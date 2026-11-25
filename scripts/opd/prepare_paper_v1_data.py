"""Convert already-downloaded DAPO/AIME data locally; no network or model loading."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ANSWER_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)

SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        ("extra_info", pa.struct([("split", pa.string()), ("index", pa.int64()), ("source_id", pa.string())])),
    ]
)


def iter_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def prompt_key(prompt):
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_row(prompt, answer, data_source, split, index, source_id):
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
        "extra_info": {"split": split, "index": index, "source_id": str(source_id)},
    }


def convert_dapo(rows):
    """Collapse exact prompt repeats only; reject conflicting answers, keep first occurrence."""
    seen = {}
    converted = []
    total = 0
    for total, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt or prompt[-1].get("role") != "user":
            raise ValueError(f"DAPO row {total}: expected chat messages ending with user")
        if any(m.get("role") not in {"system", "user"} or not isinstance(m.get("content"), str) for m in prompt):
            raise ValueError(f"DAPO row {total}: invalid prompt")
        if len([m for m in prompt if m["role"] == "user"]) != 1:
            raise ValueError(f"DAPO row {total}: expected one user turn")
        prompt = [{"role": m["role"], "content": m["content"]} for m in prompt]
        answer = row.get("reward_model", {}).get("ground_truth")
        if answer is None or not str(answer).strip():
            raise ValueError(f"DAPO row {total}: missing ground truth")
        answer = str(answer).strip()
        key = prompt_key(prompt)
        if key in seen:
            if seen[key] != answer:
                raise ValueError(f"DAPO row {total}: identical prompt has conflicting answers")
            continue
        seen[key] = answer
        source_id = (row.get("extra_info") or {}).get("index", total - 1)
        converted.append(make_row(prompt, answer, "math_dapo", "train", len(converted), source_id))
    if not converted:
        raise ValueError("DAPO training set is empty")
    return converted, {
        "raw_rows": total,
        "unique_prompts": len(converted),
        "exact_duplicates_removed": total - len(converted),
    }


def convert_aime(rows, year):
    converted = []
    seen = set()
    for i, row in enumerate(rows):
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"AIME {year} row {i}: missing problem")
        answer = str(row.get("answer", "")).strip()
        if not answer.isascii() or not answer.isdigit() or not 0 <= int(answer) <= 999:
            raise ValueError(f"AIME {year} row {i}: answer must be an integer in [0, 999]")
        if problem.strip() in seen:
            raise ValueError(f"AIME {year}: duplicated question")
        seen.add(problem.strip())
        prompt = [{"role": "user", "content": ANSWER_INSTRUCTION + problem.strip()}]
        converted.append(make_row(prompt, str(int(answer)), f"aime{year}", "test", i, row.get("id", i)))
    if len(converted) != 30:
        raise ValueError(f"AIME {year}: expected 30 problems, found {len(converted)}")
    return converted


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root):
    root = Path(root).resolve()
    output = root / "prepared"
    manifest_path = output / "manifest.json"
    inputs = {
        "dapo": root / "dapo/data/dapo-math-17k.parquet",
        "aime24": root / "aime24/test-00000-of-00001.parquet",
        "aime25": root / "aime25/test.jsonl",
        "aime26": root / "aime26/aime2026.jsonl",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    # Do not overwrite a prepared experiment's datasets on an accidental rerun.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, path in inputs.items():
            if sha256(path) != manifest["inputs"][name]["sha256"]:
                raise ValueError("Source data changed; use a new PAPER_DATA_ROOT")
        for name, info in manifest["outputs"].items():
            if sha256(output / name) != info["sha256"]:
                raise ValueError(f"Prepared file changed: {name}")
        print("Prepared files already verified:", manifest_path)
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    print("Reading DAPO in chunks and removing exact prompt duplicates...", flush=True)
    train, counts = convert_dapo(iter_rows(inputs["dapo"]))
    if len(train) < 512:
        raise ValueError(f"Only {len(train)} unique training prompts; paper batch size is 512")
    evaluation = {name: convert_aime(iter_rows(inputs[name]), name[-2:]) for name in ("aime24", "aime25", "aime26")}
    train_keys = {prompt_key(r["prompt"]) for r in train}
    overlap = {name: sum(prompt_key(r["prompt"]) in train_keys for r in rows) for name, rows in evaluation.items()}
    if any(overlap.values()):
        raise ValueError(f"Exact train/evaluation prompt overlap found: {overlap}; inspect sources before training")
    files = {"train.parquet": train, **{f"{name}.parquet": rows for name, rows in evaluation.items()}}
    files["aime.parquet"] = [row for rows in evaluation.values() for row in rows]
    manifest = {
        "paper": "https://arxiv.org/abs/2607.04751v1",
        "dapo": counts,
        "exact_prompt_overlap": overlap,
        "choices": {
            "deduplication": "exact chat prompt; first occurrence",
            "evaluation_instruction": ANSWER_INSTRUCTION,
            "scoring": "verl math_dapo default: last Answer: in last 300 characters; use acc for avg@32",
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "outputs": {},
    }
    download_manifest = root / "download_manifest.json"
    if download_manifest.is_file():
        manifest["download_manifest"] = json.loads(download_manifest.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output) as temp:
        stage = Path(temp)
        for name, rows in files.items():
            # Explicit plain Arrow schema: no datasets-version-specific HF metadata.
            path = stage / name
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
            if pq.read_table(path).to_pylist() != rows:
                raise ValueError(f"Parquet roundtrip changed {name}")
            manifest["outputs"][name] = {"rows": len(rows), "sha256": sha256(path)}
        # Publish manifest last so incomplete preparation cannot pass preflight.
        for name in files:
            os.replace(stage / name, output / name)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dapo": counts, "outputs": manifest["outputs"]}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default=os.environ.get("PAPER_DATA_ROOT"), required=not os.environ.get("PAPER_DATA_ROOT")
    )
    prepare(parser.parse_args().data_root)
