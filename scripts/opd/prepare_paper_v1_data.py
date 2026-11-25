"""Convert already-downloaded DAPO/AIME data locally; no network or model loading."""

import argparse
import hashlib
import json
import os
import re
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


def normalize_dapo_answer(answer):
    """Canonicalize plain integer notation only; do not guess algebraic equivalence."""
    answer = str(answer).strip()
    if re.fullmatch(r"[+-]?[0-9]+(?:\.0+)?", answer):
        return str(int(answer.split(".")[0]))
    return answer


def convert_dapo(rows, conflict_policy="error", conflict_report=None):
    """Audit all answers per exact prompt; optionally drop entire conflicting groups."""
    if conflict_policy not in {"error", "drop"}:
        raise ValueError(f"Unknown DAPO conflict policy: {conflict_policy}")
    seen = {}
    total = 0
    normalized_rows = 0
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
        raw_answer = str(answer).strip()
        answer = normalize_dapo_answer(raw_answer)
        normalized_rows += answer != raw_answer
        key = prompt_key(prompt)
        source_id = (row.get("extra_info") or {}).get("index", total - 1)
        group = seen.setdefault(key, {"prompt": prompt, "source_id": source_id, "answers": {}})
        variant = group["answers"].setdefault(
            answer, {"count": 0, "first_row": total, "source_id": str(source_id), "raw_answers": {}}
        )
        variant["count"] += 1
        variant["raw_answers"][raw_answer] = variant["raw_answers"].get(raw_answer, 0) + 1
    if not seen:
        raise ValueError("DAPO training set is empty")

    converted = []
    conflicts = []
    duplicate_rows = 0
    conflict_rows = 0
    for key, group in seen.items():
        if len(group["answers"]) > 1:
            conflicts.append(
                {
                    "prompt_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    "prompt": group["prompt"],
                    "answers": group["answers"],
                }
            )
            conflict_rows += sum(v["count"] for v in group["answers"].values())
            continue
        answer, variant = next(iter(group["answers"].items()))
        duplicate_rows += variant["count"] - 1
        converted.append(make_row(group["prompt"], answer, "math_dapo", "train", len(converted), group["source_id"]))
    if conflict_report is not None:
        conflict_report.extend(conflicts)
    if conflicts and conflict_policy == "error":
        first_answers = {a: v["first_row"] for a, v in conflicts[0]["answers"].items()}
        raise ValueError(
            f"DAPO: {len(conflicts)} identical prompts have conflicting answers after integer normalization; "
            f"first conflict (answer: first row): {first_answers}. "
            "See prepared/dapo-conflicts.json. Use --dapo-conflict-policy drop to exclude entire conflicting prompts."
        )
    return converted, {
        "raw_rows": total,
        "unique_prompts": len(seen),
        "training_prompts": len(converted),
        "duplicate_rows_removed": duplicate_rows,
        "answer_format_rows_normalized": normalized_rows,
        "conflicting_prompts": len(conflicts),
        "conflict_rows_removed": conflict_rows,
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


def prepare(root, conflict_policy="error"):
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
        if manifest["choices"].get("dapo_conflict_policy", "error") != conflict_policy:
            raise ValueError("Prepared data uses another conflict policy; use a new PAPER_DATA_ROOT")
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
    conflicts = []
    conflict_path = output / "dapo-conflicts.json"
    scan_complete = False
    try:
        train, counts = convert_dapo(iter_rows(inputs["dapo"]), conflict_policy, conflicts)
        scan_complete = True
    finally:
        conflict_path.write_text(
            json.dumps(
                {
                    "policy": conflict_policy,
                    "scan_complete": scan_complete or bool(conflicts),
                    "conflicting_prompts": len(conflicts),
                    "conflicts": conflicts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("DAPO conflict report:", conflict_path, flush=True)
    if conflicts:
        print(f"Excluded {len(conflicts)} conflicting prompts; retained {len(train)} training prompts.", flush=True)
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
            "deduplication": "exact chat prompt; normalize plain integer answers; first occurrence if unambiguous",
            "dapo_conflict_policy": conflict_policy,
            "evaluation_instruction": ANSWER_INSTRUCTION,
            "scoring": "verl math_dapo default: last Answer: in last 300 characters; use acc for avg@32",
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "outputs": {"dapo-conflicts.json": {"conflicting_prompts": len(conflicts), "sha256": sha256(conflict_path)}},
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
    parser.add_argument(
        "--dapo-conflict-policy",
        choices=("error", "drop"),
        default="error",
        help="error writes a conflict report and stops; drop excludes every row of conflicting prompts",
    )
    args = parser.parse_args()
    prepare(args.data_root, args.dapo_conflict_policy)
