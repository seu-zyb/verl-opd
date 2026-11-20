"""Create validated datasets-3-compatible copies; never edit the source files."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def check_rows(rows):
    if not rows:
        raise ValueError("Empty dataset")
    for i, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Row {i}: prompt must be a nonempty list")
        if any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in prompt):
            raise ValueError(f"Row {i}: invalid text message")
        if prompt[-1].get("role") != "user" or any(m.get("role") != "system" for m in prompt[:-1]):
            raise ValueError(f"Row {i}: expected one user turn, optionally preceded by system messages")
        if row.get("data_source") != "openai/gsm8k":
            raise ValueError(f"Row {i}: unexpected GSM8K reward routing: {row.get('data_source')!r}")
        reward = row.get("reward_model")
        if not isinstance(reward, dict) or not isinstance(reward.get("ground_truth"), str):
            raise ValueError(f"Row {i}: reward_model.ground_truth must be a string")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/seu_nvme/home/fenglei/213241973/data/gsm8k")
    parser.add_argument("--cache-dir", help="Optional datasets cache directory")
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from datasets import load_dataset

    source_dir = Path(args.data_dir).resolve()
    output_dir = source_dir / "compat_datasets3"
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    # Validate BOTH splits before publishing either. All temporary files stay
    # below the new output folder; the original parquet files are read only.
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output_dir) as temp:
        stage = Path(temp)
        for name in ("train.parquet", "test.parquet"):
            source = source_dir / name
            original = pq.read_table(source)
            rows = original.to_pylist()
            check_rows(rows)
            metadata = dict(original.schema.metadata or {})
            metadata.pop(b"huggingface", None)
            cleaned = original.replace_schema_metadata(metadata)
            destination = stage / name
            pq.write_table(cleaned, destination)
            roundtrip = pq.read_table(destination)
            if not original.equals(roundtrip, check_metadata=False):
                raise ValueError(f"{name}: Arrow content changed")
            loaded = load_dataset("parquet", data_files=str(destination), split="train",
                                  cache_dir=args.cache_dir or str(stage / "cache"))
            if loaded.to_list() != rows:
                raise ValueError(f"{name}: datasets decoded content changed")
            check_rows(loaded.to_list())
            counts[name] = len(loaded)
            print(f"PASS: {name}: {len(loaded)} rows, all contents identical", flush=True)
            del loaded
        for name in counts:
            os.replace(stage / name, output_dir / name)
    print("Original files unchanged.")
    print("Compatible data:", output_dir)
    print(json.dumps(counts))
    print("READY: both datasets verified; submit run_style_experiment_fixed.slurm")


if __name__ == "__main__":
    main()
