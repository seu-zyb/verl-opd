"""Read-only OPD checks. Run without --gpu on login; with --gpu inside gpuB job.

Does not install packages, change datasets, load model weights, or start Ray.
Each check runs in its own process so a failure does not hide later failures.
"""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--root", default="/seu_nvme/home/fenglei/213241973")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    args = parser.parse_args()
    root = Path(args.root)
    paths = {
        "project": os.environ.get("PROJECT_ROOT", str(root / "code/verl-opd")),
        "student": os.environ.get("STUDENT_MODEL", str(root / "models/Qwen3-1.7B-Base")),
        "teacher": os.environ.get("TEACHER_MODEL", str(root / "models/Qwen3-14B")),
        "data": str(root / "data/gsm8k"),
        "train": args.train_file or os.environ.get("TRAIN_FILE", str(root / "data/gsm8k/train.parquet")),
        "test": args.val_file or os.environ.get("VAL_FILE", str(root / "data/gsm8k/test.parquet")),
    }
    results = []
    print("Python:", sys.version, "Executable:", sys.executable, flush=True)
    print("Paths:", json.dumps(paths, ensure_ascii=False), flush=True)
    for name in ("verl", "torch", "vllm", "flash-attn", "flashinfer-python",
                 "transformers", "tokenizers", "huggingface-hub", "datasets",
                 "numpy", "pyarrow", "pandas", "ray", "peft", "accelerate",
                 "tensordict", "dill", "multiprocess", "fsspec"):
        try:
            print(name, md.version(name), flush=True)
        except md.PackageNotFoundError:
            print(name, "NOT INSTALLED", flush=True)
    for name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        print(name, repr(os.environ.get(name)), flush=True)

    def check(name, code=None, command=None):
        print("\n===== " + name + " =====", flush=True)
        if command is None:
            prefix = "import json\np = json.loads(" + repr(json.dumps(paths)) + ")\n"
            command = [sys.executable, "-u", "-c", prefix + textwrap.dedent(code)]
        try:
            run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", timeout=args.timeout)
            print(run.stdout, flush=True)
            status = "PASS" if run.returncode == 0 else "FAIL (exit %s)" % run.returncode
        except subprocess.TimeoutExpired:
            status = "TIMEOUT (inconclusive; may be filesystem/import slowness)"
        except Exception as error:
            status = "FAIL: " + repr(error)
        results.append((name, status))
        print(status, flush=True)

    check("dependency metadata", command=[sys.executable, "-m", "pip", "check"])
    check("model files and shard headers", '''
        from pathlib import Path
        from safetensors import safe_open
        for role in ("student", "teacher"):
            folder = Path(p[role])
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                json.loads((folder / name).read_text())
            index = folder / "model.safetensors.index.json"
            shards = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
                      if index.exists() else ["model.safetensors"])
            for name in shards:
                with safe_open(str(folder / name), framework="pt", device="cpu") as f:
                    assert list(f.keys()), name
            print(role, len(shards), "shard headers readable (not a checksum test)")
    ''')
    for split in ("train", "test"):
        check("parquet " + split, '''
            from pathlib import Path
            import pyarrow.parquet as pq
            from datasets import load_dataset
            file = Path(p[SPLIT_KEY])
            raw = pq.read_table(file)
            required = {"prompt", "data_source", "reward_model"}
            assert required.issubset(raw.column_names), raw.column_names
            data = load_dataset("parquet", data_files=str(file), split="train")
            assert len(data) > 0
            assert data.to_list() == raw.to_pylist(), "Decoded content differs"
            for row in data:
                assert isinstance(row["prompt"], list) and row["prompt"]
                assert all(isinstance(m, dict) and isinstance(m.get("content"), str)
                           for m in row["prompt"])
                assert row["prompt"][-1]["role"] == "user"
                assert isinstance(row["reward_model"], dict)
                assert "ground_truth" in row["reward_model"]
            print("Rows:", len(data), "Columns:", data.column_names)
        '''.replace("SPLIT_KEY", repr(split)))
    check("tokenizers and actual context helper", '''
        import importlib.util
        from pathlib import Path
        from transformers import AutoTokenizer
        s = AutoTokenizer.from_pretrained(p["student"], local_files_only=True)
        t = AutoTokenizer.from_pretrained(p["teacher"], local_files_only=True)
        assert s.get_vocab() == t.get_vocab(), "Vocabulary ID mappings differ"
        assert hasattr(t, "all_special_tokens_extended"), "vLLM tokenizer API missing"
        template = (Path(p["project"]) / "examples/on_policy_distillation_trainer/qwen3_base_chat_template.jinja").read_text()
        s.chat_template = template
        question = [{"role": "user", "content": "What is 2 + 2?"}]
        assert s.apply_chat_template(question, tokenize=True, add_generation_prompt=True)
        path = Path(p["project"]) / "verl/utils/full_rollout_context.py"
        spec = importlib.util.spec_from_file_location("opd_context_check", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        response = s.encode("2 + 2 = 4.", add_special_tokens=False)
        tokens = helper.build_full_rollout_teacher_prompt(
            question, response, s, t, "Correct errors and preserve valid expression.",
            4096, {"enable_thinking": False})
        print("Vocabulary matches; context helper passes; teacher prefix tokens:", len(tokens))
        print("Student EOS:", s.eos_token_id, "Teacher EOS:", t.eos_token_id,
              "(different EOS is not a vocabulary mismatch)")
    ''')
    if args.gpu:
        check("GPU visibility", '''
            import os
            import torch
            assert not os.environ.get("ROCR_VISIBLE_DEVICES"), "unset ROCR_VISIBLE_DEVICES"
            assert not os.environ.get("HIP_VISIBLE_DEVICES"), "unset HIP_VISIBLE_DEVICES"
            assert torch.cuda.is_available()
            assert torch.cuda.device_count() >= 4, "This experiment requests four GPUs"
            for i in range(torch.cuda.device_count()):
                print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
        ''')
        check("FlashAttention real CUDA forward and backward", '''
            import torch
            from flash_attn import flash_attn_func
            q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            out = flash_attn_func(q, q, q, causal=True)
            out.float().square().mean().backward()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
            print("BF16 forward/backward OK")
        ''')
        check("vLLM tokenizer API and verl worker imports", '''
            import sys
            sys.path.insert(0, p["project"])
            from transformers import AutoTokenizer
            from vllm.transformers_utils.tokenizer import get_cached_tokenizer
            for role in ("student", "teacher"):
                get_cached_tokenizer(AutoTokenizer.from_pretrained(p[role], local_files_only=True))
            import verl.workers.rollout.vllm_rollout.vllm_async_server
            import verl.experimental.teacher_loop.teacher_manager
            import verl.workers.fsdp_workers
            print("Tokenizers and worker imports OK (engine startup not tested)")
        ''')
    print("\n===== SUMMARY =====", flush=True)
    for name, status in results:
        print(status, "|", name, flush=True)
    print("Not tested: engine startup, NCCL collectives, full-model memory, rollout and optimizer step.", flush=True)
    return 0 if all(status == "PASS" for _, status in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
