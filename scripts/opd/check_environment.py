"""Read-only OPD checks. Run without --gpu on login; with --gpu inside gpuB job.

Does not install packages, change datasets, load model weights, or start Ray.
Each check runs in its own process so a failure does not hide later failures.
"""

import argparse
import importlib.metadata as md
import json
import os
import platform
import signal
import subprocess
import sys
import textwrap
from pathlib import Path


def run_check(command, timeout):
    """Keep crash/timeout output; a negative exit code is a signal, not success."""
    try:
        run = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout
        )
        status = "PASS" if run.returncode == 0 else f"FAIL (exit {run.returncode})"
        if run.returncode < 0:
            try:
                status += " " + signal.Signals(-run.returncode).name
            except ValueError:
                pass
        return status, run.stdout
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return "TIMEOUT (inconclusive; may be filesystem/import slowness)", output
    except OSError as error:
        return "FAIL: " + repr(error), ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--root", default="/seu_nvme/home/fenglei/213241973")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    parser.add_argument("--report-dir", type=Path, help="Save each subprocess log and a machine-readable summary")
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
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
    print("Python:", sys.version, "Executable:", sys.executable, flush=True)
    print("Host:", platform.node(), "libc:", platform.libc_ver(), flush=True)
    print("Paths:", json.dumps(paths, ensure_ascii=False), flush=True)
    for name in (
        "verl",
        "torch",
        "vllm",
        "flash-attn",
        "flashinfer-python",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "datasets",
        "numpy",
        "pyarrow",
        "pandas",
        "ray",
        "peft",
        "accelerate",
        "tensordict",
        "dill",
        "multiprocess",
        "fsspec",
    ):
        try:
            print(name, md.version(name), flush=True)
        except md.PackageNotFoundError:
            print(name, "NOT INSTALLED", flush=True)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONPATH",
        "OPD_ARROW_FIRST",
    ):
        print(name, repr(os.environ.get(name)), flush=True)

    def check(name, code=None, command=None):
        print("\n===== " + name + " =====", flush=True)
        if command is None:
            prefix = "import json\np = json.loads(" + repr(json.dumps(paths)) + ")\n"
            command = [sys.executable, "-X", "faulthandler", "-u", "-c", prefix + textwrap.dedent(code)]
        status, output = run_check(command, args.timeout)
        print(output, flush=True)
        if args.report_dir:
            (args.report_dir / f"check-{len(results) + 1:02d}.log").write_text(
                name + "\n" + output + "\n" + status + "\n", encoding="utf-8"
            )
        results.append((name, status))
        print(status, flush=True)
        return status == "PASS"

    check("dependency metadata", command=[sys.executable, "-m", "pip", "check"])
    if os.environ.get("OPD_ARROW_FIRST") == "1":
        check(
            "PyArrow startup preload",
            """
            import sys
            assert "pyarrow" in sys.modules, "PyArrow was not preloaded at interpreter startup"
            hook = sys.modules.get("sitecustomize")
            assert getattr(hook, "OPD_ARROW_PRELOADED", False), "Wrong or missing OPD startup hook"
            print("Startup hook:", hook.__file__)
            print("PyArrow:", sys.modules["pyarrow"].__file__)
            """,
        )
    check(
        "model files and shard headers",
        """
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
    """,
    )
    for split in ("train", "test"):
        check(
            "parquet " + split,
            """
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
        """.replace("SPLIT_KEY", repr(split)),
        )
    check(
        "tokenizers and actual context helper",
        """
        import importlib.util
        from pathlib import Path
        from transformers import AutoTokenizer
        s = AutoTokenizer.from_pretrained(p["student"], local_files_only=True)
        t = AutoTokenizer.from_pretrained(p["teacher"], local_files_only=True)
        assert s.get_vocab() == t.get_vocab(), "Vocabulary ID mappings differ"
        assert hasattr(t, "all_special_tokens_extended"), "vLLM tokenizer API missing"
        template_path = Path(p["project"]) / "examples/on_policy_distillation_trainer/qwen3_base_chat_template.jinja"
        template = template_path.read_text()
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
    """,
    )
    if args.gpu:
        check(
            "GPU visibility",
            """
            import os
            import torch
            assert not os.environ.get("ROCR_VISIBLE_DEVICES"), "unset ROCR_VISIBLE_DEVICES"
            assert not os.environ.get("HIP_VISIBLE_DEVICES"), "unset HIP_VISIBLE_DEVICES"
            assert torch.cuda.is_available()
            assert torch.cuda.device_count() >= 4, "This experiment requests four GPUs"
            for i in range(torch.cuda.device_count()):
                print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
        """,
        )
        check(
            "FlashAttention real CUDA forward and backward",
            """
            import torch
            from flash_attn import flash_attn_func
            q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            out = flash_attn_func(q, q, q, causal=True)
            out.float().square().mean().backward()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
            print("BF16 forward/backward OK")
        """,
        )
        startup_code = """
            import sys
            sys.path.insert(0, p["project"])
            print("STAGE 1: import transformers.AutoTokenizer", flush=True)
            from transformers import AutoTokenizer
            print("STAGE 2: import vllm tokenizer helper", flush=True)
            from vllm.transformers_utils.tokenizer import get_cached_tokenizer
            for role in ("student", "teacher"):
                print("STAGE 3: load local tokenizer:", role, flush=True)
                tokenizer = AutoTokenizer.from_pretrained(p[role], local_files_only=True)
                print("STAGE 4: get_cached_tokenizer:", role, flush=True)
                get_cached_tokenizer(tokenizer)
            print("STAGE 5: import vllm_async_server", flush=True)
            import verl.workers.rollout.vllm_rollout.vllm_async_server
            print("STAGE 6: import teacher_manager", flush=True)
            import verl.experimental.teacher_loop.teacher_manager
            print("STAGE 7: import engine_workers", flush=True)
            import verl.workers.engine_workers
            print("STAGE 8: import FSDP engine", flush=True)
            import verl.workers.engine.fsdp
            print("STAGE 9: import actual training entrypoint", flush=True)
            import verl.trainer.main_ppo
            print("Tokenizers and worker imports OK (engine startup not tested)")
        """
        imports_ok = check("vLLM tokenizer and worker import sequence (crash trace)", startup_code)
        # Keep the original import order above to detect interactions. Then
        # isolate each major component in a fresh process, even after a crash.
        # Only diagnose individual dependencies if the production import order
        # failed. Passing isolated imports never overrides that failure.
        modules = (
            ()
            if imports_ok
            else (
                "ray",
                "numba",
                "zmq",
                "flashinfer",
                "vllm.transformers_utils.tokenizer",
                "vllm._C",
                "verl.workers.rollout.vllm_rollout.vllm_async_server",
                "verl.experimental.teacher_loop.teacher_manager",
                "verl.workers.engine_workers",
                "verl.workers.engine.fsdp",
                "verl.trainer.main_ppo",
            )
        )
        for module in modules:
            check(
                "isolated import: " + module,
                """
                import sys
                import importlib
                sys.path.insert(0, p["project"])
                print("Importing:", MODULE_NAME, flush=True)
                importlib.import_module(MODULE_NAME)
                print("Import completed", flush=True)
            """.replace("MODULE_NAME", repr(module)),
            )
        if not imports_ok and args.report_dir:
            # importtime gives the last completed imports even when a native
            # crash has no useful Python stack. Keep the large trace in a file.
            prefix = "import json\np = json.loads(" + repr(json.dumps(paths)) + ")\n"
            command = [
                sys.executable,
                "-X",
                "faulthandler",
                "-X",
                "importtime",
                "-u",
                "-c",
                prefix + textwrap.dedent(startup_code),
            ]
            status, output = run_check(command, args.timeout)
            trace = args.report_dir / "import-trace.log"
            trace.write_text(output + "\n" + status + "\n", encoding="utf-8")
            print("Additional import trace:", trace, status, flush=True)
    print("\n===== SUMMARY =====", flush=True)
    for name, status in results:
        print(status, "|", name, flush=True)
    if args.report_dir:
        (args.report_dir / "summary.json").write_text(
            json.dumps(dict(results), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print("Not tested: engine startup, NCCL collectives, full-model memory, rollout and optimizer step.", flush=True)
    return 0 if all(status == "PASS" for _, status in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
