"""Collect isolated import-order comparisons; do not install packages or start training."""

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

SERVER = "verl.workers.rollout.vllm_rollout.vllm_async_server"
PRELUDE = """
import faulthandler, importlib, os
faulthandler.enable(all_threads=True)
def load(name):
    print('BEGIN IMPORT', name, flush=True)
    module = importlib.import_module(name)
    print('END IMPORT', name, getattr(module, '__version__', ''), getattr(module, '__file__', ''), flush=True)
    return module
"""
TOKENIZERS = """
load('transformers')
from transformers import AutoTokenizer
from vllm.transformers_utils.tokenizer import get_cached_tokenizer
for role in ('STUDENT_MODEL', 'TEACHER_MODEL'):
    print('LOAD TOKENIZER', role, flush=True)
    get_cached_tokenizer(AutoTokenizer.from_pretrained(os.environ[role], local_files_only=True))
"""


def run_case(name, command, output_dir, timeout):
    """Write output even on native crashes; a later passing case cannot hide a failure."""
    log_path = output_dir / f"{name}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        code = result.returncode
        status = "PASS" if code == 0 else "FAIL"
        if code < 0:
            try:
                status += " " + signal.Signals(-code).name
            except ValueError:
                status += f" signal {-code}"
    except subprocess.TimeoutExpired:
        code, status = None, "TIMEOUT"
    except OSError as error:
        code, status = None, f"LAUNCH ERROR: {error}"
    info = {"status": status, "returncode": code, "log": str(log_path)}
    print(name, json.dumps(info), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--native-backtrace", action="store_true", help="Use existing gdb after SIGSEGV, if available")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("torch", "vllm", "numpy", "pandas", "pyarrow", "ray", "transformers"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    metadata = {
        "host": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "libc": platform.libc_ver(),
        "versions": versions,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "ARROW_DEFAULT_MEMORY_POOL",
            )
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    server = f"load({SERVER!r})\n"
    cases = {
        "01-arrow": "load('pyarrow')\n",
        "02-pandas": "load('pandas')\n",
        "03-torch-arrow": "load('torch')\nload('pyarrow')\n",
        "04-server-original": server,
        "05-arrow-server": "load('pyarrow')\n" + server,
        "06-torch-arrow-server": "load('torch')\nload('pyarrow')\n" + server,
        "07-full-original": TOKENIZERS + server,
        "08-full-arrow-first": "load('pyarrow')\n" + TOKENIZERS + server,
    }
    results = {}
    for name, code in cases.items():
        results[name] = run_case(
            name, [sys.executable, "-X", "faulthandler", "-u", "-c", PRELUDE + code], args.output_dir, args.timeout
        )
    summary = {"training_tested": False, "cases": results}
    # Preserve all results before optionally requesting a native backtrace.
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.native_backtrace:
        gdb = shutil.which("gdb")
        failed = next(
            (
                name
                for name in ("07-full-original", "04-server-original")
                if results[name]["returncode"] == -signal.SIGSEGV
            ),
            None,
        )
        if gdb and failed:
            command = [
                gdb,
                "--nx",
                "--batch",
                "-ex",
                "set pagination off",
                "-ex",
                "run",
                "-ex",
                "thread apply all bt 30",
                "--args",
                sys.executable,
                "-X",
                "faulthandler",
                "-u",
                "-c",
                PRELUDE + cases[failed],
            ]
            # A debugger exit code is not the child Python process's success status.
            summary["native_backtrace"] = run_case("native-backtrace", command, args.output_dir, args.timeout)
            summary["native_backtrace"]["note"] = "Debugger completion only; inspect log for the inferior crash."
        else:
            summary["native_backtrace"] = {"status": "SKIPPED", "reason": "gdb absent or no baseline SIGSEGV"}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Diagnostic collection complete. This does not certify training readiness.", flush=True)
    print("Results:", args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
