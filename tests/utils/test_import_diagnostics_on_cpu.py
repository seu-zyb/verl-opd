"""Check subprocess outcome preservation without requiring vLLM, CUDA, or intentional crashes."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "import_diagnostics", Path(__file__).resolve().parents[2] / "scripts/opd/diagnose_import_order.py"
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_nonzero_child_retains_output_and_does_not_stop_next_case(tmp_path):
    failed = diagnostics.run_case(
        "failure", [sys.executable, "-u", "-c", "print('before failure'); raise SystemExit(3)"], tmp_path, 10
    )
    passed = diagnostics.run_case("success", [sys.executable, "-c", "print('done')"], tmp_path, 10)
    assert failed["status"] == "FAIL" and failed["returncode"] == 3
    assert "before failure" in Path(failed["log"]).read_text()
    assert passed["status"] == "PASS" and passed["returncode"] == 0


def test_missing_executable_is_reported(tmp_path):
    result = diagnostics.run_case("missing", [str(tmp_path / "nonexistent-python")], tmp_path, 10)
    assert result["returncode"] is None
    assert result["status"].startswith("LAUNCH ERROR")
