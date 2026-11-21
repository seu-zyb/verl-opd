"""Compatibility entrypoint; use the maintained checker in scripts/opd."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/opd/check_environment.py"), run_name="__main__")
