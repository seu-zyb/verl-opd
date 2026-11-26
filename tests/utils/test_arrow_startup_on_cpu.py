"""Exercise actual interpreter startup and inherited subprocesses without CUDA."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "scripts/opd/arrow_first"


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_preload_precedes_user_code_and_reaches_child(tmp_path, enabled):
    env = dict(os.environ, OPD_ARROW_FIRST=enabled, PYTHONPATH=str(HOOK))
    probe = (
        "import json, sys; "
        "print(json.dumps({'arrow': 'pyarrow' in sys.modules, "
        "'hook': getattr(sys.modules.get('sitecustomize'), 'OPD_ARROW_PRELOADED', False)}))"
    )
    parent = probe + "; import subprocess; subprocess.run([sys.executable, '-c', " + repr(probe) + "], check=True)"
    result = subprocess.run(
        [sys.executable, "-u", "-c", parent], env=env, cwd=tmp_path, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    expected = {"arrow": enabled == "1", "hook": enabled == "1"}
    assert [json.loads(line) for line in result.stdout.splitlines()] == [expected, expected]


def test_preload_failure_stops_interpreter(tmp_path):
    # Simulate an unusable Arrow installation, without crashing the test process.
    (tmp_path / "pyarrow.py").write_text("raise ImportError('simulated broken Arrow')\n", encoding="utf-8")
    env = dict(os.environ, OPD_ARROW_FIRST="1", PYTHONPATH=os.pathsep.join([str(HOOK), str(tmp_path)]))
    result = subprocess.run(
        [sys.executable, "-c", "print('USER_CODE_RAN')"],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "USER_CODE_RAN" not in result.stdout
    assert "OPD PyArrow startup preload failed" in result.stderr


def test_hydra_ray_startup_overrides_remain_strings():
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(
            config_name="opd_paper_v1",
            overrides=[
                '+ray_kwargs.ray_init.runtime_env.env_vars.OPD_ARROW_FIRST="1"',
                "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='/project/arrow_first:/project'",
            ],
        )
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.OPD_ARROW_FIRST == "1"
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH == "/project/arrow_first:/project"
