"""Check the OPD launcher/diagnostics without a GPU or installed vLLM."""

import ast
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/opd/check_environment.py"
spec = importlib.util.spec_from_file_location("opd_environment_check", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_preserve_exception_output():
    status, output = checker.run_check(
        [sys.executable, "-u", "-c", "print('BEFORE'); raise RuntimeError('failure detail')"], 10
    )
    assert status == "FAIL (exit 1)"
    assert "BEFORE" in output and "failure detail" in output


def test_preserve_timeout_output():
    status, output = checker.run_check(
        [sys.executable, "-u", "-c", "import time; print('LAST STAGE'); time.sleep(20)"], 2
    )
    assert status.startswith("TIMEOUT")
    assert "LAST STAGE" in output


def test_report_native_signal(monkeypatch):
    monkeypatch.setattr(
        checker.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], -11, "native crash trace")
    )
    status, output = checker.run_check(["python"], 2)
    assert status == "FAIL (exit -11) SIGSEGV"
    assert output == "native crash trace"


def test_checked_verl_modules_exist():
    # The previous check referenced removed fsdp_workers.py, blocking otherwise
    # healthy runs. Inspect embedded Python checks without importing GPU code.
    tree = ast.parse(CHECKER.read_text(encoding="utf-8"))
    imports = set()
    import textwrap

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("verl.") and " " not in node.value:
                imports.add(node.value)
            try:
                embedded = ast.parse(textwrap.dedent(node.value))
            except (SyntaxError, ValueError):
                continue
            for statement in ast.walk(embedded):
                if isinstance(statement, ast.Import):
                    imports.update(n.name for n in statement.names if n.name.startswith("verl."))
    assert "verl.workers.engine_workers" in imports
    assert "verl.trainer.main_ppo" in imports
    assert "verl.workers.fsdp_workers" not in imports
    for module in imports:
        path = ROOT.joinpath(*module.split("."))
        assert path.with_suffix(".py").exists() or (path / "__init__.py").exists(), module


@pytest.mark.parametrize("mode", ["style", "full", "baseline"])
def test_real_launcher_composes_four_gpu_config(tmp_path, mode):
    bash = os.environ.get("OPD_TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("bash required to exercise the actual launcher")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "argv.json"
    capture = tmp_path / "capture.py"
    capture.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['OPD_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    for name in ("python", "python3"):
        parts = [sys.executable.replace("\\", "/")]
        if name == "python3":
            parts.append(capture.as_posix())
        wrapper = fake_bin / name
        wrapper.write_text(
            "#!/bin/bash\nexec " + " ".join(map(shlex.quote, parts)) + ' "$@"\n', encoding="utf-8", newline="\n"
        )
        wrapper.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
        OPD_CAPTURE=str(captured),
        CONTEXT_MODE=mode,
        NGPUS_PER_NODE="2",
        TEACHER_WORLD_SIZE="2",
        ROLLOUT_TP="1",
        TEACHER_TP="1",
        TOTAL_TRAINING_STEPS="100",
        SEED="2026",
        SAVE_FREQ="20",
        TEST_FREQ="20",
    )
    for name in ("STUDENT_MODEL", "TEACHER_MODEL", "TRAIN_FILE", "VAL_FILE"):
        path = tmp_path / name.lower()
        path.touch()
        env[name] = path.as_posix()
    launch = ROOT / "examples/on_policy_distillation_trainer/run_full_rollout_context.sh"
    result = subprocess.run(
        [
            bash,
            str(launch),
            "trainer.val_before_train=True",
            "trainer.resume_mode=disable",
            "trainer.validation_data_dir=" + (tmp_path / "validation").as_posix(),
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(captured.read_text())
    assert args[:2] == ["-m", "verl.trainer.main_ppo"]
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=args[2:])
    assert config.trainer.n_gpus_per_node == 2
    assert config.distillation.n_gpus_per_node == 2
    assert config.trainer.total_training_steps == 100
    assert config.trainer.val_before_train and config.trainer.resume_mode == "disable"
    assert config.trainer.validation_data_dir == (tmp_path / "validation").as_posix()
    assert config.distillation.full_rollout_context == (mode != "baseline")
    assert config.distillation.teacher_models.teacher_model.inference.max_model_len == 5121
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert not config.distillation.distillation_loss.use_task_rewards
    assert config.distillation.distillation_loss.use_policy_gradient
    assert config.distillation.distillation_loss.loss_mode == "k1"
    assert config.data.train_files == [env["TRAIN_FILE"]]
    assert config.data.val_files == [env["VAL_FILE"]]
