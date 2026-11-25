"""Offline data/config checks and isolated production loss gradient checks (no Ray/GPU)."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_paper", ROOT / "scripts/opd/prepare_paper_v1_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def dapo_row(i, answer="7"):
    return {
        "prompt": [{"role": "user", "content": f"Training problem {i}"}],
        "reward_model": {"ground_truth": answer},
        "extra_info": {"index": str(i)},
    }


def test_exact_deduplication_and_conflicting_answers():
    rows, counts = prepare.convert_dapo([dapo_row(1), dapo_row(1), dapo_row(2)])
    assert len(rows) == 2
    assert counts["raw_rows"] == 3
    assert counts["unique_prompts"] == counts["training_prompts"] == 2
    assert counts["duplicate_rows_removed"] == 1
    assert counts["conflicting_prompts"] == 0
    assert rows[0]["prompt"] == dapo_row(1)["prompt"]
    with pytest.raises(ValueError, match="conflicting answers"):
        prepare.convert_dapo([dapo_row(1), dapo_row(1, "8")])


def test_integer_format_normalization_does_not_guess_other_expressions():
    rows, counts = prepare.convert_dapo([dapo_row(1, a) for a in ["7", "007", "+7", "7.00"]])
    assert len(rows) == 1
    assert rows[0]["reward_model"]["ground_truth"] == "7"
    assert counts["answer_format_rows_normalized"] == 3
    assert prepare.normalize_dapo_answer("-003.0") == "-3"
    for answer in ["0.5", "1/2", "1e3", "1,000", r"\boxed{7}"]:
        assert prepare.normalize_dapo_answer(answer) == answer


@pytest.mark.parametrize("reverse", [False, True])
def test_drop_excludes_all_conflicting_rows_including_later_repeats(reverse):
    raw = [dapo_row(1), dapo_row(1, "8"), dapo_row(2), dapo_row(1), dapo_row(2, "007")]
    report = []
    rows, counts = prepare.convert_dapo(reversed(raw) if reverse else raw, "drop", report)
    assert [r["prompt"] for r in rows] == [dapo_row(2)["prompt"]]
    assert rows[0]["extra_info"]["index"] == 0
    assert counts["conflicting_prompts"] == 1
    assert counts["conflict_rows_removed"] == 3
    assert (
        counts["raw_rows"]
        == counts["training_prompts"] + counts["duplicate_rows_removed"] + counts["conflict_rows_removed"]
    )
    assert report[0]["answers"]["7"]["count"] == 2
    assert report[0]["answers"]["8"]["count"] == 1


def aime_rows(year):
    # math-ai/aime24 uses solution + integer id; aime25/26 use answer + string id.
    return [
        {
            "problem": f"AIME {year} question {i}",
            **({"solution": r"\boxed{007}", "id": i + 60} if year == 24 else {"answer": "007", "id": str(i)}),
        }
        for i in range(30)
    ]


def test_aime_contract():
    rows = prepare.convert_aime(aime_rows(24), "24")
    assert rows[0]["reward_model"]["ground_truth"] == "7"
    assert rows[0]["data_source"] == "aime24"
    assert rows[0]["prompt"][0]["content"].startswith(prepare.ANSWER_INSTRUCTION)
    with pytest.raises(ValueError, match="expected 30"):
        prepare.convert_aime(aime_rows(24)[:-1], "24")
    invalid = aime_rows(24)
    invalid[0]["answer"] = "1000"
    with pytest.raises(ValueError, match="integer"):
        prepare.convert_aime(invalid, "24")


@pytest.mark.parametrize(
    "year, raw, expected",
    [
        ("24", {"solution": r"\boxed{204}", "id": 60}, "204"),
        ("25", {"answer": "70", "id": "0"}, "70"),
        ("26", {"answer": "277", "id": "1"}, "277"),
        ("24", {"answer": None, "solution": r"\boxed{000}"}, "0"),
        ("24", {"answer": "", "solution": r"\boxed{999}"}, "999"),
        ("25", {"answer": "007", "solution": "A long worked solution."}, "7"),
    ],
)
def test_actual_math_ai_answer_schemas(year, raw, expected):
    assert prepare.extract_aime_answer(raw, year, 0) == expected


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"answer": "1.5"},
        {"answer": "1000"},
        {"answer": "-1"},
        {"solution": r"\boxed{1000}"},
        {"solution": r"\boxed{1} or \boxed{2}"},
        {"solution": "There are 204 objects."},
        {"answer": "bad", "solution": r"\boxed{204}"},
    ],
)
def test_aime_answer_parser_rejects_missing_ambiguous_or_invalid_labels(raw):
    with pytest.raises(ValueError, match="Available fields"):
        prepare.extract_aime_answer(raw, "24", 0)


def test_prepare_roundtrip_idempotency_and_tamper_detection(tmp_path):
    for directory in ("dapo/data", "aime24", "aime25", "aime26"):
        (tmp_path / directory).mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([dapo_row(i) for i in range(512)] + [dapo_row(1), dapo_row(999), dapo_row(999, "8")]),
        tmp_path / "dapo/data/dapo-math-17k.parquet",
    )
    pq.write_table(pa.Table.from_pylist(aime_rows(24)), tmp_path / "aime24/test-00000-of-00001.parquet")
    for year, name in ((25, "test.jsonl"), (26, "aime2026.jsonl")):
        (tmp_path / f"aime{year}" / name).write_text(
            "\n".join(json.dumps(r) for r in aime_rows(year)), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="conflicting answers"):
        prepare.prepare(tmp_path)
    report_path = tmp_path / "prepared/dapo-conflicts.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["conflicting_prompts"] == 1
    assert not (tmp_path / "prepared/manifest.json").exists()
    assert not (tmp_path / "prepared/train.parquet").exists()
    manifest = prepare.prepare(tmp_path, "drop")
    assert manifest["dapo"]["unique_prompts"] == 513
    assert manifest["dapo"]["training_prompts"] == 512
    assert manifest["choices"]["dapo_conflict_policy"] == "drop"
    assert manifest["outputs"]["aime.parquet"]["rows"] == 90
    table = pq.read_table(tmp_path / "prepared/aime.parquet")
    assert not table.schema.metadata  # Avoid HF datasets metadata/version incompatibility.
    assert {r["data_source"] for r in table.to_pylist()} == {"aime24", "aime25", "aime26"}
    assert prepare.prepare(tmp_path, "drop") == manifest
    with pytest.raises(ValueError, match="another conflict policy"):
        prepare.prepare(tmp_path)
    (tmp_path / "prepared/train.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Prepared file changed"):
        prepare.prepare(tmp_path, "drop")


def test_hydra_profile_resolves_and_preserves_one_update_per_batch(monkeypatch):
    for key, value in {
        "PAPER_DATA_ROOT": "/data/paper",
        "STUDENT_MODEL": "/models/student",
        "TEACHER_MODEL": "/models/teacher",
        "RUN_OUTPUT": "/outputs/test",
        "SEED": "2026",
        "TOTAL_TRAINING_STEPS": "200",
        "VAL_BEFORE_TRAIN": "true",
        "TEST_FREQ": "20",
        "SAVE_FREQ": "20",
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="opd_paper_v1", overrides=["+trainer.token_probability_every=1"])
        OmegaConf.resolve(cfg)
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 512
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    assert cfg.actor_rollout_ref.rollout.n * cfg.data.train_batch_size == 4096
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 32
    assert cfg.actor_rollout_ref.rollout.val_kwargs.do_sample
    assert cfg.actor_rollout_ref.rollout.val_kwargs.top_p == 0.7
    assert cfg.trainer.total_epochs >= cfg.trainer.total_training_steps == 200
    assert cfg.distillation.teacher_models.teacher_model.inference.max_model_len >= (
        cfg.data.max_prompt_length + cfg.data.max_response_length + 1
    )
    assert cfg.data.train_files == ["/data/paper/prepared/train.parquet"]
    assert cfg.distillation.distillation_loss.policy_loss_mode == "gpg"
    assert cfg.distillation.distillation_loss.loss_max_clamp is None
    assert not cfg.distillation.distillation_loss.use_task_rewards
    assert not cfg.distillation.full_rollout_context
    assert cfg.actor_rollout_ref.rollout.response_length == 16384


def load_functions(relative_path, names, namespace):
    """Execute actual function bodies while excluding unrelated distributed imports."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)


def test_opd_gradient_uses_teacher_signal_without_clipping_or_reward_normalization():
    ns = {"torch": torch}
    load_functions("verl/utils/torch_functional.py", ["masked_sum"], ns)
    ns["verl_F"] = SimpleNamespace(masked_sum=ns["masked_sum"])
    load_functions("verl/trainer/ppo/core_algos.py", ["agg_loss", "compute_policy_loss_gpg"], ns)
    ns.update(
        {
            "get_distillation_loss_fn": lambda mode: lambda **kw: (
                kw["model_output"]["log_probs"] - kw["data"]["teacher_logprobs"],
                {},
            ),
            "get_policy_loss_fn": lambda mode: ns["compute_policy_loss_gpg"] if mode == "gpg" else None,
            "no_padding_2_padding": lambda value, data: value,
            "compute_distillation_loss_range": lambda **kw: {},
        }
    )
    load_functions("verl/trainer/distillation/losses.py", ["distillation_loss"], ns)
    logp = torch.tensor([[-0.2, -0.5, -0.7]], requires_grad=True)
    teacher = torch.tensor([[-25.0, -0.1, -1.0]])
    mask = torch.tensor([[True, True, False]])
    cfg = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_cfg = SimpleNamespace(
        loss_mode="k1", loss_max_clamp=None, use_policy_gradient=True, policy_loss_mode="gpg", global_batch_info={}
    )
    loss, _ = ns["distillation_loss"](
        cfg,
        SimpleNamespace(distillation_loss=loss_cfg),
        {"log_probs": logp},
        {"response_mask": mask, "teacher_logprobs": teacher, "old_log_probs": logp.detach()},
    )
    loss.backward()
    torch.testing.assert_close(logp.grad, (logp.detach() - teacher) * mask / mask.sum())
    assert logp.grad[0, 0] > 10  # Detect unintended +/-10 reward clamp.
