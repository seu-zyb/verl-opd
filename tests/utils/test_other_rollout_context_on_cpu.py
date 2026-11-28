"""MOPD pairing and real two-pass methods with mocked Ray/inference services."""

import ast
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("other_context", ROOT / "verl/utils/other_rollout_context.py")
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def objects(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class Batch:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(next(iter(self.batch.values())))

    def chunk(self, n):
        return [
            Batch(
                {k: v[indices] for k, v in self.batch.items()},
                {k: v[indices] for k, v in self.non_tensor_batch.items()},
                dict(self.meta_info),
            )
            for indices in np.array_split(np.arange(len(self)), n)
        ]

    @staticmethod
    def concat(batches):
        meta = dict(batches[0].meta_info)
        if "metrics" in meta:
            meta["metrics"] = sum([b.meta_info["metrics"] for b in batches], [])
        return Batch(
            {k: torch.cat([b.batch[k] for b in batches]) for k in batches[0].batch},
            {k: np.concatenate([b.non_tensor_batch[k] for b in batches]) for k in batches[0].non_tensor_batch},
            meta,
        )

    def union(self, other):
        self.batch.update(other.batch)
        return self


def sample_batch():
    return Batch(
        {
            "prompts": torch.tensor([[0, 1, 2]] * 3 + [[0, 3, 4]] * 3),
            "responses": torch.tensor([[11, 12, 0], [21, 0, 0], [31, 32, 33], [41, 42, 0], [51, 0, 0], [61, 62, 63]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1]] * 2),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]] * 2),
        },
        {
            "uid": objects(["q1"] * 3 + ["q2"] * 3),
            "__num_turns__": np.array([2] * 6),
            "raw_prompt": objects([[{"role": "user", "content": q}] for q in ["Q1"] * 3 + ["Q2"] * 3]),
            "data_source": objects(["math_dapo"] * 6),
        },
        {"metrics": [{} for _ in range(6)], "global_steps": 1},
    )


def test_pairing_is_same_question_other_record_and_preserves_targets():
    batch = sample_batch()
    before = {k: v.clone() for k, v in batch.batch.items()}
    audit = context.attach_other_rollout_context(batch, batch.non_tensor_batch["uid"], 3)
    assert [r["context_batch_index"] for r in audit] == [1, 2, 0, 4, 5, 3]
    assert all(r["target_id"] != r["context_id"] for r in audit)
    assert batch.non_tensor_batch["teacher_context_response_ids"][0].tolist() == [21]
    for key in before:
        torch.testing.assert_close(before[key], batch.batch[key])


def test_identical_text_is_allowed_for_distinct_records():
    batch = sample_batch()
    batch.batch["responses"][1] = batch.batch["responses"][0]
    batch.batch["attention_mask"][1] = batch.batch["attention_mask"][0]
    batch.batch["response_mask"][1] = batch.batch["response_mask"][0]
    audit = context.attach_other_rollout_context(batch, batch.non_tensor_batch["uid"], 3)
    assert audit[0]["same_token_content"]
    assert audit[0]["target_id"] != audit[0]["context_id"]


@pytest.mark.parametrize("kind", ["singleton", "wrong_count", "wrong_prompt", "tool", "multimodal", "no_uid"])
def test_invalid_pairing_fails_explicitly(kind):
    batch = sample_batch()
    n, uids = 3, batch.non_tensor_batch["uid"]
    if kind == "singleton":
        n = 1
    elif kind == "wrong_count":
        uids[0] = "another"
    elif kind == "wrong_prompt":
        batch.batch["prompts"][1, 1] = 99
    elif kind == "tool":
        batch.batch["response_mask"][0, 0] = 0
    elif kind == "multimodal":
        batch.non_tensor_batch["multi_modal_inputs"] = objects([{"image": 1}] * 6)
    else:
        uids = None
    with pytest.raises(ValueError):
        context.attach_other_rollout_context(batch, uids, n)


def actual_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
    )
    fn.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "source,validate,expected_calls", [("other", False, 0), ("self", False, 1), ("self", True, 0), ("other", True, 0)]
)
def test_first_pass_defers_only_other_context(source, validate, expected_calls):
    cfg = OmegaConf.create(
        {"distillation": {"enabled": True, "full_rollout_context": True, "full_rollout_context_source": source}}
    )
    requests = []

    async def teacher(**kwargs):
        requests.append(kwargs)
        return "ids", "probs"

    worker = SimpleNamespace(
        config=cfg,
        distillation_enabled=True,
        teacher_key="data_source",
        teacher_server_manager=SimpleNamespace(
            distillation_config=cfg.distillation, compute_teacher_logprobs_single=teacher
        ),
    )
    output = SimpleNamespace(
        num_turns=2, response_mask=[1, 1], multi_modal_data=None, mm_processor_kwargs=None, extra_fields={}
    )
    method = actual_function(
        "verl/experimental/agent_loop/agent_loop.py",
        "_compute_teacher_logprobs",
        {"uses_other_rollout": context.uses_other_rollout},
        "AgentLoopWorker",
    )
    asyncio.run(method(worker, output, [1, 2], [3, 4], validate, {"raw_prompt": []}))
    assert len(requests) == expected_calls
    assert ("teacher_ids" in output.extra_fields) == bool(expected_calls)


@pytest.mark.parametrize("validate", [False, True])
def test_real_two_pass_methods_across_worker_boundaries(monkeypatch, tmp_path, validate):
    # Six responses, two groups of three, split among three workers: groups straddle chunks.
    cfg = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "full_rollout_context": True,
                "full_rollout_context_source": "other",
                "teacher_key": "data_source",
            },
            "trainer": {"mopd_pairing_dir": str(tmp_path)},
        }
    )
    pad_ns = {"F": torch.nn.functional}
    pad = actual_function("verl/experimental/teacher_loop/teacher_manager.py", "_pad_teacher_outputs", pad_ns)
    teacher_module = ModuleType("verl.experimental.teacher_loop.teacher_manager")
    teacher_module._pad_teacher_outputs = pad
    monkeypatch.setitem(sys.modules, teacher_module.__name__, teacher_module)
    ns = {
        "asyncio": asyncio,
        "torch": torch,
        "DataProto": Batch,
        "TensorDict": lambda data, batch_size: data,
        "uses_other_rollout": context.uses_other_rollout,
        "attach_other_rollout_context": context.attach_other_rollout_context,
        "write_pairing_report": context.write_pairing_report,
    }
    path = "verl/experimental/agent_loop/agent_loop.py"
    teacher_method = actual_function(path, "compute_other_rollout_teacher", ns, "AgentLoopWorker")
    manager_method = actual_function(path, "generate_sequences", ns, "AgentLoopManager")
    requests = []

    async def teacher(**kw):
        requests.append(kw)
        sequence = kw["sequence_ids"]
        ids = torch.tensor(sequence[1:] + [0], dtype=torch.int32).reshape(-1, 1)
        probs = torch.full(ids.shape, -float(kw["context_response_ids"][0]))
        return ids, probs

    workers = []
    for _ in range(3):
        worker = SimpleNamespace(
            config=cfg,
            teacher_key="data_source",
            tokenizer=SimpleNamespace(pad_token_id=0),
            teacher_server_manager=SimpleNamespace(compute_teacher_logprobs_single=teacher),
        )

        async def generate(chunk):
            return chunk

        async def score(chunk, worker=worker):
            return await teacher_method(worker, chunk)

        workers.append(
            SimpleNamespace(
                generate_sequences=SimpleNamespace(remote=generate),
                compute_other_rollout_teacher=SimpleNamespace(remote=score),
            )
        )
    manager = SimpleNamespace(
        config=cfg,
        rollout_config=SimpleNamespace(n=3),
        agent_loop_workers=workers,
        _performance_metrics=lambda metrics, output: {},
    )
    batch = sample_batch()
    batch.meta_info["validate"] = validate
    result = asyncio.run(manager_method(manager, batch))
    if validate:
        assert not requests
        assert "teacher_ids" not in result.batch
        assert not list(tmp_path.iterdir())
    else:
        assert len(requests) == 6
        assert result.batch["teacher_ids"].shape == (6, 6, 1)
        for i, request in enumerate(requests):
            response = result.batch["responses"][i][result.batch["response_mask"][i].bool()]
            # Padded prompt width=3: first response prediction must be at position 2.
            assert result.batch["teacher_ids"][i, 2 : 2 + len(response), 0].tolist() == response.tolist()
            assert request["routing_key"] == "math_dapo"
        assert requests[0]["context_response_ids"] == [21]
        assert requests[2]["context_response_ids"] == [11, 12]
        assert (tmp_path / "1.jsonl").exists()


def test_paper_configs_keep_student_data_loss_and_validation_identical(monkeypatch):
    for key in ("PAPER_DATA_ROOT", "STUDENT_MODEL", "TEACHER_MODEL", "RUN_OUTPUT"):
        monkeypatch.setenv(key, "/test/" + key)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        baseline = compose(config_name="opd_paper_v1")
        mopd = compose(config_name="mopd_paper_v1")
        OmegaConf.resolve(baseline)
        OmegaConf.resolve(mopd)
    for key in ("data", "algorithm", "actor_rollout_ref"):
        if key == "actor_rollout_ref":
            # Trace labels follow the experiment name; all sampling/optimizer settings must match.
            for label in ("project_name", "experiment_name"):
                mopd[key].rollout.trace[label] = baseline[key].rollout.trace[label]
        assert baseline[key] == mopd[key]
    assert baseline.distillation.distillation_loss == mopd.distillation.distillation_loss
    assert context.uses_other_rollout(mopd) and not context.uses_other_rollout(baseline)
    assert mopd.distillation.full_rollout_context_instruction == ""
    assert mopd.distillation.teacher_models.teacher_model.inference.max_model_len == (
        mopd.distillation.full_rollout_context_max_prompt_length + mopd.data.max_response_length + 1
    )
    assert baseline.trainer.total_training_steps == mopd.trainer.total_training_steps
