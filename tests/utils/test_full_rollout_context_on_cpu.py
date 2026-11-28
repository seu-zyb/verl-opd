# Copyright 2026 verl-opd contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU regression tests, without loading Ray, vLLM or any model weights.

Load the lightweight helper by path to avoid verl's GPU/distributed import graph.
Manager tests execute the real manager code with mocked inference/config services.
They do not establish GPU/runtime compatibility or model quality.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path, monkeypatch=None):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


context = load_module("full_rollout_context_cpu", "verl/utils/full_rollout_context.py")


class Tokenizer:
    chat_template = "fake-native-template"

    def __init__(self, prefix_length=12):
        self.prefix_length = prefix_length
        self.messages = None
        self.decoded_ids = None

    def get_vocab(self):
        return {"a": 0, "b": 1}

    def decode(self, ids, **kwargs):
        self.decoded_ids = list(ids)
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return "3 + 5 = 9"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        self.messages = messages
        self.kwargs = kwargs
        return list(range(100, 100 + self.prefix_length))


def build(**kwargs):
    defaults = dict(
        raw_prompt=[{"role": "user", "content": "What is 3 + 5?"}],
        response_ids=[11, 12, 13],
        student_tokenizer=Tokenizer(),
        teacher_tokenizer=Tokenizer(),
        instruction="Correct mistakes while preserving valid expression.",
        max_prompt_length=12,
        chat_template_kwargs={"enable_thinking": False},
    )
    defaults.update(kwargs)
    return context.build_full_rollout_teacher_prompt(**defaults)


def test_one_user_message_contains_whole_attempt_and_preserves_system_text():
    teacher, student = Tokenizer(), Tokenizer()
    response = [11, 12, 13]
    raw_prompt = [
        {"role": "system", "content": "Use algebra."},
        {"role": "user", "content": [{"type": "text", "text": "What is 3 + 5?"}]},
    ]
    assert (
        len(build(raw_prompt=raw_prompt, response_ids=response, teacher_tokenizer=teacher, student_tokenizer=student))
        == 12
    )
    assert len(teacher.messages) == 1
    assert teacher.messages[0]["role"] == "user"
    text = teacher.messages[0]["content"]
    assert "Use algebra." in text and "What is 3 + 5?" in text and "3 + 5 = 9" in text
    assert "Reference Solution" not in text
    assert student.decoded_ids == response == [11, 12, 13]
    assert teacher.kwargs["enable_thinking"] is False


def test_empty_instruction_is_supported():
    teacher = Tokenizer()
    build(instruction="", teacher_tokenizer=teacher)
    assert teacher.messages[0]["content"].endswith("=== Student Attempt End ===")


def test_other_rollout_is_labelled_before_attempt_and_instructed_after():
    teacher = Tokenizer()
    instruction = "This is an unverified attempt. Solve the original problem independently."
    build(context_source="other", instruction=instruction, teacher_tokenizer=teacher)
    assert teacher.messages[0]["content"] == (
        "Problem:\nWhat is 3 + 5?\n\n"
        "Another solution:\n3 + 5 = 9\n\n"
        + instruction
    )


def test_context_budget_never_silently_truncates():
    with pytest.raises(ValueError, match="never silently truncated"):
        build(max_prompt_length=11)


@pytest.mark.parametrize(
    "override",
    [
        {"response_ids": []},
        {"raw_prompt": []},
        {"raw_prompt": [{"role": "assistant", "content": "history"}]},
        {"raw_prompt": [{"role": "user", "content": [{"type": "image"}]}]},
        {"chat_template_kwargs": {"continue_final_message": True}},
        {"max_prompt_length": 0},
    ],
)
def test_unsupported_inputs_fail(override):
    with pytest.raises(ValueError):
        build(**override)


def test_tokenizer_mismatch_is_rejected_even_with_same_vocabulary_size():
    other = Tokenizer()
    other.get_vocab = lambda: {"a": 1, "b": 0}
    with pytest.raises(ValueError, match="identical"):
        context.validate_shared_vocabulary(Tokenizer(), other)
    context.validate_shared_vocabulary(Tokenizer(), Tokenizer())


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("response_length", [1, 3])
def test_causal_alignment_includes_first_and_last_response_tokens(width, response_length):
    c, p, r = 12, 4, response_length
    ids = torch.arange((c + r) * width).reshape(c + r, width).int()
    logprobs = -ids.float()
    aligned_ids, aligned_probs = context.align_teacher_response_outputs(ids, logprobs, c, p, r)
    assert aligned_ids.shape == (p + r, width)
    # Position p-1 predicts the first response token; p+r-2 predicts the last.
    torch.testing.assert_close(aligned_ids[p - 1], ids[c - 1])
    torch.testing.assert_close(aligned_ids[p + r - 2], ids[c + r - 2])
    torch.testing.assert_close(aligned_probs[p - 1 : p + r - 1], logprobs[c - 1 : c + r - 1])
    assert torch.count_nonzero(aligned_ids[: p - 1]) == 0
    assert torch.count_nonzero(aligned_ids[-1]) == 0  # dummy row
    assert aligned_ids.dtype == ids.dtype and aligned_probs.dtype == logprobs.dtype


@pytest.fixture
def manager_module(monkeypatch):
    """Stub only service imports; execute the actual manager and padding code."""
    config_module = ModuleType("verl.utils.config")
    config_module.omega_conf_to_dataclass = lambda value: value
    types_module = ModuleType("verl.workers.config")
    for name in ("DistillationConfig", "DistillationLossConfig", "DistillationTeacherModelConfig"):
        setattr(types_module, name, SimpleNamespace)
    client_module = ModuleType("verl.workers.rollout.llm_server")
    client_module.LLMServerClient = object
    for name, module in {
        "verl.utils.config": config_module,
        "verl.utils.full_rollout_context": context,
        "verl.workers.config": types_module,
        "verl.workers.rollout.llm_server": client_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return load_module("teacher_manager_cpu", "verl/experimental/teacher_loop/teacher_manager.py")


class Client:
    async def generate(self, **kwargs):
        self.request = kwargs
        tokens = kwargs["prompt_ids"]
        # Actual verl engine contract: row i holds the prediction for token i+1.
        ids = [[token] for token in tokens[1:]] + [[0]]
        self.ids = torch.tensor(ids, dtype=torch.int32)
        self.probs = -torch.arange(len(tokens), dtype=torch.float32).reshape(-1, 1)
        return SimpleNamespace(extra_fields={"prompt_ids": ids, "prompt_logprobs": self.probs.tolist()})


def make_manager(module, enabled):
    manager = module.AsyncTeacherLLMServerManager.__new__(module.AsyncTeacherLLMServerManager)
    manager.distillation_config = SimpleNamespace(
        full_rollout_context=enabled,
        full_rollout_context_instruction="Check correctness and retain valid wording.",
        full_rollout_context_max_prompt_length=12,
        full_rollout_context_chat_template_kwargs={"enable_thinking": False},
    )
    manager.distillation_loss_config = SimpleNamespace(topk=4, loss_settings=SimpleNamespace(use_topk=False))
    manager.teacher_key = "data_source"
    manager.teacher_model_configs = {
        "default": SimpleNamespace(inference=SimpleNamespace(temperature=1.0, max_model_len=100))
    }
    manager.student_tokenizer = Tokenizer()
    manager.teacher_tokenizers = {"default": Tokenizer()}
    client = Client()
    manager.teacher_client = {"default": client}
    return manager, client


def test_other_rollout_manager_quotes_b_but_scores_original_a(manager_module):
    manager, client = make_manager(manager_module, True)
    manager.distillation_config.full_rollout_context_source = "other"
    sequence = [1, 2, 3, 4, 21, 22, 23]
    ids, probs = asyncio.run(
        manager.compute_teacher_logprobs_single(
            sequence_ids=sequence,
            student_prompt_length=4,
            raw_prompt=[{"role": "user", "content": "3 + 5?"}],
            context_response_ids=[31, 32],
        )
    )
    assert manager.student_tokenizer.decoded_ids == [31, 32]
    assert client.request["prompt_ids"][-3:] == [21, 22, 23]
    assert ids[3:6, 0].tolist() == [21, 22, 23]
    assert probs[3:6, 0].tolist() == [-11.0, -12.0, -13.0]
    assert sequence == [1, 2, 3, 4, 21, 22, 23]
    with pytest.raises(ValueError, match="companion"):
        asyncio.run(
            manager.compute_teacher_logprobs_single(
                sequence_ids=sequence,
                student_prompt_length=4,
                raw_prompt=[{"role": "user", "content": "Q"}],
            )
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_manager_baseline_and_context_with_padding(manager_module, enabled):
    manager, client = make_manager(manager_module, enabled)
    sequence = [1, 2, 3, 4, 21, 22, 23]
    ids, probs = asyncio.run(
        manager.compute_teacher_logprobs_single(
            sequence_ids=sequence,
            student_prompt_length=4,
            raw_prompt=[{"role": "user", "content": "3 + 5?"}],
            routing_key="gsm8k",
        )
    )
    assert sequence == [1, 2, 3, 4, 21, 22, 23]
    assert ids.shape == probs.shape == (7, 1)
    assert client.request["prompt_ids"][-3:] == [21, 22, 23]
    assert ids[3:6, 0].tolist() == [21, 22, 23]
    if not enabled:
        assert client.request["prompt_ids"] == sequence
        torch.testing.assert_close(ids, client.ids)
        torch.testing.assert_close(probs, client.probs)
    else:
        assert len(client.request["prompt_ids"]) == 15
        assert probs[3:6, 0].tolist() == [-11.0, -12.0, -13.0]
    padded_ids, padded_probs = manager_module._pad_teacher_outputs(ids, probs, 6, 5, 4, 3, 0)
    assert padded_ids.shape == padded_probs.shape == (1, 11, 1)
    assert padded_ids[0, 5:8, 0].tolist() == [21, 22, 23]


@pytest.mark.parametrize("invalid", ["multimodal", "missing_prompt", "too_long"])
def test_manager_rejects_invalid_requests_before_inference(manager_module, invalid):
    manager, client = make_manager(manager_module, True)
    kwargs = dict(
        sequence_ids=[1, 2, 3, 4, 21, 22, 23],
        student_prompt_length=4,
        raw_prompt=[{"role": "user", "content": "3 + 5?"}],
    )
    if invalid == "multimodal":
        kwargs["multi_modal_data"] = {"images": ["image"]}
    elif invalid == "missing_prompt":
        kwargs["raw_prompt"] = None
    else:
        manager.teacher_model_configs["default"].inference.max_model_len = 15  # needs 16 including dummy generation
    with pytest.raises(ValueError):
        asyncio.run(manager.compute_teacher_logprobs_single(**kwargs))
    assert not hasattr(client, "request")


def test_yaml_defaults_are_opt_in():
    config = OmegaConf.load(ROOT / "verl/trainer/config/distillation/distillation.yaml")
    assert config.full_rollout_context is False
    assert config.full_rollout_context_max_prompt_length == 4096
    assert config.full_rollout_context_chat_template_kwargs.enable_thinking is False
    assert "错误" in config.full_rollout_context_instruction


@pytest.fixture
def distillation_config_module(monkeypatch):
    # Load the real config/conversion code, replacing only the unrelated rollout
    # class import. Each test supplies an explicit inference configuration.
    base = load_module("verl.base_config", "verl/base_config.py", monkeypatch)
    conversion = load_module("verl.utils.config", "verl/utils/config.py", monkeypatch)
    rollout_module = ModuleType("cpu_distillation.rollout")
    rollout_module.RolloutConfig = base.BaseConfig
    monkeypatch.setitem(sys.modules, "cpu_distillation.rollout", rollout_module)
    assert conversion.omega_conf_to_dataclass(SimpleNamespace(value=1)).value == 1
    return load_module("cpu_distillation.distillation", "verl/workers/config/distillation.py", monkeypatch)


@pytest.mark.parametrize("enabled,expected", [(False, 49), (True, 81)])
def test_teacher_context_budget_includes_target_and_generated_token(distillation_config_module, enabled, expected):
    module = distillation_config_module
    inference = SimpleNamespace(
        prompt_length=32,
        response_length=16,
        max_model_len=expected,
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    teacher = module.DistillationTeacherModelConfig(model_path="teacher", inference=inference)
    config = module.DistillationConfig(
        enabled=True,
        full_rollout_context=enabled,
        full_rollout_context_max_prompt_length=64,
        n_gpus_per_node=1,
        nnodes=1,
        teacher_models={"teacher_model": teacher},
        distillation_loss=SimpleNamespace(loss_settings=SimpleNamespace(use_topk=False), topk=1),
    )
    assert config.teacher_models["default"].inference.prompt_length == expected - 1
    assert inference.response_length == 1


def test_teacher_budget_fails_before_server_start(distillation_config_module):
    module = distillation_config_module
    teacher = module.DistillationTeacherModelConfig(
        inference=SimpleNamespace(prompt_length=32, response_length=16, max_model_len=80)
    )
    with pytest.raises(ValueError, match="requires room"):
        teacher.validate_and_prepare_for_distillation(use_topk=False, topk=1, teacher_prompt_length=64)
    assert teacher.inference.response_length == 16  # validation happens before mutation
