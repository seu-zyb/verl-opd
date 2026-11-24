"""Small exact-distribution and shifted-teacher checks, without Ray/GPU imports."""

import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "token_probability_debug", Path(__file__).resolve().parents[2] / "verl/utils/token_probability_debug.py"
)
debug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug)


class Tokenizer:
    eos_token_id = 4

    def get_vocab(self):
        return {"<|im_end|>": 3, "<|endoftext|>": 4}

    def convert_ids_to_tokens(self, token):
        return f"token-{token}"

    def decode(self, tokens, skip_special_tokens=False):
        assert not skip_special_tokens
        return " ".join(map(str, tokens))


def test_true_full_vocabulary_normalization_and_watch_outside_topk():
    probabilities = torch.tensor([[0.50, 0.25, 0.125, 0.075, 0.05], [0.1, 0.2, 0.5, 0.1, 0.1]])
    logits = probabilities.log().requires_grad_()
    original = logits.clone()
    out = debug.summarize_logits(logits, topk=2, watch_ids=[4], chunk_size=1)
    assert out[debug.DIAGNOSTIC_KEYS[0]].tolist() == [[0, 1], [2, 1]]
    torch.testing.assert_close(out[debug.DIAGNOSTIC_KEYS[1]].exp(), torch.tensor([[0.5, 0.25], [0.5, 0.2]]))
    torch.testing.assert_close(out[debug.DIAGNOSTIC_KEYS[2]].exp(), torch.tensor([[0.05], [0.1]]))
    torch.testing.assert_close(logits, original)
    assert not out[debug.DIAGNOSTIC_KEYS[1]].requires_grad


def fixture_data():
    # Prompt width 3, left padding 2. Response length 2, one right pad.
    # Teacher outputs are shifted: column 2 predicts first response token 1.
    tensors = {
        "prompts": torch.tensor([[4, 4, 0]]),
        "responses": torch.tensor([[1, 4, 4]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 0]]),
        "response_mask": torch.tensor([[1, 1, 0]]),
        "teacher_ids": torch.tensor([[[4], [4], [1], [4], [0], [4]]]),
        "teacher_logprobs": torch.tensor([[[0.0], [0.0], [-0.5], [-1.5], [0.0], [0.0]]]),
    }
    batch = SimpleNamespace(batch=tensors, non_tensor_batch={"uid": ["question-a"]})
    diagnostics = {
        key: value.unsqueeze(0) for key, value in debug.summarize_logits(torch.zeros(3, 5), 2, [4, 3]).items()
    }
    return batch, diagnostics


def write(tmp_path, batch, diagnostics):
    return debug.write_step(
        tmp_path,
        1,
        batch,
        Tokenizer(),
        torch.tensor([[-1.0, -2.0, 0.0]]),
        torch.ones(1, 3),
        diagnostics,
        [4, 3],
        1.0,
        "baseline",
    )


def test_teacher_shift_padding_and_eos_same_as_pad(tmp_path):
    batch, diagnostics = fixture_data()
    path = write(tmp_path, batch, diagnostics)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    assert row["phase"] == "before_actor_update"
    assert row["prompt_ids"] == [0]
    assert row["response_ids"] == [1, 4]  # Valid EOS preserved, padding excluded by mask.
    assert len(row["tokens"]) == 2
    assert [t["teacher_logp"] for t in row["tokens"]] == [-0.5, -1.5]
    assert row["tokens"][0]["logp_student_minus_teacher"] == -0.5
    assert row["tokens"][0]["student_other_mass"] == pytest.approx(0.6)
    assert debug.watched_token_ids(Tokenizer()) == [4, 3]


def test_misaligned_teacher_is_rejected_instead_of_probability_one(tmp_path):
    batch, diagnostics = fixture_data()
    batch.batch["teacher_ids"][0, 2, 0] = 4
    with pytest.raises(ValueError, match="alignment mismatch"):
        write(tmp_path, batch, diagnostics)
    assert not (tmp_path / "step_000001.jsonl.gz").exists()


def test_missing_teacher_candidate_is_null_not_zero(tmp_path):
    batch, diagnostics = fixture_data()
    batch.batch["teacher_ids"] = torch.zeros(1, 6, 2, dtype=torch.long)
    batch.batch["teacher_logprobs"] = torch.full((1, 6, 2), -2.0)
    path = write(tmp_path, batch, diagnostics)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    assert row["tokens"][0]["teacher_p"] is None


def test_nonfinite_logprob_serializes_without_nan(tmp_path):
    batch, diagnostics = fixture_data()
    batch.batch["teacher_logprobs"][0, 2, 0] = -torch.inf
    path = write(tmp_path, batch, diagnostics)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    assert row["tokens"][0]["teacher_logp"] is None
    assert row["tokens"][0]["teacher_p"] == 0.0
