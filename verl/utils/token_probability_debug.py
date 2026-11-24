"""Opt-in, pre-update token diagnostics; never changes the distillation objective."""

import gzip
import json
import math
from pathlib import Path

import torch

DIAGNOSTIC_KEYS = (
    "token_diag_top_ids",
    "token_diag_top_log_probs",
    "token_diag_watch_log_probs",
)


def watched_token_ids(tokenizer):
    """Watch both Qwen terminators without assuming that both stop generation."""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    vocab = tokenizer.get_vocab()
    for token in ("<|im_end|>", "<|endoftext|>"):
        if token in vocab:
            ids.append(int(vocab[token]))
    if not ids:
        raise ValueError("Token diagnostics requires at least one known EOS token")
    return list(dict.fromkeys(ids))


@torch.no_grad()
def summarize_logits(logits, topk, watch_ids, chunk_size=64):
    """Input [positions, vocab], already temperature-scaled; full-vocab normalization.

    Only a small chunk is converted to fp32, avoiding a second full fp32 logits
    buffer. The returned top-k probabilities are NOT renormalized within top-k.
    """
    if logits.ndim != 2 or not 0 < topk <= logits.shape[-1]:
        raise ValueError("Expected 2D logits and 0 < topk <= vocabulary size")
    if chunk_size <= 0 or not watch_ids or any(i < 0 or i >= logits.shape[-1] for i in watch_ids):
        raise ValueError("Invalid chunk size or watched token IDs")
    watch = torch.tensor(watch_ids, device=logits.device, dtype=torch.long)
    top_ids, top_logps, watch_logps = [], [], []
    for part in logits.detach().split(chunk_size, dim=0):
        part = part.float()
        normalizer = torch.logsumexp(part, dim=-1, keepdim=True)
        values, ids = torch.topk(part, k=topk, dim=-1)
        top_ids.append(ids)
        top_logps.append(values - normalizer)
        watch_logps.append(part.index_select(-1, watch) - normalizer)
    return dict(zip(DIAGNOSTIC_KEYS, (torch.cat(top_ids), torch.cat(top_logps), torch.cat(watch_logps)), strict=True))


def _number(value):
    value = float(value)
    # Keep JSON standards compliant; a null logprob can signal non-finite input.
    return value if math.isfinite(value) else None


def _probability(logp):
    value = float(logp)
    return math.exp(value) if math.isfinite(value) else (0.0 if value == -math.inf else None)


def write_step(
    directory, step, batch, tokenizer, log_probs, entropy, diagnostics, watch_ids, temperature, context_mode
):
    """Write one gzip JSONL per optimizer step, one sample per line.

    Teacher tensors have already been shifted by the vLLM extractor. In padded
    coordinates, response t is stored at prompt_width - 1 + t. Verify IDs before
    using this value; zero-valued padding must never be reported as probability 1.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = batch.batch
    responses = tensors["responses"].detach().cpu()
    masks = tensors["response_mask"].detach().cpu().bool()
    prompt_width = tensors["prompts"].shape[1]
    response_width = responses.shape[1]
    log_probs, entropy = log_probs.detach().cpu(), entropy.detach().cpu()
    diagnostics = {k: v.detach().cpu() for k, v in diagnostics.items()}
    for name in DIAGNOSTIC_KEYS:
        if diagnostics[name].shape[:2] != responses.shape:
            raise ValueError(f"{name} is not aligned to responses")
    teacher_ids = tensors.get("teacher_ids")
    teacher_logps = tensors.get("teacher_logprobs")
    if (teacher_ids is None) != (teacher_logps is None):
        raise ValueError("Teacher token IDs and logprobs must be present together")
    if teacher_ids is not None:
        if teacher_ids.ndim != 3 or teacher_ids.shape != teacher_logps.shape:
            raise ValueError("Expected matching [batch, sequence, candidates] teacher tensors")
        if teacher_ids.shape[1] != prompt_width + response_width:
            raise ValueError("Expected teacher tensors aligned to the padded student sequence")
        teacher_ids = teacher_ids[:, prompt_width - 1 : prompt_width + response_width - 1].detach().cpu()
        teacher_logps = teacher_logps[:, prompt_width - 1 : prompt_width + response_width - 1].detach().cpu()
    target = directory / f"step_{step:06d}.jsonl.gz"
    temporary = directory / f"step_{step:06d}.jsonl.gz.tmp"
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for i in range(responses.shape[0]):
            tokens = []
            for t in masks[i].nonzero(as_tuple=True)[0].tolist():
                token_id = int(responses[i, t])
                student_lp = float(log_probs[i, t])
                teacher_lp = None
                if teacher_ids is not None:
                    matches = teacher_ids[i, t].eq(token_id).nonzero(as_tuple=True)[0]
                    if teacher_ids.shape[-1] == 1 and len(matches) != 1:
                        raise ValueError(f"Teacher/student token alignment mismatch at sample {i}, position {t}")
                    if len(matches):
                        teacher_lp = float(teacher_logps[i, t, int(matches[0])])
                top_ids = diagnostics["token_diag_top_ids"][i, t].tolist()
                top_lps = diagnostics["token_diag_top_log_probs"][i, t].tolist()
                top_ps = [_probability(x) for x in top_lps]
                watched_lps = diagnostics["token_diag_watch_log_probs"][i, t].tolist()
                tokens.append(
                    {
                        "position": t,
                        "token_id": token_id,
                        "token": tokenizer.convert_ids_to_tokens(token_id),
                        "student_logp": _number(student_lp),
                        "student_p": _probability(student_lp),
                        "teacher_logp": _number(teacher_lp) if teacher_lp is not None else None,
                        "teacher_p": _probability(teacher_lp) if teacher_lp is not None else None,
                        "logp_student_minus_teacher": _number(student_lp - teacher_lp)
                        if teacher_lp is not None
                        else None,
                        "student_entropy": _number(entropy[i, t]),
                        "student_topk": [
                            {"id": tid, "token": tokenizer.convert_ids_to_tokens(tid), "logp": _number(lp), "p": p}
                            for tid, lp, p in zip(top_ids, top_lps, top_ps, strict=True)
                        ],
                        "student_other_mass": max(0.0, 1.0 - sum(top_ps))
                        if all(p is not None for p in top_ps)
                        else None,
                        "student_watch": [
                            {
                                "id": tid,
                                "token": tokenizer.convert_ids_to_tokens(tid),
                                "logp": _number(lp),
                                "p": _probability(lp),
                            }
                            for tid, lp in zip(watch_ids, watched_lps, strict=True)
                        ],
                    }
                )
            prompt_mask = tensors["attention_mask"][i, :prompt_width].detach().cpu().bool()
            prompt_ids = tensors["prompts"][i].detach().cpu()[prompt_mask].tolist()
            response_attention = tensors["attention_mask"][i, prompt_width:].detach().cpu().bool()
            response_ids = responses[i][response_attention].tolist()
            uid = batch.non_tensor_batch.get("uid")
            row = {
                "step": int(step),
                "phase": "before_actor_update",
                "batch_index": i,
                "uid": str(uid[i]) if uid is not None else None,
                "temperature": float(temperature),
                "teacher_context_mode": context_mode,
                "distribution": "student_full_softmax_topk_and_watch_before_sampling_filters",
                "teacher_values": "probability_of_actual_student_token_only",
                "tokenizer_eos_id": tokenizer.eos_token_id,
                "watched_ids_are_not_necessarily_runtime_stop_ids": True,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "output": tokenizer.decode(response_ids, skip_special_tokens=False),
                "response_capacity": response_width,
                "response_at_capacity": len(response_ids) == response_width,
                "tokens": tokens,
            }
            context = batch.non_tensor_batch.get("teacher_context_response_ids")
            if context is not None:
                value = context[i]
                row["teacher_context_response_ids"] = value.tolist() if hasattr(value, "tolist") else value
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(target)
    return target
