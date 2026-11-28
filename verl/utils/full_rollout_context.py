# Copyright 2026 verl-opd contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Text-only, single-turn teacher hindsight context and causal output alignment.

No model weights or inference engines are needed by these helpers. The target
response is always the student's original token sequence, never re-tokenized.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def validate_shared_vocabulary(student_tokenizer: Any, teacher_tokenizer: Any) -> None:
    """Token-level distillation requires identical token-to-ID mappings."""
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "Full-rollout distillation requires identical student/teacher token-to-ID mappings. "
            "Choose compatible tokenizers; re-tokenizing the response would invalidate token alignment."
        )


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise ValueError("Full-rollout context supports text-only messages, not image/audio/video content.")
            parts.append(part["text"])
        return "\n".join(parts)
    raise ValueError("Full-rollout context requires string or text-block message content.")


def build_full_rollout_teacher_prompt(
    raw_prompt: Sequence[Mapping[str, Any]],
    response_ids: list[int],
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    instruction: str,
    max_prompt_length: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    context_source: str = "self",
) -> list[int]:
    """Build one teacher user message; return IDs ending at the assistant start.

    The supported input is optional system messages followed by exactly one user
    message. System text is retained, labelled, inside the single teacher message.
    Ground-truth answers, rewards and correctness flags are deliberately not read.
    """
    messages = list(raw_prompt)
    if not messages or any(not isinstance(message, Mapping) for message in messages):
        raise ValueError("Full-rollout context requires raw_prompt messages.")
    roles = [message.get("role") for message in messages]
    if roles[-1] != "user" or any(role != "system" for role in roles[:-1]):
        raise ValueError("Full-rollout context supports one user turn, optionally preceded by system messages.")
    if not response_ids:
        raise ValueError("Cannot compute full-rollout teacher supervision for an empty response.")
    if max_prompt_length <= 0:
        raise ValueError("full_rollout_context_max_prompt_length must be positive.")
    kwargs = dict(chat_template_kwargs or {})
    reserved = {"tokenize", "add_generation_prompt", "continue_final_message", "return_tensors", "return_dict"}
    if reserved.intersection(kwargs):
        raise ValueError(f"Teacher chat template kwargs cannot override {sorted(reserved.intersection(kwargs))}.")
    if not teacher_tokenizer.chat_template:
        raise ValueError("The teacher tokenizer must provide a chat template for full-rollout context.")

    problem_parts = []
    for message in messages:
        text = _text_content(message.get("content"))
        problem_parts.append(f"[Original system instructions]\n{text}" if message["role"] == "system" else text)
    # Remove conversation control tokens from the quoted context, not from the
    # actual response IDs used for teacher forcing (including EOS).
    attempt = student_tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    problem = "\n\n".join(problem_parts)
    if context_source == "other":
        content = f"Problem:\n{problem}\n\nAnother solution:\n{attempt}"
    elif context_source == "self":
        content = f"Problem:\n{problem}\n\n=== Student Attempt Begin ===\n{attempt}\n=== Student Attempt End ==="
    else:
        raise ValueError(f"Unknown full_rollout_context_source: {context_source}")
    if instruction.strip():
        content += f"\n\n{instruction.strip()}"
    tokenized = teacher_tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, **kwargs
    )
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    prompt_ids = list(tokenized)
    if not prompt_ids:
        raise ValueError("Teacher chat template produced an empty prompt.")
    if len(prompt_ids) > max_prompt_length:
        raise ValueError(
            f"Full-rollout teacher prompt has {len(prompt_ids)} tokens, exceeding "
            f"full_rollout_context_max_prompt_length={max_prompt_length}. "
            "Increase this budget and the teacher max_model_len, or reduce student lengths. "
            "The full student attempt is never silently truncated."
        )
    return prompt_ids


def align_teacher_response_outputs(
    teacher_ids: Any,
    teacher_logprobs: Any,
    teacher_prompt_length: int,
    student_prompt_length: int,
    response_length: int,
) -> tuple[Any, Any]:
    """Map next-token-aligned teacher tensors back to the student sequence.

    Both vLLM and SGLang adapters in verl shift prompt logprobs left by one and
    append a dummy row. With prefix length C, the first response prediction is at
    C-1, NOT C. Keep R rows, then place them at P-1 in the original P+R layout.
    Prompt/dummy rows are zero placeholders, excluded by the existing response mask.
    """
    if min(teacher_prompt_length, student_prompt_length, response_length) <= 0:
        raise ValueError("Teacher/student prompts and response must be non-empty.")
    if teacher_ids.ndim != 2 or teacher_ids.shape != teacher_logprobs.shape:
        raise ValueError("Teacher token IDs/logprobs must have matching [sequence, topk] shapes.")
    if teacher_ids.shape[0] != teacher_prompt_length + response_length:
        raise ValueError("Teacher output length does not match teacher prompt plus response.")
    shape = (student_prompt_length + response_length, teacher_ids.shape[1])
    aligned_ids = teacher_ids.new_zeros(shape)
    aligned_logprobs = teacher_logprobs.new_zeros(shape)
    source = slice(teacher_prompt_length - 1, teacher_prompt_length + response_length - 1)
    target = slice(student_prompt_length - 1, student_prompt_length + response_length - 1)
    aligned_ids[target] = teacher_ids[source]
    aligned_logprobs[target] = teacher_logprobs[source]
    return aligned_ids, aligned_logprobs
