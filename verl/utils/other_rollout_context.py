"""Pair independent records within a prompt group without inspecting rewards or labels."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def uses_other_rollout(config):
    distillation = config.get("distillation", {})
    return (
        distillation.get("enabled", False)
        and distillation.get("full_rollout_context", False)
        and distillation.get("full_rollout_context_source", "self") == "other"
    )


def attach_other_rollout_context(batch, uids, expected_n):
    """Called on the full, ordered rollout batch BEFORE splitting teacher requests.

    Cyclic pairing i -> (i+1) mod n uses each response once as context. Equal token
    contents are allowed: distinct stochastic samples may legitimately coincide.
    The uid identifies the input problem for each rollout.
    """
    if expected_n < 2:
        raise ValueError("MOPD requires rollout.n >= 2.")
    if uids is None or len(uids) != len(batch.batch["responses"]):
        raise ValueError("MOPD requires the input prompt uid for every rollout record.")
    tensors = batch.batch
    prompt_width = tensors["prompts"].shape[1]
    
    # 注意力掩码区分有效 token 和 padding。
    response_attention = tensors["attention_mask"][:, prompt_width:].bool()
    if not (response_attention == tensors["response_mask"].bool()).all():
        raise ValueError("MOPD supports single-turn responses without tool observations only.")
    # 检查是否为单轮对话。
    if any(turns != 2 for turns in batch.non_tensor_batch["__num_turns__"]):
        raise ValueError("MOPD requires single-turn responses.")
    # 检查是否包含多模态输入。
    if any(bool(value) for value in batch.non_tensor_batch.get("multi_modal_inputs", [])):
        raise ValueError("MOPD currently supports text-only prompts.")

    # 每个问题对应的 rollout 在 batch 中的索引。
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[str(uid)].append(i)
    # 去除 padding token。
    responses = [
        tensors["responses"][i][mask].detach().cpu().numpy().astype(np.int32, copy=True)
        for i, mask in enumerate(response_attention)
    ]
    # 确保所有 rollout 非空。
    if any(len(response) == 0 for response in responses):
        raise ValueError("MOPD cannot pair an empty rollout.")
        
    size = len(responses)  # rollout 总数。
    context_ids, target_ids, context_record_ids = (np.empty(size, dtype=object) for _ in range(3))
    # context_ids 是作为上下文的 rollout token 序列。
    # context_record_ids 是类似 A:1 的标记。
    same_content = np.zeros(size, dtype=bool)
    audit = []
    # indices 是某个 uid 对应的 rollout 下标。
    for uid, indices in groups.items():
        if len(indices) != expected_n:  # 每题的 rollout 数量必须等于预期值。
            raise ValueError(f"MOPD prompt group {uid} has {len(indices)} responses, expected {expected_n}.")
        prompt_tokens = []
        for i in indices:
            mask = tensors["attention_mask"][i, :prompt_width].bool()
            prompt_tokens.append(tensors["prompts"][i][mask].tolist())
        if any(tokens != prompt_tokens[0] for tokens in prompt_tokens[1:]):
            raise ValueError(f"MOPD uid {uid} contains different student prompts.")
        for ordinal, i in enumerate(indices):
            context_ordinal = (ordinal + 1) % len(indices)
            j = indices[context_ordinal]
            context_ids[i] = responses[j]  # 配对规则：rollout_i -> rollout_i+1。
            target_ids[i] = f"{uid}:{ordinal}"
            context_record_ids[i] = f"{uid}:{context_ordinal}"
            same_content[i] = np.array_equal(responses[i], responses[j])
            audit.append(
                {
                    "uid": uid,
                    "target_id": target_ids[i],
                    "context_id": context_record_ids[i],
                    "target_batch_index": i,
                    "context_batch_index": j,
                    "target_tokens": len(responses[i]),
                    "context_tokens": len(responses[j]),
                    "target_sha256": hashlib.sha256(responses[i].astype("<i4").tobytes()).hexdigest(),
                    "context_sha256": hashlib.sha256(responses[j].astype("<i4").tobytes()).hexdigest(),
                    "same_token_content": bool(same_content[i]),
                }
            )
    # 将配对信息合并到 batch 中。
    batch.non_tensor_batch.update(
        {
            "teacher_context_response_ids": context_ids,
            "mopd_target_id": target_ids,
            "mopd_context_id": context_record_ids,
            "teacher_context_matches_target": same_content,
        }
    )
    return audit


def write_pairing_report(directory, step, audit):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{int(step)}.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in audit:
            handle.write(json.dumps({"step": int(step), **row}, ensure_ascii=False) + "\n")
    temporary.replace(target)
