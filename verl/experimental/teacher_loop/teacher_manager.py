# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.full_rollout_context import (
    align_teacher_response_outputs,
    build_full_rollout_teacher_prompt,
    validate_shared_vocabulary,
)
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
        student_tokenizer: Any = None,
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers: dict[str, Any] = {}
        if self.distillation_config.full_rollout_context:
            if student_tokenizer is None:
                raise ValueError("Full-rollout context requires the student tokenizer.")
            from verl.utils.fs import copy_to_local
            from verl.utils.tokenizer import hf_tokenizer

            for key, model_config in self.teacher_model_configs.items():
                tokenizer = hf_tokenizer(copy_to_local(model_config.model_path), trust_remote_code=False)
                validate_shared_vocabulary(student_tokenizer, tokenizer)
                self.teacher_tokenizers[key] = tokenizer

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        student_prompt_length: Optional[int] = None,
        raw_prompt: Optional[list[dict[str, Any]]] = None,
        context_response_ids: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        original_sequence_length = len(sequence_ids)
        teacher_prompt_length = None
        if self.distillation_config.full_rollout_context:
            if multi_modal_data or mm_processor_kwargs:
                raise ValueError("Full-rollout context currently supports text-only, single-turn distillation.")
            if student_prompt_length is None or not 0 < student_prompt_length < original_sequence_length:
                raise ValueError("Full-rollout context requires a non-empty student prompt and response.")
            if raw_prompt is None:
                raise ValueError("Full-rollout context requires the original raw_prompt messages.")
            response_ids = sequence_ids[student_prompt_length:]
            context_source = getattr(self.distillation_config, "full_rollout_context_source", "self")
            if context_source == "other":
                if not context_response_ids:
                    raise ValueError("Other-rollout context requires an explicit non-empty companion response.")
                quoted_ids = context_response_ids
            elif context_source == "self":
                quoted_ids = response_ids
            else:
                raise ValueError(f"Unknown full_rollout_context_source: {context_source}")
            teacher_prompt = build_full_rollout_teacher_prompt(
                raw_prompt=raw_prompt,
                response_ids=quoted_ids,
                student_tokenizer=self.student_tokenizer,
                teacher_tokenizer=self.teacher_tokenizers[teacher_key],
                instruction=self.distillation_config.full_rollout_context_instruction,
                max_prompt_length=self.distillation_config.full_rollout_context_max_prompt_length,
                chat_template_kwargs=self.distillation_config.full_rollout_context_chat_template_kwargs,
                context_source=context_source,
            )
            teacher_prompt_length = len(teacher_prompt)
            sequence_ids = teacher_prompt + response_ids
            max_model_len = teacher_model_config.inference.max_model_len
            if max_model_len is not None and len(sequence_ids) + 1 > max_model_len:
                raise ValueError(
                    f"Teacher request requires {len(sequence_ids) + 1} tokens, exceeding {max_model_len=}."
                )
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        if teacher_prompt_length is not None:
            teacher_ids, teacher_logprobs = align_teacher_response_outputs(
                teacher_ids,
                teacher_logprobs,
                teacher_prompt_length=teacher_prompt_length,
                student_prompt_length=student_prompt_length,
                response_length=original_sequence_length - student_prompt_length,
            )
        return teacher_ids, teacher_logprobs
