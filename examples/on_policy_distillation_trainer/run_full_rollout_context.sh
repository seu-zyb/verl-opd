#!/usr/bin/env bash
# Qwen3-1.7B-Base <- Qwen3-14B, official verl v0.8.0 OPD + teacher hindsight.
# Run ONLY within an allocated GPU job. Default: 1 student GPU + 1 teacher GPU.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"

: "${STUDENT_MODEL:?Set STUDENT_MODEL to your Qwen3-1.7B-Base directory}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to your Qwen3-14B directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to a verl-format training parquet file}"
: "${VAL_FILE:?Set VAL_FILE to a verl-format validation parquet file}"
for task_path in "$STUDENT_MODEL" "$TEACHER_MODEL" "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -e "$task_path" ]]; then
        printf 'Path does not exist: %s\n' "$task_path" >&2
        exit 1
    fi
done

mode=${CONTEXT_MODE:-style}
case "$mode" in
    baseline) full_context=False ;;
    full|style) full_context=True ;;
    *) printf 'CONTEXT_MODE must be baseline, full, or style\n' >&2; exit 1 ;;
esac

export STUDENT_MODEL TEACHER_MODEL
export NNODES=1
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export TEACHER_TP=${TEACHER_TP:-1}
export TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.7}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:--1}
export PROJECT_NAME=${PROJECT_NAME:-full_rollout_opd}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_base_from_14b_${mode}}
export DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
export USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}

teacher_prefix_budget=${TEACHER_PREFIX_BUDGET:-4096}
teacher_model_len=$((teacher_prefix_budget + MAX_RESPONSE_LENGTH + 1))
# This template contains double quotes only. Hydra single quoting preserves its
# literal newlines and Jinja \n escapes (JSON-encoding it would double-escape them).
student_template="'$(cat "$script_dir/qwen3_base_chat_template.jinja")'"
# JSON lists keep dataset paths a single Hydra value.
train_file_arg=$(python -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "$TRAIN_FILE")
val_file_arg=$(python -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "$VAL_FILE")

extra=(
    "data.train_files=$train_file_arg"
    "data.val_files=$val_file_arg"
    "data.seed=${SEED:-2026}"
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED:-2026}"
    "actor_rollout_ref.actor.data_loader_seed=${SEED:-2026}"
    "actor_rollout_ref.model.custom_chat_template=$student_template"
    "actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.top_k=-1"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False"
    "trainer.logger=[console]"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-2}"
    "distillation.full_rollout_context=$full_context"
    "distillation.full_rollout_context_max_prompt_length=$teacher_prefix_budget"
    "distillation.teacher_models.teacher_model.inference.temperature=1.0"
    "distillation.teacher_models.teacher_model.inference.max_model_len=$teacher_model_len"
)
if [[ "$mode" == full ]]; then
    extra+=("distillation.full_rollout_context_instruction='以上学生作答可能包含错误。请独立判断并解答原问题。'")
fi
printf 'Mode=%s; student GPUs=%s; teacher GPUs=%s; teacher max length=%s\n' \
    "$mode" "$NGPUS_PER_NODE" "$TEACHER_WORLD_SIZE" "$teacher_model_len"
bash "$script_dir/run_qwen3_8b_fsdp.sh" "${extra[@]}" "$@"
