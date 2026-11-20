#!/bin/bash
#SBATCH --job-name=opd-style
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --chdir=/seu_nvme/home/fenglei/213241973/code/verl-opd
#SBATCH -o opd-style-%j.out
#SBATCH -e opd-style-%j.err

set -eo pipefail

echo "===== Slurm information ====="
echo "Job ID: ${SLURM_JOB_ID}"
hostname
nvidia-smi

# Do not enable nounset during conda activation (compiler hooks use optional vars).
source /seu_share/apps/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate verl-opd
source /seu_nvme/home/fenglei/213241973/experiment_configs/full_rollout_paths.sh
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

: "${PROJECT_ROOT:?Missing PROJECT_ROOT}"
: "${STUDENT_MODEL:?Missing STUDENT_MODEL}"
: "${TEACHER_MODEL:?Missing TEACHER_MODEL}"
: "${OUTPUT_ROOT:?Missing OUTPUT_ROOT}"
export PROJECT_ROOT STUDENT_MODEL TEACHER_MODEL

# Use validated compatible copies. Source data and the existing paths file stay intact.
export TRAIN_FILE=/seu_nvme/home/fenglei/213241973/data/gsm8k/compat_datasets3/train.parquet
export VAL_FILE=/seu_nvme/home/fenglei/213241973/data/gsm8k/compat_datasets3/test.parquet
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

# Short local paths avoid Ray's Unix socket path length limit.
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
export TMPDIR="/tmp/t${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

export SEED=2026
export PROJECT_NAME=full_rollout_opd
export EXPERIMENT_NAME="qwen3_1.7b_from_14b_style_seed${SEED}"
RUN_OUTPUT="${OUTPUT_ROOT}/style/seed-${SEED}-job-${SLURM_JOB_ID}"
mkdir -p "$RUN_OUTPUT"

# Keep the user's existing experiment budget and resource split.
export TOTAL_TRAINING_STEPS=100
export TOTAL_EPOCHS=1
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=1024
export TEACHER_PREFIX_BUDGET=4096
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export NGPUS_PER_NODE=2
export TEACHER_WORLD_SIZE=2
export ROLLOUT_TP=1
export TEACHER_TP=1
export ROLLOUT_GPU_MEM_UTIL=0.4
export TEACHER_GPU_MEM_UTIL=0.7
export SAVE_FREQ=20
export TEST_FREQ=20

for task_path in "$STUDENT_MODEL" "$TEACHER_MODEL" "$TRAIN_FILE" "$VAL_FILE" \
    "$PROJECT_ROOT/opd_fix_bundle/check_opd_environment.py"; do
    if [ ! -e "$task_path" ]; then
        echo "ERROR: missing $task_path"
        echo "Extract opd_fix_bundle.zip in PROJECT_ROOT and run prepare_gsm8k_compat.py first."
        exit 1
    fi
done

echo "===== Experiment configuration ====="
echo "Student model: $STUDENT_MODEL"
echo "Teacher model: $TEACHER_MODEL"
echo "Train: $TRAIN_FILE"
echo "Validation: $VAL_FILE"
echo "Output: $RUN_OUTPUT"
python -m pip list --format=json > "$RUN_OUTPUT/package_versions.json"
cp "$0" "$RUN_OUTPUT/submitted.slurm"

# One allocated job: check all GPU/import items, then go directly into training.
# pipefail stops training if any check fails, while the checker reports all items.
python -u "$PROJECT_ROOT/opd_fix_bundle/check_opd_environment.py" --gpu \
    --train-file "$TRAIN_FILE" --val-file "$VAL_FILE" \
    2>&1 | tee "$RUN_OUTPUT/environment-check.log"

echo "===== Start style experiment ====="
CONTEXT_MODE=style \
bash examples/on_policy_distillation_trainer/run_full_rollout_context.sh \
    "trainer.default_local_dir=${RUN_OUTPUT}" \
    "trainer.val_before_train=True"

echo "===== Style experiment completed ====="
echo "Results: $RUN_OUTPUT"
