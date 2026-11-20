#!/bin/bash

#SBATCH --job-name=opd-style
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH -o opd-style-%j.out
#SBATCH -e opd-style-%j.err

set -eo pipefail

echo "===== Slurm information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "===== GPU ====="
nvidia-smi

# 激活 Conda 环境
source /seu_share/apps/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate verl-opd

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# 加载本机模型、数据和输出路径
source /seu_nvme/home/fenglei/213241973/experiment_configs/full_rollout_paths.sh

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

# Ray 和其他临时文件放在个人 NVMe
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}"
export TMPDIR="/tmp/t${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

echo "===== Environment ====="
which python
python --version
python -m pip check

python - <<'PY'
import torch
import flash_attn

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Visible GPU count:", torch.cuda.device_count())
print("FlashAttention:", flash_attn.__version__)

if torch.cuda.device_count() != 4:
    raise RuntimeError(
        f"Expected 4 GPUs, but found {torch.cuda.device_count()}"
    )
PY

# 固定随机种子
export SEED=2026

# 实验名称
export PROJECT_NAME="full_rollout_opd"
export EXPERIMENT_NAME="qwen3_1.7b_from_14b_style_seed${SEED}"

# 每次作业使用独立输出目录
RUN_OUTPUT="${OUTPUT_ROOT}/style/seed-${SEED}-job-${SLURM_JOB_ID}"
mkdir -p "$RUN_OUTPUT"

# 第一轮正式可行性实验
export TOTAL_TRAINING_STEPS=100
export TOTAL_EPOCHS=1
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8

# 长度设置
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=1024
export TEACHER_PREFIX_BUDGET=4096
export PPO_MAX_TOKEN_LEN_PER_GPU=4096

# 2张学生 GPU + 2张教师 GPU
export NGPUS_PER_NODE=2
export TEACHER_WORLD_SIZE=2
export ROLLOUT_TP=1
export TEACHER_TP=1

export ROLLOUT_GPU_MEM_UTIL=0.4
export TEACHER_GPU_MEM_UTIL=0.7

# 每20步保存和验证
export SAVE_FREQ=20
export TEST_FREQ=20

echo "===== Experiment configuration ====="
echo "Mode: style"
echo "Student model: $STUDENT_MODEL"
echo "Teacher model: $TEACHER_MODEL"
echo "Training data: $TRAIN_FILE"
echo "Validation data: $VAL_FILE"
echo "Output directory: $RUN_OUTPUT"
echo "Training steps: $TOTAL_TRAINING_STEPS"
echo "Seed: $SEED"

for path in \
    "$STUDENT_MODEL" \
    "$TEACHER_MODEL" \
    "$TRAIN_FILE" \
    "$VAL_FILE"
do
    if [ ! -e "$path" ]; then
        echo "ERROR: path does not exist: $path"
        exit 1
    fi
done

echo "===== Start style experiment ====="

CONTEXT_MODE=style \
bash examples/on_policy_distillation_trainer/run_full_rollout_context.sh \
    "trainer.default_local_dir=${RUN_OUTPUT}" \
    "trainer.val_before_train=True"

echo "===== Style experiment completed ====="
echo "Results: $RUN_OUTPUT"
