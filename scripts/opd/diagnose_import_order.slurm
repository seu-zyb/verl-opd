#!/bin/bash
#SBATCH --job-name=opd-import
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --chdir=/seu_nvme/home/fenglei/213241973/code/verl-opd
#SBATCH -o opd-import-%j.out
#SBATCH -e opd-import-%j.err

set -eo pipefail
source /seu_share/apps/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate verl-opd
PROJECT_ROOT=/seu_nvme/home/fenglei/213241973/code/verl-opd
source "${OPD_PATHS_FILE:-$PROJECT_ROOT/server_debug/full_rollout_paths.sh}"
set -u
cd "$PROJECT_ROOT"
export PROJECT_ROOT STUDENT_MODEL TEACHER_MODEL
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}" TMPDIR="/tmp/t${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
DIAG_OUTPUT="$OUTPUT_ROOT/import-diagnostics/job-${SLURM_JOB_ID}"
mkdir -p "$DIAG_OUTPUT"
cp "$0" "$DIAG_OUTPUT/submitted.slurm"
git rev-parse HEAD > "$DIAG_OUTPUT/git-commit.txt"
hostname
nvidia-smi
python -u scripts/opd/diagnose_import_order.py --output-dir "$DIAG_OUTPUT" --native-backtrace "$@"
