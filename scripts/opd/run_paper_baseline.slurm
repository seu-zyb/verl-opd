#!/bin/bash
#SBATCH --job-name=opd-paper
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --chdir=/seu_nvme/home/fenglei/213241973/code/verl-opd
#SBATCH -o opd-paper-%j.out
#SBATCH -e opd-paper-%j.err

set -eo pipefail
source /seu_share/apps/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate verl-opd

PROJECT_ROOT=/seu_nvme/home/fenglei/213241973/code/verl-opd
PATHS_FILE=${OPD_PATHS_FILE:-$PROJECT_ROOT/server_debug/full_rollout_paths.sh}
source "$PATHS_FILE"
set -u
export PROJECT_ROOT STUDENT_MODEL TEACHER_MODEL
export PAPER_DATA_ROOT=${PAPER_DATA_ROOT:-/seu_nvme/home/fenglei/213241973/data/opd_paper_v1}
export TRAIN_FILE="$PAPER_DATA_ROOT/prepared/train.parquet"
export VAL_FILE="$PAPER_DATA_ROOT/prepared/aime.parquet"
export SEED=${SEED:-2026}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export RUN_OUTPUT="${OUTPUT_ROOT}/paper-v1-opd/seed-${SEED}-job-${SLURM_JOB_ID}"
mkdir -p "$RUN_OUTPUT"
cd "$PROJECT_ROOT"

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}" TMPDIR="/tmp/t${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

for path in "$STUDENT_MODEL" "$TEACHER_MODEL" "$TRAIN_FILE" "$VAL_FILE" "$PAPER_DATA_ROOT/prepared/manifest.json"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing: $path"
        echo 'First run: python scripts/opd/prepare_paper_v1_data.py --data-root "$PAPER_DATA_ROOT"'
        exit 1
    fi
done

echo "Job=$SLURM_JOB_ID; OPD: 512 prompts x 8 responses, one update/batch, response cap=16384"
echo "Student=$STUDENT_MODEL; Teacher=$TEACHER_MODEL; Output=$RUN_OUTPUT"
hostname
nvidia-smi
cp "$0" "$RUN_OUTPUT/submitted.slurm"
cp "$PATHS_FILE" "$RUN_OUTPUT/paths.sh"
cp "$PAPER_DATA_ROOT/prepared/manifest.json" "$RUN_OUTPUT/data-manifest.json"
python -m pip list --format=json > "$RUN_OUTPUT/package_versions.json"
git rev-parse HEAD > "$RUN_OUTPUT/git-commit.txt"
git diff --stat > "$RUN_OUTPUT/git-diff-stat.txt"

# Every-step token diagnostics remains available; large paper batches make it expensive.
# Explicit opt-in: sbatch --export=ALL,TOKEN_PROBABILITY_EVERY=1 scripts/opd/run_paper_baseline.slurm
diagnostics=()
if [[ ! ${TOKEN_PROBABILITY_EVERY:-0} =~ ^[0-9]+$ ]]; then
    echo 'TOKEN_PROBABILITY_EVERY must be a nonnegative integer'; exit 1
fi
if (( ${TOKEN_PROBABILITY_EVERY:-0} > 0 )); then
    diagnostics+=("+trainer.token_probability_dir=$RUN_OUTPUT/token_probs"
        "+trainer.token_probability_every=$TOKEN_PROBABILITY_EVERY" "+trainer.token_probability_topk=10")
fi

# Save fully resolved settings before spending GPU time; extra CLI args are applied last.
python -m verl.trainer.main_ppo --config-name opd_paper_v1 --cfg job --resolve \
    "${diagnostics[@]}" "$@" > "$RUN_OUTPUT/resolved-config.yaml"
python -u scripts/opd/check_environment.py --gpu --timeout 300 \
    --train-file "$TRAIN_FILE" --val-file "$VAL_FILE" --report-dir "$RUN_OUTPUT/checks" \
    2>&1 | tee "$RUN_OUTPUT/environment-check.log"
python -u -m verl.trainer.main_ppo --config-name opd_paper_v1 \
    "${diagnostics[@]}" "$@" 2>&1 | tee "$RUN_OUTPUT/training.log"
echo "Completed: $RUN_OUTPUT"
