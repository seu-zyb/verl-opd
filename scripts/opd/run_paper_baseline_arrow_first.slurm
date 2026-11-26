#!/bin/bash
#SBATCH --job-name=opd-baseline-fix
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --chdir=/seu_nvme/home/fenglei/213241973/code/verl-opd
#SBATCH -o opd-baseline-fix-%j.out
#SBATCH -e opd-baseline-fix-%j.err

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
# Self-contained job-local Arrow-first workaround; no trainer/loss changes.
# Create after conda activation so it cannot affect conda's own Python.
OPD_STARTUP_DIR="$RUN_OUTPUT/arrow-first-startup"
mkdir -p "$OPD_STARTUP_DIR"
cat > "$OPD_STARTUP_DIR/sitecustomize.py" <<'OPD_ARROW_HOOK'
"""Job-scoped workaround for the import-order SIGSEGV reproduced on h004.

Only put this directory on PYTHONPATH for an affected OPD job. Do not install it
in site-packages: it intentionally controls startup only for that job's Python
processes, including inherited subprocesses and Ray workers.
"""

import os

if os.environ.get("OPD_ARROW_FIRST") == "1":
    try:
        import pyarrow  # noqa: F401
    except Exception as error:
        # Python normally reports and ignores errors from sitecustomize. Do not
        # continue into the known bad import order if this workaround failed.
        raise SystemExit(f"OPD PyArrow startup preload failed: {error!r}") from error
    OPD_ARROW_PRELOADED = True
OPD_ARROW_HOOK
export OPD_ARROW_FIRST=1
export PYTHONPATH="$OPD_STARTUP_DIR:$PYTHONPATH"
startup_overrides=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPD_ARROW_FIRST=\"1\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$PYTHONPATH'"
)
python -c 'import sys, sitecustomize; assert "pyarrow" in sys.modules; assert sitecustomize.OPD_ARROW_PRELOADED; print("Arrow startup:", sitecustomize.__file__)'
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
    "${startup_overrides[@]}" "${diagnostics[@]}" "$@" > "$RUN_OUTPUT/resolved-config.yaml"
python -u scripts/opd/check_environment.py --gpu --timeout 300 \
    --train-file "$TRAIN_FILE" --val-file "$VAL_FILE" --report-dir "$RUN_OUTPUT/checks" \
    2>&1 | tee "$RUN_OUTPUT/environment-check.log"
python -u -m verl.trainer.main_ppo --config-name opd_paper_v1 \
    "${startup_overrides[@]}" "${diagnostics[@]}" "$@" 2>&1 | tee "$RUN_OUTPUT/training.log"
echo "Completed: $RUN_OUTPUT"
