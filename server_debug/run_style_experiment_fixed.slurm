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

# Compatibility entrypoint. The maintained experiment lives in scripts/opd/.
set -eo pipefail
exec bash /seu_nvme/home/fenglei/213241973/code/verl-opd/scripts/opd/run_style.slurm
