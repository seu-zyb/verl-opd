#!/bin/bash
#SBATCH --job-name=mopd-paper
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --chdir=/seu_nvme/home/fenglei/213241973/code/verl-opd
#SBATCH -o mopd-paper-%j.out
#SBATCH -e mopd-paper-%j.err

export OPD_CONFIG_NAME=mopd_paper_v1
export OPD_RUN_LABEL=paper-v1-mopd
# Source the shared launcher so submitted.slurm captures this Slurm submission.
source /seu_nvme/home/fenglei/213241973/code/verl-opd/scripts/opd/run_paper_baseline.slurm "$@"
