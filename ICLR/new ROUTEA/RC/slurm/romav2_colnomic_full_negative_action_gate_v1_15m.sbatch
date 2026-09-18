#!/usr/bin/env bash
#SBATCH --job-name=roma_fullneg
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_fullneg-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_fullneg-%j.err
#SBATCH --export=NIL
set -euo pipefail
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC;py=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python;export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:/hkfs/work/workspace/scratch/ap7811-benchmark" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4;cd "$root";"$py" programs/run_romav2_colnomic_full_negative_action_gate_v1.py
