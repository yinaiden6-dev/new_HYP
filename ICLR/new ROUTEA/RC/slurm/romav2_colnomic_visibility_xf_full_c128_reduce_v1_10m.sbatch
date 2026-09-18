#!/usr/bin/env bash
#SBATCH --job-name=roma_c128r
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_c128r-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/roma_c128r-%j.err
#SBATCH --export=NIL
set -euo pipefail
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC;py=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python;export PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH="$root/src:$root/programs:/hkfs/work/workspace/scratch/ap7811-benchmark" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='';cd "$root";"$py" programs/reduce_romav2_colnomic_visibility_xf_full_c128_v1.py
