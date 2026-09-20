#!/usr/bin/env bash
#SBATCH --job-name=h593_colpack
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-colpack-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-colpack-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
exec timeout --signal=TERM --kill-after=10s 540s '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' programs/publish_rc_h593_content_controls_v1.py
