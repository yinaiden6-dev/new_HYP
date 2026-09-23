#!/usr/bin/env bash
#SBATCH --job-name=h593_packfix
#SBATCH --partition=dev_accelerated
#SBATCH --gres=gpu:1
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-packfix-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-packfix-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
export PATH="$rc_root/isolated/h593_packaging_runtime_20260920_v1/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$rc_root/isolated/h593_packaging_runtime_20260920_v1/site-packages"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
cd "$rc_root"
exec timeout --signal=TERM --kill-after=10s 540s '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' programs/run_rc_h593_packaging_repair_v1.py "${1:?stage required}"
