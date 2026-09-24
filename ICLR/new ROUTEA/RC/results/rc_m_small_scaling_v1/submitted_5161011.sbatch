#!/usr/bin/env bash
#SBATCH --job-name=M_small_scale
#SBATCH --partition=cpuonly,dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/m-small-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/m-small-%j.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
timeout --signal=TERM --kill-after=10s 550s "$PY" -u programs/run_rc_m_small_scaling_v1.py "$1" "${2:-0}"
"$PY" -u programs/run_rc_m_small_scaling_v1.py "advance-$1" "${2:-0}"
