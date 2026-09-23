#!/usr/bin/env bash
#SBATCH --job-name=h593_operator
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-operator-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-operator-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
rc_root='/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'
rc_tmp="${SLURM_TMPDIR:-/tmp}/h593-operator-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$rc_tmp"
export TMPDIR="$rc_tmp"
cd "$rc_root"
exec timeout --signal=TERM --kill-after=10s 850s "$rc_python" programs/run_rc_h593_quality_operator_v1.py worker --shard "${SLURM_ARRAY_TASK_ID:-0}"
