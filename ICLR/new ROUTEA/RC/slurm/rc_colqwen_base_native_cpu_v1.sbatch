#!/usr/bin/env bash
#SBATCH --job-name=cqbase_join
#SBATCH --partition=cpuonly,dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cqbase-cpu-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cqbase-cpu-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
export PYTHONPATH="$PWD/programs:$PWD/src:/hkfs/work/workspace/scratch/ap7811-benchmark"
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
if [[ "$1" == quality ]]; then PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python'; fi
export CUDA_VISIBLE_DEVICES=
timeout --signal=TERM --kill-after=10s 550s "$PY" -u programs/run_rc_colqwen_base_native_v1.py "$1"
"$PY" -u programs/run_rc_colqwen_base_native_v1.py "advance-$1"
