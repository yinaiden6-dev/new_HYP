#!/usr/bin/env bash
#SBATCH --job-name=post_attr
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-attr-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-attr-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
export PYTHONPATH="$PWD/programs:$PWD/src:/hkfs/work/workspace/scratch/ap7811-benchmark"
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
set +e
timeout --signal=TERM --kill-after=15s 550s "$PY" -u programs/dispatch_rc_postllm_h593_decomposition_v1.py "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 80 ]]; then exit 75; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$code"
