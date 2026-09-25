#!/usr/bin/env bash
#SBATCH --job-name=post593
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post593-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post593-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
export PYTHONPATH="$PWD/programs:$PWD/src:/hkfs/work/workspace/scratch/ap7811-benchmark"
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
role="$1"
index="${SLURM_ARRAY_TASK_ID:-0}"
set +e
timeout --signal=TERM --kill-after=20s 550s "$PY" -u programs/dispatch_rc_postllm_h593_v1.py worker --role "$role" --index "$index" --budget 400
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 48 ]]; then exit 75; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$code"
