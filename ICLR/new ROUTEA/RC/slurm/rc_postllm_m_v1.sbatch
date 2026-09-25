#!/usr/bin/env bash
#SBATCH --job-name=cn_Mpost1
#SBATCH --partition=dev_accelerated,accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/postllm-M1-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/postllm-M1-%j.err
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
stage="$1"
arm="${2:-POST_REAL}"
set +e
timeout --signal=TERM --kill-after=20s 550s "$PY" -u programs/run_rc_postllm_m_v1.py "$stage" --arm "$arm" --budget 400
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 48 ]]; then exit 75; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if [[ "$code" != 0 ]]; then exit "$code"; fi
if [[ "$stage" == pilot ]]; then
  "$PY" -u programs/run_rc_postllm_m_v1.py advance
fi
