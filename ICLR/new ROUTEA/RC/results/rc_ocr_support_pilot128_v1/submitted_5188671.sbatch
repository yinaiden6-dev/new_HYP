#!/usr/bin/env bash
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --requeue
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/ocr-support-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/ocr-support-%A_%a.err
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
ws='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc="$ws/ICLR/new ROUTEA/RC"
export PYTHONPATH="$rc/programs:$rc/src:$ws"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PADDLE_PDX_HOME="$ws/cache/paddlex"
stage="$1"
shard="${SLURM_ARRAY_TASK_ID:-0}"
py="$ws/.venv-colpali/bin/python"
if [[ "$stage" == ocr || "$stage" == smoke ]]; then py=/home/hk-project-pai00054/ap7811/miniconda/envs/dotsocr/bin/python; fi
cd "$rc"
set +e
timeout --signal=TERM --kill-after=15s 550s "$py" -u programs/run_rc_ocr_support_pilot128_v1.py "$stage" --shard "$shard" --fold "$shard" --budget 430
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 64 ]]; then exit 75; fi
  task="$SLURM_JOB_ID"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then task="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
  scontrol requeue "$task"
  exit 0
fi
exit "$code"
