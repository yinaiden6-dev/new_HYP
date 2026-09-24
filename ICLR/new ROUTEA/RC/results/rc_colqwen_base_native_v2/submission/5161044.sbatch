#!/usr/bin/env bash
#SBATCH --job-name=cqbase2_native
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cqbase2-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/cqbase2-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
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
export TORCH_HOME='/hkfs/work/workspace/scratch/ap7811-benchmark/third_party/model_cache/torch'
set +e
timeout --signal=TERM --kill-after=20s 835s "$PY" -u programs/run_rc_colqwen_base_native_v2.py "$1" --budget 660
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
 if [[ "${SLURM_RESTART_COUNT:-0}" -ge 16 ]]; then exit 75; fi
 job="${SLURM_JOB_ID}"
 if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
 scontrol requeue "$job"
 exit 0
fi
if [[ "$code" != 0 ]]; then exit "$code"; fi
"$PY" -u programs/run_rc_colqwen_base_native_v2.py "advance-$1"
