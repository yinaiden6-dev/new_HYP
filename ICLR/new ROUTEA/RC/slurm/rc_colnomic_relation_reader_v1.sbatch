#!/usr/bin/env bash
#SBATCH --job-name=cnrel
#SBATCH --partition=dev_cpuonly,cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/colnomic-relation-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/colnomic-relation-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
export PYTHONPATH="$PWD/programs:$PWD/src"
cnrel_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
cnrel_role="$1"
cnrel_index="${SLURM_ARRAY_TASK_ID:-0}"
cnrel_budget=400
if [[ "$cnrel_role" == capacity ]]; then cnrel_budget=240; fi
set +e
timeout --signal=TERM --kill-after=20s 510s "$cnrel_python" -u programs/run_rc_colnomic_relation_reader_v1.py "$cnrel_role" --index "$cnrel_index" --budget "$cnrel_budget"
cnrel_code=$?
set -e
if [[ "$cnrel_code" == 75 || "$cnrel_code" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 64 ]]; then exit 75; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if [[ "$cnrel_code" != 0 ]]; then exit "$cnrel_code"; fi
if [[ "$cnrel_role" == capacity ]]; then
  # BASE uses cached scalar content only. Run it as a separate resumable stage
  # in the same Job ID; its completion is checked before the three readers run.
  set +e
  timeout --signal=TERM --kill-after=20s 60s "$cnrel_python" -u programs/run_rc_colnomic_relation_reader_v1.py fit --index 0 --budget 25
  cnrel_base_code=$?
  set -e
  if [[ "$cnrel_base_code" == 75 || "$cnrel_base_code" == 124 ]]; then
    if [[ "${SLURM_RESTART_COUNT:-0}" -ge 64 ]]; then exit 75; fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if [[ "$cnrel_base_code" != 0 ]]; then exit "$cnrel_base_code"; fi
  cnrel_dispatch=after_capacity
elif [[ "$cnrel_role" == fit ]]; then
  cnrel_dispatch=after_fit
else
  exit 0
fi
set +e
timeout --signal=TERM --kill-after=5s 20s "$cnrel_python" -u programs/dispatch_rc_colnomic_relation_reader_v1.py "$cnrel_dispatch"
cnrel_dispatch_code=$?
set -e
if [[ "$cnrel_dispatch_code" != 0 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 64 ]]; then exit "$cnrel_dispatch_code"; fi
  scontrol requeue "$SLURM_JOB_ID"
fi
