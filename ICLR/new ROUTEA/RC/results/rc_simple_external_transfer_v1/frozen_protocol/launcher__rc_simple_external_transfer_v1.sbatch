#!/usr/bin/env bash
#SBATCH --job-name=simple_external
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/simple-external-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/simple-external-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
rc_program="${1:?program required}"
shift
case "$rc_program" in
 source) rc_program=programs/run_rc_simple_external_source_freeze_v1.py ;;
 replay) rc_program=programs/run_rc_simple_external_replay_v1.py ;;
 *) exit 2 ;;
esac
rc_args=("$@")
if [[ "${1:-}" == array ]]; then
  rc_args=(replay --task "${SLURM_ARRAY_TASK_ID:?}")
fi
set +e
timeout --signal=TERM --kill-after=10s 550s "$rc_python" -u "$rc_program" "${rc_args[@]}"
rc_status=$?
set -e
if [[ "$rc_status" == 75 || "$rc_status" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 12 ]]; then exit 124; fi
  rc_job="${SLURM_JOB_ID:?}"
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then rc_job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID:?}"; fi
  scontrol requeue "$rc_job"
  exit 0
fi
exit "$rc_status"
