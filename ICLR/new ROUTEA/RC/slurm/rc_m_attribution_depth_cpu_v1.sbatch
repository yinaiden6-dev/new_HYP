#!/usr/bin/env bash
#SBATCH --job-name=M_depth
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-depth-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-depth-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
set +e
timeout --signal=TERM --kill-after=10s 540s .venv-colpali/bin/python -u 'ICLR/new ROUTEA/RC/programs/submit_rc_m_attribution_depth_v1.py' run "$1"
depth_status=$?
set -e
if [[ "$depth_status" == 75 || "$depth_status" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 6 ]]; then exit 124; fi
  depth_job="$SLURM_JOB_ID"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then depth_job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
  scontrol requeue "$depth_job"
  exit 0
fi
exit "$depth_status"
