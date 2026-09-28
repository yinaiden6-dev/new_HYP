#!/usr/bin/env bash
#SBATCH --job-name=M2_context_pack
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=52
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M2-packed-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M2-packed-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
set +e
timeout --signal=TERM --kill-after=10s 540s .venv-colpali/bin/python -u 'ICLR/new ROUTEA/RC/programs/run_rc_m_predictive_fine_packed_v1.py'
packed_status=$?
set -e
if [[ "$packed_status" == 75 || "$packed_status" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 32 ]]; then exit 124; fi
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit "$packed_status"
