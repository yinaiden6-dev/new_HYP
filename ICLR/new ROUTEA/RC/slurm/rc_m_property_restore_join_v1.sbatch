#!/usr/bin/env bash
#SBATCH --job-name=M_restore_join
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-restore-join-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-restore-join-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
restore_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
set +e
timeout --signal=TERM --kill-after=10s 540s "$restore_python" -u 'ICLR/new ROUTEA/RC/programs/join_rc_roma_property_restore_v1.py' --budget 480
restore_status=$?
set -e
if [[ "$restore_status" == 75 || "$restore_status" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 2 ]]; then exit 124; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$restore_status"
