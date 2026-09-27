#!/usr/bin/env bash
#SBATCH --job-name=M_phase_bridge
#SBATCH --partition=cpuonly,dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-phase-bridge-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-phase-bridge-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
bridge_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
set +e
timeout --signal=TERM --kill-after=10s 540s "$bridge_python" -u 'ICLR/new ROUTEA/RC/programs/run_rc_m_phase_bridge_v1.py' pool --budget 450 --workers 4 --threads 2
bridge_status=$?
set -e
if [[ "$bridge_status" == 75 || "$bridge_status" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 8 ]]; then exit 124; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$bridge_status"
