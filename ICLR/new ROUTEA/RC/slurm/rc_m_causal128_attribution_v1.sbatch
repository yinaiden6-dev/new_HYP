#!/usr/bin/env bash
#SBATCH --job-name=M_causal128
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-causal128-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-causal128-%A_%a.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
causal_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
causal_stage="$1"
causal_restarts="$2"
set +e
timeout --signal=TERM --kill-after=10s 540s "$causal_python" -u 'ICLR/new ROUTEA/RC/programs/submit_rc_m_causal128_attribution_chain_v1.py' run "$causal_stage"
causal_status=$?
set -e
if [[ "$causal_status" == 75 || "$causal_status" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge "$causal_restarts" ]]; then exit 124; fi
  causal_job="$SLURM_JOB_ID"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then causal_job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
  scontrol requeue "$causal_job"
  exit 0
fi
exit "$causal_status"
