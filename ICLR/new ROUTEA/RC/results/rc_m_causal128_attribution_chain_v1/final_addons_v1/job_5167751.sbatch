#!/usr/bin/env bash
#SBATCH --job-name=M_causal_add
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-causal128-addon-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/M-causal128-addon-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
case "$1" in
  compensate) causal_args=('ICLR/new ROUTEA/RC/programs/run_rc_m_causal128_property_compensation_v1.py' run) ;;
  bridge) causal_args=('ICLR/new ROUTEA/RC/programs/run_rc_m_causal128_newgroup_property_bridge_v1.py' worker --budget 430) ;;
  join) causal_args=('ICLR/new ROUTEA/RC/programs/run_rc_m_causal128_newgroup_property_bridge_v1.py' join) ;;
  *) exit 64 ;;
esac
set +e
timeout --signal=TERM --kill-after=10s 540s .venv-colpali/bin/python -u "${causal_args[@]}"
causal_status=$?
set -e
if [[ "$causal_status" == 75 || "$causal_status" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 6 ]]; then exit 124; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$causal_status"
