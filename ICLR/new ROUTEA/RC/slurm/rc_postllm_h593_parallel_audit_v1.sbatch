#!/usr/bin/env bash
#SBATCH --job-name=post_attr_audit
#SBATCH --partition=dev_cpuonly,cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-attr-audit-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/post-attr-audit-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
set +e
timeout --signal=TERM --kill-after=15s 550s "$PY" -u programs/parallel_audit_rc_postllm_h593_decomposition_v1.py --workers 8 --budget 440
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 80 ]]; then exit 75; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$code"
