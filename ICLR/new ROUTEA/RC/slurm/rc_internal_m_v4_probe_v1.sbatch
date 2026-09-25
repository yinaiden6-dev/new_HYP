#!/usr/bin/env bash
#SBATCH --job-name=cn_M4probe
#SBATCH --partition=dev_accelerated,accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/internal-M4-probe-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/internal-M4-probe-%j.err
#SBATCH --export=NIL
#SBATCH --requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
export PYTHONPATH="$PWD/programs:$PWD/src:/hkfs/work/workspace/scratch/ap7811-benchmark"
PY='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
set +e
timeout --signal=TERM --kill-after=20s 550s "$PY" -u programs/run_rc_internal_m_v4_probe_v1.py "$1" --lane "${2:-REAL}" --budget 380
code=$?
set -e
if [[ "$code" == 75 || "$code" == 124 ]]; then
  if [[ "${SLURM_RESTART_COUNT:-0}" -ge 8 ]]; then exit 75; fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$code"
