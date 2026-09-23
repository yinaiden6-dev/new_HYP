#!/usr/bin/env bash
#SBATCH --job-name=h71_fusion
#SBATCH --partition=cpuonly,dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h71-fusion-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h71-fusion-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_workspace='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-romav2/bin/python"
rc_runtime="${SLURM_TMPDIR:-/tmp}/h71-fusion-${SLURM_JOB_ID:?}"
mkdir -p "$rc_runtime/tmp" "$rc_runtime/xdg"
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$rc_runtime/xdg" TMPDIR="$rc_runtime/tmp"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=''
cd "$rc_root"
exec timeout --signal=TERM --kill-after=10s 575s "$rc_python" -u programs/run_rc_h71_feature_fusion_pilot_v1.py "${1:-fit}" --index "${SLURM_ARRAY_TASK_ID:-0}"
