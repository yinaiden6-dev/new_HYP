#!/usr/bin/env bash
#SBATCH --job-name=h593_ccache
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-ccache-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/h593-ccache-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
rc_workspace='/hkfs/work/workspace/scratch/ap7811-benchmark'
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-romav2/bin/python"
rc_runtime="${SLURM_TMPDIR:-/tmp}/h593-coordinate-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$rc_runtime/tmp" "$rc_runtime/xdg" "$rc_runtime/inductor" "$rc_runtime/triton"
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TORCH_HOME="$rc_workspace/third_party/model_cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$rc_runtime/xdg" TMPDIR="$rc_runtime/tmp"
export TORCHINDUCTOR_CACHE_DIR="$rc_runtime/inductor" TRITON_CACHE_DIR="$rc_runtime/triton"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$rc_root"
exec timeout --signal=TERM --kill-after=10s 850s "$rc_python" programs/run_rc_h593_roma_coordinate_cache_v2.py "${1:-worker}" --index "${SLURM_ARRAY_TASK_ID:-0}"
