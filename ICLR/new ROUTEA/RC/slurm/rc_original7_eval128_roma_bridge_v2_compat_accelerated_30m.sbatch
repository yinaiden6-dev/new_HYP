#!/usr/bin/env bash
#SBATCH --job-name=e128_roma2_bridge
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/e128_roma2_bridge-%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/e128_roma2_bridge-%A_%a.err
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
root="$workspace/ICLR/new ROUTEA/RC"
py="$workspace/.venv-romav2/bin/python"
stage_mode=${1:-produce}
runtime="${SLURM_TMPDIR:-/tmp}/rc-e128-roma-${SLURM_JOB_ID:-preflight}-${SLURM_ARRAY_TASK_ID:-0}-$$"
mkdir -p "$runtime/tmp" "$runtime/xdg" "$runtime/torchinductor" "$runtime/triton"
export PYTHONPATH="$root/src:$root/programs:$workspace"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TORCH_HOME="$workspace/third_party/model_cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$runtime/xdg" TMPDIR="$runtime/tmp"
export TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" TRITON_CACHE_DIR="$runtime/triton"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$root"
rc_remaining_seconds=$("$py" -c 'from datetime import datetime,timezone;print(max(0,int((datetime(2026,9,11,16,tzinfo=timezone.utc)-datetime.now(timezone.utc)).total_seconds())))')
if (( rc_remaining_seconds <= 0 )); then
  echo 'USER_RESEARCH_DEADLINE_REACHED'
  exit 1
fi
if [[ "$stage_mode" == preflight ]]; then
  exec timeout --signal=TERM --kill-after=10s "${rc_remaining_seconds}s" "$py" programs/materialize_rc_original7_eval128_roma_v2_runtime_compat.py --phase preflight
fi
[[ "$stage_mode" == produce ]]
[[ -n "${SLURM_JOB_ID:-}" ]]
exec timeout --signal=TERM --kill-after=10s "${rc_remaining_seconds}s" "$py" programs/materialize_rc_original7_eval128_roma_v2_runtime_compat.py --phase produce --kind bridge
