#!/bin/bash
# One submission, three isolated single-fold BAG training tasks (folds 2--4).
# Each task owns a distinct protocol, output namespace, lock and node-local cache.
#SBATCH --job-name=rcde_r1b_v18
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=04:00:00
#SBATCH --array=2-4%3
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NIL
set -euo pipefail
umask 077

rc_workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_authority="$rc_root/registry/current_authority_v14_20260814.json"
rc_runner="$rc_root/programs/run_dino_rcde_r1_main_train_v1_8_single_scope.py"
rc_validator="$rc_root/programs/validate_dino_rcde_r1_main_train_v1_8_single_scope.py"
rc_cache="$rc_root/runtime/dino_rcde_p0_v1_2/stable_fp16_cache"
rc_parent="$rc_root/results/dino_rcde_r1_main_v1_8_accelerated_folds2_4"

test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "$#" -eq 0
test -x "$rc_python"
rc_fold=${SLURM_ARRAY_TASK_ID:?}
case "$rc_fold" in
  2|3|4) ;;
  *) echo "unauthorized array fold: $rc_fold" >&2; exit 64 ;;
esac
rc_protocol="$rc_root/protocols/dino_rcde_r1_main_train_v1_8_bag_fold${rc_fold}_20260814.json"
rc_output="$rc_parent/RCDE_BAG_fold${rc_fold}"
rc_lock="$rc_parent/.RCDE_BAG_fold${rc_fold}.runner.lock"
rc_runtime_cache="${SLURM_TMPDIR:-/tmp}/rcde_r1b_v18_${SLURM_ARRAY_JOB_ID:?}_${rc_fold}"

test -f "$rc_protocol"
test -f "$rc_authority"
test -f "$rc_runner"
test -f "$rc_validator"
test -d "$rc_cache"
test ! -e "$rc_output"
mkdir -p "$rc_parent" "$rc_runtime_cache/tmp" "$rc_runtime_cache/cuda" \
  "$rc_runtime_cache/numba"

export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$rc_runtime_cache/hf"
export XDG_CACHE_HOME="$rc_runtime_cache/xdg"
export TORCH_HOME="$rc_runtime_cache/torch"
export TRITON_CACHE_DIR="$rc_runtime_cache/triton"
export TMPDIR="$rc_runtime_cache/tmp"
export CUDA_CACHE_PATH="$rc_runtime_cache/cuda"
export NUMBA_CACHE_DIR="$rc_runtime_cache/numba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4

flock -x "$rc_lock" \
  "$rc_python" "$rc_runner" \
    --protocol "$rc_protocol" \
    --authority "$rc_authority" \
    --cache-root "$rc_cache" \
    --output-dir "$rc_output" \
    --arm RCDE_BAG \
    --fold "$rc_fold" \
    --max-runtime-seconds 13800
if test ! -e "$rc_output/train_result.json"; then
  echo "fold $rc_fold did not complete within its one authorized task" >&2
  exit 91
fi
"$rc_python" "$rc_validator" \
  --protocol "$rc_protocol" \
  --authority "$rc_authority" \
  --cache-root "$rc_cache" \
  --output-dir "$rc_output" \
  --arm RCDE_BAG \
  --fold "$rc_fold" \
  --validation-output "$rc_output/training_validation.json"
