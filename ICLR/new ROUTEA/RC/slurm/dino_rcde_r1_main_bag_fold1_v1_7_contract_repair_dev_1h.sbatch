#!/bin/bash
# One immutable, contract-repaired continuation of R1 RCDE_BAG outer fold 1.
# Four copies may be submitted only as a strict afterok chain.
#SBATCH --job-name=rcde_r1b1v17
#SBATCH --partition=dev_accelerated-h100
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077

rc_workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_protocol="$rc_root/protocols/dino_rcde_r1_main_train_v1_7_contract_repair_20260813.json"
rc_authority="$rc_root/registry/current_authority_v13_20260813.json"
rc_runner="$rc_root/programs/run_dino_rcde_r1_main_train_v1_7_contract_repair.py"
rc_validator="$rc_root/programs/validate_dino_rcde_r1_main_train_v1_7_contract_repair.py"
rc_cache="$rc_root/runtime/dino_rcde_p0_v1_2/stable_fp16_cache"
rc_parent="$rc_root/results/dino_rcde_r1_main_v1_7_contract_repair"
rc_output="$rc_parent/RCDE_BAG_fold1"
rc_runtime_cache="${SLURM_TMPDIR:-/tmp}/rcde_r1b1v17_${SLURM_JOB_ID:?}"

test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "$#" -eq 0
test -x "$rc_python"
test -f "$rc_protocol"
test -f "$rc_authority"
test -f "$rc_runner"
test -f "$rc_validator"
test -d "$rc_cache"
mkdir -p "$rc_parent" "$rc_runtime_cache"

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4

if test -e "$rc_output/training_validation.json"; then
  exit 0
fi
if test ! -e "$rc_output/train_result.json"; then
  flock -x "$rc_parent/.RCDE_BAG_fold1.runner.lock" \
    "$rc_python" "$rc_runner" \
      --protocol "$rc_protocol" \
      --authority "$rc_authority" \
      --cache-root "$rc_cache" \
      --output-dir "$rc_output" \
      --arm RCDE_BAG \
      --fold 1 \
      --max-runtime-seconds 3240
fi
if test -e "$rc_output/train_result.json"; then
  "$rc_python" "$rc_validator" \
    --protocol "$rc_protocol" \
    --authority "$rc_authority" \
    --cache-root "$rc_cache" \
    --output-dir "$rc_output" \
    --validation-output "$rc_output/training_validation.json"
fi
