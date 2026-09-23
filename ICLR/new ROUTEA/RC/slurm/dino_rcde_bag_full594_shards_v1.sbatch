#!/bin/bash
#SBATCH --job-name=rcde_bag594
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH --time=01:00:00
#SBATCH --array=0-49%8
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --export=NIL
set -euo pipefail
umask 077

rc_workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_authority="$rc_root/registry/dino_rcde_bag_full594_completion_authority_v1_20260827.json"
rc_producer="$rc_root/programs/materialize_dino_rcde_bag_full594_shard_v1.py"
rc_validator="$rc_root/programs/validate_dino_rcde_bag_full594_shard_v1.py"
rc_launcher="$rc_root/slurm/dino_rcde_bag_full594_shards_v1.sbatch"
rc_producer_root="$rc_root/results/dino_rcde_bag_full594_completion_v1/prejoin"
rc_validation_root="$rc_root/results/dino_rcde_bag_full594_completion_v1/prejoin_validation"
rc_shard=${SLURM_ARRAY_TASK_ID:?}
rc_start=$((rc_shard*12))
rc_stop=$((rc_start+12))
rc_payload=$(printf '%s/shard_%03d_%03d.pt' "$rc_producer_root" "$rc_start" "$rc_stop")
rc_validation=$(printf '%s/shard_%03d_%03d.validation.json' "$rc_validation_root" "$rc_start" "$rc_stop")

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
case "${SLURM_JOB_PARTITION:?}" in
  accelerated|dev_accelerated-h100) ;;
  *) exit 71 ;;
esac
test "${SLURM_CPUS_PER_TASK:?}" -eq 8
test "${SLURM_MEM_PER_NODE:?}" -eq 128000
test -n "${CUDA_VISIBLE_DEVICES:-}"
test "$rc_shard" -ge 0
test "$rc_shard" -lt 50
test -f "$rc_authority"
test ! -e "$rc_payload"
test ! -e "$rc_validation"
cmp -s "$0" "$rc_launcher"

rc_runtime=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/rcde_bag594.XXXXXX")
mkdir -p "$rc_runtime/tmp" "$rc_runtime/cuda" "$rc_runtime/triton" "$rc_runtime/xdg"
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$rc_runtime/tmp" CUDA_CACHE_PATH="$rc_runtime/cuda"
export TRITON_CACHE_DIR="$rc_runtime/triton" XDG_CACHE_HOME="$rc_runtime/xdg"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

cd "$rc_root"
"$rc_python" "$rc_producer" \
  --authority "$rc_authority" \
  --shard-ordinal "$rc_shard" \
  --output-root "$rc_producer_root" \
  --device cuda
"$rc_python" "$rc_validator" \
  --authority "$rc_authority" \
  --shard "$rc_payload" \
  --shard-ordinal "$rc_shard" \
  --output "$rc_validation" \
  --device cuda
test -f "$rc_payload"
test -f "$rc_validation"
