#!/bin/bash
#SBATCH --job-name=rcde_bag594_fin
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077

rc_workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_authority="$rc_root/registry/dino_rcde_bag_full594_completion_authority_v1_20260827.json"
rc_launcher="$rc_root/slurm/dino_rcde_bag_full594_finalize_v1.sbatch"
rc_result_root="$rc_root/results/dino_rcde_bag_full594_completion_v1"
rc_producer_root="$rc_result_root/prejoin"
rc_validation_root="$rc_result_root/prejoin_validation"
rc_controls="$rc_result_root/target_free_controls.pt"
rc_manifest="$rc_result_root/prejoin_manifest.json"
rc_join="$rc_result_root/label_join.json"
rc_result="$rc_result_root/result.json"
rc_science_validation="$rc_result_root/independent_validation.json"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 8
test "${SLURM_MEM_PER_NODE:?}" -eq 128000
test -z "${CUDA_VISIBLE_DEVICES:-}"
test -f "$rc_authority"
test ! -e "$rc_controls"
test ! -e "$rc_manifest"
test ! -e "$rc_join"
test ! -e "$rc_result"
test ! -e "$rc_science_validation"
cmp -s "$0" "$rc_launcher"

rc_runtime=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/rcde_bag594_fin.XXXXXX")
mkdir -p "$rc_runtime/tmp" "$rc_runtime/xdg"
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$rc_runtime/tmp" XDG_CACHE_HOME="$rc_runtime/xdg"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

cd "$rc_root"
"$rc_python" programs/finalize_dino_rcde_bag_full594_prejoin_v1.py \
  --authority "$rc_authority" \
  --producer-root "$rc_producer_root" \
  --validation-root "$rc_validation_root" \
  --controls "$rc_controls" \
  --manifest "$rc_manifest"
"$rc_python" programs/join_dino_rcde_bag_full594_labels_v1.py \
  --authority "$rc_authority" \
  --producer-root "$rc_producer_root" \
  --controls "$rc_controls" \
  --manifest "$rc_manifest" \
  --output "$rc_join"
"$rc_python" programs/reduce_dino_rcde_bag_full594_science_v1.py \
  --authority "$rc_authority" \
  --join "$rc_join" \
  --output "$rc_result"
"$rc_python" programs/validate_dino_rcde_bag_full594_science_v1.py \
  --authority "$rc_authority" \
  --producer-root "$rc_producer_root" \
  --controls "$rc_controls" \
  --manifest "$rc_manifest" \
  --join "$rc_join" \
  --result "$rc_result" \
  --output "$rc_science_validation"
test -f "$rc_science_validation"
