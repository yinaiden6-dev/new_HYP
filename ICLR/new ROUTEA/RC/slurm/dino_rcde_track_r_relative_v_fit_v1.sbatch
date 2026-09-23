#!/bin/bash
#SBATCH --job-name=rcde_tr_vfit
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=03:00:00
#SBATCH --array=1-4%4
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
rc_runner="$rc_root/programs/run_dino_rcde_track_r_relative_v_fit_v1.py"
rc_validator="$rc_root/programs/validate_dino_rcde_track_r_relative_v_fit_v1.py"
rc_authority="$rc_root/registry/current_authority_v120_20260821.json"
rc_protocol="$rc_root/protocols/dino_rcde_track_r_relative_v_fit_protocol_v1_20260821.json"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_relative_v_fit_v1.sbatch"
rc_fold=${SLURM_ARRAY_TASK_ID:?}
rc_parent="$rc_root/results/dino_rcde_track_r_relative_v_fits_v1/outer_fold${rc_fold}"
rc_primary="$rc_parent/primary"
rc_replay="$rc_parent/resume_replay"
rc_validation="$rc_parent/fresh_resume_validation.json"
rc_node_root="$rc_root/protocols/dino_rcde_track_r_relative_v_fit_nodes_v2"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = accelerated
test "${SLURM_CPUS_PER_TASK:?}" -eq 4
test "${SLURM_MEM_PER_NODE:?}" -eq 64000
test -n "${CUDA_VISIBLE_DEVICES:-}"
case "$rc_fold" in 1|2|3|4) ;; *) exit 64 ;; esac
test -f "$rc_authority"
test ! -e "$rc_primary"
test ! -e "$rc_replay"
test ! -e "$rc_validation"
cmp -s "$0" "$rc_launcher"

rc_runtime=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/rcde_trackr_vfit.XXXXXX")
mkdir -p "$rc_runtime/tmp" "$rc_runtime/cuda" "$rc_runtime/triton" "$rc_runtime/xdg"
export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$rc_runtime/tmp" CUDA_CACHE_PATH="$rc_runtime/cuda"
export TRITON_CACHE_DIR="$rc_runtime/triton" XDG_CACHE_HOME="$rc_runtime/xdg"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

cd "$rc_root"
"$rc_python" "$rc_runner" --execution-manifest "$rc_node_root/fold${rc_fold}_primary.json"
"$rc_python" "$rc_runner" --execution-manifest "$rc_node_root/fold${rc_fold}_resume_part1.json"
"$rc_python" "$rc_runner" --execution-manifest "$rc_node_root/fold${rc_fold}_resume_part2.json"
"$rc_python" "$rc_validator" \
  --authority "$rc_authority" \
  --protocol "$rc_protocol" \
  --primary-output "$rc_primary" \
  --resume-output "$rc_replay" \
  --outer-fold "$rc_fold" \
  --output "$rc_validation"
test "$(jq -er '.status' "$rc_validation")" = DINO_RCDE_TRACK_R_RELATIVE_V_FIT_FRESH_RESUME_VALIDATION_PASS
test "$(jq -er '.validation_pass' "$rc_validation")" = true
