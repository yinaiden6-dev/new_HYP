#!/bin/bash
#SBATCH --job-name=rcde_tr_c3
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32768
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail

rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_self="$rc_root/slurm/dino_rcde_track_r_post_oof_controller_v1.sbatch"
rc_authority="$rc_root/registry/current_authority_v121_20260821.json"
rc_runtime_validator="$rc_root/programs/validate_dino_rcde_track_r_v121_runtime_binding_v1.py"
rc_finalizer="$rc_root/programs/finalize_dino_rcde_track_r_oof_prejoin_v120.py"
rc_producer_root="$rc_root/results/dino_rcde_track_r_oof_prejoin_v1/producer"
rc_validation_root="$rc_root/results/dino_rcde_track_r_oof_prejoin_v1/validation"
rc_output="$rc_root/results/dino_rcde_track_r_oof_prejoin_completion_v120/result.json"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test ! -e "$rc_output"
test -f "$rc_runtime_validator"
cmp -s "$0" "$rc_self"
cd "$rc_root"
"$rc_python" "$rc_runtime_validator" \
  --authority "$rc_authority" \
  --binding-name post_oof_controller \
  --runtime-file "$rc_self"
"$rc_python" "$rc_finalizer" \
  --authority "$rc_authority" \
  --producer-root "$rc_producer_root" \
  --validation-root "$rc_validation_root" \
  --output "$rc_output"
test "$(jq -er '.status' "$rc_output")" = DINO_RCDE_TRACK_R_OOF_PREJOIN_V120_COMPLETE
