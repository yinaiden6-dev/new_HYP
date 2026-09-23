#!/bin/bash
#SBATCH --job-name=rcde_tr_c1v2
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=65536
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail

rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_reduce_freezer="$rc_root/programs/freeze_dino_rcde_track_r_relative_e1_episode_reduction_authority_v118.py"
rc_reducer="$rc_root/programs/reduce_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py"
rc_validator="$rc_root/programs/validate_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py"
rc_producer_root="$rc_root/results/dino_rcde_track_r_relative_e1_shards_v1/producer"
rc_shard_validation_root="$rc_root/results/dino_rcde_track_r_relative_e1_shards_v1/validation"
rc_ledger_root="$rc_root/results/dino_rcde_track_r_relative_e1_ledgers_v1"
rc_validation="$rc_root/results/dino_rcde_track_r_relative_e1_ledgers_validation_v1/result.json"
rc_vfit_freezer="$rc_root/programs/freeze_dino_rcde_track_r_relative_v_fit_authority_v119.py"
rc_vfit_launcher="$rc_root/slurm/dino_rcde_track_r_relative_v_fit_v1.sbatch"
rc_next="$rc_root/slurm/dino_rcde_track_r_post_vfit_controller_v1.sbatch"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
cd "$rc_root"
"$rc_python" "$rc_reduce_freezer"
"$rc_python" "$rc_reducer" --authority "$rc_root/registry/current_authority_v119_20260821.json" --producer-root "$rc_producer_root" --validation-root "$rc_shard_validation_root" --output-root "$rc_ledger_root"
"$rc_python" "$rc_validator" --authority "$rc_root/registry/current_authority_v119_20260821.json" --producer-root "$rc_producer_root" --shard-validation-root "$rc_shard_validation_root" --ledger-root "$rc_ledger_root" --output "$rc_validation"
test "$(jq -er '.status' "$rc_validation")" = DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGERS_INDEPENDENT_VALIDATION_PASS
"$rc_python" "$rc_vfit_freezer"
rc_vfit=$(sbatch --parsable "$rc_vfit_launcher")
rc_controller=$(sbatch --parsable --dependency="afterok:$rc_vfit" "$rc_next")
echo "submitted V120 vfit array $rc_vfit and controller $rc_controller"

