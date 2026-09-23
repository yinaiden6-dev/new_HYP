#!/bin/bash
#SBATCH --job-name=rcde_tr_c5
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_validation="$rc_root/results/dino_rcde_track_r_p_v2_utility_shards_v1/validation/shard_000_012.validation.json"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_p_v2_utility_shards_v1.sbatch"
rc_next="$rc_root/slurm/dino_rcde_track_r_post_p_utility_controller_v1.sbatch"
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "$(jq -er '.status' "$rc_validation")" = DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARD_INDEPENDENT_VALIDATION_PASS
test "$(jq -er '.independent_full_replay' "$rc_validation")" = true
cd "$rc_root"
rc_remaining=$(sbatch --parsable --array=1-49%8 "$rc_launcher")
rc_controller=$(sbatch --parsable --dependency="afterok:$rc_remaining" "$rc_next")
echo "submitted V122 remaining array $rc_remaining and post-P controller $rc_controller"

