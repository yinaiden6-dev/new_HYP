#!/bin/bash
#SBATCH --job-name=rcde_tr_pu
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=65536
#SBATCH --time=06:00:00
#SBATCH --array=0
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
rc_authority="$rc_root/registry/current_authority_v122_20260821.json"
rc_authority_validation="$rc_root/results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/result.json"
rc_producer="$rc_root/programs/materialize_dino_rcde_track_r_p_v2_utility_shard_v1.py"
rc_validator="$rc_root/programs/validate_dino_rcde_track_r_p_v2_utility_shard_v1.py"
rc_producer_root="$rc_root/results/dino_rcde_track_r_p_v2_utility_shards_v1/producer"
rc_validation_root="$rc_root/results/dino_rcde_track_r_p_v2_utility_shards_v1/validation"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_p_v2_utility_shards_v1.sbatch"
rc_shard=${SLURM_ARRAY_TASK_ID:?}
rc_start=$((rc_shard*12))
rc_stop=$((rc_start+12))
rc_payload=$(printf '%s/shard_%03d_%03d.pt' "$rc_producer_root" "$rc_start" "$rc_stop")
rc_validation=$(printf '%s/shard_%03d_%03d.validation.json' "$rc_validation_root" "$rc_start" "$rc_stop")

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 4
test "${SLURM_MEM_PER_NODE:?}" -eq 65536
test "$rc_shard" -ge 0
test "$rc_shard" -lt 50
test "$(stat -c '%a' "$rc_authority")" = 444
test "$(stat -c '%a' "$rc_authority_validation")" = 444
test "$(jq -er '.status' "$rc_authority_validation")" = DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS
test "$(jq -er '.authority_sha256' "$rc_authority_validation")" = "$(sha256sum "$rc_authority" | cut -d' ' -f1)"
test ! -e "$rc_payload"
test ! -e "$rc_validation"
cmp -s "$0" "$rc_launcher"

export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$rc_root"
"$rc_python" "$rc_producer" --authority "$rc_authority" --shard-ordinal "$rc_shard" --output-root "$rc_producer_root"
"$rc_python" "$rc_validator" --authority "$rc_authority" --shard "$rc_payload" --shard-ordinal "$rc_shard" --output "$rc_validation"
test "$(jq -er '.status' "$rc_validation")" = DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARD_INDEPENDENT_VALIDATION_PASS
test "$(jq -er '.independent_full_replay' "$rc_validation")" = true
