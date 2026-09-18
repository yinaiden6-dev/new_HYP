#!/usr/bin/env bash
set -euo pipefail

root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
cd "$root"

array_job=$(sbatch --parsable slurm/routea_matched_three_arm_n2_token_cache_array_v1_1h.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${array_job}" slurm/routea_matched_three_arm_n2_token_cache_aggregate_v1_30m.sbatch)

printf 'array_job=%s\naggregate_job=%s\n' "$array_job" "$aggregate_job"
