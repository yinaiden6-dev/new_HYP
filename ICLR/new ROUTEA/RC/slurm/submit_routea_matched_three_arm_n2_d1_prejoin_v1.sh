#!/bin/bash
# Submit only the label authority and target-free RAW-prejoin graph.
# Usage: bash slurm/submit_routea_matched_three_arm_n2_d1_prejoin_v1.sh 5129123
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 TOKEN_AGGREGATE_JOB_ID" >&2
  exit 2
fi
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
cd "$root"
label=$(sbatch --parsable --dependency="afterok:$1" slurm/routea_matched_three_arm_n2_d1_label_join_authority_v1_15m.sbatch)
raw=$(sbatch --parsable --dependency="afterok:$1" slurm/routea_matched_three_arm_n2_d1_raw_prejoin_array_v1_1h.sbatch)
aggregate=$(sbatch --parsable --dependency="afterok:$raw" slurm/routea_matched_three_arm_n2_d1_raw_prejoin_aggregate_v1_15m.sbatch)
printf 'label_job=%s\nraw_array_job=%s\nraw_aggregate_job=%s\n' "$label" "$raw" "$aggregate"
