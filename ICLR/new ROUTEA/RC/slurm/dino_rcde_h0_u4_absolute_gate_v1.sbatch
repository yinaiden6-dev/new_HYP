#!/bin/bash
#SBATCH --job-name=rcde_h0u4gate
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=dev_cpuonly,cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC;python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python;authority=registry/h0/h0_u4_absolute_gate_authority_v24_20260823.json;raw=results/dino_rcde_h0_u4_raw_aggregate_v2/result.json;result=results/dino_rcde_h0_u4_absolute_gate_v1/result.json;validation=results/dino_rcde_h0_u4_absolute_gate_validation_v1/result.json
test "$#" -eq 0;test "${SLURM_SUBMIT_DIR:-}" = "$root";case "${SLURM_JOB_PARTITION:?}" in dev_cpuonly|cpuonly);;*)exit 71;;esac;test "${SLURM_CPUS_PER_TASK:?}" -eq 2;test "${SLURM_MEM_PER_NODE:?}" -eq 8192;cmp -s "$0" "$root/slurm/dino_rcde_h0_u4_absolute_gate_v1.sbatch";cd "$root";test "$(stat -c %a "$authority")" = 444;test ! -e "$result";test ! -e "$validation";export PYTHONPATH="$root/programs:$root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
"$python" programs/reduce_dino_rcde_h0_u4_absolute_gate_v1.py --authority "$authority" --aggregate "$raw" --output "$result"
"$python" programs/validate_dino_rcde_h0_u4_absolute_gate_v1.py --authority "$authority" --aggregate "$raw" --result "$result" --output "$validation"
jq -e '(.status=="H0_U4_CONDITIONAL_ABSOLUTE_VERIFICATION_GO" or .status=="H0_U4_CONDITIONAL_ABSOLUTE_VERIFICATION_NO_GO") and (.scientific_GO_or_NO_GO=="GO" or .scientific_GO_or_NO_GO=="NO_GO") and .metric_population.matched_scored==567 and .metric_population.unmatched_unscored==27 and .metric_population.fallback_count==0 and .retrieval_or_ownership_claim_authorized==false' "$result" >/dev/null
jq -e '.status=="H0_U4_ABSOLUTE_GATE_INDEPENDENT_VALIDATION_PASS" and .validation_pass==true and (.scientific_GO_or_NO_GO=="GO" or .scientific_GO_or_NO_GO=="NO_GO") and .unmatched_fallback_count==0' "$validation" >/dev/null
test "$(stat -c %a "$result")" = 444;test "$(stat -c %a "$validation")" = 444
