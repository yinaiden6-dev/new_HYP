#!/bin/bash
#SBATCH --job-name=rcde_h0bias
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
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC;python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python;authority=registry/h0/h0_bias_calibrator_optimization_authority_v28_20260824.json;raw=results/dino_rcde_h0_u4_raw_aggregate_v2/result.json;result=results/dino_rcde_h0_bias_calibrator_optimization_v1/result.json;validation=results/dino_rcde_h0_bias_calibrator_optimization_validation_v1/result.json
test "$#" -eq 0;test "${SLURM_SUBMIT_DIR:-}" = "$root";case "${SLURM_JOB_PARTITION:?}" in dev_cpuonly|cpuonly);;*)exit 71;;esac;test "${SLURM_CPUS_PER_TASK:?}" -eq 2;test "${SLURM_MEM_PER_NODE:?}" -eq 8192;cmp -s "$0" "$root/slurm/dino_rcde_h0_bias_calibrator_optimization_v1.sbatch";cd "$root";test "$(stat -c %a "$authority")" = 444;test ! -e "$result";test ! -e "$validation";export PYTHONPATH="$root/src:$root/programs" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
"$python" -m pytest -q tests/test_dino_rcde_h0_bias_calibrator_v1.py tests/test_fit_dino_rcde_h0_bias_calibrator_v1.py
"$python" programs/fit_dino_rcde_h0_bias_calibrator_v1.py --aggregate "$raw" --output "$result"
"$python" programs/validate_dino_rcde_h0_bias_calibrator_v1.py --aggregate "$raw" --result "$result" --output "$validation"
jq -e '.status=="H0_BIAS_CALIBRATOR_OPTIMIZATION_READY" and .all_qualification_gates_pass==true and (.qualification_gates|all) and .model_forward_count==0 and .scientific_GO_or_NO_GO==null and .calibrator_consumption_authorized==false' "$result" >/dev/null
jq -e '.status=="H0_BIAS_CALIBRATOR_OPTIMIZATION_INDEPENDENT_VALIDATION_PASS" and .validation_pass==true and .all_qualification_gates_recomputed_pass==true and .model_forward_count==0 and .scientific_GO_or_NO_GO==null' "$validation" >/dev/null
test "$(stat -c %a "$result")" = 444;test "$(stat -c %a "$validation")" = 444
