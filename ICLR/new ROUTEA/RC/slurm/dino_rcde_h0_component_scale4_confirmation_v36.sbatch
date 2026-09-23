#!/bin/bash
#SBATCH --job-name=rcde_h0s4
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
authority=registry/h0/h0_component_scale4_confirmation_authority_v36_20260824.json
result=results/dino_rcde_h0_component_scale4_confirmation_v1/result.json
validation=results/dino_rcde_h0_component_scale4_confirmation_validation_v1/result.json
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 4
test "${SLURM_MEM_PER_NODE:?}" -eq 16000
cmp -s "$0" "$root/slurm/dino_rcde_h0_component_scale4_confirmation_v36.sbatch"
cd "$root"
test "$(stat -c %a "$authority")" = 444
test ! -e "$result"
test ! -e "$validation"
export PYTHONPATH="$root/programs:$root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
"$python" -m pytest -q tests/test_dino_rcde_h0_component_scale4_v1.py tests/test_reduce_dino_rcde_h0_component_scale4_confirmation_v1.py tests/test_freeze_dino_rcde_h0_component_scale4_confirmation_authority_v36.py tests/test_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py
"$python" programs/reduce_dino_rcde_h0_component_scale4_confirmation_v1.py --authority "$authority" --output "$result"
"$python" programs/validate_dino_rcde_h0_component_scale4_confirmation_v1.py --authority "$authority" --result "$result" --output "$validation"
jq -e '(.status=="COMPONENT_SCALE4_INTERNAL_CONFIRMATION_GO" or .status=="COMPONENT_SCALE4_INTERNAL_CONFIRMATION_NO_GO") and .query_count==582 and .preregistered_scale==4 and .alternative_scale_count==0 and .next_authorized_stage==null' "$result" >/dev/null
jq -e '.status=="COMPONENT_SCALE4_INTERNAL_CONFIRMATION_INDEPENDENT_VALIDATION_PASS" and .validation_pass==true and .query_count==582 and .next_authorized_stage==null' "$validation" >/dev/null
test "$(stat -c %a "$result")" = 444
test "$(stat -c %a "$validation")" = 444
