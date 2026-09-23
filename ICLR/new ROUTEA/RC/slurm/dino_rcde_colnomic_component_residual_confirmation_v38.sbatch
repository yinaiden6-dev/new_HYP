#!/bin/bash
#SBATCH --job-name=rcde_rawres
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
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
authority=registry/h0/colnomic_component_residual_confirmation_authority_v38_20260824.json
result=results/dino_rcde_colnomic_component_residual_confirmation_v1/result.json
validation=results/dino_rcde_colnomic_component_residual_confirmation_validation_v1/result.json
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$root"
case "${SLURM_JOB_PARTITION:?}" in dev_cpuonly|cpuonly);;*)exit 71;;esac
test "${SLURM_CPUS_PER_TASK:?}" -eq 2
test "${SLURM_MEM_PER_NODE:?}" -eq 8192
cmp -s "$0" "$root/slurm/dino_rcde_colnomic_component_residual_confirmation_v38.sbatch"
cd "$root"
test "$(stat -c %a "$authority")" = 444
test ! -e "$result"
test ! -e "$validation"
export PYTHONPATH="$root/programs:$root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
"$python" -m pytest -q tests/test_dino_rcde_colnomic_component_residual_v1.py tests/test_reduce_dino_rcde_colnomic_component_residual_confirmation_v1.py tests/test_freeze_dino_rcde_colnomic_component_residual_confirmation_authority_v38.py tests/test_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py
"$python" programs/reduce_dino_rcde_colnomic_component_residual_confirmation_v1.py --authority "$authority" --output "$result"
"$python" programs/validate_dino_rcde_colnomic_component_residual_confirmation_v1.py --authority "$authority" --result "$result" --output "$validation"
jq -e '(.status=="COLNOMIC_COMPONENT_RESIDUAL_INTERNAL_GO" or .status=="COLNOMIC_COMPONENT_RESIDUAL_INTERNAL_NO_GO") and .query_count==582 and .residual_weight==1 and .alternative_weight_count==0 and .next_authorized_stage==null' "$result" >/dev/null
jq -e '.status=="COLNOMIC_COMPONENT_RESIDUAL_INDEPENDENT_VALIDATION_PASS" and .validation_pass==true and .query_count==582 and .next_authorized_stage==null' "$validation" >/dev/null
test "$(stat -c %a "$result")" = 444
test "$(stat -c %a "$validation")" = 444
