#!/bin/bash
#SBATCH --job-name=rcde_gxr0s
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077
root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
authority=registry/h0/gx_cbnr_formal_r0_science_authority_v68_20260825.json
result=results/dino_rcde_gx_formal_r0_science_v1/result.json
validation=results/dino_rcde_gx_formal_r0_science_validation_v1/result.json
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$root"
test "${SLURM_JOB_PARTITION:?}" = accelerated
test "${SLURM_CPUS_PER_TASK:?}" -eq 16
test "${SLURM_MEM_PER_NODE:?}" -eq 64000
test -n "${CUDA_VISIBLE_DEVICES:-}"
cmp -s "$0" "$root/slurm/dino_rcde_gx_formal_r0_science_v68.sbatch"
cd "$root"
test "$(stat -c %a "$authority")" = 444
test ! -e "$result"
test ! -e "$validation"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$root/programs:$root/src:$root/tests" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
"$python" programs/freeze_dino_rcde_gx_formal_r0_science_authority_v68.py --output "$authority" >/dev/null
"$python" -m pytest -q -p no:cacheprovider \
  tests/test_reduce_dino_rcde_gx_formal_r0_science_v1.py \
  tests/test_validate_dino_rcde_gx_formal_r0_science_v1.py \
  tests/test_freeze_dino_rcde_gx_formal_r0_science_authority_v68.py
"$python" programs/reduce_dino_rcde_gx_formal_r0_science_v1.py --authority "$authority" --output "$result"
"$python" programs/validate_dino_rcde_gx_formal_r0_science_v1.py --authority "$authority" --result "$result" --output "$validation"
jq -e '(.status=="GX_CBNR_R0_REPRESENTATION_GO" or .status=="GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY") and .scientific_GO_or_NO_GO==.status and .automatic_stage_advance==false' "$result" >/dev/null
jq -e '.status=="GX_CBNR_R0_SCIENTIFIC_INDEPENDENT_VALIDATION_PASS" and .validation_pass==true and (.scientific_GO_or_NO_GO=="GX_CBNR_R0_REPRESENTATION_GO" or .scientific_GO_or_NO_GO=="GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY") and .automatic_stage_advance==false' "$validation" >/dev/null
test "$(stat -c %a "$result")" = 444
test "$(stat -c %a "$validation")" = 444
