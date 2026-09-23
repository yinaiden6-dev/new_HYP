#!/bin/bash
# PJ2 freezes the PJ1-bound reduction authority, computes the registered
# C128-relative reduction, independently validates it, and stops.
#SBATCH --job-name=rcde_h0tfp2
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077
rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_self="$rc_root/slurm/dino_rcde_h0_c128_target_free_postjoin_pj2_v1.sbatch"
rc_authority="$rc_root/registry/h0/h0_c128_target_free_postjoin_pj2_authority_v1_20260824.json"
rc_result="$rc_root/results/dino_rcde_h0_c128_target_free_postjoin_pj2_v1/result.json"
rc_validation="$rc_root/results/dino_rcde_h0_c128_target_free_postjoin_pj2_validation_v1/result.json"
test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 4
test "${SLURM_MEM_PER_NODE:?}" -eq 32768
cmp -s "$0" "$rc_self"
test ! -e "$rc_authority"
test ! -e "$rc_result"
test ! -e "$rc_validation"
cd "$rc_root"
export PYTHONPATH="$rc_root/programs:$rc_root/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$rc_python" -m pytest -q \
  tests/test_freeze_dino_rcde_h0_c128_target_free_postjoin_pj2_authority_v1.py \
  tests/test_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py
"$rc_python" programs/freeze_dino_rcde_h0_c128_target_free_postjoin_pj2_authority_v1.py --output "$rc_authority"
"$rc_python" programs/reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py --authority "$rc_authority" --output "$rc_result"
"$rc_python" programs/validate_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py --authority "$rc_authority" --result "$rc_result" --output "$rc_validation"
jq -e '(.status=="RELATIVE_C128_RANK_INCREMENT_GO" or .status=="RELATIVE_C128_RANK_INCREMENT_NO_GO") and (.scientific_GO_or_NO_GO=="GO" or .scientific_GO_or_NO_GO=="NO_GO") and .automatic_stage_advance==false and .next_authorized_stage==null' "$rc_result" >/dev/null
jq -e '.validation_pass==true and (.scientific_GO_or_NO_GO=="GO" or .scientific_GO_or_NO_GO=="NO_GO") and .automatic_stage_advance==false and .next_authorized_stage==null' "$rc_validation" >/dev/null
test "$(stat -c %a "$rc_authority")" = 444
test "$(stat -c %a "$rc_result")" = 444
test "$(stat -c %a "$rc_validation")" = 444
