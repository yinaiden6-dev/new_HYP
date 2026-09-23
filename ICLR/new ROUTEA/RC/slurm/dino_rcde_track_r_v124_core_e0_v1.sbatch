#!/bin/bash
#SBATCH --job-name=rcde_tr_v124e0
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32768
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail
umask 077

rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_authority="$rc_root/registry/dino_rcde_track_r_v124_core_e0_authority_v1_20260824.json"
rc_runtime_validator="$rc_root/programs/validate_dino_rcde_track_r_v124_core_e0_v1.py"
rc_reducer="$rc_root/programs/reduce_dino_rcde_track_r_v124_core_e0_v1.py"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_v124_core_e0_v1.sbatch"
rc_rootwise_test="$rc_root/tests/test_dino_rcde_sr0_mt_v_runtime_v2.py"
rc_c128_test="$rc_root/tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py"
rc_output_root="$rc_root/results/dino_rcde_track_r_v124_core_e0_v1"
rc_junit="$rc_output_root/pytest.xml"
rc_result="$rc_output_root/result.json"
rc_validation="$rc_root/results/dino_rcde_track_r_v124_core_e0_validation_v1/result.json"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 2
test "${SLURM_MEM_PER_NODE:?}" -eq 32768
cmp -s "$0" "$rc_launcher"
for path in "$rc_output_root" "$rc_validation"; do test ! -e "$path"; done

export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_root/tests:/hkfs/work/workspace/scratch/ap7811-benchmark"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$rc_root"
"$rc_python" "$rc_runtime_validator" --authority "$rc_authority" --authority-only
mkdir -p "$rc_output_root"
"$rc_python" -m pytest -q -p no:cacheprovider \
  --junitxml "$rc_junit" \
  "$rc_rootwise_test" "$rc_c128_test"
"$rc_python" "$rc_reducer" \
  --authority "$rc_authority" --junit "$rc_junit" --output "$rc_result"
"$rc_python" "$rc_runtime_validator" \
  --authority "$rc_authority" --junit "$rc_junit" --result "$rc_result" \
  --output "$rc_validation"
test "$(jq -er '.status' "$rc_result")" = DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_PASS
jq -e '.status == "DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_INDEPENDENT_VALIDATION_PASS" and .validation_pass == true and .scientific_GO_or_NO_GO == null' "$rc_validation" >/dev/null
