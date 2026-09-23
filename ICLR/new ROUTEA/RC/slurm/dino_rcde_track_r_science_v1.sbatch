#!/bin/bash
#SBATCH --job-name=rcde_tr_sci
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=65536
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
rc_authority="$rc_root/registry/current_authority_v123_20260822.json"
rc_runtime_validator="$rc_root/programs/validate_dino_rcde_track_r_v123_runtime_binding_v1.py"
rc_sealer="$rc_root/programs/seal_dino_rcde_track_r_i0_science_metadata_v1.py"
rc_seal_validator="$rc_root/programs/validate_dino_rcde_track_r_i0_science_metadata_v1.py"
rc_reducer="$rc_root/programs/reduce_dino_rcde_track_r_science_v1.py"
rc_validator="$rc_root/programs/validate_dino_rcde_track_r_science_v1.py"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_science_v1.sbatch"
rc_prejoin="$rc_root/results/dino_rcde_track_r_oof_prejoin_v1/producer"
rc_prejoin_validation="$rc_root/results/dino_rcde_track_r_oof_prejoin_v1/validation"
rc_p="$rc_root/results/dino_rcde_track_r_p_v2_utility_comparator_v1/result.json"
rc_p_validation="$rc_root/results/dino_rcde_track_r_p_v2_utility_comparator_validation_v1/result.json"
rc_registration="$rc_root/results/dino_rcde_track_r_autochain_v1/v123_registration.json"
rc_registration_validation="$rc_root/results/dino_rcde_track_r_autochain_v1/v123_registration.validation.json"
rc_promotion="$rc_root/results/dino_rcde_track_r_autochain_v1/v123_promotion.json"
rc_i0="$rc_root/results/dino_rcde_track_r_i0_science_metadata_v1/result.json"
rc_i0_validation="$rc_root/results/dino_rcde_track_r_i0_science_metadata_validation_v1/result.json"
rc_result="$rc_root/results/dino_rcde_track_r_scientific_result_v1/result.json"
rc_validation="$rc_root/results/dino_rcde_track_r_scientific_validation_v1/result.json"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 4
test "${SLURM_MEM_PER_NODE:?}" -eq 65536
cmp -s "$0" "$rc_launcher"
for path in "$rc_registration" "$rc_registration_validation" "$rc_promotion"; do
  test -f "$path"; test ! -L "$path"; test "$(stat -c %a "$path")" = 444
done
for path in "$rc_i0" "$rc_i0_validation" "$rc_result" "$rc_validation"; do test ! -e "$path"; done
export PYTHONPATH="$rc_root/src:$rc_root/programs:/hkfs/work/workspace/scratch/ap7811-benchmark"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$rc_root"
"$rc_python" - "$rc_registration" "$rc_registration_validation" "$rc_promotion" "$rc_authority" "${SLURM_JOB_ID:?}" <<'PY'
import hashlib,json,sys
from pathlib import Path

registration,validation,promotion,authority=map(Path,sys.argv[1:5]); science_job=int(sys.argv[5])
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def logical(value):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!="logical_sha256"},sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()
reg=json.loads(registration.read_text()); val=json.loads(validation.read_text()); pro=json.loads(promotion.read_text())
assert reg.get("status")=="V123_AUTOMATIC_CONTINUATION_REGISTERED" and reg.get("logical_sha256")==logical(reg)
assert val.get("status")=="V123_AUTOMATIC_CONTINUATION_REGISTRATION_INDEPENDENT_VALIDATION_PASS" and val.get("validation_pass") is True and val.get("registration_sha256")==sha(registration)
assert pro.get("status")=="V123_AUTHORITY_FROZEN_AND_SCIENCE_DEPENDENCY_RELEASED" and pro.get("logical_sha256")==logical(pro)
assert reg.get("science_job_id")==science_job==pro.get("science_job_id")
assert reg.get("promotion_job_id")==pro.get("promotion_job_id")
assert pro.get("registration_sha256")==sha(registration) and pro.get("registration_validation_sha256")==sha(validation)
assert pro.get("v123_authority_sha256")==sha(authority)
PY
"$rc_python" "$rc_runtime_validator" --authority "$rc_authority" --binding-name launcher --runtime-file "$rc_launcher"
"$rc_python" "$rc_sealer" --authority "$rc_authority" --output "$rc_i0"
"$rc_python" "$rc_seal_validator" --authority "$rc_authority" --seal "$rc_i0" --output "$rc_i0_validation"
"$rc_python" "$rc_reducer" \
  --authority "$rc_authority" \
  --prejoin-root "$rc_prejoin" \
  --validation-root "$rc_prejoin_validation" \
  --i0-seal "$rc_i0" \
  --i0-seal-validation "$rc_i0_validation" \
  --p-utility-comparator "$rc_p" \
  --p-utility-comparator-validation "$rc_p_validation" \
  --output "$rc_result"
"$rc_python" "$rc_validator" \
  --authority "$rc_authority" \
  --prejoin-root "$rc_prejoin" \
  --prejoin-validation-root "$rc_prejoin_validation" \
  --i0-seal "$rc_i0" \
  --i0-seal-validation "$rc_i0_validation" \
  --p-utility-comparator "$rc_p" \
  --p-utility-comparator-validation "$rc_p_validation" \
  --result "$rc_result" \
  --output "$rc_validation"
test "$(jq -er '.status' "$rc_validation")" = DINO_RCDE_TRACK_R_SCIENTIFIC_INDEPENDENT_VALIDATION_PASS
test "$(jq -er '.validation_pass' "$rc_validation")" = true
test "$(jq -r '.automatic_stage_advance' "$rc_validation")" = false
