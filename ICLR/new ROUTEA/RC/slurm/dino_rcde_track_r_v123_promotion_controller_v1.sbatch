#!/bin/bash
# Scheduler-only V122 comparator -> V123 authority promotion.
#SBATCH --job-name=rcde_tr_v123p
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16384
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
rc_self="$rc_root/slurm/dino_rcde_track_r_v123_promotion_controller_v1.sbatch"
rc_registration="$rc_root/results/dino_rcde_track_r_autochain_v1/v123_registration.json"
rc_registration_validation="$rc_root/results/dino_rcde_track_r_autochain_v1/v123_registration.validation.json"
rc_registration_validator="$rc_root/programs/validate_dino_rcde_track_r_v123_autochain_registration_v1.py"
rc_post_p_receipt="$rc_root/results/dino_rcde_track_r_autochain_v1/post_p_utility_submission.json"
rc_comparator="$rc_root/results/dino_rcde_track_r_p_v2_utility_comparator_v1/result.json"
rc_comparator_validation="$rc_root/results/dino_rcde_track_r_p_v2_utility_comparator_validation_v1/result.json"
rc_freezer="$rc_root/programs/freeze_dino_rcde_track_r_science_authority_v123.py"
rc_runtime_validator="$rc_root/programs/validate_dino_rcde_track_r_v123_runtime_binding_v1.py"
rc_authority="$rc_root/registry/current_authority_v123_20260822.json"
rc_promotion_receipt="$rc_root/results/dino_rcde_track_r_autochain_v1/v123_promotion.json"
rc_i0="$rc_root/results/dino_rcde_track_r_i0_science_metadata_v1/result.json"
rc_i0_validation="$rc_root/results/dino_rcde_track_r_i0_science_metadata_validation_v1/result.json"
rc_science="$rc_root/results/dino_rcde_track_r_scientific_result_v1/result.json"
rc_science_validation="$rc_root/results/dino_rcde_track_r_scientific_validation_v1/result.json"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 2
test "${SLURM_MEM_PER_NODE:?}" -eq 16384
case "${SLURM_JOB_ID:?}" in (*[!0-9]*|'') exit 72 ;; esac
cmp -s "$0" "$rc_self"
for path in "$rc_registration" "$rc_registration_validation" "$rc_post_p_receipt" "$rc_comparator" "$rc_comparator_validation"; do
  test -f "$path"; test ! -L "$path"; test "$(stat -c %a "$path")" = 444
done
for path in "$rc_authority" "$rc_promotion_receipt" "$rc_i0" "$rc_i0_validation" "$rc_science" "$rc_science_validation"; do
  test ! -e "$path"
done

cd "$rc_root"
export PYTHONPATH="$rc_root/src:$rc_root/programs:/hkfs/work/workspace/scratch/ap7811-benchmark"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

test "$(jq -er '.status' "$rc_post_p_receipt")" = V122_P_UTILITY_COMPARATOR_COMPLETE_AWAITING_SEPARATE_V123_AUTHORITY
jq -e '.automatic_stage_advance == false' "$rc_post_p_receipt" >/dev/null
test "$(jq -er '.status' "$rc_comparator")" = DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_READY
test "$(jq -er '.query_count' "$rc_comparator")" -eq 594
test "$(jq -er '.candidate_count_per_query' "$rc_comparator")" -eq 128
test "$(jq -er '.target_free_prejoin_score_ledger' "$rc_comparator")" = true
test "$(jq -er '.status' "$rc_comparator_validation")" = DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_INDEPENDENT_VALIDATION_PASS
test "$(jq -er '.validation_pass' "$rc_comparator_validation")" = true
test "$(jq -er '.promotion_job_id' "$rc_registration")" -eq "${SLURM_JOB_ID}"
test "$(jq -er '.comparator_job_id' "$rc_registration")" -eq 5102487
test "$(jq -er '.status' "$rc_registration_validation")" = V123_AUTOMATIC_CONTINUATION_REGISTRATION_INDEPENDENT_VALIDATION_PASS
test "$(jq -er '.validation_pass' "$rc_registration_validation")" = true

"$rc_python" "$rc_registration_validator" --registration "$rc_registration" --output "$rc_registration_validation" --dry-run
"$rc_python" -m py_compile \
  programs/register_dino_rcde_track_r_v123_autochain_v1.py \
  programs/validate_dino_rcde_track_r_v123_autochain_registration_v1.py \
  programs/freeze_dino_rcde_track_r_science_authority_v123.py \
  programs/validate_dino_rcde_track_r_v123_runtime_binding_v1.py \
  programs/seal_dino_rcde_track_r_i0_science_metadata_v1.py \
  programs/validate_dino_rcde_track_r_i0_science_metadata_v1.py \
  programs/reduce_dino_rcde_track_r_science_v1.py \
  programs/validate_dino_rcde_track_r_science_v1.py
"$rc_python" -m pytest -q \
  tests/test_dino_rcde_track_r_v122_p_utility_chain_v1.py \
  tests/test_dino_rcde_track_r_v123_science_chain_v1.py

"$rc_python" "$rc_freezer" --output "$rc_authority"
test "$(stat -c %a "$rc_authority")" = 444
test "$(jq -er '.status' "$rc_authority")" = DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_AUTHORIZED
"$rc_python" "$rc_runtime_validator" \
  --authority "$rc_authority" \
  --binding-name v123_promotion_controller \
  --runtime-file "$rc_self"

"$rc_python" - "$rc_registration" "$rc_registration_validation" "$rc_authority" "$rc_comparator" "$rc_comparator_validation" "$rc_promotion_receipt" <<'PY'
import hashlib,json,os,sys,tempfile
from pathlib import Path

registration,registration_validation,authority,comparator,comparator_validation,out=map(Path,sys.argv[1:])
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def logical(value):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!="logical_sha256"},sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()
reg=json.loads(registration.read_text()); auth=json.loads(authority.read_text())
value={
    "schema_version":"rc_dino_rcde_track_r_v123_promotion_v1_20260824",
    "status":"V123_AUTHORITY_FROZEN_AND_SCIENCE_DEPENDENCY_RELEASED",
    "claim_level":"ENGINEERING_AUTHORITY_PROMOTION_ONLY",
    "promotion_job_id":int(os.environ["SLURM_JOB_ID"]),
    "science_job_id":reg["science_job_id"],
    "science_dependency":reg["science_dependency"],
    "registration_sha256":sha(registration),
    "registration_validation_sha256":sha(registration_validation),
    "v123_authority_sha256":sha(authority),
    "v123_authority_logical_sha256":auth["logical_sha256"],
    "comparator_sha256":sha(comparator),
    "comparator_validation_sha256":sha(comparator_validation),
    "model_or_science_execution_count":0,
    "scientific_GO_or_NO_GO":None,
    "automatic_stage_advance_to_v123":True,
    "automatic_stage_advance_after_v123":False,
}
assert value["promotion_job_id"]==reg["promotion_job_id"]
value["logical_sha256"]=logical(value)
if out.exists() or out.is_symlink(): raise RuntimeError("immutable promotion receipt exists")
fd,temp=tempfile.mkstemp(prefix=f".{out.name}.",suffix=".partial",dir=out.parent); os.close(fd); path=Path(temp)
try:
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    path.chmod(0o444); os.link(path,out)
finally: path.unlink(missing_ok=True)
print(json.dumps({"status":value["status"],"science_job_id":value["science_job_id"]},sort_keys=True))
PY
