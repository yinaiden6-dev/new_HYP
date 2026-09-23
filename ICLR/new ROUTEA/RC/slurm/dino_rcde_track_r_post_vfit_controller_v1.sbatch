#!/bin/bash
#SBATCH --job-name=rcde_tr_c2
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --input=/dev/null
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NIL
set -euo pipefail

rc_root=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC
rc_python=/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python
rc_self="$rc_root/slurm/dino_rcde_track_r_post_vfit_controller_v1.sbatch"
rc_v120="$rc_root/registry/current_authority_v120_20260821.json"
rc_authority="$rc_root/registry/current_authority_v121_20260821.json"
rc_runtime_validator="$rc_root/programs/validate_dino_rcde_track_r_v121_runtime_binding_v1.py"
for rc_fold in 1 2 3 4; do
  rc_validation="$rc_root/results/dino_rcde_track_r_relative_v_fits_v1/outer_fold${rc_fold}/fresh_resume_validation.json"
  test "$(jq -er '.status' "$rc_validation")" = DINO_RCDE_TRACK_R_RELATIVE_V_FIT_FRESH_RESUME_VALIDATION_PASS
  test "$(jq -er '.validation_pass' "$rc_validation")" = true
done
rc_freezer="$rc_root/programs/freeze_dino_rcde_track_r_oof_prejoin_authority_v120.py"
rc_launcher="$rc_root/slurm/dino_rcde_track_r_oof_prejoin_shards_v1.sbatch"
rc_next_controller="$rc_root/slurm/dino_rcde_track_r_post_oof_controller_v1.sbatch"
rc_receipt="$rc_root/results/dino_rcde_track_r_autochain_v1/post_vfit_submission.json"
test -f "$rc_freezer"
test -f "$rc_launcher"
test -f "$rc_next_controller"
test -f "$rc_runtime_validator"
test -f "$rc_v120"
test ! -L "$rc_v120"
test ! -e "$rc_receipt"
cmp -s "$0" "$rc_self"
cd "$rc_root"
"$rc_python" "$rc_freezer"
"$rc_python" "$rc_runtime_validator" \
  --authority "$rc_authority" \
  --binding-name post_vfit_controller \
  --runtime-file "$rc_self"
rc_oof_job=$(sbatch --parsable "$rc_launcher")
rc_controller_job=$(sbatch --parsable --dependency="afterok:$rc_oof_job" "$rc_next_controller")
mkdir -p "$(dirname "$rc_receipt")"
"$rc_python" - "$rc_receipt" "$rc_v120" "$rc_authority" "$rc_self" "$rc_launcher" "$rc_next_controller" "$rc_oof_job" "$rc_controller_job" <<'PY'
import hashlib,json,os
from pathlib import Path
import sys,tempfile
receipt,v120,v121,self_path,launcher,next_controller=map(Path,sys.argv[1:7]); oof_job=int(sys.argv[7]); controller_job=int(sys.argv[8])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def logical(v): return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!="logical_sha256"},sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()
a120=json.loads(v120.read_text(encoding="utf-8")); a121=json.loads(v121.read_text(encoding="utf-8"))
assert a120.get("authority_revision")==5
assert a121.get("parent_v120_authority_revision")==5
assert a121.get("parent_v120_authority_sha256")==sha(v120)
v={"schema_version":"rc_dino_rcde_track_r_autochain_post_vfit_v2_20260822","status":"TRACK_R_V120_OOF_ARRAY_AND_FINALIZER_SUBMITTED","claim_level":"ENGINEERING_SCHEDULER_REGISTRATION_ONLY","controller_job_id":int(os.environ["SLURM_JOB_ID"]),"v120_authority_sha256":sha(v120),"v120_authority_logical_sha256":a120["logical_sha256"],"v121_authority_sha256":sha(v121),"v121_authority_logical_sha256":a121["logical_sha256"],"oof_array_job_id":oof_job,"post_oof_controller_job_id":controller_job,"oof_dependency":None,"post_oof_dependency":f"afterok:{oof_job}","post_vfit_controller_sha256":sha(self_path),"oof_launcher_sha256":sha(launcher),"post_oof_controller_sha256":sha(next_controller),"model_or_science_execution_count":0,"scientific_GO_or_NO_GO":None,"automatic_stage_advance":False}; v["logical_sha256"]=logical(v)
if receipt.exists() or receipt.is_symlink(): raise RuntimeError("immutable post-Vfit receipt exists")
fd,t=tempfile.mkstemp(prefix=f".{receipt.name}.",suffix=".partial",dir=receipt.parent); os.close(fd); q=Path(t)
try: q.write_text(json.dumps(v,sort_keys=True,indent=2)+"\n"); q.chmod(0o444); os.link(q,receipt)
finally: q.unlink(missing_ok=True)
PY
