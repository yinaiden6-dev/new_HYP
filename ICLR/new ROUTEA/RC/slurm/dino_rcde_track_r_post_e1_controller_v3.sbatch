#!/bin/bash
# Fail-closed Track-R continuation from validated V118-r2 E1 shards to V-fit.
#SBATCH --job-name=rcde_tr_c1v3
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

rc_workspace=/hkfs/work/workspace/scratch/ap7811-benchmark
rc_root="$rc_workspace/ICLR/new ROUTEA/RC"
rc_python="$rc_workspace/.venv-colpali/bin/python"
rc_controller="$rc_root/slurm/dino_rcde_track_r_post_e1_controller_v3.sbatch"
rc_authority_v118="$rc_root/registry/current_authority_v118_20260821.json"
rc_authority_v118_validation="$rc_root/results/dino_rcde_track_r_relative_e1_authority_v118_revision2_validation_v1/result.json"
rc_authority_v119="$rc_root/registry/current_authority_v119_20260821.json"
rc_authority_v120="$rc_root/registry/current_authority_v120_20260821.json"
rc_authority_v121="$rc_root/registry/current_authority_v121_20260821.json"
rc_reduce_freezer="$rc_root/programs/freeze_dino_rcde_track_r_relative_e1_episode_reduction_authority_v119_r2.py"
rc_reducer="$rc_root/programs/reduce_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py"
rc_ledger_validator="$rc_root/programs/validate_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py"
rc_vfit_freezer="$rc_root/programs/freeze_dino_rcde_track_r_relative_v_fit_authority_v119.py"
rc_vfit_launcher="$rc_root/slurm/dino_rcde_track_r_relative_v_fit_v1.sbatch"
rc_post_vfit_controller="$rc_root/slurm/dino_rcde_track_r_post_vfit_controller_v1.sbatch"
rc_producer_root="$rc_root/results/dino_rcde_track_r_relative_e1_shards_v1/producer"
rc_shard_validation_root="$rc_root/results/dino_rcde_track_r_relative_e1_shards_v1/validation"
rc_ledger_root="$rc_root/results/dino_rcde_track_r_relative_e1_ledgers_v1"
rc_ledger_validation="$rc_root/results/dino_rcde_track_r_relative_e1_ledgers_validation_v1/result.json"
rc_vfit_root="$rc_root/results/dino_rcde_track_r_relative_v_fits_v1"
rc_oof_root="$rc_root/results/dino_rcde_track_r_oof_prejoin_v1"
rc_oof_completion="$rc_root/results/dino_rcde_track_r_oof_prejoin_completion_v120/result.json"
rc_receipt="$rc_root/results/dino_rcde_track_r_autochain_v1/post_e1_submission.json"
rc_post_vfit_receipt="$rc_root/results/dino_rcde_track_r_autochain_v1/post_vfit_submission.json"

test "$#" -eq 0
test "${SLURM_SUBMIT_DIR:-}" = "$rc_root"
test "${SLURM_JOB_PARTITION:?}" = cpuonly
test "${SLURM_CPUS_PER_TASK:?}" -eq 4
test "${SLURM_MEM_PER_NODE:?}" -eq 65536
case "${SLURM_JOB_ID:?}" in (*[!0-9]*|'') exit 72 ;; esac
test -x "$rc_python"
cmp -s "$0" "$rc_controller"

# A rerun may never adopt or overwrite a partially advanced chain.
for rc_absent in \
  "$rc_authority_v119" \
  "$rc_authority_v120" \
  "$rc_authority_v121" \
  "$rc_ledger_root" \
  "$rc_ledger_validation" \
  "$rc_vfit_root" \
  "$rc_oof_root" \
  "$rc_oof_completion" \
  "$rc_receipt" \
  "$rc_post_vfit_receipt"
do
  test ! -e "$rc_absent"
  test ! -L "$rc_absent"
done

for rc_required in \
  "$rc_authority_v118" \
  "$rc_authority_v118_validation" \
  "$rc_reduce_freezer" \
  "$rc_reducer" \
  "$rc_ledger_validator" \
  "$rc_vfit_freezer" \
  "$rc_vfit_launcher" \
  "$rc_post_vfit_controller"
do
  test -f "$rc_required"
  test ! -L "$rc_required"
done

# Independently close V118-r2, its authority validation, and all 50 immutable
# producer/validation pairs before the first downstream output is created.
"$rc_python" - "$rc_authority_v118" "$rc_authority_v118_validation" "$rc_producer_root" "$rc_shard_validation_root" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

authority_path, authority_validation_path, producer_root, validation_root = map(Path, sys.argv[1:])

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def canonical(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

def logical(value):
    return canonical({key: item for key, item in value.items() if key != "logical_sha256"})

def immutable_file(path, label):
    require(path.is_file() and not path.is_symlink(), f"{label} absent/unsafe")
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} not immutable")

immutable_file(authority_path, "V118 authority")
immutable_file(authority_validation_path, "V118 authority validation")
require(producer_root.is_dir() and not producer_root.is_symlink(), "E1 producer root absent/unsafe")
require(validation_root.is_dir() and not validation_root.is_symlink(), "E1 validation root absent/unsafe")
authority = json.loads(authority_path.read_text(encoding="utf-8"))
authority_validation = json.loads(authority_validation_path.read_text(encoding="utf-8"))
authority_sha = sha256(authority_path)
require(
    authority.get("schema_version") == "rc_current_authority_v118_20260821"
    and authority.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARDS_AUTHORIZED"
    and authority.get("authority_version") == 118
    and authority.get("authority_revision") == 2
    and authority.get("logical_sha256") == logical(authority)
    and authority.get("episode_shard_materialization_authorized") is True
    and authority.get("episode_shard_validation_authorized") is True
    and authority.get("training_authorized") is False
    and authority.get("heldout_scoring_authorized") is False
    and authority.get("scientific_GO_or_NO_GO") is None,
    "V118-r2 authority drift",
)
require(
    authority_validation.get("status")
    == "DINO_RCDE_TRACK_R_RELATIVE_E1_PARALLEL_AUTHORITY_V118_REVISION2_INDEPENDENT_VALIDATION_PASS"
    and authority_validation.get("validation_pass") is True
    and authority_validation.get("authority_version") == 118
    and authority_validation.get("authority_revision") == 2
    and authority_validation.get("authority_sha256") == authority_sha
    and authority_validation.get("authority_logical_sha256") == authority["logical_sha256"]
    and authority_validation.get("logical_sha256") == logical(authority_validation)
    and authority_validation.get("training_authorized") is False
    and authority_validation.get("scientific_GO_or_NO_GO") is None,
    "V118-r2 independent authority PASS absent",
)

expected_producers = []
expected_validations = []
eligible = excluded = episodes = 0
for ordinal in range(50):
    start = ordinal * 12
    stop = start + 12
    producer = producer_root / f"shard_{start:03d}_{stop:03d}.pt"
    validation = validation_root / f"shard_{start:03d}_{stop:03d}.validation.json"
    expected_producers.append(producer)
    expected_validations.append(validation)
    immutable_file(producer, f"E1 producer shard {ordinal}")
    immutable_file(validation, f"E1 validation shard {ordinal}")
    value = json.loads(validation.read_text(encoding="utf-8"))
    require(
        value.get("schema_version")
        == "rc_dino_rcde_track_r_relative_e1_episode_shard_validation_v1_20260821"
        and value.get("status")
        == "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_INDEPENDENT_VALIDATION_PASS"
        and value.get("validation_pass") is True
        and value.get("shard_ordinal") == ordinal
        and value.get("producer_shard_sha256") == sha256(producer)
        and value.get("logical_sha256") == logical(value),
        f"E1 shard {ordinal} independent PASS drift",
    )
    eligible += int(value["eligible_query_count"])
    excluded += int(value["excluded_query_count"])
    episodes += int(value["episode_count"])
require(
    eligible == 594 and excluded == 6 and episodes == 1782,
    "E1 594/6/1782 closure drift",
)
require(
    sorted(producer_root.glob("*.pt")) == expected_producers
    and sorted(validation_root.glob("*.validation.json")) == expected_validations
    and not list(producer_root.glob("*.partial"))
    and not list(validation_root.glob("*.partial")),
    "E1 exact 50-shard population drift",
)
PY

export PYTHONPATH="$rc_root/src:$rc_root/programs:$rc_workspace"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cd "$rc_root"
"$rc_python" "$rc_reduce_freezer"
test -f "$rc_authority_v119"
test ! -L "$rc_authority_v119"
test "$(stat -c '%a' "$rc_authority_v119")" = 444
"$rc_python" -c 'import hashlib,json,sys; ap,cp=sys.argv[1:]; sha=lambda p: hashlib.sha256(open(p,"rb").read()).hexdigest(); a=json.load(open(ap,encoding="utf-8")); b=a.get("bindings",{}).get("controller",{}); s=a.get("supersedes",{}); r=a.get("runtime_episode_address_repair",{}); assert a.get("schema_version")=="rc_current_authority_v119_20260821" and a.get("status")=="DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_REDUCTION_AUTHORIZED" and a.get("authority_version")==119 and a.get("authority_revision")==2 and a.get("parent_v118_authority_revision")==2 and a.get("all_shard_source_authority_match") is True; assert s.get("physical_sha256")=="d54b40a969a81d5706600bbb76c7cd6ea04198397ed8504e626e596b481cef51" and s.get("logical_sha256")=="e8775882e7fea0afb630dd7de75599c0d904778fb835f50510a786beaf110938" and s.get("failure_job_id")==5097842; assert r.get("namespace")=="DINO_RCDE_TRACK_R_LEDGER_RUNTIME_EPISODE_ID_V2" and r.get("source_shard_or_episode_mutation_authorized") is False and r.get("source_episode_id_collision_count",0)>0 and r.get("runtime_episode_id_collision_count")==0; assert b.get("path")=="slurm/dino_rcde_track_r_post_e1_controller_v3.sbatch" and b.get("sha256")==sha(cp)' "$rc_authority_v119" "$rc_controller"

"$rc_python" "$rc_reducer" \
  --authority "$rc_authority_v119" \
  --producer-root "$rc_producer_root" \
  --validation-root "$rc_shard_validation_root" \
  --output-root "$rc_ledger_root"
mkdir -p "$(dirname "$rc_ledger_validation")"
"$rc_python" "$rc_ledger_validator" \
  --authority "$rc_authority_v119" \
  --producer-root "$rc_producer_root" \
  --shard-validation-root "$rc_shard_validation_root" \
  --ledger-root "$rc_ledger_root" \
  --output "$rc_ledger_validation"
test "$(jq -er '.status' "$rc_ledger_validation")" = DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGERS_INDEPENDENT_VALIDATION_PASS
test "$(jq -er '.validation_pass' "$rc_ledger_validation")" = true
test "$(stat -c '%a' "$rc_ledger_validation")" = 444

"$rc_python" "$rc_vfit_freezer"
test -f "$rc_authority_v120"
test ! -L "$rc_authority_v120"
test "$(stat -c '%a' "$rc_authority_v120")" = 444
test "$(jq -er '.status' "$rc_authority_v120")" = DINO_RCDE_TRACK_R_RELATIVE_V_FIT_AUTHORIZED
test "$(jq -r '.scientific_GO_or_NO_GO' "$rc_authority_v120")" = null

rc_vfit_job=$(sbatch --parsable "$rc_vfit_launcher")
case "$rc_vfit_job" in (*[!0-9]*|'') exit 73 ;; esac
rc_post_vfit_job=$(sbatch --parsable --dependency="afterok:$rc_vfit_job" "$rc_post_vfit_controller")
case "$rc_post_vfit_job" in (*[!0-9]*|'') exit 74 ;; esac

mkdir -p "$(dirname "$rc_receipt")"
"$rc_python" - "$rc_receipt" "$rc_authority_v118" "$rc_authority_v119" "$rc_authority_v120" "$rc_controller" "$rc_vfit_launcher" "$rc_post_vfit_controller" "$rc_vfit_job" "$rc_post_vfit_job" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

receipt, v118, v119, v120, controller, vfit_launcher, post_vfit_controller = map(Path, sys.argv[1:8])
vfit_job = int(sys.argv[8])
post_vfit_job = int(sys.argv[9])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def logical(value):
    payload = json.dumps({key: item for key, item in value.items() if key != "logical_sha256"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

if receipt.exists() or receipt.is_symlink():
    raise RuntimeError("immutable post-E1 submission receipt exists")
authorities = {}
for name, path in (("v118", v118), ("v119", v119), ("v120", v120)):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("logical_sha256") != logical(value):
        raise RuntimeError(f"{name} logical hash drift before receipt")
    authorities[name] = {
        "path": path.name,
        "sha256": sha256(path),
        "logical_sha256": value["logical_sha256"],
        "status": value["status"],
    }
value = {
    "schema_version": "rc_dino_rcde_track_r_autochain_post_e1_v1_20260822",
    "status": "TRACK_R_V120_VFIT_ARRAY_AND_POST_VFIT_CONTROLLER_SUBMITTED",
    "claim_level": "ENGINEERING_SCHEDULER_CONTINUATION_RECEIPT_ONLY",
    "controller_job_id": int(os.environ["SLURM_JOB_ID"]),
    "vfit_array_job_id": vfit_job,
    "post_vfit_controller_job_id": post_vfit_job,
    "dependency": f"afterok:{vfit_job}",
    "authorities": authorities,
    "controller_sha256": sha256(controller),
    "vfit_launcher_sha256": sha256(vfit_launcher),
    "post_vfit_controller_sha256": sha256(post_vfit_controller),
    "training_result_claimed": False,
    "scientific_GO_or_NO_GO": None,
    "automatic_stage_advance": False,
}
value["logical_sha256"] = logical(value)
descriptor, temporary = tempfile.mkstemp(prefix=f".{receipt.name}.", suffix=".partial", dir=receipt.parent)
temp = Path(temporary)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.chmod(0o444)
    os.link(temp, receipt)
finally:
    temp.unlink(missing_ok=True)
print(json.dumps({"status": value["status"], "vfit_array_job_id": vfit_job, "post_vfit_controller_job_id": post_vfit_job}, sort_keys=True))
PY

test -f "$rc_receipt"
test ! -L "$rc_receipt"
test "$(stat -c '%a' "$rc_receipt")" = 444
