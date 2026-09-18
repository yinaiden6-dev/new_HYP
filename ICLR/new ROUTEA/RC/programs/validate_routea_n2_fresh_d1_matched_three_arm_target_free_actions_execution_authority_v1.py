#!/usr/bin/env python3
"""Fail-closed validation of the target-free A0 execution authority."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_EXECUTION_AUTHORITY_V1_20260904.json"
PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1.py"
A0_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1.py"
LAUNCHER = ROOT / "slurm/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1_10m.sbatch"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1"
VERSION = "routea_n2_fresh_d1_matched_three_arm_target_free_actions_execution_authority_v1_20260904"
STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_EXECUTION_AUTHORIZED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_EXECUTION"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_READY"
VALID = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_VALIDATED"

class AuthorityError(RuntimeError): pass
def req(x: bool, m: str) -> None:
    if not x: raise AuthorityError(m)
def sha(p: Path) -> str:
    req(p.is_file() and not p.is_symlink(), f"missing/nonregular {p}")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def canon(x: Any) -> str: return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def logical(x: Mapping[str,Any]) -> str:
    y=dict(x); y.pop("logical_sha256",None); return canon(y)
def read(p: Path) -> dict[str,Any]:
    req(p.is_file() and not p.is_symlink(),f"missing JSON {p}"); x=json.loads(p.read_text()); req(isinstance(x,dict),"JSON root"); return x
def bind(b: Any) -> None:
    req(isinstance(b,Mapping) and set(b) in ({"path","sha256"},{"path","sha256","logical_sha256"}),"binding schema")
    p=ROOT/str(b["path"]); req(sha(p)==b["sha256"],f"physical binding {p}")
    if "logical_sha256" in b:
        x=read(p); req(x.get("logical_sha256")==b["logical_sha256"]==logical(x),f"logical binding {p}")
def output_mode(producer_sha: str, validator_sha: str) -> str:
    if not OUT_ROOT.exists(): return "FRESH_OUTPUT_ROOT_ABSENT"
    req(OUT_ROOT.is_dir() and not OUT_ROOT.is_symlink(),"output root type")
    req({p.name for p in OUT_ROOT.iterdir()}=={"actions.json","independent_validation.json"},"partial/unexpected output")
    a,v=read(OUT_ROOT/"actions.json"),read(OUT_ROOT/"independent_validation.json")
    req(a.get("status")==READY and a.get("logical_sha256")==logical(a),"action seal")
    req(v.get("status")==VALID and v.get("logical_sha256")==logical(v),"validation seal")
    req(v.get("actions_sha256")==sha(OUT_ROOT/"actions.json") and v.get("actions_logical_sha256")==a["logical_sha256"] and v.get("producer_sha256")==producer_sha and v.get("validator_sha256")==validator_sha and all(v.get("checks",{}).values()),"output closure")
    return "EXISTING_OUTPUTS_EXACTLY_REVALIDATED"
def main() -> None:
    a=read(AUTHORITY)
    req(a.get("version")==VERSION and a.get("status")==STATUS and a.get("logical_sha256")==logical(a),"authority envelope")
    req(a.get("claim_level")=="ENGINEERING_TARGET_FREE_EVAL32_ACTION_SEAL_EXECUTION_NO_LABEL_OR_SCIENTIFIC_CLAIM","claim")
    for b in a.get("inputs",{}).values(): bind(b)
    aggregates={"current64":read(ROOT/a["inputs"]["current64_aggregate"]["path"])}
    for group,spec in a.get("input_shards",{}).items():
        req(group in aggregates,"shard group"); items=aggregates[group].get("shards")
        req(spec=={"count":8,"canonical_sha256":canon(items)} and len(items)==8,"shard seal digest")
        for item in items:
            s=int(item["shard"]); d=ROOT/f"results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/shard{s:02d}"
            req(item["payload_sha256"]==sha(d/"payload.pt") and item["receipt_sha256"]==sha(d/"receipt.json") and item["validation_sha256"]==sha(d/"validation.json"),"shard physical")
            x=read(d/"validation.json"); req(item["validation_logical_sha256"]==x.get("logical_sha256")==logical(x),"shard logical")
    ps,vs=sha(PRODUCER),sha(A0_VALIDATOR)
    req(a.get("programs")=={"producer":{"path":str(PRODUCER.relative_to(ROOT)),"sha256":ps},"independent_validator":{"path":str(A0_VALIDATOR.relative_to(ROOT)),"sha256":vs}},"programs")
    bind(a.get("authority_validator")); bind(a.get("launcher"))
    forbidden=a.get("forbidden_inputs",{})
    req(set(forbidden)=={"j1_train_label_ledger","j1_eval_label_ledger","n2_label_authority","pair64_postseal_fixed_pairs"},"forbidden roles")
    source=PRODUCER.read_text()+A0_VALIDATOR.read_text()
    for path in forbidden.values(): req(str(path) not in source and Path(str(path)).name not in source,"forbidden source reference")
    req(a.get("outputs")=={"actions":{"path":"results/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1/actions.json","status":READY},"independent_validation":{"path":"results/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1/independent_validation.json","status":VALID}},"outputs")
    req(a.get("resource_contract")=={"allowed_partitions":["cpuonly","dev_cpuonly"],"default_partition":"cpuonly","cpus_per_task":4,"memory_gib":32,"walltime_seconds":600,"gpu_count":0},"resources")
    text=LAUNCHER.read_text(); first=text.find('"$python_bin" "$authority_validator"'); prod=text.find('"$python_bin" "$producer"'); val=text.find('"$python_bin" "$independent_validator"'); last=text.rfind('"$python_bin" "$authority_validator"')
    req("#SBATCH --partition=cpuonly" in text and "#SBATCH --cpus-per-task=4" in text and "#SBATCH --mem=32G" in text and "#SBATCH --time=00:10:00" in text and "#SBATCH --export=NIL" in text and "${SLURM_TMPDIR:-/tmp}" in text and "export PATH=/usr/local/bin:/usr/bin:/bin" in text and 0<=first<prod<val<last,"launcher")
    if os.environ.get("SLURM_JOB_ID"):
        req(os.environ.get("SLURM_JOB_PARTITION") in {"cpuonly","dev_cpuonly"} and os.environ.get("SLURM_CPUS_PER_TASK")=="4","runtime resources")
    req(a.get("access_boundary")=={"label_ledger_open_count":0,"target_identity_read_count":0,"target_position_read_count":0,"model_update_count":0,"external_read_count":0,"sealed_read_count":0},"access")
    req(a.get("hardware_identity_is_authority") is False and a.get("scientific_GO_or_NO_GO") is None and a.get("automatic_stage_advance") is False and a.get("next_authorized_stage")==NEXT,"claim/next")
    print(json.dumps({"status":STATUS,"authority_sha256":sha(AUTHORITY),"output_mode":output_mode(ps,vs)},sort_keys=True))
if __name__=="__main__": main()
