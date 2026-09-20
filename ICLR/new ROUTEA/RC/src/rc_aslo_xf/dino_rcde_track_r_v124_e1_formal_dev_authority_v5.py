"""Formal E1 resource-only dev staged authority V5."""

from __future__ import annotations
import json, os
from pathlib import Path
import stat
from typing import Any, Mapping

from . import dino_rcde_track_r_v124_e1_authority_v3 as PARENT

ROOT=PARENT.RC_ROOT
AUTHORITY_PATH="registry/dino_rcde_track_r_v124_e1_formal_dev_staged_authority_v5_20260826.json"
PARENT_AUTHORITY_PATH="registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_authority_v3_20260826.json"
PARENT_VALIDATION_PATH="registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_authority_v3_20260826.independent_validation.json"
PRODUCER_NAMESPACE="results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3_committed"
VALIDATION_NAMESPACE="results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_validation_v3_committed"
OVERLAY_PATHS={
 "addendum":"plan/DINO_RCDE_TRACK_R_V124_E1_FORMAL_DEV_STAGED_RESOURCE_ADDENDUM_V5_20260826.md",
 "authority_runtime":"src/rc_aslo_xf/dino_rcde_track_r_v124_e1_formal_dev_authority_v5.py",
 "producer":"programs/run_dino_rcde_track_r_v124_e1_formal_dev_producer_v5.py",
 "validator":"programs/validate_dino_rcde_track_r_v124_e1_formal_dev_independent_v5.py",
 "producer_launcher":"slurm/dino_rcde_track_r_v124_e1_formal_dev_producer_v5.sbatch",
 "validator_launcher":"slurm/dino_rcde_track_r_v124_e1_formal_dev_validator_v5.sbatch",
 "freezer":"programs/freeze_dino_rcde_track_r_v124_e1_formal_dev_authority_v5.py",
 "authority_validator":"programs/validate_dino_rcde_track_r_v124_e1_formal_dev_authority_v5.py",
 "tests":"tests/test_dino_rcde_track_r_v124_e1_formal_dev_v5.py",
 "parent_authority":PARENT_AUTHORITY_PATH,
 "parent_validation":PARENT_VALIDATION_PATH,
 "smoke_submission":"registry/dino_rcde_track_r_v124_e1_dev_staged_smoke_v4_submission_receipt_20260826.json",
 "smoke_producer_result":"results/dino_rcde_track_r_v124_e1_v3_dev_staged_smoke_v4_producer_committed/producer_result.json",
 "smoke_producer_commit":"results/dino_rcde_track_r_v124_e1_v3_dev_staged_smoke_v4_producer_committed/family_commit.json",
 "smoke_validation_result":"results/dino_rcde_track_r_v124_e1_v3_dev_staged_smoke_v4_validation_committed/result.json",
 "smoke_validation_commit":"results/dino_rcde_track_r_v124_e1_v3_dev_staged_smoke_v4_validation_committed/family_commit.json",
}

class FormalDevAuthorityV5Error(RuntimeError): pass
def require(x:object,m:str)->None:
    if not x: raise FormalDevAuthorityV5Error(m)
def _raw(v:str|Path)->Path:
    p=Path(v); return Path(os.path.abspath(os.fspath(p if p.is_absolute() else ROOT/p)))
def _sha(p:Path)->str:return PARENT.V2.file_sha256(p)
def _regular(v:str|Path,mode:int|None=None)->Path:
    p=_raw(v); require(os.path.lexists(p),f"absent: {p}"); s=os.lstat(p); require(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode),f"type/symlink: {p}")
    if mode is not None: require(s.st_mode&0o777==mode,f"mode: {p}")
    require(p.resolve()==p,f"alias: {p}"); return p
def logical(v:Mapping[str,Any])->str:return PARENT.V2.logical_sha256(v)

def validate_candidate(v:Mapping[str,Any])->None:
    keys={"schema_version","status","claim_level","parent_authority_path","parent_authority_sha256","parent_validation_path","parent_validation_sha256","execution_scope","resource_contract","stage_sequence","output_contract","bindings","binding_modes","eligible_as_e1_result","formal_or_full594_execution_authorized","target_rival_join_authorized","postjoin_authorized","scientific_reduction_authorized","automatic_stage_advance","automatic_submit_authorized","scientific_GO_or_NO_GO","next_authorized_stage","logical_sha256"}
    require(set(v)==keys,"V5 key set")
    require(v["schema_version"]=="rc_dino_rcde_track_r_v124_e1_formal_dev_authority_v5_20260826" and v["status"]=="DINO_RCDE_TRACK_R_V124_E1_FORMAL_DEV_STAGED_V5_EXECUTION_AUTHORIZED" and v["claim_level"]=="FORMAL_E1_TARGET_FREE_PREJOIN_ENGINEERING_ONLY" and v["parent_authority_path"]==PARENT_AUTHORITY_PATH and v["parent_validation_path"]==PARENT_VALIDATION_PATH and v["execution_scope"]=="EXECUTION0_FORMAL_E1_DEV_ACCELERATED_TWO_STAGE_PREJOIN_V5" and v["resource_contract"]=={"partition":"dev_accelerated","allowed_partitions":["dev_accelerated"],"gpu_count":1,"cpus_per_task":8,"memory_megabytes":128000,"walltime_seconds_per_stage":3600} and v["stage_sequence"]==[{"stage":0,"role":"FORMAL_PRODUCER","dependency":None},{"stage":1,"role":"FORMAL_INDEPENDENT_VALIDATOR","dependency":"afterok:stage0"}] and v["output_contract"]=={"producer_family":PRODUCER_NAMESPACE,"validation_family":VALIDATION_NAMESPACE,"append_only":True,"atomic_family_required":True,"producer_commit_is_stage_checkpoint":True,"exact_committed_reuse":True,"partial_family_consumable":False,"eligible_as_e1_result":True} and v["eligible_as_e1_result"] is True and v["formal_or_full594_execution_authorized"] is False and v["target_rival_join_authorized"] is False and v["postjoin_authorized"] is False and v["scientific_reduction_authorized"] is False and v["automatic_stage_advance"] is False and v["automatic_submit_authorized"] is False and v["scientific_GO_or_NO_GO"] is None and v["next_authorized_stage"] is None and v["logical_sha256"]==logical(v),"V5 semantic drift")
    require(set(v["bindings"])==set(OVERLAY_PATHS)==set(v["binding_modes"]),"V5 binding set")
    for k,e in OVERLAY_PATHS.items():
        row=v["bindings"][k]; require(set(row)=={"path","bytes","sha256"} and row["path"]==e,f"binding {k}"); p=_regular(e,v["binding_modes"][k]); require(p.stat().st_size==row["bytes"] and _sha(p)==row["sha256"],f"hash {k}")

def read_authority(path:Path,expected_sha256:str)->dict[str,Any]:
    ap=_regular(path,0o444); require(ap==_raw(AUTHORITY_PATH) and _sha(ap)==expected_sha256,"authority physical")
    v=json.loads(ap.read_text(encoding="ascii")); validate_candidate(v)
    pp=_regular(PARENT_AUTHORITY_PATH,0o444); pv=_regular(PARENT_VALIDATION_PATH,0o444); require(_sha(pp)==v["parent_authority_sha256"] and _sha(pv)==v["parent_validation_sha256"],"parent hash")
    validation=json.loads(pv.read_text()); require(validation["status"]=="DINO_RCDE_TRACK_R_V124_E1_FORMAL_AUTHORITY_V3_INDEPENDENT_VALIDATION_PASS","parent validation")
    parent=PARENT.read_authority(pp,v["parent_authority_sha256"])
    for rel in (PRODUCER_NAMESPACE,VALIDATION_NAMESPACE):
        p=_raw(rel)
        if os.path.lexists(p): require(not stat.S_ISLNK(os.lstat(p).st_mode),f"formal output symlink {rel}")
    merged=dict(parent); merged.update({"execution_scope":v["execution_scope"],"resource_contract":v["resource_contract"],"output_contract":v["output_contract"],"eligible_as_e1_result":True,"formal_or_full594_execution_authorized":False,"target_rival_join_authorized":False,"postjoin_authorized":False,"scientific_reduction_authorized":False,"automatic_stage_advance":False,"automatic_submit_authorized":False,"scientific_GO_or_NO_GO":None,"next_authorized_stage":None}); return merged

__all__=["AUTHORITY_PATH","PRODUCER_NAMESPACE","VALIDATION_NAMESPACE","OVERLAY_PATHS","read_authority","validate_candidate","logical"]
