#!/usr/bin/env python3
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from rc_aslo_xf import dino_rcde_track_r_v124_e1_formal_dev_authority_v5 as A  # noqa:E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_authority_v3 as P  # noqa:E402
OUT=ROOT/A.AUTHORITY_PATH
def bind(r:str)->dict:
 p=ROOT/r
 if not p.is_file() or p.is_symlink():raise RuntimeError(f"absent {r}")
 return {"path":r,"bytes":p.stat().st_size,"sha256":P.V2.file_sha256(p)}
def build()->dict:
 v={"schema_version":"rc_dino_rcde_track_r_v124_e1_formal_dev_authority_v5_20260826","status":"DINO_RCDE_TRACK_R_V124_E1_FORMAL_DEV_STAGED_V5_EXECUTION_AUTHORIZED","claim_level":"FORMAL_E1_TARGET_FREE_PREJOIN_ENGINEERING_ONLY","parent_authority_path":A.PARENT_AUTHORITY_PATH,"parent_authority_sha256":P.V2.file_sha256(ROOT/A.PARENT_AUTHORITY_PATH),"parent_validation_path":A.PARENT_VALIDATION_PATH,"parent_validation_sha256":P.V2.file_sha256(ROOT/A.PARENT_VALIDATION_PATH),"execution_scope":"EXECUTION0_FORMAL_E1_DEV_ACCELERATED_TWO_STAGE_PREJOIN_V5","resource_contract":{"partition":"dev_accelerated","allowed_partitions":["dev_accelerated"],"gpu_count":1,"cpus_per_task":8,"memory_megabytes":128000,"walltime_seconds_per_stage":3600},"stage_sequence":[{"stage":0,"role":"FORMAL_PRODUCER","dependency":None},{"stage":1,"role":"FORMAL_INDEPENDENT_VALIDATOR","dependency":"afterok:stage0"}],"output_contract":{"producer_family":A.PRODUCER_NAMESPACE,"validation_family":A.VALIDATION_NAMESPACE,"append_only":True,"atomic_family_required":True,"producer_commit_is_stage_checkpoint":True,"exact_committed_reuse":True,"partial_family_consumable":False,"eligible_as_e1_result":True},"bindings":{k:bind(p) for k,p in A.OVERLAY_PATHS.items()},"binding_modes":{k:(ROOT/p).stat().st_mode&0o777 for k,p in A.OVERLAY_PATHS.items()},"eligible_as_e1_result":True,"formal_or_full594_execution_authorized":False,"target_rival_join_authorized":False,"postjoin_authorized":False,"scientific_reduction_authorized":False,"automatic_stage_advance":False,"automatic_submit_authorized":False,"scientific_GO_or_NO_GO":None,"next_authorized_stage":None};v["logical_sha256"]=A.logical(v);A.validate_candidate(v);return v
def main():
 if os.path.lexists(OUT):raise RuntimeError(f"exists {OUT}")
 v=build();OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n",encoding="ascii");OUT.chmod(0o444);print(json.dumps({"status":v["status"],"authority":str(OUT),"sha256":P.V2.file_sha256(OUT)},sort_keys=True))
if __name__=="__main__":main()

