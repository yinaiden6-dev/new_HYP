#!/usr/bin/env python3
"""Freeze compact V124 E1 dev staged smoke V4 authority."""

from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from rc_aslo_xf import dino_rcde_track_r_v124_e1_dev_staged_authority_v4 as A  # noqa:E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_authority_v2 as P  # noqa:E402

OUT=ROOT/A.AUTHORITY_PATH

def bind(relative: str) -> dict[str,object]:
    path=ROOT/relative
    if not path.is_file() or path.is_symlink(): raise RuntimeError(f"binding absent/symlink: {relative}")
    return {"path":relative,"bytes":path.stat().st_size,"sha256":P.BASE.V2.file_sha256(path)}

def build() -> dict[str,object]:
    parent=ROOT/A.PARENT_AUTHORITY_PATH
    value={
      "schema_version":"rc_dino_rcde_track_r_v124_e1_dev_staged_authority_v4_20260826",
      "status":"DINO_RCDE_TRACK_R_V124_E1_DEV_STAGED_SMOKE_V4_EXECUTION_AUTHORIZED",
      "claim_level":"DEV_ACCELERATED_STAGED_REAL_SMOKE_ENGINEERING_ONLY_NONPROMOTABLE",
      "parent_authority_path":A.PARENT_AUTHORITY_PATH,
      "parent_authority_sha256":P.BASE.V2.file_sha256(parent),
      "execution_scope":"EXECUTION0_DEV_ACCELERATED_TWO_STAGE_REAL_SMOKE_V4_ONLY_NONPROMOTABLE",
      "resource_contract":{"partition":"dev_accelerated","allowed_partitions":["dev_accelerated"],"gpu_count":1,"cpus_per_task":8,"memory_megabytes":128000,"walltime_seconds_per_stage":3600},
      "stage_sequence":[{"stage":0,"role":"PRODUCER","dependency":None},{"stage":1,"role":"INDEPENDENT_VALIDATOR","dependency":"afterok:stage0"}],
      "output_contract":{"producer_family":A.PRODUCER_NAMESPACE,"validation_family":A.VALIDATION_NAMESPACE,"append_only":True,"atomic_family_required":True,"producer_commit_is_stage_checkpoint":True,"exact_committed_reuse":True,"partial_family_consumable":False,"eligible_as_e1_result":False},
      "bindings":{key:bind(path) for key,path in A.OVERLAY_PATHS.items()},
      "binding_modes":{key:(ROOT/path).stat().st_mode & 0o777 for key,path in A.OVERLAY_PATHS.items()},
      "scientific_GO_or_NO_GO":None,"eligible_as_e1_result":False,"formal_or_full594_execution_authorized":False,"target_rival_join_authorized":False,"postjoin_authorized":False,"scientific_reduction_authorized":False,"automatic_stage_advance":False,"automatic_submit_authorized":False,"next_authorized_stage":None,
    }
    value["logical_sha256"]=A.logical(value); A.validate_candidate(value); return value

def main() -> None:
    if os.path.lexists(OUT): raise RuntimeError(f"authority exists: {OUT}")
    value=build(); OUT.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="ascii"); OUT.chmod(0o444); print(json.dumps({"authority":str(OUT),"sha256":P.BASE.V2.file_sha256(OUT),"status":value["status"]},sort_keys=True))

if __name__=="__main__": main()

