#!/usr/bin/env python3
"""Formal V124 E1 V3 producer under dev staged resource authority V5."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any,Mapping
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"programs"))
import run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2 as CORE  # noqa:E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_atomic_family_v3 as ATOMIC  # noqa:E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_formal_dev_authority_v5 as AUTH  # noqa:E402
AUTHORITY=ROOT/AUTH.AUTHORITY_PATH;PUBLIC_DIR=ROOT/AUTH.PRODUCER_NAMESPACE
ARTIFACT_SCHEMA="rc_dino_rcde_track_r_v124_e1_v3_prejoin_artifact_20260826";RESULT_SCHEMA="rc_dino_rcde_track_r_v124_e1_v3_producer_result_20260826";STATUS="DINO_RCDE_TRACK_R_V124_E1_V3_FIXED_NATURAL_OPENED_CANARY_PREJOIN_PASS"
def _validate(public_dir:Path,*,authority_sha256:str,family_role:str,producer:bool):
    del family_role;return ATOMIC.validate_family(public_dir,authority_sha256=authority_sha256,family_role="E1_V3_PRODUCER_FAMILY" if producer else "E1_V3_VALIDATION_FAMILY",producer=producer)
def run(*,authority_path:Path,authority_sha256:str,crash_after:str|None=None)->Mapping[str,Any]:
    CORE.AUTHORITY=AUTHORITY;CORE.PUBLIC_DIR=PUBLIC_DIR;CORE.ARTIFACT_SCHEMA=ARTIFACT_SCHEMA;CORE.RESULT_SCHEMA=RESULT_SCHEMA;CORE.STATUS=STATUS;CORE.read_authority=AUTH.read_authority;CORE.publish_producer_family=ATOMIC.publish_producer_family;CORE.validate_family=_validate
    return CORE.run(authority_path=authority_path,authority_sha256=authority_sha256,crash_after=crash_after)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--authority",type=Path,default=AUTHORITY);p.add_argument("--authority-sha256",required=True);p.add_argument("--crash-after");a=p.parse_args();r=run(authority_path=a.authority,authority_sha256=a.authority_sha256,crash_after=a.crash_after);print(json.dumps({"status":r["status"]},sort_keys=True))

