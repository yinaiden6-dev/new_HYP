#!/usr/bin/env python3
"""V124 E1 non-promotable dev staged independent validator V4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"programs"))
import validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_independent_v1 as CORE  # noqa:E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_dev_staged_authority_v4 as AUTH  # noqa:E402

AUTHORITY=ROOT/AUTH.AUTHORITY_PATH
PRODUCER_DIR=ROOT/AUTH.PRODUCER_NAMESPACE
PUBLIC_DIR=ROOT/AUTH.VALIDATION_NAMESPACE
ARTIFACT_SCHEMA="rc_dino_rcde_track_r_v124_e1_v3_dev_staged_smoke_artifact_v4_20260826"
PRODUCER_STATUS="DINO_RCDE_TRACK_R_V124_E1_DEV_STAGED_SMOKE_V4_PRODUCER_PASS_NONPROMOTABLE"
RESULT_SCHEMA="rc_dino_rcde_track_r_v124_e1_v3_dev_staged_smoke_independent_validation_v4_20260826"
STATUS="DINO_RCDE_TRACK_R_V124_E1_DEV_STAGED_SMOKE_V4_INDEPENDENT_VALIDATION_PASS_NONPROMOTABLE"

def _configure() -> None:
    CORE.AUTH=AUTH; CORE.AUTHORITY=AUTHORITY; CORE.PRODUCER_DIR=PRODUCER_DIR; CORE.PUBLIC_DIR=PUBLIC_DIR
    CORE.ARTIFACT_SCHEMA=ARTIFACT_SCHEMA; CORE.PRODUCER_STATUS=PRODUCER_STATUS; CORE.RESULT_SCHEMA=RESULT_SCHEMA; CORE.STATUS=STATUS
    CORE.CLAIM="DEV_ACCELERATED_STAGED_REAL_SMOKE_ENGINEERING_ONLY_NONPROMOTABLE"

def run(*, authority_path: Path, authority_sha256: str, crash_after: str | None=None) -> Mapping[str, Any]:
    _configure(); result=CORE.run(authority_path=authority_path,authority_sha256=authority_sha256,crash_after=crash_after)
    if result.get("real_smoke") is not True or result.get("eligible_as_e1_result") is not False:
        raise RuntimeError("V4 validator promotion boundary drift")
    return result

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--authority",type=Path,default=AUTHORITY); p.add_argument("--authority-sha256",required=True); p.add_argument("--crash-after"); a=p.parse_args()
    out=run(authority_path=a.authority,authority_sha256=a.authority_sha256,crash_after=a.crash_after); print(json.dumps({"status":out["status"]},sort_keys=True))
