#!/usr/bin/env python3
"""Independent validator for literal accelerated-h100 smoke successor V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_independent_v2 as CORE  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_resource_authority_v3 as AUTH  # noqa: E402


AUTHORITY = ROOT / AUTH.AUTHORITY_PATH
PRODUCER_DIR = ROOT / AUTH.PRODUCER_NAMESPACE
PUBLIC_DIR = ROOT / AUTH.VALIDATION_NAMESPACE
ARTIFACT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v3_accelerated_h100_real_smoke_artifact_v3_20260826"
PRODUCER_STATUS = "DINO_RCDE_TRACK_R_V124_E1_V3_ACCELERATED_H100_REAL_SMOKE_V3_PASS_NONPROMOTABLE"
RESULT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v3_accelerated_h100_real_smoke_independent_validation_v3_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_V3_ACCELERATED_H100_REAL_SMOKE_V3_INDEPENDENT_VALIDATION_PASS_NONPROMOTABLE"


def _configure() -> None:
    CORE.AUTH = AUTH
    CORE.AUTHORITY = AUTHORITY
    CORE.PRODUCER_DIR = PRODUCER_DIR
    CORE.PUBLIC_DIR = PUBLIC_DIR
    CORE.ARTIFACT_SCHEMA = ARTIFACT_SCHEMA
    CORE.PRODUCER_STATUS = PRODUCER_STATUS
    CORE.RESULT_SCHEMA = RESULT_SCHEMA
    CORE.STATUS = STATUS


def run(
    *,
    authority_path: Path,
    authority_sha256: str,
    crash_after: str | None = None,
) -> Mapping[str, Any]:
    _configure()
    return CORE.run(
        authority_path=authority_path,
        authority_sha256=authority_sha256,
        crash_after=crash_after,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--crash-after")
    args = parser.parse_args()
    result = run(
        authority_path=args.authority,
        authority_sha256=args.authority_sha256,
        crash_after=args.crash_after,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
