#!/usr/bin/env python3
"""Append-only exact-path-map real-smoke V2 producer entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_dino_rcde_track_r_v124_e1_accelerated_real_smoke_v1 as CORE  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_authority_v2 as AUTH  # noqa: E402


AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2_20260826.json"
PUBLIC_DIR = ROOT / AUTH.PRODUCER_NAMESPACE
ARTIFACT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_artifact_v2_20260826"
RESULT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_result_v2_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_V3_ACCELERATED_REAL_SMOKE_V2_PASS_NONPROMOTABLE"


def _configure() -> None:
    CORE.AUTH = AUTH
    CORE.AUTHORITY = AUTHORITY
    CORE.PUBLIC_DIR = PUBLIC_DIR
    CORE.ARTIFACT_SCHEMA = ARTIFACT_SCHEMA
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
