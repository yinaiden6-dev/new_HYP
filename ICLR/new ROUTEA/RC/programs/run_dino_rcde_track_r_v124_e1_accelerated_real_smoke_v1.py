#!/usr/bin/env python3
"""Non-promotable accelerated real-smoke overlay for reviewed E1 V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2 as CORE  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_atomic_family_v3 as ATOMIC  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_authority_v1 as AUTH  # noqa: E402


AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_20260826.json"
PUBLIC_DIR = ROOT / AUTH.PRODUCER_NAMESPACE
ARTIFACT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_artifact_v1_20260826"
RESULT_SCHEMA = "rc_dino_rcde_track_r_v124_e1_v3_accelerated_real_smoke_result_v1_20260826"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_V3_ACCELERATED_REAL_SMOKE_PASS_NONPROMOTABLE"
CLAIM = "ACCELERATED_REAL_SMOKE_ENGINEERING_ONLY_NONPROMOTABLE"


def _validate_adapter(
    public_dir: Path,
    *,
    authority_sha256: str,
    family_role: str,
    producer: bool,
):
    del family_role
    artifact, result = ATOMIC.validate_family(
        public_dir,
        authority_sha256=authority_sha256,
        family_role=(
            "E1_V3_PRODUCER_FAMILY"
            if producer
            else "E1_V3_VALIDATION_FAMILY"
        ),
        producer=producer,
    )
    if producer and (
        artifact is None
        or artifact.get("real_smoke") is not True
        or artifact.get("eligible_as_e1_result") is not False
        or result.get("real_smoke") is not True
        or result.get("eligible_as_e1_result") is not False
    ):
        raise RuntimeError("committed smoke family promotion boundary drift")
    return artifact, result


def _publish_adapter(
    public_dir: Path,
    *,
    authority_sha256: str,
    artifact: Mapping[str, Any],
    result_builder: Callable[[Path], Mapping[str, Any]],
    crash_after: str | None = None,
):
    smoke_artifact = dict(artifact)
    smoke_artifact.update(
        {
            "claim_level": CLAIM,
            "real_smoke": True,
            "eligible_as_e1_result": False,
            "eligible_for_scientific_or_diagnostic_reduction": False,
        }
    )
    smoke_artifact["logical_sha256"] = CORE.V1.canonical_sha256(
        {
            key: item
            for key, item in smoke_artifact.items()
            if key not in {"logical_sha256", "tensor_payloads"}
        }
    )

    def smoke_result_builder(path: Path) -> Mapping[str, Any]:
        result = dict(result_builder(path))
        result.update(
            {
                "claim_level": CLAIM,
                "real_smoke": True,
                "eligible_as_e1_result": False,
                "eligible_for_scientific_or_diagnostic_reduction": False,
            }
        )
        result["logical_sha256"] = ATOMIC.V2.logical_sha256(result)
        return result

    return ATOMIC.publish_producer_family(
        public_dir,
        authority_sha256=authority_sha256,
        artifact=smoke_artifact,
        result_builder=smoke_result_builder,
        crash_after=crash_after,
    )


def _configure() -> None:
    CORE.AUTHORITY = AUTHORITY
    CORE.PUBLIC_DIR = PUBLIC_DIR
    CORE.ARTIFACT_SCHEMA = ARTIFACT_SCHEMA
    CORE.RESULT_SCHEMA = RESULT_SCHEMA
    CORE.STATUS = STATUS
    CORE.read_authority = AUTH.read_authority
    CORE.publish_producer_family = _publish_adapter
    CORE.validate_family = _validate_adapter


def run(
    *,
    authority_path: Path,
    authority_sha256: str,
    crash_after: str | None = None,
) -> Mapping[str, Any]:
    _configure()
    result = CORE.run(
        authority_path=authority_path,
        authority_sha256=authority_sha256,
        crash_after=crash_after,
    )
    if (
        result.get("real_smoke") is not True
        or result.get("eligible_as_e1_result") is not False
    ):
        raise RuntimeError("real-smoke result promotion boundary drift")
    return result


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
