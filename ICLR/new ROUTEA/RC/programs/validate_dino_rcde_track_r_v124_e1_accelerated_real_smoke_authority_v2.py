#!/usr/bin/env python3
"""Independent exact path-map review for real-smoke authority V2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_authority_v2 as S  # noqa: E402


OUTPUT = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2_20260826.json"
V1_REJECTION = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_20260826.terminal_rejection_v1.json"
PRODUCER = ROOT / S.PRODUCER_NAMESPACE
VALIDATION = ROOT / S.VALIDATION_NAMESPACE
FORMAL_PRODUCER = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v3_committed"
FORMAL_VALIDATION = ROOT / "results/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_validation_v3_committed"
STATUS = "DINO_RCDE_TRACK_R_V124_E1_ACCELERATED_REAL_SMOKE_AUTHORITY_V2_INDEPENDENT_REVIEW_PASS"


class SmokeAuthorityV2ReviewError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise SmokeAuthorityV2ReviewError(message)


def validate_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    S.validate_authority_envelope(value)
    rejection = json.loads(V1_REJECTION.read_text(encoding="ascii"))
    require(
        rejection.get("terminal_review_decision") == "FAIL"
        and rejection.get("authority_sha256")
        == "0c646a33b76b4dc6b0af6f927a182da58bb39616aca7115db66b3aa6db143ab7"
        and rejection.get("smoke_output_created") is False
        and rejection.get("submission_executed") is False
        and value["bindings"]["smoke_authority_v1_terminal_rejection"][
            "sha256"
        ]
        == S.BASE.V2.file_sha256(V1_REJECTION),
        "smoke V1 rejection lineage drift",
    )
    resolved = []
    for key, expected_path in S.OVERLAY_PATHS.items():
        row = value["bindings"][key]
        path = S._raw_regular_file(
            row["path"],
            expected_mode=value["binding_modes"][key],
            expected_lexical=expected_path,
        )
        require(
            row["path"] == expected_path
            and path.stat().st_size == row["bytes"]
            and S.BASE.V2.file_sha256(path) == row["sha256"],
            f"smoke V2 overlay physical/key path drift: {key}",
        )
        resolved.append(path)
    require(
        len({(path.stat().st_dev, path.stat().st_ino) for path in resolved})
        == len(S.OVERLAY_PATHS),
        "smoke V2 overlay duplicate/hardlink alias drift",
    )
    require(
        not os.path.lexists(OUTPUT)
        and not os.path.lexists(PRODUCER)
        and not os.path.lexists(VALIDATION)
        and not os.path.lexists(FORMAL_PRODUCER)
        and not os.path.lexists(FORMAL_VALIDATION),
        "smoke V2 authority/output appeared before review",
    )
    result = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2_independent_review_20260826",
        "status": STATUS,
        "candidate_authority_logical_sha256": value["logical_sha256"],
        "base_v3_binding_count": 115,
        "base_v3_import_closure_count": 65,
        "overlay_binding_count": len(S.OVERLAY_PATHS),
        "overlay_path_map_sha256": S.BASE.V2.canonical_sha256(
            dict(S.OVERLAY_PATHS)
        ),
        "overlay_import_closure_count": len(
            S.discover_overlay_import_closure()
        ),
        "overlay_import_closure_sha256": S.BASE.V2.canonical_sha256(
            list(S.discover_overlay_import_closure())
        ),
        "launcher_test_path_count": len(S.LAUNCHER_TEST_PATHS),
        "launcher_executable_path_count": len(
            S.LAUNCHER_EXECUTABLE_PATHS
        ),
        "eligible_as_e1_result": False,
        "submission_authorized": False,
        "smoke_executed": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "validation_pass": True,
    }
    result["logical_sha256"] = S.BASE.V2.logical_sha256(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="ascii"))
    print(json.dumps(validate_candidate(candidate), sort_keys=True))
