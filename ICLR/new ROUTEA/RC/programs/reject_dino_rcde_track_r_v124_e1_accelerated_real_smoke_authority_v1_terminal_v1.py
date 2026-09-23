#!/usr/bin/env python3
"""Seal terminal-review rejection of the frozen smoke-authority V1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_20260826.json"
OUTPUT = ROOT / "registry/dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_20260826.terminal_rejection_v1.json"
AUTHORITY_SHA256 = "0c646a33b76b4dc6b0af6f927a182da58bb39616aca7115db66b3aa6db143ab7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "logical_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        value = json.loads(OUTPUT.read_text(encoding="ascii"))
        if value.get("logical_sha256") != logical(value):
            raise RuntimeError("existing smoke V1 rejection drift")
        return value
    authority = json.loads(AUTHORITY.read_text(encoding="ascii"))
    if sha(AUTHORITY) != AUTHORITY_SHA256 or AUTHORITY.stat().st_mode & 0o777 != 0o444:
        raise RuntimeError("frozen smoke V1 authority drift")
    overlay = {
        key: authority["bindings"][key]
        for key in (
            "smoke_producer",
            "smoke_independent_validator",
            "smoke_launcher",
            "smoke_authority_runtime",
            "smoke_tests",
        )
    }
    value = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v1_terminal_rejection_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_ACCELERATED_REAL_SMOKE_AUTHORITY_V1_TERMINAL_FAIL_REJECTED",
        "terminal_review_decision": "FAIL",
        "authority_path": AUTHORITY.relative_to(ROOT).as_posix(),
        "authority_sha256": AUTHORITY_SHA256,
        "authority_logical_sha256": authority["logical_sha256"],
        "rejected_overlay_bindings": overlay,
        "reasons": [
            "OVERLAY_KEYS_NOT_BOUND_TO_EXACT_CANONICAL_PATH_MAP",
            "LAUNCHER_EXECUTABLE_AND_TEST_PATH_UNIQUENESS_NOT_CLOSED",
            "DUPLICATE_ALIAS_OMISSION_SWAP_POISONS_ABSENT",
        ],
        "preserve_authority_bytes": True,
        "smoke_output_created": False,
        "submission_executed": False,
        "execution_authorized": False,
        "submission_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical(value)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="ascii"
    )
    temporary.chmod(0o444)
    os.link(temporary, OUTPUT)
    temporary.unlink()
    return value


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
