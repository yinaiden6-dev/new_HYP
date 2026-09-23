#!/usr/bin/env python3
"""Seal the terminal independent-review rejection of V124-E1 review V1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v1_20260826.json"
OUTPUT = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v1_20260826.terminal_rejection_v1.json"
SOURCE_SHA256 = "0476e159c039d5bfb685e392984bc61d5607ce09a68ebf7df931b186d8e1f4e7"


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
            raise RuntimeError("existing V1 rejection receipt drift")
        return value
    if sha(SOURCE) != SOURCE_SHA256 or SOURCE.stat().st_mode & 0o777 != 0o444:
        raise RuntimeError("review V1 authority physical binding drift")
    value = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_review_v1_terminal_rejection_v1_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_REVIEW_V1_TERMINAL_FAIL_REJECTED",
        "source_review_authority_path": SOURCE.relative_to(ROOT).as_posix(),
        "source_review_authority_sha256": SOURCE_SHA256,
        "source_review_status": "DINO_RCDE_TRACK_R_V124_E1_REVIEW_CANDIDATE_FROZEN_NOT_EXECUTION_AUTHORITY",
        "terminal_review_decision": "FAIL",
        "execution_authorized": False,
        "smoke_authorized": False,
        "submission_authorized": False,
        "superseded_for_execution": True,
        "preserve_source_bytes": True,
        "reasons": [
            "C_COL_FULL_128_256_DEEP_VALIDATION_ABSENT",
            "PUBLICATION_NOT_ATOMIC_ACROSS_COMPLETE_RESULT_FAMILY",
            "AUTHORITY_BINDING_WHITELIST_AND_IMPORT_CLOSURE_NOT_EXACT",
            "WRONG_FOLD_CHECKPOINT_AND_CRASH_BOUNDARY_POISONS_INCOMPLETE",
        ],
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical(value)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="ascii")
    temporary.chmod(0o444)
    os.link(temporary, OUTPUT)
    temporary.unlink()
    return value


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
