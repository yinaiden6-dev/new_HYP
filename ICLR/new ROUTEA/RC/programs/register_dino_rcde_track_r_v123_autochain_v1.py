#!/usr/bin/env python3
"""Freeze the scheduler-only V122-to-V123 continuation registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "results/dino_rcde_track_r_autochain_v1/v123_registration.json"
PARENT = "registry/current_authority_v122_20260821.json"
REPAIR_AUTHORITY = (
    "registry/dino_rcde_track_r_p_v2_targeted7_canonical_repair_authority_v10_20260824.json"
)
ADDENDUM = "plan/DINO_RCDE_TRACK_R_V123_AUTOMATIC_CONTINUATION_ADDENDUM_V1_20260824.md"
CONTRACT = "plan/DINO_RCDE_TRACK_R_V123_SCIENTIFIC_REDUCTION_CONTRACT_V1_20260822.md"
POST_P_CONTROLLER = "slurm/dino_rcde_track_r_post_p_utility_controller_v1.sbatch"
PROMOTION_CONTROLLER = "slurm/dino_rcde_track_r_v123_promotion_controller_v1.sbatch"
SCIENCE_LAUNCHER = "slurm/dino_rcde_track_r_science_v1.sbatch"
FREEZER = "programs/freeze_dino_rcde_track_r_science_authority_v123.py"
RUNTIME_VALIDATOR = "programs/validate_dino_rcde_track_r_v123_runtime_binding_v1.py"
REGISTRAR = "programs/register_dino_rcde_track_r_v123_autochain_v1.py"
REGISTRATION_VALIDATOR = (
    "programs/validate_dino_rcde_track_r_v123_autochain_registration_v1.py"
)
TEST = "tests/test_dino_rcde_track_r_v123_science_chain_v1.py"
REPAIR_JOB_ID = 5104783
COMPARATOR_JOB_ID = 5102487


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    req(isinstance(value, dict), f"JSON object absent: {relative}")
    return value


def build(*, promotion_job_id: int, science_job_id: int) -> dict[str, Any]:
    req(
        isinstance(promotion_job_id, int)
        and promotion_job_id > COMPARATOR_JOB_ID
        and isinstance(science_job_id, int)
        and science_job_id > promotion_job_id,
        "downstream job ID/order drift",
    )
    parent_path = ROOT / PARENT
    repair_path = ROOT / REPAIR_AUTHORITY
    parent = read(PARENT)
    repair = read(REPAIR_AUTHORITY)
    req(
        (parent_path.stat().st_mode & 0o777) == 0o444
        and parent.get("schema_version") == "rc_current_authority_v122_20260821"
        and parent.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARDS_AUTHORIZED"
        and parent.get("logical_sha256") == H.logical(parent),
        "V122 parent authority drift",
    )
    req(
        (repair_path.stat().st_mode & 0o777) == 0o444
        and repair.get("status") == "V122_TARGETED7_CANONICAL_REPAIR_AUTHORIZED"
        and repair.get("logical_sha256") == H.logical(repair)
        and repair.get("targeted_repair_shard_count") == 7
        and repair.get("reused_exact_shard_count") == 43
        and repair.get("bindings", {}).get("parent", {}).get("sha256")
        == H.file_sha(parent_path),
        "targeted-seven repair authority drift",
    )
    post_binding = parent.get("bindings", {}).get("post_p_controller", {})
    req(
        post_binding.get("sha256") == H.file_sha(ROOT / POST_P_CONTROLLER)
        and post_binding.get("bytes") == (ROOT / POST_P_CONTROLLER).stat().st_size,
        "submitted V122 controller binding drift",
    )
    for relative in (
        "results/dino_rcde_track_r_p_v2_utility_comparator_v1/result.json",
        "results/dino_rcde_track_r_p_v2_utility_comparator_validation_v1/result.json",
        "registry/current_authority_v123_20260822.json",
        "results/dino_rcde_track_r_i0_science_metadata_v1/result.json",
        "results/dino_rcde_track_r_i0_science_metadata_validation_v1/result.json",
        "results/dino_rcde_track_r_scientific_result_v1/result.json",
        "results/dino_rcde_track_r_scientific_validation_v1/result.json",
    ):
        req(not (ROOT / relative).exists(), f"future V122/V123 artifact already exists: {relative}")
    bindings = {
        "parent_v122_authority": H.bind(PARENT, with_logical=True, immutable=True),
        "targeted7_repair_authority": H.bind(
            REPAIR_AUTHORITY, with_logical=True, immutable=True
        ),
        "automatic_continuation_addendum": H.bind(ADDENDUM),
        "v123_contract": H.bind(CONTRACT),
        "post_p_controller": H.bind(POST_P_CONTROLLER),
        "promotion_controller": H.bind(PROMOTION_CONTROLLER),
        "science_launcher": H.bind(SCIENCE_LAUNCHER),
        "v123_freezer": H.bind(FREEZER),
        "v123_runtime_validator": H.bind(RUNTIME_VALIDATOR),
        "registrar": H.bind(REGISTRAR),
        "registration_validator": H.bind(REGISTRATION_VALIDATOR),
        "v123_test": H.bind(TEST),
    }
    value: dict[str, Any] = {
        "schema_version": "rc_dino_rcde_track_r_v123_autochain_registration_v1_20260824",
        "status": "V123_AUTOMATIC_CONTINUATION_REGISTERED",
        "claim_level": "ENGINEERING_SCHEDULER_CONTINUATION_ONLY",
        "repair_job_id": REPAIR_JOB_ID,
        "comparator_job_id": COMPARATOR_JOB_ID,
        "promotion_job_id": promotion_job_id,
        "science_job_id": science_job_id,
        "promotion_dependency": f"afterok:{COMPARATOR_JOB_ID}",
        "science_dependency": f"afterok:{promotion_job_id}",
        "automatic_stage_advance_to_v123": True,
        "automatic_stage_advance_after_v123": False,
        "model_or_science_execution_count": 0,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": "TRACK_R_C128_CONDITIONAL_SCIENTIFIC_REDUCTION_IF_PROMOTION_PASS",
        "bindings": bindings,
    }
    value["logical_sha256"] = H.logical(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion-job-id", type=int, required=True)
    parser.add_argument("--science-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    req(
        args.output.resolve(strict=False) == (ROOT / OUTPUT).resolve(strict=False),
        "registration output drift",
    )
    value = build(
        promotion_job_id=args.promotion_job_id, science_job_id=args.science_job_id
    )
    if not args.dry_run:
        H.atomic(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "promotion_job_id": value["promotion_job_id"],
                "science_job_id": value["science_job_id"],
                "logical_sha256": value["logical_sha256"],
                "dry_run": bool(args.dry_run),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
