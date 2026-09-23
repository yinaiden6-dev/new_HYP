#!/usr/bin/env python3
"""Freeze target-free authority for the V124 corrected-core E0 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = "registry/dino_rcde_track_r_v124_core_e0_authority_v1_20260824.json"
STATUS = "DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_AUTHORIZED"
SCHEMA = "rc_dino_rcde_track_r_v124_core_e0_authority_v1_20260824"


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    req(isinstance(value, dict), f"JSON object absent: {relative}")
    return value


def build() -> dict[str, Any]:
    parent_v121 = "registry/current_authority_v121_20260821.json"
    parent_v123 = "registry/current_authority_v123_20260822.json"
    v123_result = "results/dino_rcde_track_r_scientific_result_v1/result.json"
    v123_validation = "results/dino_rcde_track_r_scientific_validation_v1/result.json"
    p_aggregate = "results/dino_rcde_sr0_mt_p_v2_formal_lock_aggregate_v1/result.json"
    pair_manifest = "results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json"
    query0_artifact = "results/dino_rcde_sr0_mt_p_v2_formal_full_c128_locks_v1/query_000/locks.pt"
    v121, v123 = read(parent_v121), read(parent_v123)
    result, validation = read(v123_result), read(v123_validation)
    aggregate, pair = read(p_aggregate), read(pair_manifest)
    req(
        (ROOT / parent_v121).stat().st_mode & 0o777 == 0o444
        and v121.get("status") == "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED"
        and v121.get("logical_sha256") == H.logical(v121),
        "V121 parent drift",
    )
    req(
        (ROOT / parent_v123).stat().st_mode & 0o777 == 0o444
        and v123.get("status") == "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_AUTHORIZED"
        and v123.get("logical_sha256") == H.logical(v123)
        and result.get("status") == "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_COMPLETE"
        and validation.get("status") == "DINO_RCDE_TRACK_R_SCIENTIFIC_INDEPENDENT_VALIDATION_PASS"
        and validation.get("validation_pass") is True,
        "V123 forensic parent drift",
    )
    req(
        aggregate.get("status") == "RCDE_SR0_MT_P_V2_FORMAL_LOCK_AGGREGATE_READY"
        and aggregate.get("query_count") == 594
        and aggregate.get("logical_sha256") == H.logical(aggregate)
        and pair.get("status") == "RCDE_SR0_MT_ROLE_FREE_PAIR_ADDRESS_READY"
        and pair.get("logical_sha256") == H.logical(pair),
        "target-free P/pair source drift",
    )
    row0 = next(
        (row for row in aggregate.get("rows", []) if row.get("execution_ordinal") == 0),
        None,
    )
    req(
        isinstance(row0, dict)
        and row0.get("lock_artifact_path") == query0_artifact
        and row0.get("lock_artifact_sha256") == H.file_sha(ROOT / query0_artifact),
        "execution0 C128 artifact drift",
    )
    bindings = {
        "draft_plan": H.bind("plan/DINO_RCDE_TRACK_R_V124_CORRECTED_MECHANISM_AUTHORITY_PLAN_DRAFT_V1_20260824.md"),
        "contract": H.bind("plan/DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_CONTRACT_V1_20260824.md"),
        "parent_v121": H.bind(parent_v121, with_logical=True, immutable=True),
        "parent_v123": H.bind(parent_v123, with_logical=True, immutable=True),
        "v123_diagnostic_result": H.bind(v123_result, with_logical=True, immutable=True),
        "v123_diagnostic_validation": H.bind(v123_validation, with_logical=True, immutable=True),
        "p_lock_aggregate": H.bind(p_aggregate, with_logical=True, immutable=True),
        "role_free_pair_manifest": H.bind(pair_manifest, with_logical=True, immutable=True),
        "execution0_lock_artifact": H.bind(query0_artifact, immutable=True),
        "redacted_schedule": H.bind(v121["bindings"]["redacted_schedule"]["path"], with_logical=True),
        "redacted_cache_index": H.bind(v121["bindings"]["redacted_cache_index"]["path"], with_logical=True),
        "geometry_payload": H.bind(v121["bindings"]["geometry_payload"]["path"]),
        "runtime_v2": H.bind("src/rc_aslo_xf/dino_rcde_sr0_mt_v_runtime_v2.py"),
        "runtime_v1": H.bind("src/rc_aslo_xf/dino_rcde_sr0_mt_v_runtime_v1.py"),
        "controls_v1": H.bind("src/rc_aslo_xf/dino_rcde_sr0_mt_controls_v1.py"),
        "vdecode_v1": H.bind("src/rc_aslo_xf/dino_rcde_cw1_multitile_vdecode_v1.py"),
        "full_c128_control": H.bind("src/rc_aslo_xf/dino_rcde_track_r_full_c128_c_dino_v1.py"),
        "independent_c128_replay": H.bind("programs/replay_dino_rcde_track_r_full_c128_c_dino_receipt_v1.py"),
        "p_v2_adapter": H.bind("src/rc_aslo_xf/dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1.py"),
        "v_input": H.bind("programs/dino_rcde_sr0_mt_v_input_common_v1.py"),
        "track_input": H.bind("programs/dino_rcde_track_r_input_common_v1.py"),
        "identity_repair": H.bind("src/rc_aslo_xf/gallery_identity_repair.py"),
        "rootwise_test": H.bind("tests/test_dino_rcde_sr0_mt_v_runtime_v2.py"),
        "c128_test": H.bind("tests/test_dino_rcde_track_r_full_c128_c_dino_v1.py"),
        "reducer": H.bind("programs/reduce_dino_rcde_track_r_v124_core_e0_v1.py"),
        "validator": H.bind("programs/validate_dino_rcde_track_r_v124_core_e0_v1.py"),
        "launcher": H.bind("slurm/dino_rcde_track_r_v124_core_e0_v1.sbatch"),
        "freezer": H.bind("programs/freeze_dino_rcde_track_r_v124_core_e0_authority_v1.py"),
    }
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "stage": "V124_CORE_REPAIR_PRE_E0",
        "claim_level": "ENGINEERING_CORE_REPAIR_ONLY",
        "v123_forensic_disposition": "V123_PROTOCOL_ABORT_PENDING_CONTROL_AND_ARM_REPAIR",
        "opened594_role": "DIAGNOSTIC_ONLY_NOT_CONFIRMATION",
        "full_reference_rootwise_test_authorized": True,
        "full_c128_c_dino_test_authorized": True,
        "natural_execution0_target_free_test_authorized": True,
        "c_col_p_authorized": False,
        "model_load_authorized": False,
        "model_forward_authorized": False,
        "training_authorized": False,
        "target_rival_join_authorized": False,
        "scientific_reduction_authorized": False,
        "opened_sealed_access_authorized": False,
        "h0_access_authorized": False,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "resource_contract": {
            "partition": "cpuonly",
            "cpus_per_task": 2,
            "memory_megabytes": 32768,
            "walltime_seconds": 1800,
        },
        "bindings": bindings,
    }
    value["logical_sha256"] = H.logical(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTHORITY)
    args = parser.parse_args()
    req(args.output.resolve(strict=False) == (ROOT / AUTHORITY).resolve(strict=False), "authority output drift")
    value = build()
    H.atomic(args.output, value)
    print(json.dumps({"status": value["status"], "binding_count": len(value["bindings"]), "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
