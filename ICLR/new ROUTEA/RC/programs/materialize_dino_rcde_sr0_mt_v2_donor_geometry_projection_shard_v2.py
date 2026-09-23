#!/usr/bin/env python3
"""V113 role-validated wrapper for the frozen V112 projection core."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import materialize_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v1 as Core


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v113_20260820.json"
AUTH_SCHEMA = "rc_current_authority_v113_20260820"
AUTH_STATUS = "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_RESCOPE_REPAIR_AUTHORIZED"
OUTROOT = Core.OUTROOT
SCHEMA = "rc_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v2_20260820"
STATUS = "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_READY"


def latest_authority(path: Path) -> tuple[dict[str, Any], str]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = [
        (int(match.group(1)), item.resolve())
        for item in (ROOT / "registry").glob("current_authority_v*_*.json")
        if (match := pattern.fullmatch(item.name))
        and item.is_file()
        and not item.is_symlink()
    ]
    exact = (ROOT / AUTH).resolve(strict=True)
    Core.require(
        rows
        and max(version for version, _ in rows) == 113
        and [item for version, item in rows if version == 113] == [exact]
        and path.resolve(strict=True) == exact,
        "V113 is not unique latest authority",
    )
    value = Core.read_json(exact, "V113 authority", immutable=True)
    Core.require(
        value.get("schema_version") == AUTH_SCHEMA
        and value.get("status") == AUTH_STATUS
        and value.get("logical_sha256") == Core.logical(value),
        "V113 authority drift",
    )
    return value, Core.file_sha256(exact)


def validate_role_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    expected_true = {
        "bound_metadata_read_authorized",
        "lock_artifact_hash_stream_authorized",
        "lock_payload_deserialization_authorized",
        "natural_lock_validation_authorized",
        "p_lock_consumption_for_donor_geometry_projection_authorized",
        "sealed_v2_lock_geometry_projection_read_authorized",
        "target_member_field_read_authorized",
        "dev_successor_submit_authorized",
    }
    Core.require(
        {key for key, value in authority.items() if value is True} == expected_true
        and authority.get("consumable_lock_authorized") is False
        and authority.get("final_donor_matching_authorized") is False
        and authority.get("training_authorized") is False
        and authority.get("heldout_scoring_authorized") is False
        and authority.get("protected_access_authorized") is False,
        "V113 exact permission scope drift",
    )
    role_manifest = Core.bound_path(authority, "loss_role_manifest", "loss role manifest")
    validation = Core.bound_json(
        authority,
        "role_free_pair_address_validation",
        "role-free pair-address validation",
        immutable=True,
    )
    Core.require(
        validation.get("schema_version")
        == "rc_dino_rcde_sr0_mt_role_free_pair_address_validation_v1_20260817"
        and validation.get("status")
        == "RCDE_SR0_MT_ROLE_FREE_PAIR_ADDRESS_INDEPENDENT_VALIDATION_PASS"
        and validation.get("validation_pass") is True
        and validation.get("pair_query_count") == 594
        and validation.get("member_count") == 1188
        and validation.get("loss_role_manifest_sha256") == Core.file_sha256(role_manifest),
        "independent exact-target role validation drift",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--shard-ordinal", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTROOT)
    args = parser.parse_args()
    Core.require(0 <= args.shard_ordinal < 50, "shard ordinal drift")
    Core.require(
        args.output_root.resolve(strict=False) == (ROOT / OUTROOT).resolve(strict=False),
        "output root drift",
    )
    authority, authority_sha = latest_authority(args.authority)
    role_validation = validate_role_authority(authority)
    for key, name in (
        ("contract", "donor contract"),
        ("repair_addendum", "V113 repair addendum"),
        ("projection_core_producer", "V112 projection core producer"),
        ("projection_core_validator", "V112 projection core validator"),
        ("projection_producer", "V113 projection producer"),
        ("projection_validator", "V113 projection validator"),
        ("validation_receipt_checker", "V113 receipt checker"),
        ("dev_continuation_helper", "V113 dev continuation helper"),
        ("lock_core", "lock core"),
        ("adapter", "natural adapter"),
        ("selector", "P selector"),
        ("structure_core", "CW1 structure core"),
        ("fanout", "fanout"),
        ("runtime_core", "P runtime core"),
        ("dev_launcher", "dev launcher"),
        ("cpu_launcher", "cpu launcher"),
    ):
        Core.bound_path(authority, key, name)
    strict_aggregate = Core.bound_json(
        authority, "strict_payload_aggregate", "V111 aggregate", immutable=True
    )
    strict_validation = Core.bound_json(
        authority,
        "strict_payload_aggregate_validation",
        "V111 validation",
        immutable=True,
    )
    lock_aggregate = Core.bound_json(
        authority, "lock_aggregate", "V109 lock aggregate", immutable=True
    )
    loss_roles = Core.bound_json(
        authority, "loss_role_manifest", "loss role manifest", immutable=True
    )
    query_manifest = Core.bound_json(
        authority, "v87_query_manifest", "V87 query manifest", immutable=True
    )
    Core.require(
        strict_aggregate.get("status")
        == "RCDE_SR0_MT_P_V2_STRICT_PAYLOAD_AGGREGATE_READY"
        and strict_validation.get("status")
        == "RCDE_SR0_MT_P_V2_STRICT_PAYLOAD_AGGREGATE_INDEPENDENT_VALIDATION_PASS"
        and strict_validation.get("validation_pass") is True
        and strict_validation.get("aggregate_sha256")
        == authority["bindings"]["strict_payload_aggregate"]["sha256"]
        and lock_aggregate.get("query_count") == 594
        and loss_roles.get("status") == "RCDE_SR0_MT_LOSS_ROLE_MANIFEST_QUARANTINED"
        and query_manifest.get("status")
        == "RCDE_SR0_MT_E0_FULL594_STRUCTURE_SIDECAR_MANIFEST_READY_STRUCTURAL_ONLY",
        "V113 prerequisite drift",
    )
    start, stop = args.shard_ordinal * 12, args.shard_ordinal * 12 + 12
    source_rows = sorted(
        [
            item
            for item in lock_aggregate["rows"]
            if start <= int(item["execution_ordinal"]) < stop
        ],
        key=lambda item: int(item["execution_ordinal"]),
    )
    Core.require(10 <= len(source_rows) <= 12, "V113 shard population drift")
    projections = [
        Core.project_query(item, loss_roles, query_manifest) for item in source_rows
    ]
    projected = sum(
        item["status"] == "EXACT_CLEAN_TARGET_LOCK_PAIR_PROJECTED"
        for item in projections
    )
    misses = sum(item["status"] == Core.TARGET_MISS for item in projections)
    output = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "ENGINEERING_ROLE_VALIDATED_EXACT_CLEAN_TARGET_LOCK_DONOR_GEOMETRY_PROJECTION_ONLY",
        "authority_sha256": authority_sha,
        "authority_logical_sha256": authority["logical_sha256"],
        "role_validation_sha256": authority["bindings"][
            "role_free_pair_address_validation"
        ]["sha256"],
        "role_validation_logical_sha256": role_validation["logical_sha256"],
        "shard_ordinal": args.shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "query_count": len(projections),
        "projected_query_count": projected,
        "target_clean_not_in_natural_c128_count": misses,
        "scope_projection_count": projected * 4,
        "direction_record_decode_count": projected * 8,
        "query_projections": projections,
        "query_projection_population_sha256": Core.canonical(projections),
        "loss_role_manifest_joined_record_count": len(projections),
        "target_member_field_read_count": len(projections),
        "independent_target_role_validation_read_count": 1,
        "rival_member_used_as_donor_count": 0,
        "fixed_target_candidate_lookup_count": len(projections),
        "fixed_target_record_scan_count": len(projections) * 1024,
        "candidate_scan_for_donor_coverage_count": 0,
        "target_insertion_count": 0,
        "lock_payload_deserialization_count": len(projections),
        "model_load_count": 0,
        "model_forward_count": 0,
        "training_count": 0,
        "score_rank_winner_gap_outcome_read_count": 0,
        "identity_supergroup_read_count": 0,
        "opened_sealed_c8_s8_read_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    output["logical_sha256"] = Core.logical(output)
    Core.atomic(args.output_root / f"shard_{start:03d}_{stop:03d}.json", output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "shard_ordinal": args.shard_ordinal,
                "query_count": output["query_count"],
                "projected_query_count": projected,
                "target_miss_count": misses,
                "logical_sha256": output["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
