#!/usr/bin/env python3
"""Independent V113 validator for role-validated donor projections."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import validate_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v1 as Core


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v113_20260820.json"
AUTH_SCHEMA = "rc_current_authority_v113_20260820"
AUTH_STATUS = "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_RESCOPE_REPAIR_AUTHORIZED"
PRODUCER_ROOT = Core.PRODUCER_ROOT
OUTROOT = Core.OUTROOT
PROJECTION_SCHEMA = "rc_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v2_20260820"
PROJECTION_STATUS = "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_READY"
VALIDATION_SCHEMA = "rc_dino_rcde_sr0_mt_v2_donor_geometry_projection_validation_shard_v2_20260820"
VALIDATION_STATUS = "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_INDEPENDENT_VALIDATION_PASS"


PROJECTION_FIELDS = {
    "schema_version",
    "status",
    "claim_level",
    "authority_sha256",
    "authority_logical_sha256",
    "role_validation_sha256",
    "role_validation_logical_sha256",
    "shard_ordinal",
    "execution_start",
    "execution_stop",
    "query_count",
    "projected_query_count",
    "target_clean_not_in_natural_c128_count",
    "scope_projection_count",
    "direction_record_decode_count",
    "query_projections",
    "query_projection_population_sha256",
    "loss_role_manifest_joined_record_count",
    "target_member_field_read_count",
    "independent_target_role_validation_read_count",
    "rival_member_used_as_donor_count",
    "fixed_target_candidate_lookup_count",
    "fixed_target_record_scan_count",
    "candidate_scan_for_donor_coverage_count",
    "target_insertion_count",
    "lock_payload_deserialization_count",
    "model_load_count",
    "model_forward_count",
    "training_count",
    "score_rank_winner_gap_outcome_read_count",
    "identity_supergroup_read_count",
    "opened_sealed_c8_s8_read_count",
    "scientific_GO_or_NO_GO",
    "automatic_stage_advance",
    "next_authorized_stage",
    "logical_sha256",
}

VALIDATION_FIELDS = {
    "schema_version",
    "status",
    "claim_level",
    "validation_pass",
    "authority_sha256",
    "authority_logical_sha256",
    "role_validation_sha256",
    "role_validation_logical_sha256",
    "shard_ordinal",
    "execution_start",
    "execution_stop",
    "projection_sha256",
    "projection_logical_sha256",
    "query_count",
    "projected_query_count",
    "target_clean_not_in_natural_c128_count",
    "scope_projection_count",
    "direction_record_decode_count",
    "query_projection_population_sha256",
    "lock_payload_deserialization_count",
    "independent_target_role_validation_read_count",
    "model_load_count",
    "model_forward_count",
    "training_count",
    "score_rank_winner_gap_outcome_read_count",
    "identity_supergroup_read_count",
    "opened_sealed_c8_s8_read_count",
    "scientific_GO_or_NO_GO",
    "automatic_stage_advance",
    "next_authorized_stage",
    "logical_sha256",
}


def latest_authority(path: Path) -> tuple[dict[str, Any], str]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = []
    for item in (ROOT / "registry").glob("current_authority_v*_*.json"):
        match = pattern.fullmatch(item.name)
        if match and item.is_file() and not item.is_symlink():
            rows.append((int(match.group(1)), item.resolve()))
    exact = (ROOT / AUTH).resolve(strict=True)
    Core.check(
        rows
        and max(version for version, _ in rows) == 113
        and [item for version, item in rows if version == 113] == [exact]
        and path.resolve(strict=True) == exact,
        "V113 latest authority drift",
    )
    value = Core.load(exact, "V113 authority", mode444=True)
    Core.check(
        value.get("schema_version") == AUTH_SCHEMA
        and value.get("status") == AUTH_STATUS
        and value.get("logical_sha256") == Core.logical(value),
        "V113 authority envelope drift",
    )
    return value, Core.stream_hash(exact)


def validate_authority_and_role(authority: Mapping[str, Any]) -> dict[str, Any]:
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
    Core.check(
        {key for key, value in authority.items() if value is True} == expected_true
        and authority.get("consumable_lock_authorized") is False
        and authority.get("final_donor_matching_authorized") is False
        and authority.get("training_authorized") is False
        and authority.get("heldout_scoring_authorized") is False
        and authority.get("protected_access_authorized") is False,
        "V113 exact permission scope drift",
    )
    role_path = Core.bound_path(authority, "loss_role_manifest", "loss role manifest")
    validation = Core.bound_json(
        authority,
        "role_free_pair_address_validation",
        "role-free pair-address validation",
        mode444=True,
    )
    Core.check(
        validation.get("schema_version")
        == "rc_dino_rcde_sr0_mt_role_free_pair_address_validation_v1_20260817"
        and validation.get("status")
        == "RCDE_SR0_MT_ROLE_FREE_PAIR_ADDRESS_INDEPENDENT_VALIDATION_PASS"
        and validation.get("validation_pass") is True
        and validation.get("pair_query_count") == 594
        and validation.get("member_count") == 1188
        and validation.get("loss_role_manifest_sha256") == Core.stream_hash(role_path),
        "independent target-role validation drift",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--shard-ordinal", required=True, type=int)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTROOT)
    args = parser.parse_args()
    Core.check(0 <= args.shard_ordinal < 50, "shard ordinal drift")
    start, stop = args.shard_ordinal * 12, args.shard_ordinal * 12 + 12
    expected_projection = (
        ROOT / PRODUCER_ROOT / f"shard_{start:03d}_{stop:03d}.json"
    ).resolve(strict=True)
    Core.check(args.projection.resolve(strict=True) == expected_projection, "projection path drift")
    Core.check(
        args.output_root.resolve(strict=False) == (ROOT / OUTROOT).resolve(strict=False),
        "output root drift",
    )
    authority, authority_sha = latest_authority(args.authority)
    role_validation = validate_authority_and_role(authority)
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
    projection = Core.load(args.projection, "V113 projection", mode444=True)
    lock_aggregate = Core.bound_json(
        authority, "lock_aggregate", "V109 lock aggregate", mode444=True
    )
    roles = Core.bound_json(
        authority, "loss_role_manifest", "loss role manifest", mode444=True
    )
    manifest = Core.bound_json(
        authority, "v87_query_manifest", "V87 query manifest", mode444=True
    )
    source_rows = sorted(
        [
            item
            for item in lock_aggregate["rows"]
            if start <= int(item["execution_ordinal"]) < stop
        ],
        key=lambda item: int(item["execution_ordinal"]),
    )
    expected = [Core.reconstruct_query(item, roles, manifest) for item in source_rows]
    projected = sum(
        item["status"] == "EXACT_CLEAN_TARGET_LOCK_PAIR_PROJECTED"
        for item in expected
    )
    misses = sum(item["status"] == Core.TARGET_MISS for item in expected)
    Core.check(
        set(projection) == PROJECTION_FIELDS
        and projection.get("schema_version") == PROJECTION_SCHEMA
        and projection.get("status") == PROJECTION_STATUS
        and projection.get("claim_level")
        == "ENGINEERING_ROLE_VALIDATED_EXACT_CLEAN_TARGET_LOCK_DONOR_GEOMETRY_PROJECTION_ONLY"
        and projection.get("logical_sha256") == Core.logical(projection)
        and projection.get("authority_sha256") == authority_sha
        and projection.get("authority_logical_sha256") == authority["logical_sha256"]
        and projection.get("role_validation_sha256")
        == authority["bindings"]["role_free_pair_address_validation"]["sha256"]
        and projection.get("role_validation_logical_sha256")
        == role_validation["logical_sha256"]
        and projection.get("shard_ordinal") == args.shard_ordinal
        and projection.get("execution_start") == start
        and projection.get("execution_stop") == stop
        and projection.get("query_count") == len(expected)
        and projection.get("query_projections") == expected
        and projection.get("query_projection_population_sha256")
        == Core.digest(expected)
        and projection.get("projected_query_count") == projected
        and projection.get("target_clean_not_in_natural_c128_count") == misses
        and projection.get("scope_projection_count") == projected * 4
        and projection.get("direction_record_decode_count") == projected * 8
        and projection.get("loss_role_manifest_joined_record_count") == len(expected)
        and projection.get("target_member_field_read_count") == len(expected)
        and projection.get("independent_target_role_validation_read_count") == 1
        and projection.get("rival_member_used_as_donor_count") == 0
        and projection.get("fixed_target_candidate_lookup_count") == len(expected)
        and projection.get("fixed_target_record_scan_count") == len(expected) * 1024
        and projection.get("candidate_scan_for_donor_coverage_count") == 0
        and projection.get("target_insertion_count") == 0
        and projection.get("lock_payload_deserialization_count") == len(expected)
        and projection.get("model_load_count") == 0
        and projection.get("model_forward_count") == 0
        and projection.get("training_count") == 0
        and projection.get("score_rank_winner_gap_outcome_read_count") == 0
        and projection.get("identity_supergroup_read_count") == 0
        and projection.get("opened_sealed_c8_s8_read_count") == 0
        and projection.get("scientific_GO_or_NO_GO") is None
        and projection.get("automatic_stage_advance") is False
        and projection.get("next_authorized_stage") is None,
        "independent V113 projection reconstruction drift",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "claim_level": "INDEPENDENT_ROLE_VALIDATED_EXACT_CLEAN_TARGET_LOCK_GEOMETRY_PROJECTION_ONLY",
        "validation_pass": True,
        "authority_sha256": authority_sha,
        "authority_logical_sha256": authority["logical_sha256"],
        "role_validation_sha256": authority["bindings"][
            "role_free_pair_address_validation"
        ]["sha256"],
        "role_validation_logical_sha256": role_validation["logical_sha256"],
        "shard_ordinal": args.shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "projection_sha256": Core.stream_hash(args.projection),
        "projection_logical_sha256": projection["logical_sha256"],
        "query_count": len(expected),
        "projected_query_count": projected,
        "target_clean_not_in_natural_c128_count": misses,
        "scope_projection_count": projected * 4,
        "direction_record_decode_count": projected * 8,
        "query_projection_population_sha256": Core.digest(expected),
        "lock_payload_deserialization_count": len(expected),
        "independent_target_role_validation_read_count": 1,
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
                "query_count": len(expected),
                "projected_query_count": projected,
                "target_miss_count": misses,
                "logical_sha256": output["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
