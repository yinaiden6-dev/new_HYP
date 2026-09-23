#!/usr/bin/env python3
"""Bound lightweight checker for a completed V113 projection validation receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v2 as V


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--shard-ordinal", required=True, type=int)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    args = parser.parse_args()
    V.Core.check(0 <= args.shard_ordinal < 50, "checker shard ordinal drift")
    start, stop = args.shard_ordinal * 12, args.shard_ordinal * 12 + 12
    authority, authority_sha = V.latest_authority(args.authority)
    role_validation = V.validate_authority_and_role(authority)
    expected_projection = (
        ROOT / V.PRODUCER_ROOT / f"shard_{start:03d}_{stop:03d}.json"
    ).resolve(strict=True)
    expected_validation = (
        ROOT / V.OUTROOT / f"shard_{start:03d}_{stop:03d}.json"
    ).resolve(strict=True)
    V.Core.check(args.projection.resolve(strict=True) == expected_projection, "checker projection path drift")
    V.Core.check(args.validation.resolve(strict=True) == expected_validation, "checker validation path drift")
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
        V.Core.bound_path(authority, key, name)
    projection = V.Core.load(expected_projection, "projection", mode444=True)
    receipt = V.Core.load(expected_validation, "validation", mode444=True)
    V.Core.check(
        set(projection) == V.PROJECTION_FIELDS
        and projection.get("schema_version") == V.PROJECTION_SCHEMA
        and projection.get("status") == V.PROJECTION_STATUS
        and projection.get("claim_level")
        == "ENGINEERING_ROLE_VALIDATED_EXACT_CLEAN_TARGET_LOCK_DONOR_GEOMETRY_PROJECTION_ONLY"
        and projection.get("logical_sha256") == V.Core.logical(projection)
        and projection.get("authority_sha256") == authority_sha
        and projection.get("authority_logical_sha256") == authority["logical_sha256"]
        and projection.get("role_validation_sha256")
        == authority["bindings"]["role_free_pair_address_validation"]["sha256"]
        and projection.get("role_validation_logical_sha256")
        == role_validation["logical_sha256"]
        and projection.get("shard_ordinal") == args.shard_ordinal
        and projection.get("execution_start") == start
        and projection.get("execution_stop") == stop
        and projection.get("query_projection_population_sha256")
        == V.Core.digest(projection.get("query_projections"))
        and projection.get("query_count") in (10, 11, 12)
        and projection.get("projected_query_count")
        + projection.get("target_clean_not_in_natural_c128_count")
        == projection.get("query_count")
        and projection.get("scope_projection_count")
        == projection.get("projected_query_count") * 4
        and projection.get("direction_record_decode_count")
        == projection.get("projected_query_count") * 8
        and projection.get("loss_role_manifest_joined_record_count")
        == projection.get("query_count")
        and projection.get("target_member_field_read_count") == projection.get("query_count")
        and projection.get("independent_target_role_validation_read_count") == 1
        and projection.get("rival_member_used_as_donor_count") == 0
        and projection.get("fixed_target_candidate_lookup_count")
        == projection.get("query_count")
        and projection.get("fixed_target_record_scan_count")
        == projection.get("query_count") * 1024
        and projection.get("candidate_scan_for_donor_coverage_count") == 0
        and projection.get("target_insertion_count") == 0
        and projection.get("lock_payload_deserialization_count") == projection.get("query_count")
        and projection.get("model_load_count") == 0
        and projection.get("model_forward_count") == 0
        and projection.get("training_count") == 0
        and projection.get("score_rank_winner_gap_outcome_read_count") == 0
        and projection.get("identity_supergroup_read_count") == 0
        and projection.get("opened_sealed_c8_s8_read_count") == 0
        and projection.get("scientific_GO_or_NO_GO") is None
        and projection.get("automatic_stage_advance") is False
        and projection.get("next_authorized_stage") is None
        and set(receipt) == V.VALIDATION_FIELDS
        and receipt.get("schema_version") == V.VALIDATION_SCHEMA
        and receipt.get("status") == V.VALIDATION_STATUS
        and receipt.get("validation_pass") is True
        and receipt.get("claim_level")
        == "INDEPENDENT_ROLE_VALIDATED_EXACT_CLEAN_TARGET_LOCK_GEOMETRY_PROJECTION_ONLY"
        and receipt.get("logical_sha256") == V.Core.logical(receipt)
        and receipt.get("authority_sha256") == authority_sha
        and receipt.get("authority_logical_sha256") == authority["logical_sha256"]
        and receipt.get("role_validation_sha256")
        == authority["bindings"]["role_free_pair_address_validation"]["sha256"]
        and receipt.get("role_validation_logical_sha256")
        == role_validation["logical_sha256"]
        and receipt.get("shard_ordinal") == args.shard_ordinal
        and receipt.get("execution_start") == start
        and receipt.get("execution_stop") == stop
        and receipt.get("projection_sha256") == V.Core.stream_hash(args.projection)
        and receipt.get("projection_logical_sha256") == projection.get("logical_sha256")
        and receipt.get("query_count") == projection.get("query_count")
        and receipt.get("projected_query_count") == projection.get("projected_query_count")
        and receipt.get("target_clean_not_in_natural_c128_count")
        == projection.get("target_clean_not_in_natural_c128_count")
        and receipt.get("query_projection_population_sha256")
        == projection.get("query_projection_population_sha256")
        and receipt.get("projected_query_count")
        + receipt.get("target_clean_not_in_natural_c128_count")
        == receipt.get("query_count")
        and receipt.get("scope_projection_count")
        == receipt.get("projected_query_count") * 4
        and receipt.get("direction_record_decode_count")
        == receipt.get("projected_query_count") * 8
        and receipt.get("lock_payload_deserialization_count") == receipt.get("query_count")
        and receipt.get("independent_target_role_validation_read_count") == 1
        and receipt.get("model_load_count") == 0
        and receipt.get("model_forward_count") == 0
        and receipt.get("training_count") == 0
        and receipt.get("score_rank_winner_gap_outcome_read_count") == 0
        and receipt.get("identity_supergroup_read_count") == 0
        and receipt.get("opened_sealed_c8_s8_read_count") == 0
        and receipt.get("scientific_GO_or_NO_GO") is None
        and receipt.get("automatic_stage_advance") is False
        and receipt.get("next_authorized_stage") is None,
        "completed V113 validation receipt drift",
    )
    print(
        json.dumps(
            {
                "status": "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_VALIDATION_RECEIPT_REPLAY_PASS",
                "shard_ordinal": args.shard_ordinal,
                "validation_logical_sha256": receipt["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
