#!/usr/bin/env python3
"""Independent V112 reconstruction of one donor-geometry projection shard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.dino_rcde_sr0_mt_p_compact_lock_fanout_v1 import (  # noqa: E402
    crossfit_fanout_scopes,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import (  # noqa: E402
    candidate_p_lock_v2_from_record,
    mask_geometry_v2,
    tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_adapter_v2 import (  # noqa: E402
    validate_natural_p_lock_v2,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import P_DIRECTIONS  # noqa: E402


AUTH = "registry/current_authority_v112_20260820.json"
AUTH_SCHEMA = "rc_current_authority_v112_20260820"
AUTH_STATUS = "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_AUTHORIZED"
PRODUCER_ROOT = Path(
    "results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1/producer"
)
OUTROOT = Path(
    "results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1/validation"
)
PAIR_KEY_SCHEMA = "rc_dino_rcde_sr0_mt_v2_donor_geometry_key_v1_20260820"
TARGET_MISS = "TARGET_CLEAN_NOT_IN_NATURAL_C128"
PROJECTION_FIELDS = {
    "schema_version",
    "status",
    "claim_level",
    "authority_sha256",
    "authority_logical_sha256",
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


def validate_projection_field_set(value: Mapping[str, Any]) -> None:
    check(isinstance(value, Mapping) and set(value) == PROJECTION_FIELDS, "projection exact field-set drift")


class IndependentProjectionError(RuntimeError):
    pass


def check(condition: Any, message: str) -> None:
    if not condition:
        raise IndependentProjectionError(message)


def digest(value: Any) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return digest({key: item for key, item in value.items() if key != "logical_sha256"})


def stream_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                return result.hexdigest()
            result.update(block)


def load(path: Path, name: str, *, mode444: bool = False) -> dict[str, Any]:
    check(path.is_file() and not path.is_symlink(), f"{name} absent/non-regular")
    if mode444:
        check(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{name} mode drift")
    value = json.loads(path.read_text())
    check(isinstance(value, dict), f"{name} object drift")
    return value


def get_authority(path: Path) -> tuple[dict[str, Any], str]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = []
    for item in (ROOT / "registry").glob("current_authority_v*_*.json"):
        match = pattern.fullmatch(item.name)
        if match and item.is_file() and not item.is_symlink():
            rows.append((int(match.group(1)), item.resolve()))
    exact = (ROOT / AUTH).resolve(strict=True)
    check(
        rows
        and max(version for version, _ in rows) == 112
        and [item for version, item in rows if version == 112] == [exact]
        and path.resolve(strict=True) == exact,
        "V112 latest authority drift",
    )
    value = load(exact, "V112 authority", mode444=True)
    check(
        value.get("schema_version") == AUTH_SCHEMA
        and value.get("status") == AUTH_STATUS
        and value.get("logical_sha256") == logical(value),
        "V112 authority envelope drift",
    )
    return value, stream_hash(exact)


def bound_path(authority: Mapping[str, Any], key: str, name: str) -> Path:
    binding = authority.get("bindings", {}).get(key)
    check(isinstance(binding, Mapping), f"{name} binding absent")
    path = ROOT / binding["path"]
    check(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == binding.get("bytes")
        and stream_hash(path) == binding.get("sha256"),
        f"{name} physical binding drift",
    )
    return path


def bound_json(
    authority: Mapping[str, Any], key: str, name: str, *, mode444: bool = False
) -> dict[str, Any]:
    binding = authority["bindings"][key]
    path = bound_path(authority, key, name)
    value = load(path, name, mode444=mode444)
    if "logical_sha256" in binding:
        check(
            value.get("logical_sha256") == binding["logical_sha256"] == logical(value),
            f"{name} logical binding drift",
        )
    return value


def validate_authority_scope(authority: Mapping[str, Any]) -> None:
    expected_true = {
        "bound_metadata_read_authorized",
        "lock_artifact_hash_stream_authorized",
        "lock_payload_deserialization_authorized",
        "natural_lock_validation_authorized",
        "p_lock_consumption_for_donor_geometry_projection_authorized",
        "sealed_v2_lock_geometry_projection_read_authorized",
        "target_member_field_read_authorized",
    }
    check(
        {key for key, value in authority.items() if value is True} == expected_true
        and authority.get("consumable_lock_authorized") is False
        and authority.get("final_donor_matching_authorized") is False
        and authority.get("rival_member_as_donor_authorized") is False
        and authority.get("optimization_identity_supergroup_read_authorized") is False
        and authority.get("model_load_authorized") is False
        and authority.get("model_forward_authorized") is False
        and authority.get("training_authorized") is False
        and authority.get("heldout_scoring_authorized") is False
        and authority.get("protected_access_authorized") is False,
        "V112 exact permission scope drift",
    )


FIELDS = ("area", "row_span", "col_span", "component_count", "border_touch")


def stats(mask: torch.Tensor, grid: tuple[int, int]) -> dict[str, Any]:
    value = mask_geometry_v2(mask, grid).as_payload()
    return {key: value[key] for key in FIELDS}


def valid_stats(mask: torch.Tensor, grid: tuple[int, int]) -> dict[str, Any]:
    value = mask_geometry_v2(mask, grid).as_payload()
    return {key: value[key] for key in ("area", "row_span", "col_span")}


def rebuild_direction(record: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    validate_natural_p_lock_v2(record)
    lock = candidate_p_lock_v2_from_record(record["core_lock_record"])
    selected = record["selected"]
    roots = lock.selected_roots
    check(
        selected["lock_state"] == "P_LOCK_PROPOSAL_READY"
        and roots
        and tuple(selected["selected_root_ordinals"]) == lock.selected_root_ordinals,
        "selected lock drift",
    )
    coverage = [int(item) for item in record["fixed_denominator"]["coverage"]]
    check(len(coverage) == lock.query_valid_mask.numel(), "coverage size drift")
    root_rows = [
        {
            "root_ordinal": root.root_ordinal,
            "state": root.status,
            "query": stats(root.query_mask, root.query_grid_shape),
            "reference": stats(root.reference_mask, root.reference_grid_shape),
        }
        for root in roots
    ]
    geometry = {
        "schema_version": PAIR_KEY_SCHEMA,
        "canonical_arm": record["canonical_arm"],
        "track": record["query"]["track"],
        "fit_id": record["query"]["fit_id"],
        "p_checkpoint_sha256": record["p_checkpoint_sha256"],
        "direction": record["direction"],
        "query_grid_shape": list(lock.geometry_signature.query_grid_shape),
        "reference_grid_shape": list(lock.geometry_signature.reference_grid_shape),
        "query_valid_geometry": valid_stats(
            lock.query_valid_mask, lock.geometry_signature.query_grid_shape
        ),
        "reference_valid_geometry": valid_stats(
            lock.reference_valid_mask, lock.geometry_signature.reference_grid_shape
        ),
        "lock_state": lock.lock_state,
        "selected_radius": selected["selected_radius"],
        "selected_root_ordinals": list(lock.selected_root_ordinals),
        "selected_roots": root_rows,
        "selected_query_union_geometry": stats(
            lock.query_union_mask, lock.geometry_signature.query_grid_shape
        ),
        "overlap_count_value_sha256": digest(
            {"dtype": "int64", "shape": [len(coverage)], "values": coverage}
        ),
    }
    query_receipt = digest(
        {
            "query": record["query"],
            "query_geometry": record["query_geometry"],
            "selected_root_query_masks": [
                {
                    "root_ordinal": root.root_ordinal,
                    "query_mask_sha256": tensor_sha256(root.query_mask),
                }
                for root in roots
            ],
            "query_union_sha256": record["fixed_denominator"]["query_union_sha256"],
            "coverage_sha256": record["fixed_denominator"]["coverage_sha256"],
            "record_sha256": record["record_sha256"],
        }
    )
    reference_receipt = digest(
        {
            "candidate": record["candidate"],
            "reference_geometry": record["reference_geometry"],
            "selected_root_reference": [
                {
                    "root_ordinal": root.root_ordinal,
                    "state": root.status,
                    "action_key_sha256": root.action_key_sha256,
                    "reference_mask_sha256": tensor_sha256(root.reference_mask),
                }
                for root in roots
            ],
            "record_sha256": record["record_sha256"],
        }
    )
    return geometry, query_receipt, reference_receipt


def target_locator(
    execution: int,
    query_id: str,
    roles: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[int, str, str, str, str]:
    role = [item for item in roles["records"] if int(item["execution_ordinal"]) == execution]
    query = [item for item in manifest["query_index"] if int(item["execution_ordinal"]) == execution]
    check(len(role) == len(query) == 1, "target locator join drift")
    role_row, query_row = role[0], query[0]
    check(role_row["query_id"] == query_row["query_id"] == query_id, "target query drift")
    check(
        role_row["record_sha256"]
        == digest(
            {key: value for key, value in role_row.items() if key != "record_sha256"}
        )
        and role_row["pair_sha256"] == query_row["pair_address_sha256"]
        and {role_row["target_member_ordinal"], role_row["rival_member_ordinal"]}
        == {0, 1}
        and query_row["directional_locators_sha256"]
        == digest(query_row["directional_locators"]),
        "target role/pair closure drift",
    )
    member = role_row["target_member_ordinal"]
    check(type(member) is int and member in (0, 1), "target member drift")
    locators = sorted(
        [item for item in query_row["directional_locators"] if item["member_ordinal"] == member],
        key=lambda item: P_DIRECTIONS.index(item["direction"]),
    )
    check(
        len(locators) == 2
        and tuple(item["direction"] for item in locators) == tuple(P_DIRECTIONS)
        and len({item["candidate_key"] for item in locators}) == 1
        and len({item["candidate_reference_source_sha256"] for item in locators}) == 1,
        "target locator directions drift",
    )
    return (
        member,
        locators[0]["candidate_key"],
        locators[0]["candidate_reference_source_sha256"],
        digest(locators),
        query_row["query_source_image_sha256"],
    )


def reconstruct_query(
    aggregate_row: Mapping[str, Any],
    roles: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    execution = int(aggregate_row["execution_ordinal"])
    artifact = ROOT / aggregate_row["lock_artifact_path"]
    check(
        artifact.is_file()
        and not artifact.is_symlink()
        and stat.S_IMODE(artifact.stat().st_mode) == 0o444
        and stream_hash(artifact) == aggregate_row["lock_artifact_sha256"],
        "artifact drift",
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=True, mmap=True)
    check(
        isinstance(payload, dict)
        and payload.get("schema_version")
        == "rc_dino_rcde_sr0_mt_p_v2_formal_lock_artifact_v1_20260819"
        and payload.get("target_free") is True
        and payload.get("execution_ordinal") == execution
        and payload.get("query_id") == aggregate_row["query_id"]
        and payload.get("source_fold") == aggregate_row["source_fold"]
        and payload.get("record_count") == 1024,
        "payload envelope drift",
    )
    check(
        payload.get("logical_sha256") == aggregate_row["lock_artifact_logical_sha256"],
        "payload logical binding drift",
    )
    member, key, source, locator_hash, query_source = target_locator(
        execution, aggregate_row["query_id"], roles, manifest
    )
    records = [item for item in payload["records"] if item["candidate"]["candidate_key"] == key]
    base = {
        "execution_ordinal": execution,
        "query_id": aggregate_row["query_id"],
        "query_source_image_sha256": query_source,
        "source_fold": aggregate_row["source_fold"],
        "lock_artifact_sha256": aggregate_row["lock_artifact_sha256"],
        "lock_artifact_logical_sha256": aggregate_row["lock_artifact_logical_sha256"],
        "target_member_ordinal": member,
        "target_candidate_key": key,
        "target_reference_source_sha256": source,
        "target_directional_locator_population_sha256": locator_hash,
    }
    if not records:
        del payload
        gc.collect()
        return {**base, "status": TARGET_MISS, "scope_count": 0, "scope_projections": []}
    check(len(records) == 8, "target full-C128 record multiplicity drift")
    scopes = crossfit_fanout_scopes(int(aggregate_row["source_fold"]))
    by_fit = {scope.fit_id: [] for scope in scopes}
    for record in records:
        check(record["query"]["fit_id"] in by_fit, "unexpected target fit")
        by_fit[record["query"]["fit_id"]].append(record)
    scope_rows = []
    positions = set()
    for scope in scopes:
        pair = sorted(by_fit[scope.fit_id], key=lambda item: P_DIRECTIONS.index(item["direction"]))
        check(
            len(pair) == 2
            and tuple(item["direction"] for item in pair) == tuple(P_DIRECTIONS)
            and len({item["candidate"]["candidate_position"] for item in pair}) == 1
            and all(item["candidate"]["candidate_key"] == key for item in pair)
            and all(item["candidate"]["candidate_reference_source_sha256"] == source for item in pair)
            and all(item["query"]["query_source_image_sha256"] == query_source for item in pair),
            "target pair binding drift",
        )
        geometry_rows, direction_rows = [], []
        for record in pair:
            geometry, query_receipt, reference_receipt = rebuild_direction(record)
            geometry_rows.append(geometry)
            direction_rows.append(
                {
                    "direction": record["direction"],
                    "record_sha256": record["record_sha256"],
                    "candidate_native_content_sha256": record["candidate"][
                        "candidate_native_content_sha256"
                    ],
                    "direction_geometry_payload_sha256": digest(geometry),
                    "recipient_query_provenance_sha256": query_receipt,
                    "donor_reference_provenance_sha256": reference_receipt,
                }
            )
        position = pair[0]["candidate"]["candidate_position"]
        positions.add(position)
        scope_rows.append(
            {
                "fit_id": scope.fit_id,
                "fit_role": scope.fit_role,
                "crossfit_role": scope.crossfit_role,
                "outer_fold": scope.outer_fold,
                "inner_heldout_fold": scope.inner_heldout_fold,
                "p_checkpoint_sha256": pair[0]["p_checkpoint_sha256"],
                "track": pair[0]["query"]["track"],
                "target_candidate_position": position,
                "target_candidate_key": key,
                "target_candidate_physical_row": pair[0]["candidate"]["candidate_physical_row"],
                "target_reference_source_sha256": source,
                "donor_geometry_key_sha256": digest(
                    {"schema_version": PAIR_KEY_SCHEMA, "directions": geometry_rows}
                ),
                "direction_records": direction_rows,
            }
        )
    check(len(positions) == 1, "target position cross-head drift")
    del payload, records
    gc.collect()
    return {
        **base,
        "status": "EXACT_CLEAN_TARGET_LOCK_PAIR_PROJECTED",
        "scope_count": 4,
        "scope_projections": scope_rows,
    }


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    check(not path.exists(), "validation output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    partial = Path(temporary)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        partial.chmod(0o444)
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--shard-ordinal", required=True, type=int)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTROOT)
    args = parser.parse_args()
    check(0 <= args.shard_ordinal < 50, "shard ordinal drift")
    start, stop = args.shard_ordinal * 12, args.shard_ordinal * 12 + 12
    expected_projection = (ROOT / PRODUCER_ROOT / f"shard_{start:03d}_{stop:03d}.json").resolve(strict=True)
    check(args.projection.resolve(strict=True) == expected_projection, "projection path drift")
    check(args.output_root.resolve(strict=False) == (ROOT / OUTROOT).resolve(strict=False), "output root drift")
    auth, auth_sha = get_authority(args.authority)
    validate_authority_scope(auth)
    for key, name in (
        ("contract", "donor contract"),
        ("projection_producer", "projection producer"),
        ("projection_validator", "projection validator"),
        ("lock_core", "lock core"),
        ("adapter", "natural adapter"),
        ("selector", "P selector"),
        ("structure_core", "CW1 structure core"),
        ("fanout", "fanout"),
        ("runtime_core", "P runtime core"),
        ("launcher", "projection launcher"),
    ):
        bound_path(auth, key, name)
    projection = load(args.projection, "producer projection", mode444=True)
    lock_aggregate = bound_json(auth, "lock_aggregate", "V109 lock aggregate", mode444=True)
    roles = bound_json(auth, "loss_role_manifest", "loss roles", mode444=True)
    manifest = bound_json(auth, "v87_query_manifest", "V87 manifest", mode444=True)
    check(
        roles.get("schema_version")
        == "rc_dino_rcde_sr0_mt_loss_role_manifest_v1_20260817"
        and roles.get("status") == "RCDE_SR0_MT_LOSS_ROLE_MANIFEST_QUARANTINED"
        and roles.get("query_count") == 594
        and manifest.get("schema_version")
        == "rc_dino_rcde_sr0_mt_full594_structure_sidecar_manifest_v1_20260818"
        and manifest.get("status")
        == "RCDE_SR0_MT_E0_FULL594_STRUCTURE_SIDECAR_MANIFEST_READY_STRUCTURAL_ONLY"
        and manifest.get("query_count") == 594,
        "role/manifest prerequisite drift",
    )
    rows = sorted(
        [item for item in lock_aggregate["rows"] if start <= int(item["execution_ordinal"]) < stop],
        key=lambda item: int(item["execution_ordinal"]),
    )
    expected = [reconstruct_query(item, roles, manifest) for item in rows]
    projected_count = sum(
        item["status"] == "EXACT_CLEAN_TARGET_LOCK_PAIR_PROJECTED" for item in expected
    )
    target_miss_count = sum(item["status"] == TARGET_MISS for item in expected)
    validate_projection_field_set(projection)
    check(
        projection.get("schema_version")
        == "rc_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v1_20260820"
        and projection.get("status")
        == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_READY"
        and projection.get("claim_level")
        == "ENGINEERING_EXACT_CLEAN_TARGET_LOCK_DONOR_GEOMETRY_PROJECTION_ONLY"
        and projection.get("logical_sha256") == logical(projection)
        and projection.get("authority_sha256") == auth_sha
        and projection.get("authority_logical_sha256") == auth["logical_sha256"]
        and projection.get("shard_ordinal") == args.shard_ordinal
        and projection.get("execution_start") == start
        and projection.get("execution_stop") == stop
        and projection.get("query_projections") == expected
        and projection.get("query_projection_population_sha256") == digest(expected)
        and projection.get("query_count") == len(expected)
        and projection.get("projected_query_count")
        == projected_count
        and projection.get("target_clean_not_in_natural_c128_count")
        == target_miss_count
        and projection.get("scope_projection_count") == projected_count * 4
        and projection.get("direction_record_decode_count") == projected_count * 8
        and projection.get("lock_payload_deserialization_count") == len(expected)
        and projection.get("rival_member_used_as_donor_count") == 0
        and projection.get("loss_role_manifest_joined_record_count") == len(expected)
        and projection.get("target_member_field_read_count") == len(expected)
        and projection.get("fixed_target_candidate_lookup_count") == len(expected)
        and projection.get("fixed_target_record_scan_count") == len(expected) * 1024
        and projection.get("candidate_scan_for_donor_coverage_count") == 0
        and projection.get("target_insertion_count") == 0
        and projection.get("model_load_count") == 0
        and projection.get("model_forward_count") == 0
        and projection.get("training_count") == 0
        and projection.get("score_rank_winner_gap_outcome_read_count") == 0
        and projection.get("identity_supergroup_read_count") == 0
        and projection.get("opened_sealed_c8_s8_read_count") == 0
        and projection.get("scientific_GO_or_NO_GO") is None
        and projection.get("automatic_stage_advance") is False
        and projection.get("next_authorized_stage") is None,
        "independent projection reconstruction drift",
    )
    output = {
        "schema_version": "rc_dino_rcde_sr0_mt_v2_donor_geometry_projection_validation_shard_v1_20260820",
        "status": "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_INDEPENDENT_VALIDATION_PASS",
        "claim_level": "INDEPENDENT_EXACT_CLEAN_TARGET_LOCK_GEOMETRY_PROJECTION_ONLY",
        "validation_pass": True,
        "authority_sha256": auth_sha,
        "authority_logical_sha256": auth["logical_sha256"],
        "shard_ordinal": args.shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "projection_sha256": stream_hash(args.projection),
        "projection_logical_sha256": projection["logical_sha256"],
        "query_count": len(expected),
        "projected_query_count": projection["projected_query_count"],
        "target_clean_not_in_natural_c128_count": projection[
            "target_clean_not_in_natural_c128_count"
        ],
        "scope_projection_count": projection["scope_projection_count"],
        "direction_record_decode_count": projection["direction_record_decode_count"],
        "query_projection_population_sha256": digest(expected),
        "lock_payload_deserialization_count": len(expected),
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
    output["logical_sha256"] = logical(output)
    atomic(args.output_root / f"shard_{start:03d}_{stop:03d}.json", output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "shard_ordinal": args.shard_ordinal,
                "query_count": output["query_count"],
                "projected_query_count": output["projected_query_count"],
                "target_miss_count": output["target_clean_not_in_natural_c128_count"],
                "logical_sha256": output["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
