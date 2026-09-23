#!/usr/bin/env python3
"""Project exact-clean target P-lock pairs into donor geometry keys."""

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
OUTROOT = Path("results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1/producer")
PAIR_KEY_SCHEMA = "rc_dino_rcde_sr0_mt_v2_donor_geometry_key_v1_20260820"
PROJECTION_SCHEMA = "rc_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v1_20260820"
PROJECTION_STATUS = "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_READY"
TARGET_MISS = "TARGET_CLEAN_NOT_IN_NATURAL_C128"


class ProjectionError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ProjectionError(message)


def canonical(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("ascii")).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, name: str, *, immutable: bool = False) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{name} absent/non-regular")
    if immutable:
        require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{name} is not 0444")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{name} is not object")
    return value


def authority(path: Path) -> tuple[dict[str, Any], str]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = [
        (int(match.group(1)), item.resolve())
        for item in (ROOT / "registry").glob("current_authority_v*_*.json")
        if (match := pattern.fullmatch(item.name))
        and item.is_file()
        and not item.is_symlink()
    ]
    exact = (ROOT / AUTH).resolve(strict=True)
    require(
        rows
        and max(version for version, _ in rows) == 112
        and [item for version, item in rows if version == 112] == [exact]
        and path.resolve(strict=True) == exact,
        "V112 is not unique latest authority",
    )
    value = read_json(exact, "V112 authority", immutable=True)
    require(
        value.get("schema_version") == AUTH_SCHEMA
        and value.get("status") == AUTH_STATUS
        and value.get("logical_sha256") == logical(value),
        "V112 authority drift",
    )
    return value, file_sha256(exact)


def bound_path(authority_value: Mapping[str, Any], key: str, name: str) -> Path:
    binding = authority_value.get("bindings", {}).get(key)
    require(isinstance(binding, Mapping), f"{name} binding absent")
    path = ROOT / binding["path"]
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == binding.get("bytes")
        and file_sha256(path) == binding.get("sha256"),
        f"{name} physical binding drift",
    )
    return path


def bound_json(
    authority_value: Mapping[str, Any], key: str, name: str, *, immutable: bool = False
) -> dict[str, Any]:
    binding = authority_value["bindings"][key]
    path = bound_path(authority_value, key, name)
    value = read_json(path, name, immutable=immutable)
    if "logical_sha256" in binding:
        require(
            value.get("logical_sha256") == binding["logical_sha256"] == logical(value),
            f"{name} logical binding drift",
        )
    return value


def validate_authority_scope(authority_value: Mapping[str, Any]) -> None:
    expected_true = {
        "bound_metadata_read_authorized",
        "lock_artifact_hash_stream_authorized",
        "lock_payload_deserialization_authorized",
        "natural_lock_validation_authorized",
        "p_lock_consumption_for_donor_geometry_projection_authorized",
        "sealed_v2_lock_geometry_projection_read_authorized",
        "target_member_field_read_authorized",
    }
    require(
        {key for key, value in authority_value.items() if value is True} == expected_true
        and authority_value.get("consumable_lock_authorized") is False
        and authority_value.get("final_donor_matching_authorized") is False
        and authority_value.get("rival_member_as_donor_authorized") is False
        and authority_value.get("optimization_identity_supergroup_read_authorized")
        is False
        and authority_value.get("model_load_authorized") is False
        and authority_value.get("model_forward_authorized") is False
        and authority_value.get("training_authorized") is False
        and authority_value.get("heldout_scoring_authorized") is False
        and authority_value.get("protected_access_authorized") is False,
        "V112 exact permission scope drift",
    )


GEOMETRY_KEYS = ("area", "row_span", "col_span", "component_count", "border_touch")


def geometry_stats(mask: torch.Tensor, grid: tuple[int, int]) -> dict[str, Any]:
    payload = mask_geometry_v2(mask, grid).as_payload()
    return {key: payload[key] for key in GEOMETRY_KEYS}


def valid_geometry_stats(mask: torch.Tensor, grid: tuple[int, int]) -> dict[str, Any]:
    payload = mask_geometry_v2(mask, grid).as_payload()
    return {key: payload[key] for key in ("area", "row_span", "col_span")}


def direction_geometry(record: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    validate_natural_p_lock_v2(record)
    core = candidate_p_lock_v2_from_record(record["core_lock_record"])
    selected = record["selected"]
    selected_roots = core.selected_roots
    require(
        selected["lock_state"] == "P_LOCK_PROPOSAL_READY"
        and selected_roots
        and tuple(selected["selected_root_ordinals"]) == core.selected_root_ordinals,
        "exact-clean selected lock drift",
    )
    coverage = [int(item) for item in record["fixed_denominator"]["coverage"]]
    require(
        len(coverage) == core.query_valid_mask.numel()
        and all(item >= 0 for item in coverage),
        "coverage vector drift",
    )
    roots = []
    for root in selected_roots:
        roots.append(
            {
                "root_ordinal": root.root_ordinal,
                "state": root.status,
                "query": geometry_stats(root.query_mask, root.query_grid_shape),
                "reference": geometry_stats(
                    root.reference_mask, root.reference_grid_shape
                ),
            }
        )
    payload = {
        "schema_version": PAIR_KEY_SCHEMA,
        "canonical_arm": record["canonical_arm"],
        "track": record["query"]["track"],
        "fit_id": record["query"]["fit_id"],
        "p_checkpoint_sha256": record["p_checkpoint_sha256"],
        "direction": record["direction"],
        "query_grid_shape": list(core.geometry_signature.query_grid_shape),
        "reference_grid_shape": list(core.geometry_signature.reference_grid_shape),
        "query_valid_geometry": valid_geometry_stats(
            core.query_valid_mask, core.geometry_signature.query_grid_shape
        ),
        "reference_valid_geometry": valid_geometry_stats(
            core.reference_valid_mask, core.geometry_signature.reference_grid_shape
        ),
        "lock_state": core.lock_state,
        "selected_radius": selected["selected_radius"],
        "selected_root_ordinals": list(core.selected_root_ordinals),
        "selected_roots": roots,
        "selected_query_union_geometry": geometry_stats(
            core.query_union_mask, core.geometry_signature.query_grid_shape
        ),
        "overlap_count_value_sha256": canonical(
            {"dtype": "int64", "shape": [len(coverage)], "values": coverage}
        ),
    }
    query_provenance = canonical(
        {
            "query": record["query"],
            "query_geometry": record["query_geometry"],
            "selected_root_query_masks": [
                {
                    "root_ordinal": root.root_ordinal,
                    "query_mask_sha256": tensor_sha256(root.query_mask),
                }
                for root in selected_roots
            ],
            "query_union_sha256": record["fixed_denominator"]["query_union_sha256"],
            "coverage_sha256": record["fixed_denominator"]["coverage_sha256"],
            "record_sha256": record["record_sha256"],
        }
    )
    reference_provenance = canonical(
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
                for root in selected_roots
            ],
            "record_sha256": record["record_sha256"],
        }
    )
    return payload, query_provenance, reference_provenance


def resolve_target(
    execution: int,
    query_id: str,
    loss_roles: Mapping[str, Any],
    v87_manifest: Mapping[str, Any],
) -> tuple[int, str, str, str, str]:
    role_rows = [
        item
        for item in loss_roles["records"]
        if int(item["execution_ordinal"]) == execution
    ]
    query_rows = [
        item
        for item in v87_manifest["query_index"]
        if int(item["execution_ordinal"]) == execution
    ]
    require(len(role_rows) == len(query_rows) == 1, "target role/query join drift")
    role, query = role_rows[0], query_rows[0]
    require(role["query_id"] == query["query_id"] == query_id, "target query id drift")
    require(
        role["record_sha256"]
        == canonical({key: value for key, value in role.items() if key != "record_sha256"})
        and role["pair_sha256"] == query["pair_address_sha256"]
        and {role["target_member_ordinal"], role["rival_member_ordinal"]} == {0, 1}
        and query["directional_locators_sha256"] == canonical(query["directional_locators"]),
        "target role/pair manifest closure drift",
    )
    member = role["target_member_ordinal"]
    require(type(member) is int and member in (0, 1), "target member ordinal drift")
    locators = sorted(
        [
        item for item in query["directional_locators"] if item["member_ordinal"] == member
        ],
        key=lambda item: P_DIRECTIONS.index(item["direction"]),
    )
    require(
        len(locators) == 2
        and {item["direction"] for item in locators} == set(P_DIRECTIONS)
        and len({item["candidate_key"] for item in locators}) == 1
        and len({item["candidate_reference_source_sha256"] for item in locators}) == 1,
        "target directional locator drift",
    )
    return (
        member,
        locators[0]["candidate_key"],
        locators[0]["candidate_reference_source_sha256"],
        canonical(locators),
        query["query_source_image_sha256"],
    )


def project_query(
    aggregate_row: Mapping[str, Any],
    loss_roles: Mapping[str, Any],
    v87_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    execution = int(aggregate_row["execution_ordinal"])
    artifact = ROOT / aggregate_row["lock_artifact_path"]
    require(
        artifact.is_file()
        and not artifact.is_symlink()
        and stat.S_IMODE(artifact.stat().st_mode) == 0o444
        and file_sha256(artifact) == aggregate_row["lock_artifact_sha256"],
        f"lock artifact drift {execution}",
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(payload, dict)
        and payload.get("schema_version")
        == "rc_dino_rcde_sr0_mt_p_v2_formal_lock_artifact_v1_20260819"
        and payload.get("target_free") is True
        and payload.get("execution_ordinal") == execution
        and payload.get("query_id") == aggregate_row["query_id"]
        and payload.get("source_fold") == aggregate_row["source_fold"]
        and payload.get("record_count") == 1024,
        "lock payload envelope drift",
    )
    require(
        payload.get("logical_sha256") == aggregate_row["lock_artifact_logical_sha256"],
        "lock payload logical binding drift",
    )
    member, target_key, target_source, locator_sha, query_source = resolve_target(
        execution, aggregate_row["query_id"], loss_roles, v87_manifest
    )
    matches = [
        record for record in payload["records"] if record["candidate"]["candidate_key"] == target_key
    ]
    base = {
        "execution_ordinal": execution,
        "query_id": aggregate_row["query_id"],
        "query_source_image_sha256": query_source,
        "source_fold": aggregate_row["source_fold"],
        "lock_artifact_sha256": aggregate_row["lock_artifact_sha256"],
        "lock_artifact_logical_sha256": aggregate_row["lock_artifact_logical_sha256"],
        "target_member_ordinal": member,
        "target_candidate_key": target_key,
        "target_reference_source_sha256": target_source,
        "target_directional_locator_population_sha256": locator_sha,
    }
    if not matches:
        del payload
        gc.collect()
        return {
            **base,
            "status": TARGET_MISS,
            "scope_count": 0,
            "scope_projections": [],
        }
    require(len(matches) == 8, "target candidate must resolve to 4 heads x 2 directions")
    scopes = crossfit_fanout_scopes(int(aggregate_row["source_fold"]))
    by_fit = {scope.fit_id: [] for scope in scopes}
    for record in matches:
        require(record["query"]["fit_id"] in by_fit, "target record unexpected fit")
        by_fit[record["query"]["fit_id"]].append(record)
    scope_rows = []
    candidate_positions = set()
    for scope in scopes:
        records = sorted(by_fit[scope.fit_id], key=lambda item: P_DIRECTIONS.index(item["direction"]))
        require(
            len(records) == 2
            and tuple(item["direction"] for item in records) == tuple(P_DIRECTIONS)
            and len({item["candidate"]["candidate_position"] for item in records}) == 1
            and all(item["candidate"]["candidate_key"] == target_key for item in records)
            and all(item["query"]["query_source_image_sha256"] == query_source for item in records)
            and all(
                item["candidate"]["candidate_reference_source_sha256"] == target_source
                for item in records
            ),
            "target pair scope/direction drift",
        )
        direction_rows = []
        geometry_rows = []
        for record in records:
            geometry, query_provenance, reference_provenance = direction_geometry(record)
            geometry_rows.append(geometry)
            direction_rows.append(
                {
                    "direction": record["direction"],
                    "record_sha256": record["record_sha256"],
                    "candidate_native_content_sha256": record["candidate"][
                        "candidate_native_content_sha256"
                    ],
                    "direction_geometry_payload_sha256": canonical(geometry),
                    "recipient_query_provenance_sha256": query_provenance,
                    "donor_reference_provenance_sha256": reference_provenance,
                }
            )
        position = records[0]["candidate"]["candidate_position"]
        candidate_positions.add(position)
        scope_rows.append(
            {
                "fit_id": scope.fit_id,
                "fit_role": scope.fit_role,
                "crossfit_role": scope.crossfit_role,
                "outer_fold": scope.outer_fold,
                "inner_heldout_fold": scope.inner_heldout_fold,
                "p_checkpoint_sha256": records[0]["p_checkpoint_sha256"],
                "track": records[0]["query"]["track"],
                "target_candidate_position": position,
                "target_candidate_key": target_key,
                "target_candidate_physical_row": records[0]["candidate"][
                    "candidate_physical_row"
                ],
                "target_reference_source_sha256": target_source,
                "donor_geometry_key_sha256": canonical(
                    {"schema_version": PAIR_KEY_SCHEMA, "directions": geometry_rows}
                ),
                "direction_records": direction_rows,
            }
        )
    require(len(candidate_positions) == 1, "target candidate position drift across heads")
    del payload, matches
    gc.collect()
    return {
        **base,
        "status": "EXACT_CLEAN_TARGET_LOCK_PAIR_PROJECTED",
        "scope_count": 4,
        "scope_projections": scope_rows,
    }


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "output exists")
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
    parser.add_argument("--output-root", type=Path, default=ROOT / OUTROOT)
    args = parser.parse_args()
    require(0 <= args.shard_ordinal < 50, "shard ordinal drift")
    require(args.output_root.resolve(strict=False) == (ROOT / OUTROOT).resolve(strict=False), "output root drift")
    auth, auth_sha = authority(args.authority)
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
    strict_aggregate = bound_json(auth, "strict_payload_aggregate", "V111 aggregate", immutable=True)
    strict_validation = bound_json(auth, "strict_payload_aggregate_validation", "V111 validation", immutable=True)
    lock_aggregate = bound_json(auth, "lock_aggregate", "V109 lock aggregate", immutable=True)
    loss_roles = bound_json(auth, "loss_role_manifest", "loss role manifest", immutable=True)
    v87_manifest = bound_json(auth, "v87_query_manifest", "V87 query manifest", immutable=True)
    require(
        strict_aggregate.get("status") == "RCDE_SR0_MT_P_V2_STRICT_PAYLOAD_AGGREGATE_READY"
        and strict_validation.get("status")
        == "RCDE_SR0_MT_P_V2_STRICT_PAYLOAD_AGGREGATE_INDEPENDENT_VALIDATION_PASS"
        and strict_validation.get("validation_pass") is True
        and strict_validation.get("aggregate_sha256")
        == auth["bindings"]["strict_payload_aggregate"]["sha256"]
        and lock_aggregate.get("query_count") == 594
        and loss_roles.get("status") == "RCDE_SR0_MT_LOSS_ROLE_MANIFEST_QUARANTINED"
        and loss_roles.get("schema_version")
        == "rc_dino_rcde_sr0_mt_loss_role_manifest_v1_20260817"
        and loss_roles.get("query_count") == 594
        and v87_manifest.get("status")
        == "RCDE_SR0_MT_E0_FULL594_STRUCTURE_SIDECAR_MANIFEST_READY_STRUCTURAL_ONLY"
        and v87_manifest.get("schema_version")
        == "rc_dino_rcde_sr0_mt_full594_structure_sidecar_manifest_v1_20260818"
        and v87_manifest.get("query_count") == 594,
        "projection prerequisite drift",
    )
    start, stop = args.shard_ordinal * 12, args.shard_ordinal * 12 + 12
    rows = sorted(
        (
            item
            for item in lock_aggregate["rows"]
            if start <= int(item["execution_ordinal"]) < stop
        ),
        key=lambda item: int(item["execution_ordinal"]),
    )
    require(10 <= len(rows) <= 12, "projection shard population drift")
    projections = [project_query(item, loss_roles, v87_manifest) for item in rows]
    ready = sum(item["status"] == "EXACT_CLEAN_TARGET_LOCK_PAIR_PROJECTED" for item in projections)
    misses = sum(item["status"] == TARGET_MISS for item in projections)
    output = {
        "schema_version": PROJECTION_SCHEMA,
        "status": PROJECTION_STATUS,
        "claim_level": "ENGINEERING_EXACT_CLEAN_TARGET_LOCK_DONOR_GEOMETRY_PROJECTION_ONLY",
        "authority_sha256": auth_sha,
        "authority_logical_sha256": auth["logical_sha256"],
        "shard_ordinal": args.shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "query_count": len(projections),
        "projected_query_count": ready,
        "target_clean_not_in_natural_c128_count": misses,
        "scope_projection_count": ready * 4,
        "direction_record_decode_count": ready * 8,
        "query_projections": projections,
        "query_projection_population_sha256": canonical(projections),
        "loss_role_manifest_joined_record_count": len(projections),
        "target_member_field_read_count": len(projections),
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
    output["logical_sha256"] = logical(output)
    atomic(args.output_root / f"shard_{start:03d}_{stop:03d}.json", output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "shard_ordinal": args.shard_ordinal,
                "query_count": output["query_count"],
                "projected_query_count": ready,
                "target_miss_count": misses,
                "logical_sha256": output["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
