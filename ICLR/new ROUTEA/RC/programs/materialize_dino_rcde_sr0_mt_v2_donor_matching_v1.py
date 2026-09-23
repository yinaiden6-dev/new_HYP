#!/usr/bin/env python3
"""Construct the frozen 16-scope partial donor ledger from V113 projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v114_20260820.json"
AUTH_SCHEMA = "rc_current_authority_v114_20260820"
AUTH_STATUS = "RCDE_SR0_MT_V2_DONOR_MATCHING_REDUCER_AUTHORIZED"
PROJECTION_ROOT = Path(
    "results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1"
)
OUT = Path("results/dino_rcde_sr0_mt_v2_donor_matching_v1/result.json")
NO_EDGE_STATUS = "RCDE_SR0_MT_V2_DONOR_GEOMETRY_NO_ELIGIBLE_EDGE_STOP"


class DonorMatchingError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise DonorMatchingError(message)


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "logical_sha256"})


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, name: str, *, mode444: bool = False) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{name} absent/non-regular")
    if mode444:
        require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{name} mode drift")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{name} object drift")
    return value


def get_authority(path: Path) -> tuple[dict[str, Any], str]:
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
        and max(version for version, _ in rows) == 114
        and [item for version, item in rows if version == 114] == [exact]
        and path.resolve(strict=True) == exact,
        "V114 latest authority drift",
    )
    value = read(exact, "V114 authority", mode444=True)
    require(
        value.get("schema_version") == AUTH_SCHEMA
        and value.get("status") == AUTH_STATUS
        and value.get("logical_sha256") == logical(value),
        "V114 authority envelope drift",
    )
    return value, file_sha(exact)


def bound_json(
    authority: Mapping[str, Any], key: str, name: str, *, mode444: bool = False
) -> dict[str, Any]:
    binding = authority.get("bindings", {}).get(key)
    require(isinstance(binding, Mapping), f"{name} binding absent")
    path = ROOT / binding["path"]
    value = read(path, name, mode444=mode444)
    require(
        path.stat().st_size == binding.get("bytes")
        and file_sha(path) == binding.get("sha256"),
        f"{name} physical binding drift",
    )
    if "logical_sha256" in binding:
        require(
            value.get("logical_sha256") == binding["logical_sha256"] == logical(value),
            f"{name} logical binding drift",
        )
    return value


def bound_path(authority: Mapping[str, Any], key: str, name: str) -> Path:
    binding = authority.get("bindings", {}).get(key)
    require(isinstance(binding, Mapping) and "path" in binding, f"{name} binding absent")
    path = ROOT / binding["path"]
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == binding["bytes"]
        and file_sha(path) == binding["sha256"],
        f"{name} runtime binding drift",
    )
    return path


def validate_authority_scope(authority: Mapping[str, Any]) -> None:
    require(
        {key for key, value in authority.items() if value is True}
        == {
            "projection_receipt_read_authorized",
            "scheduler_migration_adoption_authorized",
            "optimization_identity_supergroup_read_authorized",
            "final_donor_matching_authorized",
        }
        and authority.get("lock_payload_deserialization_authorized") is False
        and authority.get("model_load_authorized") is False
        and authority.get("training_authorized") is False
        and authority.get("V_training_authorized") is False
        and authority.get("heldout_scoring_authorized") is False
        and authority.get("protected_access_authorized") is False,
        "V114 exact permission scope drift",
    )


def maximum_cardinality(
    adjacency: Sequence[Sequence[int]],
    recipients: Sequence[int] | None = None,
    donors: set[int] | None = None,
) -> int:
    left = list(range(len(adjacency))) if recipients is None else list(recipients)
    available = set(range(len(adjacency))) if donors is None else set(donors)
    right_match: dict[int, int] = {}

    def augment(recipient: int, seen: set[int]) -> bool:
        for donor in adjacency[recipient]:
            if donor not in available or donor in seen:
                continue
            seen.add(donor)
            prior = right_match.get(donor)
            if prior is None or augment(prior, seen):
                right_match[donor] = recipient
                return True
        return False

    return sum(augment(recipient, set()) for recipient in left)


def canonical_maximum_matching(adjacency: Sequence[Sequence[int]]) -> list[int | None]:
    count = len(adjacency)
    target = maximum_cardinality(adjacency)
    result: list[int | None] = []
    used: set[int] = set()
    matched = 0
    for position in range(count):
        choices: list[int | None] = [
            donor for donor in adjacency[position] if donor not in used
        ] + [None]
        chosen: int | None = None
        for candidate in choices:
            next_used = used | ({candidate} if candidate is not None else set())
            remaining = list(range(position + 1, count))
            available = set(range(count)) - next_used
            possible = maximum_cardinality(adjacency, remaining, available)
            if matched + int(candidate is not None) + possible == target:
                chosen = candidate
                break
        require(chosen is not None or matched + maximum_cardinality(
            adjacency, list(range(position + 1, count)), set(range(count)) - used
        ) == target, "canonical maximum matching construction failed")
        result.append(chosen)
        if chosen is not None:
            used.add(chosen)
            matched += 1
    require(matched == target and len(used) == target, "canonical matching cardinality drift")
    return result


def validate_migration_receipt(
    authority: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[int, dict[str, Any]]:
    migration = bound_json(
        authority, "scheduler_migration_receipt", "scheduler migration receipt"
    )
    expected_fields = {
        "schema_version", "status", "authority_sha256", "authority_logical_sha256",
        "superseded_pending_job_id", "dev_initial_job_id",
        "dev_postvalidation_failed_job_id", "migrated_cpu_array_job_id",
        "original_cpu_array_job_id", "moved_launcher_path", "moved_launcher_sha256",
        "scheduler_override_addendum_path", "scheduler_override_addendum_sha256",
        "migration_stop_path", "migration_stop_sha256",
        "migration_stop_logical_sha256", "shard_count", "query_count",
        "scope_projection_count", "direction_record_decode_count", "shard_rows",
        "shard_population_sha256", "projection_logic_changed", "model_load_count",
        "training_count", "scientific_GO_or_NO_GO", "automatic_stage_advance",
        "next_authorized_stage", "logical_sha256",
    }
    row_fields = {
        "shard_ordinal", "route", "array_job_id", "accounting", "producer_path",
        "producer_sha256", "producer_logical_sha256", "validation_path",
        "validation_sha256", "validation_logical_sha256", "query_count",
        "scope_projection_count", "direction_record_decode_count",
    }
    accounting_fields = {"job_id", "state", "exit_code", "elapsed", "start", "end"}
    rows = migration.get("shard_rows")
    require(
        set(migration) == expected_fields
        and migration.get("schema_version")
        == "rc_dino_rcde_sr0_mt_v2_projection_scheduler_migration_receipt_v1_20260820"
        and migration.get("status")
        == "RCDE_SR0_MT_V2_PROJECTION_SCHEDULER_MIGRATION_INPUT_READY"
        and migration.get("logical_sha256") == logical(migration)
        and migration.get("authority_sha256")
        == authority["bindings"]["parent_authority_v113"]["sha256"]
        and migration.get("authority_logical_sha256") == parent["logical_sha256"]
        and isinstance(rows, list)
        and len(rows) == 50
        and migration.get("shard_population_sha256") == canonical(rows)
        and migration.get("projection_logic_changed") is False
        and migration.get("model_load_count") == 0
        and migration.get("training_count") == 0
        and migration.get("scientific_GO_or_NO_GO") is None
        and migration.get("automatic_stage_advance") is False
        and migration.get("next_authorized_stage") is None,
        "scheduler migration envelope drift",
    )
    require(
        migration.get("moved_launcher_sha256")
        == authority["bindings"]["moved_launcher"]["sha256"]
        and migration.get("moved_launcher_path")
        == authority["bindings"]["moved_launcher"]["path"]
        and migration.get("scheduler_override_addendum_sha256")
        == authority["bindings"]["scheduler_override_addendum"]["sha256"]
        and migration.get("scheduler_override_addendum_path")
        == authority["bindings"]["scheduler_override_addendum"]["path"]
        and migration.get("migration_stop_sha256")
        == authority["bindings"]["migration_stop"]["sha256"]
        and migration.get("migration_stop_path")
        == authority["bindings"]["migration_stop"]["path"],
        "scheduler migration file binding drift",
    )
    by_shard = {}
    for ordinal, row in enumerate(rows):
        require(
            set(row) == row_fields
            and row.get("shard_ordinal") == ordinal
            and set(row.get("accounting", {})) == accounting_fields,
            "scheduler migration shard row drift",
        )
        expected_route = (
            "DEV_SERIAL"
            if ordinal <= 10
            else "CPU_MIGRATED_11_19"
            if ordinal <= 19
            else "CPU_ORIGINAL_20_49"
        )
        accounting = row["accounting"]
        require(row["route"] == expected_route, "scheduler migration route drift")
        if ordinal == 10:
            require(
                accounting["state"] == "FAILED"
                and accounting["exit_code"] == "1:0",
                "dev10 postvalidation state drift",
            )
        else:
            require(
                accounting["state"] == "COMPLETED"
                and accounting["exit_code"] == "0:0",
                "completed shard accounting drift",
            )
        producer_path = ROOT / row["producer_path"]
        validation_path = ROOT / row["validation_path"]
        producer = read(producer_path, "migration producer", mode444=True)
        validation = read(validation_path, "migration validation", mode444=True)
        require(
            file_sha(producer_path) == row["producer_sha256"]
            and producer.get("logical_sha256") == row["producer_logical_sha256"]
            and producer.get("logical_sha256") == logical(producer)
            and file_sha(validation_path) == row["validation_sha256"]
            and validation.get("logical_sha256") == row["validation_logical_sha256"]
            and validation.get("logical_sha256") == logical(validation)
            and row["query_count"] == validation["query_count"]
            and row["scope_projection_count"] == validation["scope_projection_count"]
            and row["direction_record_decode_count"]
            == validation["direction_record_decode_count"],
            "scheduler migration output row drift",
        )
        by_shard[ordinal] = row
    require(
        migration.get("shard_count") == 50
        and migration.get("query_count") == sum(row["query_count"] for row in rows) == 594
        and migration.get("scope_projection_count")
        == sum(row["scope_projection_count"] for row in rows)
        == 2376
        and migration.get("direction_record_decode_count")
        == sum(row["direction_record_decode_count"] for row in rows)
        == 4752,
        "scheduler migration global count drift",
    )
    return by_shard


def projection_population(authority: Mapping[str, Any]) -> list[dict[str, Any]]:
    parent = bound_json(authority, "parent_authority_v113", "V113 authority", mode444=True)
    require(
        parent.get("status")
        == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_RESCOPE_REPAIR_AUTHORIZED",
        "V113 parent status drift",
    )
    migration_rows = validate_migration_receipt(authority, parent)
    rows: list[dict[str, Any]] = []
    for shard in range(50):
        start, stop = shard * 12, shard * 12 + 12
        producer_path = ROOT / PROJECTION_ROOT / "producer" / f"shard_{start:03d}_{stop:03d}.json"
        validation_path = ROOT / PROJECTION_ROOT / "validation" / f"shard_{start:03d}_{stop:03d}.json"
        producer = read(producer_path, f"projection shard {shard}", mode444=True)
        validation = read(validation_path, f"projection validation {shard}", mode444=True)
        migration_row = migration_rows[shard]
        require(
            migration_row["producer_path"] == producer_path.relative_to(ROOT).as_posix()
            and migration_row["validation_path"]
            == validation_path.relative_to(ROOT).as_posix()
            and migration_row["producer_sha256"] == file_sha(producer_path)
            and migration_row["validation_sha256"] == file_sha(validation_path)
            and migration_row["producer_logical_sha256"]
            == producer.get("logical_sha256")
            and migration_row["validation_logical_sha256"]
            == validation.get("logical_sha256")
            and
            producer.get("schema_version")
            == "rc_dino_rcde_sr0_mt_v2_donor_geometry_projection_shard_v2_20260820"
            and producer.get("status")
            == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_READY"
            and producer.get("logical_sha256") == logical(producer)
            and producer.get("authority_logical_sha256") == parent["logical_sha256"]
            and producer.get("shard_ordinal") == shard
            and producer.get("execution_start") == start
            and producer.get("execution_stop") == stop
            and validation.get("schema_version")
            == "rc_dino_rcde_sr0_mt_v2_donor_geometry_projection_validation_shard_v2_20260820"
            and validation.get("status")
            == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_INDEPENDENT_VALIDATION_PASS"
            and validation.get("validation_pass") is True
            and validation.get("logical_sha256") == logical(validation)
            and validation.get("authority_logical_sha256") == parent["logical_sha256"]
            and validation.get("projection_sha256") == file_sha(producer_path)
            and validation.get("projection_logical_sha256") == producer["logical_sha256"]
            and validation.get("query_projection_population_sha256")
            == producer["query_projection_population_sha256"],
            "projection/validation closure drift",
        )
        rows.extend(producer["query_projections"])
    executions = [int(item["execution_ordinal"]) for item in rows]
    require(
        len(rows) == len(set(executions)) == 594
        and all(item["status"] == "EXACT_CLEAN_TARGET_LOCK_PAIR_PROJECTED" for item in rows)
        and sum(item["scope_count"] for item in rows) == 2376,
        "projection union drift",
    )
    return sorted(rows, key=lambda item: int(item["execution_ordinal"]))


def metadata_population(
    query_manifest: Mapping[str, Any],
    prejoin: Mapping[str, Any],
    roles: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    query_rows = query_manifest["query_index"]
    prejoin_by_id = {item["query_id"]: item for item in prejoin["records"]}
    roles_by_id = {item["query_id"]: item for item in roles["records"]}
    require(len(prejoin_by_id) == len(roles_by_id) == 600, "metadata source population drift")
    result = {}
    for query in query_rows:
        execution = int(query["execution_ordinal"])
        query_id = query["query_id"]
        fold = prejoin_by_id[query_id]
        role = roles_by_id[query_id]
        require(
            fold["source_image_sha256"] == query["query_source_image_sha256"]
            and role["query_ordinal"] == fold["query_ordinal"]
            and role["track"] == fold["track"],
            "metadata query/fold/role join drift",
        )
        result[execution] = {
            "query_id": query_id,
            "source_sha": query["query_source_image_sha256"],
            "identity": role["identity"],
            "supergroup": role["supergroup"],
            "track": role["track"],
            "inner_fold": role["inner_fold"],
        }
    require(len(result) == 594, "selected metadata population drift")
    return result


def build_result(authority: Mapping[str, Any], authority_sha: str) -> dict[str, Any]:
    projections = projection_population(authority)
    query_manifest = bound_json(authority, "v87_query_manifest", "V87 manifest", mode444=True)
    prejoin = bound_json(authority, "prejoin_folds", "prejoin folds")
    roles = bound_json(authority, "training_roles", "training roles")
    feasibility = bound_json(authority, "v92_feasibility", "V92 feasibility", mode444=True)
    feasibility_validation = bound_json(
        authority, "v92_feasibility_validation", "V92 validation", mode444=True
    )
    require(
        feasibility_validation.get("validation_pass") is True
        and feasibility_validation.get("producer_feasibility_logical_sha256")
        == feasibility["logical_sha256"],
        "V92 feasibility validation drift",
    )
    metadata = metadata_population(query_manifest, prejoin, roles)
    projected_by_execution = {int(item["execution_ordinal"]): item for item in projections}
    nodes_by_fit: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = {}
    for query in projections:
        execution = int(query["execution_ordinal"])
        for scope in query["scope_projections"]:
            nodes_by_fit.setdefault(scope["fit_id"], []).append(
                (query, scope, metadata[execution])
            )
    v92_by_fit = {item["fit_id"]: item for item in feasibility["scope_summaries"]}
    require(set(nodes_by_fit) == set(v92_by_fit) and len(nodes_by_fit) == 16, "scope set drift")

    scope_outputs = []
    total_static_edges = total_geometry_edges = total_matched = 0
    total_static_zero = total_geometry_zero = total_hall = 0
    for fit_id in sorted(nodes_by_fit):
        nodes = sorted(
            nodes_by_fit[fit_id],
            key=lambda item: (
                int(item[0]["execution_ordinal"]),
                item[0]["query_id"],
                item[0]["query_source_image_sha256"],
            ),
        )
        v92 = v92_by_fit[fit_id]
        membership_path = ROOT / v92["membership_path"]
        membership = read(membership_path, f"membership {fit_id}", mode444=True)
        require(
            file_sha(membership_path) == v92["membership_sha256"]
            and membership.get("logical_sha256") == v92["membership_logical_sha256"]
            and membership.get("logical_sha256") == logical(membership),
            "membership binding drift",
        )
        member_executions = sorted(int(item["execution_ordinal"]) for item in membership["heldout_addresses"])
        node_executions = [int(item[0]["execution_ordinal"]) for item in nodes]
        require(member_executions == node_executions, "membership/projection scope drift")
        static_adjacency: list[list[int]] = []
        geometry_adjacency: list[list[int]] = []
        for query, scope, meta in nodes:
            static_edges = []
            geometry_edges = []
            for donor_index, (donor_query, donor_scope, donor_meta) in enumerate(nodes):
                legal_static = (
                    query["query_id"] != donor_query["query_id"]
                    and query["query_source_image_sha256"]
                    != donor_query["query_source_image_sha256"]
                    and meta["identity"] != donor_meta["identity"]
                    and meta["supergroup"] != donor_meta["supergroup"]
                    and meta["track"] == donor_meta["track"]
                )
                if legal_static:
                    static_edges.append(donor_index)
                    if (
                        scope["donor_geometry_key_sha256"]
                        == donor_scope["donor_geometry_key_sha256"]
                        and scope["p_checkpoint_sha256"]
                        == donor_scope["p_checkpoint_sha256"]
                    ):
                        geometry_edges.append(donor_index)
            static_adjacency.append(static_edges)
            geometry_adjacency.append(geometry_edges)
        static_matched = maximum_cardinality(static_adjacency)
        geometry_assignment = canonical_maximum_matching(geometry_adjacency)
        geometry_matched = sum(item is not None for item in geometry_assignment)
        require(
            sum(len(item) for item in static_adjacency) == v92["candidate_edge_count"]
            and sum(len(item) == 0 for item in static_adjacency)
            == v92["zero_degree_count"]
            and static_matched == v92["matched_count"]
            and len(nodes) - static_matched == v92["matching_deficit_count"],
            "V92 static graph reconstruction drift",
        )
        ledger_rows = []
        reason_counts = {
            "NULL_DONOR_STATIC_ZERO_DEGREE": 0,
            "NULL_DONOR_GEOMETRY_ZERO_DEGREE": 0,
            "NULL_DONOR_GEOMETRY_HALL_UNMATCHED": 0,
            "MATCHED_DONOR": 0,
        }
        for index, ((query, scope, _), donor_index) in enumerate(
            zip(nodes, geometry_assignment, strict=True)
        ):
            if donor_index is not None:
                donor_query, donor_scope, _ = nodes[donor_index]
                status = "MATCHED_DONOR"
                donor = {
                    "execution_ordinal": donor_query["execution_ordinal"],
                    "query_id": donor_query["query_id"],
                    "query_source_image_sha256": donor_query[
                        "query_source_image_sha256"
                    ],
                    "target_candidate_key": donor_scope["target_candidate_key"],
                    "target_candidate_physical_row": donor_scope[
                        "target_candidate_physical_row"
                    ],
                    "target_reference_source_sha256": donor_scope[
                        "target_reference_source_sha256"
                    ],
                    "direction_record_sha256": [
                        item["record_sha256"]
                        for item in donor_scope["direction_records"]
                    ],
                }
            else:
                donor = None
                if not static_adjacency[index]:
                    status = "NULL_DONOR_STATIC_ZERO_DEGREE"
                elif not geometry_adjacency[index]:
                    status = "NULL_DONOR_GEOMETRY_ZERO_DEGREE"
                else:
                    status = "NULL_DONOR_GEOMETRY_HALL_UNMATCHED"
            reason_counts[status] += 1
            ledger_rows.append(
                {
                    "fit_id": fit_id,
                    "namespace": v92["scope_kind"],
                    "recipient_execution_ordinal": query["execution_ordinal"],
                    "recipient_query_id": query["query_id"],
                    "recipient_query_source_image_sha256": query[
                        "query_source_image_sha256"
                    ],
                    "recipient_target_candidate_key": scope["target_candidate_key"],
                    "recipient_geometry_key_sha256": scope[
                        "donor_geometry_key_sha256"
                    ],
                    "recipient_direction_record_sha256": [
                        item["record_sha256"] for item in scope["direction_records"]
                    ],
                    "status": status,
                    "donor": donor,
                }
            )
        static_edges = sum(len(item) for item in static_adjacency)
        geometry_edges = sum(len(item) for item in geometry_adjacency)
        scope_output = {
            "fit_id": fit_id,
            "scope_kind": v92["scope_kind"],
            "outer_fold": v92["outer_fold"],
            "inner_heldout_fold": v92["inner_heldout_fold"],
            "node_count": len(nodes),
            "static_edge_count": static_edges,
            "static_zero_degree_count": sum(not item for item in static_adjacency),
            "static_maximum_matching_count": static_matched,
            "geometry_key_count": len(
                {item[1]["donor_geometry_key_sha256"] for item in nodes}
            ),
            "geometry_edge_count": geometry_edges,
            "geometry_zero_degree_count": sum(not item for item in geometry_adjacency),
            "geometry_maximum_matching_count": geometry_matched,
            "reason_counts": reason_counts,
            "ledger_rows": ledger_rows,
            "ledger_population_sha256": canonical(ledger_rows),
        }
        scope_outputs.append(scope_output)
        total_static_edges += static_edges
        total_geometry_edges += geometry_edges
        total_matched += geometry_matched
        total_static_zero += reason_counts["NULL_DONOR_STATIC_ZERO_DEGREE"]
        total_geometry_zero += reason_counts["NULL_DONOR_GEOMETRY_ZERO_DEGREE"]
        total_hall += reason_counts["NULL_DONOR_GEOMETRY_HALL_UNMATCHED"]
    total_nodes = sum(item["node_count"] for item in scope_outputs)
    status = (
        NO_EDGE_STATUS
        if total_geometry_edges == 0
        else "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PARTIAL_MATCHING_READY"
    )
    result = {
        "schema_version": "rc_dino_rcde_sr0_mt_v2_donor_matching_v1_20260820",
        "status": status,
        "claim_level": "ENGINEERING_FROZEN_EXACT_GEOMETRY_DONOR_MATCHING_ONLY",
        "authority_sha256": authority_sha,
        "authority_logical_sha256": authority["logical_sha256"],
        "query_count": 594,
        "scope_count": 16,
        "scope_node_count": total_nodes,
        "direction_record_count": 4752,
        "static_edge_count": total_static_edges,
        "geometry_edge_count": total_geometry_edges,
        "matched_node_count": total_matched,
        "unmatched_node_count": total_nodes - total_matched,
        "null_static_zero_degree_count": total_static_zero,
        "null_geometry_zero_degree_count": total_geometry_zero,
        "null_geometry_hall_unmatched_count": total_hall,
        "scope_results": scope_outputs,
        "scope_population_sha256": canonical(scope_outputs),
        "donor_family_decision": (
            "NO_GO_EXACT_GEOMETRY_EMPTY_GRAPH"
            if total_geometry_edges == 0
            else "ELIGIBLE_PARTIAL_EXACT_GEOMETRY_DONOR_GRAPH"
        ),
        "donor_ledger_eligible": total_matched > 0,
        "repaired_relative_shared_matched_population": True,
        "identity_supergroup_serialized": False,
        "projection_shard_read_count": 50,
        "projection_validation_shard_read_count": 50,
        "membership_manifest_read_count": 16,
        "optimization_identity_supergroup_read_count": 594,
        "model_load_count": 0,
        "model_forward_count": 0,
        "training_count": 0,
        "score_rank_winner_gap_outcome_read_count": 0,
        "opened_sealed_c8_s8_read_count": 0,
        "V_training_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    require(total_nodes == 2376, "global scope node count drift")
    result["logical_sha256"] = logical(result)
    return result


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
    parser.add_argument("--output", type=Path, default=ROOT / OUT)
    args = parser.parse_args()
    require(args.output.resolve(strict=False) == (ROOT / OUT).resolve(strict=False), "output drift")
    authority, authority_sha = get_authority(args.authority)
    validate_authority_scope(authority)
    for key, name in (
        ("scheduler_migration_receipt", "migration receipt"),
        ("scheduler_migration_receipt_producer", "migration receipt producer"),
        ("scheduler_override_addendum", "scheduler override addendum"),
        ("moved_launcher", "moved launcher"),
        ("donor_contract", "donor contract"),
        ("reducer_contract", "reducer contract"),
        ("producer", "matching producer"),
        ("validator", "matching validator"),
        ("freezer", "V114 freezer"),
        ("launcher", "matching launcher"),
        ("test", "matching test"),
    ):
        bound_path(authority, key, name)
    result = build_result(authority, authority_sha)
    atomic(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scope_node_count": result["scope_node_count"],
                "geometry_edge_count": result["geometry_edge_count"],
                "matched_node_count": result["matched_node_count"],
                "logical_sha256": result["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
