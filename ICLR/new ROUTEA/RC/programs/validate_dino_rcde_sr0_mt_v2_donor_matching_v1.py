#!/usr/bin/env python3
"""Independent validator for the frozen exact-geometry donor graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v114_20260820.json"
RESULT = Path("results/dino_rcde_sr0_mt_v2_donor_matching_v1/result.json")
OUT = Path("results/dino_rcde_sr0_mt_v2_donor_matching_validation_v1/result.json")
PROJECTION_ROOT = Path(
    "results/dino_rcde_sr0_mt_v2_donor_geometry_projection_v1"
)


class IndependentDonorError(RuntimeError):
    pass


def check(condition: Any, message: str) -> None:
    if not condition:
        raise IndependentDonorError(message)


def canon(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canon({key: item for key, item in value.items() if key != "logical_sha256"})


def fsha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, name: str, *, mode444: bool = False) -> dict[str, Any]:
    check(path.is_file() and not path.is_symlink(), f"{name} absent")
    if mode444:
        check(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{name} mode drift")
    value = json.loads(path.read_text())
    check(isinstance(value, dict), f"{name} object drift")
    return value


def authority(path: Path) -> tuple[dict[str, Any], str]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = []
    for item in (ROOT / "registry").glob("current_authority_v*_*.json"):
        match = pattern.fullmatch(item.name)
        if match and item.is_file() and not item.is_symlink():
            rows.append((int(match.group(1)), item.resolve()))
    exact = (ROOT / AUTH).resolve(strict=True)
    check(
        rows
        and max(version for version, _ in rows) == 114
        and [item for version, item in rows if version == 114] == [exact]
        and path.resolve(strict=True) == exact,
        "V114 authority drift",
    )
    value = read(exact, "V114 authority", mode444=True)
    check(
        value.get("status") == "RCDE_SR0_MT_V2_DONOR_MATCHING_REDUCER_AUTHORIZED"
        and value.get("logical_sha256") == logical(value),
        "V114 authority envelope drift",
    )
    return value, fsha(exact)


def bound(authority_value: Mapping[str, Any], key: str, name: str) -> dict[str, Any]:
    binding = authority_value["bindings"][key]
    path = ROOT / binding["path"]
    value = read(path, name, mode444=binding.get("immutable", False))
    check(
        path.stat().st_size == binding["bytes"]
        and fsha(path) == binding["sha256"],
        f"{name} binding drift",
    )
    if "logical_sha256" in binding:
        check(value.get("logical_sha256") == binding["logical_sha256"] == logical(value), f"{name} logical drift")
    return value


def bound_path(authority_value: Mapping[str, Any], key: str, name: str) -> None:
    binding = authority_value.get("bindings", {}).get(key)
    check(isinstance(binding, Mapping) and "path" in binding, f"{name} binding absent")
    path = ROOT / binding["path"]
    check(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == binding["bytes"]
        and fsha(path) == binding["sha256"],
        f"{name} runtime binding drift",
    )


def validate_authority_scope(authority_value: Mapping[str, Any]) -> None:
    check(
        {key for key, value in authority_value.items() if value is True}
        == {
            "projection_receipt_read_authorized",
            "scheduler_migration_adoption_authorized",
            "optimization_identity_supergroup_read_authorized",
            "final_donor_matching_authorized",
        }
        and authority_value.get("lock_payload_deserialization_authorized") is False
        and authority_value.get("model_load_authorized") is False
        and authority_value.get("training_authorized") is False
        and authority_value.get("V_training_authorized") is False
        and authority_value.get("heldout_scoring_authorized") is False
        and authority_value.get("protected_access_authorized") is False,
        "V114 exact permission scope drift",
    )


def hopcroft_like(adjacency: list[list[int]]) -> int:
    right: dict[int, int] = {}

    def visit(left: int, seen: set[int]) -> bool:
        for donor in adjacency[left]:
            if donor in seen:
                continue
            seen.add(donor)
            if donor not in right or visit(right[donor], seen):
                right[donor] = left
                return True
        return False

    return sum(visit(left, set()) for left in range(len(adjacency)))


def metadata(manifest: Mapping[str, Any], folds: Mapping[str, Any], roles: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    fold_by_id = {item["query_id"]: item for item in folds["records"]}
    role_by_id = {item["query_id"]: item for item in roles["records"]}
    result = {}
    for query in manifest["query_index"]:
        fold = fold_by_id[query["query_id"]]
        role = role_by_id[query["query_id"]]
        check(
            fold["source_image_sha256"] == query["query_source_image_sha256"]
            and fold["query_ordinal"] == role["query_ordinal"]
            and fold["track"] == role["track"],
            "metadata join drift",
        )
        result[int(query["execution_ordinal"])] = {
            "identity": role["identity"],
            "supergroup": role["supergroup"],
            "track": role["track"],
        }
    check(len(result) == 594, "metadata population drift")
    return result


def projection_rows(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for shard in range(50):
        start, stop = shard * 12, shard * 12 + 12
        pp = ROOT / PROJECTION_ROOT / "producer" / f"shard_{start:03d}_{stop:03d}.json"
        vp = ROOT / PROJECTION_ROOT / "validation" / f"shard_{start:03d}_{stop:03d}.json"
        producer = read(pp, f"producer {shard}", mode444=True)
        validation = read(vp, f"validation {shard}", mode444=True)
        check(
            producer.get("status")
            == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_READY"
            and validation.get("status")
            == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_PROJECTION_SHARD_ROLE_VALIDATED_INDEPENDENT_VALIDATION_PASS"
            and validation.get("validation_pass") is True
            and producer.get("authority_logical_sha256") == parent["logical_sha256"]
            and validation.get("projection_sha256") == fsha(pp)
            and validation.get("projection_logical_sha256") == producer["logical_sha256"],
            "projection receipt drift",
        )
        rows.extend(producer["query_projections"])
    check(len(rows) == 594 and len({item["execution_ordinal"] for item in rows}) == 594, "projection union drift")
    return rows


def validate_migration(
    authority_value: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    replay_sacct: bool,
) -> dict[int, dict[str, Any]]:
    migration = bound(authority_value, "scheduler_migration_receipt", "migration receipt")
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
    check(
        set(migration) == expected_fields
        and migration.get("schema_version")
        == "rc_dino_rcde_sr0_mt_v2_projection_scheduler_migration_receipt_v1_20260820"
        and migration.get("status")
        == "RCDE_SR0_MT_V2_PROJECTION_SCHEDULER_MIGRATION_INPUT_READY"
        and migration.get("logical_sha256") == logical(migration)
        and migration.get("authority_sha256")
        == authority_value["bindings"]["parent_authority_v113"]["sha256"]
        and migration.get("authority_logical_sha256") == parent["logical_sha256"]
        and isinstance(rows, list)
        and len(rows) == 50
        and migration.get("shard_population_sha256") == canon(rows)
        and migration.get("projection_logic_changed") is False
        and migration.get("model_load_count") == 0
        and migration.get("training_count") == 0
        and migration.get("scientific_GO_or_NO_GO") is None
        and migration.get("automatic_stage_advance") is False
        and migration.get("next_authorized_stage") is None
        and migration.get("moved_launcher_sha256")
        == authority_value["bindings"]["moved_launcher"]["sha256"]
        and migration.get("moved_launcher_path")
        == authority_value["bindings"]["moved_launcher"]["path"]
        and migration.get("scheduler_override_addendum_sha256")
        == authority_value["bindings"]["scheduler_override_addendum"]["sha256"]
        and migration.get("scheduler_override_addendum_path")
        == authority_value["bindings"]["scheduler_override_addendum"]["path"]
        and migration.get("migration_stop_sha256")
        == authority_value["bindings"]["migration_stop"]["sha256"]
        and migration.get("migration_stop_path")
        == authority_value["bindings"]["migration_stop"]["path"],
        "independent migration envelope drift",
    )
    job_ids = sorted({str(row["array_job_id"]) for row in rows})
    live = {}
    if replay_sacct:
        completed = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(job_ids),
                "--format=JobID,State,ExitCode,Partition",
                "--parsable2",
                "--noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        check(completed.returncode == 0, "independent sacct replay failed")
        for line in completed.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 4 and "." not in fields[0]:
                live[fields[0]] = (fields[1], fields[2], fields[3])
    by_shard = {}
    for ordinal, row in enumerate(rows):
        check(
            set(row) == row_fields
            and row["shard_ordinal"] == ordinal
            and set(row["accounting"]) == accounting_fields,
            "independent migration row schema drift",
        )
        route = (
            "DEV_SERIAL"
            if ordinal <= 10
            else "CPU_MIGRATED_11_19"
            if ordinal <= 19
            else "CPU_ORIGINAL_20_49"
        )
        accounting = row["accounting"]
        expected_producer_path = (
            PROJECTION_ROOT
            / "producer"
            / f"shard_{ordinal * 12:03d}_{ordinal * 12 + 12:03d}.json"
        ).as_posix()
        expected_validation_path = (
            PROJECTION_ROOT
            / "validation"
            / f"shard_{ordinal * 12:03d}_{ordinal * 12 + 12:03d}.json"
        ).as_posix()
        check(
            row["route"] == route
            and row["producer_path"] == expected_producer_path
            and row["validation_path"] == expected_validation_path,
            "migration route/canonical path drift",
        )
        if ordinal == 0:
            check(
                row["array_job_id"] == str(migration["dev_initial_job_id"])
                and accounting["job_id"] == str(migration["dev_initial_job_id"]),
                "initial dev job mapping drift",
            )
        elif ordinal <= 10:
            continuation_path = (
                ROOT
                / PROJECTION_ROOT
                / "continuation"
                / f"dev_submit_{ordinal:02d}.json"
            )
            continuation = read(
                continuation_path, f"dev continuation {ordinal}", mode444=True
            )
            check(
                continuation.get("status")
                == "DEV_NEXT_SHARD_SUBMISSION_COMMITTED"
                and continuation.get("logical_sha256") == logical(continuation)
                and continuation.get("next_shard_ordinal") == ordinal
                and continuation.get("submitted_job_id") == accounting["job_id"]
                and row["array_job_id"] == accounting["job_id"],
                "dev continuation job mapping drift",
            )
            if ordinal == 10:
                check(
                    accounting["job_id"]
                    == str(migration["dev_postvalidation_failed_job_id"]),
                    "dev10 top-level job mapping drift",
                )
        elif ordinal <= 19:
            migrated = str(migration["migrated_cpu_array_job_id"])
            check(
                row["array_job_id"] == migrated
                and accounting["job_id"] == f"{migrated}_{ordinal}",
                "migrated CPU job mapping drift",
            )
        else:
            original = str(migration["original_cpu_array_job_id"])
            check(
                row["array_job_id"] == original
                and accounting["job_id"] == f"{original}_{ordinal}",
                "original CPU job mapping drift",
            )
        if replay_sacct:
            check(accounting["job_id"] in live, "migration live job absent")
            check(
                live[accounting["job_id"]]
                == (
                    accounting["state"],
                    accounting["exit_code"],
                    "dev_cpuonly" if ordinal <= 10 else "cpuonly",
                ),
                "migration accounting replay drift",
            )
        if ordinal == 10:
            check(
                accounting["state"] == "FAILED" and accounting["exit_code"] == "1:0",
                "dev10 state drift",
            )
        else:
            check(
                accounting["state"] == "COMPLETED"
                and accounting["exit_code"] == "0:0",
                "migration completed state drift",
            )
        producer_path = ROOT / row["producer_path"]
        validation_path = ROOT / row["validation_path"]
        producer = read(producer_path, "migration producer", mode444=True)
        validation = read(validation_path, "migration validation", mode444=True)
        check(
            fsha(producer_path) == row["producer_sha256"]
            and producer.get("logical_sha256") == row["producer_logical_sha256"]
            and producer.get("logical_sha256") == logical(producer)
            and fsha(validation_path) == row["validation_sha256"]
            and validation.get("logical_sha256") == row["validation_logical_sha256"]
            and validation.get("logical_sha256") == logical(validation)
            and row["query_count"] == validation["query_count"]
            and row["scope_projection_count"] == validation["scope_projection_count"]
            and row["direction_record_decode_count"]
            == validation["direction_record_decode_count"],
            "migration output hash/count drift",
        )
        by_shard[ordinal] = row
    check(
        migration.get("query_count") == sum(row["query_count"] for row in rows) == 594
        and migration.get("scope_projection_count")
        == sum(row["scope_projection_count"] for row in rows)
        == 2376
        and migration.get("direction_record_decode_count")
        == sum(row["direction_record_decode_count"] for row in rows)
        == 4752,
        "independent migration global closure drift",
    )
    return by_shard


def validate_graph(
    authority_value: Mapping[str, Any],
    result: Mapping[str, Any],
    authority_sha: str,
    *,
    replay_sacct: bool = True,
) -> None:
    result_fields = {
        "schema_version", "status", "claim_level", "authority_sha256",
        "authority_logical_sha256", "query_count", "scope_count",
        "scope_node_count", "direction_record_count", "static_edge_count",
        "geometry_edge_count", "matched_node_count", "unmatched_node_count",
        "null_static_zero_degree_count", "null_geometry_zero_degree_count",
        "null_geometry_hall_unmatched_count", "scope_results",
        "scope_population_sha256", "donor_family_decision", "donor_ledger_eligible",
        "repaired_relative_shared_matched_population",
        "identity_supergroup_serialized", "projection_shard_read_count",
        "projection_validation_shard_read_count", "membership_manifest_read_count",
        "optimization_identity_supergroup_read_count", "model_load_count",
        "model_forward_count", "training_count",
        "score_rank_winner_gap_outcome_read_count", "opened_sealed_c8_s8_read_count",
        "V_training_authorized", "scientific_GO_or_NO_GO",
        "automatic_stage_advance", "next_authorized_stage", "logical_sha256",
    }
    scope_fields = {
        "fit_id", "scope_kind", "outer_fold", "inner_heldout_fold", "node_count",
        "static_edge_count", "static_zero_degree_count",
        "static_maximum_matching_count", "geometry_key_count",
        "geometry_edge_count", "geometry_zero_degree_count",
        "geometry_maximum_matching_count", "reason_counts", "ledger_rows",
        "ledger_population_sha256",
    }
    ledger_fields = {
        "fit_id", "namespace", "recipient_execution_ordinal", "recipient_query_id",
        "recipient_query_source_image_sha256", "recipient_target_candidate_key",
        "recipient_geometry_key_sha256", "recipient_direction_record_sha256",
        "status", "donor",
    }
    check(set(result) == result_fields, "donor result exact field-set drift")
    parent = bound(authority_value, "parent_authority_v113", "V113 authority")
    migration_rows = validate_migration(
        authority_value, parent, replay_sacct=replay_sacct
    )
    manifest = bound(authority_value, "v87_query_manifest", "V87 manifest")
    folds = bound(authority_value, "prejoin_folds", "prejoin folds")
    roles = bound(authority_value, "training_roles", "training roles")
    feasibility = bound(authority_value, "v92_feasibility", "V92 feasibility")
    meta = metadata(manifest, folds, roles)
    nodes_by_fit: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    projections = projection_rows(parent)
    # Tie the projection population used by matching to the adopted migration rows.
    check(
        set(migration_rows) == set(range(50))
        and sum(row["query_count"] for row in migration_rows.values())
        == len(projections),
        "migration/projection population drift",
    )
    for query in projections:
        for scope in query["scope_projections"]:
            nodes_by_fit.setdefault(scope["fit_id"], []).append((query, scope))
    v92 = {item["fit_id"]: item for item in feasibility["scope_summaries"]}
    check(set(nodes_by_fit) == set(v92), "scope population drift")
    scope_list = result.get("scope_results")
    check(
        isinstance(scope_list, list)
        and len(scope_list) == 16
        and [item["fit_id"] for item in scope_list] == sorted(nodes_by_fit)
        and len({item["fit_id"] for item in scope_list}) == 16,
        "donor result scope order/uniqueness drift",
    )
    result_scopes = {item["fit_id"]: item for item in scope_list}
    total_static = total_geometry = total_static_zero = total_geometry_zero = 0
    for fit_id, raw_nodes in nodes_by_fit.items():
        nodes = sorted(
            raw_nodes,
            key=lambda item: (
                item[0]["execution_ordinal"],
                item[0]["query_id"],
                item[0]["query_source_image_sha256"],
            ),
        )
        static_adj, geometry_adj = [], []
        for query, scope in nodes:
            source_meta = meta[query["execution_ordinal"]]
            static_edges, geometry_edges = [], []
            for donor_index, (donor_query, donor_scope) in enumerate(nodes):
                donor_meta = meta[donor_query["execution_ordinal"]]
                legal = (
                    query["query_id"] != donor_query["query_id"]
                    and query["query_source_image_sha256"]
                    != donor_query["query_source_image_sha256"]
                    and source_meta["identity"] != donor_meta["identity"]
                    and source_meta["supergroup"] != donor_meta["supergroup"]
                    and source_meta["track"] == donor_meta["track"]
                )
                if legal:
                    static_edges.append(donor_index)
                    if (
                        scope["donor_geometry_key_sha256"]
                        == donor_scope["donor_geometry_key_sha256"]
                        and scope["p_checkpoint_sha256"]
                        == donor_scope["p_checkpoint_sha256"]
                    ):
                        geometry_edges.append(donor_index)
            static_adj.append(static_edges)
            geometry_adj.append(geometry_edges)
        scope = result_scopes[fit_id]
        expected_ledger = []
        expected_reasons = {
            "NULL_DONOR_STATIC_ZERO_DEGREE": 0,
            "NULL_DONOR_GEOMETRY_ZERO_DEGREE": 0,
            "NULL_DONOR_GEOMETRY_HALL_UNMATCHED": 0,
            "MATCHED_DONOR": 0,
        }
        for index, (query, node_scope) in enumerate(nodes):
            expected_status = (
                "NULL_DONOR_STATIC_ZERO_DEGREE"
                if not static_adj[index]
                else "NULL_DONOR_GEOMETRY_ZERO_DEGREE"
            )
            expected_reasons[expected_status] += 1
            expected_ledger.append(
                {
                    "fit_id": fit_id,
                    "namespace": v92[fit_id]["scope_kind"],
                    "recipient_execution_ordinal": query["execution_ordinal"],
                    "recipient_query_id": query["query_id"],
                    "recipient_query_source_image_sha256": query[
                        "query_source_image_sha256"
                    ],
                    "recipient_target_candidate_key": node_scope[
                        "target_candidate_key"
                    ],
                    "recipient_geometry_key_sha256": node_scope[
                        "donor_geometry_key_sha256"
                    ],
                    "recipient_direction_record_sha256": [
                        item["record_sha256"]
                        for item in node_scope["direction_records"]
                    ],
                    "status": expected_status,
                    "donor": None,
                }
            )
        check(
            set(scope) == scope_fields
            and scope["scope_kind"] == v92[fit_id]["scope_kind"]
            and scope["outer_fold"] == v92[fit_id]["outer_fold"]
            and scope["inner_heldout_fold"] == v92[fit_id]["inner_heldout_fold"]
            and scope["node_count"] == len(nodes)
            and
            sum(map(len, static_adj)) == v92[fit_id]["candidate_edge_count"]
            and scope["static_edge_count"] == sum(map(len, static_adj))
            and scope["static_zero_degree_count"] == sum(not item for item in static_adj)
            and scope["static_maximum_matching_count"] == v92[fit_id]["matched_count"]
            and scope["geometry_key_count"]
            == len({item[1]["donor_geometry_key_sha256"] for item in nodes})
            and hopcroft_like(static_adj) == v92[fit_id]["matched_count"]
            and sum(map(len, geometry_adj)) == 0
            and hopcroft_like(geometry_adj) == 0
            and scope["geometry_edge_count"] == 0
            and scope["geometry_zero_degree_count"] == len(nodes)
            and scope["geometry_maximum_matching_count"] == 0
            and scope["reason_counts"] == expected_reasons
            and scope["ledger_rows"] == expected_ledger
            and all(set(item) == ledger_fields for item in scope["ledger_rows"])
            and scope["ledger_population_sha256"] == canon(expected_ledger),
            "independent scope graph drift",
        )
        total_static += sum(map(len, static_adj))
        total_geometry += sum(map(len, geometry_adj))
        total_static_zero += sum(not item for item in static_adj)
        total_geometry_zero += sum(bool(static_adj[i]) and not geometry_adj[i] for i in range(len(nodes)))
    check(
        result.get("schema_version")
        == "rc_dino_rcde_sr0_mt_v2_donor_matching_v1_20260820"
        and result.get("status")
        == "RCDE_SR0_MT_V2_DONOR_GEOMETRY_NO_ELIGIBLE_EDGE_STOP"
        and result.get("claim_level")
        == "ENGINEERING_FROZEN_EXACT_GEOMETRY_DONOR_MATCHING_ONLY"
        and result.get("authority_sha256") == authority_sha
        and result.get("authority_logical_sha256") == authority_value["logical_sha256"]
        and result.get("query_count") == 594
        and result.get("scope_count") == 16
        and result.get("scope_node_count") == 2376
        and result.get("direction_record_count") == 4752
        and result.get("static_edge_count") == total_static == 225792
        and result.get("geometry_edge_count") == total_geometry == 0
        and result.get("matched_node_count") == 0
        and result.get("unmatched_node_count") == 2376
        and result.get("null_static_zero_degree_count") == total_static_zero == 84
        and result.get("null_geometry_zero_degree_count") == total_geometry_zero == 2292
        and result.get("null_geometry_hall_unmatched_count") == 0
        and result.get("donor_family_decision")
        == "NO_GO_EXACT_GEOMETRY_EMPTY_GRAPH"
        and result.get("donor_ledger_eligible") is False
        and result.get("repaired_relative_shared_matched_population") is True
        and result.get("identity_supergroup_serialized") is False
        and result.get("projection_shard_read_count") == 50
        and result.get("projection_validation_shard_read_count") == 50
        and result.get("membership_manifest_read_count") == 16
        and result.get("optimization_identity_supergroup_read_count") == 594
        and result.get("model_load_count") == 0
        and result.get("model_forward_count") == 0
        and result.get("training_count") == 0
        and result.get("score_rank_winner_gap_outcome_read_count") == 0
        and result.get("opened_sealed_c8_s8_read_count") == 0
        and result.get("V_training_authorized") is False
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("automatic_stage_advance") is False
        and result.get("next_authorized_stage") is None,
        "global independent donor decision drift",
    )
    check(result.get("scope_population_sha256") == canon(scope_list), "scope population hash drift")

    def contains_private_key(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                key in {"identity", "supergroup"} or contains_private_key(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_private_key(item) for item in value)
        return False

    check(not contains_private_key(result), "raw identity/supergroup serialized")


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    check(not path.exists(), "validation output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    partial = Path(temporary)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True)
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
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / OUT)
    args = parser.parse_args()
    check(args.result.resolve(strict=True) == (ROOT / RESULT).resolve(strict=True), "result path drift")
    check(args.output.resolve(strict=False) == (ROOT / OUT).resolve(strict=False), "output path drift")
    authority_value, authority_sha = authority(args.authority)
    validate_authority_scope(authority_value)
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
        bound_path(authority_value, key, name)
    result = read(args.result, "donor result", mode444=True)
    check(result.get("logical_sha256") == logical(result), "donor result logical drift")
    validate_graph(authority_value, result, authority_sha)
    output = {
        "schema_version": "rc_dino_rcde_sr0_mt_v2_donor_matching_validation_v1_20260820",
        "status": "RCDE_SR0_MT_V2_DONOR_GEOMETRY_NO_ELIGIBLE_EDGE_INDEPENDENT_VALIDATION_PASS",
        "claim_level": "INDEPENDENT_FROZEN_EXACT_GEOMETRY_EMPTY_GRAPH_VALIDATION_ONLY",
        "validation_pass": True,
        "authority_sha256": authority_sha,
        "authority_logical_sha256": authority_value["logical_sha256"],
        "producer_result_sha256": fsha(args.result),
        "producer_result_logical_sha256": result["logical_sha256"],
        "scope_count": 16,
        "scope_node_count": 2376,
        "static_edge_count": 225792,
        "geometry_edge_count": 0,
        "matched_node_count": 0,
        "null_static_zero_degree_count": 84,
        "null_geometry_zero_degree_count": 2292,
        "null_geometry_hall_unmatched_count": 0,
        "donor_family_decision": "NO_GO_EXACT_GEOMETRY_EMPTY_GRAPH",
        "donor_ledger_eligible": False,
        "V_training_authorized": False,
        "model_load_count": 0,
        "training_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    output["logical_sha256"] = logical(output)
    atomic(args.output, output)
    print(json.dumps({key: output[key] for key in ("status", "scope_node_count", "geometry_edge_count", "matched_node_count", "logical_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
