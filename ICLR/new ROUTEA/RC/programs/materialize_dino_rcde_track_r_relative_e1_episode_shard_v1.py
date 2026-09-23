#!/usr/bin/env python3
"""Materialize one compact Track-R inner-OOF episode shard.

This stage reads only the already validated score-free loss-role join and the
immutable Natural/Core P-lock V2 artifacts.  It selects target/rival lock bytes
for loss use, but loads no DINO tensor, model, score, rank, result, donor, or
protected endpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))

from rc_aslo_xf.dino_rcde_track_r_episode_v1 import (  # noqa: E402
    SOURCE_FOLDS,
    build_compact_episode,
    index_complete_artifact_records,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (  # noqa: E402
    inner_oof_head_spec,
)


AUTHORITY_PATH = "registry/current_authority_v118_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v118_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARDS_AUTHORIZED"
SHARD_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_shard_v1_20260821"
SHARD_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_READY"
LOCK_ARTIFACT_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_formal_lock_artifact_v1_20260819"
LOSS_JOIN_STATUS = "RCDE_SR0_MT_LOSS_JOIN_READY"
LOSS_JOIN_VALIDATION_STATUS = "RCDE_SR0_MT_LOSS_JOIN_INDEPENDENT_VALIDATION_PASS"
SHARD_SIZE = 12
SHARD_COUNT = 50
QUERY_COUNT = 600
EXCLUDED = (25, 26, 101, 346, 354, 470)
WORKER_COUNT = 4
OLD_V118_SHA256 = "2cd0954ca31804fa804c6fb6d8fed166200ff17d153598e0d5de0343eea4bc3b"
OLD_V118_LOGICAL_SHA256 = "18081218e3e06095eb74c7d1599e61608c1d9082438b23abdab7e4215496af87"
PRODUCER_PATH = "programs/materialize_dino_rcde_track_r_relative_e1_episode_shard_v1.py"
VALIDATOR_PATH = "programs/validate_dino_rcde_track_r_relative_e1_episode_shard_v1.py"
PACKAGE_INITIALIZER_PATH = "src/rc_aslo_xf/__init__.py"
PACKAGE_INITIALIZER_IMPORT_PATHS = (
    "src/rc_aslo_xf/core.py",
    "src/rc_aslo_xf/local_runtime.py",
    "src/rc_aslo_xf/verifier.py",
)
FORMAL_PRODUCER_ROOT = "results/dino_rcde_track_r_relative_e1_shards_v1/producer"
PERFCHECK_PRODUCER_ROOT = "results/dino_rcde_track_r_relative_e1_perfcheck_v1/producer"
RUNTIME_MODULE_PATHS = (
    "src/rc_aslo_xf/colnomic_proposal_tokens.py",
    "src/rc_aslo_xf/cw0_connected_region_v2.py",
    "src/rc_aslo_xf/cw0_connected_window_v1.py",
    "src/rc_aslo_xf/cw1_sr0_natural_superregion_v2.py",
    "src/rc_aslo_xf/cw1_sr0_structure_v1.py",
    "src/rc_aslo_xf/dino_rcde_colnomic_superregion_v1.py",
    "src/rc_aslo_xf/dino_rcde_cw1_multitile_pseal_e1_v1.py",
    "src/rc_aslo_xf/dino_rcde_cw1_multitile_pseal_sources_v1.py",
    "src/rc_aslo_xf/dino_rcde_cw1_multitile_sr0_p_v1.py",
    "src/rc_aslo_xf/dino_rcde_cw1_multitile_superregion_v2.py",
    "src/rc_aslo_xf/dino_rcde_cw1_multitile_vdecode_v1.py",
    "src/rc_aslo_xf/dino_rcde_sr0_mt_p_lock_v2.py",
    "src/rc_aslo_xf/dino_rcde_sr0_mt_p_natural_adapter_v2.py",
    "src/rc_aslo_xf/dino_rcde_sr0_mt_p_natural_source_v1.py",
    "src/rc_aslo_xf/dino_rcde_sr0_mt_p_runtime_v1.py",
    "src/rc_aslo_xf/dino_rcde_sr0_mt_p_selector_v2.py",
    "src/rc_aslo_xf/dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1.py",
    "src/rc_aslo_xf/dino_rcde_sr0_mt_v_runtime_v1.py",
    "src/rc_aslo_xf/dino_rcde_track_r_episode_v1.py",
    "src/rc_aslo_xf/dino_rcde_v1_2_resource_core.py",
    "src/rc_aslo_xf/geometry_hypothesis_v1.py",
    "src/rc_aslo_xf/lt_hyp_pvlock_v4.py",
    "src/rc_aslo_xf/r0_crossview_cycle_v1.py",
)


class TrackRE1ShardError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRE1ShardError(message)


def _initialize_worker() -> None:
    """Pin each spawned query worker to one numerical thread."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def safe_path(path: Path, *, must_exist: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    if must_exist:
        require(value.is_file() and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(isinstance(item, Mapping) and set(item) >= {"path", "sha256"}, f"binding absent: {name}")
    path = safe_path(RC_ROOT / str(item["path"]))
    require(file_sha256(path) == item["sha256"], f"binding hash drift: {name}")
    return path


def _verify_file_binding(binding: Any, expected_relative: str) -> None:
    require(isinstance(binding, Mapping), f"runtime binding absent: {expected_relative}")
    require(binding.get("path") == expected_relative, f"runtime binding path drift: {expected_relative}")
    path = safe_path(RC_ROOT / expected_relative)
    require(
        binding.get("bytes") == path.stat().st_size
        and binding.get("sha256") == file_sha256(path),
        f"runtime binding hash drift: {expected_relative}",
    )


def validate_revision2_runtime(value: Mapping[str, Any]) -> None:
    """Fail closed on every local executable input of the formal runtime."""

    require(
        value.get("authority_version") == 118
        and value.get("authority_revision") == 2
        and value.get("supersedes")
        == {
            "authority_version": 118,
            "authority_revision": 1,
            "physical_sha256": OLD_V118_SHA256,
            "logical_sha256": OLD_V118_LOGICAL_SHA256,
            "status": AUTHORITY_STATUS,
        },
        "formal V118 revision-2 lineage drift",
    )
    runtime = value.get("parallel_runtime_contract")
    require(
        isinstance(runtime, Mapping)
        and runtime.get("worker_process_count") == WORKER_COUNT
        and runtime.get("threads_per_worker") == 1
        and runtime.get("process_start_method") == "spawn"
        and runtime.get("deterministic_parent_sort_required") is True
        and runtime.get("atomic_output_required") is True
        and runtime.get("full_record_validation_preserved_in_producer") is True
        and runtime.get("full_record_validation_preserved_in_independent_validator") is True
        and runtime.get("serial_payload_equivalence_required") is True
        and isinstance(runtime.get("qualified_total_seconds"), int)
        and 0 < runtime["qualified_total_seconds"] <= 3600,
        "formal parallel runtime contract drift",
    )
    resource = value.get("resource_contract")
    require(
        isinstance(resource, Mapping)
        and resource.get("cpus_per_task") == WORKER_COUNT
        and resource.get("worker_process_count") == WORKER_COUNT
        and resource.get("threads_per_worker") == 1
        and resource.get("thread_environment")
        == {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "formal resource/thread contract drift",
    )
    bindings = value.get("bindings")
    require(isinstance(bindings, Mapping), "formal runtime bindings absent")
    _verify_file_binding(bindings.get("producer"), PRODUCER_PATH)
    _verify_file_binding(bindings.get("independent_shard_validator"), VALIDATOR_PATH)
    _verify_file_binding(bindings.get("package_initializer"), PACKAGE_INITIALIZER_PATH)
    package_closure = bindings.get("package_initializer_import_closure")
    require(
        isinstance(package_closure, Mapping)
        and package_closure.get("root") == PACKAGE_INITIALIZER_PATH
        and package_closure.get("module_count")
        == len(PACKAGE_INITIALIZER_IMPORT_PATHS),
        "package initializer import closure envelope drift",
    )
    package_rows = package_closure.get("modules")
    require(
        isinstance(package_rows, list)
        and [row.get("path") for row in package_rows]
        == list(PACKAGE_INITIALIZER_IMPORT_PATHS)
        and package_closure.get("module_sequence_sha256")
        == logical_sha256({"rows": package_rows}),
        "package initializer import closure sequence drift",
    )
    for relative, row in zip(PACKAGE_INITIALIZER_IMPORT_PATHS, package_rows):
        require(
            row.get("module")
            == relative.removeprefix("src/").removesuffix(".py").replace("/", "."),
            f"package initializer import module drift: {relative}",
        )
        _verify_file_binding(row, relative)
    package_edges = package_closure.get("import_edges")
    require(
        isinstance(package_edges, list)
        and package_closure.get("import_edge_sequence_sha256")
        == logical_sha256({"rows": package_edges}),
        "package initializer import-edge receipt drift",
    )
    closure = bindings.get("implementation_ast_closure")
    require(
        isinstance(closure, Mapping)
        and closure.get("root_programs") == [PRODUCER_PATH, VALIDATOR_PATH]
        and closure.get("module_count") == len(RUNTIME_MODULE_PATHS),
        "formal runtime closure envelope drift",
    )
    rows = closure.get("modules")
    require(
        isinstance(rows, list)
        and [row.get("path") for row in rows] == list(RUNTIME_MODULE_PATHS)
        and closure.get("module_sequence_sha256") == logical_sha256({"rows": rows}),
        "formal runtime closure sequence drift",
    )
    for relative, row in zip(RUNTIME_MODULE_PATHS, rows):
        require(
            row.get("module")
            == relative.removeprefix("src/").removesuffix(".py").replace("/", "."),
            f"formal runtime module name drift: {relative}",
        )
        _verify_file_binding(row, relative)
    edges = closure.get("import_edges")
    require(
        isinstance(edges, list)
        and closure.get("import_edge_sequence_sha256")
        == logical_sha256({"rows": edges}),
        "formal runtime import-edge receipt drift",
    )


def atomic_torch(path: Path, value: object) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), f"immutable shard exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, output)
    output.chmod(0o444)


def require_immutable_authority(path: Path) -> None:
    require(
        (path.stat().st_mode & 0o777) == 0o444,
        "V118 authority is not immutable",
    )


def validate_authority(
    path: Path,
    *,
    output_root: Path,
    shard_ordinal: int,
) -> tuple[dict[str, Any], str]:
    authority_path = safe_path(path)
    require(authority_path.relative_to(RC_ROOT).as_posix() == AUTHORITY_PATH, "V118 authority path drift")
    require_immutable_authority(authority_path)
    value = read_json(authority_path)
    authority_sha = file_sha256(authority_path)
    require(
        value.get("schema_version") == AUTHORITY_SCHEMA
        and value.get("status") == AUTHORITY_STATUS
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("episode_shard_materialization_authorized") is True
        and value.get("loss_role_join_read_authorized") is True
        and value.get("lock_payload_deserialization_authorized") is True
        and value.get("token_load_authorized") is False
        and value.get("model_load_authorized") is False
        and value.get("model_forward_authorized") is False
        and value.get("training_authorized") is False
        and value.get("protected_access_authorized") is False
        and value.get("automatic_stage_advance") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("next_authorized_stage") is None,
        "V118 authority boundary drift",
    )
    resolved_output_root = safe_path(output_root, must_exist=False)
    formal_root = (RC_ROOT / FORMAL_PRODUCER_ROOT).resolve(strict=False)
    perf_root = (RC_ROOT / PERFCHECK_PRODUCER_ROOT).resolve(strict=False)
    if value.get("authority_revision") == 2:
        require(resolved_output_root == formal_root, "revision-2 producer output root drift")
        validate_revision2_runtime(value)
    else:
        require(
            authority_sha == OLD_V118_SHA256
            and value.get("logical_sha256") == OLD_V118_LOGICAL_SHA256
            and shard_ordinal == 0
            and resolved_output_root == perf_root,
            "revision-1 authority is restricted to isolated shard-0 perfcheck",
        )
    return value, authority_sha


def _materialize_execution_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Independently load and fully validate one query artifact."""

    execution = int(payload["execution_ordinal"])
    fold = payload["fold"]
    aggregate_row = payload["aggregate_row"]
    role = payload["role"]
    source_fold = int(fold["inner_fold"])
    artifact_path = safe_path(RC_ROOT / str(aggregate_row["lock_artifact_path"]))
    require(
        file_sha256(artifact_path) == aggregate_row["lock_artifact_sha256"],
        "lock artifact physical hash drift",
    )
    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=True, mmap=True
    )
    require(
        isinstance(artifact, Mapping)
        and artifact.get("schema_version") == LOCK_ARTIFACT_SCHEMA
        and artifact.get("target_free") is True
        and artifact.get("execution_ordinal") == execution
        and artifact.get("source_fold") == source_fold
        and artifact.get("candidate_count") == 128
        and artifact.get("direction_count") == 2
        and artifact.get("head_count") == 4
        and artifact.get("record_count") == 1024,
        "lock artifact envelope drift",
    )
    records = artifact.get("records")
    require(
        isinstance(records, list) and len(records) == 1024,
        "lock artifact records absent",
    )
    record_query = records[0].get("query")
    require(
        isinstance(record_query, Mapping)
        and record_query.get("execution_ordinal") == execution
        and record_query.get("historical_query_ordinal") == fold.get("query_ordinal"),
        "lock record historical/execution provenance drift",
    )
    indexed = index_complete_artifact_records(records, source_fold=source_fold)
    episodes = []
    for outer_fold in SOURCE_FOLDS:
        if outer_fold == source_fold:
            continue
        spec = inner_oof_head_spec(
            outer_fold=outer_fold, source_fold=source_fold
        )
        episodes.append(
            build_compact_episode(
                selected_head_records=indexed[spec.fit_id],
                source_fold=source_fold,
                outer_fold=outer_fold,
                query_id=str(role["query_id"]),
                execution_ordinal=execution,
                query_source_image_sha256=str(role["query_source_image_sha256"]),
                track=str(fold["track"]),
                target_candidate_key=str(role["target_candidate_key"]),
                strongest_rival_candidate_key=str(role["rival_candidate_key"]),
                loss_role_record_sha256=str(role["role_record_sha256"]),
                loss_join_sha256=str(role["loss_join_sha256"]),
                lock_artifact_path=artifact_path.relative_to(RC_ROOT).as_posix(),
                lock_artifact_sha256=str(aggregate_row["lock_artifact_sha256"]),
            )
        )
    return {
        "execution_ordinal": execution,
        "episodes": episodes,
        "source_artifact": {
            "execution_ordinal": execution,
            "lock_artifact_path": artifact_path.relative_to(RC_ROOT).as_posix(),
            "lock_artifact_sha256": aggregate_row["lock_artifact_sha256"],
            "record_population_sha256": aggregate_row["record_population_sha256"],
        },
        "natural_validation_count": len(records),
    }


def materialize(authority_path: Path, shard_ordinal: int, output_root: Path) -> dict[str, Any]:
    require(0 <= shard_ordinal < SHARD_COUNT, "shard ordinal drift")
    authority, authority_sha = validate_authority(
        authority_path,
        output_root=output_root,
        shard_ordinal=shard_ordinal,
    )
    aggregate_path = binding_path(authority, "p_v2_lock_aggregate")
    loss_join_path = binding_path(authority, "loss_join")
    loss_validation_path = binding_path(authority, "loss_join_validation")
    fold_path = binding_path(authority, "fold_schedule")
    redacted_schedule_path = binding_path(authority, "redacted_schedule")
    aggregate = read_json(aggregate_path)
    loss_join = read_json(loss_join_path)
    loss_validation = read_json(loss_validation_path)
    folds = read_json(fold_path)
    redacted_schedule = read_json(redacted_schedule_path)
    require(
        aggregate.get("status") == "RCDE_SR0_MT_P_V2_FORMAL_LOCK_AGGREGATE_READY"
        and aggregate.get("query_count") == 594
        and aggregate.get("total_record_count") == 608256,
        "P-V2 aggregate drift",
    )
    require(
        loss_join.get("status") == LOSS_JOIN_STATUS
        and loss_join.get("episode_count") == 594
        and loss_join.get("excluded_execution_ordinals") == list(EXCLUDED)
        and loss_join.get("score_rank_outcome_read_count") == 0
        and loss_join.get("identity_read_count") == 0
        and loss_join.get("supergroup_read_count") == 0
        and loss_validation.get("status") == LOSS_JOIN_VALIDATION_STATUS
        and loss_validation.get("validation_pass") is True
        and loss_validation.get("join_file_sha256") == file_sha256(loss_join_path),
        "score-free loss-role join drift",
    )
    require(
        folds.get("status") == "DINO_RCDE_PREJOIN_FOLDS_600_PHYSICALLY_ISOLATED"
        and folds.get("query_count") == QUERY_COUNT
        and isinstance(folds.get("records"), list),
        "fold schedule drift",
    )
    aggregate_by_execution = {
        int(item["execution_ordinal"]): item for item in aggregate["rows"]
    }
    loss_by_execution = {
        int(item["execution_ordinal"]): item for item in loss_join["episodes"]
    }
    fold_records = tuple(folds["records"])
    require(
        len(fold_records) == QUERY_COUNT
        and len({int(item["query_ordinal"]) for item in fold_records}) == QUERY_COUNT,
        "600-query fold record population/historical-ordinal drift",
    )
    # The schedule record sequence is the frozen 0..599 execution axis.
    # query_ordinal is a sparse historical source address and must never index
    # lock/loss artifacts.
    fold_by_execution = dict(enumerate(fold_records))
    explicit_by_execution = {
        int(item["execution_ordinal"]): item
        for item in redacted_schedule["records"]
    }
    require(set(explicit_by_execution) == set(range(QUERY_COUNT)), "explicit redacted execution-axis drift")
    for execution, fold in fold_by_execution.items():
        explicit = explicit_by_execution[execution]
        require(
            explicit.get("historical_query_ordinal") == fold.get("query_ordinal")
            and explicit.get("query_id") == fold.get("query_id")
            and explicit.get("source_image_sha256") == fold.get("source_image_sha256")
            and explicit.get("heldout_fold") == fold.get("inner_fold"),
            f"fold-sequence/explicit-execution closure drift: {execution}",
        )
    require(len(aggregate_by_execution) == len(loss_by_execution) == 594, "594-query source index drift")
    require(len(fold_by_execution) == QUERY_COUNT, "600-query fold index drift")
    expected_eligible = set(range(QUERY_COUNT)) - set(EXCLUDED)
    require(
        set(aggregate_by_execution) == set(loss_by_execution) == expected_eligible,
        "eligible execution-axis population drift",
    )
    for execution, fold in fold_by_execution.items():
        if execution in EXCLUDED:
            continue
        aggregate_row = aggregate_by_execution[execution]
        role = loss_by_execution[execution]
        require(
            aggregate_row.get("query_id") == role.get("query_id") == fold.get("query_id")
            and aggregate_row.get("source_fold") == fold.get("inner_fold")
            and role.get("query_source_image_sha256") == fold.get("source_image_sha256"),
            f"execution-axis query/source/fold closure drift: {execution}",
        )

    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    episodes: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    per_outer = {str(fold): 0 for fold in SOURCE_FOLDS}
    lock_deserialization_count = 0
    natural_validation_count = 0
    work_items = []
    for execution in range(start, stop):
        fold = fold_by_execution[execution]
        source_fold = int(fold["inner_fold"])
        require(source_fold in SOURCE_FOLDS, "source fold drift")
        if execution in EXCLUDED:
            require(execution not in aggregate_by_execution and execution not in loss_by_execution, "excluded query leaked into loss/lock population")
            exclusions.append({"execution_ordinal": execution, "reason": "NATURAL_C128_TARGET_MISS"})
            continue
        require(execution in aggregate_by_execution and execution in loss_by_execution, "eligible query absent from loss/lock population")
        work_items.append(
            {
                "execution_ordinal": execution,
                "fold": fold,
                "aggregate_row": aggregate_by_execution[execution],
                "role": loss_by_execution[execution],
            }
        )
    if work_items:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(WORKER_COUNT, len(work_items)),
            mp_context=context,
            initializer=_initialize_worker,
        ) as executor:
            worker_results = list(
                executor.map(_materialize_execution_worker, work_items)
            )
        for item in sorted(worker_results, key=lambda value: value["execution_ordinal"]):
            episodes.extend(item["episodes"])
            source_artifacts.append(item["source_artifact"])
            lock_deserialization_count += 1
            natural_validation_count += int(item["natural_validation_count"])
            for episode in item["episodes"]:
                per_outer[str(episode["query"]["outer_fold"])] += 1
    episodes.sort(key=lambda item: (item["query"]["outer_fold"], item["query"]["execution_ordinal"]))
    value: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA,
        "status": SHARD_STATUS,
        "claim_level": "ENGINEERING_COMPACT_INNER_OOF_LOSS_ROLE_EPISODE_SHARD_ONLY",
        "shard_ordinal": shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "query_slot_count": stop - start,
        "eligible_query_count": len(source_artifacts),
        "excluded_query_count": len(exclusions),
        "episode_count": len(episodes),
        "per_outer_fold_episode_count": per_outer,
        "episodes": episodes,
        "episode_sequence_sha256": canonical_sha256([item["record_sha256"] for item in episodes]),
        "exclusions": exclusions,
        "source_artifacts": source_artifacts,
        "source_bindings": {
            "authority_sha256": authority_sha,
            "p_v2_lock_aggregate_sha256": file_sha256(aggregate_path),
            "loss_join_sha256": file_sha256(loss_join_path),
            "loss_join_validation_sha256": file_sha256(loss_validation_path),
            "fold_schedule_sha256": file_sha256(fold_path),
            "redacted_schedule_sha256": file_sha256(redacted_schedule_path),
        },
        "access_audit": {
            "lock_payload_deserialization_count": lock_deserialization_count,
            "natural_v2_record_validation_count": natural_validation_count,
            "loss_role_join_read_count": len(source_artifacts),
            "dino_token_load_count": 0,
            "model_load_count": 0,
            "model_forward_count": 0,
            "training_count": 0,
            "score_rank_winner_gap_outcome_read_count": 0,
            "identity_supergroup_read_count": 0,
            "donor_null_h0_hold_switch_read_count": 0,
            "protected_access_count": 0,
        },
        "training_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    output = safe_path(output_root, must_exist=False) / f"shard_{start:03d}_{stop:03d}.pt"
    atomic_torch(output, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--shard-ordinal", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.authority, args.shard_ordinal, args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "shard_ordinal": result["shard_ordinal"],
                "episode_count": result["episode_count"],
                "logical_sha256": result["logical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
