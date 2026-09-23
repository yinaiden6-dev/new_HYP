#!/usr/bin/env python3
"""Independently reconstruct one compact Track-R E1 episode shard."""

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
    validate_compact_episode,
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
VALIDATION_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_shard_validation_v1_20260821"
VALIDATION_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_INDEPENDENT_VALIDATION_PASS"
LOCK_ARTIFACT_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_formal_lock_artifact_v1_20260819"
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
FORMAL_VALIDATION_ROOT = "results/dino_rcde_track_r_relative_e1_shards_v1/validation"
PERFCHECK_PRODUCER_ROOT = "results/dino_rcde_track_r_relative_e1_perfcheck_v1/producer"
PERFCHECK_VALIDATION_ROOT = "results/dino_rcde_track_r_relative_e1_perfcheck_v1/validation"
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


class TrackRE1ShardValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRE1ShardValidationError(message)


def _initialize_worker() -> None:
    """Pin each independently spawned validation worker to one thread."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _validate_execution_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reopen and replay one execution's compact episodes."""

    execution = int(payload["execution_ordinal"])
    fold = payload["fold"]
    aggregate_row = payload["aggregate_row"]
    role = payload["role"]
    source_fold = int(fold["inner_fold"])
    artifact_path = safe_path(RC_ROOT / str(aggregate_row["lock_artifact_path"]))
    require(
        file_sha256(artifact_path) == aggregate_row["lock_artifact_sha256"],
        "lock artifact hash drift",
    )
    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=True, mmap=True
    )
    require(
        isinstance(artifact, Mapping)
        and artifact.get("schema_version") == LOCK_ARTIFACT_SCHEMA
        and artifact.get("execution_ordinal") == execution
        and artifact.get("source_fold") == source_fold
        and artifact.get("record_count") == 1024,
        "lock artifact envelope drift",
    )
    records = artifact.get("records")
    require(isinstance(records, list) and len(records) == 1024, "lock records absent")
    record_query = records[0].get("query")
    require(
        isinstance(record_query, Mapping)
        and record_query.get("execution_ordinal") == execution
        and record_query.get("historical_query_ordinal") == fold.get("query_ordinal"),
        "independent lock record provenance drift",
    )
    indexed = index_complete_artifact_records(records, source_fold=source_fold)
    reconstructed = []
    for outer_fold in SOURCE_FOLDS:
        if outer_fold == source_fold:
            continue
        spec = inner_oof_head_spec(
            outer_fold=outer_fold, source_fold=source_fold
        )
        expected = build_compact_episode(
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
        validate_compact_episode(expected)
        reconstructed.append(expected)
    return {
        "execution_ordinal": execution,
        "episodes": reconstructed,
        "source_artifact": {
            "execution_ordinal": execution,
            "lock_artifact_path": artifact_path.relative_to(RC_ROOT).as_posix(),
            "lock_artifact_sha256": aggregate_row["lock_artifact_sha256"],
            "record_population_sha256": aggregate_row["record_population_sha256"],
        },
        "natural_validation_count": len(records),
    }


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
    """Independently fail closed on every local formal-runtime input."""

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


def require_immutable_authority(path: Path) -> None:
    require(
        (path.stat().st_mode & 0o777) == 0o444,
        "V118 authority is not immutable",
    )


def validate_authority(
    authority_path: Path,
    *,
    shard_path: Path,
    output: Path,
    shard_ordinal: int,
) -> tuple[dict[str, Any], str]:
    authority_path = safe_path(authority_path)
    require(authority_path.relative_to(RC_ROOT).as_posix() == AUTHORITY_PATH, "authority path drift")
    require_immutable_authority(authority_path)
    authority = read_json(authority_path)
    authority_sha = file_sha256(authority_path)
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("logical_sha256") == logical_sha256(authority)
        and authority.get("episode_shard_validation_authorized") is True
        and authority.get("token_load_authorized") is False
        and authority.get("model_load_authorized") is False
        and authority.get("model_forward_authorized") is False
        and authority.get("training_authorized") is False
        and authority.get("protected_access_authorized") is False
        and authority.get("automatic_stage_advance") is False
        and authority.get("scientific_GO_or_NO_GO") is None
        and authority.get("next_authorized_stage") is None,
        "V118 validation authority drift",
    )
    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    filename = f"shard_{start:03d}_{stop:03d}"
    resolved_shard = safe_path(shard_path)
    resolved_output = safe_path(output, must_exist=False)
    formal_shard = (RC_ROOT / FORMAL_PRODUCER_ROOT / f"{filename}.pt").resolve(strict=False)
    formal_output = (RC_ROOT / FORMAL_VALIDATION_ROOT / f"{filename}.validation.json").resolve(strict=False)
    perf_shard = (RC_ROOT / PERFCHECK_PRODUCER_ROOT / "shard_000_012.pt").resolve(strict=False)
    perf_output = (RC_ROOT / PERFCHECK_VALIDATION_ROOT / "shard_000_012.validation.json").resolve(strict=False)
    if authority.get("authority_revision") == 2:
        require(
            resolved_shard == formal_shard and resolved_output == formal_output,
            "revision-2 validator canonical path drift",
        )
        validate_revision2_runtime(authority)
    else:
        require(
            authority_sha == OLD_V118_SHA256
            and authority.get("logical_sha256") == OLD_V118_LOGICAL_SHA256
            and shard_ordinal == 0
            and resolved_shard == perf_shard
            and resolved_output == perf_output,
            "revision-1 authority is restricted to isolated shard-0 perfcheck",
        )
    return authority, authority_sha


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), f"immutable validation exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    output.chmod(0o444)


def validate(
    authority_path: Path,
    shard_path: Path,
    shard_ordinal: int,
    output: Path,
) -> dict[str, Any]:
    require(0 <= shard_ordinal < SHARD_COUNT, "shard ordinal drift")
    authority, _ = validate_authority(
        authority_path,
        shard_path=shard_path,
        output=output,
        shard_ordinal=shard_ordinal,
    )
    authority_path = safe_path(authority_path)
    shard_path = safe_path(shard_path)
    value = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
    require(
        isinstance(value, Mapping)
        and value.get("schema_version") == SHARD_SCHEMA
        and value.get("status") == SHARD_STATUS
        and value.get("shard_ordinal") == shard_ordinal
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("training_authorized") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False,
        "producer shard envelope/hash drift",
    )

    aggregate_path = binding_path(authority, "p_v2_lock_aggregate")
    loss_join_path = binding_path(authority, "loss_join")
    loss_validation_path = binding_path(authority, "loss_join_validation")
    fold_path = binding_path(authority, "fold_schedule")
    redacted_schedule_path = binding_path(authority, "redacted_schedule")
    aggregate = read_json(aggregate_path)
    loss_join = read_json(loss_join_path)
    folds = read_json(fold_path)
    redacted_schedule = read_json(redacted_schedule_path)
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
        "independent fold record population/historical-ordinal drift",
    )
    fold_by_execution = dict(enumerate(fold_records))
    explicit_by_execution = {
        int(item["execution_ordinal"]): item
        for item in redacted_schedule["records"]
    }
    require(set(explicit_by_execution) == set(range(QUERY_COUNT)), "independent explicit execution-axis drift")
    for execution, fold in fold_by_execution.items():
        explicit = explicit_by_execution[execution]
        require(
            explicit.get("historical_query_ordinal") == fold.get("query_ordinal")
            and explicit.get("query_id") == fold.get("query_id")
            and explicit.get("source_image_sha256") == fold.get("source_image_sha256")
            and explicit.get("heldout_fold") == fold.get("inner_fold"),
            f"independent fold/explicit execution closure drift: {execution}",
        )
    expected_eligible = set(range(QUERY_COUNT)) - set(EXCLUDED)
    require(
        set(aggregate_by_execution) == set(loss_by_execution) == expected_eligible,
        "independent eligible execution-axis population drift",
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
            f"independent execution-axis closure drift: {execution}",
        )
    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    require(
        value.get("execution_start") == start
        and value.get("execution_stop") == stop
        and value.get("query_slot_count") == stop - start,
        "shard range drift",
    )
    observed_by_key = {
        (int(item["query"]["outer_fold"]), int(item["query"]["execution_ordinal"])): item
        for item in value.get("episodes", [])
    }
    require(len(observed_by_key) == value.get("episode_count"), "episode address collision")
    reconstructed: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    lock_deserialization_count = 0
    natural_validation_count = 0
    work_items = []
    for execution in range(start, stop):
        fold = fold_by_execution[execution]
        source_fold = int(fold["inner_fold"])
        if execution in EXCLUDED:
            require(execution not in aggregate_by_execution and execution not in loss_by_execution, "excluded query leaked")
            exclusions.append({"execution_ordinal": execution, "reason": "NATURAL_C128_TARGET_MISS"})
            continue
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
                executor.map(_validate_execution_worker, work_items)
            )
        for item in sorted(worker_results, key=lambda value: value["execution_ordinal"]):
            execution = int(item["execution_ordinal"])
            for expected in item["episodes"]:
                observed = observed_by_key.get(
                    (int(expected["query"]["outer_fold"]), execution)
                )
                require(
                    observed == expected,
                    "independent compact episode replay drift",
                )
                reconstructed.append(expected)
            source_artifacts.append(item["source_artifact"])
            lock_deserialization_count += 1
            natural_validation_count += int(item["natural_validation_count"])
    reconstructed.sort(key=lambda item: (item["query"]["outer_fold"], item["query"]["execution_ordinal"]))
    per_outer = {
        str(fold): sum(item["query"]["outer_fold"] == fold for item in reconstructed)
        for fold in SOURCE_FOLDS
    }
    require(
        value.get("episodes") == reconstructed
        and value.get("episode_sequence_sha256")
        == canonical_sha256([item["record_sha256"] for item in reconstructed])
        and value.get("exclusions") == exclusions
        and value.get("source_artifacts") == source_artifacts
        and value.get("eligible_query_count") == len(source_artifacts)
        and value.get("excluded_query_count") == len(exclusions)
        and value.get("episode_count") == len(reconstructed)
        and value.get("per_outer_fold_episode_count") == per_outer,
        "independent shard population/order drift",
    )
    require(
        value.get("source_bindings")
        == {
            "authority_sha256": file_sha256(authority_path),
            "p_v2_lock_aggregate_sha256": file_sha256(aggregate_path),
            "loss_join_sha256": file_sha256(loss_join_path),
            "loss_join_validation_sha256": file_sha256(loss_validation_path),
            "fold_schedule_sha256": file_sha256(fold_path),
            "redacted_schedule_sha256": file_sha256(redacted_schedule_path),
        },
        "independent shard source-binding drift",
    )
    access = value.get("access_audit")
    require(
        isinstance(access, Mapping)
        and access.get("lock_payload_deserialization_count") == lock_deserialization_count
        and access.get("natural_v2_record_validation_count") == natural_validation_count
        and access.get("loss_role_join_read_count") == len(source_artifacts)
        and all(
            access.get(field) == 0
            for field in (
                "dino_token_load_count",
                "model_load_count",
                "model_forward_count",
                "training_count",
                "score_rank_winner_gap_outcome_read_count",
                "identity_supergroup_read_count",
                "donor_null_h0_hold_switch_read_count",
                "protected_access_count",
            )
        ),
        "shard access audit drift",
    )
    validation: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "shard_ordinal": shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "producer_shard_sha256": file_sha256(shard_path),
        "producer_shard_logical_sha256": value["logical_sha256"],
        "eligible_query_count": len(source_artifacts),
        "excluded_query_count": len(exclusions),
        "episode_count": len(reconstructed),
        "per_outer_fold_episode_count": per_outer,
        "independent_lock_deserialization_count": lock_deserialization_count,
        "independent_natural_v2_record_validation_count": natural_validation_count,
        "token_load_count": 0,
        "model_load_count": 0,
        "training_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    validation["logical_sha256"] = logical_sha256(validation)
    atomic_json(output, validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--shard-ordinal", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.authority, args.shard, args.shard_ordinal, args.output)
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
