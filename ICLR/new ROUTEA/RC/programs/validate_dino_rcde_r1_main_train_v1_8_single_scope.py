#!/usr/bin/env python3
"""Independently validate one DINO-RCDE R1 V1.8 arm/fold scope.

This validator deliberately does not import the V1.8 training runner.  It
reconstructs the outer-train population, deterministic 2048-update schedule,
cache allowlist, runtime reads, atomic chunk chain, checkpoint and optimizer
state from frozen inputs.  The protocol must authorize exactly one arm and one
outer fold, so the same program can validate BAG folds 2--4 and later CONTEXT
folds without changing code.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

import validate_dino_rcde_r1_main_train_v1_7_contract_repair as common
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "rc_dino_rcde_r1_main_train_protocol_v1_8_single_scope_20260814"
QUERY_ORDER_NAMESPACE = "RCDE_QUERY_ORDER_V1_1"
EXPECTED_MODEL_SHA256 = (
    "f901d9bd056bb65e5fcb02c72e869f6d0cf8d7472467d22fbc340442feca034c"
)
EXPECTED_PARAMETERS = 65_125
UPDATES = 2_048
EPISODES_PER_UPDATE = 4
SEED = 17
QUERY_TILE_ROWS = 8
REFERENCE_TILE_ROWS = 16
ALLOWED_ARMS = {"RCDE_BAG", "RCDE_CONTEXT"}
PROTECTED_ZERO_KEYS = {
    "C8_runtime_read_count",
    "S8_runtime_read_count",
    "opened_runtime_read_count",
    "sealed_runtime_read_count",
    "unauthorized_natural_result_read_count",
    "home_files_modified",
}
CHUNK_DIRECTORY_NAME = re.compile(r"^chunk_job(.+)$")
COMMITTED_CHUNK_FILES = {
    "chunk.json",
    "input_preflight.json",
    "runtime_read_ledger.json",
    "resume_state.pt",
}

TOP_LEVEL_FIELDS = {
    "schema_version",
    "date",
    "stage",
    "claim_boundary",
    "authority",
    "bindings",
    "cache",
    "training",
    "resume",
    "execution",
    "artifact_contract",
    "forbidden",
    "decision",
    "automatic_stage_advance",
    "scientific_GO_or_NO_GO",
}
ARTIFACT_CONTRACT_KEYS = {
    "identity",
    "manifest",
    "preflight",
    "read_ledger",
    "chunk",
    "resume",
    "checkpoint",
    "result",
    "validation",
}
TRAINING_FIELDS = {
    "arms",
    "outer_folds",
    "seed",
    "parameter_count",
    "optimizer",
    "deterministic_algorithms",
    "learning_rate",
    "weight_decay",
    "gradient_clip_l2",
    "updates_total",
    "episodes_per_update",
    "raw_correct_per_update",
    "raw_wrong_per_update",
    "warmup_updates",
    "final_learning_rate",
    "loss",
    "query_tile_rows",
    "reference_tile_rows",
    "query_order_namespace",
    "query_order_key",
    "query_order_tie_break",
    "candidate_direction_source",
    "cache_query_key",
    "outer_train_only",
    "heldout_target_join_allowed",
}
EXECUTION_FIELDS = {
    "slurm_path",
    "slurm_sha256",
    "partition",
    "partition_max_time",
    "job_time",
    "runner_runtime_guard_seconds",
    "maximum_jobs_submitted_now",
    "maximum_running_tasks",
    "submission_topology",
    "array_used",
    "gpu_per_job",
    "chunk_commit",
    "committed_chunk_files",
    "finalization_recovery",
}
RESUME_FIELDS = {
    "save_at_every_job_boundary",
    "restore_model",
    "restore_optimizer",
    "restore_progress_and_loss_trace",
    "restore_python_numpy_torch_cuda_rng",
    "recompute_schedule_digest_prefix_from_frozen_ledger",
    "persistent_output",
    "concurrent_writer_lock",
    "completed_result_immutable",
}
FORBIDDEN_FIELDS = {
    "paths",
    "home_write",
    "other_fold_or_arm_training",
    "control_training",
    "heldout_label_join",
    "evaluation",
    "scientific_decision",
}
DECISION_FIELDS = {
    "completed_fold_pass",
    "next_on_pass",
    "incomplete_after_maximum_jobs",
    "automatic_stage_advance",
    "scientific_GO_or_NO_GO",
}
REQUIRED_BINDINGS = {
    "scientific_contract",
    "cache_protocol",
    "cache_result",
    "cache_validation",
    "training_roles",
    "episode_ledger",
    "resource_shape_ledger",
    "model_visible_c128",
    "rcde_core",
    "single_scope_runner",
    "single_scope_validator",
    "validation_common_v1_7",
}


class ValidationAbort(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationAbort(message)


def exact_fields(value: Any, fields: set[str], context: str) -> None:
    require(isinstance(value, dict), f"{context} is not an object")
    observed = set(value)
    require(
        observed == fields,
        f"{context} fields drift: missing={sorted(fields - observed)} "
        f"extra={sorted(observed - fields)}",
    )


def resolve_bound(path_text: str) -> Path:
    path = (ROOT / path_text).resolve()
    require(ROOT in path.parents, f"binding escapes RC root: {path}")
    lowered = {part.lower() for part in path.parts}
    require(
        not lowered.intersection({"c8", "s8", "opened", "sealed"}),
        f"protected path requested: {path}",
    )
    return path


def protected_zero_exact(value: Any, context: str) -> None:
    exact_fields(value, PROTECTED_ZERO_KEYS, f"{context} protected access")
    require(all(item == 0 for item in value.values()), f"{context} protected access nonzero")


def artifact_contract(protocol: Mapping[str, Any]) -> dict[str, str]:
    """Validate the protocol's nine schema/status pairs and flatten internally."""

    contract = protocol["artifact_contract"]
    exact_fields(contract, ARTIFACT_CONTRACT_KEYS, "artifact contract")
    flattened: dict[str, str] = {}
    for name in sorted(ARTIFACT_CONTRACT_KEYS):
        spec = contract[name]
        exact_fields(spec, {"schema_version", "status"}, f"artifact contract {name}")
        schema = spec["schema_version"]
        status = spec["status"]
        require(
            isinstance(schema, str)
            and "v1_8" in schema
            and isinstance(status, str)
            and bool(status),
            f"artifact contract value drift: {name}",
        )
        flattened[f"{name}_schema"] = schema
        flattened[f"{name}_status"] = status
    return flattened


def authorized_scope_member(
    authority: Mapping[str, Any],
    *,
    arm: str,
    fold: int,
    stage: str,
) -> Mapping[str, Any]:
    """Return the unique exact authority member for one protocol scope."""

    scopes = authority.get("authorized_scopes")
    require(isinstance(scopes, list) and scopes, "authority scope allowlist absent")
    scope_fields = {
        "stage",
        "arm",
        "outer_fold",
        "optimizer_updates_total",
        "seed",
        "maximum_submitted_jobs_in_chain",
        "maximum_concurrently_running_tasks",
    }
    scope_keys: set[tuple[str, int, str]] = set()
    matching: list[Mapping[str, Any]] = []
    for ordinal, scope in enumerate(scopes):
        exact_fields(scope, scope_fields, f"authority scope {ordinal}")
        require(scope["arm"] in ALLOWED_ARMS, "authority contains unsupported arm")
        require(scope["outer_fold"] in {1, 2, 3, 4}, "authority contains unsupported fold")
        key = (str(scope["arm"]), int(scope["outer_fold"]), str(scope["stage"]))
        require(key not in scope_keys, "authority contains duplicate scope")
        scope_keys.add(key)
        if key == (arm, fold, stage):
            matching.append(scope)
    require(len(matching) == 1, "protocol single scope is not one exact authorized member")
    return matching[0]


def validate_reverse_protocol_anchors(
    authority: Mapping[str, Any], protocol_path: Path
) -> None:
    """Validate the shared authority's exact, non-circular protocol anchors."""

    anchors = authority.get("execution_content_anchors", {}).get("protocols")
    require(isinstance(anchors, list) and anchors, "authority protocol reverse anchors absent")
    anchor_paths: set[Path] = set()
    matches: list[Mapping[str, Any]] = []
    for ordinal, anchor in enumerate(anchors):
        exact_fields(
            anchor,
            {"path", "required_schema", "digest_rule"},
            f"authority protocol reverse anchor {ordinal}",
        )
        path = resolve_bound(str(anchor["path"]))
        require(path not in anchor_paths, "authority protocol reverse anchor path duplicated")
        anchor_paths.add(path)
        require(anchor["required_schema"] == PROTOCOL_SCHEMA, "authority reverse schema drift")
        require(
            anchor["digest_rule"] == "protocol_embeds_exact_authority_sha256",
            "authority reverse digest rule drift",
        )
        if path == protocol_path.resolve():
            matches.append(anchor)
    require(len(matches) == 1, "current protocol does not uniquely match authority reverse anchors")


def validate_protocol(
    protocol_path: Path,
    authority_path: Path,
    requested_arm: str,
    requested_fold: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    protocol = common.read_json(protocol_path)
    authority = common.read_json(authority_path)
    exact_fields(protocol, TOP_LEVEL_FIELDS, "protocol")
    require(protocol["schema_version"] == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["automatic_stage_advance"] is False, "protocol auto-advance enabled")
    require(protocol["scientific_GO_or_NO_GO"] is None, "protocol scientific decision present")

    authority_spec = protocol["authority"]
    exact_fields(authority_spec, {"path", "sha256", "required_status"}, "authority binding")
    require(
        authority_path.resolve() == resolve_bound(str(authority_spec["path"])),
        "authority path drift",
    )
    authority_sha = common.file_sha256(authority_path)
    require(authority_sha == authority_spec["sha256"], "authority digest drift")
    require(authority.get("status") == authority_spec["required_status"], "authority status drift")
    require(authority.get("next_authorized_stage") == protocol["stage"], "authority stage drift")
    require(authority.get("natural_training_authorized") is True, "training authority absent")
    require(authority.get("automatic_stage_advance") is False, "authority auto-advance enabled")
    require(authority.get("scientific_GO_or_NO_GO") is None, "authority scientific decision present")
    require(
        authority.get("next_on_validation_pass") == protocol["decision"]["next_on_pass"],
        "authority validation-pass transition drift",
    )
    protected_zero_exact(authority.get("protected_access_counts"), "authority")

    training = protocol["training"]
    exact_fields(training, TRAINING_FIELDS, "training contract")
    require(training["arms"] == [requested_arm], "protocol is not the requested unique arm")
    require(training["outer_folds"] == [requested_fold], "protocol is not the requested unique fold")
    require(requested_arm in ALLOWED_ARMS, "unsupported arm")
    require(requested_fold in {1, 2, 3, 4}, "unsupported outer fold")
    expected_training = {
        "seed": SEED,
        "parameter_count": EXPECTED_PARAMETERS,
        "optimizer": "AdamW",
        "deterministic_algorithms": True,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "gradient_clip_l2": 1.0,
        "updates_total": UPDATES,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "raw_correct_per_update": 2,
        "raw_wrong_per_update": 2,
        "warmup_updates": 128,
        "final_learning_rate": 3.0e-5,
        "loss": "mean_softplus_negative_signed_pair_logit_tau_1",
        "query_tile_rows": QUERY_TILE_ROWS,
        "reference_tile_rows": REFERENCE_TILE_ROWS,
        "query_order_namespace": QUERY_ORDER_NAMESPACE,
        "query_order_key": "source_image_sha256",
        "query_order_tie_break": "stable_episode_ledger_order",
        "candidate_direction_source": "P0_frozen_model_candidate_order",
        "cache_query_key": "execution_ordinal",
        "outer_train_only": True,
        "heldout_target_join_allowed": False,
    }
    for key, expected in expected_training.items():
        require(training[key] == expected, f"training contract drift: {key}")

    matching_scope = authorized_scope_member(
        authority,
        arm=requested_arm,
        fold=requested_fold,
        stage=str(protocol["stage"]),
    )
    expected_scope = {
        "stage": protocol["stage"],
        "arm": requested_arm,
        "outer_fold": requested_fold,
        "optimizer_updates_total": UPDATES,
        "seed": SEED,
        "maximum_submitted_jobs_in_chain": protocol["execution"][
            "maximum_jobs_submitted_now"
        ],
        "maximum_concurrently_running_tasks": protocol["execution"][
            "maximum_running_tasks"
        ],
    }
    for key, expected in expected_scope.items():
        require(matching_scope.get(key) == expected, f"authority scope drift: {key}")

    cache = protocol["cache"]
    exact_fields(
        cache,
        {"root", "query_count", "reference_count", "query_key", "model_checkpoint_logical_sha256"},
        "cache contract",
    )
    require(cache["query_count"] == 600, "cache query count drift")
    require(cache["reference_count"] == 4_748, "cache reference count drift")
    require(cache["query_key"] == "execution_ordinal", "cache query key drift")
    require(cache["model_checkpoint_logical_sha256"] == EXPECTED_MODEL_SHA256, "cache model drift")

    artifact = artifact_contract(protocol)

    execution = protocol["execution"]
    exact_fields(execution, EXECUTION_FIELDS, "execution contract")
    require(
        type(execution["maximum_jobs_submitted_now"]) is int
        and execution["maximum_jobs_submitted_now"] == 1,
        "execution job budget drift",
    )
    require(
        execution["maximum_running_tasks"] == 3,
        "execution concurrency drift",
    )
    require(execution["partition"] == "accelerated", "execution partition drift")
    require(execution["partition_max_time"] == "2-00:00:00", "partition max-time drift")
    require(execution["job_time"] == "04:00:00", "job time drift")
    require(execution["runner_runtime_guard_seconds"] == 13_800, "runtime guard drift")
    require(execution["gpu_per_job"] == 1, "execution GPU count drift")
    require(execution["array_used"] is True, "execution array drift")
    require(
        execution["chunk_commit"] == "atomic_directory_rename"
        and execution["committed_chunk_files"] == sorted(COMMITTED_CHUNK_FILES)
        and execution["finalization_recovery"] is True,
        "atomic chunk contract drift",
    )
    slurm_path = resolve_bound(str(execution["slurm_path"]))
    require(slurm_path.is_file(), "bound Slurm entry absent")
    require(common.file_sha256(slurm_path) == execution["slurm_sha256"], "Slurm digest drift")

    exact_fields(protocol["resume"], RESUME_FIELDS, "resume contract")
    expected_resume_bools = {
        "save_at_every_job_boundary",
        "restore_model",
        "restore_optimizer",
        "restore_progress_and_loss_trace",
        "restore_python_numpy_torch_cuda_rng",
        "recompute_schedule_digest_prefix_from_frozen_ledger",
        "completed_result_immutable",
    }
    require(all(protocol["resume"][key] is True for key in expected_resume_bools), "resume guarantee drift")
    require(protocol["resume"]["concurrent_writer_lock"] == "flock_exclusive", "resume lock drift")
    exact_fields(protocol["forbidden"], FORBIDDEN_FIELDS, "forbidden contract")
    require(protocol["forbidden"]["home_write"] is False, "home write enabled")
    require(protocol["forbidden"]["other_fold_or_arm_training"] is False, "other scope enabled")
    require(protocol["forbidden"]["control_training"] is False, "control training enabled")
    require(protocol["forbidden"]["heldout_label_join"] is False, "heldout join enabled")
    require(protocol["forbidden"]["evaluation"] is False, "evaluation enabled")
    require(protocol["forbidden"]["scientific_decision"] is False, "scientific decision enabled")
    require(
        {str(item).lower() for item in protocol["forbidden"]["paths"]}
        >= {"c8", "s8", "opened", "sealed"},
        "protected path set incomplete",
    )
    decision = protocol["decision"]
    exact_fields(decision, DECISION_FIELDS, "decision contract")
    require(decision["completed_fold_pass"] == artifact["validation_status"], "decision PASS drift")
    require(decision["automatic_stage_advance"] is False, "decision auto-advance enabled")
    require(decision["scientific_GO_or_NO_GO"] is None, "decision contains scientific verdict")

    bindings = protocol["bindings"]
    require(isinstance(bindings, dict), "bindings absent")
    require(REQUIRED_BINDINGS.issubset(bindings), "required single-scope binding absent")
    binding_hashes: dict[str, str] = {}
    for name, binding in bindings.items():
        exact_fields(binding, {"path", "sha256"}, f"binding {name}")
        path = resolve_bound(str(binding["path"]))
        require(path.is_file(), f"bound file absent: {name}")
        observed = common.file_sha256(path)
        require(observed == binding["sha256"], f"binding digest drift: {name}")
        binding_hashes[name] = observed
    cache_validation = common.read_json(resolve_bound(bindings["cache_validation"]["path"]))
    require(cache_validation.get("status") == "DINO_RCDE_R1_CACHE_VALIDATION_PASS", "cache validation not PASS")
    validate_reverse_protocol_anchors(authority, protocol_path)
    return protocol, authority, binding_hashes


def expected_manifest(
    population: Mapping[str, Any], fold: int, artifact: Mapping[str, str]
) -> dict[str, Any]:
    value = {
        "schema_version": artifact["manifest_schema"],
        "status": artifact["manifest_status"],
        "outer_fold": fold,
        "query_order_namespace": QUERY_ORDER_NAMESPACE,
        "query_cache_key_field": "execution_ordinal",
        "historical_query_ordinal_model_visible": False,
        "eligible_query_execution_ordinals": population["eligible_queries"],
        "eligible_query_execution_ordinals_sha256": common.canonical_sha256(population["eligible_queries"]),
        "allowed_reference_physical_rows": population["allowed_references"],
        "allowed_reference_physical_rows_sha256": common.canonical_sha256(population["allowed_references"]),
        "heldout_reference_physical_rows": population["heldout_references"],
        "heldout_reference_physical_rows_sha256": common.canonical_sha256(population["heldout_references"]),
        "reference_row_intersection_count": 0,
        "eligible_query_count": len(population["eligible_queries"]),
        "allowed_reference_count": len(population["allowed_references"]),
        "heldout_reference_count": len(population["heldout_references"]),
        "all_episode_mapping_count": 1_800,
        "correct_pool_count": len(population["correct"]),
        "wrong_pool_count": len(population["wrong"]),
        "protected_access_counts": {key: 0 for key in sorted(PROTECTED_ZERO_KEYS)},
    }
    value["logical_sha256"] = common.canonical_sha256(value)
    return value


def validate_identity(
    path: Path,
    *,
    artifact: Mapping[str, str],
    arm: str,
    fold: int,
    protocol_sha: str,
    authority_sha: str,
) -> dict[str, Any]:
    value = common.read_json(path)
    expected = {
        "schema_version": artifact["identity_schema"],
        "status": artifact["identity_status"],
        "arm": arm,
        "outer_fold": fold,
        "protocol_sha256": protocol_sha,
        "authority_sha256": authority_sha,
    }
    require(value == expected, "run identity drift")
    return value


def validate_preflight(
    value: Mapping[str, Any],
    *,
    artifact: Mapping[str, str],
    job_id: str,
    fold: int,
    protocol_sha: str,
    authority_sha: str,
    manifest: Mapping[str, Any],
    manifest_file_sha: str,
    population: Mapping[str, Any],
) -> None:
    fields = {
        "schema_version",
        "status",
        "outer_fold",
        "all_episode_mapping_count",
        "eligible_query_payloads_independently_validated",
        "allowed_reference_payloads_independently_validated",
        "cache_query_key_count",
        "cache_reference_key_count",
        "cache_execution_ordinal_range",
        "historical_query_ordinal_range",
        "known_regression_mapping",
        "access_manifest_logical_sha256",
        "protected_access_counts",
        "automatic_stage_advance",
        "scientific_GO_or_NO_GO",
        "job_id",
        "protocol_sha256",
        "authority_sha256",
        "signed_reference_access_manifest_sha256",
    }
    exact_fields(value, fields, f"preflight job {job_id}")
    historical = [int(row["query_ordinal"]) for row in population["query_by_id"].values()]
    regression = population["query_by_id"].get("OUTCOME-0504")
    require(regression is not None, "known regression query absent")
    expected = {
        "schema_version": artifact["preflight_schema"],
        "status": artifact["preflight_status"],
        "outer_fold": fold,
        "all_episode_mapping_count": 1_800,
        "eligible_query_payloads_independently_validated": len(population["eligible_queries"]),
        "allowed_reference_payloads_independently_validated": len(population["allowed_references"]),
        "cache_query_key_count": 600,
        "cache_reference_key_count": 4_748,
        "cache_execution_ordinal_range": [0, 599],
        "historical_query_ordinal_range": [min(historical), max(historical)],
        "known_regression_mapping": {
            "query_id": "OUTCOME-0504",
            "query_ordinal": int(regression["query_ordinal"]),
            "execution_ordinal": int(regression["execution_ordinal"]),
        },
        "access_manifest_logical_sha256": manifest["logical_sha256"],
        "protected_access_counts": {key: 0 for key in sorted(PROTECTED_ZERO_KEYS)},
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "job_id": job_id,
        "protocol_sha256": protocol_sha,
        "authority_sha256": authority_sha,
        "signed_reference_access_manifest_sha256": manifest_file_sha,
    }
    require(dict(value) == expected, f"preflight content drift for job {job_id}")


def expected_read_rows(
    schedule: Mapping[str, Any], start: int, completed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_counts: Counter[int] = Counter()
    reference_counts: Counter[int] = Counter()
    for update in schedule["updates"][start:completed]:
        for episode in update["episodes"]:
            query_counts[int(episode["execution_ordinal"])] += 1
            for row in common.candidate_rows(episode):
                reference_counts[row] += 1
    query_rows = [
        {
            "ordinal": ordinal,
            "semantic_read_count": count,
            "file_open_count": 1,
            "cache_file": f"query_{ordinal:04d}.pt",
        }
        for ordinal, count in sorted(query_counts.items())
    ]
    reference_rows = [
        {
            "ordinal": ordinal,
            "semantic_read_count": count,
            "file_open_count": 1,
            "cache_file": f"reference_{ordinal:05d}.pt",
        }
        for ordinal, count in sorted(reference_counts.items())
    ]
    return query_rows, reference_rows


def validate_read_ledger(
    value: Mapping[str, Any],
    *,
    artifact: Mapping[str, str],
    job_id: str,
    start: int,
    completed: int,
    schedule: Mapping[str, Any],
    allowed_queries: set[int],
    allowed_references: set[int],
) -> None:
    fields = {
        "schema_version",
        "status",
        "job_id",
        "start_update",
        "completed_updates",
        "query_rows",
        "reference_rows",
        "query_read_set_sha256",
        "reference_read_set_sha256",
        "protected_access_counts",
        "logical_sha256",
    }
    exact_fields(value, fields, f"runtime read ledger job {job_id}")
    protected_zero_exact(value["protected_access_counts"], f"runtime ledger job {job_id}")
    require(value["schema_version"] == artifact["read_ledger_schema"], "read ledger schema drift")
    require(value["status"] == artifact["read_ledger_status"], "read ledger status drift")
    require(
        str(value["job_id"]) == job_id
        and value["start_update"] == start
        and value["completed_updates"] == completed,
        "read ledger range/job drift",
    )
    query_rows, reference_rows = expected_read_rows(schedule, start, completed)
    require(value["query_rows"] == query_rows, f"query read counts drift for job {job_id}")
    require(value["reference_rows"] == reference_rows, f"reference read counts drift for job {job_id}")
    query_set = [row["ordinal"] for row in query_rows]
    reference_set = [row["ordinal"] for row in reference_rows]
    require(set(query_set).issubset(allowed_queries), "query read outside eligible set")
    require(set(reference_set).issubset(allowed_references), "reference read outside allowlist")
    require(value["query_read_set_sha256"] == common.canonical_sha256(query_set), "query read-set hash drift")
    require(value["reference_read_set_sha256"] == common.canonical_sha256(reference_set), "reference read-set hash drift")
    body = dict(value)
    logical = body.pop("logical_sha256")
    require(logical == common.canonical_sha256(body), "read ledger logical hash drift")


def validate_resume_metadata(
    state: Mapping[str, Any],
    *,
    artifact: Mapping[str, str],
    arm: str,
    fold: int,
    completed: int,
    protocol_sha: str,
    authority_sha: str,
) -> None:
    fields = {
        "schema_version",
        "status",
        "arm",
        "outer_fold",
        "completed_updates",
        "protocol_sha256",
        "authority_sha256",
        "initial_state_sha256",
        "model_state_dict",
        "optimizer_state_dict",
        "loss_trace",
        "pair_count",
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
    }
    exact_fields(state, fields, "resume state")
    require(state["schema_version"] == artifact["resume_schema"], "resume schema drift")
    require(state["status"] == artifact["resume_status"], "resume status drift")
    require(state["arm"] == arm and state["outer_fold"] == fold, "resume scope drift")
    require(state["completed_updates"] == completed, "resume progress drift")
    require(state["protocol_sha256"] == protocol_sha, "resume protocol binding drift")
    require(state["authority_sha256"] == authority_sha, "resume authority binding drift")


def validate_chunks(
    output_dir: Path,
    *,
    protocol: Mapping[str, Any],
    artifact: Mapping[str, str],
    arm: str,
    fold: int,
    protocol_sha: str,
    authority_sha: str,
    manifest: Mapping[str, Any],
    manifest_file_sha: str,
    population: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates = sorted(output_dir.glob("chunk_job*"))
    require(candidates, "no committed chunk directories found")
    require(
        len(candidates) <= protocol["execution"]["maximum_jobs_submitted_now"],
        "chunk count exceeds authority",
    )
    directories: dict[str, Path] = {}
    chunks: list[dict[str, Any]] = []
    for directory in candidates:
        match = CHUNK_DIRECTORY_NAME.fullmatch(directory.name)
        require(
            match is not None and directory.is_dir() and not directory.is_symlink(),
            f"invalid committed chunk artifact: {directory.name}",
        )
        job_id = match.group(1)
        require(job_id and job_id not in directories, "duplicate/empty chunk job ID")
        entries = list(directory.iterdir())
        require(
            {path.name for path in entries} == COMMITTED_CHUNK_FILES
            and all(path.is_file() and not path.is_symlink() for path in entries),
            f"chunk directory is not exact four-file transaction: {directory.name}",
        )
        directories[job_id] = directory
        chunks.append(common.read_json(directory / "chunk.json"))

    chunks.sort(key=lambda row: (int(row.get("start_update", -1)), str(row.get("job_id", ""))))
    cursor = 0
    previous_resume: Mapping[str, Any] | None = None
    seen: set[str] = set()
    for ordinal, chunk in enumerate(chunks):
        chunk_fields = {
            "schema_version",
            "status",
            "arm",
            "outer_fold",
            "job_id",
            "start_update",
            "completed_updates",
            "training_complete",
            "restored_from_resume",
            "restored_resume_state",
            "restored_resume_state_sha256",
            "input_preflight",
            "input_preflight_sha256",
            "runtime_read_ledger",
            "runtime_read_ledger_sha256",
            "resume_state",
            "resume_state_sha256",
            "elapsed_seconds",
            "training_seconds",
            "completed_at_utc",
            "protected_access_counts",
            "automatic_stage_advance",
            "scientific_GO_or_NO_GO",
        }
        job_id = str(chunk.get("job_id"))
        exact_fields(chunk, chunk_fields, f"chunk job {job_id}")
        require(job_id in directories and job_id not in seen, "chunk job ID drift/duplicate")
        seen.add(job_id)
        directory = directories[job_id]
        relative = directory.relative_to(output_dir).as_posix()
        require(chunk["schema_version"] == artifact["chunk_schema"], "chunk schema drift")
        require(chunk["status"] == artifact["chunk_status"], "chunk status drift")
        require(chunk["arm"] == arm and chunk["outer_fold"] == fold, "chunk scope drift")
        start = int(chunk["start_update"])
        completed = int(chunk["completed_updates"])
        require(start == cursor and start < completed <= UPDATES, "chunk coverage non-contiguous")
        require(chunk["restored_from_resume"] is (ordinal > 0), "chunk resume flag drift")
        if ordinal == 0:
            require(
                chunk["restored_resume_state"] is None
                and chunk["restored_resume_state_sha256"] is None,
                "first chunk unexpectedly restored",
            )
        else:
            previous = chunks[ordinal - 1]
            require(
                chunk["restored_resume_state"] == previous["resume_state"]
                and chunk["restored_resume_state_sha256"] == previous["resume_state_sha256"],
                "chunk resume handoff drift",
            )
        require(chunk["training_complete"] is (completed == UPDATES), "chunk completion flag drift")
        require(
            type(chunk["elapsed_seconds"]) in (int, float)
            and math.isfinite(float(chunk["elapsed_seconds"]))
            and float(chunk["elapsed_seconds"]) > 0,
            "chunk elapsed time invalid",
        )
        require(
            type(chunk["training_seconds"]) in (int, float)
            and math.isfinite(float(chunk["training_seconds"]))
            and 0 < float(chunk["training_seconds"]) <= float(chunk["elapsed_seconds"]),
            "chunk training time invalid",
        )
        require(isinstance(chunk["completed_at_utc"], str) and chunk["completed_at_utc"], "chunk completion time absent")
        expected_paths = {
            "input_preflight": f"{relative}/input_preflight.json",
            "runtime_read_ledger": f"{relative}/runtime_read_ledger.json",
            "resume_state": f"{relative}/resume_state.pt",
        }
        for field, expected in expected_paths.items():
            require(chunk[field] == expected, f"chunk path drift: {field}")
        preflight_path = directory / "input_preflight.json"
        read_path = directory / "runtime_read_ledger.json"
        resume_path = directory / "resume_state.pt"
        require(common.file_sha256(preflight_path) == chunk["input_preflight_sha256"], "preflight hash drift")
        require(common.file_sha256(read_path) == chunk["runtime_read_ledger_sha256"], "read ledger hash drift")
        require(common.file_sha256(resume_path) == chunk["resume_state_sha256"], "resume hash drift")
        protected_zero_exact(chunk["protected_access_counts"], f"chunk job {job_id}")
        require(chunk["automatic_stage_advance"] is False, "chunk auto-advance enabled")
        require(chunk["scientific_GO_or_NO_GO"] is None, "chunk scientific decision present")
        validate_preflight(
            common.read_json(preflight_path),
            artifact=artifact,
            job_id=job_id,
            fold=fold,
            protocol_sha=protocol_sha,
            authority_sha=authority_sha,
            manifest=manifest,
            manifest_file_sha=manifest_file_sha,
            population=population,
        )
        validate_read_ledger(
            common.read_json(read_path),
            artifact=artifact,
            job_id=job_id,
            start=start,
            completed=completed,
            schedule=schedule,
            allowed_queries=set(population["eligible_queries"]),
            allowed_references=set(population["allowed_references"]),
        )
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_metadata(
            state,
            artifact=artifact,
            arm=arm,
            fold=fold,
            completed=completed,
            protocol_sha=protocol_sha,
            authority_sha=authority_sha,
        )
        require(isinstance(state["loss_trace"], list) and len(state["loss_trace"]) == completed, "resume loss-trace cardinality drift")
        if previous_resume is not None:
            prior_completed = int(previous_resume["completed_updates"])
            require(state["loss_trace"][:prior_completed] == previous_resume["loss_trace"], "resume loss prefix rewritten")
            require(int(state["pair_count"]) >= int(previous_resume["pair_count"]), "resume pair count regressed")
        expected_pairs = sum(int(update["pair_count"]) for update in schedule["updates"][:completed])
        require(state["pair_count"] == expected_pairs, "resume pair count drift")
        previous_resume = state
        cursor = completed
    require(cursor == UPDATES and chunks[-1]["training_complete"] is True, "chunk chain incomplete")
    require(all(not row["training_complete"] for row in chunks[:-1]), "nonfinal chunk claims completion")
    return chunks


def validate_loss_trace(value: Any, schedule: Mapping[str, Any]) -> None:
    require(isinstance(value, list) and len(value) == UPDATES, "loss trace cardinality drift")
    fields = {
        "update",
        "learning_rate",
        "mean_pair_loss",
        "gradient_norm_before_clip",
        "pair_count",
    }
    for index, row in enumerate(value):
        exact_fields(row, fields, f"loss trace update {index + 1}")
        require(row["update"] == index + 1, "loss trace update index drift")
        require(
            math.isclose(
                float(row["learning_rate"]),
                common.learning_rate(index + 1),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "learning-rate trace drift",
        )
        require(
            type(row["mean_pair_loss"]) in (int, float)
            and math.isfinite(float(row["mean_pair_loss"]))
            and float(row["mean_pair_loss"]) >= 0,
            "loss nonfinite/negative",
        )
        require(
            type(row["gradient_norm_before_clip"]) in (int, float)
            and math.isfinite(float(row["gradient_norm_before_clip"]))
            and float(row["gradient_norm_before_clip"]) >= 0,
            "gradient norm nonfinite/negative",
        )
        require(row["pair_count"] == schedule["updates"][index]["pair_count"], "per-update pair count drift")


def validate_checkpoint_and_resume(
    output_dir: Path,
    *,
    result: Mapping[str, Any],
    final_chunk: Mapping[str, Any],
    artifact: Mapping[str, str],
    arm: str,
    fold: int,
    protocol_sha: str,
    authority_sha: str,
    loss_trace: Sequence[Mapping[str, Any]],
    pair_count: int,
) -> tuple[Path, Path]:
    checkpoint_path = output_dir / str(result["checkpoint"])
    resume_path = output_dir / str(result["completed_resume_state"])
    require(
        checkpoint_path.resolve().parent == output_dir.resolve()
        and checkpoint_path.name == "checkpoint_update2048.pt"
        and checkpoint_path.is_file()
        and not checkpoint_path.is_symlink(),
        "checkpoint path drift",
    )
    require(
        result["completed_resume_state"] == final_chunk["resume_state"]
        and resume_path.is_file()
        and not resume_path.is_symlink()
        and output_dir.resolve() in resume_path.resolve().parents,
        "completed resume path drift",
    )
    require(not (output_dir / "resume_state.pt").exists(), "legacy mutable resume remains")
    require(common.file_sha256(checkpoint_path) == result["checkpoint_sha256"], "checkpoint hash drift")
    require(common.file_sha256(resume_path) == result["completed_resume_state_sha256"], "completed resume hash drift")
    require(final_chunk["resume_state_sha256"] == result["completed_resume_state_sha256"], "chunk/result resume hash drift")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_fields = {
        "schema_version",
        "status",
        "arm",
        "outer_fold",
        "update",
        "seed",
        "protocol_sha256",
        "authority_sha256",
        "initial_state_sha256",
        "final_state_sha256",
        "model_state_dict",
    }
    exact_fields(checkpoint, checkpoint_fields, "checkpoint")
    require(checkpoint["schema_version"] == artifact["checkpoint_schema"], "checkpoint schema drift")
    require(checkpoint["status"] == artifact["checkpoint_status"], "checkpoint status drift")
    require(checkpoint["arm"] == arm and checkpoint["outer_fold"] == fold, "checkpoint scope drift")
    require(checkpoint["update"] == UPDATES and checkpoint["seed"] == SEED, "checkpoint update/seed drift")
    require(checkpoint["protocol_sha256"] == protocol_sha, "checkpoint protocol binding drift")
    require(checkpoint["authority_sha256"] == authority_sha, "checkpoint authority binding drift")

    # Replay the runner's seed contract independently of incidental validator RNG.
    torch.manual_seed(SEED)
    fresh = DINO_RCDE_V1_2()
    fresh_state = fresh.state_dict()
    initial_sha = common.state_dict_sha256(fresh_state)
    final_state = checkpoint["model_state_dict"]
    require(isinstance(final_state, dict), "checkpoint model state absent")
    require(set(final_state) == set(fresh_state), "checkpoint model key drift")
    require(sum(int(value.numel()) for value in final_state.values()) == EXPECTED_PARAMETERS, "checkpoint parameter count drift")
    final_sha = common.state_dict_sha256(final_state)
    require(checkpoint["initial_state_sha256"] == result["initial_state_sha256"] == initial_sha, "initial model hash drift")
    require(checkpoint["final_state_sha256"] == result["final_state_sha256"] == final_sha, "final model hash drift")
    require(final_sha != initial_sha, "training did not change model state")

    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    validate_resume_metadata(
        resume,
        artifact=artifact,
        arm=arm,
        fold=fold,
        completed=UPDATES,
        protocol_sha=protocol_sha,
        authority_sha=authority_sha,
    )
    require(resume["initial_state_sha256"] == initial_sha, "resume initial hash drift")
    require(common.equal_state(resume["model_state_dict"], final_state), "resume/checkpoint model mismatch")
    require(resume["loss_trace"] == list(loss_trace), "resume/result loss trace drift")
    require(resume["pair_count"] == pair_count, "resume/result pair count drift")
    require(isinstance(resume["torch_cpu_rng_state"], torch.Tensor), "CPU RNG state absent")
    require(isinstance(resume["torch_cuda_rng_state_all"], list), "CUDA RNG list absent")
    require(isinstance(resume["numpy_random_state"], tuple), "NumPy RNG state absent")

    optimizer = resume["optimizer_state_dict"]
    require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"}, "optimizer fields drift")
    require(isinstance(optimizer["param_groups"], list) and len(optimizer["param_groups"]) == 1, "optimizer group drift")
    group = optimizer["param_groups"][0]
    require(math.isclose(float(group["lr"]), 3.0e-5, rel_tol=0.0, abs_tol=1e-15), "final optimizer LR drift")
    require(float(group["weight_decay"]) == 1.0e-4, "optimizer weight decay drift")
    parameter_shapes = [tuple(parameter.shape) for parameter in fresh.parameters()]
    parameter_ids = list(group["params"])
    require(len(parameter_ids) == len(parameter_shapes) and len(set(parameter_ids)) == len(parameter_ids), "optimizer parameter IDs drift")
    require(set(optimizer["state"]) == set(parameter_ids), "optimizer state coverage drift")
    for parameter_id, shape in zip(parameter_ids, parameter_shapes, strict=True):
        state = optimizer["state"][parameter_id]
        require(set(state) >= {"step", "exp_avg", "exp_avg_sq"}, "AdamW moment fields absent")
        step = state["step"]
        step_value = float(step.item()) if isinstance(step, torch.Tensor) else float(step)
        require(step_value == UPDATES, "AdamW step count drift")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state[key]
            require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape, f"AdamW {key} shape drift")
            require(bool(torch.isfinite(tensor).all()), f"AdamW {key} nonfinite")
    return checkpoint_path, resume_path


def validate_result(
    output_dir: Path,
    *,
    protocol: Mapping[str, Any],
    artifact: Mapping[str, str],
    arm: str,
    fold: int,
    protocol_sha: str,
    authority_sha: str,
    identity_path: Path,
    manifest_path: Path,
    population: Mapping[str, Any],
    schedule: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Path, Path]:
    result_path = output_dir / "train_result.json"
    result = common.read_json(result_path)
    fields = {
        "schema_version",
        "status",
        "claim_level",
        "arm",
        "outer_fold",
        "authority_sha256",
        "protocol_sha256",
        "run_identity_sha256",
        "signed_reference_access_manifest_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "completed_resume_state",
        "completed_resume_state_sha256",
        "initial_state_sha256",
        "final_state_sha256",
        "parameter_count",
        "training",
        "bag_permutation",
        "execution_chain",
        "runtime_access",
        "protected_access_counts",
        "automatic_stage_advance",
        "scientific_GO_or_NO_GO",
        "next_required_stage",
    }
    exact_fields(result, fields, "train result")
    require(result["schema_version"] == artifact["result_schema"], "result schema drift")
    require(result["status"] == artifact["result_status"], "result status drift")
    require(
        result["claim_level"]
        == "engineering_training_complete_not_scientific_GO_or_NO_GO",
        "result claim drift",
    )
    require(result["arm"] == arm and result["outer_fold"] == fold, "result scope drift")
    require(result["protocol_sha256"] == protocol_sha, "result protocol binding drift")
    require(result["authority_sha256"] == authority_sha, "result authority binding drift")
    require(result["run_identity_sha256"] == common.file_sha256(identity_path), "result identity hash drift")
    require(result["signed_reference_access_manifest_sha256"] == common.file_sha256(manifest_path), "result manifest hash drift")
    require(result["parameter_count"] == EXPECTED_PARAMETERS, "result parameter count drift")
    protected_zero_exact(result["protected_access_counts"], "train result")
    require(result["automatic_stage_advance"] is False, "result auto-advance enabled")
    require(result["scientific_GO_or_NO_GO"] is None, "result scientific decision present")
    require(result["next_required_stage"] == "R1_MAIN_TRAINING_VALIDATION", "result next-stage drift")

    training = result["training"]
    training_fields = {
        "seed",
        "update_count",
        "episodes_per_update",
        "raw_correct_per_update",
        "raw_wrong_per_update",
        "pair_count",
        "correct_pool_count",
        "wrong_pool_count",
        "query_order_namespace",
        "query_order_key",
        "query_order_tie_break",
        "candidate_direction_source",
        "cache_query_key",
        "schedule_sha256",
        "query_tile_rows",
        "reference_tile_rows",
        "loss_trace",
    }
    exact_fields(training, training_fields, "result training")
    expected_scalars = {
        "seed": SEED,
        "update_count": UPDATES,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "raw_correct_per_update": 2,
        "raw_wrong_per_update": 2,
        "pair_count": schedule["pair_count"],
        "correct_pool_count": len(population["correct"]),
        "wrong_pool_count": len(population["wrong"]),
        "query_order_namespace": QUERY_ORDER_NAMESPACE,
        "query_order_key": "source_image_sha256",
        "query_order_tie_break": "stable_episode_ledger_order",
        "candidate_direction_source": "P0_frozen_model_candidate_order",
        "cache_query_key": "execution_ordinal",
        "schedule_sha256": schedule["sha256"],
        "query_tile_rows": QUERY_TILE_ROWS,
        "reference_tile_rows": REFERENCE_TILE_ROWS,
    }
    for key, expected in expected_scalars.items():
        require(training[key] == expected, f"result training drift: {key}")
    validate_loss_trace(training["loss_trace"], schedule)

    expected_bag = {
        "enabled": arm == "RCDE_BAG",
        "valid_tokens_only": True,
        "fixed_point_free_cyclic_shift": True,
        "query_binding_namespace": "DINO_RCDE_R1_BAG_QUERY_PERMUTATION_V1_5",
        "reference_binding_namespace": "DINO_RCDE_R1_BAG_REFERENCE_PERMUTATION_V1_5",
    }
    require(result["bag_permutation"] == expected_bag, "BAG permutation receipt drift")
    chain = result["execution_chain"]
    exact_fields(chain, {"chunk_count", "resume_count", "chunks", "maximum_jobs_respected", "chunks_contiguous"}, "execution chain")
    require(chain["chunk_count"] == len(chunks), "result chunk count drift")
    require(chain["resume_count"] == len(chunks) - 1, "result resume count drift")
    require(chain["maximum_jobs_respected"] is True and chain["chunks_contiguous"] is True, "result chain closure false")
    expected_chunk_bindings = []
    for chunk in chunks:
        chunk_path = output_dir / f"chunk_job{chunk['job_id']}" / "chunk.json"
        expected_chunk_bindings.append(
            {
                "path": chunk_path.relative_to(output_dir).as_posix(),
                "sha256": common.file_sha256(chunk_path),
                "start_update": int(chunk["start_update"]),
                "completed_updates": int(chunk["completed_updates"]),
                "restored_from_resume": bool(chunk["restored_from_resume"]),
                "restored_resume_state": chunk["restored_resume_state"],
                "restored_resume_state_sha256": chunk["restored_resume_state_sha256"],
                "resume_state": chunk["resume_state"],
                "resume_state_sha256": chunk["resume_state_sha256"],
                "runtime_read_ledger": chunk["runtime_read_ledger"],
                "runtime_read_ledger_sha256": chunk["runtime_read_ledger_sha256"],
            }
        )
    require(chain["chunks"] == expected_chunk_bindings, "result chunk bindings drift")

    query_union = sorted(
        {
            int(row["ordinal"])
            for chunk in chunks
            for row in common.read_json(output_dir / chunk["runtime_read_ledger"])["query_rows"]
        }
    )
    reference_union = sorted(
        {
            int(row["ordinal"])
            for chunk in chunks
            for row in common.read_json(output_dir / chunk["runtime_read_ledger"])["reference_rows"]
        }
    )
    expected_runtime = {
        "query_read_union": query_union,
        "query_read_union_sha256": common.canonical_sha256(query_union),
        "reference_read_union": reference_union,
        "reference_read_union_sha256": common.canonical_sha256(reference_union),
        "reference_read_union_equals_signed_allowlist": reference_union == population["allowed_references"],
        "heldout_reference_intersection_count": 0,
    }
    require(result["runtime_access"] == expected_runtime, "result runtime-access drift")
    require(query_union == population["eligible_queries"], "runtime query union incomplete")
    require(reference_union == population["allowed_references"], "runtime reference union incomplete")
    require(not set(reference_union).intersection(population["heldout_references"]), "runtime read heldout reference")
    checkpoint_path, resume_path = validate_checkpoint_and_resume(
        output_dir,
        result=result,
        final_chunk=chunks[-1],
        artifact=artifact,
        arm=arm,
        fold=fold,
        protocol_sha=protocol_sha,
        authority_sha=authority_sha,
        loss_trace=training["loss_trace"],
        pair_count=training["pair_count"],
    )
    return result, checkpoint_path, resume_path


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable validation output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--arm", choices=sorted(ALLOWED_ARMS), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    authority_path = args.authority.resolve()
    output_dir = args.output_dir.resolve()
    validation_output = (
        args.validation_output.resolve()
        if args.validation_output is not None
        else output_dir / "training_validation.json"
    )
    require(ROOT in output_dir.parents, "output directory escapes RC root")
    require(ROOT in validation_output.parents, "validation path escapes RC root")
    require(output_dir.is_dir(), "completed output directory absent")
    require(not validation_output.exists(), "immutable validation output already exists")

    protocol, authority, binding_hashes = validate_protocol(
        protocol_path, authority_path, args.arm, args.fold
    )
    artifact = artifact_contract(protocol)
    require(
        resolve_bound(str(protocol["resume"]["persistent_output"])) == output_dir,
        "persistent output binding drift",
    )
    cache_root = (
        args.cache_root.resolve()
        if args.cache_root is not None
        else resolve_bound(str(protocol["cache"]["root"]))
    )
    require(ROOT in cache_root.parents, "cache root escapes RC root")
    require(cache_root == resolve_bound(str(protocol["cache"]["root"])), "cache root argument drift")
    protocol_sha = common.file_sha256(protocol_path)
    authority_sha = common.file_sha256(authority_path)

    population = common.build_population(protocol, args.fold)
    index = common.cache_index(cache_root, set(population["reference_by_row"]))
    common.validate_cache_payloads(index, population)
    manifest_path = output_dir / "signed_reference_access_manifest.json"
    manifest = common.read_json(manifest_path)
    expected = expected_manifest(population, args.fold, artifact)
    require(manifest == expected, "signed access manifest drift")
    protected_zero_exact(manifest["protected_access_counts"], "signed access manifest")
    identity_path = output_dir / "run_identity.json"
    validate_identity(
        identity_path,
        artifact=artifact,
        arm=args.arm,
        fold=args.fold,
        protocol_sha=protocol_sha,
        authority_sha=authority_sha,
    )
    schedule = common.expected_schedule(population, args.fold)
    chunks = validate_chunks(
        output_dir,
        protocol=protocol,
        artifact=artifact,
        arm=args.arm,
        fold=args.fold,
        protocol_sha=protocol_sha,
        authority_sha=authority_sha,
        manifest=manifest,
        manifest_file_sha=common.file_sha256(manifest_path),
        population=population,
        schedule=schedule,
    )
    result, checkpoint_path, resume_path = validate_result(
        output_dir,
        protocol=protocol,
        artifact=artifact,
        arm=args.arm,
        fold=args.fold,
        protocol_sha=protocol_sha,
        authority_sha=authority_sha,
        identity_path=identity_path,
        manifest_path=manifest_path,
        population=population,
        schedule=schedule,
        chunks=chunks,
    )

    require(not list(output_dir.glob("*.tmp*")), "temporary artifacts remain")
    expected_top_level = {
        "run_identity.json",
        "signed_reference_access_manifest.json",
        "checkpoint_update2048.pt",
        "train_result.json",
        *(f"chunk_job{chunk['job_id']}" for chunk in chunks),
    }
    require({path.name for path in output_dir.iterdir()} == expected_top_level, "formal result namespace drift")

    checks = {
        "single_arm_single_fold_protocol_scope_exact": True,
        "protocol_authority_and_all_bindings": True,
        "all_artifact_schemas_and_statuses_protocol_declared": True,
        "canonical_cache_validation_pass": True,
        "query_id_historical_execution_mapping_1800_of_1800": True,
        "cache_execution_ordinals_exact_0_599": True,
        "eligible_cache_payload_sources_models_and_logical_hashes": True,
        "query_order_recomputed_from_source_sha256": True,
        "candidate_directions_recomputed_from_P0_frozen_order": True,
        "schedule_2048_updates_recomputed": True,
        "learning_rates_and_pair_counts_recomputed": True,
        "run_identity_exact": True,
        "signed_reference_access_manifest_exact": True,
        "identity_supergroup_and_reference_heldout_exclusion": True,
        "atomic_chunk_directory_transactions_exact": True,
        "all_chunk_ranges_contiguous_and_bounded": True,
        "all_preflight_receipts_exact": True,
        "all_runtime_read_ledgers_match_schedule": True,
        "runtime_reference_union_equals_allowlist": True,
        "finalization_bound_to_last_committed_completed_state": True,
        "checkpoint_and_completed_resume_bound": True,
        "optimizer_completed_2048_updates": True,
        "exact_protected_zero_key_sets": True,
        "no_scientific_decision_or_automatic_advance": True,
    }
    validation = {
        "schema_version": artifact["validation_schema"],
        "status": artifact["validation_status"],
        "claim_level": "engineering_training_validation_only_not_scientific_GO_or_NO_GO",
        "arm": args.arm,
        "outer_fold": args.fold,
        "checks": checks,
        "protocol_sha256": protocol_sha,
        "authority_sha256": authority_sha,
        "binding_hashes": binding_hashes,
        "run_identity_sha256": common.file_sha256(identity_path),
        "signed_reference_access_manifest_sha256": common.file_sha256(manifest_path),
        "train_result_sha256": common.file_sha256(output_dir / "train_result.json"),
        "checkpoint_sha256": common.file_sha256(checkpoint_path),
        "completed_resume_state_sha256": common.file_sha256(resume_path),
        "chunk_count": len(chunks),
        "update_count": UPDATES,
        "pair_count": schedule["pair_count"],
        "schedule_sha256": schedule["sha256"],
        "eligible_query_count": len(population["eligible_queries"]),
        "allowed_reference_count": len(population["allowed_references"]),
        "heldout_reference_intersection_count": 0,
        "protected_access_counts": {key: 0 for key in sorted(PROTECTED_ZERO_KEYS)},
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_required_stage": protocol["decision"]["next_on_pass"],
    }
    atomic_json_no_clobber(validation_output, validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
