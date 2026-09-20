#!/usr/bin/env python3
"""Freeze matched RCDE_BAG/RCDE_CONTEXT folds as engineering provenance only.

The program reads already completed outer-train artifacts.  It never copies a
checkpoint, runs held-out examples, joins labels, computes scientific metrics,
or advances a scientific stage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "rc_dino_rcde_r1_matched_foldset_protocol_v1_20260814"
PROTOCOL_STAGE = "R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_CLOSURE"
MANIFEST_SCHEMA = "rc_dino_rcde_r1_matched_foldset_manifest_v1_0"
MANIFEST_STATUS = "DINO_RCDE_R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_FROZEN"
CLAIM_LEVEL = "engineering_matched_foldset_closure_only_not_scientific_GO_or_NO_GO"
MANIFEST_NEXT = "R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_VALIDATION"

FOLDS = (1, 2, 3, 4)
SEED = 17
PARAMETERS = 65_125
UPDATES = 2_048
PAIRS = 65_536
INITIAL_STATE_SHA256 = "4a51b45b1a29aa80be6acb18278d783d5485fbff01665f7054b518e2d870099b"
MODEL_STATE_SCHEMA_SHA256 = "2a1110ccebf0706986fe2d5377c4abadd0fc0687560ae2738bd850ea4f6af2e7"
SOURCE_CONTEXT_AUTHORITY_PATH = "registry/current_authority_v16_20260814.json"
SOURCE_CONTEXT_AUTHORITY_SHA256 = "60196af6981fd99a72d9cb8e21685b78793bda4bce7f2773772db599e1490de8"
SOURCE_CONTEXT_AUTHORITY_STATUS = "DINO_RCDE_R1_CONTEXT_FOLDS1_4_V18_ACCELERATED_TRAINING_AUTHORIZED"
AMENDMENT_PATH = "registry/runtime_scheduler_amendment_job5072108_throttle4_20260814.json"
AMENDMENT_SHA256 = "891a81b2ee0c5f286d9a53d0ae3a73b8273a0a9e1942a41003cd0483a28dae63"
AMENDMENT_STATUS = "RUNTIME_SCHEDULER_THROTTLE4_AUTHORIZED"
RECEIPT_PATH = "registry/runtime_scheduler_receipt_job5072108_throttle4_20260814.json"
RECEIPT_SHA256 = "fbaf69f52c9aa6781f26b31c3a9daa22f1c8ce9bf030f2986e51da72c0c3eba4"
RECEIPT_STATUS = "RUNTIME_SCHEDULER_THROTTLE4_APPLIED_AND_FOLD4_RUNNING"
BAG_MANIFEST_PATH = "results/dino_rcde_r1_main_bag_foldset_v1_0/foldset_manifest.json"
BAG_MANIFEST_SHA256 = "7c52eab2df096abbc61d79cbb2ea30dfd9dd851656db25b3a7cf2c5f0d6112be"
BAG_MANIFEST_STATUS = "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_FROZEN"
BAG_VALIDATION_PATH = "results/dino_rcde_r1_main_bag_foldset_v1_0/foldset_validation.json"
BAG_VALIDATION_SHA256 = "d7b868920eab4c77c50b048baa3a01cf2eb5762f618132f4a9c34b47b28f9e46"
BAG_VALIDATION_STATUS = "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_VALIDATION_PASS"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHUNK_JOB = re.compile(r"^chunk_job([0-9]+)/chunk\.json$")

COMMON_BINDINGS = (
    "scientific_contract", "cache_protocol", "cache_result", "cache_validation",
    "training_roles", "episode_ledger", "resource_shape_ledger",
    "model_visible_c128", "rcde_core",
)
CONTEXT_SOURCES = (
    "protocol", "run_identity", "access_manifest", "train_result",
    "training_validation", "checkpoint",
)
CONTEXT_INPUT_KEYS = {"outer_fold", *CONTEXT_SOURCES}
PROTECTED_ZERO = {
    "C8_runtime_read_count": 0,
    "S8_runtime_read_count": 0,
    "home_files_modified": 0,
    "opened_runtime_read_count": 0,
    "sealed_runtime_read_count": 0,
    "unauthorized_natural_result_read_count": 0,
}
FORBIDDEN = {
    "checkpoint_copy": False,
    "outer_heldout_forward": False,
    "heldout_label_join": False,
    "scientific_reduction": False,
    "scientific_decision": False,
    "protected_paths": ["C8", "S8", "opened", "sealed"],
    "home_write": False,
}
SCHEDULER_COMPLETION = {
    "source": "live_slurm_accounting_read_only_observation_20260814",
    "array_job_id": 5_072_108,
    "tasks": [
        {"array_task_id": 1, "job_id_raw": 5_072_109, "state": "COMPLETED", "exit_code": "0:0"},
        {"array_task_id": 2, "job_id_raw": 5_072_110, "state": "COMPLETED", "exit_code": "0:0"},
        {"array_task_id": 3, "job_id_raw": 5_072_111, "state": "COMPLETED", "exit_code": "0:0"},
        {"array_task_id": 4, "job_id_raw": 5_072_108, "state": "COMPLETED", "exit_code": "0:0"},
    ],
    "validator_queries_live_slurm": False,
}
BAG_MODE = {
    "enabled": True,
    "fixed_point_free_cyclic_shift": True,
    "query_binding_namespace": "DINO_RCDE_R1_BAG_QUERY_PERMUTATION_V1_5",
    "reference_binding_namespace": "DINO_RCDE_R1_BAG_REFERENCE_PERMUTATION_V1_5",
    "valid_tokens_only": True,
}
CONTEXT_MODE = {**BAG_MODE, "enabled": False}
EXPECTED_CONTEXT_CHECKS = {
    "all_artifact_schemas_and_statuses_protocol_declared",
    "all_chunk_ranges_contiguous_and_bounded", "all_preflight_receipts_exact",
    "all_runtime_read_ledgers_match_schedule", "atomic_chunk_directory_transactions_exact",
    "cache_execution_ordinals_exact_0_599", "candidate_directions_recomputed_from_P0_frozen_order",
    "canonical_cache_validation_pass", "checkpoint_and_completed_resume_bound",
    "eligible_cache_payload_sources_models_and_logical_hashes", "exact_protected_zero_key_sets",
    "finalization_bound_to_last_committed_completed_state",
    "identity_supergroup_and_reference_heldout_exclusion", "learning_rates_and_pair_counts_recomputed",
    "no_scientific_decision_or_automatic_advance", "optimizer_completed_2048_updates",
    "protocol_authority_and_all_bindings", "query_id_historical_execution_mapping_1800_of_1800",
    "query_order_recomputed_from_source_sha256", "run_identity_exact",
    "runtime_reference_union_equals_allowlist", "schedule_2048_updates_recomputed",
    "signed_reference_access_manifest_exact", "single_arm_single_fold_protocol_scope_exact",
}


class MatchedFoldsetAbort(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedFoldsetAbort(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatchedFoldsetAbort(f"invalid {context}: {path}") from exc
    require(isinstance(value, dict), f"{context} must be an object")
    return value


def resolve_relative(text: str) -> Path:
    raw = Path(text)
    require(text != "" and not raw.is_absolute(), "source path must be RC-relative")
    path = (ROOT / raw).resolve()
    require(path.is_relative_to(ROOT.resolve()), "source path escapes RC root")
    require(path.is_file() and not path.is_symlink(), f"source is not a regular file: {text}")
    return path


def exact_source(value: Any, context: str) -> tuple[dict[str, str], Path]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{context} source schema drift")
    text, digest = value.get("path"), value.get("sha256")
    require(isinstance(text, str), f"{context} path missing")
    require(isinstance(digest, str) and HEX64.fullmatch(digest) is not None, f"{context} sha256 malformed")
    path = resolve_relative(text)
    require(file_sha256(path) == digest, f"{context} physical sha256 drift")
    return {"path": text, "sha256": digest}, path


def required_source(value: Any, context: str) -> tuple[dict[str, str], Path]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256", "required_status"}, f"{context} required source schema drift")
    source, path = exact_source({"path": value["path"], "sha256": value["sha256"]}, context)
    require(isinstance(value["required_status"], str), f"{context} required status missing")
    return {**source, "required_status": value["required_status"]}, path


def no_science(value: Mapping[str, Any], context: str) -> None:
    require(value.get("scientific_GO_or_NO_GO") is None, f"{context} contains scientific decision")
    require(value.get("automatic_stage_advance") is False, f"{context} permits automatic advance")


def logical_access(access: Mapping[str, Any], context: str) -> str:
    detached = dict(access)
    observed = detached.pop("logical_sha256", None)
    require(isinstance(observed, str) and HEX64.fullmatch(observed) is not None, f"{context} logical sha malformed")
    require(canonical_sha256(detached) == observed, f"{context} logical sha drift")
    return observed


def integer_set(value: Any, context: str) -> set[int]:
    require(isinstance(value, list) and all(type(item) is int for item in value), f"{context} malformed")
    require(len(value) == len(set(value)), f"{context} contains duplicates")
    return set(value)


def training_projection(training: Any, arm: str, fold: int, context: str) -> dict[str, Any]:
    require(isinstance(training, dict), f"{context} training missing")
    require(training.get("arms") == [arm] and training.get("outer_folds") == [fold], f"{context} arm/fold drift")
    require(training.get("seed") == SEED and training.get("parameter_count") == PARAMETERS, f"{context} seed/parameter drift")
    require(training.get("updates_total") == UPDATES, f"{context} update contract drift")
    require(training.get("outer_train_only") is True and training.get("heldout_target_join_allowed") is False, f"{context} outer-train boundary drift")
    projection = dict(training)
    projection["arms"] = ["RCDE_MATCHED_ARM"]
    return projection


def source_spec_from_manifest(value: Any, context: str, *, logical: bool = False) -> tuple[dict[str, str], Path]:
    keys = {"path", "sha256", "logical_sha256"} if logical else {"path", "sha256"}
    require(isinstance(value, dict) and set(value) == keys, f"{context} manifest source schema drift")
    source, path = exact_source({"path": value["path"], "sha256": value["sha256"]}, context)
    if logical:
        require(isinstance(value["logical_sha256"], str) and HEX64.fullmatch(value["logical_sha256"]), f"{context} logical digest malformed")
        source["logical_sha256"] = value["logical_sha256"]
    return source, path


def audit_scheduler(protocol: Mapping[str, Any], context_protocols: list[dict[str, str]]) -> dict[str, Any]:
    authority_spec, authority_path = required_source(protocol.get("source_context_authority"), "source_context_authority")
    require(authority_spec["path"] == SOURCE_CONTEXT_AUTHORITY_PATH and authority_spec["sha256"] == SOURCE_CONTEXT_AUTHORITY_SHA256, "v16 source authority binding drift")
    require(authority_spec["required_status"] == SOURCE_CONTEXT_AUTHORITY_STATUS, "v16 required status drift")
    authority = read_json(authority_path, "v16 source authority")
    require(authority.get("status") == SOURCE_CONTEXT_AUTHORITY_STATUS, "v16 source authority status drift")
    require(authority.get("protected_access_counts") == PROTECTED_ZERO, "v16 protected access drift")
    no_science(authority, "v16 source authority")
    scopes = authority.get("authorized_scopes")
    require(isinstance(scopes, list), "v16 authorized scopes missing")
    for fold in FOLDS:
        matches = [row for row in scopes if isinstance(row, dict) and row.get("arm") == "RCDE_CONTEXT" and row.get("outer_fold") == fold]
        require(len(matches) == 1 and matches[0].get("optimizer_updates_total") == UPDATES and matches[0].get("seed") == SEED, f"v16 fold{fold} scope drift")

    amendment_spec, amendment_path = required_source(protocol.get("runtime_scheduler_amendment"), "runtime amendment")
    require(amendment_spec["path"] == AMENDMENT_PATH and amendment_spec["sha256"] == AMENDMENT_SHA256 and amendment_spec["required_status"] == AMENDMENT_STATUS, "runtime amendment binding drift")
    amendment = read_json(amendment_path, "runtime amendment")
    require(amendment.get("status") == AMENDMENT_STATUS and amendment.get("job_id") == 5_072_108, "runtime amendment status/job drift")
    require(amendment.get("authority") == {"path": authority_spec["path"], "sha256": authority_spec["sha256"]}, "runtime amendment authority drift")
    require(amendment.get("protocols") == context_protocols, "runtime amendment protocol list drift")
    require(amendment.get("submitted_slurm", {}).get("submitted_array") == "1-4%3", "submitted array provenance drift")
    require(amendment.get("scheduler_only_override") == {
        "field": "ArrayTaskThrottle", "old_value": 3, "new_value": 4,
        "reason": "all four folds are file_isolated single_scope jobs and the user explicitly requested immediate fold4 launch; the prior value 3 was a conservative engineering carry_over rather than a scientific requirement",
    }, "scheduler-only override drift")
    require(amendment.get("invariants_unchanged") and all(v is True for v in amendment["invariants_unchanged"].values()), "scheduler amendment invariant drift")
    require(amendment.get("protected_access_counts") == PROTECTED_ZERO, "runtime amendment protected drift")
    no_science(amendment, "runtime amendment")

    receipt_spec, receipt_path = required_source(protocol.get("runtime_scheduler_receipt"), "runtime receipt")
    require(receipt_spec["path"] == RECEIPT_PATH and receipt_spec["sha256"] == RECEIPT_SHA256 and receipt_spec["required_status"] == RECEIPT_STATUS, "runtime receipt binding drift")
    receipt = read_json(receipt_path, "runtime receipt")
    require(receipt.get("status") == RECEIPT_STATUS and receipt.get("job_id") == 5_072_108, "runtime receipt status/job drift")
    require(receipt.get("amendment") == {"path": amendment_spec["path"], "sha256": amendment_spec["sha256"]}, "runtime receipt amendment drift")
    tasks = receipt.get("observed_tasks")
    require(isinstance(tasks, list) and [row.get("array_task_id") for row in tasks if isinstance(row, dict)] == list(FOLDS), "runtime receipt task set drift")
    require(all(row.get("state") == "RUNNING" and isinstance(row.get("node"), str) for row in tasks), "runtime receipt observation drift")
    require(receipt.get("model_or_data_files_modified") == 0 and receipt.get("home_files_modified") == 0, "runtime receipt mutation drift")
    require(receipt.get("scientific_GO_or_NO_GO") is None, "runtime receipt scientific decision")
    require(protocol.get("scheduler_completion") == SCHEDULER_COMPLETION, "completed sacct attestation drift")
    return {"source_context_authority": authority_spec, "amendment": amendment_spec, "receipt": receipt_spec, "completion": SCHEDULER_COMPLETION}


def audit_bag_foldset(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    container = protocol.get("bag_foldset")
    require(isinstance(container, dict) and set(container) == {"manifest", "validation"}, "BAG foldset binding schema drift")
    manifest_spec, manifest_path = required_source(container["manifest"], "BAG foldset manifest")
    validation_spec, validation_path = required_source(container["validation"], "BAG foldset validation")
    require((manifest_spec["path"], manifest_spec["sha256"], manifest_spec["required_status"]) == (BAG_MANIFEST_PATH, BAG_MANIFEST_SHA256, BAG_MANIFEST_STATUS), "BAG manifest binding drift")
    require((validation_spec["path"], validation_spec["sha256"], validation_spec["required_status"]) == (BAG_VALIDATION_PATH, BAG_VALIDATION_SHA256, BAG_VALIDATION_STATUS), "BAG validation binding drift")
    manifest = read_json(manifest_path, "BAG foldset manifest")
    validation = read_json(validation_path, "BAG foldset validation")
    require(manifest.get("schema_version") == "rc_dino_rcde_r1_main_bag_foldset_manifest_v1_0" and manifest.get("status") == BAG_MANIFEST_STATUS, "BAG manifest status/schema drift")
    require(validation.get("schema_version") == "rc_dino_rcde_r1_main_bag_foldset_validation_v1_0" and validation.get("status") == BAG_VALIDATION_STATUS, "BAG validation status/schema drift")
    require(validation.get("foldset_manifest") == {"path": manifest_spec["path"], "sha256": manifest_spec["sha256"]}, "BAG validation/manifest binding drift")
    require(validation.get("checks") and all(flag is True for flag in validation["checks"].values()), "BAG validation checks failed")
    for value, label in ((manifest, "BAG manifest"), (validation, "BAG validation")):
        require(value.get("protected_access_counts") == PROTECTED_ZERO, f"{label} protected drift")
        no_science(value, label)
    require(manifest.get("arm") == "RCDE_BAG" and manifest.get("outer_folds") == list(FOLDS), "BAG manifest scope drift")
    shared = manifest.get("shared_training_contract")
    require(isinstance(shared, dict) and shared.get("seed") == SEED and shared.get("parameter_count") == PARAMETERS, "BAG shared training drift")
    require(shared.get("initial_state_sha256") == INITIAL_STATE_SHA256 and shared.get("model_state_schema_sha256") == MODEL_STATE_SCHEMA_SHA256, "BAG state contract drift")
    common = shared.get("common_bindings")
    require(isinstance(common, dict) and set(common) == set(COMMON_BINDINGS), "BAG common binding set drift")
    for name in COMMON_BINDINGS:
        exact_source(common[name], f"BAG common.{name}")
    closure = manifest.get("cross_fold_closure", {})
    require(closure.get("optimization_reference_universe_count") == 50 and closure.get("eligible_query_union_count") == 594, "BAG 50/594 closure drift")
    folds = manifest.get("folds")
    require(isinstance(folds, list) and len(folds) == 4, "BAG manifest folds missing")
    return {"manifest": manifest_spec, "validation": validation_spec, "common_bindings": common}, folds


def audit_context(entry: Any, source_authority: dict[str, str]) -> tuple[dict[str, Any], dict[str, set[int]], dict[str, Any]]:
    require(isinstance(entry, dict) and set(entry) == CONTEXT_INPUT_KEYS, "CONTEXT fold input schema drift")
    fold = entry.get("outer_fold")
    require(type(fold) is int and fold in FOLDS, "CONTEXT fold must be 1..4")
    specs: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for name in CONTEXT_SOURCES:
        specs[name], paths[name] = exact_source(entry[name], f"CONTEXT fold{fold}.{name}")
    protocol = read_json(paths["protocol"], f"CONTEXT fold{fold} protocol")
    result = read_json(paths["train_result"], f"CONTEXT fold{fold} result")
    validation = read_json(paths["training_validation"], f"CONTEXT fold{fold} validation")
    identity = read_json(paths["run_identity"], f"CONTEXT fold{fold} identity")
    access = read_json(paths["access_manifest"], f"CONTEXT fold{fold} access")
    require(protocol.get("schema_version") == "rc_dino_rcde_r1_main_train_protocol_v1_8_single_scope_20260814", f"CONTEXT fold{fold} protocol schema drift")
    require(result.get("schema_version") == "rc_dino_rcde_r1_main_train_result_v1_8" and result.get("status") == "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAINING_COMPLETE", f"CONTEXT fold{fold} result schema/status drift")
    require(validation.get("schema_version") == "rc_dino_rcde_r1_main_training_validation_v1_8" and validation.get("status") == "DINO_RCDE_R1_MAIN_ARM_FOLD_VALIDATION_PASS", f"CONTEXT fold{fold} validation schema/status drift")
    require(identity.get("schema_version") == "rc_dino_rcde_r1_run_identity_v1_8" and identity.get("status") == "DINO_RCDE_R1_RUN_IDENTITY_FROZEN", f"CONTEXT fold{fold} identity schema/status drift")
    require(access.get("schema_version") == "rc_dino_rcde_signed_reference_access_manifest_v1_8" and access.get("status") == "DINO_RCDE_SIGNED_REFERENCE_ACCESS_FROZEN", f"CONTEXT fold{fold} access schema/status drift")
    for payload, label in ((protocol, "protocol"), (result, "result"), (validation, "validation")):
        no_science(payload, f"CONTEXT fold{fold} {label}")
    require(protocol.get("authority") == {**source_authority, "required_status": SOURCE_CONTEXT_AUTHORITY_STATUS}, f"CONTEXT fold{fold} v16 binding drift")
    require(result.get("arm") == validation.get("arm") == identity.get("arm") == "RCDE_CONTEXT", f"CONTEXT fold{fold} arm drift")
    require(result.get("outer_fold") == validation.get("outer_fold") == identity.get("outer_fold") == access.get("outer_fold") == fold, f"CONTEXT fold{fold} scope drift")
    for payload, label in ((result, "result"), (validation, "validation"), (access, "access")):
        require(payload.get("protected_access_counts") == PROTECTED_ZERO, f"CONTEXT fold{fold} {label} protected drift")
    for payload in (result, validation, identity):
        require(payload.get("protocol_sha256") == specs["protocol"]["sha256"] and payload.get("authority_sha256") == source_authority["sha256"], f"CONTEXT fold{fold} protocol/authority chain drift")
    require(validation.get("train_result_sha256") == specs["train_result"]["sha256"], f"CONTEXT fold{fold} result binding drift")
    require(result.get("run_identity_sha256") == validation.get("run_identity_sha256") == specs["run_identity"]["sha256"], f"CONTEXT fold{fold} identity binding drift")
    require(result.get("signed_reference_access_manifest_sha256") == validation.get("signed_reference_access_manifest_sha256") == specs["access_manifest"]["sha256"], f"CONTEXT fold{fold} access binding drift")
    require(result.get("checkpoint_sha256") == validation.get("checkpoint_sha256") == specs["checkpoint"]["sha256"], f"CONTEXT fold{fold} checkpoint binding drift")
    result_dir = paths["train_result"].parent.resolve()
    require(all(paths[name].parent.resolve() == result_dir for name in ("training_validation", "run_identity", "access_manifest", "checkpoint")), f"CONTEXT fold{fold} artifacts not co-located")
    require((result_dir / str(result.get("checkpoint"))).resolve() == paths["checkpoint"], f"CONTEXT fold{fold} checkpoint path drift")
    require((ROOT / str(protocol.get("resume", {}).get("persistent_output"))).resolve() == result_dir, f"CONTEXT fold{fold} persistent output drift")
    require(result.get("parameter_count") == PARAMETERS and result.get("initial_state_sha256") == INITIAL_STATE_SHA256, f"CONTEXT fold{fold} parameter/initial drift")
    training = result.get("training")
    require(isinstance(training, dict) and training.get("update_count") == validation.get("update_count") == UPDATES, f"CONTEXT fold{fold} update drift")
    require(training.get("pair_count") == validation.get("pair_count") == PAIRS and training.get("seed") == SEED, f"CONTEXT fold{fold} pair/seed drift")
    checks = validation.get("checks")
    require(isinstance(checks, dict) and set(checks) == EXPECTED_CONTEXT_CHECKS and all(flag is True for flag in checks.values()), f"CONTEXT fold{fold} validation checks drift")
    trace = training.get("loss_trace")
    require(isinstance(trace, list) and len(trace) == UPDATES, f"CONTEXT fold{fold} loss trace length drift")
    require([row.get("update") for row in trace if isinstance(row, dict)] == list(range(1, UPDATES + 1)), f"CONTEXT fold{fold} loss trace update drift")
    require(sum(row.get("pair_count", -1) for row in trace) == PAIRS, f"CONTEXT fold{fold} loss trace pair drift")
    require(all(all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) for key in ("mean_pair_loss", "learning_rate", "gradient_norm_before_clip")) for row in trace), f"CONTEXT fold{fold} non-finite training trace")
    chain = result.get("execution_chain")
    require(isinstance(chain, dict) and chain.get("chunk_count") == 1 and chain.get("resume_count") == 0 and chain.get("chunks_contiguous") is True and chain.get("maximum_jobs_respected") is True, f"CONTEXT fold{fold} execution chain drift")
    chunks = chain.get("chunks")
    require(isinstance(chunks, list) and len(chunks) == 1 and chunks[0].get("start_update") == 0 and chunks[0].get("completed_updates") == UPDATES, f"CONTEXT fold{fold} chunk drift")
    match = CHUNK_JOB.fullmatch(str(chunks[0].get("path")))
    expected_job = SCHEDULER_COMPLETION["tasks"][fold - 1]["job_id_raw"]
    require(match is not None and int(match.group(1)) == expected_job, f"CONTEXT fold{fold} JobIDRaw/result chunk drift")
    require(result.get("bag_permutation") == CONTEXT_MODE, f"CONTEXT fold{fold} input mode drift")
    training_contract = training_projection(protocol.get("training"), "RCDE_CONTEXT", fold, f"CONTEXT fold{fold}")
    cache_contract = protocol.get("cache")
    forbidden_contract = protocol.get("forbidden")
    require(isinstance(cache_contract, dict) and isinstance(forbidden_contract, dict), f"CONTEXT fold{fold} cache/forbidden missing")
    bindings = protocol.get("bindings")
    require(isinstance(bindings, dict), f"CONTEXT fold{fold} bindings missing")
    common: dict[str, dict[str, str]] = {}
    for name in COMMON_BINDINGS:
        common[name], _ = exact_source(bindings.get(name), f"CONTEXT fold{fold}.common.{name}")
        require(validation.get("binding_hashes", {}).get(name) == common[name]["sha256"], f"CONTEXT fold{fold} common binding drift: {name}")
    logical = logical_access(access, f"CONTEXT fold{fold} access")
    eligible = integer_set(access.get("eligible_query_execution_ordinals"), f"CONTEXT fold{fold} eligible")
    allowed = integer_set(access.get("allowed_reference_physical_rows"), f"CONTEXT fold{fold} allowed")
    heldout = integer_set(access.get("heldout_reference_physical_rows"), f"CONTEXT fold{fold} heldout")
    require(len(eligible) == access.get("eligible_query_count") == validation.get("eligible_query_count"), f"CONTEXT fold{fold} eligible count drift")
    require(len(allowed) == access.get("allowed_reference_count") == validation.get("allowed_reference_count"), f"CONTEXT fold{fold} allowed count drift")
    require(len(heldout) == access.get("heldout_reference_count") and not (allowed & heldout), f"CONTEXT fold{fold} heldout access drift")
    require(access.get("reference_row_intersection_count") == validation.get("heldout_reference_intersection_count") == 0, f"CONTEXT fold{fold} heldout intersection")
    runtime = result.get("runtime_access", {})
    require(integer_set(runtime.get("query_read_union"), f"CONTEXT fold{fold} runtime query") == eligible, f"CONTEXT fold{fold} runtime query set drift")
    require(integer_set(runtime.get("reference_read_union"), f"CONTEXT fold{fold} runtime reference") == allowed and runtime.get("reference_read_union_equals_signed_allowlist") is True, f"CONTEXT fold{fold} runtime reference set drift")
    record = {
        "outer_fold": fold, **specs, "access_manifest": {**specs["access_manifest"], "logical_sha256": logical},
        "schedule_sha256": training["schedule_sha256"], "initial_state_sha256": result["initial_state_sha256"],
        "final_state_sha256": result["final_state_sha256"], "update_count": UPDATES, "pair_count": PAIRS,
    }
    meta = {"training_contract": training_contract, "cache": cache_contract, "forbidden": forbidden_contract,
            "training_receipt": {k: v for k, v in training.items() if k != "loss_trace"}, "common": common,
            "bag_mode": result["bag_permutation"]}
    return record, {"eligible": eligible, "allowed": allowed, "heldout": heldout}, meta


def match_fold(bag: Mapping[str, Any], context: dict[str, Any], context_sets: dict[str, set[int]], context_meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, set[int]]]:
    fold = context["outer_fold"]
    require(bag.get("outer_fold") == fold and bag.get("update_count") == UPDATES and bag.get("pair_count") == PAIRS, f"BAG fold{fold} record drift")
    bag_sources: dict[str, dict[str, str]] = {}
    bag_paths: dict[str, Path] = {}
    for name in ("authority", "protocol", "run_identity", "train_result", "training_validation", "checkpoint"):
        bag_sources[name], bag_paths[name] = source_spec_from_manifest(bag.get(name), f"BAG fold{fold}.{name}")
    bag_sources["access_manifest"], bag_paths["access_manifest"] = source_spec_from_manifest(bag.get("access_manifest"), f"BAG fold{fold}.access", logical=True)
    bag_protocol = read_json(bag_paths["protocol"], f"BAG fold{fold} protocol")
    bag_result = read_json(bag_paths["train_result"], f"BAG fold{fold} result")
    bag_access = read_json(bag_paths["access_manifest"], f"BAG fold{fold} access")
    require(bag_result.get("arm") == "RCDE_BAG" and bag_result.get("outer_fold") == fold, f"BAG fold{fold} result scope drift")
    require(bag_result.get("parameter_count") == PARAMETERS and bag_result.get("protected_access_counts") == PROTECTED_ZERO, f"BAG fold{fold} parameter/protected drift")
    require(bag_result.get("bag_permutation") == BAG_MODE, f"BAG fold{fold} input mode drift")
    no_science(bag_result, f"BAG fold{fold} result")
    bag_training = bag_result.get("training")
    require(isinstance(bag_training, dict) and bag_training.get("update_count") == UPDATES and bag_training.get("pair_count") == PAIRS and bag_training.get("seed") == SEED, f"BAG fold{fold} training receipt drift")
    bag_contract = training_projection(bag_protocol.get("training"), "RCDE_BAG", fold, f"BAG fold{fold}")
    require(bag_contract == context_meta["training_contract"], f"fold{fold} normalized training contract mismatch")
    require(bag_protocol.get("cache") == context_meta["cache"] and bag_protocol.get("forbidden") == context_meta["forbidden"], f"fold{fold} cache/forbidden mismatch")
    require({k: v for k, v in bag_training.items() if k != "loss_trace"} == context_meta["training_receipt"], f"fold{fold} schedule/input training receipt mismatch")
    require(bag.get("schedule_sha256") == context["schedule_sha256"] and bag.get("initial_state_sha256") == context["initial_state_sha256"] == INITIAL_STATE_SHA256, f"fold{fold} schedule/initial mismatch")
    bindings = bag_protocol.get("bindings")
    require(isinstance(bindings, dict), f"BAG fold{fold} bindings missing")
    for name in COMMON_BINDINGS:
        source, _ = exact_source(bindings.get(name), f"BAG fold{fold}.common.{name}")
        require(source == context_meta["common"][name], f"fold{fold} common binding mismatch: {name}")
    logical_access(bag_access, f"BAG fold{fold} access")
    bag_sets = {
        "eligible": integer_set(bag_access.get("eligible_query_execution_ordinals"), f"BAG fold{fold} eligible"),
        "allowed": integer_set(bag_access.get("allowed_reference_physical_rows"), f"BAG fold{fold} allowed"),
        "heldout": integer_set(bag_access.get("heldout_reference_physical_rows"), f"BAG fold{fold} heldout"),
    }
    require(bag_sets == context_sets, f"fold{fold} BAG/CONTEXT eligible/allowed/heldout sets differ")
    bag_mode_without_toggle = dict(BAG_MODE); bag_mode_without_toggle.pop("enabled")
    context_mode_without_toggle = dict(context_meta["bag_mode"]); context_mode_without_toggle.pop("enabled")
    require(bag_mode_without_toggle == context_mode_without_toggle, f"fold{fold} non-toggle input mode mismatch")
    matched = {
        "outer_fold": fold,
        "bag": {**bag_sources, "schedule_sha256": bag["schedule_sha256"], "initial_state_sha256": bag["initial_state_sha256"], "final_state_sha256": bag["final_state_sha256"]},
        "context": context,
        "matched_contract": {
            "schedule_sha256": context["schedule_sha256"], "initial_state_sha256": INITIAL_STATE_SHA256,
            "eligible_query_count": len(context_sets["eligible"]), "allowed_reference_count": len(context_sets["allowed"]),
            "heldout_reference_count": len(context_sets["heldout"]),
            "eligible_query_set_sha256": canonical_sha256(sorted(context_sets["eligible"])),
            "allowed_reference_set_sha256": canonical_sha256(sorted(context_sets["allowed"])),
            "heldout_reference_set_sha256": canonical_sha256(sorted(context_sets["heldout"])),
            "common_bindings_equal": True, "normalized_training_cache_forbidden_equal": True,
            "training_receipt_except_loss_trace_equal": True,
            "only_model_input_space_difference": {"field": "bag_permutation.enabled", "RCDE_BAG": True, "RCDE_CONTEXT": False},
            "allowed_non_model_differences": ["authority", "protocol", "artifact_path", "runtime_provenance", "trained_state_and_loss_trace"],
        },
    }
    return matched, context_sets


def build_manifest(protocol_path: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    require(protocol_path.is_file() and not protocol_path.is_symlink(), "matched protocol missing/symlinked")
    protocol = read_json(protocol_path, "matched protocol")
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA and protocol.get("stage") == PROTOCOL_STAGE, "matched protocol schema/stage drift")
    require(protocol.get("claim_level") == CLAIM_LEVEL and protocol.get("forbidden") == FORBIDDEN, "matched protocol boundary drift")
    no_science(protocol, "matched protocol")
    authority_spec, authority_path = required_source(protocol.get("authority"), "matched authority")
    authority = read_json(authority_path, "matched authority")
    require(authority.get("status") == authority_spec["required_status"] and authority.get("next_authorized_stage") == PROTOCOL_STAGE, "matched authority status/stage drift")
    require(authority.get("natural_training_authorized") is False and authority.get("protected_access_counts") == PROTECTED_ZERO, "matched authority training/protected drift")
    no_science(authority, "matched authority")
    implementations = protocol.get("implementation_bindings")
    require(isinstance(implementations, dict) and set(implementations) == {"materializer", "validator", "tests"}, "implementation binding schema drift")
    for name in ("materializer", "validator", "tests"):
        exact_source(implementations[name], f"implementation.{name}")
    output_text = protocol.get("output_dir")
    require(isinstance(output_text, str) and output_text != "" and not Path(output_text).is_absolute(), "output_dir missing/absolute")
    output_dir = (ROOT / output_text).resolve()
    require(output_dir.is_relative_to(ROOT.resolve()), "output_dir escapes RC root")
    bag_source, bag_folds = audit_bag_foldset(protocol)
    raw_context = protocol.get("context_fold_inputs")
    require(isinstance(raw_context, list) and len(raw_context) == 4, "four CONTEXT inputs required")
    context_protocol_specs = [row.get("protocol") for row in sorted(raw_context, key=lambda row: row.get("outer_fold", -1)) if isinstance(row, dict)]
    require(len(context_protocol_specs) == 4 and all(isinstance(row, dict) for row in context_protocol_specs), "CONTEXT protocol list malformed")
    scheduler = audit_scheduler(protocol, context_protocol_specs)
    source_authority = {"path": scheduler["source_context_authority"]["path"], "sha256": scheduler["source_context_authority"]["sha256"]}
    contexts = [audit_context(row, source_authority) for row in raw_context]
    contexts.sort(key=lambda row: row[0]["outer_fold"])
    require(tuple(row[0]["outer_fold"] for row in contexts) == FOLDS, "CONTEXT folds are not exactly 1..4")
    bag_by_fold = {row.get("outer_fold"): row for row in bag_folds if isinstance(row, dict)}
    require(set(bag_by_fold) == set(FOLDS), "BAG folds are not exactly 1..4")
    matched_rows, set_rows = [], []
    for record, sets, meta in contexts:
        matched, observed_sets = match_fold(bag_by_fold[record["outer_fold"]], record, sets, meta)
        matched_rows.append(matched); set_rows.append(observed_sets)
    common = contexts[0][2]["common"]
    require(all(row[2]["common"] == common for row in contexts), "CONTEXT common bindings differ across folds")
    require(common == bag_source["common_bindings"], "BAG/CONTEXT foldset common bindings differ")
    universes = [row["allowed"] | row["heldout"] for row in set_rows]
    require(all(value == universes[0] for value in universes) and len(universes[0]) == 50, "50-reference universe closure drift")
    heldout = [row["heldout"] for row in set_rows]
    require([len(value) for value in heldout] == [13, 12, 12, 13], "heldout reference counts drift")
    membership = Counter(item for values in heldout for item in values)
    require(set(membership) == universes[0] and set(membership.values()) == {1}, "heldout reference partition drift")
    require(all(row["allowed"] == universes[0] - row["heldout"] for row in set_rows), "allowed reference complement drift")
    eligible = [row["eligible"] for row in set_rows]
    eligible_union = set().union(*eligible)
    query_membership = Counter(item for values in eligible for item in values)
    require(len(eligible_union) == 594 and set(query_membership.values()) == {3}, "594-query outer-train closure drift")
    globally_ineligible = sorted(set(range(600)) - eligible_union)
    require(globally_ineligible == [25, 26, 101, 346, 354, 470], "global ineligible query drift")
    return {
        "schema_version": MANIFEST_SCHEMA, "status": MANIFEST_STATUS,
        "arms": ["RCDE_BAG", "RCDE_CONTEXT"], "outer_folds": list(FOLDS),
        "source_mode": "immutable_paths_and_sha256_no_copy", "authority": authority_spec,
        "matched_foldset_protocol": {"path": str(protocol_path.relative_to(ROOT.resolve())), "sha256": file_sha256(protocol_path)},
        "claim_level": CLAIM_LEVEL, "scientific_GO_or_NO_GO": None, "automatic_stage_advance": False,
        "protected_access_counts": PROTECTED_ZERO, "forbidden": FORBIDDEN,
        "bag_foldset": {"manifest": bag_source["manifest"], "validation": bag_source["validation"]},
        "source_context_authority": scheduler["source_context_authority"],
        "runtime_scheduler_provenance": {"amendment": scheduler["amendment"], "receipt": scheduler["receipt"], "completion": scheduler["completion"]},
        "shared_training_contract": {"seed": SEED, "parameter_count": PARAMETERS, "updates_per_arm_fold": UPDATES,
                                     "pairs_per_arm_fold": PAIRS, "initial_state_sha256": INITIAL_STATE_SHA256,
                                     "model_state_schema_sha256": MODEL_STATE_SCHEMA_SHA256, "common_bindings": common},
        "folds": matched_rows,
        "cross_fold_closure": {
            "optimization_reference_universe_count": 50, "heldout_reference_counts_by_fold": {"1": 13, "2": 12, "3": 12, "4": 13},
            "heldout_pairwise_intersection_count": 0, "heldout_union_count": 50, "heldout_membership_count_per_reference": 1,
            "allowed_is_universe_minus_heldout": True, "eligible_query_union_count": 594,
            "eligible_query_outer_train_membership_count": 3, "globally_ineligible_execution_ordinals": globally_ineligible,
            "scope": {"query_population": "600_query_training_role", "reference_population": "50_row_optimization_reference_universe",
                      "full_gallery_evaluation": False, "outer_heldout_forward": False, "heldout_label_join": False, "scientific_reduction": False},
        },
        "checkpoint_files_copied": 0, "heldout_forward_count": 0, "heldout_label_join_count": 0,
        "scientific_metric_count": 0, "next_required_stage": MANIFEST_NEXT,
    }


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, f"immutable output drift: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with partial.open("xb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def materialize(protocol_path: Path, output_dir: Path) -> Path:
    manifest = build_manifest(protocol_path)
    protocol = read_json(protocol_path.resolve(), "matched protocol")
    expected_dir = (ROOT / str(protocol.get("output_dir"))).resolve()
    output_dir = output_dir.resolve()
    require(output_dir == expected_dir and output_dir.is_relative_to(ROOT.resolve()), "CLI output-dir differs from protocol")
    output = output_dir / "matched_foldset_manifest.json"
    require(not (output_dir / "matched_foldset_validation.json").exists(), "validation exists before materialization")
    write_once(output, manifest)
    require(not output_dir.is_symlink() and not output.is_symlink(), "matched output may not be symlinked")
    require({path.name for path in output_dir.iterdir()} == {"matched_foldset_manifest.json"}, "materializer emitted undeclared files")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze matched BAG/CONTEXT engineering folds")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = materialize(args.protocol, args.output_dir)
    print(json.dumps({"manifest": str(path), "status": MANIFEST_STATUS}, sort_keys=True))


if __name__ == "__main__":
    main()
