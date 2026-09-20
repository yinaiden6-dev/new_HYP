#!/usr/bin/env python3
"""Independent validation of matched BAG/CONTEXT engineering fold closure."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import uuid


# Intentionally independent: this module does not import the materializer.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOL_SCHEMA = "rc_dino_rcde_r1_matched_foldset_protocol_v1_20260814"
PROTOCOL_STAGE = "R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_CLOSURE"
MANIFEST_SCHEMA = "rc_dino_rcde_r1_matched_foldset_manifest_v1_0"
MANIFEST_STATUS = "DINO_RCDE_R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_FROZEN"
VALIDATION_SCHEMA = "rc_dino_rcde_r1_matched_foldset_validation_v1_0"
VALIDATION_STATUS = "DINO_RCDE_R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_VALIDATION_PASS"
CLAIM_LEVEL = "engineering_matched_foldset_closure_only_not_scientific_GO_or_NO_GO"
MANIFEST_NEXT = "R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_VALIDATION"
VALIDATION_NEXT = "R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_VALIDATED_AUTHORITY_UPDATE_REQUIRED"
FOLDS = (1, 2, 3, 4)
SEED, PARAMETERS, UPDATES, PAIRS = 17, 65_125, 2_048, 65_536
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
COMMON_BINDINGS = ("scientific_contract", "cache_protocol", "cache_result", "cache_validation", "training_roles", "episode_ledger", "resource_shape_ledger", "model_visible_c128", "rcde_core")
CONTEXT_SOURCES = ("protocol", "run_identity", "access_manifest", "train_result", "training_validation", "checkpoint")
PROTECTED_ZERO = {"C8_runtime_read_count": 0, "S8_runtime_read_count": 0, "home_files_modified": 0, "opened_runtime_read_count": 0, "sealed_runtime_read_count": 0, "unauthorized_natural_result_read_count": 0}
FORBIDDEN = {"checkpoint_copy": False, "outer_heldout_forward": False, "heldout_label_join": False, "scientific_reduction": False, "scientific_decision": False, "protected_paths": ["C8", "S8", "opened", "sealed"], "home_write": False}
SCHEDULER_COMPLETION = {"source": "live_slurm_accounting_read_only_observation_20260814", "array_job_id": 5_072_108, "tasks": [
    {"array_task_id": 1, "job_id_raw": 5_072_109, "state": "COMPLETED", "exit_code": "0:0"},
    {"array_task_id": 2, "job_id_raw": 5_072_110, "state": "COMPLETED", "exit_code": "0:0"},
    {"array_task_id": 3, "job_id_raw": 5_072_111, "state": "COMPLETED", "exit_code": "0:0"},
    {"array_task_id": 4, "job_id_raw": 5_072_108, "state": "COMPLETED", "exit_code": "0:0"}], "validator_queries_live_slurm": False}
BAG_MODE = {"enabled": True, "fixed_point_free_cyclic_shift": True, "query_binding_namespace": "DINO_RCDE_R1_BAG_QUERY_PERMUTATION_V1_5", "reference_binding_namespace": "DINO_RCDE_R1_BAG_REFERENCE_PERMUTATION_V1_5", "valid_tokens_only": True}
CONTEXT_MODE = {**BAG_MODE, "enabled": False}
EXPECTED_CONTEXT_CHECKS = {
    "all_artifact_schemas_and_statuses_protocol_declared", "all_chunk_ranges_contiguous_and_bounded", "all_preflight_receipts_exact", "all_runtime_read_ledgers_match_schedule", "atomic_chunk_directory_transactions_exact", "cache_execution_ordinals_exact_0_599", "candidate_directions_recomputed_from_P0_frozen_order", "canonical_cache_validation_pass", "checkpoint_and_completed_resume_bound", "eligible_cache_payload_sources_models_and_logical_hashes", "exact_protected_zero_key_sets", "finalization_bound_to_last_committed_completed_state", "identity_supergroup_and_reference_heldout_exclusion", "learning_rates_and_pair_counts_recomputed", "no_scientific_decision_or_automatic_advance", "optimizer_completed_2048_updates", "protocol_authority_and_all_bindings", "query_id_historical_execution_mapping_1800_of_1800", "query_order_recomputed_from_source_sha256", "run_identity_exact", "runtime_reference_union_equals_allowlist", "schedule_2048_updates_recomputed", "signed_reference_access_manifest_exact", "single_arm_single_fold_protocol_scope_exact"}
CHECKPOINT_V17_FIELDS = {"schema_version", "arm", "outer_fold", "update", "seed", "protocol_sha256", "authority_sha256", "initial_state_sha256", "final_state_sha256", "model_state_dict"}
CHECKPOINT_V18_FIELDS = CHECKPOINT_V17_FIELDS | {"status"}


class MatchedFoldsetValidationAbort(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedFoldsetValidationAbort(message)


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
        raise MatchedFoldsetValidationAbort(f"invalid {context}: {path}") from exc
    require(isinstance(value, dict), f"{context} must be object")
    return value


def resolve_relative(text: str) -> Path:
    require(isinstance(text, str) and text and not Path(text).is_absolute(), "source path must be RC-relative")
    path = (ROOT / text).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file() and not path.is_symlink(), f"invalid source: {text}")
    return path


def exact_source(value: Any, context: str, *, logical: bool = False) -> tuple[dict[str, str], Path]:
    keys = {"path", "sha256", "logical_sha256"} if logical else {"path", "sha256"}
    require(isinstance(value, dict) and set(value) == keys, f"{context} source schema drift")
    require(isinstance(value.get("sha256"), str) and HEX64.fullmatch(value["sha256"]), f"{context} digest malformed")
    path = resolve_relative(value.get("path"))
    require(file_sha256(path) == value["sha256"], f"{context} physical digest drift")
    result = {"path": value["path"], "sha256": value["sha256"]}
    if logical:
        require(isinstance(value.get("logical_sha256"), str) and HEX64.fullmatch(value["logical_sha256"]), f"{context} logical digest malformed")
        result["logical_sha256"] = value["logical_sha256"]
    return result, path


def required_source(value: Any, context: str) -> tuple[dict[str, str], Path]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256", "required_status"}, f"{context} required source schema drift")
    source, path = exact_source({"path": value["path"], "sha256": value["sha256"]}, context)
    require(isinstance(value["required_status"], str), f"{context} status missing")
    return {**source, "required_status": value["required_status"]}, path


def no_science(value: Mapping[str, Any], context: str) -> None:
    require(value.get("scientific_GO_or_NO_GO") is None and value.get("automatic_stage_advance") is False, f"{context} scientific/advance boundary drift")


def integer_set(value: Any, context: str) -> set[int]:
    require(isinstance(value, list) and all(type(item) is int for item in value) and len(value) == len(set(value)), f"{context} malformed")
    return set(value)


def access_sets(access: Mapping[str, Any], context: str) -> dict[str, set[int]]:
    detached = dict(access); observed = detached.pop("logical_sha256", None)
    require(isinstance(observed, str) and canonical_sha256(detached) == observed, f"{context} logical digest drift")
    return {"eligible": integer_set(access.get("eligible_query_execution_ordinals"), f"{context}.eligible"), "allowed": integer_set(access.get("allowed_reference_physical_rows"), f"{context}.allowed"), "heldout": integer_set(access.get("heldout_reference_physical_rows"), f"{context}.heldout")}


def state_dict_sha256(state: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), f"checkpoint state value not tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0"); digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(canonical_bytes(list(tensor.shape)) + b"\0"); digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_audit(path: Path, *, arm: str, fold: int, protocol_sha: str, authority_sha: str, result: Mapping[str, Any], expected_schema: list[list[Any]] | None) -> list[list[Any]]:
    try:
        import torch
        from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2
    except Exception as exc:  # pragma: no cover
        raise MatchedFoldsetValidationAbort("torch/RCDE core unavailable") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), f"{arm} fold{fold} checkpoint not object")
    if checkpoint.get("schema_version") == "rc_dino_rcde_r1_main_checkpoint_v1_7":
        require(arm == "RCDE_BAG" and fold == 1 and set(checkpoint) == CHECKPOINT_V17_FIELDS, "V1.7 checkpoint schema/field drift")
    else:
        require(checkpoint.get("schema_version") == "rc_dino_rcde_r1_main_checkpoint_v1_8" and set(checkpoint) == CHECKPOINT_V18_FIELDS, f"{arm} fold{fold} V1.8 checkpoint field drift")
        require(checkpoint.get("status") == "DINO_RCDE_R1_MAIN_CHECKPOINT_COMPLETE", f"{arm} fold{fold} checkpoint status drift")
    require(checkpoint.get("arm") == arm and checkpoint.get("outer_fold") == fold and checkpoint.get("update") == UPDATES and checkpoint.get("seed") == SEED, f"{arm} fold{fold} checkpoint scope/update drift")
    require(checkpoint.get("protocol_sha256") == protocol_sha and checkpoint.get("authority_sha256") == authority_sha, f"{arm} fold{fold} checkpoint provenance drift")
    state = checkpoint.get("model_state_dict")
    require(isinstance(state, dict) and len(state) == 17, f"{arm} fold{fold} state cardinality drift")
    schema = [[name, list(value.shape), str(value.dtype)] for name, value in state.items()]
    require(canonical_sha256(schema) == MODEL_STATE_SCHEMA_SHA256 and sum(int(v.numel()) for v in state.values()) == PARAMETERS, f"{arm} fold{fold} state schema/parameter drift")
    if expected_schema is not None:
        require(schema == expected_schema, f"{arm} fold{fold} ordered state schema differs")
    torch.manual_seed(SEED); initial = state_dict_sha256(DINO_RCDE_V1_2().state_dict(), torch); final = state_dict_sha256(state, torch)
    require(initial == INITIAL_STATE_SHA256, "independent initialization replay drift")
    require(checkpoint.get("initial_state_sha256") == result.get("initial_state_sha256") == initial, f"{arm} fold{fold} initial state binding drift")
    require(checkpoint.get("final_state_sha256") == result.get("final_state_sha256") == final and final != initial, f"{arm} fold{fold} final state binding drift")
    return schema


def normalized_training(protocol: Mapping[str, Any], arm: str, fold: int, context: str) -> dict[str, Any]:
    training = protocol.get("training")
    require(isinstance(training, dict) and training.get("arms") == [arm] and training.get("outer_folds") == [fold], f"{context} training scope drift")
    require(training.get("seed") == SEED and training.get("parameter_count") == PARAMETERS and training.get("updates_total") == UPDATES, f"{context} training constants drift")
    require(training.get("outer_train_only") is True and training.get("heldout_target_join_allowed") is False, f"{context} outer-train boundary drift")
    result = dict(training); result["arms"] = ["RCDE_MATCHED_ARM"]
    return result


def validate(protocol_path: Path, manifest_path: Path, output_path: Path) -> Path:
    protocol_path, manifest_path, output_path = protocol_path.resolve(), manifest_path.resolve(), output_path.resolve()
    protocol = read_json(protocol_path, "matched protocol")
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA and protocol.get("stage") == PROTOCOL_STAGE and protocol.get("claim_level") == CLAIM_LEVEL, "matched protocol schema/stage/claim drift")
    require(protocol.get("forbidden") == FORBIDDEN and protocol.get("scheduler_completion") == SCHEDULER_COMPLETION, "matched protocol forbidden/completion drift")
    no_science(protocol, "matched protocol")
    required_dir = (ROOT / str(protocol.get("output_dir"))).resolve()
    require(required_dir.is_relative_to(ROOT.resolve()) and manifest_path == required_dir / "matched_foldset_manifest.json" and output_path == required_dir / "matched_foldset_validation.json", "matched output paths drift")
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "matched manifest missing/symlinked")
    manifest = read_json(manifest_path, "matched manifest")
    require(manifest.get("schema_version") == MANIFEST_SCHEMA and manifest.get("status") == MANIFEST_STATUS and manifest.get("claim_level") == CLAIM_LEVEL, "matched manifest schema/status/claim drift")
    require(manifest.get("arms") == ["RCDE_BAG", "RCDE_CONTEXT"] and manifest.get("outer_folds") == list(FOLDS), "matched manifest scope drift")
    require(manifest.get("matched_foldset_protocol") == {"path": str(protocol_path.relative_to(ROOT.resolve())), "sha256": file_sha256(protocol_path)}, "manifest/protocol binding drift")
    require(manifest.get("forbidden") == FORBIDDEN and manifest.get("protected_access_counts") == PROTECTED_ZERO, "manifest forbidden/protected drift")
    no_science(manifest, "matched manifest")
    require(manifest.get("checkpoint_files_copied") == manifest.get("heldout_forward_count") == manifest.get("heldout_label_join_count") == manifest.get("scientific_metric_count") == 0, "manifest prohibited action count drift")
    require(manifest.get("next_required_stage") == MANIFEST_NEXT, "manifest next stage drift")

    authority_spec, authority_path = required_source(protocol.get("authority"), "matched authority")
    authority = read_json(authority_path, "matched authority")
    require(authority.get("status") == authority_spec["required_status"] and authority.get("next_authorized_stage") == PROTOCOL_STAGE and authority.get("natural_training_authorized") is False, "matched authority drift")
    require(authority.get("protected_access_counts") == PROTECTED_ZERO, "matched authority protected drift"); no_science(authority, "matched authority")
    require(manifest.get("authority") == authority_spec, "manifest matched authority binding drift")
    implementations = protocol.get("implementation_bindings")
    require(isinstance(implementations, dict) and set(implementations) == {"materializer", "validator", "tests"}, "implementation binding drift")
    for name in implementations: exact_source(implementations[name], f"implementation.{name}")

    source_authority_spec, source_authority_path = required_source(protocol.get("source_context_authority"), "v16 authority")
    require((source_authority_spec["path"], source_authority_spec["sha256"], source_authority_spec["required_status"]) == (SOURCE_CONTEXT_AUTHORITY_PATH, SOURCE_CONTEXT_AUTHORITY_SHA256, SOURCE_CONTEXT_AUTHORITY_STATUS), "v16 binding drift")
    source_authority = read_json(source_authority_path, "v16 authority")
    require(source_authority.get("status") == SOURCE_CONTEXT_AUTHORITY_STATUS and source_authority.get("protected_access_counts") == PROTECTED_ZERO, "v16 authority status/protected drift"); no_science(source_authority, "v16 authority")
    amendment_spec, amendment_path = required_source(protocol.get("runtime_scheduler_amendment"), "amendment")
    receipt_spec, receipt_path = required_source(protocol.get("runtime_scheduler_receipt"), "receipt")
    require((amendment_spec["path"], amendment_spec["sha256"], amendment_spec["required_status"]) == (AMENDMENT_PATH, AMENDMENT_SHA256, AMENDMENT_STATUS), "amendment binding drift")
    require((receipt_spec["path"], receipt_spec["sha256"], receipt_spec["required_status"]) == (RECEIPT_PATH, RECEIPT_SHA256, RECEIPT_STATUS), "receipt binding drift")
    amendment, receipt = read_json(amendment_path, "amendment"), read_json(receipt_path, "receipt")
    require(amendment.get("status") == AMENDMENT_STATUS and amendment.get("job_id") == 5_072_108 and amendment.get("scheduler_only_override", {}).get("new_value") == 4, "amendment semantic drift")
    require(amendment.get("protected_access_counts") == PROTECTED_ZERO and amendment.get("invariants_unchanged") and all(v is True for v in amendment["invariants_unchanged"].values()), "amendment invariant/protected drift"); no_science(amendment, "amendment")
    require(receipt.get("status") == RECEIPT_STATUS and receipt.get("job_id") == 5_072_108 and receipt.get("amendment") == {"path": amendment_spec["path"], "sha256": amendment_spec["sha256"]}, "receipt semantic drift")
    require(receipt.get("model_or_data_files_modified") == receipt.get("home_files_modified") == 0 and receipt.get("scientific_GO_or_NO_GO") is None, "receipt mutation/science drift")
    require(manifest.get("source_context_authority") == source_authority_spec and manifest.get("runtime_scheduler_provenance") == {"amendment": amendment_spec, "receipt": receipt_spec, "completion": SCHEDULER_COMPLETION}, "manifest scheduler provenance drift")

    bag_container = protocol.get("bag_foldset")
    require(isinstance(bag_container, dict) and set(bag_container) == {"manifest", "validation"}, "BAG foldset input drift")
    bag_manifest_spec, bag_manifest_path = required_source(bag_container["manifest"], "BAG manifest")
    bag_validation_spec, bag_validation_path = required_source(bag_container["validation"], "BAG validation")
    require((bag_manifest_spec["path"], bag_manifest_spec["sha256"], bag_manifest_spec["required_status"]) == (BAG_MANIFEST_PATH, BAG_MANIFEST_SHA256, BAG_MANIFEST_STATUS), "BAG manifest binding drift")
    require((bag_validation_spec["path"], bag_validation_spec["sha256"], bag_validation_spec["required_status"]) == (BAG_VALIDATION_PATH, BAG_VALIDATION_SHA256, BAG_VALIDATION_STATUS), "BAG validation binding drift")
    bag_manifest, bag_validation = read_json(bag_manifest_path, "BAG manifest"), read_json(bag_validation_path, "BAG validation")
    require(bag_manifest.get("status") == BAG_MANIFEST_STATUS and bag_validation.get("status") == BAG_VALIDATION_STATUS, "BAG foldset status drift")
    require(bag_validation.get("foldset_manifest") == {"path": bag_manifest_spec["path"], "sha256": bag_manifest_spec["sha256"]} and bag_validation.get("checks") and all(v is True for v in bag_validation["checks"].values()), "BAG foldset validation drift")
    require(bag_manifest.get("protected_access_counts") == bag_validation.get("protected_access_counts") == PROTECTED_ZERO, "BAG foldset protected drift"); no_science(bag_manifest, "BAG manifest"); no_science(bag_validation, "BAG validation")
    require(manifest.get("bag_foldset") == {"manifest": bag_manifest_spec, "validation": bag_validation_spec}, "manifest BAG binding drift")
    bag_by_fold = {row.get("outer_fold"): row for row in bag_manifest.get("folds", []) if isinstance(row, dict)}
    contexts = protocol.get("context_fold_inputs")
    require(isinstance(contexts, list) and len(contexts) == 4 and set(bag_by_fold) == set(FOLDS), "source folds missing")
    context_by_fold = {row.get("outer_fold"): row for row in contexts if isinstance(row, dict)}
    require(set(context_by_fold) == set(FOLDS), "CONTEXT folds not exactly 1..4")
    require(amendment.get("protocols") == [context_by_fold[f]["protocol"] for f in FOLDS], "amendment/context protocol order drift")
    observed_rows = manifest.get("folds")
    require(isinstance(observed_rows, list) and [row.get("outer_fold") for row in observed_rows] == list(FOLDS), "manifest fold ordering drift")
    observed_by_fold = {row["outer_fold"]: row for row in observed_rows}
    schema: list[list[Any]] | None = None
    all_sets, common_expected = [], bag_manifest.get("shared_training_contract", {}).get("common_bindings")
    require(isinstance(common_expected, dict) and set(common_expected) == set(COMMON_BINDINGS), "BAG common bindings drift")
    for name in COMMON_BINDINGS: exact_source(common_expected[name], f"BAG common.{name}")

    for fold in FOLDS:
        context_entry, bag_entry = context_by_fold[fold], bag_by_fold[fold]
        require(set(context_entry) == {"outer_fold", *CONTEXT_SOURCES}, f"CONTEXT fold{fold} input fields drift")
        c_specs, c_paths = {}, {}
        for name in CONTEXT_SOURCES: c_specs[name], c_paths[name] = exact_source(context_entry[name], f"CONTEXT fold{fold}.{name}")
        c_protocol, c_result = read_json(c_paths["protocol"], f"CONTEXT fold{fold} protocol"), read_json(c_paths["train_result"], f"CONTEXT fold{fold} result")
        c_validation, c_identity, c_access = read_json(c_paths["training_validation"], f"CONTEXT fold{fold} validation"), read_json(c_paths["run_identity"], f"CONTEXT fold{fold} identity"), read_json(c_paths["access_manifest"], f"CONTEXT fold{fold} access")
        require(c_protocol.get("schema_version") == "rc_dino_rcde_r1_main_train_protocol_v1_8_single_scope_20260814" and c_result.get("schema_version") == "rc_dino_rcde_r1_main_train_result_v1_8" and c_result.get("status") == "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAINING_COMPLETE", f"CONTEXT fold{fold} protocol/result drift")
        require(c_validation.get("schema_version") == "rc_dino_rcde_r1_main_training_validation_v1_8" and c_validation.get("status") == "DINO_RCDE_R1_MAIN_ARM_FOLD_VALIDATION_PASS", f"CONTEXT fold{fold} validation drift")
        require(c_identity.get("schema_version") == "rc_dino_rcde_r1_run_identity_v1_8" and c_identity.get("status") == "DINO_RCDE_R1_RUN_IDENTITY_FROZEN", f"CONTEXT fold{fold} identity drift")
        require(c_access.get("schema_version") == "rc_dino_rcde_signed_reference_access_manifest_v1_8" and c_access.get("status") == "DINO_RCDE_SIGNED_REFERENCE_ACCESS_FROZEN", f"CONTEXT fold{fold} access drift")
        for value, label in ((c_protocol, "protocol"), (c_result, "result"), (c_validation, "validation")): no_science(value, f"CONTEXT fold{fold} {label}")
        require(c_protocol.get("authority") == {"path": source_authority_spec["path"], "sha256": source_authority_spec["sha256"], "required_status": SOURCE_CONTEXT_AUTHORITY_STATUS}, f"CONTEXT fold{fold} authority anchor drift")
        require(c_result.get("arm") == c_validation.get("arm") == c_identity.get("arm") == "RCDE_CONTEXT" and c_result.get("outer_fold") == c_validation.get("outer_fold") == c_identity.get("outer_fold") == c_access.get("outer_fold") == fold, f"CONTEXT fold{fold} scope drift")
        require(c_result.get("protected_access_counts") == c_validation.get("protected_access_counts") == c_access.get("protected_access_counts") == PROTECTED_ZERO, f"CONTEXT fold{fold} protected drift")
        require(c_result.get("parameter_count") == PARAMETERS and c_result.get("initial_state_sha256") == INITIAL_STATE_SHA256, f"CONTEXT fold{fold} parameter/initial drift")
        c_training = c_result.get("training")
        require(isinstance(c_training, dict) and c_training.get("update_count") == c_validation.get("update_count") == UPDATES and c_training.get("pair_count") == c_validation.get("pair_count") == PAIRS and c_training.get("seed") == SEED, f"CONTEXT fold{fold} update/pair/seed drift")
        checks = c_validation.get("checks"); require(isinstance(checks, dict) and set(checks) == EXPECTED_CONTEXT_CHECKS and all(v is True for v in checks.values()), f"CONTEXT fold{fold} checks drift")
        trace = c_training.get("loss_trace"); require(isinstance(trace, list) and len(trace) == UPDATES and [row.get("update") for row in trace] == list(range(1, UPDATES + 1)) and sum(row.get("pair_count", -1) for row in trace) == PAIRS, f"CONTEXT fold{fold} trace coverage drift")
        require(all(math.isfinite(float(row[key])) for row in trace for key in ("mean_pair_loss", "learning_rate", "gradient_norm_before_clip")), f"CONTEXT fold{fold} nonfinite trace")
        require(c_result.get("checkpoint_sha256") == c_validation.get("checkpoint_sha256") == c_specs["checkpoint"]["sha256"] and c_validation.get("train_result_sha256") == c_specs["train_result"]["sha256"], f"CONTEXT fold{fold} result/checkpoint binding drift")
        require(c_result.get("run_identity_sha256") == c_validation.get("run_identity_sha256") == c_specs["run_identity"]["sha256"] and c_result.get("signed_reference_access_manifest_sha256") == c_validation.get("signed_reference_access_manifest_sha256") == c_specs["access_manifest"]["sha256"], f"CONTEXT fold{fold} identity/access binding drift")
        for value in (c_result, c_validation, c_identity): require(value.get("protocol_sha256") == c_specs["protocol"]["sha256"] and value.get("authority_sha256") == source_authority_spec["sha256"], f"CONTEXT fold{fold} protocol/authority binding drift")
        require(c_result.get("bag_permutation") == CONTEXT_MODE, f"CONTEXT fold{fold} input mode drift")
        chain = c_result.get("execution_chain", {}); chunks = chain.get("chunks")
        require(chain.get("chunk_count") == 1 and chain.get("resume_count") == 0 and chain.get("chunks_contiguous") is True and chain.get("maximum_jobs_respected") is True and isinstance(chunks, list) and len(chunks) == 1, f"CONTEXT fold{fold} execution chain drift")
        chunk_match = CHUNK_JOB.fullmatch(str(chunks[0].get("path"))); require(chunk_match and int(chunk_match.group(1)) == SCHEDULER_COMPLETION["tasks"][fold - 1]["job_id_raw"], f"CONTEXT fold{fold} JobIDRaw drift")
        c_sets = access_sets(c_access, f"CONTEXT fold{fold} access")
        require(len(c_sets["eligible"]) == c_access.get("eligible_query_count") == c_validation.get("eligible_query_count") and len(c_sets["allowed"]) == c_access.get("allowed_reference_count") == c_validation.get("allowed_reference_count"), f"CONTEXT fold{fold} access counts drift")
        require(len(c_sets["heldout"]) == c_access.get("heldout_reference_count") and not (c_sets["allowed"] & c_sets["heldout"]) and c_access.get("reference_row_intersection_count") == c_validation.get("heldout_reference_intersection_count") == 0, f"CONTEXT fold{fold} heldout access drift")
        runtime = c_result.get("runtime_access", {}); require(integer_set(runtime.get("query_read_union"), f"CONTEXT fold{fold} runtime queries") == c_sets["eligible"] and integer_set(runtime.get("reference_read_union"), f"CONTEXT fold{fold} runtime refs") == c_sets["allowed"] and runtime.get("reference_read_union_equals_signed_allowlist") is True, f"CONTEXT fold{fold} runtime access drift")
        c_common = {}
        for name in COMMON_BINDINGS:
            c_common[name], _ = exact_source(c_protocol.get("bindings", {}).get(name), f"CONTEXT fold{fold}.common.{name}")
            require(c_common[name] == common_expected[name] and c_validation.get("binding_hashes", {}).get(name) == c_common[name]["sha256"], f"fold{fold} common binding mismatch: {name}")

        b_specs, b_paths = {}, {}
        for name in ("authority", "protocol", "run_identity", "train_result", "training_validation", "checkpoint"):
            b_specs[name], b_paths[name] = exact_source(bag_entry.get(name), f"BAG fold{fold}.{name}")
        b_specs["access_manifest"], b_paths["access_manifest"] = exact_source(bag_entry.get("access_manifest"), f"BAG fold{fold}.access", logical=True)
        b_protocol, b_result, b_access = read_json(b_paths["protocol"], f"BAG fold{fold} protocol"), read_json(b_paths["train_result"], f"BAG fold{fold} result"), read_json(b_paths["access_manifest"], f"BAG fold{fold} access")
        require(b_result.get("arm") == "RCDE_BAG" and b_result.get("outer_fold") == fold and b_result.get("parameter_count") == PARAMETERS and b_result.get("protected_access_counts") == PROTECTED_ZERO and b_result.get("bag_permutation") == BAG_MODE, f"BAG fold{fold} result drift"); no_science(b_result, f"BAG fold{fold} result")
        b_training = b_result.get("training"); require(isinstance(b_training, dict) and b_training.get("update_count") == UPDATES and b_training.get("pair_count") == PAIRS and b_training.get("seed") == SEED, f"BAG fold{fold} training drift")
        require(normalized_training(b_protocol, "RCDE_BAG", fold, f"BAG fold{fold}") == normalized_training(c_protocol, "RCDE_CONTEXT", fold, f"CONTEXT fold{fold}"), f"fold{fold} normalized training mismatch")
        require(b_protocol.get("cache") == c_protocol.get("cache") and b_protocol.get("forbidden") == c_protocol.get("forbidden"), f"fold{fold} cache/forbidden mismatch")
        require({k: v for k, v in b_training.items() if k != "loss_trace"} == {k: v for k, v in c_training.items() if k != "loss_trace"}, f"fold{fold} schedule/input receipt mismatch")
        require(bag_entry.get("schedule_sha256") == c_training.get("schedule_sha256") and bag_entry.get("initial_state_sha256") == c_result.get("initial_state_sha256") == INITIAL_STATE_SHA256, f"fold{fold} schedule/initial mismatch")
        b_sets = access_sets(b_access, f"BAG fold{fold} access"); require(b_sets == c_sets, f"fold{fold} eligible/allowed/heldout mismatch")
        for name in COMMON_BINDINGS:
            b_common, _ = exact_source(b_protocol.get("bindings", {}).get(name), f"BAG fold{fold}.common.{name}"); require(b_common == c_common[name], f"fold{fold} common binding mismatch: {name}")
        b_mode = dict(BAG_MODE); c_mode = dict(CONTEXT_MODE); b_mode.pop("enabled"); c_mode.pop("enabled"); require(b_mode == c_mode, f"fold{fold} non-toggle input mode mismatch")

        schema = checkpoint_audit(b_paths["checkpoint"], arm="RCDE_BAG", fold=fold, protocol_sha=b_specs["protocol"]["sha256"], authority_sha=b_specs["authority"]["sha256"], result=b_result, expected_schema=schema)
        checkpoint_audit(c_paths["checkpoint"], arm="RCDE_CONTEXT", fold=fold, protocol_sha=c_specs["protocol"]["sha256"], authority_sha=source_authority_spec["sha256"], result=c_result, expected_schema=schema)
        expected_context = {"outer_fold": fold, **c_specs, "access_manifest": {**c_specs["access_manifest"], "logical_sha256": c_access["logical_sha256"]}, "schedule_sha256": c_training["schedule_sha256"], "initial_state_sha256": c_result["initial_state_sha256"], "final_state_sha256": c_result["final_state_sha256"], "update_count": UPDATES, "pair_count": PAIRS}
        expected_bag = {**b_specs, "schedule_sha256": bag_entry["schedule_sha256"], "initial_state_sha256": bag_entry["initial_state_sha256"], "final_state_sha256": bag_entry["final_state_sha256"]}
        expected_match = {"schedule_sha256": c_training["schedule_sha256"], "initial_state_sha256": INITIAL_STATE_SHA256, "eligible_query_count": len(c_sets["eligible"]), "allowed_reference_count": len(c_sets["allowed"]), "heldout_reference_count": len(c_sets["heldout"]), "eligible_query_set_sha256": canonical_sha256(sorted(c_sets["eligible"])), "allowed_reference_set_sha256": canonical_sha256(sorted(c_sets["allowed"])), "heldout_reference_set_sha256": canonical_sha256(sorted(c_sets["heldout"])), "common_bindings_equal": True, "normalized_training_cache_forbidden_equal": True, "training_receipt_except_loss_trace_equal": True, "only_model_input_space_difference": {"field": "bag_permutation.enabled", "RCDE_BAG": True, "RCDE_CONTEXT": False}, "allowed_non_model_differences": ["authority", "protocol", "artifact_path", "runtime_provenance", "trained_state_and_loss_trace"]}
        require(observed_by_fold[fold] == {"outer_fold": fold, "bag": expected_bag, "context": expected_context, "matched_contract": expected_match}, f"manifest fold{fold} differs from independent reconstruction")
        all_sets.append(c_sets)

    universes = [row["allowed"] | row["heldout"] for row in all_sets]; require(all(v == universes[0] for v in universes) and len(universes[0]) == 50, "reference universe closure drift")
    heldout = [row["heldout"] for row in all_sets]; membership = Counter(x for values in heldout for x in values)
    require([len(v) for v in heldout] == [13, 12, 12, 13] and set(membership) == universes[0] and set(membership.values()) == {1}, "heldout partition closure drift")
    eligible = [row["eligible"] for row in all_sets]; union = set().union(*eligible); membership_q = Counter(x for values in eligible for x in values)
    require(len(union) == 594 and set(membership_q.values()) == {3} and sorted(set(range(600)) - union) == [25, 26, 101, 346, 354, 470], "eligible 594x3 closure drift")
    require(manifest.get("cross_fold_closure", {}).get("optimization_reference_universe_count") == 50 and manifest.get("cross_fold_closure", {}).get("eligible_query_union_count") == 594, "manifest 50/594 closure drift")
    shared = manifest.get("shared_training_contract", {}); require(shared.get("seed") == SEED and shared.get("parameter_count") == PARAMETERS and shared.get("updates_per_arm_fold") == UPDATES and shared.get("pairs_per_arm_fold") == PAIRS and shared.get("initial_state_sha256") == INITIAL_STATE_SHA256 and shared.get("model_state_schema_sha256") == MODEL_STATE_SCHEMA_SHA256 and shared.get("common_bindings") == common_expected, "manifest shared contract drift")

    validation = {"schema_version": VALIDATION_SCHEMA, "status": VALIDATION_STATUS, "arms": ["RCDE_BAG", "RCDE_CONTEXT"], "outer_folds": list(FOLDS), "claim_level": CLAIM_LEVEL,
        "matched_foldset_protocol_sha256": file_sha256(protocol_path), "matched_foldset_manifest": {"path": str(manifest_path.relative_to(ROOT.resolve())), "sha256": file_sha256(manifest_path)},
        "checks": {"all_sources_paths_and_sha256_exact": True, "bag_foldset_manifest_and_validation_bound": True, "context_v1_8_four_folds_strict": True, "updates_2048_pairs_65536_parameters_65125_seed17": True, "source_validation_checks_all_true": True, "protected_access_all_zero": True, "scheduler_amendment_receipt_and_completed_task_mapping_bound": True, "validator_does_not_query_live_slurm": True, "bag_context_schedule_initial_access_and_common_bindings_equal_per_fold": True, "bag_enabled_context_disabled_only_model_input_difference": True, "cpu_checkpoint_metadata_state_hash_and_17_tensor_schema_closed_for_eight_checkpoints": True, "reference_universe_50_partition_closure": True, "eligible_query_union_594_membership_3_closure": True, "manifest_independently_reconstructed": True, "only_two_json_outputs_no_checkpoint_copy": True, "no_heldout_forward_label_join_scientific_metric_or_decision": True},
        "protected_access_counts": PROTECTED_ZERO, "checkpoint_files_loaded_cpu": 8, "checkpoint_files_copied": 0, "heldout_forward_count": 0, "heldout_label_join_count": 0, "scientific_metric_count": 0,
        "scientific_GO_or_NO_GO": None, "automatic_stage_advance": False, "next_required_stage": VALIDATION_NEXT}
    require(output_path.parent == manifest_path.parent and not output_path.exists(), "validation output path exists/drift")
    require({p.name for p in manifest_path.parent.iterdir()} == {"matched_foldset_manifest.json"}, "undeclared pre-validation output")
    data = (json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(); partial = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with partial.open("xb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.link(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)
    require({p.name for p in manifest_path.parent.iterdir()} == {"matched_foldset_manifest.json", "matched_foldset_validation.json"} and not output_path.is_symlink(), "validator output set drift")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate matched BAG/CONTEXT foldset")
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); output = validate(args.protocol, args.manifest, args.output)
    print(json.dumps({"validation": str(output), "status": VALIDATION_STATUS}, sort_keys=True))


if __name__ == "__main__":
    main()
