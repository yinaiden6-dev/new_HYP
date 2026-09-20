#!/usr/bin/env python3
"""Independent validator for the four-fold RCDE_BAG engineering foldset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import uuid


# This file intentionally does not import the foldset materializer.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROTOCOL_SCHEMA = "rc_dino_rcde_r1_bag_foldset_protocol_v1_20260814"
PROTOCOL_STAGE = "R1_MAIN_BAG_FOLDS1_4_FOLDSET_CLOSURE"
MANIFEST_SCHEMA = "rc_dino_rcde_r1_main_bag_foldset_manifest_v1_0"
MANIFEST_STATUS = "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_FROZEN"
VALIDATION_SCHEMA = "rc_dino_rcde_r1_main_bag_foldset_validation_v1_0"
VALIDATION_STATUS = "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_VALIDATION_PASS"
CLAIM_LEVEL = "engineering_foldset_closure_only_not_scientific_GO_or_NO_GO"
MANIFEST_NEXT = "R1_MAIN_BAG_FOLDS1_4_FOLDSET_VALIDATION"
VALIDATION_NEXT = "R1_MAIN_BAG_FOLDS1_4_VALIDATED_AUTHORITY_UPDATE_REQUIRED"

ARM = "RCDE_BAG"
FOLDS = (1, 2, 3, 4)
SEED = 17
PARAMETERS = 65_125
UPDATES = 2_048
PAIRS = 65_536
INITIAL_STATE_SHA256 = "4a51b45b1a29aa80be6acb18278d783d5485fbff01665f7054b518e2d870099b"
TRAINING_SHA256 = "e0d0d8d8cb2de523559af4801e76954457c072336df1b157c65598cf6a16c5e2"
CACHE_SHA256 = "ee2559f765fd0ab1f3650d16e34477f5481e670c64413ab2977330822dabeb4a"
FORBIDDEN_SHA256 = "417fe570b2d58ff9eeaf8c4e80609950001dc0405fbcb672c65fdb847359608a"
BAG_SHA256 = "1a2142a2cb681bdbb45b29db62ab2d78f0ec38e4a9d77b573f9996ce2bca06b4"
MODEL_STATE_SCHEMA_SHA256 = "2a1110ccebf0706986fe2d5377c4abadd0fc0687560ae2738bd850ea4f6af2e7"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PROTECTED_ZERO = {
    "C8_runtime_read_count": 0,
    "S8_runtime_read_count": 0,
    "home_files_modified": 0,
    "opened_runtime_read_count": 0,
    "sealed_runtime_read_count": 0,
    "unauthorized_natural_result_read_count": 0,
}
COMMON_BINDINGS = (
    "scientific_contract", "cache_protocol", "cache_result", "cache_validation",
    "training_roles", "episode_ledger", "resource_shape_ledger",
    "model_visible_c128", "rcde_core",
)
SOURCES = (
    "authority", "protocol", "run_identity", "access_manifest", "train_result",
    "training_validation", "checkpoint",
)
FOLD_INPUT_KEYS = {"outer_fold", *SOURCES}
RESULT_SCHEMAS = {
    "rc_dino_rcde_r1_main_train_result_v1_7",
    "rc_dino_rcde_r1_main_train_result_v1_8",
}
RESULT_STATUSES = {
    "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAIN_COMPLETE",
    "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAINING_COMPLETE",
}
SOURCE_VALIDATION_SCHEMAS = {
    "rc_dino_rcde_r1_main_train_validation_v1_7_contract_repair_20260813",
    "rc_dino_rcde_r1_main_training_validation_v1_8",
}
SOURCE_VALIDATION_STATUS = "DINO_RCDE_R1_MAIN_ARM_FOLD_VALIDATION_PASS"
IDENTITY_SCHEMAS = {
    "rc_dino_rcde_r1_main_run_identity_v1_7",
    "rc_dino_rcde_r1_run_identity_v1_8",
}
ACCESS_SCHEMAS = {
    "rc_dino_rcde_signed_reference_access_manifest_v1_7",
    "rc_dino_rcde_signed_reference_access_manifest_v1_8",
}
ACCESS_STATUS = "DINO_RCDE_SIGNED_REFERENCE_ACCESS_FROZEN"
CHECKPOINT_V17_FIELDS = {
    "schema_version", "arm", "outer_fold", "update", "seed", "protocol_sha256",
    "authority_sha256", "initial_state_sha256", "final_state_sha256",
    "model_state_dict",
}
CHECKPOINT_V18_FIELDS = CHECKPOINT_V17_FIELDS | {"status"}


class FoldsetValidationAbort(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FoldsetValidationAbort(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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
        raise FoldsetValidationAbort(f"invalid {context}: {path}") from exc
    require(isinstance(value, dict), f"{context} must be an object")
    return value


def resolve_relative(text: str) -> Path:
    raw = Path(text)
    require(text != "" and not raw.is_absolute(), "source paths must be RC-relative")
    path = (ROOT / raw).resolve()
    require(path.is_relative_to(ROOT.resolve()), "source path escaped RC root")
    require(path.is_file() and not path.is_symlink(), f"source is not regular: {text}")
    return path


def exact_source(value: Any, context: str) -> tuple[dict[str, str], Path]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{context} binding schema drift")
    text, digest = value.get("path"), value.get("sha256")
    require(isinstance(text, str), f"{context} path missing")
    require(isinstance(digest, str) and HEX64.fullmatch(digest) is not None, f"{context} digest malformed")
    path = resolve_relative(text)
    require(file_sha256(path) == digest, f"{context} digest drift")
    return {"path": text, "sha256": digest}, path


def no_science(value: Mapping[str, Any], context: str) -> None:
    require(value.get("scientific_GO_or_NO_GO") is None, f"{context} scientific decision present")
    require(value.get("automatic_stage_advance") is False, f"{context} automatic advance present")


def state_dict_sha256(state: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), f"state value is not tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(canonical_bytes(list(tensor.shape)) + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_audit(
    path: Path,
    *,
    fold: int,
    protocol_sha: str,
    authority_sha: str,
    result: Mapping[str, Any],
    expected_schema: list[list[Any]] | None,
) -> list[list[Any]]:
    try:
        import torch
        from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2
    except Exception as exc:  # pragma: no cover - environment failure
        raise FoldsetValidationAbort("torch/RCDE core unavailable for checkpoint validation") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), f"fold{fold} checkpoint is not an object")
    schema_version = checkpoint.get("schema_version")
    if fold == 1 and schema_version == "rc_dino_rcde_r1_main_checkpoint_v1_7":
        require(set(checkpoint) == CHECKPOINT_V17_FIELDS, "V1.7 checkpoint field drift")
        require("status" not in checkpoint, "V1.7 checkpoint must not gain status")
    elif fold in (2, 3, 4) and schema_version == "rc_dino_rcde_r1_main_checkpoint_v1_8":
        require(set(checkpoint) == CHECKPOINT_V18_FIELDS, "V1.8 checkpoint field drift")
        require(checkpoint.get("status") == "DINO_RCDE_R1_MAIN_CHECKPOINT_COMPLETE", "V1.8 checkpoint status drift")
    else:
        raise FoldsetValidationAbort(f"fold{fold} checkpoint schema unsupported")
    require(checkpoint.get("arm") == ARM and checkpoint.get("outer_fold") == fold, f"fold{fold} checkpoint scope drift")
    require(checkpoint.get("update") == UPDATES and checkpoint.get("seed") == SEED, f"fold{fold} checkpoint update/seed drift")
    require(checkpoint.get("protocol_sha256") == protocol_sha, f"fold{fold} checkpoint protocol drift")
    require(checkpoint.get("authority_sha256") == authority_sha, f"fold{fold} checkpoint authority drift")
    state = checkpoint.get("model_state_dict")
    require(isinstance(state, dict) and len(state) == 17, f"fold{fold} checkpoint state cardinality drift")
    schema = [[name, list(value.shape), str(value.dtype)] for name, value in state.items()]
    require(canonical_sha256(schema) == MODEL_STATE_SCHEMA_SHA256, f"fold{fold} model state schema hash drift")
    if expected_schema is not None:
        require(schema == expected_schema, f"fold{fold} ordered model state schema differs")
    require(sum(int(value.numel()) for value in state.values()) == PARAMETERS, f"fold{fold} checkpoint parameter count drift")
    torch.manual_seed(SEED)
    initial_state = DINO_RCDE_V1_2().state_dict()
    initial_sha = state_dict_sha256(initial_state, torch)
    final_sha = state_dict_sha256(state, torch)
    require(initial_sha == INITIAL_STATE_SHA256, "independent initial state replay drift")
    require(checkpoint.get("initial_state_sha256") == result.get("initial_state_sha256") == initial_sha, f"fold{fold} initial state binding drift")
    require(checkpoint.get("final_state_sha256") == result.get("final_state_sha256") == final_sha, f"fold{fold} final state binding drift")
    require(final_sha != initial_sha, f"fold{fold} checkpoint equals initialization")
    return schema


def authority_scope(authority: Mapping[str, Any], protocol: Mapping[str, Any], fold: int) -> bool:
    if isinstance(authority.get("authorized_scopes"), list):
        scopes = authority["authorized_scopes"]
    elif isinstance(authority.get("authorized_scope"), dict):
        scopes = [authority["authorized_scope"]]
    else:
        return False
    matches = [
        row for row in scopes if isinstance(row, dict)
        and row.get("stage") == protocol.get("stage")
        and row.get("arm") == ARM and row.get("outer_fold") == fold
    ]
    if len(matches) != 1:
        return False
    row = matches[0]
    return row.get("seed") == SEED and row.get(
        "optimizer_updates_total", row.get("optimizer_updates_per_arm_fold")
    ) == UPDATES


def audit_fold(entry: Any, state_schema: list[list[Any]] | None) -> tuple[dict[str, Any], dict[str, set[int]], list[list[Any]]]:
    require(isinstance(entry, dict) and set(entry) == FOLD_INPUT_KEYS, "fold input schema drift")
    fold = entry.get("outer_fold")
    require(type(fold) is int and fold in FOLDS, "fold input is not one of 1..4")
    specs: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for name in SOURCES:
        specs[name], paths[name] = exact_source(entry[name], f"fold{fold}.{name}")
    result = read_json(paths["train_result"], f"fold{fold} train result")
    source_validation = read_json(paths["training_validation"], f"fold{fold} source validation")
    identity = read_json(paths["run_identity"], f"fold{fold} run identity")
    access = read_json(paths["access_manifest"], f"fold{fold} access manifest")
    source_protocol = read_json(paths["protocol"], f"fold{fold} source protocol")
    source_authority = read_json(paths["authority"], f"fold{fold} source authority")

    expected_result_schema = "rc_dino_rcde_r1_main_train_result_v1_7" if fold == 1 else "rc_dino_rcde_r1_main_train_result_v1_8"
    expected_result_status = "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAIN_COMPLETE" if fold == 1 else "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAINING_COMPLETE"
    expected_validation_schema = "rc_dino_rcde_r1_main_train_validation_v1_7_contract_repair_20260813" if fold == 1 else "rc_dino_rcde_r1_main_training_validation_v1_8"
    expected_identity_schema = "rc_dino_rcde_r1_main_run_identity_v1_7" if fold == 1 else "rc_dino_rcde_r1_run_identity_v1_8"
    expected_access_schema = "rc_dino_rcde_signed_reference_access_manifest_v1_7" if fold == 1 else "rc_dino_rcde_signed_reference_access_manifest_v1_8"
    expected_protocol_schema = "rc_dino_rcde_r1_main_train_protocol_v1_7_contract_repair_20260813" if fold == 1 else "rc_dino_rcde_r1_main_train_protocol_v1_8_single_scope_20260814"
    require(result.get("schema_version") == expected_result_schema and result.get("status") == expected_result_status, f"fold{fold} train result version/status drift")
    require(source_validation.get("schema_version") == expected_validation_schema and source_validation.get("status") == SOURCE_VALIDATION_STATUS, f"fold{fold} source validation version/status drift")
    require(identity.get("schema_version") == expected_identity_schema, f"fold{fold} identity version drift")
    require(access.get("schema_version") == expected_access_schema and access.get("status") == ACCESS_STATUS, f"fold{fold} access version drift")
    require(source_protocol.get("schema_version") == expected_protocol_schema, f"fold{fold} source protocol version drift")
    if fold != 1:
        require(identity.get("status") == "DINO_RCDE_R1_RUN_IDENTITY_FROZEN", f"fold{fold} V1.8 identity status drift")
    else:
        require("status" not in identity, "V1.7 run identity status reinterpretation")
    for payload, label in ((result, "result"), (source_validation, "validation")):
        require(payload.get("arm") == ARM and payload.get("outer_fold") == fold, f"fold{fold} {label} scope drift")
        require(payload.get("protected_access_counts") == PROTECTED_ZERO, f"fold{fold} {label} protected access drift")
        no_science(payload, f"fold{fold} {label}")
    require(identity.get("arm") == ARM and identity.get("outer_fold") == fold, f"fold{fold} identity scope drift")
    require(access.get("outer_fold") == fold and access.get("protected_access_counts") == PROTECTED_ZERO, f"fold{fold} access scope/protected drift")
    require(access.get("reference_row_intersection_count") == 0, f"fold{fold} reference access leak")
    require(result.get("runtime_access", {}).get("heldout_reference_intersection_count") == 0, f"fold{fold} result reference leak")
    require(source_validation.get("heldout_reference_intersection_count") == 0, f"fold{fold} validation reference leak")

    result_training = result.get("training")
    require(isinstance(result_training, dict), f"fold{fold} training receipt missing")
    require(result.get("parameter_count") == PARAMETERS, f"fold{fold} result parameter count drift")
    require(result_training.get("update_count") == source_validation.get("update_count") == UPDATES, f"fold{fold} update count drift")
    require(result_training.get("pair_count") == source_validation.get("pair_count") == PAIRS, f"fold{fold} pair count drift")
    require(result_training.get("seed") == SEED, f"fold{fold} result seed drift")
    require(source_validation.get("checks") and all(flag is True for flag in source_validation["checks"].values()), f"fold{fold} source validation checks failed")

    training = source_protocol.get("training")
    cache = source_protocol.get("cache")
    forbidden = source_protocol.get("forbidden")
    require(isinstance(training, dict) and training.get("arms") == [ARM] and training.get("outer_folds") == [fold], f"fold{fold} source protocol scope drift")
    training_projection = dict(training); training_projection.pop("outer_folds", None)
    require(canonical_sha256(training_projection) == TRAINING_SHA256, f"fold{fold} normalized training drift")
    require(isinstance(cache, dict) and canonical_sha256(cache) == CACHE_SHA256, f"fold{fold} cache contract drift")
    require(isinstance(forbidden, dict) and canonical_sha256(forbidden) == FORBIDDEN_SHA256, f"fold{fold} forbidden contract drift")
    no_science(source_protocol, f"fold{fold} source protocol")

    authority_anchor = source_protocol.get("authority")
    require(isinstance(authority_anchor, dict), f"fold{fold} authority anchor missing")
    require(authority_anchor.get("path") == specs["authority"]["path"] and authority_anchor.get("sha256") == specs["authority"]["sha256"], f"fold{fold} authority source chain drift")
    require(source_authority.get("status") == authority_anchor.get("required_status"), f"fold{fold} source authority status drift")
    require(authority_scope(source_authority, source_protocol, fold), f"fold{fold} exact scope unauthorized")
    no_science(source_authority, f"fold{fold} source authority")
    require(source_authority.get("protected_access_counts") == PROTECTED_ZERO, f"fold{fold} source authority protected drift")
    for payload in (result, source_validation, identity):
        require(payload.get("protocol_sha256") == specs["protocol"]["sha256"], f"fold{fold} protocol chain drift")
        require(payload.get("authority_sha256") == specs["authority"]["sha256"], f"fold{fold} authority chain drift")
    require(source_validation.get("train_result_sha256") == specs["train_result"]["sha256"], f"fold{fold} result validation binding drift")
    require(result.get("run_identity_sha256") == source_validation.get("run_identity_sha256") == specs["run_identity"]["sha256"], f"fold{fold} identity binding drift")
    require(result.get("signed_reference_access_manifest_sha256") == source_validation.get("signed_reference_access_manifest_sha256") == specs["access_manifest"]["sha256"], f"fold{fold} access binding drift")
    require(result.get("checkpoint_sha256") == source_validation.get("checkpoint_sha256") == specs["checkpoint"]["sha256"], f"fold{fold} checkpoint binding drift")

    result_dir = paths["train_result"].parent.resolve()
    require(all(paths[name].parent.resolve() == result_dir for name in ("training_validation", "run_identity", "access_manifest", "checkpoint")), f"fold{fold} result artifacts not co-located")
    require((result_dir / str(result.get("checkpoint"))).resolve() == paths["checkpoint"], f"fold{fold} checkpoint relative path drift")
    require((ROOT / str(source_protocol.get("resume", {}).get("persistent_output"))).resolve() == result_dir, f"fold{fold} persistent output drift")
    require(result.get("initial_state_sha256") == INITIAL_STATE_SHA256, f"fold{fold} result initial state drift")
    require(result_training.get("schedule_sha256") == source_validation.get("schedule_sha256"), f"fold{fold} schedule chain drift")
    require(canonical_sha256(result.get("bag_permutation")) == BAG_SHA256, f"fold{fold} BAG permutation drift")

    checkpoint_schema = checkpoint_audit(
        paths["checkpoint"], fold=fold, protocol_sha=specs["protocol"]["sha256"],
        authority_sha=specs["authority"]["sha256"], result=result,
        expected_schema=state_schema,
    )

    common: dict[str, dict[str, str]] = {}
    bindings = source_protocol.get("bindings")
    require(isinstance(bindings, dict), f"fold{fold} source bindings missing")
    for name in COMMON_BINDINGS:
        spec, _ = exact_source(bindings.get(name), f"fold{fold}.common.{name}")
        common[name] = spec
        require(source_validation.get("binding_hashes", {}).get(name) == spec["sha256"], f"fold{fold} common binding validation drift: {name}")

    detached_access = dict(access)
    access_logical = detached_access.pop("logical_sha256", None)
    require(isinstance(access_logical, str) and canonical_sha256(detached_access) == access_logical, f"fold{fold} access logical hash drift")
    eligible = access.get("eligible_query_execution_ordinals")
    allowed = access.get("allowed_reference_physical_rows")
    heldout = access.get("heldout_reference_physical_rows")
    require(all(isinstance(value, list) and all(type(item) is int for item in value) for value in (eligible, allowed, heldout)), f"fold{fold} access sets malformed")
    require(len(eligible) == access.get("eligible_query_count") == source_validation.get("eligible_query_count"), f"fold{fold} eligible count drift")
    require(len(allowed) == access.get("allowed_reference_count") == source_validation.get("allowed_reference_count"), f"fold{fold} allowed count drift")
    require(len(heldout) == access.get("heldout_reference_count"), f"fold{fold} heldout count drift")
    require(not (set(allowed) & set(heldout)), f"fold{fold} allowed/heldout overlap")
    record = {
        "outer_fold": fold,
        "authority": specs["authority"], "protocol": specs["protocol"],
        "run_identity": specs["run_identity"],
        "access_manifest": {**specs["access_manifest"], "logical_sha256": access_logical},
        "train_result": specs["train_result"],
        "training_validation": specs["training_validation"],
        "checkpoint": specs["checkpoint"],
        "schedule_sha256": result_training["schedule_sha256"],
        "initial_state_sha256": result["initial_state_sha256"],
        "final_state_sha256": result["final_state_sha256"],
        "update_count": UPDATES, "pair_count": PAIRS,
        "eligible_query_count": len(eligible),
        "allowed_reference_count": len(allowed),
        "heldout_reference_count": len(heldout),
    }
    record["_common"] = common
    return record, {"eligible": set(eligible), "allowed": set(allowed), "heldout": set(heldout)}, checkpoint_schema


def reconstruct(protocol_path: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    require(protocol_path.is_file() and not protocol_path.is_symlink(), "foldset protocol missing")
    protocol = read_json(protocol_path, "foldset protocol")
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA and protocol.get("stage") == PROTOCOL_STAGE, "foldset protocol schema/stage drift")
    require(protocol.get("claim_level") == CLAIM_LEVEL, "foldset protocol claim drift")
    no_science(protocol, "foldset protocol")

    authority_raw = protocol.get("authority")
    require(isinstance(authority_raw, dict) and set(authority_raw) == {"path", "sha256", "required_status"}, "foldset authority binding schema drift")
    authority_spec, authority_path = exact_source({"path": authority_raw["path"], "sha256": authority_raw["sha256"]}, "foldset.authority")
    authority_spec["required_status"] = str(authority_raw["required_status"])
    authority = read_json(authority_path, "foldset authority")
    require(authority.get("status") == authority_raw["required_status"], "foldset authority status drift")
    require(authority.get("next_authorized_stage") == PROTOCOL_STAGE, "foldset authority stage drift")
    require(authority.get("natural_training_authorized") is False, "foldset authority must not authorize training")
    require(authority.get("protected_access_counts") == PROTECTED_ZERO, "foldset authority protected drift")
    no_science(authority, "foldset authority")
    implementations = protocol.get("implementation_bindings")
    require(isinstance(implementations, dict) and set(implementations) == {"materializer", "validator", "tests"}, "implementation binding set drift")
    for name in ("materializer", "validator", "tests"):
        exact_source(implementations[name], f"foldset.{name}")
    output_dir_text = protocol.get("output_dir")
    require(isinstance(output_dir_text, str) and output_dir_text != "", "foldset output_dir missing")
    required_output_dir = (ROOT / output_dir_text).resolve()
    require(not Path(output_dir_text).is_absolute() and required_output_dir.is_relative_to(ROOT.resolve()), "foldset output_dir escapes RC root")

    inputs = protocol.get("fold_inputs")
    require(isinstance(inputs, list) and len(inputs) == 4, "foldset protocol does not list four folds")
    audited = []
    state_schema = None
    for entry in sorted(inputs, key=lambda row: row.get("outer_fold", -1) if isinstance(row, dict) else -1):
        record, sets, observed_schema = audit_fold(entry, state_schema)
        if state_schema is None:
            state_schema = observed_schema
        audited.append((record, sets))
    require(tuple(row[0]["outer_fold"] for row in audited) == FOLDS, "folds are not exact 1..4")
    records = [row[0] for row in audited]
    sets = [row[1] for row in audited]
    common = records[0].pop("_common")
    for record in records[1:]:
        require(record.pop("_common") == common, "common bindings differ across folds")

    universe_sets = [row["allowed"] | row["heldout"] for row in sets]
    require(all(value == universe_sets[0] for value in universe_sets), "reference universe differs")
    universe = universe_sets[0]
    heldout = [row["heldout"] for row in sets]
    require(len(universe) == 50 and [len(value) for value in heldout] == [13, 12, 12, 13], "reference universe/count drift")
    require(sum(len(heldout[i] & heldout[j]) for i in range(4) for j in range(i + 1, 4)) == 0, "heldout pairwise overlap")
    membership = Counter(item for values in heldout for item in values)
    require(set(membership) == universe and set(membership.values()) == {1}, "heldout reference partition incomplete")
    require(all(row["allowed"] == universe - row["heldout"] for row in sets), "allowed is not universe minus heldout")
    eligible = [row["eligible"] for row in sets]
    eligible_union = set().union(*eligible)
    eligible_membership = Counter(item for values in eligible for item in values)
    require(len(eligible_union) == 594 and set(eligible_membership.values()) == {3}, "eligible query 594x3 closure drift")
    globally_ineligible = sorted(set(range(600)) - eligible_union)
    require(globally_ineligible == [25, 26, 101, 346, 354, 470], "global ineligible ordinals drift")

    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": MANIFEST_STATUS,
        "arm": ARM,
        "outer_folds": list(FOLDS),
        "source_mode": "immutable_paths_and_sha256_no_copy",
        "authority": authority_spec,
        "foldset_protocol": {
            "path": str(protocol_path.relative_to(ROOT.resolve())),
            "sha256": file_sha256(protocol_path),
        },
        "claim_level": CLAIM_LEVEL,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "protected_access_counts": PROTECTED_ZERO,
        "shared_training_contract": {
            "seed": SEED, "parameter_count": PARAMETERS,
            "initial_state_sha256": INITIAL_STATE_SHA256,
            "normalized_training_semantics_sha256": TRAINING_SHA256,
            "normalized_cache_contract_sha256": CACHE_SHA256,
            "normalized_forbidden_contract_sha256": FORBIDDEN_SHA256,
            "bag_permutation_sha256": BAG_SHA256,
            "model_state_schema_sha256": MODEL_STATE_SCHEMA_SHA256,
            "common_bindings": common,
        },
        "folds": records,
        "cross_fold_closure": {
            "optimization_reference_universe_count": 50,
            "heldout_reference_counts_by_fold": {"1": 13, "2": 12, "3": 12, "4": 13},
            "heldout_pairwise_intersection_count": 0,
            "heldout_union_count": 50,
            "heldout_membership_count_per_reference": 1,
            "allowed_is_universe_minus_heldout": True,
            "eligible_query_union_count": 594,
            "eligible_query_outer_train_membership_count": 3,
            "globally_ineligible_execution_ordinals": globally_ineligible,
            "scope": {
                "query_population": "600_query_training_role",
                "reference_population": "50_row_optimization_reference_universe",
                "full_gallery_evaluation": False,
                "outer_heldout_forward": False,
                "heldout_label_join": False,
                "scientific_reduction": False,
            },
        },
        "next_required_stage": MANIFEST_NEXT,
    }


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    require(not path.exists(), "validation output already exists")
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with partial.open("xb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def validate(protocol_path: Path, manifest_path: Path, output_path: Path) -> Path:
    expected = reconstruct(protocol_path)
    manifest_path = manifest_path.resolve()
    protocol = read_json(protocol_path.resolve(), "foldset protocol")
    required_dir = (ROOT / str(protocol.get("output_dir"))).resolve()
    require(manifest_path.parent == required_dir and required_dir.is_relative_to(ROOT.resolve()), "manifest directory differs from protocol")
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "foldset manifest missing/symlinked")
    observed = read_json(manifest_path, "foldset manifest")
    require(observed == expected, "foldset manifest differs from independent reconstruction")
    require(output_path.name == "foldset_validation.json", "validation output filename drift")
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "arm": ARM,
        "outer_folds": list(FOLDS),
        "claim_level": CLAIM_LEVEL,
        "foldset_protocol_sha256": file_sha256(protocol_path.resolve()),
        "foldset_manifest": {
            "path": str(manifest_path.relative_to(ROOT.resolve())),
            "sha256": file_sha256(manifest_path),
        },
        "checks": {
            "source_paths_and_sha256_exact": True,
            "folds_exactly_1_2_3_4": True,
            "version_adaptation_v1_7_v1_8_exact": True,
            "source_training_and_validations_complete": True,
            "checkpoint_metadata_and_state_hashes_closed": True,
            "ordered_17_tensor_schema_identical": True,
            "parameter_count_65125": True,
            "shared_initial_state_core_cache_training_and_forbidden_contracts": True,
            "bag_permutation_exact": True,
            "reference_universe_50_partition_closure": True,
            "eligible_query_union_594_membership_3_closure": True,
            "no_checkpoint_copy": True,
            "no_heldout_forward_label_join_or_scientific_reducer": True,
            "manifest_independently_reconstructed": True,
        },
        "protected_access_counts": PROTECTED_ZERO,
        "checkpoint_files_copied": 0,
        "heldout_forward_count": 0,
        "heldout_label_join_count": 0,
        "scientific_metric_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_required_stage": VALIDATION_NEXT,
    }
    output_path = output_path.resolve()
    require(output_path.parent == manifest_path.parent, "validation must be co-located with manifest")
    require(not manifest_path.parent.is_symlink() and not manifest_path.is_symlink(), "foldset paths may not be symlinked")
    before = {p.name for p in manifest_path.parent.iterdir()}
    require(before == {"foldset_manifest.json"}, "undeclared pre-validation output file")
    write_once(output_path, validation)
    require(not output_path.is_symlink(), "validation output may not be symlinked")
    after = {p.name for p in manifest_path.parent.iterdir()}
    require(after == {"foldset_manifest.json", "foldset_validation.json"}, "validator output contains undeclared files")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate the RCDE_BAG foldset")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = validate(args.protocol, args.manifest, args.output)
    print(json.dumps({"validation": str(output), "status": VALIDATION_STATUS}, sort_keys=True))


if __name__ == "__main__":
    main()
