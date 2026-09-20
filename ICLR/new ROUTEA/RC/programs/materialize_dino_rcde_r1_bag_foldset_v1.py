#!/usr/bin/env python3
"""Freeze the four completed RCDE_BAG training folds as one engineering set.

This program never loads labels, runs a model, evaluates heldout queries, or
copies a checkpoint.  A future protocol must bind every source by exact
RC-relative path and physical SHA-256.  The single output is an immutable
engineering manifest; scientific reduction remains unauthorized.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "rc_dino_rcde_r1_bag_foldset_protocol_v1_20260814"
PROTOCOL_STAGE = "R1_MAIN_BAG_FOLDS1_4_FOLDSET_CLOSURE"
MANIFEST_SCHEMA = "rc_dino_rcde_r1_main_bag_foldset_manifest_v1_0"
MANIFEST_STATUS = "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_FROZEN"
CLAIM_LEVEL = "engineering_foldset_closure_only_not_scientific_GO_or_NO_GO"
MANIFEST_NEXT = "R1_MAIN_BAG_FOLDS1_4_FOLDSET_VALIDATION"

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

COMMON_BINDING_NAMES = (
    "scientific_contract", "cache_protocol", "cache_result", "cache_validation",
    "training_roles", "episode_ledger", "resource_shape_ledger",
    "model_visible_c128", "rcde_core",
)
FOLD_SOURCE_NAMES = (
    "authority", "protocol", "run_identity", "access_manifest", "train_result",
    "training_validation", "checkpoint",
)
FOLD_INPUT_KEYS = {"outer_fold", *FOLD_SOURCE_NAMES}
PROTECTED_ZERO = {
    "C8_runtime_read_count": 0,
    "S8_runtime_read_count": 0,
    "home_files_modified": 0,
    "opened_runtime_read_count": 0,
    "sealed_runtime_read_count": 0,
    "unauthorized_natural_result_read_count": 0,
}
RESULT_SCHEMAS = {
    "rc_dino_rcde_r1_main_train_result_v1_7",
    "rc_dino_rcde_r1_main_train_result_v1_8",
}
RESULT_STATUSES = {
    "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAIN_COMPLETE",
    "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAINING_COMPLETE",
}
VALIDATION_SCHEMAS = {
    "rc_dino_rcde_r1_main_train_validation_v1_7_contract_repair_20260813",
    "rc_dino_rcde_r1_main_training_validation_v1_8",
}
VALIDATION_STATUS = "DINO_RCDE_R1_MAIN_ARM_FOLD_VALIDATION_PASS"
IDENTITY_SCHEMAS = {
    "rc_dino_rcde_r1_main_run_identity_v1_7",
    "rc_dino_rcde_r1_run_identity_v1_8",
}
ACCESS_SCHEMAS = {
    "rc_dino_rcde_signed_reference_access_manifest_v1_7",
    "rc_dino_rcde_signed_reference_access_manifest_v1_8",
}
ACCESS_STATUS = "DINO_RCDE_SIGNED_REFERENCE_ACCESS_FROZEN"


class FoldsetAbort(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FoldsetAbort(message)


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
        raise FoldsetAbort(f"invalid {context}: {path}") from exc
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


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, f"immutable output drift: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with partial.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def no_science(value: Mapping[str, Any], context: str) -> None:
    require(value.get("scientific_GO_or_NO_GO") is None, f"{context} contains a scientific decision")
    require(value.get("automatic_stage_advance") is False, f"{context} permits automatic advance")


def authority_has_scope(authority: Mapping[str, Any], protocol: Mapping[str, Any], fold: int) -> bool:
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
    updates = row.get("optimizer_updates_total", row.get("optimizer_updates_per_arm_fold"))
    return updates == UPDATES and row.get("seed") == SEED


def logical_access_sha(access: Mapping[str, Any]) -> str:
    detached = dict(access)
    observed = detached.pop("logical_sha256", None)
    require(isinstance(observed, str) and HEX64.fullmatch(observed) is not None, "access logical sha256 malformed")
    require(canonical_sha256(detached) == observed, "access logical sha256 drift")
    return observed


def normalize_protocol(protocol: Mapping[str, Any], fold: int) -> tuple[dict[str, Any], dict[str, Any]]:
    training = protocol.get("training")
    cache = protocol.get("cache")
    forbidden = protocol.get("forbidden")
    require(isinstance(training, dict) and training.get("arms") == [ARM], f"fold{fold} protocol arm drift")
    require(training.get("outer_folds") == [fold], f"fold{fold} protocol fold drift")
    require(isinstance(cache, dict), f"fold{fold} cache contract missing")
    require(isinstance(forbidden, dict), f"fold{fold} forbidden contract missing")
    training_projection = dict(training)
    training_projection.pop("outer_folds", None)
    require(canonical_sha256(training_projection) == TRAINING_SHA256, f"fold{fold} normalized training contract drift")
    require(canonical_sha256(cache) == CACHE_SHA256, f"fold{fold} cache contract drift")
    require(canonical_sha256(forbidden) == FORBIDDEN_SHA256, f"fold{fold} forbidden contract drift")
    require(training.get("parameter_count") == PARAMETERS, f"fold{fold} protocol parameter count drift")
    require(training.get("updates_total") == UPDATES and training.get("seed") == SEED, f"fold{fold} protocol update/seed drift")
    require(training.get("outer_train_only") is True and training.get("heldout_target_join_allowed") is False, f"fold{fold} outer-train boundary drift")
    require(forbidden.get("home_write") is False and forbidden.get("heldout_label_join") is False, f"fold{fold} forbidden write/join drift")
    require(forbidden.get("evaluation") is False and forbidden.get("scientific_decision") is False, f"fold{fold} evaluation boundary drift")
    require(set(forbidden.get("paths", [])) == {"C8", "S8", "opened", "sealed"}, f"fold{fold} protected path set drift")
    no_science(protocol, f"fold{fold} training protocol")
    return training_projection, dict(cache)


def audit_fold(entry: Any) -> tuple[dict[str, Any], dict[str, set[int]]]:
    require(isinstance(entry, dict) and set(entry) == FOLD_INPUT_KEYS, "fold input keys drift")
    fold = entry.get("outer_fold")
    require(type(fold) is int and fold in FOLDS, "fold must be exactly one of 1..4")
    specs: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for name in FOLD_SOURCE_NAMES:
        specs[name], paths[name] = exact_source(entry[name], f"fold{fold}.{name}")

    result = read_json(paths["train_result"], f"fold{fold} train result")
    validation = read_json(paths["training_validation"], f"fold{fold} validation")
    identity = read_json(paths["run_identity"], f"fold{fold} run identity")
    access = read_json(paths["access_manifest"], f"fold{fold} access manifest")
    protocol = read_json(paths["protocol"], f"fold{fold} training protocol")
    authority = read_json(paths["authority"], f"fold{fold} authority")

    expected_result_schema = "rc_dino_rcde_r1_main_train_result_v1_7" if fold == 1 else "rc_dino_rcde_r1_main_train_result_v1_8"
    expected_result_status = "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAIN_COMPLETE" if fold == 1 else "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAINING_COMPLETE"
    expected_validation_schema = "rc_dino_rcde_r1_main_train_validation_v1_7_contract_repair_20260813" if fold == 1 else "rc_dino_rcde_r1_main_training_validation_v1_8"
    expected_identity_schema = "rc_dino_rcde_r1_main_run_identity_v1_7" if fold == 1 else "rc_dino_rcde_r1_run_identity_v1_8"
    expected_access_schema = "rc_dino_rcde_signed_reference_access_manifest_v1_7" if fold == 1 else "rc_dino_rcde_signed_reference_access_manifest_v1_8"
    expected_protocol_schema = "rc_dino_rcde_r1_main_train_protocol_v1_7_contract_repair_20260813" if fold == 1 else "rc_dino_rcde_r1_main_train_protocol_v1_8_single_scope_20260814"
    require(result.get("schema_version") == expected_result_schema and result.get("status") == expected_result_status, f"fold{fold} result version/status drift")
    require(validation.get("schema_version") == expected_validation_schema and validation.get("status") == VALIDATION_STATUS, f"fold{fold} validation version/status drift")
    require(identity.get("schema_version") == expected_identity_schema, f"fold{fold} run identity version drift")
    require(access.get("schema_version") == expected_access_schema and access.get("status") == ACCESS_STATUS, f"fold{fold} access manifest version drift")
    require(protocol.get("schema_version") == expected_protocol_schema, f"fold{fold} source protocol version drift")
    if fold != 1:
        require(identity.get("status") == "DINO_RCDE_R1_RUN_IDENTITY_FROZEN", f"fold{fold} V1.8 identity status drift")
    else:
        require("status" not in identity, "V1.7 run identity must not be reinterpreted with a status")

    for value, name in ((result, "result"), (validation, "validation")):
        require(value.get("arm") == ARM and value.get("outer_fold") == fold, f"fold{fold} {name} scope drift")
        require(value.get("protected_access_counts") == PROTECTED_ZERO, f"fold{fold} {name} protected reads")
        no_science(value, f"fold{fold} {name}")
    require(identity.get("arm") == ARM and identity.get("outer_fold") == fold, f"fold{fold} identity scope drift")
    require(access.get("outer_fold") == fold and access.get("protected_access_counts") == PROTECTED_ZERO, f"fold{fold} access scope/protected drift")
    require(access.get("reference_row_intersection_count") == 0, f"fold{fold} train/heldout reference intersection")
    require(result.get("runtime_access", {}).get("heldout_reference_intersection_count") == 0, f"fold{fold} result heldout intersection")
    require(validation.get("heldout_reference_intersection_count") == 0, f"fold{fold} validation heldout intersection")

    require(result.get("parameter_count") == PARAMETERS, f"fold{fold} result parameter count drift")
    training = result.get("training")
    require(isinstance(training, dict), f"fold{fold} result training missing")
    require(training.get("update_count") == validation.get("update_count") == UPDATES, f"fold{fold} update count drift")
    require(training.get("pair_count") == validation.get("pair_count") == PAIRS, f"fold{fold} pair count drift")
    require(training.get("seed") == SEED, f"fold{fold} result seed drift")
    require(validation.get("checks") and all(flag is True for flag in validation["checks"].values()), f"fold{fold} validation checks not all true")

    normalize_protocol(protocol, fold)
    authority_anchor = protocol.get("authority")
    require(isinstance(authority_anchor, dict), f"fold{fold} protocol authority binding missing")
    require(authority_anchor.get("path") == specs["authority"]["path"] and authority_anchor.get("sha256") == specs["authority"]["sha256"], f"fold{fold} authority chain drift")
    require(authority.get("status") == authority_anchor.get("required_status"), f"fold{fold} authority status drift")
    require(authority_has_scope(authority, protocol, fold), f"fold{fold} exact scope was not authorized")
    no_science(authority, f"fold{fold} authority")
    require(authority.get("protected_access_counts") == PROTECTED_ZERO, f"fold{fold} authority protected counts drift")

    for payload in (result, validation, identity):
        require(payload.get("protocol_sha256") == specs["protocol"]["sha256"], f"fold{fold} protocol binding drift")
        require(payload.get("authority_sha256") == specs["authority"]["sha256"], f"fold{fold} authority binding drift")
    require(validation.get("train_result_sha256") == specs["train_result"]["sha256"], f"fold{fold} validation/result binding drift")
    require(result.get("run_identity_sha256") == specs["run_identity"]["sha256"], f"fold{fold} result/identity binding drift")
    require(validation.get("run_identity_sha256") == specs["run_identity"]["sha256"], f"fold{fold} validation/identity binding drift")
    require(result.get("signed_reference_access_manifest_sha256") == specs["access_manifest"]["sha256"], f"fold{fold} result/access binding drift")
    require(validation.get("signed_reference_access_manifest_sha256") == specs["access_manifest"]["sha256"], f"fold{fold} validation/access binding drift")
    require(result.get("checkpoint_sha256") == specs["checkpoint"]["sha256"], f"fold{fold} result/checkpoint binding drift")
    require(validation.get("checkpoint_sha256") == specs["checkpoint"]["sha256"], f"fold{fold} validation/checkpoint binding drift")

    result_dir = paths["train_result"].parent.resolve()
    require(all(paths[name].parent.resolve() == result_dir for name in ("training_validation", "run_identity", "access_manifest", "checkpoint")), f"fold{fold} result artifacts are not co-located")
    require((result_dir / str(result.get("checkpoint"))).resolve() == paths["checkpoint"], f"fold{fold} checkpoint relative binding drift")
    require((ROOT / str(protocol.get("resume", {}).get("persistent_output"))).resolve() == result_dir, f"fold{fold} persistent output drift")

    initial = result.get("initial_state_sha256")
    final = result.get("final_state_sha256")
    schedule = training.get("schedule_sha256")
    require(initial == INITIAL_STATE_SHA256, f"fold{fold} initial state drift")
    require(isinstance(final, str) and HEX64.fullmatch(final) is not None, f"fold{fold} final state malformed")
    require(isinstance(schedule, str) and schedule == validation.get("schedule_sha256"), f"fold{fold} schedule binding drift")
    require(canonical_sha256(result.get("bag_permutation")) == BAG_SHA256, f"fold{fold} BAG permutation drift")
    logical_sha = logical_access_sha(access)

    protocol_bindings = protocol.get("bindings")
    require(isinstance(protocol_bindings, dict), f"fold{fold} protocol bindings missing")
    common: dict[str, dict[str, str]] = {}
    for name in COMMON_BINDING_NAMES:
        spec, _ = exact_source(protocol_bindings.get(name), f"fold{fold}.common.{name}")
        common[name] = spec
        require(validation.get("binding_hashes", {}).get(name) == spec["sha256"], f"fold{fold} validation common binding drift: {name}")

    eligible = access.get("eligible_query_execution_ordinals")
    allowed = access.get("allowed_reference_physical_rows")
    heldout = access.get("heldout_reference_physical_rows")
    require(isinstance(eligible, list) and all(type(x) is int for x in eligible), f"fold{fold} eligible queries malformed")
    require(isinstance(allowed, list) and all(type(x) is int for x in allowed), f"fold{fold} allowed references malformed")
    require(isinstance(heldout, list) and all(type(x) is int for x in heldout), f"fold{fold} heldout references malformed")
    require(len(eligible) == access.get("eligible_query_count") == validation.get("eligible_query_count"), f"fold{fold} eligible count drift")
    require(len(allowed) == access.get("allowed_reference_count") == validation.get("allowed_reference_count"), f"fold{fold} allowed count drift")
    require(len(heldout) == access.get("heldout_reference_count"), f"fold{fold} heldout count drift")
    require(not (set(allowed) & set(heldout)), f"fold{fold} allowed/heldout references overlap")

    record = {
        "outer_fold": fold,
        "authority": specs["authority"],
        "protocol": specs["protocol"],
        "run_identity": specs["run_identity"],
        "access_manifest": {**specs["access_manifest"], "logical_sha256": logical_sha},
        "train_result": specs["train_result"],
        "training_validation": specs["training_validation"],
        "checkpoint": specs["checkpoint"],
        "schedule_sha256": schedule,
        "initial_state_sha256": initial,
        "final_state_sha256": final,
        "update_count": UPDATES,
        "pair_count": PAIRS,
        "eligible_query_count": len(eligible),
        "allowed_reference_count": len(allowed),
        "heldout_reference_count": len(heldout),
    }
    sets = {"eligible": set(eligible), "allowed": set(allowed), "heldout": set(heldout)}
    record["_common_bindings"] = common
    return record, sets


def build_manifest(protocol_path: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    require(protocol_path.is_file() and not protocol_path.is_symlink(), "foldset protocol missing/symlinked")
    protocol = read_json(protocol_path, "foldset protocol")
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA, "foldset protocol schema drift")
    require(protocol.get("stage") == PROTOCOL_STAGE, "foldset protocol stage drift")
    require(protocol.get("claim_level") == CLAIM_LEVEL, "foldset protocol claim boundary drift")
    no_science(protocol, "foldset protocol")
    raw_authority_spec = protocol.get("authority")
    require(
        isinstance(raw_authority_spec, dict)
        and set(raw_authority_spec) == {"path", "sha256", "required_status"},
        "foldset authority binding schema drift",
    )
    authority_source, authority_path = exact_source(
        {"path": raw_authority_spec["path"], "sha256": raw_authority_spec["sha256"]},
        "foldset.authority",
    )
    authority_spec = {
        **authority_source,
        "required_status": str(raw_authority_spec["required_status"]),
    }
    foldset_authority = read_json(authority_path, "foldset authority")
    require(
        foldset_authority.get("status") == raw_authority_spec["required_status"],
        "foldset authority status drift",
    )
    require(
        foldset_authority.get("next_authorized_stage") == PROTOCOL_STAGE,
        "foldset authority stage drift",
    )
    require(
        foldset_authority.get("natural_training_authorized") is False,
        "foldset authority must not authorize natural training",
    )
    require(
        foldset_authority.get("protected_access_counts") == PROTECTED_ZERO,
        "foldset authority protected counts drift",
    )
    no_science(foldset_authority, "foldset authority")
    implementation_bindings = protocol.get("implementation_bindings")
    require(
        isinstance(implementation_bindings, dict)
        and set(implementation_bindings) == {"materializer", "validator", "tests"},
        "foldset implementation binding schema drift",
    )
    for name in ("materializer", "validator", "tests"):
        exact_source(implementation_bindings[name], f"foldset.{name}")
    output_dir_text = protocol.get("output_dir")
    require(isinstance(output_dir_text, str) and output_dir_text != "", "foldset output_dir missing")
    output_dir = (ROOT / output_dir_text).resolve()
    require(not Path(output_dir_text).is_absolute() and output_dir.is_relative_to(ROOT.resolve()), "foldset output_dir escapes RC root")
    inputs = protocol.get("fold_inputs")
    require(isinstance(inputs, list) and len(inputs) == 4, "foldset protocol must contain exactly four fold inputs")

    audited = [audit_fold(row) for row in inputs]
    audited.sort(key=lambda item: item[0]["outer_fold"])
    require(tuple(item[0]["outer_fold"] for item in audited) == FOLDS, "foldset must contain folds 1..4 exactly once")
    records = [item[0] for item in audited]
    set_rows = [item[1] for item in audited]

    common_bindings = records[0].pop("_common_bindings")
    for record in records[1:]:
        require(record.pop("_common_bindings") == common_bindings, "common source bindings differ across folds")
    require(len({row["initial_state_sha256"] for row in records}) == 1, "initial state differs across folds")
    require(len({row["checkpoint"]["path"] for row in records}) == 4, "checkpoint paths are not fold-isolated")

    universes = [row["allowed"] | row["heldout"] for row in set_rows]
    require(all(universe == universes[0] for universe in universes), "optimization reference universe differs across folds")
    universe = universes[0]
    require(len(universe) == 50, "optimization reference universe must contain 50 rows")
    heldout_sets = [row["heldout"] for row in set_rows]
    pairwise_intersections = sum(
        len(heldout_sets[i] & heldout_sets[j])
        for i in range(4) for j in range(i + 1, 4)
    )
    heldout_union = set().union(*heldout_sets)
    membership = Counter(value for values in heldout_sets for value in values)
    require([len(values) for values in heldout_sets] == [13, 12, 12, 13], "heldout reference counts drift")
    require(pairwise_intersections == 0 and heldout_union == universe, "heldout reference partition does not close")
    require(set(membership.values()) == {1}, "each reference must be held out exactly once")
    require(all(row["allowed"] == universe - row["heldout"] for row in set_rows), "allowed references are not universe minus heldout")

    eligible_sets = [row["eligible"] for row in set_rows]
    eligible_union = set().union(*eligible_sets)
    query_membership = Counter(value for values in eligible_sets for value in values)
    require(all(0 <= value < 600 for value in eligible_union), "eligible query ordinal out of 600-role range")
    require(len(eligible_union) == 594, "eligible query union must contain 594 ordinals")
    require(set(query_membership.values()) == {3}, "each eligible query must be outer-train in exactly three folds")
    globally_ineligible = sorted(set(range(600)) - eligible_union)
    require(globally_ineligible == [25, 26, 101, 346, 354, 470], "global ineligible query set drift")

    protocol_sha = file_sha256(protocol_path)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": MANIFEST_STATUS,
        "arm": ARM,
        "outer_folds": list(FOLDS),
        "source_mode": "immutable_paths_and_sha256_no_copy",
        "authority": authority_spec,
        "foldset_protocol": {
            "path": str(protocol_path.relative_to(ROOT.resolve())),
            "sha256": protocol_sha,
        },
        "claim_level": CLAIM_LEVEL,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "protected_access_counts": PROTECTED_ZERO,
        "shared_training_contract": {
            "seed": SEED,
            "parameter_count": PARAMETERS,
            "initial_state_sha256": INITIAL_STATE_SHA256,
            "normalized_training_semantics_sha256": TRAINING_SHA256,
            "normalized_cache_contract_sha256": CACHE_SHA256,
            "normalized_forbidden_contract_sha256": FORBIDDEN_SHA256,
            "bag_permutation_sha256": BAG_SHA256,
            "model_state_schema_sha256": MODEL_STATE_SCHEMA_SHA256,
            "common_bindings": common_bindings,
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


def materialize(protocol_path: Path, output_dir: Path) -> Path:
    manifest = build_manifest(protocol_path)
    output_dir = output_dir.resolve()
    protocol = read_json(protocol_path.resolve(), "foldset protocol")
    expected_output = (ROOT / str(protocol.get("output_dir"))).resolve()
    require(output_dir == expected_output and output_dir.is_relative_to(ROOT.resolve()), "CLI output-dir differs from protocol")
    output = output_dir / "foldset_manifest.json"
    require(not (output_dir / "foldset_validation.json").exists(), "validation already exists before freeze")
    write_once(output, manifest)
    require(not output_dir.is_symlink() and not output.is_symlink(), "foldset output may not be symlinked")
    entries = {path.name for path in output_dir.iterdir()}
    require(entries == {"foldset_manifest.json"}, "materializer output contains undeclared files")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the four RCDE_BAG engineering folds")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = materialize(args.protocol, args.output_dir)
    print(json.dumps({"manifest": str(path), "status": MANIFEST_STATUS}, sort_keys=True))


if __name__ == "__main__":
    main()
