#!/usr/bin/env python3
"""Independently validate a completed DINO-RCDE R1 V1.7 training chain.

This validator deliberately does not import the V1.7 runner.  It replays the
query namespaces, fold-local pools, 2+2 schedule, frozen candidate directions,
learning-rate schedule, pair cardinality, cache/access allowlists, and every
chunk receipt from the frozen ledgers.  A validation artifact is written only
after every check passes; failure leaves the requested output absent.
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "rc_dino_rcde_r1_main_train_protocol_v1_7_contract_repair_20260813"
AUTHORITY_STATUS = (
    "DINO_RCDE_R1_CACHE_READY_BAG_FOLD1_V17_CONTRACT_REPAIR_TRAINING_AUTHORIZED"
)
AUTHORITY_STAGE = "R1_MAIN_BAG_FOLD1_V17_CONTRACT_REPAIR_TRAINING"
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
PROTECTED_ZERO_KEYS = {
    "C8_runtime_read_count",
    "S8_runtime_read_count",
    "opened_runtime_read_count",
    "sealed_runtime_read_count",
    "unauthorized_natural_result_read_count",
    "home_files_modified",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHUNK_DIRECTORY_NAME = re.compile(r"^chunk_job(.+)$")
COMMITTED_CHUNK_FILES = {
    "chunk.json",
    "input_preflight.json",
    "runtime_read_ledger.json",
    "resume_state.pt",
}


class ValidationAbort(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationAbort(message)


def exact_fields(value: Mapping[str, Any], fields: set[str], context: str) -> None:
    observed = set(value)
    if observed != fields:
        missing = sorted(fields - observed)
        extra = sorted(observed - fields)
        raise ValidationAbort(f"{context} field drift; missing={missing}, extra={extra}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationAbort(f"missing JSON input: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValidationAbort(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationAbort(f"JSON root is not an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_parts(namespace: str, *parts: object) -> str:
    return hashlib.sha256(
        "\0".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    ).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), f"state tensor is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(canonical_bytes(list(tensor.shape)) + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def cache_logical_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "tokens": tensor_sha256(payload["tokens_fp16"]),
            "mask": tensor_sha256(payload["valid_patch_mask"]),
            "geometry": payload["geometry_receipt"],
            "model": payload["model_checkpoint_logical_sha256"],
            "source": payload["source_image_sha256"],
        }
    )


def resolve_bound(path_text: str) -> Path:
    path = (ROOT / path_text).resolve()
    require(path != ROOT and ROOT in path.parents, f"binding escapes RC root: {path}")
    require(
        re.search(
            r"(^|[/_.-])(c8|s8|opened|sealed)(?=$|[/_.-])",
            path.relative_to(ROOT).as_posix(),
            flags=re.IGNORECASE,
        )
        is None,
        f"protected binding: {path}",
    )
    return path


def fold_number(value: object) -> int:
    text = str(value).strip().upper()
    if text.startswith("F"):
        text = text[1:]
    try:
        result = int(text)
    except ValueError as error:
        raise ValidationAbort(f"invalid fold value: {value}") from error
    return result


def protected_zero_exact(value: Any, context: str) -> None:
    require(isinstance(value, dict), f"{context} protected counts are not an object")
    exact_fields(value, PROTECTED_ZERO_KEYS, f"{context} protected counts")
    require(
        all(type(item) is int and item == 0 for item in value.values()),
        f"{context} protected count is nonzero or non-integer",
    )


def learning_rate(update_one: int) -> float:
    require(1 <= update_one <= UPDATES, "learning-rate update outside frozen range")
    maximum = 3.0e-4
    minimum = 3.0e-5
    warmup = 128
    if update_one <= warmup:
        return maximum * update_one / warmup
    progress = (update_one - warmup) / (UPDATES - warmup)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def candidate_rows(episode: Mapping[str, Any]) -> list[int]:
    return [int(episode["target_physical_row"])] + [
        int(row["physical_row"]) for row in episode.get("selected_negatives", [])
    ]


def frozen_directions(episode: Mapping[str, Any]) -> list[bool]:
    rows = candidate_rows(episode)
    order = [int(row["physical_row"]) for row in episode.get("model_candidate_order", [])]
    require(len(rows) == len(set(rows)), "episode candidate rows are not unique")
    require(len(order) == len(set(order)), "frozen candidate order is not unique")
    require(len(order) == len(rows) and set(order) == set(rows), "frozen candidate order row-set drift")
    positions = {row: position for position, row in enumerate(order)}
    target = rows[0]
    return [positions[target] < positions[rival] for rival in rows[1:]]


def update_episodes(
    correct: Sequence[dict[str, Any]], wrong: Sequence[dict[str, Any]], update_zero: int
) -> list[dict[str, Any]]:
    selected = [
        correct[(2 * update_zero) % len(correct)],
        correct[(2 * update_zero + 1) % len(correct)],
        wrong[(2 * update_zero) % len(wrong)],
        wrong[(2 * update_zero + 1) % len(wrong)],
    ]
    require(
        len({str(row["query_id"]) for row in selected}) == EPISODES_PER_UPDATE,
        "expected update contains duplicate queries",
    )
    return selected


def schedule_event(
    fold: int, update_one: int, episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    queries = []
    for episode in episodes:
        rivals = [int(row["physical_row"]) for row in episode["selected_negatives"]]
        order = [int(row["physical_row"]) for row in episode["model_candidate_order"]]
        queries.append(
            {
                "query_id": str(episode["query_id"]),
                "query_ordinal": int(episode["query_ordinal"]),
                "execution_ordinal": int(episode["execution_ordinal"]),
                "source_image_sha256": str(episode["_source_image_sha256"]),
                "target_physical_row": int(episode["target_physical_row"]),
                "rivals": rivals,
                "target_first": frozen_directions(episode),
                "frozen_model_candidate_rows": order,
                "frozen_model_candidate_rows_sha256": canonical_sha256(order),
            }
        )
    return {"outer_fold": fold, "update": update_one, "queries": queries}


def validate_protocol(
    protocol_path: Path, authority_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    protocol = read_json(protocol_path)
    authority = read_json(authority_path)
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA, "protocol schema drift")
    authority_spec = protocol.get("authority")
    require(isinstance(authority_spec, dict), "protocol authority binding absent")
    require(
        authority_path.resolve() == resolve_bound(str(authority_spec.get("path", ""))),
        "authority path drift",
    )
    authority_sha = file_sha256(authority_path)
    require(authority_sha == authority_spec.get("sha256"), "authority digest drift")
    require(
        authority.get("status") == authority_spec.get("required_status") == AUTHORITY_STATUS,
        "authority status drift",
    )
    require(authority.get("next_authorized_stage") == AUTHORITY_STAGE, "authority stage drift")
    require(authority.get("natural_training_authorized") is True, "training authority absent")
    require(authority.get("automatic_stage_advance") is False, "authority auto-advance enabled")
    require(authority.get("scientific_GO_or_NO_GO") is None, "authority contains scientific decision")
    require(
        authority.get("next_on_validation_pass")
        == "R1_MAIN_BAG_FOLD2_AUTHORITY_UPDATE_REQUIRED",
        "authority validation-pass transition drift",
    )
    protected_zero_exact(authority.get("protected_access_counts"), "authority")
    forbidden = {
        "RCDE_BAG_FOLD2",
        "RCDE_BAG_FOLD3",
        "RCDE_BAG_FOLD4",
        "RCDE_CONTEXT",
        "R1_CONTROL_TRAINING",
        "R1_HELDOUT_LABEL_JOIN",
        "R1_FULL_GALLERY_EVALUATION",
        "R1_SCIENTIFIC_GO_OR_NO_GO",
        "C8",
        "S8",
        "opened",
        "sealed",
    }
    require(
        forbidden.issubset(set(authority.get("explicitly_not_authorized", []))),
        "authority forbidden-stage set is incomplete",
    )
    scope = authority.get("authorized_scope")
    require(isinstance(scope, dict), "authority scope absent")
    for key, expected in {
        "stage": AUTHORITY_STAGE,
        "arm": "RCDE_BAG",
        "outer_fold": 1,
        "optimizer_updates_total": UPDATES,
        "seed": SEED,
        "maximum_submitted_jobs_in_chain": 4,
        "maximum_concurrently_running_tasks": 1,
    }.items():
        require(scope.get(key) == expected, f"authority scope drift: {key}")
    require(protocol.get("automatic_stage_advance") is False, "protocol auto-advance enabled")

    binding_hashes: dict[str, str] = {}
    bindings = protocol.get("bindings")
    require(isinstance(bindings, dict) and bindings, "protocol bindings absent")
    required_bindings = {
        "scientific_contract",
        "cache_protocol",
        "cache_result",
        "cache_validation",
        "training_roles",
        "episode_ledger",
        "resource_shape_ledger",
        "model_visible_c128",
        "rcde_core",
        "base_runner",
        "base_validator",
        "contract_repair_runner",
        "contract_repair_validator",
    }
    require(required_bindings.issubset(bindings), "required V1.7 binding absent")
    for name, binding in bindings.items():
        require(isinstance(binding, dict), f"binding is not an object: {name}")
        require(set(binding) >= {"path", "sha256"}, f"binding fields absent: {name}")
        path = resolve_bound(str(binding["path"]))
        require(path.is_file(), f"bound file absent: {name}")
        observed = file_sha256(path)
        require(observed == binding["sha256"], f"binding digest drift: {name}")
        binding_hashes[name] = observed

    cache_validation = read_json(resolve_bound(bindings["cache_validation"]["path"]))
    require(
        cache_validation.get("status") == "DINO_RCDE_R1_CACHE_VALIDATION_PASS",
        "canonical cache validation is not PASS",
    )
    training = protocol.get("training")
    require(isinstance(training, dict), "training contract absent")
    expected_training = {
        "arms": ["RCDE_BAG"],
        "outer_folds": [1],
        "seed": SEED,
        "parameter_count": EXPECTED_PARAMETERS,
        "optimizer": "AdamW",
        "deterministic_algorithms": True,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "gradient_clip_l2": 1.0,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "raw_correct_per_update": 2,
        "raw_wrong_per_update": 2,
        "warmup_updates": 128,
        "final_learning_rate": 3.0e-5,
        "query_tile_rows": QUERY_TILE_ROWS,
        "reference_tile_rows": REFERENCE_TILE_ROWS,
        "loss": "mean_softplus_negative_signed_pair_logit_tau_1",
        "query_order_namespace": QUERY_ORDER_NAMESPACE,
        "query_order_key": "source_image_sha256",
        "query_order_tie_break": "stable_episode_ledger_order",
        "candidate_direction_source": "P0_frozen_model_candidate_order",
        "cache_query_key": "execution_ordinal",
        "outer_train_only": True,
        "heldout_target_join_allowed": False,
    }
    for key, expected in expected_training.items():
        require(training.get(key) == expected, f"training contract drift: {key}")
    update_value = training.get("updates_total", training.get("updates_per_arm_fold"))
    require(update_value == UPDATES, "training update budget drift")

    cache = protocol.get("cache")
    require(isinstance(cache, dict), "cache contract absent")
    require(cache.get("model_checkpoint_logical_sha256") == EXPECTED_MODEL_SHA256, "cache model drift")
    if "query_count" in cache:
        require(cache["query_count"] == 600, "cache query count drift")
    if "reference_count" in cache:
        require(cache["reference_count"] == 4_748, "cache reference count drift")
    execution = protocol.get("execution")
    require(isinstance(execution, dict), "execution contract absent")
    require(
        type(execution.get("maximum_jobs_submitted_now")) is int
        and 1 <= execution["maximum_jobs_submitted_now"] <= 4,
        "execution job budget drift",
    )
    require(
        execution.get("chunk_commit") == "atomic_directory_rename"
        and execution.get("committed_chunk_files")
        == sorted(COMMITTED_CHUNK_FILES)
        and execution.get("finalization_recovery") is True,
        "atomic chunk/finalization execution contract drift",
    )
    slurm_path = resolve_bound(str(execution.get("slurm_path", "")))
    require(
        slurm_path.is_file()
        and file_sha256(slurm_path) == execution.get("slurm_sha256"),
        "Slurm execution binding drift",
    )

    anchor = authority.get("execution_content_anchors", {}).get("protocol", {})
    require(
        resolve_bound(str(anchor.get("path", ""))) == protocol_path.resolve(),
        "authority reverse protocol path drift",
    )
    require(anchor.get("required_schema") == PROTOCOL_SCHEMA, "authority reverse schema drift")
    return protocol, authority, binding_hashes


def build_population(protocol: Mapping[str, Any], fold: int) -> dict[str, Any]:
    bindings = protocol["bindings"]
    roles = read_json(resolve_bound(bindings["training_roles"]["path"]))
    ledger = read_json(resolve_bound(bindings["episode_ledger"]["path"]))
    shape = read_json(resolve_bound(bindings["resource_shape_ledger"]["path"]))
    require(roles.get("query_count") == 600, "role query population drift")
    require(ledger.get("episode_count") == 1_800, "episode population drift")
    require(shape.get("query_count") == 600, "shape query population drift")
    require(shape.get("reference_union_count") == 4_748, "shape reference population drift")

    role_rows = roles.get("records", roles.get("rows", []))
    query_rows = [row for row in shape.get("rows", []) if row.get("kind") == "query"]
    reference_rows = [row for row in shape.get("rows", []) if row.get("kind") == "reference"]
    require(len(role_rows) == len(query_rows) == 600, "role/shape row count drift")
    require(len(reference_rows) == 4_748, "reference shape row count drift")
    role_by_id = {str(row["query_id"]): row for row in role_rows}
    query_by_id = {str(row["id"]): row for row in query_rows}
    query_by_execution = {int(row["execution_ordinal"]): row for row in query_rows}
    reference_by_row = {int(row["physical_row"]): row for row in reference_rows}
    require(len(role_by_id) == len(query_by_id) == len(query_by_execution) == 600, "duplicate query mapping")
    require(len(reference_by_row) == 4_748, "duplicate reference mapping")
    require(sorted(query_by_execution) == list(range(600)), "execution ordinals not dense 0..599")
    require(len({int(row["query_ordinal"]) for row in query_rows}) == 600, "historical ordinals not unique")

    episodes = ledger.get("episodes", [])
    require(isinstance(episodes, list) and len(episodes) == 1_800, "episode list drift")
    target_by_query: dict[str, int] = {}
    reference_semantics: dict[int, tuple[str, str]] = {}
    for episode in episodes:
        query_id = str(episode["query_id"])
        role = role_by_id.get(query_id)
        row = query_by_id.get(query_id)
        require(role is not None and row is not None, "episode query mapping absent")
        require(int(episode["query_ordinal"]) == int(role["query_ordinal"]) == int(row["query_ordinal"]), "historical query mapping drift")
        require(int(episode["execution_ordinal"]) == int(row["execution_ordinal"]), "execution query mapping drift")
        frozen_directions(episode)
        target = int(episode["target_physical_row"])
        require(target in reference_by_row, "target reference outside shape union")
        prior = target_by_query.setdefault(query_id, target)
        require(prior == target, "query target row changes across fold receipts")
        semantics = (str(role["identity"]), str(role["supergroup"]))
        prior_semantics = reference_semantics.setdefault(target, semantics)
        require(prior_semantics == semantics, "target reference identity/supergroup drift")
    require(len(target_by_query) == 600, "target mapping does not cover 600 queries")
    require(len(reference_semantics) == 50, "training reference semantic population drift")
    for episode in episodes:
        for negative in episode["selected_negatives"]:
            physical = int(negative["physical_row"])
            require(physical in reference_semantics, "negative reference has no role semantics")
            require(
                str(negative["corrected_identity"]) == reference_semantics[physical][0],
                "negative corrected identity drift",
            )

    correct: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    for source in episodes:
        if fold_number(source["heldout_fold"]) != fold or source.get("pair_loss_eligible") is not True:
            continue
        query_id = str(source["query_id"])
        role = role_by_id[query_id]
        require(fold_number(role["inner_fold"]) != fold, "heldout query entered outer training")
        episode = dict(source)
        episode["_source_image_sha256"] = str(query_by_id[query_id]["source_image_sha256"])
        (correct if episode.get("full_gallery_RAW_correct") is True else wrong).append(episode)
    correct.sort(key=lambda row: hash_parts(QUERY_ORDER_NAMESPACE, fold, row["_source_image_sha256"]))
    wrong.sort(key=lambda row: hash_parts(QUERY_ORDER_NAMESPACE, fold, row["_source_image_sha256"]))
    require(len(correct) >= 2 and len(wrong) >= 2, "balanced training pools incomplete")
    require(
        len({str(row["query_id"]) for row in correct + wrong})
        == len(correct) + len(wrong),
        "outer-training query appears more than once",
    )

    eligible_queries = sorted({int(row["execution_ordinal"]) for row in correct + wrong})
    require(
        len(eligible_queries) == len(correct) + len(wrong),
        "outer-training execution slot appears more than once",
    )
    allowed_references = sorted(
        {reference for episode in correct + wrong for reference in candidate_rows(episode)}
    )
    heldout_query_ids = {
        str(row["query_id"]) for row in role_rows if fold_number(row["inner_fold"]) == fold
    }
    heldout_references = sorted({target_by_query[query_id] for query_id in heldout_query_ids})
    train_identities = {reference_semantics[row][0] for row in allowed_references}
    train_groups = {reference_semantics[row][1] for row in allowed_references}
    heldout_identities = {reference_semantics[row][0] for row in heldout_references}
    heldout_groups = {reference_semantics[row][1] for row in heldout_references}
    require(not set(allowed_references).intersection(heldout_references), "train/heldout reference overlap")
    require(not train_identities.intersection(heldout_identities), "train/heldout identity overlap")
    require(not train_groups.intersection(heldout_groups), "train/heldout supergroup overlap")

    fold_receipt = ledger.get("folds", {}).get(str(fold), {})
    require(len(allowed_references) == fold_receipt.get("allowed_physical_row_count"), "P0 allowlist count drift")
    require(canonical_sha256(allowed_references) == fold_receipt.get("allowed_physical_rows_sha256"), "P0 allowlist hash drift")
    require(len(heldout_references) == fold_receipt.get("heldout_physical_row_count"), "P0 heldout row count drift")
    require(len(correct) == fold_receipt.get("raw_correct_pool"), "P0 correct pool drift")
    require(len(wrong) == fold_receipt.get("raw_wrong_pool"), "P0 wrong pool drift")

    return {
        "roles": roles,
        "ledger": ledger,
        "shape": shape,
        "role_by_id": role_by_id,
        "query_by_id": query_by_id,
        "query_by_execution": query_by_execution,
        "reference_by_row": reference_by_row,
        "correct": correct,
        "wrong": wrong,
        "eligible_queries": eligible_queries,
        "allowed_references": allowed_references,
        "heldout_references": heldout_references,
        "train_identities": train_identities,
        "heldout_identities": heldout_identities,
        "train_groups": train_groups,
        "heldout_groups": heldout_groups,
    }


def cache_index(cache_root: Path, reference_rows: set[int]) -> dict[tuple[str, int], Path]:
    require(cache_root.is_dir(), "canonical cache root absent")
    pattern = re.compile(r"^(query|reference)_(\d+)\.pt$")
    result: dict[tuple[str, int], Path] = {}
    for path in cache_root.glob("*.pt"):
        match = pattern.fullmatch(path.name)
        require(match is not None, f"unexpected torch cache artifact: {path.name}")
        key = (match.group(1), int(match.group(2)))
        require(key not in result, f"duplicate cache key: {key}")
        result[key] = path
    expected = {("query", ordinal) for ordinal in range(600)} | {
        ("reference", row) for row in reference_rows
    }
    require(set(result) == expected, "canonical cache key set is not exact")
    return result


def validate_cache_payloads(
    index: Mapping[tuple[str, int], Path], population: Mapping[str, Any]
) -> None:
    for kind, ordinals, source_map in (
        ("query", population["eligible_queries"], population["query_by_execution"]),
        ("reference", population["allowed_references"], population["reference_by_row"]),
    ):
        for ordinal in ordinals:
            payload = torch.load(index[(kind, ordinal)], map_location="cpu", weights_only=False)
            require(payload.get("source_image_sha256") == source_map[ordinal]["source_image_sha256"], f"cache source drift: {kind}:{ordinal}")
            require(payload.get("model_checkpoint_logical_sha256") == EXPECTED_MODEL_SHA256, f"cache model drift: {kind}:{ordinal}")
            receipt = payload.get("cache_receipt")
            require(isinstance(receipt, dict) and receipt.get("model_receipt_sha256") == EXPECTED_MODEL_SHA256, f"nested cache model drift: {kind}:{ordinal}")
            tokens = payload.get("tokens_fp16")
            mask = payload.get("valid_patch_mask")
            geometry = payload.get("geometry_receipt")
            require(isinstance(tokens, torch.Tensor) and tokens.dtype == torch.float16 and tokens.ndim == 4, f"cache token contract drift: {kind}:{ordinal}")
            require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and mask.ndim == 3, f"cache mask contract drift: {kind}:{ordinal}")
            require(tuple(tokens.shape[:2]) == (1, 4) and tokens.shape[-1] == 768, f"cache token shape drift: {kind}:{ordinal}")
            require(isinstance(geometry, dict) and tuple(mask.shape[1:]) == tuple(geometry["grid_hw"]), f"cache geometry drift: {kind}:{ordinal}")
            require(tokens.shape[2] == mask.numel(), f"cache token/mask length drift: {kind}:{ordinal}")
            require(payload.get("logical_sha256") == cache_logical_sha256(payload), f"cache logical hash drift: {kind}:{ordinal}")


def expected_manifest(population: Mapping[str, Any], fold: int) -> dict[str, Any]:
    value = {
        "schema_version": "rc_dino_rcde_signed_reference_access_manifest_v1_7",
        "status": "DINO_RCDE_SIGNED_REFERENCE_ACCESS_FROZEN",
        "outer_fold": fold,
        "query_order_namespace": QUERY_ORDER_NAMESPACE,
        "query_cache_key_field": "execution_ordinal",
        "historical_query_ordinal_model_visible": False,
        "eligible_query_execution_ordinals": population["eligible_queries"],
        "eligible_query_execution_ordinals_sha256": canonical_sha256(population["eligible_queries"]),
        "allowed_reference_physical_rows": population["allowed_references"],
        "allowed_reference_physical_rows_sha256": canonical_sha256(population["allowed_references"]),
        "heldout_reference_physical_rows": population["heldout_references"],
        "heldout_reference_physical_rows_sha256": canonical_sha256(population["heldout_references"]),
        "reference_row_intersection_count": 0,
        "eligible_query_count": len(population["eligible_queries"]),
        "allowed_reference_count": len(population["allowed_references"]),
        "heldout_reference_count": len(population["heldout_references"]),
        "all_episode_mapping_count": 1_800,
        "correct_pool_count": len(population["correct"]),
        "wrong_pool_count": len(population["wrong"]),
        "protected_access_counts": {key: 0 for key in sorted(PROTECTED_ZERO_KEYS)},
    }
    value["logical_sha256"] = canonical_sha256(value)
    return value


def expected_schedule(population: Mapping[str, Any], fold: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_pairs = 0
    per_update: list[dict[str, Any]] = []
    for update_zero in range(UPDATES):
        episodes = update_episodes(population["correct"], population["wrong"], update_zero)
        event = schedule_event(fold, update_zero + 1, episodes)
        digest.update(canonical_bytes(event) + b"\n")
        pair_count = sum(len(row["selected_negatives"]) for row in episodes)
        total_pairs += pair_count
        per_update.append({"episodes": episodes, "event": event, "pair_count": pair_count})
    return {"sha256": digest.hexdigest(), "pair_count": total_pairs, "updates": per_update}


def validate_identity(path: Path, protocol_sha: str, authority_sha: str) -> dict[str, Any]:
    value = read_json(path)
    expected = {
        "schema_version": "rc_dino_rcde_r1_main_run_identity_v1_7",
        "arm": "RCDE_BAG",
        "outer_fold": 1,
        "protocol_sha256": protocol_sha,
        "authority_sha256": authority_sha,
    }
    require(value == expected, "run identity drift")
    return value


def validate_preflight(
    value: Mapping[str, Any], *, job_id: str, protocol_sha: str, authority_sha: str,
    manifest: Mapping[str, Any], manifest_file_sha: str, population: Mapping[str, Any]
) -> None:
    expected_fields = {
        "schema_version", "status", "outer_fold", "all_episode_mapping_count",
        "eligible_query_payloads_independently_validated", "allowed_reference_payloads_independently_validated",
        "cache_query_key_count", "cache_reference_key_count", "cache_execution_ordinal_range",
        "historical_query_ordinal_range", "known_regression_mapping", "access_manifest_logical_sha256",
        "protected_access_counts", "automatic_stage_advance", "scientific_GO_or_NO_GO", "job_id",
        "protocol_sha256", "authority_sha256", "signed_reference_access_manifest_sha256",
    }
    exact_fields(value, expected_fields, f"preflight job {job_id}")
    protected_zero_exact(value["protected_access_counts"], f"preflight job {job_id}")
    historical = [int(row["query_ordinal"]) for row in population["query_by_id"].values()]
    regression = population["query_by_id"].get("OUTCOME-0504")
    require(regression is not None, "known regression query absent")
    expected = {
        "schema_version": "rc_dino_rcde_r1_main_input_preflight_v1_7",
        "status": "DINO_RCDE_R1_MAIN_INPUT_PREFLIGHT_PASS",
        "outer_fold": 1,
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
            for row in candidate_rows(episode):
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
    value: Mapping[str, Any], *, job_id: str, start: int, completed: int,
    schedule: Mapping[str, Any], allowed_queries: set[int], allowed_references: set[int]
) -> None:
    fields = {
        "schema_version", "status", "job_id", "start_update", "completed_updates",
        "query_rows", "reference_rows", "query_read_set_sha256", "reference_read_set_sha256",
        "protected_access_counts", "logical_sha256",
    }
    exact_fields(value, fields, f"runtime read ledger job {job_id}")
    protected_zero_exact(value["protected_access_counts"], f"runtime read ledger job {job_id}")
    require(value["schema_version"] == "rc_dino_rcde_r1_runtime_cache_read_ledger_v1_7", "read ledger schema drift")
    require(value["status"] == "DINO_RCDE_R1_RUNTIME_CACHE_READ_LEDGER_COMPLETE", "read ledger status drift")
    require(str(value["job_id"]) == job_id and value["start_update"] == start and value["completed_updates"] == completed, "read ledger range/job drift")
    query_rows, reference_rows = expected_read_rows(schedule, start, completed)
    require(value["query_rows"] == query_rows, f"query read counts drift for job {job_id}")
    require(value["reference_rows"] == reference_rows, f"reference read counts drift for job {job_id}")
    query_set = [row["ordinal"] for row in query_rows]
    reference_set = [row["ordinal"] for row in reference_rows]
    require(set(query_set).issubset(allowed_queries), "query read outside eligible set")
    require(set(reference_set).issubset(allowed_references), "reference read outside allowlist")
    require(value["query_read_set_sha256"] == canonical_sha256(query_set), "query read-set hash drift")
    require(value["reference_read_set_sha256"] == canonical_sha256(reference_set), "reference read-set hash drift")
    body = dict(value)
    logical = body.pop("logical_sha256")
    require(logical == canonical_sha256(body), "read ledger logical hash drift")


def validate_chunks(
    output_dir: Path, protocol: Mapping[str, Any], protocol_sha: str, authority_sha: str,
    manifest: Mapping[str, Any], manifest_file_sha: str, population: Mapping[str, Any],
    schedule: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidates = sorted(output_dir.glob("chunk_job*"))
    require(candidates, "no committed chunk directories found")
    require(
        len(candidates) <= int(protocol["execution"]["maximum_jobs_submitted_now"]),
        "chunk count exceeds authority",
    )
    directories_by_job: dict[str, Path] = {}
    chunks: list[dict[str, Any]] = []
    for directory in candidates:
        match = CHUNK_DIRECTORY_NAME.fullmatch(directory.name)
        require(
            match is not None and directory.is_dir() and not directory.is_symlink(),
            f"invalid committed chunk artifact: {directory.name}",
        )
        job_id = match.group(1)
        require(job_id and job_id not in directories_by_job, "duplicate/empty committed chunk job ID")
        observed_files = {path.name for path in directory.iterdir()}
        require(
            observed_files == COMMITTED_CHUNK_FILES
            and all(path.is_file() and not path.is_symlink() for path in directory.iterdir()),
            f"committed chunk directory is not an exact four-file transaction: {directory.name}",
        )
        directories_by_job[job_id] = directory
        chunks.append(read_json(directory / "chunk.json"))
    chunks.sort(key=lambda row: (int(row.get("start_update", -1)), str(row.get("job_id", ""))))
    cursor = 0
    seen: set[str] = set()
    previous_resume: Mapping[str, Any] | None = None
    for ordinal, chunk in enumerate(chunks):
        fields = {
            "schema_version", "status", "arm", "outer_fold", "job_id", "start_update",
            "completed_updates", "training_complete", "restored_from_resume", "input_preflight",
            "input_preflight_sha256", "runtime_read_ledger", "runtime_read_ledger_sha256",
            "restored_resume_state", "restored_resume_state_sha256", "resume_state",
            "resume_state_sha256", "elapsed_seconds", "completed_at_utc", "protected_access_counts",
            "training_seconds", "automatic_stage_advance", "scientific_GO_or_NO_GO",
        }
        job_id = str(chunk.get("job_id"))
        exact_fields(chunk, fields, f"chunk job {job_id}")
        require(job_id in directories_by_job and job_id not in seen, "chunk job ID drift/duplicate")
        seen.add(job_id)
        directory = directories_by_job[job_id]
        relative_directory = directory.relative_to(output_dir).as_posix()
        require(chunk["schema_version"] == "rc_dino_rcde_r1_main_dev_chunk_v1_7", "chunk schema drift")
        require(chunk["status"] == "DINO_RCDE_R1_MAIN_DEV_CHUNK_COMPLETE", "chunk status drift")
        require(chunk["arm"] == "RCDE_BAG" and chunk["outer_fold"] == 1, "chunk scope drift")
        start = int(chunk["start_update"])
        completed = int(chunk["completed_updates"])
        require(start == cursor and start < completed <= UPDATES, "chunk coverage is non-contiguous")
        require(chunk["restored_from_resume"] is (ordinal > 0), "chunk resume flag drift")
        if ordinal == 0:
            require(
                chunk["restored_resume_state"] is None
                and chunk["restored_resume_state_sha256"] is None,
                "first chunk unexpectedly restored prior state",
            )
        else:
            previous = chunks[ordinal - 1]
            require(
                chunk["restored_resume_state"] == previous["resume_state"]
                and chunk["restored_resume_state_sha256"]
                == previous["resume_state_sha256"],
                "chunk resume handoff does not bind prior committed state",
            )
        require(chunk["training_complete"] is (completed == UPDATES), "chunk completion flag drift")
        require(type(chunk["elapsed_seconds"]) in (int, float) and math.isfinite(float(chunk["elapsed_seconds"])) and float(chunk["elapsed_seconds"]) > 0, "chunk elapsed time invalid")
        require(
            type(chunk["training_seconds"]) in (int, float)
            and math.isfinite(float(chunk["training_seconds"]))
            and 0 < float(chunk["training_seconds"]) <= float(chunk["elapsed_seconds"]),
            "chunk training time invalid",
        )
        require(isinstance(chunk["completed_at_utc"], str) and chunk["completed_at_utc"], "chunk completion time absent")
        expected_preflight_relative = f"{relative_directory}/input_preflight.json"
        expected_read_relative = f"{relative_directory}/runtime_read_ledger.json"
        expected_resume_relative = f"{relative_directory}/resume_state.pt"
        preflight_path = directory / "input_preflight.json"
        read_path = directory / "runtime_read_ledger.json"
        state_path = directory / "resume_state.pt"
        require(
            chunk["resume_state"] == expected_resume_relative
            and state_path.is_file()
            and file_sha256(state_path) == chunk["resume_state_sha256"],
            "chunk resume-state binding drift",
        )
        protected_zero_exact(chunk["protected_access_counts"], f"chunk job {job_id}")
        require(chunk["automatic_stage_advance"] is False and chunk["scientific_GO_or_NO_GO"] is None, "chunk claim boundary drift")

        require(chunk["input_preflight"] == expected_preflight_relative, "chunk preflight filename drift")
        require(chunk["input_preflight_sha256"] == file_sha256(preflight_path), "chunk preflight digest drift")
        require(chunk["runtime_read_ledger"] == expected_read_relative, "chunk read-ledger filename drift")
        require(chunk["runtime_read_ledger_sha256"] == file_sha256(read_path), "chunk read-ledger digest drift")
        validate_preflight(
            read_json(preflight_path), job_id=job_id, protocol_sha=protocol_sha,
            authority_sha=authority_sha, manifest=manifest, manifest_file_sha=manifest_file_sha,
            population=population,
        )
        validate_read_ledger(
            read_json(read_path), job_id=job_id, start=start, completed=completed,
            schedule=schedule, allowed_queries=set(population["eligible_queries"]),
            allowed_references=set(population["allowed_references"]),
        )
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        state_fields = {
            "schema_version", "arm", "outer_fold", "completed_updates",
            "protocol_sha256", "authority_sha256", "initial_state_sha256",
            "model_state_dict", "optimizer_state_dict", "loss_trace", "pair_count",
            "python_random_state", "numpy_random_state", "torch_cpu_rng_state",
            "torch_cuda_rng_state_all",
        }
        require(isinstance(state, dict), "chunk resume state is not an object")
        exact_fields(state, state_fields, f"chunk resume state job {job_id}")
        require(
            state["schema_version"] == "rc_dino_rcde_r1_main_resume_state_v1_7"
            and state["arm"] == "RCDE_BAG"
            and state["outer_fold"] == 1
            and state["completed_updates"] == completed
            and state["protocol_sha256"] == protocol_sha
            and state["authority_sha256"] == authority_sha,
            "chunk resume metadata drift",
        )
        require(
            isinstance(state["loss_trace"], list)
            and len(state["loss_trace"]) == completed,
            "chunk resume loss-trace cardinality drift",
        )
        if previous_resume is not None:
            previous_completed = int(previous_resume["completed_updates"])
            require(
                state["loss_trace"][:previous_completed]
                == previous_resume["loss_trace"],
                "chunk resume loss-trace prefix was rewritten",
            )
            require(
                int(state["pair_count"]) >= int(previous_resume["pair_count"]),
                "chunk resume pair count regressed",
            )
        expected_pairs = sum(
            int(update["pair_count"]) for update in schedule["updates"][:completed]
        )
        require(state["pair_count"] == expected_pairs, "chunk resume pair-count drift")
        previous_resume = state
        cursor = completed
    require(cursor == UPDATES and chunks[-1]["training_complete"] is True, "chunk chain does not cover update 2048")
    require(all(not row["training_complete"] for row in chunks[:-1]), "nonfinal chunk claims completion")
    return chunks


def validate_loss_trace(value: Any, schedule: Mapping[str, Any]) -> None:
    require(isinstance(value, list) and len(value) == UPDATES, "loss trace cardinality drift")
    fields = {"update", "learning_rate", "mean_pair_loss", "gradient_norm_before_clip", "pair_count"}
    for index, row in enumerate(value):
        require(isinstance(row, dict), "loss trace row is not an object")
        exact_fields(row, fields, f"loss trace update {index + 1}")
        require(row["update"] == index + 1, "loss trace update index drift")
        require(math.isclose(float(row["learning_rate"]), learning_rate(index + 1), rel_tol=0.0, abs_tol=1e-15), "learning-rate trace drift")
        require(type(row["mean_pair_loss"]) in (int, float) and math.isfinite(float(row["mean_pair_loss"])) and float(row["mean_pair_loss"]) >= 0, "loss is nonfinite/negative")
        require(type(row["gradient_norm_before_clip"]) in (int, float) and math.isfinite(float(row["gradient_norm_before_clip"])) and float(row["gradient_norm_before_clip"]) >= 0, "gradient norm is nonfinite/negative")
        require(row["pair_count"] == schedule["updates"][index]["pair_count"], "per-update pair count drift")


def equal_state(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(
        isinstance(left[name], torch.Tensor)
        and isinstance(right[name], torch.Tensor)
        and torch.equal(left[name], right[name])
        for name in left
    )


def validate_checkpoint_and_resume(
    output_dir: Path, result: Mapping[str, Any], protocol_sha: str, authority_sha: str,
    loss_trace: Sequence[Mapping[str, Any]], pair_count: int, final_chunk: Mapping[str, Any]
) -> tuple[Path, Path]:
    checkpoint_path = output_dir / str(result["checkpoint"])
    resume_path = output_dir / str(result["completed_resume_state"])
    require(
        checkpoint_path.resolve().parent == output_dir.resolve()
        and checkpoint_path.name == "checkpoint_update2048.pt"
        and checkpoint_path.is_file(),
        "checkpoint path drift",
    )
    require(
        str(result["completed_resume_state"]) == final_chunk["resume_state"]
        and resume_path.resolve()
        == (output_dir / str(final_chunk["resume_state"])).resolve()
        and output_dir.resolve() in resume_path.resolve().parents
        and resume_path.is_file(),
        "completed resume path is not the final immutable chunk state",
    )
    require(not (output_dir / "resume_state.pt").exists(), "legacy mutable resume state remains after completion")
    require(file_sha256(checkpoint_path) == result["checkpoint_sha256"], "checkpoint file digest drift")
    require(file_sha256(resume_path) == result["completed_resume_state_sha256"], "resume file digest drift")
    require(
        final_chunk["resume_state_sha256"]
        == result["completed_resume_state_sha256"],
        "final chunk/resume digest drift",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_fields = {
        "schema_version", "arm", "outer_fold", "update", "seed", "protocol_sha256",
        "authority_sha256", "initial_state_sha256", "final_state_sha256", "model_state_dict",
    }
    require(isinstance(checkpoint, dict), "checkpoint is not an object")
    exact_fields(checkpoint, checkpoint_fields, "checkpoint")
    require(checkpoint["schema_version"] == "rc_dino_rcde_r1_main_checkpoint_v1_7", "checkpoint schema drift")
    require(checkpoint["arm"] == "RCDE_BAG" and checkpoint["outer_fold"] == 1, "checkpoint scope drift")
    require(checkpoint["update"] == UPDATES and checkpoint["seed"] == SEED, "checkpoint update/seed drift")
    require(checkpoint["protocol_sha256"] == protocol_sha and checkpoint["authority_sha256"] == authority_sha, "checkpoint binding drift")

    # Replay the runner's frozen seed contract before constructing the initial
    # model.  Validation must not depend on its incidental RNG position (or on
    # the current initializer happening to be seed-independent).
    torch.manual_seed(SEED)
    fresh = DINO_RCDE_V1_2()
    fresh_state = fresh.state_dict()
    initial_sha = state_dict_sha256(fresh_state)
    final_state = checkpoint["model_state_dict"]
    require(isinstance(final_state, dict), "checkpoint model state absent")
    require(set(final_state) == set(fresh_state), "checkpoint state key drift")
    require(sum(int(value.numel()) for value in final_state.values()) == EXPECTED_PARAMETERS, "checkpoint parameter count drift")
    final_sha = state_dict_sha256(final_state)
    require(checkpoint["initial_state_sha256"] == result["initial_state_sha256"] == initial_sha, "initial state binding drift")
    require(checkpoint["final_state_sha256"] == result["final_state_sha256"] == final_sha, "final state binding drift")
    require(final_sha != initial_sha, "training did not change model state")

    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    resume_fields = {
        "schema_version", "arm", "outer_fold", "completed_updates", "protocol_sha256",
        "authority_sha256", "initial_state_sha256", "model_state_dict", "optimizer_state_dict",
        "loss_trace", "pair_count", "python_random_state", "numpy_random_state",
        "torch_cpu_rng_state", "torch_cuda_rng_state_all",
    }
    require(isinstance(resume, dict), "resume state is not an object")
    exact_fields(resume, resume_fields, "completed resume state")
    require(resume["schema_version"] == "rc_dino_rcde_r1_main_resume_state_v1_7", "resume schema drift")
    require(resume["arm"] == "RCDE_BAG" and resume["outer_fold"] == 1, "resume scope drift")
    require(resume["completed_updates"] == UPDATES, "resume update drift")
    require(resume["protocol_sha256"] == protocol_sha and resume["authority_sha256"] == authority_sha, "resume binding drift")
    require(resume["initial_state_sha256"] == initial_sha, "resume initial state drift")
    require(equal_state(resume["model_state_dict"], final_state), "resume/checkpoint model states differ")
    require(resume["loss_trace"] == list(loss_trace) and resume["pair_count"] == pair_count, "resume/result training trace drift")
    require(isinstance(resume["torch_cpu_rng_state"], torch.Tensor), "CPU RNG state absent")
    require(isinstance(resume["torch_cuda_rng_state_all"], list), "CUDA RNG state list absent")
    require(isinstance(resume["numpy_random_state"], tuple), "NumPy RNG state absent")

    optimizer = resume["optimizer_state_dict"]
    require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"}, "optimizer state field drift")
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
    output_dir: Path, protocol: Mapping[str, Any], protocol_sha: str, authority_sha: str,
    identity_path: Path, manifest_path: Path, manifest: Mapping[str, Any],
    population: Mapping[str, Any], schedule: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], Path, Path]:
    result_path = output_dir / "train_result.json"
    result = read_json(result_path)
    fields = {
        "schema_version", "status", "claim_level", "arm", "outer_fold", "authority_sha256",
        "protocol_sha256", "run_identity_sha256", "signed_reference_access_manifest_sha256",
        "checkpoint", "checkpoint_sha256", "completed_resume_state", "completed_resume_state_sha256",
        "initial_state_sha256", "final_state_sha256", "parameter_count", "training",
        "bag_permutation", "execution_chain", "runtime_access", "protected_access_counts",
        "automatic_stage_advance", "scientific_GO_or_NO_GO", "next_required_stage",
    }
    exact_fields(result, fields, "train result")
    require(result["schema_version"] == "rc_dino_rcde_r1_main_train_result_v1_7", "result schema drift")
    require(result["status"] == "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAIN_COMPLETE", "result status drift")
    require(result["claim_level"] == "engineering_training_complete_not_scientific_GO_or_NO_GO", "result claim drift")
    require(result["arm"] == "RCDE_BAG" and result["outer_fold"] == 1, "result scope drift")
    require(result["protocol_sha256"] == protocol_sha and result["authority_sha256"] == authority_sha, "result binding drift")
    require(result["run_identity_sha256"] == file_sha256(identity_path), "result identity binding drift")
    require(result["signed_reference_access_manifest_sha256"] == file_sha256(manifest_path), "result manifest binding drift")
    require(result["parameter_count"] == EXPECTED_PARAMETERS, "result parameter count drift")
    protected_zero_exact(result["protected_access_counts"], "train result")
    require(result["automatic_stage_advance"] is False and result["scientific_GO_or_NO_GO"] is None, "result claim boundary drift")
    require(result["next_required_stage"] == "R1_MAIN_TRAINING_VALIDATION", "result next-stage drift")

    training = result["training"]
    training_fields = {
        "seed", "update_count", "episodes_per_update", "raw_correct_per_update",
        "raw_wrong_per_update", "pair_count", "correct_pool_count", "wrong_pool_count",
        "query_order_namespace", "query_order_key", "candidate_direction_source",
        "query_order_tie_break", "cache_query_key", "schedule_sha256",
        "query_tile_rows", "reference_tile_rows", "loss_trace",
    }
    exact_fields(training, training_fields, "result training")
    expected_scalars = {
        "seed": SEED, "update_count": UPDATES, "episodes_per_update": EPISODES_PER_UPDATE,
        "raw_correct_per_update": 2, "raw_wrong_per_update": 2,
        "pair_count": schedule["pair_count"], "correct_pool_count": len(population["correct"]),
        "wrong_pool_count": len(population["wrong"]), "query_order_namespace": QUERY_ORDER_NAMESPACE,
        "query_order_key": "source_image_sha256", "query_order_tie_break": "stable_episode_ledger_order",
        "candidate_direction_source": "P0_frozen_model_candidate_order",
        "cache_query_key": "execution_ordinal", "schedule_sha256": schedule["sha256"],
        "query_tile_rows": QUERY_TILE_ROWS, "reference_tile_rows": REFERENCE_TILE_ROWS,
    }
    for key, expected in expected_scalars.items():
        require(training[key] == expected, f"result training drift: {key}")
    validate_loss_trace(training["loss_trace"], schedule)

    bag = result["bag_permutation"]
    expected_bag = {
        "enabled": True,
        "valid_tokens_only": True,
        "fixed_point_free_cyclic_shift": True,
        "query_binding_namespace": "DINO_RCDE_R1_BAG_QUERY_PERMUTATION_V1_5",
        "reference_binding_namespace": "DINO_RCDE_R1_BAG_REFERENCE_PERMUTATION_V1_5",
    }
    require(bag == expected_bag, "BAG permutation receipt drift")

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
                "sha256": file_sha256(chunk_path),
                "start_update": int(chunk["start_update"]),
                "completed_updates": int(chunk["completed_updates"]),
                "restored_from_resume": bool(chunk["restored_from_resume"]),
                "restored_resume_state": chunk["restored_resume_state"],
                "restored_resume_state_sha256": chunk[
                    "restored_resume_state_sha256"
                ],
                "resume_state": str(chunk["resume_state"]),
                "resume_state_sha256": str(chunk["resume_state_sha256"]),
                "runtime_read_ledger": str(chunk["runtime_read_ledger"]),
                "runtime_read_ledger_sha256": str(chunk["runtime_read_ledger_sha256"]),
            }
        )
    require(chain["chunks"] == expected_chunk_bindings, "result chunk bindings drift")

    query_union = sorted(
        {
            int(row["ordinal"])
            for chunk in chunks
            for row in read_json(output_dir / str(chunk["runtime_read_ledger"]))["query_rows"]
        }
    )
    reference_union = sorted(
        {
            int(row["ordinal"])
            for chunk in chunks
            for row in read_json(output_dir / str(chunk["runtime_read_ledger"]))["reference_rows"]
        }
    )
    expected_runtime = {
        "query_read_union": query_union,
        "query_read_union_sha256": canonical_sha256(query_union),
        "reference_read_union": reference_union,
        "reference_read_union_sha256": canonical_sha256(reference_union),
        "reference_read_union_equals_signed_allowlist": reference_union == population["allowed_references"],
        "heldout_reference_intersection_count": 0,
    }
    require(result["runtime_access"] == expected_runtime, "result runtime-access receipt drift")
    require(query_union == population["eligible_queries"], "completed schedule did not read every eligible query")
    require(reference_union == population["allowed_references"], "completed schedule did not match reference allowlist")
    require(not set(reference_union).intersection(population["heldout_references"]), "runtime read heldout reference")

    checkpoint_path, resume_path = validate_checkpoint_and_resume(
        output_dir, result, protocol_sha, authority_sha, training["loss_trace"],
        training["pair_count"], chunks[-1],
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
    require(ROOT in validation_output.parents, "validation output escapes RC root")
    require(output_dir.is_dir(), "completed output directory absent")
    require(not validation_output.exists(), "immutable validation output already exists")

    protocol, authority, binding_hashes = validate_protocol(protocol_path, authority_path)
    cache_root = (
        args.cache_root.resolve()
        if args.cache_root is not None
        else resolve_bound(str(protocol["cache"]["root"]))
    )
    require(ROOT in cache_root.parents, "cache root escapes RC root")
    if "root" in protocol["cache"]:
        require(cache_root == resolve_bound(str(protocol["cache"]["root"])), "cache root argument drift")
    protocol_sha = file_sha256(protocol_path)
    authority_sha = file_sha256(authority_path)

    population = build_population(protocol, fold=1)
    index = cache_index(cache_root, set(population["reference_by_row"]))
    validate_cache_payloads(index, population)
    manifest_path = output_dir / "signed_reference_access_manifest.json"
    manifest = read_json(manifest_path)
    expected = expected_manifest(population, fold=1)
    require(manifest == expected, "signed reference access manifest drift")
    protected_zero_exact(manifest["protected_access_counts"], "signed access manifest")
    identity_path = output_dir / "run_identity.json"
    validate_identity(identity_path, protocol_sha, authority_sha)
    schedule = expected_schedule(population, fold=1)
    chunks = validate_chunks(
        output_dir, protocol, protocol_sha, authority_sha, manifest,
        file_sha256(manifest_path), population, schedule,
    )
    result, checkpoint_path, resume_path = validate_result(
        output_dir, protocol, protocol_sha, authority_sha, identity_path,
        manifest_path, manifest, population, schedule, chunks,
    )

    # The committed namespace is exact.  Per-job artifacts exist only inside
    # an atomically renamed four-file chunk directory, so a crashed staging
    # attempt outside this directory cannot masquerade as committed progress.
    require(not list(output_dir.glob("*.tmp*")), "temporary artifacts remain in result directory")
    expected_top_level = {
        "run_identity.json",
        "signed_reference_access_manifest.json",
        "checkpoint_update2048.pt",
        "train_result.json",
        *(f"chunk_job{chunk['job_id']}" for chunk in chunks),
    }
    require(
        {path.name for path in output_dir.iterdir()} == expected_top_level,
        "formal result namespace contains missing or uncommitted artifacts",
    )

    checks = {
        "protocol_authority_and_all_bindings": True,
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
        "all_chunk_ranges_contiguous_and_bounded": True,
        "atomic_chunk_directory_transactions_exact": True,
        "finalization_bound_to_last_committed_completed_state": True,
        "all_preflight_receipts_exact": True,
        "all_runtime_read_ledgers_match_schedule": True,
        "runtime_reference_union_equals_allowlist": True,
        "checkpoint_and_completed_resume_bound": True,
        "optimizer_completed_2048_updates": True,
        "exact_protected_zero_key_sets": True,
        "no_scientific_decision_or_automatic_advance": True,
    }
    validation = {
        "schema_version": "rc_dino_rcde_r1_main_train_validation_v1_7_contract_repair_20260813",
        "status": "DINO_RCDE_R1_MAIN_ARM_FOLD_VALIDATION_PASS",
        "claim_level": "engineering_training_validation_only_not_scientific_GO_or_NO_GO",
        "arm": "RCDE_BAG",
        "outer_fold": 1,
        "checks": checks,
        "protocol_sha256": protocol_sha,
        "authority_sha256": authority_sha,
        "binding_hashes": binding_hashes,
        "run_identity_sha256": file_sha256(identity_path),
        "signed_reference_access_manifest_sha256": file_sha256(manifest_path),
        "train_result_sha256": file_sha256(output_dir / "train_result.json"),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "completed_resume_state_sha256": file_sha256(resume_path),
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
        "next_required_stage": "R1_MAIN_BAG_FOLD2_AUTHORITY_UPDATE_REQUIRED",
    }
    atomic_json_no_clobber(validation_output, validation)
    print(json.dumps({
        "status": validation["status"],
        "validation_output": str(validation_output),
        "validation_sha256": file_sha256(validation_output),
        "chunk_count": len(chunks),
        "update_count": UPDATES,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
