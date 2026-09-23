#!/usr/bin/env python3
"""Contract-faithful, time-bounded R1 single-scope training.

V1.8 preserves V1.7's scientific and atomic-transaction logic, while removing
the BAG/fold-1 execution hard-coding.  Exactly one arm and one outer fold are
frozen jointly by the protocol and its bound authority; the CLI must match that
scope byte-for-byte before any cache read or output creation.  Artifact schemas
and statuses are likewise supplied by an exact protocol artifact contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import run_dino_rcde_p0_resource_v1_2 as cache_base
import run_dino_rcde_r1_main_train_v1_5 as base
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rc_dino_rcde_r1_main_train_protocol_v1_8_single_scope_20260814"
ARTIFACT_CONTRACT_KEYS = frozenset(
    {
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
)
QUERY_ORDER_NAMESPACE = "RCDE_QUERY_ORDER_V1_1"
EXPECTED_CACHE_MODEL_SHA256 = (
    "f901d9bd056bb65e5fcb02c72e869f6d0cf8d7472467d22fbc340442feca034c"
)
PROTECTED_ZERO_KEYS = (
    "C8_runtime_read_count",
    "S8_runtime_read_count",
    "opened_runtime_read_count",
    "sealed_runtime_read_count",
    "unauthorized_natural_result_read_count",
    "home_files_modified",
)


class ContractRepairAbort(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def protected_zeros() -> dict[str, int]:
    return {key: 0 for key in PROTECTED_ZERO_KEYS}


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ContractRepairAbort(f"immutable JSON output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ContractRepairAbort(f"immutable torch output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_contract(protocol: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Validate and return the exact V1.8 artifact schema/status contract."""

    contract = protocol.get("artifact_contract")
    if not isinstance(contract, dict) or set(contract) != ARTIFACT_CONTRACT_KEYS:
        raise ContractRepairAbort("V1.8 artifact contract key set is not exact")
    normalized: dict[str, dict[str, str]] = {}
    for name in sorted(ARTIFACT_CONTRACT_KEYS):
        spec = contract.get(name)
        if not isinstance(spec, dict) or set(spec) != {"schema_version", "status"}:
            raise ContractRepairAbort(f"V1.8 artifact contract field drift: {name}")
        schema_version = spec.get("schema_version")
        status = spec.get("status")
        if (
            not isinstance(schema_version, str)
            or not schema_version
            or "v1_8" not in schema_version
            or not isinstance(status, str)
            or not status
        ):
            raise ContractRepairAbort(f"V1.8 artifact contract value drift: {name}")
        normalized[name] = {
            "schema_version": schema_version,
            "status": status,
        }
    return normalized


def single_scope(
    protocol: Mapping[str, Any], authority: Mapping[str, Any]
) -> tuple[str, int]:
    """Resolve one arm/fold jointly frozen by protocol and authority."""

    training = protocol.get("training")
    if not isinstance(training, dict):
        raise ContractRepairAbort("V1.8 training contract is absent")
    arms = training.get("arms")
    folds = training.get("outer_folds")
    if not isinstance(arms, list) or len(arms) != 1 or arms[0] not in base.ALLOWED_ARMS:
        raise ContractRepairAbort("V1.8 protocol must freeze exactly one allowed arm")
    if (
        not isinstance(folds, list)
        or len(folds) != 1
        or isinstance(folds[0], bool)
        or not isinstance(folds[0], int)
        or folds[0] not in (1, 2, 3, 4)
    ):
        raise ContractRepairAbort("V1.8 protocol must freeze exactly one outer fold")
    arm = str(arms[0])
    fold = int(folds[0])
    stage = protocol.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ContractRepairAbort("V1.8 protocol stage is absent")
    scopes = authority.get("authorized_scopes")
    if not isinstance(scopes, list) or not scopes:
        raise ContractRepairAbort("V1.8 authority scope allowlist is absent")
    scope_keys: list[tuple[object, object, object]] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            raise ContractRepairAbort("V1.8 authority scope entry is not a mapping")
        scope_arm = scope.get("arm")
        scope_fold = scope.get("outer_fold")
        scope_stage = scope.get("stage")
        if (
            scope_arm not in base.ALLOWED_ARMS
            or isinstance(scope_fold, bool)
            or not isinstance(scope_fold, int)
            or scope_fold not in (1, 2, 3, 4)
            or not isinstance(scope_stage, str)
            or not scope_stage
        ):
            raise ContractRepairAbort("V1.8 authority scope value drift")
        scope_keys.append((scope_arm, scope_fold, scope_stage))
    if len(scope_keys) != len(set(scope_keys)):
        raise ContractRepairAbort("V1.8 authority contains duplicate scopes")
    if (arm, fold, stage) not in scope_keys:
        raise ContractRepairAbort("V1.8 protocol single scope is not authorized")
    if authority.get("next_authorized_stage") != stage:
        raise ContractRepairAbort("V1.8 protocol/authority stage mismatch")
    return arm, fold


def validate_contract(
    protocol_path: Path, authority_path: Path
) -> tuple[dict[str, Any], str, int]:
    protocol = base.read_json(protocol_path)
    authority = base.read_json(authority_path)
    if protocol.get("schema_version") != SCHEMA:
        raise ContractRepairAbort("V1.8 protocol schema drift")
    expected = protocol.get("authority", {})
    if authority_path.resolve() != base.resolve_bound(str(expected.get("path", ""))):
        raise ContractRepairAbort("V1.8 authority path drift")
    if base.file_sha256(authority_path) != expected.get("sha256"):
        raise ContractRepairAbort("V1.8 authority digest drift")
    if authority.get("status") != expected.get("required_status"):
        raise ContractRepairAbort("V1.8 authority status drift")
    if not isinstance(authority.get("status"), str) or not authority.get("status"):
        raise ContractRepairAbort("V1.8 authority status is not authorized")
    if authority.get("natural_training_authorized") is not True:
        raise ContractRepairAbort("V1.8 natural training is not authorized")
    if (
        protocol.get("automatic_stage_advance") is not False
        or authority.get("automatic_stage_advance") is not False
    ):
        raise ContractRepairAbort("V1.8 automatic advancement must remain disabled")
    if (
        protocol.get("scientific_GO_or_NO_GO") is not None
        or authority.get("scientific_GO_or_NO_GO") is not None
    ):
        raise ContractRepairAbort("V1.8 pretraining authority contains a science verdict")
    for name, binding in protocol.get("bindings", {}).items():
        path = base.resolve_bound(str(binding["path"]))
        if not path.is_file() or base.file_sha256(path) != binding["sha256"]:
            raise ContractRepairAbort(f"V1.8 binding drift: {name}")
    cache_validation = base.read_json(
        base.resolve_bound(protocol["bindings"]["cache_validation"]["path"])
    )
    if cache_validation.get("status") != "DINO_RCDE_R1_CACHE_VALIDATION_PASS":
        raise ContractRepairAbort("canonical cache validation is not PASS")
    if protocol.get("cache", {}).get("model_checkpoint_logical_sha256") != (
        EXPECTED_CACHE_MODEL_SHA256
    ):
        raise ContractRepairAbort("V1.8 cache model identity drift")
    artifact_contract(protocol)
    arm, fold = single_scope(protocol, authority)
    return protocol, arm, fold


def require_cli_scope(
    cli_arm: str,
    cli_fold: int,
    authorized_arm: str,
    authorized_fold: int,
) -> None:
    if cli_arm != authorized_arm or cli_fold != authorized_fold:
        raise ContractRepairAbort(
            "CLI arm/fold does not exactly match the frozen V1.8 single scope"
        )


def _fold(value: object) -> int:
    return base.fold_number(value)


def _candidate_rows(episode: Mapping[str, Any]) -> list[int]:
    return [int(episode["target_physical_row"])] + [
        int(item["physical_row"]) for item in episode.get("selected_negatives", [])
    ]


def frozen_pair_directions(episode: Mapping[str, Any]) -> list[bool]:
    """Recover target/rival slot direction from P0's frozen candidate order."""

    expected = _candidate_rows(episode)
    order = [int(item["physical_row"]) for item in episode.get("model_candidate_order", [])]
    if len(expected) != len(set(expected)) or len(order) != len(set(order)):
        raise ContractRepairAbort("episode candidate rows are not unique")
    if len(order) != len(expected) or set(order) != set(expected):
        raise ContractRepairAbort("P0 frozen candidate order is not the episode row set")
    positions = {row: index for index, row in enumerate(order)}
    target = int(episode["target_physical_row"])
    return [positions[target] < positions[row] for row in expected[1:]]


def schedule_event(
    fold: int,
    update_one_based: int,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    queries = []
    for episode in episodes:
        rivals = [int(item["physical_row"]) for item in episode["selected_negatives"]]
        order = [int(item["physical_row"]) for item in episode["model_candidate_order"]]
        queries.append(
            {
                "query_id": str(episode["query_id"]),
                "query_ordinal": int(episode["query_ordinal"]),
                "execution_ordinal": int(episode["execution_ordinal"]),
                "source_image_sha256": str(episode["_source_image_sha256"]),
                "target_physical_row": int(episode["target_physical_row"]),
                "rivals": rivals,
                "target_first": frozen_pair_directions(episode),
                "frozen_model_candidate_rows": order,
                "frozen_model_candidate_rows_sha256": canonical_sha256(order),
            }
        )
    return {"outer_fold": fold, "update": update_one_based, "queries": queries}


def ordered_pools(
    ledger: Mapping[str, Any],
    roles: Mapping[str, Any],
    shape: Mapping[str, Any],
    fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role_rows = roles.get("records", roles.get("rows", []))
    role_by_id = {str(row["query_id"]): row for row in role_rows}
    shape_rows = [row for row in shape.get("rows", []) if row.get("kind") == "query"]
    shape_by_id = {str(row["id"]): row for row in shape_rows}
    if len(role_by_id) != 600 or len(shape_by_id) != 600:
        raise ContractRepairAbort("600-query role/shape population drift")
    correct: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    for source in ledger.get("episodes", []):
        if _fold(source.get("heldout_fold")) != fold or source.get(
            "pair_loss_eligible"
        ) is not True:
            continue
        query_id = str(source["query_id"])
        role = role_by_id.get(query_id)
        row = shape_by_id.get(query_id)
        if role is None or row is None or _fold(role["inner_fold"]) == fold:
            raise ContractRepairAbort("outer-train query mapping drift")
        if int(source["query_ordinal"]) != int(role["query_ordinal"]) or int(
            source["query_ordinal"]
        ) != int(row["query_ordinal"]):
            raise ContractRepairAbort("historical query ordinal mapping drift")
        if int(source["execution_ordinal"]) != int(row["execution_ordinal"]):
            raise ContractRepairAbort("cache execution ordinal mapping drift")
        episode = dict(source)
        episode["_source_image_sha256"] = str(row["source_image_sha256"])
        frozen_pair_directions(episode)
        (correct if episode.get("full_gallery_RAW_correct") is True else wrong).append(
            episode
        )
    key = lambda row: base.hash_parts(
        QUERY_ORDER_NAMESPACE,
        fold,
        row["_source_image_sha256"],
    )
    correct.sort(key=key)
    wrong.sort(key=key)
    if len(correct) < 2 or len(wrong) < 2:
        raise ContractRepairAbort("RAW-correct/RAW-wrong pool is incomplete")
    return correct, wrong


def _validate_payload(
    path: Path,
    *,
    source_sha256: str,
    model_sha256: str,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("source_image_sha256") != source_sha256:
        raise ContractRepairAbort(f"cache source hash drift: {path.name}")
    if payload.get("model_checkpoint_logical_sha256") != model_sha256:
        raise ContractRepairAbort(f"cache model hash drift: {path.name}")
    if payload.get("logical_sha256") != cache_base._cache_logical_hash(payload):
        raise ContractRepairAbort(f"cache logical hash drift: {path.name}")
    nested = payload.get("cache_receipt")
    if not isinstance(nested, dict) or nested.get("model_receipt_sha256") != model_sha256:
        raise ContractRepairAbort(f"nested cache model hash drift: {path.name}")
    tokens = payload.get("tokens_fp16")
    mask = payload.get("valid_patch_mask")
    if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.float16:
        raise ContractRepairAbort(f"cache token dtype drift: {path.name}")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ContractRepairAbort(f"cache mask dtype drift: {path.name}")


def input_closure(
    protocol: Mapping[str, Any],
    cache_root: Path,
    fold: int,
) -> dict[str, Any]:
    artifacts = artifact_contract(protocol)
    roles = base.read_json(base.resolve_bound(protocol["bindings"]["training_roles"]["path"]))
    ledger = base.read_json(base.resolve_bound(protocol["bindings"]["episode_ledger"]["path"]))
    shape = base.read_json(base.resolve_bound(protocol["bindings"]["resource_shape_ledger"]["path"]))
    role_rows = roles.get("records", roles.get("rows", []))
    query_rows = [row for row in shape.get("rows", []) if row.get("kind") == "query"]
    reference_rows = [row for row in shape.get("rows", []) if row.get("kind") == "reference"]
    if len(role_rows) != 600 or len(query_rows) != 600 or len(reference_rows) != 4_748:
        raise ContractRepairAbort("role/shape cardinality drift")
    role_by_id = {str(row["query_id"]): row for row in role_rows}
    query_by_id = {str(row["id"]): row for row in query_rows}
    query_by_execution = {int(row["execution_ordinal"]): row for row in query_rows}
    reference_by_row = {int(row["physical_row"]): row for row in reference_rows}
    if len(role_by_id) != 600 or len(query_by_id) != 600:
        raise ContractRepairAbort("duplicate query ID in role/shape ledger")
    if sorted(query_by_execution) != list(range(600)):
        raise ContractRepairAbort("cache execution ordinals are not dense 0..599")
    if len({int(row["query_ordinal"]) for row in query_rows}) != 600:
        raise ContractRepairAbort("historical query ordinals are not unique")
    if len(reference_by_row) != 4_748:
        raise ContractRepairAbort("duplicate reference row in shape ledger")

    all_episode_mapping_count = 0
    for episode in ledger.get("episodes", []):
        query_id = str(episode["query_id"])
        row = query_by_id.get(query_id)
        role = role_by_id.get(query_id)
        if row is None or role is None:
            raise ContractRepairAbort("episode query absent from role/shape ledgers")
        if (
            int(episode["execution_ordinal"]) != int(row["execution_ordinal"])
            or int(episode["query_ordinal"]) != int(row["query_ordinal"])
            or int(role["query_ordinal"]) != int(row["query_ordinal"])
        ):
            raise ContractRepairAbort("episode query namespace closure failed")
        frozen_pair_directions(episode)
        for reference_row in _candidate_rows(episode):
            if reference_row not in reference_by_row:
                raise ContractRepairAbort("episode reference absent from cache shape ledger")
        all_episode_mapping_count += 1
    if all_episode_mapping_count != 1_800:
        raise ContractRepairAbort("episode population drift")

    correct, wrong = ordered_pools(ledger, roles, shape, fold)
    eligible = correct + wrong
    eligible_queries = sorted({int(row["execution_ordinal"]) for row in eligible})
    allowed_references = sorted(
        {reference for episode in eligible for reference in _candidate_rows(episode)}
    )
    fold_receipt = ledger.get("folds", {}).get(str(fold), {})
    if (
        len(allowed_references) != int(fold_receipt.get("allowed_physical_row_count", -1))
        or canonical_sha256(allowed_references)
        != fold_receipt.get("allowed_physical_rows_sha256")
    ):
        raise ContractRepairAbort("derived reference allowlist does not match P0 receipt")

    heldout_ids = {
        str(row["query_id"]) for row in role_rows if _fold(row["inner_fold"]) == fold
    }
    target_by_query: dict[str, int] = {}
    for episode in ledger.get("episodes", []):
        query_id = str(episode["query_id"])
        target = int(episode["target_physical_row"])
        prior = target_by_query.setdefault(query_id, target)
        if prior != target:
            raise ContractRepairAbort("query target row changes across fold receipts")
    heldout_references = sorted({target_by_query[query_id] for query_id in heldout_ids})
    if set(allowed_references) & set(heldout_references):
        raise ContractRepairAbort("outer-train and heldout reference rows overlap")

    index = base.cache_index(cache_root)
    expected_query_keys = {("query", row) for row in range(600)}
    expected_reference_keys = {("reference", row) for row in reference_by_row}
    if set(index) != expected_query_keys | expected_reference_keys:
        raise ContractRepairAbort("canonical cache key set is not exact")
    model_sha = str(protocol["cache"]["model_checkpoint_logical_sha256"])
    for execution in eligible_queries:
        row = query_by_execution[execution]
        _validate_payload(
            index[("query", execution)],
            source_sha256=str(row["source_image_sha256"]),
            model_sha256=model_sha,
        )
    for physical_row in allowed_references:
        row = reference_by_row[physical_row]
        _validate_payload(
            index[("reference", physical_row)],
            source_sha256=str(row["source_image_sha256"]),
            model_sha256=model_sha,
        )

    manifest = {
        **artifacts["manifest"],
        "outer_fold": fold,
        "query_order_namespace": QUERY_ORDER_NAMESPACE,
        "query_cache_key_field": "execution_ordinal",
        "historical_query_ordinal_model_visible": False,
        "eligible_query_execution_ordinals": eligible_queries,
        "eligible_query_execution_ordinals_sha256": canonical_sha256(eligible_queries),
        "allowed_reference_physical_rows": allowed_references,
        "allowed_reference_physical_rows_sha256": canonical_sha256(allowed_references),
        "heldout_reference_physical_rows": heldout_references,
        "heldout_reference_physical_rows_sha256": canonical_sha256(heldout_references),
        "reference_row_intersection_count": 0,
        "eligible_query_count": len(eligible_queries),
        "allowed_reference_count": len(allowed_references),
        "heldout_reference_count": len(heldout_references),
        "all_episode_mapping_count": all_episode_mapping_count,
        "correct_pool_count": len(correct),
        "wrong_pool_count": len(wrong),
        "protected_access_counts": protected_zeros(),
    }
    manifest["logical_sha256"] = canonical_sha256(manifest)
    preflight = {
        **artifacts["preflight"],
        "outer_fold": fold,
        "all_episode_mapping_count": all_episode_mapping_count,
        "eligible_query_payloads_independently_validated": len(eligible_queries),
        "allowed_reference_payloads_independently_validated": len(allowed_references),
        "cache_query_key_count": 600,
        "cache_reference_key_count": 4_748,
        "cache_execution_ordinal_range": [0, 599],
        "historical_query_ordinal_range": [
            min(int(row["query_ordinal"]) for row in query_rows),
            max(int(row["query_ordinal"]) for row in query_rows),
        ],
        "known_regression_mapping": {
            "query_id": "OUTCOME-0504",
            "query_ordinal": int(query_by_id["OUTCOME-0504"]["query_ordinal"]),
            "execution_ordinal": int(query_by_id["OUTCOME-0504"]["execution_ordinal"]),
        },
        "access_manifest_logical_sha256": manifest["logical_sha256"],
        "protected_access_counts": protected_zeros(),
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
    }
    return {
        "roles": roles,
        "ledger": ledger,
        "shape": shape,
        "correct": correct,
        "wrong": wrong,
        "index": index,
        "query_by_execution": query_by_execution,
        "reference_by_row": reference_by_row,
        "manifest": manifest,
        "preflight": preflight,
    }


class CacheAccessTracker:
    def __init__(
        self,
        *,
        index: Mapping[tuple[str, int], Path],
        query_by_execution: Mapping[int, Mapping[str, Any]],
        reference_by_row: Mapping[int, Mapping[str, Any]],
        allowed_queries: Sequence[int],
        allowed_references: Sequence[int],
        model_sha256: str,
    ) -> None:
        self.index = index
        self.query_by_execution = query_by_execution
        self.reference_by_row = reference_by_row
        self.allowed_queries = set(map(int, allowed_queries))
        self.allowed_references = set(map(int, allowed_references))
        self.model_sha256 = model_sha256
        self.resident: dict[tuple[str, int], Mapping[str, Any]] = {}
        self.semantic_counts: Counter[tuple[str, int]] = Counter()
        self.file_open_counts: Counter[tuple[str, int]] = Counter()

    def load(self, kind: str, ordinal: int) -> Mapping[str, Any]:
        ordinal = int(ordinal)
        if kind == "query":
            if ordinal not in self.allowed_queries:
                raise ContractRepairAbort("query cache read outside eligible outer-train set")
            source = self.query_by_execution[ordinal]
        elif kind == "reference":
            if ordinal not in self.allowed_references:
                raise ContractRepairAbort("reference cache read outside signed allowlist")
            source = self.reference_by_row[ordinal]
        else:
            raise ContractRepairAbort("unknown cache payload kind")
        key = (kind, ordinal)
        self.semantic_counts[key] += 1
        if key not in self.resident:
            path = self.index.get(key)
            if path is None:
                raise ContractRepairAbort(f"missing canonical cache payload: {key}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                payload.get("source_image_sha256") != source["source_image_sha256"]
                or payload.get("model_checkpoint_logical_sha256") != self.model_sha256
                or payload.get("logical_sha256") != cache_base._cache_logical_hash(payload)
            ):
                raise ContractRepairAbort(f"runtime cache closure failed: {key}")
            self.resident[key] = payload
            self.file_open_counts[key] += 1
        return self.resident[key]

    def ledger(
        self,
        *,
        job_id: str,
        start_update: int,
        completed_updates: int,
        artifact_spec: Mapping[str, str],
    ) -> dict[str, Any]:
        query_rows = []
        reference_rows = []
        for (kind, ordinal), semantic_count in sorted(self.semantic_counts.items()):
            record = {
                "ordinal": ordinal,
                "semantic_read_count": int(semantic_count),
                "file_open_count": int(self.file_open_counts[(kind, ordinal)]),
                "cache_file": self.index[(kind, ordinal)].name,
            }
            (query_rows if kind == "query" else reference_rows).append(record)
        payload = {
            **artifact_spec,
            "job_id": job_id,
            "start_update": start_update,
            "completed_updates": completed_updates,
            "query_rows": query_rows,
            "reference_rows": reference_rows,
            "query_read_set_sha256": canonical_sha256([row["ordinal"] for row in query_rows]),
            "reference_read_set_sha256": canonical_sha256(
                [row["ordinal"] for row in reference_rows]
            ),
            "protected_access_counts": protected_zeros(),
        }
        payload["logical_sha256"] = canonical_sha256(payload)
        return payload


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def state_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    arm: str,
    fold: int,
    completed: int,
    loss_trace: list[dict[str, Any]],
    pair_count: int,
    initial_state_sha256: str,
    protocol_sha256: str,
    authority_sha256: str,
    artifact_spec: Mapping[str, str],
) -> dict[str, Any]:
    return {
        **artifact_spec,
        "arm": arm,
        "outer_fold": fold,
        "completed_updates": completed,
        "protocol_sha256": protocol_sha256,
        "authority_sha256": authority_sha256,
        "initial_state_sha256": initial_state_sha256,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_trace": loss_trace,
        "pair_count": pair_count,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def save_resume_no_clobber(path: Path, **kwargs: Any) -> None:
    atomic_torch_no_clobber(path, state_payload(**kwargs))


def schedule_prefix(
    correct: Sequence[dict[str, Any]],
    wrong: Sequence[dict[str, Any]],
    fold: int,
    completed: int,
) -> hashlib._Hash:
    digest = hashlib.sha256()
    for update_zero in range(completed):
        episodes = base.update_episodes(correct, wrong, update_zero)
        digest.update(
            base.json_bytes(schedule_event(fold, update_zero + 1, episodes)) + b"\n"
        )
    return digest


def run_update(
    *,
    model: DINO_RCDE_V1_2,
    optimizer: torch.optim.Optimizer,
    tracker: CacheAccessTracker,
    correct: Sequence[dict[str, Any]],
    wrong: Sequence[dict[str, Any]],
    fold: int,
    arm: str,
    update_zero: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    update_one = update_zero + 1
    episodes = base.update_episodes(correct, wrong, update_zero)
    event = schedule_event(fold, update_one, episodes)
    denominator = sum(len(row["selected_negatives"]) for row in episodes)
    if denominator < base.EPISODES_PER_UPDATE:
        raise ContractRepairAbort("empty update denominator")
    lr = base.learning_rate(update_one)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for episode, query_event in zip(episodes, event["queries"]):
        execution_ordinal = int(episode["execution_ordinal"])
        target_row = int(episode["target_physical_row"])
        rivals = list(query_event["rivals"])
        losses.extend(
            base.staged_episode_backward(
                model,
                tracker.load("query", execution_ordinal),
                tracker.load("reference", target_row),
                [tracker.load("reference", row) for row in rivals],
                list(query_event["target_first"]),
                denominator=denominator,
                device=device,
                arm=arm,
                fold=fold,
                query_id=str(episode["query_id"]),
                target_row=target_row,
                rival_rows=rivals,
            )
        )
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise ContractRepairAbort("nonfinite gradient norm")
    optimizer.step()
    record = {
        "update": update_one,
        "learning_rate": lr,
        "mean_pair_loss": sum(losses) / len(losses),
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "pair_count": denominator,
    }
    return event, record


CHUNK_DIRECTORY = re.compile(r"^chunk_job(.+)$")


def _artifact_path(output_dir: Path, value: object) -> Path:
    path = (output_dir / str(value)).resolve()
    if output_dir not in path.parents:
        raise ContractRepairAbort("chunk artifact escapes formal output directory")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _chunk_history(
    output_dir: Path, *, chunk_spec: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not output_dir.exists():
        return rows
    for directory in output_dir.glob("chunk_job*"):
        match = CHUNK_DIRECTORY.fullmatch(directory.name)
        if match is None or not directory.is_dir() or directory.is_symlink():
            raise ContractRepairAbort("invalid committed chunk directory")
        chunk_path = directory / "chunk.json"
        row = base.read_json(chunk_path)
        if str(row.get("job_id")) != match.group(1):
            raise ContractRepairAbort("chunk directory/job ID drift")
        expected_files = {
            "chunk.json",
            "input_preflight.json",
            "runtime_read_ledger.json",
            "resume_state.pt",
        }
        entries = list(directory.iterdir())
        observed_files = {path.name for path in entries}
        if (
            observed_files != expected_files
            or not all(path.is_file() and not path.is_symlink() for path in entries)
        ):
            raise ContractRepairAbort("committed chunk file set is not exact")
        row["_chunk_json_path"] = str(chunk_path.relative_to(output_dir))
        row["_chunk_json_sha256"] = base.file_sha256(chunk_path)
        rows.append(row)
    rows.sort(key=lambda row: (int(row["start_update"]), str(row["job_id"])))
    cursor = 0
    seen_jobs: set[str] = set()
    for row in rows:
        if (
            row.get("schema_version") != chunk_spec["schema_version"]
            or row.get("status") != chunk_spec["status"]
        ):
            raise ContractRepairAbort("prior chunk status drift")
        if int(row["start_update"]) != cursor or int(row["completed_updates"]) <= cursor:
            raise ContractRepairAbort("prior chunk coverage is non-contiguous")
        state_path = _artifact_path(output_dir, row.get("resume_state", ""))
        ledger_path = _artifact_path(output_dir, row.get("runtime_read_ledger", ""))
        preflight_path = _artifact_path(output_dir, row.get("input_preflight", ""))
        if (
            not state_path.is_file()
            or base.file_sha256(state_path) != row.get("resume_state_sha256")
            or not ledger_path.is_file()
            or base.file_sha256(ledger_path) != row.get("runtime_read_ledger_sha256")
            or not preflight_path.is_file()
            or base.file_sha256(preflight_path) != row.get("input_preflight_sha256")
        ):
            raise ContractRepairAbort("prior committed chunk binding drift")
        job_id = str(row["job_id"])
        if job_id in seen_jobs:
            raise ContractRepairAbort("duplicate prior chunk job ID")
        seen_jobs.add(job_id)
        cursor = int(row["completed_updates"])
    for index, row in enumerate(rows):
        if index == 0:
            if (
                row.get("restored_from_resume") is not False
                or row.get("restored_resume_state") is not None
                or row.get("restored_resume_state_sha256") is not None
            ):
                raise ContractRepairAbort("first chunk resume handoff drift")
        else:
            prior = rows[index - 1]
            if (
                row.get("restored_from_resume") is not True
                or row.get("restored_resume_state") != prior.get("resume_state")
                or row.get("restored_resume_state_sha256")
                != prior.get("resume_state_sha256")
            ):
                raise ContractRepairAbort("chunk resume handoff is not exact")
    return rows


def validate_resume_state(
    state: Mapping[str, Any],
    *,
    arm: str,
    fold: int,
    protocol_sha256: str,
    authority_sha256: str,
    initial_state_sha256: str,
    artifact_spec: Mapping[str, str],
) -> None:
    required = {
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
    if set(state) != required:
        raise ContractRepairAbort("resume-state field contract drift")
    if (
        state["schema_version"] != artifact_spec["schema_version"]
        or state["status"] != artifact_spec["status"]
        or state["arm"] != arm
        or int(state["outer_fold"]) != fold
        or state["protocol_sha256"] != protocol_sha256
        or state["authority_sha256"] != authority_sha256
        or state["initial_state_sha256"] != initial_state_sha256
    ):
        raise ContractRepairAbort("resume-state binding drift")


def finalize_training(
    *,
    output_dir: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    authority_sha256: str,
    identity_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    correct: Sequence[dict[str, Any]],
    wrong: Sequence[dict[str, Any]],
    initial_state_sha256: str,
    arm: str,
    fold: int,
) -> None:
    artifacts = artifact_contract(protocol)
    chunks = _chunk_history(output_dir, chunk_spec=artifacts["chunk"])
    if not chunks or int(chunks[-1]["completed_updates"]) != base.UPDATE_COUNT:
        raise ContractRepairAbort("finalization requested before update 2048 commit")
    if len(chunks) > int(protocol["execution"]["maximum_jobs_submitted_now"]):
        raise ContractRepairAbort("completed chain exceeds job budget")
    final_chunk = chunks[-1]
    completed_resume = _artifact_path(output_dir, final_chunk["resume_state"])
    state = torch.load(completed_resume, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ContractRepairAbort("completed resume state is not a mapping")
    validate_resume_state(
        state,
        arm=arm,
        fold=fold,
        protocol_sha256=protocol_sha256,
        authority_sha256=authority_sha256,
        initial_state_sha256=initial_state_sha256,
        artifact_spec=artifacts["resume"],
    )
    if int(state["completed_updates"]) != base.UPDATE_COUNT:
        raise ContractRepairAbort("completed resume state update drift")
    loss_trace = list(state["loss_trace"])
    pair_count = int(state["pair_count"])
    schedule_digest = schedule_prefix(correct, wrong, fold, base.UPDATE_COUNT).hexdigest()
    final_state = state["model_state_dict"]
    probe = DINO_RCDE_V1_2()
    probe.load_state_dict(final_state, strict=True)
    final_state_sha = base.state_dict_sha256(probe)
    checkpoint_path = output_dir / "checkpoint_update2048.pt"
    expected_checkpoint = {
        **artifacts["checkpoint"],
        "arm": arm,
        "outer_fold": fold,
        "update": base.UPDATE_COUNT,
        "seed": base.SEED,
        "protocol_sha256": protocol_sha256,
        "authority_sha256": authority_sha256,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha,
        "model_state_dict": final_state,
    }
    if checkpoint_path.exists():
        existing = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(existing, dict) or set(existing) != set(expected_checkpoint):
            raise ContractRepairAbort("existing recovery checkpoint field drift")
        metadata_keys = set(expected_checkpoint) - {"model_state_dict"}
        if any(existing[key] != expected_checkpoint[key] for key in metadata_keys):
            raise ContractRepairAbort("existing recovery checkpoint metadata drift")
        existing_state = existing["model_state_dict"]
        if set(existing_state) != set(final_state) or any(
            not torch.equal(existing_state[name], final_state[name]) for name in final_state
        ):
            raise ContractRepairAbort("existing recovery checkpoint state drift")
    else:
        atomic_torch_no_clobber(checkpoint_path, expected_checkpoint)

    chunk_bindings = []
    reference_union: set[int] = set()
    query_union: set[int] = set()
    for row in chunks:
        read_path = _artifact_path(output_dir, row["runtime_read_ledger"])
        read = base.read_json(read_path)
        reference_union.update(int(record["ordinal"]) for record in read["reference_rows"])
        query_union.update(int(record["ordinal"]) for record in read["query_rows"])
        chunk_bindings.append(
            {
                "path": str(row["_chunk_json_path"]),
                "sha256": str(row["_chunk_json_sha256"]),
                "start_update": int(row["start_update"]),
                "completed_updates": int(row["completed_updates"]),
                "restored_from_resume": bool(row["restored_from_resume"]),
                "restored_resume_state": row["restored_resume_state"],
                "restored_resume_state_sha256": row["restored_resume_state_sha256"],
                "resume_state": str(row["resume_state"]),
                "resume_state_sha256": str(row["resume_state_sha256"]),
                "runtime_read_ledger": str(row["runtime_read_ledger"]),
                "runtime_read_ledger_sha256": str(row["runtime_read_ledger_sha256"]),
            }
        )
    query_rows = sorted(query_union)
    reference_rows = sorted(reference_union)
    result = {
        **artifacts["result"],
        "claim_level": "engineering_training_complete_not_scientific_GO_or_NO_GO",
        "arm": arm,
        "outer_fold": fold,
        "authority_sha256": authority_sha256,
        "protocol_sha256": protocol_sha256,
        "run_identity_sha256": base.file_sha256(identity_path),
        "signed_reference_access_manifest_sha256": base.file_sha256(manifest_path),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": base.file_sha256(checkpoint_path),
        "completed_resume_state": str(completed_resume.relative_to(output_dir)),
        "completed_resume_state_sha256": base.file_sha256(completed_resume),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha,
        "parameter_count": base.EXPECTED_PARAMETER_COUNT,
        "training": {
            "seed": base.SEED,
            "update_count": base.UPDATE_COUNT,
            "episodes_per_update": base.EPISODES_PER_UPDATE,
            "raw_correct_per_update": 2,
            "raw_wrong_per_update": 2,
            "pair_count": pair_count,
            "correct_pool_count": len(correct),
            "wrong_pool_count": len(wrong),
            "query_order_namespace": QUERY_ORDER_NAMESPACE,
            "query_order_key": "source_image_sha256",
            "query_order_tie_break": "stable_episode_ledger_order",
            "candidate_direction_source": "P0_frozen_model_candidate_order",
            "cache_query_key": "execution_ordinal",
            "schedule_sha256": schedule_digest,
            "query_tile_rows": base.QUERY_TILE_ROWS,
            "reference_tile_rows": base.REFERENCE_TILE_ROWS,
            "loss_trace": loss_trace,
        },
        "bag_permutation": {
            "enabled": arm == "RCDE_BAG",
            "valid_tokens_only": True,
            "fixed_point_free_cyclic_shift": True,
            "query_binding_namespace": base.BAG_QUERY_NAMESPACE,
            "reference_binding_namespace": base.BAG_REFERENCE_NAMESPACE,
        },
        "execution_chain": {
            "chunk_count": len(chunks),
            "resume_count": sum(bool(row["restored_from_resume"]) for row in chunks),
            "chunks": chunk_bindings,
            "maximum_jobs_respected": len(chunks)
            <= int(protocol["execution"]["maximum_jobs_submitted_now"]),
            "chunks_contiguous": chunks[0]["start_update"] == 0
            and chunks[-1]["completed_updates"] == base.UPDATE_COUNT,
        },
        "runtime_access": {
            "query_read_union": query_rows,
            "query_read_union_sha256": canonical_sha256(query_rows),
            "reference_read_union": reference_rows,
            "reference_read_union_sha256": canonical_sha256(reference_rows),
            "reference_read_union_equals_signed_allowlist": reference_rows
            == manifest["allowed_reference_physical_rows"],
            "heldout_reference_intersection_count": len(
                set(reference_rows) & set(manifest["heldout_reference_physical_rows"])
            ),
        },
        "protected_access_counts": protected_zeros(),
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_required_stage": "R1_MAIN_TRAINING_VALIDATION",
    }
    result_path = output_dir / "train_result.json"
    if result_path.exists():
        if base.read_json(result_path) != result:
            raise ContractRepairAbort("existing train result drift during recovery")
    else:
        atomic_json_no_clobber(result_path, result)


def main() -> None:
    process_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=base.ALLOWED_ARMS, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--max-runtime-seconds", type=int, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    authority_path = args.authority.resolve()
    cache_root = args.cache_root.resolve()
    output_dir = args.output_dir.resolve()
    if ROOT not in cache_root.parents or ROOT not in output_dir.parents:
        raise ContractRepairAbort("cache/output path escapes RC root")
    protocol, authorized_arm, authorized_fold = validate_contract(
        protocol_path, authority_path
    )
    require_cli_scope(args.arm, args.fold, authorized_arm, authorized_fold)
    frozen_runtime_guard = protocol.get("execution", {}).get(
        "runner_runtime_guard_seconds"
    )
    if (
        isinstance(frozen_runtime_guard, bool)
        or not isinstance(frozen_runtime_guard, int)
        or frozen_runtime_guard < 300
        or frozen_runtime_guard > 13_800
    ):
        raise ContractRepairAbort("frozen runtime guard is outside the V1.8 safe range")
    if args.max_runtime_seconds != frozen_runtime_guard:
        raise ContractRepairAbort("CLI runtime guard does not match the frozen protocol")
    deadline = process_started + frozen_runtime_guard
    artifacts = artifact_contract(protocol)

    # All mappings, all required cache keys, source/model/logical payload hashes,
    # and the signed reference allowlist are checked before formal output exists.
    frozen = input_closure(protocol, cache_root, args.fold)
    if not torch.cuda.is_available():
        raise ContractRepairAbort("CUDA is required")
    device = torch.device("cuda")
    protocol_sha = base.file_sha256(protocol_path)
    authority_sha = base.file_sha256(authority_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")

    identity = {
        **artifacts["identity"],
        "arm": args.arm,
        "outer_fold": args.fold,
        "protocol_sha256": protocol_sha,
        "authority_sha256": authority_sha,
    }
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if base.read_json(identity_path) != identity:
            raise ContractRepairAbort("persistent output identity drift")
    else:
        atomic_json_no_clobber(identity_path, identity)
    manifest_path = output_dir / "signed_reference_access_manifest.json"
    if manifest_path.exists():
        if base.read_json(manifest_path) != frozen["manifest"]:
            raise ContractRepairAbort("persistent signed access manifest drift")
    else:
        atomic_json_no_clobber(manifest_path, frozen["manifest"])
    prior_chunks = _chunk_history(output_dir, chunk_spec=artifacts["chunk"])
    maximum_chunks = int(protocol["execution"]["maximum_jobs_submitted_now"])
    if len(prior_chunks) > maximum_chunks:
        raise ContractRepairAbort("V1.8 chunk count exceeds authority")

    base.seed_everything()
    model = DINO_RCDE_V1_2().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != base.EXPECTED_PARAMETER_COUNT:
        raise ContractRepairAbort("parameter-count drift")
    initial_state_sha = base.state_dict_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    completed = 0
    pair_count = 0
    loss_trace: list[dict[str, Any]] = []
    restored = False
    restored_resume_state: str | None = None
    restored_resume_state_sha256: str | None = None
    if prior_chunks:
        resume_path = _artifact_path(output_dir, prior_chunks[-1]["resume_state"])
        restored_resume_state = str(resume_path.relative_to(output_dir))
        restored_resume_state_sha256 = base.file_sha256(resume_path)
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ContractRepairAbort("resume state is not a mapping")
        validate_resume_state(
            state,
            arm=args.arm,
            fold=args.fold,
            protocol_sha256=protocol_sha,
            authority_sha256=authority_sha,
            initial_state_sha256=initial_state_sha,
            artifact_spec=artifacts["resume"],
        )
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        completed = int(state["completed_updates"])
        loss_trace = list(state["loss_trace"])
        pair_count = int(state["pair_count"])
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
        restored = True
    if completed < 0 or completed > base.UPDATE_COUNT or len(loss_trace) != completed:
        raise ContractRepairAbort("resume progress cardinality drift")
    if (prior_chunks[-1]["completed_updates"] if prior_chunks else 0) != completed:
        raise ContractRepairAbort("resume progress does not match chunk coverage")

    correct = frozen["correct"]
    wrong = frozen["wrong"]
    if completed == base.UPDATE_COUNT:
        finalize_training(
            output_dir=output_dir,
            protocol=protocol,
            protocol_sha256=protocol_sha,
            authority_sha256=authority_sha,
            identity_path=identity_path,
            manifest_path=manifest_path,
            manifest=frozen["manifest"],
            correct=correct,
            wrong=wrong,
            initial_state_sha256=initial_state_sha,
            arm=args.arm,
            fold=args.fold,
        )
        return
    if len(prior_chunks) >= maximum_chunks:
        raise ContractRepairAbort("V1.8 chunk budget exhausted before update 2048")

    schedule_digest = schedule_prefix(correct, wrong, args.fold, completed)
    manifest = frozen["manifest"]
    tracker = CacheAccessTracker(
        index=frozen["index"],
        query_by_execution=frozen["query_by_execution"],
        reference_by_row=frozen["reference_by_row"],
        allowed_queries=manifest["eligible_query_execution_ordinals"],
        allowed_references=manifest["allowed_reference_physical_rows"],
        model_sha256=protocol["cache"]["model_checkpoint_logical_sha256"],
    )
    chunk_start = completed
    training_started = time.monotonic()
    model.train()
    while completed < base.UPDATE_COUNT:
        if completed > chunk_start and time.monotonic() >= deadline:
            break
        event, record = run_update(
            model=model,
            optimizer=optimizer,
            tracker=tracker,
            correct=correct,
            wrong=wrong,
            fold=args.fold,
            arm=args.arm,
            update_zero=completed,
            device=device,
        )
        completed += 1
        pair_count += int(record["pair_count"])
        loss_trace.append(record)
        schedule_digest.update(base.json_bytes(event) + b"\n")
        if completed == 1 or completed % 32 == 0:
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "fold": args.fold,
                        "chunk_start_update": chunk_start,
                        "completed_updates": completed,
                        "mean_pair_loss": record["mean_pair_loss"],
                        "elapsed_seconds": time.monotonic() - process_started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if completed <= chunk_start:
        raise ContractRepairAbort("runtime guard left no completed optimizer update")
    torch.cuda.synchronize(device)
    complete = completed == base.UPDATE_COUNT

    # Build all four artifacts in a private sibling directory, fsync them,
    # then atomically rename that directory.  Before the rename no official
    # chunk exists; after it the exact four-file transaction is complete.
    chunk_name = f"chunk_job{job_id}"
    chunk_dir = output_dir / chunk_name
    staging_dir = output_dir.with_name(
        f".{output_dir.name}.{chunk_name}.staging.{os.getpid()}"
    )
    if chunk_dir.exists() or staging_dir.exists():
        raise ContractRepairAbort("chunk job namespace already exists")
    staging_dir.mkdir(parents=False, exist_ok=False)
    resume_path = staging_dir / "resume_state.pt"
    save_resume_no_clobber(
        resume_path,
        model=model,
        optimizer=optimizer,
        arm=args.arm,
        fold=args.fold,
        completed=completed,
        loss_trace=loss_trace,
        pair_count=pair_count,
        initial_state_sha256=initial_state_sha,
        protocol_sha256=protocol_sha,
        authority_sha256=authority_sha,
        artifact_spec=artifacts["resume"],
    )
    read_ledger_path = staging_dir / "runtime_read_ledger.json"
    atomic_json_no_clobber(
        read_ledger_path,
        tracker.ledger(
            job_id=job_id,
            start_update=chunk_start,
            completed_updates=completed,
            artifact_spec=artifacts["read_ledger"],
        ),
    )
    preflight = dict(frozen["preflight"])
    preflight.update(
        {
            "job_id": job_id,
            "protocol_sha256": protocol_sha,
            "authority_sha256": authority_sha,
            "signed_reference_access_manifest_sha256": base.file_sha256(manifest_path),
        }
    )
    preflight_path = staging_dir / "input_preflight.json"
    atomic_json_no_clobber(preflight_path, preflight)
    chunk_path = staging_dir / "chunk.json"
    chunk = {
        **artifacts["chunk"],
        "arm": args.arm,
        "outer_fold": args.fold,
        "job_id": job_id,
        "start_update": chunk_start,
        "completed_updates": completed,
        "training_complete": complete,
        "restored_from_resume": restored,
        "restored_resume_state": restored_resume_state,
        "restored_resume_state_sha256": restored_resume_state_sha256,
        "input_preflight": f"{chunk_name}/input_preflight.json",
        "input_preflight_sha256": base.file_sha256(preflight_path),
        "runtime_read_ledger": f"{chunk_name}/runtime_read_ledger.json",
        "runtime_read_ledger_sha256": base.file_sha256(read_ledger_path),
        "resume_state": f"{chunk_name}/resume_state.pt",
        "resume_state_sha256": base.file_sha256(resume_path),
        "elapsed_seconds": time.monotonic() - process_started,
        "training_seconds": time.monotonic() - training_started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protected_access_counts": protected_zeros(),
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
    }
    atomic_json_no_clobber(chunk_path, chunk)
    _fsync_directory(staging_dir)
    os.rename(staging_dir, chunk_dir)
    _fsync_directory(output_dir)

    if complete:
        finalize_training(
            output_dir=output_dir,
            protocol=protocol,
            protocol_sha256=protocol_sha,
            authority_sha256=authority_sha,
            identity_path=identity_path,
            manifest_path=manifest_path,
            manifest=manifest,
            correct=correct,
            wrong=wrong,
            initial_state_sha256=initial_state_sha,
            arm=args.arm,
            fold=args.fold,
        )


if __name__ == "__main__":
    main()
