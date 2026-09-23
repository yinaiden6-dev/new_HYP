#!/usr/bin/env python3
"""Run one V96-authorized formal P-V2 fit and exact 1024-resume replay.

The runner first materializes and seals the complete target-free table from
the V95-routed pair-feature and structural-sidecar shards.  Only after that
seal exists does it open the independently validated loss-role join and bind
the anonymous pair members to target/rival roles.  Training retains the frozen
16-parameter P head and the single natural pairwise objective; no P-lock,
positive-only MAP/H0 decision, retrieval score, or outcome is consumed.

Every primary 64-update checkpoint is immutable.  The replay path strictly
restores the primary update-1024 artifact and must reproduce the complete
update-2048 state exactly before a non-consumable producer receipt is emitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.dino_rcde_cw1_multitile_sr0_p_v1 import (  # noqa: E402
    FEATURE_DIM,
    PARAMETER_COUNT,
    PEpisodeKey,
    P_CHECKPOINT_INTERVAL_UPDATES,
    P_EPISODES_PER_UPDATE,
    P_GRADIENT_CLIP_L2,
    P_SEED,
    P_UPDATES_TOTAL,
    SharedMultitilePHead,
    p_learning_rate,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    DeterministicEpisodeSampler,
    capture_rng_state,
    fused_direction_pair_loss,
    initialize_training,
    restore_rng_state,
    state_sha256,
    tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_fit_runtime_v1 import (  # noqa: E402
    BoundShardPayloadV1,
    JoinedTrainingEpisodeV1,
    PFitV2RuntimeError,
    SealedTrainingEpisodePopulationV1,
    ShardRouteV1,
    ValidatedV95FitManifestV1,
    assert_compiled_role_free_exact_parity,
    build_and_seal_target_free_input_table,
    join_training_roles_after_seal,
    score_compiled_direction_v1,
    validate_v95_fit_manifest,
)


AUTHORITY_SCHEMA = "rc_current_authority_v98_20260818"
AUTHORITY_STATUS = "RCDE_SR0_MT_P_V2_DEV_CONTINUATION_EXECUTION_AUTHORIZED"
INDEX_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_training_manifest_index_v1_20260818"
INDEX_STATUS = "RCDE_SR0_MT_P_V2_TRAINING_MANIFEST_INDEX_READY"
LOSS_JOIN_SCHEMA = "rc_dino_rcde_sr0_mt_loss_join_v1_20260817"
LOSS_JOIN_STATUS = "RCDE_SR0_MT_LOSS_JOIN_READY"

CHECKPOINT_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_checkpoint_v1_20260818"
CHECKPOINT_STATUS = "RCDE_SR0_MT_P_V2_COMPACT_CHECKPOINT_READY"
CHECKPOINT_CLAIM = "ENGINEERING_P_V2_EXACT_RESUME_STATE_ONLY"
RESULT_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_fit_result_v1_20260818"
RESULT_STATUS = "RCDE_SR0_MT_P_V2_FIT_PRODUCER_READY"
RESULT_CLAIM = "ENGINEERING_ONE_P_V2_FIT_AND_EXACT_RESUME_ONLY"
TRACE_NAMESPACE = "RCDE_SR0_MT_P_V2_ROLLING_TRACE_V1"
TRACE_WINDOW = 64

EXPECTED_SCOPE = {
    "p_v2_exact_resume_validation": True,
    "p_v2_formal_fit_execution": True,
}
FIT_IDS = tuple(
    fit_id
    for outer in range(1, 5)
    for fit_id in (
        *(
            f"P_OUTER{outer}_INNER{inner}_FIT"
            for inner in range(1, 5)
            if inner != outer
        ),
        f"P_OUTER{outer}_OUTER_REFIT",
    )
)
LOSS_JOIN_EPISODE_FIELDS = frozenset(
    {
        "execution_ordinal",
        "loss_join_sha256",
        "pair_address_sha256",
        "query_id",
        "query_source_image_sha256",
        "rival_candidate_key",
        "rival_direction_addresses",
        "rival_member_ordinal",
        "role_record_sha256",
        "target_candidate_key",
        "target_direction_addresses",
        "target_member_ordinal",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "fit_id",
        "fit_index",
        "path_kind",
        "completed_updates",
        "consumable_checkpoint_authorized",
        "bindings",
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "scheduler_cursor",
        "current_learning_rate",
        "rolling_trace_state",
        "state_sha256",
        "logical_sha256",
        "restored_from_checkpoint_file_sha256",
        "restored_from_checkpoint_state_sha256",
    }
)
STATE_FIELDS = frozenset(
    {
        "completed_updates",
        "current_learning_rate",
        "scheduler_cursor",
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "rolling_trace_state",
    }
)
BINDING_FIELDS = frozenset(
    {
        "authority_file_sha256",
        "authority_logical_sha256",
        "contract_file_sha256",
        "parent_authority_v95_file_sha256",
        "parent_authority_v95_logical_sha256",
        "v95_index_file_sha256",
        "v95_index_logical_sha256",
        "v95_validation_file_sha256",
        "v95_validation_logical_sha256",
        "fit_manifest_file_sha256",
        "fit_manifest_logical_sha256",
        "loss_join_file_sha256",
        "loss_join_logical_sha256",
        "loss_join_validation_file_sha256",
        "loss_join_validation_logical_sha256",
        "target_free_table_sha256",
        "role_join_sha256",
        "ordered_pair_shards_sha256",
        "ordered_sidecar_shards_sha256",
        "p_head_file_sha256",
        "role_free_scorer_file_sha256",
        "p_lock_core_file_sha256",
        "p_selector_file_sha256",
        "p_v2_adapter_file_sha256",
        "fit_runtime_file_sha256",
        "runner_file_sha256",
        "validator_file_sha256",
        "recipe_sha256",
    }
)
ROLLING_TRACE_FIELDS = frozenset(
    {"count", "chain_sha256", "window_start", "window", "window_sha256"}
)
RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "fit_id",
        "fit_index",
        "authority_sha256",
        "authority_logical_sha256",
        "fit_manifest_sha256",
        "fit_manifest_logical_sha256",
        "input_seal_sha256",
        "role_join_sha256",
        "primary_checkpoint_artifacts",
        "replay_checkpoint_artifact",
        "primary_result",
        "replay_result",
        "protected_access_audit",
        "consumable_checkpoint_authorized",
        "scientific_GO_or_NO_GO",
        "automatic_stage_advance",
        "next_authorized_stage",
        "logical_sha256",
    }
)


class PFitV2ExecutionError(RuntimeError):
    """A V96 binding, input, training, checkpoint, or replay invariant failed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise PFitV2ExecutionError(message)


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "logical_sha256"})


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(relative: str) -> Path:
    raw = Path(relative)
    require(
        isinstance(relative, str)
        and relative
        and not raw.is_absolute()
        and ".." not in raw.parts,
        f"unsafe bound path {relative!r}",
    )
    path = (ROOT / raw).resolve(strict=True)
    require(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(), f"invalid bound file {relative}")
    return path


def read_json(path: Path, *, name: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{name} is absent or symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def latest_authority(args_path: Path) -> tuple[dict[str, Any], str]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows: list[tuple[int, Path]] = []
    for path in (ROOT / "registry").glob("current_authority_v*_*.json"):
        match = pattern.fullmatch(path.name)
        if match and path.is_file() and not path.is_symlink():
            rows.append((int(match.group(1)), path.resolve()))
    exact = (ROOT / "registry/current_authority_v98_20260818.json").resolve(strict=True)
    supplied = args_path.resolve(strict=True)
    require(rows and max(version for version, _ in rows) == 98, "V98 is not latest authority")
    require([path for version, path in rows if version == 98] == [exact] and supplied == exact, "V98 exact authority path drift")
    authority = read_json(exact, name="V98 authority")
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("authorized_scope") == EXPECTED_SCOPE
        and authority.get("fit_count") == 16
        and authority.get("training_authorized") is True
        and authority.get("model_load_authorized") is True
        and authority.get("model_forward_authorized") is True
        and authority.get("model_backward_authorized") is True
        and authority.get("model_update_authorized") is True
        and authority.get("pair_feature_payload_read_authorized") is True
        and authority.get("sidecar_shard_payload_read_authorized") is True
        and authority.get("target_rival_loss_role_read_authorized") is True
        and authority.get("consumable_checkpoint_authorized") is False
        and authority.get("p_lock_materialization_authorized") is False
        and authority.get("automatic_stage_advance") is False
        and authority.get("scientific_GO_or_NO_GO") is None
        and authority.get("next_authorized_stage") is None
        and authority.get("logical_sha256") == logical(authority),
        "V96 authority scope/status/claim drift",
    )
    for forbidden in (
        "d1_score_rank_winner_read_authorized",
        "retrieval_outcome_read_authorized",
        "opened_read_authorized",
        "sealed_read_authorized",
        "C8_read_authorized",
        "S8_read_authorized",
        "target_insertion_authorized",
        "donor_matching_authorized",
        "v_training_authorized",
        "heldout_scoring_authorized",
    ):
        require(authority.get(forbidden) is False, f"V96 forbidden scope enabled: {forbidden}")
    bindings = authority.get("bindings")
    require(isinstance(bindings, Mapping), "V96 bindings absent")
    for name, raw in bindings.items():
        require(isinstance(raw, Mapping) and set(raw) >= {"path", "sha256", "bytes"}, f"V96 binding {name} schema drift")
        path = safe_path(str(raw["path"]))
        require(file_sha(path) == raw["sha256"] and path.stat().st_size == raw["bytes"], f"V96 binding {name} physical drift")
        if "logical_sha256" in raw:
            value = read_json(path, name=f"V96 binding {name}")
            require(value.get("logical_sha256") == raw["logical_sha256"] == logical(value), f"V96 binding {name} logical drift")
    return authority, file_sha(exact)


def _bound(authority: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    bindings = authority["bindings"]
    require(isinstance(bindings, Mapping) and isinstance(bindings.get(key), Mapping), f"V96 binding absent: {key}")
    return bindings[key]


def validate_index_and_manifest(
    authority: Mapping[str, Any], fit_index: int
) -> tuple[dict[str, Any], ValidatedV95FitManifestV1, dict[str, Any], Path, str]:
    index_binding = _bound(authority, "v95_manifest_index")
    index_path = safe_path(str(index_binding["path"]))
    index = read_json(index_path, name="V95 manifest index")
    require(
        index.get("schema_version") == INDEX_SCHEMA
        and index.get("status") == INDEX_STATUS
        and index.get("logical_sha256") == index_binding["logical_sha256"] == logical(index)
        and index.get("fit_count") == 16,
        "V95 index envelope drift",
    )
    rows = index.get("fits")
    require(
        isinstance(rows, list)
        and len(rows) == 16
        and [row.get("fit_id") for row in rows] == list(FIT_IDS)
        and index.get("fit_sequence_sha256") == canonical(rows),
        "V95 fit sequence drift",
    )
    row = rows[fit_index]
    fit_id = FIT_IDS[fit_index]
    require(row.get("fit_id") == fit_id, "fit-index/fit-ID drift")
    manifest_binding = _bound(authority, f"fit_manifest_{fit_index:02d}")
    require(row.get("path") == manifest_binding.get("path"), "V95 index/authority manifest path drift")
    path = safe_path(str(row["path"]))
    manifest_file_sha256 = file_sha(path)
    require(manifest_file_sha256 == manifest_binding["sha256"], "fit manifest physical hash drift")
    raw = read_json(path, name="V95 fit manifest")
    manifest = validate_v95_fit_manifest(raw, expected_fit_id=fit_id)
    require(
        manifest.manifest_logical_sha256
        == row.get("logical_sha256")
        == manifest_binding.get("logical_sha256"),
        "fit manifest logical hash drift",
    )
    return index, manifest, raw, path, manifest_file_sha256


def _validate_route_validation(route: ShardRouteV1, *, kind: str) -> dict[str, Any]:
    path = safe_path(route.validation_path)
    require(file_sha(path) == route.validation_sha256, f"{kind} route validation physical hash drift")
    value = read_json(path, name=f"{kind} route validation")
    require(value.get("logical_sha256") == logical(value), f"{kind} route validation logical drift")
    if kind == "pair":
        require(
            value.get("validation_pass") is True
            and value.get("status") == "RCDE_SR0_MT_ROLE_FREE_PAIR_FEATURE_CACHE_SHARD_INDEPENDENT_VALIDATION_PASS"
            and value.get("materialized_artifact_file_sha256") == route.artifact_sha256
            and value.get("execution_start") == route.execution_start
            and value.get("execution_stop") == route.execution_stop
            and value.get("h0_coordinate_count") == 0,
            "pair route independent validation drift",
        )
    else:
        require(
            value.get("validation_pass") is True
            and value.get("status") == "RCDE_SR0_MT_E0_FULL594_STRUCTURE_SIDECAR_SHARD_INDEPENDENT_VALIDATION_PASS"
            and value.get("sidecar_file_sha256") == route.artifact_sha256
            and value.get("execution_start") == route.execution_start
            and value.get("execution_stop") == route.execution_stop
            and value.get("legacy_h0_coordinate_count") == 0,
            "sidecar route independent validation drift",
        )
    return value


def load_routed_shards(
    manifest: ValidatedV95FitManifestV1,
) -> tuple[
    tuple[BoundShardPayloadV1, ...],
    tuple[BoundShardPayloadV1, ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[int, int, str], Mapping[str, Any]],
]:
    pair_values: list[BoundShardPayloadV1] = []
    side_values: list[BoundShardPayloadV1] = []
    pair_receipts: list[dict[str, Any]] = []
    side_receipts: list[dict[str, Any]] = []
    side_records: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for route in manifest.pair_routes:
        _validate_route_validation(route, kind="pair")
        path = safe_path(route.artifact_path)
        digest = file_sha(path)
        require(digest == route.artifact_sha256, "pair artifact physical hash drift")
        payload = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
        require(isinstance(payload, Mapping), "pair PT payload is not a mapping")
        pair_values.append(BoundShardPayloadV1(route.artifact_path, digest, payload))
        pair_receipts.append(
            {
                "shard_ordinal": route.shard_ordinal,
                "artifact_path": route.artifact_path,
                "artifact_sha256": digest,
                "validation_path": route.validation_path,
                "validation_sha256": route.validation_sha256,
            }
        )
    selected_execution = {item.execution_ordinal for item in manifest.neutral_episodes}
    for route in manifest.sidecar_routes:
        _validate_route_validation(route, kind="sidecar")
        path = safe_path(route.artifact_path)
        digest = file_sha(path)
        require(digest == route.artifact_sha256, "sidecar artifact physical hash drift")
        payload = read_json(path, name="sidecar shard")
        side_values.append(BoundShardPayloadV1(route.artifact_path, digest, payload))
        side_receipts.append(
            {
                "shard_ordinal": route.shard_ordinal,
                "artifact_path": route.artifact_path,
                "artifact_sha256": digest,
                "validation_path": route.validation_path,
                "validation_sha256": route.validation_sha256,
            }
        )
        records = payload.get("records")
        require(isinstance(records, list), "sidecar records absent")
        for raw in records:
            require(isinstance(raw, Mapping), "sidecar record is not a mapping")
            execution = raw.get("execution_ordinal")
            if execution not in selected_execution:
                continue
            key = (int(execution), int(raw.get("member_ordinal")), str(raw.get("direction")))
            require(key not in side_records, "sidecar direction address alias")
            side_records[key] = raw
    return tuple(pair_values), tuple(side_values), pair_receipts, side_receipts, side_records


def validate_loss_join_after_table_seal(
    authority: Mapping[str, Any],
    manifest: ValidatedV95FitManifestV1,
    side_records: Mapping[tuple[int, int, str], Mapping[str, Any]],
    table_sha256: str,
) -> tuple[dict[str, Any], str]:
    require(isinstance(table_sha256, str) and len(table_sha256) == 64, "target-free table was not sealed")
    binding = _bound(authority, "loss_join")
    path = safe_path(str(binding["path"]))
    value = read_json(path, name="validated loss join")
    require(
        file_sha(path) == binding["sha256"]
        and value.get("schema_version") == LOSS_JOIN_SCHEMA
        and value.get("status") == LOSS_JOIN_STATUS
        and value.get("logical_sha256") == binding["logical_sha256"] == logical(value)
        and value.get("episode_count") == 594
        and value.get("score_rank_outcome_read_count") == 0
        and value.get("target_insertion_count") == 0,
        "loss join envelope/binding drift",
    )
    validation_binding = _bound(authority, "loss_join_validation")
    validation_path = safe_path(str(validation_binding["path"]))
    validation = read_json(validation_path, name="loss join validation")
    require(
        file_sha(validation_path) == validation_binding["sha256"]
        and validation.get("logical_sha256") == validation_binding["logical_sha256"] == logical(validation)
        and validation.get("validation_pass") is True,
        "loss join independent validation drift",
    )
    rows = value.get("episodes")
    require(isinstance(rows, list) and value.get("episode_population_sha256") == canonical(rows), "loss join population drift")
    by_execution: dict[int, Mapping[str, Any]] = {}
    for raw in rows:
        require(isinstance(raw, Mapping) and set(raw) == LOSS_JOIN_EPISODE_FIELDS, "loss join episode field drift")
        require(raw.get("loss_join_sha256") == canonical({key: item for key, item in raw.items() if key != "loss_join_sha256"}), "loss join episode hash drift")
        execution = raw.get("execution_ordinal")
        require(type(execution) is int and execution not in by_execution, "loss join execution alias")
        by_execution[execution] = raw
    selected_hashes: list[str] = []
    for role in manifest.role_addresses:
        execution = role.neutral.execution_ordinal
        row = by_execution.get(execution)
        require(
            row is not None
            and row.get("query_id") == role.neutral.query_id
            and row.get("query_source_image_sha256") == role.neutral.query_source_image_sha256
            and row.get("pair_address_sha256") == role.neutral.pair_address_sha256
            and row.get("target_member_ordinal") == role.target_member_ordinal
            and row.get("rival_member_ordinal") == role.rival_member_ordinal
            and row.get("target_candidate_key") == role.target_candidate_key
            and row.get("rival_candidate_key") == role.rival_candidate_key,
            "V95/loss-join role address drift",
        )
        for role_name, member, candidate in (
            ("target", role.target_member_ordinal, role.target_candidate_key),
            ("rival", role.rival_member_ordinal, role.rival_candidate_key),
        ):
            observed = row[f"{role_name}_direction_addresses"]
            require(isinstance(observed, Mapping) and set(observed) == {"a_to_b", "b_to_a"}, "loss join direction-address schema drift")
            expected: dict[str, str] = {}
            for direction in ("a_to_b", "b_to_a"):
                sidecar = side_records.get((execution, member, direction))
                require(
                    sidecar is not None
                    and sidecar.get("candidate_key") == candidate
                    and sidecar.get("pair_address_sha256") == role.neutral.pair_address_sha256,
                    "loss join/sidecar direction candidate drift",
                )
                expected[direction] = canonical(
                    {
                        "query_id": role.neutral.query_id,
                        "query_source_image_sha256": role.neutral.query_source_image_sha256,
                        "execution_ordinal": execution,
                        "candidate_key": candidate,
                        "candidate_reference_source_sha256": sidecar.get("candidate_reference_source_sha256"),
                        "direction": direction,
                    }
                )
            require(dict(observed) == expected, "loss join direction address drift")
        selected_hashes.append(str(row["loss_join_sha256"]))
    seal = canonical(
        {
            "target_free_table_sha256": table_sha256,
            "loss_join_file_sha256": binding["sha256"],
            "loss_join_logical_sha256": binding["logical_sha256"],
            "selected_loss_join_sha256": selected_hashes,
        }
    )
    return value, seal


def initial_trace(fit_id: str) -> dict[str, Any]:
    del fit_id
    window_sha256 = canonical([])
    return {
        "count": 0,
        "chain_sha256": None,
        "window_start": 1,
        "window": [],
        "window_sha256": window_sha256,
    }


def append_trace(state: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    require(set(state) == ROLLING_TRACE_FIELDS, "rolling trace state field drift")
    count = int(state["count"]) + 1
    require(entry.get("completed_update") == count, "rolling trace update order drift")
    window = [dict(item) for item in state["window"]]
    if len(window) == TRACE_WINDOW:
        window = []
    window.append(dict(entry))
    window_start = count - len(window) + 1
    window_sha256 = canonical(window)
    if len(window) == TRACE_WINDOW:
        chain_sha256 = canonical(
            {
                "previous_chain_sha256": state["chain_sha256"],
                "window_start": window_start,
                "window_sha256": window_sha256,
            }
        )
    else:
        chain_sha256 = state["chain_sha256"]
    return {
        "count": count,
        "chain_sha256": chain_sha256,
        "window_start": window_start,
        "window": window,
        "window_sha256": window_sha256,
    }


def validate_trace(state: Mapping[str, Any], completed_updates: int) -> None:
    window = state.get("window")
    window_start = state.get("window_start")
    expected_window_count = 0 if completed_updates == 0 else ((completed_updates - 1) % TRACE_WINDOW) + 1
    require(
        set(state) == ROLLING_TRACE_FIELDS
        and state.get("count") == completed_updates
        and isinstance(window, list)
        and len(window) == expected_window_count
        and window_start == (1 if completed_updates == 0 else completed_updates - len(window) + 1)
        and all(
            isinstance(item, Mapping)
            and item.get("completed_update") == int(window_start) + index
            for index, item in enumerate(window)
        )
        and state.get("window_sha256") == canonical(window)
        and (
            completed_updates == 0
            or isinstance(state.get("chain_sha256"), str)
            and len(state["chain_sha256"]) == 64
        ),
        "rolling trace state drift",
    )


def component_state_hash(value: object) -> str:
    return state_sha256(value)


def checkpoint_state_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {key: checkpoint[key] for key in STATE_FIELDS}


def checkpoint_logical_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "logical_sha256",
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "rolling_trace_state",
    }
    value = {key: item for key, item in checkpoint.items() if key not in excluded}
    value.update(
        {
            "model_state_sha256": component_state_hash(checkpoint["model_state"]),
            "optimizer_state_sha256": component_state_hash(checkpoint["optimizer_state"]),
            "sampler_state_sha256": component_state_hash(checkpoint["sampler_state"]),
            "rng_state_sha256": component_state_hash(checkpoint["rng_state"]),
            "rolling_trace_state_sha256": canonical(checkpoint["rolling_trace_state"]),
        }
    )
    return value


def make_checkpoint(
    *,
    fit_id: str,
    fit_index: int,
    path_kind: str,
    completed_updates: int,
    bindings: Mapping[str, str],
    model: SharedMultitilePHead,
    optimizer: torch.optim.AdamW,
    sampler: DeterministicEpisodeSampler,
    rolling_trace_state: Mapping[str, Any],
    restored_from_checkpoint_file_sha256: str | None,
    restored_from_checkpoint_state_sha256: str | None,
) -> dict[str, Any]:
    require(path_kind in {"PRIMARY", "REPLAY"}, "checkpoint path kind drift")
    require(set(bindings) == BINDING_FIELDS, "checkpoint binding field set drift")
    validate_trace(rolling_trace_state, completed_updates)
    current_lr = 0.0 if completed_updates == 0 else p_learning_rate(completed_updates)
    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": CHECKPOINT_STATUS,
        "claim_level": CHECKPOINT_CLAIM,
        "fit_id": fit_id,
        "fit_index": fit_index,
        "path_kind": path_kind,
        "completed_updates": completed_updates,
        "consumable_checkpoint_authorized": False,
        "bindings": dict(bindings),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sampler_state": sampler.state(),
        "rng_state": capture_rng_state(),
        "scheduler_cursor": completed_updates,
        "current_learning_rate": current_lr,
        "rolling_trace_state": dict(rolling_trace_state),
        "restored_from_checkpoint_file_sha256": restored_from_checkpoint_file_sha256,
        "restored_from_checkpoint_state_sha256": restored_from_checkpoint_state_sha256,
    }
    checkpoint["state_sha256"] = component_state_hash(checkpoint_state_payload(checkpoint))
    checkpoint["logical_sha256"] = canonical(checkpoint_logical_payload(checkpoint))
    require(set(checkpoint) == CHECKPOINT_FIELDS, "checkpoint field set drift")
    return checkpoint


def validate_checkpoint(
    value: Mapping[str, Any],
    *,
    fit_id: str,
    fit_index: int,
    path_kind: str,
    completed_updates: int,
    bindings: Mapping[str, str],
) -> None:
    require(
        set(value) == CHECKPOINT_FIELDS
        and value.get("schema_version") == CHECKPOINT_SCHEMA
        and value.get("status") == CHECKPOINT_STATUS
        and value.get("claim_level") == CHECKPOINT_CLAIM
        and value.get("fit_id") == fit_id
        and value.get("fit_index") == fit_index
        and value.get("path_kind") == path_kind
        and value.get("completed_updates") == completed_updates
        and value.get("scheduler_cursor") == completed_updates
        and value.get("bindings") == dict(bindings)
        and value.get("consumable_checkpoint_authorized") is False,
        "checkpoint envelope/binding drift",
    )
    validate_trace(value["rolling_trace_state"], completed_updates)
    require(
        value.get("state_sha256") == component_state_hash(checkpoint_state_payload(value))
        and value.get("logical_sha256") == canonical(checkpoint_logical_payload(value)),
        "checkpoint state/logical hash drift",
    )


def exclusive_torch(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    temp = Path(temporary)
    try:
        torch.save(dict(value), temp)
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        temp.chmod(0o444)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable JSON exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(0o444)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def artifact_receipt(path: Path, root: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "physical_sha256": file_sha(path),
        "logical_sha256": checkpoint["logical_sha256"],
        "state_sha256": checkpoint["state_sha256"],
        "completed_updates": checkpoint["completed_updates"],
    }


def episode_loss(
    model: SharedMultitilePHead, episode: JoinedTrainingEpisodeV1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = tuple(score_compiled_direction_v1(model, item).candidate_utility for item in episode.target)
    rival = tuple(score_compiled_direction_v1(model, item).candidate_utility for item in episode.rival)
    bundle = fused_direction_pair_loss(
        target_a_to_b=target[0],
        target_b_to_a=target[1],
        rival_a_to_b=rival[0],
        rival_b_to_a=rival[1],
    )
    return bundle.loss, bundle.target_utility, bundle.rival_utility, bundle.margin


def run_updates(
    *,
    model: SharedMultitilePHead,
    optimizer: torch.optim.AdamW,
    episodes: tuple[JoinedTrainingEpisodeV1, ...],
    sampler: DeterministicEpisodeSampler,
    start_update: int,
    stop_update: int,
    rolling_trace_state: Mapping[str, Any],
    checkpoint_callback: Any | None,
) -> dict[str, Any]:
    trace = dict(rolling_trace_state)
    validate_trace(trace, start_update)
    require(0 <= start_update < stop_update <= P_UPDATES_TOTAL, "training update interval drift")
    for update in range(start_update + 1, stop_update + 1):
        learning_rate = p_learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        selected = sampler.draw(P_EPISODES_PER_UPDATE)
        optimizer.zero_grad(set_to_none=True)
        bundles = [episode_loss(model, episodes[index]) for index in selected]
        losses = [item[0] for item in bundles]
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient = model.linear.weight.grad
        require(
            gradient is not None
            and gradient.shape == (1, FEATURE_DIM)
            and gradient.dtype == torch.float64
            and bool(torch.isfinite(gradient).all()),
            "P-V2 gradient absent/nonfinite/dtype drift",
        )
        raw_gradient_sha256 = tensor_sha256(gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), P_GRADIENT_CLIP_L2)
        clipped_gradient_sha256 = tensor_sha256(model.linear.weight.grad)
        optimizer.step()
        entry = {
            "completed_update": update,
            "episode_indices": list(selected),
            "loss_float64_hex": float(loss.detach()).hex(),
            "loss_tensor_sha256": tensor_sha256(loss.detach()),
            "target_utility_population_sha256": canonical(
                [tensor_sha256(item[1].detach()) for item in bundles]
            ),
            "rival_utility_population_sha256": canonical(
                [tensor_sha256(item[2].detach()) for item in bundles]
            ),
            "margin_population_sha256": canonical(
                [tensor_sha256(item[3].detach()) for item in bundles]
            ),
            "raw_gradient_sha256": raw_gradient_sha256,
            "clipped_gradient_sha256": clipped_gradient_sha256,
            "parameter_sha256": tensor_sha256(model.linear.weight),
            "optimizer_state_sha256": component_state_hash(optimizer.state_dict()),
            "next_sampler_state_sha256": component_state_hash(sampler.state()),
            "learning_rate_float64_hex": float(learning_rate).hex(),
        }
        trace = append_trace(trace, entry)
        if checkpoint_callback is not None and update % P_CHECKPOINT_INTERVAL_UPDATES == 0:
            checkpoint_callback(update, trace, model, optimizer, sampler)
    return trace


def restore_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    episode_keys: tuple[PEpisodeKey, ...],
    fit_id: str,
) -> tuple[SharedMultitilePHead, torch.optim.AdamW, DeterministicEpisodeSampler, dict[str, Any]]:
    model, optimizer = initialize_training(P_SEED)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    sampler_raw = checkpoint["sampler_state"]
    require(isinstance(sampler_raw, Mapping) and sampler_raw.get("fit_id") == fit_id, "resume sampler binding drift")
    sampler = DeterministicEpisodeSampler.restore(episode_keys, sampler_raw)
    rng = checkpoint["rng_state"]
    require(isinstance(rng, Mapping), "resume RNG state absent")
    restore_rng_state(rng)
    require(
        model.linear.weight.dtype == torch.float64
        and sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT,
        "restored P head dtype/parameter drift",
    )
    return model, optimizer, sampler, dict(checkpoint["rolling_trace_state"])


def terminal_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "completed_updates": checkpoint["completed_updates"],
        "state_sha256": checkpoint["state_sha256"],
        "model_state_sha256": component_state_hash(checkpoint["model_state"]),
        "optimizer_state_sha256": component_state_hash(checkpoint["optimizer_state"]),
        "sampler_state_sha256": component_state_hash(checkpoint["sampler_state"]),
        "rng_state_sha256": component_state_hash(checkpoint["rng_state"]),
        "scheduler_cursor": checkpoint["scheduler_cursor"],
        "rolling_trace_state_sha256": canonical(checkpoint["rolling_trace_state"]),
    }


def build_bindings(
    *,
    authority: Mapping[str, Any],
    authority_file_sha256: str,
    manifest: ValidatedV95FitManifestV1,
    manifest_file_sha256: str,
    table_sha256: str,
    role_join_sha256: str,
    pair_receipts: Sequence[Mapping[str, Any]],
    side_receipts: Sequence[Mapping[str, Any]],
    full_recipe: Mapping[str, Any],
) -> dict[str, str]:
    bindings = {
        "authority_file_sha256": authority_file_sha256,
        "authority_logical_sha256": authority["logical_sha256"],
        "contract_file_sha256": _bound(authority, "contract")["sha256"],
        "parent_authority_v95_file_sha256": _bound(authority, "parent_authority_v95")["sha256"],
        "parent_authority_v95_logical_sha256": _bound(authority, "parent_authority_v95")["logical_sha256"],
        "v95_index_file_sha256": _bound(authority, "v95_manifest_index")["sha256"],
        "v95_index_logical_sha256": _bound(authority, "v95_manifest_index")["logical_sha256"],
        "v95_validation_file_sha256": _bound(authority, "v95_manifest_validation")["sha256"],
        "v95_validation_logical_sha256": _bound(authority, "v95_manifest_validation")["logical_sha256"],
        "fit_manifest_file_sha256": manifest_file_sha256,
        "fit_manifest_logical_sha256": manifest.manifest_logical_sha256,
        "loss_join_file_sha256": _bound(authority, "loss_join")["sha256"],
        "loss_join_logical_sha256": _bound(authority, "loss_join")["logical_sha256"],
        "loss_join_validation_file_sha256": _bound(authority, "loss_join_validation")["sha256"],
        "loss_join_validation_logical_sha256": _bound(authority, "loss_join_validation")["logical_sha256"],
        "target_free_table_sha256": table_sha256,
        "role_join_sha256": role_join_sha256,
        "ordered_pair_shards_sha256": canonical(list(pair_receipts)),
        "ordered_sidecar_shards_sha256": canonical(list(side_receipts)),
        "p_head_file_sha256": _bound(authority, "p_head")["sha256"],
        "role_free_scorer_file_sha256": _bound(authority, "role_free_scorer")["sha256"],
        "p_lock_core_file_sha256": _bound(authority, "p_lock_core")["sha256"],
        "p_selector_file_sha256": _bound(authority, "p_selector")["sha256"],
        "p_v2_adapter_file_sha256": _bound(authority, "p_v2_adapter")["sha256"],
        "fit_runtime_file_sha256": _bound(authority, "fit_runtime")["sha256"],
        "runner_file_sha256": _bound(authority, "runner")["sha256"],
        "validator_file_sha256": _bound(authority, "validator")["sha256"],
        "recipe_sha256": canonical(dict(full_recipe)),
    }
    require(set(bindings) == BINDING_FIELDS and all(isinstance(item, str) and len(item) == 64 for item in bindings.values()), "checkpoint binding construction drift")
    return bindings


def run_fit(args: argparse.Namespace) -> dict[str, Any]:
    require(type(args.fit_index) is int and 0 <= args.fit_index < 16, "--fit-index must be 0..15")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        require(torch.get_num_interop_threads() == 1, "CPU interop thread policy drift")
    require(torch.get_num_threads() == 1, "CPU thread policy drift")
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    random.seed(P_SEED)
    np.random.seed(P_SEED)
    torch.manual_seed(P_SEED)

    authority, authority_file_sha256 = latest_authority(args.authority)
    expected_output_root = (ROOT / str(authority["fit_output_root"])).resolve(strict=False)
    output_root = args.output_root.resolve(strict=False)
    require(output_root == expected_output_root, "--output-root differs from V96")
    index, manifest, raw_manifest, manifest_path, manifest_file_sha256 = validate_index_and_manifest(
        authority, args.fit_index
    )
    del index, manifest_path

    # All feature/geometry payloads are opened and sealed before loss roles.
    pair_shards, sidecar_shards, pair_receipts, side_receipts, side_records = load_routed_shards(manifest)
    table = build_and_seal_target_free_input_table(
        manifest, pair_shards=pair_shards, sidecar_shards=sidecar_shards
    )
    require(
        table.role_join_count == 0
        and len(table.inputs) == 4 * len(manifest.neutral_episodes)
        and all(bool(item.input.structure.eligible.all()) for item in table.inputs.values()),
        "target-free table seal/population/zero-H0 gate drift",
    )
    input_seal_sha256 = table.table_sha256

    _loss_join, loss_role_seal = validate_loss_join_after_table_seal(
        authority, manifest, side_records, input_seal_sha256
    )
    joined = join_training_roles_after_seal(table, manifest)
    role_join_sha256 = canonical(
        {
            "runtime_role_join_sha256": joined.role_join_sha256,
            "loss_role_seal_sha256": loss_role_seal,
        }
    )
    require(
        joined.target_free_table_sha256 == input_seal_sha256
        and len(joined.episodes) == len(manifest.neutral_episodes),
        "post-seal joined population drift",
    )
    pair_shard_read_count = len(pair_shards)
    sidecar_shard_read_count = len(sidecar_shards)

    model, optimizer = initialize_training(P_SEED)
    require(
        model.linear.weight.dtype == torch.float64
        and model.linear.weight.device.type == "cpu"
        and sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT == 16,
        "formal P head is not CPU float64/16-parameter",
    )
    # The compiler already closes every input row.  One result-blind probe per
    # routed pair shard independently compares compiled and frozen scorers,
    # avoiding an otherwise redundant all-input second pass for every fit.
    parity_probe_keys: list[tuple[int, int, str]] = []
    with torch.no_grad():
        for route in manifest.pair_routes:
            candidates = [
                key
                for key in table.inputs
                if key[0] in set(route.selected_execution_ordinals)
            ]
            require(candidates, "pair route has no deterministic parity probe")
            probe = min(candidates)
            parity_probe_keys.append(probe)
            assert_compiled_role_free_exact_parity(model, table.inputs[probe])
    require(
        len(parity_probe_keys) == len(manifest.pair_routes)
        and len(set(parity_probe_keys)) == len(parity_probe_keys),
        "per-route parity probe population drift",
    )
    # Compiled inputs own their selected tensors/masks.  Release the much
    # larger immutable shard envelopes before the 2048-update loop.
    del pair_shards, sidecar_shards, side_records, _loss_join
    gc.collect()

    episodes = joined.episodes
    episode_keys = tuple(
        PEpisodeKey(
            query_id=item.neutral.query_id,
            source_image_sha256=item.neutral.query_source_image_sha256,
            execution_ordinal=item.neutral.execution_ordinal,
        )
        for item in episodes
    )
    sampler = DeterministicEpisodeSampler(episode_keys, fit_id=manifest.fit_id)
    bindings = build_bindings(
        authority=authority,
        authority_file_sha256=authority_file_sha256,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        table_sha256=input_seal_sha256,
        role_join_sha256=role_join_sha256,
        pair_receipts=pair_receipts,
        side_receipts=side_receipts,
        full_recipe=raw_manifest["recipe"],
    )

    output_root.mkdir(parents=True, exist_ok=True)
    fit_root = output_root / manifest.fit_id
    require(not fit_root.exists(), f"immutable fit output exists: {fit_root}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{manifest.fit_id}.", suffix=".partial", dir=output_root)
    )
    primary_receipts: list[dict[str, Any]] = []
    primary_checkpoints: dict[int, dict[str, Any]] = {}
    try:
        def primary_writer(
            update: int,
            trace: Mapping[str, Any],
            live_model: SharedMultitilePHead,
            live_optimizer: torch.optim.AdamW,
            live_sampler: DeterministicEpisodeSampler,
        ) -> None:
            checkpoint = make_checkpoint(
                fit_id=manifest.fit_id,
                fit_index=args.fit_index,
                path_kind="PRIMARY",
                completed_updates=update,
                bindings=bindings,
                model=live_model,
                optimizer=live_optimizer,
                sampler=live_sampler,
                rolling_trace_state=trace,
                restored_from_checkpoint_file_sha256=None,
                restored_from_checkpoint_state_sha256=None,
            )
            path = staging / "primary" / f"checkpoint_update{update:04d}.pt"
            exclusive_torch(path, checkpoint)
            primary_receipts.append(artifact_receipt(path, staging, checkpoint))
            if update in (1024, 2048):
                primary_checkpoints[update] = checkpoint

        primary_trace = run_updates(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            sampler=sampler,
            start_update=0,
            stop_update=P_UPDATES_TOTAL,
            rolling_trace_state=initial_trace(manifest.fit_id),
            checkpoint_callback=primary_writer,
        )
        require(
            len(primary_receipts) == P_UPDATES_TOTAL // P_CHECKPOINT_INTERVAL_UPDATES == 32
            and [item["completed_updates"] for item in primary_receipts]
            == list(range(64, 2049, 64))
            and set(primary_checkpoints) == {1024, 2048},
            "primary checkpoint cadence/population drift",
        )
        primary_1024_path = staging / str(primary_receipts[15]["relative_path"])
        primary_2048_path = staging / str(primary_receipts[31]["relative_path"])
        primary_1024 = torch.load(
            primary_1024_path, map_location="cpu", weights_only=True, mmap=True
        )
        primary_2048 = torch.load(
            primary_2048_path, map_location="cpu", weights_only=True, mmap=True
        )
        validate_checkpoint(primary_1024, fit_id=manifest.fit_id, fit_index=args.fit_index, path_kind="PRIMARY", completed_updates=1024, bindings=bindings)
        validate_checkpoint(primary_2048, fit_id=manifest.fit_id, fit_index=args.fit_index, path_kind="PRIMARY", completed_updates=2048, bindings=bindings)
        require(primary_trace == primary_2048["rolling_trace_state"], "primary final trace/checkpoint drift")

        primary_1024_file_sha256 = file_sha(primary_1024_path)
        replay_model, replay_optimizer, replay_sampler, replay_trace = restore_from_checkpoint(
            primary_1024, episode_keys=episode_keys, fit_id=manifest.fit_id
        )
        replay_trace = run_updates(
            model=replay_model,
            optimizer=replay_optimizer,
            episodes=episodes,
            sampler=replay_sampler,
            start_update=1024,
            stop_update=P_UPDATES_TOTAL,
            rolling_trace_state=replay_trace,
            checkpoint_callback=None,
        )
        replay_2048 = make_checkpoint(
            fit_id=manifest.fit_id,
            fit_index=args.fit_index,
            path_kind="REPLAY",
            completed_updates=2048,
            bindings=bindings,
            model=replay_model,
            optimizer=replay_optimizer,
            sampler=replay_sampler,
            rolling_trace_state=replay_trace,
            restored_from_checkpoint_file_sha256=primary_1024_file_sha256,
            restored_from_checkpoint_state_sha256=primary_1024["state_sha256"],
        )
        replay_path = staging / "replay/checkpoint_update2048.pt"
        exclusive_torch(replay_path, replay_2048)
        replay_loaded = torch.load(
            replay_path, map_location="cpu", weights_only=True, mmap=True
        )
        validate_checkpoint(replay_loaded, fit_id=manifest.fit_id, fit_index=args.fit_index, path_kind="REPLAY", completed_updates=2048, bindings=bindings)

        primary_summary = terminal_summary(primary_2048)
        replay_summary = terminal_summary(replay_loaded)
        require(primary_summary == replay_summary, "primary/replay terminal state is not exact")
        primary_1024_receipt = artifact_receipt(primary_1024_path, staging, primary_1024)
        primary_2048_receipt = artifact_receipt(primary_2048_path, staging, primary_2048)
        replay_receipt = artifact_receipt(replay_path, staging, replay_loaded) | {
            "restored_from_checkpoint_file_sha256": primary_1024_file_sha256,
            "restored_from_checkpoint_state_sha256": primary_1024["state_sha256"],
        }
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "status": RESULT_STATUS,
            "claim_level": RESULT_CLAIM,
            "fit_id": manifest.fit_id,
            "fit_index": args.fit_index,
            "authority_sha256": authority_file_sha256,
            "authority_logical_sha256": authority["logical_sha256"],
            "fit_manifest_sha256": manifest_file_sha256,
            "fit_manifest_logical_sha256": manifest.manifest_logical_sha256,
            "input_seal_sha256": input_seal_sha256,
            "role_join_sha256": role_join_sha256,
            "primary_checkpoint_artifacts": primary_receipts,
            "replay_checkpoint_artifact": replay_receipt,
            "primary_result": primary_summary,
            "replay_result": replay_summary,
            "protected_access_audit": {
                "pair_feature_payload_read_count": pair_shard_read_count,
                "sidecar_shard_payload_read_count": sidecar_shard_read_count,
                "loss_role_read_count": 1,
                "model_load_count": 2,
                "model_forward_count": (
                    (2048 + 1024) * P_EPISODES_PER_UPDATE * 4
                    + 2 * len(parity_probe_keys)
                ),
                "model_backward_count": 2048 + 1024,
                "model_update_count": 2048 + 1024,
                "d1_score_rank_winner_read_count": 0,
                "retrieval_outcome_read_count": 0,
                "opened_read_count": 0,
                "sealed_read_count": 0,
                "C8_read_count": 0,
                "S8_read_count": 0,
                "target_insertion_count": 0,
                "p_lock_materialization_count": 0,
                "identity_specific_parameter_count": 0,
            },
            "consumable_checkpoint_authorized": False,
            "scientific_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": None,
        }
        result["logical_sha256"] = logical(result)
        require(set(result) == RESULT_FIELDS, "producer result field set drift")
        exclusive_json(staging / "result.json", result)
        os.rename(staging, fit_root)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--fit-index", required=True, type=int, choices=range(16))
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run_fit(args)
    except (PFitV2ExecutionError, PFitV2RuntimeError) as error:
        print(
            json.dumps(
                {"status": "RCDE_SR0_MT_P_V2_FIT_EXECUTION_ABORT", "error": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    print(
        json.dumps(
            {
                "status": result["status"],
                "fit_id": result["fit_id"],
                "fresh_resume_exact": (
                    result["primary_result"] == result["replay_result"]
                ),
                "result": str(args.output_root / str(result["fit_id"]) / "result.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
