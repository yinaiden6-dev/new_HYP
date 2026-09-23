#!/usr/bin/env python3
"""Independent validator for one V96 formal P-V2 fit.

The validator deliberately does not import the V2 fit producer, runner, or
fit-runtime module.  It reconstructs the target-free pair-cache/sidecar seal
from the V95 inputs, verifies that the loss role is joined only after that
seal, and then validates the uninterrupted and restored checkpoint states.

Passing this validator is necessary, but not sufficient, for a checkpoint to
be consumed later: V96 itself keeps ``consumable_checkpoint_authorized``
false, and a later aggregate authority must validate all sixteen PASS rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.dino_rcde_cw1_multitile_sr0_p_v1 import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_SCHEMA_SHA256,
    SharedMultitilePHead,
)
from rc_aslo_xf.cw1_sr0_structure_v1 import (  # noqa: E402
    enumerate_superregion_bank,
    superregion_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_compact_catalog_v1 import (  # noqa: E402
    P_DIRECTIONS,
    deserialize_role_free_pair_feature_cache,
)


AUTH_SCHEMA = "rc_current_authority_v98_20260818"
AUTH_STATUS = "RCDE_SR0_MT_P_V2_DEV_CONTINUATION_EXECUTION_AUTHORIZED"
EXPECTED_SCOPE = {
    "p_v2_exact_resume_validation": True,
    "p_v2_formal_fit_execution": True,
}
V95_INDEX_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_v2_training_manifest_index_v1_20260818"
)
V95_INDEX_STATUS = "RCDE_SR0_MT_P_V2_TRAINING_MANIFEST_INDEX_READY"
V95_MANIFEST_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_training_manifest_v1_20260818"
V95_MANIFEST_STATUS = "RCDE_SR0_MT_P_V2_TRAINING_MANIFEST_READY"
LOSS_JOIN_SCHEMA = "rc_dino_rcde_sr0_mt_loss_join_v1_20260817"
LOSS_JOIN_STATUS = "RCDE_SR0_MT_LOSS_JOIN_READY"
PAIR_SHARD_SCHEMA = (
    "rc_dino_rcde_sr0_mt_role_free_pair_feature_cache_shard_v1_20260817"
)
PAIR_SHARD_STATUS = "RCDE_SR0_MT_ROLE_FREE_PAIR_FEATURE_CACHE_SHARD_READY"
SIDECAR_SHARD_SCHEMA = (
    "rc_dino_rcde_sr0_mt_training_consumer_e0_full594_structure_sidecar_shard_v1_20260818"
)
SIDECAR_SHARD_STATUS = (
    "RCDE_SR0_MT_TRAINING_CONSUMER_E0_FULL594_STRUCTURE_SIDECAR_SHARD_READY"
)
FIT_RUNTIME_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_fit_runtime_v1_20260818"
CHECKPOINT_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_checkpoint_v1_20260818"
CHECKPOINT_STATUS = "RCDE_SR0_MT_P_V2_COMPACT_CHECKPOINT_READY"
CHECKPOINT_CLAIM = "ENGINEERING_P_V2_EXACT_RESUME_STATE_ONLY"
RESULT_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_fit_result_v1_20260818"
RESULT_STATUS = "RCDE_SR0_MT_P_V2_FIT_PRODUCER_READY"
RESULT_CLAIM = "ENGINEERING_ONE_P_V2_FIT_AND_EXACT_RESUME_ONLY"
PASS_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_fit_validation_v1_20260818"
PASS_STATUS = "RCDE_SR0_MT_P_V2_FIT_INDEPENDENT_VALIDATION_PASS"
OLD_TENSOR_HASH_NAMESPACE = "rc_dino_rcde_sr0_mt_p_runtime_v1_20260815"
PRIMARY_UPDATES = tuple(range(64, 2049, 64))
SPLIT_UPDATE = 1024
FINAL_UPDATE = 2048

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
STATE_PAYLOAD_FIELDS = (
    "completed_updates",
    "current_learning_rate",
    "scheduler_cursor",
    "model_state",
    "optimizer_state",
    "sampler_state",
    "rng_state",
    "rolling_trace_state",
)
RAW_LOGICAL_STATE_FIELDS = frozenset(
    {
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "rolling_trace_state",
        "logical_sha256",
    }
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
PRIMARY_ARTIFACT_FIELDS = frozenset(
    {
        "relative_path",
        "physical_sha256",
        "logical_sha256",
        "state_sha256",
        "completed_updates",
    }
)
REPLAY_ARTIFACT_FIELDS = frozenset(
    set(PRIMARY_ARTIFACT_FIELDS)
    | {
        "restored_from_checkpoint_file_sha256",
        "restored_from_checkpoint_state_sha256",
    }
)
TERMINAL_SUMMARY_FIELDS = frozenset(
    {
        "completed_updates",
        "state_sha256",
        "model_state_sha256",
        "optimizer_state_sha256",
        "sampler_state_sha256",
        "rng_state_sha256",
        "rolling_trace_state_sha256",
        "scheduler_cursor",
    }
)


class ValidationError(RuntimeError):
    """One authority, source, checkpoint, resume, or probe invariant failed."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def canonical(value: Any) -> str:
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
    return canonical(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def require_sha(value: Any, name: str) -> str:
    require(is_sha(value), f"{name} SHA drift")
    return value


def safe_path(relative: str, *, immutable: bool = False) -> Path:
    raw = Path(relative)
    require(
        not raw.is_absolute() and ".." not in raw.parts,
        f"unsafe relative path {relative}",
    )
    path = (ROOT / raw).resolve(strict=True)
    require(
        path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(),
        f"unsafe or absent file {relative}",
    )
    if immutable:
        require(
            (path.stat().st_mode & 0o777) == 0o444,
            f"non-immutable input {relative}",
        )
    return path


def read_json(path: Path, *, name: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe {name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    """Independent reconstruction of the frozen sidecar tensor receipt."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(OLD_TENSOR_HASH_NAMESPACE.encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_bytes(value: object) -> bytes:
    output = bytearray()

    def append(tag: bytes, payload: bytes) -> None:
        output.extend(len(tag).to_bytes(4, "big"))
        output.extend(tag)
        output.extend(len(payload).to_bytes(8, "big"))
        output.extend(payload)

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        append(
            b"tensor-meta",
            json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        append(
            b"tensor-data",
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
    elif isinstance(value, Mapping):
        rows = sorted(
            ((_state_bytes(key), _state_bytes(item)) for key, item in value.items()),
            key=lambda item: item[0],
        )
        for key, item in rows:
            append(b"key", key)
            append(b"value", item)
    elif isinstance(value, tuple):
        for item in value:
            append(b"tuple", _state_bytes(item))
    elif isinstance(value, list):
        for item in value:
            append(b"list", _state_bytes(item))
    elif value is None:
        append(b"none", b"")
    elif isinstance(value, bool):
        append(b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        append(b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        require(math.isfinite(value), "non-finite float in canonical state")
        append(b"float", value.hex().encode("ascii"))
    elif isinstance(value, str):
        append(b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        append(b"bytes", value)
    else:
        raise ValidationError(
            f"unsupported canonical state type {type(value).__name__}"
        )
    return bytes(output)


def state_sha256(value: object) -> str:
    return hashlib.sha256(_state_bytes(value)).hexdigest()


def expected_fit_ids() -> list[str]:
    output: list[str] = []
    for outer in range(1, 5):
        output.extend(
            f"P_OUTER{outer}_INNER{inner}_FIT"
            for inner in range(1, 5)
            if inner != outer
        )
        output.append(f"P_OUTER{outer}_OUTER_REFIT")
    return output


def check_authority(
    path: Path, *, fit_index: int, fit_root: Path, output: Path
) -> tuple[dict[str, Any], str]:
    authority_path = path.resolve(strict=True)
    authority_pattern = re.compile(
        r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json"
    )
    discovered = [
        (int(match.group(1)), candidate.resolve())
        for candidate in (ROOT / "registry").glob("current_authority_v*_*.json")
        if (match := authority_pattern.fullmatch(candidate.name)) is not None
        and candidate.is_file()
        and not candidate.is_symlink()
    ]
    latest_version = max((item[0] for item in discovered), default=-1)
    latest_paths = [item[1] for item in discovered if item[0] == latest_version]
    require(
        authority_path == (ROOT / "registry/current_authority_v98_20260818.json").resolve(strict=True)
        and authority_path.is_file()
        and not authority_path.is_symlink(),
        "V98 authority path drift",
    )
    require(
        latest_version == 98 and latest_paths == [authority_path],
        "V98 is not the unique latest authority",
    )
    value = read_json(authority_path, name="V98 authority")
    require(
        value.get("schema_version") == AUTH_SCHEMA
        and value.get("status") == AUTH_STATUS
        and value.get("authorized_scope") == EXPECTED_SCOPE
        and value.get("logical_sha256") == logical(value),
        "V98 authority identity/scope/hash drift",
    )
    require(type(fit_index) is int and 0 <= fit_index < 16, "fit index drift")
    fit_id = expected_fit_ids()[fit_index]
    expected_root = (ROOT / str(value.get("fit_output_root")) / fit_id).resolve(
        strict=True
    )
    expected_output = (
        ROOT
        / str(value.get("validation_output_root"))
        / fit_id
        / "result.json"
    ).resolve(strict=False)
    require(
        fit_root.resolve(strict=True) == expected_root
        and output.resolve(strict=False) == expected_output,
        "fit root or validation output drift",
    )
    for key in (
        "pair_feature_payload_read_authorized",
        "sidecar_shard_payload_read_authorized",
        "validated_loss_join_read_authorized",
        "validated_membership_read_authorized",
        "target_rival_loss_role_read_authorized",
        "model_load_authorized",
        "model_forward_authorized",
        "model_backward_authorized",
        "model_update_authorized",
        "training_authorized",
        "checkpoint_deserialization_authorized",
        "resume_state_read_authorized",
    ):
        require(value.get(key) is True, f"V96 required capability drift {key}")
    for key in (
        "d1_score_rank_winner_read_authorized",
        "retrieval_outcome_read_authorized",
        "opened_read_authorized",
        "sealed_read_authorized",
        "C8_read_authorized",
        "S8_read_authorized",
        "target_insertion_authorized",
        "p_lock_materialization_authorized",
        "donor_matching_authorized",
        "v_training_authorized",
        "heldout_scoring_authorized",
        "consumable_checkpoint_authorized",
        "automatic_stage_advance",
    ):
        require(value.get(key) is False, f"V96 forbidden capability drift {key}")
    require(
        value.get("scientific_GO_or_NO_GO") is None
        and value.get("next_authorized_stage") is None,
        "V96 claim boundary drift",
    )
    resource = value.get("resource_contract")
    require(
        resource
        == {
            "partition": "dev_cpuonly",
            "continuation_policy": "submit_next_after_validation_pass",
            "job_count": 16,
            "cpus_per_task": 1,
            "memory_megabytes": 16384,
            "walltime_seconds": 3600,
        },
        "V98 resource contract drift",
    )
    bindings = value.get("bindings")
    require(isinstance(bindings, Mapping), "V96 bindings absent")
    required_bindings = {
        "parent_authority_v95",
        "v95_manifest_index",
        "v95_manifest_validation",
        "contract",
        "loss_join",
        "loss_join_validation",
        "pair_aggregate",
        "pair_aggregate_validation",
        "sidecar_manifest",
        "p_head",
        "role_free_scorer",
        "p_lock_core",
        "p_selector",
        "p_v2_adapter",
        "fit_runtime",
        "runner",
        "validator",
        "launcher",
        "freezer",
        *(f"fit_manifest_{index:02d}" for index in range(16)),
    }
    require(required_bindings.issubset(bindings), "V96 implementation/input binding absent")
    for name, binding in bindings.items():
        require(isinstance(binding, Mapping), f"V96 binding is not a mapping {name}")
        bound = safe_path(str(binding.get("path")), immutable=name.startswith("fit_manifest_") or name in {"parent_authority_v95", "v95_manifest_index", "v95_manifest_validation", "loss_join", "loss_join_validation", "pair_aggregate", "pair_aggregate_validation", "sidecar_manifest"})
        require(
            file_sha(bound) == binding.get("sha256")
            and bound.stat().st_size == binding.get("bytes"),
            f"V96 physical binding drift {name}",
        )
        if "logical_sha256" in binding:
            payload = read_json(bound, name=f"V96 binding {name}")
            require(
                payload.get("logical_sha256")
                == binding.get("logical_sha256")
                == logical(payload),
                f"V96 logical binding drift {name}",
            )
    v95_validation_binding = bindings["v95_manifest_validation"]
    v95_validation = read_json(
        safe_path(v95_validation_binding["path"], immutable=True),
        name="V95 manifest validation",
    )
    require(
        v95_validation.get("status")
        == "RCDE_SR0_MT_P_V2_TRAINING_MANIFESTS_INDEPENDENT_VALIDATION_PASS"
        and v95_validation.get("validation_pass") is True
        and v95_validation.get("fit_count") == 16
        and v95_validation.get("inner_fit_count") == 12
        and v95_validation.get("outer_refit_count") == 4
        and v95_validation.get("total_episode_instances") == 5346
        and v95_validation.get("training_authorized") is False
        and v95_validation.get("consumable_checkpoint_authorized") is False,
        "V95 independent validation boundary drift",
    )
    return value, file_sha(authority_path)


def load_fit_manifest(
    authority: Mapping[str, Any], fit_index: int
) -> tuple[dict[str, Any], str, str]:
    index_binding = authority["bindings"]["v95_manifest_index"]
    index = read_json(safe_path(index_binding["path"], immutable=True), name="V95 index")
    rows = index.get("fits")
    require(
        index.get("schema_version") == V95_INDEX_SCHEMA
        and index.get("status") == V95_INDEX_STATUS
        and index.get("logical_sha256") == logical(index)
        and isinstance(rows, list)
        and len(rows) == 16
        and [row.get("fit_id") for row in rows] == expected_fit_ids()
        and index.get("fit_sequence_sha256") == canonical(rows),
        "V95 fit index drift",
    )
    row = rows[fit_index]
    binding = authority["bindings"][f"fit_manifest_{fit_index:02d}"]
    require(row.get("path") == binding.get("path"), "fit manifest address drift")
    manifest_path = safe_path(binding["path"], immutable=True)
    manifest = read_json(manifest_path, name="V95 fit manifest")
    require(
        manifest.get("schema_version") == V95_MANIFEST_SCHEMA
        and manifest.get("status") == V95_MANIFEST_STATUS
        and manifest.get("fit_id") == expected_fit_ids()[fit_index]
        and manifest.get("logical_sha256")
        == logical(manifest)
        == row.get("logical_sha256")
        == binding.get("logical_sha256"),
        "V95 fit manifest drift",
    )
    return manifest, file_sha(manifest_path), str(manifest["logical_sha256"])


def validate_all_scope_topology(authority: Mapping[str, Any]) -> dict[str, int]:
    """Independently prove the 12-inner/4-outer set equations."""

    manifests: dict[str, dict[str, Any]] = {}
    memberships: dict[str, dict[str, Any]] = {}
    usage: dict[int, int] = {}
    universe: set[int] | None = None
    parent_binding = authority["bindings"]["parent_authority_v95"]
    parent = read_json(
        safe_path(parent_binding["path"], immutable=True), name="V95 authority"
    )
    require(
        parent.get("schema_version") == "rc_current_authority_v95_20260818"
        and parent.get("status")
        == "RCDE_SR0_MT_P_V2_TRAINING_MANIFESTS_AUTHORIZED"
        and parent.get("logical_sha256") == logical(parent),
        "V95 parent authority drift",
    )
    for index, fit_id in enumerate(expected_fit_ids()):
        binding = authority["bindings"][f"fit_manifest_{index:02d}"]
        manifest = read_json(
            safe_path(binding["path"], immutable=True), name=f"manifest {fit_id}"
        )
        require(
            manifest.get("fit_id") == fit_id
            and manifest.get("logical_sha256") == logical(manifest),
            f"scope manifest drift {fit_id}",
        )
        membership_binding = parent["bindings"].get(f"membership_{index:02d}")
        require(
            isinstance(membership_binding, Mapping)
            and membership_binding.get("path") == manifest.get("membership_path"),
            f"scope membership binding absent {fit_id}",
        )
        membership_path = safe_path(membership_binding["path"], immutable=True)
        require(
            file_sha(membership_path)
            == membership_binding.get("sha256")
            == manifest.get("membership_file_sha256"),
            f"scope membership physical drift {fit_id}",
        )
        membership = read_json(membership_path, name=f"membership {fit_id}")
        require(
            membership.get("fit_id") == fit_id
            and membership.get("logical_sha256")
            == logical(membership)
            == membership_binding.get("logical_sha256")
            == manifest.get("membership_logical_sha256")
            and membership.get("component_proof", {}).get(
                "identity_intersection_count"
            )
            == 0
            and membership.get("component_proof", {}).get(
                "supergroup_intersection_count"
            )
            == 0,
            f"scope membership logical/disjointness drift {fit_id}",
        )
        train = {int(row["execution_ordinal"]) for row in manifest["episodes"]}
        membership_train = {
            int(row["execution_ordinal"])
            for row in membership["train_addresses"]
        }
        require(train == membership_train, f"scope train population drift {fit_id}")
        for execution in train:
            usage[execution] = usage.get(execution, 0) + 1
        manifests[fit_id] = manifest
        memberships[fit_id] = membership

    for outer in range(1, 5):
        outer_id = f"P_OUTER{outer}_OUTER_REFIT"
        outer_train = {
            int(row["execution_ordinal"])
            for row in manifests[outer_id]["episodes"]
        }
        outer_heldout = {
            int(row["execution_ordinal"])
            for row in memberships[outer_id]["heldout_addresses"]
        }
        current_universe = outer_train | outer_heldout
        require(
            not (outer_train & outer_heldout)
            and len(current_universe) == 594,
            f"outer train/heldout partition drift {outer}",
        )
        if universe is None:
            universe = current_universe
        else:
            require(current_universe == universe, "outer universes disagree")
        inner_heldout_sets: list[set[int]] = []
        for inner in range(1, 5):
            if inner == outer:
                continue
            fit_id = f"P_OUTER{outer}_INNER{inner}_FIT"
            inner_train = {
                int(row["execution_ordinal"])
                for row in manifests[fit_id]["episodes"]
            }
            inner_heldout = {
                int(row["execution_ordinal"])
                for row in memberships[fit_id]["heldout_addresses"]
            }
            require(
                inner_train == outer_train - inner_heldout,
                f"inner train set equation drift {fit_id}",
            )
            inner_heldout_sets.append(inner_heldout)
        require(
            sum(len(item) for item in inner_heldout_sets)
            == len(set().union(*inner_heldout_sets))
            and set().union(*inner_heldout_sets) == outer_train,
            f"inner-heldout OOF partition drift outer {outer}",
        )
    require(
        universe is not None
        and set(usage) == universe
        and set(usage.values()) == {9}
        and sum(usage.values()) == 5346,
        "16-scope multiplicity/universe drift",
    )
    return {
        "fit_count": 16,
        "universe_count": 594,
        "episode_instance_count": 5346,
        "per_query_training_scope_multiplicity": 9,
    }


def _decode_bool_rle(value: Mapping[str, Any]) -> torch.Tensor:
    shape = value.get("shape")
    runs = value.get("true_runs")
    require(
        value.get("flatten_order") == "ROW_MAJOR"
        and isinstance(shape, list)
        and len(shape) == 2
        and all(type(item) is int and item > 0 for item in shape)
        and isinstance(runs, list),
        "eligibility RLE schema drift",
    )
    flat = torch.zeros(math.prod(shape), dtype=torch.bool)
    cursor = 0
    for run in runs:
        require(
            isinstance(run, list)
            and len(run) == 2
            and all(type(item) is int for item in run),
            "eligibility RLE run drift",
        )
        start, length = run
        require(
            start >= cursor and length > 0 and start + length <= flat.numel(),
            "eligibility RLE range/overlap drift",
        )
        flat[start : start + length] = True
        cursor = start + length
    return flat.reshape(shape)


def _decode_compact_mask(value: Mapping[str, Any]) -> tuple[torch.Tensor, tuple[int, int]]:
    grid = value.get("grid_shape")
    runs = value.get("rle")
    require(
        isinstance(grid, list)
        and len(grid) == 2
        and all(type(item) is int and item > 0 for item in grid)
        and isinstance(runs, list),
        "compact mask schema drift",
    )
    shape = (grid[0], grid[1])
    mask = torch.zeros(math.prod(shape), dtype=torch.bool)
    cursor = 0
    for run in runs:
        require(
            isinstance(run, list)
            and len(run) == 2
            and all(type(item) is int for item in run),
            "compact mask RLE drift",
        )
        start, length = run
        require(
            start >= cursor and length > 0 and start + length <= mask.numel(),
            "compact mask RLE range/overlap drift",
        )
        mask[start : start + length] = True
        cursor = start + length
    require(
        value.get("tensor_sha256") == tensor_sha256(mask),
        "compact mask tensor hash drift",
    )
    return mask, shape


def _component_is_legal(mask: torch.Tensor, shape: tuple[int, int]) -> bool:
    image = torch.as_tensor(mask, dtype=torch.bool).reshape(shape)
    coordinates = torch.nonzero(image, as_tuple=False)
    if coordinates.shape[0] < 4:
        return False
    if int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1) < 2:
        return False
    if int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1) < 2:
        return False
    active = {(int(row), int(col)) for row, col in coordinates.tolist()}
    stack = [next(iter(active))]
    visited: set[tuple[int, int]] = set()
    while stack:
        row, col = stack.pop()
        if (row, col) in visited:
            continue
        visited.add((row, col))
        for neighbour in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbour in active and neighbour not in visited:
                stack.append(neighbour)
    return visited == active


def reconstruct_input_seals(
    authority: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, str, dict[str, int]]:
    """Independently rebuild target-free and post-seal role hashes."""

    raw_episodes = manifest.get("episodes")
    require(isinstance(raw_episodes, list) and raw_episodes, "manifest episodes absent")
    neutral = [
        {
            "execution_ordinal": row["execution_ordinal"],
            "query_id": row["query_id"],
            "query_source_image_sha256": row["query_source_image_sha256"],
            "pair_address_sha256": row["pair_address_sha256"],
        }
        for row in raw_episodes
    ]
    neutral_by_execution = {int(row["execution_ordinal"]): row for row in neutral}
    require(len(neutral_by_execution) == len(neutral), "neutral execution alias")

    # Independently reopen the bound full loss join and verify the V95 role
    # projection before reading any role while compiling the anonymous table.
    loss_binding = authority["bindings"]["loss_join"]
    loss = read_json(safe_path(loss_binding["path"], immutable=True), name="loss join")
    loss_rows = loss.get("episodes")
    require(
        loss.get("schema_version") == LOSS_JOIN_SCHEMA
        and loss.get("status") == LOSS_JOIN_STATUS
        and isinstance(loss_rows, list)
        and len(loss_rows) == 594
        and loss.get("episode_population_sha256") == canonical(loss_rows)
        and loss.get("logical_sha256") == logical(loss),
        "loss join envelope drift",
    )
    loss_by_execution = {int(row["execution_ordinal"]): row for row in loss_rows}
    require(len(loss_by_execution) == 594, "loss join execution alias")

    pair_records: dict[int, Any] = {}
    for route in manifest.get("pair_feature_shards", []):
        artifact = safe_path(route["artifact_path"], immutable=True)
        validation = safe_path(route["validation_path"], immutable=True)
        require(
            file_sha(artifact) == route.get("artifact_sha256")
            and file_sha(validation) == route.get("validation_sha256"),
            "pair shard/validation physical hash drift",
        )
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict), "pair shard payload is not a mapping")
        records = payload.get("records")
        summaries = payload.get("record_summaries")
        core = {
            key: item
            for key, item in payload.items()
            if key not in {"records", "logical_sha256"}
        }
        require(
            payload.get("schema_version") == PAIR_SHARD_SCHEMA
            and payload.get("status") == PAIR_SHARD_STATUS
            and payload.get("role_free") is True
            and payload.get("target_free") is True
            and payload.get("synthetic") is False
            and isinstance(records, list)
            and isinstance(summaries, list)
            and payload.get("logical_sha256") == canonical(core)
            and payload.get("record_population_sha256") == canonical(summaries),
            "pair shard envelope drift",
        )
        selected = set(route["selected_execution_ordinals"])
        for raw in records:
            if int(raw["execution_ordinal"]) not in selected:
                continue
            execution = int(raw["execution_ordinal"])
            address = neutral_by_execution.get(execution)
            require(
                address is not None
                and raw.get("query_id") == address["query_id"]
                and raw.get("pair_address_sha256")
                == address["pair_address_sha256"]
                and execution not in pair_records,
                "pair record/neutral address drift",
            )
            pair_records[execution] = deserialize_role_free_pair_feature_cache(
                raw["pair_feature_cache"]
            )

    side_records: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for route in manifest.get("structure_sidecar_shards", []):
        artifact = safe_path(route["artifact_path"], immutable=True)
        validation = safe_path(route["validation_path"], immutable=True)
        require(
            file_sha(artifact) == route.get("artifact_sha256")
            and file_sha(validation) == route.get("validation_sha256"),
            "sidecar shard/validation physical hash drift",
        )
        payload = read_json(artifact, name="sidecar shard")
        records = payload.get("records")
        require(
            payload.get("schema_version") == SIDECAR_SHARD_SCHEMA
            and payload.get("status") == SIDECAR_SHARD_STATUS
            and payload.get("target_free") is True
            and isinstance(records, list)
            and payload.get("record_population_sha256") == canonical(records)
            and payload.get("logical_sha256") == logical(payload),
            "sidecar shard envelope drift",
        )
        selected = set(route["selected_execution_ordinals"])
        for raw in records:
            if raw.get("execution_ordinal") not in selected:
                continue
            key = (
                int(raw["execution_ordinal"]),
                int(raw["member_ordinal"]),
                str(raw["direction"]),
            )
            require(
                key[1] in (0, 1)
                and key[2] in P_DIRECTIONS
                and key not in side_records
                and raw.get("record_sha256")
                == canonical(
                    {name: item for name, item in raw.items() if name != "record_sha256"}
                ),
                "sidecar record address/hash drift",
            )
            side_records[key] = raw

    require(
        set(pair_records) == set(neutral_by_execution),
        "pair shard selected population incomplete",
    )
    expected_blocks = {
        (execution, member, direction)
        for execution in neutral_by_execution
        for member in (0, 1)
        for direction in P_DIRECTIONS
    }
    require(set(side_records) == expected_blocks, "sidecar selected population incomplete")

    semantic_rows = []
    zero_h0_coordinates = 0
    direction_address: dict[tuple[int, int, str], str] = {}
    row_geometry_cache: dict[
        tuple[object, ...],
        tuple[list[bool], list[torch.Tensor], list[str], list[int]],
    ] = {}
    for execution in sorted(pair_records):
        cache = pair_records[execution]
        address = neutral_by_execution[execution]
        require(
            cache.pair_address_sha256 == address["pair_address_sha256"],
            "pair-cache address drift",
        )
        blocks = {(item.member_ordinal, item.direction): item for item in cache.blocks}
        require(
            set(blocks)
            == {(member, direction) for member in (0, 1) for direction in P_DIRECTIONS},
            "pair-cache four-block population drift",
        )
        query_mask_hashes: set[str] = set()
        for member in (0, 1):
            for direction in P_DIRECTIONS:
                block = blocks[(member, direction)]
                side = side_records[(execution, member, direction)]
                start, stop = block.feature_offset, block.feature_offset + block.coordinate_count
                features = cache.features[start:stop].reshape(
                    block.root_count, block.action_count, FEATURE_DIM
                )
                eligibility = cache.eligibility[start:stop].reshape(
                    block.root_count, block.action_count
                )
                decoded = _decode_bool_rle(side["eligibility_rle"])
                require(
                    side.get("query_id") == address["query_id"]
                    and side.get("query_source_image_sha256")
                    == address["query_source_image_sha256"]
                    and side.get("pair_address_sha256")
                    == address["pair_address_sha256"]
                    and side.get("candidate_key") == block.candidate_key
                    and side.get("source_population_sha256")
                    == block.source_population_sha256
                    and side.get("cache_sha256") == cache.cache_sha256
                    and side.get("cache_block_sha256") == block.block_sha256
                    and side.get("root_count") == block.root_count
                    and side.get("action_count") == block.action_count
                    and side.get("coordinate_count") == block.coordinate_count
                    and torch.equal(decoded, eligibility)
                    and side.get("cache_feature_tensor_sha256")
                    == tensor_sha256(features)
                    and side.get("cache_eligibility_sha256")
                    == tensor_sha256(eligibility),
                    "pair-cache/sidecar block seal drift",
                )
                require(bool(eligibility.all()), "formal selected population contains geometry-H0")
                zero_h0_coordinates += int((~eligibility).sum())
                query_mask_hashes.add(str(side["deployment_query_root_masks_sha256"]))
                raw_query_masks = side.get("deployment_query_root_masks")
                colnomic_grid = side.get("colnomic_query_grid_shape")
                deployment_grid = side.get("deployment_query_grid_shape")
                require(
                    isinstance(raw_query_masks, list)
                    and len(raw_query_masks) == block.root_count
                    and side.get("deployment_query_root_masks_sha256")
                    == canonical(raw_query_masks),
                    "sidecar query-root mask population drift",
                )
                require(
                    isinstance(colnomic_grid, list)
                    and len(colnomic_grid) == 2
                    and all(type(item) is int and item > 0 for item in colnomic_grid),
                    "ColNomic query grid drift",
                )
                require(
                    isinstance(deployment_grid, list)
                    and len(deployment_grid) == 2
                    and all(type(item) is int and item > 0 for item in deployment_grid),
                    "query deployment grid drift",
                )
                root_available = tuple(
                    bool(eligibility[root].any()) for root in range(block.root_count)
                )
                geometry_key = (
                    side["query_id"],
                    side["query_geometry_sha256"],
                    tuple(colnomic_grid),
                    tuple(deployment_grid),
                    side["deployment_query_root_masks_sha256"],
                    root_available,
                )
                cached_geometry = row_geometry_cache.get(geometry_key)
                if cached_geometry is None:
                    decoded_masks = [
                        _decode_compact_mask(item) for item in raw_query_masks
                    ]
                    query_shapes = {shape for _mask, shape in decoded_masks}
                    require(len(query_shapes) == 1, "query-root grid drift")
                    query_grid = next(iter(query_shapes))
                    require(
                        list(query_grid) == deployment_grid,
                        "query deployment grid drift",
                    )
                    query_masks = [mask for mask, _shape in decoded_masks]
                    regions = tuple(
                        enumerate_superregion_bank(
                            (colnomic_grid[0], colnomic_grid[1]),
                            include_r0_control=False,
                        )
                    )
                    require(
                        regions
                        and int(regions[0].tile_weights.numel()) == block.root_count,
                        "superregion bank/root population drift",
                    )
                    row_ready = []
                    row_weights = []
                    for region in regions:
                        masks = [
                            query_masks[root]
                            for root in region.contributing_root_ordinals
                            if root_available[root]
                        ]
                        legal = len(masks) >= 2
                        if legal:
                            union = torch.stack(masks).any(dim=0)
                            legal = _component_is_legal(
                                union, query_grid
                            ) and int(union.sum()) > max(
                                int(mask.sum()) for mask in masks
                            )
                        row_ready.append(bool(legal))
                        row_weights.append(
                            region.aggregation_weights.detach()
                            .to(torch.float64)
                            .cpu()
                            .contiguous()
                        )
                    cached_geometry = (
                        row_ready,
                        row_weights,
                        [superregion_sha256(region) for region in regions],
                        [region.radius for region in regions],
                    )
                    row_geometry_cache[geometry_key] = cached_geometry
                row_ready, row_weights, row_hashes, row_radii = cached_geometry
                neutral_key_payload = {
                    "query_id": side["query_id"],
                    "query_source_image_sha256": side["query_source_image_sha256"],
                    "execution_ordinal": execution,
                    "candidate_key": side["candidate_key"],
                    "candidate_reference_source_sha256": side[
                        "candidate_reference_source_sha256"
                    ],
                    "direction": direction,
                }
                direction_address[(execution, member, direction)] = canonical(
                    neutral_key_payload
                )
                semantic = canonical(
                    {
                        "schema_version": FIT_RUNTIME_SCHEMA,
                        "execution_ordinal": execution,
                        "member_ordinal": member,
                        "direction": direction,
                        "candidate_key": block.candidate_key,
                        "pair_address_sha256": address["pair_address_sha256"],
                        "cache_sha256": cache.cache_sha256,
                        "block_sha256": block.block_sha256,
                        "structure_record_sha256": side["record_sha256"],
                        "source_population_sha256": block.source_population_sha256,
                        "feature_tensor_sha256": tensor_sha256(features),
                        "eligibility_tensor_sha256": tensor_sha256(eligibility),
                        "row_ready": row_ready,
                        "row_weight_sha256": [
                            tensor_sha256(weight) for weight in row_weights
                        ],
                        "row_hashes": row_hashes,
                        "row_radii": row_radii,
                    }
                )
                semantic_rows.append(
                    {
                        "execution_ordinal": execution,
                        "member_ordinal": member,
                        "direction": direction,
                        "semantic_sha256": semantic,
                    }
                )
        require(
            len(query_mask_hashes) == 1,
            "query-root masks are candidate/direction dependent",
        )
    require(
        len(row_geometry_cache) == len(neutral_by_execution),
        "query geometry was not candidate/direction invariant",
    )

    neutral_sha = canonical(neutral)
    table_sha = canonical(
        {
            "schema_version": FIT_RUNTIME_SCHEMA,
            "fit_id": manifest["fit_id"],
            "neutral_population_sha256": neutral_sha,
            "rows": sorted(
                semantic_rows,
                key=lambda row: (
                    row["execution_ordinal"],
                    row["member_ordinal"],
                    row["direction"],
                ),
            ),
            "role_fields_serialized": False,
            "geometry_h0_coordinate_count": 0,
            "role_join_count": 0,
        }
    )

    roles = []
    selected_loss_join_sha256: list[str] = []
    for row in raw_episodes:
        execution = int(row["execution_ordinal"])
        source = loss_by_execution.get(execution)
        require(
            source is not None
            and source.get("loss_join_sha256")
            == canonical(
                {name: item for name, item in source.items() if name != "loss_join_sha256"}
            )
            and all(source.get(name) == row.get(name) for name in (
                "execution_ordinal",
                "query_id",
                "query_source_image_sha256",
                "pair_address_sha256",
                "target_member_ordinal",
                "target_candidate_key",
                "rival_member_ordinal",
                "rival_candidate_key",
            )),
            "manifest/loss role join drift",
        )
        for role_name in ("target", "rival"):
            member = int(row[f"{role_name}_member_ordinal"])
            expected_addresses = {
                direction: direction_address[(execution, member, direction)]
                for direction in P_DIRECTIONS
            }
            require(
                source.get(f"{role_name}_direction_addresses")
                == expected_addresses,
                f"loss join {role_name} direction-address drift",
            )
        selected_loss_join_sha256.append(str(source["loss_join_sha256"]))
        roles.append(
            {
                "execution_ordinal": execution,
                "target_member_ordinal": row["target_member_ordinal"],
                "target_candidate_key": row["target_candidate_key"],
                "rival_member_ordinal": row["rival_member_ordinal"],
                "rival_candidate_key": row["rival_candidate_key"],
            }
        )
    runtime_role_join_sha256 = canonical(
        {"target_free_table_sha256": table_sha, "roles": roles}
    )
    loss_role_seal_sha256 = canonical(
        {
            "target_free_table_sha256": table_sha,
            "loss_join_file_sha256": loss_binding["sha256"],
            "loss_join_logical_sha256": loss_binding["logical_sha256"],
            "selected_loss_join_sha256": selected_loss_join_sha256,
        }
    )
    role_sha = canonical(
        {
            "runtime_role_join_sha256": runtime_role_join_sha256,
            "loss_role_seal_sha256": loss_role_seal_sha256,
        }
    )
    return table_sha, role_sha, {
        "episode_count": len(raw_episodes),
        "directional_block_count": len(semantic_rows),
        "geometry_h0_coordinate_count": zero_h0_coordinates,
    }


def expected_checkpoint_bindings(
    *,
    authority: Mapping[str, Any],
    authority_sha256: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    manifest_logical_sha256: str,
    input_seal_sha256: str,
    role_join_sha256: str,
) -> dict[str, str]:
    bindings = authority["bindings"]
    pair_receipts = [
        {
            "shard_ordinal": row["shard_ordinal"],
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
            "validation_path": row["validation_path"],
            "validation_sha256": row["validation_sha256"],
        }
        for row in manifest["pair_feature_shards"]
    ]
    side_receipts = [
        {
            "shard_ordinal": row["shard_ordinal"],
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
            "validation_path": row["validation_path"],
            "validation_sha256": row["validation_sha256"],
        }
        for row in manifest["structure_sidecar_shards"]
    ]
    return {
        "authority_file_sha256": authority_sha256,
        "authority_logical_sha256": require_sha(
            authority["logical_sha256"], "authority logical"
        ),
        "contract_file_sha256": require_sha(
            bindings["contract"]["sha256"], "fit execution contract"
        ),
        "parent_authority_v95_file_sha256": require_sha(
            bindings["parent_authority_v95"]["sha256"], "V95 authority file"
        ),
        "parent_authority_v95_logical_sha256": require_sha(
            bindings["parent_authority_v95"]["logical_sha256"],
            "V95 authority logical",
        ),
        "v95_index_file_sha256": require_sha(
            bindings["v95_manifest_index"]["sha256"], "V95 index file"
        ),
        "v95_index_logical_sha256": require_sha(
            bindings["v95_manifest_index"]["logical_sha256"],
            "V95 index logical",
        ),
        "v95_validation_file_sha256": require_sha(
            bindings["v95_manifest_validation"]["sha256"],
            "V95 validation file",
        ),
        "v95_validation_logical_sha256": require_sha(
            bindings["v95_manifest_validation"]["logical_sha256"],
            "V95 validation logical",
        ),
        "fit_manifest_file_sha256": manifest_sha256,
        "fit_manifest_logical_sha256": manifest_logical_sha256,
        "loss_join_file_sha256": require_sha(
            bindings["loss_join"]["sha256"], "loss join file"
        ),
        "loss_join_logical_sha256": require_sha(
            bindings["loss_join"]["logical_sha256"], "loss join logical"
        ),
        "loss_join_validation_file_sha256": require_sha(
            bindings["loss_join_validation"]["sha256"],
            "loss join validation file",
        ),
        "loss_join_validation_logical_sha256": require_sha(
            bindings["loss_join_validation"]["logical_sha256"],
            "loss join validation logical",
        ),
        "target_free_table_sha256": input_seal_sha256,
        "role_join_sha256": role_join_sha256,
        "ordered_pair_shards_sha256": canonical(pair_receipts),
        "ordered_sidecar_shards_sha256": canonical(side_receipts),
        "recipe_sha256": canonical(manifest["recipe"]),
        "p_head_file_sha256": require_sha(bindings["p_head"]["sha256"], "P head"),
        "role_free_scorer_file_sha256": require_sha(
            bindings["role_free_scorer"]["sha256"], "role-free scorer"
        ),
        "p_lock_core_file_sha256": require_sha(
            bindings["p_lock_core"]["sha256"], "V2 core"
        ),
        "p_selector_file_sha256": require_sha(
            bindings["p_selector"]["sha256"], "V2 selector"
        ),
        "p_v2_adapter_file_sha256": require_sha(
            bindings["p_v2_adapter"]["sha256"], "V2 adapter"
        ),
        "fit_runtime_file_sha256": require_sha(
            bindings["fit_runtime"]["sha256"], "fit runtime"
        ),
        "runner_file_sha256": require_sha(bindings["runner"]["sha256"], "fit runner"),
        "validator_file_sha256": require_sha(
            bindings["validator"]["sha256"], "fit validator"
        ),
    }


def _load_torch(path: Path, *, name: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe {name}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(value, dict), f"{name} is not a tensor-safe mapping")
    return value


def learning_rate_at(completed_update: int) -> float:
    require(1 <= completed_update <= FINAL_UPDATE, "scheduler cursor outside recipe")
    if completed_update <= 128:
        return 0.0003 * completed_update / 128
    progress = (completed_update - 128) / (FINAL_UPDATE - 128)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return 0.00003 + (0.0003 - 0.00003) * cosine


def checkpoint_logical_sha256(value: Mapping[str, Any]) -> str:
    """Reconstruct the runner's metadata-only checkpoint logical receipt."""

    payload = {
        key: item
        for key, item in value.items()
        if key not in RAW_LOGICAL_STATE_FIELDS
    }
    payload.update(
        {
            "model_state_sha256": state_sha256(value["model_state"]),
            "optimizer_state_sha256": state_sha256(value["optimizer_state"]),
            "sampler_state_sha256": state_sha256(value["sampler_state"]),
            "rng_state_sha256": state_sha256(value["rng_state"]),
            "rolling_trace_state_sha256": canonical(value["rolling_trace_state"]),
        }
    )
    return canonical(payload)


def validate_checkpoint(
    value: Mapping[str, Any],
    *,
    fit_id: str,
    fit_index: int,
    path_kind: str,
    completed_updates: int,
    expected_bindings: Mapping[str, Any],
    restored_file_sha256: str | None,
    restored_state_sha256: str | None,
    episode_count: int | None = None,
) -> dict[str, Any]:
    """Validate one checkpoint independently from the fit runtime/runner."""

    require(set(value) == CHECKPOINT_FIELDS, "checkpoint exact field set drift")
    require(
        value.get("schema_version") == CHECKPOINT_SCHEMA
        and value.get("status") == CHECKPOINT_STATUS
        and value.get("claim_level") == CHECKPOINT_CLAIM
        and value.get("fit_id") == fit_id
        and value.get("fit_index") == fit_index
        and value.get("path_kind") == path_kind
        and value.get("completed_updates") == completed_updates
        and value.get("consumable_checkpoint_authorized") is False,
        "checkpoint envelope drift",
    )
    require(path_kind in {"PRIMARY", "REPLAY"}, "checkpoint path kind drift")
    require(
        value.get("restored_from_checkpoint_file_sha256")
        == restored_file_sha256
        and value.get("restored_from_checkpoint_state_sha256")
        == restored_state_sha256,
        "checkpoint restore lineage drift",
    )
    bindings = value.get("bindings")
    require(
        isinstance(bindings, Mapping)
        and set(bindings) == set(expected_bindings)
        and dict(bindings) == dict(expected_bindings),
        "checkpoint binding closure drift",
    )
    require(
        value.get("scheduler_cursor") == completed_updates
        and isinstance(value.get("current_learning_rate"), float)
        and value.get("current_learning_rate") == learning_rate_at(completed_updates),
        "checkpoint scheduler/LR drift",
    )
    for name in (
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "rolling_trace_state",
    ):
        require(isinstance(value.get(name), Mapping), f"checkpoint {name} absent")
    state_payload = {name: value[name] for name in STATE_PAYLOAD_FIELDS}
    require(
        value.get("state_sha256") == state_sha256(state_payload),
        "checkpoint canonical state hash drift",
    )
    require(
        value.get("logical_sha256") == checkpoint_logical_sha256(value),
        "checkpoint logical hash drift",
    )
    trace = value["rolling_trace_state"]
    require(
        set(trace)
        == {
            "count",
            "chain_sha256",
            "window_start",
            "window",
            "window_sha256",
        }
        and trace.get("count") == completed_updates
        and isinstance(trace.get("window"), list)
        and 0 < len(trace["window"]) <= 64
        and trace.get("window_start")
        == completed_updates - len(trace["window"]) + 1
        and trace.get("window_sha256") == canonical(trace["window"])
        and is_sha(trace.get("chain_sha256")),
        "checkpoint rolling trace drift",
    )
    if episode_count is not None:
        require(episode_count > 0, "checkpoint episode population is empty")
        trace_entry_fields = {
            "completed_update",
            "episode_indices",
            "loss_float64_hex",
            "loss_tensor_sha256",
            "target_utility_population_sha256",
            "rival_utility_population_sha256",
            "margin_population_sha256",
            "raw_gradient_sha256",
            "clipped_gradient_sha256",
            "parameter_sha256",
            "optimizer_state_sha256",
            "next_sampler_state_sha256",
            "learning_rate_float64_hex",
        }
        for offset, entry in enumerate(trace["window"]):
            update = trace["window_start"] + offset
            indices = entry.get("episode_indices") if isinstance(entry, Mapping) else None
            require(
                isinstance(entry, Mapping)
                and set(entry) == trace_entry_fields
                and entry.get("completed_update") == update
                and isinstance(indices, list)
                and len(indices) == 4
                and all(type(index) is int and 0 <= index < episode_count for index in indices)
                and math.isfinite(
                    float.fromhex(str(entry.get("loss_float64_hex")))
                )
                and entry.get("learning_rate_float64_hex")
                == learning_rate_at(update).hex()
                and all(
                    is_sha(entry.get(name))
                    for name in trace_entry_fields
                    if name.endswith("sha256")
                ),
                "checkpoint rolling trace entry drift",
            )
        terminal_entry = trace["window"][-1]
        model_state = value["model_state"]
        require(
            isinstance(model_state, Mapping)
            and "linear.weight" in model_state
            and terminal_entry["parameter_sha256"]
            == tensor_sha256(model_state["linear.weight"])
            and terminal_entry["optimizer_state_sha256"]
            == state_sha256(value["optimizer_state"])
            and terminal_entry["next_sampler_state_sha256"]
            == state_sha256(value["sampler_state"]),
            "checkpoint terminal trace/state closure drift",
        )
    sampler = value["sampler_state"]
    require(
        sampler.get("fit_id") == fit_id
        and sampler.get("namespace") == "RCDE_SR0_MT_P_QUERY_ORDER_V1"
        and type(sampler.get("epoch")) is int
        and type(sampler.get("position")) is int
        and is_sha(sampler.get("order_sha256")),
        "checkpoint sampler state drift",
    )
    rng = value["rng_state"]
    require(
        set(rng)
        == {
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
        }
        and tuple(rng.get("torch_cuda_rng_states", ())) == (),
        "checkpoint RNG/device contract drift",
    )
    probe_sha = fixed_probe(value["model_state"])
    return {
        "state_sha256": value["state_sha256"],
        "logical_sha256": value["logical_sha256"],
        "model_state_sha256": state_sha256(value["model_state"]),
        "optimizer_state_sha256": state_sha256(value["optimizer_state"]),
        "sampler_state_sha256": state_sha256(value["sampler_state"]),
        "rng_state_sha256": state_sha256(value["rng_state"]),
        "rolling_trace_state_sha256": canonical(value["rolling_trace_state"]),
        "scheduler_cursor": value["scheduler_cursor"],
        "current_learning_rate_hex": value["current_learning_rate"].hex(),
        "fixed_probe_sha256": probe_sha,
    }


def _artifact_path(fit_root: Path, relative: str) -> Path:
    raw = Path(relative)
    require(
        not raw.is_absolute() and ".." not in raw.parts,
        f"unsafe checkpoint relative path {relative}",
    )
    path = (fit_root / raw).resolve(strict=True)
    require(
        path.is_relative_to(fit_root.resolve(strict=True))
        and path.is_file()
        and not path.is_symlink()
        and (path.stat().st_mode & 0o777) == 0o444,
        f"unsafe or non-immutable checkpoint {relative}",
    )
    return path


def _validate_artifact_summary(
    value: Mapping[str, Any],
    *,
    fields: frozenset[str],
    fit_root: Path,
    completed_updates: int,
) -> tuple[Path, dict[str, Any]]:
    require(set(value) == fields, "checkpoint artifact summary field drift")
    require(
        value.get("completed_updates") == completed_updates,
        "checkpoint artifact update drift",
    )
    path = _artifact_path(fit_root, str(value.get("relative_path")))
    require(
        value.get("physical_sha256") == file_sha(path),
        "checkpoint artifact physical hash drift",
    )
    checkpoint = _load_torch(path, name=f"checkpoint@{completed_updates}")
    require(
        value.get("logical_sha256") == checkpoint.get("logical_sha256")
        and value.get("state_sha256") == checkpoint.get("state_sha256"),
        "checkpoint artifact logical/state summary drift",
    )
    return path, checkpoint


def _validate_trace_chain(
    checkpoints: Sequence[Mapping[str, Any]], *, initial_chain: str | None = None
) -> str:
    previous = initial_chain
    for checkpoint in checkpoints:
        trace = checkpoint["rolling_trace_state"]
        expected = canonical(
            {
                "previous_chain_sha256": previous,
                "window_start": trace["window_start"],
                "window_sha256": trace["window_sha256"],
            }
        )
        require(trace.get("chain_sha256") == expected, "rolling trace chain drift")
        expected_updates = list(
            range(trace["window_start"], checkpoint["completed_updates"] + 1)
        )
        require(
            [row.get("completed_update") for row in trace["window"]]
            == expected_updates,
            "rolling trace update sequence drift",
        )
        previous = expected
    require(previous is not None, "empty rolling trace chain")
    return previous


def _terminal_summary(
    *,
    path_kind: str,
    artifact: Mapping[str, Any],
    checkpoint_summary: Mapping[str, Any],
    restored_file_sha256: str | None,
    restored_state_sha256: str | None,
) -> dict[str, Any]:
    del path_kind, artifact, restored_file_sha256, restored_state_sha256
    return {
        "completed_updates": FINAL_UPDATE,
        "state_sha256": checkpoint_summary["state_sha256"],
        "model_state_sha256": checkpoint_summary["model_state_sha256"],
        "optimizer_state_sha256": checkpoint_summary["optimizer_state_sha256"],
        "sampler_state_sha256": checkpoint_summary["sampler_state_sha256"],
        "rng_state_sha256": checkpoint_summary["rng_state_sha256"],
        "rolling_trace_state_sha256": checkpoint_summary[
            "rolling_trace_state_sha256"
        ],
        "scheduler_cursor": checkpoint_summary["scheduler_cursor"],
    }


def validate_fit_outputs(
    *,
    authority: Mapping[str, Any],
    authority_sha256: str,
    fit_index: int,
    fit_root: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    manifest_logical_sha256: str,
    independently_reconstructed_input_seal_sha256: str,
    independently_reconstructed_role_join_sha256: str,
    input_counts: Mapping[str, int],
    scope_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Validate all 32 primary checkpoints and the exact restored final."""

    result_path = fit_root / "result.json"
    result = read_json(result_path, name="formal fit result")
    require(set(result) == RESULT_FIELDS, "formal fit result exact field set drift")
    fit_id = expected_fit_ids()[fit_index]
    require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == RESULT_STATUS
        and result.get("claim_level") == RESULT_CLAIM
        and result.get("fit_id") == fit_id
        and result.get("fit_index") == fit_index
        and result.get("authority_sha256") == authority_sha256
        and result.get("authority_logical_sha256")
        == authority.get("logical_sha256")
        and result.get("fit_manifest_sha256") == manifest_sha256
        and result.get("fit_manifest_logical_sha256")
        == manifest_logical_sha256
        and result.get("input_seal_sha256")
        == independently_reconstructed_input_seal_sha256
        and result.get("role_join_sha256")
        == independently_reconstructed_role_join_sha256
        and result.get("consumable_checkpoint_authorized") is False
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("automatic_stage_advance") is False
        and result.get("next_authorized_stage") is None
        and result.get("logical_sha256") == logical(result),
        "formal fit result envelope/lineage drift",
    )
    expected_bindings = expected_checkpoint_bindings(
        authority=authority,
        authority_sha256=authority_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        manifest_logical_sha256=manifest_logical_sha256,
        input_seal_sha256=independently_reconstructed_input_seal_sha256,
        role_join_sha256=independently_reconstructed_role_join_sha256,
    )

    artifacts = result.get("primary_checkpoint_artifacts")
    require(
        isinstance(artifacts, list) and len(artifacts) == len(PRIMARY_UPDATES),
        "primary checkpoint population drift",
    )
    primary_values: list[dict[str, Any]] = []
    primary_summaries: list[dict[str, Any]] = []
    primary_paths: list[Path] = []
    for artifact, update in zip(artifacts, PRIMARY_UPDATES, strict=True):
        require(
            isinstance(artifact, Mapping)
            and artifact.get("relative_path")
            == f"primary/checkpoint_update{update:04d}.pt",
            f"primary checkpoint address drift update {update}",
        )
        path, checkpoint = _validate_artifact_summary(
            artifact,
            fields=PRIMARY_ARTIFACT_FIELDS,
            fit_root=fit_root,
            completed_updates=update,
        )
        summary = validate_checkpoint(
            checkpoint,
            fit_id=fit_id,
            fit_index=fit_index,
            path_kind="PRIMARY",
            completed_updates=update,
            expected_bindings=expected_bindings,
            restored_file_sha256=None,
            restored_state_sha256=None,
            episode_count=len(manifest["episodes"]),
        )
        primary_paths.append(path)
        primary_values.append(checkpoint)
        primary_summaries.append(summary)
    primary_chain = _validate_trace_chain(primary_values)
    split_index = PRIMARY_UPDATES.index(SPLIT_UPDATE)
    split_path = primary_paths[split_index]
    split_value = primary_values[split_index]
    primary_final = primary_values[-1]
    primary_final_summary = primary_summaries[-1]

    replay_artifact = result.get("replay_checkpoint_artifact")
    require(
        isinstance(replay_artifact, Mapping)
        and replay_artifact.get("relative_path")
        == "replay/checkpoint_update2048.pt"
        and replay_artifact.get("restored_from_checkpoint_file_sha256")
        == file_sha(split_path)
        and replay_artifact.get("restored_from_checkpoint_state_sha256")
        == split_value["state_sha256"],
        "replay artifact restore receipt drift",
    )
    _replay_path, replay = _validate_artifact_summary(
        replay_artifact,
        fields=REPLAY_ARTIFACT_FIELDS,
        fit_root=fit_root,
        completed_updates=FINAL_UPDATE,
    )
    replay_summary = validate_checkpoint(
        replay,
        fit_id=fit_id,
        fit_index=fit_index,
        path_kind="REPLAY",
        completed_updates=FINAL_UPDATE,
        expected_bindings=expected_bindings,
        restored_file_sha256=file_sha(split_path),
        restored_state_sha256=split_value["state_sha256"],
        episode_count=len(manifest["episodes"]),
    )
    # The replay begins with the primary 1024 chain and must extend it with
    # the same update 1025..2048 trace as the uninterrupted primary path.
    require(
        replay["rolling_trace_state"] == primary_final["rolling_trace_state"],
        "fresh/replay rolling trace final drift",
    )
    replay_chain = replay["rolling_trace_state"]["chain_sha256"]
    require(replay_chain == primary_chain, "fresh/replay trace chain drift")

    exact_keys = (
        "state_sha256",
        "model_state_sha256",
        "optimizer_state_sha256",
        "sampler_state_sha256",
        "rng_state_sha256",
        "rolling_trace_state_sha256",
        "scheduler_cursor",
        "current_learning_rate_hex",
        "fixed_probe_sha256",
    )
    require(
        all(primary_final_summary[key] == replay_summary[key] for key in exact_keys),
        "fresh/replay terminal state is not exact",
    )
    expected_primary_terminal = _terminal_summary(
        path_kind="PRIMARY",
        artifact=artifacts[-1],
        checkpoint_summary=primary_final_summary,
        restored_file_sha256=None,
        restored_state_sha256=None,
    )
    expected_replay_terminal = _terminal_summary(
        path_kind="REPLAY",
        artifact=replay_artifact,
        checkpoint_summary=replay_summary,
        restored_file_sha256=file_sha(split_path),
        restored_state_sha256=split_value["state_sha256"],
    )
    require(
        isinstance(result.get("primary_result"), Mapping)
        and set(result["primary_result"]) == TERMINAL_SUMMARY_FIELDS
        and result["primary_result"] == expected_primary_terminal
        and isinstance(result.get("replay_result"), Mapping)
        and set(result["replay_result"]) == TERMINAL_SUMMARY_FIELDS
        and result["replay_result"] == expected_replay_terminal,
        "terminal result summary drift",
    )

    access = result.get("protected_access_audit")
    require(isinstance(access, Mapping), "protected access audit absent")
    expected_access = {
        "pair_feature_payload_read_count": len(manifest["pair_feature_shards"]),
        "sidecar_shard_payload_read_count": len(
            manifest["structure_sidecar_shards"]
        ),
        "loss_role_read_count": 1,
        "model_load_count": 2,
        "model_forward_count": (
            (2048 + 1024) * 4 * 4
            + 2 * len(manifest["pair_feature_shards"])
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
    }
    require(
        dict(access) == expected_access, "protected access audit field/count drift"
    )

    return {
        "schema_version": PASS_SCHEMA,
        "status": PASS_STATUS,
        "claim_level": "INDEPENDENT_ONE_P_V2_FIT_AND_EXACT_RESUME_VALIDATION_ONLY",
        "validation_pass": True,
        "fit_id": fit_id,
        "fit_index": fit_index,
        "authority_sha256": authority_sha256,
        "authority_logical_sha256": authority["logical_sha256"],
        "fit_manifest_sha256": manifest_sha256,
        "fit_manifest_logical_sha256": manifest_logical_sha256,
        "producer_result_sha256": file_sha(result_path),
        "producer_result_logical_sha256": result["logical_sha256"],
        "independently_reconstructed_input_seal_sha256": independently_reconstructed_input_seal_sha256,
        "independently_reconstructed_role_join_sha256": independently_reconstructed_role_join_sha256,
        "episode_count": input_counts["episode_count"],
        "directional_block_count": input_counts["directional_block_count"],
        "geometry_h0_coordinate_count": input_counts[
            "geometry_h0_coordinate_count"
        ],
        "validated_scope_fit_count": scope_counts["fit_count"],
        "validated_scope_universe_count": scope_counts["universe_count"],
        "validated_scope_episode_instance_count": scope_counts[
            "episode_instance_count"
        ],
        "per_query_training_scope_multiplicity": scope_counts[
            "per_query_training_scope_multiplicity"
        ],
        "primary_checkpoint_count": len(primary_values),
        "primary_split_checkpoint_physical_sha256": file_sha(split_path),
        "primary_split_checkpoint_state_sha256": split_value["state_sha256"],
        "primary_final_checkpoint_physical_sha256": artifacts[-1][
            "physical_sha256"
        ],
        "replay_final_checkpoint_physical_sha256": replay_artifact[
            "physical_sha256"
        ],
        "fresh_resume_exact": True,
        "fixed_target_free_probe_sha256": primary_final_summary[
            "fixed_probe_sha256"
        ],
        "checkpoint_eligible_for_later_aggregate": True,
        "consumable_checkpoint_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }


def fixed_probe(model_state: Mapping[str, Any]) -> str:
    """Strictly load the shared head and hash a fixed target-free probe."""

    model = SharedMultitilePHead()
    model.load_state_dict(model_state, strict=True)
    require(
        sum(parameter.numel() for parameter in model.parameters()) == FEATURE_DIM
        and model.linear.bias is None
        and model.linear.weight.dtype == torch.float64,
        "checkpoint head architecture/dtype drift",
    )
    probe = torch.stack(
        (
            torch.zeros(FEATURE_DIM, dtype=torch.float64),
            torch.arange(FEATURE_DIM, dtype=torch.float64) / FEATURE_DIM,
            -torch.arange(FEATURE_DIM, dtype=torch.float64) / FEATURE_DIM,
            torch.eye(FEATURE_DIM, dtype=torch.float64)[0]
            - torch.eye(FEATURE_DIM, dtype=torch.float64)[-1],
        )
    )
    with torch.no_grad():
        output = model(probe)
    require(output.shape == (4,) and bool(torch.isfinite(output).all()), "fixed probe failed")
    return tensor_sha256(output)


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable validation output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(0o444)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--fit-index", type=int, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authority, authority_sha = check_authority(
        args.authority,
        fit_index=args.fit_index,
        fit_root=args.fit_root,
        output=args.output,
    )
    manifest, manifest_sha, manifest_logical = load_fit_manifest(
        authority, args.fit_index
    )
    scope_counts = validate_all_scope_topology(authority)
    input_seal, role_join, counts = reconstruct_input_seals(authority, manifest)
    result = validate_fit_outputs(
        authority=authority,
        authority_sha256=authority_sha,
        fit_index=args.fit_index,
        fit_root=args.fit_root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        manifest_logical_sha256=manifest_logical,
        independently_reconstructed_input_seal_sha256=input_seal,
        independently_reconstructed_role_join_sha256=role_join,
        input_counts=counts,
        scope_counts=scope_counts,
    )
    result["logical_sha256"] = logical(result)
    atomic(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "fit_id": result["fit_id"],
                "logical_sha256": result["logical_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
