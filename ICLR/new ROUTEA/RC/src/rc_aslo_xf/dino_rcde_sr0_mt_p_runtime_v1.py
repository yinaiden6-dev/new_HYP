"""Formal SR0-MT runtime primitives for the shared ColNomic P head.

This module closes the gap between the frozen P mathematical core and a
deployable nested-cross-fit execution.  It deliberately keeps three objects
separate:

* a target-free, complete candidate-axis feature ledger;
* a label-joined fit episode that only addresses two already materialized
  candidates; and
* an erased P lock that contains masks and provenance, never scores or labels.

The P core scores the complete CW1 r1--r4 bank on the ColNomic raster.  Formal
training and locking additionally apply the canonical ColNomic-to-DINO
geometry closure.  A selected action/row which cannot be executed on the DINO
raster is exact H0 in both paths.  There is no second-best fallback.

No model or natural payload is loaded at import time.  All functions operate on
explicit, caller-supplied tensors/mappings and are therefore suitable for the
I1 synthetic/static qualification as well as a later separately authorized
natural runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .cw1_sr0_structure_v1 import (
    CW1SR0SuperRegion,
    enumerate_superregion_bank,
    superregion_bank_sha256,
    superregion_sha256,
)
from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    FEATURE_DIM,
    FEATURE_SCHEMA_SHA256,
    HARD_MAX_TIE_ULPS,
    P_EPISODES_PER_UPDATE,
    P_GRADIENT_CLIP_L2,
    P_LOCK_ERASED_FIELDS,
    P_LOCK_REQUIRED_FIELDS,
    P_SAMPLER_NAMESPACE,
    P_SEED,
    P_UPDATES_TOTAL,
    CandidateActionFeatureTable,
    PEpisodeKey,
    SharedMultitilePHead,
    SharedPContractError,
    canonical_training_state_bytes,
    deterministic_p_epoch_order,
    make_frozen_p_optimizer,
    p_learning_rate,
    scale_balanced_hierarchical_logmeanexp,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_runtime_v1_20260815"
RAW_SOURCE_SCHEMA = "rc_dino_rcde_sr0_mt_p_raw_source_v1_20260815"
FEATURE_LEDGER_SCHEMA = "rc_dino_rcde_sr0_mt_p_feature_ledger_v1_20260815"
FEATURE_VALIDATION_SCHEMA = "rc_dino_rcde_sr0_mt_p_feature_validation_v1_20260815"
FIT_MANIFEST_SCHEMA = "rc_dino_rcde_sr0_mt_p_fit_manifest_v1_20260815"
PAIR_JOIN_SCHEMA = "rc_dino_rcde_sr0_mt_p_pair_join_v1_20260815"
RESUME_SCHEMA = "rc_dino_rcde_sr0_mt_p_resume_v1_20260815"
CHECKPOINT_SCHEMA = "rc_dino_rcde_sr0_mt_p_checkpoint_v1_20260815"
FIT_RESULT_SCHEMA = "rc_dino_rcde_sr0_mt_p_fit_result_v1_20260815"
FIT_VALIDATION_SCHEMA = "rc_dino_rcde_sr0_mt_p_fit_validation_v1_20260815"
LOCK_LEDGER_SCHEMA = "rc_dino_rcde_sr0_mt_p_lock_ledger_v1_20260815"
LOCK_VALIDATION_SCHEMA = "rc_dino_rcde_sr0_mt_p_lock_validation_v1_20260815"
LOCK_MANIFEST_SCHEMA = "rc_dino_rcde_sr0_mt_p_lock_manifest_v1_20260815"
EXECUTION_NODE_SCHEMA = "rc_dino_rcde_sr0_mt_p_execution_node_v1_20260815"
COMPACT_SOURCE_INDEX_EXECUTION_NODE_SCHEMA = (
    "rc_dino_rcde_sr0_mt_compact_source_index_execution_node_v1_20260816"
)
_SUPPORTED_EXECUTION_NODE_SCHEMAS = frozenset(
    {EXECUTION_NODE_SCHEMA, COMPACT_SOURCE_INDEX_EXECUTION_NODE_SCHEMA}
)

_RC_ROOT = Path(__file__).resolve().parents[2]
_AUTHORITY_REGISTRY = _RC_ROOT / "registry"
_AUTHORITY_NAME = re.compile(r"current_authority_v([1-9][0-9]*)_([0-9]{8})\.json")
_I1_SYNTHETIC_AUTHORITY_STATUS = (
    "RCDE_SR0_MT_I1_FORMAL_IMPLEMENTATION_PREFLIGHT_AUTHORIZED"
)
_I1_SYNTHETIC_NEXT_STAGE = "SR0_MT_I1_FORMAL_IMPLEMENTATION_PREFLIGHT_ONLY"
_NATURAL_AUTHORITY_STATUS = "RCDE_SR0_MT_FORMAL_TRAINING_EXECUTION_AUTHORIZED"
_NATURAL_NEXT_STAGE = "SR0_MT_FORMAL_TRAINING"
_CATALOG_PREP_AUTHORITY_STATUS = "RCDE_SR0_MT_CATALOG_PREP_EXECUTION_AUTHORIZED"
_CATALOG_PREP_NEXT_STAGE = "SR0_MT_CATALOG_PREP"
_CATALOG_PREP_STAGE_SCOPE = {
    "P_FEATURE": "natural_p_feature_materialization",
    "P_FEATURE_VALIDATE": "natural_p_feature_validation",
    "P_FEATURE_AGGREGATE": "natural_p_feature_materialization",
    "P_FEATURE_AGGREGATE_VALIDATE": "natural_p_feature_validation",
    "P_CROSSFIT_MANIFEST": "natural_p_execution_manifest_materialization",
    "P_CROSSFIT_MANIFEST_VALIDATE": "natural_p_execution_manifest_validation",
    "EXECUTION_CATALOG": "natural_p_execution_manifest_materialization",
    "EXECUTION_CATALOG_VALIDATE": "natural_p_execution_manifest_validation",
}
_AUTHORITY_EXECUTION_SCOPE_EXCEPTIONS = frozenset(
    {"run_synthetic_tests", "metadata_postjoin", "science_reduction"}
)
LATEST_AUTHORITY_PATH_SENTINEL = "__LATEST_EXECUTION_AUTHORITY__"
LATEST_AUTHORITY_SHA_SENTINEL = "__LATEST_EXECUTION_AUTHORITY_SHA256__"
EXECUTION_AUTHORITY_ENV = "SR0_MT_EXECUTION_AUTHORITY"

P_DIRECTIONS = ("a_to_b", "b_to_a")
ROOT_READY = "ROOT_LOCAL_COMPONENT_READY"
ROOT_H0 = "ROOT_LOCAL_COMPONENT_H0"
LOCK_READY = "P_LOCK_READY"
LOCK_H0 = "P_LOCK_H0"
FIXED_DIRECTION_WEIGHT = 0.5
MASK_RLE_FLATTEN_ORDER = "ROW_MAJOR"
MASK_RLE_INDEX_BASE = 0
MASK_RLE_RUN_SEMANTICS = "HALF_OPEN_MAXIMAL_TRUE_RUN_AS_START_LENGTH"
VECTOR_SCALAR_MAX_ULPS = 8
MIN_COMPONENT_PATCHES = 4
MIN_READY_ROOTS = 2

_SHA256 = frozenset("0123456789abcdef")
_FORBIDDEN_PREJOIN_KEYS = frozenset(
    {
        "target",
        "target_label",
        "target_identity",
        "target_row",
        "target_physical_row",
        "target_corrected_identity",
        "exact_target_row",
        "ground_truth",
        "gt",
        "label",
        "reference_label",
        "identity",
        "candidate_identity",
        "corrected_identity",
        "rival",
        "strongest_rival",
        "strongest_rival_identity",
        "strongest_rival_physical_row",
        "correctness",
        "correct",
        "is_correct",
        "d1_score",
        "d1_rank",
        "d1_slot",
        "slot",
        "winner",
        "d1_gap",
        "rank",
        "score",
        "confidence",
        "retrieval_outcome",
    }
)


class FormalPContractError(ValueError):
    """A formal P artifact, geometry, fold, or training invariant failed."""


def require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _SHA256 for c in value):
        raise FormalPContractError(f"{name} must be a lowercase SHA256")
    return value


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(tensor.shape)))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_bytes(value: object) -> bytes:
    """Canonical encoding for tensor-bearing fit receipts."""

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
            canonical_json_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}),
        )
        append(b"tensor-data", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        entries = sorted(
            ((_state_bytes(key), _state_bytes(item)) for key, item in value.items()),
            key=lambda item: item[0],
        )
        for key, item in entries:
            append(b"key", key)
            append(b"value", item)
    elif isinstance(value, (tuple, list)):
        tag = b"tuple" if isinstance(value, tuple) else b"list"
        for item in value:
            append(tag, _state_bytes(item))
    elif value is None:
        append(b"none", b"")
    elif isinstance(value, bool):
        append(b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        append(b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise FormalPContractError("non-finite float in canonical state")
        append(b"float", value.hex().encode("ascii"))
    elif isinstance(value, str):
        append(b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        append(b"bytes", value)
    else:
        raise FormalPContractError(
            f"unsupported canonical state type: {type(value).__name__}"
        )
    return bytes(output)


def state_sha256(value: object) -> str:
    return hashlib.sha256(_state_bytes(value)).hexdigest()


def exclusive_json(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _exclusive_bytes(destination, payload.encode("utf-8"))


def exclusive_torch(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_torch_mapping(path: Path, *, name: str) -> dict[str, object]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FormalPContractError(f"{name} is absent or unsafe: {source}")
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise FormalPContractError(f"{name} must be a tensor-safe mapping")
    return value


def load_json_mapping(path: Path, *, name: str) -> dict[str, object]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FormalPContractError(f"{name} is absent or unsafe: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FormalPContractError(f"{name} must be a JSON object")
    return value


def _require_latest_authority_path(path: Path) -> Path:
    source = Path(path)
    registry = _AUTHORITY_REGISTRY.resolve()
    if (
        not source.is_file()
        or source.is_symlink()
        or source.parent.resolve() != registry
        or source.resolve().parent != registry
    ):
        raise FormalPContractError(
            "execution authority must be a non-symlink file in RC/registry"
        )
    match = _AUTHORITY_NAME.fullmatch(source.name)
    if match is None:
        raise FormalPContractError("execution authority filename is not canonical")
    discovered: list[tuple[int, int, Path]] = []
    for candidate in _AUTHORITY_REGISTRY.iterdir():
        candidate_match = _AUTHORITY_NAME.fullmatch(candidate.name)
        if (
            candidate_match is not None
            and candidate.is_file()
            and not candidate.is_symlink()
        ):
            discovered.append(
                (
                    int(candidate_match.group(1)),
                    int(candidate_match.group(2)),
                    candidate.resolve(),
                )
            )
    if not discovered:
        raise FormalPContractError("RC authority registry is empty")
    latest_version = max(item[0] for item in discovered)
    latest = [item for item in discovered if item[0] == latest_version]
    if len(latest) != 1 or source.resolve() != latest[0][2]:
        raise FormalPContractError(
            "execution authority is not the unique latest numeric vN authority"
        )
    return source.resolve()


def resolve_execution_authority_arguments(
    arguments: Mapping[str, object],
) -> dict[str, object]:
    """Resolve the frozen natural-authority sentinel pair from one env path."""

    result = dict(arguments)
    authority = result.get("authority")
    authority_sha = result.get("authority_sha256")
    sentinel_requested = (
        authority == LATEST_AUTHORITY_PATH_SENTINEL
        or authority_sha == LATEST_AUTHORITY_SHA_SENTINEL
    )
    if not sentinel_requested:
        if (
            authority == LATEST_AUTHORITY_SHA_SENTINEL
            or authority_sha == LATEST_AUTHORITY_PATH_SENTINEL
        ):
            raise FormalPContractError("execution authority sentinel pair drift")
        return result
    if (
        authority != LATEST_AUTHORITY_PATH_SENTINEL
        or authority_sha != LATEST_AUTHORITY_SHA_SENTINEL
    ):
        raise FormalPContractError("execution authority sentinels must be an exact pair")
    raw = os.environ.get(EXECUTION_AUTHORITY_ENV)
    if not isinstance(raw, str) or not raw:
        raise FormalPContractError(
            f"{EXECUTION_AUTHORITY_ENV} is required for natural manifest execution"
        )
    resolved = _require_latest_authority_path(Path(raw))
    result["authority"] = str(resolved)
    result["authority_sha256"] = file_sha256(resolved)
    return result


def validate_natural_authority(
    path: Path,
    *,
    expected_file_sha256: str,
    required_scope: str,
    execution_protocol_sha256: str | None = None,
) -> dict[str, object]:
    """Validate one later-stage authority before any natural P access.

    I1 itself has every natural capability set to false, so this check also
    prevents a self-consistent execution manifest from bypassing the live
    registry.  A later authority must explicitly opt in to the exact scope and
    bind the formal execution protocol used by the node.
    """

    source = _require_latest_authority_path(Path(path))
    if file_sha256(source) != require_sha256(
        expected_file_sha256, name="natural authority file"
    ):
        raise FormalPContractError("natural authority file hash drift")
    authority = load_json_mapping(source, name="natural execution authority")
    scope = authority.get("authorized_scope")
    common_valid = (
        not isinstance(required_scope, str)
        or not required_scope
        or not isinstance(scope, Mapping)
        or scope.get(required_scope) is not True
        or authority.get("automatic_stage_advance") is not False
        or authority.get("scientific_GO_or_NO_GO") is not None
    )
    if common_valid:
        raise FormalPContractError(
            f"natural authority does not authorize scope {required_scope}"
        )
    authority_status = authority.get("status")
    if required_scope == "run_synthetic_tests":
        if (
            authority_status != _I1_SYNTHETIC_AUTHORITY_STATUS
            or authority.get("next_authorized_stage") != _I1_SYNTHETIC_NEXT_STAGE
            or authority.get("natural_training_authorized") is not False
            or authority.get("training_authorized") is not False
            or authority.get("checkpoint_deserialization_authorized") is not False
            or authority.get("model_load_authorized") is not False
            or authority.get("model_forward_authorized") is not False
            or authority.get("model_backward_authorized") is not False
            or authority.get("model_update_authorized") is not False
        ):
            raise FormalPContractError(
                "I1 synthetic authority has natural/model capability drift"
            )
    elif authority_status == _CATALOG_PREP_AUTHORITY_STATUS:
        prep_stage = scope.get("catalog_prep_stage")
        phase_ordinal = scope.get("catalog_prep_phase_ordinal")
        expected_scope = (
            _CATALOG_PREP_STAGE_SCOPE.get(prep_stage)
            if isinstance(prep_stage, str)
            else None
        )
        enabled_execution_scopes = {
            key
            for key, enabled in scope.items()
            if isinstance(key, str)
            and enabled is True
            and (
                key.startswith("natural_")
                or key in _AUTHORITY_EXECUTION_SCOPE_EXCEPTIONS
            )
        }
        if (
            authority.get("next_authorized_stage") != _CATALOG_PREP_NEXT_STAGE
            or expected_scope != required_scope
            or isinstance(phase_ordinal, bool)
            or not isinstance(phase_ordinal, int)
            or phase_ordinal < 0
            or enabled_execution_scopes != {required_scope}
            or authority.get("natural_training_authorized") is not False
            or authority.get("training_authorized") is not False
            or authority.get("checkpoint_deserialization_authorized") is not False
            or authority.get("model_load_authorized") is not False
            or authority.get("model_forward_authorized") is not False
            or authority.get("model_backward_authorized") is not False
            or authority.get("model_update_authorized") is not False
        ):
            raise FormalPContractError(
                "catalog-prep authority stage/ordinal/scope/capability drift"
            )
    else:
        if (
            authority_status != _NATURAL_AUTHORITY_STATUS
            or authority.get("next_authorized_stage") != _NATURAL_NEXT_STAGE
            or authority.get("natural_training_authorized") is not True
        ):
            raise FormalPContractError(
                "natural execution requires the formal training authority"
            )
    if execution_protocol_sha256 is not None:
        required_protocol = require_sha256(
            execution_protocol_sha256, name="natural execution protocol"
        )
        bindings = authority.get("bindings")
        protocol = (
            bindings.get("sr0_mt_execution_protocol")
            if isinstance(bindings, Mapping)
            else None
        )
        if (
            not isinstance(protocol, Mapping)
            or protocol.get("sha256") != required_protocol
        ):
            raise FormalPContractError(
                "natural authority does not bind the execution protocol"
            )
    return authority


def _validated_execution_manifest(
    argv: Sequence[str],
    *,
    expected_program: str,
    expected_schema: str = EXECUTION_NODE_SCHEMA,
) -> tuple[Path, dict[str, object]] | None:
    if expected_schema not in _SUPPORTED_EXECUTION_NODE_SCHEMAS:
        raise FormalPContractError("unsupported P execution manifest schema")
    values = list(argv)
    if "--execution-manifest" not in values:
        return None
    if len(values) != 2 or values[0] != "--execution-manifest":
        raise FormalPContractError(
            "execution-manifest mode cannot be mixed with explicit CLI arguments"
        )
    path = Path(values[1])
    manifest = load_json_mapping(path, name="P execution manifest")
    if manifest.get("schema_version") != expected_schema:
        raise FormalPContractError("P execution manifest schema drift")
    if manifest.get("program") != expected_program:
        raise FormalPContractError("P execution manifest targets another program")
    if manifest.get("logical_sha256") != canonical_sha256(
        {key: item for key, item in manifest.items() if key != "logical_sha256"}
    ):
        raise FormalPContractError("P execution manifest logical hash drift")
    return path, manifest


def execution_manifest_file_sha256(
    argv: Sequence[str],
    *,
    expected_program: str,
    expected_schema: str = EXECUTION_NODE_SCHEMA,
) -> str | None:
    """Return the physical manifest hash after exact manifest validation.

    Explicit-CLI invocations have no physical execution manifest and therefore
    return ``None``.  The helper has no process-global state, so callers can
    bind the exact submitted file into their own immutable result lineage.
    """

    loaded = _validated_execution_manifest(
        argv,
        expected_program=expected_program,
        expected_schema=expected_schema,
    )
    return None if loaded is None else file_sha256(loaded[0])


def argv_from_execution_manifest(
    argv: Sequence[str],
    *,
    expected_program: str,
    expected_schema: str = EXECUTION_NODE_SCHEMA,
) -> list[str]:
    """Resolve one JSON execution node without eval or shell expansion.

    Manifest mode is intentionally exclusive: the command line must contain
    exactly ``--execution-manifest PATH``.  The JSON object then provides a
    scalar ``arguments`` mapping whose keys are converted to long options.
    One invocation still executes exactly one program/stage.
    """

    values = list(argv)
    loaded = _validated_execution_manifest(
        values,
        expected_program=expected_program,
        expected_schema=expected_schema,
    )
    if loaded is None:
        return values
    _, manifest = loaded
    arguments = manifest.get("arguments")
    if not isinstance(arguments, Mapping) or not arguments:
        raise FormalPContractError("P execution manifest arguments are absent")
    arguments = resolve_execution_authority_arguments(arguments)
    result: list[str] = []
    for key, item in arguments.items():
        if (
            not isinstance(key, str)
            or not key
            or key.startswith("-")
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in key)
        ):
            raise FormalPContractError("P execution manifest argument name is unsafe")
        option = "--" + key.replace("_", "-")
        if isinstance(item, bool):
            if item:
                result.append(option)
        elif item is None:
            continue
        elif isinstance(item, (str, int, float)):
            result.extend((option, str(item)))
        else:
            raise FormalPContractError(
                "P execution manifest arguments must be scalar JSON values"
            )
    return result


def encode_mask_rle(value: torch.Tensor) -> list[list[int]]:
    mask = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous().flatten()
    result: list[list[int]] = []
    start: int | None = None
    for index, active in enumerate(mask.tolist()):
        if active and start is None:
            start = index
        elif not active and start is not None:
            result.append([start, index - start])
            start = None
    if start is not None:
        result.append([start, mask.numel() - start])
    return result


def decode_mask_rle(value: object, *, numel: int) -> torch.Tensor:
    if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
        raise FormalPContractError("RLE destination length must be positive")
    if not isinstance(value, list):
        raise FormalPContractError("mask RLE must be a list")
    mask = torch.zeros(numel, dtype=torch.bool)
    previous_end = 0
    for ordinal, run in enumerate(value):
        if (
            not isinstance(run, list)
            or len(run) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in run)
        ):
            raise FormalPContractError("mask RLE run must be [start,length]")
        start, length = run
        if start < 0 or length <= 0 or start + length > numel:
            raise FormalPContractError("mask RLE run is outside the declared tensor")
        if ordinal and start <= previous_end:
            raise FormalPContractError("mask RLE runs are not maximal and separated")
        mask[start : start + length] = True
        previous_end = start + length
    if encode_mask_rle(mask) != value:
        raise FormalPContractError("mask RLE is not canonical")
    return mask


def _grid(value: object, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or min(value) <= 0
    ):
        raise FormalPContractError(f"{name} must be a positive [height,width]")
    return int(value[0]), int(value[1])


def _mask(value: object, *, numel: int, name: str, allow_empty: bool) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous().flatten()
    if tensor.shape != (numel,):
        raise FormalPContractError(f"{name} mask length drift")
    if not allow_empty and not bool(tensor.any()):
        raise FormalPContractError(f"{name} mask must be nonempty")
    return tensor


def _component_is_legal(mask: torch.Tensor, grid_shape: tuple[int, int]) -> bool:
    value = torch.as_tensor(mask, dtype=torch.bool).reshape(grid_shape)
    points = torch.nonzero(value, as_tuple=False)
    if points.shape[0] < MIN_COMPONENT_PATCHES:
        return False
    if torch.unique(points[:, 0]).numel() < 2 or torch.unique(points[:, 1]).numel() < 2:
        return False
    active = {tuple(int(v) for v in point) for point in points.tolist()}
    frontier = [next(iter(active))]
    visited = {frontier[0]}
    while frontier:
        row, column = frontier.pop()
        for neighbour in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if neighbour in active and neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return visited == active


def assert_no_forbidden_prejoin_keys(value: object, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _FORBIDDEN_PREJOIN_KEYS:
                raise FormalPContractError(f"forbidden prejoin key at {path}.{key}")
            assert_no_forbidden_prejoin_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_forbidden_prejoin_keys(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class RootActionDeployment:
    action_key: str
    eligible: bool
    query_mask: torch.Tensor
    reference_mask: torch.Tensor
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_geometry_sha256: str
    reference_geometry_sha256: str
    binding_status: str
    binding_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.action_key, name="action key")
        q_shape = _grid(self.query_grid_shape, name="deployment query grid")
        r_shape = _grid(self.reference_grid_shape, name="deployment reference grid")
        query = _mask(
            self.query_mask,
            numel=math.prod(q_shape),
            name="deployment query",
            allow_empty=not self.eligible,
        )
        reference = _mask(
            self.reference_mask,
            numel=math.prod(r_shape),
            name="deployment reference",
            allow_empty=not self.eligible,
        )
        require_sha256(self.query_geometry_sha256, name="query geometry")
        require_sha256(self.reference_geometry_sha256, name="reference geometry")
        if self.eligible:
            if self.binding_status != ROOT_READY:
                raise FormalPContractError("eligible deployment must be ROOT_READY")
            if not _component_is_legal(query, q_shape) or not _component_is_legal(
                reference, r_shape
            ):
                raise FormalPContractError(
                    "eligible deployment masks must be >=4-patch 4CC with 2-D span"
                )
        elif self.binding_status != ROOT_H0 or bool(query.any()) or bool(reference.any()):
            raise FormalPContractError("H0 deployment must contain exact empty masks")
        payload = {
            "action_key": self.action_key,
            "eligible": bool(self.eligible),
            "query_mask_sha256": tensor_sha256(query),
            "reference_mask_sha256": tensor_sha256(reference),
            "query_grid_shape": list(q_shape),
            "reference_grid_shape": list(r_shape),
            "query_geometry_sha256": self.query_geometry_sha256,
            "reference_geometry_sha256": self.reference_geometry_sha256,
            "binding_status": self.binding_status,
        }
        if self.binding_sha256 != canonical_sha256(payload):
            raise FormalPContractError("root action deployment hash drift")
        object.__setattr__(self, "query_mask", query)
        object.__setattr__(self, "reference_mask", reference)
        object.__setattr__(self, "query_grid_shape", q_shape)
        object.__setattr__(self, "reference_grid_shape", r_shape)


def make_root_action_deployment(
    *,
    action_key: str,
    eligible: bool,
    query_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    query_geometry_sha256: str,
    reference_geometry_sha256: str,
) -> RootActionDeployment:
    q = torch.as_tensor(query_mask, dtype=torch.bool).detach().cpu().contiguous().flatten()
    r = torch.as_tensor(reference_mask, dtype=torch.bool).detach().cpu().contiguous().flatten()
    if not eligible:
        q = torch.zeros(math.prod(query_grid_shape), dtype=torch.bool)
        r = torch.zeros(math.prod(reference_grid_shape), dtype=torch.bool)
    status = ROOT_READY if eligible else ROOT_H0
    payload = {
        "action_key": action_key,
        "eligible": bool(eligible),
        "query_mask_sha256": tensor_sha256(q),
        "reference_mask_sha256": tensor_sha256(r),
        "query_grid_shape": list(query_grid_shape),
        "reference_grid_shape": list(reference_grid_shape),
        "query_geometry_sha256": query_geometry_sha256,
        "reference_geometry_sha256": reference_geometry_sha256,
        "binding_status": status,
    }
    return RootActionDeployment(
        action_key=action_key,
        eligible=eligible,
        query_mask=q,
        reference_mask=r,
        query_grid_shape=query_grid_shape,
        reference_grid_shape=reference_grid_shape,
        query_geometry_sha256=query_geometry_sha256,
        reference_geometry_sha256=reference_geometry_sha256,
        binding_status=status,
        binding_sha256=canonical_sha256(payload),
    )


@dataclass(frozen=True)
class DirectionalFeatureRecord:
    query_id: str
    historical_query_ordinal: int
    execution_ordinal: int
    query_source_image_sha256: str
    source_fold: int
    candidate_position: int
    candidate_key: str
    candidate_physical_row: int
    candidate_reference_source_sha256: str
    direction: str
    colnomic_query_grid_shape: tuple[int, int]
    action_table: CandidateActionFeatureTable
    deployments_by_root: tuple[tuple[RootActionDeployment, ...], ...]
    complete_bank_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise FormalPContractError("query ID must be nonempty")
        for name, value in (
            ("historical query ordinal", self.historical_query_ordinal),
            ("execution ordinal", self.execution_ordinal),
            ("candidate position", self.candidate_position),
            ("candidate physical row", self.candidate_physical_row),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FormalPContractError(f"{name} must be nonnegative")
        if self.source_fold not in (1, 2, 3, 4):
            raise FormalPContractError("source fold must be 1..4")
        require_sha256(self.query_source_image_sha256, name="query source image")
        require_sha256(
            self.candidate_reference_source_sha256, name="candidate reference source image"
        )
        if self.direction not in P_DIRECTIONS:
            raise FormalPContractError("P direction must be a_to_b or b_to_a")
        if self.candidate_key != self.action_table.candidate_key:
            raise FormalPContractError("candidate/action-table key drift")
        cq_shape = _grid(self.colnomic_query_grid_shape, name="ColNomic query grid")
        regions = tuple(enumerate_superregion_bank(cq_shape, include_r0_control=False))
        if self.complete_bank_sha256 != superregion_bank_sha256(regions):
            raise FormalPContractError("complete CW1 bank hash drift")
        deployments = tuple(tuple(root) for root in self.deployments_by_root)
        if len(deployments) != len(self.action_table.root_ordinals):
            raise FormalPContractError("deployment/action root count drift")
        query_geometry: set[str] = set()
        reference_geometry: set[str] = set()
        for root, items in enumerate(deployments):
            if len(items) != len(self.action_table.action_keys_by_root[root]):
                raise FormalPContractError("deployment/action population length drift")
            if tuple(item.action_key for item in items) != self.action_table.action_keys_by_root[root]:
                raise FormalPContractError("deployment/action key order drift")
            expected = self.action_table.eligible_by_root[root].tolist()
            if [item.eligible for item in items] != expected:
                raise FormalPContractError("feature/deployment eligibility drift")
            query_geometry.update(item.query_geometry_sha256 for item in items)
            reference_geometry.update(item.reference_geometry_sha256 for item in items)
        if len(query_geometry) != 1 or len(reference_geometry) != 1:
            raise FormalPContractError("one candidate direction must bind one geometry pair")
        if self.record_sha256 != directional_feature_record_sha256(self):
            raise FormalPContractError("directional feature record hash drift")
        object.__setattr__(self, "colnomic_query_grid_shape", cq_shape)
        object.__setattr__(self, "deployments_by_root", deployments)

    @property
    def query_geometry_sha256(self) -> str:
        return self.deployments_by_root[0][0].query_geometry_sha256

    @property
    def reference_geometry_sha256(self) -> str:
        return self.deployments_by_root[0][0].reference_geometry_sha256

    @property
    def dino_query_grid_shape(self) -> tuple[int, int]:
        return self.deployments_by_root[0][0].query_grid_shape

    @property
    def dino_reference_grid_shape(self) -> tuple[int, int]:
        return self.deployments_by_root[0][0].reference_grid_shape


def _directional_feature_record_payload(value: DirectionalFeatureRecord) -> dict[str, object]:
    return {
        "query_id": value.query_id,
        "historical_query_ordinal": value.historical_query_ordinal,
        "execution_ordinal": value.execution_ordinal,
        "query_source_image_sha256": value.query_source_image_sha256,
        "source_fold": value.source_fold,
        "candidate_position": value.candidate_position,
        "candidate_key": value.candidate_key,
        "candidate_physical_row": value.candidate_physical_row,
        "candidate_reference_source_sha256": value.candidate_reference_source_sha256,
        "direction": value.direction,
        "colnomic_query_grid_shape": list(value.colnomic_query_grid_shape),
        "complete_bank_sha256": value.complete_bank_sha256,
        "action_keys_by_root": [list(item) for item in value.action_table.action_keys_by_root],
        "feature_tensor_sha256_by_root": [
            tensor_sha256(item) for item in value.action_table.features_by_root
        ],
        "eligible_tensor_sha256_by_root": [
            tensor_sha256(item) for item in value.action_table.eligible_by_root
        ],
        "deployment_binding_sha256_by_root": [
            [item.binding_sha256 for item in root] for root in value.deployments_by_root
        ],
    }


def directional_feature_record_sha256(value: DirectionalFeatureRecord) -> str:
    return canonical_sha256(_directional_feature_record_payload(value))


def make_directional_feature_record(
    *,
    query_id: str,
    historical_query_ordinal: int,
    execution_ordinal: int,
    query_source_image_sha256: str,
    source_fold: int,
    candidate_position: int,
    candidate_key: str,
    candidate_physical_row: int,
    candidate_reference_source_sha256: str,
    direction: str,
    colnomic_query_grid_shape: tuple[int, int],
    action_table: CandidateActionFeatureTable,
    deployments_by_root: tuple[tuple[RootActionDeployment, ...], ...],
) -> DirectionalFeatureRecord:
    bank = tuple(
        enumerate_superregion_bank(colnomic_query_grid_shape, include_r0_control=False)
    )
    complete_bank_sha = superregion_bank_sha256(bank)
    # Compute the same payload without relying on a partially initialized
    # dataclass.
    payload = {
        "query_id": query_id,
        "historical_query_ordinal": historical_query_ordinal,
        "execution_ordinal": execution_ordinal,
        "query_source_image_sha256": query_source_image_sha256,
        "source_fold": source_fold,
        "candidate_position": candidate_position,
        "candidate_key": candidate_key,
        "candidate_physical_row": candidate_physical_row,
        "candidate_reference_source_sha256": candidate_reference_source_sha256,
        "direction": direction,
        "colnomic_query_grid_shape": list(colnomic_query_grid_shape),
        "complete_bank_sha256": complete_bank_sha,
        "action_keys_by_root": [list(item) for item in action_table.action_keys_by_root],
        "feature_tensor_sha256_by_root": [tensor_sha256(item) for item in action_table.features_by_root],
        "eligible_tensor_sha256_by_root": [tensor_sha256(item) for item in action_table.eligible_by_root],
        "deployment_binding_sha256_by_root": [
            [item.binding_sha256 for item in root] for root in deployments_by_root
        ],
    }
    return DirectionalFeatureRecord(
        query_id=query_id,
        historical_query_ordinal=historical_query_ordinal,
        execution_ordinal=execution_ordinal,
        query_source_image_sha256=query_source_image_sha256,
        source_fold=source_fold,
        candidate_position=candidate_position,
        candidate_key=candidate_key,
        candidate_physical_row=candidate_physical_row,
        candidate_reference_source_sha256=candidate_reference_source_sha256,
        direction=direction,
        colnomic_query_grid_shape=colnomic_query_grid_shape,
        action_table=action_table,
        deployments_by_root=deployments_by_root,
        complete_bank_sha256=complete_bank_sha,
        record_sha256=canonical_sha256(payload),
    )


def serialize_directional_feature_record(value: DirectionalFeatureRecord) -> dict[str, object]:
    if not isinstance(value, DirectionalFeatureRecord):
        raise FormalPContractError("feature record serializer needs DirectionalFeatureRecord")
    return {
        **{
            "query_id": value.query_id,
            "historical_query_ordinal": value.historical_query_ordinal,
            "execution_ordinal": value.execution_ordinal,
            "query_source_image_sha256": value.query_source_image_sha256,
            "source_fold": value.source_fold,
            "candidate_position": value.candidate_position,
            "candidate_key": value.candidate_key,
            "candidate_physical_row": value.candidate_physical_row,
            "candidate_reference_source_sha256": value.candidate_reference_source_sha256,
            "direction": value.direction,
            "colnomic_query_grid_shape": list(value.colnomic_query_grid_shape),
            "complete_bank_sha256": value.complete_bank_sha256,
            "record_sha256": value.record_sha256,
        },
        "roots": [
            {
                "root_ordinal": root,
                "action_keys": list(value.action_table.action_keys_by_root[root]),
                "features": value.action_table.features_by_root[root].detach().cpu(),
                "eligible": value.action_table.eligible_by_root[root].detach().cpu(),
                "deployments": [
                    {
                        "action_key": item.action_key,
                        "eligible": item.eligible,
                        "query_mask": item.query_mask,
                        "reference_mask": item.reference_mask,
                        "query_grid_shape": list(item.query_grid_shape),
                        "reference_grid_shape": list(item.reference_grid_shape),
                        "query_geometry_sha256": item.query_geometry_sha256,
                        "reference_geometry_sha256": item.reference_geometry_sha256,
                        "binding_status": item.binding_status,
                        "binding_sha256": item.binding_sha256,
                    }
                    for item in value.deployments_by_root[root]
                ],
            }
            for root in value.action_table.root_ordinals
        ],
    }


def deserialize_directional_feature_record(value: Mapping[str, object]) -> DirectionalFeatureRecord:
    roots = value.get("roots")
    if not isinstance(roots, list) or not roots:
        raise FormalPContractError("serialized feature record roots are absent")
    keys: list[tuple[str, ...]] = []
    features: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []
    deployments: list[tuple[RootActionDeployment, ...]] = []
    for expected_root, raw_root in enumerate(roots):
        if not isinstance(raw_root, Mapping) or raw_root.get("root_ordinal") != expected_root:
            raise FormalPContractError("serialized feature root order drift")
        raw_keys = raw_root.get("action_keys")
        raw_deployments = raw_root.get("deployments")
        if not isinstance(raw_keys, list) or not isinstance(raw_deployments, list):
            raise FormalPContractError("serialized feature action population is absent")
        root_deployments = []
        for item in raw_deployments:
            if not isinstance(item, Mapping):
                raise FormalPContractError("serialized deployment must be a mapping")
            root_deployments.append(
                RootActionDeployment(
                    action_key=str(item["action_key"]),
                    eligible=bool(item["eligible"]),
                    query_mask=torch.as_tensor(item["query_mask"], dtype=torch.bool),
                    reference_mask=torch.as_tensor(item["reference_mask"], dtype=torch.bool),
                    query_grid_shape=_grid(item["query_grid_shape"], name="query grid"),
                    reference_grid_shape=_grid(item["reference_grid_shape"], name="reference grid"),
                    query_geometry_sha256=str(item["query_geometry_sha256"]),
                    reference_geometry_sha256=str(item["reference_geometry_sha256"]),
                    binding_status=str(item["binding_status"]),
                    binding_sha256=str(item["binding_sha256"]),
                )
            )
        keys.append(tuple(str(item) for item in raw_keys))
        features.append(torch.as_tensor(raw_root["features"], dtype=torch.float64))
        eligible.append(torch.as_tensor(raw_root["eligible"], dtype=torch.bool))
        deployments.append(tuple(root_deployments))
    table = CandidateActionFeatureTable(
        candidate_key=str(value["candidate_key"]),
        root_ordinals=tuple(range(len(roots))),
        action_keys_by_root=tuple(keys),
        features_by_root=tuple(features),
        eligible_by_root=tuple(eligible),
    )
    record = DirectionalFeatureRecord(
        query_id=str(value["query_id"]),
        historical_query_ordinal=int(value["historical_query_ordinal"]),
        execution_ordinal=int(value["execution_ordinal"]),
        query_source_image_sha256=str(value["query_source_image_sha256"]),
        source_fold=int(value["source_fold"]),
        candidate_position=int(value["candidate_position"]),
        candidate_key=str(value["candidate_key"]),
        candidate_physical_row=int(value["candidate_physical_row"]),
        candidate_reference_source_sha256=str(value["candidate_reference_source_sha256"]),
        direction=str(value["direction"]),
        colnomic_query_grid_shape=_grid(value["colnomic_query_grid_shape"], name="ColNomic query grid"),
        action_table=table,
        deployments_by_root=tuple(deployments),
        complete_bank_sha256=str(value["complete_bank_sha256"]),
        record_sha256=str(value["record_sha256"]),
    )
    return record


@dataclass(frozen=True)
class DeployableDirectionOutput:
    record: DirectionalFeatureRecord
    selected_action_indices: tuple[int | None, ...]
    selected_action_keys: tuple[str | None, ...]
    root_scores: torch.Tensor
    row_scores: torch.Tensor
    row_ready: torch.Tensor
    candidate_utility: torch.Tensor
    map_row_index: int | None
    map_row_sha256: str | None


def _exact_zero(model: SharedMultitilePHead) -> torch.Tensor:
    return model.linear.weight.square().sum() * 0.0


def score_deployable_direction(
    model: SharedMultitilePHead,
    record: DirectionalFeatureRecord,
) -> DeployableDirectionOutput:
    """Score one direction after exact deployment-geometry H0 closure."""

    if not isinstance(model, SharedMultitilePHead) or not isinstance(
        record, DirectionalFeatureRecord
    ):
        raise FormalPContractError("deployable P scorer received wrong object type")
    regions = tuple(
        enumerate_superregion_bank(
            record.colnomic_query_grid_shape, include_r0_control=False
        )
    )
    base = model.score_candidate(record.action_table, regions)
    selected_deployments: list[RootActionDeployment | None] = []
    for root, index in enumerate(base.selected_action_indices):
        selected_deployments.append(
            None if index is None else record.deployments_by_root[root][index]
        )

    row_scores: list[torch.Tensor] = []
    row_ready: list[bool] = []
    for region in regions:
        selected = [selected_deployments[root] for root in region.contributing_root_ordinals]
        ready = [item for item in selected if item is not None and item.eligible]
        legal = len(ready) >= MIN_READY_ROOTS
        if legal:
            union = torch.stack([item.query_mask for item in ready]).any(dim=0)
            legal = _component_is_legal(union, record.dino_query_grid_shape) and int(
                union.sum()
            ) > max(int(item.query_mask.sum()) for item in ready)
        if legal:
            weights = region.aggregation_weights.to(
                device=base.root_scores.device, dtype=base.root_scores.dtype
            )
            row_scores.append(torch.dot(weights, base.root_scores))
            row_ready.append(True)
        else:
            row_scores.append(_exact_zero(model))
            row_ready.append(False)
    scores = torch.stack(row_scores)
    ready_tensor = torch.tensor(row_ready, dtype=torch.bool, device=scores.device)
    utility = scale_balanced_hierarchical_logmeanexp(
        scores, tuple(item.radius for item in regions)
    )
    legal_rows = torch.nonzero(ready_tensor, as_tuple=False).flatten().tolist()
    map_index: int | None = None
    if legal_rows:
        detached = scores.detach()
        maximum = detached[
            torch.tensor(legal_rows, dtype=torch.long, device=detached.device)
        ].max()
        tolerance = (
            HARD_MAX_TIE_ULPS
            * torch.finfo(detached.dtype).eps
            * max(1.0, abs(float(maximum)))
        )
        if float(maximum) > tolerance:
            tied = [
                index
                for index in legal_rows
                if abs(float(detached[index]) - float(maximum)) <= tolerance
            ]
            map_index = min(tied, key=lambda index: superregion_sha256(regions[index]))
    return DeployableDirectionOutput(
        record=record,
        selected_action_indices=base.selected_action_indices,
        selected_action_keys=base.selected_action_keys,
        root_scores=base.root_scores,
        row_scores=scores,
        row_ready=ready_tensor,
        candidate_utility=utility,
        map_row_index=map_index,
        map_row_sha256=(
            None if map_index is None else superregion_sha256(regions[map_index])
        ),
    )


def fuse_direction_utilities(
    a_to_b: torch.Tensor, b_to_a: torch.Tensor
) -> torch.Tensor:
    first = torch.as_tensor(a_to_b)
    second = torch.as_tensor(b_to_a)
    if first.ndim != 0 or second.ndim != 0 or first.device != second.device:
        raise FormalPContractError("direction utilities must be scalar on one device")
    if not first.is_floating_point() or not second.is_floating_point():
        raise FormalPContractError("direction utilities must be floating point")
    if not bool(torch.isfinite(first)) or not bool(torch.isfinite(second)):
        raise FormalPContractError("direction utilities must be finite")
    return FIXED_DIRECTION_WEIGHT * (
        first + second.to(device=first.device, dtype=first.dtype)
    )


@dataclass(frozen=True)
class FusedPairLoss:
    target_utility: torch.Tensor
    rival_utility: torch.Tensor
    margin: torch.Tensor
    loss: torch.Tensor


def fused_direction_pair_loss(
    *,
    target_a_to_b: torch.Tensor,
    target_b_to_a: torch.Tensor,
    rival_a_to_b: torch.Tensor,
    rival_b_to_a: torch.Tensor,
) -> FusedPairLoss:
    target = fuse_direction_utilities(target_a_to_b, target_b_to_a)
    rival = fuse_direction_utilities(rival_a_to_b, rival_b_to_a)
    margin = target - rival
    return FusedPairLoss(
        target_utility=target,
        rival_utility=rival,
        margin=margin,
        loss=F.softplus(-margin),
    )


def feature_ledger_index(
    ledger: Mapping[str, object],
) -> dict[tuple[str, str, str], DirectionalFeatureRecord]:
    if ledger.get("schema_version") != FEATURE_LEDGER_SCHEMA:
        raise FormalPContractError("P feature ledger schema drift")
    if ledger.get("target_free") is not True:
        raise FormalPContractError("P feature ledger is not target-free")
    raw_records = ledger.get("records")
    if not isinstance(raw_records, list):
        raise FormalPContractError("P feature ledger records are absent")
    index: dict[tuple[str, str, str], DirectionalFeatureRecord] = {}
    per_query_positions: dict[str, dict[int, str]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise FormalPContractError("P feature ledger record is not a mapping")
        record = deserialize_directional_feature_record(raw)
        key = (record.query_id, record.candidate_key, record.direction)
        if key in index:
            raise FormalPContractError("duplicate query/candidate/direction record")
        index[key] = record
        position = per_query_positions.setdefault(record.query_id, {})
        prior = position.setdefault(record.candidate_position, record.candidate_key)
        if prior != record.candidate_key:
            raise FormalPContractError("candidate position/key collision")
    query_count = len(per_query_positions)
    expected_candidates = int(ledger.get("candidate_count_per_query", -1))
    if expected_candidates <= 0:
        raise FormalPContractError("candidate count per query is invalid")
    for query_id, positions in per_query_positions.items():
        if tuple(sorted(positions)) != tuple(range(expected_candidates)):
            raise FormalPContractError(f"query {query_id} does not retain full candidate axis")
        for candidate_key in positions.values():
            if any((query_id, candidate_key, direction) not in index for direction in P_DIRECTIONS):
                raise FormalPContractError("candidate is missing a checkerboard direction")
    if ledger.get("query_count") != query_count or ledger.get("record_count") != len(index):
        raise FormalPContractError("P feature ledger population counts drift")
    if len(index) != query_count * expected_candidates * len(P_DIRECTIONS):
        raise FormalPContractError("P feature ledger is not complete query x candidate x direction")
    expected_logical = feature_ledger_logical_sha256(ledger)
    if ledger.get("logical_sha256") != expected_logical:
        raise FormalPContractError("P feature ledger logical hash drift")
    return index


def feature_ledger_logical_sha256(ledger: Mapping[str, object]) -> str:
    raw = ledger.get("records")
    if not isinstance(raw, list):
        raise FormalPContractError("feature ledger records are absent")
    core = {key: item for key, item in ledger.items() if key not in {"records", "logical_sha256"}}
    core["record_sha256_sequence"] = [
        item.get("record_sha256") if isinstance(item, Mapping) else None for item in raw
    ]
    return canonical_sha256(core)


def pair_join_episodes(
    pair_join: Mapping[str, object],
    feature_index: Mapping[tuple[str, str, str], DirectionalFeatureRecord],
) -> tuple[dict[str, object], ...]:
    if pair_join.get("schema_version") != PAIR_JOIN_SCHEMA:
        raise FormalPContractError("P pair join schema drift")
    raw_episodes = pair_join.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise FormalPContractError("P fit has no eligible pair episodes")
    episodes: list[dict[str, object]] = []
    seen_queries: set[str] = set()
    for raw in raw_episodes:
        if not isinstance(raw, Mapping):
            raise FormalPContractError("P pair episode must be a mapping")
        required = {
            "query_id",
            "query_source_image_sha256",
            "execution_ordinal",
            "target_candidate_key",
            "rival_candidate_key",
        }
        if set(raw) != required:
            raise FormalPContractError("P pair episode field set drift")
        query_id = str(raw["query_id"])
        if query_id in seen_queries:
            raise FormalPContractError("P pair join contains duplicate query")
        seen_queries.add(query_id)
        target = str(raw["target_candidate_key"])
        rival = str(raw["rival_candidate_key"])
        if target == rival:
            raise FormalPContractError("P target and rival candidate are identical")
        records = []
        for candidate in (target, rival):
            directional = []
            for direction in P_DIRECTIONS:
                key = (query_id, candidate, direction)
                if key not in feature_index:
                    raise FormalPContractError("joined candidate/direction absent from target-free ledger")
                directional.append(feature_index[key])
            records.append(tuple(directional))
        anchor = records[0][0]
        if (
            raw["query_source_image_sha256"] != anchor.query_source_image_sha256
            or raw["execution_ordinal"] != anchor.execution_ordinal
        ):
            raise FormalPContractError("P pair join query addressing drift")
        episodes.append(
            {
                "key": PEpisodeKey(
                    query_id=query_id,
                    source_image_sha256=anchor.query_source_image_sha256,
                    execution_ordinal=anchor.execution_ordinal,
                ),
                "target": records[0],
                "rival": records[1],
            }
        )
    return tuple(episodes)


@dataclass
class DeterministicEpisodeSampler:
    keys: tuple[PEpisodeKey, ...]
    fit_id: str
    epoch: int = 0
    position: int = 0

    def __post_init__(self) -> None:
        if not self.keys:
            raise FormalPContractError("P sampler population is empty")
        if not isinstance(self.fit_id, str) or not self.fit_id:
            raise FormalPContractError("P sampler fit ID is empty")
        self._refresh()

    def _refresh(self) -> None:
        self.order = deterministic_p_epoch_order(
            self.keys, fit_id=self.fit_id, epoch=self.epoch
        )
        self.order_sha256 = canonical_sha256(list(self.order))
        if not 0 <= self.position <= len(self.order):
            raise FormalPContractError("P sampler position is outside current epoch")

    def draw(self, count: int) -> tuple[int, ...]:
        result: list[int] = []
        while len(result) < count:
            if self.position == len(self.order):
                self.epoch += 1
                self.position = 0
                self._refresh()
            take = min(count - len(result), len(self.order) - self.position)
            result.extend(self.order[self.position : self.position + take])
            self.position += take
        return tuple(result)

    def state(self) -> dict[str, object]:
        return {
            "namespace": P_SAMPLER_NAMESPACE,
            "fit_id": self.fit_id,
            "epoch": self.epoch,
            "position": self.position,
            "order_sha256": self.order_sha256,
        }

    @classmethod
    def restore(
        cls, keys: tuple[PEpisodeKey, ...], value: Mapping[str, object]
    ) -> "DeterministicEpisodeSampler":
        if value.get("namespace") != P_SAMPLER_NAMESPACE:
            raise FormalPContractError("P sampler namespace drift on resume")
        sampler = cls(
            keys=keys,
            fit_id=str(value["fit_id"]),
            epoch=int(value["epoch"]),
            position=int(value["position"]),
        )
        if sampler.order_sha256 != value.get("order_sha256"):
            raise FormalPContractError("P sampler order drift on resume")
        return sampler


def capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": (
            str(numpy_state[0]),
            torch.from_numpy(numpy_state[1].copy()),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch_cpu_rng_state": torch.random.get_rng_state().clone(),
        "torch_cuda_rng_states": tuple(
            item.clone() for item in torch.cuda.get_rng_state_all()
        )
        if torch.cuda.is_available()
        else (),
    }


def restore_rng_state(value: Mapping[str, object]) -> None:
    random.setstate(value["python_rng_state"])  # type: ignore[arg-type]
    numpy_state = value["numpy_rng_state"]
    if not isinstance(numpy_state, (list, tuple)) or len(numpy_state) != 5:
        raise FormalPContractError("NumPy RNG state schema drift")
    np.random.set_state(
        (
            str(numpy_state[0]),
            torch.as_tensor(numpy_state[1], dtype=torch.uint32).cpu().numpy(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        )
    )
    torch.random.set_rng_state(torch.as_tensor(value["torch_cpu_rng_state"], dtype=torch.uint8))
    cuda_states = value["torch_cuda_rng_states"]
    if torch.cuda.is_available() and isinstance(cuda_states, (list, tuple)):
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8) for item in cuda_states]
        )


def initialize_training(seed: int = P_SEED) -> tuple[SharedMultitilePHead, torch.optim.AdamW]:
    if seed != P_SEED:
        raise FormalPContractError("formal P seed is fixed at 17")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = SharedMultitilePHead()
    optimizer = make_frozen_p_optimizer(model)
    return model, optimizer


def episode_loss(
    model: SharedMultitilePHead, episode: Mapping[str, object]
) -> FusedPairLoss:
    target = episode["target"]
    rival = episode["rival"]
    if not isinstance(target, tuple) or not isinstance(rival, tuple):
        raise FormalPContractError("P episode directional records are absent")
    target_outputs = tuple(score_deployable_direction(model, item) for item in target)
    rival_outputs = tuple(score_deployable_direction(model, item) for item in rival)
    return fused_direction_pair_loss(
        target_a_to_b=target_outputs[0].candidate_utility,
        target_b_to_a=target_outputs[1].candidate_utility,
        rival_a_to_b=rival_outputs[0].candidate_utility,
        rival_b_to_a=rival_outputs[1].candidate_utility,
    )


def run_training_updates(
    *,
    model: SharedMultitilePHead,
    optimizer: torch.optim.AdamW,
    episodes: tuple[dict[str, object], ...],
    sampler: DeterministicEpisodeSampler,
    completed_updates: int,
    stop_update: int,
    trace: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if not 0 <= completed_updates <= stop_update <= P_UPDATES_TOTAL:
        raise FormalPContractError("P update interval is outside frozen budget")
    history = [] if trace is None else list(trace)
    if len(history) != completed_updates:
        raise FormalPContractError("P trace length/completed-update drift")
    for update in range(completed_updates + 1, stop_update + 1):
        learning_rate = p_learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        selected = sampler.draw(P_EPISODES_PER_UPDATE)
        optimizer.zero_grad(set_to_none=True)
        losses = [episode_loss(model, episodes[index]).loss for index in selected]
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient = model.linear.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FormalPContractError("P gradient is absent or non-finite")
        raw_gradient_sha = tensor_sha256(gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), P_GRADIENT_CLIP_L2)
        clipped_gradient_sha = tensor_sha256(model.linear.weight.grad)
        optimizer.step()
        history.append(
            {
                "completed_update": update,
                "episode_indices": list(selected),
                "loss_float64_hex": float(loss.detach()).hex(),
                "loss_tensor_sha256": tensor_sha256(loss.detach()),
                "raw_gradient_sha256": raw_gradient_sha,
                "clipped_gradient_sha256": clipped_gradient_sha,
                "parameter_sha256": tensor_sha256(model.linear.weight),
                "optimizer_state_sha256": state_sha256(optimizer.state_dict()),
                "next_sampler_state_sha256": state_sha256(sampler.state()),
                "learning_rate_float64_hex": float(learning_rate).hex(),
            }
        )
    return history


def make_training_state(
    *,
    model: SharedMultitilePHead,
    optimizer: torch.optim.AdamW,
    completed_updates: int,
    sampler: DeterministicEpisodeSampler,
    trace: Sequence[Mapping[str, object]],
    eligible_population_sha256: str,
    execution_protocol_sha256: str,
    p_implementation_sha256: str,
    fit_manifest_sha256: str,
    feature_ledger_sha256: str,
    pair_join_sha256: str,
) -> dict[str, object]:
    for name, value in (
        ("eligible population", eligible_population_sha256),
        ("execution protocol", execution_protocol_sha256),
        ("P implementation", p_implementation_sha256),
        ("fit manifest", fit_manifest_sha256),
        ("feature ledger", feature_ledger_sha256),
        ("pair join", pair_join_sha256),
    ):
        require_sha256(value, name=name)
    current_lr = 0.0 if completed_updates == 0 else p_learning_rate(completed_updates)
    rng = capture_rng_state()
    canonical = canonical_training_state_bytes(
        model,
        optimizer,
        completed_updates=completed_updates,
        current_learning_rate=current_lr,
        scheduler_cursor=completed_updates,
        sampler_state=sampler.state(),
        rng_state=rng,
        eligible_population_sha256=eligible_population_sha256,
        execution_protocol_sha256=execution_protocol_sha256,
        p_implementation_sha256=p_implementation_sha256,
    )
    state: dict[str, object] = {
        "schema_version": RESUME_SCHEMA,
        "completed_updates": completed_updates,
        "current_learning_rate": current_lr,
        "scheduler_cursor": completed_updates,
        "sampler_state": sampler.state(),
        "rng_state": rng,
        "eligible_population_sha256": eligible_population_sha256,
        "execution_protocol_sha256": execution_protocol_sha256,
        "p_implementation_sha256": p_implementation_sha256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "fit_manifest_sha256": fit_manifest_sha256,
        "feature_ledger_sha256": feature_ledger_sha256,
        "pair_join_sha256": pair_join_sha256,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "trace": [dict(item) for item in trace],
        "canonical_training_state_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    state["state_sha256"] = state_sha256(
        {key: item for key, item in state.items() if key != "state_sha256"}
    )
    return state


def restore_training_state(
    value: Mapping[str, object],
    *,
    episode_keys: tuple[PEpisodeKey, ...],
    expected_fit_manifest_sha256: str,
    expected_feature_ledger_sha256: str,
    expected_pair_join_sha256: str,
    expected_execution_protocol_sha256: str,
    expected_p_implementation_sha256: str,
) -> tuple[
    SharedMultitilePHead,
    torch.optim.AdamW,
    DeterministicEpisodeSampler,
    int,
    list[dict[str, object]],
]:
    if value.get("schema_version") != RESUME_SCHEMA:
        raise FormalPContractError("P resume schema drift")
    expected = {
        "fit_manifest_sha256": expected_fit_manifest_sha256,
        "feature_ledger_sha256": expected_feature_ledger_sha256,
        "pair_join_sha256": expected_pair_join_sha256,
        "execution_protocol_sha256": expected_execution_protocol_sha256,
        "p_implementation_sha256": expected_p_implementation_sha256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise FormalPContractError("P resume binding drift")
    expected_state_hash = state_sha256(
        {key: item for key, item in value.items() if key != "state_sha256"}
    )
    if value.get("state_sha256") != expected_state_hash:
        raise FormalPContractError("P resume state hash drift")
    model, optimizer = initialize_training()
    model.load_state_dict(value["model_state"], strict=True)  # type: ignore[arg-type]
    optimizer.load_state_dict(value["optimizer_state"])  # type: ignore[arg-type]
    sampler_state = value["sampler_state"]
    if not isinstance(sampler_state, Mapping):
        raise FormalPContractError("P resume sampler state absent")
    sampler = DeterministicEpisodeSampler.restore(episode_keys, sampler_state)
    completed = int(value["completed_updates"])
    if value.get("scheduler_cursor") != completed or len(value.get("trace", [])) != completed:
        raise FormalPContractError("P resume cursor/trace drift")
    rng = value["rng_state"]
    if not isinstance(rng, Mapping):
        raise FormalPContractError("P resume RNG state absent")
    restore_rng_state(rng)
    return model, optimizer, sampler, completed, [dict(item) for item in value["trace"]]  # type: ignore[arg-type]


def build_p_lock_record(
    output: DeployableDirectionOutput,
    *,
    crossfit_role: str,
    outer_fold: int,
    p_checkpoint_sha256: str,
    p_training_manifest_sha256: str,
) -> dict[str, object]:
    record = output.record
    require_sha256(p_checkpoint_sha256, name="P checkpoint")
    require_sha256(p_training_manifest_sha256, name="P training manifest")
    if not isinstance(crossfit_role, str) or not crossfit_role:
        raise FormalPContractError("P lock crossfit role is empty")
    if outer_fold not in (1, 2, 3, 4):
        raise FormalPContractError("P lock outer fold must be 1..4")
    regions = tuple(
        enumerate_superregion_bank(record.colnomic_query_grid_shape, include_r0_control=False)
    )
    all_root_ordinals = tuple(range(len(output.selected_action_indices)))
    all_root_action_keys: list[str | None] = []
    all_root_deployments: list[RootActionDeployment] = []
    for root, action_index in zip(
        all_root_ordinals, output.selected_action_indices, strict=True
    ):
        if action_index is None:
            deployment = make_root_action_deployment(
                action_key="0" * 64,
                eligible=False,
                query_mask=torch.zeros(
                    math.prod(record.dino_query_grid_shape), dtype=torch.bool
                ),
                reference_mask=torch.zeros(
                    math.prod(record.dino_reference_grid_shape), dtype=torch.bool
                ),
                query_grid_shape=record.dino_query_grid_shape,
                reference_grid_shape=record.dino_reference_grid_shape,
                query_geometry_sha256=record.query_geometry_sha256,
                reference_geometry_sha256=record.reference_geometry_sha256,
            )
            all_root_action_keys.append(None)
        else:
            deployment = record.deployments_by_root[root][action_index]
            if (
                not deployment.eligible
                or deployment.binding_status != ROOT_READY
                or output.selected_action_keys[root] != deployment.action_key
            ):
                raise FormalPContractError(
                    "selected P root action/deployment binding drift"
                )
            all_root_action_keys.append(deployment.action_key)
        all_root_deployments.append(deployment)
    if output.map_row_index is None:
        selected_ordinal: int | None = None
        selected_sha: str | None = None
        roots: tuple[int, ...] = ()
        deployments: tuple[RootActionDeployment, ...] = ()
        query_union = torch.zeros(math.prod(record.dino_query_grid_shape), dtype=torch.bool)
        status = LOCK_H0
    else:
        selected_ordinal = output.map_row_index
        region = regions[selected_ordinal]
        selected_sha = superregion_sha256(region)
        roots = region.contributing_root_ordinals
        chosen = []
        for root in roots:
            action_index = output.selected_action_indices[root]
            if action_index is None:
                raise FormalPContractError("READY P row references an absent root action")
            deployment = record.deployments_by_root[root][action_index]
            if not deployment.eligible:
                raise FormalPContractError("READY P row references geometry-H0 action")
            chosen.append(deployment)
        deployments = tuple(chosen)
        query_union = torch.stack([item.query_mask for item in deployments]).any(dim=0)
        if not _component_is_legal(query_union, record.dino_query_grid_shape):
            raise FormalPContractError("READY P lock query union is not connected")
        status = LOCK_READY
    per_radius = {
        str(radius): sum(item.radius == radius for item in regions) for radius in (1, 2, 3, 4)
    }
    core: dict[str, object] = {
        "query_id": record.query_id,
        "historical_query_ordinal": record.historical_query_ordinal,
        "execution_ordinal": record.execution_ordinal,
        "query_source_image_sha256": record.query_source_image_sha256,
        "outer_fold": outer_fold,
        "crossfit_role": crossfit_role,
        "p_checkpoint_sha256": p_checkpoint_sha256,
        "p_training_manifest_sha256": p_training_manifest_sha256,
        "candidate_physical_row": record.candidate_physical_row,
        "candidate_reference_source_sha256": record.candidate_reference_source_sha256,
        "direction": record.direction,
        "complete_bank_sha256": record.complete_bank_sha256,
        "selected_bank_ordinal": selected_ordinal,
        "selected_row_sha256": selected_sha,
        "status": status,
        "query_union_mask_rle": encode_mask_rle(query_union),
        "query_union_mask_sha256": tensor_sha256(query_union),
        "ordered_root_ordinals": list(roots),
        "root_query_tile_mask_rle": [encode_mask_rle(item.query_mask) for item in deployments],
        "root_query_tile_mask_sha256": [tensor_sha256(item.query_mask) for item in deployments],
        "root_reference_component_mask_rle": [
            encode_mask_rle(item.reference_mask) for item in deployments
        ],
        "root_reference_component_mask_sha256": [
            tensor_sha256(item.reference_mask) for item in deployments
        ],
        "root_binding_status": [item.binding_status for item in deployments],
        "all_root_ordinals": list(all_root_ordinals),
        "all_root_action_key_sha256": all_root_action_keys,
        "all_root_query_tile_mask_rle": [
            encode_mask_rle(item.query_mask) for item in all_root_deployments
        ],
        "all_root_query_tile_mask_sha256": [
            tensor_sha256(item.query_mask) for item in all_root_deployments
        ],
        "all_root_reference_component_mask_rle": [
            encode_mask_rle(item.reference_mask) for item in all_root_deployments
        ],
        "all_root_reference_component_mask_sha256": [
            tensor_sha256(item.reference_mask) for item in all_root_deployments
        ],
        "all_root_binding_status": [
            item.binding_status for item in all_root_deployments
        ],
        "query_geometry_sha256": record.query_geometry_sha256,
        "reference_geometry_sha256": record.reference_geometry_sha256,
        "fixed_denominator": {
            "radii": [1, 2, 3, 4],
            "row_count_by_radius": per_radius,
            "total_row_count": len(regions),
            "ineligible_rows_retained_as_exact_zero": True,
            "available_root_renormalization": False,
        },
        "erased_fields": list(P_LOCK_ERASED_FIELDS),
    }
    if set(core) != set(P_LOCK_REQUIRED_FIELDS) - {"record_sha256"}:
        raise FormalPContractError("P lock core field set drift")
    core["record_sha256"] = canonical_sha256(core)
    validate_p_lock_record(
        core,
        query_grid_shape=record.dino_query_grid_shape,
        reference_grid_shape=record.dino_reference_grid_shape,
    )
    return core


def validate_p_lock_record(
    value: Mapping[str, object],
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> None:
    if set(value) != set(P_LOCK_REQUIRED_FIELDS):
        raise FormalPContractError("P lock required field set drift")
    if value.get("record_sha256") != canonical_sha256(
        {key: item for key, item in value.items() if key != "record_sha256"}
    ):
        raise FormalPContractError("P lock record hash drift")
    if tuple(value.get("erased_fields", [])) != P_LOCK_ERASED_FIELDS:
        raise FormalPContractError("P lock erased-field receipt drift")
    if any(key in value for key in P_LOCK_ERASED_FIELDS):
        raise FormalPContractError("P lock exposes erased field")
    q_numel = math.prod(_grid(query_grid_shape, name="P lock query grid"))
    r_numel = math.prod(_grid(reference_grid_shape, name="P lock reference grid"))
    query_union = decode_mask_rle(value["query_union_mask_rle"], numel=q_numel)
    if tensor_sha256(query_union) != value.get("query_union_mask_sha256"):
        raise FormalPContractError("P lock query union tensor hash drift")
    roots = value.get("ordered_root_ordinals")
    q_rle = value.get("root_query_tile_mask_rle")
    q_sha = value.get("root_query_tile_mask_sha256")
    r_rle = value.get("root_reference_component_mask_rle")
    r_sha = value.get("root_reference_component_mask_sha256")
    statuses = value.get("root_binding_status")
    if not all(isinstance(item, list) for item in (roots, q_rle, q_sha, r_rle, r_sha, statuses)):
        raise FormalPContractError("P lock root arrays are absent")
    count = len(roots)
    if any(len(item) != count for item in (q_rle, q_sha, r_rle, r_sha, statuses)):
        raise FormalPContractError("P lock root-array arity drift")
    if roots != sorted(set(roots)):
        raise FormalPContractError("P lock root ordinals are not canonical")
    q_masks = []
    for q_encoded, q_hash, r_encoded, r_hash, status in zip(
        q_rle, q_sha, r_rle, r_sha, statuses, strict=True
    ):
        q_mask = decode_mask_rle(q_encoded, numel=q_numel)
        r_mask = decode_mask_rle(r_encoded, numel=r_numel)
        if tensor_sha256(q_mask) != q_hash or tensor_sha256(r_mask) != r_hash:
            raise FormalPContractError("P lock root mask tensor hash drift")
        if status != ROOT_READY:
            raise FormalPContractError("sealed H1 root binding is not READY")
        if not _component_is_legal(q_mask, query_grid_shape) or not _component_is_legal(
            r_mask, reference_grid_shape
        ):
            raise FormalPContractError("P lock root component is illegal")
        q_masks.append(q_mask)
    all_roots = value.get("all_root_ordinals")
    all_keys = value.get("all_root_action_key_sha256")
    all_q_rle = value.get("all_root_query_tile_mask_rle")
    all_q_sha = value.get("all_root_query_tile_mask_sha256")
    all_r_rle = value.get("all_root_reference_component_mask_rle")
    all_r_sha = value.get("all_root_reference_component_mask_sha256")
    all_statuses = value.get("all_root_binding_status")
    all_arrays = (
        all_roots,
        all_keys,
        all_q_rle,
        all_q_sha,
        all_r_rle,
        all_r_sha,
        all_statuses,
    )
    if not all(isinstance(item, list) for item in all_arrays):
        raise FormalPContractError("P lock complete root ledger is absent")
    all_count = len(all_roots)
    if all_count == 0 or any(len(item) != all_count for item in all_arrays[1:]):
        raise FormalPContractError("P lock complete root-ledger arity drift")
    if all_roots != list(range(all_count)):
        raise FormalPContractError("P lock complete root ordinals are not canonical")
    all_q_masks: dict[int, torch.Tensor] = {}
    all_r_masks: dict[int, torch.Tensor] = {}
    for root, action_key, q_encoded, q_hash, r_encoded, r_hash, status in zip(
        all_roots,
        all_keys,
        all_q_rle,
        all_q_sha,
        all_r_rle,
        all_r_sha,
        all_statuses,
        strict=True,
    ):
        q_mask = decode_mask_rle(q_encoded, numel=q_numel)
        r_mask = decode_mask_rle(r_encoded, numel=r_numel)
        if tensor_sha256(q_mask) != q_hash or tensor_sha256(r_mask) != r_hash:
            raise FormalPContractError("P lock complete-root mask hash drift")
        if status == ROOT_READY:
            require_sha256(action_key, name="P lock selected root action")
            if not _component_is_legal(q_mask, query_grid_shape) or not _component_is_legal(
                r_mask, reference_grid_shape
            ):
                raise FormalPContractError("P lock complete READY root is illegal")
        elif status == ROOT_H0:
            if action_key is not None or bool(q_mask.any()) or bool(r_mask.any()):
                raise FormalPContractError(
                    "P lock complete H0 root must erase action and masks"
                )
        else:
            raise FormalPContractError("P lock complete root status drift")
        all_q_masks[int(root)] = q_mask
        all_r_masks[int(root)] = r_mask
    for index, root in enumerate(roots):
        if (
            all_statuses[root] != ROOT_READY
            or not torch.equal(q_masks[index], all_q_masks[root])
            or not torch.equal(
                decode_mask_rle(r_rle[index], numel=r_numel), all_r_masks[root]
            )
        ):
            raise FormalPContractError(
                "selected row root does not close against complete root ledger"
            )
    if value.get("status") == LOCK_H0:
        if count or bool(query_union.any()) or value.get("selected_bank_ordinal") is not None or value.get("selected_row_sha256") is not None:
            raise FormalPContractError("P H0 lock exposes a selected hypothesis")
    elif value.get("status") == LOCK_READY:
        if count < MIN_READY_ROOTS or value.get("selected_bank_ordinal") is None:
            raise FormalPContractError("P READY lock lacks a multi-root hypothesis")
        if not torch.equal(torch.stack(q_masks).any(dim=0), query_union):
            raise FormalPContractError("P lock query union does not close root tiles")
        if not _component_is_legal(query_union, query_grid_shape):
            raise FormalPContractError("P lock query union is not connected")
    else:
        raise FormalPContractError("unknown P lock status")
    require_sha256(value["query_geometry_sha256"], name="P lock query geometry")
    require_sha256(value["reference_geometry_sha256"], name="P lock reference geometry")


def checkpoint_model(value: Mapping[str, object]) -> SharedMultitilePHead:
    if value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise FormalPContractError("P checkpoint schema drift")
    model = SharedMultitilePHead()
    model.load_state_dict(value["model_state"], strict=True)  # type: ignore[arg-type]
    if tensor_sha256(model.linear.weight) != value.get("model_weight_sha256"):
        raise FormalPContractError("P checkpoint weight hash drift")
    expected_logical = canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"model_state", "logical_sha256"}
        }
        | {"model_state_sha256": tensor_sha256(model.linear.weight)}
    )
    if value.get("logical_sha256") != expected_logical:
        raise FormalPContractError("P checkpoint logical hash drift")
    return model


__all__ = [
    "SCHEMA_VERSION",
    "RAW_SOURCE_SCHEMA",
    "FEATURE_LEDGER_SCHEMA",
    "FEATURE_VALIDATION_SCHEMA",
    "FIT_MANIFEST_SCHEMA",
    "PAIR_JOIN_SCHEMA",
    "RESUME_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "FIT_RESULT_SCHEMA",
    "FIT_VALIDATION_SCHEMA",
    "LOCK_LEDGER_SCHEMA",
    "LOCK_VALIDATION_SCHEMA",
    "LOCK_MANIFEST_SCHEMA",
    "EXECUTION_NODE_SCHEMA",
    "COMPACT_SOURCE_INDEX_EXECUTION_NODE_SCHEMA",
    "LATEST_AUTHORITY_PATH_SENTINEL",
    "LATEST_AUTHORITY_SHA_SENTINEL",
    "EXECUTION_AUTHORITY_ENV",
    "P_DIRECTIONS",
    "ROOT_READY",
    "ROOT_H0",
    "LOCK_READY",
    "LOCK_H0",
    "VECTOR_SCALAR_MAX_ULPS",
    "FormalPContractError",
    "RootActionDeployment",
    "DirectionalFeatureRecord",
    "DeployableDirectionOutput",
    "FusedPairLoss",
    "DeterministicEpisodeSampler",
    "canonical_sha256",
    "file_sha256",
    "tensor_sha256",
    "state_sha256",
    "exclusive_json",
    "exclusive_torch",
    "load_torch_mapping",
    "load_json_mapping",
    "validate_natural_authority",
    "resolve_execution_authority_arguments",
    "execution_manifest_file_sha256",
    "argv_from_execution_manifest",
    "encode_mask_rle",
    "decode_mask_rle",
    "assert_no_forbidden_prejoin_keys",
    "make_root_action_deployment",
    "make_directional_feature_record",
    "serialize_directional_feature_record",
    "deserialize_directional_feature_record",
    "score_deployable_direction",
    "fuse_direction_utilities",
    "fused_direction_pair_loss",
    "feature_ledger_index",
    "feature_ledger_logical_sha256",
    "pair_join_episodes",
    "capture_rng_state",
    "restore_rng_state",
    "initialize_training",
    "episode_loss",
    "run_training_updates",
    "make_training_state",
    "restore_training_state",
    "build_p_lock_record",
    "validate_p_lock_record",
    "checkpoint_model",
]
