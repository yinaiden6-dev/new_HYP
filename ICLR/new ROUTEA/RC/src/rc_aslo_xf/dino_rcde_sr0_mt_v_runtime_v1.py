"""Formal sequential V runtime for the SR0 connected-multitile experiment.

This module is an additive orchestration layer over the frozen ordered-root
decoder in :mod:`dino_rcde_cw1_multitile_vdecode_v1`.  It deliberately does
not know how natural cache files are located and it never joins an outer
held-out target.  Natural training adapters may construct :class:`VPairEpisode`
objects only from inner-OOF P locks; target-free inference adapters construct a
complete :class:`VQueryBundle` and enumerate every canonical unordered C128
pair.

The runtime fixes four properties that were easy to lose in older runners:

* V is trained on the paired ordered-root arm only, after P is frozen;
* the symmetric four-term logit is computed before the single pair loss;
* one checkpoint/comparator/reducer lineage is reused for every arm/control;
* continuation state contains the sampler cursor, RNG, AdamW and LR state, so
  an interrupted run is bit-identical to an uninterrupted run on deterministic
  hardware.

There is no D1 score/rank input, target mask, fallback, arm-specific model, or
automatic stage advancement here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
    MANDATORY_ARMS,
)
from .dino_rcde_cw1_multitile_vdecode_v1 import (
    BINDING_REAL,
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    OrderedRootArmTermV1,
    QueryTokenFieldV1,
    RootPairDecodeReceiptV1,
    TERM_H0,
    TERM_NO_COMPARABLE_ROOT,
    TERM_READY,
    ThreeArmFixedDenominatorEvidenceV1,
    _decode_pair_masks,
    _reduce_arm,
    _zero_term,
    registered_reducer_source_sha256,
    token_tensor_sha256,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    LOCK_H0,
    LOCK_READY,
    ROOT_H0,
    ROOT_READY,
    decode_mask_rle,
    resolve_execution_authority_arguments,
    tensor_sha256 as p_tensor_sha256,
    validate_p_lock_record,
)
from .dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_v_runtime_v1"
TRAINING_EPISODE_SCHEMA = "rc_dino_rcde_sr0_mt_v_pair_episode_v1"
QUERY_BUNDLE_SCHEMA = "rc_dino_rcde_sr0_mt_v_query_bundle_v1"
EXECUTION_NODE_SCHEMA = "rc_dino_rcde_sr0_mt_p_execution_node_v1_20260815"
RESUME_SCHEMA = "rc_dino_rcde_sr0_mt_v_resume_v1"
CHECKPOINT_SCHEMA = "rc_dino_rcde_sr0_mt_v_checkpoint_v1"

SEED = 17
UPDATES = 2_048
QUERIES_PER_UPDATE = 4
WARMUP_UPDATES = 128
MAX_LR = 3.0e-4
FINAL_LR = 3.0e-5
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP_L2 = 1.0
TAU = 1.0
DELTA = 0.0
CANDIDATE_COUNT = 128
PAIR_COUNT = CANDIDATE_COUNT * (CANDIDATE_COUNT - 1) // 2
ORDER_NAMESPACE = "RCDE_SR0_MT_V_EPISODE_ORDER_V1"
DIRECTION_NAMESPACE = "RCDE_SR0_MT_V_PAIR_DIRECTION_V1"
QUERY_SCHEDULE_NAMESPACE = "RCDE_SR0_MT_V_TARGET_FREE_QUERY_ADDRESS_V1"

FORBIDDEN_TARGET_FREE_FIELDS = frozenset(
    {
        "target",
        "target_key",
        "target_label",
        "identity",
        "exact_label",
        "rival",
        "rival_key",
        "correctness",
        "d1_score",
        "d1_rank",
        "slot",
        "winner",
        "d1_gap",
    }
)


class SR0MTVContractError(ValueError):
    """A formal V training/inference contract was violated."""


def execution_manifest_argv(
    argv: Sequence[str], *, expected_program: str
) -> tuple[list[str], Path | None]:
    """Expand one exact JSON execution node without eval or shell expansion.

    Manifest mode is exclusive, exactly as in the formal P programs.  The
    returned path lets a runner bind the physical manifest SHA in its result
    without adding a second command-line source of truth.
    """

    values = list(argv)
    if "--execution-manifest" not in values:
        return values, None
    if len(values) != 2 or values[0] != "--execution-manifest":
        raise SR0MTVContractError(
            "execution-manifest mode cannot be mixed with explicit arguments"
        )
    path = Path(values[1]).resolve()
    if not path.is_file() or path.is_symlink():
        raise SR0MTVContractError("execution manifest is absent or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "program",
        "arguments",
        "logical_sha256",
    }:
        raise SR0MTVContractError("execution manifest top-level schema drift")
    logical = hashlib.sha256(
        json.dumps(
            {key: item for key, item in payload.items() if key != "logical_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        payload.get("schema_version") != EXECUTION_NODE_SCHEMA
        or payload.get("program") != expected_program
        or payload.get("logical_sha256") != logical
    ):
        raise SR0MTVContractError("execution manifest program/hash drift")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping) or not arguments:
        raise SR0MTVContractError("execution manifest arguments are absent")
    arguments = resolve_execution_authority_arguments(arguments)
    expanded: list[str] = []
    for key, item in arguments.items():
        if (
            not isinstance(key, str)
            or not key
            or key.startswith("-")
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in key)
        ):
            raise SR0MTVContractError("execution manifest argument name is unsafe")
        option = "--" + key.replace("_", "-")
        if isinstance(item, bool):
            if item:
                expanded.append(option)
        elif item is None:
            continue
        elif isinstance(item, (str, int, float)):
            expanded.extend((option, str(item)))
        else:
            raise SR0MTVContractError(
                "execution manifest arguments must be scalar JSON values"
            )
    return expanded, path


def execution_manifest_arguments(
    path: Path, *, expected_program: str
) -> Mapping[str, object]:
    """Independently replay and return the exact manifest argument mapping."""

    _, observed = execution_manifest_argv(
        ["--execution-manifest", str(path)], expected_program=expected_program
    )
    if observed != path.resolve():
        raise SR0MTVContractError("execution manifest path replay drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MappingProxyType(
        resolve_execution_authority_arguments(dict(payload["arguments"]))
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def hash_parts(namespace: str, *parts: object) -> str:
    payload = b"\0".join(
        (namespace.encode("utf-8"), *(str(item).encode("utf-8") for item in parts))
    )
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(_json_bytes(list(tensor.shape)) + b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor] | torch.nn.Module) -> str:
    mapping = state.state_dict() if isinstance(state, torch.nn.Module) else state
    digest = hashlib.sha256()
    for name, value in sorted(mapping.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(_json_bytes(list(tensor.shape)) + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def registered_pair_comparator_sha256() -> str:
    """Hash the exact comparator definition inherited by every V arm."""

    return hashlib.sha256(
        inspect.getsource(DINO_RCDE_V1_2.compare_relational).encode("utf-8")
    ).hexdigest()


def bind_runtime_lineage(model: torch.nn.Module) -> tuple[str, str, str]:
    """Bind the live state and frozen comparator/reducer to decoder receipts."""

    model_sha = state_dict_sha256(model)
    comparator_sha = registered_pair_comparator_sha256()
    reducer_sha = registered_reducer_source_sha256()
    # The frozen decoder checks these receipts immediately before forward.
    setattr(model, "model_checkpoint_sha256", model_sha)
    setattr(model, "pair_comparator_sha256", comparator_sha)
    return model_sha, comparator_sha, reducer_sha


class DeviceMaskRuntimeAdapter:
    """Move boolean masks to the token device without changing membership.

    The frozen provenance dataclasses intentionally store canonical CPU masks.
    Natural V execution is on CUDA, so the adapter performs the sole allowed
    device transfer at the model boundary.  It delegates every learned
    operation to the same underlying model and introduces no parameter.
    """

    def __init__(self, model: torch.nn.Module, model_sha: str, comparator_sha: str):
        self.model = model
        self.model_checkpoint_sha256 = model_sha
        self.pair_comparator_sha256 = comparator_sha

    def decode_candidate(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        **kwargs: Any,
    ) -> Any:
        return self.model.decode_candidate(
            query_layers,
            reference_layers,
            torch.as_tensor(query_mask, dtype=torch.bool, device=query_layers.device),
            torch.as_tensor(
                reference_mask, dtype=torch.bool, device=reference_layers.device
            ),
            query_grid,
            reference_grid,
            **kwargs,
        )

    def compare_relational(
        self,
        relational_g: torch.Tensor,
        relational_c: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> Any:
        return self.model.compare_relational(
            relational_g,
            relational_c,
            torch.as_tensor(query_mask, dtype=torch.bool, device=relational_g.device),
        )


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SR0MTVContractError(f"{name} must be a lowercase SHA256")
    return value


def _canonical_mask(
    value: torch.Tensor, shape: tuple[int, int], *, name: str
) -> torch.Tensor:
    mask = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous().flatten()
    if mask.shape != (math.prod(shape),):
        raise SR0MTVContractError(f"{name} mask/grid drift")
    return mask


@dataclass(frozen=True)
class SealedRootDinoScopeV1:
    """One score-erased, root-addressed DINO scope emitted by frozen P."""

    root_ordinal: int
    action_key_sha256: str | None
    query_mask: torch.Tensor
    reference_mask: torch.Tensor
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_mask_p_sha256: str
    reference_mask_p_sha256: str
    query_geometry_sha256: str
    reference_geometry_sha256: str
    binding_status: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.root_ordinal, bool)
            or not isinstance(self.root_ordinal, int)
            or self.root_ordinal < 0
        ):
            raise SR0MTVContractError("sealed root ordinal is invalid")
        q_shape = tuple(int(item) for item in self.query_grid_shape)
        r_shape = tuple(int(item) for item in self.reference_grid_shape)
        if len(q_shape) != 2 or len(r_shape) != 2 or any(
            item <= 0 for item in (*q_shape, *r_shape)
        ):
            raise SR0MTVContractError("sealed root grid shape is invalid")
        query = _canonical_mask(self.query_mask, q_shape, name="sealed root query")
        reference = _canonical_mask(
            self.reference_mask, r_shape, name="sealed root reference"
        )
        _require_sha256(self.query_mask_p_sha256, name="sealed query mask")
        _require_sha256(self.reference_mask_p_sha256, name="sealed reference mask")
        _require_sha256(self.query_geometry_sha256, name="sealed query geometry")
        _require_sha256(
            self.reference_geometry_sha256, name="sealed reference geometry"
        )
        if (
            p_tensor_sha256(query) != self.query_mask_p_sha256
            or p_tensor_sha256(reference) != self.reference_mask_p_sha256
        ):
            raise SR0MTVContractError("sealed root mask/P receipt drift")
        if self.binding_status == ROOT_READY:
            _require_sha256(self.action_key_sha256, name="sealed root action")
            if not bool(query.any()) or not bool(reference.any()):
                raise SR0MTVContractError("sealed READY root is empty")
        elif self.binding_status == ROOT_H0:
            if (
                self.action_key_sha256 is not None
                or bool(query.any())
                or bool(reference.any())
            ):
                raise SR0MTVContractError("sealed H0 root is not exact zero")
        else:
            raise SR0MTVContractError("sealed root binding status drift")
        object.__setattr__(self, "query_grid_shape", q_shape)
        object.__setattr__(self, "reference_grid_shape", r_shape)
        object.__setattr__(self, "query_mask", query)
        object.__setattr__(self, "reference_mask", reference)


@dataclass(frozen=True)
class SealedDirectionPLockV1:
    """Direct V-facing view of one immutable P-lock direction.

    It contains the selected bank-row hypothesis plus a score-erased action
    for every canonical root.  The latter is required when the opponent chose
    a different row: V reads the opponent component at the *owner's* root,
    never reconstructs a population or chooses a fallback.
    """

    candidate_key: str
    candidate_physical_row: int
    query_source_image_sha256: str
    candidate_reference_source_sha256: str
    direction: str
    status: str
    selected_bank_ordinal: int | None
    selected_row_sha256: str | None
    query_union_mask: torch.Tensor
    ordered_root_ordinals: tuple[int, ...]
    all_roots: Mapping[int, SealedRootDinoScopeV1]
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_geometry_sha256: str
    reference_geometry_sha256: str
    p_lock_record_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise SR0MTVContractError("sealed direction candidate key is empty")
        if (
            isinstance(self.candidate_physical_row, bool)
            or not isinstance(self.candidate_physical_row, int)
            or self.candidate_physical_row < 0
        ):
            raise SR0MTVContractError("sealed direction physical row is invalid")
        if self.direction not in FIXED_DIRECTIONS:
            raise SR0MTVContractError("sealed direction name drift")
        for name, digest in (
            ("query source image", self.query_source_image_sha256),
            ("reference source image", self.candidate_reference_source_sha256),
            ("query geometry", self.query_geometry_sha256),
            ("reference geometry", self.reference_geometry_sha256),
            ("P-lock record", self.p_lock_record_sha256),
        ):
            _require_sha256(digest, name=name)
        q_shape = tuple(int(item) for item in self.query_grid_shape)
        r_shape = tuple(int(item) for item in self.reference_grid_shape)
        union = _canonical_mask(
            self.query_union_mask, q_shape, name="sealed direction query union"
        )
        roots = dict(self.all_roots)
        if not roots or tuple(sorted(roots)) != tuple(range(len(roots))):
            raise SR0MTVContractError("sealed direction complete roots are not canonical")
        for root, scope in roots.items():
            if (
                root != scope.root_ordinal
                or scope.query_grid_shape != q_shape
                or scope.reference_grid_shape != r_shape
                or scope.query_geometry_sha256 != self.query_geometry_sha256
                or scope.reference_geometry_sha256 != self.reference_geometry_sha256
            ):
                raise SR0MTVContractError("sealed direction root binding drift")
        selected = tuple(int(item) for item in self.ordered_root_ordinals)
        if selected != tuple(sorted(set(selected))) or not set(selected).issubset(roots):
            raise SR0MTVContractError("sealed selected root order drift")
        if self.status == LOCK_H0:
            if (
                selected
                or bool(union.any())
                or self.selected_bank_ordinal is not None
                or self.selected_row_sha256 is not None
            ):
                raise SR0MTVContractError("sealed direction H0 exposes an H1")
        elif self.status == LOCK_READY:
            if (
                len(selected) < 2
                or self.selected_bank_ordinal is None
                or self.selected_bank_ordinal < 0
            ):
                raise SR0MTVContractError("sealed READY direction lacks a multi-root row")
            _require_sha256(self.selected_row_sha256, name="selected P row")
            selected_scopes = [roots[root] for root in selected]
            if any(item.binding_status != ROOT_READY for item in selected_scopes):
                raise SR0MTVContractError("sealed READY row contains an H0 root")
            expected_union = torch.stack(
                [item.query_mask for item in selected_scopes]
            ).any(dim=0)
            if not torch.equal(expected_union, union):
                raise SR0MTVContractError("sealed selected roots do not close query union")
        else:
            raise SR0MTVContractError("sealed direction lock status drift")
        object.__setattr__(self, "query_grid_shape", q_shape)
        object.__setattr__(self, "reference_grid_shape", r_shape)
        object.__setattr__(self, "query_union_mask", union)
        object.__setattr__(self, "ordered_root_ordinals", selected)
        object.__setattr__(self, "all_roots", MappingProxyType(roots))

    @property
    def selected_roots(self) -> tuple[SealedRootDinoScopeV1, ...]:
        return tuple(self.all_roots[root] for root in self.ordered_root_ordinals)

    def __reduce__(self):
        return (
            type(self),
            (
                self.candidate_key,
                self.candidate_physical_row,
                self.query_source_image_sha256,
                self.candidate_reference_source_sha256,
                self.direction,
                self.status,
                self.selected_bank_ordinal,
                self.selected_row_sha256,
                self.query_union_mask,
                self.ordered_root_ordinals,
                dict(self.all_roots),
                self.query_grid_shape,
                self.reference_grid_shape,
                self.query_geometry_sha256,
                self.reference_geometry_sha256,
                self.p_lock_record_sha256,
            ),
        )


@dataclass(frozen=True)
class CandidatePLockV1:
    """One candidate's DINO tokens and two direct, score-erased P locks."""

    candidate: CandidateReferenceFieldV1
    direction_locks: Mapping[str, SealedDirectionPLockV1]

    def __post_init__(self) -> None:
        locks = dict(self.direction_locks)
        if set(locks) != set(FIXED_DIRECTIONS):
            raise SR0MTVContractError("candidate P lock must contain exactly two directions")
        for direction in FIXED_DIRECTIONS:
            lock = locks[direction]
            if (
                lock.direction != direction
                or lock.candidate_key != self.candidate.candidate_key
                or lock.candidate_physical_row != self.candidate.physical_gallery_row
                or lock.candidate_reference_source_sha256
                != self.candidate.source_image_sha256
                or lock.reference_grid_shape != self.candidate.grid_shape
                or lock.reference_geometry_sha256
                != self.candidate.geometry_record_sha256
            ):
                raise SR0MTVContractError("candidate/direct P-lock binding drift")
        query_sources = {item.query_source_image_sha256 for item in locks.values()}
        query_grids = {item.query_grid_shape for item in locks.values()}
        query_geometries = {item.query_geometry_sha256 for item in locks.values()}
        if len(query_sources) != 1 or len(query_grids) != 1 or len(query_geometries) != 1:
            raise SR0MTVContractError("candidate directions disagree on query provenance")
        object.__setattr__(self, "direction_locks", MappingProxyType(locks))

    @property
    def p_lock_record_sha256_by_direction(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                direction: lock.p_lock_record_sha256
                for direction, lock in self.direction_locks.items()
            }
        )

    @property
    def selected_bank_ordinal_by_direction(self) -> Mapping[str, int | None]:
        return MappingProxyType(
            {
                direction: lock.selected_bank_ordinal
                for direction, lock in self.direction_locks.items()
            }
        )

    def __reduce__(self):
        return (type(self), (self.candidate, dict(self.direction_locks)))


def sealed_direction_lock_from_p_record(
    value: Mapping[str, object],
    *,
    query: QueryTokenFieldV1,
    candidate: CandidateReferenceFieldV1,
) -> SealedDirectionPLockV1:
    """Validate and decode one formal JSON P lock without a P population."""

    validate_p_lock_record(
        value,
        query_grid_shape=query.grid_shape,
        reference_grid_shape=candidate.grid_shape,
    )
    if (
        value.get("query_source_image_sha256") != query.source_image_sha256
        or value.get("candidate_physical_row") != candidate.physical_gallery_row
        or value.get("candidate_reference_source_sha256")
        != candidate.source_image_sha256
        or value.get("query_geometry_sha256") != query.geometry_record_sha256
        or value.get("reference_geometry_sha256")
        != candidate.geometry_record_sha256
    ):
        raise SR0MTVContractError("P lock/token provenance join drift")
    all_roots: dict[int, SealedRootDinoScopeV1] = {}
    for root, action, q_rle, q_sha, r_rle, r_sha, status in zip(
        value["all_root_ordinals"],
        value["all_root_action_key_sha256"],
        value["all_root_query_tile_mask_rle"],
        value["all_root_query_tile_mask_sha256"],
        value["all_root_reference_component_mask_rle"],
        value["all_root_reference_component_mask_sha256"],
        value["all_root_binding_status"],
        strict=True,
    ):
        ordinal = int(root)
        all_roots[ordinal] = SealedRootDinoScopeV1(
            root_ordinal=ordinal,
            action_key_sha256=None if action is None else str(action),
            query_mask=decode_mask_rle(q_rle, numel=math.prod(query.grid_shape)),
            reference_mask=decode_mask_rle(
                r_rle, numel=math.prod(candidate.grid_shape)
            ),
            query_grid_shape=query.grid_shape,
            reference_grid_shape=candidate.grid_shape,
            query_mask_p_sha256=str(q_sha),
            reference_mask_p_sha256=str(r_sha),
            query_geometry_sha256=query.geometry_record_sha256,
            reference_geometry_sha256=candidate.geometry_record_sha256,
            binding_status=str(status),
        )
    return SealedDirectionPLockV1(
        candidate_key=candidate.candidate_key,
        candidate_physical_row=candidate.physical_gallery_row,
        query_source_image_sha256=query.source_image_sha256,
        candidate_reference_source_sha256=candidate.source_image_sha256,
        direction=str(value["direction"]),
        status=str(value["status"]),
        selected_bank_ordinal=(
            None
            if value["selected_bank_ordinal"] is None
            else int(value["selected_bank_ordinal"])
        ),
        selected_row_sha256=(
            None
            if value["selected_row_sha256"] is None
            else str(value["selected_row_sha256"])
        ),
        query_union_mask=decode_mask_rle(
            value["query_union_mask_rle"], numel=math.prod(query.grid_shape)
        ),
        ordered_root_ordinals=tuple(int(item) for item in value["ordered_root_ordinals"]),
        all_roots=all_roots,
        query_grid_shape=query.grid_shape,
        reference_grid_shape=candidate.grid_shape,
        query_geometry_sha256=query.geometry_record_sha256,
        reference_geometry_sha256=candidate.geometry_record_sha256,
        p_lock_record_sha256=str(value["record_sha256"]),
    )


@dataclass(frozen=True)
class VPairEpisode:
    """One training-only exact-pair/strongest-rival episode.

    ``target_key`` and ``strongest_rival_key`` are legal only in the inner-OOF
    training namespace.  This type must never be serialized into outer-heldout
    prejoin output.
    """

    episode_id: str
    query: QueryTokenFieldV1
    locks: Mapping[str, CandidatePLockV1]
    target_key: str
    strongest_rival_key: str
    outer_fold: int
    identity_key: str
    supergroup_key: str

    def __post_init__(self) -> None:
        locks = dict(self.locks)
        if not self.episode_id or not self.identity_key or not self.supergroup_key:
            raise SR0MTVContractError("training episode identifiers must be nonempty")
        if self.target_key == self.strongest_rival_key or set(locks) != {
            self.target_key,
            self.strongest_rival_key,
        }:
            raise SR0MTVContractError("training episode must contain target and one rival")
        if self.outer_fold not in (1, 2, 3, 4):
            raise SR0MTVContractError("outer fold must be one of 1..4")
        object.__setattr__(self, "locks", MappingProxyType(locks))

    def __reduce__(self):
        return (
            type(self),
            (
                self.episode_id,
                self.query,
                dict(self.locks),
                self.target_key,
                self.strongest_rival_key,
                self.outer_fold,
                self.identity_key,
                self.supergroup_key,
            ),
        )


@dataclass(frozen=True)
class VQueryBundle:
    """One target-free outer-heldout query with a complete natural C128 axis."""

    query_id: str
    execution_ordinal: int
    outer_fold: int
    query: QueryTokenFieldV1
    locks: Mapping[str, CandidatePLockV1]
    candidate_axis_sha256: str

    def __post_init__(self) -> None:
        locks = dict(self.locks)
        if not self.query_id or self.execution_ordinal < 0:
            raise SR0MTVContractError("query identity/address is invalid")
        if self.outer_fold not in (1, 2, 3, 4):
            raise SR0MTVContractError("outer fold must be one of 1..4")
        if len(locks) != CANDIDATE_COUNT:
            raise SR0MTVContractError("target-free inference requires complete natural C128")
        axis = canonical_candidate_axis(locks)
        observed = candidate_axis_sha256(
            [locks[key].candidate.physical_gallery_row for key in axis]
        )
        if observed != self.candidate_axis_sha256:
            raise SR0MTVContractError("candidate-axis receipt drift")
        object.__setattr__(self, "locks", MappingProxyType(locks))

    def __reduce__(self):
        return (
            type(self),
            (
                self.query_id,
                self.execution_ordinal,
                self.outer_fold,
                self.query,
                dict(self.locks),
                self.candidate_axis_sha256,
            ),
        )


def canonical_candidate_axis(locks: Mapping[str, CandidatePLockV1]) -> tuple[str, ...]:
    rows = []
    physical_rows: set[int] = set()
    for key, lock in locks.items():
        candidate = lock.candidate
        if key != candidate.candidate_key or candidate.physical_gallery_row in physical_rows:
            raise SR0MTVContractError("candidate key/physical-row axis is not one-to-one")
        physical_rows.add(candidate.physical_gallery_row)
        rows.append(
            (
                int(candidate.physical_gallery_row),
                candidate.source_image_sha256,
                candidate.candidate_key,
            )
        )
    return tuple(item[2] for item in sorted(rows))


def candidate_axis_sha256(physical_rows: Sequence[int]) -> str:
    """Replay the I0 canonical ascending-physical-row axis receipt exactly."""

    rows = [int(item) for item in physical_rows]
    if rows != sorted(rows) or len(rows) != len(set(rows)):
        raise SR0MTVContractError("candidate physical-row axis is not canonical ascending")
    return hashlib.sha256(_json_bytes(rows)).hexdigest()


def canonical_unordered_pairs(
    locks: Mapping[str, CandidatePLockV1],
) -> tuple[tuple[int, str, str], ...]:
    axis = canonical_candidate_axis(locks)
    pairs: list[tuple[int, str, str]] = []
    ordinal = 0
    for left_index, left in enumerate(axis[:-1]):
        for right in axis[left_index + 1 :]:
            pairs.append((ordinal, left, right))
            ordinal += 1
    expected = len(axis) * (len(axis) - 1) // 2
    if len(pairs) != expected:
        raise SR0MTVContractError("unordered pair enumeration drift")
    return tuple(pairs)


def _validate_decode_inputs(
    query: QueryTokenFieldV1,
    locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
) -> tuple[CandidatePLockV1, CandidatePLockV1]:
    if left_key == right_key:
        raise SR0MTVContractError("pair candidates must be distinct")
    try:
        left = locks[left_key]
        right = locks[right_key]
    except KeyError as error:
        raise SR0MTVContractError("pair candidate is absent") from error
    if (
        left.candidate.candidate_key != left_key
        or right.candidate.candidate_key != right_key
        or left.candidate.layers.device != query.layers.device
        or right.candidate.layers.device != query.layers.device
        or left.candidate.layers.dtype != query.layers.dtype
        or right.candidate.layers.dtype != query.layers.dtype
    ):
        raise SR0MTVContractError("pair candidate key/dtype/device drift")
    if token_tensor_sha256(query.layers) != query.tokens_sha256:
        raise SR0MTVContractError("query token tensor mutated after provenance seal")
    for lock in (left, right):
        candidate = lock.candidate
        if token_tensor_sha256(candidate.layers) != candidate.tokens_sha256:
            raise SR0MTVContractError(
                "reference token tensor mutated after provenance seal"
            )
        for direction in FIXED_DIRECTIONS:
            sealed = lock.direction_locks[direction]
            if (
                sealed.query_source_image_sha256 != query.source_image_sha256
                or sealed.query_grid_shape != query.grid_shape
                or sealed.query_geometry_sha256 != query.geometry_record_sha256
            ):
                raise SR0MTVContractError("query token/direct P-lock provenance drift")
    return left, right


def _all_patch_term(
    adapter: DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner: CandidateReferenceFieldV1,
    opponent: CandidateReferenceFieldV1,
    *,
    direction: str,
    streaming_chunk_size: int | None,
    consensus_tile_shape: tuple[int, int, int, int] | None,
) -> OrderedRootArmTermV1:
    pair, receipt = _decode_pair_masks(
        adapter,
        adapter,
        query,
        owner,
        opponent,
        query_mask=query.valid_patch_mask,
        owner_reference_mask=owner.valid_patch_mask,
        opponent_reference_mask=opponent.valid_patch_mask,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )
    valid = query.valid_patch_mask.to(pair.contributions.device)
    patch = pair.contributions * valid.to(pair.contributions.dtype)
    scalar = patch / valid.sum().to(patch.dtype)
    rebound = RootPairDecodeReceiptV1(
        root_ordinal=-1,
        owner_component_source_root_ordinal=None,
        opponent_component_source_root_ordinal=None,
        binding_mode=BINDING_REAL,
        query_mask_sha256=receipt.query_mask_sha256,
        owner_reference_mask_sha256=receipt.owner_reference_mask_sha256,
        opponent_reference_mask_sha256=receipt.opponent_reference_mask_sha256,
    )
    return OrderedRootArmTermV1(
        arm_name=ARM_ALL_PATCH,
        owner_candidate_key=owner.candidate_key,
        opponent_candidate_key=opponent.candidate_key,
        direction=direction,
        status=TERM_READY,
        query_mask=query.valid_patch_mask,
        patch_evidence=patch,
        scalar_contributions=scalar,
        logit=pair.logit,
        decoded_root_ordinals=(-1,),
        decode_receipts=(rebound,),
    )


def _query_full_term(
    adapter: DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner: CandidateReferenceFieldV1,
    opponent: CandidateReferenceFieldV1,
    owner_lock: SealedDirectionPLockV1,
    *,
    streaming_chunk_size: int | None,
    consensus_tile_shape: tuple[int, int, int, int] | None,
) -> OrderedRootArmTermV1:
    union = owner_lock.query_union_mask
    if owner_lock.status == LOCK_H0:
        return _zero_term(
            arm_name=ARM_QUERY_FULL_REFERENCE,
            owner_key=owner.candidate_key,
            opponent_key=opponent.candidate_key,
            direction=owner_lock.direction,
            status=TERM_H0,
            query_mask=union,
            exemplar=query.layers,
        )
    pair, receipt = _decode_pair_masks(
        adapter,
        adapter,
        query,
        owner,
        opponent,
        query_mask=union,
        owner_reference_mask=owner.valid_patch_mask,
        opponent_reference_mask=opponent.valid_patch_mask,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )
    qmask = union.to(pair.contributions.device)
    patch = pair.contributions * qmask.to(pair.contributions.dtype)
    scalar = patch / qmask.sum().to(patch.dtype)
    rebound = RootPairDecodeReceiptV1(
        root_ordinal=-1,
        owner_component_source_root_ordinal=None,
        opponent_component_source_root_ordinal=None,
        binding_mode=BINDING_REAL,
        query_mask_sha256=receipt.query_mask_sha256,
        owner_reference_mask_sha256=receipt.owner_reference_mask_sha256,
        opponent_reference_mask_sha256=receipt.opponent_reference_mask_sha256,
    )
    return OrderedRootArmTermV1(
        arm_name=ARM_QUERY_FULL_REFERENCE,
        owner_candidate_key=owner.candidate_key,
        opponent_candidate_key=opponent.candidate_key,
        direction=owner_lock.direction,
        status=TERM_READY,
        query_mask=union,
        patch_evidence=patch,
        scalar_contributions=scalar,
        logit=pair.logit,
        decoded_root_ordinals=(-1,),
        decode_receipts=(rebound,),
    )


def _paired_term(
    adapter: DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner: CandidateReferenceFieldV1,
    opponent: CandidateReferenceFieldV1,
    owner_lock: SealedDirectionPLockV1,
    opponent_lock: SealedDirectionPLockV1,
    *,
    streaming_chunk_size: int | None,
    consensus_tile_shape: tuple[int, int, int, int] | None,
) -> OrderedRootArmTermV1:
    union = owner_lock.query_union_mask
    if owner_lock.status == LOCK_H0:
        return _zero_term(
            arm_name=ARM_QUERY_LOCAL_COMPONENTS,
            owner_key=owner.candidate_key,
            opponent_key=opponent.candidate_key,
            direction=owner_lock.direction,
            status=TERM_H0,
            query_mask=union,
            exemplar=query.layers,
        )
    total = query.layers.new_zeros(query.valid_patch_mask.numel())
    coverage = torch.zeros_like(query.valid_patch_mask, dtype=torch.int64)
    for owner_root in owner_lock.selected_roots:
        coverage += owner_root.query_mask.to(torch.int64)
    receipts: list[RootPairDecodeReceiptV1] = []
    decoded: list[int] = []
    for root in owner_lock.ordered_root_ordinals:
        owner_root = owner_lock.all_roots[root]
        opponent_root = opponent_lock.all_roots[root]
        if opponent_root.binding_status != ROOT_READY:
            continue
        if not torch.equal(owner_root.query_mask, opponent_root.query_mask):
            raise SR0MTVContractError(
                "candidate-specific direct locks disagree on a fixed query root"
            )
        pair, receipt = _decode_pair_masks(
            adapter,
            adapter,
            query,
            owner,
            opponent,
            query_mask=owner_root.query_mask,
            owner_reference_mask=owner_root.reference_mask,
            opponent_reference_mask=opponent_root.reference_mask,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        qmask = owner_root.query_mask.to(pair.contributions.device)
        contribution = pair.contributions * qmask.to(pair.contributions.dtype)
        total = total + contribution
        receipts.append(
            RootPairDecodeReceiptV1(
                root_ordinal=root,
                owner_component_source_root_ordinal=root,
                opponent_component_source_root_ordinal=root,
                binding_mode=BINDING_REAL,
                query_mask_sha256=receipt.query_mask_sha256,
                owner_reference_mask_sha256=receipt.owner_reference_mask_sha256,
                opponent_reference_mask_sha256=receipt.opponent_reference_mask_sha256,
            )
        )
        decoded.append(root)
    if not decoded:
        return _zero_term(
            arm_name=ARM_QUERY_LOCAL_COMPONENTS,
            owner_key=owner.candidate_key,
            opponent_key=opponent.candidate_key,
            direction=owner_lock.direction,
            status=TERM_NO_COMPARABLE_ROOT,
            query_mask=union,
            exemplar=query.layers,
        )
    reciprocal = torch.zeros_like(total)
    covered = coverage > 0
    reciprocal[covered.to(reciprocal.device)] = 1.0 / coverage[covered].to(
        device=reciprocal.device, dtype=reciprocal.dtype
    )
    patch = total * reciprocal
    qmask = union.to(device=patch.device)
    patch = patch * qmask.to(patch.dtype)
    scalar = patch / qmask.sum().to(patch.dtype)
    return OrderedRootArmTermV1(
        arm_name=ARM_QUERY_LOCAL_COMPONENTS,
        owner_candidate_key=owner.candidate_key,
        opponent_candidate_key=opponent.candidate_key,
        direction=owner_lock.direction,
        status=TERM_READY,
        query_mask=union,
        patch_evidence=patch,
        scalar_contributions=scalar,
        logit=scalar.sum(),
        decoded_root_ordinals=tuple(decoded),
        decode_receipts=tuple(receipts),
    )


def decode_direct_arm_term(
    adapter: DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner_lock: CandidatePLockV1,
    opponent_lock: CandidatePLockV1,
    *,
    direction: str,
    arm_name: str,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> OrderedRootArmTermV1:
    """Decode one direct sealed-lock term; controls reuse this exact path."""

    owner = owner_lock.candidate
    opponent = opponent_lock.candidate
    if arm_name == ARM_ALL_PATCH:
        return _all_patch_term(
            adapter,
            query,
            owner,
            opponent,
            direction=direction,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
    sealed_owner = owner_lock.direction_locks[direction]
    sealed_opponent = opponent_lock.direction_locks[direction]
    if arm_name == ARM_QUERY_FULL_REFERENCE:
        return _query_full_term(
            adapter,
            query,
            owner,
            opponent,
            sealed_owner,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
    if arm_name == ARM_QUERY_LOCAL_COMPONENTS:
        return _paired_term(
            adapter,
            query,
            owner,
            opponent,
            sealed_owner,
            sealed_opponent,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
    raise SR0MTVContractError("unknown direct V arm")


def decode_pair(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> ThreeArmFixedDenominatorEvidenceV1:
    left_lock, right_lock = _validate_decode_inputs(
        query, locks, left_key, right_key
    )
    model_sha, comparator_sha, reducer_sha = bind_runtime_lineage(model)
    adapter = DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    arms = []
    for arm_name in MANDATORY_ARMS:
        forward = tuple(
            decode_direct_arm_term(
                adapter,
                query,
                left_lock,
                right_lock,
                direction=direction,
                arm_name=arm_name,
                streaming_chunk_size=streaming_chunk_size,
                consensus_tile_shape=consensus_tile_shape,
            )
            for direction in FIXED_DIRECTIONS
        )
        reverse = tuple(
            decode_direct_arm_term(
                adapter,
                query,
                right_lock,
                left_lock,
                direction=direction,
                arm_name=arm_name,
                streaming_chunk_size=streaming_chunk_size,
                consensus_tile_shape=consensus_tile_shape,
            )
            for direction in FIXED_DIRECTIONS
        )
        arms.append(
            _reduce_arm(
                arm_name,
                left_key,
                right_key,
                forward,  # type: ignore[arg-type]
                reverse,  # type: ignore[arg-type]
            )
        )
    return ThreeArmFixedDenominatorEvidenceV1(
        left_candidate_key=left_key,
        right_candidate_key=right_key,
        model_checkpoint_sha256=model_sha,
        pair_comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
        binding_mode=BINDING_REAL,
        arms=tuple(arms),  # type: ignore[arg-type]
    )


def decode_pair_training_selective_autograd(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> ThreeArmFixedDenominatorEvidenceV1:
    """Execute all frozen arms while retaining autograd only for the loss arm.

    V120's frozen pair loss reads only ``ARM_QUERY_LOCAL_COMPONENTS``.  The
    other two arms remain mandatory execution/receipt controls, but retaining
    their activation graphs cannot affect that loss or any parameter gradient.
    Running those controls under ``no_grad`` therefore preserves their exact
    forward values and receipts while removing zero-utility graph residency.
    The public ``decode_pair`` inference path above is deliberately unchanged.
    """

    left_lock, right_lock = _validate_decode_inputs(
        query, locks, left_key, right_key
    )
    model_sha, comparator_sha, reducer_sha = bind_runtime_lineage(model)
    adapter = DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    arms = []
    for arm_name in MANDATORY_ARMS:
        context = (
            torch.enable_grad()
            if arm_name == ARM_QUERY_LOCAL_COMPONENTS
            else torch.no_grad()
        )
        with context:
            forward = tuple(
                decode_direct_arm_term(
                    adapter,
                    query,
                    left_lock,
                    right_lock,
                    direction=direction,
                    arm_name=arm_name,
                    streaming_chunk_size=streaming_chunk_size,
                    consensus_tile_shape=consensus_tile_shape,
                )
                for direction in FIXED_DIRECTIONS
            )
            reverse = tuple(
                decode_direct_arm_term(
                    adapter,
                    query,
                    right_lock,
                    left_lock,
                    direction=direction,
                    arm_name=arm_name,
                    streaming_chunk_size=streaming_chunk_size,
                    consensus_tile_shape=consensus_tile_shape,
                )
                for direction in FIXED_DIRECTIONS
            )
            arms.append(
                _reduce_arm(
                    arm_name,
                    left_key,
                    right_key,
                    forward,  # type: ignore[arg-type]
                    reverse,  # type: ignore[arg-type]
                )
            )
    return ThreeArmFixedDenominatorEvidenceV1(
        left_candidate_key=left_key,
        right_candidate_key=right_key,
        model_checkpoint_sha256=model_sha,
        pair_comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
        binding_mode=BINDING_REAL,
        arms=tuple(arms),  # type: ignore[arg-type]
    )


def move_episode_to_device(
    episode: VPairEpisode, device: torch.device | str
) -> VPairEpisode:
    """Move token tensors only; immutable P/geometry masks remain canonical CPU."""

    target_device = torch.device(device)
    query = QueryTokenFieldV1(
        layers=episode.query.layers.to(target_device),
        grid_shape=episode.query.grid_shape,
        valid_patch_mask=episode.query.valid_patch_mask,
        source_image_sha256=episode.query.source_image_sha256,
        source_key=episode.query.source_key,
        cache_payload_sha256=episode.query.cache_payload_sha256,
        geometry_record_sha256=episode.query.geometry_record_sha256,
        tokens_sha256=episode.query.tokens_sha256,
    )
    locks: dict[str, CandidatePLockV1] = {}
    for key, lock in episode.locks.items():
        candidate = CandidateReferenceFieldV1(
            candidate_key=lock.candidate.candidate_key,
            layers=lock.candidate.layers.to(target_device),
            grid_shape=lock.candidate.grid_shape,
            valid_patch_mask=lock.candidate.valid_patch_mask,
            physical_gallery_row=lock.candidate.physical_gallery_row,
            source_image_sha256=lock.candidate.source_image_sha256,
            source_key=lock.candidate.source_key,
            cache_payload_sha256=lock.candidate.cache_payload_sha256,
            geometry_record_sha256=lock.candidate.geometry_record_sha256,
            tokens_sha256=lock.candidate.tokens_sha256,
        )
        locks[key] = CandidatePLockV1(
            candidate=candidate,
            direction_locks=lock.direction_locks,
        )
    return VPairEpisode(
        episode_id=episode.episode_id,
        query=query,
        locks=locks,
        target_key=episode.target_key,
        strongest_rival_key=episode.strongest_rival_key,
        outer_fold=episode.outer_fold,
        identity_key=episode.identity_key,
        supergroup_key=episode.supergroup_key,
    )


def move_locks_to_device(
    locks: Mapping[str, CandidatePLockV1], device: torch.device | str
) -> Mapping[str, CandidatePLockV1]:
    target_device = torch.device(device)
    output: dict[str, CandidatePLockV1] = {}
    for key, lock in locks.items():
        source = lock.candidate
        candidate = CandidateReferenceFieldV1(
            candidate_key=source.candidate_key,
            layers=source.layers.to(target_device),
            grid_shape=source.grid_shape,
            valid_patch_mask=source.valid_patch_mask,
            physical_gallery_row=source.physical_gallery_row,
            source_image_sha256=source.source_image_sha256,
            source_key=source.source_key,
            cache_payload_sha256=source.cache_payload_sha256,
            geometry_record_sha256=source.geometry_record_sha256,
            tokens_sha256=source.tokens_sha256,
        )
        output[key] = CandidatePLockV1(
            candidate=candidate,
            direction_locks=lock.direction_locks,
        )
    return MappingProxyType(output)


def move_query_bundle_to_device(
    bundle: VQueryBundle, device: torch.device | str
) -> VQueryBundle:
    target_device = torch.device(device)
    query = QueryTokenFieldV1(
        layers=bundle.query.layers.to(target_device),
        grid_shape=bundle.query.grid_shape,
        valid_patch_mask=bundle.query.valid_patch_mask,
        source_image_sha256=bundle.query.source_image_sha256,
        source_key=bundle.query.source_key,
        cache_payload_sha256=bundle.query.cache_payload_sha256,
        geometry_record_sha256=bundle.query.geometry_record_sha256,
        tokens_sha256=bundle.query.tokens_sha256,
    )
    return VQueryBundle(
        query_id=bundle.query_id,
        execution_ordinal=bundle.execution_ordinal,
        outer_fold=bundle.outer_fold,
        query=query,
        locks=move_locks_to_device(bundle.locks, target_device),
        candidate_axis_sha256=bundle.candidate_axis_sha256,
    )


def target_signed_paired_logit(
    model: torch.nn.Module, episode: VPairEpisode, *, target_first: bool
) -> torch.Tensor:
    left, right = (
        (episode.target_key, episode.strongest_rival_key)
        if target_first
        else (episode.strongest_rival_key, episode.target_key)
    )
    evidence = decode_pair_training_selective_autograd(
        model, episode.query, episode.locks, left, right
    )
    logit = evidence.by_name()[ARM_QUERY_LOCAL_COMPONENTS].logit
    return logit if target_first else -logit


def pair_loss(signed_logit: torch.Tensor) -> torch.Tensor:
    if signed_logit.ndim != 0 or not bool(torch.isfinite(signed_logit)):
        raise SR0MTVContractError("pair logit must be one finite scalar")
    return F.softplus(-((signed_logit - DELTA) / TAU))


def learning_rate(update_one_based: int) -> float:
    if not 1 <= update_one_based <= UPDATES:
        raise SR0MTVContractError("learning-rate update is outside 1..2048")
    if update_one_based <= WARMUP_UPDATES:
        return MAX_LR * update_one_based / WARMUP_UPDATES
    progress = (update_one_based - WARMUP_UPDATES) / (UPDATES - WARMUP_UPDATES)
    return FINAL_LR + 0.5 * (MAX_LR - FINAL_LR) * (
        1.0 + math.cos(math.pi * progress)
    )


def ordered_training_pool(
    episodes: Sequence[VPairEpisode], outer_fold: int
) -> tuple[VPairEpisode, ...]:
    if any(item.outer_fold != outer_fold for item in episodes):
        raise SR0MTVContractError("V fit mixed outer folds")
    keys = [episode_schedule_key(item) for item in episodes]
    if len(set(keys)) != len(episodes):
        raise SR0MTVContractError("duplicate target-free V query address")
    ordered = sorted(
        episodes,
        key=lambda item: hash_parts(
            ORDER_NAMESPACE, outer_fold, episode_schedule_key(item)
        ),
    )
    if len(ordered) < QUERIES_PER_UPDATE:
        raise SR0MTVContractError("unified V episode pool has fewer than four queries")
    return tuple(ordered)


def update_episodes(
    ordered: Sequence[VPairEpisode],
    update_zero_based: int,
) -> tuple[VPairEpisode, ...]:
    start = (QUERIES_PER_UPDATE * update_zero_based) % len(ordered)
    selected = tuple(
        ordered[(start + offset) % len(ordered)]
        for offset in range(QUERIES_PER_UPDATE)
    )
    if len({episode_schedule_key(item) for item in selected}) != QUERIES_PER_UPDATE:
        raise SR0MTVContractError("unified V update contains duplicate queries")
    return selected


def episode_schedule_key(episode: VPairEpisode) -> str:
    """A target/rival/identity-free address used by the V sampler only."""

    return hash_parts(
        QUERY_SCHEDULE_NAMESPACE,
        episode.outer_fold,
        episode.query.source_image_sha256,
        episode.query.source_key,
        episode.query.cache_payload_sha256,
    )


def target_first(outer_fold: int, update: int, query_schedule_key: str) -> bool:
    return int(
        hash_parts(DIRECTION_NAMESPACE, outer_fold, update, query_schedule_key), 16
    ) % 2 == 0


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass(frozen=True)
class VFitProgress:
    completed_updates: int
    loss_trace: tuple[Mapping[str, object], ...]
    schedule_sha256: str
    next_sampler_query_keys: tuple[str, ...]


def make_resume_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    outer_fold: int,
    completed_updates: int,
    loss_trace: Sequence[Mapping[str, object]],
    schedule_events: Sequence[Mapping[str, object]],
    ordered: Sequence[VPairEpisode],
    initialization_checkpoint_sha256: str,
) -> dict[str, Any]:
    next_items = update_episodes(ordered, completed_updates)
    schedule_digest = hashlib.sha256()
    for event in schedule_events:
        schedule_digest.update(_json_bytes(event) + b"\n")
    state = {
        "schema_version": RESUME_SCHEMA,
        "outer_fold": outer_fold,
        "seed": SEED,
        "completed_updates": completed_updates,
        "initialization_checkpoint_sha256": initialization_checkpoint_sha256,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_trace": list(loss_trace),
        "schedule_events": list(schedule_events),
        "schedule_sha256": schedule_digest.hexdigest(),
        "next_sampler_query_keys": [episode_schedule_key(item) for item in next_items],
        "next_learning_rate": learning_rate(completed_updates + 1)
        if completed_updates < UPDATES
        else None,
        "rng_state": capture_rng_state(),
    }
    state["model_state_sha256"] = state_dict_sha256(state["model_state_dict"])
    return state


def validate_resume_state(
    state: Mapping[str, Any],
    *,
    outer_fold: int,
    ordered: Sequence[VPairEpisode],
    initialization_checkpoint_sha256: str,
) -> None:
    required = {
        "schema_version",
        "outer_fold",
        "seed",
        "completed_updates",
        "initialization_checkpoint_sha256",
        "model_state_dict",
        "optimizer_state_dict",
        "loss_trace",
        "schedule_events",
        "schedule_sha256",
        "next_sampler_query_keys",
        "next_learning_rate",
        "rng_state",
        "model_state_sha256",
    }
    if set(state) != required:
        raise SR0MTVContractError("resume state field contract drift")
    completed = int(state["completed_updates"])
    if (
        state["schema_version"] != RESUME_SCHEMA
        or int(state["outer_fold"]) != outer_fold
        or int(state["seed"]) != SEED
        or not 0 <= completed <= UPDATES
        or state["initialization_checkpoint_sha256"]
        != initialization_checkpoint_sha256
        or state_dict_sha256(state["model_state_dict"])
        != state["model_state_sha256"]
        or len(state["loss_trace"]) != completed
        or len(state["schedule_events"]) != completed
    ):
        raise SR0MTVContractError("resume state binding/progress drift")
    digest = hashlib.sha256()
    for event in state["schedule_events"]:
        digest.update(_json_bytes(event) + b"\n")
    next_items = update_episodes(ordered, completed)
    expected_lr = learning_rate(completed + 1) if completed < UPDATES else None
    if (
        digest.hexdigest() != state["schedule_sha256"]
        or list(state["next_sampler_query_keys"])
        != [episode_schedule_key(item) for item in next_items]
        or state["next_learning_rate"] != expected_lr
    ):
        raise SR0MTVContractError("resume schedule/cursor drift")


def train_updates(
    model: torch.nn.Module,
    episodes: Sequence[VPairEpisode],
    *,
    outer_fold: int,
    initialization_checkpoint_sha256: str,
    stop_after_updates: int,
    resume_state: Mapping[str, Any] | None = None,
    loss_fn: Callable[[torch.nn.Module, VPairEpisode, bool], torch.Tensor] | None = None,
    update_observer: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
    """Run a deterministic prefix of the frozen 2048-update V recipe.

    ``loss_fn`` is dependency injection for the synthetic I1 continuation
    fixture only.  Natural execution must leave it ``None`` so the paired
    ordered-root decoder and four-term reducer are unavoidable.
    """

    if not 0 <= stop_after_updates <= UPDATES:
        raise SR0MTVContractError("requested V stop is outside 0..2048")
    ordered = ordered_training_pool(episodes, outer_fold)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY
    )
    loss_trace: list[Mapping[str, object]] = []
    schedule_events: list[Mapping[str, object]] = []
    completed = 0
    if resume_state is None:
        seed_everything()
    else:
        validate_resume_state(
            resume_state,
            outer_fold=outer_fold,
            ordered=ordered,
            initialization_checkpoint_sha256=initialization_checkpoint_sha256,
        )
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        loss_trace = list(resume_state["loss_trace"])
        schedule_events = list(resume_state["schedule_events"])
        completed = int(resume_state["completed_updates"])
        restore_rng_state(resume_state["rng_state"])
    if stop_after_updates < completed:
        raise SR0MTVContractError("resume cannot move the update cursor backwards")

    actual_loss = loss_fn or (
        lambda active_model, item, order: pair_loss(
            target_signed_paired_logit(active_model, item, target_first=order)
        )
    )
    model.train()
    while completed < stop_after_updates:
        update_one = completed + 1
        selected = update_episodes(ordered, completed)
        directions = tuple(
            target_first(
                outer_fold, update_one, episode_schedule_key(item)
            )
            for item in selected
        )
        event = {
            "update": update_one,
            "query_schedule_keys": [
                episode_schedule_key(item) for item in selected
            ],
            "target_first": list(directions),
        }
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(update_one)
        optimizer.zero_grad(set_to_none=True)
        # Natural episodes remain canonical CPU objects.  Moving the complete
        # fold pool to CUDA duplicates recurrent reference tensors and makes
        # activation residency depend on fold size rather than the frozen
        # four-query effective batch.  Stream exactly one selected episode at
        # a time and accumulate its one-quarter gradient.  This is the same
        # arithmetic-mean objective, with one clip and one optimizer step only
        # after all four scheduled queries have contributed.
        detached_losses: list[torch.Tensor] = []
        for item, direction in zip(selected, directions, strict=True):
            active_item = (
                move_episode_to_device(item, next(model.parameters()).device)
                if loss_fn is None
                else item
            )
            value = actual_loss(model, active_item, direction)
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise SR0MTVContractError("V loss is nonfinite or nonscalar")
            detached_losses.append(value.detach().cpu())
            (value / QUERIES_PER_UPDATE).backward()
            del value
            if active_item is not item:
                del active_item
        loss = torch.stack(detached_losses).mean()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP_L2
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise SR0MTVContractError("V gradient norm is nonfinite")
        optimizer.step()
        loss_trace.append(
            {
                "update": update_one,
                "mean_pair_loss": float(loss.detach().cpu()),
                "learning_rate": learning_rate(update_one),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "query_schedule_keys": event["query_schedule_keys"],
            }
        )
        schedule_events.append(event)
        completed = update_one
        if update_observer is not None:
            update_observer(
                {
                    "update": update_one,
                    "mean_pair_loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": float(
                        gradient_norm.detach().cpu()
                    ),
                    "query_schedule_keys": tuple(event["query_schedule_keys"]),
                }
            )
    return (
        make_resume_state(
            model,
            optimizer,
            outer_fold=outer_fold,
            completed_updates=completed,
            loss_trace=loss_trace,
            schedule_events=schedule_events,
            ordered=ordered,
            initialization_checkpoint_sha256=initialization_checkpoint_sha256,
        ),
        optimizer,
    )


def assert_target_free_payload(value: object, *, path: str = "root") -> None:
    """Reject any target/rival/D1 field before exhaustive pair inference."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_TARGET_FREE_FIELDS:
                raise SR0MTVContractError(
                    f"forbidden target-free field at {path}.{key}"
                )
            assert_target_free_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_target_free_payload(item, path=f"{path}[{index}]")


def serialize_arm_evidence(
    evidence: ThreeArmFixedDenominatorEvidenceV1,
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for runtime_name, arm in evidence.by_name().items():
        terms = (*arm.forward_by_direction, *arm.reverse_by_direction)
        reference_scope_payload = [
            [
                {
                    "root_ordinal": receipt.root_ordinal,
                    "owner_reference_mask_sha256": receipt.owner_reference_mask_sha256,
                    "opponent_reference_mask_sha256": receipt.opponent_reference_mask_sha256,
                }
                for receipt in term.decode_receipts
            ]
            for term in terms
        ]
        arms[runtime_name] = {
            "logit": float(arm.logit.detach().cpu()),
            "scalar_contributions": arm.scalar_contributions.detach().cpu(),
            "signed_term_logits": [
                float(item.detach().cpu()) for item in arm.signed_term_logits
            ],
            "forward_status": [item.status for item in arm.forward_by_direction],
            "reverse_status": [item.status for item in arm.reverse_by_direction],
            "forward_query_mask_sha256": [
                tensor_sha256(item.query_mask.to(torch.uint8))
                for item in arm.forward_by_direction
            ],
            "reverse_query_mask_sha256": [
                tensor_sha256(item.query_mask.to(torch.uint8))
                for item in arm.reverse_by_direction
            ],
            "reference_scope_sha256": hashlib.sha256(
                _json_bytes(reference_scope_payload)
            ).hexdigest(),
            "dual_candidate_legal_h1": all(
                item.status == "ORDERED_ROOT_TERM_READY" for item in terms
            ),
        }
    return {
        "left_candidate_key": evidence.left_candidate_key,
        "right_candidate_key": evidence.right_candidate_key,
        "model_checkpoint_sha256": evidence.model_checkpoint_sha256,
        "pair_comparator_sha256": evidence.pair_comparator_sha256,
        "reducer_sha256": evidence.reducer_sha256,
        "binding_mode": evidence.binding_mode,
        "arms": arms,
    }


__all__ = [
    "SCHEMA_VERSION",
    "TRAINING_EPISODE_SCHEMA",
    "QUERY_BUNDLE_SCHEMA",
    "EXECUTION_NODE_SCHEMA",
    "RESUME_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "SEED",
    "UPDATES",
    "QUERIES_PER_UPDATE",
    "WARMUP_UPDATES",
    "MAX_LR",
    "FINAL_LR",
    "WEIGHT_DECAY",
    "GRADIENT_CLIP_L2",
    "CANDIDATE_COUNT",
    "PAIR_COUNT",
    "FORBIDDEN_TARGET_FREE_FIELDS",
    "SR0MTVContractError",
    "execution_manifest_argv",
    "execution_manifest_arguments",
    "SealedRootDinoScopeV1",
    "SealedDirectionPLockV1",
    "CandidatePLockV1",
    "sealed_direction_lock_from_p_record",
    "VPairEpisode",
    "VQueryBundle",
    "canonical_candidate_axis",
    "candidate_axis_sha256",
    "canonical_unordered_pairs",
    "decode_direct_arm_term",
    "decode_pair",
    "decode_pair_training_selective_autograd",
    "target_signed_paired_logit",
    "pair_loss",
    "learning_rate",
    "ordered_training_pool",
    "update_episodes",
    "episode_schedule_key",
    "target_first",
    "seed_everything",
    "capture_rng_state",
    "restore_rng_state",
    "state_dict_sha256",
    "registered_pair_comparator_sha256",
    "bind_runtime_lineage",
    "DeviceMaskRuntimeAdapter",
    "move_episode_to_device",
    "move_locks_to_device",
    "move_query_bundle_to_device",
    "make_resume_state",
    "validate_resume_state",
    "train_updates",
    "assert_target_free_payload",
    "serialize_arm_evidence",
]
