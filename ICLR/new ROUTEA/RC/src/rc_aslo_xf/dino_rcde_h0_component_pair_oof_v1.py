"""Anonymous two-member component-unary scoring primitives.

The core consumes exactly two candidate bundles.  Each bundle carries one
candidate token field and its two sealed directional P-locks.  Directional
component logits are delegated unchanged to the qualified C128 unary core;
the member score always uses the fixed two-direction denominator.

This module is an in-memory scoring primitive only.  It contains no data
loading, reduction, training, or execution orchestration.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    QueryTokenFieldV1,
    token_tensor_sha256,
)
from .dino_rcde_h0_c128_tournament_v1 import decode_component_unary
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    LOCK_H0,
    SealedDirectionPLockV1,
    p_tensor_sha256,
)
from .dino_rcde_v1_2_resource_core import CandidateEvidence, DINO_RCDE_V1_2


SCHEMA_VERSION = "rc_dino_rcde_h0_component_pair_oof_core_v1_20260824"


class ComponentPairOOFError(ValueError):
    """The anonymous pair or one of its sealed bindings is malformed."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ComponentPairOOFError(message)


def _require_sha256(value: object, *, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA256",
    )
    return value


@dataclass(frozen=True)
class ComponentRootReceiptV1:
    """One decoded candidate-specific reference component."""

    root_ordinal: int
    reference_component_mask_sha256: str

    def __post_init__(self) -> None:
        _require(
            isinstance(self.root_ordinal, int)
            and not isinstance(self.root_ordinal, bool)
            and self.root_ordinal >= 0,
            "component root ordinal drift",
        )
        _require_sha256(
            self.reference_component_mask_sha256,
            name="reference component mask",
        )


@dataclass(frozen=True)
class DirectionalComponentRawLogitV1:
    """Raw unary result and provenance for one fixed direction."""

    direction: str
    raw_logit: torch.Tensor
    forward_count: int
    root_receipts: tuple[ComponentRootReceiptV1, ...]
    structural_h0: bool
    candidate_specific_spatial_component_consumed: bool
    query_union_mask_p_sha256: str
    p_lock_record_sha256: str

    def __post_init__(self) -> None:
        _require(self.direction in FIXED_DIRECTIONS, "direction drift")
        _require(
            isinstance(self.raw_logit, torch.Tensor)
            and self.raw_logit.ndim == 0
            and self.raw_logit.is_floating_point()
            and bool(torch.isfinite(self.raw_logit)),
            "directional raw logit drift",
        )
        receipts = tuple(self.root_receipts)
        _require(
            isinstance(self.forward_count, int)
            and not isinstance(self.forward_count, bool)
            and self.forward_count == len(receipts),
            "directional forward count drift",
        )
        _require(
            tuple(item.root_ordinal for item in receipts)
            == tuple(sorted({item.root_ordinal for item in receipts})),
            "component root receipt order drift",
        )
        _require(isinstance(self.structural_h0, bool), "structural-H0 flag drift")
        _require(
            isinstance(self.candidate_specific_spatial_component_consumed, bool),
            "component-consumption flag drift",
        )
        if self.structural_h0:
            _require(
                self.forward_count == 0
                and not receipts
                and not self.candidate_specific_spatial_component_consumed
                and torch.equal(self.raw_logit, torch.zeros_like(self.raw_logit)),
                "structural-H0 is not exact zero",
            )
        else:
            _require(
                self.forward_count > 0
                and self.candidate_specific_spatial_component_consumed,
                "READY direction did not consume a component",
            )
        _require_sha256(
            self.query_union_mask_p_sha256, name="query union mask receipt"
        )
        _require_sha256(self.p_lock_record_sha256, name="P-lock record")
        object.__setattr__(self, "root_receipts", receipts)


@dataclass(frozen=True)
class AnonymousComponentMemberScoreV1:
    """Two directional raw logits and their fixed arithmetic mean."""

    candidate_key: str
    directional_raw_logits: Mapping[str, DirectionalComponentRawLogitV1]
    mean_score: torch.Tensor
    forward_count: int

    def __post_init__(self) -> None:
        _require(
            isinstance(self.candidate_key, str) and bool(self.candidate_key),
            "anonymous member key is empty",
        )
        directions = dict(self.directional_raw_logits)
        _require(
            set(directions) == set(FIXED_DIRECTIONS),
            "anonymous member must contain exactly two directions",
        )
        directions = {direction: directions[direction] for direction in FIXED_DIRECTIONS}
        _require(
            all(item.direction == direction for direction, item in directions.items()),
            "directional result key drift",
        )
        first, second = (directions[direction].raw_logit for direction in FIXED_DIRECTIONS)
        _require(
            first.device == second.device and first.dtype == second.dtype,
            "directional raw-logit device or dtype drift",
        )
        expected_mean = (first + second) * 0.5
        _require(
            isinstance(self.mean_score, torch.Tensor)
            and self.mean_score.ndim == 0
            and self.mean_score.device == first.device
            and self.mean_score.dtype == first.dtype
            and bool(torch.isfinite(self.mean_score))
            and torch.equal(self.mean_score, expected_mean),
            "fixed two-direction mean drift",
        )
        _require(
            isinstance(self.forward_count, int)
            and not isinstance(self.forward_count, bool)
            and self.forward_count
            == sum(item.forward_count for item in directions.values()),
            "member forward count drift",
        )
        object.__setattr__(
            self, "directional_raw_logits", MappingProxyType(directions)
        )

    @property
    def raw_logits_by_direction(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(
            {
                direction: item.raw_logit
                for direction, item in self.directional_raw_logits.items()
            }
        )

    @property
    def root_receipts_by_direction(
        self,
    ) -> Mapping[str, tuple[ComponentRootReceiptV1, ...]]:
        return MappingProxyType(
            {
                direction: item.root_receipts
                for direction, item in self.directional_raw_logits.items()
            }
        )


@dataclass(frozen=True)
class ComponentPairOOFV1(Mapping[str, AnonymousComponentMemberScoreV1]):
    """Key-addressable output for exactly two anonymous members."""

    members: Mapping[str, AnonymousComponentMemberScoreV1]

    def __post_init__(self) -> None:
        members = dict(self.members)
        _require(len(members) == 2, "component pair must contain exactly two members")
        _require(
            all(key == item.candidate_key for key, item in members.items()),
            "component pair key binding drift",
        )
        object.__setattr__(self, "members", MappingProxyType(members))

    def __getitem__(self, key: str) -> AnonymousComponentMemberScoreV1:
        return self.members[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    def aligned_by_candidate_key(
        self,
    ) -> Mapping[str, AnonymousComponentMemberScoreV1]:
        return MappingProxyType(
            {key: self.members[key] for key in sorted(self.members)}
        )


CandidateBundleV1 = tuple[
    CandidateReferenceFieldV1, Mapping[str, SealedDirectionPLockV1]
]


def _preflight_candidate_bundles(
    model: DINO_RCDE_V1_2,
    query: QueryTokenFieldV1,
    candidates: Mapping[str, CandidateBundleV1],
) -> tuple[
    tuple[
        str,
        CandidateReferenceFieldV1,
        Mapping[str, SealedDirectionPLockV1],
    ],
    tuple[
        str,
        CandidateReferenceFieldV1,
        Mapping[str, SealedDirectionPLockV1],
    ],
]:
    _require(isinstance(model, DINO_RCDE_V1_2), "component pair model drift")
    _require(isinstance(query, QueryTokenFieldV1), "component pair query drift")
    _require(isinstance(candidates, Mapping), "component pair input must be a mapping")
    pairs = tuple(candidates.items())
    _require(len(pairs) == 2, "component pair input must contain exactly two members")
    _require(
        token_tensor_sha256(query.layers) == query.tokens_sha256,
        "query token tensor provenance drift",
    )
    prepared = []
    physical_rows = []
    for candidate_key, bundle in pairs:
        _require(
            isinstance(candidate_key, str)
            and bool(candidate_key)
            and isinstance(bundle, tuple)
            and len(bundle) == 2,
            "component pair bundle drift",
        )
        candidate, raw_locks = bundle
        _require(
            isinstance(candidate, CandidateReferenceFieldV1)
            and candidate.candidate_key == candidate_key,
            "component pair candidate key drift",
        )
        _require(
            token_tensor_sha256(candidate.layers) == candidate.tokens_sha256,
            "candidate token tensor provenance drift",
        )
        _require(
            isinstance(raw_locks, Mapping),
            "candidate direction locks must be a mapping",
        )
        locks = dict(raw_locks)
        _require(
            set(locks) == set(FIXED_DIRECTIONS),
            "candidate must contain exactly two direction locks",
        )
        ordered_locks = {direction: locks[direction] for direction in FIXED_DIRECTIONS}
        for direction, lock in ordered_locks.items():
            _require(
                isinstance(lock, SealedDirectionPLockV1)
                and lock.direction == direction
                and lock.candidate_key == candidate_key
                and lock.candidate_physical_row == candidate.physical_gallery_row
                and lock.candidate_reference_source_sha256
                == candidate.source_image_sha256
                and lock.query_source_image_sha256 == query.source_image_sha256
                and lock.query_grid_shape == query.grid_shape
                and lock.reference_grid_shape == candidate.grid_shape
                and lock.query_geometry_sha256 == query.geometry_record_sha256
                and lock.reference_geometry_sha256
                == candidate.geometry_record_sha256,
                "candidate/query direction-lock provenance drift",
            )
        physical_rows.append(candidate.physical_gallery_row)
        prepared.append((candidate_key, candidate, MappingProxyType(ordered_locks)))
    _require(
        len(set(physical_rows)) == 2,
        "component pair members alias one physical candidate",
    )
    return prepared[0], prepared[1]


def score_component_pair_oof(
    model: DINO_RCDE_V1_2,
    query: QueryTokenFieldV1,
    candidates: Mapping[str, CandidateBundleV1],
    *,
    streaming_chunk_size: int | None = 64,
    decoder: Callable[..., CandidateEvidence] | None = None,
) -> ComponentPairOOFV1:
    """Score exactly two anonymous members without position-dependent logic."""

    prepared = _preflight_candidate_bundles(model, query, candidates)
    members: dict[str, AnonymousComponentMemberScoreV1] = {}
    for candidate_key, candidate, locks in prepared:
        directional: dict[str, DirectionalComponentRawLogitV1] = {}
        for direction in FIXED_DIRECTIONS:
            lock = locks[direction]
            unary = decode_component_unary(
                model,
                query,
                candidate,
                lock,
                streaming_chunk_size=streaming_chunk_size,
                decoder=decoder,
            )
            receipts = tuple(
                ComponentRootReceiptV1(root_ordinal, mask_sha256)
                for root_ordinal, mask_sha256 in zip(
                    unary.decoded_root_ordinals,
                    unary.reference_component_mask_sha256s,
                    strict=True,
                )
            )
            directional[direction] = DirectionalComponentRawLogitV1(
                direction=direction,
                raw_logit=unary.score,
                forward_count=unary.model_forward_count,
                root_receipts=receipts,
                structural_h0=lock.status == LOCK_H0,
                candidate_specific_spatial_component_consumed=(
                    unary.candidate_specific_spatial_component_consumed
                ),
                query_union_mask_p_sha256=p_tensor_sha256(unary.query_union_mask),
                p_lock_record_sha256=lock.p_lock_record_sha256,
            )
        first, second = (
            directional[direction].raw_logit for direction in FIXED_DIRECTIONS
        )
        members[candidate_key] = AnonymousComponentMemberScoreV1(
            candidate_key=candidate_key,
            directional_raw_logits=directional,
            mean_score=(first + second) * 0.5,
            forward_count=sum(item.forward_count for item in directional.values()),
        )
    return ComponentPairOOFV1(members)


def align_component_pair_scores_by_candidate_key(
    value: ComponentPairOOFV1
    | Mapping[str, AnonymousComponentMemberScoreV1],
) -> Mapping[str, AnonymousComponentMemberScoreV1]:
    """Return the canonical key view used for candidate-reorder comparisons."""

    members = dict(value.members if isinstance(value, ComponentPairOOFV1) else value)
    _require(len(members) == 2, "aligned component pair must contain two members")
    _require(
        all(
            isinstance(item, AnonymousComponentMemberScoreV1)
            and key == item.candidate_key
            for key, item in members.items()
        ),
        "aligned component pair key binding drift",
    )
    return MappingProxyType({key: members[key] for key in sorted(members)})


__all__ = [
    "SCHEMA_VERSION",
    "ComponentPairOOFError",
    "ComponentRootReceiptV1",
    "DirectionalComponentRawLogitV1",
    "AnonymousComponentMemberScoreV1",
    "ComponentPairOOFV1",
    "CandidateBundleV1",
    "score_component_pair_oof",
    "align_component_pair_scores_by_candidate_key",
]
