"""Target-free full-C128 component-unary Stage-A primitives.

This module is deliberately smaller than a retrieval reducer.  It validates
one complete natural C128 address axis and emits the two raw directional
component logits for one candidate at a time.  It does not average directions,
compare candidates, read a role, or compute a scientific metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    QueryTokenFieldV1,
    token_tensor_sha256,
)
from .dino_rcde_h0_c128_tournament_v1 import (
    CandidateOuterRecordsV1,
    decode_component_unary,
)
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    LOCK_H0,
    LOCK_READY,
    SealedDirectionPLockV1,
    candidate_axis_sha256,
    p_tensor_sha256,
)
from .dino_rcde_v1_2_resource_core import CandidateEvidence, DINO_RCDE_V1_2


SCHEMA_VERSION = "rc_dino_rcde_h0_c128_target_free_prejoin_core_v1_20260824"
CANDIDATE_COUNT = 128


class C128TargetFreePrejoinError(ValueError):
    """A full-C128 address or candidate-specific P-lock is malformed."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise C128TargetFreePrejoinError(message)


def require_sha256(value: object, *, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA256",
    )
    return value


@dataclass(frozen=True)
class FullC128AxisV1:
    """The exact, canonical natural-C128 address axis for one query."""

    records: tuple[CandidateOuterRecordsV1, ...]
    physical_rows: tuple[int, ...]
    candidate_axis_sha256: str

    def __post_init__(self) -> None:
        records = tuple(self.records)
        rows = tuple(int(value) for value in self.physical_rows)
        require(len(records) == len(rows) == CANDIDATE_COUNT, "C128 population drift")
        require(
            tuple(item.candidate_physical_row for item in records) == rows,
            "record/schedule candidate order drift",
        )
        require(rows == tuple(sorted(rows)) and len(set(rows)) == CANDIDATE_COUNT, "candidate rows are not canonical one-to-one C128")
        require(
            len({item.candidate_key for item in records}) == CANDIDATE_COUNT,
            "candidate key address is not one-to-one",
        )
        require(
            candidate_axis_sha256(rows) == self.candidate_axis_sha256,
            "candidate-axis SHA drift",
        )
        require_sha256(self.candidate_axis_sha256, name="candidate axis")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "physical_rows", rows)


def validate_full_c128_axis(
    records: Sequence[CandidateOuterRecordsV1],
    *,
    expected_physical_rows: Sequence[int],
    expected_candidate_axis_sha256: str,
) -> FullC128AxisV1:
    """Fail before model forward unless the complete frozen C128 axis closes."""

    return FullC128AxisV1(
        tuple(records),
        tuple(int(value) for value in expected_physical_rows),
        expected_candidate_axis_sha256,
    )


@dataclass(frozen=True)
class DirectionalRawLogitV1:
    direction: str
    raw_logit: torch.Tensor
    lock_status: str
    p_lock_record_sha256: str
    query_union_mask_p_sha256: str
    decoded_root_ordinals: tuple[int, ...]
    reference_component_mask_sha256s: tuple[str, ...]
    model_forward_count: int
    candidate_specific_spatial_component_consumed: bool

    def __post_init__(self) -> None:
        require(self.direction in FIXED_DIRECTIONS, "direction drift")
        require(
            isinstance(self.raw_logit, torch.Tensor)
            and self.raw_logit.ndim == 0
            and self.raw_logit.is_floating_point()
            and bool(torch.isfinite(self.raw_logit)),
            "raw directional logit drift",
        )
        require_sha256(self.p_lock_record_sha256, name="P-lock record")
        require_sha256(self.query_union_mask_p_sha256, name="query union mask")
        roots = tuple(self.decoded_root_ordinals)
        masks = tuple(self.reference_component_mask_sha256s)
        require(
            roots == tuple(sorted(set(roots)))
            and len(roots) == len(masks) == self.model_forward_count,
            "component receipt population/order drift",
        )
        for digest in masks:
            require_sha256(digest, name="reference component mask")
        if self.lock_status == LOCK_H0:
            require(
                not roots
                and self.model_forward_count == 0
                and not self.candidate_specific_spatial_component_consumed
                and torch.equal(self.raw_logit, torch.zeros_like(self.raw_logit)),
                "structural H0 is not exact zero",
            )
        else:
            require(
                self.lock_status == LOCK_READY
                and self.model_forward_count > 0
                and self.candidate_specific_spatial_component_consumed,
                "READY direction did not consume candidate-specific components",
            )
        object.__setattr__(self, "decoded_root_ordinals", roots)
        object.__setattr__(self, "reference_component_mask_sha256s", masks)


@dataclass(frozen=True)
class CandidateDirectionalRawLogitsV1:
    candidate_key: str
    candidate_physical_row: int
    candidate_reference_source_sha256: str
    directions: Mapping[str, DirectionalRawLogitV1]
    model_forward_count: int

    def __post_init__(self) -> None:
        require(isinstance(self.candidate_key, str) and self.candidate_key, "candidate key absent")
        require(isinstance(self.candidate_physical_row, int) and self.candidate_physical_row >= 0, "candidate row drift")
        require_sha256(self.candidate_reference_source_sha256, name="candidate source")
        directions = dict(self.directions)
        require(set(directions) == set(FIXED_DIRECTIONS), "two-direction population drift")
        directions = {direction: directions[direction] for direction in FIXED_DIRECTIONS}
        require(
            all(item.direction == direction for direction, item in directions.items())
            and self.model_forward_count
            == sum(item.model_forward_count for item in directions.values()),
            "direction binding/forward count drift",
        )
        object.__setattr__(self, "directions", MappingProxyType(directions))


def score_candidate_directional_raw(
    model: DINO_RCDE_V1_2,
    query: QueryTokenFieldV1,
    candidate: CandidateReferenceFieldV1,
    locks: Mapping[str, SealedDirectionPLockV1],
    *,
    streaming_chunk_size: int | None = 64,
    decoder: Callable[..., CandidateEvidence] | None = None,
) -> CandidateDirectionalRawLogitsV1:
    """Emit raw A2B/B2A logits without a candidate or direction reducer."""

    require(isinstance(model, DINO_RCDE_V1_2), "model drift")
    require(isinstance(query, QueryTokenFieldV1), "query field drift")
    require(isinstance(candidate, CandidateReferenceFieldV1), "candidate field drift")
    require(token_tensor_sha256(query.layers) == query.tokens_sha256, "query token provenance drift")
    require(token_tensor_sha256(candidate.layers) == candidate.tokens_sha256, "candidate token provenance drift")
    raw_locks = dict(locks)
    require(set(raw_locks) == set(FIXED_DIRECTIONS), "candidate direction locks drift")
    ordered_locks = {direction: raw_locks[direction] for direction in FIXED_DIRECTIONS}
    for direction, lock in ordered_locks.items():
        require(
            isinstance(lock, SealedDirectionPLockV1)
            and lock.direction == direction
            and lock.candidate_key == candidate.candidate_key
            and lock.candidate_physical_row == candidate.physical_gallery_row
            and lock.candidate_reference_source_sha256 == candidate.source_image_sha256
            and lock.query_source_image_sha256 == query.source_image_sha256
            and lock.query_grid_shape == query.grid_shape
            and lock.reference_grid_shape == candidate.grid_shape
            and lock.query_geometry_sha256 == query.geometry_record_sha256
            and lock.reference_geometry_sha256 == candidate.geometry_record_sha256,
            "candidate/query P-lock provenance drift",
        )
    output: dict[str, DirectionalRawLogitV1] = {}
    for direction in FIXED_DIRECTIONS:
        lock = ordered_locks[direction]
        unary = decode_component_unary(
            model,
            query,
            candidate,
            lock,
            streaming_chunk_size=streaming_chunk_size,
            decoder=decoder,
        )
        output[direction] = DirectionalRawLogitV1(
            direction=direction,
            raw_logit=unary.score,
            lock_status=lock.status,
            p_lock_record_sha256=lock.p_lock_record_sha256,
            query_union_mask_p_sha256=p_tensor_sha256(unary.query_union_mask),
            decoded_root_ordinals=unary.decoded_root_ordinals,
            reference_component_mask_sha256s=unary.reference_component_mask_sha256s,
            model_forward_count=unary.model_forward_count,
            candidate_specific_spatial_component_consumed=unary.candidate_specific_spatial_component_consumed,
        )
    return CandidateDirectionalRawLogitsV1(
        candidate_key=candidate.candidate_key,
        candidate_physical_row=candidate.physical_gallery_row,
        candidate_reference_source_sha256=candidate.source_image_sha256,
        directions=output,
        model_forward_count=sum(item.model_forward_count for item in output.values()),
    )


__all__ = [
    "SCHEMA_VERSION",
    "CANDIDATE_COUNT",
    "C128TargetFreePrejoinError",
    "FullC128AxisV1",
    "DirectionalRawLogitV1",
    "CandidateDirectionalRawLogitsV1",
    "validate_full_c128_axis",
    "score_candidate_directional_raw",
]
