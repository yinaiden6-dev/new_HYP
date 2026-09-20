"""Target-free C128 tournament engineering primitives.

The Stage-1 spatial arm consumes the candidate-specific reference components
of each sealed P-lock.  The U4 full-reference unary is intentionally not used
as a substitute for these components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    QueryTokenFieldV1,
    token_tensor_sha256,
)
from .dino_rcde_h0_full_reference_unary_v1 import exact_zero_unary
from .dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256
from .dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (
    HeadSelectionSpecV1,
    select_natural_v2_head_records,
)
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    LOCK_H0,
    LOCK_READY,
    ROOT_READY,
    SealedDirectionPLockV1,
    p_tensor_sha256,
)
from .dino_rcde_v1_2_resource_core import CandidateEvidence, DINO_RCDE_V1_2


SCHEMA_VERSION = "rc_dino_rcde_h0_c128_tournament_core_v1_20260823"
ORDER_NAMESPACE = "H0_C128_TOURNAMENT_PREJOIN_HASH_ORDER_V1"


class C128TournamentError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise C128TournamentError(message)


@dataclass(frozen=True)
class CandidateOuterRecordsV1:
    candidate_key: str
    candidate_physical_row: int
    candidate_reference_source_sha256: str
    direction_records: Mapping[str, Mapping[str, Any]]
    record_sha256s: tuple[str, str]
    order_sha256: str

    def __post_init__(self) -> None:
        require(set(self.direction_records) == set(FIXED_DIRECTIONS), "two-direction record drift")
        require(tuple(self.direction_records) == FIXED_DIRECTIONS, "direction order drift")
        require(tuple(self.direction_records[d]["record_sha256"] for d in FIXED_DIRECTIONS) == self.record_sha256s, "direction SHA drift")


def group_selected_records(
    records: Sequence[Mapping[str, Any]], *, query_id: str, expected_candidate_count: int
) -> tuple[CandidateOuterRecordsV1, ...]:
    require(len(records) == expected_candidate_count * 2, "selected record population drift")
    buckets: dict[int, dict[str, Mapping[str, Any]]] = {}
    addresses: dict[int, tuple[str, int, str]] = {}
    for record in records:
        query = record["query"]
        candidate = record["candidate"]
        require(query["query_id"] == query_id, "selected records mix queries")
        position = int(candidate["candidate_position"])
        direction = str(record["direction"])
        require(0 <= position < expected_candidate_count and direction in FIXED_DIRECTIONS, "candidate position/direction drift")
        address = (str(candidate["candidate_key"]), int(candidate["candidate_physical_row"]), str(candidate["candidate_reference_source_sha256"]))
        require(addresses.setdefault(position, address) == address, "candidate address differs across directions")
        bucket = buckets.setdefault(position, {})
        require(direction not in bucket, "duplicate candidate direction")
        bucket[direction] = record
    require(set(buckets) == set(range(expected_candidate_count)), "candidate-axis position drift")
    rows = []
    for position in range(expected_candidate_count):
        require(set(buckets[position]) == set(FIXED_DIRECTIONS), "candidate direction incomplete")
        key, physical, source = addresses[position]
        directions = {direction: buckets[position][direction] for direction in FIXED_DIRECTIONS}
        shas = tuple(str(directions[d]["record_sha256"]) for d in FIXED_DIRECTIONS)
        order = canonical_sha256({"namespace": ORDER_NAMESPACE, "query_id": query_id, "candidate_key": key, "candidate_physical_row": physical, "direction_record_sha256s": list(shas)})
        rows.append(CandidateOuterRecordsV1(key, physical, source, directions, shas, order))
    require(len({row.candidate_key for row in rows}) == expected_candidate_count and len({row.candidate_physical_row for row in rows}) == expected_candidate_count, "candidate axis not one-to-one")
    return tuple(rows)


def select_outer_c128_records(
    artifact_records: Sequence[Mapping[str, Any]], *, query_id: str, spec: HeadSelectionSpecV1
) -> tuple[CandidateOuterRecordsV1, ...]:
    selected = select_natural_v2_head_records(artifact_records, spec=spec, expected_candidate_count=128)
    return group_selected_records(selected, query_id=query_id, expected_candidate_count=128)


def hash_order_fixture(
    records: Sequence[CandidateOuterRecordsV1], *, count: int
) -> tuple[CandidateOuterRecordsV1, ...]:
    require(0 < count <= len(records), "fixture count drift")
    return tuple(sorted(records, key=lambda row: (row.order_sha256, row.candidate_key))[:count])


@dataclass(frozen=True)
class ComponentUnaryV1:
    score: torch.Tensor
    query_union_mask: torch.Tensor
    patch_evidence: torch.Tensor
    decoded_root_ordinals: tuple[int, ...]
    reference_component_mask_sha256s: tuple[str, ...]
    model_forward_count: int
    candidate_specific_spatial_component_consumed: bool


def decode_component_unary(
    model: DINO_RCDE_V1_2,
    query: QueryTokenFieldV1,
    candidate: CandidateReferenceFieldV1,
    lock: SealedDirectionPLockV1,
    *,
    streaming_chunk_size: int | None = 64,
    decoder: Callable[..., CandidateEvidence] | None = None,
) -> ComponentUnaryV1:
    require(isinstance(model, DINO_RCDE_V1_2), "component unary model drift")
    require(
        candidate.candidate_key == lock.candidate_key
        and candidate.physical_gallery_row == lock.candidate_physical_row
        and candidate.source_image_sha256 == lock.candidate_reference_source_sha256
        and query.source_image_sha256 == lock.query_source_image_sha256
        and query.grid_shape == lock.query_grid_shape
        and candidate.grid_shape == lock.reference_grid_shape,
        "candidate/query P-lock provenance drift",
    )
    require(token_tensor_sha256(query.layers) == query.tokens_sha256 and token_tensor_sha256(candidate.layers) == candidate.tokens_sha256, "token tensor provenance drift")
    union_cpu = lock.query_union_mask.detach().cpu().to(torch.bool).flatten()
    if lock.status == LOCK_H0:
        zero = exact_zero_unary(model)
        return ComponentUnaryV1(zero, union_cpu, torch.zeros_like(union_cpu, dtype=zero.dtype, device=zero.device), (), (), 0, False)
    require(lock.status == LOCK_READY and bool(union_cpu.any()), "component unary lock state drift")
    decode = model.decode_candidate if decoder is None else decoder
    total = query.layers.new_zeros(query.valid_patch_mask.numel())
    coverage = torch.zeros_like(query.valid_patch_mask, dtype=torch.int64, device=total.device)
    decoded: list[int] = []
    reference_hashes: list[str] = []
    for root in lock.selected_roots:
        require(root.binding_status == ROOT_READY and bool(root.reference_mask.any()), "selected component is not candidate-specific READY")
        qmask = root.query_mask.to(device=query.layers.device, dtype=torch.bool)
        rmask = root.reference_mask.to(device=candidate.layers.device, dtype=torch.bool)
        evidence = decode(
            query.layers,
            candidate.layers,
            qmask.reshape(query.grid_shape),
            rmask.reshape(candidate.grid_shape),
            query.grid_shape,
            candidate.grid_shape,
            streaming_chunk_size=streaming_chunk_size,
        )
        require(evidence.relational.shape[0] == qmask.numel(), "component relational shape drift")
        signed = model.signed_head(evidence.relational).squeeze(-1)
        require(signed.shape == qmask.shape and bool(torch.isfinite(signed).all()), "component signed evidence drift")
        total = total + signed * qmask.to(signed.dtype)
        coverage = coverage + qmask.to(torch.int64)
        decoded.append(root.root_ordinal)
        reference_hashes.append(root.reference_mask_p_sha256)
    require(bool(decoded), "READY lock decoded no candidate-specific components")
    reciprocal = torch.zeros_like(total)
    covered = coverage > 0
    reciprocal[covered] = 1.0 / coverage[covered].to(total.dtype)
    union = union_cpu.to(device=total.device)
    patch = total * reciprocal * union.to(total.dtype)
    score = patch.sum() / union.sum().to(dtype=patch.dtype)
    require(bool(torch.isfinite(score)) and all(p_tensor_sha256(root.reference_mask) == digest for root, digest in zip(lock.selected_roots, reference_hashes, strict=True)), "component score/hash drift")
    return ComponentUnaryV1(score, union_cpu, patch, tuple(decoded), tuple(reference_hashes), len(decoded), True)


def align_scores_by_candidate_key(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output = {str(row["candidate_key"]): row for row in rows}
    require(len(output) == len(rows), "candidate score key collision")
    return output


__all__ = [
    "SCHEMA_VERSION",
    "ORDER_NAMESPACE",
    "C128TournamentError",
    "CandidateOuterRecordsV1",
    "ComponentUnaryV1",
    "group_selected_records",
    "select_outer_c128_records",
    "hash_order_fixture",
    "decode_component_unary",
    "align_scores_by_candidate_key",
]
