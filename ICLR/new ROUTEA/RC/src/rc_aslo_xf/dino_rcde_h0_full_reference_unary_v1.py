"""Candidate-unary full-reference H0 path for the existing RCDE model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    QueryTokenFieldV1,
    token_tensor_sha256,
)
from .dino_rcde_sr0_mt_semantic_h0_v1 import candidate_unary_evidence
from .dino_rcde_v1_2_resource_core import CandidateEvidence, DINO_RCDE_V1_2


SCHEMA_VERSION = "rc_dino_rcde_h0_full_reference_unary_core_v1_20260821"


class FullReferenceUnaryError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise FullReferenceUnaryError(message)


@dataclass(frozen=True)
class FullReferenceUnaryV1:
    score: torch.Tensor
    evidence: CandidateEvidence
    query_mask: torch.Tensor
    reference_mask: torch.Tensor


def decode_full_reference_unary(
    model: DINO_RCDE_V1_2,
    query: QueryTokenFieldV1,
    reference: CandidateReferenceFieldV1,
    query_union_mask: torch.Tensor,
    *,
    structural_ready: bool,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> FullReferenceUnaryV1:
    """Decode one candidate alone on a sealed query union and full reference."""

    require(isinstance(model, DINO_RCDE_V1_2), "full-reference unary model drift")
    query_mask = torch.as_tensor(
        query_union_mask, dtype=torch.bool, device=query.layers.device
    ).flatten()
    reference_mask = torch.as_tensor(
        reference.valid_patch_mask, dtype=torch.bool, device=reference.layers.device
    ).flatten()
    valid_query = torch.as_tensor(
        query.valid_patch_mask, dtype=torch.bool, device=query.layers.device
    ).flatten()
    require(
        query_mask.shape == valid_query.shape
        and bool(query_mask.any()) == bool(structural_ready)
        and not bool((query_mask & ~valid_query).any()),
        "sealed query-union mask drift",
    )
    require(
        reference_mask.shape == reference.valid_patch_mask.shape
        and bool(reference_mask.any())
        and torch.equal(reference_mask.cpu(), reference.valid_patch_mask.cpu()),
        "reference is not complete native valid support",
    )
    require(
        token_tensor_sha256(query.layers) == query.tokens_sha256
        and token_tensor_sha256(reference.layers) == reference.tokens_sha256,
        "token provenance drift",
    )
    if not structural_ready:
        # A structural-H0 row is not decoded against a reference.
        raise FullReferenceUnaryError(
            "structural-H0 has no full-reference unary forward; use exact zero receipt"
        )
    evidence = model.decode_candidate(
        query.layers,
        reference.layers,
        query_mask.reshape(query.grid_shape),
        reference_mask.reshape(reference.grid_shape),
        query.grid_shape,
        reference.grid_shape,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )
    score = candidate_unary_evidence(
        model.signed_head,
        evidence.relational,
        query_mask,
        structural_ready=True,
    )
    require(score.ndim == 0 and bool(torch.isfinite(score)), "unary score drift")
    return FullReferenceUnaryV1(score, evidence, query_mask, reference_mask)


def exact_zero_unary(model: DINO_RCDE_V1_2) -> torch.Tensor:
    """The bias-free semantic-H0 value without constructing fake tokens."""

    require(isinstance(model, DINO_RCDE_V1_2), "zero-unary model drift")
    return model.signed_head.weight.square().sum() * 0.0


def matched_absolute_h0_loss(
    *,
    pair_logit: torch.Tensor,
    positive_scores: tuple[torch.Tensor, torch.Tensor],
    donor_scores: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Frozen equal-weight pair + positive-unary + donor-unary objective."""

    pair = torch.as_tensor(pair_logit)
    values = (*positive_scores, *donor_scores)
    require(
        pair.ndim == 0
        and pair.is_floating_point()
        and bool(torch.isfinite(pair))
        and all(
            item.ndim == 0
            and item.device == pair.device
            and item.dtype == pair.dtype
            and bool(torch.isfinite(item))
            for item in values
        ),
        "absolute-H0 loss population drift",
    )
    relative = torch.nn.functional.softplus(-pair)
    positive = 0.5 * sum(torch.nn.functional.softplus(-item) for item in positive_scores)
    negative = 0.5 * sum(torch.nn.functional.softplus(item) for item in donor_scores)
    return relative + positive + negative


def unmatched_relative_loss(pair_logit: torch.Tensor) -> torch.Tensor:
    pair = torch.as_tensor(pair_logit)
    require(pair.ndim == 0 and bool(torch.isfinite(pair)), "relative-only fallback drift")
    return torch.nn.functional.softplus(-pair)


__all__ = [
    "SCHEMA_VERSION",
    "FullReferenceUnaryError",
    "FullReferenceUnaryV1",
    "decode_full_reference_unary",
    "exact_zero_unary",
    "matched_absolute_h0_loss",
    "unmatched_relative_loss",
]
