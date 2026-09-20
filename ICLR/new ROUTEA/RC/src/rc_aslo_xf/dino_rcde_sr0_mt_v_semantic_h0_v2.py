"""Candidate-unary semantic-H0 adapter for the existing DINO-RCDE V graph."""
from __future__ import annotations

import torch

from .dino_rcde_v1_2_resource_core import (
    CandidateEvidence,
    DINO_RCDE_V1_2,
    StreamedCandidateEvidence,
)
from .dino_rcde_sr0_mt_semantic_h0_v1 import candidate_unary_evidence


def unary_from_candidate_evidence(
    model: DINO_RCDE_V1_2,
    evidence: CandidateEvidence | StreamedCandidateEvidence,
    query_mask: torch.Tensor,
    *,
    structural_ready: bool,
) -> torch.Tensor:
    if not isinstance(model, DINO_RCDE_V1_2):
        raise TypeError("semantic-H0 V2 requires DINO_RCDE_V1_2")
    if not isinstance(evidence, (CandidateEvidence, StreamedCandidateEvidence)):
        raise TypeError("semantic-H0 V2 requires candidate-specific evidence")
    return candidate_unary_evidence(
        model.signed_head,
        evidence.relational,
        query_mask,
        structural_ready=structural_ready,
    )


__all__ = ["unary_from_candidate_evidence"]

