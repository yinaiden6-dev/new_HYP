"""Raw-preserving domain alignment for identity-disjoint difficult retrieval."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .domain_alignment import (
    DomainAdapterOutput,
    DomainRankingLoss,
    deployment_maxsim_ranking_loss,
)


@dataclass
class SafeMarginTerms:
    preservation: torch.Tensor
    correction: torch.Tensor
    raw_correct_count: int
    raw_error_count: int


@dataclass
class DomainSafeTrainingLoss:
    total: torch.Tensor
    ranking: DomainRankingLoss
    raw_ranking: DomainRankingLoss
    margins: SafeMarginTerms
    drift: torch.Tensor
    adapted: DomainAdapterOutput


def safe_margin_terms(
    raw_scores: torch.Tensor,
    adapted_scores: torch.Tensor,
    *,
    normalization_count: int,
    positive_index: int = 0,
    preserve_cap: float = 0.03,
    error_margin: float = 0.02,
) -> SafeMarginTerms:
    """Preserve raw-correct margins and repair raw-misordered competitors."""
    if (
        raw_scores.ndim != 1
        or adapted_scores.shape != raw_scores.shape
        or raw_scores.numel() < 2
        or normalization_count < 1
        or not 0 <= positive_index < raw_scores.numel()
        or preserve_cap <= 0.0
        or error_margin <= 0.0
    ):
        raise ValueError("invalid Domain-Safe margin inputs")
    competitor_mask = torch.ones(
        raw_scores.numel(),
        dtype=torch.bool,
        device=raw_scores.device,
    )
    competitor_mask[int(positive_index)] = False
    raw_margin = (
        raw_scores[int(positive_index)].detach()
        - raw_scores[competitor_mask].detach()
    ) / float(normalization_count)
    adapted_margin = (
        adapted_scores[int(positive_index)]
        - adapted_scores[competitor_mask]
    ) / float(normalization_count)
    raw_correct = raw_margin > 0.0
    raw_error = ~raw_correct
    zero = adapted_scores.sum() * 0.0
    preservation = (
        F.relu(
            raw_margin[raw_correct].clamp(max=float(preserve_cap))
            - adapted_margin[raw_correct]
        ).mean()
        if bool(raw_correct.any())
        else zero
    )
    correction = (
        F.relu(float(error_margin) - adapted_margin[raw_error]).mean()
        if bool(raw_error.any())
        else zero
    )
    if not bool(torch.isfinite(preservation + correction)):
        raise AssertionError("Domain-Safe margin loss is non-finite")
    return SafeMarginTerms(
        preservation=preservation,
        correction=correction,
        raw_correct_count=int(raw_correct.sum().item()),
        raw_error_count=int(raw_error.sum().item()),
    )


def domain_safe_training_loss(
    adapter: torch.nn.Module,
    raw_image_tokens: torch.Tensor,
    template_tokens: torch.Tensor,
    references: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    temperature: float = 0.05,
    preserve_cap: float = 0.03,
    error_margin: float = 0.02,
    preservation_weight: float = 10.0,
    correction_weight: float = 2.0,
    drift_weight: float = 0.05,
) -> DomainSafeTrainingLoss:
    if min(preservation_weight, correction_weight, drift_weight) < 0.0:
        raise ValueError("Domain-Safe loss weights must be non-negative")
    adapted = adapter(
        raw_image_tokens,
        int(grid_h),
        int(grid_w),
        return_output=True,
    )
    if not isinstance(adapted, DomainAdapterOutput):
        raise TypeError("adapter must return DomainAdapterOutput")
    with torch.no_grad():
        raw_ranking = deployment_maxsim_ranking_loss(
            raw_image_tokens,
            template_tokens,
            references,
            reference_mask,
            temperature=temperature,
        )
    ranking = deployment_maxsim_ranking_loss(
        adapted.tokens,
        template_tokens,
        references,
        reference_mask,
        temperature=temperature,
    )
    normalization_count = (
        raw_image_tokens.shape[0] + template_tokens.shape[0]
    )
    margins = safe_margin_terms(
        raw_ranking.scores,
        ranking.scores,
        normalization_count=normalization_count,
        preserve_cap=preserve_cap,
        error_margin=error_margin,
    )
    drift = adapted.residual_ratio.square().mean()
    total = (
        ranking.total
        + float(preservation_weight) * margins.preservation
        + float(correction_weight) * margins.correction
        + float(drift_weight) * drift
    )
    if not bool(torch.isfinite(total)):
        raise AssertionError("Domain-Safe total loss is non-finite")
    return DomainSafeTrainingLoss(
        total=total,
        ranking=ranking,
        raw_ranking=raw_ranking,
        margins=margins,
        drift=drift,
        adapted=adapted,
    )
