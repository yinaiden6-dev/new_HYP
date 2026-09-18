"""Domain-first photo-query alignment for frozen full-reference retrieval.

The reference gallery remains fixed. Only visual query tokens are adapted.
Adapters are exactly identity at initialization and use a bounded non-sigmoid
residual, avoiding B2's under-actuated scalar.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DomainAdapterOutput:
    tokens: torch.Tensor
    delta: torch.Tensor
    residual_ratio: torch.Tensor


@dataclass
class DomainRankingLoss:
    total: torch.Tensor
    scores: torch.Tensor
    target_margin: torch.Tensor
    target_rank: torch.Tensor


@dataclass
class TokenAlignmentLoss:
    total: torch.Tensor
    anchor_count: int
    mean_margin: torch.Tensor


def _validate_grid(tokens: torch.Tensor, grid_h: int, grid_w: int) -> None:
    if tokens.ndim != 2:
        raise ValueError("query tokens must be [P,D]")
    if grid_h <= 0 or grid_w <= 0 or grid_h * grid_w != tokens.shape[0]:
        raise ValueError(
            f"grid {(grid_h, grid_w)} does not match {tokens.shape[0]} tokens"
        )


def _bounded_unit_delta(raw_delta: torch.Tensor) -> torch.Tensor:
    """Keep each residual vector at norm <= 1 with stable zero gradients."""
    norm = raw_delta.norm(dim=-1, keepdim=True)
    return raw_delta / norm.clamp_min(1.0)


def tangent_permuted_same_drift(
    tokens: torch.Tensor,
    delta: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Destroy tangent direction while preserving exact per-token drift geometry.

    Relative to unit token ``x``, this preserves both ``delta·x`` and
    ``||delta||``. Normalization of ``x + ratio*delta`` therefore has the same
    cosine drift from ``x`` before and after the control.
    """
    if tokens.shape != delta.shape or tokens.ndim != 2:
        raise ValueError("tokens and delta must have the same [P,D] shape")
    if permutation.shape != (tokens.shape[-1],):
        raise ValueError("permutation must have shape [D]")
    x = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    d = delta.float()
    parallel = (d * x).sum(dim=-1, keepdim=True)
    perpendicular = d - parallel * x
    perpendicular_norm = perpendicular.norm(dim=-1, keepdim=True)

    shuffled = d.index_select(-1, permutation.to(d.device))
    shuffled = shuffled - (shuffled * x).sum(dim=-1, keepdim=True) * x
    shuffled_norm = shuffled.norm(dim=-1, keepdim=True)

    fallback = torch.roll(x, shifts=1, dims=-1)
    fallback = fallback - (fallback * x).sum(dim=-1, keepdim=True) * x
    fallback = F.normalize(fallback, dim=-1, eps=1e-6)
    direction = torch.where(
        shuffled_norm > 1e-8,
        shuffled / shuffled_norm.clamp_min(1e-8),
        fallback,
    )
    controlled = parallel * x + perpendicular_norm * direction
    return controlled.to(delta.dtype)


class _BoundedQueryAdapter(nn.Module):
    def __init__(self, dim: int = 128, max_ratio: float = 0.50):
        super().__init__()
        if max_ratio <= 0.0:
            raise ValueError("max_ratio must be positive")
        self.dim = int(dim)
        self.max_ratio = float(max_ratio)

    def raw_delta(
        self, tokens: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
        *,
        tangent_permutation: torch.Tensor | None = None,
        return_output: bool = False,
    ) -> torch.Tensor | DomainAdapterOutput:
        _validate_grid(tokens, grid_h, grid_w)
        delta = _bounded_unit_delta(self.raw_delta(tokens, grid_h, grid_w))
        if tangent_permutation is not None:
            delta = tangent_permuted_same_drift(
                tokens,
                delta,
                tangent_permutation,
            )
        base = tokens.float()
        base_norm = base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        candidate = F.normalize(
            base + self.max_ratio * delta.float(),
            dim=-1,
            eps=1e-6,
        ) * base_norm
        # Encoder tokens are nearly, but not bitwise, unit norm. At the
        # zero-initialized head the deployed B0 path must be numerically exact,
        # while gradients must still flow through ``candidate``. A detached
        # value correction supplies both properties.
        zero_delta = (
            delta.detach().abs().amax(dim=-1, keepdim=True) == 0.0
        ).to(candidate.dtype)
        adapted = (
            candidate + zero_delta * (base - candidate).detach()
        ).to(tokens.dtype)
        ratio = (
            self.max_ratio
            * delta.float().norm(dim=-1)
            / tokens.float().norm(dim=-1).clamp_min(1e-6)
        )
        output = DomainAdapterOutput(adapted, delta, ratio)
        return output if return_output else output.tokens


class PointwiseQueryDomainAdapter(_BoundedQueryAdapter):
    """High-capacity per-token photo-to-reference map."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        max_ratio: float = 0.50,
    ):
        super().__init__(dim, max_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.delta_head = nn.Linear(hidden, dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def raw_delta(
        self, tokens: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        _validate_grid(tokens, grid_h, grid_w)
        hidden = F.gelu(self.fc1(self.norm(tokens)))
        hidden = F.gelu(self.fc2(hidden))
        return self.delta_head(hidden)


class SpatialQueryDomainAdapter(_BoundedQueryAdapter):
    """Query-domain adapter with exact-grid local and global spatial context."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        heads: int = 4,
        layers: int = 2,
        max_ratio: float = 0.50,
    ):
        super().__init__(dim, max_ratio)
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.norm = nn.LayerNorm(dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.depthwise = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            block,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.delta_head = nn.Linear(dim, dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    @staticmethod
    def coordinates(
        grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, grid_h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, grid_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1).reshape(-1, 2)

    def raw_delta(
        self, tokens: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        _validate_grid(tokens, grid_h, grid_w)
        normalized = self.norm(tokens)
        grid = normalized.T.reshape(1, self.dim, grid_h, grid_w)
        local = self.pointwise(F.gelu(self.depthwise(grid)))
        local = local.reshape(self.dim, -1).T
        coords = self.coordinates(
            grid_h,
            grid_w,
            tokens.device,
            tokens.dtype,
        )
        hidden = normalized + local + self.coord_mlp(coords)
        contextual = self.context(hidden.unsqueeze(0)).squeeze(0)
        return self.delta_head(contextual)


def build_query_domain_adapter(
    mode: str,
    *,
    dim: int = 128,
    hidden: int = 512,
    heads: int = 4,
    layers: int = 2,
    max_ratio: float = 0.50,
) -> _BoundedQueryAdapter:
    if mode == "pointwise":
        return PointwiseQueryDomainAdapter(dim, hidden, max_ratio)
    if mode == "spatial":
        return SpatialQueryDomainAdapter(
            dim, hidden, heads, layers, max_ratio
        )
    raise ValueError(f"unknown domain adapter mode: {mode}")


def fixed_channel_permutation(
    dim: int,
    *,
    seed: int = 20260727,
    device: torch.device | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(dim, generator=generator)
    if bool(torch.equal(permutation, torch.arange(dim))):
        permutation = torch.roll(permutation, shifts=1)
    return permutation.to(device=device)


def deployment_maxsim_ranking_loss(
    image_tokens: torch.Tensor,
    template_tokens: torch.Tensor,
    references: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> DomainRankingLoss:
    """Exact deployed SUM-MaxSim over positive-first training candidates."""
    if image_tokens.ndim != 2 or template_tokens.ndim != 2:
        raise ValueError("image/template tokens must be [P,D]")
    if references.ndim != 3 or reference_mask.shape != references.shape[:-1]:
        raise ValueError("references/mask must be [C,Q,D] and [C,Q]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if image_tokens.shape[-1] != references.shape[-1]:
        raise ValueError("embedding dimensions differ")
    image_similarity = torch.einsum(
        "pd,cqd->pcq",
        image_tokens.float(),
        references.float(),
    ).masked_fill(~reference_mask[None, :, :], float("-inf"))
    scores = image_similarity.amax(dim=-1).sum(dim=0)
    if template_tokens.shape[0]:
        template_similarity = torch.einsum(
            "td,cqd->tcq",
            template_tokens.float(),
            references.float(),
        ).masked_fill(~reference_mask[None, :, :], float("-inf"))
        scores = scores + template_similarity.amax(dim=-1).sum(dim=0)
    if not bool(torch.isfinite(scores).all()):
        raise AssertionError("candidate MaxSim scores contain NaN or Inf")
    normalized_scores = scores / float(
        image_tokens.shape[0] + template_tokens.shape[0]
    )
    total = F.cross_entropy(
        (normalized_scores / float(temperature)).view(1, -1),
        torch.zeros(1, dtype=torch.long, device=scores.device),
    )
    margin = scores[0] - scores[1:].amax()
    rank = (scores[1:] > scores[0]).sum()
    return DomainRankingLoss(total, scores, margin, rank)


@torch.no_grad()
def conservative_alignment_indices(
    raw_query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    positive_margin: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return query anchor indices, paired-reference targets, and margins."""
    q = F.normalize(raw_query.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    n = F.normalize(negatives.float(), dim=-1)
    qp = q @ p.T
    pos_score, q_to_p = qp.max(dim=-1)
    negative_similarity = torch.einsum("pd,nqd->pnq", q, n)
    negative_similarity = negative_similarity.masked_fill(
        ~negative_mask[None, :, :],
        float("-inf"),
    )
    neg_score = negative_similarity.amax(dim=(-1, -2))
    margin = pos_score - neg_score
    p_to_q = qp.argmax(dim=0)
    query_ids = torch.arange(q.shape[0], device=q.device)
    mutual = p_to_q[q_to_p] == query_ids
    anchors = mutual & (margin >= float(positive_margin))
    return anchors.nonzero(as_tuple=False).flatten(), q_to_p[anchors], margin


def paired_token_alignment_loss(
    raw_query: torch.Tensor,
    adapted_query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    positive_margin: float = 0.08,
) -> TokenAlignmentLoss:
    query_indices, reference_indices, margin = conservative_alignment_indices(
        raw_query,
        positive,
        negatives,
        negative_mask,
        positive_margin=positive_margin,
    )
    if query_indices.numel() == 0:
        zero = adapted_query.sum() * 0.0
        return TokenAlignmentLoss(zero, 0, margin.mean())
    predicted = F.normalize(
        adapted_query.index_select(0, query_indices).float(),
        dim=-1,
    )
    target = F.normalize(
        positive.index_select(0, reference_indices).float().detach(),
        dim=-1,
    )
    total = (1.0 - (predicted * target).sum(dim=-1)).mean()
    return TokenAlignmentLoss(
        total,
        int(query_indices.numel()),
        margin.index_select(0, query_indices).mean(),
    )
