"""Shared, candidate-order-equivariant evidence readers for REBUT QR/QRR.

No dataset, identity IDs, retrieval ranks, or trainable backbone enter readers.
Query-side inputs must already be aligned to ONE COMMON QUERY GRID. Reference
inputs stay on each reference's native grid and are independently pooled; native
reference token indices are never compared across candidates. Valid masks exclude
padding/geometric invalidity only, NEVER low confidence. Descriptors and support
values are immutable cached inputs; this module learns only their readout.

The zero-initialized final residual projection initially blocks gradients to its
upstream reader. A nonzero first update opens that path; checks must span several
updates. These interface checks do not constitute scientific evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Arm = Literal["B_CAL", "QR", "QR_VEC", "QRR"]


@dataclass(frozen=True)
class EvidenceBatch:
    query: Tensor                 # [batch, candidates, common_query_tokens, Dq]
    reference: Tensor             # [batch, candidates, reference_tokens, Dr]
    query_valid: Tensor           # bool [batch, candidates, common_query_tokens]
    reference_valid: Tensor       # bool [batch, candidates, reference_tokens]

    def validate(self) -> None:
        if self.query.ndim != 4 or self.reference.ndim != 4:
            raise ValueError("evidence must have shape [B,C,N,D]")
        if self.query.shape[:2] != self.reference.shape[:2]:
            raise ValueError("query/reference candidate axes differ")
        if min(*self.query.shape, *self.reference.shape) < 1:
            raise ValueError("empty evidence dimensions")
        for values, mask in ((self.query, self.query_valid),
                             (self.reference, self.reference_valid)):
            if mask.dtype != torch.bool or mask.shape != values.shape[:-1]:
                raise ValueError("validity must be a bool mask on the token axis")
            if values.device != mask.device:
                raise ValueError("evidence and mask devices differ")
            if not torch.is_floating_point(values):
                raise ValueError("evidence must be floating point")
            if not torch.isfinite(values[mask]).all():
                raise ValueError("valid evidence contains nonfinite values")
        if (self.query.dtype != self.reference.dtype or
                self.query.device != self.reference.device):
            raise ValueError("both sides must have matching dtype/device")


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    clean = torch.where(mask.unsqueeze(-1), values, 0.0)
    return clean.sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)


class MeanAttentionPool(nn.Module):
    """Concatenated masked mean and learned attention; all-invalid returns zero."""

    def __init__(self, width: int):
        super().__init__()
        self.attention = nn.Linear(width, 1, bias=False)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        clean = torch.where(mask.unsqueeze(-1), values, 0.0)
        logits = self.attention(clean).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = logits.softmax(-1) * mask.to(logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(
            torch.finfo(logits.dtype).tiny)
        return torch.cat((masked_mean(clean, mask),
                          (clean * weights.unsqueeze(-1)).sum(-2)), -1)


@dataclass(frozen=True)
class EncodedEvidence:
    query_tokens: Tensor
    query_valid: Tensor
    reference_summary: Tensor
    independent_summary: Tensor


class EvidenceEncoder(nn.Module):
    """Same useful backbone in all three reader arms; no padding statistics fit."""

    def __init__(self, query_dim: int, reference_dim: int,
                 width: int = 64, input_width: int = 32):
        super().__init__()
        self.query_dim, self.reference_dim = query_dim, reference_dim
        self.width = width
        def stem(dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(dim, input_width), nn.GELU(),
                                 nn.Linear(input_width, width), nn.GELU())
        self.query_stem = stem(query_dim)
        # Equal-dimensional forward/reverse bundles use the same token MLP.
        self.reference_stem = (self.query_stem if query_dim == reference_dim
                               else stem(reference_dim))
        self.pool = MeanAttentionPool(width)
        self.summary = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())

    def forward(self, batch: EvidenceBatch) -> EncodedEvidence:
        if batch.query.shape[-1] != self.query_dim:
            raise ValueError("query descriptor dimension changed")
        if batch.reference.shape[-1] != self.reference_dim:
            raise ValueError("reference descriptor dimension changed")
        q = self.query_stem(torch.where(batch.query_valid[..., None],
                                       batch.query, 0.0))
        r = self.reference_stem(torch.where(batch.reference_valid[..., None],
                                           batch.reference, 0.0))
        q = torch.where(batch.query_valid[..., None], q, 0.0)
        r = torch.where(batch.reference_valid[..., None], r, 0.0)
        qp = self.pool(q, batch.query_valid)
        rp = self.pool(r, batch.reference_valid)
        independent = self.summary(torch.cat((qp, rp), -1))
        return EncodedEvidence(q, batch.query_valid, rp, independent)


def _gather(values: Tensor, index: Tensor) -> Tensor:
    """Gather candidate positions [B,P] without confusing native token axes."""
    if index.ndim != 2 or index.shape[0] != values.shape[0]:
        raise ValueError("candidate indices must have shape [B,P]")
    if index.dtype != torch.long:
        raise ValueError("candidate indices must be long")
    return values[torch.arange(values.shape[0], device=values.device)[:, None], index]


class EvidenceResidual(nn.Module):
    """QR scalar / QR_VEC late interaction / QRR local early interaction.

    encode -> pair_difference permits arbitrary rival checks; forward compares
    every candidate with the single original anchor, using bounded pair chunks.
    No dropout or batch normalization: a single pair is independent of its batch.
    """

    def __init__(self, arm: Arm, query_dim: int, reference_dim: int,
                 width: int = 64, input_width: int = 32, vector_dim: int = 16,
                 comparison_width: int = 32,
                 max_parameters: int = 250_000):
        super().__init__()
        if arm not in ("QR", "QR_VEC", "QRR"):
            raise ValueError("residual arm must be QR, QR_VEC, or QRR")
        self.arm = arm
        self.encoder = EvidenceEncoder(query_dim, reference_dim, width, input_width)
        if arm == "QR":
            self.output = nn.Linear(width, 1, bias=False)
        elif arm == "QR_VEC":
            self.vector = nn.Linear(width, vector_dim)
            self.comparison = nn.Sequential(nn.Linear(4 * vector_dim, comparison_width),
                                            nn.GELU())
            self.output = nn.Linear(comparison_width, 1, bias=False)
        else:
            # Presence bits retain one-sided valid evidence. Union pooling avoids
            # dropping locations supported by just one competing reference.
            self.local_comparison = nn.Sequential(
                nn.Linear(4 * width + 2, comparison_width), nn.GELU(),
                nn.Linear(comparison_width, comparison_width), nn.GELU())
            self.local_pool = MeanAttentionPool(comparison_width)
            self.comparison = nn.Sequential(
                nn.Linear(2 * comparison_width + 4 * width, comparison_width), nn.GELU(),
                nn.Linear(comparison_width, comparison_width), nn.GELU())
            self.output = nn.Linear(comparison_width, 1, bias=False)
            # QRR pools AFTER joint local comparison; the independent summary
            # path is intentionally absent rather than counted as dead capacity.
            self.encoder.summary = nn.Identity()
        nn.init.zeros_(self.output.weight)
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if count > max_parameters:
            raise ValueError(f"reader has {count} parameters, exceeds {max_parameters}; "
                             "freeze a smaller shared input_width across all arms")

    def encode(self, batch: EvidenceBatch) -> EncodedEvidence:
        return self.encoder(batch)

    def independent_score(self, encoded: EncodedEvidence) -> Tensor:
        if self.arm != "QR":
            raise ValueError("only QR defines rival-independent scalar evidence")
        return self.output(encoded.independent_summary).squeeze(-1)

    def _ordered(self, encoded: EncodedEvidence, left: Tensor, right: Tensor) -> Tensor:
        if self.arm == "QR_VEC":
            z = self.vector(encoded.independent_summary)
            a, b = _gather(z, left), _gather(z, right)
            features = torch.cat((a, b, a - b, a * b), -1)
        else:
            a = _gather(encoded.query_tokens, left)
            b = _gather(encoded.query_tokens, right)
            ma, mb = _gather(encoded.query_valid, left), _gather(encoded.query_valid, right)
            inputs = torch.cat((a, b, a - b, a * b,
                                ma[..., None].to(a.dtype),
                                mb[..., None].to(a.dtype)), -1)
            local = self.local_comparison(inputs)
            joint = self.local_pool(local, ma | mb)
            # Independently pooled reference sides only; no native-grid subtraction.
            features = torch.cat((joint, _gather(encoded.reference_summary, left),
                                  _gather(encoded.reference_summary, right)), -1)
        return self.output(self.comparison(features)).squeeze(-1)

    def pair_difference(self, encoded: EncodedEvidence,
                        left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("left/right candidate index shapes differ")
        if self.arm == "QR":
            score = self.independent_score(encoded)
            return _gather(score, left) - _gather(score, right)
        return 0.5 * (self._ordered(encoded, left, right) -
                      self._ordered(encoded, right, left))

    def forward(self, batch: EvidenceBatch, anchor: Tensor,
                pair_chunk_size: int = 16) -> Tensor:
        b, c = batch.query.shape[:2]
        _check_anchor(anchor, b, c)
        if pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be positive")
        encoded = self.encode(batch)
        pieces = []
        for start in range(0, c, pair_chunk_size):
            positions = torch.arange(start, min(start + pair_chunk_size, c),
                                     device=batch.query.device).expand(b, -1)
            pieces.append(self.pair_difference(encoded, positions,
                                              anchor[:, None].expand_as(positions)))
        return torch.cat(pieces, 1)


def _check_anchor(anchor: Tensor, batch: int, candidates: int) -> None:
    if anchor.shape != (batch,) or anchor.dtype != torch.long:
        raise ValueError("anchor must be long [B]")
    if (anchor < 0).any() or (anchor >= candidates).any():
        raise ValueError("anchor outside natural candidate axis")


class RebutScorer(nn.Module):
    """Shared linear decision head plus a zero-initialized evidence residual.

    x0[B,C,F] contains the SAME existing evidence for all arms, expressed against
    the original anchor. It may include M0, Lcur and retrieval scores; evidence
    readers receive none of those rank/identity bookkeeping values separately.
    Initial parameters can copy any dimension-compatible baseline head exactly.
    """

    def __init__(self, arm: Arm, existing_dim: int, query_dim: int = 1,
                 reference_dim: int = 1, initial_weight: Tensor | None = None,
                 initial_bias: float = 0.0, head_bias: bool = True, **reader_options):
        super().__init__()
        if arm not in ("B_CAL", "QR", "QR_VEC", "QRR"):
            raise ValueError("unknown experiment arm")
        self.arm = arm
        self.head = nn.Linear(existing_dim, 1, bias=head_bias)
        with torch.no_grad():
            self.head.weight.zero_()
            if self.head.bias is not None:
                self.head.bias.fill_(initial_bias)
            elif initial_bias != 0:
                raise ValueError("initial_bias requires head_bias=True")
            if initial_weight is not None:
                if initial_weight.numel() != existing_dim:
                    raise ValueError("initial head dimension mismatch")
                self.head.weight.copy_(initial_weight.reshape(1, -1))
        self.reader = (None if arm == "B_CAL" else
                       EvidenceResidual(arm, query_dim, reference_dim, **reader_options))

    def forward(self, existing: Tensor, evidence: EvidenceBatch | None,
                anchor: Tensor, pair_chunk_size: int = 16) -> Tensor:
        if existing.ndim != 3:
            raise ValueError("existing evidence must be [B,C,F]")
        b, c = existing.shape[:2]
        _check_anchor(anchor, b, c)
        logits = self.head(existing).squeeze(-1)
        if self.reader is not None:
            if evidence is None or evidence.query.shape[:2] != (b, c):
                raise ValueError("reader evidence candidate axis mismatch")
            logits = logits + self.reader(evidence, anchor, pair_chunk_size)
        anchor_mask = torch.arange(c, device=existing.device)[None] == anchor[:, None]
        return torch.where(anchor_mask, 0.0, logits)


def shared_hit_miss_loss(logits: Tensor, correct_identity: Tensor, anchor: Tensor,
                         candidate_valid: Tensor | None = None,
                         reduction: str = "mean") -> Tensor:
    """Per-query loss: identity probability SUM for hits, softplus for misses.

    correct_identity is bool[B,C], possibly multiple true physical references.
    Zero true entries means target absent and NEVER labels HOLD as correct.
    Candidate validity masks only padding, never missing-target injection.
    """
    if logits.ndim != 2 or correct_identity.shape != logits.shape:
        raise ValueError("logits and identity mask must be [B,C]")
    if correct_identity.dtype != torch.bool:
        raise ValueError("identity membership must be boolean")
    b, c = logits.shape
    _check_anchor(anchor, b, c)
    valid = (torch.ones_like(correct_identity) if candidate_valid is None
             else candidate_valid)
    if valid.dtype != torch.bool or valid.shape != logits.shape:
        raise ValueError("candidate validity must be bool [B,C]")
    if (correct_identity & ~valid).any():
        raise ValueError("correct identity cannot refer to padding")
    anchor_mask = torch.arange(c, device=logits.device)[None] == anchor[:, None]
    if not valid[anchor_mask].all() or (valid.sum(-1) < 2).any():
        raise ValueError("each query needs its anchor and at least one challenger")
    if not torch.isfinite(logits[valid]).all():
        raise ValueError("nonfinite candidate logits")
    if not (logits[anchor_mask] == 0).all():
        raise ValueError("HOLD anchor must have score zero")
    hit = correct_identity.any(-1)
    # Avoid logsumexp(all -inf) on miss rows, whose backward would be undefined.
    safe_correct = correct_identity | ((~hit)[:, None] & anchor_mask)
    clean = logits.masked_fill(~valid, -torch.inf)
    hit_loss = torch.logsumexp(clean, -1) - torch.logsumexp(
        clean.masked_fill(~safe_correct, -torch.inf), -1)
    challengers = valid & ~anchor_mask
    miss_loss = (F.softplus(torch.where(challengers, logits, 0.0)) *
                 challengers).sum(-1) / challengers.sum(-1)
    result = torch.where(hit, hit_loss, miss_loss)
    if reduction == "none":
        return result
    if reduction == "mean":
        return result.mean()
    if reduction == "sum":
        return result.sum()
    raise ValueError("reduction must be none, mean or sum")


@torch.no_grad()
def fixed_anchor_decision(logits: Tensor, anchor: Tensor, threshold: float | Tensor = 0.0,
                          candidate_keys: Tensor | None = None,
                          candidate_valid: Tensor | None = None) -> Tensor:
    """One-shot decision, strict threshold; equal top scores retain the anchor.

    Scores at the threshold retain HOLD. Equal highest challenger scores also
    retain HOLD, avoiding candidate-order dependence. candidate_keys are optional
    immutable physical row IDs used ONLY to validate that no physical row repeats;
    they are never features or learned tie breakers. Returns candidate positions.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [B,C]")
    b, c = logits.shape
    _check_anchor(anchor, b, c)
    valid = (torch.ones_like(logits, dtype=torch.bool) if candidate_valid is None
             else candidate_valid)
    if valid.shape != logits.shape or valid.dtype != torch.bool:
        raise ValueError("candidate validity must be bool [B,C]")
    anchor_mask = torch.arange(c, device=logits.device)[None] == anchor[:, None]
    if not valid[anchor_mask].all() or not torch.isfinite(logits[valid]).all():
        raise ValueError("invalid anchor or logits")
    if not (logits[anchor_mask] == 0).all():
        raise ValueError("HOLD anchor must have score zero")
    if candidate_keys is not None:
        if candidate_keys.shape != logits.shape:
            raise ValueError("candidate keys must have shape [B,C]")
        for keys, mask in zip(candidate_keys, valid):
            if keys[mask].unique().numel() != int(mask.sum()):
                raise ValueError("physical candidate IDs must be unique")
    challenger_logits = logits.masked_fill(~valid | anchor_mask, -torch.inf)
    maximum, choice = challenger_logits.max(-1)
    tied = (challenger_logits == maximum[:, None]).sum(-1) != 1
    tau = torch.as_tensor(threshold, dtype=logits.dtype, device=logits.device)
    if tau.ndim > 1 or (tau.ndim == 1 and tau.shape != (b,)) or not torch.isfinite(tau).all():
        raise ValueError("threshold must be finite scalar or [B]")
    switch = (maximum > tau) & ~tied & torch.isfinite(maximum)
    return torch.where(switch, choice, anchor)


def parameter_report(models: dict[str, nn.Module]) -> dict:
    counts = {name: sum(p.numel() for p in model.parameters() if p.requires_grad)
              for name, model in models.items()}
    readers = {name: n for name, n in counts.items() if name != "B_CAL"}
    mean = sum(readers.values()) / max(len(readers), 1)
    relative = {name: n / mean - 1.0 for name, n in readers.items()}
    return {"trainable_parameters": counts, "relative_to_reader_arm_mean": relative,
            "all_under_250k": all(n <= 250000 for n in counts.values()),
            "reader_arms_within_10_percent_of_mean": all(abs(x) <= .1 for x in relative.values()),
            "capacity_note": "Same token backbone; all counted parameters are active. "
            "Any real capacity mismatch is reported, never hidden with unused parameters."}
