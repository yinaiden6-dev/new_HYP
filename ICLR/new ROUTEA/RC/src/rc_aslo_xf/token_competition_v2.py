"""Frozen-head, original-query-token competition readers for TOKEN V2.

The 264-vector is normalized q128, its free-MaxSim matched reference128 and
eight cached scalar channels. No spatial pooling precedes the shared token stem.
Only physical IDs used to resolve frozen-baseline ties enter rival selection;
they never enter the learned reader. Reference placeholders are never consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .rebut_qr_qrr_v1 import (
    EvidenceBatch, MeanAttentionPool, _check_anchor, _gather,
    fixed_anchor_decision, shared_hit_miss_loss,
)

Arm = Literal["TOKEN_QR", "TOKEN_QRR_ANCHOR", "TOKEN_QRR_MULTI"]
ARMS = ("TOKEN_QR", "TOKEN_QRR_ANCHOR", "TOKEN_QRR_MULTI")


@dataclass(frozen=True)
class TokenEvidence(EvidenceBatch):
    """EvidenceBatch-compatible input; reference fields may both be None.

    query is FP64 [B,C,N,264], query_valid is bool [B,C,N]. The N axis must
    denote the same original query tokens in every candidate. physical_axis is
    long [B,C], containing unique immutable physical IDs (never slot IDs).
    """

    reference: Tensor | None
    reference_valid: Tensor | None
    physical_axis: Tensor | None = None

    def validate(self, check_finite: bool = True) -> None:
        _validate_query(self, check_finite)
        if self.physical_axis is not None:
            _validate_axis(self.physical_axis, self.query.shape[:2], self.query.device)


def _validate_query(batch: EvidenceBatch, check_finite: bool = False) -> None:
    q, mask = batch.query, batch.query_valid
    if q.ndim != 4 or q.shape[-1] != 264 or min(q.shape) < 1:
        raise ValueError("query must have nonempty shape [B,C,N,264]")
    if q.dtype != torch.float64:
        raise ValueError("TOKEN V2 requires FP64 query evidence")
    if mask.dtype != torch.bool or mask.shape != q.shape[:-1]:
        raise ValueError("query_valid must be bool [B,C,N]")
    if mask.device != q.device:
        raise ValueError("query and query_valid devices differ")
    if check_finite:
        for start in range(0, q.shape[1], 16):
            values, valid = q[:, start:start + 16], mask[:, start:start + 16]
            if not torch.isfinite(values[valid]).all():
                raise ValueError("valid query evidence contains nonfinite values")


def _validate_axis(axis: Tensor, shape, device) -> None:
    if axis.shape != shape or axis.dtype != torch.long or axis.device != device:
        raise ValueError("physical_axis must be long [B,C] on the evidence device")
    if any(row.unique().numel() != row.numel() for row in axis):
        raise ValueError("physical_axis contains duplicate physical IDs")


@dataclass(frozen=True)
class EncodedTokens:
    query_tokens: Tensor
    query_valid: Tensor


class TokenEncoder(nn.Module):
    def __init__(self, checkpoint_chunks: bool = True):
        super().__init__()
        self.query_stem = nn.Sequential(
            nn.Linear(264, 24), nn.GELU(), nn.Linear(24, 24), nn.GELU())
        self.checkpoint_chunks = checkpoint_chunks

    def _chunk(self, values: Tensor, mask: Tensor) -> Tensor:
        encoded = self.query_stem(torch.where(mask[..., None], values, 0.0))
        return torch.where(mask[..., None], encoded, 0.0)

    def forward(self, batch: EvidenceBatch, chunk_size: int = 16) -> EncodedTokens:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        pieces = []
        for start in range(0, batch.query.shape[1], chunk_size):
            q = batch.query[:, start:start + chunk_size]
            m = batch.query_valid[:, start:start + chunk_size]
            if self.checkpoint_chunks and torch.is_grad_enabled():
                z = checkpoint(self._chunk, q, m, use_reentrant=False)
            else:
                z = self._chunk(q, m)
            pieces.append(z)
        return EncodedTokens(torch.cat(pieces, 1), batch.query_valid)


class TokenResidual(nn.Module):
    """QR independently pools; QRR compares aligned tokens before pooling.

    ANCHOR and MULTI have identical parameter names/shapes and initialization.
    Their only difference is the deterministic, target-free rival set.
    """

    def __init__(self, arm: Arm, checkpoint_chunks: bool = True):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown token arm: {arm}")
        self.arm = arm
        self.checkpoint_chunks = checkpoint_chunks
        self.encoder = TokenEncoder(checkpoint_chunks)
        if arm == "TOKEN_QR":
            self.local_comparison = nn.Sequential(
                nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24), nn.GELU())
            self.comparison = nn.Sequential(
                nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 24), nn.GELU())
        else:
            self.local_comparison = nn.Sequential(
                nn.Linear(98, 24), nn.GELU(), nn.Linear(24, 24), nn.GELU())
            self.comparison = nn.Sequential(
                nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 24), nn.GELU())
        self.local_pool = MeanAttentionPool(24)
        self.output = nn.Linear(24, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def encode(self, batch: EvidenceBatch, chunk_size: int = 16) -> EncodedTokens:
        return self.encoder(batch, chunk_size)

    def _independent(self, q: Tensor, mask: Tensor) -> Tensor:
        local = self.local_comparison(q)
        return self.output(self.comparison(self.local_pool(local, mask))).squeeze(-1)

    def independent_score(self, encoded: EncodedTokens, chunk_size: int = 16) -> Tensor:
        if self.arm != "TOKEN_QR":
            raise ValueError("only TOKEN_QR defines an independent candidate score")
        pieces = []
        for start in range(0, encoded.query_tokens.shape[1], chunk_size):
            q = encoded.query_tokens[:, start:start + chunk_size]
            m = encoded.query_valid[:, start:start + chunk_size]
            if self.checkpoint_chunks and torch.is_grad_enabled():
                value = checkpoint(self._independent, q, m, use_reentrant=False)
            else:
                value = self._independent(q, m)
            pieces.append(value)
        return torch.cat(pieces, 1)

    def _ordered(self, q: Tensor, mask: Tensor, left: Tensor, right: Tensor) -> Tensor:
        a, b = _gather(q, left), _gather(q, right)
        ma, mb = _gather(mask, left), _gather(mask, right)
        local = self.local_comparison(torch.cat(
            (a, b, a - b, a * b, ma[..., None].to(a.dtype),
             mb[..., None].to(a.dtype)), -1))
        pooled = self.local_pool(local, ma | mb)
        return self.output(self.comparison(pooled)).squeeze(-1)

    def _pair(self, q: Tensor, mask: Tensor, left: Tensor, right: Tensor) -> Tensor:
        return 0.5 * (self._ordered(q, mask, left, right) -
                      self._ordered(q, mask, right, left))

    def pair_difference(self, encoded: EncodedTokens, left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("left/right shapes differ")
        if self.arm == "TOKEN_QR":
            scores = self.independent_score(encoded)
            return _gather(scores, left) - _gather(scores, right)
        args = (encoded.query_tokens, encoded.query_valid, left, right)
        if self.checkpoint_chunks and torch.is_grad_enabled():
            return checkpoint(self._pair, *args, use_reentrant=False)
        return self._pair(*args)

    @staticmethod
    def select_rivals(base: Tensor, anchor: Tensor, physical_axis: Tensor,
                      multi: bool) -> tuple[Tensor, Tensor]:
        """RAW anchor plus best frozen-base rival excluding self AND anchor.

        Ties use ascending immutable physical ID. Scores are the CURRENT frozen
        head applied to this forward's x0, including its stage normalization.
        The C=2 fallback keeps the anchor-only comparison for the challenger.
        """
        b, c = base.shape
        positions = torch.arange(c, device=base.device)[None].expand(b, -1)
        anchors = anchor[:, None].expand_as(positions)
        rivals = torch.stack((anchors, anchors), -1).clone()
        valid = torch.stack((positions != anchors, torch.zeros_like(positions,
                                                                   dtype=torch.bool)), -1)
        if multi:
            if c < 2:
                raise ValueError("competition requires at least two candidates")
            physical_order = torch.argsort(physical_axis, dim=1, stable=True)
            sorted_scores = base.detach().gather(1, physical_order)
            score_order = torch.argsort(sorted_scores, dim=1, descending=True, stable=True)
            order = physical_order.gather(1, score_order)
            nonanchor = order[order != anchor[:, None]].reshape(b, c - 1)
            best = nonanchor[:, 0, None].expand_as(positions)
            second = (nonanchor[:, 1, None].expand_as(positions) if c > 2 else anchors)
            extra = torch.where(positions == best, second, best)
            rivals[..., 1] = extra
            valid[..., 1] = (extra != positions) & (extra != anchors)
        return rivals, valid

    def forward(self, batch: EvidenceBatch, anchor: Tensor, base_logits: Tensor,
                physical_axis: Tensor, pair_chunk_size: int = 16,
                return_trace: bool = False):
        b, c = batch.query.shape[:2]
        if pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be positive")
        encoded = self.encode(batch, pair_chunk_size)
        rivals, valid = self.select_rivals(base_logits, anchor, physical_axis,
                                         self.arm == "TOKEN_QRR_MULTI")
        if self.arm == "TOKEN_QR":
            d = self.independent_score(encoded, pair_chunk_size)
            pair = d[..., None] - _gather(d, rivals.reshape(b, -1)).reshape(b, c, 2)
            pair = torch.where(valid, pair, 0.0)
        else:
            pieces = []
            for start in range(0, c, pair_chunk_size):
                stop = min(start + pair_chunk_size, c)
                left = torch.arange(start, stop, device=batch.query.device)[None].expand(b, -1)
                edge_pieces = []
                for edge in range(2 if self.arm == "TOKEN_QRR_MULTI" else 1):
                    edge_value = self.pair_difference(encoded, left, rivals[:, start:stop, edge])
                    edge_pieces.append(torch.where(valid[:, start:stop, edge], edge_value, 0.0))
                if len(edge_pieces) == 1:
                    edge_pieces.append(torch.zeros_like(edge_pieces[0]))
                pieces.append(torch.stack(edge_pieces, -1))
            pair = torch.cat(pieces, 1)
            d = pair.sum(-1) / valid.sum(-1).clamp_min(1)
        if return_trace:
            return d, {"D": d, "rivals": rivals, "rival_valid": valid,
                       "pair_differences": pair}
        return d


class TokenCompetitionScorer(nn.Module):
    def __init__(self, arm: Arm, initial_weight: Tensor | None = None,
                 pair_chunk_size: int = 16, checkpoint_chunks: bool = True):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown token arm: {arm}")
        if pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be positive")
        self.arm = arm
        self.pair_chunk_size = pair_chunk_size
        self.head = nn.Linear(18, 1, bias=False)
        self.reader = TokenResidual(arm, checkpoint_chunks)
        self.double()
        with torch.no_grad():
            self.head.weight.zero_()
            if initial_weight is not None:
                if initial_weight.numel() != 18:
                    raise ValueError("initial head must contain 18 weights")
                self.head.weight.copy_(initial_weight.reshape(1, 18))
        self.head.requires_grad_(False)

    def _forward(self, existing: Tensor, evidence: EvidenceBatch, anchor: Tensor,
                 pair_chunk_size: int | None, physical_axis: Tensor | None,
                 return_trace: bool):
        if existing.ndim != 3 or existing.shape[-1] != 18:
            raise ValueError("existing x0 must be [B,C,18]")
        if existing.dtype != torch.float64:
            raise ValueError("TOKEN V2 requires FP64 existing x0")
        b, c = existing.shape[:2]
        _check_anchor(anchor, b, c)
        _validate_query(evidence)
        if evidence.query.shape[:2] != (b, c) or evidence.query.device != existing.device:
            raise ValueError("evidence and x0 candidate axes/devices differ")
        axis = physical_axis if physical_axis is not None else getattr(evidence, "physical_axis", None)
        if axis is None:
            if self.arm == "TOKEN_QRR_MULTI":
                raise ValueError("MULTI requires immutable physical_axis; slots are not physical IDs")
            axis = torch.arange(c, device=existing.device)[None].expand(b, -1)
        _validate_axis(axis, (b, c), existing.device)
        # RAW anchor is fixed; the baseline's anchor slot is its exact HOLD zero.
        anchor_mask = torch.arange(c, device=existing.device)[None] == anchor[:, None]
        base = torch.where(anchor_mask, 0.0, self.head(existing).squeeze(-1))
        if not torch.isfinite(base).all():
            raise ValueError("nonfinite frozen baseline logits")
        chunk = self.pair_chunk_size if pair_chunk_size is None else pair_chunk_size
        if chunk < 1:
            raise ValueError("pair_chunk_size must be positive")
        # Epoch-zero replay is exactly the inherited baseline. Preserve the
        # normal differentiable path for the first training gradient.
        if not torch.is_grad_enabled() and torch.count_nonzero(self.reader.output.weight) == 0:
            if not return_trace:
                return base
            rivals, valid = self.reader.select_rivals(base, anchor, axis,
                                                     self.arm == "TOKEN_QRR_MULTI")
            return base, {"base_logits": base, "D": torch.zeros_like(base),
                          "residual": torch.zeros_like(base), "rivals": rivals,
                          "rival_valid": valid,
                          "pair_differences": torch.zeros((*base.shape, 2),
                                                          dtype=base.dtype, device=base.device),
                          "physical_axis": axis}
        result = self.reader(evidence, anchor, base, axis, chunk, return_trace)
        d, trace = result if return_trace else (result, None)
        residual = d - d.gather(1, anchor[:, None])
        residual = torch.where(anchor_mask, 0.0, residual)
        logits = torch.where(anchor_mask, 0.0, base + residual)
        if return_trace:
            trace.update(base_logits=base, residual=residual, physical_axis=axis)
            return logits, trace
        return logits

    def forward(self, existing: Tensor, evidence: EvidenceBatch, anchor: Tensor,
                pair_chunk_size: int | None = None, physical_axis: Tensor | None = None) -> Tensor:
        return self._forward(existing, evidence, anchor, pair_chunk_size, physical_axis, False)

    def forward_with_trace(self, existing: Tensor, evidence: EvidenceBatch, anchor: Tensor,
                           pair_chunk_size: int | None = None,
                           physical_axis: Tensor | None = None) -> tuple[Tensor, dict]:
        return self._forward(existing, evidence, anchor, pair_chunk_size, physical_axis, True)


def new_model(arm: Arm, seed: int, optimizer_config: dict):
    """Old trainrunner factory signature; only reader parameters are optimized."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TokenCompetitionScorer(
        arm, pair_chunk_size=int(optimizer_config.get("pair_chunk_size", 16)),
        checkpoint_chunks=bool(optimizer_config.get("checkpoint_chunks", True)))
    optimizer = torch.optim.AdamW(
        model.reader.parameters(), lr=float(optimizer_config.get("lr", optimizer_config.get("learning_rate", .001))),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)))
    return model, optimizer


def parameter_report(models: dict[str, nn.Module]) -> dict:
    counts = {name: sum(p.numel() for p in model.parameters() if p.requires_grad)
              for name, model in models.items()}
    readers = {name: count for name, count in counts.items() if name in ARMS}
    mean = sum(readers.values()) / max(len(readers), 1)
    relative = {name: count / mean - 1.0 for name, count in readers.items()}
    shapes = {name: {key: list(value.shape) for key, value in model.reader.state_dict().items()}
              for name, model in models.items() if name in ARMS}
    return {
        "trainable_parameters": counts,
        "frozen_head_parameters": {name: sum(p.numel() for p in model.head.parameters())
                                   for name, model in models.items()},
        "relative_to_reader_arm_mean": relative,
        "all_under_250k": all(count <= 250000 for count in counts.values()),
        "reader_arms_within_10_percent_of_mean": all(abs(x) <= .1 for x in relative.values()),
        "anchor_multi_identical_parameter_shapes": (
            shapes.get("TOKEN_QRR_ANCHOR") == shapes.get("TOKEN_QRR_MULTI")),
        "capacity_note": "All counted reader parameters belong to active forward paths; "
                         "ANCHOR and MULTI differ only in rival selection. Frozen head has 18 parameters.",
    }


__all__ = ["ARMS", "TokenEvidence", "TokenCompetitionScorer", "TokenResidual",
           "new_model", "parameter_report", "fixed_anchor_decision", "shared_hit_miss_loss"]
