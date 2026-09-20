"""Candidate-binding contrastive no-regret representation-loss primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


BREAK_WEIGHT = 1.0
BINDING_WEIGHT = 1.0


class CBNRError(ValueError):
    pass


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise CBNRError(message)


def _scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    _require(
        isinstance(value, torch.Tensor)
        and value.ndim == 0
        and value.is_floating_point()
        and bool(torch.isfinite(value)),
        f"{name} must be a finite floating scalar",
    )
    return value


@dataclass(frozen=True)
class PairCenteredResidualV1:
    left: torch.Tensor
    right: torch.Tensor
    margin_left_minus_right: torch.Tensor

    def __post_init__(self) -> None:
        left = _scalar(self.left, "left residual")
        right = _scalar(self.right, "right residual")
        margin = _scalar(self.margin_left_minus_right, "residual margin")
        _require(
            left.device == right.device == margin.device
            and left.dtype == right.dtype == margin.dtype,
            "pair-centered dtype/device drift",
        )
        _require(torch.equal(left + right, torch.zeros_like(left)), "residual is not exact zero-sum")
        _require(torch.equal(left - right, margin), "residual margin reconstruction drift")


def pair_centered_residual(
    left_evidence: torch.Tensor, right_evidence: torch.Tensor
) -> PairCenteredResidualV1:
    left = _scalar(left_evidence, "left evidence")
    right = _scalar(right_evidence, "right evidence")
    _require(left.device == right.device and left.dtype == right.dtype, "evidence dtype/device drift")
    margin = left - right
    centered_left = margin * 0.5
    centered_right = -centered_left
    return PairCenteredResidualV1(centered_left, centered_right, margin)


def target_oriented_margin(
    centered: PairCenteredResidualV1, *, target_member_ordinal: int
) -> torch.Tensor:
    _require(target_member_ordinal in (0, 1), "target member ordinal drift")
    return (
        centered.margin_left_minus_right
        if target_member_ordinal == 0
        else -centered.margin_left_minus_right
    )


@dataclass(frozen=True)
class CBNRLossV1:
    total: torch.Tensor
    rank: torch.Tensor
    keep: torch.Tensor
    binding: torch.Tensor
    raw_margin: torch.Tensor
    real_residual_margin: torch.Tensor
    binding_control_residual_margin: torch.Tensor
    final_margin: torch.Tensor

    def __post_init__(self) -> None:
        values = (
            self.total,
            self.rank,
            self.keep,
            self.binding,
            self.raw_margin,
            self.real_residual_margin,
            self.binding_control_residual_margin,
            self.final_margin,
        )
        _require(
            all(_scalar(value, "CBNR loss field") is value for value in values),
            "loss field drift",
        )
        _require(
            torch.equal(self.total, self.rank + self.keep + self.binding),
            "CBNR total reconstruction drift",
        )


@dataclass(frozen=True)
class StratifiedCBNRObjectiveV1:
    total: torch.Tensor
    retrieval_no_regret: torch.Tensor
    binding: torch.Tensor
    raw_correct_count: int
    raw_wrong_count: int


def stratified_cbnr_objective(
    losses: Sequence[CBNRLossV1],
) -> StratifiedCBNRObjectiveV1:
    """Equal-weight RAW-correct/wrong retrieval strata plus unit binding mean."""

    values = tuple(losses)
    _require(bool(values), "empty CBNR batch")
    correct = tuple(item for item in values if bool(item.raw_margin > 0))
    wrong = tuple(item for item in values if not bool(item.raw_margin > 0))
    _require(bool(correct) and bool(wrong), "CBNR batch must contain both RAW strata")
    device, dtype = values[0].total.device, values[0].total.dtype
    _require(
        all(item.total.device == device and item.total.dtype == dtype for item in values),
        "CBNR batch dtype/device drift",
    )
    correct_ret = torch.stack(tuple(item.rank + item.keep for item in correct)).mean()
    wrong_ret = torch.stack(tuple(item.rank + item.keep for item in wrong)).mean()
    retrieval = 0.5 * (correct_ret + wrong_ret)
    binding = torch.stack(tuple(item.binding for item in values)).mean()
    total = retrieval + binding
    return StratifiedCBNRObjectiveV1(
        total=total,
        retrieval_no_regret=retrieval,
        binding=binding,
        raw_correct_count=len(correct),
        raw_wrong_count=len(wrong),
    )
def cbnr_loss(
    *,
    raw_target_margin: torch.Tensor,
    real_left_evidence: torch.Tensor,
    real_right_evidence: torch.Tensor,
    binding_left_evidence: torch.Tensor,
    binding_right_evidence: torch.Tensor,
    target_member_ordinal: int,
) -> CBNRLossV1:
    """Compute the single frozen CBNR loss; RAW is utility-only supervision."""

    raw = _scalar(raw_target_margin, "RAW target margin")
    _require(not raw.requires_grad, "RAW margin must not enter the trainable graph")
    real = pair_centered_residual(real_left_evidence, real_right_evidence)
    control = pair_centered_residual(binding_left_evidence, binding_right_evidence)
    real_margin = target_oriented_margin(real, target_member_ordinal=target_member_ordinal)
    control_margin = target_oriented_margin(control, target_member_ordinal=target_member_ordinal)
    _require(
        raw.device == real_margin.device == control_margin.device
        and raw.dtype == real_margin.dtype == control_margin.dtype,
        "RAW/representation dtype-device drift",
    )
    final_margin = raw + real_margin
    rank = F.softplus(-final_margin)
    keep = torch.where(
        raw > 0,
        torch.as_tensor(BREAK_WEIGHT, dtype=raw.dtype, device=raw.device)
        * F.relu(raw - final_margin),
        torch.zeros_like(raw),
    )
    binding = torch.as_tensor(
        BINDING_WEIGHT, dtype=raw.dtype, device=raw.device
    ) * F.softplus(control_margin - real_margin)
    total = rank + keep + binding
    return CBNRLossV1(
        total=total,
        rank=rank,
        keep=keep,
        binding=binding,
        raw_margin=raw,
        real_residual_margin=real_margin,
        binding_control_residual_margin=control_margin,
        final_margin=final_margin,
    )


__all__ = [
    "BINDING_WEIGHT",
    "BREAK_WEIGHT",
    "CBNRError",
    "CBNRLossV1",
    "PairCenteredResidualV1",
    "StratifiedCBNRObjectiveV1",
    "cbnr_loss",
    "pair_centered_residual",
    "stratified_cbnr_objective",
    "target_oriented_margin",
]
