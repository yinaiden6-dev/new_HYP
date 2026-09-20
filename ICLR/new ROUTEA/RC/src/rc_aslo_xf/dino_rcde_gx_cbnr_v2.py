"""Matched-null calibrated connected DINO relational-energy primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F


NULL_COUNT = 4
NULL_NAMES = (
    "C_BIND",
    "P_COORD",
    "T_SHAPE_MATCHED_ROOT_ASSIGNMENT",
    "N_REGION_RESAMPLE",
)


class GXCBNRError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise GXCBNRError(message)


def scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    require(
        isinstance(value, torch.Tensor)
        and value.ndim == 0
        and value.is_floating_point()
        and bool(torch.isfinite(value)),
        f"{name} must be a finite floating scalar",
    )
    return value


@dataclass(frozen=True)
class GeometricExclusivityV1:
    real: torch.Tensor
    null_logmeanexp: torch.Tensor
    z: torch.Tensor
    nulls: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    def __post_init__(self) -> None:
        values = (self.real, self.null_logmeanexp, self.z, *self.nulls)
        require(
            all(scalar(value, "geometric exclusivity field") is value for value in values),
            "geometric exclusivity scalar drift",
        )
        require(
            all(value.device == self.real.device and value.dtype == self.real.dtype for value in values),
            "geometric exclusivity dtype/device drift",
        )
        require(torch.equal(self.z, self.real - self.null_logmeanexp), "Z reconstruction drift")


def geometric_exclusivity(
    real: torch.Tensor, nulls: Sequence[torch.Tensor]
) -> GeometricExclusivityV1:
    real_value = scalar(real, "real geometric energy")
    values = tuple(nulls)
    require(len(values) == NULL_COUNT, "GX-CBNR requires exactly four nulls")
    require(
        all(
            scalar(value, "null geometric energy") is value
            and value.device == real_value.device
            and value.dtype == real_value.dtype
            for value in values
        ),
        "null energy dtype/device drift",
    )
    stacked = torch.stack(values)
    null_lme = torch.logsumexp(stacked, dim=0) - math.log(NULL_COUNT)
    return GeometricExclusivityV1(
        real=real_value,
        null_logmeanexp=null_lme,
        z=real_value - null_lme,
        nulls=values,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class PairCenteredGXResidualV1:
    target: torch.Tensor
    rival: torch.Tensor
    target_minus_rival: torch.Tensor

    def __post_init__(self) -> None:
        require(torch.equal(self.target + self.rival, torch.zeros_like(self.target)), "GX residual not zero-sum")
        require(torch.equal(self.target - self.rival, self.target_minus_rival), "GX residual margin drift")


def pair_centered_gx_residual(
    target_z: torch.Tensor, rival_z: torch.Tensor
) -> PairCenteredGXResidualV1:
    margin = scalar(target_z, "target Z") - scalar(rival_z, "rival Z")
    target = 0.5 * margin
    return PairCenteredGXResidualV1(target, -target, margin)


@dataclass(frozen=True)
class GXCBNRLossV1:
    total: torch.Tensor
    rank: torch.Tensor
    keep: torch.Tensor
    nce: torch.Tensor
    raw_margin: torch.Tensor
    geometric_margin: torch.Tensor
    final_margin: torch.Tensor
    target_z: torch.Tensor
    rival_z: torch.Tensor


@dataclass(frozen=True)
class GXBranchCoefficientsV1:
    target_real: torch.Tensor
    target_nulls: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    rival_real: torch.Tensor
    rival_nulls: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    detached_loss: GXCBNRLossV1


def gx_cbnr_loss(
    *,
    raw_target_margin: torch.Tensor,
    target: GeometricExclusivityV1,
    rival: GeometricExclusivityV1,
) -> GXCBNRLossV1:
    raw = scalar(raw_target_margin, "RAW target margin")
    require(not raw.requires_grad, "RAW margin entered the trainable graph")
    require(
        raw.device == target.z.device == rival.z.device
        and raw.dtype == target.z.dtype == rival.z.dtype,
        "RAW/Z dtype-device drift",
    )
    geometric_margin = target.z - rival.z
    final_margin = raw + geometric_margin
    rank = F.softplus(-final_margin)
    keep = torch.where(raw > 0, F.relu(raw - final_margin), torch.zeros_like(raw))
    nce = F.softplus(-target.z)
    total = rank + keep + nce
    require(torch.equal(total, rank + keep + nce), "GX-CBNR loss reconstruction drift")
    return GXCBNRLossV1(
        total=total,
        rank=rank,
        keep=keep,
        nce=nce,
        raw_margin=raw,
        geometric_margin=geometric_margin,
        final_margin=final_margin,
        target_z=target.z,
        rival_z=rival.z,
    )


def gx_branch_coefficients(
    *,
    raw_target_margin: torch.Tensor,
    target_real: torch.Tensor,
    target_nulls: Sequence[torch.Tensor],
    rival_real: torch.Tensor,
    rival_nulls: Sequence[torch.Tensor],
) -> GXBranchCoefficientsV1:
    """Return exact scalar chain-rule coefficients for branchwise replay."""

    raw = scalar(raw_target_margin, "RAW target margin").detach()
    target_values = (target_real, *tuple(target_nulls))
    rival_values = (rival_real, *tuple(rival_nulls))
    require(
        len(target_values) == 5 and len(rival_values) == 5,
        "GX coefficient branch population drift",
    )
    leaves = tuple(
        scalar(value, "GX coefficient energy").detach().clone().requires_grad_(True)
        for value in (*target_values, *rival_values)
    )
    target = geometric_exclusivity(leaves[0], leaves[1:5])
    rival = geometric_exclusivity(leaves[5], leaves[6:10])
    loss = gx_cbnr_loss(raw_target_margin=raw, target=target, rival=rival)
    gradients = torch.autograd.grad(loss.total, leaves, create_graph=False)

    def detached(value: torch.Tensor) -> torch.Tensor:
        return value.detach().clone()

    detached_loss = GXCBNRLossV1(
        total=detached(loss.total),
        rank=detached(loss.rank),
        keep=detached(loss.keep),
        nce=detached(loss.nce),
        raw_margin=detached(loss.raw_margin),
        geometric_margin=detached(loss.geometric_margin),
        final_margin=detached(loss.final_margin),
        target_z=detached(loss.target_z),
        rival_z=detached(loss.rival_z),
    )
    return GXBranchCoefficientsV1(
        target_real=gradients[0].detach(),
        target_nulls=tuple(item.detach() for item in gradients[1:5]),  # type: ignore[arg-type]
        rival_real=gradients[5].detach(),
        rival_nulls=tuple(item.detach() for item in gradients[6:10]),  # type: ignore[arg-type]
        detached_loss=detached_loss,
    )


@dataclass(frozen=True)
class GXActionV1:
    action: str
    final_margin: torch.Tensor
    challenger_z: torch.Tensor
    all_nulls_eligible: bool


def gx_action(
    *,
    raw_challenger_minus_winner: torch.Tensor,
    challenger_z: torch.Tensor,
    winner_z: torch.Tensor,
    all_nulls_eligible: bool,
) -> GXActionV1:
    raw = scalar(raw_challenger_minus_winner, "RAW challenger margin")
    challenger = scalar(challenger_z, "challenger Z")
    winner = scalar(winner_z, "winner Z")
    final = raw + challenger - winner
    switch = all_nulls_eligible and bool(challenger > 0) and bool(final > 0)
    return GXActionV1(
        action="SWITCH" if switch else "RELATIVE_NULL_HOLD",
        final_margin=final,
        challenger_z=challenger,
        all_nulls_eligible=bool(all_nulls_eligible),
    )


__all__ = [
    "GXActionV1",
    "GXCBNRError",
    "GXCBNRLossV1",
    "GXBranchCoefficientsV1",
    "GeometricExclusivityV1",
    "NULL_COUNT",
    "NULL_NAMES",
    "PairCenteredGXResidualV1",
    "geometric_exclusivity",
    "gx_action",
    "gx_branch_coefficients",
    "gx_cbnr_loss",
    "pair_centered_gx_residual",
]
