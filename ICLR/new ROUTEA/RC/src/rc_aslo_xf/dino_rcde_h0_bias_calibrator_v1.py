"""Shared bias-only calibration for absolute H0 logits.

The fitted map is deliberately restricted to ``z -> z + b``.  Consequently it
can move the absolute zero decision boundary without changing any candidate
ranking or pairwise margin.  Structural-H0 rows bypass the fitted bias and
remain exact zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


SCHEMA_VERSION = "rc_dino_rcde_h0_bias_calibrator_core_v1_20260824"


class BiasCalibratorError(ValueError):
    """Raised when the calibration population or arithmetic is unsafe."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise BiasCalibratorError(message)


@dataclass(frozen=True)
class BiasCalibratorFitV1:
    bias: float
    initial_loss: float
    final_loss: float
    gradient: float
    hessian: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        require(
            all(
                math.isfinite(value)
                for value in (
                    self.bias,
                    self.initial_loss,
                    self.final_loss,
                    self.gradient,
                    self.hessian,
                )
            ),
            "bias-calibrator fit contains non-finite diagnostics",
        )
        require(self.hessian > 0.0, "bias-calibrator Hessian is not strictly positive")
        require(self.iterations >= 0, "bias-calibrator iteration count is invalid")
        require(isinstance(self.converged, bool), "bias-calibrator convergence flag drift")


def _population(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # CPU float64 fixes both the arithmetic precision and execution order used
    # by every producer/validator, independently of the model device/dtype.
    z = torch.as_tensor(logits).detach().to(device="cpu", dtype=torch.float64)
    y = torch.as_tensor(labels).detach().to(device="cpu", dtype=torch.float64)
    w = torch.as_tensor(weights).detach().to(device="cpu", dtype=torch.float64)
    require(
        z.ndim == y.ndim == w.ndim == 1 and z.shape == y.shape == w.shape and z.numel() > 1,
        "bias-calibrator population must be non-empty aligned vectors",
    )
    require(
        bool(torch.isfinite(z).all())
        and bool(torch.isfinite(y).all())
        and bool(torch.isfinite(w).all()),
        "bias-calibrator population is non-finite",
    )
    require(bool(((y == 0.0) | (y == 1.0)).all()), "bias-calibrator labels are not binary")
    require(bool((w > 0.0).all()), "bias-calibrator weights must be strictly positive")
    total = w.sum()
    require(bool(torch.isfinite(total)) and float(total) > 0.0, "bias-calibrator weight mass is invalid")
    w = w / total
    positive_mass = (w * y).sum()
    negative_mass = (w * (1.0 - y)).sum()
    require(
        0.0 < float(positive_mass) < 1.0 and 0.0 < float(negative_mass) < 1.0,
        "bias-calibrator requires positive and negative effective mass",
    )
    return z.contiguous(), y.contiguous(), w.contiguous()


def _loss_state(
    z: torch.Tensor, y: torch.Tensor, w: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eta = z + bias
    require(bool(torch.isfinite(eta).all()), "bias-calibrator shifted logits are non-finite")
    signed_eta = torch.where(y == 1.0, -eta, eta)
    loss = (w * torch.nn.functional.softplus(signed_eta)).sum()
    # The split form avoids the cancellation in sigmoid(eta) - 1 for a
    # confident positive.  h > 0 is the float64 witness of strict convexity.
    positive = y == 1.0
    gradient = (w[~positive] * torch.sigmoid(eta[~positive])).sum() - (
        w[positive] * torch.sigmoid(-eta[positive])
    ).sum()
    curvature = torch.sigmoid(eta) * torch.sigmoid(-eta)
    hessian = (w * curvature).sum()
    require(
        bool(torch.isfinite(loss))
        and bool(torch.isfinite(gradient))
        and bool(torch.isfinite(hessian)),
        "bias-calibrator Newton state is non-finite",
    )
    return loss, gradient, hessian


def weighted_log_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    bias: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Return normalized weighted binary log loss in deterministic float64."""

    z, y, w = _population(logits, labels, weights)
    b = torch.as_tensor(bias, dtype=torch.float64, device="cpu")
    require(b.ndim == 0 and bool(torch.isfinite(b)), "bias-calibrator bias is not finite scalar")
    loss, _, _ = _loss_state(z, y, w, b)
    return loss


def fit_bias_calibrator(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_iterations: int = 64,
    tolerance: float = 1e-12,
) -> BiasCalibratorFitV1:
    """Fit the unique unregularized intercept with safeguarded Newton steps.

    All weights are normalized once.  With positive weight on both binary
    classes, the objective is coercive and its scalar Hessian is strictly
    positive at every finite point, hence the minimizer is unique.  A
    deterministic Armijo backtrack prevents an unsafe full Newton step.
    """

    require(
        isinstance(max_iterations, int) and not isinstance(max_iterations, bool) and max_iterations > 0,
        "bias-calibrator max_iterations is invalid",
    )
    require(
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(float(tolerance))
        and float(tolerance) > 0.0,
        "bias-calibrator tolerance is invalid",
    )
    z, y, w = _population(logits, labels, weights)
    bias = torch.zeros((), dtype=torch.float64)
    loss, gradient, hessian = _loss_state(z, y, w, bias)
    require(float(hessian) > 0.0, "bias-calibrator Hessian lost strict convexity")
    initial_loss = float(loss)
    iterations = 0
    converged = abs(float(gradient)) <= float(tolerance)

    while not converged and iterations < max_iterations:
        require(float(hessian) > 0.0, "bias-calibrator Hessian lost strict convexity")
        full_step = -gradient / hessian
        require(bool(torch.isfinite(full_step)), "bias-calibrator Newton step is non-finite")
        directional = gradient * full_step
        require(float(directional) < 0.0, "bias-calibrator Newton direction is not descending")

        scale = 1.0
        accepted = False
        for _ in range(80):
            candidate = bias + scale * full_step
            if bool(torch.isfinite(candidate)):
                candidate_loss, candidate_gradient, candidate_hessian = _loss_state(
                    z, y, w, candidate
                )
                armijo = loss + 1e-4 * scale * directional
                if float(candidate_hessian) > 0.0 and float(candidate_loss) <= float(armijo):
                    accepted = True
                    break
            scale *= 0.5
        require(accepted, "bias-calibrator safe Newton step was not found")
        bias, loss, gradient, hessian = (
            candidate,
            candidate_loss,
            candidate_gradient,
            candidate_hessian,
        )
        iterations += 1
        converged = abs(float(gradient)) <= float(tolerance)

    return BiasCalibratorFitV1(
        bias=float(bias),
        initial_loss=initial_loss,
        final_loss=float(loss),
        gradient=float(gradient),
        hessian=float(hessian),
        iterations=iterations,
        converged=converged,
    )


def apply_bias_calibrator(
    score: torch.Tensor,
    *,
    structural_ready: bool,
    bias: float | torch.Tensor,
) -> torch.Tensor:
    """Apply ``z + b`` to READY scores and preserve structural H0 as +0."""

    require(isinstance(structural_ready, bool), "structural-ready flag drift")
    value = torch.as_tensor(score)
    require(value.is_floating_point() and bool(torch.isfinite(value).all()), "calibration score drift")
    offset = torch.as_tensor(bias, dtype=value.dtype, device=value.device)
    require(offset.ndim == 0 and bool(torch.isfinite(offset)), "calibration bias drift")
    if not structural_ready:
        return torch.zeros_like(value)
    return value + offset


__all__ = [
    "SCHEMA_VERSION",
    "BiasCalibratorError",
    "BiasCalibratorFitV1",
    "weighted_log_loss",
    "fit_bias_calibrator",
    "apply_bias_calibrator",
]
