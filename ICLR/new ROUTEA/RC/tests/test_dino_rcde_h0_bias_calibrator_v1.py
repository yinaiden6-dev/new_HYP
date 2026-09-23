from __future__ import annotations

import math

import pytest
import torch

from rc_aslo_xf.dino_rcde_h0_bias_calibrator_v1 import (
    BiasCalibratorError,
    apply_bias_calibrator,
    fit_bias_calibrator,
    weighted_log_loss,
)


def population() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.tensor([-1.7, -0.4, 0.2, 0.8, 1.9, -0.2], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    weights = torch.tensor([1, 2, 1, 3, 2, 1], dtype=torch.float32)
    return logits, labels, weights


def test_float64_strictly_convex_newton_fit_is_deterministic() -> None:
    logits, labels, weights = population()
    first = fit_bias_calibrator(logits, labels, weights)
    second = fit_bias_calibrator(logits.clone(), labels.clone(), 7.0 * weights)
    assert first == second
    assert first.converged and 0 < first.iterations <= 64
    assert first.hessian > 0.0
    assert abs(first.gradient) <= 1e-12
    assert first.final_loss < first.initial_loss
    assert weighted_log_loss(logits, labels, weights, bias=first.bias).dtype == torch.float64
    assert weighted_log_loss(logits, labels, weights, bias=first.bias).item() == first.final_loss


def test_zero_logit_solution_is_weighted_prevalence_log_odds() -> None:
    logits = torch.zeros(5)
    labels = torch.tensor([1, 1, 0, 0, 0])
    weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0])
    fit = fit_bias_calibrator(logits, labels, weights)
    assert fit.converged
    assert fit.bias == pytest.approx(math.log(3.0 / 3.0), abs=1e-12)


def test_ready_adds_only_shared_bias_and_preserves_rank_and_margin() -> None:
    raw = torch.tensor([-0.7, 0.1, 1.4, -0.2], dtype=torch.float64)
    bias = 0.625
    calibrated = apply_bias_calibrator(raw, structural_ready=True, bias=bias)
    assert torch.equal(calibrated, raw + bias)
    assert torch.equal(torch.argsort(calibrated), torch.argsort(raw))
    raw_margins = raw[:, None] - raw[None, :]
    calibrated_margins = calibrated[:, None] - calibrated[None, :]
    assert torch.allclose(calibrated_margins, raw_margins, atol=2e-16, rtol=0.0)


def test_structural_h0_bypasses_bias_as_exact_positive_zero() -> None:
    output = apply_bias_calibrator(
        torch.tensor([-9.0, 2.0], dtype=torch.float64),
        structural_ready=False,
        bias=123.0,
    )
    assert torch.equal(output, torch.zeros(2, dtype=torch.float64))
    assert all(math.copysign(1.0, item) == 1.0 for item in output.tolist())


@pytest.mark.parametrize(
    ("logits", "labels", "weights", "message"),
    [
        ([0.0, 1.0], [1, 1], [1.0, 1.0], "positive and negative"),
        ([0.0, 1.0], [0, 1], [1.0, 0.0], "strictly positive"),
        ([0.0, float("nan")], [0, 1], [1.0, 1.0], "non-finite"),
    ],
)
def test_invalid_or_non_identifiable_population_fails_closed(
    logits: list[float], labels: list[int], weights: list[float], message: str
) -> None:
    with pytest.raises(BiasCalibratorError, match=message):
        fit_bias_calibrator(torch.tensor(logits), torch.tensor(labels), torch.tensor(weights))
