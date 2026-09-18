"""Fold-local joint match/no-match density-ratio verifier.

The verifier is deliberately small and identity-free.  It is fitted only on a
training-fold prejoin ledger; labels select loss rows but are never accepted by
``forward``.  Its output is an estimated log density ratio, not a posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_DIM = 4  # appearance, geometry, edge/glyph-visual, interaction
STEPS = 256
LEARNING_RATE = 3.0e-3


class JointDensityRatio(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(FEATURE_DIM, 1, bias=True, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != FEATURE_DIM or not bool(torch.isfinite(features).all()):
            raise ValueError("joint verifier requires finite [...,4] prejoin features")
        return self.linear(features).squeeze(-1)


@dataclass(frozen=True)
class VerifierFit:
    state_dict: dict[str, torch.Tensor]
    sampling_log_odds: float
    initial_nll: float
    final_nll: float
    parameter_count: int


def fit_training_fold(features: torch.Tensor, labels: torch.Tensor, *, seed: int) -> VerifierFit:
    """Fit one registered fold-local likelihood-ratio calibrator.

    ``labels`` may only enter this routine's logistic loss.  The inverse
    empirical sampling-prior correction is applied to the output logit so that
    a balanced training batch does not silently become the deployment prior.
    """

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or labels.shape != (features.shape[0],):
        raise ValueError("verifier training feature/label shape drift")
    if labels.dtype != torch.bool or features.shape[0] < 2 or not bool(labels.any()) or not bool((~labels).any()):
        raise ValueError("verifier training requires both match and no-match rows")
    torch.manual_seed(seed)
    model = JointDensityRatio().to(features.device)
    target = labels.to(dtype=torch.float64)
    prior = float(target.mean())
    correction = math.log(prior / (1.0 - prior))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    with torch.no_grad():
        initial = float(F.binary_cross_entropy_with_logits(model(features), target))
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(features), target)
        loss.backward()
        if not all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise RuntimeError("non-finite or inactive joint-verifier gradient")
        optimizer.step()
    with torch.no_grad():
        final = float(F.binary_cross_entropy_with_logits(model(features), target))
    return VerifierFit(
        state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        sampling_log_odds=correction,
        initial_nll=initial,
        final_nll=final,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def estimated_log_ratio(model: JointDensityRatio, features: torch.Tensor, *, sampling_log_odds: float) -> torch.Tensor:
    if not math.isfinite(sampling_log_odds):
        raise ValueError("sampling prior correction must be finite")
    return model(features) - sampling_log_odds
