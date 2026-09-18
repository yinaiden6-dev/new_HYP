"""D_FULL_MASS_FREE_IDENTITY: retain full visibility mass, free identity matching.

This is a new zero-parameter scorer, not a bit-equivalent replacement for old C.
The canonical numerical definition is the cached scalar bridge rho=M_C/M_B,
then D.score=B.score*rho for each of the three score fields and D.mass=M_C.
The raw helper first produces original B_QUERY fields and invokes that bridge.
It must not independently recompute a rolled C mass or reassociate operations.
P retains both complete visibility maps. Reference visibility enters only via
its full-grid mass; it never multiplies the identity similarities a second time.

Finite-support implication: if fixed old local values b_target[i] < b_wrong[i]
at every query cell, then on any nonempty common support, any identical
nonnegative weighting with positive total weight preserves that strict order.
This says nothing about candidate-specific query weights/masses, different
supports, or this new scorer's outcomes. No natural HYP GO follows from it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch.nn import functional as F

from .reference_visibility_pv_lossless_v1 import (
    FEATURE_DIMENSION, VisibilityProposal, tensor_sha256, validate_visibility_proposal,
)

SCORER = "D_FULL_MASS_FREE_IDENTITY"


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class FullMassFreeIdentityEvidence:
    real_score: torch.Tensor
    visibility_mass: torch.Tensor
    query_control_score: torch.Tensor
    reference_control_score: torch.Tensor
    identity_values: torch.Tensor
    query_visibility_mass: torch.Tensor
    mass_ratio: torch.Tensor
    proposal_logical_sha256: str
    scorer: str = SCORER

    def as_old_feature_record(self):
        """Same four-field interface, with explicitly changed score semantics."""
        return {field: float(getattr(self, field).detach()) for field in (
            "real_score", "visibility_mass", "query_control_score", "reference_control_score")}


def _scalar(value, name):
    if isinstance(value, torch.Tensor):
        _need(value.dtype == torch.float64 and value.device.type == "cpu" and value.shape == (), name + "_NOT_FP64_CPU_SCALAR")
        result = value.detach()
    else:
        _need(isinstance(value, (float, int)) and not isinstance(value, bool), name + "_NOT_NUMERIC_SCALAR")
        result = torch.tensor(value, dtype=torch.float64)
    _need(bool(torch.isfinite(result)), name + "_NONFINITE")
    return result


def combine_full_mass_with_free_identity(b_record: Mapping[str, object], c_mass) -> dict[str, float]:
    """Canonical cached-only D bridge; passed Q/R fields retain their provenance.

When M_B is zero, only the consistent all-zero B scores and M_C=0 case is
accepted, returning zero evidence. No positive floor or tunable threshold is
introduced. The R score is multiplied as supplied, never overwritten by REAL.
"""
    fields = ("real_score", "visibility_mass", "query_control_score", "reference_control_score")
    _need(isinstance(b_record, Mapping) and set(fields).issubset(b_record), "B_RECORD_FIELDS_MISSING")
    b = {field: _scalar(b_record[field], "B_" + field) for field in fields}
    mc = _scalar(c_mass, "C_VISIBILITY_MASS")
    mb = b["visibility_mass"]
    _need(bool(mb >= 0) and bool(mc >= 0), "NEGATIVE_VISIBILITY_MASS")
    if bool(mb == 0):
        _need(bool(mc == 0) and all(bool(b[field] == 0) for field in fields if field != "visibility_mass"),
              "ZERO_B_MASS_INCONSISTENT_WITH_C_MASS_OR_B_SCORES")
        return {"real_score": 0.0, "visibility_mass": float(mc),
                "query_control_score": 0.0, "reference_control_score": 0.0}
    ratio = mc / mb
    _need(bool(torch.isfinite(ratio)), "NONFINITE_MASS_RATIO")
    output = {field: float(b[field] * ratio) for field in fields if field != "visibility_mass"}
    output["visibility_mass"] = float(mc)
    _need(all(math.isfinite(value) for value in output.values()), "NONFINITE_COMBINED_EVIDENCE")
    return output


def verify_visibility(proposal: VisibilityProposal, query_tokens: torch.Tensor,
                      reference_tokens: torch.Tensor, *,
                      expected_source_binding: Mapping[str, str] | None = None) -> FullMassFreeIdentityEvidence:
    proposal = validate_visibility_proposal(proposal)
    if expected_source_binding is not None:
        _need(isinstance(expected_source_binding, Mapping)
              and dict(expected_source_binding) == dict(proposal.source_binding), "EXPECTED_SOURCE_BINDING_MISMATCH")
    for tokens, grid, key in ((query_tokens, proposal.query_grid_shape, "query_tokens_sha256"),
                              (reference_tokens, proposal.reference_grid_shape, "reference_tokens_sha256")):
        _need(isinstance(tokens, torch.Tensor) and tokens.device.type == "cpu" and tokens.is_floating_point()
              and tokens.shape == (math.prod(grid), FEATURE_DIMENSION) and bool(torch.isfinite(tokens).all()), "RAW_TOKEN_FULL_GRID_INVALID")
        _need(tensor_sha256(tokens) == proposal.source_binding[key], "RAW_TOKEN_SOURCE_BINDING_MISMATCH:" + key)
    qn = F.normalize(query_tokens.to(torch.float64), dim=1)
    rn = F.normalize(reference_tokens.to(torch.float64), dim=1)
    similarity = qn @ rn.T
    wq, wr = proposal.query_visibility, proposal.reference_visibility
    ones_r = torch.ones_like(wr, dtype=torch.float64)
    # This is original B_QUERY, including multiplication by its all-one
    # reference weights and its own mass before the canonical scalar bridge.
    identity = (similarity * ones_r[None]).max(dim=1).values
    mass = proposal.visibility_mass
    product_sum = (wq * identity).sum()
    denominator = wq.sum().clamp_min(1e-12)
    bmass = torch.sqrt(wq.mean() * ones_r.mean())
    b_real = (bmass * product_sum) / denominator
    qroll = wq.roll(max(1, wq.numel() // 2))
    bqmass = torch.sqrt(qroll.mean() * ones_r.mean())
    b_query = (bqmass * (qroll * identity).sum()) / qroll.sum().clamp_min(1e-12)
    b_record = {"real_score": b_real, "visibility_mass": bmass,
                "query_control_score": b_query, "reference_control_score": b_real}
    combined = combine_full_mass_with_free_identity(b_record, mass)
    ratio = mass / bmass if bool(bmass > 0) else torch.zeros((), dtype=torch.float64)
    _need(bool(torch.isfinite(identity).all()), "NONFINITE_IDENTITY_EVIDENCE")
    return FullMassFreeIdentityEvidence(*(torch.tensor(combined[field], dtype=torch.float64) for field in
        ("real_score", "visibility_mass", "query_control_score", "reference_control_score")),
        identity, bmass, ratio, proposal.logical_sha256)


score_full_mass_free_identity = verify_visibility
