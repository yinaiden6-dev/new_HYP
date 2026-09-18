"""Shared query target prior over the unchanged full-reference C scorer.

The only trainable object is theta[133].  The prior reads the original query
tokens and cell positions, never a candidate.  It multiplies each candidate's
original full-grid query visibility; all means retain their original axes.
This soft field is not asserted to be connected or to prove pixel ownership.

Numerical contract: CPU FP64, one candidate and one 1-D reduction at a time.
At theta=0 the prior is exactly one and the literal old scalar operation order
is replayed.  Q rolls the complete effective query visibility alpha*wq; R
rolls the original reference visibility before full-reference weighted MaxSim.
The separate prior-position control rolls only alpha, then recomputes all four
fields, all six action features, and the frozen action logit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from . import romav2_colnomic_frozen_gate_v1 as frozen_gate

FEATURE_DIMENSION = 128
PRIOR_PARAMETER_COUNT = 133
POSITION_FEATURES = ("x", "y", "x_squared", "y_squared", "x_times_y")
SCORE_FIELDS = ("real_score", "visibility_mass", "query_control_score", "reference_control_score")


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid_shape(value, name):
    _need(isinstance(value, (tuple, list)) and len(value) == 2
          and all(type(x) is int and x > 0 for x in value), name + "_INVALID")
    return tuple(value)


def _axis(value, count, name):
    _need(isinstance(value, torch.Tensor) and value.dtype == torch.int64
          and value.device.type == "cpu" and value.shape == (count,)
          and torch.equal(value, torch.arange(count, dtype=torch.int64)), name + "_NOT_CANONICAL_FULL_AXIS")


def _fp64(value, shape, name, *, frozen=False):
    _need(isinstance(value, torch.Tensor) and value.dtype == torch.float64
          and value.device.type == "cpu" and value.shape == shape
          and bool(torch.isfinite(value.detach()).all()), name + "_NOT_FINITE_FP64_CPU")
    _need(not frozen or not value.requires_grad, name + "_SOURCE_MUST_BE_FROZEN")


def _tokens(value, shape, name):
    _need(isinstance(value, torch.Tensor) and value.is_floating_point()
          and value.device.type == "cpu" and value.shape == (math.prod(shape), FEATURE_DIMENSION)
          and bool(torch.isfinite(value).all()) and not value.requires_grad,
          name + "_NOT_FROZEN_FULL_GRID_TOKENS")


def query_features(query_tokens: torch.Tensor, query_grid_shape, *, query_axis=None) -> torch.Tensor:
    """FP64 normalized original tokens plus row-major cell centers in [-1,1]."""
    shape = _grid_shape(query_grid_shape, "QUERY_GRID")
    _tokens(query_tokens, shape, "QUERY")
    n = math.prod(shape)
    if query_axis is not None:
        _axis(query_axis, n, "QUERY")
    height, width = shape
    index = torch.arange(n, dtype=torch.int64)
    x = (index.remainder(width).to(torch.float64) + .5) * (2. / width) - 1.
    y = (index.div(width, rounding_mode="floor").to(torch.float64) + .5) * (2. / height) - 1.
    positions = torch.stack((x, y, x * x, y * y, x * y), dim=1)
    return torch.cat((F.normalize(query_tokens.to(torch.float64), dim=1), positions), dim=1)


class SharedQueryTargetPrior(nn.Module):
    """No candidate input; one query evaluation supplies every candidate."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(PRIOR_PARAMETER_COUNT, dtype=torch.float64))

    def from_features(self, features: torch.Tensor) -> torch.Tensor:
        _need(isinstance(features, torch.Tensor) and features.ndim == 2 and features.shape[0] > 0,
              "QUERY_FEATURE_AXIS_INVALID")
        _fp64(features, (features.shape[0], PRIOR_PARAMETER_COUNT), "QUERY_FEATURES", frozen=True)
        _fp64(self.theta, (PRIOR_PARAMETER_COUNT,), "THETA")
        scores = features @ self.theta
        _need(bool(torch.isfinite(scores.detach()).all()), "NONFINITE_PRIOR_SCORES")
        return torch.exp(scores - scores.max())

    def forward(self, query_tokens: torch.Tensor, query_grid_shape, *, query_axis=None) -> torch.Tensor:
        return self.from_features(query_features(query_tokens, query_grid_shape, query_axis=query_axis))


@dataclass(frozen=True)
class CandidateEvidenceSource:
    """Frozen, full-query-axis MaxSim caches, independently source-bound by caller.

    reference_control_mean must be computed from the actually rolled wr: even
    a permutation's FP64 mean can differ in its last bit.  Cached local arrays
    are full-reference weighted maxima, not hard correspondence similarities.
    """
    local_values: torch.Tensor
    reference_control_local_values: torch.Tensor
    query_visibility: torch.Tensor
    reference_mean: torch.Tensor
    reference_control_mean: torch.Tensor
    query_axis: torch.Tensor
    query_control_shift: int


def validate_candidate_source(source: CandidateEvidenceSource, query_count: int) -> CandidateEvidenceSource:
    _need(isinstance(source, CandidateEvidenceSource), "CANDIDATE_SOURCE_TYPE_INVALID")
    _need(type(query_count) is int and query_count > 0, "QUERY_COUNT_INVALID")
    _axis(source.query_axis, query_count, "SOURCE_QUERY")
    for name in ("local_values", "reference_control_local_values", "query_visibility"):
        _fp64(getattr(source, name), (query_count,), name.upper(), frozen=True)
    for name in ("reference_mean", "reference_control_mean"):
        value = getattr(source, name)
        _fp64(value, (), name.upper(), frozen=True)
        _need(bool((value >= 0.) & (value <= 1.)), name.upper() + "_OUTSIDE_UNIT_INTERVAL")
    _need(bool(((source.query_visibility >= 0.) & (source.query_visibility <= 1.)).all()),
          "QUERY_VISIBILITY_OUTSIDE_UNIT_INTERVAL")
    _need(type(source.query_control_shift) is int, "QUERY_CONTROL_SHIFT_NOT_INTEGER")
    return source


def precompute_candidate_evidence(query_tokens: torch.Tensor, reference_tokens: torch.Tensor,
                                 query_visibility: torch.Tensor, reference_visibility: torch.Tensor,
                                 query_grid_shape, reference_grid_shape, *,
                                 query_control_shift=None, reference_control_shift=None,
                                 query_axis=None, reference_axis=None) -> CandidateEvidenceSource:
    """Compute caches from original full arrays; this function performs no I/O."""
    qshape = _grid_shape(query_grid_shape, "QUERY_GRID")
    rshape = _grid_shape(reference_grid_shape, "REFERENCE_GRID")
    _tokens(query_tokens, qshape, "QUERY")
    _tokens(reference_tokens, rshape, "REFERENCE")
    nq, nr = math.prod(qshape), math.prod(rshape)
    qa = torch.arange(nq, dtype=torch.int64) if query_axis is None else query_axis
    ra = torch.arange(nr, dtype=torch.int64) if reference_axis is None else reference_axis
    _axis(qa, nq, "QUERY")
    _axis(ra, nr, "REFERENCE")
    for value, count, name in ((query_visibility, nq, "QUERY_VISIBILITY"),
                               (reference_visibility, nr, "REFERENCE_VISIBILITY")):
        _fp64(value, (count,), name, frozen=True)
        _need(bool(((value >= 0.) & (value <= 1.)).all()), name + "_OUTSIDE_UNIT_INTERVAL")
    qs = max(1, nq // 2) if query_control_shift is None else query_control_shift
    rs = max(1, nr // 2) if reference_control_shift is None else reference_control_shift
    _need(type(qs) is int and type(rs) is int, "CONTROL_SHIFT_NOT_INTEGER")
    similarity = F.normalize(query_tokens.to(torch.float64), dim=1) @ F.normalize(reference_tokens.to(torch.float64), dim=1).T
    rolled_reference = reference_visibility.roll(rs)
    result = CandidateEvidenceSource(
        (similarity * reference_visibility[None]).max(1).values,
        (similarity * rolled_reference[None]).max(1).values,
        query_visibility.detach().clone(), reference_visibility.mean(), rolled_reference.mean(),
        qa.detach().clone(), qs)
    return validate_candidate_source(result, nq)


@dataclass(frozen=True)
class CandidateScores:
    real_score: torch.Tensor
    visibility_mass: torch.Tensor
    query_control_score: torch.Tensor
    reference_control_score: torch.Tensor

    def as_old_feature_record(self) -> dict[str, float]:
        return {name: float(getattr(self, name).detach()) for name in SCORE_FIELDS}


def _literal_score(weight, local, reference_mean):
    product = weight.mean() * reference_mean
    # sqrt(0) has an infinite derivative.  For frozen zero visibility its
    # derivative with respect to theta is zero; keep the literal zero value
    # while avoiding the undefined 0*inf autograd chain. No positive floor.
    mass = product * 0. if bool(product.detach() == 0.) else torch.sqrt(product)
    return mass * (weight * local).sum() / weight.sum().clamp_min(1e-12), mass


def score_candidate(alpha: torch.Tensor, source: CandidateEvidenceSource) -> CandidateScores:
    _need(isinstance(alpha, torch.Tensor) and alpha.ndim == 1 and alpha.numel() > 0,
          "PRIOR_QUERY_AXIS_INVALID")
    _fp64(alpha, (alpha.numel(),), "ALPHA")
    _need(bool(((alpha.detach() >= 0.) & (alpha.detach() <= 1.)).all()), "ALPHA_OUTSIDE_UNIT_INTERVAL")
    source = validate_candidate_source(source, alpha.numel())
    effective_query = alpha * source.query_visibility
    real, mass = _literal_score(effective_query, source.local_values, source.reference_mean)
    query, _ = _literal_score(effective_query.roll(source.query_control_shift), source.local_values, source.reference_mean)
    reference, _ = _literal_score(effective_query, source.reference_control_local_values, source.reference_control_mean)
    return CandidateScores(real, mass, query, reference)


@dataclass(frozen=True)
class QueryScores:
    alpha: torch.Tensor
    records: Mapping[Hashable, CandidateScores]


def score_query(prior: SharedQueryTargetPrior, query_tokens: torch.Tensor, query_grid_shape,
                sources: Mapping[Hashable, CandidateEvidenceSource], *,
                prior_position_control: bool = False, prior_control_shift=None) -> QueryScores:
    _need(isinstance(sources, Mapping) and len(sources) > 0, "EMPTY_OR_INVALID_CANDIDATE_MAPPING")
    _need(isinstance(prior, SharedQueryTargetPrior), "PRIOR_TYPE_INVALID")
    _need(type(prior_position_control) is bool, "PRIOR_POSITION_CONTROL_NOT_BOOLEAN")
    _need(prior_position_control or prior_control_shift is None, "CONTROL_SHIFT_WITHOUT_CONTROL")
    alpha = prior(query_tokens, query_grid_shape)
    if prior_position_control:
        shift = max(1, alpha.numel() // 2) if prior_control_shift is None else prior_control_shift
        _need(type(shift) is int, "PRIOR_CONTROL_SHIFT_NOT_INTEGER")
        alpha = alpha.roll(shift)
    # Deliberately no candidate-axis batched sum: preserve old 1-D reductions.
    return QueryScores(alpha, {key: score_candidate(alpha, source) for key, source in sources.items()})


def standardized_raw_gap(raw: Sequence[float], challenger: int, winner: int) -> torch.Tensor:
    """Use the original public implementation for its Python scalar raw std."""
    _need(len(raw) > 0 and type(challenger) is int and type(winner) is int
          and 0 <= challenger < len(raw) and 0 <= winner < len(raw), "RAW_CANDIDATE_AXIS_INVALID")
    _need(all(math.isfinite(float(value)) for value in raw), "RAW_NONFINITE")
    zeros = {name: 0. for name in SCORE_FIELDS}
    return frozen_gate.candidate_feature(raw, {challenger: zeros, winner: zeros}, challenger, winner)[0]


def frozen_action_features(raw: Sequence[float], records: Mapping[int, CandidateScores],
                           challenger: int, winner: int, *, fixed_raw_gap=None) -> torch.Tensor:
    """Original six features with gradients through the four new C scalars."""
    gap = standardized_raw_gap(raw, challenger, winner) if fixed_raw_gap is None else fixed_raw_gap
    _fp64(gap, (), "FIXED_RAW_GAP", frozen=True)
    lc, lw = records[challenger], records[winner]
    _need(isinstance(lc, CandidateScores) and isinstance(lw, CandidateScores), "ACTION_RECORD_TYPE_INVALID")
    for record in (lc, lw):
        for field in SCORE_FIELDS:
            _fp64(getattr(record, field), (), field.upper())
    def symmetric(a, b):
        return (a - b) / (a.abs() + b.abs() + 1e-12)
    sc = lc.real_score / lc.visibility_mass.clamp_min(1e-12)
    sw = lw.real_score / lw.visibility_mass.clamp_min(1e-12)
    qc, qw = lc.real_score - lc.query_control_score, lw.real_score - lw.query_control_score
    rc, rw = lc.real_score - lc.reference_control_score, lw.real_score - lw.reference_control_score
    return torch.stack((gap, symmetric(lc.real_score, lw.real_score),
        symmetric(lc.visibility_mass, lw.visibility_mass), symmetric(sc, sw),
        symmetric(qc, qw), symmetric(rc, rw)))


def frozen_action_logit(raw: Sequence[float], records: Mapping[int, CandidateScores],
                        challenger: int, winner: int, *, fixed_raw_gap=None, head=None) -> torch.Tensor:
    features = frozen_action_features(raw, records, challenger, winner, fixed_raw_gap=fixed_raw_gap)
    return (FixedActionHead() if head is None else head)(features)


class FixedActionHead(nn.Module):
    """Externally chosen six-weight/bias head, stored only as frozen buffers.

    The default is the historical FROZEN_C head.  To reproduce a separately
    sealed NATIVE7 bundle, pass that bundle's exact weight and bias values;
    their provenance and qualification are the caller's responsibility.
    """

    def __init__(self, weight=None, bias=None):
        super().__init__()
        w = frozen_gate.WEIGHT if weight is None else weight
        b = frozen_gate.BIAS if bias is None else bias
        _fp64(w, (6,), "FIXED_HEAD_WEIGHT", frozen=True)
        if not isinstance(b, torch.Tensor):
            _need(isinstance(b, (float, int)) and not isinstance(b, bool), "FIXED_HEAD_BIAS_INVALID")
            b = torch.tensor(b, dtype=torch.float64)
        _fp64(b, (), "FIXED_HEAD_BIAS", frozen=True)
        self.register_buffer("weight", w.detach().clone())
        self.register_buffer("bias", b.detach().clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _fp64(self.weight, (6,), "FIXED_HEAD_WEIGHT", frozen=True)
        _fp64(self.bias, (), "FIXED_HEAD_BIAS", frozen=True)
        _need(isinstance(features, torch.Tensor) and features.ndim in (1, 2)
              and features.shape[-1] == 6, "ACTION_FEATURE_AXES_INVALID")
        _fp64(features, tuple(features.shape), "ACTION_FEATURES")
        return (self.weight * features).sum(dim=-1) + self.bias


@dataclass(frozen=True)
class BatchedEvidenceSource:
    candidate_keys: tuple[Hashable, ...]
    local_values: torch.Tensor
    reference_control_local_values: torch.Tensor
    query_visibility: torch.Tensor
    reference_mean: torch.Tensor
    reference_control_mean: torch.Tensor
    query_axis: torch.Tensor
    query_control_shift: int


def validate_batched_source(source: BatchedEvidenceSource) -> BatchedEvidenceSource:
    _need(isinstance(source, BatchedEvidenceSource) and isinstance(source.candidate_keys, tuple)
          and len(source.candidate_keys) > 0 and len(set(source.candidate_keys)) == len(source.candidate_keys),
          "BATCH_CANDIDATE_KEYS_INVALID")
    _need(isinstance(source.query_visibility, torch.Tensor) and source.query_visibility.ndim == 2,
          "BATCH_SOURCE_AXES_INVALID")
    nc, nq = source.query_visibility.shape
    _need(nc == len(source.candidate_keys) and nq > 0, "BATCH_SOURCE_AXIS_SIZE_MISMATCH")
    _axis(source.query_axis, nq, "BATCH_QUERY")
    for name in ("local_values", "reference_control_local_values", "query_visibility"):
        _fp64(getattr(source, name), (nc, nq), "BATCH_" + name.upper(), frozen=True)
    for name in ("reference_mean", "reference_control_mean"):
        value = getattr(source, name)
        _fp64(value, (nc,), "BATCH_" + name.upper(), frozen=True)
        _need(bool(((value >= 0.) & (value <= 1.)).all()), name.upper() + "_OUTSIDE_UNIT_INTERVAL")
    _need(bool(((source.query_visibility >= 0.) & (source.query_visibility <= 1.)).all()),
          "BATCH_QUERY_VISIBILITY_OUTSIDE_UNIT_INTERVAL")
    _need(type(source.query_control_shift) is int, "BATCH_QUERY_CONTROL_SHIFT_NOT_INTEGER")
    return source


def pack_candidate_sources(sources: Mapping[Hashable, CandidateEvidenceSource]) -> BatchedEvidenceSource:
    """Pack once before optimization; keeps explicit caller candidate order."""
    _need(isinstance(sources, Mapping) and len(sources) > 0, "EMPTY_OR_INVALID_CANDIDATE_MAPPING")
    values = list(sources.values())
    _need(isinstance(values[0], CandidateEvidenceSource) and values[0].query_visibility.ndim == 1,
          "BATCH_SOURCE_TYPE_INVALID")
    nq = values[0].query_visibility.numel()
    for value in values:
        validate_candidate_source(value, nq)
    shifts = {value.query_control_shift for value in values}
    _need(len(shifts) == 1, "BATCH_QUERY_CONTROL_SHIFTS_DISAGREE")
    stack = lambda name: torch.stack([getattr(value, name) for value in values])
    return validate_batched_source(BatchedEvidenceSource(tuple(sources),
        stack("local_values"), stack("reference_control_local_values"), stack("query_visibility"),
        stack("reference_mean"), stack("reference_control_mean"), values[0].query_axis.clone(), shifts.pop()))


@dataclass(frozen=True)
class BatchedCandidateScores:
    real_score: torch.Tensor
    visibility_mass: torch.Tensor
    query_control_score: torch.Tensor
    reference_control_score: torch.Tensor

    def as_records(self, candidate_keys: Sequence[Hashable]) -> dict[Hashable, CandidateScores]:
        _need(len(candidate_keys) == self.real_score.numel(), "OUTPUT_CANDIDATE_AXIS_MISMATCH")
        return {key: CandidateScores(*(getattr(self, field)[index] for field in SCORE_FIELDS))
                for index, key in enumerate(candidate_keys)}


def _batched_literal_score(weight, local, reference_mean):
    product = weight.mean(dim=1) * reference_mean
    positive = product > 0.
    safe_product = torch.where(positive, product, torch.ones_like(product))
    mass = torch.where(positive, torch.sqrt(safe_product), product * 0.)
    return mass * (weight * local).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-12), mass


def score_candidates_batched(alpha: torch.Tensor, source: BatchedEvidenceSource, *,
                             validate_source: bool = True) -> BatchedCandidateScores:
    """FP64 vectorized training path, not claimed to be bit-identical to 1-D.

    PyTorch may use different reductions for CxQ versus Q arrays.  The literal
    score_candidate path remains the independent numerical oracle.  A caller
    must measure any natural parity differences rather than substitute cached
    baseline constants.  No reduction result is copied from the old scorer.
    """
    if validate_source:
        validate_batched_source(source)
    _need(isinstance(source, BatchedEvidenceSource), "BATCH_SOURCE_TYPE_INVALID")
    _fp64(alpha, (source.query_visibility.shape[1],), "ALPHA")
    _need(bool(((alpha.detach() >= 0.) & (alpha.detach() <= 1.)).all()), "ALPHA_OUTSIDE_UNIT_INTERVAL")
    effective = source.query_visibility * alpha[None, :]
    real, mass = _batched_literal_score(effective, source.local_values, source.reference_mean)
    query, _ = _batched_literal_score(effective.roll(source.query_control_shift, dims=1),
                                      source.local_values, source.reference_mean)
    reference, _ = _batched_literal_score(effective, source.reference_control_local_values, source.reference_control_mean)
    return BatchedCandidateScores(real, mass, query, reference)


def frozen_action_features_batched(raw_gaps: torch.Tensor, scores: BatchedCandidateScores,
                                   winner: int) -> torch.Tensor:
    """Candidate rows keep packed-source order; raw_gaps use that same order."""
    _need(isinstance(scores, BatchedCandidateScores) and scores.real_score.ndim == 1,
          "BATCH_SCORES_INVALID")
    count = scores.real_score.numel()
    _need(type(winner) is int and 0 <= winner < count, "BATCH_WINNER_OUTSIDE_CANDIDATE_AXIS")
    _fp64(raw_gaps, (count,), "RAW_GAPS", frozen=True)
    for field in SCORE_FIELDS:
        _fp64(getattr(scores, field), (count,), "BATCH_" + field.upper())
    def symmetric(value):
        return (value - value[winner]) / (value.abs() + value[winner].abs() + 1e-12)
    normalized = scores.real_score / scores.visibility_mass.clamp_min(1e-12)
    query = scores.real_score - scores.query_control_score
    reference = scores.real_score - scores.reference_control_score
    return torch.stack((raw_gaps, symmetric(scores.real_score), symmetric(scores.visibility_mass),
                        symmetric(normalized), symmetric(query), symmetric(reference)), dim=1)


def frozen_action_logits_batched(raw_gaps: torch.Tensor, scores: BatchedCandidateScores,
                                 winner: int, head: FixedActionHead) -> torch.Tensor:
    _need(isinstance(head, FixedActionHead), "FROZEN_HEAD_TYPE_INVALID")
    return head(frozen_action_features_batched(raw_gaps, scores, winner))
