"""CW0 V2 reference-first connected-region E0 core.

This module implements the target-free mathematical core frozen by
``CW0_REFERENCE_FIRST_CONNECTED_REGION_CONTRACT_V2_20260809.md`` and its JSON
protocol.  It deliberately contains no natural-data I/O, label join, training
runner, D1 field, endpoint access, or Slurm integration.

V2 retains the proven low-level numerical primitives from the immutable V1
matched arm, but does not call V1's descriptor-conditioned query-first
hypothesis enumerator or its single-endpoint cross-fit scorer.  A V2 proposal
is the complete Cartesian product of fixed 4x4 query and reference macro
windows.  It seals one complete connected ``(W_q,W_r)`` pair before evaluating
the two fixed double-endpoint phases.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch

from .cw0_connected_window_v1 import (
    CW0ConnectedWindow,
    CW0ConnectedWindowCore,
    CW0WeightedAffineFit,
    apply_affine,
    canonical_connected_window,
    target_free_schema_sha256,
    weighted_ridge_affine_fit,
)
from .geometry_hypothesis_v1 import fixed_macro_bank_split, grid_cell_centres
from .r0_crossview_cycle_v1 import R0SoftCorrespondence


CW0V2_SCHEMA_VERSION = "rc_cw0_reference_first_connected_region_core_v2"
CW0V2_SEED = 17
CW0V2_REFERENCE_MACRO_SIDE = 4
CW0V2_QUERY_MACRO_SIDE = 4
CW0V2_PHASE_COUNT = 2
CW0V2_DIRECTIONS_PER_PHASE = 2
CW0V2_RIDGE = 1.0e-3
CW0V2_EVIDENCE_CLIP = 4.0
CW0V2_HYPOTHESIS_TEMPERATURE = 0.10
CW0V2_CANDIDATE_TEMPERATURE = 0.10
CW0V2_H0_LOG_EVIDENCE = 0.0
CW0V2_ZERO_RETURN_CYCLE_PENALTY = 1.0
CW0V2_MIN_UNIQUE_ENDPOINTS = 4
CW0V2_MIN_EFFECTIVE_MATCHES = 8.0
CW0V2_MIN_COVARIANCE_EIGENVALUE = 1.0e-4
CW0V2_MAX_AFFINE_CONDITION = 1.0e4
CW0V2_MAX_REPROJECTION_RMSE = 0.04
CW0V2_MAX_REPROJECTION_ERROR = 0.08
CW0V2_MAX_DIRECTION_DISAGREEMENT_CELL_DIAGONALS = 2.0
CW0V2_TRAINABLE_PARAMETER_COUNT = 4_357
CW0V2_EPS = 1.0e-12


def _grid_shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError(f"{name} must be a positive (height,width) tuple")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class CW0V2MacroSeed:
    macro_row: int
    macro_column: int
    seed_index: int
    window: CW0ConnectedWindow

    def __post_init__(self) -> None:
        if self.macro_row < 0 or self.macro_column < 0:
            raise ValueError("macro coordinates must be non-negative")
        if self.seed_index != self.window.seed_index:
            raise ValueError("macro seed and window seed disagree")


CW0V2ReferenceSeed = CW0V2MacroSeed
CW0V2QuerySeed = CW0V2MacroSeed


def _enumerate_macro_seeds(
    grid_shape: tuple[int, int],
    *,
    name: str,
    macro_side: int,
) -> tuple[CW0V2MacroSeed, ...]:
    height, width = _grid_shape(grid_shape, name=name)
    output: list[CW0V2MacroSeed] = []
    macro_rows = math.ceil(height / macro_side)
    macro_columns = math.ceil(width / macro_side)
    for macro_y in range(macro_rows):
        y0 = macro_y * macro_side
        y1 = min(height, y0 + macro_side)
        seed_y = (y0 + y1 - 1) // 2
        for macro_x in range(macro_columns):
            x0 = macro_x * macro_side
            x1 = min(width, x0 + macro_side)
            seed_x = (x0 + x1 - 1) // 2
            seed = seed_y * width + seed_x
            output.append(
                CW0V2MacroSeed(
                    macro_row=macro_y,
                    macro_column=macro_x,
                    seed_index=seed,
                    window=canonical_connected_window(seed, (height, width)),
                )
            )
    return tuple(output)


def enumerate_reference_macro_seeds(
    reference_grid_shape: tuple[int, int],
) -> tuple[CW0V2ReferenceSeed, ...]:
    """Enumerate one lower row-major centre per fixed 4x4 macro tile."""

    return _enumerate_macro_seeds(
        reference_grid_shape,
        name="reference grid",
        macro_side=CW0V2_REFERENCE_MACRO_SIDE,
    )


def enumerate_query_macro_seeds(
    query_grid_shape: tuple[int, int],
) -> tuple[CW0V2QuerySeed, ...]:
    """Enumerate every fixed query window without reading descriptors."""

    return _enumerate_macro_seeds(
        query_grid_shape,
        name="query grid",
        macro_side=CW0V2_QUERY_MACRO_SIDE,
    )


def maximum_reference_seed_coverage_distance(
    reference_grid_shape: tuple[int, int],
) -> int:
    """Return the exact maximum Manhattan distance to the nearest fixed seed."""

    height, width = _grid_shape(reference_grid_shape, name="reference grid")
    seeds = enumerate_reference_macro_seeds((height, width))
    coordinates = [(seed.seed_index // width, seed.seed_index % width) for seed in seeds]
    return max(
        min(abs(y - sy) + abs(x - sx) for sy, sx in coordinates)
        for y in range(height)
        for x in range(width)
    )


def maximum_query_seed_coverage_distance(query_grid_shape: tuple[int, int]) -> int:
    height, width = _grid_shape(query_grid_shape, name="query grid")
    seeds = enumerate_query_macro_seeds((height, width))
    coordinates = [(seed.seed_index // width, seed.seed_index % width) for seed in seeds]
    return max(
        min(abs(y - sy) + abs(x - sx) for sy, sx in coordinates)
        for y in range(height)
        for x in range(width)
    )

@dataclass(frozen=True)
class CW0V2EndpointPartition:
    phase: int
    direction: int
    fit_query: torch.Tensor
    fit_reference: torch.Tensor
    verify_query: torch.Tensor
    verify_reference: torch.Tensor

    def __post_init__(self) -> None:
        if not 0 <= self.phase < CW0V2_PHASE_COUNT or not 0 <= self.direction < CW0V2_DIRECTIONS_PER_PHASE:
            raise ValueError("cross-fit phase or direction is out of range")
        values = []
        for item in (
            self.fit_query,
            self.fit_reference,
            self.verify_query,
            self.verify_reference,
        ):
            value = torch.as_tensor(item, dtype=torch.bool).detach().cpu().contiguous()
            if value.ndim != 1:
                raise ValueError("endpoint masks must be flattened")
            values.append(value)
        fit_q, fit_r, verify_q, verify_r = values
        if fit_q.shape != verify_q.shape or fit_r.shape != verify_r.shape:
            raise ValueError("fit/verify endpoint capacity drift")
        if bool((fit_q & verify_q).any() or (fit_r & verify_r).any()):
            raise ValueError("fit and verification endpoints must be exact-disjoint")
        object.__setattr__(self, "fit_query", fit_q)
        object.__setattr__(self, "fit_reference", fit_r)
        object.__setattr__(self, "verify_query", verify_q)
        object.__setattr__(self, "verify_reference", verify_r)


def double_endpoint_crossfit_partitions(
    query_window: torch.Tensor,
    reference_window: torch.Tensor,
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> tuple[tuple[CW0V2EndpointPartition, CW0V2EndpointPartition], ...]:
    """Return both frozen checkerboard phases over byte-identical windows."""

    q_split = fixed_macro_bank_split(query_window, query_grid_shape)
    r_split = fixed_macro_bank_split(reference_window, reference_grid_shape)
    phases = (
        (
            (q_split.a, r_split.a, q_split.b, r_split.b),
            (q_split.b, r_split.b, q_split.a, r_split.a),
        ),
        (
            (q_split.b, r_split.a, q_split.a, r_split.b),
            (q_split.a, r_split.b, q_split.b, r_split.a),
        ),
    )
    return tuple(
        tuple(
            CW0V2EndpointPartition(
                phase=phase,
                direction=direction,
                fit_query=items[0],
                fit_reference=items[1],
                verify_query=items[2],
                verify_reference=items[3],
            )
            for direction, items in enumerate(directions)
        )
        for phase, directions in enumerate(phases)
    )


@dataclass(frozen=True)
class CW0V2RegionHypothesis:
    candidate_index: int
    query_seed_ordinal: int
    query_seed_index: int
    reference_seed_ordinal: int
    reference_seed_index: int
    query_window: CW0ConnectedWindow
    reference_window: CW0ConnectedWindow
    partitions: tuple[tuple[CW0V2EndpointPartition, CW0V2EndpointPartition], ...]
    mask_pair_sha256: str

    def __post_init__(self) -> None:
        if (
            self.candidate_index < 0
            or self.query_seed_ordinal < 0
            or self.reference_seed_ordinal < 0
        ):
            raise ValueError("candidate and seed ordinals must be non-negative")
        if self.reference_seed_index != self.reference_window.seed_index:
            raise ValueError("reference hypothesis seed drift")
        if self.query_seed_index != self.query_window.seed_index:
            raise ValueError("query hypothesis seed drift")
        if len(self.partitions) != CW0V2_PHASE_COUNT or any(
            len(phase) != CW0V2_DIRECTIONS_PER_PHASE for phase in self.partitions
        ):
            raise ValueError("V2 requires two phases with two directions each")
        if not self.query_window.legal or not self.reference_window.legal:
            raise ValueError("V2 region membership must be a legal dense single 4CC")
        if (
            not isinstance(self.mask_pair_sha256, str)
            or len(self.mask_pair_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.mask_pair_sha256)
        ):
            raise ValueError("mask-pair SHA256 must be lowercase hexadecimal")
        if self.mask_pair_sha256 != _window_pair_sha256(
            self.query_window, self.reference_window
        ):
            raise ValueError("mask-pair integrity hash drift")
        q_mask = self.query_window.mask
        r_mask = self.reference_window.mask
        for phase in self.partitions:
            for direction in phase:
                if not torch.equal(direction.fit_query | direction.verify_query, q_mask):
                    raise ValueError("query cross-fit endpoints must partition full Wq")
                if not torch.equal(direction.fit_reference | direction.verify_reference, r_mask):
                    raise ValueError("reference cross-fit endpoints must partition full Wr")


def _window_pair_sha256(query: CW0ConnectedWindow, reference: CW0ConnectedWindow) -> str:
    digest = hashlib.sha256()
    digest.update(bytes(int(item) for item in query.mask.tolist()))
    digest.update(b"|")
    digest.update(bytes(int(item) for item in reference.mask.tolist()))
    return digest.hexdigest()


def enumerate_reference_first_hypotheses(
    *,
    candidate_index: int,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> tuple[CW0V2RegionHypothesis, ...]:
    """Return the complete descriptor-free query x reference Cartesian product."""

    query_seeds = enumerate_query_macro_seeds(query_grid_shape)
    reference_seeds = enumerate_reference_macro_seeds(reference_grid_shape)
    output: list[CW0V2RegionHypothesis] = []
    # Reference-major order matches Z_b = mean_a exp(S_ab/tau).
    for reference_ordinal, reference_seed in enumerate(reference_seeds):
        for query_ordinal, query_seed in enumerate(query_seeds):
            partitions = double_endpoint_crossfit_partitions(
                query_seed.window.mask,
                reference_seed.window.mask,
                query_grid_shape=query_grid_shape,
                reference_grid_shape=reference_grid_shape,
            )
            output.append(
                CW0V2RegionHypothesis(
                    candidate_index=candidate_index,
                    query_seed_ordinal=query_ordinal,
                    query_seed_index=query_seed.seed_index,
                    reference_seed_ordinal=reference_ordinal,
                    reference_seed_index=reference_seed.seed_index,
                    query_window=query_seed.window,
                    reference_window=reference_seed.window,
                    partitions=partitions,
                    mask_pair_sha256=_window_pair_sha256(
                        query_seed.window, reference_seed.window
                    ),
                )
            )
    expected = len(query_seeds) * len(reference_seeds)
    if len(output) != expected:
        raise RuntimeError("Cartesian connected-hypothesis count drift")
    return tuple(output)


@dataclass(frozen=True)
class CW0V2HeldoutEvidence:
    signed_raw: torch.Tensor
    signed_clipped: torch.Tensor
    reliability: torch.Tensor
    evidence: torch.Tensor
    reciprocal_mass: torch.Tensor
    return_mass: torch.Tensor
    reprojection_distance: torch.Tensor
    cycle_error: torch.Tensor
    dustbin_failure: torch.Tensor
    conditional_entropy: torch.Tensor
    zero_return: torch.Tensor


@dataclass(frozen=True)
class CW0V2HardDirectionDiagnostics:
    structural_fit_query_endpoints: int
    structural_fit_reference_endpoints: int
    structural_verify_query_endpoints: int
    structural_verify_reference_endpoints: int
    fit_unique_query_endpoints: int
    fit_unique_reference_endpoints: int
    verify_unique_query_endpoints: int
    verify_unique_reference_endpoints: int
    effective_matches: torch.Tensor
    source_min_eigenvalue: torch.Tensor
    target_min_eigenvalue: torch.Tensor
    affine_condition: torch.Tensor
    reprojection_rmse: torch.Tensor
    maximum_reprojection_error: torch.Tensor
    affine_determinant: torch.Tensor
    eligible: bool


@dataclass(frozen=True)
class CW0V2DirectionOutput:
    score: torch.Tensor
    affine_fit: CW0WeightedAffineFit
    heldout: CW0V2HeldoutEvidence
    hard: CW0V2HardDirectionDiagnostics


def stable_reprojection_metrics(
    reprojection_delta: torch.Tensor,
    zero_return: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return squared, per-row, RMS, and maximum reprojection errors.

    ``sqrt(mean(delta**2))`` has an undefined autograd derivative at an exact
    zero residual and PyTorch consequently returns NaN gradients there.  The
    vector-norm form is mathematically identical, but uses the finite zero
    subgradient selected by ``torch.linalg.vector_norm``.  A zero-return row
    remains the contract's exact unit failure before any aggregate is formed.
    """

    delta = torch.as_tensor(reprojection_delta)
    missing = torch.as_tensor(zero_return, dtype=torch.bool, device=delta.device)
    if (
        delta.ndim != 2
        or delta.shape[1] != 2
        or delta.shape[0] == 0
        or missing.shape != (delta.shape[0],)
        or not bool(torch.isfinite(delta).all())
    ):
        raise ValueError("reprojection metrics require finite [N,2] deltas")
    distance = torch.linalg.vector_norm(delta, dim=1)
    distance = torch.where(missing, torch.ones_like(distance), distance)
    squared = distance.square()
    rmse = torch.linalg.vector_norm(distance) / math.sqrt(distance.numel())
    maximum = distance.max()
    return squared, distance, rmse, maximum


def _zero_affine_fit(value: torch.Tensor) -> CW0WeightedAffineFit:
    zero = value.sum() * 0.0
    matrix = torch.eye(3, dtype=value.dtype, device=value.device)
    pair = torch.zeros((2, 2), dtype=value.dtype, device=value.device)
    return CW0WeightedAffineFit(matrix, zero, zero, pair, pair, zero, zero, zero)


def _zero_direction_output(
    correspondence: R0SoftCorrespondence,
    partition: CW0V2EndpointPartition,
) -> CW0V2DirectionOutput:
    zero = correspondence.q_to_r.sum() * 0.0
    count = int(partition.verify_reference.sum())
    vector = torch.zeros(count, dtype=zero.dtype, device=zero.device)
    heldout = CW0V2HeldoutEvidence(
        signed_raw=vector,
        signed_clipped=vector,
        reliability=vector,
        evidence=vector,
        reciprocal_mass=vector,
        return_mass=vector,
        reprojection_distance=vector,
        cycle_error=vector,
        dustbin_failure=vector,
        conditional_entropy=vector,
        zero_return=torch.ones(count, dtype=torch.bool, device=zero.device),
    )
    hard = CW0V2HardDirectionDiagnostics(
        int(partition.fit_query.sum()),
        int(partition.fit_reference.sum()),
        int(partition.verify_query.sum()),
        int(partition.verify_reference.sum()),
        0,
        0,
        0,
        0,
        zero,
        zero,
        zero,
        torch.full_like(zero, 1.0e30),
        torch.full_like(zero, CW0V2_ZERO_RETURN_CYCLE_PENALTY),
        torch.full_like(zero, CW0V2_ZERO_RETURN_CYCLE_PENALTY),
        zero,
        False,
    )
    return CW0V2DirectionOutput(zero, _zero_affine_fit(zero), heldout, hard)


def score_crossfit_direction(
    correspondence: R0SoftCorrespondence,
    hypothesis: CW0V2RegionHypothesis,
    partition: CW0V2EndpointPartition,
) -> CW0V2DirectionOutput:
    """Fit and verify one exact-disjoint endpoint block with no renormalization."""

    candidate = hypothesis.candidate_index
    pair = correspondence.reciprocal_pair_mass
    if not 0 <= candidate < pair.shape[0]:
        raise ValueError("candidate index is out of range")
    fit_q = torch.nonzero(partition.fit_query.to(pair.device), as_tuple=False).flatten()
    fit_r = torch.nonzero(partition.fit_reference.to(pair.device), as_tuple=False).flatten()
    verify_q = torch.nonzero(partition.verify_query.to(pair.device), as_tuple=False).flatten()
    verify_r = torch.nonzero(partition.verify_reference.to(pair.device), as_tuple=False).flatten()
    if min(fit_q.numel(), fit_r.numel(), verify_q.numel(), verify_r.numel()) == 0:
        return _zero_direction_output(correspondence, partition)

    q_xy = grid_cell_centres(
        hypothesis.query_window.grid_shape, dtype=pair.dtype, device=pair.device
    )
    r_xy = grid_cell_centres(
        hypothesis.reference_window.grid_shape, dtype=pair.dtype, device=pair.device
    )
    fit_weight = pair[candidate][fit_q][:, fit_r]
    source = r_xy[fit_r][None, :, :].expand(fit_q.numel(), -1, -1).reshape(-1, 2)
    target = q_xy[fit_q][:, None, :].expand(-1, fit_r.numel(), -1).reshape(-1, 2)
    fit = weighted_ridge_affine_fit(
        source, target, fit_weight.reshape(-1), ridge=CW0V2_RIDGE
    )

    predicted = apply_affine(fit.matrix, r_xy[verify_r])
    reverse = correspondence.r_to_q[candidate, verify_r][:, verify_q]
    return_mass = reverse.sum(dim=1)
    zero_return = return_mass.eq(0.0)
    conditional = reverse / return_mass[:, None].clamp_min(CW0V2_EPS)
    expected_query = conditional @ q_xy[verify_q]
    (
        reprojection_squared,
        reprojection_distance,
        reprojection_rmse,
        maximum_reprojection_error,
    ) = stable_reprojection_metrics(
        expected_query - predicted,
        zero_return,
    )
    squared_distance = torch.cdist(predicted, q_xy[verify_q]).square()
    conditional_cycle = (conditional * squared_distance).sum(dim=1)
    cycle_error = conditional_cycle + (1.0 - return_mass.clamp(0.0, 1.0))
    cycle_error = torch.where(
        zero_return,
        torch.full_like(cycle_error, CW0V2_ZERO_RETURN_CYCLE_PENALTY),
        cycle_error,
    )
    entropy = -(conditional * conditional.clamp_min(CW0V2_EPS).log()).sum(dim=1)
    if verify_q.numel() > 1:
        entropy = entropy / math.log(verify_q.numel())
    else:
        entropy = entropy * 0.0

    forward = correspondence.q_to_r[candidate, verify_q][:, verify_r]
    verify_pair = forward * reverse.transpose(0, 1)
    reciprocal = verify_pair.sum(dim=0)
    outside_return = (
        correspondence.r_to_q[candidate, verify_r].sum(dim=1) - return_mass
    ).clamp_min(0.0)
    dustbin = correspondence.reference_dustbin[candidate, verify_r]
    nonreciprocal = (return_mass - reciprocal).clamp_min(0.0)
    failure = dustbin + outside_return + nonreciprocal

    rho_q = correspondence.query_reliability[candidate, verify_q]
    conditional_rho_q = (conditional * rho_q[None, :]).sum(dim=1)
    conditional_rho_q = torch.where(
        zero_return,
        rho_q.mean().expand_as(conditional_rho_q),
        conditional_rho_q,
    )
    rho_r = correspondence.reference_reliability[candidate, verify_r]
    reliability = torch.sqrt((conditional_rho_q * rho_r).clamp_min(0.0))
    q_height, q_width = hypothesis.query_window.grid_shape
    cell_scale_squared = (1.0 / q_height) ** 2 + (1.0 / q_width) ** 2
    signed_raw = (
        torch.log((reciprocal + CW0V2_EPS) / (failure + CW0V2_EPS))
        - reprojection_squared / cell_scale_squared
        - cycle_error / cell_scale_squared
        - entropy
    )
    signed_clipped = signed_raw.clamp(-CW0V2_EVIDENCE_CLIP, CW0V2_EVIDENCE_CLIP)
    evidence = reliability * signed_clipped
    # Fixed full held-out reference-bank denominator; no reliability or
    # positive-cell renormalization is allowed.
    score = evidence.sum() / float(verify_r.numel())
    heldout = CW0V2HeldoutEvidence(
        signed_raw,
        signed_clipped,
        reliability,
        evidence,
        reciprocal,
        return_mass,
        reprojection_distance,
        cycle_error,
        failure,
        entropy,
        zero_return,
    )

    linear = fit.matrix[:2, :2]
    singular = torch.linalg.svdvals(linear)
    affine_condition = torch.where(
        singular[-1].gt(CW0V2_EPS),
        singular[0] / singular[-1].clamp_min(CW0V2_EPS),
        torch.full_like(singular[0], 1.0e30),
    )
    determinant = torch.linalg.det(linear)
    rmse = reprojection_rmse
    maximum = maximum_reprojection_error
    structural_endpoint_counts = (
        int(fit_q.numel()),
        int(fit_r.numel()),
        int(verify_q.numel()),
        int(verify_r.numel()),
    )
    # "Unique endpoint" means an endpoint with actual strictly positive
    # reciprocal marginal mass in the prescribed block.  No unregistered mass
    # threshold is introduced; structural bank sizes remain separately
    # receipted so dense-softmax behaviour is auditable.
    endpoint_counts = (
        int(fit_weight.sum(dim=1).detach().gt(0.0).sum()),
        int(fit_weight.sum(dim=0).detach().gt(0.0).sum()),
        int(verify_pair.sum(dim=1).detach().gt(0.0).sum()),
        int(verify_pair.sum(dim=0).detach().gt(0.0).sum()),
    )
    eligible = bool(
        min(endpoint_counts) >= CW0V2_MIN_UNIQUE_ENDPOINTS
        and float(fit.effective_matches.detach()) >= CW0V2_MIN_EFFECTIVE_MATCHES
        and float(fit.source_min_eigenvalue.detach()) >= CW0V2_MIN_COVARIANCE_EIGENVALUE
        and float(fit.target_min_eigenvalue.detach()) >= CW0V2_MIN_COVARIANCE_EIGENVALUE
        and float(affine_condition.detach()) < CW0V2_MAX_AFFINE_CONDITION
        and float(rmse.detach()) < CW0V2_MAX_REPROJECTION_RMSE
        and float(maximum.detach()) < CW0V2_MAX_REPROJECTION_ERROR
        and float(determinant.detach()) > 0.0
    )
    hard = CW0V2HardDirectionDiagnostics(
        structural_endpoint_counts[0],
        structural_endpoint_counts[1],
        structural_endpoint_counts[2],
        structural_endpoint_counts[3],
        endpoint_counts[0],
        endpoint_counts[1],
        endpoint_counts[2],
        endpoint_counts[3],
        fit.effective_matches,
        fit.source_min_eigenvalue,
        fit.target_min_eigenvalue,
        affine_condition,
        rmse,
        maximum,
        determinant,
        eligible,
    )
    return CW0V2DirectionOutput(score, fit, heldout, hard)


def _normalized_lme(
    scores: torch.Tensor,
    *,
    temperature: float,
    include_h0: bool,
) -> torch.Tensor:
    value = torch.as_tensor(scores)
    if value.ndim != 1 or value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("normalized LME requires a nonempty finite vector")
    if bool(value.detach().eq(0.0).all()):
        return value.sum() * 0.0
    if include_h0:
        value = torch.cat(
            (torch.zeros(1, dtype=value.dtype, device=value.device), value), dim=0
        )
    return temperature * (
        torch.logsumexp(value / temperature, dim=0) - math.log(value.numel())
    )


def hierarchical_candidate_reducer(
    hypothesis_scores: torch.Tensor,
    *,
    reference_window_count: int,
    query_window_count: int,
) -> torch.Tensor:
    """Reference-major two-axis mean reducer with fixed H0 prior 0.5.

    ``hypothesis_scores`` must be ``[N_r,N_q]``.  No mask hash or hard-H1
    status changes its mathematical weight.
    """

    value = torch.as_tensor(hypothesis_scores)
    if (
        isinstance(reference_window_count, bool)
        or isinstance(query_window_count, bool)
        or reference_window_count <= 0
        or query_window_count <= 0
        or value.shape != (reference_window_count, query_window_count)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("hierarchical reducer shape/count drift")
    if bool(value.detach().eq(0.0).all()):
        return value.sum() * 0.0
    tau = CW0V2_CANDIDATE_TEMPERATURE
    log_z_per_reference = (
        torch.logsumexp(value / tau, dim=1) - math.log(query_window_count)
    )
    log_z_bar = (
        torch.logsumexp(log_z_per_reference, dim=0)
        - math.log(reference_window_count)
    )
    log_half = math.log(0.5)
    return tau * torch.logaddexp(
        torch.tensor(log_half, dtype=value.dtype, device=value.device),
        log_z_bar + log_half,
    )


@dataclass(frozen=True)
class CW0V2PhaseOutput:
    score: torch.Tensor
    directions: tuple[CW0V2DirectionOutput, CW0V2DirectionOutput]
    direction_disagreement_mse: torch.Tensor
    direction_agreement_slack: torch.Tensor
    hard_eligible: bool


@dataclass(frozen=True)
class CW0V2HypothesisOutput:
    score: torch.Tensor
    phases: tuple[CW0V2PhaseOutput, CW0V2PhaseOutput]
    hard_h1: bool


def score_region_hypothesis(
    correspondence: R0SoftCorrespondence,
    hypothesis: CW0V2RegionHypothesis,
) -> CW0V2HypothesisOutput:
    phases: list[CW0V2PhaseOutput] = []
    for phase_partitions in hypothesis.partitions:
        directions = tuple(
            score_crossfit_direction(correspondence, hypothesis, partition)
            for partition in phase_partitions
        )
        direction_scores = torch.stack([item.score for item in directions])
        phase_score = direction_scores.mean()
        # The two exact-disjoint fits must explain the same immutable full Wr.
        # Agreement is measured on fixed reference coordinates, never on a
        # descriptor-selected or nearest-neighbour subset.  MSE avoids a
        # square-root singularity while remaining exactly equivalent to the
        # registered RMS < two query-cell-diagonals hard gate.
        pair = correspondence.reciprocal_pair_mass
        reference_xy = grid_cell_centres(
            hypothesis.reference_window.grid_shape,
            dtype=pair.dtype,
            device=pair.device,
        )
        full_reference = torch.nonzero(
            hypothesis.reference_window.mask.to(pair.device), as_tuple=False
        ).flatten()
        prediction_0 = apply_affine(
            directions[0].affine_fit.matrix, reference_xy[full_reference]
        )
        prediction_1 = apply_affine(
            directions[1].affine_fit.matrix, reference_xy[full_reference]
        )
        direction_disagreement_mse = (
            prediction_0 - prediction_1
        ).square().sum(dim=1).mean()
        query_height, query_width = hypothesis.query_window.grid_shape
        query_cell_diagonal_squared = (
            (1.0 / query_height) ** 2 + (1.0 / query_width) ** 2
        )
        direction_threshold_squared = (
            CW0V2_MAX_DIRECTION_DISAGREEMENT_CELL_DIAGONALS**2
            * query_cell_diagonal_squared
        )
        direction_agreement_slack = (
            1.0 - direction_disagreement_mse / direction_threshold_squared
        )
        direction_agreement_hard = bool(
            torch.isfinite(direction_disagreement_mse).detach()
            and float(direction_disagreement_mse.detach())
            < direction_threshold_squared
        )
        phases.append(
            CW0V2PhaseOutput(
                score=phase_score,
                directions=(directions[0], directions[1]),
                direction_disagreement_mse=direction_disagreement_mse,
                direction_agreement_slack=direction_agreement_slack,
                hard_eligible=(
                    all(item.hard.eligible for item in directions)
                    and direction_agreement_hard
                ),
            )
        )
    phase_scores = torch.stack([item.score for item in phases])
    hypothesis_score = _normalized_lme(
        phase_scores,
        temperature=CW0V2_HYPOTHESIS_TEMPERATURE,
        include_h0=False,
    )
    return CW0V2HypothesisOutput(
        score=hypothesis_score,
        phases=(phases[0], phases[1]),
        hard_h1=any(item.hard_eligible for item in phases),
    )


@dataclass(frozen=True)
class CW0V2CandidateOutput:
    candidate_score: torch.Tensor
    hypotheses: tuple[CW0V2RegionHypothesis, ...]
    outputs: tuple[CW0V2HypothesisOutput, ...]
    query_window_count: int
    reference_window_count: int
    soft_map_non_h0_index: int | None
    hard_map_non_h0_index: int | None

    @property
    def map_non_h0_index(self) -> int | None:
        """Compatibility spelling for the only reportable, hard-qualified MAP."""

        return self.hard_map_non_h0_index

    @property
    def reported_masks(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.hard_map_non_h0_index is None:
            return None
        hypothesis = self.hypotheses[self.hard_map_non_h0_index]
        return hypothesis.query_window.mask, hypothesis.reference_window.mask


def score_candidate_reference_first(
    correspondence: R0SoftCorrespondence,
    *,
    candidate_index: int,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> CW0V2CandidateOutput:
    hypotheses = enumerate_reference_first_hypotheses(
        candidate_index=candidate_index,
        query_grid_shape=query_grid_shape,
        reference_grid_shape=reference_grid_shape,
    )
    if not hypotheses:
        raise RuntimeError("fixed reference-first enumeration produced no hypotheses")
    outputs = tuple(
        score_region_hypothesis(correspondence, hypothesis)
        for hypothesis in hypotheses
    )
    scores = torch.stack([item.score for item in outputs])
    query_count = len(enumerate_query_macro_seeds(query_grid_shape))
    reference_count = len(enumerate_reference_macro_seeds(reference_grid_shape))
    if scores.numel() != query_count * reference_count:
        raise RuntimeError("candidate score grid is not the complete Cartesian product")
    candidate_score = hierarchical_candidate_reducer(
        scores.reshape(reference_count, query_count),
        reference_window_count=reference_count,
        query_window_count=query_count,
    )
    maximum, index = torch.max(scores.detach(), dim=0)
    soft_map = int(index) if float(maximum) > CW0V2_H0_LOG_EVIDENCE else None
    hard_indices = [index for index, item in enumerate(outputs) if item.hard_h1]
    hard_map: int | None = None
    if hard_indices:
        hard_scores = scores.detach()[hard_indices]
        hard_maximum, local_index = torch.max(hard_scores, dim=0)
        if float(hard_maximum) > CW0V2_H0_LOG_EVIDENCE:
            hard_map = hard_indices[int(local_index)]
    return CW0V2CandidateOutput(
        candidate_score,
        hypotheses,
        outputs,
        query_count,
        reference_count,
        soft_map,
        hard_map,
    )


class CW0ReferenceFirstConnectedRegionCore(CW0ConnectedWindowCore):
    """V2 alias of the shared, identity-free 4,357-parameter token core."""

    def __init__(self, *, temperature: float = 0.07, seed: int = CW0V2_SEED) -> None:
        super().__init__(temperature=temperature, seed=seed)
        if sum(parameter.numel() for parameter in self.parameters()) != CW0V2_TRAINABLE_PARAMETER_COUNT:
            raise RuntimeError("CW0 V2 trainable parameter count drift")


def hypothesis_payload(hypothesis: CW0V2RegionHypothesis) -> dict[str, Any]:
    """Return the deterministic, target-free V2 hypothesis receipt."""

    return {
        "schema_version": CW0V2_SCHEMA_VERSION,
        "candidate_index": hypothesis.candidate_index,
        "query_seed_ordinal": hypothesis.query_seed_ordinal,
        "query_seed_index": hypothesis.query_seed_index,
        "reference_seed_ordinal": hypothesis.reference_seed_ordinal,
        "reference_seed_index": hypothesis.reference_seed_index,
        "mask_pair_sha256": hypothesis.mask_pair_sha256,
        "query_window_indices": torch.nonzero(
            hypothesis.query_window.mask, as_tuple=False
        ).flatten().tolist(),
        "reference_window_indices": torch.nonzero(
            hypothesis.reference_window.mask, as_tuple=False
        ).flatten().tolist(),
        "phase_endpoint_indices": [
            [
                {
                    "fit_query": torch.nonzero(item.fit_query, as_tuple=False).flatten().tolist(),
                    "fit_reference": torch.nonzero(item.fit_reference, as_tuple=False).flatten().tolist(),
                    "verify_query": torch.nonzero(item.verify_query, as_tuple=False).flatten().tolist(),
                    "verify_reference": torch.nonzero(item.verify_reference, as_tuple=False).flatten().tolist(),
                }
                for item in phase
            ]
            for phase in hypothesis.partitions
        ],
        "target_free": True,
        "reference_first": True,
        "descriptor_conditioned_proposal": False,
        "deduplicated_for_scoring": False,
        "complete_membership_not_redefined": True,
    }


def hypothesis_sha256(hypothesis: CW0V2RegionHypothesis) -> str:
    return target_free_schema_sha256(hypothesis_payload(hypothesis))


__all__ = [
    "CW0V2_SCHEMA_VERSION",
    "CW0V2_SEED",
    "CW0V2_REFERENCE_MACRO_SIDE",
    "CW0V2_QUERY_MACRO_SIDE",
    "CW0V2_PHASE_COUNT",
    "CW0V2_DIRECTIONS_PER_PHASE",
    "CW0V2_RIDGE",
    "CW0V2_EVIDENCE_CLIP",
    "CW0V2_HYPOTHESIS_TEMPERATURE",
    "CW0V2_CANDIDATE_TEMPERATURE",
    "CW0V2_H0_LOG_EVIDENCE",
    "CW0V2_MAX_DIRECTION_DISAGREEMENT_CELL_DIAGONALS",
    "CW0V2_TRAINABLE_PARAMETER_COUNT",
    "CW0V2MacroSeed",
    "CW0V2ReferenceSeed",
    "CW0V2QuerySeed",
    "CW0V2EndpointPartition",
    "CW0V2RegionHypothesis",
    "CW0V2HeldoutEvidence",
    "CW0V2HardDirectionDiagnostics",
    "CW0V2DirectionOutput",
    "CW0V2PhaseOutput",
    "CW0V2HypothesisOutput",
    "CW0V2CandidateOutput",
    "CW0ReferenceFirstConnectedRegionCore",
    "enumerate_reference_macro_seeds",
    "enumerate_query_macro_seeds",
    "maximum_reference_seed_coverage_distance",
    "maximum_query_seed_coverage_distance",
    "double_endpoint_crossfit_partitions",
    "enumerate_reference_first_hypotheses",
    "score_crossfit_direction",
    "score_region_hypothesis",
    "hierarchical_candidate_reducer",
    "score_candidate_reference_first",
    "stable_reprojection_metrics",
    "hypothesis_payload",
    "hypothesis_sha256",
]
