"""CW0 soft connected-window proposal-recall primitives.

This module implements only the mathematical E0 core frozen in
``CW0_CONNECTED_WINDOW_PROPOSAL_RECALL_CONTRACT_V1_20260809.md``.  It does
not read labels, D1 fields, candidate ranks, natural-data manifests, or any
evaluation endpoint.  In particular, it does not replace or mutate the
historical hard-prejoin lineage.

The latent object represented here is always one *complete* pair of dense,
four-neighbour-connected masks ``(W_q, W_r)``.  A seed initializes that pair;
fit anchors and held-out rows are measurements of the sealed pair and can
never redefine its membership.  The two cross-fit directions use the same
``W_q`` and ``W_r`` and differ only in which fixed reference macro bank fits
or verifies the affine map.

Discrete proposal construction is deliberately detached.  Gradients flow
through the shared adapter, soft correspondence, weighted ridge fits, and
held-out evidence, but never through BFS, centre quantisation, rounding, or
mask membership.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn

from .geometry_hypothesis_v1 import (
    GEOMETRY_CANONICAL_RESOLUTION,
    MACRO_TILE_SIDE,
    connected_components_4,
    fixed_macro_bank_split,
    grid_cell_centres,
)
from .r0_crossview_cycle_v1 import (
    ADAPTER_RANK,
    DESCRIPTOR_DIM,
    AsymmetricRank8ResidualAdapter,
    R0SoftCorrespondence,
    soft_correspondence_with_dustbin,
)


CW0_SCHEMA_VERSION = "rc_cw0_connected_window_core_v1"
CW0_SEED = 17
CW0_GRAPH_RADIUS = 6
CW0_WINDOW_CAP = 64
CW0_MIN_WINDOW_CELLS = 8
CW0_CROSSFIT_DIRECTIONS = 2
CW0_H0_LOG_EVIDENCE = 0.0
CW0_DEFAULT_CORRESPONDENCE_TEMPERATURE = 0.07
CW0_DEFAULT_WINDOW_TEMPERATURE = 0.10
CW0_DEFAULT_RIDGE = 1.0e-3
CW0_ZERO_RETURN_CYCLE_PENALTY = 1.0
CW0_EPS = 1.0e-12
CW0_TRAINABLE_PARAMETER_COUNT = 4_357

REASON_LEGAL = "LEGAL"
REASON_QUERY_WINDOW = "QUERY_WINDOW_ILLEGAL"
REASON_REFERENCE_WINDOW = "REFERENCE_WINDOW_ILLEGAL"
REASON_NO_2D_SPAN = "WINDOW_NO_2D_SPAN"

_FORBIDDEN_TARGET_FREE_KEYS = frozenset(
    {
        "target",
        "target_id",
        "target_label",
        "positive_candidate_index",
        "ground_truth",
        "d1_score",
        "d1_rank",
        "d1_slot",
        "d1_winner",
        "d1_gap",
        "opened",
        "sealed",
    }
)


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


def _finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _canonicalize_coordinate(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Detach and quantise a two-vector on the frozen 1e-10 lattice."""

    coordinate = torch.as_tensor(value, dtype=torch.float64).detach().cpu().contiguous()
    if coordinate.shape != (2,) or not bool(torch.isfinite(coordinate).all()):
        raise ValueError("a finite two-dimensional centre is required")
    canonical = (
        torch.round(coordinate / GEOMETRY_CANONICAL_RESOLUTION)
        * GEOMETRY_CANONICAL_RESOLUTION
    )
    canonical = torch.where(canonical.eq(0.0), torch.zeros_like(canonical), canonical)
    return canonical.clamp(0.0, 1.0)


SeedEndpoint = Literal["query", "reference"]


def canonical_grid_cell(
    continuous_xy: torch.Tensor | Sequence[float],
    grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, int]:
    """Map a detached centre to one row-major cell on either endpoint."""

    height, width = _grid_shape(grid_shape, name="coordinate grid")
    centre = _canonicalize_coordinate(continuous_xy)
    # A centre exactly on the upper image boundary belongs to the last cell.
    x = min(width - 1, max(0, int(math.floor(float(centre[0]) * width))))
    y = min(height - 1, max(0, int(math.floor(float(centre[1]) * height))))
    return centre, y * width + x


def canonical_reference_cell(
    continuous_xy: torch.Tensor | Sequence[float],
    grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, int]:
    """Compatibility alias spelling out the contract's query->reference use."""

    return canonical_grid_cell(continuous_xy, grid_shape)


@dataclass(frozen=True)
class CW0ConnectedWindow:
    """Canonical BFS window; membership is immutable CPU boolean state."""

    seed_index: int
    grid_shape: tuple[int, int]
    mask: torch.Tensor
    member_indices: torch.Tensor
    graph_distances: torch.Tensor
    legal: bool
    reason: str

    def __post_init__(self) -> None:
        shape = _grid_shape(self.grid_shape, name="connected-window grid")
        count = math.prod(shape)
        if isinstance(self.seed_index, bool) or not isinstance(self.seed_index, int):
            raise ValueError("seed index must be an integer")
        if not 0 <= self.seed_index < count:
            raise ValueError("seed index is out of range")
        mask = torch.as_tensor(self.mask, dtype=torch.bool).detach().cpu().contiguous()
        members = torch.as_tensor(
            self.member_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        distances = torch.as_tensor(
            self.graph_distances, dtype=torch.long
        ).detach().cpu().contiguous()
        if mask.shape != (count,) or members.ndim != 1 or distances.shape != members.shape:
            raise ValueError("connected-window tensor schema drift")
        expected_members = torch.nonzero(mask, as_tuple=False).flatten()
        if not torch.equal(expected_members, members.sort().values):
            raise ValueError("member indices and connected mask disagree")
        if members.numel() > CW0_WINDOW_CAP:
            raise ValueError("connected window exceeds the frozen cap")
        if members.numel() and (
            int(members.min()) < 0
            or int(members.max()) >= count
            or members.unique().numel() != members.numel()
            or bool(distances.lt(0).any() or distances.gt(CW0_GRAPH_RADIUS).any())
        ):
            raise ValueError("connected-window members or distances are invalid")
        ordered = sorted(
            zip(distances.tolist(), members.tolist()), key=lambda item: (item[0], item[1])
        )
        if list(zip(distances.tolist(), members.tolist())) != ordered:
            raise ValueError("BFS members must use (distance,row-major) order")
        diagnostics = connected_components_4(mask, shape)[1]
        if self.legal:
            if (
                self.reason != REASON_LEGAL
                or members.numel() < CW0_MIN_WINDOW_CELLS
                or not diagnostics.connected_valid
                or diagnostics.component_count != 1
                or not diagnostics.has_2d_span
                or not bool(mask[self.seed_index])
            ):
                raise ValueError("legal CW0 window must be a dense single 4CC with 2-D span")
        elif self.reason == REASON_LEGAL:
            raise ValueError("an illegal window needs an explicit failure reason")
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "member_indices", members)
        object.__setattr__(self, "graph_distances", distances)


def canonical_connected_window(
    seed_index: int,
    grid_shape: tuple[int, int],
    *,
    radius: int = CW0_GRAPH_RADIUS,
    cap: int = CW0_WINDOW_CAP,
    minimum_cells: int = CW0_MIN_WINDOW_CELLS,
) -> CW0ConnectedWindow:
    """Construct the frozen radius-6/cap-64 canonical 4-neighbour BFS.

    The optional arguments exist only so engineering tests can prove that a
    caller cannot silently drift the frozen contract: values other than the
    registered constants are rejected.
    """

    if radius != CW0_GRAPH_RADIUS or cap != CW0_WINDOW_CAP or minimum_cells != CW0_MIN_WINDOW_CELLS:
        raise ValueError("CW0 radius, cap, and minimum membership are frozen")
    height, width = _grid_shape(grid_shape, name="connected-window grid")
    count = height * width
    if isinstance(seed_index, bool) or not isinstance(seed_index, int) or not 0 <= seed_index < count:
        raise ValueError("seed index is out of range")
    seed_y, seed_x = divmod(seed_index, width)

    # In an obstacle-free rectangular grid, ordering every cell by Manhattan
    # distance and then row-major index is exactly canonical breadth-first
    # traversal with queue order (graph_distance,row_major_index).
    ordered = sorted(
        (
            (abs(y - seed_y) + abs(x - seed_x), y * width + x)
            for y in range(height)
            for x in range(width)
            if abs(y - seed_y) + abs(x - seed_x) <= radius
        ),
        key=lambda item: (item[0], item[1]),
    )[:cap]
    distances = torch.tensor([item[0] for item in ordered], dtype=torch.long)
    members = torch.tensor([item[1] for item in ordered], dtype=torch.long)
    mask = torch.zeros(count, dtype=torch.bool)
    mask[members] = True
    diagnostics = connected_components_4(mask, (height, width))[1]
    if members.numel() < minimum_cells:
        legal = False
        reason = REASON_QUERY_WINDOW
    elif not diagnostics.connected_valid or diagnostics.component_count != 1:
        legal = False
        reason = REASON_QUERY_WINDOW
    elif not diagnostics.has_2d_span:
        legal = False
        reason = REASON_NO_2D_SPAN
    else:
        legal = True
        reason = REASON_LEGAL
    return CW0ConnectedWindow(
        seed_index=seed_index,
        grid_shape=(height, width),
        mask=mask,
        member_indices=members,
        graph_distances=distances,
        legal=legal,
        reason=reason,
    )


def _cell_centre(index: int, grid_shape: tuple[int, int]) -> torch.Tensor:
    shape = _grid_shape(grid_shape, name="cell-centre grid")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < math.prod(shape):
        raise ValueError("cell-centre index is out of range")
    return grid_cell_centres(shape, dtype=torch.float64, device="cpu")[index]


@dataclass(frozen=True)
class CW0ConnectedWindowPair:
    """Endpoint-configurable low-level construction of one sealed mask pair.

    CW0's frozen high-level path uses ``seed_endpoint='query'``.  Keeping this
    lower-level constructor symmetric makes that scientific choice explicit
    rather than baking it into BFS or membership semantics.
    """

    seed_endpoint: SeedEndpoint
    query_centre_xy: torch.Tensor
    reference_centre_xy: torch.Tensor
    query: CW0ConnectedWindow
    reference: CW0ConnectedWindow

    def __post_init__(self) -> None:
        if self.seed_endpoint not in ("query", "reference"):
            raise ValueError("seed endpoint must be query or reference")
        q_centre = _canonicalize_coordinate(self.query_centre_xy)
        r_centre = _canonicalize_coordinate(self.reference_centre_xy)
        _, q_index = canonical_grid_cell(q_centre, self.query.grid_shape)
        _, r_index = canonical_grid_cell(r_centre, self.reference.grid_shape)
        if q_index != self.query.seed_index or r_index != self.reference.seed_index:
            raise ValueError("window-pair centres must map to their seed cells")
        object.__setattr__(self, "query_centre_xy", q_centre)
        object.__setattr__(self, "reference_centre_xy", r_centre)


def build_connected_window_pair(
    *,
    seed_endpoint: SeedEndpoint,
    seed_index: int,
    predicted_other_centre_xy: torch.Tensor | Sequence[float],
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> CW0ConnectedWindowPair:
    """Build symmetric endpoint windows without assigning target evidence."""

    if seed_endpoint == "query":
        query_seed = seed_index
        reference_centre, reference_seed = canonical_grid_cell(
            predicted_other_centre_xy, reference_grid_shape
        )
        query_centre = _cell_centre(query_seed, query_grid_shape)
    elif seed_endpoint == "reference":
        reference_seed = seed_index
        query_centre, query_seed = canonical_grid_cell(
            predicted_other_centre_xy, query_grid_shape
        )
        reference_centre = _cell_centre(reference_seed, reference_grid_shape)
    else:
        raise ValueError("seed endpoint must be query or reference")
    return CW0ConnectedWindowPair(
        seed_endpoint=seed_endpoint,
        query_centre_xy=query_centre,
        reference_centre_xy=reference_centre,
        query=canonical_connected_window(query_seed, query_grid_shape),
        reference=canonical_connected_window(reference_seed, reference_grid_shape),
    )


@dataclass(frozen=True)
class CW0DeterministicHypothesis:
    """One target-free seed proposal with one shared connected footprint."""

    candidate_index: int
    query_seed_index: int
    reference_seed_index: int
    reference_centre_xy: torch.Tensor
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_window: torch.Tensor
    reference_window: torch.Tensor
    reference_bank_a: torch.Tensor
    reference_bank_b: torch.Tensor
    legal: bool
    reason: str

    def __post_init__(self) -> None:
        q_shape = _grid_shape(self.query_grid_shape, name="query grid")
        r_shape = _grid_shape(self.reference_grid_shape, name="reference grid")
        q_count, r_count = math.prod(q_shape), math.prod(r_shape)
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate index must be a non-negative integer")
        if not 0 <= self.query_seed_index < q_count or not 0 <= self.reference_seed_index < r_count:
            raise ValueError("hypothesis seed is out of range")
        centre = _canonicalize_coordinate(self.reference_centre_xy)
        query = torch.as_tensor(self.query_window, dtype=torch.bool).detach().cpu().contiguous()
        reference = torch.as_tensor(
            self.reference_window, dtype=torch.bool
        ).detach().cpu().contiguous()
        bank_a = torch.as_tensor(self.reference_bank_a, dtype=torch.bool).detach().cpu().contiguous()
        bank_b = torch.as_tensor(self.reference_bank_b, dtype=torch.bool).detach().cpu().contiguous()
        if query.shape != (q_count,) or reference.shape != (r_count,):
            raise ValueError("hypothesis window schema drift")
        if bank_a.shape != (r_count,) or bank_b.shape != (r_count,):
            raise ValueError("hypothesis bank schema drift")
        expected_split = fixed_macro_bank_split(reference, r_shape)
        if not torch.equal(bank_a, expected_split.a) or not torch.equal(bank_b, expected_split.b):
            raise ValueError("A/B must be the fixed 4x4 reference macro split")
        if bool((bank_a & bank_b).any()) or not torch.equal(bank_a | bank_b, reference):
            raise ValueError("A/B must exactly partition the sealed reference window")
        q_diag = connected_components_4(query, q_shape)[1]
        r_diag = connected_components_4(reference, r_shape)[1]
        if self.legal:
            if (
                self.reason != REASON_LEGAL
                or q_diag.component_count != 1
                or r_diag.component_count != 1
                or not q_diag.connected_valid
                or not r_diag.connected_valid
                or not q_diag.has_2d_span
                or not r_diag.has_2d_span
                or not CW0_MIN_WINDOW_CELLS <= q_diag.active_count <= CW0_WINDOW_CAP
                or not CW0_MIN_WINDOW_CELLS <= r_diag.active_count <= CW0_WINDOW_CAP
                or not bool(query[self.query_seed_index])
                or not bool(reference[self.reference_seed_index])
            ):
                raise ValueError("legal CW0 hypothesis requires two complete dense single 4CC masks")
        elif self.reason == REASON_LEGAL:
            raise ValueError("illegal CW0 hypothesis needs an explicit reason")
        object.__setattr__(self, "reference_centre_xy", centre)
        object.__setattr__(self, "query_grid_shape", q_shape)
        object.__setattr__(self, "reference_grid_shape", r_shape)
        object.__setattr__(self, "query_window", query)
        object.__setattr__(self, "reference_window", reference)
        object.__setattr__(self, "reference_bank_a", bank_a)
        object.__setattr__(self, "reference_bank_b", bank_b)


def build_connected_hypothesis(
    *,
    candidate_index: int,
    query_seed_index: int,
    reference_centre_xy: torch.Tensor | Sequence[float],
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> CW0DeterministicHypothesis:
    """Seal one pair of full windows before either cross-fit direction runs."""

    pair = build_connected_window_pair(
        seed_endpoint="query",
        seed_index=query_seed_index,
        predicted_other_centre_xy=reference_centre_xy,
        query_grid_shape=query_grid_shape,
        reference_grid_shape=reference_grid_shape,
    )
    q_window = pair.query
    r_window = pair.reference
    centre = pair.reference_centre_xy
    reference_seed = r_window.seed_index
    legal = q_window.legal and r_window.legal
    if not q_window.legal:
        reason = REASON_QUERY_WINDOW if q_window.reason != REASON_NO_2D_SPAN else REASON_NO_2D_SPAN
    elif not r_window.legal:
        reason = REASON_REFERENCE_WINDOW if r_window.reason != REASON_NO_2D_SPAN else REASON_NO_2D_SPAN
    else:
        reason = REASON_LEGAL
    split = fixed_macro_bank_split(r_window.mask, r_window.grid_shape)
    return CW0DeterministicHypothesis(
        candidate_index=candidate_index,
        query_seed_index=query_seed_index,
        reference_seed_index=reference_seed,
        reference_centre_xy=centre,
        query_grid_shape=q_window.grid_shape,
        reference_grid_shape=r_window.grid_shape,
        query_window=q_window.mask,
        reference_window=r_window.mask,
        reference_bank_a=split.a,
        reference_bank_b=split.b,
        legal=legal,
        reason=reason,
    )


def enumerate_connected_hypotheses(
    correspondence: R0SoftCorrespondence,
    *,
    candidate_index: int,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> tuple[CW0DeterministicHypothesis, ...]:
    """Enumerate every query cell result-blindly; no top-k is permitted."""

    q_shape = _grid_shape(query_grid_shape, name="query grid")
    r_shape = _grid_shape(reference_grid_shape, name="reference grid")
    expected = torch.as_tensor(correspondence.expected_reference_xy)
    if expected.ndim != 3 or not 0 <= candidate_index < expected.shape[0]:
        raise ValueError("candidate index is out of range for correspondence")
    if expected.shape[1:] != (math.prod(q_shape), 2):
        raise ValueError("expected-reference coordinate schema drift")
    return tuple(
        build_connected_hypothesis(
            candidate_index=candidate_index,
            query_seed_index=seed,
            reference_centre_xy=expected[candidate_index, seed],
            query_grid_shape=q_shape,
            reference_grid_shape=r_shape,
        )
        for seed in range(math.prod(q_shape))
    )


@dataclass(frozen=True)
class CW0WeightedAffineFit:
    """Differentiable weighted ridge fit and its soft anti-collapse receipt."""

    matrix: torch.Tensor  # [3,3], reference xy -> query xy
    total_mass: torch.Tensor
    effective_matches: torch.Tensor
    source_covariance: torch.Tensor  # [2,2]
    target_covariance: torch.Tensor  # [2,2]
    source_min_eigenvalue: torch.Tensor
    target_min_eigenvalue: torch.Tensor
    normalized_entropy: torch.Tensor


def _weighted_covariance_2d(
    coordinates: torch.Tensor,
    weights: torch.Tensor,
    total: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = (weights[:, None] * coordinates).sum(dim=0) / total.clamp_min(CW0_EPS)
    centered = coordinates - mean
    covariance = torch.einsum("n,ni,nj->ij", weights, centered, centered) / total.clamp_min(
        CW0_EPS
    )
    return mean, covariance


def weighted_ridge_affine_fit(
    source_xy: torch.Tensor,
    target_xy: torch.Tensor,
    weights: torch.Tensor,
    *,
    ridge: float = CW0_DEFAULT_RIDGE,
) -> CW0WeightedAffineFit:
    """Fit ``target = affine(source)`` without a hard match-count gate.

    The weighted centring form leaves the intercept unregularised and applies
    ridge only to the two-dimensional linear map.  Exact zero mass returns a
    finite canonical zero map (with homogeneous bottom row) while preserving a
    differentiable graph for nonzero soft mass.
    """

    ridge_value = _finite_positive(ridge, name="ridge")
    source = torch.as_tensor(source_xy)
    target = torch.as_tensor(target_xy, dtype=source.dtype, device=source.device)
    weight = torch.as_tensor(weights, dtype=source.dtype, device=source.device)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
        raise ValueError("weighted affine coordinates must both be [N,2]")
    if weight.shape != (source.shape[0],):
        raise ValueError("weighted affine weights must be [N]")
    if not bool(torch.isfinite(source).all() and torch.isfinite(target).all() and torch.isfinite(weight).all()):
        raise ValueError("weighted affine inputs must be finite")
    if bool(weight.lt(0).any()):
        raise ValueError("weighted affine weights must be non-negative")

    total = weight.sum()
    source_mean, source_covariance = _weighted_covariance_2d(source, weight, total)
    target_mean, target_covariance = _weighted_covariance_2d(target, weight, total)
    source_centered = source - source_mean
    target_centered = target - target_mean
    cross = torch.einsum("n,ni,nj->ij", weight, source_centered, target_centered) / total.clamp_min(
        CW0_EPS
    )
    identity = torch.eye(2, dtype=source.dtype, device=source.device)
    linear = torch.linalg.solve(
        source_covariance + ridge_value * identity,
        cross,
    )
    intercept = target_mean - source_mean @ linear
    top = torch.cat((linear.transpose(0, 1), intercept[:, None]), dim=1)
    bottom = torch.tensor([[0.0, 0.0, 1.0]], dtype=source.dtype, device=source.device)
    matrix = torch.cat((top, bottom), dim=0)

    square_sum = weight.square().sum()
    effective = total.square() / square_sum.clamp_min(CW0_EPS)
    probability = weight / total.clamp_min(CW0_EPS)
    entropy = -(probability * probability.clamp_min(CW0_EPS).log()).sum()
    if source.shape[0] > 1:
        entropy = entropy / math.log(source.shape[0])
    else:
        entropy = entropy * 0.0
    source_min = torch.linalg.eigvalsh(source_covariance)[0]
    target_min = torch.linalg.eigvalsh(target_covariance)[0]
    return CW0WeightedAffineFit(
        matrix=matrix,
        total_mass=total,
        effective_matches=effective,
        source_covariance=source_covariance,
        target_covariance=target_covariance,
        source_min_eigenvalue=source_min,
        target_min_eigenvalue=target_min,
        normalized_entropy=entropy,
    )


def apply_affine(matrix: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous reference-to-query affine to row-vector points."""

    value = torch.as_tensor(xy)
    transform = torch.as_tensor(matrix, dtype=value.dtype, device=value.device)
    if value.ndim != 2 or value.shape[1] != 2 or transform.shape != (3, 3):
        raise ValueError("affine application expects [N,2] coordinates and [3,3] matrix")
    homogeneous = torch.cat(
        (value, torch.ones((value.shape[0], 1), dtype=value.dtype, device=value.device)),
        dim=1,
    )
    return (homogeneous @ transform.transpose(0, 1))[:, :2]


@dataclass(frozen=True)
class CW0HeldoutEvidence:
    """Per-cell signed evidence; reliability is part of the actual score."""

    evidence: torch.Tensor
    signed_before_reliability: torch.Tensor
    reliability: torch.Tensor
    reciprocal_mass: torch.Tensor
    return_mass: torch.Tensor
    cycle_error: torch.Tensor
    reprojection_error: torch.Tensor
    dustbin_failure: torch.Tensor
    conditional_entropy: torch.Tensor
    zero_return: torch.Tensor


def heldout_reference_evidence(
    *,
    correspondence: R0SoftCorrespondence,
    candidate_index: int,
    query_rows: torch.Tensor,
    reference_rows: torch.Tensor,
    predicted_query_xy: torch.Tensor,
    query_xy_all: torch.Tensor,
    query_cell_scale_squared: float,
) -> CW0HeldoutEvidence:
    """Score held-out cells without allowing zero return to become zero error."""

    scale = _finite_positive(query_cell_scale_squared, name="query cell scale squared")
    q_rows = torch.as_tensor(query_rows, dtype=torch.long, device=correspondence.q_to_r.device)
    r_rows = torch.as_tensor(reference_rows, dtype=torch.long, device=correspondence.q_to_r.device)
    predicted = torch.as_tensor(
        predicted_query_xy,
        dtype=correspondence.q_to_r.dtype,
        device=correspondence.q_to_r.device,
    )
    q_coordinates = torch.as_tensor(
        query_xy_all,
        dtype=correspondence.q_to_r.dtype,
        device=correspondence.q_to_r.device,
    )
    if q_rows.ndim != 1 or r_rows.ndim != 1 or predicted.shape != (r_rows.numel(), 2):
        raise ValueError("held-out evidence row schema drift")
    if q_rows.numel() == 0 or r_rows.numel() == 0:
        empty = torch.zeros(r_rows.numel(), dtype=predicted.dtype, device=predicted.device)
        return CW0HeldoutEvidence(empty, empty, empty, empty, empty, empty, empty, empty, empty, empty.bool())

    reverse_local = correspondence.r_to_q[candidate_index, r_rows][:, q_rows]
    return_mass = reverse_local.sum(dim=1)
    zero_return = return_mass.eq(0.0)
    conditional = reverse_local / return_mass[:, None].clamp_min(CW0_EPS)
    expected_query = conditional @ q_coordinates[q_rows]
    reprojection = (expected_query - predicted).square().sum(dim=1)
    reprojection = torch.where(
        zero_return,
        torch.full_like(reprojection, CW0_ZERO_RETURN_CYCLE_PENALTY),
        reprojection,
    )
    squared_distance = torch.cdist(predicted, q_coordinates[q_rows]).square()
    conditional_cycle = (conditional * squared_distance).sum(dim=1)
    # Return lost to dustbin/outside W_q is an explicit cycle failure.  An
    # exact zero-return row therefore has penalty one, not zero.
    cycle_error = conditional_cycle + (1.0 - return_mass.clamp(0.0, 1.0))
    cycle_error = torch.where(
        zero_return,
        torch.full_like(cycle_error, CW0_ZERO_RETURN_CYCLE_PENALTY),
        cycle_error,
    )
    entropy = -(conditional * conditional.clamp_min(CW0_EPS).log()).sum(dim=1)
    if q_rows.numel() > 1:
        entropy = entropy / math.log(q_rows.numel())
    else:
        entropy = entropy * 0.0

    forward_local = correspondence.q_to_r[candidate_index, q_rows][:, r_rows]
    pair = forward_local * reverse_local.transpose(0, 1)
    reciprocal = pair.sum(dim=0)
    outside_return = (
        correspondence.r_to_q[candidate_index, r_rows].sum(dim=1) - return_mass
    ).clamp_min(0.0)
    dustbin = correspondence.reference_dustbin[candidate_index, r_rows]
    nonreciprocal = (return_mass - reciprocal).clamp_min(0.0)
    failure = dustbin + outside_return + nonreciprocal

    q_reliability = correspondence.query_reliability[candidate_index, q_rows]
    conditional_q_reliability = (conditional * q_reliability[None, :]).sum(dim=1)
    conditional_q_reliability = torch.where(
        zero_return,
        q_reliability.mean().expand_as(conditional_q_reliability),
        conditional_q_reliability,
    )
    r_reliability = correspondence.reference_reliability[candidate_index, r_rows]
    reliability = torch.sqrt(
        (conditional_q_reliability * r_reliability).clamp_min(0.0)
    )
    signed = (
        torch.log((reciprocal + CW0_EPS) / (failure + CW0_EPS))
        - reprojection / scale
        - cycle_error / scale
        - entropy
    )
    evidence = reliability * signed
    return CW0HeldoutEvidence(
        evidence=evidence,
        signed_before_reliability=signed,
        reliability=reliability,
        reciprocal_mass=reciprocal,
        return_mass=return_mass,
        cycle_error=cycle_error,
        reprojection_error=reprojection,
        dustbin_failure=failure,
        conditional_entropy=entropy,
        zero_return=zero_return,
    )


@dataclass(frozen=True)
class CW0SoftWindowDiagnostics:
    """Differentiable score and diagnostics for one sealed window pair."""

    score: torch.Tensor
    direction_scores: torch.Tensor  # [2]
    fits: tuple[CW0WeightedAffineFit, CW0WeightedAffineFit] | tuple[()]
    heldout: tuple[CW0HeldoutEvidence, CW0HeldoutEvidence] | tuple[()]
    total_reciprocal_mass: torch.Tensor
    effective_matches: torch.Tensor
    source_min_eigenvalue: torch.Tensor
    target_min_eigenvalue: torch.Tensor
    fit_entropy: torch.Tensor
    legal: bool
    reason: str


def _zero_window_diagnostics(
    correspondence: R0SoftCorrespondence,
    *,
    reason: str,
) -> CW0SoftWindowDiagnostics:
    zero = correspondence.q_to_r.sum() * 0.0
    pair = torch.stack((zero, zero))
    return CW0SoftWindowDiagnostics(
        score=zero,
        direction_scores=pair,
        fits=(),
        heldout=(),
        total_reciprocal_mass=pair,
        effective_matches=pair,
        source_min_eigenvalue=pair,
        target_min_eigenvalue=pair,
        fit_entropy=pair,
        legal=False,
        reason=reason,
    )


def score_connected_hypothesis(
    correspondence: R0SoftCorrespondence,
    hypothesis: CW0DeterministicHypothesis,
    *,
    ridge: float = CW0_DEFAULT_RIDGE,
) -> CW0SoftWindowDiagnostics:
    """Cross-fit one sealed ``(W_q,W_r)`` in both fixed A/B directions."""

    if not hypothesis.legal:
        return _zero_window_diagnostics(correspondence, reason=hypothesis.reason)
    candidate = hypothesis.candidate_index
    pair_mass = correspondence.reciprocal_pair_mass
    if not 0 <= candidate < pair_mass.shape[0]:
        raise ValueError("hypothesis candidate is out of correspondence range")
    q_mask = hypothesis.query_window.to(pair_mass.device)
    r_mask = hypothesis.reference_window.to(pair_mass.device)
    if q_mask.shape != (pair_mass.shape[1],) or r_mask.numel() > pair_mass.shape[2]:
        raise ValueError("hypothesis and correspondence grid sizes disagree")
    if bool(pair_mass[candidate, :, r_mask.numel() :].ne(0).any()):
        raise ValueError("reference padding carries reciprocal mass")
    q_rows = torch.nonzero(q_mask, as_tuple=False).flatten()
    q_xy = grid_cell_centres(
        hypothesis.query_grid_shape,
        dtype=pair_mass.dtype,
        device=pair_mass.device,
    )
    r_xy = grid_cell_centres(
        hypothesis.reference_grid_shape,
        dtype=pair_mass.dtype,
        device=pair_mass.device,
    )
    q_height, q_width = hypothesis.query_grid_shape
    cell_scale_squared = (1.0 / q_height) ** 2 + (1.0 / q_width) ** 2
    banks = (
        (hypothesis.reference_bank_a, hypothesis.reference_bank_b),
        (hypothesis.reference_bank_b, hypothesis.reference_bank_a),
    )
    direction_scores: list[torch.Tensor] = []
    fits: list[CW0WeightedAffineFit] = []
    heldouts: list[CW0HeldoutEvidence] = []
    for fit_mask_cpu, verify_mask_cpu in banks:
        fit_rows = torch.nonzero(fit_mask_cpu.to(pair_mass.device), as_tuple=False).flatten()
        verify_rows = torch.nonzero(verify_mask_cpu.to(pair_mass.device), as_tuple=False).flatten()
        if fit_rows.numel() == 0 or verify_rows.numel() == 0:
            # This is exact H0 for the direction; no alternate footprint or
            # nearest patch may be searched.
            direction_scores.append(pair_mass.sum() * 0.0)
            continue
        local_weight = pair_mass[candidate][q_rows][:, fit_rows]
        source = r_xy[fit_rows][None, :, :].expand(q_rows.numel(), -1, -1).reshape(-1, 2)
        target = q_xy[q_rows][:, None, :].expand(-1, fit_rows.numel(), -1).reshape(-1, 2)
        fit = weighted_ridge_affine_fit(
            source,
            target,
            local_weight.reshape(-1),
            ridge=ridge,
        )
        predicted = apply_affine(fit.matrix, r_xy[verify_rows])
        heldout = heldout_reference_evidence(
            correspondence=correspondence,
            candidate_index=candidate,
            query_rows=q_rows,
            reference_rows=verify_rows,
            predicted_query_xy=predicted,
            query_xy_all=q_xy,
            query_cell_scale_squared=cell_scale_squared,
        )
        # Every held-out member participates.  Zero-return cells are negative
        # evidence; they are not silently removed from this fixed denominator.
        direction_scores.append(heldout.evidence.mean())
        fits.append(fit)
        heldouts.append(heldout)
    if len(direction_scores) != CW0_CROSSFIT_DIRECTIONS:
        raise RuntimeError("CW0 cross-fit direction count drift")
    direction_tensor = torch.stack(direction_scores)
    score = direction_tensor.mean()
    if len(fits) != CW0_CROSSFIT_DIRECTIONS:
        # A macro bank empty at a tiny/boundary grid cannot authorize evidence.
        return _zero_window_diagnostics(correspondence, reason="EMPTY_CROSSFIT_BANK")
    return CW0SoftWindowDiagnostics(
        score=score,
        direction_scores=direction_tensor,
        fits=(fits[0], fits[1]),
        heldout=(heldouts[0], heldouts[1]),
        total_reciprocal_mass=torch.stack([fit.total_mass for fit in fits]),
        effective_matches=torch.stack([fit.effective_matches for fit in fits]),
        source_min_eigenvalue=torch.stack([fit.source_min_eigenvalue for fit in fits]),
        target_min_eigenvalue=torch.stack([fit.target_min_eigenvalue for fit in fits]),
        fit_entropy=torch.stack([fit.normalized_entropy for fit in fits]),
        legal=True,
        reason=REASON_LEGAL,
    )


def soft_noncollapse_barrier(diagnostics: CW0SoftWindowDiagnostics) -> torch.Tensor:
    """Differentiable mass/effective/2-D/entropy/cycle/dustbin barrier.

    This is a training loss, never a discrete window-deletion gate.  The
    thresholds define a smooth qualification pressure and do not alter the
    complete target masks.
    """

    if not diagnostics.legal:
        return diagnostics.score * 0.0
    mass = torch.nn.functional.softplus(8.0 - diagnostics.total_reciprocal_mass)
    effective = torch.nn.functional.softplus(8.0 - diagnostics.effective_matches)
    covariance = torch.nn.functional.softplus(
        1.0e-4 - diagnostics.source_min_eigenvalue
    ) + torch.nn.functional.softplus(1.0e-4 - diagnostics.target_min_eigenvalue)
    entropy = torch.nn.functional.softplus(diagnostics.fit_entropy - 0.90)
    heldout_penalties = []
    for item in diagnostics.heldout:
        heldout_penalties.append(
            item.cycle_error.mean()
            + item.dustbin_failure.mean()
            + torch.nn.functional.softplus(item.conditional_entropy.mean() - 0.90)
        )
    return (
        mass.mean()
        + effective.mean()
        + covariance.mean()
        + entropy.mean()
        + torch.stack(heldout_penalties).mean()
    )


def normalized_h0_window_reducer(
    window_scores: torch.Tensor,
    *,
    temperature: float = CW0_DEFAULT_WINDOW_TEMPERATURE,
) -> torch.Tensor:
    """Fixed-temperature normalized LME over all seeds plus one exact H0."""

    tau = _finite_positive(temperature, name="window temperature")
    scores = torch.as_tensor(window_scores)
    if scores.ndim != 1 or scores.numel() == 0 or not bool(torch.isfinite(scores).all()):
        raise ValueError("window reducer requires one finite score per result-blind seed")
    if bool(scores.detach().eq(0.0).all()):
        # Prevent logsumexp roundoff from making all-zero evidence depend on
        # the number of seeds/windows while retaining a well-typed graph.
        return scores.sum() * 0.0
    values = torch.cat(
        (torch.zeros(1, dtype=scores.dtype, device=scores.device), scores), dim=0
    )
    return tau * (
        torch.logsumexp(values / tau, dim=0) - math.log(values.numel())
    )


@dataclass(frozen=True)
class CW0CandidateOutput:
    candidate_score: torch.Tensor
    window_scores: torch.Tensor
    hypotheses: tuple[CW0DeterministicHypothesis, ...]
    diagnostics: tuple[CW0SoftWindowDiagnostics, ...]


def score_candidate_connected_windows(
    correspondence: R0SoftCorrespondence,
    *,
    candidate_index: int,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    ridge: float = CW0_DEFAULT_RIDGE,
    window_temperature: float = CW0_DEFAULT_WINDOW_TEMPERATURE,
) -> CW0CandidateOutput:
    """Enumerate and score every query seed for one candidate reference."""

    hypotheses = enumerate_connected_hypotheses(
        correspondence,
        candidate_index=candidate_index,
        query_grid_shape=query_grid_shape,
        reference_grid_shape=reference_grid_shape,
    )
    diagnostics = tuple(
        score_connected_hypothesis(correspondence, hypothesis, ridge=ridge)
        for hypothesis in hypotheses
    )
    window_scores = torch.stack([item.score for item in diagnostics])
    candidate_score = normalized_h0_window_reducer(
        window_scores, temperature=window_temperature
    )
    return CW0CandidateOutput(
        candidate_score=candidate_score,
        window_scores=window_scores,
        hypotheses=hypotheses,
        diagnostics=diagnostics,
    )


class CW0ConnectedWindowCore(nn.Module):
    """The frozen 4,357-parameter adapter/correspondence E0 core."""

    def __init__(
        self,
        *,
        temperature: float = CW0_DEFAULT_CORRESPONDENCE_TEMPERATURE,
        seed: int = CW0_SEED,
    ) -> None:
        super().__init__()
        self.temperature = _finite_positive(temperature, name="correspondence temperature")
        self.adapter = AsymmetricRank8ResidualAdapter(
            descriptor_dim=DESCRIPTOR_DIM,
            rank=ADAPTER_RANK,
            seed=seed,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1)
        self.query_reliability = nn.Linear(DESCRIPTOR_DIM, 1, dtype=torch.float64)
        self.reference_reliability = nn.Linear(DESCRIPTOR_DIM, 1, dtype=torch.float64)
        with torch.no_grad():
            self.query_reliability.weight.copy_(
                0.01
                * torch.randn(
                    self.query_reliability.weight.shape,
                    generator=generator,
                    dtype=torch.float64,
                )
            )
            self.reference_reliability.weight.copy_(
                0.01
                * torch.randn(
                    self.reference_reliability.weight.shape,
                    generator=generator,
                    dtype=torch.float64,
                )
            )
            self.query_reliability.bias.zero_()
            self.reference_reliability.bias.zero_()
        self.b_null = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        if sum(parameter.numel() for parameter in self.parameters()) != CW0_TRAINABLE_PARAMETER_COUNT:
            raise RuntimeError("CW0 trainable parameter count drift")

    def build_correspondence(
        self,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        *,
        query_grid_shape: tuple[int, int],
        reference_grid_shapes: Sequence[tuple[int, int]],
        reference_valid_mask: torch.Tensor,
    ) -> R0SoftCorrespondence:
        """Adapt frozen tokens and build candidate-bound soft correspondence."""

        query = self.adapter.adapt_query(query_tokens)
        reference = self.adapter.adapt_reference(reference_tokens)
        valid = torch.as_tensor(
            reference_valid_mask, dtype=torch.bool, device=reference.device
        )
        if valid.shape != reference.shape[:2]:
            raise ValueError("reference validity mask and token padding disagree")
        reference = torch.where(valid[..., None], reference, torch.zeros_like(reference))
        rho_q = torch.sigmoid(self.query_reliability(query).squeeze(-1))
        rho_r = torch.sigmoid(self.reference_reliability(reference).squeeze(-1))
        rho_r = torch.where(valid, rho_r, torch.full_like(rho_r, 0.5))
        return soft_correspondence_with_dustbin(
            query,
            reference,
            query_grid_shape=query_grid_shape,
            reference_grid_shapes=reference_grid_shapes,
            reference_valid_mask=valid,
            query_reliability=rho_q,
            reference_reliability=rho_r,
            dustbin_logit=self.b_null,
            temperature=self.temperature,
        )


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bool:
            return [bool(item) for item in tensor.flatten().tolist()]
        if tensor.dtype.is_floating_point:
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("canonical target-free payload must be finite")
            canonical = (
                torch.round(tensor.to(torch.float64) / GEOMETRY_CANONICAL_RESOLUTION)
                * GEOMETRY_CANONICAL_RESOLUTION
            )
            canonical = torch.where(canonical.eq(0.0), torch.zeros_like(canonical), canonical)
            return [float(item) for item in canonical.flatten().tolist()]
        return [int(item) for item in tensor.flatten().tolist()]
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in _FORBIDDEN_TARGET_FREE_KEYS:
                raise ValueError(f"target-free payload contains forbidden key: {name}")
            output[name] = _canonical_json_value(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical target-free payload must be finite")
        canonical = round(value / GEOMETRY_CANONICAL_RESOLUTION) * GEOMETRY_CANONICAL_RESOLUTION
        return 0.0 if canonical == 0.0 else canonical
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical payload type: {type(value).__name__}")


def hypothesis_schema_payload(hypothesis: CW0DeterministicHypothesis) -> dict[str, Any]:
    """Return a target-free logical receipt for one deterministic proposal."""

    return {
        "schema_version": CW0_SCHEMA_VERSION,
        "candidate_index": hypothesis.candidate_index,
        "query_seed_index": hypothesis.query_seed_index,
        "reference_seed_index": hypothesis.reference_seed_index,
        "reference_centre_xy": [float(item) for item in hypothesis.reference_centre_xy],
        "query_grid_shape": list(hypothesis.query_grid_shape),
        "reference_grid_shape": list(hypothesis.reference_grid_shape),
        "query_window_indices": torch.nonzero(
            hypothesis.query_window, as_tuple=False
        ).flatten().tolist(),
        "reference_window_indices": torch.nonzero(
            hypothesis.reference_window, as_tuple=False
        ).flatten().tolist(),
        "reference_bank_a_indices": torch.nonzero(
            hypothesis.reference_bank_a, as_tuple=False
        ).flatten().tolist(),
        "reference_bank_b_indices": torch.nonzero(
            hypothesis.reference_bank_b, as_tuple=False
        ).flatten().tolist(),
        "legal": hypothesis.legal,
        "reason": hypothesis.reason,
        "target_free": True,
        "seed_is_initialization_only": True,
        "membership_is_complete_dense_single_4cc": bool(hypothesis.legal),
        "macro_tile_side": MACRO_TILE_SIDE,
    }


def target_free_schema_sha256(value: Mapping[str, Any]) -> str:
    """Hash a canonical target-free mapping and fail on target/D1 fields."""

    canonical = _canonical_json_value(value)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hypothesis_sha256(hypothesis: CW0DeterministicHypothesis) -> str:
    return target_free_schema_sha256(hypothesis_schema_payload(hypothesis))


__all__ = [
    "CW0_SCHEMA_VERSION",
    "CW0_SEED",
    "CW0_GRAPH_RADIUS",
    "CW0_WINDOW_CAP",
    "CW0_MIN_WINDOW_CELLS",
    "CW0_CROSSFIT_DIRECTIONS",
    "CW0_H0_LOG_EVIDENCE",
    "CW0_DEFAULT_CORRESPONDENCE_TEMPERATURE",
    "CW0_DEFAULT_WINDOW_TEMPERATURE",
    "CW0_DEFAULT_RIDGE",
    "CW0_ZERO_RETURN_CYCLE_PENALTY",
    "CW0_TRAINABLE_PARAMETER_COUNT",
    "CW0ConnectedWindow",
    "CW0ConnectedWindowPair",
    "CW0DeterministicHypothesis",
    "CW0WeightedAffineFit",
    "CW0HeldoutEvidence",
    "CW0SoftWindowDiagnostics",
    "CW0CandidateOutput",
    "CW0ConnectedWindowCore",
    "SeedEndpoint",
    "canonical_grid_cell",
    "canonical_reference_cell",
    "canonical_connected_window",
    "build_connected_window_pair",
    "build_connected_hypothesis",
    "enumerate_connected_hypotheses",
    "weighted_ridge_affine_fit",
    "apply_affine",
    "heldout_reference_evidence",
    "score_connected_hypothesis",
    "soft_noncollapse_barrier",
    "normalized_h0_window_reducer",
    "score_candidate_connected_windows",
    "hypothesis_schema_payload",
    "target_free_schema_sha256",
    "hypothesis_sha256",
]
