"""Deterministic geometry for connected candidate-conditioned footprints.

This module is a new lineage.  It does not import the Qwen conditional gate or
the historical PVLock readout.  Its contract separates three objects:

``DiscreteGeometrySlots``
    Sparse proposal anchors plus dense query/reference footprints.  Anchors
    initialize geometry; they are never, by themselves, target evidence.

``FixedGeometryHypotheses``
    Exactly eight affine or homography slots fitted in FP64 with torch only.
    A legal slot requires a single dense 4-neighbour component on both sides,
    genuine 2-D extent, a non-degenerate fit, and bounded reprojection error.

``CrossBankVerificationCoordinates``
    Coordinates predicted in an independent verification grid.  Verification
    is restricted to the sealed footprint and excludes cells used to fit the
    transform.  There is no whole-image nearest-neighbour search here.

Illegal slots are canonical H0: identity placeholder matrix, false masks,
``-1`` indices, zero coordinates, and exact zero log evidence.  Connectivity
diagnostics remain available even for rejected proposal slots so a failure is
auditable without turning the rejected footprint into evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence, TypeVar

import torch


GEOMETRY_SLOTS = 8
MIN_FOOTPRINT_CELLS = 8
MIN_AFFINE_ANCHORS = 4
MIN_HOMOGRAPHY_ANCHORS = 8
MIN_VERIFICATION_CELLS = 4
MACRO_TILE_SIDE = 4
EPS = 1.0e-12
# Pre-registered serialization and decision resolution for every continuous
# geometry quantity.  This is deliberately much larger than the observed
# 1e-17--1e-14 LAPACK/CUDA last-bit drift, while remaining far below one token
# cell on every registered grid.  Logical indices and masks are never rounded.
GEOMETRY_CANONICAL_RESOLUTION = 1.0e-10
GEOMETRY_DECISION_GUARD = GEOMETRY_CANONICAL_RESOLUTION

MODEL_AFFINE = "affine"
MODEL_HOMOGRAPHY = "homography"
GeometryModel = Literal["affine", "homography"]

REASON_LEGAL = "LEGAL"
REASON_H0 = "H0_PADDING"
REASON_FOOTPRINT_DISCONNECTED = "FOOTPRINT_DISCONNECTED"
REASON_FOOTPRINT_SPARSE = "FOOTPRINT_SPARSE"
REASON_FOOTPRINT_NO_2D_SPAN = "FOOTPRINT_NO_2D_SPAN"
REASON_FOOTPRINT_AREA = "FOOTPRINT_AREA"
REASON_FOOTPRINT_SHAPE = "FOOTPRINT_SHAPE"
REASON_INSUFFICIENT_ANCHORS = "INSUFFICIENT_ANCHORS"
REASON_ANCHOR_NO_2D_SPAN = "ANCHOR_NO_2D_SPAN"
REASON_FIT_DEGENERATE = "FIT_DEGENERATE"
REASON_FIT_NONFINITE = "FIT_NONFINITE"
REASON_FIT_CONDITION = "FIT_CONDITION"
REASON_REPROJECTION = "REPROJECTION"
REASON_PROJECTED_AREA = "PROJECTED_AREA"
REASON_ORIENTATION = "ORIENTATION"
REASON_LOCAL_ANCHOR_COVERAGE = "LOCAL_ANCHOR_COVERAGE"
REASON_LOCAL_REGION_TOO_LARGE = "LOCAL_REGION_TOO_LARGE"


def _canonicalize_geometry(value: torch.Tensor | float) -> torch.Tensor:
    """Quantize continuous FP64 geometry to the registered replay lattice."""

    tensor = torch.as_tensor(value, dtype=torch.float64)
    canonical = (
        torch.round(tensor / GEOMETRY_CANONICAL_RESOLUTION)
        * GEOMETRY_CANONICAL_RESOLUTION
    )
    # Do not serialize platform-dependent signed zero.
    return torch.where(canonical.eq(0.0), torch.zeros_like(canonical), canonical)


def _canonical_geometry_scalar(value: torch.Tensor | float) -> float:
    return float(_canonicalize_geometry(value))


def _passes_upper_geometry_gate(value: float, maximum: float) -> bool:
    """Pass only below an upper threshold and outside its fail-closed halo."""

    canonical_value = _canonical_geometry_scalar(value)
    canonical_maximum = _canonical_geometry_scalar(maximum)
    return bool(
        math.isfinite(canonical_value)
        and math.isfinite(canonical_maximum)
        and canonical_value < canonical_maximum - GEOMETRY_DECISION_GUARD
    )


def _passes_geometry_interval(value: float, minimum: float, maximum: float) -> bool:
    """Pass an interval only when separated from both decision thresholds."""

    canonical_value = _canonical_geometry_scalar(value)
    canonical_minimum = _canonical_geometry_scalar(minimum)
    canonical_maximum = _canonical_geometry_scalar(maximum)
    return bool(
        math.isfinite(canonical_value)
        and canonical_value > canonical_minimum + GEOMETRY_DECISION_GUARD
        and canonical_value < canonical_maximum - GEOMETRY_DECISION_GUARD
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


def _indices(value: torch.Tensor, *, shape: tuple[int, ...], name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.long).detach().cpu().contiguous()
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _mask(value: torch.Tensor, *, shape: tuple[int, ...], name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous()
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def grid_cell_centres(
    grid_shape: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return canonical normalized ``(x,y)`` cell centres in row-major order."""

    height, width = _grid_shape(grid_shape, name="grid")
    y = (torch.arange(height, dtype=dtype, device=device) + 0.5) / float(height)
    x = (torch.arange(width, dtype=dtype, device=device) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def _xy(indices: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    height, width = _grid_shape(grid_shape, name="grid")
    value = torch.as_tensor(indices, dtype=torch.long)
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= height * width):
        raise ValueError("grid index is out of range")
    return torch.stack(
        (value.remainder(width), torch.div(value, width, rounding_mode="floor")),
        dim=-1,
    )


def _has_2d_span(indices: torch.Tensor, grid_shape: tuple[int, int]) -> bool:
    value = torch.as_tensor(indices, dtype=torch.long)
    if value.numel() < 3:
        return False
    coordinates = _xy(value, grid_shape)
    if coordinates[:, 0].unique().numel() < 2 or coordinates[:, 1].unique().numel() < 2:
        return False
    # Integer grid coordinates admit an exact rank-two test.  Avoid a floating
    # SVD tolerance becoming part of the logical footprint contract.
    origin = coordinates[0]
    vectors = coordinates[1:] - origin
    for first in range(vectors.shape[0]):
        cross = (
            vectors[first, 0] * vectors[first + 1 :, 1]
            - vectors[first, 1] * vectors[first + 1 :, 0]
        )
        if bool(cross.ne(0).any()):
            return True
    return False


@dataclass(frozen=True)
class FootprintDiagnostics:
    component_count: int
    largest_component_fraction: float
    compactness: float
    diameter: int
    perimeter: int
    isoperimetric: float
    connected_valid: bool
    has_2d_span: bool
    active_count: int


def connected_components_4(
    footprint: torch.Tensor,
    grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, FootprintDiagnostics]:
    """Label a flattened footprint using the fixed 4-neighbour graph.

    Labels are canonical: components are numbered by their smallest row-major
    cell.  The implementation uses torch tensors only and is deterministic on
    CPU or accelerator tensors (the returned ledger is canonical CPU data).
    """

    height, width = _grid_shape(grid_shape, name="footprint grid")
    value = torch.as_tensor(footprint, dtype=torch.bool).detach().cpu().contiguous()
    if value.shape != (height * width,):
        raise ValueError("footprint must be a flattened grid mask")
    labels = torch.full((height * width,), -1, dtype=torch.long)
    active = torch.nonzero(value, as_tuple=False).flatten()
    components: list[torch.Tensor] = []
    remaining = value.clone()
    while bool(remaining.any()):
        seed = int(torch.nonzero(remaining, as_tuple=False)[0])
        frontier = torch.zeros_like(remaining)
        frontier[seed] = True
        component = torch.zeros_like(remaining)
        while bool(frontier.any()):
            component |= frontier
            cells = frontier.reshape(height, width)
            neighbours = torch.zeros_like(cells)
            neighbours[1:] |= cells[:-1]
            neighbours[:-1] |= cells[1:]
            neighbours[:, 1:] |= cells[:, :-1]
            neighbours[:, :-1] |= cells[:, 1:]
            frontier = neighbours.reshape(-1) & remaining & ~component
        members = torch.nonzero(component, as_tuple=False).flatten()
        labels[members] = len(components)
        remaining[members] = False
        components.append(members)

    count = int(active.numel())
    if count == 0:
        diagnostics = FootprintDiagnostics(0, 0.0, 0.0, 0, 0, 0.0, False, False, 0)
        return labels, diagnostics
    sizes = [int(component.numel()) for component in components]
    coordinates = _xy(active, (height, width))
    span = coordinates.max(dim=0).values - coordinates.min(dim=0).values + 1
    bounding_area = int(span.prod())
    pair_distance = torch.cdist(
        coordinates.to(torch.float64), coordinates.to(torch.float64), p=1
    )
    cells = value.reshape(height, width)
    adjacent = int((cells[:, :-1] & cells[:, 1:]).sum()) + int(
        (cells[:-1] & cells[1:]).sum()
    )
    perimeter = 4 * count - 2 * adjacent
    diagnostics = FootprintDiagnostics(
        component_count=len(components),
        largest_component_fraction=float(max(sizes)) / float(count),
        compactness=float(count) / float(bounding_area),
        diameter=int(pair_distance.max()) if pair_distance.numel() else 0,
        perimeter=perimeter,
        isoperimetric=float(perimeter * perimeter) / (4.0 * math.pi * float(count)),
        connected_valid=len(components) == 1,
        has_2d_span=_has_2d_span(active, (height, width)),
        active_count=count,
    )
    return labels, diagnostics


def validate_connected_footprint(
    footprint: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    minimum_cells: int = MIN_FOOTPRINT_CELLS,
    minimum_compactness: float = 0.5,
) -> FootprintDiagnostics:
    """Return diagnostics; callers decide whether a failed footprint is H0."""

    if isinstance(minimum_cells, bool) or not isinstance(minimum_cells, int) or minimum_cells < 1:
        raise ValueError("minimum_cells must be a positive integer")
    if not math.isfinite(minimum_compactness) or not 0.0 < minimum_compactness <= 1.0:
        raise ValueError("minimum_compactness must be in (0,1]")
    _, diagnostics = connected_components_4(footprint, grid_shape)
    valid = (
        diagnostics.connected_valid
        and diagnostics.active_count >= minimum_cells
        and diagnostics.compactness >= minimum_compactness
        and diagnostics.has_2d_span
    )
    return FootprintDiagnostics(
        component_count=diagnostics.component_count,
        largest_component_fraction=diagnostics.largest_component_fraction,
        compactness=diagnostics.compactness,
        diameter=diagnostics.diameter,
        perimeter=diagnostics.perimeter,
        isoperimetric=diagnostics.isoperimetric,
        connected_valid=valid,
        has_2d_span=diagnostics.has_2d_span,
        active_count=diagnostics.active_count,
    )


@dataclass(frozen=True)
class GeneratedPairedFootprint:
    """Result of deterministic sparse-anchor to dense-footprint construction."""

    query_footprint: torch.Tensor
    reference_footprint: torch.Tensor
    included_anchor_mask: torch.Tensor
    query_diagnostics: FootprintDiagnostics
    reference_diagnostics: FootprintDiagnostics
    legal: bool
    reason: str

    def __post_init__(self) -> None:
        query = torch.as_tensor(self.query_footprint, dtype=torch.bool).detach().cpu().contiguous()
        reference = torch.as_tensor(self.reference_footprint, dtype=torch.bool).detach().cpu().contiguous()
        included = torch.as_tensor(self.included_anchor_mask, dtype=torch.bool).detach().cpu().contiguous()
        if query.ndim != 1 or reference.ndim != 1 or included.ndim != 1:
            raise ValueError("generated footprint tensors must be flattened")
        if self.legal:
            if (
                self.reason != REASON_LEGAL
                or not self.query_diagnostics.connected_valid
                or not self.reference_diagnostics.connected_valid
                or not bool(query.any())
                or not bool(reference.any())
            ):
                raise ValueError("legal generated footprint contract drift")
        elif bool(query.any() or reference.any()):
            raise ValueError("failed footprint generation must return exact empty H0")
        object.__setattr__(self, "query_footprint", query)
        object.__setattr__(self, "reference_footprint", reference)
        object.__setattr__(self, "included_anchor_mask", included)


def _empty_footprint_diagnostics() -> FootprintDiagnostics:
    return FootprintDiagnostics(0, 0.0, 0.0, 0, 0, 0.0, False, False, 0)


def project_reference_footprint_to_query(
    matrix: torch.Tensor,
    reference_footprint: torch.Tensor,
    *,
    reference_grid_shape: tuple[int, int],
    query_grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Project sealed reference cell squares to a dense query-grid mask.

    The reference cell *area*, not only its centre, is transformed.  Each
    transformed quadrilateral is conservatively rasterized by its query-cell
    overlap bounding interval.  Connectivity is validated by the caller after
    the complete reference component has been projected.
    """

    reference_height, reference_width = _grid_shape(
        reference_grid_shape, name="reference projection grid"
    )
    query_height, query_width = _grid_shape(
        query_grid_shape, name="query projection grid"
    )
    footprint = torch.as_tensor(reference_footprint, dtype=torch.bool).detach().cpu().contiguous()
    if footprint.shape != (reference_height * reference_width,):
        raise ValueError("reference projection footprint shape drift")
    output = torch.zeros(query_height * query_width, dtype=torch.bool)
    for reference_index in torch.nonzero(footprint, as_tuple=False).flatten().tolist():
        y, x = divmod(reference_index, reference_width)
        corners = torch.tensor(
            (
                (x / reference_width, y / reference_height),
                ((x + 1) / reference_width, y / reference_height),
                ((x + 1) / reference_width, (y + 1) / reference_height),
                (x / reference_width, (y + 1) / reference_height),
            ),
            dtype=torch.float64,
        )
        mapped = _canonicalize_geometry(apply_geometry(matrix, corners))
        left_value = float(mapped[:, 0].min())
        right_value = float(mapped[:, 0].max())
        top_value = float(mapped[:, 1].min())
        bottom_value = float(mapped[:, 1].max())
        if right_value <= 0.0 or left_value >= 1.0 or bottom_value <= 0.0 or top_value >= 1.0:
            continue
        tolerance = GEOMETRY_CANONICAL_RESOLUTION
        left = math.floor(left_value * query_width + tolerance)
        right = math.ceil(right_value * query_width - tolerance) - 1
        top = math.floor(top_value * query_height + tolerance)
        bottom = math.ceil(bottom_value * query_height - tolerance) - 1
        left = max(0, min(query_width - 1, left))
        right = max(0, min(query_width - 1, right))
        top = max(0, min(query_height - 1, top))
        bottom = max(0, min(query_height - 1, bottom))
        if left > right or top > bottom:
            continue
        rows = torch.arange(top, bottom + 1)[:, None]
        columns = torch.arange(left, right + 1)[None, :]
        output[(rows * query_width + columns).reshape(-1)] = True
    return output


def _convex_hull_grid_mask(
    anchor_indices: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Rasterize the canonical convex hull of discrete anchor centres."""

    height, width = _grid_shape(grid_shape, name="hull grid")
    points = sorted({tuple(value) for value in _xy(anchor_indices, grid_shape).tolist()})
    if len(points) < 3:
        return torch.zeros(height * width, dtype=torch.bool)

    def cross(origin: tuple[int, int], first: tuple[int, int], second: tuple[int, int]) -> int:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return torch.zeros(height * width, dtype=torch.bool)
    coordinates = _xy(torch.arange(height * width), grid_shape).to(torch.float64)
    inside = torch.ones(height * width, dtype=torch.bool)
    for start, end in zip(hull, hull[1:] + hull[:1], strict=True):
        edge_x = float(end[0] - start[0])
        edge_y = float(end[1] - start[1])
        relative_x = coordinates[:, 0] - float(start[0])
        relative_y = coordinates[:, 1] - float(start[1])
        inside &= (edge_x * relative_y - edge_y * relative_x).ge(-1.0e-12)
    return inside


def build_seed_local_connected_footprints(
    query_anchor_indices: torch.Tensor,
    reference_anchor_indices: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    seed_query_index: int,
    seed_reference_index: int,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    maximum_seed_radius: int = 6,
    minimum_anchors: int | None = None,
    maximum_bbox_cells: int = 64,
    maximum_aspect_ratio: float = 4.0,
    padding: int = 0,
    geometry_model: GeometryModel = MODEL_AFFINE,
    maximum_reprojection_rmse: float = 0.04,
    maximum_reprojection_error: float = 0.08,
    reference_admissible_mask: torch.Tensor | None = None,
) -> GeneratedPairedFootprint:
    """Generate ``R_r``, fit ``T(r->q)``, then project the unique ``R_q``.

    Query anchors constrain geometry but never independently draw a query
    bounding box.  The final query footprint therefore comes only from the
    candidate reference footprint and the fitted transform.  The filled cells
    are still a hypothesis, not evidence; an independent V bank must verify
    held-out cells inside this sealed region.
    """

    query_shape = _grid_shape(query_grid_shape, name="query grid")
    reference_shape = _grid_shape(reference_grid_shape, name="reference grid")
    query = torch.as_tensor(query_anchor_indices, dtype=torch.long).detach().cpu().contiguous()
    reference = torch.as_tensor(reference_anchor_indices, dtype=torch.long).detach().cpu().contiguous()
    active = torch.as_tensor(anchor_mask, dtype=torch.bool).detach().cpu().contiguous()
    if query.ndim != 1 or reference.shape != query.shape or active.shape != query.shape:
        raise ValueError("paired footprint anchors must share one [S] shape")
    if (
        isinstance(maximum_seed_radius, bool)
        or not isinstance(maximum_seed_radius, int)
        or maximum_seed_radius < 1
        or isinstance(maximum_bbox_cells, bool)
        or not isinstance(maximum_bbox_cells, int)
        or maximum_bbox_cells < MIN_FOOTPRINT_CELLS
        or isinstance(padding, bool)
        or not isinstance(padding, int)
        or padding != 0
        or not math.isfinite(maximum_aspect_ratio)
        or maximum_aspect_ratio < 1.0
        or geometry_model not in {MODEL_AFFINE, MODEL_HOMOGRAPHY}
        or not math.isfinite(maximum_reprojection_rmse)
        or maximum_reprojection_rmse <= 0.0
        or not math.isfinite(maximum_reprojection_error)
        or maximum_reprojection_error < maximum_reprojection_rmse
    ):
        raise ValueError("invalid seed-local footprint construction thresholds")
    required_anchors = (
        MIN_AFFINE_ANCHORS if geometry_model == MODEL_AFFINE else MIN_HOMOGRAPHY_ANCHORS
    )
    if minimum_anchors is None:
        minimum_anchors = required_anchors
    if (
        isinstance(minimum_anchors, bool)
        or not isinstance(minimum_anchors, int)
        or minimum_anchors < required_anchors
    ):
        raise ValueError("minimum anchors are below the registered model family")
    query_count = math.prod(query_shape)
    reference_count = math.prod(reference_shape)
    if not 0 <= seed_query_index < query_count or not 0 <= seed_reference_index < reference_count:
        raise ValueError("seed index is outside its grid")
    if bool(active.any()):
        query_active = query[active]
        reference_active = reference[active]
        if (
            int(query_active.min()) < 0
            or int(query_active.max()) >= query_count
            or int(reference_active.min()) < 0
            or int(reference_active.max()) >= reference_count
            or query_active.unique().numel() != query_active.numel()
            or reference_active.unique().numel() != reference_active.numel()
        ):
            raise ValueError("active footprint anchors must be in-range and one-to-one")
    seed_pair = active & query.eq(seed_query_index) & reference.eq(seed_reference_index)
    if not bool(seed_pair.any()):
        raise ValueError("seed pair must be active")
    query_xy = _xy(query.clamp_min(0), query_shape)
    reference_xy = _xy(reference.clamp_min(0), reference_shape)
    seed_query_xy = _xy(torch.tensor([seed_query_index]), query_shape)[0]
    seed_reference_xy = _xy(torch.tensor([seed_reference_index]), reference_shape)[0]
    local = (
        active
        & (query_xy - seed_query_xy).abs().amax(dim=1).le(maximum_seed_radius)
        & (reference_xy - seed_reference_xy).abs().amax(dim=1).le(maximum_seed_radius)
    )

    empty_query = torch.zeros(query_count, dtype=torch.bool)
    empty_reference = torch.zeros(reference_count, dtype=torch.bool)

    def failed(reason: str) -> GeneratedPairedFootprint:
        return GeneratedPairedFootprint(
            query_footprint=empty_query,
            reference_footprint=empty_reference,
            included_anchor_mask=local,
            query_diagnostics=_empty_footprint_diagnostics(),
            reference_diagnostics=_empty_footprint_diagnostics(),
            legal=False,
            reason=reason,
        )

    if int(local.sum()) < minimum_anchors:
        return failed(REASON_LOCAL_ANCHOR_COVERAGE)
    local_query = query[local]
    local_reference = reference[local]
    if not _has_2d_span(local_query, query_shape) or not _has_2d_span(local_reference, reference_shape):
        return failed(REASON_ANCHOR_NO_2D_SPAN)

    admissible = (
        torch.ones(reference_count, dtype=torch.bool)
        if reference_admissible_mask is None
        else _mask(
            reference_admissible_mask,
            shape=(reference_count,),
            name="reference admissible mask",
        )
    )
    reference_hull = _convex_hull_grid_mask(local_reference, reference_shape)
    all_reference_xy = _xy(torch.arange(reference_count), reference_shape)
    radius_mask = (all_reference_xy - seed_reference_xy).abs().amax(dim=1).le(
        maximum_seed_radius
    )
    reference_allowed = reference_hull & radius_mask & admissible
    reference_labels, _ = connected_components_4(reference_allowed, reference_shape)
    seed_component = int(reference_labels[seed_reference_index])
    if seed_component < 0:
        return failed(REASON_LOCAL_ANCHOR_COVERAGE)
    reference_footprint = reference_labels.eq(seed_component)
    # An inadmissible gap may separate some original anchor endpoints.  Those
    # pairs do not silently bridge the gap and are removed before fitting.
    local &= reference_footprint[reference.clamp_min(0)]
    if int(local.sum()) < minimum_anchors:
        return failed(REASON_LOCAL_ANCHOR_COVERAGE)
    local_query = query[local]
    local_reference = reference[local]
    if not _has_2d_span(local_query, query_shape) or not _has_2d_span(
        local_reference, reference_shape
    ):
        return failed(REASON_ANCHOR_NO_2D_SPAN)
    _, reference_diagnostics = connected_components_4(reference_footprint, reference_shape)
    reference_active_xy = _xy(
        torch.nonzero(reference_footprint, as_tuple=False).flatten(), reference_shape
    )
    reference_span = (
        reference_active_xy.max(dim=0).values
        - reference_active_xy.min(dim=0).values
        + 1
    )
    reference_bbox_area = int(reference_span.prod())
    reference_aspect = max(
        float(reference_span[0]) / float(reference_span[1]),
        float(reference_span[1]) / float(reference_span[0]),
    )
    if (
        reference_bbox_area > maximum_bbox_cells
        or reference_aspect > maximum_aspect_ratio
    ):
        return failed(REASON_LOCAL_REGION_TOO_LARGE)
    if not reference_diagnostics.connected_valid or not reference_diagnostics.has_2d_span:
        return failed(REASON_FOOTPRINT_NO_2D_SPAN)
    reference_area_fraction = reference_diagnostics.active_count / reference_count
    if (
        reference_diagnostics.active_count < MIN_FOOTPRINT_CELLS
        or not 0.005 <= reference_area_fraction <= 0.65
    ):
        return failed(REASON_FOOTPRINT_AREA)
    if (
        reference_diagnostics.compactness < 0.35
        or reference_diagnostics.isoperimetric > 4.0
    ):
        return failed(REASON_FOOTPRINT_SHAPE)
    query_centres = grid_cell_centres(query_shape)[local_query]
    reference_centres = grid_cell_centres(reference_shape)[local_reference]
    weights = torch.full((int(local.sum()),), 1.0 / float(local.sum()), dtype=torch.float64)
    try:
        matrix = (
            _fit_affine(reference_centres, query_centres, weights)
            if geometry_model == MODEL_AFFINE
            else _fit_homography(reference_centres, query_centres, weights)
        )
        design_condition = _geometry_design_condition(
            reference_centres, query_centres, weights, geometry_model
        )
        design_condition = _canonical_geometry_scalar(design_condition)
        if not _passes_upper_geometry_gate(design_condition, 1.0e4):
            return failed(REASON_FIT_CONDITION)
        projected_anchors = apply_geometry(matrix, reference_centres)
        residual = torch.linalg.vector_norm(projected_anchors - query_centres, dim=1)
        rmse = _canonical_geometry_scalar(
            torch.sqrt((residual.square() * weights).sum())
        )
        maximum = _canonical_geometry_scalar(residual.max())
        if not _passes_upper_geometry_gate(
            rmse, maximum_reprojection_rmse
        ) or not _passes_upper_geometry_gate(maximum, maximum_reprojection_error):
            return failed(REASON_REPROJECTION)
        query_footprint = project_reference_footprint_to_query(
            matrix,
            reference_footprint,
            reference_grid_shape=reference_shape,
            query_grid_shape=query_shape,
        )
    except (RuntimeError, ValueError):
        return failed(REASON_FIT_DEGENERATE)
    query_labels, _ = connected_components_4(query_footprint, query_shape)
    projected_seed = apply_geometry(
        matrix,
        grid_cell_centres(reference_shape)[torch.tensor([seed_reference_index])],
    )[0]
    if not bool(
        projected_seed[0].gt(GEOMETRY_DECISION_GUARD)
        & projected_seed[0].lt(1.0 - GEOMETRY_DECISION_GUARD)
        & projected_seed[1].gt(GEOMETRY_DECISION_GUARD)
        & projected_seed[1].lt(1.0 - GEOMETRY_DECISION_GUARD)
    ):
        return failed(REASON_REPROJECTION)
    projected_seed_scaled = _canonicalize_geometry(
        projected_seed * torch.tensor((query_shape[1], query_shape[0]))
    )
    projected_seed_x = min(query_shape[1] - 1, int(projected_seed_scaled[0]))
    projected_seed_y = min(query_shape[0] - 1, int(projected_seed_scaled[1]))
    projected_seed_index = projected_seed_y * query_shape[1] + projected_seed_x
    query_component = int(query_labels[projected_seed_index])
    if query_component < 0:
        return failed(REASON_REPROJECTION)
    query_footprint = query_labels.eq(query_component)
    _, query_diagnostics = connected_components_4(query_footprint, query_shape)
    if not query_diagnostics.connected_valid or not query_diagnostics.has_2d_span:
        return failed(REASON_FOOTPRINT_NO_2D_SPAN)
    query_area_fraction = query_diagnostics.active_count / query_count
    if (
        query_diagnostics.active_count < MIN_FOOTPRINT_CELLS
        or not 0.005 <= query_area_fraction <= 0.65
    ):
        return failed(REASON_FOOTPRINT_AREA)
    if (
        query_diagnostics.compactness < 0.35
        or query_diagnostics.isoperimetric > 4.0
    ):
        return failed(REASON_FOOTPRINT_SHAPE)
    query_active = torch.nonzero(query_footprint, as_tuple=False).flatten()
    query_xy = _xy(query_active, query_shape)
    query_span = query_xy.max(dim=0).values - query_xy.min(dim=0).values + 1
    query_area = int(query_span.prod())
    query_aspect = max(
        float(query_span[0]) / float(query_span[1]),
        float(query_span[1]) / float(query_span[0]),
    )
    if query_area > maximum_bbox_cells or query_aspect > maximum_aspect_ratio:
        return failed(REASON_LOCAL_REGION_TOO_LARGE)
    if not bool(query_footprint[local_query].all()):
        # If paired anchors do not lie in the projected reference region, the
        # reference did not actually propose those query cells.
        return failed(REASON_REPROJECTION)
    return GeneratedPairedFootprint(
        query_footprint=query_footprint,
        reference_footprint=reference_footprint,
        included_anchor_mask=local,
        query_diagnostics=query_diagnostics,
        reference_diagnostics=reference_diagnostics,
        legal=True,
        reason=REASON_LEGAL,
    )


@dataclass(frozen=True)
class FixedParitySplit:
    a: torch.Tensor
    b: torch.Tensor

    def __post_init__(self) -> None:
        a = torch.as_tensor(self.a, dtype=torch.bool).detach().cpu().contiguous()
        b = torch.as_tensor(self.b, dtype=torch.bool).detach().cpu().contiguous()
        if a.ndim < 1 or b.shape != a.shape or bool((a & b).any()):
            raise ValueError("fixed parity masks must be disjoint masks")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)


def fixed_macro_bank_split(
    footprint: torch.Tensor,
    grid_shape: tuple[int, int],
) -> FixedParitySplit:
    """Formal 4x4-cell macro-tile checkerboard A/B split."""

    height, width = _grid_shape(grid_shape, name="macro-bank grid")
    value = torch.as_tensor(footprint, dtype=torch.bool).detach().cpu().contiguous()
    if value.shape != (height * width,):
        raise ValueError("macro-bank footprint shape drift")
    index = torch.arange(height * width)
    x = index.remainder(width)
    y = torch.div(index, width, rounding_mode="floor")
    bank_a = (
        torch.div(x, MACRO_TILE_SIDE, rounding_mode="floor")
        + torch.div(y, MACRO_TILE_SIDE, rounding_mode="floor")
    ).remainder(2).eq(0)
    return FixedParitySplit(value & bank_a, value & ~bank_a)


def fixed_anchor_pair_split(slots: "DiscreteGeometrySlots") -> FixedParitySplit:
    """Split paired anchors by the formal reference 4x4 macro bank."""

    _, width = slots.reference_grid_shape
    reference = slots.anchor_reference_indices.clamp_min(0)
    bank_a = (
        torch.div(reference.remainder(width), MACRO_TILE_SIDE, rounding_mode="floor")
        + torch.div(
            torch.div(reference, width, rounding_mode="floor"),
            MACRO_TILE_SIDE,
            rounding_mode="floor",
        )
    ).remainder(2).eq(0)
    return FixedParitySplit(slots.anchor_mask & bank_a, slots.anchor_mask & ~bank_a)


@dataclass(frozen=True)
class DiscreteGeometrySlots:
    """Eight sparse anchor sets backed by dense connected footprints."""

    seed_query_indices: torch.Tensor
    seed_reference_indices: torch.Tensor
    anchor_query_indices: torch.Tensor
    anchor_reference_indices: torch.Tensor
    anchor_weights: torch.Tensor
    anchor_mask: torch.Tensor
    query_footprints: torch.Tensor
    reference_footprints: torch.Tensor
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]

    def __post_init__(self) -> None:
        query_shape = _grid_shape(self.query_grid_shape, name="query grid")
        reference_shape = _grid_shape(self.reference_grid_shape, name="reference grid")
        query_count = math.prod(query_shape)
        reference_count = math.prod(reference_shape)
        anchor_query = torch.as_tensor(self.anchor_query_indices, dtype=torch.long).detach().cpu().contiguous()
        if anchor_query.ndim != 2 or anchor_query.shape[0] != GEOMETRY_SLOTS:
            raise ValueError("anchor query tensor must be [8,S]")
        capacity = int(anchor_query.shape[1])
        if capacity < MIN_HOMOGRAPHY_ANCHORS:
            raise ValueError("anchor capacity must permit a homography")
        anchor_reference = _indices(
            self.anchor_reference_indices,
            shape=(GEOMETRY_SLOTS, capacity),
            name="anchor reference",
        )
        anchor_mask = _mask(
            self.anchor_mask,
            shape=(GEOMETRY_SLOTS, capacity),
            name="anchor mask",
        )
        weights = torch.as_tensor(self.anchor_weights, dtype=torch.float64).detach().cpu().contiguous()
        if weights.shape != anchor_query.shape or not bool(torch.isfinite(weights).all()):
            raise ValueError("anchor weights must be finite [8,S]")
        seed_query = _indices(self.seed_query_indices, shape=(GEOMETRY_SLOTS,), name="seed query")
        seed_reference = _indices(self.seed_reference_indices, shape=(GEOMETRY_SLOTS,), name="seed reference")
        query_footprints = _mask(
            self.query_footprints,
            shape=(GEOMETRY_SLOTS, query_count),
            name="query footprints",
        )
        reference_footprints = _mask(
            self.reference_footprints,
            shape=(GEOMETRY_SLOTS, reference_count),
            name="reference footprints",
        )
        for slot in range(GEOMETRY_SLOTS):
            active = anchor_mask[slot]
            footprint_active = bool(query_footprints[slot].any() or reference_footprints[slot].any())
            if not bool(active.any()):
                if (
                    footprint_active
                    or int(seed_query[slot]) != -1
                    or int(seed_reference[slot]) != -1
                    or bool(anchor_query[slot].ne(-1).any())
                    or bool(anchor_reference[slot].ne(-1).any())
                    or bool(weights[slot].ne(0).any())
                ):
                    raise ValueError("inactive slot must be canonical H0 padding")
                continue
            query = anchor_query[slot, active]
            reference = anchor_reference[slot, active]
            active_weights = weights[slot, active]
            if (
                int(query.min()) < 0
                or int(query.max()) >= query_count
                or int(reference.min()) < 0
                or int(reference.max()) >= reference_count
            ):
                raise ValueError("anchor index is outside its grid")
            if query.unique().numel() != query.numel() or reference.unique().numel() != reference.numel():
                raise ValueError("anchors must be one-to-one within each slot")
            if not bool(query_footprints[slot, query].all()) or not bool(reference_footprints[slot, reference].all()):
                raise ValueError("every anchor must lie inside its sealed footprint")
            seed_match = (query == seed_query[slot]) & (reference == seed_reference[slot])
            if not bool(seed_match.any()):
                raise ValueError("the seed pair must remain among the anchors")
            if not bool(active_weights.gt(0).all()) or float(active_weights.sum()) <= 0.0:
                raise ValueError("active anchor weights must be positive")
            if (
                bool(anchor_query[slot, ~active].ne(-1).any())
                or bool(anchor_reference[slot, ~active].ne(-1).any())
                or bool(weights[slot, ~active].ne(0).any())
            ):
                raise ValueError("inactive anchor entries must be -1/-1/0")
        object.__setattr__(self, "seed_query_indices", seed_query)
        object.__setattr__(self, "seed_reference_indices", seed_reference)
        object.__setattr__(self, "anchor_query_indices", anchor_query)
        object.__setattr__(self, "anchor_reference_indices", anchor_reference)
        object.__setattr__(self, "anchor_weights", weights)
        object.__setattr__(self, "anchor_mask", anchor_mask)
        object.__setattr__(self, "query_footprints", query_footprints)
        object.__setattr__(self, "reference_footprints", reference_footprints)
        object.__setattr__(self, "query_grid_shape", query_shape)
        object.__setattr__(self, "reference_grid_shape", reference_shape)

    @property
    def active(self) -> torch.Tensor:
        return self.anchor_mask.any(dim=1)


@dataclass(frozen=True)
class GeometryFitConfig:
    model: GeometryModel = MODEL_AFFINE
    minimum_footprint_cells: int = MIN_FOOTPRINT_CELLS
    minimum_compactness: float = 0.35
    minimum_area_fraction: float = 0.005
    maximum_area_fraction: float = 0.65
    maximum_isoperimetric: float = 4.0
    maximum_condition: float = 1.0e4
    maximum_reprojection_rmse: float = 0.04
    maximum_reprojection_error: float = 0.08
    minimum_projected_area: float = 1.0e-4
    maximum_projected_area: float = 16.0
    require_orientation_preserving: bool = True

    def __post_init__(self) -> None:
        if self.model not in {MODEL_AFFINE, MODEL_HOMOGRAPHY}:
            raise ValueError("geometry model must be affine or homography")
        if (
            isinstance(self.minimum_footprint_cells, bool)
            or not isinstance(self.minimum_footprint_cells, int)
            or self.minimum_footprint_cells < MIN_FOOTPRINT_CELLS
        ):
            raise ValueError("minimum footprint must contain at least eight cells")
        positive = (
            self.minimum_compactness,
            self.minimum_area_fraction,
            self.maximum_area_fraction,
            self.maximum_isoperimetric,
            self.maximum_condition,
            self.maximum_reprojection_rmse,
            self.maximum_reprojection_error,
            self.minimum_projected_area,
            self.maximum_projected_area,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("geometry thresholds must be finite and positive")
        if self.minimum_compactness > 1.0:
            raise ValueError("minimum compactness cannot exceed one")
        if (
            self.maximum_area_fraction > 1.0
            or self.minimum_area_fraction >= self.maximum_area_fraction
        ):
            raise ValueError("footprint area-fraction interval is invalid")
        if self.maximum_reprojection_error < self.maximum_reprojection_rmse:
            raise ValueError("maximum reprojection error must cover the RMSE")
        if self.maximum_projected_area <= self.minimum_projected_area:
            raise ValueError("projected area interval is empty")


@dataclass(frozen=True)
class FixedGeometryHypotheses:
    matrices: torch.Tensor
    legal: torch.Tensor
    reasons: tuple[str, ...]
    fit_anchor_mask: torch.Tensor
    reprojection_rmse: torch.Tensor
    reprojection_max: torch.Tensor
    condition: torch.Tensor
    query_component_count: torch.Tensor
    reference_component_count: torch.Tensor
    largest_component_fraction: torch.Tensor
    compactness: torch.Tensor
    diameter: torch.Tensor
    perimeter: torch.Tensor
    isoperimetric: torch.Tensor
    connected_valid: torch.Tensor
    h0_log_evidence: torch.Tensor
    model: GeometryModel

    def __post_init__(self) -> None:
        matrices = _canonicalize_geometry(
            torch.as_tensor(self.matrices, dtype=torch.float64)
        ).detach().cpu().contiguous()
        legal = _mask(self.legal, shape=(GEOMETRY_SLOTS,), name="geometry legal")
        fit_mask = torch.as_tensor(self.fit_anchor_mask, dtype=torch.bool).detach().cpu().contiguous()
        if matrices.shape != (GEOMETRY_SLOTS, 3, 3) or fit_mask.ndim != 2 or fit_mask.shape[0] != GEOMETRY_SLOTS:
            raise ValueError("fixed geometry schema drift")
        if len(self.reasons) != GEOMETRY_SLOTS:
            raise ValueError("geometry reason count drift")
        one = (GEOMETRY_SLOTS,)
        rmse = _canonicalize_geometry(self.reprojection_rmse).detach().cpu().contiguous()
        maximum = _canonicalize_geometry(self.reprojection_max).detach().cpu().contiguous()
        condition = _canonicalize_geometry(self.condition).detach().cpu().contiguous()
        q_components = _indices(self.query_component_count, shape=one, name="query component count")
        r_components = _indices(self.reference_component_count, shape=one, name="reference component count")
        largest = torch.as_tensor(self.largest_component_fraction, dtype=torch.float64).detach().cpu().contiguous()
        compactness = torch.as_tensor(self.compactness, dtype=torch.float64).detach().cpu().contiguous()
        diameter = torch.as_tensor(self.diameter, dtype=torch.long).detach().cpu().contiguous()
        perimeter = torch.as_tensor(self.perimeter, dtype=torch.long).detach().cpu().contiguous()
        isoperimetric = torch.as_tensor(self.isoperimetric, dtype=torch.float64).detach().cpu().contiguous()
        connected = _mask(self.connected_valid, shape=one, name="connected valid")
        h0 = torch.as_tensor(self.h0_log_evidence, dtype=torch.float64).detach().cpu().contiguous()
        for value, name in (
            (rmse, "reprojection rmse"),
            (maximum, "reprojection max"),
            (condition, "condition"),
            (h0, "H0 log evidence"),
        ):
            if value.shape != one or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} schema drift")
        if (
            largest.shape != (GEOMETRY_SLOTS, 2)
            or compactness.shape != (GEOMETRY_SLOTS, 2)
            or diameter.shape != (GEOMETRY_SLOTS, 2)
            or perimeter.shape != (GEOMETRY_SLOTS, 2)
            or isoperimetric.shape != (GEOMETRY_SLOTS, 2)
            or not bool(torch.isfinite(isoperimetric).all())
        ):
            raise ValueError("footprint diagnostic schema drift")
        identity = torch.eye(3, dtype=torch.float64).expand(GEOMETRY_SLOTS, -1, -1)
        if not torch.equal(matrices[~legal], identity[~legal]):
            raise ValueError("illegal geometry matrices must be exact identity H0")
        if bool(rmse[~legal].ne(0).any() or maximum[~legal].ne(0).any() or condition[~legal].ne(0).any()):
            raise ValueError("illegal geometry numerical payload must be exact zero")
        if bool(h0.ne(0).any()):
            raise ValueError("H0 log evidence must be exact zero")
        if self.model not in {MODEL_AFFINE, MODEL_HOMOGRAPHY}:
            raise ValueError("unknown geometry model")
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "legal", legal)
        object.__setattr__(self, "fit_anchor_mask", fit_mask)
        object.__setattr__(self, "reprojection_rmse", rmse)
        object.__setattr__(self, "reprojection_max", maximum)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "query_component_count", q_components)
        object.__setattr__(self, "reference_component_count", r_components)
        object.__setattr__(self, "largest_component_fraction", largest)
        object.__setattr__(self, "compactness", compactness)
        object.__setattr__(self, "diameter", diameter)
        object.__setattr__(self, "perimeter", perimeter)
        object.__setattr__(self, "isoperimetric", isoperimetric)
        object.__setattr__(self, "connected_valid", connected)
        object.__setattr__(self, "h0_log_evidence", h0)


def _normalization(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = points.mean(dim=0)
    distance = torch.linalg.vector_norm(points - center, dim=1).mean()
    canonical_distance = _canonical_geometry_scalar(distance)
    if (
        not math.isfinite(canonical_distance)
        or canonical_distance <= EPS + GEOMETRY_DECISION_GUARD
    ):
        raise ValueError(REASON_FIT_DEGENERATE)
    scale = math.sqrt(2.0) / canonical_distance
    transform = torch.tensor(
        (
            (scale, 0.0, -scale * float(center[0])),
            (0.0, scale, -scale * float(center[1])),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    homogeneous = torch.cat((points, torch.ones((points.shape[0], 1), dtype=torch.float64)), dim=1)
    return (homogeneous @ transform.T)[:, :2], transform


def _fit_affine(reference: torch.Tensor, query: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    source, source_transform = _normalization(reference)
    target, target_transform = _normalization(query)
    design = torch.cat((source, torch.ones((source.shape[0], 1), dtype=torch.float64)), dim=1)
    weighted_design = design * weights.sqrt()[:, None]
    weighted_query = target * weights.sqrt()[:, None]
    singular = _canonicalize_geometry(torch.linalg.svdvals(weighted_design))
    if float(singular[-1]) <= EPS + GEOMETRY_DECISION_GUARD:
        raise ValueError(REASON_FIT_DEGENERATE)
    solution = torch.linalg.lstsq(weighted_design, weighted_query, driver="gelsd").solution
    normalized = torch.eye(3, dtype=torch.float64)
    normalized[:2] = solution.T
    matrix = torch.linalg.inv(target_transform) @ normalized @ source_transform
    matrix = matrix / matrix[2, 2]
    return _canonicalize_geometry(matrix)


def _geometry_design_condition(
    reference: torch.Tensor,
    query: torch.Tensor,
    weights: torch.Tensor,
    model: GeometryModel,
) -> float:
    source, _ = _normalization(reference)
    target, _ = _normalization(query)
    if model == MODEL_AFFINE:
        design = torch.cat(
            (source, torch.ones((source.shape[0], 1), dtype=torch.float64)), dim=1
        )
        singular = torch.linalg.svdvals(design * weights.sqrt()[:, None])
        denominator = singular[-1]
    else:
        x, y = source[:, 0], source[:, 1]
        u, v = target[:, 0], target[:, 1]
        one = torch.ones_like(x)
        zero = torch.zeros_like(x)
        first = torch.stack(
            (-x, -y, -one, zero, zero, zero, u * x, u * y, u), dim=1
        )
        second = torch.stack(
            (zero, zero, zero, -x, -y, -one, v * x, v * y, v), dim=1
        )
        design = torch.stack((first, second), dim=1).reshape(-1, 9)
        design = design * weights.sqrt().repeat_interleave(2)[:, None]
        singular = torch.linalg.svdvals(design)
        # The final singular direction is the intended homogeneous nullspace.
        denominator = singular[-2]
    denominator_value = _canonical_geometry_scalar(denominator)
    if denominator_value <= EPS + GEOMETRY_DECISION_GUARD:
        return math.inf
    return _canonical_geometry_scalar(singular[0] / denominator_value)


def _fit_homography(reference: torch.Tensor, query: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    source, source_transform = _normalization(reference)
    target, target_transform = _normalization(query)
    x, y = source[:, 0], source[:, 1]
    u, v = target[:, 0], target[:, 1]
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    first = torch.stack((-x, -y, -one, zero, zero, zero, u * x, u * y, u), dim=1)
    second = torch.stack((zero, zero, zero, -x, -y, -one, v * x, v * y, v), dim=1)
    design = torch.stack((first, second), dim=1).reshape(-1, 9)
    design = design * weights.sqrt().repeat_interleave(2)[:, None]
    singular = _canonicalize_geometry(torch.linalg.svdvals(design))
    if float(singular[-2]) <= EPS + GEOMETRY_DECISION_GUARD:
        raise ValueError(REASON_FIT_DEGENERATE)
    _, _, vh = torch.linalg.svd(design, full_matrices=True)
    normalized = vh[-1].reshape(3, 3)
    matrix = torch.linalg.inv(target_transform) @ normalized @ source_transform
    if (
        not bool(torch.isfinite(matrix).all())
        or abs(_canonical_geometry_scalar(matrix[2, 2]))
        <= EPS + GEOMETRY_DECISION_GUARD
    ):
        raise ValueError(REASON_FIT_NONFINITE)
    return _canonicalize_geometry(matrix / matrix[2, 2])


def apply_geometry(matrix: torch.Tensor, reference_xy: torch.Tensor) -> torch.Tensor:
    value = _canonicalize_geometry(torch.as_tensor(matrix, dtype=torch.float64))
    points = torch.as_tensor(reference_xy, dtype=torch.float64)
    if value.shape != (3, 3) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("geometry application requires [3,3] and [N,2]")
    homogeneous = torch.cat((points, torch.ones((points.shape[0], 1), dtype=torch.float64, device=points.device)), dim=1)
    mapped = homogeneous @ value.to(points.device).T
    denominator = _canonicalize_geometry(mapped[:, 2])
    if not bool(torch.isfinite(mapped).all()) or bool(
        denominator.abs().le(EPS + GEOMETRY_DECISION_GUARD).any()
    ):
        raise ValueError(REASON_FIT_NONFINITE)
    result = mapped[:, :2] / denominator[:, None]
    if not bool(torch.isfinite(result).all()):
        raise ValueError(REASON_FIT_NONFINITE)
    return _canonicalize_geometry(result)


def _projected_area(matrix: torch.Tensor) -> tuple[float, bool]:
    corners = torch.tensor(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=torch.float64)
    mapped = apply_geometry(matrix, corners)
    shifted = torch.roll(mapped, shifts=-1, dims=0)
    area = 0.5 * (mapped[:, 0] * shifted[:, 1] - mapped[:, 1] * shifted[:, 0]).sum()
    signed_area = _canonical_geometry_scalar(area)
    return abs(signed_area), signed_area > GEOMETRY_DECISION_GUARD


def fit_fixed_geometry_hypotheses(
    slots: DiscreteGeometrySlots,
    *,
    config: GeometryFitConfig = GeometryFitConfig(),
    fit_anchor_mask: torch.Tensor | None = None,
) -> FixedGeometryHypotheses:
    """Fit one deterministic geometry per fixed slot; failures become H0."""

    selected = slots.anchor_mask.clone() if fit_anchor_mask is None else _mask(
        fit_anchor_mask,
        shape=tuple(slots.anchor_mask.shape),
        name="fit anchor mask",
    )
    if bool((selected & ~slots.anchor_mask).any()):
        raise ValueError("fit mask cannot introduce anchors")
    matrices = torch.eye(3, dtype=torch.float64).repeat(GEOMETRY_SLOTS, 1, 1)
    legal = torch.zeros(GEOMETRY_SLOTS, dtype=torch.bool)
    reasons = [REASON_H0] * GEOMETRY_SLOTS
    rmse = torch.zeros(GEOMETRY_SLOTS, dtype=torch.float64)
    maximum = torch.zeros(GEOMETRY_SLOTS, dtype=torch.float64)
    condition = torch.zeros(GEOMETRY_SLOTS, dtype=torch.float64)
    q_components = torch.zeros(GEOMETRY_SLOTS, dtype=torch.long)
    r_components = torch.zeros(GEOMETRY_SLOTS, dtype=torch.long)
    largest = torch.zeros((GEOMETRY_SLOTS, 2), dtype=torch.float64)
    compactness = torch.zeros((GEOMETRY_SLOTS, 2), dtype=torch.float64)
    diameter = torch.zeros((GEOMETRY_SLOTS, 2), dtype=torch.long)
    perimeter = torch.zeros((GEOMETRY_SLOTS, 2), dtype=torch.long)
    isoperimetric = torch.zeros((GEOMETRY_SLOTS, 2), dtype=torch.float64)
    connected = torch.zeros(GEOMETRY_SLOTS, dtype=torch.bool)
    query_centres = grid_cell_centres(slots.query_grid_shape)
    reference_centres = grid_cell_centres(slots.reference_grid_shape)
    minimum_anchors = MIN_AFFINE_ANCHORS if config.model == MODEL_AFFINE else MIN_HOMOGRAPHY_ANCHORS

    for slot in range(GEOMETRY_SLOTS):
        if not bool(slots.active[slot]):
            continue
        _, query_diagnostics = connected_components_4(slots.query_footprints[slot], slots.query_grid_shape)
        _, reference_diagnostics = connected_components_4(slots.reference_footprints[slot], slots.reference_grid_shape)
        q_components[slot] = query_diagnostics.component_count
        r_components[slot] = reference_diagnostics.component_count
        largest[slot] = torch.tensor((query_diagnostics.largest_component_fraction, reference_diagnostics.largest_component_fraction))
        compactness[slot] = torch.tensor((query_diagnostics.compactness, reference_diagnostics.compactness))
        diameter[slot] = torch.tensor((query_diagnostics.diameter, reference_diagnostics.diameter))
        perimeter[slot] = torch.tensor((query_diagnostics.perimeter, reference_diagnostics.perimeter))
        isoperimetric[slot] = torch.tensor(
            (query_diagnostics.isoperimetric, reference_diagnostics.isoperimetric),
            dtype=torch.float64,
        )
        connected[slot] = query_diagnostics.connected_valid and reference_diagnostics.connected_valid
        if not bool(connected[slot]):
            reasons[slot] = REASON_FOOTPRINT_DISCONNECTED
            continue
        if (
            query_diagnostics.active_count < config.minimum_footprint_cells
            or reference_diagnostics.active_count < config.minimum_footprint_cells
        ):
            reasons[slot] = REASON_FOOTPRINT_SPARSE
            continue
        query_area_fraction = query_diagnostics.active_count / math.prod(
            slots.query_grid_shape
        )
        reference_area_fraction = reference_diagnostics.active_count / math.prod(
            slots.reference_grid_shape
        )
        if not (
            config.minimum_area_fraction
            <= query_area_fraction
            <= config.maximum_area_fraction
            and config.minimum_area_fraction
            <= reference_area_fraction
            <= config.maximum_area_fraction
        ):
            reasons[slot] = REASON_FOOTPRINT_AREA
            continue
        if (
            query_diagnostics.compactness < config.minimum_compactness
            or reference_diagnostics.compactness < config.minimum_compactness
            or query_diagnostics.isoperimetric > config.maximum_isoperimetric
            or reference_diagnostics.isoperimetric > config.maximum_isoperimetric
        ):
            reasons[slot] = REASON_FOOTPRINT_SHAPE
            continue
        if not query_diagnostics.has_2d_span or not reference_diagnostics.has_2d_span:
            reasons[slot] = REASON_FOOTPRINT_NO_2D_SPAN
            continue
        active = selected[slot]
        if int(active.sum()) < minimum_anchors:
            reasons[slot] = REASON_INSUFFICIENT_ANCHORS
            continue
        query_indices = slots.anchor_query_indices[slot, active]
        reference_indices = slots.anchor_reference_indices[slot, active]
        if not _has_2d_span(query_indices, slots.query_grid_shape) or not _has_2d_span(reference_indices, slots.reference_grid_shape):
            reasons[slot] = REASON_ANCHOR_NO_2D_SPAN
            continue
        query = query_centres[query_indices]
        reference = reference_centres[reference_indices]
        weights = slots.anchor_weights[slot, active]
        weights = weights / weights.sum()
        try:
            matrix = (
                _fit_affine(reference, query, weights)
                if config.model == MODEL_AFFINE
                else _fit_homography(reference, query, weights)
            )
            if not bool(torch.isfinite(matrix).all()):
                reasons[slot] = REASON_FIT_NONFINITE
                continue
            matrix_condition = _geometry_design_condition(
                reference, query, weights, config.model
            )
            matrix_condition = _canonical_geometry_scalar(matrix_condition)
            if not _passes_upper_geometry_gate(
                matrix_condition, config.maximum_condition
            ):
                reasons[slot] = REASON_FIT_CONDITION
                continue
            projected = apply_geometry(matrix, reference)
            residual = torch.linalg.vector_norm(projected - query, dim=1)
            slot_rmse = _canonical_geometry_scalar(
                torch.sqrt((residual.square() * weights).sum())
            )
            slot_maximum = _canonical_geometry_scalar(residual.max())
            if not _passes_upper_geometry_gate(
                slot_rmse, config.maximum_reprojection_rmse
            ) or not _passes_upper_geometry_gate(
                slot_maximum, config.maximum_reprojection_error
            ):
                reasons[slot] = REASON_REPROJECTION
                continue
            area, orientation_preserving = _projected_area(matrix)
            if not _passes_geometry_interval(
                area,
                config.minimum_projected_area,
                config.maximum_projected_area,
            ):
                reasons[slot] = REASON_PROJECTED_AREA
                continue
            if config.require_orientation_preserving and not orientation_preserving:
                reasons[slot] = REASON_ORIENTATION
                continue
        except (RuntimeError, ValueError):
            reasons[slot] = REASON_FIT_DEGENERATE
            continue
        matrices[slot] = matrix
        legal[slot] = True
        reasons[slot] = REASON_LEGAL
        rmse[slot] = slot_rmse
        maximum[slot] = slot_maximum
        condition[slot] = matrix_condition

    return FixedGeometryHypotheses(
        matrices=matrices,
        legal=legal,
        reasons=tuple(reasons),
        fit_anchor_mask=selected,
        reprojection_rmse=rmse,
        reprojection_max=maximum,
        condition=condition,
        query_component_count=q_components,
        reference_component_count=r_components,
        largest_component_fraction=largest,
        compactness=compactness,
        diameter=diameter,
        perimeter=perimeter,
        isoperimetric=isoperimetric,
        connected_valid=connected,
        h0_log_evidence=torch.zeros(GEOMETRY_SLOTS, dtype=torch.float64),
        model=config.model,
    )


def rasterize_footprint_to_grid(
    footprint: torch.Tensor,
    source_shape: tuple[int, int],
    destination_shape: tuple[int, int],
) -> torch.Tensor:
    """Area-rasterize a sealed footprint between unequal P/V grids."""

    source_height, source_width = _grid_shape(source_shape, name="source grid")
    destination_height, destination_width = _grid_shape(destination_shape, name="destination grid")
    destination_count = destination_height * destination_width
    output = torch.zeros(destination_count, dtype=torch.bool)
    active = torch.nonzero(footprint, as_tuple=False).flatten()
    if active.numel() == 0:
        return output
    # Rasterize source-cell *areas*, rather than mapping only their centres.
    # Centre-only mapping breaks a connected 32x24 footprint into alternating
    # columns on a 36x48 V grid.  Positive-area overlap preserves the dense
    # component under both upsampling and downsampling.
    source_xy = _xy(active, (source_height, source_width))
    for x, y in source_xy.tolist():
        destination_left = int(math.floor(x * destination_width / source_width))
        destination_right = int(math.ceil((x + 1) * destination_width / source_width)) - 1
        destination_top = int(math.floor(y * destination_height / source_height))
        destination_bottom = int(math.ceil((y + 1) * destination_height / source_height)) - 1
        destination_left = max(0, min(destination_width - 1, destination_left))
        destination_right = max(0, min(destination_width - 1, destination_right))
        destination_top = max(0, min(destination_height - 1, destination_top))
        destination_bottom = max(0, min(destination_height - 1, destination_bottom))
        rows = torch.arange(destination_top, destination_bottom + 1)[:, None]
        columns = torch.arange(destination_left, destination_right + 1)[None, :]
        output[(rows * destination_width + columns).reshape(-1)] = True
    return output


def _dilate_4(mask: torch.Tensor, grid_shape: tuple[int, int], radius: int) -> torch.Tensor:
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("expansion radius must be a nonnegative integer")
    height, width = _grid_shape(grid_shape, name="dilation grid")
    result = torch.as_tensor(mask, dtype=torch.bool).reshape(height, width).clone()
    for _ in range(radius):
        prior = result.clone()
        result[1:] |= prior[:-1]
        result[:-1] |= prior[1:]
        result[:, 1:] |= prior[:, :-1]
        result[:, :-1] |= prior[:, 1:]
    return result.reshape(-1)


def _dilate_chebyshev(
    mask: torch.Tensor,
    grid_shape: tuple[int, int],
    radius: int,
) -> torch.Tensor:
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("halo radius must be a nonnegative integer")
    height, width = _grid_shape(grid_shape, name="halo grid")
    source = torch.as_tensor(mask, dtype=torch.bool).reshape(height, width)
    result = source.clone()
    for _ in range(radius):
        prior = result.clone()
        padded = torch.nn.functional.pad(prior, (1, 1, 1, 1), value=False)
        result = torch.zeros_like(prior)
        for y_offset in range(3):
            for x_offset in range(3):
                result |= padded[
                    y_offset : y_offset + height,
                    x_offset : x_offset + width,
                ]
    return result.reshape(-1)


@dataclass(frozen=True)
class CrossBankVerificationCoordinates:
    predicted_query_xy: torch.Tensor
    predicted_query_indices: torch.Tensor
    reference_indices: torch.Tensor
    verification_mask: torch.Tensor
    query_region_mask: torch.Tensor
    reference_region_mask: torch.Tensor
    query_verification_subset_mask: torch.Tensor
    reference_verification_subset_mask: torch.Tensor
    verification_legal: torch.Tensor
    h0_log_evidence: torch.Tensor

    def __post_init__(self) -> None:
        xy = _canonicalize_geometry(self.predicted_query_xy).detach().cpu().contiguous()
        query = torch.as_tensor(self.predicted_query_indices, dtype=torch.long).detach().cpu().contiguous()
        reference = torch.as_tensor(self.reference_indices, dtype=torch.long).detach().cpu().contiguous()
        mask = torch.as_tensor(self.verification_mask, dtype=torch.bool).detach().cpu().contiguous()
        query_region = torch.as_tensor(self.query_region_mask, dtype=torch.bool).detach().cpu().contiguous()
        reference_region = torch.as_tensor(self.reference_region_mask, dtype=torch.bool).detach().cpu().contiguous()
        query_subset = torch.as_tensor(self.query_verification_subset_mask, dtype=torch.bool).detach().cpu().contiguous()
        reference_subset = torch.as_tensor(self.reference_verification_subset_mask, dtype=torch.bool).detach().cpu().contiguous()
        legal = _mask(self.verification_legal, shape=(GEOMETRY_SLOTS,), name="verification legal")
        h0 = torch.as_tensor(self.h0_log_evidence, dtype=torch.float64).detach().cpu().contiguous()
        if xy.ndim != 3 or xy.shape[0] != GEOMETRY_SLOTS or xy.shape[2] != 2:
            raise ValueError("predicted coordinate schema drift")
        if query.shape != xy.shape[:2] or reference.shape != query.shape or mask.shape != query.shape:
            raise ValueError("cross-bank index schema drift")
        if query_region.ndim != 2 or query_region.shape[0] != GEOMETRY_SLOTS:
            raise ValueError("query region schema drift")
        if reference_region.shape != mask.shape:
            raise ValueError("reference region schema drift")
        if query_subset.shape != query_region.shape or reference_subset.shape != reference_region.shape:
            raise ValueError("verification subset schema drift")
        if bool((query_subset & ~query_region).any()) or bool((reference_subset & ~reference_region).any()):
            raise ValueError("verification subset must stay inside the dense footprint")
        if h0.shape != (GEOMETRY_SLOTS,) or bool(h0.ne(0).any()):
            raise ValueError("cross-bank H0 must be exact zero")
        illegal = ~legal
        if (
            bool(mask[illegal].any())
            or bool(query_region[illegal].any())
            or bool(reference_region[illegal].any())
            or bool(query_subset[illegal].any())
            or bool(reference_subset[illegal].any())
            or bool(query[illegal].ne(-1).any())
            or bool(reference[illegal].ne(-1).any())
            or bool(xy[illegal].ne(0).any())
        ):
            raise ValueError("verification-illegal slots must be exact H0")
        object.__setattr__(self, "predicted_query_xy", xy)
        object.__setattr__(self, "predicted_query_indices", query)
        object.__setattr__(self, "reference_indices", reference)
        object.__setattr__(self, "verification_mask", mask)
        object.__setattr__(self, "query_region_mask", query_region)
        object.__setattr__(self, "reference_region_mask", reference_region)
        object.__setattr__(self, "query_verification_subset_mask", query_subset)
        object.__setattr__(self, "reference_verification_subset_mask", reference_subset)
        object.__setattr__(self, "verification_legal", legal)
        object.__setattr__(self, "h0_log_evidence", h0)


def predict_cross_bank_verification_coordinates(
    slots: DiscreteGeometrySlots,
    hypotheses: FixedGeometryHypotheses,
    *,
    query_verification_grid_shape: tuple[int, int],
    reference_verification_grid_shape: tuple[int, int],
    verification_bank: int | None = None,
    region_expansion: int = 0,
    fit_exclusion_radius: int = 1,
    minimum_verification_cells: int = MIN_VERIFICATION_CELLS,
) -> CrossBankVerificationCoordinates:
    """Predict held-out V coordinates strictly inside the sealed footprint.

    The transform maps every eligible reference V cell to a query coordinate.
    A row survives only when both endpoints lie in the mapped footprint and
    neither endpoint belongs to the fit-anchor exclusion.  No descriptor or
    whole-image search is performed.
    """

    query_v_shape = _grid_shape(query_verification_grid_shape, name="query V grid")
    reference_v_shape = _grid_shape(reference_verification_grid_shape, name="reference V grid")
    if hypotheses.fit_anchor_mask.shape != slots.anchor_mask.shape:
        raise ValueError("geometry/support fit-mask schema drift")
    if verification_bank not in {None, 0, 1}:
        raise ValueError("verification bank must be None, zero, or one")
    if region_expansion != 0:
        raise ValueError(
            "V cannot expand a sealed footprint; use result-blind proposal padding before sealing"
        )
    if isinstance(minimum_verification_cells, bool) or not isinstance(minimum_verification_cells, int) or minimum_verification_cells < 1:
        raise ValueError("minimum verification cells must be positive")
    query_count = math.prod(query_v_shape)
    reference_count = math.prod(reference_v_shape)
    predicted_xy = torch.zeros((GEOMETRY_SLOTS, reference_count, 2), dtype=torch.float64)
    predicted_query = torch.full((GEOMETRY_SLOTS, reference_count), -1, dtype=torch.long)
    reference_indices = torch.full_like(predicted_query, -1)
    verification_mask = torch.zeros_like(predicted_query, dtype=torch.bool)
    query_regions = torch.zeros((GEOMETRY_SLOTS, query_count), dtype=torch.bool)
    reference_regions = torch.zeros((GEOMETRY_SLOTS, reference_count), dtype=torch.bool)
    query_subsets = torch.zeros((GEOMETRY_SLOTS, query_count), dtype=torch.bool)
    reference_subsets = torch.zeros((GEOMETRY_SLOTS, reference_count), dtype=torch.bool)
    verification_legal = torch.zeros(GEOMETRY_SLOTS, dtype=torch.bool)
    reference_centres = grid_cell_centres(reference_v_shape)
    q_height, q_width = query_v_shape
    for slot in range(GEOMETRY_SLOTS):
        if not bool(hypotheses.legal[slot]):
            continue
        query_region = rasterize_footprint_to_grid(slots.query_footprints[slot], slots.query_grid_shape, query_v_shape)
        reference_region = rasterize_footprint_to_grid(slots.reference_footprints[slot], slots.reference_grid_shape, reference_v_shape)
        query_region = _dilate_4(query_region, query_v_shape, region_expansion)
        reference_region = _dilate_4(reference_region, reference_v_shape, region_expansion)
        query_region_diagnostics = connected_components_4(query_region, query_v_shape)[1]
        reference_region_diagnostics = connected_components_4(reference_region, reference_v_shape)[1]
        if not query_region_diagnostics.connected_valid or not reference_region_diagnostics.connected_valid:
            # A remap that breaks the sealed component is not a legal V region.
            continue
        query_subset = query_region.clone()
        reference_subset = reference_region.clone()
        if verification_bank is not None:
            reference_bank_p = fixed_macro_bank_split(
                slots.reference_footprints[slot], slots.reference_grid_shape
            )
            selected_bank_p = (
                reference_bank_p.a if verification_bank == 0 else reference_bank_p.b
            )
            selected_bank_v = rasterize_footprint_to_grid(
                selected_bank_p,
                slots.reference_grid_shape,
                reference_v_shape,
            )
            reference_subset &= selected_bank_v

        active_fit = hypotheses.fit_anchor_mask[slot]
        fit_query = torch.zeros(query_count, dtype=torch.bool)
        fit_reference = torch.zeros(reference_count, dtype=torch.bool)
        if bool(active_fit.any()):
            fit_query_source = torch.zeros(math.prod(slots.query_grid_shape), dtype=torch.bool)
            fit_reference_source = torch.zeros(math.prod(slots.reference_grid_shape), dtype=torch.bool)
            fit_query_source[slots.anchor_query_indices[slot, active_fit]] = True
            fit_reference_source[slots.anchor_reference_indices[slot, active_fit]] = True
            fit_query_source = _dilate_chebyshev(
                fit_query_source,
                slots.query_grid_shape,
                fit_exclusion_radius,
            )
            fit_reference_source = _dilate_chebyshev(
                fit_reference_source,
                slots.reference_grid_shape,
                fit_exclusion_radius,
            )
            fit_query = rasterize_footprint_to_grid(fit_query_source, slots.query_grid_shape, query_v_shape)
            fit_reference = rasterize_footprint_to_grid(fit_reference_source, slots.reference_grid_shape, reference_v_shape)
        query_subset &= ~fit_query
        reference_subset &= ~fit_reference

        candidate_reference = torch.nonzero(reference_subset, as_tuple=False).flatten()
        if candidate_reference.numel() == 0:
            continue
        mapped = _canonicalize_geometry(
            apply_geometry(
                hypotheses.matrices[slot], reference_centres[candidate_reference]
            )
        )
        inside = (
            mapped[:, 0].gt(GEOMETRY_DECISION_GUARD)
            & mapped[:, 0].lt(1.0 - GEOMETRY_DECISION_GUARD)
            & mapped[:, 1].gt(GEOMETRY_DECISION_GUARD)
            & mapped[:, 1].lt(1.0 - GEOMETRY_DECISION_GUARD)
        )
        scaled_x = _canonicalize_geometry(mapped[:, 0] * q_width)
        scaled_y = _canonicalize_geometry(mapped[:, 1] * q_height)
        x = torch.floor(scaled_x).to(torch.long).clamp(0, q_width - 1)
        y = torch.floor(scaled_y).to(torch.long).clamp(0, q_height - 1)
        query_index = y * q_width + x
        allowed = inside & query_subset[query_index]
        selected_positions: list[int] = []
        seen_query: set[int] = set()
        for position in torch.nonzero(allowed, as_tuple=False).flatten().tolist():
            destination = int(query_index[position])
            if destination in seen_query:
                continue
            seen_query.add(destination)
            selected_positions.append(position)
        if len(selected_positions) < minimum_verification_cells:
            continue
        # Fixed geometry-only farthest-point traversal.  V similarity is not
        # available here and therefore cannot move a prediction to a nicer
        # looking cell.  At most eight real rows survive; remaining capacity is
        # represented by exact zero padding in the downstream fixed reducer.
        pool = torch.tensor(selected_positions, dtype=torch.long)
        pool_reference = candidate_reference[pool]
        pool_xy = _xy(pool_reference, reference_v_shape).to(torch.float64)
        fit_reference_indices = torch.nonzero(fit_reference, as_tuple=False).flatten()
        fit_reference_xy = _xy(fit_reference_indices, reference_v_shape).to(torch.float64)
        chosen_local: list[int] = []
        available = torch.ones(pool.numel(), dtype=torch.bool)
        while bool(available.any()) and len(chosen_local) < GEOMETRY_SLOTS:
            if chosen_local:
                selected_xy = pool_xy[torch.tensor(chosen_local)]
                distance = torch.cdist(pool_xy, selected_xy, p=2).min(dim=1).values
            elif fit_reference_xy.numel():
                distance = torch.cdist(pool_xy, fit_reference_xy, p=2).min(dim=1).values
            else:
                center = pool_xy.mean(dim=0, keepdim=True)
                distance = torch.cdist(pool_xy, center, p=2).squeeze(1)
            distance[~available] = -1.0
            maximum_distance = distance.max()
            tied = torch.nonzero(distance.eq(maximum_distance), as_tuple=False).flatten()
            # Canonical reference row breaks exact geometric ties.
            chosen = int(tied[pool_reference[tied].argmin()])
            chosen_local.append(chosen)
            available[chosen] = False
        selected_tensor = pool[torch.tensor(chosen_local)]
        output_rows = candidate_reference[selected_tensor]
        predicted_xy[slot, output_rows] = mapped[selected_tensor]
        predicted_query[slot, output_rows] = query_index[selected_tensor]
        reference_indices[slot, output_rows] = output_rows
        verification_mask[slot, output_rows] = True
        query_regions[slot] = query_region
        reference_regions[slot] = reference_region
        query_subsets[slot] = query_subset
        reference_subsets[slot] = reference_subset
        verification_legal[slot] = True

    return CrossBankVerificationCoordinates(
        predicted_query_xy=predicted_xy,
        predicted_query_indices=predicted_query,
        reference_indices=reference_indices,
        verification_mask=verification_mask,
        query_region_mask=query_regions,
        reference_region_mask=reference_regions,
        query_verification_subset_mask=query_subsets,
        reference_verification_subset_mask=reference_subsets,
        verification_legal=verification_legal,
        h0_log_evidence=torch.zeros(GEOMETRY_SLOTS, dtype=torch.float64),
    )


T = TypeVar("T")


def reorder_candidates(values: Sequence[T], order: torch.Tensor) -> tuple[T, ...]:
    """Joint candidate reorder; spatial/reference destruction is separate."""

    indices = torch.as_tensor(order, dtype=torch.long).detach().cpu().contiguous()
    if indices.shape != (len(values),) or not torch.equal(
        torch.sort(indices).values, torch.arange(len(values), dtype=torch.long)
    ):
        raise ValueError("candidate reorder must be a complete permutation")
    return tuple(values[int(index)] for index in indices)


def grid_symmetry_permutation(
    grid_shape: tuple[int, int],
    *,
    transform: Literal["horizontal_flip", "vertical_flip", "half_turn"],
) -> torch.Tensor:
    """Fixed topology-preserving spatial relabeling for equivariance tests."""

    height, width = _grid_shape(grid_shape, name="symmetry grid")
    index = torch.arange(height * width)
    y = torch.div(index, width, rounding_mode="floor")
    x = index.remainder(width)
    if transform == "horizontal_flip":
        x = width - 1 - x
    elif transform == "vertical_flip":
        y = height - 1 - y
    elif transform == "half_turn":
        x = width - 1 - x
        y = height - 1 - y
    else:
        raise ValueError("unregistered spatial symmetry")
    return y * width + x


def spatially_relabel_slots(
    slots: DiscreteGeometrySlots,
    *,
    query_permutation: torch.Tensor,
    reference_permutation: torch.Tensor,
) -> DiscreteGeometrySlots:
    """Jointly relabel anchors and footprints under grid graph automorphisms."""

    query_count = math.prod(slots.query_grid_shape)
    reference_count = math.prod(slots.reference_grid_shape)
    query_order = _indices(query_permutation, shape=(query_count,), name="query permutation")
    reference_order = _indices(reference_permutation, shape=(reference_count,), name="reference permutation")
    if not torch.equal(torch.sort(query_order).values, torch.arange(query_count)) or not torch.equal(
        torch.sort(reference_order).values, torch.arange(reference_count)
    ):
        raise ValueError("spatial relabeling must be bijective")

    def require_graph_automorphism(order: torch.Tensor, grid_shape: tuple[int, int], name: str) -> None:
        height, width = grid_shape
        index = torch.arange(height * width).reshape(height, width)
        source_edges = torch.cat(
            (
                torch.stack((index[:, :-1].reshape(-1), index[:, 1:].reshape(-1)), dim=1),
                torch.stack((index[:-1].reshape(-1), index[1:].reshape(-1)), dim=1),
            ),
            dim=0,
        )
        mapped = order[source_edges]
        mapped_xy = _xy(mapped.reshape(-1), grid_shape).reshape(-1, 2, 2)
        if not bool((mapped_xy[:, 0] - mapped_xy[:, 1]).abs().sum(dim=1).eq(1).all()):
            raise ValueError(f"{name} spatial relabeling must preserve the 4-neighbour graph")

    require_graph_automorphism(query_order, slots.query_grid_shape, "query")
    require_graph_automorphism(reference_order, slots.reference_grid_shape, "reference")

    def transform_index(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        active = result.ge(0)
        result[active] = order[result[active]]
        return result

    query_footprints = torch.zeros_like(slots.query_footprints)
    reference_footprints = torch.zeros_like(slots.reference_footprints)
    query_footprints[:, query_order] = slots.query_footprints
    reference_footprints[:, reference_order] = slots.reference_footprints
    return DiscreteGeometrySlots(
        seed_query_indices=transform_index(slots.seed_query_indices, query_order),
        seed_reference_indices=transform_index(slots.seed_reference_indices, reference_order),
        anchor_query_indices=transform_index(slots.anchor_query_indices, query_order),
        anchor_reference_indices=transform_index(slots.anchor_reference_indices, reference_order),
        anchor_weights=slots.anchor_weights,
        anchor_mask=slots.anchor_mask,
        query_footprints=query_footprints,
        reference_footprints=reference_footprints,
        query_grid_shape=slots.query_grid_shape,
        reference_grid_shape=slots.reference_grid_shape,
    )
