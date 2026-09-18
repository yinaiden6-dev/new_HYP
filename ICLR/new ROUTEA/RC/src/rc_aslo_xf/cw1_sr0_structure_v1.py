"""CW1/SR0 result-blind connected super-region structural core.

This module contains no descriptor, score, label, candidate-rank, D1, natural
manifest, opened, or sealed input.  It reuses CW0's fixed local query tiles as
geometry atoms and constructs a separate, multiscale query target hypothesis
bank.  A target super-region is a fixed connected union of multiple atoms; it
is never assembled from positive evidence.

Only the SR0-E0-Structure contract is implemented here.  P/V scoring,
training objectives, natural data and retrieval actions are intentionally out
of scope and must fail closed in the stage runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Sequence

import torch

from .cw0_connected_region_v2 import enumerate_query_macro_seeds
from .geometry_hypothesis_v1 import connected_components_4


CW1_SR0_STRUCTURE_SCHEMA_VERSION = "rc_cw1_sr0_structure_core_v1"
CW1_SR0_MACRO_SIDE = 4
CW1_SR0_LOCAL_CONTROL_RADIUS = 0
CW1_SR0_TARGET_RADII = (1, 2, 3, 4)
CW1_SR0_ALL_RADII = (CW1_SR0_LOCAL_CONTROL_RADIUS,) + CW1_SR0_TARGET_RADII

STATUS_SINGLE = "SINGLE_CONNECTED_TARGET_CANDIDATE"
STATUS_H0 = "H0_HOLD"
STATUS_MULTI_TARGET = "MULTI_TARGET_AMBIGUOUS_HOLD"
STATUS_MULTI_REGION_SAME_CANDIDATE = "MULTI_REGION_SAME_CANDIDATE_HOLD"


def _grid_shape(value: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError("grid shape must be a positive (height,width) tuple")
    return int(value[0]), int(value[1])


def macro_grid_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = _grid_shape(grid_shape)
    return math.ceil(height / CW1_SR0_MACRO_SIDE), math.ceil(
        width / CW1_SR0_MACRO_SIDE
    )


@lru_cache(maxsize=32)
def _query_seeds(grid_shape: tuple[int, int]):
    """Cache the immutable result-blind local tile bank by grid shape."""

    return enumerate_query_macro_seeds(_grid_shape(grid_shape))


def _mask_sha256(mask: torch.Tensor, grid_shape: tuple[int, int]) -> str:
    shape = _grid_shape(grid_shape)
    value = torch.as_tensor(mask, dtype=torch.bool).detach().cpu().contiguous()
    if value.shape != (math.prod(shape),):
        raise ValueError("mask shape does not match grid")
    digest = hashlib.sha256()
    digest.update(CW1_SR0_STRUCTURE_SCHEMA_VERSION.encode("utf-8"))
    digest.update(json.dumps(shape, separators=(",", ":")).encode("utf-8"))
    digest.update(value.to(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _root_ball_ordinals(
    root_ordinal: int,
    radius: int,
    grid_shape: tuple[int, int],
) -> tuple[int, ...]:
    if (
        isinstance(radius, bool)
        or not isinstance(radius, int)
        or radius not in CW1_SR0_ALL_RADII
    ):
        raise ValueError("SR0 radius must be one of 0,1,2,3,4")
    macro_height, macro_width = macro_grid_shape(grid_shape)
    root_count = macro_height * macro_width
    if (
        isinstance(root_ordinal, bool)
        or not isinstance(root_ordinal, int)
        or not 0 <= root_ordinal < root_count
    ):
        raise ValueError("macro root ordinal is out of range")
    root_y, root_x = divmod(root_ordinal, macro_width)
    return tuple(
        ordinal
        for ordinal in range(root_count)
        if abs(divmod(ordinal, macro_width)[0] - root_y)
        + abs(divmod(ordinal, macro_width)[1] - root_x)
        <= radius
    )


def _union_and_overlap(
    grid_shape: tuple[int, int], root_ordinals: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seeds = _query_seeds(_grid_shape(grid_shape))
    roots = tuple(int(item) for item in root_ordinals)
    if not roots or tuple(sorted(set(roots))) != roots:
        raise ValueError("contributing roots must be a nonempty sorted unique tuple")
    if roots[0] < 0 or roots[-1] >= len(seeds):
        raise ValueError("contributing root ordinal is out of range")
    stacked = torch.stack([seeds[index].window.mask for index in roots]).to(
        dtype=torch.int64
    )
    coverage = stacked.sum(dim=0)
    mask = coverage.gt(0)
    reciprocal = torch.zeros_like(coverage, dtype=torch.float64)
    reciprocal[mask] = coverage[mask].to(torch.float64).reciprocal()
    tile_weights = torch.zeros(len(seeds), dtype=torch.float64)
    for ordinal in roots:
        tile_weights[ordinal] = reciprocal[seeds[ordinal].window.mask].sum()
    return mask, coverage, tile_weights


def exact_overlap_identity(
    grid_shape: tuple[int, int], root_ordinals: Sequence[int]
) -> bool:
    """Verify sum_u sum_{p in Tu} 1/c(p) == |union Tu| exactly.

    Python ``Fraction`` is used only for this structural validator so the
    identity is not weakened to a floating tolerance.  Runtime aggregation
    uses the equivalent float64 weights stored in each region.
    """

    roots = tuple(int(item) for item in root_ordinals)
    mask, coverage, _ = _union_and_overlap(grid_shape, roots)
    # Group identical denominators.  A cell covered by c tiles contributes c
    # incidences of 1/c, so this is exact while avoiding one Fraction per
    # tile-cell incidence.
    total = Fraction(0, 1)
    for count in torch.unique(coverage[mask]).tolist():
        cells = int(coverage[mask].eq(int(count)).sum())
        incidences = int(count) * cells
        total += Fraction(incidences, int(count))
    return total == Fraction(int(mask.sum()), 1)


def validate_connected_target_mask(
    mask: torch.Tensor, grid_shape: tuple[int, int]
) -> None:
    """Reject any supplied target membership that is not one dense 4CC."""

    shape = _grid_shape(grid_shape)
    value = torch.as_tensor(mask, dtype=torch.bool).detach().cpu().contiguous()
    if value.shape != (math.prod(shape),):
        raise ValueError("target mask shape does not match grid")
    diagnostics = connected_components_4(value, shape)[1]
    if (
        int(value.sum()) == 0
        or not diagnostics.connected_valid
        or diagnostics.component_count != 1
        or not diagnostics.has_2d_span
    ):
        raise ValueError("target super-region must be one 4CC with 2-D span")


@dataclass(frozen=True)
class CW1SR0SuperRegion:
    """One immutable, result-blind query target hypothesis."""

    grid_shape: tuple[int, int]
    radius: int
    canonical_root_ordinal: int
    canonical_root_seed_index: int
    source_root_ordinals: tuple[int, ...]
    contributing_root_ordinals: tuple[int, ...]
    mask: torch.Tensor
    member_indices: torch.Tensor
    coverage_count: torch.Tensor
    tile_weights: torch.Tensor
    mask_sha256: str
    reportable_target: bool

    def __post_init__(self) -> None:
        shape = _grid_shape(self.grid_shape)
        seeds = _query_seeds(shape)
        if self.radius not in CW1_SR0_ALL_RADII:
            raise ValueError("super-region radius is not frozen")
        sources = tuple(int(item) for item in self.source_root_ordinals)
        roots = tuple(int(item) for item in self.contributing_root_ordinals)
        if tuple(sorted(set(sources))) != sources or not sources:
            raise ValueError("source roots must be sorted, unique and nonempty")
        if self.canonical_root_ordinal != sources[0]:
            raise ValueError("canonical root must be the lowest source root")
        expected_roots = _root_ball_ordinals(
            self.canonical_root_ordinal, self.radius, shape
        )
        if roots != expected_roots:
            raise ValueError("contributing roots drifted from the fixed graph ball")
        expected_mask, expected_coverage, expected_weights = _union_and_overlap(
            shape, roots
        )
        mask = torch.as_tensor(self.mask, dtype=torch.bool).detach().cpu().contiguous()
        members = torch.as_tensor(
            self.member_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        coverage = torch.as_tensor(
            self.coverage_count, dtype=torch.long
        ).detach().cpu().contiguous()
        weights = torch.as_tensor(
            self.tile_weights, dtype=torch.float64
        ).detach().cpu().contiguous()
        if not torch.equal(mask, expected_mask):
            raise ValueError("super-region mask is not the canonical tile union")
        if not torch.equal(
            members, torch.nonzero(mask, as_tuple=False).flatten()
        ):
            raise ValueError("member indices disagree with the super-region mask")
        if not torch.equal(coverage, expected_coverage):
            raise ValueError("coverage count disagrees with constituent tiles")
        if not torch.equal(weights, expected_weights):
            raise ValueError("tile overlap weights disagree with the fixed formula")
        if self.canonical_root_seed_index != seeds[
            self.canonical_root_ordinal
        ].seed_index:
            raise ValueError("canonical macro root seed index drift")
        validate_connected_target_mask(mask, shape)
        if self.mask_sha256 != _mask_sha256(mask, shape):
            raise ValueError("super-region mask hash drift")
        if self.reportable_target != (self.radius in CW1_SR0_TARGET_RADII):
            raise ValueError("r0/control versus target role drift")
        if self.radius == CW1_SR0_LOCAL_CONTROL_RADIUS:
            if len(roots) != 1 or not torch.equal(mask, seeds[roots[0]].window.mask):
                raise ValueError("r0 must be exactly one local-tile control")
        else:
            if len(roots) < 2:
                raise ValueError("a target super-region needs multiple local tiles")
            largest_constituent = max(
                int(seeds[index].window.mask.sum()) for index in roots
            )
            if int(mask.sum()) <= largest_constituent:
                raise ValueError("target super-region must be larger than each tile")
        if not exact_overlap_identity(shape, roots):
            raise ValueError("exact unique-patch overlap identity failed")
        floating_error = abs(float(weights.sum()) - float(mask.sum()))
        if floating_error > 1.0e-12:
            raise ValueError("float64 overlap weights exceed frozen tolerance")
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "source_root_ordinals", sources)
        object.__setattr__(self, "contributing_root_ordinals", roots)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "member_indices", members)
        object.__setattr__(self, "coverage_count", coverage)
        object.__setattr__(self, "tile_weights", weights)

    @property
    def cell_count(self) -> int:
        return int(self.mask.sum())

    @property
    def aggregation_weights(self) -> torch.Tensor:
        """Return one row over all query atoms; the row sums to one."""

        return self.tile_weights / float(self.cell_count)


def _candidate_region(
    grid_shape: tuple[int, int], radius: int, root_ordinal: int
) -> dict[str, Any]:
    seeds = _query_seeds(grid_shape)
    roots = _root_ball_ordinals(root_ordinal, radius, grid_shape)
    mask, coverage, weights = _union_and_overlap(grid_shape, roots)
    return {
        "grid_shape": grid_shape,
        "radius": radius,
        "canonical_root_ordinal": root_ordinal,
        "canonical_root_seed_index": seeds[root_ordinal].seed_index,
        "source_root_ordinals": (root_ordinal,),
        "contributing_root_ordinals": roots,
        "mask": mask,
        "member_indices": torch.nonzero(mask, as_tuple=False).flatten(),
        "coverage_count": coverage,
        "tile_weights": weights,
        "mask_sha256": _mask_sha256(mask, grid_shape),
        "reportable_target": radius in CW1_SR0_TARGET_RADII,
    }


def enumerate_superregions_for_radius(
    grid_shape: tuple[int, int], radius: int
) -> tuple[CW1SR0SuperRegion, ...]:
    """Enumerate and structurally deduplicate one frozen radius level."""

    shape = _grid_shape(grid_shape)
    if radius not in CW1_SR0_ALL_RADII:
        raise ValueError("SR0 radius must be one of 0,1,2,3,4")
    root_count = math.prod(macro_grid_shape(shape))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for root in range(root_count):
        item = _candidate_region(shape, radius, root)
        key = item["mask_sha256"]
        if key in grouped and not torch.equal(grouped[key][0]["mask"], item["mask"]):
            raise RuntimeError("cryptographic mask hash collision")
        grouped.setdefault(key, []).append(item)
    regions: list[CW1SR0SuperRegion] = []
    for candidates in grouped.values():
        candidates.sort(key=lambda item: item["canonical_root_ordinal"])
        canonical = dict(candidates[0])
        canonical["source_root_ordinals"] = tuple(
            item["canonical_root_ordinal"] for item in candidates
        )
        regions.append(CW1SR0SuperRegion(**canonical))
    regions.sort(key=lambda item: item.canonical_root_ordinal)
    return tuple(regions)


def enumerate_superregion_bank(
    grid_shape: tuple[int, int], *, include_r0_control: bool = True
) -> tuple[CW1SR0SuperRegion, ...]:
    """Return the complete result-blind r0..4 structural bank."""

    radii = CW1_SR0_ALL_RADII if include_r0_control else CW1_SR0_TARGET_RADII
    return tuple(
        region
        for radius in radii
        for region in enumerate_superregions_for_radius(grid_shape, radius)
    )


def structural_aggregation_matrix(
    regions: Sequence[CW1SR0SuperRegion],
) -> torch.Tensor:
    """Return fixed overlap-corrected region-by-query-atom weights."""

    items = tuple(regions)
    if not items:
        raise ValueError("at least one super-region is required")
    shape = items[0].grid_shape
    atom_count = len(_query_seeds(shape))
    if any(item.grid_shape != shape for item in items):
        raise ValueError("all aggregation rows must share one query grid")
    matrix = torch.stack([item.aggregation_weights for item in items])
    if matrix.shape != (len(items), atom_count):
        raise RuntimeError("structural aggregation matrix shape drift")
    if not torch.allclose(
        matrix.sum(dim=1),
        torch.ones(len(items), dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("every structural aggregation row must sum to one")
    return matrix


def superregion_payload(region: CW1SR0SuperRegion) -> dict[str, Any]:
    return {
        "schema_version": CW1_SR0_STRUCTURE_SCHEMA_VERSION,
        "grid_shape": list(region.grid_shape),
        "radius": region.radius,
        "canonical_root_ordinal": region.canonical_root_ordinal,
        "canonical_root_seed_index": region.canonical_root_seed_index,
        "source_root_ordinals": list(region.source_root_ordinals),
        "contributing_root_ordinals": list(region.contributing_root_ordinals),
        "member_indices": region.member_indices.tolist(),
        "coverage_count": region.coverage_count.tolist(),
        "tile_weights_hex": [float(item).hex() for item in region.tile_weights],
        "mask_sha256": region.mask_sha256,
        "reportable_target": region.reportable_target,
        "target_free": True,
    }


def superregion_sha256(region: CW1SR0SuperRegion) -> str:
    rendered = json.dumps(
        superregion_payload(region), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def superregion_bank_sha256(regions: Sequence[CW1SR0SuperRegion]) -> str:
    payload = [superregion_sha256(item) for item in regions]
    rendered = json.dumps(payload, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CW1SR0SelectedRegion:
    """One reportable selected query region keyed by exact identity.

    ``physical_candidate_id`` remains a receipt field because multiple gallery
    rows may share one exact identity.  Ambiguity logic groups by
    ``exact_identity`` before comparing masks, so duplicate physical rows can
    never manufacture a multi-target result.
    """

    exact_identity: str
    physical_candidate_id: str
    region: CW1SR0SuperRegion

    def __post_init__(self) -> None:
        if not isinstance(self.exact_identity, str) or not self.exact_identity:
            raise ValueError("selected region needs a nonempty exact identity")
        if (
            not isinstance(self.physical_candidate_id, str)
            or not self.physical_candidate_id
        ):
            raise ValueError("selected region needs a physical candidate ID")
        if not isinstance(self.region, CW1SR0SuperRegion):
            raise TypeError("selected region must contain a CW1/SR0 region")
        if not self.region.reportable_target or self.region.radius == 0:
            raise ValueError("r0/local control cannot enter ambiguity decisions")


def macro_root_components(
    root_ordinals: Sequence[int], grid_shape: tuple[int, int]
) -> tuple[tuple[int, ...], ...]:
    """Return exact 4-neighbour components of a qualified-root set."""

    macro_height, macro_width = macro_grid_shape(grid_shape)
    root_count = macro_height * macro_width
    remaining = set(int(item) for item in root_ordinals)
    if any(item < 0 or item >= root_count for item in remaining):
        raise ValueError("qualified macro root is out of range")
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        component: list[int] = []
        while queue:
            current = queue.pop(0)
            component.append(current)
            y, x = divmod(current, macro_width)
            neighbours = []
            if y > 0:
                neighbours.append(current - macro_width)
            if x > 0:
                neighbours.append(current - 1)
            if x + 1 < macro_width:
                neighbours.append(current + 1)
            if y + 1 < macro_height:
                neighbours.append(current + macro_width)
            for neighbour in sorted(neighbours):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
        components.append(tuple(sorted(component)))
    components.sort(key=lambda item: item[0])
    return tuple(components)


def _masks_are_exactly_spatially_separate(
    left: CW1SR0SuperRegion, right: CW1SR0SuperRegion
) -> bool:
    """Return true only for nonoverlapping, non-4-adjacent complete masks."""

    if left.grid_shape != right.grid_shape:
        raise ValueError("selected query regions must share one grid")
    if bool((left.mask & right.mask).any()):
        return False
    diagnostics = connected_components_4(left.mask | right.mask, left.grid_shape)[1]
    # ``connected_valid`` is intentionally false for a two-component union;
    # both individual inputs have already passed the one-4CC/2-D constructor.
    return diagnostics.component_count == 2


def ambiguity_status(selections: Sequence[CW1SR0SelectedRegion]) -> str:
    """Classify exact spatial ambiguity after grouping physical rows by label.

    Different roots, candidates, or physical gallery rows do not imply two
    objects.  Two selected reportable masks count as spatially separate only
    when their intersection is empty and their union has exactly two 4CCs.
    Overlap and even disjoint-but-4-adjacent masks remain ordinary competition.
    A same-identity ambiguity additionally needs two byte-distinct masks.
    """

    items = tuple(selections)
    if not items:
        return STATUS_H0
    if any(not isinstance(item, CW1SR0SelectedRegion) for item in items):
        raise TypeError("ambiguity input must contain selected SR0 regions")
    grid_shape = items[0].region.grid_shape
    if any(item.region.grid_shape != grid_shape for item in items):
        raise ValueError("all selected query regions must share one grid")

    grouped: dict[str, list[CW1SR0SelectedRegion]] = {}
    for item in items:
        grouped.setdefault(item.exact_identity, []).append(item)
    for identity in grouped:
        grouped[identity].sort(
            key=lambda item: (
                item.region.mask_sha256,
                item.physical_candidate_id,
            )
        )

    # Same identity: duplicate rows or a duplicate mask are one explanation.
    # A HOLD needs two independent (byte-distinct) complete masks.
    for identity in sorted(grouped):
        candidates = grouped[identity]
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                if (
                    left.region.mask_sha256 != right.region.mask_sha256
                    and _masks_are_exactly_spatially_separate(
                        left.region, right.region
                    )
                ):
                    return STATUS_MULTI_REGION_SAME_CANDIDATE

    # Distinct exact identities are compared only after same-label grouping.
    identities = sorted(grouped)
    for left_index, left_identity in enumerate(identities):
        for right_identity in identities[left_index + 1 :]:
            for left in grouped[left_identity]:
                for right in grouped[right_identity]:
                    if _masks_are_exactly_spatially_separate(
                        left.region, right.region
                    ):
                        return STATUS_MULTI_TARGET
    return STATUS_SINGLE


__all__ = [
    "CW1_SR0_STRUCTURE_SCHEMA_VERSION",
    "CW1_SR0_MACRO_SIDE",
    "CW1_SR0_LOCAL_CONTROL_RADIUS",
    "CW1_SR0_TARGET_RADII",
    "CW1_SR0_ALL_RADII",
    "STATUS_SINGLE",
    "STATUS_H0",
    "STATUS_MULTI_TARGET",
    "STATUS_MULTI_REGION_SAME_CANDIDATE",
    "CW1SR0SuperRegion",
    "CW1SR0SelectedRegion",
    "macro_grid_shape",
    "exact_overlap_identity",
    "validate_connected_target_mask",
    "enumerate_superregions_for_radius",
    "enumerate_superregion_bank",
    "structural_aggregation_matrix",
    "superregion_payload",
    "superregion_sha256",
    "superregion_bank_sha256",
    "macro_root_components",
    "ambiguity_status",
]
