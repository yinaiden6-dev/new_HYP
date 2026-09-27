"""Parameter-free reference-generated superregion bridge.

This module is deliberately narrower than either the historical CW0 scorer or
the current DINO/GX path.  It consumes an already-computed, candidate-bound
``R0SoftCorrespondence`` and turns every frozen CW0 reference macro atom into
two cross-fit geometry proposals:

``A_TO_B`` fits on the atom's A cells and ``B_TO_A`` fits on its B cells.

For each direction, the candidate reference cells are the affine *source* and
their soft expected query coordinates are the target.  The affine is then
used to project the complete reference-cell areas into the query.  Only the
four-connected component containing the projected reference seed is retained.
Consequently, a query patch can enter a proposal only through an auditable
specific-reference cell projection; no query-only region bank or learned mask
head exists here.

The bridge has no trainable parameters, reads no labels/ranks/winner fields,
and does not call a verifier.  Its hard masks are detached visualization/E0
artifacts.  Upstream representation learning, downstream DINO/GX, and natural
scientific gates remain separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch

from .cw0_connected_region_v2 import enumerate_reference_macro_seeds
from .cw0_connected_window_v1 import (
    CW0_DEFAULT_RIDGE,
    CW0WeightedAffineFit,
    apply_affine,
    weighted_ridge_affine_fit,
)
from .geometry_hypothesis_v1 import (
    FootprintDiagnostics,
    connected_components_4,
    fixed_macro_bank_split,
    grid_cell_centres,
    project_reference_footprint_to_query,
)
from .r0_crossview_cycle_v1 import (
    R0SoftCorrespondence,
    soft_correspondence_with_dustbin,
)


RGH_REFERENCE_GENERATED_SUPERREGION_SCHEMA_VERSION = (
    "rc_rgh_reference_generated_superregion_v1"
)
RGH_DIRECTIONS = ("A_TO_B", "B_TO_A")
RGH_MINIMUM_TOTAL_MASS = 1.0e-10
RGH_MINIMUM_COVARIANCE_EIGENVALUE = 1.0e-12

RGH_REASON_LEGAL = "LEGAL"
RGH_REASON_ZERO_MASS = "ZERO_MASS"
RGH_REASON_DEGENERATE_FIT = "DEGENERATE_FIT"
RGH_REASON_NONFINITE_AFFINE = "NONFINITE_AFFINE"
RGH_REASON_SEED_OUTSIDE_QUERY = "SEED_OUTSIDE_QUERY"
RGH_REASON_SEED_NOT_PROJECTED = "SEED_NOT_PROJECTED"
RGH_REASON_SEED_COMPONENT_NO_2D_SPAN = "SEED_COMPONENT_NO_2D_SPAN"

RGHDirection = Literal["A_TO_B", "B_TO_A"]


def _positive_grid(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError(f"{name} must be a positive (height,width) tuple")
    return int(value[0]), int(value[1])


def _empty_diagnostics() -> FootprintDiagnostics:
    return FootprintDiagnostics(0, 0.0, 0.0, 0, 0, 0.0, False, False, 0)


@dataclass(frozen=True)
class RGHReferenceGeneratedSuperregion:
    """One candidate/reference-atom/direction visual proposal and provenance."""

    schema_version: str
    candidate_index: int
    reference_atom_ordinal: int
    direction: RGHDirection
    fit_bank: Literal["A", "B"]
    verify_bank: Literal["A", "B"]
    reference_seed_index: int
    query_seed_index: int
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    reference_footprint: torch.Tensor
    reference_fit_mask: torch.Tensor
    reference_verify_mask: torch.Tensor
    raw_projected_query_footprint: torch.Tensor
    query_footprint: torch.Tensor
    # [Q,R].  A true entry means this retained query cell is covered by the
    # affine projection of that exact specific-reference cell area.
    query_reference_provenance: torch.Tensor
    affine_fit: CW0WeightedAffineFit
    fit_reference_indices: torch.Tensor
    fit_expected_query_xy: torch.Tensor
    fit_weights: torch.Tensor
    projected_seed_xy: torch.Tensor
    raw_query_diagnostics: FootprintDiagnostics
    query_diagnostics: FootprintDiagnostics
    reference_diagnostics: FootprintDiagnostics
    legal: bool
    reason: str

    def __post_init__(self) -> None:
        if self.schema_version != RGH_REFERENCE_GENERATED_SUPERREGION_SCHEMA_VERSION:
            raise ValueError("RGH schema version drift")
        q_shape = _positive_grid(self.query_grid_shape, name="query grid")
        r_shape = _positive_grid(self.reference_grid_shape, name="reference grid")
        q_count, r_count = math.prod(q_shape), math.prod(r_shape)
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
            or isinstance(self.reference_atom_ordinal, bool)
            or not isinstance(self.reference_atom_ordinal, int)
            or self.reference_atom_ordinal < 0
        ):
            raise ValueError("candidate and reference-atom ordinals must be non-negative")
        if self.direction not in RGH_DIRECTIONS:
            raise ValueError("unknown RGH cross-fit direction")
        expected_banks = ("A", "B") if self.direction == "A_TO_B" else ("B", "A")
        if (self.fit_bank, self.verify_bank) != expected_banks:
            raise ValueError("RGH direction/bank contract drift")
        if not 0 <= self.reference_seed_index < r_count:
            raise ValueError("reference seed is out of range")
        if not -1 <= self.query_seed_index < q_count:
            raise ValueError("query seed is out of range")

        def mask(value: torch.Tensor, count: int, name: str) -> torch.Tensor:
            result = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous()
            if result.shape != (count,):
                raise ValueError(f"{name} shape drift")
            return result

        reference = mask(self.reference_footprint, r_count, "reference footprint")
        fit_mask = mask(self.reference_fit_mask, r_count, "reference fit mask")
        verify_mask = mask(self.reference_verify_mask, r_count, "reference verify mask")
        raw_query = mask(
            self.raw_projected_query_footprint, q_count, "raw projected query footprint"
        )
        query = mask(self.query_footprint, q_count, "query footprint")
        provenance = torch.as_tensor(
            self.query_reference_provenance, dtype=torch.bool
        ).detach().cpu().contiguous()
        if provenance.shape != (q_count, r_count):
            raise ValueError("query/reference provenance shape drift")
        if bool((fit_mask & verify_mask).any()) or not torch.equal(
            fit_mask | verify_mask, reference
        ):
            raise ValueError("A/B masks must exactly partition the full reference atom")
        if bool((query & ~raw_query).any()):
            raise ValueError("retained seed component escaped the raw projection")
        if not torch.equal(provenance.any(dim=1), query):
            raise ValueError("query footprint/provenance union drift")
        if bool(provenance[:, ~reference].any()) or bool(provenance[~query].any()):
            raise ValueError("provenance escaped the paired query/reference footprints")

        fit_indices = torch.as_tensor(
            self.fit_reference_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        fit_expected = torch.as_tensor(
            self.fit_expected_query_xy, dtype=torch.float64
        ).detach().cpu().contiguous()
        fit_weights = torch.as_tensor(
            self.fit_weights, dtype=torch.float64
        ).detach().cpu().contiguous()
        projected_seed = torch.as_tensor(
            self.projected_seed_xy, dtype=torch.float64
        ).detach().cpu().contiguous()
        if (
            fit_indices.ndim != 1
            or fit_expected.shape != (fit_indices.numel(), 2)
            or fit_weights.shape != (fit_indices.numel(),)
            or projected_seed.shape != (2,)
            or not torch.equal(fit_indices, torch.nonzero(fit_mask, as_tuple=False).flatten())
            or not bool(torch.isfinite(fit_expected).all())
            or not bool(torch.isfinite(fit_weights).all())
            or not bool(torch.isfinite(projected_seed).all())
            or bool(fit_weights.lt(0).any())
        ):
            raise ValueError("RGH fit provenance drift")

        _, expected_reference_diagnostics = connected_components_4(reference, r_shape)
        _, expected_raw_diagnostics = connected_components_4(raw_query, q_shape)
        _, expected_query_diagnostics = connected_components_4(query, q_shape)
        if self.reference_diagnostics != expected_reference_diagnostics:
            raise ValueError("reference diagnostics drift")
        if self.raw_query_diagnostics != expected_raw_diagnostics:
            raise ValueError("raw query diagnostics drift")
        if self.query_diagnostics != expected_query_diagnostics:
            raise ValueError("retained query diagnostics drift")
        if self.legal:
            if (
                self.reason != RGH_REASON_LEGAL
                or self.query_seed_index < 0
                or not bool(query[self.query_seed_index])
                or not self.query_diagnostics.connected_valid
                or not self.query_diagnostics.has_2d_span
                or not self.reference_diagnostics.connected_valid
                or not self.reference_diagnostics.has_2d_span
            ):
                raise ValueError("legal RGH proposal must be a paired connected 2-D region")
        elif self.reason == RGH_REASON_LEGAL:
            raise ValueError("illegal RGH proposal requires an explicit H0 reason")
        elif bool(query.any() or provenance.any()):
            raise ValueError("illegal RGH proposal must expose an exact empty query H0")

        object.__setattr__(self, "query_grid_shape", q_shape)
        object.__setattr__(self, "reference_grid_shape", r_shape)
        object.__setattr__(self, "reference_footprint", reference)
        object.__setattr__(self, "reference_fit_mask", fit_mask)
        object.__setattr__(self, "reference_verify_mask", verify_mask)
        object.__setattr__(self, "raw_projected_query_footprint", raw_query)
        object.__setattr__(self, "query_footprint", query)
        object.__setattr__(self, "query_reference_provenance", provenance)
        object.__setattr__(self, "fit_reference_indices", fit_indices)
        object.__setattr__(self, "fit_expected_query_xy", fit_expected)
        object.__setattr__(self, "fit_weights", fit_weights)
        object.__setattr__(self, "projected_seed_xy", projected_seed)


def build_rgh_soft_assignment(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shapes: Sequence[tuple[int, int]],
    reference_valid_mask: torch.Tensor,
    query_reliability: torch.Tensor,
    reference_reliability: torch.Tensor,
    dustbin_logit: torch.Tensor,
    temperature: float = 0.07,
) -> R0SoftCorrespondence:
    """Thin named bridge to the existing candidate-bound R0 assignment.

    No alternate matcher or hidden parameters are introduced by RGH.
    """

    return soft_correspondence_with_dustbin(
        query_tokens,
        reference_tokens,
        query_grid_shape=query_grid_shape,
        reference_grid_shapes=reference_grid_shapes,
        reference_valid_mask=reference_valid_mask,
        query_reliability=query_reliability,
        reference_reliability=reference_reliability,
        dustbin_logit=dustbin_logit,
        temperature=temperature,
    )


def _project_reference_cells_with_provenance(
    matrix: torch.Tensor,
    reference_footprint: torch.Tensor,
    *,
    reference_grid_shape: tuple[int, int],
    query_grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project full cell areas and retain the exact reference-cell lineage."""

    r_count = math.prod(reference_grid_shape)
    q_count = math.prod(query_grid_shape)
    reference = torch.as_tensor(reference_footprint, dtype=torch.bool).detach().cpu()
    full = project_reference_footprint_to_query(
        matrix,
        reference,
        reference_grid_shape=reference_grid_shape,
        query_grid_shape=query_grid_shape,
    )
    provenance = torch.zeros((q_count, r_count), dtype=torch.bool)
    for reference_index in torch.nonzero(reference, as_tuple=False).flatten().tolist():
        singleton = torch.zeros(r_count, dtype=torch.bool)
        singleton[reference_index] = True
        provenance[:, reference_index] = project_reference_footprint_to_query(
            matrix,
            singleton,
            reference_grid_shape=reference_grid_shape,
            query_grid_shape=query_grid_shape,
        )
    if not torch.equal(provenance.any(dim=1), full):
        raise RuntimeError("full-area projection and per-cell provenance do not close")
    return full, provenance


def _projected_seed_index(
    matrix: torch.Tensor,
    reference_seed_index: int,
    *,
    reference_grid_shape: tuple[int, int],
    query_grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, int]:
    reference_xy = grid_cell_centres(reference_grid_shape, dtype=torch.float64)
    projected = apply_affine(matrix.to(torch.float64), reference_xy[[reference_seed_index]])[0]
    if not bool(torch.isfinite(projected).all()):
        return projected, -1
    x, y = float(projected[0]), float(projected[1])
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        return projected, -1
    height, width = query_grid_shape
    column = min(width - 1, max(0, int(math.floor(x * width))))
    row = min(height - 1, max(0, int(math.floor(y * height))))
    return projected, row * width + column


def _retain_seed_component(
    raw_query: torch.Tensor,
    provenance: torch.Tensor,
    *,
    query_seed_index: int,
    query_grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, FootprintDiagnostics, str]:
    labels, _ = connected_components_4(raw_query, query_grid_shape)
    if query_seed_index < 0:
        empty = torch.zeros_like(raw_query)
        return empty, torch.zeros_like(provenance), _empty_diagnostics(), RGH_REASON_SEED_OUTSIDE_QUERY
    if not bool(raw_query[query_seed_index]):
        empty = torch.zeros_like(raw_query)
        return empty, torch.zeros_like(provenance), _empty_diagnostics(), RGH_REASON_SEED_NOT_PROJECTED
    component = labels.eq(int(labels[query_seed_index]))
    retained_provenance = provenance & component[:, None]
    retained = retained_provenance.any(dim=1)
    _, diagnostics = connected_components_4(retained, query_grid_shape)
    if not diagnostics.has_2d_span:
        empty = torch.zeros_like(raw_query)
        return (
            empty,
            torch.zeros_like(provenance),
            _empty_diagnostics(),
            RGH_REASON_SEED_COMPONENT_NO_2D_SPAN,
        )
    return retained, retained_provenance, diagnostics, RGH_REASON_LEGAL


def generate_reference_superregions_from_assignment(
    correspondence: R0SoftCorrespondence,
    *,
    candidate_index: int,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> tuple[RGHReferenceGeneratedSuperregion, ...]:
    """Generate all CW0 reference-atom proposals for one anonymous candidate.

    The input assignment is the only content-bearing input.  Candidate scores,
    ranks, labels, and target joins are neither accepted nor inferred.
    """

    q_shape = _positive_grid(query_grid_shape, name="query grid")
    r_shape = _positive_grid(reference_grid_shape, name="reference grid")
    q_count, r_count = math.prod(q_shape), math.prod(r_shape)
    expected = torch.as_tensor(correspondence.expected_query_xy)
    match_mass = torch.as_tensor(correspondence.reference_match_mass)
    if (
        expected.ndim != 3
        or expected.shape[2] != 2
        or match_mass.shape != expected.shape[:2]
        or not 0 <= candidate_index < expected.shape[0]
        or expected.shape[1] < r_count
        or correspondence.q_to_r.shape[1] != q_count
    ):
        raise ValueError("RGH soft-assignment/schema mismatch")

    candidate_expected = expected[candidate_index, :r_count].detach().cpu().to(torch.float64)
    candidate_mass = match_mass[candidate_index, :r_count].detach().cpu().to(torch.float64)
    reference_xy = grid_cell_centres(r_shape, dtype=torch.float64)
    output: list[RGHReferenceGeneratedSuperregion] = []
    for atom_ordinal, atom in enumerate(enumerate_reference_macro_seeds(r_shape)):
        reference_footprint = atom.window.mask
        split = fixed_macro_bank_split(reference_footprint, r_shape)
        _, reference_diagnostics = connected_components_4(reference_footprint, r_shape)
        for direction in RGH_DIRECTIONS:
            fit_bank, verify_bank = (("A", "B") if direction == "A_TO_B" else ("B", "A"))
            fit_mask = split.a if fit_bank == "A" else split.b
            verify_mask = split.b if verify_bank == "B" else split.a
            fit_indices = torch.nonzero(fit_mask, as_tuple=False).flatten()
            target_xy = candidate_expected[fit_indices]
            weights = candidate_mass[fit_indices]
            affine_fit = weighted_ridge_affine_fit(
                reference_xy[fit_indices], target_xy, weights, ridge=CW0_DEFAULT_RIDGE
            )

            raw_query = torch.zeros(q_count, dtype=torch.bool)
            provenance = torch.zeros((q_count, r_count), dtype=torch.bool)
            # Exact finite padding keeps an H0 record serializable and avoids
            # a diagnostic NaN becoming an accidental downstream feature.
            projected_seed = torch.zeros((2,), dtype=torch.float64)
            query_seed = -1
            retained = torch.zeros(q_count, dtype=torch.bool)
            retained_provenance = torch.zeros_like(provenance)
            retained_diagnostics = _empty_diagnostics()
            reason = RGH_REASON_LEGAL

            if float(affine_fit.total_mass) <= RGH_MINIMUM_TOTAL_MASS:
                reason = RGH_REASON_ZERO_MASS
            elif (
                float(affine_fit.source_min_eigenvalue)
                <= RGH_MINIMUM_COVARIANCE_EIGENVALUE
                or float(affine_fit.target_min_eigenvalue)
                <= RGH_MINIMUM_COVARIANCE_EIGENVALUE
            ):
                reason = RGH_REASON_DEGENERATE_FIT
            elif not bool(torch.isfinite(affine_fit.matrix).all()):
                reason = RGH_REASON_NONFINITE_AFFINE
            else:
                raw_query, provenance = _project_reference_cells_with_provenance(
                    affine_fit.matrix,
                    reference_footprint,
                    reference_grid_shape=r_shape,
                    query_grid_shape=q_shape,
                )
                projected_seed, query_seed = _projected_seed_index(
                    affine_fit.matrix,
                    atom.seed_index,
                    reference_grid_shape=r_shape,
                    query_grid_shape=q_shape,
                )
                retained, retained_provenance, retained_diagnostics, reason = (
                    _retain_seed_component(
                        raw_query,
                        provenance,
                        query_seed_index=query_seed,
                        query_grid_shape=q_shape,
                    )
                )
            _, raw_diagnostics = connected_components_4(raw_query, q_shape)
            legal = reason == RGH_REASON_LEGAL
            output.append(
                RGHReferenceGeneratedSuperregion(
                    schema_version=RGH_REFERENCE_GENERATED_SUPERREGION_SCHEMA_VERSION,
                    candidate_index=candidate_index,
                    reference_atom_ordinal=atom_ordinal,
                    direction=direction,
                    fit_bank=fit_bank,
                    verify_bank=verify_bank,
                    reference_seed_index=atom.seed_index,
                    query_seed_index=query_seed,
                    query_grid_shape=q_shape,
                    reference_grid_shape=r_shape,
                    reference_footprint=reference_footprint,
                    reference_fit_mask=fit_mask,
                    reference_verify_mask=verify_mask,
                    raw_projected_query_footprint=raw_query,
                    query_footprint=retained if legal else torch.zeros_like(retained),
                    query_reference_provenance=(
                        retained_provenance if legal else torch.zeros_like(retained_provenance)
                    ),
                    affine_fit=affine_fit,
                    fit_reference_indices=fit_indices,
                    fit_expected_query_xy=target_xy,
                    fit_weights=weights,
                    projected_seed_xy=projected_seed,
                    raw_query_diagnostics=raw_diagnostics,
                    query_diagnostics=(retained_diagnostics if legal else _empty_diagnostics()),
                    reference_diagnostics=reference_diagnostics,
                    legal=legal,
                    reason=reason,
                )
            )
    expected_count = 2 * len(enumerate_reference_macro_seeds(r_shape))
    if len(output) != expected_count:
        raise RuntimeError("RGH lost a frozen reference atom/direction")
    return tuple(output)


__all__ = [
    "RGH_DIRECTIONS",
    "RGH_REFERENCE_GENERATED_SUPERREGION_SCHEMA_VERSION",
    "RGHReferenceGeneratedSuperregion",
    "build_rgh_soft_assignment",
    "generate_reference_superregions_from_assignment",
]
