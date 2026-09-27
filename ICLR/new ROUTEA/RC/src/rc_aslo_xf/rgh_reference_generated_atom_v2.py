"""Cross-fitted evidence for reference-generated RGH atoms.

The V1 RGH bridge deliberately stopped at geometry visualization.  This
module supplies the missing differentiable, proposal-side atom score without
adding a backbone, assignment module, or scoring head.  It reuses the frozen
CW0 4,357-parameter correspondence family and evaluates each fixed reference
atom on reference endpoints that were not used to fit its affine map.

The score is *not* called a likelihood ratio.  A deployable zero point is
formed only by subtracting the same atom score recomputed under a separately
frozen matched spatial-null path.  Candidate binding and the formal spatial
destruction remain independently replayed evaluation controls.

Labels, ranks, winners, D1 values, DINO/GX tensors, and target masks are not
accepted by any inference function in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
from torch.nn import functional as F

from .cw0_connected_region_v2 import (
    CW0V2_CANDIDATE_TEMPERATURE,
    CW0V2_EPS,
    CW0V2_EVIDENCE_CLIP,
    CW0V2_MAX_AFFINE_CONDITION,
    CW0V2_MAX_REPROJECTION_ERROR,
    CW0V2_MAX_REPROJECTION_RMSE,
    CW0V2_MIN_COVARIANCE_EIGENVALUE,
    CW0V2_MIN_EFFECTIVE_MATCHES,
    CW0V2_MIN_UNIQUE_ENDPOINTS,
    CW0V2_RIDGE,
    CW0V2_ZERO_RETURN_CYCLE_PENALTY,
    enumerate_reference_macro_seeds,
    stable_reprojection_metrics,
)
from .cw0_connected_window_v1 import (
    CW0WeightedAffineFit,
    apply_affine,
    weighted_ridge_affine_fit,
)
from .geometry_hypothesis_v1 import fixed_macro_bank_split, grid_cell_centres
from .r0_crossview_cycle_v1 import R0SoftCorrespondence
from .rgh_reference_generated_superregion_v1 import (
    RGH_REASON_LEGAL,
    _project_reference_cells_with_provenance,
    _projected_seed_index,
    _retain_seed_component,
)


RGH_ATOM_EVIDENCE_SCHEMA_VERSION = "rc_rgh_reference_generated_atom_evidence_v2"
RGH_ATOM_DIRECTIONS = ("A_TO_B", "B_TO_A")
RGHAtomDirection = Literal["A_TO_B", "B_TO_A"]
RGH_REASON_HELDOUT_STRUCTURAL_H0 = "HELDOUT_STRUCTURAL_H0"


def polynomial_soft_clip(
    value: torch.Tensor, *, bound: float = CW0V2_EVIDENCE_CLIP
) -> torch.Tensor:
    """Monotone bounded robust clip with no finite-input flat gradient.

    Historical hard clipping is byte-compatible on synthetic in-range values
    but gives exactly zero gradient once natural reprojection errors push every
    held-out cell outside ``[-4,4]``.  ``b*x/(b+|x|)`` preserves sign and order,
    remains strictly inside the same bound, and has polynomial rather than
    exponential tails.  It is frozen for the new RGH V2 atom score only; no
    historical CW0 result is rewritten.
    """

    result = torch.as_tensor(value)
    if not result.is_floating_point() or not bool(torch.isfinite(result).all()):
        raise ValueError("RGH polynomial soft clip requires finite floating input")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("RGH polynomial soft-clip bound must be positive")
    return bound * result / (bound + result.abs())


def _grid(value: tuple[int, int], *, name: str) -> tuple[int, int]:
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
class RGHAtomDirectionEvidence:
    """One differentiable held-out score plus detached structural diagnostics."""

    schema_version: str
    candidate_index: int
    reference_atom_ordinal: int
    direction: RGHAtomDirection
    score: torch.Tensor
    affine_fit: CW0WeightedAffineFit
    fit_reference_indices: torch.Tensor
    verify_reference_indices: torch.Tensor
    heldout_signed_raw: torch.Tensor
    heldout_reliability: torch.Tensor
    heldout_evidence: torch.Tensor
    reciprocal_mass: torch.Tensor
    failure_mass: torch.Tensor
    reprojection_distance: torch.Tensor
    cycle_error: torch.Tensor
    conditional_entropy: torch.Tensor
    fit_unique_query_endpoints: int
    fit_unique_reference_endpoints: int
    verify_unique_query_endpoints: int
    verify_unique_reference_endpoints: int
    affine_condition: torch.Tensor
    reprojection_rmse: torch.Tensor
    maximum_reprojection_error: torch.Tensor
    affine_determinant: torch.Tensor
    hard_eligible: bool

    def __post_init__(self) -> None:
        if self.schema_version != RGH_ATOM_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("RGH atom evidence schema drift")
        if self.direction not in RGH_ATOM_DIRECTIONS:
            raise ValueError("RGH atom evidence direction drift")
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
            or isinstance(self.reference_atom_ordinal, bool)
            or not isinstance(self.reference_atom_ordinal, int)
            or self.reference_atom_ordinal < 0
        ):
            raise ValueError("RGH atom evidence ordinals must be non-negative")
        scalar_fields = (
            self.score,
            self.affine_condition,
            self.reprojection_rmse,
            self.maximum_reprojection_error,
            self.affine_determinant,
        )
        if any(
            torch.as_tensor(value).ndim != 0
            or not bool(torch.isfinite(torch.as_tensor(value)).all())
            for value in scalar_fields
        ):
            raise ValueError("RGH atom scalar diagnostics must be finite")
        vector_fields = (
            self.heldout_signed_raw,
            self.heldout_reliability,
            self.heldout_evidence,
            self.reciprocal_mass,
            self.failure_mass,
            self.reprojection_distance,
            self.cycle_error,
            self.conditional_entropy,
        )
        lengths = {int(torch.as_tensor(value).numel()) for value in vector_fields}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("RGH atom held-out vectors must share a nonzero length")
        if any(not bool(torch.isfinite(torch.as_tensor(value)).all()) for value in vector_fields):
            raise ValueError("RGH atom held-out diagnostics must be finite")
        fit = torch.as_tensor(self.fit_reference_indices, dtype=torch.long)
        verify = torch.as_tensor(self.verify_reference_indices, dtype=torch.long)
        if fit.ndim != 1 or verify.ndim != 1 or fit.numel() == 0 or verify.numel() == 0:
            raise ValueError("RGH atom fit/verify reference banks must be nonempty")
        if bool(torch.isin(fit, verify).any()):
            raise ValueError("RGH atom fit and verification reference banks overlap")
        if not isinstance(self.hard_eligible, bool):
            raise ValueError("RGH atom hard eligibility must be boolean")


@dataclass(frozen=True)
class RGHAtomEvidence:
    """The two complementary reference-endpoint directions for one atom."""

    candidate_index: int
    reference_atom_ordinal: int
    directions: tuple[RGHAtomDirectionEvidence, RGHAtomDirectionEvidence]

    def __post_init__(self) -> None:
        if len(self.directions) != 2 or tuple(item.direction for item in self.directions) != RGH_ATOM_DIRECTIONS:
            raise ValueError("RGH atom requires canonical A_TO_B/B_TO_A evidence")
        if any(
            item.candidate_index != self.candidate_index
            or item.reference_atom_ordinal != self.reference_atom_ordinal
            for item in self.directions
        ):
            raise ValueError("RGH atom directional binding drift")


@dataclass(frozen=True)
class RGHProjectedAtomV2:
    """A frozen paired atom generated with reciprocal/reliability-weighted fit."""

    candidate_index: int
    reference_atom_ordinal: int
    direction: RGHAtomDirection
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    reference_footprint: torch.Tensor
    query_footprint: torch.Tensor
    query_reference_provenance: torch.Tensor
    query_seed_index: int
    evidence: RGHAtomDirectionEvidence
    legal: bool
    reason: str

    def __post_init__(self) -> None:
        q_shape = _grid(self.query_grid_shape, name="query grid")
        r_shape = _grid(self.reference_grid_shape, name="reference grid")
        q_count, r_count = math.prod(q_shape), math.prod(r_shape)
        reference = torch.as_tensor(
            self.reference_footprint, dtype=torch.bool
        ).detach().cpu().contiguous()
        query = torch.as_tensor(
            self.query_footprint, dtype=torch.bool
        ).detach().cpu().contiguous()
        provenance = torch.as_tensor(
            self.query_reference_provenance, dtype=torch.bool
        ).detach().cpu().contiguous()
        if reference.shape != (r_count,) or query.shape != (q_count,):
            raise ValueError("RGH V2 projected-atom mask shape drift")
        if provenance.shape != (q_count, r_count):
            raise ValueError("RGH V2 projected-atom provenance shape drift")
        if not torch.equal(provenance.any(dim=1), query):
            raise ValueError("RGH V2 query/provenance union drift")
        if bool(provenance[:, ~reference].any() or provenance[~query].any()):
            raise ValueError("RGH V2 provenance escaped its paired atom")
        if (
            self.evidence.candidate_index != self.candidate_index
            or self.evidence.reference_atom_ordinal != self.reference_atom_ordinal
            or self.evidence.direction != self.direction
        ):
            raise ValueError("RGH V2 projected-atom/evidence binding drift")
        if self.legal:
            if (
                self.reason != RGH_REASON_LEGAL
                or not self.evidence.hard_eligible
                or not 0 <= self.query_seed_index < q_count
                or not bool(query[self.query_seed_index])
            ):
                raise ValueError("RGH V2 legal projected atom is not structurally closed")
        elif self.reason == RGH_REASON_LEGAL or bool(query.any() or provenance.any()):
            raise ValueError("RGH V2 H0 atom must expose exact empty query geometry")
        object.__setattr__(self, "query_grid_shape", q_shape)
        object.__setattr__(self, "reference_grid_shape", r_shape)
        object.__setattr__(self, "reference_footprint", reference)
        object.__setattr__(self, "query_footprint", query)
        object.__setattr__(self, "query_reference_provenance", provenance)

    @property
    def affine_fit(self) -> CW0WeightedAffineFit:
        return self.evidence.affine_fit


def _score_direction(
    correspondence: R0SoftCorrespondence,
    *,
    candidate_index: int,
    reference_atom_ordinal: int,
    direction: RGHAtomDirection,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    reference_coordinate_permutation: torch.Tensor | None,
) -> RGHAtomDirectionEvidence:
    q_shape = _grid(query_grid_shape, name="query grid")
    r_shape = _grid(reference_grid_shape, name="reference grid")
    q_count, r_count = math.prod(q_shape), math.prod(r_shape)
    pair = torch.as_tensor(correspondence.reciprocal_pair_mass)
    if (
        pair.ndim != 3
        or pair.shape[1] != q_count
        or pair.shape[2] < r_count
        or not 0 <= candidate_index < pair.shape[0]
    ):
        raise ValueError("RGH atom/correspondence shape drift")
    atoms = enumerate_reference_macro_seeds(r_shape)
    if not 0 <= reference_atom_ordinal < len(atoms):
        raise ValueError("RGH reference atom ordinal is out of range")
    atom = atoms[reference_atom_ordinal]
    split = fixed_macro_bank_split(atom.window.mask, r_shape)
    fit_mask, verify_mask = (
        (split.a, split.b) if direction == "A_TO_B" else (split.b, split.a)
    )
    fit_r = torch.nonzero(fit_mask.to(pair.device), as_tuple=False).flatten()
    verify_r = torch.nonzero(verify_mask.to(pair.device), as_tuple=False).flatten()
    if fit_r.numel() == 0 or verify_r.numel() == 0:
        raise ValueError("RGH reference atom cannot form complementary endpoint banks")

    q_xy = grid_cell_centres(q_shape, dtype=pair.dtype, device=pair.device)
    r_xy = grid_cell_centres(r_shape, dtype=pair.dtype, device=pair.device)
    if reference_coordinate_permutation is not None:
        permutation = torch.as_tensor(
            reference_coordinate_permutation, dtype=torch.long, device=pair.device
        )
        if (
            permutation.shape != (r_count,)
            or not torch.equal(
                permutation.sort().values,
                torch.arange(r_count, dtype=torch.long, device=pair.device),
            )
        ):
            raise ValueError("RGH reference-coordinate control is not a permutation")
        r_xy = r_xy[permutation]
    rho_r_fit = correspondence.reference_reliability[candidate_index, fit_r]
    fit_weight = pair[candidate_index, :, fit_r] * rho_r_fit[None, :]
    source = r_xy[fit_r][None, :, :].expand(q_count, -1, -1).reshape(-1, 2)
    target = q_xy[:, None, :].expand(-1, fit_r.numel(), -1).reshape(-1, 2)
    affine_fit = weighted_ridge_affine_fit(
        source, target, fit_weight.reshape(-1), ridge=CW0V2_RIDGE
    )

    predicted = apply_affine(affine_fit.matrix, r_xy[verify_r])
    reverse = correspondence.r_to_q[candidate_index, verify_r, :q_count]
    return_mass = reverse.sum(dim=1)
    zero_return = return_mass.eq(0.0)
    conditional = reverse / return_mass[:, None].clamp_min(CW0V2_EPS)
    expected_query = conditional @ q_xy
    (
        reprojection_squared,
        reprojection_distance,
        reprojection_rmse,
        maximum_reprojection_error,
    ) = stable_reprojection_metrics(expected_query - predicted, zero_return)
    squared_distance = torch.cdist(predicted, q_xy).square()
    conditional_cycle = (conditional * squared_distance).sum(dim=1)
    cycle_error = conditional_cycle + (1.0 - return_mass.clamp(0.0, 1.0))
    cycle_error = torch.where(
        zero_return,
        torch.full_like(cycle_error, CW0V2_ZERO_RETURN_CYCLE_PENALTY),
        cycle_error,
    )
    entropy = -(conditional * conditional.clamp_min(CW0V2_EPS).log()).sum(dim=1)
    if q_count > 1:
        entropy = entropy / math.log(q_count)
    else:
        entropy = entropy * 0.0

    forward = correspondence.q_to_r[candidate_index, :q_count, verify_r]
    verify_pair = forward * reverse.transpose(0, 1)
    reciprocal = verify_pair.sum(dim=0)
    dustbin = correspondence.reference_dustbin[candidate_index, verify_r]
    nonreciprocal = (return_mass - reciprocal).clamp_min(0.0)
    failure = dustbin + nonreciprocal

    rho_q = correspondence.query_reliability[candidate_index, :q_count]
    conditional_rho_q = (conditional * rho_q[None, :]).sum(dim=1)
    conditional_rho_q = torch.where(
        zero_return, rho_q.mean().expand_as(conditional_rho_q), conditional_rho_q
    )
    rho_r = correspondence.reference_reliability[candidate_index, verify_r]
    reliability = torch.sqrt((conditional_rho_q * rho_r).clamp_min(0.0))
    q_height, q_width = q_shape
    cell_scale_squared = (1.0 / q_height) ** 2 + (1.0 / q_width) ** 2
    signed_raw = (
        torch.log((reciprocal + CW0V2_EPS) / (failure + CW0V2_EPS))
        - reprojection_squared / cell_scale_squared
        - cycle_error / cell_scale_squared
        - entropy
    )
    signed_clipped = polynomial_soft_clip(signed_raw)
    evidence = reliability * signed_clipped
    score = evidence.sum() / float(verify_r.numel())

    fit_endpoint_counts = (
        int(fit_weight.sum(dim=1).detach().gt(0.0).sum()),
        int(fit_weight.sum(dim=0).detach().gt(0.0).sum()),
    )
    verify_endpoint_counts = (
        int(verify_pair.sum(dim=1).detach().gt(0.0).sum()),
        int(verify_pair.sum(dim=0).detach().gt(0.0).sum()),
    )
    linear = affine_fit.matrix[:2, :2]
    singular = torch.linalg.svdvals(linear)
    affine_condition = torch.where(
        singular[-1].gt(CW0V2_EPS),
        singular[0] / singular[-1].clamp_min(CW0V2_EPS),
        torch.full_like(singular[0], 1.0e30),
    )
    determinant = torch.linalg.det(linear)
    hard_eligible = bool(
        min(*fit_endpoint_counts, *verify_endpoint_counts) >= CW0V2_MIN_UNIQUE_ENDPOINTS
        and float(affine_fit.effective_matches.detach()) >= CW0V2_MIN_EFFECTIVE_MATCHES
        and float(affine_fit.source_min_eigenvalue.detach())
        >= CW0V2_MIN_COVARIANCE_EIGENVALUE
        and float(affine_fit.target_min_eigenvalue.detach())
        >= CW0V2_MIN_COVARIANCE_EIGENVALUE
        and float(affine_condition.detach()) < CW0V2_MAX_AFFINE_CONDITION
        and float(reprojection_rmse.detach()) < CW0V2_MAX_REPROJECTION_RMSE
        and float(maximum_reprojection_error.detach()) < CW0V2_MAX_REPROJECTION_ERROR
        and float(determinant.detach()) > 0.0
    )
    return RGHAtomDirectionEvidence(
        schema_version=RGH_ATOM_EVIDENCE_SCHEMA_VERSION,
        candidate_index=candidate_index,
        reference_atom_ordinal=reference_atom_ordinal,
        direction=direction,
        score=score,
        affine_fit=affine_fit,
        fit_reference_indices=fit_r,
        verify_reference_indices=verify_r,
        heldout_signed_raw=signed_raw,
        heldout_reliability=reliability,
        heldout_evidence=evidence,
        reciprocal_mass=reciprocal,
        failure_mass=failure,
        reprojection_distance=reprojection_distance,
        cycle_error=cycle_error,
        conditional_entropy=entropy,
        fit_unique_query_endpoints=fit_endpoint_counts[0],
        fit_unique_reference_endpoints=fit_endpoint_counts[1],
        verify_unique_query_endpoints=verify_endpoint_counts[0],
        verify_unique_reference_endpoints=verify_endpoint_counts[1],
        affine_condition=affine_condition,
        reprojection_rmse=reprojection_rmse,
        maximum_reprojection_error=maximum_reprojection_error,
        affine_determinant=determinant,
        hard_eligible=hard_eligible,
    )


def score_reference_generated_atoms(
    correspondence: R0SoftCorrespondence,
    *,
    candidate_index: int,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    reference_coordinate_permutation: torch.Tensor | None = None,
) -> tuple[RGHAtomEvidence, ...]:
    """Score the complete fixed reference-atom population for one candidate."""

    atoms = enumerate_reference_macro_seeds(_grid(reference_grid_shape, name="reference grid"))
    output: list[RGHAtomEvidence] = []
    for atom_ordinal in range(len(atoms)):
        directions = tuple(
            _score_direction(
                correspondence,
                candidate_index=candidate_index,
                reference_atom_ordinal=atom_ordinal,
                direction=direction,
                query_grid_shape=query_grid_shape,
                reference_grid_shape=reference_grid_shape,
                reference_coordinate_permutation=reference_coordinate_permutation,
            )
            for direction in RGH_ATOM_DIRECTIONS
        )
        output.append(
            RGHAtomEvidence(
                candidate_index=candidate_index,
                reference_atom_ordinal=atom_ordinal,
                directions=(directions[0], directions[1]),
            )
        )
    return tuple(output)


def generate_projected_atoms_v2(
    correspondence: R0SoftCorrespondence,
    *,
    candidate_index: int,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> tuple[RGHProjectedAtomV2, ...]:
    """Freeze hard projected atoms from the held-out-qualified V2 evidence.

    This function is for post-training geometry sealing.  Natural training
    uses the differentiable atom scores above and must not hard-prune an atom
    merely because an early checkpoint fails the detached geometry gate.
    """

    q_shape = _grid(query_grid_shape, name="query grid")
    r_shape = _grid(reference_grid_shape, name="reference grid")
    q_count, r_count = math.prod(q_shape), math.prod(r_shape)
    atoms = enumerate_reference_macro_seeds(r_shape)
    evidence = score_reference_generated_atoms(
        correspondence,
        candidate_index=candidate_index,
        query_grid_shape=q_shape,
        reference_grid_shape=r_shape,
    )
    output: list[RGHProjectedAtomV2] = []
    for atom_evidence in evidence:
        atom = atoms[atom_evidence.reference_atom_ordinal]
        reference = atom.window.mask
        for direction_evidence in atom_evidence.directions:
            query = torch.zeros(q_count, dtype=torch.bool)
            provenance = torch.zeros((q_count, r_count), dtype=torch.bool)
            query_seed = -1
            reason = RGH_REASON_HELDOUT_STRUCTURAL_H0
            if direction_evidence.hard_eligible:
                raw_query, raw_provenance = _project_reference_cells_with_provenance(
                    direction_evidence.affine_fit.matrix,
                    reference,
                    reference_grid_shape=r_shape,
                    query_grid_shape=q_shape,
                )
                _, query_seed = _projected_seed_index(
                    direction_evidence.affine_fit.matrix,
                    atom.seed_index,
                    reference_grid_shape=r_shape,
                    query_grid_shape=q_shape,
                )
                query, provenance, _, reason = _retain_seed_component(
                    raw_query,
                    raw_provenance,
                    query_seed_index=query_seed,
                    query_grid_shape=q_shape,
                )
            legal = direction_evidence.hard_eligible and reason == RGH_REASON_LEGAL
            if not legal:
                query = torch.zeros(q_count, dtype=torch.bool)
                provenance = torch.zeros((q_count, r_count), dtype=torch.bool)
            output.append(
                RGHProjectedAtomV2(
                    candidate_index=candidate_index,
                    reference_atom_ordinal=atom_evidence.reference_atom_ordinal,
                    direction=direction_evidence.direction,
                    query_grid_shape=q_shape,
                    reference_grid_shape=r_shape,
                    reference_footprint=reference,
                    query_footprint=query,
                    query_reference_provenance=provenance,
                    query_seed_index=query_seed,
                    evidence=direction_evidence,
                    legal=legal,
                    reason=(reason if legal else reason),
                )
            )
    if len(output) != 2 * len(atoms):
        raise RuntimeError("RGH V2 projected-atom axis lost a direction")
    return tuple(output)


def matched_null_direction_contrasts(
    real: Sequence[RGHAtomEvidence],
    matched_spatial_null: Sequence[RGHAtomEvidence],
) -> torch.Tensor:
    """Return ``[B,2]`` REAL-minus-training-null atom contrasts.

    The caller must construct ``matched_spatial_null`` by rebuilding the
    assignment under the separately registered training-only coordinate/token
    destruction.  This function never aliases the formal held-out P control.
    """

    real_items, null_items = tuple(real), tuple(matched_spatial_null)
    if len(real_items) == 0 or len(real_items) != len(null_items):
        raise ValueError("RGH REAL/null atom populations must be nonempty and aligned")
    rows: list[torch.Tensor] = []
    for ordinal, (real_atom, null_atom) in enumerate(zip(real_items, null_items)):
        if (
            real_atom.reference_atom_ordinal != ordinal
            or null_atom.reference_atom_ordinal != ordinal
            or real_atom.candidate_index != null_atom.candidate_index
        ):
            raise ValueError("RGH REAL/null atom axis drift")
        rows.append(
            torch.stack(
                tuple(
                    real_direction.score - null_direction.score
                    for real_direction, null_direction in zip(
                        real_atom.directions, null_atom.directions
                    )
                )
            )
        )
    return torch.stack(rows)


def normalized_two_direction_softmin(
    direction_contrasts: torch.Tensor,
    *,
    temperature: float = CW0V2_CANDIDATE_TEMPERATURE,
) -> torch.Tensor:
    """Smooth AND over the two directions; exact zero maps to exact zero."""

    value = torch.as_tensor(direction_contrasts)
    if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] == 0:
        raise ValueError("RGH direction contrast must be [B,2]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("RGH direction contrasts must be finite floating values")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("RGH direction soft-min temperature must be positive")
    output = -temperature * (
        torch.logsumexp(-value / temperature, dim=1) - math.log(2.0)
    )
    exact_zero = value.eq(0.0).all(dim=1)
    return torch.where(exact_zero, value.sum(dim=1) * 0.0, output)


def h0_uniform_atom_marginal(
    joint_atom_contrasts: torch.Tensor,
    *,
    temperature: float = CW0V2_CANDIDATE_TEMPERATURE,
) -> torch.Tensor:
    """MIL candidate evidence with fixed 1/2 H0 and uniform atom prior.

    ``all atom contrasts == 0`` is returned byte-exactly as zero.  A larger
    reference-atom population cannot acquire prior bonus because atom mass is
    normalized before mixing with H0.
    """

    value = torch.as_tensor(joint_atom_contrasts)
    if value.ndim != 1 or value.numel() == 0 or not value.is_floating_point():
        raise ValueError("RGH atom marginal requires a nonempty floating vector")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("RGH atom marginal received non-finite evidence")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("RGH atom marginal temperature must be positive")
    if bool(value.detach().eq(0.0).all()):
        return value.sum() * 0.0
    log_atom_mean = torch.logsumexp(value / temperature, dim=0) - math.log(
        value.numel()
    )
    log_half = torch.tensor(
        math.log(0.5), dtype=value.dtype, device=value.device
    )
    return temperature * torch.logaddexp(log_half, log_half + log_atom_mean)


@dataclass(frozen=True)
class RGHC128NaturalLoss:
    total: torch.Tensor
    listwise_rank: torch.Tensor
    target_above_h0: torch.Tensor
    training_binding_contrast: torch.Tensor


def rgh_c128_natural_loss(
    candidate_scores: torch.Tensor,
    *,
    target_index: int,
    target_training_c_bind_score: torch.Tensor,
) -> RGHC128NaturalLoss:
    """Frozen three-term natural loss without treating every rival as absent.

    Wrong candidates participate in exact-label ranking, but are not forced
    below H0 because a blind photograph may contain another gallery object.
    """

    if isinstance(target_index, bool) or not isinstance(target_index, int):
        raise ValueError("RGH target index must be an integer")

    return rgh_c128_natural_multitarget_loss(
        candidate_scores,
        target_indices=(target_index,),
        target_training_c_bind_score=target_training_c_bind_score,
    )


def rgh_c128_natural_multitarget_loss(
    candidate_scores: torch.Tensor,
    *,
    target_indices: Sequence[int],
    target_training_c_bind_score: torch.Tensor,
) -> RGHC128NaturalLoss:
    """Natural loss with a fixed uniform reduction over target-equivalent rows."""

    scores = torch.as_tensor(candidate_scores)
    c_bind = torch.as_tensor(
        target_training_c_bind_score, dtype=scores.dtype, device=scores.device
    )
    raw_indices = tuple(target_indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indices):
        raise ValueError("RGH target-equivalent indices must be integers")
    indices = tuple(int(index) for index in raw_indices)
    if (
        scores.ndim != 1
        or scores.numel() < 2
        or not scores.is_floating_point()
        or not bool(torch.isfinite(scores).all())
        or not indices
        or tuple(sorted(set(indices))) != indices
        or any(not 0 <= index < scores.numel() for index in indices)
        or c_bind.ndim != 0
        or not bool(torch.isfinite(c_bind))
    ):
        raise ValueError("RGH C128 natural-loss input drift")
    target_rows = scores[torch.tensor(indices, dtype=torch.long, device=scores.device)]
    target = torch.logsumexp(target_rows, dim=0) - math.log(len(indices))
    rival_mask = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    rival_mask[torch.tensor(indices, dtype=torch.long, device=scores.device)] = False
    rivals = scores[rival_mask]
    if rivals.numel() == 0:
        raise ValueError("RGH C128 loss requires at least one different-label rival")
    listwise = torch.logaddexp(target, torch.logsumexp(rivals, dim=0)) - target
    presence = F.softplus(-target)
    binding = F.softplus(c_bind - target)
    return RGHC128NaturalLoss(
        total=listwise + presence + binding,
        listwise_rank=listwise,
        target_above_h0=presence,
        training_binding_contrast=binding,
    )


__all__ = [
    "RGH_ATOM_DIRECTIONS",
    "RGH_ATOM_EVIDENCE_SCHEMA_VERSION",
    "RGHAtomDirectionEvidence",
    "RGHAtomEvidence",
    "RGHProjectedAtomV2",
    "RGHC128NaturalLoss",
    "polynomial_soft_clip",
    "score_reference_generated_atoms",
    "generate_projected_atoms_v2",
    "matched_null_direction_contrasts",
    "normalized_two_direction_softmin",
    "h0_uniform_atom_marginal",
    "rgh_c128_natural_loss",
    "rgh_c128_natural_multitarget_loss",
]
