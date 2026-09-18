"""Natural candidate-binding qualification around the frozen R0 geometry core.

This module is deliberately a representation gate, not a retrieval policy.  It
adds no trainable parameters to :class:`R0CrossViewCycleCore` and never reads a
D1/AP score, candidate rank, winner flag, label, or identity in ``forward``.
Exact identities are used only by the authority freezer to exclude a donor
with the same corrected identity.  They are discarded before model forward.
The model receives one separately loaded donor-content tensor per destination;
those donors are not candidates and never receive a retrieval score.

The deployable response is content-null centred candidate by candidate::

    response(content, geometry_g)
      = core(content, geometry_g) - core(zero_content, geometry_g)

Consequently a zero-content candidate has exact zero response even when its
frozen geometry differs from another candidate's geometry.  REAL, candidate
binding C, and spatial P all use the same destination geometry and the same
zero-content baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import inspect
import math
from typing import Hashable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .r0_crossview_cycle_v1 import (
    DESCRIPTOR_DIM,
    R0CrossViewCycleCore,
    R0ForwardOutput,
    R0PrejoinGeometry,
    connected_slot_positive_barrier,
)


PAIR_TEMPERATURE = 0.10
CONTROL_TEMPERATURE = 0.10
CONTROL_MARGIN = 0.05
CONTROL_PAIR_WEIGHT = 0.50
CONNECTED_BARRIER_WEIGHT = 0.01


@dataclass(frozen=True)
class CandidateBindingPlan:
    """Sealed external-gallery C eligibility with no identity in its payload.

    Donor rows and source hashes live in the target-free manifest and are
    resolved before forward.  ``eligible[d]`` states whether the separately
    supplied donor tensor for destination ``d`` is authorized.  Ineligible
    destinations must carry exact-zero donor content.
    """

    namespace: str
    phase: str
    ledger_logical_sha256: str
    eligible: torch.Tensor  # [A]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or not self.namespace
            or self.phase not in {"TRAIN", "EVALUATION"}
            or self.phase not in self.namespace
            or not isinstance(self.ledger_logical_sha256, str)
            or len(self.ledger_logical_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.ledger_logical_sha256)
        ):
            raise ValueError("candidate-binding namespace/phase/ledger seal is invalid")
        eligible = torch.as_tensor(self.eligible, dtype=torch.bool).detach().cpu().contiguous()
        if eligible.ndim != 1 or eligible.numel() < 2:
            raise ValueError("candidate-binding plan must contain at least two candidates")
        object.__setattr__(self, "eligible", eligible)


def build_external_candidate_binding_plan(
    eligible: Sequence[bool] | torch.Tensor,
    *,
    namespace: str,
    phase: str,
    ledger_logical_sha256: str,
) -> CandidateBindingPlan:
    """Construct the forward-only plan from a sealed external donor ledger."""

    return CandidateBindingPlan(
        namespace=namespace,
        phase=phase,
        ledger_logical_sha256=ledger_logical_sha256,
        eligible=torch.as_tensor(eligible, dtype=torch.bool),
    )


def reorder_candidate_binding_plan(
    plan: CandidateBindingPlan, order: torch.Tensor
) -> CandidateBindingPlan:
    """Jointly reorder a sealed plan; ``order[new] == old``."""

    value = torch.as_tensor(order, dtype=torch.long).detach().cpu().contiguous()
    count = int(plan.eligible.numel())
    if value.shape != (count,) or not torch.equal(
        torch.sort(value).values, torch.arange(count)
    ):
        raise ValueError("candidate reorder must be a complete permutation")
    return CandidateBindingPlan(
        namespace=plan.namespace,
        phase=plan.phase,
        ledger_logical_sha256=plan.ledger_logical_sha256,
        eligible=plan.eligible[value],
    )


def validate_train_evaluation_candidate_separation(
    train: CandidateBindingPlan, evaluation: CandidateBindingPlan
) -> None:
    """Require phase-separated C controls with independently sealed ledgers."""

    if (
        train.phase != "TRAIN"
        or evaluation.phase != "EVALUATION"
        or train.namespace == evaluation.namespace
        or train.ledger_logical_sha256 == evaluation.ledger_logical_sha256
        or train.eligible.shape != evaluation.eligible.shape
    ):
        raise ValueError("TRAIN-C and EVALUATION-C contracts are not separated")


@dataclass(frozen=True)
class SpatialControlPlan:
    """Presealed target-free P permutations in an explicit phase namespace."""

    namespace: str
    phase: str
    reference_grid_shapes: tuple[tuple[int, int], ...]
    permutations: tuple[torch.Tensor, ...]
    eligible: torch.Tensor

    def __post_init__(self) -> None:
        if not self.namespace or self.phase not in {"TRAIN", "EVALUATION"}:
            raise ValueError("spatial control requires an explicit TRAIN/EVALUATION namespace")
        if self.phase not in self.namespace:
            raise ValueError("spatial namespace must encode its phase")
        eligible = torch.as_tensor(self.eligible, dtype=torch.bool).detach().cpu().contiguous()
        shapes = tuple(tuple(int(item) for item in shape) for shape in self.reference_grid_shapes)
        permutations = tuple(
            torch.as_tensor(value, dtype=torch.long).detach().cpu().contiguous()
            for value in self.permutations
        )
        if (
            len(permutations) < 2
            or len(shapes) != len(permutations)
            or eligible.shape != (len(permutations),)
        ):
            raise ValueError("one spatial permutation is required per compact candidate")
        for candidate, (shape, order) in enumerate(zip(shapes, permutations, strict=True)):
            if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
                raise ValueError("spatial plan grid shapes must be positive")
            count = math.prod(shape)
            if order.shape != (count,):
                raise ValueError("spatial permutation/grid shape mismatch")
            if not torch.equal(torch.sort(order).values, torch.arange(count)):
                raise ValueError("spatial control must be a complete candidate-local permutation")
            if bool(eligible[candidate]) and bool(order.eq(torch.arange(count)).any()):
                raise ValueError("eligible spatial destruction must have no fixed cells")
            if bool(eligible[candidate]) and spatial_permutation_is_affine(order, shape):
                raise ValueError("eligible spatial destruction must be non-affine")
        object.__setattr__(self, "reference_grid_shapes", shapes)
        object.__setattr__(self, "permutations", permutations)
        object.__setattr__(self, "eligible", eligible)


def spatial_permutation_is_affine(
    permutation: torch.Tensor, grid_shape: tuple[int, int]
) -> bool:
    """Exactly test whether a destination-to-source grid map is 2-D affine."""

    height, width = (int(grid_shape[0]), int(grid_shape[1]))
    order = torch.as_tensor(permutation, dtype=torch.long).detach().cpu()
    if height < 2 or width < 2 or order.shape != (height * width,):
        return True

    def xy(index: int) -> tuple[int, int]:
        return index % width, index // width

    origin = xy(int(order[0]))
    x_one = xy(int(order[1]))
    y_one = xy(int(order[width]))
    x_step = (x_one[0] - origin[0], x_one[1] - origin[1])
    y_step = (y_one[0] - origin[0], y_one[1] - origin[1])
    for destination, source in enumerate(order.tolist()):
        x = destination % width
        y = destination // width
        expected = (
            origin[0] + x * x_step[0] + y * y_step[0],
            origin[1] + x * x_step[1] + y * y_step[1],
        )
        if xy(source) != expected:
            return False
    return True


def _sattolo_permutation(count: int, key: bytes) -> torch.Tensor:
    """Hash-fixed one-cycle derangement, implemented without global RNG state."""

    values = list(range(count))
    counter = 0
    for index in range(count - 1, 0, -1):
        digest = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        source = int.from_bytes(digest[:8], "big") % index
        values[index], values[source] = values[source], values[index]
        counter += 1
    output = torch.tensor(values, dtype=torch.long)
    if bool(output.eq(torch.arange(count)).any()):  # Sattolo invariant
        raise RuntimeError("Sattolo construction produced a fixed point")
    return output


def build_spatial_control_plan(
    reference_grid_shapes: Sequence[tuple[int, int]],
    candidate_keys: Sequence[Hashable],
    *,
    namespace: str,
    phase: str,
) -> SpatialControlPlan:
    """Seal deterministic, phase-separated, no-fixed spatial permutations.

    The stable candidate key may be a physical-row receipt or another
    target-free key.  It is used only to seal the permutation and is not passed
    to model ``forward``.  Each path is a hash-fixed Sattolo derangement, not a
    global roll that an affine verifier could absorb.  If the first draw is
    affine (or the EVALUATION draw equals TRAIN), deterministic nonces are tried
    until a non-affine, phase-distinct permutation is found.
    """

    shapes = tuple(tuple(int(value) for value in shape) for shape in reference_grid_shapes)
    if len(shapes) < 2 or len(candidate_keys) != len(shapes):
        raise ValueError("one target-free key is required per compact candidate")
    if phase not in {"TRAIN", "EVALUATION"} or phase not in namespace:
        raise ValueError("spatial plan namespace/phase mismatch")
    permutations: list[torch.Tensor] = []
    eligible = torch.zeros(len(shapes), dtype=torch.bool)
    for candidate, (shape, key) in enumerate(zip(shapes, candidate_keys, strict=True)):
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("reference grid shapes must be positive (height,width) pairs")
        count = math.prod(shape)
        if shape[0] < 2 or shape[1] < 2 or count < 4:
            permutations.append(torch.arange(count, dtype=torch.long))
            continue
        base = f"{key!r}|{shape[0]}|{shape[1]}|R0_NATURAL_BINDING_P_V1".encode()
        train_order: torch.Tensor | None = None
        for nonce in range(256):
            trial = _sattolo_permutation(
                count, hashlib.sha256(base + b"|TRAIN|" + nonce.to_bytes(4, "big")).digest()
            )
            if not spatial_permutation_is_affine(trial, shape):
                train_order = trial
                break
        if train_order is None:  # pragma: no cover - combinatorially unreachable here
            raise RuntimeError("could not seal a non-affine TRAIN spatial control")
        order = train_order
        if phase == "EVALUATION":
            order = torch.empty(0, dtype=torch.long)
            for nonce in range(256):
                trial = _sattolo_permutation(
                    count,
                    hashlib.sha256(
                        base + b"|EVALUATION|" + nonce.to_bytes(4, "big")
                    ).digest(),
                )
                if (
                    not torch.equal(trial, train_order)
                    and not spatial_permutation_is_affine(trial, shape)
                ):
                    order = trial
                    break
            if order.numel() == 0:  # inverse of an n>2 cycle is distinct and no-fixed
                inverse = torch.empty_like(train_order)
                inverse[train_order] = torch.arange(count)
                if spatial_permutation_is_affine(inverse, shape):
                    raise RuntimeError("could not seal a distinct non-affine EVALUATION control")
                order = inverse
        permutations.append(order)
        eligible[candidate] = True
    return SpatialControlPlan(
        namespace=namespace,
        phase=phase,
        reference_grid_shapes=shapes,
        permutations=tuple(permutations),
        eligible=eligible,
    )


def reorder_spatial_control_plan(
    plan: SpatialControlPlan, order: torch.Tensor
) -> SpatialControlPlan:
    """Jointly reorder a presealed spatial control plan."""

    value = torch.as_tensor(order, dtype=torch.long).detach().cpu().contiguous()
    count = len(plan.permutations)
    if value.shape != (count,) or not torch.equal(
        torch.sort(value).values, torch.arange(count)
    ):
        raise ValueError("candidate reorder must be a complete permutation")
    return SpatialControlPlan(
        namespace=plan.namespace,
        phase=plan.phase,
        reference_grid_shapes=tuple(plan.reference_grid_shapes[int(index)] for index in value),
        permutations=tuple(plan.permutations[int(index)] for index in value),
        eligible=plan.eligible[value],
    )


def validate_train_evaluation_spatial_separation(
    train: SpatialControlPlan, evaluation: SpatialControlPlan
) -> None:
    """Require separate namespaces and distinct eligible P realizations."""

    if (
        train.phase != "TRAIN"
        or evaluation.phase != "EVALUATION"
        or train.namespace == evaluation.namespace
        or train.reference_grid_shapes != evaluation.reference_grid_shapes
        or len(train.permutations) != len(evaluation.permutations)
        or not torch.equal(train.eligible, evaluation.eligible)
    ):
        raise ValueError("TRAIN-P and EVALUATION-P contracts are not separated")
    for eligible, first, second in zip(
        train.eligible, train.permutations, evaluation.permutations, strict=True
    ):
        if bool(eligible) and torch.equal(first, second):
            raise ValueError("TRAIN-P and EVALUATION-P permutations must differ")


@dataclass(frozen=True)
class SelectedCandidateSubset:
    """Active candidates compacted from a right-padded output request."""

    reference_tokens: torch.Tensor  # [A,Rmax,D], canonical zero row padding
    geometry: R0PrejoinGeometry  # A active candidates only
    source_indices: torch.Tensor  # [A], indices in the source C population
    output_positions: torch.Tensor  # [A], positions in requested S capacity
    output_capacity: int


def subset_r0_geometry(
    geometry: R0PrejoinGeometry, source_indices: torch.Tensor
) -> R0PrejoinGeometry:
    """Materialize a compact geometry ledger without loading all references.

    This is the public runner boundary for sparse candidate loading.  A caller
    may first subset the frozen geometry by source row, then load/pad only the
    corresponding REAL candidates and any explicitly sealed C donors.  The
    returned ledger contains no source-row identifiers.
    """

    indices = source_indices.detach().cpu().to(torch.long)
    if (
        indices.ndim != 1
        or indices.numel() < 2
        or indices.unique().numel() != indices.numel()
        or bool(indices.lt(0).any() or indices.ge(geometry.candidate_count).any())
    ):
        raise ValueError("geometry subset requires at least two unique in-range indices")
    return R0PrejoinGeometry(
        predicted_query_xy=geometry.predicted_query_xy[indices],
        reference_indices=geometry.reference_indices[indices],
        verification_mask=geometry.verification_mask[indices],
        slot_legal=geometry.slot_legal[indices],
        query_footprints=geometry.query_footprints[indices],
        reference_footprints=geometry.reference_footprints[indices],
        affine_matrices=geometry.affine_matrices[indices],
        reference_valid_mask=geometry.reference_valid_mask[indices],
        query_grid_shape=geometry.query_grid_shape,
        reference_grid_shapes=tuple(
            geometry.reference_grid_shapes[int(index)] for index in indices
        ),
    )


def select_candidate_subset(
    reference_tokens: torch.Tensor,
    geometry: R0PrejoinGeometry,
    candidate_indices: torch.Tensor,
) -> SelectedCandidateSubset:
    """Compact a unique candidate subset and preserve exact output padding.

    ``candidate_indices`` is a one-dimensional fixed-capacity vector.  Active
    source candidate indices must come first; trailing ``-1`` values are
    canonical candidate-level padding.  Reference-row padding is independently
    canonicalized to exact zero from the geometry validity mask.
    """

    references = torch.as_tensor(reference_tokens)
    requested = torch.as_tensor(candidate_indices, dtype=torch.long).detach().cpu().contiguous()
    if (
        references.ndim != 3
        or references.shape[-1] != DESCRIPTOR_DIM
        or references.shape[:2] != geometry.reference_valid_mask.shape
        or requested.ndim != 1
        or requested.numel() < 2
    ):
        raise ValueError("candidate subset/reference schema drift")
    if not bool(torch.isfinite(references).all()):
        raise ValueError("candidate reference tokens must be finite")
    padding = requested.eq(-1)
    if bool(requested.lt(-1).any()):
        raise ValueError("candidate padding sentinel must be -1")
    first_padding = int(torch.nonzero(padding, as_tuple=False)[0]) if bool(padding.any()) else len(requested)
    if bool(requested[first_padding:].ne(-1).any()):
        raise ValueError("candidate padding must be canonical right padding")
    source = requested[:first_padding]
    if source.numel() < 2 or source.unique().numel() != source.numel():
        raise ValueError("at least two unique active candidates are required")
    if bool(source.lt(0).any() or source.ge(geometry.candidate_count).any()):
        raise ValueError("candidate subset index is out of range")

    compact_geometry = subset_r0_geometry(geometry, source)
    compact = references[source.to(references.device)]
    valid = compact_geometry.reference_valid_mask.to(compact.device)
    compact = torch.where(valid[..., None], compact, torch.zeros_like(compact))
    positions = torch.arange(source.numel(), dtype=torch.long)
    return SelectedCandidateSubset(
        reference_tokens=compact,
        geometry=compact_geometry,
        source_indices=source,
        output_positions=positions,
        output_capacity=int(requested.numel()),
    )


def _candidate_binding_content(
    donor_references: torch.Tensor,
    geometry: R0PrejoinGeometry,
    plan: CandidateBindingPlan,
) -> torch.Tensor:
    donors = torch.as_tensor(donor_references)
    count = geometry.candidate_count
    if (
        donors.ndim != 3
        or donors.shape[-1] != DESCRIPTOR_DIM
        or donors.shape[:2] != geometry.reference_valid_mask.shape
        or plan.eligible.shape != (count,)
        or not bool(torch.isfinite(donors).all())
    ):
        raise ValueError("candidate-binding plan does not match compact subset")
    valid = geometry.reference_valid_mask.to(donors.device)
    if bool(donors[~valid].ne(0).any()):
        raise ValueError("external donor reference padding must be exact zero")
    ineligible = ~plan.eligible.to(donors.device)
    if bool(ineligible.any()) and bool(donors[ineligible].ne(0).any()):
        raise ValueError("ineligible external C donors must be exact zero")
    output = torch.zeros_like(donors)
    for destination in range(count):
        if not bool(plan.eligible[destination]):
            continue
        cell_count = math.prod(geometry.reference_grid_shapes[destination])
        if not bool(valid[destination, :cell_count].all()):
            raise ValueError("destination geometry validity mask is inconsistent")
        output[destination, :cell_count] = donors[destination, :cell_count]
    return output


def no_fixed_spatial_control(
    references: torch.Tensor,
    geometry: R0PrejoinGeometry,
    plan: SpatialControlPlan,
) -> torch.Tensor:
    """Destroy reference content-to-coordinate binding with no fixed cell.

    A one-cell reference cannot support a no-fixed permutation and is returned
    as exact zero with ``eligible=False``.  Natural multi-patch references are
    cyclically shifted by one cell; this rule depends only on grid size and is
    therefore candidate-reorder equivariant.
    """

    if references.shape[:2] != geometry.reference_valid_mask.shape:
        raise ValueError("spatial control/reference geometry mismatch")
    if (
        len(plan.permutations) != references.shape[0]
        or plan.reference_grid_shapes != geometry.reference_grid_shapes
    ):
        raise ValueError("spatial control plan does not match compact subset")
    output = torch.zeros_like(references)
    for candidate, shape in enumerate(geometry.reference_grid_shapes):
        count = math.prod(shape)
        order = plan.permutations[candidate]
        if order.shape != (count,):
            raise ValueError("sealed spatial permutation/grid shape mismatch")
        if not bool(plan.eligible[candidate]):
            continue
        if bool(order.eq(torch.arange(count)).any()):
            raise ValueError("sealed spatial destruction contains a fixed cell")
        output[candidate, :count] = references[
            candidate, order.to(references.device)
        ]
    return output


def _scatter_compact(
    compact: torch.Tensor,
    positions: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    output = compact.new_zeros((capacity,) + tuple(compact.shape[1:]))
    return output.index_copy(0, positions.to(compact.device), compact)


@dataclass(frozen=True)
class R0NaturalBindingOutput:
    real_scores: torch.Tensor  # [S], content-null centred
    candidate_control_scores: torch.Tensor  # [S], same destination geometry
    spatial_control_scores: torch.Tensor  # [S], same destination geometry
    zero_content_scores: torch.Tensor  # [S], exact zero
    real_minus_candidate_control_scores: torch.Tensor  # [S]
    real_minus_spatial_control_scores: torch.Tensor  # [S]
    candidate_mask: torch.Tensor  # [S]
    candidate_binding_eligible: torch.Tensor  # [S]
    spatial_eligible: torch.Tensor  # [S]
    active_source_indices: torch.Tensor  # [A], receipt only
    active_output_positions: torch.Tensor  # [A]
    spatial_permutations: tuple[torch.Tensor, ...]
    compact_real_output: R0ForwardOutput
    compact_candidate_control_output: R0ForwardOutput
    compact_spatial_control_output: R0ForwardOutput


class R0NaturalBindingCore(nn.Module):
    """Parameter-neutral wrapper for REAL/C/P content-response qualification."""

    def __init__(self, *, temperature: float = 0.07, seed: int = 17) -> None:
        super().__init__()
        self.core = R0CrossViewCycleCore(temperature=temperature, seed=seed)

    def forward(
        self,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        candidate_control_tokens: torch.Tensor,
        geometry: R0PrejoinGeometry,
        candidate_indices: torch.Tensor,
        binding_plan: CandidateBindingPlan,
        spatial_plan: SpatialControlPlan,
    ) -> R0NaturalBindingOutput:
        subset = select_candidate_subset(reference_tokens, geometry, candidate_indices)
        references = subset.reference_tokens
        if binding_plan.eligible.numel() != references.shape[0]:
            raise ValueError("binding plan and active candidate subset disagree")

        binding_content = _candidate_binding_content(
            candidate_control_tokens, subset.geometry, binding_plan
        )
        if binding_content.dtype != references.dtype or binding_content.device != references.device:
            raise ValueError("external donor content must match REAL dtype and device")
        if len(spatial_plan.permutations) != references.shape[0]:
            raise ValueError("spatial plan and active candidate subset disagree")
        spatial_content = no_fixed_spatial_control(
            references, subset.geometry, spatial_plan
        )
        zero_content = torch.zeros_like(references)

        real_raw = self.core(query_tokens, references, subset.geometry)
        zero_raw = self.core(query_tokens, zero_content, subset.geometry)
        binding_raw = self.core(query_tokens, binding_content, subset.geometry)
        spatial_raw = self.core(query_tokens, spatial_content, subset.geometry)

        zero_response = zero_raw.scores - zero_raw.scores
        real_response = real_raw.scores - zero_raw.scores
        content_is_zero = references.eq(0).all(dim=(1, 2))
        real_response = torch.where(content_is_zero, zero_response, real_response)

        binding_response = binding_raw.scores - zero_raw.scores
        binding_same_as_real = binding_content.eq(references).all(dim=(1, 2))
        binding_response = torch.where(
            binding_same_as_real, real_response, binding_response
        )
        spatial_response = spatial_raw.scores - zero_raw.scores
        spatial_same_as_real = spatial_content.eq(references).all(dim=(1, 2))
        spatial_response = torch.where(
            spatial_same_as_real, real_response, spatial_response
        )

        capacity = subset.output_capacity
        positions = subset.output_positions
        candidate_mask = torch.zeros(capacity, dtype=torch.bool, device=real_response.device)
        candidate_mask[positions.to(candidate_mask.device)] = True
        binding_eligible_padded = torch.zeros_like(candidate_mask)
        binding_eligible_padded[positions.to(candidate_mask.device)] = binding_plan.eligible.to(
            candidate_mask.device
        )
        spatial_eligible_padded = torch.zeros_like(candidate_mask)
        spatial_eligible_padded[positions.to(candidate_mask.device)] = spatial_plan.eligible.to(
            candidate_mask.device
        )

        padded_real = _scatter_compact(real_response, positions, capacity)
        padded_binding = _scatter_compact(binding_response, positions, capacity)
        padded_spatial = _scatter_compact(spatial_response, positions, capacity)

        return R0NaturalBindingOutput(
            real_scores=padded_real,
            candidate_control_scores=padded_binding,
            spatial_control_scores=padded_spatial,
            zero_content_scores=_scatter_compact(
                zero_response, positions, capacity
            ),
            real_minus_candidate_control_scores=padded_real - padded_binding,
            real_minus_spatial_control_scores=padded_real - padded_spatial,
            candidate_mask=candidate_mask,
            candidate_binding_eligible=binding_eligible_padded,
            spatial_eligible=spatial_eligible_padded,
            active_source_indices=subset.source_indices,
            active_output_positions=positions,
            spatial_permutations=spatial_plan.permutations,
            compact_real_output=real_raw,
            compact_candidate_control_output=binding_raw,
            compact_spatial_control_output=spatial_raw,
        )


def sealed_pair_margin(scores: torch.Tensor, first_slot: int, second_slot: int) -> torch.Tensor:
    """Return a sealed ordered-pair margin; swapping slots negates it."""

    value = torch.as_tensor(scores)
    if value.ndim != 1 or not 0 <= first_slot < value.numel() or not 0 <= second_slot < value.numel():
        raise ValueError("pair slots are outside the candidate score vector")
    if first_slot == second_slot:
        raise ValueError("a candidate cannot compete with itself")
    return value[first_slot] - value[second_slot]


@dataclass(frozen=True)
class R0NaturalBindingLoss:
    total: torch.Tensor
    pairwise: torch.Tensor
    candidate_binding_pair: torch.Tensor
    spatial_pair: torch.Tensor
    connected_slot_barrier: torch.Tensor
    real_pair_margin: torch.Tensor
    candidate_control_pair_margin: torch.Tensor
    spatial_control_pair_margin: torch.Tensor
    real_minus_candidate_control_pair: torch.Tensor
    real_minus_spatial_control_pair: torch.Tensor
    compact_positive_index: int
    positive_repairable_directions: torch.Tensor


def natural_binding_loss(
    output: R0NaturalBindingOutput,
    positive_slot: int,
    rival_slot: int,
) -> R0NaturalBindingLoss:
    """Fixed successor loss: pair + positive REAL>C/P + local anti-collapse."""

    capacity = int(output.real_scores.numel())
    if not 0 <= positive_slot < capacity or not 0 <= rival_slot < capacity:
        raise ValueError("positive/rival slot is outside the candidate subset")
    if positive_slot == rival_slot:
        raise ValueError("positive and rival slots must differ")
    if not bool(output.candidate_mask[positive_slot] and output.candidate_mask[rival_slot]):
        raise ValueError("positive and rival must both be active, non-padding candidates")
    pair_slots = torch.tensor(
        [positive_slot, rival_slot], dtype=torch.long, device=output.real_scores.device
    )
    if not bool(output.candidate_binding_eligible[pair_slots].all()):
        raise ValueError("positive/rival pair lacks shape-matched identity-disjoint C donors")
    if not bool(output.spatial_eligible[pair_slots].all()):
        raise ValueError("positive/rival pair cannot support no-fixed spatial controls")

    positions = output.active_output_positions.tolist()
    try:
        compact_positive = positions.index(positive_slot)
    except ValueError as error:  # pragma: no cover - guarded by candidate_mask
        raise ValueError("positive candidate compact mapping is missing") from error

    real_pair = sealed_pair_margin(output.real_scores, positive_slot, rival_slot)
    binding_pair = sealed_pair_margin(
        output.candidate_control_scores, positive_slot, rival_slot
    )
    spatial_pair = sealed_pair_margin(
        output.spatial_control_scores, positive_slot, rival_slot
    )
    binding_pair_increment = real_pair - binding_pair
    spatial_pair_increment = real_pair - spatial_pair
    pairwise = F.softplus(-real_pair / PAIR_TEMPERATURE)
    candidate_binding_pair = F.softplus(
        (CONTROL_MARGIN - binding_pair_increment) / CONTROL_TEMPERATURE
    )
    spatial_pair_loss = F.softplus(
        (CONTROL_MARGIN - spatial_pair_increment) / CONTROL_TEMPERATURE
    )
    barrier_result = connected_slot_positive_barrier(
        output.compact_real_output, compact_positive
    )
    if not bool(barrier_result.repairable_directions.any()):
        raise ValueError("positive candidate is connected-slot coverage-ineligible")
    barrier = barrier_result.loss
    total = (
        pairwise
        + CONTROL_PAIR_WEIGHT * (candidate_binding_pair + spatial_pair_loss)
        + CONNECTED_BARRIER_WEIGHT * barrier
    )
    return R0NaturalBindingLoss(
        total=total,
        pairwise=pairwise,
        candidate_binding_pair=candidate_binding_pair,
        spatial_pair=spatial_pair_loss,
        connected_slot_barrier=barrier,
        real_pair_margin=real_pair,
        candidate_control_pair_margin=binding_pair,
        spatial_control_pair_margin=spatial_pair,
        real_minus_candidate_control_pair=binding_pair_increment,
        real_minus_spatial_control_pair=spatial_pair_increment,
        compact_positive_index=compact_positive,
        positive_repairable_directions=barrier_result.repairable_directions,
    )


def assert_natural_binding_forward_is_target_free() -> None:
    """Fail closed if a forbidden retrieval/label field enters forward."""

    forbidden = {"target", "label", "identity", "d1", "ap", "rank", "winner"}
    parameter_names = {
        name.lower() for name in inspect.signature(R0NaturalBindingCore.forward).parameters
    }
    output_names = {field.name.lower() for field in fields(R0NaturalBindingOutput)}
    observed = parameter_names | output_names
    if any(token in name for token in forbidden for name in observed):
        raise RuntimeError("R0 natural-binding forward schema contains a forbidden shortcut")


assert_natural_binding_forward_is_target_free()
