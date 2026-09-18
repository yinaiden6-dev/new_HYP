"""Target-free R0 cross-view coordinate/cycle primitives.

The R0 representation is deliberately narrower than a retrieval model.  It
adapts frozen 128-dimensional ColNomic spatial descriptors and asks whether a
*pre-existing, connected* candidate-conditioned affine hypothesis predicts
independent query/reference content.  It never selects a target, reads a D1
field, or turns a sparse correspondence into a target region.

There are three hard boundaries in this module:

* :class:`R0PrejoinGeometry` contains only result-blind geometry.  Every active
  slot owns one dense four-neighbour-connected query footprint and one dense
  four-neighbour-connected reference footprint.  A slot with fewer than four
  geometry-only held-out samples is canonical H0.
* :class:`R0CrossViewCycleCore` consumes that frozen schema.  Its soft
  correspondences contain a shared dustbin and produce per-patch reliability;
  background is therefore allowed to remain unmatched.
* :func:`smooth_candidate_pair_loss` is a separate post-join operation.  Labels
  cannot alter proposal coordinates, components, masks, or forward scores.

Reference grids may differ between candidates.  The tensor interface uses
right padding plus an exact validity mask; padding never participates in a
softmax, footprint, null, or score.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import math
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .geometry_hypothesis_v1 import (
    GEOMETRY_SLOTS,
    MIN_FOOTPRINT_CELLS,
    MIN_VERIFICATION_CELLS,
    CrossBankVerificationCoordinates,
    FixedGeometryHypotheses,
    connected_components_4,
    grid_cell_centres,
)


DESCRIPTOR_DIM = 128
ADAPTER_RANK = 8
DIRECTIONS = 2
VERIFICATION_SLOTS = 8
MAX_RESIDUAL_SCALE = 0.1
H0_LOG_EVIDENCE = 0.0
_EPS = 1.0e-12


def _positive_grid_shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError(f"{name} must be a positive (height,width) tuple")
    return int(value[0]), int(value[1])


def _canonical_tensor(value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous()


def _grid_index_from_normalized_xy(
    xy: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    height, width = _positive_grid_shape(grid_shape, name="coordinate grid")
    value = torch.as_tensor(xy, dtype=torch.float64)
    x = torch.floor(value[..., 0] * width).to(torch.long).clamp(0, width - 1)
    y = torch.floor(value[..., 1] * height).to(torch.long).clamp(0, height - 1)
    return y * width + x


@dataclass(frozen=True)
class R0PrejoinGeometry:
    """Fixed-capacity, target-free affine/connected-footprint ledger.

    Shapes are ``C`` candidates, two cross-fit directions, eight geometry
    slots, and eight verification rows.  Reference tensors use ``Rmax`` right
    padding because natural references do not share one grid shape.
    """

    predicted_query_xy: torch.Tensor  # [C,2,8,8,2]
    reference_indices: torch.Tensor  # [C,2,8,8], candidate-local indices
    verification_mask: torch.Tensor  # [C,2,8,8]
    slot_legal: torch.Tensor  # [C,2,8]
    query_footprints: torch.Tensor  # [C,2,8,Q]
    reference_footprints: torch.Tensor  # [C,2,8,Rmax]
    affine_matrices: torch.Tensor  # [C,2,8,3,3], illegal == identity
    reference_valid_mask: torch.Tensor  # [C,Rmax]
    query_grid_shape: tuple[int, int]
    reference_grid_shapes: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        query_shape = _positive_grid_shape(self.query_grid_shape, name="query grid")
        predicted = _canonical_tensor(self.predicted_query_xy, dtype=torch.float64)
        reference_indices = _canonical_tensor(self.reference_indices, dtype=torch.long)
        verification = _canonical_tensor(self.verification_mask, dtype=torch.bool)
        legal = _canonical_tensor(self.slot_legal, dtype=torch.bool)
        query_footprints = _canonical_tensor(self.query_footprints, dtype=torch.bool)
        reference_footprints = _canonical_tensor(self.reference_footprints, dtype=torch.bool)
        matrices = _canonical_tensor(self.affine_matrices, dtype=torch.float64)
        valid = _canonical_tensor(self.reference_valid_mask, dtype=torch.bool)

        if predicted.ndim != 5 or predicted.shape[1:] != (
            DIRECTIONS,
            GEOMETRY_SLOTS,
            VERIFICATION_SLOTS,
            2,
        ):
            raise ValueError("predicted query coordinates must be [C,2,8,8,2]")
        candidates = int(predicted.shape[0])
        compact_shape = (candidates, DIRECTIONS, GEOMETRY_SLOTS, VERIFICATION_SLOTS)
        if (
            reference_indices.shape != compact_shape
            or verification.shape != compact_shape
            or legal.shape != compact_shape[:-1]
        ):
            raise ValueError("R0 verification capacity schema drift")
        query_count = math.prod(query_shape)
        if query_footprints.shape != (
            candidates,
            DIRECTIONS,
            GEOMETRY_SLOTS,
            query_count,
        ):
            raise ValueError("query footprint schema drift")
        if valid.ndim != 2 or valid.shape[0] != candidates:
            raise ValueError("reference validity mask must be [C,Rmax]")
        reference_capacity = int(valid.shape[1])
        if reference_footprints.shape != (
            candidates,
            DIRECTIONS,
            GEOMETRY_SLOTS,
            reference_capacity,
        ):
            raise ValueError("reference footprint schema drift")
        if matrices.shape != (candidates, DIRECTIONS, GEOMETRY_SLOTS, 3, 3):
            raise ValueError("affine matrix schema drift")
        if len(self.reference_grid_shapes) != candidates:
            raise ValueError("one reference grid shape is required per candidate")
        shapes = tuple(
            _positive_grid_shape(shape, name=f"reference grid {index}")
            for index, shape in enumerate(self.reference_grid_shapes)
        )
        if not bool(torch.isfinite(predicted).all() and torch.isfinite(matrices).all()):
            raise ValueError("prejoin geometry must be finite")

        identity = torch.eye(3, dtype=torch.float64)
        for candidate, shape in enumerate(shapes):
            reference_count = math.prod(shape)
            expected_valid = torch.zeros(reference_capacity, dtype=torch.bool)
            expected_valid[:reference_count] = True
            if not torch.equal(valid[candidate], expected_valid):
                raise ValueError("reference padding must be canonical right padding")
            if bool(reference_footprints[candidate, ..., reference_count:].any()):
                raise ValueError("reference padding cannot enter a footprint")
            for direction in range(DIRECTIONS):
                for slot in range(GEOMETRY_SLOTS):
                    active = bool(legal[candidate, direction, slot])
                    row_mask = verification[candidate, direction, slot]
                    q_footprint = query_footprints[candidate, direction, slot]
                    r_footprint = reference_footprints[candidate, direction, slot, :reference_count]
                    matrix = matrices[candidate, direction, slot]
                    if not active:
                        if (
                            bool(row_mask.any())
                            or bool(q_footprint.any())
                            or bool(r_footprint.any())
                            or bool(reference_indices[candidate, direction, slot].ne(-1).any())
                            or bool(predicted[candidate, direction, slot].ne(0).any())
                            or not torch.equal(matrix, identity)
                        ):
                            raise ValueError("illegal R0 slot must be canonical exact H0")
                        continue

                    q_diagnostics = connected_components_4(q_footprint, query_shape)[1]
                    r_diagnostics = connected_components_4(r_footprint, shape)[1]
                    if (
                        not q_diagnostics.connected_valid
                        or not r_diagnostics.connected_valid
                        or q_diagnostics.component_count != 1
                        or r_diagnostics.component_count != 1
                        or q_diagnostics.active_count < MIN_FOOTPRINT_CELLS
                        or r_diagnostics.active_count < MIN_FOOTPRINT_CELLS
                        or not q_diagnostics.has_2d_span
                        or not r_diagnostics.has_2d_span
                    ):
                        raise ValueError(
                            "active evidence requires one dense connected 2-D footprint on both sides"
                        )
                    row_count = int(row_mask.sum())
                    if not MIN_VERIFICATION_CELLS <= row_count <= VERIFICATION_SLOTS:
                        raise ValueError(
                            "a seed or sparse patch list cannot authorize R0 evidence"
                        )
                    rows = reference_indices[candidate, direction, slot, row_mask]
                    if (
                        int(rows.min()) < 0
                        or int(rows.max()) >= reference_count
                        or rows.unique().numel() != rows.numel()
                        or not bool(r_footprint[rows].all())
                    ):
                        raise ValueError("held-out reference rows must be unique and inside R_r")
                    xy = predicted[candidate, direction, slot, row_mask]
                    if not bool(
                        (
                            xy.ge(0.0)
                            & xy.lt(1.0)
                        ).all()
                    ):
                        raise ValueError("predicted held-out coordinates must be inside the query grid")
                    query_rows = _grid_index_from_normalized_xy(xy, query_shape)
                    if not bool(q_footprint[query_rows].all()):
                        raise ValueError("held-out query evidence cannot leave connected R_q")
                    if (
                        bool(reference_indices[candidate, direction, slot, ~row_mask].ne(-1).any())
                        or bool(predicted[candidate, direction, slot, ~row_mask].ne(0).any())
                    ):
                        raise ValueError("unused verification capacity must be exact H0 padding")

        object.__setattr__(self, "predicted_query_xy", predicted)
        object.__setattr__(self, "reference_indices", reference_indices)
        object.__setattr__(self, "verification_mask", verification)
        object.__setattr__(self, "slot_legal", legal)
        object.__setattr__(self, "query_footprints", query_footprints)
        object.__setattr__(self, "reference_footprints", reference_footprints)
        object.__setattr__(self, "affine_matrices", matrices)
        object.__setattr__(self, "reference_valid_mask", valid)
        object.__setattr__(self, "query_grid_shape", query_shape)
        object.__setattr__(self, "reference_grid_shapes", shapes)

    @property
    def candidate_count(self) -> int:
        return int(self.predicted_query_xy.shape[0])

    @property
    def reference_capacity(self) -> int:
        return int(self.reference_valid_mask.shape[1])

    @classmethod
    def from_geometry_api(
        cls,
        candidate_directions: Sequence[
            Sequence[tuple[FixedGeometryHypotheses, CrossBankVerificationCoordinates]]
        ],
        *,
        query_grid_shape: tuple[int, int],
        reference_grid_shapes: Sequence[tuple[int, int]],
    ) -> "R0PrejoinGeometry":
        """Compact stable geometry API objects into the padded R0 schema."""

        candidates = len(candidate_directions)
        if candidates == 0 or len(reference_grid_shapes) != candidates:
            raise ValueError("candidate geometry cannot be empty or shape-misaligned")
        shapes = tuple(
            _positive_grid_shape(shape, name=f"reference grid {index}")
            for index, shape in enumerate(reference_grid_shapes)
        )
        reference_capacity = max(math.prod(shape) for shape in shapes)
        query_count = math.prod(_positive_grid_shape(query_grid_shape, name="query grid"))
        predicted = torch.zeros(
            candidates,
            DIRECTIONS,
            GEOMETRY_SLOTS,
            VERIFICATION_SLOTS,
            2,
            dtype=torch.float64,
        )
        reference_indices = torch.full(predicted.shape[:-1], -1, dtype=torch.long)
        verification = torch.zeros_like(reference_indices, dtype=torch.bool)
        legal = torch.zeros((candidates, DIRECTIONS, GEOMETRY_SLOTS), dtype=torch.bool)
        query_footprints = torch.zeros(
            (candidates, DIRECTIONS, GEOMETRY_SLOTS, query_count), dtype=torch.bool
        )
        reference_footprints = torch.zeros(
            (candidates, DIRECTIONS, GEOMETRY_SLOTS, reference_capacity), dtype=torch.bool
        )
        matrices = torch.eye(3, dtype=torch.float64).expand(
            candidates, DIRECTIONS, GEOMETRY_SLOTS, 3, 3
        ).clone()
        valid = torch.zeros((candidates, reference_capacity), dtype=torch.bool)

        for candidate, payloads in enumerate(candidate_directions):
            if len(payloads) != DIRECTIONS:
                raise ValueError("each candidate requires A-to-B and B-to-A payloads")
            reference_count = math.prod(shapes[candidate])
            valid[candidate, :reference_count] = True
            for direction, (hypotheses, coordinates) in enumerate(payloads):
                if coordinates.query_region_mask.shape[1] != query_count:
                    raise ValueError("query geometry API grid does not match R0 query grid")
                if coordinates.reference_region_mask.shape[1] != reference_count:
                    raise ValueError("reference geometry API grid does not match candidate grid")
                active = hypotheses.legal & coordinates.verification_legal
                for slot in range(GEOMETRY_SLOTS):
                    if not bool(active[slot]):
                        continue
                    rows = torch.nonzero(
                        coordinates.verification_mask[slot], as_tuple=False
                    ).flatten()
                    if not MIN_VERIFICATION_CELLS <= int(rows.numel()) <= VERIFICATION_SLOTS:
                        continue
                    legal[candidate, direction, slot] = True
                    matrices[candidate, direction, slot] = hypotheses.matrices[slot]
                    query_footprints[candidate, direction, slot] = coordinates.query_region_mask[slot]
                    reference_footprints[candidate, direction, slot, :reference_count] = (
                        coordinates.reference_region_mask[slot]
                    )
                    verification[candidate, direction, slot, : rows.numel()] = True
                    reference_indices[candidate, direction, slot, : rows.numel()] = rows
                    predicted[candidate, direction, slot, : rows.numel()] = (
                        coordinates.predicted_query_xy[slot, rows]
                    )
        return cls(
            predicted_query_xy=predicted,
            reference_indices=reference_indices,
            verification_mask=verification,
            slot_legal=legal,
            query_footprints=query_footprints,
            reference_footprints=reference_footprints,
            affine_matrices=matrices,
            reference_valid_mask=valid,
            query_grid_shape=query_grid_shape,
            reference_grid_shapes=shapes,
        )


class AsymmetricRank8ResidualAdapter(nn.Module):
    """Shared asymmetric low-rank branch with exact-identity initialization.

    The trainable rank-eight branch is recentered by its frozen seed value.
    Consequently the effective residual is bit-exact zero at construction,
    while every trainable factor (including the bounded scale) has a first-step
    gradient.  Recentring is identity-independent and is stored in the
    checkpoint as immutable buffers.
    """

    def __init__(
        self,
        descriptor_dim: int = DESCRIPTOR_DIM,
        rank: int = ADAPTER_RANK,
        *,
        seed: int = 17,
        initial_scale_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if descriptor_dim != DESCRIPTOR_DIM or rank != ADAPTER_RANK:
            raise ValueError("R0 freezes a 128-dimensional rank-8 adapter")
        if not 0.0 < initial_scale_fraction < 1.0:
            raise ValueError("initial scale fraction must lie strictly inside (0,1)")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        magnitude = 1.0 / math.sqrt(descriptor_dim)

        def factor(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.randn(shape, generator=generator, dtype=torch.float64) * magnitude

        self.query_u = nn.Parameter(factor((descriptor_dim, rank)))
        self.query_v = nn.Parameter(factor((rank, descriptor_dim)))
        self.reference_u = nn.Parameter(factor((descriptor_dim, rank)))
        self.reference_v = nn.Parameter(factor((rank, descriptor_dim)))
        alpha_value = math.atanh(initial_scale_fraction)
        self.query_alpha = nn.Parameter(torch.tensor(alpha_value, dtype=torch.float64))
        self.reference_alpha = nn.Parameter(torch.tensor(alpha_value, dtype=torch.float64))
        self.register_buffer("_query_u_seed", self.query_u.detach().clone(), persistent=True)
        self.register_buffer("_query_v_seed", self.query_v.detach().clone(), persistent=True)
        self.register_buffer("_reference_u_seed", self.reference_u.detach().clone(), persistent=True)
        self.register_buffer("_reference_v_seed", self.reference_v.detach().clone(), persistent=True)
        self.register_buffer(
            "_query_alpha_seed", self.query_alpha.detach().clone(), persistent=True
        )
        self.register_buffer(
            "_reference_alpha_seed", self.reference_alpha.detach().clone(), persistent=True
        )

    @staticmethod
    def _adapt(
        tokens: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        u_seed: torch.Tensor,
        v_seed: torch.Tensor,
        alpha_seed: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.as_tensor(tokens)
        if value.ndim < 2 or value.shape[-1] != DESCRIPTOR_DIM:
            raise ValueError("ColNomic spatial descriptors must end in dimension 128")
        dtype = value.dtype
        current = F.linear(F.linear(value, v.to(dtype)), u.to(dtype))
        seed = F.linear(F.linear(value, v_seed.to(dtype)), u_seed.to(dtype))
        current_scale = MAX_RESIDUAL_SCALE * torch.tanh(alpha.to(dtype))
        seed_scale = MAX_RESIDUAL_SCALE * torch.tanh(alpha_seed.to(dtype))
        residual = current_scale * current - seed_scale * seed
        return F.normalize(value + residual, dim=-1)

    def adapt_query(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._adapt(
            tokens,
            self.query_u,
            self.query_v,
            self.query_alpha,
            self._query_u_seed,
            self._query_v_seed,
            self._query_alpha_seed,
        )

    def adapt_reference(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._adapt(
            tokens,
            self.reference_u,
            self.reference_v,
            self.reference_alpha,
            self._reference_u_seed,
            self._reference_v_seed,
            self._reference_alpha_seed,
        )

    @property
    def bounded_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            MAX_RESIDUAL_SCALE * torch.tanh(self.query_alpha),
            MAX_RESIDUAL_SCALE * torch.tanh(self.reference_alpha),
        )


@dataclass(frozen=True)
class R0SoftCorrespondence:
    q_to_r: torch.Tensor  # [C,Q,Rmax], unconditional real mass
    r_to_q: torch.Tensor  # [C,Rmax,Q], unconditional real mass
    query_dustbin: torch.Tensor  # [C,Q]
    reference_dustbin: torch.Tensor  # [C,Rmax]
    query_reliability: torch.Tensor  # learned rho_q [C,Q]
    reference_reliability: torch.Tensor  # learned rho_r [C,Rmax]
    query_match_mass: torch.Tensor  # 1-dustbin [C,Q]
    reference_match_mass: torch.Tensor  # 1-dustbin [C,Rmax]
    expected_reference_xy: torch.Tensor  # [C,Q,2], conditional on match
    expected_query_xy: torch.Tensor  # [C,Rmax,2], conditional on match
    roundtrip_query_xy: torch.Tensor  # [C,Q,2]
    query_cycle_displacement: torch.Tensor  # [C,Q]
    query_cycle_squared_displacement: torch.Tensor  # E[||q'-q||^2], [C,Q]
    reciprocal_mass: torch.Tensor  # [C,Q]
    conditional_query_entropy: torch.Tensor  # [C,Q]
    conditional_reference_entropy: torch.Tensor  # [C,Rmax]
    reciprocal_pair_mass: torch.Tensor  # [C,Q,Rmax], A_ij B_ji


def _padded_reference_coordinates(
    grid_shapes: Sequence[tuple[int, int]],
    capacity: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    output = torch.zeros((len(grid_shapes), capacity, 2), dtype=dtype, device=device)
    for candidate, shape in enumerate(grid_shapes):
        coordinates = grid_cell_centres(shape, dtype=dtype, device=device)
        output[candidate, : coordinates.shape[0]] = coordinates
    return output


def soft_correspondence_with_dustbin(
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
    """Build differentiable, candidate-bound soft Q<->R correspondences.

    The shared dustbin logit includes ``log(number_of_real_cells)``.  Uniform
    weak similarities therefore do not acquire near-one reliability merely
    because a reference contains hundreds of cells.
    """

    query = torch.as_tensor(query_tokens)
    reference = torch.as_tensor(reference_tokens)
    valid = torch.as_tensor(reference_valid_mask, dtype=torch.bool, device=reference.device)
    query_shape = _positive_grid_shape(query_grid_shape, name="query grid")
    if (
        query.ndim != 2
        or reference.ndim != 3
        or query.shape[-1] != DESCRIPTOR_DIM
        or reference.shape[-1] != DESCRIPTOR_DIM
        or query.shape[0] != math.prod(query_shape)
        or valid.shape != reference.shape[:2]
        or len(reference_grid_shapes) != reference.shape[0]
    ):
        raise ValueError("soft correspondence input schema drift")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    rho_q = torch.as_tensor(query_reliability, dtype=query.dtype, device=query.device)
    rho_r = torch.as_tensor(reference_reliability, dtype=query.dtype, device=query.device)
    b_null = torch.as_tensor(dustbin_logit, dtype=query.dtype, device=query.device)
    if rho_q.shape != (query.shape[0],) or rho_r.shape != reference.shape[:2]:
        raise ValueError("patch reliability schema drift")
    if (
        not bool(torch.isfinite(rho_q).all() and torch.isfinite(rho_r).all())
        or bool(rho_q.le(0).any() or rho_q.ge(1).any())
        or bool(rho_r[valid].le(0).any() or rho_r[valid].ge(1).any())
        or b_null.numel() != 1
        or not bool(torch.isfinite(b_null).all())
    ):
        raise ValueError("reliability must lie strictly inside (0,1) and b_null be finite")
    for candidate, shape in enumerate(reference_grid_shapes):
        count = math.prod(_positive_grid_shape(shape, name="reference grid"))
        expected = torch.zeros(reference.shape[1], dtype=torch.bool, device=valid.device)
        expected[:count] = True
        if not torch.equal(valid[candidate], expected):
            raise ValueError("reference validity mask/grid-shape mismatch")

    similarity = torch.einsum("qd,crd->cqr", query, reference)
    logits = (
        similarity / temperature
        + rho_q.log()[None, :, None]
        + rho_r.clamp_min(_EPS).log()[:, None, :]
    )
    negative_infinity = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
    logits = torch.where(valid[:, None, :], logits, negative_infinity)
    real_counts = valid.sum(dim=1).to(logits.dtype)
    query_dustbin_logit = (
        b_null.reshape(1, 1, 1)
        + (1.0 - rho_q).clamp_min(_EPS).log()[None, :, None]
        + real_counts.log()[:, None, None]
    ).expand(reference.shape[0], -1, -1)
    q_probability = torch.softmax(torch.cat((logits, query_dustbin_logit), dim=2), dim=2)
    q_to_r = q_probability[:, :, :-1]
    q_dustbin = q_probability[:, :, -1]

    reverse_logits = logits.transpose(1, 2)
    reverse_dustbin_logit = (
        b_null.reshape(1, 1, 1)
        + (1.0 - rho_r).clamp_min(_EPS).log()[..., None]
        + math.log(query.shape[0])
    )
    r_probability = torch.softmax(
        torch.cat((reverse_logits, reverse_dustbin_logit), dim=2), dim=2
    )
    r_to_q = r_probability[:, :, :-1]
    r_dustbin = r_probability[:, :, -1]
    r_to_q = torch.where(valid[:, :, None], r_to_q, torch.zeros_like(r_to_q))
    r_dustbin = torch.where(valid, r_dustbin, torch.ones_like(r_dustbin))
    query_match_mass = 1.0 - q_dustbin
    reference_match_mass = torch.where(valid, 1.0 - r_dustbin, torch.zeros_like(r_dustbin))

    dtype = query.dtype
    device = query.device
    query_xy = grid_cell_centres(query_shape, dtype=dtype, device=device)
    reference_xy = _padded_reference_coordinates(
        reference_grid_shapes, reference.shape[1], dtype=dtype, device=device
    )
    expected_reference = torch.einsum("cqr,crd->cqd", q_to_r, reference_xy)
    expected_reference = expected_reference / query_match_mass[..., None].clamp_min(_EPS)
    expected_query = torch.einsum("crq,qd->crd", r_to_q, query_xy)
    expected_query = expected_query / reference_match_mass[..., None].clamp_min(_EPS)

    # Compose Q->R->Q through first and second coordinate moments without
    # materialising a potentially enormous [C,Q,Q] bridge.
    bridge_mass = torch.einsum("cqr,cr->cq", q_to_r, reference_match_mass)
    roundtrip = torch.einsum(
        "cqr,cr,crd->cqd", q_to_r, reference_match_mass, expected_query
    )
    roundtrip = roundtrip / bridge_mass[..., None].clamp_min(_EPS)
    cycle = torch.linalg.vector_norm(roundtrip - query_xy[None, :, :], dim=2)
    query_norm2 = query_xy.square().sum(dim=1)
    reference_return_second = torch.einsum("crq,q->cr", r_to_q, query_norm2)
    second = torch.einsum("cqr,cr->cq", q_to_r, reference_return_second)
    second = second / bridge_mass.clamp_min(_EPS)
    cycle_squared = (
        second
        - 2.0 * (roundtrip * query_xy[None, :, :]).sum(dim=2)
        + query_norm2[None, :]
    ).clamp_min(0.0)
    reciprocal = torch.einsum("cqr,crq->cq", q_to_r, r_to_q)

    q_conditional = q_to_r / query_match_mass[..., None].clamp_min(_EPS)
    r_conditional = r_to_q / reference_match_mass[..., None].clamp_min(_EPS)
    q_entropy = -(q_conditional * q_conditional.clamp_min(_EPS).log()).sum(dim=2)
    r_entropy = -(r_conditional * r_conditional.clamp_min(_EPS).log()).sum(dim=2)
    q_norm = real_counts.log().clamp_min(1.0)[:, None]
    r_norm = torch.tensor(
        math.log(max(2, query.shape[0])), dtype=dtype, device=device
    )
    q_entropy = q_entropy / q_norm
    r_entropy = torch.where(valid, r_entropy / r_norm, torch.ones_like(r_entropy))

    return R0SoftCorrespondence(
        q_to_r=q_to_r,
        r_to_q=r_to_q,
        query_dustbin=q_dustbin,
        reference_dustbin=r_dustbin,
        query_reliability=rho_q[None, :].expand(reference.shape[0], -1),
        reference_reliability=torch.where(valid, rho_r, torch.zeros_like(rho_r)),
        query_match_mass=query_match_mass,
        reference_match_mass=reference_match_mass,
        expected_reference_xy=expected_reference,
        expected_query_xy=expected_query,
        roundtrip_query_xy=roundtrip,
        query_cycle_displacement=cycle,
        query_cycle_squared_displacement=cycle_squared,
        reciprocal_mass=reciprocal,
        conditional_query_entropy=q_entropy,
        conditional_reference_entropy=r_entropy,
        reciprocal_pair_mass=q_to_r * r_to_q.transpose(1, 2),
    )


@dataclass(frozen=True)
class R0ReciprocalMassDiagnostics:
    total_mass: torch.Tensor  # [C]
    effective_matches: torch.Tensor  # [C]
    query_covariance: torch.Tensor  # [C,2,2]
    reference_covariance: torch.Tensor  # [C,2,2]
    query_min_eigenvalue: torch.Tensor  # [C]
    reference_min_eigenvalue: torch.Tensor  # [C]
    eligible: torch.Tensor  # [C]


def _weighted_covariance(coordinates: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total = weights.sum(dim=1).clamp_min(_EPS)
    mean = torch.einsum("cn,cnd->cd", weights, coordinates) / total[:, None]
    centered = coordinates - mean[:, None, :]
    return torch.einsum("cn,cni,cnj->cij", weights, centered, centered) / total[:, None, None]


def reciprocal_mass_diagnostics(
    correspondence: R0SoftCorrespondence,
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shapes: Sequence[tuple[int, int]],
    reference_valid_mask: torch.Tensor,
    minimum_mass: float = 8.0,
    minimum_effective_matches: float = 8.0,
) -> R0ReciprocalMassDiagnostics:
    """Target-free all-dustbin and coordinate-collapse barrier."""

    pair_mass = correspondence.reciprocal_pair_mass
    candidates, query_count, capacity = pair_mass.shape
    valid = torch.as_tensor(
        reference_valid_mask, dtype=torch.bool, device=pair_mass.device
    )
    if valid.shape != (candidates, capacity):
        raise ValueError("reciprocal mass validity schema drift")
    total = pair_mass.sum(dim=(1, 2))
    effective = total.square() / pair_mass.square().sum(dim=(1, 2)).clamp_min(_EPS)
    query_weight = pair_mass.sum(dim=2)
    reference_weight = pair_mass.sum(dim=1)
    query_xy = grid_cell_centres(
        _positive_grid_shape(query_grid_shape, name="query grid"),
        dtype=pair_mass.dtype,
        device=pair_mass.device,
    )
    if query_xy.shape[0] != query_count:
        raise ValueError("query coordinate/mass mismatch")
    query_coordinates = query_xy[None, :, :].expand(candidates, -1, -1)
    reference_coordinates = _padded_reference_coordinates(
        reference_grid_shapes,
        capacity,
        dtype=pair_mass.dtype,
        device=pair_mass.device,
    )
    query_covariance = _weighted_covariance(query_coordinates, query_weight)
    reference_covariance = _weighted_covariance(reference_coordinates, reference_weight)
    query_eigen = torch.linalg.eigvalsh(query_covariance)
    reference_eigen = torch.linalg.eigvalsh(reference_covariance)
    q_height, q_width = _positive_grid_shape(query_grid_shape, name="query grid")
    query_floor = (1.0 / (2.0 * max(q_height, q_width))) ** 2
    reference_floor = torch.tensor(
        [
            (1.0 / (2.0 * max(shape))) ** 2
            for shape in reference_grid_shapes
        ],
        dtype=pair_mass.dtype,
        device=pair_mass.device,
    )
    eligible = (
        total.ge(minimum_mass)
        & effective.ge(minimum_effective_matches)
        & query_eigen[:, 0].ge(query_floor)
        & reference_eigen[:, 0].ge(reference_floor)
    )
    return R0ReciprocalMassDiagnostics(
        total_mass=total,
        effective_matches=effective,
        query_covariance=query_covariance,
        reference_covariance=reference_covariance,
        query_min_eigenvalue=query_eigen[:, 0],
        reference_min_eigenvalue=reference_eigen[:, 0],
        eligible=eligible,
    )


def reciprocal_mass_hinge_barrier(
    diagnostics: R0ReciprocalMassDiagnostics,
    *,
    minimum_mass: float = 8.0,
    minimum_effective_matches: float = 8.0,
) -> torch.Tensor:
    """Positive-pair train barrier; all-dustbin cannot minimize it."""

    return (
        F.relu(minimum_mass - diagnostics.total_mass)
        + F.relu(minimum_effective_matches - diagnostics.effective_matches)
    ).mean()


@dataclass(frozen=True)
class R0NoncollapseDiagnostics:
    passed: torch.Tensor  # [C]
    reliable_fraction: torch.Tensor
    mean_entropy: torch.Tensor
    coordinate_variance_xy: torch.Tensor  # [C,2]
    known_warp_error: torch.Tensor


def known_warp_noncollapse_gate(
    correspondence: R0SoftCorrespondence,
    *,
    reference_grid_shapes: Sequence[tuple[int, int]],
    reference_valid_mask: torch.Tensor,
    reference_to_query_affine: torch.Tensor,
    minimum_reliability: float = 0.55,
    maximum_entropy: float = 0.90,
    minimum_coordinate_variance: float = 1.0e-4,
    maximum_warp_error: float = 0.12,
) -> R0NoncollapseDiagnostics:
    """Reject uniform-centre and arbitrary-permutation cycle shortcuts.

    A bijective but arbitrary token permutation can have perfect algebraic
    round-trip cycle.  It still fails this gate because expected query
    coordinates must agree with an independently frozen, result-blind affine.
    """

    valid = torch.as_tensor(
        reference_valid_mask,
        dtype=torch.bool,
        device=correspondence.expected_query_xy.device,
    )
    matrices = torch.as_tensor(
        reference_to_query_affine,
        dtype=correspondence.expected_query_xy.dtype,
        device=correspondence.expected_query_xy.device,
    )
    candidates, capacity = valid.shape
    if matrices.shape != (candidates, 3, 3):
        raise ValueError("known affine must be [C,3,3]")
    reference_xy = _padded_reference_coordinates(
        reference_grid_shapes,
        capacity,
        dtype=correspondence.expected_query_xy.dtype,
        device=correspondence.expected_query_xy.device,
    )
    homogeneous = torch.cat(
        (reference_xy, torch.ones_like(reference_xy[..., :1])), dim=2
    )
    mapped_h = torch.einsum("cij,crj->cri", matrices, homogeneous)
    mapped = mapped_h[..., :2] / mapped_h[..., 2:].clamp_min(_EPS)
    error = torch.linalg.vector_norm(correspondence.expected_query_xy - mapped, dim=2)
    weights = correspondence.reference_reliability * valid
    mean_error = (error * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(_EPS)

    reliable = correspondence.query_reliability.ge(minimum_reliability)
    reliable_fraction = reliable.to(correspondence.query_reliability.dtype).mean(dim=1)
    entropy = (
        correspondence.conditional_query_entropy * reliable
    ).sum(dim=1) / reliable.sum(dim=1).clamp_min(1)
    expected = correspondence.expected_reference_xy
    count = reliable.sum(dim=1).clamp_min(1).to(expected.dtype)
    mean = (expected * reliable[..., None]).sum(dim=1) / count[:, None]
    variance = (
        (expected - mean[:, None, :]).square() * reliable[..., None]
    ).sum(dim=1) / count[:, None]
    passed = (
        reliable_fraction.ge(0.10)
        & entropy.le(maximum_entropy)
        & variance.amin(dim=1).ge(minimum_coordinate_variance)
        & mean_error.le(maximum_warp_error)
    )
    return R0NoncollapseDiagnostics(
        passed=passed,
        reliable_fraction=reliable_fraction,
        mean_entropy=entropy,
        coordinate_variance_xy=variance,
        known_warp_error=mean_error,
    )


@dataclass(frozen=True)
class NaturalQueryReferenceConsistency:
    q1_to_q2_mass: torch.Tensor  # [C,Q1]
    predicted_q2_xy: torch.Tensor  # [C,Q1,2]
    q1_to_q2_entropy: torch.Tensor  # [C,Q1]


def natural_queries_via_same_reference(
    first: R0SoftCorrespondence,
    second: R0SoftCorrespondence,
    *,
    second_query_grid_shape: tuple[int, int],
) -> NaturalQueryReferenceConsistency:
    """Compose ``q1 -> same canonical reference -> q2`` target-free.

    Occluded q1 patches carry little real-reference mass and consequently
    remain low-mass instead of being forced onto q2.
    """

    if first.q_to_r.shape[0] != second.r_to_q.shape[0] or first.q_to_r.shape[2] != second.r_to_q.shape[1]:
        raise ValueError("the two natural queries must share the same reference atlas")
    bridge = torch.einsum("cir,crj->cij", first.q_to_r, second.r_to_q)
    mass = bridge.sum(dim=2)
    q2_xy = grid_cell_centres(
        _positive_grid_shape(second_query_grid_shape, name="second query grid"),
        dtype=bridge.dtype,
        device=bridge.device,
    )
    if q2_xy.shape[0] != bridge.shape[2]:
        raise ValueError("second query grid/token count mismatch")
    predicted = torch.einsum("cij,jd->cid", bridge, q2_xy)
    predicted = predicted / mass[..., None].clamp_min(_EPS)
    conditional = bridge / mass[..., None].clamp_min(_EPS)
    entropy = -(conditional * conditional.clamp_min(_EPS).log()).sum(dim=2)
    entropy = entropy / math.log(max(2, bridge.shape[2]))
    return NaturalQueryReferenceConsistency(
        q1_to_q2_mass=mass,
        predicted_q2_xy=predicted,
        q1_to_q2_entropy=entropy,
    )


def _bilinear_sample_query(
    query_tokens: torch.Tensor,
    xy: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    height, width = _positive_grid_shape(grid_shape, name="sampling grid")
    if query_tokens.shape != (height * width, DESCRIPTOR_DIM):
        raise ValueError("query sampling grid/token mismatch")
    feature = query_tokens.transpose(0, 1).reshape(1, DESCRIPTOR_DIM, height, width)
    grid = (2.0 * xy - 1.0).reshape(1, -1, 1, 2).to(query_tokens.dtype)
    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.reshape(DESCRIPTOR_DIM, -1).transpose(0, 1)


@dataclass(frozen=True)
class R0ForwardOutput:
    scores: torch.Tensor  # [C]
    direction_scores: torch.Tensor  # [C,2]
    slot_evidence: torch.Tensor  # [C,2,8]
    patch_evidence: torch.Tensor  # [C,2,8,8]
    patch_reliability: torch.Tensor  # [C,2,8,8]
    footprint_mass: torch.Tensor  # [C,2,8]
    footprint_cycle_response: torch.Tensor  # [C,2,8]
    footprint_weight: torch.Tensor  # normalized reliability/cycle weight [C,2,8]
    slot_total_mass: torch.Tensor  # local reciprocal mass [C,2,8]
    slot_effective_matches: torch.Tensor  # [C,2,8]
    slot_query_min_eigenvalue: torch.Tensor  # [C,2,8]
    slot_reference_min_eigenvalue: torch.Tensor  # [C,2,8]
    slot_query_eigen_floor: torch.Tensor  # [C,2,8]
    slot_reference_eigen_floor: torch.Tensor  # [C,2,8]
    slot_coordinate_eligible: torch.Tensor  # [C,2,8]
    prejoin_slot_legal: torch.Tensor  # byte-frozen forward copy [C,2,8]
    mass_diagnostics: R0ReciprocalMassDiagnostics
    correspondence: R0SoftCorrespondence


@dataclass(frozen=True)
class ConnectedSlotPositiveBarrier:
    """Post-join local anti-collapse result for one positive candidate."""

    loss: torch.Tensor
    per_slot_hinge: torch.Tensor  # [2,8]
    selected_slots: torch.Tensor  # [2], -1 means no frozen legal slot
    repairable_directions: torch.Tensor  # [2]


def connected_slot_positive_barrier(
    output: R0ForwardOutput,
    positive_candidate_index: int,
    *,
    minimum_mass: float = 8.0,
    minimum_effective_matches: float = 8.0,
) -> ConnectedSlotPositiveBarrier:
    """Train-local M/N_eff/2-D barrier over frozen connected slots.

    This is deliberately post-join: the positive index selects which already
    frozen candidate ledger receives anti-collapse supervision, but cannot
    alter its component, transform, footprint, or slots.  Per direction the
    lowest total hinge among frozen legal slots is used.  A direction with no
    legal connected slot is an unrepairable coverage failure and contributes
    exact zero (not a fabricated gradient).

    ``output.mass_diagnostics`` remains a global-atlas audit only and must not
    be substituted for this local barrier.
    """

    candidate_count = int(output.slot_total_mass.shape[0])
    if not 0 <= positive_candidate_index < candidate_count:
        raise ValueError("positive candidate index is outside the frozen candidate ledger")
    legal = output.prejoin_slot_legal.to(output.slot_total_mass.device)
    if legal.shape != output.slot_total_mass.shape:
        raise ValueError("frozen connected-slot legal mask schema drift")
    if not math.isfinite(minimum_mass) or minimum_mass <= 0.0:
        raise ValueError("minimum local reciprocal mass must be positive")
    if not math.isfinite(minimum_effective_matches) or minimum_effective_matches <= 0.0:
        raise ValueError("minimum local effective-match count must be positive")

    mass = output.slot_total_mass[positive_candidate_index]
    effective = output.slot_effective_matches[positive_candidate_index]
    q_eigen = output.slot_query_min_eigenvalue[positive_candidate_index]
    r_eigen = output.slot_reference_min_eigenvalue[positive_candidate_index]
    q_floor = output.slot_query_eigen_floor[positive_candidate_index]
    r_floor = output.slot_reference_eigen_floor[positive_candidate_index]
    per_slot = (
        F.relu(minimum_mass - mass) / minimum_mass
        + F.relu(minimum_effective_matches - effective) / minimum_effective_matches
        + F.relu(q_floor - q_eigen) / q_floor.clamp_min(_EPS)
        + F.relu(r_floor - r_eigen) / r_floor.clamp_min(_EPS)
    )
    candidate_legal = legal[positive_candidate_index]
    selected = torch.full((DIRECTIONS,), -1, dtype=torch.long, device=mass.device)
    repairable = candidate_legal.any(dim=1)
    losses: list[torch.Tensor] = []
    for direction in range(DIRECTIONS):
        legal_slots = torch.nonzero(candidate_legal[direction], as_tuple=False).flatten()
        if legal_slots.numel() == 0:
            continue
        legal_hinges = per_slot[direction, legal_slots]
        local_choice = torch.argmin(legal_hinges.detach())
        chosen = legal_slots[local_choice]
        selected[direction] = chosen
        losses.append(per_slot[direction, chosen])
    if losses:
        loss = torch.stack(losses).mean()
    else:
        # Preserve a well-typed scalar receipt while intentionally providing
        # no training signal for a candidate without any prejoin legal slot.
        loss = output.slot_total_mass.sum() * 0.0
    return ConnectedSlotPositiveBarrier(
        loss=loss,
        per_slot_hinge=per_slot,
        selected_slots=selected,
        repairable_directions=repairable,
    )


def fixed_capacity_candidate_evidence(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    geometry: R0PrejoinGeometry,
    correspondence: R0SoftCorrespondence,
    *,
    evidence_clip: float = 4.0,
) -> R0ForwardOutput:
    """Score only geometry-held-out cells inside the dense connected object.

    The eight rows are coverage samples, not the latent target.  Every patch
    contribution is multiplied by reliability and by whole-footprint
    match/cycle mass.  A collection of isolated hot patches therefore cannot
    bypass the dense connected footprint contract.
    """

    query = torch.as_tensor(query_tokens)
    references = torch.as_tensor(reference_tokens)
    if (
        references.shape[:2] != geometry.reference_valid_mask.shape
        or references.shape[2] != DESCRIPTOR_DIM
        or query.shape != (math.prod(geometry.query_grid_shape), DESCRIPTOR_DIM)
    ):
        raise ValueError("R0 evidence token/geometry schema mismatch")
    if not math.isfinite(evidence_clip) or evidence_clip <= 0.0:
        raise ValueError("evidence clip must be finite and positive")
    device = query.device
    dtype = query.dtype
    verification = geometry.verification_mask.to(device)
    legal = geometry.slot_legal.to(device)
    q_footprints = geometry.query_footprints.to(device)
    r_footprints = geometry.reference_footprints.to(device)
    predicted = geometry.predicted_query_xy.to(device=device, dtype=dtype)
    reference_indices = geometry.reference_indices.to(device)
    valid = geometry.reference_valid_mask.to(device)

    patch_evidence = torch.zeros(verification.shape, dtype=dtype, device=device)
    patch_reliability = torch.zeros_like(patch_evidence)
    footprint_mass = torch.zeros(legal.shape, dtype=dtype, device=device)
    footprint_cycle = torch.zeros_like(footprint_mass)
    footprint_weight = torch.zeros_like(footprint_mass)
    slot_total_mass = torch.zeros_like(footprint_mass)
    slot_effective = torch.zeros_like(footprint_mass)
    slot_q_min_eigen = torch.zeros_like(footprint_mass)
    slot_r_min_eigen = torch.zeros_like(footprint_mass)
    slot_q_eigen_floor = torch.zeros_like(footprint_mass)
    slot_r_eigen_floor = torch.zeros_like(footprint_mass)
    slot_coordinate_eligible = torch.zeros_like(legal)

    mass_diagnostics = reciprocal_mass_diagnostics(
        correspondence,
        query_grid_shape=geometry.query_grid_shape,
        reference_grid_shapes=geometry.reference_grid_shapes,
        reference_valid_mask=geometry.reference_valid_mask,
    )
    query_xy_all = grid_cell_centres(
        geometry.query_grid_shape, dtype=dtype, device=device
    )
    query_height, query_width = geometry.query_grid_shape
    sigma_cell_squared = (1.0 / query_height) ** 2 + (1.0 / query_width) ** 2

    for candidate in range(geometry.candidate_count):
        candidate_valid = valid[candidate]
        for direction in range(DIRECTIONS):
            for slot in range(GEOMETRY_SLOTS):
                if not bool(legal[candidate, direction, slot]):
                    continue
                q_mask = q_footprints[candidate, direction, slot]
                r_mask = r_footprints[candidate, direction, slot] & candidate_valid
                q_rows_dense = torch.nonzero(q_mask, as_tuple=False).flatten()
                r_rows_dense = torch.nonzero(r_mask, as_tuple=False).flatten()
                # Local closure is Rq -> Rr -> Rq.  The global softmax is kept
                # so flow to a perfect-looking patch outside the proposed
                # object counts as failure, but it can never add local mass or
                # improve the connected-slot score.
                local_a = correspondence.q_to_r[candidate][q_rows_dense][:, r_rows_dense]
                local_b = correspondence.r_to_q[candidate][r_rows_dense][:, q_rows_dense]
                dense_pair_mass = local_a * local_b.transpose(0, 1)
                dense_mass = dense_pair_mass.sum()
                dense_effective = dense_mass.square() / dense_pair_mass.square().sum().clamp_min(
                    _EPS
                )
                q_endpoint_weight = dense_pair_mass.sum(dim=1)
                r_endpoint_weight = dense_pair_mass.sum(dim=0)

                def minimum_covariance_eigenvalue(
                    coordinates: torch.Tensor, weights: torch.Tensor
                ) -> torch.Tensor:
                    total_weight = weights.sum().clamp_min(_EPS)
                    mean_coordinate = (coordinates * weights[:, None]).sum(dim=0) / total_weight
                    centered = coordinates - mean_coordinate
                    covariance = torch.einsum(
                        "n,ni,nj->ij", weights, centered, centered
                    ) / total_weight
                    return torch.linalg.eigvalsh(covariance)[0]

                q_min_eigen = minimum_covariance_eigenvalue(
                    query_xy_all[q_rows_dense], q_endpoint_weight
                )
                reference_xy_all = grid_cell_centres(
                    geometry.reference_grid_shapes[candidate],
                    dtype=dtype,
                    device=device,
                )
                r_min_eigen = minimum_covariance_eigenvalue(
                    reference_xy_all[r_rows_dense], r_endpoint_weight
                )
                reference_height, reference_width = geometry.reference_grid_shapes[candidate]
                q_floor = (1.0 / (2.0 * max(query_height, query_width))) ** 2
                r_floor = (
                    1.0 / (2.0 * max(reference_height, reference_width))
                ) ** 2
                coordinate_eligible = bool(
                    dense_mass.ge(8.0)
                    & dense_effective.ge(8.0)
                    & q_min_eigen.ge(q_floor)
                    & r_min_eigen.ge(r_floor)
                )
                q_region_mass = local_a.sum(dim=1)
                r_region_mass = local_b.sum(dim=1)

                # Expected local cycle is normalized only over successful
                # returns inside Rq.  Outflow is retained below in the signed
                # failure denominator, never renormalized into success.
                local_bridge = local_a @ local_b
                local_return_mass = local_bridge.sum(dim=1)
                local_q_coordinates = query_xy_all[q_rows_dense]
                local_cycle_squared = (
                    local_bridge
                    * torch.cdist(local_q_coordinates, local_q_coordinates).square()
                ).sum(dim=1) / local_return_mass.clamp_min(_EPS)
                dense_cycle = torch.exp(
                    -local_cycle_squared.mean() / sigma_cell_squared
                )
                dense_match_mass = torch.sqrt(
                    q_region_mass.mean() * r_region_mass.mean()
                )
                dense_weight = dense_match_mass * dense_cycle
                footprint_mass[candidate, direction, slot] = dense_match_mass
                footprint_cycle[candidate, direction, slot] = dense_cycle
                footprint_weight[candidate, direction, slot] = dense_weight
                slot_total_mass[candidate, direction, slot] = dense_mass
                slot_effective[candidate, direction, slot] = dense_effective
                slot_q_min_eigen[candidate, direction, slot] = q_min_eigen
                slot_r_min_eigen[candidate, direction, slot] = r_min_eigen
                slot_q_eigen_floor[candidate, direction, slot] = q_floor
                slot_r_eigen_floor[candidate, direction, slot] = r_floor
                slot_coordinate_eligible[candidate, direction, slot] = coordinate_eligible
                if not coordinate_eligible:
                    # A dense mask alone is insufficient: reciprocal mass must
                    # cover that same connected region in two dimensions.
                    continue

                active = verification[candidate, direction, slot]
                rows = reference_indices[candidate, direction, slot, active]
                xy = predicted[candidate, direction, slot, active]
                query_rows = _grid_index_from_normalized_xy(xy, geometry.query_grid_shape).to(device)
                q_reliability = correspondence.query_reliability[candidate, query_rows]
                r_reliability = correspondence.reference_reliability[candidate, rows]
                reliability = torch.sqrt(q_reliability * r_reliability)
                # Formal R0 evidence: non-dustbin reciprocal mass versus
                # dustbin return, followed by expected cycle and heldout affine
                # penalties.  Candidate competition belongs to postjoin loss;
                # no all-token generic pair scorer is hidden here.
                q_lookup = torch.full(
                    (query.shape[0],), -1, dtype=torch.long, device=device
                )
                q_lookup[q_rows_dense] = torch.arange(q_rows_dense.numel(), device=device)
                r_lookup = torch.full(
                    (references.shape[1],), -1, dtype=torch.long, device=device
                )
                r_lookup[r_rows_dense] = torch.arange(r_rows_dense.numel(), device=device)
                local_q_rows = q_lookup[query_rows]
                local_r_rows = r_lookup[rows]
                if bool(local_q_rows.lt(0).any() or local_r_rows.lt(0).any()):
                    raise RuntimeError("heldout rows escaped the validated connected footprint")
                reciprocal = dense_pair_mass[local_q_rows].sum(dim=1)

                a_full = correspondence.q_to_r[candidate, query_rows]
                a_local = a_full[:, r_rows_dense]
                a_outside = a_full.sum(dim=1) - a_local.sum(dim=1)
                b_local_rows = correspondence.r_to_q[candidate, rows]
                # All selected Rr endpoints contribute their dustbin/outside
                # return failure.  A-flow outside Rr is also explicit failure.
                failure = (
                    correspondence.query_dustbin[candidate, query_rows]
                    + a_outside
                    + (a_local * (
                        correspondence.reference_dustbin[candidate, r_rows_dense]
                        + correspondence.r_to_q[candidate, r_rows_dense].sum(dim=1)
                        - local_b.sum(dim=1)
                    )[None, :]).sum(dim=1)
                )
                cycle_squared = local_cycle_squared[local_q_rows]
                heldout_local_b = local_b[local_r_rows]
                heldout_local_mass = heldout_local_b.sum(dim=1)
                heldout_expected_query = (
                    heldout_local_b @ local_q_coordinates
                ) / heldout_local_mass[:, None].clamp_min(_EPS)
                affine_squared = (
                    heldout_expected_query - xy
                ).square().sum(dim=1)
                signed = (
                    torch.log((reciprocal + _EPS) / (failure + _EPS))
                    - cycle_squared / sigma_cell_squared
                    - affine_squared / sigma_cell_squared
                )
                patch_evidence[candidate, direction, slot, active] = dense_weight * signed.clamp(
                    -evidence_clip, evidence_clip
                )
                patch_reliability[candidate, direction, slot, active] = reliability

    # Fixed denominator: invalid slots and unused rows are exact zero.  A
    # candidate with all H0 remains bit-exact zero independent of legal count.
    slot_evidence = patch_evidence.sum(dim=3) / float(VERIFICATION_SLOTS)
    direction_scores = torch.logsumexp(
        torch.cat(
            (
                torch.zeros(
                    (*slot_evidence.shape[:2], 1), dtype=dtype, device=device
                ),
                slot_evidence,
            ),
            dim=2,
        ),
        dim=2,
    ) - math.log(GEOMETRY_SLOTS + 1)
    scores = direction_scores.mean(dim=1)
    return R0ForwardOutput(
        scores=scores,
        direction_scores=direction_scores,
        slot_evidence=slot_evidence,
        patch_evidence=patch_evidence,
        patch_reliability=patch_reliability,
        footprint_mass=footprint_mass,
        footprint_cycle_response=footprint_cycle,
        footprint_weight=footprint_weight,
        slot_total_mass=slot_total_mass,
        slot_effective_matches=slot_effective,
        slot_query_min_eigenvalue=slot_q_min_eigen,
        slot_reference_min_eigenvalue=slot_r_min_eigen,
        slot_query_eigen_floor=slot_q_eigen_floor,
        slot_reference_eigen_floor=slot_r_eigen_floor,
        slot_coordinate_eligible=slot_coordinate_eligible,
        prejoin_slot_legal=legal.detach().clone(),
        mass_diagnostics=mass_diagnostics,
        correspondence=correspondence,
    )


class R0CrossViewCycleCore(nn.Module):
    """Frozen-token R0 adapter plus target-free connected geometry scorer."""

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        seed: int = 17,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)
        self.adapter = AsymmetricRank8ResidualAdapter(seed=seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1)
        self.query_reliability = nn.Linear(
            DESCRIPTOR_DIM, 1, bias=True, dtype=torch.float64
        )
        self.reference_reliability = nn.Linear(
            DESCRIPTOR_DIM, 1, bias=True, dtype=torch.float64
        )
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

    def forward(
        self,
        query_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        geometry: R0PrejoinGeometry,
    ) -> R0ForwardOutput:
        if reference_tokens.shape[:2] != geometry.reference_valid_mask.shape:
            raise ValueError("candidate/reference padding does not match prejoin geometry")
        query = self.adapter.adapt_query(query_tokens)
        reference = self.adapter.adapt_reference(reference_tokens)
        # Exact padding masking is applied by correspondence and geometry; zero
        # padded vectors avoid accidental finite payload in receipts.
        reference = torch.where(
            geometry.reference_valid_mask.to(reference.device)[..., None],
            reference,
            torch.zeros_like(reference),
        )
        rho_q = torch.sigmoid(self.query_reliability(query).squeeze(-1))
        rho_r = torch.sigmoid(self.reference_reliability(reference).squeeze(-1))
        rho_r = torch.where(
            geometry.reference_valid_mask.to(rho_r.device),
            rho_r,
            torch.full_like(rho_r, 0.5),
        )
        correspondence = soft_correspondence_with_dustbin(
            query,
            reference,
            query_grid_shape=geometry.query_grid_shape,
            reference_grid_shapes=geometry.reference_grid_shapes,
            reference_valid_mask=geometry.reference_valid_mask,
            query_reliability=rho_q,
            reference_reliability=rho_r,
            dustbin_logit=self.b_null,
            temperature=self.temperature,
        )
        return fixed_capacity_candidate_evidence(
            query,
            reference,
            geometry,
            correspondence,
        )


def smooth_candidate_pair_loss(
    scores: torch.Tensor,
    positive_candidate_index: int,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Post-join all-natural-negative loss; never called by prejoin forward."""

    value = torch.as_tensor(scores)
    if value.ndim != 1 or value.numel() < 2:
        raise ValueError("candidate loss requires at least two candidate scores")
    if not 0 <= positive_candidate_index < value.numel():
        raise ValueError("positive candidate index is out of range")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("loss temperature must be finite and positive")
    positive = value[positive_candidate_index]
    negative = torch.cat(
        (value[:positive_candidate_index], value[positive_candidate_index + 1 :])
    )
    return torch.logsumexp(
        torch.cat(
            (
                torch.zeros(1, dtype=value.dtype, device=value.device),
                (negative - positive) / temperature,
            )
        ),
        dim=0,
    )


class ControlKind(str, Enum):
    CANDIDATE_BINDING = "CANDIDATE_BINDING"
    SPATIAL = "SPATIAL"


class ControlPhase(str, Enum):
    TRAIN = "TRAIN"
    EVALUATION = "EVALUATION"


@dataclass(frozen=True)
class R0ControlSpec:
    namespace: str
    kind: ControlKind
    phase: ControlPhase

    def __post_init__(self) -> None:
        if not self.namespace or self.kind.value not in self.namespace or self.phase.value not in self.namespace:
            raise ValueError("control namespace must explicitly encode kind and phase")


def validate_control_namespace_separation(specs: Iterable[R0ControlSpec]) -> None:
    values = tuple(specs)
    namespaces = [spec.namespace for spec in values]
    if len(namespaces) != len(set(namespaces)):
        raise ValueError("train/evaluation C/P controls require disjoint namespaces")
    by_pair = {(spec.kind, spec.phase) for spec in values}
    if len(by_pair) != len(values):
        raise ValueError("duplicate control kind/phase contract")


def _complete_permutation(value: torch.Tensor, size: int, *, name: str) -> torch.Tensor:
    permutation = torch.as_tensor(value, dtype=torch.long).detach().cpu().contiguous()
    if permutation.shape != (size,) or not torch.equal(
        torch.sort(permutation).values, torch.arange(size)
    ):
        raise ValueError(f"{name} must be a complete permutation")
    return permutation


def reorder_candidates(
    reference_tokens: torch.Tensor,
    geometry: R0PrejoinGeometry,
    order: torch.Tensor,
) -> tuple[torch.Tensor, R0PrejoinGeometry]:
    """Joint equivariant reorder; this is not candidate-binding control C."""

    permutation = _complete_permutation(
        order, geometry.candidate_count, name="candidate reorder"
    )
    shapes = tuple(geometry.reference_grid_shapes[int(index)] for index in permutation)
    reordered = R0PrejoinGeometry(
        predicted_query_xy=geometry.predicted_query_xy[permutation],
        reference_indices=geometry.reference_indices[permutation],
        verification_mask=geometry.verification_mask[permutation],
        slot_legal=geometry.slot_legal[permutation],
        query_footprints=geometry.query_footprints[permutation],
        reference_footprints=geometry.reference_footprints[permutation],
        affine_matrices=geometry.affine_matrices[permutation],
        reference_valid_mask=geometry.reference_valid_mask[permutation],
        query_grid_shape=geometry.query_grid_shape,
        reference_grid_shapes=shapes,
    )
    return reference_tokens[permutation.to(reference_tokens.device)], reordered


def apply_candidate_binding_control(
    reference_tokens: torch.Tensor,
    permutation: torch.Tensor,
    spec: R0ControlSpec,
    *,
    reference_grid_shapes: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Derange whole content only between exactly shape-matched candidates."""

    if spec.kind is not ControlKind.CANDIDATE_BINDING:
        raise ValueError("a spatial namespace cannot authorize candidate binding control")
    order = _complete_permutation(permutation, reference_tokens.shape[0], name="C control")
    if bool(order.eq(torch.arange(order.numel())).any()):
        raise ValueError("candidate-binding destruction must be a derangement")
    if len(reference_grid_shapes) != reference_tokens.shape[0]:
        raise ValueError("candidate-binding control shape ledger mismatch")
    shapes = tuple(
        _positive_grid_shape(shape, name="C-control reference grid")
        for shape in reference_grid_shapes
    )
    if any(shapes[destination] != shapes[int(source)] for destination, source in enumerate(order)):
        raise ValueError(
            "candidate-binding control forbids padding, truncation, resize, or unequal-grid swaps"
        )
    return reference_tokens[order.to(reference_tokens.device)]


@dataclass(frozen=True)
class ShapeMatchedCandidateBindingControl:
    reference_tokens: torch.Tensor
    permutation: torch.Tensor
    eligible: torch.Tensor


def shape_matched_candidate_binding_control(
    reference_tokens: torch.Tensor,
    reference_grid_shapes: Sequence[tuple[int, int]],
    spec: R0ControlSpec,
) -> ShapeMatchedCandidateBindingControl:
    """Canonical within-shape derangement plus explicit singleton coverage.

    Groups with one member cannot support a scientifically valid whole-atlas C
    control.  Their content is exact zero and ``eligible=False``; evaluators
    must keep them in the coverage denominator and score them as H0 rather than
    silently padding/truncating another atlas.
    """

    if spec.kind is not ControlKind.CANDIDATE_BINDING:
        raise ValueError("a spatial namespace cannot authorize candidate binding control")
    if len(reference_grid_shapes) != reference_tokens.shape[0]:
        raise ValueError("candidate-binding control shape ledger mismatch")
    shapes = tuple(
        _positive_grid_shape(shape, name="C-control reference grid")
        for shape in reference_grid_shapes
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for index, shape in enumerate(shapes):
        groups.setdefault(shape, []).append(index)
    permutation = torch.arange(reference_tokens.shape[0], dtype=torch.long)
    eligible = torch.zeros(reference_tokens.shape[0], dtype=torch.bool)
    output = reference_tokens.clone()
    for members in groups.values():
        if len(members) < 2:
            output[members[0]].zero_()
            continue
        for offset, destination in enumerate(members):
            source = members[(offset + 1) % len(members)]
            permutation[destination] = source
            eligible[destination] = True
            output[destination] = reference_tokens[source]
    return ShapeMatchedCandidateBindingControl(
        reference_tokens=output,
        permutation=permutation,
        eligible=eligible,
    )


def apply_spatial_control(
    reference_tokens: torch.Tensor,
    permutations: Sequence[torch.Tensor],
    reference_valid_mask: torch.Tensor,
    spec: R0ControlSpec,
) -> torch.Tensor:
    """Destroy coordinate-to-content binding within each candidate only."""

    if spec.kind is not ControlKind.SPATIAL:
        raise ValueError("a candidate-binding namespace cannot authorize spatial control")
    if len(permutations) != reference_tokens.shape[0]:
        raise ValueError("one spatial permutation is required per candidate")
    output = reference_tokens.clone()
    valid = torch.as_tensor(reference_valid_mask, dtype=torch.bool)
    for candidate, raw_order in enumerate(permutations):
        count = int(valid[candidate].sum())
        order = _complete_permutation(raw_order, count, name="P control")
        if bool(order.eq(torch.arange(count)).any()):
            raise ValueError("spatial destruction must have no fixed cells")
        output[candidate, :count] = reference_tokens[candidate, order.to(reference_tokens.device)]
    return output


def target_free_tensor_hash(*values: Any) -> str:
    """Hash only tensor/prejoin state; post-join labels have no API path here."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, R0PrejoinGeometry):
            for field in fields(value):
                update(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            digest.update(str(len(value)).encode())
            for item in value:
                update(item)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            digest.update(repr(value).encode())
        else:
            raise TypeError(f"unsupported target-free hash type: {type(value)!r}")

    for item in values:
        update(item)
    return digest.hexdigest()


def assert_forward_schema_is_target_free() -> None:
    """Fail closed if a future forward ledger grows target/D1 shortcut fields."""

    forbidden = ("target", "label", "d1", "rank", "slot_winner", "base_gap")
    names = [field.name.lower() for field in fields(R0PrejoinGeometry)]
    names += [field.name.lower() for field in fields(R0ForwardOutput)]
    contaminated = [name for name in names if any(token in name for token in forbidden)]
    if contaminated:
        raise RuntimeError(f"R0 target-free schema contaminated: {contaminated}")


assert_forward_schema_is_target_free()
