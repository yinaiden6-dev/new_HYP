"""Target-free multi-support P-lock primitives for the conditional V gate.

This module starts a new, immutable lineage.  V1--V3 sealed eight *points* and
then treated every slot as H1.  V4 instead separates three objects:

``SeedCorrespondenceSet``
    P-only reciprocal correspondences.  A seed is an initializer, not target
    evidence.

``SealedMultiPatchHypotheses``
    Up to eight seed-local hypotheses.  Every legal hypothesis contains at
    least four one-to-one query/reference correspondences with genuine 2-D
    extent and bounded evidence concentration.

``AcceptedTargetLock``
    A V-validated subset.  Proposal code never returns this type.  Validation
    uses patch-level features and excludes the seed and its halo, so a single
    high-similarity point cannot validate itself.

All public proposal and verification APIs are target-free.  They deliberately
have no target, identity label, D1 score, rank, slot, winner, or retrieval
action argument.  Grid sizes and channel dimensions are runtime values so the
same core can qualify Qwen3 local tokens or a ColPali fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


MAX_HYPOTHESES = 8
DEFAULT_MAX_SUPPORT = 12
MIN_SUPPORT = 4
MIN_EFFECTIVE_SUPPORT = 3.0
MAX_SINGLE_WEIGHT = 0.5
MIN_VERIFICATION_PATCHES = 3
PATCH_FEATURE_DIM = 4
DEFAULT_SUPPORT_RADIUS = 4
DEFAULT_LEAVEOUT_HALO = 1
_DIRECTIONS = frozenset({"a_to_b", "b_to_a"})


def _shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError(f"{name} must be a positive (height,width) tuple")
    return int(value[0]), int(value[1])


def _finite_tokens(
    value: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
    name: str,
) -> torch.Tensor:
    height, width = _shape(grid_shape, name=f"{name} grid")
    tensor = torch.as_tensor(value)
    if (
        tensor.ndim != 2
        or tensor.shape[0] != height * width
        or tensor.shape[1] <= 0
        or not tensor.is_floating_point()
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError(
            f"{name} must be finite floating [{height * width},D]"
        )
    return tensor.detach().contiguous()


def _selection_mask(
    value: torch.Tensor | None,
    *,
    count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.ones(count, dtype=torch.bool, device=device)
    mask = torch.as_tensor(value, dtype=torch.bool, device=device)
    if mask.shape != (count,) or not bool(mask.any()):
        raise ValueError(f"{name} must select at least one of {count} cells")
    return mask


def _xy(indices: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    _, width = _shape(grid_shape, name="grid")
    value = torch.as_tensor(indices, dtype=torch.long)
    return torch.stack(
        (value.remainder(width), torch.div(value, width, rounding_mode="floor")),
        dim=-1,
    )


def _map_grid_indices(
    indices: torch.Tensor,
    *,
    source_grid_shape: tuple[int, int],
    destination_grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Map cell centres between independently resized P and V grids."""

    source_h, source_w = _shape(source_grid_shape, name="source grid")
    destination_h, destination_w = _shape(
        destination_grid_shape, name="destination grid"
    )
    value = torch.as_tensor(indices, dtype=torch.long)
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= source_h * source_w):
        raise ValueError("source grid index is out of range")
    source_xy = _xy(value, (source_h, source_w)).to(torch.float64)
    normalized_x = (source_xy[..., 0] + 0.5) / float(source_w)
    normalized_y = (source_xy[..., 1] + 0.5) / float(source_h)
    destination_x = torch.floor(normalized_x * destination_w).to(torch.long)
    destination_y = torch.floor(normalized_y * destination_h).to(torch.long)
    destination_x.clamp_(0, destination_w - 1)
    destination_y.clamp_(0, destination_h - 1)
    return destination_y * destination_w + destination_x


def _has_2d_span(indices: torch.Tensor, grid_shape: tuple[int, int]) -> bool:
    coordinates = _xy(indices, grid_shape)
    return bool(
        coordinates[:, 0].unique().numel() >= 2
        and coordinates[:, 1].unique().numel() >= 2
    )


def _effective_size(weights: torch.Tensor) -> float:
    value = torch.as_tensor(weights, dtype=torch.float64)
    return float(value.sum().square() / value.square().sum())


@dataclass(frozen=True)
class SeedCorrespondenceSet:
    """P-only reciprocal seed pairs; scores intentionally do not cross P->V."""

    query_indices: torch.Tensor
    reference_indices: torch.Tensor
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    direction: str

    def __post_init__(self) -> None:
        query_shape = _shape(self.query_grid_shape, name="query grid")
        reference_shape = _shape(self.reference_grid_shape, name="reference grid")
        query = torch.as_tensor(
            self.query_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        reference = torch.as_tensor(
            self.reference_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        if (
            query.ndim != 1
            or reference.shape != query.shape
            or query.numel() > MAX_HYPOTHESES
        ):
            raise ValueError("seed set must contain between zero and eight pairs")
        if self.direction not in _DIRECTIONS:
            raise ValueError("direction must be a_to_b or b_to_a")
        if query.numel():
            if (
                int(query.min()) < 0
                or int(query.max()) >= math.prod(query_shape)
                or int(reference.min()) < 0
                or int(reference.max()) >= math.prod(reference_shape)
            ):
                raise ValueError("seed index is outside its grid")
            if (
                query.unique().numel() != query.numel()
                or reference.unique().numel() != reference.numel()
            ):
                raise ValueError("seed pairs must be one-to-one")
        object.__setattr__(self, "query_indices", query)
        object.__setattr__(self, "reference_indices", reference)
        object.__setattr__(self, "query_grid_shape", query_shape)
        object.__setattr__(self, "reference_grid_shape", reference_shape)

    def __len__(self) -> int:
        return int(self.query_indices.numel())


@dataclass(frozen=True)
class SealedMultiPatchHypotheses:
    """Fixed-capacity P->V interface with explicit H1 and support masks."""

    seed_query_indices: torch.Tensor
    seed_reference_indices: torch.Tensor
    support_query_indices: torch.Tensor
    support_reference_indices: torch.Tensor
    support_weights: torch.Tensor
    support_mask: torch.Tensor
    legal: torch.Tensor
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    direction: str

    def __post_init__(self) -> None:
        query_shape = _shape(self.query_grid_shape, name="query grid")
        reference_shape = _shape(self.reference_grid_shape, name="reference grid")
        seed_query = torch.as_tensor(
            self.seed_query_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        seed_reference = torch.as_tensor(
            self.seed_reference_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        support_query = torch.as_tensor(
            self.support_query_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        support_reference = torch.as_tensor(
            self.support_reference_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        weights = torch.as_tensor(
            self.support_weights, dtype=torch.float64
        ).detach().cpu().contiguous()
        support_mask = torch.as_tensor(
            self.support_mask, dtype=torch.bool
        ).detach().cpu().contiguous()
        legal = torch.as_tensor(self.legal, dtype=torch.bool).detach().cpu().contiguous()

        if seed_query.shape != (MAX_HYPOTHESES,) or seed_reference.shape != (
            MAX_HYPOTHESES,
        ):
            raise ValueError("multi-support interface requires exactly eight H1 slots")
        if (
            support_query.ndim != 2
            or support_query.shape[0] != MAX_HYPOTHESES
            or support_query.shape[1] < MIN_SUPPORT
            or support_reference.shape != support_query.shape
            or weights.shape != support_query.shape
            or support_mask.shape != support_query.shape
            or legal.shape != (MAX_HYPOTHESES,)
        ):
            raise ValueError("multi-support tensor schema drift")
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("support weights must be finite")
        if self.direction not in _DIRECTIONS:
            raise ValueError("direction must be a_to_b or b_to_a")

        query_count = math.prod(query_shape)
        reference_count = math.prod(reference_shape)
        for hypothesis_index in range(MAX_HYPOTHESES):
            active = support_mask[hypothesis_index]
            if not bool(legal[hypothesis_index]):
                if (
                    int(seed_query[hypothesis_index]) != -1
                    or int(seed_reference[hypothesis_index]) != -1
                    or bool(active.any())
                    or bool(weights[hypothesis_index].ne(0).any())
                    or bool(support_query[hypothesis_index].ne(-1).any())
                    or bool(support_reference[hypothesis_index].ne(-1).any())
                ):
                    raise ValueError("illegal H1 slots must be canonical exact H0 padding")
                continue

            query = support_query[hypothesis_index, active]
            reference = support_reference[hypothesis_index, active]
            active_weights = weights[hypothesis_index, active]
            if int(active.sum()) < MIN_SUPPORT:
                raise ValueError("a legal hypothesis needs at least four correspondences")
            if (
                int(query.min()) < 0
                or int(query.max()) >= query_count
                or int(reference.min()) < 0
                or int(reference.max()) >= reference_count
            ):
                raise ValueError("support index is outside its grid")
            if (
                query.unique().numel() != query.numel()
                or reference.unique().numel() != reference.numel()
            ):
                raise ValueError("legal support must be one-to-one in query and reference")
            if not bool(
                (
                    (query == seed_query[hypothesis_index])
                    & (reference == seed_reference[hypothesis_index])
                ).any()
            ):
                raise ValueError("the initiating seed must remain in its support")
            if not _has_2d_span(query, query_shape) or not _has_2d_span(
                reference, reference_shape
            ):
                raise ValueError("legal support needs nondegenerate query/reference 2-D span")
            if (
                not bool(active_weights.gt(0).all())
                or not math.isclose(float(active_weights.sum()), 1.0, abs_tol=1.0e-10)
                or float(active_weights.max()) > MAX_SINGLE_WEIGHT + 1.0e-12
                or _effective_size(active_weights) < MIN_EFFECTIVE_SUPPORT - 1.0e-12
            ):
                raise ValueError("legal support weight concentration violates the contract")
            if (
                bool(weights[hypothesis_index, ~active].ne(0).any())
                or bool(support_query[hypothesis_index, ~active].ne(-1).any())
                or bool(support_reference[hypothesis_index, ~active].ne(-1).any())
            ):
                raise ValueError("inactive support entries must use canonical zero/-1 padding")

        object.__setattr__(self, "seed_query_indices", seed_query)
        object.__setattr__(self, "seed_reference_indices", seed_reference)
        object.__setattr__(self, "support_query_indices", support_query)
        object.__setattr__(self, "support_reference_indices", support_reference)
        object.__setattr__(self, "support_weights", weights)
        object.__setattr__(self, "support_mask", support_mask)
        object.__setattr__(self, "legal", legal)
        object.__setattr__(self, "query_grid_shape", query_shape)
        object.__setattr__(self, "reference_grid_shape", reference_shape)

    @property
    def legal_count(self) -> int:
        return int(self.legal.sum())


@dataclass(frozen=True)
class MultiPatchVerificationFeatures:
    """Patch-level V output with proposal and post-leaveout masks retained."""

    patch_features: torch.Tensor
    support_weights: torch.Tensor
    support_mask: torch.Tensor
    proposal_legal: torch.Tensor
    verification_mask: torch.Tensor
    verification_legal: torch.Tensor
    direction: str

    def __post_init__(self) -> None:
        features = torch.as_tensor(self.patch_features).detach().contiguous()
        weights = torch.as_tensor(
            self.support_weights, dtype=torch.float64
        ).detach().cpu().contiguous()
        support = torch.as_tensor(
            self.support_mask, dtype=torch.bool
        ).detach().cpu().contiguous()
        proposal_legal = torch.as_tensor(
            self.proposal_legal, dtype=torch.bool
        ).detach().cpu().contiguous()
        verification = torch.as_tensor(
            self.verification_mask, dtype=torch.bool
        ).detach().cpu().contiguous()
        verification_legal = torch.as_tensor(
            self.verification_legal, dtype=torch.bool
        ).detach().cpu().contiguous()
        expected_prefix = (MAX_HYPOTHESES, support.shape[1]) if support.ndim == 2 else ()
        if (
            features.ndim != 3
            or tuple(features.shape[:2]) != expected_prefix
            or features.shape[2] != PATCH_FEATURE_DIM
            or weights.shape != support.shape
            or proposal_legal.shape != (MAX_HYPOTHESES,)
            or verification.shape != support.shape
            or verification_legal.shape != (MAX_HYPOTHESES,)
            or not features.is_floating_point()
            or not bool(torch.isfinite(features).all())
            or not bool(torch.isfinite(weights).all())
        ):
            raise ValueError("multi-patch verification feature schema drift")
        if bool((support & ~proposal_legal[:, None]).any()):
            raise ValueError("proposal-illegal H1 slots cannot retain support")
        if bool((verification & ~support).any()):
            raise ValueError("verification mask must be a subset of proposal support")
        expected_legal = proposal_legal & verification.sum(dim=1).ge(
            MIN_VERIFICATION_PATCHES
        )
        if not torch.equal(verification_legal, expected_legal):
            raise ValueError("verification legal mask does not match patch evidence")
        feature_mask = verification.to(features.device).unsqueeze(-1)
        if bool(features.masked_select(~feature_mask).ne(0).any()):
            raise ValueError("non-verification feature rows must be exact zero")
        if self.direction not in _DIRECTIONS:
            raise ValueError("direction must be a_to_b or b_to_a")
        object.__setattr__(self, "patch_features", features)
        object.__setattr__(self, "support_weights", weights)
        object.__setattr__(self, "support_mask", support)
        object.__setattr__(self, "proposal_legal", proposal_legal)
        object.__setattr__(self, "verification_mask", verification)
        object.__setattr__(self, "verification_legal", verification_legal)


@dataclass(frozen=True)
class AcceptedTargetLock:
    """V-authorized multi-patch locks; proposal functions never create this."""

    hypotheses: SealedMultiPatchHypotheses
    verification: MultiPatchVerificationFeatures
    accepted: torch.Tensor

    def __post_init__(self) -> None:
        accepted = torch.as_tensor(
            self.accepted, dtype=torch.bool
        ).detach().cpu().contiguous()
        if accepted.shape != (MAX_HYPOTHESES,):
            raise ValueError("accepted-lock mask must have eight entries")
        if self.hypotheses.direction != self.verification.direction:
            raise ValueError("proposal and verification directions differ")
        if not torch.equal(
            self.hypotheses.legal, self.verification.proposal_legal
        ):
            raise ValueError("proposal legal mask was not retained into V")
        if bool((accepted & ~self.verification.verification_legal).any()):
            raise ValueError("V cannot accept an illegal or single-point lock")
        object.__setattr__(self, "accepted", accepted)


def propose_seed_correspondences(
    query_proposal: torch.Tensor,
    reference_proposal: torch.Tensor,
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    direction: str,
    query_selection_mask: torch.Tensor | None = None,
    reference_selection_mask: torch.Tensor | None = None,
    maximum_hypotheses: int = MAX_HYPOTHESES,
) -> SeedCorrespondenceSet:
    """Select deterministic reciprocal-nearest P seeds without exporting scores."""

    query = _finite_tokens(
        query_proposal, grid_shape=query_grid_shape, name="query proposal"
    )
    reference = _finite_tokens(
        reference_proposal, grid_shape=reference_grid_shape, name="reference proposal"
    ).to(query.device)
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference proposal dimensions differ")
    if not 0 <= int(maximum_hypotheses) <= MAX_HYPOTHESES:
        raise ValueError("maximum hypotheses must be between zero and eight")
    query_mask = _selection_mask(
        query_selection_mask,
        count=query.shape[0],
        device=query.device,
        name="query selection mask",
    )
    reference_mask = _selection_mask(
        reference_selection_mask,
        count=reference.shape[0],
        device=query.device,
        name="reference selection mask",
    )
    query_active = torch.nonzero(query_mask, as_tuple=False).flatten()
    reference_active = torch.nonzero(reference_mask, as_tuple=False).flatten()
    query_unit = F.normalize(query[query_active].to(torch.float64), dim=1, eps=1.0e-12)
    reference_unit = F.normalize(
        reference[reference_active].to(device=query.device, dtype=torch.float64),
        dim=1,
        eps=1.0e-12,
    )
    similarity = query_unit @ reference_unit.T
    query_best = similarity.argmax(dim=1)
    reference_best = similarity.argmax(dim=0)
    candidates: list[tuple[float, int, int]] = []
    for query_position in range(query_active.numel()):
        reference_position = int(query_best[query_position])
        if int(reference_best[reference_position]) != query_position:
            continue
        candidates.append(
            (
                -float(similarity[query_position, reference_position]),
                int(query_active[query_position]),
                int(reference_active[reference_position]),
            )
        )
    candidates.sort()
    selected = candidates[: int(maximum_hypotheses)]
    return SeedCorrespondenceSet(
        query_indices=torch.tensor([item[1] for item in selected], dtype=torch.long),
        reference_indices=torch.tensor(
            [item[2] for item in selected], dtype=torch.long
        ),
        query_grid_shape=query_grid_shape,
        reference_grid_shape=reference_grid_shape,
        direction=direction,
    )


def _local_cells(
    center_index: int,
    *,
    grid_shape: tuple[int, int],
    radius: int,
    selection_mask: torch.Tensor,
) -> torch.Tensor:
    coordinates = _xy(
        torch.arange(math.prod(grid_shape), device=selection_mask.device), grid_shape
    )
    center = _xy(torch.tensor([center_index]), grid_shape)[0].to(coordinates.device)
    local = (coordinates - center).abs().amax(dim=1).le(radius) & selection_mask
    return torch.nonzero(local, as_tuple=False).flatten()


def build_seed_local_hypotheses(
    query_proposal: torch.Tensor,
    reference_proposal: torch.Tensor,
    seeds: SeedCorrespondenceSet,
    *,
    query_selection_mask: torch.Tensor | None = None,
    reference_selection_mask: torch.Tensor | None = None,
    support_radius: int = DEFAULT_SUPPORT_RADIUS,
    maximum_support: int = DEFAULT_MAX_SUPPORT,
) -> SealedMultiPatchHypotheses:
    """Expand reciprocal seeds into structurally qualified local supports."""

    query = _finite_tokens(
        query_proposal, grid_shape=seeds.query_grid_shape, name="query proposal"
    )
    reference = _finite_tokens(
        reference_proposal,
        grid_shape=seeds.reference_grid_shape,
        name="reference proposal",
    ).to(query.device)
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference proposal dimensions differ")
    if isinstance(support_radius, bool) or not isinstance(support_radius, int) or support_radius < 1:
        raise ValueError("support radius must be a positive integer")
    if (
        isinstance(maximum_support, bool)
        or not isinstance(maximum_support, int)
        or maximum_support < MIN_SUPPORT
    ):
        raise ValueError("maximum support must be at least four")
    query_mask = _selection_mask(
        query_selection_mask,
        count=query.shape[0],
        device=query.device,
        name="query selection mask",
    )
    reference_mask = _selection_mask(
        reference_selection_mask,
        count=reference.shape[0],
        device=query.device,
        name="reference selection mask",
    )
    if (
        seeds.query_grid_shape != _shape(seeds.query_grid_shape, name="query grid")
        or seeds.reference_grid_shape
        != _shape(seeds.reference_grid_shape, name="reference grid")
    ):
        raise RuntimeError("seed grid schema drift")

    seed_query = torch.full((MAX_HYPOTHESES,), -1, dtype=torch.long)
    seed_reference = torch.full((MAX_HYPOTHESES,), -1, dtype=torch.long)
    support_query = torch.full(
        (MAX_HYPOTHESES, maximum_support), -1, dtype=torch.long
    )
    support_reference = torch.full_like(support_query, -1)
    support_weights = torch.zeros(
        (MAX_HYPOTHESES, maximum_support), dtype=torch.float64
    )
    support_mask = torch.zeros(
        (MAX_HYPOTHESES, maximum_support), dtype=torch.bool
    )
    legal = torch.zeros(MAX_HYPOTHESES, dtype=torch.bool)
    query_unit = F.normalize(query.to(torch.float64), dim=1, eps=1.0e-12)
    reference_unit = F.normalize(
        reference.to(device=query.device, dtype=torch.float64), dim=1, eps=1.0e-12
    )
    seen_supports: set[tuple[tuple[int, int], ...]] = set()
    output_index = 0

    for seed_index in range(len(seeds)):
        if output_index == MAX_HYPOTHESES:
            break
        query_seed = int(seeds.query_indices[seed_index])
        reference_seed = int(seeds.reference_indices[seed_index])
        local_query = _local_cells(
            query_seed,
            grid_shape=seeds.query_grid_shape,
            radius=support_radius,
            selection_mask=query_mask,
        )
        local_reference = _local_cells(
            reference_seed,
            grid_shape=seeds.reference_grid_shape,
            radius=support_radius,
            selection_mask=reference_mask,
        )
        similarity = query_unit[local_query] @ reference_unit[local_reference].T
        query_best = similarity.argmax(dim=1)
        reference_best = similarity.argmax(dim=0)
        pairs: list[tuple[float, int, int]] = []
        for query_position in range(local_query.numel()):
            reference_position = int(query_best[query_position])
            if int(reference_best[reference_position]) != query_position:
                continue
            pairs.append(
                (
                    -float(similarity[query_position, reference_position]),
                    int(local_query[query_position]),
                    int(local_reference[reference_position]),
                )
            )
        pairs.sort()
        seed_pair = (query_seed, reference_seed)
        ordered = [pair for pair in pairs if (pair[1], pair[2]) == seed_pair]
        ordered.extend(pair for pair in pairs if (pair[1], pair[2]) != seed_pair)
        selected = ordered[:maximum_support]
        if len(selected) < MIN_SUPPORT or not selected or (selected[0][1], selected[0][2]) != seed_pair:
            continue
        selected_query = torch.tensor([item[1] for item in selected], dtype=torch.long)
        selected_reference = torch.tensor(
            [item[2] for item in selected], dtype=torch.long
        )
        if not _has_2d_span(
            selected_query, seeds.query_grid_shape
        ) or not _has_2d_span(selected_reference, seeds.reference_grid_shape):
            continue
        signature = tuple(
            sorted(
                zip(
                    selected_query.tolist(),
                    selected_reference.tolist(),
                    strict=True,
                )
            )
        )
        if signature in seen_supports:
            continue
        seen_supports.add(signature)
        count = len(selected)
        seed_query[output_index] = query_seed
        seed_reference[output_index] = reference_seed
        support_query[output_index, :count] = selected_query
        support_reference[output_index, :count] = selected_reference
        support_weights[output_index, :count] = 1.0 / float(count)
        support_mask[output_index, :count] = True
        legal[output_index] = True
        output_index += 1

    return SealedMultiPatchHypotheses(
        seed_query_indices=seed_query,
        seed_reference_indices=seed_reference,
        support_query_indices=support_query,
        support_reference_indices=support_reference,
        support_weights=support_weights,
        support_mask=support_mask,
        legal=legal,
        query_grid_shape=seeds.query_grid_shape,
        reference_grid_shape=seeds.reference_grid_shape,
        direction=seeds.direction,
    )


def propose_multisupport_p_only(
    query_proposal: torch.Tensor,
    reference_proposal: torch.Tensor,
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    direction: str,
    query_selection_mask: torch.Tensor | None = None,
    reference_selection_mask: torch.Tensor | None = None,
    support_radius: int = DEFAULT_SUPPORT_RADIUS,
    maximum_support: int = DEFAULT_MAX_SUPPORT,
) -> SealedMultiPatchHypotheses:
    """P-only convenience wrapper; it never returns an accepted lock."""

    seeds = propose_seed_correspondences(
        query_proposal,
        reference_proposal,
        query_grid_shape=query_grid_shape,
        reference_grid_shape=reference_grid_shape,
        direction=direction,
        query_selection_mask=query_selection_mask,
        reference_selection_mask=reference_selection_mask,
    )
    return build_seed_local_hypotheses(
        query_proposal,
        reference_proposal,
        seeds,
        query_selection_mask=query_selection_mask,
        reference_selection_mask=reference_selection_mask,
        support_radius=support_radius,
        maximum_support=maximum_support,
    )


def verify_multisupport_v_only(
    query_verification: torch.Tensor,
    reference_verification: torch.Tensor,
    hypotheses: SealedMultiPatchHypotheses,
    *,
    query_verification_grid_shape: tuple[int, int] | None = None,
    reference_verification_grid_shape: tuple[int, int] | None = None,
    query_verification_mask: torch.Tensor | None = None,
    reference_verification_mask: torch.Tensor | None = None,
    leaveout_halo: int = DEFAULT_LEAVEOUT_HALO,
) -> MultiPatchVerificationFeatures:
    """Build patch-level V features after seed-and-halo leave-out."""

    query_v_shape = (
        hypotheses.query_grid_shape
        if query_verification_grid_shape is None
        else _shape(query_verification_grid_shape, name="query verification grid")
    )
    reference_v_shape = (
        hypotheses.reference_grid_shape
        if reference_verification_grid_shape is None
        else _shape(
            reference_verification_grid_shape, name="reference verification grid"
        )
    )
    query = _finite_tokens(
        query_verification,
        grid_shape=query_v_shape,
        name="query verification",
    )
    reference = _finite_tokens(
        reference_verification,
        grid_shape=reference_v_shape,
        name="reference verification",
    ).to(query.device)
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference verification dimensions differ")
    if isinstance(leaveout_halo, bool) or not isinstance(leaveout_halo, int) or leaveout_halo < 0:
        raise ValueError("leaveout halo must be a nonnegative integer")
    query_mask = _selection_mask(
        query_verification_mask,
        count=query.shape[0],
        device=query.device,
        name="query verification mask",
    )
    reference_mask = _selection_mask(
        reference_verification_mask,
        count=reference.shape[0],
        device=query.device,
        name="reference verification mask",
    )
    support_capacity = hypotheses.support_mask.shape[1]
    features = torch.zeros(
        (MAX_HYPOTHESES, support_capacity, PATCH_FEATURE_DIM),
        dtype=torch.float32,
        device=query.device,
    )
    verification_mask = torch.zeros_like(hypotheses.support_mask)
    query_unit = F.normalize(query.to(torch.float32), dim=1, eps=1.0e-12)
    reference_unit = F.normalize(
        reference.to(device=query.device, dtype=torch.float32), dim=1, eps=1.0e-12
    )

    for hypothesis_index in range(MAX_HYPOTHESES):
        if not bool(hypotheses.legal[hypothesis_index]):
            continue
        active_positions = torch.nonzero(
            hypotheses.support_mask[hypothesis_index], as_tuple=False
        ).flatten()
        proposal_query_indices = hypotheses.support_query_indices[
            hypothesis_index, active_positions
        ]
        proposal_reference_indices = hypotheses.support_reference_indices[
            hypothesis_index, active_positions
        ]
        query_seed_xy = _xy(
            hypotheses.seed_query_indices[hypothesis_index].reshape(1),
            hypotheses.query_grid_shape,
        )[0]
        reference_seed_xy = _xy(
            hypotheses.seed_reference_indices[hypothesis_index].reshape(1),
            hypotheses.reference_grid_shape,
        )[0]
        query_distance = (
            _xy(proposal_query_indices, hypotheses.query_grid_shape) - query_seed_xy
        ).abs().amax(dim=1)
        reference_distance = (
            _xy(proposal_reference_indices, hypotheses.reference_grid_shape)
            - reference_seed_xy
        ).abs().amax(dim=1)
        mapped_query_indices = _map_grid_indices(
            proposal_query_indices,
            source_grid_shape=hypotheses.query_grid_shape,
            destination_grid_shape=query_v_shape,
        )
        mapped_reference_indices = _map_grid_indices(
            proposal_reference_indices,
            source_grid_shape=hypotheses.reference_grid_shape,
            destination_grid_shape=reference_v_shape,
        )
        allowed = (
            query_distance.gt(leaveout_halo)
            & reference_distance.gt(leaveout_halo)
            & query_mask[mapped_query_indices.to(query.device)].detach().cpu()
            & reference_mask[
                mapped_reference_indices.to(query.device)
            ].detach().cpu()
        )
        # A coarse V grid may merge multiple P cells.  Retain the first
        # canonical support position for each mapped query/reference pair so
        # verification evidence remains one-to-one after cross-backbone map.
        verified_position_values: list[int] = []
        seen_query: set[int] = set()
        seen_reference: set[int] = set()
        for local_position in torch.nonzero(allowed, as_tuple=False).flatten().tolist():
            query_index = int(mapped_query_indices[local_position])
            reference_index = int(mapped_reference_indices[local_position])
            if query_index in seen_query or reference_index in seen_reference:
                continue
            seen_query.add(query_index)
            seen_reference.add(reference_index)
            verified_position_values.append(int(active_positions[local_position]))
        verified_positions = torch.tensor(
            verified_position_values, dtype=torch.long
        )
        if verified_positions.numel() < MIN_VERIFICATION_PATCHES:
            continue
        verification_mask[hypothesis_index, verified_positions] = True
        selected_in_active = torch.tensor(
            [
                int((active_positions == position).nonzero().item())
                for position in verified_position_values
            ],
            dtype=torch.long,
        )
        verified_query = mapped_query_indices[selected_in_active].to(query.device)
        verified_reference = mapped_reference_indices[selected_in_active].to(
            query.device
        )
        similarity = query_unit[verified_query] @ reference_unit[verified_reference].T
        diagonal = similarity.diagonal()
        count = int(diagonal.numel())
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=query.device)
        row_mean = similarity.masked_select(off_diagonal).reshape(count, count - 1).mean(dim=1)
        column_mean = (
            similarity.T.masked_select(off_diagonal)
            .reshape(count, count - 1)
            .mean(dim=1)
        )
        other_diagonal = (diagonal.sum() - diagonal) / float(count - 1)
        patch_features = torch.stack(
            (
                diagonal,
                diagonal - row_mean,
                diagonal - column_mean,
                other_diagonal,
            ),
            dim=1,
        )
        features[hypothesis_index, verified_positions.to(query.device)] = patch_features

    verification_legal = hypotheses.legal & verification_mask.sum(dim=1).ge(
        MIN_VERIFICATION_PATCHES
    )
    return MultiPatchVerificationFeatures(
        patch_features=features,
        support_weights=hypotheses.support_weights,
        support_mask=hypotheses.support_mask,
        proposal_legal=hypotheses.legal,
        verification_mask=verification_mask,
        verification_legal=verification_legal,
        direction=hypotheses.direction,
    )


def score_candidate_features(
    verification: MultiPatchVerificationFeatures,
    patch_head: nn.Module | Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Score only legal V rows; return exact H0=0 when none are legal."""

    if not bool(verification.verification_legal.any()):
        if isinstance(patch_head, nn.Module):
            first_parameter = next(patch_head.parameters(), None)
            if first_parameter is not None:
                return torch.zeros(
                    (), device=first_parameter.device, dtype=first_parameter.dtype
                )
        return verification.patch_features.new_zeros(())
    mask = verification.verification_mask.to(verification.patch_features.device)
    valid_features = verification.patch_features[mask]
    if isinstance(patch_head, nn.Module):
        first_parameter = next(patch_head.parameters(), None)
        if first_parameter is not None:
            valid_features = valid_features.to(
                device=first_parameter.device, dtype=first_parameter.dtype
            )
    valid_logits = torch.as_tensor(patch_head(valid_features))
    if valid_logits.shape != (valid_features.shape[0],) or not bool(
        torch.isfinite(valid_logits).all()
    ):
        raise ValueError("patch head must return one finite scalar per legal V row")
    mask = mask.to(valid_logits.device)
    patch_logits = valid_logits.new_zeros(mask.shape)
    patch_logits[mask] = valid_logits
    hypothesis_scores: list[torch.Tensor] = []
    weights = verification.support_weights.to(
        device=patch_logits.device, dtype=patch_logits.dtype
    )
    for hypothesis_index in range(MAX_HYPOTHESES):
        if not bool(verification.verification_legal[hypothesis_index]):
            continue
        active = mask[hypothesis_index]
        active_weights = weights[hypothesis_index, active]
        active_weights = active_weights / active_weights.sum()
        hypothesis_scores.append(
            (patch_logits[hypothesis_index, active] * active_weights).sum()
        )
    # H0 is an explicit zero logit.  Illegal H1 slots are absent, not zero
    # feature rows passed through a biased head.
    logits = torch.stack((patch_logits.new_zeros(()), *hypothesis_scores))
    return torch.logsumexp(logits, dim=0) - math.log(float(logits.numel()))


def seal_verified_locks(
    hypotheses: SealedMultiPatchHypotheses,
    verification: MultiPatchVerificationFeatures,
    accept_mask: torch.Tensor,
) -> AcceptedTargetLock:
    """Create the accepted-lock type only after an explicit V decision."""

    return AcceptedTargetLock(hypotheses, verification, accept_mask)


def spatially_transform_hypotheses(
    hypotheses: SealedMultiPatchHypotheses,
    *,
    transform: str,
    transform_query: bool = False,
    transform_reference: bool = True,
) -> SealedMultiPatchHypotheses:
    """Apply a topology-preserving spatial destruction before V feature formation."""

    if not transform_query and not transform_reference:
        raise ValueError("spatial control must transform at least one side")
    if transform not in {"horizontal_flip", "vertical_flip", "half_width_cycle"}:
        raise ValueError("unregistered spatial transform")

    def apply(indices: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
        value = indices.clone()
        active = value.ge(0)
        height, width = grid_shape
        y = torch.div(value[active], width, rounding_mode="floor")
        x = value[active].remainder(width)
        if transform == "horizontal_flip":
            mapped = y * width + (width - 1 - x)
        elif transform == "vertical_flip":
            mapped = (height - 1 - y) * width + x
        else:
            if width < 2:
                raise ValueError("half-width cycle requires reference width >= 2")
            shift = max(1, width // 2)
            mapped = y * width + (x + shift).remainder(width)
        value[active] = mapped
        return value

    seed_query = hypotheses.seed_query_indices
    support_query = hypotheses.support_query_indices
    seed_reference = hypotheses.seed_reference_indices
    support_reference = hypotheses.support_reference_indices
    if transform_query:
        seed_query = apply(seed_query, hypotheses.query_grid_shape)
        support_query = apply(support_query, hypotheses.query_grid_shape)
    if transform_reference:
        seed_reference = apply(seed_reference, hypotheses.reference_grid_shape)
        support_reference = apply(
            support_reference, hypotheses.reference_grid_shape
        )
    return SealedMultiPatchHypotheses(
        seed_query_indices=seed_query,
        seed_reference_indices=seed_reference,
        support_query_indices=support_query,
        support_reference_indices=support_reference,
        support_weights=hypotheses.support_weights,
        support_mask=hypotheses.support_mask,
        legal=hypotheses.legal,
        query_grid_shape=hypotheses.query_grid_shape,
        reference_grid_shape=hypotheses.reference_grid_shape,
        direction=hypotheses.direction,
    )


def candidate_binding_control_features(
    query_verification: torch.Tensor,
    reference_verifications: Sequence[torch.Tensor],
    hypotheses: Sequence[SealedMultiPatchHypotheses],
    source_for_destination: torch.Tensor,
    *,
    query_verification_grid_shape: tuple[int, int] | None = None,
    reference_verification_grid_shapes: Sequence[tuple[int, int]] | None = None,
    query_verification_mask: torch.Tensor | None = None,
    reference_verification_masks: Sequence[torch.Tensor | None] | None = None,
    leaveout_halo: int = DEFAULT_LEAVEOUT_HALO,
) -> list[MultiPatchVerificationFeatures]:
    """Break P-to-candidate binding while leaving candidate slots fixed."""

    count = len(reference_verifications)
    masks = (
        list(reference_verification_masks)
        if reference_verification_masks is not None
        else [None] * count
    )
    reference_shapes = (
        list(reference_verification_grid_shapes)
        if reference_verification_grid_shapes is not None
        else [item.reference_grid_shape for item in hypotheses]
    )
    order = torch.as_tensor(source_for_destination, dtype=torch.long)
    if (
        count < 2
        or len(hypotheses) != count
        or len(masks) != count
        or len(reference_shapes) != count
        or order.shape != (count,)
        or order.unique().numel() != count
        or bool(order.eq(torch.arange(count)).any())
    ):
        raise ValueError("candidate-binding control requires a complete derangement")
    return [
        verify_multisupport_v_only(
            query_verification,
            reference_verifications[int(order[destination])],
            hypotheses[destination],
            query_verification_grid_shape=query_verification_grid_shape,
            reference_verification_grid_shape=reference_shapes[
                int(order[destination])
            ],
            query_verification_mask=query_verification_mask,
            reference_verification_mask=masks[int(order[destination])],
            leaveout_halo=leaveout_halo,
        )
        for destination in range(count)
    ]


def candidate_reorder_equivariance_error(
    query_verification: torch.Tensor,
    reference_verifications: Sequence[torch.Tensor],
    hypotheses: Sequence[SealedMultiPatchHypotheses],
    patch_head: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    permutation: torch.Tensor,
    *,
    query_verification_grid_shape: tuple[int, int] | None = None,
    reference_verification_grid_shapes: Sequence[tuple[int, int]] | None = None,
    query_verification_mask: torch.Tensor | None = None,
    reference_verification_masks: Sequence[torch.Tensor | None] | None = None,
    leaveout_halo: int = DEFAULT_LEAVEOUT_HALO,
) -> torch.Tensor:
    """Reorder references and their hypotheses together; this is not C."""

    count = len(reference_verifications)
    masks = (
        list(reference_verification_masks)
        if reference_verification_masks is not None
        else [None] * count
    )
    reference_shapes = (
        list(reference_verification_grid_shapes)
        if reference_verification_grid_shapes is not None
        else [item.reference_grid_shape for item in hypotheses]
    )
    order = torch.as_tensor(permutation, dtype=torch.long)
    if (
        len(hypotheses) != count
        or len(masks) != count
        or len(reference_shapes) != count
        or order.shape != (count,)
        or order.unique().numel() != count
    ):
        raise ValueError("candidate reorder must be a complete permutation")

    def score(
        reference: torch.Tensor,
        item: SealedMultiPatchHypotheses,
        reference_mask: torch.Tensor | None,
        reference_shape: tuple[int, int],
    ) -> torch.Tensor:
        features = verify_multisupport_v_only(
            query_verification,
            reference,
            item,
            query_verification_grid_shape=query_verification_grid_shape,
            reference_verification_grid_shape=reference_shape,
            query_verification_mask=query_verification_mask,
            reference_verification_mask=reference_mask,
            leaveout_halo=leaveout_halo,
        )
        return score_candidate_features(features, patch_head)

    base = torch.stack(
        [
            score(
                reference_verifications[index], hypotheses[index], masks[index],
                reference_shapes[index]
            )
            for index in range(count)
        ]
    )
    reordered = torch.stack(
        [
            score(
                reference_verifications[int(index)],
                hypotheses[int(index)],
                masks[int(index)],
                reference_shapes[int(index)],
            )
            for index in order
        ]
    )
    return (reordered - base[order.to(base.device)]).abs().max()
