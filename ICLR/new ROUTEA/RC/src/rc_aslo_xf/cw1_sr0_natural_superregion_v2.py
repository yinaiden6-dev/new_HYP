"""Target-free natural connected-superregion bridge (V2).

This module is a clean successor to ``cw1_sr0_natural_native_span_v1``.  It
does not modify that historical implementation.  The important separation is
explicit:

* P sees frozen ColNomic local descriptors and proposes exactly one r1--r4
  connected query super-region for one anonymous reference candidate;
* V is not allowed to select another region and reads only the complementary
  query cells belonging to that sealed region;
* every r1--r4 P feature remains available in a fixed-width ledger, while V
  columns are marked observed only for the one P-selected row.

There is deliberately no label join, D1/rank input, ownership head, retrieval
fusion, SWITCH/HOLD policy, natural threshold, or endpoint-specific tuning in
this core.  It is a representation/mechanism primitive, not a scientific
GO/NO-GO reducer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .cw0_connected_region_v2 import (
    enumerate_query_macro_seeds,
    enumerate_reference_macro_seeds,
)
from .cw1_sr0_structure_v1 import (
    CW1SR0SuperRegion,
    enumerate_superregion_bank,
    superregion_bank_sha256,
    superregion_sha256,
)
from .geometry_hypothesis_v1 import (
    connected_components_4,
    fixed_macro_bank_split,
    grid_cell_centres,
)


SCHEMA_VERSION = "rc_cw1_sr0_natural_superregion_bridge_v2"
DESCRIPTOR_DIMENSION = 128
TEMPERATURE = 0.07
MINIMUM_P_CELLS = 4
MINIMUM_V_CELLS = 4
REFERENCE_FRINGE_RADIUS = 1

CONTROL_BASE = "BASE"
CONTROL_V_C_BIND = "V_C_BIND"
CONTROL_V_REF_SPATIAL = "V_REF_SPATIAL"
CONTROL_MODES = (CONTROL_BASE, CONTROL_V_C_BIND, CONTROL_V_REF_SPATIAL)

PROTECTED_ZERO: Mapping[str, int] = {
    "labels": 0,
    "D1": 0,
    "candidate_rank": 0,
    "opened": 0,
    "sealed": 0,
    "human_spatial_annotation": 0,
}

REGION_FEATURE_COLUMNS = (
    "p_score",
    "p_proposal_support",
    "p_action_margin",
    "p_eligible_mass",
    "p_root_count",
    "p_eligible_root_count",
    "p_reference_iou",
    "p_affine_agreement",
    "v_score",
    "v_valid_mass",
    "v_positive_mass",
    "v_observed",
)


class NaturalSuperregionV2Error(RuntimeError):
    pass


def _grid_shape(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(value) != 2 or any(isinstance(item, bool) for item in value):
        raise NaturalSuperregionV2Error(f"{name} is not a grid shape")
    height, width = map(int, value)
    if min(height, width) <= 0:
        raise NaturalSuperregionV2Error(f"{name} is not positive")
    return height, width


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _unit_tokens(
    value: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().contiguous()
    if (
        tensor.shape != (math.prod(grid_shape), DESCRIPTOR_DIMENSION)
        or tensor.dtype not in {torch.float16, torch.float32, torch.float64}
        or not bool(torch.isfinite(tensor).all())
    ):
        raise NaturalSuperregionV2Error(
            f"{name} must be finite native-grid [cells,128]"
        )
    return F.normalize(tensor.to(torch.float32), p=2, dim=1, eps=1.0e-12)


def _dilate_once(mask: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    height, width = grid_shape
    source = torch.as_tensor(mask, dtype=torch.bool).reshape(height, width)
    padded = F.pad(source, (1, 1, 1, 1), value=False)
    output = torch.zeros_like(source)
    for y_offset in range(3):
        for x_offset in range(3):
            output |= padded[
                y_offset : y_offset + height,
                x_offset : x_offset + width,
            ]
    return output.reshape(-1).contiguous()


@dataclass(frozen=True)
class NativeReferenceActionV2:
    ordinal: int
    source_root_ordinal: int
    transform: str
    mask: torch.Tensor
    member_indices: torch.Tensor
    mask_sha256: str

    def __post_init__(self) -> None:
        if self.transform not in {"I", "ROT180"}:
            raise ValueError("unknown native reference transform")
        mask = torch.as_tensor(self.mask, dtype=torch.bool).detach().cpu().contiguous()
        members = torch.as_tensor(
            self.member_indices, dtype=torch.long
        ).detach().cpu().contiguous()
        if mask.ndim != 1 or not torch.equal(
            members, torch.nonzero(mask, as_tuple=False).flatten()
        ):
            raise ValueError("native reference action membership drift")
        if self.mask_sha256 != tensor_sha256(mask):
            raise ValueError("native reference action hash drift")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "member_indices", members)


def enumerate_native_reference_actions_v2(
    reference_grid_shape: tuple[int, int],
) -> tuple[NativeReferenceActionV2, ...]:
    """Enumerate and deduplicate the native I/ROT180 action population.

    Deduplication happens *after* the registered one-cell fringe.  Therefore a
    small or unusual native grid cannot obtain extra proposal prior merely
    because multiple semantic actions collapse to the same physical mask.
    """

    shape = _grid_shape(reference_grid_shape, name="reference grid")
    height, width = shape
    bases = enumerate_reference_macro_seeds(shape)
    output: list[NativeReferenceActionV2] = []
    seen: dict[str, torch.Tensor] = {}
    for transform in ("I", "ROT180"):
        for root, item in enumerate(bases):
            mask = item.window.mask
            if transform == "ROT180":
                mask = torch.flip(
                    mask.reshape(height, width), (0, 1)
                ).reshape(-1)
            mask = _dilate_once(mask, shape)
            digest = tensor_sha256(mask)
            if digest in seen:
                if not torch.equal(seen[digest], mask):
                    raise RuntimeError("reference action hash collision")
                continue
            diagnostics = connected_components_4(mask, shape)[1]
            if (
                not diagnostics.connected_valid
                or diagnostics.component_count != 1
                or not diagnostics.has_2d_span
            ):
                raise NaturalSuperregionV2Error(
                    "native reference action is not one connected 2-D region"
                )
            seen[digest] = mask
            output.append(
                NativeReferenceActionV2(
                    ordinal=len(output),
                    source_root_ordinal=root,
                    transform=transform,
                    mask=mask,
                    member_indices=torch.nonzero(mask, as_tuple=False).flatten(),
                    mask_sha256=digest,
                )
            )
    if not output:
        raise NaturalSuperregionV2Error("native reference action bank is empty")
    return tuple(output)


def _apply_affine(affine: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    value = xy.detach().cpu().to(torch.float64)
    design = torch.cat(
        (value, torch.ones((value.shape[0], 1), dtype=torch.float64)), dim=1
    )
    return design @ affine[:2, :].T


def _fit_affine(
    query_xy: torch.Tensor, reference_xy: torch.Tensor
) -> tuple[torch.Tensor, float, bool]:
    q = query_xy.detach().cpu().to(torch.float64)
    r = reference_xy.detach().cpu().to(torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    if q.shape[0] < MINIMUM_P_CELLS:
        return identity, float("inf"), False
    design = torch.cat(
        (q, torch.ones((q.shape[0], 1), dtype=torch.float64)), dim=1
    )
    if int(torch.linalg.matrix_rank(design)) < 3:
        return identity, float("inf"), False
    gram = design.T @ design + 1.0e-9 * torch.eye(3, dtype=torch.float64)
    try:
        solution = torch.linalg.solve(gram, design.T @ r)
    except RuntimeError:
        return identity, float("inf"), False
    predicted = design @ solution
    rmse = float(torch.linalg.vector_norm(predicted - r, dim=1).square().mean().sqrt())
    affine = torch.eye(3, dtype=torch.float64)
    affine[:2, :] = solution.T
    determinant = float(torch.linalg.det(affine[:2, :2]))
    legal = bool(torch.isfinite(affine).all() and math.isfinite(rmse) and determinant > 0.0)
    return affine, rmse, legal


def _bilinear_sample(
    tokens: torch.Tensor,
    grid_shape: tuple[int, int],
    xy: torch.Tensor,
) -> torch.Tensor:
    height, width = grid_shape
    field = tokens.reshape(height, width, DESCRIPTOR_DIMENSION).permute(2, 0, 1)[None]
    coordinate = xy.to(device=tokens.device, dtype=torch.float32)
    grid = coordinate.mul(2.0).sub(1.0).reshape(1, 1, -1, 2)
    sampled = F.grid_sample(
        field,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0, :, 0, :].T.contiguous()


@dataclass(frozen=True)
class NaturalPDirectionV2:
    direction: int
    p_query_indices: torch.Tensor
    v_query_indices: torch.Tensor
    selected_action: int
    proposal_support: float
    action_margin: float
    affine: torch.Tensor
    affine_rmse: float
    eligible: bool


@dataclass(frozen=True)
class NaturalPRootV2:
    root_ordinal: int
    root_mask: torch.Tensor
    directions: tuple[NaturalPDirectionV2, NaturalPDirectionV2]
    proposal_support: float
    action_margin: float
    reference_iou: float
    affine_disagreement: float
    affine_agreement: float
    p_score: float
    eligible: bool


@dataclass(frozen=True)
class NaturalPRegionV2:
    region_ordinal: int
    radius: int
    cell_count: int
    region_sha256: str
    contributing_root_ordinals: tuple[int, ...]
    p_score: float
    proposal_support: float
    action_margin: float
    eligible_mass: float
    root_count: int
    eligible_root_count: int
    reference_iou: float
    affine_agreement: float


@dataclass(frozen=True)
class NaturalSuperregionPreparationV2:
    schema_version: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_tokens_sha256: str
    reference_tokens_sha256: str
    query_root_mask_sha256: tuple[str, ...]
    reference_action_mask_sha256: tuple[str, ...]
    region_bank_sha256: str
    roots: tuple[NaturalPRootV2, ...]
    regions: tuple[NaturalPRegionV2, ...]
    selected_region_ordinal: int | None
    seal_sha256: str
    protected_access_counts: Mapping[str, int]


@dataclass(frozen=True)
class NaturalVRootV2:
    root_ordinal: int
    direction_scores: tuple[float, float]
    valid_v_patch_counts: tuple[int, int]
    total_v_patch_counts: tuple[int, int]
    score: float
    valid_fraction: float
    positive_mass: float
    eligible: bool


@dataclass(frozen=True)
class NaturalSuperregionVerificationV2:
    schema_version: str
    preparation_seal_sha256: str
    control_mode: str
    verification_reference_sha256: str
    selected_region_ordinal: int | None
    roots: tuple[NaturalVRootV2, ...]
    selected_v_score: float
    selected_v_valid_mass: float
    selected_v_positive_mass: float
    v_read_query_mask: torch.Tensor
    seal_sha256: str
    protected_access_counts: Mapping[str, int]


@dataclass(frozen=True)
class NaturalSuperregionFeatureLedgerV2:
    schema_version: str
    preparation_seal_sha256: str
    verification_seal_sha256: str
    region_keys: tuple[tuple[int, int, str, int], ...]
    feature_columns: tuple[str, ...]
    features: torch.Tensor
    selected_region_ordinal: int | None
    logical_sha256: str


def _action_support_table(
    similarity: torch.Tensor,
    actions: Sequence[NativeReferenceActionV2],
) -> torch.Tensor:
    masks = torch.stack([item.mask for item in actions]).to(
        device=similarity.device, dtype=similarity.dtype
    )
    counts = masks.sum(dim=1).clamp_min(1.0)
    # Cosine similarity lies in [-1,1], hence exp(sim/0.07) is finite in
    # float32.  The dense formulation avoids a Python loop over query cells.
    return TEMPERATURE * torch.log(
        (torch.exp(similarity / TEMPERATURE) @ masks.T) / counts[None]
    )


def _prepare_direction(
    *,
    direction: int,
    root_mask: torch.Tensor,
    query_bank: torch.Tensor,
    action_support: torch.Tensor,
    similarity: torch.Tensor,
    query_xy: torch.Tensor,
    reference_xy: torch.Tensor,
    actions: Sequence[NativeReferenceActionV2],
) -> NaturalPDirectionV2:
    p_mask = root_mask & query_bank
    v_mask = root_mask & ~query_bank
    p_indices = torch.nonzero(p_mask, as_tuple=False).flatten()
    v_indices = torch.nonzero(v_mask, as_tuple=False).flatten()
    if p_indices.numel() == 0:
        ranking = tuple(range(len(actions)))
        support = float("-inf")
        margin = 0.0
        affine = torch.eye(3, dtype=torch.float64)
        rmse = float("inf")
        eligible = False
    else:
        scores = action_support[p_indices.to(action_support.device)].mean(dim=0)
        ranking = tuple(
            sorted(
                range(len(actions)),
                key=lambda ordinal: (-float(scores[ordinal].detach()), ordinal),
            )
        )
        selected = ranking[0]
        support = float(scores[selected].detach())
        margin = (
            support - float(scores[ranking[1]].detach())
            if len(ranking) > 1
            else 0.0
        )
        reference_indices = actions[selected].member_indices
        local = similarity[p_indices.to(similarity.device)][
            :, reference_indices.to(similarity.device)
        ]
        weights = torch.softmax(local / TEMPERATURE, dim=1)
        decoded = weights.to(torch.float64) @ reference_xy[reference_indices].to(
            device=weights.device, dtype=torch.float64
        )
        affine, rmse, fit_legal = _fit_affine(query_xy[p_indices], decoded)
        eligible = bool(
            p_indices.numel() >= MINIMUM_P_CELLS
            and v_indices.numel() >= MINIMUM_V_CELLS
            and fit_legal
            and math.isfinite(support)
        )
    return NaturalPDirectionV2(
        direction=direction,
        p_query_indices=p_indices,
        v_query_indices=v_indices,
        selected_action=ranking[0],
        proposal_support=support,
        action_margin=margin,
        affine=affine,
        affine_rmse=rmse,
        eligible=eligible,
    )


def _prepare_root(
    *,
    root_ordinal: int,
    root_mask: torch.Tensor,
    query_banks: tuple[torch.Tensor, torch.Tensor],
    action_support: torch.Tensor,
    similarity: torch.Tensor,
    query_xy: torch.Tensor,
    reference_xy: torch.Tensor,
    actions: Sequence[NativeReferenceActionV2],
    reference_grid_shape: tuple[int, int],
) -> NaturalPRootV2:
    directions = tuple(
        _prepare_direction(
            direction=direction,
            root_mask=root_mask,
            query_bank=query_banks[direction],
            action_support=action_support,
            similarity=similarity,
            query_xy=query_xy,
            reference_xy=reference_xy,
            actions=actions,
        )
        for direction in (0, 1)
    )
    left = actions[directions[0].selected_action].mask
    right = actions[directions[1].selected_action].mask
    union = int((left | right).sum())
    reference_iou = 0.0 if union == 0 else float((left & right).sum()) / float(union)
    indices = torch.nonzero(root_mask, as_tuple=False).flatten()
    prediction_0 = _apply_affine(directions[0].affine, query_xy[indices])
    prediction_1 = _apply_affine(directions[1].affine, query_xy[indices])
    disagreement = float(
        torch.linalg.vector_norm(prediction_0 - prediction_1, dim=1).mean()
    )
    height, width = reference_grid_shape
    cell_diagonal = math.sqrt(1.0 / height**2 + 1.0 / width**2)
    affine_agreement = math.exp(-disagreement / cell_diagonal)
    support = 0.5 * sum(item.proposal_support for item in directions)
    action_margin = 0.5 * sum(item.action_margin for item in directions)
    eligible = all(item.eligible for item in directions)
    # Region proposal selection is driven by P support alone.  Reference-mask
    # IoU and affine agreement remain separately auditable features; fixing
    # them as multiplicative gates here would erase otherwise useful P signal.
    p_score = support if eligible else 0.0
    return NaturalPRootV2(
        root_ordinal=root_ordinal,
        root_mask=root_mask,
        directions=(directions[0], directions[1]),
        proposal_support=support,
        action_margin=action_margin,
        reference_iou=reference_iou,
        affine_disagreement=disagreement,
        affine_agreement=affine_agreement,
        p_score=p_score,
        eligible=eligible,
    )


def _region_rows(
    regions: Sequence[CW1SR0SuperRegion],
    roots: Sequence[NaturalPRootV2],
) -> tuple[NaturalPRegionV2, ...]:
    p_score = torch.tensor([item.p_score for item in roots], dtype=torch.float64)
    support = torch.tensor(
        [item.proposal_support if math.isfinite(item.proposal_support) else 0.0 for item in roots],
        dtype=torch.float64,
    )
    margin = torch.tensor([item.action_margin for item in roots], dtype=torch.float64)
    eligible = torch.tensor([item.eligible for item in roots], dtype=torch.float64)
    reference_iou = torch.tensor([item.reference_iou for item in roots], dtype=torch.float64)
    affine_agreement = torch.tensor(
        [item.affine_agreement for item in roots], dtype=torch.float64
    )
    output: list[NaturalPRegionV2] = []
    for ordinal, region in enumerate(regions):
        weights = region.aggregation_weights
        output.append(
            NaturalPRegionV2(
                region_ordinal=ordinal,
                radius=region.radius,
                cell_count=region.cell_count,
                region_sha256=superregion_sha256(region),
                contributing_root_ordinals=region.contributing_root_ordinals,
                p_score=float(weights @ p_score),
                proposal_support=float(weights @ support),
                action_margin=float(weights @ margin),
                eligible_mass=float(weights @ eligible),
                root_count=len(region.contributing_root_ordinals),
                eligible_root_count=sum(
                    int(roots[root].eligible)
                    for root in region.contributing_root_ordinals
                ),
                reference_iou=float(weights @ reference_iou),
                affine_agreement=float(weights @ affine_agreement),
            )
        )
    return tuple(output)


def _preparation_seal(
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    query_hash: str,
    reference_hash: str,
    root_hashes: Sequence[str],
    action_hashes: Sequence[str],
    region_bank_hash: str,
    roots: Sequence[NaturalPRootV2],
    regions: Sequence[NaturalPRegionV2],
    selected: int | None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "query_grid_shape": list(query_grid_shape),
        "reference_grid_shape": list(reference_grid_shape),
        "query_tokens_sha256": query_hash,
        "reference_tokens_sha256": reference_hash,
        "query_root_mask_sha256": list(root_hashes),
        "reference_action_mask_sha256": list(action_hashes),
        "region_bank_sha256": region_bank_hash,
        "roots": [
            {
                "root": item.root_ordinal,
                "p_score": float(item.p_score).hex(),
                "support": float(item.proposal_support).hex(),
                "margin": float(item.action_margin).hex(),
                "iou": float(item.reference_iou).hex(),
                "agreement": float(item.affine_agreement).hex(),
                "eligible": item.eligible,
                "directions": [
                    {
                        "direction": direction.direction,
                        "p": direction.p_query_indices.tolist(),
                        "v": direction.v_query_indices.tolist(),
                        "action": direction.selected_action,
                        "support": float(direction.proposal_support).hex(),
                        "margin": float(direction.action_margin).hex(),
                        "affine_sha256": tensor_sha256(direction.affine),
                        "eligible": direction.eligible,
                    }
                    for direction in item.directions
                ],
            }
            for item in roots
        ],
        "regions": [
            {
                "ordinal": item.region_ordinal,
                "sha256": item.region_sha256,
                "p_score": float(item.p_score).hex(),
                "support": float(item.proposal_support).hex(),
                "margin": float(item.action_margin).hex(),
                "eligible_mass": float(item.eligible_mass).hex(),
                "root_count": item.root_count,
                "eligible_root_count": item.eligible_root_count,
            }
            for item in regions
        ],
        "selected_region_ordinal": selected,
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare_natural_superregion_candidate_v2(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
) -> NaturalSuperregionPreparationV2:
    """Create and seal one anonymous candidate's P-only region proposal."""

    q_shape = _grid_shape(query_grid_shape, name="query grid")
    r_shape = _grid_shape(reference_grid_shape, name="reference grid")
    query = _unit_tokens(query_tokens, q_shape, name="query tokens")
    reference = _unit_tokens(reference_tokens, r_shape, name="reference tokens").to(
        query.device
    )
    q_hash = tensor_sha256(query)
    r_hash = tensor_sha256(reference)
    atoms = enumerate_query_macro_seeds(q_shape)
    root_masks = tuple(item.window.mask for item in atoms)
    actions = enumerate_native_reference_actions_v2(r_shape)
    regions = tuple(
        sorted(
            enumerate_superregion_bank(q_shape, include_r0_control=False),
            key=lambda item: (
                item.radius,
                item.cell_count,
                bytes.fromhex(item.mask_sha256),
                item.canonical_root_ordinal,
            ),
        )
    )
    if any(region.radius not in (1, 2, 3, 4) for region in regions):
        raise RuntimeError("natural target bank contains an r0 row")
    split = fixed_macro_bank_split(
        torch.ones(math.prod(q_shape), dtype=torch.bool), q_shape
    )
    similarity = query @ reference.T
    action_support = _action_support_table(similarity, actions)
    q_xy = grid_cell_centres(q_shape, dtype=torch.float64)
    r_xy = grid_cell_centres(r_shape, dtype=torch.float64)
    roots = tuple(
        _prepare_root(
            root_ordinal=ordinal,
            root_mask=mask,
            query_banks=(split.a, split.b),
            action_support=action_support,
            similarity=similarity,
            query_xy=q_xy,
            reference_xy=r_xy,
            actions=actions,
            reference_grid_shape=r_shape,
        )
        for ordinal, mask in enumerate(root_masks)
    )
    region_rows = _region_rows(regions, roots)
    # One isolated local root can initialize a proposal but can never authorize
    # a reportable target.  Selection requires evidence from two distinct
    # constituent 4x4-macro roots inside the connected super-region.
    eligible_regions = [
        item for item in region_rows if item.eligible_root_count >= 2
    ]
    selected = (
        min(
            eligible_regions,
            key=lambda item: (-item.p_score, item.region_ordinal),
        ).region_ordinal
        if eligible_regions
        else None
    )
    root_hashes = tuple(tensor_sha256(mask) for mask in root_masks)
    action_hashes = tuple(item.mask_sha256 for item in actions)
    bank_hash = superregion_bank_sha256(regions)
    seal = _preparation_seal(
        query_grid_shape=q_shape,
        reference_grid_shape=r_shape,
        query_hash=q_hash,
        reference_hash=r_hash,
        root_hashes=root_hashes,
        action_hashes=action_hashes,
        region_bank_hash=bank_hash,
        roots=roots,
        regions=region_rows,
        selected=selected,
    )
    return NaturalSuperregionPreparationV2(
        schema_version=SCHEMA_VERSION,
        query_grid_shape=q_shape,
        reference_grid_shape=r_shape,
        query_tokens_sha256=q_hash,
        reference_tokens_sha256=r_hash,
        query_root_mask_sha256=root_hashes,
        reference_action_mask_sha256=action_hashes,
        region_bank_sha256=bank_hash,
        roots=roots,
        regions=region_rows,
        selected_region_ordinal=selected,
        seal_sha256=seal,
        protected_access_counts=dict(PROTECTED_ZERO),
    )


def _validate_preparation(prepared: NaturalSuperregionPreparationV2) -> None:
    if not isinstance(prepared, NaturalSuperregionPreparationV2):
        raise TypeError("expected a natural V2 preparation")
    if (
        prepared.schema_version != SCHEMA_VERSION
        or prepared.protected_access_counts != PROTECTED_ZERO
    ):
        raise ValueError("natural V2 preparation schema drift")
    atoms = enumerate_query_macro_seeds(prepared.query_grid_shape)
    actions = enumerate_native_reference_actions_v2(prepared.reference_grid_shape)
    regions = tuple(
        sorted(
            enumerate_superregion_bank(
                prepared.query_grid_shape, include_r0_control=False
            ),
            key=lambda item: (
                item.radius,
                item.cell_count,
                bytes.fromhex(item.mask_sha256),
                item.canonical_root_ordinal,
            ),
        )
    )
    if (
        len(prepared.roots) != len(atoms)
        or len(prepared.regions) != len(regions)
        or prepared.query_root_mask_sha256
        != tuple(tensor_sha256(item.window.mask) for item in atoms)
        or prepared.reference_action_mask_sha256
        != tuple(item.mask_sha256 for item in actions)
        or prepared.region_bank_sha256 != superregion_bank_sha256(regions)
    ):
        raise ValueError("natural V2 structural receipt drift")
    expected = _preparation_seal(
        query_grid_shape=prepared.query_grid_shape,
        reference_grid_shape=prepared.reference_grid_shape,
        query_hash=prepared.query_tokens_sha256,
        reference_hash=prepared.reference_tokens_sha256,
        root_hashes=prepared.query_root_mask_sha256,
        action_hashes=prepared.reference_action_mask_sha256,
        region_bank_hash=prepared.region_bank_sha256,
        roots=prepared.roots,
        regions=prepared.regions,
        selected=prepared.selected_region_ordinal,
    )
    if prepared.seal_sha256 != expected:
        raise ValueError("natural V2 P-only seal drift")


def _verification_seal(
    *,
    preparation_seal: str,
    control_mode: str,
    reference_hash: str,
    selected: int | None,
    roots: Sequence[NaturalVRootV2],
    v_read_query_mask: torch.Tensor,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "preparation_seal_sha256": preparation_seal,
        "control_mode": control_mode,
        "verification_reference_sha256": reference_hash,
        "selected_region_ordinal": selected,
        "roots": [
            {
                "root": item.root_ordinal,
                "direction_scores": [float(value).hex() for value in item.direction_scores],
                "valid_v_patch_counts": list(item.valid_v_patch_counts),
                "total_v_patch_counts": list(item.total_v_patch_counts),
                "score": float(item.score).hex(),
                "valid_fraction": float(item.valid_fraction).hex(),
                "positive_mass": float(item.positive_mass).hex(),
                "eligible": item.eligible,
            }
            for item in roots
        ],
        "v_read_query_mask_sha256": tensor_sha256(v_read_query_mask),
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_natural_superregion_candidate_v2(
    prepared: NaturalSuperregionPreparationV2,
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    control_mode: str = CONTROL_BASE,
    reference_permutation: torch.Tensor | None = None,
) -> NaturalSuperregionVerificationV2:
    """Verify only the sealed P-selected region on complementary query cells."""

    _validate_preparation(prepared)
    if control_mode not in CONTROL_MODES:
        raise ValueError("unknown natural V2 verification control")
    query = _unit_tokens(
        query_tokens, prepared.query_grid_shape, name="verification query tokens"
    )
    source_reference = _unit_tokens(
        reference_tokens,
        prepared.reference_grid_shape,
        name="verification reference tokens",
    ).to(query.device)
    if tensor_sha256(query) != prepared.query_tokens_sha256:
        raise ValueError("V cannot change the query bound to P")
    source_hash = tensor_sha256(source_reference)
    if control_mode == CONTROL_BASE:
        if reference_permutation is not None or source_hash != prepared.reference_tokens_sha256:
            raise ValueError("BASE V must use the P-bound reference")
        reference = source_reference
    elif control_mode == CONTROL_V_C_BIND:
        if reference_permutation is not None or source_hash == prepared.reference_tokens_sha256:
            raise ValueError("V_C_BIND must use different reference content")
        reference = source_reference
    else:
        if source_hash != prepared.reference_tokens_sha256 or reference_permutation is None:
            raise ValueError("V_REF_SPATIAL requires P-bound content and a permutation")
        permutation = torch.as_tensor(
            reference_permutation, dtype=torch.long
        ).detach().cpu().contiguous()
        count = math.prod(prepared.reference_grid_shape)
        if (
            permutation.shape != (count,)
            or permutation.unique().numel() != count
            or int(permutation.min()) != 0
            or int(permutation.max()) != count - 1
            or bool(permutation.eq(torch.arange(count)).any())
        ):
            raise ValueError("V_REF_SPATIAL requires a complete no-fixed-point permutation")
        reference = source_reference[permutation.to(source_reference.device)].contiguous()
        if tensor_sha256(reference) == prepared.reference_tokens_sha256:
            raise ValueError("V_REF_SPATIAL permutation did not change reference content binding")
    verification_hash = tensor_sha256(reference)
    actions = enumerate_native_reference_actions_v2(prepared.reference_grid_shape)
    query_xy = grid_cell_centres(prepared.query_grid_shape, dtype=torch.float64)
    v_read_mask = torch.zeros(math.prod(prepared.query_grid_shape), dtype=torch.bool)
    rows: list[NaturalVRootV2] = []
    selected_row = (
        None
        if prepared.selected_region_ordinal is None
        else prepared.regions[prepared.selected_region_ordinal]
    )
    for root_ordinal in (
        () if selected_row is None else selected_row.contributing_root_ordinals
    ):
        p_root = prepared.roots[root_ordinal]
        direction_scores: list[float] = []
        valid_counts: list[int] = []
        total_counts: list[int] = []
        evidence_rows: list[torch.Tensor] = []
        for direction in p_root.directions:
            v_indices = direction.v_query_indices
            total_counts.append(int(v_indices.numel()))
            predicted = _apply_affine(direction.affine, query_xy[v_indices])
            inside = predicted.gt(0.0).all(dim=1) & predicted.lt(1.0).all(dim=1)
            r_height, r_width = prepared.reference_grid_shape
            nearest_x = torch.floor(predicted[:, 0] * r_width).to(torch.long).clamp(0, r_width - 1)
            nearest_y = torch.floor(predicted[:, 1] * r_height).to(torch.long).clamp(0, r_height - 1)
            nearest = nearest_y * r_width + nearest_x
            action = actions[direction.selected_action]
            inside &= action.mask[nearest]
            valid_indices = v_indices[inside]
            valid_xy = predicted[inside]
            valid_counts.append(int(valid_indices.numel()))
            # V uses the complete complementary bank as its fixed denominator.
            # Cells whose P transport leaves the image/action receive exact
            # zero evidence; they are never silently removed as an apparently
            # favourable subset.  ``valid_counts`` is diagnostic only.
            evidence = torch.zeros(
                v_indices.numel(), dtype=query.dtype, device=query.device
            )
            if valid_indices.numel() and direction.eligible:
                sampled = F.normalize(
                    _bilinear_sample(reference, prepared.reference_grid_shape, valid_xy),
                    p=2,
                    dim=1,
                    eps=1.0e-12,
                )
                evidence[inside.to(evidence.device)] = (
                    query[valid_indices.to(query.device)] * sampled
                ).sum(dim=1)
            score = (
                0.0
                if evidence.numel() == 0
                else float(evidence.sum().detach()) / float(evidence.numel())
            )
            direction_scores.append(score)
            evidence_rows.append(evidence)
            # Reading the entire held-out bank is part of the P/V contract,
            # even when a transported coordinate yields zero evidence.
            v_read_mask[v_indices] = True
        total = sum(total_counts)
        valid = sum(valid_counts)
        concatenated = torch.cat(evidence_rows) if any(row.numel() for row in evidence_rows) else torch.empty(0)
        eligible = all(
            direction.eligible and count >= MINIMUM_V_CELLS
            for direction, count in zip(p_root.directions, valid_counts, strict=True)
        )
        rows.append(
            NaturalVRootV2(
                root_ordinal=root_ordinal,
                direction_scores=(direction_scores[0], direction_scores[1]),
                valid_v_patch_counts=(valid_counts[0], valid_counts[1]),
                total_v_patch_counts=(total_counts[0], total_counts[1]),
                score=0.5 * (direction_scores[0] + direction_scores[1]),
                valid_fraction=0.0 if total == 0 else float(valid) / float(total),
                positive_mass=(
                    0.0
                    if concatenated.numel() == 0
                    else float(concatenated.clamp_min(0.0).sum().detach())
                ),
                eligible=eligible,
            )
        )
    if selected_row is None:
        v_score = valid_mass = positive_mass = 0.0
    else:
        by_root = {item.root_ordinal: item for item in rows}
        weights = enumerate_superregion_bank(
            prepared.query_grid_shape, include_r0_control=False
        )
        # Region ordering in the structural module differs from the sealed
        # score ordering; bind by region hash, never by incidental position.
        region_lookup = {superregion_sha256(item): item for item in weights}
        region = region_lookup[selected_row.region_sha256]
        v_score = sum(
            float(region.aggregation_weights[root]) * by_root[root].score
            for root in region.contributing_root_ordinals
        )
        valid_mass = sum(
            float(region.aggregation_weights[root]) * by_root[root].valid_fraction
            for root in region.contributing_root_ordinals
        )
        positive_mass = sum(
            float(region.aggregation_weights[root]) * by_root[root].positive_mass
            for root in region.contributing_root_ordinals
        )
    seal = _verification_seal(
        preparation_seal=prepared.seal_sha256,
        control_mode=control_mode,
        reference_hash=verification_hash,
        selected=prepared.selected_region_ordinal,
        roots=rows,
        v_read_query_mask=v_read_mask,
    )
    return NaturalSuperregionVerificationV2(
        schema_version=SCHEMA_VERSION,
        preparation_seal_sha256=prepared.seal_sha256,
        control_mode=control_mode,
        verification_reference_sha256=verification_hash,
        selected_region_ordinal=prepared.selected_region_ordinal,
        roots=tuple(rows),
        selected_v_score=v_score,
        selected_v_valid_mass=valid_mass,
        selected_v_positive_mass=positive_mass,
        v_read_query_mask=v_read_mask,
        seal_sha256=seal,
        protected_access_counts=dict(PROTECTED_ZERO),
    )


def materialize_natural_superregion_feature_ledger_v2(
    prepared: NaturalSuperregionPreparationV2,
    verified: NaturalSuperregionVerificationV2,
) -> NaturalSuperregionFeatureLedgerV2:
    """Return one finite feature row for every sealed r1--r4 region."""

    _validate_preparation(prepared)
    if (
        verified.schema_version != SCHEMA_VERSION
        or verified.preparation_seal_sha256 != prepared.seal_sha256
        or verified.selected_region_ordinal != prepared.selected_region_ordinal
        or verified.protected_access_counts != PROTECTED_ZERO
    ):
        raise ValueError("natural V2 verification/preparation binding drift")
    expected_verification_seal = _verification_seal(
        preparation_seal=verified.preparation_seal_sha256,
        control_mode=verified.control_mode,
        reference_hash=verified.verification_reference_sha256,
        selected=verified.selected_region_ordinal,
        roots=verified.roots,
        v_read_query_mask=verified.v_read_query_mask,
    )
    if verified.seal_sha256 != expected_verification_seal:
        raise ValueError("natural V2 verification seal drift")
    features = torch.zeros(
        (len(prepared.regions), len(REGION_FEATURE_COLUMNS)), dtype=torch.float64
    )
    keys: list[tuple[int, int, str, int]] = []
    for row in prepared.regions:
        keys.append((row.radius, row.cell_count, row.region_sha256, row.region_ordinal))
        features[row.region_ordinal, :8] = torch.tensor(
            (
                row.p_score,
                row.proposal_support,
                row.action_margin,
                row.eligible_mass,
                float(row.root_count),
                float(row.eligible_root_count),
                row.reference_iou,
                row.affine_agreement,
            ),
            dtype=torch.float64,
        )
    if prepared.selected_region_ordinal is not None:
        ordinal = prepared.selected_region_ordinal
        features[ordinal, 8:] = torch.tensor(
            (
                verified.selected_v_score,
                verified.selected_v_valid_mass,
                verified.selected_v_positive_mass,
                1.0,
            ),
            dtype=torch.float64,
        )
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("natural V2 feature ledger is not finite")
    logical = hashlib.sha256(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "preparation_seal_sha256": prepared.seal_sha256,
                "verification_seal_sha256": verified.seal_sha256,
                "region_keys": keys,
                "feature_columns": REGION_FEATURE_COLUMNS,
                "features_sha256": tensor_sha256(features),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return NaturalSuperregionFeatureLedgerV2(
        schema_version=SCHEMA_VERSION,
        preparation_seal_sha256=prepared.seal_sha256,
        verification_seal_sha256=verified.seal_sha256,
        region_keys=tuple(keys),
        feature_columns=REGION_FEATURE_COLUMNS,
        features=features,
        selected_region_ordinal=prepared.selected_region_ordinal,
        logical_sha256=logical,
    )


__all__ = [
    "SCHEMA_VERSION",
    "CONTROL_BASE",
    "CONTROL_V_C_BIND",
    "CONTROL_V_REF_SPATIAL",
    "PROTECTED_ZERO",
    "REGION_FEATURE_COLUMNS",
    "NaturalSuperregionV2Error",
    "NativeReferenceActionV2",
    "NaturalSuperregionPreparationV2",
    "NaturalSuperregionVerificationV2",
    "NaturalSuperregionFeatureLedgerV2",
    "enumerate_native_reference_actions_v2",
    "prepare_natural_superregion_candidate_v2",
    "verify_natural_superregion_candidate_v2",
    "materialize_natural_superregion_feature_ledger_v2",
    "tensor_sha256",
]
