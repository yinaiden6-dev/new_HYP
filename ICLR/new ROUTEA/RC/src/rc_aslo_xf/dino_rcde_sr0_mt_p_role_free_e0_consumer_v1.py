"""Source-index-free compact P scorer for the N16 E0 consumer.

The validated sidecar supplies action keys, eligibility and compact deployment
geometry.  Pair-cache tensors supply only the frozen 16-D features.  No fold is
accepted: source fold is irrelevant to P training and must not be fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from .cw1_sr0_structure_v1 import enumerate_superregion_bank, superregion_sha256
from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    CandidateActionFeatureTable,
    FEATURE_DIM,
    HARD_MAX_TIE_ULPS,
    SharedMultitilePHead,
    canonical_action_argmax,
    scale_balanced_hierarchical_logmeanexp,
)
from .dino_rcde_sr0_mt_p_compact_catalog_v1 import CompactMask
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    MIN_READY_ROOTS,
    FormalPContractError,
    _component_is_legal,
    canonical_sha256,
    tensor_sha256,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_role_free_e0_consumer_v1_20260818"


def _require(condition: object, message: str) -> None:
    if not condition:
        raise FormalPContractError(message)


def decode_bool_rle(value: Mapping[str, object]) -> torch.Tensor:
    shape_raw = value.get("shape")
    runs_raw = value.get("true_runs")
    _require(
        value.get("flatten_order") == "ROW_MAJOR"
        and isinstance(shape_raw, list)
        and len(shape_raw) == 2
        and isinstance(runs_raw, list),
        "sidecar eligibility RLE schema drift",
    )
    shape = tuple(int(item) for item in shape_raw)
    _require(shape[0] > 0 and shape[1] > 0, "sidecar eligibility shape drift")
    flat = torch.zeros(math.prod(shape), dtype=torch.bool)
    cursor = 0
    for raw in runs_raw:
        _require(isinstance(raw, list) and len(raw) == 2, "sidecar eligibility run drift")
        start, length = (int(raw[0]), int(raw[1]))
        _require(start >= cursor and length > 0 and start + length <= flat.numel(), "sidecar eligibility run overlap/range drift")
        flat[start : start + length] = True
        cursor = start + length
    result = flat.reshape(shape)
    _require(value.get("tensor_sha256") == tensor_sha256(result), "sidecar eligibility tensor hash drift")
    return result


@dataclass(frozen=True)
class RoleFreeCompactStructure:
    query_id: str
    query_source_image_sha256: str
    execution_ordinal: int
    member_ordinal: int
    candidate_key: str
    direction: str
    source_population_sha256: str
    action_keys: tuple[tuple[str, ...], ...]
    eligible: torch.Tensor
    colnomic_query_grid_shape: tuple[int, int]
    deployment_query_grid_shape: tuple[int, int]
    deployment_reference_grid_shape: tuple[int, int]
    deployment_query_root_masks: tuple[CompactMask, ...]
    deployment_reference_mask_table: tuple[CompactMask, ...]
    deployment_reference_mask_index: torch.Tensor
    query_geometry_sha256: str
    reference_geometry_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        keys = tuple(tuple(row) for row in self.action_keys)
        flags = torch.as_tensor(self.eligible, dtype=torch.bool).detach().cpu().contiguous()
        index = torch.as_tensor(self.deployment_reference_mask_index, dtype=torch.int64).detach().cpu().contiguous()
        query_masks = tuple(self.deployment_query_root_masks)
        reference_masks = tuple(self.deployment_reference_mask_table)
        _require(
            bool(self.query_id)
            and self.execution_ordinal >= 0
            and self.member_ordinal in (0, 1)
            and self.direction in ("a_to_b", "b_to_a")
            and bool(keys)
            and len(keys) == len(query_masks)
            and all(len(row) == len(keys[0]) for row in keys)
            and flags.shape == index.shape == (len(keys), len(keys[0]))
            and bool(reference_masks)
            and bool(index.ge(0).all())
            and bool(index.lt(len(reference_masks)).all()),
            "role-free compact structure population drift",
        )
        _require(
            all(mask.grid_shape == self.deployment_query_grid_shape for mask in query_masks)
            and all(mask.grid_shape == self.deployment_reference_grid_shape for mask in reference_masks),
            "role-free compact deployment grid drift",
        )
        object.__setattr__(self, "action_keys", keys)
        object.__setattr__(self, "eligible", flags)
        object.__setattr__(self, "deployment_reference_mask_index", index)
        object.__setattr__(self, "deployment_query_root_masks", query_masks)
        object.__setattr__(self, "deployment_reference_mask_table", reference_masks)

    @property
    def root_count(self) -> int:
        return len(self.action_keys)

    @property
    def action_count(self) -> int:
        return len(self.action_keys[0])


@dataclass(frozen=True)
class RoleFreeCompactDirectionalInput:
    structure: RoleFreeCompactStructure
    features: torch.Tensor

    def __post_init__(self) -> None:
        value = torch.as_tensor(self.features, dtype=torch.float64).detach().cpu().contiguous().clone()
        expected = (self.structure.root_count, self.structure.action_count, FEATURE_DIM)
        _require(value.shape == expected and bool(torch.isfinite(value).all()), "role-free compact feature shape/finite drift")
        _require(bool(value[~self.structure.eligible].eq(0.0).all()), "role-free compact H0 feature is nonzero")
        object.__setattr__(self, "features", value)


@dataclass(frozen=True)
class RoleFreeCompactDirectionOutput:
    compact_input: RoleFreeCompactDirectionalInput
    selected_action_indices: tuple[int | None, ...]
    selected_action_keys: tuple[str | None, ...]
    root_scores: torch.Tensor
    row_hashes: tuple[str, ...]
    row_radii: tuple[int, ...]
    row_scores: torch.Tensor
    row_ready: torch.Tensor
    candidate_utility: torch.Tensor
    map_row_index: int | None
    map_row_sha256: str | None


def structure_from_sidecar_record(value: Mapping[str, object]) -> RoleFreeCompactStructure:
    query_masks_raw = value.get("deployment_query_root_masks")
    reference_masks_raw = value.get("deployment_reference_mask_table")
    action_keys_raw = value.get("action_keys")
    _require(
        isinstance(query_masks_raw, list)
        and isinstance(reference_masks_raw, list)
        and isinstance(action_keys_raw, list)
        and value.get("record_sha256")
        == canonical_sha256({key: item for key, item in value.items() if key != "record_sha256"}),
        "sidecar structure record schema/hash drift",
    )
    action_keys = tuple(tuple(str(item) for item in row) for row in action_keys_raw)
    _require(value.get("action_keys_sha256") == canonical_sha256(action_keys_raw), "sidecar action-key hash drift")
    _require(
        value.get("deployment_query_root_masks_sha256") == canonical_sha256(query_masks_raw)
        and value.get("deployment_reference_mask_table_sha256") == canonical_sha256(reference_masks_raw)
        and value.get("deployment_reference_mask_index_sha256")
        == canonical_sha256(value.get("deployment_reference_mask_index")),
        "sidecar deployment structure hash drift",
    )
    reference_index = torch.as_tensor(
        value["deployment_reference_mask_index"], dtype=torch.int64
    )
    _require(
        value.get("deployment_reference_mask_index_tensor_sha256")
        == tensor_sha256(reference_index),
        "sidecar deployment index tensor hash drift",
    )
    return RoleFreeCompactStructure(
        query_id=str(value["query_id"]),
        query_source_image_sha256=str(value["query_source_image_sha256"]),
        execution_ordinal=int(value["execution_ordinal"]),
        member_ordinal=int(value["member_ordinal"]),
        candidate_key=str(value["candidate_key"]),
        direction=str(value["direction"]),
        source_population_sha256=str(value["source_population_sha256"]),
        action_keys=action_keys,
        eligible=decode_bool_rle(value["eligibility_rle"]),  # type: ignore[arg-type]
        colnomic_query_grid_shape=tuple(int(item) for item in value["colnomic_query_grid_shape"]),  # type: ignore[arg-type]
        deployment_query_grid_shape=tuple(int(item) for item in value["deployment_query_grid_shape"]),  # type: ignore[arg-type]
        deployment_reference_grid_shape=tuple(int(item) for item in value["deployment_reference_grid_shape"]),  # type: ignore[arg-type]
        deployment_query_root_masks=tuple(CompactMask.from_payload(item) for item in query_masks_raw),
        deployment_reference_mask_table=tuple(CompactMask.from_payload(item) for item in reference_masks_raw),
        deployment_reference_mask_index=reference_index,
        query_geometry_sha256=str(value["query_geometry_sha256"]),
        reference_geometry_sha256=str(value["reference_geometry_sha256"]),
        record_sha256=str(value["record_sha256"]),
    )


def _exact_zero(model: SharedMultitilePHead) -> torch.Tensor:
    return model.linear.weight.square().sum() * 0.0


def score_role_free_compact_direction(
    model: SharedMultitilePHead,
    value: RoleFreeCompactDirectionalInput,
) -> RoleFreeCompactDirectionOutput:
    _require(isinstance(model, SharedMultitilePHead), "role-free compact scorer model type drift")
    structure = value.structure
    regions = tuple(enumerate_superregion_bank(structure.colnomic_query_grid_shape, include_r0_control=False))
    _require(regions and len(regions[0].tile_weights) == structure.root_count, "role-free compact root/grid drift")
    table = CandidateActionFeatureTable(
        candidate_key=structure.candidate_key,
        root_ordinals=tuple(range(structure.root_count)),
        action_keys_by_root=structure.action_keys,
        features_by_root=tuple(value.features[root] for root in range(structure.root_count)),
        eligible_by_root=tuple(structure.eligible[root] for root in range(structure.root_count)),
    )
    selected_indices: list[int | None] = []
    selected_keys: list[str | None] = []
    root_scores: list[torch.Tensor] = []
    for keys, features, eligible in zip(
        table.action_keys_by_root,
        table.features_by_root,
        table.eligible_by_root,
        strict=True,
    ):
        logits = model(features)
        index = canonical_action_argmax(logits, keys, eligible.to(logits.device))
        selected_indices.append(index)
        selected_keys.append(None if index is None else keys[index])
        root_scores.append(_exact_zero(model) if index is None else logits[index])
    root_tensor = torch.stack(root_scores)
    query_masks = tuple(mask.decode() for mask in structure.deployment_query_root_masks)
    row_scores: list[torch.Tensor] = []
    row_ready: list[bool] = []
    row_hashes = tuple(superregion_sha256(region) for region in regions)
    row_radii = tuple(region.radius for region in regions)
    for region in regions:
        selected = [query_masks[root] for root in region.contributing_root_ordinals if selected_indices[root] is not None]
        legal = len(selected) >= MIN_READY_ROOTS
        if legal:
            union = torch.stack(selected).any(dim=0)
            legal = _component_is_legal(union, structure.deployment_query_grid_shape) and int(union.sum()) > max(int(mask.sum()) for mask in selected)
        if legal:
            weights = region.aggregation_weights.to(device=root_tensor.device, dtype=root_tensor.dtype)
            row_scores.append(torch.dot(weights, root_tensor))
        else:
            row_scores.append(_exact_zero(model))
        row_ready.append(bool(legal))
    scores = torch.stack(row_scores)
    ready = torch.tensor(row_ready, dtype=torch.bool, device=scores.device)
    utility = scale_balanced_hierarchical_logmeanexp(scores, row_radii)
    legal_rows = torch.nonzero(ready, as_tuple=False).flatten().tolist()
    map_index: int | None = None
    if legal_rows:
        detached = scores.detach()
        maximum = detached[torch.tensor(legal_rows, dtype=torch.long, device=detached.device)].max()
        tolerance = HARD_MAX_TIE_ULPS * torch.finfo(detached.dtype).eps * max(1.0, abs(float(maximum)))
        if float(maximum) > tolerance:
            tied = [index for index in legal_rows if abs(float(detached[index]) - float(maximum)) <= tolerance]
            map_index = min(tied, key=lambda index: row_hashes[index])
    return RoleFreeCompactDirectionOutput(
        compact_input=value,
        selected_action_indices=tuple(selected_indices),
        selected_action_keys=tuple(selected_keys),
        root_scores=root_tensor,
        row_hashes=row_hashes,
        row_radii=row_radii,
        row_scores=scores,
        row_ready=ready,
        candidate_utility=utility,
        map_row_index=map_index,
        map_row_sha256=None if map_index is None else row_hashes[map_index],
    )


def output_semantic_sha256(value: RoleFreeCompactDirectionOutput) -> str:
    return canonical_sha256(
        {
            "selected_action_indices": list(value.selected_action_indices),
            "selected_action_keys": list(value.selected_action_keys),
            "root_scores_sha256": tensor_sha256(value.root_scores),
            "row_hashes": list(value.row_hashes),
            "row_radii": list(value.row_radii),
            "row_scores_sha256": tensor_sha256(value.row_scores),
            "row_ready_sha256": tensor_sha256(value.row_ready),
            "candidate_utility_sha256": tensor_sha256(value.candidate_utility),
            "map_row_index": value.map_row_index,
            "map_row_sha256": value.map_row_sha256,
        }
    )


__all__ = [
    "SCHEMA_VERSION",
    "RoleFreeCompactStructure",
    "RoleFreeCompactDirectionalInput",
    "RoleFreeCompactDirectionOutput",
    "decode_bool_rle",
    "structure_from_sidecar_record",
    "score_role_free_compact_direction",
    "output_semantic_sha256",
]
