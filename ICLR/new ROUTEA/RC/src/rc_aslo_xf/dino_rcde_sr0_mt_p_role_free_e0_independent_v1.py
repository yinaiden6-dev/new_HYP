"""Independent source-fold-free deployable scorer for V83 replay."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from .cw1_sr0_structure_v1 import enumerate_superregion_bank, superregion_sha256
from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    CandidateActionFeatureTable, FEATURE_DIM, HARD_MAX_TIE_ULPS,
    SharedMultitilePHead, canonical_action_argmax,
    scale_balanced_hierarchical_logmeanexp,
)
from .dino_rcde_sr0_mt_p_compact_catalog_v1 import CompactMask
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    MIN_READY_ROOTS, FormalPContractError, _component_is_legal,
    canonical_sha256, tensor_sha256,
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise FormalPContractError(message)


def decode_eligibility(value: Mapping[str, object]) -> torch.Tensor:
    shape_raw = value.get("shape")
    runs_raw = value.get("true_runs")
    require(
        value.get("flatten_order") == "ROW_MAJOR"
        and isinstance(shape_raw, list)
        and len(shape_raw) == 2
        and isinstance(runs_raw, list),
        "independent eligibility schema",
    )
    shape = tuple(int(item) for item in shape_raw)
    require(shape[0] > 0 and shape[1] > 0, "independent eligibility shape")
    flat = torch.zeros(math.prod(shape), dtype=torch.bool)
    cursor = 0
    for run in runs_raw:
        require(isinstance(run, list) and len(run) == 2, "independent eligibility run schema")
        start, length = int(run[0]), int(run[1])
        require(start >= cursor and length > 0 and start + length <= flat.numel(), "independent eligibility RLE")
        flat[start:start + length] = True; cursor = start + length
    result = flat.reshape(shape)
    require(value.get("tensor_sha256") == tensor_sha256(result), "independent eligibility hash")
    return result


@dataclass(frozen=True)
class IndependentStructure:
    query_id: str
    query_source_image_sha256: str
    execution_ordinal: int
    member_ordinal: int
    candidate_key: str
    direction: str
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
        require(
            bool(self.query_id) and self.execution_ordinal >= 0
            and self.member_ordinal in (0, 1) and self.direction in ("a_to_b", "b_to_a")
            and bool(keys) and len(keys) == len(self.deployment_query_root_masks)
            and all(len(row) == len(keys[0]) for row in keys)
            and flags.shape == index.shape == (len(keys), len(keys[0]))
            and bool(self.deployment_reference_mask_table)
            and bool(index.ge(0).all()) and bool(index.lt(len(self.deployment_reference_mask_table)).all()),
            "independent structure population",
        )
        require(
            all(mask.grid_shape == self.deployment_query_grid_shape for mask in self.deployment_query_root_masks)
            and all(mask.grid_shape == self.deployment_reference_grid_shape for mask in self.deployment_reference_mask_table),
            "independent deployment grid",
        )
        object.__setattr__(self, "action_keys", keys)
        object.__setattr__(self, "eligible", flags)
        object.__setattr__(self, "deployment_reference_mask_index", index)

    @property
    def root_count(self) -> int: return len(self.action_keys)
    @property
    def action_count(self) -> int: return len(self.action_keys[0])


@dataclass(frozen=True)
class IndependentInput:
    structure: IndependentStructure
    features: torch.Tensor
    def __post_init__(self) -> None:
        value = torch.as_tensor(self.features, dtype=torch.float64).detach().cpu().contiguous().clone()
        require(value.shape == (self.structure.root_count, self.structure.action_count, FEATURE_DIM), "independent feature shape")
        require(bool(torch.isfinite(value).all()) and bool(value[~self.structure.eligible].eq(0.0).all()), "independent feature finite/H0")
        object.__setattr__(self, "features", value)


@dataclass(frozen=True)
class IndependentOutput:
    compact_input: IndependentInput
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


def structure_from_record(value: Mapping[str, object]) -> IndependentStructure:
    query_masks = value["deployment_query_root_masks"]  # type: ignore[index]
    reference_masks = value["deployment_reference_mask_table"]  # type: ignore[index]
    keys = value["action_keys"]  # type: ignore[index]
    require(value.get("record_sha256") == canonical_sha256({k:v for k,v in value.items() if k!="record_sha256"}), "independent record hash")
    require(value.get("action_keys_sha256") == canonical_sha256(keys), "independent action hash")
    require(value.get("deployment_query_root_masks_sha256") == canonical_sha256(query_masks), "independent query mask hash")
    require(value.get("deployment_reference_mask_table_sha256") == canonical_sha256(reference_masks), "independent reference mask hash")
    reference_index = torch.as_tensor(value["deployment_reference_mask_index"], dtype=torch.int64)  # type: ignore[index]
    require(
        value.get("deployment_reference_mask_index_sha256")
        == canonical_sha256(value.get("deployment_reference_mask_index")),
        "independent index payload hash",
    )
    require(value.get("deployment_reference_mask_index_tensor_sha256") == tensor_sha256(reference_index), "independent index hash")
    return IndependentStructure(
        query_id=str(value["query_id"]), query_source_image_sha256=str(value["query_source_image_sha256"]),
        execution_ordinal=int(value["execution_ordinal"]), member_ordinal=int(value["member_ordinal"]),
        candidate_key=str(value["candidate_key"]), direction=str(value["direction"]),
        action_keys=tuple(tuple(str(item) for item in row) for row in keys),  # type: ignore[arg-type]
        eligible=decode_eligibility(value["eligibility_rle"]),  # type: ignore[arg-type]
        colnomic_query_grid_shape=tuple(int(item) for item in value["colnomic_query_grid_shape"]),  # type: ignore[arg-type]
        deployment_query_grid_shape=tuple(int(item) for item in value["deployment_query_grid_shape"]),  # type: ignore[arg-type]
        deployment_reference_grid_shape=tuple(int(item) for item in value["deployment_reference_grid_shape"]),  # type: ignore[arg-type]
        deployment_query_root_masks=tuple(CompactMask.from_payload(item) for item in query_masks),  # type: ignore[arg-type]
        deployment_reference_mask_table=tuple(CompactMask.from_payload(item) for item in reference_masks),  # type: ignore[arg-type]
        deployment_reference_mask_index=reference_index,
        query_geometry_sha256=str(value["query_geometry_sha256"]), reference_geometry_sha256=str(value["reference_geometry_sha256"]),
        record_sha256=str(value["record_sha256"]),
    )


def zero(model: SharedMultitilePHead) -> torch.Tensor: return model.linear.weight.square().sum()*0.0


def score(model: SharedMultitilePHead, value: IndependentInput) -> IndependentOutput:
    require(isinstance(model, SharedMultitilePHead), "independent scorer model type")
    structure=value.structure
    regions=tuple(enumerate_superregion_bank(structure.colnomic_query_grid_shape,include_r0_control=False))
    require(regions and len(regions[0].tile_weights)==structure.root_count,"independent root/grid")
    table=CandidateActionFeatureTable(
        candidate_key=structure.candidate_key,root_ordinals=tuple(range(structure.root_count)),
        action_keys_by_root=structure.action_keys,
        features_by_root=tuple(value.features[root] for root in range(structure.root_count)),
        eligible_by_root=tuple(structure.eligible[root] for root in range(structure.root_count)),
    )
    indices=[];keys=[];root_scores=[]
    for action_keys,features,eligible in zip(table.action_keys_by_root,table.features_by_root,table.eligible_by_root,strict=True):
        logits=model(features);index=canonical_action_argmax(logits,action_keys,eligible.to(logits.device));indices.append(index);keys.append(None if index is None else action_keys[index]);root_scores.append(zero(model) if index is None else logits[index])
    roots=torch.stack(root_scores);query_masks=tuple(mask.decode() for mask in structure.deployment_query_root_masks)
    row_scores=[];row_ready=[];row_hashes=tuple(superregion_sha256(region) for region in regions);row_radii=tuple(region.radius for region in regions)
    for region in regions:
        selected=[query_masks[root] for root in region.contributing_root_ordinals if indices[root] is not None]
        legal=len(selected)>=MIN_READY_ROOTS
        if legal:
            union=torch.stack(selected).any(dim=0);legal=_component_is_legal(union,structure.deployment_query_grid_shape) and int(union.sum())>max(int(mask.sum()) for mask in selected)
        row_scores.append(torch.dot(region.aggregation_weights.to(device=roots.device,dtype=roots.dtype),roots) if legal else zero(model));row_ready.append(bool(legal))
    scores=torch.stack(row_scores);ready=torch.tensor(row_ready,dtype=torch.bool,device=scores.device);utility=scale_balanced_hierarchical_logmeanexp(scores,row_radii)
    legal_rows=torch.nonzero(ready,as_tuple=False).flatten().tolist();map_index=None
    if legal_rows:
        detached=scores.detach();maximum=detached[torch.tensor(legal_rows,dtype=torch.long,device=detached.device)].max();tolerance=HARD_MAX_TIE_ULPS*torch.finfo(detached.dtype).eps*max(1.0,abs(float(maximum)))
        if float(maximum)>tolerance:
            tied=[index for index in legal_rows if abs(float(detached[index])-float(maximum))<=tolerance];map_index=min(tied,key=lambda index:row_hashes[index])
    return IndependentOutput(value,tuple(indices),tuple(keys),roots,row_hashes,row_radii,scores,ready,utility,map_index,None if map_index is None else row_hashes[map_index])


def output_sha(value: IndependentOutput) -> str:
    return canonical_sha256({"selected_action_indices":list(value.selected_action_indices),"selected_action_keys":list(value.selected_action_keys),"root_scores_sha256":tensor_sha256(value.root_scores),"row_hashes":list(value.row_hashes),"row_radii":list(value.row_radii),"row_scores_sha256":tensor_sha256(value.row_scores),"row_ready_sha256":tensor_sha256(value.row_ready),"candidate_utility_sha256":tensor_sha256(value.candidate_utility),"map_row_index":value.map_row_index,"map_row_sha256":value.map_row_sha256})


__all__=["IndependentStructure","IndependentInput","IndependentOutput","structure_from_record","score","output_sha"]
