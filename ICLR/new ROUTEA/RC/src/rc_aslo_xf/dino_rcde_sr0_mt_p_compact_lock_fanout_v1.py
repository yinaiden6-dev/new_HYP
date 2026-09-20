"""Single-pass four-head P-lock fanout for the compact SR0-MT repair.

This additive module does not replace or mutate the I1-sealed P scorer.  It
only changes loop nesting at lock time: one target-free, complete candidate
feature stream for a query is consumed once, and every decoded directional
record is immediately scored by the four checkpoints assigned to that
query's frozen cross-fit role.

The four assignments are fixed by the query source fold ``s``:

* ``P_OUTER{s}_OUTER_REFIT`` for the query's outer-heldout deployment lock;
* ``P_OUTER{o}_INNER{s}_FIT`` for every ``o != s``, producing the three
  complementary inner-OOF locks.

No target, rival, rank, D1 value, label, outcome, candidate pruning, or
fallback is accepted.  The legacy ``score_deployable_direction`` remains the
executable scoring oracle, including its complete r1--r4 fixed denominator,
post-mapping H0 closure, strict-positive MAP rule, and no-second-best policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from typing import Iterable, Mapping, Sequence

import torch

from .cw1_sr0_structure_v1 import (
    CW1SR0SuperRegion,
    enumerate_superregion_bank,
    superregion_bank_sha256,
    superregion_sha256,
)
from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    HARD_MAX_TIE_ULPS,
    P_LOCK_ERASED_FIELDS,
    P_LOCK_REQUIRED_FIELDS,
    SharedMultitilePHead,
    canonical_action_argmax,
    scale_balanced_hierarchical_logmeanexp,
)
from .dino_rcde_sr0_mt_p_compact_catalog_v1 import (
    NATURAL_CANDIDATE_COUNT,
    CandidateFeatureTensor,
    FactorizedDirectionalSourceIndex,
    candidate_action_table_from_compact_tensor,
    validate_candidate_feature_tensor,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    LOCK_H0,
    LOCK_READY,
    MIN_READY_ROOTS,
    P_DIRECTIONS,
    ROOT_H0,
    ROOT_READY,
    DeployableDirectionOutput,
    DirectionalFeatureRecord,
    FormalPContractError,
    RootActionDeployment,
    _component_is_legal,
    canonical_sha256,
    encode_mask_rle,
    make_root_action_deployment,
    require_sha256,
    score_deployable_direction,
    tensor_sha256,
    validate_p_lock_record,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_compact_lock_fanout_v1_20260815"
INNER_FIT_ROLE = "INNER_TRAIN_FIT"
OUTER_REFIT_ROLE = "OUTER_TRAIN_REFIT"
INNER_LOCK_ROLE = "INNER_HELDOUT_OOF"
OUTER_LOCK_ROLE = "OUTER_HELDOUT_DEPLOYMENT"

_DIRECTION_ORDINAL = {name: ordinal for ordinal, name in enumerate(P_DIRECTIONS)}


@dataclass(frozen=True)
class CrossfitFanoutScope:
    """One of the four frozen checkpoints allowed to score one source fold."""

    fit_id: str
    fit_role: str
    crossfit_role: str
    source_fold: int
    outer_fold: int
    inner_heldout_fold: int | None

    def __post_init__(self) -> None:
        if self.source_fold not in (1, 2, 3, 4) or self.outer_fold not in (
            1,
            2,
            3,
            4,
        ):
            raise FormalPContractError("fanout source/outer fold must be 1..4")
        if self.fit_role == OUTER_REFIT_ROLE:
            expected = (
                self.outer_fold == self.source_fold
                and self.inner_heldout_fold is None
                and self.crossfit_role == OUTER_LOCK_ROLE
                and self.fit_id == f"P_OUTER{self.source_fold}_OUTER_REFIT"
            )
        elif self.fit_role == INNER_FIT_ROLE:
            expected = (
                self.outer_fold != self.source_fold
                and self.inner_heldout_fold == self.source_fold
                and self.crossfit_role == INNER_LOCK_ROLE
                and self.fit_id
                == f"P_OUTER{self.outer_fold}_INNER{self.source_fold}_FIT"
            )
        else:
            expected = False
        if not expected:
            raise FormalPContractError("P lock fanout cross-fit scope drift")


def crossfit_fanout_scopes(source_fold: int) -> tuple[CrossfitFanoutScope, ...]:
    """Return own outer refit followed by the three complementary inner heads."""

    if source_fold not in (1, 2, 3, 4):
        raise FormalPContractError("fanout source fold must be 1..4")
    own = CrossfitFanoutScope(
        fit_id=f"P_OUTER{source_fold}_OUTER_REFIT",
        fit_role=OUTER_REFIT_ROLE,
        crossfit_role=OUTER_LOCK_ROLE,
        source_fold=source_fold,
        outer_fold=source_fold,
        inner_heldout_fold=None,
    )
    complementary = tuple(
        CrossfitFanoutScope(
            fit_id=f"P_OUTER{outer}_INNER{source_fold}_FIT",
            fit_role=INNER_FIT_ROLE,
            crossfit_role=INNER_LOCK_ROLE,
            source_fold=source_fold,
            outer_fold=outer,
            inner_heldout_fold=source_fold,
        )
        for outer in (1, 2, 3, 4)
        if outer != source_fold
    )
    return (own, *complementary)


@dataclass(frozen=True)
class ScopedPHead:
    """A frozen shared P head bound to exactly one cross-fit scope."""

    scope: CrossfitFanoutScope
    model: SharedMultitilePHead
    p_checkpoint_sha256: str | None = None
    p_training_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CrossfitFanoutScope) or not isinstance(
            self.model, SharedMultitilePHead
        ):
            raise FormalPContractError("fanout needs a scoped SharedMultitilePHead")
        hashes = (self.p_checkpoint_sha256, self.p_training_manifest_sha256)
        if (hashes[0] is None) != (hashes[1] is None):
            raise FormalPContractError(
                "fanout checkpoint/training-manifest hashes must be paired"
            )
        if hashes[0] is not None:
            require_sha256(hashes[0], name="fanout P checkpoint")
            require_sha256(hashes[1], name="fanout P training manifest")


def _cpu_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().contiguous().clone()
    if not bool(torch.isfinite(tensor).all()):
        raise FormalPContractError(f"fanout {name} is non-finite")
    return tensor


@dataclass(frozen=True)
class CompactDirectionalFeatureInput:
    """One role-free compact source plus its complete Cartesian feature tensor."""

    source_fold: int
    source_index: FactorizedDirectionalSourceIndex
    candidate_features: CandidateFeatureTensor

    def __post_init__(self) -> None:
        if self.source_fold not in (1, 2, 3, 4):
            raise FormalPContractError("compact fanout source fold must be 1..4")
        if not isinstance(self.source_index, FactorizedDirectionalSourceIndex):
            raise FormalPContractError("compact fanout source-index type drift")
        if self.source_index.candidate_count_per_query != NATURAL_CANDIDATE_COUNT:
            raise FormalPContractError("compact fanout source is not natural C128")
        validate_candidate_feature_tensor(
            self.source_index, self.candidate_features
        )


@dataclass(frozen=True)
class _PreparedCompactStructure:
    """Canonical query-grid metadata shared by every source and P head."""

    query_grid_shape: tuple[int, int]
    root_count: int
    regions: tuple[CW1SR0SuperRegion, ...]
    row_hashes: tuple[str, ...]
    row_radii: tuple[int, ...]
    row_root_ordinals: tuple[tuple[int, ...], ...]
    row_aggregation_weights: tuple[torch.Tensor, ...]
    complete_bank_sha256: str
    row_count_by_radius: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        rows = len(self.regions)
        if (
            self.root_count <= 0
            or rows <= 0
            or len(self.row_hashes) != rows
            or len(self.row_radii) != rows
            or len(self.row_root_ordinals) != rows
            or len(self.row_aggregation_weights) != rows
            or set(self.row_radii) != {1, 2, 3, 4}
            or sum(self.row_count_by_radius) != rows
        ):
            raise FormalPContractError("prepared compact structure population drift")
        if any(
            weights.shape != (self.root_count,)
            or weights.dtype != torch.float64
            or not bool(torch.isfinite(weights).all())
            for weights in self.row_aggregation_weights
        ):
            raise FormalPContractError("prepared compact structure weight drift")


@lru_cache(maxsize=64)
def _prepare_compact_structure(
    query_grid_shape: tuple[int, int],
) -> _PreparedCompactStructure:
    """Materialize the immutable CW1 bank and its hashes once per query grid."""

    regions = tuple(
        enumerate_superregion_bank(
            tuple(query_grid_shape), include_r0_control=False
        )
    )
    if not regions:
        raise FormalPContractError("prepared compact structure bank is empty")
    root_count = int(regions[0].tile_weights.numel())
    row_hashes = tuple(superregion_sha256(item) for item in regions)
    row_radii = tuple(item.radius for item in regions)
    return _PreparedCompactStructure(
        query_grid_shape=tuple(query_grid_shape),
        root_count=root_count,
        regions=regions,
        row_hashes=row_hashes,
        row_radii=row_radii,
        row_root_ordinals=tuple(
            item.contributing_root_ordinals for item in regions
        ),
        row_aggregation_weights=tuple(
            item.aggregation_weights for item in regions
        ),
        complete_bank_sha256=superregion_bank_sha256(regions),
        row_count_by_radius=tuple(
            sum(item.radius == radius for item in regions)
            for radius in (1, 2, 3, 4)
        ),
    )


@dataclass(frozen=True)
class _PreparedCompactRowGeometry:
    """Head-invariant deployment-query geometry for one factorized source."""

    geometry_key: tuple[object, ...]
    deployment_query_grid_shape: tuple[int, int]
    root_ready: tuple[bool, ...]
    root_query_masks: tuple[torch.Tensor, ...]
    row_ready: tuple[bool, ...]
    row_query_unions: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        roots = len(self.root_ready)
        rows = len(self.row_ready)
        token_count = math.prod(self.deployment_query_grid_shape)
        if (
            roots <= 0
            or rows <= 0
            or len(self.root_query_masks) != roots
            or len(self.row_query_unions) != rows
            or any(
                mask.shape != (token_count,) or mask.dtype != torch.bool
                for mask in (*self.root_query_masks, *self.row_query_unions)
            )
        ):
            raise FormalPContractError("prepared compact row geometry drift")


def _compact_row_geometry_key(
    source: FactorizedDirectionalSourceIndex,
    structure: _PreparedCompactStructure,
) -> tuple[object, ...]:
    root_ready = tuple(bool(item) for item in source.eligible.any(dim=1).tolist())
    return (
        structure.complete_bank_sha256,
        structure.query_grid_shape,
        source.query_geometry_sha256,
        source.colnomic_query_root_masks[0].grid_shape,
        source.deployment_query_root_masks[0].grid_shape,
        tuple(item.logical_sha256 for item in source.deployment_query_root_masks),
        root_ready,
    )


def _prepare_compact_row_geometry(
    source: FactorizedDirectionalSourceIndex,
    structure: _PreparedCompactStructure,
) -> _PreparedCompactRowGeometry:
    """Evaluate every row's deployment legality once, before head fanout."""

    if source.root_count != structure.root_count:
        raise FormalPContractError("prepared compact source root/grid drift")
    deployment_grid = source.deployment_query_root_masks[0].grid_shape
    root_ready = tuple(bool(item) for item in source.eligible.any(dim=1).tolist())
    root_masks = tuple(item.decode() for item in source.deployment_query_root_masks)
    zero = torch.zeros(math.prod(deployment_grid), dtype=torch.bool)
    row_ready: list[bool] = []
    row_unions: list[torch.Tensor] = []
    for roots in structure.row_root_ordinals:
        ready_masks = [root_masks[root] for root in roots if root_ready[root]]
        union = (
            torch.stack(ready_masks).any(dim=0)
            if ready_masks
            else zero.clone()
        )
        legal = len(ready_masks) >= MIN_READY_ROOTS
        if legal:
            legal = _component_is_legal(union, deployment_grid) and int(
                union.sum()
            ) > max(int(item.sum()) for item in ready_masks)
        row_ready.append(bool(legal))
        row_unions.append(union)
    return _PreparedCompactRowGeometry(
        geometry_key=_compact_row_geometry_key(source, structure),
        deployment_query_grid_shape=deployment_grid,
        root_ready=root_ready,
        root_query_masks=root_masks,
        row_ready=tuple(row_ready),
        row_query_unions=tuple(row_unions),
    )


@dataclass(frozen=True)
class CompactDeployableDirectionOutput:
    """Old-scorer-equivalent output with only selected deployments decoded."""

    compact_input: CompactDirectionalFeatureInput
    prepared_structure: _PreparedCompactStructure
    prepared_row_geometry: _PreparedCompactRowGeometry
    selected_action_indices: tuple[int | None, ...]
    selected_action_keys: tuple[str | None, ...]
    selected_deployments: tuple[RootActionDeployment | None, ...]
    root_scores: torch.Tensor
    row_scores: torch.Tensor
    row_ready: torch.Tensor
    candidate_utility: torch.Tensor
    map_row_index: int | None
    map_row_sha256: str | None
    selected_deployment_reference_count: int
    selected_deployment_decode_count: int

    def __post_init__(self) -> None:
        source = self.compact_input.source_index
        roots = source.root_count
        structure = self.prepared_structure
        geometry = self.prepared_row_geometry
        regions = structure.regions
        if (
            structure.query_grid_shape
            != source.colnomic_query_root_masks[0].grid_shape
            or structure.root_count != roots
            or geometry.deployment_query_grid_shape
            != source.deployment_query_root_masks[0].grid_shape
            or geometry.geometry_key != _compact_row_geometry_key(
                source, structure
            )
            or len(geometry.root_ready) != roots
            or len(geometry.row_ready) != len(regions)
            or len(self.selected_action_indices) != roots
            or len(self.selected_action_keys) != roots
            or len(self.selected_deployments) != roots
            or self.root_scores.shape != (roots,)
            or self.row_scores.shape != (len(regions),)
            or self.row_ready.shape != (len(regions),)
            or self.row_ready.dtype != torch.bool
            or self.candidate_utility.ndim != 0
            or self.selected_deployment_reference_count
            != sum(item is not None for item in self.selected_deployments)
            or not 0
            <= self.selected_deployment_decode_count
            <= self.selected_deployment_reference_count
            <= roots
        ):
            raise FormalPContractError("compact deployable output population drift")
        if not torch.equal(
            self.row_ready.detach().cpu(),
            torch.tensor(geometry.row_ready, dtype=torch.bool),
        ):
            raise FormalPContractError("compact deployable row geometry drift")
        for root, deployment in enumerate(self.selected_deployments):
            action = self.selected_action_indices[root]
            key = self.selected_action_keys[root]
            if (action is not None) != geometry.root_ready[root]:
                raise FormalPContractError("compact prepared root READY/H0 drift")
            if deployment is None:
                if action is not None or key is not None or bool(
                    self.root_scores[root].ne(0.0)
                ):
                    raise FormalPContractError(
                        "compact selected root H0/action drift"
                    )
            elif (
                action is None
                or key != deployment.action_key
                or not deployment.eligible
                or deployment.binding_status != ROOT_READY
            ):
                raise FormalPContractError(
                    "compact selected deployment/action binding drift"
                )
        if bool(self.row_scores[~self.row_ready].ne(0.0).any()):
            raise FormalPContractError("compact row H0 is not exact zero")
        if self.map_row_index is None:
            if self.map_row_sha256 is not None:
                raise FormalPContractError("compact MAP H0 hash drift")
        elif (
            self.map_row_index not in range(len(regions))
            or self.map_row_sha256
            != structure.row_hashes[self.map_row_index]
            or not bool(self.row_ready[self.map_row_index])
        ):
            raise FormalPContractError("compact MAP READY binding drift")


def _model_exact_zero(model: SharedMultitilePHead) -> torch.Tensor:
    return model.linear.weight.square().sum() * 0.0


def _score_compact_roots(
    model: SharedMultitilePHead,
    action_table,
    geometry: _PreparedCompactRowGeometry,
) -> tuple[tuple[int | None, ...], tuple[str | None, ...], torch.Tensor]:
    """Run only the head-dependent linear/action selection part of shared P."""

    if len(action_table.root_ordinals) != len(geometry.root_ready):
        raise FormalPContractError("compact action table root/grid drift")
    selected_indices: list[int | None] = []
    selected_keys: list[str | None] = []
    root_scores: list[torch.Tensor] = []
    for root, (keys, features, eligible) in enumerate(
        zip(
            action_table.action_keys_by_root,
            action_table.features_by_root,
            action_table.eligible_by_root,
            strict=True,
        )
    ):
        logits = model(features)
        index = canonical_action_argmax(
            logits, keys, eligible.to(logits.device)
        )
        if (index is not None) != geometry.root_ready[root]:
            raise FormalPContractError("compact prepared root selection drift")
        selected_indices.append(index)
        if index is None:
            selected_keys.append(None)
            root_scores.append(_model_exact_zero(model))
        else:
            selected_keys.append(keys[index])
            root_scores.append(logits[index])
    return (
        tuple(selected_indices),
        tuple(selected_keys),
        torch.stack(root_scores),
    )


def _score_compact_with_table(
    model: SharedMultitilePHead,
    value: CompactDirectionalFeatureInput,
    action_table,
    deployment_cache: dict[tuple[int, int], RootActionDeployment] | None = None,
    prepared_structure: _PreparedCompactStructure | None = None,
    prepared_row_geometry: _PreparedCompactRowGeometry | None = None,
) -> CompactDeployableDirectionOutput:
    """Exact legacy replay using prepared, head-invariant row metadata."""

    if not isinstance(model, SharedMultitilePHead) or not isinstance(
        value, CompactDirectionalFeatureInput
    ):
        raise FormalPContractError("compact P scorer received wrong object type")
    source = value.source_index
    query_grid = source.colnomic_query_root_masks[0].grid_shape
    structure = (
        _prepare_compact_structure(query_grid)
        if prepared_structure is None
        else prepared_structure
    )
    geometry = (
        _prepare_compact_row_geometry(source, structure)
        if prepared_row_geometry is None
        else prepared_row_geometry
    )
    if (
        structure.query_grid_shape != query_grid
        or structure.root_count != source.root_count
        or geometry.geometry_key != _compact_row_geometry_key(
            source, structure
        )
        or len(geometry.root_ready) != source.root_count
        or action_table.candidate_key != source.candidate_key
    ):
        raise FormalPContractError("compact prepared scorer binding drift")
    selected_indices, selected_keys, root_score_tensor = _score_compact_roots(
        model, action_table, geometry
    )
    selected_deployments: list[RootActionDeployment | None] = []
    decode_count = 0
    for root, index in enumerate(selected_indices):
        if index is None:
            selected_deployments.append(None)
            continue
        cache_key = (root, index)
        deployment = (
            None
            if deployment_cache is None
            else deployment_cache.get(cache_key)
        )
        if deployment is None:
            deployment = source.decode_deployment(root, index)
            decode_count += 1
            if deployment_cache is not None:
                deployment_cache[cache_key] = deployment
        if (
            not deployment.eligible
            or deployment.binding_status != ROOT_READY
            or selected_keys[root] != deployment.action_key
        ):
            raise FormalPContractError(
                "compact selected P root action/deployment binding drift"
            )
        selected_deployments.append(deployment)

    row_scores: list[torch.Tensor] = []
    for legal, weights in zip(
        geometry.row_ready,
        structure.row_aggregation_weights,
        strict=True,
    ):
        if legal:
            device_weights = weights.to(
                device=root_score_tensor.device, dtype=root_score_tensor.dtype
            )
            row_scores.append(torch.dot(device_weights, root_score_tensor))
        else:
            row_scores.append(_model_exact_zero(model))
    scores = torch.stack(row_scores)
    ready_tensor = torch.tensor(
        geometry.row_ready, dtype=torch.bool, device=scores.device
    )
    utility = scale_balanced_hierarchical_logmeanexp(
        scores, structure.row_radii
    )
    legal_rows = torch.nonzero(ready_tensor, as_tuple=False).flatten().tolist()
    map_index: int | None = None
    if legal_rows:
        detached = scores.detach()
        maximum = detached[
            torch.tensor(legal_rows, dtype=torch.long, device=detached.device)
        ].max()
        tolerance = (
            HARD_MAX_TIE_ULPS
            * torch.finfo(detached.dtype).eps
            * max(1.0, abs(float(maximum)))
        )
        if float(maximum) > tolerance:
            tied = [
                index
                for index in legal_rows
                if abs(float(detached[index]) - float(maximum)) <= tolerance
            ]
            map_index = min(
                tied, key=lambda index: structure.row_hashes[index]
            )
    return CompactDeployableDirectionOutput(
        compact_input=value,
        prepared_structure=structure,
        prepared_row_geometry=geometry,
        selected_action_indices=selected_indices,
        selected_action_keys=selected_keys,
        selected_deployments=tuple(selected_deployments),
        root_scores=root_score_tensor,
        row_scores=scores,
        row_ready=ready_tensor,
        candidate_utility=utility,
        map_row_index=map_index,
        map_row_sha256=(
            None if map_index is None else structure.row_hashes[map_index]
        ),
        selected_deployment_reference_count=sum(
            item is not None for item in selected_deployments
        ),
        selected_deployment_decode_count=decode_count,
    )


def score_compact_deployable_direction(
    model: SharedMultitilePHead,
    value: CompactDirectionalFeatureInput,
) -> CompactDeployableDirectionOutput:
    """Score compact features and decode at most one deployment per root."""

    if not isinstance(value, CompactDirectionalFeatureInput):
        raise FormalPContractError("compact scorer input type drift")
    action_table = candidate_action_table_from_compact_tensor(
        value.source_index, value.candidate_features
    )
    return _score_compact_with_table(model, value, action_table)


def decision_from_compact_output(
    output: CompactDeployableDirectionOutput,
) -> FanoutDirectionDecision:
    """Produce the same compact decision schema as the legacy scorer path."""

    if not isinstance(output, CompactDeployableDirectionOutput):
        raise FormalPContractError("compact decision requires compact output")
    source = output.compact_input.source_index
    structure = output.prepared_structure
    return FanoutDirectionDecision(
        candidate_position=source.candidate_position,
        candidate_key=source.candidate_key,
        candidate_physical_row=source.candidate_physical_row,
        candidate_reference_source_sha256=source.candidate_reference_source_sha256,
        direction=source.direction,
        selected_action_indices=output.selected_action_indices,
        selected_action_keys=output.selected_action_keys,
        root_statuses=tuple(
            ROOT_READY if item is not None else ROOT_H0
            for item in output.selected_action_indices
        ),
        root_scores=_cpu_tensor(output.root_scores, name="compact root scores"),
        row_hashes=structure.row_hashes,
        row_radii=structure.row_radii,
        row_scores=_cpu_tensor(output.row_scores, name="compact row scores"),
        row_ready=torch.as_tensor(output.row_ready, dtype=torch.bool)
        .detach()
        .cpu()
        .contiguous()
        .clone(),
        candidate_utility=_cpu_tensor(
            output.candidate_utility, name="compact candidate utility"
        ),
        selected_bank_ordinal=output.map_row_index,
        selected_row_sha256=output.map_row_sha256,
        lock_status=LOCK_H0 if output.map_row_index is None else LOCK_READY,
    )


@dataclass(frozen=True)
class FanoutDirectionDecision:
    """Compact lock-ready result for one head/candidate/direction coordinate."""

    candidate_position: int
    candidate_key: str
    candidate_physical_row: int
    candidate_reference_source_sha256: str
    direction: str
    selected_action_indices: tuple[int | None, ...]
    selected_action_keys: tuple[str | None, ...]
    root_statuses: tuple[str, ...]
    root_scores: torch.Tensor
    row_hashes: tuple[str, ...]
    row_radii: tuple[int, ...]
    row_scores: torch.Tensor
    row_ready: torch.Tensor
    candidate_utility: torch.Tensor
    selected_bank_ordinal: int | None
    selected_row_sha256: str | None
    lock_status: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_position, bool)
            or not isinstance(self.candidate_position, int)
            or self.candidate_position < 0
            or not self.candidate_key
            or isinstance(self.candidate_physical_row, bool)
            or not isinstance(self.candidate_physical_row, int)
            or self.candidate_physical_row < 0
            or len(self.candidate_reference_source_sha256) != 64
            or self.direction not in P_DIRECTIONS
        ):
            raise FormalPContractError("fanout candidate/direction addressing drift")
        roots = len(self.selected_action_indices)
        rows = len(self.row_hashes)
        if (
            len(self.selected_action_keys) != roots
            or len(self.root_statuses) != roots
            or self.root_scores.shape != (roots,)
            or len(self.row_radii) != rows
            or self.row_scores.shape != (rows,)
            or self.row_ready.shape != (rows,)
            or self.row_ready.dtype != torch.bool
            or self.candidate_utility.ndim != 0
            or set(self.row_radii) != {1, 2, 3, 4}
        ):
            raise FormalPContractError("fanout score/denominator population drift")
        for index, status in enumerate(self.root_statuses):
            selected = (
                self.selected_action_indices[index] is not None
                and self.selected_action_keys[index] is not None
            )
            if (status == ROOT_READY) != selected or status not in (
                ROOT_READY,
                ROOT_H0,
            ):
                raise FormalPContractError("fanout selected-action/H0 drift")
            if status == ROOT_H0 and bool(self.root_scores[index].ne(0.0)):
                raise FormalPContractError("fanout root H0 is not exact zero")
        if bool(self.row_scores[~self.row_ready].ne(0.0).any()):
            raise FormalPContractError("fanout row H0 is not exact zero")
        if self.selected_bank_ordinal is None:
            if self.selected_row_sha256 is not None or self.lock_status != LOCK_H0:
                raise FormalPContractError("fanout MAP H0/lock status drift")
        elif (
            self.selected_bank_ordinal not in range(rows)
            or self.selected_row_sha256
            != self.row_hashes[self.selected_bank_ordinal]
            or not bool(self.row_ready[self.selected_bank_ordinal])
            or self.lock_status != LOCK_READY
        ):
            raise FormalPContractError("fanout MAP READY/lock status drift")


def decision_from_legacy_output(
    output: DeployableDirectionOutput,
) -> FanoutDirectionDecision:
    """Erase the heavy feature record while preserving every lock decision bit."""

    if not isinstance(output, DeployableDirectionOutput):
        raise FormalPContractError("fanout decision requires legacy deployable output")
    record = output.record
    regions = tuple(
        enumerate_superregion_bank(
            record.colnomic_query_grid_shape, include_r0_control=False
        )
    )
    root_statuses = tuple(
        ROOT_READY if index is not None else ROOT_H0
        for index in output.selected_action_indices
    )
    return FanoutDirectionDecision(
        candidate_position=record.candidate_position,
        candidate_key=record.candidate_key,
        candidate_physical_row=record.candidate_physical_row,
        candidate_reference_source_sha256=record.candidate_reference_source_sha256,
        direction=record.direction,
        selected_action_indices=output.selected_action_indices,
        selected_action_keys=output.selected_action_keys,
        root_statuses=root_statuses,
        root_scores=_cpu_tensor(output.root_scores, name="root scores"),
        row_hashes=tuple(superregion_sha256(item) for item in regions),
        row_radii=tuple(item.radius for item in regions),
        row_scores=_cpu_tensor(output.row_scores, name="row scores"),
        row_ready=torch.as_tensor(output.row_ready, dtype=torch.bool)
        .detach()
        .cpu()
        .contiguous()
        .clone(),
        candidate_utility=_cpu_tensor(
            output.candidate_utility, name="candidate utility"
        ),
        selected_bank_ordinal=output.map_row_index,
        selected_row_sha256=output.map_row_sha256,
        lock_status=LOCK_H0 if output.map_row_index is None else LOCK_READY,
    )


def _tensor_bytes_payload(value: torch.Tensor) -> dict[str, object]:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes_hex": tensor.numpy().tobytes(order="C").hex(),
    }


def decision_bytes(value: FanoutDirectionDecision) -> bytes:
    """Canonical bytes used by the exact four-pass equivalence validator."""

    if not isinstance(value, FanoutDirectionDecision):
        raise FormalPContractError("fanout byte encoder received wrong decision type")
    payload: dict[str, object] = {
        "candidate_position": value.candidate_position,
        "candidate_key": value.candidate_key,
        "candidate_physical_row": value.candidate_physical_row,
        "candidate_reference_source_sha256": value.candidate_reference_source_sha256,
        "direction": value.direction,
        "selected_action_indices": list(value.selected_action_indices),
        "selected_action_keys": list(value.selected_action_keys),
        "root_statuses": list(value.root_statuses),
        "root_scores": _tensor_bytes_payload(value.root_scores),
        "row_hashes": list(value.row_hashes),
        "row_radii": list(value.row_radii),
        "row_scores": _tensor_bytes_payload(value.row_scores),
        "row_ready": _tensor_bytes_payload(value.row_ready),
        "candidate_utility": _tensor_bytes_payload(value.candidate_utility),
        "selected_bank_ordinal": value.selected_bank_ordinal,
        "selected_row_sha256": value.selected_row_sha256,
        "lock_status": value.lock_status,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def build_compact_p_lock_record(
    output: CompactDeployableDirectionOutput,
    *,
    crossfit_role: str,
    outer_fold: int,
    p_checkpoint_sha256: str,
    p_training_manifest_sha256: str,
) -> dict[str, object]:
    """Build the byte-exact legacy lock from selected compact deployments.

    No unselected deployment is decoded.  ROOT_H0 entries use the unchanged
    legacy exact-zero synthetic deployment, while every ROOT_READY entry uses
    the single deployment selected by its P head.
    """

    if not isinstance(output, CompactDeployableDirectionOutput):
        raise FormalPContractError("compact lock builder received wrong output type")
    require_sha256(p_checkpoint_sha256, name="P checkpoint")
    require_sha256(p_training_manifest_sha256, name="P training manifest")
    if not isinstance(crossfit_role, str) or not crossfit_role:
        raise FormalPContractError("P lock crossfit role is empty")
    if outer_fold not in (1, 2, 3, 4):
        raise FormalPContractError("P lock outer fold must be 1..4")

    source = output.compact_input.source_index
    query_grid = source.deployment_query_root_masks[0].grid_shape
    reference_grid = source.deployment_reference_mask_table[0].grid_shape
    structure = output.prepared_structure
    geometry = output.prepared_row_geometry
    regions = structure.regions
    complete_bank_sha256 = structure.complete_bank_sha256
    all_root_ordinals = tuple(range(source.root_count))
    all_root_action_keys: list[str | None] = []
    all_root_deployments: list[RootActionDeployment] = []
    for root, deployment in enumerate(output.selected_deployments):
        if deployment is None:
            deployment = make_root_action_deployment(
                action_key="0" * 64,
                eligible=False,
                query_mask=torch.zeros(math.prod(query_grid), dtype=torch.bool),
                reference_mask=torch.zeros(
                    math.prod(reference_grid), dtype=torch.bool
                ),
                query_grid_shape=query_grid,
                reference_grid_shape=reference_grid,
                query_geometry_sha256=source.query_geometry_sha256,
                reference_geometry_sha256=source.reference_geometry_sha256,
            )
            all_root_action_keys.append(None)
        else:
            action_index = output.selected_action_indices[root]
            if (
                action_index is None
                or not deployment.eligible
                or deployment.binding_status != ROOT_READY
                or output.selected_action_keys[root] != deployment.action_key
            ):
                raise FormalPContractError(
                    "selected compact root action/deployment binding drift"
                )
            all_root_action_keys.append(deployment.action_key)
        all_root_deployments.append(deployment)

    if output.map_row_index is None:
        selected_ordinal: int | None = None
        selected_sha: str | None = None
        roots: tuple[int, ...] = ()
        deployments: tuple[RootActionDeployment, ...] = ()
        query_union = torch.zeros(math.prod(query_grid), dtype=torch.bool)
        status = LOCK_H0
    else:
        selected_ordinal = output.map_row_index
        region = regions[selected_ordinal]
        selected_sha = structure.row_hashes[selected_ordinal]
        roots = region.contributing_root_ordinals
        chosen: list[RootActionDeployment] = []
        for root in roots:
            deployment = output.selected_deployments[root]
            if deployment is None or not deployment.eligible:
                raise FormalPContractError(
                    "READY compact row references geometry-H0 action"
                )
            chosen.append(deployment)
        deployments = tuple(chosen)
        query_union = geometry.row_query_unions[selected_ordinal]
        if not geometry.row_ready[selected_ordinal]:
            raise FormalPContractError(
                "READY compact P lock query union is not connected"
            )
        status = LOCK_READY

    per_radius = {
        str(radius): count
        for radius, count in zip(
            (1, 2, 3, 4), structure.row_count_by_radius, strict=True
        )
    }
    core: dict[str, object] = {
        "query_id": source.query_id,
        "historical_query_ordinal": source.historical_query_ordinal,
        "execution_ordinal": source.execution_ordinal,
        "query_source_image_sha256": source.query_source_image_sha256,
        "outer_fold": outer_fold,
        "crossfit_role": crossfit_role,
        "p_checkpoint_sha256": p_checkpoint_sha256,
        "p_training_manifest_sha256": p_training_manifest_sha256,
        "candidate_physical_row": source.candidate_physical_row,
        "candidate_reference_source_sha256": source.candidate_reference_source_sha256,
        "direction": source.direction,
        "complete_bank_sha256": complete_bank_sha256,
        "selected_bank_ordinal": selected_ordinal,
        "selected_row_sha256": selected_sha,
        "status": status,
        "query_union_mask_rle": encode_mask_rle(query_union),
        "query_union_mask_sha256": tensor_sha256(query_union),
        "ordered_root_ordinals": list(roots),
        "root_query_tile_mask_rle": [
            encode_mask_rle(item.query_mask) for item in deployments
        ],
        "root_query_tile_mask_sha256": [
            tensor_sha256(item.query_mask) for item in deployments
        ],
        "root_reference_component_mask_rle": [
            encode_mask_rle(item.reference_mask) for item in deployments
        ],
        "root_reference_component_mask_sha256": [
            tensor_sha256(item.reference_mask) for item in deployments
        ],
        "root_binding_status": [item.binding_status for item in deployments],
        "all_root_ordinals": list(all_root_ordinals),
        "all_root_action_key_sha256": all_root_action_keys,
        "all_root_query_tile_mask_rle": [
            encode_mask_rle(item.query_mask) for item in all_root_deployments
        ],
        "all_root_query_tile_mask_sha256": [
            tensor_sha256(item.query_mask) for item in all_root_deployments
        ],
        "all_root_reference_component_mask_rle": [
            encode_mask_rle(item.reference_mask)
            for item in all_root_deployments
        ],
        "all_root_reference_component_mask_sha256": [
            tensor_sha256(item.reference_mask)
            for item in all_root_deployments
        ],
        "all_root_binding_status": [
            item.binding_status for item in all_root_deployments
        ],
        "query_geometry_sha256": source.query_geometry_sha256,
        "reference_geometry_sha256": source.reference_geometry_sha256,
        "fixed_denominator": {
            "radii": [1, 2, 3, 4],
            "row_count_by_radius": per_radius,
            "total_row_count": len(regions),
            "ineligible_rows_retained_as_exact_zero": True,
            "available_root_renormalization": False,
        },
        "erased_fields": list(P_LOCK_ERASED_FIELDS),
    }
    if set(core) != set(P_LOCK_REQUIRED_FIELDS) - {"record_sha256"}:
        raise FormalPContractError("compact P lock core field set drift")
    core["record_sha256"] = canonical_sha256(core)
    validate_p_lock_record(
        core,
        query_grid_shape=query_grid,
        reference_grid_shape=reference_grid,
    )
    return core


def compact_lock_record_bytes(value: Mapping[str, object]) -> bytes:
    """Canonical complete lock bytes; no score or feature is persisted."""

    if set(value) != set(P_LOCK_REQUIRED_FIELDS):
        raise FormalPContractError("compact lock byte encoder field-set drift")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class CompactFanoutDirectionDecision:
    decision: FanoutDirectionDecision
    lock_record: Mapping[str, object]
    selected_deployment_decode_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.decision, FanoutDirectionDecision):
            raise FormalPContractError("compact fanout decision schema drift")
        lock = dict(self.lock_record)
        if (
            set(lock) != set(P_LOCK_REQUIRED_FIELDS)
            or lock.get("candidate_physical_row")
            != self.decision.candidate_physical_row
            or lock.get("candidate_reference_source_sha256")
            != self.decision.candidate_reference_source_sha256
            or lock.get("direction") != self.decision.direction
            or lock.get("selected_bank_ordinal")
            != self.decision.selected_bank_ordinal
            or lock.get("selected_row_sha256")
            != self.decision.selected_row_sha256
            or lock.get("status") != self.decision.lock_status
            or not 0
            <= self.selected_deployment_decode_count
            <= sum(
                status == ROOT_READY for status in self.decision.root_statuses
            )
        ):
            raise FormalPContractError(
                "compact fanout decision/complete-lock binding drift"
            )
        if lock.get("record_sha256") != canonical_sha256(
            {key: item for key, item in lock.items() if key != "record_sha256"}
        ):
            raise FormalPContractError("compact fanout lock hash drift")
        object.__setattr__(self, "lock_record", lock)


@dataclass(frozen=True)
class FanoutScopeResult:
    scope: CrossfitFanoutScope
    decisions: tuple[FanoutDirectionDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CrossfitFanoutScope) or not self.decisions:
            raise FormalPContractError("fanout scope result is empty or unbound")


@dataclass(frozen=True)
class QueryLockFanoutResult:
    """Four exact decision populations emitted from one input stream pass."""

    query_id: str
    historical_query_ordinal: int
    execution_ordinal: int
    query_source_image_sha256: str
    source_fold: int
    candidate_count: int
    candidate_axis: tuple[tuple[int, str, int, str], ...]
    scope_results: tuple[FanoutScopeResult, ...]
    input_record_count: int

    def __post_init__(self) -> None:
        expected_scopes = crossfit_fanout_scopes(self.source_fold)
        if (
            not self.query_id
            or self.candidate_count <= 0
            or len(self.candidate_axis) != self.candidate_count
            or tuple(item.scope for item in self.scope_results) != expected_scopes
            or self.input_record_count
            != self.candidate_count * len(P_DIRECTIONS)
        ):
            raise FormalPContractError("query lock fanout population drift")
        if tuple(item[0] for item in self.candidate_axis) != tuple(
            range(self.candidate_count)
        ):
            raise FormalPContractError("fanout candidate axis is not contiguous")
        expected_coordinates = tuple(
            (position, direction)
            for position in range(self.candidate_count)
            for direction in P_DIRECTIONS
        )
        for result in self.scope_results:
            coordinates = tuple(
                (item.candidate_position, item.direction)
                for item in result.decisions
            )
            if coordinates != expected_coordinates:
                raise FormalPContractError(
                    "fanout scope is not complete candidate x direction"
                )

    def by_fit_id(self) -> dict[str, FanoutScopeResult]:
        return {item.scope.fit_id: item for item in self.scope_results}


def _canonical_scoped_heads(
    scoped_heads: Sequence[ScopedPHead], *, source_fold: int
) -> tuple[ScopedPHead, ...]:
    expected = crossfit_fanout_scopes(source_fold)
    values = tuple(scoped_heads)
    if len(values) != 4 or any(not isinstance(item, ScopedPHead) for item in values):
        raise FormalPContractError("lock fanout requires exactly four scoped P heads")
    by_id = {item.scope.fit_id: item for item in values}
    if len(by_id) != 4 or set(by_id) != {item.fit_id for item in expected}:
        raise FormalPContractError("lock fanout checkpoint scope set drift")
    ordered = tuple(by_id[item.fit_id] for item in expected)
    if tuple(item.scope for item in ordered) != expected:
        raise FormalPContractError("lock fanout checkpoint metadata drift")
    return ordered


def fanout_query_feature_stream(
    records: Iterable[DirectionalFeatureRecord],
    scoped_heads: Sequence[ScopedPHead],
    *,
    expected_candidate_count: int,
) -> QueryLockFanoutResult:
    """Consume one query feature stream once and score all four frozen heads.

    The function intentionally iterates ``records`` exactly once.  It retains
    only compact decision outputs and addressing receipts; the heavyweight
    ``DirectionalFeatureRecord`` is released after its four head evaluations.
    Input order may vary, but the returned natural candidate axis is canonical
    by ``candidate_position`` and checkerboard direction.
    """

    if (
        isinstance(expected_candidate_count, bool)
        or not isinstance(expected_candidate_count, int)
        or expected_candidate_count <= 0
    ):
        raise FormalPContractError("fanout expected candidate count is invalid")

    ordered_heads: tuple[ScopedPHead, ...] | None = None
    decisions: dict[str, list[FanoutDirectionDecision]] = {}
    coordinates: set[tuple[int, str]] = set()
    candidate_axis: dict[int, tuple[int, str, int, str]] = {}
    key_positions: dict[str, int] = {}
    physical_positions: dict[int, int] = {}
    query_anchor: tuple[str, int, int, str, int] | None = None
    input_record_count = 0

    for record in records:
        input_record_count += 1
        if not isinstance(record, DirectionalFeatureRecord):
            raise FormalPContractError("fanout stream item is not a feature record")
        anchor = (
            record.query_id,
            record.historical_query_ordinal,
            record.execution_ordinal,
            record.query_source_image_sha256,
            record.source_fold,
        )
        if query_anchor is None:
            query_anchor = anchor
            ordered_heads = _canonical_scoped_heads(
                scoped_heads, source_fold=record.source_fold
            )
            decisions = {item.scope.fit_id: [] for item in ordered_heads}
        elif anchor != query_anchor:
            raise FormalPContractError("fanout stream crossed query/source-fold boundary")

        coordinate = (record.candidate_position, record.direction)
        if coordinate in coordinates:
            raise FormalPContractError("fanout stream contains duplicate coordinate")
        coordinates.add(coordinate)
        axis_item = (
            record.candidate_position,
            record.candidate_key,
            record.candidate_physical_row,
            record.candidate_reference_source_sha256,
        )
        prior = candidate_axis.setdefault(record.candidate_position, axis_item)
        if prior != axis_item:
            raise FormalPContractError("fanout candidate directions disagree")
        prior_key = key_positions.setdefault(
            record.candidate_key, record.candidate_position
        )
        prior_row = physical_positions.setdefault(
            record.candidate_physical_row, record.candidate_position
        )
        if prior_key != record.candidate_position or prior_row != record.candidate_position:
            raise FormalPContractError("fanout candidate key/physical row is duplicated")

        assert ordered_heads is not None
        with torch.no_grad():
            for item in ordered_heads:
                output = score_deployable_direction(item.model, record)
                decisions[item.scope.fit_id].append(
                    decision_from_legacy_output(output)
                )

    if query_anchor is None or ordered_heads is None:
        raise FormalPContractError("fanout query feature stream is empty")
    if set(candidate_axis) != set(range(expected_candidate_count)):
        raise FormalPContractError("fanout stream does not retain complete candidate axis")
    expected_coordinates = {
        (position, direction)
        for position in range(expected_candidate_count)
        for direction in P_DIRECTIONS
    }
    if coordinates != expected_coordinates:
        raise FormalPContractError("fanout stream is missing candidate/direction records")

    scope_results = []
    for item in ordered_heads:
        ordered_decisions = tuple(
            sorted(
                decisions[item.scope.fit_id],
                key=lambda value: (
                    value.candidate_position,
                    _DIRECTION_ORDINAL[value.direction],
                ),
            )
        )
        scope_results.append(
            FanoutScopeResult(scope=item.scope, decisions=ordered_decisions)
        )
    return QueryLockFanoutResult(
        query_id=query_anchor[0],
        historical_query_ordinal=query_anchor[1],
        execution_ordinal=query_anchor[2],
        query_source_image_sha256=query_anchor[3],
        source_fold=query_anchor[4],
        candidate_count=expected_candidate_count,
        candidate_axis=tuple(candidate_axis[index] for index in range(expected_candidate_count)),
        scope_results=tuple(scope_results),
        input_record_count=input_record_count,
    )


@dataclass(frozen=True)
class CompactFanoutScopeResult:
    scope: CrossfitFanoutScope
    decisions: tuple[CompactFanoutDirectionDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CrossfitFanoutScope) or not self.decisions:
            raise FormalPContractError("compact fanout scope result is empty")


@dataclass(frozen=True)
class CompactFanoutExecutionReceipt:
    input_source_count: int
    candidate_action_table_build_count: int
    full_deployment_object_count: int
    selected_deployment_decode_count: int
    selected_deployment_decode_limit: int
    maximum_selected_decodes_per_source: int
    maximum_selected_decode_limit_per_source: int

    def __post_init__(self) -> None:
        if (
            self.input_source_count != NATURAL_CANDIDATE_COUNT * len(P_DIRECTIONS)
            or self.candidate_action_table_build_count != self.input_source_count
            or self.full_deployment_object_count != 0
            or self.selected_deployment_decode_count < 0
            or self.selected_deployment_decode_count
            > self.selected_deployment_decode_limit
            or self.maximum_selected_decodes_per_source
            > self.maximum_selected_decode_limit_per_source
            or self.maximum_selected_decode_limit_per_source <= 0
        ):
            raise FormalPContractError("compact fanout execution receipt drift")


@dataclass(frozen=True)
class CompactQueryLockFanoutResult:
    query_id: str
    historical_query_ordinal: int
    execution_ordinal: int
    query_source_image_sha256: str
    source_fold: int
    candidate_axis_sha256: str
    candidate_axis: tuple[tuple[int, str, int, str], ...]
    scope_results: tuple[CompactFanoutScopeResult, ...]
    execution_receipt: CompactFanoutExecutionReceipt

    def __post_init__(self) -> None:
        require_sha256(self.candidate_axis_sha256, name="compact candidate axis")
        expected_scopes = crossfit_fanout_scopes(self.source_fold)
        if (
            not self.query_id
            or len(self.candidate_axis) != NATURAL_CANDIDATE_COUNT
            or tuple(item.scope for item in self.scope_results) != expected_scopes
            or tuple(item[0] for item in self.candidate_axis)
            != tuple(range(NATURAL_CANDIDATE_COUNT))
        ):
            raise FormalPContractError("compact query fanout population drift")
        expected_coordinates = tuple(
            (position, direction)
            for position in range(NATURAL_CANDIDATE_COUNT)
            for direction in P_DIRECTIONS
        )
        for result in self.scope_results:
            coordinates = tuple(
                (
                    item.decision.candidate_position,
                    item.decision.direction,
                )
                for item in result.decisions
            )
            if coordinates != expected_coordinates:
                raise FormalPContractError(
                    "compact scope is not complete natural C128 x direction"
                )

    def by_fit_id(self) -> dict[str, CompactFanoutScopeResult]:
        return {item.scope.fit_id: item for item in self.scope_results}


def fanout_compact_query_sources(
    values: Iterable[CompactDirectionalFeatureInput],
    scoped_heads: Sequence[ScopedPHead],
) -> CompactQueryLockFanoutResult:
    """Single-pass natural-C128 compact fanout with complete erased locks.

    Each source/feature pair is consumed exactly once and converted to one old
    action table.  Four heads reuse that table.  Only each head's selected
    action is decoded for a root; the full root x action deployment population
    is never materialized.
    """

    ordered_heads: tuple[ScopedPHead, ...] | None = None
    decisions: dict[str, list[CompactFanoutDirectionDecision]] = {}
    coordinates: set[tuple[int, str]] = set()
    candidate_axis: dict[int, tuple[int, str, int, str]] = {}
    key_positions: dict[str, int] = {}
    physical_positions: dict[int, int] = {}
    query_anchor: tuple[str, int, int, str, int, str] | None = None
    query_side_by_direction: dict[str, tuple[object, ...]] = {}
    input_source_count = 0
    action_table_build_count = 0
    selected_decode_count = 0
    selected_decode_limit = 0
    maximum_selected = 0
    maximum_limit = 0
    row_geometry_cache: dict[
        tuple[object, ...], _PreparedCompactRowGeometry
    ] = {}

    for value in values:
        input_source_count += 1
        if not isinstance(value, CompactDirectionalFeatureInput):
            raise FormalPContractError(
                "compact fanout stream item has the wrong type"
            )
        source = value.source_index
        anchor = (
            source.query_id,
            source.historical_query_ordinal,
            source.execution_ordinal,
            source.query_source_image_sha256,
            value.source_fold,
            source.candidate_axis_sha256,
        )
        if query_anchor is None:
            query_anchor = anchor
            ordered_heads = _canonical_scoped_heads(
                scoped_heads, source_fold=value.source_fold
            )
            if any(
                item.p_checkpoint_sha256 is None
                or item.p_training_manifest_sha256 is None
                for item in ordered_heads
            ):
                raise FormalPContractError(
                    "compact lock fanout requires bound checkpoint/manifest hashes"
                )
            decisions = {item.scope.fit_id: [] for item in ordered_heads}
        elif anchor != query_anchor:
            raise FormalPContractError(
                "compact fanout stream crossed query/fold/candidate-axis boundary"
            )

        query_side = (
            source.query_token_sha256,
            source.query_valid_mask_sha256,
            source.query_geometry_sha256,
            source.colnomic_query_root_masks[0].grid_shape,
            tuple(
                item.logical_sha256 for item in source.colnomic_query_root_masks
            ),
            source.deployment_query_root_masks[0].grid_shape,
            tuple(
                item.logical_sha256 for item in source.deployment_query_root_masks
            ),
        )
        prior_query_side = query_side_by_direction.setdefault(
            source.direction, query_side
        )
        if prior_query_side != query_side:
            raise FormalPContractError(
                "compact fanout query-side source drift within direction"
            )

        coordinate = (source.candidate_position, source.direction)
        if coordinate in coordinates:
            raise FormalPContractError(
                "compact fanout stream contains duplicate coordinate"
            )
        coordinates.add(coordinate)
        axis_item = (
            source.candidate_position,
            source.candidate_key,
            source.candidate_physical_row,
            source.candidate_reference_source_sha256,
        )
        prior = candidate_axis.setdefault(source.candidate_position, axis_item)
        if prior != axis_item:
            raise FormalPContractError("compact candidate directions disagree")
        prior_key = key_positions.setdefault(
            source.candidate_key, source.candidate_position
        )
        prior_row = physical_positions.setdefault(
            source.candidate_physical_row, source.candidate_position
        )
        if prior_key != source.candidate_position or prior_row != source.candidate_position:
            raise FormalPContractError(
                "compact candidate key/physical row is duplicated"
            )

        action_table = candidate_action_table_from_compact_tensor(
            source, value.candidate_features
        )
        action_table_build_count += 1
        structure = _prepare_compact_structure(
            source.colnomic_query_root_masks[0].grid_shape
        )
        geometry_key = _compact_row_geometry_key(source, structure)
        prepared_geometry = row_geometry_cache.get(geometry_key)
        if prepared_geometry is None:
            prepared_geometry = _prepare_compact_row_geometry(
                source, structure
            )
            row_geometry_cache[geometry_key] = prepared_geometry
        source_selected = 0
        source_limit = 4 * source.root_count
        deployment_cache: dict[tuple[int, int], RootActionDeployment] = {}
        assert ordered_heads is not None
        with torch.no_grad():
            for item in ordered_heads:
                output = _score_compact_with_table(
                    item.model,
                    value,
                    action_table,
                    deployment_cache=deployment_cache,
                    prepared_structure=structure,
                    prepared_row_geometry=prepared_geometry,
                )
                source_selected += output.selected_deployment_decode_count
                lock = build_compact_p_lock_record(
                    output,
                    crossfit_role=item.scope.crossfit_role,
                    outer_fold=item.scope.outer_fold,
                    p_checkpoint_sha256=str(item.p_checkpoint_sha256),
                    p_training_manifest_sha256=str(
                        item.p_training_manifest_sha256
                    ),
                )
                decisions[item.scope.fit_id].append(
                    CompactFanoutDirectionDecision(
                        decision=decision_from_compact_output(output),
                        lock_record=lock,
                        selected_deployment_decode_count=(
                            output.selected_deployment_decode_count
                        ),
                    )
                )
        if source_selected > source_limit:
            raise FormalPContractError(
                "compact fanout decoded more than four selected actions per root"
            )
        selected_decode_count += source_selected
        selected_decode_limit += source_limit
        maximum_selected = max(maximum_selected, source_selected)
        maximum_limit = max(maximum_limit, source_limit)

    if query_anchor is None or ordered_heads is None:
        raise FormalPContractError("compact query source stream is empty")
    expected_coordinates = {
        (position, direction)
        for position in range(NATURAL_CANDIDATE_COUNT)
        for direction in P_DIRECTIONS
    }
    if coordinates != expected_coordinates or set(candidate_axis) != set(
        range(NATURAL_CANDIDATE_COUNT)
    ):
        raise FormalPContractError(
            "compact fanout stream is incomplete natural C128 x direction"
        )

    scope_results: list[CompactFanoutScopeResult] = []
    for item in ordered_heads:
        ordered_decisions = tuple(
            sorted(
                decisions[item.scope.fit_id],
                key=lambda value: (
                    value.decision.candidate_position,
                    _DIRECTION_ORDINAL[value.decision.direction],
                ),
            )
        )
        scope_results.append(
            CompactFanoutScopeResult(
                scope=item.scope, decisions=ordered_decisions
            )
        )
    receipt = CompactFanoutExecutionReceipt(
        input_source_count=input_source_count,
        candidate_action_table_build_count=action_table_build_count,
        full_deployment_object_count=0,
        selected_deployment_decode_count=selected_decode_count,
        selected_deployment_decode_limit=selected_decode_limit,
        maximum_selected_decodes_per_source=maximum_selected,
        maximum_selected_decode_limit_per_source=maximum_limit,
    )
    return CompactQueryLockFanoutResult(
        query_id=query_anchor[0],
        historical_query_ordinal=query_anchor[1],
        execution_ordinal=query_anchor[2],
        query_source_image_sha256=query_anchor[3],
        source_fold=query_anchor[4],
        candidate_axis_sha256=query_anchor[5],
        candidate_axis=tuple(
            candidate_axis[index] for index in range(NATURAL_CANDIDATE_COUNT)
        ),
        scope_results=tuple(scope_results),
        execution_receipt=receipt,
    )


def scope_decision_bytes(
    result: FanoutScopeResult,
) -> tuple[bytes, ...]:
    """Return candidate/direction bytes in canonical natural-axis order."""

    if not isinstance(result, FanoutScopeResult):
        raise FormalPContractError("scope byte encoder received wrong result type")
    return tuple(decision_bytes(item) for item in result.decisions)


def decision_index(
    result: FanoutScopeResult,
) -> Mapping[tuple[str, str], FanoutDirectionDecision]:
    """Read-only-style candidate-key/direction lookup for synthetic validators."""

    if not isinstance(result, FanoutScopeResult):
        raise FormalPContractError("fanout decision index received wrong result type")
    return {(item.candidate_key, item.direction): item for item in result.decisions}


__all__ = [
    "SCHEMA_VERSION",
    "INNER_FIT_ROLE",
    "OUTER_REFIT_ROLE",
    "INNER_LOCK_ROLE",
    "OUTER_LOCK_ROLE",
    "CrossfitFanoutScope",
    "ScopedPHead",
    "FanoutDirectionDecision",
    "FanoutScopeResult",
    "QueryLockFanoutResult",
    "CompactDirectionalFeatureInput",
    "CompactDeployableDirectionOutput",
    "CompactFanoutDirectionDecision",
    "CompactFanoutScopeResult",
    "CompactFanoutExecutionReceipt",
    "CompactQueryLockFanoutResult",
    "crossfit_fanout_scopes",
    "decision_from_legacy_output",
    "decision_from_compact_output",
    "decision_bytes",
    "compact_lock_record_bytes",
    "score_compact_deployable_direction",
    "build_compact_p_lock_record",
    "scope_decision_bytes",
    "decision_index",
    "fanout_query_feature_stream",
    "fanout_compact_query_sources",
]
