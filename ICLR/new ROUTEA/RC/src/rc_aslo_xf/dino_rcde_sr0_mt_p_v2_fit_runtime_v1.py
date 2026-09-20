"""Role-safe V95 input compiler for the successor P-V2 fit.

This module is deliberately narrower than a trainer.  It validates one V95
fit manifest, reconstructs the complete target-free two-member/two-direction
input table from the bound pair-feature and structural-sidecar shards, seals
that table, and only then permits the target/rival role join.  It also exposes
an independently written compiled scorer whose output is required to be exact
to the already qualified role-free scorer.

There is no optimizer, update loop, checkpoint writer, natural file reader, or
lock materializer here.  In particular the obsolete V1 MAP/H0 decision is not
consumed by either the compiler or the role join.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .cw1_sr0_structure_v1 import enumerate_superregion_bank, superregion_sha256
from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    CandidateActionFeatureTable,
    FEATURE_DIM,
    FEATURE_SCHEMA_SHA256,
    HARD_MAX_TIE_ULPS,
    SharedMultitilePHead,
    canonical_action_argmax,
    scale_balanced_hierarchical_logmeanexp,
)
from .dino_rcde_sr0_mt_p_compact_catalog_v1 import (
    P_DIRECTIONS,
    RoleFreePairFeatureCache,
    deserialize_role_free_pair_feature_cache,
)
from .dino_rcde_sr0_mt_p_role_free_e0_consumer_v1 import (
    RoleFreeCompactDirectionalInput,
    RoleFreeCompactDirectionOutput,
    score_role_free_compact_direction,
    structure_from_sidecar_record,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    MIN_READY_ROOTS,
    FormalPContractError,
    _component_is_legal,
    canonical_sha256,
    tensor_sha256,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_v2_fit_runtime_v1_20260818"
V95_MANIFEST_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_training_manifest_v1_20260818"
V95_MANIFEST_STATUS = "RCDE_SR0_MT_P_V2_TRAINING_MANIFEST_READY"
PAIR_SHARD_SCHEMA = "rc_dino_rcde_sr0_mt_role_free_pair_feature_cache_shard_v1_20260817"
PAIR_SHARD_STATUS = "RCDE_SR0_MT_ROLE_FREE_PAIR_FEATURE_CACHE_SHARD_READY"
SIDECAR_SHARD_SCHEMA = "rc_dino_rcde_sr0_mt_training_consumer_e0_full594_structure_sidecar_shard_v1_20260818"
SIDECAR_SHARD_STATUS = "RCDE_SR0_MT_TRAINING_CONSUMER_E0_FULL594_STRUCTURE_SIDECAR_SHARD_READY"
FIT_ROLES = ("INNER_TRAIN_FIT", "OUTER_TRAIN_REFIT")


class PFitV2RuntimeError(FormalPContractError):
    """A V95 manifest, shard join, seal, or parity invariant failed."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise PFitV2RuntimeError(message)


def _sha(value: object, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{name} SHA drift",
    )
    return value


def _logical(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "fit_id",
        "fit_role",
        "outer_fold",
        "inner_heldout_fold",
        "identity_disjoint",
        "supergroup_disjoint",
        "train_query_count",
        "eligible_episode_count",
        "episodes",
        "episode_population_sha256",
        "membership_path",
        "membership_file_sha256",
        "membership_logical_sha256",
        "pair_feature_shards",
        "pair_feature_shards_sha256",
        "structure_sidecar_shards",
        "structure_sidecar_shards_sha256",
        "source_bindings",
        "recipe",
        "output_namespace",
        "fresh_resume_exact_required",
        "independent_fit_validation_required",
        "training_authorized",
        "consumable_checkpoint_authorized",
        "scientific_GO_or_NO_GO",
        "automatic_stage_advance",
        "next_authorized_stage",
        "logical_sha256",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        "execution_ordinal",
        "query_id",
        "query_source_image_sha256",
        "pair_address_sha256",
        "target_member_ordinal",
        "target_candidate_key",
        "rival_member_ordinal",
        "rival_candidate_key",
    }
)
_ROUTE_FIELDS = frozenset(
    {
        "shard_ordinal",
        "execution_start",
        "execution_stop",
        "selected_execution_ordinals",
        "artifact_path",
        "artifact_sha256",
        "validation_path",
        "validation_sha256",
    }
)
_SOURCE_BINDING_FIELDS = frozenset(
    {
        "pair_aggregate_logical_sha256",
        "pair_validation_logical_sha256",
        "loss_join_logical_sha256",
        "loss_validation_logical_sha256",
        "membership_validation_logical_sha256",
        "sidecar_manifest_logical_sha256",
        "v94_adapter_validation_logical_sha256",
        "feature_schema_sha256",
        "feature_implementation_sha256",
        "numeric_policy_sha256",
        "p_v2_adapter_sha256",
        "p_runtime_sha256",
        "role_free_scorer_sha256",
    }
)
_RECIPE = {
    "seed": 17,
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "gradient_clip_l2": 1.0,
    "updates_total": 2048,
    "episodes_per_update": 4,
    "warmup_updates": 128,
    "final_learning_rate": 0.00003,
    "checkpoint_interval_updates": 64,
    "early_stop": False,
    "auxiliary_loss": False,
    "direction_fusion_before_one_pair_loss": True,
    "loss": "softplus(-(U_target-U_rival))",
}


@dataclass(frozen=True)
class NeutralEpisodeAddressV1:
    execution_ordinal: int
    query_id: str
    query_source_image_sha256: str
    pair_address_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "execution_ordinal": self.execution_ordinal,
            "query_id": self.query_id,
            "query_source_image_sha256": self.query_source_image_sha256,
            "pair_address_sha256": self.pair_address_sha256,
        }


@dataclass(frozen=True)
class TrainingRoleAddressV1:
    neutral: NeutralEpisodeAddressV1
    target_member_ordinal: int
    target_candidate_key: str
    rival_member_ordinal: int
    rival_candidate_key: str


@dataclass(frozen=True)
class ShardRouteV1:
    shard_ordinal: int
    execution_start: int
    execution_stop: int
    selected_execution_ordinals: tuple[int, ...]
    artifact_path: str
    artifact_sha256: str
    validation_path: str
    validation_sha256: str


@dataclass(frozen=True)
class ValidatedV95FitManifestV1:
    fit_id: str
    fit_role: str
    outer_fold: int
    inner_heldout_fold: int | None
    neutral_episodes: tuple[NeutralEpisodeAddressV1, ...]
    role_addresses: tuple[TrainingRoleAddressV1, ...]
    pair_routes: tuple[ShardRouteV1, ...]
    sidecar_routes: tuple[ShardRouteV1, ...]
    output_namespace: str
    manifest_logical_sha256: str
    source_bindings: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_bindings", MappingProxyType(dict(self.source_bindings))
        )


def _route(raw: object, *, name: str) -> ShardRouteV1:
    _require(isinstance(raw, Mapping) and set(raw) == _ROUTE_FIELDS, f"{name} field drift")
    start, stop = raw["execution_start"], raw["execution_stop"]
    ordinal = raw["shard_ordinal"]
    selected_raw = raw["selected_execution_ordinals"]
    _require(
        type(start) is int
        and type(stop) is int
        and 0 <= start < stop <= 600
        and type(ordinal) is int
        and ordinal >= 0
        and isinstance(selected_raw, list),
        f"{name} range drift",
    )
    selected = tuple(selected_raw)
    _require(
        selected
        and all(type(item) is int and start <= item < stop for item in selected)
        and selected == tuple(sorted(set(selected))),
        f"{name} selected execution drift",
    )
    for field in ("artifact_path", "validation_path"):
        value = raw[field]
        _require(
            isinstance(value, str)
            and value
            and not value.startswith("/")
            and ".." not in value.split("/"),
            f"{name} unsafe {field}",
        )
    return ShardRouteV1(
        shard_ordinal=ordinal,
        execution_start=start,
        execution_stop=stop,
        selected_execution_ordinals=selected,
        artifact_path=raw["artifact_path"],
        artifact_sha256=_sha(raw["artifact_sha256"], f"{name} artifact"),
        validation_path=raw["validation_path"],
        validation_sha256=_sha(raw["validation_sha256"], f"{name} validation"),
    )


def validate_v95_fit_manifest(
    value: Mapping[str, Any], *, expected_fit_id: str | None = None
) -> ValidatedV95FitManifestV1:
    """Strictly decode one immutable V95 per-fit input manifest."""

    _require(set(value) == _MANIFEST_FIELDS, "V95 fit manifest field set drift")
    _require(
        value.get("schema_version") == V95_MANIFEST_SCHEMA
        and value.get("status") == V95_MANIFEST_STATUS
        and value.get("claim_level") == "FORMAL_P_V2_TRAINING_INPUT_MANIFEST_ONLY"
        and value.get("logical_sha256") == _logical(value),
        "V95 fit manifest schema/status/hash drift",
    )
    fit_id = value.get("fit_id")
    fit_role = value.get("fit_role")
    outer = value.get("outer_fold")
    inner = value.get("inner_heldout_fold")
    _require(isinstance(fit_id, str) and fit_id, "V95 fit ID drift")
    _require(expected_fit_id is None or fit_id == expected_fit_id, "V95 fit ID mismatch")
    _require(fit_role in FIT_ROLES and type(outer) is int and 1 <= outer <= 4, "V95 fit role/fold drift")
    if fit_role == "INNER_TRAIN_FIT":
        _require(type(inner) is int and 1 <= inner <= 4 and inner != outer, "V95 inner fold drift")
    else:
        _require(inner is None, "V95 outer refit names an inner fold")
    _require(
        value.get("identity_disjoint") is True
        and value.get("supergroup_disjoint") is True
        and value.get("fresh_resume_exact_required") is True
        and value.get("independent_fit_validation_required") is True,
        "V95 disjointness/resume validation gate drift",
    )
    _require(
        value.get("training_authorized") is False
        and value.get("consumable_checkpoint_authorized") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False
        and value.get("next_authorized_stage") is None,
        "V95 input-only boundary drift",
    )
    _require(value.get("recipe") == _RECIPE, "V95 frozen recipe drift")
    source = value.get("source_bindings")
    _require(isinstance(source, Mapping) and set(source) == _SOURCE_BINDING_FIELDS, "V95 source binding field drift")
    source_values = {str(key): _sha(item, f"V95 source {key}") for key, item in source.items()}
    _require(source_values["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256, "V95 feature schema drift")

    raw_episodes = value.get("episodes")
    _require(isinstance(raw_episodes, list) and raw_episodes, "V95 episodes absent")
    _require(value.get("episode_population_sha256") == canonical_sha256(raw_episodes), "V95 episode population drift")
    neutral: list[NeutralEpisodeAddressV1] = []
    roles: list[TrainingRoleAddressV1] = []
    for index, raw in enumerate(raw_episodes):
        _require(isinstance(raw, Mapping) and set(raw) == _EPISODE_FIELDS, f"V95 episode {index} field drift")
        execution = raw["execution_ordinal"]
        _require(type(execution) is int and 0 <= execution < 600, f"V95 episode {index} execution drift")
        _require(isinstance(raw["query_id"], str) and raw["query_id"], f"V95 episode {index} query drift")
        address = NeutralEpisodeAddressV1(
            execution_ordinal=execution,
            query_id=raw["query_id"],
            query_source_image_sha256=_sha(raw["query_source_image_sha256"], f"episode {index} query source"),
            pair_address_sha256=_sha(raw["pair_address_sha256"], f"episode {index} pair"),
        )
        target_member, rival_member = raw["target_member_ordinal"], raw["rival_member_ordinal"]
        _require(
            type(target_member) is int
            and type(rival_member) is int
            and {target_member, rival_member} == {0, 1},
            f"V95 episode {index} role members drift",
        )
        target_key = _sha(raw["target_candidate_key"], f"episode {index} target candidate")
        rival_key = _sha(raw["rival_candidate_key"], f"episode {index} rival candidate")
        _require(target_key != rival_key, f"V95 episode {index} candidate alias")
        neutral.append(address)
        roles.append(
            TrainingRoleAddressV1(
                neutral=address,
                target_member_ordinal=target_member,
                target_candidate_key=target_key,
                rival_member_ordinal=rival_member,
                rival_candidate_key=rival_key,
            )
        )
    _require(
        tuple(item.execution_ordinal for item in neutral)
        == tuple(sorted({item.execution_ordinal for item in neutral})),
        "V95 episode execution order/uniqueness drift",
    )
    _require(
        value.get("train_query_count") == len(neutral)
        and value.get("eligible_episode_count") == len(neutral),
        "V95 episode count drift",
    )

    def routes(field: str, digest_field: str) -> tuple[ShardRouteV1, ...]:
        raw_rows = value.get(field)
        _require(isinstance(raw_rows, list) and raw_rows, f"V95 {field} absent")
        _require(value.get(digest_field) == canonical_sha256(raw_rows), f"V95 {field} hash drift")
        rows = tuple(_route(item, name=f"{field}[{index}]") for index, item in enumerate(raw_rows))
        _require(
            tuple(item.shard_ordinal for item in rows)
            == tuple(sorted({item.shard_ordinal for item in rows})),
            f"V95 {field} order/alias drift",
        )
        covered = [execution for item in rows for execution in item.selected_execution_ordinals]
        _require(
            sorted(covered) == sorted(item.execution_ordinal for item in neutral)
            and len(covered) == len(set(covered)),
            f"V95 {field} coverage drift",
        )
        return rows

    pair_routes = routes("pair_feature_shards", "pair_feature_shards_sha256")
    sidecar_routes = routes("structure_sidecar_shards", "structure_sidecar_shards_sha256")
    namespace = value.get("output_namespace")
    _require(
        namespace == f"results/dino_rcde_sr0_mt_p_v2_formal_fits_v1/{fit_id}",
        "V95 output namespace drift",
    )
    _sha(value.get("membership_file_sha256"), "V95 membership file")
    _sha(value.get("membership_logical_sha256"), "V95 membership logical")
    _require(isinstance(value.get("membership_path"), str) and fit_id in value["membership_path"], "V95 membership path drift")
    return ValidatedV95FitManifestV1(
        fit_id=fit_id,
        fit_role=fit_role,
        outer_fold=outer,
        inner_heldout_fold=inner,
        neutral_episodes=tuple(neutral),
        role_addresses=tuple(roles),
        pair_routes=pair_routes,
        sidecar_routes=sidecar_routes,
        output_namespace=namespace,
        manifest_logical_sha256=value["logical_sha256"],
        source_bindings=source_values,
    )


@dataclass(frozen=True)
class BoundShardPayloadV1:
    artifact_path: str
    artifact_file_sha256: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _sha(self.artifact_file_sha256, "bound shard file")
        _require(isinstance(self.payload, Mapping), "bound shard payload is not a mapping")


@dataclass(frozen=True)
class CompiledDirectionalInputV1:
    execution_ordinal: int
    member_ordinal: int
    direction: str
    candidate_key: str
    pair_address_sha256: str
    cache_sha256: str
    block_sha256: str
    input: RoleFreeCompactDirectionalInput
    action_table: CandidateActionFeatureTable
    regions: tuple[Any, ...]
    row_ready: tuple[bool, ...]
    row_weights: tuple[torch.Tensor, ...]
    row_hashes: tuple[str, ...]
    row_radii: tuple[int, ...]
    feature_tensor_sha256: str
    eligibility_tensor_sha256: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        _require(self.member_ordinal in (0, 1) and self.direction in P_DIRECTIONS, "compiled address drift")
        for value, name in (
            (self.candidate_key, "compiled candidate"),
            (self.pair_address_sha256, "compiled pair"),
            (self.cache_sha256, "compiled cache"),
            (self.block_sha256, "compiled block"),
            (self.feature_tensor_sha256, "compiled features"),
            (self.eligibility_tensor_sha256, "compiled eligibility"),
            (self.semantic_sha256, "compiled semantic"),
        ):
            _sha(value, name)


def _compiled_semantic_payload(value: CompiledDirectionalInputV1) -> dict[str, object]:
    structure = value.input.structure
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_ordinal": value.execution_ordinal,
        "member_ordinal": value.member_ordinal,
        "direction": value.direction,
        "candidate_key": value.candidate_key,
        "pair_address_sha256": value.pair_address_sha256,
        "cache_sha256": value.cache_sha256,
        "block_sha256": value.block_sha256,
        "structure_record_sha256": structure.record_sha256,
        "source_population_sha256": structure.source_population_sha256,
        "feature_tensor_sha256": value.feature_tensor_sha256,
        "eligibility_tensor_sha256": value.eligibility_tensor_sha256,
        "row_ready": list(value.row_ready),
        "row_weight_sha256": [tensor_sha256(item) for item in value.row_weights],
        "row_hashes": list(value.row_hashes),
        "row_radii": list(value.row_radii),
    }


def _make_compiled(
    *,
    neutral: NeutralEpisodeAddressV1,
    member: int,
    direction: str,
    cache: RoleFreePairFeatureCache,
    block: Any,
    sidecar: Mapping[str, Any],
    row_geometry_cache: dict[
        tuple[object, ...],
        tuple[
            tuple[Any, ...],
            tuple[bool, ...],
            tuple[torch.Tensor, ...],
            tuple[str, ...],
            tuple[int, ...],
        ],
    ],
) -> CompiledDirectionalInputV1:
    start = block.feature_offset
    stop = start + block.coordinate_count
    features_flat = cache.features[start:stop]
    eligible_flat = cache.eligibility[start:stop]
    features = features_flat.reshape(block.root_count, block.action_count, FEATURE_DIM)
    eligibility = eligible_flat.reshape(block.root_count, block.action_count)
    _require(
        sidecar.get("query_id") == neutral.query_id
        and sidecar.get("query_source_image_sha256") == neutral.query_source_image_sha256
        and sidecar.get("execution_ordinal") == neutral.execution_ordinal
        and sidecar.get("pair_address_sha256") == neutral.pair_address_sha256
        and sidecar.get("member_ordinal") == member
        and sidecar.get("candidate_key") == block.candidate_key
        and sidecar.get("direction") == direction
        and sidecar.get("source_population_sha256") == block.source_population_sha256
        and sidecar.get("root_count") == block.root_count
        and sidecar.get("action_count") == block.action_count
        and sidecar.get("coordinate_count") == block.coordinate_count
        and sidecar.get("cache_sha256") == cache.cache_sha256
        and sidecar.get("cache_block_sha256") == block.block_sha256,
        "pair block/sidecar address or population drift",
    )
    _require(
        sidecar.get("cache_feature_tensor_sha256") == tensor_sha256(features)
        and sidecar.get("cache_eligibility_sha256") == tensor_sha256(eligibility),
        "pair block/sidecar tensor hash drift",
    )
    structure = structure_from_sidecar_record(sidecar)
    _require(torch.equal(structure.eligible, eligibility), "pair/sidecar eligibility value drift")
    _require(bool(eligibility.all()), "formal V95 input contains a geometry-H0 coordinate")
    role_free = RoleFreeCompactDirectionalInput(structure=structure, features=features)
    table = CandidateActionFeatureTable(
        candidate_key=structure.candidate_key,
        root_ordinals=tuple(range(structure.root_count)),
        action_keys_by_root=structure.action_keys,
        features_by_root=tuple(role_free.features[root] for root in range(structure.root_count)),
        eligible_by_root=tuple(structure.eligible[root] for root in range(structure.root_count)),
    )
    root_available = tuple(bool(structure.eligible[root].any()) for root in range(structure.root_count))
    query_mask_population_sha256 = sidecar.get(
        "deployment_query_root_masks_sha256"
    )
    _sha(query_mask_population_sha256, "compiled query-mask population")
    geometry_key = (
        structure.query_id,
        structure.query_geometry_sha256,
        structure.colnomic_query_grid_shape,
        structure.deployment_query_grid_shape,
        query_mask_population_sha256,
        root_available,
    )
    cached_geometry = row_geometry_cache.get(geometry_key)
    if cached_geometry is None:
        regions = tuple(
            enumerate_superregion_bank(
                structure.colnomic_query_grid_shape, include_r0_control=False
            )
        )
        query_masks = tuple(
            item.decode() for item in structure.deployment_query_root_masks
        )
        row_ready_list: list[bool] = []
        row_weight_list: list[torch.Tensor] = []
        for region in regions:
            selected_masks = [
                query_masks[root]
                for root in region.contributing_root_ordinals
                if root_available[root]
            ]
            legal = len(selected_masks) >= MIN_READY_ROOTS
            if legal:
                union = torch.stack(selected_masks).any(dim=0)
                legal = _component_is_legal(
                    union, structure.deployment_query_grid_shape
                ) and int(union.sum()) > max(
                    int(mask.sum()) for mask in selected_masks
                )
            row_ready_list.append(bool(legal))
            row_weight_list.append(
                region.aggregation_weights.detach()
                .to(torch.float64)
                .cpu()
                .contiguous()
            )
        cached_geometry = (
            regions,
            tuple(row_ready_list),
            tuple(row_weight_list),
            tuple(superregion_sha256(region) for region in regions),
            tuple(region.radius for region in regions),
        )
        row_geometry_cache[geometry_key] = cached_geometry
    regions, row_ready, row_weights, row_hashes, row_radii = cached_geometry
    provisional = object.__new__(CompiledDirectionalInputV1)
    for name, item in {
        "execution_ordinal": neutral.execution_ordinal,
        "member_ordinal": member,
        "direction": direction,
        "candidate_key": block.candidate_key,
        "pair_address_sha256": neutral.pair_address_sha256,
        "cache_sha256": cache.cache_sha256,
        "block_sha256": block.block_sha256,
        "input": role_free,
        "action_table": table,
        "regions": regions,
        "row_ready": row_ready,
        "row_weights": row_weights,
        "row_hashes": row_hashes,
        "row_radii": row_radii,
        "feature_tensor_sha256": tensor_sha256(role_free.features),
        "eligibility_tensor_sha256": tensor_sha256(structure.eligible),
    }.items():
        object.__setattr__(provisional, name, item)
    semantic = canonical_sha256(_compiled_semantic_payload(provisional))
    return CompiledDirectionalInputV1(
        execution_ordinal=neutral.execution_ordinal,
        member_ordinal=member,
        direction=direction,
        candidate_key=block.candidate_key,
        pair_address_sha256=neutral.pair_address_sha256,
        cache_sha256=cache.cache_sha256,
        block_sha256=block.block_sha256,
        input=role_free,
        action_table=table,
        regions=regions,
        row_ready=row_ready,
        row_weights=row_weights,
        row_hashes=row_hashes,
        row_radii=row_radii,
        feature_tensor_sha256=tensor_sha256(role_free.features),
        eligibility_tensor_sha256=tensor_sha256(structure.eligible),
        semantic_sha256=semantic,
    )


@dataclass(frozen=True)
class SealedTargetFreeInputTableV1:
    fit_id: str
    neutral_population_sha256: str
    inputs: Mapping[tuple[int, int, str], CompiledDirectionalInputV1]
    table_sha256: str
    geometry_h0_coordinate_count: int = 0
    role_join_count: int = 0

    def __post_init__(self) -> None:
        _require(
            self.role_join_count == 0 and self.geometry_h0_coordinate_count == 0,
            "target-free table role/H0 seal drift",
        )
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


def _table_payload(
    fit_id: str,
    neutral_population_sha256: str,
    inputs: Mapping[tuple[int, int, str], CompiledDirectionalInputV1],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fit_id": fit_id,
        "neutral_population_sha256": neutral_population_sha256,
        "rows": [
            {
                "execution_ordinal": key[0],
                "member_ordinal": key[1],
                "direction": key[2],
                "semantic_sha256": inputs[key].semantic_sha256,
            }
            for key in sorted(inputs)
        ],
        "role_fields_serialized": False,
        "geometry_h0_coordinate_count": 0,
        "role_join_count": 0,
    }


def _payload_by_path(
    values: Sequence[BoundShardPayloadV1], routes: Sequence[ShardRouteV1], *, name: str
) -> dict[str, Mapping[str, Any]]:
    supplied = {item.artifact_path: item for item in values}
    expected = {item.artifact_path: item for item in routes}
    _require(len(supplied) == len(values) and set(supplied) == set(expected), f"{name} shard path population drift")
    output: dict[str, Mapping[str, Any]] = {}
    for path, route in expected.items():
        item = supplied[path]
        _require(item.artifact_file_sha256 == route.artifact_sha256, f"{name} shard file hash drift")
        output[path] = item.payload
    return output


def build_and_seal_target_free_input_table(
    manifest: ValidatedV95FitManifestV1,
    *,
    pair_shards: Sequence[BoundShardPayloadV1],
    sidecar_shards: Sequence[BoundShardPayloadV1],
) -> SealedTargetFreeInputTableV1:
    """Build all 2 x 2 inputs without consulting target/rival roles."""

    pair_payloads = _payload_by_path(pair_shards, manifest.pair_routes, name="pair")
    side_payloads = _payload_by_path(sidecar_shards, manifest.sidecar_routes, name="sidecar")
    neutral_by_execution = {item.execution_ordinal: item for item in manifest.neutral_episodes}
    pair_records: dict[int, tuple[NeutralEpisodeAddressV1, RoleFreePairFeatureCache]] = {}
    for route in manifest.pair_routes:
        payload = pair_payloads[route.artifact_path]
        records = payload.get("records")
        summaries = payload.get("record_summaries")
        _require(
            payload.get("schema_version") == PAIR_SHARD_SCHEMA
            and payload.get("status") == PAIR_SHARD_STATUS
            and payload.get("role_free") is True
            and payload.get("target_free") is True
            and payload.get("synthetic") is False
            and payload.get("execution_start") == route.execution_start
            and payload.get("execution_stop") == route.execution_stop
            and isinstance(records, list)
            and isinstance(summaries, list)
            and payload.get("logical_sha256")
            == canonical_sha256({key: item for key, item in payload.items() if key not in {"records", "logical_sha256"}})
            and payload.get("record_population_sha256") == canonical_sha256(summaries),
            "pair shard envelope drift",
        )
        record_index = {int(item["execution_ordinal"]): item for item in records if isinstance(item, Mapping)}
        _require(len(record_index) == len(records), "pair shard execution alias")
        for execution in route.selected_execution_ordinals:
            raw = record_index.get(execution)
            neutral = neutral_by_execution[execution]
            _require(
                isinstance(raw, Mapping)
                and set(raw) == {"query_id", "execution_ordinal", "pair_address_sha256", "pair_feature_cache"}
                and raw.get("query_id") == neutral.query_id
                and raw.get("pair_address_sha256") == neutral.pair_address_sha256,
                "pair record/neutral address drift",
            )
            cache = deserialize_role_free_pair_feature_cache(raw["pair_feature_cache"])
            _require(execution not in pair_records, "pair record duplicated across shards")
            pair_records[execution] = (neutral, cache)

    side_records: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for route in manifest.sidecar_routes:
        payload = side_payloads[route.artifact_path]
        records = payload.get("records")
        _require(
            payload.get("schema_version") == SIDECAR_SHARD_SCHEMA
            and payload.get("status") == SIDECAR_SHARD_STATUS
            and payload.get("target_free") is True
            and payload.get("execution_start") == route.execution_start
            and payload.get("execution_stop") == route.execution_stop
            and isinstance(records, list)
            and payload.get("logical_sha256") == _logical(payload)
            and payload.get("record_population_sha256") == canonical_sha256(records),
            "sidecar shard envelope drift",
        )
        selected = set(route.selected_execution_ordinals)
        for raw in records:
            _require(isinstance(raw, Mapping), "sidecar record is not a mapping")
            execution = raw.get("execution_ordinal")
            if execution not in selected:
                continue
            key = (execution, raw.get("member_ordinal"), raw.get("direction"))
            _require(
                type(key[0]) is int
                and key[1] in (0, 1)
                and key[2] in P_DIRECTIONS
                and key not in side_records,
                "sidecar member/direction address drift",
            )
            side_records[key] = raw

    _require(set(pair_records) == set(neutral_by_execution), "pair neutral population incomplete")
    expected_keys = {
        (execution, member, direction)
        for execution in neutral_by_execution
        for member in (0, 1)
        for direction in P_DIRECTIONS
    }
    _require(set(side_records) == expected_keys, "sidecar neutral population incomplete")
    inputs: dict[tuple[int, int, str], CompiledDirectionalInputV1] = {}
    row_geometry_cache: dict[
        tuple[object, ...],
        tuple[
            tuple[Any, ...],
            tuple[bool, ...],
            tuple[torch.Tensor, ...],
            tuple[str, ...],
            tuple[int, ...],
        ],
    ] = {}
    for execution in sorted(pair_records):
        neutral, cache = pair_records[execution]
        by_address = {(item.member_ordinal, item.direction): item for item in cache.blocks}
        _require(set(by_address) == {(member, direction) for member in (0, 1) for direction in P_DIRECTIONS}, "pair block population incomplete")
        for member in (0, 1):
            for direction in P_DIRECTIONS:
                block = by_address[(member, direction)]
                inputs[(execution, member, direction)] = _make_compiled(
                    neutral=neutral,
                    member=member,
                    direction=direction,
                    cache=cache,
                    block=block,
                    sidecar=side_records[(execution, member, direction)],
                    row_geometry_cache=row_geometry_cache,
                )
    _require(
        len(row_geometry_cache) == len(neutral_by_execution),
        "query geometry was not candidate/direction invariant",
    )
    neutral_sha = canonical_sha256([item.payload() for item in manifest.neutral_episodes])
    table_sha = canonical_sha256(_table_payload(manifest.fit_id, neutral_sha, inputs))
    return SealedTargetFreeInputTableV1(
        fit_id=manifest.fit_id,
        neutral_population_sha256=neutral_sha,
        inputs=inputs,
        table_sha256=table_sha,
        geometry_h0_coordinate_count=0,
    )


@dataclass(frozen=True)
class JoinedTrainingEpisodeV1:
    neutral: NeutralEpisodeAddressV1
    target: tuple[CompiledDirectionalInputV1, CompiledDirectionalInputV1]
    rival: tuple[CompiledDirectionalInputV1, CompiledDirectionalInputV1]


@dataclass(frozen=True)
class SealedTrainingEpisodePopulationV1:
    fit_id: str
    target_free_table_sha256: str
    episodes: tuple[JoinedTrainingEpisodeV1, ...]
    role_join_sha256: str


def join_training_roles_after_seal(
    table: SealedTargetFreeInputTableV1,
    manifest: ValidatedV95FitManifestV1,
) -> SealedTrainingEpisodePopulationV1:
    """Join labels only after the complete target-free table is sealed."""

    _require(table.fit_id == manifest.fit_id and table.role_join_count == 0, "role join/table state drift")
    expected_table = canonical_sha256(
        _table_payload(table.fit_id, table.neutral_population_sha256, table.inputs)
    )
    _require(table.table_sha256 == expected_table, "target-free table seal drift")
    joined = []
    role_receipts = []
    for role in manifest.role_addresses:
        execution = role.neutral.execution_ordinal
        target = tuple(table.inputs[(execution, role.target_member_ordinal, direction)] for direction in P_DIRECTIONS)
        rival = tuple(table.inputs[(execution, role.rival_member_ordinal, direction)] for direction in P_DIRECTIONS)
        _require(
            all(item.candidate_key == role.target_candidate_key for item in target)
            and all(item.candidate_key == role.rival_candidate_key for item in rival),
            "post-seal target/rival candidate binding drift",
        )
        joined.append(JoinedTrainingEpisodeV1(role.neutral, target, rival))
        role_receipts.append(
            {
                "execution_ordinal": execution,
                "target_member_ordinal": role.target_member_ordinal,
                "target_candidate_key": role.target_candidate_key,
                "rival_member_ordinal": role.rival_member_ordinal,
                "rival_candidate_key": role.rival_candidate_key,
            }
        )
    return SealedTrainingEpisodePopulationV1(
        fit_id=manifest.fit_id,
        target_free_table_sha256=table.table_sha256,
        episodes=tuple(joined),
        role_join_sha256=canonical_sha256(
            {"target_free_table_sha256": table.table_sha256, "roles": role_receipts}
        ),
    )


def _exact_zero(model: SharedMultitilePHead) -> torch.Tensor:
    return model.linear.weight.square().sum() * 0.0


def score_compiled_direction_v1(
    model: SharedMultitilePHead, value: CompiledDirectionalInputV1
) -> RoleFreeCompactDirectionOutput:
    """Score a precompiled input with exact role-free-scorer semantics."""

    _require(isinstance(model, SharedMultitilePHead), "compiled scorer model drift")
    _require(
        value.semantic_sha256 == canonical_sha256(_compiled_semantic_payload(value))
        and value.feature_tensor_sha256 == tensor_sha256(value.input.features)
        and value.eligibility_tensor_sha256 == tensor_sha256(value.input.structure.eligible),
        "compiled input seal drift",
    )
    selected_indices: list[int | None] = []
    selected_keys: list[str | None] = []
    root_scores: list[torch.Tensor] = []
    for keys, features, eligible in zip(
        value.action_table.action_keys_by_root,
        value.action_table.features_by_root,
        value.action_table.eligible_by_root,
        strict=True,
    ):
        logits = model(features)
        index = canonical_action_argmax(logits, keys, eligible.to(logits.device))
        selected_indices.append(index)
        selected_keys.append(None if index is None else keys[index])
        root_scores.append(_exact_zero(model) if index is None else logits[index])
    root_tensor = torch.stack(root_scores)
    # Root availability and row geometry are immutable input facts.  They are
    # compiled once above; every update therefore performs only the registered
    # per-row dot product, with no mask stacking or connected-component pass.
    _require(
        tuple(index is not None for index in selected_indices)
        == tuple(bool(item.any()) for item in value.input.structure.eligible),
        "compiled root availability drift",
    )
    row_scores: list[torch.Tensor] = []
    for legal, stored_weights in zip(value.row_ready, value.row_weights, strict=True):
        if legal:
            weights = stored_weights.to(
                device=root_tensor.device, dtype=root_tensor.dtype
            )
            row_scores.append(torch.dot(weights, root_tensor))
        else:
            row_scores.append(_exact_zero(model))
    scores = torch.stack(row_scores)
    ready = torch.tensor(value.row_ready, dtype=torch.bool, device=scores.device)
    utility = scale_balanced_hierarchical_logmeanexp(scores, value.row_radii)
    legal_rows = torch.nonzero(ready, as_tuple=False).flatten().tolist()
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
            map_index = min(tied, key=lambda index: value.row_hashes[index])
    return RoleFreeCompactDirectionOutput(
        compact_input=value.input,
        selected_action_indices=tuple(selected_indices),
        selected_action_keys=tuple(selected_keys),
        root_scores=root_tensor,
        row_hashes=value.row_hashes,
        row_radii=value.row_radii,
        row_scores=scores,
        row_ready=ready,
        candidate_utility=utility,
        map_row_index=map_index,
        map_row_sha256=None if map_index is None else value.row_hashes[map_index],
    )


def assert_compiled_role_free_exact_parity(
    model: SharedMultitilePHead, value: CompiledDirectionalInputV1
) -> None:
    """Fail closed unless the compiled and qualified scorers are exact."""

    compiled = score_compiled_direction_v1(model, value)
    reference = score_role_free_compact_direction(model, value.input)
    _require(compiled.selected_action_indices == reference.selected_action_indices, "compiled selected-action index parity drift")
    _require(compiled.selected_action_keys == reference.selected_action_keys, "compiled selected-action key parity drift")
    for name in ("root_scores", "row_scores", "row_ready", "candidate_utility"):
        _require(torch.equal(getattr(compiled, name), getattr(reference, name)), f"compiled {name} parity drift")
    _require(
        compiled.map_row_index == reference.map_row_index
        and compiled.map_row_sha256 == reference.map_row_sha256,
        "compiled MAP diagnostic parity drift",
    )


__all__ = [
    "SCHEMA_VERSION",
    "V95_MANIFEST_SCHEMA",
    "V95_MANIFEST_STATUS",
    "PFitV2RuntimeError",
    "NeutralEpisodeAddressV1",
    "TrainingRoleAddressV1",
    "ShardRouteV1",
    "ValidatedV95FitManifestV1",
    "BoundShardPayloadV1",
    "CompiledDirectionalInputV1",
    "SealedTargetFreeInputTableV1",
    "JoinedTrainingEpisodeV1",
    "SealedTrainingEpisodePopulationV1",
    "validate_v95_fit_manifest",
    "build_and_seal_target_free_input_table",
    "join_training_roles_after_seal",
    "score_compiled_direction_v1",
    "assert_compiled_role_free_exact_parity",
]
