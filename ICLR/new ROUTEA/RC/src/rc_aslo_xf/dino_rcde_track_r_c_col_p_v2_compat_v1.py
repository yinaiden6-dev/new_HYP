"""Strict C_COL_P to P-V2 compatibility core for Track R.

This append-only module keeps the registered full-C128 ColNomic donor
intervention, but deliberately replaces the obsolete V1 P-lock serializer with
the current P-V2 path::

    donor-bound ColNomic features
      -> SharedMultitilePHead loaded from a P-V2 ``model_state``
      -> legacy/scalar direction scorer (the sealed feature oracle)
      -> score-erased fanout decision
      -> sign-independent Natural/Core P-V2 lock builder

The scalar scorer's V1 MAP/H0 bit is never consumed.  Only its selected
reference actions and complete row-score vector enter ``build_natural_p_lock_v2``.
In particular, a donor query root remains present when reference-mask transport
leaves no legal action.  The P-V2 record therefore emits
``ROOT_REFERENCE_MISSING`` instead of silently converting the root to V1 H0.

There is no file I/O, V model, target/rival role, result reduction, or automatic
stage advance here.  A separately authorised natural runner must bind physical
artifacts and independently replay the returned records.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_sr0_p_v1 import SharedMultitilePHead
from .dino_rcde_sr0_mt_c_col_p_v1 import (
    build_control_feature_records,
    canonical_donor_positions,
    donor_receipt,
)
from .dino_rcde_sr0_mt_p_compact_catalog_v1 import CompactMask
from .dino_rcde_sr0_mt_p_compact_lock_fanout_v1 import (
    FanoutDirectionDecision,
    decision_from_legacy_output,
)
from .dino_rcde_sr0_mt_p_lock_v2 import (
    ROOT_QUERY_UNMAPPABLE,
    ROOT_READY,
    ROOT_REFERENCE_MISSING,
    candidate_p_lock_v2_from_record,
)
from .dino_rcde_sr0_mt_p_natural_adapter_v2 import (
    build_natural_p_lock_v2,
    validate_natural_p_lock_v2,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    RAW_SOURCE_SCHEMA,
    DirectionalFeatureRecord,
    assert_no_forbidden_prejoin_keys,
    canonical_sha256,
    score_deployable_direction,
    state_sha256,
    tensor_sha256,
)


SCHEMA_VERSION = "rc_dino_rcde_track_r_c_col_p_v2_compat_v1_20260824"
NAMESPACE = "DINO_RCDE_TRACK_R_C_COL_P_STRICT_P_V2_COMPAT_V1"
P_V2_CHECKPOINT_SCHEMA = "rc_dino_rcde_sr0_mt_p_v2_checkpoint_v1_20260818"
P_V2_CHECKPOINT_STATUS = "RCDE_SR0_MT_P_V2_COMPACT_CHECKPOINT_READY"
P_V2_CHECKPOINT_CLAIM = "ENGINEERING_P_V2_EXACT_RESUME_STATE_ONLY"
P_V2_COMPLETED_UPDATES = 2048

P_V2_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "fit_id",
        "fit_index",
        "path_kind",
        "completed_updates",
        "consumable_checkpoint_authorized",
        "bindings",
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "scheduler_cursor",
        "current_learning_rate",
        "rolling_trace_state",
        "state_sha256",
        "logical_sha256",
        "restored_from_checkpoint_file_sha256",
        "restored_from_checkpoint_state_sha256",
    }
)
P_V2_CHECKPOINT_STATE_FIELDS = frozenset(
    {
        "completed_updates",
        "current_learning_rate",
        "scheduler_cursor",
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "rolling_trace_state",
    }
)
P_V2_CHECKPOINT_BINDING_FIELDS = frozenset(
    {
        "authority_file_sha256",
        "authority_logical_sha256",
        "contract_file_sha256",
        "parent_authority_v95_file_sha256",
        "parent_authority_v95_logical_sha256",
        "v95_index_file_sha256",
        "v95_index_logical_sha256",
        "v95_validation_file_sha256",
        "v95_validation_logical_sha256",
        "fit_manifest_file_sha256",
        "fit_manifest_logical_sha256",
        "loss_join_file_sha256",
        "loss_join_logical_sha256",
        "loss_join_validation_file_sha256",
        "loss_join_validation_logical_sha256",
        "target_free_table_sha256",
        "role_join_sha256",
        "ordered_pair_shards_sha256",
        "ordered_sidecar_shards_sha256",
        "p_head_file_sha256",
        "role_free_scorer_file_sha256",
        "p_lock_core_file_sha256",
        "p_selector_file_sha256",
        "p_v2_adapter_file_sha256",
        "fit_runtime_file_sha256",
        "runner_file_sha256",
        "validator_file_sha256",
        "recipe_sha256",
    }
)


class StrictCColPV2CompatError(RuntimeError):
    """The C_COL_P intervention cannot close against strict P-V2."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise StrictCColPV2CompatError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value  # type: ignore[return-value]


def _sequence(value: object, name: str) -> Sequence[Any]:
    _require(isinstance(value, (list, tuple)), f"{name} must be a sequence")
    return value  # type: ignore[return-value]


def _sha(value: object, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} SHA256 drift",
    )
    return value


def _grid(value: object, name: str) -> tuple[int, int]:
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(item) is int and item > 0 for item in value),
        f"{name} grid drift",
    )
    return int(value[0]), int(value[1])  # type: ignore[index]


def p_v2_checkpoint_state_sha256(value: Mapping[str, Any]) -> str:
    """Recompute the frozen P-V2 checkpoint state digest."""

    _require(
        P_V2_CHECKPOINT_STATE_FIELDS.issubset(value),
        "P-V2 checkpoint state fields are incomplete",
    )
    return state_sha256(
        {key: value[key] for key in P_V2_CHECKPOINT_STATE_FIELDS}
    )


def p_v2_checkpoint_logical_sha256(value: Mapping[str, Any]) -> str:
    """Recompute the frozen P-V2 checkpoint logical digest."""

    excluded = {
        "logical_sha256",
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "rolling_trace_state",
    }
    payload = {key: item for key, item in value.items() if key not in excluded}
    for name in (
        "model_state",
        "optimizer_state",
        "sampler_state",
        "rng_state",
        "rolling_trace_state",
    ):
        _require(name in value, f"P-V2 checkpoint {name} is absent")
    payload.update(
        {
            "model_state_sha256": state_sha256(value["model_state"]),
            "optimizer_state_sha256": state_sha256(value["optimizer_state"]),
            "sampler_state_sha256": state_sha256(value["sampler_state"]),
            "rng_state_sha256": state_sha256(value["rng_state"]),
            "rolling_trace_state_sha256": canonical_sha256(
                value["rolling_trace_state"]
            ),
        }
    )
    return canonical_sha256(payload)


def load_strict_outer_refit_p_v2_model(
    checkpoint: Mapping[str, Any],
    *,
    source_fold: int,
    p_training_manifest_sha256: str,
) -> SharedMultitilePHead:
    """Validate a complete P-V2 checkpoint envelope and load ``model_state``.

    Physical checkpoint and manifest hashes remain the natural runner's
    responsibility.  This pure boundary proves that old V1 ``checkpoint_model``
    is not used and that the supplied mapping is the exact outer-refit state.
    """

    _require(source_fold in (1, 2, 3, 4), "P-V2 source fold drift")
    manifest_sha = _sha(p_training_manifest_sha256, "P training manifest")
    fit_id = f"P_OUTER{source_fold}_OUTER_REFIT"
    bindings = _mapping(checkpoint.get("bindings"), "P-V2 checkpoint bindings")
    _require(
        set(checkpoint) == P_V2_CHECKPOINT_FIELDS
        and checkpoint.get("schema_version") == P_V2_CHECKPOINT_SCHEMA
        and checkpoint.get("status") == P_V2_CHECKPOINT_STATUS
        and checkpoint.get("claim_level") == P_V2_CHECKPOINT_CLAIM
        and checkpoint.get("fit_id") == fit_id
        and checkpoint.get("fit_index") == source_fold * 4 - 1
        and checkpoint.get("path_kind") == "PRIMARY"
        and checkpoint.get("completed_updates") == P_V2_COMPLETED_UPDATES
        and checkpoint.get("scheduler_cursor") == P_V2_COMPLETED_UPDATES
        and checkpoint.get("consumable_checkpoint_authorized") is False
        and set(bindings) == P_V2_CHECKPOINT_BINDING_FIELDS
        and bindings.get("fit_manifest_file_sha256") == manifest_sha,
        "P-V2 outer-refit checkpoint envelope/binding drift",
    )
    for key, item in bindings.items():
        _sha(item, f"P-V2 checkpoint binding {key}")
    _require(
        checkpoint.get("state_sha256") == p_v2_checkpoint_state_sha256(checkpoint),
        "P-V2 checkpoint state hash drift",
    )
    _require(
        checkpoint.get("logical_sha256")
        == p_v2_checkpoint_logical_sha256(checkpoint),
        "P-V2 checkpoint logical hash drift",
    )
    model_state = _mapping(checkpoint.get("model_state"), "P-V2 model state")
    model = SharedMultitilePHead()
    try:
        model.load_state_dict(model_state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise StrictCColPV2CompatError("P-V2 model_state load drift") from error
    _require(
        state_sha256(model.state_dict()) == state_sha256(model_state),
        "P-V2 loaded model state differs from checkpoint",
    )
    model.eval()
    return model


def _select_raw_query(
    raw_source: Mapping[str, Any], query_id: str
) -> Mapping[str, Any]:
    _require(
        raw_source.get("schema_version") == RAW_SOURCE_SCHEMA
        and raw_source.get("target_free") is True,
        "C_COL strict raw source schema/role drift",
    )
    assert_no_forbidden_prejoin_keys(raw_source)
    selected = [
        _mapping(item, "raw P query")
        for item in _sequence(raw_source.get("queries"), "raw P queries")
        if isinstance(item, Mapping) and item.get("query_id") == query_id
    ]
    _require(len(selected) == 1, "C_COL strict query is absent or duplicated")
    return selected[0]


def _geometry_valid_mask(
    record: Mapping[str, Any], *, expected_grid: tuple[int, int], name: str
) -> torch.Tensor:
    geometry = _mapping(record.get("dino_geometry"), f"{name} DINO geometry")
    _require(
        _grid(geometry.get("grid_shape"), f"{name} DINO geometry") == expected_grid,
        f"{name} DINO geometry grid drift",
    )
    mask = (
        torch.as_tensor(geometry.get("valid_patch_mask"), dtype=torch.bool)
        .detach()
        .cpu()
        .contiguous()
        .flatten()
    )
    _require(
        mask.shape == (math.prod(expected_grid),) and bool(mask.any()),
        f"{name} DINO valid mask drift",
    )
    return mask


def _bool_rle(value: torch.Tensor) -> dict[str, Any]:
    tensor = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous()
    flat = tensor.flatten()
    runs: list[list[int]] = []
    start: int | None = None
    for index, active in enumerate([*flat.tolist(), False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append([start, index - start])
            start = None
    return {
        "shape": list(tensor.shape),
        "flatten_order": "ROW_MAJOR",
        "true_runs": runs,
        "tensor_sha256": tensor_sha256(tensor),
    }


def _mask_table(
    values: Sequence[torch.Tensor], *, grid_shape: tuple[int, int]
) -> tuple[list[dict[str, object]], list[int]]:
    table: list[torch.Tensor] = []
    payloads: list[dict[str, object]] = []
    by_hash: dict[str, int] = {}
    indices: list[int] = []
    for raw in values:
        mask = (
            torch.as_tensor(raw, dtype=torch.bool)
            .detach()
            .cpu()
            .contiguous()
            .flatten()
        )
        _require(mask.shape == (math.prod(grid_shape),), "strict mask table grid drift")
        digest = tensor_sha256(mask)
        if digest in by_hash:
            _require(torch.equal(table[by_hash[digest]], mask), "strict mask hash alias")
        else:
            by_hash[digest] = len(table)
            table.append(mask)
            payloads.append(CompactMask.from_tensor(mask, grid_shape=grid_shape).payload())
        indices.append(by_hash[digest])
    return payloads, indices


def _donor_query_root_masks(
    donor: Mapping[str, Any],
    *,
    direction: str,
    query_grid_shape: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    directions = _mapping(donor.get("directions"), "C_COL donor directions")
    roots = _sequence(directions.get(direction), "C_COL donor root bank")
    output: list[torch.Tensor] = []
    for expected_root, raw_root in enumerate(roots):
        root = _mapping(raw_root, "C_COL donor root")
        _require(root.get("root_ordinal") == expected_root, "C_COL donor root order drift")
        actions = _sequence(root.get("actions"), "C_COL donor actions")
        _require(bool(actions), "C_COL donor root has no actions")
        masks = [
            torch.as_tensor(
                _mapping(action, "C_COL donor action").get("deployment_query_mask"),
                dtype=torch.bool,
            )
            .detach()
            .cpu()
            .contiguous()
            .flatten()
            for action in actions
        ]
        _require(
            all(mask.shape == (math.prod(query_grid_shape),) for mask in masks),
            "C_COL donor query-root grid drift",
        )
        nonempty = [mask for mask in masks if bool(mask.any())]
        if nonempty:
            chosen = nonempty[0]
            _require(
                all(torch.equal(chosen, mask) for mask in nonempty),
                "C_COL donor query root changed across reference actions",
            )
        else:
            chosen = torch.zeros(math.prod(query_grid_shape), dtype=torch.bool)
        output.append(chosen.clone())
    return tuple(output)


def _strict_source_record(
    *,
    raw_query: Mapping[str, Any],
    destination: Mapping[str, Any],
    donor: Mapping[str, Any],
    feature_record: DirectionalFeatureRecord,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize one control direction while preserving donor query roots."""

    query_grid = _grid(raw_query.get("dino_query_grid_shape"), "strict query DINO")
    reference_grid = _grid(
        destination.get("dino_reference_grid_shape"), "strict destination DINO"
    )
    destination_query_masks = _donor_query_root_masks(
        destination, direction=feature_record.direction, query_grid_shape=query_grid
    )
    donor_query_masks = _donor_query_root_masks(
        donor, direction=feature_record.direction, query_grid_shape=query_grid
    )
    root_count = len(feature_record.deployments_by_root)
    _require(
        len(destination_query_masks) == len(donor_query_masks) == root_count,
        "strict donor query-root population drift",
    )
    _require(
        all(
            torch.equal(destination_mask, donor_mask)
            for destination_mask, donor_mask in zip(
                destination_query_masks, donor_query_masks, strict=True
            )
        ),
        "C_COL query structure is not candidate-invariant",
    )
    action_counts = tuple(len(root) for root in feature_record.deployments_by_root)
    _require(
        bool(action_counts) and len(set(action_counts)) == 1 and action_counts[0] > 0,
        "strict C_COL action bank is not rectangular",
    )
    action_count = action_counts[0]
    eligibility = torch.stack(feature_record.action_table.eligible_by_root).to(torch.bool)
    _require(
        eligibility.shape == (root_count, action_count),
        "strict C_COL eligibility shape drift",
    )
    reference_masks = [
        deployment.reference_mask
        for root in feature_record.deployments_by_root
        for deployment in root
    ]
    reference_table, flat_reference_indices = _mask_table(
        reference_masks, grid_shape=reference_grid
    )
    reference_indices = [
        flat_reference_indices[root * action_count : (root + 1) * action_count]
        for root in range(root_count)
    ]
    query_payloads = [
        CompactMask.from_tensor(mask, grid_shape=query_grid).payload()
        for mask in destination_query_masks
    ]
    action_keys = [list(row) for row in feature_record.action_table.action_keys_by_root]
    donor_tokens = torch.as_tensor(donor.get("reference_tokens")).detach().cpu().contiguous()
    donor_valid = (
        torch.as_tensor(donor.get("reference_valid_patch_mask"), dtype=torch.bool)
        .detach()
        .cpu()
        .contiguous()
        .flatten()
    )
    source_population = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "query_id": feature_record.query_id,
        "destination_candidate_key": feature_record.candidate_key,
        "donor_candidate_key": donor.get("candidate_key"),
        "direction": feature_record.direction,
        "donor_reference_tokens_sha256": tensor_sha256(donor_tokens),
        "donor_reference_valid_mask_sha256": tensor_sha256(donor_valid),
        "query_root_mask_sha256": [
            tensor_sha256(item) for item in destination_query_masks
        ],
        "reference_mask_table_sha256": [
            item["tensor_sha256"] for item in reference_table
        ],
        "reference_mask_index": reference_indices,
        "action_keys": action_keys,
        "eligibility_sha256": tensor_sha256(eligibility),
    }
    cache_sha = canonical_sha256(source_population)
    source_record = {
        "query_id": feature_record.query_id,
        "historical_query_ordinal": feature_record.historical_query_ordinal,
        "execution_ordinal": feature_record.execution_ordinal,
        "query_source_image_sha256": feature_record.query_source_image_sha256,
        "candidate_axis_sha256": raw_query.get("candidate_axis_sha256"),
        "candidate_count_per_query": len(
            _sequence(raw_query.get("candidates"), "strict candidate axis")
        ),
        "candidate_position": feature_record.candidate_position,
        "candidate_key": feature_record.candidate_key,
        "candidate_physical_row": feature_record.candidate_physical_row,
        # This remains the destination DINO/reference address.  Donor content is
        # recorded separately in the control source-population receipt above.
        "candidate_reference_source_sha256": (
            feature_record.candidate_reference_source_sha256
        ),
        "direction": feature_record.direction,
        "query_geometry_sha256": feature_record.query_geometry_sha256,
        "reference_geometry_sha256": feature_record.reference_geometry_sha256,
        "deployment_query_grid_shape": list(query_grid),
        "deployment_reference_grid_shape": list(reference_grid),
        "colnomic_query_grid_shape": list(feature_record.colnomic_query_grid_shape),
        "root_count": root_count,
        "action_count": action_count,
        "deployment_query_root_masks": query_payloads,
        "deployment_reference_mask_table": reference_table,
        "deployment_reference_mask_index": reference_indices,
        "action_keys": action_keys,
        "eligibility_rle": _bool_rle(eligibility),
        "cache_sha256": cache_sha,
    }
    return source_record, source_population


def _assert_v2_root_semantics(
    record: Mapping[str, Any],
    decision: FanoutDirectionDecision,
    source_record: Mapping[str, Any],
) -> tuple[int, int]:
    core = candidate_p_lock_v2_from_record(record["core_lock_record"])
    query_payloads = _sequence(
        source_record.get("deployment_query_root_masks"), "strict query-root payloads"
    )
    _require(
        len(core.roots) == len(decision.selected_action_indices) == len(query_payloads),
        "strict P-V2 root population drift",
    )
    missing = 0
    unmappable = 0
    for root, selected, payload in zip(
        core.roots, decision.selected_action_indices, query_payloads, strict=True
    ):
        compact = CompactMask.from_payload(_mapping(payload, "strict query-root mask"))
        expected_query = compact.decode()
        _require(
            torch.equal(root.query_mask, expected_query),
            "strict P-V2 erased or changed the donor query root",
        )
        if selected is not None:
            _require(root.status == ROOT_READY, "selected control action is not ROOT_READY")
        elif bool(expected_query.any()):
            _require(
                root.status == ROOT_REFERENCE_MISSING,
                "nonempty query root without a reference was not preserved",
            )
            missing += 1
        else:
            _require(
                root.status == ROOT_QUERY_UNMAPPABLE,
                "empty query root was not ROOT_QUERY_UNMAPPABLE",
            )
            unmappable += 1
    return missing, unmappable


@dataclass(frozen=True)
class StrictCColPV2Replay:
    records: tuple[Mapping[str, Any], ...]
    decisions: tuple[FanoutDirectionDecision, ...]
    source_records: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.records)
        _require(
            count > 0
            and count == len(self.decisions) == len(self.source_records)
            and count % len(P_DIRECTIONS) == 0,
            "strict C_COL replay population drift",
        )
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "source_records", tuple(self.source_records))
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))


def replay_strict_c_col_p_v2(
    *,
    raw_source: Mapping[str, Any],
    query_id: str,
    checkpoint: Mapping[str, Any],
    p_checkpoint_sha256: str,
    p_training_manifest_sha256: str,
    corrected_identity_by_physical_row: Mapping[int, str],
    query_geometry_record: Mapping[str, Any],
    reference_geometry_by_physical_row: Mapping[int, Mapping[str, Any]],
    fold_record: Mapping[str, Any],
    membership: Mapping[str, Any],
    expected_count: int = 128,
) -> StrictCColPV2Replay:
    """Build a complete target-free C_COL_P outer-refit Natural/Core V2 axis."""

    checkpoint_sha = _sha(p_checkpoint_sha256, "P checkpoint")
    manifest_sha = _sha(p_training_manifest_sha256, "P training manifest")
    raw_query = _select_raw_query(raw_source, query_id)
    source_fold = int(raw_query.get("source_fold", -1))
    model = load_strict_outer_refit_p_v2_model(
        checkpoint,
        source_fold=source_fold,
        p_training_manifest_sha256=manifest_sha,
    )
    fit_id = f"P_OUTER{source_fold}_OUTER_REFIT"
    _require(
        fold_record.get("query_id") == query_id
        and fold_record.get("query_ordinal")
        == raw_query.get("historical_query_ordinal")
        and fold_record.get("source_image_sha256")
        == raw_query.get("query_source_image_sha256")
        and fold_record.get("inner_fold") == source_fold,
        "strict C_COL fold/query binding drift",
    )
    candidates = tuple(
        _mapping(item, "strict raw candidate")
        for item in _sequence(raw_query.get("candidates"), "strict candidate axis")
    )
    _require(
        len(candidates) == expected_count
        and tuple(int(item.get("candidate_position", -1)) for item in candidates)
        == tuple(range(expected_count)),
        "strict C_COL candidate axis drift",
    )
    donor_positions = canonical_donor_positions(
        raw_query,
        corrected_identity_by_physical_row,
        expected_count=expected_count,
    )
    source_donor_receipt = donor_receipt(
        raw_query,
        corrected_identity_by_physical_row,
        expected_count=expected_count,
    )
    query_grid = _grid(raw_query.get("dino_query_grid_shape"), "strict query DINO")
    _require(
        query_geometry_record.get("query_id") == query_id
        and query_geometry_record.get("execution_ordinal")
        == raw_query.get("execution_ordinal")
        and query_geometry_record.get("source_image_sha256")
        == raw_query.get("query_source_image_sha256"),
        "strict C_COL query geometry address drift",
    )
    _geometry_valid_mask(
        query_geometry_record, expected_grid=query_grid, name="strict query"
    )
    destination_valid: dict[str, torch.Tensor] = {}
    reference_records: dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        row = int(candidate.get("candidate_physical_row", -1))
        geometry = reference_geometry_by_physical_row.get(row)
        _require(isinstance(geometry, Mapping), "strict destination geometry absent")
        reference_grid = _grid(
            candidate.get("dino_reference_grid_shape"), "strict destination DINO"
        )
        _require(
            geometry.get("physical_row") == row
            and geometry.get("source_image_sha256")
            == candidate.get("candidate_reference_source_sha256")
            and _mapping(geometry.get("dino_geometry"), "strict destination DINO").get(
                "geometry_sha256"
            )
            == candidate.get("reference_geometry_sha256"),
            "strict destination geometry/source binding drift",
        )
        key = str(candidate.get("candidate_key"))
        destination_valid[key] = _geometry_valid_mask(
            geometry, expected_grid=reference_grid, name="strict destination"
        )
        reference_records[row] = geometry

    feature_records = build_control_feature_records(
        raw_query,
        corrected_identity_by_physical_row=corrected_identity_by_physical_row,
        destination_dino_valid_by_candidate_key=destination_valid,
        expected_count=expected_count,
    )
    _require(
        len(feature_records) == expected_count * len(P_DIRECTIONS),
        "strict C_COL feature population drift",
    )
    observed_coordinates: list[tuple[int, str]] = []
    records: list[Mapping[str, Any]] = []
    decisions: list[FanoutDirectionDecision] = []
    source_records: list[Mapping[str, Any]] = []
    source_populations: list[Mapping[str, Any]] = []
    missing_count = 0
    unmappable_count = 0
    with torch.no_grad():
        for feature in feature_records:
            _require(
                isinstance(feature, DirectionalFeatureRecord),
                "strict C_COL feature record type drift",
            )
            position = feature.candidate_position
            destination = candidates[position]
            donor = candidates[donor_positions[position]]
            source_record, source_population = _strict_source_record(
                raw_query=raw_query,
                destination=destination,
                donor=donor,
                feature_record=feature,
            )
            decision = decision_from_legacy_output(
                score_deployable_direction(model, feature)
            )
            reference_geometry = reference_records[feature.candidate_physical_row]
            record = build_natural_p_lock_v2(
                source_record=source_record,
                query_geometry_record=query_geometry_record,
                reference_geometry_record=reference_geometry,
                fold_record=fold_record,
                membership=membership,
                row_scores=decision.row_scores,
                selected_action_indices=decision.selected_action_indices,
                fit_id=fit_id,
                crossfit_role="OUTER_HELDOUT_DEPLOYMENT",
                p_checkpoint_sha256=checkpoint_sha,
                p_training_manifest_sha256=manifest_sha,
            )
            validate_natural_p_lock_v2(record)
            missing, unmappable = _assert_v2_root_semantics(
                record, decision, source_record
            )
            missing_count += missing
            unmappable_count += unmappable
            observed_coordinates.append((position, feature.direction))
            records.append(MappingProxyType(dict(record)))
            decisions.append(decision)
            source_records.append(MappingProxyType(dict(source_record)))
            source_populations.append(MappingProxyType(dict(source_population)))
    expected_coordinates = [
        (position, direction)
        for position in range(expected_count)
        for direction in P_DIRECTIONS
    ]
    _require(
        observed_coordinates == expected_coordinates,
        "strict C_COL candidate/direction order drift",
    )
    record_hashes = [str(record["record_sha256"]) for record in records]
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "query_id": query_id,
        "execution_ordinal": raw_query.get("execution_ordinal"),
        "source_fold": source_fold,
        "candidate_count": expected_count,
        "direction_count": len(P_DIRECTIONS),
        "rebuilt_lock_record_count": len(records),
        "model_forward_count": len(records),
        "fit_id": fit_id,
        "p_checkpoint_sha256": checkpoint_sha,
        "p_training_manifest_sha256": manifest_sha,
        "donor_positions": list(donor_positions),
        "donor_receipt_logical_sha256": canonical_sha256(source_donor_receipt),
        "donor_receipt": source_donor_receipt,
        "control_source_population_sha256": canonical_sha256(
            [dict(item) for item in source_populations]
        ),
        "record_sha256_sequence": record_hashes,
        "record_sha256_sequence_sha256": canonical_sha256(record_hashes),
        "root_reference_missing_count": missing_count,
        "root_query_unmappable_count": unmappable_count,
        "query_mask_preserved_for_reference_missing": True,
        "complete_p_feature_head_selector_v2_lock_rerun": True,
        "target_rival_read_count": 0,
        "rank_score_outcome_read_count": 0,
        "v_model_load_count": 0,
        "v_model_forward_count": 0,
        "scientific_reduction_count": 0,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return StrictCColPV2Replay(
        records=tuple(records),
        decisions=tuple(decisions),
        source_records=tuple(source_records),
        receipt=receipt,
    )


__all__ = [
    "SCHEMA_VERSION",
    "NAMESPACE",
    "P_V2_CHECKPOINT_SCHEMA",
    "P_V2_CHECKPOINT_STATUS",
    "P_V2_CHECKPOINT_CLAIM",
    "P_V2_COMPLETED_UPDATES",
    "P_V2_CHECKPOINT_FIELDS",
    "P_V2_CHECKPOINT_STATE_FIELDS",
    "P_V2_CHECKPOINT_BINDING_FIELDS",
    "StrictCColPV2CompatError",
    "StrictCColPV2Replay",
    "p_v2_checkpoint_state_sha256",
    "p_v2_checkpoint_logical_sha256",
    "load_strict_outer_refit_p_v2_model",
    "replay_strict_c_col_p_v2",
]
