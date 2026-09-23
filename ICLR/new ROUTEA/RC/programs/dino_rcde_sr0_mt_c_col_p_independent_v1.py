"""Validator-only, source-rebuilt C_COL_P replay.

This module intentionally does not import the production C_COL_P core.  It
constructs the cyclic donor payload independently and routes that payload
through the already frozen generic P feature materializer, P head, resolver,
and lock serializer.  It is therefore a second implementation of the control
composition while sharing only the scientific P primitives being validated.
"""

from __future__ import annotations

import copy
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    RAW_SOURCE_SCHEMA,
    SharedMultitilePHead,
    assert_no_forbidden_prejoin_keys,
    build_p_lock_record,
    canonical_sha256,
    feature_ledger_index,
    score_deployable_direction,
    tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (
    CandidatePLockV1,
    VQueryBundle,
    canonical_candidate_axis,
    sealed_direction_lock_from_p_record,
)
from materialize_dino_rcde_sr0_mt_p_features_v1 import materialize_source_payload


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_c_col_p_v2_20260815"
NAMESPACE = "RCDE_SR0_MT_C_COL_P_IDDISJOINT_CYCLIC_NATIVE_GRID_V2"
EXPECTED_CANDIDATE_COUNT = 128


class IndependentCColPError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentCColPError(message)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value  # type: ignore[return-value]


def _sequence(value: object, name: str) -> Sequence[object]:
    _require(isinstance(value, (list, tuple)), f"{name} must be a sequence")
    return value  # type: ignore[return-value]


def _grid(value: object, name: str) -> tuple[int, int]:
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and min(value) > 0,
        f"{name} grid drift",
    )
    return int(value[0]), int(value[1])  # type: ignore[index]


def independent_donor_positions(
    query: Mapping[str, object],
    corrected_identity_by_physical_row: Mapping[int, str],
    count: int = EXPECTED_CANDIDATE_COUNT,
) -> tuple[int, ...]:
    candidates = _candidate_axis(query, count)
    identities = []
    for item in candidates:
        value = corrected_identity_by_physical_row.get(
            int(item["candidate_physical_row"])
        )
        _require(isinstance(value, str) and bool(value), "independent identity absent")
        identities.append(value)
    offsets = [
        offset
        for offset in range(1, count)
        if all(
            identities[position] != identities[(position + offset) % count]
            for position in range(count)
        )
    ]
    _require(bool(offsets), "independent C_COL_P has no legal cyclic matching")
    order = tuple((position + offsets[0]) % count for position in range(count))
    _require(
        sorted(order) == list(range(count))
        and all(position != donor for position, donor in enumerate(order)),
        "independent C_COL_P donor map drift",
    )
    return order


def _component_legal(mask: torch.Tensor, grid: tuple[int, int]) -> bool:
    value = torch.as_tensor(mask, dtype=torch.bool).reshape(grid)
    points = torch.nonzero(value, as_tuple=False)
    if (
        points.shape[0] < 4
        or torch.unique(points[:, 0]).numel() < 2
        or torch.unique(points[:, 1]).numel() < 2
    ):
        return False
    active = {tuple(map(int, point)) for point in points.tolist()}
    frontier = [next(iter(active))]
    visited = {frontier[0]}
    while frontier:
        row, column = frontier.pop()
        for neighbour in (
            (row - 1, column), (row + 1, column),
            (row, column - 1), (row, column + 1),
        ):
            if neighbour in active and neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return visited == active


def _transport(
    mask: torch.Tensor,
    source_grid: tuple[int, int],
    destination_grid: tuple[int, int],
    destination_valid: torch.Tensor,
) -> torch.Tensor:
    source = torch.as_tensor(mask, dtype=torch.bool).flatten().cpu()
    valid_mask = torch.as_tensor(destination_valid, dtype=torch.bool).flatten().cpu()
    _require(
        source.shape == (math.prod(source_grid),)
        and valid_mask.shape == (math.prod(destination_grid),),
        "independent C_COL_P transport grid drift",
    )
    output = torch.zeros_like(valid_mask)
    active = torch.nonzero(source, as_tuple=False).flatten()
    valid = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if active.numel() == 0 or valid.numel() == 0:
        return output
    sh, sw = source_grid
    dh, dw = destination_grid
    sr, sc = active // sw, active % sw
    dr, dc = valid // dw, valid % dw
    x = (sc[:, None] * dw < (dc[None, :] + 1) * sw) & (
        dc[None, :] * sw < (sc[:, None] + 1) * dw
    )
    y = (sr[:, None] * dh < (dr[None, :] + 1) * sh) & (
        dr[None, :] * sh < (sr[:, None] + 1) * dh
    )
    output[valid[(x & y).any(dim=0)]] = True
    return output.contiguous()


def _action_key(
    destination: Mapping[str, object],
    donor: Mapping[str, object],
    donor_action_key: str,
    direction: str,
    root_ordinal: int,
) -> str:
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "namespace": NAMESPACE,
            "destination_candidate_key": str(destination["candidate_key"]),
            "donor_candidate_key": str(donor["candidate_key"]),
            "donor_action_key": donor_action_key,
            "direction": direction,
            "root_ordinal": root_ordinal,
            "destination_reference_geometry_sha256": str(
                destination["reference_geometry_sha256"]
            ),
        }
    )


def _candidate_axis(query: Mapping[str, object], count: int) -> tuple[Mapping[str, object], ...]:
    candidates = tuple(
        _mapping(item, "independent raw candidate")
        for item in _sequence(query.get("candidates"), "independent candidate axis")
    )
    _require(len(candidates) == count, "independent C_COL_P C axis incomplete")
    _require(
        [int(item.get("candidate_position", -1)) for item in candidates]
        == list(range(count)),
        "independent C_COL_P position axis drift",
    )
    rows = [int(item.get("candidate_physical_row", -1)) for item in candidates]
    _require(
        rows == sorted(set(rows))
        and query.get("candidate_axis_sha256") == canonical_sha256(rows),
        "independent C_COL_P physical axis drift",
    )
    return candidates


def _reference(candidate: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    tokens = torch.as_tensor(candidate.get("reference_tokens")).detach().cpu().contiguous()
    valid = torch.as_tensor(
        candidate.get("reference_valid_patch_mask"), dtype=torch.bool
    ).detach().cpu().flatten().contiguous()
    grid = _grid(candidate.get("colnomic_reference_grid_shape"), "ColNomic reference")
    _require(
        tokens.ndim == 2
        and tokens.is_floating_point()
        and bool(torch.isfinite(tokens).all())
        and tokens.shape[0] == math.prod(grid)
        and valid.shape == (tokens.shape[0],)
        and bool(valid.any()),
        "independent C_COL_P reference tensor/grid drift",
    )
    return tokens, valid, grid


def build_independent_control_source(
    raw_source: Mapping[str, object],
    query_id: str,
    count: int,
    corrected_identity_by_physical_row: Mapping[int, str],
    destination_dino_valid_by_candidate_key: Mapping[str, torch.Tensor],
) -> tuple[dict[str, object], Mapping[str, object], dict[str, object]]:
    """Independently rebuild native donor actions on destination DINO grids."""

    _require(
        raw_source.get("schema_version") == RAW_SOURCE_SCHEMA
        and raw_source.get("target_free") is True,
        "independent C_COL_P source schema/role drift",
    )
    assert_no_forbidden_prejoin_keys(raw_source)
    selected = [
        _mapping(item, "independent raw query")
        for item in _sequence(raw_source.get("queries"), "independent raw queries")
        if isinstance(item, Mapping) and str(item.get("query_id", "")) == query_id
    ]
    _require(len(selected) == 1, "independent C_COL_P query absent/duplicated")
    source_query = selected[0]
    candidates = _candidate_axis(source_query, count)
    order = independent_donor_positions(
        source_query, corrected_identity_by_physical_row, count
    )
    control_query = copy.deepcopy(dict(source_query))
    control_candidates = control_query["candidates"]
    _require(isinstance(control_candidates, list), "independent control candidate axis drift")
    bindings = []
    input_hashes = []
    output_hashes = []
    geometry_rows = []
    changed = 0
    for position, donor_position in enumerate(order):
        destination = candidates[position]
        donor = candidates[donor_position]
        destination_tokens, destination_valid, destination_grid = _reference(destination)
        donor_tokens, donor_valid, donor_grid = _reference(donor)
        _require(
            destination_tokens.shape[1] == donor_tokens.shape[1]
            and destination_tokens.dtype == donor_tokens.dtype,
            "independent C_COL_P donor descriptor dimension/dtype drift",
        )
        destination_hash = tensor_sha256(destination_tokens)
        donor_hash = tensor_sha256(donor_tokens)
        changed += int(destination_hash != donor_hash)
        input_hashes.append(destination_hash)
        output_hashes.append(donor_hash)
        controlled = control_candidates[position]
        controlled["reference_tokens"] = donor_tokens.clone()
        controlled["reference_valid_patch_mask"] = donor_valid.clone()
        controlled["colnomic_reference_grid_shape"] = list(donor_grid)
        destination_dino_grid = _grid(
            destination.get("dino_reference_grid_shape"), "DINO reference"
        )
        donor_dino_grid = _grid(
            donor.get("dino_reference_grid_shape"), "donor DINO reference"
        )
        destination_dino_valid = destination_dino_valid_by_candidate_key.get(
            str(destination["candidate_key"])
        )
        _require(
            isinstance(destination_dino_valid, torch.Tensor)
            and destination_dino_valid.numel() == math.prod(destination_dino_grid),
            "independent destination DINO valid mask absent",
        )
        rebuilt_directions: dict[str, list[dict[str, object]]] = {}
        donor_directions = _mapping(donor.get("directions"), "donor directions")
        for direction in P_DIRECTIONS:
            roots = []
            for root_ordinal, raw_root in enumerate(
                _sequence(donor_directions.get(direction), "donor root bank")
            ):
                donor_root = _mapping(raw_root, "donor root")
                _require(
                    donor_root.get("root_ordinal") == root_ordinal,
                    "independent donor root order drift",
                )
                actions = []
                for raw_action in _sequence(donor_root.get("actions"), "donor actions"):
                    donor_action = _mapping(raw_action, "donor action")
                    donor_query_mask = torch.as_tensor(
                        donor_action.get("deployment_query_mask"), dtype=torch.bool
                    ).flatten()
                    donor_reference_mask = torch.as_tensor(
                        donor_action.get("deployment_reference_mask"), dtype=torch.bool
                    ).flatten()
                    transported = _transport(
                        donor_reference_mask,
                        donor_dino_grid,
                        destination_dino_grid,
                        destination_dino_valid,
                    )
                    eligible = bool(
                        donor_action.get("deployment_eligible")
                        and _component_legal(
                            donor_query_mask,
                            _grid(source_query.get("dino_query_grid_shape"), "DINO query"),
                        )
                        and _component_legal(transported, destination_dino_grid)
                    )
                    actions.append(
                        {
                            "action_key": _action_key(
                                destination,
                                donor,
                                str(donor_action.get("action_key", "")),
                                direction,
                                root_ordinal,
                            ),
                            "colnomic_query_action_mask": torch.as_tensor(
                                donor_action.get("colnomic_query_action_mask"),
                                dtype=torch.bool,
                            ).clone(),
                            "colnomic_reference_action_mask": torch.as_tensor(
                                donor_action.get("colnomic_reference_action_mask"),
                                dtype=torch.bool,
                            ).clone(),
                            "deployment_eligible": eligible,
                            "deployment_query_mask": (
                                donor_query_mask.clone()
                                if eligible
                                else torch.zeros_like(donor_query_mask)
                            ),
                            "deployment_reference_mask": (
                                transported.clone()
                                if eligible
                                else torch.zeros_like(transported)
                            ),
                        }
                    )
                roots.append(
                    {
                        "root_ordinal": root_ordinal,
                        "reference_action_bank_sha256": canonical_sha256(
                            {
                                "namespace": NAMESPACE,
                                "donor_bank": donor_root.get(
                                    "reference_action_bank_sha256"
                                ),
                                "destination": str(destination["candidate_key"]),
                            }
                        ),
                        "actions": actions,
                    }
                )
            rebuilt_directions[direction] = roots
        controlled["directions"] = rebuilt_directions
        geometry = {
            "destination_colnomic_grid": list(destination_grid),
            "donor_colnomic_grid": list(donor_grid),
            "destination_valid_mask_sha256": tensor_sha256(destination_valid),
            "donor_valid_mask_sha256": tensor_sha256(donor_valid),
            "destination_dino_grid": list(destination_dino_grid),
            "destination_dino_geometry_sha256": str(
                destination.get("reference_geometry_sha256", "")
            ),
            "donor_dino_grid": list(donor_dino_grid),
        }
        _require(
            len(geometry["destination_dino_geometry_sha256"]) == 64,
            "independent destination geometry hash drift",
        )
        geometry_rows.append(geometry)
        bindings.append(
            {
                "destination_position": position,
                "donor_position": donor_position,
                "destination_candidate_key": str(destination["candidate_key"]),
                "donor_candidate_key": str(donor["candidate_key"]),
                "destination_physical_row": int(destination["candidate_physical_row"]),
                "donor_physical_row": int(donor["candidate_physical_row"]),
                "destination_source_sha256": str(
                    destination["candidate_reference_source_sha256"]
                ),
                "donor_source_sha256": str(donor["candidate_reference_source_sha256"]),
                "destination_colnomic_tokens_sha256": destination_hash,
                "donor_colnomic_tokens_sha256": donor_hash,
                "destination_geometry": geometry,
            }
        )
    _require(changed > 0, "independent C_COL_P source is globally content-identical")
    control_source = {
        "schema_version": RAW_SOURCE_SCHEMA,
        "target_free": True,
        "synthetic": bool(raw_source.get("synthetic", False)),
        "candidate_count_per_query": count,
        "queries": [control_query],
    }
    control_source["logical_sha256"] = canonical_sha256(
        {
            "schema_version": RAW_SOURCE_SCHEMA,
            "namespace": NAMESPACE,
            "query_id": query_id,
            "binding_sequence": bindings,
        }
    )
    assert_no_forbidden_prejoin_keys(control_source)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "query_id": query_id,
        "candidate_count": count,
        "donor_positions": list(order),
        "cyclic_offset": int(order[0]),
        "fixed_point_free": True,
        "complete_permutation": True,
        "corrected_identity_disjoint": True,
        "corrected_identity_axis_sha256": canonical_sha256(
            [
                corrected_identity_by_physical_row[int(item["candidate_physical_row"])]
                for item in candidates
            ]
        ),
        "content_multiset_preserved": sorted(input_hashes) == sorted(output_hashes),
        "input_reference_content_axis_sha256": canonical_sha256(input_hashes),
        "output_reference_content_axis_sha256": canonical_sha256(output_hashes),
        "reference_content_multiset_sha256": canonical_sha256(sorted(input_hashes)),
        "destination_geometry_axis_sha256": canonical_sha256(geometry_rows),
        "bindings": bindings,
        "binding_sequence_sha256": canonical_sha256(bindings),
        "response_dependent_retry_count": 0,
        "fallback_donor_count": 0,
        "native_colnomic_token_resize_count": 0,
        "reference_mask_transport_rule": (
            "NORMALIZED_CELL_RECTANGLE_STRICT_POSITIVE_AREA_OVERLAP"
        ),
        "cross_grid_binding_count": sum(
            geometry["destination_dino_grid"] != geometry["donor_dino_grid"]
            or geometry["destination_colnomic_grid"] != geometry["donor_colnomic_grid"]
            for geometry in geometry_rows
        ),
    }
    _require(receipt["content_multiset_preserved"] is True, "independent multiset drift")
    return control_source, source_query, receipt


def _fingerprint_payload(bundle: VQueryBundle) -> dict[str, object]:
    axis = canonical_candidate_axis(bundle.locks)
    candidates = []
    for key in axis:
        candidate_lock = bundle.locks[key]
        candidate = candidate_lock.candidate
        directions = []
        for direction in P_DIRECTIONS:
            lock = candidate_lock.direction_locks[direction]
            directions.append(
                {
                    "direction": direction,
                    "record_sha256": lock.p_lock_record_sha256,
                    "status": lock.status,
                    "selected_bank_ordinal": lock.selected_bank_ordinal,
                    "selected_row_sha256": lock.selected_row_sha256,
                    "query_union_mask_sha256": tensor_sha256(lock.query_union_mask),
                    "ordered_root_ordinals": list(lock.ordered_root_ordinals),
                    "all_roots": [
                        {
                            "root_ordinal": root,
                            "action_key_sha256": scope.action_key_sha256,
                            "query_mask_sha256": tensor_sha256(scope.query_mask),
                            "reference_mask_sha256": tensor_sha256(scope.reference_mask),
                            "binding_status": scope.binding_status,
                        }
                        for root, scope in lock.all_roots.items()
                    ],
                }
            )
        candidates.append(
            {
                "candidate_key": key,
                "physical_gallery_row": candidate.physical_gallery_row,
                "source_image_sha256": candidate.source_image_sha256,
                "source_key": candidate.source_key,
                "cache_payload_sha256": candidate.cache_payload_sha256,
                "geometry_record_sha256": candidate.geometry_record_sha256,
                "tokens_sha256": candidate.tokens_sha256,
                "valid_patch_mask_sha256": tensor_sha256(candidate.valid_patch_mask),
                "grid_shape": list(candidate.grid_shape),
                "directions": directions,
            }
        )
    return {
        "query_id": bundle.query_id,
        "execution_ordinal": bundle.execution_ordinal,
        "outer_fold": bundle.outer_fold,
        "candidate_axis_sha256": bundle.candidate_axis_sha256,
        "query": {
            "source_image_sha256": bundle.query.source_image_sha256,
            "source_key": bundle.query.source_key,
            "cache_payload_sha256": bundle.query.cache_payload_sha256,
            "geometry_record_sha256": bundle.query.geometry_record_sha256,
            "tokens_sha256": bundle.query.tokens_sha256,
            "valid_patch_mask_sha256": tensor_sha256(bundle.query.valid_patch_mask),
            "grid_shape": list(bundle.query.grid_shape),
        },
        "candidates": candidates,
    }


def independent_bundle_fingerprint(bundle: VQueryBundle) -> str:
    return canonical_sha256(_fingerprint_payload(bundle))


def independent_dino_fingerprint(bundle: VQueryBundle) -> str:
    payload = _fingerprint_payload(bundle)
    candidates = []
    for item in payload["candidates"]:  # type: ignore[index]
        row = dict(item)  # type: ignore[arg-type]
        row.pop("directions")
        candidates.append(row)
    return canonical_sha256(
        {
            "query": payload["query"],
            "candidate_axis_sha256": payload["candidate_axis_sha256"],
            "candidates": candidates,
        }
    )


def independent_replay_c_col_p(
    *,
    raw_source: Mapping[str, object],
    query_id: str,
    real_bundle: VQueryBundle,
    model: SharedMultitilePHead,
    outer_fold: int,
    p_checkpoint_sha256: str,
    p_training_manifest_sha256: str,
    corrected_identity_by_physical_row: Mapping[int, str],
    expected_count: int = EXPECTED_CANDIDATE_COUNT,
) -> tuple[VQueryBundle, dict[str, object], tuple[Mapping[str, object], ...]]:
    control_source, raw_query, receipt = build_independent_control_source(
        raw_source,
        query_id,
        expected_count,
        corrected_identity_by_physical_row,
        {
            key: real_bundle.locks[key].candidate.valid_patch_mask
            for key in canonical_candidate_axis(real_bundle.locks)
        },
    )
    candidates = _candidate_axis(raw_query, expected_count)
    axis = canonical_candidate_axis(real_bundle.locks)
    _require(
        tuple(str(item["candidate_key"]) for item in candidates) == axis
        and real_bundle.query_id == query_id
        and real_bundle.execution_ordinal == int(raw_query["execution_ordinal"])
        and real_bundle.outer_fold == outer_fold
        and int(raw_query["source_fold"]) == outer_fold
        and real_bundle.query.source_image_sha256
        == raw_query["query_source_image_sha256"],
        "independent raw/DINO query candidate binding drift",
    )
    for item in candidates:
        key = str(item["candidate_key"])
        field = real_bundle.locks[key].candidate
        _require(
            field.physical_gallery_row == int(item["candidate_physical_row"])
            and field.source_image_sha256
            == item["candidate_reference_source_sha256"]
            and field.geometry_record_sha256 == item["reference_geometry_sha256"]
            and field.grid_shape
            == _grid(item["dino_reference_grid_shape"], "DINO reference"),
            "independent destination DINO binding drift",
        )
    ledger = materialize_source_payload(
        control_source,
        source_file_sha256=canonical_sha256(
            {"namespace": NAMESPACE, "query_id": query_id}
        ),
        source_role="C_COL_P_INDEPENDENT_REPLAY_ONLY",
    )
    index = feature_ledger_index(ledger)
    records = sorted(
        index.values(),
        key=lambda item: (
            item.candidate_position,
            P_DIRECTIONS.index(item.direction),
        ),
    )
    _require(
        len(records) == expected_count * len(P_DIRECTIONS),
        "independent C_COL_P directional feature population drift",
    )
    lock_records = tuple(
        build_p_lock_record(
            score_deployable_direction(model, record),
            crossfit_role="OUTER_HELDOUT_DEPLOYMENT",
            outer_fold=outer_fold,
            p_checkpoint_sha256=p_checkpoint_sha256,
            p_training_manifest_sha256=p_training_manifest_sha256,
        )
        for record in records
    )
    by_address = {
        (int(item["candidate_physical_row"]), str(item["direction"])): item
        for item in lock_records
    }
    _require(
        len(by_address) == expected_count * len(P_DIRECTIONS),
        "independent C_COL_P lock address population drift",
    )
    locks = {}
    for key in axis:
        candidate = real_bundle.locks[key].candidate
        directions = {
            direction: sealed_direction_lock_from_p_record(
                by_address[(candidate.physical_gallery_row, direction)],
                query=real_bundle.query,
                candidate=candidate,
            )
            for direction in P_DIRECTIONS
        }
        locks[key] = CandidatePLockV1(candidate=candidate, direction_locks=directions)
    controlled = VQueryBundle(
        query_id=real_bundle.query_id,
        execution_ordinal=real_bundle.execution_ordinal,
        outer_fold=real_bundle.outer_fold,
        query=real_bundle.query,
        locks=locks,
        candidate_axis_sha256=real_bundle.candidate_axis_sha256,
    )
    input_dino = independent_dino_fingerprint(real_bundle)
    output_dino = independent_dino_fingerprint(controlled)
    _require(input_dino == output_dino, "independent C_COL_P changed DINO content")
    real_hashes = [
        real_bundle.locks[key].direction_locks[direction].p_lock_record_sha256
        for key in axis
        for direction in P_DIRECTIONS
    ]
    control_hashes = [
        controlled.locks[key].direction_locks[direction].p_lock_record_sha256
        for key in axis
        for direction in P_DIRECTIONS
    ]
    changed = sum(first != second for first, second in zip(real_hashes, control_hashes))
    receipt.update(
        {
            "real_bundle_fingerprint": independent_bundle_fingerprint(real_bundle),
            "control_bundle_fingerprint": independent_bundle_fingerprint(controlled),
            "dino_axis_content_input_sha256": input_dino,
            "dino_axis_content_output_sha256": output_dino,
            "dino_axis_content_byte_identical": True,
            "real_p_lock_sequence_sha256": canonical_sha256(real_hashes),
            "control_p_lock_sequence_sha256": canonical_sha256(control_hashes),
            "rerun_p_lock_count": len(control_hashes),
            "changed_p_lock_count": changed,
            "unchanged_p_lock_count": len(control_hashes) - changed,
            "complete_p_feature_head_resolver_lock_rerun": True,
            "p_checkpoint_sha256": p_checkpoint_sha256,
            "p_training_manifest_sha256": p_training_manifest_sha256,
        }
    )
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return controlled, receipt, tuple(
        MappingProxyType(dict(item)) for item in lock_records
    )


def assert_observed_matches_independent(
    *,
    observed_bundle: VQueryBundle,
    observed_receipt: Mapping[str, object],
    rebuilt_bundle: VQueryBundle,
    rebuilt_receipt: Mapping[str, object],
) -> None:
    """Fail closed if a serialized producer result differs from source replay.

    Keeping this comparison in the validator-only module makes receipt tamper
    tests exercise the same boundary as the command-line validator without
    importing or calling the production replay implementation.
    """

    _require(
        dict(observed_receipt) == dict(rebuilt_receipt),
        "observed C_COL_P receipt differs from independent source replay",
    )
    _require(
        independent_bundle_fingerprint(observed_bundle)
        == independent_bundle_fingerprint(rebuilt_bundle),
        "observed C_COL_P bundle differs from independent source replay",
    )
    _require(
        independent_dino_fingerprint(observed_bundle)
        == independent_dino_fingerprint(rebuilt_bundle),
        "observed C_COL_P DINO axis/content differs from independent replay",
    )


__all__ = [
    "SCHEMA_VERSION",
    "NAMESPACE",
    "EXPECTED_CANDIDATE_COUNT",
    "IndependentCColPError",
    "independent_donor_positions",
    "build_independent_control_source",
    "independent_bundle_fingerprint",
    "independent_dino_fingerprint",
    "independent_replay_c_col_p",
    "assert_observed_matches_independent",
]
