"""Canonical pre-P ColNomic candidate-binding destruction for SR0-MT.

``C_COL_P`` is deliberately not a post-hoc mask shuffle.  It chooses the first
pre-registered cyclic offset that is corrected-identity-disjoint at every
destination, binds each donor's complete native ColNomic token/valid/action
bank before P, then reruns the frozen P feature definition, P head, resolver,
and lock seal.

The destination DINO candidate field is never copied or transformed here.  A
caller must provide the already validated target-free :class:`VQueryBundle`;
the returned bundle reuses its query and candidate objects byte-for-byte and
changes only the sealed P locks.  Donor ColNomic tensors are never resized.
Their already canonical DINO reference footprints are area-rasterized from the
donor DINO grid onto the destination DINO grid by normalized positive overlap,
then re-sealed with destination geometry.  There is no ordinal guessing,
fallback donor, or response-dependent retry.

This module contains the canonical scientific transform and deterministic
model replay.  Formal materialization and independent validation live in
separate programs and independently load every bound source artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    FEATURE_DIM,
    CandidateActionFeatureTable,
    candidate_action_colnomic_features,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    RAW_SOURCE_SCHEMA,
    SharedMultitilePHead,
    assert_no_forbidden_prejoin_keys,
    build_p_lock_record,
    canonical_sha256,
    make_directional_feature_record,
    make_root_action_deployment,
    score_deployable_direction,
    tensor_sha256,
)
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    CandidatePLockV1,
    VQueryBundle,
    canonical_candidate_axis,
    sealed_direction_lock_from_p_record,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_c_col_p_v2_20260815"
NAMESPACE = "RCDE_SR0_MT_C_COL_P_IDDISJOINT_CYCLIC_NATIVE_GRID_V2"
CONTROL_INPUT_SCHEMA = "rc_dino_rcde_sr0_mt_c_col_p_query_input_v1"
CONTROL_VALIDATION_SCHEMA = "rc_dino_rcde_sr0_mt_c_col_p_query_validation_v1"
EXPECTED_CANDIDATE_COUNT = 128


class CColPContractError(RuntimeError):
    """Raised when a C_COL_P transform cannot satisfy the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CColPContractError(message)


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


def _candidate_axis(raw_query: Mapping[str, object], expected_count: int) -> tuple[Mapping[str, object], ...]:
    candidates = tuple(
        _mapping(item, "raw C_COL_P candidate")
        for item in _sequence(raw_query.get("candidates"), "raw candidate axis")
    )
    _require(len(candidates) == expected_count, "C_COL_P candidate axis is not complete")
    _require(
        tuple(int(item.get("candidate_position", -1)) for item in candidates)
        == tuple(range(expected_count)),
        "C_COL_P candidate positions are not canonical 0..C-1",
    )
    rows = tuple(int(item.get("candidate_physical_row", -1)) for item in candidates)
    keys = tuple(str(item.get("candidate_key", "")) for item in candidates)
    sources = tuple(
        str(item.get("candidate_reference_source_sha256", "")) for item in candidates
    )
    _require(
        rows == tuple(sorted(rows))
        and len(set(rows)) == expected_count
        and len(set(keys)) == expected_count
        and all(len(item) == 64 for item in sources),
        "C_COL_P physical/key/source axis drift",
    )
    _require(
        raw_query.get("candidate_axis_sha256") == canonical_sha256(list(rows)),
        "C_COL_P physical candidate-axis receipt drift",
    )
    return candidates


def canonical_donor_positions(
    raw_query: Mapping[str, object],
    corrected_identity_by_physical_row: Mapping[int, str],
    *,
    expected_count: int = EXPECTED_CANDIDATE_COUNT,
) -> tuple[int, ...]:
    """Return the frozen result-blind identity-disjoint cyclic matching.

    Candidate order and corrected gallery identities are inference-visible
    reference metadata.  No query result, score, rank, label join, or target is
    consulted.  The smallest legal nonzero cyclic offset is the registered
    deterministic tie break.
    """

    candidates = _candidate_axis(raw_query, expected_count)
    identities: list[str] = []
    for candidate in candidates:
        row = int(candidate["candidate_physical_row"])
        identity = corrected_identity_by_physical_row.get(row)
        _require(
            isinstance(identity, str) and bool(identity),
            "C_COL_P corrected gallery identity is absent",
        )
        identities.append(identity)
    legal_offsets = [
        offset
        for offset in range(1, expected_count)
        if all(
            identities[position] != identities[(position + offset) % expected_count]
            for position in range(expected_count)
        )
    ]
    _require(
        bool(legal_offsets),
        "C_COL_P candidate axis has no identity-disjoint cyclic perfect matching",
    )
    offset = legal_offsets[0]
    order = tuple(
        (position + offset) % expected_count for position in range(expected_count)
    )
    _require(
        tuple(sorted(order)) == tuple(range(expected_count))
        and all(position != donor for position, donor in enumerate(order))
        and all(identities[position] != identities[donor] for position, donor in enumerate(order)),
        "C_COL_P donor map is not an identity-disjoint derangement",
    )
    return order


def _reference_content(candidate: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    tokens = torch.as_tensor(candidate.get("reference_tokens"))
    valid = torch.as_tensor(
        candidate.get("reference_valid_patch_mask"), dtype=torch.bool
    ).flatten()
    grid = _grid(candidate.get("colnomic_reference_grid_shape"), "ColNomic reference")
    _require(
        tokens.ndim == 2
        and tokens.is_floating_point()
        and bool(torch.isfinite(tokens).all())
        and tokens.shape[0] == math.prod(grid)
        and valid.shape == (tokens.shape[0],)
        and bool(valid.any()),
        "C_COL_P ColNomic reference tensor/grid/validity drift",
    )
    return tokens.detach().cpu().contiguous(), valid.detach().cpu().contiguous(), grid


def _binding_rows(
    raw_query: Mapping[str, object],
    corrected_identity_by_physical_row: Mapping[int, str],
    expected_count: int,
) -> tuple[tuple[Mapping[str, object], Mapping[str, object]], ...]:
    candidates = _candidate_axis(raw_query, expected_count)
    order = canonical_donor_positions(
        raw_query,
        corrected_identity_by_physical_row,
        expected_count=expected_count,
    )
    rows = []
    changed = 0
    for position, donor_position in enumerate(order):
        destination, donor = candidates[position], candidates[donor_position]
        destination_tokens, _, _ = _reference_content(destination)
        donor_tokens, _, _ = _reference_content(donor)
        _require(
            destination_tokens.shape[1] == donor_tokens.shape[1]
            and destination_tokens.dtype == donor_tokens.dtype,
            "C_COL_P donor descriptor dtype/dimension drift",
        )
        changed += int(not torch.equal(destination_tokens, donor_tokens))
        rows.append((destination, donor))
    _require(changed > 0, "C_COL_P cyclic binding is content-identical everywhere")
    return tuple(rows)


def donor_receipt(
    raw_query: Mapping[str, object],
    corrected_identity_by_physical_row: Mapping[int, str],
    *,
    expected_count: int = EXPECTED_CANDIDATE_COUNT,
) -> dict[str, object]:
    """Build a source-only receipt for the frozen cyclic donor assignment."""

    rows = _binding_rows(
        raw_query, corrected_identity_by_physical_row, expected_count
    )
    order = canonical_donor_positions(
        raw_query,
        corrected_identity_by_physical_row,
        expected_count=expected_count,
    )
    bindings = []
    input_hashes = []
    output_hashes = []
    geometry_rows = []
    for destination, donor in rows:
        destination_tokens, destination_valid, destination_grid = _reference_content(
            destination
        )
        donor_tokens, donor_valid, donor_grid = _reference_content(donor)
        input_hash = tensor_sha256(destination_tokens)
        output_hash = tensor_sha256(donor_tokens)
        input_hashes.append(input_hash)
        output_hashes.append(output_hash)
        geometry = {
            "destination_colnomic_grid": list(destination_grid),
            "donor_colnomic_grid": list(donor_grid),
            "destination_valid_mask_sha256": tensor_sha256(destination_valid),
            "donor_valid_mask_sha256": tensor_sha256(donor_valid),
            "destination_dino_grid": list(
                _grid(destination.get("dino_reference_grid_shape"), "DINO reference")
            ),
            "destination_dino_geometry_sha256": str(
                destination.get("reference_geometry_sha256", "")
            ),
            "donor_dino_grid": list(
                _grid(donor.get("dino_reference_grid_shape"), "donor DINO reference")
            ),
        }
        _require(
            len(geometry["destination_dino_geometry_sha256"]) == 64,
            "C_COL_P destination DINO geometry receipt drift",
        )
        geometry_rows.append(geometry)
        bindings.append(
            {
                "destination_position": int(destination["candidate_position"]),
                "donor_position": int(donor["candidate_position"]),
                "destination_candidate_key": str(destination["candidate_key"]),
                "donor_candidate_key": str(donor["candidate_key"]),
                "destination_physical_row": int(destination["candidate_physical_row"]),
                "donor_physical_row": int(donor["candidate_physical_row"]),
                "destination_source_sha256": str(
                    destination["candidate_reference_source_sha256"]
                ),
                "donor_source_sha256": str(donor["candidate_reference_source_sha256"]),
                "destination_colnomic_tokens_sha256": input_hash,
                "donor_colnomic_tokens_sha256": output_hash,
                "destination_geometry": geometry,
            }
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "query_id": str(raw_query.get("query_id", "")),
        "candidate_count": expected_count,
        "donor_positions": list(order),
        "cyclic_offset": int((order[0] - 0) % expected_count),
        "fixed_point_free": True,
        "complete_permutation": True,
        "corrected_identity_disjoint": True,
        "corrected_identity_axis_sha256": canonical_sha256(
            [
                corrected_identity_by_physical_row[int(item["candidate_physical_row"])]
                for item in _candidate_axis(raw_query, expected_count)
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
    _require(receipt["content_multiset_preserved"] is True, "C_COL_P content multiset drift")
    return receipt


def _component_is_legal(mask: torch.Tensor, grid: tuple[int, int]) -> bool:
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


def transport_reference_mask(
    source_mask: torch.Tensor,
    source_grid: tuple[int, int],
    destination_grid: tuple[int, int],
    destination_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact positive-area cell transport between normalized native grids."""

    source = torch.as_tensor(source_mask, dtype=torch.bool).flatten().cpu()
    destination_valid = torch.as_tensor(
        destination_valid_mask, dtype=torch.bool
    ).flatten().cpu()
    _require(
        source.shape == (math.prod(source_grid),)
        and destination_valid.shape == (math.prod(destination_grid),),
        "C_COL_P reference-mask transport grid drift",
    )
    output = torch.zeros_like(destination_valid)
    active = torch.nonzero(source, as_tuple=False).flatten()
    valid = torch.nonzero(destination_valid, as_tuple=False).flatten()
    if active.numel() == 0 or valid.numel() == 0:
        return output
    sh, sw = source_grid
    dh, dw = destination_grid
    sr, sc = torch.div(active, sw, rounding_mode="floor"), active % sw
    dr, dc = torch.div(valid, dw, rounding_mode="floor"), valid % dw
    # Strict positive normalized rectangle overlap, written with integer cross
    # products so shared boundaries never become false overlaps by float error.
    horizontal = (sc[:, None] * dw < (dc[None, :] + 1) * sw) & (
        dc[None, :] * sw < (sc[:, None] + 1) * dw
    )
    vertical = (sr[:, None] * dh < (dr[None, :] + 1) * sh) & (
        dr[None, :] * sh < (sr[:, None] + 1) * dh
    )
    output[valid[(horizontal & vertical).any(dim=0)]] = True
    return output.contiguous()


def _control_action_key(
    *,
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


def _directional_record(
    raw_query: Mapping[str, object],
    destination: Mapping[str, object],
    donor: Mapping[str, object],
    destination_dino_valid_mask: torch.Tensor,
    *,
    direction: str,
):
    q_tokens = torch.as_tensor(raw_query.get("query_tokens"))
    q_valid = torch.as_tensor(
        raw_query.get("query_valid_patch_mask"), dtype=torch.bool
    ).flatten()
    cq_grid = _grid(raw_query.get("colnomic_query_grid_shape"), "ColNomic query")
    dq_grid = _grid(raw_query.get("dino_query_grid_shape"), "DINO query")
    donor_tokens, donor_valid, reference_grid = _reference_content(donor)
    dr_grid = _grid(destination.get("dino_reference_grid_shape"), "DINO reference")
    donor_dr_grid = _grid(
        donor.get("dino_reference_grid_shape"), "donor DINO reference"
    )
    destination_dino_valid = torch.as_tensor(
        destination_dino_valid_mask, dtype=torch.bool
    ).flatten()
    _require(
        q_tokens.ndim == 2
        and q_tokens.shape[0] == math.prod(cq_grid)
        and q_valid.shape == (q_tokens.shape[0],)
        and bool(torch.isfinite(q_tokens).all()),
        "C_COL_P query token/grid drift",
    )
    _require(
        destination_dino_valid.shape == (math.prod(dr_grid),)
        and bool(destination_dino_valid.any()),
        "C_COL_P destination DINO valid-mask drift",
    )
    directions = _mapping(donor.get("directions"), "donor candidate directions")
    roots = _sequence(directions.get(direction), f"{direction} root bank")
    _require(bool(roots), "C_COL_P root bank is empty")
    action_keys_by_root = []
    features_by_root = []
    eligible_by_root = []
    deployments_by_root = []
    for expected_root, raw_root in enumerate(roots):
        root = _mapping(raw_root, "C_COL_P root")
        _require(root.get("root_ordinal") == expected_root, "C_COL_P root order drift")
        actions = _sequence(root.get("actions"), "C_COL_P actions")
        _require(bool(actions), "C_COL_P root has no action/H0 row")
        keys = []
        features = []
        eligible = []
        deployments = []
        for raw_action in actions:
            action = _mapping(raw_action, "C_COL_P action")
            donor_action_key = str(action.get("action_key", ""))
            action_key = _control_action_key(
                destination=destination,
                donor=donor,
                donor_action_key=donor_action_key,
                direction=direction,
                root_ordinal=expected_root,
            )
            donor_deployable = bool(action.get("deployment_eligible"))
            qmask = torch.as_tensor(
                action.get("deployment_query_mask"), dtype=torch.bool
            ).flatten()
            donor_rmask = torch.as_tensor(
                action.get("deployment_reference_mask"), dtype=torch.bool
            ).flatten()
            transported = transport_reference_mask(
                donor_rmask,
                donor_dr_grid,
                dr_grid,
                destination_dino_valid,
            )
            deployable = bool(
                donor_deployable
                and _component_is_legal(qmask, dq_grid)
                and _component_is_legal(transported, dr_grid)
            )
            deployment = make_root_action_deployment(
                action_key=action_key,
                eligible=deployable,
                query_mask=qmask,
                reference_mask=transported,
                query_grid_shape=dq_grid,
                reference_grid_shape=dr_grid,
                query_geometry_sha256=str(raw_query.get("query_geometry_sha256", "")),
                reference_geometry_sha256=str(
                    destination.get("reference_geometry_sha256", "")
                ),
            )
            if deployable:
                feature = candidate_action_colnomic_features(
                    q_tokens,
                    donor_tokens,
                    query_action_mask=torch.as_tensor(
                        action.get("colnomic_query_action_mask"), dtype=torch.bool
                    ),
                    reference_action_mask=torch.as_tensor(
                        action.get("colnomic_reference_action_mask"), dtype=torch.bool
                    ),
                    query_valid_patch_mask=q_valid,
                    reference_valid_patch_mask=donor_valid,
                )
            else:
                feature = torch.zeros(FEATURE_DIM, dtype=torch.float64)
            keys.append(action_key)
            features.append(feature.to(torch.float64))
            eligible.append(deployable)
            deployments.append(deployment)
        action_keys_by_root.append(tuple(keys))
        features_by_root.append(torch.stack(features).to(torch.float64))
        eligible_by_root.append(torch.tensor(eligible, dtype=torch.bool))
        deployments_by_root.append(tuple(deployments))
    candidate_key = str(destination.get("candidate_key", ""))
    table = CandidateActionFeatureTable(
        candidate_key=candidate_key,
        root_ordinals=tuple(range(len(roots))),
        action_keys_by_root=tuple(action_keys_by_root),
        features_by_root=tuple(features_by_root),
        eligible_by_root=tuple(eligible_by_root),
    )
    return make_directional_feature_record(
        query_id=str(raw_query.get("query_id", "")),
        historical_query_ordinal=int(raw_query.get("historical_query_ordinal", -1)),
        execution_ordinal=int(raw_query.get("execution_ordinal", -1)),
        query_source_image_sha256=str(raw_query.get("query_source_image_sha256", "")),
        source_fold=int(raw_query.get("source_fold", -1)),
        candidate_position=int(destination.get("candidate_position", -1)),
        candidate_key=candidate_key,
        candidate_physical_row=int(destination.get("candidate_physical_row", -1)),
        candidate_reference_source_sha256=str(
            destination.get("candidate_reference_source_sha256", "")
        ),
        direction=direction,
        colnomic_query_grid_shape=cq_grid,
        action_table=table,
        deployments_by_root=tuple(deployments_by_root),
    )


def build_control_feature_records(
    raw_query: Mapping[str, object],
    *,
    corrected_identity_by_physical_row: Mapping[int, str],
    destination_dino_valid_by_candidate_key: Mapping[str, torch.Tensor],
    expected_count: int = EXPECTED_CANDIDATE_COUNT,
) -> tuple[object, ...]:
    """Recompute every destination feature row from its cyclic donor content."""

    assert_no_forbidden_prejoin_keys(raw_query)
    records = []
    for destination, donor in _binding_rows(
        raw_query, corrected_identity_by_physical_row, expected_count
    ):
        candidate_key = str(destination["candidate_key"])
        destination_valid = destination_dino_valid_by_candidate_key.get(candidate_key)
        _require(
            isinstance(destination_valid, torch.Tensor),
            "C_COL_P destination DINO valid mask is absent",
        )
        for direction in P_DIRECTIONS:
            records.append(
                _directional_record(
                    raw_query,
                    destination,
                    donor,
                    destination_valid,
                    direction=direction,
                )
            )
    _require(
        len(records) == expected_count * len(P_DIRECTIONS),
        "C_COL_P directional feature population is incomplete",
    )
    return tuple(records)


def _bundle_fingerprint_payload(bundle: VQueryBundle) -> dict[str, object]:
    axis = canonical_candidate_axis(bundle.locks)
    candidates = []
    for key in axis:
        lock = bundle.locks[key]
        candidate = lock.candidate
        directions = []
        for direction in P_DIRECTIONS:
            sealed = lock.direction_locks[direction]
            directions.append(
                {
                    "direction": direction,
                    "record_sha256": sealed.p_lock_record_sha256,
                    "status": sealed.status,
                    "selected_bank_ordinal": sealed.selected_bank_ordinal,
                    "selected_row_sha256": sealed.selected_row_sha256,
                    "query_union_mask_sha256": tensor_sha256(
                        sealed.query_union_mask
                    ),
                    "ordered_root_ordinals": list(sealed.ordered_root_ordinals),
                    "all_roots": [
                        {
                            "root_ordinal": root,
                            "action_key_sha256": scope.action_key_sha256,
                            "query_mask_sha256": tensor_sha256(scope.query_mask),
                            "reference_mask_sha256": tensor_sha256(
                                scope.reference_mask
                            ),
                            "binding_status": scope.binding_status,
                        }
                        for root, scope in sealed.all_roots.items()
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
                "valid_patch_mask_sha256": tensor_sha256(
                    candidate.valid_patch_mask
                ),
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
            "valid_patch_mask_sha256": tensor_sha256(
                bundle.query.valid_patch_mask
            ),
            "grid_shape": list(bundle.query.grid_shape),
        },
        "candidates": candidates,
    }


def bundle_fingerprint(bundle: VQueryBundle) -> str:
    return canonical_sha256(_bundle_fingerprint_payload(bundle))


def dino_axis_content_fingerprint(bundle: VQueryBundle) -> str:
    payload = _bundle_fingerprint_payload(bundle)
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


@dataclass(frozen=True)
class CColPReplay:
    bundle: VQueryBundle
    receipt: Mapping[str, object]
    lock_records: tuple[Mapping[str, object], ...]


def replay_c_col_p(
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
) -> CColPReplay:
    """Run the complete fixed C_COL_P feature -> P -> lock -> V join path."""

    _require(
        raw_source.get("schema_version") == RAW_SOURCE_SCHEMA
        and raw_source.get("target_free") is True,
        "C_COL_P raw P source schema/role drift",
    )
    assert_no_forbidden_prejoin_keys(raw_source)
    queries = [
        _mapping(item, "raw P query")
        for item in _sequence(raw_source.get("queries"), "raw P queries")
        if isinstance(item, Mapping) and str(item.get("query_id", "")) == query_id
    ]
    _require(len(queries) == 1, "C_COL_P requested query is absent or duplicated")
    raw_query = queries[0]
    candidates = _candidate_axis(raw_query, expected_count)
    real_axis = canonical_candidate_axis(real_bundle.locks)
    _require(
        real_bundle.query_id == query_id
        and real_bundle.outer_fold == outer_fold
        and int(raw_query.get("source_fold", -1)) == outer_fold
        and int(raw_query.get("execution_ordinal", -1))
        == real_bundle.execution_ordinal
        and str(raw_query.get("query_source_image_sha256", ""))
        == real_bundle.query.source_image_sha256,
        "C_COL_P query/fold/address drift",
    )
    raw_keys = tuple(str(item["candidate_key"]) for item in candidates)
    _require(
        raw_keys == real_axis,
        "C_COL_P raw and DINO candidate axes are not the same canonical C axis",
    )
    for item in candidates:
        key = str(item["candidate_key"])
        field = real_bundle.locks[key].candidate
        _require(
            int(item["candidate_physical_row"]) == field.physical_gallery_row
            and str(item["candidate_reference_source_sha256"])
            == field.source_image_sha256
            and str(item["reference_geometry_sha256"])
            == field.geometry_record_sha256
            and _grid(item["dino_reference_grid_shape"], "DINO reference")
            == field.grid_shape,
            "C_COL_P ColNomic source and DINO destination binding drift",
        )

    feature_records = build_control_feature_records(
        raw_query,
        corrected_identity_by_physical_row=corrected_identity_by_physical_row,
        destination_dino_valid_by_candidate_key={
            key: real_bundle.locks[key].candidate.valid_patch_mask
            for key in real_axis
        },
        expected_count=expected_count,
    )
    lock_records = tuple(
        build_p_lock_record(
            score_deployable_direction(model, record),
            # Keep all non-intervened lock metadata identical to REAL.  The
            # control identity lives in the outer receipt, never in a field
            # that would make every record hash differ by construction.
            crossfit_role="OUTER_HELDOUT_DEPLOYMENT",
            outer_fold=outer_fold,
            p_checkpoint_sha256=p_checkpoint_sha256,
            p_training_manifest_sha256=p_training_manifest_sha256,
        )
        for record in feature_records
    )
    by_address = {
        (int(item["candidate_physical_row"]), str(item["direction"])): item
        for item in lock_records
    }
    _require(
        len(by_address) == expected_count * len(P_DIRECTIONS),
        "C_COL_P P-lock address population is incomplete",
    )
    locks = {}
    for key in real_axis:
        candidate = real_bundle.locks[key].candidate
        directions = {
            direction: sealed_direction_lock_from_p_record(
                by_address[(candidate.physical_gallery_row, direction)],
                query=real_bundle.query,
                candidate=candidate,
            )
            for direction in P_DIRECTIONS
        }
        locks[key] = CandidatePLockV1(
            candidate=candidate,
            direction_locks=directions,
        )
    controlled = VQueryBundle(
        query_id=real_bundle.query_id,
        execution_ordinal=real_bundle.execution_ordinal,
        outer_fold=real_bundle.outer_fold,
        query=real_bundle.query,
        locks=locks,
        candidate_axis_sha256=real_bundle.candidate_axis_sha256,
    )
    input_dino = dino_axis_content_fingerprint(real_bundle)
    output_dino = dino_axis_content_fingerprint(controlled)
    _require(input_dino == output_dino, "C_COL_P changed DINO query/candidate axis or content")
    real_hashes = [
        real_bundle.locks[key].direction_locks[direction].p_lock_record_sha256
        for key in real_axis
        for direction in P_DIRECTIONS
    ]
    control_hashes = [
        controlled.locks[key].direction_locks[direction].p_lock_record_sha256
        for key in real_axis
        for direction in P_DIRECTIONS
    ]
    changed = sum(first != second for first, second in zip(real_hashes, control_hashes))
    source_receipt = donor_receipt(
        raw_query,
        corrected_identity_by_physical_row,
        expected_count=expected_count,
    )
    receipt = {
        **source_receipt,
        "real_bundle_fingerprint": bundle_fingerprint(real_bundle),
        "control_bundle_fingerprint": bundle_fingerprint(controlled),
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
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return CColPReplay(
        bundle=controlled,
        receipt=MappingProxyType(receipt),
        lock_records=tuple(MappingProxyType(dict(item)) for item in lock_records),
    )


__all__ = [
    "SCHEMA_VERSION",
    "NAMESPACE",
    "CONTROL_INPUT_SCHEMA",
    "CONTROL_VALIDATION_SCHEMA",
    "EXPECTED_CANDIDATE_COUNT",
    "CColPContractError",
    "CColPReplay",
    "canonical_donor_positions",
    "donor_receipt",
    "transport_reference_mask",
    "build_control_feature_records",
    "bundle_fingerprint",
    "dino_axis_content_fingerprint",
    "replay_c_col_p",
]
