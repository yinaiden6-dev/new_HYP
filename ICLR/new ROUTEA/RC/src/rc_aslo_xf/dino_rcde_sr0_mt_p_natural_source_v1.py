"""Full-600 target-free source adapter for the formal SR0-MT P path.

This module does not train a model and does not join labels.  It turns one
immutable full-600 geometry/cache address row into the exact raw structure
consumed by :mod:`materialize_dino_rcde_sr0_mt_p_features_v1`:

* the complete natural C128 axis is retained in canonical order;
* both checkerboard directions and every canonical query-root/reference-action
  coordinate are retained;
* ColNomic masks remain on their native raster;
* deployment masks are produced only by the canonical ColNomic-to-DINO
  geometry mapper; a mapping failure is an exact H0 row with empty masks; and
* no target, identity, score, rank, slot, winner, or outcome is read.

Natural payload access is intentionally confined to ``load_raw_source_shard``.
I1 imports and tests the pure builders below with synthetic sources, but must
never call that loader.  A later, separately authorised execution may call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .cw0_connected_region_v2 import enumerate_query_macro_seeds
from .dino_rcde_cw1_multitile_pseal_e1_v1 import (
    _direction_mask,
    _reference_action_bank,
)
from .dino_rcde_cw1_multitile_pseal_sources_v1 import (
    CW1MultiTilePSealSourceLoaderV1,
    CandidateReferenceSourceV1,
    ColNomicTokenSourceV1,
)
from .dino_rcde_cw1_multitile_superregion_v2 import (
    ROOT_READY,
    _make_root_binding,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    RAW_SOURCE_SCHEMA,
    FormalPContractError,
    assert_no_forbidden_prejoin_keys,
    canonical_sha256,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_natural_source_v1_20260815"
NATURAL_SOURCE_MANIFEST_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_natural_source_manifest_v1_20260815"
)
EXPECTED_QUERY_COUNT = 600
EXPECTED_CANDIDATE_COUNT = 128
EXPECTED_OUTER_FOLDS = frozenset({1, 2, 3, 4})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalPContractError(message)


def candidate_key_v1(*, physical_row: int, source_image_sha256: str) -> str:
    """Return the only candidate address used by P, V, and three-arm scoring."""

    _require(
        isinstance(physical_row, int)
        and not isinstance(physical_row, bool)
        and physical_row >= 0,
        "candidate physical row drift",
    )
    _require(
        isinstance(source_image_sha256, str) and len(source_image_sha256) == 64,
        "candidate source hash drift",
    )
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "physical_row": physical_row,
            "source_image_sha256": source_image_sha256,
        }
    )


def _action_key(
    *,
    candidate_key: str,
    direction: str,
    root_ordinal: int,
    action_ordinal: int,
    structural_action_sha256: str,
    query_geometry_sha256: str,
    reference_geometry_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_key": candidate_key,
            "direction": direction,
            "root_ordinal": root_ordinal,
            "action_ordinal": action_ordinal,
            "structural_action_sha256": structural_action_sha256,
            "query_geometry_sha256": query_geometry_sha256,
            "reference_geometry_sha256": reference_geometry_sha256,
        }
    )


def _candidate_directions(
    *,
    candidate_key: str,
    query: ColNomicTokenSourceV1,
    reference: ColNomicTokenSourceV1,
) -> dict[str, list[dict[str, object]]]:
    _require(query.kind == "query", "P natural query source kind drift")
    _require(reference.kind == "reference", "P natural reference source kind drift")
    roots = enumerate_query_macro_seeds(query.colnomic_geometry.grid_shape)
    output: dict[str, list[dict[str, object]]] = {}
    for direction in P_DIRECTIONS:
        action_bank, action_bank_sha256 = _reference_action_bank(
            grid_shape=reference.colnomic_geometry.grid_shape,
            valid_patch_mask=reference.colnomic_geometry.valid_patch_mask,
            direction=direction,
        )
        root_rows: list[dict[str, object]] = []
        for root_ordinal, root in enumerate(roots):
            full_query_root = (
                root.window.mask & query.colnomic_geometry.valid_patch_mask
            ).detach().cpu().contiguous()
            directional_query_root = _direction_mask(
                full_query_root,
                query.colnomic_geometry.grid_shape,
                direction,
            )
            actions: list[dict[str, object]] = []
            if not action_bank:
                missing = _make_root_binding(
                    candidate_key=candidate_key,
                    direction=direction,
                    root_ordinal=root_ordinal,
                    source_reference_component=None,
                    colnomic_query_geometry=query.colnomic_geometry,
                    colnomic_reference_geometry=reference.colnomic_geometry,
                    dino_query_geometry=query.dino_geometry,
                    dino_reference_geometry=reference.dino_geometry,
                )
                actions.append(
                    {
                        "action_key": _action_key(
                            candidate_key=candidate_key,
                            direction=direction,
                            root_ordinal=root_ordinal,
                            action_ordinal=0,
                            structural_action_sha256=action_bank_sha256,
                            query_geometry_sha256=query.dino_geometry.sha256,
                            reference_geometry_sha256=reference.dino_geometry.sha256,
                        ),
                        "colnomic_query_action_mask": torch.zeros_like(
                            directional_query_root
                        ),
                        "colnomic_reference_action_mask": torch.zeros_like(
                            reference.colnomic_geometry.valid_patch_mask
                        ),
                        "deployment_eligible": False,
                        "deployment_query_mask": torch.zeros_like(
                            missing.dino_query_tile_mask
                        ),
                        "deployment_reference_mask": torch.zeros_like(
                            missing.dino_reference_component_mask
                        ),
                    }
                )
            else:
                for action in action_bank:
                    binding = _make_root_binding(
                        candidate_key=candidate_key,
                        direction=direction,
                        root_ordinal=root_ordinal,
                        source_reference_component=action.component_mask,
                        colnomic_query_geometry=query.colnomic_geometry,
                        colnomic_reference_geometry=reference.colnomic_geometry,
                        dino_query_geometry=query.dino_geometry,
                        dino_reference_geometry=reference.dino_geometry,
                    )
                    deployment_eligible = binding.status == ROOT_READY
                    actions.append(
                        {
                            "action_key": _action_key(
                                candidate_key=candidate_key,
                                direction=direction,
                                root_ordinal=root_ordinal,
                                action_ordinal=action.action_ordinal,
                                structural_action_sha256=action.action_sha256,
                                query_geometry_sha256=query.dino_geometry.sha256,
                                reference_geometry_sha256=reference.dino_geometry.sha256,
                            ),
                            "colnomic_query_action_mask": directional_query_root,
                            "colnomic_reference_action_mask": action.selector_mask,
                            "deployment_eligible": deployment_eligible,
                            "deployment_query_mask": (
                                binding.dino_query_tile_mask
                                if deployment_eligible
                                else torch.zeros_like(binding.dino_query_tile_mask)
                            ),
                            "deployment_reference_mask": (
                                binding.dino_reference_component_mask
                                if deployment_eligible
                                else torch.zeros_like(
                                    binding.dino_reference_component_mask
                                )
                            ),
                        }
                    )
            root_rows.append(
                {
                    "root_ordinal": root_ordinal,
                    "reference_action_bank_sha256": action_bank_sha256,
                    "actions": actions,
                }
            )
        output[direction] = root_rows
    return output


def build_target_free_raw_query(
    *,
    query_id: str,
    historical_query_ordinal: int,
    execution_ordinal: int,
    heldout_fold: int,
    candidate_axis_sha256: str,
    query: ColNomicTokenSourceV1,
    candidates: Sequence[CandidateReferenceSourceV1],
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
) -> dict[str, object]:
    """Build one complete target-free raw query without a label-side join."""

    _require(isinstance(query_id, str) and query_id, "query id absent")
    _require(
        heldout_fold in EXPECTED_OUTER_FOLDS,
        "target-free source fold is outside frozen outer folds",
    )
    _require(
        query.kind == "query" and query.native_key == historical_query_ordinal,
        "query source address drift",
    )
    ordered = tuple(candidates)
    _require(
        len(ordered) == expected_candidate_count
        and tuple(item.candidate_position for item in ordered)
        == tuple(range(expected_candidate_count))
        and len({item.physical_row for item in ordered}) == expected_candidate_count,
        "natural candidate axis is incomplete, reordered, or duplicated",
    )
    candidate_rows: list[dict[str, object]] = []
    for item in ordered:
        reference = item.source
        key = candidate_key_v1(
            physical_row=item.physical_row,
            source_image_sha256=reference.source_image_sha256,
        )
        candidate_rows.append(
            {
                "candidate_position": item.candidate_position,
                "candidate_key": key,
                "candidate_physical_row": item.physical_row,
                "candidate_reference_source_sha256": reference.source_image_sha256,
                "reference_tokens": reference.tokens,
                "reference_valid_patch_mask": (
                    reference.colnomic_geometry.valid_patch_mask
                ),
                "colnomic_reference_grid_shape": list(reference.grid_shape),
                "dino_reference_grid_shape": list(
                    reference.dino_geometry.grid_shape
                ),
                "reference_geometry_sha256": reference.dino_geometry.sha256,
                "directions": _candidate_directions(
                    candidate_key=key,
                    query=query,
                    reference=reference,
                ),
            }
        )
    result: dict[str, object] = {
        "query_id": query_id,
        "historical_query_ordinal": historical_query_ordinal,
        "execution_ordinal": execution_ordinal,
        "query_source_image_sha256": query.source_image_sha256,
        "source_fold": heldout_fold,
        "candidate_axis_sha256": candidate_axis_sha256,
        "query_tokens": query.tokens,
        "query_valid_patch_mask": query.colnomic_geometry.valid_patch_mask,
        "colnomic_query_grid_shape": list(query.grid_shape),
        "dino_query_grid_shape": list(query.dino_geometry.grid_shape),
        "query_geometry_sha256": query.dino_geometry.sha256,
        "candidates": candidate_rows,
    }
    assert_no_forbidden_prejoin_keys(result)
    return result


def raw_source_from_queries(
    queries: Sequence[Mapping[str, object]],
    *,
    candidate_count: int,
    source_address_sha256: str,
) -> dict[str, object]:
    ordered = list(queries)
    _require(bool(ordered), "target-free raw source shard is empty")
    _require(
        [int(item["execution_ordinal"]) for item in ordered]
        == sorted(int(item["execution_ordinal"]) for item in ordered),
        "target-free source execution order drift",
    )
    value: dict[str, object] = {
        "schema_version": RAW_SOURCE_SCHEMA,
        "target_free": True,
        "synthetic": False,
        "candidate_count_per_query": candidate_count,
        "source_address_sha256": source_address_sha256,
        "queries": ordered,
    }
    value["logical_sha256"] = canonical_sha256(
        {
            "schema_version": RAW_SOURCE_SCHEMA,
            "target_free": True,
            "candidate_count_per_query": candidate_count,
            "source_address_sha256": source_address_sha256,
            "query_addresses": [
                {
                    "query_id": item["query_id"],
                    "historical_query_ordinal": item["historical_query_ordinal"],
                    "execution_ordinal": item["execution_ordinal"],
                    "query_source_image_sha256": item[
                        "query_source_image_sha256"
                    ],
                    "source_fold": item["source_fold"],
                    "candidate_axis_sha256": item["candidate_axis_sha256"],
                    "candidate_keys": [
                        row["candidate_key"] for row in item["candidates"]
                    ],
                }
                for item in ordered
            ],
        }
    )
    assert_no_forbidden_prejoin_keys(value)
    return value


def load_raw_source_shard(
    manifest: Mapping[str, object],
    *,
    rc_root: str | None = None,
) -> dict[str, object]:
    """Load a later-authorised natural shard from immutable geometry addresses.

    This is the sole natural token-reading entry point.  Callers must enforce
    the live authority before invoking it; I1 is limited to importing and
    synthetic-testing this function.
    """

    from pathlib import Path

    from .dino_rcde_cw1_multitile_pseal_sources_v1 import sha256_file

    _require(
        manifest.get("schema_version") == NATURAL_SOURCE_MANIFEST_SCHEMA
        and manifest.get("target_free") is True,
        "natural P source manifest schema drift",
    )
    expected_logical = canonical_sha256(
        {key: item for key, item in manifest.items() if key != "logical_sha256"}
    )
    _require(
        manifest.get("logical_sha256") == expected_logical,
        "natural P source manifest logical hash drift",
    )
    geometry_path = Path(str(manifest.get("geometry_payload"))).resolve()
    geometry_sha256 = str(manifest.get("geometry_payload_sha256"))
    _require(
        sha256_file(geometry_path) == geometry_sha256,
        "natural P geometry payload file hash drift",
    )
    start = int(manifest.get("execution_start", -1))
    stop = int(manifest.get("execution_stop", -1))
    _require(
        0 <= start < stop <= EXPECTED_QUERY_COUNT,
        "natural P source execution shard bounds drift",
    )
    root = Path(rc_root).resolve() if rc_root is not None else None
    loader = CW1MultiTilePSealSourceLoaderV1(
        payload_path=geometry_path,
        rc_root=root,
        expected_payload_file_sha256=geometry_sha256,
    )
    payload = loader._load_payload()  # canonical loader; later authority only
    query_records = payload.get("query_records")
    reference_records = payload.get("reference_records")
    _require(
        isinstance(query_records, list)
        and len(query_records) == EXPECTED_QUERY_COUNT
        and isinstance(reference_records, list),
        "full-600 source population drift",
    )
    query_by_execution = {
        int(item["execution_ordinal"]): item
        for item in query_records
        if isinstance(item, Mapping)
    }
    reference_by_row = {
        int(item["physical_row"]): item
        for item in reference_records
        if isinstance(item, Mapping)
    }
    _require(
        set(query_by_execution) == set(range(EXPECTED_QUERY_COUNT)),
        "full-600 execution addressing drift",
    )
    output: list[dict[str, object]] = []
    for execution_ordinal in range(start, stop):
        record = query_by_execution[execution_ordinal]
        historical = int(record["historical_query_ordinal"])
        axis = list(map(int, record["candidate_physical_rows"]))
        _require(
            len(axis) == EXPECTED_CANDIDATE_COUNT
            and len(set(axis)) == EXPECTED_CANDIDATE_COUNT,
            "natural C128 address drift",
        )
        query_source = loader._build_source(
            record=record,
            kind="query",
            native_key=historical,
        )
        candidates: list[CandidateReferenceSourceV1] = []
        for position, physical_row in enumerate(axis):
            _require(
                physical_row in reference_by_row,
                "candidate row absent from reference geometry union",
            )
            if physical_row not in loader._reference_sources:
                loader._reference_sources[physical_row] = loader._build_source(
                    record=reference_by_row[physical_row],
                    kind="reference",
                    native_key=physical_row,
                )
            candidates.append(
                CandidateReferenceSourceV1(
                    candidate_position=position,
                    physical_row=physical_row,
                    source=loader._reference_sources[physical_row],
                )
            )
        output.append(
            build_target_free_raw_query(
                query_id=str(record["query_id"]),
                historical_query_ordinal=historical,
                execution_ordinal=execution_ordinal,
                heldout_fold=int(record["heldout_fold"]),
                candidate_axis_sha256=str(record["candidate_axis_sha256"]),
                query=query_source,
                candidates=candidates,
            )
        )
    return raw_source_from_queries(
        output,
        candidate_count=EXPECTED_CANDIDATE_COUNT,
        source_address_sha256=expected_logical,
    )


__all__ = [
    "SCHEMA_VERSION",
    "NATURAL_SOURCE_MANIFEST_SCHEMA",
    "EXPECTED_QUERY_COUNT",
    "EXPECTED_CANDIDATE_COUNT",
    "candidate_key_v1",
    "build_target_free_raw_query",
    "raw_source_from_queries",
    "load_raw_source_shard",
]
