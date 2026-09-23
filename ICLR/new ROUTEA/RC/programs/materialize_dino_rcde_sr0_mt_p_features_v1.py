#!/usr/bin/env python3
"""Materialize the formal target-free SR0-MT ColNomic P feature ledger.

The input is an explicit tensor-safe source payload.  It contains frozen
ColNomic tokens, canonical action masks, and geometry-mapped deployment masks;
it must not contain labels or retrieval outcomes.  Every query retains the
complete candidate axis and both checkerboard directions.  The output contains
features and discrete deployment support, but no source tokens.

Natural execution requires a separately frozen authority.  ``--synthetic`` is
provided only for I1 engineering qualification and never emits a scientific
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))

from rc_aslo_xf.cw0_connected_region_v2 import enumerate_query_macro_seeds  # noqa: E402
from rc_aslo_xf.dino_rcde_cw1_multitile_sr0_p_v1 import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_SHA256,
    CandidateActionFeatureTable,
    candidate_action_colnomic_features,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    FEATURE_LEDGER_SCHEMA,
    P_DIRECTIONS,
    RAW_SOURCE_SCHEMA,
    VECTOR_SCALAR_MAX_ULPS,
    FormalPContractError,
    assert_no_forbidden_prejoin_keys,
    argv_from_execution_manifest,
    canonical_sha256,
    exclusive_json,
    exclusive_torch,
    execution_manifest_file_sha256,
    feature_ledger_index,
    feature_ledger_logical_sha256,
    file_sha256,
    load_torch_mapping,
    load_json_mapping,
    make_directional_feature_record,
    make_root_action_deployment,
    serialize_directional_feature_record,
    tensor_sha256,
    validate_natural_authority,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_source_v1 import (  # noqa: E402
    NATURAL_SOURCE_MANIFEST_SCHEMA,
    load_raw_source_shard,
)


RESULT_SCHEMA = "rc_dino_rcde_sr0_mt_p_feature_materialization_result_v1_20260815"
READY = "RCDE_SR0_MT_P_FEATURE_LEDGER_READY"
CLAIM = "ENGINEERING_TARGET_FREE_P_FEATURE_MATERIALIZATION_ONLY"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FormalPContractError(f"{name} must be a mapping")
    return value


def _sequence(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise FormalPContractError(f"{name} must be a list")
    return value


def _grid(value: object, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or min(value) <= 0
    ):
        raise FormalPContractError(f"{name} must be a positive grid shape")
    return int(value[0]), int(value[1])


def _source_feature_record(
    query: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    direction: str,
) -> object:
    q_tokens = torch.as_tensor(query["query_tokens"])
    r_tokens = torch.as_tensor(candidate["reference_tokens"])
    q_valid = torch.as_tensor(query["query_valid_patch_mask"], dtype=torch.bool)
    r_valid = torch.as_tensor(candidate["reference_valid_patch_mask"], dtype=torch.bool)
    cq_shape = _grid(query["colnomic_query_grid_shape"], name="ColNomic query grid")
    dq_shape = _grid(query["dino_query_grid_shape"], name="DINO query grid")
    dr_shape = _grid(candidate["dino_reference_grid_shape"], name="DINO reference grid")
    if q_tokens.shape[0] != math.prod(cq_shape):
        raise FormalPContractError("query token/grid shape drift")
    cr_shape = _grid(
        candidate["colnomic_reference_grid_shape"], name="ColNomic reference grid"
    )
    if r_tokens.shape[0] != math.prod(cr_shape):
        raise FormalPContractError("reference token/grid shape drift")
    directions = _mapping(candidate.get("directions"), name="candidate directions")
    raw_roots = _sequence(directions.get(direction), name=f"{direction} roots")
    expected_roots = len(enumerate_query_macro_seeds(cq_shape))
    if len(raw_roots) != expected_roots:
        raise FormalPContractError("source does not retain every canonical P root")
    keys_by_root: list[tuple[str, ...]] = []
    features_by_root: list[torch.Tensor] = []
    eligible_by_root: list[torch.Tensor] = []
    deployments_by_root = []
    for root_ordinal, raw_root in enumerate(raw_roots):
        root = _mapping(raw_root, name="source root")
        if root.get("root_ordinal") != root_ordinal:
            raise FormalPContractError("source root ordinal drift")
        actions = _sequence(root.get("actions"), name="source actions")
        if not actions:
            raise FormalPContractError("every P root must retain at least one action/H0 row")
        root_keys = []
        root_features = []
        root_eligible = []
        root_deployments = []
        for raw_action in actions:
            action = _mapping(raw_action, name="source action")
            key = str(action["action_key"])
            deployment_eligible = bool(action["deployment_eligible"])
            deployment = make_root_action_deployment(
                action_key=key,
                eligible=deployment_eligible,
                query_mask=torch.as_tensor(action["deployment_query_mask"], dtype=torch.bool),
                reference_mask=torch.as_tensor(
                    action["deployment_reference_mask"], dtype=torch.bool
                ),
                query_grid_shape=dq_shape,
                reference_grid_shape=dr_shape,
                query_geometry_sha256=str(query["query_geometry_sha256"]),
                reference_geometry_sha256=str(candidate["reference_geometry_sha256"]),
            )
            if deployment_eligible:
                feature = candidate_action_colnomic_features(
                    q_tokens,
                    r_tokens,
                    query_action_mask=torch.as_tensor(
                        action["colnomic_query_action_mask"], dtype=torch.bool
                    ),
                    reference_action_mask=torch.as_tensor(
                        action["colnomic_reference_action_mask"], dtype=torch.bool
                    ),
                    query_valid_patch_mask=q_valid,
                    reference_valid_patch_mask=r_valid,
                )
            else:
                feature = torch.zeros(FEATURE_DIM, dtype=torch.float64)
            root_keys.append(key)
            root_features.append(feature)
            root_eligible.append(deployment_eligible)
            root_deployments.append(deployment)
        keys_by_root.append(tuple(root_keys))
        features_by_root.append(torch.stack(root_features).to(torch.float64))
        eligible_by_root.append(torch.tensor(root_eligible, dtype=torch.bool))
        deployments_by_root.append(tuple(root_deployments))
    table = CandidateActionFeatureTable(
        candidate_key=str(candidate["candidate_key"]),
        root_ordinals=tuple(range(expected_roots)),
        action_keys_by_root=tuple(keys_by_root),
        features_by_root=tuple(features_by_root),
        eligible_by_root=tuple(eligible_by_root),
    )
    return make_directional_feature_record(
        query_id=str(query["query_id"]),
        historical_query_ordinal=int(query["historical_query_ordinal"]),
        execution_ordinal=int(query["execution_ordinal"]),
        query_source_image_sha256=str(query["query_source_image_sha256"]),
        source_fold=int(query["source_fold"]),
        candidate_position=int(candidate["candidate_position"]),
        candidate_key=str(candidate["candidate_key"]),
        candidate_physical_row=int(candidate["candidate_physical_row"]),
        candidate_reference_source_sha256=str(
            candidate["candidate_reference_source_sha256"]
        ),
        direction=direction,
        colnomic_query_grid_shape=cq_shape,
        action_table=table,
        deployments_by_root=tuple(deployments_by_root),
    )


def materialize_source_payload(
    source: Mapping[str, object],
    *,
    source_file_sha256: str,
    source_role: str,
) -> dict[str, object]:
    if source.get("schema_version") != RAW_SOURCE_SCHEMA or source.get("target_free") is not True:
        raise FormalPContractError("raw P source schema/target-free contract drift")
    assert_no_forbidden_prejoin_keys(source)
    queries = _sequence(source.get("queries"), name="source queries")
    expected_candidates = int(source.get("candidate_count_per_query", -1))
    if not queries or expected_candidates <= 0:
        raise FormalPContractError("raw P source population is empty")
    records = []
    query_axes = []
    for raw_query in queries:
        query = _mapping(raw_query, name="source query")
        candidates = _sequence(query.get("candidates"), name="query candidates")
        if len(candidates) != expected_candidates:
            raise FormalPContractError("query did not retain the complete candidate axis")
        positions = []
        candidate_keys = []
        candidate_physical_rows = []
        for position, raw_candidate in enumerate(candidates):
            candidate = _mapping(raw_candidate, name="source candidate")
            if candidate.get("candidate_position") != position:
                raise FormalPContractError("candidate axis is not canonical 0..C-1")
            positions.append(position)
            candidate_keys.append(str(candidate["candidate_key"]))
            candidate_physical_rows.append(int(candidate["candidate_physical_row"]))
            for direction in P_DIRECTIONS:
                records.append(
                    serialize_directional_feature_record(
                        _source_feature_record(query, candidate, direction=direction)
                    )
                )
        physical_axis_sha256 = str(
            query.get(
                "candidate_axis_sha256",
                canonical_sha256(candidate_physical_rows),
            )
        )
        if physical_axis_sha256 != canonical_sha256(candidate_physical_rows):
            raise FormalPContractError("physical candidate-axis hash drift")
        query_axes.append(
            {
                "query_id": str(query["query_id"]),
                "execution_ordinal": int(query["execution_ordinal"]),
                "candidate_positions": positions,
                "candidate_physical_rows": candidate_physical_rows,
                "candidate_axis_sha256": physical_axis_sha256,
                "candidate_key_axis_sha256": canonical_sha256(candidate_keys),
            }
        )
    records.sort(
        key=lambda item: (
            int(item["execution_ordinal"]),
            int(item["candidate_position"]),
            P_DIRECTIONS.index(str(item["direction"])),
        )
    )
    ledger: dict[str, object] = {
        "schema_version": FEATURE_LEDGER_SCHEMA,
        "claim_level": CLAIM,
        "target_free": True,
        "source_role": source_role,
        "source_file_sha256": source_file_sha256,
        "source_logical_sha256": str(source.get("logical_sha256", canonical_sha256(query_axes))),
        "feature_schema": list(FEATURE_SCHEMA),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_dtype": "torch.float64",
        "vector_scalar_max_ulps": VECTOR_SCALAR_MAX_ULPS,
        "query_count": len(queries),
        "candidate_count_per_query": expected_candidates,
        "direction_count": len(P_DIRECTIONS),
        "record_count": len(records),
        "query_candidate_axes": query_axes,
        "records": records,
        "protected_access_audit": {
            "target_label_read_count": 0,
            "D1_or_rank_read_count": 0,
            "dino_token_read_count": 0,
            "model_load_count": 0,
            "model_forward_count": 0,
            "model_update_count": 0,
            "target_insertion_count": 0,
        },
    }
    ledger["logical_sha256"] = feature_ledger_logical_sha256(ledger)
    feature_ledger_index(ledger)
    return ledger


def synthetic_source(*, query_count: int = 2, candidate_count: int = 3) -> dict[str, object]:
    """Small deterministic source used only by I1/tests."""

    generator = torch.Generator().manual_seed(1701)
    shape = (8, 8)
    roots = enumerate_query_macro_seeds(shape)
    queries = []
    for query_ordinal in range(query_count):
        q_tokens = torch.randn((64, 12), generator=generator, dtype=torch.float32)
        candidates = []
        for candidate_position in range(candidate_count):
            r_tokens = q_tokens.roll(candidate_position + 1, dims=0) + 0.01 * torch.randn(
                (64, 12), generator=generator
            )
            directions = {}
            for direction_index, direction in enumerate(P_DIRECTIONS):
                root_rows = []
                for root_ordinal, root in enumerate(roots):
                    query_mask = root.window.mask.clone()
                    # Keep the synthetic deployment component connected.  The
                    # candidate/direction distinction is carried by tokens and
                    # action addresses, not by a wrap-around flat-grid roll.
                    reference_mask = query_mask.clone()
                    root_rows.append(
                        {
                            "root_ordinal": root_ordinal,
                            "actions": [
                                {
                                    "action_key": _sha(
                                        f"q{query_ordinal}:c{candidate_position}:{direction}:r{root_ordinal}"
                                    ),
                                    "colnomic_query_action_mask": query_mask,
                                    "colnomic_reference_action_mask": reference_mask,
                                    "deployment_eligible": True,
                                    "deployment_query_mask": query_mask,
                                    "deployment_reference_mask": reference_mask,
                                }
                            ],
                        }
                    )
                directions[direction] = root_rows
            candidates.append(
                {
                    "candidate_position": candidate_position,
                    "candidate_key": _sha(f"candidate-{query_ordinal}-{candidate_position}"),
                    "candidate_physical_row": 1000 + candidate_position,
                    "candidate_reference_source_sha256": _sha(
                        f"reference-source-{candidate_position}"
                    ),
                    "reference_tokens": r_tokens,
                    "reference_valid_patch_mask": torch.ones(64, dtype=torch.bool),
                    "colnomic_reference_grid_shape": list(shape),
                    "dino_reference_grid_shape": list(shape),
                    "reference_geometry_sha256": _sha(
                        f"reference-geometry-{candidate_position}"
                    ),
                    "directions": directions,
                }
            )
        queries.append(
            {
                "query_id": f"SYNTH-{query_ordinal:04d}",
                "historical_query_ordinal": query_ordinal,
                "execution_ordinal": query_ordinal,
                "query_source_image_sha256": _sha(f"query-source-{query_ordinal}"),
                "source_fold": 1 + query_ordinal % 4,
                "query_tokens": q_tokens,
                "query_valid_patch_mask": torch.ones(64, dtype=torch.bool),
                "colnomic_query_grid_shape": list(shape),
                "dino_query_grid_shape": list(shape),
                "query_geometry_sha256": _sha(f"query-geometry-{query_ordinal}"),
                "candidate_axis_sha256": canonical_sha256(
                    [1000 + position for position in range(candidate_count)]
                ),
                "candidates": candidates,
            }
        )
    source: dict[str, object] = {
        "schema_version": RAW_SOURCE_SCHEMA,
        "target_free": True,
        "synthetic": True,
        "candidate_count_per_query": candidate_count,
        "queries": queries,
    }
    source["logical_sha256"] = canonical_sha256(
        {
            "schema_version": RAW_SOURCE_SCHEMA,
            "query_count": query_count,
            "candidate_count": candidate_count,
            "seed": 1701,
        }
    )
    return source


def main() -> None:
    invocation_manifest_sha256 = execution_manifest_file_sha256(
        sys.argv[1:], expected_program=Path(__file__).name
    )
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=Path)
    source_group.add_argument("--natural-source-manifest", type=Path)
    source_group.add_argument("--synthetic", action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--natural-source-manifest-sha256")
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--authority-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--synthetic-query-count", type=int, default=2)
    parser.add_argument("--synthetic-candidate-count", type=int, default=3)
    args = parser.parse_args(
        argv_from_execution_manifest(
            sys.argv[1:], expected_program=Path(__file__).name
        )
    )

    if args.synthetic:
        source = synthetic_source(
            query_count=args.synthetic_query_count,
            candidate_count=args.synthetic_candidate_count,
        )
        source_file_hash = canonical_sha256(
            {
                "synthetic_query_count": args.synthetic_query_count,
                "synthetic_candidate_count": args.synthetic_candidate_count,
            }
        )
        source_role = "I1_SYNTHETIC_ONLY"
    elif args.natural_source_manifest is not None:
        if (
            args.natural_source_manifest_sha256 is None
            or file_sha256(args.natural_source_manifest)
            != args.natural_source_manifest_sha256
        ):
            raise SystemExit("natural source manifest SHA256 is absent or drifted")
        if args.authority is None or args.authority_sha256 is None:
            raise SystemExit("natural source execution requires a bound authority")
        validate_natural_authority(
            args.authority,
            expected_file_sha256=args.authority_sha256,
            required_scope="natural_p_feature_materialization",
        )
        source_manifest = load_json_mapping(
            args.natural_source_manifest, name="natural P source manifest"
        )
        if source_manifest.get("schema_version") != NATURAL_SOURCE_MANIFEST_SCHEMA:
            raise SystemExit("natural P source manifest schema drift")
        source = load_raw_source_shard(source_manifest, rc_root=str(RC_ROOT))
        source_file_hash = args.natural_source_manifest_sha256
        source_role = "FORMAL_TARGET_FREE_NATURAL_SOURCE"
    else:
        if args.authority is None or args.authority_sha256 is None:
            raise SystemExit("natural source execution requires a bound authority")
        validate_natural_authority(
            args.authority,
            expected_file_sha256=args.authority_sha256,
            required_scope="natural_p_feature_materialization",
        )
        if args.source_sha256 is None or file_sha256(args.source) != args.source_sha256:
            raise SystemExit("raw source file SHA256 is absent or drifted")
        source = load_torch_mapping(args.source, name="raw target-free P source")
        source_file_hash = args.source_sha256
        source_role = "FORMAL_TARGET_FREE_NATURAL_SOURCE"
    ledger = materialize_source_payload(
        source, source_file_sha256=source_file_hash, source_role=source_role
    )
    ledger["execution_manifest_sha256"] = invocation_manifest_sha256
    ledger["logical_sha256"] = feature_ledger_logical_sha256(ledger)
    if (
        source_role == "FORMAL_TARGET_FREE_NATURAL_SOURCE"
        and invocation_manifest_sha256 is None
    ):
        raise SystemExit("formal P feature materialization requires a physical execution manifest")
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"immutable output root exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    exclusive_torch(output / "feature_ledger.pt", ledger)
    receipt: dict[str, object] = {
        "schema_version": RESULT_SCHEMA,
        "status": READY,
        "claim_level": CLAIM,
        "scientific_GO_or_NO_GO": None,
        "feature_ledger_file_sha256": file_sha256(output / "feature_ledger.pt"),
        "feature_ledger_logical_sha256": ledger["logical_sha256"],
        "query_count": ledger["query_count"],
        "candidate_count_per_query": ledger["candidate_count_per_query"],
        "record_count": ledger["record_count"],
        "source_role": source_role,
        "execution_manifest_sha256": invocation_manifest_sha256,
        "next_authorized_stage": None,
        "automatic_stage_advance": False,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    exclusive_json(output / "result.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
