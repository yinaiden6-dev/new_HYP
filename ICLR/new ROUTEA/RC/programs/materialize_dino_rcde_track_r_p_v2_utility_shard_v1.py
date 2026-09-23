#!/usr/bin/env python3
"""Materialize target-free outer-refit P-V2 utilities for one 12-query shard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
sys.path.insert(0, str(RC_ROOT / "programs"))

from rc_aslo_xf.dino_rcde_cw1_multitile_sr0_p_v1 import SharedMultitilePHead  # noqa: E402
from rc_aslo_xf.dino_rcde_cw1_multitile_pseal_sources_v1 import (  # noqa: E402
    CW1MultiTilePSealSourceLoaderV1,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_compact_catalog_v1 import (  # noqa: E402
    P_DIRECTIONS,
    deserialize_factorized_directional_source_index,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_compact_lock_fanout_v1 import (  # noqa: E402
    CompactDirectionalFeatureInput,
    crossfit_fanout_scopes,
    decision_from_compact_output,
    score_compact_deployable_direction,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_source_v1 import candidate_key_v1  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    fuse_direction_utilities,
    tensor_sha256,
)

from materialize_dino_rcde_sr0_mt_role_free_pair_feature_cache_shard_v1 import (  # noqa: E402
    BatchedNaturalPairFeatureProducer,
    project_token_record,
)
from dino_rcde_track_r_p_v2_utility_runtime_v1 import (  # noqa: E402
    source_bindings as runtime_source_bindings,
    validate_authority as validate_v122_authority,
)


AUTHORITY_PATH = "registry/current_authority_v122_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v122_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARDS_AUTHORIZED"
OUTPUT_SCHEMA = "rc_dino_rcde_track_r_p_v2_utility_shard_v1_20260821"
OUTPUT_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARD_READY"
SHARD_SIZE = 12
SHARD_COUNT = 50
QUERY_COUNT = 600
EXCLUDED = (25, 26, 101, 346, 354, 470)


class TrackRPUtilityError(RuntimeError):
    pass


class QueryResolver:
    """Per-query token resolver backed by one shared full600 payload load."""

    def __init__(
        self,
        loader: CW1MultiTilePSealSourceLoaderV1,
        query_record: Mapping[str, Any],
        reference_records: Mapping[int, Mapping[str, Any]],
    ) -> None:
        self.loader = loader
        self.query_record = query_record
        self.reference_records = reference_records
        self.query = None
        self.references: dict[int, Any] = {}

    def __call__(self, source):
        if self.query is None:
            self.query = self.loader._build_source(
                record=project_token_record(self.query_record, kind="query"),
                kind="query",
                native_key=source.historical_query_ordinal,
            )
        row = source.candidate_physical_row
        if row not in self.references:
            self.references[row] = self.loader._build_source(
                record=project_token_record(self.reference_records[row], kind="reference"),
                kind="reference",
                native_key=row,
            )
        query = self.query
        reference = self.references[row]
        require(
            candidate_key_v1(
                physical_row=row,
                source_image_sha256=reference.source_image_sha256,
            )
            == source.candidate_key,
            "P utility candidate key drift",
        )
        require(
            tensor_sha256(query.tokens) == source.query_token_sha256
            and tensor_sha256(reference.tokens) == source.reference_token_sha256,
            "P utility token hash drift",
        )
        return (
            query.tokens,
            reference.tokens,
            query.colnomic_geometry.valid_patch_mask,
            reference.colnomic_geometry.valid_patch_mask,
        )

    def release_candidate(self, row: int) -> None:
        self.references.pop(row, None)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRPUtilityError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def safe_path(path: Path, *, must_exist: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    if must_exist:
        require(value.is_file() and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(isinstance(item, Mapping) and set(item) >= {"path", "sha256"}, f"binding absent: {name}")
    path = safe_path(RC_ROOT / str(item["path"]))
    require(file_sha256(path) == item["sha256"], f"binding hash drift: {name}")
    return path


def atomic_torch(path: Path, value: object) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), "immutable P utility shard exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, output)
    output.chmod(0o444)


def materialize(
    authority_path: Path,
    shard_ordinal: int,
    output_root: Path,
    *,
    replay_only: bool = False,
) -> dict[str, Any]:
    require(0 <= shard_ordinal < SHARD_COUNT, "shard ordinal drift")
    authority_path = safe_path(authority_path)
    authority, authority_sha = validate_v122_authority(
        authority_path,
        required_flag="p_v2_utility_materialization_authorized",
        runtime_binding="producer",
        runtime_path="programs/materialize_dino_rcde_track_r_p_v2_utility_shard_v1.py",
    )
    expected_output_root = (RC_ROOT / str(authority["producer_root"])).resolve(strict=False)
    require(safe_path(output_root, must_exist=False) == expected_output_root, "V122 producer root drift")
    manifest_index = read_json(binding_path(authority, "lock_manifest_index"))
    folds = read_json(binding_path(authority, "fold_schedule"))
    fold_records = tuple(folds["records"])
    require(
        len(fold_records) == QUERY_COUNT
        and len({int(item["query_ordinal"]) for item in fold_records}) == QUERY_COUNT,
        "P utility fold record population/historical-ordinal drift",
    )
    fold_by_execution = dict(enumerate(fold_records))
    start = shard_ordinal * SHARD_SIZE
    stop = min(QUERY_COUNT, start + SHARD_SIZE)
    shard_manifest_rows = {
        int(item["shard_ordinal"]): item for item in manifest_index["shards"]
    }
    source_manifest_row = shard_manifest_rows[start // SHARD_SIZE]
    source_manifest_path = safe_path(RC_ROOT / str(source_manifest_row["path"]))
    require(file_sha256(source_manifest_path) == source_manifest_row["sha256"], "source manifest hash drift")
    source_manifest = read_json(source_manifest_path)
    query_manifest = {
        int(item["execution_ordinal"]): item
        for item in source_manifest["selected_queries"]
    }
    source_artifact_path = safe_path(RC_ROOT / str(source_manifest["source_artifact_path"]))
    require(file_sha256(source_artifact_path) == source_manifest["source_artifact_sha256"], "compact source artifact hash drift")
    raw = torch.load(
        source_artifact_path, map_location="cpu", weights_only=True, mmap=True
    )
    raw_records = raw.get("records") if isinstance(raw, Mapping) else None
    require(isinstance(raw_records, list), "compact source records absent")
    by_execution: dict[int, list[Any]] = {}
    for item in raw_records:
        execution = int(item["execution_ordinal"])
        if start <= execution < stop:
            by_execution.setdefault(execution, []).append(
                deserialize_factorized_directional_source_index(item)
            )
    del raw, raw_records
    gc.collect()
    source_descriptor = read_json(
        safe_path(RC_ROOT / str(source_manifest["source_manifest_path"]))
    )
    geometry_payload_path = safe_path(RC_ROOT / str(source_descriptor["geometry_payload"]))
    require(file_sha256(geometry_payload_path) == source_descriptor["geometry_payload_sha256"], "geometry payload hash drift")
    require(
        source_descriptor["geometry_payload_sha256"]
        == authority["bindings"]["full600_payload"]["sha256"],
        "V122 full600 payload authority binding drift",
    )
    shared_loader = CW1MultiTilePSealSourceLoaderV1(
        payload_path=geometry_payload_path,
        rc_root=RC_ROOT,
        expected_payload_file_sha256=source_descriptor["geometry_payload_sha256"],
    )
    shared_payload = shared_loader._load_payload()
    shared_queries = {
        int(item["execution_ordinal"]): item
        for item in shared_payload["query_records"]
    }
    shared_references = {
        int(item["physical_row"]): item
        for item in shared_payload["reference_records"]
    }
    rows = []
    exclusions = []
    model_forward_count = 0
    for execution in range(start, stop):
        if execution in EXCLUDED:
            require(execution not in by_execution, "excluded query has compact source records")
            exclusions.append({"execution_ordinal": execution, "reason": "NATURAL_C128_TARGET_MISS"})
            continue
        sources = by_execution.get(execution)
        qrow = query_manifest.get(execution)
        require(isinstance(sources, list) and len(sources) == 256 and isinstance(qrow, Mapping), "query C128x2 compact source drift")
        source_fold = int(qrow["source_fold"])
        source_by_coordinate = {
            (item.candidate_position, item.direction): item for item in sources
        }
        require(
            set(source_by_coordinate)
            == {(position, direction) for position in range(128) for direction in P_DIRECTIONS},
            "P utility source coordinate closure drift",
        )
        require(
            len(
                {
                    (
                        item.query_id,
                        item.historical_query_ordinal,
                        item.execution_ordinal,
                        item.query_source_image_sha256,
                        item.candidate_axis_sha256,
                    )
                    for item in sources
                }
            )
            == 1,
            "P utility source query/candidate-axis anchor drift",
        )
        candidate_bindings = []
        for position in range(128):
            pair = [
                source_by_coordinate[(position, direction)]
                for direction in P_DIRECTIONS
            ]
            binding = (
                pair[0].candidate_key,
                pair[0].candidate_physical_row,
                pair[0].candidate_reference_source_sha256,
            )
            require(
                binding
                == (
                    pair[1].candidate_key,
                    pair[1].candidate_physical_row,
                    pair[1].candidate_reference_source_sha256,
                ),
                "P utility candidate binding differs across directions",
            )
            candidate_bindings.append(binding)
        require(
            len({item[0] for item in candidate_bindings}) == 128
            and len({item[1] for item in candidate_bindings}) == 128,
            "P utility natural C128 candidate binding collision",
        )
        outer_scopes = [
            scope
            for scope in crossfit_fanout_scopes(source_fold)
            if scope.fit_role == "OUTER_TRAIN_REFIT"
        ]
        require(len(outer_scopes) == 1, "outer-refit P scope drift")
        scope = outer_scopes[0]
        assignments = {item["fit_id"]: item for item in qrow["head_assignments"]}
        assignment = assignments[scope.fit_id]
        checkpoint_path = safe_path(RC_ROOT / str(assignment["checkpoint_path"]))
        require(file_sha256(checkpoint_path) == assignment["checkpoint_sha256"], "P checkpoint hash drift")
        require(
            assignment["checkpoint_sha256"]
            == authority["bindings"][f"p_outer_refit_checkpoint_fold{source_fold}"]["sha256"],
            "P outer-refit checkpoint authority drift",
        )
        require(
            assignment["training_manifest_sha256"]
            == authority["bindings"][f"p_outer_refit_manifest_fold{source_fold}"]["sha256"],
            "P outer-refit training-manifest authority drift",
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        require(
            isinstance(checkpoint, Mapping)
            and checkpoint.get("schema_version")
            == "rc_dino_rcde_sr0_mt_p_v2_checkpoint_v1_20260818"
            and checkpoint.get("status") == "RCDE_SR0_MT_P_V2_COMPACT_CHECKPOINT_READY"
            and checkpoint.get("fit_id") == scope.fit_id
            and checkpoint.get("completed_updates") == 2048
            and checkpoint.get("logical_sha256")
            == assignment["checkpoint_logical_sha256"]
            and checkpoint.get("state_sha256")
            == assignment["checkpoint_state_sha256"],
            "P outer-refit checkpoint schema/state drift",
        )
        model = SharedMultitilePHead()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        resolver = QueryResolver(
            shared_loader, shared_queries[execution], shared_references
        )
        feature_producer = BatchedNaturalPairFeatureProducer(resolver)

        def stream():
            for position in range(128):
                row = source_by_coordinate[(position, P_DIRECTIONS[0])].candidate_physical_row
                for direction in P_DIRECTIONS:
                    source = source_by_coordinate[(position, direction)]
                    yield CompactDirectionalFeatureInput(
                        source_fold=source_fold,
                        source_index=source,
                        candidate_features=feature_producer(source),
                    )
                feature_producer._prepared.clear()
                resolver.release_candidate(row)

        decisions = []
        with torch.no_grad():
            for compact_input in stream():
                output = score_compact_deployable_direction(model, compact_input)
                decisions.append(decision_from_compact_output(output))
        require(
            [(item.candidate_position, item.direction) for item in decisions]
            == [
                (position, direction)
                for position in range(128)
                for direction in P_DIRECTIONS
            ],
            "P utility single-outer decision population drift",
        )
        directional: dict[int, dict[str, Any]] = {}
        for decision in decisions:
            source = source_by_coordinate[(decision.candidate_position, decision.direction)]
            directional.setdefault(decision.candidate_position, {})[
                decision.direction
            ] = (decision.candidate_utility.detach().cpu(), source)
            model_forward_count += 1
        candidate_rows = []
        for position in range(128):
            values = directional[position]
            require(set(values) == set(P_DIRECTIONS), "P utility direction pair drift")
            utility = fuse_direction_utilities(
                values[P_DIRECTIONS[0]][0], values[P_DIRECTIONS[1]][0]
            )
            first = values[P_DIRECTIONS[0]][1]
            candidate_rows.append(
                {
                    "candidate_position": position,
                    "candidate_key": first.candidate_key,
                    "candidate_physical_row": first.candidate_physical_row,
                    "candidate_reference_source_sha256": first.candidate_reference_source_sha256,
                    "utility": float(utility),
                    "direction_utility": {
                        direction: float(values[direction][0]) for direction in P_DIRECTIONS
                    },
                }
            )
        row_value = {
            "query_id": sources[0].query_id,
            "execution_ordinal": execution,
            "query_source_image_sha256": sources[0].query_source_image_sha256,
            "outer_fold": source_fold,
            "fit_id": scope.fit_id,
            "p_checkpoint_sha256": assignment["checkpoint_sha256"],
            "target_free": True,
            "candidate_count": 128,
            "candidates": candidate_rows,
            "candidate_axis_sha256": canonical_sha256(
                [item["candidate_physical_row"] for item in candidate_rows]
            ),
        }
        row_value["record_sha256"] = logical_sha256(row_value)
        rows.append(row_value)
        del resolver, feature_producer, decisions, model, checkpoint
        gc.collect()
    value = {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "claim_level": "TARGET_FREE_OUTER_REFIT_P_V2_C128_UTILITY_COMPARATOR_SHARD_ONLY",
        "shard_ordinal": shard_ordinal,
        "execution_start": start,
        "execution_stop": stop,
        "query_count": len(rows),
        "excluded_query_count": len(exclusions),
        "rows": rows,
        "row_sequence_sha256": canonical_sha256([item["record_sha256"] for item in rows]),
        "exclusions": exclusions,
        "source_bindings": runtime_source_bindings(authority, authority_sha),
        "model_forward_count": model_forward_count,
        "target_rival_read_count": 0,
        "dino_token_read_count": 0,
        "rank_winner_gap_outcome_read_count": 0,
        "scientific_reduction_count": 0,
        "protected_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    require(model_forward_count == 256 * len(rows), "V122 forward-count closure drift")
    value["logical_sha256"] = logical_sha256(value)
    output = safe_path(output_root, must_exist=False) / f"shard_{start:03d}_{stop:03d}.pt"
    if not replay_only:
        atomic_torch(output, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--shard-ordinal", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    value = materialize(args.authority, args.shard_ordinal, args.output_root)
    print(
        json.dumps(
            {
                "status": value["status"],
                "shard_ordinal": value["shard_ordinal"],
                "query_count": value["query_count"],
                "logical_sha256": value["logical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
