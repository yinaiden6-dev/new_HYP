"""Resumable, semantics-preserving production primitives for V124 Phase A.

This append-only module deliberately reuses the frozen C_COL donor equations,
P-V2 feature/head/resolver, Natural/Core serializer, and missing-root checks.
It changes only execution granularity: the complete C128 donor plan is computed
first, while expensive source construction and P forwards are sealed in fixed
16-candidate shards.  A resumed shard is accepted only after every materialized
record and every semantic hash has been recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import resource
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from . import dino_rcde_sr0_mt_c_col_p_v1 as c_col_v1
from . import dino_rcde_sr0_mt_p_natural_source_v1 as natural_v1
from . import dino_rcde_track_r_c_col_p_v2_compat_v1 as strict_v1
from .dino_rcde_sr0_mt_p_compact_lock_fanout_v1 import (
    FanoutDirectionDecision,
    decision_from_legacy_output,
)
from .dino_rcde_sr0_mt_p_natural_adapter_v2 import (
    build_natural_p_lock_v2,
    validate_natural_p_lock_v2,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    DirectionalFeatureRecord,
    assert_no_forbidden_prejoin_keys,
    canonical_sha256,
    score_deployable_direction,
    tensor_sha256,
)
from .dino_rcde_track_r_c_col_p_phase_a_resume_schema_v2 import (
    CANDIDATE_COUNT,
    CONTROL_CHUNK_SCHEMA,
    CONTROL_CHUNK_STATUS,
    CONTROL_SHARD_SCHEMA,
    CONTROL_SHARD_STATUS,
    DIRECTION_COUNT,
    NAMESPACE,
    PROGRESS_INTERVAL,
    SOURCE_SHARD_SCHEMA,
    SOURCE_SHARD_STATUS,
    SOURCE_CHUNK_SCHEMA,
    SOURCE_CHUNK_STATUS,
    control_shard_ranges,
    progress_chunk_ranges,
    source_shard_ranges,
)


class ResumablePhaseAError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise ResumablePhaseAError(message)


def rss_bytes() -> int:
    """Return process maximum RSS in bytes on the Linux execution platform."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    require(value >= 0, "Phase-A V2 RSS counter drift")
    return value * 1024


def emit_progress(
    *,
    stage: str,
    shard_start: int,
    shard_stop: int,
    local_completed: int,
    global_completed: int,
    elapsed_seconds: float,
    output_path: str,
    committed_count: int,
    commit_sha256: str,
    run_id: str,
) -> dict[str, Any]:
    require(
        isinstance(stage, str)
        and bool(stage)
        and isinstance(run_id, str)
        and bool(run_id)
        and type(shard_start) is int
        and type(shard_stop) is int
        and type(local_completed) is int
        and type(global_completed) is int
        and 0 <= shard_start < shard_stop <= CANDIDATE_COUNT
        and local_completed in {PROGRESS_INTERVAL, shard_stop - shard_start}
        and shard_start + local_completed == global_completed
        and isinstance(elapsed_seconds, (int, float))
        and float(elapsed_seconds) >= 0.0
        and isinstance(output_path, str)
        and bool(output_path)
        and type(committed_count) is int
        and committed_count in {PROGRESS_INTERVAL, shard_stop - shard_start}
        and committed_count <= local_completed
        and isinstance(commit_sha256, str)
        and len(commit_sha256) == 64,
        "Phase-A V2 progress drift",
    )
    value = {
        "namespace": NAMESPACE,
        "stage": stage,
        "run_id": run_id,
        "shard_start": shard_start,
        "shard_stop": shard_stop,
        "local_completed": local_completed,
        "shard_local_completed": local_completed,
        "chunk_candidate_count": committed_count,
        "global_completed": global_completed,
        "total_candidates": CANDIDATE_COUNT,
        "elapsed_seconds": float(elapsed_seconds),
        "output_path": output_path,
        "committed_count": committed_count,
        "commit_sha256": commit_sha256,
        "rss_bytes": rss_bytes(),
    }
    print(json.dumps(value, sort_keys=True), flush=True)
    return value


def tensor_tree(value: Any) -> Any:
    """Canonical JSON-safe semantic projection for tensors and nested records."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": tensor_sha256(tensor),
        }
    if isinstance(value, Mapping):
        return {str(key): tensor_tree(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ResumablePhaseAError(
        f"Phase-A V2 semantic tree rejects type: {type(value).__name__}"
    )


def semantic_sha256(value: Any) -> str:
    return canonical_sha256(tensor_tree(value))


def decision_sha256(value: FanoutDirectionDecision) -> str:
    require(isinstance(value, FanoutDirectionDecision), "Phase-A V2 decision type drift")
    return canonical_sha256(
        {
            "candidate_position": value.candidate_position,
            "candidate_key": value.candidate_key,
            "candidate_physical_row": value.candidate_physical_row,
            "candidate_reference_source_sha256": (
                value.candidate_reference_source_sha256
            ),
            "direction": value.direction,
            "selected_action_indices": list(value.selected_action_indices),
            "selected_action_keys": list(value.selected_action_keys),
            "root_statuses": list(value.root_statuses),
            "root_scores_sha256": tensor_sha256(value.root_scores),
            "row_hashes": list(value.row_hashes),
            "row_radii": list(value.row_radii),
            "row_scores_sha256": tensor_sha256(value.row_scores),
            "row_ready_sha256": tensor_sha256(value.row_ready),
            "candidate_utility_sha256": tensor_sha256(value.candidate_utility),
            "selected_bank_ordinal": value.selected_bank_ordinal,
            "selected_row_sha256": value.selected_row_sha256,
            "lock_status": value.lock_status,
        }
    )


def build_candidate_raw_row(
    *, query_source: Any, candidate_position: int, physical_row: int, reference_source: Any
) -> dict[str, Any]:
    """Build one row byte-semantically equivalent to V1 full-query construction."""

    require(
        type(candidate_position) is int
        and 0 <= candidate_position < CANDIDATE_COUNT
        and type(physical_row) is int
        and physical_row >= 0,
        "Phase-A V2 candidate source address drift",
    )
    key = natural_v1.candidate_key_v1(
        physical_row=physical_row,
        source_image_sha256=reference_source.source_image_sha256,
    )
    value = {
        "candidate_position": candidate_position,
        "candidate_key": key,
        "candidate_physical_row": physical_row,
        "candidate_reference_source_sha256": reference_source.source_image_sha256,
        "reference_tokens": reference_source.tokens,
        "reference_valid_patch_mask": (
            reference_source.colnomic_geometry.valid_patch_mask
        ),
        "colnomic_reference_grid_shape": list(reference_source.grid_shape),
        "dino_reference_grid_shape": list(
            reference_source.dino_geometry.grid_shape
        ),
        "reference_geometry_sha256": reference_source.dino_geometry.sha256,
        "directions": natural_v1._candidate_directions(
            candidate_key=key,
            query=query_source,
            reference=reference_source,
        ),
    }
    assert_no_forbidden_prejoin_keys(value)
    return value


def build_source_shard_payload(
    *,
    candidates: Sequence[Mapping[str, Any]],
    start: int,
    stop: int,
    authority_sha256: str,
    preflight_logical_sha256: str,
    query_stage_logical_sha256: str,
    run_id: str,
    artifact_kind: str = "SHARD",
) -> dict[str, Any]:
    allowed = (
        source_shard_ranges() if artifact_kind == "SHARD" else progress_chunk_ranges()
    )
    require(
        artifact_kind in {"SHARD", "CHUNK"} and (start, stop) in allowed,
        "source shard/chunk range drift",
    )
    rows = [dict(item) for item in candidates]
    require(
        len(rows) == stop - start
        and [int(item.get("candidate_position", -1)) for item in rows]
        == list(range(start, stop)),
        "source shard candidate coverage drift",
    )
    hashes = [semantic_sha256(item) for item in rows]
    value: dict[str, Any] = {
        "schema_version": (
            SOURCE_SHARD_SCHEMA if artifact_kind == "SHARD" else SOURCE_CHUNK_SCHEMA
        ),
        "status": (
            SOURCE_SHARD_STATUS if artifact_kind == "SHARD" else SOURCE_CHUNK_STATUS
        ),
        "artifact_kind": artifact_kind,
        "authority_sha256": authority_sha256,
        "preflight_logical_sha256": preflight_logical_sha256,
        "query_stage_logical_sha256": query_stage_logical_sha256,
        "candidate_start": start,
        "candidate_stop": stop,
        "candidate_count": len(rows),
        "candidate_positions": list(range(start, stop)),
        "candidate_semantic_sha256_sequence": hashes,
        "candidate_semantic_sha256_sequence_sha256": canonical_sha256(hashes),
        "candidates": rows,
        "run_id": run_id,
        "rss_bytes_at_commit": rss_bytes(),
    }
    value["logical_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key not in {"candidates", "logical_sha256"}}
    )
    return value


def validate_source_shard_payload(
    value: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    authority_sha256: str,
    preflight_logical_sha256: str,
    query_stage_logical_sha256: str,
    artifact_kind: str = "SHARD",
) -> tuple[Mapping[str, Any], ...]:
    allowed = (
        source_shard_ranges() if artifact_kind == "SHARD" else progress_chunk_ranges()
    )
    require(
        artifact_kind in {"SHARD", "CHUNK"} and (start, stop) in allowed,
        "source resume shard/chunk range drift",
    )
    candidates = value.get("candidates")
    require(
        value.get("schema_version")
        == (SOURCE_SHARD_SCHEMA if artifact_kind == "SHARD" else SOURCE_CHUNK_SCHEMA)
        and value.get("status")
        == (SOURCE_SHARD_STATUS if artifact_kind == "SHARD" else SOURCE_CHUNK_STATUS)
        and value.get("artifact_kind") == artifact_kind
        and value.get("authority_sha256") == authority_sha256
        and value.get("preflight_logical_sha256") == preflight_logical_sha256
        and value.get("query_stage_logical_sha256") == query_stage_logical_sha256
        and value.get("candidate_start") == start
        and value.get("candidate_stop") == stop
        and value.get("candidate_positions") == list(range(start, stop))
        and isinstance(candidates, list)
        and len(candidates) == stop - start,
        "source resume envelope drift",
    )
    rows = tuple(dict(item) for item in candidates if isinstance(item, Mapping))
    require(len(rows) == stop - start, "source resume row type drift")
    hashes = [semantic_sha256(item) for item in rows]
    require(
        [int(item.get("candidate_position", -1)) for item in rows]
        == list(range(start, stop))
        and value.get("candidate_semantic_sha256_sequence") == hashes
        and value.get("candidate_semantic_sha256_sequence_sha256")
        == canonical_sha256(hashes)
        and value.get("logical_sha256")
        == canonical_sha256(
            {
                key: item
                for key, item in value.items()
                if key not in {"candidates", "logical_sha256"}
            }
        ),
        "source resume per-candidate hash drift",
    )
    for row in rows:
        assert_no_forbidden_prejoin_keys(row)
    return rows


def merge_source_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    start: int,
    stop: int,
    authority_sha256: str,
    preflight_logical_sha256: str,
    query_stage_logical_sha256: str,
    run_id: str,
) -> dict[str, Any]:
    require((start, stop) in source_shard_ranges(), "source merge range drift")
    ranges = ((start, start + PROGRESS_INTERVAL), (start + PROGRESS_INTERVAL, stop))
    require(len(chunks) == len(ranges), "source merge chunk count drift")
    rows: list[Mapping[str, Any]] = []
    for raw, (chunk_start, chunk_stop) in zip(chunks, ranges, strict=True):
        rows.extend(
            validate_source_shard_payload(
                raw,
                start=chunk_start,
                stop=chunk_stop,
                authority_sha256=authority_sha256,
                preflight_logical_sha256=preflight_logical_sha256,
                query_stage_logical_sha256=query_stage_logical_sha256,
                artifact_kind="CHUNK",
            )
        )
    value = build_source_shard_payload(
        candidates=rows,
        start=start,
        stop=stop,
        authority_sha256=authority_sha256,
        preflight_logical_sha256=preflight_logical_sha256,
        query_stage_logical_sha256=query_stage_logical_sha256,
        run_id=run_id,
        artifact_kind="SHARD",
    )
    value["source_chunk_logical_sha256_sequence"] = [
        str(item["logical_sha256"]) for item in chunks
    ]
    value["logical_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"candidates", "logical_sha256"}
        }
    )
    validate_source_shard_payload(
        value,
        start=start,
        stop=stop,
        authority_sha256=authority_sha256,
        preflight_logical_sha256=preflight_logical_sha256,
        query_stage_logical_sha256=query_stage_logical_sha256,
        artifact_kind="SHARD",
    )
    return value


@dataclass(frozen=True)
class ControlContextV2:
    raw_query: Mapping[str, Any]
    source_fold: int
    model: torch.nn.Module | None
    fit_id: str
    candidates: tuple[Mapping[str, Any], ...]
    binding_rows: tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]
    donor_positions: tuple[int, ...]
    donor_receipt: Mapping[str, Any]
    query_geometry_record: Mapping[str, Any]
    reference_records: Mapping[int, Mapping[str, Any]]
    destination_valid: Mapping[str, torch.Tensor]
    fold_record: Mapping[str, Any]
    membership: Mapping[str, Any]
    p_checkpoint_sha256: str
    p_training_manifest_sha256: str


def prepare_control_context(
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
    load_model: bool = True,
) -> ControlContextV2:
    """Prepare the exact complete-axis state used by V1 before its record loop."""

    checkpoint_sha = strict_v1._sha(p_checkpoint_sha256, "P checkpoint")
    manifest_sha = strict_v1._sha(p_training_manifest_sha256, "P training manifest")
    raw_query = strict_v1._select_raw_query(raw_source, query_id)
    source_fold = int(raw_query.get("source_fold", -1))
    model = (
        strict_v1.load_strict_outer_refit_p_v2_model(
            checkpoint,
            source_fold=source_fold,
            p_training_manifest_sha256=manifest_sha,
        )
        if load_model
        else None
    )
    fit_id = f"P_OUTER{source_fold}_OUTER_REFIT"
    require(
        fold_record.get("query_id") == query_id
        and fold_record.get("query_ordinal")
        == raw_query.get("historical_query_ordinal")
        and fold_record.get("source_image_sha256")
        == raw_query.get("query_source_image_sha256")
        and fold_record.get("inner_fold") == source_fold,
        "resumable strict C_COL fold/query binding drift",
    )
    candidates = tuple(
        strict_v1._mapping(item, "resumable raw candidate")
        for item in strict_v1._sequence(
            raw_query.get("candidates"), "resumable candidate axis"
        )
    )
    require(
        len(candidates) == CANDIDATE_COUNT
        and tuple(int(item.get("candidate_position", -1)) for item in candidates)
        == tuple(range(CANDIDATE_COUNT)),
        "resumable strict C_COL candidate axis drift",
    )
    donor_positions = strict_v1.canonical_donor_positions(
        raw_query,
        corrected_identity_by_physical_row,
        expected_count=CANDIDATE_COUNT,
    )
    donor = strict_v1.donor_receipt(
        raw_query,
        corrected_identity_by_physical_row,
        expected_count=CANDIDATE_COUNT,
    )
    binding_rows = tuple(
        c_col_v1._binding_rows(
            raw_query,
            corrected_identity_by_physical_row,
            CANDIDATE_COUNT,
        )
    )
    require(
        len(binding_rows) == CANDIDATE_COUNT
        and all(
            int(source["candidate_position"]) == donor_positions[position]
            for position, (_, source) in enumerate(binding_rows)
        ),
        "resumable binding rows differ from complete C128 donor plan",
    )
    query_grid = strict_v1._grid(
        raw_query.get("dino_query_grid_shape"), "resumable query DINO"
    )
    require(
        query_geometry_record.get("query_id") == query_id
        and query_geometry_record.get("execution_ordinal")
        == raw_query.get("execution_ordinal")
        and query_geometry_record.get("source_image_sha256")
        == raw_query.get("query_source_image_sha256"),
        "resumable strict C_COL query geometry drift",
    )
    strict_v1._geometry_valid_mask(
        query_geometry_record, expected_grid=query_grid, name="resumable query"
    )
    destination_valid: dict[str, torch.Tensor] = {}
    reference_records: dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        row = int(candidate.get("candidate_physical_row", -1))
        geometry = reference_geometry_by_physical_row.get(row)
        require(isinstance(geometry, Mapping), "resumable destination geometry absent")
        reference_grid = strict_v1._grid(
            candidate.get("dino_reference_grid_shape"),
            "resumable destination DINO",
        )
        require(
            geometry.get("physical_row") == row
            and geometry.get("source_image_sha256")
            == candidate.get("candidate_reference_source_sha256")
            and strict_v1._mapping(
                geometry.get("dino_geometry"), "resumable destination DINO"
            ).get("geometry_sha256")
            == candidate.get("reference_geometry_sha256"),
            "resumable destination geometry/source binding drift",
        )
        key = str(candidate.get("candidate_key"))
        destination_valid[key] = strict_v1._geometry_valid_mask(
            geometry, expected_grid=reference_grid, name="resumable destination"
        )
        reference_records[row] = geometry
    return ControlContextV2(
        raw_query=MappingProxyType(dict(raw_query)),
        source_fold=source_fold,
        model=model,
        fit_id=fit_id,
        candidates=candidates,
        binding_rows=binding_rows,
        donor_positions=tuple(donor_positions),
        donor_receipt=MappingProxyType(dict(donor)),
        query_geometry_record=MappingProxyType(dict(query_geometry_record)),
        reference_records=MappingProxyType(reference_records),
        destination_valid=MappingProxyType(destination_valid),
        fold_record=MappingProxyType(dict(fold_record)),
        membership=MappingProxyType(dict(membership)),
        p_checkpoint_sha256=checkpoint_sha,
        p_training_manifest_sha256=manifest_sha,
    )


def _control_shard_logical(value: Mapping[str, Any]) -> str:
    excluded = {
        "records",
        "decisions",
        "source_records",
        "source_populations",
        "logical_sha256",
    }
    return canonical_sha256({key: item for key, item in value.items() if key not in excluded})


def build_control_shard_payload(
    context: ControlContextV2,
    *,
    start: int,
    stop: int,
    authority_sha256: str,
    preflight_logical_sha256: str,
    context_stage_logical_sha256: str,
    run_id: str,
    artifact_kind: str = "SHARD",
) -> dict[str, Any]:
    allowed = (
        control_shard_ranges() if artifact_kind == "SHARD" else progress_chunk_ranges()
    )
    require(
        artifact_kind in {"SHARD", "CHUNK"} and (start, stop) in allowed,
        "control shard/chunk range drift",
    )
    require(
        isinstance(context.model, torch.nn.Module),
        "control shard build requires one frozen P-V2 model load",
    )
    bindings = context.binding_rows
    # The reconstructed binding rows must agree exactly with the complete donor
    # plan sealed before sharding.  This catches a future accidental local map.
    require(
        len(bindings) == CANDIDATE_COUNT
        and all(
            int(donor["candidate_position"]) == context.donor_positions[position]
            for position, (_, donor) in enumerate(bindings)
        ),
        "control shard donor map differs from complete C128 plan",
    )
    records: list[Mapping[str, Any]] = []
    decisions: list[FanoutDirectionDecision] = []
    source_records: list[Mapping[str, Any]] = []
    source_populations: list[Mapping[str, Any]] = []
    coordinates: list[list[Any]] = []
    missing_count = 0
    unmappable_count = 0
    with torch.no_grad():
        for position in range(start, stop):
            destination, donor = bindings[position]
            destination_key = str(destination["candidate_key"])
            destination_valid = context.destination_valid[destination_key]
            for direction in P_DIRECTIONS:
                feature = c_col_v1._directional_record(
                    context.raw_query,
                    destination,
                    donor,
                    destination_valid,
                    direction=direction,
                )
                require(
                    isinstance(feature, DirectionalFeatureRecord),
                    "control shard feature type drift",
                )
                source_record, source_population = strict_v1._strict_source_record(
                    raw_query=context.raw_query,
                    destination=destination,
                    donor=donor,
                    feature_record=feature,
                )
                decision = decision_from_legacy_output(
                    score_deployable_direction(context.model, feature)
                )
                reference_geometry = context.reference_records[
                    feature.candidate_physical_row
                ]
                record = build_natural_p_lock_v2(
                    source_record=source_record,
                    query_geometry_record=context.query_geometry_record,
                    reference_geometry_record=reference_geometry,
                    fold_record=context.fold_record,
                    membership=context.membership,
                    row_scores=decision.row_scores,
                    selected_action_indices=decision.selected_action_indices,
                    fit_id=context.fit_id,
                    crossfit_role="OUTER_HELDOUT_DEPLOYMENT",
                    p_checkpoint_sha256=context.p_checkpoint_sha256,
                    p_training_manifest_sha256=context.p_training_manifest_sha256,
                )
                validate_natural_p_lock_v2(record)
                missing, unmappable = strict_v1._assert_v2_root_semantics(
                    record, decision, source_record
                )
                missing_count += missing
                unmappable_count += unmappable
                coordinates.append([position, direction])
                records.append(dict(record))
                decisions.append(decision)
                source_records.append(dict(source_record))
                source_populations.append(dict(source_population))
            # Durable progress is emitted by the file-owning runner only after
            # an 8-candidate chunk has been atomically sealed and hashed.
    record_hashes = [str(item["record_sha256"]) for item in records]
    decision_hashes = [decision_sha256(item) for item in decisions]
    source_hashes = [str(item["cache_sha256"]) for item in source_records]
    population_hashes = [canonical_sha256(item) for item in source_populations]
    value: dict[str, Any] = {
        "schema_version": (
            CONTROL_SHARD_SCHEMA if artifact_kind == "SHARD" else CONTROL_CHUNK_SCHEMA
        ),
        "status": (
            CONTROL_SHARD_STATUS if artifact_kind == "SHARD" else CONTROL_CHUNK_STATUS
        ),
        "artifact_kind": artifact_kind,
        "authority_sha256": authority_sha256,
        "preflight_logical_sha256": preflight_logical_sha256,
        "context_stage_logical_sha256": context_stage_logical_sha256,
        "candidate_start": start,
        "candidate_stop": stop,
        "candidate_count": stop - start,
        "direction_count": DIRECTION_COUNT,
        "record_count": len(records),
        "coordinates": coordinates,
        "record_sha256_sequence": record_hashes,
        "record_sha256_sequence_sha256": canonical_sha256(record_hashes),
        "decision_sha256_sequence": decision_hashes,
        "decision_sha256_sequence_sha256": canonical_sha256(decision_hashes),
        "source_record_sha256_sequence": source_hashes,
        "source_record_sha256_sequence_sha256": canonical_sha256(source_hashes),
        "source_population_sha256_sequence": population_hashes,
        "source_population_sha256_sequence_sha256": canonical_sha256(
            population_hashes
        ),
        "root_reference_missing_count": missing_count,
        "root_query_unmappable_count": unmappable_count,
        "donor_positions": list(context.donor_positions[start:stop]),
        "donor_receipt_logical_sha256": canonical_sha256(
            dict(context.donor_receipt)
        ),
        "records": records,
        "decisions": decisions,
        "source_records": source_records,
        "source_populations": source_populations,
        "run_id": run_id,
        "rss_bytes_at_commit": rss_bytes(),
    }
    value["logical_sha256"] = _control_shard_logical(value)
    return value


def validate_control_shard_payload(
    value: Mapping[str, Any],
    context: ControlContextV2,
    *,
    start: int,
    stop: int,
    authority_sha256: str,
    preflight_logical_sha256: str,
    context_stage_logical_sha256: str,
    artifact_kind: str = "SHARD",
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[FanoutDirectionDecision, ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    allowed = (
        control_shard_ranges() if artifact_kind == "SHARD" else progress_chunk_ranges()
    )
    require(
        artifact_kind in {"SHARD", "CHUNK"} and (start, stop) in allowed,
        "control resume shard/chunk range drift",
    )
    expected_coordinates = [
        [position, direction]
        for position in range(start, stop)
        for direction in P_DIRECTIONS
    ]
    require(
        value.get("schema_version")
        == (CONTROL_SHARD_SCHEMA if artifact_kind == "SHARD" else CONTROL_CHUNK_SCHEMA)
        and value.get("status")
        == (CONTROL_SHARD_STATUS if artifact_kind == "SHARD" else CONTROL_CHUNK_STATUS)
        and value.get("artifact_kind") == artifact_kind
        and value.get("authority_sha256") == authority_sha256
        and value.get("preflight_logical_sha256") == preflight_logical_sha256
        and value.get("context_stage_logical_sha256")
        == context_stage_logical_sha256
        and value.get("candidate_start") == start
        and value.get("candidate_stop") == stop
        and value.get("candidate_count") == stop - start
        and value.get("direction_count") == DIRECTION_COUNT
        and value.get("record_count") == len(expected_coordinates)
        and value.get("coordinates") == expected_coordinates
        and value.get("donor_positions")
        == list(context.donor_positions[start:stop])
        and value.get("donor_receipt_logical_sha256")
        == canonical_sha256(dict(context.donor_receipt)),
        "control resume envelope/donor drift",
    )
    records_raw = value.get("records")
    decisions_raw = value.get("decisions")
    source_records_raw = value.get("source_records")
    populations_raw = value.get("source_populations")
    require(
        isinstance(records_raw, list)
        and isinstance(decisions_raw, list)
        and isinstance(source_records_raw, list)
        and isinstance(populations_raw, list)
        and len(records_raw)
        == len(decisions_raw)
        == len(source_records_raw)
        == len(populations_raw)
        == len(expected_coordinates),
        "control resume materialized population drift",
    )
    records = tuple(dict(item) for item in records_raw if isinstance(item, Mapping))
    decisions = tuple(
        item for item in decisions_raw if isinstance(item, FanoutDirectionDecision)
    )
    source_records = tuple(
        dict(item) for item in source_records_raw if isinstance(item, Mapping)
    )
    populations = tuple(
        dict(item) for item in populations_raw if isinstance(item, Mapping)
    )
    require(
        len(records)
        == len(decisions)
        == len(source_records)
        == len(populations)
        == len(expected_coordinates),
        "control resume typed population drift",
    )
    missing_count = 0
    unmappable_count = 0
    for coordinate, record, decision, source_record, population in zip(
        expected_coordinates,
        records,
        decisions,
        source_records,
        populations,
        strict=True,
    ):
        validate_natural_p_lock_v2(record)
        require(
            [decision.candidate_position, decision.direction] == coordinate
            and int(record["candidate"]["candidate_position"]) == coordinate[0]
            and str(record["direction"]) == coordinate[1]
            and source_record.get("candidate_position") == coordinate[0]
            and source_record.get("direction") == coordinate[1]
            and source_record.get("cache_sha256") == canonical_sha256(population),
            "control resume coordinate/source drift",
        )
        missing, unmappable = strict_v1._assert_v2_root_semantics(
            record, decision, source_record
        )
        missing_count += missing
        unmappable_count += unmappable
    record_hashes = [str(item["record_sha256"]) for item in records]
    decision_hashes = [decision_sha256(item) for item in decisions]
    source_hashes = [str(item["cache_sha256"]) for item in source_records]
    population_hashes = [canonical_sha256(item) for item in populations]
    require(
        value.get("record_sha256_sequence") == record_hashes
        and value.get("record_sha256_sequence_sha256")
        == canonical_sha256(record_hashes)
        and value.get("decision_sha256_sequence") == decision_hashes
        and value.get("decision_sha256_sequence_sha256")
        == canonical_sha256(decision_hashes)
        and value.get("source_record_sha256_sequence") == source_hashes
        and value.get("source_record_sha256_sequence_sha256")
        == canonical_sha256(source_hashes)
        and value.get("source_population_sha256_sequence") == population_hashes
        and value.get("source_population_sha256_sequence_sha256")
        == canonical_sha256(population_hashes)
        and value.get("root_reference_missing_count") == missing_count
        and value.get("root_query_unmappable_count") == unmappable_count
        and value.get("logical_sha256") == _control_shard_logical(value),
        "control resume per-record hash/root-semantics drift",
    )
    return records, decisions, source_records, populations


def merge_control_chunks(
    chunks: Sequence[Mapping[str, Any]],
    context: ControlContextV2,
    *,
    start: int,
    stop: int,
    authority_sha256: str,
    preflight_logical_sha256: str,
    context_stage_logical_sha256: str,
    run_id: str,
) -> dict[str, Any]:
    """Merge two already-durable 8-candidate chunks into one 16-candidate shard."""

    require((start, stop) in control_shard_ranges(), "control merge range drift")
    ranges = ((start, start + PROGRESS_INTERVAL), (start + PROGRESS_INTERVAL, stop))
    require(len(chunks) == len(ranges), "control merge chunk count drift")
    records: list[Mapping[str, Any]] = []
    decisions: list[FanoutDirectionDecision] = []
    source_records: list[Mapping[str, Any]] = []
    populations: list[Mapping[str, Any]] = []
    for raw, (chunk_start, chunk_stop) in zip(chunks, ranges, strict=True):
        observed = validate_control_shard_payload(
            raw,
            context,
            start=chunk_start,
            stop=chunk_stop,
            authority_sha256=authority_sha256,
            preflight_logical_sha256=preflight_logical_sha256,
            context_stage_logical_sha256=context_stage_logical_sha256,
            artifact_kind="CHUNK",
        )
        records.extend(observed[0])
        decisions.extend(observed[1])
        source_records.extend(observed[2])
        populations.extend(observed[3])
    coordinates = [
        [position, direction]
        for position in range(start, stop)
        for direction in P_DIRECTIONS
    ]
    record_hashes = [str(item["record_sha256"]) for item in records]
    decision_hashes = [decision_sha256(item) for item in decisions]
    source_hashes = [str(item["cache_sha256"]) for item in source_records]
    population_hashes = [canonical_sha256(item) for item in populations]
    value: dict[str, Any] = {
        "schema_version": CONTROL_SHARD_SCHEMA,
        "status": CONTROL_SHARD_STATUS,
        "artifact_kind": "SHARD",
        "authority_sha256": authority_sha256,
        "preflight_logical_sha256": preflight_logical_sha256,
        "context_stage_logical_sha256": context_stage_logical_sha256,
        "candidate_start": start,
        "candidate_stop": stop,
        "candidate_count": stop - start,
        "direction_count": DIRECTION_COUNT,
        "record_count": len(records),
        "coordinates": coordinates,
        "record_sha256_sequence": record_hashes,
        "record_sha256_sequence_sha256": canonical_sha256(record_hashes),
        "decision_sha256_sequence": decision_hashes,
        "decision_sha256_sequence_sha256": canonical_sha256(decision_hashes),
        "source_record_sha256_sequence": source_hashes,
        "source_record_sha256_sequence_sha256": canonical_sha256(source_hashes),
        "source_population_sha256_sequence": population_hashes,
        "source_population_sha256_sequence_sha256": canonical_sha256(
            population_hashes
        ),
        "root_reference_missing_count": sum(
            int(item["root_reference_missing_count"]) for item in chunks
        ),
        "root_query_unmappable_count": sum(
            int(item["root_query_unmappable_count"]) for item in chunks
        ),
        "donor_positions": list(context.donor_positions[start:stop]),
        "donor_receipt_logical_sha256": canonical_sha256(
            dict(context.donor_receipt)
        ),
        "records": records,
        "decisions": decisions,
        "source_records": source_records,
        "source_populations": populations,
        "run_id": run_id,
        "source_chunk_logical_sha256_sequence": [
            str(item["logical_sha256"]) for item in chunks
        ],
        "rss_bytes_at_commit": rss_bytes(),
    }
    value["logical_sha256"] = _control_shard_logical(value)
    validate_control_shard_payload(
        value,
        context,
        start=start,
        stop=stop,
        authority_sha256=authority_sha256,
        preflight_logical_sha256=preflight_logical_sha256,
        context_stage_logical_sha256=context_stage_logical_sha256,
        artifact_kind="SHARD",
    )
    return value


def combine_control_shards(
    shards: Sequence[Mapping[str, Any]],
    context: ControlContextV2,
    *,
    authority_sha256: str,
    preflight_logical_sha256: str,
    context_stage_logical_sha256: str,
) -> strict_v1.StrictCColPV2Replay:
    require(len(shards) == len(control_shard_ranges()), "control shard count drift")
    records: list[Mapping[str, Any]] = []
    decisions: list[FanoutDirectionDecision] = []
    source_records: list[Mapping[str, Any]] = []
    populations: list[Mapping[str, Any]] = []
    for raw, (start, stop) in zip(shards, control_shard_ranges(), strict=True):
        observed = validate_control_shard_payload(
            raw,
            context,
            start=start,
            stop=stop,
            authority_sha256=authority_sha256,
            preflight_logical_sha256=preflight_logical_sha256,
            context_stage_logical_sha256=context_stage_logical_sha256,
        )
        records.extend(observed[0])
        decisions.extend(observed[1])
        source_records.extend(observed[2])
        populations.extend(observed[3])
    expected_coordinates = [
        (position, direction)
        for position in range(CANDIDATE_COUNT)
        for direction in P_DIRECTIONS
    ]
    require(
        len(records) == CANDIDATE_COUNT * DIRECTION_COUNT
        and [
            (int(item["candidate"]["candidate_position"]), str(item["direction"]))
            for item in records
        ]
        == expected_coordinates,
        "combined control shard coverage drift",
    )
    record_hashes = [str(record["record_sha256"]) for record in records]
    missing_count = sum(
        int(item["root_reference_missing_count"]) for item in shards
    )
    unmappable_count = sum(
        int(item["root_query_unmappable_count"]) for item in shards
    )
    receipt: dict[str, Any] = {
        "schema_version": strict_v1.SCHEMA_VERSION,
        "namespace": strict_v1.NAMESPACE,
        "query_id": str(context.raw_query["query_id"]),
        "execution_ordinal": context.raw_query["execution_ordinal"],
        "source_fold": context.source_fold,
        "candidate_count": CANDIDATE_COUNT,
        "direction_count": DIRECTION_COUNT,
        "rebuilt_lock_record_count": len(records),
        "model_forward_count": len(records),
        "fit_id": context.fit_id,
        "p_checkpoint_sha256": context.p_checkpoint_sha256,
        "p_training_manifest_sha256": context.p_training_manifest_sha256,
        "donor_positions": list(context.donor_positions),
        "donor_receipt_logical_sha256": canonical_sha256(
            dict(context.donor_receipt)
        ),
        "donor_receipt": dict(context.donor_receipt),
        "control_source_population_sha256": canonical_sha256(populations),
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
    return strict_v1.StrictCColPV2Replay(
        records=tuple(MappingProxyType(dict(item)) for item in records),
        decisions=tuple(decisions),
        source_records=tuple(
            MappingProxyType(dict(item)) for item in source_records
        ),
        receipt=receipt,
    )


__all__ = [
    "ControlContextV2",
    "ResumablePhaseAError",
    "build_candidate_raw_row",
    "build_control_shard_payload",
    "build_source_shard_payload",
    "combine_control_shards",
    "decision_sha256",
    "emit_progress",
    "merge_control_chunks",
    "merge_source_chunks",
    "prepare_control_context",
    "rss_bytes",
    "semantic_sha256",
    "tensor_tree",
    "validate_control_shard_payload",
    "validate_source_shard_payload",
]
