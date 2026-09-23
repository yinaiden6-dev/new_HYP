#!/usr/bin/env python3
"""Precompute one quarantined H0-U4 fold-local score chunk.

This program deliberately stops before any scientific aggregation.  It replays
the frozen outer-refit P lock for the exact target, applies the same sealed
query union to the exact target full reference and the Q0 full-reference
donor, and records the two directional unary logits for INIT, Track-R and
Track-H.  No loss, margin, winner, GO/NO-GO or automatic stage transition is
computed here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import torch

import dino_rcde_sr0_mt_v_input_common_v1 as v_input
import dino_rcde_track_r_input_common_v1 as track_input
from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    QueryTokenFieldV1,
)
from rc_aslo_xf.dino_rcde_h0_full_reference_unary_v1 import (
    decode_full_reference_unary,
    exact_zero_unary,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (
    GeometryNamespaceBindingV1,
    outer_refit_head_spec,
    project_natural_v2_candidate_to_v1_runtime,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (
    LOCK_H0,
    LOCK_READY,
    state_dict_sha256,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rc_dino_rcde_h0_u4_precompute_chunk_v1_20260823"
STATUS = "H0_U4_PRECOMPUTE_CHUNK_READY_QUARANTINED"
AUTH_STATUS = "H0_U4_OUTER_OOF_SPECULATIVE_PRECOMPUTE_QUARANTINE_AUTHORIZED"
LEDGER_STATUS = "H0_U4_OUTER_OOF_METADATA_LEDGER_READY"
CACHE_MODEL_CHECKPOINT_LOGICAL_SHA256 = (
    "f901d9bd056bb65e5fcb02c72e869f6d0cf8d7472467d22fbc340442feca034c"
)
ARM_ORDER = ("INIT", "TRACK_R", "TRACK_H")
CACHE_ROOT_RELATIVE = "runtime/dino_rcde_p0_v1_2/stable_fp16_cache"


class U4PrecomputeError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise U4PrecomputeError(message)


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


def safe_file(path: Path, name: str) -> Path:
    value = path.resolve()
    require(ROOT == value or ROOT in value.parents, f"{name} escapes RC root")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"{name} requests a protected path",
    )
    require(value.is_file() and not value.is_symlink(), f"{name} absent/unsafe")
    return value


def safe_directory(path: Path, name: str) -> Path:
    value = path.resolve()
    require(ROOT == value or ROOT in value.parents, f"{name} escapes RC root")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"{name} requests a protected path",
    )
    require(value.is_dir() and not value.is_symlink(), f"{name} absent/unsafe")
    return value


def read_json(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(safe_file(path, name).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{name} is not an object")
    return value


def bound_path(authority: Mapping[str, Any], key: str) -> Path:
    binding = authority.get("bindings", {}).get(key)
    require(isinstance(binding, Mapping), f"authority binding absent: {key}")
    path = safe_file(ROOT / str(binding.get("path", "")), f"binding {key}")
    require(
        path.stat().st_size == int(binding.get("bytes", -1))
        and file_sha256(path) == binding.get("sha256"),
        f"authority binding drift: {key}",
    )
    return path


def load_authority(
    path: Path,
    *,
    fold: int,
    start: int,
    stop: int,
    ledger: Path,
    schedule: Path,
    cache_index: Path,
    geometry: Path,
    cache_root: Path,
    init_checkpoint: Path,
    relative_checkpoint: Path,
    h0_checkpoint: Path,
    output: Path,
) -> tuple[dict[str, Any], str]:
    authority_path = safe_file(path, "V15 authority")
    require(
        stat.S_IMODE(authority_path.stat().st_mode) == 0o444,
        "V15 authority not immutable",
    )
    authority = read_json(authority_path, "V15 authority")
    require(
        authority.get("status") == AUTH_STATUS
        and authority.get("logical_sha256") == logical_sha256(authority)
        and authority.get("quarantine_precompute_authorized") is True
        and authority.get("outer_oof_model_forward_authorized") is True
        and authority.get("training_authorized") is False
        and authority.get("model_backward_authorized") is False
        and authority.get("model_update_authorized") is False
        and authority.get("metric_computation_authorized") is False
        and authority.get("aggregate_reducer_authorized") is False
        and authority.get("unquarantine_authorized") is False
        and authority.get("scientific_decision_authorized") is False
        and authority.get("scientific_GO_or_NO_GO") is None
        and authority.get("automatic_stage_advance") is False
        and authority.get("next_authorized_stage") is None,
        "V15 quarantine authority envelope drift",
    )
    scoring = authority.get("scoring_contract", {})
    require(
        scoring.get("arms") == list(ARM_ORDER)
        and scoring.get("roles") == ["TARGET", "Q0_DONOR"]
        and scoring.get("directions_per_role") == 2
        and scoring.get("raw_logits_only") is True
        and scoring.get("torch_eval_required") is True
        and scoring.get("torch_no_grad_required") is True
        and scoring.get("target_lock_query_union_reused_for_target_and_donor")
        is True
        and scoring.get("structural_h0_exact_zero_required") is True
        and scoring.get("structural_h0_target_and_donor_forward_count") == 0,
        "V15 scoring contract drift",
    )
    cache_contract = authority.get("cache_contract", {})
    require(
        cache_contract.get("relative_path") == CACHE_ROOT_RELATIVE
        and cache_contract.get("directory_not_hash_bound") is True
        and cache_contract.get("payloads_verified_via_redacted_cache_index") is True
        and cache_root.resolve() == (ROOT / CACHE_ROOT_RELATIVE).resolve(),
        "V15 cache-root contract drift",
    )
    expected_paths = {
        "outer_ledger": ledger,
        "redacted_schedule": schedule,
        "redacted_cache_index": cache_index,
        "geometry_payload": geometry,
        f"init_checkpoint_fold{fold}": init_checkpoint,
        f"relative_primary_checkpoint_fold{fold}": relative_checkpoint,
        f"h0_checkpoint_fold{fold}": h0_checkpoint,
        "scorer": Path(__file__),
    }
    for key, supplied in expected_paths.items():
        require(
            supplied.resolve(strict=True) == bound_path(authority, key),
            f"V15 supplied path differs from binding: {key}",
        )
    chunks = authority.get("chunk_contract", {}).get("chunks", [])
    matches = [
        row
        for row in chunks
        if int(row.get("outer_fold", -1)) == fold
        and int(row.get("chunk_start", -1)) == start
        and int(row.get("chunk_stop", -1)) == stop
    ]
    require(
        len(matches) == 1 and int(matches[0].get("row_count", -1)) == stop - start,
        "V15 chunk authorization absent",
    )
    pattern = authority["chunk_contract"]["output_pattern"]
    expected_output = ROOT / pattern.format(
        outer_fold=fold, chunk_start=start, chunk_stop=stop
    )
    require(
        output.resolve(strict=False) == expected_output.resolve(strict=False),
        "V15 output path drift",
    )
    return authority, file_sha256(authority_path)


def fold_matched_rows(
    ledger: Mapping[str, Any], fold: int, start: int, stop: int
) -> tuple[Mapping[str, Any], ...]:
    require(fold in (1, 2, 3, 4), "outer fold drift")
    require(
        ledger.get("status") == LEDGER_STATUS
        and ledger.get("logical_sha256") == logical_sha256(ledger)
        and ledger.get("query_count") == 594
        and ledger.get("matched_count") == 567,
        "outer ledger envelope drift",
    )
    rows = tuple(
        sorted(
            (
                row
                for row in ledger.get("rows", ())
                if int(row.get("outer_fold", -1)) == fold
                and row.get("status") == "MATCHED"
            ),
            key=lambda row: (int(row["outer_fold"]), int(row["execution_ordinal"])),
        )
    )
    expected = {1: 145, 2: 140, 3: 136, 4: 146}[fold]
    require(len(rows) == expected, "fold matched population drift")
    require(
        type(start) is int
        and type(stop) is int
        and 0 <= start < stop <= len(rows)
        and stop - start <= 12,
        "fold-local chunk bounds drift",
    )
    return rows[start:stop]


def select_target_direction_records(
    artifact: Mapping[str, Any], outer_row: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolve exactly the two outer-refit target records sealed by U4."""

    require(
        artifact.get("schema_version")
        == "rc_dino_rcde_sr0_mt_p_v2_formal_lock_artifact_v1_20260819"
        and artifact.get("target_free") is True
        and artifact.get("record_count") == 1024
        and int(artifact.get("execution_ordinal", -1))
        == int(outer_row["execution_ordinal"])
        and artifact.get("query_id") == outer_row["query_id"],
        "P-lock artifact envelope drift",
    )
    target = outer_row["target"]
    expected_hashes = tuple(target["direction_record_sha256"])
    require(len(expected_hashes) == 2 and len(set(expected_hashes)) == 2, "target direction hash drift")
    index = {
        record["record_sha256"]: record
        for record in artifact.get("records", ())
        if isinstance(record, Mapping)
    }
    require(len(index) == 1024, "P-lock record hash collision")
    records = tuple(index.get(digest) for digest in expected_hashes)
    require(all(isinstance(record, Mapping) for record in records), "target direction record absent")
    by_direction = {record["direction"]: record for record in records}  # type: ignore[index]
    require(set(by_direction) == set(FIXED_DIRECTIONS), "target direction population drift")
    ordered = tuple(by_direction[direction] for direction in FIXED_DIRECTIONS)
    fold = int(outer_row["outer_fold"])
    for record in ordered:
        candidate = record["candidate"]
        query = record["query"]
        require(
            candidate["candidate_key"] == target["candidate_key"]
            and int(candidate["candidate_physical_row"])
            == int(target["candidate_physical_row"])
            and candidate["candidate_reference_source_sha256"]
            == target["candidate_reference_source_sha256"]
            and query["fit_id"] == f"P_OUTER{fold}_OUTER_REFIT"
            and query["crossfit_role"] == "OUTER_HELDOUT_DEPLOYMENT"
            and int(query["outer_fold"]) == fold
            and query["inner_heldout_fold"] is None
            and query["query_source_image_sha256"]
            == outer_row["query_source_image_sha256"],
            "target outer-refit record drift",
        )
    return ordered  # type: ignore[return-value]


def validate_checkpoint_payload(
    value: Mapping[str, Any], *, arm: str, fold: int, checkpoint_sha256: str
) -> Mapping[str, torch.Tensor]:
    """Validate a checkpoint without relying on a training validator result."""

    expected = {
        "INIT": (
            "DINO_RCDE_R1_MAIN_CHECKPOINT_COMPLETE",
            {"RCDE_CONTEXT", "RCDE_CONTEXT_INITIALIZATION"},
        ),
        "TRACK_R": (
            "DINO_RCDE_TRACK_R_RELATIVE_CHECKPOINT_COMPLETE",
            {"RCDE_CONTEXT_TRACK_R_RELATIVE"},
        ),
        "TRACK_H": (
            "H0_FULL_REFERENCE_UNARY_CHECKPOINT_COMPLETE",
            {"RCDE_CONTEXT_H0_FULL_REFERENCE_UNARY"},
        ),
    }
    require(arm in expected, "checkpoint arm drift")
    status, names = expected[arm]
    state = value.get("model_state_dict")
    require(
        value.get("status") == status
        and value.get("arm") in names
        and int(value.get("outer_fold", -1)) == fold
        and int(value.get("update", -1)) == 2048
        and int(value.get("seed", -1)) == 17
        and isinstance(value.get("final_state_sha256"), str)
        and isinstance(state, Mapping)
        and len(state) == 17
        and len(checkpoint_sha256) == 64,
        f"{arm} checkpoint envelope drift",
    )
    return state  # type: ignore[return-value]


@dataclass(frozen=True)
class MaterializedRow:
    outer_row: Mapping[str, Any]
    query: QueryTokenFieldV1
    target: CandidateReferenceFieldV1
    donor: CandidateReferenceFieldV1
    query_masks: Mapping[str, torch.Tensor]
    structural_ready: Mapping[str, bool]
    directional_receipts: tuple[Mapping[str, Any], Mapping[str, Any]]


def _move_query(value: QueryTokenFieldV1, device: torch.device) -> QueryTokenFieldV1:
    return QueryTokenFieldV1(
        layers=value.layers.to(device),
        grid_shape=value.grid_shape,
        valid_patch_mask=value.valid_patch_mask,
        source_image_sha256=value.source_image_sha256,
        source_key=value.source_key,
        cache_payload_sha256=value.cache_payload_sha256,
        geometry_record_sha256=value.geometry_record_sha256,
        tokens_sha256=value.tokens_sha256,
    )


def _move_candidate(
    value: CandidateReferenceFieldV1, device: torch.device
) -> CandidateReferenceFieldV1:
    return CandidateReferenceFieldV1(
        candidate_key=value.candidate_key,
        layers=value.layers.to(device),
        grid_shape=value.grid_shape,
        valid_patch_mask=value.valid_patch_mask,
        physical_gallery_row=value.physical_gallery_row,
        source_image_sha256=value.source_image_sha256,
        source_key=value.source_key,
        cache_payload_sha256=value.cache_payload_sha256,
        geometry_record_sha256=value.geometry_record_sha256,
        tokens_sha256=value.tokens_sha256,
    )


def materialize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_root: Path,
    cache_index: Mapping[tuple[str, int], Mapping[str, object]],
    schedule_by_query: Mapping[str, Mapping[str, Any]],
    query_geometry: Mapping[int, Mapping[str, Any]],
    reference_geometry: Mapping[int, Mapping[str, Any]],
) -> tuple[MaterializedRow, ...]:
    queries: dict[int, QueryTokenFieldV1] = {}
    references: dict[int, CandidateReferenceFieldV1] = {}
    output = []
    for outer_row in rows:
        fold = int(outer_row["outer_fold"])
        execution = int(outer_row["execution_ordinal"])
        schedule = schedule_by_query.get(str(outer_row["query_id"]))
        require(
            isinstance(schedule, Mapping)
            and int(schedule["execution_ordinal"]) == execution
            and int(schedule["heldout_fold"]) == fold
            and schedule["source_image_sha256"]
            == outer_row["query_source_image_sha256"],
            "outer ledger/redacted schedule drift",
        )
        target_address = outer_row["target"]
        donor_address = outer_row["donor"]
        target_row = int(target_address["candidate_physical_row"])
        donor_key, donor_row_raw, donor_source = donor_address["payload_key"]
        donor_row = int(donor_row_raw)
        c128 = {int(item) for item in schedule["candidate_physical_rows"]}
        require(target_row in c128 and donor_row not in c128, "target/donor C128 role drift")
        q_geometry = query_geometry.get(execution)
        target_geometry = reference_geometry.get(target_row)
        donor_geometry = reference_geometry.get(donor_row)
        require(
            isinstance(q_geometry, Mapping)
            and isinstance(target_geometry, Mapping)
            and isinstance(donor_geometry, Mapping),
            "canonical geometry absent",
        )
        query = queries.get(execution)
        if query is None:
            query = v_input.make_query_field(
                {
                    "execution_ordinal": execution,
                    "query_source_image_sha256": outer_row[
                        "query_source_image_sha256"
                    ],
                },
                cache_root=cache_root,
                index=cache_index,
                expected_model_sha256=CACHE_MODEL_CHECKPOINT_LOGICAL_SHA256,
            )
            require(
                query.source_image_sha256 == q_geometry["source_image_sha256"],
                "query token/geometry source drift",
            )
            queries[execution] = query

        def reference(row: int, source: str) -> CandidateReferenceFieldV1:
            value = references.get(row)
            if value is None:
                value = v_input.make_candidate_field(
                    row,
                    source,
                    cache_root=cache_root,
                    index=cache_index,
                    expected_model_sha256=CACHE_MODEL_CHECKPOINT_LOGICAL_SHA256,
                )
                references[row] = value
            require(value.source_image_sha256 == source, "reference cache source drift")
            return value

        target = reference(
            target_row, str(target_address["candidate_reference_source_sha256"])
        )
        donor = reference(donor_row, str(donor_source))
        require(
            target.candidate_key == target_address["candidate_key"]
            and donor.candidate_key == donor_key
            and donor.candidate_key != target.candidate_key
            and target.source_image_sha256 == target_geometry["source_image_sha256"]
            and donor.source_image_sha256 == donor_geometry["source_image_sha256"],
            "target/donor token binding drift",
        )
        artifact_path = safe_file(ROOT / outer_row["lock_artifact_path"], "P-lock artifact")
        require(file_sha256(artifact_path) == outer_row["lock_artifact_sha256"], "P-lock artifact hash drift")
        artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=True, mmap=True
        )
        require(isinstance(artifact, Mapping), "P-lock artifact object drift")
        records = select_target_direction_records(artifact, outer_row)
        projection = project_natural_v2_candidate_to_v1_runtime(
            query=query,
            candidate=target,
            direction_records=records,
            spec=outer_refit_head_spec(source_fold=fold),
            query_geometry_binding=GeometryNamespaceBindingV1(
                source_image_sha256=query.source_image_sha256,
                grid_shape=query.grid_shape,
                canonical_geometry_sha256=q_geometry["dino_geometry"][
                    "geometry_sha256"
                ],
                cache_geometry_record_sha256=query.geometry_record_sha256,
            ),
            reference_geometry_binding=GeometryNamespaceBindingV1(
                source_image_sha256=target.source_image_sha256,
                grid_shape=target.grid_shape,
                canonical_geometry_sha256=target_geometry["dino_geometry"][
                    "geometry_sha256"
                ],
                cache_geometry_record_sha256=target.geometry_record_sha256,
            ),
        )
        direction_locks = projection.runtime_lock.direction_locks
        masks = {
            direction: direction_locks[direction].query_union_mask
            for direction in FIXED_DIRECTIONS
        }
        ready = {}
        for direction in FIXED_DIRECTIONS:
            lock = direction_locks[direction]
            require(lock.status in {LOCK_READY, LOCK_H0}, "direction lock status drift")
            ready[direction] = lock.status == LOCK_READY
            require(
                bool(masks[direction].any()) == ready[direction],
                "direction structural-ready/mask drift",
            )
        receipts = tuple(
            {
                "direction": direction,
                "p_lock_record_sha256": record["record_sha256"],
                "query_union_sha256": record["fixed_denominator"][
                    "query_union_sha256"
                ],
                "structural_ready": ready[direction],
                "forward_count_per_model": 2 if ready[direction] else 0,
            }
            for direction, record in zip(FIXED_DIRECTIONS, records)
        )
        output.append(
            MaterializedRow(
                outer_row=outer_row,
                query=query,
                target=target,
                donor=donor,
                query_masks=masks,
                structural_ready=ready,
                directional_receipts=receipts,  # type: ignore[arg-type]
            )
        )
    require(len(output) == len(rows), "materialized row count drift")
    return tuple(output)


def score_one_model(
    model: DINO_RCDE_V1_2,
    rows: Sequence[MaterializedRow],
    *,
    checkpoint_sha256: str,
    device: torch.device,
    decoder: Callable[..., Any] = decode_full_reference_unary,
) -> tuple[dict[str, Mapping[str, Any]], str, str, int]:
    """Return raw unary logits only; never compute a scientific statistic."""

    model.eval()
    before = state_dict_sha256(model)
    output: dict[str, Mapping[str, Any]] = {}
    forward_count = 0
    with torch.no_grad():
        for item in rows:
            query = _move_query(item.query, device)
            target = _move_candidate(item.target, device)
            donor = _move_candidate(item.donor, device)
            target_logits: dict[str, float] = {}
            donor_logits: dict[str, float] = {}
            for direction in FIXED_DIRECTIONS:
                if item.structural_ready[direction]:
                    target_value = decoder(
                        model,
                        query,
                        target,
                        item.query_masks[direction],
                        structural_ready=True,
                        streaming_chunk_size=64,
                    ).score
                    donor_value = decoder(
                        model,
                        query,
                        donor,
                        item.query_masks[direction],
                        structural_ready=True,
                        streaming_chunk_size=64,
                    ).score
                    forward_count += 2
                else:
                    require(
                        not bool(item.query_masks[direction].any()),
                        "structural-H0 query union is not empty",
                    )
                    target_value = exact_zero_unary(model)
                    donor_value = exact_zero_unary(model)
                require(
                    target_value.ndim == donor_value.ndim == 0
                    and bool(torch.isfinite(target_value))
                    and bool(torch.isfinite(donor_value)),
                    "nonfinite unary precompute logit",
                )
                target_logits[direction] = float(target_value.detach().cpu())
                donor_logits[direction] = float(donor_value.detach().cpu())
                if not item.structural_ready[direction]:
                    require(
                        target_logits[direction] == 0.0
                        and donor_logits[direction] == 0.0,
                        "structural-H0 unary is not exact zero",
                    )
            key = str(item.outer_row["record_sha256"])
            require(key not in output, "outer row score collision")
            output[key] = {
                "checkpoint_sha256": checkpoint_sha256,
                "target": target_logits,
                "donor": donor_logits,
            }
            del query, target, donor
    after = state_dict_sha256(model)
    require(before == after, "no-grad precompute mutated model")
    return output, before, after, forward_count


def load_model(
    path: Path, *, arm: str, fold: int, device: torch.device
) -> tuple[DINO_RCDE_V1_2, dict[str, Any]]:
    checkpoint_path = safe_file(path, f"{arm} checkpoint")
    checkpoint_sha = file_sha256(checkpoint_path)
    value = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    require(isinstance(value, Mapping), f"{arm} checkpoint object drift")
    state = validate_checkpoint_payload(
        value, arm=arm, fold=fold, checkpoint_sha256=checkpoint_sha
    )
    model = DINO_RCDE_V1_2().to(device)
    model.load_state_dict(state, strict=True)
    observed = state_dict_sha256(model)
    require(observed == value["final_state_sha256"], f"{arm} checkpoint state drift")
    receipt = {
        "checkpoint_sha256": checkpoint_sha,
        "final_state_sha256": value["final_state_sha256"],
        "status": value["status"],
        "arm": value["arm"],
        "update": 2048,
    }
    return model, receipt


def build_output_rows(
    rows: Sequence[MaterializedRow],
    scores: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for item in rows:
        outer = item.outer_row
        target = outer["target"]
        donor = outer["donor"]
        outer_sha = str(outer["record_sha256"])
        arms = {arm: scores[arm][outer_sha] for arm in ARM_ORDER}
        row = {
            "outer_ledger_record_sha256": outer_sha,
            "outer_fold": int(outer["outer_fold"]),
            "execution_ordinal": int(outer["execution_ordinal"]),
            "query_id": outer["query_id"],
            "query_source_image_sha256": outer["query_source_image_sha256"],
            "recipient_supergroup_sha256": outer["recipient_supergroup_sha256"],
            "track": outer["track"],
            "p_lock_artifact_sha256": outer["lock_artifact_sha256"],
            "target": {
                "candidate_key": target["candidate_key"],
                "candidate_physical_row": int(target["candidate_physical_row"]),
                "candidate_reference_source_sha256": target[
                    "candidate_reference_source_sha256"
                ],
            },
            "donor": {
                "candidate_key": donor["payload_key"][0],
                "candidate_physical_row": int(donor["payload_key"][1]),
                "candidate_reference_source_sha256": donor["payload_key"][2],
            },
            "directional_receipts": list(item.directional_receipts),
            "arms": arms,
        }
        row["record_sha256"] = canonical_sha256(row)
        output.append(row)
    return output


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = path.resolve()
    require(ROOT in output.parents and not output.exists(), "output path drift/exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--chunk-start", type=int, required=True)
    parser.add_argument("--chunk-stop", type=int, required=True)
    parser.add_argument("--redacted-schedule", type=Path, required=True)
    parser.add_argument("--redacted-cache-index", type=Path, required=True)
    parser.add_argument("--geometry-payload", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--relative-checkpoint", type=Path, required=True)
    parser.add_argument("--h0-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    authority, authority_sha = load_authority(
        args.authority,
        fold=args.fold,
        start=args.chunk_start,
        stop=args.chunk_stop,
        ledger=args.ledger,
        schedule=args.redacted_schedule,
        cache_index=args.redacted_cache_index,
        geometry=args.geometry_payload,
        cache_root=args.cache_root,
        init_checkpoint=args.init_checkpoint,
        relative_checkpoint=args.relative_checkpoint,
        h0_checkpoint=args.h0_checkpoint,
        output=args.output,
    )

    ledger_path = safe_file(args.ledger, "U4 outer ledger")
    ledger = read_json(ledger_path, "U4 outer ledger")
    selected = fold_matched_rows(
        ledger, args.fold, args.chunk_start, args.chunk_stop
    )
    schedule_path = safe_file(args.redacted_schedule, "redacted schedule")
    cache_index_path = safe_file(args.redacted_cache_index, "redacted cache index")
    geometry_path = safe_file(args.geometry_payload, "canonical geometry")
    cache_root = safe_directory(args.cache_root, "stable token cache")
    cache_index, schedule_by_query, cache_index_sha, schedule_sha, schedule_logical = (
        v_input.load_schedule_cache_index(schedule_path, cache_index_path)
    )
    query_geometry, reference_geometry, geometry_sha = track_input.load_geometry_payload(
        geometry_path
    )
    materialized = materialize_rows(
        selected,
        cache_root=cache_root,
        cache_index=cache_index,
        schedule_by_query=schedule_by_query,
        query_geometry=query_geometry,
        reference_geometry=reference_geometry,
    )

    checkpoint_paths = {
        "INIT": args.init_checkpoint,
        "TRACK_R": args.relative_checkpoint,
        "TRACK_H": args.h0_checkpoint,
    }
    scores: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    checkpoint_receipts: dict[str, Any] = {}
    model_forward_count = 0
    for arm in ARM_ORDER:
        model, receipt = load_model(
            checkpoint_paths[arm], arm=arm, fold=args.fold, device=device
        )
        arm_scores, before, after, arm_forward_count = score_one_model(
            model,
            materialized,
            checkpoint_sha256=receipt["checkpoint_sha256"],
            device=device,
        )
        receipt["model_state_before_sha256"] = before
        receipt["model_state_after_sha256"] = after
        receipt["no_grad_state_unchanged"] = before == after
        receipt["model_forward_count"] = arm_forward_count
        checkpoint_receipts[arm] = receipt
        scores[arm] = arm_scores
        model_forward_count += arm_forward_count
        del model
        torch.cuda.empty_cache()
    require(
        checkpoint_receipts["TRACK_R"].get("model_state_before_sha256")
        != checkpoint_receipts["TRACK_H"].get("model_state_before_sha256"),
        "Track-R/Track-H states unexpectedly identical",
    )
    ready_direction_count = sum(
        int(item.structural_ready[direction])
        for item in materialized
        for direction in FIXED_DIRECTIONS
    )
    require(
        model_forward_count == ready_direction_count * len(ARM_ORDER) * 2,
        "model forward accounting drift",
    )
    rows = build_output_rows(materialized, scores)
    value = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "QUARANTINED_RAW_OUTER_OOF_UNARY_LOGITS_ONLY",
        "authority_sha256": authority_sha,
        "authority_logical_sha256": authority["logical_sha256"],
        "outer_fold": args.fold,
        "chunk_start": args.chunk_start,
        "chunk_stop": args.chunk_stop,
        "row_count": len(rows),
        "outer_ledger_sha256": file_sha256(ledger_path),
        "outer_ledger_logical_sha256": ledger["logical_sha256"],
        "input_receipts": {
            "redacted_schedule_sha256": schedule_sha,
            "redacted_schedule_logical_sha256": schedule_logical,
            "redacted_cache_index_sha256": cache_index_sha,
            "geometry_payload_sha256": geometry_sha,
            "cache_model_checkpoint_logical_sha256": CACHE_MODEL_CHECKPOINT_LOGICAL_SHA256,
        },
        "checkpoint_receipts": checkpoint_receipts,
        "rows": rows,
        "row_population_sha256": canonical_sha256(rows),
        "model_load_count": 3,
        "structural_ready_direction_count": ready_direction_count,
        "structural_h0_direction_count": sum(
            int(not item.structural_ready[direction])
            for item in materialized
            for direction in FIXED_DIRECTIONS
        ),
        "model_forward_count": model_forward_count,
        "no_grad_forward_count": model_forward_count,
        "model_backward_count": 0,
        "model_update_count": 0,
        "scientific_metric_count": 0,
        "reducer_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(args.output, value)
    print(
        json.dumps(
            {
                "status": STATUS,
                "outer_fold": args.fold,
                "chunk_start": args.chunk_start,
                "chunk_stop": args.chunk_stop,
                "row_count": len(rows),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
