"""Shared, fail-closed materialization helpers for formal SR0-MT V inputs.

Natural cache payloads are opened only by explicit caller invocation.  The
candidate address is imported from the frozen P natural-source module; this
file never invents a second candidate namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

RC_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RC_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: E402
    CandidateReferenceFieldV1,
    QueryTokenFieldV1,
    token_tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_source_v1 import (  # noqa: E402
    candidate_key_v1,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    LOCK_LEDGER_SCHEMA,
    P_DIRECTIONS,
    canonical_sha256,
    file_sha256,
    load_json_mapping,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CandidatePLockV1,
    VPairEpisode,
    VQueryBundle,
    candidate_axis_sha256,
    sealed_direction_lock_from_p_record,
)

# Reuse the already independently qualified redacted-cache parser rather than
# implementing a subtly different cache addressing convention.
import materialize_dino_rcde_r1_oof_prejoin_v3 as redacted  # noqa: E402


EPISODE_LEDGER_SCHEMA = "rc_dino_rcde_sr0_mt_v_training_ledger_v1"
EPISODE_VALIDATION_SCHEMA = "rc_dino_rcde_sr0_mt_v_training_ledger_validation_v1"
QUERY_INPUT_SCHEMA = "rc_dino_rcde_sr0_mt_three_arm_query_input_v1"
QUERY_VALIDATION_SCHEMA = "rc_dino_rcde_sr0_mt_query_bundle_validation_v1"
CONTROL_INPUT_SCHEMA = "rc_dino_rcde_sr0_mt_control_query_input_v1"
CONTROL_INPUT_VALIDATION_SCHEMA = "rc_dino_rcde_sr0_mt_control_input_validation_v1"
CONTROL_SCORE_SOURCE_INDEX_SCHEMA = (
    "rc_dino_rcde_sr0_mt_control_score_source_index_v1"
)
CONTROL_SCORE_MANIFEST_SCHEMA = "rc_dino_rcde_sr0_mt_control_score_manifest_v1"
CONTROL_SCORE_VALIDATION_SCHEMA = (
    "rc_dino_rcde_sr0_mt_control_score_validation_v1"
)
EXPECTED_REDACTED_QUERY_COUNT = 600
EXPECTED_CANDIDATE_COUNT = 128
EXPECTED_REFERENCE_UNION_COUNT = 4_748
EXPECTED_FOLD_COUNTS = {1: 151, 2: 150, 3: 149, 4: 150}
P_LOCK_VALIDATION_PASS = "RCDE_SR0_MT_P_LOCK_INDEPENDENT_VALIDATION_PASS"
I0_POSTJOIN_PASS = "RCDE_SR0_MT_I0_POSTJOIN_INDEPENDENT_VALIDATION_PASS"


class VInputContractError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VInputContractError(f"JSON input is not an object: {path}")
    return value


def safe_path(path: Path, *, must_exist: bool = True) -> Path:
    value = Path(path).resolve()
    if RC_ROOT != value and RC_ROOT not in value.parents:
        raise VInputContractError(f"path escapes RC root: {value}")
    if {"c8", "s8", "opened", "sealed"}.intersection(
        part.lower() for part in value.parts
    ):
        raise VInputContractError(f"protected path requested: {value}")
    if must_exist and (not value.is_file() or value.is_symlink()):
        raise VInputContractError(f"input is absent or unsafe: {value}")
    return value


def atomic_torch(path: Path, value: object) -> None:
    output = safe_path(path, must_exist=False)
    if output.exists():
        raise VInputContractError(f"immutable output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, output)


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    output = safe_path(path, must_exist=False)
    if output.exists():
        raise VInputContractError(f"immutable output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _logical(value: Mapping[str, object], *, exclude: Sequence[str] = ()) -> str:
    excluded = {"logical_sha256", *exclude}
    return canonical_sha256({key: item for key, item in value.items() if key not in excluded})


def validate_lock_ledger(
    ledger_path: Path, validation_path: Path, *, expected_outer_fold: int
) -> dict[str, Any]:
    ledger_path = safe_path(ledger_path)
    validation_path = safe_path(validation_path)
    ledger = read_json(ledger_path)
    validation = read_json(validation_path)
    if (
        ledger.get("schema_version") != LOCK_LEDGER_SCHEMA
        or ledger.get("target_free") is not True
        or int(ledger.get("outer_fold", -1)) != expected_outer_fold
        or validation.get("status") != P_LOCK_VALIDATION_PASS
        or validation.get("validation_pass") is not True
        or validation.get("lock_ledger_file_sha256") != file_sha256(ledger_path)
        or ledger.get("logical_sha256")
        != _logical(ledger, exclude=("records",))
    ):
        raise VInputContractError("P lock independent-validation closure failed")
    records = ledger.get("records")
    metadata = ledger.get("query_metadata")
    if not isinstance(records, list) or not isinstance(metadata, list):
        raise VInputContractError("P lock records/query metadata absent")
    expected = int(ledger.get("query_count", -1)) * 128 * 2
    if len(records) != expected or int(ledger.get("record_count", -1)) != expected:
        raise VInputContractError("P lock ledger is not complete query x C128 x 2")
    return ledger


def lock_indices(
    ledgers: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Mapping[str, object]], dict[tuple[str, int, str], Mapping[str, object]]]:
    metadata: dict[str, Mapping[str, object]] = {}
    records: dict[tuple[str, int, str], Mapping[str, object]] = {}
    for ledger in ledgers:
        for raw in ledger["query_metadata"]:
            if not isinstance(raw, Mapping):
                raise VInputContractError("P lock query metadata row drift")
            query_id = str(raw["query_id"])
            if query_id in metadata:
                raise VInputContractError("P lock query duplicated across scopes")
            metadata[query_id] = raw
        for raw in ledger["records"]:
            if not isinstance(raw, Mapping):
                raise VInputContractError("P lock record row drift")
            key = (
                str(raw["query_id"]),
                int(raw["candidate_physical_row"]),
                str(raw["direction"]),
            )
            if key in records:
                raise VInputContractError("P lock record address collision")
            records[key] = raw
    return metadata, records


def load_schedule_cache_index(
    schedule_path: Path,
    cache_index_path: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
    str,
    str,
]:
    """Load the one frozen 600-query schedule and its exact 4,748-row cache.

    V materializers must not accept a cache index merely because every address
    they happen to touch exists.  The full cache is part of the experimental
    population: it must be bound to the exact redacted schedule and to the
    complete physical-reference union before any query or episode is decoded.
    """

    schedule_path = safe_path(schedule_path)
    cache_index_path = safe_path(cache_index_path)
    schedule, _ = redacted.load_redacted_schedule(
        redacted_schedule=schedule_path,
        prejoin_folds=None,
        model_visible_c128=None,
        expected_query_count=EXPECTED_REDACTED_QUERY_COUNT,
        expected_candidate_count=EXPECTED_CANDIDATE_COUNT,
        expected_fold_counts=EXPECTED_FOLD_COUNTS,
    )
    schedule_payload = read_json(schedule_path)
    schedule_logical = str(schedule_payload.get("logical_sha256", ""))
    reference_rows = {
        int(row)
        for item in schedule
        for row in item["candidate_physical_rows"]
    }
    if len(reference_rows) != EXPECTED_REFERENCE_UNION_COUNT:
        raise VInputContractError(
            "redacted schedule reference union is not the frozen 4,748 rows"
        )
    index, index_sha = redacted.load_redacted_cache_index(
        cache_index_path,
        expected_query_count=EXPECTED_REDACTED_QUERY_COUNT,
        expected_reference_rows=reference_rows,
        expected_schedule_logical_sha256=schedule_logical,
    )
    redacted.validate_schedule_cache_index_global_join(schedule, index)
    by_query: dict[str, dict[str, Any]] = {}
    for item in schedule:
        query_id = str(item["query_id"])
        if query_id in by_query:
            raise VInputContractError("redacted schedule query ID collision")
        by_query[query_id] = dict(item)
    if len(by_query) != EXPECTED_REDACTED_QUERY_COUNT:
        raise VInputContractError("redacted schedule query population drift")
    return (
        index,
        by_query,
        index_sha,
        file_sha256(schedule_path),
        schedule_logical,
    )


def assert_meta_matches_schedule(
    meta: Mapping[str, object], schedule_row: Mapping[str, object]
) -> None:
    rows = [int(item) for item in meta.get("candidate_physical_rows", [])]
    if not (
        str(meta.get("query_id", "")) == str(schedule_row.get("query_id", ""))
        and int(meta.get("execution_ordinal", -1))
        == int(schedule_row.get("execution_ordinal", -2))
        and int(meta.get("outer_fold", -1))
        == int(schedule_row.get("heldout_fold", -2))
        and str(meta.get("query_source_image_sha256", ""))
        == str(schedule_row.get("source_image_sha256", ""))
        and rows == [int(item) for item in schedule_row.get("candidate_physical_rows", [])]
        and meta.get("candidate_axis_sha256")
        == schedule_row.get("candidate_axis_sha256")
    ):
        raise VInputContractError("P-lock metadata/redacted schedule closure drift")


def _cache_payload(
    *,
    cache_root: Path,
    index: Mapping[tuple[str, int], Mapping[str, object]],
    kind: str,
    key: int,
    expected_model_sha256: str,
    expected_source_image_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], Mapping[str, object], str]:
    address = (kind, int(key))
    if address not in index:
        raise VInputContractError(f"redacted cache address absent: {address}")
    path = redacted.canonical_cache_path(safe_path(cache_root, must_exist=False), kind, key)
    path = safe_path(path)
    payload = redacted.safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise VInputContractError("redacted cache payload is not a mapping")
    layers, mask, grid = redacted.validate_cache_payload(
        payload,
        expected_model_sha256=expected_model_sha256,
        index_record=index[address],
        expected_source_image_sha256=expected_source_image_sha256,
    )
    geometry = payload.get("geometry_receipt")
    if not isinstance(geometry, Mapping) or not isinstance(
        geometry.get("logical_sha256"), str
    ):
        raise VInputContractError("redacted cache geometry receipt absent")
    return (
        layers.to(dtype=torch.float32).contiguous(),
        mask.to(dtype=torch.bool).contiguous(),
        grid,
        payload,
        file_sha256(path),
    )


def make_query_field(
    meta: Mapping[str, object],
    *,
    cache_root: Path,
    index: Mapping[tuple[str, int], Mapping[str, object]],
    expected_model_sha256: str,
) -> QueryTokenFieldV1:
    execution = int(meta["execution_ordinal"])
    source = str(meta["query_source_image_sha256"])
    layers, mask, grid, payload, payload_sha = _cache_payload(
        cache_root=cache_root,
        index=index,
        kind="query",
        key=execution,
        expected_model_sha256=expected_model_sha256,
        expected_source_image_sha256=source,
    )
    geometry = payload["geometry_receipt"]
    return QueryTokenFieldV1(
        layers=layers,
        grid_shape=grid,
        valid_patch_mask=mask,
        source_image_sha256=source,
        source_key=f"query_execution:{execution}",
        cache_payload_sha256=payload_sha,
        geometry_record_sha256=str(geometry["logical_sha256"]),
        tokens_sha256=token_tensor_sha256(layers),
    )


def make_candidate_field(
    row: int,
    source_sha256: str,
    *,
    cache_root: Path,
    index: Mapping[tuple[str, int], Mapping[str, object]],
    expected_model_sha256: str,
) -> CandidateReferenceFieldV1:
    layers, mask, grid, payload, payload_sha = _cache_payload(
        cache_root=cache_root,
        index=index,
        kind="reference",
        key=row,
        expected_model_sha256=expected_model_sha256,
        expected_source_image_sha256=source_sha256,
    )
    geometry = payload["geometry_receipt"]
    return CandidateReferenceFieldV1(
        candidate_key=candidate_key_v1(
            physical_row=row, source_image_sha256=source_sha256
        ),
        layers=layers,
        grid_shape=grid,
        valid_patch_mask=mask,
        physical_gallery_row=row,
        source_image_sha256=source_sha256,
        source_key=f"reference_physical_row:{row}",
        cache_payload_sha256=payload_sha,
        geometry_record_sha256=str(geometry["logical_sha256"]),
        tokens_sha256=token_tensor_sha256(layers),
    )


def candidate_source_by_row(meta: Mapping[str, object]) -> dict[int, str]:
    axis = meta.get("candidate_axis")
    if not isinstance(axis, list) or len(axis) != 128:
        raise VInputContractError("P lock candidate binding axis is not C128")
    result: dict[int, str] = {}
    for item in axis:
        if not isinstance(item, Mapping):
            raise VInputContractError("candidate binding row drift")
        row = int(item["candidate_physical_row"])
        source = str(item["candidate_reference_source_sha256"])
        expected_key = candidate_key_v1(physical_row=row, source_image_sha256=source)
        if row in result or not expected_key:
            raise VInputContractError("candidate binding axis collision")
        result[row] = source
    rows = list(map(int, meta.get("candidate_physical_rows", [])))
    if rows != sorted(result) or meta.get("candidate_axis_sha256") != canonical_sha256(rows):
        raise VInputContractError("physical-row candidate axis receipt drift")
    return result


def make_candidate_lock(
    query: QueryTokenFieldV1,
    meta: Mapping[str, object],
    records: Mapping[tuple[str, int, str], Mapping[str, object]],
    row: int,
    *,
    cache_root: Path,
    index: Mapping[tuple[str, int], Mapping[str, object]],
    expected_model_sha256: str,
) -> CandidatePLockV1:
    query_id = str(meta["query_id"])
    source = candidate_source_by_row(meta)[row]
    candidate = make_candidate_field(
        row,
        source,
        cache_root=cache_root,
        index=index,
        expected_model_sha256=expected_model_sha256,
    )
    directions = {}
    for direction in P_DIRECTIONS:
        raw = records.get((query_id, row, direction))
        if raw is None:
            raise VInputContractError("candidate direction P lock is absent")
        directions[direction] = sealed_direction_lock_from_p_record(
            raw, query=query, candidate=candidate
        )
    return CandidatePLockV1(candidate=candidate, direction_locks=directions)


def make_query_bundle(
    meta: Mapping[str, object],
    records: Mapping[tuple[str, int, str], Mapping[str, object]],
    *,
    cache_root: Path,
    index: Mapping[tuple[str, int], Mapping[str, object]],
    expected_model_sha256: str,
) -> VQueryBundle:
    query = make_query_field(
        meta,
        cache_root=cache_root,
        index=index,
        expected_model_sha256=expected_model_sha256,
    )
    sources = candidate_source_by_row(meta)
    locks = {}
    for row in sorted(sources):
        lock = make_candidate_lock(
            query,
            meta,
            records,
            row,
            cache_root=cache_root,
            index=index,
            expected_model_sha256=expected_model_sha256,
        )
        locks[lock.candidate.candidate_key] = lock
    rows = sorted(sources)
    return VQueryBundle(
        query_id=str(meta["query_id"]),
        execution_ordinal=int(meta["execution_ordinal"]),
        outer_fold=int(meta["outer_fold"]),
        query=query,
        locks=locks,
        candidate_axis_sha256=candidate_axis_sha256(rows),
    )


def make_pair_episode(
    post: Mapping[str, object],
    meta: Mapping[str, object],
    records: Mapping[tuple[str, int, str], Mapping[str, object]],
    *,
    cache_root: Path,
    index: Mapping[tuple[str, int], Mapping[str, object]],
    expected_model_sha256: str,
) -> VPairEpisode:
    if post.get("target_hit") is not True:
        raise VInputContractError("target-miss query is ineligible for V pair training")
    query = make_query_field(
        meta,
        cache_root=cache_root,
        index=index,
        expected_model_sha256=expected_model_sha256,
    )
    target_row = int(post["target_physical_row"])
    rival_row = int(post["strongest_rival_physical_row"])
    if target_row == rival_row:
        raise VInputContractError("V target/rival physical rows are identical")
    locks = {}
    by_row = {}
    for row in (target_row, rival_row):
        lock = make_candidate_lock(
            query,
            meta,
            records,
            row,
            cache_root=cache_root,
            index=index,
            expected_model_sha256=expected_model_sha256,
        )
        locks[lock.candidate.candidate_key] = lock
        by_row[row] = lock.candidate.candidate_key
    return VPairEpisode(
        episode_id=canonical_sha256(
            {
                "query_id": post["query_id"],
                "execution_ordinal": meta["execution_ordinal"],
                "outer_fold": meta["outer_fold"],
            }
        ),
        query=query,
        locks=locks,
        target_key=by_row[target_row],
        strongest_rival_key=by_row[rival_row],
        outer_fold=int(meta["outer_fold"]),
        identity_key=str(post["identity"]),
        supergroup_key=str(post["supergroup"]),
    )


def bundle_receipt(bundle: VQueryBundle) -> dict[str, object]:
    axis = sorted(
        (lock.candidate.physical_gallery_row, key) for key, lock in bundle.locks.items()
    )
    return {
        "query_id": bundle.query_id,
        "execution_ordinal": bundle.execution_ordinal,
        "outer_fold": bundle.outer_fold,
        "query_tokens_sha256": bundle.query.tokens_sha256,
        "candidate_axis_sha256": bundle.candidate_axis_sha256,
        "candidate_binding_axis_sha256": canonical_sha256(axis),
        "candidate_count": len(axis),
        "p_lock_record_sha256": [
            bundle.locks[key].direction_locks[direction].p_lock_record_sha256
            for _, key in axis
            for direction in P_DIRECTIONS
        ],
    }


__all__ = [
    "EPISODE_LEDGER_SCHEMA",
    "EPISODE_VALIDATION_SCHEMA",
    "QUERY_INPUT_SCHEMA",
    "QUERY_VALIDATION_SCHEMA",
    "CONTROL_INPUT_SCHEMA",
    "CONTROL_INPUT_VALIDATION_SCHEMA",
    "CONTROL_SCORE_SOURCE_INDEX_SCHEMA",
    "CONTROL_SCORE_MANIFEST_SCHEMA",
    "CONTROL_SCORE_VALIDATION_SCHEMA",
    "VInputContractError",
    "read_json",
    "safe_path",
    "atomic_torch",
    "atomic_json",
    "validate_lock_ledger",
    "lock_indices",
    "load_schedule_cache_index",
    "assert_meta_matches_schedule",
    "make_query_bundle",
    "make_pair_episode",
    "bundle_receipt",
    "file_sha256",
    "canonical_sha256",
    "_logical",
]
