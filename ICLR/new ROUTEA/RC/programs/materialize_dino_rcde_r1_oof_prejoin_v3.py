#!/usr/bin/env python3
"""Materialize one immutable, label-free R1 OOF evidence shard.

V3 changes only immutable materialization publication.  Model, data, payload,
receipt, and scientific semantics remain byte-compatible with the
authority-frozen V1.  ``MATERIALIZATION_COMMITTED.json`` binds shard+receipt;
it is deliberately not the later full-scope ``SCOPE_COMMITTED.json``.

Formal execution requires an exact protocol/authority binding; an explicit
test mode exists only for tiny fixtures.  The runner accepts one frozen
arm/fold/checkpoint and an explicit set of *historical* query ordinals.  Cache
lookup is always by the separately frozen dense
``execution_ordinal``.  No target, identity, RAW score/rank, winner, or
correctness field is read or serialized.

The production path uses ``decode_candidate_true_streaming`` (the audited
decoder) for every candidate.  ``decode_candidate_true_streaming_fast`` is
intentionally absent.  The complete pair matrix is evaluated once per
unordered pair and then populated by exact negation.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = (ROOT / "src/rc_aslo_xf/dino_rcde_v1_2_resource_core.py").resolve()
CORE_CANONICAL_MODULE = "rc_aslo_xf.dino_rcde_v1_2_resource_core"
CORE_PINNED_MODULE = "_routea_rcde_r1_oof_prejoin_pinned_core"


class PrejoinAbort(RuntimeError):
    pass


def _load_pinned_core() -> Any:
    """Load the audited core by exact file path, never by ambient import order."""

    ambient = sys.modules.get(CORE_CANONICAL_MODULE)
    if ambient is not None:
        ambient_file = getattr(ambient, "__file__", None)
        if ambient_file is None or Path(ambient_file).resolve() != CORE_PATH:
            raise PrejoinAbort("ambient DINO-RCDE core shadow detected")
    if not CORE_PATH.is_file() or CORE_PATH.is_symlink():
        raise PrejoinAbort("pinned DINO-RCDE core path is absent or symlinked")
    prior = sys.modules.get(CORE_PINNED_MODULE)
    if prior is not None:
        prior_file = getattr(prior, "__file__", None)
        if prior_file is None or Path(prior_file).resolve() != CORE_PATH:
            raise PrejoinAbort("pinned DINO-RCDE core module-name collision")
        module = prior
    else:
        spec = importlib.util.spec_from_file_location(CORE_PINNED_MODULE, CORE_PATH)
        if spec is None or spec.loader is None or Path(str(spec.origin)).resolve() != CORE_PATH:
            raise PrejoinAbort("cannot construct exact DINO-RCDE core import")
        module = importlib.util.module_from_spec(spec)
        sys.modules[CORE_PINNED_MODULE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(CORE_PINNED_MODULE, None)
            raise
    if (
        Path(getattr(module, "__file__", "")).resolve() != CORE_PATH
        or Path(str(getattr(getattr(module, "__spec__", None), "origin", ""))).resolve()
        != CORE_PATH
        or getattr(module.DINO_RCDE_V1_2, "__module__", None)
        != CORE_PINNED_MODULE
    ):
        raise PrejoinAbort("DINO-RCDE core import provenance mismatch")
    return module


_RCDE_CORE = _load_pinned_core()
DINO_RCDE_V1_2 = _RCDE_CORE.DINO_RCDE_V1_2
EXPECTED_PARAMETER_COUNT = _RCDE_CORE.EXPECTED_PARAMETER_COUNT

SCHEMA = "rc_dino_rcde_r1_oof_prejoin_shard_v1_20260814"
RECEIPT_SCHEMA = "rc_dino_rcde_r1_oof_prejoin_receipt_v1_20260814"
STATUS = "DINO_RCDE_R1_OOF_PREJOIN_SHARD_COMPLETE"
REDACTED_SCHEDULE_SCHEMA = "rc_dino_rcde_r1_oof_redacted_schedule_v1_20260814"
REDACTED_CACHE_INDEX_SCHEMA = (
    "rc_dino_rcde_r1_oof_redacted_cache_index_v1_20260814"
)
REDACTED_SCHEDULE_STATUS = "DINO_RCDE_R1_OOF_REDACTED_SCHEDULE_READY"
REDACTED_CACHE_INDEX_STATUS = "DINO_RCDE_R1_OOF_REDACTED_CACHE_INDEX_READY"
REDACTED_INPUTS_VALIDATION_SCHEMA = (
    "rc_dino_rcde_r1_oof_redacted_inputs_validation_v1_20260814"
)
REDACTED_INPUTS_VALIDATION_STATUS = (
    "DINO_RCDE_R1_OOF_REDACTED_INPUTS_VALIDATION_PASS"
)
MATCHED_FOLDSET_MANIFEST_SCHEMA = (
    "rc_dino_rcde_r1_matched_foldset_manifest_v1_0"
)
MATCHED_FOLDSET_MANIFEST_STATUS = (
    "DINO_RCDE_R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_FROZEN"
)
MATCHED_FOLDSET_VALIDATION_SCHEMA = (
    "rc_dino_rcde_r1_matched_foldset_validation_v1_0"
)
MATCHED_FOLDSET_VALIDATION_STATUS = (
    "DINO_RCDE_R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_VALIDATION_PASS"
)
PROTOCOL_SCHEMA = "rc_dino_rcde_r1_oof_prejoin_protocol_v1_20260814"
FORMAL_STAGE = "R1_OOF_PREJOIN_E0"
NEXT_VALIDATION_STAGE = "R1_OOF_PREJOIN_INDEPENDENT_VALIDATION_ONLY"
ALLOWED_ARMS = ("RCDE_BAG", "RCDE_CONTEXT")
EXPECTED_FOLD_COUNTS = {1: 151, 2: 150, 3: 149, 4: 150}
EXPECTED_QUERY_COUNT = 600
EXPECTED_CANDIDATE_COUNT = 128
EXPECTED_UPDATE = 2_048
EXPECTED_SEED = 17
SUMMARY_DIM = 16
PAIR_BATCH_SIZE = 128
EXPECTED_QUERY_TILE_ROWS = 8
EXPECTED_REFERENCE_TILE_ROWS = 8
BAG_QUERY_NAMESPACE = "DINO_RCDE_R1_BAG_QUERY_PERMUTATION_V1_5"
BAG_REFERENCE_NAMESPACE = "DINO_RCDE_R1_BAG_REFERENCE_PERMUTATION_V1_5"
EXPECTED_CACHE_MODEL_SHA256 = (
    "f901d9bd056bb65e5fcb02c72e869f6d0cf8d7472467d22fbc340442feca034c"
)
MATRIX_HASH_KEYS = frozenset({"Q", "G", "P", "v"})
PROTECTED_PATH_PARTS = frozenset({"c8", "s8", "opened", "sealed"})
FORBIDDEN_OUTPUT_KEY_FRAGMENTS = (
    "target",
    "identity",
    "supergroup",
    "raw_score",
    "raw_rank",
    "rank_slot",
    "winner",
    "correctness",
)
def _logical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def _zero_access_audit(value: object) -> bool:
    return isinstance(value, Mapping) and value == {
        "restricted_path_reads": 0,
        "external_user_directory_writes": 0,
        "label_joins": 0,
    }


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise PrejoinAbort("tensor hash input is not a tensor")
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
        )
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise PrejoinAbort(f"missing JSON input: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PrejoinAbort(f"JSON input is not an object: {path}")
    return value


def exact_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Accept an actual JSON integer, never bool/string/float coercions."""

    if type(value) is not int:
        raise PrejoinAbort(f"{label} must be an exact JSON integer")
    result = value
    if minimum is not None and result < minimum:
        raise PrejoinAbort(f"{label} is below {minimum}")
    if maximum is not None and result > maximum:
        raise PrejoinAbort(f"{label} is above {maximum}")
    return result


def safe_torch_load(path: Path) -> Any:
    """Load tensor-only checkpoints/caches without permitting arbitrary pickle."""

    safe = ensure_unprotected_path(path)
    if not safe.is_file():
        raise PrejoinAbort(f"missing torch input: {safe}")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(safe, map_location="cpu", weights_only=True)


def assert_pinned_core_import() -> None:
    """Recheck provenance at execution time in case modules were mutated."""

    module = sys.modules.get(CORE_PINNED_MODULE)
    if (
        module is not _RCDE_CORE
        or Path(getattr(module, "__file__", "")).resolve() != CORE_PATH
        or DINO_RCDE_V1_2 is not getattr(module, "DINO_RCDE_V1_2", None)
    ):
        raise PrejoinAbort("pinned DINO-RCDE core runtime provenance drift")
    ambient = sys.modules.get(CORE_CANONICAL_MODULE)
    if ambient is not None and Path(getattr(ambient, "__file__", "")).resolve() != CORE_PATH:
        raise PrejoinAbort("ambient DINO-RCDE core shadow detected at runtime")


def ensure_unprotected_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise PrejoinAbort(f"symlinked path component is forbidden: {current}")
    resolved = absolute.resolve(strict=False)
    if {part.lower() for part in resolved.parts} & PROTECTED_PATH_PARTS:
        raise PrejoinAbort(f"protected path is not allowed: {resolved}")
    return absolute


def resolve_bound_path(path_text: object) -> Path:
    candidate = Path(str(path_text))
    if candidate.is_absolute():
        raise PrejoinAbort("formal bindings must be RC-root-relative")
    current = ROOT
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            raise PrejoinAbort(f"symlinked formal binding is forbidden: {current}")
    resolved = (ROOT / candidate).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise PrejoinAbort(f"binding escapes RC root: {resolved}")
    return ensure_unprotected_path(resolved)


def hash_parts(namespace: str, *parts: object) -> str:
    payload = "\0".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_query_ordinals(text: str) -> list[int]:
    """Parse a canonical comma list with optional inclusive ``a-b`` ranges."""

    if not isinstance(text, str) or not text.strip():
        raise PrejoinAbort("an explicit query-ordinal shard is required")
    values: list[int] = []
    for item in text.split(","):
        token = item.strip()
        if re.fullmatch(r"\d+", token):
            values.append(int(token))
        elif re.fullmatch(r"\d+-\d+", token):
            left_text, right_text = token.split("-", 1)
            left, right = int(left_text), int(right_text)
            if right < left:
                raise PrejoinAbort("descending query-ordinal range is forbidden")
            values.extend(range(left, right + 1))
        else:
            raise PrejoinAbort(f"invalid query ordinal token: {token!r}")
    if not values or len(values) != len(set(values)):
        raise PrejoinAbort("query ordinals must be nonempty and unique")
    if values != sorted(values):
        raise PrejoinAbort("query ordinals must be in strictly increasing order")
    return values


def _fold_number(value: object) -> int:
    text = str(value).strip().upper()
    if text.startswith("F"):
        text = text[1:]
    try:
        result = int(text)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise PrejoinAbort(f"invalid fold value: {value!r}") from exc
    if result not in (1, 2, 3, 4):
        raise PrejoinAbort(f"fold outside 1..4: {result}")
    return result


def _normalize_schedule_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_query_count: int,
    expected_candidate_count: int,
    expected_fold_counts: Mapping[int, int] | None,
) -> list[dict[str, Any]]:
    required = {
        "query_id",
        "historical_query_ordinal",
        "execution_ordinal",
        "heldout_fold",
        "source_image_sha256",
        "candidate_physical_rows",
        "candidate_axis_sha256",
    }
    if len(records) != expected_query_count:
        raise PrejoinAbort("redacted schedule query count drift")
    normalized: list[dict[str, Any]] = []
    for record_index, source in enumerate(records):
        if not isinstance(source, Mapping):
            raise PrejoinAbort("redacted schedule record is not an object")
        if set(source) != required:
            raise PrejoinAbort("redacted schedule record fields are not exact")
        raw_rows = source["candidate_physical_rows"]
        if not isinstance(raw_rows, list):
            raise PrejoinAbort("candidate physical rows must be a JSON list")
        source_rows = [
            exact_int(
                value,
                f"schedule[{record_index}].candidate_physical_rows[{row_index}]",
                minimum=0,
            )
            for row_index, value in enumerate(raw_rows)
        ]
        if (
            len(source_rows) != expected_candidate_count
            or len(source_rows) != len(set(source_rows))
        ):
            raise PrejoinAbort("candidate physical-row set is not exact")
        canonical_rows = sorted(source_rows)
        if source_rows != canonical_rows:
            raise PrejoinAbort("redacted candidate axis is not numeric canonical")
        if source["candidate_axis_sha256"] != canonical_sha256(canonical_rows):
            raise PrejoinAbort("redacted candidate-axis hash drift")
        normalized.append(
            {
                "query_id": str(source["query_id"]),
                "historical_query_ordinal": exact_int(
                    source["historical_query_ordinal"],
                    f"schedule[{record_index}].historical_query_ordinal",
                    minimum=0,
                ),
                "execution_ordinal": exact_int(
                    source["execution_ordinal"],
                    f"schedule[{record_index}].execution_ordinal",
                    minimum=0,
                ),
                "heldout_fold": exact_int(
                    source["heldout_fold"],
                    f"schedule[{record_index}].heldout_fold",
                    minimum=1,
                    maximum=4,
                ),
                "source_image_sha256": str(source["source_image_sha256"]),
                "candidate_physical_rows": canonical_rows,
                "candidate_axis_sha256": str(source["candidate_axis_sha256"]),
            }
        )
    if len({row["query_id"] for row in normalized}) != expected_query_count:
        raise PrejoinAbort("duplicate query ID in redacted schedule")
    if len({row["historical_query_ordinal"] for row in normalized}) != expected_query_count:
        raise PrejoinAbort("duplicate historical query ordinal")
    execution = sorted(row["execution_ordinal"] for row in normalized)
    if execution != list(range(expected_query_count)):
        raise PrejoinAbort("execution ordinals must be dense 0..N-1")
    if expected_fold_counts is not None:
        observed = {
            fold: sum(row["heldout_fold"] == fold for row in normalized)
            for fold in (1, 2, 3, 4)
        }
        if observed != dict(expected_fold_counts):
            raise PrejoinAbort(f"OOF fold-count drift: {observed}")
    return normalized


def load_redacted_schedule(
    *,
    redacted_schedule: Path | None,
    prejoin_folds: Path | None,
    model_visible_c128: Path | None,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    expected_fold_counts: Mapping[int, int] | None = EXPECTED_FOLD_COUNTS,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load a frozen redacted schedule or compile the equivalent E0 view.

    The two-source mode exists only to exercise E0 before the immutable formal
    schedule is frozen.  Its dense execution ordinal is the jointly verified
    record-sequence position, never the historical query ordinal.
    """

    if redacted_schedule is not None:
        if prejoin_folds is not None or model_visible_c128 is not None:
            raise PrejoinAbort("redacted and two-source schedule modes are exclusive")
        path = ensure_unprotected_path(redacted_schedule)
        payload = read_json(path)
        exact_top = {
            "schema_version",
            "status",
            "query_count",
            "fold_query_counts",
            "record_fields",
            "records",
            "record_sequence_sha256",
            "source_bindings",
            "access_audit",
            "logical_sha256",
        }
        if set(payload) != exact_top:
            raise PrejoinAbort("redacted schedule top-level field drift")
        query_count = exact_int(
            payload.get("query_count"), "redacted schedule query_count", minimum=0
        )
        raw_fold_counts = payload.get("fold_query_counts")
        if not isinstance(raw_fold_counts, Mapping) or set(raw_fold_counts) != {
            "1",
            "2",
            "3",
            "4",
        }:
            raise PrejoinAbort("redacted schedule fold-count schema drift")
        fold_counts = {
            str(fold): exact_int(
                raw_fold_counts[str(fold)],
                f"redacted schedule fold_query_counts[{fold}]",
                minimum=0,
            )
            for fold in (1, 2, 3, 4)
        }
        if (
            payload.get("schema_version") != REDACTED_SCHEDULE_SCHEMA
            or payload.get("status") != REDACTED_SCHEDULE_STATUS
            or query_count != expected_query_count
            or (
                expected_fold_counts is not None
                and fold_counts
                != {
                    str(key): value
                    for key, value in dict(expected_fold_counts).items()
                }
            )
            or payload.get("record_fields")
            != [
                "query_id",
                "historical_query_ordinal",
                "execution_ordinal",
                "heldout_fold",
                "source_image_sha256",
                "candidate_physical_rows",
                "candidate_axis_sha256",
            ]
            or not _zero_access_audit(payload.get("access_audit"))
            or not isinstance(payload.get("source_bindings"), Mapping)
            or payload.get("logical_sha256") != _logical_mapping_sha256(payload)
        ):
            raise PrejoinAbort("redacted schedule envelope drift")
        records = payload.get("records")
        if not isinstance(records, list):
            raise PrejoinAbort("redacted schedule records are absent")
        if payload.get("record_sequence_sha256") != canonical_sha256(records):
            raise PrejoinAbort("redacted schedule record-sequence drift")
        assert_output_keys_prejoin_only(payload)
        normalized = _normalize_schedule_records(
            records,
            expected_query_count=expected_query_count,
            expected_candidate_count=expected_candidate_count,
            expected_fold_counts=expected_fold_counts,
        )
        return normalized, {"redacted_schedule_sha256": file_sha256(path)}

    if prejoin_folds is None or model_visible_c128 is None:
        raise PrejoinAbort("both prejoin-fold and C128 sources are required")
    fold_path = ensure_unprotected_path(prejoin_folds)
    c128_path = ensure_unprotected_path(model_visible_c128)
    folds = read_json(fold_path)
    c128 = read_json(c128_path)
    fold_rows = folds.get("records")
    candidate_rows = c128.get("rows")
    if not isinstance(fold_rows, list) or not isinstance(candidate_rows, list):
        raise PrejoinAbort("two-source schedule rows are absent")
    if len(fold_rows) != expected_query_count or len(candidate_rows) != expected_query_count:
        raise PrejoinAbort("two-source schedule population drift")
    compiled: list[dict[str, Any]] = []
    for execution_ordinal, (fold_row, candidate_row) in enumerate(
        zip(fold_rows, candidate_rows)
    ):
        if (
            str(fold_row.get("query_id")) != str(candidate_row.get("query_id"))
            or int(fold_row.get("query_ordinal"))
            != int(candidate_row.get("query_ordinal"))
        ):
            raise PrejoinAbort("two-source schedule sequence alignment failed")
        compiled.append(
            {
                "query_id": str(fold_row["query_id"]),
                "historical_query_ordinal": int(fold_row["query_ordinal"]),
                "execution_ordinal": execution_ordinal,
                "heldout_fold": _fold_number(fold_row["inner_fold"]),
                "source_image_sha256": str(fold_row["source_image_sha256"]),
                "candidate_physical_rows": sorted(
                    map(int, candidate_row["candidate_physical_rows"])
                ),
                "candidate_axis_sha256": canonical_sha256(
                    sorted(map(int, candidate_row["candidate_physical_rows"]))
                ),
            }
        )
    normalized = _normalize_schedule_records(
        compiled,
        expected_query_count=expected_query_count,
        expected_candidate_count=expected_candidate_count,
        expected_fold_counts=expected_fold_counts,
    )
    return normalized, {
        "prejoin_folds_sha256": file_sha256(fold_path),
        "candidate_ledger_sha256": file_sha256(c128_path),
    }


def canonical_cache_path(cache_root: Path, kind: str, cache_key: int) -> Path:
    if kind == "query":
        name = f"query_{int(cache_key):04d}.pt"
    elif kind == "reference":
        name = f"reference_{int(cache_key):05d}.pt"
    else:
        raise PrejoinAbort(f"unknown cache kind: {kind}")
    return ensure_unprotected_path(Path(cache_root) / name)


def load_redacted_cache_index(
    path: Path,
    *,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    expected_reference_rows: set[int] | None = None,
    expected_schedule_logical_sha256: str | None = None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], str]:
    source = ensure_unprotected_path(path)
    payload = read_json(source)
    exact_top = {
        "schema_version",
        "status",
        "query_count",
        "reference_count",
        "record_count",
        "record_fields",
        "records",
        "record_sequence_sha256",
        "source_bindings",
        "access_audit",
        "logical_sha256",
    }
    if set(payload) != exact_top:
        raise PrejoinAbort("redacted cache-index top-level field drift")
    query_count = exact_int(
        payload.get("query_count"), "redacted cache-index query_count", minimum=0
    )
    reference_count = exact_int(
        payload.get("reference_count"),
        "redacted cache-index reference_count",
        minimum=0,
    )
    record_count = exact_int(
        payload.get("record_count"), "redacted cache-index record_count", minimum=0
    )
    if (
        payload.get("schema_version") != REDACTED_CACHE_INDEX_SCHEMA
        or payload.get("status") != REDACTED_CACHE_INDEX_STATUS
        or query_count != expected_query_count
        or payload.get("record_fields")
        != [
            "kind",
            "cache_key",
            "source_image_sha256",
            "tokens_fp16_payload_sha256",
            "cache_logical_sha256",
            "valid_patch_mask_sha256",
            "geometry_logical_sha256",
        ]
        or not _zero_access_audit(payload.get("access_audit"))
        or not isinstance(payload.get("source_bindings"), Mapping)
        or payload.get("logical_sha256") != _logical_mapping_sha256(payload)
    ):
        raise PrejoinAbort("redacted cache-index envelope drift")
    records = payload.get("records")
    if not isinstance(records, list):
        raise PrejoinAbort("redacted cache-index records are absent")
    if (
        record_count != len(records)
        or payload.get("record_sequence_sha256") != canonical_sha256(records)
    ):
        raise PrejoinAbort("redacted cache-index record-sequence drift")
    assert_output_keys_prejoin_only(payload)
    required = {
        "kind",
        "cache_key",
        "source_image_sha256",
        "tokens_fp16_payload_sha256",
        "cache_logical_sha256",
        "valid_patch_mask_sha256",
        "geometry_logical_sha256",
    }
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PrejoinAbort("redacted cache-index record is not an object")
        if set(record) != required or record.get("kind") not in {"query", "reference"}:
            raise PrejoinAbort("redacted cache-index record field drift")
        cache_key = exact_int(
            record["cache_key"],
            f"redacted cache-index records[{record_index}].cache_key",
            minimum=0,
        )
        key = (str(record["kind"]), cache_key)
        if key in result:
            raise PrejoinAbort("duplicate redacted cache-index key")
        for name in required - {"kind", "cache_key"}:
            if not _is_hex_digest(record[name]):
                raise PrejoinAbort(f"redacted cache-index digest drift: {name}")
        result[key] = dict(record)
    if sorted(key for kind, key in result if kind == "query") != list(
        range(expected_query_count)
    ):
        raise PrejoinAbort("redacted query cache keys are not dense execution ordinals")
    observed_references = {key for kind, key in result if kind == "reference"}
    if expected_reference_rows is not None and observed_references != set(
        expected_reference_rows
    ):
        raise PrejoinAbort("redacted reference cache keys differ from schedule union")
    if (
        reference_count != len(observed_references)
        or record_count != expected_query_count + len(observed_references)
    ):
        raise PrejoinAbort("redacted cache-index population drift")
    if (
        expected_schedule_logical_sha256 is not None
        and payload["source_bindings"].get("redacted_schedule_logical_sha256")
        != expected_schedule_logical_sha256
    ):
        raise PrejoinAbort("cache index is not bound to the redacted schedule")
    return result, file_sha256(source)


def validate_schedule_cache_index_global_join(
    schedule: Sequence[Mapping[str, Any]],
    cache_index: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    """Close every query schedule row to its dense cache key and source hash."""

    seen_execution: set[int] = set()
    for row_index, row in enumerate(schedule):
        execution = exact_int(
            row.get("execution_ordinal"),
            f"normalized schedule[{row_index}].execution_ordinal",
            minimum=0,
        )
        if execution in seen_execution:
            raise PrejoinAbort("duplicate execution ordinal in global cache join")
        seen_execution.add(execution)
        index_record = cache_index.get(("query", execution))
        if not isinstance(index_record, Mapping):
            raise PrejoinAbort(
                f"global cache join missing query execution ordinal: {execution}"
            )
        if index_record.get("source_image_sha256") != row.get(
            "source_image_sha256"
        ):
            raise PrejoinAbort(
                f"global cache source-image join drift: execution {execution}"
            )
    query_keys = {key for kind, key in cache_index if kind == "query"}
    if query_keys != seen_execution:
        raise PrejoinAbort("global cache query-key population differs from schedule")


def validate_formal_parameters(
    *,
    arm: str,
    fold: int,
    requested_query_ordinals: Sequence[int],
    protocol: Path | None,
    authority: Path | None,
    redacted_schedule: Path | None,
    redacted_cache_index: Path | None,
    prejoin_folds: Path | None,
    model_visible_c128: Path | None,
    query_tile_rows: int,
    reference_tile_rows: int,
    pair_batch_size: int,
) -> None:
    """Close formal CLI choices before any schedule/cache content is parsed."""

    if arm not in ALLOWED_ARMS or type(fold) is not int or fold not in (1, 2, 3, 4):
        raise PrejoinAbort("formal arm/fold parameter closure failed")
    if (
        not requested_query_ordinals
        or any(type(value) is not int or value < 0 for value in requested_query_ordinals)
        or list(requested_query_ordinals)
        != sorted(set(requested_query_ordinals))
    ):
        raise PrejoinAbort("formal query ordinals are not canonical exact integers")
    if (
        protocol is None
        or authority is None
        or redacted_schedule is None
        or redacted_cache_index is None
        or prejoin_folds is not None
        or model_visible_c128 is not None
    ):
        raise PrejoinAbort(
            "formal mode requires protocol, authority, redacted schedule, "
            "and redacted cache index only"
        )
    if (
        type(query_tile_rows) is not int
        or query_tile_rows != EXPECTED_QUERY_TILE_ROWS
        or type(reference_tile_rows) is not int
        or reference_tile_rows != EXPECTED_REFERENCE_TILE_ROWS
        or type(pair_batch_size) is not int
        or pair_batch_size != PAIR_BATCH_SIZE
    ):
        raise PrejoinAbort("formal streaming/batching parameters are not frozen")


def validate_formal_contract(
    *,
    protocol_path: Path,
    authority_path: Path,
    arm: str,
    fold: int,
    checkpoint_path: Path,
    schedule_path: Path,
    cache_index_path: Path,
    output_dir: Path,
    requested_query_ordinals: Sequence[int],
) -> tuple[dict[str, str], dict[str, Path]]:
    assert_pinned_core_import()
    protocol_path = ensure_unprotected_path(protocol_path)
    authority_path = ensure_unprotected_path(authority_path)
    protocol = read_json(protocol_path)
    authority = read_json(authority_path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise PrejoinAbort("formal prejoin protocol schema drift")
    if (
        protocol.get("stage") != FORMAL_STAGE
        or protocol.get("scientific_GO_or_NO_GO") is not None
        or protocol.get("automatic_stage_advance") is not False
        or protocol.get("heldout_label_join") is not False
        or protocol.get("natural_training_authorized") is not False
    ):
        raise PrejoinAbort("formal prejoin protocol boundary drift")
    authority_binding = protocol.get("authority")
    if (
        not isinstance(authority_binding, Mapping)
        or set(authority_binding) != {"path", "sha256", "required_status"}
    ):
        raise PrejoinAbort("formal authority binding absent")
    required_status = authority_binding.get("required_status")
    if not isinstance(required_status, str) or not required_status.strip():
        raise PrejoinAbort("formal authority required_status must be nonempty")
    if (
        not isinstance(authority.get("schema_version"), str)
        or re.fullmatch(
            r"rc_current_authority_v\d+_20260814",
            str(authority.get("schema_version")),
        )
        is None
        or
        resolve_bound_path(authority_binding.get("path", "")) != authority_path
        or authority_binding.get("sha256") != file_sha256(authority_path)
        or authority.get("status") != required_status
        or authority.get("next_authorized_stage") != FORMAL_STAGE
        or authority.get("scientific_GO_or_NO_GO") is not None
        or authority.get("automatic_stage_advance") is not False
    ):
        raise PrejoinAbort("formal prejoin authority closure failed")
    anchors = authority.get("execution_content_anchors")
    if not isinstance(anchors, Mapping):
        raise PrejoinAbort("authority execution-content anchors absent")
    leaf_protocols = anchors.get("leaf_protocols")
    reverse_materializer = anchors.get("materializer")
    if (
        not isinstance(leaf_protocols, Mapping)
        or set(leaf_protocols) != {"paths", "required_schema", "digest_rule"}
        or leaf_protocols.get("required_schema") != PROTOCOL_SCHEMA
        or leaf_protocols.get("digest_rule")
        != "each_leaf_protocol_embeds_exact_authority_sha256"
        or not isinstance(reverse_materializer, Mapping)
        or resolve_bound_path(reverse_materializer.get("path", ""))
        != Path(__file__).resolve()
        or reverse_materializer.get("sha256") != file_sha256(Path(__file__))
    ):
        raise PrejoinAbort("authority reverse execution-content anchor drift")
    raw_leaf_paths = leaf_protocols.get("paths")
    if (
        not isinstance(raw_leaf_paths, list)
        or len(raw_leaf_paths) != len(ALLOWED_ARMS) * 4
        or any(not isinstance(value, str) or not value for value in raw_leaf_paths)
        or raw_leaf_paths != sorted(set(raw_leaf_paths))
    ):
        raise PrejoinAbort("authority leaf-protocol path set is not canonical")
    resolved_leaf_paths = [resolve_bound_path(value) for value in raw_leaf_paths]
    if (
        len(resolved_leaf_paths) != len(set(resolved_leaf_paths))
        or protocol_path not in resolved_leaf_paths
    ):
        raise PrejoinAbort("current protocol is not an authorized leaf protocol")
    scope = protocol.get("scope")
    if not isinstance(scope, Mapping) or set(scope) != {"arm", "outer_fold"}:
        raise PrejoinAbort("formal prejoin scope field drift")
    protocol_fold = exact_int(
        scope.get("outer_fold"), "formal protocol scope outer_fold", minimum=1, maximum=4
    )
    if scope.get("arm") != arm or protocol_fold != fold:
        raise PrejoinAbort("formal prejoin scope mismatch")
    authorized_scope = authority.get("authorized_scope")
    if (
        not isinstance(authorized_scope, Mapping)
        or authorized_scope.get("stage") != FORMAL_STAGE
    ):
        raise PrejoinAbort("formal prejoin scope is not authorized")
    authorized_arms = authorized_scope.get("arms")
    if (
        not isinstance(authorized_arms, list)
        or not authorized_arms
        or len(authorized_arms) != len(set(map(str, authorized_arms)))
        or any(value not in ALLOWED_ARMS for value in authorized_arms)
        or arm not in authorized_arms
    ):
        raise PrejoinAbort("authority does not explicitly authorize current arm")
    raw_authorized_folds = authorized_scope.get("outer_folds")
    if not isinstance(raw_authorized_folds, list) or not raw_authorized_folds:
        raise PrejoinAbort("authority outer_folds authorization is absent")
    authorized_folds = [
        exact_int(value, "authority outer_folds member", minimum=1, maximum=4)
        for value in raw_authorized_folds
    ]
    if authorized_folds != sorted(set(authorized_folds)) or fold not in authorized_folds:
        raise PrejoinAbort("authority does not explicitly authorize current fold")
    raw_cohorts = authorized_scope.get("historical_query_ordinals_by_fold")
    if not isinstance(raw_cohorts, Mapping) or set(raw_cohorts) != {
        "1",
        "2",
        "3",
        "4",
    }:
        raise PrejoinAbort("authority cohort-by-fold authorization is absent")
    authorized_cohorts: dict[int, list[int]] = {}
    for cohort_fold in (1, 2, 3, 4):
        raw_ordinals = raw_cohorts[str(cohort_fold)]
        if not isinstance(raw_ordinals, list):
            raise PrejoinAbort("authority cohort must be a JSON list")
        cohort = [
            exact_int(
                value,
                f"authority fold{cohort_fold} historical query ordinal",
                minimum=0,
            )
            for value in raw_ordinals
        ]
        if cohort != sorted(set(cohort)):
            raise PrejoinAbort("authority cohort ordinals are not canonical")
        authorized_cohorts[cohort_fold] = cohort
    if list(requested_query_ordinals) != authorized_cohorts[fold]:
        raise PrejoinAbort("requested query cohort differs from authority")
    bindings = protocol.get("bindings")
    required_bindings = {
        "checkpoint": checkpoint_path,
        "redacted_schedule": schedule_path,
        "redacted_cache_index": cache_index_path,
        "core": ROOT / "src/rc_aslo_xf/dino_rcde_v1_2_resource_core.py",
        "runner": Path(__file__).resolve(),
    }
    dependency_names = {
        "redacted_inputs_validation",
        "matched_foldset_manifest",
        "matched_foldset_validation",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != (
        set(required_bindings) | dependency_names | {"output_directory"}
    ):
        raise PrejoinAbort("formal prejoin binding set drift")
    for name, expected_path in required_bindings.items():
        binding = bindings[name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or resolve_bound_path(binding.get("path", ""))
            != Path(expected_path).resolve()
            or binding.get("sha256") != file_sha256(Path(expected_path))
        ):
            raise PrejoinAbort(f"formal prejoin binding drift: {name}")
    dependency_paths: dict[str, Path] = {}
    for name in sorted(dependency_names):
        binding = bindings[name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise PrejoinAbort(f"formal dependency binding field drift: {name}")
        dependency_path = resolve_bound_path(binding.get("path", ""))
        if not dependency_path.is_file() or dependency_path.is_symlink():
            raise PrejoinAbort(f"formal dependency path is absent/symlinked: {name}")
        if binding.get("sha256") != file_sha256(dependency_path):
            raise PrejoinAbort(f"formal dependency binding hash drift: {name}")
        dependency_paths[name] = dependency_path
    output_binding = bindings["output_directory"]
    if (
        not isinstance(output_binding, Mapping)
        or set(output_binding) != {"path"}
        or resolve_bound_path(output_binding.get("path", ""))
        != ensure_unprotected_path(output_dir)
    ):
        raise PrejoinAbort("formal output-directory binding drift")

    # Only after authority, scope, and exact content hashes close may these
    # predecessor receipts be parsed.  Schedule/cache JSON remains unread.
    redacted_validation = read_json(
        dependency_paths["redacted_inputs_validation"]
    )
    if (
        redacted_validation.get("schema_version")
        != REDACTED_INPUTS_VALIDATION_SCHEMA
        or redacted_validation.get("status")
        != REDACTED_INPUTS_VALIDATION_STATUS
        or redacted_validation.get("schedule_file_sha256")
        != file_sha256(schedule_path)
        or redacted_validation.get("cache_index_file_sha256")
        != file_sha256(cache_index_path)
        or redacted_validation.get("automatic_stage_advance") is not False
        or redacted_validation.get("scientific_decision") is not None
    ):
        raise PrejoinAbort("redacted-input validation predecessor drift")
    matched_manifest = read_json(dependency_paths["matched_foldset_manifest"])
    if (
        matched_manifest.get("schema_version") != MATCHED_FOLDSET_MANIFEST_SCHEMA
        or matched_manifest.get("status") != MATCHED_FOLDSET_MANIFEST_STATUS
        or matched_manifest.get("arms") != list(ALLOWED_ARMS)
        or matched_manifest.get("outer_folds") != [1, 2, 3, 4]
        or matched_manifest.get("scientific_GO_or_NO_GO") is not None
        or matched_manifest.get("automatic_stage_advance") is not False
    ):
        raise PrejoinAbort("matched-foldset manifest predecessor drift")
    validate_matched_checkpoint_binding(
        matched_manifest,
        arm=arm,
        fold=fold,
        checkpoint_path=checkpoint_path,
    )
    matched_validation = read_json(
        dependency_paths["matched_foldset_validation"]
    )
    manifest_anchor = matched_validation.get("matched_foldset_manifest")
    if (
        matched_validation.get("schema_version")
        != MATCHED_FOLDSET_VALIDATION_SCHEMA
        or matched_validation.get("status")
        != MATCHED_FOLDSET_VALIDATION_STATUS
        or matched_validation.get("arms") != list(ALLOWED_ARMS)
        or matched_validation.get("outer_folds") != [1, 2, 3, 4]
        or not isinstance(manifest_anchor, Mapping)
        or set(manifest_anchor) != {"path", "sha256"}
        or resolve_bound_path(manifest_anchor.get("path", ""))
        != dependency_paths["matched_foldset_manifest"]
        or manifest_anchor.get("sha256")
        != file_sha256(dependency_paths["matched_foldset_manifest"])
        or matched_validation.get("scientific_GO_or_NO_GO") is not None
        or matched_validation.get("automatic_stage_advance") is not False
    ):
        raise PrejoinAbort("matched-foldset validation predecessor drift")
    hashes = {
        "protocol_sha256": file_sha256(protocol_path),
        "authority_sha256": file_sha256(authority_path),
    }
    hashes.update(
        {
            f"{name}_sha256": file_sha256(path)
            for name, path in dependency_paths.items()
        }
    )
    return hashes, dependency_paths


def validate_matched_checkpoint_binding(
    manifest: Mapping[str, Any],
    *,
    arm: str,
    fold: int,
    checkpoint_path: Path,
) -> None:
    """Bind one execution checkpoint to its exact matched arm/fold cell.

    The matched-foldset manifest is the authority that selected the eight
    comparable checkpoints.  Merely checking that a supplied checkpoint says
    the requested arm/fold internally is insufficient: an otherwise valid
    checkpoint from an older run could still be substituted.  This check
    therefore requires both the canonical path and the file digest selected by
    the manifest cell.
    """

    arm_key = {"RCDE_BAG": "bag", "RCDE_CONTEXT": "context"}.get(arm)
    if arm_key is None:
        raise PrejoinAbort("checkpoint/manifest arm-fold binding drift")
    requested_fold = exact_int(
        fold, "checkpoint/manifest requested fold", minimum=1, maximum=4
    )
    raw_folds = manifest.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != 4:
        raise PrejoinAbort("checkpoint/manifest fold table drift")
    indexed: dict[int, Mapping[str, Any]] = {}
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, Mapping):
            raise PrejoinAbort("checkpoint/manifest fold table drift")
        fold_number = exact_int(
            raw_fold.get("outer_fold"),
            "matched-foldset outer_fold",
            minimum=1,
            maximum=4,
        )
        if fold_number in indexed:
            raise PrejoinAbort("checkpoint/manifest fold table drift")
        indexed[fold_number] = raw_fold
    if set(indexed) != {1, 2, 3, 4}:
        raise PrejoinAbort("checkpoint/manifest fold table drift")
    arm_record = indexed[requested_fold].get(arm_key)
    binding = arm_record.get("checkpoint") if isinstance(arm_record, Mapping) else None
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise PrejoinAbort("checkpoint/manifest arm-fold binding drift")
    digest = binding.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PrejoinAbort("checkpoint/manifest arm-fold binding drift")
    expected_path = ensure_unprotected_path(checkpoint_path)
    if (
        resolve_bound_path(binding.get("path", "")) != expected_path
        or digest != file_sha256(expected_path)
    ):
        raise PrejoinAbort("checkpoint/manifest arm-fold binding drift")


def cache_logical_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "tokens": tensor_sha256(payload["tokens_fp16"]),
            "mask": tensor_sha256(payload["valid_patch_mask"]),
            "geometry": payload["geometry_receipt"],
            "model": payload["model_checkpoint_logical_sha256"],
            "source": payload["source_image_sha256"],
        }
    )


def validate_cache_payload(
    payload: Mapping[str, Any],
    *,
    expected_model_sha256: str,
    index_record: Mapping[str, Any] | None = None,
    expected_source_image_sha256: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    tokens = payload.get("tokens_fp16")
    mask = payload.get("valid_patch_mask")
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.dtype != torch.float16
        or tokens.ndim != 4
        or tuple(tokens.shape[:2]) != (1, 4)
        or tokens.shape[-1] != 768
        or not bool(torch.isfinite(tokens).all())
    ):
        raise PrejoinAbort("cache token payload drift")
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 3
        or mask.shape[0] != 1
        or tokens.shape[2] != mask[0].numel()
        or not bool(mask.any())
    ):
        raise PrejoinAbort("cache geometry mask drift")
    if payload.get("model_checkpoint_logical_sha256") != expected_model_sha256:
        raise PrejoinAbort("cache backbone identity drift")
    if payload.get("logical_sha256") != cache_logical_sha256(payload):
        raise PrejoinAbort("cache logical hash drift")
    if (
        expected_source_image_sha256 is not None
        and payload.get("source_image_sha256") != expected_source_image_sha256
    ):
        raise PrejoinAbort("cache source-image binding drift")
    if index_record is not None:
        cache_receipt = payload.get("cache_receipt")
        geometry = payload.get("geometry_receipt")
        if not isinstance(cache_receipt, Mapping) or not isinstance(geometry, Mapping):
            raise PrejoinAbort("cache nested receipts are absent")
        if (
            index_record.get("source_image_sha256")
            != payload.get("source_image_sha256")
            or index_record.get("tokens_fp16_payload_sha256")
            != cache_receipt.get("tokens_fp16_payload_sha256")
            or index_record.get("cache_logical_sha256")
            != payload.get("logical_sha256")
            or index_record.get("valid_patch_mask_sha256")
            != cache_receipt.get("valid_patch_mask_sha256")
            or index_record.get("geometry_logical_sha256")
            != geometry.get("logical_sha256")
        ):
            raise PrejoinAbort("redacted cache-index payload closure failed")
    return tokens[0], mask[0], tuple(int(value) for value in mask[0].shape)


def permutation_shift(namespace: str, fold: int, item: object, valid_count: int) -> int:
    if valid_count < 2:
        raise PrejoinAbort("BAG permutation requires at least two valid tokens")
    return 1 + int(hash_parts(namespace, fold, item), 16) % (valid_count - 1)


def core_inputs(
    payload: Mapping[str, Any],
    device: torch.device,
    *,
    arm: str,
    namespace: str,
    fold: int,
    item: object,
    expected_cache_model_sha256: str,
    index_record: Mapping[str, Any] | None = None,
    expected_source_image_sha256: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], int | None]:
    tokens, mask, grid = validate_cache_payload(
        payload,
        expected_model_sha256=expected_cache_model_sha256,
        index_record=index_record,
        expected_source_image_sha256=expected_source_image_sha256,
    )
    tokens = tokens.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device, dtype=torch.bool)
    shift: int | None = None
    if arm == "RCDE_BAG":
        valid = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        shift = permutation_shift(namespace, fold, item, int(valid.numel()))
        source = valid.roll(shifts=shift)
        permuted = tokens.clone()
        permuted[:, valid, :] = tokens[:, source, :]
        tokens = permuted
    return tokens, mask, grid, shift


def _is_hex_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def audited_candidate_record(evidence: Any) -> dict[str, Any]:
    if (
        getattr(evidence, "matrix_audit_enabled", None) is not True
        or getattr(evidence, "analytic_null", None) is not False
        or getattr(evidence, "assignment", None) is not None
    ):
        raise PrejoinAbort("candidate was not produced by the audited streaming path")
    raw = evidence.raw_summary.detach().cpu().contiguous()
    null = evidence.null_summary.detach().cpu().contiguous()
    relational = evidence.relational.detach().cpu().contiguous()
    u = evidence.u.detach().cpu().contiguous()
    v = evidence.v.detach().cpu().contiguous()
    if (
        raw.dtype != torch.float32
        or null.dtype != torch.float32
        or relational.dtype != torch.float32
        or u.dtype != torch.float32
        or v.dtype != torch.float32
        or raw.ndim != 2
        or raw.shape[1] != SUMMARY_DIM
        or tuple(null.shape) != tuple(raw.shape)
        or tuple(relational.shape) != tuple(raw.shape)
        or tuple(u.shape) != (raw.shape[0],)
        or v.ndim != 1
        or not all(
            bool(torch.isfinite(value).all())
            for value in (raw, null, relational, u, v)
        )
        or not torch.equal(relational, raw - null)
    ):
        raise PrejoinAbort("candidate evidence tensor closure failed")
    hashes = dict(evidence.matrix_sha256)
    if set(hashes) != MATRIX_HASH_KEYS or not all(
        _is_hex_digest(value) for value in hashes.values()
    ):
        raise PrejoinAbort("audited Q/G/P/v hash contract drift")
    slices = [dict(item) for item in evidence.matrix_audit_slices]
    if len(slices) != 3:
        raise PrejoinAbort("audited matrix slice cardinality drift")
    for item in slices:
        required = {"flat_index", "query_index", "reference_index", "Q", "G", "P", "v"}
        if set(item) != required or not all(
            math.isfinite(float(item[name])) for name in ("Q", "G", "P", "v")
        ):
            raise PrejoinAbort("audited matrix slice field drift")
    return {
        "raw_summary": raw,
        "null_summary": null,
        "relational": relational,
        "u": u,
        "v": v,
        "matrix_sha256": hashes,
        "matrix_audit_slices": slices,
        "tensor_sha256": {
            "raw_summary": tensor_sha256(raw),
            "null_summary": tensor_sha256(null),
            "relational": tensor_sha256(relational),
            "u": tensor_sha256(u),
            "v": tensor_sha256(v),
        },
        "streaming_receipt": {
            "logical_cost_volume_build_count": int(
                evidence.logical_cost_volume_build_count
            ),
            "consensus_tile_count_per_pass": int(evidence.consensus_tile_count_per_pass),
            "real_pass_tile_counts": list(map(int, evidence.real_pass_tile_counts)),
            "null_pass_tile_counts": list(map(int, evidence.null_pass_tile_counts)),
            "maximum_resident_cost_elements": int(
                evidence.maximum_resident_cost_elements
            ),
            "summary_pass_count": int(evidence.summary_pass_count),
            "full_consensus_logits_resident": bool(
                evidence.full_consensus_logits_resident
            ),
        },
    }


def build_delta_matrix(
    model: Any,
    relational: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    pair_batch_size: int = PAIR_BATCH_SIZE,
) -> tuple[torch.Tensor, int]:
    candidate_count = int(relational.shape[0])
    left, right = torch.triu_indices(candidate_count, candidate_count, offset=1)
    delta = torch.zeros((candidate_count, candidate_count), dtype=torch.float32)
    evaluated = 0
    with torch.no_grad():
        for start in range(0, left.numel(), pair_batch_size):
            stop = min(left.numel(), start + pair_batch_size)
            batch_left = left[start:stop].to(relational.device)
            batch_right = right[start:stop].to(relational.device)
            values = model.compare_relational_batch(
                relational, batch_left, batch_right, query_mask
            ).detach().cpu().to(torch.float32)
            if tuple(values.shape) != (stop - start,) or not bool(
                torch.isfinite(values).all()
            ):
                raise PrejoinAbort("pairwise comparator output drift")
            delta[left[start:stop], right[start:stop]] = values
            delta[right[start:stop], left[start:stop]] = -values
            evaluated += stop - start
    delta.diagonal().zero_()
    if evaluated != candidate_count * (candidate_count - 1) // 2:
        raise PrejoinAbort("unordered pair evaluation count drift")
    if not torch.equal(delta, -delta.T) or not torch.equal(
        torch.diagonal(delta), torch.zeros(candidate_count, dtype=torch.float32)
    ):
        raise PrejoinAbort("pairwise Delta is not strictly antisymmetric")
    return delta, evaluated


def _record_logical_sha256(record: Mapping[str, Any]) -> str:
    candidates = []
    for candidate in record["candidates"]:
        candidates.append(
            {
                "physical_row": candidate["physical_row"],
                "reference_grid": candidate["reference_grid"],
                "content_grid_shift": candidate["content_grid_shift"],
                "matrix_sha256": candidate["matrix_sha256"],
                "matrix_audit_slices": candidate["matrix_audit_slices"],
                "tensor_sha256": candidate["tensor_sha256"],
                "streaming_receipt": candidate["streaming_receipt"],
                "cache_file_sha256": candidate["cache_file_sha256"],
            }
        )
    return canonical_sha256(
        {
            "query_id": record["query_id"],
            "source_image_sha256": record["source_image_sha256"],
            "historical_query_ordinal": record["historical_query_ordinal"],
            "execution_ordinal": record["execution_ordinal"],
            "heldout_fold": record["heldout_fold"],
            "query_grid": record["query_grid"],
            "query_mask_sha256": tensor_sha256(record["query_valid_patch_mask"]),
            "query_content_grid_shift": record["query_content_grid_shift"],
            "query_cache_file_sha256": record["query_cache_file_sha256"],
            "candidate_physical_rows": record["candidate_physical_rows"],
            "candidate_axis_sha256": record["candidate_axis_sha256"],
            "candidates": candidates,
            "delta_sha256": tensor_sha256(record["delta"]),
            "unordered_pair_evaluation_count": record[
                "unordered_pair_evaluation_count"
            ],
        }
    )


def assert_output_keys_prejoin_only(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_OUTPUT_KEY_FRAGMENTS):
                raise PrejoinAbort(f"forbidden postjoin field at {path}.{key}")
            assert_output_keys_prejoin_only(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            assert_output_keys_prejoin_only(nested, f"{path}[{index}]")


def load_frozen_model(
    checkpoint_path: Path,
    *,
    arm: str,
    fold: int,
    device: torch.device,
    model_factory: Callable[[], Any] = DINO_RCDE_V1_2,
    expected_parameter_count: int = EXPECTED_PARAMETER_COUNT,
) -> tuple[Any, dict[str, Any], str]:
    assert_pinned_core_import()
    checkpoint = safe_torch_load(checkpoint_path)
    if (
        not isinstance(checkpoint, Mapping)
        or
        checkpoint.get("arm") != arm
        or int(checkpoint.get("outer_fold", -1)) != fold
        or int(checkpoint.get("update", -1)) != EXPECTED_UPDATE
        or int(checkpoint.get("seed", -1)) != EXPECTED_SEED
        or not isinstance(checkpoint.get("model_state_dict"), Mapping)
    ):
        raise PrejoinAbort("checkpoint arm/fold/update/seed contract drift")
    state = checkpoint["model_state_dict"]
    actual_state_sha256 = state_dict_sha256(state)
    if checkpoint.get("final_state_sha256") != actual_state_sha256:
        raise PrejoinAbort("checkpoint final-state hash drift")
    model = model_factory()
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != expected_parameter_count:
        raise PrejoinAbort("model parameter count drift")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, actual_state_sha256


def materialize_payload(
    *,
    arm: str,
    fold: int,
    requested_query_ordinals: Sequence[int],
    schedule: Sequence[Mapping[str, Any]],
    cache_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    source_hashes: Mapping[str, str],
    redacted_cache_index: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    test_mode: bool = False,
    model_factory: Callable[[], Any] = DINO_RCDE_V1_2,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    expected_parameter_count: int = EXPECTED_PARAMETER_COUNT,
    expected_cache_model_sha256: str = EXPECTED_CACHE_MODEL_SHA256,
    query_tile_rows: int = 8,
    reference_tile_rows: int = 8,
    pair_batch_size: int = PAIR_BATCH_SIZE,
) -> dict[str, Any]:
    if arm not in ALLOWED_ARMS or fold not in (1, 2, 3, 4):
        raise PrejoinAbort("arm/fold scope is invalid")
    if list(requested_query_ordinals) != sorted(set(requested_query_ordinals)):
        raise PrejoinAbort("requested query ordinals are not canonical")
    by_historical = {int(row["historical_query_ordinal"]): row for row in schedule}
    selected = []
    for ordinal in requested_query_ordinals:
        row = by_historical.get(int(ordinal))
        if row is None:
            raise PrejoinAbort(f"requested historical query ordinal absent: {ordinal}")
        if int(row["heldout_fold"]) != fold:
            raise PrejoinAbort(
                f"requested query {ordinal} is not held out by outer fold {fold}"
            )
        selected.append(row)
    if not selected:
        raise PrejoinAbort("empty OOF shard is forbidden")

    if not test_mode and redacted_cache_index is None:
        raise PrejoinAbort("formal replay requires a redacted cache index")
    cache_root = ensure_unprotected_path(cache_root)
    checkpoint_path = ensure_unprotected_path(checkpoint_path)
    model, checkpoint, final_state_sha256 = load_frozen_model(
        checkpoint_path,
        arm=arm,
        fold=fold,
        device=device,
        model_factory=model_factory,
        expected_parameter_count=expected_parameter_count,
    )
    records: list[dict[str, Any]] = []
    for schedule_row in selected:
        execution = int(schedule_row["execution_ordinal"])
        query_index_record = (
            None
            if redacted_cache_index is None
            else redacted_cache_index.get(("query", execution))
        )
        if not test_mode and query_index_record is None:
            raise PrejoinAbort(f"query absent from redacted cache index: {execution}")
        query_path = canonical_cache_path(cache_root, "query", execution)
        if not query_path.is_file():
            raise PrejoinAbort(f"query cache payload absent: execution {execution}")
        query_payload = safe_torch_load(query_path)
        if not isinstance(query_payload, Mapping):
            raise PrejoinAbort("query cache payload is not an object")
        q, qm, qg, query_shift = core_inputs(
            query_payload,
            device,
            arm=arm,
            namespace=BAG_QUERY_NAMESPACE,
            fold=fold,
            item=schedule_row["query_id"],
            expected_cache_model_sha256=expected_cache_model_sha256,
            index_record=query_index_record,
            expected_source_image_sha256=schedule_row["source_image_sha256"],
        )
        candidate_records: list[dict[str, Any]] = []
        relational_rows: list[torch.Tensor] = []
        for physical_row in schedule_row["candidate_physical_rows"]:
            reference_index_record = (
                None
                if redacted_cache_index is None
                else redacted_cache_index.get(("reference", int(physical_row)))
            )
            if not test_mode and reference_index_record is None:
                raise PrejoinAbort(
                    f"reference absent from redacted cache index: {physical_row}"
                )
            reference_path = canonical_cache_path(
                cache_root, "reference", int(physical_row)
            )
            if not reference_path.is_file():
                raise PrejoinAbort(f"reference cache payload absent: {physical_row}")
            reference_payload = safe_torch_load(reference_path)
            if not isinstance(reference_payload, Mapping):
                raise PrejoinAbort("reference cache payload is not an object")
            r, rm, rg, reference_shift = core_inputs(
                reference_payload,
                device,
                arm=arm,
                namespace=BAG_REFERENCE_NAMESPACE,
                fold=fold,
                item=int(physical_row),
                expected_cache_model_sha256=expected_cache_model_sha256,
                index_record=reference_index_record,
                expected_source_image_sha256=(
                    None
                    if reference_index_record is None
                    else str(reference_index_record["source_image_sha256"])
                ),
            )
            with torch.no_grad():
                evidence = model.decode_candidate_true_streaming(
                    q,
                    r,
                    qm,
                    rm,
                    qg,
                    rg,
                    query_tile_rows=query_tile_rows,
                    reference_tile_rows=reference_tile_rows,
                    materialize_assignment_for_audit=False,
                )
            candidate = audited_candidate_record(evidence)
            candidate.update(
                {
                    "physical_row": int(physical_row),
                    "reference_grid": list(rg),
                    "reference_valid_patch_mask": rm.detach().cpu().contiguous(),
                    "content_grid_shift": reference_shift,
                    "cache_file_sha256": file_sha256(reference_path),
                }
            )
            candidate_records.append(candidate)
            relational_rows.append(candidate["relational"])
            del r, rm, evidence
        if len(candidate_records) != expected_candidate_count:
            raise PrejoinAbort("materialized candidate count drift")
        relational = torch.stack(relational_rows, dim=0).to(device)
        delta, pair_count = build_delta_matrix(
            model,
            relational,
            qm,
            pair_batch_size=pair_batch_size,
        )
        record: dict[str, Any] = {
            "query_id": str(schedule_row["query_id"]),
            "source_image_sha256": str(schedule_row["source_image_sha256"]),
            "historical_query_ordinal": int(
                schedule_row["historical_query_ordinal"]
            ),
            "execution_ordinal": execution,
            "heldout_fold": fold,
            "query_grid": list(qg),
            "query_valid_patch_mask": qm.detach().cpu().contiguous(),
            "query_content_grid_shift": query_shift,
            "query_cache_file_sha256": file_sha256(query_path),
            "candidate_physical_rows": list(
                schedule_row["candidate_physical_rows"]
            ),
            "candidate_axis_sha256": str(schedule_row["candidate_axis_sha256"]),
            "candidates": candidate_records,
            "delta": delta,
            "delta_sha256": tensor_sha256(delta),
            "unordered_pair_evaluation_count": pair_count,
        }
        record["logical_sha256"] = _record_logical_sha256(record)
        records.append(record)
        del q, qm, relational

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "arm": arm,
        "outer_fold": fold,
        "historical_query_ordinals": list(map(int, requested_query_ordinals)),
        "query_count": len(records),
        "candidate_count_per_query": expected_candidate_count,
        "pair_count_per_query": expected_candidate_count
        * (expected_candidate_count - 1)
        // 2,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_final_state_sha256": final_state_sha256,
        "checkpoint_initial_state_sha256": checkpoint.get("initial_state_sha256"),
        "source_hashes": dict(source_hashes),
        "cache_backbone_sha256": expected_cache_model_sha256,
        "claim_level": "ENGINEERING_PREJOIN_ONLY",
        "execution_mode": "TEST_FIXTURE" if test_mode else "FORMAL_BOUND",
        "records": records,
        "record_logical_sha256s": [row["logical_sha256"] for row in records],
        "record_sequence_sha256": canonical_sha256(
            [row["logical_sha256"] for row in records]
        ),
        "label_read_count": 0,
        "protected_access_counts": {
            "C8_runtime_read_count": 0,
            "S8_runtime_read_count": 0,
            "opened_runtime_read_count": 0,
            "sealed_runtime_read_count": 0,
            "home_files_modified": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "natural_training_authorized": False,
        "next_authorized_stage": NEXT_VALIDATION_STAGE,
    }
    assert_output_keys_prejoin_only(payload)
    return payload


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(Path(path), flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


MATERIALIZATION_COMMIT_FILENAME = "MATERIALIZATION_COMMITTED.json"
MATERIALIZATION_COMMIT_SCHEMA = (
    "rc_dino_rcde_r1_oof_materialization_commit_v1_20260814"
)
MATERIALIZATION_COMMIT_STATUS = "DINO_RCDE_R1_OOF_MATERIALIZATION_COMMITTED"
_MATERIALIZATION_CONTENT_FILENAMES = ("shard.pt", "receipt.json")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_NOREPLACE_UNSUPPORTED_ERRNOS = frozenset(
    {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}
)


def _renameat2_noreplace(staging: Path, output: Path) -> None:
    """Use the kernel no-clobber primitive and preserve its exact errno."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(staging),
        _AT_FDCWD,
        os.fsencode(output),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def _materialization_commit_logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PrejoinAbort(f"{label} is absent or not a regular file")


def _replay_scope_content_hashes(
    scope: Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
) -> None:
    artifact_path = scope / "shard.pt"
    receipt_path = scope / "receipt.json"
    _require_regular_file(artifact_path, "published shard")
    _require_regular_file(receipt_path, "published receipt")
    observed_receipt_sha256 = file_sha256(receipt_path)
    observed_artifact_sha256 = file_sha256(artifact_path)
    if observed_receipt_sha256 != expected_receipt_sha256:
        raise PrejoinAbort("published receipt hash replay mismatch")
    if observed_artifact_sha256 != expected_artifact_sha256:
        raise PrejoinAbort("published shard hash replay mismatch")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PrejoinAbort("published receipt replay is unreadable") from error
    artifact = receipt.get("artifact") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("path") != "shard.pt"
        or artifact.get("sha256") != observed_artifact_sha256
    ):
        raise PrejoinAbort("published receipt-to-shard binding mismatch")


def _write_materialization_commit_manifest(
    staging: Path,
    *,
    artifact_sha256: str,
    receipt_sha256: str,
    arm: str,
    outer_fold: int,
    historical_query_ordinals: Sequence[int],
    publication_mode: str,
) -> tuple[Path, str]:
    marker_path = staging / MATERIALIZATION_COMMIT_FILENAME
    publisher_path = Path(__file__).resolve()
    marker: dict[str, Any] = {
        "schema_version": MATERIALIZATION_COMMIT_SCHEMA,
        "status": MATERIALIZATION_COMMIT_STATUS,
        "commit_point": MATERIALIZATION_COMMIT_FILENAME,
        "arm": arm,
        "outer_fold": outer_fold,
        "historical_query_ordinals": list(historical_query_ordinals),
        "files": {
            "receipt.json": {"sha256": receipt_sha256},
            "shard.pt": {"sha256": artifact_sha256},
        },
        "publisher": {
            "path": str(publisher_path),
            "sha256": file_sha256(publisher_path),
        },
        "publication_mode": publication_mode,
        "auditable_staging_retained": publication_mode
        == "MKDIR_HARDLINK_MATERIALIZATION_COMMIT",
        "publication_semantics": (
            "MATERIALIZATION_ONLY_NOT_FINAL_SCOPE_COMMIT"
        ),
        "claim_level": "ENGINEERING_PREJOIN_ONLY",
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
    }
    marker["logical_sha256"] = _materialization_commit_logical_sha256(marker)
    assert_output_keys_prejoin_only(marker)
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with marker_path.open("rb") as handle:
        os.fsync(handle.fileno())
    return marker_path, file_sha256(marker_path)


def _rewrite_materialization_commit_for_fallback(staging: Path) -> str:
    """Atomically switch the still-private marker to the GPFS fallback mode."""

    marker_path = staging / MATERIALIZATION_COMMIT_FILENAME
    _require_regular_file(marker_path, "private materialization commit marker")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PrejoinAbort("private materialization marker is unreadable") from error
    if not isinstance(marker, Mapping):
        raise PrejoinAbort("private materialization marker is not an object")
    replacement = dict(marker)
    replacement["publication_mode"] = "MKDIR_HARDLINK_MATERIALIZATION_COMMIT"
    replacement["auditable_staging_retained"] = True
    replacement["logical_sha256"] = _materialization_commit_logical_sha256(
        replacement
    )
    temporary = staging / f".{MATERIALIZATION_COMMIT_FILENAME}.tmp.{os.getpid()}"
    if os.path.lexists(temporary):
        raise PrejoinAbort("private materialization marker temporary exists")
    temporary.write_text(
        json.dumps(replacement, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, marker_path)
    fsync_directory(staging)
    return file_sha256(marker_path)


def validate_materialization_commit(
    scope_dir: Path,
    *,
    expected_marker_sha256: str | None = None,
    expected_publication_mode: str | None = None,
) -> dict[str, Any]:
    """Validate the materialization checkpoint and replay content hashes."""

    scope = Path(scope_dir)
    if scope.is_symlink() or not scope.is_dir():
        raise PrejoinAbort("published scope is absent or unsafe")
    marker_path = scope / MATERIALIZATION_COMMIT_FILENAME
    _require_regular_file(marker_path, "materialization commit marker")
    observed_marker_sha256 = file_sha256(marker_path)
    if (
        expected_marker_sha256 is not None
        and observed_marker_sha256 != expected_marker_sha256
    ):
        raise PrejoinAbort("materialization commit marker file hash mismatch")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PrejoinAbort("materialization commit marker is unreadable") from error
    if not isinstance(marker, Mapping):
        raise PrejoinAbort("materialization commit marker is not an object")
    if (
        marker.get("schema_version") != MATERIALIZATION_COMMIT_SCHEMA
        or marker.get("status") != MATERIALIZATION_COMMIT_STATUS
        or marker.get("commit_point") != MATERIALIZATION_COMMIT_FILENAME
        or marker.get("publication_semantics")
        != "MATERIALIZATION_ONLY_NOT_FINAL_SCOPE_COMMIT"
        or marker.get("claim_level") != "ENGINEERING_PREJOIN_ONLY"
        or marker.get("scientific_GO_or_NO_GO") is not None
        or marker.get("automatic_stage_advance") is not False
        or marker.get("logical_sha256")
        != _materialization_commit_logical_sha256(marker)
    ):
        raise PrejoinAbort(
            "materialization commit marker schema or logical hash drift"
        )
    publication_mode = marker.get("publication_mode")
    if publication_mode not in {
        "RENAMEAT2_NOREPLACE",
        "MKDIR_HARDLINK_MATERIALIZATION_COMMIT",
    } or (
        expected_publication_mode is not None
        and publication_mode != expected_publication_mode
    ):
        raise PrejoinAbort("materialization publication mode drift")
    if marker.get("auditable_staging_retained") is not (
        publication_mode == "MKDIR_HARDLINK_MATERIALIZATION_COMMIT"
    ):
        raise PrejoinAbort("materialization staging-retention receipt drift")
    if (
        marker.get("arm") not in ALLOWED_ARMS
        or type(marker.get("outer_fold")) is not int
        or marker["outer_fold"] not in EXPECTED_FOLD_COUNTS
        or not isinstance(marker.get("historical_query_ordinals"), list)
        or not marker["historical_query_ordinals"]
        or any(type(value) is not int for value in marker["historical_query_ordinals"])
    ):
        raise PrejoinAbort("materialization scope identity drift")
    publisher = marker.get("publisher")
    current_publisher = Path(__file__).resolve()
    if (
        not isinstance(publisher, Mapping)
        or set(publisher) != {"path", "sha256"}
        or publisher.get("path") != str(current_publisher)
        or publisher.get("sha256") != file_sha256(current_publisher)
    ):
        raise PrejoinAbort("materialization publisher provenance drift")
    files = marker.get("files")
    if (
        not isinstance(files, Mapping)
        or set(files) != set(_MATERIALIZATION_CONTENT_FILENAMES)
    ):
        raise PrejoinAbort("materialization commit marker file set drift")
    for filename in _MATERIALIZATION_CONTENT_FILENAMES:
        entry = files.get(filename)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"sha256"}
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise PrejoinAbort("materialization commit marker hash entry drift")
    _replay_scope_content_hashes(
        scope,
        expected_artifact_sha256=files["shard.pt"]["sha256"],
        expected_receipt_sha256=files["receipt.json"]["sha256"],
    )
    return dict(marker)


def _mkdir_hardlink_commit_publish(
    staging: Path,
    output: Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
    expected_marker_sha256: str,
) -> None:
    """GPFS-safe publication with an atomic materialization marker."""

    if staging.parent != output.parent:
        raise PrejoinAbort("publication fallback requires one common parent")
    parent = output.parent
    if not staging.is_dir() or staging.is_symlink():
        raise PrejoinAbort("publication staging directory is absent or unsafe")
    if os.stat(staging, follow_symlinks=False).st_dev != os.stat(parent).st_dev:
        raise PrejoinAbort("publication fallback is not same-filesystem")
    try:
        # mkdir is the no-clobber ownership primitive for the final scope name.
        os.mkdir(output, mode=0o700)
    except FileExistsError as error:
        raise PrejoinAbort(f"immutable shard output race: {output}") from error
    fsync_directory(parent)

    # A crash before the final os.link leaves a directory with no materialized
    # checkpoint.  It can never be mistaken for a final committed scope.
    for filename in _MATERIALIZATION_CONTENT_FILENAMES:
        os.link(staging / filename, output / filename, follow_symlinks=False)
    fsync_directory(output)
    _replay_scope_content_hashes(
        output,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    os.link(
        staging / MATERIALIZATION_COMMIT_FILENAME,
        output / MATERIALIZATION_COMMIT_FILENAME,
        follow_symlinks=False,
    )
    fsync_directory(output)
    fsync_directory(parent)
    validate_materialization_commit(
        output,
        expected_marker_sha256=expected_marker_sha256,
        expected_publication_mode="MKDIR_HARDLINK_MATERIALIZATION_COMMIT",
    )
    # The hidden staging directory is intentionally retained.  It consists of
    # hard links, adds no payload copy, and remains a complete audit record if
    # the caller or a later stage fails after materialization commit.


def _publish_staging_directory(
    staging: Path,
    output: Path,
    *,
    expected_artifact_sha256: str,
    expected_receipt_sha256: str,
    expected_marker_sha256: str,
) -> str:
    """Publish once; use marker transactions only when renameat2 is unsupported."""

    try:
        _renameat2_noreplace(staging, output)
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise PrejoinAbort(f"immutable shard output race: {output}") from error
        if error.errno not in _RENAME_NOREPLACE_UNSUPPORTED_ERRNOS:
            raise PrejoinAbort(
                f"atomic no-replace publication failed: errno={error.errno}"
            ) from error
        if not staging.is_dir() or staging.is_symlink():
            raise PrejoinAbort(
                "unsupported renameat2 returned after staging disappeared"
            ) from error
        expected_marker_sha256 = _rewrite_materialization_commit_for_fallback(
            staging
        )
        _mkdir_hardlink_commit_publish(
            staging,
            output,
            expected_artifact_sha256=expected_artifact_sha256,
            expected_receipt_sha256=expected_receipt_sha256,
            expected_marker_sha256=expected_marker_sha256,
        )
        return "MKDIR_HARDLINK_MATERIALIZATION_COMMIT"

    try:
        fsync_directory(output.parent)
        validate_materialization_commit(
            output,
            expected_marker_sha256=expected_marker_sha256,
            expected_publication_mode="RENAMEAT2_NOREPLACE",
        )
    except BaseException as publication_error:
        try:
            _renameat2_noreplace(output, staging)
            fsync_directory(output.parent)
        except BaseException as rollback_error:
            raise PrejoinAbort(
                "fast publication failed and no-replace rollback failed"
            ) from rollback_error
        raise publication_error
    return "RENAMEAT2_NOREPLACE"


def atomic_commit_shard(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    input_paths: Mapping[str, Path],
) -> tuple[Path, Path]:
    output = ensure_unprotected_path(output_dir)
    if os.path.lexists(output):
        raise PrejoinAbort(f"immutable shard output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.incomplete"
    if os.path.lexists(staging):
        raise PrejoinAbort(f"incomplete prior attempt blocks resume: {staging}")
    staging.mkdir(mode=0o700)
    artifact_path = staging / "shard.pt"
    receipt_path = staging / "receipt.json"
    torch.save(dict(payload), artifact_path)
    with artifact_path.open("rb") as handle:
        os.fsync(handle.fileno())
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": STATUS,
        "arm": payload["arm"],
        "outer_fold": payload["outer_fold"],
        "historical_query_ordinals": payload["historical_query_ordinals"],
        "source_files": {
            name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for name, path in sorted(input_paths.items())
        },
        "artifact": {
            "path": "shard.pt",
            "sha256": file_sha256(artifact_path),
            "record_sequence_sha256": payload["record_sequence_sha256"],
        },
        "label_read_count": 0,
        "claim_level": payload["claim_level"],
        "execution_mode": payload["execution_mode"],
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "natural_training_authorized": False,
        "next_authorized_stage": NEXT_VALIDATION_STAGE,
    }
    assert_output_keys_prejoin_only(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with receipt_path.open("rb") as handle:
        os.fsync(handle.fileno())
    expected_artifact_sha256 = receipt["artifact"]["sha256"]
    expected_receipt_sha256 = file_sha256(receipt_path)
    _marker_path, expected_marker_sha256 = _write_materialization_commit_manifest(
        staging,
        artifact_sha256=expected_artifact_sha256,
        receipt_sha256=expected_receipt_sha256,
        arm=str(payload["arm"]),
        outer_fold=int(payload["outer_fold"]),
        historical_query_ordinals=payload["historical_query_ordinals"],
        publication_mode="RENAMEAT2_NOREPLACE",
    )
    # Every file and directory entry is durable before either publication path.
    fsync_directory(staging)
    _publish_staging_directory(
        staging,
        output,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_marker_sha256=expected_marker_sha256,
    )
    return output / "shard.pt", output / "receipt.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--query-ordinals", required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--redacted-schedule", type=Path)
    parser.add_argument("--redacted-cache-index", type=Path)
    parser.add_argument("--prejoin-folds", type=Path)
    parser.add_argument("--model-visible-c128", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--query-tile-rows", type=int, default=EXPECTED_QUERY_TILE_ROWS
    )
    parser.add_argument(
        "--reference-tile-rows", type=int, default=EXPECTED_REFERENCE_TILE_ROWS
    )
    parser.add_argument("--pair-batch-size", type=int, default=PAIR_BATCH_SIZE)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    ordinals = parse_query_ordinals(args.query_ordinals)
    input_paths: dict[str, Path] = {"checkpoint": args.checkpoint}
    redacted_cache_records: dict[tuple[str, int], dict[str, Any]] | None = None
    if args.test_mode:
        if args.protocol is not None or args.authority is not None:
            raise PrejoinAbort("test mode cannot claim a formal authority binding")
        schedule, source_hashes = load_redacted_schedule(
            redacted_schedule=args.redacted_schedule,
            prejoin_folds=args.prejoin_folds,
            model_visible_c128=args.model_visible_c128,
        )
        schedule_reference_rows = {
            int(row)
            for schedule_row in schedule
            for row in schedule_row["candidate_physical_rows"]
        }
        schedule_logical_sha256 = (
            None
            if args.redacted_schedule is None
            else read_json(args.redacted_schedule).get("logical_sha256")
        )
        if args.redacted_schedule is not None:
            input_paths["redacted_schedule"] = args.redacted_schedule
        else:
            if args.prejoin_folds is None or args.model_visible_c128 is None:
                raise PrejoinAbort("test two-source schedule inputs are incomplete")
            input_paths["prejoin_folds"] = args.prejoin_folds
            input_paths["candidate_ledger"] = args.model_visible_c128
        if args.redacted_cache_index is not None:
            redacted_cache_records, cache_index_sha256 = load_redacted_cache_index(
                args.redacted_cache_index,
                expected_query_count=len(schedule),
                expected_reference_rows=schedule_reference_rows,
                expected_schedule_logical_sha256=schedule_logical_sha256,
            )
            validate_schedule_cache_index_global_join(
                schedule, redacted_cache_records
            )
            source_hashes["redacted_cache_index_sha256"] = cache_index_sha256
            input_paths["redacted_cache_index"] = args.redacted_cache_index
    else:
        validate_formal_parameters(
            arm=args.arm,
            fold=args.fold,
            requested_query_ordinals=ordinals,
            protocol=args.protocol,
            authority=args.authority,
            redacted_schedule=args.redacted_schedule,
            redacted_cache_index=args.redacted_cache_index,
            prejoin_folds=args.prejoin_folds,
            model_visible_c128=args.model_visible_c128,
            query_tile_rows=args.query_tile_rows,
            reference_tile_rows=args.reference_tile_rows,
            pair_batch_size=args.pair_batch_size,
        )
        if (
            args.protocol is None
            or args.authority is None
            or args.redacted_schedule is None
            or args.redacted_cache_index is None
        ):  # pragma: no cover - narrowed by validate_formal_parameters
            raise PrejoinAbort("formal parameter narrowing failed")
        # The authority/protocol and exact path hashes close before either
        # redacted JSON payload is parsed.
        contract_hashes, dependency_paths = validate_formal_contract(
            protocol_path=args.protocol,
            authority_path=args.authority,
            arm=args.arm,
            fold=args.fold,
            checkpoint_path=args.checkpoint,
            schedule_path=args.redacted_schedule,
            cache_index_path=args.redacted_cache_index,
            output_dir=args.output_dir,
            requested_query_ordinals=ordinals,
        )
        schedule, source_hashes = load_redacted_schedule(
            redacted_schedule=args.redacted_schedule,
            prejoin_folds=None,
            model_visible_c128=None,
        )
        schedule_reference_rows = {
            int(row)
            for schedule_row in schedule
            for row in schedule_row["candidate_physical_rows"]
        }
        schedule_logical_sha256 = read_json(args.redacted_schedule).get(
            "logical_sha256"
        )
        redacted_cache_records, cache_index_sha256 = load_redacted_cache_index(
            args.redacted_cache_index,
            expected_reference_rows=schedule_reference_rows,
            expected_schedule_logical_sha256=schedule_logical_sha256,
        )
        validate_schedule_cache_index_global_join(schedule, redacted_cache_records)
        source_hashes.update(contract_hashes)
        source_hashes["redacted_cache_index_sha256"] = cache_index_sha256
        input_paths.update(
            {
                "protocol": args.protocol,
                "authority": args.authority,
                "redacted_cache_index": args.redacted_cache_index,
                "redacted_schedule": args.redacted_schedule,
                **dependency_paths,
            }
        )
    payload = materialize_payload(
        arm=args.arm,
        fold=args.fold,
        requested_query_ordinals=ordinals,
        schedule=schedule,
        cache_root=args.cache_root,
        checkpoint_path=args.checkpoint,
        device=torch.device(args.device),
        source_hashes=source_hashes,
        redacted_cache_index=redacted_cache_records,
        test_mode=args.test_mode,
        query_tile_rows=args.query_tile_rows,
        reference_tile_rows=args.reference_tile_rows,
        pair_batch_size=args.pair_batch_size,
    )
    artifact, receipt = atomic_commit_shard(
        args.output_dir, payload, input_paths=input_paths
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "artifact": str(artifact),
                "receipt": str(receipt),
                "query_count": payload["query_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
