#!/usr/bin/env python3
"""Independent validator for one R1 OOF prejoin shard.

This module intentionally does not import the materializer.  All hashes,
schedule joins, cache checks, tensor invariants, pair closure, and candidate
reorder checks are recomputed independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import torch

import rc_aslo_xf.dino_rcde_v1_2_resource_core as _rcde_core


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = (ROOT / "src/rc_aslo_xf/dino_rcde_v1_2_resource_core.py").resolve()
if Path(getattr(_rcde_core, "__file__", "")).resolve() != CORE_PATH:
    raise RuntimeError("DINO-RCDE core import provenance mismatch")
DINO_RCDE_V1_2 = _rcde_core.DINO_RCDE_V1_2
EXPECTED_PARAMETER_COUNT = _rcde_core.EXPECTED_PARAMETER_COUNT


SHARD_SCHEMA = "rc_dino_rcde_r1_oof_prejoin_shard_v1_20260814"
RECEIPT_SCHEMA = "rc_dino_rcde_r1_oof_prejoin_receipt_v1_20260814"
VALIDATION_SCHEMA = "rc_dino_rcde_r1_oof_prejoin_validation_v1_20260814"
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
MATCHED_FOLDSET_MANIFEST_SCHEMA = "rc_dino_rcde_r1_matched_foldset_manifest_v1_0"
MATCHED_FOLDSET_MANIFEST_STATUS = (
    "DINO_RCDE_R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_FROZEN"
)
MATCHED_FOLDSET_VALIDATION_SCHEMA = "rc_dino_rcde_r1_matched_foldset_validation_v1_0"
MATCHED_FOLDSET_VALIDATION_STATUS = (
    "DINO_RCDE_R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_VALIDATION_PASS"
)
SHARD_STATUS = "DINO_RCDE_R1_OOF_PREJOIN_SHARD_COMPLETE"
PASS_STATUS = "DINO_RCDE_R1_OOF_PREJOIN_VALIDATION_PASS"
EXPECTED_QUERY_COUNT = 600
EXPECTED_CANDIDATE_COUNT = 128
EXPECTED_FOLD_COUNTS = {1: 151, 2: 150, 3: 149, 4: 150}
EXPECTED_UPDATE = 2_048
EXPECTED_SEED = 17
SUMMARY_DIM = 16
MATRIX_HASH_KEYS = frozenset({"Q", "G", "P", "v"})
EXPECTED_CACHE_MODEL_SHA256 = (
    "f901d9bd056bb65e5fcb02c72e869f6d0cf8d7472467d22fbc340442feca034c"
)
FORBIDDEN_KEY_FRAGMENTS = (
    "target",
    "identity",
    "supergroup",
    "raw_score",
    "raw_rank",
    "rank_slot",
    "winner",
    "correctness",
)
EXPECTED_CLAIM_LEVEL = "ENGINEERING_PREJOIN_ONLY"
EXPECTED_NEXT_STAGE = "R1_OOF_PREJOIN_INDEPENDENT_VALIDATION_ONLY"
PROTECTED_PATH_PARTS = frozenset({"c8", "s8", "opened", "sealed"})


class ValidationAbort(RuntimeError):
    pass


def exact_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValidationAbort(f"{label} must be an exact JSON integer")
    result = value
    if minimum is not None and result < minimum:
        raise ValidationAbort(f"{label} is below {minimum}")
    return result


def exact_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValidationAbort(f"{label} is not an exact SHA-256 digest")
    return value


def ensure_unprotected_path(path: Path, *, must_exist: bool = True) -> Path:
    resolved = Path(path).resolve()
    if {part.lower() for part in resolved.parts} & PROTECTED_PATH_PARTS:
        raise ValidationAbort(f"protected path is forbidden: {resolved}")
    if must_exist and not resolved.exists():
        raise ValidationAbort(f"required path is absent: {resolved}")
    return resolved


def safe_torch_load(path: Path) -> Any:
    safe = ensure_unprotected_path(path)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(safe, map_location="cpu", weights_only=True)


def exact_fold_counts(value: object, expected: Mapping[int, int]) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {str(key) for key in expected}
    if set(value) != expected_keys:
        return False
    return all(
        type(value[str(key)]) is int and value[str(key)] == count
        for key, count in expected.items()
    )


def exact_int_list(value: object, expected: Sequence[int]) -> bool:
    return (
        isinstance(value, list)
        and all(type(item) is int for item in value)
        and value == list(expected)
    )


def logical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def zero_access_audit(value: object) -> bool:
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
        raise ValidationAbort("hash input is not a tensor")
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


def parse_ordinals(text: str) -> list[int]:
    values: list[int] = []
    for piece in text.split(","):
        token = piece.strip()
        if re.fullmatch(r"\d+", token):
            values.append(int(token))
        elif re.fullmatch(r"\d+-\d+", token):
            left, right = map(int, token.split("-", 1))
            if right < left:
                raise ValidationAbort("descending ordinal range")
            values.extend(range(left, right + 1))
        else:
            raise ValidationAbort("invalid ordinal syntax")
    if not values or values != sorted(set(values)):
        raise ValidationAbort("ordinals are not canonical unique increasing values")
    return values


def assert_prejoin_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise ValidationAbort(f"forbidden postjoin key at {path}.{key}")
            assert_prejoin_keys(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            assert_prejoin_keys(nested, f"{path}[{index}]")


def _fold(value: object) -> int:
    text = str(value).strip().upper()
    if text.startswith("F"):
        text = text[1:]
    result = int(text)
    if result not in (1, 2, 3, 4):
        raise ValidationAbort("fold outside 1..4")
    return result


def normalize_schedule(
    source_files: Mapping[str, Mapping[str, str]],
    *,
    expected_query_count: int,
    expected_candidate_count: int,
    expected_fold_counts: Mapping[int, int] | None,
) -> list[dict[str, Any]]:
    for name, binding in source_files.items():
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValidationAbort(f"receipt source binding field drift: {name}")
        if not isinstance(binding["path"], str):
            raise ValidationAbort(f"receipt source path drift: {name}")
        expected_sha = exact_digest(binding["sha256"], f"receipt source {name}")
        path = ensure_unprotected_path(Path(binding["path"]))
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValidationAbort("receipt source-file hash drift")
    if "redacted_schedule" in source_files:
        payload = json.loads(
            Path(source_files["redacted_schedule"]["path"]).read_text(encoding="utf-8")
        )
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
        expected_fields = [
            "query_id",
            "historical_query_ordinal",
            "execution_ordinal",
            "heldout_fold",
            "source_image_sha256",
            "candidate_physical_rows",
            "candidate_axis_sha256",
        ]
        if (
            set(payload) != exact_top
            or payload.get("schema_version") != REDACTED_SCHEDULE_SCHEMA
            or payload.get("status") != REDACTED_SCHEDULE_STATUS
            or exact_int(payload.get("query_count"), "schedule query_count", minimum=0)
            != expected_query_count
            or (
                expected_fold_counts is not None
                and not exact_fold_counts(
                    payload.get("fold_query_counts"), expected_fold_counts
                )
            )
            or payload.get("record_fields") != expected_fields
            or not isinstance(payload.get("source_bindings"), Mapping)
            or not zero_access_audit(payload.get("access_audit"))
            or payload.get("logical_sha256") != logical_mapping_sha256(payload)
        ):
            raise ValidationAbort("redacted schedule envelope drift")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValidationAbort("redacted schedule records absent")
        if payload.get("record_sequence_sha256") != canonical_sha256(raw_records):
            raise ValidationAbort("redacted schedule record-sequence drift")
        assert_prejoin_keys(payload)
    else:
        if set(source_files) != {"checkpoint", "prejoin_folds", "candidate_ledger"}:
            raise ValidationAbort("two-source schedule binding set drift")
        folds = json.loads(
            Path(source_files["prejoin_folds"]["path"]).read_text(encoding="utf-8")
        )
        candidates = json.loads(
            Path(source_files["candidate_ledger"]["path"]).read_text(encoding="utf-8")
        )
        fold_rows, candidate_rows = folds.get("records"), candidates.get("rows")
        if not isinstance(fold_rows, list) or not isinstance(candidate_rows, list):
            raise ValidationAbort("two-source schedule rows absent")
        if len(fold_rows) != len(candidate_rows):
            raise ValidationAbort("two-source schedule length drift")
        raw_records = []
        for execution, (left, right) in enumerate(zip(fold_rows, candidate_rows)):
            if (
                str(left.get("query_id")) != str(right.get("query_id"))
                or int(left.get("query_ordinal")) != int(right.get("query_ordinal"))
            ):
                raise ValidationAbort("two-source record sequence mismatch")
            raw_records.append(
                {
                    "query_id": str(left["query_id"]),
                    "historical_query_ordinal": int(left["query_ordinal"]),
                    "execution_ordinal": execution,
                    "heldout_fold": _fold(left["inner_fold"]),
                    "source_image_sha256": str(left["source_image_sha256"]),
                    "candidate_physical_rows": sorted(
                        map(int, right["candidate_physical_rows"])
                    ),
                    "candidate_axis_sha256": canonical_sha256(
                        sorted(map(int, right["candidate_physical_rows"]))
                    ),
                }
            )
    if not isinstance(raw_records, list) or len(raw_records) != expected_query_count:
        raise ValidationAbort("schedule population drift")
    exact_fields = {
        "query_id",
        "historical_query_ordinal",
        "execution_ordinal",
        "heldout_fold",
        "source_image_sha256",
        "candidate_physical_rows",
        "candidate_axis_sha256",
    }
    records = []
    for source in raw_records:
        if not isinstance(source, Mapping) or set(source) != exact_fields:
            raise ValidationAbort("redacted schedule field drift")
        if not isinstance(source["candidate_physical_rows"], list):
            raise ValidationAbort("candidate row-set is not a list")
        source_rows = [
            exact_int(value, "candidate physical row", minimum=0)
            for value in source["candidate_physical_rows"]
        ]
        if (
            len(source_rows) != expected_candidate_count
            or len(set(source_rows)) != len(source_rows)
        ):
            raise ValidationAbort("candidate row-set drift")
        canonical = sorted(source_rows)
        if source_rows != canonical:
            raise ValidationAbort("redacted candidate axis is not numeric canonical")
        if source["candidate_axis_sha256"] != canonical_sha256(canonical):
            raise ValidationAbort("redacted candidate-axis hash drift")
        if not isinstance(source["query_id"], str) or not source["query_id"]:
            raise ValidationAbort("query ID is not an exact nonempty string")
        source_sha = exact_digest(source["source_image_sha256"], "query source")
        axis_sha = exact_digest(source["candidate_axis_sha256"], "candidate axis")
        historical = exact_int(
            source["historical_query_ordinal"], "historical query ordinal", minimum=0
        )
        execution = exact_int(
            source["execution_ordinal"], "execution ordinal", minimum=0
        )
        heldout = exact_int(source["heldout_fold"], "heldout fold", minimum=1)
        if heldout not in (1, 2, 3, 4):
            raise ValidationAbort("heldout fold outside 1..4")
        records.append(
            {
                "query_id": source["query_id"],
                "historical_query_ordinal": historical,
                "execution_ordinal": execution,
                "heldout_fold": heldout,
                "source_image_sha256": source_sha,
                "candidate_physical_rows": canonical,
                "candidate_axis_sha256": axis_sha,
            }
        )
    if len({row["query_id"] for row in records}) != expected_query_count:
        raise ValidationAbort("duplicate schedule query ID")
    if len({row["historical_query_ordinal"] for row in records}) != expected_query_count:
        raise ValidationAbort("duplicate historical ordinal")
    if sorted(row["execution_ordinal"] for row in records) != list(
        range(expected_query_count)
    ):
        raise ValidationAbort("execution ordinal is not dense 0..N-1")
    if expected_fold_counts is not None:
        counts = {
            fold: sum(row["heldout_fold"] == fold for row in records)
            for fold in (1, 2, 3, 4)
        }
        if counts != dict(expected_fold_counts):
            raise ValidationAbort("OOF fold-count drift")
    return records


def canonical_cache_path(root: Path, kind: str, cache_key: int) -> Path:
    key = exact_int(cache_key, "canonical cache key", minimum=0)
    if kind == "query":
        name = f"query_{key:04d}.pt"
    elif kind == "reference":
        name = f"reference_{key:05d}.pt"
    else:
        raise ValidationAbort("unknown cache kind")
    return Path(root) / name


def load_redacted_cache_index(
    source_files: Mapping[str, Mapping[str, str]],
    *,
    expected_query_count: int,
    expected_reference_rows: set[int] | None,
    expected_schedule_logical_sha256: str | None,
) -> dict[tuple[str, int], dict[str, Any]] | None:
    binding = source_files.get("redacted_cache_index")
    if binding is None:
        return None
    payload = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
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
    expected_fields = [
        "kind",
        "cache_key",
        "source_image_sha256",
        "tokens_fp16_payload_sha256",
        "cache_logical_sha256",
        "valid_patch_mask_sha256",
        "geometry_logical_sha256",
    ]
    if (
        set(payload) != exact_top
        or payload.get("schema_version") != REDACTED_CACHE_INDEX_SCHEMA
        or payload.get("status") != REDACTED_CACHE_INDEX_STATUS
        or exact_int(payload.get("query_count"), "cache-index query_count", minimum=0)
        != expected_query_count
        or payload.get("record_fields") != expected_fields
        or not isinstance(payload.get("source_bindings"), Mapping)
        or not zero_access_audit(payload.get("access_audit"))
        or payload.get("logical_sha256") != logical_mapping_sha256(payload)
    ):
        raise ValidationAbort("redacted cache-index envelope drift")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValidationAbort("redacted cache-index records absent")
    if (
        exact_int(payload.get("record_count"), "cache-index record_count", minimum=0)
        != len(records)
        or payload.get("record_sequence_sha256") != canonical_sha256(records)
    ):
        raise ValidationAbort("redacted cache-index record-sequence drift")
    assert_prejoin_keys(payload)
    exact = {
        "kind",
        "cache_key",
        "source_image_sha256",
        "tokens_fp16_payload_sha256",
        "cache_logical_sha256",
        "valid_patch_mask_sha256",
        "geometry_logical_sha256",
    }
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in records:
        if (
            not isinstance(row, Mapping)
            or set(row) != exact
            or row.get("kind") not in {"query", "reference"}
        ):
            raise ValidationAbort("redacted cache-index record field drift")
        key = (
            row["kind"],
            exact_int(row["cache_key"], "cache key", minimum=0),
        )
        if key in result:
            raise ValidationAbort("duplicate redacted cache-index key")
        for name in exact - {"kind", "cache_key"}:
            exact_digest(row[name], f"redacted cache-index {name}")
        result[key] = dict(row)
    if sorted(key for kind, key in result if kind == "query") != list(
        range(expected_query_count)
    ):
        raise ValidationAbort("redacted query cache keys are not dense execution ordinals")
    observed_references = {key for kind, key in result if kind == "reference"}
    if expected_reference_rows is not None and observed_references != set(
        expected_reference_rows
    ):
        raise ValidationAbort("redacted reference keys differ from schedule union")
    if (
        exact_int(
            payload.get("reference_count"), "cache-index reference_count", minimum=0
        )
        != len(observed_references)
        or exact_int(
            payload.get("record_count"), "cache-index record_count", minimum=0
        )
        != expected_query_count + len(observed_references)
    ):
        raise ValidationAbort("redacted cache-index population drift")
    if (
        expected_schedule_logical_sha256 is not None
        and payload["source_bindings"].get("redacted_schedule_logical_sha256")
        != expected_schedule_logical_sha256
    ):
        raise ValidationAbort("cache index is not bound to redacted schedule")
    return result


def validate_schedule_cache_index_join(
    schedule: Sequence[Mapping[str, Any]],
    cache_index: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    expected_query_keys = {
        ("query", exact_int(row["execution_ordinal"], "execution ordinal", minimum=0))
        for row in schedule
    }
    observed_query_keys = {key for key in cache_index if key[0] == "query"}
    if observed_query_keys != expected_query_keys:
        raise ValidationAbort("global schedule/cache query-key join drift")
    for row in schedule:
        key = ("query", row["execution_ordinal"])
        index_row = cache_index[key]
        if index_row.get("source_image_sha256") != row["source_image_sha256"]:
            raise ValidationAbort(
                f"global schedule/cache query-source join drift: {row['query_id']}"
            )


def validate_matched_checkpoint_binding(
    manifest: Mapping[str, Any],
    *,
    arm: str,
    fold: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> None:
    """Independently bind a leaf checkpoint to the matched arm/fold cell."""

    arm_key = {"RCDE_BAG": "bag", "RCDE_CONTEXT": "context"}.get(arm)
    if arm_key is None:
        raise ValidationAbort("checkpoint/manifest arm-fold binding drift")
    requested_fold = exact_int(
        fold, "checkpoint/manifest requested fold", minimum=1
    )
    if requested_fold not in (1, 2, 3, 4):
        raise ValidationAbort("checkpoint/manifest arm-fold binding drift")
    raw_folds = manifest.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != 4:
        raise ValidationAbort("checkpoint/manifest fold table drift")
    indexed: dict[int, Mapping[str, Any]] = {}
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, Mapping):
            raise ValidationAbort("checkpoint/manifest fold table drift")
        fold_number = exact_int(
            raw_fold.get("outer_fold"), "matched-foldset outer_fold", minimum=1
        )
        if fold_number not in (1, 2, 3, 4) or fold_number in indexed:
            raise ValidationAbort("checkpoint/manifest fold table drift")
        indexed[fold_number] = raw_fold
    if set(indexed) != {1, 2, 3, 4}:
        raise ValidationAbort("checkpoint/manifest fold table drift")
    arm_record = indexed[requested_fold].get(arm_key)
    binding = arm_record.get("checkpoint") if isinstance(arm_record, Mapping) else None
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ValidationAbort("checkpoint/manifest arm-fold binding drift")
    path_text = binding.get("path")
    if not isinstance(path_text, str) or not path_text or Path(path_text).is_absolute():
        raise ValidationAbort("checkpoint/manifest arm-fold binding drift")
    relative_path = Path(path_text)
    current = ROOT
    for part in relative_path.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            raise ValidationAbort("checkpoint/manifest arm-fold binding drift")
    bound_path = (ROOT / relative_path).resolve()
    if bound_path != ROOT and ROOT not in bound_path.parents:
        raise ValidationAbort("checkpoint/manifest arm-fold binding drift")
    digest = exact_digest(binding.get("sha256"), "matched checkpoint digest")
    expected_path = ensure_unprotected_path(checkpoint_path)
    expected_digest = exact_digest(checkpoint_sha256, "receipt checkpoint digest")
    if (
        ensure_unprotected_path(bound_path) != expected_path
        or digest != expected_digest
        or digest != file_sha256(expected_path)
    ):
        raise ValidationAbort("checkpoint/manifest arm-fold binding drift")


def validate_formal_predecessors(
    source_files: Mapping[str, Mapping[str, str]],
    *,
    arm: str,
    fold: int,
) -> None:
    redacted = json.loads(
        ensure_unprotected_path(
            Path(source_files["redacted_inputs_validation"]["path"])
        ).read_text(encoding="utf-8")
    )
    if (
        redacted.get("schema_version") != REDACTED_INPUTS_VALIDATION_SCHEMA
        or redacted.get("status") != REDACTED_INPUTS_VALIDATION_STATUS
        or redacted.get("schedule_file_sha256")
        != source_files["redacted_schedule"]["sha256"]
        or redacted.get("cache_index_file_sha256")
        != source_files["redacted_cache_index"]["sha256"]
        or redacted.get("automatic_stage_advance") is not False
        or redacted.get("scientific_decision") is not None
    ):
        raise ValidationAbort("redacted-input predecessor closure drift")
    manifest_path = ensure_unprotected_path(
        Path(source_files["matched_foldset_manifest"]["path"])
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != MATCHED_FOLDSET_MANIFEST_SCHEMA
        or manifest.get("status") != MATCHED_FOLDSET_MANIFEST_STATUS
        or manifest.get("arms") != ["RCDE_BAG", "RCDE_CONTEXT"]
        or manifest.get("outer_folds") != [1, 2, 3, 4]
        or manifest.get("automatic_stage_advance") is not False
        or manifest.get("scientific_GO_or_NO_GO") is not None
    ):
        raise ValidationAbort("matched-foldset manifest predecessor drift")
    validate_matched_checkpoint_binding(
        manifest,
        arm=arm,
        fold=fold,
        checkpoint_path=Path(source_files["checkpoint"]["path"]),
        checkpoint_sha256=source_files["checkpoint"]["sha256"],
    )
    matched_validation = json.loads(
        ensure_unprotected_path(
            Path(source_files["matched_foldset_validation"]["path"])
        ).read_text(encoding="utf-8")
    )
    anchor = matched_validation.get("matched_foldset_manifest")
    anchor_path_text = anchor.get("path") if isinstance(anchor, Mapping) else None
    anchor_path = (
        None
        if not isinstance(anchor_path_text, str) or Path(anchor_path_text).is_absolute()
        else (ROOT / anchor_path_text).resolve()
    )
    if (
        matched_validation.get("schema_version") != MATCHED_FOLDSET_VALIDATION_SCHEMA
        or matched_validation.get("status") != MATCHED_FOLDSET_VALIDATION_STATUS
        or matched_validation.get("arms") != ["RCDE_BAG", "RCDE_CONTEXT"]
        or matched_validation.get("outer_folds") != [1, 2, 3, 4]
        or not isinstance(anchor, Mapping)
        or anchor_path != manifest_path
        or anchor.get("sha256") != source_files["matched_foldset_manifest"]["sha256"]
        or matched_validation.get("automatic_stage_advance") is not False
        or matched_validation.get("scientific_GO_or_NO_GO") is not None
    ):
        raise ValidationAbort("matched-foldset validation predecessor drift")


def validate_cache_payload(
    payload: Mapping[str, Any],
    expected_model_sha256: str,
    *,
    index_record: Mapping[str, Any] | None,
    expected_source_image_sha256: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, mask = payload.get("tokens_fp16"), payload.get("valid_patch_mask")
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.dtype != torch.float16
        or tokens.ndim != 4
        or tuple(tokens.shape[:2]) != (1, 4)
        or tokens.shape[-1] != 768
        or not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 3
        or mask.shape[0] != 1
        or tokens.shape[2] != mask[0].numel()
        or not bool(torch.isfinite(tokens).all())
        or not bool(mask.any())
    ):
        raise ValidationAbort("cache tensor structure drift")
    if payload.get("model_checkpoint_logical_sha256") != expected_model_sha256:
        raise ValidationAbort("cache model identity drift")
    if payload.get("logical_sha256") != cache_logical_sha256(payload):
        raise ValidationAbort("cache logical hash drift")
    if (
        expected_source_image_sha256 is not None
        and payload.get("source_image_sha256") != expected_source_image_sha256
    ):
        raise ValidationAbort("cache source-image binding drift")
    if index_record is not None:
        receipt = payload.get("cache_receipt")
        geometry = payload.get("geometry_receipt")
        if not isinstance(receipt, Mapping) or not isinstance(geometry, Mapping):
            raise ValidationAbort("cache nested receipts absent")
        if (
            index_record.get("source_image_sha256")
            != payload.get("source_image_sha256")
            or index_record.get("tokens_fp16_payload_sha256")
            != receipt.get("tokens_fp16_payload_sha256")
            or index_record.get("cache_logical_sha256")
            != payload.get("logical_sha256")
            or index_record.get("valid_patch_mask_sha256")
            != receipt.get("valid_patch_mask_sha256")
            or index_record.get("geometry_logical_sha256")
            != geometry.get("logical_sha256")
        ):
            raise ValidationAbort("redacted cache-index payload closure failed")
    return tokens, mask


def load_model(
    checkpoint_path: Path,
    *,
    arm: str,
    fold: int,
    model_factory: Callable[[], Any],
    expected_parameter_count: int,
) -> tuple[Any, Mapping[str, Any]]:
    checkpoint = safe_torch_load(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise ValidationAbort("checkpoint is not a mapping")
    if (
        checkpoint.get("arm") != arm
        or exact_int(checkpoint.get("outer_fold"), "checkpoint outer_fold") != fold
        or exact_int(checkpoint.get("update"), "checkpoint update", minimum=0)
        != EXPECTED_UPDATE
        or exact_int(checkpoint.get("seed"), "checkpoint seed", minimum=0)
        != EXPECTED_SEED
    ):
        raise ValidationAbort("checkpoint scope drift")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValidationAbort("checkpoint state absent")
    if checkpoint.get("final_state_sha256") != state_dict_sha256(state):
        raise ValidationAbort("checkpoint state hash drift")
    model = model_factory()
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != expected_parameter_count:
        raise ValidationAbort("model parameter count drift")
    model.eval()
    return model, checkpoint


def sampled_pairs(candidate_count: int) -> list[tuple[int, int]]:
    proposed = [
        (0, 1),
        (0, candidate_count - 1),
        (candidate_count // 2, candidate_count - 1),
        (candidate_count // 3, (2 * candidate_count) // 3),
    ]
    result = []
    for left, right in proposed:
        if 0 <= left < right < candidate_count and (left, right) not in result:
            result.append((left, right))
    return result


def validate_candidate(
    candidate: Mapping[str, Any],
    *,
    query_mask: torch.Tensor,
    reference_mask: torch.Tensor,
) -> None:
    raw = candidate.get("raw_summary")
    null = candidate.get("null_summary")
    relational = candidate.get("relational")
    u, v = candidate.get("u"), candidate.get("v")
    nq, nr = query_mask.numel(), reference_mask.numel()
    if not all(isinstance(value, torch.Tensor) for value in (raw, null, relational, u, v)):
        raise ValidationAbort("candidate evidence tensor absent")
    if (
        raw.dtype != torch.float32
        or null.dtype != torch.float32
        or relational.dtype != torch.float32
        or u.dtype != torch.float32
        or v.dtype != torch.float32
        or tuple(raw.shape) != (nq, SUMMARY_DIM)
        or tuple(null.shape) != (nq, SUMMARY_DIM)
        or tuple(relational.shape) != (nq, SUMMARY_DIM)
        or tuple(u.shape) != (nq,)
        or tuple(v.shape) != (nr,)
        or not all(bool(torch.isfinite(value).all()) for value in (raw, null, relational, u, v))
        or not torch.equal(relational, raw - null)
    ):
        raise ValidationAbort("candidate evidence shape/dtype/finite closure failed")
    q_valid = query_mask.flatten()
    r_valid = reference_mask.flatten()
    if (
        not torch.equal(raw[~q_valid], torch.zeros_like(raw[~q_valid]))
        or not torch.equal(null[~q_valid], torch.zeros_like(null[~q_valid]))
        or not torch.equal(relational[~q_valid], torch.zeros_like(relational[~q_valid]))
        or not torch.equal(u[~q_valid], torch.zeros_like(u[~q_valid]))
        or not torch.equal(v[~r_valid], torch.zeros_like(v[~r_valid]))
        or bool((u < 0).any())
        or bool((u > 1).any())
        or bool((v < 0).any())
        or bool((v > 1).any())
    ):
        raise ValidationAbort("candidate padding/probability closure failed")
    hashes = candidate.get("matrix_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != MATRIX_HASH_KEYS or not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in hashes.values()
    ):
        raise ValidationAbort("Q/G/P/v hash contract drift")
    tensor_hashes = candidate.get("tensor_sha256")
    expected_tensors = {
        "raw_summary": raw,
        "null_summary": null,
        "relational": relational,
        "u": u,
        "v": v,
    }
    if not isinstance(tensor_hashes, Mapping) or set(tensor_hashes) != set(
        expected_tensors
    ):
        raise ValidationAbort("candidate tensor hash field drift")
    for name, tensor in expected_tensors.items():
        if tensor_hashes[name] != tensor_sha256(tensor):
            raise ValidationAbort("candidate tensor hash mismatch")
    slices = candidate.get("matrix_audit_slices")
    expected_indices = [0, (nq * nr) // 2, nq * nr - 1]
    if not isinstance(slices, list) or [int(row["flat_index"]) for row in slices] != expected_indices:
        raise ValidationAbort("candidate audit index drift")
    for item in slices:
        if set(item) != {"flat_index", "query_index", "reference_index", "Q", "G", "P", "v"}:
            raise ValidationAbort("candidate audit field drift")
        flat = int(item["flat_index"])
        qi, ri = divmod(flat, nr)
        if int(item["query_index"]) != qi or int(item["reference_index"]) != ri:
            raise ValidationAbort("candidate audit coordinate drift")
        q, g, p, dust = map(float, (item["Q"], item["G"], item["P"], item["v"]))
        if not all(torch.isfinite(torch.tensor(value)) for value in (q, g, p, dust)):
            raise ValidationAbort("nonfinite candidate audit scalar")
        if min(q, g, p, dust) < -1e-7 or max(q, g, p, dust) > 1.0 + 1e-7:
            raise ValidationAbort("candidate audit probability range drift")
        if abs(p - q * g) > 1e-6 or abs(dust - float(v[ri])) > 1e-6:
            raise ValidationAbort("candidate audit algebraic closure failed")
    stream = candidate.get("streaming_receipt")
    if (
        not isinstance(stream, Mapping)
        or int(stream.get("logical_cost_volume_build_count", -1)) != 1
        or int(stream.get("summary_pass_count", -1)) != 3
        or stream.get("full_consensus_logits_resident") is not False
        or len(stream.get("real_pass_tile_counts", [])) != 3
        or len(stream.get("null_pass_tile_counts", [])) != 3
        or min(map(int, stream["real_pass_tile_counts"])) <= 0
        or min(map(int, stream["null_pass_tile_counts"])) <= 0
    ):
        raise ValidationAbort("audited streaming receipt drift")


def record_logical_sha256(record: Mapping[str, Any]) -> str:
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
            "unordered_pair_evaluation_count": record["unordered_pair_evaluation_count"],
        }
    )


def validate_shard(
    *,
    shard_dir: Path,
    cache_root: Path,
    arm: str,
    fold: int,
    requested_query_ordinals: Sequence[int],
    model_factory: Callable[[], Any] = DINO_RCDE_V1_2,
    expected_parameter_count: int = EXPECTED_PARAMETER_COUNT,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    expected_fold_counts: Mapping[int, int] | None = EXPECTED_FOLD_COUNTS,
    expected_cache_model_sha256: str = EXPECTED_CACHE_MODEL_SHA256,
    test_mode: bool = False,
) -> dict[str, Any]:
    requested = [
        exact_int(value, "requested historical query ordinal", minimum=0)
        for value in requested_query_ordinals
    ]
    if not requested or requested != sorted(set(requested)):
        raise ValidationAbort("requested ordinals are not canonical")
    shard_root = ensure_unprotected_path(shard_dir)
    cache_root = ensure_unprotected_path(cache_root)
    receipt_path = shard_root / "receipt.json"
    artifact_path = shard_root / "shard.pt"
    if not receipt_path.is_file() or not artifact_path.is_file():
        raise ValidationAbort("committed shard files are incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != SHARD_STATUS:
        raise ValidationAbort("receipt schema/status drift")
    assert_prejoin_keys(receipt)
    if (
        receipt.get("arm") != arm
        or exact_int(receipt.get("outer_fold"), "receipt outer_fold") != fold
        or not exact_int_list(receipt.get("historical_query_ordinals"), requested)
        or receipt.get("artifact", {}).get("sha256") != file_sha256(artifact_path)
    ):
        raise ValidationAbort("receipt scope/artifact binding drift")
    source_files = receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValidationAbort("receipt source bindings absent")
    if test_mode:
        allowed_sets = (
            {"checkpoint", "redacted_schedule"},
            {"checkpoint", "redacted_schedule", "redacted_cache_index"},
            {"checkpoint", "prejoin_folds", "candidate_ledger"},
        )
        if set(source_files) not in allowed_sets:
            raise ValidationAbort("test receipt source binding set drift")
    elif set(source_files) != {
        "checkpoint",
        "redacted_schedule",
        "redacted_cache_index",
        "protocol",
        "authority",
        "redacted_inputs_validation",
        "matched_foldset_manifest",
        "matched_foldset_validation",
    }:
        raise ValidationAbort("formal receipt source binding set drift")
    schedule = normalize_schedule(
        source_files,
        expected_query_count=expected_query_count,
        expected_candidate_count=expected_candidate_count,
        expected_fold_counts=expected_fold_counts,
    )
    if not test_mode:
        validate_formal_predecessors(source_files, arm=arm, fold=fold)
    expected_source_hashes: dict[str, str] = {}
    if "redacted_schedule" in source_files:
        expected_source_hashes["redacted_schedule_sha256"] = source_files[
            "redacted_schedule"
        ]["sha256"]
    else:
        expected_source_hashes.update(
            {
                "prejoin_folds_sha256": source_files["prejoin_folds"]["sha256"],
                "candidate_ledger_sha256": source_files["candidate_ledger"][
                    "sha256"
                ],
            }
        )
    for source_name, hash_name in (
        ("redacted_cache_index", "redacted_cache_index_sha256"),
        ("protocol", "protocol_sha256"),
        ("authority", "authority_sha256"),
        ("redacted_inputs_validation", "redacted_inputs_validation_sha256"),
        ("matched_foldset_manifest", "matched_foldset_manifest_sha256"),
        ("matched_foldset_validation", "matched_foldset_validation_sha256"),
    ):
        if source_name in source_files:
            expected_source_hashes[hash_name] = source_files[source_name]["sha256"]
    by_historical = {row["historical_query_ordinal"]: row for row in schedule}
    selected = [by_historical.get(value) for value in requested]
    if any(row is None for row in selected) or any(
        row["heldout_fold"] != fold for row in selected
    ):
        raise ValidationAbort("requested shard is not exact OOF")

    payload = safe_torch_load(artifact_path)
    if not isinstance(payload, Mapping):
        raise ValidationAbort("shard payload is not a mapping")
    if payload.get("schema_version") != SHARD_SCHEMA or payload.get("status") != SHARD_STATUS:
        raise ValidationAbort("shard schema/status drift")
    assert_prejoin_keys(payload)
    if (
        payload.get("arm") != arm
        or exact_int(payload.get("outer_fold"), "payload outer_fold") != fold
        or not exact_int_list(payload.get("historical_query_ordinals"), requested)
        or exact_int(payload.get("query_count"), "payload query_count", minimum=0)
        != len(selected)
        or exact_int(
            payload.get("candidate_count_per_query"),
            "payload candidate_count_per_query",
            minimum=0,
        )
        != expected_candidate_count
        or exact_int(
            payload.get("pair_count_per_query"),
            "payload pair_count_per_query",
            minimum=0,
        )
        != expected_candidate_count * (expected_candidate_count - 1) // 2
        or receipt["artifact"].get("record_sequence_sha256")
        != payload.get("record_sequence_sha256")
        or payload.get("claim_level") != EXPECTED_CLAIM_LEVEL
        or receipt.get("claim_level") != EXPECTED_CLAIM_LEVEL
        or payload.get("execution_mode")
        != ("TEST_FIXTURE" if test_mode else "FORMAL_BOUND")
        or receipt.get("execution_mode") != payload.get("execution_mode")
        or payload.get("automatic_stage_advance") is not False
        or receipt.get("automatic_stage_advance") is not False
        or payload.get("natural_training_authorized") is not False
        or receipt.get("natural_training_authorized") is not False
        or payload.get("next_authorized_stage") != EXPECTED_NEXT_STAGE
        or receipt.get("next_authorized_stage") != EXPECTED_NEXT_STAGE
        or payload.get("source_hashes") != expected_source_hashes
    ):
        raise ValidationAbort("shard top-level closure drift")
    checkpoint_path = Path(source_files["checkpoint"]["path"])
    model, checkpoint = load_model(
        checkpoint_path,
        arm=arm,
        fold=fold,
        model_factory=model_factory,
        expected_parameter_count=expected_parameter_count,
    )
    if (
        payload.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or payload.get("checkpoint_final_state_sha256")
        != checkpoint.get("final_state_sha256")
        or payload.get("cache_backbone_sha256") != expected_cache_model_sha256
    ):
        raise ValidationAbort("checkpoint/cache binding drift")
    schedule_reference_rows = {
        int(row)
        for schedule_row in schedule
        for row in schedule_row["candidate_physical_rows"]
    }
    schedule_logical_sha256 = (
        None
        if "redacted_schedule" not in source_files
        else str(
            json.loads(
                Path(source_files["redacted_schedule"]["path"]).read_text(
                    encoding="utf-8"
                )
            )["logical_sha256"]
        )
    )
    redacted_index = load_redacted_cache_index(
        source_files,
        expected_query_count=expected_query_count,
        expected_reference_rows=schedule_reference_rows,
        expected_schedule_logical_sha256=schedule_logical_sha256,
    )
    if not test_mode and redacted_index is None:
        raise ValidationAbort("formal validation requires redacted cache index")
    if redacted_index is not None:
        validate_schedule_cache_index_join(schedule, redacted_index)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(selected):
        raise ValidationAbort("shard record count drift")
    observed_hashes = []
    sampled_closure_count = 0
    reorder_closure_count = 0
    for expected, record in zip(selected, records):
        assert expected is not None
        for key in (
            "query_id",
            "historical_query_ordinal",
            "execution_ordinal",
            "heldout_fold",
            "source_image_sha256",
            "candidate_physical_rows",
            "candidate_axis_sha256",
        ):
            if record.get(key) != expected.get(key):
                raise ValidationAbort(f"schedule/shard record mismatch: {key}")
        raw_rows = record.get("candidate_physical_rows")
        if not isinstance(raw_rows, list):
            raise ValidationAbort("candidate axis is not a list")
        rows = [
            exact_int(value, "shard candidate physical row", minimum=0)
            for value in raw_rows
        ]
        if rows != sorted(rows) or len(rows) != len(set(rows)):
            raise ValidationAbort("candidate axis is not numeric canonical")
        if record["candidate_axis_sha256"] != canonical_sha256(rows):
            raise ValidationAbort("candidate axis hash drift")
        execution = int(record["execution_ordinal"])
        query_index_record = (
            None if redacted_index is None else redacted_index.get(("query", execution))
        )
        if not test_mode and query_index_record is None:
            raise ValidationAbort("query absent from redacted cache index")
        query_path = canonical_cache_path(cache_root, "query", execution)
        if not query_path.is_file() or file_sha256(query_path) != record["query_cache_file_sha256"]:
            raise ValidationAbort("query cache binding drift")
        query_payload = safe_torch_load(query_path)
        if not isinstance(query_payload, Mapping):
            raise ValidationAbort("query cache payload is not a mapping")
        _, query_mask_payload = validate_cache_payload(
            query_payload,
            expected_cache_model_sha256,
            index_record=query_index_record,
            expected_source_image_sha256=record["source_image_sha256"],
        )
        query_mask = record.get("query_valid_patch_mask")
        if (
            not isinstance(query_mask, torch.Tensor)
            or query_mask.dtype != torch.bool
            or not torch.equal(query_mask, query_mask_payload[0])
            or list(query_mask.shape) != list(record.get("query_grid", []))
        ):
            raise ValidationAbort("query mask/grid closure drift")
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != expected_candidate_count:
            raise ValidationAbort("candidate artifact count drift")
        relational_rows = []
        for physical_row, candidate in zip(rows, candidates):
            if exact_int(candidate.get("physical_row"), "candidate physical_row") != physical_row:
                raise ValidationAbort("candidate artifact row/order drift")
            reference_index_record = (
                None
                if redacted_index is None
                else redacted_index.get(("reference", int(physical_row)))
            )
            if not test_mode and reference_index_record is None:
                raise ValidationAbort("reference absent from redacted cache index")
            reference_path = canonical_cache_path(
                cache_root, "reference", int(physical_row)
            )
            if not reference_path.is_file() or file_sha256(reference_path) != candidate.get(
                "cache_file_sha256"
            ):
                raise ValidationAbort("reference cache binding drift")
            reference_payload = safe_torch_load(reference_path)
            if not isinstance(reference_payload, Mapping):
                raise ValidationAbort("reference cache payload is not a mapping")
            _, reference_mask_payload = validate_cache_payload(
                reference_payload,
                expected_cache_model_sha256,
                index_record=reference_index_record,
                expected_source_image_sha256=(
                    None
                    if reference_index_record is None
                    else str(reference_index_record["source_image_sha256"])
                ),
            )
            reference_mask = candidate.get("reference_valid_patch_mask")
            if (
                not isinstance(reference_mask, torch.Tensor)
                or reference_mask.dtype != torch.bool
                or not torch.equal(reference_mask, reference_mask_payload[0])
                or list(reference_mask.shape) != list(candidate.get("reference_grid", []))
            ):
                raise ValidationAbort("reference mask/grid closure drift")
            validate_candidate(
                candidate, query_mask=query_mask, reference_mask=reference_mask
            )
            relational_rows.append(candidate["relational"])
        delta = record.get("delta")
        if (
            not isinstance(delta, torch.Tensor)
            or delta.dtype != torch.float32
            or tuple(delta.shape) != (expected_candidate_count, expected_candidate_count)
            or not bool(torch.isfinite(delta).all())
            or record.get("delta_sha256") != tensor_sha256(delta)
            or not torch.equal(delta, -delta.T)
            or not torch.equal(
                torch.diagonal(delta),
                torch.zeros(expected_candidate_count, dtype=torch.float32),
            )
            or exact_int(
                record.get("unordered_pair_evaluation_count"),
                "unordered_pair_evaluation_count",
                minimum=0,
            )
            != expected_candidate_count * (expected_candidate_count - 1) // 2
        ):
            raise ValidationAbort("full Delta closure failed")
        relational = torch.stack(relational_rows, dim=0)
        pairs = sampled_pairs(expected_candidate_count)
        with torch.no_grad():
            for left, right in pairs:
                scalar = model.compare_relational(
                    relational[left], relational[right], query_mask
                ).logit.detach().cpu().to(torch.float32)
                if not torch.allclose(
                    scalar, delta[left, right], rtol=1e-6, atol=1e-6
                ):
                    raise ValidationAbort("sampled compare_relational closure failed")
                sampled_closure_count += 1
            reverse = torch.arange(expected_candidate_count - 1, -1, -1)
            reversed_relational = relational.index_select(0, reverse)
            for left, right in pairs:
                reverse_left = expected_candidate_count - 1 - left
                reverse_right = expected_candidate_count - 1 - right
                value = model.compare_relational_batch(
                    reversed_relational,
                    torch.tensor([reverse_left]),
                    torch.tensor([reverse_right]),
                    query_mask,
                )[0].detach().cpu()
                if not torch.allclose(
                    value, delta[left, right], rtol=1e-6, atol=1e-6
                ):
                    raise ValidationAbort("candidate reorder closure failed")
                reorder_closure_count += 1
        logical = record_logical_sha256(record)
        if record.get("logical_sha256") != logical:
            raise ValidationAbort("record logical hash drift")
        observed_hashes.append(logical)
    if (
        payload.get("record_logical_sha256s") != observed_hashes
        or payload.get("record_sequence_sha256") != canonical_sha256(observed_hashes)
        or payload.get("label_read_count") != 0
        or payload.get("scientific_GO_or_NO_GO") is not None
    ):
        raise ValidationAbort("record sequence/protected boundary drift")
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": PASS_STATUS,
        "arm": arm,
        "outer_fold": fold,
        "historical_query_ordinals": requested,
        "query_count": len(records),
        "candidate_count_per_query": expected_candidate_count,
        "unordered_pairs_per_query": expected_candidate_count
        * (expected_candidate_count - 1)
        // 2,
        "sampled_comparator_closure_count": sampled_closure_count,
        "candidate_reorder_closure_count": reorder_closure_count,
        "numeric_physical_row_order": True,
        "execution_ordinal_cache_binding": True,
        "full_delta_strict_antisymmetry": True,
        "postjoin_fields_absent": True,
        "artifact_sha256": file_sha256(artifact_path),
        "receipt_sha256": file_sha256(receipt_path),
        "scientific_GO_or_NO_GO": None,
    }


def atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    output = ensure_unprotected_path(path, must_exist=False)
    if os.path.lexists(output):
        raise ValidationAbort(f"validation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if os.path.lexists(temporary):
        raise ValidationAbort(f"validation temporary already exists: {temporary}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.link(temporary, output)
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    temporary.unlink()
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("RCDE_BAG", "RCDE_CONTEXT"), required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--query-ordinals", required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    result = validate_shard(
        shard_dir=args.shard_dir,
        cache_root=args.cache_root,
        arm=args.arm,
        fold=args.fold,
        requested_query_ordinals=parse_ordinals(args.query_ordinals),
        test_mode=args.test_mode,
    )
    atomic_json_no_clobber(args.validation_output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
