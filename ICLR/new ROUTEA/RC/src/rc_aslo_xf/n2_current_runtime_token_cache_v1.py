"""Pure helpers for the N2 987-query current-runtime token cache.

This module deliberately contains no label join, model training, gallery
lookup, or model loading.  The materializer and the independent validators use
the same frozen data schema, while validators independently reconstruct the
actual image decode and encoder forward.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import torch


SHARD_COUNT = 16
QUERY_COUNT = 987
CURRENT64_COUNT = 64
FRESH_COUNT = QUERY_COUNT - CURRENT64_COUNT
EXPECTED_D1_PARAMETER_COUNT = 49_792
EXPECTED_D1_PARAMETER_SCHEMA_SHA256 = "8594609ae4028a220fe3720a73545e1e6e654b79dda0ba0e39a78ea2541473f1"

READY_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_SHARD_READY"
VALIDATED_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_SHARD_VALIDATED"
AGGREGATE_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_AGGREGATE_READY"
FINAL_STATUS = "ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_INDEPENDENT_VALIDATION_PASS"

RECORD_KEYS = frozenset(
    {
        "query_id",
        "query_ordinal",
        "heldout_fold",
        "track",
        "source_image_sha256",
        "source_exif_orientation",
        "decode_frame",
        "raw_size_hw",
        "oriented_size_hw",
        "grid_shape",
        "image_tokens",
        "image_tokens_sha256",
        "template_tokens",
        "template_tokens_sha256",
        "materialization_mode",
        "current64_source_shard",
        "current64_source_payload_sha256",
        "model_update_count",
    }
)

# These fields are forbidden in every prejoin record.  The list is exact and
# intentionally broader than the fields used by historical pipelines.
FORBIDDEN_RECORD_KEYS = frozenset(
    {
        "target",
        "target_id",
        "target_identity",
        "target_label",
        "label",
        "exact_label",
        "identity",
        "identity_id",
        "supergroup",
        "supergroup_id",
        "product_stem",
        "role",
        "correct",
        "outcome",
        "winner",
        "challenger",
        "candidate",
        "candidate_id",
        "candidate_row",
        "physical_row",
        "gallery_row",
        "slot",
        "switch",
        "action",
    }
)

# One hash-selected member from every contiguous shard, plus all four E0
# fixtures.  This is frozen before any 987-query cache is produced.
FRESH_FORWARD_AUDIT_QUERY_IDS = (
    "DIFFICULT-0023",
    "DIFFICULT-0079",
    "OUTCOME-0005",
    "OUTCOME-0028",
    "OUTCOME-0089",
    "OUTCOME-0185",
    "OUTCOME-0244",
    "OUTCOME-0296",
    "OUTCOME-0344",
    "OUTCOME-0392",
    "OUTCOME-0450",
    "OUTCOME-0568",
    "OUTCOME-0604",
    "OUTCOME-0675",
    "OUTCOME-0749",
    "OUTCOME-0760",
    "OUTCOME-0533",
    "DIFFICULT-0128",
    "NDV2-007-P01",
    "DIFFICULT-0025",
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def shard_bounds(shard: int) -> tuple[int, int]:
    if shard not in range(SHARD_COUNT):
        raise ValueError(f"shard must be in 0..{SHARD_COUNT - 1}")
    return shard * QUERY_COUNT // SHARD_COUNT, (shard + 1) * QUERY_COUNT // SHARD_COUNT


def shard_for_ordinal(ordinal: int) -> int:
    if ordinal not in range(QUERY_COUNT):
        raise ValueError(f"query ordinal must be in 0..{QUERY_COUNT - 1}")
    # The explicit search avoids a boundary bug when QUERY_COUNT is not
    # divisible by SHARD_COUNT.
    for shard in range(SHARD_COUNT):
        begin, end = shard_bounds(shard)
        if begin <= ordinal < end:
            return shard
    raise AssertionError("unreachable shard lookup")


def expected_query_ordinals(shard: int) -> tuple[int, ...]:
    begin, end = shard_bounds(shard)
    return tuple(range(begin, end))


def nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.add(str(key))
            found.update(nested_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(nested_keys(item))
    return found


def validate_prejoin_record_schema(record: Mapping[str, Any]) -> None:
    keys = set(map(str, record.keys()))
    if keys != RECORD_KEYS:
        raise ValueError(
            f"prejoin record schema drift missing={sorted(RECORD_KEYS - keys)} "
            f"extra={sorted(keys - RECORD_KEYS)}"
        )
    forbidden = nested_keys(record).intersection(FORBIDDEN_RECORD_KEYS)
    if forbidden:
        raise ValueError(f"identity/outcome field entered prejoin record: {sorted(forbidden)}")


def parameter_schema(module: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in module.named_parameters()
    ]


def parameter_schema_sha256(module: torch.nn.Module) -> str:
    return canonical_sha256(parameter_schema(module))


def validate_audit_population(query_ids: Sequence[str], ledger_rows: Sequence[Mapping[str, Any]]) -> None:
    if tuple(query_ids) != FRESH_FORWARD_AUDIT_QUERY_IDS:
        raise ValueError("fresh-forward audit population drift")
    by_id = {str(row["query_id"]): row for row in ledger_rows}
    if len(by_id) != QUERY_COUNT or any(query_id not in by_id for query_id in query_ids):
        raise ValueError("fresh-forward audit member absent from canonical ledger")
    covered_shards = {shard_for_ordinal(int(by_id[query_id]["query_ordinal"])) for query_id in query_ids}
    if covered_shards != set(range(SHARD_COUNT)):
        raise ValueError("fresh-forward audit does not cover every shard")
    if not {"OUTCOME-0533", "DIFFICULT-0128", "NDV2-007-P01", "DIFFICULT-0025"}.issubset(query_ids):
        raise ValueError("fresh-forward audit lost an E0 fixture")


def assert_parameter_names_reference_defined(names: Iterable[str]) -> None:
    bad_fragments = ("identity", "gallery_row", "physical_row", "candidate_slot", "reference_slot")
    bad = [name for name in names if any(fragment in name.lower() for fragment in bad_fragments)]
    if bad:
        raise ValueError(f"identity/gallery-specific trainable parameter names found: {bad[:8]}")
