"""Immutable redacted query-token cache for the L0-V2 target-free path.

The historical query shards are mixed metadata/tensor dictionaries.  They are
allowed only at this *one-way source-redaction boundary*: the materializer
copies the two frozen tensors and non-target provenance fields into an
append-only cache.  Target-free C0, training forward and heldout scoring then
read only this cache; they never deserialize a legacy query shard.

The final cache index is deliberately unavailable until every one of the 987
ledger records has been independently materialized.  This prevents a partial
cache from silently changing the population seen by a target-free runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


ARTIFACT_VERSION = "l0_v2_redacted_query_token_artifact_v1"
INDEX_VERSION = "l0_v2_redacted_query_token_cache_v1"
SEGMENT_RECEIPT_VERSION = "l0_v2_redacted_query_token_cache_segment_v2"
VALIDATION_RECEIPT_VERSION = "l0_v2_redacted_query_token_cache_validation_v2"
INDEX_NAME = "index_v1.json"
ARTIFACT_KEYS = {
    "version",
    "query_ordinal",
    "query_id",
    "path",
    "grid_h",
    "grid_w",
    "source_image_sha256",
    "source_shard_sha256",
    "image_tokens",
    "template_tokens",
    "image_tokens_sha256",
    "template_tokens_sha256",
    "logical_sha256",
}
ARTIFACT_LOGICAL_KEYS = ARTIFACT_KEYS - {"image_tokens", "template_tokens", "logical_sha256"}
INDEX_KEYS = {
    "version",
    "query_count",
    "query_ledger_logical_sha256",
    "query_ledger_sha256",
    "records",
    "target_bearing_fields_accessed_during_redaction",
    "contains_target_bearing_fields",
    "legacy_query_shard_redaction_reads",
    "target_free_runtime_legacy_query_shard_reads",
    "target_free_runtime_target_bearing_field_reads",
    "opened_runtime_read_count",
    "sealed_runtime_read_count",
    "home_files_modified",
    "logical_sha256",
}
INDEX_RECORD_KEYS = {
    "query_ordinal",
    "query_id",
    "artifact",
    "artifact_file_sha256",
    "artifact_logical_sha256",
    "image_tokens_sha256",
    "template_tokens_sha256",
}
SEGMENT_KEYS = {
    "version",
    "start",
    "stop",
    "query_ledger_logical_sha256",
    "query_ledger_sha256",
    "records",
    "target_bearing_fields_accessed_during_redaction",
    "legacy_query_shard_redaction_reads",
    "target_free_runtime_legacy_query_shard_reads",
    "opened_runtime_read_count",
    "sealed_runtime_read_count",
    "home_files_modified",
    "logical_sha256",
}


class L0RedactedQueryCacheError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype, shape and exact CPU tensor bytes without lossy conversion."""

    if not isinstance(tensor, torch.Tensor):
        raise L0RedactedQueryCacheError("redacted query cache value is not a tensor")
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cache_index_path(cache_root: Path) -> Path:
    return cache_root / INDEX_NAME


def artifact_path(cache_root: Path, ordinal: int) -> Path:
    return cache_root / "rows" / f"{ordinal:04d}.pt"


def segment_receipt_path(cache_root: Path, start: int, stop: int) -> Path:
    return cache_root / "receipts" / f"segment_{start:04d}_{stop:04d}_v2.json"


def validation_receipt_path(cache_root: Path, start: int, stop: int) -> Path:
    return cache_root / "validation" / f"segment_{start:04d}_{stop:04d}_v2.json"


def full_validation_receipt_path(cache_root: Path) -> Path:
    return cache_root / "validation" / "validation_complete.json"


def _logical(payload: Mapping[str, Any], *, excluded: Iterable[str] = ("logical_sha256",)) -> str:
    return canonical_sha256({key: payload[key] for key in payload if key not in set(excluded)})


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create exactly once; a pre-existing different receipt is a hard error."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise L0RedactedQueryCacheError(f"immutable cache receipt drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text() != rendered:
            raise L0RedactedQueryCacheError(f"concurrent immutable cache receipt drift: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_torch(path: Path, payload: Mapping[str, Any]) -> None:
    """Create exactly once.  Existing artifacts are validated by callers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _spec_fields(spec: Any) -> dict[str, Any]:
    """Return the only provenance fields allowed in a redacted artifact."""

    return {
        "query_ordinal": int(spec.query_ordinal),
        "query_id": str(spec.query_id),
        "path": str(spec.path),
        "grid_h": int(spec.grid_h),
        "grid_w": int(spec.grid_w),
        "source_image_sha256": str(spec.source_image_sha256),
        "source_shard_sha256": str(spec.source_shard_sha256),
    }


def _expected_tensor_shapes(spec: Any) -> tuple[tuple[int, int], int]:
    return (int(spec.grid_h) * int(spec.grid_w), 128), 128


def make_artifact_payload(
    spec: Any,
    *,
    image_tokens: torch.Tensor,
    template_tokens: torch.Tensor,
) -> dict[str, Any]:
    """Build a redacted cache artifact after source-redaction validation."""

    image = image_tokens.detach().cpu().to(torch.float16).contiguous()
    template = template_tokens.detach().cpu().to(torch.float16).contiguous()
    expected_image, dimension = _expected_tensor_shapes(spec)
    if (
        tuple(image.shape) != expected_image
        or not image.is_floating_point()
        or template.ndim != 2
        or int(template.shape[1]) != dimension
        or not template.is_floating_point()
        or not bool(torch.isfinite(image).all())
        or not bool(torch.isfinite(template).all())
    ):
        raise L0RedactedQueryCacheError("legacy source tensors fail redacted cache schema")
    payload: dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        **_spec_fields(spec),
        "image_tokens": image,
        "template_tokens": template,
        "image_tokens_sha256": tensor_sha256(image),
        "template_tokens_sha256": tensor_sha256(template),
    }
    payload["logical_sha256"] = canonical_sha256(
        {key: payload[key] for key in ARTIFACT_LOGICAL_KEYS}
    )
    return payload


def _validate_artifact_payload(payload: Any, spec: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != ARTIFACT_KEYS:
        raise L0RedactedQueryCacheError("redacted query artifact schema drift")
    if payload.get("version") != ARTIFACT_VERSION:
        raise L0RedactedQueryCacheError("redacted query artifact version drift")
    expected_fields = _spec_fields(spec)
    if any(payload.get(key) != value for key, value in expected_fields.items()):
        raise L0RedactedQueryCacheError("redacted query artifact provenance drift")
    image, template = payload["image_tokens"], payload["template_tokens"]
    expected_image, dimension = _expected_tensor_shapes(spec)
    if (
        not isinstance(image, torch.Tensor)
        or tuple(image.shape) != expected_image
        or image.dtype != torch.float16
        or not isinstance(template, torch.Tensor)
        or template.ndim != 2
        or int(template.shape[1]) != dimension
        or template.dtype != torch.float16
        or not bool(torch.isfinite(image).all())
        or not bool(torch.isfinite(template).all())
        or payload.get("image_tokens_sha256") != tensor_sha256(image)
        or payload.get("template_tokens_sha256") != tensor_sha256(template)
        or payload.get("logical_sha256")
        != canonical_sha256({key: payload[key] for key in ARTIFACT_LOGICAL_KEYS})
    ):
        raise L0RedactedQueryCacheError("redacted query artifact tensor/hash drift")
    return payload


def materialize_artifact_from_legacy_source(spec: Any, *, cache_root: Path) -> dict[str, Any]:
    """One-way redaction boundary.  No target-free runtime calls this function."""

    if not spec.path.is_file() or not spec.shard.is_file():
        raise L0RedactedQueryCacheError("legacy query source is missing at redaction boundary")
    if sha256_file(spec.path) != spec.source_image_sha256 or sha256_file(spec.shard) != spec.source_shard_sha256:
        raise L0RedactedQueryCacheError("legacy query source hash drift at redaction boundary")
    legacy = torch.load(spec.shard, map_location="cpu", weights_only=True)
    if not isinstance(legacy, dict):
        raise L0RedactedQueryCacheError("legacy query source is not a tensor dictionary")
    try:
        image = legacy["image_tokens"]
        template = legacy["template_tokens"]
        path = str(legacy["path"])
        grid_h, grid_w = int(legacy["grid_h"]), int(legacy["grid_w"])
    except (KeyError, TypeError, ValueError) as exc:
        raise L0RedactedQueryCacheError("legacy query source non-target tensor schema drift") from exc
    # The legacy dictionary may physically contain labels; do not serialize it
    # or inspect any field other than the fixed source-redaction whitelist.
    if path != str(spec.path) or grid_h != int(spec.grid_h) or grid_w != int(spec.grid_w):
        raise L0RedactedQueryCacheError("legacy query source path/grid drift at redaction boundary")
    payload = make_artifact_payload(spec, image_tokens=image, template_tokens=template)
    del legacy
    destination = artifact_path(cache_root, int(spec.query_ordinal))
    _immutable_torch(destination, payload)
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    checked = _validate_artifact_payload(loaded, spec)
    return artifact_record(cache_root, spec, checked)


def artifact_record(cache_root: Path, spec: Any, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = artifact_path(cache_root, int(spec.query_ordinal))
    if not path.is_file():
        raise L0RedactedQueryCacheError("redacted query artifact is missing")
    if payload is None:
        payload = _validate_artifact_payload(torch.load(path, map_location="cpu", weights_only=True), spec)
    else:
        payload = _validate_artifact_payload(dict(payload), spec)
    return {
        "query_ordinal": int(spec.query_ordinal),
        "query_id": str(spec.query_id),
        "artifact": str(path.relative_to(cache_root)),
        "artifact_file_sha256": sha256_file(path),
        "artifact_logical_sha256": str(payload["logical_sha256"]),
        "image_tokens_sha256": str(payload["image_tokens_sha256"]),
        "template_tokens_sha256": str(payload["template_tokens_sha256"]),
    }


def _validate_index(index: Any, specs: list[Any], *, cache_root: Path, query_ledger_logical_sha256: str, query_ledger_sha256: str) -> dict[str, Any]:
    if not isinstance(index, dict) or set(index) != INDEX_KEYS:
        raise L0RedactedQueryCacheError("redacted query cache index schema drift")
    if (
        index.get("version") != INDEX_VERSION
        or index.get("query_count") != 987
        or index.get("query_ledger_logical_sha256") != query_ledger_logical_sha256
        or index.get("query_ledger_sha256") != query_ledger_sha256
        or index.get("target_bearing_fields_accessed_during_redaction") is not False
        or index.get("contains_target_bearing_fields") is not False
        or index.get("legacy_query_shard_redaction_reads") != 987
        or index.get("target_free_runtime_legacy_query_shard_reads") != 0
        or index.get("target_free_runtime_target_bearing_field_reads") != 0
        or index.get("opened_runtime_read_count") != 0
        or index.get("sealed_runtime_read_count") != 0
        or index.get("home_files_modified") != 0
        or index.get("logical_sha256") != _logical(index)
        or not isinstance(index.get("records"), list)
        or len(index["records"]) != 987
    ):
        raise L0RedactedQueryCacheError("redacted query cache index receipt drift")
    expected_ordinals = list(range(987))
    if [record.get("query_ordinal") if isinstance(record, dict) else None for record in index["records"]] != expected_ordinals:
        raise L0RedactedQueryCacheError("redacted query cache index order drift")
    for record, spec in zip(index["records"], specs, strict=True):
        if not isinstance(record, dict) or set(record) != INDEX_RECORD_KEYS:
            raise L0RedactedQueryCacheError("redacted query cache index record schema drift")
        if int(record["query_ordinal"]) != int(spec.query_ordinal) or record["query_id"] != str(spec.query_id):
            raise L0RedactedQueryCacheError("redacted query cache index query binding drift")
        expected_relative = str(artifact_path(cache_root, int(spec.query_ordinal)).relative_to(cache_root))
        if record["artifact"] != expected_relative:
            raise L0RedactedQueryCacheError("redacted query cache index artifact path drift")
        for name in ("artifact_file_sha256", "artifact_logical_sha256", "image_tokens_sha256", "template_tokens_sha256"):
            if not isinstance(record.get(name), str) or len(str(record[name])) != 64:
                raise L0RedactedQueryCacheError("redacted query cache index hash schema drift")
    return index


def build_final_index(
    specs: list[Any],
    *,
    cache_root: Path,
    query_ledger_logical_sha256: str,
    query_ledger_sha256: str,
) -> dict[str, Any]:
    if len(specs) != 987 or [int(spec.query_ordinal) for spec in specs] != list(range(987)):
        raise L0RedactedQueryCacheError("redacted cache finalization requires all 987 canonical ledger records")
    records = [artifact_record(cache_root, spec) for spec in specs]
    payload: dict[str, Any] = {
        "version": INDEX_VERSION,
        "query_count": 987,
        "query_ledger_logical_sha256": query_ledger_logical_sha256,
        "query_ledger_sha256": query_ledger_sha256,
        "records": records,
        "target_bearing_fields_accessed_during_redaction": False,
        "contains_target_bearing_fields": False,
        "legacy_query_shard_redaction_reads": 987,
        "target_free_runtime_legacy_query_shard_reads": 0,
        "target_free_runtime_target_bearing_field_reads": 0,
        "opened_runtime_read_count": 0,
        "sealed_runtime_read_count": 0,
        "home_files_modified": 0,
    }
    payload["logical_sha256"] = _logical(payload)
    destination = cache_index_path(cache_root)
    write_immutable_json(destination, payload)
    return _validate_index(
        json.loads(destination.read_text()),
        specs,
        cache_root=cache_root,
        query_ledger_logical_sha256=query_ledger_logical_sha256,
        query_ledger_sha256=query_ledger_sha256,
    )


def load_redacted_query_artifact(
    spec: Any,
    *,
    specs: list[Any],
    cache_root: Path,
    query_ledger_logical_sha256: str,
    query_ledger_sha256: str,
) -> dict[str, Any]:
    """Load a target-free query only from the finalized redacted cache."""

    index_path = cache_index_path(cache_root)
    if not index_path.is_file():
        raise L0RedactedQueryCacheError(
            "finalized redacted query-token cache is missing; target-free runtime is fail-closed"
        )
    index = _validate_index(
        json.loads(index_path.read_text()),
        specs,
        cache_root=cache_root,
        query_ledger_logical_sha256=query_ledger_logical_sha256,
        query_ledger_sha256=query_ledger_sha256,
    )
    ordinal = int(spec.query_ordinal)
    if ordinal not in range(987) or specs[ordinal].query_id != spec.query_id:
        raise L0RedactedQueryCacheError("redacted query runtime spec is not canonical ledger record")
    record = index["records"][ordinal]
    path = cache_root / str(record["artifact"])
    if not path.is_file() or sha256_file(path) != record["artifact_file_sha256"]:
        raise L0RedactedQueryCacheError("redacted query cache artifact file hash drift")
    payload = _validate_artifact_payload(torch.load(path, map_location="cpu", weights_only=True), spec)
    if (
        payload["logical_sha256"] != record["artifact_logical_sha256"]
        or payload["image_tokens_sha256"] != record["image_tokens_sha256"]
        or payload["template_tokens_sha256"] != record["template_tokens_sha256"]
    ):
        raise L0RedactedQueryCacheError("redacted query cache index-to-artifact hash drift")
    return payload


def validate_final_cache(
    specs: list[Any],
    *,
    cache_root: Path,
    query_ledger_logical_sha256: str,
    query_ledger_sha256: str,
) -> dict[str, Any]:
    """Independently revalidate the finalized cache without opening legacy shards."""

    if len(specs) != 987 or [int(spec.query_ordinal) for spec in specs] != list(range(987)):
        raise L0RedactedQueryCacheError("final redacted cache validation requires all 987 ledger records")
    index_path = cache_index_path(cache_root)
    if not index_path.is_file():
        raise L0RedactedQueryCacheError("finalized redacted query-token cache index is missing")
    index = _validate_index(
        json.loads(index_path.read_text()),
        specs,
        cache_root=cache_root,
        query_ledger_logical_sha256=query_ledger_logical_sha256,
        query_ledger_sha256=query_ledger_sha256,
    )
    actual = [artifact_record(cache_root, spec) for spec in specs]
    if actual != index["records"]:
        raise L0RedactedQueryCacheError("final redacted cache index-to-artifact coverage drift")
    return index


def make_segment_receipt(
    records: list[dict[str, Any]],
    *,
    start: int,
    stop: int,
    query_ledger_logical_sha256: str,
    query_ledger_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": SEGMENT_RECEIPT_VERSION,
        "start": start,
        "stop": stop,
        "query_ledger_logical_sha256": query_ledger_logical_sha256,
        "query_ledger_sha256": query_ledger_sha256,
        "records": records,
        "target_bearing_fields_accessed_during_redaction": False,
        "legacy_query_shard_redaction_reads": stop - start,
        "target_free_runtime_legacy_query_shard_reads": 0,
        "opened_runtime_read_count": 0,
        "sealed_runtime_read_count": 0,
        "home_files_modified": 0,
    }
    payload["logical_sha256"] = _logical(payload)
    if set(payload) != SEGMENT_KEYS:
        raise AssertionError("internal segment receipt schema drift")
    return payload


def validate_segment_receipt(
    receipt: Any,
    specs: list[Any],
    *,
    cache_root: Path,
    start: int,
    stop: int,
    query_ledger_logical_sha256: str,
    query_ledger_sha256: str,
) -> dict[str, Any]:
    if not isinstance(receipt, dict) or set(receipt) != SEGMENT_KEYS:
        raise L0RedactedQueryCacheError("redacted query segment receipt schema drift")
    if (
        receipt.get("version") != SEGMENT_RECEIPT_VERSION
        or receipt.get("start") != start
        or receipt.get("stop") != stop
        or receipt.get("query_ledger_logical_sha256") != query_ledger_logical_sha256
        or receipt.get("query_ledger_sha256") != query_ledger_sha256
        or receipt.get("target_bearing_fields_accessed_during_redaction") is not False
        or receipt.get("legacy_query_shard_redaction_reads") != stop - start
        or receipt.get("target_free_runtime_legacy_query_shard_reads") != 0
        or receipt.get("opened_runtime_read_count") != 0
        or receipt.get("sealed_runtime_read_count") != 0
        or receipt.get("home_files_modified") != 0
        or receipt.get("logical_sha256") != _logical(receipt)
        or not isinstance(receipt.get("records"), list)
        or len(receipt["records"]) != stop - start
    ):
        raise L0RedactedQueryCacheError("redacted query segment receipt drift")
    actual = [artifact_record(cache_root, spec) for spec in specs[start:stop]]
    if actual != receipt["records"]:
        raise L0RedactedQueryCacheError("redacted query segment artifact receipt drift")
    return receipt
