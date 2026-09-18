#!/usr/bin/env python3
"""Strict read-only loader for one sanitized-V2 S0 input shard.

The loader deliberately returns two disjoint object graphs:

* ``math`` contains only opaque keys, content-bound ColNomic tensors, grids,
  and canonical coordinates;
* ``io`` contains only image locators and decode metadata.

In particular, the broker's legacy ``physical_row`` is validated while the
file is parsed and is then discarded.  It can never reach an S0 mathematical
record.  This module does not import any historical RGH/P implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch


P_SCHEMA = "rc_lth_p_only_sanitized_train_input_shard_v2_20260906"
P_STATUS = "RC_LTH_P_ONLY_SANITIZED_TRAIN_INPUT_V2_SHARD_READY"
BROKER_SCHEMA = "rc_lth_p_only_sanitized_train_broker_shard_v2_20260906"
BROKER_STATUS = "RC_LTH_P_ONLY_SANITIZED_TRAIN_BROKER_V2_SHARD_READY"

SHARD_COUNT = 8
QUERY_COUNT = 4
CANDIDATE_COUNT = 128
TOKEN_DIMENSION = 128

HEX64 = re.compile(r"^[0-9a-f]{64}$")
OPAQUE_QUERY_KEY = re.compile(r"^q_[0-9a-f]{64}$")
OPAQUE_CANDIDATE_KEY = re.compile(r"^c_[0-9a-f]{64}$")

P_KEYS = frozenset(
    {
        "candidate_count_per_record",
        "coordinate_semantics",
        "provenance_digest_sha256",
        "record_count",
        "records",
        "reference_assets",
        "schema_version",
        "shard_count",
        "shard_index",
        "status",
    }
)
RECORD_KEYS = frozenset(
    {
        "candidate_keys",
        "candidate_set_sha256",
        "content_sha256",
        "grid_shape",
        "patch_coordinates",
        "patch_coordinates_sha256",
        "resource_key",
        "tokens",
        "tokens_sha256",
    }
)
ASSET_KEYS = frozenset(
    {
        "content_sha256",
        "grid_shape",
        "patch_coordinates",
        "patch_coordinates_sha256",
        "source_binding_sha256",
        "tokens",
        "tokens_sha256",
    }
)
BROKER_KEYS = frozenset(
    {
        "candidate_locator_map",
        "record_count",
        "schema_version",
        "shard_count",
        "shard_index",
        "source_seal",
        "status",
        "subject_locator_map",
    }
)
SUBJECT_LOCATOR_KEYS = frozenset(
    {"colnomic_frame", "content_sha256", "locator", "roma_decode_recipe"}
)
CANDIDATE_LOCATOR_KEYS = frozenset(
    {
        "colnomic_frame",
        "content_sha256",
        "locator",
        "physical_row",
        "roma_decode_recipe",
        "source_binding_sha256",
    }
)
SOURCE_SEAL_KEYS = frozenset(
    {
        "contract_authority_sha256",
        "e0_incident_sha256",
        "execution_authority_sha256",
        "reference_sidecar_receipt_sha256",
        "reference_sidecar_sha256",
        "reference_sidecar_validation_sha256",
        "source_payload_sha256",
        "source_receipt_sha256",
    }
)
COORDINATE_SEMANTICS = MappingProxyType(
    {
        "axis_order": "XY",
        "cell_center_formula": (
            "X_EQ_COLUMN_PLUS_HALF_OVER_WIDTH__Y_EQ_ROW_PLUS_HALF_OVER_HEIGHT"
        ),
        "grid_shape_order": "HEIGHT_WIDTH",
        "linearization": "ROW_MAJOR_I_EQ_ROW_TIMES_WIDTH_PLUS_COLUMN",
        "tensor_dtype": "torch.float64",
    }
)

# These fields must not occur anywhere in either serialized envelope.  The
# exact-key schemas below are the primary whitelist; this scan makes leakage
# failures explicit rather than reporting them as a generic schema mismatch.
FORBIDDEN_LEAK_FIELD_FRAGMENTS = (
    "target",
    "label",
    "identity",
    "rank",
    "score",
    "winner",
)


class SanitizedS0LoaderAbort(RuntimeError):
    """Fail-closed error with a stable machine-readable reason code."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SanitizedS0LoaderAbort(code)


@dataclass(frozen=True, slots=True)
class SanitizedS0TensorAsset:
    """Mathematical tensor asset with no locator or physical-row metadata."""

    content_sha256: str
    source_binding_sha256: str | None
    grid_shape: tuple[int, int]
    patch_coordinates: torch.Tensor
    patch_coordinates_sha256: str
    tokens: torch.Tensor
    tokens_sha256: str


@dataclass(frozen=True, slots=True)
class SanitizedS0MathRecord:
    """One anonymous query and its exact serialized opaque C128 axis."""

    resource_key: str
    content_sha256: str
    grid_shape: tuple[int, int]
    patch_coordinates: torch.Tensor
    patch_coordinates_sha256: str
    tokens: torch.Tensor
    tokens_sha256: str
    candidate_keys: tuple[str, ...]
    candidate_set_sha256: str


@dataclass(frozen=True, slots=True)
class SanitizedS0MathShard:
    """The only object graph that an S0 mathematical consumer should receive."""

    shard_index: int
    provenance_digest_sha256: str
    records: tuple[SanitizedS0MathRecord, ...]
    reference_assets: Mapping[str, SanitizedS0TensorAsset]


@dataclass(frozen=True, slots=True)
class SanitizedS0IOLocator:
    """Image I/O information intentionally separated from mathematical data."""

    locator: str
    content_sha256: str
    colnomic_frame: str
    roma_decode_recipe: str
    source_binding_sha256: str | None


@dataclass(frozen=True, slots=True)
class SanitizedS0IOBroker:
    """Broker-only lookup table; legacy physical rows are never retained."""

    subjects: Mapping[str, SanitizedS0IOLocator]
    candidates: Mapping[str, SanitizedS0IOLocator]

    def subject_locator(self, resource_key: str) -> SanitizedS0IOLocator:
        _require(OPAQUE_QUERY_KEY.fullmatch(resource_key) is not None, "QUERY_KEY_INVALID")
        try:
            return self.subjects[resource_key]
        except KeyError as exc:
            raise SanitizedS0LoaderAbort("QUERY_LOCATOR_ABSENT") from exc

    def candidate_locator(self, candidate_key: str) -> SanitizedS0IOLocator:
        _require(
            OPAQUE_CANDIDATE_KEY.fullmatch(candidate_key) is not None,
            "CANDIDATE_KEY_INVALID",
        )
        try:
            return self.candidates[candidate_key]
        except KeyError as exc:
            raise SanitizedS0LoaderAbort("CANDIDATE_LOCATOR_ABSENT") from exc


@dataclass(frozen=True, slots=True)
class SanitizedS0ShardBundle:
    """Validated shard with explicitly disjoint math and I/O views."""

    math: SanitizedS0MathShard
    io: SanitizedS0IOBroker
    p_input_sha256: str
    broker_sha256: str


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_coordinates(grid_shape: Sequence[int]) -> torch.Tensor:
    height, width = _validated_grid_shape(grid_shape)
    rows = torch.arange(height, dtype=torch.float64)[:, None].expand(height, width)
    columns = torch.arange(width, dtype=torch.float64)[None, :].expand(height, width)
    return (
        torch.stack(
            ((columns + 0.5) / width, (rows + 0.5) / height), dim=-1
        )
        .reshape(-1, 2)
        .contiguous()
    )


def opaque_query_key(
    content_sha256: str, tokens_sha256: str, grid_shape: Sequence[int]
) -> str:
    payload = {
        "content_sha256": content_sha256,
        "grid_shape": list(_validated_grid_shape(grid_shape)),
        "tokens_sha256": tokens_sha256,
    }
    return "q_" + hashlib.sha256(
        b"rc_lth_p_only_subject_key_v1\x00"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def opaque_candidate_key(
    content_sha256: str,
    tokens_sha256: str,
    grid_shape: Sequence[int],
    source_binding_sha256: str,
) -> str:
    payload = {
        "content_sha256": content_sha256,
        "grid_shape": list(_validated_grid_shape(grid_shape)),
        "source_binding_sha256": source_binding_sha256,
        "tokens_sha256": tokens_sha256,
    }
    return "c_" + hashlib.sha256(
        b"rc_lth_p_only_candidate_key_v1\x00"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _validated_grid_shape(value: Any) -> tuple[int, int]:
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        ),
        "GRID_SHAPE_INVALID",
    )
    return int(value[0]), int(value[1])


def _hex64(value: Any, code: str) -> str:
    _require(isinstance(value, str) and HEX64.fullmatch(value) is not None, code)
    return value


def _reject_leak_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(isinstance(key, str), "NONSTRING_FIELD_NAME")
            lowered = key.lower()
            _require(
                not any(term in lowered for term in FORBIDDEN_LEAK_FIELD_FRAGMENTS),
                "TARGET_RANK_SCORE_OR_WINNER_FIELD_PRESENT",
            )
            _reject_leak_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_leak_fields(child)


def _read_regular_readonly_file(path: Path) -> tuple[bytes, str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require(absolute.exists(), "INPUT_FILE_ABSENT")
    _require(not absolute.is_symlink(), "INPUT_SYMLINK_DENIED")
    before_lstat = os.lstat(absolute)
    _require(stat.S_ISREG(before_lstat.st_mode), "INPUT_NOT_REGULAR")
    _require(stat.S_IMODE(before_lstat.st_mode) == 0o444, "INPUT_NOT_MODE_0444")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise SanitizedS0LoaderAbort("INPUT_OPEN_FAILED") from exc
    try:
        before = os.fstat(descriptor)
        target = os.readlink(f"/proc/self/fd/{descriptor}")
        _require(
            os.path.normpath(target) == os.path.normpath(str(absolute)),
            "INPUT_FD_TARGET_MISMATCH",
        )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "INPUT_CHANGED_DURING_READ",
        )
    finally:
        os.close(descriptor)
    after_lstat = os.lstat(absolute)
    _require(
        (before_lstat.st_dev, before_lstat.st_ino, before_lstat.st_size, before_lstat.st_mtime_ns)
        == (after_lstat.st_dev, after_lstat.st_ino, after_lstat.st_size, after_lstat.st_mtime_ns),
        "INPUT_CHANGED_DURING_READ",
    )
    encoded = b"".join(chunks)
    return encoded, hashlib.sha256(encoded).hexdigest()


def _load_torch_dict(
    path: Path, *, expected_sha256: str | None
) -> tuple[dict[str, Any], str]:
    encoded, digest = _read_regular_readonly_file(path)
    if expected_sha256 is not None:
        _hex64(expected_sha256, "EXPECTED_FILE_SHA256_INVALID")
        _require(digest == expected_sha256, "INPUT_FILE_SHA256_MISMATCH")
    try:
        value = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SanitizedS0LoaderAbort("TORCH_ENVELOPE_LOAD_FAILED") from exc
    _require(isinstance(value, dict), "TORCH_ENVELOPE_NOT_DICT")
    return value, digest


def _validated_tensor_asset(
    value: Any, *, source_binding_required: bool
) -> SanitizedS0TensorAsset:
    expected_keys = ASSET_KEYS if source_binding_required else RECORD_KEYS - {
        "candidate_keys",
        "candidate_set_sha256",
        "resource_key",
    }
    _require(isinstance(value, dict) and set(value) == expected_keys, "TENSOR_ASSET_KEYS_INVALID")
    grid = _validated_grid_shape(value["grid_shape"])
    tokens = value["tokens"]
    tokens_sha = _hex64(value["tokens_sha256"], "TOKENS_SHA256_INVALID")
    _require(
        isinstance(tokens, torch.Tensor)
        and tokens.dtype == torch.float16
        and tokens.ndim == 2
        and tuple(tokens.shape) == (math.prod(grid), TOKEN_DIMENSION)
        and tokens.device.type == "cpu"
        and tokens.is_contiguous()
        and bool(torch.isfinite(tokens).all()),
        "TOKEN_GRID_OR_DTYPE_INVALID",
    )
    _require(tensor_sha256(tokens) == tokens_sha, "TOKEN_HASH_INVALID")

    coordinates = value["patch_coordinates"]
    coordinates_sha = _hex64(
        value["patch_coordinates_sha256"], "COORDINATES_SHA256_INVALID"
    )
    expected_coordinates = canonical_coordinates(grid)
    _require(
        isinstance(coordinates, torch.Tensor)
        and coordinates.dtype == torch.float64
        and tuple(coordinates.shape) == (math.prod(grid), 2)
        and coordinates.device.type == "cpu"
        and coordinates.is_contiguous()
        and bool(torch.isfinite(coordinates).all()),
        "COORDINATE_GRID_OR_DTYPE_INVALID",
    )
    _require(torch.equal(coordinates, expected_coordinates), "COORDINATE_VALUES_INVALID")
    _require(tensor_sha256(coordinates) == coordinates_sha, "COORDINATE_HASH_INVALID")

    content_sha = _hex64(value["content_sha256"], "CONTENT_SHA256_INVALID")
    source_binding: str | None = None
    if source_binding_required:
        source_binding = _hex64(
            value["source_binding_sha256"], "SOURCE_BINDING_SHA256_INVALID"
        )
    return SanitizedS0TensorAsset(
        content_sha256=content_sha,
        source_binding_sha256=source_binding,
        grid_shape=grid,
        patch_coordinates=coordinates.detach().clone().contiguous(),
        patch_coordinates_sha256=coordinates_sha,
        tokens=tokens.detach().clone().contiguous(),
        tokens_sha256=tokens_sha,
    )


def _validated_locator_string(value: Any) -> str:
    _require(
        isinstance(value, str)
        and value
        and "\x00" not in value
        and os.path.isabs(value),
        "BROKER_LOCATOR_INVALID",
    )
    return value


def load_sanitized_s0_shard(
    p_input_path: os.PathLike[str] | str,
    broker_path: os.PathLike[str] | str,
    *,
    expected_p_input_sha256: str | None = None,
    expected_broker_sha256: str | None = None,
) -> SanitizedS0ShardBundle:
    """Load and validate exactly one sanitized-V2 shard without asset I/O.

    ``candidate_keys`` are preserved byte-for-byte and position-for-position
    from each serialized record.  No sort, rank reconstruction, physical-row
    join, target join, or image decode occurs here.
    """

    p_path = Path(p_input_path)
    b_path = Path(broker_path)
    _require(p_path.name == "p_input.pt", "P_INPUT_FILENAME_INVALID")
    _require(b_path.name == "broker.pt", "BROKER_FILENAME_INVALID")
    _require(
        Path(os.path.abspath(os.fspath(p_path.parent)))
        == Path(os.path.abspath(os.fspath(b_path.parent))),
        "SHARD_FILE_DIRECTORY_MISMATCH",
    )
    p_input, p_digest = _load_torch_dict(
        p_path, expected_sha256=expected_p_input_sha256
    )
    broker, broker_digest = _load_torch_dict(
        b_path, expected_sha256=expected_broker_sha256
    )

    _reject_leak_fields(p_input)
    _reject_leak_fields(broker)
    _require(set(p_input) == P_KEYS, "P_INPUT_KEYS_INVALID")
    _require(set(broker) == BROKER_KEYS, "BROKER_KEYS_INVALID")
    _require(
        p_input["schema_version"] == P_SCHEMA
        and p_input["status"] == P_STATUS
        and broker["schema_version"] == BROKER_SCHEMA
        and broker["status"] == BROKER_STATUS,
        "SHARD_SCHEMA_OR_STATUS_INVALID",
    )
    shard_index = p_input["shard_index"]
    _require(
        isinstance(shard_index, int)
        and not isinstance(shard_index, bool)
        and shard_index in range(SHARD_COUNT)
        and broker["shard_index"] == shard_index
        and p_input["shard_count"] == broker["shard_count"] == SHARD_COUNT,
        "SHARD_IDENTITY_INVALID",
    )
    _require(
        p_input["record_count"] == broker["record_count"] == QUERY_COUNT
        and p_input["candidate_count_per_record"] == CANDIDATE_COUNT,
        "QUERY_OR_CANDIDATE_COUNT_INVALID",
    )
    _require(
        p_input["coordinate_semantics"] == dict(COORDINATE_SEMANTICS),
        "COORDINATE_SEMANTICS_INVALID",
    )
    source_seal = broker["source_seal"]
    _require(
        isinstance(source_seal, dict)
        and set(source_seal) == SOURCE_SEAL_KEYS
        and all(isinstance(value, str) and HEX64.fullmatch(value) for value in source_seal.values()),
        "SOURCE_SEAL_INVALID",
    )
    provenance = _hex64(
        p_input["provenance_digest_sha256"], "PROVENANCE_DIGEST_INVALID"
    )
    _require(provenance == canonical_sha256(source_seal), "SOURCE_SEAL_BINDING_INVALID")

    raw_records = p_input["records"]
    raw_assets = p_input["reference_assets"]
    subject_map = broker["subject_locator_map"]
    candidate_map = broker["candidate_locator_map"]
    _require(
        isinstance(raw_records, list) and len(raw_records) == QUERY_COUNT,
        "QUERY_RECORD_COUNT_INVALID",
    )
    _require(
        isinstance(raw_assets, dict)
        and isinstance(subject_map, dict)
        and isinstance(candidate_map, dict),
        "ASSET_OR_BROKER_MAP_INVALID",
    )

    math_records: list[SanitizedS0MathRecord] = []
    seen_queries: set[str] = set()
    referenced_candidates: set[str] = set()
    for raw_record in raw_records:
        _require(
            isinstance(raw_record, dict) and set(raw_record) == RECORD_KEYS,
            "QUERY_RECORD_KEYS_INVALID",
        )
        query_key = raw_record["resource_key"]
        _require(
            isinstance(query_key, str)
            and OPAQUE_QUERY_KEY.fullmatch(query_key) is not None
            and query_key not in seen_queries,
            "QUERY_KEY_INVALID",
        )
        seen_queries.add(query_key)
        candidate_keys_value = raw_record["candidate_keys"]
        _require(
            isinstance(candidate_keys_value, list)
            and len(candidate_keys_value) == CANDIDATE_COUNT
            and len(set(candidate_keys_value)) == CANDIDATE_COUNT
            and all(
                isinstance(key, str)
                and OPAQUE_CANDIDATE_KEY.fullmatch(key) is not None
                for key in candidate_keys_value
            ),
            "CANDIDATE_AXIS_INVALID",
        )
        # Deliberately do not sort this tuple: its serialized order is S0's
        # anonymous candidate axis.
        candidate_axis = tuple(candidate_keys_value)
        candidate_set_sha = _hex64(
            raw_record["candidate_set_sha256"], "CANDIDATE_SET_SHA256_INVALID"
        )
        _require(
            candidate_set_sha == canonical_sha256(list(candidate_axis)),
            "CANDIDATE_AXIS_HASH_INVALID",
        )
        referenced_candidates.update(candidate_axis)

        query_asset = _validated_tensor_asset(
            {
                key: raw_record[key]
                for key in RECORD_KEYS
                if key not in {"candidate_keys", "candidate_set_sha256", "resource_key"}
            },
            source_binding_required=False,
        )
        _require(
            query_key
            == opaque_query_key(
                query_asset.content_sha256,
                query_asset.tokens_sha256,
                query_asset.grid_shape,
            ),
            "QUERY_KEY_CONTENT_BINDING_INVALID",
        )
        math_records.append(
            SanitizedS0MathRecord(
                resource_key=query_key,
                content_sha256=query_asset.content_sha256,
                grid_shape=query_asset.grid_shape,
                patch_coordinates=query_asset.patch_coordinates,
                patch_coordinates_sha256=query_asset.patch_coordinates_sha256,
                tokens=query_asset.tokens,
                tokens_sha256=query_asset.tokens_sha256,
                candidate_keys=candidate_axis,
                candidate_set_sha256=candidate_set_sha,
            )
        )

    _require(set(raw_assets) == referenced_candidates, "REFERENCE_ASSET_COVERAGE_INVALID")
    _require(set(subject_map) == seen_queries, "SUBJECT_BROKER_COVERAGE_INVALID")
    _require(set(candidate_map) == referenced_candidates, "CANDIDATE_BROKER_COVERAGE_INVALID")

    math_assets: dict[str, SanitizedS0TensorAsset] = {}
    for candidate_key in referenced_candidates:
        _require(
            isinstance(candidate_key, str)
            and OPAQUE_CANDIDATE_KEY.fullmatch(candidate_key) is not None,
            "REFERENCE_ASSET_KEY_INVALID",
        )
        asset = _validated_tensor_asset(
            raw_assets[candidate_key], source_binding_required=True
        )
        assert asset.source_binding_sha256 is not None
        _require(
            candidate_key
            == opaque_candidate_key(
                asset.content_sha256,
                asset.tokens_sha256,
                asset.grid_shape,
                asset.source_binding_sha256,
            ),
            "CANDIDATE_KEY_CONTENT_BINDING_INVALID",
        )
        math_assets[candidate_key] = asset

    io_subjects: dict[str, SanitizedS0IOLocator] = {}
    for query_key in seen_queries:
        item = subject_map[query_key]
        _require(
            isinstance(item, dict) and set(item) == SUBJECT_LOCATOR_KEYS,
            "SUBJECT_LOCATOR_KEYS_INVALID",
        )
        content_sha = _hex64(item["content_sha256"], "BROKER_CONTENT_SHA256_INVALID")
        record = next(value for value in math_records if value.resource_key == query_key)
        _require(content_sha == record.content_sha256, "SUBJECT_CONTENT_BINDING_INVALID")
        _require(
            item["colnomic_frame"]
            in {"DECODED_RAW_BEFORE_EXIF", "EXIF_ORIENTED_BEFORE_RESIZE"}
            and item["roma_decode_recipe"] == "PIL_EXIF_ORIENTED_RGB_V1",
            "SUBJECT_DECODE_CONTRACT_INVALID",
        )
        io_subjects[query_key] = SanitizedS0IOLocator(
            locator=_validated_locator_string(item["locator"]),
            content_sha256=content_sha,
            colnomic_frame=item["colnomic_frame"],
            roma_decode_recipe=item["roma_decode_recipe"],
            source_binding_sha256=None,
        )

    io_candidates: dict[str, SanitizedS0IOLocator] = {}
    physical_rows: set[int] = set()
    for candidate_key in referenced_candidates:
        item = candidate_map[candidate_key]
        _require(
            isinstance(item, dict) and set(item) == CANDIDATE_LOCATOR_KEYS,
            "CANDIDATE_LOCATOR_KEYS_INVALID",
        )
        physical_row = item["physical_row"]
        _require(
            isinstance(physical_row, int)
            and not isinstance(physical_row, bool)
            and physical_row >= 0
            and physical_row not in physical_rows,
            "BROKER_PHYSICAL_ROW_INVALID",
        )
        physical_rows.add(physical_row)
        content_sha = _hex64(item["content_sha256"], "BROKER_CONTENT_SHA256_INVALID")
        source_binding = _hex64(
            item["source_binding_sha256"], "BROKER_SOURCE_BINDING_SHA256_INVALID"
        )
        asset = math_assets[candidate_key]
        _require(
            content_sha == asset.content_sha256
            and source_binding == asset.source_binding_sha256,
            "CANDIDATE_CONTENT_BINDING_INVALID",
        )
        _require(
            item["colnomic_frame"] == "DECODED_RAW_BEFORE_EXIF"
            and item["roma_decode_recipe"] == "PIL_EXIF_ORIENTED_RGB_V1",
            "CANDIDATE_DECODE_CONTRACT_INVALID",
        )
        # physical_row is intentionally not copied into this returned object.
        io_candidates[candidate_key] = SanitizedS0IOLocator(
            locator=_validated_locator_string(item["locator"]),
            content_sha256=content_sha,
            colnomic_frame=item["colnomic_frame"],
            roma_decode_recipe=item["roma_decode_recipe"],
            source_binding_sha256=source_binding,
        )

    return SanitizedS0ShardBundle(
        math=SanitizedS0MathShard(
            shard_index=shard_index,
            provenance_digest_sha256=provenance,
            records=tuple(math_records),
            reference_assets=MappingProxyType(math_assets),
        ),
        io=SanitizedS0IOBroker(
            subjects=MappingProxyType(io_subjects),
            candidates=MappingProxyType(io_candidates),
        ),
        p_input_sha256=p_digest,
        broker_sha256=broker_digest,
    )


__all__ = [
    "CANDIDATE_COUNT",
    "QUERY_COUNT",
    "SanitizedS0IOBroker",
    "SanitizedS0IOLocator",
    "SanitizedS0LoaderAbort",
    "SanitizedS0MathRecord",
    "SanitizedS0MathShard",
    "SanitizedS0ShardBundle",
    "SanitizedS0TensorAsset",
    "canonical_coordinates",
    "canonical_sha256",
    "load_sanitized_s0_shard",
    "opaque_candidate_key",
    "opaque_query_key",
    "tensor_sha256",
]
