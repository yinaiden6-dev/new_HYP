#!/usr/bin/env python3
"""Shared fail-closed lineage checks for the Track-R V121 OOF stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class V121LineageError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise V121LineageError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def read_authority(
    authority_path: Path,
    *,
    root: Path,
    expected_relative: str,
    expected_schema: str,
    expected_status: str,
) -> tuple[dict[str, Any], str]:
    resolved_root = root.resolve()
    resolved = authority_path.resolve(strict=True)
    require(
        resolved == (resolved_root / expected_relative).resolve(strict=True)
        and resolved.is_file()
        and not resolved.is_symlink(),
        "V121 authority path drift",
    )
    require(
        (resolved.stat().st_mode & 0o777) == 0o444,
        "V121 authority is not immutable",
    )
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(
        isinstance(value, dict)
        and value.get("schema_version") == expected_schema
        and value.get("status") == expected_status
        and value.get("logical_sha256") == logical_sha256(value),
        "V121 authority envelope/logical drift",
    )
    return value, file_sha256(resolved)


def _bound_path(root: Path, binding: Mapping[str, Any]) -> Path:
    relative = binding.get("path")
    require(isinstance(relative, str), "runtime binding path absent")
    raw = Path(relative)
    require(
        not raw.is_absolute() and ".." not in raw.parts,
        f"unsafe runtime binding path: {relative}",
    )
    path = (root.resolve() / raw).resolve(strict=True)
    require(
        path.is_relative_to(root.resolve())
        and path.is_file()
        and not path.is_symlink()
        and binding.get("bytes") == path.stat().st_size
        and binding.get("sha256") == file_sha256(path),
        f"runtime binding drift: {relative}",
    )
    return path


def validate_runtime_bindings(
    authority: Mapping[str, Any],
    *,
    root: Path,
    required_names: Sequence[str],
) -> None:
    bindings = authority.get("bindings")
    require(isinstance(bindings, Mapping), "V121 authority bindings absent")
    for name in required_names:
        binding = bindings.get(name)
        require(isinstance(binding, Mapping), f"V121 binding absent: {name}")
        _bound_path(root, binding)

    closure = bindings.get("runtime_import_closure")
    require(isinstance(closure, Mapping), "V121 runtime import closure absent")
    rows = closure.get("rows")
    require(
        isinstance(rows, list)
        and closure.get("count") == len(rows)
        and closure.get("logical_sha256")
        == canonical_sha256({"rows": rows}),
        "V121 runtime import closure receipt drift",
    )
    observed_paths: list[str] = []
    for row in rows:
        require(isinstance(row, Mapping), "V121 import binding row drift")
        path = _bound_path(root, row)
        observed_paths.append(path.relative_to(root.resolve()).as_posix())
    require(
        observed_paths == sorted(observed_paths)
        and len(observed_paths) == len(set(observed_paths)),
        "V121 runtime import closure order/collision drift",
    )


def require_shard_source_authority(
    shard: Mapping[str, Any], expected_sha256: str, shard_ordinal: int
) -> None:
    source_bindings = shard.get("source_bindings")
    require(
        isinstance(source_bindings, Mapping)
        and source_bindings.get("authority_sha256") == expected_sha256,
        f"OOF shard{shard_ordinal} V121 source-authority SHA drift",
    )


__all__ = [
    "V121LineageError",
    "canonical_sha256",
    "file_sha256",
    "logical_sha256",
    "read_authority",
    "require_shard_source_authority",
    "validate_runtime_bindings",
]
