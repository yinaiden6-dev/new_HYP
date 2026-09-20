#!/usr/bin/env python3
"""Shared fail-closed I/O for the BAG-only completion executables."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_REL = "registry/dino_rcde_bag_full594_predecessor_status_repair_authority_v2_20260827.json"
AUTHORITY_SCHEMA = "rc_dino_rcde_bag_full594_predecessor_status_repair_authority_v2_20260827"
AUTHORITY_STATUS = "DINO_RCDE_BAG_FULL594_PREDECESSOR_STATUS_REPAIR_AUTHORIZED"


class BagCompletionIOError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise BagCompletionIOError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(path: Path, *, must_exist: bool = True, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    if must_exist:
        require(
            (value.is_file() if file else value.is_dir()) and not value.is_symlink(),
            f"input absent/unsafe: {value}",
        )
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def read_authority(path: Path) -> tuple[dict[str, Any], str]:
    authority_path = safe_path(path)
    require(authority_path == (RC_ROOT / AUTHORITY_REL).resolve(), "authority path drift")
    authority = read_json(authority_path)
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS,
        "BAG completion authority schema/status drift",
    )
    return authority, file_sha256(authority_path)


def binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(
        isinstance(item, Mapping) and set(item) >= {"path", "sha256"},
        f"authority binding absent: {name}",
    )
    path = safe_path(RC_ROOT / str(item["path"]))
    require(file_sha256(path) == item["sha256"], f"authority binding hash drift: {name}")
    return path


def external_binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(
        isinstance(item, Mapping) and set(item) >= {"path", "sha256"},
        f"external authority binding absent: {name}",
    )
    path = Path(str(item["path"])).resolve()
    require(path.is_file() and not path.is_symlink(), f"external binding absent/unsafe: {name}")
    require(file_sha256(path) == item["sha256"], f"external binding hash drift: {name}")
    return path


def logical_sha256(value: Mapping[str, Any]) -> str:
    from rc_aslo_xf.dino_rcde_bag_completion_v1 import canonical_sha256

    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), f"immutable JSON output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    output.chmod(0o444)


def atomic_torch(path: Path, value: object) -> None:
    output = safe_path(path, must_exist=False)
    require(not output.exists(), f"immutable tensor output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".partial.{os.getpid()}")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    output.chmod(0o444)


__all__ = [
    "AUTHORITY_REL",
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "BagCompletionIOError",
    "RC_ROOT",
    "atomic_json",
    "atomic_torch",
    "binding_path",
    "external_binding_path",
    "file_sha256",
    "logical_sha256",
    "read_authority",
    "read_json",
    "require",
    "safe_path",
]
