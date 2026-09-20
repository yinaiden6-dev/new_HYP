"""Narrow permission repair for the historical C128 prejoin aggregate.

This module changes no payload, schema, hash, population, or output contract.
It only models a historical provenance fact: the single frozen redacted OOF
schedule was materialized with owner-write permissions.  That exact file may
be mode 0600 or 0644.  The fifty shard results and their fifty independent
validation results must remain exactly 0444.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import stat
from typing import Type


HISTORICAL_REDACTED_SCHEDULE = PurePosixPath(
    "results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_oof_schedule.json"
)
HISTORICAL_REDACTED_SCHEDULE_SCHEMA = (
    "rc_dino_rcde_r1_oof_redacted_schedule_v1_20260814"
)
HISTORICAL_REDACTED_SCHEDULE_BYTES = 1_281_283
HISTORICAL_REDACTED_SCHEDULE_SHA256 = (
    "3d226d89b32fe79b9b490e10cfaecb7b2e8a22dbcbc8e46cde20a79f129e548f"
)
HISTORICAL_REDACTED_SCHEDULE_LOGICAL_SHA256 = (
    "350201bf540272af1533e79543dafeba5b620a3a28ed05790f73e2c9cd11be09"
)
HISTORICAL_REDACTED_SCHEDULE_MODES = frozenset((0o600, 0o644))

SHARD_RESULT_PREFIX = PurePosixPath(
    "results/dino_rcde_h0_c128_target_free_prejoin_full_v1/shards"
)
SHARD_VALIDATION_PREFIX = PurePosixPath(
    "results/dino_rcde_h0_c128_target_free_prejoin_full_v1/validations"
)
AGGREGATE_RESULT = PurePosixPath(
    "results/dino_rcde_h0_c128_target_free_prejoin_aggregate_v1/result.json"
)
AGGREGATE_VALIDATION = PurePosixPath(
    "results/dino_rcde_h0_c128_target_free_prejoin_aggregate_validation_v1/result.json"
)
STRICT_ARTIFACT_MODE = 0o444


class AggregatePermissionV2Error(RuntimeError):
    """Raised when the narrow permission contract is violated."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> PurePosixPath:
    try:
        return PurePosixPath(path.relative_to(root).as_posix())
    except ValueError as error:
        raise AggregatePermissionV2Error("input escapes RC root") from error


def _is_numbered_artifact(
    relative: PurePosixPath,
    *,
    prefix: PurePosixPath,
    directory_prefix: str,
    filename: str,
) -> bool:
    parts = relative.parts
    base = prefix.parts
    if len(parts) != len(base) + 2 or parts[: len(base)] != base:
        return False
    ordinal = parts[-2]
    return (
        ordinal.startswith(directory_prefix)
        and len(ordinal) == len(directory_prefix) + 3
        and ordinal[len(directory_prefix) :].isdigit()
        and 0 <= int(ordinal[len(directory_prefix) :]) < 50
        and parts[-1] == filename
    )


def is_strict_0444_artifact(relative: PurePosixPath) -> bool:
    """Return whether *relative* is one of the exact aggregate artifacts."""

    return (
        relative in {AGGREGATE_RESULT, AGGREGATE_VALIDATION}
        or _is_numbered_artifact(
            relative,
            prefix=SHARD_RESULT_PREFIX,
            directory_prefix="shard_",
            filename="result.json",
        )
        or _is_numbered_artifact(
            relative,
            prefix=SHARD_VALIDATION_PREFIX,
            directory_prefix="shard_",
            filename="validation.json",
        )
    )


def validate_permission_mode(relative: PurePosixPath, mode: int) -> None:
    """Validate only the pre-registered permission exception and hard modes."""

    normalized = stat.S_IMODE(mode)
    if relative == HISTORICAL_REDACTED_SCHEDULE:
        if normalized not in HISTORICAL_REDACTED_SCHEDULE_MODES:
            raise AggregatePermissionV2Error(
                "historical redacted schedule mode must be exactly 0600 or 0644"
            )
        return
    if is_strict_0444_artifact(relative):
        if normalized != STRICT_ARTIFACT_MODE:
            raise AggregatePermissionV2Error(
                f"aggregate shard/result artifact must be exactly 0444: {relative}"
            )
        return
    if normalized & 0o222:
        raise AggregatePermissionV2Error(f"immutable input is mutable: {relative}")


def safe_file_v2(
    root: Path,
    path: Path,
    name: str,
    *,
    immutable: bool = True,
    error_type: Type[RuntimeError] = AggregatePermissionV2Error,
) -> Path:
    """Resolve an input under *root* and apply the V2 permission contract.

    Payload validation deliberately remains in the frozen V1 producer or the
    separately frozen V1 replay validator.  This helper only replaces their
    generic write-bit check.
    """

    root_value = root.resolve()
    original = path
    value = path.resolve()
    try:
        relative = _relative(root_value, value)
        if original.is_symlink() or not value.is_file():
            raise AggregatePermissionV2Error(f"{name} absent/unsafe")
        if any(
            token in part.lower()
            for part in value.parts
            for token in ("c8", "s8", "opened", "sealed")
        ):
            raise AggregatePermissionV2Error(f"{name} requests protected data")
        if immutable:
            validate_permission_mode(relative, value.stat().st_mode)
        return value
    except AggregatePermissionV2Error as error:
        raise error_type(str(error)) from error


def assert_frozen_legacy(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    """Fail closed if a V2 wrapper is pointed at a changed V1 implementation."""

    value = path.resolve()
    if (
        not value.is_file()
        or value.is_symlink()
        or value.stat().st_size != expected_bytes
        or file_sha256(value) != expected_sha256
    ):
        raise AggregatePermissionV2Error(f"frozen V1 implementation drift: {path}")


__all__ = [
    "AGGREGATE_RESULT",
    "AGGREGATE_VALIDATION",
    "AggregatePermissionV2Error",
    "HISTORICAL_REDACTED_SCHEDULE",
    "HISTORICAL_REDACTED_SCHEDULE_BYTES",
    "HISTORICAL_REDACTED_SCHEDULE_LOGICAL_SHA256",
    "HISTORICAL_REDACTED_SCHEDULE_MODES",
    "HISTORICAL_REDACTED_SCHEDULE_SCHEMA",
    "HISTORICAL_REDACTED_SCHEDULE_SHA256",
    "STRICT_ARTIFACT_MODE",
    "assert_frozen_legacy",
    "file_sha256",
    "is_strict_0444_artifact",
    "safe_file_v2",
    "validate_permission_mode",
]
