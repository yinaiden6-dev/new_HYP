"""Strict durable commit markers for the V124 Phase-A V3 retry-1 repair.

The historical Phase-A retry helper is intentionally scoped to its V2 retry
namespace.  V3 repair outputs therefore use this separate helper.  It accepts
only the three authority-declared retry-1 payload paths, preserves the existing
receipt and commit schemas, and never repairs or overwrites a partial family.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


REPAIR_NAMESPACE = (
    "dino_rcde_track_r_v124_c_col_p_phase_a_finalizer_repair_v3_retry1"
)
REPAIR_PHASE = "PHASE_A_V124_FINALIZER_REPAIR_V3_ONLY"
COMMIT_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_retry_commit_v2_20260824"
)
COMMIT_STATUS = "PHASE_A_RETRY_STAGE_COMMITTED"
RECEIPT_SCHEMA = (
    "rc_dino_rcde_track_r_v124_phase_a_finalizer_repair_receipt_v3_20260825"
)
RECEIPT_STATUS = "V124_PHASE_A_FINALIZER_REPAIR_RECEIPT_READY"


class RepairCommitV3Retry1Error(RuntimeError):
    """A repair path, payload, receipt, authority, or marker drifted."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise RepairCommitV3Retry1Error(message)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} SHA256 drift",
    )
    return value


@dataclass(frozen=True)
class _RoleSpec:
    role: str
    stage: str
    relative_payload: str
    payload_kind: str
    payload_schema: str
    payload_status: str


_ROLE_SPECS = {
    "REPAIR_NORMALIZED_AGGREGATE": _RoleSpec(
        role="repair_normalized_aggregate",
        stage="REPAIR_NORMALIZED_AGGREGATE",
        relative_payload="normalized/aggregate.pt",
        payload_kind="torch",
        payload_schema=(
            "rc_dino_rcde_track_r_v124_c_col_p_phase_a_independent_"
            "normalized_v3_20260825"
        ),
        payload_status=(
            "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_INDEPENDENT_"
            "NORMALIZED_V3_READY"
        ),
    ),
    "REPAIR_NORMALIZED_VALIDATION": _RoleSpec(
        role="repair_normalized_validation",
        stage="REPAIR_NORMALIZED_VALIDATION",
        relative_payload="normalized/validation.json",
        payload_kind="json",
        payload_schema=(
            "rc_dino_rcde_track_r_v124_c_col_p_phase_a_independent_"
            "normalized_validation_v3_20260825"
        ),
        payload_status=(
            "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_INDEPENDENT_"
            "NORMALIZED_V3_VALIDATION_PASS"
        ),
    ),
    "REPAIR_FINAL_RESULT": _RoleSpec(
        role="repair_final_result",
        stage="REPAIR_FINAL_RESULT",
        relative_payload="final/result.json",
        payload_kind="json",
        payload_schema=(
            "rc_dino_rcde_track_r_v124_c_col_p_phase_a_finalizer_"
            "repair_v3_result_20260825"
        ),
        payload_status=(
            "DINO_RCDE_TRACK_R_V124_C_COL_P_P_V2_COMPATIBILITY_E0_"
            "INDEPENDENT_VALIDATION_PASS"
        ),
    ),
}


def _strict_root(repair_root: Path) -> Path:
    raw = Path(repair_root)
    require(raw.name == REPAIR_NAMESPACE, "V3 retry-1 repair namespace drift")
    require(
        raw.exists() and raw.is_dir() and not raw.is_symlink(),
        "V3 retry-1 repair root absent or unsafe",
    )
    root = raw.resolve(strict=True)
    require(root.name == REPAIR_NAMESPACE, "resolved V3 retry-1 namespace drift")
    return root


def safe_repair_path(repair_root: Path, path: Path) -> Path:
    """Resolve one non-root path below the exact retry-1 namespace."""

    root = _strict_root(repair_root)
    raw = Path(path)
    require(".." not in raw.parts, "V3 retry-1 path traversal forbidden")
    require(not raw.is_symlink(), "V3 retry-1 path symlink forbidden")
    lexical = Path(os.path.abspath(raw))
    require(
        lexical != root and root in lexical.parents,
        "V3 retry-1 path escapes the repair namespace",
    )
    relative = lexical.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "V3 retry-1 path symlink forbidden")
    resolved = lexical.resolve(strict=False)
    require(
        resolved == lexical and root in resolved.parents,
        "V3 retry-1 resolved path escapes the repair namespace",
    )
    return resolved


def _immutable(path: Path, name: str) -> Path:
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_mode & 0o777 == 0o444,
        f"{name} absent, mutable or unsafe",
    )
    return path


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except Exception as error:  # pragma: no cover - decoder detail varies
        raise RepairCommitV3Retry1Error(f"{name} JSON drift") from error
    require(isinstance(value, dict), f"{name} root drift")
    return value


def _read_payload(path: Path, kind: str) -> Mapping[str, Any]:
    if kind == "json":
        return _read_json(path, "payload")
    require(kind == "torch", "repair payload kind drift")
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as error:  # pragma: no cover - torch detail varies
        raise RepairCommitV3Retry1Error("repair torch payload drift") from error
    require(isinstance(value, Mapping), "repair torch payload root drift")
    return value


def _validate_inputs(
    *,
    repair_root: Path,
    payload_path: Path,
    receipt_path: Path,
    marker_path: Path,
    authority_sha256: str,
    phase: str,
    stage: str,
    shard_ordinal: int | None,
    candidate_start: int | None,
    candidate_stop: int | None,
) -> tuple[Path, Path, Path, str, _RoleSpec]:
    root = _strict_root(repair_root)
    spec = _ROLE_SPECS.get(stage)
    require(spec is not None, "unsupported V3 retry-1 repair stage")
    require(
        phase == REPAIR_PHASE
        and shard_ordinal is None
        and candidate_start == 0
        and candidate_stop == 128,
        "V3 retry-1 phase or coordinates drift",
    )
    payload = safe_repair_path(root, payload_path)
    receipt = safe_repair_path(root, receipt_path)
    marker = safe_repair_path(root, marker_path)
    expected_payload = root / spec.relative_payload
    require(
        payload == expected_payload
        and receipt == payload.with_name(f"{payload.name}.receipt.json")
        and marker == payload.with_name(f"{payload.name}.commit.json"),
        "V3 retry-1 role path drift",
    )
    _immutable(payload, "repair payload")
    _immutable(receipt, "repair receipt")
    authority = _sha(authority_sha256, "repair authority")
    return payload, receipt, marker, authority, spec


def _expected_commit(
    *,
    root: Path,
    payload: Path,
    receipt: Path,
    authority_sha256: str,
    spec: _RoleSpec,
) -> dict[str, Any]:
    payload_value = _read_payload(payload, spec.payload_kind)
    payload_logical = _sha(payload_value.get("logical_sha256"), "payload logical")
    require(
        payload_value.get("schema_version") == spec.payload_schema
        and payload_value.get("status") == spec.payload_status
        and payload_value.get("authority_sha256") == authority_sha256
        and payload_value.get("scientific_GO_or_NO_GO") is None
        and payload_value.get("automatic_stage_advance") is False
        and payload_value.get("next_authorized_stage") is None,
        "V3 retry-1 payload schema/status/authority/boundary drift",
    )
    if spec.payload_kind == "json":
        require(
            payload_logical == logical_sha256(payload_value),
            "V3 retry-1 JSON payload logical drift",
        )

    receipt_value = _read_json(receipt, "receipt")
    required_receipt = {
        "schema_version",
        "status",
        "role",
        "payload_path",
        "payload_sha256",
        "payload_bytes",
        "payload_logical_sha256",
        "logical_sha256",
    }
    require(
        set(receipt_value) == required_receipt
        and receipt_value.get("schema_version") == RECEIPT_SCHEMA
        and receipt_value.get("status") == RECEIPT_STATUS
        and receipt_value.get("role") == spec.role
        and receipt_value.get("payload_path") == spec.relative_payload
        and receipt_value.get("payload_sha256") == file_sha256(payload)
        and receipt_value.get("payload_bytes") == payload.stat().st_size
        and receipt_value.get("payload_logical_sha256") == payload_logical
        and receipt_value.get("logical_sha256") == logical_sha256(receipt_value),
        "V3 retry-1 receipt schema/role/payload drift",
    )

    value: dict[str, Any] = {
        "schema_version": COMMIT_SCHEMA,
        "status": COMMIT_STATUS,
        "authority_sha256": authority_sha256,
        "phase": REPAIR_PHASE,
        "stage": spec.stage,
        "shard_ordinal": None,
        "candidate_start": 0,
        "candidate_stop": 128,
        "payload_path": payload.relative_to(root).as_posix(),
        "payload_bytes": payload.stat().st_size,
        "payload_sha256": file_sha256(payload),
        "receipt_path": receipt.relative_to(root).as_posix(),
        "receipt_bytes": receipt.stat().st_size,
        "receipt_sha256": file_sha256(receipt),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    return value


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "repair commit marker exists")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(0o444)
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def validate_commit(
    marker_path: Path,
    *,
    repair_root: Path,
    payload_path: Path,
    receipt_path: Path,
    authority_sha256: str,
    stage: str,
) -> Mapping[str, Any]:
    """Validate one exact retry-1 marker without writing or repairing."""

    payload, receipt, marker, authority, spec = _validate_inputs(
        repair_root=repair_root,
        payload_path=payload_path,
        receipt_path=receipt_path,
        marker_path=marker_path,
        authority_sha256=authority_sha256,
        phase=REPAIR_PHASE,
        stage=stage,
        shard_ordinal=None,
        candidate_start=0,
        candidate_stop=128,
    )
    _immutable(marker, "repair commit")
    root = _strict_root(repair_root)
    expected = _expected_commit(
        root=root,
        payload=payload,
        receipt=receipt,
        authority_sha256=authority,
        spec=spec,
    )
    observed = _read_json(marker, "commit")
    require(observed == expected, "existing V3 retry-1 commit/context drift")
    return observed


def commit_stage(
    *,
    resume_root: Path,
    payload_path: Path,
    receipt_path: Path,
    marker_path: Path,
    authority_sha256: str,
    phase: str,
    stage: str,
    shard_ordinal: int | None,
    candidate_start: int | None,
    candidate_stop: int | None,
) -> Mapping[str, Any]:
    """Create one marker, or exactly validate an already complete triple."""

    payload, receipt, marker, authority, spec = _validate_inputs(
        repair_root=resume_root,
        payload_path=payload_path,
        receipt_path=receipt_path,
        marker_path=marker_path,
        authority_sha256=authority_sha256,
        phase=phase,
        stage=stage,
        shard_ordinal=shard_ordinal,
        candidate_start=candidate_start,
        candidate_stop=candidate_stop,
    )
    root = _strict_root(resume_root)
    expected = _expected_commit(
        root=root,
        payload=payload,
        receipt=receipt,
        authority_sha256=authority,
        spec=spec,
    )
    if marker.exists() or marker.is_symlink():
        return validate_commit(
            marker,
            repair_root=root,
            payload_path=payload,
            receipt_path=receipt,
            authority_sha256=authority,
            stage=stage,
        )
    _exclusive_json(marker, expected)
    return validate_commit(
        marker,
        repair_root=root,
        payload_path=payload,
        receipt_path=receipt,
        authority_sha256=authority,
        stage=stage,
    )


__all__ = [
    "COMMIT_SCHEMA",
    "COMMIT_STATUS",
    "RECEIPT_SCHEMA",
    "RECEIPT_STATUS",
    "REPAIR_NAMESPACE",
    "REPAIR_PHASE",
    "RepairCommitV3Retry1Error",
    "commit_stage",
    "safe_repair_path",
    "validate_commit",
]
