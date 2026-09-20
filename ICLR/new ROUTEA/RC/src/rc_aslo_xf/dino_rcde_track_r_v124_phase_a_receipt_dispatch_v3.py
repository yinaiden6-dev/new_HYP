"""Read-only receipt-family dispatch for the V124 Phase-A finalizer repair.

This module deliberately does not import any producer or reducer helper.  Some
historical producer helpers repair an interrupted three-file family by writing
the missing receipt or commit marker.  A finalizer validator must never do
that: an incomplete family is evidence of an incomplete commit and fails
closed.

Dispatch is selected by two caller-supplied values, ``family`` and ``role``.
Neither a filename nor a directory name is used to infer either value.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


PRODUCTION_DURABLE_V2 = "production_durable_v2"
INDEPENDENT_V2 = "independent_v2"
REPAIR_V3 = "repair_v3"

PRODUCTION_PHASE_A_ARTIFACT = "production_phase_a_artifact"
PRODUCTION_CONTROL_AGGREGATE = "production_control_aggregate"
INDEPENDENT_AGGREGATE = "independent_aggregate"
INDEPENDENT_SOURCE_SHARD = "independent_source_shard"
REPAIR_NORMALIZED_AGGREGATE = "repair_normalized_aggregate"
REPAIR_NORMALIZED_VALIDATION = "repair_normalized_validation"
REPAIR_FINAL_RESULT = "repair_final_result"

PRODUCTION_RECEIPT_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_durable_receipt_v2_20260824"
)
PRODUCTION_RECEIPT_STATUS = (
    "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_DURABLE_COMMIT"
)
INDEPENDENT_RECEIPT_SCHEMA = "rc_v124_ccol_phase_a_v2_independent_receipt_v1"
INDEPENDENT_RECEIPT_STATUS = "INDEPENDENT_STAGE_RECEIPT_READY"

REPAIR_RECEIPT_SCHEMA_V3 = (
    "rc_dino_rcde_track_r_v124_phase_a_finalizer_repair_receipt_v3_20260825"
)
REPAIR_RECEIPT_STATUS_V3 = "V124_PHASE_A_FINALIZER_REPAIR_RECEIPT_READY"
REPAIR_PHASE_V3 = "PHASE_A_V124_FINALIZER_REPAIR_V3_ONLY"

COMMIT_SCHEMA_V2 = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_retry_commit_v2_20260824"
)
COMMIT_STATUS_V2 = "PHASE_A_RETRY_STAGE_COMMITTED"


class ReceiptDispatchV3Error(RuntimeError):
    """A receipt family is incomplete, unsafe, cross-wired, or drifted."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise ReceiptDispatchV3Error(message)


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
    family: str
    role: str
    payload_kind: str
    receipt_schema: str
    receipt_status: str
    commit_phase: str
    commit_stage: str
    payload_schema: str | None = None
    payload_status: str | None = None


@dataclass(frozen=True)
class ValidatedReceiptFamilyV3:
    family: str
    role: str
    payload_path: Path
    receipt_path: Path
    commit_path: Path
    payload_sha256: str
    payload_bytes: int
    payload_logical_sha256: str
    receipt_sha256: str
    commit_logical_sha256: str
    authority_sha256: str
    resume_root: Path
    receipt: Mapping[str, Any]
    commit: Mapping[str, Any]


_ROLE_SPECS = {
    (PRODUCTION_DURABLE_V2, PRODUCTION_PHASE_A_ARTIFACT): _RoleSpec(
        family=PRODUCTION_DURABLE_V2,
        role=PRODUCTION_PHASE_A_ARTIFACT,
        payload_kind="torch",
        receipt_schema=PRODUCTION_RECEIPT_SCHEMA,
        receipt_status=PRODUCTION_RECEIPT_STATUS,
        commit_phase="PHASE_A_P_V2_LOCK_REBUILD_ONLY",
        commit_stage="FINAL_ARTIFACT",
        payload_schema=(
            "rc_dino_rcde_track_r_v124_c_col_p_p_v2_phase_a_artifact_v1_20260824"
        ),
        payload_status="DINO_RCDE_TRACK_R_V124_C_COL_P_P_V2_PHASE_A_READY",
    ),
    (PRODUCTION_DURABLE_V2, PRODUCTION_CONTROL_AGGREGATE): _RoleSpec(
        family=PRODUCTION_DURABLE_V2,
        role=PRODUCTION_CONTROL_AGGREGATE,
        payload_kind="json",
        receipt_schema=PRODUCTION_RECEIPT_SCHEMA,
        receipt_status=PRODUCTION_RECEIPT_STATUS,
        commit_phase="PHASE_A_P_V2_LOCK_REBUILD_ONLY",
        commit_stage="CONTROL_AGGREGATE",
        payload_schema=(
            "rc_dino_rcde_track_r_v124_c_col_p_phase_a_control_aggregate_v2_20260824"
        ),
        payload_status=(
            "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_CONTROL_AGGREGATE_READY"
        ),
    ),
    (INDEPENDENT_V2, INDEPENDENT_AGGREGATE): _RoleSpec(
        family=INDEPENDENT_V2,
        role=INDEPENDENT_AGGREGATE,
        payload_kind="torch",
        receipt_schema=INDEPENDENT_RECEIPT_SCHEMA,
        receipt_status=INDEPENDENT_RECEIPT_STATUS,
        commit_phase="PHASE_A_P_V2_INDEPENDENT_REPLAY_ONLY",
        commit_stage="INDEPENDENT_AGGREGATE",
        payload_schema=(
            "rc_dino_rcde_track_r_v124_c_col_p_phase_a_v2_"
            "independent_aggregate_v1_20260824"
        ),
        payload_status=(
            "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_V2_"
            "INDEPENDENT_AGGREGATE_READY"
        ),
    ),
    (INDEPENDENT_V2, INDEPENDENT_SOURCE_SHARD): _RoleSpec(
        family=INDEPENDENT_V2,
        role=INDEPENDENT_SOURCE_SHARD,
        payload_kind="torch",
        receipt_schema=INDEPENDENT_RECEIPT_SCHEMA,
        receipt_status=INDEPENDENT_RECEIPT_STATUS,
        commit_phase="PHASE_A_P_V2_INDEPENDENT_REPLAY_ONLY",
        commit_stage="INDEPENDENT_SOURCE_SHARD",
        payload_schema=(
            "rc_dino_rcde_track_r_v124_c_col_p_phase_a_v2_"
            "independent_source_shard_v1_20260824"
        ),
        payload_status=(
            "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_V2_"
            "INDEPENDENT_SOURCE_SHARD_READY"
        ),
    ),
    (REPAIR_V3, REPAIR_NORMALIZED_AGGREGATE): _RoleSpec(
        family=REPAIR_V3,
        role=REPAIR_NORMALIZED_AGGREGATE,
        payload_kind="torch",
        receipt_schema=REPAIR_RECEIPT_SCHEMA_V3,
        receipt_status=REPAIR_RECEIPT_STATUS_V3,
        commit_phase=REPAIR_PHASE_V3,
        commit_stage="REPAIR_NORMALIZED_AGGREGATE",
    ),
    (REPAIR_V3, REPAIR_NORMALIZED_VALIDATION): _RoleSpec(
        family=REPAIR_V3,
        role=REPAIR_NORMALIZED_VALIDATION,
        payload_kind="json",
        receipt_schema=REPAIR_RECEIPT_SCHEMA_V3,
        receipt_status=REPAIR_RECEIPT_STATUS_V3,
        commit_phase=REPAIR_PHASE_V3,
        commit_stage="REPAIR_NORMALIZED_VALIDATION",
    ),
    (REPAIR_V3, REPAIR_FINAL_RESULT): _RoleSpec(
        family=REPAIR_V3,
        role=REPAIR_FINAL_RESULT,
        payload_kind="json",
        receipt_schema=REPAIR_RECEIPT_SCHEMA_V3,
        receipt_status=REPAIR_RECEIPT_STATUS_V3,
        commit_phase=REPAIR_PHASE_V3,
        commit_stage="REPAIR_FINAL_RESULT",
    ),
}


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except Exception as error:  # pragma: no cover - exact JSON failure varies
        raise ReceiptDispatchV3Error(f"{name} JSON drift") from error
    require(isinstance(value, dict), f"{name} root is not a mapping")
    return value


def _read_payload(path: Path, kind: str) -> Mapping[str, Any]:
    if kind == "json":
        return _read_json(path, "payload")
    require(kind == "torch", "payload kind drift")
    try:
        import torch

        value = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except Exception as error:  # pragma: no cover - exact torch failure varies
        raise ReceiptDispatchV3Error("torch payload decode drift") from error
    require(isinstance(value, Mapping), "torch payload root is not a mapping")
    return value


def _inside(root: Path, path: Path, name: str) -> Path:
    root_resolved = root.resolve(strict=True)
    raw = Path(path)
    require(not raw.is_symlink(), f"{name} symlink forbidden")
    lexical = Path(os.path.abspath(raw))
    require(
        lexical != root_resolved and root_resolved in lexical.parents,
        f"{name} escapes resume root",
    )
    relative = lexical.relative_to(root_resolved)
    cursor = root_resolved
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), f"{name} symlink forbidden")
    resolved = lexical.resolve(strict=True)
    require(
        resolved == lexical and root_resolved in resolved.parents,
        f"{name} resolved path drift",
    )
    return resolved


def _immutable_file(root: Path, path: Path, name: str) -> Path:
    resolved = _inside(root, path, name)
    require(
        resolved.is_file() and resolved.stat().st_mode & 0o777 == 0o444,
        f"{name} absent, mutable or unsafe",
    )
    return resolved


def _resolve_relative(root: Path, raw: object, name: str) -> Path:
    require(isinstance(raw, str) and bool(raw), f"{name} path absent")
    relative = Path(raw)
    require(not relative.is_absolute() and ".." not in relative.parts, f"{name} path drift")
    return (root / relative).resolve(strict=False)


def _validate_commit(
    *,
    spec: _RoleSpec,
    payload: Path,
    receipt: Path,
    commit_path: Path,
    resume_root: Path,
    authority_sha256: str,
    payload_sha256: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    value = _read_json(commit_path, "commit")
    authority = _sha(authority_sha256, "expected authority")
    required = {
        "schema_version", "status", "authority_sha256", "phase", "stage",
        "shard_ordinal", "candidate_start", "candidate_stop", "payload_path",
        "payload_bytes", "payload_sha256", "receipt_path", "receipt_bytes",
        "receipt_sha256", "scientific_GO_or_NO_GO",
        "automatic_stage_advance", "next_authorized_stage", "logical_sha256",
    }
    require(set(value) == required, "commit field set drift")
    require(
        value.get("schema_version") == COMMIT_SCHEMA_V2
        and value.get("status") == COMMIT_STATUS_V2
        and value.get("authority_sha256") == authority
        and value.get("phase") == spec.commit_phase
        and value.get("stage") == spec.commit_stage
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("payload_bytes") == payload.stat().st_size
        and value.get("payload_sha256") == payload_sha256
        and value.get("receipt_bytes") == receipt.stat().st_size
        and value.get("receipt_sha256") == receipt_sha256
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False
        and value.get("next_authorized_stage") is None,
        "commit envelope/hash/authority/role drift",
    )
    root = resume_root.resolve(strict=True)
    require(
        _resolve_relative(root, value.get("payload_path"), "commit payload") == payload
        and _resolve_relative(root, value.get("receipt_path"), "commit receipt") == receipt,
        "commit resume-root path binding drift",
    )
    if spec.role == INDEPENDENT_SOURCE_SHARD:
        start = value.get("candidate_start")
        stop = value.get("candidate_stop")
        ordinal = value.get("shard_ordinal")
        require(
            type(start) is int
            and type(stop) is int
            and type(ordinal) is int
            and 0 <= start < stop <= 128
            and stop - start == 16
            and start % 16 == 0
            and ordinal == start // 16,
            "independent source-shard coordinate drift",
        )
    elif spec.role == INDEPENDENT_AGGREGATE:
        require(
            value.get("shard_ordinal") is None
            and value.get("candidate_start") == 0
            and value.get("candidate_stop") == 128,
            "independent aggregate coordinate drift",
        )
    elif spec.family == PRODUCTION_DURABLE_V2:
        require(
            value.get("shard_ordinal") is None
            and value.get("candidate_start") is None
            and value.get("candidate_stop") is None,
            "production aggregate/artifact coordinate drift",
        )
    else:
        require(
            value.get("shard_ordinal") is None,
            "repair finalizer commit shard coordinate drift",
        )
    return value


def _validate_receipt(
    *,
    spec: _RoleSpec,
    payload_path: Path,
    receipt_path: Path,
    payload: Mapping[str, Any],
    resume_root: Path,
    repository_root: Path | None,
    payload_sha256: str,
    expected_resume_key_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    receipt = _read_json(receipt_path, "receipt")
    require(
        receipt.get("schema_version") == spec.receipt_schema
        and receipt.get("status") == spec.receipt_status
        and receipt.get("logical_sha256") == logical_sha256(receipt),
        "receipt family/status/logical drift",
    )
    payload_logical = _sha(payload.get("logical_sha256"), "payload logical")
    if spec.payload_kind == "json":
        require(
            payload_logical == logical_sha256(payload),
            "JSON payload logical drift",
        )
    if spec.payload_schema is not None:
        require(
            payload.get("schema_version") == spec.payload_schema
            and payload.get("status") == spec.payload_status,
            "payload schema/status role drift",
        )
    if spec.family == PRODUCTION_DURABLE_V2:
        required = {
            "schema_version", "status", "payload_path", "payload_sha256",
            "payload_schema_version", "payload_status",
            "payload_logical_sha256", "logical_sha256",
        }
        require(set(receipt) == required, "production receipt field set drift")
        require(repository_root is not None, "production repository root is required")
        repo = Path(repository_root).resolve(strict=True)
        require(repo.is_dir() and not Path(repository_root).is_symlink(), "repository root drift")
        receipt_payload = _resolve_relative(
            repo, receipt.get("payload_path"), "production receipt payload"
        )
        require(
            receipt_payload == payload_path
            and receipt.get("payload_sha256") == payload_sha256
            and receipt.get("payload_schema_version") == payload.get("schema_version")
            and receipt.get("payload_status") == payload.get("status")
            and receipt.get("payload_logical_sha256") == payload_logical,
            "production receipt payload binding drift",
        )
        require(
            expected_resume_key_sha256 is None,
            "production family cannot satisfy an independent resume key",
        )
    elif spec.family == INDEPENDENT_V2:
        required = {
            "schema_version", "status", "data_path", "data_bytes",
            "data_sha256", "data_logical_sha256", "resume_key_sha256",
            "logical_sha256",
        }
        require(set(receipt) == required, "independent receipt field set drift")
        resume_key = _sha(payload.get("resume_key_sha256"), "payload resume key")
        require(
            isinstance(receipt.get("data_path"), str)
            and Path(receipt["data_path"]).is_absolute()
            and Path(receipt["data_path"]).resolve(strict=False) == payload_path
            and receipt.get("data_bytes") == payload_path.stat().st_size
            and receipt.get("data_sha256") == payload_sha256
            and receipt.get("data_logical_sha256") == payload_logical
            and receipt.get("resume_key_sha256") == resume_key,
            "independent receipt payload/logical/resume drift",
        )
        if expected_resume_key_sha256 is not None:
            require(
                receipt.get("resume_key_sha256")
                == _sha(expected_resume_key_sha256, "expected resume key"),
                "independent expected resume key drift",
            )
    else:
        required = {
            "schema_version", "status", "role", "payload_path",
            "payload_sha256", "payload_bytes", "payload_logical_sha256",
            "logical_sha256",
        }
        require(set(receipt) == required, "repair V3 receipt field set drift")
        root = resume_root.resolve(strict=True)
        require(
            receipt.get("role") == spec.role
            and _resolve_relative(root, receipt.get("payload_path"), "repair receipt payload")
            == payload_path
            and receipt.get("payload_sha256") == payload_sha256
            and receipt.get("payload_bytes") == payload_path.stat().st_size
            and receipt.get("payload_logical_sha256") == payload_logical,
            "repair V3 receipt role/payload binding drift",
        )
        require(
            expected_resume_key_sha256 is None,
            "repair V3 family cannot satisfy an independent resume key",
        )
    return receipt, payload_logical


def validate_receipt_family(
    *,
    family: str,
    role: str,
    payload_path: Path,
    receipt_path: Path,
    commit_path: Path,
    resume_root: Path,
    authority_sha256: str,
    repository_root: Path | None = None,
    expected_resume_key_sha256: str | None = None,
) -> ValidatedReceiptFamilyV3:
    """Validate one explicit payload/receipt/commit family without writing.

    ``repository_root`` is mandatory only for production V2 because its
    historical receipt stores a repository-relative payload path.  Independent
    V2 stores an absolute data path; repair V3 stores a resume-root-relative
    path.  These conventions are validated rather than inferred.
    """

    require(isinstance(family, str) and isinstance(role, str), "family/role type drift")
    spec = _ROLE_SPECS.get((family, role))
    require(spec is not None, "unsupported or cross-family role dispatch")
    root_input = Path(resume_root)
    require(
        root_input.exists()
        and root_input.is_dir()
        and not root_input.is_symlink(),
        "resume root absent or unsafe",
    )
    root = root_input.resolve(strict=True)
    payload = _immutable_file(root, Path(payload_path), "payload")
    receipt = _immutable_file(root, Path(receipt_path), "receipt")
    commit = _immutable_file(root, Path(commit_path), "commit")
    require(len({payload, receipt, commit}) == 3, "payload/receipt/commit alias drift")

    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, file_sha256(path))
        for path in (payload, receipt, commit)
    }
    payload_mapping = _read_payload(payload, spec.payload_kind)
    payload_sha = before[payload][2]
    receipt_mapping, payload_logical = _validate_receipt(
        spec=spec,
        payload_path=payload,
        receipt_path=receipt,
        payload=payload_mapping,
        resume_root=root,
        repository_root=repository_root,
        payload_sha256=payload_sha,
        expected_resume_key_sha256=expected_resume_key_sha256,
    )
    receipt_sha = before[receipt][2]
    commit_mapping = _validate_commit(
        spec=spec,
        payload=payload,
        receipt=receipt,
        commit_path=commit,
        resume_root=root,
        authority_sha256=authority_sha256,
        payload_sha256=payload_sha,
        receipt_sha256=receipt_sha,
    )
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, file_sha256(path))
        for path in (payload, receipt, commit)
    }
    require(after == before, "read-only validation changed an artifact family")
    return ValidatedReceiptFamilyV3(
        family=family,
        role=role,
        payload_path=payload,
        receipt_path=receipt,
        commit_path=commit,
        payload_sha256=payload_sha,
        payload_bytes=payload.stat().st_size,
        payload_logical_sha256=payload_logical,
        receipt_sha256=receipt_sha,
        commit_logical_sha256=str(commit_mapping["logical_sha256"]),
        authority_sha256=_sha(authority_sha256, "expected authority"),
        resume_root=root,
        receipt=MappingProxyType(dict(receipt_mapping)),
        commit=MappingProxyType(dict(commit_mapping)),
    )


__all__ = [
    "COMMIT_SCHEMA_V2", "COMMIT_STATUS_V2", "INDEPENDENT_AGGREGATE",
    "INDEPENDENT_RECEIPT_SCHEMA", "INDEPENDENT_RECEIPT_STATUS",
    "INDEPENDENT_SOURCE_SHARD", "INDEPENDENT_V2", "PRODUCTION_CONTROL_AGGREGATE",
    "PRODUCTION_DURABLE_V2", "PRODUCTION_PHASE_A_ARTIFACT",
    "PRODUCTION_RECEIPT_SCHEMA", "PRODUCTION_RECEIPT_STATUS", "REPAIR_FINAL_RESULT",
    "REPAIR_NORMALIZED_AGGREGATE", "REPAIR_NORMALIZED_VALIDATION", "REPAIR_PHASE_V3",
    "REPAIR_RECEIPT_SCHEMA_V3", "REPAIR_RECEIPT_STATUS_V3", "REPAIR_V3",
    "ReceiptDispatchV3Error", "ValidatedReceiptFamilyV3", "canonical_sha256",
    "file_sha256", "logical_sha256", "validate_receipt_family",
]
