"""Append-only progress and immutable commit receipts for Phase-A retry V2.

Progress JSONL is diagnostic only.  A stage is reusable exclusively when its
payload, receipt and commit marker all close under :func:`validate_commit`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


JOURNAL_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_retry_progress_v2_20260824"
)
COMMIT_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_retry_commit_v2_20260824"
)
RETRY_NAMESPACE = "dino_rcde_track_r_v124_c_col_p_phase_a_retry_v2"


class PhaseARetryProgressError(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise PhaseARetryProgressError(message)


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


def safe_retry_path(resume_root: Path, path: Path) -> Path:
    root = Path(resume_root).resolve(strict=False)
    value = Path(path).resolve(strict=False)
    require(
        RETRY_NAMESPACE in root.parts
        and (root == value.parent or root in value.parents),
        "Phase-A V2 path escapes the new retry namespace",
    )
    require(
        "dino_rcde_track_r_v124_c_col_p_phase_a_e0_v1" not in value.parts
        and "5106209" not in value.as_posix(),
        "V1 timeout residue entered the V2 retry namespace",
    )
    return value


def append_progress(
    journal_path: Path,
    event: Mapping[str, Any],
    *,
    resume_root: Path,
    authority_sha256: str,
    job_id: str,
) -> Mapping[str, Any]:
    """Append one fsynced JSON line with one atomic ``write(2)`` call."""

    journal = safe_retry_path(resume_root, journal_path)
    authority = _sha(authority_sha256, "progress authority")
    require(isinstance(job_id, str) and bool(job_id), "progress job ID absent")
    required = {"phase", "stage", "status"}
    require(required.issubset(event), "progress event fields incomplete")
    value: dict[str, Any] = {
        "schema_version": JOURNAL_SCHEMA,
        "authority_sha256": authority,
        "job_id": job_id,
        **dict(event),
    }
    value["event_sha256"] = canonical_sha256(value)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    require(len(payload) <= 4096, "progress line exceeds atomic append budget")
    journal.parent.mkdir(parents=True, exist_ok=True)
    require(not journal.is_symlink(), "progress journal symlink forbidden")
    descriptor = os.open(
        journal,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
        0o600,
    )
    try:
        written = os.write(descriptor, payload)
        require(written == len(payload), "short progress journal append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return value


def read_progress(journal_path: Path) -> tuple[Mapping[str, Any], ...]:
    """Validate every complete diagnostic line; never infer stage completion."""

    rows = []
    with Path(journal_path).open("r", encoding="ascii") as handle:
        for ordinal, line in enumerate(handle):
            value = json.loads(line)
            require(
                isinstance(value, dict)
                and value.get("schema_version") == JOURNAL_SCHEMA
                and value.get("event_sha256")
                == canonical_sha256(
                    {key: item for key, item in value.items() if key != "event_sha256"}
                ),
                f"progress journal line drift: {ordinal}",
            )
            rows.append(value)
    return tuple(rows)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "commit marker exists")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    finally:
        temp.unlink(missing_ok=True)


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
    """Create or exactly validate one immutable stage commit marker."""

    payload = safe_retry_path(resume_root, payload_path)
    receipt = safe_retry_path(resume_root, receipt_path)
    marker = safe_retry_path(resume_root, marker_path)
    require(
        payload.is_file()
        and receipt.is_file()
        and not payload.is_symlink()
        and not receipt.is_symlink()
        and payload.stat().st_mode & 0o777 == 0o444
        and receipt.stat().st_mode & 0o777 == 0o444,
        "commit payload/receipt absent or unsafe",
    )
    authority = _sha(authority_sha256, "commit authority")
    value: dict[str, Any] = {
        "schema_version": COMMIT_SCHEMA,
        "status": "PHASE_A_RETRY_STAGE_COMMITTED",
        "authority_sha256": authority,
        "phase": phase,
        "stage": stage,
        "shard_ordinal": shard_ordinal,
        "candidate_start": candidate_start,
        "candidate_stop": candidate_stop,
        "payload_path": payload.relative_to(Path(resume_root).resolve()).as_posix(),
        "payload_bytes": payload.stat().st_size,
        "payload_sha256": file_sha256(payload),
        "receipt_path": receipt.relative_to(Path(resume_root).resolve()).as_posix(),
        "receipt_bytes": receipt.stat().st_size,
        "receipt_sha256": file_sha256(receipt),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="ascii"))
        require(existing == value, "existing commit marker/context drift")
        validate_commit(marker, resume_root=resume_root, authority_sha256=authority)
        return existing
    _exclusive_json(marker, value)
    validate_commit(marker, resume_root=resume_root, authority_sha256=authority)
    return value


def validate_commit(
    marker_path: Path,
    *,
    resume_root: Path,
    authority_sha256: str,
) -> Mapping[str, Any]:
    marker = safe_retry_path(resume_root, marker_path)
    require(
        marker.is_file()
        and not marker.is_symlink()
        and marker.stat().st_mode & 0o777 == 0o444,
        "commit marker absent, mutable or unsafe",
    )
    value = json.loads(marker.read_text(encoding="ascii"))
    require(
        isinstance(value, dict)
        and value.get("schema_version") == COMMIT_SCHEMA
        and value.get("status") == "PHASE_A_RETRY_STAGE_COMMITTED"
        and value.get("authority_sha256")
        == _sha(authority_sha256, "commit authority")
        and value.get("logical_sha256") == logical_sha256(value)
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("automatic_stage_advance") is False
        and value.get("next_authorized_stage") is None,
        "commit marker envelope drift",
    )
    root = Path(resume_root).resolve()
    payload = safe_retry_path(root, root / value["payload_path"])
    receipt = safe_retry_path(root, root / value["receipt_path"])
    require(
        payload.is_file()
        and receipt.is_file()
        and not payload.is_symlink()
        and not receipt.is_symlink()
        and payload.stat().st_mode & 0o777 == 0o444
        and receipt.stat().st_mode & 0o777 == 0o444
        and payload.stat().st_size == value["payload_bytes"]
        and receipt.stat().st_size == value["receipt_bytes"]
        and file_sha256(payload) == value["payload_sha256"]
        and file_sha256(receipt) == value["receipt_sha256"],
        "committed payload/receipt hash drift",
    )
    return value


__all__ = [
    "COMMIT_SCHEMA",
    "JOURNAL_SCHEMA",
    "RETRY_NAMESPACE",
    "PhaseARetryProgressError",
    "append_progress",
    "canonical_sha256",
    "commit_stage",
    "file_sha256",
    "logical_sha256",
    "read_progress",
    "safe_retry_path",
    "validate_commit",
]
