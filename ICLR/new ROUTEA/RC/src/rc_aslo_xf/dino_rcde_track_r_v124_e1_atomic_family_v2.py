"""Crash-safe, append-only publication for V124-E1 V2 result families."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import torch


PRODUCER_MEMBERS = (
    "prejoin_artifact.pt",
    "prejoin_artifact.pt.receipt.json",
    "prejoin_artifact.pt.commit.json",
    "producer_result.json",
    "producer_result.json.receipt.json",
    "producer_result.json.commit.json",
    "family_commit.json",
)
VALIDATION_MEMBERS = (
    "result.json",
    "result.json.receipt.json",
    "result.json.commit.json",
    "family_commit.json",
)
CRASH_BOUNDARIES = (
    "AFTER_ARTIFACT_PAYLOAD",
    "AFTER_ARTIFACT_RECEIPT",
    "AFTER_ARTIFACT_COMMIT",
    "AFTER_RESULT_PAYLOAD",
    "AFTER_RESULT_RECEIPT",
    "AFTER_RESULT_COMMIT",
    "AFTER_FAMILY_COMMIT",
    "AFTER_ATOMIC_RENAME",
)


class E1AtomicFamilyV2Error(RuntimeError):
    pass


class InjectedFamilyCrash(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1AtomicFamilyV2Error(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _crash(boundary: str, requested: str | None) -> None:
    require(boundary in CRASH_BOUNDARIES, "unknown publication boundary")
    if requested == boundary:
        raise InjectedFamilyCrash(boundary)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hidden_stage(public_dir: Path) -> Path:
    parent = public_dir.parent.resolve()
    require(parent.is_dir() and not parent.is_symlink(), "family parent absent/unsafe")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{public_dir.name}.stage-", dir=parent
        )
    )
    require(stage.name.startswith("."), "staging directory is public")
    return stage


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "staged JSON exists")
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("x", encoding="ascii") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o444)
    os.link(temporary, path)
    temporary.unlink()
    _fsync_file(path)


def _write_torch(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "staged torch exists")
    temporary = path.with_name(f".{path.name}.partial")
    torch.save(dict(value), temporary)
    _fsync_file(temporary)
    temporary.chmod(0o444)
    os.link(temporary, path)
    temporary.unlink()
    _fsync_file(path)


def _receipt(
    payload: Path,
    value: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    result = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_v2_receipt_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_V2_RECEIPT_READY",
        "role": role,
        "payload_path": payload.name,
        "payload_bytes": payload.stat().st_size,
        "payload_sha256": file_sha256(payload),
        "payload_logical_sha256": value["logical_sha256"],
    }
    result["logical_sha256"] = logical_sha256(result)
    return result


def _commit(
    payload: Path,
    receipt: Path,
    *,
    authority_sha256: str,
    role: str,
) -> dict[str, Any]:
    result = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_v2_payload_commit_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_V2_PAYLOAD_COMMITTED",
        "authority_sha256": authority_sha256,
        "role": role,
        "payload_path": payload.name,
        "payload_bytes": payload.stat().st_size,
        "payload_sha256": file_sha256(payload),
        "receipt_path": receipt.name,
        "receipt_bytes": receipt.stat().st_size,
        "receipt_sha256": file_sha256(receipt),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    return result


def _family_commit(
    stage: Path,
    members: tuple[str, ...],
    *,
    authority_sha256: str,
    family_role: str,
) -> dict[str, Any]:
    rows = []
    for name in members:
        if name == "family_commit.json":
            continue
        path = stage / name
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_mode & 0o777 == 0o444,
            f"staged family member absent/mutable: {name}",
        )
        rows.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    result = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_v2_family_commit_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_V2_FAMILY_COMMITTED",
        "authority_sha256": authority_sha256,
        "family_role": family_role,
        "ordered_members": rows,
        "member_count": len(rows),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    return result


def validate_family(
    public_dir: Path,
    *,
    authority_sha256: str,
    family_role: str,
    producer: bool,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any]]:
    directory = public_dir.resolve()
    require(
        directory.is_dir() and not directory.is_symlink(),
        "committed family absent/unsafe",
    )
    expected = PRODUCER_MEMBERS if producer else VALIDATION_MEMBERS
    observed_names = tuple(sorted(path.name for path in directory.iterdir()))
    require(
        observed_names == tuple(sorted(expected)),
        "committed family member set drift",
    )
    for name in expected:
        path = directory / name
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_mode & 0o777 == 0o444,
            f"committed member absent/mutable: {name}",
        )
    family = json.loads(
        (directory / "family_commit.json").read_text(encoding="ascii")
    )
    require(
        family.get("schema_version")
        == "rc_dino_rcde_track_r_v124_e1_v2_family_commit_20260826"
        and family.get("status")
        == "DINO_RCDE_TRACK_R_V124_E1_V2_FAMILY_COMMITTED"
        and family.get("authority_sha256") == authority_sha256
        and family.get("family_role") == family_role
        and family.get("logical_sha256") == logical_sha256(family)
        and family.get("scientific_GO_or_NO_GO") is None
        and family.get("automatic_stage_advance") is False
        and family.get("next_authorized_stage") is None,
        "family commit envelope drift",
    )
    rows = family.get("ordered_members")
    require(isinstance(rows, list), "family member rows absent")
    expected_rows = []
    for name in expected:
        if name == "family_commit.json":
            continue
        path = directory / name
        expected_rows.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    require(
        rows == expected_rows and family.get("member_count") == len(expected_rows),
        "family commit/member bytes drift",
    )

    if producer:
        artifact_path = directory / "prejoin_artifact.pt"
        result_path = directory / "producer_result.json"
        artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=False, mmap=True
        )
        require(isinstance(artifact, Mapping), "committed artifact root drift")
        result = json.loads(result_path.read_text(encoding="ascii"))
        payloads = (
            (artifact_path, artifact, "E1_V2_PREJOIN_ARTIFACT"),
            (result_path, result, "E1_V2_PRODUCER_RESULT"),
        )
    else:
        artifact = None
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="ascii"))
        payloads = ((result_path, result, "E1_V2_INDEPENDENT_RESULT"),)

    for payload, value, role in payloads:
        receipt_path = payload.with_name(payload.name + ".receipt.json")
        commit_path = payload.with_name(payload.name + ".commit.json")
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        commit = json.loads(commit_path.read_text(encoding="ascii"))
        require(
            receipt == _receipt(payload, value, role=role)
            and commit
            == _commit(
                payload,
                receipt_path,
                authority_sha256=authority_sha256,
                role=role,
            ),
            f"committed {role} triple drift",
        )
    return artifact, result


def publish_producer_family(
    public_dir: Path,
    *,
    authority_sha256: str,
    artifact: Mapping[str, Any],
    result_builder: Callable[[Path], Mapping[str, Any]],
    crash_after: str | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], bool]:
    if public_dir.exists() or public_dir.is_symlink():
        artifact_value, result = validate_family(
            public_dir,
            authority_sha256=authority_sha256,
            family_role="E1_V2_PRODUCER_FAMILY",
            producer=True,
        )
        require(artifact_value is not None, "reused artifact absent")
        return artifact_value, result, True
    stage = _hidden_stage(public_dir)
    artifact_path = stage / "prejoin_artifact.pt"
    _write_torch(artifact_path, artifact)
    _crash("AFTER_ARTIFACT_PAYLOAD", crash_after)
    artifact_receipt = artifact_path.with_name(
        artifact_path.name + ".receipt.json"
    )
    _write_json(
        artifact_receipt,
        _receipt(artifact_path, artifact, role="E1_V2_PREJOIN_ARTIFACT"),
    )
    _crash("AFTER_ARTIFACT_RECEIPT", crash_after)
    _write_json(
        artifact_path.with_name(artifact_path.name + ".commit.json"),
        _commit(
            artifact_path,
            artifact_receipt,
            authority_sha256=authority_sha256,
            role="E1_V2_PREJOIN_ARTIFACT",
        ),
    )
    _crash("AFTER_ARTIFACT_COMMIT", crash_after)
    result = dict(result_builder(artifact_path))
    result_path = stage / "producer_result.json"
    _write_json(result_path, result)
    _crash("AFTER_RESULT_PAYLOAD", crash_after)
    result_receipt = result_path.with_name(result_path.name + ".receipt.json")
    _write_json(
        result_receipt,
        _receipt(result_path, result, role="E1_V2_PRODUCER_RESULT"),
    )
    _crash("AFTER_RESULT_RECEIPT", crash_after)
    _write_json(
        result_path.with_name(result_path.name + ".commit.json"),
        _commit(
            result_path,
            result_receipt,
            authority_sha256=authority_sha256,
            role="E1_V2_PRODUCER_RESULT",
        ),
    )
    _crash("AFTER_RESULT_COMMIT", crash_after)
    _write_json(
        stage / "family_commit.json",
        _family_commit(
            stage,
            PRODUCER_MEMBERS,
            authority_sha256=authority_sha256,
            family_role="E1_V2_PRODUCER_FAMILY",
        ),
    )
    _crash("AFTER_FAMILY_COMMIT", crash_after)
    _fsync_directory(stage)
    require(not public_dir.exists() and not public_dir.is_symlink(), "public family appeared before commit")
    os.rename(stage, public_dir)
    _fsync_directory(public_dir.parent)
    _crash("AFTER_ATOMIC_RENAME", crash_after)
    artifact_value, result_value = validate_family(
        public_dir,
        authority_sha256=authority_sha256,
        family_role="E1_V2_PRODUCER_FAMILY",
        producer=True,
    )
    require(artifact_value is not None, "published artifact absent")
    return artifact_value, result_value, False


def publish_validation_family(
    public_dir: Path,
    *,
    authority_sha256: str,
    result: Mapping[str, Any],
    crash_after: str | None = None,
) -> tuple[Mapping[str, Any], bool]:
    if public_dir.exists() or public_dir.is_symlink():
        _, observed = validate_family(
            public_dir,
            authority_sha256=authority_sha256,
            family_role="E1_V2_VALIDATION_FAMILY",
            producer=False,
        )
        return observed, True
    stage = _hidden_stage(public_dir)
    result_path = stage / "result.json"
    _write_json(result_path, result)
    _crash("AFTER_RESULT_PAYLOAD", crash_after)
    receipt_path = result_path.with_name(result_path.name + ".receipt.json")
    _write_json(
        receipt_path,
        _receipt(result_path, result, role="E1_V2_INDEPENDENT_RESULT"),
    )
    _crash("AFTER_RESULT_RECEIPT", crash_after)
    _write_json(
        result_path.with_name(result_path.name + ".commit.json"),
        _commit(
            result_path,
            receipt_path,
            authority_sha256=authority_sha256,
            role="E1_V2_INDEPENDENT_RESULT",
        ),
    )
    _crash("AFTER_RESULT_COMMIT", crash_after)
    _write_json(
        stage / "family_commit.json",
        _family_commit(
            stage,
            VALIDATION_MEMBERS,
            authority_sha256=authority_sha256,
            family_role="E1_V2_VALIDATION_FAMILY",
        ),
    )
    _crash("AFTER_FAMILY_COMMIT", crash_after)
    _fsync_directory(stage)
    os.rename(stage, public_dir)
    _fsync_directory(public_dir.parent)
    _crash("AFTER_ATOMIC_RENAME", crash_after)
    _, observed = validate_family(
        public_dir,
        authority_sha256=authority_sha256,
        family_role="E1_V2_VALIDATION_FAMILY",
        producer=False,
    )
    return observed, False


__all__ = [
    "CRASH_BOUNDARIES",
    "E1AtomicFamilyV2Error",
    "InjectedFamilyCrash",
    "canonical_sha256",
    "file_sha256",
    "logical_sha256",
    "publish_producer_family",
    "publish_validation_family",
    "validate_family",
]
