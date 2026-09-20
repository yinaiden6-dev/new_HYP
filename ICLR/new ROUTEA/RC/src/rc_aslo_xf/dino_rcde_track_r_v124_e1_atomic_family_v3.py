"""Symlink-safe atomic-family publication with member logical hashes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping

import torch

from . import dino_rcde_track_r_v124_e1_atomic_family_v2 as V2


PRODUCER_MEMBERS = V2.PRODUCER_MEMBERS
VALIDATION_MEMBERS = V2.VALIDATION_MEMBERS
CRASH_BOUNDARIES = V2.CRASH_BOUNDARIES
InjectedFamilyCrash = V2.InjectedFamilyCrash


class E1AtomicFamilyV3Error(RuntimeError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise E1AtomicFamilyV3Error(message)


def _raw_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lstat(path: Path):
    try:
        return os.lstat(path)
    except OSError as error:
        raise E1AtomicFamilyV3Error(f"lstat failed: {path}") from error


def _assert_directory_no_symlink(path: Path, *, name: str) -> Path:
    raw = _raw_absolute(path)
    metadata = _lstat(raw)
    require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{name} is not a raw regular directory",
    )
    return raw


def _assert_regular_no_symlink(path: Path, *, name: str) -> Path:
    raw = _raw_absolute(path)
    metadata = _lstat(raw)
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{name} is not a raw regular file",
    )
    return raw


def _assert_lexical_components(parent: Path, child: Path) -> None:
    base = _assert_directory_no_symlink(parent, name="trusted family parent")
    raw = _raw_absolute(child)
    require(base in raw.parents, "family/staging path escapes trusted parent")
    cursor = base
    for part in raw.relative_to(base).parts:
        cursor = cursor / part
        if os.path.lexists(cursor):
            metadata = _lstat(cursor)
            require(
                not stat.S_ISLNK(metadata.st_mode),
                f"symlink component forbidden: {cursor}",
            )


def _hidden_stage(public_dir: Path) -> Path:
    raw_public = _raw_absolute(public_dir)
    parent = _assert_directory_no_symlink(
        raw_public.parent, name="public family parent"
    )
    require(not os.path.lexists(raw_public), "public family already lexically exists")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{raw_public.name}.stage-", dir=parent)
    )
    stage = _raw_absolute(stage)
    _assert_lexical_components(parent, stage)
    metadata = _lstat(stage)
    require(
        stage.name.startswith(".")
        and stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_dev == _lstat(parent).st_dev,
        "hidden staging directory safety drift",
    )
    return stage


def _member_logical(path: Path) -> str:
    member = _assert_regular_no_symlink(path, name="family member")
    if member.suffix == ".pt":
        value = torch.load(
            member, map_location="cpu", weights_only=False, mmap=True
        )
        require(isinstance(value, Mapping), "torch member root drift")
        logical = value.get("logical_sha256")
        require(
            isinstance(logical, str) and len(logical) == 64,
            "torch member logical SHA absent",
        )
        if "tensor_payloads" in value:
            expected = V2.canonical_sha256(
                {
                    key: item
                    for key, item in value.items()
                    if key not in {"logical_sha256", "tensor_payloads"}
                }
            )
            require(
                logical == expected,
                "torch payload semantic logical SHA drift",
            )
        return logical
    value = json.loads(member.read_text(encoding="ascii"))
    require(
        isinstance(value, Mapping)
        and value.get("logical_sha256") == V2.logical_sha256(value),
        "JSON member logical SHA drift",
    )
    return str(value["logical_sha256"])


def _member_row(path: Path) -> dict[str, object]:
    member = _assert_regular_no_symlink(path, name="family member")
    return {
        "path": member.name,
        "bytes": member.stat().st_size,
        "sha256": V2.file_sha256(member),
        "logical_sha256": _member_logical(member),
    }


def _family_commit(
    stage: Path,
    members: tuple[str, ...],
    *,
    authority_sha256: str,
    family_role: str,
) -> dict[str, Any]:
    rows = [
        _member_row(stage / name)
        for name in members
        if name != "family_commit.json"
    ]
    value = {
        "schema_version": "rc_dino_rcde_track_r_v124_e1_v3_family_commit_20260826",
        "status": "DINO_RCDE_TRACK_R_V124_E1_V3_FAMILY_COMMITTED",
        "authority_sha256": authority_sha256,
        "family_role": family_role,
        "ordered_members": rows,
        "member_count": len(rows),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = V2.logical_sha256(value)
    return value


def _validate_member_triple(
    payload: Path,
    value: Mapping[str, Any],
    *,
    role: str,
    authority_sha256: str,
) -> None:
    receipt_path = payload.with_name(payload.name + ".receipt.json")
    commit_path = payload.with_name(payload.name + ".commit.json")
    _assert_regular_no_symlink(payload, name=f"{role} payload")
    _assert_regular_no_symlink(receipt_path, name=f"{role} receipt")
    _assert_regular_no_symlink(commit_path, name=f"{role} commit")
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    commit = json.loads(commit_path.read_text(encoding="ascii"))
    require(
        receipt == V2._receipt(payload, value, role=role)
        and commit
        == V2._commit(
            payload,
            receipt_path,
            authority_sha256=authority_sha256,
            role=role,
        ),
        f"{role} triple drift",
    )


def validate_family(
    public_dir: Path,
    *,
    authority_sha256: str,
    family_role: str,
    producer: bool,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any]]:
    raw_public = _raw_absolute(public_dir)
    parent = _assert_directory_no_symlink(
        raw_public.parent, name="public family parent"
    )
    _assert_lexical_components(parent, raw_public)
    directory = _assert_directory_no_symlink(
        raw_public, name="raw public family"
    )
    expected = PRODUCER_MEMBERS if producer else VALIDATION_MEMBERS
    names = tuple(sorted(path.name for path in directory.iterdir()))
    require(names == tuple(sorted(expected)), "family member set drift")
    for name in expected:
        member = _assert_regular_no_symlink(
            directory / name, name=f"family member {name}"
        )
        require(
            member.stat().st_mode & 0o777 == 0o444,
            f"family member mutable: {name}",
        )
    family_path = directory / "family_commit.json"
    family = json.loads(family_path.read_text(encoding="ascii"))
    expected_rows = [
        _member_row(directory / name)
        for name in expected
        if name != "family_commit.json"
    ]
    require(
        family.get("schema_version")
        == "rc_dino_rcde_track_r_v124_e1_v3_family_commit_20260826"
        and family.get("status")
        == "DINO_RCDE_TRACK_R_V124_E1_V3_FAMILY_COMMITTED"
        and family.get("authority_sha256") == authority_sha256
        and family.get("family_role") == family_role
        and family.get("ordered_members") == expected_rows
        and family.get("member_count") == len(expected_rows)
        and family.get("scientific_GO_or_NO_GO") is None
        and family.get("automatic_stage_advance") is False
        and family.get("next_authorized_stage") is None
        and family.get("logical_sha256") == V2.logical_sha256(family),
        "family commit/logical member rows drift",
    )
    if producer:
        artifact_path = directory / "prejoin_artifact.pt"
        artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=False, mmap=True
        )
        require(isinstance(artifact, Mapping), "artifact root drift")
        result_path = directory / "producer_result.json"
        result = json.loads(result_path.read_text(encoding="ascii"))
        _validate_member_triple(
            artifact_path,
            artifact,
            role="E1_V3_PREJOIN_ARTIFACT",
            authority_sha256=authority_sha256,
        )
        _validate_member_triple(
            result_path,
            result,
            role="E1_V3_PRODUCER_RESULT",
            authority_sha256=authority_sha256,
        )
        return artifact, result
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    _validate_member_triple(
        result_path,
        result,
        role="E1_V3_INDEPENDENT_RESULT",
        authority_sha256=authority_sha256,
    )
    return None, result


def publish_producer_family(
    public_dir: Path,
    *,
    authority_sha256: str,
    artifact: Mapping[str, Any],
    result_builder: Callable[[Path], Mapping[str, Any]],
    crash_after: str | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], bool]:
    raw_public = _raw_absolute(public_dir)
    if os.path.lexists(raw_public):
        artifact_value, result = validate_family(
            raw_public,
            authority_sha256=authority_sha256,
            family_role="E1_V3_PRODUCER_FAMILY",
            producer=True,
        )
        require(artifact_value is not None, "reused artifact absent")
        return artifact_value, result, True
    stage = _hidden_stage(raw_public)
    artifact_path = stage / "prejoin_artifact.pt"
    V2._write_torch(artifact_path, artifact)
    V2._crash("AFTER_ARTIFACT_PAYLOAD", crash_after)
    artifact_receipt = artifact_path.with_name(
        artifact_path.name + ".receipt.json"
    )
    V2._write_json(
        artifact_receipt,
        V2._receipt(
            artifact_path, artifact, role="E1_V3_PREJOIN_ARTIFACT"
        ),
    )
    V2._crash("AFTER_ARTIFACT_RECEIPT", crash_after)
    V2._write_json(
        artifact_path.with_name(artifact_path.name + ".commit.json"),
        V2._commit(
            artifact_path,
            artifact_receipt,
            authority_sha256=authority_sha256,
            role="E1_V3_PREJOIN_ARTIFACT",
        ),
    )
    V2._crash("AFTER_ARTIFACT_COMMIT", crash_after)
    result = dict(result_builder(artifact_path))
    result_path = stage / "producer_result.json"
    V2._write_json(result_path, result)
    V2._crash("AFTER_RESULT_PAYLOAD", crash_after)
    result_receipt = result_path.with_name(result_path.name + ".receipt.json")
    V2._write_json(
        result_receipt,
        V2._receipt(
            result_path, result, role="E1_V3_PRODUCER_RESULT"
        ),
    )
    V2._crash("AFTER_RESULT_RECEIPT", crash_after)
    V2._write_json(
        result_path.with_name(result_path.name + ".commit.json"),
        V2._commit(
            result_path,
            result_receipt,
            authority_sha256=authority_sha256,
            role="E1_V3_PRODUCER_RESULT",
        ),
    )
    V2._crash("AFTER_RESULT_COMMIT", crash_after)
    V2._write_json(
        stage / "family_commit.json",
        _family_commit(
            stage,
            PRODUCER_MEMBERS,
            authority_sha256=authority_sha256,
            family_role="E1_V3_PRODUCER_FAMILY",
        ),
    )
    V2._crash("AFTER_FAMILY_COMMIT", crash_after)
    V2._fsync_directory(stage)
    require(not os.path.lexists(raw_public), "public family appeared early")
    os.rename(stage, raw_public)
    V2._fsync_directory(raw_public.parent)
    V2._crash("AFTER_ATOMIC_RENAME", crash_after)
    artifact_value, result_value = validate_family(
        raw_public,
        authority_sha256=authority_sha256,
        family_role="E1_V3_PRODUCER_FAMILY",
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
    raw_public = _raw_absolute(public_dir)
    if os.path.lexists(raw_public):
        _, observed = validate_family(
            raw_public,
            authority_sha256=authority_sha256,
            family_role="E1_V3_VALIDATION_FAMILY",
            producer=False,
        )
        return observed, True
    stage = _hidden_stage(raw_public)
    result_path = stage / "result.json"
    V2._write_json(result_path, result)
    V2._crash("AFTER_RESULT_PAYLOAD", crash_after)
    receipt = result_path.with_name(result_path.name + ".receipt.json")
    V2._write_json(
        receipt,
        V2._receipt(result_path, result, role="E1_V3_INDEPENDENT_RESULT"),
    )
    V2._crash("AFTER_RESULT_RECEIPT", crash_after)
    V2._write_json(
        result_path.with_name(result_path.name + ".commit.json"),
        V2._commit(
            result_path,
            receipt,
            authority_sha256=authority_sha256,
            role="E1_V3_INDEPENDENT_RESULT",
        ),
    )
    V2._crash("AFTER_RESULT_COMMIT", crash_after)
    V2._write_json(
        stage / "family_commit.json",
        _family_commit(
            stage,
            VALIDATION_MEMBERS,
            authority_sha256=authority_sha256,
            family_role="E1_V3_VALIDATION_FAMILY",
        ),
    )
    V2._crash("AFTER_FAMILY_COMMIT", crash_after)
    V2._fsync_directory(stage)
    os.rename(stage, raw_public)
    V2._fsync_directory(raw_public.parent)
    V2._crash("AFTER_ATOMIC_RENAME", crash_after)
    _, observed = validate_family(
        raw_public,
        authority_sha256=authority_sha256,
        family_role="E1_V3_VALIDATION_FAMILY",
        producer=False,
    )
    return observed, False


__all__ = [
    "CRASH_BOUNDARIES",
    "E1AtomicFamilyV3Error",
    "InjectedFamilyCrash",
    "publish_producer_family",
    "publish_validation_family",
    "validate_family",
]
