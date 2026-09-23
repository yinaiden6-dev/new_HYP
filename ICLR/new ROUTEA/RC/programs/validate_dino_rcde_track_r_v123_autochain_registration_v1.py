#!/usr/bin/env python3
"""Independently validate the immutable V123 scheduler registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = "results/dino_rcde_track_r_autochain_v1/v123_registration.json"
OUTPUT = "results/dino_rcde_track_r_autochain_v1/v123_registration.validation.json"
SCHEMA = "rc_dino_rcde_track_r_v123_autochain_registration_v1_20260824"
STATUS = "V123_AUTOMATIC_CONTINUATION_REGISTERED"
REPAIR_JOB_ID = 5104783
COMPARATOR_JOB_ID = 5102487
REQUIRED_BINDINGS = (
    "parent_v122_authority",
    "targeted7_repair_authority",
    "automatic_continuation_addendum",
    "v123_contract",
    "post_p_controller",
    "promotion_controller",
    "science_launcher",
    "v123_freezer",
    "v123_runtime_validator",
    "registrar",
    "registration_validator",
    "v123_test",
)


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "logical_sha256"})


def build(registration_path: Path) -> dict[str, Any]:
    resolved = registration_path.resolve(strict=True)
    req(
        resolved == (ROOT / REGISTRATION).resolve(strict=True)
        and resolved.is_file()
        and not resolved.is_symlink()
        and (resolved.stat().st_mode & 0o777) == 0o444,
        "registration path/mode drift",
    )
    value = json.loads(resolved.read_text(encoding="utf-8"))
    req(
        isinstance(value, dict)
        and value.get("schema_version") == SCHEMA
        and value.get("status") == STATUS
        and value.get("logical_sha256") == logical(value),
        "registration envelope/logical drift",
    )
    promotion = value.get("promotion_job_id")
    science = value.get("science_job_id")
    req(
        value.get("repair_job_id") == REPAIR_JOB_ID
        and value.get("comparator_job_id") == COMPARATOR_JOB_ID
        and isinstance(promotion, int)
        and promotion > COMPARATOR_JOB_ID
        and isinstance(science, int)
        and science > promotion
        and value.get("promotion_dependency") == f"afterok:{COMPARATOR_JOB_ID}"
        and value.get("science_dependency") == f"afterok:{promotion}"
        and value.get("automatic_stage_advance_to_v123") is True
        and value.get("automatic_stage_advance_after_v123") is False
        and value.get("model_or_science_execution_count") == 0
        and value.get("scientific_GO_or_NO_GO") is None,
        "registration job/dependency/claim drift",
    )
    bindings = value.get("bindings")
    req(isinstance(bindings, Mapping), "registration bindings absent")
    observed: list[str] = []
    for name in REQUIRED_BINDINGS:
        binding = bindings.get(name)
        req(isinstance(binding, Mapping), f"registration binding absent: {name}")
        relative = binding.get("path")
        req(isinstance(relative, str), f"registration binding path absent: {name}")
        raw = Path(relative)
        req(not raw.is_absolute() and ".." not in raw.parts, f"unsafe binding: {name}")
        path = (ROOT / raw).resolve(strict=True)
        req(
            path.is_relative_to(ROOT.resolve())
            and path.is_file()
            and not path.is_symlink()
            and binding.get("bytes") == path.stat().st_size
            and binding.get("sha256") == file_sha(path),
            f"registration binding drift: {name}",
        )
        if "logical_sha256" in binding:
            bound = json.loads(path.read_text(encoding="utf-8"))
            req(
                isinstance(bound, dict)
                and binding.get("logical_sha256") == bound.get("logical_sha256")
                and bound.get("logical_sha256") == logical(bound),
                f"registration logical binding drift: {name}",
            )
        observed.append(relative)
    req(len(observed) == len(set(observed)), "registration binding path collision")
    result: dict[str, Any] = {
        "schema_version": "rc_dino_rcde_track_r_v123_autochain_registration_validation_v1_20260824",
        "status": "V123_AUTOMATIC_CONTINUATION_REGISTRATION_INDEPENDENT_VALIDATION_PASS",
        "validation_pass": True,
        "registration_sha256": file_sha(resolved),
        "registration_logical_sha256": value["logical_sha256"],
        "promotion_job_id": promotion,
        "science_job_id": science,
        "verified_binding_count": len(observed),
        "model_or_science_execution_count": 0,
        "scientific_GO_or_NO_GO": None,
    }
    result["logical_sha256"] = logical(result)
    return result


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    req(not path.exists() and not path.is_symlink(), "immutable validation already exists")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(0o444)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, default=ROOT / REGISTRATION)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    req(
        args.output.resolve(strict=False) == (ROOT / OUTPUT).resolve(strict=False),
        "validation output drift",
    )
    result = build(args.registration)
    if not args.dry_run:
        atomic(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "promotion_job_id": result["promotion_job_id"],
                "science_job_id": result["science_job_id"],
                "dry_run": bool(args.dry_run),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
