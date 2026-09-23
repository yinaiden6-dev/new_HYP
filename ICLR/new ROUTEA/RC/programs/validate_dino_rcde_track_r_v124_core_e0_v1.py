#!/usr/bin/env python3
"""Independent envelope and binding validation for V124 core-repair E0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "registry/dino_rcde_track_r_v124_core_e0_authority_v1_20260824.json"
JUNIT = ROOT / "results/dino_rcde_track_r_v124_core_e0_v1/pytest.xml"
RESULT = ROOT / "results/dino_rcde_track_r_v124_core_e0_v1/result.json"
OUTPUT = ROOT / "results/dino_rcde_track_r_v124_core_e0_validation_v1/result.json"


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    ).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "logical_sha256"})


def validate_binding(binding: Mapping[str, Any]) -> None:
    relative = binding.get("path")
    req(isinstance(relative, str), "authority binding path absent")
    path = (ROOT / relative).resolve(strict=True)
    req(
        path.is_relative_to(ROOT.resolve())
        and path.is_file()
        and not path.is_symlink()
        and binding.get("bytes") == path.stat().st_size
        and binding.get("sha256") == sha(path),
        f"authority binding drift: {relative}",
    )
    if "logical_sha256" in binding:
        value = json.loads(path.read_text(encoding="utf-8"))
        req(
            value.get("logical_sha256") == logical(value) == binding["logical_sha256"],
            f"authority logical binding drift: {relative}",
        )


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    req(not path.exists() and not path.is_symlink(), "immutable E0 validation exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        temp.chmod(0o444); os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate_authority_only(authority_path: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    req(
        authority_path.resolve(strict=True) == AUTHORITY.resolve(strict=True),
        "E0 authority path drift",
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    req(
        (authority_path.stat().st_mode & 0o777) == 0o444
        and authority.get("status") == "DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_AUTHORIZED"
        and authority.get("logical_sha256") == logical(authority)
        and authority.get("scientific_GO_or_NO_GO") is None,
        "E0 authority drift",
    )
    bindings = authority.get("bindings")
    req(isinstance(bindings, Mapping), "E0 authority bindings absent")
    for binding in bindings.values():
        req(isinstance(binding, Mapping), "E0 binding object absent")
        validate_binding(binding)
    return authority, bindings


def build(authority_path: Path, junit_path: Path, result_path: Path) -> dict[str, Any]:
    for observed, expected in (
        (authority_path, AUTHORITY), (junit_path, JUNIT), (result_path, RESULT)
    ):
        req(observed.resolve(strict=True) == expected.resolve(strict=True), "E0 validation path drift")
    authority, bindings = validate_authority_only(authority_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    root = ET.parse(junit_path).getroot()
    cases = root.findall(".//testcase")
    req(
        (junit_path.stat().st_mode & 0o777) == 0o444
        and len(cases) == 10
        and not root.findall(".//failure")
        and not root.findall(".//error")
        and not root.findall(".//skipped"),
        "independent JUnit replay drift",
    )
    req(
        (result_path.stat().st_mode & 0o777) == 0o444
        and result.get("status") == "DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_PASS"
        and result.get("logical_sha256") == logical(result)
        and result.get("test_count") == 10
        and result.get("junit_sha256") == sha(junit_path)
        and result.get("authority_sha256") == sha(authority_path)
        and result.get("model_forward_count") == 0
        and result.get("target_rival_read_count") == 0
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("automatic_stage_advance") is False,
        "E0 result drift",
    )
    validation: dict[str, Any] = {
        "schema_version": "rc_dino_rcde_track_r_v124_core_e0_validation_v1_20260824",
        "status": "DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_INDEPENDENT_VALIDATION_PASS",
        "validation_pass": True,
        "authority_sha256": sha(authority_path),
        "result_sha256": sha(result_path),
        "result_logical_sha256": result["logical_sha256"],
        "junit_sha256": sha(junit_path),
        "verified_binding_count": len(bindings),
        "test_count": len(cases),
        "model_or_science_execution_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    validation["logical_sha256"] = logical(validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--junit", type=Path, default=JUNIT)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--authority-only", action="store_true")
    args = parser.parse_args()
    if args.authority_only:
        authority, bindings = validate_authority_only(args.authority)
        print(json.dumps({"status": "DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_AUTHORITY_RUNTIME_PASS", "binding_count": len(bindings), "logical_sha256": authority["logical_sha256"]}, sort_keys=True))
        return
    req(args.output.resolve(strict=False) == OUTPUT.resolve(strict=False), "E0 validation output drift")
    result = build(args.authority, args.junit, args.result)
    atomic(args.output, result)
    print(json.dumps({"status": result["status"], "validation_pass": True, "logical_sha256": result["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
