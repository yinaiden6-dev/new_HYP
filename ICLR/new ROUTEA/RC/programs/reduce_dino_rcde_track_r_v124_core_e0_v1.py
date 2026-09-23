#!/usr/bin/env python3
"""Reduce the frozen V124 core-repair pytest receipt without scientific reads."""

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
OUTPUT = ROOT / "results/dino_rcde_track_r_v124_core_e0_v1/result.json"
EXPECTED_TEST_COUNT = 10


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


def atomic(path: Path, value: Mapping[str, Any]) -> None:
    req(not path.exists() and not path.is_symlink(), "immutable E0 result exists")
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


def build(authority_path: Path, junit_path: Path) -> dict[str, Any]:
    req(
        authority_path.resolve(strict=True) == AUTHORITY.resolve(strict=True)
        and junit_path.resolve(strict=True) == JUNIT.resolve(strict=True),
        "E0 authority/JUnit path drift",
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    req(
        (authority_path.stat().st_mode & 0o777) == 0o444
        and authority.get("status") == "DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_AUTHORIZED"
        and authority.get("logical_sha256") == logical(authority)
        and authority.get("scientific_GO_or_NO_GO") is None,
        "E0 authority drift",
    )
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(item.attrib.get("tests", 0)) for item in suites)
    failures = sum(int(item.attrib.get("failures", 0)) for item in suites)
    errors = sum(int(item.attrib.get("errors", 0)) for item in suites)
    skipped = sum(int(item.attrib.get("skipped", 0)) for item in suites)
    cases = root.findall(".//testcase")
    req(
        tests == len(cases) == EXPECTED_TEST_COUNT
        and failures == errors == skipped == 0,
        "V124 core E0 pytest gate failed",
    )
    classes = sorted({str(item.attrib.get("classname", "")) for item in cases})
    req(
        any("test_dino_rcde_sr0_mt_v_runtime_v2" in item for item in classes)
        and any("test_dino_rcde_track_r_full_c128_c_dino_v1" in item for item in classes),
        "V124 core E0 test population drift",
    )
    junit_path.chmod(0o444)
    result: dict[str, Any] = {
        "schema_version": "rc_dino_rcde_track_r_v124_core_e0_result_v1_20260824",
        "status": "DINO_RCDE_TRACK_R_V124_CORE_REPAIR_E0_PASS",
        "claim_level": "ENGINEERING_CORE_REPAIR_ONLY",
        "test_count": tests,
        "failure_count": failures,
        "error_count": errors,
        "skipped_count": skipped,
        "test_classnames": classes,
        "junit_sha256": sha(junit_path),
        "authority_sha256": sha(authority_path),
        "authority_logical_sha256": authority["logical_sha256"],
        "full_reference_rootwise_qualified": True,
        "full_c128_c_dino_qualified": True,
        "c_col_p_qualified": False,
        "natural_execution_count": 1,
        "model_load_count": 0,
        "model_forward_count": 0,
        "target_rival_read_count": 0,
        "scientific_reduction_count": 0,
        "protected_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--junit", type=Path, default=JUNIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    req(args.output.resolve(strict=False) == OUTPUT.resolve(strict=False), "E0 output drift")
    result = build(args.authority, args.junit)
    atomic(args.output, result)
    print(json.dumps({"status": result["status"], "test_count": result["test_count"], "logical_sha256": result["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
