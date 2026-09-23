#!/usr/bin/env python3
"""Independent content/claim validator for the connected-superregion E0 result.

This validator does not import the new superregion implementation.  Behavioral
contracts are carried by the recorded pytest XML; this program independently
checks immutable content hashes, test counts, claim boundaries, protected-access
zeros, and the geometry/materialization fail-closed state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


EXPECTED_RESULT_SCHEMA = (
    "rc_dino_rcde_colnomic_connected_superregion_e0_result_v1_20260814"
)
EXPECTED_STATUS = "RCDE_SR_E0_CORE_READY_GEOMETRY_MATERIALIZATION_REQUIRED"
PASS_STATUS = "RCDE_SR_E0_INDEPENDENT_VALIDATION_PASS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    result_path = arguments.result.resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if payload.get("schema_version") != EXPECTED_RESULT_SCHEMA:
        errors.append("result schema mismatch")
    if payload.get("status") != EXPECTED_STATUS:
        errors.append("result status mismatch")
    if payload.get("claim_level") != "ENGINEERING_E0_ONLY":
        errors.append("claim level escaped engineering E0")
    if payload.get("scientific_GO_or_NO_GO") is not None:
        errors.append("E0 must not carry a scientific decision")
    if payload.get("natural_training_authorized") is not False:
        errors.append("E0 must not authorize natural training")
    for key in ("opened_access_count", "sealed_access_count", "target_mask_access_count"):
        if payload.get(key) != 0:
            errors.append(f"protected access must be zero: {key}")

    for name, record in payload.get("content_anchors", {}).items():
        path = root / record.get("path", "")
        if not path.is_file():
            errors.append(f"missing content anchor: {name}")
        elif _sha256(path) != record.get("sha256"):
            errors.append(f"content hash mismatch: {name}")

    tests = payload.get("engineering_tests", {})
    junit_path = root / tests.get("junit_path", "")
    if not junit_path.is_file() or _sha256(junit_path) != tests.get("junit_sha256"):
        errors.append("pytest JUnit receipt missing or changed")
    else:
        document = ET.parse(junit_path).getroot()
        suite = document if document.tag == "testsuite" else document.find("testsuite")
        if suite is None:
            errors.append("pytest JUnit has no testsuite")
        else:
            observed_tests = int(suite.attrib.get("tests", "-1"))
            observed_failures = int(suite.attrib.get("failures", "-1"))
            observed_errors = int(suite.attrib.get("errors", "-1"))
            if observed_tests != tests.get("combined_tests"):
                errors.append("pytest test-count mismatch")
            if observed_failures != 0 or observed_errors != 0:
                errors.append("pytest receipt contains failures/errors")

    geometry = payload.get("geometry_audit", {})
    if geometry.get("simple_grid_resize_authorized") is not False:
        errors.append("simple cross-backbone grid resize must remain forbidden")
    if geometry.get("natural_geometry_receipts_complete") is not False:
        errors.append("E0 cannot claim natural geometry receipts are complete")
    if geometry.get("dino_query_count") != geometry.get(
        "dino_query_with_existing_colnomic_spatial_tokens"
    ):
        errors.append("query token coverage is not complete")
    if geometry.get("missing_reference_spatial_token_count") != 46:
        errors.append("reference spatial-token deficit drift")

    boundary = payload.get("failure_boundary", {})
    expected_boundary = {
        "engineering_core_ready": True,
        "natural_geometry_bridge_ready": False,
        "natural_connected_region_signal_tested": False,
        "retrieval_improvement_tested": False,
    }
    if boundary != expected_boundary:
        errors.append("E0 claim boundary drift")
    if payload.get("next_authorized_stage") != (
        "RCDE_SR_CANONICAL_GEOMETRY_RECEIPT_MATERIALIZATION_CONTRACT"
    ):
        errors.append("next-stage authorization drift")

    output = {
        "schema_version": (
            "rc_dino_rcde_colnomic_connected_superregion_e0_validation_v1_20260814"
        ),
        "status": PASS_STATUS if not errors else "RCDE_SR_E0_INDEPENDENT_VALIDATION_FAIL",
        "errors": errors,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "scientific_GO_or_NO_GO": None,
        "natural_training_authorized": False,
    }
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()

