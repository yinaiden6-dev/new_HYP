#!/usr/bin/env python3
"""Independent P0 validator; deliberately does not import the P0 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_ZERO = {
    "C8_runtime_read_count": 0,
    "opened_runtime_read_count": 0,
    "sealed_runtime_read_count": 0,
    "home_files_modified": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"validation artifact drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(rendered); handle.flush(); os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def corrected_identities(setids: Sequence[str], repair: Mapping[str, Any]) -> list[str]:
    corrected = [str(value) for value in setids]
    for row in repair["filename_collision_overrides"]:
        corrected[int(row["physical_row"])] = str(row["corrected_identity"])
    for component in repair["verified_byte_identical_duplicate_components"]:
        for physical_row in component["physical_rows"]:
            corrected[int(physical_row)] = str(component["canonical_identity"])
    return corrected


def reduce_one(scores: Sequence[float], corrected: Sequence[str]) -> tuple[list[int], list[float]]:
    winners: dict[str, tuple[float, int]] = {}
    for row, identity in enumerate(corrected):
        score = float(scores[row]); current = winners.get(identity)
        if current is None or score > current[0] or (score == current[0] and row < current[1]):
            winners[identity] = (score, row)
    ordered = sorted(winners.items(), key=lambda item: (-item[1][0], item[0]))[:128]
    return [value[1][1] for value in ordered], [value[1][0] for value in ordered]


def confidence(scores: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in scores)
    median = (ordered[63] + ordered[64]) * 0.5
    deviations = sorted(abs(value - median) for value in ordered)
    mad = (deviations[63] + deviations[64]) * 0.5
    return (float(scores[0]) - float(scores[1])) / (1.4826 * mad + 1.0e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    protocol_path = args.protocol.resolve(); result_dir = args.result_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    contract = json.loads((result_dir / "contract.json").read_text(encoding="utf-8"))
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    prejoin_receipt = json.loads((result_dir / "base_prejoin_receipt.json").read_text(encoding="utf-8"))
    low = json.loads((result_dir / "rcde_raw_low_margin_receipt.json").read_text(encoding="utf-8"))
    data = json.loads((result_dir / "rcde_data_role_receipt.json").read_text(encoding="utf-8"))
    statistics = json.loads((result_dir / "rcde_statistics_receipt.json").read_text(encoding="utf-8"))
    resource = json.loads((result_dir / "rcde_resource_receipt.json").read_text(encoding="utf-8"))
    prejoin_path = result_dir / "base_prejoin.pt"
    prejoin = torch.load(prejoin_path, map_location="cpu", weights_only=False)
    gallery_binding = protocol["source_bindings"]["gallery_token_cache"]
    gallery_path = (ROOT / gallery_binding["path"]).resolve()
    gallery = torch.load(gallery_path, map_location="cpu", weights_only=False, mmap=True)
    repair_path = (ROOT / protocol["source_bindings"]["gallery_identity_repair"]["path"]).resolve()
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    corrected = corrected_identities(gallery["setids"], repair)
    rows_match = True; scores_match = True; confidence_match = True
    independent_confidences = []
    for index in range(600):
        rows, scores = reduce_one(prejoin["physical_scores"][index].tolist(), corrected)
        rows_match &= rows == prejoin["c128_rows_by_raw_rank"][index].tolist()
        recorded_scores = prejoin["c128_scores_by_raw_rank"][index].tolist()
        scores_match &= all(float(left) == float(right) for left, right in zip(scores, recorded_scores, strict=True))
        value = confidence(scores); independent_confidences.append(value)
        confidence_match &= abs(value - float(low["rows"][index]["u_RAW"])) <= 1.0e-15
    threshold_match = True; flag_match = True
    folds = [int(row["fold"]) for row in low["rows"]]
    for heldout in (1, 2, 3, 4):
        members = [index for index, fold in enumerate(folds) if fold != heldout]
        ordered = sorted(members, key=lambda index: (independent_confidences[index], low["rows"][index]["source_image_sha256"], low["rows"][index]["query_id"]))
        k = math.ceil(0.25 * len(ordered)); tau = independent_confidences[ordered[k - 1]]
        recorded = low["thresholds"][str(heldout)]
        threshold_match &= recorded["k"] == k and float(recorded["tau"]) == tau
    final_order = sorted(range(600), key=lambda index: (independent_confidences[index], low["rows"][index]["source_image_sha256"], low["rows"][index]["query_id"]))
    threshold_match &= float(low["tau_final"]) == independent_confidences[final_order[149]]
    for index, row in enumerate(low["rows"]):
        expected = independent_confidences[index] <= float(low["thresholds"][str(folds[index])]["tau"])
        flag_match &= bool(row["oof_low_margin"]) is expected
    role_ok = all(data["role_checks"].values())
    matched_ok = bool(data["matched_colnomic_control"]["eligible"])
    if not role_ok:
        expected_status = "DINO_RCDE_DATA_ROLE_PREFLIGHT_ABORT"
    elif not data["negative_gate"] or not data["balanced_pool_gate"]:
        expected_status = "DINO_RCDE_NEGATIVE_BAND_COVERAGE_INELIGIBLE"
    elif not data["n1_hall_gate"]:
        expected_status = "N1_PERMUTATION_CONTROL_INELIGIBLE"
    elif not matched_ok:
        expected_status = "COLNOMIC_RCDE_MATCHED_CONTROL_INELIGIBLE"
    elif not statistics["statistics_gate"]:
        expected_status = "DINO_RCDE_STATISTICALLY_INELIGIBLE"
    elif not resource["resource_gate"]:
        expected_status = "DINO_RCDE_RESOURCE_BLOCKED"
    else:
        expected_status = "DINO_RCDE_P0_READY"
    checks = {
        "protocol_hash": contract["protocol_sha256"] == sha256_file(protocol_path) == result["protocol_sha256"],
        "contract_frozen_before_score": contract["status"] == "P0_MACHINE_CONTRACT_VALIDATED_BEFORE_RAW_SCORE_READ"
        and contract["RAW_score_read_count"] == 0 and contract["label_read_count"] == 0
        and int(contract["created_ns"]) <= int(prejoin_receipt["created_ns"]),
        "low_margin_namespace_exact": low["namespace"] == "RAW_LOW_MARGIN_V1_1_Q25_MAD_C128",
        "low_margin_prejoin_before_label": low["label_read_count"] == 0,
        "base_prejoin_hash": prejoin_receipt["base_prejoin_sha256"] == sha256_file(prejoin_path),
        "independent_corrected_identity_reduction": rows_match and scores_match,
        "independent_low_margin_values": confidence_match,
        "independent_low_margin_thresholds": threshold_match and flag_match,
        "population_receipts": len(low["rows"]) == 600 and data["label_read_count"] == 600,
        "protected_access_zero": all(
            value.get("protected_access_counts") == PROTECTED_ZERO
            for value in (contract, prejoin_receipt, low, data, statistics, resource, result)
        ),
        "decision_status_exact": result["status"] == expected_status,
        "next_stage_fail_closed": result["next_authorized_stage"] == ("E0" if expected_status == "DINO_RCDE_P0_READY" else None),
        "no_natural_training_authority": result["natural_training_authorized"] is False,
    }
    validation_ok = all(checks.values())
    validation = {
        "schema_version": "rc_dino_rcde_p0_validation_v1_1_20260812",
        "status": "DINO_RCDE_P0_VALIDATION_PASS" if validation_ok else "DINO_RCDE_P0_VALIDATION_ABORT",
        "decision_status": expected_status, "checks": checks,
        "result_sha256": sha256_file(result_dir / "result.json"),
        "protocol_sha256": sha256_file(protocol_path),
        "next_authorized_stage": "E0" if validation_ok and expected_status == "DINO_RCDE_P0_READY" else None,
        "authority_update_eligible": validation_ok and expected_status == "DINO_RCDE_P0_READY",
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    exclusive_json(result_dir / "validation.json", validation)
    print(json.dumps({"status": validation["status"], "decision_status": expected_status, "authority_update_eligible": validation["authority_update_eligible"]}, sort_keys=True))
    return 0 if validation_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
