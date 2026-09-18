#!/usr/bin/env python3
"""Validate the frozen contract without touching sealed endpoint bytes."""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
CONTRACT = ROOT / "plan/ROMAV2_COLNOMIC_FULL_NEGATIVE_EXTERNAL_CONFIRMATION_CONTRACT_V1_20260831.md"
AUTHORITY = ROOT / "registry/romav2_colnomic_full_negative_external_confirmation_authority_v1_20260831.json"
RESULT = ROOT / "results/romav2_colnomic_full_negative_action_gate_v1/result.json"
VALIDATION = ROOT / "results/romav2_colnomic_full_negative_action_gate_v1/independent_validation.json"
TRAINER = ROOT / "programs/run_romav2_colnomic_full_negative_action_gate_v1.py"
SCORER = ROOT / "programs/run_romav2_colnomic_visibility_xf_six_case_v1.py"
CHECKPOINT = WORKSPACE / "third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt"
OUT = ROOT / "results/romav2_colnomic_external_confirmation_contract_v1/preflight.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def logical(value: dict) -> str:
    projection = dict(value)
    projection.pop("logical_sha256", None)
    payload = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable preflight already exists: {OUT}")

    authority = json.loads(AUTHORITY.read_text())
    result = json.loads(RESULT.read_text())
    validation = json.loads(VALIDATION.read_text())

    expected_weights = [
        0.9812819097198987,
        -3.812628067996388,
        5.826320131219255,
        5.860970045711693,
        -0.14878670951682949,
        -0.2442796885686483,
    ]
    expected_bias = -1.5548565799571785

    checks = {
        "contract_hash": sha(CONTRACT) == authority["contract_sha256"],
        "source_result_hash": sha(RESULT) == authority["source_result"]["sha256"],
        "source_validation_hash": sha(VALIDATION)
        == authority["source_independent_validation"]["sha256"],
        "source_status": result["status"]
        == "ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_HEADROOM",
        "independent_validation_status": validation["status"]
        == "ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_INDEPENDENT_VALIDATION_PASS",
        "trainer_hash": sha(TRAINER)
        == "3fc3491c05562f59bbca3e983522f3b75d7a19e4f2ee121d89bbb3bc8a7b0310",
        "scorer_hash": sha(SCORER)
        == "fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c",
        "checkpoint_hash": sha(CHECKPOINT)
        == "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7",
        "parameter_count": result["parameter_count"] == 7
        and authority["frozen_action"]["parameter_count"] == 7,
        "weights_exact": result["head"]["weight"] == expected_weights
        and authority["frozen_action"]["weight"] == expected_weights,
        "bias_exact": result["head"]["bias"] == expected_bias
        and authority["frozen_action"]["bias"] == expected_bias,
        "endpoint_metadata": authority["frozen_endpoint_metadata"]
        == {
            "name": "new_difficult_test_v1",
            "identity_count": 8,
            "primary_query_count": 31,
            "complete_gallery_physical_row_count": 5413,
            "role": "ONE_SHOT_SEALED_DIRECTIONAL_CONFIRMATION",
        },
        "all_numbers_finite": all(math.isfinite(x) for x in expected_weights + [expected_bias]),
        "sealed_access_disabled": authority["permissions"]
        == {
            "sealed_manifest_metadata_binding_authorized": True,
            "sealed_query_bytes_read_authorized": False,
            "sealed_label_read_authorized": False,
            "sealed_scoring_authorized": False,
            "model_or_threshold_change_authorized": False,
        },
        "prior_access_zero": result["opened_read_count"] == 0
        and result["sealed_read_count"] == 0
        and validation["sealed_data_read_authorized"] is False,
    }
    passed = all(checks.values())
    value = {
        "schema_version": "rc_romav2_colnomic_external_confirmation_contract_preflight_v1_20260831",
        "status": (
            "SEALED_INPUT_MANIFEST_BINDING_PREFLIGHT_READY"
            if passed
            else "EXTERNAL_CONFIRMATION_CONTRACT_INVALID"
        ),
        "claim_level": "CONTRACT_PREFLIGHT_ONLY_NO_SEALED_ACCESS",
        "checks": checks,
        "contract_sha256": sha(CONTRACT),
        "authority_sha256": sha(AUTHORITY),
        "sealed_query_bytes_read_count": 0,
        "sealed_label_read_count": 0,
        "sealed_scoring_count": 0,
        "sealed_data_read_authorized": False,
        "next_authorized_stage": "BIND_PREEXISTING_SEALED_MANIFEST_METADATA" if passed else None,
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical(value)
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True))
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
