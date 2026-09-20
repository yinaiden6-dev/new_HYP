#!/usr/bin/env python3
"""Freeze executable Scale-4 confirmation authority after PJ1 PASS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = Path(
    "registry/h0/h0_component_scale4_confirmation_authority_v36_20260824.json"
)
STATUS = "H0_COMPONENT_SCALE4_INTERNAL_CONFIRMATION_EXECUTION_AUTHORIZED"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {relative}")
    return value


def build() -> dict[str, Any]:
    prereg_relative = "registry/h0/h0_component_scale4_prereg_authority_v35_20260824.json"
    pj1_authority_relative = (
        "registry/h0/h0_c128_target_free_postjoin_pj1_authority_v1_20260824.json"
    )
    pj1_result_relative = "results/dino_rcde_h0_c128_target_free_postjoin_pj1_v1/result.json"
    pj1_validation_relative = (
        "results/dino_rcde_h0_c128_target_free_postjoin_pj1_validation_v1/result.json"
    )
    pj2_result_relative = "results/dino_rcde_h0_c128_target_free_postjoin_pj2_v1/result.json"
    pj2_validation_relative = (
        "results/dino_rcde_h0_c128_target_free_postjoin_pj2_validation_v1/result.json"
    )
    prereg = read(prereg_relative)
    pj1_authority = read(pj1_authority_relative)
    pj1 = read(pj1_result_relative)
    pj1_validation = read(pj1_validation_relative)
    pj2 = read(pj2_result_relative)
    pj2_validation = read(pj2_validation_relative)
    require(
        prereg.get("status")
        == "H0_COMPONENT_SCALE4_SUCCESSOR_PREREGISTERED_NOT_EXECUTABLE"
        and prereg.get("logical_sha256") == H.logical(prereg)
        and prereg.get("scale") == 4.0
        and prereg.get("confirmation_execution_ordinals_excluded") == list(range(12))
        and prereg.get("expected_confirmation_query_count") == 582
        and prereg.get("alternative_scale_count") == 0,
        "Scale-4 preregistration drift",
    )
    require(
        pj1_authority.get("status")
        == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_EXECUTION_AUTHORIZED"
        and pj1_authority.get("logical_sha256") == H.logical(pj1_authority)
        and pj1.get("status") == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_READY"
        and pj1.get("logical_sha256") == H.logical(pj1)
        and pj1.get("query_count") == 594
        and pj1.get("candidate_count") == 76032
        and pj1_validation.get("status")
        == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_INDEPENDENT_VALIDATION_PASS"
        and pj1_validation.get("validation_pass") is True
        and pj1_validation.get("result_sha256") == H.file_sha(ROOT / pj1_result_relative),
        "PJ1 immutable qualification drift",
    )
    require(
        pj2.get("status") == "RELATIVE_C128_RANK_INCREMENT_NO_GO"
        and pj2.get("scientific_GO_or_NO_GO") == "NO_GO"
        and pj2.get("logical_sha256") == H.logical(pj2)
        and pj2_validation.get("validation_pass") is True
        and pj2_validation.get("result_sha256") == H.file_sha(ROOT / pj2_result_relative),
        "unscaled PJ2 negative comparator drift",
    )
    bindings = {
        "contract": H.bind(
            "plan/DINO_RCDE_H0_COMPONENT_SCALE4_SUCCESSOR_CONTRACT_V1_20260824.md"
        ),
        "preregistration_authority": H.bind(
            prereg_relative, with_logical=True, immutable=True
        ),
        "pj1_authority": H.bind(
            pj1_authority_relative, with_logical=True, immutable=True
        ),
        "pj1_result": H.bind(
            pj1_result_relative, with_logical=True, immutable=True
        ),
        "pj1_validation": H.bind(
            pj1_validation_relative, with_logical=True, immutable=True
        ),
        "unscaled_pj2_result": H.bind(
            pj2_result_relative, with_logical=True, immutable=True
        ),
        "unscaled_pj2_validation": H.bind(
            pj2_validation_relative, with_logical=True, immutable=True
        ),
        "scale4_core": H.bind(
            "src/rc_aslo_xf/dino_rcde_h0_component_scale4_v1.py"
        ),
        "base_pj2_reducer": H.bind(
            "programs/reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py"
        ),
        "reducer": H.bind(
            "programs/reduce_dino_rcde_h0_component_scale4_confirmation_v1.py"
        ),
        "validator": H.bind(
            "programs/validate_dino_rcde_h0_component_scale4_confirmation_v1.py"
        ),
        "freezer": H.bind(
            "programs/freeze_dino_rcde_h0_component_scale4_confirmation_authority_v36.py"
        ),
        "launcher": H.bind(
            "slurm/dino_rcde_h0_component_scale4_confirmation_v36.sbatch"
        ),
        "test_core": H.bind("tests/test_dino_rcde_h0_component_scale4_v1.py"),
        "test_reducer": H.bind(
            "tests/test_reduce_dino_rcde_h0_component_scale4_confirmation_v1.py"
        ),
        "test_authority": H.bind(
            "tests/test_freeze_dino_rcde_h0_component_scale4_confirmation_authority_v36.py"
        ),
        "base_pj2_tests": H.bind(
            "tests/test_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py"
        ),
    }
    value: dict[str, Any] = {
        "schema_version": "rc_h0_component_scale4_confirmation_authority_v36_20260824",
        "status": STATUS,
        "claim_level": "N582_INTERNAL_C128_CONFIRMATION_ONLY",
        "scale4_reduction_authorized": True,
        "semantic_join_authorized": True,
        "model_load_authorized": False,
        "model_forward_authorized": False,
        "model_backward_authorized": False,
        "model_update_authorized": False,
        "training_authorized": False,
        "alternative_scale_count": 0,
        "threshold_or_hold_scan_authorized": False,
        "protected_access_authorized": False,
        "full_gallery_retrieval_or_ownership_claim_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "population_contract": {
            "source_query_count": 594,
            "development_execution_ordinals_excluded": list(range(12)),
            "confirmation_query_count": 582,
            "candidate_count": 74496,
            "fold_query_counts": {"1": 149, "2": 140, "3": 145, "4": 148},
            "track_query_counts": {
                "difficult": 58,
                "new_difficult_train": 24,
                "outcome": 500,
            },
            "supergroup_count": 48,
        },
        "scale_contract": {
            "scale": 4.0,
            "formula": "S_INIT + 4.0 * (S_TRACK_H - S_INIT)",
            "candidate_level_before_identity_reduction": True,
            "binary64_fixed_operation_order": True,
        },
        "gate_contract": prereg["gate_contract"],
        "output": "results/dino_rcde_h0_component_scale4_confirmation_v1/result.json",
        "validation_output": (
            "results/dino_rcde_h0_component_scale4_confirmation_validation_v1/result.json"
        ),
        "resource_contract": {
            "partition": "cpuonly",
            "gpus": 0,
            "cpus_per_task": 4,
            "memory_megabytes": 16000,
            "walltime_seconds": 3600,
        },
        "bindings": bindings,
    }
    value["logical_sha256"] = H.logical(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTHORITY)
    args = parser.parse_args()
    require(
        args.output.resolve(strict=False) == (ROOT / AUTHORITY).resolve(strict=False),
        "V36 authority output drift",
    )
    value = build()
    if args.output.exists():
        require(read(str(AUTHORITY)) == value, "existing V36 authority drift")
    else:
        H.atomic(args.output, value)
    print(json.dumps({"status": STATUS, "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
