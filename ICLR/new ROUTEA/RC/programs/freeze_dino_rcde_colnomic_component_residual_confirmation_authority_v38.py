#!/usr/bin/env python3
"""Freeze executable unit residual fusion authority V38."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = Path(
    "registry/h0/colnomic_component_residual_confirmation_authority_v38_20260824.json"
)
STATUS = "COLNOMIC_COMPONENT_RESIDUAL_INTERNAL_CONFIRMATION_EXECUTION_AUTHORIZED"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {relative}")
    return value


def build() -> dict[str, Any]:
    prereg_relative = "registry/h0/colnomic_component_residual_prereg_authority_v37_20260824.json"
    raw_validation_relative = (
        "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_validation.json"
    )
    pj1_relative = "results/dino_rcde_h0_c128_target_free_postjoin_pj1_v1/result.json"
    pj1_validation_relative = (
        "results/dino_rcde_h0_c128_target_free_postjoin_pj1_validation_v1/result.json"
    )
    prereg = read(prereg_relative)
    raw_validation = read(raw_validation_relative)
    pj1 = read(pj1_relative)
    pj1_validation = read(pj1_validation_relative)
    require(
        prereg.get("status") == "COLNOMIC_COMPONENT_RESIDUAL_PREREGISTERED_NOT_EXECUTABLE"
        and prereg.get("logical_sha256") == H.logical(prereg)
        and prereg.get("residual_weight") == 1.0
        and prereg.get("alternative_weight_count") == 0
        and prereg.get("confirmation_execution_ordinals_excluded") == list(range(12)),
        "V37 preregistration drift",
    )
    raw_relative = "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_ledger.pt"
    require(
        raw_validation.get("status") == "RCDE_SR0_MT_I0_PREJOIN_INDEPENDENT_VALIDATION_PASS"
        and raw_validation.get("prejoin_ledger_sha256") == H.file_sha(ROOT / raw_relative)
        and pj1.get("status") == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_READY"
        and pj1.get("logical_sha256") == H.logical(pj1)
        and pj1.get("query_count") == 594
        and pj1_validation.get("validation_pass") is True
        and pj1_validation.get("result_sha256") == H.file_sha(ROOT / pj1_relative),
        "RAW/PJ1 qualification drift",
    )
    bindings = {
        "contract": H.bind(
            "plan/DINO_RCDE_COLNOMIC_COMPONENT_RESIDUAL_FUSION_CONTRACT_V1_20260824.md"
        ),
        "preregistration_authority": H.bind(
            prereg_relative, with_logical=True, immutable=True
        ),
        "raw_prejoin_ledger": H.bind(raw_relative, immutable=True),
        "raw_prejoin_receipt": H.bind(
            "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_receipt.json",
            with_logical=True,
            immutable=True,
        ),
        "raw_prejoin_validation": H.bind(
            raw_validation_relative, with_logical=True, immutable=True
        ),
        "pj1_result": H.bind(pj1_relative, with_logical=True, immutable=True),
        "pj1_validation": H.bind(
            pj1_validation_relative, with_logical=True, immutable=True
        ),
        "fusion_core": H.bind(
            "src/rc_aslo_xf/dino_rcde_colnomic_component_residual_v1.py"
        ),
        "base_pj2_reducer": H.bind(
            "programs/reduce_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py"
        ),
        "scale4_statistics_reducer": H.bind(
            "programs/reduce_dino_rcde_h0_component_scale4_confirmation_v1.py"
        ),
        "reducer": H.bind(
            "programs/reduce_dino_rcde_colnomic_component_residual_confirmation_v1.py"
        ),
        "validator": H.bind(
            "programs/validate_dino_rcde_colnomic_component_residual_confirmation_v1.py"
        ),
        "freezer": H.bind(
            "programs/freeze_dino_rcde_colnomic_component_residual_confirmation_authority_v38.py"
        ),
        "launcher": H.bind(
            "slurm/dino_rcde_colnomic_component_residual_confirmation_v38.sbatch"
        ),
        "test_core": H.bind(
            "tests/test_dino_rcde_colnomic_component_residual_v1.py"
        ),
        "test_reducer": H.bind(
            "tests/test_reduce_dino_rcde_colnomic_component_residual_confirmation_v1.py"
        ),
        "test_authority": H.bind(
            "tests/test_freeze_dino_rcde_colnomic_component_residual_confirmation_authority_v38.py"
        ),
        "base_pj2_tests": H.bind(
            "tests/test_dino_rcde_h0_c128_target_free_postjoin_pj2_v1.py"
        ),
    }
    value: dict[str, Any] = {
        "schema_version": "rc_colnomic_component_residual_confirmation_authority_v38_20260824",
        "status": STATUS,
        "claim_level": "N582_INTERNAL_NATURAL_C128_INCREMENT_ONLY",
        "residual_reduction_authorized": True,
        "semantic_join_authorized": True,
        "residual_weight": 1.0,
        "alternative_weight_count": 0,
        "normalization_threshold_hold_scan_authorized": False,
        "model_load_authorized": False,
        "model_forward_authorized": False,
        "model_backward_authorized": False,
        "model_update_authorized": False,
        "training_authorized": False,
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
        "score_contract": {
            "formula": "S_RAW_COLNOMIC + (S_TRACK_H_COMPONENT - S_INIT_COMPONENT)",
            "residual_weight": 1.0,
            "candidate_level_before_identity_reduction": True,
            "binary64_fixed_operation_order": True,
        },
        "gate_contract": prereg["gate_contract"],
        "output": "results/dino_rcde_colnomic_component_residual_confirmation_v1/result.json",
        "validation_output": (
            "results/dino_rcde_colnomic_component_residual_confirmation_validation_v1/result.json"
        ),
        "resource_contract": {
            "partitions": ["dev_cpuonly", "cpuonly"],
            "gpus": 0,
            "cpus_per_task": 2,
            "memory_megabytes": 8192,
            "walltime_seconds": 600,
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
        "V38 authority output drift",
    )
    value = build()
    if args.output.exists():
        require(read(str(AUTHORITY)) == value, "existing V38 authority drift")
    else:
        H.atomic(args.output, value)
    print(json.dumps({"status": STATUS, "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
