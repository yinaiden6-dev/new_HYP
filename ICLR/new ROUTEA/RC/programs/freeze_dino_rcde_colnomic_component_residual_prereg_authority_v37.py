#!/usr/bin/env python3
"""Freeze unit-weight ColNomic/component residual intent before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = Path(
    "registry/h0/colnomic_component_residual_prereg_authority_v37_20260824.json"
)
STATUS = "COLNOMIC_COMPONENT_RESIDUAL_PREREGISTERED_NOT_EXECUTABLE"


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {relative}")
    return value


def build() -> dict[str, Any]:
    canary_relative = (
        "results/dino_rcde_h0_component_pair_canary12_postjoin_v1/result.json"
    )
    scale4_relative = "results/dino_rcde_h0_component_scale4_confirmation_v1/result.json"
    raw_validation_relative = (
        "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_validation.json"
    )
    pj1_relative = "results/dino_rcde_h0_c128_target_free_postjoin_pj1_v1/result.json"
    pj1_validation_relative = (
        "results/dino_rcde_h0_c128_target_free_postjoin_pj1_validation_v1/result.json"
    )
    canary = read(canary_relative)
    scale4 = read(scale4_relative)
    raw_validation = read(raw_validation_relative)
    pj1 = read(pj1_relative)
    pj1_validation = read(pj1_validation_relative)
    require(
        canary.get("status") == "H0_COMPONENT_PAIR_CANARY12_POSTJOIN_STOP"
        and canary.get("logical_sha256") == H.logical(canary)
        and scale4.get("status") == "COMPONENT_SCALE4_INTERNAL_CONFIRMATION_NO_GO"
        and scale4.get("logical_sha256") == H.logical(scale4)
        and scale4.get("scale4_vs_init", {}).get("transition_counts", {}).get("rescue") == 10
        and scale4.get("scale4_vs_init", {}).get("transition_counts", {}).get("break") == 1,
        "component development/negative evidence drift",
    )
    require(
        raw_validation.get("status") == "RCDE_SR0_MT_I0_PREJOIN_INDEPENDENT_VALIDATION_PASS"
        and raw_validation.get("prejoin_ledger_sha256")
        == H.file_sha(
            ROOT / "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_ledger.pt"
        )
        and pj1.get("status") == "H0_C128_TARGET_FREE_POSTJOIN_PJ1_READY"
        and pj1.get("logical_sha256") == H.logical(pj1)
        and pj1_validation.get("validation_pass") is True,
        "RAW/PJ1 source qualification drift",
    )
    bindings = {
        "contract": H.bind(
            "plan/DINO_RCDE_COLNOMIC_COMPONENT_RESIDUAL_FUSION_CONTRACT_V1_20260824.md"
        ),
        "development_canary12": H.bind(
            canary_relative, with_logical=True, immutable=True
        ),
        "scale4_negative_result": H.bind(
            scale4_relative, with_logical=True, immutable=True
        ),
        "raw_prejoin_ledger": H.bind(
            "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_ledger.pt",
            immutable=True,
        ),
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
        "freezer": H.bind(
            "programs/freeze_dino_rcde_colnomic_component_residual_prereg_authority_v37.py"
        ),
        "test_authority": H.bind(
            "tests/test_freeze_dino_rcde_colnomic_component_residual_prereg_authority_v37.py"
        ),
    }
    value: dict[str, Any] = {
        "schema_version": "rc_colnomic_component_residual_prereg_authority_v37_20260824",
        "status": STATUS,
        "claim_level": "RESULT_BLIND_UNIT_RESIDUAL_INTENT_ONLY",
        "residual_weight": 1.0,
        "score_formula": "S_RAW_COLNOMIC + (S_TRACK_H_COMPONENT - S_INIT_COMPONENT)",
        "alternative_weight_count": 0,
        "normalization_threshold_hold_scan_authorized": False,
        "development_execution_ordinals": list(range(12)),
        "confirmation_execution_ordinals_excluded": list(range(12)),
        "expected_confirmation_query_count": 582,
        "execution_authorized": False,
        "semantic_reduction_authorized": False,
        "model_load_authorized": False,
        "model_forward_authorized": False,
        "model_backward_authorized": False,
        "model_update_authorized": False,
        "protected_access_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "gate_contract": {
            "group_balanced_top1_gain_vs_raw_minimum": 0.03,
            "top1_and_mrr_bootstrap_lower_strictly_positive": True,
            "top1_and_mrr_group_signflip_p_strictly_below": 0.05,
            "positive_fold_count_minimum": 3,
            "rescue_strictly_above_break_vs_raw": True,
            "difficult_rescue_at_least_break_vs_raw": True,
            "absolute_top1_and_mrr_strictly_above_raw": True,
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
        "V37 authority output drift",
    )
    value = build()
    if args.output.exists():
        require(read(str(AUTHORITY)) == value, "existing V37 authority drift")
    else:
        H.atomic(args.output, value)
    print(json.dumps({"status": STATUS, "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
