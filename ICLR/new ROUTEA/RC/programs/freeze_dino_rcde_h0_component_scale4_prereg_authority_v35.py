#!/usr/bin/env python3
"""Freeze Scale-4 intent before full-C128 semantic outcomes exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = Path(
    "registry/h0/h0_component_scale4_prereg_authority_v35_20260824.json"
)
STATUS = "H0_COMPONENT_SCALE4_SUCCESSOR_PREREGISTERED_NOT_EXECUTABLE"


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
    canary_validation_relative = (
        "results/dino_rcde_h0_component_pair_canary12_postjoin_validation_v1/result.json"
    )
    track_r_relative = "results/dino_rcde_track_r_scientific_result_v1/result.json"
    track_r_validation_relative = (
        "results/dino_rcde_track_r_scientific_validation_v1/result.json"
    )
    full_authority_relative = (
        "registry/h0/h0_c128_target_free_full_prejoin_authority_v2_20260824.json"
    )
    handoff_relative = (
        "registry/h0/h0_c128_target_free_postjoin_handoff_authority_v1_20260824.json"
    )
    canary = read(canary_relative)
    canary_validation = read(canary_validation_relative)
    track_r = read(track_r_relative)
    track_r_validation = read(track_r_validation_relative)
    full_authority = read(full_authority_relative)
    handoff = read(handoff_relative)
    require(
        canary.get("status") == "H0_COMPONENT_PAIR_CANARY12_POSTJOIN_STOP"
        and canary.get("logical_sha256") == H.logical(canary)
        and canary.get("init_correct_count") == canary.get("track_h_correct_count") == 5
        and canary.get("transition_counts", {}).get("rescue") == 0
        and canary.get("transition_counts", {}).get("break") == 0
        and canary.get("margin_improved_count") == 9
        and canary_validation.get("validation_pass") is True
        and canary_validation.get("result_sha256") == H.file_sha(ROOT / canary_relative),
        "canary12 development receipt drift",
    )
    require(
        track_r.get("decision_A") == "DINO_SPECIFIC_REFERENCE_EVIDENCE_NO_GO"
        and track_r.get("decision_B") == "P_LOCK_REGIONAL_INCREMENT_NO_INCREMENT"
        and track_r_validation.get("validation_pass") is True,
        "Track-R permanent negative evidence drift",
    )
    require(
        full_authority.get("status")
        == "H0_C128_TARGET_FREE_PREJOIN_STAGE_A_EXECUTION_AUTHORIZED"
        and full_authority.get("logical_sha256") == H.logical(full_authority)
        and full_authority.get("scientific_GO_or_NO_GO") is None
        and handoff.get("status")
        == "H0_C128_TARGET_FREE_POSTJOIN_AUTOMATIC_HANDOFF_AUTHORIZED"
        and handoff.get("logical_sha256") == H.logical(handoff),
        "isolated full-C128 lineage drift",
    )
    bindings = {
        "contract": H.bind(
            "plan/DINO_RCDE_H0_COMPONENT_SCALE4_SUCCESSOR_CONTRACT_V1_20260824.md"
        ),
        "canary12_result": H.bind(
            canary_relative, with_logical=True, immutable=True
        ),
        "canary12_validation": H.bind(
            canary_validation_relative, with_logical=True, immutable=True
        ),
        "track_r_negative_result": H.bind(
            track_r_relative, with_logical=True, immutable=True
        ),
        "track_r_negative_validation": H.bind(
            track_r_validation_relative, with_logical=True, immutable=True
        ),
        "full_c128_prejoin_authority": H.bind(
            full_authority_relative, with_logical=True, immutable=True
        ),
        "full_c128_postjoin_handoff_authority": H.bind(
            handoff_relative, with_logical=True, immutable=True
        ),
        "freezer": H.bind(
            "programs/freeze_dino_rcde_h0_component_scale4_prereg_authority_v35.py"
        ),
        "test_authority": H.bind(
            "tests/test_freeze_dino_rcde_h0_component_scale4_prereg_authority_v35.py"
        ),
    }
    value: dict[str, Any] = {
        "schema_version": "rc_h0_component_scale4_prereg_authority_v35_20260824",
        "status": STATUS,
        "claim_level": "RESULT_BLIND_SUCCESSOR_INTENT_ONLY",
        "scale": 4.0,
        "score_formula": "S_INIT + 4.0 * (S_TRACK_H - S_INIT)",
        "development_execution_ordinals": list(range(12)),
        "confirmation_execution_ordinals_excluded": list(range(12)),
        "expected_confirmation_query_count": 582,
        "alternative_scale_count": 0,
        "threshold_or_hold_scan_authorized": False,
        "execution_authorized": False,
        "semantic_join_authorized": False,
        "scientific_reduction_authorized": False,
        "model_load_authorized": False,
        "model_forward_authorized": False,
        "model_backward_authorized": False,
        "model_update_authorized": False,
        "protected_access_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "gate_contract": {
            "group_balanced_top1_gain_vs_init_minimum": 0.03,
            "top1_and_mrr_bootstrap_lower_strictly_positive": True,
            "top1_and_mrr_group_signflip_p_strictly_below": 0.05,
            "positive_fold_count_minimum": 3,
            "rescue_strictly_above_break_vs_init": True,
            "difficult_rescue_at_least_break_vs_init": True,
            "top1_mrr_and_net_rescue_strictly_above_track_h": True,
        },
        "future_input_requirements": {
            "full_50_shard_aggregate_immutable": True,
            "full_aggregate_independent_validation_pass": True,
            "pj1_exact_metadata_join_immutable": True,
            "pj1_independent_validation_pass": True,
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
        "V35 authority output drift",
    )
    value = build()
    if args.output.exists():
        require(read(str(AUTHORITY)) == value, "existing V35 authority drift")
    else:
        H.atomic(args.output, value)
    print(json.dumps({"status": STATUS, "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
