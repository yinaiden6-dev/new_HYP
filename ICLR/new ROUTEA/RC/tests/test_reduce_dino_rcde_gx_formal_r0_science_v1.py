from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))

import reduce_dino_rcde_gx_formal_r0_science_v1 as R


def records():
    output = []
    execution = 0
    for fold in R.FOLDS:
        for member in range(8):
            raw_correct = member < 4
            track = "difficult" if member in (0, 4) else (
                "new_difficult_train" if member in (1, 5) else "outcome"
            )
            target = {
                "REAL": 2.0,
                "C_BIND": 0.2,
                "P_COORD": 0.3,
                "N_REGION_RESAMPLE": 0.4,
                "null_logmeanexp_CPN": 0.31,
                "Z_CPN": 1.69,
            }
            rival = {
                "REAL": 0.5,
                "C_BIND": 0.2,
                "P_COORD": 0.3,
                "N_REGION_RESAMPLE": 0.4,
                "null_logmeanexp_CPN": 0.31,
                "Z_CPN": 0.19,
            }
            row = {
                "outer_fold": fold,
                "execution_ordinal": execution,
                "query_id": f"Q{execution}",
                "episode_id": f"E{execution}",
                "group_sha256": f"{execution:064x}",
                "track": track,
                "raw_correct": raw_correct,
                "raw_target_margin": 0.5 if raw_correct else -0.5,
                "candidate_energies": {"target": target, "rival": rival},
                "target_real_minus_null": {
                    "C_BIND": 1.8,
                    "P_COORD": 1.7,
                    "N_REGION_RESAMPLE": 1.6,
                },
                "rival_real_minus_null": {
                    "C_BIND": 0.3,
                    "P_COORD": 0.2,
                    "N_REGION_RESAMPLE": 0.1,
                },
                "target_z_minus_rival_z": 1.5,
                "final_target_margin": 2.0 if raw_correct else 1.0,
                "raw_wrong_final_direction": not raw_correct,
                "action": "RELATIVE_NULL_HOLD" if raw_correct else "SWITCH",
                "deployed_target_margin": 0.5 if raw_correct else 1.0,
                "final_correct": True,
                "rescue": not raw_correct,
                "break": False,
                "correctness_increment": 0.0 if raw_correct else 1.0,
                "null_eligibility": {name: True for name in R.CONTROLS},
                "invariants": {
                    "candidate_swap_exact": True,
                    "candidate_reorder_exact": True,
                    "pair_zero_sum_exact": True,
                    "forced_ineligible_hold_exact": True,
                    "natural_hold_preserves_raw_exact": True,
                },
            }
            row["record_sha256"] = R.canonical_sha256(row)
            output.append(row)
            execution += 1
    return output


def test_exact_contract_population_is_go_and_t_is_diagnostic_only() -> None:
    rows = records()
    t = {
        (row["outer_fold"], row["execution_ordinal"]): (True, False)
        for row in rows
    }
    value = R.summarize_records(rows, auxiliary_t_by_record=t)
    assert value["gates"]["all_gates_pass"] is True
    assert value["totals"] == {
        "query_count": 32,
        "raw_correct_count": 16,
        "raw_wrong_count": 16,
        "raw_wrong_final_direction_count": 16,
        "raw_correct_retained_count": 16,
        "rescue_count": 16,
        "break_count": 0,
        "net_rescue": 16,
    }
    assert set(value["bootstrap"]) == set(R.BOOTSTRAP_ENDPOINTS)
    assert all(item["replicates"] == 10_000 for item in value["bootstrap"].values())
    assert all(item["lower_strictly_positive"] for item in value["bootstrap"].values())
    assert value["auxiliary_t_diagnostic"]["present_count"] == 32
    assert all(
        value["auxiliary_t_diagnostic"][field] is False
        for field in ("enters_calibration", "enters_loss", "enters_action", "enters_gate", "enters_bootstrap")
    )


def test_bootstrap_is_exactly_pcg64_seed17_linear_10000() -> None:
    first = R.registered_bootstrap(records(), "target_z_minus_rival_z")
    second = R.registered_bootstrap(records(), "target_z_minus_rival_z")
    assert first == second
    assert first["rng"] == "numpy.PCG64"
    assert first["seed"] == 17
    assert first["replicates"] == 10_000
    assert first["quantile_method"] == "linear"


def test_one_fold_without_net_rescue_is_fail_closed() -> None:
    rows = records()
    for row in rows:
        if row["outer_fold"] == 2 and not row["raw_correct"]:
            row["raw_wrong_final_direction"] = False
            row["action"] = "RELATIVE_NULL_HOLD"
            row["deployed_target_margin"] = row["raw_target_margin"]
            row["final_correct"] = False
            row["rescue"] = False
            row["correctness_increment"] = 0.0
            row["record_sha256"] = R.canonical_sha256(
                {key: value for key, value in row.items() if key != "record_sha256"}
            )
    value = R.summarize_records(rows)
    assert value["fold_summaries"]["2"]["gates"]["net_rescue_positive"] is False
    assert value["gates"]["all_gates_pass"] is False


def test_hold_preserves_raw_and_cannot_count_overlay_only_final_direction() -> None:
    rows = records()
    row = next(item for item in rows if not item["raw_correct"])
    # The learned overlay is positive, but the registered action is HOLD, so
    # the deployed result remains the RAW-wrong negative margin.
    assert row["final_target_margin"] > 0.0
    row["action"] = "RELATIVE_NULL_HOLD"
    row["deployed_target_margin"] = row["raw_target_margin"]
    row["final_correct"] = False
    row["raw_wrong_final_direction"] = False
    row["rescue"] = False
    row["correctness_increment"] = 0.0
    row["record_sha256"] = R.canonical_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"}
    )
    value = R.summarize_records(rows)
    assert value["totals"]["raw_wrong_final_direction_count"] == 15
    assert value["totals"]["rescue_count"] == 15


def test_inconsistent_hold_outcome_fails_closed() -> None:
    rows = records()
    row = next(item for item in rows if not item["raw_correct"])
    row["action"] = "RELATIVE_NULL_HOLD"
    row["record_sha256"] = R.canonical_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"}
    )
    with pytest.raises(R.R0ScienceError, match="action/outcome/null/invariant drift"):
        R.summarize_records(rows)


def test_population_or_record_hash_drift_aborts() -> None:
    with pytest.raises(R.R0ScienceError, match="Balanced-32 population drift"):
        R.summarize_records(records()[:-1])
    rows = records()
    rows[0]["query_id"] = "tampered"
    with pytest.raises(R.R0ScienceError, match="record SHA drift"):
        R.summarize_records(rows)
