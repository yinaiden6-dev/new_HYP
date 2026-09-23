from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))

import validate_dino_rcde_gx_formal_r0_science_v1 as V
import reduce_dino_rcde_gx_formal_r0_science_v1 as R
import reduce_dino_rcde_gx_formal_r0_science_v1 as P


def _records() -> list[dict]:
    rows = []
    tracks = ("outcome", "difficult", "new_difficult_train", "outcome")
    for fold in V.FOLDS:
        for slot in range(8):
            raw_correct = slot >= 4
            row = {
                "outer_fold": fold,
                "execution_ordinal": (fold - 1) * 8 + slot,
                "group_sha256": f"{(fold - 1) * 8 + slot:064x}",
                "track": tracks[(fold - 1) % len(tracks)],
                "raw_correct": raw_correct,
                "raw_target_margin": 1.0 if raw_correct else -1.0,
                "action": "RELATIVE_NULL_HOLD" if raw_correct else "SWITCH",
                "deployed_target_margin": 1.0,
                "final_correct": True,
                "rescue": not raw_correct,
                "break": False,
                "correctness_increment": 0.0 if raw_correct else 1.0,
                "raw_wrong_final_direction": not raw_correct,
                "target_z_minus_rival_z": 1.0,
                "target_real_minus_null": {name: 1.0 for name in V.NULLS},
                "null_eligibility": {name: True for name in V.NULLS},
                "invariants": {
                    "candidate_swap_exact": True,
                    "candidate_reorder_exact": True,
                    "pair_zero_sum_exact": True,
                    "forced_ineligible_hold_exact": True,
                    "natural_hold_preserves_raw_exact": True,
                },
                "record_sha256": f"{slot + 8 * fold:064x}",
                "correctness_increment": float(not raw_correct),
            }
            row["record_sha256"] = V.canonical_sha256(
                {key: value for key, value in row.items() if key != "record_sha256"}
            )
            rows.append(row)
    return rows


def test_independent_summary_serialization_matches_reducer_contract() -> None:
    records = _records()
    for row in records:
        row["record_sha256"] = R.canonical_sha256(
            {key: value for key, value in row.items() if key != "record_sha256"}
        )
    production = R.summarize_records(records)
    folds = V._fold_summaries(records)
    tracks = V._track_summaries(records)
    bootstrap = V._bootstrap(records)
    totals = V._totals(records)
    invariants = {
        "record_count": len(records),
        **{
            f"{name}_count": len(records)
            for name in records[0]["invariants"]
        },
        "all_pass": True,
    }
    eligibility = production["null_eligibility_summary"]
    gates = V._gates(
        folds, tracks, invariants, bootstrap, eligibility, totals
    )
    assert folds == production["fold_summaries"]
    assert tracks == production["track_summaries"]
    assert bootstrap == production["bootstrap"]
    assert totals == production["totals"]
    assert gates == production["gates"]


def test_independent_bootstrap_and_all_registered_gates_pass() -> None:
    records = _records()
    folds = V._fold_summaries(records)
    tracks = V._track_summaries(records)
    invariants = {
        "record_count": 32,
        **{f"{name}_count": 32 for name in records[0]["invariants"]},
        "all_pass": True,
    }
    eligibility = {
        "silent_denominator_drop_count": 0,
    }
    bootstrap = V._bootstrap(records)
    totals = V._totals(records)
    gates = V._gates(
        folds, tracks, invariants, bootstrap, eligibility, totals
    )
    assert all(row["seed"] == 17 for row in bootstrap.values())
    assert all(row["replicates"] == 10_000 for row in bootstrap.values())
    assert all(row["lower_strictly_positive"] for row in bootstrap.values())
    assert gates["all_gates_pass"] is True


def test_break_or_nonpositive_causal_fold_fails_closed() -> None:
    records = _records()
    records[4]["final_correct"] = False
    records[4]["break"] = True
    for row in records[:8]:
        row["target_real_minus_null"]["P_COORD"] = -1.0
    folds = V._fold_summaries(records)
    tracks = V._track_summaries(records)
    invariants = {
        "record_count": 32,
        **{f"{name}_count": 32 for name in records[0]["invariants"]},
        "all_pass": True,
    }
    bootstrap = V._bootstrap(records)
    gates = V._gates(
        folds,
        tracks,
        invariants,
        bootstrap,
        {"silent_denominator_drop_count": 0},
        V._totals(records),
    )
    assert gates["raw_correct_retention_exactly_16_of_16"] is False
    assert gates["every_fold_target_real_minus_each_null_positive"] is False
    assert gates["all_gates_pass"] is False


def test_independent_summary_serialization_matches_production_schema() -> None:
    records = _records()
    auxiliary = {
        (int(row["outer_fold"]), int(row["execution_ordinal"])): (True, False)
        for row in records
    }
    production = P.summarize_records(records, auxiliary_t_by_record=auxiliary)
    folds = V._fold_summaries(records)
    tracks = V._track_summaries(records)
    bootstrap = V._bootstrap(records)
    totals = V._totals(records)
    assert folds == production["fold_summaries"]
    assert tracks == production["track_summaries"]
    assert bootstrap == production["bootstrap"]
    assert totals == production["totals"]


def test_validator_is_independent_and_cli_is_authority_bound() -> None:
    source = Path(V.__file__).read_text(encoding="utf-8")
    assert "import reduce_dino_rcde_gx_formal_r0_science_v1" not in source
    assert "dino_rcde_gx_relational_head_cache_v1 import" not in source
    assert "make_candidate_branch_relational_cache" not in source
    assert "--authority" in source
    assert "--result" in source
    assert "--output" in source
    assert '"--training-root"' not in source
