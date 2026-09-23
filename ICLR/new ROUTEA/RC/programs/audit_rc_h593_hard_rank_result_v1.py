#!/usr/bin/env python3
"""Replay a post-join loss decomposition from the sealed H593 hard-rank result.

This reads only result.json and validation.json. It does not fit a model,
reselect parameters, inspect caches, or produce a HOLD/SWITCH decision.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_h593_hard_rank_v1"
RESULT_SHA = "d9f32d6f7befcca4663ad5e001bd45d460d9dc706fa8c30fbab732393b9e582f"
VALIDATION_SHA = "66cfa01ffe469872fa6f0b9551aa9edbf317b20d32ceaf56486cc8d2d0ea0e71"
OLD, NEW = "COST1_FULL", "RANK_HARD6"
DENOMINATOR = 570
CATEGORIES = ("still_correct", "still_wrong", "gain", "loss")
TOLERANCE = 1e-12


def need(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def close(actual, expected, message):
    need(math.isfinite(actual) and math.isfinite(expected)
         and abs(actual - expected) <= TOLERANCE, message)


def category(row):
    old, new = row["target_is_top"][OLD], row["target_is_top"][NEW]
    need(type(old) is bool and type(new) is bool, "non-boolean target-top indicator")
    return ("still_correct" if new else "loss") if old else ("gain" if new else "still_wrong")


def summarize(rows):
    old = math.fsum(row["ranking_losses"][OLD]["hard_softplus"] for row in rows)
    new = math.fsum(row["ranking_losses"][NEW]["hard_softplus"] for row in rows)
    improvement = math.fsum(row["ranking_losses"][OLD]["hard_softplus"] -
                            row["ranking_losses"][NEW]["hard_softplus"] for row in rows)
    close(improvement, old - new, "group loss difference and per-query differences disagree")
    return {
        "count": len(rows),
        "old_hard_loss_sum": old,
        "new_hard_loss_sum": new,
        "hard_loss_improvement_sum": improvement,
        "old_hard_loss_per_present570": old / DENOMINATOR,
        "new_hard_loss_per_present570": new / DENOMINATOR,
        "hard_loss_improvement_per_present570": improvement / DENOMINATOR,
        "query_ids": [row["query_id"] for row in rows],
        "original_query_ids": [row["original_query_id"] for row in rows],
    }


def analyze(result, validation):
    need(result["status"] == "H593_HARD_RANK_ALL593_COMPLETE", "result not complete")
    need(validation["status"] == "HARD_RANK_ALL_COUNTS_PASS", "validation not passed")
    need(validation["authority"] == result["authority"], "authority binding mismatch")
    need(result["action_trained"] is False and result["new_deployed_accuracy"] is None
         and result["external_GO"] is False, "rank-only scientific boundary changed")
    rows = sorted(result["rows"], key=lambda row: row["query_id"])
    need(len(rows) == 593 and len({row["query_id"] for row in rows}) == 593,
         "expected 593 unique queries")
    present = [row for row in rows if row["target_present"]]
    eligible = [row for row in rows if row["ranking_eligible"]]
    need(len(present) == DENOMINATOR and len(eligible) == 144, "population count mismatch")
    need(sum(row["RAW_correct"] for row in rows) == 426, "RAW correct count mismatch")
    for row in rows:
        need(row["ranking_eligible"] == (row["target_present"] and not row["RAW_correct"]),
             "eligibility definition mismatch")
        need(not row["RAW_correct"] or row["target_present"], "RAW correct target absent")
        for model in (OLD, NEW):
            losses = row["ranking_losses"][model]
            value = losses["hard_softplus"]
            if not row["target_present"]:
                need(value is None, "target-absent loss must be null")
            elif row["RAW_correct"]:
                need(value == 0.0, "RAW-correct conditional loss must be zero")
            else:
                need(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0,
                     "invalid hard loss")
                margin = losses["target_margin"]
                need(isinstance(margin, (int, float)) and math.isfinite(margin), "invalid target margin")
                # target_margin = target minus strongest wrong; stable softplus(-margin).
                expected = max(-margin, 0.0) + math.log1p(math.exp(-abs(margin)))
                close(value, expected, "recorded hard loss disagrees with recorded margin")
    groups = {name: summarize([row for row in eligible if category(row) == name])
              for name in CATEGORIES}
    total = summarize(present)
    total.pop("query_ids")
    total.pop("original_query_ids")
    total.update({"ranking_eligible_count": len(eligible), "RAW_correct_zero_loss_count": 426,
                  "target_absent_excluded_count": 23, "denominator": DENOMINATOR})
    need(sum(group["count"] for group in groups.values()) == 144, "group partition incomplete")
    for field in ("old_hard_loss_sum", "new_hard_loss_sum", "hard_loss_improvement_sum"):
        close(math.fsum(group[field] for group in groups.values()), total[field],
              "group sum does not reproduce total: " + field)
    old_top = groups["still_correct"]["count"] + groups["loss"]["count"]
    new_top = groups["still_correct"]["count"] + groups["gain"]["count"]
    need(old_top == result["summary"][OLD]["target_top_among144"] == 104, "old top count mismatch")
    need(new_top == result["summary"][NEW]["target_top_among144"] == 102, "new top count mismatch")
    comparison = result["comparisons"][OLD]
    for field, count in (("ranking_gained", groups["gain"]["count"]),
                         ("ranking_lost", groups["loss"]["count"]),
                         ("ranking_net", new_top - old_top)):
        need(count == comparison[field], "sealed comparison mismatch: " + field)
    close(total["old_hard_loss_per_present570"], result["summary"][OLD]["hard_loss_per_present570"],
          "old total differs from sealed summary")
    close(total["new_hard_loss_per_present570"], result["summary"][NEW]["hard_loss_per_present570"],
          "new total differs from sealed summary")
    unchanged = math.fsum(groups[name]["hard_loss_improvement_sum"]
                          for name in ("still_correct", "still_wrong"))
    changed = math.fsum(groups[name]["hard_loss_improvement_sum"] for name in ("gain", "loss"))
    close(unchanged + changed, total["hard_loss_improvement_sum"], "changed partition sum mismatch")
    improvement = total["hard_loss_improvement_sum"]
    return {
        "status": "H593_HARD_RANK_POST_JOIN_LOSS_DECOMPOSITION_PASS",
        "authority": result["authority"],
        "evidence_level": "Post-join descriptive decomposition of opened grouped OOF ranking results",
        "baseline": OLD, "model": NEW,
        "improvement_sign": "old hard-softplus loss minus new hard-softplus loss; positive is lower new loss",
        "group_scope": "144 RAW-wrong queries with target in RAW C128; ranking is over 127 challengers",
        "category_definition": {"still_correct": "both heads rank target first",
                                "still_wrong": "neither head ranks target first",
                                "gain": "only new head ranks target first",
                                "loss": "only old head ranks target first"},
        "groups": groups, "TOTAL": total,
        "ranking": {"old_target_top": old_top, "new_target_top": new_top,
                    "gained": groups["gain"]["count"], "lost": groups["loss"]["count"],
                    "net": new_top - old_top},
        "decomposition": {
            "unchanged_correctness_count": groups["still_correct"]["count"] + groups["still_wrong"]["count"],
            "unchanged_correctness_improvement_sum": unchanged,
            "changed_correctness_count": groups["gain"]["count"] + groups["loss"]["count"],
            "changed_correctness_improvement_sum": changed,
            "unchanged_correctness_share_of_net_improvement": unchanged / improvement if improvement != 0.0 else None,
        },
        "checks": {"SHA_verified": True, "recorded_margin_loss_agreement": True,
                   "group_sum_matches_total": True, "sealed_summary_agreement": True,
                   "absolute_float_tolerance": TOLERANCE},
        "new_deployed_accuracy": None, "action_trained": False, "external_GO": False,
        "limits": ["This decomposition does not retrain, select, or change any head.",
                   "The 570 denominator includes 426 RAW-correct zero-loss rows and 144 ranking-eligible rows.",
                   "Gain and loss here refer to challenger target-top ranking, not deployed HOLD/SWITCH decisions.",
                   "A lower held-out continuous margin loss need not improve the count of target-top rankings.",
                   "The percentage is a descriptive attribution of the net loss difference, not causal evidence."]
    }


def main():
    result_path, validation_path = OUT / "result.json", OUT / "validation.json"
    result_binding, validation_binding = binding(result_path), binding(validation_path)
    need(result_binding["sha256"] == RESULT_SHA, "sealed result SHA mismatch")
    need(validation_binding["sha256"] == VALIDATION_SHA, "sealed validation SHA mismatch")
    result, validation = json.loads(result_path.read_text()), json.loads(validation_path.read_text())
    need(validation["result"] == result_binding, "validation does not bind this result")
    report = analyze(result, validation)
    report["sources"] = {"result": result_binding, "validation": validation_binding,
                         "audit_program": binding(Path(__file__))}
    path = OUT / "post_join_analysis.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": binding(path),
                      "ranking": report["ranking"], "decomposition": report["decomposition"],
                      "TOTAL": report["TOTAL"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
