#!/usr/bin/env python3
"""Postseal E0 reducer for fresh-D1 NATIVE7 A/B/C actions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib, json, math, os, tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT_V1_20260904.md"
)
A0_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1"
ACTIONS = A0_ROOT / "actions.json"
A0_VALIDATION = A0_ROOT / "independent_validation.json"
J1_ROOT = (
    ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1"
)
J1_EVAL = J1_ROOT / "J1_EVAL32_LABEL_LEDGER.json"
J1_VALIDATION = J1_ROOT / "independent_validation.json"
COMPARATOR_ROOT = ROOT / "results/routea_matched_three_arm_common3_native7_crossfit_v1"
COMPARATOR = COMPARATOR_ROOT / "result.json"
COMPARATOR_VALIDATION = COMPARATOR_ROOT / "independent_validation.json"
CANDIDATE_RESULT = (
    ROOT
    / "results/routea_matched_three_arm_n2_current_runtime_d1_candidate_recall_gate_v1/result.json"
)
CANDIDATE_VALIDATION = CANDIDATE_RESULT.parent / "independent_validation.json"
GALLERY = (
    ROOT.parents[2]
    / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
)
REPAIR_REGISTRY = ROOT / "registry/gallery_identity_repair_v1.json"
REPAIR_RUNTIME = ROOT / "src/rc_aslo_xf/gallery_identity_repair.py"
REPAIR_CONTRACT = (
    ROOT / "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"
)
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1"
OUT = OUT_ROOT / "result.json"
VERSION = "routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1_20260904"
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
PATHS = ("REAL", "C_BIND")
GO = "N2_FRESH_D1_CANDIDATE_BOUND_ACTION_CONDITIONAL_GO"
RETRIEVAL_ONLY = "N2_FRESH_D1_RETRIEVAL_ONLY_NO_CANDIDATE_BINDING"
NO_GO = "N2_FRESH_D1_ACTION_CONDITIONAL_NO_GO"
SPATIAL_NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_CONDITIONAL_SPATIAL_CONTROL_CONTRACT"
EXPECTED = {
    CONTRACT: "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79",
    COMPARATOR: "591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3",
    COMPARATOR_VALIDATION: "1a7b08edfad7823db62f86c5f0b213106daead6f2ddba2d299102b69e546d8df",
    CANDIDATE_RESULT: "0544287c2e3ef327143fa5619f28d5df50ad2a5835004717bde38776aeb81b3d",
    CANDIDATE_VALIDATION: "26506c7c5bff8f76ade79625f9744efd4c46dac8bc573f8504ca53dbc1be60d1",
    GALLERY: "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    REPAIR_REGISTRY: "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f",
    REPAIR_RUNTIME: "995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d",
    REPAIR_CONTRACT: "867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650",
}


class ReducerError(RuntimeError):
    pass


def req(x: bool, m: str) -> None:
    if not x:
        raise ReducerError(m)


def sha(p: Path) -> str:
    req(p.is_file() and not p.is_symlink(), f"missing {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canon(x: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            x, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def logical(x: Mapping[str, Any]) -> str:
    y = dict(x)
    y.pop("logical_sha256", None)
    return canon(y)


def read(p: Path) -> dict[str, Any]:
    x = json.loads(p.read_text())
    req(isinstance(x, dict), f"json root {p}")
    return x


def atomic(p: Path, x: Mapping[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    s = (
        json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    if p.exists():
        req(p.read_text() == s, "immutable result drift")
        return
    fd, n = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".partial", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(s)
            f.flush()
            os.fsync(f.fileno())
        os.link(n, p)
    finally:
        Path(n).unlink(missing_ok=True)


def corrected_labels() -> tuple[str, ...]:
    import torch
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from rc_aslo_xf.n2_corrected_d1_runtime_v1 import (
        corrected_labels_from_legacy,
        validate_corrected_identity_axis,
    )

    labels = corrected_labels_from_legacy(
        torch.load(GALLERY, map_location="cpu", weights_only=False, mmap=True)["setids"]
    )
    a = validate_corrected_identity_axis(labels)
    req(
        a["corrected_identity_count"] == 5412
        and a["corrected_axis_sha256"]
        == "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
        "corrected axis",
    )
    return labels


def summarize(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    req(len(cases) > 0, "empty summary")
    n = len(cases)
    base = sum(bool(x["base_correct"]) for x in cases)
    final = sum(bool(x["final_correct"]) for x in cases)
    rescue = sum((not x["base_correct"]) and x["final_correct"] for x in cases)
    breaks = sum(x["base_correct"] and (not x["final_correct"]) for x in cases)
    switch = sum(x["decision"] == "SWITCH" for x in cases)
    return {
        "query_count": n,
        "base_top1": base,
        "base_R@1": base / n,
        "base_MRR": sum(1 / int(x["base_target_rank"]) for x in cases) / n,
        "final_top1": final,
        "final_R@1": final / n,
        "final_MRR": sum(1 / int(x["final_target_rank"]) for x in cases) / n,
        "rescue": rescue,
        "break": breaks,
        "retained_correct": sum(
            x["base_correct"] and x["final_correct"] for x in cases
        ),
        "retained_wrong": sum(
            (not x["base_correct"]) and (not x["final_correct"]) for x in cases
        ),
        "wrong_to_wrong": sum(
            (not x["base_correct"])
            and (not x["final_correct"])
            and x["decision"] == "SWITCH"
            for x in cases
        ),
        "switch_count": switch,
        "hold_count": n - switch,
        "correctness_increment": final - base,
    }


def grouped(cases: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups = defaultdict(list)
    for row in cases:
        groups[str(row[key])].append(row)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def action_cases(
    action_records: list[dict[str, Any]],
    labels_by_query: Mapping[str, Mapping[str, Any]],
    arm: str,
    path: str,
) -> list[dict[str, Any]]:
    out = []
    for record in action_records:
        q = str(record["query_id"])
        label = labels_by_query[q]
        target = int(label["target_candidate_position"])
        rows = list(map(int, record["candidate_physical_rows"]))
        req(
            label["target_state"] != "TARGET_ABSENT"
            and rows[target] == label["target_representative_physical_row"]
            and record["base_winner_position"] == label["base_winner_position"]
            and canon(rows)
            == label["targetfree_record_binding"]["candidate_axis_sha256"],
            f"action/label binding {q}",
        )
        action = record["actions"][arm][path]
        base_order = list(map(int, record["base_ranking_positions"]))
        final_order = list(map(int, action["final_ranking_positions"]))
        req(
            set(base_order) == set(final_order) == set(range(128))
            and action["final_ranking_physical_rows"] == [rows[i] for i in final_order],
            f"ranking {q}",
        )
        br = base_order.index(target) + 1
        fr = final_order.index(target) + 1
        out.append(
            {
                "query_id": q,
                "execution_ordinal": int(record["execution_ordinal"]),
                "canonical_heldout_fold": int(label["canonical_heldout_fold"]),
                "canonical_track": str(label["canonical_track"]),
                "target_candidate_position": target,
                "target_representative_physical_row": rows[target],
                "base_target_rank": br,
                "final_target_rank": fr,
                "base_correct": br == 1,
                "final_correct": fr == 1,
                "decision": action["decision"],
                "proposed_challenger_position": int(
                    action["proposed_challenger_position"]
                ),
                "proposed_challenger_physical_row": int(
                    action["proposed_challenger_physical_row"]
                ),
                "wrong_to_wrong": br != 1
                and fr != 1
                and action["decision"] == "SWITCH",
            }
        )
    return out


def base_cases(
    action_records: list[dict[str, Any]],
    labels_by_query: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Construct the D1-alone endpoint without consulting any local arm."""
    out = []
    for record in action_records:
        query_id = str(record["query_id"])
        label = labels_by_query[query_id]
        target = int(label["target_candidate_position"])
        rows = list(map(int, record["candidate_physical_rows"]))
        base_order = list(map(int, record["base_ranking_positions"]))
        req(
            label["target_state"] != "TARGET_ABSENT"
            and rows[target] == label["target_representative_physical_row"]
            and record["base_winner_position"] == label["base_winner_position"]
            and canon(rows)
            == label["targetfree_record_binding"]["candidate_axis_sha256"]
            and set(base_order) == set(range(128)),
            f"D1 base/label binding {query_id}",
        )
        rank = base_order.index(target) + 1
        out.append(
            {
                "query_id": query_id,
                "execution_ordinal": int(record["execution_ordinal"]),
                "canonical_heldout_fold": int(label["canonical_heldout_fold"]),
                "canonical_track": str(label["canonical_track"]),
                "target_candidate_position": target,
                "target_representative_physical_row": rows[target],
                "base_target_rank": rank,
                "final_target_rank": rank,
                "base_correct": rank == 1,
                "final_correct": rank == 1,
                "decision": "HOLD",
                "proposed_challenger_position": None,
                "proposed_challenger_physical_row": None,
                "wrong_to_wrong": False,
            }
        )
    return out


def retention(
    real: list[dict[str, Any]], control: list[dict[str, Any]]
) -> dict[str, Any]:
    rr = {x["query_id"] for x in real if (not x["base_correct"]) and x["final_correct"]}
    cr = {
        x["query_id"] for x in control if (not x["base_correct"]) and x["final_correct"]
    }
    return {
        "real_rescue_count": len(rr),
        "cbind_rescue_count": len(cr),
        "retained_real_rescue_count": len(rr & cr),
        "cbind_rescue_retention": len(rr & cr) / len(rr) if rr else None,
    }


def decide(gates: Mapping[str, bool]) -> str:
    if all(gates.values()):
        return GO
    retrieval = all(
        gates[name]
        for name in (
            "G1_strictly_beats_RAW_C_28",
            "G2_MRR_beats_fixed_and_D1",
            "G3_no_regret",
            "G4_each_fold_nonnegative_vs_RAW_C",
            "G5_fresh_D1_recall_not_below_RAW",
            "G8_all_replay_and_binding_checks",
        )
    )
    binding = (
        gates["G6_beats_D1_and_C_BIND"]
        and gates["G7_real_rescue_and_C_BIND_destruction"]
    )
    return RETRIEVAL_ONLY if retrieval and not binding else NO_GO


def build() -> dict[str, Any]:
    for p, h in EXPECTED.items():
        req(sha(p) == h, f"fixed hash {p}")
    actions, av = read(ACTIONS), read(A0_VALIDATION)
    req(
        actions.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_READY"
        and actions.get("logical_sha256") == logical(actions)
        and av.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_VALIDATED"
        and av.get("actions_sha256") == sha(ACTIONS)
        and av.get("actions_logical_sha256") == actions["logical_sha256"]
        and av.get("logical_sha256") == logical(av)
        and all(av.get("checks", {}).values()),
        "A0 authority",
    )
    evals, j1v = read(J1_EVAL), read(J1_VALIDATION)
    req(
        evals.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_EVAL32_LABEL_LEDGER_READY"
        and evals.get("role") == "EVAL"
        and evals.get("population", {}).get("target_absent_count") == 0
        and j1v.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_VALIDATED"
        and j1v.get("eval32_ledger_sha256") == sha(J1_EVAL)
        and j1v.get("eval32_ledger_logical_sha256") == evals["logical_sha256"]
        and j1v.get("logical_sha256") == logical(j1v)
        and all(j1v.get("checks", {}).values()),
        "J1 EVAL authority",
    )
    labels = corrected_labels()
    labels_by_query = {str(x["query_id"]): x for x in evals["records"]}
    action_records = actions["records"]
    req(
        len(labels_by_query) == len(action_records) == 32
        and set(labels_by_query) == {str(x["query_id"]) for x in action_records},
        "query join",
    )
    for x in evals["records"]:
        req(
            labels[int(x["target_physical_row_provenance"])] == x["target_identity"],
            "target provenance",
        )
    d1 = base_cases(action_records, labels_by_query)
    evaluations = {}
    for arm in ARMS:
        real = action_cases(action_records, labels_by_query, arm, "REAL")
        control = action_cases(action_records, labels_by_query, arm, "C_BIND")
        evaluations[arm] = {
            "REAL": {
                "summary": summarize(real),
                "by_fold": grouped(real, "canonical_heldout_fold"),
                "by_track": grouped(real, "canonical_track"),
                "actions": real,
            },
            "C_BIND": {
                "summary": summarize(control),
                "by_fold": grouped(control, "canonical_heldout_fold"),
                "by_track": grouped(control, "canonical_track"),
                "actions": control,
            },
            "rescue_retention": retention(real, control),
        }
    comp, compv = read(COMPARATOR), read(COMPARATOR_VALIDATION)
    req(
        comp.get("status")
        == "ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_COMPLETE"
        and comp.get("logical_sha256") == logical(comp)
        and compv.get("status")
        == "ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION_PASS"
        and compv.get("producer_result_sha256") == sha(COMPARATOR)
        and compv.get("logical_sha256") == logical(compv)
        and all(compv.get("checks", {}).values()),
        "comparator",
    )
    fallback = comp["evaluations"]["NATIVE7"]["C_PAIRED"]
    fixed_summary = fallback["real"]
    req(
        fixed_summary["base_top1"] == 25
        and fixed_summary["final_top1"] == 28
        and fixed_summary["rescue"] == 3
        and fixed_summary["break"] == 0
        and fixed_summary["switch_count"] == 4
        and fixed_summary["wrong_to_wrong"] == 1
        and abs(fixed_summary["final_MRR"] - 0.9014136904761905) <= 1e-15,
        "fixed comparator metrics",
    )
    fallback_by_query = {str(x["query_id"]): x for x in fallback["actions"]}
    req(set(fallback_by_query) == set(labels_by_query), "comparator query axis")
    c = evaluations["C_PAIRED"]
    cs = c["REAL"]["summary"]
    cc = c["C_BIND"]["summary"]
    ds = summarize(d1)
    fold_deltas = {}
    for fold in sorted(
        {int(x["canonical_heldout_fold"]) for x in c["REAL"]["actions"]}
    ):
        rows = [
            x for x in c["REAL"]["actions"] if int(x["canonical_heldout_fold"]) == fold
        ]
        delta = sum(
            int(x["final_correct"])
            - int(fallback_by_query[x["query_id"]]["final_correct"])
            for x in rows
        )
        fold_deltas[str(fold)] = {
            "query_count": len(rows),
            "correctness_delta_sum": delta,
            "nonnegative": delta >= 0,
        }
    fresh_recall = sum(
        x["target_candidate_position"] is not None for x in evals["records"]
    )
    raw_recall = sum(int(x["target_position"]) >= 0 for x in fallback["actions"])
    ret = c["rescue_retention"]
    gates = {
        "G1_strictly_beats_RAW_C_28": cs["final_top1"] > 28,
        "G2_MRR_beats_fixed_and_D1": cs["final_MRR"] >= 0.9014136904761905
        and cs["final_MRR"] >= ds["final_MRR"],
        "G3_no_regret": cs["rescue"] > cs["break"] and cs["break"] <= 1,
        "G4_each_fold_nonnegative_vs_RAW_C": all(
            x["nonnegative"] for x in fold_deltas.values()
        ),
        "G5_fresh_D1_recall_not_below_RAW": fresh_recall >= raw_recall,
        "G6_beats_D1_and_C_BIND": cs["final_top1"] > ds["final_top1"]
        and cs["final_top1"] > cc["final_top1"]
        and cs["correctness_increment"] > cc["correctness_increment"],
        "G7_real_rescue_and_C_BIND_destruction": ret["real_rescue_count"] >= 1
        and ret["cbind_rescue_retention"] is not None
        and ret["cbind_rescue_retention"] < 1.0,
        "G8_all_replay_and_binding_checks": True,
    }
    status = decide(gates)
    candidate, candidatev = read(CANDIDATE_RESULT), read(CANDIDATE_VALIDATION)
    fixed235 = candidate.get("metrics", {}).get("fixed_r5_old_d1_wrong_235", {})
    deep22 = candidate.get("metrics", {}).get("fixed_r5_old_deep_tail_22", {})
    req(
        candidate.get("status") == "N2_CANDIDATE_RECALL_NO_GO"
        and candidatev.get("status")
        == "N2_CANDIDATE_RECALL_NO_GO_INDEPENDENTLY_VALIDATED"
        and candidatev.get("result_sha256") == sha(CANDIDATE_RESULT)
        and all(candidatev.get("checks", {}).values())
        and fixed235.get("query_count") == 235
        and fixed235.get("fresh_d1", {}).get("c128_target_coverage_count") == 190
        and deep22.get("query_count") == 22
        and deep22.get("fresh_d1_c128_recovered_count") == 0
        and candidatev.get("fixed235_fresh_d1_c128_coverage_count") == 190
        and candidatev.get("deep22_fresh_d1_c128_recovered_count") == 0,
        "candidate recall boundary",
    )
    return {
        "version": VERSION,
        "status": status,
        "claim_level": "INTERNAL_OPENED_TARGET_PRESENT_CONDITIONAL_ACTION_EVALUATION_NOT_END_TO_END_OR_SPATIAL_OWNERSHIP",
        "primary_arm": "C_PAIRED",
        "control_arms": ["A_ALL", "B_QUERY"],
        "d1_alone": {
            "summary": ds,
            "by_fold": grouped(d1, "canonical_heldout_fold"),
            "by_track": grouped(d1, "canonical_track"),
            "actions": d1,
        },
        "evaluations": evaluations,
        "fixed_RAW_C_comparator": {
            "source_family": "NATIVE7",
            "source_arm": "C_PAIRED",
            "summary": fixed_summary,
            "actions": fallback["actions"],
            "result_sha256": sha(COMPARATOR),
            "validation_sha256": sha(COMPARATOR_VALIDATION),
        },
        "fold_gate_vs_RAW_C": fold_deltas,
        "candidate_recall": {
            "status": candidate["status"],
            "validation_status": candidatev["status"],
            "fixed235_fresh_d1_c128_coverage_count": fixed235["fresh_d1"][
                "c128_target_coverage_count"
            ],
            "fixed235_query_count": fixed235["query_count"],
            "deep22_recovered_count": deep22["fresh_d1_c128_recovered_count"],
            "end_to_end_claim_authorized": False,
            "product_or_external_scoring_authorized": False,
        },
        "gate_checks": gates,
        "conditional_action_status": status,
        "bindings": {
            "contract_sha256": sha(CONTRACT),
            "a0_actions_sha256": sha(ACTIONS),
            "a0_actions_logical_sha256": actions["logical_sha256"],
            "a0_validation_sha256": sha(A0_VALIDATION),
            "a0_validation_logical_sha256": av["logical_sha256"],
            "j1_eval_ledger_sha256": sha(J1_EVAL),
            "j1_eval_ledger_logical_sha256": evals["logical_sha256"],
            "j1_public_validation_sha256": sha(J1_VALIDATION),
            "j1_public_validation_logical_sha256": j1v["logical_sha256"],
            "fixed_comparator_result_sha256": sha(COMPARATOR),
            "fixed_comparator_validation_sha256": sha(COMPARATOR_VALIDATION),
            "candidate_recall_result_sha256": sha(CANDIDATE_RESULT),
            "candidate_recall_validation_sha256": sha(CANDIDATE_VALIDATION),
            "corrected_axis_sha256": "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
            "reducer_sha256": sha(Path(__file__).resolve()),
        },
        "access": {
            "j1_eval_ledger_open_count": 1,
            "j1_train_ledger_open_count": 0,
            "pair_label_or_fixed_pair_open_count": 0,
            "optimizer_or_training_result_mutation_count": 0,
            "target_identity_read_count": 32,
            "target_position_read_count": 32,
            "model_update_count": 0,
            "external_read_count": 0,
            "sealed_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": SPATIAL_NEXT if status == GO else None,
        "logical_sha256": "",
    }


def selftest() -> None:
    cases = [
        {
            "base_correct": True,
            "final_correct": True,
            "base_target_rank": 1,
            "final_target_rank": 1,
            "decision": "HOLD",
        },
        {
            "base_correct": False,
            "final_correct": True,
            "base_target_rank": 2,
            "final_target_rank": 1,
            "decision": "SWITCH",
        },
        {
            "base_correct": False,
            "final_correct": False,
            "base_target_rank": 3,
            "final_target_rank": 3,
            "decision": "HOLD",
        },
    ]
    s = summarize(cases)
    req(
        s["base_top1"] == 1
        and s["final_top1"] == 2
        and s["rescue"] == 1
        and s["break"] == 0
        and s["hold_count"] == 2,
        "summary fixture",
    )
    g = {
        "G1_strictly_beats_RAW_C_28": True,
        "G2_MRR_beats_fixed_and_D1": True,
        "G3_no_regret": True,
        "G4_each_fold_nonnegative_vs_RAW_C": True,
        "G5_fresh_D1_recall_not_below_RAW": True,
        "G6_beats_D1_and_C_BIND": False,
        "G7_real_rescue_and_C_BIND_destruction": False,
        "G8_all_replay_and_binding_checks": True,
    }
    req(decide(g) == RETRIEVAL_ONLY, "priority fixture")
    print(json.dumps({"status": "E0_REDUCER_SYNTHETIC_SELF_TEST_PASS"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-self-test", action="store_true")
    args = ap.parse_args()
    if args.synthetic_self_test:
        selftest()
        return
    out = build()
    out["logical_sha256"] = logical(out)
    atomic(OUT, out)
    print(
        json.dumps(
            {"status": out["status"], "gates": out["gate_checks"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
