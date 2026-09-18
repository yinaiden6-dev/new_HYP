#!/usr/bin/env python3
"""Independent validation of the N2 postseal conditional-action reduction.

This file deliberately does not import the producer.  It reconstructs ranks,
retrieval metrics, causal-control gates, and the terminal status from sealed
inputs and compares them with the published reduction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT / "results/routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1"
)
RESULT = RESULT_ROOT / "result.json"
VALIDATION = RESULT_ROOT / "independent_validation.json"
ACTIONS = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1/actions.json"
)
ACTIONS_VALIDATION = ACTIONS.parent / "independent_validation.json"
J1_EVAL = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1/J1_EVAL32_LABEL_LEDGER.json"
)
J1_VALIDATION = J1_EVAL.parent / "independent_validation.json"
COMPARATOR = (
    ROOT / "results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json"
)
COMPARATOR_VALIDATION = COMPARATOR.parent / "independent_validation.json"
CANDIDATE = (
    ROOT
    / "results/routea_matched_three_arm_n2_current_runtime_d1_candidate_recall_gate_v1/result.json"
)
CANDIDATE_VALIDATION = CANDIDATE.parent / "independent_validation.json"
REDUCER = (
    ROOT
    / "programs/reduce_routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1.py"
)
CONTRACT = (
    ROOT
    / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT_V1_20260904.md"
)
GALLERY = (
    ROOT.parents[2]
    / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
)
REPAIR_REGISTRY = ROOT / "registry/gallery_identity_repair_v1.json"
REPAIR_RUNTIME = ROOT / "src/rc_aslo_xf/gallery_identity_repair.py"
REPAIR_CONTRACT = (
    ROOT / "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"
)

ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
PATHS = ("REAL", "C_BIND")
GO = "N2_FRESH_D1_CANDIDATE_BOUND_ACTION_CONDITIONAL_GO"
RETRIEVAL_ONLY = "N2_FRESH_D1_RETRIEVAL_ONLY_NO_CANDIDATE_BINDING"
NO_GO = "N2_FRESH_D1_ACTION_CONDITIONAL_NO_GO"
VALIDATED = "N2_FRESH_D1_MATCHED_THREE_ARM_POSTSEAL_EVALUATION_VALIDATED"
SPATIAL_NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_CONDITIONAL_SPATIAL_CONTROL_CONTRACT"
FIXED = {
    CONTRACT: "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79",
    COMPARATOR: "591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3",
    COMPARATOR_VALIDATION: "1a7b08edfad7823db62f86c5f0b213106daead6f2ddba2d299102b69e546d8df",
    CANDIDATE: "0544287c2e3ef327143fa5619f28d5df50ad2a5835004717bde38776aeb81b3d",
    CANDIDATE_VALIDATION: "26506c7c5bff8f76ade79625f9744efd4c46dac8bc573f8504ca53dbc1be60d1",
    GALLERY: "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    REPAIR_REGISTRY: "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f",
    REPAIR_RUNTIME: "995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d",
    REPAIR_CONTRACT: "867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    copy = dict(value)
    copy.pop("logical_sha256", None)
    return canonical(copy)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"non-object JSON: {path}")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    if path.exists():
        require(path.read_text() == text, "immutable validation drift")
        return
    fd, partial = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, path)
    finally:
        Path(partial).unlink(missing_ok=True)


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(bool(rows), "empty metric population")
    count = len(rows)
    base = sum(bool(x["base_correct"]) for x in rows)
    final = sum(bool(x["final_correct"]) for x in rows)
    switched = sum(x["decision"] == "SWITCH" for x in rows)
    return {
        "query_count": count,
        "base_top1": base,
        "base_R@1": base / count,
        "base_MRR": sum(1 / int(x["base_target_rank"]) for x in rows) / count,
        "final_top1": final,
        "final_R@1": final / count,
        "final_MRR": sum(1 / int(x["final_target_rank"]) for x in rows) / count,
        "rescue": sum(not x["base_correct"] and x["final_correct"] for x in rows),
        "break": sum(x["base_correct"] and not x["final_correct"] for x in rows),
        "retained_correct": sum(x["base_correct"] and x["final_correct"] for x in rows),
        "retained_wrong": sum(
            not x["base_correct"] and not x["final_correct"] for x in rows
        ),
        "wrong_to_wrong": sum(
            not x["base_correct"]
            and not x["final_correct"]
            and x["decision"] == "SWITCH"
            for x in rows
        ),
        "switch_count": switched,
        "hold_count": count - switched,
        "correctness_increment": final - base,
    }


def partitions(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: metrics(group) for name, group in sorted(groups.items())}


def reconstruct(
    action_records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    arm: str,
    path: str,
) -> list[dict[str, Any]]:
    rows_out = []
    for record in action_records:
        query_id = str(record["query_id"])
        label = labels[query_id]
        axis = list(map(int, record["candidate_physical_rows"]))
        target = int(label["target_candidate_position"])
        require(label["target_state"] != "TARGET_ABSENT", f"absent target: {query_id}")
        require(
            axis[target] == int(label["target_representative_physical_row"]),
            f"target binding: {query_id}",
        )
        require(
            canonical(axis)
            == label["targetfree_record_binding"]["candidate_axis_sha256"],
            f"axis binding: {query_id}",
        )
        require(
            int(record["base_winner_position"]) == int(label["base_winner_position"]),
            f"winner binding: {query_id}",
        )
        action = record["actions"][arm][path]
        base_order = list(map(int, record["base_ranking_positions"]))
        final_order = list(map(int, action["final_ranking_positions"]))
        require(
            set(base_order) == set(final_order) == set(range(128)),
            f"rank permutation: {query_id}",
        )
        require(
            list(map(int, action["final_ranking_physical_rows"]))
            == [axis[i] for i in final_order],
            f"row rank binding: {query_id}",
        )
        base_rank = base_order.index(target) + 1
        final_rank = final_order.index(target) + 1
        rows_out.append(
            {
                "query_id": query_id,
                "execution_ordinal": int(record["execution_ordinal"]),
                "canonical_heldout_fold": int(label["canonical_heldout_fold"]),
                "canonical_track": str(label["canonical_track"]),
                "target_candidate_position": target,
                "target_representative_physical_row": axis[target],
                "base_target_rank": base_rank,
                "final_target_rank": final_rank,
                "base_correct": base_rank == 1,
                "final_correct": final_rank == 1,
                "decision": str(action["decision"]),
                "proposed_challenger_position": int(
                    action["proposed_challenger_position"]
                ),
                "proposed_challenger_physical_row": int(
                    action["proposed_challenger_physical_row"]
                ),
                "wrong_to_wrong": base_rank != 1
                and final_rank != 1
                and action["decision"] == "SWITCH",
            }
        )
    return rows_out


def reconstruct_base(
    action_records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Independently reconstruct D1-alone without reading an arm action."""
    rows_out = []
    for record in action_records:
        query_id = str(record["query_id"])
        label = labels[query_id]
        axis = list(map(int, record["candidate_physical_rows"]))
        target = int(label["target_candidate_position"])
        base_order = list(map(int, record["base_ranking_positions"]))
        require(label["target_state"] != "TARGET_ABSENT", f"absent target: {query_id}")
        require(
            axis[target] == int(label["target_representative_physical_row"])
            and canonical(axis)
            == label["targetfree_record_binding"]["candidate_axis_sha256"]
            and int(record["base_winner_position"])
            == int(label["base_winner_position"])
            and set(base_order) == set(range(128)),
            f"D1 base binding: {query_id}",
        )
        rank = base_order.index(target) + 1
        rows_out.append(
            {
                "query_id": query_id,
                "execution_ordinal": int(record["execution_ordinal"]),
                "canonical_heldout_fold": int(label["canonical_heldout_fold"]),
                "canonical_track": str(label["canonical_track"]),
                "target_candidate_position": target,
                "target_representative_physical_row": axis[target],
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
    return rows_out


def rescue_retention(
    real: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    real_ids = {
        x["query_id"] for x in real if not x["base_correct"] and x["final_correct"]
    }
    control_ids = {
        x["query_id"] for x in control if not x["base_correct"] and x["final_correct"]
    }
    return {
        "real_rescue_count": len(real_ids),
        "cbind_rescue_count": len(control_ids),
        "retained_real_rescue_count": len(real_ids & control_ids),
        "cbind_rescue_retention": (
            len(real_ids & control_ids) / len(real_ids) if real_ids else None
        ),
    }


def terminal(gates: Mapping[str, bool]) -> str:
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


def independently_corrected_labels() -> tuple[str, ...]:
    import sys
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from rc_aslo_xf.n2_corrected_d1_runtime_v1 import (
        corrected_labels_from_legacy,
        validate_corrected_identity_axis,
    )

    legacy = torch.load(GALLERY, map_location="cpu", weights_only=False, mmap=True)[
        "setids"
    ]
    labels = corrected_labels_from_legacy(legacy)
    audit = validate_corrected_identity_axis(labels)
    require(audit["corrected_identity_count"] == 5412, "corrected identity count")
    require(
        audit["corrected_axis_sha256"]
        == "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
        "corrected identity axis",
    )
    return labels


def independently_recompute() -> dict[str, Any]:
    for path, expected in FIXED.items():
        require(file_sha(path) == expected, f"fixed input hash: {path}")
    result = load(RESULT)
    require(result.get("logical_sha256") == logical(result), "result logical hash")
    require(
        result.get("bindings", {}).get("reducer_sha256") == file_sha(REDUCER),
        "reducer binding",
    )

    actions, action_validation = load(ACTIONS), load(ACTIONS_VALIDATION)
    require(
        actions.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_READY",
        "A0 status",
    )
    require(actions.get("logical_sha256") == logical(actions), "A0 logical")
    require(
        action_validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_VALIDATED",
        "A0 validation status",
    )
    require(
        action_validation.get("actions_sha256") == file_sha(ACTIONS),
        "A0 physical binding",
    )
    require(
        action_validation.get("actions_logical_sha256") == actions["logical_sha256"],
        "A0 logical binding",
    )
    require(
        action_validation.get("logical_sha256") == logical(action_validation),
        "A0 validation logical",
    )
    require(all(action_validation.get("checks", {}).values()), "A0 checks")

    ledger, ledger_validation = load(J1_EVAL), load(J1_VALIDATION)
    require(
        ledger.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_EVAL32_LABEL_LEDGER_READY",
        "J1 status",
    )
    require(
        ledger.get("role") == "EVAL"
        and ledger.get("population", {}).get("target_absent_count") == 0,
        "J1 role/population",
    )
    require(ledger.get("logical_sha256") == logical(ledger), "J1 logical")
    require(
        ledger_validation.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_VALIDATED",
        "J1 validation status",
    )
    require(
        ledger_validation.get("eval32_ledger_sha256") == file_sha(J1_EVAL),
        "J1 physical binding",
    )
    require(
        ledger_validation.get("eval32_ledger_logical_sha256")
        == ledger["logical_sha256"],
        "J1 logical binding",
    )
    require(
        ledger_validation.get("logical_sha256") == logical(ledger_validation),
        "J1 validation logical",
    )
    require(all(ledger_validation.get("checks", {}).values()), "J1 checks")

    action_records = list(actions["records"])
    labels = {str(row["query_id"]): row for row in ledger["records"]}
    require(len(action_records) == len(labels) == 32, "EVAL32 size")
    require(
        {str(row["query_id"]) for row in action_records} == set(labels), "query-id join"
    )
    corrected = independently_corrected_labels()
    for row in ledger["records"]:
        require(
            corrected[int(row["target_physical_row_provenance"])]
            == row["target_identity"],
            f"corrected target provenance: {row['query_id']}",
        )

    d1 = reconstruct_base(action_records, labels)
    expected_evaluations = {}
    for arm in ARMS:
        real = reconstruct(action_records, labels, arm, "REAL")
        control = reconstruct(action_records, labels, arm, "C_BIND")
        expected_evaluations[arm] = {
            "REAL": {
                "summary": metrics(real),
                "by_fold": partitions(real, "canonical_heldout_fold"),
                "by_track": partitions(real, "canonical_track"),
                "actions": real,
            },
            "C_BIND": {
                "summary": metrics(control),
                "by_fold": partitions(control, "canonical_heldout_fold"),
                "by_track": partitions(control, "canonical_track"),
                "actions": control,
            },
            "rescue_retention": rescue_retention(real, control),
        }

    comparator, comparator_validation = load(COMPARATOR), load(COMPARATOR_VALIDATION)
    require(
        comparator.get("logical_sha256") == logical(comparator), "comparator logical"
    )
    require(
        comparator_validation.get("producer_result_sha256") == file_sha(COMPARATOR),
        "comparator validation binding",
    )
    fixed = comparator["evaluations"]["NATIVE7"]["C_PAIRED"]
    fixed_summary = fixed["real"]
    require(
        fixed_summary["base_top1"] == 25
        and fixed_summary["final_top1"] == 28
        and fixed_summary["rescue"] == 3
        and fixed_summary["break"] == 0
        and fixed_summary["switch_count"] == 4
        and fixed_summary["wrong_to_wrong"] == 1
        and math.isclose(fixed_summary["final_MRR"], 0.9014136904761905, abs_tol=1e-15),
        "frozen RAW+C metrics",
    )
    fixed_by_query = {str(row["query_id"]): row for row in fixed["actions"]}
    require(set(fixed_by_query) == set(labels), "fixed comparator query axis")

    c_real = expected_evaluations["C_PAIRED"]["REAL"]
    c_control = expected_evaluations["C_PAIRED"]["C_BIND"]
    d1_summary = metrics(d1)
    fold_gate = {}
    for fold_name, fold_metrics in c_real["by_fold"].items():
        fold_rows = [
            x
            for x in c_real["actions"]
            if str(x["canonical_heldout_fold"]) == fold_name
        ]
        delta = sum(
            int(x["final_correct"])
            - int(fixed_by_query[x["query_id"]]["final_correct"])
            for x in fold_rows
        )
        fold_gate[fold_name] = {
            "query_count": fold_metrics["query_count"],
            "correctness_delta_sum": delta,
            "nonnegative": delta >= 0,
        }
    real_summary = c_real["summary"]
    control_summary = c_control["summary"]
    retention = expected_evaluations["C_PAIRED"]["rescue_retention"]
    raw_recall = sum(int(row["target_position"]) >= 0 for row in fixed["actions"])
    fresh_recall = sum(
        row["target_candidate_position"] is not None for row in ledger["records"]
    )
    gates = {
        "G1_strictly_beats_RAW_C_28": real_summary["final_top1"] > 28,
        "G2_MRR_beats_fixed_and_D1": real_summary["final_MRR"] >= 0.9014136904761905
        and real_summary["final_MRR"] >= d1_summary["final_MRR"],
        "G3_no_regret": real_summary["rescue"] > real_summary["break"]
        and real_summary["break"] <= 1,
        "G4_each_fold_nonnegative_vs_RAW_C": all(
            row["nonnegative"] for row in fold_gate.values()
        ),
        "G5_fresh_D1_recall_not_below_RAW": fresh_recall >= raw_recall,
        "G6_beats_D1_and_C_BIND": real_summary["final_top1"] > d1_summary["final_top1"]
        and real_summary["final_top1"] > control_summary["final_top1"]
        and real_summary["correctness_increment"]
        > control_summary["correctness_increment"],
        "G7_real_rescue_and_C_BIND_destruction": retention["real_rescue_count"] >= 1
        and retention["cbind_rescue_retention"] is not None
        and retention["cbind_rescue_retention"] < 1.0,
        "G8_all_replay_and_binding_checks": True,
    }
    expected_status = terminal(gates)

    candidate, candidate_validation = load(CANDIDATE), load(CANDIDATE_VALIDATION)
    fixed235 = candidate.get("metrics", {}).get("fixed_r5_old_d1_wrong_235", {})
    deep22 = candidate.get("metrics", {}).get("fixed_r5_old_deep_tail_22", {})
    require(
        candidate.get("status") == "N2_CANDIDATE_RECALL_NO_GO",
        "candidate recall status",
    )
    require(
        candidate_validation.get("status")
        == "N2_CANDIDATE_RECALL_NO_GO_INDEPENDENTLY_VALIDATED",
        "candidate recall validation",
    )
    require(
        candidate_validation.get("result_sha256") == file_sha(CANDIDATE),
        "candidate recall binding",
    )
    require(
        all(candidate_validation.get("checks", {}).values()), "candidate recall checks"
    )
    require(
        fixed235.get("query_count") == 235
        and fixed235.get("fresh_d1", {}).get("c128_target_coverage_count") == 190
        and deep22.get("query_count") == 22
        and deep22.get("fresh_d1_c128_recovered_count") == 0
        and candidate_validation.get("fixed235_fresh_d1_c128_coverage_count") == 190
        and candidate_validation.get("deep22_fresh_d1_c128_recovered_count") == 0,
        "candidate recall counts",
    )

    checks = {
        "result_logical_hash": True,
        "fixed_input_hashes": True,
        "a0_target_free_action_authority": True,
        "physical_eval_only_j1_authority": True,
        "query_and_candidate_axis_join": True,
        "corrected_5412_axis_rebuilt": True,
        "d1_metrics_recomputed": result["d1_alone"]
        == {
            "summary": d1_summary,
            "by_fold": partitions(d1, "canonical_heldout_fold"),
            "by_track": partitions(d1, "canonical_track"),
            "actions": d1,
        },
        "all_three_arms_both_paths_recomputed": result["evaluations"]
        == expected_evaluations,
        "fixed_raw_c_comparator_recomputed": result["fixed_RAW_C_comparator"]["summary"]
        == fixed_summary
        and result["fixed_RAW_C_comparator"]["actions"] == fixed["actions"],
        "fold_gate_recomputed": result["fold_gate_vs_RAW_C"] == fold_gate,
        "eight_gates_recomputed": result["gate_checks"] == gates,
        "status_priority_recomputed": result["status"] == expected_status
        and result["conditional_action_status"] == expected_status,
        "candidate_recall_no_go_preserved": result["candidate_recall"][
            "fixed235_fresh_d1_c128_coverage_count"
        ]
        == fixed235["fresh_d1"]["c128_target_coverage_count"]
        and result["candidate_recall"]["fixed235_query_count"]
        == fixed235["query_count"]
        and result["candidate_recall"]["deep22_recovered_count"]
        == deep22["fresh_d1_c128_recovered_count"]
        and result["candidate_recall"]["end_to_end_claim_authorized"] is False,
        "primary_and_controls_frozen": result["primary_arm"] == "C_PAIRED"
        and result["control_arms"] == ["A_ALL", "B_QUERY"],
        "no_forbidden_access": result["access"]["j1_train_ledger_open_count"] == 0
        and result["access"]["pair_label_or_fixed_pair_open_count"] == 0
        and result["access"]["optimizer_or_training_result_mutation_count"] == 0
        and result["access"]["model_update_count"] == 0,
        "claim_boundary": result["scientific_GO_or_NO_GO"] is None
        and result["ownership_GO_or_NO_GO"] is None
        and result["automatic_stage_advance"] is False,
        "next_stage_exact": result["next_authorized_stage"]
        == (SPATIAL_NEXT if expected_status == GO else None),
    }
    require(all(checks.values()), "independent postseal validation failed")
    return {
        "version": "routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_independent_validation_v1_20260904",
        "status": VALIDATED,
        "claim_level": "INDEPENDENT_VALIDATION_OF_INTERNAL_CONDITIONAL_ACTION_EVALUATION_ONLY",
        "conditional_action_status": expected_status,
        "checks": checks,
        "gate_checks": gates,
        "result_sha256": file_sha(RESULT),
        "result_logical_sha256": result["logical_sha256"],
        "reducer_sha256": file_sha(REDUCER),
        "validator_sha256": file_sha(Path(__file__).resolve()),
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
        "next_authorized_stage": SPATIAL_NEXT if expected_status == GO else None,
        "logical_sha256": "",
    }


def synthetic_self_test() -> None:
    rows = [
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
            "final_target_rank": 2,
            "decision": "SWITCH",
        },
    ]
    summary = metrics(rows)
    require(
        summary["rescue"] == 1
        and summary["break"] == 0
        and summary["wrong_to_wrong"] == 1,
        "metric fixture",
    )
    gates = {
        "G1_strictly_beats_RAW_C_28": True,
        "G2_MRR_beats_fixed_and_D1": True,
        "G3_no_regret": True,
        "G4_each_fold_nonnegative_vs_RAW_C": True,
        "G5_fresh_D1_recall_not_below_RAW": True,
        "G6_beats_D1_and_C_BIND": False,
        "G7_real_rescue_and_C_BIND_destruction": False,
        "G8_all_replay_and_binding_checks": True,
    }
    require(terminal(gates) == RETRIEVAL_ONLY, "status-priority fixture")
    gates["G6_beats_D1_and_C_BIND"] = True
    gates["G7_real_rescue_and_C_BIND_destruction"] = True
    require(terminal(gates) == GO, "GO fixture")
    print(json.dumps({"status": "E0_POSTSEAL_VALIDATOR_SYNTHETIC_SELF_TEST_PASS"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_self_test:
        synthetic_self_test()
        return
    validation = independently_recompute()
    validation["logical_sha256"] = logical(validation)
    write_once(VALIDATION, validation)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "conditional_action_status": validation["conditional_action_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
