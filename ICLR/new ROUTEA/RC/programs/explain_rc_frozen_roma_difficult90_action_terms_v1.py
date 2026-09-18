#!/usr/bin/env python3
"""Exact cached explanation of the original frozen difficult90 seven-parameter head.

First reproduce every original REAL R1 action. Then replay six prespecified
interventions: set exactly one derived feature to +0 for all 127 challengers,
preserving the old weights, bias, raw winner, tie rule and switch threshold.
This is an input-feature intervention on an existing frozen action rule, not a
new trained model, pixel intervention, formal action approval or HYP GO.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_frozen_roma_difficult90_action_terms_v1"
PINS = {
    "original_result": ("results/romav2_colnomic_difficult90_frozen_regression_v1/result.json", "bdb962e4b07e6b27a86a35723465141c1f34f4b6dbda8f41810adb5199b79f31"),
    "original_validation": ("results/romav2_colnomic_difficult90_frozen_regression_v1/independent_validation.json", "be8d3a2eb55cf3e7054529c51730b426054720dddab86d7ec8b464275bec6234"),
    "real_prejoin_validation": ("results/romav2_colnomic_difficult90_real_prejoin_v1/validation.json", "28160696964403ee73aec9fb3edebd28b8e2eb56af1c85b4173788e04d6fbf9d"),
    "target_join": ("results/romav2_colnomic_difficult90_regression_manifest_v1/target_join_manifest.json", "f16d9bfb77202c4167966a68067b668ae76d758aba9d04bc28fdfaecf04acc0f"),
    "identity_repair": ("registry/gallery_identity_repair_v1.json", "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f"),
    "frozen_gate": ("src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py", "96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7"),
    "original_reduction_authority": ("registry/romav2_colnomic_difficult90_reduction_authority_v1_20260901.json", "b9cf06c2e2523cde53581a9d079895f1e936f8b9528eb76c04d2277c7478f811"),
    "original_reducer": ("programs/reduce_romav2_colnomic_difficult90_regression_v1.py", "6b9a9e16fc0d8fc6d6fd0835debf4df1fad1221cf9ef7a970a7648cc02630d2a"),
    "original_action_function": ("programs/reduce_romav2_colnomic_new_difficult_sealed_v1.py", "4e92e331e25ebf3dd0c7a62fa202371d84ffde17d078c77f5ee3fb514c642f35"),
}
FEATURE_NAMES = ("standardized_raw_gap", "symmetric_local_score", "symmetric_visibility_mass",
    "symmetric_visibility_normalized_local_similarity", "symmetric_query_spatial_robustness",
    "symmetric_reference_spatial_robustness")
INTERVENTIONS = tuple("SET_FEATURE_TO_ZERO:" + name for name in FEATURE_NAMES)
FIELDS_REPLAYED = ("target_present_c128", "target_position", "base_winner_position",
    "proposed_challenger_position", "raw_action", "effective_action", "base_correct", "final_correct", "wrong_to_wrong")


def need(value, reason):
    if not bool(value):
        raise RuntimeError(reason)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def logical(value):
    return hashlib.sha256(encode({key: item for key, item in value.items() if key != "logical_sha256"})).hexdigest()


def same_float(a, b):
    return a is None and b is None or a is not None and b is not None and float(a).hex() == float(b).hex()


def identity(physical_row):
    # Frozen repair states 5413 physical rows, 5412 corrected identities, and
    # exactly this one duplicate component; all filename collisions were split.
    return 714 if int(physical_row) in (714, 715) else int(physical_row)


def score_action(candidates, raw, axis, winner, logits, target_row, gate):
    best = max(logits, key=lambda position: (logits[position], -axis[position]))
    value = logits[best]
    raw_switch = value > gate.SWITCH_THRESHOLD
    effective_switch = raw_switch and identity(axis[best]) != identity(axis[winner])
    final_position = best if effective_switch else winner
    positions = [position for position, physical in enumerate(axis) if identity(physical) == identity(target_row)]
    target_position = positions[0] if positions else None
    base_correct = identity(axis[winner]) == identity(target_row)
    final_correct = identity(axis[final_position]) == identity(target_row)
    return {"target_present_c128": target_position is not None, "target_position": target_position,
        "base_winner_position": winner, "base_winner_physical_row": axis[winner],
        "proposed_challenger_position": best, "proposed_challenger_physical_row": axis[best],
        "switch_logit": value, "switch_logit_binary64": float(value).hex(),
        "target_switch_logit": None if target_position is None or target_position == winner else logits[target_position],
        "target_switch_logit_binary64": None if target_position is None or target_position == winner else float(logits[target_position]).hex(),
        "raw_action": "SWITCH" if raw_switch else "HOLD", "effective_action": "SWITCH" if effective_switch else "HOLD",
        "final_prediction_position": final_position, "final_prediction_physical_row": axis[final_position],
        "final_prediction_identity_representative": identity(axis[final_position]),
        "base_correct": base_correct, "final_correct": final_correct,
        "wrong_to_wrong": not base_correct and not final_correct and effective_switch}


def r1_summary(actions):
    rescue = [row["query_ordinal"] for row in actions if not row["base_correct"] and row["final_correct"]]
    broken = [row["query_ordinal"] for row in actions if row["base_correct"] and not row["final_correct"]]
    return {"query_count": len(actions), "base_top1": sum(row["base_correct"] for row in actions),
        "final_top1": sum(row["final_correct"] for row in actions),
        "target_present_c128_count": sum(row["target_present_c128"] for row in actions),
        "rescue": len(rescue), "break": len(broken), "rescue_ordinals": rescue, "break_ordinals": broken,
        "switch_count": sum(row["effective_action"] == "SWITCH" for row in actions),
        "wrong_to_wrong": sum(row["wrong_to_wrong"] for row in actions)}


def run(out):
    import torch
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not out.exists(), "APPEND_ONLY_OUTPUT_ALREADY_EXISTS")
    sources = {}
    def checked(name):
        relative, expected = PINS[name]
        path = ROOT / relative
        need(sha(path) == expected, "FROZEN_SOURCE_HASH_DRIFT:" + name)
        sources[name] = {"path": relative, "sha256": expected}
        return path
    for name in PINS:
        checked(name)
    original = json.loads(checked("original_result").read_text())
    validation = json.loads(checked("original_validation").read_text())
    real_validation = json.loads(checked("real_prejoin_validation").read_text())
    target_join = json.loads(checked("target_join").read_text())
    repair = json.loads(checked("identity_repair").read_text())
    authority = json.loads(checked("original_reduction_authority").read_text())
    need(validation["result_sha256"] == PINS["original_result"][1]
         and validation["status"] == "ROMAV2_COLNOMIC_DIFFICULT90_REGRESSION_INDEPENDENT_VALIDATION_PASS"
         and all(validation["checks"].values()), "ORIGINAL_INDEPENDENT_VALIDATION_NOT_CLOSED")
    need(original["authority_sha256"] == PINS["original_reduction_authority"][1]
         and original["logical_sha256"] == logical(original), "ORIGINAL_RESULT_AUTHORITY_OR_LOGICAL_SEAL")
    need(authority["manifest"]["target_join_sha256"] == PINS["target_join"][1]
         and authority["source_sha256"][PINS["frozen_gate"][0]] == PINS["frozen_gate"][1], "ORIGINAL_JOIN_OR_GATE_AUTHORITY_DRIFT")
    need(target_join["logical_sha256"] == logical(target_join)
         and target_join["status"] == "ROMAV2_COLNOMIC_DIFFICULT90_TARGET_JOIN_BOUND_REDUCER_ONLY", "TARGET_JOIN_SEAL")
    duplicates = repair["verified_byte_identical_duplicate_components"]
    need(repair["physical_row_count"] == 5413 and repair["corrected_identity_count"] == 5412
         and len(duplicates) == 1 and sorted(duplicates[0]["physical_rows"]) == [714, 715], "UNVERIFIED_PHYSICAL_IDENTITY_EQUIVALENCE")
    need(real_validation["status"] == "ROMAV2_COLNOMIC_DIFFICULT90_REAL_PREJOIN_VALIDATION_PASS"
         and real_validation["query_count"] == 90 and all(real_validation["checks"].values()), "REAL_PREJOIN_VALIDATION")
    source_rows = {}
    for shard in real_validation["shards"]:
        relative = f"results/romav2_colnomic_difficult90_real_prejoin_v1/shard{shard['shard']:02d}/result.json"
        path = ROOT / relative
        need(shard["pass"] and all(shard["checks"].values()) and sha(path) == shard["sha256"], "REAL_SHARD_SOURCE_DRIFT")
        sources["real_shard_" + str(shard["shard"])] = {"path": relative, "sha256": shard["sha256"]}
        data = json.loads(path.read_text())
        need(data["logical_sha256"] == logical(data) and data["target_role_read_count"] == 0
             and data["target_insertion_count"] == data["model_update_count"] == 0, "REAL_SHARD_TARGET_FREE_SOURCE_SEAL")
        for row in data["rows"]:
            ordinal = int(row["query_ordinal"])
            need(ordinal not in source_rows, "DUPLICATE_SOURCE_QUERY")
            source_rows[ordinal] = row
    targets = {int(row["query_ordinal"]): row for row in target_join["rows"]}
    old_actions = {int(row["query_ordinal"]): row for row in original["actions"]["REAL"]}
    need(sorted(source_rows) == sorted(targets) == sorted(old_actions) == list(range(90)), "ALL90_AXES_REQUIRED")
    spec = importlib.util.spec_from_file_location("rc_original_difficult90_frozen_gate_for_terms", checked("frozen_gate"))
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    need(tuple(gate.FEATURE_NAMES) == FEATURE_NAMES and gate.PARAMETER_COUNT == 7 and gate.WEIGHT.dtype == torch.float64
         and gate.SWITCH_THRESHOLD == 0.0, "FROZEN_SEVEN_PARAMETER_GATE_SCHEMA")
    program_sha = sha(Path(__file__))
    records, all_actions, altered_actions = [], [], {name: [] for name in INTERVENTIONS}
    for ordinal in range(90):
        source, target, old = source_rows[ordinal], targets[ordinal], old_actions[ordinal]
        need(source["query_id"] == target["query_id"] == old["query_id"]
             and source["opened_split"] == target["opened_split"] == old["opened_split"], "ORIGINAL_QUERY_OR_SPLIT_JOIN_DRIFT")
        candidates = {int(row["candidate_position"]): row for row in source["candidates"]}
        need(sorted(candidates) == list(range(128)), "FULL_C128_CANDIDATE_POSITION_AXIS_REQUIRED")
        axis = [int(candidates[i]["physical_row"]) for i in range(128)]
        need(len(set(axis)) == 128 and all(0 <= row < 5413 for row in axis), "PHYSICAL_ROW_AXIS_DRIFT")
        raw = [float(candidates[i]["raw_score"]) for i in range(128)]
        winner = max(range(128), key=lambda i: (raw[i], -axis[i]))
        original_logits, zero_logits, terms = {}, [{} for _ in FEATURE_NAMES], []
        for challenger in range(128):
            if challenger == winner:
                continue
            features = gate.candidate_feature(raw, candidates, challenger, winner)
            products = gate.WEIGHT * features
            value = float(products.sum() + gate.BIAS)
            need(same_float(value, gate.logit(raw, candidates, challenger, winner)), "EXACT_ALL_CHALLENGER_LOGIT_REPLAY_FAILED")
            original_logits[challenger] = value
            changed_values = []
            for feature_index in range(6):
                modified = features.clone()
                modified[feature_index] = 0.0
                altered = float((gate.WEIGHT * modified).sum() + gate.BIAS)
                zero_logits[feature_index][challenger] = altered
                changed_values.append(float(altered).hex())
            terms.append({"challenger_position": challenger, "challenger_physical_row": axis[challenger],
                "features": features.tolist(), "features_binary64": [float(x).hex() for x in features],
                "weighted_terms": products.tolist(), "weighted_terms_binary64": [float(x).hex() for x in products],
                "bias_binary64": float(gate.BIAS).hex(), "original_logit_binary64": float(value).hex(),
                "single_feature_zero_logits_binary64": changed_values})
        action = score_action(candidates, raw, axis, winner, original_logits, target["target_gallery_physical_row"], gate)
        for field in FIELDS_REPLAYED:
            need(action[field] == old[field], "ORIGINAL_ACTION_MISMATCH:" + str(ordinal) + ":" + field)
        for field in ("switch_logit", "target_switch_logit"):
            need(same_float(action[field], old[field]), "ORIGINAL_ACTION_FLOAT_BITS_MISMATCH:" + str(ordinal) + ":" + field)
        implicit_original_final_position = old["proposed_challenger_position"] if old["effective_action"] == "SWITCH" else old["base_winner_position"]
        need(action["final_prediction_position"] == implicit_original_final_position
             and action["base_correct"] == (old["base_rank"] == 1)
             and action["final_correct"] == (old["final_rank"] == 1), "ORIGINAL_FINAL_PREDICTION_OR_R1_REPLAY_FAILED")
        common = {"query_ordinal": ordinal, "query_id": source["query_id"], "opened_split": source["opened_split"]}
        action.update(common)
        all_actions.append(action)
        interventions = {}
        for index, name in enumerate(INTERVENTIONS):
            changed = score_action(candidates, raw, axis, winner, zero_logits[index], target["target_gallery_physical_row"], gate)
            changed.update(common)
            interventions[name] = changed
            altered_actions[name].append(changed)
        selected = next(row for row in terms if row["challenger_position"] == action["proposed_challenger_position"])
        target_terms = next((row for row in terms if row["challenger_position"] == action["target_position"]), None)
        records.append({**common, "target_gallery_physical_row": target["target_gallery_physical_row"],
            "target_identity_equivalent_physical_rows": [714, 715] if int(target["target_gallery_physical_row"]) in (714, 715) else [target["target_gallery_physical_row"]],
            "candidate_physical_rows": axis, "original_action": action, "selected_challenger_terms": selected,
            "target_challenger_terms": target_terms, "all_127_challenger_terms": terms,
            "single_feature_zero_actions": interventions})
    baseline = r1_summary(all_actions)
    need(all(baseline[key] == original["summaries"]["REAL"][key] for key in baseline), "ORIGINAL_ALL90_R1_SUMMARY_DRIFT")
    need(baseline["base_top1"] == 61 and baseline["final_top1"] == 69 and baseline["rescue"] == 8 and baseline["break"] == 0, "ORIGINAL_61_TO_69_EIGHT_RESCUES_NOT_REPRODUCED")
    print(json.dumps({"event": "ORIGINAL_90_ACTIONS_EXACTLY_REPLAYED", "base_top1": 61, "final_top1": 69,
        "rescue": 8, "break": 0, "elapsed_seconds": time.monotonic() - started}), flush=True)
    original_rescues = set(baseline["rescue_ordinals"])
    original_success = {row["query_ordinal"] for row in all_actions if row["final_correct"]}
    summaries = {}
    for name, actions in altered_actions.items():
        summary = r1_summary(actions)
        changed_success = {row["query_ordinal"] for row in actions if row["final_correct"]}
        retained, lost = original_rescues & changed_success, original_rescues - changed_success
        summaries[name] = {**summary, "original_eight_rescues_retained": len(retained), "original_eight_rescues_lost": len(lost),
            "retained_original_rescue_ordinals": sorted(retained), "lost_original_rescue_ordinals": sorted(lost),
            "original_final_correct_broken_count": len(original_success - changed_success),
            "original_final_correct_broken_ordinals": sorted(original_success - changed_success),
            "new_correct_vs_original_frozen_system_count": len(changed_success - original_success),
            "new_correct_vs_original_frozen_system_ordinals": sorted(changed_success - original_success),
            "final_prediction_identity_changed_count": sum(old["final_prediction_identity_representative"] != new["final_prediction_identity_representative"]
                for old, new in zip(all_actions, actions, strict=True))}
    for source in sources.values():
        need(sha(ROOT / source["path"]) == source["sha256"], "SOURCE_CHANGED_DURING_EXPLANATION")
    need(sha(Path(__file__)) == program_sha, "PROGRAM_CHANGED_DURING_EXPLANATION")
    out.mkdir()
    detail = out / "query_action_terms.jsonl"
    with detail.open("xb") as handle:
        for record in records:
            handle.write(encode(record) + b"\n")
    result = {"status": "RC_FROZEN_ROMA_DIFFICULT90_ACTION_TERMS_EXACT_REPLAY_COMPLETE",
        "claim_level": "OPENED_ORIGINAL_DIFFICULT90_FROZEN_HEAD_INPUT_TERM_INTERVENTION_ONLY",
        "sources": sources, "program_sha256": program_sha, "original_population": original["population"],
        "original_R1_summary_exact_replay": baseline, "single_feature_zero_interventions": summaries,
        "intervention_order_frozen_before_execution": list(INTERVENTIONS), "feature_names": list(FEATURE_NAMES),
        "weights": gate.WEIGHT.tolist(), "weights_binary64": [float(x).hex() for x in gate.WEIGHT],
        "bias": gate.BIAS, "bias_binary64": float(gate.BIAS).hex(), "switch_threshold": 0.0,
        "tie_rule": "MAX_LOGIT_THEN_SMALLEST_PHYSICAL_ROW; RAW_WINNER_MAX_RAW_THEN_SMALLEST_PHYSICAL_ROW",
        "neutralization": "CLONE_DERIVED_FEATURE_VECTOR_SET_ONE_COMPONENT_PLUS_ZERO_RECOMPUTE_ORIGINAL_FP64_WEIGHTED_SUM_AND_BIAS",
        "candidate_count": 128, "challengers_per_query": 127, "query_count": 90,
        "exact_original_challenger_logit_count": 11430, "counterfactual_challenger_logit_count": 68580,
        "all_query_terms": {"path": detail.name, "sha256": sha(detail)},
        "checks": {"same_original_shards_join_gate_and_result_hashes": True,
            "all90_original_actions_final_predictions_R1_and_logit_bits_replayed": True,
            "all_127_challengers_evaluated_in_each_prespecified_intervention": True,
            "physical_label_equivalence_from_frozen_unique_714_715_duplicate": True,
            "original_effective_switch_duplicate_label_rule_preserved": True,
            "no_logit_minus_term_shortcut_original_reduction_order_preserved": True},
        "new_model_or_parameter_update_count": 0, "threshold_scan_count": 0, "raw_token_or_fullrank_payload_read_count": 0,
        "MRR_recomputed": False, "HYP_GO_claimed": False, "formal_action_authorized": False,
        "limits": ["This explains the actual original frozen head on all90 opened regression queries; it is not a fresh-D1 or TRAIN32 surrogate.",
            "R1 and complete action choices are exactly replayed from cached C128 scalars. Full-gallery MRR is deliberately not recomputed.",
            "Setting a derived symmetric feature to zero is a frozen-head input intervention, not erasing a physical region or modifying RoMa geometry.",
            "Zeroing the explicit visibility-mass feature removes only that head term. Other local-score and Q/R-derived features can still contain mass information; this is not a complete geometry-quality channel ablation.",
            "The six interventions are descriptive necessity/sensitivity checks of existing terms, not six fitted replacement methods or new GO gates.",
            "Feature dependence alone cannot establish spatial HYP causality or a guaranteed recognition improvement."]}
    output = out / "result.json"
    output.write_bytes(encode(result) + b"\n")
    detail.chmod(0o444)
    output.chmod(0o444)
    print(json.dumps({"status": result["status"], "original": baseline, "interventions": summaries,
        "result_sha256": sha(output), "all_query_terms_sha256": sha(detail), "out": str(out)}, sort_keys=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
