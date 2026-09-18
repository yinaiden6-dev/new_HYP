#!/usr/bin/env python3
"""Exact cached old RAW/RoMa action replay, then six fixed input-term ablations.

No token payloads, encoder calls, retraining, new retrieval scores, or new GO
criterion. Single-feature zeroing is an internal computation intervention, not
a model proposal or a pixel/geometry causal claim.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_frozen_roma_action_terms_v1"
PRE = "results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1"
ROLE_ROOT = "results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_shards"
PINS = {
    "feature_program": ("programs/run_romav2_colnomic_no_regret_action_gate_v1.py", "a28ba978cbf55cbd9d5daf811327ce19bfa36909d301b81413a7b8b1cc043b29"),
    "reducer": ("programs/reduce_romav2_colnomic_current_runtime_frozen_gate_v1.py", "adf5fc2e39c924e12ba7d0b359dc5b7de43ea4cc5632c13e7a184c9efebb94c1"),
    "frozen_gate": ("src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py", "96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7"),
    "head": ("results/romav2_colnomic_full_negative_action_gate_v1/result.json", "b296e946ff84ef574605ca5a6f6c6d6892b06dc8f38002b33b913a5afc6f059f"),
    "head_validation": ("results/romav2_colnomic_full_negative_action_gate_v1/independent_validation.json", "2f112c8daa7b0c199e7e7ca2933116bdaf261b547277140950c809cc3bd264de"),
    "old_result": ("results/romav2_colnomic_current_runtime_frozen_gate_v1/result.json", "12dccec38392d45da0a34aece6bc0be8e089ef0f07eca49feb764e8a014d6445"),
    "old_validation": ("results/romav2_colnomic_current_runtime_frozen_gate_v1/independent_validation.json", "83e14a12a8647e08856c2890bfaa638eece1cb3876ba88656e6d755e2f07f87f"),
    "prejoin_validation": (PRE + "/validation.json", "7693c5b09f014f1189ad9ee5722b98bb19e0dd68947afadab2f7e556b197ff52"),
    "old_source_manifest": ("results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json", "e0be35125eddec391a92398e84103e77ca0a65f661452b9190c2b81f3fdbea23"),
    "identity_repair": ("registry/gallery_identity_repair_v1.json", "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f"),
}
FEATURE_NAMES = (
    "standardized_raw_gap", "symmetric_local_score", "symmetric_visibility_mass",
    "symmetric_visibility_normalized_local_similarity", "symmetric_query_spatial_robustness",
    "symmetric_reference_spatial_robustness",
)


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def logical(value):
    return hashlib.sha256(canonical({k: v for k, v in value.items() if k != "logical_sha256"})).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


SOURCES = {}


def read_json(relative, expected=None):
    path = ROOT / relative
    need(path.suffix == ".json" and path.is_file() and not path.is_symlink(), "JSON_ONLY_SOURCE")
    digest = sha(path)
    need(expected is None or digest == expected, "FROZEN_JSON_SHA_DRIFT:" + relative)
    SOURCES[relative] = digest
    return json.loads(path.read_text())


def action(query, logits, target_rows):
    axis = query["axis"]
    winner = query["winner"]
    best_logit, best = max(zip(logits, query["challengers"]), key=lambda pair: (pair[0], -axis[pair[1]]))
    final = best if best_logit > 0.0 else winner
    targets = [i for i, row in enumerate(axis) if row in target_rows]
    return {"execution_ordinal": query["execution_ordinal"], "query_id": query["query_id"],
        "target_present": bool(targets), "target_position": targets[0] if targets else None,
        "base_winner": winner, "proposed_challenger": best, "switch_logit": best_logit,
        "decision": "SWITCH" if best_logit > 0.0 else "HOLD", "final_position": final,
        "base_correct": axis[winner] in target_rows, "final_correct": axis[final] in target_rows}


def summary(actions):
    eligible = [row for row in actions if row["target_present"]]
    return {"query_count": len(actions), "eligible_target_count": len(eligible),
        "target_absent_count": len(actions) - len(eligible),
        "base_top1": sum(row["base_correct"] for row in eligible),
        "final_top1": sum(row["final_correct"] for row in eligible),
        "rescue": sum(not row["base_correct"] and row["final_correct"] for row in eligible),
        "break": sum(row["base_correct"] and not row["final_correct"] for row in eligible),
        "switch_count": sum(row["decision"] == "SWITCH" for row in eligible)}


def compare(full, changed):
    fields = {name: [] for name in ("full_rescues_retained", "full_rescues_lost", "new_rescues",
        "new_errors_from_full_correct", "full_errors_corrected", "full_breaks_retained", "full_breaks_removed",
        "correct_HOLDs_retained", "correct_HOLDs_broken", "prediction_changed")}
    for old, new in zip(full, changed, strict=True):
        need(old["query_id"] == new["query_id"], "INTERVENTION_QUERY_AXIS")
        key = old["query_id"]
        old_rescue = not old["base_correct"] and old["final_correct"]
        new_rescue = not new["base_correct"] and new["final_correct"]
        predicates = {
            "full_rescues_retained": old_rescue and new_rescue,
            "full_rescues_lost": old_rescue and not new_rescue,
            "new_rescues": new_rescue and not old_rescue,
            "new_errors_from_full_correct": old["final_correct"] and not new["final_correct"],
            "full_errors_corrected": not old["final_correct"] and new["final_correct"],
            "full_breaks_retained": old["base_correct"] and not old["final_correct"] and not new["final_correct"],
            "full_breaks_removed": old["base_correct"] and not old["final_correct"] and new["final_correct"],
            "correct_HOLDs_retained": old["base_correct"] and old["decision"] == "HOLD" and new["final_correct"],
            "correct_HOLDs_broken": old["base_correct"] and old["decision"] == "HOLD" and not new["final_correct"],
            "prediction_changed": old["final_position"] != new["final_position"],
        }
        for field, value in predicates.items():
            if value:
                fields[field].append(key)
    return {"counts": {k: len(v) for k, v in fields.items()}, "query_sets": fields}


def save(relative, value):
    path = OUT / relative
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return {"path": relative, "sha256": sha(path)}


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("60_SECOND_ARITHMETIC_BUDGET")))
    signal.alarm(60)
    need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    for relative, expected in PINS.values():
        need(sha(ROOT / relative) == expected, "FROZEN_SOURCE_SHA_DRIFT:" + relative)
        SOURCES[relative] = expected
    SOURCES[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    docs = {name: read_json(relative, digest) for name, (relative, digest) in PINS.items() if relative.endswith(".json")}
    old = docs["old_result"]
    need(docs["old_validation"]["status"] == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_PASS"
        and docs["old_validation"]["result_sha256"] == PINS["old_result"][1]
        and all(docs["old_validation"]["checks"].values()), "OLD_REPLAY_NOT_VALIDATED")
    val = docs["prejoin_validation"]
    need(val["status"] == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS"
        and all(val["checks"].values()) and val["train_count"] == val["eval_count"] == 32, "OLD_FEATURE_PREJOIN_NOT_CLOSED")
    # Extract the original pure feature functions verbatim, without importing
    # their unrelated gallery/token/training modules or executing their main.
    tree = ast.parse((ROOT / PINS["feature_program"][0]).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("sym", "candidate_feature")]
    need(len(functions) == 2, "ORIGINAL_FEATURE_FUNCTIONS_NOT_FOUND")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), PINS["feature_program"][0], "exec"), namespace)
    feature = namespace["candidate_feature"]
    weights = torch.tensor(docs["head"]["head"]["weight"], dtype=torch.float64)
    bias = float(docs["head"]["head"]["bias"])
    need(weights.shape == (6,) and docs["head"]["parameter_count"] == 7, "FROZEN_HEAD_SHAPE")
    rows = []
    for shard in range(8):
        entry = next(item for item in val["shards"] if item["shard"] == shard)
        doc = read_json(PRE + f"/shard{shard:02d}/result.json", entry["sha256"])
        need(doc["logical_sha256"] == logical(doc) and doc["shard"] == shard
            and doc["target_role_read_count"] == doc["target_insertion_count"] == 0, "FEATURE_SHARD_ENVELOPE")
        rows.extend(doc["rows"])
    need(len(rows) == len({row["query_id"] for row in rows}) == 64, "CURRENT64_AXIS")
    need({role: sum(row["role"] == role for row in rows) for role in ("TRAIN", "EVAL")} == {"TRAIN": 32, "EVAL": 32}, "ROLE_COUNTS")
    queries = []
    for row in rows:
        candidates = {int(item["candidate_position"]): item for item in row["candidates"]}
        need(set(candidates) == set(range(128)) and len(row["candidates"]) == 128, "FULL_C128_CANDIDATES")
        axis = [int(candidates[i]["physical_row"]) for i in range(128)]
        need(len(set(axis)) == 128, "DUPLICATE_PHYSICAL_CANDIDATE")
        raw = [float(candidates[i]["raw_score"]) for i in range(128)]
        winner = max(range(128), key=lambda i: (raw[i], -axis[i]))
        challengers = [i for i in range(128) if i != winner]
        features = [feature(raw, candidates, candidate, winner) for candidate in challengers]
        logits = [float((weights * vector).sum() + bias) for vector in features]
        need(all(torch.isfinite(vector).all() for vector in features) and all(__import__("math").isfinite(z) for z in logits), "NONFINITE_FEATURE_LOGIT")
        queries.append({"query_id": row["query_id"], "execution_ordinal": int(row["execution_ordinal"]),
            "role": row["role"], "axis": axis, "raw": raw, "candidates": candidates,
            "winner": winner, "challengers": challengers, "features": features, "logits": logits})
    # All full-model candidate calculations are complete before any role join.
    source = docs["old_source_manifest"]
    source_rows = {int(row["execution_ordinal"]): row for row in source["records"]}
    repair = docs["identity_repair"]
    need(repair["physical_row_count"] == 5413 and repair["corrected_identity_count"] == 5412
        and repair["verified_byte_identical_duplicate_components"] == [{"canonical_identity": "Biogen_21",
            "physical_rows": [714, 715], "relative_paths": ["homeopathic/Biogen_21 .jpg", "homeopathic/Biogen_21.jpg"],
            "sha256": "3972cbccd14da1942e6d64dfeb460322a8ad3e766f6dedb58a233f45ffca77ea"}], "CORRECTED_IDENTITY_EQUIVALENCE_DRIFT")
    for query in queries:
        execution = query["execution_ordinal"]
        role = read_json(ROLE_ROOT + f"/role_exec{execution:03d}.json")
        prior = source_rows[execution]
        need(role["logical_sha256"] == logical(role) and role["target_naturally_present"] is True
            and role["query_id"] == prior["query_id"] == query["query_id"]
            and role["candidate_axis_sha256"] == prior["candidate_axis_sha256"]
            and role["source_manifest_logical_sha256"] == source["logical_sha256"]
            and role["prejoin_source_record_logical_sha256"] == prior["record_logical_sha256"], "ORIGINAL_ROLE_SOURCE_BINDING")
        target_row = int(prior["candidate_physical_rows"][int(role["target_candidate_position"])])
        target_rows = {target_row}
        if target_row in (714, 715):
            need(role["identity"] == "Biogen_21", "DUPLICATE_TARGET_IDENTITY_DRIFT")
            target_rows = {714, 715}
        need(sum(row in target_rows for row in query["axis"]) <= 1, "CURRENT_AXIS_TARGET_IDENTITY_NOT_UNIQUE")
        query.update(target_rows=target_rows, supergroup=role["supergroup"], target_identity=role["identity"])
        query["full_action"] = action(query, query["logits"], target_rows)
    eval_actions = [query["full_action"] for query in queries if query["role"] == "EVAL"]
    eval_summary = summary(eval_actions)
    gates = {"eligible_target_count_ge_30": eval_summary["eligible_target_count"] >= 30,
        "strict_top1_gain": eval_summary["final_top1"] > eval_summary["base_top1"],
        "rescue_gt_break": eval_summary["rescue"] > eval_summary["break"],
        "break_at_most_one": eval_summary["break"] <= 1}
    passed = all(gates.values())
    replay = {"schema_version": "rc_romav2_colnomic_current_runtime_frozen_gate_v1_20260901",
        "status": "ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_GO" if passed else "ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_NO_GO",
        "claim_level": "ADAPTIVE_INTERNAL_OOF_DEPLOYMENT_COMPATIBILITY_ONLY", "frozen_head_sha256": PINS["head"][1],
        "frozen_head_validation_sha256": PINS["head_validation"][1], "parameter_count": 7, "model_update_count": 0,
        "evaluation_summary": eval_summary, "gates": gates, "actions": eval_actions,
        "target_join_after_all_prejoin_shards": True, "sealed_read_count": 0, "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": "NEW_DIFFICULT_SEALED_QUERY_TOKENIZATION_E0" if passed else None, "logical_sha256": ""}
    replay["logical_sha256"] = logical(replay)
    need(replay == old and canonical(replay) == canonical(old), "STOP_OLD_COMPLETE_RESULT_NOT_EXACTLY_REPRODUCED")
    for actual, expected in zip(eval_actions, old["actions"], strict=True):
        need(actual["switch_logit"].hex() == float(expected["switch_logit"]).hex(), "OLD_ACTION_LOGIT_BIT_DRIFT")
    print(json.dumps({"event": "OLD_COMPLETE_BASELINE_EXACT_REPLAY_PASS", "EVAL": eval_summary,
        "elapsed_seconds": time.monotonic() - started}), flush=True)
    # Only after the complete old result reproduces do we compute the six
    # predeclared interventions. Re-evaluate every challenger, not only the
    # original proposed challenger. No old logit-minus-term approximation.
    outputs = []
    for query in queries:
        terms = [weights * vector for vector in query["features"]]
        removed = {}
        for index, name in enumerate(FEATURE_NAMES):
            logits = []
            for vector in query["features"]:
                changed = vector.clone()
                changed[index] = 0.0
                logits.append(float((weights * changed).sum() + bias))
            removed[name] = {"zeroed_input_index": index, "complete_127_logits_binary64": [z.hex() for z in logits],
                "action": action(query, logits, query["target_rows"])}
        proposal_index = query["challengers"].index(query["full_action"]["proposed_challenger"])
        challengers = []
        for position, vector, term, logit in zip(query["challengers"], query["features"], terms, query["logits"], strict=True):
            need(float((torch.tensor(term.tolist(), dtype=torch.float64).sum() + bias)).hex() == logit.hex(), "TERM_LOGIT_RECOMPOSITION_BITS")
            challengers.append({"candidate_position": position, "physical_row": query["axis"][position],
                "features_binary64": [float(x).hex() for x in vector], "weighted_terms_binary64": [float(x).hex() for x in term],
                "weighted_terms": term.tolist(), "bias_binary64": bias.hex(), "logit_binary64": logit.hex(), "logit": logit})
        outputs.append({"query_id": query["query_id"], "execution_ordinal": query["execution_ordinal"], "role": query["role"],
            "supergroup": query["supergroup"], "target_identity": query["target_identity"],
            "candidate_physical_rows": query["axis"], "raw_scores_binary64": [x.hex() for x in query["raw"]],
            "target_equivalent_physical_rows": sorted(query["target_rows"]), "full_action": query["full_action"],
            "selected_challenger_terms": dict(zip(FEATURE_NAMES, terms[proposal_index].tolist())),
            "selected_challenger_bias": bias, "selected_challenger_logit": query["logits"][proposal_index],
            "HOLD_explanation": "When HOLD, the maximal rejected challenger logit and its six terms explain the no-switch decision; the unchanged base winner has no self-challenge logit.",
            "complete_127_challengers": challengers, "single_feature_zero_interventions": removed})
    summaries = {}
    for role in ("TRAIN", "EVAL"):
        selected = [row for row in outputs if row["role"] == role]
        full = [row["full_action"] for row in selected]
        summaries[role] = {"full_model": summary(full), "single_feature_zero": {}}
        for name in FEATURE_NAMES:
            changed = [row["single_feature_zero_interventions"][name]["action"] for row in selected]
            summaries[role]["single_feature_zero"][name] = {"summary": summary(changed), "comparison_to_full_model": compare(full, changed)}
    for relative, expected in SOURCES.items():
        need(sha(ROOT / relative) == expected, "SOURCE_CHANGED_DURING_REPLAY:" + relative)
    OUT.mkdir()
    replay_binding = save("legacy_complete_result_replay.json", replay)
    records_binding = save("complete_action_terms_and_interventions.json", {"feature_names": FEATURE_NAMES,
        "weight_binary64": [float(x).hex() for x in weights], "bias_binary64": bias.hex(), "records": outputs})
    result = {"status": "RC_FROZEN_ROMA_ACTION_TERMS_EXACT_REPLAY_AND_ACCOUNTING_COMPLETE",
        "claim_level": "OLD_FROZEN_RAW_ROMA_CURRENT_RUNTIME_INTERNAL_COMPUTATION_INTERVENTIONS_ONLY",
        "sources": [{"path": k, "sha256": v} for k, v in sorted(SOURCES.items())],
        "legacy_complete_result_replay": replay_binding, "complete_terms": records_binding,
        "checks": {"old_complete_EVAL_result_canonical_payload_exact": True, "old_32_EVAL_action_logits_FP64_bits_exact": True,
            "original_feature_function_source_executed_without_importing_token_loader": True,
            "all_64_C128_candidate_axes_and_127_challengers_retained": True,
            "all_64_127_weighted_term_logit_recompositions_exact": True,
            "all_six_feature_zero_interventions_rescore_all_127_with_original_ties_and_threshold": True,
            "all_source_hashes_unchanged": True},
        "label_join": "Original source-axis target physical row plus frozen corrected-identity duplicate equivalence [714,715], then membership on the current RAW C128; EVAL target positions and complete outputs reproduce the original gallery-label reducer exactly.",
        "summaries": summaries, "parameter_updates": 0, "new_model_or_scoring_definition": False,
        "token_payload_reads": 0, "raw_image_reads": 0, "new_encoder_or_RoMa_forwards": 0,
        "D1_or_protected_data_reads": 0, "Slurm_jobs": 0,
        "intervention_definition": "For each of six existing candidate-difference input coordinates separately set that coordinate to positive zero for every challenger and reevaluate the original dot-product, bias, full challenger selection and zero-threshold action. No retraining or selection among interventions.",
        "limits": ["TRAIN and adaptive internal EVAL are reported separately; this is not a new independent endpoint.",
            "Term sign/magnitude is not a spatial-causal attribution; correlated terms and the fixed intercept affect interpretation.",
            "Zeroing the explicit visibility-mass feature does not remove mass from real_score, query/reference differences, or other derived features; it deletes only this one head input term.",
            "Feature zeroing tests this frozen computation, not deletion of pixels, removal of RoMa, removal of geometry, or absence of HYP.",
            "No intervention replaces the frozen deployed model, no new GO rule, threshold, tuning or HYP conclusion."],
        "scientific_GO_or_NO_GO": None, "HYP_GO_claimed": False, "elapsed_seconds": time.monotonic() - started}
    final = save("result.json", result)
    signal.alarm(0)
    print(json.dumps({"status": result["status"], "result_sha256": final["sha256"], "summaries": {
        role: {"full": item["full_model"], "zero_feature": {name: {
            "summary": detail["summary"], "changes": detail["comparison_to_full_model"]["counts"]}
            for name, detail in item["single_feature_zero"].items()}} for role, item in summaries.items()},
        "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
