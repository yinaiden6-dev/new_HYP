#!/usr/bin/env python3
"""Post-join F128 analysis. No labels or predictions are read before the final gate.

This reporting program is deliberately outside the frozen scientific source set.
It hashes, but never deserializes or exports, model/checkpoint/trace tensors.
Exit 75 means the final independent join gate is absent or has not passed.
Exit 1 means a binding, integrity or consistency check failed. No output is
published in either case. Run this only after the existing join has completed.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = RC / "results/rc_token_competition_f128_v2"
GATE = "TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS"
ARMS = ("TOKEN_QR", "TOKEN_QRR_ANCHOR", "TOKEN_QRR_MULTI")
POINTS = ("zero", "inherited_tau")
REPORT = "REPORT_TOKEN_COMPETITION_F128_V2_ANALYSIS.md"
SPLITS = ("train_query_ids", "inner_fit_query_ids", "inner_val_query_ids", "heldout_query_ids")


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


class Audit:
    """Hash every explicit binding once, reject conflicting hashes and changes."""

    def __init__(self):
        self.files = {}

    @staticmethod
    def signature(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def check(self, item):
        need(isinstance(item, dict) and {"path", "sha256"} <= item.keys(), "MISSING_SHA_BINDING")
        path = Path(item["path"]).resolve()
        need(path.is_file(), "BOUND_FILE_MISSING:" + str(path))
        expected = item["sha256"]
        need(isinstance(expected, str) and len(expected) == 64, "BAD_SHA:" + str(path))
        before = self.signature(path)
        prior = self.files.get(str(path))
        if prior:
            need(prior[0] == expected and prior[1] == before, "BINDING_CHANGED:" + str(path))
        else:
            need(sha(path) == expected, "SHA_MISMATCH:" + str(path))
            need(self.signature(path) == before, "CHANGED_DURING_HASH:" + str(path))
            self.files[str(path)] = (expected, before)
        return path

    def tree(self, value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.check(value)
            for item in value.values():
                self.tree(item)
        elif isinstance(value, list):
            for item in value:
                self.tree(item)

    def read(self, item):
        value = read_json(self.check(item))
        self.tree(value)
        return value

    def unchanged(self):
        for path, (_, signature) in self.files.items():
            need(self.signature(Path(path)) == signature, "CHANGED_BEFORE_PUBLISH:" + path)


def keyed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        need(key not in result, "DUPLICATE_KEY:" + str(key))
        result[key] = row
    return result


def close_number(a, b, label):
    need(math.isfinite(float(a)) and math.isfinite(float(b)) and
         math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=2e-10), label)


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def decision(row, threshold):
    scores, anchor = row["logits"], row["winner"]
    need(len(scores) == 128 and 0 <= anchor < 128 and scores[anchor] == 0.0 and
         all(math.isfinite(x) for x in scores), "INVALID_PREDICTION_SCORES")
    contenders = [p for p in range(128) if p != anchor]
    best = max(contenders, key=lambda p: scores[p])
    unique = sum(scores[p] == scores[best] for p in contenders) == 1
    return best if unique and scores[best] > threshold else anchor


def load_verified(root, validation, audit):
    """Called only after the exact final status has been checked in main."""
    audit.tree(validation)
    need(validation.get("fits_complete") == 15 and validation.get("panel_queries") == 128 and
         validation.get("seed") == 0 and validation.get("held_labels_read_only_after_all15_checks") is True,
         "FINAL_GATE_SCOPE")
    need(validation.get("development_analysis") is True and validation.get("external_GO") is False and
         validation.get("formal_H593_replacement") is False, "FINAL_EVIDENCE_SCOPE")
    protocol = audit.read(validation["protocol"])
    need(audit.check(validation["protocol"]) == root / "protocol.json", "PROTOCOL_LOCATION")
    need(Path(protocol["output_root"]).resolve() == root and protocol["version"] == "TOKEN_COMPETITION_F128_V2",
         "PROTOCOL_ROOT_OR_VERSION")
    need(tuple(protocol["arms"]) == ARMS and protocol["seeds"] == [0], "ARMS_OR_SEEDS")
    panel = sorted(protocol["panel_query_ids"])
    need(len(panel) == len(set(panel)) == 128, "PANEL128")
    need(protocol["sources"] == protocol["code_sources"], "FROZEN_SOURCE_REGISTRY")
    code = protocol["code_sources"]
    need(str(Path(__file__).resolve()) not in {b["path"] for b in code.values()}, "REPORTER_MUST_NOT_BE_FROZEN_SOURCE")
    for relative, value in code.items():
        need(audit.check(value) == (RC / relative).resolve(), "SOURCE_LOCATION:" + relative)
    need(validation["sources"]["verifier"] == code["programs/join_token_competition_v2.py"], "JOIN_SOURCE_BINDING")
    need(validation["sources"]["baseline_validation"] == protocol["source_validation"], "BASELINE_GATE_BINDING")
    baseline_gate = audit.read(protocol["source_validation"])
    need(baseline_gate["status"] == "F128_BASELINE15_INDEPENDENT_PASS" and
         baseline_gate["protocol"] == protocol["source_protocol"], "SAME_F128_BASELINE_GATE")
    audit.read(protocol["source_protocol"])
    artifact_names = ("metrics.csv", "paired_vs_B_CAL.csv", "paired_structural.csv",
                      "joined_predictions.json", "candidate_predictions.json", "trace_manifest.json",
                      "REPORT_TOKEN_COMPETITION_V2.md")
    need(set(artifact_names) <= validation["artifacts"].keys(), "FINAL_ARTIFACT_SET")
    for name in artifact_names:
        need(audit.check(validation["artifacts"][name]) == root / name, "FINAL_ARTIFACT_LOCATION:" + name)
    audit.read(validation["artifacts"]["trace_manifest.json"])
    receipts = keyed(validation["receipts"], ("fold", "seed", "arm"))
    expected = {(f, 0, a) for f in range(5) for a in ARMS}
    need(set(receipts) == expected, "ALL15_RECEIPT_KEYS")
    results, bases, fold_of, manifests = {}, {}, {}, set()
    source_roles = {"module": "src/rc_aslo_xf/token_competition_v2.py",
                    "data": "programs/token_competition_data_v2.py",
                    "runner": "programs/run_token_competition_v2.py",
                    "verifier": "programs/join_token_competition_v2.py",
                    "math_v1": "programs/join_rebut_qr_qrr_v1.py",
                    "math_v2": "programs/join_rebut_qr_qrr_frozenbase_v2.py",
                    "f128_join": "programs/join_rebut_qr_qrr_f128_v1.py"}
    result_source_roles = {"runner_sha256": "runner", "module_sha256": "module", "data_adapter_sha256": "data"}
    for fold in range(5):
        split = protocol["folds"][str(fold)]
        train, held = set(split[SPLITS[0]]), set(split[SPLITS[3]])
        fit, val = set(split[SPLITS[1]]), set(split[SPLITS[2]])
        need(not train & held and train | held == set(panel) and not fit & val and fit | val == train,
             "NESTED_FOLD_PARTITION")
        for q in split["heldout_query_ids"]:
            need(q not in fold_of, "DUPLICATE_OOF_QUERY")
            fold_of[q] = fold
        base_binding = protocol["baseline_bindings"][str(fold)]["0"]
        base = audit.read(base_binding["source_result"])
        need(base["binding"]["protocol"] == protocol["source_protocol"] and
             base["binding"]["fold"] == fold and base["binding"]["seed"] == 0 and
             base["binding"]["arm"] == "B_CAL", "BASELINE_MODEL_LINEAGE")
        need(base["model_sha256"] == base_binding["source_model"]["sha256"], "BASELINE_MODEL_SHA")
        bases[fold] = base
        for arm in ARMS:
            item = receipts[(fold, 0, arm)]
            receipt = audit.read(item["receipt"])
            need(receipt["status"] == "TOKEN_FIT_INDEPENDENT_PASS" and receipt["held_labels_opened"] is False,
                 "FIT_GATE")
            need((receipt["fold"], receipt["seed"], receipt["arm"]) == (fold, 0, arm), "FIT_KEY")
            inputs, sources = receipt["inputs"], receipt["inputs"]["sources"]
            need((inputs["fold"], inputs["seed"], inputs["arm"]) == (fold, 0, arm), "FIT_INPUT_KEY")
            need(sources["protocol"] == validation["protocol"] and inputs["baseline"] == base_binding and
                 inputs["baseline_validation"] == protocol["source_validation"], "FIT_PROTOCOL_BASELINE")
            need(inputs["train_roles"] == split["train_roles"], "FIT_TRAIN_ROLES")
            for role, relative in source_roles.items():
                need(sources[role] == code[relative], "FIT_SOURCE:" + role)
            for role in ("common_features", "gallery"):
                need(sources[role] == protocol[role], "FIT_DATA_SOURCE:" + role)
            manifest_binding = sources["evidence_manifest"]
            manifest_path = Path(protocol["evidence_manifest"]["path"] if isinstance(protocol["evidence_manifest"], dict)
                                 else protocol["evidence_manifest"]).resolve()
            need(audit.check(manifest_binding) == manifest_path, "EVIDENCE_MANIFEST_LOCATION")
            if str(manifest_path) not in manifests:
                manifest = audit.read(manifest_binding)
                evidence_gate = audit.read(sources["evidence_validation"])
                need(manifest["status"] == evidence_gate["status"] == "TOKEN_COMPETITION_EVIDENCE128_PASS",
                     "EVIDENCE_STATUS")
                need(evidence_gate["protocol"] == validation["protocol"] and
                     evidence_gate["manifest"] == manifest_binding and evidence_gate["labels_read"] == 0,
                     "EVIDENCE_GATE_BINDING")
                need(len(manifest["records"]) == 128 and {r["query_id"] for r in manifest["records"]} == set(panel),
                     "EVIDENCE_PANEL")
                manifests.add(str(manifest_path))
            folder = root / f"fold{fold}/seed0/{arm}"
            for name, value in inputs["artifacts"].items():
                need(audit.check(value) == (folder / name).resolve(), "FIT_ARTIFACT_LOCATION")
            need(receipt["result"] == inputs["artifacts"]["result.json"], "RESULT_RECEIPT_BINDING")
            result = audit.read(receipt["result"])
            b = result["binding"]
            need(result["status"] == "FROZEN_BASE_RESIDUAL_LABEL_FREE_COMPLETE" and
                 b["protocol"] == validation["protocol"] and (b["fold"], b["seed"], b["arm"]) == (fold, 0, arm),
                 "RESULT_LINEAGE")
            need(result["baseline_bindings"] == b["baseline_bindings"] == base_binding and
                 b["evidence_manifest"] == manifest_binding and result["evidence_bindings"] == inputs["evidence"],
                 "RESULT_DATA_LINEAGE")
            for field, role in result_source_roles.items():
                need(b[field] == sources[role]["sha256"], "RESULT_SOURCE:" + field)
            for field, relative in (("source_runner_sha256", "programs/run_rebut_qr_qrr_frozenbase_v2.py"),
                                    ("v1_runner_sha256", "programs/run_rebut_qr_qrr_v1.py")):
                need(b[field] == code[relative]["sha256"], "RESULT_SOURCE:" + field)
            for field in SPLITS:
                need(result[field] == base[field] == split[field], "RESULT_SPLIT:" + field)
            need(result["outer_labels_opened"] is False and result["head_exact_unchanged"] is True and
                 result["frozen_head_parameters"] == 18 and result["population_queries"] == 128,
                 "RESULT_SCOPE_OR_FROZEN_HEAD")
            need(result["model_sha256"] == inputs["artifacts"]["model.pt"]["sha256"], "RESULT_MODEL_SHA")
            need(result["calibration"]["threshold"] == 0.0 and result["calibration"]["new_threshold_fitting"] is False,
                 "PRIMARY_THRESHOLD_CHANGED")
            close_number(result["calibration"]["inherited_tau_secondary"], base["calibration"]["threshold"], "SECONDARY_TAU")
            selection = result["selection"]
            need(selection["main_epoch"] == result["best_epoch"] == receipt["selected_epoch"] == item["selected_epoch"],
                 "SELECTED_EPOCH_BINDING")
            need(selection["epoch0_fallback"] == receipt["epoch0_fallback"] == (selection["main_epoch"] == 0) and
                 result["reader_enabled"] == (selection["main_epoch"] > 0), "EPOCH0_FALLBACK_BINDING")
            results[(fold, arm)] = result
        need(results[(fold, ARMS[1])]["trainable_parameters"] == results[(fold, ARMS[2])]["trainable_parameters"],
             "ANCHOR_MULTI_PARAMETER_COUNT")
    need(set(fold_of) == set(panel), "OOF128_COVERAGE")
    return protocol, panel, results, bases, fold_of


def pair_stats(new, old, ids, joined, identities, sampled=None, np=None):
    n = len(ids)
    a = [bool(joined[(*new, q)]["correct"]) for q in ids]
    b = [bool(joined[(*old, q)]["correct"]) for q in ids]
    rescue, breaks = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    correct = sum(b)
    result = {"queries": n, "correct": sum(a), "accuracy": sum(a) / n, "baseline_correct": correct,
              "rescues": rescue, "breaks": breaks, "net": rescue - breaks,
              "gain_percentage_points": 100 * (rescue - breaks) / n,
              "baseline_correct_loss_rate": breaks / correct if correct else None,
              "changed_decisions": sum(joined[(*new, q)]["selected"] != joined[(*old, q)]["selected"] for q in ids)}
    if sampled is not None:
        clusters = sorted(set(identities[q] for q in ids))
        sums = np.array([sum(int(x) - int(y) for q, x, y in zip(ids, a, b) if identities[q] == c) for c in clusters])
        counts = np.array([sum(identities[q] == c for q in ids) for c in clusters])
        draws = sums[sampled].sum(1) / counts[sampled].sum(1)
        result.update(identity_cluster_gain_ci_low=float(np.quantile(draws, .025)),
                      identity_cluster_gain_ci_high=float(np.quantile(draws, .975)))
    return result


def compare_official(computed, official, fields):
    for field in fields:
        actual = computed[field]
        if actual is None:
            need(official[field] in (None, "", "None"), "OFFICIAL_NULL:" + field)
        else:
            close_number(actual, official[field], "OFFICIAL_MISMATCH:" + field)


def analyze(root, validation, audit, protocol, panel, results, bases, fold_of):
    # Both label files come exclusively from SHA-verified final/protocol bindings.
    curator = audit.read(validation["sources"]["curator"])
    gallery_document = audit.read(protocol["gallery"])
    curator_rows = keyed(curator["records"], ("query_id",))
    gallery = {int(r["physical_row"]): r["identity"] for r in gallery_document["records"]}
    need(len(gallery) == len(gallery_document["records"]), "DUPLICATE_GALLERY_ROW")
    identities = {q: curator_rows[(q,)]["identity"] for q in panel}
    need(all(curator_rows[(q,)]["outer_fold"] == fold_of[q] for q in panel), "CURATOR_FOLD")
    joined_doc = audit.read(validation["artifacts"]["joined_predictions.json"])
    candidates_doc = audit.read(validation["artifacts"]["candidate_predictions.json"])
    joined = keyed(joined_doc["records"], ("model", "operating_point", "query_id"))
    expected_keys = {(a, p, q) for a in ("B_CAL", *ARMS) for p in POINTS for q in panel}
    expected_keys |= {("RAW", "untrained", q) for q in panel}
    need(set(joined) == expected_keys, "JOINED_POPULATION")
    candidates = keyed(candidates_doc["records"], ("model", "query_id", "position"))
    need(set(candidates) == {(a, q, p) for a in ARMS for q in panel for p in range(128)}, "CANDIDATE_POPULATION")
    predictions = {}
    for fold in range(5):
        for arm, result in [("B_CAL", bases[fold]), *[(a, results[(fold, a)]) for a in ARMS]]:
            rows = keyed(result["predictions"], ("query_id",))
            need(set(rows) == {(q,) for q in protocol["folds"][str(fold)]["heldout_query_ids"]}, "RESULT_PREDICTION_PANEL")
            predictions.update({(arm, q[0]): row for q, row in rows.items()})
    query_info = {}
    for q in panel:
        base = predictions[("B_CAL", q)]
        axis, winner = base["axis"], base["winner"]
        need(len(axis) == len(set(axis)) == 128 and len({gallery[x] for x in axis}) == 128, "C128_IDENTITY_AXIS")
        target = [p for p, physical in enumerate(axis) if gallery[physical] == identities[q]]
        target_position = target[0] if target else None
        ranked = sorted(range(128), key=lambda p: (-base["logits"][p], axis[p]))
        rank = ranked.index(target_position) + 1 if target else None
        query_info[q] = {"target_in_c128": bool(target), "target_position": target_position,
                         "target_physical_row": axis[target_position] if target else None,
                         "target_B_CAL_rank": rank,
                         "target_rank_bucket": "absent" if rank is None else ("top2" if rank <= 2 else "rank3_128"),
                         "raw_physical_row": axis[winner]}
        raw = joined[("RAW", "untrained", q)]
        need(raw["selected"] == axis[winner] and raw["correct"] == (gallery[axis[winner]] == identities[q]), "RAW_JOIN")
        for arm in ("B_CAL", *ARMS):
            row = predictions[(arm, q)]
            need(row["axis"] == axis and row["winner"] == winner, "PREDICTION_AXIS_BINDING")
            for point in POINTS:
                threshold = 0.0 if point == "zero" else bases[fold_of[q]]["calibration"]["threshold"]
                position = decision(row, threshold)
                official = joined[(arm, point, q)]
                need(official["seed"] == 0 and official["fold"] == fold_of[q] and
                     official["selected"] == axis[position] and
                     official["correct"] == (gallery[axis[position]] == identities[q]), "JOINED_DECISION_OR_LABEL")
                close_number(official["threshold"], threshold, "JOINED_THRESHOLD")
            if arm == "B_CAL":
                continue
            for pos in range(128):
                c = candidates[(arm, q, pos)]
                need(c["seed"] == 0 and c["fold"] == fold_of[q] and c["physical_row"] == axis[pos] and
                     c["raw_anchor"] == (pos == winner), "CANDIDATE_AXIS_BINDING")
                close_number(c["zero_base"], base["logits"][pos], "CANDIDATE_BASE")
                close_number(c["logit"], row["logits"][pos], "CANDIDATE_LOGIT")
                close_number(c["logit"], float.fromhex(c["logit_hex"]), "CANDIDATE_HEX")
                close_number(c["logit"], c["zero_base"] + c["residual"], "CANDIDATE_RESIDUAL")
                need(math.isfinite(c["D"]), "CANDIDATE_D")
                for edge in c["edges"]:
                    need(edge["rival_physical_row"] == axis[edge["rival_position"]] and
                         math.isfinite(edge["pair_difference"]), "CANDIDATE_EDGE")
    metrics = csv_rows(root / "metrics.csv")
    official_metrics = keyed(metrics, ("model", "operating_point"))
    need(set(official_metrics) == {(a, p) for a in ("B_CAL", *ARMS) for p in POINTS} | {("RAW", "untrained")}, "METRICS_KEYS")
    for (model, point), row in official_metrics.items():
        calculated = pair_stats((model, point), ("RAW", "untrained"), panel, joined, identities)
        compare_official(calculated, row, ("queries", "correct", "accuracy"))
        for source, target in (("rescues", "rescues_vs_RAW"), ("breaks", "breaks_vs_RAW"), ("net", "net_vs_RAW")):
            close_number(calculated[source], row[target], "METRICS_RAW:" + target)
    import numpy as np
    clusters = sorted(set(identities.values()))
    need(validation["identity_cluster_bootstrap_resamples"] == 10000, "BOOTSTRAP_COUNT")
    sampled = np.random.default_rng(20260927).integers(0, len(clusters), size=(10000, len(clusters)))
    pair_rows = csv_rows(root / "paired_vs_B_CAL.csv") + csv_rows(root / "paired_structural.csv")
    official_pairs = keyed(pair_rows, ("model", "operating_point", "baseline"))
    comparisons = [(a, "B_CAL") for a in ARMS] + [(ARMS[2], ARMS[1])]
    need(set(official_pairs) == {(a, p, "same_F128_" + b + "_seed0") for a, b in comparisons for p in POINTS}, "PAIRED_KEYS")
    pairs, per_fold = [], []
    for arm, baseline in comparisons:
        for point in POINTS:
            calculated = pair_stats((arm, point), (baseline, point), panel, joined, identities, sampled, np)
            official = official_pairs[(arm, point, "same_F128_" + baseline + "_seed0")]
            compare_official(calculated, official, ("queries", "rescues", "breaks", "net", "baseline_correct",
                "baseline_correct_loss_rate", "changed_decisions", "identity_cluster_gain_ci_low", "identity_cluster_gain_ci_high"))
            row = {"model": arm, "baseline": baseline, "seed": 0, "operating_point": point, **calculated}
            fold_nets = []
            for fold in range(5):
                qs = [q for q in panel if fold_of[q] == fold]
                f = {"model": arm, "baseline": baseline, "seed": 0, "operating_point": point, "fold": fold,
                     **pair_stats((arm, point), (baseline, point), qs, joined, identities)}
                sel = results[(fold, arm)]["selection"]
                f.update(main_epoch=sel["main_epoch"], best_ce_epoch=sel["best_ce_epoch"],
                         epoch0_fallback=sel["epoch0_fallback"],
                         trainable_parameters=results[(fold, arm)]["trainable_parameters"])
                per_fold.append(f)
                fold_nets.append(f["net"])
            row.update(fold_nets=fold_nets, positive_folds=sum(n > 0 for n in fold_nets),
                       negative_folds=sum(n < 0 for n in fold_nets), unchanged_folds=sum(n == 0 for n in fold_nets))
            pairs.append(row)
    diagnostics = []
    for q in panel:
        info = query_info[q]
        for arm in ARMS:
            for point in POINTS:
                current, baseline = joined[(arm, point, q)], joined[("B_CAL", point, q)]
                rescue = bool(current["correct"] and not baseline["correct"])
                broken = bool(baseline["correct"] and not current["correct"])
                pred = predictions[(arm, q)]
                chosen = pred["axis"].index(current["selected"])
                target = candidates[(arm, q, info["target_position"])] if info["target_in_c128"] else None
                c = candidates[(arm, q, chosen)]
                diagnostics.append({"query_id": q, "identity": identities[q], "fold": fold_of[q], "seed": 0,
                    "model": arm, "operating_point": point, **info,
                    "B_CAL_selected": baseline["selected"], "B_CAL_correct": baseline["correct"],
                    "selected": current["selected"], "correct": current["correct"],
                    "action": "HOLD" if current["selected"] == info["raw_physical_row"] else "SWITCH",
                    "changed_decision": current["selected"] != baseline["selected"],
                    "rescue": rescue, "break": broken, "net": int(rescue) - int(broken),
                    "target_base_logit": target["zero_base"] if target else None,
                    "target_residual": target["residual"] if target else None,
                    "target_final_logit": target["logit"] if target else None,
                    "selected_residual": c["residual"], "selected_final_logit": c["logit"],
                    "selected_rival_positions": json.dumps([e["rival_position"] for e in c["edges"]]),
                    "main_epoch": results[(fold_of[q], arm)]["selection"]["main_epoch"],
                    "epoch0_fallback": results[(fold_of[q], arm)]["selection"]["epoch0_fallback"]})
    strata = []
    for arm in ARMS:
        for point in POINTS:
            for bucket in ("top2", "rank3_128", "absent"):
                rows = [r for r in diagnostics if r["model"] == arm and r["operating_point"] == point and r["target_rank_bucket"] == bucket]
                strata.append({"model": arm, "operating_point": point, "target_rank_bucket": bucket,
                    "queries": len(rows), "baseline_wrong": sum(not r["B_CAL_correct"] for r in rows),
                    "baseline_correct": sum(r["B_CAL_correct"] for r in rows), "correct": sum(r["correct"] for r in rows),
                    "rescues": sum(r["rescue"] for r in rows), "breaks": sum(r["break"] for r in rows),
                    "net": sum(r["net"] for r in rows)})
    selections = [{"fold": f, "seed": 0, "model": a, "trainable_parameters": r["trainable_parameters"],
                   "reader_enabled": r["reader_enabled"], **r["selection"]} for (f, a), r in sorted(results.items())]
    coverage = Counter(info["target_rank_bucket"] for info in query_info.values())
    summary = {"status": "TOKEN_COMPETITION_ANALYSIS_PASS", "validation_status": GATE,
        "validation": binding(root / "validation.json"), "protocol": validation["protocol"],
        "panel": "F128 opened development; original execution ordinals 0..127, original grouped five folds",
        "seed": 0, "queries": 128, "candidate_source": "frozen natural ColNomic C128",
        "primary_operating_point": "zero", "secondary_operating_point": "inherited_tau",
        "baseline": "SAME F128 seed0 phase-correct frozen 18-dimensional B_CAL, original RMS and M preserved",
        "metrics": metrics, "comparisons": pairs, "selections": selections,
        "candidate_membership": {"queries": 128, "present": 128 - coverage["absent"], "absent": coverage["absent"],
            "coverage": (128 - coverage["absent"]) / 128, "B_CAL_target_top2": coverage["top2"],
            "B_CAL_target_rank3_128": coverage["rank3_128"],
            "ranking_rule": "descending frozen B_CAL logit, tie ascending physical row; diagnostic only; decision ties HOLD"},
        "target_rank_strata": strata, "bootstrap": {"unit": "target identity cluster", "clusters": len(clusters),
            "resamples": 10000, "rng_seed": 20260927, "interval": "percentile95", "units": "accuracy difference"},
        "scope": {"development_only": True, "old_EVAL128": False, "H593_confirmation": False,
                  "external_confirmation": False, "M_redesign_claim": False, "compression_unique_causal_claim": False,
                  "held_results_select_model_epoch_threshold_seed": False, "model_weights_exported": False},
        "inputs": {"validation": binding(root / "validation.json"), "protocol": validation["protocol"],
                   "analysis_source": binding(__file__), "final_artifacts": validation["artifacts"],
                   "curator": validation["sources"]["curator"], "gallery": protocol["gallery"]}}
    return summary, per_fold, diagnostics


def conclusion(row):
    if row["net"] > 0:
        evidence = ("配对 identity-cluster 95% CI 完全高于 0，支持本开发面板上的正向差异"
                    if row["identity_cluster_gain_ci_low"] > 0 else "配对 identity-cluster 95% CI 未完全高于 0，尚不能据此确认稳定增益")
        return f"净增加 {row['net']} 个正确决策（+{row['gain_percentage_points']:.2f} 个百分点）；{evidence}。"
    if row["net"] < 0:
        return f"净减少 {-row['net']} 个正确决策（{row['gain_percentage_points']:.2f} 个百分点），当前配置不支持优于对照的主张。"
    return f"正确数净变化为 0；救回 {row['rescues']}、破坏 {row['breaks']}，改变 {row['changed_decisions']} 个决策，没有净收益证据。"


def report_text(s):
    primary = [r for r in s["comparisons"] if r["operating_point"] == "zero"]
    structural = next(r for r in primary if r["baseline"] == ARMS[1])
    lines = ["# TOKEN competition V2：F128 开发面板完整分析", "",
        "本报告只在完整 independent join PASS 且输入 SHA/来源绑定全部复核后生成。主结果使用预先固定的 zero 阈值；继承阈值单独列作次要结果。", ""]
    for row in primary:
        lines.append(f"- **{row['model']} 对 {row['baseline']}**：" + conclusion(row))
    lines += ["", "## 主结果：fixed0", "",
        "模型为 seed0 的三个 native-token residual reader，叠加 SAME F128 分阶段冻结的 18 维 B_CAL；"
        "候选来自冻结的自然 ColNomic C128，全部 128 个候选均可选择。F128 是已打开的开发面板，保持原分组五折 OOF。", "",
        "| 模型 | 对照 | 正确 /128 | 救回 | 破坏 | 净变化 | 改变决策 | 对照正确样本损失率 | 配对增益 95% CI（百分点） |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        loss = "NA" if r["baseline_correct_loss_rate"] is None else f"{100*r['baseline_correct_loss_rate']:.2f}%"
        lines.append(f"| {r['model']} | {r['baseline']} ({r['baseline_correct']}/128) | {r['correct']} | {r['rescues']} | {r['breaks']} | {r['net']:+d} | {r['changed_decisions']} | {loss} | [{100*r['identity_cluster_gain_ci_low']:.2f}, {100*r['identity_cluster_gain_ci_high']:.2f}] |")
    lines += ["", "救回指对照错误而该模型正确；破坏指对照正确而该模型错误；净变化 = 救回 − 破坏。"
              "对照正确样本损失率完整纳入已有正确样本的回归损失。改变决策也包含两者都错误的情况。"
              "CI 以目标 identity 为重采样簇、10,000 次 bootstrap、随机种子 20260927；单位是准确率差，表中转为百分点。"
              "它是开发面板上的描述性不确定性估计，不是跨 seed、跨数据集确认，也未做多重比较校正。", "",
              "## 折间一致性与训练选择", "", "| 模型 | 对照 | 五折净变化（fold0→4） | 正 / 零 / 负折数 |", "|---|---|---|---|"]
    for r in primary:
        lines.append(f"| {r['model']} | {r['baseline']} | {', '.join(f'{n:+d}' for n in r['fold_nets'])} | {r['positive_folds']} / {r['unchanged_folds']} / {r['negative_folds']} |")
    lines += ["", "五折净变化仅用于检查增益是否集中于少数折，不把五折当成五次独立复现。", "",
              "| 模型 | 主 epoch（fold0→4） | CE 最优 epoch | epoch0 回退数 /5 | reader 参数量 |", "|---|---|---|---:|---:|"]
    for arm in ARMS:
        rows = sorted([r for r in s["selections"] if r["model"] == arm], key=lambda r: r["fold"])
        lines.append(f"| {arm} | {', '.join(str(r['main_epoch']) for r in rows)} | {', '.join(str(r['best_ce_epoch']) for r in rows)} | {sum(r['epoch0_fallback'] for r in rows)} | {rows[0]['trainable_parameters']} |")
    lines += ["", "选择规则在训练前冻结：仅 inner-validation fixed0 正确数严格超过 B_CAL 的 epoch 合格，"
              "从合格 epoch 中取 CE 最低者，平局取最早者；无合格者回退 epoch0，精确保留 B_CAL。"
              "best-CE epoch 是选择诊断，不能直接替换主 epoch；outer refit 只执行选定 epoch 数。"
              "没有使用这里的 held 结果重新挑选 arm、epoch、阈值或 seed。", "", "## whole-C128 成员覆盖与目标深度", ""]
    c = s["candidate_membership"]
    lines += [f"目标 identity 在 C128 内：**{c['present']}/128 ({100*c['coverage']:.2f}%)**；缺失 {c['absent']}。"
              f"按冻结 B_CAL 全 C128 logit 排名，目标在 top2 的有 {c['B_CAL_target_top2']}，在 rank3–128 的有 {c['B_CAL_target_rank3_128']}。",
              "成员覆盖是当前固定候选池的上限，独立于 reader 的排序/动作能力。目标深度采用 logit 降序、"
              "相同分数按 physical row 升序，仅作诊断；实际 HOLD/SWITCH 仍使用已冻结的唯一最大 challenger 及严格阈值规则，平局 HOLD。", "",
              "| 模型（fixed0） | 目标 B_CAL 深度 | 查询数 | 对照错误数 | 救回 | 破坏 | 净变化 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in s["target_rank_strata"]:
        if r["operating_point"] == "zero":
            lines.append(f"| {r['model']} | {r['target_rank_bucket']} | {r['queries']} | {r['baseline_wrong']} | {r['rescues']} | {r['breaks']} | {r['net']:+d} |")
    lines += ["", "rank3–128 的救回可证明当前模型确实纠正过较深候选，但不能单独证明新的候选召回、空间所有权或因果定位能力。"
              "缺失组在该固定候选轴上无法救回。逐查询表保留目标深度、动作、候选 residual 和所选 rival，便于检查具体行为。", "",
              "## 次要结果：继承 B_CAL 阈值", "", "继承阈值直接沿用 SAME F128 各折 B_CAL 阈值，没有新拟合。以下结果与 fixed0 分开解释，不据此改换主 operating point。", "",
              "| 模型 | 对照 | 正确 /128 | 救回 / 破坏 / 净变化 | 配对增益 95% CI（百分点） |", "|---|---|---:|---|---|"]
    for r in s["comparisons"]:
        if r["operating_point"] == "inherited_tau":
            lines.append(f"| {r['model']} | {r['baseline']} ({r['baseline_correct']}/128) | {r['correct']} | {r['rescues']} / {r['breaks']} / {r['net']:+d} | [{100*r['identity_cluster_gain_ci_low']:.2f}, {100*r['identity_cluster_gain_ci_high']:.2f}] |")
    lines += ["", "## 能支持什么，尚不能支持什么", "",
        "**最直接的隔离比较是 MULTI 对 ANCHOR。** 两者参数结构、初始化规则、native token 输入、B_CAL、损失、训练及选择规则相同，"
        "预先定义的差别是 target-free rival graph：ANCHOR 使用 RAW anchor；MULTI 在每个候选处额外加入冻结 B_CAL 最强的非自身、非 RAW rival，去重后取均值。"
        "训练轨迹和最终选择 epoch 可以随这一结构干预而变化。其当前 fixed0 结论为：" + conclusion(structural),
        "", "**QR 是辅助结构对照。** 它使用相同局部输入，但独立候选读出与联合比较读出的参数结构不同，仅近似匹配有效容量；"
        "因此 QR 与 joint reader 的差别不能宣称是完全等参数的单因素因果结论。",
        "", "**不能将与旧 QRR 的差别唯一归因于压缩。** native-token V2 同时改变局部内容表示、匹配/支持交互、读出结构及候选竞争机制等因素。"
        "本报告没有将旧 QRR 数值混入这组主配对比较；若要证明压缩是唯一原因，仍需同输入、同结构、同训练/选择规则的受控压缩消融。",
        "", "**保留 original M。** 原 common features、M 与阶段 RMS、18 维 B_CAL 都保持原绑定；当前改变的是 native-token residual 读出与竞争。"
        "这些结果不构成 M 重设计的证据，也不自动证明 P/HYP、空间归属或全图库候选召回改善。",
        "", "**证据范围仅为 seed0、已打开的 F128 开发面板。** 这不是旧 EVAL128，不是 H593 正式结果替换，也没有外部数据确认。"
        "不从本表选择后续最佳 arm 后再把同一面板称为 untouched confirmation。",
        "", "## 完整性与交付", "",
        f"最终 gate：`{GATE}`。程序复核所有显式 SHA 绑定，包括 protocol/source、15 个独立 receipt、各 fit artifact、baseline、证据 manifest、trace 与最终汇总，"
        "并从最终绑定 curator/gallery 复核 identity 判定、官方计数、配对统计与 CI。该程序不替代原来的模型数值 replay。",
        "", "交付：[结构化摘要](analysis_summary.json)、[分折表](per_fold.csv)、[逐查询诊断](per_query_diagnostics.csv)。"
        "15 份 model.pt 及 checkpoint/trace 大文件保持本地；此目录只含分析摘要和表格，不复制模型权重。", ""]
    return "\n".join(lines)


def encode_csv(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def publish(root, summary, per_fold, diagnostics, audit):
    output = root / "analysis"
    contents = {REPORT: report_text(summary), "per_fold.csv": encode_csv(per_fold),
                "per_query_diagnostics.csv": encode_csv(diagnostics)}
    summary["outputs"] = {name: {"path": str(output / name), "sha256": hashlib.sha256(data.encode()).hexdigest()}
                          for name, data in contents.items()}
    summary["artifacts"] = {"report": summary["outputs"][REPORT],
                            "per_fold": summary["outputs"]["per_fold.csv"],
                            "per_query": summary["outputs"]["per_query_diagnostics.csv"]}
    manifest = [{"path": path, "sha256": value[0]} for path, value in sorted(audit.files.items())]
    summary["integrity"] = {"verified_sha_files": len(manifest), "verified_bindings": manifest,
                            "per_fold_rows": len(per_fold), "per_query_diagnostic_rows": len(diagnostics)}
    contents["analysis_summary.json"] = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    audit.unchanged()
    need(sha(root / "validation.json") == summary["inputs"]["validation"]["sha256"], "FINAL_GATE_CHANGED")
    output.mkdir(parents=True, exist_ok=True)
    # Publish summary last: its hashes describe a complete consistent output set.
    for name, data in contents.items():
        path = output / name
        temporary = path.with_name("." + path.name + f".{os.getpid()}.tmp")
        try:
            with temporary.open("x") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    print(json.dumps({"status": summary["status"], "analysis": str(output), "verified_sha_files": len(manifest),
                      "per_fold_rows": len(per_fold), "per_query_diagnostic_rows": len(diagnostics)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Completed token V2 result root")
    args = parser.parse_args()
    root = args.root.resolve()
    gate_path = root / "validation.json"
    # This must be the first experiment file read. Refusal creates no files.
    if not gate_path.is_file():
        print(json.dumps({"status": "ANALYSIS_REFUSED_FINAL_GATE_MISSING", "required": GATE,
                          "path": str(gate_path), "held_labels_opened": False}))
        return 75
    gate_binding = binding(gate_path)
    validation = read_json(gate_path)
    if validation.get("status") != GATE:
        print(json.dumps({"status": "ANALYSIS_REFUSED_FINAL_GATE_NOT_PASS", "required": GATE,
                          "actual": validation.get("status"), "held_labels_opened": False}))
        return 75
    audit = Audit()
    audit.check(gate_binding)
    protocol, panel, results, bases, fold_of = load_verified(root, validation, audit)
    summary, per_fold, diagnostics = analyze(root, validation, audit, protocol, panel, results, bases, fold_of)
    publish(root, summary, per_fold, diagnostics, audit)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError, ImportError) as error:
        print(json.dumps({"status": "ANALYSIS_FAILED_CLOSED", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
