"""Posthoc frozen-operator diagnostics; no model training, cache loads, or OOF claim.

The two interventions are fixed in advance: MULTI weights + RAW-only edges, and
MULTI weights + common {RAW, best frozen-base nonanchor} pool with self-edge zero.
All inputs are existing finalized artifacts. No coefficient search is performed.
"""
import hashlib
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARMS = ("TOKEN_QR", "TOKEN_QRR_ANCHOR", "TOKEN_QRR_MULTI")
LOST = {"H593-1af45e7f0d56a52adeb60b07", "H593-2e5153cd0e9bd1dddbb75dea"}
bindings = {}


def read(path, expected=None):
    path = Path(path)
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if expected:
        assert sha == expected, (path, "SHA mismatch")
    bindings[str(path)] = sha
    return json.loads(payload)


def bound(binding):
    return read(binding["path"], binding["sha256"])


def decision(rows, scores):
    anchor = next(r["position"] for r in rows if r["raw_anchor"])
    assert scores[anchor] == 0.0
    choices = [i for i in range(len(rows)) if i != anchor]
    maximum = max(scores[i] for i in choices)
    tops = [i for i in choices if scores[i] == maximum]
    # Immutable physical IDs only choose rivals; tied best action must HOLD.
    return tops[0] if maximum > 0.0 and len(tops) == 1 else anchor


def cycle_summary(cycles):
    absolute = sorted(abs(r["cycle"]) for r in cycles)
    edge_scale = [abs(r[k]) for r in cycles for k in ("target_candidate_vs_RAW", "RAW_vs_best_base", "target_candidate_vs_best_base")]
    rms = math.sqrt(statistics.mean(x * x for x in absolute))
    edge_rms = math.sqrt(statistics.mean(x * x for x in edge_scale))
    return {"triangles": len(cycles), "nonzero_at_1e_10": sum(x > 1e-10 for x in absolute),
            "mean_absolute_cycle": statistics.mean(absolute), "median_absolute_cycle": statistics.median(absolute),
            "p95_absolute_cycle_nearest_rank": absolute[math.ceil(0.95 * len(absolute)) - 1],
            "max_absolute_cycle": max(absolute), "rms_cycle": rms, "rms_three_edges": edge_rms,
            "rms_cycle_over_rms_edge": rms / edge_rms if edge_rms else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-originals", action="store_true", help="Revalidate original bound files and refresh the portable input snapshot.")
    args = parser.parse_args()
    snapshot_path = OUT / "portable_input_snapshot.json"
    if args.from_originals or not snapshot_path.exists():
        validation = read(ROOT / "validation.json")
        assert validation["held_labels_read_only_after_all15_checks"] is True
        protocol = bound(validation["protocol"])
        candidates = bound(validation["artifacts"]["candidate_predictions.json"])
        joined_doc = bound(validation["artifacts"]["joined_predictions.json"])
        gallery = {str(r["physical_row"]): r["identity"] for r in bound(protocol["gallery"])["records"]}
        identities = {r["query_id"]: r["identity"] for r in bound(validation["sources"]["curator"])["records"]}
        fits = {}
        for fold in (0, 4):
            for arm in ARMS:
                fit = read(ROOT / f"fold{fold}/seed0/{arm}/result.json")
                fits[f"{fold}/{arm}"] = {k: fit[k] for k in ("best_epoch", "model_sha256")}
        # Keep exactly the values consumed by this script. QR/ANCHOR pair edges
        # are not used by interventions on frozen MULTI and are not duplicated.
        candidates["records"] = [{k: v for k, v in r.items() if r["model"] == "TOKEN_QRR_MULTI" or k in ("model", "query_id", "position", "physical_row", "logit")} for r in candidates["records"]]
        snapshot = {"description": "Portable derived snapshot; numeric values copied without rounding from the SHA-verified final artifacts. No token tensors or model states are required to replay these frozen-edge operators.",
                    "original_source_bindings_sha256": dict(bindings), "protocol": protocol,
                    "candidates": candidates, "joined_doc": joined_doc, "gallery": gallery, "identities": identities, "critical_fit_metadata": fits}
        snapshot_path.write_text(json.dumps(snapshot, separators=(",", ":")) + "\n")
    snapshot = read(snapshot_path)
    bindings.update(snapshot["original_source_bindings_sha256"])
    protocol, candidates, joined_doc = (snapshot[k] for k in ("protocol", "candidates", "joined_doc"))
    gallery = {int(k): v for k, v in snapshot["gallery"].items()}
    identities, fits = snapshot["identities"], snapshot["critical_fit_metadata"]
    joined = {(r["model"], r["query_id"]): r for r in joined_doc["records"] if r["operating_point"] == "zero"}
    grouped = defaultdict(list)
    for row in candidates["records"]:
        grouped[row["model"], row["query_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["position"])
        assert [r["position"] for r in rows] == list(range(128))
        assert len({r["physical_row"] for r in rows}) == 128

    output, all_cycles, target_cycles, max_error = [], [], [], 0.0
    for q in sorted(protocol["panel_query_ids"]):
        rows = grouped["TOKEN_QRR_MULTI", q]
        a = next(r["position"] for r in rows if r["raw_anchor"])
        rank = sorted((r for r in rows if not r["raw_anchor"]), key=lambda r: (-r["zero_base"], r["physical_row"]))
        b, c = (rank[i]["position"] for i in (0, 1))
        edge = [{r["rival_position"]: r["pair_difference"] for r in row["edges"]} for row in rows]
        for i in range(128):
            assert set(edge[i]) == ({b} if i == a else {a, c if i == b else b})
        d_native = [sum(e.values()) / len(e) for e in edge]
        s_native = [0.0 if i == a else r["zero_base"] + d_native[i] - d_native[a] for i, r in enumerate(rows)]
        max_error = max(max_error, max(abs(s_native[i] - row["logit"]) for i, row in enumerate(rows)))
        assert max_error < 2e-12
        assert rows[decision(rows, s_native)]["physical_row"] == joined["TOKEN_QRR_MULTI", q]["selected"]
        assert abs(edge[a][b] + edge[b][a]) < 2e-12
        d_anchor = [0.0 if i == a else edge[i][a] for i in range(128)]
        # Every candidate uses exactly the same two opponent slots including zero self-edges.
        d_common = [0.5 * ((0.0 if i == a else edge[i][a]) + (0.0 if i == b else edge[i][b])) for i in range(128)]
        sets = {"native_MULTI": s_native,
                "same_MULTI_weights_RAW_only": [0.0 if i == a else rows[i]["zero_base"] + d_anchor[i] for i in range(128)],
                "same_MULTI_weights_common_opponents": [0.0 if i == a else rows[i]["zero_base"] + d_common[i] - d_common[a] for i in range(128)]}
        g = next((i for i, r in enumerate(rows) if gallery[r["physical_row"]] == identities[q]), None)
        cycles = [{"query_id": q, "physical_row": rows[i]["physical_row"], "cycle": edge[i][a] + edge[a][b] - edge[i][b],
                   "target_candidate_vs_RAW": edge[i][a], "RAW_vs_best_base": edge[a][b], "target_candidate_vs_best_base": edge[i][b]}
                  for i in range(128) if i not in (a, b)]
        all_cycles.extend(cycles)
        if g not in (None, a, b):
            target_cycles.append(next(r for r in cycles if r["physical_row"] == rows[g]["physical_row"]))
        cases = {}
        for name, scores in sets.items():
            selected = decision(rows, scores)
            cases[name] = {"selected_position": selected, "selected_physical_row": rows[selected]["physical_row"],
                           "correct": gallery[rows[selected]["physical_row"]] == identities[q],
                           "target_score": None if g is None else scores[g],
                           "best_base_nonanchor_score": scores[b],
                           "target_minus_best_base_nonanchor": None if g is None else scores[g] - scores[b],
                           "logits": scores}
        cross_arm = {}
        for arm in ARMS:
            rr = grouped[arm, q]
            assert [r["physical_row"] for r in rr] == [r["physical_row"] for r in rows]
            cross_arm[arm] = {"selected": joined[arm, q]["selected"], "correct": joined[arm, q]["correct"],
                             "target_logit": None if g is None else rr[g]["logit"], "best_base_nonanchor_logit": rr[b]["logit"],
                             "target_minus_best_base_nonanchor": None if g is None else rr[g]["logit"] - rr[b]["logit"]}
        record = {"query_id": q, "fold": rows[0]["fold"], "raw_anchor": rows[a]["physical_row"],
                  "target": None if g is None else rows[g]["physical_row"],
                  "best_base_nonanchor": rows[b]["physical_row"], "second_base_nonanchor": rows[c]["physical_row"],
                  "baseline": joined["B_CAL", q], "cross_trained_arm_comparison": cross_arm, "probes": cases,
                  "triangle_cycle_summary": cycle_summary(cycles)}
        if q in LOST:
            for arm in ARMS:
                fit = fits[f"{rows[0]['fold']}/{arm}"]
                cross_arm[arm].update(selected_epoch=fit["best_epoch"], model_sha256=fit["model_sha256"])
            assert g not in (a, b, c)
            da, db = edge[g][a], edge[b][a]
            eg, eb = edge[g][b], edge[b][c]
            record["critical_edges"] = {"target_vs_RAW": da, "target_vs_best_base": eg,
                                        "best_base_vs_RAW": db, "best_base_vs_second_base": eb,
                                        "second_base_vs_RAW": edge[c][a], "RAW_vs_best_base": edge[a][b]}
            record["margin_decomposition"] = {
                "base_target_minus_best_base": rows[g]["zero_base"] - rows[b]["zero_base"],
                "RAW_only_residual_margin": da - db,
                "extra_edge_residual_margin": eg - eb,
                "native_residual_margin": 0.5 * ((da - db) + (eg - eb)),
                "change_in_target_D_from_mean": 0.5 * (eg - da),
                "change_in_best_base_D_from_mean": 0.5 * (eb - db),
                "RAW_D_subtraction_common_shift_nonanchors": -edge[a][b],
                "net_target_vs_RAW_change_native_minus_RAW_only": cases["native_MULTI"]["target_score"] - cases["same_MULTI_weights_RAW_only"]["target_score"],
                "target_pair_transitivity_defect": eg - (da - db),
                "triangle_cycle_target_RAW_best_base": da + edge[a][b] - eg,
                "triangle_cycle_abs_over_mean_abs_three_edges": abs(da + edge[a][b] - eg) / statistics.mean((abs(da), abs(edge[a][b]), abs(eg))),
                "best_base_second_pair_transitivity_defect": eb - (db - edge[c][a]),
                "transitive_opponent_asymmetry_margin_term": -0.5 * (db - edge[c][a]),
                "pair_nontransitivity_correction_to_asymmetry_term": 0.5 * ((eg - (da - db)) - (eb - (db - edge[c][a]))),
                "native_minus_RAW_only_margin": cases["native_MULTI"]["target_minus_best_base_nonanchor"] - cases["same_MULTI_weights_RAW_only"]["target_minus_best_base_nonanchor"],
                "native_minus_common_opponents_margin": cases["native_MULTI"]["target_minus_best_base_nonanchor"] - cases["same_MULTI_weights_common_opponents"]["target_minus_best_base_nonanchor"]}
            record["critical_candidate_native_records"] = {name: rows[i] for name, i in (("a_RAW", a), ("b_base_best", b), ("c_base_second", c), ("g_target", g))}
        output.append(record)

    summaries = {}
    for name in output[0]["probes"]:
        summary = {"queries": len(output), "correct": sum(r["probes"][name]["correct"] for r in output), "comparisons": {}}
        for ref in ("B_CAL", "TOKEN_QRR_MULTI", "TOKEN_QRR_ANCHOR"):
            rescued = [r["query_id"] for r in output if r["probes"][name]["correct"] and not joined[ref, r["query_id"]]["correct"]]
            broken = [r["query_id"] for r in output if not r["probes"][name]["correct"] and joined[ref, r["query_id"]]["correct"]]
            changed = [r["query_id"] for r in output if r["probes"][name]["selected_physical_row"] != joined[ref, r["query_id"]]["selected"]]
            bc = sum(joined[ref, r["query_id"]]["correct"] for r in output)
            summary["comparisons"][ref] = {"baseline_correct": bc, "rescues": len(rescued), "breaks": len(broken),
                "net": len(rescued) - len(broken), "changed_decisions": len(changed), "baseline_correct_loss_rate": len(broken) / bc,
                "rescue_query_ids": rescued, "break_query_ids": broken, "changed_query_ids": changed}
        summaries[name] = summary
    result = {"status": "POSTHOC_SAVED_EDGE_DIAGNOSTICS_COMPLETE", "evidence_level": "Posthoc mechanism probes on existing F128 seed0 held predictions; not a new OOF evaluation or validated model.",
              "scope": "Frozen trained MULTI weights; fixed saved-edge operator interventions only. Full-gallery recall, retraining, and external evaluation were not performed.",
              "source_bindings_sha256": bindings, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "native_reconstruction_max_abs_error": max_error,
              "decision_rule": "strict challenger > 0; equal highest challenger scores HOLD; rival tie uses ascending immutable physical row",
              "architecture_limits": ["Cached scalar mean_v and baseline M remain unchanged.", "MULTI evaluates RAW plus one frozen-base rival per candidate, not all-pairs competition.", "ANCHOR/MULTI have identical parameter shapes but separately trained weights; cross-arm margin differences are not isolated operator causal effects.", "No learned set-aware aggregation or opponent-strength adjustment exists in the native mean."],
              "triangle_cycle_definition": "delta(g,RAW)+delta(RAW,b)-delta(g,b); every g outside {RAW,b}, b = highest frozen B_CAL nonanchor. QR is a potential-difference model by construction; QRR enforces antisymmetry but not zero triangle cycles.",
              "triangle_cycles_all_candidates": cycle_summary(all_cycles), "triangle_cycles_present_targets_outside_RAW_and_best": cycle_summary(target_cycles),
              "summaries": summaries, "two_lost_queries": [r for r in output if r["query_id"] in LOST], "all_query_probes": output}
    (OUT / "saved_edge_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    compact = {"native_reconstruction_max_abs_error": max_error, "summaries": summaries,
               "two_lost_queries": [{k: v for k, v in r.items() if k != "critical_candidate_native_records" and k != "probes"} | {"probes": {n: {k: v for k, v in x.items() if k != "logits"} for n, x in r["probes"].items()}} for r in output if r["query_id"] in LOST]}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
