"""Read-only numerical replay of the completed pilot's TRAIN endpoint.

Uses sealed scalar predictions, no model forward, no probe labels, no training.
Writes a separate audit; never modifies the frozen v2 experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/rc_prellm_m_adapter_v2"
OUTPUT = ROOT / "results/rc_internal_m_v2_learning_scope_audit_20260924.json"


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def main():
    manifest_path = SOURCE / "input_manifest.json"
    manifest = read(manifest_path)
    theta = manifest["frozen_head"]["theta"]
    rows = manifest["train_rows"]
    assert len(rows) == 16 and len({r["query_id"] for r in rows}) == 16
    collected, sources, max_error = {}, [binding(manifest_path)], 0.0
    shuffled = {arm: {"abs_delta_L": [], "abs_delta_logit": [], "changed": 0}
                for arm in ("PRE_REAL", "POST_REAL")}
    for row in rows:
        path = SOURCE / "independent_numpy" / (row["query_id"] + ".json")
        data = read(path)
        sources.append(binding(path))
        assert data["candidate_ids"] == row["candidate_ids"]
        assert data["M"] == row["M"] and data["theta"] == theta
        winner = row["winner_index"]
        target, = row["target_positions"]
        ids = [i for i in range(128) if i != winner]
        raw = row["raw_scores"]
        center = statistics.mean(raw)
        sd = max(math.sqrt(statistics.mean((x - center) ** 2 for x in raw)), 1e-12)
        mass = row["M"]
        base = data["models"]["FRESH_ZERO"]
        for arm, prediction in data["models"].items():
            content = prediction["L"]
            assert len(content) == 128 and prediction["challenger_positions"] == ids
            def sym(a, b):
                return (a - b) / (abs(a) + abs(b) + 1e-12)
            scores = []
            for i in ids:
                features = [(raw[i] - raw[winner]) / sd,
                            sym(mass[i] * content[i], mass[winner] * content[winner]),
                            sym(mass[i], mass[winner]), sym(content[i], content[winner]), 1.0]
                scores.append(math.fsum(a * b for a, b in zip(features, theta)))
            max_error = max(max_error, max(abs(a - b) for a, b in zip(scores, prediction["logits"])))
            largest = max(range(127), key=scores.__getitem__)
            chosen = ids[largest] if scores[largest] > 0 else winner
            assert chosen == prediction["prediction_position"]
            loss = (softplus(max(scores)) if target == winner else
                    softplus(-scores[ids.index(target)]) +
                    softplus(max(z for i, z in zip(ids, scores) if i != target)))
            rec = collected.setdefault(arm, {"loss": [], "abs_delta_L": [],
                                             "abs_delta_logit": [], "changed": 0, "correct": 0})
            rec["loss"].append(loss)
            rec["abs_delta_L"].extend(abs(a - b) for a, b in zip(content, base["L"]))
            rec["abs_delta_logit"].extend(abs(a - b) for a, b in zip(scores, base["logits"]))
            rec["changed"] += chosen != base["prediction_position"]
            rec["correct"] += row["candidate_identities"][chosen] == row["candidate_identities"][target]
        for arm, rec in shuffled.items():
            native, intervention = data["models"][arm], data["models"][arm + "_SHUFFLED"]
            rec["abs_delta_L"].extend(abs(a-b) for a,b in zip(native["L"], intervention["L"]))
            rec["abs_delta_logit"].extend(abs(a-b) for a,b in zip(native["logits"], intervention["logits"]))
            rec["changed"] += native["prediction_position"] != intervention["prediction_position"]
    assert max_error < 1e-10
    initial = statistics.mean(collected["FRESH_ZERO"]["loss"])
    summaries = {}
    for arm, rec in collected.items():
        mean = statistics.mean(rec["loss"])
        summaries[arm] = {"queries": 16, "correct": rec["correct"], "mean_cost1": mean,
                          "relative_loss_decrease": (initial-mean)/initial,
                          "changed_decisions_vs_zero": rec["changed"],
                          "mean_abs_delta_L": statistics.mean(rec["abs_delta_L"]),
                          "max_abs_delta_L": max(rec["abs_delta_L"]),
                          "mean_abs_delta_logit": statistics.mean(rec["abs_delta_logit"])}
    sensitivity = {arm: {"changed_decisions": rec["changed"],
                         "mean_abs_delta_L": statistics.mean(rec["abs_delta_L"]),
                         "max_abs_delta_L": max(rec["abs_delta_L"]),
                         "max_abs_delta_logit": max(rec["abs_delta_logit"])}
                   for arm, rec in shuffled.items()}
    result = {"status": "TRAIN16_SCALAR_REPLAY_PASS", "script": binding(Path(__file__)),
              "sources": sources, "panel": "Original v2 TRAIN16, natural ColNomic C128, fold0",
              "head": "frozen COST1_REFIT_M1Q0R0", "updates_per_arm": 16,
              "query_passes": 1, "max_logit_replay_error": max_error,
              "new_model_forwards": 0, "new_training": False, "probe_labels_read": False,
              "same_panel_endpoint_metrics": summaries, "train_internal_M_shuffle": sensitivity,
              "interpretation": ["Nonzero changes do not establish adequate fitting.",
                                 "All heads still directly read real external M.",
                                 "This audit neither tests M-free inference nor establishes internal-use impossibility."]}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "status": result["status"],
                      "max_logit_replay_error": max_error, "metrics": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
