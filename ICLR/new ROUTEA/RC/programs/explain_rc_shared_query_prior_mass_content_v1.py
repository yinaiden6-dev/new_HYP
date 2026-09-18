#!/usr/bin/env python3
"""Frozen post-hoc two-factor accounting; zero fitting and no model selection.

Cross original/learned candidate M with original/learned (S,Q,R)/max(M,1e-12).
These are interventions at the existing scalar interface, not image-space
counterfactuals or proposals. Same-source corners reuse saved scalar bits.
"""
from pathlib import Path
import json
import sys
import hashlib
import argparse

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))
import run_rc_shared_query_target_prior_development_v1 as base

OUT = ROOT / "results/rc_shared_query_prior_mass_content_v1"
SOURCE = ROOT / "results/rc_shared_query_target_prior_development_v1"
RESULT_SHA = "977ffe3dcd76deb62bb0dde880ee4e9c5694ec25ab0323bd8acfa9a64e05d2f0"
RUNNER_SHA = "b004d6b3c518773d9f31c65f8e9a40f0d43593fe8e06b39b99bd4f33bb951307"
MODES = {"OLD_M_OLD_N": ("UNIFORM", "UNIFORM"),
         "NEW_M_OLD_N": ("REAL", "UNIFORM"),
         "OLD_M_NEW_N": ("UNIFORM", "REAL"),
         "NEW_M_NEW_N": ("REAL", "REAL")}


def main(validate=False):
    import torch
    import numpy as np
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    base.need(base.sha(Path(base.__file__)) == RUNNER_SHA, "RUNNER_SOURCE_DRIFT")
    base.need(base.sha(SOURCE / "result.json") == RESULT_SHA, "RESULT_SOURCE_DRIFT")
    source = base.read(SOURCE / "result.json")
    validated = base.read(SOURCE / "independent_validation.json")
    base.need(validated["result_sha256"] == RESULT_SHA and all(validated["checks"].values()), "SOURCE_NOT_VALIDATED")
    core, frozen, pure = base.modules()
    head, _ = base.load_head({"head_parameter_seal": base.binding(base.HEAD)}, core, pure)
    full = {}
    full_validation = base.read(base.FULL / "validation.json")
    for shard in full_validation["shards"]:
        p = base.FULL / f"shard{shard['shard']:02d}/payload.pt"
        base.need(base.sha(p) == shard["payload_sha256"], "FULL_SHARD_DRIFT")
        for row in torch.load(p, map_location="cpu", mmap=True, weights_only=True)["records"]:
            full[row["execution_ordinal"]] = row
    old = {r["execution_ordinal"]: r for r in source["actions"]["UNIFORM"]}
    real = {r["execution_ordinal"]: r for r in source["actions"]["REAL"]}
    prejoin = base.read(SOURCE / "eval_prejoin.json")
    rows = {mode: [] for mode in MODES}
    records = []
    floors = {mode: 0 for mode in ("UNIFORM", "REAL")}
    corner_logit_checks = 0
    for entry in prejoin:
        ex = entry["execution_ordinal"]
        row = full[ex]
        axis = row["candidate_physical_rows"]
        base.need(axis == entry["candidate_physical_rows"] and row["query_id"] == entry["query_id"], "QUERY_AXIS")
        winner = int(row["base_winner_position"])
        challengers = list(map(int, row["challenger_positions"]))
        raw = row["base_scores"].tolist()
        with np.load(base.checked(entry["arrays"]), allow_pickle=False) as saved:
            arrays = {k: torch.from_numpy(saved[k].copy()) for k in saved.files}
        by_mode = {}
        for mode, (mass_source, normalized_source) in MODES.items():
            m = arrays[mass_source + "__scalars"][:, 1]
            nsource = arrays[normalized_source + "__scalars"]
            if mass_source == normalized_source:
                transformed = nsource.clone()
            else:
                transformed = nsource.clone()
                transformed[:, 1] = m
                for i in (0, 2, 3):
                    transformed[:, i] = m * (nsource[:, i] / nsource[:, 1].clamp_min(1e-12))
            plain = {p: {name: float(transformed[p, j]) for j, name in enumerate(core.SCORE_FIELDS)} for p in range(128)}
            feats = torch.stack([frozen.candidate_feature(raw, plain, c, winner) for c in challengers])
            logits = base.native_logits(feats, head)
            if mass_source == normalized_source:
                base.exact_tensor(feats, arrays[mass_source + "__features"], "CORNER_FEATURE_REPLAY")
                base.exact_tensor(logits, arrays[mass_source + "__logits"], "CORNER_LOGIT_REPLAY")
                corner_logit_checks += logits.numel()
            rows[mode].append({**row, "target_position": old[ex]["target_position"],
                               "real_native_features": {"C_PAIRED": feats}})
            by_mode[mode] = {"features": feats, "logits": logits}
        for mode in floors:
            floors[mode] += int((arrays[mode + "__scalars"][:, 1] <= 1e-12).sum())
        positions = set([old[ex]["proposed_challenger"], real[ex]["proposed_challenger"], old[ex]["target_position"]])
        positions.discard(winner)
        probes = []
        for p in sorted(positions):
            ix = challengers.index(p)
            z = {mode: float(value["logits"][ix]) for mode, value in by_mode.items()}
            oldf = by_mode["OLD_M_OLD_N"]["features"][ix]
            newf = by_mode["NEW_M_NEW_N"]["features"][ix]
            m_effect = .5 * ((z["NEW_M_OLD_N"] - z["OLD_M_OLD_N"]) + (z["NEW_M_NEW_N"] - z["OLD_M_NEW_N"]))
            n_effect = .5 * ((z["OLD_M_NEW_N"] - z["OLD_M_OLD_N"]) + (z["NEW_M_NEW_N"] - z["NEW_M_OLD_N"]))
            base.need(abs(m_effect + n_effect - (z["NEW_M_NEW_N"] - z["OLD_M_OLD_N"])) < 1e-10, "TWO_FACTOR_ACCOUNTING")
            probes.append({"position": p, "physical_row": int(axis[p]), "is_target": p == old[ex]["target_position"],
                           "logits": z, "mass_factor_shapley_logit_delta": m_effect,
                           "normalized_evidence_factor_shapley_logit_delta": n_effect,
                           "feature_weighted_delta": dict(zip(frozen.FEATURE_NAMES, ((newf - oldf) * head.weight).tolist()))})
        records.append({"query_id": entry["query_id"], "execution_ordinal": ex,
                        "uniform_correct": old[ex]["final_correct"], "real_correct": real[ex]["final_correct"],
                        "uniform_final_position": old[ex]["final_position"], "real_final_position": real[ex]["final_position"],
                        "target_position": old[ex]["target_position"], "probes": probes})
    actions = {mode: pure["actions"](head.weight, float(head.bias), value, "NATIVE7", "C_PAIRED") for mode, value in rows.items()}
    for mode, original in (("OLD_M_OLD_N", "UNIFORM"), ("NEW_M_NEW_N", "REAL")):
        base.need(base.encode(actions[mode]) == base.encode(source["actions"][original]), "EXACT_ORIGINAL_ACTION_REPLAY")
    comparisons = {}
    for mode, value in actions.items():
        gains = [r["query_id"] for r in value if r["final_correct"] and not old[r["execution_ordinal"]]["final_correct"]]
        losses = [r["query_id"] for r in value if not r["final_correct"] and old[r["execution_ordinal"]]["final_correct"]]
        comparisons[mode] = {"new_correct_vs_original": gains, "lost_correct_vs_original": losses, "paired_net": len(gains) - len(losses)}
    result = {"status": "FROZEN_QUERY_PRIOR_MASS_NORMALIZED_EVIDENCE_ACCOUNTING_COMPLETE",
              "sources": {"program": base.binding(Path(__file__)), "runner": base.binding(Path(base.__file__)),
                          "result": base.binding(SOURCE / "result.json"), "validation": base.binding(SOURCE / "independent_validation.json"),
                          "prejoin": base.binding(SOURCE / "eval_prejoin.json"), "head": base.binding(base.HEAD)},
              "interventions": MODES, "formula": "M_destination*(S,Q,R)_source/max(M_source,1e-12); same-source corners reuse exact scalar bits",
              "floor_candidate_counts": floors, "exact_corner_logit_checks": corner_logit_checks,
              "metrics": {mode: pure["summary"](value) for mode, value in actions.items()},
              "comparisons_to_original": comparisons, "actions": actions, "records": records,
              "new_training_updates": 0, "model_selection_or_deployment": False, "HYP_GO_claimed": False,
              "limitations": ["Post-hoc computational interface interventions on opened EVAL32, not a new trained model.",
                              "Normalized evidence still includes RoMa reference visibility and query weighting.",
                              "No pixel intervention or connected-HYP inference."]}
    if validate:
        base.need(base.read(OUT / "result.json") == result, "ACCOUNTING_REPLAY_DRIFT")
        base.atomic(OUT / "validation.json", {"status": "FROZEN_QUERY_PRIOR_FACTOR_ACCOUNTING_REPLAY_PASS", "result_sha256": base.sha(OUT / "result.json"), "same_program_replay": True, "original_both_corner_actions_exact": True})
    else:
        base.need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
        base.atomic(OUT / "result.json", result)
    print(json.dumps({"metrics": result["metrics"], "comparisons": comparisons, "floor_candidate_counts": floors}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    main(p.parse_args().validate)
