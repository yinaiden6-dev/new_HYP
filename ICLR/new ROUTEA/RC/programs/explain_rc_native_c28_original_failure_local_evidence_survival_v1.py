#!/usr/bin/env python3
"""Append the three remaining original C28 failures to the fixed-cue trace.

The critical wrong is the ORIGINAL C28 final reference. Formula, complete
query/candidate axes, saved prior, and all scientific boundaries are unchanged.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "programs/explain_rc_shared_query_prior_local_evidence_survival_v1.py"
CORE_SHA = "2f3e15b94767364d381c5daa9278e9f3026ee30ee65935764e374a65fdd46ebd"
spec = importlib.util.spec_from_file_location("frozen_cue_trace", CORE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.need(base.sha(CORE) == CORE_SHA, "FROZEN_TRACE_FORMULA_DRIFT")
CASES = {194: (5, 46), 307: (50, 90), 495: (52, 50)}
PHYSICAL = {194: (1024, 2461), 307: (4865, 5165), 495: (221, 219)}
base.CASES = CASES
OUT = ROOT / "results/rc_native_c28_original_failure_local_evidence_survival_v1"
REPORT = ROOT / "reports/REPORT_RC_NATIVE_C28_ORIGINAL_FAILURE_LOCAL_EVIDENCE_SURVIVAL_20260909.md"
VISUAL = ROOT / "results/rc_native_c28_original_failure_visual_facts_v1/result.json"
VISUAL_SHA = "a969e6f13e6729b3da23f228e596dfc040571498039487b7e89c180b257bf373"
EARLIER = ROOT / "results/rc_shared_query_prior_local_evidence_survival_v1/result.json"


def inputs():
    import numpy as np
    need, sha, read, bind, checked = base.need, base.sha, base.read, base.bind, base.checked
    need(sha(base.SOURCE / "result.json") == base.RESULT_SHA, "ORIGINAL_RESULT_PIN")
    result = read(base.SOURCE / "result.json")
    need(sha(VISUAL) == VISUAL_SHA and sha(EARLIER) == "408c611ff5e4895868369eeb7abdd61002296d363773740aeddecffe1af835ba", "EXISTING_EVIDENCE_PIN")
    sources = {"program": bind(__file__), "fixed_formula_source": bind(CORE), "result": bind(base.SOURCE / "result.json"),
               "visual_facts": bind(VISUAL), "earlier_ex27_trace": bind(EARLIER)}
    authority = ROOT / "registry/rc_shared_query_target_prior_development_authority_v1_20260909.json"
    need(sha(authority) == result["authority_sha256"], "AUTHORITY_PIN")
    auth = read(authority); sources["authority"] = bind(authority)
    for key in ("cache_manifest", "cache_validation", "head_parameter_seal"):
        sources[key] = auth["sources"][key]; checked(sources[key])
    cache = read(checked(sources["cache_manifest"])); cv = read(checked(sources["cache_validation"]))
    need(cv["manifest_sha256"] == sources["cache_manifest"]["sha256"] and all(cv["checks"].values()), "CACHE_VALIDATION")
    for key in ("parameters", "eval_prejoin", "eval_prejoin_seal", "independent_validation"):
        sources[key] = bind(base.SOURCE / (key + ".json"))
    need(sources["parameters"]["sha256"] == result["parameters_sha256"] and sources["eval_prejoin_seal"]["sha256"] == result["eval_prejoin_seal_sha256"], "FROZEN_OUTPUT_PIN")
    seal = read(base.SOURCE / "eval_prejoin_seal.json")
    need(sources["eval_prejoin"]["sha256"] == seal["eval_prejoin_sha256"], "PREJOIN_PIN")
    v = read(base.SOURCE / "independent_validation.json")
    need(v["result_sha256"] == base.RESULT_SHA and all(v["checks"].values()), "SOURCE_VALIDATION")
    head = read(checked(sources["head_parameter_seal"]))
    pre = {r["execution_ordinal"]: r for r in read(base.SOURCE / "eval_prejoin.json")}
    original = {r["execution_ordinal"]: r for r in result["actions"]["UNIFORM"]}
    need({ex for ex, row in original.items() if not row["final_correct"]} == {27, 194, 307, 495}, "ORIGINAL_C28_FAILURE_POPULATION")
    episodes = {}
    for entry in cache["records"]:
        ex = entry["execution_ordinal"]
        if entry["kind"] != "FULL" or ex not in CASES:
            continue
        meta = read(checked(entry)); prediction = pre[ex]; old = original[ex]
        need(meta["query_id"] == prediction["query_id"] == old["query_id"] and meta["axis"] == prediction["candidate_physical_rows"], "QUERY_AXIS")
        need((old["target_position"], old["final_position"]) == CASES[ex]
             and (old["target_physical_row"], old["final_physical_row"]) == PHYSICAL[ex], "ORIGINAL_CRITICAL_WRONG_PIN")
        with np.load(checked(meta["arrays"]), allow_pickle=False) as handle:
            x = {k: handle[k].copy() for k in handle.files}
        with np.load(checked(prediction["arrays"]), allow_pickle=False) as handle:
            saved = {k: handle[k].copy() for k in handle.files}
        need(x["a"].shape == x["b"].shape == x["wq"].shape and x["a"].shape[0] == 128 and np.array_equal(x["candidate_positions"], np.arange(128)), "FULL_CANDIDATE_AXIS")
        episodes[ex] = {"meta": meta, "input_arrays": x, "saved": saved,
            "sources": {"metadata": entry, "cache_arrays": meta["arrays"], "frozen_outputs": prediction["arrays"]}}
    need(set(episodes) == set(CASES), "THREE_REMAINING_CASES")
    return sources, head, episodes


def run():
    import numpy as np
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    base.need(not OUT.exists() and not REPORT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    sources, head, episodes = inputs()
    records = []
    for ex in sorted(episodes):
        stages = base.torch_stages(episodes[ex], head)
        row, arrays = base.make_case(ex, episodes[ex], stages)
        row["critical_wrong_definition"] = "ORIGINAL_NATIVE_C28_FINAL_REFERENCE"
        path = OUT / f"exec{ex:04d}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o444); row["arrays"] = base.bind(path); records.append(row)
    earlier = base.read(EARLIER)
    result = {"status": "RC_NATIVE_C28_ORIGINAL_FAILURE_LOCAL_EVIDENCE_SURVIVAL_V1_COMPLETE", "sources": sources,
        "scope": "Remaining three failures of original fixed RAW_C128_NATIVE7_C_PAIRED_28_of_32; ex27 already traced",
        "case_count": 3, "all_original_failure_executions": [27, 194, 307, 495], "earlier_completed_execution": 27,
        "candidate_count": 128, "rivals_per_subject": 127, "records": records,
        "stage_definitions": earlier["stage_definitions"], "fixed_sets": earlier["fixed_sets"],
        "producer_exact_checks": {"old_new_scalar_values": 3072, "old_new_feature_values": 4572, "old_new_logits": 762},
        "numpy_validation_tolerances": earlier["numpy_validation_tolerances"], "limits": earlier["limits"],
        "new_training_updates": 0, "new_encoder_or_RoMa_forwards": 0, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
    base.save_json(OUT / "result.json", result)
    lines = ["# 原最佳 Native C28 其余三个失败案例的固定证据追踪", "", "固定原 RAW C128、NATIVE7/C_PAIRED、opened EVAL32。critical wrong 严格使用原28/32系统的最终候选；不使用新prior后来换出的其它wrong。原失败集合为 execution27/194/307/495；27已在上一份完整追踪中封存，本次仅补其余三个。", "",
        "公式完整复用既有 A→B(reference visibility)→原C逐token贡献→新C逐token贡献。同一完整query轴与127 rivals，不选K。", "",
        "| Query | 对象 | 固定A独占集：A→B→原C→新C | α均值 | α ESS | 原→新visibility占比 |", "|---|---|---|---:|---:|---|"]
    for row in records:
        for name, item in row["subjects"].items():
            chain = " → ".join(str(item["stages"][s]["fixed_A_positive_survivor_count"]) for s in base.STAGES)
            st = item["alpha_and_visibility"]
            lines.append(f"| {row['query_id']} | {name} ({item['physical_row']}) | {chain} | {st['selected_alpha_mean']:.6g} | {st['selected_alpha_ESS']:.3f} | {st['original_selected_wq_fraction']:.6g} → {st['effective_selected_alpha_wq_fraction']:.6g} |")
    lines += ["", "A→B只改变固定reference visibility；B→C包含query visibility、mass和归一化。局部贡献次序不等于最终六feature动作次序。wrong的独占token不能证明图像中真实存在该物体，token坐标不等于像素感受野或ownership。", "",
              f"完整margin向量、所有正集合、固定最佳witness的128候选值及source SHA：[result.json]({OUT / 'result.json'})。结果SHA：`{base.sha(OUT / 'result.json')}`。零forward、零训练、没有新模型。", ""]
    with REPORT.open("x") as stream:
        stream.write("\n".join(lines)); stream.flush(); os.fsync(stream.fileno())
    REPORT.chmod(0o444)
    print(json.dumps({"status": result["status"], "result": base.bind(OUT / "result.json"), "report": base.bind(REPORT)}, sort_keys=True), flush=True)


def validate():
    import numpy as np
    sources, head, episodes = inputs()
    result = base.read(OUT / "result.json")
    base.need(result["sources"] == sources, "VALIDATION_SOURCE_DRIFT")
    errors = {"scalar_max_abs": 0., "feature_max_abs": 0., "margin_max_abs": 0., "stage_max_abs": 0.}
    counts = {"queries": 0, "subject_stages": 0, "complete127_margin_entries": 0, "old_new_scalar_values": 0, "old_new_feature_values": 0}
    for row in result["records"]:
        ex = row["execution_ordinal"]; ep = episodes[ex]; x, frozen = ep["input_arrays"], ep["saved"]
        alpha = frozen["REAL__alpha"]
        stages = {base.STAGES[0]: x["a"], base.STAGES[1]: x["b"]}
        for mode, stage in (("UNIFORM", base.STAGES[2]), ("REAL", base.STAGES[3])):
            prior = np.ones_like(alpha) if mode == "UNIFORM" else alpha
            ew = x["wq"] * prior[None, :]
            rolled = np.stack([np.roll(ew[g], int(x["qshift"][g])) for g in range(128)])
            den = np.maximum(ew.sum(axis=1), 1e-12)
            mass = np.sqrt(ew.mean(axis=1) * x["wr_mean"])
            qm = np.sqrt(rolled.mean(axis=1) * x["wr_mean"])
            rm = np.sqrt(ew.mean(axis=1) * x["wr_rolled_mean"])
            scalars = np.stack((mass * (ew * x["b"]).sum(axis=1) / den, mass,
                qm * (rolled * x["b"]).sum(axis=1) / np.maximum(rolled.sum(axis=1), 1e-12),
                rm * (ew * x["br"]).sum(axis=1) / den), axis=1)
            errors["scalar_max_abs"] = max(errors["scalar_max_abs"], float(np.abs(scalars - frozen[mode + "__scalars"]).max()))
            base.need(np.allclose(scalars, frozen[mode + "__scalars"], rtol=base.RTOL, atol=base.ATOL), "NUMPY_SCALAR_DRIFT")
            winner = ep["meta"]["winner"]; cs = np.array([g for g in range(128) if g != winner]); raw = x["raw_scores"]
            mean = sum(map(float, raw)) / 128; std = (sum((float(v) - mean) ** 2 for v in raw) / 128) ** .5
            sym = lambda v: (v[cs] - v[winner]) / (np.abs(v[cs]) + abs(v[winner]) + 1e-12)
            features = np.stack(((raw[cs] - raw[winner]) / max(std, 1e-12), sym(scalars[:, 0]), sym(scalars[:, 1]),
                sym(scalars[:, 0] / np.maximum(scalars[:, 1], 1e-12)), sym(scalars[:, 0] - scalars[:, 2]), sym(scalars[:, 0] - scalars[:, 3])), axis=1)
            errors["feature_max_abs"] = max(errors["feature_max_abs"], float(np.abs(features - frozen[mode + "__features"]).max()))
            base.need(np.allclose(features, frozen[mode + "__features"], rtol=base.RTOL, atol=base.FEATURE_ATOL), "NUMPY_FEATURE_DRIFT")
            stages[stage] = mass[:, None] * ew * x["b"] / den[:, None]
            counts["old_new_scalar_values"] += scalars.size; counts["old_new_feature_values"] += features.size
        with np.load(base.checked(row["arrays"]), allow_pickle=False) as saved:
            base.need(np.array_equal(saved["alpha"], alpha), "ALPHA_DRIFT")
            for subject, g in zip(("TARGET", "CRITICAL_WRONG"), CASES[ex]):
                rivals = np.array([h for h in range(128) if h != g]); item = row["subjects"][subject]
                base.need(np.array_equal(saved[subject + "__rival_positions"], rivals), "RIVALS_DRIFT")
                am = x["a"][g] - x["a"][rivals].max(axis=0); fixed = np.flatnonzero(am > 0); best = int(am.argmax())
                base.need(fixed.tolist() == item["fixed_A_positive_indices"] and np.array_equal(saved[subject + "__fixed_A_positive_indices"], fixed), "FIXED_A_SET_DRIFT")
                base.need(best == item["best_A_witness_index"] and base.weight_stats(alpha, x["wq"][g], fixed) == item["alpha_and_visibility"], "WITNESS_OR_ALPHA_STATS_DRIFT")
                for stage, values in stages.items():
                    key = subject + "__" + stage
                    errors["stage_max_abs"] = max(errors["stage_max_abs"], float(np.abs(values - saved[stage + "__values"]).max()))
                    base.need(np.allclose(values, saved[stage + "__values"], rtol=base.RTOL, atol=base.ATOL), "NUMPY_STAGE_DRIFT")
                    margins = np.stack([values[g] - values[h] for h in rivals]); worst = margins.min(axis=0)
                    errors["margin_max_abs"] = max(errors["margin_max_abs"], float(np.abs(margins - saved[key + "__all127_margins"]).max()))
                    base.need(np.allclose(margins, saved[key + "__all127_margins"], rtol=base.RTOL, atol=base.ATOL), "NUMPY_MARGIN_DRIFT")
                    base.need(np.array_equal(np.flatnonzero(worst > 0), saved[key + "__positive_indices"]) and fixed[worst[fixed] > 0].tolist() == item["stages"][stage]["fixed_A_positive_survivor_indices"], "ALL_POSITIVE_SET_MEMBERSHIP_DRIFT")
                    base.need(np.allclose(values[:, best], saved[key + "__all128_values_at_fixed_A_best_witness"], rtol=base.RTOL, atol=base.ATOL), "ALL128_WITNESS_DRIFT")
                    counts["subject_stages"] += 1; counts["complete127_margin_entries"] += margins.size
        counts["queries"] += 1
    base.need(counts["queries"] == 3 and counts["subject_stages"] == 24 and counts["old_new_scalar_values"] == 3072 and counts["old_new_feature_values"] == 4572, "REPLAY_COUNTS")
    receipt = {"status": "RC_NATIVE_C28_ORIGINAL_FAILURE_LOCAL_EVIDENCE_SURVIVAL_V1_NUMPY64_VALIDATION_PASS",
        "result": base.bind(OUT / "result.json"), "report": base.bind(REPORT), "sources": sources, "counts": counts,
        "observed_errors": errors, "strict_positive_memberships_recomputed_exactly": True,
        "new_training_updates": 0, "new_encoder_or_RoMa_forwards": 0, "HYP_GO_claimed": False}
    base.save_json(OUT / "validation.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(); validate() if args.validate else run()
