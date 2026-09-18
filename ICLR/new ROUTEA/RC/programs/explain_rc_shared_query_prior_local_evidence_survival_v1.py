#!/usr/bin/env python3
"""Track fixed complete local-evidence sets; no forwards, fits, or proposals.

All 127 rivals remain present. A and B are the original cached full-reference
MaxSim vectors without and with wr. C stages are per-token contributions to
the original or alpha-weighted score, including its candidate-specific mass.
Independent validation uses NumPy64 reductions, separately from the producer's
literal Torch64 reductions. Its numerical tolerances are engineering bounds,
not correctness, selection, or scientific thresholds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "rc_shared_query_prior_local_evidence_survival_v1"
OUT = ROOT / "results" / PREFIX
REPORT = ROOT / "reports/REPORT_RC_SHARED_QUERY_PRIOR_LOCAL_EVIDENCE_SURVIVAL_20260909.md"
SOURCE = ROOT / "results/rc_shared_query_target_prior_development_v1"
RESULT_SHA = "977ffe3dcd76deb62bb0dde880ee4e9c5694ec25ab0323bd8acfa9a64e05d2f0"
CASES = {27: (17, 51), 193: (18, 33), 200: (15, 24), 449: (66, 124)}
STAGES = ("A_CONTENT", "B_REFERENCE_VISIBILITY", "C_ORIGINAL_CONTRIBUTION", "C_PRIOR_CONTRIBUTION")
RTOL, ATOL, FEATURE_ATOL = 1e-11, 1e-13, 1e-10


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def safe(path):
    path = Path(path).resolve()
    need(path.is_relative_to(ROOT) and not any(x in str(path).lower() for x in ("d1_mi", "d1-mi", "grozi", "gisc_prerecall_universe")), "PROTECTED_OR_OUTSIDE_PATH")
    return path


def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(safe(path).read_text())


def bind(path):
    return {"path": str(safe(path).relative_to(ROOT)), "sha256": sha(path)}


def checked(item):
    path = ROOT / item["path"]
    need(sha(path) == item["sha256"], "SOURCE_SHA_DRIFT:" + item["path"])
    return path


def save_json(path, value):
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def load_inputs():
    import numpy as np
    need(sha(SOURCE / "result.json") == RESULT_SHA, "RESULT_PIN")
    result = read(SOURCE / "result.json")
    sources = {"result": bind(SOURCE / "result.json"), "program": bind(__file__)}
    authpath = ROOT / "registry/rc_shared_query_target_prior_development_authority_v1_20260909.json"
    need(sha(authpath) == result["authority_sha256"], "AUTHORITY_PIN")
    auth = read(authpath); sources["authority"] = bind(authpath)
    for key in ("cache_manifest", "cache_validation", "head_parameter_seal"):
        sources[key] = auth["sources"][key]; checked(sources[key])
    cv = read(checked(sources["cache_validation"]))
    need(cv["manifest_sha256"] == sources["cache_manifest"]["sha256"] and all(cv["checks"].values()), "CACHE_VALIDATION")
    cache = read(checked(sources["cache_manifest"]))
    need(sha(SOURCE / "parameters.json") == result["parameters_sha256"] and sha(SOURCE / "eval_prejoin_seal.json") == result["eval_prejoin_seal_sha256"], "PREJOIN_PARAMETERS")
    seal = read(SOURCE / "eval_prejoin_seal.json")
    need(sha(SOURCE / "eval_prejoin.json") == seal["eval_prejoin_sha256"], "PREJOIN_INDEX")
    for key in ("parameters", "eval_prejoin", "eval_prejoin_seal", "independent_validation"):
        sources[key] = bind(SOURCE / (key + ".json"))
    v = read(SOURCE / "independent_validation.json")
    need(v["result_sha256"] == RESULT_SHA and all(v["checks"].values()), "SOURCE_VALIDATION")
    head = read(checked(sources["head_parameter_seal"]))
    pre = {row["execution_ordinal"]: row for row in read(SOURCE / "eval_prejoin.json")}
    real = {row["execution_ordinal"]: row for row in result["actions"]["REAL"]}
    episodes = {}
    for entry in cache["records"]:
        ex = entry["execution_ordinal"]
        if entry["kind"] != "FULL" or ex not in CASES:
            continue
        meta = read(checked(entry)); pred = pre[ex]
        need(meta["axis"] == pred["candidate_physical_rows"] and meta["query_id"] == pred["query_id"], "COMMON_QUERY_AXIS")
        target, wrong = CASES[ex]
        need(real[ex]["target_position"] == target and wrong != target, "FIXED_CASE_TARGET")
        need(wrong == (real[ex]["base_winner"] if ex in (27, 200) else real[ex]["proposed_challenger"]), "FIXED_CRITICAL_WRONG")
        with np.load(checked(meta["arrays"]), allow_pickle=False) as src:
            arrays = {k: src[k].copy() for k in src.files}
        with np.load(checked(pred["arrays"]), allow_pickle=False) as src:
            saved = {k: src[k].copy() for k in src.files}
        need(arrays["a"].shape == arrays["b"].shape == arrays["wq"].shape and arrays["a"].shape[0] == 128 and np.array_equal(arrays["candidate_positions"], np.arange(128)), "FULL128_QUERY_AXES")
        episodes[ex] = {"meta": meta, "input_arrays": arrays, "saved": saved,
                        "sources": {"metadata": entry, "cache_arrays": meta["arrays"], "frozen_outputs": pred["arrays"]}}
    need(set(episodes) == set(CASES), "CASE_POPULATION")
    return sources, head, episodes


def old_feature(raw, scalars, challenger, winner):
    mean = sum(map(float, raw)) / len(raw)
    std = (sum((float(x) - mean) ** 2 for x in raw) / len(raw)) ** .5
    c, w = list(map(float, scalars[challenger])), list(map(float, scalars[winner]))
    sym = lambda a, b: (a - b) / (abs(a) + abs(b) + 1e-12)
    return [(raw[challenger] - raw[winner]) / max(std, 1e-12), sym(c[0], w[0]), sym(c[1], w[1]),
            sym(c[0] / max(c[1], 1e-12), w[0] / max(w[1], 1e-12)),
            sym(c[0] - c[2], w[0] - w[2]), sym(c[0] - c[3], w[0] - w[3])]


def torch_stages(episode, head):
    import numpy as np
    import torch
    x = {k: torch.from_numpy(v) for k, v in episode["input_arrays"].items()}
    alpha = torch.from_numpy(episode["saved"]["REAL__alpha"])
    stages = {"A_CONTENT": x["a"], "B_REFERENCE_VISIBILITY": x["b"]}
    weight = torch.tensor([float.fromhex(v) for v in head["weight_binary64"]], dtype=torch.float64)
    bias = float.fromhex(head["bias_binary64"])
    winner = episode["meta"]["winner"]
    challengers = [p for p in range(128) if p != winner]
    raw = x["raw_scores"].tolist()
    for mode, field in (("UNIFORM", "C_ORIGINAL_CONTRIBUTION"), ("REAL", "C_PRIOR_CONTRIBUTION")):
        scalars, contributions = [], []
        prior = torch.ones_like(alpha) if mode == "UNIFORM" else alpha
        for g in range(128):
            effective = prior * x["wq"][g]
            shifted = effective.roll(int(x["qshift"][g]))
            mass = torch.sqrt(effective.mean() * x["wr_mean"][g])
            qm = torch.sqrt(shifted.mean() * x["wr_mean"][g])
            rm = torch.sqrt(effective.mean() * x["wr_rolled_mean"][g])
            score = mass * (effective * x["b"][g]).sum() / effective.sum().clamp_min(1e-12)
            qs = qm * (shifted * x["b"][g]).sum() / shifted.sum().clamp_min(1e-12)
            rs = rm * (effective * x["br"][g]).sum() / effective.sum().clamp_min(1e-12)
            scalars.append(torch.stack((score, mass, qs, rs)))
            contributions.append(mass * (effective * x["b"][g]) / effective.sum().clamp_min(1e-12))
        scalars = torch.stack(scalars)
        need(scalars.numpy().tobytes() == episode["saved"][mode + "__scalars"].tobytes(), "OLD_NEW_SCALARS_BIT_DRIFT")
        features = torch.tensor([old_feature(raw, scalars.tolist(), c, winner) for c in challengers], dtype=torch.float64)
        need(features.numpy().tobytes() == episode["saved"][mode + "__features"].tobytes(), "OLD_NEW_FEATURES_BIT_DRIFT")
        need((features @ weight + bias).numpy().tobytes() == episode["saved"][mode + "__logits"].tobytes(), "OLD_NEW_LOGITS_BIT_DRIFT")
        stages[field] = torch.stack(contributions)
        need(np.allclose(stages[field].sum(dim=1).numpy(), scalars[:, 0].numpy(), rtol=RTOL, atol=ATOL), "CONTRIBUTION_SUM")
    return {k: v.numpy() for k, v in stages.items()}


def weight_stats(alpha, visibility, indices):
    import numpy as np
    alpha = np.asarray(alpha, dtype=np.float64)
    selected = alpha[indices]
    effective = alpha * visibility
    total = float(alpha.sum()); part = float(selected.sum())
    ess = lambda a: float(a.sum() ** 2 / np.square(a).sum()) if np.square(a).sum() > 0 else 0.
    original_mass, effective_mass = float(visibility.sum()), float(effective.sum())
    return {"full_alpha_sum": total, "selected_alpha_sum": part,
            "selected_alpha_mean": float(selected.mean()) if len(indices) else 0.,
            "selected_alpha_fraction": part / total if total else 0.,
            "full_alpha_ESS": ess(alpha), "selected_alpha_ESS": ess(selected),
            "selected_alpha_zero_count": int((selected == 0).sum()),
            "original_wq_sum": original_mass, "effective_alpha_wq_sum": effective_mass,
            "original_selected_wq_sum": float(visibility[indices].sum()),
            "effective_selected_alpha_wq_sum": float(effective[indices].sum()),
            "original_selected_wq_fraction": float(visibility[indices].sum()) / original_mass if original_mass else 0.,
            "effective_selected_alpha_wq_fraction": float(effective[indices].sum()) / effective_mass if effective_mass else 0.,
            "candidate_visibility_retention_rho": effective_mass / original_mass if original_mass else 0.}


def make_case(ex, episode, stages):
    import numpy as np
    arrays = {key + "__values": value for key, value in stages.items()}
    alpha = episode["saved"]["REAL__alpha"]
    arrays["alpha"] = alpha
    meta = episode["meta"]
    record = {"execution_ordinal": ex, "query_id": meta["query_id"], "query_grid_shape": meta["q_grid_shape"],
              "candidate_physical_rows": meta["axis"], "sources": episode["sources"], "subjects": {}}
    for subject, g in zip(("TARGET", "CRITICAL_WRONG"), CASES[ex]):
        rivals = np.array([h for h in range(128) if h != g], dtype=np.int64)
        arrays[subject + "__rival_positions"] = rivals
        a_margin = stages[STAGES[0]][g][None, :] - stages[STAGES[0]][rivals]
        fixed = np.flatnonzero(a_margin.min(axis=0) > 0)
        best = int(a_margin.min(axis=0).argmax())
        arrays[subject + "__fixed_A_positive_indices"] = fixed
        item = {"position": g, "physical_row": meta["axis"][g], "fixed_A_positive_indices": fixed.tolist(),
                "fixed_A_positive_count": len(fixed), "best_A_witness_index": best,
                "best_A_witness_margin": float(a_margin[:, best].min()),
                "alpha_and_visibility": weight_stats(alpha, episode["input_arrays"]["wq"][g], fixed), "stages": {}}
        for stage, values in stages.items():
            margins = values[g][None, :] - values[rivals]
            worst = margins.min(axis=0)
            arrays[subject + "__" + stage + "__all127_margins"] = margins
            arrays[subject + "__" + stage + "__worst_margin"] = worst
            arrays[subject + "__" + stage + "__positive_indices"] = np.flatnonzero(worst > 0)
            arrays[subject + "__" + stage + "__all128_values_at_fixed_A_best_witness"] = values[:, best]
            survivors = fixed[worst[fixed] > 0]
            item["stages"][stage] = {"all_axis_positive_count": int((worst > 0).sum()),
                "fixed_A_positive_survivor_count": len(survivors), "fixed_A_positive_survivor_indices": survivors.tolist(),
                "fixed_A_best_witness_margin": float(worst[best]),
                "candidate_value_sum_on_fixed_A_set": float(values[g, fixed].sum()),
                "candidate_value_sum_all_query": float(values[g].sum())}
        record["subjects"][subject] = item
    return record, arrays


def run():
    import numpy as np
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    need(not OUT.exists() and not REPORT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    sources, head, episodes = load_inputs()
    records = []
    for ex in sorted(episodes):
        stages = torch_stages(episodes[ex], head)
        record, arrays = make_case(ex, episodes[ex], stages)
        path = OUT / f"exec{ex:04d}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o444)
        record["arrays"] = bind(path); records.append(record)
    result = {"status": "RC_SHARED_QUERY_PRIOR_LOCAL_EVIDENCE_SURVIVAL_V1_COMPLETE", "sources": sources,
        "case_count": 4, "candidate_count": 128, "rivals_per_subject": 127, "records": records,
        "stage_definitions": {"A_CONTENT": "max_j cosine(q_p,r_gj), original unweighted full-reference cache",
            "B_REFERENCE_VISIBILITY": "max_j wr_gj*cosine(q_p,r_gj), original full-reference cache",
            "C_ORIGINAL_CONTRIBUTION": "M_g*wq_gp*b_gp/max(sum_p wq_gp,1e-12)",
            "C_PRIOR_CONTRIBUTION": "M'_g*alpha_p*wq_gp*b_gp/max(sum_p alpha_p*wq_gp,1e-12)"},
        "fixed_sets": "all query tokens with strict positive A margin against every one of the original 127 rivals; no top-K",
        "producer_exact_checks": {"old_new_scalar_values": 4096, "old_new_feature_values": 6096, "old_new_logits": 1016},
        "numpy_validation_tolerances": {"rtol": RTOL, "scalar_or_stage_atol": ATOL, "feature_atol": FEATURE_ATOL,
            "strict_positive_set_membership_must_match_exactly": True},
        "limits": ["A positive token is a contextual representation witness, not pixel-local semantic ownership.",
            "Wrong-reference exclusive tokens do not prove that reference object is physically present.",
            "B to C includes query visibility, candidate mass and normalization together.",
            "Contribution ranking is not the nonlinear six-feature action ranking.",
            "The frozen head and all experiment outputs are unchanged; no new model or accuracy estimate is produced."],
        "new_training_updates": 0, "new_encoder_or_RoMa_forwards": 0, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
    save_json(OUT / "result.json", result)
    lines = ["# 固定完整局部证据集合的存活追踪（2026-09-09）", "", "原 RAW C128、opened EVAL32 的四个已打开案例。只使用既有 ColNomic/RoMa 缓存和冻结 prior 输出；没有新训练、forward、top-K 或模型选择。", "",
        "所有 A 独占 token 都固定保留；以下数字依次表示该集合在 A、B、原 C 逐 token 贡献、新 C 逐 token 贡献中仍严格超过完整127个 rival 的数量。", "",
        "| Query | 对象 | A → B → 原 C → 新 C | α均值 | α有效数量 ESS | 原→新 query visibility 占比 |", "|---|---|---|---:|---:|---|"]
    for row in records:
        for subject, item in row["subjects"].items():
            nums = " → ".join(str(item["stages"][s]["fixed_A_positive_survivor_count"]) for s in STAGES)
            st = item["alpha_and_visibility"]
            lines.append(f"| {row['query_id']} | {subject} | {nums} | {st['selected_alpha_mean']:.6g} | {st['selected_alpha_ESS']:.3f} | {st['original_selected_wq_fraction']:.6g} → {st['effective_selected_alpha_wq_fraction']:.6g} |")
    lines.extend(["", "A→B仅改变原 reference visibility；B→C同时引入原 query visibility、mass和归一化，不能将这一步全归于某一个因素。新旧 C 比较固定同一原来源，仅改变已训练 α 的作用。逐 token 贡献求和复原 scorer，但局部贡献排序不是六 feature action 的候选排序。", "", "target 与 wrong 的 A 独占证据都只是上下文 token 表征中的可区分信号。wrong 独占不能证明图中真实存在该物体，token 坐标也不能直接当作像素感受野或 ownership。", "", f"完整127×Q margin、全部正 token 集合、固定最佳 A witness 的128候选值、α总量/ESS和来源SHA见 [result.json]({OUT / 'result.json'})。结果SHA：`{sha(OUT / 'result.json')}`。", ""])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x") as stream:
        stream.write("\n".join(lines)); stream.flush(); os.fsync(stream.fileno())
    REPORT.chmod(0o444)
    print(json.dumps({"status": result["status"], "result": bind(OUT / "result.json"), "report": bind(REPORT)}, sort_keys=True), flush=True)


def validate():
    """Separate NumPy64 implementation; no Torch producer reductions called."""
    import numpy as np
    sources, head, episodes = load_inputs()
    result = read(OUT / "result.json")
    need(result["sources"] == sources, "VALIDATION_SOURCE_DRIFT")
    errors = {"scalar_max_abs": 0., "feature_max_abs": 0., "stage_max_abs": 0., "margin_max_abs": 0.}
    checks = {"queries": 0, "subject_stages": 0, "complete127_margin_entries": 0, "old_new_scalar_values": 0, "old_new_feature_values": 0}
    for record in result["records"]:
        ex = record["execution_ordinal"]; episode = episodes[ex]
        x, predicted = episode["input_arrays"], episode["saved"]
        alpha = predicted["REAL__alpha"]
        stages = {STAGES[0]: x["a"], STAGES[1]: x["b"]}
        for mode, stage in (("UNIFORM", STAGES[2]), ("REAL", STAGES[3])):
            prior = np.ones_like(alpha) if mode == "UNIFORM" else alpha
            ew = x["wq"] * prior[None, :]
            shifted = np.stack([np.roll(ew[g], int(x["qshift"][g])) for g in range(128)])
            den = np.maximum(ew.sum(axis=1), 1e-12)
            mass = np.sqrt(ew.mean(axis=1) * x["wr_mean"])
            qm = np.sqrt(shifted.mean(axis=1) * x["wr_mean"])
            rm = np.sqrt(ew.mean(axis=1) * x["wr_rolled_mean"])
            scalars = np.stack((mass * (ew * x["b"]).sum(axis=1) / den, mass,
                qm * (shifted * x["b"]).sum(axis=1) / np.maximum(shifted.sum(axis=1), 1e-12),
                rm * (ew * x["br"]).sum(axis=1) / den), axis=1)
            diff = float(np.abs(scalars - predicted[mode + "__scalars"]).max())
            errors["scalar_max_abs"] = max(errors["scalar_max_abs"], diff)
            need(np.allclose(scalars, predicted[mode + "__scalars"], rtol=RTOL, atol=ATOL), "NUMPY_SCALAR_REDUCTION_DRIFT")
            # Independent vector form of all six features, from the NumPy-reduced scalars.
            winner = episode["meta"]["winner"]; cs = np.array([g for g in range(128) if g != winner])
            raw = x["raw_scores"]
            mean = sum(map(float, raw)) / 128
            std = (sum((float(v) - mean) ** 2 for v in raw) / 128) ** .5
            sym = lambda values: (values[cs] - values[winner]) / (np.abs(values[cs]) + abs(values[winner]) + 1e-12)
            features = np.stack(((raw[cs] - raw[winner]) / max(std, 1e-12), sym(scalars[:, 0]), sym(scalars[:, 1]),
                sym(scalars[:, 0] / np.maximum(scalars[:, 1], 1e-12)), sym(scalars[:, 0] - scalars[:, 2]), sym(scalars[:, 0] - scalars[:, 3])), axis=1)
            errors["feature_max_abs"] = max(errors["feature_max_abs"], float(np.abs(features - predicted[mode + "__features"]).max()))
            need(np.allclose(features, predicted[mode + "__features"], rtol=RTOL, atol=FEATURE_ATOL), "NUMPY_FEATURE_REDUCTION_DRIFT")
            stages[stage] = mass[:, None] * (ew * x["b"]) / den[:, None]
            checks["old_new_scalar_values"] += scalars.size; checks["old_new_feature_values"] += features.size
        with np.load(checked(record["arrays"]), allow_pickle=False) as saved:
            need(np.array_equal(saved["alpha"], alpha), "ALPHA_SOURCE_DRIFT")
            for subject, g in zip(("TARGET", "CRITICAL_WRONG"), CASES[ex]):
                rivals = np.array([h for h in range(128) if h != g])
                need(np.array_equal(saved[subject + "__rival_positions"], rivals), "ALL127_RIVAL_AXIS")
                am = x["a"][g] - x["a"][rivals].max(axis=0)
                fixed = np.flatnonzero(am > 0)
                item = record["subjects"][subject]
                need(fixed.tolist() == item["fixed_A_positive_indices"] and np.array_equal(saved[subject + "__fixed_A_positive_indices"], fixed), "COMPLETE_A_SET_DRIFT")
                stats = weight_stats(alpha, x["wq"][g], fixed)
                need(stats == item["alpha_and_visibility"], "ALPHA_MASS_ESS_DRIFT")
                best = int(am.argmax()); need(best == item["best_A_witness_index"], "FIXED_WITNESS_DRIFT")
                for stage, values in stages.items():
                    diff = float(np.abs(values - saved[stage + "__values"]).max())
                    errors["stage_max_abs"] = max(errors["stage_max_abs"], diff)
                    need(np.allclose(values, saved[stage + "__values"], rtol=RTOL, atol=ATOL), "NUMPY_STAGE_DRIFT")
                    margins = np.stack([values[g] - values[h] for h in rivals])
                    key = subject + "__" + stage
                    errors["margin_max_abs"] = max(errors["margin_max_abs"], float(np.abs(margins - saved[key + "__all127_margins"]).max()))
                    need(np.allclose(margins, saved[key + "__all127_margins"], rtol=RTOL, atol=ATOL), "NUMPY_MARGIN_DRIFT")
                    worst = margins.min(axis=0)
                    need(np.array_equal(np.flatnonzero(worst > 0), saved[key + "__positive_indices"]), "STRICT_POSITIVE_SET_DRIFT")
                    need(fixed[worst[fixed] > 0].tolist() == item["stages"][stage]["fixed_A_positive_survivor_indices"], "SAME_LOCATION_SURVIVAL_DRIFT")
                    need(np.allclose(values[:, best], saved[key + "__all128_values_at_fixed_A_best_witness"], rtol=RTOL, atol=ATOL), "ALL128_WITNESS_DRIFT")
                    checks["subject_stages"] += 1; checks["complete127_margin_entries"] += margins.size
        checks["queries"] += 1
    need(checks["queries"] == 4 and checks["subject_stages"] == 32 and checks["old_new_scalar_values"] == 4096 and checks["old_new_feature_values"] == 6096, "VALIDATION_POPULATION")
    receipt = {"status": "RC_SHARED_QUERY_PRIOR_LOCAL_EVIDENCE_SURVIVAL_V1_NUMPY64_VALIDATION_PASS",
        "result": bind(OUT / "result.json"), "report": bind(REPORT), "sources": sources, "checks": checks, "observed_errors": errors,
        "independent_numerical_path": "NumPy64 means/sums and vector features; no Torch producer reduction or scorer import",
        "strict_positive_memberships_recomputed_exactly": True, "new_training_updates": 0, "HYP_GO_claimed": False}
    save_json(OUT / "validation.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate() if args.validate else run()
