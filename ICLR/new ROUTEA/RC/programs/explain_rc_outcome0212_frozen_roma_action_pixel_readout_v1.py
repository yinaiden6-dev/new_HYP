#!/usr/bin/env python3
"""Read existing pixel-intervention query tokens through original frozen C.

do(query encoder output), conditional on ORIGINAL RoMa visibility, RAW C128
scores, reference tokens and NATIVE7/C action parameters. This does not rerun
the end-to-end image pipeline. No encoder/RoMa forward or fitting occurs.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_outcome0212_frozen_roma_action_pixel_readout_v1"
REPORT = ROOT / "reports/REPORT_RC_OUTCOME0212_FROZEN_ROMA_ACTION_PIXEL_READOUT_20260909.md"
PIXEL = ROOT / "results/rc_outcome0212_pixel_source_diagnostic_v1/result.json"
PIXEL_SHA = "81bbaaafb37e898524ac42b959e07bc322baf00ce0aaad4ed46e384fdd949388"
PARENT = ROOT / "results/rc_shared_query_target_prior_development_v1/result.json"
PARENT_SHA = "977ffe3dcd76deb62bb0dde880ee4e9c5694ec25ab0323bd8acfa9a64e05d2f0"
META = ROOT / "results/rc_shared_query_target_prior_cache_v1/episodes/FULL_0193/metadata.json"
META_SHA = "16aab84e1b4d1ca34247104e95b6866dd383bfde6087360de644e8e0c75d724e"
OLD_SCORE = ROOT / "programs/run_romav2_colnomic_visibility_xf_six_case_v1.py"
SCORE_SHA = "fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c"
MODES = ("ORIGINAL", "KEEP_TARGET", "ERASE_TARGET", "ALL_GRAY")
FIELDS = ("real_score", "visibility_mass", "query_control_score", "reference_control_score")


def need(value, message):
    if not bool(value): raise RuntimeError(message)


def safe(path):
    path = Path(path).resolve()
    need(path.is_relative_to(ROOT) and not any(x in str(path).lower() for x in ("d1_mi", "d1-mi", "grozi", "gisc_prerecall_universe")), "PROTECTED_PATH")
    return path


def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(safe(path).read_text())
def bind(path): return {"path": str(safe(path).relative_to(ROOT)), "sha256": sha(path)}
def checked(item):
    path = safe(ROOT / item["path"])
    need(sha(path) == item["sha256"], "SOURCE_HASH_DRIFT")
    return path


def token_sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str(t.dtype).encode() + json.dumps(list(t.shape), separators=(",", ":")).encode() + t.numpy().tobytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def load_inputs():
    import torch
    import numpy as np
    from torch.nn import functional as F
    for path, expected in ((PIXEL, PIXEL_SHA), (PARENT, PARENT_SHA), (META, META_SHA), (OLD_SCORE, SCORE_SHA)):
        need(sha(path) == expected, "PIN_DRIFT")
    pix, parent, meta = read(PIXEL), read(PARENT), read(META)
    sources = {"program": bind(__file__), "pixel_result": bind(PIXEL), "parent_result": bind(PARENT), "metadata": bind(META), "original_score_source": bind(OLD_SCORE)}
    validation = PIXEL.with_name("independent_validation.json")
    v = read(validation)
    need(v["result_sha256"] == PIXEL_SHA and all(v["checks"].values()) and pix["original_token_SHA_and_a_bit_exact"], "PIXEL_NOT_QUALIFIED")
    sources["pixel_validation"] = bind(validation)
    for key in ("maps", "tokens"):
        sources[key] = meta["source"][key]; checked(sources[key])
    maps = torch.load(checked(sources["maps"]), map_location="cpu", mmap=True, weights_only=True)
    rows = [row for row in maps["records"] if row["execution_ordinal"] == 193]
    need(len(rows) == 1, "MAP_QUERY_COUNT"); row = rows[0]
    archive = torch.load(checked(sources["tokens"]), map_location="cpu", mmap=True, weights_only=True)
    query = archive["records"][sources["tokens"]["record_index"]]
    need(query["query_id"] == row["query_id"] == meta["query_id"] == "OUTCOME-0212"
         and query["candidate_physical_rows"] == row["candidate_physical_rows"] == meta["axis"] == pix["candidate_physical_rows"], "QUERY_AND_FULL128_AXIS")
    need(token_sha(query["query_tokens"]) == meta["q_tokens_sha256"], "ORIGINAL_QUERY_TOKEN_SHA")
    candidates = []
    for p, c in enumerate(row["candidates"]):
        rt = archive["references"][c["physical_row"]]["tokens"]
        need(p == c["candidate_position"] and c["physical_row"] == meta["axis"][p] and token_sha(rt) == c["reference_tokens_sha256"], "REFERENCE_BINDING")
        wq, wr = c["query_visibility"], c["reference_visibility"]
        need(wq.dtype == wr.dtype == torch.float64 and wq.shape == (720,) and wr.ndim == 1, "ORIGINAL_VISIBILITY_AXES")
        need(hashlib.sha256(wq.numpy().tobytes()).hexdigest() == c["query_map_sha256"] and hashlib.sha256(wr.numpy().tobytes()).hexdigest() == c["reference_map_sha256"], "ORIGINAL_VISIBILITY_SHA")
        candidates.append({"reference": rt, "wq": wq, "wr": wr, "qshift": 360, "rshift": max(1, len(wr) // 2)})
    need(len(candidates) == 128, "C128_COUNT")
    old = next(a for a in parent["actions"]["UNIFORM"] if a["execution_ordinal"] == 193)
    need(old["final_correct"] and old["decision"] == "SWITCH" and old["target_physical_row"] == 1024, "ORIGINAL_RESCUE")
    parameters = read(PARENT.with_name("parameters.json"))
    need(sha(PARENT.with_name("parameters.json")) == parent["parameters_sha256"], "PARENT_PARAMETERS")
    head = read(checked(parameters["head_parameter_seal"])); sources["head"] = parameters["head_parameter_seal"]
    weight = torch.tensor([float.fromhex(v) for v in head["weight_binary64"]], dtype=torch.float64); bias = float.fromhex(head["bias_binary64"])
    need(head["parameter_sha256"] == parameters["head_parameter_sha256"] and not head["trainable_in_query_prior_branch"], "FIXED_HEAD")
    sys.path.insert(0, str(ROOT / "src"))
    from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as frozen
    need(sha(Path(frozen.__file__)) == "96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7", "ORIGINAL_FEATURE_SOURCE")
    sources["original_feature_source"] = bind(frozen.__file__)
    node = next(n for n in ast.parse(OLD_SCORE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "score")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(OLD_SCORE), "exec"), namespace)
    query_modes = {}
    for entry in pix["entries"]:
        with np.load(checked(entry["arrays"]), allow_pickle=False) as arrays: qt = torch.from_numpy(arrays["q_tokens"].copy())
        need(qt.shape == (720, 128) and qt.dtype == torch.float16 and token_sha(qt) == entry["query_tokens_sha256"], "PIXEL_QUERY_TOKEN_SHA")
        query_modes[entry["mode"]] = {"tokens": qt, "source": entry["arrays"]}
    need(tuple(query_modes) == MODES and torch.equal(query_modes["ORIGINAL"]["tokens"], query["query_tokens"]), "ORIGINAL_TOKEN_REGRESSION")
    with np.load(checked(meta["arrays"]), allow_pickle=False) as arrays: cached = {k: arrays[k].copy() for k in arrays.files}
    need(cached["raw_scores"].tolist() == row["candidate_raw_scores"], "FROZEN_RAW_SCORE_BINDING")
    return sources, row, candidates, query_modes, old, weight, bias, frozen, namespace["score"], cached


def action(raw, axis, logits, original):
    winner = original["base_winner"]; target = original["target_position"]
    challengers = [i for i in range(128) if i != winner]
    ranking = sorted(range(128), key=lambda i: (-raw[i], axis[i]))
    need(ranking[0] == winner, "FROZEN_RAW_WINNER")
    ix = max(range(127), key=lambda i: (float(logits[i]), -axis[challengers[i]]))
    best, z = challengers[ix], float(logits[ix]); decision = "SWITCH" if z > 0 else "HOLD"
    final = best if decision == "SWITCH" else winner; final_order = list(ranking)
    if decision == "SWITCH": final_order.remove(best); final_order.insert(0, best)
    return {**{k: original[k] for k in ("execution_ordinal", "query_id", "track", "heldout_fold")},
        "base_winner": winner, "base_winner_physical_row": axis[winner], "target_position": target, "target_physical_row": axis[target],
        "base_target_rank": ranking.index(target) + 1, "final_target_rank": final_order.index(target) + 1,
        "proposed_challenger": best, "proposed_challenger_physical_row": axis[best], "proposed_challenger_base_rank": ranking.index(best) + 1,
        "switch_logit": z, "decision": decision, "final_position": final, "final_physical_row": axis[final],
        "base_correct": winner == target, "final_correct": final == target, "wrong_to_wrong": winner != target and final != target and final != winner}


def calculate(validator=False):
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    sources, row, candidates, modes, old, weight, bias, frozen, score, cached = load_inputs()
    axis, raw = row["candidate_physical_rows"], row["candidate_raw_scores"]
    challengers = [p for p in range(128) if p != old["base_winner"]]
    records = []; original_mass = None
    for mode in MODES:
        qt = modes[mode]["tokens"]; scalars = []; locals_ = []; rlocals = []
        for c in candidates:
            wq, wr = c["wq"], c["wr"]
            if validator:
                sim = F.normalize(qt.double(), dim=1) @ F.normalize(c["reference"].double(), dim=1).T
                local = (sim * wr[None]).max(1).values
                rolled = wr.roll(c["rshift"]); br = (sim * rolled[None]).max(1).values
                mass = torch.sqrt(wq.mean() * wr.mean())
                s = mass * (wq * local).sum() / wq.sum().clamp_min(1e-12)
                rs = torch.sqrt(wq.mean() * rolled.mean()) * (wq * br).sum() / wq.sum().clamp_min(1e-12)
            else:
                s, mass, local = score(qt, c["reference"], wq, wr)
                rs, _, br = score(qt, c["reference"], wq, wr.roll(c["rshift"]))
            shifted = wq.roll(c["qshift"])
            qs = torch.sqrt(shifted.mean() * wr.mean()) * (shifted * local).sum() / shifted.sum().clamp_min(1e-12)
            scalars.append(torch.stack((s, mass, qs, rs))); locals_.append(local); rlocals.append(br)
        scalars, b, br = torch.stack(scalars), torch.stack(locals_), torch.stack(rlocals)
        plain = {p: {name: float(scalars[p, i]) for i, name in enumerate(FIELDS)} for p in range(128)}
        features = torch.stack([frozen.candidate_feature(raw, plain, c, old["base_winner"]) for c in challengers])
        logits = features @ weight + bias
        prediction = action(raw, axis, logits, old)
        if mode == "ORIGINAL":
            need(b.numpy().tobytes() == cached["b"].tobytes() and br.numpy().tobytes() == cached["br"].tobytes(), "ORIGINAL_LOCAL_CACHE_BIT_REGRESSION")
            for p, entry in enumerate(row["candidates"]):
                need(all(float(scalars[p, i]).hex() == float(entry["old_scores"][name]).hex() for i, name in enumerate(FIELDS)), "ORIGINAL_FOUR_SCALAR_BITS")
            need(prediction == old, "ORIGINAL_NATIVE_C_ACTION_BIT_REGRESSION")
            original_mass = scalars[:, 1].clone()
        need(torch.equal(scalars[:, 1], original_mass), "FROZEN_VISIBILITY_MASS_CHANGED")
        arrays = {"b": b.numpy(), "br": br.numpy(), "scalars": scalars.numpy(), "features": features.numpy(), "logits": logits.numpy()}
        path = OUT / (mode + ".npz")
        if validator:
            with np.load(path, allow_pickle=False) as saved:
                need(set(saved.files) == set(arrays), "SAVED_ARRAY_KEYS")
                for k, value in arrays.items(): need(value.tobytes() == saved[k].tobytes(), "INDEPENDENT_READOUT_BYTE_REPLAY:" + k)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
            path.chmod(0o444)
        target_ix = challengers.index(old["target_position"])
        records.append({"mode": mode, "query_token_source": modes[mode]["source"], "arrays": bind(path), "action": prediction,
            "target_vs_original_RAW_winner_logit": float(logits[target_ix]), "target_candidate_scalars": plain[old["target_position"]],
            "original_RAW_winner_scalars": plain[old["base_winner"]], "target_action_feature_contributions": (features[target_ix] * weight).tolist()})
        print(json.dumps({"event": "FROZEN_ORIGINAL_P_RAW_ACTION_READOUT", "mode": mode, "validate": validator, "action": prediction["decision"],
                          "final_physical_row": prediction["final_physical_row"], "switch_logit": prediction["switch_logit"]}), flush=True)
    result = {"status": "RC_OUTCOME0212_FROZEN_ROMA_ACTION_PIXEL_READOUT_V1_COMPLETE", "sources": sources,
        "query_id": "OUTCOME-0212", "execution_ordinal": 193, "candidate_physical_rows": axis, "raw_scores": raw,
        "head": {"weight": weight.tolist(), "bias": bias, "updated": False}, "records": records,
        "original_b_br_four_scalars_and_native_action_bit_exact": True, "all_conditions_visibility_mass_bit_identical": True,
        "intervention": "do(query encoder output) conditional on ORIGINAL RoMa wq/wr, reference tokens, RAW scores/C128 and NATIVE7/C action",
        "frozen_P_definition": "Original RoMa visibility maps; not a newly certified connected proposal",
        "limits": ["This is not a rerun of the complete altered-image pipeline: RoMa and RAW retain information from the original image.",
            "The original query was already correctly rescued by Native C28; altered-image action behavior is not an accuracy improvement.",
            "No claim of deployable GT use, spatial ownership, or HYP GO."],
        "additional_encoder_forward_count": 0, "additional_RoMa_forward_count": 0, "training_updates": 0,
        "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if not args.validate: need(not OUT.exists() and not REPORT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    result = calculate(args.validate)
    if args.validate:
        need(read(OUT / "result.json") == result, "RESULT_REPLAY_DRIFT")
        receipt = {"status": "RC_OUTCOME0212_FROZEN_ROMA_ACTION_PIXEL_READOUT_V1_INDEPENDENT_REPLAY_PASS",
            "result": bind(OUT / "result.json"), "full_candidate_conditions": 512, "four_scalar_values": 2048, "action_logit_values": 508,
            "independent_path": "Direct normalized FP64 matrix and original fixed-weight reductions, versus source-extracted old scorer",
            "additional_encoder_forward_count": 0, "additional_RoMa_forward_count": 0, "training_updates": 0}
        save_json(OUT / "validation.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    else:
        save_json(OUT / "result.json", result)
        lines = ["# OUTCOME-0212：像素条件token接回原固定RoMa/RAW/action", "", "只替换已经生成的query encoder输出。原128 references、RoMa wq/wr、RAW C128分数及Native7/C head全部冻结。没有新encoder、RoMa或训练。", "",
            "| token输入条件 | 动作 | 最终reference | 最大switch logit | target对原RAW winner的logit |", "|---|---|---:|---:|---:|"]
        for rec in result["records"]:
            a = rec["action"]
            lines.append(f"| {rec['mode']} | {a['decision']} | {a['final_physical_row']} | {a['switch_logit']:+.6f} | {rec['target_vs_original_RAW_winner_logit']:+.6f} |")
        lines += ["", "ORIGINAL的b/br、四scalar和原完整action逐bit回归；四条件mass完全相同，因为RoMa visibility被冻结。", "",
            "这是 do(query encoder output) 在原P/RAW/action条件下的有限干预，不是重跑完整图像pipeline：冻结RoMa与RAW仍携带原图信息。该例原NativeC28本来已成功救回，因此不能将干预后的结果称为识别提升或HYP GO。", "",
            f"来源SHA、四条件完整b/br、512候选四scalar、全部508 action logits：[result.json]({OUT / 'result.json'})。SHA：`{sha(OUT / 'result.json')}`。", ""]
        with REPORT.open("x") as stream: stream.write("\n".join(lines)); stream.flush(); os.fsync(stream.fileno())
        REPORT.chmod(0o444)
        print(json.dumps({"status": result["status"], "result": bind(OUT / "result.json"), "report": bind(REPORT)}, sort_keys=True), flush=True)
