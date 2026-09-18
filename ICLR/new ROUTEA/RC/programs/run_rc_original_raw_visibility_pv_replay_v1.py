#!/usr/bin/env python3
"""Recompute the old RAW visibility score through the repaired P/V interface."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
INPUT = ROOT / "results/rc_original_raw_visibility_pv_inputs_v1"
OUT = ROOT / "results/rc_original_raw_visibility_pv_replay_v1"
AUTH = ROOT / "registry/rc_original_raw_visibility_pv_replay_authority_v1_20260909.json"
LAUNCH = ROOT / "slurm/rc_original_raw_visibility_pv_replay_v1_dev_cpuonly_59m.sbatch"
PLAN = ROOT / "plan/RC_ORIGINAL_RAW_VISIBILITY_PV_LOSSLESS_REPAIR_V1_20260909.md"
CORE = ROOT / "src/rc_aslo_xf/reference_visibility_pv_lossless_v1.py"
CORE_SHA = "f5fcfa1ce6fa6582a3628035e2ba54414185160fa36515fa27fd55d8aa81adcc"
GATE = ROOT / "src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py"
GATE_SHA = "96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7"
OLD = ROOT / "results/rc_frozen_roma_action_terms_v1/result.json"
OLD_SHA = "9af91982d06630da09e84120dce9cdd673d06f0c11cb019750e818dfa59300a6"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def bind(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(item):
    path = ROOT / item["path"]
    assert path.resolve().is_relative_to(ROOT) and sha(path) == item["sha256"], str(path)
    return path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    return bind(path)


def source_pins():
    assert sha(CORE) == CORE_SHA and sha(GATE) == GATE_SHA and sha(OLD) == OLD_SHA
    return {name: bind(path) for name, path in {
        "program": Path(__file__), "core": CORE, "gate": GATE,
        "input_manifest": INPUT / "manifest.json", "plan": PLAN, "launcher": LAUNCH,
        "old_action_replay": OLD,
    }.items()}


def freeze():
    assert not AUTH.exists() and not OUT.exists()
    inputs = read(INPUT / "manifest.json")
    assert inputs["status"] == "RC_ORIGINAL_RAW_VISIBILITY_PV_INPUTS_FULL64_EXACT_EXPORT_PASS"
    save(AUTH, {"status": "RC_ORIGINAL_RAW_VISIBILITY_PV_LOSSLESS_REPLAY_AUTHORIZED",
        "sources": source_pins(), "query_count": 64, "candidate_count": 128,
        "claim": "ORIGINAL_RAW_SYSTEM_INTERFACE_REPAIR_ONLY", "new_training": False,
        "new_HYP_or_scientific_GO": False, "formal392_authorized": False})
    print(json.dumps({"status": "FROZEN", "authority_sha256": sha(AUTH)}), flush=True)


def authority():
    value = read(AUTH)
    assert value["sources"] == source_pins()
    for item in value["sources"].values():
        checked(item)
    return value


def predict(axis, raw, candidates, gate):
    winner = max(range(128), key=lambda i: (raw[i], -axis[i]))
    challengers = [i for i in range(128) if i != winner]
    logits = [gate.logit(raw, candidates, c, winner) for c in challengers]
    best_logit, best = max(zip(logits, challengers), key=lambda x: (x[0], -axis[x[1]]))
    return {"base_winner": winner, "proposed_challenger": best,
        "switch_logit_binary64": best_logit.hex(), "decision": "SWITCH" if best_logit > 0 else "HOLD",
        "final_position": best if best_logit > 0 else winner,
        "challengers": challengers, "complete_logits_binary64": [z.hex() for z in logits]}


def run():
    import torch
    from rc_aslo_xf import reference_visibility_pv_lossless_v1 as pv
    from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as gate
    auth = authority()
    assert not OUT.exists()
    started = time.monotonic()
    manifest = read(checked(auth["sources"]["input_manifest"]))
    assert len(manifest["shards"]) == 8
    records = []
    for item in sorted(manifest["shards"], key=lambda x: x["shard"]):
        path = INPUT / item["path"]
        assert sha(path) == item["sha256"]
        assert sha(INPUT / item["receipt_path"]) == item["receipt_sha256"]
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        original_cache = {}
        for query in data["records"]:
            assert time.monotonic() - started < 3000, "INCOMPLETE_3000S_BUDGET"
            source = query["token_source"]
            if source["path"] not in original_cache:
                original_cache[source["path"]] = torch.load(checked(source), map_location="cpu", weights_only=False, mmap=True)
            tokens = original_cache[source["path"]]
            qrecord = tokens["records"][source["record_index"]]
            assert qrecord["query_id"] == query["query_id"]
            assert int(qrecord["execution_ordinal"]) == query["execution_ordinal"]
            assert list(qrecord["candidate_physical_rows"]) == query["candidate_physical_rows"]
            assert pv.tensor_sha256(qrecord["query_tokens"]) == query["query_tokens_sha256"]
            axis, raw = query["candidate_physical_rows"], query["candidate_raw_scores"]
            assert len(axis) == len(set(axis)) == len(raw) == len(query["candidates"]) == 128
            computed, rows = {}, []
            for i, candidate in enumerate(query["candidates"]):
                assert candidate["candidate_position"] == i and candidate["physical_row"] == axis[i]
                reference = tokens["references"][axis[i]]
                proposal = pv.make_visibility_proposal(
                    query_resource_key=query["query_resource_key"],
                    candidate_resource_key=candidate["candidate_resource_key"],
                    reference_resource_key=candidate["reference_resource_key"],
                    query_grid_shape=query["query_grid_shape"], reference_grid_shape=candidate["reference_grid_shape"],
                    query_visibility=candidate["query_visibility"], reference_visibility=candidate["reference_visibility"],
                    source_binding=candidate["source_binding"])
                assert float(proposal.visibility_mass).hex() == candidate["visibility_mass_binary64"]
                evidence = pv.verify_visibility(proposal, qrecord["query_tokens"], reference["tokens"],
                    expected_source_binding=candidate["source_binding"])
                fields = evidence.as_old_feature_record()
                old = candidate["old_scores"]
                for name, value in fields.items():
                    assert value.hex() == float(old[name]).hex(), (query["query_id"], i, name, value.hex(), float(old[name]).hex())
                assert float(raw[i]).hex() == float(old["raw_score"]).hex()
                computed[i] = {**fields, "raw_score": raw[i], "physical_row": axis[i], "candidate_position": i}
                rows.append({"candidate_position": i, "physical_row": axis[i],
                    "proposal_logical_sha256": proposal.logical_sha256,
                    "query_full_mean_binary64": float(proposal.query_full_mean).hex(),
                    "reference_full_mean_binary64": float(proposal.reference_full_mean).hex(),
                    "verified_scores_binary64": {k: v.hex() for k, v in fields.items()}})
            prediction = predict(axis, raw, computed, gate)
            old_candidates = {i: c["old_scores"] for i, c in enumerate(query["candidates"])}
            assert prediction == predict(axis, raw, old_candidates, gate), "ACTION_INTERFACE_NOT_LOSSLESS"
            result = {"query_id": query["query_id"], "role": query["role"], "execution_ordinal": query["execution_ordinal"],
                "candidate_physical_rows": axis, "candidates": rows, "prediction": prediction,
                "input_shard_sha256": item["sha256"], "token_source": source, "target_roles_read": 0}
            saved = save(OUT / "prejoin" / (str(query["execution_ordinal"]) + ".json"), result)
            records.append({**saved, "query_id": query["query_id"], "role": query["role"], "execution_ordinal": query["execution_ordinal"]})
            print(json.dumps({"event": "RAW_VISIBILITY_PV_QUERY_EXACT", "queries": len(records),
                "elapsed_seconds": time.monotonic() - started}), flush=True)
        del original_cache, data
    assert len(records) == len({r["query_id"] for r in records}) == 64
    assert {role: sum(r["role"] == role for r in records) for role in ("TRAIN", "EVAL")} == {"TRAIN": 32, "EVAL": 32}
    save(OUT / "prejoin_seal.json", {"status": "RC_ORIGINAL_RAW_VISIBILITY_PV_FULL64_EXACT_PREJOIN",
        "authority_sha256": sha(AUTH), "records": records, "new_V_raw_token_pair_recomputations": 8192,
        "scalar_bit_exact_comparisons": 32768, "action_logit_bit_exact_comparisons": 8128,
        "target_roles_read_before_full64_seal": 0, "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": "FULL64_REPLAY_COMPLETE", "new_HYP_or_scientific_GO": False}), flush=True)


def validate():
    from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as gate
    auth = authority()
    seal = read(OUT / "prejoin_seal.json")
    assert seal["authority_sha256"] == sha(AUTH) and len(seal["records"]) == 64
    prior = read(checked(auth["sources"]["old_action_replay"]))
    detail = ROOT / "results/rc_frozen_roma_action_terms_v1" / prior["complete_terms"]["path"]
    assert sha(detail) == prior["complete_terms"]["sha256"]
    originals = {r["query_id"]: r for r in read(detail)["records"]}
    assert len(originals) == 64
    original_scores = {}
    prefix = "results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1/shard"
    for source in prior["sources"]:
        if source["path"].startswith(prefix) and source["path"].endswith("/result.json"):
            for row in read(checked(source))["rows"]:
                assert row["query_id"] not in original_scores
                original_scores[row["query_id"]] = row
    assert len(original_scores) == 64
    summaries = {}
    for role in ("TRAIN", "EVAL"):
        actions = []
        for record in seal["records"]:
            if record["role"] != role:
                continue
            row = read(checked(record))
            old = originals[row["query_id"]]
            assert row["candidate_physical_rows"] == old["candidate_physical_rows"]
            source_rows = original_scores[row["query_id"]]["candidates"]
            assert len(row["candidates"]) == len(source_rows) == 128
            reconstructed = {}
            for index, (candidate, source) in enumerate(zip(row["candidates"], source_rows, strict=True)):
                assert candidate["candidate_position"] == source["candidate_position"] == index
                assert candidate["physical_row"] == source["physical_row"] == row["candidate_physical_rows"][index]
                fields = candidate["verified_scores_binary64"]
                assert set(fields) == {"real_score", "visibility_mass", "query_control_score", "reference_control_score"}
                assert all(value == float(source[key]).hex() for key, value in fields.items())
                reconstructed[index] = {key: float.fromhex(value) for key, value in fields.items()}
            predicted, action = row["prediction"], old["full_action"]
            raw = [float(c["raw_score"]) for c in source_rows]
            assert predicted == predict(row["candidate_physical_rows"], raw, reconstructed, gate)
            for field in ("base_winner", "proposed_challenger", "final_position", "decision"):
                assert predicted[field] == action[field]
            assert predicted["switch_logit_binary64"] == float(action["switch_logit"]).hex()
            assert predicted["complete_logits_binary64"] == [c["logit_binary64"] for c in old["complete_127_challengers"]]
            actions.append(action)
        assert len(actions) == 32
        summaries[role] = {"base_top1": sum(a["base_correct"] for a in actions),
            "final_top1": sum(a["final_correct"] for a in actions),
            "rescue": sum(not a["base_correct"] and a["final_correct"] for a in actions),
            "break": sum(a["base_correct"] and not a["final_correct"] for a in actions)}
    save(OUT / "result.json", {"status": "RC_ORIGINAL_RAW_VISIBILITY_PV_LOSSLESS_REPLAY_PASS",
        "authority_sha256": sha(AUTH), "prejoin_seal_sha256": sha(OUT / "prejoin_seal.json"),
        "summaries": summaries, "checks": {"all64_original_RAW_C128_axes": True,
            "all8192_pairs_recomputed_from_original_RAW_tokens_through_new_P_V": True,
            "all32768_original_score_scalars_bit_exact": True, "all8128_original_action_logits_bit_exact": True,
            "all64_original_predictions_unchanged": True},
        "new_training_updates": 0, "encoder_or_RoMa_forwards": 0,
        "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None,
        "limit": "Lossless engineering repair of original soft visibility mechanism; not connected HYP, spatial causality, or a new retrieval gain."})
    print(json.dumps({"status": "RC_ORIGINAL_RAW_VISIBILITY_PV_LOSSLESS_REPLAY_PASS", "summaries": summaries}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "run", "validate", "e0"), required=True)
    args = parser.parse_args()
    if args.phase == "e0":
        from rc_aslo_xf.reference_visibility_pv_lossless_v1 import verify_visibility
        assert callable(verify_visibility) and sha(CORE) == CORE_SHA and sha(GATE) == GATE_SHA
        print(json.dumps({"status": "LOSSLESS_PV_LAUNCHER_IMPORT_E0_PASS", "natural_data_reads": 0}))
        return
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    {"freeze": freeze, "run": run, "validate": validate}[args.phase]()


if __name__ == "__main__":
    main()
