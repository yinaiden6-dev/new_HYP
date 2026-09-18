#!/usr/bin/env python3
"""Complete fixed SAM component-bank capacity accounting, not a selector."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_cached_sam3_region_bank_capacity_v1"
WORKER = ROOT / "programs/run_rc_cached_sam3_region_closure_v1.py"
PARENT = ROOT / "results/rc_cached_sam3_region_closure_v1"
PINS = {
    "worker": (WORKER, "d282fa5be4926c6c1e2ea28f24b5790ffefedadaca0cbbfee10fdbcc8d73273b"),
    "authority": (ROOT / "registry/rc_cached_sam3_region_closure_authority_v1_20260909.json", "f64aee43f5d24fa5256945a349ed73d4414aa6dd218526e72c6b634214263fa1"),
    "parent_result": (PARENT / "result.json", "bf7970d5df75809f55b095a6d9d28d70ca821b195dfdf1f30cd0a3c5abb5a1ad"),
    "parent_validation": (PARENT / "independent_validation.json", "04c2bff0a19eb4f912dd0c4abcd1a1ee0e5678e000dbc97af1baff25a50abcfd"),
    "parent_prejoin": (PARENT / "prejoin.json", "0c9bfca90d650e43d54bfc9b66cb88040e6dd49cf90f2bde9a83189336f9f607"),
}


def load_source():
    import hashlib
    if hashlib.sha256(WORKER.read_bytes()).hexdigest() != PINS["worker"][1]: raise RuntimeError("SOURCE_WORKER_PIN")
    spec = importlib.util.spec_from_file_location("fixed_sam_bank_source", WORKER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.BARRIER = module.LabelBarrier(); module.BARRIER.paths.add(str((PARENT / "result.json").resolve()))
    sys.addaudithook(module.BARRIER.hook)
    return module


def inputs(source):
    bindings = {}
    for name, (path, expected) in PINS.items():
        source.need(source.sha(path) == expected, "PIN_DRIFT:" + name); bindings[name] = source.bind(path)
    source.authority()
    validation = source.read(PARENT / "independent_validation.json")
    source.need(validation["status"] == "RC_CACHED_SAM3_REGION_CLOSURE_V1_INDEPENDENT_ARTIFACT_VALIDATION_PASS"
        and validation["result_sha256"] == PINS["parent_result"][1] and all(x is True for x in validation["checks"].values()), "PARENT_NOT_VALIDATED")
    prior, parameters = source.load_prior()
    manifest = source.load_qualified_manifest()
    parent_prejoin = {row["execution_ordinal"]: row for row in source.read(PARENT / "prejoin.json")}
    bindings.update(program=source.bind(Path(__file__)), manifest=source.bind(source.INPUTS), prior_parameters=source.bind(source.PARAMETERS))
    return bindings, prior, parameters, manifest, parent_prejoin


def component_bank(source, prior, parameters, manifest, parent_prejoin):
    import numpy as np
    import torch
    records = []; total_masks = 0; total_components = 0
    for entry in manifest["records"]:
        meta = source.read(source.checked(entry["local_cache_metadata"]))
        source.need(meta["query_id"] == entry["query_id"] and meta["axis"] == entry["candidate_physical_rows"]
                    and meta["q_grid_shape"] == entry["native_grid_shape"], "CACHE_AXIS")
        with np.load(source.checked(entry["local_cache_arrays"]), allow_pickle=False) as values:
            tokens = torch.from_numpy(values["q_tokens"].copy()); a = torch.from_numpy(values["a"].copy())
        source.need(source.tensor_sha(tokens) == entry["query_tokens_sha256"] and a.dtype == torch.float64
                    and a.shape == (128, tokens.shape[0]), "RAW_TOKENS_OR_A_SHAPE")
        with torch.no_grad(): alpha = prior(tokens, entry["native_grid_shape"])
        seed = int(torch.argmax(alpha)); bank = source.prepare_masks(entry, entry["native_grid_shape"])
        old = parent_prejoin[entry["execution_ordinal"]]
        source.need(old["alpha_sha256"] == source.tensor_sha(alpha)
            and old["arms"]["QUERY_REGION"]["P"]["shared_region"]["seed"] == seed, "ORIGINAL_ALPHA_AND_SEED_REPLAY")
        all_union = torch.zeros(len(alpha), dtype=torch.bool); legal_union = torch.zeros_like(all_union)
        components = []
        for mask in bank:
            all_union |= mask["component_by_token"].ge(0)
            for component_index, positions in enumerate(mask["components"]):
                if len(positions) < 4: continue
                legal_union[positions] = True
                scores = [float(a[g, positions].mean()) for g in range(128)]
                region_id = f"p{mask['prompt_index']}_m{mask['mask_index']}_c{component_index}"
                component = {"region_id": region_id, "prompt_index": mask["prompt_index"], "mask_index": mask["mask_index"],
                    "component_index": component_index, "query_token_indices": positions, "area_tokens": len(positions),
                    "SAM_confidence_binary64": source.hx(mask["confidence"]), "alpha_mass_binary64": source.hx(alpha[positions].sum()),
                    "contains_original_argmax_alpha": seed in positions, "sampled_mask_sha256": mask["sampled_mask_sha256"],
                    **source.ranked_readout(scores, entry["candidate_physical_rows"])}
                source.need(len(component["scores_binary64"]) == 128 and component["supported_candidate_count"] == 128, "ALL_CANDIDATES_REQUIRED")
                components.append(component)
        source.need(bool(legal_union[seed]) == any(seed in c["query_token_indices"] for c in components), "INDEPENDENT_SEED_MEMBERSHIP")
        stats = {"argmax_alpha_query_index": seed, "argmax_alpha_in_any_native_mask": bool(all_union[seed]),
            "argmax_alpha_in_any_legal_component": bool(legal_union[seed]), "all_mask_union_token_count": int(all_union.sum()),
            "legal_component_union_token_count": int(legal_union.sum()), "alpha_total_binary64": source.hx(alpha.sum()),
            "alpha_inside_all_mask_union_binary64": source.hx(alpha[all_union].sum()),
            "alpha_outside_all_mask_union_binary64": source.hx(alpha[~all_union].sum()),
            "alpha_inside_legal_union_binary64": source.hx(alpha[legal_union].sum()),
            "alpha_outside_legal_union_binary64": source.hx(alpha[~legal_union].sum())}
        total_masks += len(bank); total_components += len(components)
        records.append({"execution_ordinal": entry["execution_ordinal"], "query_id": entry["query_id"], "role": entry["role"],
            "candidate_physical_rows": entry["candidate_physical_rows"], "query_grid_shape": entry["native_grid_shape"],
            "prior_theta_sha256": parameters["theta_sha256"], "original_mask_count": len(bank),
            "all_mask_union_token_indices": all_union.nonzero().flatten().tolist(),
            "legal_component_union_token_indices": legal_union.nonzero().flatten().tolist(),
            "alpha_statistics": stats, "components": components, "target_label_reads": 0})
    source.need(len(records) == 9 and total_masks == 246, "COMPLETE_9_QUERY_246_MASK_BANK")
    return {"records": records, "total_mask_count": total_masks, "total_legal_component_count": total_components,
            "all_component_scores_saved_before_target_join": True, "target_label_reads": 0}


def postjoin(source, prejoin, bindings):
    seal = source.read(OUT / "prejoin_seal.json")
    source.need(seal["prejoin_sha256"] == source.sha(OUT / "prejoin.json") and seal["sources"] == bindings
                and source.BARRIER.blocked == 0, "COMPLETE_BANK_PREJOIN_SEAL")
    source.BARRIER.released = True
    parent = source.read(PARENT / "result.json"); targets = {row["execution_ordinal"]: row for row in parent["rows"]}
    rows = []
    for row in prejoin["records"]:
        old = targets[row["execution_ordinal"]]; target = old["target_position"]
        source.need(old["query_id"] == row["query_id"] and row["candidate_physical_rows"][target] == old["target_physical_row"]
                    and old["role"] == row["role"], "PARENT_TARGET_AXIS")
        outcomes = []
        for component in row["components"]:
            scores = [float.fromhex(x) for x in component["scores_binary64"]]
            rank = component["supported_rank_by_candidate"][target]
            rivals = [i for i in range(128) if i != target]
            rival = max(rivals, key=lambda i: (scores[i], -row["candidate_physical_rows"][i]))
            margin = scores[target] - scores[rival]
            outcomes.append({"region_id": component["region_id"], "target_rank": rank,
                "target_top1_under_fixed_tie_rule": rank == 1, "target_strictly_beats_all127": margin > 0,
                "target_minus_strongest_rival_binary64": source.hx(margin),
                "strongest_rival_physical_row": row["candidate_physical_rows"][rival],
                "contains_original_argmax_alpha": component["contains_original_argmax_alpha"]})
        winners = [item["region_id"] for item in outcomes if item["target_top1_under_fixed_tie_rule"]]
        strict = [item["region_id"] for item in outcomes if item["target_strictly_beats_all127"]]
        rows.append({"execution_ordinal": row["execution_ordinal"], "query_id": row["query_id"], "role": row["role"],
            "target_position": target, "target_physical_row": old["target_physical_row"], "original_mask_count": row["original_mask_count"],
            "legal_component_count": len(outcomes), "target_known_oracle_best_rank": min((x["target_rank"] for x in outcomes), default=None),
            "exists_target_top1_component": bool(winners), "exists_strict_target_winning_component": bool(strict),
            "target_top1_component_ids": winners, "strict_target_winning_component_ids": strict,
            "original_QUERY_REGION_outcome": old["arms"]["QUERY_REGION"], "alpha_statistics": row["alpha_statistics"],
            "all_component_target_outcomes": outcomes})
    return {"status": "RC_CACHED_SAM3_REGION_BANK_CAPACITY_V1_COMPLETE", "sources": bindings,
        "prejoin_seal": source.bind(OUT / "prejoin_seal.json"), "scope": "COMPLETE_FIXED9_QUERY_COMPONENT_BANK_CAPACITY_ORACLE_ONLY_NOT_A_SELECTOR",
        "total_original_masks": prejoin["total_mask_count"], "total_legal_components": prejoin["total_legal_component_count"],
        "rows": rows, "population_counts": {role: sum(row["role"] == role for row in rows) for role in ("TRAIN", "EVAL")},
        "limits": ["Target-known best rank locates capacity in the fixed bank; it is not an achieved retrieval accuracy or a deployed selection rule.",
            "Every component of every original mask is retained, including duplicate supports across prompts.",
            "No seed, prompt, threshold, minimum area, prior parameter, or V rule was changed.",
            "Historical SAM input image bytes remain unsealed; this remains a nine-query internal diagnostic."],
        "new_selector_rules": 0, "training_updates": 0, "encoder_or_RoMa_forward_count": 0,
        "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", choices=("run", "validate"), required=True)
    validate = parser.parse_args().phase == "validate"
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    source = load_source(); bindings, prior, parameters, manifest, parent_prejoin = inputs(source)
    source.need(OUT.exists() if validate else not OUT.exists(), "APPEND_ONLY_OUTPUT_STATE")
    prejoin = component_bank(source, prior, parameters, manifest, parent_prejoin)
    if validate: source.need(source.read(OUT / "prejoin.json") == prejoin, "INDEPENDENT_COMPLETE_COMPONENT_SCORE_RANK_AND_SEED_REPLAY")
    else:
        OUT.mkdir(parents=True)
        source.save(OUT / "prejoin.json", prejoin)
        source.save(OUT / "prejoin_seal.json", {"status": "ALL_COMPONENTS_FULL128_SCORES_PREJOIN_SEALED", "prejoin_sha256": source.sha(OUT / "prejoin.json"),
            "sources": bindings, "query_count": 9, "mask_count": 246, "component_count": prejoin["total_legal_component_count"], "target_label_reads": 0})
    result = postjoin(source, prejoin, bindings)
    if validate:
        source.need(source.read(OUT / "result.json") == result, "INDEPENDENT_ALL_TARGET_RANKS_AND_ORACLE_CAPACITY_REPLAY")
        receipt = {"status": "RC_CACHED_SAM3_REGION_BANK_CAPACITY_V1_INDEPENDENT_VALIDATION_PASS", "result_sha256": source.sha(OUT / "result.json"),
            "checks": {"complete246_mask_component_bank_rebuilt": True, "all_component_full128_scores_and_ranks_recomputed": True,
                "alpha_argmax_membership_and_union_mass_recomputed": True, "targets_joined_only_after_complete_bank_seal": True,
                "all9_queries_reported_without_selection_rule_change": True}, "training_updates": 0, "encoder_or_RoMa_forward_count": 0,
            "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
        source.save(OUT / "independent_validation.json", receipt); print(json.dumps(receipt), flush=True)
    else:
        source.save(OUT / "result.json", result)
        print(json.dumps({"status": result["status"], "total_legal_components": result["total_legal_components"], "rows": [
            {key: row[key] for key in ("execution_ordinal", "query_id", "role", "legal_component_count", "target_known_oracle_best_rank",
                "exists_target_top1_component", "exists_strict_target_winning_component", "alpha_statistics")} for row in result["rows"]]}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
