#!/usr/bin/env python3
"""Frozen cached SAM3 proposal regions and full-reference V-only readout."""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
PREFIX = "rc_cached_sam3_region_closure_v1"
OUT = ROOT / "results" / PREFIX
AUTH = ROOT / "registry/rc_cached_sam3_region_closure_authority_v1_20260909.json"
PLAN = ROOT / "plan/RC_CACHED_SAM3_REGION_CLOSURE_V1_20260909.md"
INPUTS = ROOT / "results/rc_existing_sam3_full64_overlap_v1/manifest.json"
PARAMETERS = ROOT / "results/rc_shared_query_target_prior_development_v1/parameters.json"
EVAL_PARENT = ROOT / "results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json"
TRAIN_PARENT = ROOT / "results/rc_shared_query_prior_train_eval_accounting_v1/result.json"
ARMS = ("ALL", "QUERY_REGION", "REF_REGION", "QUERY_REGION_PRIOR_HALF_ROLL", "REF_GEOMETRY_SHIFT")
BARRIER = None
PINS = {
    "mask_manifest": (INPUTS, "11248cb09c2d599061186093df2c55eebf5644f9d788c1fce9715755bd1faac7"),
    "prior_parameters": (PARAMETERS, "06b927f5d531f20712ec5e4875bad13b8bcd1d82260a31d8609d80719ffd1958"),
    "prior_core": (ROOT / "src/rc_aslo_xf/shared_query_target_prior_v1.py", "c3d759a5837b2b0111ec4a97d7c3538444010c13c855f813355cb6c67059f128"),
    "prior_validation": (ROOT / "results/rc_shared_query_target_prior_development_v1/independent_validation.json", "a71bea2cf6af73d1aa4610c8b2e773b05820b124f224a585b04aff58db78d6bd"),
    "EVAL_label_parent": (EVAL_PARENT, "591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3"),
    "TRAIN_label_parent": (TRAIN_PARENT, "603e4b764a5d58c23263b7b46878ac3df4ed35616e95adccbd52e59c16a91df5"),
}
CONTRACT = {"scope": "ONLY_FIXED_EXISTING_MASK_CACHE_AVAILABLE_FULL64_QUERIES_INTERNAL_READOUT",
    "query_count": 9, "TRAIN_count": 2, "EVAL_count": 7, "candidate_count": 128, "arms": list(ARMS),
    "mask_threshold": .5, "native_center_sampling": "floor((row+.5)*Hm/Hq),floor((col+.5)*Wm/Wq)",
    "native_center_integer_formula": "((2*row+1)*Hm)//(2*Hq),((2*col+1)*Wm)//(2*Wq)",
    "all_original_prompts_and_masks_retained": True, "prompt_count": 4,
    "selector": "LEGAL_FIRST_SEED_COMPONENT_SIZE_GE4_THEN_HIGHEST_SAM_SCORE_TIE_ORIGINAL_PROMPT_AND_MASK_INDEX",
    "component": "ENTIRE_SEED_CONTAINING_4_NEIGHBOR_COMPONENT_OF_NATIVE_CENTER_SAMPLED_BINARY_MASK",
    "minimum_component_size": 4, "query_seed": "FIRST_ARGMAX_FROZEN_ALPHA",
    "reference_seed": "FIRST_ARGMAX_FROZEN_ALPHA_TIMES_ORIGINAL_WQ;ALL_ZERO_WQ_IS_H0",
    "V": "UNIFORM_MEAN_ORIGINAL_FULL_REFERENCE_UNWEIGHTED_MAXSIM_A_ON_SELECTED_COMPONENT",
    "ALL": "ORIGINAL_A_MEAN_DIM1_BIT_EXACT", "H0_score": None, "H0_prediction": None,
    "H0_target_rank": None, "H0_target_MRR": 0., "candidate_tie": "LOWEST_PHYSICAL_ROW",
    "prior_control": "HALF_ROLL_ALPHA_BEFORE_QUERY_SEED_ONLY",
    "reference_control": "WQ_CANDIDATE_AXIS_SHIFT64_BEFORE_REFERENCE_SEED;A_UNCHANGED",
    "postjoin": "ALL_P_COMPONENTS_AND_FULL_C128_SCORES_SEALED_BEFORE_ANY_PARENT_TARGET_READ",
    "historical_SAM_input_image_bytes_sealed": False, "new_encoder_forward_count": 0, "new_RoMa_forward_count": 0,
    "training_updates": 0, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}


def need(value, message):
    if not bool(value): raise RuntimeError(message)


def safe(path):
    path = Path(path).resolve()
    need(path.is_relative_to(WORKSPACE), "SOURCE_OUTSIDE_WORKSPACE")
    need(not any(x in str(path).lower() for x in ("d1_mi", "d1-mi", "grozi", "gisc_prerecall_universe")), "PROTECTED_SOURCE")
    return path


def sha(path):
    h = hashlib.sha256()
    if BARRIER is not None: BARRIER.hash_only += 1
    try:
        with safe(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    finally:
        if BARRIER is not None: BARRIER.hash_only -= 1
    return h.hexdigest()


def bind(path): return {"path": str(safe(path)), "sha256": sha(path)}
def checked(value):
    path = safe(value["path"]); need(sha(path) == value["sha256"], "SOURCE_HASH_DRIFT:" + str(path)); return path
def read(path): return json.loads(safe(path).read_text())
def hx(value): return float(value).hex()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream: stream.write(canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444); return bind(path)


def tensor_sha(value):
    import torch
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() + canonical(list(value.shape)) + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def native_center_mask(mask, query_grid):
    """Sample the existing binary native mask; no resize or interpolation."""
    import torch
    need(mask.ndim == 2 and len(query_grid) == 2 and all(type(x) is int and x > 0 for x in query_grid), "MASK_OR_QUERY_GRID_SHAPE")
    need(bool(torch.isfinite(mask).all()) and bool(((mask == 0) | (mask == 1)).all()), "CACHE_MASK_NOT_ALREADY_BINARY")
    height, width = query_grid; mh, mw = mask.shape
    iy = ((2 * torch.arange(height, dtype=torch.int64) + 1) * mh) // (2 * height)
    ix = ((2 * torch.arange(width, dtype=torch.int64) + 1) * mw) // (2 * width)
    return mask[iy[:, None], ix[None, :]].gt(.5).contiguous()


def connected_components(mask):
    """All 4-neighbor components; no score, candidate, alpha or label input."""
    import torch
    height, width = mask.shape; flat = mask.flatten()
    labels = torch.full((height * width,), -1, dtype=torch.int64); components = []
    for seed in range(height * width):
        if not bool(flat[seed]) or int(labels[seed]) >= 0: continue
        index = len(components); labels[seed] = index; queue = deque([seed]); members = []
        while queue:
            point = queue.popleft(); members.append(point); row, column = divmod(point, width)
            neighbors = []
            if row > 0: neighbors.append(point - width)
            if row + 1 < height: neighbors.append(point + width)
            if column > 0: neighbors.append(point - 1)
            if column + 1 < width: neighbors.append(point + 1)
            for neighbor in neighbors:
                if bool(flat[neighbor]) and int(labels[neighbor]) < 0:
                    labels[neighbor] = index; queue.append(neighbor)
        components.append(sorted(members))
    return labels, components


def choose_region(seed, prepared_masks):
    """Highest-confidence legal mask; complete containing component only."""
    if seed is None: return {"state": "H0", "seed": None, "reason": "ZERO_WQ", "query_token_indices": []}
    for mask in prepared_masks:
        component_index = int(mask["component_by_token"][seed])
        if component_index < 0: continue
        component = mask["components"][component_index]
        if len(component) < 4: continue
        return {"state": "H1", "seed": seed, "prompt_index": mask["prompt_index"], "mask_index": mask["mask_index"],
                "SAM_confidence_binary64": hx(mask["confidence"]), "component_index": component_index,
                "query_token_indices": component, "sampled_mask_sha256": mask["sampled_mask_sha256"]}
    return {"state": "H0", "seed": seed, "reason": "NO_LEGAL_SEED_COMPONENT", "query_token_indices": []}


def ranked_readout(scores, axis):
    valid = [i for i, score in enumerate(scores) if score is not None]
    ranked = sorted(valid, key=lambda i: (-scores[i], axis[i]))
    h0 = sorted((i for i in range(len(axis)) if scores[i] is None), key=lambda i: axis[i])
    ranks = [None] * len(axis)
    for rank, position in enumerate(ranked, 1): ranks[position] = rank
    return {"scores_binary64": [None if score is None else hx(score) for score in scores],
            "supported_rank_by_candidate": ranks, "ranked_candidate_positions_with_H0_last": ranked + h0,
            "H0_positions": h0, "supported_candidate_count": len(valid), "prediction_position": ranked[0] if ranked else None}


def e0():
    import torch
    native = torch.zeros((8, 8), dtype=torch.float32); native[2:6, 2:6] = 1.
    sampled = native_center_mask(native, [4, 4]); labels, components = connected_components(sampled)
    need(components == [[5, 6, 9, 10]], "NATIVE_CENTER_AND_4_COMPONENT")
    illegal_labels, illegal_components = connected_components(torch.tensor([[True, False], [False, False]]))
    legal_labels, legal_components = connected_components(torch.ones((2, 2), dtype=torch.bool))
    masks = [{"confidence": .9, "prompt_index": 0, "mask_index": 0, "component_by_token": illegal_labels, "components": illegal_components, "sampled_mask_sha256": "small"},
             {"confidence": .8, "prompt_index": 1, "mask_index": 0, "component_by_token": legal_labels, "components": legal_components, "sampled_mask_sha256": "legal"}]
    chosen = choose_region(0, masks)
    need(chosen["state"] == "H1" and chosen["prompt_index"] == 1 and chosen["query_token_indices"] == [0, 1, 2, 3], "LEGAL_FIRST_MASK_SELECTION")
    need(choose_region(None, masks)["state"] == "H0", "ZERO_WQ_H0")
    order = ranked_readout([.2, None, .2], [20, 1, 10])
    need(order["prediction_position"] == 2 and order["supported_rank_by_candidate"] == [2, None, 1], "V_SCORE_TIE_H0_RULE")
    need(ranked_readout([None, None], [1, 2])["prediction_position"] is None, "NO_H0_FALLBACK")
    return {"status": "RC_CACHED_SAM3_REGION_CLOSURE_E0_PASS", "checks": {"native_center_sampling_without_interpolation": True,
        "entire_four_neighbor_component": True, "legal_first_confidence_selection": True,
        "zero_WQ_and_empty_family_H0": True, "candidate_tie_and_H0_no_fallback": True},
        "encoder_forward_count": 0, "training_updates": 0}


class LabelBarrier:
    def __init__(self):
        self.paths = {str(EVAL_PARENT.resolve()), str(TRAIN_PARENT.resolve())}
        self.hash_only = 0; self.released = False; self.blocked = 0

    def hook(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        if str(Path(os.fsdecode(args[0])).resolve()) in self.paths and not self.released and not self.hash_only:
            self.blocked += 1; raise RuntimeError("PARENT_LABEL_OR_OUTCOME_READ_BEFORE_FULL_P_V_SEAL")


def source_bindings():
    values = {}
    for name, (path, expected) in PINS.items():
        need(sha(path) == expected, "SOURCE_PIN_DRIFT:" + name); values[name] = bind(path)
    values.update(program=bind(Path(__file__)), plan=bind(PLAN))
    return values


def load_qualified_manifest():
    value = read(INPUTS)
    need(value["status"] == "RC_EXISTING_SAM3_FULL64_QUERY_OVERLAP_MANIFEST_V1_COMPLETE"
         and value["matched_query_count"] == len(value["records"]) == 9
         and value["counts"] == {"EVAL": 7, "TRAIN": 2, "mask_candidates": 246, "mask_files": 36}, "FIXED_AVAILABLE_POPULATION")
    need(value["prompt_order"] == ["product package", "box", "medicine package", "medicine box"]
         and value["legacy_semantics"]["observed_all36_masks_binary_zero_one"] is True
         and value["native_center_reuse"]["all_original_masks_preserved"] is True, "ORIGINAL_MASK_BANK_CONTRACT")
    return value


def load_prior():
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from rc_aslo_xf.shared_query_target_prior_v1 import SharedQueryTargetPrior
    validation = read(PINS["prior_validation"][0])
    need(validation["status"] == "RC_SHARED_QUERY_TARGET_PRIOR_INDEPENDENT_FROZEN_REPLAY_VALIDATION_PASS"
         and all(v is True for v in validation["checks"].values()), "FROZEN_PRIOR_NOT_VALIDATED")
    parameters = read(PARAMETERS); prior = SharedQueryTargetPrior()
    with torch.no_grad(): prior.theta.copy_(torch.tensor([float.fromhex(x) for x in parameters["theta_binary64"]], dtype=torch.float64))
    prior.theta.requires_grad_(False)
    need(tensor_sha(prior.theta) == parameters["theta_sha256"], "PRIOR_THETA_SHA")
    return prior, parameters


def prepare_masks(entry, query_grid):
    import torch
    masks = []; file_count = 0
    for prompt_index, item in enumerate(entry["masks"]):
        need(item["prompt_index"] == prompt_index and item["prompt"] == ["product package", "box", "medicine package", "medicine box"][prompt_index], "ORIGINAL_PROMPT_ORDER")
        payload = torch.load(checked(item), map_location="cpu", mmap=True, weights_only=True)
        need(payload["prompt"] == item["prompt"] and int(payload["image_index"]) == entry["sam_image_index"], "SAM_IMAGE_ADDRESS")
        values, scores = payload["masks"], payload["scores"]
        need(list(values.shape) == item["masks_shape"] and str(values.dtype) == item["masks_dtype"]
             and list(scores.shape) == item["scores_shape"] and [hx(x) for x in scores] == item["scores_binary64"], "ORIGINAL_MASK_OR_SCORE_BYTES")
        need(values.shape[0] == item["mask_count"] and item["original_mask_indices"] == list(range(values.shape[0])), "ORIGINAL_MASK_ORDER")
        need(values.ndim == 4 and values.shape[1] == 1 and values.dtype == torch.float32 and bool(torch.isfinite(scores).all()), "MASK_TENSOR_FORMAT")
        mh, mw = values.shape[-2:]; qh, qw = query_grid
        need(item["native_center_sample_y_indices"] == [((2*r+1)*mh)//(2*qh) for r in range(qh)]
             and item["native_center_sample_x_indices"] == [((2*c+1)*mw)//(2*qw) for c in range(qw)], "MANIFEST_NATIVE_CENTER_RULE")
        for mask_index in range(values.shape[0]):
            sampled = native_center_mask(values[mask_index, 0], query_grid)
            labels, components = connected_components(sampled)
            masks.append({"prompt_index": prompt_index, "mask_index": mask_index, "confidence": float(scores[mask_index]),
                          "component_by_token": labels, "components": components, "sampled_mask_sha256": tensor_sha(sampled)})
        file_count += 1
    need(file_count == 4 and len(masks) == entry["total_mask_candidates_across_prompts"], "ALL_ORIGINAL_MASKS_REQUIRED")
    return sorted(masks, key=lambda row: (-row["confidence"], row["prompt_index"], row["mask_index"]))


def build_prejoin(manifest, prior, parameters):
    import numpy as np
    import torch
    output = []; total_masks = 0
    for entry in manifest["records"]:
        meta = read(checked(entry["local_cache_metadata"]))
        need(meta["query_id"] == entry["query_id"] and meta["execution_ordinal"] == entry["execution_ordinal"]
             and meta["role"] == entry["role"] and meta["axis"] == entry["candidate_physical_rows"]
             and meta["q_grid_shape"] == entry["native_grid_shape"], "CACHE_QUERY_FRAME_AND_AXIS")
        need(sha(entry["source_path"]) == entry["current_source_image_sha256"] == meta["query_source_image_sha256"], "CURRENT_IMAGE_BYTES")
        with np.load(checked(entry["local_cache_arrays"]), allow_pickle=False) as arrays:
            tokens = torch.from_numpy(arrays["q_tokens"].copy()); a = torch.from_numpy(arrays["a"].copy())
            wq = torch.from_numpy(arrays["wq"].copy())
            need(arrays["candidate_positions"].tolist() == list(range(128)), "CACHE_FULL128_AXIS")
        need(tensor_sha(tokens) == meta["q_tokens_sha256"] == entry["query_tokens_sha256"], "ORIGINAL_RAW_QUERY_TOKEN_BYTES")
        query_grid = entry["native_grid_shape"]; count = math.prod(query_grid)
        need(a.shape == wq.shape == (128, count) and a.dtype == wq.dtype == torch.float64
             and bool(torch.isfinite(a).all()) and bool(torch.isfinite(wq).all()) and bool(((wq >= 0) & (wq <= 1)).all()), "A_AND_WQ_FULL_AXIS")
        alpha = prior(tokens, query_grid).detach()
        bank = prepare_masks(entry, query_grid); total_masks += len(bank)
        query_seed = int(torch.argmax(alpha)); rolled_seed = int(torch.argmax(alpha.roll(max(1, count // 2))))
        query_region = choose_region(query_seed, bank); rolled_region = choose_region(rolled_seed, bank)
        whole_region = {"state": "H1", "seed": None, "kind": "ALL_QUERY_TOKENS", "query_token_indices": list(range(count))}
        arms = {}; whole_scores = a.mean(dim=1)
        for arm in ARMS:
            if arm == "ALL": regions = [whole_region] * 128
            elif arm == "QUERY_REGION": regions = [query_region] * 128
            elif arm == "QUERY_REGION_PRIOR_HALF_ROLL": regions = [rolled_region] * 128
            else:
                geometry = wq if arm == "REF_REGION" else wq.roll(64, dims=0)
                regions = [choose_region(None if not bool(geometry[g].any()) else int(torch.argmax(alpha * geometry[g])), bank) for g in range(128)]
            scores = [None if region["state"] == "H0" else float(a[g, region["query_token_indices"]].mean()) for g, region in enumerate(regions)]
            if arm == "ALL":
                # Canonical ALL is exactly the original full a.mean(dim=1).
                need([hx(x) for x in scores] == [hx(x) for x in whole_scores], "ALL_UNIFORM_FULL_AXIS_MEAN_BIT_DRIFT")
            if arm in ("ALL", "QUERY_REGION", "QUERY_REGION_PRIOR_HALF_ROLL"):
                choices = {"shared_region": regions[0]}
            else: choices = {"per_candidate_regions": regions}
            arms[arm] = {"P": choices, **ranked_readout(scores, entry["candidate_physical_rows"])}
        output.append({"query_id": entry["query_id"], "execution_ordinal": entry["execution_ordinal"], "role": entry["role"],
            "candidate_physical_rows": entry["candidate_physical_rows"], "query_grid_shape": query_grid,
            "alpha_sha256": tensor_sha(alpha), "prior_theta_sha256": parameters["theta_sha256"], "source_a_sha256": tensor_sha(a),
            "original_SAM_mask_count": len(bank), "mask_bank_ledger": [{"prompt_index": item["prompt_index"], "mask_index": item["mask_index"],
                "confidence_binary64": hx(item["confidence"]), "sampled_mask_sha256": item["sampled_mask_sha256"],
                "all_component_sizes": [len(component) for component in item["components"]]} for item in bank],
            "arms": arms, "target_label_reads": 0})
    need(len(output) == 9 and total_masks == 246 and sum(x["role"] == "TRAIN" for x in output) == 2, "COMPLETE_FIXED_POPULATION")
    return output


def postjoin(prejoin):
    seal = read(OUT / "prejoin_seal.json")
    need(seal["authority_sha256"] == sha(AUTH) and seal["prejoin_sha256"] == sha(OUT / "prejoin.json")
         and seal["query_count"] == 9 and BARRIER.blocked == 0, "FULL_P_V_PREJOIN_SEAL_REQUIRED")
    BARRIER.released = True
    eval_parent = read(EVAL_PARENT); train_parent = read(TRAIN_PARENT)
    parents = {"EVAL": {row["execution_ordinal"]: row for row in eval_parent["evaluations"]["NATIVE7"]["C_PAIRED"]["actions"]},
               "TRAIN": {row["execution_ordinal"]: row for row in train_parent["FULL_TRAIN32_actions"]["UNIFORM"]}}
    joined = []
    for row in prejoin:
        parent = parents[row["role"]][row["execution_ordinal"]]
        target = parent["target_position"]
        need(parent["query_id"] == row["query_id"] and row["candidate_physical_rows"][target] == parent["target_physical_row"], "POSTJOIN_TARGET_AXIS")
        arms = {}
        for arm, values in row["arms"].items():
            valid = values["scores_binary64"][target] is not None
            rank = values["supported_rank_by_candidate"][target]
            prediction = values["prediction_position"]
            arms[arm] = {"target_valid": valid, "target_rank": rank, "prediction_position": prediction,
                         "prediction_physical_row": None if prediction is None else row["candidate_physical_rows"][prediction],
                         "correct": valid and prediction == target, "MRR": 0. if rank is None else 1. / rank,
                         "valid_candidate_count": values["supported_candidate_count"], "H0_candidate_count": len(values["H0_positions"])}
        joined.append({"query_id": row["query_id"], "execution_ordinal": row["execution_ordinal"], "role": row["role"],
            "target_position": target, "target_physical_row": parent["target_physical_row"], "arms": arms,
            "original_NATIVE7_C_complete_system_reference": {"final_correct": parent["final_correct"], "final_position": parent["final_position"],
                "final_physical_row": parent["final_physical_row"], "decision": parent["decision"]}})
    metrics = {}; comparisons = {}; baseline = {}
    for role in ("TRAIN", "EVAL"):
        rows = [row for row in joined if row["role"] == role]; metrics[role] = {}; comparisons[role] = {}
        baseline[role] = {"query_count": len(rows), "correct": sum(row["original_NATIVE7_C_complete_system_reference"]["final_correct"] for row in rows),
                          "scope": "INDEPENDENT_COMPLETE_SYSTEM_REFERENCE_NOT_THE_V_ONLY_BASELINE"}
        for arm in ARMS:
            values = [row["arms"][arm] for row in rows]
            metrics[role][arm] = {"query_count": len(rows), "correct": sum(x["correct"] for x in values),
                "MRR": sum(x["MRR"] for x in values) / len(rows), "target_valid_count": sum(x["target_valid"] for x in values),
                "prediction_available_count": sum(x["prediction_position"] is not None for x in values),
                "valid_candidate_count_total": sum(x["valid_candidate_count"] for x in values), "H0_candidate_count_total": sum(x["H0_candidate_count"] for x in values),
                "correct_execution_ordinals": [row["execution_ordinal"] for row in rows if row["arms"][arm]["correct"]]}
            if arm != "ALL":
                rescued = [row["execution_ordinal"] for row in rows if row["arms"][arm]["correct"] and not row["arms"]["ALL"]["correct"]]
                broken = [row["execution_ordinal"] for row in rows if not row["arms"][arm]["correct"] and row["arms"]["ALL"]["correct"]]
                comparisons[role][arm + "_vs_ALL"] = {"rescue": len(rescued), "break": len(broken), "paired_net": len(rescued) - len(broken),
                    "rescued_execution_ordinals": rescued, "broken_execution_ordinals": broken}
    return {"status": "RC_CACHED_SAM3_REGION_CLOSURE_V1_COMPLETE", "authority_sha256": sha(AUTH), "prejoin_seal_sha256": sha(OUT / "prejoin_seal.json"),
            "scope": "CACHE_AVAILABLE_ONLY_INTERNAL_2TRAIN_7EVAL_FROZEN_PRIOR_AUTOMATIC_REGION_V_READOUT",
            "metrics": metrics, "paired_comparisons": comparisons, "original_NATIVE7_C_reference": baseline, "rows": joined,
            "candidate_count": 128, "mask_manifest": bind(INPUTS), "all_four_prompts_and_all246_masks_used": True,
            "historical_SAM_input_image_bytes_sealed": False, "historical_SAM_confidence_threshold_sealed": False,
            "limits": ["The9query subset is fixed by existing mask availability and covers only1of4 original NativeC28 failures.",
                "Historical SAM PT files do not seal original image bytes or producer/checkpoint provenance.",
                "These V-only region ranks cannot be presented as an improvement of the full NativeC action system.",
                "No fallback is added on missing25EVAL queries, and no32query GO, ownership, or external confirmation is claimed."],
            "new_encoder_forward_count": 0, "new_SAM_or_RoMa_forward_count": 0, "training_updates": 0,
            "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None, "automatic_stage_advance": False}


def freeze():
    need(not AUTH.exists() and not OUT.exists(), "APPEND_ONLY_AUTHORITY_OR_OUTPUT_EXISTS")
    sources = source_bindings(); load_qualified_manifest(); load_prior()
    test = e0(); value = {"status": "RC_CACHED_SAM3_REGION_CLOSURE_V1_AUTHORIZED", "sources": sources,
                         "contract": CONTRACT, "e0": test, "scientific_GO_or_NO_GO": None}
    save(AUTH, value); print(json.dumps({"status": value["status"], "authority_sha256": sha(AUTH)}), flush=True)


def authority():
    value = read(AUTH)
    need(value["status"] == "RC_CACHED_SAM3_REGION_CLOSURE_V1_AUTHORIZED" and value["sources"] == source_bindings()
         and value["contract"] == CONTRACT and value["e0"]["status"] == "RC_CACHED_SAM3_REGION_CLOSURE_E0_PASS"
         and all(v is True for v in value["e0"]["checks"].values()), "AUTHORITY_OR_E0_DRIFT")
    return value


def execute(validate=False):
    authority(); manifest = load_qualified_manifest(); prior, parameters = load_prior()
    if validate: need(OUT.is_dir(), "OUTPUT_MISSING")
    else: need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    prejoin = build_prejoin(manifest, prior, parameters)
    if validate: need(read(OUT / "prejoin.json") == prejoin, "INDEPENDENT_ALL_MASK_SELECTION_COMPONENT_AND_SCORE_REPLAY")
    else:
        OUT.mkdir(parents=True)
        save(OUT / "prejoin.json", prejoin)
        save(OUT / "prejoin_seal.json", {"status": "ALL_NINE_QUERY_P_AND_FULL_C128_V_SCORES_PREJOIN_SEALED", "authority_sha256": sha(AUTH),
            "prejoin_sha256": sha(OUT / "prejoin.json"), "query_count": 9, "candidate_count": 128, "arms": list(ARMS), "target_label_reads": 0})
    result = postjoin(prejoin)
    if validate:
        need(read(OUT / "result.json") == result, "INDEPENDENT_POSTJOIN_RANK_METRIC_REPLAY")
        receipt = {"status": "RC_CACHED_SAM3_REGION_CLOSURE_V1_INDEPENDENT_ARTIFACT_VALIDATION_PASS",
            "authority_sha256": sha(AUTH), "result_sha256": sha(OUT / "result.json"), "checks": {
                "all246_original_masks_and_scores_read_and_binary_checked": True,
                "all5_arms_seed_component_selection_recomputed_without_labels_or_V_scores": True,
                "all_full128_V_scores_and_ranks_replayed": True, "ALL_full_axis_mean_bit_exact": True,
                "H0_invalid_targets_never_counted_correct": True, "labels_only_after_complete_prejoin": True,
                "2TRAIN_7EVAL_populations_and_parent_system_reference_separate": True},
            "training_updates": 0, "new_encoder_or_RoMa_forward_count": 0, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
        save(OUT / "independent_validation.json", receipt); print(json.dumps(receipt), flush=True)
    else:
        save(OUT / "result.json", result); print(json.dumps({"status": result["status"], "metrics": result["metrics"],
            "paired_comparisons": result["paired_comparisons"], "original_NATIVE7_C_reference": result["original_NATIVE7_C_reference"]}), flush=True)


def main():
    global BARRIER
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", choices=("e0", "freeze", "run", "validate"), required=True)
    phase = parser.parse_args().phase
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    if phase == "e0": print(json.dumps(e0(), sort_keys=True)); return
    BARRIER = LabelBarrier(); sys.addaudithook(BARRIER.hook)
    if phase == "freeze": freeze()
    else: execute(validate=phase == "validate")


if __name__ == "__main__": main()
