#!/usr/bin/env python3
"""Three fixed pixel views for reference V on an existing automatic H."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "rc_difficult0006_fixed_H_pixel_recovery_v1"
OUT = ROOT / "results" / PREFIX
AUTH = ROOT / "registry/rc_difficult0006_fixed_H_pixel_recovery_authority_v1_20260909.json"
PLAN = ROOT / "plan/RC_DIFFICULT0006_FIXED_H_PIXEL_RECOVERY_V1_20260909.md"
LAUNCH = ROOT / "slurm/rc_difficult0006_fixed_H_pixel_recovery_v1_gpu_30m.sbatch"
BASE_SOURCE = ROOT / "programs/run_rc_outcome0212_pixel_source_diagnostic_v1.py"
BASE_SHA = "f10d035fcabcb4dbf849977b7ffe71f586e8d504ac98f9d50528fc37dd7de0a2"
MODES = ("ORIGINAL", "KEEP_H_ON_GRAY", "TIGHT_CROP_OF_KEEP_H")
# These previously opened case identities are used only in the final report.
TARGET, OLD_WRONG = 599, 594
PINS = {
    "base_worker": (BASE_SOURCE, BASE_SHA),
    "mask_manifest": (ROOT / "results/rc_existing_sam3_full64_overlap_v1/manifest.json", "11248cb09c2d599061186093df2c55eebf5644f9d788c1fce9715755bd1faac7"),
    "P_prejoin": (ROOT / "results/rc_cached_sam3_region_closure_v1/prejoin.json", "0c9bfca90d650e43d54bfc9b66cb88040e6dd49cf90f2bde9a83189336f9f607"),
    "P_prejoin_seal": (ROOT / "results/rc_cached_sam3_region_closure_v1/prejoin_seal.json", "f1360acf466b1cc911404002800b950bd7268a34ecb4d0fa166ac68a73a83bd6"),
    "P_validation": (ROOT / "results/rc_cached_sam3_region_closure_v1/independent_validation.json", "04c2bff0a19eb4f912dd0c4abcd1a1ee0e5678e000dbc97af1baff25a50abcfd"),
}
CONTRACT = {
    "query_id": "DIFFICULT-0006", "execution_ordinal": 6,
    "H_source": "FROZEN_QUERY_REGION_BOX_MASK13_COMPONENT0_SEED471_147_CELLS",
    "H_selection_repeated_or_changed": False, "GT_polygon_or_oracle_support_selection": False,
    "raw_size_wh": [4032, 3024], "old_query_grid_hw": [24, 32], "raw_pixels_per_old_cell": 126,
    "pixel_mask": "UNION_OF_ALL147_ORIGINAL_NATIVE_CELL_RECTANGLES_HALF_OPEN_INTEGER_BOUNDS",
    "conditions": list(MODES), "gray_RGB": [127, 127, 127],
    "crop": "EXACT_BOUNDING_RECTANGLE_OF_H_PIXEL_UNION_ON_KEEP_H_IMAGE_NO_PADDING",
    "crop_bbox_xyxy": [252, 1386, 3906, 2142], "crop_size_wh": [3654, 756],
    "image_frame": "ORIGINAL_RAW_RGB_NO_EXIF_TRANSPOSE", "batch_size": 1,
    "encoder": "UNCHANGED_PARENT_LOCAL_BF16_COLQWEN2_5_AND_PROCESSOR_LITERAL_EXTRACTION",
    "retained_encoder_outputs": "SAME_SINGLE_FORWARD_IMAGE_TOKENS_PLUS_TEMPLATE_TOKENS_AND_RAW_IMAGE_THEN_TEMPLATE_CONCAT",
    "template_tokens_used_for_current_P_or_V": False, "additional_template_or_RAW_scoring": False,
    "original_guard": "QTOKEN_SHA_AND_FULL_C128_A_BIT_EXACT_BEFORE_OTHER_TWO_FORWARDS",
    "primary_readout_fullcanvas": "UNIFORM_A_MEAN_ON_THE_FROZEN147_CELLS",
    "primary_readout_crop": "UNIFORM_A_MEAN_ON_NEW_GRID_CENTERS_INSIDE_THE_SAME_ORIGINAL_H_PIXEL_UNION",
    "crop_center_mapping": "ORIGINAL_XY=CROP_ORIGIN+((col+.5)*crop_width/grid_width,(row+.5)*crop_height/grid_height);CONTAINING_PIXEL_FLOOR",
    "auxiliary_readout": "ALL_NEW_QUERY_TOKENS_MEAN_SEPARATELY_LABELED",
    "candidate_axis": "UNCHANGED_ORIGINAL_RAW_C128_AND_REFERENCE_TOKENS",
    "maximum_encoder_forward_count": 3, "model_load_count": 1, "cpu_threads": 8,
    "new_RoMa_or_SAM_forward_count": 0, "training_updates": 0,
    "crop_changes_effective_resolution_and_encoder_context": True,
    "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None,
}


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def helper():
    need(hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest() == BASE_SHA, "BASE_WORKER_PIN")
    spec = importlib.util.spec_from_file_location("fixed_H_original_encoder_helpers", BASE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources(b):
    output = {}
    for key, (path, expected) in PINS.items():
        need(b.sha(path) == expected, "SOURCE_PIN_DRIFT:" + key)
        output[key] = b.binding(path)
    output.update(program=b.binding(Path(__file__)), plan=b.binding(PLAN), launcher=b.binding(LAUNCH))
    return output


def pixel_union(indices, grid_hw, image_wh):
    import numpy as np
    gh, gw = grid_hw; width, height = image_wh
    need(width % gw == height % gh == 0, "NONINTEGER_NATIVE_PIXEL_TILES")
    need(indices == sorted(set(indices)) and indices and min(indices) >= 0 and max(indices) < gh * gw, "FIXED_H_AXIS")
    dx, dy = width // gw, height // gh
    mask = np.zeros((height, width), dtype=np.bool_)
    for index in indices:
        row, col = divmod(index, gw)
        mask[row * dy:(row + 1) * dy, col * dx:(col + 1) * dx] = True
    rows, cols = [i // gw for i in indices], [i % gw for i in indices]
    bbox = [min(cols) * dx, min(rows) * dy, (max(cols) + 1) * dx, (max(rows) + 1) * dy]
    need(int(mask.sum()) == len(indices) * dx * dy, "PIXEL_UNION_AREA")
    return mask, bbox


def load_inputs(b):
    import numpy as np
    import torch
    from PIL import Image
    manifest = b.read(PINS["mask_manifest"][0])
    entry = next(r for r in manifest["records"] if r["execution_ordinal"] == 6)
    need(entry["query_id"] == "DIFFICULT-0006" and entry["native_grid_shape"] == [24, 32], "QUERY_BINDING")
    seal = b.read(PINS["P_prejoin_seal"][0]); validation = b.read(PINS["P_validation"][0])
    need(seal["prejoin_sha256"] == PINS["P_prejoin"][1] and seal["query_count"] == 9
         and validation["status"] == "RC_CACHED_SAM3_REGION_CLOSURE_V1_INDEPENDENT_ARTIFACT_VALIDATION_PASS"
         and validation["authority_sha256"] == seal["authority_sha256"]
         and all(value is True for value in validation["checks"].values()), "OLD_P_NOT_VALIDATED")
    row = next(r for r in b.read(PINS["P_prejoin"][0]) if r["execution_ordinal"] == 6)
    need(row["query_id"] == entry["query_id"] and row["candidate_physical_rows"] == entry["candidate_physical_rows"], "P_C128_AXIS")
    selected = row["arms"]["QUERY_REGION"]["P"]["shared_region"]
    H = selected["query_token_indices"]
    need(selected["state"] == "H1" and selected["seed"] == 471 and selected["prompt_index"] == 1
         and selected["mask_index"] == 13 and selected["component_index"] == 0 and len(H) == 147, "FROZEN_AUTOMATIC_H_DRIFT")
    path = Path(entry["source_path"])
    need(b.sha(path) == entry["current_source_image_sha256"], "ORIGINAL_IMAGE_SHA")
    with Image.open(path) as image:
        need(image.size == (4032, 3024) and int(image.getexif().get(274, 1)) == 6, "RAW_IMAGE_FRAME")
        rgb = np.array(image.convert("RGB"), copy=True)
    mask, bbox = pixel_union(H, [24, 32], [4032, 3024])
    need(bbox == CONTRACT["crop_bbox_xyxy"], "H_DERIVED_BBOX")
    gray = np.full_like(rgb, 127)
    keep = np.where(mask[..., None], rgb, gray)
    x0, y0, x1, y1 = bbox
    images = {"ORIGINAL": rgb, "KEEP_H_ON_GRAY": keep,
              "TIGHT_CROP_OF_KEEP_H": keep[y0:y1, x0:x1].copy()}
    need(np.array_equal(keep[mask], rgb[mask]) and np.array_equal(keep[~mask], gray[~mask]), "KEEP_H_PIXEL_IDENTITY")
    source = entry["original_RAW_token_source"]
    archive = torch.load(b.checked(source), map_location="cpu", mmap=True, weights_only=True)
    q = archive["records"][source["record_index"]]
    need(q["query_id"] == entry["query_id"] and q["execution_ordinal"] == 6
         and q["candidate_physical_rows"] == entry["candidate_physical_rows"]
         and list(q["query_grid_shape"]) == [24, 32]
         and b.tensor_sha(q["query_tokens"]) == entry["query_tokens_sha256"], "ORIGINAL_QUERY_TOKENS")
    metadata = b.read(b.checked(entry["local_cache_metadata"]))
    need(metadata["query_id"] == entry["query_id"] and metadata["execution_ordinal"] == 6
         and metadata["axis"] == entry["candidate_physical_rows"]
         and metadata["q_tokens_sha256"] == entry["query_tokens_sha256"], "LOCAL_CACHE_METADATA_BINDING")
    refs, ref_ledger = [], []
    for position, physical in enumerate(entry["candidate_physical_rows"]):
        ref = archive["references"][physical]
        need(metadata["candidates"][position]["candidate_position"] == position
             and metadata["candidates"][position]["physical_row"] == physical
             and b.tensor_sha(ref["tokens"]) == metadata["candidates"][position]["reference_tokens_sha256"], "REFERENCE_TOKEN_SHA")
        refs.append(ref["tokens"])
        ref_ledger.append({"candidate_position": position, "physical_row": physical, "token_sha256": b.tensor_sha(ref["tokens"]), "shape": list(ref["tokens"].shape)})
    with np.load(b.checked(entry["local_cache_arrays"]), allow_pickle=False) as arrays:
        original_a = torch.from_numpy(arrays["a"].copy())
        need(arrays["candidate_positions"].tolist() == list(range(128)), "A_CANDIDATE_AXIS")
    need(original_a.shape == (128, 768) and original_a.dtype == torch.float64, "ORIGINAL_A_SHAPE")
    closure = {"query_id": entry["query_id"], "execution_ordinal": 6, "entry": entry, "selected_H": selected,
               "crop_bbox_xyxy": bbox, "H_pixel_count": int(mask.sum()), "H_pixel_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
               "references": ref_ledger, "condition_pixel_SHAs": {mode: hashlib.sha256(value.tobytes()).hexdigest() for mode, value in images.items()},
               "condition_shapes": {mode: list(value.shape) for mode, value in images.items()}, "target_or_GT_used_for_H_or_bbox": False}
    return closure, mask, images, q["query_tokens"], refs, original_a, row["arms"]["QUERY_REGION"]["scores_binary64"]


def support_for_condition(mode, grid_hw, H, full_mask, bbox):
    import numpy as np
    gh, gw = grid_hw
    if mode != "TIGHT_CROP_OF_KEEP_H":
        need(grid_hw == [24, 32], "FULLCANVAS_QUERY_GRID_DRIFT")
        return list(H), {"mapping": "ORIGINAL_FROZEN_NATIVE_H", "selected_cell_count": len(H)}
    x0, y0, x1, y1 = bbox; width, height = x1 - x0, y1 - y0
    # Integer arithmetic is the exact containing-pixel sample of the new
    # query-cell centers mapped back into the original H pixel union.
    iy = y0 + ((2 * np.arange(gh, dtype=np.int64) + 1) * height) // (2 * gh)
    ix = x0 + ((2 * np.arange(gw, dtype=np.int64) + 1) * width) // (2 * gw)
    selected = np.flatnonzero(full_mask[iy[:, None], ix[None, :]].reshape(-1)).tolist()
    need(selected, "CROP_GRID_HAS_NO_CENTER_IN_ORIGINAL_H")
    return selected, {"mapping": "NEW_GRID_CENTERS_IN_SAME_ORIGINAL_H_PIXEL_UNION", "selected_cell_count": len(selected),
                      "new_grid_cell_count": gh * gw, "original_pixel_y_indices": iy.tolist(), "original_pixel_x_indices": ix.tolist(),
                      "unselected_gray_corner_cell_count": gh * gw - len(selected)}


def summarize(a, support, axis, b):
    import torch
    primary = torch.stack([row[support].mean() for row in a])
    whole = a.mean(dim=1)
    def ranked(values):
        order = sorted(range(128), key=lambda i: (-float(values[i]), axis[i]))
        return {"scores_binary64": [b.hx(x) for x in values], "ranked_physical_rows": [axis[i] for i in order],
                "target599_rank": order.index(axis.index(TARGET)) + 1, "old_wrong594_rank": order.index(axis.index(OLD_WRONG)) + 1,
                "top1_physical_row": axis[order[0]]}
    return {"same_H_physical_domain": ranked(primary), "all_query_tokens_auxiliary": ranked(whole)}


def split_encoder_outputs(encoded, image_mask):
    """The original producer's FP32 -> selected FP16 CPU extraction order."""
    import torch
    image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
    template_tokens = encoded[~image_mask].detach().half().cpu().contiguous()
    raw_concat = torch.cat((image_tokens, template_tokens), dim=0).contiguous()
    return image_tokens, template_tokens, raw_concat


def encode_with_template(encoder, processor, image, b):
    """One unchanged literal encoder call; retain both original token subsets."""
    import torch
    inputs = processor.process_images([image]).to(torch.device("cuda"))
    encoded = encoder(**inputs)[0].float()
    image_mask = inputs["input_ids"][0] == processor.image_token_id
    image_tokens, template_tokens, raw_concat = split_encoder_outputs(encoded, image_mask)
    _, height, width = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    receipt = {"grid": [height // merge, width // merge], "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
               "input_ids_sha256": b.tensor_sha(inputs["input_ids"]), "attention_mask_sha256": b.tensor_sha(inputs["attention_mask"]),
               "pixel_values_sha256": b.tensor_sha(inputs["pixel_values"]), "image_token_indices": image_mask.nonzero().flatten().cpu().tolist(),
               "template_token_indices": (~image_mask).nonzero().flatten().cpu().tolist(),
               "image_token_id": int(processor.image_token_id), "encoder_sequence_length": int(encoded.shape[0]),
               "raw_concat_order": "IMAGE_TOKENS_THEN_TEMPLATE_TOKENS_AS_ORIGINAL_RAW_PRODUCER"}
    return image_tokens, template_tokens, raw_concat, receipt


def e0(b):
    import numpy as np
    import torch
    mask, bbox = pixel_union([0, 1, 4], [2, 4], [8, 4])
    need(int(mask.sum()) == 12 and bbox == [0, 0, 4, 4], "SYNTHETIC_H_PIXEL_UNION")
    support, mapping = support_for_condition("TIGHT_CROP_OF_KEEP_H", [4, 4], [0, 1, 4], mask, bbox)
    need(support == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13] and mapping["unselected_gray_corner_cell_count"] == 4, "SYNTHETIC_CROP_SAME_PHYSICAL_DOMAIN")
    rgb = np.arange(96, dtype=np.uint8).reshape(4, 8, 3); gray = np.full_like(rgb, 127)
    keep = np.where(mask[..., None], rgb, gray)
    need(np.array_equal(keep[mask], rgb[mask]) and np.array_equal(keep[~mask], gray[~mask]), "SYNTHETIC_KEEP_H_PIXEL_IDENTITY")
    encoded = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    image_mask = torch.tensor([False, True, True, False, True])
    image_tokens, template_tokens, raw_concat = split_encoder_outputs(encoded, image_mask)
    need(torch.equal(image_tokens, encoded[[1, 2, 4]].half())
         and torch.equal(template_tokens, encoded[[0, 3]].half())
         and torch.equal(raw_concat, encoded[[1, 2, 4, 0, 3]].half()), "SYNTHETIC_ORIGINAL_TOKEN_SPLIT_AND_CONCAT")
    return {"status": "RC_DIFFICULT0006_FIXED_H_PIXEL_RECOVERY_E0_PASS", "checks": {"exact_H_pixel_union": True,
            "crop_centers_exclude_gray_corners_outside_H": True, "KEEP_preserves_H_and_grays_outside": True,
            "same_forward_image_template_split_and_original_RAW_concat_order": True},
            "new_encoder_forward_count": 0, "training_updates": 0}


def freeze(b):
    need(not AUTH.exists() and not OUT.exists(), "APPEND_ONLY_AUTHORITY_OR_OUTPUT_EXISTS")
    value = {"status": "RC_DIFFICULT0006_FIXED_H_PIXEL_RECOVERY_AUTHORIZED", "sources": sources(b),
             "model_manifest": b.model_manifest(), "contract": CONTRACT, "e0": e0(b), "scientific_GO_or_NO_GO": None}
    b.save(AUTH, value)
    print(json.dumps({"status": value["status"], "authority_sha256": b.sha(AUTH)}), flush=True)


def authority(b, rehash_weights=True):
    value = b.read(AUTH)
    need(value["status"] == "RC_DIFFICULT0006_FIXED_H_PIXEL_RECOVERY_AUTHORIZED"
         and value["sources"] == sources(b) and value["contract"] == CONTRACT
         and all(value["e0"]["checks"].values()), "AUTHORITY_DRIFT")
    if rehash_weights:
        need(value["model_manifest"] == b.model_manifest(), "FROZEN_MODEL_SOFTWARE_DRIFT")
    return value


def run(b):
    import torch
    from PIL import Image
    authority(b); need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    closure, mask, images, original_tokens, references, original_a, original_H_scores = load_inputs(b)
    need(torch.cuda.is_available(), "CUDA_REQUIRED")
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    encoder = ColQwen2_5.from_pretrained(str(b.MODEL), torch_dtype=torch.bfloat16).to(torch.device("cuda")).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(b.MODEL))
    need(not any(p.requires_grad for p in encoder.parameters()), "ENCODER_NOT_FROZEN")
    OUT.mkdir(parents=True)
    b.save(OUT / "input_closure.json", closure)
    b.save_npz(OUT / "fixed_H_pixel_mask.npz", {"mask": mask})
    runtime = {"gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__, "CUDA": torch.version.cuda,
               "threads": torch.get_num_threads(), "attention_implementation": getattr(encoder.config, "_attn_implementation", None),
               "float32_matmul_precision": torch.get_float32_matmul_precision(), "model_dtype": str(encoder.dtype)}
    b.save(OUT / "runtime.json", runtime)
    records = []; first_axis = None
    H = closure["selected_H"]["query_token_indices"]; axis = closure["entry"]["candidate_physical_rows"]
    for mode in MODES:
        image = Image.fromarray(images[mode], mode="RGB")
        path = OUT / (mode + ".png"); image.save(path); path.chmod(0o444)
        tokens, template_tokens, raw_concat, preprocessing = encode_with_template(encoder, processor, image, b)
        need(template_tokens.dtype == raw_concat.dtype == torch.float16
             and template_tokens.ndim == 2 and template_tokens.shape[1] == 128
             and raw_concat.shape == (len(tokens) + len(template_tokens), 128)
             and bool(torch.isfinite(template_tokens).all()), "RETAINED_TEMPLATE_TOKEN_DOMAIN")
        if mode == "ORIGINAL" and b.tensor_sha(tokens) != b.tensor_sha(original_tokens):
            saved = b.save_npz(OUT / "ORIGINAL_token_mismatch.npz", {"actual": tokens.numpy(), "expected": original_tokens.numpy(),
                                  "q_template_tokens": template_tokens.numpy(), "q_raw_concat_tokens": raw_concat.numpy()})
            b.save(OUT / "result.json", {"status": "ORIGINAL_TOKEN_BIT_REPLAY_MISMATCH_STOP", "authority_sha256": b.sha(AUTH),
                   "encoder_forward_count": 1, "subsequent_intervention_forward_count": 0, "arrays": saved,
                   "actual_sha256": b.tensor_sha(tokens), "expected_sha256": b.tensor_sha(original_tokens), "scientific_interpretation_authorized": False})
            return 4
        grid = preprocessing["grid"]
        need(tokens.dtype == torch.float16 and list(tokens.shape) == [grid[0] * grid[1], 128]
             and bool(torch.isfinite(tokens).all()), "NEW_QUERY_TOKEN_GRID")
        axes = {key: preprocessing[key] for key in ("grid", "image_grid_thw", "input_ids_sha256", "attention_mask_sha256", "image_token_indices", "image_token_id")}
        if mode == "ORIGINAL": first_axis = axes
        elif mode != "TIGHT_CROP_OF_KEEP_H": need(axes == first_axis, "FULLCANVAS_TOKEN_AXIS_DRIFT")
        support, mapping = support_for_condition(mode, grid, H, mask, closure["crop_bbox_xyxy"])
        a = b.compute_a(tokens, references)
        need(a.shape == (128, len(tokens)) and bool(torch.isfinite(a).all()), "NONFINITE_OR_MALFORMED_A")
        if mode == "ORIGINAL" and b.tensor_sha(a) != b.tensor_sha(original_a):
            saved = b.save_npz(OUT / "ORIGINAL_a_mismatch.npz", {"actual": a.numpy(), "expected": original_a.numpy(),
                                  "q_tokens": tokens.numpy(), "q_template_tokens": template_tokens.numpy(), "q_raw_concat_tokens": raw_concat.numpy()})
            b.save(OUT / "result.json", {"status": "ORIGINAL_A_BIT_REPLAY_MISMATCH_STOP", "authority_sha256": b.sha(AUTH),
                   "encoder_forward_count": 1, "subsequent_intervention_forward_count": 0, "arrays": saved, "scientific_interpretation_authorized": False})
            return 4
        description = summarize(a, support, axis, b)
        if mode == "ORIGINAL":
            need(description["same_H_physical_domain"]["scores_binary64"] == original_H_scores, "ORIGINAL_147_CELL_V_READOUT_DRIFT")
        saved = b.save_npz(OUT / (mode + ".npz"), {"q_tokens": tokens.numpy(), "q_template_tokens": template_tokens.numpy(),
                                                   "q_raw_concat_tokens": raw_concat.numpy(), "a": a.numpy()})
        records.append({"mode": mode, "input_png": b.binding(path), "input_RGB_sha256": closure["condition_pixel_SHAs"][mode],
                        "input_size_wh": [image.width, image.height], "arrays": saved, "query_tokens_sha256": b.tensor_sha(tokens),
                        "query_template_tokens_sha256": b.tensor_sha(template_tokens), "query_raw_concat_tokens_sha256": b.tensor_sha(raw_concat),
                        "template_tokens_used_for_current_P_or_V": False,
                        "a_sha256": b.tensor_sha(a), "preprocessing": preprocessing, "physical_H_query_token_indices": support,
                        "physical_H_mapping": mapping, "readout": description})
        print(json.dumps({"event": "FIXED_H_PIXEL_CONDITION_COMPLETE", "mode": mode, "forward_count": len(records),
                          "query_grid": grid, "physical_H_cells": len(support)}), flush=True)
    result = {"status": "RC_DIFFICULT0006_FIXED_H_PIXEL_RECOVERY_COMPLETE", "authority_sha256": b.sha(AUTH),
              "input_closure": b.binding(OUT / "input_closure.json"), "runtime": b.binding(OUT / "runtime.json"),
              "fixed_H": closure["selected_H"], "crop_bbox_xyxy": closure["crop_bbox_xyxy"], "candidate_physical_rows": axis,
              "records": records, "original_token_A_and_147cell_V_bits_exact": True, "encoder_forward_count": 3,
              "full_single_forward_token_outputs_retained": True, "template_or_RAW_concat_scoring_count": 0,
              "new_SAM_or_RoMa_forward_count": 0, "training_updates": 0,
              "scope": "ONE_OPENED_CASE_REFERENCE_V_REENCODING_OF_EXISTING_AUTOMATIC_H",
              "limits": ["Crop changes effective resolution, token grid and encoder context; it does not isolate a single causal factor.",
                         "The original complete NATIVE7 C action was already correct on this case; the baseline failure is this V-only region mean.",
                         "The primary readout samples the same physical H but its discretization changes with the crop grid.",
                         "All-token crop readout is auxiliary and includes gray corner cells; it is not substituted for the H-domain readout.",
                         "No new proposal, GT/oracle-selected region, retrieval system deployment or HYP GO is claimed."],
              "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
    b.save(OUT / "result.json", result)
    print(json.dumps({"status": result["status"], "result_sha256": b.sha(OUT / "result.json")}), flush=True)
    return 0


def validate(b):
    import numpy as np
    import torch
    from PIL import Image
    authority(b, rehash_weights=False)
    result = b.read(OUT / "result.json")
    need(result["status"] == "RC_DIFFICULT0006_FIXED_H_PIXEL_RECOVERY_COMPLETE" and result["authority_sha256"] == b.sha(AUTH), "RESULT_INCOMPLETE")
    closure, mask, images, original_tokens, refs, original_a, original_H_scores = load_inputs(b)
    need(b.read(b.checked(result["input_closure"])) == closure, "SOURCE_INPUT_CLOSURE_REPLAY")
    need([r["mode"] for r in result["records"]] == list(MODES), "THREE_CONDITION_AXIS")
    for row in result["records"]:
        mode = row["mode"]
        with Image.open(b.checked(row["input_png"])) as image:
            need(np.array_equal(np.asarray(image.convert("RGB")), images[mode]), "SAVED_PIXEL_CONDITION")
        with np.load(b.checked(row["arrays"]), allow_pickle=False) as arrays:
            tokens = torch.from_numpy(arrays["q_tokens"].copy()); saved_a = torch.from_numpy(arrays["a"].copy())
            template = torch.from_numpy(arrays["q_template_tokens"].copy())
            raw_concat = torch.from_numpy(arrays["q_raw_concat_tokens"].copy())
        need(template.dtype == raw_concat.dtype == torch.float16 and template.ndim == 2 and template.shape[1] == 128
             and b.tensor_sha(template) == row["query_template_tokens_sha256"]
             and b.tensor_sha(raw_concat) == row["query_raw_concat_tokens_sha256"]
             and b.tensor_sha(raw_concat) == b.tensor_sha(torch.cat((tokens, template), dim=0)), "RETAINED_TOKEN_SPLIT_CONCAT_REPLAY")
        image_indices = row["preprocessing"]["image_token_indices"]
        template_indices = row["preprocessing"]["template_token_indices"]
        need(len(image_indices) == len(tokens) and len(template_indices) == len(template)
             and not set(image_indices).intersection(template_indices)
             and sorted(image_indices + template_indices) == list(range(row["preprocessing"]["encoder_sequence_length"])), "ENCODER_TOKEN_PARTITION_COMPLETE")
        grid = row["preprocessing"]["grid"]
        need(tokens.dtype == torch.float16 and list(tokens.shape) == [grid[0] * grid[1], 128]
             and b.tensor_sha(tokens) == row["query_tokens_sha256"], "SAVED_NEW_QUERY_TOKEN_AXIS")
        support, mapping = support_for_condition(mode, grid, closure["selected_H"]["query_token_indices"], mask, closure["crop_bbox_xyxy"])
        need(support == row["physical_H_query_token_indices"] and mapping == row["physical_H_mapping"], "SAME_PHYSICAL_H_MAPPING_REPLAY")
        a = b.compute_a(tokens, refs)
        need(b.tensor_sha(a) == b.tensor_sha(saved_a) == row["a_sha256"], "FULL_C128_A_REPLAY")
        description = summarize(a, support, closure["entry"]["candidate_physical_rows"], b)
        need(description == row["readout"], "H_AND_AUXILIARY_READOUT_REPLAY")
        if mode == "ORIGINAL":
            need(b.tensor_sha(tokens) == b.tensor_sha(original_tokens) and b.tensor_sha(a) == b.tensor_sha(original_a)
                 and description["same_H_physical_domain"]["scores_binary64"] == original_H_scores, "ORIGINAL_BASELINE_REPLAY")
    receipt = {"status": "RC_DIFFICULT0006_FIXED_H_PIXEL_RECOVERY_ARTIFACT_VALIDATION_PASS", "result_sha256": b.sha(OUT / "result.json"),
               "authority_sha256": b.sha(AUTH), "checks": {"fixed_automatic_H_and_pixels_replayed": True,
               "original_query_tokens_A_and_H_scores_bit_exact": True, "all_three_C128_A_recomputed_from_saved_tokens_original_refs": True,
               "same_forward_template_tokens_and_original_RAW_concat_partition_preserved": True,
               "crop_H_domain_center_mapping_and_gray_corner_exclusion_replayed": True, "all_primary_and_auxiliary_scores_ranks_replayed": True},
               "additional_encoder_forward_count": 0, "encoder_forward_values_independently_recomputed": False,
               "weight_files_rehashed_in_artifact_validator": False, "training_updates": 0, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
    b.save(OUT / "independent_validation.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("e0", "freeze", "run", "validate"), required=True)
    phase = parser.parse_args().phase
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    b = helper()
    if phase == "e0": print(json.dumps(e0(b), sort_keys=True)); return 0
    if phase == "freeze": freeze(b); return 0
    if phase == "run": return run(b)
    validate(b); return 0


if __name__ == "__main__":
    raise SystemExit(main())
