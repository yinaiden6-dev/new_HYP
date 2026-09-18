#!/usr/bin/env python3
"""CPU-only processor replay and exact per-output-token vision input blocks."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import sysconfig

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
PARENT = ROOT / "results/rc_outcome0212_pixel_source_diagnostic_v1"
OUT = ROOT / "results/rc_outcome0212_packed_pixel_block_accounting_v1"
MODEL = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b"
AUTH = ROOT / "registry/rc_outcome0212_pixel_source_diagnostic_authority_v1_20260909.json"
MODES = ("ORIGINAL", "KEEP_TARGET", "ERASE_TARGET", "ALL_GRAY")
PINS = {
    "pixel_result": (PARENT / "result.json", "81bbaaafb37e898524ac42b959e07bc322baf00ce0aaad4ed46e384fdd949388"),
    "pixel_validation": (PARENT / "independent_validation.json", "b9b83bdc328ab9196b6751c52a3a830f1461f94bb9e3f30152fbee1dfc90c9bd"),
    "pixel_authority": (AUTH, "e3abbd8bd4446359fe9713a42d4f70a8aecfd54661ffda51919ac81f512e3c9b"),
}
WEIGHT_READ_ATTEMPTS = []


def need(value, message):
    if not bool(value): raise RuntimeError(message)


def safe(path):
    path = Path(path).resolve()
    need(path.is_relative_to(WORKSPACE), "SOURCE_OUTSIDE_WORKSPACE")
    need(not any(x in str(path).lower() for x in ("d1_mi", "d1-mi", "grozi", "gisc_prerecall_universe")), "PROTECTED_SOURCE")
    return path


def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def bind(path): return {"path": str(safe(path)), "sha256": sha(path)}
def read(path): return json.loads(safe(path).read_text())
def checked(value):
    path = safe(value["path"]); need(sha(path) == value["sha256"], "SOURCE_HASH_DRIFT:" + str(path)); return path


def tensor_sha(tensor):
    import torch
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(x.dtype).encode() + json.dumps(list(x.shape), separators=(",", ":")).encode()
                          + x.view(torch.uint8).numpy().tobytes()).hexdigest()


def raw_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def bit_changed_rows(actual, original):
    import torch
    need(actual.shape == original.shape and actual.dtype == original.dtype, "COMPARISON_ARRAY_AXIS")
    a = actual.contiguous().view(torch.uint8).reshape(actual.shape[0], -1)
    b = original.contiguous().view(torch.uint8).reshape(original.shape[0], -1)
    return a.ne(b).any(dim=1)


def prohibit_weight_read(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
    path = Path(os.fsdecode(args[0]))
    if path.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth"):
        WEIGHT_READ_ATTEMPTS.append(str(path))
        raise RuntimeError("ENCODER_OR_TENSOR_CHECKPOINT_READ_FORBIDDEN_IN_PROCESSOR_ONLY_AUDIT")


def merged_blocks(pixel_values, image_grid_thw, patch_size, temporal_patch_size, merge_size):
    """Processor order: T,Hmerge,Wmerge,merge_h,merge_w,C,Tpatch,PH,PW.

    Every successive merge_size**2 patch rows belongs to one row-major merged
    image token. Vision window reordering is reversed after the merger before
    image embeddings are scattered into image_token_id positions.
    """
    t, h, w = [int(x) for x in image_grid_thw[0].tolist()]
    need((t, h, w, patch_size, temporal_patch_size, merge_size) == (1, 72, 40, 14, 2, 2), "FROZEN_PACKING_AXES_DRIFT")
    need(pixel_values.shape == (1, t * h * w, 3 * temporal_patch_size * patch_size * patch_size), "PIXEL_VALUES_PACKING_SHAPE")
    return pixel_values[0].contiguous().reshape(720, 4, 1176)


def e0():
    import torch
    # Independent coordinate enumeration checks the source view/permute order
    # for a small synthetic image, including channels and repeated time slots.
    image = torch.arange(1 * 2 * 3 * 56 * 84, dtype=torch.float64).reshape(1, 2, 3, 56, 84)
    packed = image.view(1, 1, 2, 3, 2, 2, 14, 3, 2, 14).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(6, 4, 1176)
    for r in range(2):
        for c in range(3):
            pieces = []
            for mr in range(2):
                for mc in range(2):
                    patch = image[0, :, :, (2*r+mr)*14:(2*r+mr+1)*14, (2*c+mc)*14:(2*c+mc+1)*14]
                    pieces.append(patch.permute(1, 0, 2, 3).contiguous().flatten())
            need(torch.equal(packed[r*3+c], torch.stack(pieces)), "SYNTHETIC_PIXEL_BLOCK_ORDER")
    signed = torch.tensor([[0., -0.]], dtype=torch.float32)
    changed = torch.tensor([[0., 0.]], dtype=torch.float32)
    need(bool(bit_changed_rows(changed, signed)[0]), "BIT_COMPARISON_MUST_INCLUDE_SIGNED_ZERO")
    return {"status": "PACKED_PIXEL_BLOCK_MAPPING_E0_PASS", "synthetic_merged_blocks_checked": 6,
            "bit_comparison_includes_signed_zero": True, "encoder_forward_count": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("e0", "run"), default="run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    smoke = e0()
    if args.phase == "e0": print(json.dumps(smoke)); return
    need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    sys.addaudithook(prohibit_weight_read)
    sources = {}
    for name, (path, expected) in PINS.items():
        need(sha(path) == expected, "PIN_DRIFT:" + name); sources[name] = bind(path)
    parent = read(PARENT / "result.json"); validation = read(PARENT / "independent_validation.json"); authority = read(AUTH)
    need(parent["status"] == "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_COMPLETE"
         and validation["status"] == "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_INDEPENDENT_ARTIFACT_VALIDATION_PASS"
         and validation["result_sha256"] == sources["pixel_result"]["sha256"]
         and all(v is True for v in validation["checks"].values()), "PIXEL_EXPERIMENT_NOT_VALIDATED")
    manifest = authority["model_manifest"]
    need(all(importlib.metadata.version(name) == version for name, version in manifest["versions"].items()), "PROCESSOR_SOFTWARE_VERSION_DRIFT")
    # Configuration and actual implementation are checked; no model tensors
    # are opened or instantiated. The checkpoint read barrier enforces this.
    checked_sources = []
    for item in manifest["files"]:
        if Path(item["path"]).suffix != ".safetensors": checked(item); checked_sources.append(item)
    from colpali_engine.models import ColQwen2_5_Processor
    processor = ColQwen2_5_Processor.from_pretrained(str(MODEL), local_files_only=True)
    need(not torch.cuda.is_initialized(), "CUDA_CONTEXT_MUST_REMAIN_UNINITIALIZED")
    packing = {"patch_size": int(processor.image_processor.patch_size),
               "temporal_patch_size": int(processor.image_processor.temporal_patch_size),
               "merge_size": int(processor.image_processor.merge_size)}
    entries = {}; tokens = {}; raw_blocks = {}; model_dtype_blocks = {}
    for entry in parent["entries"]:
        mode = entry["mode"]
        with Image.open(checked(entry["input_png"])) as image:
            inputs = processor.process_images([image.convert("RGB")])
        need(all(value.device.type == "cpu" for value in inputs.values() if isinstance(value, torch.Tensor)), "PROCESSOR_NOT_CPU")
        need(tensor_sha(inputs["pixel_values"]) == entry["preprocessing"]["pixel_values_sha256"], "CPU_PIXEL_VALUES_VS_GPU_RECEIPT_BIT_DRIFT:" + mode)
        for key in ("input_ids", "attention_mask"):
            need(tensor_sha(inputs[key]) == entry["preprocessing"][key + "_sha256"], "TOKEN_PROMPT_AXIS_DRIFT:" + mode)
        indices = (inputs["input_ids"][0] == processor.image_token_id).nonzero().flatten().tolist()
        need(indices == entry["preprocessing"]["image_token_indices"] == list(range(4, 724)), "IMAGE_TOKEN_SEQUENCE_AXIS")
        block = merged_blocks(inputs["pixel_values"], inputs["image_grid_thw"], **packing)
        need(block.dtype == torch.float32, "PRECAST_PIXEL_VALUES_DTYPE")
        raw_blocks[mode] = block.clone()
        # The pinned vision path casts pixel_values to visual BF16 before its
        # disjoint Conv3d patch embedding; this is only a deterministic cast.
        model_dtype_blocks[mode] = block.to(torch.bfloat16)
        with np.load(checked(entry["arrays"]), allow_pickle=False) as arrays:
            tokens[mode] = torch.from_numpy(arrays["q_tokens"].copy())
        need(tensor_sha(tokens[mode]) == entry["query_tokens_sha256"], "ENCODED_TOKEN_SOURCE_SHA")
        entries[mode] = {"input_png": entry["input_png"], "query_tokens": entry["arrays"],
                         "whole_pixel_values_sha256": tensor_sha(inputs["pixel_values"]), "whole_pixel_values_match_GPU_receipt": True}
    need(tuple(entries) == MODES, "FOUR_CONDITION_ORDER")
    output_arrays = {}; summaries = {}; ledgers = {}
    original_raw = raw_blocks["ORIGINAL"]; original_bf16 = model_dtype_blocks["ORIGINAL"]; original_tokens = tokens["ORIGINAL"]
    fixed = {physical: [row["query_token_index"] for row in parent["descriptions"]["ORIGINAL"]["fixed_original_exclusive_sets"][physical]["tokens"]]
             for physical in ("1024", "1631")}
    for mode in MODES:
        raw_changed = bit_changed_rows(raw_blocks[mode], original_raw)
        bf16_changed = bit_changed_rows(model_dtype_blocks[mode], original_bf16)
        encoded_changed = bit_changed_rows(tokens[mode], original_tokens)
        evidence = (~raw_changed) & encoded_changed
        output_arrays[mode + "__raw_block_changed"] = raw_changed.numpy()
        output_arrays[mode + "__BF16_block_changed"] = bf16_changed.numpy()
        output_arrays[mode + "__encoded_token_changed"] = encoded_changed.numpy()
        output_arrays[mode + "__local_input_unchanged_encoded_changed"] = evidence.numpy()
        rows = []
        for index in range(720):
            rows.append({"query_token_index": index, "image_input_id_sequence_position": index + 4,
                "merged_grid_row": index // 20, "merged_grid_column": index % 20,
                "vision_input_patch_rows": list(range(4*index, 4*index+4)),
                "raw_FP32_pixel_block_sha256": raw_sha(raw_blocks[mode][index]),
                "effective_BF16_pixel_block_sha256": raw_sha(model_dtype_blocks[mode][index]),
                "raw_FP32_input_changed": bool(raw_changed[index]), "effective_BF16_input_changed": bool(bf16_changed[index]),
                "encoded_token_changed": bool(encoded_changed[index]),
                "raw_input_unchanged_but_encoding_changed": bool(evidence[index])})
        ledgers[mode] = rows
        summaries[mode] = {"raw_FP32_unchanged_block_count": int((~raw_changed).sum()),
            "effective_BF16_unchanged_block_count": int((~bf16_changed).sum()), "changed_encoded_token_count": int(encoded_changed.sum()),
            "raw_input_unchanged_but_encoding_changed_count": int(evidence.sum()),
            "fixed_original_exclusive_sets": {physical: {"indices": ids,
                "raw_input_unchanged_indices": [i for i in ids if not bool(raw_changed[i])],
                "raw_input_unchanged_encoding_changed_indices": [i for i in ids if bool(evidence[i])]} for physical, ids in fixed.items()}}
    need(not WEIGHT_READ_ATTEMPTS and not torch.cuda.is_initialized(), "FORBIDDEN_MODEL_OR_CUDA_ACTIVITY")
    OUT.mkdir(parents=True)
    path = OUT / "per_token_block_change_flags.npz"
    with path.open("xb") as stream: np.savez(stream, **output_arrays)
    path.chmod(0o444)
    result = {"status": "RC_OUTCOME0212_PACKED_PIXEL_BLOCK_ACCOUNTING_COMPLETE", "sources": sources,
        "program": bind(Path(__file__)), "checked_processor_and_model_code_without_weights": checked_sources,
        "e0": smoke, "packing": {**packing, "image_grid_thw": [1, 72, 40], "output_grid": [36, 20],
            "resized_image_hw": [1008, 560], "processor_pixel_values_shape": [1, 2880, 1176],
            "per_output_token_block_shape": [4, 1176], "map": "output_token_p -> pixel_values[0,4*p:4*p+4,:]",
            "processor_order": "B,T,Hmerge,Wmerge,merge_h,merge_w,C,Tpatch,patch_h,patch_w",
            "vision_window_order_restored_before_image_token_scatter": True,
            "original_pixel_center_or_GT_membership_used_to_define_local_block": False},
        "entries": entries, "summaries": summaries, "all720_token_pixel_block_ledgers": ledgers,
        "change_flags": bind(path), "new_encoder_load_count": 0, "new_encoder_forward_count": 0,
        "GPU_context_initialized": False, "model_weight_read_attempts": [], "training_updates": 0,
        "interpretation_limits": ["Identical actual preprocessed input blocks plus changed encoded tokens demonstrates dependence beyond that block for this intervention.",
            "The four-block group is a disjoint patch-embedding input group, not the receptive field of the later vision transformer or language model.",
            "This does not identify which remote layer or pixel caused the change, nor certify target ownership or generalization."],
        "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
    path = OUT / "result.json"
    with path.open("x") as stream: json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    path.chmod(0o444)
    print(json.dumps({"status": result["status"], "result_sha256": sha(path), "summaries": summaries}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
