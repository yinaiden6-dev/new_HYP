#!/usr/bin/env python3
"""Four fixed pixel conditions, the original frozen ColNomic encoder, no fitting."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import sysconfig

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
PREFIX = "rc_outcome0212_pixel_source_diagnostic_v1"
OUT = ROOT / "results" / PREFIX
AUTH = ROOT / "registry/rc_outcome0212_pixel_source_diagnostic_authority_v1_20260909.json"
PLAN = ROOT / "plan/RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_V1_20260909.md"
LAUNCH = ROOT / "slurm/rc_outcome0212_pixel_source_diagnostic_v1_gpu_30m.sbatch"
MODEL = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b"
BASE = WORKSPACE / "models/downloaded_models/colqwen2.5-7B-base"
QUAL = ROOT / "results/rc_outcome0212_exact_pixel_mask_qualification_v1/receipt.json"
OLD = ROOT / "results/rc_outcome0212_exclusive_token_gold_polygon_v1/result.json"
UTILITY = ROOT / "programs/rc_outcome0212_exact_pixel_polygon_mask_v1.py"
MODES = ("ORIGINAL", "KEEP_TARGET", "ERASE_TARGET", "ALL_GRAY")
QUAL_MODES = {"ORIGINAL": "ORIGINAL", "KEEP_TARGET": "KEEP_TARGET_ON_GRAY", "ERASE_TARGET": "ERASE_TARGET", "ALL_GRAY": "WHOLE_GRAY"}
PINNED = {
    "input_qualification": (QUAL, "725b3bdcd824222a1295716b440daedb042bc984c2ca6dc0f601330911abedc6"),
    "old_exclusive_sets": (OLD, "e3253768658172bb3b6d247d69a2f1259400c9f6122eb8d91097480aa1ecc92b"),
    "mask_utility": (UTILITY, "43edc05600d2e504e59968fb43e417aa220560c356ba0e7d34ab1813b8eabfe4"),
    "original_token_producer": (ROOT / "programs/materialize_romav2_colnomic_current_runtime_bridge_shard_v1.py", "3df9c314df8e1da6fa9b4f60c41e3db7117c6cad3784f35a012c76fdb2fde03e"),
}
CONTRACT = {"query_id": "OUTCOME-0212", "execution_ordinal": 193, "conditions": list(MODES),
    "gray_RGB": [127, 127, 127], "mask": "EXACT_ORIGINAL_PIXEL_CENTERS_BOUNDARY_INCLUDED_TARGET_POLYGON_UNION",
    "input_size_wh": [2160, 3840], "input_frame": "DECODED_RAW_RGB_NO_EXIF_TRANSFORM", "batch_size": 1,
    "model_constructor": "ColQwen2_5.from_pretrained(local_adapter,torch_dtype=torch.bfloat16).to(cuda).eval()",
    "processor_constructor": "ColQwen2_5_Processor.from_pretrained(local_adapter)",
    "extraction": "encoder(**processor.process_images([image]).to(cuda))[0].float();image_token_id_mask;detach().half().cpu().contiguous()",
    "model_parameters_trainable": False, "model_load_count": 1, "maximum_encoder_forward_count": 4,
    "ORIGINAL_token_and_a_bit_gate_before_other_three_forwards": True,
    "query_token_dtype": "torch.float16", "a_dtype": "torch.float64", "a_shape": [128, 720],
    "a_definition": "CPU_FP64_NORMALIZE_Q_AND_R_THEN_FULL_REFERENCE_Q_MATMUL_RT_MAX_DIM1",
    "candidate_axis": "ORIGINAL_RAW_C128_UNCHANGED", "fixed_sets": "ORIGINAL_TARGET1024_12_AND_WRONG1631_5_NO_RESELECTION",
    "cpu_threads": 8, "new_RoMa_forward_count": 0, "training_updates": 0,
    "scientific_GO_or_NO_GO": None, "HYP_GO_claimed": False, "automatic_stage_advance": False}


def need(value, message):
    if not bool(value): raise RuntimeError(message)


def safe(path):
    path = Path(path).resolve()
    need(path.is_relative_to(WORKSPACE), "SOURCE_OUTSIDE_WORKSPACE")
    need(not any(s in str(path).lower() for s in ("d1_mi", "d1-mi", "grozi", "gisc_prerecall_universe")), "PROTECTED_SOURCE")
    return path


def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def binding(path): return {"path": str(safe(path)), "sha256": sha(path)}
def checked(value):
    path = safe(value["path"]); need(sha(path) == value["sha256"], "SOURCE_HASH_DRIFT:" + str(path)); return path
def read(path): return json.loads(safe(path).read_text())
def hx(value): return float(value).hex()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444); return binding(path)


def tensor_sha(tensor):
    import torch
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(tensor.dtype).encode() + canonical(list(tensor.shape)) + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def save_npz(path, arrays):
    import numpy as np
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444); return binding(path)


def utility():
    need(sha(UTILITY) == PINNED["mask_utility"][1], "MASK_UTILITY_PIN")
    spec = importlib.util.spec_from_file_location("exact_outcome0212_mask", UTILITY)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def model_manifest():
    adapter = read(MODEL / "adapter_config.json")
    need(Path(adapter["base_model_name_or_path"]).resolve() == BASE, "LOCAL_BASE_MODEL_PATH_DRIFT")
    index = read(BASE / "model.safetensors.index.json")
    weights = sorted(set(index["weight_map"].values()))
    need(len(weights) == 7 and all(Path(name).name == name for name in weights), "BASE_WEIGHT_SHARD_INDEX")
    files = [MODEL / name for name in ("adapter_config.json", "adapter_model.safetensors", "added_tokens.json", "chat_template.json",
        "merges.txt", "preprocessor_config.json", "processor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")]
    files += [BASE / name for name in ("config.json", "generation_config.json", "model.safetensors.index.json", *weights)]
    site = Path(sysconfig.get_paths()["purelib"])
    code_paths = ["colpali_engine/models/qwen2_5/colqwen2_5/modeling_colqwen2_5.py",
        "colpali_engine/models/qwen2_5/colqwen2_5/processing_colqwen2_5.py", "colpali_engine/utils/processing_utils.py",
        "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", "transformers/models/qwen2_vl/processing_qwen2_vl.py",
        "transformers/models/qwen2_vl/image_processing_qwen2_vl.py", "transformers/integrations/peft.py"]
    files += [site / name for name in code_paths]
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "colpali-engine", "peft", "accelerate", "safetensors", "Pillow", "numpy")}
    return {"model_path": str(MODEL), "base_path": str(BASE), "files": [binding(path) for path in files], "versions": versions,
            "python_executable": sys.executable, "python_version": sys.version, "base_weight_shards": len(weights)}


def source_bindings():
    result = {}
    for key, (path, expected) in PINNED.items():
        need(sha(path) == expected, "PIN_DRIFT:" + key); result[key] = binding(path)
    for key, path in (("program", Path(__file__)), ("plan", PLAN), ("launcher", LAUNCH)):
        result[key] = binding(path)
    return result


def load_inputs():
    import numpy as np
    import torch
    from PIL import Image
    q = read(QUAL); old = read(OLD)
    need(q["status"] == "RC_OUTCOME0212_EXACT_PIXEL_MASK_AND_SOURCE_QUALIFICATION_PASS" and q["query_id"] == "OUTCOME-0212"
         and q["gray_RGB"] == [127, 127, 127] and q["source_frame"]["EXIF_orientation"] == 1, "INPUT_QUALIFICATION")
    for key, value in q["checks"].items():
        need(value is True if key != "run_boundary_neighbor_exact_point_tests" else value > 0, "INPUT_CHECK:" + key)
    for value in q["sources"].values(): checked(value)
    mask = np.load(checked(q["mask"]["file"]), allow_pickle=False)
    with Image.open(checked(q["sources"]["original_query_image"])) as image:
        need(image.size == (2160, 3840) and image.getexif().get(274, 1) == 1, "ORIGINAL_PIXEL_FRAME")
        rgb = np.array(image.convert("RGB"), copy=True)
    ann = read(checked(q["sources"]["annotation"]))
    rebuilt = utility().rasterize_target_mask(utility().target_polygons(ann), 2160, 3840)
    need(mask.dtype == np.bool_ and mask.shape == (3840, 2160) and np.array_equal(mask, rebuilt), "EXACT_TARGET_MASK_REPLAY")
    gray = np.full_like(rgb, 127); keep = np.where(mask[..., None], rgb, gray); erase = np.where(mask[..., None], gray, rgb)
    images = {"ORIGINAL": rgb, "KEEP_TARGET": keep, "ERASE_TARGET": erase, "ALL_GRAY": gray}
    need(np.array_equal(keep.astype(np.int16) + erase.astype(np.int16) - gray.astype(np.int16), rgb.astype(np.int16)), "COMPLEMENTARY_PIXEL_IDENTITY")
    for mode, value in images.items():
        need(hashlib.sha256(value.tobytes()).hexdigest() == q["conditions"][QUAL_MODES[mode]]["raw_contiguous_bytes_sha256"], "PIXEL_CONDITION_SHA:" + mode)
    archive = torch.load(checked(q["sources"]["original_RAW_token_archive"]), map_location="cpu", mmap=True, weights_only=True)
    record = archive["records"][q["original_token_record_index"]]
    need(record["query_id"] == q["query_id"] and record["execution_ordinal"] == 193
         and record["candidate_physical_rows"] == q["candidate_physical_rows"] and list(record["query_grid_shape"]) == [36, 20], "RAW_QUERY_AXIS")
    need(tensor_sha(record["query_tokens"]) == q["query_tokens_sha256"], "ORIGINAL_TOKEN_SHA")
    references = []
    for position, entry in enumerate(q["references"]):
        ref = archive["references"][entry["physical_row"]]
        need(entry["candidate_position"] == position and q["candidate_physical_rows"][position] == entry["physical_row"]
             and list(ref["tokens"].shape) == entry["tokens_shape"] and tensor_sha(ref["tokens"]) == entry["tokens_sha256"], "REFERENCE_TOKEN_BINDING")
        references.append(ref["tokens"])
    with np.load(checked(q["sources"]["original_local_cache_arrays"]), allow_pickle=False) as arrays:
        expected_a = torch.from_numpy(arrays["a"].copy())
    need(expected_a.shape == (128, 720) and expected_a.dtype == torch.float64, "ORIGINAL_A_AXIS")
    need(len(old["candidates"]["1024"]["all_exclusive_records"]) == 12 and len(old["candidates"]["1631"]["all_exclusive_records"]) == 5, "FIXED_EXCLUSIVE_SET_COUNTS")
    return q, old, mask, images, record["query_tokens"], references, expected_a


def compute_a(query, references):
    import torch
    from torch.nn import functional as F
    qn = F.normalize(query.to(torch.float64), dim=1)
    return torch.stack([(qn @ F.normalize(reference.to(torch.float64), dim=1).T).max(1).values for reference in references])


def describe(a, tokens, original, q, old):
    import torch
    axis = q["candidate_physical_rows"]; summaries = {}
    for physical in (1024, 1631):
        source = old["candidates"][str(physical)]; position = axis.index(physical)
        rival_positions = [i for i in range(128) if i != position]
        rivals, indices = a[rival_positions].max(dim=0)
        rows = []
        for item in source["all_exclusive_records"]:
            p = item["query_token_index"]; margin = a[position, p] - rivals[p]
            rp = rival_positions[int(indices[p])]
            rows.append({"query_token_index": p, "original_center_inside_GT": item["center_inside_target_polygon"],
                "subject_physical_row": physical, "subject_score_binary64": hx(a[position, p]),
                "strongest_rival_position": rp, "strongest_rival_physical_row": axis[rp],
                "strongest_rival_score_binary64": hx(rivals[p]), "margin_binary64": hx(margin), "survives_strict_positive": bool(margin > 0)})
        summaries[str(physical)] = {"fixed_token_count": len(rows), "surviving_count": sum(row["survives_strict_positive"] for row in rows), "tokens": rows}
    pooled = a.mean(dim=1); order = sorted(range(128), key=lambda i: (-float(pooled[i]), axis[i]))
    delta = (tokens.to(torch.float64) - original.to(torch.float64)).norm(dim=1)
    changed = tokens.ne(original).any(dim=1)
    return {"fixed_original_exclusive_sets": summaries,
        "mean_MaxSim_scores_binary64": [hx(x) for x in pooled], "mean_MaxSim_ranked_physical_rows": [axis[i] for i in order],
        "target_mean_MaxSim_rank": order.index(axis.index(1024)) + 1, "wrong_mean_MaxSim_rank": order.index(axis.index(1631)) + 1,
        "token_feature_change": {"changed_token_count": int(changed.sum()), "query_token_count": 720,
            "L2_mean": float(delta.mean()), "L2_max": float(delta.max()), "L2_per_token_binary64": [hx(x) for x in delta]}}, delta


def factorial(values):
    o, k, e, g = (values[mode] for mode in MODES)
    return {"target_given_background": o - e, "target_given_gray_background": k - g,
            "background_given_target": o - k, "background_given_gray_target": e - g, "interaction": o - e - k + g}


def reduce_effects(descriptions):
    effects = {}
    for physical in ("1024", "1631"):
        rows = []
        for i, item in enumerate(descriptions["ORIGINAL"]["fixed_original_exclusive_sets"][physical]["tokens"]):
            values = {mode: float.fromhex(descriptions[mode]["fixed_original_exclusive_sets"][physical]["tokens"][i]["margin_binary64"]) for mode in MODES}
            rows.append({"query_token_index": item["query_token_index"], "margin_effects_binary64": {k: hx(v) for k, v in factorial(values).items()}})
        effects[physical] = rows
    pooled = [{key: hx(value) for key, value in factorial({mode: float.fromhex(descriptions[mode]["mean_MaxSim_scores_binary64"][i]) for mode in MODES}).items()} for i in range(128)]
    return {"fixed_original_token_margin_effects": effects, "full_C128_mean_MaxSim_effects": pooled,
            "arithmetic": "ORIGINAL-ERASE;KEEP-GRAY;ORIGINAL-KEEP;ERASE-GRAY;ORIGINAL-ERASE-KEEP+GRAY"}


def encode_original_path(encoder, processor, image):
    """Literal steps from the hash-pinned original producer, no new settings."""
    import torch
    inputs = processor.process_images([image]).to(torch.device("cuda"))
    encoded = encoder(**inputs)[0].float()
    image_mask = inputs["input_ids"][0] == processor.image_token_id
    image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
    _, height, width = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    receipt = {"grid": [height // merge, width // merge], "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
               "input_ids_sha256": tensor_sha(inputs["input_ids"]), "attention_mask_sha256": tensor_sha(inputs["attention_mask"]),
               "pixel_values_sha256": tensor_sha(inputs["pixel_values"]), "image_token_indices": image_mask.nonzero().flatten().cpu().tolist(),
               "image_token_id": int(processor.image_token_id)}
    return image_tokens, receipt


def e0():
    import numpy as np
    import torch
    raster = utility().synthetic_e0()
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3); mask = utility().rasterize_target_mask([[[.5, .5], [3.5, .5], [.5, 2.5]]], 4, 3)
    gray = np.full_like(rgb, 127); keep = np.where(mask[..., None], rgb, gray); erase = np.where(mask[..., None], gray, rgb)
    need(np.array_equal(keep.astype(np.int16) + erase.astype(np.int16) - gray.astype(np.int16), rgb.astype(np.int16)), "SYNTHETIC_PIXEL_COMPLEMENT")
    query = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float16)
    refs = [query, query.flip(0)]
    need(torch.equal(compute_a(query, refs), torch.ones((2, 2), dtype=torch.float64)), "SYNTHETIC_FULL_REFERENCE_A")
    need(factorial(dict(zip(MODES, (10., 5., 4., 1.))))["interaction"] == 2., "SYNTHETIC_FACTORIAL")
    return {"status": "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_E0_PASS", "checks": {"exact_mask_utility": raster["status"] == "EXACT_PIXEL_CENTER_RASTER_E0_PASS",
        "complementary_pixels_exact": True, "FP16_tokens_FP64_full_reference_a": True, "fixed_four_condition_factorial_arithmetic": True},
        "encoder_load_count": 0, "encoder_forward_count": 0, "training_updates": 0}


def freeze():
    need(not AUTH.exists() and not OUT.exists(), "APPEND_ONLY_AUTHORITY_OR_OUTPUT_EXISTS")
    sources = source_bindings(); qualification = read(QUAL)
    need(qualification["status"] == "RC_OUTCOME0212_EXACT_PIXEL_MASK_AND_SOURCE_QUALIFICATION_PASS", "MASK_NOT_QUALIFIED")
    manifest = model_manifest()
    value = {"status": "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_AUTHORIZED", "sources": sources, "model_manifest": manifest,
             "contract": CONTRACT, "e0": e0(), "scientific_GO_or_NO_GO": None}
    save(AUTH, value); print(json.dumps({"status": value["status"], "authority_sha256": sha(AUTH)}), flush=True)


def authority(rehash_weights=True):
    value = read(AUTH)
    need(value["status"] == "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_AUTHORIZED" and value["contract"] == CONTRACT
         and value["sources"] == source_bindings(), "AUTHORITY_SOURCE_DRIFT")
    need(value["e0"]["status"] == "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_E0_PASS" and all(value["e0"]["checks"].values()), "E0_NOT_CLOSED")
    if rehash_weights: need(value["model_manifest"] == model_manifest(), "MODEL_WEIGHT_OR_SOFTWARE_DRIFT")
    return value


def run():
    import numpy as np
    import torch
    from PIL import Image
    auth = authority(); need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    q, old, mask, images, expected_tokens, references, expected_a = load_inputs()
    need(torch.cuda.is_available(), "CUDA_REQUIRED_FOR_ORIGINAL_ENCODER")
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    encoder = ColQwen2_5.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16).to(torch.device("cuda")).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(MODEL))
    need(not any(p.requires_grad for p in encoder.parameters()), "ENCODER_NOT_FROZEN")
    OUT.mkdir(parents=True)
    runtime = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
        "GPU_capability": list(torch.cuda.get_device_capability(0)), "torch_num_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(), "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "attention_implementation": getattr(encoder.config, "_attn_implementation", None),
        "processor_class": type(processor).__module__ + "." + type(processor).__name__, "encoder_class": type(encoder).__module__ + "." + type(encoder).__name__,
        "model_dtype": str(encoder.dtype), "model_training": encoder.training}
    save(OUT / "runtime.json", runtime)
    entries = []; descriptions = {}; first_axis = None
    for mode in MODES:
        image = Image.fromarray(images[mode], mode="RGB")
        path = OUT / (mode + ".png"); image.save(path); path.chmod(0o444)
        tokens, prep = encode_original_path(encoder, processor, image)
        if mode == "ORIGINAL":
            parity = tokens.shape == expected_tokens.shape and tokens.dtype == expected_tokens.dtype and tensor_sha(tokens) == q["query_tokens_sha256"]
            if not parity:
                difference = {"actual_shape": list(tokens.shape), "expected_shape": list(expected_tokens.shape), "actual_sha256": tensor_sha(tokens), "expected_sha256": q["query_tokens_sha256"]}
                if tokens.shape == expected_tokens.shape:
                    delta = (tokens.float() - expected_tokens.float()).abs()
                    difference.update(different_scalar_count=int(tokens.ne(expected_tokens).sum()), max_absolute_difference=float(delta.max()), mean_absolute_difference=float(delta.mean()))
                saved = save_npz(OUT / "ORIGINAL_token_replay_mismatch.npz", {"actual_tokens": tokens.numpy(), "expected_tokens": expected_tokens.numpy()})
                result = {"status": "RC_OUTCOME0212_ORIGINAL_ENCODER_TOKEN_REPLAY_MISMATCH_STOP", "authority_sha256": sha(AUTH), "encoder_forward_count": 1,
                    "subsequent_intervention_forward_count": 0, "difference": difference, "tokens": saved, "preprocessing": prep,
                    "scientific_interpretation_authorized": False, "scientific_GO_or_NO_GO": None, "HYP_GO_claimed": False}
                save(OUT / "result.json", result); print(json.dumps(result), flush=True); return 4
        need(list(tokens.shape) == [720, 128] and tokens.dtype == torch.float16 and prep["grid"] == [36, 20], "FOUR_INPUT_TOKEN_GRID_DRIFT")
        axis_receipt = {key: prep[key] for key in ("grid", "image_grid_thw", "input_ids_sha256", "attention_mask_sha256", "image_token_indices", "image_token_id")}
        if first_axis is None: first_axis = axis_receipt
        else: need(axis_receipt == first_axis, "FOUR_INPUT_IMAGE_TOKEN_ORDER_DRIFT")
        a = compute_a(tokens, references)
        if mode == "ORIGINAL" and tensor_sha(a) != tensor_sha(expected_a):
            saved = save_npz(OUT / "ORIGINAL_a_replay_mismatch.npz", {"q_tokens": tokens.numpy(), "actual_a": a.numpy(), "expected_a": expected_a.numpy()})
            result = {"status": "RC_OUTCOME0212_ORIGINAL_A_REPLAY_MISMATCH_STOP", "authority_sha256": sha(AUTH), "encoder_forward_count": 1,
                "subsequent_intervention_forward_count": 0, "arrays": saved, "scientific_interpretation_authorized": False,
                "scientific_GO_or_NO_GO": None, "HYP_GO_claimed": False}
            save(OUT / "result.json", result); print(json.dumps(result), flush=True); return 4
        description, delta = describe(a, tokens, expected_tokens, q, old)
        if mode == "ORIGINAL":
            for physical in ("1024", "1631"):
                need([x["margin_binary64"] for x in description["fixed_original_exclusive_sets"][physical]["tokens"]]
                     == [x["margin_binary64"] for x in old["candidates"][physical]["all_exclusive_records"]], "ORIGINAL_FIXED_MARGIN_REPLAY")
        arrays = save_npz(OUT / (mode + ".npz"), {"q_tokens": tokens.numpy(), "a": a.numpy(), "token_delta_L2": delta.numpy()})
        entries.append({"mode": mode, "input_png": binding(path), "input_RGB_sha256": hashlib.sha256(images[mode].tobytes()).hexdigest(),
                        "arrays": arrays, "query_tokens_sha256": tensor_sha(tokens), "a_sha256": tensor_sha(a), "preprocessing": prep})
        descriptions[mode] = description
        print(json.dumps({"event": "FIXED_PIXEL_CONDITION_COMPLETE", "mode": mode, "encoder_forward_count": len(entries),
            "fixed_target_surviving": description["fixed_original_exclusive_sets"]["1024"]["surviving_count"],
            "fixed_wrong_surviving": description["fixed_original_exclusive_sets"]["1631"]["surviving_count"]}), flush=True)
    result = {"status": "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_COMPLETE", "authority_sha256": sha(AUTH), "input_qualification": binding(QUAL),
        "runtime": binding(OUT / "runtime.json"), "candidate_physical_rows": q["candidate_physical_rows"], "entries": entries,
        "descriptions": descriptions, "factorial_effects": reduce_effects(descriptions), "original_token_SHA_and_a_bit_exact": True,
        "encoder_load_count": 1, "encoder_forward_count": 4, "RoMa_forward_count": 0, "training_updates": 0,
        "scope": "ONE_OPENED_QUERY_FIXED_GT_PIXEL_DEPENDENCE_ONLY_NOT_A_RETRIEVAL_SYSTEM",
        "limits": ["Token centers do not define independent pixel receptive fields.", "Gray fill and mask boundaries can change the input distribution.",
                   "Only the pre-existing target12/wrong5 token sets are analyzed for survival; no new best tokens are selected.",
                   "These interventions do not prove an actual wrong object exists or certify connected HYP or ownership."],
        "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None, "automatic_stage_advance": False}
    save(OUT / "result.json", result); print(json.dumps({"status": result["status"], "result_sha256": sha(OUT / "result.json")}), flush=True); return 0


def validate():
    import numpy as np
    import torch
    from PIL import Image
    authority(rehash_weights=False)
    result = read(OUT / "result.json")
    need(result["status"] == "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_COMPLETE" and result["authority_sha256"] == sha(AUTH), "RESULT_NOT_COMPLETE")
    q, old, mask, images, original, references, expected_a = load_inputs()
    descriptions = {}
    for entry in result["entries"]:
        mode = entry["mode"]
        with Image.open(checked(entry["input_png"])) as im: pixels = np.array(im.convert("RGB"))
        need(np.array_equal(pixels, images[mode]), "SAVED_PIXEL_INTERVENTION_REPLAY")
        with np.load(checked(entry["arrays"]), allow_pickle=False) as arrays:
            tokens = torch.from_numpy(arrays["q_tokens"].copy()); saved_a = torch.from_numpy(arrays["a"].copy())
            saved_delta = torch.from_numpy(arrays["token_delta_L2"].copy())
        need(tokens.shape == (720, 128) and tokens.dtype == torch.float16 and tensor_sha(tokens) == entry["query_tokens_sha256"], "SAVED_QUERY_TOKEN_BINDING")
        actual = compute_a(tokens, references)
        need(tensor_sha(actual) == tensor_sha(saved_a) == entry["a_sha256"], "ALL_C128_A_INDEPENDENT_REPLAY")
        if mode == "ORIGINAL": need(tensor_sha(tokens) == tensor_sha(original) and tensor_sha(actual) == tensor_sha(expected_a), "HISTORICAL_BASELINE_REPLAY")
        description, delta = describe(actual, tokens, original, q, old)
        need(tensor_sha(delta) == tensor_sha(saved_delta), "TOKEN_CHANGE_REPLAY")
        descriptions[mode] = description
    need(tuple(descriptions) == MODES and descriptions == result["descriptions"] and reduce_effects(descriptions) == result["factorial_effects"], "FIXED_SET_EFFECTS_REPLAY")
    receipt = {"status": "RC_OUTCOME0212_PIXEL_SOURCE_DIAGNOSTIC_INDEPENDENT_ARTIFACT_VALIDATION_PASS", "result_sha256": sha(OUT / "result.json"),
        "authority_sha256": sha(AUTH), "checks": {"exact_original_mask_and_pixel_complements": True,
        "all_four_saved_query_token_SHA_and_historical_original_token_SHA": True, "all_four_full_C128_a_recomputed_from_saved_tokens_and_original_refs": True,
        "fixed_original_sets_subject_rival_margins_and_2x2_effects_replayed": True},
        "additional_encoder_forward_count": 0, "encoder_forward_values_independently_recomputed": False,
        "weight_files_rehashed_in_artifact_validator": False, "training_updates": 0, "scientific_GO_or_NO_GO": None, "HYP_GO_claimed": False}
    save(OUT / "independent_validation.json", receipt); print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", choices=("e0", "freeze", "run", "validate"), required=True)
    phase = parser.parse_args().phase
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    if phase == "e0": print(json.dumps(e0(), sort_keys=True)); return 0
    if phase == "freeze": freeze(); return 0
    if phase == "run": return run()
    validate(); return 0


if __name__ == "__main__": raise SystemExit(main())
