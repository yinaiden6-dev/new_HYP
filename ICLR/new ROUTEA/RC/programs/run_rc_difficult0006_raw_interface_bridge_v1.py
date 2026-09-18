#!/usr/bin/env python3
"""Frozen RAW score-interface bridge; no encoder, proposal training or action."""
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
WORKSPACE = ROOT.parents[2]
PREFIX = "rc_difficult0006_raw_interface_bridge_v1"
OUT = ROOT / "results" / PREFIX
AUTH = ROOT / "registry/rc_difficult0006_raw_interface_bridge_authority_v1_20260909.json"
PLAN = ROOT / "plan/RC_DIFFICULT0006_RAW_INTERFACE_BRIDGE_V1_20260909.md"
LAUNCH = ROOT / "slurm/rc_difficult0006_raw_interface_bridge_v1_gpu_30m.sbatch"
RAW_SOURCE = ROOT / "programs/run_romav2_colnomic_sealed_source_e0_v2.py"
GALLERY = WORKSPACE / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
PARENT = ROOT / "results/rc_difficult0006_fixed_H_pixel_recovery_v1"
RAW_ARCHIVE = ROOT / "results/romav2_colnomic_current_runtime_bridge_prejoin_v1/shard03/payload.pt"
MODES = ("ORIGINAL", "KEEP_H_ON_GRAY", "TIGHT_CROP_OF_KEEP_H")
RAW_SOURCE_SHA = "2f43fcb16b4ed3345af5799aaf5f458ea87aaddce551b6fb738e7ea0c2c20327"
GALLERY_SHA = "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc"
REF_BRIDGE = ROOT / "results/rc_difficult0006_RAW_full_vs_image_reference_bridge_v1"
Q_GROUPS = ("FULL", "IMAGE", "TEMPLATE", "H", "H_PLUS_TEMPLATE")
R_GROUPS = ("FULL", "IMAGE", "TEMPLATE")
BACKENDS = ("GPU_FP32_DOT", "CPU_FP64_DOT", "CPU_FP64_RENORM")
PINS = {
    "original_RAW_function": (RAW_SOURCE, RAW_SOURCE_SHA), "full_gallery": (GALLERY, GALLERY_SHA),
    "original_RAW_archive": (RAW_ARCHIVE, "5c05fb4cf96bda8214fe4dbfdd0d237e90ef9fa571713f034e1005005182ca5d"),
    "three_query_result": (PARENT / "result.json", "cc68ecbeac2de1395b9187fdec709de6c8bf434b39716f7498a0c6384ba5d8de"),
    "three_query_validation": (PARENT / "independent_validation.json", "f7d2536b8e719874d196f492052f78c297427acf47c66ef95bc3558e86ca5792"),
    "reference_byte_bridge": (REF_BRIDGE / "result.json", "f529603a0023afe031b24adaa883a2fed3be4e5fb79b4a724e93f2af098ee303"),
    "reference_index_validation": (REF_BRIDGE / "image_template_index_validation.json", "14c5fdb404975c58722f346d8d8b7dbf9d35f5a5c75170901969e2924c3f38c9"),
    "reference_full_source_validation": (REF_BRIDGE / "full_passage_source_hash_validation.json", "2fd0f3bc925358d43a4665a3b2599e123cf7cea904ad13185d442465bc63f661"),
}
CONTRACT = {"query_id": "DIFFICULT-0006", "execution_ordinal": 6, "conditions": list(MODES),
    "Q_groups": list(Q_GROUPS), "R_groups": list(R_GROUPS), "numeric_paths": list(BACKENDS),
    "original_RAW_gate": "EXACT_OLD_AST_FULL5413_GALLERY_PHYSICAL_BATCH16_FP32_SUM_MAXSIM_C128_BITS",
    "failed_ORIGINAL_gate": "STOP_BEFORE_ALL_OTHER_CONTROL_SCORING",
    "GPU_control_geometry": "ORIGINAL_PHYSICAL_GALLERY_BATCH16_FULL_NEIGHBORS_AND_PADDING;SAME_FULL_Q_BY_FULL_R_EINSUM_BEFORE_SUBSET_MASKS",
    "R_ablation": "MASK_REFERENCE_COLUMNS_AFTER_SAME_SIMILARITY_MATRIX", "Q_ablation": "SUM_SELECTED_QUERY_ROWS_AFTER_REFERENCE_MAX",
    "Q_order": "ALL_IMAGE_TOKENS_THEN_ORIGINAL_NONIMAGE_TEMPLATE_TOKENS", "R_order": "ORIGINAL_SAVED_VALID_PASSAGE_ORDER",
    "FP64_comparison": "CPU_FP64_DOT_TO_CPU_FP64_RENORM_ONLY_ADDS_L2_NORMALIZATION",
    "FP32_to_FP64_limit": "GPU32_TO_CPU64_CHANGES_BOTH_PRECISION_AND_BACKEND",
    "H_indices": "SAME_FROZEN_PHYSICAL_H_MAPPING_FROM_THREE_QUERY_PIXEL_PARENT",
    "H_PLUS_TEMPLATE_limit": "ORIGINAL_TEMPLATES_SEE_ORIGINAL_FULL_IMAGE_NOT_REGION_ONLY",
    "SUM_to_MEAN": "SAME_SELECTED_QUERY_SET_POSITIVE_COMMON_SCALE_NOT_AN_INDEPENDENT_RANKING_MECHANISM",
    "candidate_count": 128, "gallery_layout_count": 5413, "batch_size": 16, "CPU_threads": 8,
    "new_encoder_load_count": 0, "new_encoder_forward_count": 0, "RoMa_forward_count": 0, "training_updates": 0,
    "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}


def need(value, message):
    if not bool(value): raise RuntimeError(message)


def safe(path):
    path = Path(path).resolve()
    need(path.is_relative_to(WORKSPACE), "SOURCE_OUTSIDE_WORKSPACE")
    need(not any(s in str(path).lower() for s in ("d1_mi", "d1-mi", "grozi", "gisc_prerecall_universe")), "PROTECTED_SOURCE")
    return path


def sha(path):
    digest = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): digest.update(block)
    return digest.hexdigest()


def binding(path): return {"path": str(safe(path)), "sha256": sha(path)}
def checked(value):
    path = safe(value["path"]); need(sha(path) == value["sha256"], "SOURCE_HASH_DRIFT:" + str(path)); return path
def read(path): return json.loads(safe(path).read_text())
def hx(value): return float(value).hex()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def tensor_sha(tensor):
    import torch
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(tensor.dtype).encode() + canonical(list(tensor.shape)) + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def save(path, value):
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream: stream.write(canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444); return binding(path)


def save_npz(path, arrays):
    import numpy as np
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream: np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444); return binding(path)


def old_raw_function():
    import torch
    need(sha(RAW_SOURCE) == RAW_SOURCE_SHA, "ORIGINAL_RAW_FUNCTION_SOURCE_PIN")
    nodes = [node for node in ast.parse(RAW_SOURCE.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "full_gallery_scores"]
    need(len(nodes) == 1, "RAW_FUNCTION_AST_NOT_UNIQUE")
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(RAW_SOURCE), "exec"), namespace)
    return namespace["full_gallery_scores"]


def original_RAW_gate(full_query, passages, candidate_axis, expected_scores, device):
    """Run the exact original full-gallery batch16 function, expose only C128."""
    import torch
    scores = old_raw_function()(full_query, passages, device, batch_size=16)[candidate_axis]
    passed = tensor_sha(scores) == tensor_sha(expected_scores)
    delta = (scores - expected_scores).abs()
    receipt = {"passed": passed, "candidate_count": 128, "gallery_count": len(passages), "batch_size": 16,
        "actual_scores_sha256": tensor_sha(scores), "expected_scores_sha256": tensor_sha(expected_scores),
        "different_scalar_count": int(scores.ne(expected_scores).sum()), "max_absolute_difference": float(delta.max()),
        "actual_scores_binary64": [hx(x) for x in scores], "expected_scores_binary64": [hx(x) for x in expected_scores]}
    need(scores.dtype == torch.float64, "ORIGINAL_RAW_OUTPUT_DTYPE")
    return scores, receipt


def sources():
    result = {}
    for name, (path, expected) in PINS.items():
        need(sha(path) == expected, "PIN_DRIFT:" + name); result[name] = binding(path)
    result.update(program=binding(Path(__file__)), plan=binding(PLAN), launcher=binding(LAUNCH))
    return result


def load_inputs():
    import numpy as np
    import torch
    parent = read(PARENT / "result.json"); validation = read(PARENT / "independent_validation.json")
    need(parent["status"].endswith("COMPLETE") and validation["result_sha256"] == PINS["three_query_result"][1]
         and all(v is True for v in validation["checks"].values()), "THREE_QUERY_OUTPUT_NOT_VALIDATED")
    bridge = read(REF_BRIDGE / "result.json"); indices = read(REF_BRIDGE / "image_template_index_validation.json")
    full = read(REF_BRIDGE / "full_passage_source_hash_validation.json")
    need(indices["image_index_ledger_matches"] == 128 and indices["all_original_cached_image_indices_equal_byte_verified_passage_range4_to4plusN"] is True
         and indices["gallery_saved_padding_tokens"] is False and full["matches"] == full["count"] == 128, "REFERENCE_FULL_IMAGE_TEMPLATE_INDEX_QUALIFICATION")
    gallery = torch.load(GALLERY, map_location="cpu", mmap=True, weights_only=False)["passage_emb"]
    need(len(gallery) == 5413, "ORIGINAL_FULL_GALLERY_AXIS")
    raw_archive = torch.load(RAW_ARCHIVE, map_location="cpu", mmap=True, weights_only=True)
    raw = raw_archive["records"][7]
    axis = parent["candidate_physical_rows"]
    need(raw["execution_ordinal"] == 6 and raw["query_id"] == "DIFFICULT-0006"
         and raw["candidate_physical_rows"] == bridge["candidate_physical_rows"] == axis and len(axis) == len(set(axis)) == 128, "ORIGINAL_RAW_C128_QUERY_AXIS")
    reference_indices = {}
    for row in bridge["reference_rows"]:
        physical = row["physical_row"]; passage = gallery[physical]
        image_idx = row["image_token_indices_in_valid_passage"]; template_idx = row["template_token_indices_in_valid_passage"]
        need(tensor_sha(passage) == row["full_passage_tokens_sha256"]
             and tensor_sha(passage[image_idx]) == row["image_subset_sha256"] == tensor_sha(raw_archive["references"][physical]["tokens"])
             and tensor_sha(passage[template_idx]) == row["template_subset_sha256"], "REFERENCE_PARTITION_BYTES")
        need(not set(image_idx).intersection(template_idx) and sorted(image_idx + template_idx) == list(range(len(passage)))
             and len(template_idx) == 11, "REFERENCE_VALID_PASSAGE_PARTITION")
        reference_indices[physical] = {"IMAGE": image_idx, "TEMPLATE": template_idx}
    queries = {}
    for record in parent["records"]:
        mode = record["mode"]
        with np.load(checked(record["arrays"]), allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays["q_tokens"].copy()); template = torch.from_numpy(arrays["q_template_tokens"].copy())
            fullq = torch.from_numpy(arrays["q_raw_concat_tokens"].copy()); original_a = torch.from_numpy(arrays["a"].copy())
        need(tensor_sha(image) == record["query_tokens_sha256"] and tensor_sha(template) == record["query_template_tokens_sha256"]
             and tensor_sha(fullq) == record["query_raw_concat_tokens_sha256"] == tensor_sha(torch.cat((image, template))), "QUERY_FULL_PARTITION_BYTES")
        ni, nt = len(image), len(template); H = record["physical_H_query_token_indices"]
        need(nt == 11 and all(0 <= i < ni for i in H) and len(H) == len(set(H)), "QUERY_H_AND_TEMPLATE_AXIS")
        groups = {"FULL": list(range(ni + nt)), "IMAGE": list(range(ni)), "TEMPLATE": list(range(ni, ni + nt)),
                  "H": H, "H_PLUS_TEMPLATE": H + list(range(ni, ni + nt))}
        queries[mode] = {"full": fullq, "image_count": ni, "template_count": nt, "groups": groups, "source": record["arrays"],
                         "old_image_only_a": original_a, "old_H_indices": H}
    need(tuple(queries) == MODES and tensor_sha(queries["ORIGINAL"]["full"][:queries["ORIGINAL"]["image_count"]]) == raw["query_tokens_sha256"], "ORIGINAL_QUERY_IMAGE_REPLAY")
    expected = raw["candidate_raw_scores"]
    need(expected.dtype == torch.float64 and [hx(x) for x in expected] == bridge["original_C128_RAW_scores_binary64"], "ORIGINAL_RAW_SCORE_SOURCE")
    return queries, gallery, axis, reference_indices, expected


def fixed_layout_readout(query, gallery, axis, reference_indices, groups, backend):
    """Keep original physical batch16 and full-Q/full-R GEMM geometry.

    Only batches containing one of the frozen C128 are recomputed here;
    every such batch retains all original neighboring gallery passages.
    Reference subsets are masked after GEMM, query subsets after reference max.
    """
    import torch
    from torch.nn import functional as F
    dtype = torch.float32 if backend == "GPU_FP32_DOT" else torch.float64
    device = torch.device("cuda" if backend == "GPU_FP32_DOT" else "cpu")
    q = query.to(dtype=dtype, device=device)
    if backend == "CPU_FP64_RENORM": q = F.normalize(q, dim=1)
    destination = {physical: i for i, physical in enumerate(axis)}
    score_outputs = {f"Q_{qg}__R_{rg}": torch.empty(128, dtype=torch.float64) for qg in groups for rg in R_GROUPS}
    local_outputs = {rg: torch.empty((128, len(query)), dtype=dtype) for rg in R_GROUPS}
    batch_layout = []
    with torch.inference_mode():
        for start in range(0, len(gallery), 16):
            physical_batch = list(range(start, min(start + 16, len(gallery))))
            selected = [physical for physical in physical_batch if physical in destination]
            if not selected: continue
            refs = [gallery[physical].to(dtype=dtype, device=device) for physical in physical_batch]
            lengths = torch.tensor([len(ref) for ref in refs], device=device)
            padded = torch.nn.utils.rnn.pad_sequence(refs, batch_first=True)
            if backend == "CPU_FP64_RENORM": padded = F.normalize(padded, dim=2)
            valid = torch.arange(padded.shape[1], device=device)[None] < lengths[:, None]
            similarity = torch.einsum("pd,bqd->pbq", q, padded)
            masks = {"FULL": valid, "IMAGE": valid.clone(), "TEMPLATE": valid.clone()}
            for physical in selected:
                offset = physical - start
                for rg in ("IMAGE", "TEMPLATE"):
                    masks[rg][offset].zero_(); masks[rg][offset, reference_indices[physical][rg]] = True
            batch_layout.append({"start": start, "physical_rows": physical_batch, "reference_lengths": lengths.cpu().tolist(),
                                 "padded_reference_count": padded.shape[1], "query_count": len(query)})
            for rg in R_GROUPS:
                local = similarity.masked_fill(~masks[rg][None], float("-inf")).amax(dim=-1)
                for qg, positions in groups.items():
                    # FULL deliberately uses the unchanged original sum(0).
                    selected_rows = local if qg == "FULL" else local[positions]
                    scores = selected_rows.sum(dim=0).cpu().to(torch.float64)
                    for physical in selected: score_outputs[f"Q_{qg}__R_{rg}"][destination[physical]] = scores[physical-start]
                for physical in selected: local_outputs[rg][destination[physical]] = local[:, physical-start].cpu()
    need(all(bool(torch.isfinite(value).all()) for value in (*score_outputs.values(), *local_outputs.values())), "NONFINITE_INTERFACE_READOUT")
    return score_outputs, local_outputs, batch_layout


def summarize(scores, axis, groups):
    output = {}
    for name, values in scores.items():
        order = sorted(range(128), key=lambda i: (-float(values[i]), axis[i]))
        qg = name.split("__R_")[0][2:]; count = len(groups[qg])
        target = axis.index(599); rival = max((i for i in range(128) if i != target), key=lambda i: (float(values[i]), -axis[i]))
        output[name] = {"query_rows": count, "scores_SUM_binary64": [hx(x) for x in values],
            "scores_MEAN_same_scale_binary64": [hx(x / count) for x in values],
            "ranked_physical_rows": [axis[i] for i in order], "target599_rank": order.index(target) + 1,
            "old_V_wrong594_rank": order.index(axis.index(594)) + 1, "top1_physical_row": axis[order[0]],
            "target_minus_strongest_rival_SUM_binary64": hx(values[target] - values[rival]),
            "strongest_rival_physical_row": axis[rival]}
    return output


def e0():
    import torch
    rng = torch.Generator().manual_seed(17)
    query = torch.randn((13, 128), generator=rng, dtype=torch.float16)
    gallery = [torch.randn((3 + i % 11, 128), generator=rng, dtype=torch.float16) for i in range(35)]
    actual = old_raw_function()(query, gallery, torch.device("cpu"), batch_size=16)
    expected = []
    for start in range(0, len(gallery), 16):
        refs = [x.float() for x in gallery[start:start+16]]
        padded = torch.nn.utils.rnn.pad_sequence(refs, batch_first=True)
        lengths = torch.tensor([len(x) for x in refs])
        valid = torch.arange(padded.shape[1])[None] < lengths[:, None]
        similarity = torch.einsum("pd,bqd->pbq", query.float(), padded).masked_fill(~valid[None], float("-inf"))
        expected.append(similarity.amax(-1).sum(0))
    need(tensor_sha(actual) == tensor_sha(torch.cat(expected).double()), "ORIGINAL_RAW_BATCH16_LITERAL_E0")
    # Exercise every fixed Q/R mask without an encoder or CUDA. Dyadic FP16
    # inputs make these small FP64 dot/sum oracles exactly representable.
    refs = [torch.randn((3 + i % 11, 128), generator=rng, dtype=torch.float16) for i in range(130)]
    axis = list(range(127)) + [129]
    indices = {physical: {"IMAGE": list(range(1, len(refs[physical])-1)), "TEMPLATE": [0, len(refs[physical])-1]} for physical in axis}
    groups = {"FULL": list(range(13)), "IMAGE": list(range(9)), "TEMPLATE": list(range(9, 13)), "H": [0, 2, 4], "H_PLUS_TEMPLATE": [0, 2, 4, 9, 10, 11, 12]}
    scores, local, layout = fixed_layout_readout(query, refs, axis, indices, groups, "CPU_FP64_DOT")
    for position, physical in enumerate(axis):
        similarity = query.double() @ refs[physical].double().T
        for rg in R_GROUPS:
            columns = list(range(len(refs[physical]))) if rg == "FULL" else indices[physical][rg]
            expected_local = similarity[:, columns].amax(dim=1)
            need(tensor_sha(local[rg][position]) == tensor_sha(expected_local), "SYNTHETIC_R_COLUMN_MASK")
            for qg, rows in groups.items():
                need(hx(scores[f"Q_{qg}__R_{rg}"][position]) == hx(expected_local[rows].sum()), "SYNTHETIC_Q_ROW_SUM")
    need(layout[-1]["physical_rows"] == [128, 129] and layout[-1]["reference_lengths"] == [len(refs[128]), len(refs[129])], "ORIGINAL_PHYSICAL_BATCH_NEIGHBOR_AND_TAIL")
    return {"status": "RC_DIFFICULT0006_RAW_INTERFACE_BRIDGE_E0_PASS", "original_full_gallery_AST_batch16_exact": True,
        "all5_Q_groups_and_all3_R_groups_literal_oracle": True, "physical_batch_neighbors_and_short_tail_preserved": True,
        "all_per_token_max_vectors_retained": True,
        "synthetic_gallery_count": 35, "encoder_load_count": 0, "encoder_forward_count": 0, "training_updates": 0}


def freeze():
    need(not AUTH.exists() and not OUT.exists(), "APPEND_ONLY_AUTHORITY_OR_OUTPUT_EXISTS")
    value = {"status": "RC_DIFFICULT0006_RAW_INTERFACE_BRIDGE_AUTHORIZED", "sources": sources(), "contract": CONTRACT,
             "e0": e0(), "scientific_GO_or_NO_GO": None}
    save(AUTH, value); print(json.dumps({"status": value["status"], "authority_sha256": sha(AUTH)}), flush=True)


def authority():
    value = read(AUTH)
    need(value["status"] == "RC_DIFFICULT0006_RAW_INTERFACE_BRIDGE_AUTHORIZED" and value["sources"] == sources()
         and value["contract"] == CONTRACT and value["e0"]["original_full_gallery_AST_batch16_exact"] is True, "FROZEN_AUTHORITY_DRIFT")
    return value


def execute(validate=False):
    import numpy as np
    import torch
    authority(); queries, gallery, axis, reference_indices, expected = load_inputs()
    need(torch.cuda.is_available(), "GPU_REQUIRED_ONLY_FOR_ORIGINAL_FP32_EINSUM")
    need(OUT.exists() if validate else not OUT.exists(), "APPEND_ONLY_OUTPUT_STATE")
    if not validate: OUT.mkdir(parents=True)
    original_scores, gate = original_RAW_gate(queries["ORIGINAL"]["full"], gallery, axis, expected, torch.device("cuda"))
    if validate: need(read(OUT / "original_RAW_gate.json") == gate, "INDEPENDENT_ORIGINAL_RAW_GATE_REPLAY")
    else: save(OUT / "original_RAW_gate.json", gate)
    if not gate["passed"]:
        if not validate:
            save(OUT / "result.json", {"status": "RC_DIFFICULT0006_ORIGINAL_RAW_SCORE_BIT_MISMATCH_STOP", "authority_sha256": sha(AUTH),
                "gate": binding(OUT / "original_RAW_gate.json"), "other_control_scoring_count": 0,
                "scientific_interpretation_authorized": False, "encoder_load_count": 0, "encoder_forward_count": 0,
                "scientific_GO_or_NO_GO": None, "HYP_GO_claimed": False})
        print(json.dumps({"status": "ORIGINAL_RAW_BIT_GATE_STOP", "gate": gate}), flush=True); return 4
    records = []
    for mode in MODES:
        query = queries[mode]; direct_raw = original_scores if mode == "ORIGINAL" else old_raw_function()(query["full"], gallery, torch.device("cuda"), batch_size=16)[axis]
        arrays = {}; descriptions = {}; layouts = None
        for backend in BACKENDS:
            scores, locals_, layout = fixed_layout_readout(query["full"], gallery, axis, reference_indices, query["groups"], backend)
            if layouts is None: layouts = layout
            else: need(layout == layouts, "NUMERIC_PATHS_MUST_KEEP_IDENTICAL_BATCH_GEOMETRY")
            if backend == "GPU_FP32_DOT": need(tensor_sha(scores["Q_FULL__R_FULL"]) == tensor_sha(direct_raw), "FULL_LAYOUT_FP32_VS_OLD_AST_BIT_DRIFT")
            arrays.update({backend + "__SCORE__" + name: value.numpy() for name, value in scores.items()})
            arrays.update({backend + "__LOCAL__R_" + name: value.numpy() for name, value in locals_.items()})
            descriptions[backend] = summarize(scores, axis, query["groups"])
        # Compare the renormalized image-only matrix with the old CPU V cache;
        # full-Q/full-R GEMM geometry can have last-bit arithmetic differences.
        current_a = torch.from_numpy(arrays["CPU_FP64_RENORM__LOCAL__R_IMAGE"])[:, :query["image_count"]]
        old_a = query["old_image_only_a"]
        compatibility = {"exact_a_bits": tensor_sha(current_a) == tensor_sha(old_a),
            "max_absolute_a_difference": float((current_a - old_a).abs().max()),
            "old_current_V_same_H_target599_rank": read(PARENT / "result.json")["records"][MODES.index(mode)]["readout"]["same_H_physical_domain"]["target599_rank"]}
        path = OUT / (mode + ".npz")
        if validate:
            with np.load(path, allow_pickle=False) as saved:
                need(set(saved.files) == set(arrays), "SAVED_CONTROL_ARRAY_KEYS")
                for name, values in arrays.items(): need(saved[name].dtype == values.dtype and saved[name].shape == values.shape and saved[name].tobytes() == values.tobytes(), "FULL_INTERFACE_ARRAY_REPLAY:" + name)
            array_binding = binding(path)
        else: array_binding = save_npz(path, arrays)
        record = {"mode": mode, "query_source": query["source"], "query_full_tokens_sha256": tensor_sha(query["full"]),
            "query_image_count": query["image_count"], "query_template_count": query["template_count"],
            "Q_group_indices": query["groups"], "original_physical_batch16_layout": layouts,
            "arrays": array_binding, "readouts": descriptions, "old_V_renorm_image_matrix_comparison": compatibility}
        records.append(record)
        print(json.dumps({"event": "FIXED_RAW_INTERFACE_CONDITION_COMPLETE", "mode": mode,
            "ranks": {backend: {name: row["target599_rank"] for name, row in values.items()} for backend, values in descriptions.items()}}), flush=True)
    result = {"status": "RC_DIFFICULT0006_RAW_INTERFACE_BRIDGE_COMPLETE", "authority_sha256": sha(AUTH),
        "original_RAW_gate": binding(OUT / "original_RAW_gate.json"), "original_C128_RAW_score_bit_exact": True,
        "candidate_physical_rows": axis, "records": records, "only_original_C128_scores_exposed": True,
        "scope": "ONE_OPENED_QUERY_THREE_ALREADY_ENCODED_INPUTS_FIXED_RAW_INTERFACE_FACTOR_DECOMPOSITION",
        "limits": ["Original full-query RAW was already correct; repeating that result is a regression check, not HYP success.",
            "Q and R template ablations share the same full similarity matrix and original physical batch16 padding geometry.",
            "GPU_FP32_DOT versus CPU_FP64_DOT changes precision and backend together; only CPU_FP64_DOT versus CPU_FP64_RENORM isolates renormalization.",
            "SUM and MEAN on the same Q set differ by one candidate-independent positive scale, not a separate ranking mechanism.",
            "ORIGINAL H+template includes full-image contextual template features, so it is not region-only evidence.",
            "KEEP and CROP remain one opened diagnostic, and no settings or checkpoints are selected."],
        "encoder_load_count": 0, "encoder_forward_count": 0, "RoMa_forward_count": 0, "training_updates": 0,
        "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
    if validate:
        need(read(OUT / "result.json") == result, "INDEPENDENT_ALL_INTERFACE_RANK_AND_SCORE_REPLAY")
        receipt = {"status": "RC_DIFFICULT0006_RAW_INTERFACE_BRIDGE_INDEPENDENT_VALIDATION_PASS", "authority_sha256": sha(AUTH),
            "result_sha256": sha(OUT / "result.json"), "checks": {"original_RAW_C128_scores_bit_exact": True,
                "original_5413_gallery_batch16_function_replayed_for_all3_inputs": True,
                "all_Q_R_numeric_interfaces_and_full_C128_arrays_recomputed_exactly": True,
                "no_encoder_or_RoMa_loaded_or_run": True}, "training_updates": 0, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None}
        save(OUT / "independent_validation.json", receipt); print(json.dumps(receipt), flush=True)
    else: save(OUT / "result.json", result); print(json.dumps({"status": result["status"], "result_sha256": sha(OUT / "result.json")}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", choices=("e0", "freeze", "run", "validate"), required=True)
    phase = parser.parse_args().phase
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    if phase == "e0": print(json.dumps(e0(), sort_keys=True)); return 0
    if phase == "freeze": freeze(); return 0
    return execute(validate=phase == "validate")


if __name__ == "__main__": raise SystemExit(main())
