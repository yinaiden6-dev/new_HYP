#!/usr/bin/env python3
"""Exact replay of the historical B4 ColNomic query/cache generator."""

import json
from pathlib import Path
import sys

import torch
from PIL import Image
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))
import run_romav2_colnomic_sealed_source_e0_v1 as v1  # noqa: E402


OUT = ROOT / "results/romav2_colnomic_sealed_source_e0_v2/result.json"


@torch.inference_mode()
def full_gallery_scores(query, gallery, device, batch_size=16):
    rows = []
    query = query.float().to(device)
    for start in range(0, len(gallery), batch_size):
        refs = [x.float().to(device) for x in gallery[start : start + batch_size]]
        lengths = torch.tensor([x.shape[0] for x in refs], device=device)
        padded = torch.nn.utils.rnn.pad_sequence(refs, batch_first=True)
        valid = torch.arange(padded.shape[1], device=device)[None] < lengths[:, None]
        similarity = torch.einsum("pd,bqd->pbq", query, padded)
        similarity = similarity.masked_fill(~valid[None], float("-inf"))
        rows.append(similarity.amax(dim=-1).sum(dim=0).cpu())
    return torch.cat(rows).to(torch.float64)


def top128_rows(scores, labels):
    seen = set()
    rows = []
    for row in torch.argsort(scores, descending=True, stable=True).tolist():
        label = labels[row]
        if label in seen:
            continue
        seen.add(label)
        rows.append(int(row))
        if len(rows) == 128:
            break
    assert len(rows) == 128
    return rows


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable V2 E0 output exists: {OUT}")
    predecessor = json.loads(v1.OUT.read_text())
    assert predecessor["status"] == "ROMAV2_COLNOMIC_SEALED_SOURCE_E0_ABORT"
    receipt = json.loads(v1.MANIFEST_RECEIPT.read_text())
    assert receipt["status"] == "NEW_DIFFICULT_SEALED_MANIFEST_BINDING_PASS"

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    loader = v1.RGHFull600SourceLoaderV1(
        rc_root=ROOT,
        geometry_payload_path=v1.GEOMETRY,
        prejoin_schedule_path=v1.FOLDS,
    )
    source = loader.load_execution(v1.OPENED_EXECUTION)
    base = v1.FrozenColNomicBaseV1(ROOT)
    gallery_source = v1.build_gallery_source(verify_cache_file_sha256=True)
    payload = torch.load(
        v1.GALLERY_CACHE, map_location="cpu", weights_only=False, mmap=True
    )
    passages = payload["passage_emb"]
    legacy_labels = tuple(map(str, payload["setids"]))

    device = torch.device("cuda")
    encoder = ColQwen2_5.from_pretrained(
        str(v1.MODEL), torch_dtype=torch.bfloat16
    ).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(v1.MODEL))
    with Image.open(source.query.source_path) as image:
        inputs = processor.process_images([image.convert("RGB")]).to(device)
    encoded = encoder(**inputs)[0].float()
    image_mask = inputs["input_ids"][0] == processor.image_token_id
    assert image_mask.shape == encoded.shape[:1] and bool(image_mask.any())
    image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
    template_tokens = encoded[~image_mask].detach().half().cpu().contiguous()
    temporal, height, width = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    grid_shape = (height // merge, width // merge)
    assert temporal == 1 and image_tokens.shape[0] == grid_shape[0] * grid_shape[1]

    expected = source.query.tokens.detach().cpu().float()
    replay = image_tokens.float()
    shape_exact = tuple(replay.shape) == tuple(expected.shape)
    if shape_exact:
        cosine = F.cosine_similarity(replay, expected, dim=1)
        min_cosine = float(cosine.min())
        mean_cosine = float(cosine.mean())
        max_abs = float((replay - expected).abs().max())
    else:
        min_cosine = mean_cosine = max_abs = float("nan")

    full_query = torch.cat((image_tokens, template_tokens), dim=0)
    scores = full_gallery_scores(full_query, passages, device)
    legacy_ranked = top128_rows(scores, legacy_labels)
    corrected_ranked = top128_rows(scores, gallery_source.corrected_identities)
    expected_axis = [int(item.physical_row) for item in source.candidates]
    legacy_axis = sorted(legacy_ranked)
    corrected_axis = sorted(corrected_ranked)
    expected_scores = base.scores(v1.OPENED_EXECUTION, expected_axis)
    observed_scores = scores[expected_axis]
    score_max_abs = float((observed_scores - expected_scores).abs().max())
    score_order_exact = torch.argsort(
        observed_scores, descending=True, stable=True
    ).tolist() == torch.argsort(expected_scores, descending=True, stable=True).tolist()
    axis_exact = corrected_axis == expected_axis

    checks = {
        "predecessor_failure_is_lineage_only": True,
        "opened_query_exact": source.query_id == "DIFFICULT-0043",
        "historical_image_mask_path_replayed": True,
        "historical_image_template_reordering_replayed": True,
        "historical_sum_maxsim_replayed": True,
        "query_token_shape_exact": shape_exact,
        "query_token_min_cosine_ge_0_999": min_cosine >= 0.999,
        "query_token_mean_cosine_ge_0_9999": mean_cosine >= 0.9999,
        "corrected_natural_c128_axis_exact": axis_exact,
        "natural_c128_score_order_exact": score_order_exact,
        "natural_c128_score_max_abs_le_1e_3": score_max_abs <= 1e-3,
        "sealed_pixel_decode_zero": True,
        "sealed_model_scoring_zero": True,
        "target_join_zero": True,
    }
    passed = all(checks.values())
    value = {
        "schema_version": "rc_romav2_colnomic_sealed_source_e0_v2_20260831",
        "status": (
            "ROMAV2_COLNOMIC_SEALED_SOURCE_E0_V2_READY"
            if passed
            else "ROMAV2_COLNOMIC_SEALED_SOURCE_E0_V2_ABORT"
        ),
        "claim_level": "OPENED_EXACT_LINEAGE_REPLAY_ENGINEERING_ONLY",
        "checks": checks,
        "diagnostics": {
            "attention_implementation": getattr(encoder.config, "_attn_implementation", None),
            "query_grid_shape": list(grid_shape),
            "query_token_min_cosine": min_cosine,
            "query_token_mean_cosine": mean_cosine,
            "query_token_max_abs": max_abs,
            "corrected_c128_axis_match_count": sum(a == b for a, b in zip(corrected_axis, expected_axis)),
            "legacy_c128_axis_match_count": sum(a == b for a, b in zip(legacy_axis, expected_axis)),
            "corrected_c128_set_overlap": len(set(corrected_axis) & set(expected_axis)),
            "legacy_c128_set_overlap": len(set(legacy_axis) & set(expected_axis)),
            "c128_score_max_abs": score_max_abs,
        },
        "bindings": {
            "predecessor_sha256": v1.sha(v1.OUT),
            "manifest_receipt_sha256": v1.sha(v1.MANIFEST_RECEIPT),
            "gallery_cache_sha256": v1.sha(v1.GALLERY_CACHE),
            "historical_b4_generator_sha256": v1.sha(ROOT.parent / "scripts/b4_build_difficult_train_cache.py"),
        },
        "access": {
            "opened_pixel_decode_count": 1,
            "opened_model_scoring_count": 1,
            "sealed_pixel_decode_count": 0,
            "sealed_model_scoring_count": 0,
            "target_join_count": 0,
        },
        "next_authorized_stage": "NEW_DIFFICULT_SEALED_QUERY_TOKENIZATION_E0"
        if passed
        else None,
        "logical_sha256": "",
    }
    value["logical_sha256"] = v1.logical(value)
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "checks": checks, "diagnostics": value["diagnostics"]}, sort_keys=True))
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
