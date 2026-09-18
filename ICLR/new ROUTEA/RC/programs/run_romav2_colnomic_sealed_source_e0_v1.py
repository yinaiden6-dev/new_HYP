#!/usr/bin/env python3
"""Replay one opened query through the exact sealed-source ColNomic path."""

import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from PIL import Image
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

from rc_aslo_xf.colnomic_proposal_tokens import (  # noqa: E402
    proposal_grid_sha256,
    split_cached_spatial_tokens,
)
from rc_aslo_xf.conditional_rep_sources import build_gallery_source  # noqa: E402
from rc_aslo_xf.rgh_full600_source_v1 import RGHFull600SourceLoaderV1  # noqa: E402
from rc_aslo_xf.rgh_v9_frozen_colnomic_base_v1 import FrozenColNomicBaseV1  # noqa: E402


MODEL = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b"
GALLERY_CACHE = (
    WORKSPACE / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
)
GEOMETRY = ROOT / "cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt"
FOLDS = ROOT / "protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json"
MANIFEST_RECEIPT = ROOT / "results/romav2_colnomic_new_difficult_sealed_manifest_v1/binding_receipt.json"
OUT = ROOT / "results/romav2_colnomic_sealed_source_e0_v1/result.json"
OPENED_EXECUTION = 20


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def logical(value: dict) -> str:
    projection = dict(value)
    projection.pop("logical_sha256", None)
    return hashlib.sha256(
        json.dumps(
            projection, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable E0 output exists: {OUT}")
    receipt = json.loads(MANIFEST_RECEIPT.read_text())
    assert receipt["status"] == "NEW_DIFFICULT_SEALED_MANIFEST_BINDING_PASS"
    assert receipt["access"]["sealed_model_scoring_count"] == 0
    assert receipt["access"]["sealed_pixel_decode_count"] == 0

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from transformers.utils.import_utils import is_flash_attn_2_available

    loader = RGHFull600SourceLoaderV1(
        rc_root=ROOT,
        geometry_payload_path=GEOMETRY,
        prejoin_schedule_path=FOLDS,
    )
    source = loader.load_execution(OPENED_EXECUTION)
    base = FrozenColNomicBaseV1(ROOT)
    gallery_source = build_gallery_source(verify_cache_file_sha256=True)
    payload = torch.load(
        GALLERY_CACHE, map_location="cpu", weights_only=False, mmap=True
    )
    passages = payload["passage_emb"]
    assert len(passages) == 5413
    assert tuple(map(str, payload["setids"])) == gallery_source.legacy_setids

    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    model = ColQwen2_5.from_pretrained(
        str(MODEL),
        torch_dtype=torch.bfloat16,
        attn_implementation=attention,
        local_files_only=True,
    ).cuda().eval()
    processor = ColQwen2_5_Processor.from_pretrained(
        str(MODEL), local_files_only=True
    )
    with Image.open(source.query.source_path) as image:
        inputs = processor.process_images([image.convert("RGB")]).to("cuda")
    with torch.inference_mode():
        embedding = model(**inputs)
    valid = embedding[0][inputs["attention_mask"][0].bool()].detach().to(torch.float16).cpu().contiguous()
    grid = split_cached_spatial_tokens(
        valid,
        input_ids=inputs["input_ids"].detach().cpu(),
        attention_mask=inputs["attention_mask"].detach().cpu(),
        image_grid_thw=inputs["image_grid_thw"].detach().cpu(),
        image_token_id=int(processor.image_token_id),
        merge_size=int(processor.image_processor.merge_size),
    )

    expected_tokens = source.query.tokens.detach().cpu().to(torch.float32)
    replay_tokens = grid.tokens.to(torch.float32)
    shape_exact = tuple(replay_tokens.shape) == tuple(expected_tokens.shape)
    if shape_exact:
        cosine = F.cosine_similarity(replay_tokens, expected_tokens, dim=1)
        min_cosine = float(cosine.min())
        mean_cosine = float(cosine.mean())
        max_abs = float((replay_tokens - expected_tokens).abs().max())
    else:
        min_cosine = mean_cosine = max_abs = float("nan")

    with torch.inference_mode():
        raw_all = processor.score_multi_vector(
            [valid], passages, batch_size=32, device=torch.device("cuda")
        )[0].detach().to(torch.float64).cpu()
    best_by_identity = {}
    for physical_row, identity in enumerate(gallery_source.corrected_identities):
        score = float(raw_all[physical_row])
        previous = best_by_identity.get(identity)
        if previous is None or score > previous[0] or (
            score == previous[0] and physical_row < previous[1]
        ):
            best_by_identity[identity] = (score, physical_row)
    ranked = sorted(best_by_identity.values(), key=lambda item: (-item[0], item[1]))
    observed_ranked_rows = [row for _, row in ranked[:128]]
    observed_axis = sorted(observed_ranked_rows)
    expected_axis = [int(item.physical_row) for item in source.candidates]
    axis_exact = observed_axis == expected_axis
    expected_scores = base.scores(OPENED_EXECUTION, expected_axis)
    observed_scores = torch.tensor(
        [float(raw_all[row]) for row in expected_axis], dtype=torch.float64
    )
    score_max_abs = float((observed_scores - expected_scores).abs().max())
    score_order_exact = torch.argsort(
        observed_scores, descending=True, stable=True
    ).tolist() == torch.argsort(expected_scores, descending=True, stable=True).tolist()

    checks = {
        "opened_query_only": source.query_id == "OUTCOME-0031",
        "query_token_shape_exact": shape_exact,
        "query_token_min_cosine_ge_0_999": min_cosine >= 0.999,
        "query_token_mean_cosine_ge_0_9999": mean_cosine >= 0.9999,
        "natural_c128_axis_exact": axis_exact,
        "natural_c128_score_order_exact": score_order_exact,
        "natural_c128_score_max_abs_le_1e_3": score_max_abs <= 1e-3,
        "sealed_pixel_decode_zero": True,
        "sealed_model_scoring_zero": True,
        "target_join_zero": True,
    }
    passed = all(checks.values())
    value = {
        "schema_version": "rc_romav2_colnomic_sealed_source_e0_v1_20260831",
        "status": (
            "ROMAV2_COLNOMIC_SEALED_SOURCE_E0_READY"
            if passed
            else "ROMAV2_COLNOMIC_SEALED_SOURCE_E0_ABORT"
        ),
        "claim_level": "OPENED_REPLAY_ENGINEERING_ONLY",
        "opened_execution": OPENED_EXECUTION,
        "opened_query_id": source.query_id,
        "checks": checks,
        "diagnostics": {
            "attention_implementation": attention,
            "query_grid_shape": list(grid.grid_shape),
            "query_grid_sha256": proposal_grid_sha256(grid),
            "query_token_min_cosine": min_cosine,
            "query_token_mean_cosine": mean_cosine,
            "query_token_max_abs": max_abs,
            "c128_axis_match_count": sum(a == b for a, b in zip(observed_axis, expected_axis)),
            "c128_score_max_abs": score_max_abs,
        },
        "bindings": {
            "manifest_receipt_sha256": sha(MANIFEST_RECEIPT),
            "gallery_cache_sha256": sha(GALLERY_CACHE),
            "model_path": str(MODEL),
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
    value["logical_sha256"] = logical(value)
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "checks": checks, "diagnostics": value["diagnostics"]}, sort_keys=True))
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
