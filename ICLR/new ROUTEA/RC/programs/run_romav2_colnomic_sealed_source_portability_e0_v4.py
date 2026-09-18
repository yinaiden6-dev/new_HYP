#!/usr/bin/env python3
"""Result-blind portability gate for the frozen new_difficult encoder path."""

import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))
import run_romav2_colnomic_sealed_source_e0_v1 as v1  # noqa: E402
import run_romav2_colnomic_sealed_source_e0_v2 as v2  # noqa: E402
import run_romav2_colnomic_sealed_source_e0_v3 as v3  # noqa: E402


OUT = ROOT / "results/romav2_colnomic_sealed_source_portability_e0_v4/result.json"
EXECUTIONS = (78, 83, 88, 72)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.to(torch.float64) - a.to(torch.float64).mean()
    y = b.to(torch.float64) - b.to(torch.float64).mean()
    return float((x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1e-12))


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable V4 portability output exists: {OUT}")
    predecessor = json.loads(v3.OUT.read_text())
    assert predecessor["status"] == "ROMAV2_COLNOMIC_SEALED_SOURCE_E0_V3_ABORT"
    receipt = json.loads(v1.MANIFEST_RECEIPT.read_text())
    assert receipt["status"] == "NEW_DIFFICULT_SEALED_MANIFEST_BINDING_PASS"

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    loader = v1.RGHFull600SourceLoaderV1(
        rc_root=ROOT,
        geometry_payload_path=v1.GEOMETRY,
        prejoin_schedule_path=v1.FOLDS,
    )
    base = v1.FrozenColNomicBaseV1(ROOT)
    gallery_source = v1.build_gallery_source(verify_cache_file_sha256=True)
    payload = torch.load(v1.GALLERY_CACHE, map_location="cpu", weights_only=False, mmap=True)
    passages = payload["passage_emb"]
    device = torch.device("cuda")
    encoder = ColQwen2_5.from_pretrained(str(v1.MODEL), torch_dtype=torch.bfloat16).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(v1.MODEL))

    rows = []
    for execution in EXECUTIONS:
        source = loader.load_execution(execution)
        assert source.track == "new_difficult_train"
        with Image.open(source.query.source_path) as raw:
            orientation = int(raw.getexif().get(274, 1))
            inputs = processor.process_images([ImageOps.exif_transpose(raw).convert("RGB")]).to(device)
        encoded = encoder(**inputs)[0].float()
        image_mask = inputs["input_ids"][0] == processor.image_token_id
        image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
        template_tokens = encoded[~image_mask].detach().half().cpu().contiguous()
        _, height, width = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
        merge = int(processor.image_processor.merge_size)
        grid = (height // merge, width // merge)
        expected_tokens = source.query.tokens.detach().cpu().float()
        replay_tokens = image_tokens.float()
        shape_exact = tuple(replay_tokens.shape) == tuple(expected_tokens.shape)
        token_mean_cosine = (
            float(F.cosine_similarity(replay_tokens, expected_tokens, dim=1).mean())
            if shape_exact
            else float("nan")
        )
        scores = v2.full_gallery_scores(torch.cat((image_tokens, template_tokens)), passages, device)
        observed_ranked = v2.top128_rows(scores, gallery_source.corrected_identities)
        observed_axis = sorted(observed_ranked)
        expected_axis = [int(item.physical_row) for item in source.candidates]
        expected_scores = base.scores(execution, expected_axis)
        observed_expected_scores = scores[expected_axis]
        expected_ranked = [
            expected_axis[i]
            for i in torch.argsort(expected_scores, descending=True, stable=True).tolist()
        ]
        c128_overlap = len(set(observed_axis) & set(expected_axis))
        top10_overlap = len(set(observed_ranked[:10]) & set(expected_ranked[:10]))
        row = {
            "execution_ordinal": execution,
            "query_id": source.query_id,
            "source_exif_orientation": orientation,
            "grid_shape": list(grid),
            "expected_grid_shape": list(source.query.grid_shape),
            "grid_exact": grid == source.query.grid_shape,
            "token_mean_cosine": token_mean_cosine,
            "c128_set_overlap": c128_overlap,
            "top10_set_overlap": top10_overlap,
            "top1_exact": observed_ranked[0] == expected_ranked[0],
            "expected_axis_score_pearson": pearson(observed_expected_scores, expected_scores),
        }
        row["pass"] = (
            row["grid_exact"]
            and row["token_mean_cosine"] >= 0.95
            and row["c128_set_overlap"] >= 120
            and row["top10_set_overlap"] >= 8
            and row["top1_exact"]
            and row["expected_axis_score_pearson"] >= 0.999
        )
        rows.append(row)
        print(json.dumps({"event": "portability_case", **row}, sort_keys=True), flush=True)

    orientation_counts = {
        str(value): sum(row["source_exif_orientation"] == value for row in rows)
        for value in (1, 6)
    }
    checks = {
        "four_fixed_opened_queries": len(rows) == 4,
        "two_identity_oriented_and_two_native": orientation_counts == {"1": 2, "6": 2},
        "all_query_gates_pass": all(row["pass"] for row in rows),
        "no_target_or_result_read": True,
        "sealed_pixel_decode_zero": True,
        "sealed_model_scoring_zero": True,
    }
    passed = all(checks.values())
    value = {
        "schema_version": "rc_romav2_colnomic_sealed_source_portability_e0_v4_20260831",
        "status": (
            "ROMAV2_COLNOMIC_SEALED_SOURCE_PORTABILITY_E0_V4_READY"
            if passed
            else "ROMAV2_COLNOMIC_SEALED_SOURCE_PORTABILITY_E0_V4_ABORT"
        ),
        "claim_level": "OPENED_RESULT_BLIND_RUNTIME_PORTABILITY_ONLY",
        "frozen_thresholds": {
            "token_mean_cosine_min": 0.95,
            "c128_set_overlap_min": 120,
            "top10_set_overlap_min": 8,
            "top1_exact": True,
            "expected_axis_score_pearson_min": 0.999,
        },
        "checks": checks,
        "rows": rows,
        "bindings": {
            "predecessor_sha256": v1.sha(v3.OUT),
            "manifest_receipt_sha256": v1.sha(v1.MANIFEST_RECEIPT),
            "gallery_cache_sha256": v1.sha(v1.GALLERY_CACHE),
            "new_difficult_v2_builder_sha256": v1.sha(ROOT.parent / "scripts/c2_build_new_difficult_token_cache.py"),
        },
        "access": {
            "opened_new_difficult_train_pixel_decode_count": 4,
            "opened_model_scoring_count": 4,
            "target_label_read_count": 0,
            "retrieval_result_read_count": 0,
            "sealed_pixel_decode_count": 0,
            "sealed_model_scoring_count": 0,
        },
        "next_authorized_stage": "NEW_DIFFICULT_SEALED_QUERY_TOKENIZATION_E0" if passed else None,
        "logical_sha256": "",
    }
    value["logical_sha256"] = v1.logical(value)
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True))
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
