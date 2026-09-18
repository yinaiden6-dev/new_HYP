#!/usr/bin/env python3
"""Materialize current-runtime target-free query/C128/reference tokens."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))
import run_romav2_colnomic_sealed_source_e0_v1 as v1  # noqa: E402
import run_romav2_colnomic_sealed_source_e0_v2 as v2  # noqa: E402
from rc_aslo_xf.cw1_sr0_s8_feature_runtime_v1 import HybridSpatialReferenceResolver  # noqa: E402


TRAIN = ROOT / "results/romav2_colnomic_visibility_xf_full_c128_prejoin_v1"
EVAL = ROOT / "results/romav2_colnomic_visibility_xf_fullnegative_eval_prejoin_v1"
PORTABILITY = ROOT / "results/romav2_colnomic_sealed_source_portability_e0_v5/result.json"
CONTRACT = ROOT / "plan/ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_V1_20260831.md"
OUTROOT = ROOT / "results/romav2_colnomic_current_runtime_bridge_prejoin_v1"
SHARD_COUNT = 8


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    x = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def selections():
    train_exec = []
    train_hashes = []
    for shard in range(4):
        path = TRAIN / f"shard{shard:02d}/result.json"
        doc = json.loads(path.read_text())
        train_hashes.append(sha(path))
        train_exec.extend(int(row["execution_ordinal"]) for row in doc["rows"])
    eval_exec = []
    eval_hashes = []
    for shard in range(4):
        path = EVAL / f"shard{shard:02d}/result.json"
        doc = json.loads(path.read_text())
        eval_hashes.append(sha(path))
        eval_exec.extend(int(row["execution_ordinal"]) for row in doc["rows"])
    assert len(train_exec) == len(eval_exec) == 32
    assert len(set(train_exec)) == len(set(eval_exec)) == 32
    assert not (set(train_exec) & set(eval_exec))
    return [("TRAIN", x) for x in train_exec] + [("EVAL", x) for x in eval_exec], train_hashes, eval_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shard not in range(SHARD_COUNT):
        raise ValueError("shard outside 0..7")
    out = OUTROOT / f"shard{args.shard:02d}/payload.pt"
    receipt_path = OUTROOT / f"shard{args.shard:02d}/receipt.json"
    if out.exists() or receipt_path.exists():
        raise RuntimeError("immutable runtime-bridge shard exists")
    portability = json.loads(PORTABILITY.read_text())
    assert portability["status"] == "ROMAV2_COLNOMIC_SEALED_SOURCE_PORTABILITY_E0_V5_ABORT"
    assert portability["access"]["sealed_model_scoring_count"] == 0
    selection, train_hashes, eval_hashes = selections()
    work = [(role, execution) for index, (role, execution) in enumerate(selection) if index % SHARD_COUNT == args.shard]
    assert len(work) == 8

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    loader = v1.RGHFull600SourceLoaderV1(
        rc_root=ROOT, geometry_payload_path=v1.GEOMETRY, prejoin_schedule_path=v1.FOLDS
    )
    gallery_source = v1.build_gallery_source(verify_cache_file_sha256=True)
    gallery_payload = torch.load(v1.GALLERY_CACHE, map_location="cpu", weights_only=False, mmap=True)
    passages = gallery_payload["passage_emb"]
    labels = gallery_source.corrected_identities
    device = torch.device("cuda")
    encoder = ColQwen2_5.from_pretrained(str(v1.MODEL), torch_dtype=torch.bfloat16).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(v1.MODEL))
    resolver = HybridSpatialReferenceResolver(
        gallery_source=gallery_source,
        processor_factory=lambda: processor,
        gallery_embedding_loader=lambda: passages,
    )

    records = []
    reference_rows = set()
    for role, execution in work:
        source = loader.load_execution(execution)
        with Image.open(source.query.source_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB") if source.track == "new_difficult_train" else raw.convert("RGB")
            inputs = processor.process_images([image]).to(device)
        encoded = encoder(**inputs)[0].float()
        image_mask = inputs["input_ids"][0] == processor.image_token_id
        image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
        template_tokens = encoded[~image_mask].detach().half().cpu().contiguous()
        _, height, width = [int(x) for x in inputs["image_grid_thw"][0].tolist()]
        merge = int(processor.image_processor.merge_size)
        grid = (height // merge, width // merge)
        scores = v2.full_gallery_scores(torch.cat((image_tokens, template_tokens)), passages, device)
        ranked = v2.top128_rows(scores, labels)
        axis = sorted(ranked)
        reference_rows.update(axis)
        records.append(
            {
                "role": role,
                "execution_ordinal": execution,
                "query_id": source.query_id,
                "track": source.track,
                "query_source_path": source.query.source_path,
                "query_source_sha256": source.query.source_image_sha256,
                "query_grid_shape": grid,
                "query_tokens": image_tokens,
                "query_tokens_sha256": tensor_sha(image_tokens),
                "candidate_physical_rows": axis,
                "candidate_raw_scores": torch.tensor([float(scores[row]) for row in axis], dtype=torch.float64),
                "candidate_ranked_physical_rows": ranked,
                "target_or_label_read_count": 0,
            }
        )
        print(json.dumps({"event": "query_ready", "shard": args.shard, "role": role, "execution": execution, "candidate_count": len(axis)}, sort_keys=True), flush=True)

    references = {}
    for index, row in enumerate(sorted(reference_rows)):
        resolved = resolver.resolve(row)
        references[row] = {
            "physical_row": row,
            "source_path": str(gallery_source.raw_paths[row]),
            "grid_shape": resolved.grid_shape,
            "tokens": resolved.tokens,
            "tokens_sha256": resolved.tokens_sha256,
            "source_kind": resolved.source_kind,
            "source_logical_sha256": resolved.source_logical_sha256,
        }
        if (index + 1) % 128 == 0 or index + 1 == len(reference_rows):
            print(json.dumps({"event": "references_ready", "shard": args.shard, "done": index + 1, "total": len(reference_rows)}, sort_keys=True), flush=True)

    value = {
        "schema_version": "rc_romav2_colnomic_current_runtime_bridge_prejoin_shard_v1_20260831",
        "status": "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_SHARD_READY",
        "shard": args.shard,
        "shard_count": SHARD_COUNT,
        "records": records,
        "references": references,
        "bindings": {
            "contract_sha256": sha(CONTRACT),
            "train_prejoin_sha256": train_hashes,
            "eval_prejoin_sha256": eval_hashes,
            "portability_sha256": sha(PORTABILITY),
            "gallery_cache_sha256": sha(v1.GALLERY_CACHE),
        },
        "access": {
            "target_label_read_count": 0,
            "retrieval_result_read_count": 0,
            "sealed_pixel_decode_count": 0,
            "sealed_model_scoring_count": 0,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=False)
    torch.save(value, out)
    receipt = {
        "schema_version": "rc_romav2_colnomic_current_runtime_bridge_prejoin_receipt_v1_20260831",
        "status": "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_SHARD_READY",
        "shard": args.shard,
        "query_count": len(records),
        "reference_count": len(references),
        "payload_sha256": sha(out),
        "target_label_read_count": 0,
        "sealed_model_scoring_count": 0,
        "next_authorized_stage": "CURRENT_RUNTIME_BRIDGE_ROMA_VISIBILITY_MATERIALIZATION",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
