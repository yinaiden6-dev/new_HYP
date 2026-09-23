# One-off diagnostic: re-run text extraction + scoring for a handful of queries where
# MonoQwen2-VL reranking made the rank worse than ColNomic's original order, to see
# whether the cause is bad OCR text or the reranker itself failing to discriminate.

import argparse
import json
import os

import torch

from rerank_monoqwen import (
    build_gallery_setid_to_path,
    clean_query_text,
    load_image,
    load_reranker,
    score_candidates,
    vlm_read_text,
)

DEFAULT_PREDS_JSONL = "/hkfs/work/workspace/scratch/ap7811-benchmark/colnomic/result/raw_gallery/run_colnomic_retrieval_detail_raw_gallery_name_preds_image_only_7b.jsonl"
DEFAULT_RERANK_JSONL = "/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank/result/rerank_preds_image_only_7b.jsonl"
DEFAULT_GALLERY_DIR = "/hkfs/work/workspace/scratch/ap7811-benchmark/dailymed/data/box_flat_20000_images/data/raw_images"
DEFAULT_BASE_MODEL = "/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/Qwen2-VL-2B-Instruct"
DEFAULT_ADAPTER = "/hkfs/work/workspace/scratch/ap7811-benchmark/models/downloaded_models/MonoQwen2-VL-v0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_jsonl", default=DEFAULT_PREDS_JSONL)
    ap.add_argument("--rerank_jsonl", default=DEFAULT_RERANK_JSONL)
    ap.add_argument("--gallery_dir", default=DEFAULT_GALLERY_DIR)
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--n_worse", type=int, default=8)
    ap.add_argument("--n_better", type=int, default=4)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch_dtype = torch.bfloat16

    rerank_records = []
    with open(args.rerank_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rerank_records.append(json.loads(line))

    worse, better = [], []
    for rec in rerank_records:
        if not rec.get("reranked"):
            continue
        gt = rec["query_setid"]
        pc, pr = rec["preds_colnomic"], rec["preds_rerank"]
        rc = pc.index(gt) + 1 if gt in pc else 999
        rr = pr.index(gt) + 1 if gt in pr else 999
        if rr > rc:
            worse.append((rec, rc, rr))
        elif rr < rc:
            better.append((rec, rc, rr))

    cases = worse[: args.n_worse] + better[: args.n_better]
    print(f"[INFO] picked {len(worse[:args.n_worse])} worse + {len(better[:args.n_better])} better cases")

    gallery_map = build_gallery_setid_to_path(args.gallery_dir)
    model, processor = load_reranker(
        args.base_model, args.adapter, device, torch_dtype, args.min_pixels, args.max_pixels
    )
    true_id = processor.tokenizer.convert_tokens_to_ids("True")
    false_id = processor.tokenizer.convert_tokens_to_ids("False")

    for rec, rc, rr in cases:
        qpath = rec["query_path"]
        gt = rec["query_setid"]
        kind = "WORSE" if rr > rc else "BETTER"
        print("\n" + "=" * 100)
        print(f"[{kind}] query={qpath}")
        print(f"gt_setid={gt!r} colnomic_rank={rc} rerank_rank={rr}")

        qimg = load_image(qpath)
        text = "" if qimg is None else vlm_read_text(model, processor, device, qimg, args.max_new_tokens)
        text_clean = clean_query_text(text, args.max_chars)
        print(f"extracted_text (raw, {len(text)} chars): {text[:500]!r}")

        cand_setids = rec["preds_colnomic"][:10]
        cand_paths = []
        for sid in cand_setids:
            p = gallery_map.get(sid)
            if p:
                cand_paths.append((sid, p))
        images = [load_image(p) for _, p in cand_paths]
        keep = [(sid, img) for (sid, _), img in zip(cand_paths, images) if img is not None]
        keep_sids, keep_imgs = zip(*keep)
        scores = score_candidates(model, processor, device, text_clean, list(keep_imgs), true_id, false_id)

        print("colnomic_rank  setid  monoqwen_score")
        for i, (sid, sc) in enumerate(zip(keep_sids, scores), start=1):
            marker = " <-- GT" if sid == gt else ""
            print(f"  {i:>2}            {sid!r:<45} {sc:.4f}{marker}")


if __name__ == "__main__":
    main()
