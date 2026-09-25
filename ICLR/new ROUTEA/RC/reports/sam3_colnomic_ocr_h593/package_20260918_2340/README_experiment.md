# H593 SAM3 + ColNomic + DeepSeek OCR Experiment Package

Created: 2026-09-18

## 1. Experiment Scope

This package summarizes the H593 exact-instance retrieval experiment built from three query splits:

| Split | Queries | Gallery | Notes |
|---|---:|---:|---|
| outcome | 478 | 5412 raw-gallery images | H593 subset from the outcome query set |
| difficult | 86 | 5412 raw-gallery images | hard/difficult query subset |
| new_difficult_train | 29 | 5412 raw-gallery images | diagnostic train subset |
| ALL_593 | 593 | 5412 raw-gallery images | weighted merge of the three splits |

The task is exact-instance retrieval: given a query image, retrieve the matching original gallery image/product instance. Evaluation uses Recall@1, Recall@5, Recall@10, and MRR.

## 2. Models and Signals

Main visual retriever:

- ColNomic-7B image retrieval over the raw gallery.
- Gallery embedding cache: `/hkfs/work/workspace/scratch/ap7811-benchmark/colnomic/result/raw_gallery/cache_7b/colnomic_gallery_emb.pt`

OCR signal:

- DeepSeek-OCR text extraction.
- Gallery OCR cache: `/hkfs/work/workspace/scratch/ap7811-benchmark/ocr_cache/gallery_ocr_deepseek_raw_gallery.jsonl`
- Query OCR and SAM3 crop OCR are stored under each split's `ocr/` directory in the original result tree.

SAM3 signal:

- SAM3 generated candidate crops/masks from query images.
- Candidate crops were evaluated with ColNomic visual retrieval and/or DeepSeek OCR text matching.
- The package keeps result JSONL and summary files, but excludes large intermediate crop/mask image caches.

## 3. Method Families

The experiment includes 30 methods per split.

- `colnomic_full`: full query image retrieval with ColNomic.
- `ocr_text_only`: query OCR text versus gallery OCR text, using text-only matching.
- `colnomic_full_ocr_a0p9` / `a0p6`: full-image ColNomic score fused with full-query OCR score, using alpha 0.9 or 0.6.
- `sam3_single_crop_*`: use one SAM3 crop per query.
- `sam3_product_package_*`: use SAM3 product/package prompts.
- `sam3_all_prompts_*`: use all available SAM3 prompts.
- `top_score`: choose the SAM3 candidate with the highest SAM3 score before retrieval.
- `topk`: aggregate over top-k SAM3 candidates.
- `cropocr`: OCR is run on SAM3 crop text.
- `ocr`: OCR is run on the full query image text.
- `oracle_any`: diagnostic upper bound: whether any SAM3 proposal contains the correct answer. It is not a deployable ranking result.

## 4. Key ALL_593 Results

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| ColNomic full | 0.7201 | 0.8702 | 0.9056 | 0.7886 |
| OCR text-only | 0.3997 | 0.5228 | 0.5531 | 0.4521 |
| ColNomic full + OCR alpha=0.9 | 0.7184 | 0.8752 | 0.9056 | 0.7846 |
| ColNomic full + OCR alpha=0.6 | 0.5852 | 0.7403 | 0.7976 | 0.6496 |
| SAM3 single crop + ColNomic | 0.6577 | 0.7875 | 0.8061 | 0.7117 |
| SAM3 single crop + OCR | 0.3322 | 0.4536 | 0.4857 | 0.3854 |
| SAM3 single crop + ColNomic + cropOCR alpha=0.9 | 0.6408 | 0.7841 | 0.8027 | 0.6991 |
| SAM3 product-package top-k + ColNomic | 0.6644 | 0.8044 | 0.8246 | 0.7221 |
| SAM3 product-package top-k + ColNomic + OCR alpha=0.9 | 0.6779 | 0.8145 | 0.8432 | 0.7352 |
| SAM3 all-prompts top-k + ColNomic | 0.6644 | 0.7892 | 0.8196 | 0.7190 |
| SAM3 all-prompts top-k + ColNomic + OCR alpha=0.9 | 0.6661 | 0.8061 | 0.8364 | 0.7236 |

## 5. Main Findings

1. ColNomic full-image retrieval remains the strongest deployed baseline on ALL_593: R@1 = 0.7201.
2. Light OCR fusion with full-image ColNomic, alpha=0.9, is almost tied with the baseline: R@1 = 0.7184, R@5 = 0.8752.
3. Heavier OCR fusion, alpha=0.6, clearly hurts retrieval, suggesting OCR is useful as a weak auxiliary signal but not reliable enough to dominate ranking.
4. SAM3 cropping does not improve over full-image retrieval. The best SAM3 deployed setting in this package is product-package top-k + ColNomic + OCR alpha=0.9 with R@1 = 0.6779.
5. OCR-only and SAM3-crop OCR-only are much weaker than image retrieval, which supports the conclusion that exact-instance matching depends strongly on visual structure, layout, and product appearance, not only readable text.
6. Oracle SAM3 metrics are diagnostic upper bounds, not final retrieval metrics. They show whether segmentation proposals contain useful candidates, but they do not solve ranking by themselves.

## 6. Package Contents

- `results/h593_summary.csv`: full 120-row summary, including 30 methods for each split and ALL_593.
- `results/h593_colnomic_full_593_summary.csv`: ColNomic-only 593 summary with correct counts.
- `results/<split>/summary.json`: original split-level metric summary.
- `results/<split>/per_sample.jsonl`: per-query predictions and scores.
- `scripts/`: exact sbatch and Python programs used for this experiment.
- `logs/`: selected Slurm logs for the latest successful/repair runs.

## 7. Reproducibility Notes

The final `difficult` run used:

- Final eval job: `5151948`
- Output status: `COMPLETED`
- `difficult/per_sample.jsonl`: 86 / 86 records
- `difficult/summary.json`: generated successfully

An older job `5150822` remained in Slurm `COMPLETING` after timeout on node `hkn0610`; it is not part of the final result and does not affect the packaged outputs.
