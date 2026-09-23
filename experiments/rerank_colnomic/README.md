# ColNomic Rerank Experiments

This directory contains the ColNomic-7B reranking experiments copied from:

`/hkfs/work/workspace/scratch/ap7811-benchmark/rerank/colnomc_rerank`

## Scope

The experiments test whether rerankers can improve the ColNomic-7B raw-gallery retrieval baseline on:

- `outcome` / image-only query set: 1132 queries.
- `difficult` raw-gallery query set: 224 queries.

The main corrected result directory is:

- `result_csv_gt_v2/`

The older uncorrected/raw result copy is:

- `result/`

Use `result_csv_gt_v2/` for reporting unless intentionally comparing against earlier GT handling.

## Main Files

- `RERANK_WORK_LOG.md`: detailed experimental worklog and interpretation.
- `metrics_overview.csv`: compact metric table extracted from `result_csv_gt_v2/*metrics*.json`.
- `rerank_monoqwen.py`: MonoQwen reranker.
- `rerank_monoqwen_deepseek_ocr.py`: OCR-text reranker variant.
- `rerank_monoqwen_hybrid.py`: hybrid ColNomic + reranker score variant.
- `rerank_qwen3vl_hybrid.py`: Qwen3-VL hybrid reranker variant.
- `run_rerank_csv_gt_v2_all_horeka.sbatch`: corrected GT batch launcher.
- `result_csv_gt_v2/*metrics*.json`: final metrics.
- `result_csv_gt_v2/*preds*.jsonl`: per-query rerank predictions.

## Corrected GT Results

Summary from `metrics_overview.csv`:

| Metrics file | Method block | N | R@1 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| `rerank_metrics_image_only_7b.json` | `colnomic_baseline` | 1132 | 0.7562 | 0.8728 | 0.9011 | 0.8084 |
| `rerank_metrics_image_only_7b.json` | `monoqwen_rerank` | 1132 | 0.6873 | 0.8613 | 0.9011 | 0.7626 |
| `rerank_metrics_deepseek_ocr_image_only_7b.json` | `monoqwen_deepseek_ocr_rerank` | 1132 | 0.6714 | 0.8534 | 0.9011 | 0.7488 |
| `rerank_metrics_hybrid_queryimg_ocr_image_only_7b.json` | `monoqwen_hybrid` | 1132 | 0.7562 | 0.8728 | 0.9011 | 0.8084 |
| `rerank_metrics_qwen3vl_hybrid_image_only_7b.json` | `qwen3vl_hybrid` | 1132 | 0.7580 | 0.8737 | 0.9011 | 0.8093 |
| `rerank_metrics_difficult_raw_gallery_7b.json` | `colnomic_baseline` | 224 | 0.6473 | 0.8750 | 0.9062 | 0.7504 |
| `rerank_metrics_difficult_raw_gallery_7b.json` | `monoqwen_rerank` | 224 | 0.6116 | 0.8348 | 0.9062 | 0.7022 |
| `rerank_metrics_deepseek_ocr_difficult_raw_gallery_7b.json` | `monoqwen_deepseek_ocr_rerank` | 224 | 0.4643 | 0.7679 | 0.9062 | 0.5859 |
| `rerank_metrics_hybrid_queryimg_ocr_difficult_raw_gallery_7b.json` | `monoqwen_hybrid` | 224 | 0.6473 | 0.8750 | 0.9062 | 0.7504 |
| `rerank_metrics_qwen3vl_hybrid_difficult_raw_gallery_7b.json` | `qwen3vl_hybrid` | 224 | 0.6473 | 0.8750 | 0.9062 | 0.7504 |

## Interpretation

The rerank experiments show that naive reranking generally does not improve ColNomic-7B top-1 accuracy. MonoQwen and OCR-only reranking reduce R@1. Hybrid gating preserves the ColNomic baseline. Qwen3-VL hybrid gives only a negligible improvement on the 1132-query image-only set and no improvement on difficult.

This supports the working conclusion that the main failure is not simply rank-order correction inside the existing top-10 candidate set. The model still needs better part-to-whole visual grounding or candidate generation, not a reranker that fully overrides ColNomic.
