# OCR spatial-support pilot — submitted, not a result

Frozen first128 H593 queries, natural C128, original group folds. A single offline PP-OCRv5 run for each of3885 unique images supplies identical text, boxes and recognition confidence to all arms. Original RoMa u/v and ColNomic scores are reused.

Controls: calibration-only, text, text+M interaction, true u/v-weighted text, and shuffled box-support binding. Original frozen COST1 is retained as the baseline. The main question is whether candidate-conditioned support helps use text beyond text alone and matched-budget shuffled support.

[Chinese submission report](../../ICLR/new%20ROUTEA/RC/reports/REPORT_OCR_SUPPORT_PILOT128_SUBMITTED_20261009.md) · [Frozen protocol](../../ICLR/new%20ROUTEA/RC/results/rc_ocr_support_pilot128_v1/protocol.json) · [Task receipts](../../ICLR/new%20ROUTEA/RC/results/rc_ocr_support_pilot128_v1/submission.json)

Jobs5188667–5188672: source-qualified smoke,32 OCR shards, indexing,16 support-replay shards,5-fold training, and independent summary. First real pair and loss/geometry checks passed. No new retrieval result is claimed. Source images, model weights, OCR image outputs, tensors and execution logs remain in the workspace.
