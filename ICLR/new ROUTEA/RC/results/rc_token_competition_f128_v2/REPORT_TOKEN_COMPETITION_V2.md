# Native token competition V2: F128 development, seed0

Population: original execution ordinals 0–127, original grouped five folds, frozen natural ColNomic C128. Each residual freezes the existing same-F128, seed0 B_CAL head and normalization separately for inner fitting and outer refit. Native query tokens remain unpooled. Full C128 hit/miss loss and the original AdamW budget are preserved.

The main operating point is fixed zero; inherited B_CAL thresholds are secondary. Inner-validation correctness must strictly improve over frozen B_CAL to enable an epoch; otherwise epoch zero exactly preserves B_CAL. All 15 residual fits passed independent replay before held labels were opened.

| Model | Seed | Operating point | Correct /128 | RAW rescue / break |
|---|---:|---|---:|---:|
| B_CAL | 0 | zero | 103/128 | 11/1 |
| B_CAL | 0 | inherited_tau | 93/128 | 11/11 |
| TOKEN_QR | 0 | zero | 108/128 | 19/4 |
| TOKEN_QR | 0 | inherited_tau | 101/128 | 18/10 |
| TOKEN_QRR_ANCHOR | 0 | zero | 108/128 | 20/5 |
| TOKEN_QRR_ANCHOR | 0 | inherited_tau | 102/128 | 19/10 |
| TOKEN_QRR_MULTI | 0 | zero | 106/128 | 18/5 |
| TOKEN_QRR_MULTI | 0 | inherited_tau | 105/128 | 18/6 |
| RAW |  | untrained | 93/128 | 0/0 |

This is a seed0 development analysis. It does not establish full-H593 performance or external confirmation. Held results do not select an arm, epoch, threshold or seed.

[Paired comparisons](paired_vs_B_CAL.csv) · [MULTI versus ANCHOR](paired_structural.csv) · [Candidate scores and edges](candidate_predictions.json) · [Trace tensors](trace_manifest.json) · [Validation](validation.json)
