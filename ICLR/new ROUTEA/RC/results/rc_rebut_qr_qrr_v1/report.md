# QR / QR-vec / QRR: F71 exploratory grouped development result

This is the frozen existing ordinal0–70 panel (71 queries), not a new593-query result or an external GO. All60 predeclared arm/seed/fold fits completed before outer labels were joined.

The original query identity/component folds remain unchanged. Gallery references of held identities may occur as training negatives: this is a fixed-gallery evaluation, not unseen-reference-gallery generalization. See fold_exposure_audit.json.

Each seed is reported separately;71 queries are not counted as213 independent observations. Historical481/492/496 heads are evaluated on these same71 queries but were trained on larger original H593 outer-TRAIN populations.

| Model | Seed | Operating point | Correct /71 | RAW rescue / break | Wrong-to-wrong switches |
|---|---:|---|---:|---:|---:|
| B_CAL | 0 | selected_tau | 52/71 | 6/5 | 6 |
| B_CAL | 0 | zero | 57/71 | 6/0 | 7 |
| QR | 0 | selected_tau | 44/71 | 1/8 | 4 |
| QR | 0 | zero | 29/71 | 3/25 | 16 |
| QR_VEC | 0 | selected_tau | 47/71 | 1/5 | 5 |
| QR_VEC | 0 | zero | 29/71 | 3/25 | 16 |
| QRR | 0 | selected_tau | 50/71 | 1/2 | 2 |
| QRR | 0 | zero | 31/71 | 4/24 | 16 |
| B_CAL | 1 | selected_tau | 52/71 | 6/5 | 6 |
| B_CAL | 1 | zero | 57/71 | 6/0 | 7 |
| QR | 1 | selected_tau | 49/71 | 2/4 | 6 |
| QR | 1 | zero | 34/71 | 4/21 | 15 |
| QR_VEC | 1 | selected_tau | 51/71 | 3/3 | 3 |
| QR_VEC | 1 | zero | 31/71 | 3/23 | 17 |
| QRR | 1 | selected_tau | 50/71 | 0/1 | 3 |
| QRR | 1 | zero | 33/71 | 4/22 | 16 |
| B_CAL | 2 | selected_tau | 52/71 | 6/5 | 6 |
| B_CAL | 2 | zero | 57/71 | 6/0 | 7 |
| QR | 2 | selected_tau | 53/71 | 2/0 | 3 |
| QR | 2 | zero | 33/71 | 5/23 | 14 |
| QR_VEC | 2 | selected_tau | 53/71 | 2/0 | 4 |
| QR_VEC | 2 | zero | 32/71 | 3/22 | 16 |
| QRR | 2 | selected_tau | 51/71 | 1/1 | 3 |
| QRR | 2 | zero | 27/71 | 2/26 | 17 |
| RAW |  | historical | 51/71 | 0/0 | 0 |
| COST1_481 |  | historical | 57/71 | 6/0 | 2 |
| GAP_BIAS2_492 |  | historical | 59/71 | 8/0 | 5 |
| CONTENT18_496 |  | historical | 59/71 | 9/1 | 6 |

Primary structural comparison: QRR versus QR_VEC with the same seed and operating point. Secondary: QR versus B_CAL. Both zero-threshold and inner-validation-selected-threshold results are retained; no held-based operating-point or seed selection is performed.

Paired intervals use2000 identity-cluster bootstrap resamples; component-cluster sensitivity and equal-group estimates are included. Intervals are descriptive conditional on these fixed predictions and do not account for repeated H593 development or overlapping training sets.

A negative pilot cannot establish absence of useful information in richer evidence or establish M sufficiency. A positive pilot remains developmental and does not replace the original full-population results. No external GO is declared.

- [Main table](main_results.csv)
- [Paired comparisons](paired_comparisons.csv)
- [Independent validation](validation.json)
- [Full joined predictions](joined_predictions.json)
