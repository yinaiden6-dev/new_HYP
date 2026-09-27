# F128 same-population B_CAL and frozen residual readers

Development panel: execution ordinals 0–127, original grouped five folds, natural ColNomic C128. B_CAL is retrained on this same panel; each residual freezes its phase-matched B_CAL. Three seeds are separate repeats, not 384 independent images.

Main operating point is fixed zero. Inherited B_CAL threshold is secondary. An inner-validation epoch must improve fixed-zero correctness to enable the residual; otherwise epoch zero preserves its baseline exactly. Best-CE diagnostics are not extra held models.

| Model | Seed | Operating point | Correct /128 | RAW rescue / break |
|---|---:|---|---:|---:|
| B_CAL | 0 | zero | 103/128 | 11/1 |
| B_CAL | 0 | inherited_tau | 93/128 | 11/11 |
| B_CAL | 1 | zero | 103/128 | 11/1 |
| B_CAL | 1 | inherited_tau | 93/128 | 11/11 |
| B_CAL | 2 | zero | 103/128 | 11/1 |
| B_CAL | 2 | inherited_tau | 93/128 | 11/11 |
| QR | 0 | zero | 102/128 | 10/1 |
| QR | 0 | inherited_tau | 92/128 | 9/10 |
| QR_VEC | 0 | zero | 100/128 | 9/2 |
| QR_VEC | 0 | inherited_tau | 90/128 | 9/12 |
| QRR | 0 | zero | 102/128 | 11/2 |
| QRR | 0 | inherited_tau | 92/128 | 10/11 |
| QR | 1 | zero | 102/128 | 11/2 |
| QR | 1 | inherited_tau | 92/128 | 10/11 |
| QR_VEC | 1 | zero | 103/128 | 11/1 |
| QR_VEC | 1 | inherited_tau | 93/128 | 10/10 |
| QRR | 1 | zero | 102/128 | 11/2 |
| QRR | 1 | inherited_tau | 92/128 | 10/11 |
| QR | 2 | zero | 100/128 | 9/2 |
| QR | 2 | inherited_tau | 92/128 | 10/11 |
| QR_VEC | 2 | zero | 99/128 | 10/4 |
| QR_VEC | 2 | inherited_tau | 93/128 | 10/10 |
| QRR | 2 | zero | 102/128 | 11/2 |
| QRR | 2 | inherited_tau | 92/128 | 11/12 |
| RAW |  | untrained | 93/128 | 0/0 |

This is a development analysis, not full-H593 or external confirmation. All 60 saved models were independently replayed before held labels were opened. No head, epoch, threshold, arm or seed is selected from these held results.

[Paired comparisons](paired_vs_B_CAL.csv) · [All predictions](joined_predictions.json) · [Validation](validation.json)
