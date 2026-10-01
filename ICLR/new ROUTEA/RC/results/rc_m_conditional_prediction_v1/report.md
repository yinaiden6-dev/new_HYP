# M conditional prediction: H593 grouped OOF

Opened grouped OOF; fitting probes only. NLL benefit is not a direct Shannon CMI estimate, causal proof, or deployed model improvement.

Natural ColNomic C128;570 target-present queries,23 candidate misses excluded and reported. Six scalar probes, original five group folds; inner selection sees outer TRAIN only.

| Contrast | Group mean held NLL difference | Exploratory group95% interval |
|---|---:|---|
| C_PLUS_M minus C_ONLY | -0.55873549 | [-0.7825041631721182, -0.3592490758030267] |
| C_PLUS_M minus C_RICH | -0.61867524 | [-0.8494770450149678, -0.4139057719284934] |
| C_PLUS_RESIDUAL minus C_ONLY | -0.55499026 | [-0.7819192605034128, -0.3535355471964253] |
| RAW_C_M minus RAW_C | -0.56069729 | [-0.7655126781580041, -0.3763216035548419] |

Negative differences favour M. Primary is C_PLUS_M vs C_ONLY; other contrasts are declared sensitivities, not independent confirmations. No operating threshold, encoder, adapter, or deployed head changed. All128 scores, fitting convergence and original split identities are retained.
