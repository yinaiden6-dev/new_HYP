# H0 U4 NO-GO: literature review and bias-only repair decision

Date: 2026-08-24

## Decision

The reliable narrow repair direction is a single shared post-hoc intercept on
frozen Track-H logits. This follows the measured failure mode: Track-H improved
target-donor separation but displaced both roles below zero. Retraining a more
flexible calibrator, changing the head, or scanning temperatures is not
supported by the evidence and adds avoidable selection capacity.

The fix is mathematically well posed but **not yet scientifically verified**.
Current U4 is now an optimization/development set; only a new untouched
candidate-specific component-tournament endpoint can establish transfer.

## Evidence and formula

Registered U4 values were:

| Arm | loss | target logit | donor logit | margin | win rate |
|---|---:|---:|---:|---:|---:|
| INIT | 0.6904257062 | +0.1462470788 | +0.0861800386 | 0.0600670402 | 0.6023472751 |
| Track-H | 0.6903228272 | -0.2855525123 | -0.4702416536 | 0.1846891413 | 0.6601978087 |

For one shared scalar `b`, optimize the exact group-balanced U4 log loss

```text
L(b) = mean_group mean_query 0.5 * mean_direction [
       softplus(-(target_logit + b)) + softplus(donor_logit + b)] .
```

Its second derivative is a positive weighted sum of Bernoulli variances, so
the objective is strictly convex and has a unique finite optimum. Since `b=0`
is feasible, the optimum cannot worsen development NLL.

On the already-opened U4, the diagnostic optimum is:

| quantity | value |
|---|---:|
| shared `b_dev` | +0.3846088654 |
| loss before | 0.6903228272 |
| loss after | 0.6728133962 |
| development-only improvement | 0.0175094311 |
| calibrated target logit | +0.0990563530 |
| calibrated donor logit | -0.0856327883 |

The direction-specific diagnostic optima (`0.384104` and `0.385113`) are
nearly identical, which is positive evidence for one shared bias. All four
fold diagnostic optima are positive, though variable (`0.50027, 0.45711,
0.24354, 0.31533`); this is why four fold-specific parameters are not allowed.

The stricter leave-one-outer-fold-out diagnostic fits are approximately
`0.3422, 0.3605, 0.4240, 0.4096`.  Calibrated Track-H improves over raw
Track-H in all four held-out folds, their range is below `0.15`, and the
combined OOF loss improves over INIT by about `0.01666`.  These are frozen
development qualification checks, not a new scientific endpoint.

## What this repair cannot change

Adding the same bias to a target and donor leaves their difference exactly
unchanged. Therefore Track-H margin `0.1846891413`, win rate `0.6601978087`,
90 rescues and 56 breaks remain exactly the same. It also leaves pure
all-READY C128 rank, winner, top-1 and MRR unchanged. Bias repairs only the
absolute zero point; it does not create evidence.

This is a decisive falsifiability check: if an implementation reports new
rescues/breaks or all-READY ranking gains from shared bias alone, it is wrong
or has changed something beyond bias.

## Why the direction is theoretically justified

Platt calibration introduced fitting a low-dimensional sigmoid after a fixed
classifier; the proposed map is its slope-frozen, intercept-only restriction
([Platt, 1999](https://www.cs.cornell.edu/courses/cs678/2007sp/platt.pdf)).
Guo et al. show that low-parameter post-hoc calibration can correct neural
network confidence, while requiring separate calibration data ([ICML
2017](https://proceedings.mlr.press/v70/guo17a.html)). Log loss is strictly
proper, so BCE/NLL is the correct objective for the claimed absolute
probability zero point ([Gneiting and Raftery,
2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)).

The decision to forbid flexible calibration is also evidence based. Kull et
al. document failure/overfitting risks for calibration families and motivate a
family containing the identity map ([AISTATS
2017](https://proceedings.mlr.press/v54/kull17a.html)). Cawley and Talbot and
Varma and Simon show why optimizing and evaluating on the same sample yields
selection bias ([JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html);
[BMC Bioinformatics
2006](https://pubmed.ncbi.nlm.nih.gov/16504092/)). Ovadia et al. show that
post-hoc calibration can fail under distribution shift, directly motivating a
new component-level endpoint rather than reuse of full-reference U4
([NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)).

## Reliability boundary and required next evidence

The scalar's convex optimum and exact invariances make the implementation
repair reliable and auditable. They do not make its generalization guaranteed.
The following boundary is mandatory:

1. Freeze `b_dev`, optimizer, scorer and reducer; no further family or
   threshold scan on U4.
2. Mark every current-U4-derived number optimization/development only.
3. Seal a genuinely new endpoint with zero identity, recipient-supergroup,
   image/reference-source and donor overlap with current U4.
4. Score the candidate-specific component tournament target-free, using each
   candidate's own P-lock/components; full-reference logits are comparator
   only.
5. Join labels only after immutable raw-score validation and reduce under a
   pre-registered authority.
6. Require raw/calibrated rank parity for all-READY candidates and evaluate
   absolute NLL/threshold behavior only on valid positives plus verified Q0
   nulls. A merely wrong exact label is not automatically absent in a complex
   query.

Until that untouched component-tournament gate passes independently, the
formal conclusion remains U4 NO-GO, the 594x128 scientific tournament remains
unauthorized, and no HOLD/SWITCH, retrieval or ownership claim is available.

## Artifacts consulted

- `plan/DINO_RCDE_H0_U4_OUTER_OOF_ABSOLUTE_GATE_CONTRACT_V1_20260823.md`
- `plan/DINO_RCDE_H0_FULL_REFERENCE_UNARY_TRAINING_CONTRACT_V1_20260821.md`
- `plan/DINO_RCDE_H0_C128_TOURNAMENT_E0_CONTRACT_V1_20260823.md`
- `results/dino_rcde_h0_u4_absolute_gate_v1/result.json`
- `results/dino_rcde_h0_u4_absolute_gate_validation_v1/result.json`
- `results/dino_rcde_h0_u4_raw_aggregate_v2/result.json`
