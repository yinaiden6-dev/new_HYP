# DINO-RCDE H0 bias-calibration successor contract V1

Date: 2026-08-24  
Predecessor: `H0_U4_CONDITIONAL_ABSOLUTE_VERIFICATION_NO_GO`  
Scope: one shared post-hoc logit intercept; no backbone/head retraining, slope,
temperature, direction/fold/group/candidate-specific parameter, or retrieval
claim

## 1. Decision and role of the opened U4

The U4 failure is a common-zero-point failure, not a failure of all relative
signal. Track-H has group-balanced target/donor logits `-0.2855525123 /
-0.4702416536`, margin `0.1846891413`, and win rate `0.6601978087`, but its
positive-reference BCE regresses while donor BCE improves.

The only authorized successor is therefore a **bias-only** map

```text
calibrated_logit = raw_track_h_logit + structural_ready * b
```

with one scalar `b` shared across directions, folds, groups, queries,
candidates and target/donor roles. `structural_ready=0` retains exact zero and
zero model forwards. There is no fallback.

The already-unblinded 594-query U4 is hereby demoted from confirmation to
**optimization/development only**. Its target/donor labels may optimize the
single scalar and diagnose mechanism sufficiency. It can never again issue a
successor GO, estimate unbiased generalization, select among calibrator
families, or serve as the final endpoint.

## 2. Registered objective and mathematical guarantees

Let `z+_qd` and `z-_qd` be frozen Track-H target and Q0-donor logits, and
`m_qd` the frozen structural-ready indicator. For recipient supergroup `g`,
the exact U4-aligned objective is

```text
L(b) = mean_g mean_{q in g} 0.5 * mean_d [
         softplus(-(z+_qd + m_qd*b))
       + softplus( (z-_qd + m_qd*b))
       ]
```

No available-item renormalization is allowed. Structural directions contribute
the same constant `log(2)` and zero derivative.

For nonstructural pairs,

```text
L'(b)  = mean_weighted 0.5 * [sigmoid(z+ + b) - 1
                              + sigmoid(z- + b)]
L''(b) = mean_weighted 0.5 * [sigmoid(z+ + b)(1-sigmoid(z+ + b))
                              + sigmoid(z- + b)(1-sigmoid(z- + b))] > 0
```

Thus the one-dimensional objective is strictly convex whenever at least one
finite nonstructural pair exists; the optimum is unique. Because `b=0` is in
the family, exact optimization cannot worsen calibration-set NLL. This is an
empirical optimization guarantee only, not an untouched-population guarantee.

The bias cannot create relative evidence:

```text
(z+ + b) - (z- + b) = z+ - z-
```

Consequently target/donor margin, win rate, and every margin-sign rescue/break
are exactly unchanged. The opened-U4 values remain 90 rescues and 56 breaks.
Any claimed rescue/break, pure-rank, top-1 or MRR improvement from this shared
bias alone is a contract violation (apart from separately reported
structural-zero exceptions in a mixed READY/H0 candidate set).

## 3. Frozen opened-U4 development result

Minimizing the registered group-balanced objective on the already-opened raw
U4 logits gives the single development candidate

```text
b_dev = +0.3846088654
L_U4(0)     = 0.6903228272
L_U4(b_dev) = 0.6728133962
development-only gain = 0.0175094311
```

The calibrated target/donor means are `+0.0990563530 / -0.0856327883`; their
BCE terms are `0.6713582242 / 0.6742685682`. The two separately diagnosed
direction optima are `+0.3841039747 / +0.3851132694`; this near equality
supports one shared scalar rather than direction-specific parameters. Foldwise
oracle optima are all positive (`0.50027, 0.45711, 0.24354, 0.31533`) but are
diagnostics, not four authorized parameters.

These numbers establish that a scalar intercept is mechanically sufficient to
repair the opened U4 loss. They do not establish transfer to unseen identities
or to candidate-specific component logits.

## 4. Qualification state machine

### B0 — scope and leakage freeze

- bind predecessor raw/result/validation hashes and preserve Track-H bytes;
- freeze exactly one parameter and the objective above before any run;
- prohibit current-U4 model-family search, slope/temperature scans and
  per-fold/per-direction variants;
- designate current U4 as optimization-only in every artifact.

### B1 — deterministic optimizer qualification

- solve `L'(b)=0` with a deterministic bracketed root solver in float64;
- independently recompute `L`, gradient sign around the root and strict
  convexity;
- require fresh/replay equality of `b`, metrics and logical hashes;
- require `L(b) <= L(0)`, finite output, and exact no mutation of Track-H;
- require structural-H0 exact-zero/zero-forward preservation.

### B2 — opened-U4 development closure

- reproduce `b_dev` and `L_U4(b_dev)` above within a prebound numeric
  tolerance;
- run leave-one-outer-fold-out fitting using exactly the same fixed one-bias
  family; require combined OOF improvement versus INIT `>= 0.01`;
- require calibrated Track-H loss to improve over raw Track-H in all `4/4`
  held-out folds;
- require the LOFO bias range `max(b_k)-min(b_k) <= 0.15`;
- require exact equality before/after calibration for every target-donor
  margin, win indicator, rescue/break label and nonstructural candidate order;
- publish **no GO/NO-GO** and authorize no tournament science.

### B3 — new untouched endpoint prerequisite

Before any successor scientific evaluation, a separate authority must bind a
new endpoint manifest and hashes. It must have no query-image, query identity,
recipient supergroup, target identity/reference-source or donor-assignment
overlap with the opened U4 optimization population. Its identities, labels,
rank/winner and outcomes remain inaccessible until `b_dev`, scorer, reducer,
thresholds and exclusions are frozen. No current U4 row, resample, bootstrap,
repartition or relabeled view counts as untouched.

The endpoint primary path must be the candidate-specific **component
tournament** path, with each candidate's own sealed P-lock and bound reference
components. Full-reference U4 logits are a calibration comparator only. A
natural wrong C128 candidate must not be labeled absent merely because it is
not the exact target; any absolute-negative claim requires a separately
verified Q0/null assignment.

### B4 — untouched scientific qualification

The future endpoint contract must freeze its numeric gates before label access.
At minimum it must require:

- complete target-free component scoring and independent validation;
- bias frozen before raw endpoint scoring/reduction, with no refit;
- exact raw-versus-calibrated rank/winner parity on all-all-READY candidate
  sets, with structural exceptions isolated rather than pooled;
- separately reported component absolute NLL/zero-threshold behavior on only
  valid positive and verified-null labels;
- no worse primary component loss than uncalibrated Track-H, and the frozen
  minimum improvement versus INIT/Track-R required by that new authority;
- positive effect consistency over preregistered unseen folds/supergroups;
- fail-closed NO-GO on any overlap, adaptive threshold, missing null label,
  fallback, hash drift or independent-validation failure.

Only B4 may decide whether the repaired absolute zero point transfers to the
actual component tournament. B0--B3 are engineering/development states.

## 5. Literature basis

- Platt fits a sigmoid after a fixed score-producing classifier, establishing
  post-hoc logit calibration as a separate optimization stage: [Platt,
  1999](https://www.cs.cornell.edu/courses/cs678/2007sp/platt.pdf).
- Held-out, low-dimensional post-hoc calibration is effective for neural
  networks, but its evidence is validation-set dependent: [Guo et al., ICML
  2017](https://proceedings.mlr.press/v70/guo17a.html).
- Log loss is a strictly proper scoring rule, supporting the registered BCE/NLL
  objective for an absolute probability zero point: [Gneiting and Raftery,
  JASA 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
- More flexible calibration families can overfit and can even worsen an
  already calibrated score; this motivates the identity-containing one-bias
  family here: [Kull et al., AISTATS
  2017](https://proceedings.mlr.press/v54/kull17a.html).
- Optimizing and evaluating on the same finite sample creates selection bias;
  hence opened U4 must be optimization-only and the endpoint new: [Cawley and
  Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html), [Varma
  and Simon, BMC Bioinformatics
  2006](https://pubmed.ncbi.nlm.nih.gov/16504092/).
- Calibration may fail under distribution shift, so full-reference U4 success
  cannot substitute for untouched component evidence: [Ovadia et al., NeurIPS
  2019](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).

## 6. Claim boundary

Bias-only repair can move the absolute zero point while preserving learned
relative separation. It cannot generate missing component evidence, repair a
wrong P-lock, rescue a negative margin, change an all-READY tournament rank,
prove physical absence, or establish retrieval/ownership gain. Until a new
untouched component-tournament endpoint passes B4, formal 594x128 science and
HOLD/SWITCH use remain unauthorized.
