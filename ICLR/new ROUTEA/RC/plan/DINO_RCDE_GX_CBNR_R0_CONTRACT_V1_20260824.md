# DINO-RCDE GX-CBNR R0 contract V1

## 1. Scientific bottleneck

The missing evidence is **reference-specific geometric exclusivity**, not an
additional semantic similarity score.  ColNomic supplies global identity
prior and DINO supplies dense appearance/correspondence features, but
near-duplicate packages can score highly under both.  R0 must estimate:

```text
Z(q,g) = log p(candidate g's connected local hypothesis is explained by one
               coherent projection)
         - log p(the same evidence is produced by matched accidental nulls)
```

This contract supersedes only the R0 mechanism section of
`DINO_RCDE_CBNR_REPRESENTATION_GATE_CONTRACT_V1_20260824.md`.  Synthetic CBNR
E0, pair-centering, RAW no-regret, identity-disjoint folds and all permanent
historical negative evidence remain binding.

## 2. Candidate-specific geometric energy

For candidate `g`, direction `d` and every selected connected root `r`, use
the sealed multi-root P lock:

- connected query component with nonzero area;
- connected candidate-reference component with nonzero area;
- the unchanged DINO-RCDE 4-D consensus/cycle representation;
- fixed complete-root denominator and overlap correction;
- the unchanged shared candidate comparator against an exact-zero relational
  field, retaining signed, modulation and pair heads with no identity
  parameter.

The two direction scores are averaged into one candidate energy
`E_real(q,g)`.  A single patch cannot constitute evidence.  Structural H0,
no comparable roots or an incomplete fixed denominator gives exact H0, not a
fabricated score.

## 3. Four matched accidental nulls

For the same query, candidate, root population and shared decoder, compute:

1. `C_BIND`: full-C128 corrected-identity-disjoint reference-content
   derangement; destination P locks are byte-identical and only two required
   donor payloads reach the device;
2. `P_COORD`: result-blind spatial content derangement inside each registered
   reference component, preserving its mask, token multiset and magnitude;
3. `T_TOPOLOGY`: fixed-point-free derangement of reference-root assignments
   across the candidate's ordered multi-root hypothesis, preserving the root
   population, per-root areas, query roots and token contents while destroying
   part/line topology;
4. `N_MATCHED`: result-blind connected matched-shape translation with the same
   area, shape and border-distance stratum.

Candidate reorder is an invariance and is not `C_BIND`.  Coordinate
permutation and topology derangement are distinct interventions.  No null may
be selected or retried according to a model response.

Define the calibrated exclusivity log-ratio:

```text
E_null(q,g) = logmeanexp(
    E_C_BIND(q,g), E_P_COORD(q,g), E_T_TOPOLOGY(q,g), E_N_MATCHED(q,g))

Z(q,g) = E_real(q,g) - E_null(q,g)
```

The `logmeanexp` includes the fixed `-log(4)` term, giving an equal-prior NCE
zero point.  If any mandatory null is geometrically ineligible, candidate `g`
is `NO_MATCH_UNCALIBRATED` and cannot trigger a switch.

## 4. Retrieval and no-regret objective

For exact target `y`, frozen RAW strongest identity-disjoint rival `c` and
`m0 = RAW(y)-RAW(c)`:

```text
r_geo   = Z(q,y) - Z(q,c)
m_final = m0 + r_geo

L_rank  = softplus(-m_final)
L_keep  = I[m0>0] * relu(m0-m_final)
L_nce   = softplus(-Z(q,y))

L_GX_CBNR = L_rank + L_keep + L_nce
```

All weights are exactly one.  Batches contain equal RAW-correct and RAW-wrong
strata and form a `0.5/0.5` retrieval/no-regret mean.  The target label and RAW
margin enter only after all real/null inputs and receipts are sealed.  They
never enter the model forward.

The deployed overlay remains pair-centered:

```text
delta_g = +0.5 * (Z_g-Z_c)
delta_c = -0.5 * (Z_g-Z_c)
```

No absolute bias, scale, threshold, top-k scan or selector is authorized.

## 5. Target-free action and HOLD

RAW first supplies the natural winner and challengers.  A challenger may
replace the RAW winner only if all are true:

- all four nulls are eligible and independently receipted;
- `Z_challenger > 0`;
- `RAW(challenger)-RAW(winner)+Z_challenger-Z_winner > 0`;
- candidate reorder/swap invariants close exactly.

Otherwise the local path emits exact `NO_MATCH/HOLD` and preserves RAW.  This
is an NCE zero point, not a validation-tuned threshold.  Multi-object/ambiguous
queries therefore remain conservative unless one reference has exclusive
geometric evidence sufficient to win.

## 6. Frozen R0 topology

R0 cannot start until the append-only natural E0 OOM repair is independently
validated PASS.

For each outer fold `f`:

- training reads only compact episodes with
  `outer_fold=f`, `source_fold!=f`,
  `P_OUTERf_INNER{source_fold}_FIT`, role
  `INNER_HELDOUT_DEPLOYMENT`;
- training query-source and group sets must be disjoint from fold-f
  Balanced-32 evaluation;
- exactly 128 updates, AdamW, peak LR `3e-4`, weight decay `1e-4`, L2 clip
  `1.0`, linear warmup across all 128 updates;
- each update contains two queries: one RAW-correct and one RAW-wrong,
  selected by a frozen result-blind hash schedule from the complete eligible
  fold-local population;
- each query and each real/null branch is CUDA-streamed separately, gradients
  are accumulated under the exact joint objective, and only one optimizer
  step occurs per update;
- only update 128 is evaluated; no checkpoint, update or threshold selection.

Balanced-32 evaluation uses `P_OUTERf_OUTER_REFIT` and exactly eight records
per outer fold.  It supersedes the older wording “every inner heldout split”
for this exact-32 development screen: the four required split directions are
outer folds `1..4`.

## 7. Statistical and causal GO gates

All must pass:

- RAW-wrong target-vs-rival final direction at least `11/16`;
- RAW-correct retention exactly `16/16` (the finite-sample meaning of
  `>=0.98`);
- rescues strictly exceed breaks and net rescue is positive in every outer
  fold;
- mean `Z_target-Z_rival > 0` in every outer fold;
- mean real-minus-each-null exclusivity is positive for `C_BIND`, `P_COORD`,
  `T_TOPOLOGY` and `N_MATCHED` in every fold;
- difficult-track rescue is at least break and every available track is
  nonnegative;
- exact swap, reorder, zero-sum and HOLD invariants pass;
- null eligibility is reported per control/fold/track; no ineligible record is
  silently dropped from denominators.

Overall confidence uses `group_sha256` as the cluster.  Cluster means are
equally weighted; PCG64 seed `17`; `10,000` bootstrap resamples; two-sided 95%
interval; lower endpoint is percentile `2.5%` using NumPy quantile
`method="linear"`; required lower endpoints are strictly greater than zero.
Track comes from the frozen schedule/metadata join, never a query-ID prefix.

R0 PASS is `GX_CBNR_R0_REPRESENTATION_GO`.  Failure is
`GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY`.  PASS authorizes only a
separately frozen action/full-C128 experiment; automatic stage advance,
opened/sealed access and paper claims remain false.
