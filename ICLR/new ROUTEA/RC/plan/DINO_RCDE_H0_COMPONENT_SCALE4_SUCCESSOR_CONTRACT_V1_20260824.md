# H0 candidate-specific component Scale-4 successor contract V1

## 1. Decision and novelty boundary

The frozen component-unary Track-H readout at its native magnitude did not
change a single decision on the 12-query difficult development canary:

```text
INIT correct = 5/12
TRACK_H correct = 5/12
rescue = 0
break = 0
```

It nevertheless increased the target-rival margin on 9/12 queries and had
zero breaks.  Five of seven INIT-wrong cases had a positive Track-H-minus-INIT
margin delta.  The failure is therefore not pure absence of direction; on
this canary the new evidence is under-magnitude.

This successor tests exactly one new object and no alternative arm:

```text
S_SCALE4(q,g) = S_INIT(q,g) + 4 * [S_TRACK_H(q,g) - S_INIT(q,g)]
```

This is not an additive bias.  Candidate-common bias cancels from ranking;
Scale-4 amplifies the candidate-relative component delta and can change rank.
No model, checkpoint, representation, component, candidate set or P-lock is
changed.

The old Track-R local-component experiment remains permanent negative
evidence (`111 rescue / 107 break`, candidate-binding and spatial gates
failed).  Scale-4 does not reuse that pair logit.  It acts only on the newer
candidate-specific component-unary INIT/TRACK_H scores.

## 2. Development selection and quarantine

Scale `4.0` is a development choice made after the fixed 12-query canary role
join.  Under the affine diagnostic

```text
m_alpha = m_INIT + alpha * (m_TRACK_H - m_INIT)
```

the fixed canary has a broad zero-break interval in which additional wrong
cases cross zero; `4.0` is the smallest integer scale satisfying the canary
resource gate.  This is explicitly model selection, not confirmation.

Consequently execution ordinals `0..11` are permanently excluded from every
Scale-4 evaluation statistic, plot, confidence interval and decision.  They
may appear only in a development receipt.  No alternative scale, interpolation
rule, threshold or HOLD gate may be scanned after this contract is frozen.

## 3. Confirmation population

The input is the independently validated target-free full-C128 raw-score
ledger being produced by the isolated prejoin workstream:

- 594 natural target-hit queries;
- exactly 128 natural candidates per query;
- INIT and TRACK_H, two directions per candidate;
- no target insertion and no scorer-side semantic access.

After removing development executions `0..11`, the Scale-4 confirmation
population is exactly 582 queries.  The six original natural-C128 target
misses remain absent as before.

This contract is frozen before reading any 594-query C128 semantic reduction
or outcome.  It does not modify, cancel or reinterpret the concurrently
running raw-score jobs or their automatic INIT/TRACK_H postjoin.

## 4. Fixed reducer

Use the same binary64 direction reducer, repaired exact-label mapping,
duplicate-label reduction and stable tie rule as the frozen C128 PJ2 contract:

```text
S_A(q,row) = 0.5 * [z_A(a_to_b) + z_A(b_to_a)]
S_SCALE4 = S_INIT + 4.0 * (S_TRACK_H - S_INIT)
```

For each arm independently:

- duplicate physical rows of one corrected identity reduce by maximum score;
- an exact score tie selects the smaller physical row;
- the strongest wrong identity is selected independently for that arm;
- strict top-1 requires `target_score > strongest_wrong_score`;
- target rank and reciprocal rank use the frozen stable ordering.

The reducer reports INIT, TRACK_H and SCALE4 top-1/MRR, SCALE4-vs-INIT rescue,
break, both-correct and both-wrong, fold/track/supergroup summaries, and the
same quantities for SCALE4-vs-TRACK_H.  It must not subtract margins whose
strongest wrong identities differ across arms.

## 5. Registered confirmation gate

`COMPONENT_SCALE4_INTERNAL_CONFIRMATION_GO` requires all of:

1. group-balanced SCALE4-minus-INIT top-1 gain at least `0.03`;
2. its 9,999-supergroup-bootstrap 95% lower bound strictly positive;
3. its one-sided 9,999 group sign-flip `p < 0.05`;
4. group-balanced SCALE4-minus-INIT MRR gain strictly positive with bootstrap
   lower bound positive and sign-flip `p < 0.05`;
5. top-1 and MRR fold gain positive in at least three of four folds;
6. SCALE4-vs-INIT rescues strictly exceed breaks;
7. difficult-track SCALE4-vs-INIT rescues are at least breaks;
8. SCALE4 group-balanced top-1 and MRR are each strictly above unscaled
   TRACK_H, and SCALE4-vs-TRACK_H rescues exceed breaks.

Any failure is `COMPONENT_SCALE4_INTERNAL_CONFIRMATION_NO_GO` and permanently
stops this scalar-amplification family.  The result is internal confirmation
only because the broader 600-query resource has informed prior Route-A model
development.  It is not sealed or external-dataset evidence.

## 6. Claim and execution boundary

- labels enter only after the full anonymous ledger and its independent
  validation are immutable;
- model load/forward/backward/update counts are zero;
- no opened or sealed data;
- no full-gallery claim: this remains natural-C128 target determination;
- no spatial/ownership claim until candidate-binding and spatial destruction
  controls pass in a separately frozen stage;
- `automatic_stage_advance=false` and `next_authorized_stage=null`.

The authority may be prepared now, but execution must fail closed until the
existing 50-shard aggregate, its independent validation, and the exact PJ1
metadata join are immutable.
