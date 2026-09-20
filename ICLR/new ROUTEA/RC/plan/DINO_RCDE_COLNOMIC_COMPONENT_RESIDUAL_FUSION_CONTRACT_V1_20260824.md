# ColNomic-preserving candidate-component residual fusion contract V1

## 1. Root-cause decision

Standalone candidate-component ranking is not viable.  On the independently
validated 582-query Scale-4 confirmation, candidate-specific evidence produced
`10 rescue / 1 break` and significant MRR/top-1 gains, but absolute top-1 was
only `10/582`; difficult and new_difficult_train had zero rescues.  The scalar
family therefore stops permanently.

The failure came from discarding the strong ColNomic retrieval score after it
generated C128 and asking a weak local branch to rank all 128 candidates by
itself.  The next mechanism restores the project baseline and uses the local
branch only as a residual:

```text
S_FUSED(q,g) = S_RAW_COLNOMIC(q,g)
             + [S_TRACK_H_COMPONENT(q,g) - S_INIT_COMPONENT(q,g)]
```

The single fixed residual weight is `1.0`.  No other weight, scale, threshold,
normalization, top-k, HOLD rule or selector arm is authorized.

## 2. Development evidence and quarantine

The fixed difficult canary executions `0..11` are development-only.  Their RAW
target-versus-strongest-rival baseline was `11/12`; the only RAW error had
margin about `-0.170`, while its component delta was about `+0.371`.  Unit
residual weight rescued it with zero break.  This observation selects the
mechanism and is not confirmation.

Executions `0..11` are permanently excluded from fusion evaluation,
statistics, plots and gates.  Confirmation uses exactly the other 582 natural
target-hit queries.  The six pre-existing target misses remain absent; there
is no target insertion.

## 3. Frozen sources

### ColNomic base

Use the independently validated I0 target-free prejoin ledger.  Before label
access it contains, for all 600 queries:

- the natural 128 physical-row axis;
- all 128 immutable ColNomic `candidate_raw_score_bits`;
- query/fold/source receipts;
- zero target/label join.

The postjoin target/rival-only RAW ledger may be used as lineage validation,
but it is not the score source because fusion must score every natural C128
candidate.

### Component residual

Use the immutable PJ1 ledger containing INIT/TRACK_H candidate-specific raw
directional logits and corrected identities after the independently validated
target-free C128 prejoin.  No model is rerun.

For each `(execution_ordinal, physical_row)`, RAW and PJ1 candidate axes,
candidate key/source receipts, query ID and fold must close exactly before
scores are combined.

## 4. Fixed candidate and identity reducer

At each physical row, in binary64 and fixed operation order:

```text
s_I = 0.5 * (z_I_a_to_b + z_I_b_to_a)
s_H = 0.5 * (z_H_a_to_b + z_H_b_to_a)
delta = s_H - s_I
s_FUSED = s_RAW + delta
```

Do not algebraically reorder the expression or normalize either score family.
Store every score and its big-endian binary64 bit pattern.

RAW and FUSED independently reduce duplicate physical rows of one corrected
identity by maximum score; exact ties choose the smaller physical row.  Each
arm independently selects its strongest wrong corrected identity using stable
ordering `(-score, physical_row, UTF8(label))`.

```text
strict_top1 = target_score > strongest_wrong_score
rank = 1 + count(wrong_identity_score >= target_score)
MRR = 1 / rank
```

Cross-arm top-1 and MRR differences are allowed.  Cross-arm margin differences
are forbidden because strongest-wrong identities may differ.

## 5. Confirmation population and statistics

- confirmation queries: 582;
- candidates: 74,496;
- folds: `{1:149, 2:140, 3:145, 4:148}`;
- tracks: `{difficult:58, new_difficult_train:24, outcome:500}`;
- supergroups: 48.

Inference reuses the registered PJ2 procedure: query mean within supergroup,
then equal supergroup mean; 9,999 PCG64(seed=17) bootstrap resamples with linear
95% quantiles; a separate 9,999-draw PCG64(seed=17) one-sided group sign-flip
test.

`COLNOMIC_COMPONENT_RESIDUAL_INTERNAL_GO` requires all:

1. group-balanced FUSED-minus-RAW top-1 gain at least `0.03`;
2. top-1 bootstrap lower bound `>0` and sign-flip `p<0.05`;
3. group-balanced MRR gain `>0`, bootstrap lower `>0`, sign-flip `p<0.05`;
4. top-1 and MRR gains positive in at least three of four folds;
5. FUSED-vs-RAW rescues strictly exceed breaks;
6. difficult-track rescues are at least breaks;
7. FUSED absolute top-1 and MRR are each strictly above RAW.

Failure is `COLNOMIC_COMPONENT_RESIDUAL_INTERNAL_NO_GO` and stops this fixed
fusion mechanism.  The confirmation remains internal because Route-A has used
this broader data resource for prior development.

## 6. Claim boundary

This experiment can establish only an internal natural-C128 retrieval
increment over ColNomic.  It cannot establish full-gallery gain, ownership or
spatial causality.  A GO must still pass candidate-binding and spatial
destruction controls, then a separately frozen full-gallery endpoint.

Model load/forward/backward/update counts are zero.  No opened/sealed access,
identity parameter, target insertion or automatic stage advance is allowed.
