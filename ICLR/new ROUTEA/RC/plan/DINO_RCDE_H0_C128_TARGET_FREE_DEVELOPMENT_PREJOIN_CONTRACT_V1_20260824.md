# DINO-RCDE H0 C128 target-free development prejoin contract V1

## 1. Decision and claim boundary

Job `5104781` independently validated the four-candidate C128 component-unary
engineering fixture.  It did not authorize formal C128 scoring.  The old
Stage-1 condition (an independently validated U4 GO) is not satisfied: U4 is a
scientific NO-GO on absolute loss.  This contract does not overwrite or
reinterpret that result.

At the user's direction, this successor authorizes a separate **development
diagnostic** which answers the actual target-free question before any label is
visible:

> For every natural candidate already present in a query's frozen C128, what
> raw candidate-specific component score is produced by the frozen INIT and
> TRACK_H outer-OOF models?

This workstream is isolated from the concurrent rescue-improvement work.  It
does not read, modify, consume or validate a successor checkpoint produced by
that workstream.  Its output is a frozen baseline which a later separately
authorized successor may compare against.

A PASS proves only that a complete target-free score ledger was produced and
validated.  It is not a retrieval GO, a rescue/break improvement, an ownership
claim or paper evidence.  Scientific metrics and label join require a later
authority.

Proceeding with raw C128 prejoin is logically compatible with the U4 NO-GO.
U4 failed an **absolute-zero/log-loss** requirement.  C128 target selection is
an ordering problem and is invariant to a candidate-common additive offset:

```text
argmax_g [s(q,g) + b(q)] = argmax_g s(q,g).
```

Therefore this diagnostic may test relative target-free candidate evidence
without claiming that absolute calibration or rescue safety has been repaired.

## 2. Frozen population and models

- redacted 600-query schedule;
- fixed natural-C128 target misses `{25,26,101,346,354,470}` are excluded;
- exactly 594 eligible queries and 128 natural candidates per query;
- exactly two fixed directions per candidate;
- outer folds `1..4`, with expected query counts `149/148/149/148`;
- only the frozen `INIT` and current `TRACK_H` update-2048 checkpoint for the
  query's outer fold;
- exactly the candidate's own `P_OUTER{o}_OUTER_REFIT` lock;
- no candidate insertion, target-conditioned pair construction or fallback
  reference mask.

The additive bias calibrator is not consumed.  A common candidate-independent
bias cancels in a C128 ranking and cannot change the winner, rank, rescue or
break.

## 3. Target-free score

For candidate `g`, direction `d`, arm `a`, and its own sealed lock:

```text
z_a(q,g,d) = decode_component_unary(model_a, q, reference_g, P_lock_g,d)
```

The runner must call the byte-bound V26 component-unary core without changing
its reducer.  READY roots consume their candidate-specific reference
components.  Structural H0 emits bit-exact zero with zero model forwards.
There is no full-reference, all-patch, RAW, D1 or rank fallback.

Prejoin output stores the two raw directional logits separately.  It may not
average directions, choose a winner, compute a rank or identify a target.

## 4. Access and execution order

Before every score ledger is closed and independently validated, the scorer
and validator may not read:

- target label, target candidate position or correctness;
- target/rival/donor role manifests;
- RAW score, rank, slot, winner or gap;
- rescue, break, outcome or any retrieval metric;
- opened or sealed data.

Allowed inputs are the redacted schedule/cache index, canonical geometry,
strictly validated full-C128 P-lock receipts/artifacts, fold assignment and
the frozen fold-local checkpoints.  All forwards run under `eval` and
`no_grad`; model state, backward count and update count must remain unchanged.

## 5. Scale canary

The first execution is frozen to schedule execution ordinal `55`, the same
real query used by the independently validated E0, but now it scores all 128
natural candidates under both arms and both directions.  Candidate selection
is therefore absent: the whole natural axis is consumed.

The scale canary must establish:

1. exactly `128` candidates and `512` raw directional arm logits;
2. exact schedule/candidate-axis/source/P-lock/token/geometry/checkpoint hash
   closure;
3. each candidate consumes only its own two outer-refit locks;
4. every READY nonzero path consumes a candidate-specific reference component;
5. structural H0 is exact zero with zero forwards;
6. all logits are finite;
7. candidate reorder is jointly equivariant after candidate-key alignment;
8. dense/streaming absolute error is at most `1e-6` on the fixed sentinel;
9. model state is unchanged and backward/update counts are zero;
10. all forbidden-access counters and scientific-metric counts are zero.

PASS is `H0_C128_TARGET_FREE_SCALE_CANARY_READY`.  Failure is
`H0_C128_TARGET_FREE_ENGINEERING_ABORT`, never a scientific NO-GO.  The canary
does not automatically authorize the full population.

## 6. Full prejoin after canary PASS

Only a separate authority issued after independent canary PASS may run the
full ledger.  Its topology is 50 fixed shards, each covering 12 schedule
execution slots; exclusions remain absent rather than being replaced.  The
aggregate must contain exactly:

- 594 queries;
- 76,032 anonymous candidate rows;
- 152,064 candidate-directions;
- 304,128 raw directional arm logits.

Every shard writes raw scores and receipts only.  A byte-bound independent
validator must close all 50 shards, all candidate axes and all upstream hashes
before a separate postjoin contract may be written.

No threshold, pooling, root count, score scale, direction reducer, checkpoint
or candidate subset may be selected after looking at the ledger.

## 7. State machine

```text
5104781 E0 independent validation PASS
  -> freeze this development contract
  -> one-query full-C128 scale canary
  -> independent canary validation
  -> separate full-prejoin authority
  -> 50 target-free raw-score shards
  -> immutable aggregate + independent validation
  -> separate label-join/reducer authority
```

Every stage records `claim_level`, hashes, data role, access audit, status,
failure reason, `scientific_GO_or_NO_GO=null`,
`automatic_stage_advance=false` and `next_authorized_stage=null`.
