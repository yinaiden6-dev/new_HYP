# DINO-RCDE BAG full-594 completion contract V1

Date: 2026-08-27  
Stage: `RCDE_BAG_FIXED_DECODER_GIVEN_C128_OUTER_OOF_COMPLETION`  
Claim boundary: natural target-hit queries, given that the target is already in the frozen C128.

## Purpose and historical disposition

`RCDE_BAG` completed four identity/supergroup-disjoint folds at exactly 2,048
updates and passed checkpoint/training/foldset engineering validation.  Its only
outer-OOF inference was the eight-query, target-free E0 prejoin.  There was no
heldout label join or scientific reducer.  V22 then changed the mainline to
candidate-conditioned connected superregions and made `RCDE_CONTEXT` the sole V
initialization.  Therefore BAG is not a scientific NO-GO; it is an unevaluated
fixed decoder.

This sidecar completion does not supersede the connected-superregion, Track-R,
V124, RGH, or GX authorities.  It answers only:

> Does the already-frozen BAG decoder expose candidate-bound exact-instance
> evidence after real token-content-to-grid organization has been destroyed?

No training, checkpoint selection, hyperparameter choice, layer choice, target
proposal, connected-region construction, retrieval fusion, HOLD/SWITCH, C8,
opened, or sealed access is authorized.

## Frozen population and order

- Use the existing 600-query redacted OOF schedule and the four validated BAG
  checkpoints.
- The scientific population is the same 594 natural C128 target-hit executions
  used by the frozen I0 metadata seal; executions `25,26,101,346,354,470` are
  excluded before the producer and are never silently inserted.
- For each eligible query, run the fold-matched BAG checkpoint over every one of
  its 128 natural candidates, in numeric physical-row order.
- Seal every candidate evidence record and the complete antisymmetric
  `128 x 128` pairwise Delta matrix before reading target, rival, RAW score,
  correctness, group, or scientific outcome.
- Use 50 immutable shards covering execution intervals of 12.  Shards may run
  concurrently but their population is fixed before execution.

## Frozen controls

The target-free finalizer creates both controls before label join.

1. `C_BIND`: on the complete C128 axis, deterministically assign every
   destination a different, corrected-identity-disjoint donor reference using
   namespace `RCDE_BAG_FULL594_C_BIND_V1_SEED17`.  Because the frozen decoder
   evaluates every candidate independently before the pair comparator and has
   no candidate-key input, full-reference rebinding is exactly the corresponding
   row/column reindex of the sealed Delta matrix.  The complete donor map and
   control matrix are sealed.  It is not a coordinate-only permutation.
2. `CANDIDATE_REORDER`: deterministically reorder the complete axis using
   namespace `RCDE_BAG_FULL594_CANDIDATE_REORDER_V1_SEED17`.  After inverse
   indexing, Delta must be exactly unchanged.  Reorder is an invariance gate,
   not C_BIND and not a scientific null.

Strict antisymmetry, zero diagonal, candidate swap sign, numeric-axis closure,
checkpoint state hashes, query/fold binding, source hashes, and sampled pair
comparator replay must pass for every shard.  Any failure aborts before label
join.

## Label join and reducer

Only after all 50 producer shards, independent shard validations, and the
target-free control aggregate pass may the joiner read the frozen 594-query I0
metadata seal.  For each query it joins the frozen target physical row and the
deployed strongest non-target RAW rival physical row, then reads:

```text
REAL margin   = Delta_REAL[target, rival]
C_BIND margin = Delta_C_BIND[target, rival]
RAW margin    = frozen RAW(target) - RAW(rival)
```

No forward, model load, update, candidate selection, or control construction is
allowed after this join.

The independent unit is the frozen supergroup.  Statistics use 9,999 fixed-seed
supergroup resamples/sign flips with seed 17.  Statistical eligibility requires
at least 40 groups, at least eight per fold, at least 12 groups in each RAW-wrong
and RAW-correct stratum, and at least two such groups per stratum per fold.

The fixed BAG decoder is GO only when all conditions hold:

- overall group-balanced target-over-rival direction is at least 0.60, mean
  margin is positive, the group-bootstrap 95% lower bound on direction is above
  0.50, and the single-arm studentized sign-flip p-value is below 0.05;
- at least three of four folds have direction above 0.50;
- RAW-wrong direction is at least 0.60 with positive mean margin;
- RAW-correct direction is at least 0.70;
- C_BIND lowers direction by at least 0.05, has positive mean margin drop,
  positive cluster-bootstrap lower bound, one-sided sign-flip p below 0.05,
  REAL greater than C_BIND in at least 0.60 of group-balanced cases, and positive
  fold mean drop in at least three folds;
- every candidate-reorder and swap closure passes.

The single-arm test is the preregistered BAG-only completion family; there is no
post-result choice between BAG and CONTEXT.  The primary reducer and independent
validator must agree byte-for-byte on rows, statistics, and decision.

## Allowed decisions and claim boundary

- Pass: `DINO_RCDE_BAG_FIXED_DECODER_CANDIDATE_BOUND_NONSPATIAL_EVIDENCE_GO`.
- Fail with adequate population: `DINO_RCDE_BAG_FIXED_DECODER_CANDIDATE_BOUND_EVIDENCE_NO_GO`.
- Insufficient population: `DINO_RCDE_BAG_COMPLETION_STATISTICALLY_INELIGIBLE`.

A GO means that this fixed, OOF BAG decoder found dispersed candidate-bound
information in DINO tokens and justifies designing a later set/component
marginal.  It does not validate a deployable dispersed multi-component model,
spatial correspondence, a target region, full-gallery recall, target absence,
retrieval gain, ownership, or HOLD/SWITCH.

A NO-GO rejects only this frozen BAG decoder and its frozen representation
recipe.  It does not reject all dispersed-patch, set, OT, or multi-component
mechanisms.

There is no automatic downstream stage.
