# D1 current-runtime matched three-arm prejoin V1

## Purpose

Test whether local reference-conditioned evidence complements RAW and D1 even
though D1 alone did not improve EVAL full-gallery top-1. This stage is entirely
target-free and does not train or select a head.

## Authorized inputs

- independently validated 64-query RAW/D1 current-runtime prejoin;
- independently validated 64-query label-join handoff only. The target-bearing
  producer result is forbidden; the handoff may provide status and next-stage
  authorization but no target identity;
- independently validated 3,004-row reference spatial-token union cache;
- frozen RoMa-v2 checkpoint and canonical ColNomic geometry;
- prior target-free RAW+C RoMa prejoin as an exact replay authority.

The RoMa checkpoint is bound directly by SHA-256
`1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`.
No historical RoMa result containing target rows or target pairs may be read,
even if only its checkpoint field would otherwise be consumed.

## Two base axes

Each query retains two independent, physical-row-sorted C128 axes:

- RAW axis from frozen current-runtime ColNomic;
- D1 axis from the query's assigned fold-local D1 adapter.

In this 64-query experiment the D1 checkpoint was trained under the historical
token runtime and is applied to exact current-runtime query tokens. Therefore
the D1 axis is a checkpoint-portability diagnostic, not the final N2 D1 base.
No result from this arm may replace the requirement to retrain fresh fold-local
D1 on current-runtime tokens before final system promotion.

RoMa is run once per query/reference pair on the union of these axes. Candidate
membership is never expanded after target join.

## Matched arms

For query tokens `q`, reference tokens `r`, candidate-specific RoMa query map
`wq` and reference map `wr`:

- A ALL: `wq=1`, `wr=1`;
- B QUERY: use `wq`, set `wr=1`;
- C PAIRED: use both `wq` and `wr`.

The same maps, normalization, candidate population and score formula are used
for RAW and D1. Only the query token tensor differs. Similarity is computed
once per base/candidate, then all three arm scores are derived from that same
matrix. No arm receives a fitted scale, threshold, top-k or alternate pooling.

## Replay and controls

For every RAW-axis candidate, the new RAW+C score and both map hashes must
replay the frozen current-runtime RAW+C authority exactly (score tolerance
`1e-12`). Save the actual `wq` and `wr` tensors for independent formula replay
and later spatial controls.

For each base axis save a fixed-point-free shift-64 candidate-binding ledger:
destination position `i` receives the local evidence at source position
`(i+64) mod 128`. This is a later inference-only distribution-preserving
control; it is not used for training in this stage.

## Output boundary

Eight immutable shards each contain eight queries. Outputs include union rows,
axis memberships, maps, A/B/C scores for RAW and D1, and C_BIND ledgers.
Query target identity/role/supergroup reads, target insertion, score fusion,
model updates and sealed access are all zero. Intrinsic corrected gallery-row
identity metadata may be validated for row lineage but is never joined to a
query target in this stage.

Aggregate validation authorizes only a post-seal matched-arm diagnostic. It
does not authorize a no-regret head, D1 promotion, full-600, ISIC or sealed
claims.
