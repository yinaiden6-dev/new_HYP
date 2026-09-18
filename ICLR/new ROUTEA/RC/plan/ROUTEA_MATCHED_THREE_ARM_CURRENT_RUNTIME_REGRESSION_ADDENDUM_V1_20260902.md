# Matched three-arm current-runtime regression addendum V1

This successor addendum preserves the byte-frozen primary cross-fit contract
used by the already sealed pair64 and full-negative feature artifacts, but
corrects one training-regression condition that is impossible across runtimes.

The historical frozen C head was trained with historical-runtime full-negative
TRAIN32. The sealed current-runtime TRAIN32 contains the same executions but
different query tokens, C128 axes and local scores. Therefore a newly trained C
head on current-runtime features is not required to reproduce the historical
weights.

Before interpreting any newly trained A/B/C head, the historical sanitized C
head itself, without retraining, must reproduce on sealed current-runtime EVAL
features:

- RAW base `25/32`;
- final `27/32`;
- rescue `2`;
- break `0`;
- switch `3`;
- all current C actions under threshold zero, with maximum logit reconstruction
  error no greater than `1e-12` to allow equivalent binary64 reduction order.

After this regression passes, common-3 and native-7 A/B/C heads may be trained
with identical pair64 plus current-runtime TRAIN32 data and identical optimizer
settings. Their weights are new current-runtime models. The frozen historical C
head remains the deployment floor and cannot be replaced by an internal tie.

Pair64 rows retain their sealed payload order. Current TRAIN32 and EVAL32 rows
are sorted by execution ordinal before loss/evaluation. This order is shared by
all arms and both head families and cannot be changed after results.

Because the historical pair64 artifact was produced before the current
full-negative feature aggregate completed, training requires a fresh
append-only Pair64 V2 materialization created after the full validator PASS.
V2 must be independently validated and bit-exact to V1. The trainer consumes
the V2 payload and validation, not the earlier Pair64 artifact directly.

Frozen-C regression is evaluated before constructing an optimizer. On failure,
the runner writes an abort with zero model updates and exits; no A/B/C head may
be trained.

For MRR, SWITCH has a fixed ranking meaning: move the proposed challenger from
its base rank to rank one and shift all intervening candidates down by one.
HOLD preserves the base ranking. This same transform is used by all arms and
C_BIND. Report base/final target rank, MRR, retained-correct and retained-wrong.

For each arm/family report the observed feature-matrix rank, exact zero columns
and exact duplicate columns on pair64 plus TRAIN32. Native-7 remains nominal
capacity; these observed degeneracies are part of the result.

This addendum must be hash-bound by the training runner and its independent
validator. It does not alter or invalidate upstream feature-artifact hashes.
