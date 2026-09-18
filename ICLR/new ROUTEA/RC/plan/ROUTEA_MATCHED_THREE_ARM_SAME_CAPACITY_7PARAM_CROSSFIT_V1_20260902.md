# Matched A/B/C same-capacity seven-parameter cross-fit V1

## Objective

Fairly compare A ALL, B QUERY and C PAIRED under the exact deployable
HOLD/SWITCH capacity and training data that produced the frozen RAW+C
`25/32 -> 27/32` result. The standalone local C score (`24/32`) is not the
deployed endpoint.

The primary capacity-matched comparison is a common non-degenerate linear head
with two inputs plus bias (three parameters): standardized base gap and
symmetric arm score. A secondary native-mechanism comparison uses six features
plus bias. The latter has seven nominal parameters but not identical effective
capacity: A's mass/normalized-score columns are degenerate and both robustness
columns are zero; B's reference robustness is zero. These degeneracies must be
reported, and the native comparison must not be called strict
same-effective-capacity evidence.

## Immutable primary scope

- RAW current-runtime base only;
- 32 frozen TRAIN queries and 32 identity/supergroup-disjoint EVAL queries;
- two historical balanced pair cohorts of 32 each, disjoint from both current
  TRAIN and EVAL;
- common head: two features and one bias, exactly three trainable scalars;
- native head: six features and one bias, exactly seven nominal trainable
  scalars;
- zero initialization, seed 17, AdamW, learning rate `0.03`, weight decay
  `1e-3`, 2,000 updates, switch threshold zero;
- the same pair loss, full-negative no-regret loss, batches and update count for
  all arms.

The historical D1 checkpoint is excluded. Fresh current-runtime D1 belongs to
the later N2 experiment after an evidence arm is selected.

## Native six-feature schema

For base winner `w` and challenger `c`:

1. standardized RAW score gap;
2. symmetric arm-local score;
3. symmetric effective visibility mass;
4. symmetric arm score divided by effective mass;
5. symmetric real-minus-fixed-query-roll response;
6. symmetric real-minus-fixed-reference-roll response.

Effective weights are:

- A: query ones, reference ones;
- B: RoMa query map, reference ones;
- C: RoMa query map, RoMa reference map.

Thus A naturally has mass one and two zero control responses; B has only query
response; C has both. No arm receives additional dimensions. Pair cohort V1
uses its historical roll-by-one control; pair cohort V2 uses its historical
half-map roll. Current full-negative cohorts also use the frozen half-map roll.
Every arm uses the same shift within a cohort. These conventions are immutable
and cannot be homogenized after observing results, because native-C regression
depends on their historical values. Common-3 does not consume either control.

For C_BIND, reindex the complete local bundle—score, mass, score/mass, query
response and reference response—by the fixed shift-64 ledger. Preserve the
destination candidate's base score, label and base winner. C_BIND is evaluated
with frozen heads only and never enters training.

## Two-stage feature boundary

1. Materialize and independently validate target-free full-negative features
   for all 64 current queries before reading any target-bearing role shard.
   The already-sealed result-blind TRAIN/EVAL data-split role may be preserved;
   it contains no identity, supergroup, target row or outcome.
2. Materialize the supervised training-only pair64 features on the fixed pair
   rows. Pair rows and labels cannot enter current EVAL candidate generation.

Pair64 materialization is governed by
`ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_ADDENDUM_V1_20260902.md`;
that addendum is part of this frozen contract.

C features must replay their existing pair and current full-negative feature
authorities exactly. A/B are accepted only after that regression closes.

## Training and hard regression

Train A/B/C independently with identical initialization and ordering within
each registered head family. Before
interpreting A/B:

- the newly trained C weights and bias must exactly replay the frozen sanitized
  C-head authority, subject only to a pre-registered `1e-12` scalar tolerance;
- C EVAL actions must replay base `25/32`, final `27/32`, rescue `2`, break `0`;
- all C feature rows and logits must replay.

Failure is `C_REGRESSION_ABORT`, not an A/B scientific result.

## Evaluation

On EVAL32 report for each arm:

- final R@1/MRR, rescue, break, retained correct/wrong, wrong-to-wrong;
- switch count and explicit HOLD count;
- shift-64 C_BIND action metrics and rescue retention;
- fold and track direction.

An arm is internally actionable only if final top-1 exceeds RAW 25, rescue is
greater than break and break is at most one. The frozen C model remains the
deployment default. A/B cannot replace it merely by tying 27/32 on this opened
internal panel. Any provisional strict improvement still requires a fresh
external confirmation.

## Claim boundary

This is internal arm selection, not final scientific or external evidence. It
does not authorize sealed reuse, ISIC claims, full-600 claims or final D1.
