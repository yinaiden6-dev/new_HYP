# Matched A/B/C pair64 training-feature materialization addendum V1

This addendum governs only stage 2 (the supervised historical pair64
materialization) of
`ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md`.
It does not authorize training, current-query evaluation, candidate mining, or
candidate generation.

## Frozen inputs and target-bearing exception

The only target-bearing inputs are the already materialized
`romav2_colnomic_visibility_xf_balanced32_v1/result.json` and
`romav2_colnomic_visibility_xf_balanced32_v2/result.json` authorities.  They
provide exactly one fixed TARGET row and one fixed COMPETITOR row for each of
32 queries.  Their physical hashes, logical hashes, schemas, statuses, row
orders, positions and physical rows are immutable inputs.

The two cohorts must contain exactly 64 distinct execution ordinals.  They
must also be disjoint from the 64 execution ordinals in the validated current
runtime TRAIN/EVAL source.  This is a population audit only: no current query
candidate, role, target, score or feature may enter this materialization.

Historical query/reference tokens, geometry and C128 axes are loaded only
through the frozen `RGHFull600SourceLoaderV1`.  RAW C128 scores are loaded only
through `FrozenColNomicBaseV1`.  The RoMaV2 checkpoint is fixed to SHA-256
`1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`.
All authorities, source modules and checkpoint bytes are direct-hash bound.

The producer may read TARGET/COMPETITOR roles to derive the single supervised
`switch_label`.  It must not read an identity or supergroup.  Its output must
not expose TARGET/COMPETITOR role fields, target positions, base-correct flags
or target identities.  It may expose only the fixed base-winner/challenger
pair, its training switch label, arm features and the provenance needed for
exact validation.

## Fixed pair and arm formula

No row is searched or generated.  If the frozen RAW winner is the fixed
TARGET, the fixed COMPETITOR is the challenger and `switch_label=false`.
Otherwise the fixed COMPETITOR is the winner, the fixed TARGET is the
challenger and `switch_label=true`.

For normalized ColNomic query/reference tokens, let `S` be their cosine
similarity matrix.  For effective query/reference weights `q,r`:

```
local_i = max_j S_ij r_j
mass = sqrt(mean(q) mean(r))
score = mass * sum_i q_i local_i / max(sum_i q_i, 1e-12)
```

RoMaV2 is rerun exactly once on each fixed query/reference row to obtain
`wq,wr`.  The three arms are:

- A_ALL: `q=ones_like(wq), r=ones_like(wr)`;
- B_QUERY: `q=wq, r=ones_like(wr)`;
- C_PAIRED: `q=wq, r=wr`.

Each candidate/arm stores real score, effective mass, score divided by mass,
query-control score, reference-control score, real-minus-query-control and
real-minus-reference-control.

## Cohort-specific historical control ledger

Exact C replay takes precedence over an incorrectly generalized uniform
shift.  The registered historical shifts are:

- balanced32 V1: query shift `1`, reference shift `1`;
- balanced32 V2: query shift `max(1, len(wq)//2)`, reference shift
  `max(1, len(wr)//2)`.

Within a cohort the same shifts apply to A, B and C.  They are fixed before
training and cannot be tuned.  A therefore has zero query/reference response;
B has zero reference response.  The V2 half-map authority must not be called a
shift-one authority.

## Feature ledgers

For base winner `w`, fixed challenger `c`, population standard deviation over
the frozen RAW C128 vector, and
`sym(a,b)=(a-b)/(|a|+|b|+1e-12)`:

- common3 input (two features plus bias):
  `[z_raw(c,w), sym(score_c,score_w)]`;
- native7 input (six features plus bias):
  `[z_raw, sym(score), sym(mass), sym(score/mass),
  sym(real-query_control), sym(real-reference_control)]`.

Both vectors are materialized independently for A, B and C.  Feature ordering
is fixed.  No standardization other than `z_raw` is allowed.

## Mandatory C exact replay

For every one of the 128 fixed candidate rows, C must exactly replay the
historical pair authority's query-map hash, reference-map hash, real score,
mass, query-control score and reference-control score.  Binary64 equality is
required; a tolerance is not permitted.

For all 64 pairs, the six-dimensional C native vector must be bit-exact to
`rc_aslo_xf.romav2_colnomic_frozen_gate_v1.candidate_feature` evaluated on the
frozen pair authority.  The producer fails before publication on any mismatch.
The independent validator reconstructs all A/B/C metrics and features from
the stored maps and frozen source tokens without importing the producer.

## Output and claim boundary

The artifact is `TRAINING_ONLY_PAIR64_FEATURES_NO_EVAL_CANDIDATE_GENERATION`.
It contains exactly 64 `switch_label` values and no evaluation labels.  It is
permitted only as a hash-bound input to the registered common3/native7
cross-fit training and frozen-C regression.  In particular it cannot:

- propose, filter, score or reorder a current EVAL candidate;
- supply target rows to current full-negative materialization;
- change a current C128 axis or winner/challenger search;
- establish a scientific, external, deployment or D1 result.

Only an independent-validation PASS may authorize the training consumer.
This addendum and its launcher do not submit a Slurm job.
