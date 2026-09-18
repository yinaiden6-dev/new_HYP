# Route A N2 fresh-D1 matched three-arm Pair64 training-only execution contract V1

## 1. Authority and scope

This frozen contract is the executable successor to the append-only Pair64
scope correction:

- corrective addendum:
  `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_PAIR_SCOPE_CORRECTIVE_ADDENDUM_V1_20260904.md`;
- corrective-addendum SHA256:
  `19fa1ca3e5b898adec22c9148c397d7765af48aa23a5057d155115649fdbe124`.

The unmodified parent remains:

- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT_V1_20260904.md`;
- SHA256
  `6d1eb7515ed739cb74a1d955c8fa077fc1033df8a7e1b06757b02c1eabdf54f3`.

Where the parent can be read as requiring Pair64 full-C128 target-free local
maps, the corrective addendum and this contract control. Current64 remains
unchanged.

This contract authorizes implementation and, only after a separate hash-bound
execution authority passes, execution through:

`PAIR64_TARGET_FREE_BASE_PRESEAL -> PAIR64_POSTSEAL_FIXED_TRAINING_PAIR -> PAIR64_TRAINING_ONLY_FEATURE_AGGREGATE_VALIDATED`.

It does not authorize seven-parameter head training, label use before the base
preseal, current64 modification, external access or a scientific claim.

## 2. Immutable current64 prerequisite

The current64 full-C128 target-free path is complete and immutable:

- path:
  `results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/validation.json`;
- physical SHA256:
  `24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859`;
- logical SHA256:
  `f28de8e8a61506dafc14cfb6e47f2c14097cc5a7553bef265b5636e779392af1`;
- required status:
  `ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED`;
- required population: 64 queries, 8,192 candidates, A/B/C REAL and C_BIND;
- required boundary:
  `contract_population_complete=false`, `pair64_pending_query_count=64`.

Every current64 check and shard seal must remain true. Pair64 may only add the
training side of the later handoff; it cannot rewrite current64 features,
candidate axes, controls or validation.

## 3. Pair64 population and roster authority

### 3.1 Fixed population

Pair64 contains exactly 64 queries in the already sealed Pair64 order. It must
remain disjoint from current64. No query may be removed, duplicated,
rebalanced or reordered because of fresh-D1 correctness, target presence or a
later training label.

The primary origin bundle is:

- `results/routea_matched_three_arm_pair64_training_features_v2/payload.pt`,
  SHA256
  `d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3`;
- its `receipt.json`, `post_full_lineage.json` and
  `independent_validation.json` must be bound by the later execution authority;
- independent-validation SHA256:
  `657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b`;
- required status:
  `ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED`.

Only the ordered `query_id` projection may be consumed from that target-bearing
bundle. Its candidates, member rows, labels, target/competitor roles, scores,
maps, features, folds and actions are forbidden as fresh inputs.

### 3.2 Independent order cross-check

The ordered 64-query projection must independently equal the concatenated
query-ID order of:

1. `results/romav2_colnomic_visibility_xf_balanced32_v1/result.json`, SHA256
   `77367e63ac835bb38340c1ae9c9d16018fde1c22b16a14a12f346d56201e404a`,
   exactly 32 rows;
2. `results/romav2_colnomic_visibility_xf_balanced32_v2/result.json`, SHA256
   `fd02c73f508f3f16acbafefb5b703dc864a7793ff7f688ff70efffdffa80f108`,
   exactly 32 rows.

These Balanced documents are target-bearing. The projection may read only
list position and `rows[].query_id`. Candidate/member fields, target or
competitor roles, base ranks, maps, scores and outcomes have semantic read
count zero.

The role-free address manifest

`results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json`

is an independent roster check only. Its member payload read count is exactly
zero. Role-free members must never supply a fresh candidate, training endpoint,
map or feature.

### 3.3 Canonical source and fold authority

For each projected query ID, only

`cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json`, SHA256
`df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec`,

may supply canonical query ordinal, source path/hash, track, grid and held-out
fold. Pair64 `inner_fold`, execution ordinal or historical candidate metadata
cannot route a checkpoint or source image.

The fresh numerical axis comes only from:

- `results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1/independent_aggregate_validation.json`;
- physical SHA256
  `fc5ac9422ae6f2fe6485b906525f7ddccd316ba9b77c481bb65f679ea706982f`;
- logical SHA256
  `46680768b95e9f350369900175c945df4c0c5252f3e1432379238d0a543b8b18`;
- required status
  `ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED`.

Its independently sealed shards supply each query's fold-local checkpoint,
adapted query tokens, all 5,413 physical-row scores, corrected 5,412-identity
ranking and natural C128.

## 4. Target-free Pair64 base preseal

Before any target or training-role join, materialize and independently validate
exactly 64 records containing:

- roster position and opaque query ID;
- canonical source/grid/fold receipts;
- OOF shard and checkpoint receipts;
- full 5,413-row scores and corrected 5,412-identity ranking hashes;
- the natural 128-identity C128 and base winner;
- adapted query-token hashes;
- a target-free Pair64 reference-token union.

Each C128 contains 128 distinct corrected identities in fresh-D1 rank order.
Target insertion, candidate widening, historical Pair64 candidate carryover,
RAW/old-D1 axis carryover and result-dependent filtering are forbidden.

This stage does not compute RoMa maps or A/B/C features.

Successful statuses are:

- `ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY`;
- `ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED`.

Only the independently validated preseal may authorize the fixed postseal
training-pair join.

## 5. Pair64 reference-token union

The target-free Pair64 natural-C128 axes have the following frozen reference
population:

- union: exactly 2,962 physical rows;
- covered by the independently validated 4,976-row spatial cache: 2,957;
- missing from that cache: exactly five rows,
  `[1366, 1826, 3528, 4796, 5411]`.

The five missing rows must be completed with
`HybridSpatialReferenceResolver` from the frozen gallery embeddings and frozen
processor path, with zero encoder-model load or forward. They may not be
dropped, substituted or copied from a nearby physical row.

For regression only:

- the old 3,004-row reference cache overlaps the Pair64 union in 2,043 rows;
- the validated current64 2,805-row cache overlaps it in 1,932 rows;
- their combined reusable row union is 2,045.

Those 2,045 rows may provide a byte-reuse regression only when physical row,
source bytes, preprocessing frame, grid, tokens and metadata replay exactly.
The primary coverage authority remains the validated 4,976-row cache plus the
five resolver completions, not the old/current cache union.

## 6. Roster poison matrix

The preseal validator must run this fixed mutation matrix before accepting the
roster:

| Source mutation | Required response |
|---|---|
| Pair64 V2 target identity, switch label, target/competitor row, map, feature, score, fold or action | ordered query-ID roster and every preseal output unchanged |
| Balanced V1/V2 fields other than list position and `query_id` | ordered query-ID roster unchanged |
| Role-free member payload, member candidate or member address | member read count remains zero; preseal unchanged |
| Historical Pair64 execution ordinal or inner fold | preseal unchanged |
| Pair64 ordered `query_id` sequence | fail closed |
| Canonical ledger query ID, source hash, grid or held-out fold | fail closed |
| OOF checkpoint, adapted-token, full-score, ranking or C128 hash | fail closed |
| Current64/Pair64 overlap injection | fail closed |

The receipt must separately report artifact deserialization and semantic-field
consumption. A container read cannot be reported as permission to consume its
target-bearing members.

## 7. Postseal target join and fixed training pair

Only after the Pair64 base preseal validates may a separate process consume
targets from the already fixed role authority:

- `results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_manifest.json`;
- `results/cw0_rgh_xf_v2_p0_a0_manifest_v2/independent_validation.json`;
- its sealed `role_shards/role_exec*.json` files.

The later execution authority must bind every physical/logical hash and exact
role-shard population before the join.

For every Pair64 query, choose exactly one counterpart by the frozen rule:

1. target equals the fresh-D1 winner: winner versus strongest wrong, negative
   HOLD;
2. target is naturally in C128 but is not winner: winner versus target,
   positive challenger;
3. target is absent from C128: winner versus strongest non-winner, negative
   HOLD.

All 64 queries remain. The three branch counts are descriptive outputs and
must sum to 64; they cannot be used to rebalance or filter training. Target
insertion remains zero. Postseal target reads may change only branch and
counterpart; they cannot change the presealed scores, ranking, C128, winner,
fold, source, tokens or reference union.

## 8. Exactly 128 fresh RoMa computations

For each query, compute local evidence for the two distinct endpoints: base
winner and fixed training counterpart. This produces exactly 128 fresh RoMa
query-reference evaluations across Pair64.

All 128 must be recomputed through the frozen RoMa-v2 path:

- RoMa checkpoint SHA256
  `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`;
- ColNomic preprocessing configuration SHA256
  `1a427e12a15406a5c4701273c91a0b319528de5c78523ae5ba5329d86e5d5557`.

The checkpoint hash above must be validated against the frozen historical
authority before execution. A later execution authority must fail closed if
the file hash differs.

Historical Pair64 postjoin maps, Balanced maps, role-free member maps and
current64 maps are forbidden. Required receipts are:

- `roma_pair_evaluation_count=128`;
- `historical_postjoin_map_reuse_count=0`;
- `role_free_member_map_read_count=0`;
- finite query/reference maps with canonical geometry and exact source rows.

## 9. Training-only A/B/C features

Using both freshly computed endpoints, construct one registered six-feature
winner-versus-counterpart vector for each of:

- A ALL: full query x full reference;
- B QUERY: candidate-specific query region x full reference;
- C PAIRED: candidate-specific query region x paired reference region.

The feature implementation remains target-independent. The target label is
used only by the fixed sampler in Section 7. Preserve the parent score formula,
normalization, visibility mass, score/mass, query-roll response and
reference-roll response exactly.

Pair64 does not produce:

- 127-challenger fullnegative matrices;
- C_BIND features or evaluation;
- candidate-reorder retrieval metrics;
- a challenger selected by a trained head;
- HOLD/SWITCH inference outcomes.

Current64 remains the sole full-C128 target-free selection/C_BIND population.

## 10. Independent validation

The independent validator must not import producer roster, join, map or feature
functions. It must independently prove:

- exact 64-query roster/order and zero overlap with current64;
- every roster poison-matrix response;
- canonical-ledger and fold-local OOF routing;
- complete full-gallery/C128 base preseal before target access;
- exact 2,962-row reference union, 2,957 validated-cache rows and the five
  fixed resolver completions;
- all three fixed branch choices and branch total 64;
- preseal scores/axes/winners unchanged by postseal target join;
- exactly 128 fresh RoMa evaluations and zero historical postjoin-map reuse;
- A/B/C six-feature reconstruction for every training pair;
- zero target insertion and zero model update;
- no current64 modification, external access or sealed-test access;
- no retrieval, candidate-binding, ownership or scientific result claim.

The training-only feature aggregate succeeds only with:

`ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_ONLY_FEATURE_AGGREGATE_VALIDATED`.

Its claim level is supervised internal training-input preparation only.

## 11. Next-stage boundary

Only the independently validated Pair64 training-only aggregate, together with
the immutable current64 aggregate in Section 2, may authorize creation of:

`N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT`.

This contract does not construct that optimizer, train a head, join current64
targets, or automatically submit the next stage. A later execution authority
must bind this contract, all producer/validator/launcher hashes, the role
authority and exact outputs before any Pair64 job starts.

At creation, the frozen status is:

`N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_ONLY_EXECUTION_CONTRACT_FROZEN_NOT_EXECUTED`.

`automatic_stage_advance=false`, `scientific_GO_or_NO_GO=null`, and
`ownership_GO_or_NO_GO=null`.
