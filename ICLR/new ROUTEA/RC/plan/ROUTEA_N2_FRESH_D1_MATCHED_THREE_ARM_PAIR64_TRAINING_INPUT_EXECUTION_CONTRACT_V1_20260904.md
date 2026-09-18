# Route A N2 fresh-D1 matched three-arm Pair64 training-input execution contract V1

## 1. Status, purpose and authority

This frozen contract replaces the non-executable Pair64 draft after the
current64 aggregate passed independent validation. It governs:

`Pair64 target-free fresh-D1 base preseal -> postseal fixed training-pair join -> fresh A/B/C pair features -> independent training-input handoff`.

It does not authorize a full-C128 Pair64 local-feature matrix, Pair64 C_BIND,
Pair64 fullnegative training, action-head training, label-aware evaluation or a
scientific claim.

The governing corrective addendum is:

- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_PAIR_SCOPE_CORRECTIVE_ADDENDUM_V1_20260904.md`;
- SHA256
  `19fa1ca3e5b898adec22c9148c397d7765af48aa23a5057d155115649fdbe124`.

The parent contract and exact-map addendum remain bound:

- parent rematerialization contract SHA256
  `6d1eb7515ed739cb74a1d955c8fa077fc1033df8a7e1b06757b02c1eabdf54f3`;
- current64 exact-map reuse addendum SHA256
  `4e418ea658b517d2ce02e4ffa7337eab29833b513350167e2c42f847eee0dea6`.

The corrective addendum controls whenever the parent can be read as requiring
Pair64 full-C128 target-free maps.

## 2. Passed current64 prerequisite

The current64 stage is immutable:

- path:
  `results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/validation.json`;
- physical SHA256:
  `24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859`;
- logical SHA256:
  `f28de8e8a61506dafc14cfb6e47f2c14097cc5a7553bef265b5636e779392af1`;
- required status:
  `ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED`;
- required next stage:
  `N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_PREJOIN_CONTRACT`;
- required population:
  `current64=64`, `Pair64 pending=64`, `contract_population_complete=false`.

Any mismatch aborts before Pair64 target-free roster construction. Pair64 may
not alter, regenerate or reinterpret the current64 tensors or controls.

## 3. Pair64 roster authorities and prohibited reuse

### 3.1 Primary roster/order authority

The post-full Pair64 V2 bundle fixes only cohort membership and order:

- `results/routea_matched_three_arm_pair64_training_features_v2/payload.pt`,
  SHA256
  `d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3`;
- `results/routea_matched_three_arm_pair64_training_features_v2/receipt.json`;
- `results/routea_matched_three_arm_pair64_training_features_v2/post_full_lineage.json`;
- `results/routea_matched_three_arm_pair64_training_features_v2/independent_validation.json`,
  SHA256
  `657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b`,
  required status `ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED`.

The target-free roster projection may semantically consume only each record's
list position and `query_id`. It must report zero consumption of historical
member/candidate rows, target/competitor identity, label, base rank, candidate
position, map, feature, fold, switch label, logit or action.

### 3.2 Balanced order cross-check

The exact 32+32 query-ID sequence must be independently cross-checked against:

- `results/romav2_colnomic_visibility_xf_balanced32_v1/result.json`, SHA256
  `77367e63ac835bb38340c1ae9c9d16018fde1c22b16a14a12f346d56201e404a`,
  status `ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_NO_GO`;
- `results/romav2_colnomic_visibility_xf_balanced32_v2/result.json`, SHA256
  `fd02c73f508f3f16acbafefb5b703dc864a7793ff7f688ff70efffdffa80f108`,
  status `ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_V2_GO`.

These target-bearing documents may contribute only `rows[].query_id` in
existing V1-then-V2 row order. Their candidate/member identities, roles,
scores, ranks, maps, target flags and outcomes are forbidden.

### 3.3 Role-free cross-validation only

`results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json`
may be used only to independently validate the 64-query roster. Its member
rows cannot define either endpoint of a fresh Pair64 training pair. The receipt
must report `role_free_member_read_count=0` and zero candidate-axis
construction from that artifact.

### 3.4 Canonical source, fold and axis authorities

The roster artifacts do not supply query sources, folds or candidates. Those
come only from:

- canonical ledger
  `cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json`, SHA256
  `df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec`;
- fresh-D1 OOF aggregate
  `results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1/independent_aggregate_validation.json`,
  physical SHA256
  `fc5ac9422ae6f2fe6485b906525f7ddccd316ba9b77c481bb65f679ea706982f`,
  logical SHA256
  `46680768b95e9f350369900175c945df4c0c5252f3e1432379238d0a543b8b18`,
  status `ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED`;
- its independently sealed OOF shards and fold-checkpoint bindings.

The canonical ledger supplies query source, grid, query ordinal and held-out
fold. The OOF record supplies adapted query tokens, complete scores, corrected
ranking and natural C128. Historical Pair64 candidates never enter this join.

## 4. Stage P0: target-free Pair64 base preseal

For each of the 64 query IDs in frozen order, P0 must uniquely join the
canonical ledger and fresh OOF record and seal:

- query ID and Pair64 ordinal;
- canonical query source path/hash, preprocessing frame and grid;
- held-out fold and OOF checkpoint hash;
- adapted float32 query-token hash;
- all 5,413 physical-row scores and their hash;
- the corrected 5,412-identity ranking;
- the 128 distinct natural C128 identities and representative physical rows;
- the fresh-D1 base winner and deterministic score/physical-row tie rule.

P0 cannot read any target-bearing role manifest. It cannot compute RoMa maps,
A/B/C features, a training pair, C_BIND, fullnegative rows or an action.

The producer status is
`ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY`.
An independent validator that does not import the producer must reconstruct
the 64-query order, canonical joins, OOF routing, full ranking and natural C128.
Its success status is
`ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED`.

Both outputs must report zero target, target-position, supergroup, outcome,
action, insertion, model-update, external and sealed-test access. Failure sets
`next_authorized_stage=null`.

## 5. Stage J0: postseal target join and fixed pair selection

Only after P0 independent validation may J0 read exact query targets from:

- `results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_manifest.json`;
- its `independent_validation.json`;
- its sealed `role_shards/role_exec*.json` target records.

The later execution authority must bind all physical/logical hashes and the
independent-validation status before J0. J0 may consume only the target identity
needed for supervised pair construction. Historical candidate pairs, ranks,
switch labels, outcomes and model decisions in any role source remain
forbidden.

For each presealed query, select exactly two distinct endpoints:

1. if target equals the fresh-D1 winner, choose winner plus fresh D1's strongest
   wrong and label the pair negative HOLD;
2. if target is naturally present in C128 but is not the winner, choose winner
   plus target and label the target challenger positive;
3. if target is absent from C128, choose winner plus fresh D1's strongest
   non-winner and label the pair negative HOLD.

The target is never inserted. No query is removed or rebalanced. The branch,
target-presence state, two endpoint rows and training label are sealed before
any local map is computed.

## 6. Stage M0: fresh training-only A/B/C pair features

M0 consumes only the independently validated pair ledger. It performs exactly
`64 x 2 = 128` fresh RoMa query-reference evaluations: one for the base winner
and one for its fixed counterpart on every query.

All 128 maps must be recomputed with the frozen RoMa-v2 checkpoint and
canonical geometry. Historical Pair64 postjoin maps, cached pair maps, arm
scores, feature tensors or actions are prohibited, even if their image pair
appears identical. Required counts are:

- `roma_pair_evaluation_count=128`;
- `historical_postjoin_map_reuse_count=0`;
- `query_count=64`;
- `candidate_endpoint_count=128`.

For both endpoints, compute all three fixed arms:

- A ALL: all query patches x all reference patches;
- B QUERY: candidate-specific query region x full reference;
- C PAIRED: candidate-specific query region x paired reference region.

Preserve the parent formula, normalization, canonical geometry, roll controls
and six native features. M0 may not compute Pair64 C_BIND, a 127-challenger
matrix, fullnegative rows, deployed challenger selection or retrieval metrics.

The feature function is target-independent and receives only query/reference
content. The artifact is nevertheless labelled
`POSTSEAL_TRAINING_ONLY`, because the supervised target selected the pair.

## 7. Independent validation and training-input handoff

The independent validator must not import the P0, J0 or M0 producer. It must:

- revalidate the passed current64 aggregate and all its source bindings;
- reconstruct the 64-query roster using only allowed query-ID/order fields;
- prove role-free member consumption and historical candidate consumption are
  zero;
- replay every canonical ledger/OOF join and prove fold exclusion;
- prove the P0 scores, rankings, C128 axes and winner remain byte/logically
  unchanged after the target join;
- independently reproduce all three fixed pair-selection branches;
- verify target insertion zero and retain all target-absent HOLD rows;
- rerun or independently replay all 128 fresh RoMa evaluations;
- reconstruct A/B/C scores and six-feature vectors for both endpoints;
- prove no Pair64 C_BIND or fullnegative artifact was created;
- report zero model updates, external reads and sealed-test reads.

The validated handoff contains two explicitly separate namespaces:

- `CURRENT64_TARGET_FREE_FULL_C128_EVALUATION_FEATURES`;
- `PAIR64_POSTSEAL_TRAINING_ONLY_PAIR_FEATURES`.

It must never claim that all 128 queries have target-free local features. Its
success status is
`ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUTS_VALIDATED`.

Only that independent status may authorize creation of
`N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT`.
It does not construct the optimizer or train a head.

## 8. Prohibitions and fixed state

This contract does not authorize:

- Pair64 full-C128 RoMa/A/B/C materialization;
- Pair64 fullnegative or C_BIND;
- role-free/Balanced member or candidate reuse for fresh pair construction;
- historical postjoin Pair64 map/feature/action reuse;
- target insertion, candidate widening or candidate-source scan;
- pooling, temperature, top-k, map, loss or threshold tuning;
- label-aware EVAL32 processing;
- action-head training or inference;
- external, ISIC, sealed or `new_difficult` sealed-test access;
- candidate-recall, candidate-binding, ownership, deployment or scientific
  claims.

At creation this is a frozen mechanism/data contract, not an execution
authority. No Pair64 producer, validator or launcher may run until a separate
authority binds this contract hash, all roster/role source hashes, code hashes,
output paths and target-free/postseal access counters.

Current status:

`N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_CONTRACT_FROZEN_NOT_EXECUTED`.
