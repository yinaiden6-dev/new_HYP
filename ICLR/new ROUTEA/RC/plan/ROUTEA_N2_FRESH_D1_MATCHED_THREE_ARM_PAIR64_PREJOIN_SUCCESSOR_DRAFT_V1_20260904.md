# Route A N2 fresh-D1 matched three-arm Pair64 prejoin successor V1

## Draft state

`DRAFT_WAITING_CURRENT64_AGGREGATE`

This document is a non-executable draft. It cannot be frozen, hash-bound by an
execution authority, implemented as a formal job, or used to read Pair64
targets until the current64 aggregate at

`results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/validation.json`

exists and independently reports

`ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED`.

Its physical and logical hashes are intentionally absent from this draft. They
must be inserted only after that result passes every check. Until then:

- `contract_frozen=false`;
- `execution_authorized=false`;
- `pair64_prejoin_authorized=false`;
- `postseal_label_join_authorized=false`;
- `head_training_authorized=false`.

## 1. Purpose and fixed boundary

After current64 staging validates, the successor will do exactly two things:

1. rematerialize the fixed Pair64 population target-free on each query's
   fresh-D1 natural C128;
2. independently combine the validated 64-query Pair64 and 64-query current64
   prejoins into one target-free 128-query aggregate.

It does not join a target, construct a supervised episode, train an A/B/C head,
select a challenger, execute HOLD/SWITCH, evaluate retrieval outcomes, or make
a scientific claim.

The governing candidate-recall result remains
`N2_CANDIDATE_RECALL_NO_GO`. Therefore even a completed 128-query prejoin is
only infrastructure for a later target-present conditional experiment. It
does not repair candidate recall or authorize an end-to-end claim.

## 2. Governing lineage

The frozen parent documents are:

- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT_V1_20260904.md`,
  SHA256
  `6d1eb7515ed739cb74a1d955c8fa077fc1033df8a7e1b06757b02c1eabdf54f3`;
- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_EXACT_ROMA_MAP_CACHE_REUSE_ADDENDUM_V1_20260904.md`,
  SHA256
  `4e418ea658b517d2ce02e4ffa7337eab29833b513350167e2c42f847eee0dea6`;
- `plan/ROUTEA_N2_CANDIDATE_RECALL_AND_OWNERSHIP_TWO_GATE_ADDENDUM_V1_20260903.md`,
  SHA256
  `cb39aeaf3b0826b408b84063f0345bbcb384cb8ceef9ff5ae8760e60d1a86765`.

The current64 execution authority is fixed only as an engineering predecessor:

- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_STAGING_EXECUTION_AUTHORITY_V1_20260904.json`;
- physical SHA256
  `f1c7a39c23dd776cdc8edffdbc55012aacea3e33d8db7c9933f1044d1b13191a`;
- logical SHA256
  `99430821e99bfeea43e17b2de74f53333b60286c1cf11204899931a3bc816079`;
- required boundary:
  `current64_query_count=64`, `pair64_pending_query_count=64`,
  `contract_population_complete=false`.

The fresh-D1 numerical source remains:

- `results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1/independent_aggregate_validation.json`;
- physical SHA256
  `fc5ac9422ae6f2fe6485b906525f7ddccd316ba9b77c481bb65f679ea706982f`;
- logical SHA256
  `46680768b95e9f350369900175c945df4c0c5252f3e1432379238d0a543b8b18`;
- required status
  `ROUTEA_N2_CURRENT_RUNTIME_D1_OOF_PREJOIN_AGGREGATE_VALIDATED`.

The fixed Pair64 membership/order authority is the independently validated
Pair64 V2 lineage. Its historical scores, candidate positions, target pair,
maps and feature tensors are not numerical inputs to this successor.

## 3. Freeze prerequisites

This draft may become a frozen contract only after all of the following are
true:

1. current64 aggregate validation has the exact status named above, all checks
   true, target/group/outcome/action reads zero, 64 queries, 8,192 candidate
   pairs and `pair64_pending_query_count=64`;
2. its physical and logical hashes are inserted into the frozen successor;
3. the Pair64 query-ID-only roster and order are independently reconstructed
   without consuming target, switch label, historical target/competitor,
   historical fold or historical candidate positions;
4. the Pair64 fresh-C128 reference union and exact-map reuse/missing partition
   are sealed before any target read;
5. producer, independent validator, launchers and output paths are implemented,
   audited and hash-bound by a separate execution authority.

Failure of any prerequisite leaves this draft non-executable.

## 4. Target-free Pair64 roster

The population is exactly the registered 64 Pair64 queries in frozen payload
order. Each must join by unique `query_id` to the canonical 987-query ledger
and to exactly one fresh-D1 OOF record. Only the canonical ledger supplies
`query_ordinal` and `heldout_fold`.

Pair64 containers may be deserialized only under explicit access accounting.
Before scoring, materialize an anonymous roster containing only information
needed for target-free execution, such as query ID, source receipt, canonical
fold and fixed Pair64 order. It cannot publish or consume target identity,
target row/position, switch label, correctness, target/rival pair, outcome or
supergroup.

Poisoning every forbidden Pair64 field must leave the anonymous roster and all
prejoin outputs unchanged. Pair64 and current64 must remain disjoint 64-query
sets.

## 5. Pair64 fresh-D1 candidate axis

For every Pair64 query, consume only its sealed OOF record and preserve:

- all 5,413 physical-row scores;
- the corrected 5,412-identity ranking;
- the natural fresh-D1 C128 identities and representative physical rows;
- the OOF-adapted float32 query tokens and hashes;
- its canonical held-out fold and checkpoint binding.

Each axis contains 128 distinct corrected identities in fresh-D1 rank order.
Target insertion, candidate widening, old Pair64 candidate carryover, RAW or
old-D1 axis carryover, historical winner/rival carryover and per-identity or
per-slot parameters are forbidden.

## 6. Reference and RoMa source manifest

Before Pair64 feature materialization, create and independently validate a
Pair64-only source manifest that reports, without labels:

- 64 x 128 exact query-reference pair keys;
- the Pair64 fresh reference-row union;
- its exact overlap with the validated current64 reference cache;
- any additional reference rows requiring frozen resolver completion;
- exact historical RoMa-map reuse pairs eligible under the narrow reuse
  addendum;
- every missing pair requiring frozen RoMa recomputation.

No Pair64 reuse or missing counts are preregistered in this draft. They must be
derived once from the sealed candidate axes and exact pair keys, then frozen.
They cannot be tuned or changed after labels or outcomes are read.

An existing reference token may be reused only after byte, row, source-image,
preprocessing-frame, grid and metadata replay. Historical RoMa maps may be
reused only when every eligibility condition in the exact-map reuse addendum
passes. An ineligible pair must be recomputed by the frozen RoMa path or fail
closed; it cannot be dropped or approximated by a nearby row/map.

## 7. Fixed target-free A/B/C rematerialization

For every Pair64 query and all 128 natural candidates, preserve the same three
arms and one shared similarity matrix:

- A ALL: all query patches x all reference patches;
- B QUERY: candidate-specific query region x full reference;
- C PAIRED: candidate-specific query region x paired reference region.

Preserve the parent contract's normalization, scoring formula, canonical
geometry, RoMa checkpoint, roll controls, six native feature semantics and
fixed shift-64 C_BIND ledger. Candidate reorder and C_BIND remain distinct.
C_BIND is materialized before any later challenger selection and is never a
target/rival swap.

The Pair64 prejoin must save sufficient tensors and hashes for independent
formula, geometry, map-source, feature, reorder and C_BIND replay. All fields
that reveal a target, outcome or action remain absent.

## 8. Pair64 independent validation

The independent validator cannot import the producer. It must reconstruct and
verify at least:

- exact 64-query roster/order and fresh OOF routing;
- 8,192 natural candidate pairs and 128 distinct identities per query;
- complete reference union with no dropped row;
- exact cached-map reuse and independently rerun missing-map completion;
- A/B/C score and six-feature reconstruction;
- candidate reorder invariance;
- complete shift-64 C_BIND bundle with unchanged base gap;
- zero target, supergroup, outcome, action, insertion, update, external and
  sealed-test access;
- `current64_pending=0`, `pair64_query_count=64`, while combined population is
  still incomplete until the next aggregate validates.

A successful Pair64-only status remains staging-only and may authorize only
the combined target-free aggregate validator.

## 9. Combined 128-query target-free aggregate

The combined validator consumes only:

1. the future frozen current64 aggregate validation and its eight shard seals;
2. the future independently validated Pair64 aggregate and its shard seals.

It must prove:

- exactly 64 current64 plus 64 Pair64 queries and zero query overlap;
- exactly 16,384 query-candidate pairs;
- complete A/B/C REAL and C_BIND tensors for every pair;
- one combined reference union whose component cache entries replay exactly;
- correct fresh-D1 OOF checkpoint routing for all 128 queries;
- no target insertion and zero target/group/outcome/action reads;
- all component claims remain target-free and engineering-only.

Only this independent combined validation may set
`contract_population_complete=true` and publish:

`ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PREJOIN_VALIDATED`.

Its only permitted next stage is creation of a separate
`N2_FRESH_D1_MATCHED_THREE_ARM_POSTSEAL_JOIN_CONTRACT`. It does not execute the
join and cannot construct an optimizer.

## 10. Prohibitions and eventual mechanical state

No stage in this draft may:

- read a Pair64/current64 target, target row/rank, correctness or action;
- insert a target or alter the natural C128;
- scan candidate width, pooling, top-k, temperature, mask, map or threshold;
- reuse historical arm scores/features/actions rather than reconstructing them;
- train or load an A/B/C action head;
- evaluate rescue, break, MRR or retrieval correctness;
- access external, ISIC, sealed or `new_difficult` sealed-test data;
- make an ownership, candidate-recall, deployment or scientific claim.

When all freeze prerequisites are met, this draft must be superseded by a
separately hashed frozen contract. It must never be edited in place and must
never itself be named as execution authority.
