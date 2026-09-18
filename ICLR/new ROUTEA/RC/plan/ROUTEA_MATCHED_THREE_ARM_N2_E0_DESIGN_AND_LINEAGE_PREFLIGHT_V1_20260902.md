# Route A matched three-arm N2 E0 design and lineage preflight V1

## Authority and claim boundary

This append-only contract implements only
`N2_CURRENT_RUNTIME_D1_C_E0_DESIGN_AND_LINEAGE_PREFLIGHT` from the independently
validated N2 design.  It is an engineering gate.  It does not train D1, fit an
action head, score an external query, open a sealed endpoint, join a query
target, or make a scientific GO/NO-GO decision.

The sole predecessor is:

- design: `plan/ROUTEA_MATCHED_THREE_ARM_N2_TO_UNTOUCHED_EXTERNAL_CONFIRMATION_DESIGN_V1_20260902.md`, SHA256
  `0b796b5a88481d7189ae026fc4e3464622f24f4e15b043e1e919a02f3df15ebd`;
- independent validation:
  `results/routea_matched_three_arm_n2_external_design_v1/independent_validation.json`, SHA256
  `5681b3863c56bb8b7fa7ebd02f16c4253ba2412d4cc4ea3cd216bb90e420e0a5`;
- required predecessor status:
  `ROUTEA_MATCHED_THREE_ARM_N2_EXTERNAL_DESIGN_VALIDATED`;
- required predecessor next stage:
  `N2_CURRENT_RUNTIME_D1_C_E0_DESIGN_AND_LINEAGE_PREFLIGHT`.

Additional frozen internal inputs are:

- Pair64 V2 payload SHA256
  `d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3`;
- Pair64 V2 independent-validation SHA256
  `657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b`;
- 5,413-row ColNomic gallery cache SHA256
  `11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc`.

No automatic stage advance is permitted.  A PASS may authorize only a fresh
987-query current-runtime token-cache materialization contract; it does not
authorize D1 training by itself.

## Frozen D1 algorithm lineage

E0 binds, without modifying, these algorithm sources:

- `programs/run_a0_fold_d1.py`, SHA256
  `d8d58a0d96e1c3bc8457cf16f8b6aa0be46ceb7893a0cb6c0b75c50042eb0174`;
- `../programs/run_routea_v3_1_c6direct_m1.py`, SHA256
  `cdbac5833699c5904475cc308d1fbae6587a94765f9096d993093b2c1f621e41`;
- `../route_a_core/route_a/o1_c6direct_m1_d1.py`, SHA256
  `a3973e45f97f2157883acb96b64c614d9fa93ec80187e6608f462fe841737274`;
- `registry/upstream_inputs.json`, SHA256
  `6f0399baa46a0c774a7214486467320eb0e778fe91cefd2b2010bc7512f75eac`;
- `cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json`, SHA256
  `df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec`.

The inherited recipe must remain a shared 49,792-parameter pointwise query
adapter, 800 updates, batch size four with fixed track order
`outcome,outcome,difficult,new_difficult_train`, 63 target-free label-unique
hard negatives, AdamW learning rate `3e-4`, weight decay `0.01`, gradient clip
`1.0`, final-step-only checkpointing, and fold seed `17 + 1009*fold`.

The inherited sources freeze the optimizer, loss, schedule, split semantics and
hard-negative semantics only.  Their `5,404` legacy filename-label population
and full-gallery reducer are explicitly invalid for N2.  E0 must demonstrate
that the legacy reducer rejects the corrected axis; it may not monkey-patch or
relax the old constant.

## Corrected 5,412-identity interface

The only N2 gallery interface is
`src/rc_aslo_xf/n2_corrected_d1_runtime_v1.py`.  It must consume the validated
5,413 physical-row gallery and the frozen corrected-identity registry.  It must
require exactly 5,412 corrected identities, with the sole repeated identity
`Biogen_21` at physical rows 714 and 715.

For each corrected identity, the identity score is the maximum score of its
physical rows.  If physical rows tie at that maximum, the representative is the
lowest physical row.  Identities are ranked by score descending and then by
representative physical row ascending.  The natural C128 is the first 128
identities and contains exactly one representative row for each identity.
Target insertion, RAW-axis carryover, D1-winner carryover, and slot carryover
are forbidden.

E0 must test all of the following:

- 5,413 physical rows reduce to exactly 5,412 corrected identities;
- max pooling selects the higher duplicate-row score;
- an exact duplicate-row tie selects physical row 714;
- an all-identity score tie ranks by representative physical row;
- the C128 has 128 distinct identities and physical representatives;
- the legacy 5,404-label axis fails closed.

## Canonical query-to-fold join

The canonical ledger contains exactly 987 unique `query_id` values and unique
`query_ordinal` values in ordinal order.  Its heldout-fold population is fixed
at `{0:212,1:205,2:189,3:188,4:193}`.

Every current64 and Pair64 record is resolved by a unique `query_id` join to
that ledger.  Only the ledger supplies `query_ordinal`, `heldout_fold`, track,
grid and source receipt.  Pair64 `inner_fold`, Pair64 execution ordinal and
current64 execution ordinal are not fold or query authorities and must not be
passed to the join interface.  E0 must verify 64/64 unique Pair64 joins and
64/64 unique current64 joins.  Mutating or deleting Pair64 `inner_fold` and
execution ordinal must not change the canonical join result.

The Pair64 tensor artifact is necessarily deserialized once.  Receipts must
therefore distinguish artifact reads from semantic consumption: report one
Pair64 artifact read, but zero `inner_fold`, execution-ordinal, switch-label
and target-identity consumption by the canonical join.

## Current-runtime and geometry fixtures

E0 binds
`results/routea_d1_current_runtime_64_prejoin_v1/validation.json`, physical
SHA256 `6c031e5a82b8057a48b48794d53dd76f4888b76ed03f081a925425c64e9621cb`,
including all eight recorded shard seals.  It independently compares all 64
current64 records with the frozen current-runtime bridge: query identity,
track, grid, raw image-token bytes/hash, template-token bytes/hash, natural
corrected C128 axis and candidate scores must replay exactly, except the pre-registered `1e-6`
floating score tolerance inherited from the original replay validator.

The bridge-to-current64 match is a unique `query_id` join.  Execution ordinal
is diagnostic metadata only; changing it cannot change the join.  Every
current64 query is also checked directly against the canonical ledger for its
query ordinal, heldout fold, track, grid, source path and source SHA.

One shared current-runtime encoder and processor are loaded once.  The four
fixed fresh fixtures are:

1. `OUTCOME-0533`, grid `36x20`, raw decoded frame;
2. `DIFFICULT-0128`, grid `24x32`, raw decoded frame even when EXIF orientation
   is non-one;
3. `NDV2-007-P01`, grid `32x24`, EXIF-oriented before resize;
4. `DIFFICULT-0025`, grid `25x29`, raw decoded frame and repeated fresh forward.

The first three must reproduce both their current64 raw image tokens and
template tokens byte-exactly.  `DIFFICULT-0025` is the only canonical 987-query member with the otherwise
uncovered `25x29` grid; its two fresh forwards must be byte-exact.  Collectively
the fixtures must cover all four observed grids, all three tracks, and both
track-specific EXIF decode paths.  No behavior may be inferred for the fourth
grid from current64.

## Required outputs and hard status

The producer writes one immutable `payload.pt` and one `result.json`.  The
independent validator must reconstruct the corrected reducer, 987 joins,
current64 replay and fixture receipts without importing the producer program.

A producer failure before atomic publication leaves the output absent and
authorizes nothing.  A validator failure either leaves validation absent or
publishes an abort with `next_authorized_stage=null`; output absence is itself
fail-closed.  A PASS status is
`ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_READY`; independent
PASS is
`ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_VALIDATED`.
Both PASS envelopes may name only
`N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT` as the next
authorized stage.  Any failed hard check yields an abort with
`next_authorized_stage=null`.

All outputs must report zero D1 model loads, zero D1 updates, zero action-head
updates, zero external reads, zero sealed reads, and zero target-label joins.
The only model execution is the shared current-runtime encoder used by the four
internal fixtures.
