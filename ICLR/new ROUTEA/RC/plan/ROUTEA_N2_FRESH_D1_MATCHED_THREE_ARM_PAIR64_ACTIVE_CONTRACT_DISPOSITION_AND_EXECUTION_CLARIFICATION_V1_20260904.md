# Route A N2 fresh-D1 matched three-arm Pair64 active-contract disposition and execution clarification V1

## 1. Highest-precedence disposition

This append-only clarification has highest precedence over every Pair64 plan or
contract written on 2026-09-04 when deciding which lineage may be implemented,
validated or named by an execution authority.

The only active Pair64 contract is:

- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_CONTRACT_V1_20260904.md`;
- SHA256
  `bc53ef74ec71283d79247a2117f25e2c708c2d30bb6fda1fad194b5668c97ade`;
- disposition: `ACTIVE_PAIR64_CONTRACT`.

Its two mandatory append-only corrections are:

1. scope correction:
   `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_PAIR_SCOPE_CORRECTIVE_ADDENDUM_V1_20260904.md`,
   SHA256
   `19fa1ca3e5b898adec22c9148c397d7765af48aa23a5057d155115649fdbe124`;
2. label-authority correction:
   `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_LABEL_AUTHORITY_CORRECTION_ADDENDUM_V1_20260904.md`,
   SHA256
   `9c94f79cb718e7a111a983fa045814028cba1d737553a9c60daa46f118f39035`.

The active contract is valid only when both corrections are enforced.

## 2. Permanently superseded documents

The following documents are `SUPERSEDED_BEFORE_EXECUTION`:

- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_ONLY_EXECUTION_CONTRACT_V1_20260904.md`,
  SHA256
  `07cff47995dfa838f6465c856b05aadbf787f3902ac09ccab1f17436fbd98ac8`;
- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_PREJOIN_SUCCESSOR_DRAFT_V1_20260904.md`,
  SHA256
  `aacf113cdd31902a3a2d0d2cab0df7d91824a4e2b75a37626be6ac57ee20e759`.

Neither superseded document may be:

- bound by an execution authority, model card, launcher or validator;
- imported or parsed by production code;
- used to define a population, reference union, map count, target authority,
  output status or next stage;
- described as an active fallback or alternate arm.

Any authority or code hash graph that binds either superseded document fails
closed before reading Pair64 data.

## 3. P0 must retain the real OOF tensor authority

The active P0 remains target-free and seals the fresh-D1 base before label
access. A summary-only record is insufficient. For each of 64 Pair64 queries,
P0 must store an immutable pointer to the actual OOF record with at least:

- OOF shard ordinal and relative payload path;
- exact OOF payload physical SHA256;
- record ID consisting of canonical `query_id` and `query_ordinal`;
- held-out fold and OOF checkpoint SHA256;
- adapted-image-token dtype, shape and exact tensor SHA256;
- full 5,413-row score dtype, shape and exact tensor SHA256;
- corrected ranking and natural-C128 hashes;
- representative physical rows and base winner/tie-rule receipt.

The P0 artifact may additionally embed tensors, but the pointer cannot be
replaced by derived statistics, top-128-only scores or copied historical
Pair64 tensors.

The independent P0 validator must reopen the referenced OOF payload, locate
exactly one record by both query ID and ordinal, recompute the adapted-token and
full-score tensor hashes from contiguous bytes, and replay the corrected
ranking/C128. A pointer whose payload hash, record ID or reopened tensor hash
differs fails closed.

The OOF authority remains:

- `results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1/independent_aggregate_validation.json`;
- physical SHA256
  `fc5ac9422ae6f2fe6485b906525f7ddccd316ba9b77c481bb65f679ea706982f`;
- logical SHA256
  `46680768b95e9f350369900175c945df4c0c5252f3e1432379238d0a543b8b18`.

## 4. Corrected 5,412-identity axis is mandatory

P0 and J0 must bind the complete corrected gallery lineage:

- frozen gallery cache SHA256:
  `11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc`;
- corrected 5,413-row to 5,412-identity mapping SHA256:
  `935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4`;
- repair registry SHA256:
  `9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f`;
- repair runtime SHA256:
  `995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d`;
- repair contract SHA256:
  `867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650`.

Target presence, winner equality and challenger selection operate on corrected
exact identity. The primary N2 label authority's `target_physical_row` is
provenance only. If one exact identity has multiple physical rows, J0 uses the
representative row already selected by the sealed fresh C128; it never inserts
the authority's physical row.

Legacy 5,404-label reduction or direct physical-row equality is forbidden.

## 5. J0 join and routing clarification

The primary target source is the N2 fivefold label authority specified by the
label-authority correction addendum. J0 joins it to P0 only by canonical
`query_id`, then requires the four non-heldout N2 records to agree on target
identity, supergroup and target-row provenance.

J0 must not route or select using any historical:

- Pair64 or Balanced execution ordinal;
- `inner_fold`;
- target or competitor candidate position;
- CW0 `target_candidate_position` or `target_naturally_present`;
- old candidate-axis hash, base rank, map, score, action or member row.

Canonical ledger and OOF receipts provide source, fold, score and axis. The
target identity determines only which of the three fixed postseal branches
applies on that already sealed axis.

CW0 remains a 64/64 identity/supergroup agreement check only and cannot become
the primary target or routing authority.

## 6. Selected-endpoint reference resolution

The active Pair64 path does not materialize a target-free full 2,962-row
Pair64 reference-token union. That count belongs to a superseded planning
interpretation and is not an active execution prerequisite.

After J0 seals one fixed training pair per query, M0 resolves only the two
selected endpoint references for each of the 64 queries. Each selected
physical row must be resolved independently through:

1. the independently validated 4,976-row spatial cache when present;
2. otherwise `HybridSpatialReferenceResolver` using frozen gallery embeddings
   and the frozen processor path, with zero encoder forward.

For every endpoint, seal physical row, corrected identity, source image/path
hash, preprocessing frame, grid, reference-token bytes/hash, source kind and
source logical hash. A missing selected endpoint cannot be dropped, replaced
or mapped to a nearby row.

Reference resolution occurs after the fixed supervised pair ledger is sealed;
therefore no target-free full-reference union is claimed. The deterministic
resolver itself receives only the selected physical row, not target label,
branch or outcome.

## 7. Fixed Pair64 counts and branch audit

The active frozen population is:

- Pair64 queries: `64`;
- fixed training pairs: `64`;
- selected candidate endpoints: `128`;
- fresh RoMa query-reference map evaluations: `128`;
- historical postjoin Pair64 map reuse: `0`;
- Pair64 C_BIND evaluations: `0`;
- Pair64 fullnegative candidate rows: `0`.

The independently audited three-state population is:

- target equals fresh-D1 winner, negative winner-HOLD: `37`;
- target naturally present as a non-winner, positive SWITCH training pair:
  `27`;
- target absent from fresh C128, negative HOLD: `0`.

These counts are validation invariants, not filters, sampling quotas or a
permission to alter the three-branch rule. All 64 records remain in fixed
Pair64 order. If recomputation does not produce exactly `37/27/0`, execution
fails closed; it must not drop cases, scan candidate width, insert targets,
rebalance labels or change a threshold.

The 27 SWITCH entries are supervised positive training pairs. They are not
deployed switches, retrieval rescues or evidence of scientific improvement.

## 8. Training-only feature and next-stage boundary

M0 computes A ALL, B QUERY and C PAIRED endpoint evidence and the registered
six-feature winner-versus-counterpart vector only after P0 and J0 validate.
The feature function remains target-independent; the postseal target selects
the fixed pair.

Pair64 still does not produce C_BIND, a 127-challenger matrix, deployed
own-strongest selection, retrieval metrics or action outcomes. Current64
remains the full-C128 target-free evaluation population.

Successful Pair64 output is training-input preparation only. Its sole next
stage remains creation of:

`N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT`.

No optimizer, seven-parameter head, post-training evaluation, external access,
candidate-binding claim, ownership claim or scientific GO/NO-GO is authorized
by this clarification.

## 9. Authority validation requirement

Any future Pair64 execution authority and its independent validator must bind:

- this highest-precedence clarification;
- the active bc53 contract;
- the mandatory 19fa scope correction;
- the mandatory 9c94 label-authority correction;
- the OOF and corrected-gallery lineages above;
- every producer, independent validator, launcher and output path.

They must explicitly reject both superseded document hashes before execution.
The authority must report `active_pair64_contract_count=1`, name only the bc53
contract as active, and set `automatic_stage_advance=false`.
