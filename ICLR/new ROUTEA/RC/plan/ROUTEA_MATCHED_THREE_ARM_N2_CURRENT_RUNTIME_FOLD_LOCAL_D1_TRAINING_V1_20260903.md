# Route A N2 current-runtime fold-local D1 training V1

## Claim and predecessor boundary

This contract implements
`N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_CONTRACT`.  It is an internal,
identity/supergroup-disjoint training stage.  It does not evaluate the N2
product gate, train an A/B/C action head, access an external or sealed query,
or make a scientific GO/NO-GO claim.

Execution is fail-closed until both of the following immutable files exist and
pass their complete checks:

- `results/routea_matched_three_arm_n2_token_cache_v1/aggregate.json`, status
  `ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_AGGREGATE_READY`;
- `results/routea_matched_three_arm_n2_token_cache_v1/independent_validation.json`,
  status `ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_INDEPENDENT_VALIDATION_PASS`
  and `next_authorized_stage=N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_CONTRACT`.

The predecessor physical and logical hashes are measured once at execution and
sealed into every descendant.  Their values are deliberately not guessed while
Job 5129123 is pending.  Absence, non-PASS status, failed check, logical-hash
drift, or shard-seal drift yields
`N2_D1_PREDECESSOR_NOT_AUTHORIZED_ABORT`, with zero model updates.

## Frozen model and recipe

The D1 algorithm remains the historical Domain-Safe query adapter; only its
data/runtime boundary is replaced.

The executable fails unless the complete algorithm dependency set remains
byte-exact: D1 core `a3973e45f97f2157883acb96b64c614d9fa93ec80187e6608f462fe841737274`,
domain alignment `2967270e1959966413f64902d07425e543e1bead72e0f08097b1f443a5a0ebe7`,
Domain-Safe loss `53bd5436ac8a8992bf244ad422be217eb355330cc38d3479b1d83d60a674cea2`,
MaxSim runtime `a8e267a9c033b3d6a116b2445e34614eb5f7f173df90d44952bf7a6befd054df`,
corrected D1 runtime `fb0cb8f1fdeefc66aaaa587529a66b02a25331ceab706c6c05be8572da0a8c6b`,
gallery-repair runtime `995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d`,
registry `9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f`
and repair contract `867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650`.

- shared pointwise query-only adapter: `128 -> 128 -> 128 -> 128`, bounded
  residual ratio `0.25`, exactly `49,792` trainable scalars;
- parameter names and shapes are independent of identity, gallery size,
  physical row, candidate, slot and reference enrollment;
- reference tokens, template tokens and the current-runtime ColNomic encoder
  are frozen;
- 800 optimizer updates, batch size four, fixed track order
  `outcome,outcome,difficult,new_difficult_train`;
- AdamW, learning rate `3e-4`, weight decay `0.01`, gradient clip `1.0`;
- temperature `0.05`, preserve cap `0.03`, error margin `0.02`, preservation
  weight `10`, correction weight `2`, drift weight `0.05`;
- 63 natural, legal, corrected-identity-unique RAW hard negatives;
- zero-initialized delta head and final-step-only model selection;
- fresh initialization seed is fixed to `17` in every fold, exactly as in the
  historical executable; only the sampling schedule uses
  `17 + 1009*fold`.

Adding a new reference may update only its external token/cache/index entry.
It cannot add or update any D1 parameter.

## Current-runtime inputs and corrected gallery

The numerical query authority is exclusively the independently validated
16-shard 987-query cache.  Historical 987/600 query token caches and historical
D1 checkpoints are prohibited numerical inputs.  Successor scorer/trainer
processes reconstruct membership and fold routing only from the validated token
aggregate's sanitized `query_id,ordinal,fold,track` records.  They do not open
the canonical ledger's path-bearing rows.  The isolated label-join authority
uses only this sanitized axis as its join destination.

D1 scores all 5,413 physical gallery rows.  The frozen gallery repair registry
maps them to exactly 5,412 corrected exact identities, with only physical rows
714 and 715 sharing `Biogen_21`.  Legacy 5,404-label reduction is forbidden.
Target optimization identities must each map to exactly one physical gallery
row.  A corrected identity may not be partly legal and partly excluded.

The old fold authority remains the split authority, not a numerical-token
authority.  A separate label-join authority process uniquely joins all three
optimization manifests to the canonical ledger by `query_id`, checks their
identity/supergroup semantics against that split, and publishes a different
train-only payload for each fold.  Each payload contains only that fold's
training query IDs, target identities, target physical rows, tracks and
supergroups; it contains no held-out query labels or paths.  The D1 trainer is
forbidden from opening the three manifests, historical token shards, query
paths, or the full split preflight.

For every fold, train and held-out identities are disjoint, train and held-out
supergroups are disjoint, every training target row is legal and every held-out
target row is excluded from training negative mining.  The physical legal mask
remains the registered fold mask; its label counts are recomputed under the
corrected 5,412-identity axis instead of accepting the old 5,404-label
assertions.  The exact corrected eligible-identity counts for folds 0--4 are
`5314, 5325, 5319, 5300, 5330`.

The current N2 Pair64 64-query and current64 64-query feature populations both
contain folds 1--4 only; their 128 records contain no fold-0 member.  Therefore
the primary array is `1-4`.  Every later rematerializer must join each query by
canonical held-out fold and hard-check that it consumed that fold's checkpoint.
Fold 0 is supported by identical code only as an optional full-987 internal
diagnostic.  It is not a deployment prerequisite: an all-987 deployment D1 is
a separate, later refit contract.  This is checkpoint coverage for the current
N2 panel, not a complete 987-query OOF result, and may not be described as one.

## Mandatory target-free RAW prejoin

Before any target identity, correctness or supergroup is read, every one of the
987 query records is scored against all 5,413 physical gallery rows with the
unadapted current-runtime image-plus-template sum-MaxSim path.

The prejoin output contains only opaque query ID, canonical ordinal, held-out
fold, track, one finite float32 `[5413]` physical-row score tensor, tensor hash
and source bindings.  It must not contain identity, target, group, target row,
winner correctness, positive/negative status, selected negatives or outcomes.
It is materialized as 16 immutable shards with durable per-query progress and
an atomic final commit.  A separate validator that does not import the producer
recomputes schema, tensor hashes, population, bindings and a fixed RAW score
replay before an aggregate validation authorizes label join.

Only after this independent prejoin validation may the training process read
its fold's train-only label-join payload.  It applies the fold legal mask to the
already sealed scores; first reduces every legal corrected identity by maximum
physical-row score with lowest-row tie break; then excludes the paired
corrected target identity; and only then takes the stable top 63 corrected
identities.  Therefore the target affects the loss and post-seal exclusion
only; it cannot change RAW scores or their order.

## Training, resume and publication

Each fold is a separate process and checkpoint lineage.  The training process
must verify the 987/987 query-ID join, current token hashes, split counts,
identity and supergroup disjointness, corrected gallery axis, legal-mask
closure, schedule and negative-selection hashes before its first update.

Every 25 completed updates, the process atomically publishes a resume record
containing adapter state, optimizer state, CPU/CUDA RNG states, history and all
input/schedule/negative/code hashes.  Resume accepts only the greatest complete
checkpoint whose complete binding equals the current execution.  It never
repairs or overwrites a committed file.  The final checkpoint, history and
result are also write-once and atomically committed.

No held-out score, target rank, R@1, MRR or action outcome selects a checkpoint.
The only selected checkpoint is update 800.  Held-out query tokens are not
needed by the fold training loop.

## Independent fold validation

The fold validator must not import the training producer.  It independently:

1. revalidates both predecessor authorities and all 16 token/prejoin shards;
2. reconstructs corrected labels, the fold split, legal mask, schedule and all
   top-63 negative rows;
3. verifies 800 history entries, finite gradients/losses, exact recipe and
   exactly 49,792 shared parameters with no identity/gallery/slot table;
4. reconstructs the fresh initial state from fixed seed 17 and proves the final
   state changed;
5. loads the update-775 resume state and independently executes updates
   776--800.  Schema, tensor dtype/shape, IDs, schedule, negative rows and all
   non-floating state are exact.  Floating parameter, optimizer and history
   values use preregistered `atol=1e-6, rtol=1e-6`, with maximum absolute and
   relative drift reported separately; GPU model is diagnostic, not frozen;
6. reports zero external/sealed access and no held-out model selection.

Passing a fold produces
`ROUTEA_N2_CURRENT_RUNTIME_D1_FOLD_TRAINING_VALIDATED`.  All required folds
must pass before D1 scoring or local-feature rematerialization is authorized.
The next stage is
`N2_CURRENT_RUNTIME_D1_OOF_SCORING_AND_THREE_ARM_REMATERIALIZATION_CONTRACT`;
there is no automatic stage transition.

The later product gate, not this training stage, is preregistered against the
frozen RAW+C comparator: base `25/32`, MRR `0.828199`; final `28/32`, MRR
`0.90141369`, rescue/break/switch/wrong-to-wrong `3/0/4/1`, RAW C128 recall
`32/32`.  D1+C must reach at least `29/32`; these values do not enter D1 loss or
checkpoint selection.  An eventual all-987 deployment refit requires another
contract and a separately audited all-internal mask (current diagnostic union:
5,348 physical rows / 5,347 corrected identities, all 81 targets legal).  It is
not part of the fold-training authority here.

## Execution graph

1. one isolated three-manifest label-join/split authority plus independent
   validation (no model update);
2. 16-way target-free RAW prejoin shard production and validation;
3. one independent RAW-prejoin aggregate validator;
4. folds 1--4 D1 training and independent validation in parallel;
5. optional fold 0 through the identical launcher.

The launchers use explicit paths under `--export=NIL`, require CUDA, never
modify HOME, and write only below the Route A RC result/work/log directories.
They are prepared by this contract but are not submitted until the pending
987-cache aggregate independently passes.
