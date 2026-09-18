# Route A matched three-arm N2 to untouched external confirmation design V1

## Authority and scope

This append-only document is a **design-only authority**. It may be written and
audited without reading any external query image, label or outcome. It does not
authorize external scoring, sealed reuse, model replacement or a scientific
GO/NO-GO.

The direct predecessor is the independently validated internal N1 result:

- result:
  `results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json`;
- result SHA256:
  `591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3`;
- result logical SHA256:
  `63a0df2be54283292ee9a79a86517931f99d9ba7648f3cd4f6820c0c8c9e7a7e`;
- independent validation SHA256:
  `1a7b08edfad7823db62f86c5f0b213106daead6f2ddba2d299102b69e546d8df`;
- validation logical SHA256:
  `cb4cb02bd4da71555e67136a050c2fc9e74d9644d74d5d194efc58eb9d5d4643`;
- required status:
  `ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION_PASS`;
- required next stage:
  `MATCHED_THREE_ARM_EXTERNAL_CONFIRMATION_DESIGN`.

The provisional N1 arm is `NATIVE7/C_PAIRED`, with seven parameters:

- weight:
  `[1.007810106300991, -3.7801241834284802, 5.716638282754007,
  5.9125939065333615, -0.14872291353025213, -0.23872814933012354]`;
- bias: `-1.4778745133604176`;
- parameter SHA256:
  `ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263`.

N1 is internal evidence only. RAW was `25/32`; the historical frozen C head was
`27/32`; the new NATIVE7-C head was `28/32`, with three rescues, zero breaks,
four switches and one wrong-to-wrong action. Its C_BIND control was `20/32`
and retained none of the three REAL rescues. Relative to the frozen C head,
however, N1 adds only one correct query. The exact paired evidence is therefore
too small for a scientific claim.

The common three-parameter comparison did not beat frozen C: A/B/C were
`25/26/26`. Native A/B/C have effective design ranks `3/6/7` including bias.
Thus N1 supports a provisional mechanism-specific C candidate, not a strict
same-effective-capacity proof.

## Permanent data-status audit

The following are not untouched external endpoints and cannot be reused as
paper-strength confirmation:

- `outcome`, `difficult`, and `new_difficult_train`: opened and used in the
  600-query source and multiple optimization/diagnostic stages;
- `difficult90`: previously evaluated by the historical head;
- `new_difficult` validation: previously opened for C2 model selection;
- `new_difficult` 31-query test/sealed endpoint: already consumed once by the
  historical head and was ceiling-limited at RAW `31/31`;
- B7: previously inspected, uses existing gallery identities, and still lacks
  a closed human-confirmed mapping;
- synthetic corruption folders: useful for robustness diagnostics but not a
  natural product endpoint;
- ISIC/IMA++: a generality dataset, not a natural product external endpoint.

These populations may be used only for explicitly named opened regression or
exploratory diagnostics. They cannot choose a threshold, feature, arm, base or
final product model and cannot be renamed as independent confirmation.

## Frozen sequence

The only authorized scientific sequence is:

`N1 complete -> N2 fresh current-runtime D1 -> freeze one product model ->`
`one untouched external / primary N3 -> optional disjoint N3B ->`
`opened exploratory G1/ISIC generality`.

External data collection and manifest construction may occur in parallel with
N2, but no external score or label may be read until one final product model
hash has been frozen. Parallel collection must be performed by an isolated
curator; the N2/model team cannot inspect external query bytes, filenames,
targets or acquisition outcomes before product-hash freeze.

## N2: fresh current-runtime D1 plus selected local evidence

### N2 purpose

N2 asks whether photo-to-reference domain alignment and candidate-bound paired
visibility are complementary. It does not reopen the N1 arm choice: C is the
registered primary local arm. A and B remain matched mechanism controls and
cannot replace C based on N2 results.

### D1 requirements

- First materialize and independently validate a 987-query current-runtime
  compact token cache. Existing historical 987/600 caches are not numeric
  authorities. The 64 already validated current-runtime query tokens must
  replay byte-exactly; only the remaining 923 queries are newly encoded.
- Train fresh fold-local D1 adapters on these current-runtime query tokens.
- The immutable D1 implementation lineage is:
  - `programs/run_a0_fold_d1.py`, SHA256
    `d8d58a0d96e1c3bc8457cf16f8b6aa0be46ceb7893a0cb6c0b75c50042eb0174`;
  - `../programs/run_routea_v3_1_c6direct_m1.py`, SHA256
    `cdbac5833699c5904475cc308d1fbae6587a94765f9096d993093b2c1f621e41`;
  - `../route_a_core/route_a/o1_c6direct_m1_d1.py`, SHA256
    `a3973e45f97f2157883acb96b64c614d9fa93ec80187e6608f462fe841737274`;
  - identity/supergroup fold authority `registry/upstream_inputs.json`, SHA256
    `6f0399baa46a0c774a7214486467320eb0e778fe91cefd2b2010bc7512f75eac`;
  - 987-query membership ledger
    `cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json`,
    SHA256
    `df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec`.
- These historical D1 files freeze the algorithm and recipe only. Their
  executable gallery assertions use the obsolete `5404` legacy-label count.
  N2 must add a successor wrapper using the independently validated `5412`
  corrected-identity mapping; it may not relax the count or run the legacy
  label reducer unchanged. The corrected wrapper must preserve the optimizer,
  loss, split and hard-negative semantics.
- Bind the current-runtime 64-query numerical authority at
  `results/routea_d1_current_runtime_64_prejoin_v1/validation.json`, physical
  SHA256
  `6c031e5a82b8057a48b48794d53dd76f4888b76ed03f081a925425c64e9621cb`,
  including all eight shard seals.
- The frozen recipe is a shared 49,792-parameter pointwise query adapter,
  800 updates, batch size four with fixed `2 outcome + 1 difficult + 1
  new_difficult_train`, AdamW `lr=3e-4`, weight decay `0.01`, gradient clip
  `1.0`, 63 target-free label-unique hard negatives and final-step-only
  checkpointing. Fold seed is `17 + 1009*fold`.
- Each Pair64/TRAIN32/EVAL32 query must be scored only by a fold checkpoint
  whose training identities and supergroups exclude that query. The immutable
  query-to-fold ledger is sealed before scoring. Historical checkpoint bytes
  are diagnostic only and cannot enter N2.
- Resolve every Pair64/current64 query by unique `query_id` join to the
  canonical 987-query ledger, then use its `query_ordinal` and `heldout_fold`.
  Pair64's historical `inner_fold` and execution ordinal are not D1 fold/query
  authorities and must be ignored. Require 64/64 unique joins and independently
  prove identity and supergroup exclusion for every consumed checkpoint.
- D1 must independently score the complete `5413` physical-row gallery with
  `5412` corrected exact identities.
- Collapse physical rows to corrected exact identities by maximum score. If
  physical rows of one identity tie at the maximum, its representative is the
  lowest physical row. Rank identities by score descending, then representative
  physical row ascending. The natural C128 is the first 128 identities and
  contains exactly one representative row per identity.
- D1 itself defines its winner, score gap, full ranking and natural C128.
- The D1 C128 must contain 128 distinct corrected exact identities; target
  insertion, RAW-axis carryover and winner/slot carryover are forbidden.
- N2 E0 fixtures must cover outcome, difficult, new_difficult_train, the
  track-specific EXIF path, and all four observed query grids. In particular,
  include `DIFFICULT-0025` with its otherwise-uncovered `25x29` grid in addition
  to `36x20`, `24x32` and `32x24`; do not infer its behavior from current64.
- Every scoring/materialization process may read only an opaque query-to-fold
  assignment; it cannot read that query's target identity, correctness or
  supergroup. D1 training and fold construction may use identities and
  supergroups only in the isolated split-authority process. All target-free D1
  scores, C128 axes and local A/B/C feature bundles are sealed before the
  separate target/outcome join.

### N2 rematerialization and training

- Re-materialize the fixed 64-query Pair64 population under its fold-compatible
  fresh D1 base; do not copy RAW base scores, winners, axes or the historical
  TARGET/COMPETITOR candidate choice. Candidate features are target-free.
- After the D1 ranking and C128 are sealed, join only the Pair64 target identity:
  - if target is the D1 winner, use the D1 strongest wrong as a negative HOLD;
  - if target is naturally in C128 but is not winner, use target as the positive
    challenger;
  - if target is absent, use the D1 strongest non-winner as a negative HOLD.
  No target insertion or result-dependent rebalance is allowed; all 64 queries
  remain in frozen order even if the D1 correct/wrong balance changes.
- Re-materialize current TRAIN32/EVAL32 A/B/C features on each D1 natural C128.
- Use the same A/B/C definitions and six native features. C_BIND reuses only
  the frozen shift-64 permutation algorithm: instantiate the fixed-point-free
  payload permutation anew on each D1 natural axis. RAW physical-row pairings
  are forbidden.
- Preserve
  exact HOLD, one SWITCH, seed 17, zero initialization, AdamW `lr=0.03`,
  `weight_decay=1e-3`, HOLD weight 4 and 2,000 updates.
- Train independent seven-parameter A/B/C heads. C is primary; A/B are fixed
  matched controls only.
- Report D1 alone, D1+C REAL, D1+C C_BIND, frozen RAW+C fallback and A/B
  diagnostics. No threshold, top-K, feature, loss or fold scan is authorized.
- Report C128 recall, R@1, MRR, rescue, break, wrong-to-wrong, rank transform,
  track/fold direction, feature degeneracy and all finite-state receipts.

### N2 product-model decision

`D1+C` becomes the single product candidate only if all conditions hold on the
internal identity/supergroup-disjoint OOF panel:

1. D1+C final top-1 strictly exceeds frozen RAW+C `28/32`;
2. D1+C MRR is not below RAW+C MRR;
3. rescues exceed breaks and breaks are at most one;
4. every fold containing at least one query whose D1+C and RAW+C final actions
   differ has non-negative top-1 net change versus RAW+C;
5. D1 C128 target recall is not below RAW C128 target recall;
6. D1+C REAL final exact-identity top-1 count strictly exceeds both D1-alone
   final top-1 and D1+C C_BIND final top-1;
7. at least one REAL rescue occurs and C_BIND rescue retention is below one;
8. all independent replay and causal-binding checks pass.

If any condition fails, N2 mechanically falls back to the already frozen
RAW+C NATIVE7-C model. N2 may not tune the RAW+C fallback or reinterpret a tie
as promotion. The selected model and fallback reason must be frozen before any
external scoring.

### Single deployment model

Fold-local OOF checkpoints cannot score an external query because it has no
internal fold. After the above N2 decision, freeze exactly one deployment
bundle before external access:

- If N2 fails, the bundle is the existing RAW+C head and current RAW encoder;
  no refit is permitted.
- If D1+C passes, refit one 49,792-parameter D1 adapter on all 987 internal
  current-runtime query/reference pairs with the identical D1 recipe above,
  seed 17 and 800 updates. External identities are absent by construction.
- Re-materialize D1 features with this deployment D1 and independently refit
  A, B and C heads on the fixed Pair64 plus all current64 queries, using the
  registered NATIVE7 architecture, seed 17, zero initialization and 2,000
  updates. C is the only product action head; A/B are frozen diagnostics for
  the external mechanism comparison. Architecture, threshold, loss weights and
  candidate rules are unchanged.
- Freeze encoder/runtime, D1, C head, preprocessing, corrected gallery reducer,
  candidate width, C_BIND permutation and all code/data hashes into one product
  bundle. The refit receives no internal performance claim; only the untouched
  external endpoint evaluates it.

## Untouched natural-product external confirmation

### Required population

The external endpoint must be newly collected or independently sourced and
must satisfy all of the following before inference:

- exactly `240` primary natural queries;
- exactly `48` new exact product identities;
- exactly `24` independent product supergroups;
- exactly two identities and ten queries in every supergroup;
- exactly five preregistered primary queries per identity;
- at least two acquisition locations with at least 60 primary queries per
  location, and at least three device classes with at least 40 primary queries
  per device class. Each identity's five queries must span at least two
  locations and all three device classes, under a curator-frozen assignment;
- complete byte-SHA and perceptual-near-duplicate disjointness from historical
  query/reference images, plus separately curated identity and supergroup
  disjointness;
- the legacy corrected gallery plus one preregistered reference for every new
  identity;
- full-gallery scoring and natural candidate generation, with zero oracle
  insertion and zero identity-specific parameter update.

Before any score is computed, freeze two physically isolated manifests:

1. A scorer-readable anonymous prejoin manifest containing opaque query IDs,
   identity-free staged paths, byte/perceptual hashes, acquisition metadata and
   reference-enrollment rows, but no query target identity, supergroup or
   correctness field. Original filenames and directory stems are forbidden.
2. A curator-only target-join manifest mapping opaque query IDs to target
   identity and supergroup. The scorer process cannot open this file.

Scores, features, actions and complete rankings are sealed and independently
validated before a separate process receives the curator manifest. SHA
disjointness alone is insufficient: the curator must also sign identity,
supergroup and perceptual-near-duplicate audits.

### Frozen paths

Primary paths:

1. `FINAL_BASE`;
2. `FINAL_BASE + C_REAL`;
3. `FINAL_BASE + C_BIND`.

Engineering control: candidate reorder invariance. Candidate reorder is not
C_BIND and cannot substitute for it.

Secondary mechanism controls: A_REAL and B_REAL using their already frozen
matched definitions. They only decide whether a paired-visibility-specific
subclaim is supported; they cannot rescue a failure of the primary retrieval
path. If N2 selected D1+C, report D1 alone and the frozen RAW+C fallback, but
never choose between them on the external endpoint.

This primary external contract targets candidate-bound non-spatial evidence.
Query/reference coordinate-destruction or S11 controls require a separate
spatial-ownership claim contract and cannot be added after inspecting results.

### Information sufficiency

The endpoint is `UNINFORMATIVE_CEILING` only when FINAL_BASE makes fewer than
48 errors in the fixed N=240 panel. The endpoint is consumed and cannot be
expanded or supplemented after this status.

If at least 48 base errors exist but fewer than 36 have the target naturally in
C128, the result is `EXTERNAL_CANDIDATE_RECALL_NO_GO`, not uninformative. Hard
cases may not be appended. Any later attempt requires an entirely new disjoint
endpoint and contract.

### Retrieval and binding gates

Primary retrieval GO requires all of:

- R@1 increase at least `+0.04` over FINAL_BASE;
- a paired supergroup-clustered 100,000-replicate percentile 95% bootstrap CI
  lower bound above zero, using seed 17 and resampling exactly 24 supergroups
  with replacement while retaining all their queries;
- MRR point estimate improves and its CI lower bound is at least `-0.005`;
- rescues exceed breaks;
- break count divided by base-correct count is at most `0.01`;
- every acquisition location has non-negative net change;
- rescues cover at least six identities and four supergroups.

Candidate-binding support additionally requires:

- the same clustered CI lower bound above zero for the paired per-query value
  `([REAL_final_correct]-[BASE_correct]) -
  ([CBIND_final_correct]-[BASE_correct])`; and
- REAL rescue retention under C_BIND below `0.50`.

Failure of candidate binding narrows the claim to retrieval-only; it does not
rewrite a real retrieval gain as an engineering failure. Failure of retrieval
is an external NO-GO for this frozen product model and cannot be repaired on
the same endpoint.

Exact zero differences are ties. No alternative CI, seed, clustering unit,
one-sided interval or mid-run sample extension is allowed.

## N3 memory enrollment and later scale replication

The untouched product external above is also the primary N3 one-shot enrollment
test: 48 unseen identities are enrolled with one reference each, only reference
caches/indexes are rebuilt, and query scoring uses zero model update and no
identity-specific parameter. It cannot be counted twice as two experiments.

Any later N3B claim must use a second, newly frozen and fully disjoint identity
batch or a preregistered gallery-scale expansion. Existing B7 identities cannot
establish either claim.

## G1 / ISIC generality

ISIC is a later architectural/task-generality experiment, not a substitute for
the untouched product endpoint. Existing ISIC/IMA++ assets are opened and may
support only an exploratory architectural-generality result. Before use, audit
that every eligible lesion has at least two independently captured,
non-duplicate images. Use patient-disjoint folds, lesion ID as exact identity,
one frozen reference per lesion, distinct query images, and no
diagnosis/mask/metadata at inference. The same architectural stages may be
trained on optimization lesions, but G1 cannot alter the frozen product model
or product claims. A confirmatory medical-domain claim requires a later
untouched medical endpoint.

## Interfaces and fail-closed outputs

Every N2/external stage must emit append-only `result.json`, access receipts,
physical/logical hashes and an independently implemented validator. Required
fields include status, claim level, input/model/data hashes, target-free and
postjoin access counts, fold/track/location/device metrics, rescue/break,
controls, model-update count, scientific claim boundary and exactly one
`next_authorized_stage` or null.

No stage auto-advances. Engineering abort, uninformative ceiling, retrieval
NO-GO, binding-only failure and scientific GO are distinct statuses.

## Explicit prohibitions

- no external query byte, target or outcome access under this design-only file;
- no reuse of old new_difficult/difficult90 outcomes as fresh evidence;
- no target insertion, target mask/box/point/polygon or filename/stem decision;
- no D1 rank/slot/winner leakage into local feature generation beyond the
  registered base gap and candidate axis;
- no applying the RAW-trained C head directly to D1 features;
- no model selection, threshold tuning or fallback choice on external data;
- no ISIC result used to repair the product endpoint;
- no claim of spatial ownership from C_BIND alone;
- no scientific GO from the internal `28/32` result.

The only next executable stage authorized after independent validation of this
design is `N2_CURRENT_RUNTIME_D1_C_E0_DESIGN_AND_LINEAGE_PREFLIGHT`.
