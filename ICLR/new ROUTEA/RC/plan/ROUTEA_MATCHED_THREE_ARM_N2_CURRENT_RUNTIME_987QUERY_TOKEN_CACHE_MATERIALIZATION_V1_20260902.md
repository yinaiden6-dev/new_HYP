# Route A N2 current-runtime 987-query token-cache materialization V1

## Authority and claim boundary

This contract implements only
`N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT`.  It is an
engineering, target-free, prejoin cache stage.  It does not train D1, update an
encoder, construct a candidate axis, score the gallery, join a target, choose a
model, or make a scientific GO/NO-GO decision.

The complete predecessor is the pair:

- N2 E0 result
  `results/routea_matched_three_arm_n2_e0_v1/result.json`, SHA256
  `91b5685e2d3906dd86fac87705cefcb134d863248b2f9c989086feff6d1c4797`,
  status `ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_READY`;
- N2 E0 independent validation
  `results/routea_matched_three_arm_n2_e0_v1/independent_validation.json`,
  SHA256
  `6e2975fa61642d07654b876d44d24bb803d46d085b7f1cd236acc08aaa00f1b1`,
  status
  `ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_VALIDATED`;
- scope-correction addendum
  `plan/ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_ADDENDUM_V1_20260902.md`,
  SHA256
  `dd3b667907ecebce34aec4d37cbfb5520c4e432ef71a630b74b9fcffd8b00264`;
- scope-correction independent validation
  `results/routea_matched_three_arm_n2_e0_scope_correction_v1/independent_validation.json`,
  physical SHA256
  `c9ecdb33d94b68e8bcecf99384830b23e22b3c4eb0c593f828fdcb0b0bf8aa59`,
  logical SHA256
  `936cf64caa5ce47f808a3a981112deb7967583a690f78c631dc9e762da5cbe99`,
  status `ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_VALIDATED`.

Every predecessor must name this stage as its only successor.  This contract
inherits the correction: E0 proved all 64 image-token replays but only three
fresh current64 image-and-template replays.  This stage therefore does not
claim an E0 all-64 template replay.

## Fixed population and sharding

The sole membership authority is
`cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json`, SHA256
`df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec`.
It contains 987 unique query IDs and ordinals with tracks
`outcome=813`, `difficult=134`, `new_difficult_train=40`, folds
`{0:212,1:205,2:189,3:188,4:193}`, and grids
`36x20=813`, `24x32=162`, `32x24=11`, `25x29=1`.

There are exactly 16 immutable shards.  Shard `s` contains the contiguous
half-open ordinal interval

`[floor(987*s/16), floor(987*(s+1)/16))`.

This gives only 61- or 62-query shards, complete coverage, and no overlap.
Each shard is published by a temporary sibling directory followed by one
atomic rename.  A rerun may accept an already complete byte-sealed shard, but
may never overwrite or partially repair it.  A corrupt committed shard fails
closed.

## Current64 reuse and 923 fresh forwards

Exactly the 64 query IDs sealed by
`results/routea_d1_current_runtime_64_prejoin_v1/validation.json`, SHA256
`6c031e5a82b8057a48b48794d53dd76f4888b76ed03f081a925425c64e9621cb`,
are `CURRENT64_EXACT_REUSE`.  Their image and template tensors are copied
byte-for-byte from the sealed current64 shard payload, and their image-token
tensor, grid, source hash, decode frame and canonical-ledger join are replayed.
The copied template tensor is newly self-hashed and sealed; it is not presented
as an old bridge proof.

The other 923 members are `FRESH_CURRENT_RUNTIME_FORWARD`.  Every process loads
one shared current-runtime ColNomic encoder and processor, decodes the live
source whose bytes match the ledger hash, and creates both image and template
tokens.  `new_difficult_train` alone applies `ImageOps.exif_transpose` before
RGB conversion; `outcome` and `difficult` use the decoded raw frame.  Stored
image tokens are exactly `grid_h*grid_w x 128`; template tokens are `T x 128`;
both are finite contiguous float16 tensors with a dtype-and-shape-aware SHA256.

No historical 987/600 token cache is a numerical source.  No external or
sealed query is readable.

## Record schema and prejoin isolation

Each record contains only:

- query ID, canonical query ordinal, heldout fold and track;
- source-image SHA256, EXIF orientation, decode frame, raw/oriented image size
  and canonical grid;
- image and template tensors and their hashes;
- materialization mode and, only for reused members, the old current64 shard
  and payload SHA256;
- zero model-update count.

Source paths, filenames, target or exact labels, identity/supergroup fields,
candidate rows, gallery rows, slots, winners, challengers, outcomes, actions
and switches are forbidden in every record.  Query ID is an opaque query key,
not a trainable identity parameter.  No target insertion or target lookup is
possible in this stage.

## Model lineage and reference-defined boundary

The current-runtime encoder is the same shared ColNomic adapter used by E0.
Its stable model identity is bound by
`results/romav2_colnomic_new_difficult_model_lineage_v1/result.json`, SHA256
`6d5670561a2188deaea01f9c4373f2d4435ba45114307d95bcde555e5e7f2c82`,
whose ColNomic encoder fingerprint logical SHA256 is
`af1f3f83297ac2ab737a50704455fea1da10635b7b14cf101ba0dc02ec30191f`.
GPU model, node, runtime cache path and temporary directory are diagnostic and
must not enter the stable model identity.

The cache performs zero optimizer steps and zero parameter updates.  The
encoder is frozen.  D1 is not loaded or applied.  Its future schema remains a
single shared 49,792-parameter adapter with parameter-schema SHA256
`8594609ae4028a220fe3720a73545e1e6e654b79dda0ba0e39a78ea2541473f1`,
and the local A/B/C heads remain shared
seven-parameter functions.  No per-identity, per-gallery-row, per-candidate,
per-slot or per-reference trainable table is allowed.  Parameter names/shapes
cannot depend on gallery order, gallery size or enrollment.  Adding a new
reference may rebuild only its external token/cache/index entries; it cannot
change encoder, D1 or action-head parameters.

The aggregate and independent validation record the shared encoder parameter
schema hash and independently reconstruct the D1 parameter schema/count.
Neither schema may contain identity/gallery-row/reference-slot parameters.

## Independent validation

Each shard validator, without loading an encoder, independently reconstructs
the canonical interval and checks:

- exact record schema and forbidden-field absence;
- source bytes, raw/oriented size, EXIF/decode semantics and ledger grid;
- tensor dtype, shape, finiteness and byte hash;
- exact current64 image-and-template copy for reused members;
- immutable payload/receipt hashes, predecessor hashes and zero-access counts.

After all 16 shard validations, an aggregate seals the 987-query population,
64/923 mode split, fold/track/grid distributions, ordinal/query uniqueness,
per-shard hashes and the tensor-content manifest.

The final validator must not import either producer or aggregate program.  It
independently reopens all shard payloads, recomputes every schema/hash/population
check, then loads one shared encoder and fresh-forwards this preregistered
20-query audit set:

`DIFFICULT-0023, DIFFICULT-0079, OUTCOME-0005, OUTCOME-0028,
OUTCOME-0089, OUTCOME-0185, OUTCOME-0244, OUTCOME-0296, OUTCOME-0344,
OUTCOME-0392, OUTCOME-0450, OUTCOME-0568, OUTCOME-0604, OUTCOME-0675,
OUTCOME-0749, OUTCOME-0760, OUTCOME-0533, DIFFICULT-0128,
NDV2-007-P01, DIFFICULT-0025`.

The first 16 provide one fixed hash-selected query per shard; the final four
are all E0 fixtures.  Stored tensor hashes remain byte-exact integrity checks.
Fresh-forward semantic replay is accepted only when shape/dtype/grid/decode are
exact, maximum absolute token drift is at most `0.00390625`, and flattened
cosine similarity is at least `0.99999` for both image and template tensors.
Hardware identity is diagnostic only.

## Status and execution graph

Shard production and validation use an `accelerated` array `0-15%8`, one GPU
per task.  Aggregate plus final independent replay is a separate dependent
`accelerated` job.  The submission helper only submits this two-job graph; it
does not advance beyond validation.

The final PASS status is
`ROUTEA_N2_CURRENT_RUNTIME_987_TOKEN_CACHE_INDEPENDENT_VALIDATION_PASS`.
It authorizes only authoring/executing a fresh
`N2_CURRENT_RUNTIME_FOLD_LOCAL_D1_TRAINING_CONTRACT`.  It does not authorize
automatic training, label join, external access or a scientific claim.

Every artifact reports zero encoder updates, zero D1 loads/updates, zero action
head updates, zero target/identity/supergroup joins, zero candidate/gallery
semantic-field consumption, zero external reads and zero sealed reads.  The
implementation must also report the unavoidable eight current64 torch-artifact
deserializations separately: deserializing those historical payloads is not
misreported as zero access, while their RAW/D1 scores, ranks, C128 axes,
candidates, winners and actions remain semantically unconsumed.  Any mismatch
yields an ABORT with `next_authorized_stage=null`.
