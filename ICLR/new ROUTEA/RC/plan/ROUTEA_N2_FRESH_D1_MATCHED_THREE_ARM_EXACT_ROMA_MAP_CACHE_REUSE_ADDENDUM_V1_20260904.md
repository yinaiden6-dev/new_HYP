# Route A N2 fresh-D1 matched three-arm exact RoMa-map cache reuse addendum V1

## Scope

This addendum creates one narrow exception to the sentence in Section 3 of
`ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_REMATERIALIZATION_CONTRACT_V1_20260904.md`
that forbids historical maps as fresh-D1 numerical inputs.  The parent
contract is SHA-256
`6d1eb7515ed739cb74a1d955c8fa077fc1033df8a7e1b06757b02c1eabdf54f3`.

The exception applies only to a RoMa map that is an exact cached evaluation of
the same identity-independent pair function:

```
(query image bytes, reference image bytes, canonical geometry,
 preprocessing contract, frozen RoMa checkpoint) -> (query map, reference map)
```

RoMa does not consume D1 tokens, scores, ranks, candidate slots, target labels
or action outcomes.  Reusing this exact function value therefore does not
reuse a historical D1 decision.  Every other prohibition in the parent
contract remains unchanged.

## Exact eligibility

A cached pair may be reused only when all of the following are identical and
hash-bound:

- unique `query_id` and physical reference row;
- query source image bytes and reference source image bytes;
- query and reference EXIF/decode frames;
- query and reference canonical grid geometry, valid-patch masks and cell
  coordinates;
- ColNomic preprocessing configuration, SHA-256
  `1a427e12a15406a5c4701273c91a0b319528de5c78523ae5ba5329d86e5d5557`;
- frozen RoMa-v2 checkpoint, SHA-256
  `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`;
- map dtype, shape, finite range, query-map SHA-256 and reference-map SHA-256;
- the complete historical target-free map-shard payload seal.

The frozen historical producer and validator are:

- `programs/materialize_routea_d1_current_runtime_matched_three_arm_shard_v1.py`,
  SHA-256
  `ca10470f271da2d6fef5cd59d11ad930c5d9045774774d2caac7df8d1b339383`;
- `programs/validate_routea_d1_current_runtime_matched_three_arm_shard_v1.py`,
  SHA-256
  `bd4cda9d350796cb6a7a7f2ba3c26047254a9566ea481e20338ad49ea0e9008d`;
- aggregate validation
  `results/routea_d1_current_runtime_matched_three_arm_prejoin_v1/validation.json`,
  SHA-256
  `60bd07480bf4963503993ecb926b696e3fc568c588189c557c0bc9975f32c66f`;
- canonical geometry implementation
  `src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py`, SHA-256
  `c732dc933c502f17fd5cb8cc80ae45d116716fe801cc8f3034a1e6a49cef6508`.

Any mismatch makes that pair ineligible for reuse.  It must be recomputed
through the frozen RoMa path or the stage must fail closed; it may never be
dropped, substituted or assigned a nearby map.

## Preregistered current64 population

This exception is fixed before current64 feature reduction:

- fresh-D1 candidate pairs: `64 * 128 = 8,192`;
- exact eligible cached pairs: `8,168`;
- missing pairs requiring frozen RoMa recomputation: `24`;
- queries containing a missing pair: `17`;
- fresh current64 reference union: `2,805` physical rows;
- byte-reusable reference rows: `2,801`;
- missing reference rows: exactly `467, 1488, 2435, 5159`.

The 8,168/24 partition is derived only from the sealed fresh-D1 C128 axes and
the target-free old pair-key set.  It cannot be changed after labels or
retrieval outcomes are read.  The complete missing-pair list and the complete
reuse-pair hash ledger must be published by the source-only manifest.

## What remains forbidden

This exception does not authorize reuse of any historical:

- RAW, old-D1 or union candidate axis;
- base score, target score, rank, winner, challenger or slot;
- adapted query token;
- A/B/C arm score, visibility mass, normalized score or roll response;
- common/native feature tensor;
- C_BIND destination/source ledger;
- trained head, weight, bias, logit, threshold, HOLD/SWITCH action or metric.

All of those quantities must be reconstructed from the sealed fresh-D1
adapted tokens, full-gallery scores and natural C128.  C_BIND is rebuilt on
the fresh candidate positions using the fixed shift by 64.

## Target-free validation contract

Before a cached map is consumed, the source-only manifest and independent
validator must independently reconstruct the full 8,192-pair partition by
`query_id` and physical row.  For every reused pair they must verify the old
shard seal, both map hashes, both map lengths and the exact reference-token
hash.  For every missing pair the later prejoin validator must rerun the
frozen RoMa pair function, verify geometry and finite map ranges, and compare
the resulting maps under a declared numerical replay tolerance.  The replay
tolerance is validation-only and may not change a map, feature or score.

The validator may reuse schema/hash helpers but must not import the producer.
It must report target, supergroup, retrieval-outcome and action semantic reads
as zero; target insertion, model updates, external reads and sealed-test reads
are also zero.  Candidate reorder and C_BIND remain separate controls.

## Claim and stage boundary

This addendum covers only the 64-query current64 staging branch.  Pair64 is
still pending (`pair64_pending_count=64`) and
`contract_population_complete=false`.  Passing this cache check does not
produce the parent contract's final 128-query PREJOIN status, authorize a
label join, train a head, establish candidate recall, establish ownership or
support a scientific GO/NO-GO.

The only next stage permitted here is current64 reference/map completion and
current64 target-free prejoin validation.  A separate Pair64 contract and a
complete 128-query aggregate remain mandatory.
