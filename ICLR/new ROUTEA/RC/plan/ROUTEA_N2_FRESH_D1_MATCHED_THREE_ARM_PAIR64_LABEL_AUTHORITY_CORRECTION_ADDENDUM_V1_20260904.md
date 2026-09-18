# Route A N2 fresh-D1 matched three-arm Pair64 label-authority correction addendum V1

## Purpose and precedence

This append-only addendum corrects the primary postseal label source in
Section 5 of:

- `plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_CONTRACT_V1_20260904.md`;
- contract SHA256
  `bc53ef74ec71283d79247a2117f25e2c708c2d30bb6fda1fad194b5668c97ade`.

The frozen mechanism, Pair64 roster, target-free preseal, three-branch pair
rule, 128 fresh RoMa evaluations, A/B/C features and current64 boundary remain
unchanged. Only the target-label authority and its consumption rules are
corrected here.

This addendum controls wherever the frozen contract names the CW0 RGH role
manifest as the primary Pair64 target source.

## 1. Primary label authority

The sole primary Pair64 target authority is the independently validated N2
fivefold label authority:

- result:
  `results/routea_matched_three_arm_n2_d1_label_join_authority_v1/result.json`;
- result SHA256:
  `519cea43968abf8083f3d6c4b98e108d592f29976c470b01f464054d7c9a5861`;
- result logical SHA256:
  `f65130fa5c9d5ee349485f466bb3fa9b098b3941db812b4d789363df2a12c604`;
- required status:
  `ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_READY`;
- independent validation:
  `results/routea_matched_three_arm_n2_d1_label_join_authority_v1/independent_validation.json`;
- independent-validation SHA256:
  `0b78e01d6940c19db40d606ce6012be190628b31d56e71851c22a97e83cfe3a8`;
- independent-validation logical SHA256:
  `a5c320ff698c8fd4e781976ddcc40ae10866d66a2efbb2b8c0623b9731b01fda`;
- required status:
  `ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_VALIDATED`;
- required query population: `987`.

The five validated fold payloads are:

| Fold | Payload SHA256 | Train queries | Record-manifest SHA256 |
|---|---|---:|---|
| 0 | `0a9c72582f0af05b9a83d28a6217ef62670a2a89b7bf433247e35e46d0e983f3` | 775 | `5d574dfef7572ea22ad2fc2fcd7dc3e1a6b67b45096419493313880051de0a88` |
| 1 | `c411861a4ca6b8f41792893f747d5f66561af718c6681b3d4c4bbece6b40ab72` | 782 | `fd8b76e42c91da651ae9c9d1d6197269e21d6d24f70b7a87c1c74da44f51ca22` |
| 2 | `08b89f23d61012460da86191951202dbdb4e7aa4cffb75e2c673e5bb8c5f9a0f` | 798 | `54788159dc855fe1a3d44cd6af0eac2b0e1dbfaebeefa27a74d61f1943195083` |
| 3 | `a559a678fe056353de607ba92b2bda1e675470be7a477ccdc54513c1395afb94` | 799 | `756f0d69cd6d8053645a7c6471b5bd484a2b0ea9e8f2e98e602ddcaae6abdd35` |
| 4 | `d6014cb1c141efd7cb57a19eed9692b406efddf8ef0f52449690e06b1090c6b3` | 794 | `747d857129686a01b794c2f8f42119a728b03f91b69f92065287e2377f974328` |

Every payload and the result/validation envelopes must replay physically and
logically before any Pair64 target is read.

## 2. Four-record consensus per Pair64 query

For a Pair64 query whose canonical held-out fold is `h`, that query must be
absent from fold `h`'s training payload and appear exactly once in each of the
other four fold payloads.

The four records must agree exactly on:

- `query_id`;
- canonical `query_ordinal`;
- canonical `heldout_fold`;
- `track`;
- corrected `target_identity`;
- `supergroup`;
- `target_physical_row`.

Therefore every one of the 64 Pair64 query IDs has exactly four non-heldout
label records, for exactly 256 record occurrences. Any missing, duplicate or
disagreeing record fails before pair selection.

The held-out-fold relationship is a consistency check only. Pair64 historical
`inner_fold`, Pair64 execution ordinal, Balanced execution ordinal and CW0
execution ordinal cannot route this join.

## 3. Corrected-identity membership rule

Fresh-D1 C128 membership and the three postseal states are determined by
corrected exact identity, not by equality to the authority's physical target
row.

For each C128 representative row, recover its corrected exact identity through
the independently validated 5,413-row to 5,412-identity reducer. Then:

1. if `target_identity` equals the fresh-D1 winner identity, use winner versus
   strongest wrong and negative HOLD;
2. if `target_identity` equals any non-winner C128 identity, use that C128
   representative row as the positive challenger;
3. otherwise retain target-absent and use winner versus strongest non-winner
   with negative HOLD.

`target_physical_row` is provenance for validating the target identity. It is
not inserted into C128 and cannot replace the natural representative row. This
is mandatory for corrected identities with multiple physical gallery rows.

## 4. CW0 is secondary cross-check only

The following CW0 artifacts are retained only as an independent label
agreement check:

- source manifest:
  `results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json`, SHA256
  `e0be35125eddec391a92398e84103e77ca0a65f661452b9190c2b81f3fdbea23`,
  logical SHA256
  `fd2681a4ceb88ea3a412952377ea7629b3ae48b7704f6ac6bf394f741856e0ac`;
- role manifest:
  `results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_manifest.json`, SHA256
  `2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454`,
  logical SHA256
  `9bcc4e84042d9cede6b8ee28ecf36f2937054f788c46184076759e08dbd75ffc`;
- result:
  `results/cw0_rgh_xf_v2_p0_a0_manifest_v2/result.json`, SHA256
  `1c3cd17e1349a84be7fb7af5d5b796874f3ec3d7a61539b3c6e535dddac04e08`,
  logical SHA256
  `b86c6d50fcddf140ddf44f47f2e434a59ddc335b6d5b7d3ad90f5f7f899adeab`;
- independent validation:
  `results/cw0_rgh_xf_v2_p0_a0_manifest_v2/independent_validation.json`,
  SHA256
  `ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac`,
  logical SHA256
  `6c157956369e0343eb71705575cbb5a31216aea0c759869aa00ececa2733f0c3`.

The 64 Pair64 query IDs must all occur exactly once in the 600 CW0 role shards.
For those 64, CW0 `identity` and `supergroup` must equal the N2 four-record
consensus. The observed frozen cross-check is `64/64` agreement for both
fields, and the later independent validator must recompute it.

CW0 cannot supply the primary target, target row, fold, source or candidate
axis. In particular these CW0 fields are forbidden for routing or pair
construction:

- `target_candidate_position`;
- `target_naturally_present`;
- `candidate_axis_sha256`;
- `execution_ordinal`;
- `inner_fold`;
- old prejoin source record;
- old candidate/member rows.

Those fields belong to the earlier CW0/RGH axis. Fresh target presence and
position are always recomputed from N2 `target_identity` on the sealed
fresh-D1 corrected-identity C128.

The CW0 role manifest itself states
`combined_role_runtime_read_authorized=false`; this addendum does not turn it
into the primary runtime label authority.

## 5. Target-free and postseal boundary

The Pair64 target-free base preseal remains unchanged and must complete before
opening any N2 label-authority payload or CW0 role shard.

After preseal validation, the postseal process may read the four N2 records per
Pair64 query only to form the consensus target identity/group/row and apply the
fixed three-state rule. It may then perform the already frozen 128 fresh RoMa
training-only computations.

Required access receipts include:

- `pair64_query_count=64`;
- `n2_label_record_occurrence_count=256`;
- `n2_nonheldout_records_per_query=4`;
- `n2_four_record_disagreement_count=0`;
- `cw0_secondary_query_count=64`;
- `cw0_identity_disagreement_count=0`;
- `cw0_supergroup_disagreement_count=0`;
- `cw0_target_candidate_position_consumption_count=0`;
- `cw0_execution_or_inner_fold_consumption_count=0`;
- `target_insertion_count=0`.

Changing any CW0 forbidden field must leave the N2-derived three-state pair
ledger unchanged. Changing one of the four N2 consensus labels must cause an
independent disagreement abort, not silently change a pair.

## 6. Claim and next-stage boundary

This addendum changes no model, target-free score, C128 axis, A/B/C definition,
RoMa computation count, loss, threshold or current64 result. It authorizes no
execution by itself.

A future Pair64 execution authority must bind this addendum, the frozen
training-input contract, both N2 label-authority envelopes, all five payload
hashes and the CW0 secondary evidence before postseal target access.

Successful Pair64 validation remains training-input preparation only. Its only
permitted successor is creation of
`N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT`; no optimizer
or head training is authorized here.
