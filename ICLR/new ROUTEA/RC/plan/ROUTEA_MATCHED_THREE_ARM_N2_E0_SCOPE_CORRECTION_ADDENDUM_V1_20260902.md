# Route A matched three-arm N2 E0 scope-correction addendum V1

## Purpose

This append-only addendum corrects one over-broad sentence in
`ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_V1_20260902.md`.
It does not modify E0 code or artifacts, relax a failed check, train a model,
or authorize external/sealed access.

Bound E0 artifacts:

- contract SHA256:
  `9c136e001334e82a885ee9aa8f247dd9b70a05a936edacce73db4d2cfcc61dcd`;
- payload SHA256:
  `9313e9290f4337c28e90fcbb2adcb8dc30ff2ef970c9202d9a1715e44ae0bca7`;
- producer result SHA256:
  `91b5685e2d3906dd86fac87705cefcb134d863248b2f9c989086feff6d1c4797`;
- independent validation SHA256:
  `6e2975fa61642d07654b876d44d24bb803d46d085b7f1cd236acc08aaa00f1b1`;
- required validation status:
  `ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_VALIDATED`.

## Correct replay scope

The original contract's lines 112--116 can be read as requiring all 64
template-token tensors to replay against the earlier query-only bridge. That
bridge has no template-token field, so that literal claim is unsupported.

The validated E0 evidence is exactly:

1. all 64 current-runtime queries replay raw image-token bytes, corrected C128
   axes, candidate scores, grid and canonical-ledger fields against their
   independently sealed authorities;
2. three fresh current64 fixtures (`OUTCOME-0533`, `DIFFICULT-0128`,
   `NDV2-007-P01`) replay both image and template token bytes against the
   current64 payload;
3. the uncovered `DIFFICULT-0025` `25x29` fixture replays both image and
   template token bytes across two independent fresh forwards;
4. the 987-query successor materializer must fresh-generate and seal both image
   and template tensors for every query. Its independent validator must replay
   tensor shape/dtype/hash and use a preregistered fresh-forward audit subset;
   it cannot inherit an all-64 template replay claim from E0.

E0 is therefore registered as
`ENGINEERING_PASS_WITH_SCOPE_CORRECTION`. No scientific conclusion changes.

## Fail-closed publication clarification

A producer failure may either leave the output absent or atomically publish an
ABORT with `next_authorized_stage=null`. Both are fail-closed. Only the existing
READY plus independent VALIDATED pair, followed by validation of this addendum,
may authorize the token-cache contract.

## Reference-defined, non-memorizing model boundary

The successor token cache and D1 training must preserve the following hard
invariants:

- gallery/query tokens are external data, not trainable identity parameters;
- D1 is one shared 49,792-parameter query adapter for every identity;
- the local A/B/C action heads are shared seven-parameter functions, with no
  per-identity, per-gallery-row, per-slot or per-reference trainable table;
- parameter names, shapes and count are invariant to gallery row order, gallery
  size and identity enrollment;
- identity/supergroup fields may enter isolated split construction and loss
  only; they cannot enter target-free inference features or parameters;
- adding a new reference may rebuild only reference tokens/cache/indexes and
  cannot update D1 or action-head parameters;
- candidate reorder must jointly reorder candidate payloads and leave scores
  equivariant; no row ordinal may serve as an identity parameter.

The 987 token-cache stage has zero model updates and cannot establish
generalization by itself. Later identity/supergroup-disjoint OOF and untouched
one-reference enrollment are required to test the reference-defined claim.

After independent validation, the only authorized successor remains
`N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT`.
