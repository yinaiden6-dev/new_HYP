# RC-LTH P-only pre-execution overlap and lineage audit V1

Date: 2026-09-06  
Status: `PREEXECUTION_AUDIT_COMPLETE`  
Claim level: audit only; no new scientific result  
Automatic stage advance: false

## 1. Audited proposition

This audit covers only the Proposal proposition:

```text
(complete query q, arbitrary natural-C128 candidate reference g)
-> H_g = {H0, H_g1, ..., H_g8}
```

Every legal `H_gk` must contain a connected multi-patch query region, the
corresponding reference atoms/region, and replayable query-reference assignment
and geometry provenance.  H0 is explicit.  No V, retrieval action,
HOLD/SWITCH, or ownership claim is in scope.

The following three plans were read in full before this audit:

| File | SHA-256 |
|---|---|
| `plan/RouteA_Latent_Target_Hypothesis_最终方案.md` | `d921286719e66e10e113dfeb6d8c6bc33db6f8907f37240891b4eb9c5eef95d9` |
| `plan/RC_LTR_ASLO_XF_V3_COMPLETE_PLAN_20260805.md` | `b96ceeef4f3a71e1e19c935e97ebf4f3b8ea6498bb5b08eaa046c4baad587229` |
| `plan/CW0_RGH_XF_V2_FROZEN_SELECTED_SUPERREGION_NATURAL_P0_CONTRACT_V1_20260825.md` | `e57628ec796c89dbfc005a9ea2b8a5c43bf1b58ebdc521492658dfde6e8d9b84` |

The user-supplied joint-closure report was also read in full:

| File | SHA-256 |
|---|---|
| `reports/REPORT_ROUTEA_HYP_ROMAV2_CURRENT_EVIDENCE_AND_JOINT_CLOSURE_PLAN_20260906.md` | `dcacab8272efcadc7b12542efb3dbf40212290342bea0896896706e809b23457` |

That report describes a later deployment-level chain through D1-MI and
NATIVE7.  The present authority deliberately stops earlier: it freezes an
existing ColNomic natural-C128 membership and tests P only.  It does not wait
for, consume, or modify D1-MI, and it does not execute the report's V/action,
HOLD/SWITCH, deletion, or ownership stages.

## 2. Non-duplication decision

The complete new proposition has **not** been executed previously.

| Historical line | What it established | Why it is not this P proposition |
|---|---|---|
| G1 V3 / old HYP-XF | Eight homography slots and inlier counts | No connected query/reference region masks; no region-only candidate score; old producer/validator isolation is insufficient. |
| P-V2 / CW1 | A fixed query root/row bank and a selected row | The bank is query-side fixed geometry, not generated from arbitrary specific-reference RoMa/ColNomic atoms.  P-V2 can select a legal row even at a negative maximum. |
| CMPS | A shared 17-parameter selector over the same fixed CW1 bank | It does not generate the new atom bank; final natural OOF selector conclusion is not available. |
| RGH frozen-selected-superregion | Candidate-bound projected atoms, dual-end connected merge, provenance, exact H0 | The only completed selected-superregion artifact is synthetic E0.  The natural V8/V9/V9B line never materialized the selected region used for scoring. |
| RGH V9B | A one-fold natural pilot | It fused a uniform-atom marginal into a ColNomic base: `27/32 -> 26/32`, `0` rescue, `1` break.  It is not P-only and its score bypasses a selected connected region. |
| RGH AS1 | Six-parameter sparse selector synthetic E0 | `natural_data_read_count=0`; no natural RoMa/ColNomic atom bank, hard connected merge, provenance replay, P_COORD, all-patch, or query-only comparator. |
| RoMaV2+ColNomic frozen head | Candidate-bound no-regret correction and C_BIND dependence | The seven-parameter action head reads standardized raw gap and winner/challenger state.  It is an ordinary candidate decoder/action head, not HYP. |
| CGXF / R0-P3 | Connected paired-footprint synthetic E0 and a natural scale-only coverage screen | The engineering primitive exists, but the natural target H1 coverage was only `2/32`, rival H1 `2/32`, and dual H1 `0/32`; the frozen status was coverage NO-GO.  Its top-seed/mutual/scale-only proposer is not the new RoMa atom selector. |

The closest reusable historical engineering primitive is
`src/rc_aslo_xf/rgh_frozen_selected_superregion_v1.py`, which implements
dual-end component merging and provenance validation for CW0/RGH typed atoms.
It cannot consume RoMa atoms without importing the failed hard-geometry RGH
contract, so this audit permits reuse of its test ideas, not its scientific
result or its typed reducer.

## 3. Positive evidence inherited, with boundary

`results/romav2_colnomic_difficult90_frozen_regression_v1/result.json` is
accepted as positive mechanism evidence only:

```text
REAL:   61/90 -> 69/90, 8 rescue, 0 break
C_BIND: 61/90 -> 50/90, all 8 REAL rescues disappear
```

The result itself states:

```text
claim_level = OPENED_REGRESSION_ONLY_NOT_INDEPENDENT_EVIDENCE
candidate_binding_support = true
strict_spatial_causal_claim_authorized = false
target_label_read_count = 90
```

Therefore it supports only the premise that a specific reference contributes
additional information.  It is forbidden as a P input because its action
records contain target, rank, winner, correctness, gap, and postjoin fields.

The frozen bundle is audited as follows:

```text
isolated/romav2_colnomic_candidate_bound_no_regret_v1_20260901/bundle_manifest.json
physical SHA-256 = 8bff8ee124aad6d87946cc284e8827a60e4a130356d43c2f0b9443a6276298c8
logical SHA-256  = c545bf32933114e054c3c5eb3ee59aab33e1383864d0799fe2902127e4addee4
```

It is byte-identical to
`registry/romav2_colnomic_candidate_bound_no_regret_model_card_v1_20260901.json`.
Its RoMa checkpoint SHA is
`1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7`.
The model card and all referenced source/evidence hashes currently close.

Only the frozen encoders, preprocessing lineage, and the empirical fact of
candidate binding may be inherited.  The following frozen-head features are
not inherited by P:

```text
standardized_raw_gap
winner/challenger selection
HOLD/SWITCH logit and threshold
move-to-front ranking mutation
```

The separate balanced-pair RoMa V2 result is retained as representation
headroom, not as P qualification:

```text
results/romav2_colnomic_visibility_xf_balanced32_v2/result.json
pair correct = 29/32
REAL > query-coordinate control = 25/32
REAL > reference-coordinate control = 27/32
rescue / break = 14 / 1
```

It is a postjoin pair diagnostic (`full_c128_scoring_count=0`) and has neither
a full anonymous C128 proposal tournament nor connected H1 seals.

## 4. Negative evidence inherited

1. `results/dino_rcde_track_r_scientific_result_v1/result.json` records
   `DINO_SPECIFIC_REFERENCE_EVIDENCE_NO_GO`,
   `P_LOCK_REGIONAL_INCREMENT_NO_INCREMENT`, and `NO_SPATIAL_CLAIM`.
2. `results/rgh_v9b_balanced_fold2_oof_pilot_v1/result.json` records
   `RGH_V9B_BALANCED_FOLD2_OOF_ABORT`; base `27/32`, REAL `26/32`, rescue `0`,
   break `1`.
3. `results/rgh_v9b_reference_visibility_headroom_v1/result.json` records
   `both_direction_visibility_reprojection_atom_count=0`.  The old strict
   visibility/reprojection atom definition has no natural coverage headroom.
4. `results/rgh_as1_sparse_atom_selector_e0_v2/result.json` is
   `SYNTHETIC_ENGINEERING_SELECTOR_ONLY`, has `natural_data_read_count=0`, and
   remains `AS1_WAIT_HARD_GEOMETRY_COVERAGE_REPAIR`.
5. `results/r0_p3_scale_only_fixed32_coverage_gate_v1/formal_job5064412/result.json`
   is the closest previous natural connected-H1 coverage attempt and failed at
   target `2/32`, rival `2/32`, dual `0/32` with no threshold scan.
6. `results/cw0_rgh_xf_v2_all_legal_merge_diagnostic_v1/result.json` shows that
   merging all legal atoms is non-deployable: transitive components cover
   roughly `528--754` active reference cells.  A sparse selector cannot be
   replaced by merge-all.

These failures forbid importing the old DINO/CW0 hard geometry atom, the
uniform-atom marginal, or the AS1 pre-merge candidate score as the new P.

## 5. Frozen natural axis and clean-interface finding

The current-runtime bridge contains a useful fixed ColNomic natural-C128 axis:

```text
results/romav2_colnomic_current_runtime_bridge_prejoin_v1/shard00..07/payload.pt
validation status = ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_VALIDATION_PASS
query count = 64
target insertion count = 0
```

The 32 optimization records are all `inner_fold=1`, span 12 identities and 12
supergroups, and have natural target coverage `32/32`.  The 32 evaluation
records are all `inner_fold=2`, span 11 identities and 11 supergroups, and have
natural target coverage `32/32`.  Optimization/evaluation identity overlap and
supergroup overlap are both zero.

The evaluation population has already been adaptively opened by an older
seven-parameter head.  It may therefore support only a frozen **development
P0**, never a fresh or paper-level confirmation.  Its identity/supergroup
disjointness from the optimization bank remains a valid engineering and
development-generalization property for this new head.

However, the source payload also contains `candidate_raw_scores`,
`candidate_ranked_physical_rows`, and path strings.  It therefore cannot be
opened by the P runtime.  A separate candidate-axis projection process must
write a sanitized, content-addressed input containing only:

```text
opaque query/reference resource keys and content hashes;
query/reference pixels accessed only through bound resource locators;
ColNomic image tokens and canonical grids/coordinates;
the fixed natural-C128 membership as a canonical set;
physical-row key only as opaque reference binding provenance.
```

The projection process may read the candidate-source envelope in order to
erase fields.  The P process itself must report zero reads of raw score, rank,
slot, winner, correctness, target, label, filename/name stem, and identity.

The existing RoMa bridge JSON files are not sufficient: they retain only
candidate scalar scores and visibility-map hashes, not warp/cycle assignment,
atom tensors, connected regions, or replayable provenance.  RoMa must be rerun
from the frozen checkpoint and immediately compressed into a new P atom bank.

## 6. Reuse and non-reuse decision

Permitted engineering ideas/primitives:

- deterministic 4-connected-component validation;
- sparsemax arithmetic only, source-hash bound;
- hash-fixed, no-fixed-point, non-affine coordinate permutations;
- candidate reorder replay;
- exact H0 and fixed-slot padding tests;
- cell-level query-reference provenance closure.

Forbidden imports into the P core/runtime:

- `romav2_colnomic_frozen_gate_v1` and every D1/action module;
- `FrozenColNomicBaseV1` or any base score/rank loader;
- RGH V9/V9B candidate scoring and uniform-atom marginal;
- AS1's pre-connected-region candidate score;
- old G1 homography-only proposal;
- ColNomic-only rank/top-seed connected decoders;
- identity/filename-based gallery-label helpers;
- V, HOLD/SWITCH, ownership, opened/sealed, IMA++, or external dataset readers.

## 7. Protected parallel lineage

This P lineage uses new names prefixed `rc_lth_p_only` and does not modify,
overwrite, cancel, import, or depend on:

```text
results/routea_n2_d1_mi_*
plan/ROUTEA_N2_D1_MI_*
programs/*n2_d1_mi*
Job 5133448 or any successor from that dialogue
```

It also has a hard path deny for
`external/_staging/grozi120_v1`, including the `DO_NOT_CONSUME` marker.  IMA++
is not read or evaluated in this P lineage.

## 8. Audit decision

```text
OLD_EXPERIMENT_DUPLICATES_NEW_P_PROPOSITION = false
POSITIVE_CANDIDATE_BINDING_PREMISE_AVAILABLE = true
CLEAN_P_INPUT_ALREADY_AVAILABLE = false
OLD_NATURAL_CONNECTED_REGION_EVIDENCE_AVAILABLE = false
NEW_APPEND_ONLY_P_CONTRACT_REQUIRED = true
NEXT_ALLOWED_STAGE = WRITE_AND_VALIDATE_P_ONLY_CONTRACT
```
