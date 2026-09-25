# CW0-RGH-XF V2 Successor: Reference-Generated Connected-Superregion Contract V1

Date: 2026-08-25  
Status: `FROZEN_FOR_RGH_E0_ONLY`  
Scientific claim: none  
Natural training authorized: no  
DINO/GX execution authorized: no  
Automatic continuation: false

## 1. Decision and sole unresolved edge

This successor does **not** introduce another cross-image assignment module,
slot family, adapter, backbone, loss, verifier, or pooling scan.  Those objects
have already been implemented and, in several lineages, naturally trained.

The sole unresolved edge is:

```text
existing complete proposal-side soft query/reference assignment
-> weighted reference-to-query geometry
-> projection of reference cell areas into the query
-> candidate-specific paired connected superregion
```

For candidate physical row `g`, let `P^g_ij` be the already-defined reciprocal
soft assignment between query cell `i` and reference cell `j`, including its
shared dustbin and reliability terms.  For every frozen connected reference
atom `W^b_r`, define

\[
w^g_j=\rho^r_j\sum_iP^g_{ij},\qquad
\widehat x^g_j=
\frac{\sum_iP^g_{ij}x_i}{\sum_iP^g_{ij}+\epsilon}.
\]

The frozen weighted affine is

\[
T^g_b=\arg\min_T
\sum_{j\in W^b_r}w^g_j\|\widehat x^g_j-Ty_j\|^2
+\lambda\|T\|^2.
\]

The query hypothesis is generated only by projecting complete reference cell
areas and retaining the component containing the projected seed:

\[
Q^g_b=CC_4\left(
\bigcup_{j\in W^b_r}T^g_b(\operatorname{cellarea}(j)),
\operatorname{projectedSeed}(b)
\right).
\]

Every retained query cell must close to the exact provenance chain:

```text
candidate physical row
-> reference atom
-> reference cell
-> soft-assignment row/column
-> affine
-> projected reference cell area
-> query cell
```

No-provenance query cells are forbidden.

## 2. Parent lineages and non-duplication

### 2.1 Reused CW0 objects

The successor reuses, after fresh hash and semantic qualification:

- the shared 4,357-parameter query/reference adapter, reliability, and dustbin;
- complete bidirectional soft assignment and reciprocal mass;
- checkerboard A/B fit/verification endpoint isolation;
- weighted ridge affine and anti-collapse diagnostics;
- candidate reorder, pair swap, dense/streaming, and exact-H0 contracts.

The historical CW0 core E0 remains engineering evidence only.  Its natural
objective, natural S0, and P32 were not executed and are not inherited.

### 2.2 Reused August-14 geometry

The successor reuses, after fresh qualification:

- reference-cell-area projection;
- projected-seed single-4CC retention;
- canonical EXIF-oriented geometry;
- query/reference connectedness and two-axis span;
- no gap fill, dilation, morphological closing, or fabricated bridge;
- merging only when query and reference components are both compatible.

The historical hard-anchor E0 does not qualify a natural soft-assignment bridge.

### 2.3 Historical assignment and slot results remain binding

The following are permanent regression evidence:

- CG-XF hard prejoin: some candidate H1 on 32/32 queries, but correct-reference
  any-H1 on 7/32 and two-direction H1 on 3/32;
- LT-HYP/PVLock V3: fixed eight-slot/H0 verifier `NO_LEARNABILITY`;
- M0: fixed H0 plus eight H1 slots `NO_SIGNAL`;
- CW1-S8: variable H1 family and candidate-binding `NO_GO`;
- DINO-RCDE: learned 4-D assignment/dustbin was naturally trained, but its
  ALL/FULL/LOCAL Track-R scientific reduction was `NO_GO`;
- P-V2: fixed query bank and forced hard maximum can lock background regions.

Therefore assignment, dustbin, eight slots, H0, affine, 4CC, and DINO/GX are
not new claims.  The only tested increment is soft-assignment-to-reference-area
query geometry.

## 3. E0 access and isolation boundary

RGH-E0 is isolated from the V124 and GX workstream.

- It must not modify, cancel, overwrite, or consume job `5107346` outputs.
- It must not load or run DINO/GX during the proposal E0.
- It must write only under:
  - `src/rc_aslo_xf/rgh_xf_v2*`;
  - `programs/*rgh_xf_v2*`;
  - `tests/*rgh_xf_v2*`;
  - `results/cw0_rgh_xf_v2_*`;
  - `reports/figures/cw0_rgh_xf_v2_*`;
  - `registry/rgh_xf/`.
- No HOME file or base environment configuration may be modified.

Proposal prejoin reads of target/rival roles, labels, outcomes, RAW/D1 scores,
ranks, winner flags, gaps, opened/sealed resources, or human spatial annotations
must all be zero.

## 4. Fresh E0 input rule

E0 may use only a pre-existing, hash-bound complete assignment implementation
or artifact.  E0 introduces zero trainable parameters and zero updates.

If no compatible assignment source can be closed against current query and
reference token/geometry hashes, E0 must emit:

```text
RGH_E0_ASSIGNMENT_INPUT_ABSENT_ABORT
```

It may not initialize a replacement assignment, train a temporary model, copy
a V checkpoint into P, relax a hash, or fall back to independent cosine/LME.

A frozen DINO-RCDE assignment may be used only as a clearly named read-only
visual headroom diagnostic.  It cannot qualify proposal-side P because it was
trained for pair evidence rather than reference-generated geometry.

## 5. Hypothesis invariants

Each directional H1 must satisfy all of the following:

- query and reference are each one exact 4-neighbour connected component;
- source ColNomic reference atom has at least eight cells;
- mapped DINO query and reference masks each have at least four cells;
- both masks have two-axis span;
- effective assignment mass is at least eight;
- fit and verify endpoints each contain at least four cells;
- affine condition is below `1e4`;
- normalized RMSE is below `0.04` and maximum error below `0.08`;
- orientation is preserving;
- every projected query cell carries reference-cell provenance.

There is no dilation, gap fill, free graph growth, learned patch threshold, or
result-dependent retry.  Invalid hypotheses are exact H0.

E0 retains the complete frozen reference-atom axis.  It does not select or scan
top-K or slot count.  The historical eight-slot interface is a later deployment
compatibility object, not an E0 mechanism or scientific increment.

## 6. RGH-E0-A: fixture qualification

The fixture suite must verify:

1. new bridge parameter count is zero;
2. assignment input bytes are unchanged;
3. known-affine reference-area projection closes exactly;
4. EXIF-oriented geometry and cell areas close;
5. query/reference masks are exact 4CC with two-axis span;
6. no morphology, dilation, or bridge is called;
7. candidate reorder and pair swap close;
8. `C_BIND` is applied before assignment and rebuilds geometry;
9. `P_COORD` changes reference coordinates and rebuilds geometry;
10. dense/streaming error is at most `1e-7`;
11. source/token/geometry/provenance tampering fails closed;
12. all protected-access counters are zero.

Engineering failure is `RGH_E0_ENGINEERING_ABORT` and does not authorize a
natural visual run.

## 7. RGH-E0-B: natural read-only visual headroom

Exactly eight optimization-role queries are selected before role join by a
frozen hash/geometry/EXIF stratification.  Previously visualized queries and
groups must be recorded as opened diagnostics and cannot be presented as
untouched evidence.

For every selected query:

1. generate target-free RGH records for the complete natural C128;
2. freeze all candidate/atom/direction masks, geometry, assignment, and hashes;
3. only after freezing, join target and RAW-winner roles for diagnostic titles;
4. render target, RAW winner, and one hash-fixed random candidate;
5. render REAL, `C_BIND`, and `P_COORD` from independently rebuilt proposal paths;
6. render the matched historical P-V2 overlay without changing either path.

Each high-resolution visual must expose:

- raw query and specific reference;
- assignment mass and dustbin;
- fit and verify cells;
- reference atom and projected reference-cell boundaries;
- final query 4CC;
- reference-to-query provenance;
- H0 reason;
- matched old P-V2 proposal.

Visual appearance is diagnostic only and cannot enter any gate or parameter
choice.  No IoU is reported because no human target mask is available.

Successful completion is:

```text
RGH_E0_NATURAL_VISUAL_READY
```

This is not proposal GO, target determination, retrieval gain, or ownership.

## 8. P0 boundary

RGH-P0 requires a separate authority, natural objective completion, fresh
identity/supergroup-disjoint training, and a newly frozen 32-query panel that
excludes historical O32/P32 and manually inspected groups.

Minimum P0 gates are:

- joint two-direction correct-reference H1 coverage at least 26/32;
- zero single-patch, disconnected, or missing-provenance H1;
- target proposal score beats strongest natural rival on at least 21/32;
- group-balanced target-rival mean margin is positive;
- REAL beats C_BIND and P_COORD on at least 21/32 each, with positive
  group-balanced mean margins;
- target coverage does not fall below the fixed-bank P-V2 comparator and gains
  at least four net cases;
- candidate reorder and prejoin access audits pass.

Any P0 failure stops the lineage.  Top-K, slot count, radius, cap, temperature,
pooling, threshold, and backbone are not scanned on that panel.

## 9. V0 waits for frozen V124/GX

RGH-V0 is unauthorized until the repaired V124/GX checkpoint, decoder, reducer,
three-arm semantics, and null definitions are complete and byte-frozen.

One frozen checkpoint must later compare:

1. `ALL_PATCH_SAME_MODEL`;
2. `RGH_QUERY_REGION_FULL_REFERENCE`;
3. `RGH_PAIRED_QUERY_REFERENCE_REGION`.

The corrected P-V2/V124 three arms remain the fixed-bank matched baseline and
are never overwritten.  New RGH masks require new regional forwards and new
REAL/null materialization, but not a second V architecture.

## 10. Required result fields

Every stage result must contain at least:

```json
{
  "schema_version": "cw0_rgh_xf_v2_*",
  "stage": "E0_A|E0_B|P0|V0",
  "status": "...",
  "claim_level": "...",
  "next_authorized_stage": null,
  "automatic_stage_advance": false,
  "parent_contract_hashes": {},
  "assignment_lineage": {
    "source": "...",
    "checkpoint_sha256": null,
    "parameter_count": 4357,
    "new_parameter_count": 0
  },
  "source_bindings": {},
  "access_audit": {
    "prejoin_target_label_reads": 0,
    "prejoin_raw_d1_reads": 0,
    "dino_reads": 0,
    "gx_reads": 0,
    "opened_reads": 0,
    "sealed_reads": 0,
    "human_spatial_annotation_reads": 0
  },
  "hypothesis_counts": {},
  "h0_reason_counts": {},
  "provenance_closure": {},
  "control_metrics": {},
  "visualization_manifest_sha256": null,
  "scientific_GO_or_NO_GO": null
}
```

Every hypothesis record binds query/candidate keys, reference source SHA,
direction, reference atom, assignment hashes, affine and fit/verify statistics,
query/reference masks, per-query-cell reference provenance, geometry hashes,
REAL/C_BIND/P_COORD namespace, and an H1/H0 status with one exact reason.

## 11. Stop rules

- E0-A failure stops E0-B and P0.
- Missing compatible assignment input is an engineering abort, not a scientific
  assignment or geometry NO-GO.
- E0-B may complete as a visual diagnostic even when the regions look poor; it
  does not authorize result-driven tuning.
- P0 cannot start without a separately frozen natural objective and new panel.
- V0 cannot start while V124/GX semantics are still changing.
- No runner automatically advances a stage.
- Historical results are never overwritten or reinterpreted.

