# Route A RC: ColNomic-P x DINO-RCDE-V Connected-Superregion Mainline V1

Date: 2026-08-14  
Status: design and E0 implementation authority only  
Scientific claim: none  
Natural training authorized: no  
Opened/sealed access: forbidden

## 1. Decision

Connected superregion is restored as a **hard mainline evidence bottleneck**. It is
not a post-hoc visualization and it is not deferred until after an all-patch
candidate decision.

The frozen role split is:

```text
frozen ColNomic full-gallery prior
        -> natural target-free candidate set
        -> ColNomic candidate/reference local tokens (P only)
        -> candidate-conditioned multi-patch connected superregion seal
        -> learned DINO-RCDE query/reference tokens (V only)
        -> region-only independent verification
        -> candidate competition
        -> later, and only after a scientific GO, D1 HOLD/SWITCH
```

ColNomic supplies identity-sensitive proposal evidence. DINO-RCDE supplies a
different, learned dense correspondence field. A candidate can obtain local
evidence only through a sealed connected query footprint and its paired connected
reference footprint. Evidence outside those footprints is exactly zero.

The current V3 job remains valid only as an evidence-source/materialization
prerequisite. It does not itself implement this superregion scorer and cannot
authorize a target, ownership, or retrieval claim.

## 2. Why this is the required correction

The current DINO-RCDE comparator averages learned contributions over every valid
query patch. That can learn candidate-relative evidence, but it does not force the
decision to identify one coherent target explanation. Consequently background and
multiple-object evidence can still enter the same scalar.

The historical failures do not justify removing superregion:

- The old S8 family used a connected proposal, but compressed V into a few coarse
  mean/positive-mass statistics. Candidate-binding destruction was effectively
  unchanged. This falsified the old V/readout, not the connected-region premise.
- The DINO-Structure x ColNomic-Evidence D32 experiment used query-only DINO MST
  structure and never read candidate-reference DINO tokens. It produced giant or
  tiny regions and degraded `18/32` to `14/32` with `0 rescue / 4 break`. That
  query-only fixed heuristic remains permanently stopped.
- The prior ColNomic multi-support + Qwen verifier improved on a single seed but
  did not beat all-patch and failed candidate-binding/spatial controls. The
  proposal interface remains useful; the Qwen verifier/readout does not.

This V1 therefore combines only the non-falsified pieces: candidate-conditioned
ColNomic P, an explicit connected mask, and learned natural-pair DINO-RCDE V.

## 3. Non-negotiable target and supervision contract

- No human mask, box, point, polygon, target-derived pseudo-mask, or oracle
  candidate insertion.
- Proposal and inference are target-free. Target identity may enter only the
  fold-local natural exact-pair loss and offline evaluation after prejoin sealing.
- No identity-specific parameters.
- One patch is never target evidence. Every legal H1 must contain at least four
  patches, span at least two rows and two columns, and be exactly one 4-neighbour
  connected component in both query and reference.
- Disconnected components are never bridged, silently unioned, or filled. If two
  disconnected explanations both remain actionable, the state is
  `MULTI_COMPONENT_AMBIGUITY_HOLD`.
- H0/no-match is an explicit byte-exact zero path.
- Candidate reorder is an invariance/equivariance contract. Candidate-binding
  shuffle is a destruction control. They are never conflated.

## 4. Canonical geometry is a hard prerequisite

ColNomic and DINO grids cannot be aligned by token ordinal, reshape, fixed
transpose, bilinear resize, or grid-centre scaling.

Current evidence already contains at least these cases:

- ColNomic `24x32`, DINO `37x28`, EXIF orientation 6;
- ColNomic `24x32`, DINO `28x37`, EXIF orientation 1;
- ColNomic `36x20`, DINO `37x21`, EXIF orientation 1.

The common frame is the complete image after canonical EXIF orientation, expressed
in normalized pixel-boundary coordinates. Every backbone must supply a per-image
geometry receipt containing:

1. source-image SHA256 and query ID or physical gallery row;
2. raw dimensions, EXIF orientation, and raw-to-oriented transform;
3. frozen processor/config SHA256 and exact resize/crop/pad semantics;
4. patch grid shape, valid-patch mask SHA256, and the normalized oriented-image
   rectangle covered by every patch;
5. token SHA256 and candidate-axis/source-ledger binding.

A ColNomic footprint is mapped to DINO only by positive-area overlap of patch
rectangles in this common frame, followed by intersection with the DINO valid mask.
The mapped mask must independently pass connectivity, size, span, finite-boundary,
and source-SHA checks. Failure yields exact H0; code may not repair connectivity.

The current 600-query DINO ledger is fully covered by existing ColNomic spatial
query tokens. Of 4,748 DINO reference rows, 4,702 already have old ColNomic spatial
tokens. The 46 missing rows, including five rows needed by the current eight-query
E0 cohort, must be split from the frozen full 5,413-row ColNomic cache by replaying
the frozen CPU processor; no 7B re-encoding, zero fill, or row skipping is allowed.

## 5. P: candidate-conditioned connected proposal

For each anonymous candidate `g`, ColNomic P produces at most eight seed-local,
one-to-one multi-patch correspondences in both fixed checkerboard directions.
Scores, ranks, winner flags, D1 gaps, target labels, and DINO V values cannot cross
the P seal.

Each legal sparse H1 is converted to a dense paired footprint with the already
tested reference-to-query construction in `geometry_hypothesis_v1`:

1. form the connected reference convex-hull component around the initiating seed;
2. fit the frozen affine family from the one-to-one ColNomic anchor pairs;
3. project the **reference cell areas** into the query grid and retain only the
   projected seed component;
4. reject unless query and reference footprints are each a single 4CC, have at
   least four cells, have 2-D span, contain the retained anchors, and pass the
   already frozen fit/shape/area contracts;
5. map cell areas through the canonical geometry receipts into the DINO grids;
6. intersect with destination valid masks and rerun all structural checks;
7. merge two slots only when their query footprints touch/overlap **and** their
   reference footprints touch/overlap; otherwise retain separate components;
8. deduplicate by the joint query-mask/reference-mask hash, retaining all legal
   components and exact H0.

This reuses only the old geometry and seal, not its affine/cosine score. Fixed
local hypotheses can grow into a larger superregion only through overlapping or
touching candidate-consistent components; no gap is filled and no new halo is
invented. If the fixed footprint is structurally ineligible, V1 stops rather than
tuning geometry on the same heldout set.

P uses a small shared ColNomic-only head to rank the sealed legal population. For
candidate `g` and legal region `H`, let `A_g(H)` be the head's overlap-corrected
mean of candidate-relative ColNomic local summaries. To prevent larger region
banks from winning by multiplicity, candidate evidence is the fixed hierarchical
log-mean-exp

```text
U_g = log [ mean_scale mean_H_in_scale exp A_g(H) ].
```

The P loss on natural target/strongest-rival pairs is

```text
L_P = softplus(-(U_y - U_c)).
```

The deployed lock is the canonical MAP under the same scale-balanced population;
ties use the joint region hash. P parameters are shared across identities. The
target label enters only this fold-local loss, never the proposal inputs or
inference. A P score/confidence is discarded at the seal and cannot enter V.

## 6. V: learned region-only DINO-RCDE verification

P and V use different backbones and share only immutable boolean masks, geometry,
candidate/source keys, and hashes. They share no feature vector, parameter, score,
or gradient.

For a sealed hypothesis `H_g=(Q_g,R_g)`:

- DINO-RCDE decodes candidate `g` using only `Q_g x R_g`;
- a competitor `c` is evaluated on the same query region with its own sealed
  reference footprint;
- outside-region costs, assignments, relational rows, and patch contributions are
  exact zero;
- V retains signed patch contributions, positive and negative mass, cancellation,
  correspondence residuals, and concentration. It cannot be reduced to the old
  four S8 scalars.

Let `A_d(g,c;H_g^d)` be the learned RCDE pair evidence read only on candidate
`g`'s sealed query explanation in cross-fit direction `d`, with the two candidates'
sealed reference content. The deployed regional pair logit is

```text
Delta_SR(g,c) = 1/4 * sum_d [ A_d(g,c;H_g^d) - A_d(c,g;H_c^d) ].
```

Therefore `Delta_SR(c,g) = -Delta_SR(g,c)` by construction. The four terms have a
fixed denominator: a missing/illegal H1 contributes exact zero and may not be
removed by available-item renormalization. If both directions are H0, the action is
HOLD. P/V cross-fit uses fixed complementary checkerboards: ColNomic P-A selects
and DINO V-B scores; then P-B selects and V-A scores; the two directions are
averaged. Outer heldout data never select a hypothesis, backbone, threshold, or
checkpoint.

## 7. Training objective

Training is sequential and identity/supergroup cross-fitted, never joint:

1. fit the shared ColNomic P head on an inner-train identity set;
2. freeze P and materialize byte-sealed locks for inner-heldout identities;
3. concatenate only these OOF locks to train DINO-RCDE V;
4. for outer-heldout evaluation, fit P/V on outer-train data only, freeze both,
   and materialize every target-free candidate lock and score before label join.

There is no V-to-P gradient and no joint fine-tuning. This prevents P from choosing
a training-sample-specific favourable region that V can merely memorize.

The V loss is strongest-rival pairwise no-regret on the symmetric regional logit:

```text
L_pair = softplus(-(Delta_SR(q,y,c) - delta) / tau)
```

with `tau=1`. The margin `delta`, optimizer, update budget, sampler, and all loss
weights must be frozen in the natural-stage contract before label join. Correct
and incorrect ColNomic cases both train the shared verifier. There is no mask loss,
foreground loss, identity classifier, D1 imitation, or target-derived region loss.

The current DINO-RCDE checkpoint is the V initialization/evidence source. P and V
have disjoint learnable parameters. P receives only `L_P`; V receives only
`L_pair`. This contract does not authorize retraining either one yet. First the
region path must pass E0 and an optimization-only given-pair screen.

## 8. Fixed controls

All controls run with a frozen model and the same target-free P seal:

- `ALL_PATCH_SAME_MODEL`: same DINO-RCDE capacity without the region bottleneck;
- `C_COL_P`: shuffle candidate binding before P and rebuild the proposal;
- `C_DINO_V`: keep P byte-identical and replace only the DINO reference binding;
- `P_QUERY`: spatially permute DINO query content while keeping mask statistics;
- `P_REFERENCE`: spatially permute reference content inside the sealed footprint;
- `DISCONNECTED_MATCHED_AREA`: same patch count and marginal values split into
  disconnected pieces;
- `RANDOM_CONNECTED_MATCHED_SHAPE`: result-blind connected mask with matched shape;
- `SINGLE_SEED`: illegal as target evidence and retained only as a negative control;
- candidate reorder and pair swap: exact contracts, not destruction controls.

`C_DINO_V` must not rerun P. `C_COL_P` must not reuse the original P seal. Spatial
controls must preserve candidate binding. Otherwise the causal questions collapse.

## 9. Staged execution

### E0 -- structural and engineering qualification

Synthetic fixtures plus two result-blind natural geometry receipts only. Required:

- canonical geometry source/hash binding;
- positive-area rasterization and padding exclusion;
- exact one-4CC, minimum size, 2-D span, support containment;
- disconnected components never auto-bridged;
- outside-region contribution exact zero and scalar reconstruction closure;
- pair-swap antisymmetry `<=1e-7`;
- candidate-reorder closure `<=1e-7`;
- exact H0 and multi-component ambiguity HOLD;
- no labels, D1 scores, opened, sealed, target masks, or training.

Failure status: `RCDE_SR_E0_ENGINEERING_ABORT`.  
Pass status: `RCDE_SR_E0_READY`.  
Neither is a scientific conclusion.

### SR0 -- natural given-pair representation screen

Only after E0 and geometry/cache completion. Use optimization identities with
identity+supergroup inner OOF. Give the model the natural correct reference and the
natural strongest rival; do not change full-gallery scores.

The connected arm must strictly beat the same-checkpoint all-patch arm, the old
ColNomic/Qwen multi-support result, magnitude-only, and random-connected controls;
both candidate-binding and spatial destruction must lower group-balanced margin.
It must improve correct-over-rival direction without concentrating more than 50%
of positive mass on one patch. Exact gates and sample eligibility are frozen before
execution.

Failure status: `RCDE_SR_NATURAL_REGION_EVIDENCE_NO_GO`. This falsifies this fixed
P-to-region-to-V mechanism, not all connected target hypotheses.

### T1 -- target determination within natural C128

Only after SR0 GO. Apply the frozen regional scorer to all candidate hypotheses.
It must determine the correct reference more reliably than all-patch RCDE and the
frozen ColNomic prior, including on prior-wrong cases, with C/P degradation.

### A1 -- no-regret retrieval action

Only after T1 GO. D1 remains the frozen full-gallery base. The regional mechanism
may issue one SWITCH only when its target-free evidence passes frozen calibration;
otherwise byte-identical HOLD. Report rescue, break, wrong-to-wrong, and all matched
controls. Ownership is an interpretation of the winning connected `g-vs-c`
evidence, not a separate pre-target map.

## 10. Implementation boundary for the current turn

Authorized now:

- this additive mainline contract;
- a new geometry/connected-footprint/reducer module;
- E0 unit tests and local fixtures;
- read-only audit/materialization design for missing ColNomic geometry and five
  current reference rows.

Not authorized now:

- edits to files hash-bound to the running/pending V3 job;
- natural label join or training;
- full C128 scoring, D1 SWITCH/HOLD, opened/sealed access;
- changing the stopped S8 or D32 mechanisms and calling them new evidence.
