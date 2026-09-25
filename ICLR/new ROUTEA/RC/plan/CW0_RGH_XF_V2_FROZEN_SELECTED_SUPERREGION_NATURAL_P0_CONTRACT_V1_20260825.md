# CW0-RGH-XF V2 Frozen-Selected-Superregion Natural P0 Contract V1

Date: 2026-08-25  
Status: `FROZEN_DESIGN_ONLY_NOT_EXECUTION_AUTHORITY`  
Parent contract: `CW0_RGH_XF_V2_SUCCESSOR_REFERENCE_GENERATED_SUPERREGION_CONTRACT_V1_20260825.md`  
Interpretation addendum: `CW0_RGH_XF_V2_E0_ATOM_VS_SUPERREGION_CLARIFICATION_ADDENDUM_V1_20260825.md`  
Scientific claim: none  
Natural training authorized by this document: no  
DINO/V124/GX execution authorized by this document: no  
Automatic continuation: false

## 1. Decision and exact missing mechanism

The existing RGH E0 artifact has established only the following object:

```text
RGH_PROJECTED_ATOM
specific reference atom
-> existing soft assignment
-> fitted reference-to-query affine
-> projected connected query atom
```

It has not established a selected target region.  Merging every structurally
legal atom is permanently a non-deployable diagnostic: on the opened canary it
formed one transitive component per candidate and direction and covered almost
the complete reference.

This contract adds the missing natural proposal stage:

```text
natural exact pair + all natural C128 rivals
-> train the existing 4,357-parameter CW0 assignment only
-> held-out atom score relative to an independent matched coordinate null
-> freeze visible atoms and exact H0 before role join
-> merge only visible atoms that are compatible on both endpoints
-> seal every separated connected component for a future V marginal
```

This is not authority for another selector head.  It introduces:

- no new backbone;
- no new adapter;
- no new atom-quality head;
- no slot embedding or identity parameter;
- no top-K, learned threshold, or pooling scan; and
- no DINO, V124, GX, RAW/D1, SWITCH, or HOLD input.

The only trainable object is the inherited CW0 assignment core:

```text
rank-8 asymmetric query/reference residual adapter
+ query reliability head
+ reference reliability head
+ one shared dustbin scalar
= exactly 4,357 trainable scalars
```

Its parameter count, initialization, descriptor temperature, and assignment
equations are inherited and may not be enlarged under this contract.

## 2. Existing assignment is dual-softmax, not Sinkhorn

For candidate `g`, the inherited assignment computes query-to-reference and
reference-to-query probabilities with separate dustbins.  Reciprocal pair mass
is the product

\[
P^g_{ij}=A^g_{ij}B^g_{ji}.
\]

This is a dual-softmax partial-assignment construction.  It is **not** a
doubly-stochastic transport plan and must not be described as Sinkhorn or
optimal transport.  Query and reference dustbins allow unmatched mass, but do
not by themselves establish natural absence or candidate-level H0.

No replacement matcher is authorized.  E0 must byte-bind the existing CW0
assignment equations and prove that RGH consumes `reciprocal_pair_mass`,
reliability, and dustbin outputs with no independent cosine/LME fallback.

## 3. Fixed reference atoms and cross-fit directions

The complete result-blind reference atom population is inherited from CW0:

```text
one seed per fixed 4x4 reference macro tile
canonical row-major seed
radius = 6
cap = 64 reference cells
minimum = 8 reference cells
one exact reference-side 4CC with two-axis span
```

Every atom remains in the fixed uniform atom axis.  Structural failure produces
an exact-zero H0 atom at the same ordinal; it is not removed from the marginal.
No seed score, label, rank, top-K, NMS, or hash deduplication changes this axis.

For atom `b`, the fixed reference checkerboard creates two directions:

```text
A_TO_B: fit on reference A; score on held-out reference B
B_TO_A: fit on reference B; score on held-out reference A
```

The complete soft assignment is sealed once before atom scoring.  The fixed
reference-cell rows are then partitioned into fit and held-out banks without
renormalizing either bank.  A reference descriptor/logit/reliability row from
the held-out bank may not enter the affine fit.

The checkerboard split prevents literal reuse of the same reference endpoints
in fit and score.  Query endpoints are shared through the complete soft
assignment.  It is **not statistical independence**: neighbouring image
patches are correlated, and both banks originate from the same natural image
pair.  Results may therefore be called reference-endpoint-cross-fit, not an
independent dataset test.

## 4. Reference-generated geometry

For direction `d`, let `F_d` and `V_d` be its fit and held-out reference banks.
For every `j` in `W_r^b intersect F_d`, define

\[
w^{g,d}_j
=
\rho^{r,g}_j\sum_{i\in Q}P^{g}_{ij},
\qquad
\widehat x^{g,d}_j
=
\frac{\sum_{i\in Q}P^{g}_{ij}x_i}
     {\sum_{i\in Q}P^{g}_{ij}+\epsilon}.
\]

The inherited weighted ridge affine is

\[
T^{g,d}_b
=
\arg\min_T
\sum_{j\in W_r^b\cap F_d}
w^{g,d}_j\|\widehat x^{g,d}_j-Ty_j\|^2
+\lambda\|T\|^2.
\]

The complete reference-cell areas in `W_r^b`, not only high-mass endpoints,
are projected through `T`.  The query atom is the single 4CC containing the
projected reference seed.  Every retained query cell must retain exact
reference-cell provenance.

Structural H1 requires, in each direction:

- one reference 4CC and one projected query 4CC;
- two-axis span on both endpoints;
- at least four unique fit endpoints and four unique held-out endpoints;
- reciprocal effective mass at least eight;
- source and target covariance minimum eigenvalue at least `1e-4`;
- affine condition below `1e4`;
- held-out normalized RMSE below `0.04`;
- held-out maximum error below `0.08`;
- orientation preserving; and
- complete query-cell reference provenance.

There is no area cap in this contract.  There is also no dilation, gap fill,
morphological closing, free graph growth, or result-dependent retry.

## 5. Held-out signed atom score

For a structurally legal atom and direction, score only held-out reference
cells in `W_r^b intersect V_d`.  Let the inherited signed held-out cell evidence
include reciprocal return, cycle/reprojection response, dustbin failure,
conditional entropy, and explicit query/reference reliability.  Evidence is
mapped by the frozen bounded function below before reliability multiplication
and reduced with the complete held-out-cell denominator:

\[
S^{g,d}_{b,\mathrm{REAL}}
=
\frac{1}{|W_r^b\cap V_d|}
\sum_{j\in W_r^b\cap V_d}
\rho_j e_j.
\]

The immutable historical CW0 hard clamp cannot be reused unchanged here.  A
pre-training natural plumbing canary showed that its REAL and matched-coordinate
paths both saturated at the same lower bound on every atom, producing exact
zero contrast and zero gradient before any natural label was opened.  RGH V2
therefore freezes the bounded monotone polynomial soft clip

\[
\operatorname{softclip}_4(x)=\frac{4x}{4+|x|}.
\]

It preserves sign and order, stays strictly within `(-4,4)`, and has nonzero
derivative for every finite input.  This is a preregistered learnability repair
inside the new RGH atom score; it does not overwrite or reinterpret historical
CW0 results.  E0 must retain the failed hard-clamp replay as a regression test
and prove that natural REAL/P_TRAIN contrasts and all 4,357 parameter gradients
are no longer identically zero.

Low reliability may reduce this signed evidence but may not delete a held-out
cell or renormalize the remaining hotspots.

During natural optimization, every fixed atom whose arithmetic remains finite
keeps this differentiable signed surrogate even when its detached hard geometry
gate is not yet satisfied.  The detached hard state is receipted but does not
erase the training gradient.  This is required to avoid repeating the historic
`7/32` dead loop in which an initially illegal correct-reference hypothesis
could never improve.  After the assignment checkpoint is frozen, an atom that
fails any structural H1 condition becomes exact-zero H0 with one
machine-readable reason and cannot be visible or enter a merge.

## 6. Independent training matched-coordinate destruction

Raw `S_REAL` has no identified absolute zero.  A separate, result-blind
training coordinate destruction `P_TRAIN` supplies a matched reference point.

For every query, candidate, atom, and direction, `P_TRAIN`:

1. preserves query/reference token tensors, assignment values, dustbin values,
   reliability values, atom membership, valid masks, grid shape, and coordinate
   multiset;
2. changes only the association between reference assignment rows and reference
   coordinates using the inherited hash-fixed, no-fixed-point, non-affine
   spatial control;
3. uses a `TRAIN` namespace and realization fixed before label join;
4. reruns affine fitting, reference-area projection, held-out scoring, and H0;
5. never retries a different permutation when the controlled atom becomes H0;
   and
6. is different from every later evaluation-coordinate-destruction namespace
   and realization.

Training/evaluation spatial separation must be independently validated using
the existing phase-separated spatial-control contract.

Define

\[
u^{g,d}_b
=
S^{g,d}_{b,\mathrm{REAL}}
-S^{g,d}_{b,P_{\mathrm{TRAIN}}}.
\]

`u` is named **matched-null contrast**.  It is not a likelihood ratio, log
likelihood ratio, calibrated probability, foreground score, segmentation
score, ownership probability, or proof of target presence.

For differentiable natural training, the two-direction atom score is the
normalized smooth minimum at the already inherited `tau=0.10`:

\[
m^g_b
=-	au\log\left[
\frac{
e^{-u^{g,A\to B}_b/\tau}+e^{-u^{g,B\to A}_b/\tau}
}{2}
\right].
\]

An explicit all-zero branch returns positive-zero bit-exactly.  This smooth
surrogate gives both directions gradient; it does not define deployment
visibility.

Consequently, after the assignment checkpoint is frozen, atom `b` is visible
if and only if:

```text
both directional structural states are H1
and u_A_TO_B > 0
and u_B_TO_A > 0
```

Zero is the fixed mathematical boundary.  It is not scanned or calibrated on
P0.  Exact ties belong to H0.

## 7. H0 and fixed-uniform-atom MIL marginal

For candidate `g`, let `B_g` be its complete fixed reference-atom count,
including structural H0 atoms at their original ordinals.  Candidate evidence
uses a fixed `0.5` H0 prior and a fixed `0.5` uniform prior over the complete
atom axis:

\[
G_g
=
\tau
\log\left[
0.5e^{0/\tau}
+0.5\frac{1}{B_g}\sum_{b=1}^{B_g}e^{m^g_b/\tau}
\right],
\qquad \tau=0.10.
\]

A dedicated all-zero branch must return positive-zero bit-exactly when every
`m_b` is zero.  Uniform replication of the complete atom axis must leave the
score invariant.  Candidate-specific removal of H0 atoms, available-item
renormalization, positive clamp, top-K, max pooling, or learned atom prior is
forbidden.

This marginal is a multiple-instance-learning construction: exact-pair labels
supervise the candidate bag, while atom membership remains latent.  It does not
provide atom-level ground truth and cannot by itself prove that the MAP or
visible atom is the physical target.

## 8. Natural full-C128 training objective

### 8.1 Pre-loss target-free forward

For every training query, the complete natural C128 axis is reconstructed in
canonical physical-row order.  The model computes `G_g` for all 128 anonymous
candidates before exact-label role selection.

The target is never inserted.  If its exact label is naturally absent from
C128, the query is recorded as `TARGET_NATURALLY_ABSENT_FROM_C128` and is
ineligible for the retrieval loss.  No retry, wider candidate set, or donor
replacement is permitted.

Physical rows sharing the target exact label are target-equivalent, not rivals.
Every different-label natural C128 member remains in the rival sum.

### 8.2 Listwise rank loss

After the anonymous forward, the exact label selects target-equivalent rows and
different-label rivals.  If several physical rows carry the target label, their
candidate evidence is reduced by the same fixed uniform log-mean-exp rule.

The full-C128 rank loss is

\[
L_{\mathrm{rank}}
=
\log\left[
1+\sum_{c:\ell_c\ne y}
\exp(G_c-G_y)
\right].
\]

This objective directly exposes the assignment to the complete natural rival
population rather than a detached single strongest rival.

### 8.3 Target-above-H0 loss

The correct candidate must have evidence above the exact H0 zero point:

\[
L_{\mathrm{target>H0}}=\operatorname{softplus}(-G_y).
\]

No loss forces every wrong candidate to H0.  A natural query may contain a
second gallery drug box or another legitimately explainable candidate; those
candidate hypotheses remain representable.  The retrieval label identifies the
primary exact-instance target, not universal absence of every rival.

### 8.4 Independent training candidate-binding contrast

An independent `C_BIND_TRAIN` namespace constructs one complete fixed-point-free,
corrected-identity-disjoint donor permutation over the natural C128 reference
axis before assignment.  It preserves the destination candidate axis and query,
but replaces complete reference token/content/geometry binding and reruns the
entire assignment-to-atom path.

It must not shuffle already sealed atoms or scores.  Its namespace, donors, and
hashes are independent of evaluation C_BIND.

For the target destination define

\[
L_{C\_BIND}
=
\operatorname{softplus}
(G^{C\_BIND\_TRAIN}_y-G^{\mathrm{REAL}}_y).
\]

The once-frozen objective is

```text
L = 1.00 L_rank
  + 1.00 L_target>H0
  + 1.00 L_C_BIND
```

No other loss, weight, margin, entropy regularizer, atom sparsity penalty, or
threshold is authorized by this contract.  Optimizer, schedule, update budget,
sampler, and fold manifests require a later execution authority and must be
frozen before natural training.

## 9. Freeze-before-join evaluation path

After fold-local assignment training:

1. freeze the 4,357-parameter checkpoint and its complete state hash;
2. for every held-out query, materialize all natural C128 REAL assignments,
   `P_TRAIN` contrasts, structural states, `u` values, and visibility without
   reading labels, ranks, outcomes, or RAW/D1;
3. materialize separate evaluation C_BIND and coordinate-destruction paths
   under namespaces that were never used in training;
4. freeze every atom record and candidate component seal;
5. independently reconstruct the full candidate axis, controls, hashes, and
   protected-access counters; and
6. only after the validated prejoin population seal exists, join exact labels
   for P0 metrics and titles.

An evaluation producer may not open the postjoin role ledger and promise not to
use it.  The role ledger must be deferred and physically unopened until the
prejoin population seal is committed.

## 10. Frozen visible-atom merge

The merge input is exactly the set of frozen visible atoms.  Structurally legal
but non-visible atoms cannot act as bridges.

Two visible atoms are adjacent only if all of the following hold:

1. their query masks touch or overlap by 4-neighbour adjacency in both
   directions;
2. their reference masks touch or overlap by 4-neighbour adjacency;
3. both directional affine families are compatible; and
4. candidate, query, reference source, checkpoint, direction schedule, and
   geometry receipts are identical.

Affine compatibility is inherited from the CW0 direction-agreement scale.  For
each direction, evaluate both affine maps on the complete union of the two
reference atoms.  The RMS disagreement must be strictly below two normalized
query-cell diagonals.  No compatibility threshold is scanned.

Every connected component of this dual-end, affine-compatible graph is merged
independently.  For each component and direction:

- query mask is the exact union of its visible query atoms;
- reference mask is the exact union of its visible reference atoms;
- the union must independently remain one 4CC with two-axis span;
- all query-cell provenance edges are retained; and
- source atom ordinals are canonical and identical across the two directions.

If there are several separated components, all of them are sealed.  P does not
choose one, average their masks, bridge them, or convert them into ambiguity H0.
A future independently frozen V may marginalize the component axis.  This
contract does not authorize that V marginal.

Candidate H0 occurs only when no visible component survives exact structural
validation.  There is no query/reference area cap.

The only deployable proposal artifact name authorized by this stage is:

```text
RGH_FROZEN_SELECTED_SUPERREGION
```

Each seal contains the complete component population, not one post-hoc chosen
component.

## 11. Required immutable records

Every target-free atom record must bind:

- query ID, execution ordinal, candidate position, physical row, and source SHA;
- fold-local checkpoint SHA and assignment implementation SHA;
- reference atom ordinal and immutable membership hash;
- direction and exact fit/held-out endpoint IDs;
- REAL, P_TRAIN, evaluation-P, and C_BIND namespace hashes;
- reciprocal assignment, dustbin, reliability, and coordinate-binding hashes;
- affine matrix, effective mass, covariance, condition, orientation, held-out
  RMSE/max error, and held-out signed score;
- REAL and P_TRAIN scores, matched-null contrast `u`, structural H1/H0, and
  frozen visibility;
- query/reference masks and cell-level provenance hash; and
- one exact H0 reason when ineligible.

Every candidate seal must bind:

- all atom record hashes in canonical order;
- fixed uniform atom count and H0 marginal receipt;
- visibility bit vector;
- dual-end compatibility graph hash;
- every separated merged component and its source atoms;
- component query/reference mask and provenance hashes for both directions;
- candidate-level `G_g` and exact all-zero closure;
- candidate reorder receipt; and
- zero protected-read counters.

Required top-level fields include:

```json
{
  "schema_version": "cw0_rgh_xf_v2_frozen_selected_superregion_*",
  "stage": "E0|S0|P0_PREJOIN|P0_POSTJOIN",
  "status": "...",
  "claim_level": "...",
  "training_parameter_count": 4357,
  "new_backbone_count": 0,
  "new_head_parameter_count": 0,
  "candidate_count": 128,
  "target_insert_count": 0,
  "prejoin_target_role_read_count": 0,
  "prejoin_raw_d1_read_count": 0,
  "dino_read_count": 0,
  "gx_read_count": 0,
  "v124_read_count": 0,
  "opened_read_count": 0,
  "sealed_read_count": 0,
  "scientific_GO_or_NO_GO": null,
  "automatic_stage_advance": false,
  "next_authorized_stage": null
}
```

## 12. E0 gates

E0 uses synthetic fixtures plus a result-blind natural plumbing canary.  It
does not train on natural labels.

It passes only if:

1. exactly 4,357 assignment parameters exist and the new bridge/merge has zero
   trainable parameters;
2. dual-softmax, reciprocal mass, reliability, and dustbin arithmetic match the
   inherited implementation;
3. known-affine REAL atoms have positive matched-null contrast in both
   directions, while coordinate destruction lowers the score;
4. all-zero atom contrast and candidate marginal are positive-zero bit-exactly;
5. fit and held-out reference-endpoint populations are exact-disjoint and
   held-out rows actually enter `S`, not only a receipt;
6. a fixture whose useful atom is not initially maximal provides finite,
   nonzero gradient to the inherited assignment parameters;
7. every one of the 4,357 parameters receives finite gradient and one update
   changes the assignment state;
8. TRAIN and EVALUATION coordinate controls are distinct, no-fixed-point, and
   independently reconstructed;
9. C_BIND is applied before assignment over a complete candidate axis;
10. visible-only merge cannot use a non-visible atom as a transitive bridge;
11. multiple separated components are all retained with no forced winner;
12. candidate reorder, fresh/resume, dense/streaming, source, geometry,
    provenance, and access contracts pass; and
13. DINO/GX/V124 reads and all protected prejoin reads are zero.

Pass status:

```text
RGH_FROZEN_SELECTED_SUPERREGION_E0_READY
```

Failure statuses distinguish:

```text
RGH_E0_ASSIGNMENT_INPUT_ABORT
RGH_E0_MATCHED_NULL_DIRECTION_ABORT
RGH_E0_HELDOUT_SCORE_NOT_CROSSFIT_ABORT
RGH_E0_H0_MARGINAL_ABORT
RGH_E0_DUAL_END_MERGE_ABORT
RGH_E0_ACCESS_OR_REPLAY_ABORT
```

All are engineering/learnability aborts, never scientific RGH NO-GO.

## 13. Natural S0 and P0 gates

E0 only authorizes a separately hash-bound natural execution authority.  S0
must first prove, on two optimization-role queries with complete natural C128:

- target is naturally present or recorded absent without insertion;
- all 128 candidates receive target-free REAL/P_TRAIN/C_BIND records;
- full-C128 listwise loss, target-above-H0, and C_BIND loss are finite;
- RAW-correct and RAW-wrong examples both provide assignment gradient;
- every assignment parameter has finite gradient;
- checkpoint/resume and protected-access receipts close; and
- no V/GX/V124 input is read.

Only then may a frozen, identity-and-supergroup-disjoint 32-query development
panel be evaluated.  P0 requires all of:

1. the correct-reference candidate has at least one frozen selected component
   on at least `26/32` queries;
2. zero selected atoms violate dual-direction positivity, structural H1,
   provenance, or candidate/source binding;
3. target candidate `G_y` exceeds the strongest different-label natural C128
   rival on at least `21/32` queries;
4. group-balanced target-minus-strongest-rival mean is positive;
5. independent evaluation C_BIND lowers target-versus-rival margin on at least
   `21/32`, with positive group-balanced REAL-minus-control mean;
6. independent evaluation coordinate destruction lowers target-versus-rival
   margin on at least `21/32`, with positive group-balanced REAL-minus-control
   mean;
7. every split has more target-rank improvements than degradations versus the
   frozen raw-assignment initialization;
8. candidate reorder, exact-label mutation after prejoin, fresh/resume, and
   independent full-C128 reconstruction pass; and
9. no top-K, zero threshold, atom count, radius, cap, temperature, loss weight,
   component count, or backbone was changed after panel opening.

P0 statuses are:

```text
RGH_P0_REFERENCE_GENERATION_COVERAGE_NO_GO
RGH_P0_FULL_C128_RANK_NO_GO
RGH_P0_CANDIDATE_BINDING_NO_GO
RGH_P0_COORDINATE_CAUSAL_NO_GO
RGH_P0_READY_WAITING_FOR_FROZEN_V124_GX
```

P0 is development evidence only if its identities, supergroups, or queries were
previously opened.  It cannot be promoted to a paper-level claim without a
separately frozen untouched endpoint.

## 14. V124/GX isolation and downstream boundary

This lineage writes only new `rgh_xf` plan/source/program/test/result/registry
paths.  It must not modify, replace, cancel, or consume in-progress V124/GX
files or job outputs, including job `5107346` lineage artifacts.

RGH E0/S0/P0 may proceed without DINO/GX.  A future V stage is blocked until:

- V124 corrected FULL/LOCAL and complete-C128 controls are independently valid;
- the DINO/GX checkpoint, decoder, reducer, null definitions, and three-arm
  semantics are byte-frozen; and
- a new authority binds those final hashes and the RGH selected-superregion
  seals.

The corrected historical fixed-bank P-V2 arms remain immutable matched controls.
They are not overwritten by RGH.

## 15. Stop rules

- E0 failure stops natural S0.
- S0 failure stops P0.
- P0 failure stops RGH before any DINO/GX forward.
- A target naturally absent from C128 is recorded, never inserted.
- Wrong candidates are never universally forced to H0.
- Multiple separated visible components are preserved, never resolved by P.
- No P0 outcome may change the zero visibility boundary, atom axis, merge rule,
  temperature, loss, or controls.
- No runner advances automatically.
- Hash, candidate-axis, fold, identity, supergroup, source, geometry, control,
  checkpoint, or access drift is an engineering abort, not a scientific NO-GO.

## 16. Theoretical support and claim limits

The shared dustbin and reciprocal assignment follow the unmatched-correspondence
principle exemplified by SuperGlue, but this implementation remains its existing
dual-softmax formulation and makes no Sinkhorn claim.

The H0-plus-uniform-atom marginal is a multiple-instance-learning construction:
candidate labels supervise a bag of latent reference-generated explanations
without human masks.  MIL makes latent supervision possible; it does not prove
that the selected component is semantically the complete target.

Endpoint cross-fit reduces direct self-scoring by separating affine fit cells
from held-out score cells.  Spatial correlation prevents treating it as an
independent-sample guarantee.

The dual-end connectivity rule encodes the task prior that one drug-box
explanation should remain coherent in both query and specific reference.  It
reduces disconnected evidence combinations but cannot create missing visual
evidence or guarantee exact identity recognition.

References:

- [Sarlin et al., *SuperGlue: Learning Feature Matching with Graph Neural
  Networks*, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html).
- [Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML
  2018](https://proceedings.mlr.press/v80/ilse18a.html).
- [Chernozhukov et al., *Double/debiased machine learning for treatment and
  structural parameters*, 2018](https://academic.oup.com/ectj/article/21/1/C1/5056401),
  cited only for the general sample-splitting motivation; the RGH checkerboard
  is not claimed to inherit its statistical guarantees.

Passing this contract may establish only that a naturally trained existing CW0
assignment can generate candidate-bound, coordinate-sensitive, frozen connected
proposal components for identities held out from its fold.  Target determination,
retrieval gain, HOLD/SWITCH, ownership, and paper-level generalization require
later independent V, action, and untouched-endpoint contracts.
