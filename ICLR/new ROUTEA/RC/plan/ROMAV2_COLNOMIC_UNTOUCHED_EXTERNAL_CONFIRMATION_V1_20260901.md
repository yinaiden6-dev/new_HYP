# RoMaV2-ColNomic untouched external confirmation V1

## Claim

Confirm the frozen candidate-bound, target-free, no-regret exact-instance
retrieval correction mechanism against frozen ColNomic. Spatial ownership is a
separate optional claim and is not required for retrieval GO.

## Frozen mechanism

Consume only the immutable model bundle
`isolated/romav2_colnomic_candidate_bound_no_regret_v1_20260901`.
No retraining, calibration, checkpoint selection, threshold change, C128
change, pooling change or preprocessing change is permitted.

## Minimum paper population

- 240 natural queries;
- 48 exact identities;
- at least 24 independent supergroups;
- exactly 5 primary queries per identity;
- at least 2 collection locations and 3 device classes;
- query/reference/identity/supergroup SHA-disjoint from all historical Route A
  populations;
- no synthetic augmentation as an independent primary query.

Each identity must be acquired under a result-blind fixed schedule: frontal or
oblique view, partial occlusion, clutter/multiple objects, distance/blur/crop,
and one unconstrained blind-style capture. Identities and images cannot be
selected by ColNomic/RoMa outcomes.

## Information sufficiency

After the one allowed scoring attempt, report but do not tune on:

- at least 48 base-wrong queries;
- at least 36 base-wrong queries whose target naturally enters C128.

If either is absent, output `EXTERNAL_CONFIRMATION_UNINFORMATIVE_CEILING`.
This is not model failure and the endpoint cannot be extended after inspection
while retaining untouched status.

## Gallery and inference

Freeze the complete 5,413 legacy physical rows plus all external references
before query scoring. Preserve every competing reference. Save complete
corrected exact-label rankings; local inference sees only natural C128 and may
not insert the target. All queries remain in R@1/MRR denominators.

Required paths are BASE, REAL and fixed-point-free exact-label-disjoint C_BIND.
Q/R/S11 controls are additionally required only for a strict spatial claim.
Candidate reorder is an engineering equivariance test, not C_BIND.

## Isolation

Materialize all target-free scores, features, logits, actions, controls and
complete rankings before a separate target join. Model-visible target, label,
stem and identity reads must be zero. An independent implementation recomputes
all final actions and metrics.

## Statistics

Primary endpoint: `R@1_REAL - R@1_BASE`.

- 100,000 supergroup-cluster bootstrap replicates;
- stratify by collection location;
- two-sided 95% CI;
- cluster-bootstrap MRR;
- query-level exact McNemar is descriptive only;
- hierarchical testing: REAL vs BASE first, then REAL vs C_BIND.

## Paper GO

`EXTERNAL_CANDIDATE_BOUND_NONSPATIAL_GO` requires:

1. R@1 increment at least +0.04;
2. supergroup-cluster 95% CI lower bound above zero;
3. REAL MRR point estimate above BASE and CI lower bound no worse than -0.005;
4. rescues greater than breaks;
5. breaks/base-correct no more than 1%;
6. neither collection location has negative net gain;
7. rescues span at least 6 identities and 4 supergroups;
8. REAL significantly exceeds C_BIND;
9. C_BIND retains fewer than 50% of REAL rescues.

Only if REAL also beats Q, R and S11 with rescue retention below 0.50 may the
status advance to `EXTERNAL_CANDIDATE_BOUND_SPATIAL_GO`. Otherwise the claim
remains candidate-bound and non-spatial.

## Failure taxonomy

- invalid data/manifest: `EXTERNAL_INPUT_ABORT`;
- insufficient base errors: `EXTERNAL_CONFIRMATION_UNINFORMATIVE_CEILING`;
- primary effect/CI failure: `EXTERNAL_RETRIEVAL_NO_GO`;
- excess breaks: `EXTERNAL_SAFETY_NO_GO`;
- retrieval GO without binding: `EXTERNAL_RETRIEVAL_GO_BINDING_UNPROVEN`;
- retrieval plus binding, spatial controls fail:
  `EXTERNAL_CANDIDATE_BOUND_NONSPATIAL_GO`;
- all controls pass: `EXTERNAL_CANDIDATE_BOUND_SPATIAL_GO`.

## Prohibitions

No external fitting, result-based image selection, target insertion, deletion
of misses, filename/stem inference, post-result query addition, endpoint reuse
for repair, or promotion from retrieval GO to spatial ownership is allowed.
