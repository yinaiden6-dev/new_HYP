# RGH atom-selection historical audit and AS1 decision

Date: 2026-08-31

## Current finding

V9B did not learn a deployable spatial selection.  On the six audited OOF
cases (five ColNomic errors and one V9B break), all 12 target/competitor
candidates had:

- top-atom hard legality false in both directions;
- top-8 legal directional atom count `0/16`;
- query-active count zero;
- target median atom ESS ratio `0.9969` and top mass `0.0246`;
- competitor median atom ESS ratio `0.9956` and top mass `0.0239`.

The colored reference-side cells in the audit are fixed reference macro seeds.
They are not evidence that a corresponding query region exists.  Current V9B
therefore has neither a concentrated soft selection nor a hard materializable
query-reference selected superregion.

Figures:
`reports/figures/rgh_v9b_oof_region_information_audit_v1/`.

## Similar historical mechanisms

| Mechanism | Bank | Selection | H0 | Current evidence | Reuse decision |
|---|---|---|---|---|---|
| P-V2 | fixed CW1 roots and r1-r4 rows | deterministic hard argmax; always selects a legal row | structural only | historical compact/background and candidate-specificity failures | do not reuse selector |
| CMPS | same fixed CW1 row bank | shared 17-parameter root/action head and unique-coverage row score | structural plus semantic H0 | engineering gates exist; formal OOF/reducer is still incomplete | reuse principles only, not checkpoint/result |
| RGH frozen selection | reference macro atoms | keep hard-visible atoms and merge exact dual-end compatible components | exact candidate H0 | synthetic E0 passed; historical natural canary had zero hard-joint-H1 atoms | reuse geometry/merge after natural visibility is repaired |
| RGH V8/V9/V9B | complete reference atom bank | fixed-temperature H0 plus uniform-atom log-mean-exp; entropy only auxiliary | soft H0 marginal | OOF atom weights near-uniform; selected superregion never materialized | close this marginal as selector |
| fixed geometry slots | at most eight fitted hypotheses | deterministic geometry slot construction | canonical illegal-slot H0 | different hypothesis family; prior failures do not test RGH sparse atom selection | no direct reuse |

No existing RC implementation of sparsemax/entmax or an H0-competing sparse
selector over the complete RGH atom bank was found.

## Mathematical failure

The deployed RGH candidate score uses a fixed-temperature atom log-mean-exp at
temperature `0.10`.  Natural matched-null atom contrasts are usually on the
order of `0.01--0.03`, so the atom logit differences are small relative to the
temperature and the marginal is necessarily close to uniform.  V8 target-atom
entropy does not replace that deployed marginal.

The post-V9B hard-eligibility audit further localized the spatial failure over
1,222 atom directions from six target/competitor cases:

- unique endpoints, source/target covariance and affine condition: 1,222/1,222
  pass;
- effective matches: 1,081/1,222 pass;
- reprojection RMSE and maximum error: 0/1,222 pass;
- positive determinant: 665/1,222 pass.

Median RMSE is about `0.166` against the fixed `0.04` gate; median maximum
error is about `0.330` against `0.08`.  This is not a near-threshold failure.
It is consistent with complete-reference verify cells being required even when
the natural query only observes part of the reference object.

## AS1 successor

AS1 is a six-parameter shared selector over the RGH atom bank:

1. use signed two-direction `REAL-P_TRAIN` atom evidence;
2. divide by detached candidate-local RMS to remove the arbitrary evidence
   scale while retaining the matched-null zero point;
3. map five signed/symmetry features through one shared linear head;
4. compete the complete atom axis against a fixed exact-zero H0 slot with
   sparsemax;
5. retain signed candidate evidence and exact-zero inactive atoms;
6. allow only exact-positive-weight, geometry-legal atoms into the existing
   frozen dual-end merge.

AS1 differs from CMPS because it selects RGH reference-generated atoms rather
than CW1 rows, and differs from historical slots because it does not cap or
enumerate a fixed top-K.

## Fail-closed sequence

1. Synthetic AS1 E0 validates sparse/H0/signed gradients.
2. The hard-eligibility audit identifies which geometry condition currently
   removes every natural atom.
3. Natural AS1 feature materialization is forbidden until at least one legal
   atom exists at useful coverage.  Selecting structural-H0 atoms would be a
   false success.
4. After visibility coverage closes, train only the six selector parameters on
   frozen assignment features and evaluate identity-disjoint OOF.
5. Joint 4,357-parameter assignment training is allowed only after the frozen
   feature selector passes retrieval, C_BIND, spatial and visual gates.

No P0/opened/sealed access or full training is authorized by this report.

AS1 E0 V2 subsequently passed all seven engineering gates and independent
validation, including six finite nonzero parameter gradients, exact H0,
sparsity, update, and atom reorder.  Its next state remains
`AS1_WAIT_HARD_GEOMETRY_COVERAGE_REPAIR`; this does not authorize natural AS1
training.
