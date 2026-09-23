# RoMa-v2 + ColNomic difficult90 frozen regression

Date: 2026-09-01

## Result

The frozen current-runtime mechanism was applied without fitting or tuning to
the already-opened difficult validation and test populations: 90 queries from
9 identities.

| path | R@1 | MRR | rescue | break | switch | wrong-to-wrong |
|---|---:|---:|---:|---:|---:|---:|
| ColNomic base | 61/90 (67.78%) | 0.7580 | — | — | — | — |
| REAL frozen gate | 69/90 (76.67%) | 0.8111 | 8 | 0 | 11 | 3 |
| C_BIND | 50/90 (55.56%) | 0.6911 | 0 | 11 | 21 | 10 |
| Query-coordinate diagnostic | 70/90 (77.78%) | 0.8215 | 9 | 0 | 12 | 3 |
| Reference-coordinate diagnostic | 69/90 (76.67%) | 0.8111 | 8 | 0 | 12 | 4 |

Status: `DIFFICULT90_FROZEN_REGRESSION_PASS`.

REAL improves R@1 by 8/90 = 8.89 percentage points and preserves every one of
the 61 base-correct queries.  The paired 8-rescue/0-break query-level exact
McNemar sign probability is 0.0078125; this is descriptive because queries are
clustered within identities and the dataset is opened development data.

## Split consistency

- validation: `21/38 -> 23/38`, 2 rescues, 0 breaks;
- test: `40/52 -> 46/52`, 6 rescues, 0 breaks.

Rescues occur in 5 of 9 identities.  Natural C128 contains the target for
84/90 queries.  Among the 23 base-wrong queries whose target is actionable in
C128, REAL rescues 8 (34.78%).  The remaining six target-absent queries cannot
be rescued by this local route.

## Mechanistic interpretation

The candidate-binding control is decisive:

- all eight REAL rescues disappear under C_BIND;
- C_BIND produces 11 breaks and no rescues;
- REAL rescue retention under C_BIND is 0;
- candidate keys and raw ColNomic scores remain fixed while reference payloads
  alone are shifted by 64 positions.

This supports a candidate-bound exact-reference correction mechanism rather
than a query-only objectness bonus.

It does **not** support a spatial-coordinate claim.  Query-coordinate
destruction retains all REAL rescues and adds one; reference-coordinate
destruction retains all eight.  The available controls are single-axis
diagnostics without an S11 cell.  The justified claim is therefore
candidate-bound, non-spatial retrieval correction with no-regret abstention.

## Integrity and non-degradation audit

- same RoMa checkpoint, ColNomic fingerprint, gallery and identity repair;
- same natural C128, visibility-XF score and half-axis diagnostics;
- same six features, seven parameters, zero threshold and one-SWITCH limit;
- zero training/model updates and zero threshold scans;
- complete 5,412-label MRR with move-to-front action semantics;
- all 90 queries remain in denominators, including six C128 misses;
- REAL and C_BIND target-free prejoins independently validated;
- C_BIND was generated as an exact REAL payload reindex with zero model
  forwards and independently checked field by field;
- reducer and independent validator agree on every action, metric and status.

## Evidence boundary

This is a strong opened regression result, not untouched paper evidence.  The
new_difficult sealed endpoint was ceiling-saturated at 31/31 and therefore
could test preservation but not improvement.  The next paper-level step is a
new untouched external exact-instance population containing natural base-wrong
and low-margin cases.  Neither difficult90 nor new_difficult may now be used to
tune this frozen mechanism.
