# GX R1P direct REAL-vs-P spatial-pair qualification V1

Adaptive development-only follow-up to the immutable R0 NO-GO and R1Q
optimization insufficiency.  It tests one implementation repair: use the
RCDE comparator in its native pairwise form rather than construct unary
energies against an exact-zero tensor.

For candidate `g`:

```text
S_P(q,g) = PairComparator(REAL(q,g), P_COORD(q,g))
```

The fixed denominator, roots, two directions, caches, candidate binding and
848 `ell/rho/F` parameters are unchanged.  Train four fresh outer folds for
1,024 fixed-LR (`3e-4`) updates, one correct and one wrong episode per update,
using:

```text
L = softplus(-(S_target-S_rival)) + softplus(-S_target)
```

RAW, scale, SWITCH/HOLD, C_BIND and N_REGION_RESAMPLE do not enter this narrow
P-spatial decodability decision.  Only the final update is evaluated.

PASS requires in every fold: final-cycle loss <= 90% of first-cycle loss,
mean target `S_P>0`, and mean `S_target-S_rival>0`; across 16 RAW-wrong
development records at least 11 must have `S_target-S_rival>0`.

PASS is `GX_R1P_DIRECT_SPATIAL_PAIR_DECODABILITY_GO`; failure is
`GX_R1P_DIRECT_SPATIAL_PAIR_INSUFFICIENT`.  Neither is a scientific endpoint
or authorization for RAW deployment/opened/sealed access.
