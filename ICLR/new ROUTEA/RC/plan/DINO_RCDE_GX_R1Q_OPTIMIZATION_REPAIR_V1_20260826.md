# DINO-RCDE GX R1Q optimization-repair qualification V1

Date: 2026-08-26

This is an adaptive development qualification after the immutable R0 status
`GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY`.  It is not independent
scientific evidence and cannot overwrite or reinterpret R0.

## Single repaired factor

Keep the R0 post-consensus relational cache, 848 trainable `ell/rho/F` head
parameters, C/P/N branches, exact loss, 0.5/0.5 RAW strata, fold split, seed,
weight decay and gradient clip unchanged.  Change only the optimization
budget/schedule:

- four fresh outer-fold models from the original fold-local INIT checkpoint;
- 1,024 updates;
- repeat the immutable 128-record schedule exactly eight times;
- fixed learning rate `3e-4` from update 1 through 1,024;
- evaluate only after update 1,024;
- no checkpoint selection, threshold scan, scale scan or action tuning.

RAW fusion and SWITCH/HOLD are reported only as diagnostics and do not enter
this qualification decision.

## Frozen development gates

Every fold must satisfy all of:

1. final-cycle mean combined loss is at most 95% of first-cycle mean;
2. final-cycle mean RAW-wrong loss is at most 90% of first-cycle mean;
3. mean target `REAL-C_BIND`, `REAL-P_COORD` and
   `REAL-N_REGION_RESAMPLE` are each positive;
4. mean `Z_target-Z_rival` is positive.

Across the 16 RAW-wrong evaluation records, at least 11 must have positive
`Z_target-Z_rival`.  RAW margin crossing is not a gate at R1Q because RAW and
GX units are not yet calibrated.

PASS is `GX_R1Q_OPTIMIZATION_REPAIR_DECODABILITY_GO`.  Failure is
`GX_R1Q_OPTIMIZATION_REPAIR_INSUFFICIENT`.  Neither authorizes opened/sealed
access, a paper claim, RAW action deployment or automatic stage advance.
