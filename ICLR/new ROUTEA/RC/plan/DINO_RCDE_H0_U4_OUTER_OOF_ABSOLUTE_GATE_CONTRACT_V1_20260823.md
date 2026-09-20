# H0 U4 outer-OOF absolute-verification gate contract V1

U4 evaluates only whether Track-H learned an absolute target-versus-Q0-donor
verification zero point on identity/supergroup-disjoint outer OOF data.

Population:

- four outer-refit scopes, 594 queries;
- 567 Q0-matched target/donor pairs: folds 145/140/136/146;
- 27 unmatched rows are retained as unscored, with no fallback;
- exact target and labels enter only offline evaluation after target-free P-lock
  seal; no D1 score/rank/winner/gap is read.

For each matched query, the exact same sealed target P-lock query union is used
for target full reference and donor full reference. Each of INIT, Track-R and
Track-H emits two directional target logits and two directional donor logits.

The registered per-query unary log-loss is aligned with training:

```text
Lq = 0.5 * mean_d [softplus(-target_d) + softplus(donor_d)]
margin_q = mean_d(target_d - donor_d)
```

The factor `0.5` makes `Lq` the mean over the two roles (target/donor) and the
two directions.  Therefore an all-zero, no-information decoder has exactly
`Lq = log(2)`.  Omitting this factor while retaining a `log(2)` gate would
silently make the null baseline `2*log(2)` and create a spurious NO-GO.

If a sealed target P-lock direction is structural H0 (empty query union), no
reference forward is permitted.  INIT, Track-R and Track-H must all emit the
same exact-zero target and donor logits for that direction, record zero model
forwards, and retain the row in the fixed population.  No fallback mask or
alternate proposal may be substituted.

Primary aggregation is recipient-supergroup -> query balanced. Raw query and
track summaries are reported but are not the primary gate.

U4 GO requires all:

- Track-H group-balanced log-loss < log(2);
- Track-H log-loss at least 0.01 below Track-R;
- Track-H log-loss at least 0.01 below INIT;
- Track-H group-balanced target>donor win rate >= 0.625;
- Track-H group-balanced margin > 0 overall;
- Track-H fold margin > 0 in at least 3/4 folds.

The Q0 donor substitution is the registered C_BIND/null control and is counted
once, not presented as an independent fifth causal test.

GO means only: conditional on target in C128 and the correct candidate's sealed
P-lock being given, Track-H learned unseen-identity outer-OOF absolute
verification evidence superior to same-capacity relative-only training and
initialization. It does not prove target-free target finding, retrieval gain,
ownership, spatial causality, physical absence or C128-miss recovery.
