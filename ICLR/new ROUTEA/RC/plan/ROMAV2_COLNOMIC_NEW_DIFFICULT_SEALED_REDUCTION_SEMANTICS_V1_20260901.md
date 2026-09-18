# Sealed reduction semantics V1

This file is frozen before target-join access.

## Ranking

- The base ranking is the target-free, complete 5,412 corrected exact-label
  order sealed in the full-rank prejoin.
- Physical rows are reduced to corrected exact labels by max score; ties use
  the lower physical row.  All 5,413 physical rows remain score inputs.
- HOLD preserves the base ranking byte-for-byte.
- SWITCH uses the single challenger selected by the frozen seven-parameter
  head from natural C128.  Its corrected exact label is removed from its base
  position and moved to rank 1.  Every other exact label retains its relative
  order.  No other score or rank changes.
- If the selected challenger shares the base winner's corrected exact label,
  the action is treated as HOLD for label-level metrics.

Strict R@1 and reciprocal rank are computed from these complete exact-label
orders.  A target absent from natural C128 cannot be selected, but its base and
final reciprocal rank remain measurable from the complete order.

## Actions and controls

- REAL, C_BIND, query-coordinate and reference-coordinate paths all use the
  same frozen weights, bias, zero threshold, one-SWITCH limit and move-to-front
  ranking rule.
- REAL consumes each candidate's sealed `(real, query_control,
  reference_control)` tuple.  The query-coordinate path substitutes
  `(query_control, query_control, reference_control)`, making query robustness
  exactly zero.  The reference-coordinate path substitutes `(reference_control,
  query_control, reference_control)`, making reference robustness exactly zero.
  C_BIND consumes its independently recomputed mismatched-reference
  `(real, query_control, reference_control)` tuple.  Visibility mass comes from
  the corresponding REAL or C_BIND map pair; cyclic coordinate shifts preserve
  that mass.
- Candidate key/raw score remains fixed under C_BIND; only its reference
  payload is deranged by `(destination + 64) mod 128`.
- Query/reference coordinate controls use the already sealed half-axis cyclic
  derangements.  Control failures affect causal claim level, not the validity
  of the REAL retrieval metric.

## Layered result statuses

- Directional retrieval GO requires REAL R@1 above base, REAL MRR not below
  base, rescue greater than break, and at most one break.
- Candidate-binding support additionally requires the REAL increment to exceed
  C_BIND and REAL rescue retention under C_BIND to be below one.
- Spatial support additionally requires REAL to exceed both coordinate-control
  increments with rescue retention below one for each.
- Retrieval GO without causal-control support is reported only at its narrower
  retrieval-only or candidate-bound nonspatial claim level.
