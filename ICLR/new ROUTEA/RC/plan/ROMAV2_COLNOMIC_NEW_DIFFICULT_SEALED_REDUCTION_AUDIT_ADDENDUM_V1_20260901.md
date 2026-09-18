# Sealed reduction audit addendum V1

Frozen before target join.

1. The base order is the complete 5,412 corrected exact-label full-rank seal.
   HOLD preserves it. SWITCH removes the selected challenger's exact label
   from its old position, moves it to rank 1, and stably preserves every other
   label. All 31 queries remain in every denominator, including target-absent
   C128 cases and wrong-to-wrong actions.
2. REAL features use `(S00,S10,S01)`. Query-coordinate diagnostic features use
   `(S10,S10,S01)` and reference-coordinate diagnostic features use
   `(S01,S10,S01)`. Thus the destroyed-axis robustness is exactly zero and no
   S00 value re-enters that robustness coordinate. C_BIND uses its independently
   recomputed mismatched-reference `(S00,S10,S01)` tuple.
3. The Q/R paths lack an independently materialized S11 cell. They are
   single-axis coordinate-sensitivity diagnostics, not a strict matched 2x2
   spatial-causal or spatial-ownership proof.
4. Corrected exact-label reduction uses descending score and lower physical row
   for exact ties. The sole duplicate identity is byte-identical rows 714/715.
5. Before reduction, the target join validator must prove ordinal/query-ID
   equality, `corrected_label[target_row] == target_exact_label`, and target
   reference file SHA equality. Target names, stems, paths and ordinals are not
   model or gate features.
