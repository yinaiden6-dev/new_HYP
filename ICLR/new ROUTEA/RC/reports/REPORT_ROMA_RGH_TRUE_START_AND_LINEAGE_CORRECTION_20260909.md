# RoMa-RGH true start and lineage correction

Date: 2026-09-09  
Status: implementation in progress; no natural result yet

## Corrected lineage

The historical `RGH V7/V8/V9` code is not RoMa-based.  It consumes frozen
128-dimensional ColNomic query/reference tokens and learns the existing
4,357-parameter assignment.  Its natural score path reduces the complete atom
axis and records `selected_superregion_materialized=false`.  It therefore
cannot be used as a scientific result for a selected connected RoMa
hypothesis.

The `21/32` result previously attributed to RGH belongs to the later RC-LTH
P-only legal-family V3.  That branch uses query-first RoMa/ColNomic atoms and a
ten-parameter proposal scorer; it is not the historical RGH mechanism and is
not the new reference-first mechanism below.

The earlier pure-RoMa P plus RAW-ColNomic V M0G experiment is also distinct. It
was conditional on supplied target/rival pairs and used query-first atoms.  Its
NO-GO remains negative evidence and must not be overwritten.

## New mechanism being implemented

The new lineage is `ROMA_RGH_MAXTREE_RAWCOL_V1`:

```text
specific reference
-> RoMa reference-to-query dense field
-> one pure-RoMa atom per reference cell
-> complete threshold-free reference-grid max-tree
-> connected query/reference hypothesis family + exact H0
-> sealed masks only
-> independent RAW-ColNomic unary V
-> full-C128 target-free scores
```

P is reference-first and cannot read ColNomic tokens, retrieval scores, ranks,
winner state or labels.  V cannot read RoMa reliability or proposal scores.
The primary V scope keeps the selected query region but searches the complete
candidate reference; a paired-reference scope remains a diagnostic so the old
coordinate-lock failure cannot be hidden.

The complete frozen design is in
`plan/ROMA_RGH_XF_REFERENCE_FIRST_HYPOTHESIS_V1_20260909.md`.

## Reused assets and exclusions

Reused as read-only engineering sources:

- the validated 64-query original-RAW C128 bridge and RAW ColNomic tokens;
- the frozen RoMa-v2 checkpoint;
- canonical EXIF-aware ColNomic geometry;
- existing H0/connectivity/provenance tests and RAW-ColNomic MaxSim arithmetic.

Not reused as model outputs or scientific evidence:

- historical RGH V7/V8/V9 checkpoints and candidate scores;
- RC-LTH P-only V3 head or its `21/32` result;
- conditional-pair M0G checkpoints;
- any human/SAM mask, target crop or target-informed candidate insertion.

## Current boundary

The existing RAW+RoMa+ColNomic retrieval result remains unchanged and is the
parent comparator, not an input to P.  The new core must first pass synthetic
E0, including a `Q != R` reference-axis test, token-independence of P,
candidate reorder, C_BIND, P_COORD, H0, dual connectivity and unary V
arithmetic.  Only then may the two lowest-ordinal TRAIN queries be run through
a natural full-C128 canary.

No HYP, ownership or retrieval-improvement claim exists at this stage.
