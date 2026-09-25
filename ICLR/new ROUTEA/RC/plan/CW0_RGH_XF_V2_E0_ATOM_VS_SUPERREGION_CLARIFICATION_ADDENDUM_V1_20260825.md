# CW0-RGH-XF V2 E0 Atom-vs-Superregion Clarification Addendum V1

Date: 2026-08-25  
Status: `FROZEN_INTERPRETATION_CORRECTION`  
Scientific claim: none

## 1. Corrected interpretation

The immutable raw visual canary at
`results/cw0_rgh_xf_v2_raw_visual_canary_v1` renders one projected connected
proposal for each `(candidate, reference atom, direction)`.  Those colored
regions are **reference-generated proposal atoms**, not a deployed merged
superregion.

The class name `RGHReferenceGeneratedSuperregion` and the original E0 prose do
not authorize interpreting an individual atom as the final target or the
complete superregion.  The precise object produced at E0 is:

```text
specific-reference atom
-> soft-assignment affine
-> projected connected query atom
```

## 2. Why E0 did not merge everything

Applying the August-14 pairwise touch/overlap merge rule to every legal raw atom
creates one transitive component per candidate and direction in all four opened
candidate diagnostics.  The observed closures are:

| Query / role | Direction | Legal atoms | Components | Merged Q cells | Merged R cells |
|---|---|---:|---:|---:|---:|
| exec18 target | A_TO_B | 54 | 1 | 141 | 754 |
| exec18 target | B_TO_A | 53 | 1 | 75 | 754 |
| exec18 rival | A_TO_B | 53 | 1 | 140 | 754 |
| exec18 rival | B_TO_A | 50 | 1 | 77 | 753 |
| exec89 target | A_TO_B | 48 | 1 | 187 | 700 |
| exec89 target | B_TO_A | 49 | 1 | 148 | 700 |
| exec89 rival | A_TO_B | 36 | 1 | 213 | 528 |
| exec89 rival | B_TO_A | 35 | 1 | 138 | 528 |

This is a transitive-overlap closure over almost the complete reference, not a
candidate-selected target explanation.  Presenting it as the final superregion
would be a false success.

## 3. Missing step before a deployable superregion exists

A deployable superregion requires a frozen, naturally trained proposal-side
assignment/objective that assigns mass to a sparse compatible atom subset
*before* merging.  Only atoms retained by that target-free frozen mechanism may
enter the exact query-and-reference touch/overlap merge.

The missing edge is therefore:

```text
natural exact-pair plus all-rival proposal training
-> frozen atom weights / H0
-> sparse compatible reference-generated atoms
-> exact connected merge
-> independent V
```

No top-K, mass threshold, slot scan, visual selection, or result-driven pruning
may be introduced to manufacture that subset from the opened canary.

## 4. Required output naming

Future artifacts must distinguish:

- `RGH_PROJECTED_ATOM`: one reference atom projected into query geometry;
- `RGH_ALL_LEGAL_TRANSITIVE_MERGE_DIAGNOSTIC`: the non-deployable union of all
  legal touching atoms;
- `RGH_FROZEN_SELECTED_SUPERREGION`: allowed only after a separately authorized
  natural proposal stage freezes atom selection before role join;
- `RGH_H0`: no selected candidate-specific explanation.

The raw visual canary authorizes only the first object.  An append-only all-legal
merge visualization may be generated to expose the transitive-union failure, but
it cannot authorize P0, V0, retrieval, target determination, or ownership.

