# Verified outcome matrices for new HYP

## Figure A: NATIVE7 lineage, opened EVAL32

**Suggested caption.** Query-level correctness and actions on all 32 queries of the
previously opened matched EVAL32, in original execution order. All conditions use
the original RAW C128 and all 127 challengers under zero-threshold HOLD/SWITCH.
Green cells are correct, red cells are wrong, and black dots indicate SWITCH.
Column totals count correct queries. RAW is the original base winner; M3 denotes
RAW_PLUS_M3, L3 denotes RAW_PLUS_L3, and the remaining refitted columns are JOINT4,
PRODUCT5, RESPONSE6 and ORIGINAL7. The final three columns remove the original
S contrast, Q/R-response contrasts, or both from ORIGINAL7 while retaining its
other coefficients; these are fixed-head interventions, not retrained models.
Bold query labels identify the four errors of ORIGINAL7. Equal totals need not
represent equal correctness sets. These panels visualize existing validated
results and do not constitute new evaluation or model adoption.

M3 still has the RAW content prior; L3 still has visibility-conditioned content.
They should not be described as fully isolated quality-only/content-only models.
All columns are REAL; C_BIND outcomes are not shown in this figure.

Refitted sources: jobs 5138847 and 5138852. Fixed-head interventions use the
validated `rc_frozen_group_effect_native64_v1` result. Duplicate ORIGINAL7 and
refitted source actions are cross-checked before combining columns. The old
matched32 DROP_QR bridge is not included.

The full-axis display makes two previously established facts visible without
selecting a favorable subset: M3 and L3 both score 26/32 but differ on
OUTCOME-0220 and OUTCOME-0618; the four ORIGINAL7 errors (DIFFICULT-0050,
OUTCOME-0213, OUTCOME-0373, OUTCOME-0676) remain wrong in all displayed conditions.
These are descriptions of the source outcomes, not newly tested mechanisms.

## Figure B: historical FROZEN_C lineage, opened difficult90

**Suggested caption.** All 90 original difficult90 query ordinals, shown in two
consecutive 45-row panels. The historical FROZEN_C head is distinct from NATIVE7
in Figure A. The five columns are the original RAW winner, original FROZEN_C,
and fixed-head removal of the S contrast, Q/R-response contrasts, or both.
Green/red cells and SWITCH dots have the same meaning as in Figure A. Each
column heading reports the total across all 90 queries, not the individual
45-row panel. Daggers mark the six queries whose target is absent from the
natural C128. The 70/90 DROP_QR entry is an already-opened intervention outcome;
it is not an adopted improved model or independent confirmation.

## Provenance and reproduction

Each figure has SVG/PDF publication exports, a 300 dpi PNG preview, and a CSV with
one row per original query. CSV fields preserve query identifiers/ordinals,
correctness, stored action and final candidate position for every displayed
column; `switch_logit` is the existing stored value in hexadecimal binary64
notation, not a new model output. The RAW column uses recorded base correctness
and has no challenger logit.

`manifest.json` binds all four source results and their independent validations
by SHA256, lists per-column sources and original query axes, and records artifact
hashes and the plotting program. Source validation checks must all be true. The
plotting program creates no model, fits no parameters, and makes no new
prediction or scheduler query.

From workspace root:

```bash
env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 .venv-colpali/bin/python \
  'ICLR/new ROUTEA/RC/programs/plot_new_hyp_verified_outcome_matrices_v1.py'
```

Figure A and B intentionally remain separate. Their counts must not be combined
as repeated measurements of one head or one evaluation population.
