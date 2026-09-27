# H593: all 112 COST1 failures — English gallery

- [All figures and clickable case index](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_h593_failures_en_20260927_v1/README.md)
- [PNG, PDF and SVG figures](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_h593_failures_en_20260927_v1/medicine/)
- [112-page PDF](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_h593_failures_en_20260927_v1/failures112.pdf)
- [CSV](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_h593_failures_en_20260927_v1/data/cases.csv) · [all-case manifest](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_h593_failures_en_20260927_v1/cases_manifest.json)
- [Offline HTML gallery](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_h593_failures_en_20260927_v1/index.html)
- [Independent validation](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_h593_failures_en_20260927_v1/independent_validation.json)

Same medicine-model source as the previous 90 rescue figures: original grouped OOF **NATIVE7-COST1**, H593 and ColNomic natural C128. RAW 426/593, COST1 481/593. This is a complete presentation of its 112 errors, not the post-LLM model's error set.

| Error | Target inside C128 | Target outside C128 | Total |
|---|---:|---:|---:|
| Wrong HOLD | 74 | 18 | 92 |
| Wrong-to-wrong SWITCH | 12 | 5 | 17 |
| RAW-correct break | 3 | 0 | 3 |
| All | 89 | 23 | 112 |

9 DIFFICULT + 1 NDV2 + 102 OUTCOME. Difficult cases appear first. All three breaks are included: OUTCOME-0334, OUTCOME-0337 and NDV2-012-P03.

The images retain the English rescue90 style and enlarged reference panels. Each contrasts the **final incorrect reference** with the correct reference, including complete query photographs. Target-absent cases show the correct reference photograph with a clearly unavailable pair map; no additional target is inserted or scored. Native maps keep a fixed 0–1 color scale.

Per-case data preserves all 127 sealed challenger logits plus HOLD=0, head parameters, candidate identities, source fold and native query/reference maps. Small NPZ files hold visualization arrays and 127×6 decision features only. Original photos are embedded in the presentation panels; standalone source photos, model weights, large caches and execution logs are excluded. The renderer ran no encoder, matcher or training.

[files.json](files.json) records exact copied source bytes and [validation.json](validation.json) records the export. Existing experimental results and previous figures were not modified.
