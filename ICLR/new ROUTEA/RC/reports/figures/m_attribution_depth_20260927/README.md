# M attribution depth: three-panel scientific figure

`post_M_level_gap_three_panels.png` is the 300-dpi raster figure; the matching PDF is vector output with embedded fonts. `plotted_values.csv` contains the exact nine points and confidence limits. `plot_m_attribution_depth_v1.py` is the plotting source; `figure_manifest.json` records source and output SHA256 hashes.

The endpoint is the **change in frozen POST target-minus-fixed-wrong content gap**, not retrieval accuracy. Means and exploratory95% group-bootstrap intervals are copied from accepted results. Each panel has its own clearly labelled y-scale. The phase panel has9queries/9groups; the two J/P panels use all120 target-present queries of the existing128panel and45groups. They overlap and are not pooled.

The log-M common-level and relative-gap terms are symmetric finite-change contributions whose sum equals total. The figure is a post-hoc, coordinate-dependent mechanism diagnosis and does not establish unique causal attribution. Full details, log-odds sensitivity, domain limits and negative results are in the companion report `reports/REPORT_POST_M_LEVEL_GAP_INTERPRETATION_20260927.md`.

Rerun with explicit paths after downloading:

```sh
python plot_m_attribution_depth_v1.py --source /path/to/rc_post_m_level_gap_v1 --output /path/to/figure_output
```
