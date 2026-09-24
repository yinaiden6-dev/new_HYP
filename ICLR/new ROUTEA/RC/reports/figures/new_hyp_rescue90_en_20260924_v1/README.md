# new HYP: 90 successful COST1 corrections — English edition

30 medicine queries, 30 GroZi queries and 30 ISIC queries. Each case has a 3200×1800 PNG, vector SVG and one-page PDF. The package also contains a 90-page PDF, three 30-page PDFs, contact sheets, an offline HTML gallery, CSV/JSON decisions, native FP64 heatmaps and image thumbnails. No numbered fold labels appear in the figures. Both references use enlarged full-column panels, approximately twice the original displayed size, below their corresponding heatmaps.

## Selection and model provenance

- Medicine: the original H593 grouped held-out COST1 predictions, RAW 426→481/593 (58 rescues, 3 breaks). Excluding medicine queries already in the original 18 figures leaves 48 eligible rescues. We include all 12 remaining DIFFICULT cases and one NDV2 case, then 17 OUTCOME cases, prioritizing new identities. The 30 queries cover 24 identities. The exact source fold, head parameters and training exclusion remain in the per-case JSON.
- GroZi: frozen full-H593 COST1, RAW 321→355/480 (34 rescues, no breaks), with no GroZi training or calibration. The 30 selected queries cover 23 product identities. Query images are the dataset product crops used by the original experiment. The natural C128 comes from the original mixed gallery, so an incorrect reference can be a legacy medicine image.
- ISIC: frozen full-H593 COST1, RAW 466→500/537 (34 rescues, no breaks). We exclude the previously shown ISIC-Q-0013 and select 30 queries spanning 26 lesions and 25 patients. This is an opened exploratory same-lesion retrieval panel, not a clinical diagnosis experiment or a new untouched confirmation set. Original ISIC IDs, attribution and existing per-image license metadata are preserved for query, incorrect reference and correct reference.

Success is defined by the sealed retrieval identity labels, not by a new visual judgment. The original 18 figures and experimental files are unchanged. This selected collection does not estimate overall accuracy.

## Reading the figures

The same query is shown alongside its cached query-token visibility grids for the RAW incorrect reference and the correct reference. S is the original weighted content score. M = sqrt(mean(query visibility) × mean(reference visibility)). The COST1 head combines several signals; neither S nor M alone is the final action score. The grids retain the original FP64 values and native shape, use nearest-pixel display and a fixed 0–1 scale, and are not segmentation masks or ownership evidence.

Each case retains all 127 sealed challenger logits plus HOLD=0, the physical candidate axis, head provenance and both query/reference visibility arrays. The frozen external head JSON is in data/heads. Source identifiers and license records are preserved verbatim; presentation labels are in English.

## Reproduction

source/render_rc_cost1_rescue90_english_v1.py redraws the verified data without encoder or RoMa inference, retraining, candidate changes or new experiment results. The English figures use the same arrays, decisions and image thumbnails as the verified source collection. The JSON validation receipts document byte comparisons and independent checks.
