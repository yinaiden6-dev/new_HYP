# Completed POST-LLM experiments and mechanism attribution

2026-09-25. This increment contains the completed internal-M V4 follow-ups, POST-LLM pilot and H593 five-fold experiments, fixed-model attribution, RoMa distribution-path interventions, and the RoMa-free relation-reader pilot. The original source records are preserved without rewriting their numbers, paths, or sealed hashes.

## Start here

- [Full H593 decomposition and interpretation](../../ICLR/new%20ROUTEA/RC/reports/REPORT_POSTLLM_H593_DECOMPOSITION_INTERPRETATION_20260925.md)
- [Unified attribution report](../../ICLR/new%20ROUTEA/RC/reports/REPORT_NEW_HYP_ATTRIBUTION_SYNTHESIS_20260925.md)
- [Full POST-LLM H593 results](../../ICLR/new%20ROUTEA/RC/reports/REPORT_POSTLLM_H593_V1_20260924.md)
- [RoMa-free relation-reader pilot and limitations](../../ICLR/new%20ROUTEA/RC/reports/REPORT_COLNOMIC_ROMA_FREE_RELATION_READER_REVIEW_20260925.md)
- [RoMa matching-distribution intervention report](../../ICLR/new%20ROUTEA/RC/results/rc_roma_position_factor_v1/report.md)
- [Scope, per-file hashes, archive inventory, and exclusions](files.json)
- [Export verification](validation.json)

## Main results in this increment

All rows below use the original H593 grouped OOF folds and ColNomic natural C128. All 593 queries are included; 570 have a target in C128. These are opened development results, not a new untouched external confirmation.

| Frozen endpoint or replay | Correct / 593 | RAW rescues / breaks | Original 54 internal rescues retained |
|---|---:|---:|---:|
| RAW | 426 | 0 / 0 | — |
| Full POST_REAL | 478 | 54 / 2 | 54 |
| Common response across patches | 477 | 53 / 2 | 52 |
| Patch-specific remainder only | 430 | 7 / 3 | 6 |
| Full response, fixed content matches | 469 | 44 / 1 | 44 |
| Common response, fixed content matches | 468 | 43 / 1 | 43 |

The common response remains query-dependent. “Fixed matches” means ColNomic MaxSim indices selected with constant M, not RoMa hard coordinates. POST_REAL retains RoMa and injects M after the LLM; its final head does not directly read M. The experiment does not establish that RoMa can be removed. Reports distinguish this endpoint from the external heads, the earlier small-panel pilots, and failed controls.

## Complete candidate records and restoration

Reports, aggregate results, per-query C128 scores, challenger logits, scalar heads, protocols, code, Slurm launchers, and independent validation records are directly browsable. The 75,904 detailed H593 candidate JSON records are preserved losslessly as `queries/<query_id>/candidates.tar.gz`; smaller pilot candidate directories use the same convention. Each archive contains the original JSON bytes plus `_EXPORT_MEMBERS.json` with SHA-256 and size for every member. This avoids tens of thousands of individual Git entries.

Text files larger than 60 MiB are stored as `.gz`; the inventory maps each archive to its original path. Verify the entire increment from the repository root:

```bash
python tools/verify_restore_mechanism_20260925.py
```

To restore compressed text to the original relative paths, without overwriting conflicting files:

```bash
python tools/verify_restore_mechanism_20260925.py --restore-text
```

Binary adapter/optimizer weights, encoder/token caches, per-patch NPZ intermediates, raw photos, and err/out logs remain in the workspace and are excluded from this Git increment. Their original paths and available SHA seals remain in the result records. The archive supports inspection and score-level analysis; a full model rerun or patch-level audit also needs those separately retained inputs. Absolute paths in frozen scientific records have deliberately not been rewritten.

## Verified payload

The export verifier checked 152,890 original text records, including all 75,904 H593 decomposition candidates. The increment contains 70,720 exported scientific/code files; detailed candidates are grouped into 648 lossless archives. The exported payload is approximately 1.85 GB before Git transport compression. No binary model/token caches or execution logs are included. See `validation.json` for the machine-readable check.
