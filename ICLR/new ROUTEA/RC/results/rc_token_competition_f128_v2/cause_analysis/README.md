# F128 cause analysis: code, scores and fixed-weight operator probes

Main interpretation / 主要解释：[完整原因分析](../../../reports/REPORT_TOKEN_COMPETITION_F128_V2_CAUSE_ANALYSIS_20260927.md).

Original result: same-F128 B_CAL103, QR108, ANCHOR108, MULTI106 out of128; grouped fivefold, seed0, natural ColNomic C128, fixed threshold0. The original15-fit scientific validation and promotion NO_GO are unchanged.

This addition explains the observed gains and losses. It does not introduce a newly trained model. In particular, the frozen-MULTI operator probes yielding107/128 are posthoc diagnostics on opened outcomes, not a new registered OOF result or a reason to replace the promotion gate.

## Code and supporting data

| Folder | Script | Evidence |
|---|---|---|
| `cases/` | `build_case_analysis.py` | 384 decisions; target/wrong/base/residual margins; nine rescues and four breaks; two lost MULTI rescues |
| `training/` | `analyze_training.py` | 15 completed fits,247 epoch records, actual CE/selection/gradient checks and held-loss decomposition |
| `mechanism/` | `probe_saved_edges.py` | Same trained MULTI weights with original, RAW-only and common-opponent aggregation; all128 score vectors and complete source provenance |
| `uncertainty/` | `../../../programs/analyze_token128_group_uncertainty_v1.py` | 48 identity groups, rescues/breaks/net effects without treating same-identity images as independent |

The uncertainty script's location is `ICLR/new ROUTEA/RC/programs/analyze_token128_group_uncertainty_v1.py`.

## Reproduce after cloning

All four analysis programs use only the Python standard library. No new RoMa/encoder inference, tensor cache, model weights or GPU are needed for these saved-result analyses. Original model training still needs the separately retained image/model/cache assets described by the frozen protocol.

From the repository root:

```bash
token_result='ICLR/new ROUTEA/RC/results/rc_token_competition_f128_v2'
python3 "$token_result/cause_analysis/cases/build_case_analysis.py" --root "$token_result"
python3 "$token_result/cause_analysis/training/analyze_training.py" --root "$token_result"
python3 "$token_result/cause_analysis/mechanism/probe_saved_edges.py"
python3 'ICLR/new ROUTEA/RC/programs/analyze_token128_group_uncertainty_v1.py' --root "$token_result"
```

These scripts rewrite derived outputs in their own analysis subdirectories. To preserve the checked-in derived files byte-for-byte, reproduce in a separate working copy. Tables and predictions reproduce; provenance paths can legitimately reflect the local clone. The mechanism script uses the included `portable_input_snapshot.json` and reproduces its parsed output exactly; the snapshot is scalar candidate/decision evidence, not a token or embedding cache.

The case and training folders include relocation checks. Source path strings in older protocols remain original provenance, while source-byte checks and the narrow documented training-source mapping identify corresponding files in a relocated clone.

## What is and is not established

- New readers rescue actual deeper candidates, but all main grouped gain intervals include0.
- Selected checkpoints are the recorded minimum validation CE states; there was no epoch0 fallback or demonstrated checkpoint-selection bug.
- Unequal opponent sets followed by unweighted means can distort comparisons. One of the two lost rescues is recovered by a fixed-weight operator intervention; the other remains wrong.
- A nonzero triangle residual measures non-additive pair scores, not automatically a cyclic ranking or a unique cause of error.
- M and the original candidate pool were held fixed. These diagnostics do not establish that M is uniquely causal, that visual information is missing, or that more training/data alone solves all errors.

Original experiment source: `ICLR/new ROUTEA/RC/src/rc_aslo_xf/token_competition_v2.py` plus protocol-bound programs and launcher already archived. Prepared seed-confirmation source is also archived for completeness; that30-fit follow-up was **not submitted**, because the current promotion gate did not pass.
