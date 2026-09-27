# Deeper M attribution: context registration upstream, level versus gap downstream

**Status: both attribution branches, frozen-model propagation, independent arithmetic reviews and combined closure are complete.** The existing coarse-path correction recount is also independently complete. This is a separate increment after the completed [128-image attribution archive](../m_causal128_attribution_20260927/README.md), published in commit `6daf8d9`. It does not rerun or duplicate that archive's approximately 17,000 result files.

The two new branches address different remaining explanations:

1. **Upstream:** does P help because of its internal field structure, or because it is registered with the appearance and joint-context inputs that the predictor reads?
2. **Downstream:** when an intervention changes M, how much of the frozen model's response follows the candidates' common M level, their relative gap, or interaction between those two coordinates?

The files contain frozen designs, completed results, numerical checks, independent scientific readings and scheduling provenance. Upstream acceptance is `CONTEXT_BINDING_COMPLETE_INDEPENDENT_PASS`; combined acceptance is `M_ATTRIBUTION_DEPTH_BOTH_BRANCHES_PASS`. After a snapshot is generated, `EXPERIMENT_STATUS.md` records the captured receipts. Completion concerns these finite computational interventions; it does not imply unique causality, independent external confirmation or improved full-H593 accuracy.

Start with the [final synthesis](../../ICLR/new%20ROUTEA/RC/reports/REPORT_M_ATTRIBUTION_DEPTH_FINAL_20260927.md), its [source-bound validation](../../ICLR/new%20ROUTEA/RC/results/rc_m_attribution_depth_v1/final_analysis_v1/validation.json), and the [independent final numerical and scientific review](../../ICLR/new%20ROUTEA/RC/results/rc_m_attribution_depth_v1/final_analysis_v1/second_review.json). It connects the upstream scale-dependent support effects, downstream level/gap decomposition and independently recounted original corrections.

The [predictive-theory review](../../ICLR/new%20ROUTEA/RC/plan/RC_M_ATTRIBUTION_DEPTH_PREDICTIVE_THEORY_REVIEW_20260927.md) records a possible next research framework, **not another submitted or completed experiment**. It separates predicting downstream responses when intervention M is already supplied from predicting the entire chain before the intervention. Prediction and causal identification are different claims; the already opened nine-group and 128-image panels remain development evidence.

## Effective task chain after CPU scheduling revision

| Stage | Job | Input scope | Required output |
|---|---|---|---|
| Context-registration interventions | `5167771_[0-17]` | Nine opened mechanism groups; nine target-versus-wrong comparisons = 18 query/reference candidate pairs; 29 input configurations | All fixed DPT outputs and input invariants |
| Level/gap interventions | `5167800_[0-7]` | 120 target-present queries from the 128-image panel; 45 components; phase9 also reported separately | Frozen POST response at all valid crossed M inputs |
| Upstream join | `5167808` | After `5167771`; all 18 candidate-pair shards | Independent upstream acceptance |
| Upstream frozen propagation | `5167809_[0-8]` | After `5167808`; same nine groups, 29 configurations, original external/internal models | Original models' responses to each changed M |
| Upstream propagation join | `5167810` | After `5167809`; all nine propagation shards | `CONTEXT_BINDING_COMPLETE_INDEPENDENT_PASS` |
| Downstream join and second arithmetic review | `5167811`, `5167812` | After `5167800`, then after `5167811`; all 120 queries and both coordinate systems | Primary and independent arithmetic receipts |
| Combined closure | `5167813` | After `5167810` and `5167812`; both branches and independent review | `M_ATTRIBUTION_DEPTH_BOTH_BRANCHES_PASS` |

The original downstream array `5167772` was replaced by `5167800`; the original pending postprocessing jobs `5167773`–`5167778` were canceled and replaced by `5167808`–`5167813`. The table above is the effective recorded chain, not the superseded initial submission.

The upstream worker array retains its original `accelerated` scheduling request with one GPU for placement. The replacement downstream and postprocessing jobs use `cpuonly`, with eligible downstream shards also offered to `dev_cpuonly` as recorded in the refill files. All computation remains on CPU; requests retain four CPUs, 32 GB and 10-minute resumable slices. The CPU launcher changes only resource directives, and its non-`#SBATCH` body is byte-identical to the original launcher. Frozen numerical programs and protocols are unchanged. This is a scheduling record, not a live display or completion assertion.

Revision provenance is retained in [CPU launcher validation](../../ICLR/new%20ROUTEA/RC/results/rc_m_attribution_depth_v1/cpu_launcher_validation.json), [downstream replacement](../../ICLR/new%20ROUTEA/RC/results/rc_m_attribution_depth_v1/downstream_cpu_replacement.json), and [effective postprocessing chain](../../ICLR/new%20ROUTEA/RC/results/rc_m_attribution_depth_v1/cpu_postprocess_chain.json). The orchestration result directory also retains the dev-partition refill and superseded scheduler records, so the reason for each replacement remains inspectable.

## Upstream: controlled registration of A, J and P

The frozen predictor reads two appearance layers, with its later input equal to `A1 + J + P`. It then applies per-patch LayerNorm and spatial decoding. The experiment applies the same cyclic grid shift independently to A, J and P in a complete binary factorial design. A includes **both** appearance layers. Four shifts are fixed in advance: ±4 coarse-grid positions along X and Y. The unchanged configuration is shared, producing `1 + 4 × 7 = 29` distinct configurations.

The primary contrast averages `[g000 + g111 − g001 − g110] / 2` across directions, where g is the target-minus-fixed-wrong logM gap and bits indicate A/J/P shifts. It compares aligned and misaligned inputs at matched absolute placements. `110 − 000` keeps the P tensor exactly unchanged; conditional J×P and three-way interactions are also retained.

Exact inverse-roll recovery, P Fourier-magnitude preservation and a fixed small-channel Gram check verify input properties. Per-patch variances, cross-covariances, actual summed-input LayerNorm output, confidence maps and pooled weights are retained in the workspace. These diagnostics track how field registration changes the actual predictor input rather than assigning meaning from the heatmaps alone.

**Common shifting of all three fields is not assumed to leave M unchanged.** Convolution, padding, multiscale resizing and fixed token pooling can respond to grid position. The `111 − 000` control, dense-confidence means, exact-pooling means and output-shift discrepancy quantify these effects. They do not make the experiment a unique test of physical geometry or semantic correspondence. Equal `A0` and equal `A1 + J + P` are structurally indistinguishable at this decoder.

The same changed M values then enter the original fold's frozen external NATIVE7/M_FREE and internal POST_REAL models. External local-content weighting shape remains fixed, so this is propagation through M, not the total effect of altering every spatial-weight path. Only target and the fixed wrong candidate are changed; any third RAW winner retains its original HR1 anchor. No new head is trained.

## Completed upstream finding: registration changes support and its use

All **18 query/reference candidate pairs × 29 configurations = 522 candidate outputs** and propagation for all **nine target-versus-wrong comparisons** are complete. Four directions are averaged within each of the nine opened identity groups. The [scientific reading](../../ICLR/new%20ROUTEA/RC/reports/REPORT_M_CONTEXT_BINDING_SCIENTIFIC_READING_20260927.md), [independent seven-contrast review](../../ICLR/new%20ROUTEA/RC/reports/REPORT_M_CONTEXT_BINDING_INDEPENDENT_INTERPRETATION_20260927.md), and [all independently recomputed values](../../ICLR/new%20ROUTEA/RC/results/rc_m_context_binding_v1/independent_interpretation/result.json) retain every specified direction and comparison.

For the primary matched registration contrast `[000 + 111 − 001 − 110] / 2`:

| Endpoint | Mean change | Exploratory 95% group interval |
|---|---:|---:|
| Target logM | +0.056083 | [+0.039478, +0.073280] |
| Fixed-wrong logM | +0.057445 | [+0.038435, +0.078460] |
| Target−wrong logM gap | −0.001363 | [−0.020763, +0.017139] |
| Target−wrong raw M gap | +0.002857 | [+0.000871, +0.005062] |
| NATIVE7 action gap | −0.002825 | [−0.023383, +0.015466] |
| M_FREE action gap | −0.003845 | [−0.027895, +0.017706] |
| POST content gap | +0.002533 | [+0.000399, +0.005406] |
| POST action gap | +0.131071 | [−0.001990, +0.323616] |

Registration raises target and wrong support levels in all nine groups. **The raw-M gap and POST content gap show positive signals; the primary log-ratio gap and primary action gap do not have intervals excluding zero.** Thus this is neither “no identity-related tendency” nor evidence of stable improved final correction. An increase in absolute support, an increase in relative ratio, a larger content gap and a changed decision are different endpoints.

With the entire P tensor held fixed, shifting A/J reduces target and wrong logM by approximately 0.048596 and 0.048348, respectively. This rules out P alone determining the tested output. The matched contrast rejects a purely additive placement explanation for absolute support, but it does not isolate semantic registration from every decoder/grid interaction. Raw M and logM are also different scales: a common multiplicative change could enlarge an existing raw difference without changing the ratio. The present results do not identify an exact common multiplier.

The predefined secondary J×P interactions also remain visible: POST content/action effects are +0.005080/+0.264525 with A fixed, and +0.004990/+0.260904 with A shifted; all four exploratory intervals are positive. These do not replace the primary contrast or constitute an independent confirmation. The three-way interaction intervals include zero.

All 576 inverse-shift checks pass; the maximum checked Gram relative error is 6.05e-16, and the full-P Fourier-amplitude relative error is 1.37e-16. Independent contrast and frozen-action recomputation agrees within 3.55e-15. Actual SUM and LayerNorm commute with common shifting, while the complete DPT need not; the common-grid control remains necessary. These are latent-input interventions on an opened small panel, not photographs transformed and re-encoded, or a full-C128 ranking test. Fine-grained covariance and normalization diagnostics locate interface changes but do not uniquely assign their mediation of M.

## Downstream: common level and relative candidate gap

For transformed candidate values `z_target` and `z_wrong`, define:

`mu = (z_target + z_wrong) / 2`, `delta = z_target − z_wrong`.

The four cells LL/LG/GL/GG combine the low/high endpoint's mu and delta. LL and GG reproduce the old candidate M values exactly; original patch MaxSim and argmax must replay. Actions are recomputed under the new fixed-third-RAW-anchor protocol, so their endpoint values are not silently equated to old full-world actions.

Two coordinate systems are reported in full:

- `z = log(M)`, the primary decomposition. On the J-with-P edge, 21 of 120 queries have an invalid cross-cell M above 1. Those cells are recorded, never clipped or forwarded. Their original endpoints remain available; complete decomposition of that edge is a clearly labelled 99-query domain-conditional appendix. The other three log-M edges retain all 120 queries.
- `z = log(M / (1 − M))`, a separately frozen sensitivity analysis for **all** four J/P edges and the phase9 contrasts. It is not applied only to the 21 invalid cases, and results are not selected between coordinates.

The program retains both orders of conditional effects, their interaction, the total GG−LL effect, and symmetric level/gap allocations. Symmetric allocations add to the total but are coordinate-dependent accounting conventions, **not a unique causal decomposition**. M is not a calibrated identity probability. A valid bounded M also need not be inside the original training support; the descriptive `condition_range_audit.md` records the model's actual transformed input ranges without filtering observations or fitting thresholds.

Four phase-direction repeats are averaged within groups; they are not independent images. The nine phase groups overlap the 128-image panel and are never pooled as independent evidence. All these panels have already been opened during mechanism research.

## Interpretation rules

The [recorded interpretation rules](../../ICLR/new%20ROUTEA/RC/plan/RC_M_ATTRIBUTION_DEPTH_INTERPRETATION_RULES_20260927.md) define the finite explanations these experiments can challenge. They are part of the required archive, alongside the experimental protocols.

Upstream, a response while P is exactly fixed can reject an explanation using only P; the matched four-cell contrast can reject additive separability of context placement and P placement at the measured endpoint. Decoder/grid interactions remain possible, and the sum interface does not identify a unique decomposition of J and P.

Downstream, nonzero common-level effects can reject a gap-only response in the specified coordinate system. The hypothesis `L_g = a * log(M_g) + b_g` with a shared slope predicts zero common-level effect and zero interaction **in log-M coordinates**; logit contrasts are coordinate sensitivity and do not, by themselves, test that same affine hypothesis. Candidate-specific but unequal affine slopes can already produce a common-level effect. Observing such an effect alone does not establish nonlinear processing.

These comparisons can identify inadequate finite explanations for this frozen system. They do not establish unique causality, universal necessity, or a new accuracy result. Acceptance receipts verify execution and arithmetic; the final scientific interpretation still depends on the signed results and controls.

## Completed findings and remaining interpretation limits

| Question | Current conclusion |
|---|---|
| Does registration matter with P's internal field fixed? | Yes for the tested support values: fixing P exactly and moving A/J changes M; this does not uniquely identify semantic correspondence |
| Can common grid/decoder effects account for the contrast? | The matched contrast excludes purely additive placement effects; decoder interactions remain possible. Dense-mean and exact-pooling log-gap results both have intervals including zero |
| Does the same upstream effect reach both frozen downstream uses? | Propagation is complete: primary POST content gap is positive, whereas external action gaps and primary POST action gap retain intervals including zero |
| Does common M level explain response reversals, or is interaction required? | Downstream complete: negative common-level contribution outweighs positive mean gap contribution in native-amplitude phase9; interaction is not established as its main explanation |
| Are any conclusions robust across coordinates and allowed domains? | Downstream complete: several directions agree across coordinates, but component allocations, interactions and some intervals differ; report both and the common legal subset |

Negative effects, endpoint disagreements, invalid-domain counts and coordinate sensitivity must remain alongside any positive result. These are attribution experiments, not accuracy optimization, QRR training, full-gallery evaluation, or untouched external confirmation.

## Completed downstream interpretation and scientific figures

The [downstream interpretation report](../../ICLR/new%20ROUTEA/RC/reports/REPORT_POST_M_LEVEL_GAP_INTERPRETATION_20260927.md) explains how the same M input is used differently by the external and internal models. The external normalized comparison is effectively invariant to common positive rescaling of the two candidate M values in this panel, where every fixed pair contains RAW. Frozen POST reads absolute logM as well as its candidate difference; shared-level changes can reinforce or offset the relative-gap path. This is a difference in how one interface is used, not equivalence of the two functions.

For POST content target-minus-wrong changes in the primary log-M decomposition:

| Existing panel / intervention | Symmetric common-level contribution | Symmetric relative-gap contribution | Total |
|---|---:|---:|---:|
| Phase9, native amplitude, LOCAL→GLOBAL | −0.000667263 | +0.000237960 | −0.000429303 |
| Phase9, permuted amplitude, LOCAL→GLOBAL | +0.001486313 | +0.000692945 | +0.002179257 |
| All 120 target-present queries, P restored with J | +0.03541213 | +0.01315276 | +0.04856489 |
| All 120 target-present queries, P restored without direct J | −0.00118992 | −0.00627983 | −0.00746975 |
| All 120 target-present queries, J restored without P | −0.02477472 | +0.02645358 | +0.00167886 |

These are content-gap changes, **not accuracy gains**. Phase9 overlaps the 128-image panel; repeats are averaged within groups. The original-amplitude phase9 gap contribution has an interval containing zero despite its positive mean. The J-with-P edge requires the separately reported common legal 99-query comparison because 21 primary-coordinate cross cells exceed M=1. Log-odds results and numerical-domain versus TRAIN-support limits remain in the full report. The displayed rounded components may differ in their last digit from the rounded total; unrounded values close within the independently validated arithmetic tolerance.

The three-panel scientific figure is available as [PNG](../../ICLR/new%20ROUTEA/RC/reports/figures/m_attribution_depth_20260927/post_M_level_gap_three_panels.png) and [vector PDF](../../ICLR/new%20ROUTEA/RC/reports/figures/m_attribution_depth_20260927/post_M_level_gap_three_panels.pdf). It is accompanied by [exact plotted values and intervals](../../ICLR/new%20ROUTEA/RC/reports/figures/m_attribution_depth_20260927/plotted_values.csv), [plotting source](../../ICLR/new%20ROUTEA/RC/reports/figures/m_attribution_depth_20260927/plot_m_attribution_depth_v1.py), and the [source/output SHA256 manifest](../../ICLR/new%20ROUTEA/RC/reports/figures/m_attribution_depth_20260927/figure_manifest.json). The [figure README](../../ICLR/new%20ROUTEA/RC/reports/figures/m_attribution_depth_20260927/README.md) records panel definitions and regeneration commands. These are generated scientific plots; no source photographs are included.

## Completed existing-cache audit: coarse pathways and actual corrections

The [independent coarse-path recount](../../ICLR/new%20ROUTEA/RC/reports/REPORT_M_COARSE_PATH_CORRECTION_ACCOUNTING_20260927.md) covers the **opened first 128 H593 queries, 46 identity groups, original held-fold frozen models and natural ColNomic C128**. This is not the historical EVAL128 panel. RAW is **93/128**; targets enter C128 for 120 queries across 45 groups, with 27 potentially recoverable RAW errors. The remaining eight queries cannot be corrected inside their fixed C128.

The audit independently recomputes all **1,920 full-C128 decision vectors**: three models × five cached worlds × 128 queries, including the original zero-threshold HOLD/SWITCH rule. Feature error is zero; maximum logit discrepancy is **7.11e-15**. Decisions, switches and correctness agree with the sealed results. This is an accounting check of existing cached experiments, with no new model inference or training.

In the table, each entry is **correct / 128 (rescued RAW errors, broken RAW-correct queries)**. The original HR1 anchor is shown separately from the complete coarse-stage replacement.

| Cached M world | Frozen NATIVE7 | Frozen M_FREE | Frozen POST |
|---|---:|---:|---:|
| Original HR1 | 102 (10, 1) | 102 (10, 1) | 101 (8, 0) |
| Complete coarse A + J + P | 102 (10, 1) | 102 (10, 1) | 100 (8, 1) |
| Coarse A + J, P removed | 100 (7, 0) | 100 (7, 0) | 94 (2, 1) |
| Coarse A + P, J removed | 93 (0, 0) | 93 (0, 0) | 93 (1, 1) |
| Coarse A alone | 93 (0, 0) | 93 (0, 0) | 93 (1, 1) |

Complete coarse M preserves all original rescues for the three models, but POST also introduces a break at H593 execution index 84: **101→100 is real replacement drift**, not a consequence of subsequently removing J or P. Removing P preserves 7/10 original NATIVE7 rescues, 6/10 original M_FREE rescues and 2/8 original POST rescues. M_FREE's seventh rescue under P removal is a different query, so equal rescue totals do not mean equal corrected sets. Removing J loses all original rescues, although POST retains one different rescue and one different break.

Conditional restoration of P when J is present recovers **3 / 4 / 6** original rescues for NATIVE7 / M_FREE / POST; restoration of P without J recovers **0 / 0 / 0**. These overlapping restoration sets show conditional pathway participation. They are not additive or unique attribution shares.

The intervention propagates changes in candidate M and its consistent descendants while retaining the original local weighting shapes and content inputs; POST uses the changed M through its frozen internal adapter. Thus this is **M-only propagation of coarse-path interventions**, not removal of every downstream J/P-dependent path. It does not assign the full-C128 corrections uniquely to phase, amplitude or context registration, prove J/P irreplaceable, or extrapolate to the full H593 totals 481/478. Full query identities, correction/break sets and source hashes are retained in [result.json](../../ICLR/new%20ROUTEA/RC/results/rc_m_coarse_path_correction_audit_v1/result.json), with [independent validation](../../ICLR/new%20ROUTEA/RC/results/rc_m_coarse_path_correction_audit_v1/validation.json).

## Files, replay and export scope

The source-relative layout remains `ICLR/new ROUTEA/RC/`:

- `plan/RC_M_CONTEXT_BINDING_V1_20260927.md`
- `plan/RC_M_ATTRIBUTION_DEPTH_INTERPRETATION_RULES_20260927.md`
- `plan/RC_M_ATTRIBUTION_DEPTH_PREDICTIVE_THEORY_REVIEW_20260927.md`
- `reports/REPORT_POST_M_LEVEL_GAP_DESIGN_20260927.md`
- `reports/REPORT_POST_M_LEVEL_GAP_INTERPRETATION_20260927.md`
- `reports/REPORT_M_COARSE_PATH_CORRECTION_ACCOUNTING_20260927.md`
- `reports/REPORT_M_CONTEXT_BINDING_INDEPENDENT_INTERPRETATION_20260927.md`
- `reports/REPORT_M_CONTEXT_BINDING_SCIENTIFIC_READING_20260927.md`
- `reports/REPORT_M_ATTRIBUTION_DEPTH_FINAL_20260927.md`
- `reports/figures/m_attribution_depth_20260927/` (only this figure directory admits PNG/PDF)
- `programs/rc_m_context_binding_v1.py`
- `programs/run_rc_post_m_level_gap_v1.py`
- `programs/check_rc_post_m_level_gap_v1.py`
- `programs/submit_rc_m_attribution_depth_v1.py`
- `programs/audit_rc_m_coarse_path_corrections_v1.py`
- `slurm/rc_m_attribution_depth_v1.sbatch`
- `slurm/rc_m_attribution_depth_cpu_v1.sbatch`
- `results/rc_m_context_binding_v1/`
- `results/rc_post_m_level_gap_v1/`
- `results/rc_m_attribution_depth_v1/`
- `results/rc_m_coarse_path_correction_audit_v1/`

The four explicitly named result roots retain protocols, source hashes, scalar outputs, all available per-case JSON/table records, validation receipts and orchestration metadata. The fourth root is limited to the coarse-path correction audit; it does not open up the old causal128 result tree. Source dependencies already identical to committed files are referenced by commit/blob/hash rather than copied as another historical increment. Frozen absolute paths and input hashes are retained; running from a fresh clone requires restoring the separately stored data and execution environment.

The temporary `independent_interpretation/before_scale_explanation_refresh/` directory is excluded explicitly: it contains superseded derived drafts, not additional current evidence. The accepted interpretation, scientific reading and their current hash-bound receipts are retained.

No `.pt`/`.npz`/other feature or patch tensors, model weights, raw photos, optimizer states, or `err/out/log` files are uploaded. Their paths and sizes appear in the exclusions inventory. PNG/PDF are allowed only in `reports/figures/m_attribution_depth_20260927/`, together with the generating code, source tables and SHA manifest. In particular, the workspace retains detailed LayerNorm, confidence and MaxSim/argmax tensors, but the Git snapshot alone is not a full tensor replay package.

From the export checkout:

```bash
# Inspect the exact scope without copying.
python3 tools/export_m_attribution_depth_20260927.py --preview

# Refresh only this new increment, including byte verification and captured status.
python3 tools/export_m_attribution_depth_20260927.py
```

The exporter performs no experiments, modifies no frozen source/protocol, and never stages, commits or pushes Git. It traverses only the four named result roots; the old causal128 result tree remains excluded by an explicit scope assertion. `files.json`, `validation.json` and `EXPERIMENT_STATUS.md` are generated by the first export and refreshed on subsequent snapshots. Both new chains have passed acceptance; the export independently checks the receipts captured in its own snapshot. A local snapshot and passing scientific receipts are not themselves proof of a successful GitHub push.
