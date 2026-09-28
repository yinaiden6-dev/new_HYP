# Prospective POST prediction and full-C128 property attribution

**Initial publication: programs and protocols are frozen, the task chain is submitted, and final experimental results are still pending. No scientific GO or new accuracy claim is made.** The generated [captured status](EXPERIMENT_STATUS.md) is authoritative for each subsequent snapshot. Completion of a job, engineering review or source export is not itself a positive scientific result.

This increment follows the completed [context/level-gap attribution archive](../m_attribution_depth_20260927/README.md). It adds two bounded tests rather than repeating the earlier whole-path ablations or re-exporting approximately 17,000 historical result files.

The [September 28 submission and partial-results report](../../ICLR/new%20ROUTEA/RC/reports/REPORT_M_PREDICTIVE_FINE_SUBMISSION_ARCHIVE_20260928.md) records the latest execution change and findings. Per the user's updated request, interactive monitoring stops after submission and this archive push. Existing Slurm dependencies and budget-triggered continuation remain active; this publication does not promise an automatic future GitHub refresh.

Completed phase/amplitude accounting distinguishes pathway dependence from finer-property necessity: on the three selected historical rescue cases, NATIVE7 / M_FREE / POST retain 3 / 3 / 2 correct answers under every tested amplitude and global/local phase change. Zeroing the whole P encoding leaves 2 / 0 / 0 correct. MAIN8 has no original rescues, so its rescue-retention denominator is undefined. This is not a full128 causal-share estimate; see the [coverage and decision analysis](../../ICLR/new%20ROUTEA/RC/results/rc_m_fine_c128_attribution_v1/phase_scope_analysis.md).

## The two questions

| Branch | Question and fixed population | What is new; what is reused |
|---|---|---|
| Prospective POST response | Given M, can probe-only predictions forecast the frozen internal model's response at new M values? 120 opened target-present queries, 45 groups, two preselected candidate references per query | Five local probes and four previously uncomputed synthetic endpoints per candidate; original held-fold POST models, cached hidden states, reference tokens and original HR1 content remain fixed |
| Fine-property accounting | How do phase, amplitude and context registration affect candidate competition and actual original corrections on complete natural C128 axes? Eleven opened queries: MAIN8 historical outcome-independent group representatives and three separately reported selected rescue cases | Historical phase/amplitude GPU outputs and original eight-query POST results are reused; missing frozen POST responses are completed for three cases; context registration is replayed on CPU for all 11 × 128 candidate pairs |

Here **fine-property means a more detailed decomposition of the coarse DPT predictor's mechanism**. It does not mean a new experiment on RoMa's fine-matching/refinement stage. Eleven queries with complete C128 are not 128 queries or all H593. MAIN8 and the three outcome-selected cases must never be pooled as an unbiased sample.

The predictive branch tests downstream response **given M**, not how RoMa generates M. The query groups are already opened and the two-candidate selection is inherited from a label-conditioned diagnostic. New synthetic endpoints do not turn this into untouched-query confirmation or target-free retrieval evaluation.

## Frozen design and code

- [Combined plan and task chain](../../ICLR/new%20ROUTEA/RC/plan/RC_M_PREDICTIVE_AND_FINE_C128_CHAIN_V1_20260927.md)
- [Prospective prediction plan](../../ICLR/new%20ROUTEA/RC/plan/RC_POST_M_PREDICTIVE_RESPONSE_V1_20260927.md), [program](../../ICLR/new%20ROUTEA/RC/programs/run_rc_post_m_predictive_response_v1.py), [protocol](../../ICLR/new%20ROUTEA/RC/results/rc_post_m_predictive_response_v1/protocol.json)
- [Full-C128 property program](../../ICLR/new%20ROUTEA/RC/programs/run_rc_m_fine_c128_attribution_v1.py), [protocol](../../ICLR/new%20ROUTEA/RC/results/rc_m_fine_c128_attribution_v1/protocol.json)
- [Orchestrator](../../ICLR/new%20ROUTEA/RC/programs/submit_rc_m_predictive_and_fine_c128_v1.py), [CPU launcher](../../ICLR/new%20ROUTEA/RC/slurm/rc_m_predictive_and_fine_c128_v1.sbatch), [sealed chain protocol](../../ICLR/new%20ROUTEA/RC/results/rc_m_predictive_and_fine_c128_chain_v1/protocol.json)

No encoder, LLM, complete RoMa matcher or GPU forward is required. The context branch executes the frozen **CPU DPT head**, and both branches use frozen POST adapters/projectors and cached content. They do not train replacement models or tune decision thresholds. Detailed tensor intermediates are retained in the workspace, but excluded from Git as specified below.

## Prediction chronology and controls

For each of 240 candidate references, the prediction program observes log-M shifts `−0.02, −0.01, 0, +0.01, +0.02` around original HR1. CONSTANT, shared linear/quadratic, local linear/quadratic and patch-token linear/quadratic MaxSim forecasts are all fixed before evaluating shifts `−0.08, −0.04, +0.04, +0.08`.

All 120 query probes and all predictions must be saved and hashed in a **global prediction seal before any endpoint forward**. Shared coefficients use probe responses only, weighted equally across groups, then queries and their two candidates. Identity roles enter only the later evaluation; ascending candidate positions determine intervention assignment.

Patch forecasts extrapolate already normalized query-token paths, compute reference dot products and take free MaxSim. They are intentionally not normalized again. The baseline top-two gap versus predicted token displacement gives a stability flag for the **forecast**, not a guarantee about the actual nonlinear/BF16 endpoint. Both successful and failed matching-position forecasts are retained.

Reports will include all predeclared predictors and radii, continuous errors, matching changes, shared-slope and constant controls, and group-level exploratory intervals. For action reconstruction, the other 126 candidates' **content scores L** stay at original HR1. All action logits are recomputed with the original frozen INTERNAL3 head; changing RAW's content can therefore affect the other candidates' action logits too. This is prediction agreement, not new retrieval accuracy.

## Full-C128 attribution controls and reuse

The phase branch retains all twelve historical worlds: NATIVE, ZERO, amplitude flattening/permutation, and global/local phase changes in four directions. For MAIN8, the original POST content and logits are reused and checked. The three selected cases complete only their missing frozen POST responses.

The context branch uses thirteen worlds per candidate: native `000`, plus `001`, `110` and `111` for each of four cyclic shifts. Its matched contrast is the four-direction mean of `(000 + 111 − 001 − 110) / 2`. P-fixed A/J shifts, P-only shifts and joint-shift grid controls are also reported. Both branches propagate M and all M-derived terms consistently into the original external and internal models.

Historical GPU phase outputs and new CPU context outputs retain **separate native anchors**. No cross-backend factorial contrast is formed. Original HR1 is a third explicit anchor. Native parity, input invariants, actual LayerNorm hooks, exact pooling and frozen POST patch evidence are retained for checks.

Every completed world gets the full 128-candidate decision, RAW rescue/break sets, retention of each model's original rescues, target-versus-fixed-wrong and strongest-wrong margins. Continuous matched effects are reported separately from integer rescue sets. Zero original-rescue denominators remain undefined. Conditional restoration sets can overlap; they are not additive or unique causal shares.

## Submitted jobs and continuation

| Stage | Recorded job | Dependencies |
|---|---|---|
| Predictive probes | `5168058_[0-7]` | None |
| Global prediction seal | `5168059` | All probes |
| New prediction endpoints | `5168060_[0-7]` | Global seal |
| Predictive join / independent check | `5168061`, `5168062` | Endpoints, then join |
| Historical phase replay/completion | `5168063_[0-10]` | None |
| Phase accounting | `5168064` | Phase replay |
| One-candidate context pilot | `5168065` | None |
| Context computation | `5168066_[0-49%50]` | Pilot |
| Context aggregation | `5168067` | All context shards |
| Context frozen-model propagation | `5168068_[0-10]` | Context aggregation |
| Fine-property final accounting | `5168069` | Context propagation and phase accounting |
| Combined independent closure | `5168070` | Predictive independent check and fine-property accounting |

This is the original submitted chain, **not a live progress table**. Its scientific workers use four CPUs, 32 GB and 10-minute CPU slices, a 430-second worker budget, and same-ID requeue for budget/timeout exits only, at most 32 restarts. Independent failures stop downstream dependencies. The first candidate pilot is reused; completed per-candidate/world outputs are not recomputed.

**Execution-only replacement:** `5168921_[0-3%4]` packages 48 formerly pending context shards into four allocations of 52 CPUs / 128 GB / 10 minutes, with up to 13 unchanged four-thread child workers each. Original running shards `5168066_1` and `5168066_27` continue separately. `5168067` now waits for the packed array and both retained shards. The canceled pending jobs are recorded explicitly, and no running shard was duplicated. See the [packing program](../../ICLR/new%20ROUTEA/RC/programs/run_rc_m_predictive_fine_packed_v1.py), [submission repair](../../ICLR/new%20ROUTEA/RC/programs/submit_rc_m_predictive_fine_packed_v1.py), [launcher](../../ICLR/new%20ROUTEA/RC/slurm/rc_m_predictive_fine_packed_v1.sbatch), and [verified execution record](../../ICLR/new%20ROUTEA/RC/results/rc_m_predictive_and_fine_c128_chain_v1/packed_execution_submission.json). Two packs were offered to both dev and ordinary CPU partitions; the other two remain on ordinary CPU. No periodic monitoring process was created.

Eligible ready jobs may be offered to `dev_cpuonly,cpuonly` by the [scheduling-only refill helper](../../ICLR/new%20ROUTEA/RC/programs/refill_rc_m_predictive_fine_dev_v1.py). It handles Slurm's parenthesized pending reasons, which the original helper did not recognize. It changes only recorded jobs' placement, checks CPU/no-GPU resources, and leaves frozen scientific programs/protocols unchanged. The chain's JSON/JSONL records retain both the initial and corrected refill history.

See [submission](../../ICLR/new%20ROUTEA/RC/results/rc_m_predictive_and_fine_c128_chain_v1/submission.json), [scheduler verification](../../ICLR/new%20ROUTEA/RC/results/rc_m_predictive_and_fine_c128_chain_v1/scheduler_validation.json), [dynamic launcher review](../../ICLR/new%20ROUTEA/RC/results/rc_m_predictive_and_fine_c128_chain_v1/engineering_review/orchestrator_dynamic_validation.json), and [independent predictive review](../../ICLR/new%20ROUTEA/RC/results/rc_post_m_predictive_response_v1/engineering_review/pre_submit_independent_review.json). Those engineering passes do not establish a positive mechanism effect.

## Earlier source coverage and archive boundaries

The [source coverage audit](../../ICLR/new%20ROUTEA/RC/reports/M_ATTRIBUTION_SOURCE_COVERAGE_20260927.json) follows 37 protocol/authority/revision records and recursively identified code dependencies: **341 files, all present byte-for-byte in `d6b11c3`**, including 179 RC files, 156 vendor sources and six explicitly pinned runtime source snapshots. The eleven files absent from the preceding `6daf8d9` were included in `d6b11c3`; no prior attribution source was missing at that audit. This bounded audit excludes the separate QRR branch.

The new exporter traverses exactly three result directories:

```text
ICLR/new ROUTEA/RC/results/rc_post_m_predictive_response_v1/
ICLR/new ROUTEA/RC/results/rc_m_fine_c128_attribution_v1/
ICLR/new ROUTEA/RC/results/rc_m_predictive_and_fine_c128_chain_v1/
```

It retains programs, launcher, plans, protocols, submission and continuation provenance, scalar/per-query/per-candidate JSON or tables, all available reports and independent receipts. Historical code is verified against committed blobs and reused at existing paths. Historical result trees are not rescanned or recopied.

Model weights, `.pt`/`.pth`/`.npz`/other token and feature tensors, confidence/LayerNorm/patch arrays, raw photographs, execution `err/out/log` files, locks and temporary files are excluded. Their paths and sizes appear in the snapshot exclusions. Full numerical replay requires separately restoring those workspace inputs and the recorded environment; this Git archive alone is not a complete tensor/data backup.

From the export checkout:

```bash
# Inventory and verify bounded historical dependencies; copy nothing.
python3 tools/export_m_predictive_fine_c128_20260928.py --preview

# Refresh this increment, then verify exported bytes and captured statuses.
python3 tools/export_m_predictive_fine_c128_20260928.py
```

The exporter never submits experiments, alters frozen sources, stages, commits or pushes Git. Re-run after the final branch receipts pass to add the completed results; update this introduction and interpretation with the signed findings then. The generated `files.json`, `validation.json` and `EXPERIMENT_STATUS.md` distinguish a submitted snapshot from completed numerical acceptance and from an independently verified GitHub push.
