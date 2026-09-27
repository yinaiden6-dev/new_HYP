# Training and selection cause analysis

Scope: TOKEN_QR / TOKEN_QRR_ANCHOR / TOKEN_QRR_MULTI, native-token V2; opened F128 development panel, original identity-grouped five folds, seed0; unchanged natural ColNomic C128; fixed0 action; matched phase-specific frozen same-F128 B_CAL. These are development observations, not external confirmation.

## Supported explanation

The residual learns a useful average correction, but the objective and selection rule do not protect each previously correct query. The supervised loss is identity-set full-C128 cross entropy when the target is **present in C128**, and mean challenger softplus when absent. This hit/miss distinction is candidate membership, not RAW correctness. Training gives queries equal weight; it has no extra B_CAL-correct preservation term, no hard margin restriction, and no baseline-specific break penalty. Frozen B_CAL weights do not freeze final decisions: the learned residual is added to their logits. Source: `src/rc_aslo_xf/rebut_qr_qrr_v1.py:280-317`, `src/rc_aslo_xf/token_competition_v2.py:298-301`, `programs/run_token_competition_v2.py:485-505` (all source SHA values verified against protocol).

Selection requires total inner correct count to exceed B_CAL, then minimizes inner validation loss. A rescue can pay for a break. MULTI fold2 actually selects epoch4 with two rescues and one break, and no eligible zero-break epoch exists in that fit. Therefore a zero-break rule would have forced fallback there; its held effect is an untested counterfactual. For the other fourteen selected inner fits there are zero inner breaks, yet each arm has four held breaks. Thus even observed zero inner breaks do not ensure held preservation. Source: per-fit `result.json` history and `inner_validation.json`; condensed in `per_fit.csv`.

Crucially, **all 15 selected epochs equal the global minimum inner validation loss**, so the accuracy eligibility gate did not change checkpoint choice in this run. It cannot explain MULTI weakness by having blocked a lower-CE checkpoint. CE itself is not top1 accuracy: MULTI fold1 chooses 26/31 at epoch5 (CE .2256), although epoch7 reaches 27/31 (CE .2697); fold2 chooses 18/24 at epoch4 although epoch7 reaches 19/24. These alternatives are not held-evaluated results or a justification for post hoc reselection.

## Optimization versus generalization

All fits learn: every training update has a nonzero reader gradient; all except each stage's first update have a nonzero stem gradient. The initial zero stem gradient is expected from the zero-initialized output layer. Update totals equal sample count times completed epochs; no incomplete-training explanation is supported.

All 15 histories show training loss continuing down after the selected epoch while validation CE ends above its minimum. Example ANCHOR fold0: selected epoch5 train loss .2462 / validation .3802; final inner epoch15 train .00645 / validation .5744. This is direct evidence of late inner overfitting. However early stopping already selects epoch5, so it does **not** establish that the deployed four breaks are caused by accidentally using the last, overfit checkpoint.

Selected epochs by folds0..4: QR [5,7,7,6,6], ANCHOR [5,6,5,6,6], MULTI [5,5,4,5,4]. MULTI reaches its validation optimum earlier and has weaker inner gains; it is not simply failing to finish learning. The supplied traces cannot establish that training longer, another learning rate, or more data would fix it.

Each inner fit has only 71–86 queries / 28–32 identities and each inner validation has 20–31 queries but **eight identities**. Validation identity sets are disjoint from fitting identities, and held identities are disjoint from both. Across five inner validations there are 130 query occurrences but only 38 unique queries; summed inner gains are selected-on, overlapping observations. The 11,744/11,760 reader parameters and limited identity support make limited generalization plausible, but parameter count alone is not proof of a causal explanation.

The held model is a fresh same-seed outer-TRAIN refit with the outer-phase B_CAL and normalization, trained for the selected epoch count. It is not the saved selected inner model. Each epoch uses 91–112 queries rather than 71–86, increasing updates per epoch by 1.28–1.42x. Thus inner-to-held differences combine unseen-identity transfer and the prescribed refit; the artifacts do not isolate their contributions. Source: `programs/run_token_competition_v2.py:473-475,548-583`.

## Where transfer fails

Selected inner rescue/break occurrences versus final held rescue/break counts: QR 14/0 → 9/4; ANCHOR 15/0 → 9/4; MULTI 10/1 → 7/4. These numbers describe different populations and models, not paired inner/held observations.

Fold3 is the clearest shared failure: inner gains +3 / +4 / +3 for QR / ANCHOR / MULTI become zero rescues and one break for every arm on its 16 held queries. Fold2 yields one rescue and one break for every arm. MULTI fold0 has inner +2 but no held rescues or breaks; ANCHOR and QR each gain one. MULTI therefore already has weaker selected inner performance and loses additional gains in transfer.

Average held objective still improves for all three arms: B_CAL .59594 → QR .34779 / ANCHOR .37224 / MULTI .37626. Fold3 likewise has lower mean held loss despite a net top1 loss. Average probabilistic fit and per-query action preservation are distinct; this is consistent with the missing preservation constraint, not a finding of global optimization failure.

Inherited B_CAL tau is secondary and was not the selection target. Its ordering cannot replace fixed0 or prove a better primary model. Threshold/residual-scale interaction requires the separate score/action analysis.

## Verification and reproducibility

Run `python /path/to/result/cause_analysis/training/analyze_training.py --root /path/to/result` from any directory, or omit arguments to use the result tree containing the script. It uses only Python standard libraries, original saved logits/metrics, the already opened current-panel analysis mapping, and SHA-verified frozen source text. It writes only its derived training outputs; no model forward, training, scheduler operation, or protected-data access occurs.

For an exported checkout retaining `RC/programs`, `RC/src`, and `RC/results/rc_token_competition_f128_v2`, the same command works with `--root` pointing to its result directory. For another layout, add `--source-root /path/to/checkout/RC`. The resolver maps only the known original RC prefix to that source directory, checks the relative filename and original expected SHA, and records both the unchanged declared binding and resolved filename. It never falls back to absolute workspace sources if a checkout source is absent. Required result inputs are `protocol.json`, `joined_predictions.json`, `candidate_predictions.json`, `analysis/per_query_diagnostics.csv`, all 15 per-fit `result.json` / `progress.json` / `inner_validation.json`, all 247 `inner/epoch*.json`, and the 15 independent receipt JSON files. Model weights, token caches, images and external role files are not needed.

`portability_check.json` records an executed copied-checkout test: all 15 fits and 247 epochs pass, all three derived CSVs are byte-identical, and removing an exported source fails without reading the original workspace copy. This test copied only the declared JSON/CSV/script inputs and five frozen source files.

The replay independently reconstructs all 247 inner epoch losses and decisions: maximum loss difference 7.77e-16; rescue/break/correct counts all match. It reproduces all 15 selected epochs, verifies progress/update accounting, matches per-fit result SHA against existing independent receipts, and checks five frozen source hashes. Existing model-forward receipts report zero replay error; a new model forward was not performed by this analysis. No implementation error was found by these checks; they are not an exhaustive code proof.

Outputs: `per_fit.csv` (15 fits), `epoch_trajectories.csv` (inner plus outer-refit epochs), `per_query_loss.csv` (selected-inner and held losses/actions), `audit_summary.json` (checks, bindings, script SHA), and this note. Original experiment artifacts are unchanged.
