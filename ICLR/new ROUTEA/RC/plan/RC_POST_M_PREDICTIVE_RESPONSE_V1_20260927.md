# Frozen POST response: prospective new-M prediction, v1

This is a computational prediction test, declared before evaluating any new endpoint. It predicts the downstream effect **given M**; it does not predict RoMa's generation of M, establish identity sufficiency, or claim untouched-query generalization.

## Existing evidence and incremental question

`rc_mass_information_transport_v1` already fitted label-free, TRAIN16 polynomial surrogates of a frozen POST model's real-versus-constant, fixed-reference-hit content change, and evaluated the previously opened PROBE8. That positive result remains evidence. Its coefficients belong to that small-panel model and are not interchangeable with the five H593 fold models.

The present increment is prospective local extrapolation around each candidate's original HR1 M, through the actual BF16 adapter/projector and free MaxSim, with explicit predictions of matching-location changes. The previous level/gap four-cell intervention is not repeated.

## Fixed population and inputs

- All 120 target-present queries / 45 components in the completed level-gap panel; the same fixed two candidate references per query. These groups and their original outcomes are already opened.
- Original held-fold POST_REAL, projection and INTERNAL3 parameters; original cached hidden states and reference tokens. No training, new encoder/RoMa/LLM forward, GPU computation, threshold fitting or new image collection.
- Candidate prediction is independent: it sees its own cached content, baseline M and local probes. Target/wrong roles are withheld from prediction functions and joined only for evaluation. Candidate selection itself is inherited and label-conditioned, so this is not target-free full-C128 performance validation.

## Interventions frozen before endpoint evaluation

Let x = log M and t be the additive shift from the original HR1 x.

- Probes: t = [-0.02, -0.01, 0, 0.01, 0.02].
- Unseen endpoints: t = [-0.08, -0.04, 0.04, 0.08].
- All actual inputs M0 exp(t) must satisfy 0 < M <= 1. No clipping or selective coordinate replacement.
- Paired diagnostics at radii 0.04 and 0.08 use (+r,+r), (-r,-r), (+r,-r), (-r,+r), where candidate order is ascending physical candidate position, never target identity.
- These are interface interventions, not necessarily realizable upstream image edits. No endpoint is reused as a local probe or used to choose its radius.

## Predeclared predictors

Every predictor returns each candidate's content score, without identity labels:

1. CONSTANT: the measured baseline score.
2. SHARED_LINEAR: one slope, weighted equally across components, fit to probe-only score changes with zero response at t=0.
3. SHARED_QUADRATIC: the corresponding common slope/curvature, using the same probes and weights. This is the current-model analogue of the historical scalar-surrogate family, not a refit of the old model.
4. LOCAL_LINEAR: centered finite-difference score slope at h=0.02.
5. LOCAL_QUADRATIC: that slope plus centered score curvature at h=0.02.
6. PATCH_LINEAR_MAX: centered first differences of normalized projected query tokens, then reference dot products and MaxSim on the predicted similarities.
7. PATCH_QUADRATIC_MAX: the corresponding second-order token forecast, then reference dot products and MaxSim.

The h=0.01 probe estimates are saved as a predeclared slope/curvature consistency diagnostic, not used to choose a predictor, radius or winner. No final normalization is applied to polynomial token forecasts: they forecast the already normalized token path, not a new inference model. The shared-linear paired-gap prediction depends only on the difference of the two x shifts; its common-shift effect is exactly zero.

The patch predictors forecast per-patch winning reference positions. A predicted unchanged winner is a prospective flag, not a mathematical guarantee. Actual endpoint winner changes and nonlinear score response are revealed only after the prediction seal. BF16 casting, normalization and MaxSim mean a global differentiable Taylor remainder is not assumed.

The token derivative is also dotted with the baseline winning and runner-up reference tokens. This directly measures how the predicted token motion aligns with each candidate's reference content; its mean is compared with the scalar score slope. A second pre-endpoint stability flag is `baseline top1-top2 gap > 2 * norm(predicted normalized-token displacement)`. Unit reference norms make this sufficient for the **forecast** to keep the same winner, not for the actual nonlinear model. Actual switch rates and prediction errors inside/outside that flag are all reported. No result-dependent threshold is fitted.

## Mandatory chronology and artifact closure

1. `prepare`: seal this protocol, program and exact cache/model/history bindings; no new forward.
2. `worker --stage probe`: compute only the five local conditions and replay the original HR1 endpoint; save normalized query/reference tokens, all patch MaxSim/argmax, baseline top-two scores, fixed-baseline-winner scores, exact M and head.
3. `seal`: require all 120 probe receipts; fit only the shared coefficients, compute all seven endpoint predictions and predicted matching winners, and hash every prediction before any endpoint forward. No target/wrong metadata is read.
4. `worker --stage endpoint`: require the complete global prediction seal, then compute the four new conditions, saving the same patch evidence.
5. `join` and `check`: independently reconstruct scores, forecasts and original INTERNAL3 actions from saved arrays. The check does not rerun any model.

Per-candidate atomic checkpoints; eight deterministic shards, four CPUs, ten-minute allocations with a 430-second soft budget. Exit 75 means resume, not scientific failure. At most 240 x 9 = 2160 new lightweight candidate POST/projector/MaxSim calls; old full-C128 content vectors are reused for action reconstruction. Estimated total worker wall time 3–20 minutes excluding queueing; estimate is not a measured benchmark. Expected saved arrays approximately 1–2 GB. No new GPU forward.

## Endpoints and reporting rules

- Primary: candidate content-change MAE/RMSE for every predictor and every signed radius, averaged within query then equally across components; paired group-bootstrap differences versus CONSTANT and SHARED_LINEAR (10,000 draws, fixed seed 20260927). Report all predictors; no test-set model selection.
- Secondary: target-minus-fixed-wrong content-gap change MAE and direction agreement in every paired pattern. Zero is predeclared as absolute change <= 1e-6 for sign reporting; always report zero counts and all continuous errors.
- Matching: patch winner forecast accuracy, missed switches, spurious switches, actual switch fractions, and content forecast error split by the pre-endpoint predicted-stability flag. This is descriptive calibration, not a proof of a no-switch radius.
- Original INTERNAL3: predicted versus actual action gaps and full-C128 chosen-position agreement when only the two preselected references are intervened upon and all other reference scores stay at original HR1. These synthetic, label-conditioned interventions are not new retrieval accuracy or rescue claims.
- Report both radii and the small-versus-large-radius error comparison. A stronger extrapolation claim requires improvement over shared-slope and constant with paired exploratory intervals, not merely high correlation with the baseline score.
- A successful local prediction does not show that the adapter has acquired new identity information, remove RoMa, establish a universal common direction, or provide an efficient replacement. Probe cost is reported.

Finite-difference coefficient fitting and prediction occur before new endpoint values. An after-the-fact path integral or algebraic closure is not counted as prospective evidence. Conclusions remain conditional on the frozen fold models, opened population, local log-M radii and cached representation.
