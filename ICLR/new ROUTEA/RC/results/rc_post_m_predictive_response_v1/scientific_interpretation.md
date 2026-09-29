# Prospective POST response: scientific interpretation

Prospective synthetic log-M endpoints on already opened fixed pairs. Not unseen-query, full-natural-C128 accuracy, upstream M prediction, or unique causal proof.

The prior TRAIN16/PROBE8 scalar transport experiment already established predictive response structure. This experiment tests new local conditions and matching-location forecasts, with a global prediction barrier.

For a fixed matching index, content-score response is the normalized query-token movement projected onto the selected reference token. With free MaxSim, switching reference indices adds another contribution. Scalar M alone does not set the response sign: token motion and reference content jointly do so. The following endpoint tests determine how well local measurements predict that response.

## Candidate prediction errors

| model | shift | MAE | MAE minus shared slope [95% group CI] |
|---|---:|---:|---|
| CONSTANT | -0.04 | 0.0018937853 | 0.00028278393 [1.6242822e-05, 0.00054514576] |
| CONSTANT | -0.08 | 0.0037646836 | 0.00053251261 [2.0063918e-06, 0.0010591066] |
| CONSTANT | 0.04 | 0.0019192134 | 0.00031832823 [5.3876664e-05, 0.00057748977] |
| CONSTANT | 0.08 | 0.0038613919 | 0.00066889402 [0.00013983101, 0.0011870594] |
| LOCAL_LINEAR | -0.04 | 2.2917198e-05 | -0.0015880841 [-0.0017719755, -0.0014153109] |
| LOCAL_LINEAR | -0.08 | 8.5139529e-05 | -0.0031470314 [-0.0035094495, -0.0028040215] |
| LOCAL_LINEAR | 0.04 | 2.3422859e-05 | -0.0015774623 [-0.001766988, -0.0014011683] |
| LOCAL_LINEAR | 0.08 | 8.8956838e-05 | -0.0031035411 [-0.0034859432, -0.002747933] |
| LOCAL_QUADRATIC | -0.04 | 2.01958e-05 | -0.0015908055 [-0.0017743595, -0.0014187251] |
| LOCAL_QUADRATIC | -0.08 | 8.8844269e-05 | -0.0031433267 [-0.0035034237, -0.0028016511] |
| LOCAL_QUADRATIC | 0.04 | 1.8760868e-05 | -0.0015821243 [-0.0017703248, -0.0014062535] |
| LOCAL_QUADRATIC | 0.08 | 8.498268e-05 | -0.0031075152 [-0.0034860087, -0.0027549362] |
| PATCH_LINEAR_MAX | -0.04 | 2.345117e-05 | -0.0015875502 [-0.0017718171, -0.0014136969] |
| PATCH_LINEAR_MAX | -0.08 | 8.812261e-05 | -0.0031440484 [-0.0035086257, -0.0028001745] |
| PATCH_LINEAR_MAX | 0.04 | 2.4332298e-05 | -0.0015765529 [-0.0017656025, -0.0013994995] |
| PATCH_LINEAR_MAX | 0.08 | 9.3087652e-05 | -0.0030994103 [-0.0034824063, -0.0027413173] |
| PATCH_QUADRATIC_MAX | -0.04 | 2.0135171e-05 | -0.0015908662 [-0.0017742496, -0.0014187289] |
| PATCH_QUADRATIC_MAX | -0.08 | 0.00011883603 | -0.0031133349 [-0.0034766208, -0.0027727175] |
| PATCH_QUADRATIC_MAX | 0.04 | 1.8962381e-05 | -0.0015819228 [-0.0017706936, -0.0014060065] |
| PATCH_QUADRATIC_MAX | 0.08 | 0.00011758647 | -0.0030749114 [-0.0034577079, -0.0027191061] |
| SHARED_LINEAR | -0.04 | 0.0016110013 | 0 [0, 0] |
| SHARED_LINEAR | -0.08 | 0.003232171 | 0 [0, 0] |
| SHARED_LINEAR | 0.04 | 0.0016008852 | 0 [0, 0] |
| SHARED_LINEAR | 0.08 | 0.0031924979 | 0 [0, 0] |
| SHARED_QUADRATIC | -0.04 | 0.0016091173 | -1.8840467e-06 [-4.1220765e-06, 3.4698501e-07] |
| SHARED_QUADRATIC | -0.08 | 0.0032246348 | -7.536187e-06 [-1.6488306e-05, 1.38794e-06] |
| SHARED_QUADRATIC | 0.04 | 0.0016025355 | 1.6503311e-06 [-5.8357284e-07, 3.8807139e-06] |
| SHARED_QUADRATIC | 0.08 | 0.0031991958 | 6.6978927e-06 [-2.2848709e-06, 1.5691474e-05] |

## Candidate-specific response under a common M change

A shared linear slope predicts zero paired-gap change for the same shift applied to both candidates. Candidate-local reference-alignment slopes predict a nonzero gap when their responses differ. This tests a measurable content-dependent condition rather than relabeling M as quality.

| shift | actual absolute gap change | alignment forecast MAE | MAE minus shared-zero [95% group CI] |
|---:|---:|---:|---|
| -0.08 | 0.0037382076 | 0.00010879452 | -0.0036294131 [-0.0045158219, -0.0028140433] |
| -0.04 | 0.0018714241 | 2.8749715e-05 | -0.0018426744 [-0.0022843017, -0.0014342947] |
| 0.04 | 0.0018762484 | 3.1664025e-05 | -0.0018445843 [-0.0022851872, -0.0014388727] |
| 0.08 | 0.0037491744 | 0.00011477373 | -0.0036344007 [-0.004519507, -0.0028288991] |

## Matching forecast, with no-switch baseline

| model / shift | winner accuracy | no-switch accuracy | difference [95% group CI] | forecast bound-stable fraction |
|---|---:|---:|---|---:|
| PATCH_LINEAR_MAX/-0.08 | 0.991740 | 0.978143 | 0.013597 [0.010718, 0.016543] | 0.365520 |
| PATCH_QUADRATIC_MAX/-0.08 | 0.958384 | 0.978143 | -0.019760 [-0.022555, -0.016877] | 0.064953 |
| PATCH_LINEAR_MAX/-0.04 | 0.995156 | 0.987894 | 0.007263 [0.005723, 0.008913] | 0.563085 |
| PATCH_QUADRATIC_MAX/-0.04 | 0.990762 | 0.987894 | 0.002868 [0.001329, 0.004525] | 0.400643 |
| PATCH_LINEAR_MAX/0.04 | 0.995443 | 0.988455 | 0.006988 [0.005556, 0.008492] | 0.563085 |
| PATCH_QUADRATIC_MAX/0.04 | 0.990146 | 0.988455 | 0.001691 [0.000518, 0.002895] | 0.398844 |
| PATCH_LINEAR_MAX/0.08 | 0.991293 | 0.977764 | 0.013529 [0.010782, 0.016394] | 0.365520 |
| PATCH_QUADRATIC_MAX/0.08 | 0.957559 | 0.977764 | -0.020204 [-0.022683, -0.017684] | 0.064438 |

## Evidence boundary

- Probe slope equals derivative/reference alignment only within finite-difference and assignment constraints; that algebra itself is not prospective evidence.
- Useful evidence is its performance on new M endpoints sealed after predictions, compared to shared and constant predictors.
- Forecast no-switch flags are not actual-model guarantees; miss/switch counts must accompany overall winner accuracy.
- This predicts the frozen response given M. It does not explain why upstream M contains identity evidence or remove dependence on RoMa.
- Coordinates, opened 120/45 population, two selected candidates and local 4/8-percent log shifts limit the conclusion.
- Additional geometry summaries are descriptive; no model is selected by these endpoint results.
