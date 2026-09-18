# RoMa-v2/ColNomic full-negative action gate V1

Date: 2026-08-31

The pair-trained seven-parameter gate passed fold3/fold4 cross-fit but failed
full C128 because it never saw third-party challengers.  V1 retains the same
features, head, zero HOLD threshold and fourfold cost for breaks.

Training uses:

- fold3/fold4 balanced target-versus-base-rival pairs for rescue supervision;
- the previously opened inner-fold1 full-C128 population as optimization data;
  each query penalizes its maximum non-target challenger, and a wrong base
  winner additionally requires the true target switch logit to be positive.

Evaluation uses newly sealed inner-fold2 C128 scores.  The identities are
disjoint from inner-fold1 training, but this is adaptive development rather
than untouched evidence because the mechanism was designed after prior Route A
experiments.  Final paper confirmation requires a new external dataset.

Promotion requires final top-1 strictly above frozen ColNomic, rescue greater
than break, at most one break, and at least 30 naturally present targets.

