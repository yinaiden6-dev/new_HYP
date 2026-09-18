# RoMa-v2/ColNomic no-regret action gate V1

Date: 2026-08-31

The standalone full-C128 local scorer is closed at 19/32, with 3 rescues and
11 breaks.  Its new strongest wrong has median ColNomic rank 56, so fixed
local ranking and equal-rank fusion are not deployable.

Train one shared seven-parameter linear switch gate.  For challenger `c` and
frozen ColNomic winner `w`, six antisymmetric target-free features are:

1. standardized ColNomic score gap;
2. normalized visibility-XF local-score difference;
3. normalized visibility-mass difference;
4. normalized local-similarity (`score/mass`) difference;
5. normalized query-spatial robustness difference;
6. normalized reference-spatial robustness difference.

The head outputs one switch logit.  Logit `<=0` is exact HOLD; otherwise the
single challenger with maximum logit replaces the winner.  Labels enter only
weighted BCE during training; HOLD/baseline-correct examples have weight four.

Qualification first trains fold3 and evaluates fold4, then reverses.  Each
direction requires at least 24/32 pair accuracy, at least 10/16 rescues and at
least 14/16 correct holds.  Only if both pass may a combined head be evaluated
on the disjoint inner-fold1 full-C128 score seals.  Final headroom requires
top-1 at least 28/32 and rescue strictly greater than break.  This is internal
actionability evidence, not P0 or an untouched endpoint.

