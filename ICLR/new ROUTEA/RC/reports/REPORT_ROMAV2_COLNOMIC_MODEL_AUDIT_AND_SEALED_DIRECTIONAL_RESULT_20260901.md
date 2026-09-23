# RoMa-v2 + ColNomic model audit and sealed directional result

Date: 2026-09-01

## Final result

The one-shot frozen `new_difficult` test endpoint contains 31 primary queries
from 8 identities.  Complete-gallery corrected exact-label evaluation gave:

| path | R@1 | MRR | rescues | breaks | switches |
|---|---:|---:|---:|---:|---:|
| Frozen ColNomic base | 31/31 | 1.0000 | — | — | — |
| REAL frozen gate | 31/31 | 1.0000 | 0 | 0 | 0 |
| C_BIND destruction | 20/31 | 0.8226 | 0 | 11 | 11 |
| Query-coordinate diagnostic | 31/31 | 1.0000 | 0 | 0 | 0 |
| Reference-coordinate diagnostic | 31/31 | 1.0000 | 0 | 0 | 0 |

Registered status: `SEALED_DIRECTIONAL_RETRIEVAL_NO_GO`.

The status follows mechanically because the contract requires strict R@1
improvement.  The base is already perfect, so no model can satisfy that gate
on this endpoint.  REAL preserved all 31 correct decisions and made no action.
This is a ceiling-limited directional NO-GO, not evidence that the mechanism
breaks retrieval and not a positive retrieval-increment result.

## What remains positive

- Original adaptive internal OOF: `24/32 -> 27/32`, 3 rescues, 0 breaks.
- Current-runtime frozen-head OOF: `25/32 -> 27/32`, 2 rescues, 0 breaks,
  with zero retraining and independent action-by-action validation.
- Under sealed C_BIND, wrong reference payloads caused 11 harmful switches,
  while REAL made none.  Candidate/reference binding materially affects the
  frozen decision path.

C_BIND does not by itself establish a retrieval gain because the REAL endpoint
has no errors to rescue.  The Q/R controls are single-axis diagnostics without
an independently materialized S11 cell and do not authorize a strict spatial
ownership claim.

## Audit closure

The audit closed the following issues before target join:

1. The six feature names, ordering, weights, bias, zero threshold and one-
   SWITCH rule were frozen in a standalone module.  All 8,128 internal feature
   vectors matched the predecessor implementation bit-exactly.
2. RoMa checkpoint/source, DINO source, ColNomic encoder fingerprint, gallery
   cache and 5,413-row/5,412-label identity repair were hash-bound.
3. All 31 token, REAL RoMa, full-rank and C_BIND prejoin records independently
   validated before labels were read.
4. Full-rank and C128 scores matched exactly for 31/31 queries with maximum
   absolute difference 0.
5. SWITCH ranking was frozen as move-selected-challenger-to-front with stable
   remainder; HOLD preserves the complete ranking.
6. Every query remains in the denominator, including target-absent C128 and
   wrong-to-wrong cases.
7. Every target row, corrected exact label and reference-image SHA matched at
   join time.
8. Reducer and independent validator produced identical actions, metrics,
   controls and final status.

No target label, filename stem or identity was used by the model or gate.
Semantic source paths were visible to the I/O layer, but static code audit
confirmed that their strings were never consumed as features.

## Scientific boundary and next step

This endpoint is a directional check and is too small and too easy for a paper-
strength effect claim.  The internal OOF result remains evidence of deployable
headroom, while the sealed test establishes no-regret preservation under a
perfect base and strong sensitivity to candidate binding.

The next legitimate confirmation must use either:

- the already opened `difficult` population as a clearly labelled regression/
  diagnostic set, without tuning; and
- a newly untouched external exact-instance population containing natural
  low-margin and base-wrong cases for final paper evidence.

The frozen sealed endpoint must not be reused to tune the threshold, features,
candidate width, loss, pooling or model.
