# GX R2A fold-1 live-consensus 72-update diagnostic V1

## Question

Does the R2A fold-1 failure mainly reflect an under-trained live 4D consensus,
or does the frozen representation/mechanism remain insufficient after a longer
optimization trajectory?

## Paired trajectory

This diagnostic continues the exact 32-update state to update 72.  It does not
restart training.  The model tensors, AdamW moments, first 32 losses, 24 raw
contexts, initial evaluation, train/evaluation identities, episode order, LR,
weight decay, loss, clipping, and final gates remain unchanged.  The only
changed variable is the total update budget: 32 versus 72.

The 72-update result is compared against the immutable 16- and 32-update
results.  Its purpose is to measure the learning curve, not to select a
publishable checkpoint on this development fold.

## Interpretation

- If 32 and 72 progressively improve held-out wrong margins and 72 passes the
  fixed gates, under-training is a credible bottleneck.  A fresh fold-local
  contract is still required before any scientific promotion.
- If training loss falls but held-out target/rival margins remain flat or
  reverse, additional optimization is not the repair; the evidence,
  representation, or objective is insufficient.
- If correct examples degrade while wrong examples improve, the unresolved
  bottleneck is no-regret calibration rather than raw decodability.

This is an adaptive single-fold development diagnostic.  It cannot establish
scientific GO, target-free C128 retrieval, ownership, or full-gallery rescue.
No further update-budget scan is authorized from this result.

