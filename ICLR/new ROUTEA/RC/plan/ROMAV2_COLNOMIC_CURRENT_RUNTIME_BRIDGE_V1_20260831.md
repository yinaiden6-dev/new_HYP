# RoMa-v2 + ColNomic current-runtime bridge V1

Date: 2026-08-31

The sealed-source portability audit found that current frozen ColNomic weights
preserve grids, top-10, at least 126/128 candidates and score correlation above
0.999 on four result-blind `new_difficult_train` queries, but one exact-label
winner changed relative to the July cache.  The old and current runtime cannot
therefore be mixed in a one-shot sealed endpoint.

This bridge re-materializes, under one current runtime, only the already-opened
32 full-negative training queries and the distinct 32-query adaptive OOF panel.
It keeps the RoMa checkpoint, visibility-XF formula, six target-free features,
seven-parameter linear HOLD/SWITCH form, zero threshold and one-switch limit.
Labels remain unavailable until every target-free C128 record is sealed.

The bridge must reproduce a strict internal no-regret gain with rescue greater
than break and at most one break before sealed scoring is authorized.  It may
refit the same seven parameters on the original training roles because this is
a deployment-consistency rebuild, not external evidence.  It may not change
candidate width, local score, controls, loss weights, threshold or model.

`new_difficult` sealed pixel decode, model scoring and target join remain zero
throughout this bridge.
