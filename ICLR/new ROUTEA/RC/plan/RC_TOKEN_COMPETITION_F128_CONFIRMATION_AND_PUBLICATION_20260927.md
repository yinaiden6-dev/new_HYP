# Native-token QR/QRR: completion, publication and conditional seed confirmation

This document implements the user's instruction to analyze the current completed experiment, publish the result to `https://github.com/yinaiden6-dev/new_HYP.git`, and automatically conduct one follow-up round only if the pre-result gate passes. The follow-up must also be analyzed and published. It does not change the running V2 scientific protocol.

## What the current experiment tests

The population is **F128: acquisition indices 0–127**, with the original identity-grouped five folds and ColNomic's natural C128 candidate sets. It is not the historical EVAL128 panel. The baseline is the phase-correct frozen B_CAL fitted under the same F128 protocol and seed. The original pair support scalar M0 and baseline decision remain available.

Three zero-initialized residual readers use cached native query tokens, their freely matched full-reference content descriptors, and the registered local scalar channels:

- **TOKEN_QR:** reads each query–candidate pair.
- **TOKEN_QRR_ANCHOR:** adds comparison against the original RAW anchor.
- **TOKEN_QRR_MULTI:** adds the registered strongest baseline rival, excluding self and anchor, while all C128 candidates remain eligible.

ANCHOR and MULTI have the same parameter count and initialization. Their direct paired comparison isolates this change in competitor access. Comparing a native-token arm with an older compressed QRR implementation changes more than one factor and must not be described as a single-variable mechanism proof. This experiment uses query-mediated candidate competition; it does not run a new RoMa reference–reference matcher.

## Completion and reporting

All 15 seed0 fits must finish and the existing independent join must report `TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS`. Only then may the post-join analyzer read the final outcome labels. Training completion, independent replay, statistical evidence and GitHub publication are separate statuses.

Primary results use **fixed threshold 0**. The inherited-threshold results are supplementary. The report records full OOF coverage, correct counts, rescues/breaks relative to the same baseline, baseline-correct loss rate, changed decisions, identity-cluster intervals, per-fold differences, selected epochs and any epoch0 fallback. Candidate coverage and target rank strata explain which decisions were eligible for rescue; they do not authorize candidate-set changes.

## Gate frozen before final results

The exact executable policy is `results/rc_token_competition_f128_v2/promotion_policy.json`. Under the default stated to the user before the final join, at least one of the three registered arms must have:

1. Positive net gain over **same-F128 B_CAL, seed0, threshold0**.
2. A strictly positive lower endpoint of the registered identity-cluster 95% gain interval.

This is a **development replication trigger**, not an external GO claim, a test of unique causation, or a familywise-corrected claim that one of several models is superior. MULTI versus ANCHOR is reported separately; passing against B_CAL does not itself establish a MULTI advantage. A failed gate still leads to publication of all completed results, with no next experiment automatically submitted.

## Exactly one conditional next round

If the gate passes, execute all three arms on **seeds1 and2 across all five folds: 30 fits**. Reuse the already validated F128 baseline heads for the corresponding seeds and the original V2 native-token evidence cache. Change no candidate set, fold, features, loss, training budget or selection rule. The additional seeds check stability on the same opened development sample; they are not new independent queries.

The confirmation writes to `results/rc_token_competition_f128_confirm_v1`. It retains original seed0 results and reports all seeds separately and without choosing the best seed. Only after all30 independent replay checks may the confirmation join read outcome labels. Its exact final status is `TOKEN_COMPETITION_F128_CONFIRM_ALL30_INDEPENDENT_JOIN_PASS`.

Scheduling uses 8 CPUs, 32 GB and 10-minute resumable slices. The bulk array starts in cpuonly; a controller fills available dev_cpuonly submission slots using only pending jobs from this new chain. No new encoder/RoMa forward passes are required. The existing seed0 tasks' user-requested accelerated placement is preserved.

## Publication and recovery

The publisher prepares a SHA-verified snapshot, then commits and pushes only this experiment's source, protocol, scalar parameters, complete candidate scores, validation records, tables and reports. Images, backbone/reader weights, token and trace tensors, caches, and err/out logs are excluded. Export validation does not substitute for scientific validation. A source checkout without excluded tensors is not a complete tensor-replay package; manifests retain their original paths and hashes.

Publication uses the separate checkout `github_exports/new_HYP_token128_publication_20260927` so concurrently prepared experiments in the original checkout are untouched. A push counts as complete only after remote main is verified against the publication commit. Both positive and negative scientific results are published.

`programs/complete_token128_pipeline_20260927.py` coordinates the sequence. Its source pins, step receipts, process identity and recovery state are kept in `operations/token128_completion_20260927` at workspace level. It performs at most one confirmation round. On integrity or unrecoverable execution failure it records the error and stops for review; it never turns a failure into a scientific GO.
