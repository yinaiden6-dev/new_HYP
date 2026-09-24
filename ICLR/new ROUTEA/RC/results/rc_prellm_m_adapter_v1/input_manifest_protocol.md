# Real-M adapter pilot input protocol

2026-09-24. This is a bounded development pilot using the existing H593 fold0,
original RAW C128 candidate sets, original ColNomic tokens, and cached true
RoMa M. It is not a full-fold or external evaluation.

The original four outcome-blind TRAIN examples remain byte-for-byte archived
in `engineering_manifest4.json`, with their validation and preparation source
saved as `engineering_manifest_validation4.json` and `engineering_prepare4.py`.
They remain the first four rows of the expanded manifest and are used for
engineering checks. All four happen to have correct original RAW winners.

The sixteen-query TRAIN pilot intentionally contains eight original RAW-correct
and eight original RAW-incorrect queries. Only fold0 TRAIN identity labels and
original candidate scores determine these strata. Target-absent queries are
ineligible for this small supervised fit; their IDs are retained in the
selection record, and no targets are inserted into any candidate set. After
preserving the original four examples, each stratum is filled in ascending
SHA256 order of `RC_PRELLM_M_ADAPTER_V1_20260924|query_id`. Distinct TRAIN
components are preferred globally; any required repeated component is reported.
No new-model result or held-probe outcome is used for this selection.

The eight original outer-held probe IDs, their candidate axes, and all their
input values are preserved. They were selected only by the same query-ID hash
order. Held target labels and group identities are absent from the manifest;
component counts and outcomes are only reported after predictions are sealed.
The probe remains an opened development diagnostic rather than fresh evidence.

Each TRAIN and probe row retains all 128 original physical candidate IDs,
candidate identities, RAW scores, true M, original free-content L0, reference
token bindings, source hashes, and original query image preprocessing frame.
The same frozen fold0 COST1_REFIT_M1Q0R0 output head is used for every arm;
its true M input remains available regardless of internal adapter condition.

Adapter log-M normalization is fitted only on the 16 x 128 TRAIN mass values:
`log(max(M, 1e-8))`, with the population mean and standard deviation recorded
at `mass_normalization`. No probe input contributes to normalization.

The training schedule is fixed at sixteen optimizer updates, one per TRAIN
query in recorded order, with the final checkpoint predetermined. Probe
outcomes do not select a checkpoint or hyperparameter. Runtime must first pass
fresh-encoder replay parity against the original query tokens, then run the
archived four-query engineering checks. This preparation submits no jobs and
does not expand the run to all 593 queries.
