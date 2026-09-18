# Matched three-arm post-seal diagnostic V1

## Scope

After all 64 target-free A/B/C payloads are sealed and independently
validated, join the frozen query roles and measure local evidence selectivity.
No model is trained and no threshold, pooling rule or candidate width is
selected here.

## Metrics

For RAW and legacy-D1-portability C128 independently, and for A ALL, B QUERY
and C PAIRED, report on ALL/TRAIN/EVAL:

- strict target rank, R@1 and MRR under local arm scores;
- target versus the arm's strongest wrong margin;
- target versus the frozen full-gallery base's strongest wrong margin;
- shift-64 C_BIND strict rank and margin;
- REAL-minus-C_BIND margin direction and retention;
- C-versus-A and C-versus-B rank/margin direction.

EVAL 32 is primary. TRAIN is descriptive only. Per-track and EVAL heldout-fold
summaries are mandatory. Physical rows are reduced with the corrected
5,413-row/5,412-identity mapping, maximum-over-row pooling and pessimistic
strict ties.

C_BIND top-1 retention is conditional on REAL top-1: report the REAL top-1
denominator, the intersection also top-1 under C_BIND, and their ratio. If the
REAL top-1 denominator is zero, the ratio is frozen to `1.0` while the zero
denominator remains explicitly reported. C-versus-A/B paired direction is
reported for ALL, TRAIN and EVAL.

## Boundary

RAW is the deployable base. The D1 axis uses a historical-runtime checkpoint
on current-runtime tokens and is a portability diagnostic only. This stage
cannot promote that D1 checkpoint. C_BIND is inference-only and never enters
training.

Independent validation authorizes only identical-capacity seven-parameter
cross-fit for A/B/C. Final arm selection, no-regret retrieval gain and causal
claims require that successor experiment.
