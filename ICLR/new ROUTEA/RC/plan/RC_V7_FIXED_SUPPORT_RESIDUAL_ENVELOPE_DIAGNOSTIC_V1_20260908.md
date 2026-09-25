# V7 fixed-support residual envelope diagnostic V1

Status: frozen one-formula diagnostic protocol, before opening new diagnostic values. Scope is the existing opened TRAIN32/C128, fixed V7 final checkpoint and saved REAL decisions. No training, formal392 read, threshold/temperature/top-K scan, support reselection, reranking or scientific GO is authorized.

For each candidate retain its saved V7 selected H exactly. With residual e[i,d], positive final weights w_mu[d], w_M[d] and source reference cells j, the original score is

    E = 1 - .5 * (sum_d w_mu[d] * mean_j max_(i in j) e[i,d]
                  + sum_d w_M[d] * max_i e[i,d]).

The only diagnostic score on the same H is

    C = 1 - .5 * (mean_j max_(i in j) sum_d w_mu[d] * e[i,d]
                  + max_i sum_d w_M[d] * e[i,d]).

Every tie among actual atoms resolves to the smallest physical source query index. Reference groups are ordered by source reference index. Empty H retains the original fixed -256 score, with zero excess penalty and no fabricated support.

For every reference group, each channelwise maximum dominates the selected worst actual atom componentwise; the same holds globally. Positive weights therefore prove C >= E over exact reals. Runtime checks this certificate directly. FP64 scoring preserves the frozen E operation order exactly, verifies the saved original full family score vector and selected ordinal, and records mu/global decompositions and rounding residuals. The declared absolute score comparison bound is gamma_(16*(N+128)+128) * (1 + sum_d weights[d] * max_i,d residual[i,d]), with FP64 unit roundoff 2^-53 and N the H atom count. Numerical gaps are never clamped; a negative gap inside this bound is labeled rounding-limited, and any larger contradiction aborts.

All 32/C128 candidate diagnostics are saved and sealed before private target postjoin. For each query, use SAVED original E scores to choose the best target candidate and strongest wrong candidate, tie-breaking by the existing canonical candidate order. Both candidates and both saved H supports stay fixed when computing the C margin. Report crossings of this fixed-pair zero margin and target-versus-wrong excess-penalty mean/median. This is an anchored diagnostic, not a new retrieval accuracy or rescue result. No coherent-score candidate winner is selected.

Input closure reuses the frozen V7 source adapter and original source hashes. The immutable V7 result and any available independent validation receipt are interpreted only after diagnostic prejoin closure. Pending or failed independent validation is reported, never promoted. Runtime uses one CPU Slurm allocation and writes append-only outputs with resumable per-query seals. It does not monitor jobs or advance any stage.
