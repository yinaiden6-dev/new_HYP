# D1 + three-arm current-runtime diagnostic V1

## Purpose

Restore the missing mechanism comparison without changing the validated raw-
ColNomic C-arm model. Reuse the existing 600-query population under its frozen
identity/supergroup folds. This stage has no HOLD/SWITCH training.

## Base axes

- RAW: current-runtime frozen ColNomic full-gallery scores and natural C128.
- D1: each query uses its assigned fresh fold-local 49,792-parameter query-only
  domain adapter; template and gallery reference tokens remain unchanged. D1
  full-gallery scores define its own winner, gap and natural C128.

Old-runtime D1 score/C128 artifacts are engineering references only. A new
current-runtime D1 bridge is mandatory before comparison.
The bridge must preserve both adapted image tokens and unchanged template
tokens; old D1 companion files contain only C128 row/label hashes and cannot
feed RoMa or the seven-parameter gate directly.

### E0 bridge qualification

Before multi-query materialization, run exactly two target-free opened
executions: one `difficult` raw-frame query and one `new_difficult_train` EXIF-
oriented query. RAW and D1 must use the same current-runtime Sum-MaxSim scorer;
the only allowed numerical difference is replacing raw query image tokens with
the assigned fold-local D1-adapted image tokens. Template tokens and gallery
reference tokens are unchanged.

E0 must serialize and independently reload:

- raw query image tokens;
- D1-adapted query image tokens;
- unchanged query template tokens;
- complete RAW and D1 5,413-row score vectors;
- C128 rows both in retrieval-rank order and physical-row-sorted order.

The physical-row-sorted C128 is the only axis later consumed by RoMa and
`C_BIND`. E0 is engineering-only and reads no query target, identity role or
supergroup.

The July/August historical token cache is not a numerical replay authority for
this bridge. The frozen current-runtime lineage was created precisely because
the historical runtime is not portable to the current runtime. Historical
token cosine and score drift must be recorded as diagnostics, but must not be
used as a GO/NO-GO threshold. The hard comparison is within one process and one
current runtime: RAW and D1 use the same decoded image, encoder output,
template tokens, gallery tokens and scorer; only the fold-local image-token
adapter differs.

The two E0 executions must already exist in the independently validated frozen
current-runtime bridge. Live raw image tokens must replay that authority
exactly. The RAW ranked C128, physical-row-sorted C128, top-1 and the 128 RAW
scores on the authority axis must also replay; score tolerance is fixed at
`1e-6`. This protects the validated RAW+C baseline while allowing the known
historical-runtime drift to remain diagnostic-only.

## Matched local arms

For each base candidate and the same RoMa overlap maps:

- A ALL: `wq=1`, `wr=1`;
- B QUERY: `wq=RoMa(query)`, `wr=1`;
- C PAIRED: `wq=RoMa(query)`, `wr=RoMa(reference)`.

C is the existing successful mechanism and remains byte/hash frozen. A/B are
comparators, not replacements. All arms preserve signed candidate-specific
ColNomic evidence and use the same complete candidate population.

## No-action outputs

Before target join save RAW/D1 full rankings, C128, A/B/C scores, maps and
C_BIND. After join report independently for RAW and D1:

- base R@1/MRR and target-in-C128 recall;
- target versus that base's own strongest wrong for A/B/C;
- candidate-binding destruction;
- fold, identity and supergroup directions.

## Promotion

Only if D1 improves base R@1 or C128 recall without negative fold direction,
or improves C-arm target-versus-own-strongest-wrong margin, may a D1-specific
seven-parameter head be trained. C must also outperform or complement A/B;
otherwise keep the current RAW+C frozen model.

No C-head weights may be applied directly to A/B or a D1 axis because their
score distributions and winner/challenger populations differ. Any trained
heads must be same-capacity, same-loss, same-batch and identity-disjoint.
