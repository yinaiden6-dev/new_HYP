# RoMa-v2/ColNomic visibility-XF fresh full-C128 gate V1

Date: 2026-08-31

Predecessor: `ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_V2_GO`.

## Population

Use `inner_fold==1`, exclude every six-case and balanced32 execution, sort by
`sha256("ROMA_COLXF_FULLC128_V1|query_id|execution")`, take the first 32.
Selection does not read target presence, correctness, rank or score.

## Prejoin

Four shards independently score all natural 128 candidates for eight queries.
Each candidate records the unchanged visibility-XF REAL score, half-axis
query/reference spatial controls, visibility mass and map hashes.  No role or
label file is opened.  All four target-free shards must commit before join.

## Postjoin decision

After all score seals close, join natural target roles without insertion.
Report local full-C128 top-1/MRR/margin, frozen ColNomic top-1/MRR,
rescue/break/wrong-to-wrong and target-versus-own-strongest-rival C/P effects.

Representation headroom requires at least 30 naturally present targets,
local top-1 at least 21, positive mean target margin, at least 20 REAL>C_QUERY
and 20 REAL>C_REFERENCE directions, and distinct target/rival maps throughout.

- If headroom passes and rescue exceeds break, authorize direct full-C128
  confirmation.
- If headroom passes but rescue does not exceed break, authorize only a new
  target-free HOLD/SWITCH action gate.
- Otherwise stop this visibility-XF family.

No opened/sealed endpoint or P0 is accessed.

