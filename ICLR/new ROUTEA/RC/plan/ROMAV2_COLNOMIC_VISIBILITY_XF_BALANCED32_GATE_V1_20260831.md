# RoMa-v2/ColNomic visibility-XF fresh balanced32 gate V1

Date: 2026-08-31

Predecessor: `ROMAV2_COLNOMIC_VISIBILITY_XF_HEADROOM` on six opened cases.

Select natural-target-present `inner_fold==3` records excluding every prior
six-case execution.  Within frozen ColNomic correctness strata, sort by
`sha256("ROMA_COLXF_BAL32_V1|query_id|execution")` and take 16 wrong plus 16
correct.  For each query score only target and frozen ColNomic strongest wrong;
the subset is postjoin diagnostic and cannot be called target-free C128.

The score and query/reference cyclic spatial controls are byte-identical to
the six-case predecessor.  No parameters are trained.

Promotion to a separately authorized full-C128 gate requires:

- at least 11/16 wrong episodes choose target (rescues);
- at most 1/16 correct episode chooses competitor (breaks);
- at least 26/32 overall pair accuracy;
- REAL target-minus-competitor margin exceeds query and reference spatial
  control margins on at least 20/32 each;
- all candidate maps are distinct and every score is finite.

