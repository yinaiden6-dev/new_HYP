# RoMa-v2/ColNomic visibility-XF balanced32 V2 spatial-control repair

Date: 2026-08-31

V1 remains `ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_NO_GO`: 15 rescues,
2 breaks and 29/32 pair accuracy passed the representation headroom gates, but
adjacent `roll(1)` controls gave only 16/32 and 17/32 REAL-control directions.

The V1 control is not graph destroying for smooth overlap maps: it moves each
weight to an adjacent flattened cell.  V2 changes no REAL formula, model,
threshold or action gate.  It replaces only each control permutation by a
cyclic shift of `floor(N/2)` cells.  For every `N>1` this is fixed-point-free,
preserves the exact weight multiset and creates large coordinate displacement.

V2 uses a fresh `inner_fold==4` balanced32 population, excluding every V1 and
six-case execution, sorted under namespace `ROMA_COLXF_BAL32_V2`.  Promotion
keeps the original strict gates: at least 11 rescues, at most one break,
26/32 pair accuracy, 20/32 REAL improvements over each spatial control and
32/32 distinct candidate maps.

