# H128: single-image scores versus pair-specific support

If M(q,r)=a(q)b(r), every four-cycle logM(q,r)-logM(q,s)-logM(t,r)+logM(t,s) is zero. Nonzero beyond the numerical control rules out this single-image multiplicative explanation on the observed entries; it does not establish identity usefulness by itself.

| Input | Query-side candidate range max | Ref-side query range max | Cycle RMS | Cycle max abs |
|---|---:|---:|---:|---:|
| A1J1P1 | 0.3698542 | 0.26522336 | 0.87411378 | 3.7480968 |
| A1J1P0 | 0.14572105 | 0.048002117 | 0.63532382 | 3.1148677 |
| A1J0P1 | 0.38177038 | 0.34677684 | 0.16353198 | 0.66911193 |
| A1J0P0 | 0 | 0 | 1.6379335e-16 | 4.4408921e-16 |

Deterministically retained query-pair cycles: 3047.
For every pair of query rows with >=2 common physical references, take smallest/largest common physical row. Selection is independent of M and labels.

Overlapping cycles are dependent; no significance test or effective sample-size claim.
A-only empirical numerical variation is retained as control, never thresholded to manufacture exact separability.
Nonseparability measures pair interaction, not correctness or unique causality.
The native coarse matcher may provide interaction through J and through derived P; these are not independent information sources.
