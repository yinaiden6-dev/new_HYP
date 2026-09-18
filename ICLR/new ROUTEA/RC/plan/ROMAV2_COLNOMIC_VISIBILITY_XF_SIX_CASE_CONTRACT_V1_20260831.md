# RoMa-v2/ColNomic visibility-XF six-case contract V1

Date: 2026-08-31

RoMa alone is not an exact-instance scorer: its six-case target advantage was
3/6 for overlap, 2/6 for hard-visible mass and 3/6 for cycle.  Visual audit
showed that it nevertheless localizes the correct object/panel and gives
similar masks to true near-duplicate references.  This successor tests the
pre-registered complementary decomposition:

- RoMa provides candidate-conditioned soft visibility on query/reference;
- frozen ColNomic patch cosine provides exact-content evidence only inside
  that visibility.

For ColNomic query/reference cell visibility `wq,wr`, the only REAL score is:

`sqrt(mean(wq)*mean(wr)) * sum_i(wq_i*max_j(wr_j*cos(q_i,r_j))) / (sum_i wq_i+eps)`.

No alpha, temperature, top-K or threshold is scanned.  Two matched controls
cyclically shift query or reference visibility weights by one flattened cell,
preserving their multisets and all token content.

Six-case headroom requires target score above the deployed competitor in at
least 4/6, target REAL above both spatial controls in at least 4/6, complete
finite maps and candidate-distinct score receipts.  It is zero-training
diagnostic evidence only and cannot modify retrieval or authorize P0.

