# RC-LTH P-only legal-family V3 final result and failure analysis

Date: 2026-09-08  
Population: opened TRAIN32, post-hoc development only  
Final status: `P_RERANKER_SHORTCUT_NO_GO`  
P0/V/action/ownership authorized: no

## Final engineering closure

- Fresh and resumed 512-update checkpoints are byte-identical.
- The reducer completed with exit 0.
- Independent checkpoint reload and full-C128 statistics/taxonomy replay passed.
- Final parity passed on 12,288 REAL/C_BIND/P_COORD candidate-control records.
- Legal seed count/axis hash, complete FP64 seed-score vector, family score,
  MAP seed ordinal/hash and structural-H0 reason are bit-exact between the
  serialized cache, training checkpoint and deployment path.
- Candidate reorder is exact; forbidden prejoin/protected reads and target
  insertion are zero.

Therefore the NO-GO is not caused by the former atom-sparsemax/best-positive
train--deploy mismatch, cache drift, checkpoint drift, serialization drift,
MAP tie breaking, H0 semantics, or reducer arithmetic.

## Frozen gate result

| Gate | Result |
|---|---:|
| Target legal connected family | 32/32 |
| REAL target over own strongest wrong | 21/32 |
| Group-balanced REAL margin | +3.1546 |
| REAL vs all-patch paired net | +5 |
| REAL vs query-only paired net | +1 (required >=4) |
| C_BIND positive margin drop | 27/32 |
| P_COORD positive margin drop | 26/32 |

Only `P_RERANKER_SHORTCUT_NO_GO` fires.  Candidate binding, coordinate
sensitivity, legal connected generation, all-patch separation and candidate
reorder pass, but the complete P-only qualification conjunction does not.

## Why the large margin gain produced only one extra flip

The binary success sets are nested:

- REAL and query-only both pass 20 queries;
- both fail 11 queries;
- REAL alone passes one query;
- query-only alone passes zero.

The continuous scores are not similar.  Mean REAL margin is `+3.077`; mean
query-only margin is `+0.086`; mean REAL-minus-query-only is `+2.991`.
REAL therefore increases confidence on already-correct examples without
changing most decision signs.

An exact log-mean-exp decomposition of target minus strongest-wrong shows:

- on the 21 REAL successes, mean REAL margin is `+5.207`; the largest positive
  terms are precision quality (`+2.215`), assigned cosine (`+1.013`) and
  centered cosine (`+1.013`);
- on the 11 REAL failures, mean REAL margin is `-0.990` while query-only is
  only `-0.021`; almost every learned feature favors the wrong candidate;
- the largest average negative term on failures is cycle quality (`-0.563`),
  followed by assigned/centered cosine and precision;
- every failed target has thousands of legal connected seeds, so this is not
  geometry coverage loss;
- in three failed queries the target MAP seed is slightly stronger than the
  wrong MAP seed, but the complete wrong family has enough broadly high seeds
  to win the required log-mean-exp score.

The shared candidate-conditioned head is thus a strong amplifier: it amplifies
correct specific-reference evidence on 21 queries and systematically amplifies
wrong-reference matchability/quality on the remaining 11.  It does not yet
separate target-specific geometry from generic high-quality correspondence.

## Implementation versus design conclusion

The implemented REAL semantics are closed and parity-correct.  The failure is
a mechanism/data-discrimination limitation under a deliberately strong
query-only comparator and a strict predeclared binary paired-net gate.

The query-only arm does not read candidate geometry, rank or winner.  It uses
one candidate-independent complete query-seed family and evaluates candidate
atom evidence inside those fixed seeds.  This is stronger than the older
fixed-small-region query-only baseline, but it was frozen before this result.
Changing it now or reducing the required paired net would be post-hoc gate
tuning and cannot repair this version.

## Stop and future mechanism boundary

This result may support only the partial statement that the proposed family is
candidate-bound, connected and coordinate-sensitive in margin.  It cannot
support P-only qualification because it fails to establish an indispensable
advantage over query-only regional evidence.

No threshold, temperature, top-K, feature deletion, added layer, checkpoint or
query-only weakening may be tried on TRAIN32.  A successor, if separately
authorized, must be frozen before a new group-disjoint panel and address the
identified failure structurally: distinguish identity-bearing geometry-content
agreement from generic cycle/precision matchability, and prevent a broad wrong
hypothesis family from winning solely through diffuse high-quality matches.
It must retain the same full-C128 controls and parity gates.

