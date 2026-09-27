# Route A RC — CW0 Reference-First Connected-Region Contract V2

Date: 2026-08-09  
Status: **FROZEN FOR E0 ONLY; NATURAL TRAINING NOT YET AUTHORIZED**  
Seed: `17`  
Supersedes: `CW0_CONNECTED_WINDOW_PROPOSAL_RECALL_CONTRACT_V1_20260809.md`
before any V1 natural training or scientific endpoint was run

## 1. Decision and claim boundary

The raw connected proposer is engineering-valid but puts the true reference in
an any-direction legal H1 on only `7/32` fixed optimization-role queries and a
joint two-direction H1 on only `3/32`.  Training only after
that hard selection is circular: a target with no raw slot receives no local
representation gradient.  V2 therefore trains a shared representation on soft
connected hypotheses first and regenerates strict H1 only after the adapter is
frozen.

V1 is retained as an immutable query-first matched arm.  It is not a natural
training entry point because it starts from a query seed and reuses the same
query cells in fit and verification.  V2 fixes both issues.  Passing this
contract can qualify a proposal mechanism; it cannot by itself establish
target determination, ownership, retrieval gain, or an 80% scientific success
probability.

## 2. The target is one complete connected region

For candidate `g` and hypothesis `h`, the latent object is exactly

\[
h=(W_q,W_r,\phi),
\]

where `W_q` and `W_r` are immutable Boolean membership masks and `phi` is one
of two fixed cross-fit phases.

Hard invariants:

1. `W_q` and `W_r` each contain `8..64` cells, have two-dimensional span, and
   are exactly one 4-neighbour-connected component.
2. A seed patch only initializes the region.  Seeds, anchors, high-score
   cells, attention, reliability and verification rows never redefine target
   membership.
3. The candidate reducer marginalizes mutually exclusive connected
   hypotheses.  It never unions patches from different hypotheses.  The
   reported target mask is the single MAP non-H0 hypothesis.
4. Every verification member remains in a fixed region-size denominator.
   Reliability may reduce its signed contribution but may not remove the cell
   or renormalize the remaining hotspots.  Signed per-cell evidence is clipped
   to `[-4,4]` before multiplication by reliability.
5. An H1 requires region-level reciprocal effective mass, two-dimensional
   covariance and both cross-fit directions.  A single patch or a disconnected
   patch set presented in place of `W_q/W_r` cannot authorize H1.  Sparse
   hotspots inside a legal region remain bounded by the full-region
   denominator and must still pass the effective-mass/covariance gates.  If the
   complete region cannot be supported under these rules, the exact result is
   H0/HOLD.

This definition permits partial occlusion without pretending that the
remaining visible hotspots themselves are the complete target.  Connectivity
belongs to the selected target membership; uncertainty is represented inside
that connected object and can force hypothesis-level abstention.

No human mask, box, point, polygon, target-derived pseudo-mask or morphology
bridge is used.

## 3. Inputs and forbidden shortcuts

- frozen ColNomic spatial query/reference tokens;
- natural fold-local C128 in its frozen order;
- one shared identity-free rank-8 asymmetric adapter, two reliability heads
  and one shared dustbin logit: exactly `4,357` trainable scalars;
- no D1 score, rank, slot, winner flag or gap before the later action stage;
- no global pooled pair feature, candidate/slot embedding, identity table,
  oracle target insertion, Qwen, opened, sealed or outer endpoint;
- labels enter only after target-free hypothesis tensors and hashes exist, and
  only select the positive candidate in losses/diagnostics.

## 4. Reference-conditioned exhaustive connected hypotheses

### 4.1 Fixed reference windows

For each candidate reference grid, enumerate the centre cell of every fixed
`4x4` macro tile in row-major order.  For a tile with half-open row interval
`[y0,y1)` and column interval `[x0,x1)`, the seed is exactly
`(floor((y0+y1-1)/2), floor((x0+x1-1)/2))`; this also fixes partial boundary
tiles without a floating tie.  This gives `28..56` seeds on the audited O32 grids and
guarantees that every valid reference cell is within graph distance at most
four of at least one seed.

Around each reference seed, construct `W_r` by canonical 4-neighbour BFS:

```text
radius = 6
cap = 64
minimum cells = 8
queue order = (graph distance, row-major cell)
```

No descriptor score, target, candidate rank, top-k or learned seed selector may
change this enumeration.

### 4.2 Fixed exhaustive query windows

The query grid uses the same result-blind macro-seed rule as the reference:
one exact seed per `4x4` macro tile, with the same
`floor((start+stop-1)/2)` coordinate formula.  Around every seed, build one
canonical radius-6/cap-64 `W_q`.  On the registered `24x32` query grid this is
exactly 48 overlapping connected windows; every query cell is within graph
distance at most four of at least one seed.

For every candidate, retain the complete Cartesian product

\[
H_g=\{(W_q^a,W_r^b):a=1\ldots N_q,\ b=1\ldots N_r(g)\}.
\]

There is no descriptor-conditioned centre, argmax, top-k, NMS, learned seed,
hash deduplication or hard H1 deletion in the warm-start forward.  Therefore:

- a correct region that is not the initial winner still receives gradient;
- two drug boxes remain separate hypotheses instead of being averaged into a
  background centre;
- held-out verification evidence cannot influence where its own region is
  placed;
- each candidate is judged by how its own complete reference windows explain
  each complete query window.

Exact mask hashes are used for integrity and MAP reporting only.  They never
change a hypothesis's scoring weight.

### 4.3 One connected object, two fixed cross-fit phases

Both complete windows are split by the fixed macro checkerboard

\[
b(x,y)=(\lfloor y/4\rfloor+\lfloor x/4\rfloor)\bmod 2.
\]

Two phases are retained because an unknown affine mapping need not preserve
checkerboard colour:

```text
phase 0, direction 0: fit Q_A x R_A; verify Q_B x R_B
phase 0, direction 1: fit Q_B x R_B; verify Q_A x R_A
phase 1, direction 0: fit Q_B x R_A; verify Q_A x R_B
phase 1, direction 1: fit Q_A x R_B; verify Q_B x R_A
```

Forbidden cross-bank mass is exactly zero at the fit/verify API.  In every
direction, fit-query IDs and verify-query IDs are disjoint and fit-reference
IDs and verify-reference IDs are disjoint.  Both phases and both directions
reference byte-identical full `W_q/W_r`; no direction may select a new region.

## 5. Scoring and no-island protection

For each direction, a weighted ridge affine is fit only on its fit endpoint
pair.  Verification reads all cells in the held-out reference bank and only
the held-out query bank.  It uses reciprocal mass, predicted-coordinate
reprojection, cycle, dustbin failure, conditional entropy and explicit query
and reference reliability.  Zero return has cycle penalty `1`, never zero.

For held-out cells `V_h`, the direction score is

\[
S_{h,d}=\frac{1}{|V_h|}\sum_{j\in V_h}
\rho_j\,\operatorname{clip}(e_j,-4,4).
\]

The denominator is `|V_h|`, not `sum(rho)` and not the number of positive
cells.  A low-reliability bridge therefore cannot be deleted to make distant
hotspots dominate.  A phase score is the mean of its two directions.  The
hypothesis score is the normalized log-mean-exp of the two phase scores at
temperature `0.10`.

To remove the previously observed legal-H1-count shortcut, reduction is
hierarchical and uses means, never sums:

\[
Z_b={1\over N_q}\sum_a \exp(S_{ab}/0.10),\qquad
\overline Z_g={1\over N_r(g)}\sum_b Z_b,
\]

\[
G_g=0.10\log[0.5+0.5\,\overline Z_g].
\]

The fixed H0 prior is `0.5`.  If every `S_h` is zero, `G_g` must be bit-exact
zero.  Uniformly replicating the complete query-window axis, or uniformly
replicating the complete reference-window axis, any positive integer number of
times must leave the mathematical score unchanged; an explicit all-zero branch
makes the floating result exact.  This is not an arbitrary single-slot
replication invariant: duplicating only one hypothesis changes its empirical
measure under an arithmetic mean.  Requiring invariance to that operation
would require unique-mask reweighting and would contradict the frozen
`deduplication_for_scoring=false` contract.  Grid shape or uniform execution
replication cannot become candidate priors.

Hard H1 after adapter freeze additionally requires, in both directions:

```text
full Wq/Wr exact single 4CC and 2-D span
fit unique query/reference endpoints >= 4
verify unique query/reference endpoints >= 4
reciprocal effective matches >= 8
source and target covariance minimum eigenvalue >= 1e-4
affine condition < 1e4
reprojection RMSE < 0.04 and max error < 0.08
orientation preserving
the two direction fits predict the same complete Wr within RMS < 2 query-cell diagonals
```

No reliability threshold creates a second learned mask.  Instead, insufficient
mass/coverage makes the complete hypothesis H0.

## 6. Warm-start objective aligned to hard H1

All soft quantities are computed before label join.  Every hard condition above
has the following fixed dimensionless signed slack before the `0.10` soft-min:

```text
effective matches:          (N_eff - 8) / 8
source covariance:          (lambda_src - 1e-4) / 1e-4
target covariance:          (lambda_tgt - 1e-4) / 1e-4
affine condition:           (1e4 - condition) / 1e4
reprojection RMSE:          (0.04 - RMSE) / 0.04
maximum reprojection error: (0.08 - max_error) / 0.08
orientation:                determinant / 0.10
direction agreement:        1 - disagreement_MSE
                                / (4*query_cell_diagonal_squared)
```

For one phase, `disagreement_MSE` is the fixed-denominator mean over every
cell coordinate in the immutable complete `Wr` of the squared distance between
the two independently fitted `R->Q` affine predictions.  It reads no label,
reliability, nearest-neighbour subset or descriptor-selected endpoint.  The
hard phase passes only when both directions pass their individual gates and
this MSE is finite and strictly below four query-cell-diagonal-squared.  The
two phases remain alternative checkerboard hypotheses and are not compared to
each other.

The differentiable eligibility is the `0.10` soft-min of these dimensionless
slacks followed by a `0.10` sigmoid.  Membership and endpoint counts are fixed
structural constants.  Every listed soft gate has the same pass direction as
its hard gate; no unlisted mass/count gate may be silently added.

For the target candidate, `L_prop=-log(mean_h(e_h)+1e-12)`.  The hierarchical
mean makes it invariant to uniform replication of a complete query or
reference hypothesis axis, not to duplicating an arbitrary single slot.  The
label selects the target candidate only after all `e_h` are materialized
target-free.  This prevents the adapter
from improving a soft retrieval loss while leaving all hard target slots
absent without reintroducing an H1-count shortcut.

The once-frozen objective is:

```text
L = 1.00 L_rank_all_C128
  + 0.50 L_prop
  + 0.25 L_candidate_binding
  + 0.25 L_spatial_permutation
  + 0.50 L_same_reference_crossview
  + 0.10 L_noncollapse
```

`L_rank_all_C128` uses the exact target and all different-identity natural C128
rivals.  Training C/P maps use namespaces disjoint from frozen evaluation C/P
maps and require real target evidence to exceed destroyed evidence by the
fixed margin.  The 4,724 legal same-reference multiview episodes prefer a
common candidate-bound geometric explanation over a same-shape wrong
reference.  No detached strongest-rival, supplemental target, identity-specific
parameter or target spatial supervision is allowed.

## 7. Staged authorization

### E0-Core — synthetic mathematical core only

Required before any natural training:

- exact reference-first seed enumeration and full reference-cell coverage;
- fixed query-window enumeration covers every query cell and retains two-object
  alternatives rather than averaging them;
- every mask is one dense 4CC; bridge removal makes the altered mask illegal;
- A/B exact-disjoint on both endpoints and both phases use identical masks;
- no fit/verify cross-bank mass;
- single-patch/disconnected membership and effective support below eight
  cannot beat H0;
- reliability scaling, fixed denominator, signed clipping and zero-return
  failure are numerically exact;
- known affine connected candidate beats near-rival, COMMON, C, P and
  DISCONNECTED fixtures;
- dense/one-candidate-streaming error `<=1e-7`, candidate reorder
  equivariance, target-free hash invariance, all `4,357` gradients finite with
  absolute magnitude above `1e-12`, and one real update.

Status is `CW0V2_E0_CORE_READY` or `CW0V2_E0_CORE_ABORT`.  Passing only
authorizes E0-Objective; it does not authorize natural S0.

### E0-Objective — synthetic training-path qualification

Required after E0-Core and before any natural read:

- implement every Section 6 soft gate with the registered sign, scale,
  `0.10` soft-min and `0.10` sigmoid; no hard argmax/H1 mask may enter the
  gradient path;
- implement hierarchical `L_prop`, all-C128 pairwise rank, candidate-binding,
  spatial-permutation, same-reference cross-view and noncollapse terms with the
  frozen weights, while labels remain a post-materialization selector only;
- on a fixture whose correct connected region is not the initial soft MAP,
  the fixed objective gives that region and target candidate finite nonzero
  gradient and improves their margin after the registered update sequence;
- known connected affine evidence beats COMMON, C, P and DISCONNECTED after
  training; destroying binding or geometry cannot improve the trained
  positive margin;
- full candidate scores, soft eligibility, loss, gradients and updated
  parameters agree between dense and one-candidate streaming within `1e-7`;
- every trainable scalar has finite gradient with absolute value above
  `1e-12`, and all access/hash/reorder invariants remain valid.

Status is `CW0V2_E0_READY` or `CW0V2_E0_OBJECTIVE_ABORT`.  Only
`CW0V2_E0_READY` authorizes S0.  Both E0 layers remain synthetic engineering
qualifications and carry no target-determination or retrieval claim.

### S0 — two optimization-role queries

Only after E0.  Require all 128 candidates, finite loss and gradients, target
gradient even when raw target H1 is absent, no target/D1/Qwen/outer/opened/
sealed access in prejoin, and a replayable checkpoint/receipt.  It has no
retrieval or proposal GO claim.

### O32 — reused optimization-panel proposal screen

The existing 32-query panel has already been target-joined and directly caused
the query-first mechanism to be replaced.  It is therefore permanently an
`optimization/development` panel, not an untouched P32 and not independent
scientific evidence.  Its identities and supergroups are excluded from later
OOF scoring.  It cannot select a checkpoint or change any V2 constant.

After the fixed-budget adapter is frozen, regenerate strict H1 target-free,
freeze artifacts/hashes, then join labels.  Development qualification requires
all of:

1. joint two-direction target H1 coverage `>=26/32` (raw joint reference is
   `3/32`; raw any-direction is `7/32`);
2. every target H1 satisfies full connected membership and double-endpoint
   cross-fit contracts; single-patch/disconnected counts are zero;
3. target local score beats the strongest natural rival on `>=20/32`;
4. target-rank and target-margin improvements outnumber degradations versus
   RAW and COMMON, with positive group-balanced mean increment;
5. frozen C and P each reduce the real target-rival margin on `>=20/32`, and
   group-balanced mean `REAL-control > 0`;
6. candidate reorder, label-mutation and all access-boundary checks pass.

Failure is `CW0V2_OPTIMIZATION_PROPOSAL_NO_GO`; no threshold, seed count,
radius, cap, temperature, pooling, loss-weight or backbone change is allowed on
O32.  Passing authorizes five-fold inner OOF only.  It does not authorize
SWITCH/HOLD, ownership, opened or sealed evaluation.

Any later untouched panel is named `P32_external`, receives a new manifest and
SHA before use, and must be identity- and supergroup-disjoint from all training,
calibration and O32 data.  If no such panel exists, the next evidence is only
the pre-registered five-fold group OOF followed by a newly collected external
dataset.

## 8. Probability statement

The design removes the known circular slot-selection failure, held-out
selection leak, non-differentiable relocation bottleneck, H1-count shortcut and
disconnected target membership.  Natural exact-pair supervision may still be
too weak to learn local geometry.  O32 cannot update a scientific success
probability because it has already influenced the design.  A probability near
0.8 becomes defensible only conditionally after the mechanism passes E0, S0,
all pre-registered five-fold group-OOF gates and an untouched external panel.
