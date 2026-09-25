# DINO-RCDE CBNR representation qualification contract V1

## 1. Why a new representation objective is necessary

All fixed-readout families are closed:

- Track-R candidate-relative local components: `111 rescue / 107 break`,
  candidate-binding and spatial gates failed;
- component Track-H native readout: no decision flips on the difficult canary;
- standalone Scale-4: `10 rescue / 1 break` but only `10/582` top-1 and zero
  difficult/new_difficult rescues;
- ColNomic RAW plus unit component residual: RAW `466/582` became `464/582`,
  with `1 rescue / 3 break` and negative top-1/MRR gains.

More importantly, on the 582-query confirmation, the target-vs-RAW-rival
component delta was positive for `58/116 = 50%` RAW errors but also for
`252/466 = 54%` RAW-correct cases; their mean deltas were nearly equal.
Therefore the frozen representation does not contain a deployable indicator
of when to overturn ColNomic.  No further scale, threshold, interpolation or
small selector is authorized.

The last justified DINO-RCDE successor changes training semantics rather than
readout: **candidate-binding contrastive no-regret representation learning
(CBNR)**.

## 2. Fixed mechanism

For a natural query, current RAW candidate pair `(g,c)`, and each candidate's
own connected component lock, the shared decoder emits uncentered evidence
`e_g,e_c`.  Convert it to a pair-centered residual:

```text
r = e_g - e_c
delta_g = +0.5 * r
delta_c = -0.5 * r
delta_g + delta_c = 0
```

The deployed pair margin is:

```text
m_final = m_RAW + r
```

This removes the common downward-shift shortcut observed in U4.  The residual
is antisymmetric under candidate swap and cannot change both candidates in the
same direction.

At inference labels are absent.  The system evaluates RAW winner against
natural challengers with the same shared decoder; a later action layer may
select at most one switch or exact HOLD.  This contract qualifies only the
representation/loss, not the final action layer.

## 3. Fixed training loss

Training knows the exact target identity only inside the loss.  Orient every
episode as target `y` versus natural strongest RAW rival `c` and define:

```text
m0      = RAW(y) - RAW(c)
r_real  = centered candidate-specific component margin
r_bind  = the same decoder after candidate-reference content derangement,
          retaining destination P-locks
m_final = m0 + r_real
```

The only loss is:

```text
L_rank = softplus(-m_final)
L_keep = I[m0 > 0] * relu(m0 - m_final)
L_bind = softplus(r_bind - r_real)
L_CBNR = L_rank + L_keep + L_bind
```

- `L_rank` trains retrieval against the actual natural hard rival;
- `L_keep` adds an explicit penalty on top of the retrieval loss whenever the
  residual lowers a ColNomic-correct margin;
- `L_bind` requires correct candidate binding to outperform destroyed binding;
- the C control remains a separately evaluated destruction after training;
- no Q0 absolute unary term, bias, identity embedding, scale or threshold.

The binding control must remain differentiable with respect to decoder
parameters.  It may not train invariance to shuffled binding.

Training batches report RAW-correct and RAW-wrong strata separately and form
their retrieval/no-regret contribution as an equal `0.5/0.5` stratum mean.
This prevents the larger RAW-correct population from hiding failure to rescue
errors.  The binding term has unit weight.  No loss coefficient is scanned.

## 4. Staged funnel

### E0 — core and gradient qualification

Synthetic two-candidate fixtures only.  Hard requirements:

1. pair residual is exact zero-sum and candidate-swap antisymmetric;
2. common evidence translation leaves the residual unchanged;
3. decreasing target support increases `L_rank`;
4. worsening a RAW-correct margin activates the additional unit `L_keep`;
5. making destroyed binding competitive increases `L_bind`;
6. all functional decoder parameters receive finite nonzero gradients;
7. one optimizer step changes parameters and lowers the fixed fixture loss;
8. target label never enters model inputs.

PASS is `CBNR_E0_CORE_READY`; failure is engineering abort only.

### R0 — optimization-only identity-disjoint representation gate

Only after E0 PASS, reuse the immutable Track-R natural pair episode/lock/token
infrastructure.  Outer heldout, opened and sealed outcomes remain inaccessible.
Use existing optimization identities and supergroups with a frozen inner
identity+supergroup cross-fit.  Balance RAW-correct and RAW-wrong episodes in
loss reporting without changing their natural frequencies in deployment.

R0 evaluates heldout optimization groups and must satisfy all:

1. RAW-wrong target-vs-rival direction accuracy at least `0.65`;
2. RAW-correct retention at least `0.98`;
3. rescues strictly exceed breaks and net rescue is positive in every inner
   heldout split;
4. mean `r_real-r_bind > 0` in every split, with group-bootstrap lower bound
   positive overall;
5. difficult-track rescue is at least break;
6. candidate swap, reorder and exact-HOLD invariants pass;
7. no identity-specific parameters or inference-time label fields.

R0 PASS only authorizes a separately frozen outer-OOF action experiment.  R0
failure permanently stops the current DINO-RCDE representation family; it may
not be followed by loss-weight, threshold, top-k or pooling scans.

## 5. Reuse and non-repetition boundary

Reuse is allowed for tokens, candidate-specific P-locks, natural strongest
rivals, identity/supergroup folds, model initialization and optimizer
infrastructure.  The new scientific object is exactly the combination of:

- pair-centered zero-sum residual;
- RAW-margin no-regret correct protection;
- correctly directed candidate-binding contrast.

Without all three, the experiment is a repetition of Track-R/U3 and is
forbidden.

No full-gallery, ownership or spatial claim is authorized by E0/R0.  Spatial
permutation is deferred until candidate-binding qualification passes.
