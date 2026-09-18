# Matched three-arm final execution audit addendum V1

This append-only successor governs only the final consumer of the already
validated Pair64 V2 and full-negative feature artifacts.  It does not modify
the byte-frozen primary contract or the current-runtime regression addendum
bound by Pair64 V2.

## Fail-closed input and frozen-head authority

Before any optimizer is constructed, the consumer must validate:

- the Pair64 V2 payload, receipt, post-full lineage and independent validator,
  including every wrapper/validator/launcher/contract hash recorded by that
  lineage;
- the full-negative aggregate and every payload/receipt/validation shard seal;
- the TRAIN/EVAL role manifest, every role-shard physical and logical hash,
  and execution/identity/supergroup disjointness;
- the target join's gallery cache, raw-path sequence, legacy-setid sequence,
  corrected identity mapping, repair contract and repair manifest hashes;
- the frozen-C module, source-head artifact and validation envelope, including
  exact feature names, weights, bias, parameter count and zero threshold;
- all base scores and all REAL/C_BIND feature tensors are finite and have the
  frozen C128/challenger shapes.

Any failure before training is a zero-update abort.  In particular, a frozen-C
regression failure must not construct an optimizer.

The role ledger supplies only the hash-bound query identity, supergroup, query
ID and track.  Its legacy RGH candidate axis and target position are not
consumed: the unique target position is recomputed on the sealed current-runtime
C128 axis through the six-hash corrected gallery source.

## Training and feature-degeneracy receipt

For every family and arm, report the observed design matrix formed by the 64
Pair64 rows followed by all 127 challengers of each TRAIN32 row.  Report its
shape, feature rank, rank after adding the bias column, exact-zero columns and
exact-duplicate column pairs.  These are diagnostics, not permission to remove
features or change the registered parameter counts.

At each of the 2,000 registered updates, loss, gradients and updated parameters
must be finite.  Final weights, bias and evaluation logits must also be finite.
No clipping, early stopping, replacement objective or retry is authorized.

## Action replay

Every evaluation row has exactly 128 distinct physical candidates and exactly
127 challenger positions excluding the frozen RAW winner.  The RAW winner and
base ranks use score-descending, physical-row-ascending tie breaking.  The
proposed challenger uses logit-descending, physical-row-ascending tie breaking.
HOLD preserves the full base order.  SWITCH moves the proposed challenger to
rank one and shifts only candidates that previously preceded it down by one.

C_BIND preserves the query, destination candidate axis, RAW scores, target
join and winner; it changes only the sealed local feature bundle.  Its actions,
summary and rescue retention are independently replayed.

The result stores all 32 frozen-C REAL actions and all 32 frozen-C C_BIND
actions, including their finite logits and ranks.  The independent validator
reconstructs and exactly replays both ledgers as well as the pre-existing
current-runtime frozen action authority.

## Decision boundary

The historical frozen C head remains the deployment default.  A newly trained
head may authorize only an external-confirmation *design* when, on the internal
EVAL32 panel, it strictly exceeds frozen C's `27/32`, has more rescues than
breaks and has at most one break.  It must additionally outperform its own
C_BIND result, have at least one REAL rescue and retain fewer than all REAL
rescues under C_BIND.  A tie is not a promotion.  The final model cannot be
replaced and no scientific GO/NO-GO may be claimed by this run.

The independent validator must not import or call producer data-loading,
training, action, summary, ranking or retention functions.  It must reconstruct
those computations and exact output schemas independently.  Only its PASS may
release the conditional next stage above.
