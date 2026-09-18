# Sealed candidate-binding destruction control V1

For every natural C128 axis, destination candidate position `j` retains its
candidate key, physical row and raw ColNomic score, but consumes the reference
image/tokens from source position `(j + 64) mod 128`.  The permutation is
fixed-point-free and shared by all queries.  RoMa visibility, local ColNomic
evidence and both coordinate controls are recomputed from the mismatched
reference.  No target role or result is read before all outputs are sealed.

This is distinct from candidate reorder invariance: keys do not move in C_BIND;
only the reference payload bound to each key is destroyed.
