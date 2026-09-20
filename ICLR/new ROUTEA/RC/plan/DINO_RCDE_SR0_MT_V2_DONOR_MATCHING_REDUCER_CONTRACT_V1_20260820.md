# V2 geometry-matched donor reducer contract V1

This stage consumes all 50 independently validated V113 projection shards and
constructs the 16 frozen scope graphs defined by the V2 donor contract.

The geometry key, exact-clean target locator, metadata inequality predicate and
scope populations are immutable. No pooling, field, tolerance, threshold or
fallback may be changed after viewing projection results.

For every scope, producer and independent validator must reconstruct:

- recipient population and namespace;
- static metadata edge count and V92 maximum-cardinality closure;
- exact geometry-key edge count and degree sequence;
- deterministic maximum-cardinality partial matching;
- one ledger row per recipient, with exact-null donor fields when unmatched;
- separate `STATIC_ZERO_DEGREE`, `GEOMETRY_ZERO_DEGREE` and `HALL_UNMATCHED`
  reasons; and
- identical matched population for any future repaired/relative-only consumer.

If the global exact-geometry edge count is zero, the only valid decision is:

```text
RCDE_SR0_MT_V2_DONOR_GEOMETRY_NO_ELIGIBLE_EDGE_STOP
donor_ledger_eligible = false
V_training_authorized = false
next_authorized_stage = null
```

This is a mechanism-qualification engineering NO-GO, not a retrieval or paper
scientific result. It does not invalidate the V2 P locks or reference-
conditioned target hypotheses. It forbids tuning this geometry key on the same
594-query population.
