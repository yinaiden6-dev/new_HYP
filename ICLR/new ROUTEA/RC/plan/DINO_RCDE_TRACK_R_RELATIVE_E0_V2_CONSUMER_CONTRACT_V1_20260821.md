# DINO-RCDE Track R relative E0 V2 consumer contract V1

Date: 2026-08-21  
Stage: `TRACK_R_RELATIVE_E0_V2_CONSUMER_QUALIFICATION`  
Scientific claim: none

## 1. Purpose

E0 qualifies one explicit Natural/Core P-lock V2 consumer for the existing
three-arm DINO-RCDE runtime.  It tests schema, provenance, masks, fixed
denominator, controls and numerical execution only.

E0 does not read target/rival roles, RAW score/rank/outcome, does not train on
natural labels and does not make a scientific GO/NO-GO decision.

## 2. Authority lineage

The future E0 authority must bind both:

- V114 as the current live parent and the immutable closure of the failed
  exact-geometry donor family;
- V111 plus its strict P-V2 aggregate validation as the explicit data fork.

This is a new relative-only successor, not a V114 donor repair.  It may not
relax, reconstruct or consume a donor ledger.

## 3. Frozen fixture

The producer has two layers.

### 3.1 Synthetic core

- two candidates;
- two directions;
- one legal head;
- at least four canonical roots;
- one connected multi-root selected row;
- one small shared RCDE model/decoder;
- no natural identity or role.

The fixture executes all three arms and one optimizer step on a synthetic
relative pair loss solely to prove finite gradient and update plumbing.  This
does not count as natural training.

### 3.2 Result-blind natural schema smoke

Natural ordinals are fixed before execution by the canonical query-source hash,
two per outer fold.  For each query E0 validates the complete C128 axis, then
materializes only hash-fixed candidate pairs.  It never chooses a pair using
identity, label, score, rank, proposal status or model output.

Each query uses its legal outer-refit head.  At least one separately fixed query
also exercises a legal inner-OOF head.  Natural smoke may deserialize the
registered DINO tokens and matched R1 CONTEXT checkpoint and perform forward
only when the future authority explicitly permits it.

## 4. V2 projection boundary

The adapter must first call `validate_natural_p_lock_v2`, then decode
`core_lock_record` using `candidate_p_lock_v2_from_record`.

For the current frozen 608,256-record population only, it may emit an explicit
V1-runtime compatibility view because every record is proposal-READY and the
independent count of `ROOT_REFERENCE_MISSING` is zero.  The adapter must persist
a projection receipt; it is never an implicit schema cast.

Any `ROOT_REFERENCE_MISSING` causes exact
`TRACK_R_V2_REFERENCE_MISSING_REQUIRES_NATIVE_RUNTIME_ABORT`.  It may not be
turned into V1 H0.

## 5. Mandatory checks

E0 producer and independent validator must separately establish:

1. exact natural wrapper, nested core and record hashes;
2. exact head/fit/crossfit-role selection;
3. complete candidate axis and two directions;
4. query/source/grid/valid-mask/token-cache closure;
5. canonical-geometry SHA and token-cache geometry SHA as separate namespaces;
6. V1 and V2 mask hashes independently recomputed from the same mask bytes;
7. query union equals the OR of selected structural query roots;
8. full-reference/local-component query masks are byte-identical;
9. full-reference masks contain every and only valid reference patch;
10. local components retain root address and complete overlap denominator;
11. exact zero outside registered masks;
12. candidate reorder and pair swap exact contracts;
13. dense/streaming equivalence within the registered tolerance;
14. C_DINO_V, P_QUERY and P_REFERENCE each alter only their registered input;
15. model state is unchanged by natural forward;
16. target/rival/identity/score/rank/slot/winner/gap/outcome and protected reads
    are all zero.

No REAL-versus-control direction is inspected in E0.

## 6. Poison tests

Tests must fail closed for:

- V1/unknown wrapper or core schemas;
- extra target/rival/outcome fields;
- wrong fit/head/role/fold;
- missing or duplicated candidate/direction;
- source, physical row, grid or token hash mismatch;
- canonical geometry SHA substituted for token-cache geometry SHA;
- V2 mask SHA substituted for V1 P-mask SHA;
- selected-row/core-row hash namespace confusion;
- nonfinite token or decoder output;
- injected reference-missing selected root;
- available-root or available-patch renormalization.

## 7. Output and states

Required immutable outputs:

```text
fixture_manifest.json
synthetic_receipt.json
natural_projection_receipts.pt
three_arm_receipts.pt
control_isolation_receipts.json
access_audit.json
result.json
independent_validation.json
pytest.xml
```

PASS:

```text
DINO_RCDE_TRACK_R_RELATIVE_E0_V2_CONSUMER_READY
```

FAIL:

```text
DINO_RCDE_TRACK_R_RELATIVE_E0_ENGINEERING_ABORT
```

Both states retain:

```text
training_authorized        = false
heldout_scoring_authorized = false
scientific_GO_or_NO_GO     = null
automatic_stage_advance    = false
next_authorized_stage      = null
```

PASS may name only
`next_eligible_contract=TRACK_R_INNER_OOF_EPISODE_LEDGER_PREFLIGHT`.

## 8. Resource and reproducibility boundary

Partition, node and GPU model are runtime receipts, not scientific/model/data
hash inputs.  E0 may run on an available dev or accelerated GPU partition with
the same frozen token/checkpoint bytes and deterministic numerical contract.

