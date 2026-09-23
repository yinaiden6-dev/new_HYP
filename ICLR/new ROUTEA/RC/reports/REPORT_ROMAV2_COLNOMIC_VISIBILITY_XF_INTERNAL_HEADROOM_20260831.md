# RoMa-v2 + ColNomic visibility-XF internal headroom report

Date: 2026-08-31

## Result

The adaptive identity-disjoint inner-fold2 gate reached:

- frozen ColNomic: `24/32`;
- final HOLD/SWITCH decision: `27/32`;
- rescues: `3`;
- breaks: `0`;
- switches: `4` (three correct rescues and one wrong-to-different-wrong).

Status: `ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_HEADROOM`.

Independent validation status:
`ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_INDEPENDENT_VALIDATION_PASS`.  The
validator recomputed source hashes, fold isolation, the seven-parameter
ledger, every HOLD/SWITCH action, all retrieval counts and the evidence
boundary.  All seven checks passed.

## Mechanism

RoMa v2 supplies candidate-conditioned query/reference visibility.  ColNomic
supplies exact-content evidence inside the same visibility.  A seven-parameter
shared gate consumes only target-free winner/challenger differences in frozen
ColNomic gap, local evidence, visibility, local similarity and two spatial
robustness terms.  It may HOLD or perform one SWITCH.

The decisive repair was full-negative training.  Pair-only cross-fit learned
target versus the frozen strongest rival but failed against new third-party
candidates.  Adding complete inner-fold1 C128 challenger populations taught
the gate to suppress those candidates without losing the pair rescue signal.

## Evidence boundary

This is strong internal OOF headroom, not untouched paper evidence.  The
mechanism was developed after extensive use of the RC dataset.  It may justify
freezing one external confirmation contract, but external/sealed data must not
be read until that contract and independent validation close.

The external confirmation contract is now frozen at
`plan/ROMAV2_COLNOMIC_FULL_NEGATIVE_EXTERNAL_CONFIRMATION_CONTRACT_V1_20260831.md`.
Its current authority is contract-only: sealed input binding is the next
stage, while sealed query/label reads and scoring remain unauthorized.

The contract-only preflight subsequently passed all 15 checks with status
`SEALED_INPUT_MANIFEST_BINDING_PREFLIGHT_READY`.  It verified the source
result and independent validation hashes, frozen six-feature/seven-parameter
head, scorer/trainer bytes, RoMa checkpoint, endpoint counts and zero prior
sealed access.  A filename-only search found no pre-existing
`new_difficult` sealed manifest to bind.  Therefore the current hard boundary
is not the model: constructing such a manifest would require a new, explicit
authorization to enumerate sealed metadata.  No sealed query bytes, labels or
scores were read.
