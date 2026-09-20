"""Memory-bounded C_DINO LOCAL training path for CBNR.

The deployed/diagnostic three-arm decoder remains
``decode_c_dino_pair_v2``.  CBNR's loss consumes only the LOCAL arm, so this
module reproduces exactly that root/mask/reducer path while allowing the same
global assignment streaming used by REAL.  It never changes the full-C128
donor plan or the destination P locks.
"""

from __future__ import annotations

from typing import Mapping

import torch

from .dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_QUERY_LOCAL_COMPONENTS,
)
from .dino_rcde_cw1_multitile_vdecode_v1 import (
    BINDING_REAL,
    FIXED_DIRECTIONS,
    FixedDenominatorArmEvidenceV1,
    OrderedRootArmTermV1,
    QueryTokenFieldV1,
    RootPairDecodeReceiptV1,
    TERM_H0,
    TERM_NO_COMPARABLE_ROOT,
    TERM_READY,
    _decode_pair_masks,
    _zero_term,
)
from . import dino_rcde_sr0_mt_controls_v1 as _controls
from . import dino_rcde_sr0_mt_v_runtime_v2 as _runtime
from . import dino_rcde_track_r_control_decoder_v2 as _decoder
from .dino_rcde_sr0_mt_p_runtime_v1 import LOCK_H0, ROOT_READY


SCHEMA_VERSION = "rc_dino_rcde_cbnr_control_runtime_v1_20260824"
STREAMING_CHUNK_SIZE = 64
C_DINO_TRAINING_STREAMING_CHUNK_SIZE = None


def decode_real_local_training_v1(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, _runtime.CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    streaming_chunk_size: int = STREAMING_CHUNK_SIZE,
) -> FixedDenominatorArmEvidenceV1:
    """Decode only REAL's loss-bearing LOCAL arm with the V2 reducer."""

    if type(streaming_chunk_size) is not int or streaming_chunk_size <= 0:
        raise _runtime.SR0MTVContractError("CBNR streaming chunk drift")
    left, right = _runtime._v1._validate_decode_inputs(
        query, locks, left_key, right_key
    )
    model_sha, comparator_sha, _ = _runtime.bind_runtime_lineage(model)
    adapter = _runtime.DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    forward = tuple(
        _runtime.decode_direct_arm_term(
            adapter,
            query,
            left,
            right,
            direction=direction,
            arm_name=ARM_QUERY_LOCAL_COMPONENTS,
            streaming_chunk_size=streaming_chunk_size,
        )
        for direction in FIXED_DIRECTIONS
    )
    reverse = tuple(
        _runtime.decode_direct_arm_term(
            adapter,
            query,
            right,
            left,
            direction=direction,
            arm_name=ARM_QUERY_LOCAL_COMPONENTS,
            streaming_chunk_size=streaming_chunk_size,
        )
        for direction in FIXED_DIRECTIONS
    )
    return _runtime._reduce_arm(
        ARM_QUERY_LOCAL_COMPONENTS,
        left_key,
        right_key,
        forward,  # type: ignore[arg-type]
        reverse,  # type: ignore[arg-type]
    )


def _local_term_streaming(
    adapter: _runtime.DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner: _controls.CDinoCandidateBindingV1,
    opponent: _controls.CDinoCandidateBindingV1,
    *,
    direction: str,
    streaming_chunk_size: int | None,
) -> tuple[
    OrderedRootArmTermV1,
    tuple[_controls.ReferenceMaskTransportReceiptV1, ...],
]:
    owner_lock = owner.direction_locks[direction]
    opponent_lock = opponent.direction_locks[direction]
    union = owner_lock.query_union_mask
    if owner_lock.status == LOCK_H0:
        return (
            _zero_term(
                arm_name=ARM_QUERY_LOCAL_COMPONENTS,
                owner_key=owner.candidate.candidate_key,
                opponent_key=opponent.candidate.candidate_key,
                direction=direction,
                status=TERM_H0,
                query_mask=union,
                exemplar=query.layers,
            ),
            (),
        )
    total = query.layers.new_zeros(query.valid_patch_mask.numel())
    coverage = torch.zeros_like(query.valid_patch_mask, dtype=torch.int64)
    for owner_root in owner_lock.selected_roots:
        coverage += owner_root.query_mask.to(torch.int64)
    decode_receipts: list[RootPairDecodeReceiptV1] = []
    transport_receipts: list[_controls.ReferenceMaskTransportReceiptV1] = []
    decoded: list[int] = []
    for root in owner_lock.ordered_root_ordinals:
        owner_root = owner_lock.all_roots[root]
        opponent_root = opponent_lock.all_roots[root]
        if opponent_root.binding_status != ROOT_READY:
            continue
        if not torch.equal(owner_root.query_mask, opponent_root.query_mask):
            raise _runtime.SR0MTVContractError(
                "CBNR C_DINO destination P locks disagree on a fixed query root"
            )
        owner_reference, owner_transport = (
            _controls.transport_reference_mask_positive_overlap(
                owner_root.reference_mask,
                owner_root.reference_grid_shape,
                owner.candidate.grid_shape,
                owner.candidate.valid_patch_mask,
                p_lock_record_sha256=owner_lock.p_lock_record_sha256,
            )
        )
        opponent_reference, opponent_transport = (
            _controls.transport_reference_mask_positive_overlap(
                opponent_root.reference_mask,
                opponent_root.reference_grid_shape,
                opponent.candidate.grid_shape,
                opponent.candidate.valid_patch_mask,
                p_lock_record_sha256=opponent_lock.p_lock_record_sha256,
            )
        )
        transport_receipts.extend((owner_transport, opponent_transport))
        if not owner_transport.decoder_legal or not opponent_transport.decoder_legal:
            continue
        pair, receipt = _decode_pair_masks(
            adapter,
            adapter,
            query,
            owner.candidate,
            opponent.candidate,
            query_mask=owner_root.query_mask,
            owner_reference_mask=owner_reference,
            opponent_reference_mask=opponent_reference,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=None,
        )
        qmask = owner_root.query_mask.to(pair.contributions.device)
        total = total + pair.contributions * qmask.to(pair.contributions.dtype)
        decode_receipts.append(
            RootPairDecodeReceiptV1(
                root_ordinal=root,
                owner_component_source_root_ordinal=root,
                opponent_component_source_root_ordinal=root,
                binding_mode=BINDING_REAL,
                query_mask_sha256=receipt.query_mask_sha256,
                owner_reference_mask_sha256=receipt.owner_reference_mask_sha256,
                opponent_reference_mask_sha256=(
                    receipt.opponent_reference_mask_sha256
                ),
            )
        )
        decoded.append(root)
    if not decoded:
        return (
            _zero_term(
                arm_name=ARM_QUERY_LOCAL_COMPONENTS,
                owner_key=owner.candidate.candidate_key,
                opponent_key=opponent.candidate.candidate_key,
                direction=direction,
                status=TERM_NO_COMPARABLE_ROOT,
                query_mask=union,
                exemplar=query.layers,
            ),
            tuple(transport_receipts),
        )
    reciprocal = torch.zeros_like(total)
    covered = coverage > 0
    reciprocal[covered.to(reciprocal.device)] = 1.0 / coverage[covered].to(
        device=reciprocal.device, dtype=reciprocal.dtype
    )
    patch = total * reciprocal
    qmask = union.to(device=patch.device)
    patch = patch * qmask.to(patch.dtype)
    scalar = patch / qmask.sum().to(patch.dtype)
    return (
        OrderedRootArmTermV1(
            arm_name=ARM_QUERY_LOCAL_COMPONENTS,
            owner_candidate_key=owner.candidate.candidate_key,
            opponent_candidate_key=opponent.candidate.candidate_key,
            direction=direction,
            status=TERM_READY,
            query_mask=union,
            patch_evidence=patch,
            scalar_contributions=scalar,
            logit=scalar.sum(),
            decoded_root_ordinals=tuple(decoded),
            decode_receipts=tuple(decode_receipts),
        ),
        tuple(transport_receipts),
    )


def decode_c_dino_local_training_v1(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, _controls.CDinoCandidateBindingV1],
    left_key: str,
    right_key: str,
    *,
    streaming_chunk_size: int | None = C_DINO_TRAINING_STREAMING_CHUNK_SIZE,
) -> tuple[FixedDenominatorArmEvidenceV1, str]:
    """Decode only the loss-bearing LOCAL arm with autograd and streaming."""

    if streaming_chunk_size is not None and (
        type(streaming_chunk_size) is not int or streaming_chunk_size <= 0
    ):
        raise _runtime.SR0MTVContractError("CBNR streaming chunk drift")
    left, right = _decoder._validate_pair_only_c_dino_inputs(
        query, locks, left_key, right_key
    )
    model_sha, comparator_sha, _ = _runtime.bind_runtime_lineage(model)
    adapter = _runtime.DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    forward: list[OrderedRootArmTermV1] = []
    reverse: list[OrderedRootArmTermV1] = []
    receipts: list[_controls.ReferenceMaskTransportReceiptV1] = []
    for direction in FIXED_DIRECTIONS:
        term, rows = _local_term_streaming(
            adapter,
            query,
            left,
            right,
            direction=direction,
            streaming_chunk_size=streaming_chunk_size,
        )
        _decoder._validate_local_transport_term(
            left, right, direction=direction, term=term, receipts=rows
        )
        forward.append(term)
        receipts.extend(rows)
        term, rows = _local_term_streaming(
            adapter,
            query,
            right,
            left,
            direction=direction,
            streaming_chunk_size=streaming_chunk_size,
        )
        _decoder._validate_local_transport_term(
            right, left, direction=direction, term=term, receipts=rows
        )
        reverse.append(term)
        receipts.extend(rows)
    arm = _runtime._reduce_arm(
        ARM_QUERY_LOCAL_COMPONENTS,
        left_key,
        right_key,
        tuple(forward),  # type: ignore[arg-type]
        tuple(reverse),  # type: ignore[arg-type]
    )
    transport_sha256 = _decoder._transport_receipt_sha256(
        left_key=left_key,
        right_key=right_key,
        arm_name=ARM_QUERY_LOCAL_COMPONENTS,
        receipts=tuple(receipts),
    )
    return arm, transport_sha256


__all__ = [
    "SCHEMA_VERSION",
    "STREAMING_CHUNK_SIZE",
    "C_DINO_TRAINING_STREAMING_CHUNK_SIZE",
    "decode_c_dino_local_training_v1",
    "decode_real_local_training_v1",
]
