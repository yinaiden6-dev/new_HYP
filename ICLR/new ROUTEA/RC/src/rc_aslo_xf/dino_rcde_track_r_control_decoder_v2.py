"""Append-only Track-R control decoders routed through the V2 runtime.

The historical controls module bound ``decode_direct_arm_term`` and
``decode_pair`` from the V1 runtime at import time.  Consequently its
``C_DINO_V`` and ``P_REFERENCE`` FULL arms retained the union-once decoder even
after the rootwise V2 runtime was added.

This module is the narrow successor boundary:

* ordinary REAL/C_COL/P_QUERY pairs use :func:`decode_pair_v2`, a named wrapper
  around the V2 runtime;
* pair-only, full-C128-planned C_DINO bindings use
  :func:`decode_c_dino_pair_v2`;
* P_REFERENCE uses :func:`p_reference_score_v2` and calls the V2 direct-term
  entry point for every arm and direction.

For C_DINO, ALL and FULL call the V2 direct-term entry point.  LOCAL retains
the frozen mask-transport implementation from controls V1, then fails closed
unless FULL and LOCAL have the same status, ordered roots, query masks, and
fixed-denominator path.  No historical runtime, controls module, materializer,
or launcher is modified here.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

import torch

from .dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
    MANDATORY_ARMS,
)
from .dino_rcde_cw1_multitile_vdecode_v1 import (
    BINDING_REAL,
    FIXED_DIRECTIONS,
    FixedDenominatorArmEvidenceV1,
    OrderedRootArmTermV1,
    QueryTokenFieldV1,
    ThreeArmFixedDenominatorEvidenceV1,
    token_tensor_sha256,
)
from . import dino_rcde_sr0_mt_controls_v1 as _controls_v1
from . import dino_rcde_sr0_mt_v_runtime_v2 as _runtime_v2


SCHEMA_VERSION = "rc_dino_rcde_track_r_control_decoder_v2_20260824"
PARENT_CONTROLS_SCHEMA_VERSION = _controls_v1.SCHEMA_VERSION
PARENT_RUNTIME_SCHEMA_VERSION = _runtime_v2.SCHEMA_VERSION


def decode_pair_v2(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, _runtime_v2.CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> ThreeArmFixedDenominatorEvidenceV1:
    """Decode an ordinary pair through the corrected rootwise V2 runtime."""

    return _runtime_v2.decode_pair(
        model,
        query,
        locks,
        left_key,
        right_key,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )


def decode_all_patch_pair_v2(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, _runtime_v2.CandidatePLockV1],
    left_key: str,
    right_key: str,
) -> FixedDenominatorArmEvidenceV1:
    """Decode only ALL_PATCH without reading either candidate's P lock.

    This is the Phase-B invariance boundary.  It intentionally does not call
    the three-arm runtime validator because that validator reads directional
    P-lock provenance before dispatch.  Only the query and the two candidate
    DINO fields are validated here; the V2 direct-term function's ALL_PATCH
    branch likewise dereferences only ``candidate``.
    """

    if left_key == right_key:
        raise _runtime_v2.SR0MTVContractError(
            "ALL_PATCH pair candidates must be distinct"
        )
    try:
        left = locks[left_key]
        right = locks[right_key]
    except KeyError as error:
        raise _runtime_v2.SR0MTVContractError(
            "ALL_PATCH pair candidate is absent"
        ) from error
    if token_tensor_sha256(query.layers) != query.tokens_sha256:
        raise _runtime_v2.SR0MTVContractError(
            "ALL_PATCH query token tensor mutated after provenance seal"
        )
    for key, lock in ((left_key, left), (right_key, right)):
        candidate = lock.candidate
        if (
            candidate.candidate_key != key
            or candidate.layers.device != query.layers.device
            or candidate.layers.dtype != query.layers.dtype
            or token_tensor_sha256(candidate.layers) != candidate.tokens_sha256
        ):
            raise _runtime_v2.SR0MTVContractError(
                "ALL_PATCH candidate key/content/dtype/device drift"
            )
    model_sha, comparator_sha, _ = _runtime_v2.bind_runtime_lineage(model)
    adapter = _runtime_v2.DeviceMaskRuntimeAdapter(
        model, model_sha, comparator_sha
    )
    forward = tuple(
        _runtime_v2.decode_direct_arm_term(
            adapter,
            query,
            left,
            right,
            direction=direction,
            arm_name=ARM_ALL_PATCH,
        )
        for direction in FIXED_DIRECTIONS
    )
    reverse = tuple(
        _runtime_v2.decode_direct_arm_term(
            adapter,
            query,
            right,
            left,
            direction=direction,
            arm_name=ARM_ALL_PATCH,
        )
        for direction in FIXED_DIRECTIONS
    )
    return _runtime_v2._reduce_arm(
        ARM_ALL_PATCH,
        left_key,
        right_key,
        forward,  # type: ignore[arg-type]
        reverse,  # type: ignore[arg-type]
    )


def _validate_pair_only_c_dino_inputs(
    query: QueryTokenFieldV1,
    locks: Mapping[str, _controls_v1.CDinoCandidateBindingV1],
    left_key: str,
    right_key: str,
) -> tuple[
    _controls_v1.CDinoCandidateBindingV1,
    _controls_v1.CDinoCandidateBindingV1,
]:
    """Accept exactly the two payloads emitted by the full-C128 materializer."""

    pair = (left_key, right_key)
    if (
        left_key == right_key
        or len(locks) != 2
        or set(locks) != set(pair)
    ):
        raise _runtime_v2.SR0MTVContractError(
            "C_DINO V2 requires exactly two materialized pair bindings"
        )
    left = locks[left_key]
    right = locks[right_key]
    if not isinstance(left, _controls_v1.CDinoCandidateBindingV1) or not isinstance(
        right, _controls_v1.CDinoCandidateBindingV1
    ):
        raise _runtime_v2.SR0MTVContractError(
            "C_DINO V2 input is not a materialized pair binding"
        )
    if token_tensor_sha256(query.layers) != query.tokens_sha256:
        raise _runtime_v2.SR0MTVContractError(
            "C_DINO query token tensor mutated after provenance seal"
        )
    for key, binding in ((left_key, left), (right_key, right)):
        candidate = binding.candidate
        if (
            candidate.candidate_key != key
            or candidate.layers.device != query.layers.device
            or candidate.layers.dtype != query.layers.dtype
            or token_tensor_sha256(candidate.layers) != candidate.tokens_sha256
        ):
            raise _runtime_v2.SR0MTVContractError(
                "C_DINO candidate content/device drift"
            )
        if set(binding.direction_locks) != set(FIXED_DIRECTIONS):
            raise _runtime_v2.SR0MTVContractError(
                "C_DINO destination P-lock direction axis drift"
            )
        for direction in FIXED_DIRECTIONS:
            sealed = binding.direction_locks[direction]
            if (
                sealed.query_source_image_sha256 != query.source_image_sha256
                or sealed.query_grid_shape != query.grid_shape
                or sealed.query_geometry_sha256
                != query.geometry_record_sha256
            ):
                raise _runtime_v2.SR0MTVContractError(
                    "C_DINO query/P-lock provenance drift"
                )
    return left, right


def _validate_local_transport_term(
    owner: _controls_v1.CDinoCandidateBindingV1,
    opponent: _controls_v1.CDinoCandidateBindingV1,
    *,
    direction: str,
    term: OrderedRootArmTermV1,
    receipts: tuple[_controls_v1.ReferenceMaskTransportReceiptV1, ...],
) -> None:
    """Close each decoded LOCAL root against its two transport receipts."""

    if len(receipts) != 2 * len(term.decode_receipts):
        raise _runtime_v2.SR0MTVContractError(
            "C_DINO LOCAL mask-transport/root receipt count drift"
        )
    owner_sealed = owner.direction_locks[direction]
    opponent_sealed = opponent.direction_locks[direction]
    for index, decoded in enumerate(term.decode_receipts):
        root = decoded.root_ordinal
        if root not in owner_sealed.all_roots or root not in opponent_sealed.all_roots:
            raise _runtime_v2.SR0MTVContractError(
                "C_DINO LOCAL decoded root is outside the destination P lock"
            )
        owner_root = owner_sealed.all_roots[root]
        opponent_root = opponent_sealed.all_roots[root]
        owner_receipt = receipts[2 * index]
        opponent_receipt = receipts[2 * index + 1]
        if (
            not owner_receipt.decoder_legal
            or not opponent_receipt.decoder_legal
            or owner_receipt.source_grid_shape
            != owner_root.reference_grid_shape
            or opponent_receipt.source_grid_shape
            != opponent_root.reference_grid_shape
            or owner_receipt.destination_grid_shape
            != owner.candidate.grid_shape
            or opponent_receipt.destination_grid_shape
            != opponent.candidate.grid_shape
            or owner_receipt.p_lock_record_sha256
            != owner_sealed.p_lock_record_sha256
            or opponent_receipt.p_lock_record_sha256
            != opponent_sealed.p_lock_record_sha256
            or owner_receipt.token_resize_or_interpolation_count != 0
            or opponent_receipt.token_resize_or_interpolation_count != 0
        ):
            raise _runtime_v2.SR0MTVContractError(
                "C_DINO LOCAL mask-transport receipt binding drift"
            )


def _transport_receipt_sha256(
    *,
    left_key: str,
    right_key: str,
    arm_name: str,
    receipts: tuple[_controls_v1.ReferenceMaskTransportReceiptV1, ...],
) -> str:
    """Preserve the controls-V1 mask-transport receipt payload exactly."""

    return _controls_v1._payload_sha(
        {
            "contract_sha256": _controls_v1.MASK_TRANSPORT_CONTRACT_SHA256,
            "left_candidate_key": left_key,
            "right_candidate_key": right_key,
            "arm_name": arm_name,
            "receipt_count": len(receipts),
            "receipts": [item.logical_record() for item in receipts],
        }
    )


def decode_c_dino_pair_v2(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, _controls_v1.CDinoCandidateBindingV1],
    left_key: str,
    right_key: str,
) -> tuple[ThreeArmFixedDenominatorEvidenceV1, Mapping[str, str]]:
    """Decode a pair-only full-C128 C_DINO materialization via rootwise FULL.

    The complete C128 intervention is planned and receipted by the separate
    materializer.  This device-side boundary deliberately accepts only its two
    anonymous destination bindings; it must never receive or reconstruct the
    full candidate axis.
    """

    left, right = _validate_pair_only_c_dino_inputs(
        query, locks, left_key, right_key
    )
    model_sha, comparator_sha, reducer_sha = _runtime_v2.bind_runtime_lineage(
        model
    )
    adapter = _runtime_v2.DeviceMaskRuntimeAdapter(
        model, model_sha, comparator_sha
    )
    arms = []
    transport_hashes: dict[str, str] = {}
    for arm_name in MANDATORY_ARMS:
        forward: list[OrderedRootArmTermV1] = []
        reverse: list[OrderedRootArmTermV1] = []
        arm_receipts: list[_controls_v1.ReferenceMaskTransportReceiptV1] = []
        for direction in FIXED_DIRECTIONS:
            if arm_name == ARM_QUERY_LOCAL_COMPONENTS:
                term, receipts = _controls_v1._c_dino_paired_term(
                    adapter, query, left, right, direction=direction
                )
                _validate_local_transport_term(
                    left,
                    right,
                    direction=direction,
                    term=term,
                    receipts=receipts,
                )
                forward.append(term)
                arm_receipts.extend(receipts)
                term, receipts = _controls_v1._c_dino_paired_term(
                    adapter, query, right, left, direction=direction
                )
                _validate_local_transport_term(
                    right,
                    left,
                    direction=direction,
                    term=term,
                    receipts=receipts,
                )
                reverse.append(term)
                arm_receipts.extend(receipts)
            else:
                # ALL and, crucially, FULL are dynamically routed through V2;
                # no controls-V1 static runtime entry is reachable here.
                forward.append(
                    _runtime_v2.decode_direct_arm_term(
                        adapter,
                        query,
                        left,  # structural CandidatePLockV1-compatible binding
                        right,
                        direction=direction,
                        arm_name=arm_name,
                    )
                )
                reverse.append(
                    _runtime_v2.decode_direct_arm_term(
                        adapter,
                        query,
                        right,
                        left,
                        direction=direction,
                        arm_name=arm_name,
                    )
                )
        arms.append(
            _runtime_v2._reduce_arm(
                arm_name,
                left_key,
                right_key,
                tuple(forward),  # type: ignore[arg-type]
                tuple(reverse),  # type: ignore[arg-type]
            )
        )
        transport_hashes[arm_name] = _transport_receipt_sha256(
            left_key=left_key,
            right_key=right_key,
            arm_name=arm_name,
            receipts=tuple(arm_receipts),
        )
    evidence = ThreeArmFixedDenominatorEvidenceV1(
        left_candidate_key=left_key,
        right_candidate_key=right_key,
        model_checkpoint_sha256=model_sha,
        pair_comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
        binding_mode=BINDING_REAL,
        arms=tuple(arms),  # type: ignore[arg-type]
    )
    _runtime_v2._assert_matched_regional_structure(evidence)
    return evidence, MappingProxyType(transport_hashes)


def p_reference_score_v2(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, _runtime_v2.CandidatePLockV1],
    left_key: str,
    right_key: str,
    arm_name: str,
    *,
    namespace: str,
    query_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score P_REFERENCE while routing every direct term through runtime V2."""

    # The V2 pair validator supplies the complete key, token, dtype, device,
    # and query/P-lock provenance checks without performing a model forward.
    left, right = _runtime_v2._v1._validate_decode_inputs(
        query, locks, left_key, right_key
    )
    model_sha, comparator_sha, _ = _runtime_v2.bind_runtime_lineage(model)
    adapter = _runtime_v2.DeviceMaskRuntimeAdapter(
        model, model_sha, comparator_sha
    )
    by_direction: dict[
        tuple[str, str], _runtime_v2.CandidatePLockV1
    ] = {}
    for key, lock in ((left_key, left), (right_key, right)):
        for direction in FIXED_DIRECTIONS:
            _, footprint = _controls_v1._base_masks(
                query, lock, direction, arm_name
            )
            if int(footprint.sum()) < 2:
                candidate = lock.candidate
            else:
                candidate = _controls_v1.permute_reference_content(
                    lock.candidate,
                    footprint,
                    namespace=namespace,
                    query_id=query_id,
                    direction=direction,
                    arm_name=arm_name,
                )[0]
            by_direction[(key, direction)] = replace(
                lock, candidate=candidate
            )
    forward = tuple(
        _runtime_v2.decode_direct_arm_term(
            adapter,
            query,
            by_direction[(left_key, direction)],
            by_direction[(right_key, direction)],
            direction=direction,
            arm_name=arm_name,
        )
        for direction in FIXED_DIRECTIONS
    )
    reverse = tuple(
        _runtime_v2.decode_direct_arm_term(
            adapter,
            query,
            by_direction[(right_key, direction)],
            by_direction[(left_key, direction)],
            direction=direction,
            arm_name=arm_name,
        )
        for direction in FIXED_DIRECTIONS
    )
    result = _runtime_v2._reduce_arm(
        arm_name,
        left_key,
        right_key,
        forward,  # type: ignore[arg-type]
        reverse,  # type: ignore[arg-type]
    )
    return result.scalar_contributions, result.logit


__all__ = [
    "SCHEMA_VERSION",
    "PARENT_CONTROLS_SCHEMA_VERSION",
    "PARENT_RUNTIME_SCHEMA_VERSION",
    "decode_pair_v2",
    "decode_all_patch_pair_v2",
    "decode_c_dino_pair_v2",
    "p_reference_score_v2",
]
