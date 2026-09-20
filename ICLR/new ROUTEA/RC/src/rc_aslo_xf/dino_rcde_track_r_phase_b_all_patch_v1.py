"""Phase-B exact ALL_PATCH invariance for REAL versus C_COL bundles.

This append-only helper consumes two target-free :class:`VQueryBundle`
instances, one anonymous pair, and one already loaded V checkpoint.  It proves
that the complete DINO axis and the ALL_PATCH request are byte-identical before
running the P-independent single-arm decoder twice.  FULL, LOCAL, C_DINO,
postjoin, and scientific reduction are outside this module.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_superregion_v2 import ARM_ALL_PATCH
from .dino_rcde_cw1_multitile_vdecode_v1 import (
    FIXED_DIRECTIONS,
    FixedDenominatorArmEvidenceV1,
    token_tensor_sha256,
)
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    VQueryBundle,
    canonical_candidate_axis,
    tensor_sha256,
)
from . import dino_rcde_sr0_mt_v_runtime_v2 as _runtime_v2
from . import dino_rcde_track_r_control_decoder_v2 as _decoder_v2


SCHEMA_VERSION = "rc_dino_rcde_track_r_phase_b_all_patch_v1_20260824"
STATUS = "DINO_RCDE_TRACK_R_PHASE_B_ALL_PATCH_EXACT_INVARIANCE_PASS"
NUMERICAL_POLICY = "SAME_DEVICE_DTYPE_EXACT_BYTES_ZERO_TOLERANCE_V1"
RECEIPT_FIELDS = frozenset(
    {
        "real_request_sha256",
        "control_request_sha256",
        "real_patch_tensor_sha256",
        "control_patch_tensor_sha256",
        "real_direction_sequence_sha256",
        "control_direction_sequence_sha256",
        "real_pair_score_sha256",
        "control_pair_score_sha256",
    }
)


class PhaseBAllPatchInvariantError(RuntimeError):
    """REAL and C_COL cannot share one exact ALL_PATCH request/result."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise PhaseBAllPatchInvariantError(message)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _payload_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    return tensor_sha256(torch.as_tensor(value))


def _query_record(bundle: VQueryBundle) -> dict[str, object]:
    query = bundle.query
    _require(
        token_tensor_sha256(query.layers) == query.tokens_sha256,
        "query DINO token content/provenance drift",
    )
    return {
        "source_image_sha256": query.source_image_sha256,
        "source_key": query.source_key,
        "cache_payload_sha256": query.cache_payload_sha256,
        "geometry_record_sha256": query.geometry_record_sha256,
        "tokens_sha256": query.tokens_sha256,
        "layers_exact_sha256": _tensor_sha256(query.layers),
        "valid_patch_mask_exact_sha256": _tensor_sha256(
            query.valid_patch_mask
        ),
        "grid_shape": list(query.grid_shape),
        "dtype": str(query.layers.dtype),
        "device": str(query.layers.device),
    }


def _candidate_record(lock: object) -> dict[str, object]:
    candidate = lock.candidate  # type: ignore[attr-defined]
    _require(
        token_tensor_sha256(candidate.layers) == candidate.tokens_sha256,
        "candidate DINO token content/provenance drift",
    )
    return {
        "candidate_key": candidate.candidate_key,
        "physical_gallery_row": candidate.physical_gallery_row,
        "source_image_sha256": candidate.source_image_sha256,
        "source_key": candidate.source_key,
        "cache_payload_sha256": candidate.cache_payload_sha256,
        "geometry_record_sha256": candidate.geometry_record_sha256,
        "tokens_sha256": candidate.tokens_sha256,
        "layers_exact_sha256": _tensor_sha256(candidate.layers),
        "valid_patch_mask_exact_sha256": _tensor_sha256(
            candidate.valid_patch_mask
        ),
        "grid_shape": list(candidate.grid_shape),
        "dtype": str(candidate.layers.dtype),
        "device": str(candidate.layers.device),
    }


def _dino_axis_record(bundle: VQueryBundle) -> dict[str, object]:
    axis = canonical_candidate_axis(bundle.locks)
    return {
        "query_id": bundle.query_id,
        "execution_ordinal": bundle.execution_ordinal,
        "outer_fold": bundle.outer_fold,
        "candidate_axis_sha256": bundle.candidate_axis_sha256,
        "candidate_keys": list(axis),
        "query": _query_record(bundle),
        "candidates": [
            _candidate_record(bundle.locks[key]) for key in axis
        ],
    }


def _assert_dino_axis_exact(
    real_bundle: VQueryBundle,
    control_bundle: VQueryBundle,
) -> tuple[dict[str, object], dict[str, object]]:
    real = _dino_axis_record(real_bundle)
    control = _dino_axis_record(control_bundle)
    _require(real == control, "REAL/C_COL complete DINO axis/request drift")

    # Hash equality above is sealed by direct tensor equality here, so this is
    # a zero-tolerance byte claim rather than only producer-declared metadata.
    _require(
        torch.equal(real_bundle.query.layers, control_bundle.query.layers)
        and torch.equal(
            real_bundle.query.valid_patch_mask,
            control_bundle.query.valid_patch_mask,
        ),
        "REAL/C_COL query DINO tensor bytes drift",
    )
    axis = canonical_candidate_axis(real_bundle.locks)
    for key in axis:
        real_candidate = real_bundle.locks[key].candidate
        control_candidate = control_bundle.locks[key].candidate
        _require(
            torch.equal(real_candidate.layers, control_candidate.layers)
            and torch.equal(
                real_candidate.valid_patch_mask,
                control_candidate.valid_patch_mask,
            ),
            "REAL/C_COL candidate DINO tensor bytes drift",
        )
    return real, control


def _request_record(
    *,
    axis_record: Mapping[str, object],
    bundle: VQueryBundle,
    pair_member_keys: tuple[str, str],
    model_checkpoint_sha256: str,
    comparator_sha256: str,
    reducer_sha256: str,
) -> dict[str, object]:
    left_key, right_key = pair_member_keys
    axis = canonical_candidate_axis(bundle.locks)
    position = {key: index for index, key in enumerate(axis)}
    _require(
        left_key != right_key
        and left_key in position
        and right_key in position,
        "anonymous ALL_PATCH pair axis drift",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "arm_name": ARM_ALL_PATCH,
        "pair_member_keys": [left_key, right_key],
        "pair_member_axis_positions": [position[left_key], position[right_key]],
        "directions": list(FIXED_DIRECTIONS),
        "dino_axis_sha256": _payload_sha256(axis_record),
        "query": _query_record(bundle),
        "left_candidate": _candidate_record(bundle.locks[left_key]),
        "right_candidate": _candidate_record(bundle.locks[right_key]),
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "pair_comparator_sha256": comparator_sha256,
        "reducer_sha256": reducer_sha256,
        "adapter_source_sha256": _sha256(
            inspect.getsource(_runtime_v2.DeviceMaskRuntimeAdapter).encode(
                "utf-8"
            )
        ),
        "single_arm_entry_source_sha256": _sha256(
            inspect.getsource(_decoder_v2.decode_all_patch_pair_v2).encode(
                "utf-8"
            )
        ),
        "direct_decoder_source_sha256": _sha256(
            inspect.getsource(_runtime_v2.decode_direct_arm_term).encode(
                "utf-8"
            )
        ),
        "all_patch_term_source_sha256": _sha256(
            inspect.getsource(_runtime_v2._v1._all_patch_term).encode("utf-8")
        ),
        "numerical_policy": NUMERICAL_POLICY,
    }


def _direction_sequence_record(
    arm: FixedDenominatorArmEvidenceV1,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    terms = (
        *(("FORWARD", item) for item in arm.forward_by_direction),
        *(("REVERSE", item) for item in arm.reverse_by_direction),
    )
    for orientation, term in terms:
        _require(
            term.arm_name == ARM_ALL_PATCH
            and term.decoded_root_ordinals == (-1,),
            "Phase-B decoder executed a non-ALL_PATCH regional path",
        )
        output.append(
            {
                "orientation": orientation,
                "direction": term.direction,
                "owner_candidate_key": term.owner_candidate_key,
                "opponent_candidate_key": term.opponent_candidate_key,
                "status": term.status,
                "query_mask_sha256": _tensor_sha256(term.query_mask),
                "patch_evidence_sha256": _tensor_sha256(
                    term.patch_evidence
                ),
                "scalar_contributions_sha256": _tensor_sha256(
                    term.scalar_contributions
                ),
                "logit_sha256": _tensor_sha256(term.logit),
                "decoded_root_ordinals": list(term.decoded_root_ordinals),
                "decode_receipts": [
                    {
                        "root_ordinal": item.root_ordinal,
                        "owner_component_source_root_ordinal": (
                            item.owner_component_source_root_ordinal
                        ),
                        "opponent_component_source_root_ordinal": (
                            item.opponent_component_source_root_ordinal
                        ),
                        "binding_mode": item.binding_mode,
                        "query_mask_sha256": item.query_mask_sha256,
                        "owner_reference_mask_sha256": (
                            item.owner_reference_mask_sha256
                        ),
                        "opponent_reference_mask_sha256": (
                            item.opponent_reference_mask_sha256
                        ),
                    }
                    for item in term.decode_receipts
                ],
            }
        )
    return output


def _assert_arm_exact(
    real: FixedDenominatorArmEvidenceV1,
    control: FixedDenominatorArmEvidenceV1,
) -> None:
    _require(
        real.arm_name == control.arm_name == ARM_ALL_PATCH
        and real.left_candidate_key == control.left_candidate_key
        and real.right_candidate_key == control.right_candidate_key,
        "REAL/C_COL ALL_PATCH pair/arm drift",
    )
    real_terms = (*real.forward_by_direction, *real.reverse_by_direction)
    control_terms = (
        *control.forward_by_direction,
        *control.reverse_by_direction,
    )
    for first, second in zip(real_terms, control_terms, strict=True):
        _require(
            first.status == second.status
            and first.direction == second.direction
            and first.owner_candidate_key == second.owner_candidate_key
            and first.opponent_candidate_key == second.opponent_candidate_key
            and first.decoded_root_ordinals == second.decoded_root_ordinals
            and first.decode_receipts == second.decode_receipts
            and torch.equal(first.query_mask, second.query_mask)
            and torch.equal(first.patch_evidence, second.patch_evidence)
            and torch.equal(
                first.scalar_contributions, second.scalar_contributions
            )
            and torch.equal(first.logit, second.logit),
            "REAL/C_COL ALL_PATCH direction bytes drift",
        )
    _require(
        all(
            torch.equal(first, second)
            for first, second in zip(
                real.signed_term_logits,
                control.signed_term_logits,
                strict=True,
            )
        )
        and torch.equal(
            real.scalar_contributions, control.scalar_contributions
        )
        and torch.equal(real.logit, control.logit),
        "REAL/C_COL ALL_PATCH patch/pair bytes drift",
    )


def phase_b_all_patch_invariance(
    *,
    model: torch.nn.Module,
    real_bundle: VQueryBundle,
    control_bundle: VQueryBundle,
    pair_member_keys: Sequence[str],
    expected_model_checkpoint_sha256: str,
) -> Mapping[str, str]:
    """Run the authorized single-arm exact REAL/C_COL invariance check."""

    _require(
        isinstance(real_bundle, VQueryBundle)
        and isinstance(control_bundle, VQueryBundle),
        "Phase-B inputs must be REAL and C_COL VQueryBundle values",
    )
    pair = tuple(pair_member_keys)
    _require(
        len(pair) == 2
        and all(isinstance(item, str) and bool(item) for item in pair),
        "Phase-B anonymous pair must contain exactly two keys",
    )
    _require(not model.training, "Phase-B V checkpoint must be in eval mode")
    expected = str(expected_model_checkpoint_sha256)
    _require(
        len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected),
        "expected V checkpoint SHA256 drift",
    )
    model_sha, comparator_sha, reducer_sha = _runtime_v2.bind_runtime_lineage(
        model
    )
    _require(model_sha == expected, "Phase-B V checkpoint binding drift")
    real_axis, control_axis = _assert_dino_axis_exact(
        real_bundle, control_bundle
    )
    real_request = _request_record(
        axis_record=real_axis,
        bundle=real_bundle,
        pair_member_keys=pair,  # type: ignore[arg-type]
        model_checkpoint_sha256=model_sha,
        comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
    )
    control_request = _request_record(
        axis_record=control_axis,
        bundle=control_bundle,
        pair_member_keys=pair,  # type: ignore[arg-type]
        model_checkpoint_sha256=model_sha,
        comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
    )
    real_request_sha = _payload_sha256(real_request)
    control_request_sha = _payload_sha256(control_request)
    _require(
        real_request == control_request
        and real_request_sha == control_request_sha,
        "REAL/C_COL ALL_PATCH request bytes drift",
    )

    with torch.no_grad():
        real_arm = _decoder_v2.decode_all_patch_pair_v2(
            model,
            real_bundle.query,
            real_bundle.locks,
            pair[0],
            pair[1],
        )
    _require(
        _runtime_v2.state_dict_sha256(model) == model_sha,
        "V checkpoint changed after REAL ALL_PATCH",
    )
    with torch.no_grad():
        control_arm = _decoder_v2.decode_all_patch_pair_v2(
            model,
            control_bundle.query,
            control_bundle.locks,
            pair[0],
            pair[1],
        )
    _require(
        _runtime_v2.state_dict_sha256(model) == model_sha,
        "V checkpoint changed after C_COL ALL_PATCH",
    )
    _assert_arm_exact(real_arm, control_arm)

    real_direction = _payload_sha256(_direction_sequence_record(real_arm))
    control_direction = _payload_sha256(
        _direction_sequence_record(control_arm)
    )
    real_patch = _tensor_sha256(real_arm.scalar_contributions)
    control_patch = _tensor_sha256(control_arm.scalar_contributions)
    real_pair = _tensor_sha256(real_arm.logit)
    control_pair = _tensor_sha256(control_arm.logit)
    _require(
        real_direction == control_direction
        and real_patch == control_patch
        and real_pair == control_pair,
        "REAL/C_COL ALL_PATCH serialized bytes drift",
    )
    receipt = {
        "real_request_sha256": real_request_sha,
        "control_request_sha256": control_request_sha,
        "real_patch_tensor_sha256": real_patch,
        "control_patch_tensor_sha256": control_patch,
        "real_direction_sequence_sha256": real_direction,
        "control_direction_sequence_sha256": control_direction,
        "real_pair_score_sha256": real_pair,
        "control_pair_score_sha256": control_pair,
    }
    _require(set(receipt) == RECEIPT_FIELDS, "Phase-B receipt field drift")
    return MappingProxyType(receipt)


__all__ = [
    "SCHEMA_VERSION",
    "STATUS",
    "NUMERICAL_POLICY",
    "RECEIPT_FIELDS",
    "PhaseBAllPatchInvariantError",
    "phase_b_all_patch_invariance",
]
