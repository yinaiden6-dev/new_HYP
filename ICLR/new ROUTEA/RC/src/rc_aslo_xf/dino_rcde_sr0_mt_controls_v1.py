"""Inference-only destruction controls for formal SR0 multitile V.

The functions in this module transform exactly one registered variable and
return explicit receipts.  They never train a model, inspect a target/rival, or
select a transform by its response.  Candidate reorder is deliberately kept
separate from candidate-binding destruction.

Mask controls are emitted as arm/direction-specific overrides.  A formal
prejoin runner must re-run the regional decoder with those masks; applying an
override to a previously decoded all-patch tensor is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)
from .dino_rcde_cw1_multitile_vdecode_v1 import (
    BINDING_REAL,
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    OrderedRootArmTermV1,
    QueryTokenFieldV1,
    RootPairDecodeReceiptV1,
    TERM_H0,
    TERM_NO_COMPARABLE_ROOT,
    TERM_READY,
    ThreeArmFixedDenominatorEvidenceV1,
    _decode_pair_masks,
    _reduce_arm,
    _zero_term,
    token_tensor_sha256,
)
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    CandidatePLockV1,
    DeviceMaskRuntimeAdapter,
    SR0MTVContractError,
    bind_runtime_lineage,
    decode_direct_arm_term,
    decode_pair,
    hash_parts,
    tensor_sha256,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import LOCK_H0, ROOT_READY
from .geometry_hypothesis_v1 import connected_components_4
from .dino_rcde_colnomic_superregion_v1 import (
    FloatReductionClosureError,
    _region_is_legal,
    validate_conditioned_float_decomposition,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_controls_v1"


def _validate_control_decomposition(
    observed: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    absolute_mass: torch.Tensor,
    operation_count: int,
    name: str,
) -> None:
    try:
        validate_conditioned_float_decomposition(
            observed,
            reconstructed,
            absolute_mass=absolute_mass,
            operation_count=operation_count,
            name=name,
        )
    except FloatReductionClosureError as error:
        raise SR0MTVContractError(str(error)) from error

C_DINO_V = "C_DINO_V"
C_COL_P = "C_COL_P"
P_QUERY = "P_QUERY"
P_REFERENCE = "P_REFERENCE"
RANDOM_CONNECTED_MATCHED_SHAPE = "RANDOM_CONNECTED_MATCHED_SHAPE"
DISCONNECTED_MATCHED_AREA = "DISCONNECTED_MATCHED_AREA"
MAGNITUDE_ONLY = "MAGNITUDE_ONLY"
SINGLE_SEED = "SINGLE_SEED"
CANDIDATE_REORDER = "CANDIDATE_REORDER"
PAIR_SWAP = "PAIR_SWAP"

MANDATORY_CONTROLS = (
    C_DINO_V,
    C_COL_P,
    P_QUERY,
    P_REFERENCE,
    RANDOM_CONNECTED_MATCHED_SHAPE,
    DISCONNECTED_MATCHED_AREA,
    MAGNITUDE_ONLY,
    SINGLE_SEED,
    CANDIDATE_REORDER,
    PAIR_SWAP,
)

ARM_NAMES = (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
    "QUERY_REGION_FULL_REFERENCE_SAME_MODEL",
    "PAIRED_QUERY_REFERENCE_REGION",
)

MASK_TRANSPORT_METHOD = "NORMALIZED_CELL_RECTANGLE_POSITIVE_OVERLAP_V1"


@dataclass(frozen=True)
class TransformReceiptV1:
    control_name: str
    namespace: str
    changed_variable: str
    destination_to_source: tuple[int, ...]
    input_sha256: str
    output_sha256: str
    fixed_point_free: bool
    content_multiset_preserved: bool
    candidate_axis_preserved: bool
    p_lock_preserved: bool
    model_checkpoint_preserved: bool = True
    inference_only: bool = True
    changed_count: int | None = None
    structural_change_observed: bool | None = None

    def __post_init__(self) -> None:
        if self.control_name not in MANDATORY_CONTROLS:
            raise SR0MTVContractError("unknown SR0 control")
        if not self.namespace or not self.changed_variable:
            raise SR0MTVContractError("control namespace/variable is absent")
        order = tuple(int(item) for item in self.destination_to_source)
        if order and tuple(sorted(order)) != tuple(range(len(order))):
            raise SR0MTVContractError("control permutation is incomplete")
        if self.fixed_point_free and any(index == source for index, source in enumerate(order)):
            raise SR0MTVContractError("control claimed a fixed-point-free mapping with a fixed point")
        if len(self.input_sha256) != 64 or len(self.output_sha256) != 64:
            raise SR0MTVContractError("control hash receipt drift")
        if not self.inference_only or not self.model_checkpoint_preserved:
            raise SR0MTVContractError("control changed training/model state")
        if self.changed_count is not None and (
            isinstance(self.changed_count, bool) or self.changed_count < 0
        ):
            raise SR0MTVContractError("control changed-count receipt drift")
        object.__setattr__(self, "destination_to_source", order)


@dataclass(frozen=True)
class ArmMaskOverrideV1:
    control_name: str
    candidate_key: str
    direction: str
    arm_name: str
    query_mask: torch.Tensor
    reference_mask: torch.Tensor
    query_mask_sha256: str
    reference_mask_sha256: str
    status: str

    def __post_init__(self) -> None:
        if self.control_name not in {
            RANDOM_CONNECTED_MATCHED_SHAPE,
            DISCONNECTED_MATCHED_AREA,
            SINGLE_SEED,
        }:
            raise SR0MTVContractError("not a mask control")
        if self.direction not in FIXED_DIRECTIONS or self.arm_name not in ARM_NAMES:
            raise SR0MTVContractError("mask-control direction/arm drift")
        query = torch.as_tensor(self.query_mask, dtype=torch.bool).detach().cpu().flatten()
        reference = torch.as_tensor(self.reference_mask, dtype=torch.bool).detach().cpu().flatten()
        if tensor_sha256(query.to(torch.uint8)) != self.query_mask_sha256 or tensor_sha256(
            reference.to(torch.uint8)
        ) != self.reference_mask_sha256:
            raise SR0MTVContractError("mask-control receipt drift")
        if self.status not in {"READY", "H0_CONTROL_INELIGIBLE"}:
            raise SR0MTVContractError("mask-control status drift")
        if self.status == "READY" and (not bool(query.any()) or not bool(reference.any())):
            raise SR0MTVContractError("READY mask control is empty")
        object.__setattr__(self, "query_mask", query)
        object.__setattr__(self, "reference_mask", reference)


def _payload_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


MASK_TRANSPORT_CONTRACT_SHA256 = _payload_sha(
    {
        "method": MASK_TRANSPORT_METHOD,
        "selection_rule": "STRICT_POSITIVE_2D_CELL_RECTANGLE_OVERLAP",
        "destination_valid_intersection": True,
        "decoder_illegal_transport_policy": "SKIP_ROOT",
        "token_resize_or_interpolation_count": 0,
    }
)


@dataclass(frozen=True)
class CDinoCandidateBindingV1:
    """Destination candidate address carrying one donor's native DINO field.

    The original destination P lock is retained as a nested immutable object.
    Its reference masks deliberately remain on the destination grid; the
    C_DINO decoder transports only those boolean masks when the donor grid is
    different.  DINO tensors are never resized or interpolated.
    """

    candidate: CandidateReferenceFieldV1
    destination_p_lock: CandidatePLockV1
    donor_candidate_key: str
    donor_physical_gallery_row: int
    donor_corrected_identity_sha256: str
    destination_corrected_identity_sha256: str

    def __post_init__(self) -> None:
        destination = self.destination_p_lock.candidate
        if (
            self.candidate.candidate_key != destination.candidate_key
            or self.candidate.physical_gallery_row
            != destination.physical_gallery_row
        ):
            raise SR0MTVContractError("C_DINO destination candidate address drift")
        if (
            not self.donor_candidate_key
            or self.donor_candidate_key == destination.candidate_key
            or self.donor_physical_gallery_row
            == destination.physical_gallery_row
        ):
            raise SR0MTVContractError("C_DINO donor has a fixed point")
        for value in (
            self.donor_corrected_identity_sha256,
            self.destination_corrected_identity_sha256,
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SR0MTVContractError("C_DINO corrected-identity digest drift")
        if (
            self.donor_corrected_identity_sha256
            == self.destination_corrected_identity_sha256
        ):
            raise SR0MTVContractError("C_DINO donor is not corrected-identity-disjoint")

    @property
    def direction_locks(self) -> Mapping[str, object]:
        return self.destination_p_lock.direction_locks

    @property
    def p_lock_record_sha256_by_direction(self) -> Mapping[str, str]:
        return self.destination_p_lock.p_lock_record_sha256_by_direction

    def __reduce__(self):
        return (
            type(self),
            (
                self.candidate,
                self.destination_p_lock,
                self.donor_candidate_key,
                self.donor_physical_gallery_row,
                self.donor_corrected_identity_sha256,
                self.destination_corrected_identity_sha256,
            ),
        )


@dataclass(frozen=True)
class ReferenceMaskTransportReceiptV1:
    source_grid_shape: tuple[int, int]
    destination_grid_shape: tuple[int, int]
    source_mask_sha256: str
    destination_valid_mask_sha256: str
    transported_mask_sha256: str
    p_lock_record_sha256: str
    decoder_legal: bool
    method: str = MASK_TRANSPORT_METHOD
    token_resize_or_interpolation_count: int = 0

    def __post_init__(self) -> None:
        for shape in (self.source_grid_shape, self.destination_grid_shape):
            if len(shape) != 2 or any(int(item) <= 0 for item in shape):
                raise SR0MTVContractError("C_DINO mask-transport grid drift")
        for value in (
            self.source_mask_sha256,
            self.destination_valid_mask_sha256,
            self.transported_mask_sha256,
            self.p_lock_record_sha256,
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SR0MTVContractError("C_DINO mask-transport SHA drift")
        if (
            self.method != MASK_TRANSPORT_METHOD
            or self.token_resize_or_interpolation_count != 0
        ):
            raise SR0MTVContractError("C_DINO token resize/interpolation is forbidden")

    def logical_record(self) -> dict[str, object]:
        return {
            "source_grid_shape": list(self.source_grid_shape),
            "destination_grid_shape": list(self.destination_grid_shape),
            "source_mask_sha256": self.source_mask_sha256,
            "destination_valid_mask_sha256": self.destination_valid_mask_sha256,
            "transported_mask_sha256": self.transported_mask_sha256,
            "p_lock_record_sha256": self.p_lock_record_sha256,
            "decoder_legal": self.decoder_legal,
            "method": self.method,
            "token_resize_or_interpolation_count": 0,
        }


_REFERENCE_MASK_TRANSPORT_CACHE: dict[
    tuple[str, tuple[int, int], tuple[int, int], str], torch.Tensor
] = {}


def transport_reference_mask_positive_overlap(
    source_mask: torch.Tensor,
    source_grid_shape: tuple[int, int],
    destination_grid_shape: tuple[int, int],
    destination_valid_mask: torch.Tensor,
    *,
    p_lock_record_sha256: str,
) -> tuple[torch.Tensor, ReferenceMaskTransportReceiptV1]:
    """Transport a P reference mask by exact normalized-cell overlap.

    This transports membership only.  It never resizes, interpolates, pools,
    pads, or otherwise changes donor token content.
    """

    source_grid = tuple(int(item) for item in source_grid_shape)
    destination_grid = tuple(int(item) for item in destination_grid_shape)
    source = torch.as_tensor(source_mask, dtype=torch.bool).detach().cpu().flatten()
    valid = (
        torch.as_tensor(destination_valid_mask, dtype=torch.bool)
        .detach()
        .cpu()
        .flatten()
    )
    if (
        len(source_grid) != 2
        or len(destination_grid) != 2
        or any(item <= 0 for item in (*source_grid, *destination_grid))
        or source.numel() != math.prod(source_grid)
        or valid.numel() != math.prod(destination_grid)
    ):
        raise SR0MTVContractError("C_DINO mask-transport mask/grid drift")
    source_sha = tensor_sha256(source.to(torch.uint8))
    valid_sha = tensor_sha256(valid.to(torch.uint8))
    cache_key = (source_sha, source_grid, destination_grid, valid_sha)
    cached = _REFERENCE_MASK_TRANSPORT_CACHE.get(cache_key)
    if cached is None:
        source_2d = source.reshape(source_grid)
        source_h, source_w = source_grid
        destination_h, destination_w = destination_grid
        row_overlap = torch.zeros(
            (destination_h, source_h), dtype=torch.bool
        )
        column_overlap = torch.zeros(
            (destination_w, source_w), dtype=torch.bool
        )
        # Exact integer inequalities implement strict overlap between
        # [r/H,(r+1)/H) rectangles; no floating rounding is involved.
        for destination_row in range(destination_h):
            for source_row in range(source_h):
                row_overlap[destination_row, source_row] = max(
                    source_row * destination_h,
                    destination_row * source_h,
                ) < min(
                    (source_row + 1) * destination_h,
                    (destination_row + 1) * source_h,
                )
        for destination_column in range(destination_w):
            for source_column in range(source_w):
                column_overlap[destination_column, source_column] = max(
                    source_column * destination_w,
                    destination_column * source_w,
                ) < min(
                    (source_column + 1) * destination_w,
                    (destination_column + 1) * source_w,
                )
        transported_2d = (
            row_overlap.to(torch.int16)
            @ source_2d.to(torch.int16)
            @ column_overlap.to(torch.int16).transpose(0, 1)
        ) > 0
        cached = (transported_2d.flatten() & valid).contiguous()
        _REFERENCE_MASK_TRANSPORT_CACHE[cache_key] = cached
    transported = cached.clone()
    receipt = ReferenceMaskTransportReceiptV1(
        source_grid_shape=source_grid,
        destination_grid_shape=destination_grid,
        source_mask_sha256=source_sha,
        destination_valid_mask_sha256=valid_sha,
        transported_mask_sha256=tensor_sha256(transported.to(torch.uint8)),
        p_lock_record_sha256=p_lock_record_sha256,
        decoder_legal=_region_is_legal(transported, destination_grid),
    )
    return transported, receipt


def c_dino_mask_transport_preflight_sha256(
    locks: Mapping[str, CDinoCandidateBindingV1],
) -> str:
    """Seal every destination-P to native-donor-grid mask transport."""

    rows: list[dict[str, object]] = []
    for key in sorted(
        locks,
        key=lambda item: (
            locks[item].candidate.physical_gallery_row,
            item,
        ),
    ):
        lock = locks[key]
        for direction in FIXED_DIRECTIONS:
            sealed = lock.direction_locks[direction]
            for root in sorted(sealed.all_roots):
                scope = sealed.all_roots[root]
                _, receipt = transport_reference_mask_positive_overlap(
                    scope.reference_mask,
                    scope.reference_grid_shape,
                    lock.candidate.grid_shape,
                    lock.candidate.valid_patch_mask,
                    p_lock_record_sha256=sealed.p_lock_record_sha256,
                )
                rows.append(
                    {
                        "candidate_key": key,
                        "direction": direction,
                        "root_ordinal": root,
                        "receipt": receipt.logical_record(),
                    }
                )
    return _payload_sha(rows)


def deterministic_derangement(length: int, namespace: str, *parts: object) -> tuple[int, ...]:
    if length < 2:
        raise SR0MTVContractError("a derangement needs at least two items")
    shift = 1 + int(hash_parts(namespace, *parts), 16) % (length - 1)
    return tuple((index + shift) % length for index in range(length))


def _permute_selected_layers(
    layers: torch.Tensor,
    selected_mask: torch.Tensor,
    order: Sequence[int],
) -> torch.Tensor:
    source = torch.as_tensor(layers)
    mask = torch.as_tensor(selected_mask, dtype=torch.bool, device=source.device).flatten()
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    permutation = tuple(int(item) for item in order)
    if tuple(sorted(permutation)) != tuple(range(int(selected.numel()))):
        raise SR0MTVContractError("content permutation does not cover the selected footprint")
    output = source.clone()
    output[:, selected, :] = source[:, selected[list(permutation)], :]
    if not torch.equal(output[:, ~mask, :], source[:, ~mask, :]):
        raise SR0MTVContractError("content control changed values outside its footprint")
    return output


def permute_query_content(
    query: QueryTokenFieldV1,
    *,
    namespace: str,
    query_id: str,
) -> tuple[QueryTokenFieldV1, TransformReceiptV1]:
    valid = query.valid_patch_mask
    count = int(valid.sum())
    order = deterministic_derangement(count, namespace, query_id)
    layers = _permute_selected_layers(query.layers, valid, order)
    output_hash = token_tensor_sha256(layers)
    controlled = replace(
        query,
        layers=layers,
        cache_payload_sha256=hash_parts(namespace, query.cache_payload_sha256, output_hash),
        tokens_sha256=output_hash,
    )
    receipt = TransformReceiptV1(
        control_name=P_QUERY,
        namespace=namespace,
        changed_variable="DINO_QUERY_CONTENT_WITHIN_VALID_GRID",
        destination_to_source=order,
        input_sha256=query.tokens_sha256,
        output_sha256=output_hash,
        fixed_point_free=True,
        content_multiset_preserved=True,
        candidate_axis_preserved=True,
        p_lock_preserved=True,
    )
    return controlled, receipt


def permute_reference_content(
    candidate: CandidateReferenceFieldV1,
    selected_mask: torch.Tensor,
    *,
    namespace: str,
    query_id: str,
    direction: str,
    arm_name: str,
) -> tuple[CandidateReferenceFieldV1, TransformReceiptV1]:
    mask = torch.as_tensor(selected_mask, dtype=torch.bool).flatten()
    if mask.shape != candidate.valid_patch_mask.shape or bool(
        (mask & ~candidate.valid_patch_mask).any()
    ):
        raise SR0MTVContractError("reference permutation footprint is outside valid content")
    count = int(mask.sum())
    order = deterministic_derangement(
        count, namespace, query_id, candidate.candidate_key, direction, arm_name
    )
    layers = _permute_selected_layers(candidate.layers, mask, order)
    output_hash = token_tensor_sha256(layers)
    controlled = replace(
        candidate,
        layers=layers,
        cache_payload_sha256=hash_parts(
            namespace, candidate.cache_payload_sha256, direction, arm_name, output_hash
        ),
        tokens_sha256=output_hash,
    )
    return controlled, TransformReceiptV1(
        control_name=P_REFERENCE,
        namespace=namespace,
        changed_variable="DINO_REFERENCE_CONTENT_WITHIN_REGISTERED_ARM_FOOTPRINT",
        destination_to_source=order,
        input_sha256=candidate.tokens_sha256,
        output_sha256=output_hash,
        fixed_point_free=True,
        content_multiset_preserved=True,
        candidate_axis_preserved=True,
        p_lock_preserved=True,
    )


def c_dino_v_derangement(
    locks: Mapping[str, CandidatePLockV1],
    *,
    namespace: str,
    query_id: str,
    corrected_identity_by_physical_row: Mapping[int, str],
) -> tuple[
    Mapping[str, CDinoCandidateBindingV1],
    tuple[TransformReceiptV1, ...],
    tuple[str, ...],
]:
    """Derange native DINO fields over the complete natural C128 axis.

    The matching is independent of query targets and has no geometry strata:
    every destination receives one donor with a different corrected identity.
    The donor's complete native token/grid/valid/geometry payload is rebound to
    the destination candidate key/physical-row address.  The destination's P
    lock remains the exact original object and is not rewritten to the donor
    grid.
    """

    axis = tuple(
        sorted(
            locks,
            key=lambda key: (
                locks[key].candidate.physical_gallery_row,
                locks[key].candidate.source_image_sha256,
                key,
            ),
        )
    )
    if len(axis) < 2:
        raise SR0MTVContractError(
            "C_DINO_V full-axis has no identity-disjoint perfect matching"
        )
    identities: dict[str, str] = {}
    for key in axis:
        row = locks[key].candidate.physical_gallery_row
        value = corrected_identity_by_physical_row.get(row)
        if not isinstance(value, str) or not value:
            raise SR0MTVContractError("C_DINO_V corrected identity is absent")
        identities[key] = value
    donor_for: dict[str, str] = {}
    destination_for_donor: dict[str, str] = {}

    def assign(destination: str, seen: set[str]) -> bool:
        donors = sorted(
            (
                donor
                for donor in axis
                if donor != destination
                and identities[donor] != identities[destination]
            ),
            key=lambda donor: hash_parts(
                namespace, query_id, "FULL_C128_DONOR", destination, donor
            ),
        )
        for donor in donors:
            if donor in seen:
                continue
            seen.add(donor)
            prior = destination_for_donor.get(donor)
            if prior is None or assign(prior, seen):
                destination_for_donor[donor] = destination
                donor_for[destination] = donor
                return True
        return False

    destination_order = sorted(
        axis,
        key=lambda key: hash_parts(namespace, query_id, "FULL_C128_DEST", key),
    )
    if any(not assign(destination, set()) for destination in destination_order):
        raise SR0MTVContractError(
            "C_DINO_V full-axis has no identity-disjoint perfect matching"
        )
    order = tuple(axis.index(donor_for[key]) for key in axis)
    if sorted(order) != list(range(len(axis))) or any(
        source == index for index, source in enumerate(order)
    ):
        raise SR0MTVContractError("C_DINO_V donor assignment is not a derangement")

    output: dict[str, CDinoCandidateBindingV1] = {}
    for destination_key in axis:
        donor_key = donor_for[destination_key]
        destination_lock = locks[destination_key]
        destination = destination_lock.candidate
        donor = locks[donor_key].candidate
        controlled_candidate = CandidateReferenceFieldV1(
            candidate_key=destination.candidate_key,
            physical_gallery_row=destination.physical_gallery_row,
            layers=donor.layers.clone(),
            grid_shape=donor.grid_shape,
            valid_patch_mask=donor.valid_patch_mask.clone(),
            source_image_sha256=donor.source_image_sha256,
            source_key=donor.source_key,
            cache_payload_sha256=donor.cache_payload_sha256,
            geometry_record_sha256=donor.geometry_record_sha256,
            tokens_sha256=donor.tokens_sha256,
        )
        output[destination_key] = CDinoCandidateBindingV1(
            candidate=controlled_candidate,
            destination_p_lock=destination_lock,
            donor_candidate_key=donor_key,
            donor_physical_gallery_row=donor.physical_gallery_row,
            donor_corrected_identity_sha256=hashlib.sha256(
                identities[donor_key].encode("utf-8")
            ).hexdigest(),
            destination_corrected_identity_sha256=hashlib.sha256(
                identities[destination_key].encode("utf-8")
            ).hexdigest(),
        )
    if set(output) != set(locks):
        raise SR0MTVContractError("C_DINO_V output axis is incomplete")
    input_multiset = sorted(
        (
            locks[key].candidate.source_image_sha256,
            locks[key].candidate.cache_payload_sha256,
            locks[key].candidate.geometry_record_sha256,
            locks[key].candidate.tokens_sha256,
            tuple(locks[key].candidate.grid_shape),
            tensor_sha256(
                locks[key].candidate.valid_patch_mask.to(torch.uint8)
            ),
        )
        for key in axis
    )
    output_multiset = sorted(
        (
            output[key].candidate.source_image_sha256,
            output[key].candidate.cache_payload_sha256,
            output[key].candidate.geometry_record_sha256,
            output[key].candidate.tokens_sha256,
            tuple(output[key].candidate.grid_shape),
            tensor_sha256(
                output[key].candidate.valid_patch_mask.to(torch.uint8)
            ),
        )
        for key in axis
    )
    if output_multiset != input_multiset:
        raise SR0MTVContractError("C_DINO_V full native-content multiset drift")
    p_input = _payload_sha(
        [
            [
                locks[key].p_lock_record_sha256_by_direction[direction]
                for direction in FIXED_DIRECTIONS
            ]
            for key in axis
        ]
    )
    p_output = _payload_sha(
        [
            [
                output[key].p_lock_record_sha256_by_direction[direction]
                for direction in FIXED_DIRECTIONS
            ]
            for key in axis
        ]
    )
    if p_input != p_output:
        raise SR0MTVContractError("C_DINO_V changed the destination P-lock population")
    receipt = TransformReceiptV1(
        control_name=C_DINO_V,
        namespace=namespace,
        changed_variable="DINO_REFERENCE_NATIVE_CONTENT_BINDING_ACROSS_FULL_C128",
        destination_to_source=order,
        input_sha256=_payload_sha(input_multiset),
        output_sha256=_payload_sha(output_multiset),
        fixed_point_free=True,
        content_multiset_preserved=True,
        candidate_axis_preserved=True,
        p_lock_preserved=True,
    )
    return MappingProxyType(output), (receipt,), ()


def validate_c_col_p_rebuild(
    real: Mapping[str, CandidatePLockV1],
    rebuilt: Mapping[str, CandidatePLockV1],
) -> TransformReceiptV1:
    """Validate a separately materialized pre-P candidate-binding control."""

    if set(real) != set(rebuilt):
        raise SR0MTVContractError("C_COL_P changed the natural candidate axis")
    real_receipts: list[str] = []
    rebuilt_receipts: list[str] = []
    structural_change = False
    for key in sorted(real):
        first, second = real[key], rebuilt[key]
        if (
            first.candidate.physical_gallery_row != second.candidate.physical_gallery_row
            or first.candidate.tokens_sha256 != second.candidate.tokens_sha256
            or first.candidate.cache_payload_sha256 != second.candidate.cache_payload_sha256
        ):
            raise SR0MTVContractError("C_COL_P changed DINO candidate content/axis")
        real_receipts.extend(first.p_lock_record_sha256_by_direction.values())
        rebuilt_receipts.extend(second.p_lock_record_sha256_by_direction.values())
        structural_change = structural_change or any(
            _direction_structure_sha(first, direction)
            != _direction_structure_sha(second, direction)
            for direction in FIXED_DIRECTIONS
        )
    changed_count = sum(
        first != second
        for first, second in zip(real_receipts, rebuilt_receipts, strict=True)
    )
    return TransformReceiptV1(
        control_name=C_COL_P,
        namespace="RCDE_SR0_MT_C_COL_P_V1",
        changed_variable="COLNOMIC_CANDIDATE_BINDING_BEFORE_P_AND_REBUILT_LOCK",
        destination_to_source=(),
        input_sha256=_payload_sha(real_receipts),
        output_sha256=_payload_sha(rebuilt_receipts),
        fixed_point_free=False,
        content_multiset_preserved=True,
        candidate_axis_preserved=True,
        # C_COL is defined by a fresh pre-P binding intervention and P rerun.
        # Even an accidental byte-identical result is a scientific null, not
        # evidence that the original P lock was the executed object.
        p_lock_preserved=False,
        changed_count=changed_count,
        structural_change_observed=structural_change,
    )


def _indices(mask: torch.Tensor, grid: tuple[int, int]) -> list[tuple[int, int]]:
    width = int(grid[1])
    return [(int(index) // width, int(index) % width) for index in torch.nonzero(mask, as_tuple=False).flatten()]


def _mask_from_cells(cells: Sequence[tuple[int, int]], grid: tuple[int, int]) -> torch.Tensor:
    output = torch.zeros(math.prod(grid), dtype=torch.bool)
    width = grid[1]
    for row, column in cells:
        output[row * width + column] = True
    return output


def _bbox(mask: torch.Tensor, grid: tuple[int, int]) -> tuple[int, int, int, int]:
    cells = _indices(mask, grid)
    if not cells:
        raise SR0MTVContractError("mask has no bounding box")
    rows, columns = zip(*cells)
    return min(rows), max(rows), min(columns), max(columns)


def _border_distance(mask: torch.Tensor, grid: tuple[int, int]) -> int:
    h, w = grid
    return min(min(row, column, h - 1 - row, w - 1 - column) for row, column in _indices(mask, grid))


def random_connected_matched_shape(
    source_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    grid: tuple[int, int],
    *,
    namespace: str,
    key: str,
) -> torch.Tensor | None:
    """Translate one connected shape result-blindly, preserving shape/stratum.

    There is no response-dependent retry.  All legal translations are computed
    from masks alone, ordered by a hash, and the first non-identity member is
    selected.  If none exists, the registered control is H0.
    """

    source = torch.as_tensor(source_mask, dtype=torch.bool).flatten()
    valid = torch.as_tensor(valid_mask, dtype=torch.bool).flatten()
    if source.shape != valid.shape or source.shape != (math.prod(grid),) or bool(
        (source & ~valid).any()
    ):
        raise SR0MTVContractError("connected-control source/valid mask drift")
    _, diagnostic = connected_components_4(source, grid)
    if diagnostic.component_count != 1:
        raise SR0MTVContractError("connected control requires one source 4CC")
    cells = _indices(source, grid)
    r0, r1, c0, c1 = _bbox(source, grid)
    source_border = _border_distance(source, grid)
    legal: list[torch.Tensor] = []
    for dr in range(-r0, grid[0] - r1):
        for dc in range(-c0, grid[1] - c1):
            if dr == 0 and dc == 0:
                continue
            candidate = _mask_from_cells([(r + dr, c + dc) for r, c in cells], grid)
            if bool((candidate & ~valid).any()) or _border_distance(candidate, grid) != source_border:
                continue
            legal.append(candidate)
    if not legal:
        return None
    legal.sort(key=lambda value: hash_parts(namespace, key, tensor_sha256(value.to(torch.uint8))))
    return legal[0]


def disconnected_matched_area(
    source_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    grid: tuple[int, int],
    *,
    namespace: str,
    key: str,
) -> torch.Tensor | None:
    """Select a result-blind same-area valid mask with >=2 non-touching 4CCs."""

    source = torch.as_tensor(source_mask, dtype=torch.bool).flatten()
    valid = torch.as_tensor(valid_mask, dtype=torch.bool).flatten()
    count = int(source.sum())
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    if count < 2 or len(valid_indices) < count:
        return None
    ordered = sorted(valid_indices, key=lambda index: hash_parts(namespace, key, index))
    # Deterministic sliding windows are a mask-only enumeration, never a model
    # response retry.  This remains tractable for natural DINO grids.
    for offset in range(len(ordered)):
        chosen = [ordered[(offset + item) % len(ordered)] for item in range(count)]
        candidate = torch.zeros_like(valid)
        candidate[chosen] = True
        _, diagnostic = connected_components_4(candidate, grid)
        if diagnostic.component_count >= 2:
            return candidate
    return None


def single_seed_mask(source_mask: torch.Tensor, *, namespace: str, key: str) -> torch.Tensor:
    source = torch.as_tensor(source_mask, dtype=torch.bool).flatten()
    indices = torch.nonzero(source, as_tuple=False).flatten().tolist()
    if not indices:
        return torch.zeros_like(source)
    chosen = min(indices, key=lambda index: hash_parts(namespace, key, index))
    output = torch.zeros_like(source)
    output[chosen] = True
    return output


def mask_override(
    *,
    control_name: str,
    candidate_key: str,
    direction: str,
    arm_name: str,
    source_query_mask: torch.Tensor,
    source_reference_mask: torch.Tensor,
    query_valid_mask: torch.Tensor,
    reference_valid_mask: torch.Tensor,
    query_grid: tuple[int, int],
    reference_grid: tuple[int, int],
    namespace: str,
) -> ArmMaskOverrideV1:
    if control_name == RANDOM_CONNECTED_MATCHED_SHAPE:
        query = random_connected_matched_shape(
            source_query_mask, query_valid_mask, query_grid,
            namespace=namespace, key=f"{candidate_key}:{direction}:{arm_name}:Q",
        )
        reference = random_connected_matched_shape(
            source_reference_mask, reference_valid_mask, reference_grid,
            namespace=namespace, key=f"{candidate_key}:{direction}:{arm_name}:R",
        )
    elif control_name == DISCONNECTED_MATCHED_AREA:
        query = disconnected_matched_area(
            source_query_mask, query_valid_mask, query_grid,
            namespace=namespace, key=f"{candidate_key}:{direction}:{arm_name}:Q",
        )
        reference = disconnected_matched_area(
            source_reference_mask, reference_valid_mask, reference_grid,
            namespace=namespace, key=f"{candidate_key}:{direction}:{arm_name}:R",
        )
    elif control_name == SINGLE_SEED:
        query = single_seed_mask(
            source_query_mask, namespace=namespace,
            key=f"{candidate_key}:{direction}:{arm_name}:Q",
        )
        reference = single_seed_mask(
            source_reference_mask, namespace=namespace,
            key=f"{candidate_key}:{direction}:{arm_name}:R",
        )
    else:
        raise SR0MTVContractError("requested control does not produce mask overrides")
    status = "READY"
    if query is None or reference is None or not bool(query.any()) or not bool(reference.any()):
        query = torch.zeros_like(torch.as_tensor(query_valid_mask, dtype=torch.bool).flatten())
        reference = torch.zeros_like(torch.as_tensor(reference_valid_mask, dtype=torch.bool).flatten())
        status = "H0_CONTROL_INELIGIBLE"
    return ArmMaskOverrideV1(
        control_name=control_name,
        candidate_key=candidate_key,
        direction=direction,
        arm_name=arm_name,
        query_mask=query,
        reference_mask=reference,
        query_mask_sha256=tensor_sha256(query.to(torch.uint8)),
        reference_mask_sha256=tensor_sha256(reference.to(torch.uint8)),
        status=status,
    )


def magnitude_only_contributions(
    signed_contributions: torch.Tensor,
    *,
    unordered_pair_key: str,
    canonical_orientation: int,
) -> torch.Tensor:
    """Remove learned sign while retaining the exact magnitude vector/scale.

    A result-blind, pair-keyed Rademacher sequence supplies signs.  The
    canonical orientation is +1 for the stored unordered pair and -1 only for
    its derived PAIR_SWAP, preserving exact antisymmetry without a second
    forward.
    """

    values = torch.as_tensor(signed_contributions)
    if values.ndim != 1 or canonical_orientation not in (-1, 1):
        raise SR0MTVContractError("magnitude-only input/orientation drift")
    signs = torch.tensor(
        [
            1.0
            if int(hash_parts("RCDE_SR0_MT_MAGNITUDE_ONLY_V1", unordered_pair_key, i), 16) % 2
            else -1.0
            for i in range(values.numel())
        ],
        dtype=values.dtype,
        device=values.device,
    )
    return canonical_orientation * signs * values.abs()


def candidate_reorder(
    axis: Sequence[str], *, namespace: str, query_id: str
) -> tuple[tuple[str, ...], TransformReceiptV1]:
    axis = tuple(axis)
    order = deterministic_derangement(len(axis), namespace, query_id)
    reordered = tuple(axis[index] for index in order)
    return reordered, TransformReceiptV1(
        control_name=CANDIDATE_REORDER,
        namespace=namespace,
        changed_variable="CONTAINER_PRESENTATION_ORDER_ONLY",
        destination_to_source=order,
        input_sha256=_payload_sha(axis),
        output_sha256=_payload_sha(reordered),
        fixed_point_free=True,
        content_multiset_preserved=True,
        candidate_axis_preserved=True,
        p_lock_preserved=True,
    )


def pair_swap_record(
    left_key: str, right_key: str, logit: float, contribution_sha256: str
) -> Mapping[str, object]:
    if left_key == right_key or len(contribution_sha256) != 64:
        raise SR0MTVContractError("pair-swap source drift")
    return MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "control_name": PAIR_SWAP,
            "source_pair": [left_key, right_key],
            "derived_pair": [right_key, left_key],
            "source_logit": float(logit),
            "derived_logit": -float(logit),
            "source_contribution_sha256": contribution_sha256,
            "second_model_forward_count": 0,
            "exact_antisymmetry": True,
        }
    )


def _direction_structure_sha(lock: CandidatePLockV1, direction: str) -> str:
    sealed = lock.direction_locks[direction]
    return _payload_sha(
        {
            "status": sealed.status,
            "selected_bank_ordinal": sealed.selected_bank_ordinal,
            "selected_row_sha256": sealed.selected_row_sha256,
            "query_union_sha256": tensor_sha256(
                sealed.query_union_mask.to(torch.uint8)
            ),
            "roots": [
                {
                    "root": root,
                    "action": scope.action_key_sha256,
                    "query": scope.query_mask_p_sha256,
                    "reference": scope.reference_mask_p_sha256,
                    "status": scope.binding_status,
                }
                for root, scope in sealed.all_roots.items()
            ],
        }
    )


def _reference_union(lock: CandidatePLockV1, direction: str) -> torch.Tensor:
    sealed = lock.direction_locks[direction]
    output = torch.zeros_like(lock.candidate.valid_patch_mask)
    if sealed.status != "P_LOCK_READY":
        return output
    for root in sealed.selected_roots:
        output |= root.reference_mask
    return output


def _base_masks(
    query: QueryTokenFieldV1,
    lock: CandidatePLockV1,
    direction: str,
    arm_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if arm_name == ARM_ALL_PATCH:
        return query.valid_patch_mask, lock.candidate.valid_patch_mask
    sealed = lock.direction_locks[direction]
    if sealed.status != "P_LOCK_READY":
        return (
            torch.zeros_like(query.valid_patch_mask),
            torch.zeros_like(lock.candidate.valid_patch_mask),
        )
    if arm_name == ARM_QUERY_FULL_REFERENCE:
        return sealed.query_union_mask, lock.candidate.valid_patch_mask
    if arm_name == ARM_QUERY_LOCAL_COMPONENTS:
        return sealed.query_union_mask, _reference_union(lock, direction)
    raise SR0MTVContractError("unknown runtime arm for control scoring")


def _raw_mask_term(
    adapter: DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner: CandidateReferenceFieldV1,
    opponent: CandidateReferenceFieldV1,
    query_mask: torch.Tensor,
    owner_reference_mask: torch.Tensor,
    opponent_reference_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmask = torch.as_tensor(query_mask, dtype=torch.bool).flatten()
    owner_mask = torch.as_tensor(owner_reference_mask, dtype=torch.bool).flatten()
    opponent_mask = torch.as_tensor(opponent_reference_mask, dtype=torch.bool).flatten()
    zero = query.layers.new_zeros(query.valid_patch_mask.numel())
    if not bool(qmask.any()) or not bool(owner_mask.any()) or not bool(opponent_mask.any()):
        return zero, zero.new_zeros(())
    owner_evidence = adapter.decode_candidate(
        query.layers,
        owner.layers,
        qmask.reshape(query.grid_shape),
        owner_mask.reshape(owner.grid_shape),
        query.grid_shape,
        owner.grid_shape,
    )
    opponent_evidence = adapter.decode_candidate(
        query.layers,
        opponent.layers,
        qmask.reshape(query.grid_shape),
        opponent_mask.reshape(opponent.grid_shape),
        query.grid_shape,
        opponent.grid_shape,
    )
    pair = adapter.compare_relational(
        owner_evidence.relational, opponent_evidence.relational, qmask
    )
    contributions = pair.contributions * qmask.to(
        device=pair.contributions.device, dtype=pair.contributions.dtype
    )
    scalar = contributions / qmask.sum().to(
        device=contributions.device, dtype=contributions.dtype
    )
    _validate_control_decomposition(
        pair.logit,
        scalar.sum(),
        absolute_mass=scalar.abs().to(torch.float64).sum(),
        operation_count=scalar.numel() + 2,
        name="control raw-mask scalar reconstruction",
    )
    return scalar, pair.logit


def _mask_control_score(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
    arm_name: str,
    control_name: str,
    *,
    namespace: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    model_sha, comparator_sha, _ = bind_runtime_lineage(model)
    adapter = DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    signed_parts: list[torch.Tensor] = []
    signed_logits: list[torch.Tensor] = []
    eligible = True
    for direction in FIXED_DIRECTIONS:
        for owner_key, opponent_key, sign in (
            (left_key, right_key, 1.0),
            (right_key, left_key, -1.0),
        ):
            owner = locks[owner_key]
            opponent = locks[opponent_key]
            if arm_name == ARM_QUERY_LOCAL_COMPONENTS:
                parts, logit, term_eligible = _local_mask_control_term(
                    adapter,
                    query,
                    owner,
                    opponent,
                    direction=direction,
                    control_name=control_name,
                    namespace=namespace,
                )
                eligible = eligible and term_eligible
                signed_parts.append(parts * sign)
                signed_logits.append(logit * sign)
                continue
            owner_q, owner_r = _base_masks(query, owner, direction, arm_name)
            _, opponent_r = _base_masks(query, opponent, direction, arm_name)
            owner_override = mask_override(
                control_name=control_name,
                candidate_key=owner_key,
                direction=direction,
                arm_name=arm_name,
                source_query_mask=owner_q,
                source_reference_mask=owner_r,
                query_valid_mask=query.valid_patch_mask,
                reference_valid_mask=owner.candidate.valid_patch_mask,
                query_grid=query.grid_shape,
                reference_grid=owner.candidate.grid_shape,
                namespace=namespace,
            )
            opponent_override = mask_override(
                control_name=control_name,
                candidate_key=opponent_key,
                direction=direction,
                arm_name=arm_name,
                source_query_mask=_base_masks(query, opponent, direction, arm_name)[0],
                source_reference_mask=opponent_r,
                query_valid_mask=query.valid_patch_mask,
                reference_valid_mask=opponent.candidate.valid_patch_mask,
                query_grid=query.grid_shape,
                reference_grid=opponent.candidate.grid_shape,
                namespace=namespace,
            )
            eligible = eligible and owner_override.status == "READY" and opponent_override.status == "READY"
            parts, logit = _raw_mask_term(
                adapter,
                query,
                owner.candidate,
                opponent.candidate,
                owner_override.query_mask,
                owner_override.reference_mask,
                opponent_override.reference_mask,
            )
            signed_parts.append(parts * sign)
            signed_logits.append(logit * sign)
    contributions = sum(signed_parts, start=query.layers.new_zeros(query.valid_patch_mask.numel())) / 4.0
    margin = sum(signed_logits, start=query.layers.new_zeros(())) / 4.0
    absolute_mass = sum(
        (item.abs().to(torch.float64).sum() for item in signed_parts),
        start=torch.zeros((), dtype=torch.float64, device=contributions.device),
    ) / 4.0
    _validate_control_decomposition(
        margin,
        contributions.sum(),
        absolute_mass=absolute_mass,
        operation_count=contributions.numel() + 8,
        name="mask control fixed-four-term reconstruction",
    )
    return contributions, margin, eligible


def _one_control_mask(
    control_name: str,
    source: torch.Tensor,
    valid: torch.Tensor,
    grid: tuple[int, int],
    *,
    namespace: str,
    key: str,
) -> torch.Tensor | None:
    if control_name == RANDOM_CONNECTED_MATCHED_SHAPE:
        return random_connected_matched_shape(
            source, valid, grid, namespace=namespace, key=key
        )
    if control_name == DISCONNECTED_MATCHED_AREA:
        return disconnected_matched_area(
            source, valid, grid, namespace=namespace, key=key
        )
    if control_name == SINGLE_SEED:
        value = single_seed_mask(source, namespace=namespace, key=key)
        return value if bool(value.any()) else None
    raise SR0MTVContractError("unknown root-local mask control")


def _local_mask_control_term(
    adapter: DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner: CandidatePLockV1,
    opponent: CandidatePLockV1,
    *,
    direction: str,
    control_name: str,
    namespace: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Execute matched mask controls root-by-root for the paired arm.

    A union of individually connected reference components need not itself be
    connected.  Treating that union as the source of RANDOM_CONNECTED is a
    category error.  The paired arm therefore transforms each sealed root
    scope, decodes it, and applies the same frozen overlap correction as the
    real direct-lock path.
    """

    sealed_owner = owner.direction_locks[direction]
    sealed_opponent = opponent.direction_locks[direction]
    zero = query.layers.new_zeros(query.valid_patch_mask.numel())
    if sealed_owner.status != "P_LOCK_READY":
        return zero, zero.new_zeros(()), True
    total = zero.clone()
    coverage = torch.zeros_like(query.valid_patch_mask, dtype=torch.int64)
    comparable = 0
    for root in sealed_owner.ordered_root_ordinals:
        owner_root = sealed_owner.all_roots[root]
        opponent_root = sealed_opponent.all_roots[root]
        query_mask = _one_control_mask(
            control_name,
            owner_root.query_mask,
            query.valid_patch_mask,
            query.grid_shape,
            namespace=namespace,
            key=f"{owner.candidate.candidate_key}:{direction}:{root}:Q",
        )
        owner_reference = _one_control_mask(
            control_name,
            owner_root.reference_mask,
            owner.candidate.valid_patch_mask,
            owner.candidate.grid_shape,
            namespace=namespace,
            key=f"{owner.candidate.candidate_key}:{direction}:{root}:R",
        )
        if query_mask is None or owner_reference is None:
            return zero, zero.new_zeros(()), False
        coverage += query_mask.to(torch.int64)
        if opponent_root.binding_status != "ROOT_LOCAL_COMPONENT_READY":
            continue
        opponent_reference = _one_control_mask(
            control_name,
            opponent_root.reference_mask,
            opponent.candidate.valid_patch_mask,
            opponent.candidate.grid_shape,
            namespace=namespace,
            key=f"{opponent.candidate.candidate_key}:{direction}:{root}:R",
        )
        if opponent_reference is None:
            return zero, zero.new_zeros(()), False
        scalar, _ = _raw_mask_term(
            adapter,
            query,
            owner.candidate,
            opponent.candidate,
            query_mask,
            owner_reference,
            opponent_reference,
        )
        total = total + scalar * int(query_mask.sum())
        comparable += 1
    if comparable == 0:
        return zero, zero.new_zeros(()), True
    reciprocal = torch.zeros_like(total)
    covered = coverage > 0
    reciprocal[covered.to(reciprocal.device)] = 1.0 / coverage[covered].to(
        device=reciprocal.device, dtype=reciprocal.dtype
    )
    patch = total * reciprocal
    union = covered.to(device=patch.device)
    scalar = patch / union.sum().to(dtype=patch.dtype)
    return scalar, scalar.sum(), True


def _p_reference_score(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
    arm_name: str,
    *,
    namespace: str,
    query_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_sha, comparator_sha, _ = bind_runtime_lineage(model)
    adapter = DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    by_direction: dict[tuple[str, str], CandidatePLockV1] = {}
    for key in (left_key, right_key):
        for direction in FIXED_DIRECTIONS:
            _, footprint = _base_masks(query, locks[key], direction, arm_name)
            if int(footprint.sum()) < 2:
                candidate = locks[key].candidate
            else:
                candidate = permute_reference_content(
                    locks[key].candidate,
                    footprint,
                    namespace=namespace,
                    query_id=query_id,
                    direction=direction,
                    arm_name=arm_name,
                )[0]
            by_direction[(key, direction)] = replace(
                locks[key], candidate=candidate
            )
    forward = []
    reverse = []
    for direction in FIXED_DIRECTIONS:
        forward.append(
            decode_direct_arm_term(
                adapter,
                query,
                by_direction[(left_key, direction)],
                by_direction[(right_key, direction)],
                direction=direction,
                arm_name=arm_name,
            )
        )
        reverse.append(
            decode_direct_arm_term(
                adapter,
                query,
                by_direction[(right_key, direction)],
                by_direction[(left_key, direction)],
                direction=direction,
                arm_name=arm_name,
            )
        )
    result = _reduce_arm(
        arm_name,
        left_key,
        right_key,
        tuple(forward),  # type: ignore[arg-type]
        tuple(reverse),  # type: ignore[arg-type]
    )
    return result.scalar_contributions, result.logit


def _c_dino_paired_term(
    adapter: DeviceMaskRuntimeAdapter,
    query: QueryTokenFieldV1,
    owner: CDinoCandidateBindingV1,
    opponent: CDinoCandidateBindingV1,
    *,
    direction: str,
) -> tuple[OrderedRootArmTermV1, tuple[ReferenceMaskTransportReceiptV1, ...]]:
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
    transport_receipts: list[ReferenceMaskTransportReceiptV1] = []
    decoded: list[int] = []
    for root in owner_lock.ordered_root_ordinals:
        owner_root = owner_lock.all_roots[root]
        opponent_root = opponent_lock.all_roots[root]
        if opponent_root.binding_status != ROOT_READY:
            continue
        if not torch.equal(owner_root.query_mask, opponent_root.query_mask):
            raise SR0MTVContractError(
                "C_DINO destination P locks disagree on a fixed query root"
            )
        owner_reference, owner_transport = (
            transport_reference_mask_positive_overlap(
                owner_root.reference_mask,
                owner_root.reference_grid_shape,
                owner.candidate.grid_shape,
                owner.candidate.valid_patch_mask,
                p_lock_record_sha256=owner_lock.p_lock_record_sha256,
            )
        )
        opponent_reference, opponent_transport = (
            transport_reference_mask_positive_overlap(
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
            streaming_chunk_size=None,
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
                opponent_reference_mask_sha256=receipt.opponent_reference_mask_sha256,
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


def decode_c_dino_pair(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    locks: Mapping[str, CDinoCandidateBindingV1],
    left_key: str,
    right_key: str,
) -> tuple[ThreeArmFixedDenominatorEvidenceV1, Mapping[str, str]]:
    """Decode one C_DINO pair without rewriting its destination P locks."""

    if left_key == right_key or left_key not in locks or right_key not in locks:
        raise SR0MTVContractError("C_DINO pair axis drift")
    left, right = locks[left_key], locks[right_key]
    for key, lock in ((left_key, left), (right_key, right)):
        if (
            lock.candidate.candidate_key != key
            or lock.candidate.layers.device != query.layers.device
            or lock.candidate.layers.dtype != query.layers.dtype
            or token_tensor_sha256(lock.candidate.layers)
            != lock.candidate.tokens_sha256
        ):
            raise SR0MTVContractError("C_DINO candidate content/device drift")
        for direction in FIXED_DIRECTIONS:
            sealed = lock.direction_locks[direction]
            if (
                sealed.query_source_image_sha256 != query.source_image_sha256
                or sealed.query_grid_shape != query.grid_shape
                or sealed.query_geometry_sha256 != query.geometry_record_sha256
            ):
                raise SR0MTVContractError("C_DINO query/P-lock provenance drift")
    model_sha, comparator_sha, reducer_sha = bind_runtime_lineage(model)
    adapter = DeviceMaskRuntimeAdapter(model, model_sha, comparator_sha)
    arms = []
    transport_hashes: dict[str, str] = {}
    for arm_name in (
        ARM_ALL_PATCH,
        ARM_QUERY_FULL_REFERENCE,
        ARM_QUERY_LOCAL_COMPONENTS,
    ):
        forward: list[OrderedRootArmTermV1] = []
        reverse: list[OrderedRootArmTermV1] = []
        receipts: list[ReferenceMaskTransportReceiptV1] = []
        for direction in FIXED_DIRECTIONS:
            if arm_name == ARM_QUERY_LOCAL_COMPONENTS:
                term, term_receipts = _c_dino_paired_term(
                    adapter, query, left, right, direction=direction
                )
                forward.append(term)
                receipts.extend(term_receipts)
                term, term_receipts = _c_dino_paired_term(
                    adapter, query, right, left, direction=direction
                )
                reverse.append(term)
                receipts.extend(term_receipts)
            else:
                forward.append(
                    decode_direct_arm_term(
                        adapter,
                        query,
                        left,  # type: ignore[arg-type]
                        right,  # type: ignore[arg-type]
                        direction=direction,
                        arm_name=arm_name,
                    )
                )
                reverse.append(
                    decode_direct_arm_term(
                        adapter,
                        query,
                        right,  # type: ignore[arg-type]
                        left,  # type: ignore[arg-type]
                        direction=direction,
                        arm_name=arm_name,
                    )
                )
        arms.append(
            _reduce_arm(
                arm_name,
                left_key,
                right_key,
                tuple(forward),  # type: ignore[arg-type]
                tuple(reverse),  # type: ignore[arg-type]
            )
        )
        transport_hashes[arm_name] = _payload_sha(
            {
                "contract_sha256": MASK_TRANSPORT_CONTRACT_SHA256,
                "left_candidate_key": left_key,
                "right_candidate_key": right_key,
                "arm_name": arm_name,
                "receipt_count": len(receipts),
                "receipts": [item.logical_record() for item in receipts],
            }
        )
    return (
        ThreeArmFixedDenominatorEvidenceV1(
            left_candidate_key=left_key,
            right_candidate_key=right_key,
            model_checkpoint_sha256=model_sha,
            pair_comparator_sha256=comparator_sha,
            reducer_sha256=reducer_sha,
            binding_mode=BINDING_REAL,
            arms=tuple(arms),  # type: ignore[arg-type]
        ),
        MappingProxyType(transport_hashes),
    )


def move_c_dino_locks_to_device(
    locks: Mapping[str, CDinoCandidateBindingV1], device: torch.device | str
) -> Mapping[str, CDinoCandidateBindingV1]:
    target_device = torch.device(device)
    output: dict[str, CDinoCandidateBindingV1] = {}
    for key, lock in locks.items():
        source = lock.candidate
        candidate = CandidateReferenceFieldV1(
            candidate_key=source.candidate_key,
            layers=source.layers.to(target_device),
            grid_shape=source.grid_shape,
            valid_patch_mask=source.valid_patch_mask,
            physical_gallery_row=source.physical_gallery_row,
            source_image_sha256=source.source_image_sha256,
            source_key=source.source_key,
            cache_payload_sha256=source.cache_payload_sha256,
            geometry_record_sha256=source.geometry_record_sha256,
            tokens_sha256=source.tokens_sha256,
        )
        output[key] = replace(lock, candidate=candidate)
    return MappingProxyType(output)


def evaluate_all_control_scores(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    real_locks: Mapping[str, CandidatePLockV1],
    c_dino_locks: Mapping[str, CDinoCandidateBindingV1],
    c_col_locks: Mapping[str, CandidatePLockV1],
    left_key: str,
    right_key: str,
    *,
    query_id: str,
    c_dino_ineligible_candidate_keys: Sequence[str] = (),
) -> Mapping[str, Mapping[str, Mapping[str, object]]]:
    """Execute all 3x8 inference-only controls under one frozen model."""

    validate_c_col_p_rebuild(real_locks, c_col_locks)
    for key in real_locks:
        if c_dino_locks[key].p_lock_record_sha256_by_direction != real_locks[
            key
        ].p_lock_record_sha256_by_direction:
            raise SR0MTVContractError("C_DINO_V changed the P lock")
    ineligible = frozenset(c_dino_ineligible_candidate_keys)
    if ineligible:
        raise SR0MTVContractError(
            "C_DINO_V full-C128 control must be executable for every candidate"
        )
    real = decode_pair(model, query, real_locks, left_key, right_key)
    c_dino, c_dino_transport = decode_c_dino_pair(
        model, query, c_dino_locks, left_key, right_key
    )
    c_col = decode_pair(model, query, c_col_locks, left_key, right_key)
    p_query_value, _ = permute_query_content(
        query, namespace="RCDE_SR0_MT_P_QUERY_V1", query_id=query_id
    )
    p_query = decode_pair(model, p_query_value, real_locks, left_key, right_key)
    output: dict[str, dict[str, dict[str, object]]] = {}
    for arm_name in (ARM_ALL_PATCH, ARM_QUERY_FULL_REFERENCE, ARM_QUERY_LOCAL_COMPONENTS):
        arm_output: dict[str, dict[str, object]] = {}

        def add(
            name: str,
            contributions: torch.Tensor,
            margin: torch.Tensor,
            *,
            mask_transport_receipt_sha256: str | None = None,
        ) -> None:
            values = torch.as_tensor(contributions).detach().cpu().contiguous()
            scalar = torch.as_tensor(margin).detach().cpu()
            _validate_control_decomposition(
                scalar,
                values.sum(),
                absolute_mass=values.abs().to(torch.float64).sum(),
                operation_count=values.numel() + 8,
                name=f"{name} control reconstruction",
            )
            arm_output[name] = {
                "status": "READY",
                "margin": scalar.to(torch.float64),
                "scalar_contributions": values.to(torch.float64),
                "only_registered_variable_changed": True,
            }
            if name == C_DINO_V:
                if (
                    not isinstance(mask_transport_receipt_sha256, str)
                    or len(mask_transport_receipt_sha256) != 64
                ):
                    raise SR0MTVContractError("C_DINO mask-transport receipt absent")
                arm_output[name].update(
                    {
                        "mask_transport_method": MASK_TRANSPORT_METHOD,
                        "mask_transport_contract_sha256": (
                            MASK_TRANSPORT_CONTRACT_SHA256
                        ),
                        "mask_transport_receipt_sha256": (
                            mask_transport_receipt_sha256
                        ),
                        "token_resize_or_interpolation_count": 0,
                        "p_lock_byte_identical": True,
                    }
                )

        def add_ineligible(name: str, reason: str) -> None:
            arm_output[name] = {
                "status": "NOT_APPLICABLE_CONTRACT",
                "reason": reason,
                "only_registered_variable_changed": True,
            }

        c_dino_arm = c_dino.by_name()[arm_name]
        add(
            C_DINO_V,
            c_dino_arm.scalar_contributions,
            c_dino_arm.logit,
            mask_transport_receipt_sha256=c_dino_transport[arm_name],
        )
        for name, decoded in ((C_COL_P, c_col), (P_QUERY, p_query)):
            arm = decoded.by_name()[arm_name]
            add(name, arm.scalar_contributions, arm.logit)
        p_ref_parts, p_ref_margin = _p_reference_score(
            model,
            query,
            real_locks,
            left_key,
            right_key,
            arm_name,
            namespace="RCDE_SR0_MT_P_REFERENCE_V1",
            query_id=query_id,
        )
        add(P_REFERENCE, p_ref_parts, p_ref_margin)
        for name in (RANDOM_CONNECTED_MATCHED_SHAPE, DISCONNECTED_MATCHED_AREA):
            parts, margin, mask_eligible = _mask_control_score(
                model,
                query,
                real_locks,
                left_key,
                right_key,
                arm_name,
                name,
                namespace=f"RCDE_SR0_MT_{name}_V1",
            )
            if mask_eligible:
                add(name, parts, margin)
            else:
                add_ineligible(name, "MASK_CONTROL_GEOMETRY_INELIGIBLE")
        real_arm = real.by_name()[arm_name]
        unordered_key = "|".join(sorted((left_key, right_key)))
        magnitude = magnitude_only_contributions(
            real_arm.scalar_contributions,
            unordered_pair_key=unordered_key,
            canonical_orientation=1,
        )
        add(MAGNITUDE_ONLY, magnitude, magnitude.sum())
        single_parts, single_margin, single_eligible = _mask_control_score(
            model,
            query,
            real_locks,
            left_key,
            right_key,
            arm_name,
            SINGLE_SEED,
            namespace="RCDE_SR0_MT_SINGLE_SEED_V1",
        )
        if single_eligible:
            add(SINGLE_SEED, single_parts, single_margin)
        else:
            add_ineligible(SINGLE_SEED, "MASK_CONTROL_GEOMETRY_INELIGIBLE")
        output[arm_name] = arm_output
    return MappingProxyType(
        {
            "ALL_PATCH_SAME_MODEL": MappingProxyType(output[ARM_ALL_PATCH]),
            "QUERY_REGION_FULL_REFERENCE_SAME_MODEL": MappingProxyType(
                output[ARM_QUERY_FULL_REFERENCE]
            ),
            "PAIRED_QUERY_REFERENCE_REGION": MappingProxyType(
                output[ARM_QUERY_LOCAL_COMPONENTS]
            ),
        }
    )


__all__ = [
    "SCHEMA_VERSION",
    "C_DINO_V",
    "C_COL_P",
    "P_QUERY",
    "P_REFERENCE",
    "RANDOM_CONNECTED_MATCHED_SHAPE",
    "DISCONNECTED_MATCHED_AREA",
    "MAGNITUDE_ONLY",
    "SINGLE_SEED",
    "CANDIDATE_REORDER",
    "PAIR_SWAP",
    "MANDATORY_CONTROLS",
    "MASK_TRANSPORT_METHOD",
    "MASK_TRANSPORT_CONTRACT_SHA256",
    "TransformReceiptV1",
    "ArmMaskOverrideV1",
    "CDinoCandidateBindingV1",
    "ReferenceMaskTransportReceiptV1",
    "deterministic_derangement",
    "permute_query_content",
    "permute_reference_content",
    "c_dino_v_derangement",
    "transport_reference_mask_positive_overlap",
    "c_dino_mask_transport_preflight_sha256",
    "decode_c_dino_pair",
    "move_c_dino_locks_to_device",
    "validate_c_col_p_rebuild",
    "random_connected_matched_shape",
    "disconnected_matched_area",
    "single_seed_mask",
    "mask_override",
    "magnitude_only_contributions",
    "candidate_reorder",
    "pair_swap_record",
    "evaluate_all_control_scores",
]
