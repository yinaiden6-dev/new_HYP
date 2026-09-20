"""Ordered-root three-arm DINO-RCDE verification for CW1 multitile locks.

This module is deliberately narrower than an SR0 runner.  It consumes already
sealed, target-free CW1 multitile populations and reruns one *shared* decoder
and pair comparator under the three mandatory mask contracts:

* ``ALL_PATCH_SAME_MODEL``;
* ``CW1_QUERY_MULTITILE_FULL_REFERENCE``; and
* ``CW1_QUERY_MULTITILE_LOCAL_COMPONENT_SET``.

The local-component arm never constructs a reference union.  Each query root is
decoded against the reference component bound to that same root, after which
query-patch evidence is overlap-corrected on the sealed query union.  Missing
roots and inactive H0 hypotheses contribute exact zero under the fixed four-term
two-direction denominator.

There is no target/label join, optimizer, model selection, retrieval action, or
protected-endpoint access in this engineering core.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import torch

from .dino_rcde_colnomic_superregion_v1 import (
    FloatReductionClosureError,
    decode_candidate_in_superregion,
    compare_regional_fields,
    validate_conditioned_float_decomposition,
)
from .dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
    MANDATORY_ARMS,
    ROOT_READY,
    STATUS_READY,
    ArmScopeV2,
    CW1MultiTilePopulationV2,
    ThreeArmFamilyV2,
    aggregate_mapped_root_contributions,
)
from .dino_rcde_v1_2_resource_core import PairEvidence


SCHEMA_VERSION = "rc_dino_rcde_cw1_multitile_vdecode_v1"
FIXED_DIRECTIONS = ("a_to_b", "b_to_a")
PHYSICAL_GALLERY_ROWS = 5413

BINDING_REAL = "REAL"
BINDING_JOINT_ROOT_COMPONENT_REORDER = "JOINT_ROOT_COMPONENT_REORDER"
BINDING_C_LOCAL_COMPONENT = "C_LOCAL_COMPONENT_BIND"
BINDING_MODES = (
    BINDING_REAL,
    BINDING_JOINT_ROOT_COMPONENT_REORDER,
    BINDING_C_LOCAL_COMPONENT,
)

TERM_READY = "ORDERED_ROOT_TERM_READY"
TERM_H0 = "ORDERED_ROOT_TERM_H0_EXACT_ZERO"
TERM_NO_COMPARABLE_ROOT = "ORDERED_ROOT_TERM_NO_COMPARABLE_ROOT_EXACT_ZERO"


class CW1MultiTileVDecodeError(ValueError):
    """The ordered-root, matched-model, or fixed-denominator contract failed."""


def _validate_scalar_decomposition(
    observed: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    absolute_mass: torch.Tensor,
    operation_count: int,
    name: str,
) -> tuple[float, float]:
    try:
        return validate_conditioned_float_decomposition(
            observed,
            reconstructed,
            absolute_mass=absolute_mass,
            operation_count=operation_count,
            name=name,
        )
    except FloatReductionClosureError as error:
        raise CW1MultiTileVDecodeError(str(error)) from error


def _sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CW1MultiTileVDecodeError(f"{name} must be a lowercase SHA256")
    return value


def token_tensor_sha256(value: torch.Tensor) -> str:
    """Hash exact token dtype, shape and bytes using the cache convention."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class _RegionalDecoder(Protocol):
    def decode_candidate(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        **kwargs: Any,
    ) -> Any: ...


class _PairComparator(Protocol):
    def compare_relational(
        self,
        relational_g: torch.Tensor,
        relational_c: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> PairEvidence: ...


def _shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    shape = tuple(int(item) for item in value)
    if len(shape) != 2 or any(item <= 0 for item in shape):
        raise CW1MultiTileVDecodeError(f"{name} must be a positive HxW pair")
    return shape


def _mask(
    value: torch.Tensor,
    shape: tuple[int, int],
    *,
    name: str,
    allow_empty: bool = False,
) -> torch.Tensor:
    mask = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous().flatten()
    if mask.shape != (math.prod(shape),):
        raise CW1MultiTileVDecodeError(f"{name} mask/grid mismatch")
    if not allow_empty and not bool(mask.any()):
        raise CW1MultiTileVDecodeError(f"{name} mask is empty")
    return mask


def _mask_sha256(value: torch.Tensor, shape: tuple[int, int]) -> str:
    mask = torch.as_tensor(value, dtype=torch.uint8).detach().cpu().contiguous().flatten()
    payload = (
        int(shape[0]).to_bytes(8, "little")
        + int(shape[1]).to_bytes(8, "little")
        + mask.numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class QueryTokenFieldV1:
    layers: torch.Tensor
    grid_shape: tuple[int, int]
    valid_patch_mask: torch.Tensor
    source_image_sha256: str
    source_key: str
    cache_payload_sha256: str
    geometry_record_sha256: str
    tokens_sha256: str

    def __post_init__(self) -> None:
        shape = _shape(self.grid_shape, name="query grid")
        layers = torch.as_tensor(self.layers)
        mask = _mask(self.valid_patch_mask, shape, name="query valid")
        if (
            layers.ndim != 3
            or layers.shape[1] != math.prod(shape)
            or not layers.is_floating_point()
            or not bool(torch.isfinite(layers).all())
        ):
            raise CW1MultiTileVDecodeError(
                "query layers must be finite floating [L,H*W,D]"
            )
        _sha256(self.source_image_sha256, name="query source image")
        if not isinstance(self.source_key, str) or not self.source_key:
            raise CW1MultiTileVDecodeError("query source key must be nonempty")
        _sha256(self.cache_payload_sha256, name="query cache payload")
        _sha256(self.geometry_record_sha256, name="query geometry record")
        _sha256(self.tokens_sha256, name="query tokens")
        if token_tensor_sha256(layers) != self.tokens_sha256:
            raise CW1MultiTileVDecodeError("query token tensor/cache binding drift")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "valid_patch_mask", mask)


@dataclass(frozen=True)
class CandidateReferenceFieldV1:
    candidate_key: str
    layers: torch.Tensor
    grid_shape: tuple[int, int]
    valid_patch_mask: torch.Tensor
    physical_gallery_row: int
    source_image_sha256: str
    source_key: str
    cache_payload_sha256: str
    geometry_record_sha256: str
    tokens_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise CW1MultiTileVDecodeError("candidate key must be nonempty")
        if (
            isinstance(self.physical_gallery_row, bool)
            or not isinstance(self.physical_gallery_row, int)
            or not 0 <= self.physical_gallery_row < PHYSICAL_GALLERY_ROWS
        ):
            raise CW1MultiTileVDecodeError(
                "reference physical gallery row must be non-negative"
            )
        shape = _shape(self.grid_shape, name="reference grid")
        layers = torch.as_tensor(self.layers)
        mask = _mask(self.valid_patch_mask, shape, name="reference valid")
        if (
            layers.ndim != 3
            or layers.shape[1] != math.prod(shape)
            or not layers.is_floating_point()
            or not bool(torch.isfinite(layers).all())
        ):
            raise CW1MultiTileVDecodeError(
                "reference layers must be finite floating [L,H*W,D]"
            )
        _sha256(self.source_image_sha256, name="reference source image")
        if not isinstance(self.source_key, str) or not self.source_key:
            raise CW1MultiTileVDecodeError("reference source key must be nonempty")
        _sha256(self.cache_payload_sha256, name="reference cache payload")
        _sha256(self.geometry_record_sha256, name="reference geometry record")
        _sha256(self.tokens_sha256, name="reference tokens")
        if token_tensor_sha256(layers) != self.tokens_sha256:
            raise CW1MultiTileVDecodeError("reference token tensor/cache binding drift")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "valid_patch_mask", mask)


@dataclass(frozen=True)
class DirectionalCandidateScopeV1:
    """One candidate/direction's complete P population and selected arm family."""

    population: CW1MultiTilePopulationV2
    family: ThreeArmFamilyV2
    query_source_image_sha256: str
    query_source_key: str
    query_cache_payload_sha256: str
    query_geometry_record_sha256: str
    query_tokens_sha256: str
    reference_physical_gallery_row: int
    reference_source_image_sha256: str
    reference_source_key: str
    reference_cache_payload_sha256: str
    reference_geometry_record_sha256: str
    reference_tokens_sha256: str
    binding_mode: str = BINDING_REAL
    root_component_order: tuple[int, ...] | None = None
    component_source_root_ordinals: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            self.population.candidate_key != self.family.candidate_key
            or self.population.direction != self.family.direction
            or self.population.population_sha256 != self.family.population_sha256
            or not 0 <= self.family.bank_ordinal < len(self.population.rows)
        ):
            raise CW1MultiTileVDecodeError("population/family binding drift")
        for name, value in (
            ("query source image", self.query_source_image_sha256),
            ("query cache payload", self.query_cache_payload_sha256),
            ("query geometry record", self.query_geometry_record_sha256),
            ("query tokens", self.query_tokens_sha256),
            ("reference source image", self.reference_source_image_sha256),
            ("reference cache payload", self.reference_cache_payload_sha256),
            ("reference geometry record", self.reference_geometry_record_sha256),
            ("reference tokens", self.reference_tokens_sha256),
        ):
            _sha256(value, name=name)
        if (
            not isinstance(self.query_source_key, str)
            or not self.query_source_key
            or not isinstance(self.reference_source_key, str)
            or not self.reference_source_key
        ):
            raise CW1MultiTileVDecodeError("scope source key is absent")
        if (
            isinstance(self.reference_physical_gallery_row, bool)
            or not isinstance(self.reference_physical_gallery_row, int)
            or not 0
            <= self.reference_physical_gallery_row
            < PHYSICAL_GALLERY_ROWS
        ):
            raise CW1MultiTileVDecodeError(
                "scope reference physical gallery row must be non-negative"
            )
        query_geometry_hashes = {
            item.dino_query_geometry_sha256 for item in self.population.root_bindings
        }
        reference_geometry_hashes = {
            item.dino_reference_geometry_sha256
            for item in self.population.root_bindings
        }
        if query_geometry_hashes != {self.query_geometry_record_sha256}:
            raise CW1MultiTileVDecodeError(
                "query token provenance/population geometry drift"
            )
        if reference_geometry_hashes != {self.reference_geometry_record_sha256}:
            raise CW1MultiTileVDecodeError(
                "reference token provenance/population geometry drift"
            )
        row = self.population.rows[self.family.bank_ordinal]
        full, query_full, query_local = self.family.arms
        if (
            tuple(item.name for item in self.family.arms) != MANDATORY_ARMS
            or query_full.active != (row.status == STATUS_READY)
            or query_local.active != (row.status == STATUS_READY)
            or query_full.root_ordinals
            != row.structural_region.contributing_root_ordinals
            or query_local.root_ordinals != query_full.root_ordinals
            or not torch.equal(query_full.query_mask, row.dino_query_union_mask)
            or not torch.equal(query_local.query_mask, query_full.query_mask)
            or full.name != ARM_ALL_PATCH
        ):
            raise CW1MultiTileVDecodeError("selected three-arm row binding drift")

        roots = tuple(row.structural_region.contributing_root_ordinals)
        order = roots if self.root_component_order is None else tuple(
            int(item) for item in self.root_component_order
        )
        component_sources = roots if self.component_source_root_ordinals is None else tuple(
            int(item) for item in self.component_source_root_ordinals
        )
        if tuple(sorted(order)) != tuple(sorted(roots)) or len(order) != len(roots):
            raise CW1MultiTileVDecodeError(
                "root/component presentation order is not a complete permutation"
            )
        if (
            tuple(sorted(component_sources)) != tuple(sorted(roots))
            or len(component_sources) != len(roots)
        ):
            raise CW1MultiTileVDecodeError(
                "local component source ledger is not a complete multiset permutation"
            )
        if self.binding_mode not in BINDING_MODES:
            raise CW1MultiTileVDecodeError("unknown root/component binding mode")
        if self.binding_mode == BINDING_REAL:
            if order != roots or component_sources != roots:
                raise CW1MultiTileVDecodeError(
                    "REAL binding must retain canonical root/component addresses"
                )
        elif self.binding_mode == BINDING_JOINT_ROOT_COMPONENT_REORDER:
            if len(roots) < 2 or order == roots or component_sources != roots:
                raise CW1MultiTileVDecodeError(
                    "joint reorder must reorder complete root/component pairs only"
                )
        else:
            if order != roots or len(roots) < 2 or any(
                root == source
                for root, source in zip(roots, component_sources, strict=True)
            ):
                raise CW1MultiTileVDecodeError(
                    "C_LOCAL_COMPONENT_BIND requires a fixed-point-free component-only derangement"
                )
            binding_by_root = {item.root_ordinal: item for item in row.root_bindings}
            if not any(
                not torch.equal(
                    binding_by_root[root].dino_reference_component_mask,
                    binding_by_root[source].dino_reference_component_mask,
                )
                for root, source in zip(roots, component_sources, strict=True)
            ):
                raise CW1MultiTileVDecodeError(
                    "C_LOCAL_COMPONENT_BIND did not change addressed content"
                )
        object.__setattr__(self, "root_component_order", order)
        object.__setattr__(self, "component_source_root_ordinals", component_sources)

    @property
    def candidate_key(self) -> str:
        return self.population.candidate_key

    @property
    def direction(self) -> str:
        return self.population.direction

    @property
    def row(self):
        return self.population.rows[self.family.bank_ordinal]

    def component_source_for_root(self, root_ordinal: int) -> int:
        roots = self.row.structural_region.contributing_root_ordinals
        try:
            index = roots.index(int(root_ordinal))
        except ValueError as error:
            raise CW1MultiTileVDecodeError("component source requested for foreign root") from error
        assert self.component_source_root_ordinals is not None
        return self.component_source_root_ordinals[index]


def jointly_reorder_root_component_scope(
    scope: DirectionalCandidateScopeV1,
    root_order: tuple[int, ...],
) -> DirectionalCandidateScopeV1:
    """Presentation-only reorder: each root travels with its bound component."""

    return replace(
        scope,
        binding_mode=BINDING_JOINT_ROOT_COMPONENT_REORDER,
        root_component_order=tuple(root_order),
        component_source_root_ordinals=tuple(
            scope.row.structural_region.contributing_root_ordinals
        ),
    )


def derange_local_component_binding_scope(
    scope: DirectionalCandidateScopeV1,
    component_source_root_ordinals: tuple[int, ...],
) -> DirectionalCandidateScopeV1:
    """Executable C_LOCAL_COMPONENT_BIND without rewriting the clean family."""

    return replace(
        scope,
        binding_mode=BINDING_C_LOCAL_COMPONENT,
        root_component_order=tuple(
            scope.row.structural_region.contributing_root_ordinals
        ),
        component_source_root_ordinals=tuple(component_source_root_ordinals),
    )


@dataclass(frozen=True)
class RootPairDecodeReceiptV1:
    root_ordinal: int
    owner_component_source_root_ordinal: int | None
    opponent_component_source_root_ordinal: int | None
    binding_mode: str
    query_mask_sha256: str
    owner_reference_mask_sha256: str
    opponent_reference_mask_sha256: str

    def __post_init__(self) -> None:
        if self.binding_mode not in BINDING_MODES:
            raise CW1MultiTileVDecodeError("root receipt binding mode drift")


@dataclass(frozen=True)
class OrderedRootArmTermV1:
    arm_name: str
    owner_candidate_key: str
    opponent_candidate_key: str
    direction: str
    status: str
    query_mask: torch.Tensor
    patch_evidence: torch.Tensor
    scalar_contributions: torch.Tensor
    logit: torch.Tensor
    decoded_root_ordinals: tuple[int, ...]
    decode_receipts: tuple[RootPairDecodeReceiptV1, ...]

    def __post_init__(self) -> None:
        if self.arm_name not in MANDATORY_ARMS:
            raise CW1MultiTileVDecodeError("term arm drift")
        if self.direction not in FIXED_DIRECTIONS:
            raise CW1MultiTileVDecodeError("term direction drift")
        if self.status not in {TERM_READY, TERM_H0, TERM_NO_COMPARABLE_ROOT}:
            raise CW1MultiTileVDecodeError("term status drift")
        mask = torch.as_tensor(self.query_mask, dtype=torch.bool).detach().cpu().flatten()
        patch = torch.as_tensor(self.patch_evidence)
        scalar_parts = torch.as_tensor(self.scalar_contributions)
        logit = torch.as_tensor(self.logit)
        if (
            patch.ndim != 1
            or scalar_parts.shape != patch.shape
            or mask.shape != patch.shape
            or logit.ndim != 0
            or not patch.is_floating_point()
            or patch.dtype != scalar_parts.dtype
            or not bool(torch.isfinite(patch).all())
            or not bool(torch.isfinite(scalar_parts).all())
            or not bool(torch.isfinite(logit))
        ):
            raise CW1MultiTileVDecodeError("term tensor schema drift")
        outside = ~mask.to(patch.device)
        if bool(patch[outside].ne(0).any()) or bool(scalar_parts[outside].ne(0).any()):
            raise CW1MultiTileVDecodeError("term evidence leaked outside its query mask")
        _validate_scalar_decomposition(
            logit,
            scalar_parts.sum(),
            absolute_mass=scalar_parts.abs().to(torch.float64).sum(),
            operation_count=scalar_parts.numel() + 2,
            name="term scalar reconstruction",
        )
        if self.status != TERM_READY and (
            bool(patch.ne(0).any()) or bool(logit.ne(0)) or self.decode_receipts
        ):
            raise CW1MultiTileVDecodeError("inactive/missing term is not exact zero")
        roots = tuple(int(item) for item in self.decoded_root_ordinals)
        receipts = tuple(self.decode_receipts)
        if roots != tuple(item.root_ordinal for item in receipts):
            raise CW1MultiTileVDecodeError("term root receipt order drift")
        object.__setattr__(self, "query_mask", mask)
        object.__setattr__(self, "patch_evidence", patch)
        object.__setattr__(self, "scalar_contributions", scalar_parts)
        object.__setattr__(self, "logit", logit)
        object.__setattr__(self, "decoded_root_ordinals", roots)
        object.__setattr__(self, "decode_receipts", receipts)


@dataclass(frozen=True)
class FixedDenominatorArmEvidenceV1:
    arm_name: str
    left_candidate_key: str
    right_candidate_key: str
    forward_by_direction: tuple[OrderedRootArmTermV1, OrderedRootArmTermV1]
    reverse_by_direction: tuple[OrderedRootArmTermV1, OrderedRootArmTermV1]
    signed_term_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    scalar_contributions: torch.Tensor
    logit: torch.Tensor

    def __post_init__(self) -> None:
        forward = tuple(self.forward_by_direction)
        reverse = tuple(self.reverse_by_direction)
        if (
            self.arm_name not in MANDATORY_ARMS
            or len(forward) != 2
            or len(reverse) != 2
            or tuple(item.direction for item in forward) != FIXED_DIRECTIONS
            or tuple(item.direction for item in reverse) != FIXED_DIRECTIONS
            or any(item.arm_name != self.arm_name for item in (*forward, *reverse))
            or any(
                item.owner_candidate_key != self.left_candidate_key
                or item.opponent_candidate_key != self.right_candidate_key
                for item in forward
            )
            or any(
                item.owner_candidate_key != self.right_candidate_key
                or item.opponent_candidate_key != self.left_candidate_key
                for item in reverse
            )
        ):
            raise CW1MultiTileVDecodeError("fixed-denominator term order drift")
        signed = tuple(torch.as_tensor(item) for item in self.signed_term_logits)
        if (
            len(signed) != 4
            or any(item.ndim != 0 for item in signed)
            or any(item.dtype not in (torch.float32, torch.float64) for item in signed)
            or any(item.dtype != signed[0].dtype or item.device != signed[0].device for item in signed)
            or not all(bool(torch.isfinite(item)) for item in signed)
        ):
            raise CW1MultiTileVDecodeError("fixed denominator requires exactly four scalars")
        expected_signed = (
            forward[0].logit,
            -reverse[0].logit,
            forward[1].logit,
            -reverse[1].logit,
        )
        if any(
            not torch.equal(observed, expected)
            for observed, expected in zip(signed, expected_signed, strict=True)
        ):
            raise CW1MultiTileVDecodeError("signed four-term ledger drift")
        parts = torch.as_tensor(self.scalar_contributions)
        logit = torch.as_tensor(self.logit)
        expected_logit = sum(signed, start=logit.new_zeros(())) / 4.0
        if (
            parts.ndim != 1
            or logit.ndim != 0
            or parts.dtype != signed[0].dtype
            or logit.dtype != signed[0].dtype
            or parts.device != signed[0].device
            or logit.device != signed[0].device
            or not bool(torch.isfinite(parts).all())
            or not bool(torch.isfinite(logit))
        ):
            raise CW1MultiTileVDecodeError("fixed four-term reconstruction failed")
        signed_absolute_mass = sum(
            (item.abs().to(torch.float64) for item in signed),
            start=torch.zeros((), dtype=torch.float64, device=logit.device),
        ) / 4.0
        _validate_scalar_decomposition(
            logit,
            expected_logit,
            absolute_mass=signed_absolute_mass,
            operation_count=8,
            name="fixed four-term scalar ledger reconstruction",
        )
        term_absolute_mass = sum(
            (
                item.scalar_contributions.abs().to(torch.float64).sum()
                for item in (*forward, *reverse)
            ),
            start=torch.zeros((), dtype=torch.float64, device=parts.device),
        ) / 4.0
        _validate_scalar_decomposition(
            logit,
            parts.sum(),
            absolute_mass=term_absolute_mass,
            operation_count=parts.numel() + 8,
            name="fixed four-term contribution reconstruction",
        )
        object.__setattr__(self, "forward_by_direction", forward)
        object.__setattr__(self, "reverse_by_direction", reverse)
        object.__setattr__(self, "signed_term_logits", signed)
        object.__setattr__(self, "scalar_contributions", parts)
        object.__setattr__(self, "logit", logit)


@dataclass(frozen=True)
class ThreeArmFixedDenominatorEvidenceV1:
    left_candidate_key: str
    right_candidate_key: str
    model_checkpoint_sha256: str
    pair_comparator_sha256: str
    reducer_sha256: str
    binding_mode: str
    arms: tuple[
        FixedDenominatorArmEvidenceV1,
        FixedDenominatorArmEvidenceV1,
        FixedDenominatorArmEvidenceV1,
    ]

    def __post_init__(self) -> None:
        arms = tuple(self.arms)
        if tuple(item.arm_name for item in arms) != MANDATORY_ARMS:
            raise CW1MultiTileVDecodeError("all three decoded arms must be present in order")
        if any(
            item.left_candidate_key != self.left_candidate_key
            or item.right_candidate_key != self.right_candidate_key
            for item in arms
        ):
            raise CW1MultiTileVDecodeError("decoded pair key drift")
        for value in (
            self.model_checkpoint_sha256,
            self.pair_comparator_sha256,
            self.reducer_sha256,
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise CW1MultiTileVDecodeError("decoded model/comparator/reducer hash drift")
        if self.binding_mode not in BINDING_MODES:
            raise CW1MultiTileVDecodeError("decoded binding mode drift")
        object.__setattr__(self, "arms", arms)

    def by_name(self) -> Mapping[str, FixedDenominatorArmEvidenceV1]:
        return MappingProxyType({item.arm_name: item for item in self.arms})


def _scope_arm(scope: DirectionalCandidateScopeV1, arm_name: str) -> ArmScopeV2:
    return scope.family.arms[MANDATORY_ARMS.index(arm_name)]


def _zero_term(
    *,
    arm_name: str,
    owner_key: str,
    opponent_key: str,
    direction: str,
    status: str,
    query_mask: torch.Tensor,
    exemplar: torch.Tensor,
) -> OrderedRootArmTermV1:
    count = int(torch.as_tensor(query_mask).numel())
    zero = exemplar.new_zeros((count,))
    return OrderedRootArmTermV1(
        arm_name=arm_name,
        owner_candidate_key=owner_key,
        opponent_candidate_key=opponent_key,
        direction=direction,
        status=status,
        query_mask=query_mask,
        patch_evidence=zero,
        scalar_contributions=zero.clone(),
        logit=exemplar.new_zeros(()),
        decoded_root_ordinals=(),
        decode_receipts=(),
    )


def _decode_pair_masks(
    decoder: _RegionalDecoder,
    comparator: _PairComparator,
    query: QueryTokenFieldV1,
    owner: CandidateReferenceFieldV1,
    opponent: CandidateReferenceFieldV1,
    *,
    query_mask: torch.Tensor,
    owner_reference_mask: torch.Tensor,
    opponent_reference_mask: torch.Tensor,
    streaming_chunk_size: int | None,
    consensus_tile_shape: tuple[int, int, int, int] | None,
) -> tuple[PairEvidence, RootPairDecodeReceiptV1]:
    qmask = _mask(query_mask, query.grid_shape, name="decode query")
    owner_mask = _mask(
        owner_reference_mask, owner.grid_shape, name="decode owner reference"
    )
    opponent_mask = _mask(
        opponent_reference_mask,
        opponent.grid_shape,
        name="decode opponent reference",
    )
    qsha = _mask_sha256(qmask, query.grid_shape)
    owner_sha = _mask_sha256(owner_mask, owner.grid_shape)
    opponent_sha = _mask_sha256(opponent_mask, opponent.grid_shape)
    owner_field = decode_candidate_in_superregion(
        decoder,
        query.layers,
        owner.layers,
        candidate_key=owner.candidate_key,
        query_mask=qmask,
        reference_mask=owner_mask,
        query_grid_shape=query.grid_shape,
        reference_grid_shape=owner.grid_shape,
        query_region_sha256=qsha,
        reference_region_sha256=owner_sha,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )
    opponent_field = decode_candidate_in_superregion(
        decoder,
        query.layers,
        opponent.layers,
        candidate_key=opponent.candidate_key,
        query_mask=qmask,
        reference_mask=opponent_mask,
        query_grid_shape=query.grid_shape,
        reference_grid_shape=opponent.grid_shape,
        query_region_sha256=qsha,
        reference_region_sha256=opponent_sha,
        streaming_chunk_size=streaming_chunk_size,
        consensus_tile_shape=consensus_tile_shape,
    )
    pair = compare_regional_fields(comparator, owner_field, opponent_field)
    receipt = RootPairDecodeReceiptV1(
        root_ordinal=-1,
        owner_component_source_root_ordinal=None,
        opponent_component_source_root_ordinal=None,
        binding_mode=BINDING_REAL,
        query_mask_sha256=qsha,
        owner_reference_mask_sha256=owner_sha,
        opponent_reference_mask_sha256=opponent_sha,
    )
    return pair, receipt


def _decode_arm_term(
    decoder: _RegionalDecoder,
    comparator: _PairComparator,
    query: QueryTokenFieldV1,
    owner: CandidateReferenceFieldV1,
    opponent: CandidateReferenceFieldV1,
    owner_scope: DirectionalCandidateScopeV1,
    opponent_scope: DirectionalCandidateScopeV1,
    *,
    arm_name: str,
    streaming_chunk_size: int | None,
    consensus_tile_shape: tuple[int, int, int, int] | None,
) -> OrderedRootArmTermV1:
    arm = _scope_arm(owner_scope, arm_name)
    if arm_name == ARM_ALL_PATCH:
        pair, receipt = _decode_pair_masks(
            decoder,
            comparator,
            query,
            owner,
            opponent,
            query_mask=query.valid_patch_mask,
            owner_reference_mask=owner.valid_patch_mask,
            opponent_reference_mask=opponent.valid_patch_mask,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        valid = query.valid_patch_mask.to(pair.contributions.device)
        patch = pair.contributions * valid.to(pair.contributions.dtype)
        scalar_parts = patch / valid.sum().to(patch.dtype)
        receipt = RootPairDecodeReceiptV1(
            root_ordinal=-1,
            owner_component_source_root_ordinal=None,
            opponent_component_source_root_ordinal=None,
            binding_mode=owner_scope.binding_mode,
            query_mask_sha256=receipt.query_mask_sha256,
            owner_reference_mask_sha256=receipt.owner_reference_mask_sha256,
            opponent_reference_mask_sha256=receipt.opponent_reference_mask_sha256,
        )
        return OrderedRootArmTermV1(
            arm_name=arm_name,
            owner_candidate_key=owner.candidate_key,
            opponent_candidate_key=opponent.candidate_key,
            direction=owner_scope.direction,
            status=TERM_READY,
            query_mask=query.valid_patch_mask,
            patch_evidence=patch,
            scalar_contributions=scalar_parts,
            logit=pair.logit,
            decoded_root_ordinals=(-1,),
            decode_receipts=(receipt,),
        )

    row = owner_scope.row
    if not arm.active or row.status != STATUS_READY:
        return _zero_term(
            arm_name=arm_name,
            owner_key=owner.candidate_key,
            opponent_key=opponent.candidate_key,
            direction=owner_scope.direction,
            status=TERM_H0,
            query_mask=arm.query_mask,
            exemplar=query.layers,
        )

    if owner_scope.binding_mode != opponent_scope.binding_mode:
        raise CW1MultiTileVDecodeError(
            "owner/opponent root-component binding modes do not match"
        )
    owner_binding_by_root = {item.root_ordinal: item for item in row.root_bindings}
    opponent_binding_by_root = {
        item.root_ordinal: item for item in opponent_scope.row.root_bindings
    }
    owner_query_tile_by_root = dict(zip(
        arm.root_ordinals, arm.query_tile_masks, strict=True
    ))
    owner_reference_mask_by_root = dict(zip(
        arm.root_ordinals, arm.reference_masks_by_root, strict=True
    ))
    opponent_arm = _scope_arm(opponent_scope, arm_name)
    opponent_reference_mask_by_root = dict(zip(
        opponent_arm.root_ordinals,
        opponent_arm.reference_masks_by_root,
        strict=True,
    ))
    contribution_by_root: dict[int, torch.Tensor] = {}
    receipt_by_root: dict[int, RootPairDecodeReceiptV1] = {}
    assert owner_scope.root_component_order is not None
    for root in owner_scope.root_component_order:
        owner_binding = owner_binding_by_root[root]
        opponent_binding = opponent_binding_by_root[root]
        query_tile = owner_query_tile_by_root[root]
        if (
            not torch.equal(query_tile, owner_binding.dino_query_tile_mask)
            or not torch.equal(query_tile, opponent_binding.dino_query_tile_mask)
        ):
            raise CW1MultiTileVDecodeError(
                "candidate-specific populations disagree on a fixed query root"
            )
        if owner_binding.status != ROOT_READY:
            continue
        owner_component_source: int | None = None
        opponent_component_source: int | None = None
        if arm_name == ARM_QUERY_FULL_REFERENCE:
            owner_ref_mask = owner_reference_mask_by_root[root]
            opponent_ref_mask = opponent.valid_patch_mask
            if not torch.equal(owner_ref_mask, owner.valid_patch_mask):
                raise CW1MultiTileVDecodeError(
                    "full-reference arm did not use the complete owner valid mask"
                )
        elif arm_name == ARM_QUERY_LOCAL_COMPONENTS:
            owner_component_source = owner_scope.component_source_for_root(root)
            opponent_component_source = opponent_scope.component_source_for_root(root)
            owner_source_binding = owner_binding_by_root[owner_component_source]
            opponent_source_binding = opponent_binding_by_root[
                opponent_component_source
            ]
            if (
                owner_source_binding.status != ROOT_READY
                or opponent_source_binding.status != ROOT_READY
            ):
                continue
            owner_ref_mask = owner_reference_mask_by_root[owner_component_source]
            opponent_ref_mask = opponent_reference_mask_by_root[
                opponent_component_source
            ]
            if not torch.equal(
                owner_ref_mask, owner_source_binding.dino_reference_component_mask
            ) or not torch.equal(
                opponent_ref_mask,
                opponent_source_binding.dino_reference_component_mask,
            ):
                raise CW1MultiTileVDecodeError(
                    "local arm component-source binding drift"
                )
        else:  # pragma: no cover - guarded by the public arm enumeration
            raise CW1MultiTileVDecodeError("unknown regional arm")
        pair, receipt = _decode_pair_masks(
            decoder,
            comparator,
            query,
            owner,
            opponent,
            query_mask=query_tile,
            owner_reference_mask=owner_ref_mask,
            opponent_reference_mask=opponent_ref_mask,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        contribution_by_root[root] = pair.contributions
        receipt_by_root[root] = RootPairDecodeReceiptV1(
            root_ordinal=root,
            owner_component_source_root_ordinal=owner_component_source,
            opponent_component_source_root_ordinal=opponent_component_source,
            binding_mode=owner_scope.binding_mode,
            query_mask_sha256=receipt.query_mask_sha256,
            owner_reference_mask_sha256=receipt.owner_reference_mask_sha256,
            opponent_reference_mask_sha256=receipt.opponent_reference_mask_sha256,
        )

    if not contribution_by_root:
        return _zero_term(
            arm_name=arm_name,
            owner_key=owner.candidate_key,
            opponent_key=opponent.candidate_key,
            direction=owner_scope.direction,
            status=TERM_NO_COMPARABLE_ROOT,
            query_mask=arm.query_mask,
            exemplar=query.layers,
        )
    patch, logit = aggregate_mapped_root_contributions(row, contribution_by_root)
    denominator = row.dino_query_union_mask.sum().to(
        device=patch.device, dtype=patch.dtype
    )
    scalar_parts = patch / denominator
    return OrderedRootArmTermV1(
        arm_name=arm_name,
        owner_candidate_key=owner.candidate_key,
        opponent_candidate_key=opponent.candidate_key,
        direction=owner_scope.direction,
        status=TERM_READY,
        query_mask=arm.query_mask,
        patch_evidence=patch,
        scalar_contributions=scalar_parts,
        logit=logit,
        decoded_root_ordinals=tuple(sorted(contribution_by_root)),
        decode_receipts=tuple(
            receipt_by_root[root] for root in sorted(receipt_by_root)
        ),
    )


def _reduce_arm(
    arm_name: str,
    left_key: str,
    right_key: str,
    forward: tuple[OrderedRootArmTermV1, OrderedRootArmTermV1],
    reverse: tuple[OrderedRootArmTermV1, OrderedRootArmTermV1],
) -> FixedDenominatorArmEvidenceV1:
    signed = (
        forward[0].logit,
        -reverse[0].logit,
        forward[1].logit,
        -reverse[1].logit,
    )
    scalar_parts = (
        forward[0].scalar_contributions
        - reverse[0].scalar_contributions
        + forward[1].scalar_contributions
        - reverse[1].scalar_contributions
    ) / 4.0
    logit = sum(signed, start=scalar_parts.new_zeros(())) / 4.0
    return FixedDenominatorArmEvidenceV1(
        arm_name=arm_name,
        left_candidate_key=left_key,
        right_candidate_key=right_key,
        forward_by_direction=forward,
        reverse_by_direction=reverse,
        signed_term_logits=signed,
        scalar_contributions=scalar_parts,
        logit=logit,
    )


def registered_reducer_source_sha256() -> str:
    """Byte hash of the exact registered fixed-four-term reducer source."""

    return hashlib.sha256(inspect.getsource(_reduce_arm).encode("utf-8")).hexdigest()


def _runtime_lineage_sha256(value: object, attribute: str, *, name: str) -> str:
    try:
        digest = getattr(value, attribute)
    except AttributeError as error:
        raise CW1MultiTileVDecodeError(
            f"{name} runtime lineage receipt is absent"
        ) from error
    return _sha256(digest, name=f"{name} runtime lineage")


def _query_provenance(value: QueryTokenFieldV1) -> tuple[object, ...]:
    return (
        value.source_image_sha256,
        value.source_key,
        value.cache_payload_sha256,
        value.geometry_record_sha256,
        value.tokens_sha256,
    )


def _scope_query_provenance(value: DirectionalCandidateScopeV1) -> tuple[object, ...]:
    return (
        value.query_source_image_sha256,
        value.query_source_key,
        value.query_cache_payload_sha256,
        value.query_geometry_record_sha256,
        value.query_tokens_sha256,
    )


def _reference_provenance(value: CandidateReferenceFieldV1) -> tuple[object, ...]:
    return (
        value.candidate_key,
        value.physical_gallery_row,
        value.source_image_sha256,
        value.source_key,
        value.cache_payload_sha256,
        value.geometry_record_sha256,
        value.tokens_sha256,
    )


def _scope_reference_provenance(
    value: DirectionalCandidateScopeV1,
) -> tuple[object, ...]:
    return (
        value.candidate_key,
        value.reference_physical_gallery_row,
        value.reference_source_image_sha256,
        value.reference_source_key,
        value.reference_cache_payload_sha256,
        value.reference_geometry_record_sha256,
        value.reference_tokens_sha256,
    )


def decode_ordered_root_three_arm_pair(
    decoder: _RegionalDecoder,
    comparator: _PairComparator,
    query: QueryTokenFieldV1,
    candidates: Mapping[str, CandidateReferenceFieldV1],
    scopes: Mapping[tuple[str, str], DirectionalCandidateScopeV1],
    *,
    left_candidate_key: str,
    right_candidate_key: str,
    streaming_chunk_size: int | None = None,
    consensus_tile_shape: tuple[int, int, int, int] | None = None,
) -> ThreeArmFixedDenominatorEvidenceV1:
    """Rerun all three arms with ordered roots and a fixed four-term reducer.

    Candidate containers are addressed only by immutable keys, so changing the
    insertion/order of ``candidates`` or ``scopes`` cannot change the result.
    """

    if left_candidate_key == right_candidate_key:
        raise CW1MultiTileVDecodeError("pair candidates must be distinct")
    try:
        left = candidates[left_candidate_key]
        right = candidates[right_candidate_key]
    except KeyError as exc:
        raise CW1MultiTileVDecodeError("pair candidate is absent") from exc
    if left.candidate_key != left_candidate_key or right.candidate_key != right_candidate_key:
        raise CW1MultiTileVDecodeError("candidate mapping key drift")
    if (
        left.layers.device != query.layers.device
        or right.layers.device != query.layers.device
        or left.layers.dtype != query.layers.dtype
        or right.layers.dtype != query.layers.dtype
    ):
        raise CW1MultiTileVDecodeError("query/reference dtype or device drift")

    # ``frozen=True`` protects the dataclass fields, but not the storage of the
    # tensors held by those fields.  Recompute the payload digests immediately
    # before any model call so an in-place mutation after construction cannot
    # silently execute under a previously sealed provenance receipt.
    if token_tensor_sha256(query.layers) != query.tokens_sha256:
        raise CW1MultiTileVDecodeError(
            "query token tensor mutated after provenance seal"
        )
    for candidate in (left, right):
        if token_tensor_sha256(candidate.layers) != candidate.tokens_sha256:
            raise CW1MultiTileVDecodeError(
                "reference token tensor mutated after provenance seal"
            )

    bundles: dict[tuple[str, str], DirectionalCandidateScopeV1] = {}
    for candidate_key in (left_candidate_key, right_candidate_key):
        for direction in FIXED_DIRECTIONS:
            key = (candidate_key, direction)
            try:
                bundle = scopes[key]
            except KeyError as exc:
                raise CW1MultiTileVDecodeError(
                    f"missing candidate/direction scope: {key}"
                ) from exc
            if bundle.candidate_key != candidate_key or bundle.direction != direction:
                raise CW1MultiTileVDecodeError("scope mapping key drift")
            bundles[key] = bundle

    binding_modes = {item.binding_mode for item in bundles.values()}
    if len(binding_modes) != 1:
        raise CW1MultiTileVDecodeError(
            "pair scopes mix REAL/reorder/destruction binding modes"
        )
    binding_mode = next(iter(binding_modes))

    runtime_model_sha256 = _runtime_lineage_sha256(
        decoder, "model_checkpoint_sha256", name="decoder"
    )
    runtime_comparator_sha256 = _runtime_lineage_sha256(
        comparator, "pair_comparator_sha256", name="comparator"
    )
    runtime_reducer_sha256 = registered_reducer_source_sha256()
    model_hashes = {item.family.model_checkpoint_sha256 for item in bundles.values()}
    comparator_hashes = {item.family.pair_comparator_sha256 for item in bundles.values()}
    reducer_hashes = {item.family.reducer_sha256 for item in bundles.values()}
    if (
        model_hashes != {runtime_model_sha256}
        or comparator_hashes != {runtime_comparator_sha256}
        or reducer_hashes != {runtime_reducer_sha256}
    ):
        raise CW1MultiTileVDecodeError(
            "three-arm family lineage does not match runtime model/comparator/reducer"
        )
    for candidate_key, candidate in (
        (left_candidate_key, left),
        (right_candidate_key, right),
    ):
        for direction in FIXED_DIRECTIONS:
            bundle = bundles[(candidate_key, direction)]
            if _query_provenance(query) != _scope_query_provenance(bundle):
                raise CW1MultiTileVDecodeError(
                    "query token provenance/scope binding drift"
                )
            if _reference_provenance(candidate) != _scope_reference_provenance(
                bundle
            ):
                raise CW1MultiTileVDecodeError(
                    "reference token provenance/scope binding drift"
                )
            all_patch = _scope_arm(bundle, ARM_ALL_PATCH)
            if (
                not torch.equal(all_patch.query_mask, query.valid_patch_mask)
                or not torch.equal(all_patch.full_reference_mask, candidate.valid_patch_mask)
            ):
                raise CW1MultiTileVDecodeError(
                    "three-arm family is not bound to the supplied valid masks"
                )

    arm_results: list[FixedDenominatorArmEvidenceV1] = []
    for arm_name in MANDATORY_ARMS:
        forward_items: list[OrderedRootArmTermV1] = []
        reverse_items: list[OrderedRootArmTermV1] = []
        for direction in FIXED_DIRECTIONS:
            forward_items.append(
                _decode_arm_term(
                    decoder,
                    comparator,
                    query,
                    left,
                    right,
                    bundles[(left_candidate_key, direction)],
                    bundles[(right_candidate_key, direction)],
                    arm_name=arm_name,
                    streaming_chunk_size=streaming_chunk_size,
                    consensus_tile_shape=consensus_tile_shape,
                )
            )
            reverse_items.append(
                _decode_arm_term(
                    decoder,
                    comparator,
                    query,
                    right,
                    left,
                    bundles[(right_candidate_key, direction)],
                    bundles[(left_candidate_key, direction)],
                    arm_name=arm_name,
                    streaming_chunk_size=streaming_chunk_size,
                    consensus_tile_shape=consensus_tile_shape,
                )
            )
        arm_results.append(
            _reduce_arm(
                arm_name,
                left_candidate_key,
                right_candidate_key,
                tuple(forward_items),  # type: ignore[arg-type]
                tuple(reverse_items),  # type: ignore[arg-type]
            )
        )
    return ThreeArmFixedDenominatorEvidenceV1(
        left_candidate_key=left_candidate_key,
        right_candidate_key=right_candidate_key,
        model_checkpoint_sha256=next(iter(model_hashes)),
        pair_comparator_sha256=next(iter(comparator_hashes)),
        reducer_sha256=next(iter(reducer_hashes)),
        binding_mode=binding_mode,
        arms=tuple(arm_results),  # type: ignore[arg-type]
    )


__all__ = [
    "SCHEMA_VERSION",
    "FIXED_DIRECTIONS",
    "BINDING_REAL",
    "BINDING_JOINT_ROOT_COMPONENT_REORDER",
    "BINDING_C_LOCAL_COMPONENT",
    "BINDING_MODES",
    "TERM_READY",
    "TERM_H0",
    "TERM_NO_COMPARABLE_ROOT",
    "CW1MultiTileVDecodeError",
    "token_tensor_sha256",
    "QueryTokenFieldV1",
    "CandidateReferenceFieldV1",
    "DirectionalCandidateScopeV1",
    "RootPairDecodeReceiptV1",
    "OrderedRootArmTermV1",
    "FixedDenominatorArmEvidenceV1",
    "ThreeArmFixedDenominatorEvidenceV1",
    "jointly_reorder_root_component_scope",
    "derange_local_component_binding_scope",
    "registered_reducer_source_sha256",
    "decode_ordered_root_three_arm_pair",
]
