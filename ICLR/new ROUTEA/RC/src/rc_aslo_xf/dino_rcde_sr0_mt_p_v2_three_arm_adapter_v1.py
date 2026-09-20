"""Exact Natural/Core P-lock V2 projection into the frozen V1 three-arm runtime.

This module is deliberately a small compatibility boundary, not a schema
upgrade.  It validates the complete score-erased Natural V2 wrapper and nested
Core V2 record, then constructs the historical :class:`CandidatePLockV1`
runtime view only when that projection is lossless.

In particular, ``ROOT_REFERENCE_MISSING`` is *not* representable by the V1
runtime: V2 preserves its query mask in the fixed denominator whereas V1 H0
requires both masks to be empty.  Such a record therefore fails closed.  No
target, rival, score, rank, outcome, model forward, or file I/O exists here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    QueryTokenFieldV1,
)
from .dino_rcde_sr0_mt_p_lock_v2 import (
    LOCK_PROPOSAL_READY as V2_LOCK_READY,
    LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE as V2_LOCK_UNAVAILABLE,
    ROOT_QUERY_UNMAPPABLE as V2_ROOT_QUERY_UNMAPPABLE,
    ROOT_READY as V2_ROOT_READY,
    ROOT_REFERENCE_MISSING as V2_ROOT_REFERENCE_MISSING,
    CandidatePLockV2,
    candidate_p_lock_v2_from_record,
    canonical_sha256,
    tensor_sha256 as v2_tensor_sha256,
)
from .dino_rcde_sr0_mt_p_natural_adapter_v2 import (
    ERASED_FIELDS,
    SCHEMA_VERSION as NATURAL_V2_SCHEMA,
    validate_natural_p_lock_v2,
)
from .dino_rcde_sr0_mt_p_natural_source_v1 import candidate_key_v1
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    LOCK_H0 as V1_LOCK_H0,
    LOCK_READY as V1_LOCK_READY,
    ROOT_H0 as V1_ROOT_H0,
    ROOT_READY as V1_ROOT_READY,
    tensor_sha256 as v1_p_tensor_sha256,
)
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    CandidatePLockV1,
    SealedDirectionPLockV1,
    SealedRootDinoScopeV1,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1_20260821"

_QUERY_FIELDS = frozenset(
    {
        "query_id",
        "historical_query_ordinal",
        "execution_ordinal",
        "query_source_image_sha256",
        "outer_fold",
        "inner_heldout_fold",
        "fit_id",
        "crossfit_role",
        "track",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_position",
        "candidate_key",
        "candidate_physical_row",
        "candidate_native_content_sha256",
        "candidate_reference_source_sha256",
    }
)
_SELECTED_FIELDS = frozenset(
    {
        "lock_state",
        "selected_bank_ordinal",
        "selected_bank_row_sha256",
        "selected_core_row_sha256",
        "selected_radius",
        "selected_root_ordinals",
    }
)
_GEOMETRY_FIELDS = frozenset(
    {
        "geometry_sha256",
        "grid_shape",
        "source_valid_mask_sha256",
        "mask_sha256",
        "area",
        "row_span",
        "col_span",
        "component_count",
        "border_touch",
    }
)
_FIXED_DENOMINATOR_FIELDS = frozenset(
    {
        "complete_root_coverage_used",
        "missing_root_contribution",
        "available_root_renormalization",
        "available_patch_renormalization",
        "outside_union_evidence",
        "score_sign_used_as_availability",
        "coverage",
        "coverage_sha256",
        "query_union",
        "query_union_sha256",
    }
)


class NaturalPLockV2ThreeArmAdapterError(ValueError):
    """Natural/Core V2 cannot be represented exactly by the V1 runtime."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise NaturalPLockV2ThreeArmAdapterError(message)


def _sha(value: object, *, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA256",
    )
    return value


def _grid(value: object, *, name: str) -> tuple[int, int]:
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(item) is int and item > 0 for item in value),
        f"{name} grid drift",
    )
    return int(value[0]), int(value[1])  # type: ignore[index]


def _exact_mapping(
    value: object, expected: frozenset[str], *, name: str
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == expected, f"{name} field-set drift")
    return value  # type: ignore[return-value]


def _canonical_mask(value: object, shape: tuple[int, int], *, name: str) -> torch.Tensor:
    mask = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous().flatten()
    _require(mask.shape == (shape[0] * shape[1],), f"{name} mask/grid drift")
    return mask


@dataclass(frozen=True)
class GeometryNamespaceBindingV1:
    """Bridge two intentionally distinct geometry-hash namespaces.

    ``canonical_geometry_sha256`` is the canonical full600/V2 geometry hash.
    ``cache_geometry_record_sha256`` is the DINO token-cache geometry receipt
    consumed by the historical V1 token field.  They are never substituted for
    one another.
    """

    source_image_sha256: str
    grid_shape: tuple[int, int]
    canonical_geometry_sha256: str
    cache_geometry_record_sha256: str

    def __post_init__(self) -> None:
        _sha(self.source_image_sha256, name="geometry namespace source")
        object.__setattr__(self, "grid_shape", _grid(self.grid_shape, name="geometry namespace"))
        _sha(self.canonical_geometry_sha256, name="canonical geometry")
        _sha(self.cache_geometry_record_sha256, name="cache geometry record")


@dataclass(frozen=True)
class TensorHashNamespaceReceiptV1:
    """The same tensor sealed independently under V2 and V1 hash conventions."""

    v2_tensor_sha256: str
    v1_p_tensor_sha256: str

    def __post_init__(self) -> None:
        _sha(self.v2_tensor_sha256, name="V2 tensor")
        _sha(self.v1_p_tensor_sha256, name="V1 P tensor")


def tensor_hash_namespace_receipt(value: torch.Tensor) -> TensorHashNamespaceReceiptV1:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return TensorHashNamespaceReceiptV1(
        v2_tensor_sha256=v2_tensor_sha256(tensor),
        v1_p_tensor_sha256=v1_p_tensor_sha256(tensor),
    )


@dataclass(frozen=True)
class HeadSelectionSpecV1:
    outer_fold: int
    source_fold: int
    fit_id: str
    crossfit_role: str
    inner_heldout_fold: int | None

    def __post_init__(self) -> None:
        _require(self.outer_fold in (1, 2, 3, 4), "head outer fold drift")
        _require(self.source_fold in (1, 2, 3, 4), "head source fold drift")
        _require(isinstance(self.fit_id, str) and bool(self.fit_id), "head fit ID absent")
        _require(
            self.crossfit_role in (
                "INNER_HELDOUT_DEPLOYMENT",
                "OUTER_HELDOUT_DEPLOYMENT",
            ),
            "head crossfit role drift",
        )


def inner_oof_head_spec(*, outer_fold: int, source_fold: int) -> HeadSelectionSpecV1:
    _require(outer_fold != source_fold, "inner-OOF source fold must differ from outer fold")
    return HeadSelectionSpecV1(
        outer_fold=outer_fold,
        source_fold=source_fold,
        fit_id=f"P_OUTER{outer_fold}_INNER{source_fold}_FIT",
        crossfit_role="INNER_HELDOUT_DEPLOYMENT",
        inner_heldout_fold=source_fold,
    )


def outer_refit_head_spec(*, source_fold: int) -> HeadSelectionSpecV1:
    return HeadSelectionSpecV1(
        outer_fold=source_fold,
        source_fold=source_fold,
        fit_id=f"P_OUTER{source_fold}_OUTER_REFIT",
        crossfit_role="OUTER_HELDOUT_DEPLOYMENT",
        inner_heldout_fold=None,
    )


def _validate_natural_record(value: Mapping[str, Any]) -> CandidatePLockV2:
    """Run the registered validators plus stricter nested target-free fields."""

    try:
        validate_natural_p_lock_v2(value)
        core = candidate_p_lock_v2_from_record(value["core_lock_record"])
    except Exception as error:
        raise NaturalPLockV2ThreeArmAdapterError(
            f"Natural/Core V2 validation failed: {error}"
        ) from error
    _require(value.get("schema_version") == NATURAL_V2_SCHEMA, "Natural V2 schema drift")
    _exact_mapping(value.get("query"), _QUERY_FIELDS, name="Natural V2 query")
    candidate = _exact_mapping(
        value.get("candidate"), _CANDIDATE_FIELDS, name="Natural V2 candidate"
    )
    _sha(
        candidate["candidate_native_content_sha256"],
        name="candidate directional source population",
    )
    _sha(
        candidate["candidate_reference_source_sha256"],
        name="candidate reference source",
    )
    _require(
        candidate["candidate_key"] == core.candidate_key
        and candidate["candidate_physical_row"] == core.candidate_physical_row
        and candidate["candidate_reference_source_sha256"]
        == core.geometry_signature.reference_source_image_sha256,
        "candidate wrapper/core binding drift",
    )
    _exact_mapping(value.get("selected"), _SELECTED_FIELDS, name="Natural V2 selected")
    _exact_mapping(value.get("query_geometry"), _GEOMETRY_FIELDS, name="Natural V2 query geometry")
    _exact_mapping(
        value.get("reference_geometry"),
        _GEOMETRY_FIELDS,
        name="Natural V2 reference geometry",
    )
    _exact_mapping(
        value.get("fixed_denominator"),
        _FIXED_DENOMINATOR_FIELDS,
        name="Natural V2 fixed denominator",
    )
    _require(value.get("erased_fields") == list(ERASED_FIELDS), "erased-field receipt drift")
    return core


def select_natural_v2_head_records(
    records: Sequence[Mapping[str, Any]],
    *,
    spec: HeadSelectionSpecV1,
    expected_candidate_count: int,
) -> tuple[Mapping[str, Any], ...]:
    """Select exactly one cross-fit head and close its candidate/direction axis."""

    _require(
        type(expected_candidate_count) is int and expected_candidate_count >= 2,
        "expected candidate count drift",
    )
    selected: list[Mapping[str, Any]] = []
    for value in records:
        _validate_natural_record(value)
        query = value["query"]
        if (
            query["fit_id"] == spec.fit_id
            and query["crossfit_role"] == spec.crossfit_role
            and query["outer_fold"] == spec.outer_fold
            and query["inner_heldout_fold"] == spec.inner_heldout_fold
        ):
            selected.append(value)
    _require(
        len(selected) == expected_candidate_count * len(FIXED_DIRECTIONS),
        "selected head is not candidate_count x two directions",
    )
    query_keys = {
        (
            item["query"]["query_id"],
            item["query"]["historical_query_ordinal"],
            item["query"]["execution_ordinal"],
            item["query"]["query_source_image_sha256"],
            item["query"]["track"],
        )
        for item in selected
    }
    _require(len(query_keys) == 1, "selected head mixes query provenance")
    checkpoint_keys = {
        (item["p_checkpoint_sha256"], item["p_training_manifest_sha256"])
        for item in selected
    }
    _require(len(checkpoint_keys) == 1, "selected head mixes P checkpoint lineage")
    by_position: dict[int, dict[str, Mapping[str, Any]]] = {}
    candidate_receipts: dict[int, tuple[object, ...]] = {}
    directional_population_receipts: dict[int, dict[str, str]] = {}
    for item in selected:
        candidate = item["candidate"]
        position = candidate["candidate_position"]
        _require(type(position) is int and 0 <= position < expected_candidate_count, "candidate position drift")
        direction = item["direction"]
        _require(direction in FIXED_DIRECTIONS, "candidate direction drift")
        receipt = (
            candidate["candidate_key"],
            candidate["candidate_physical_row"],
            candidate["candidate_reference_source_sha256"],
        )
        previous = candidate_receipts.setdefault(position, receipt)
        _require(previous == receipt, "candidate binding differs across directions")
        population_bucket = directional_population_receipts.setdefault(position, {})
        _require(
            direction not in population_bucket,
            "duplicate candidate directional population receipt",
        )
        population_bucket[direction] = candidate[
            "candidate_native_content_sha256"
        ]
        bucket = by_position.setdefault(position, {})
        _require(direction not in bucket, "duplicate candidate direction record")
        bucket[direction] = item
    _require(set(by_position) == set(range(expected_candidate_count)), "candidate position population drift")
    rows = [int(candidate_receipts[position][1]) for position in range(expected_candidate_count)]
    _require(rows == sorted(rows) and len(set(rows)) == len(rows), "candidate physical-row axis drift")
    _require(
        all(set(by_position[position]) == set(FIXED_DIRECTIONS) for position in by_position),
        "candidate direction population drift",
    )
    _require(
        all(
            set(directional_population_receipts[position])
            == set(FIXED_DIRECTIONS)
            and len(set(directional_population_receipts[position].values()))
            == len(FIXED_DIRECTIONS)
            for position in directional_population_receipts
        ),
        "candidate directional source populations are duplicated or incomplete",
    )
    return tuple(
        by_position[position][direction]
        for position in range(expected_candidate_count)
        for direction in FIXED_DIRECTIONS
    )


@dataclass(frozen=True)
class DirectionProjectionReceiptV1:
    direction: str
    natural_record_sha256: str
    core_record_sha256: str
    core_geometry_signature_sha256: str
    query_geometry_namespace: GeometryNamespaceBindingV1
    reference_geometry_namespace: GeometryNamespaceBindingV1
    query_valid_hashes: TensorHashNamespaceReceiptV1
    reference_valid_hashes: TensorHashNamespaceReceiptV1
    query_union_hashes: TensorHashNamespaceReceiptV1
    root_mask_hashes: tuple[Mapping[str, object], ...]
    logical_sha256: str

    def __post_init__(self) -> None:
        _require(self.direction in FIXED_DIRECTIONS, "projection receipt direction drift")
        _sha(self.natural_record_sha256, name="projection Natural record")
        _sha(self.core_record_sha256, name="projection Core record")
        _sha(
            self.core_geometry_signature_sha256,
            name="projection Core geometry signature",
        )
        _sha(self.logical_sha256, name="projection receipt logical")
        _require(
            self.logical_sha256 == canonical_sha256(self.payload(include_hash=False)),
            "projection receipt logical hash drift",
        )

    def payload(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "direction": self.direction,
            "natural_record_sha256": self.natural_record_sha256,
            "core_record_sha256": self.core_record_sha256,
            "core_geometry_signature_sha256": self.core_geometry_signature_sha256,
            "query_canonical_geometry_sha256": self.query_geometry_namespace.canonical_geometry_sha256,
            "query_cache_geometry_record_sha256": self.query_geometry_namespace.cache_geometry_record_sha256,
            "reference_canonical_geometry_sha256": self.reference_geometry_namespace.canonical_geometry_sha256,
            "reference_cache_geometry_record_sha256": self.reference_geometry_namespace.cache_geometry_record_sha256,
            "query_valid_hashes": dict(self.query_valid_hashes.__dict__),
            "reference_valid_hashes": dict(self.reference_valid_hashes.__dict__),
            "query_union_hashes": dict(self.query_union_hashes.__dict__),
            "root_mask_hashes": [dict(item) for item in self.root_mask_hashes],
        }
        if include_hash:
            value["logical_sha256"] = self.logical_sha256
        return value


@dataclass(frozen=True)
class CandidateRuntimeProjectionV1:
    runtime_lock: CandidatePLockV1
    direction_receipts: Mapping[str, DirectionProjectionReceiptV1]

    def __post_init__(self) -> None:
        receipts = dict(self.direction_receipts)
        _require(set(receipts) == set(FIXED_DIRECTIONS), "projection receipt directions drift")
        object.__setattr__(self, "direction_receipts", MappingProxyType(receipts))


def _geometry_bridge(
    *,
    role: str,
    natural_geometry: Mapping[str, Any],
    core_geometry_sha256: str,
    valid_mask: torch.Tensor,
    source_image_sha256: str,
    runtime_geometry_sha256: str,
    runtime_grid_shape: tuple[int, int],
    binding: GeometryNamespaceBindingV1,
) -> tuple[GeometryNamespaceBindingV1, TensorHashNamespaceReceiptV1]:
    hashes = tensor_hash_namespace_receipt(valid_mask)
    _require(binding.source_image_sha256 == source_image_sha256, f"{role} geometry source drift")
    _require(binding.grid_shape == runtime_grid_shape, f"{role} geometry grid drift")
    _require(
        binding.canonical_geometry_sha256
        == natural_geometry["geometry_sha256"]
        == core_geometry_sha256,
        f"{role} canonical geometry namespace drift",
    )
    _require(
        binding.cache_geometry_record_sha256 == runtime_geometry_sha256,
        f"{role} cache geometry namespace drift",
    )
    _require(_grid(natural_geometry["grid_shape"], name=f"{role} natural geometry") == runtime_grid_shape, f"{role} natural/runtime grid drift")
    _require(natural_geometry["mask_sha256"] == hashes.v2_tensor_sha256, f"{role} valid-mask V2 hash drift")
    _sha(natural_geometry["source_valid_mask_sha256"], name=f"{role} source valid mask")
    return binding, hashes


def project_natural_v2_candidate_to_v1_runtime(
    *,
    query: QueryTokenFieldV1,
    candidate: CandidateReferenceFieldV1,
    direction_records: Mapping[str, Mapping[str, Any]],
    spec: HeadSelectionSpecV1,
    query_geometry_binding: GeometryNamespaceBindingV1,
    reference_geometry_binding: GeometryNamespaceBindingV1,
) -> CandidateRuntimeProjectionV1:
    """Build one exact V1 runtime view from two Natural/Core V2 directions."""

    records = dict(direction_records)
    _require(set(records) == set(FIXED_DIRECTIONS), "candidate projection needs exactly two directions")
    expected_key = candidate_key_v1(
        physical_row=candidate.physical_gallery_row,
        source_image_sha256=candidate.source_image_sha256,
    )
    _require(candidate.candidate_key == expected_key, "runtime candidate key/source drift")
    projected: dict[str, SealedDirectionPLockV1] = {}
    receipts: dict[str, DirectionProjectionReceiptV1] = {}
    for direction in FIXED_DIRECTIONS:
        value = records[direction]
        core = _validate_natural_record(value)
        query_address = value["query"]
        candidate_address = value["candidate"]
        selected = value["selected"]
        fixed = value["fixed_denominator"]
        _require(value["direction"] == direction, "direction key/record drift")
        _require(
            query_address["fit_id"] == spec.fit_id
            and query_address["crossfit_role"] == spec.crossfit_role
            and query_address["outer_fold"] == spec.outer_fold
            and query_address["inner_heldout_fold"] == spec.inner_heldout_fold,
            "projection selected the wrong cross-fit head",
        )
        _require(query_address["query_source_image_sha256"] == query.source_image_sha256, "query token/Natural V2 source drift")
        _require(
            candidate_address["candidate_key"] == candidate.candidate_key
            and candidate_address["candidate_physical_row"] == candidate.physical_gallery_row
            and candidate_address["candidate_reference_source_sha256"] == candidate.source_image_sha256,
            "candidate token/Natural V2 address drift",
        )
        _require(
            core.candidate_key == candidate.candidate_key
            and core.candidate_physical_row == candidate.physical_gallery_row,
            "candidate token/Core V2 address drift",
        )
        _require(torch.equal(core.query_valid_mask, query.valid_patch_mask), "query valid-mask membership drift")
        _require(torch.equal(core.reference_valid_mask, candidate.valid_patch_mask), "reference valid-mask membership drift")
        q_binding, q_valid_hashes = _geometry_bridge(
            role="query",
            natural_geometry=value["query_geometry"],
            core_geometry_sha256=core.geometry_signature.query_geometry_sha256,
            valid_mask=query.valid_patch_mask,
            source_image_sha256=query.source_image_sha256,
            runtime_geometry_sha256=query.geometry_record_sha256,
            runtime_grid_shape=query.grid_shape,
            binding=query_geometry_binding,
        )
        r_binding, r_valid_hashes = _geometry_bridge(
            role="reference",
            natural_geometry=value["reference_geometry"],
            core_geometry_sha256=core.geometry_signature.reference_geometry_sha256,
            valid_mask=candidate.valid_patch_mask,
            source_image_sha256=candidate.source_image_sha256,
            runtime_geometry_sha256=candidate.geometry_record_sha256,
            runtime_grid_shape=candidate.grid_shape,
            binding=reference_geometry_binding,
        )
        _require(
            core.geometry_signature.query_source_image_sha256 == query.source_image_sha256
            and core.geometry_signature.reference_source_image_sha256 == candidate.source_image_sha256,
            "Core V2 geometry/source provenance drift",
        )
        all_roots: dict[int, SealedRootDinoScopeV1] = {}
        root_hash_rows: list[Mapping[str, object]] = []
        for root in core.roots:
            if root.status == V2_ROOT_REFERENCE_MISSING:
                raise NaturalPLockV2ThreeArmAdapterError(
                    "ROOT_REFERENCE_MISSING cannot be projected to V1 without erasing its query mask"
                )
            if root.status == V2_ROOT_READY:
                binding_status = V1_ROOT_READY
            elif root.status == V2_ROOT_QUERY_UNMAPPABLE:
                binding_status = V1_ROOT_H0
            else:  # guarded by Core V2 validation, retained as fail-closed defense
                raise NaturalPLockV2ThreeArmAdapterError("unknown Core V2 root state")
            query_hashes = tensor_hash_namespace_receipt(root.query_mask)
            reference_hashes = tensor_hash_namespace_receipt(root.reference_mask)
            all_roots[root.root_ordinal] = SealedRootDinoScopeV1(
                root_ordinal=root.root_ordinal,
                action_key_sha256=root.action_key_sha256,
                query_mask=root.query_mask,
                reference_mask=root.reference_mask,
                query_grid_shape=query.grid_shape,
                reference_grid_shape=candidate.grid_shape,
                query_mask_p_sha256=query_hashes.v1_p_tensor_sha256,
                reference_mask_p_sha256=reference_hashes.v1_p_tensor_sha256,
                query_geometry_sha256=query.geometry_record_sha256,
                reference_geometry_sha256=candidate.geometry_record_sha256,
                binding_status=binding_status,
            )
            root_hash_rows.append(
                MappingProxyType(
                    {
                        "root_ordinal": root.root_ordinal,
                        "query_v2_tensor_sha256": query_hashes.v2_tensor_sha256,
                        "query_v1_p_tensor_sha256": query_hashes.v1_p_tensor_sha256,
                        "reference_v2_tensor_sha256": reference_hashes.v2_tensor_sha256,
                        "reference_v1_p_tensor_sha256": reference_hashes.v1_p_tensor_sha256,
                    }
                )
            )
        union = _canonical_mask(fixed["query_union"], query.grid_shape, name="fixed query union")
        _require(torch.equal(union, core.query_union_mask), "fixed/core query-union drift")
        union_hashes = tensor_hash_namespace_receipt(union)
        _require(fixed["query_union_sha256"] == union_hashes.v2_tensor_sha256, "query-union V2 hash drift")
        if core.lock_state == V2_LOCK_READY:
            v1_status = V1_LOCK_READY
            _require(len(core.selected_root_ordinals) >= 2, "V1 READY projection requires a multi-root row")
            _require(all(all_roots[root].binding_status == V1_ROOT_READY for root in core.selected_root_ordinals), "V1 READY selected row contains an unrepresentable root")
            selected_bank_ordinal = selected["selected_bank_ordinal"]
            selected_row_sha256 = selected["selected_bank_row_sha256"]
        elif core.lock_state == V2_LOCK_UNAVAILABLE:
            v1_status = V1_LOCK_H0
            _require(not bool(union.any()), "structural-unavailable lock has a nonempty union")
            selected_bank_ordinal = None
            selected_row_sha256 = None
        else:
            raise NaturalPLockV2ThreeArmAdapterError("unknown Core V2 lock state")
        natural_record_sha = _sha(value["record_sha256"], name="Natural V2 record")
        runtime_direction = SealedDirectionPLockV1(
            candidate_key=candidate.candidate_key,
            candidate_physical_row=candidate.physical_gallery_row,
            query_source_image_sha256=query.source_image_sha256,
            candidate_reference_source_sha256=candidate.source_image_sha256,
            direction=direction,
            status=v1_status,
            selected_bank_ordinal=selected_bank_ordinal,
            selected_row_sha256=selected_row_sha256,
            query_union_mask=union,
            ordered_root_ordinals=core.selected_root_ordinals,
            all_roots=all_roots,
            query_grid_shape=query.grid_shape,
            reference_grid_shape=candidate.grid_shape,
            query_geometry_sha256=query.geometry_record_sha256,
            reference_geometry_sha256=candidate.geometry_record_sha256,
            p_lock_record_sha256=natural_record_sha,
        )
        core_record_sha = _sha(value["core_lock_record"]["record_sha256"], name="Core V2 record")
        receipt_payload = {
            "schema_version": SCHEMA_VERSION,
            "direction": direction,
            "natural_record_sha256": natural_record_sha,
            "core_record_sha256": core_record_sha,
            "core_geometry_signature_sha256": core.geometry_signature.signature_sha256,
            "query_canonical_geometry_sha256": q_binding.canonical_geometry_sha256,
            "query_cache_geometry_record_sha256": q_binding.cache_geometry_record_sha256,
            "reference_canonical_geometry_sha256": r_binding.canonical_geometry_sha256,
            "reference_cache_geometry_record_sha256": r_binding.cache_geometry_record_sha256,
            "query_valid_hashes": q_valid_hashes.__dict__,
            "reference_valid_hashes": r_valid_hashes.__dict__,
            "query_union_hashes": union_hashes.__dict__,
            "root_mask_hashes": [dict(item) for item in root_hash_rows],
        }
        receipts[direction] = DirectionProjectionReceiptV1(
            direction=direction,
            natural_record_sha256=natural_record_sha,
            core_record_sha256=core_record_sha,
            core_geometry_signature_sha256=core.geometry_signature.signature_sha256,
            query_geometry_namespace=q_binding,
            reference_geometry_namespace=r_binding,
            query_valid_hashes=q_valid_hashes,
            reference_valid_hashes=r_valid_hashes,
            query_union_hashes=union_hashes,
            root_mask_hashes=tuple(root_hash_rows),
            logical_sha256=canonical_sha256(receipt_payload),
        )
        projected[direction] = runtime_direction
    return CandidateRuntimeProjectionV1(
        runtime_lock=CandidatePLockV1(candidate=candidate, direction_locks=projected),
        direction_receipts=receipts,
    )


__all__ = [
    "SCHEMA_VERSION",
    "NaturalPLockV2ThreeArmAdapterError",
    "GeometryNamespaceBindingV1",
    "TensorHashNamespaceReceiptV1",
    "HeadSelectionSpecV1",
    "DirectionProjectionReceiptV1",
    "CandidateRuntimeProjectionV1",
    "tensor_hash_namespace_receipt",
    "inner_oof_head_spec",
    "outer_refit_head_spec",
    "select_natural_v2_head_records",
    "project_natural_v2_candidate_to_v1_runtime",
]
