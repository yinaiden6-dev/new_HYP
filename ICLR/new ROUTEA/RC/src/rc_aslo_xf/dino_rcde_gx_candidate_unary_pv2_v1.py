"""Natural/Core P-V2 geometric energies with true complete-root denominators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    QueryTokenFieldV1,
)
from .dino_rcde_gx_candidate_unary_v2 import (
    C_BIND,
    N_REGION_RESAMPLE,
    P_COORD,
    REAL,
    T_SHAPE_MATCHED_ROOT_ASSIGNMENT,
    T_ROOT_ASSIGNMENT_POPULATION,
    CandidateGeometricEnergyV1,
    RootDecodeSpecV1,
    _global_candidate_permutation,
    _matched_topology_derangement,
    _zero,
)
from .geometry_hypothesis_v1 import connected_components_4
from .dino_rcde_sr0_mt_controls_v1 import (
    CDinoCandidateBindingV1,
    transport_reference_mask_positive_overlap,
)
from .dino_rcde_sr0_mt_p_lock_v2 import (
    CandidatePLockV2,
    ROOT_READY,
    ROOT_REFERENCE_MISSING,
    aggregate_fixed_denominator_v2,
    candidate_p_lock_v2_from_record,
    tensor_sha256,
)
from .dino_rcde_sr0_mt_p_natural_adapter_v2 import (
    validate_natural_p_lock_v2,
)
from .dino_rcde_sr0_mt_v_runtime_v1 import (
    hash_parts,
    tensor_sha256 as v1_tensor_sha256,
)


SCHEMA_VERSION = "rc_dino_rcde_gx_candidate_unary_pv2_v1_20260824"


class GXPV2Error(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise GXPV2Error(message)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class NaturalCandidatePLockV2:
    candidate: CandidateReferenceFieldV1
    natural_record_by_direction: Mapping[str, Mapping[str, Any]]
    core_lock_by_direction: Mapping[str, CandidatePLockV2]

    def __post_init__(self) -> None:
        require(
            set(self.natural_record_by_direction) == set(FIXED_DIRECTIONS)
            and set(self.core_lock_by_direction) == set(FIXED_DIRECTIONS),
            "natural P-V2 direction axis drift",
        )
        for direction in FIXED_DIRECTIONS:
            record = self.natural_record_by_direction[direction]
            core = self.core_lock_by_direction[direction]
            require(
                record["direction"] == direction
                and record["candidate"]["candidate_key"]
                == core.candidate_key
                == self.candidate.candidate_key
                and int(record["candidate"]["candidate_physical_row"])
                == core.candidate_physical_row
                == self.candidate.physical_gallery_row
                and torch.equal(
                    core.reference_valid_mask,
                    self.candidate.valid_patch_mask.cpu(),
                ),
                "natural P-V2 candidate/core drift",
            )
        object.__setattr__(
            self,
            "natural_record_by_direction",
            MappingProxyType(dict(self.natural_record_by_direction)),
        )
        object.__setattr__(
            self,
            "core_lock_by_direction",
            MappingProxyType(dict(self.core_lock_by_direction)),
        )


@dataclass(frozen=True)
class ConnectedTranslationV1:
    role: str
    source_mask: torch.Tensor
    translated_mask: torch.Tensor
    source_indices: tuple[int, ...]
    destination_indices: tuple[int, ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        source = torch.as_tensor(self.source_mask, dtype=torch.bool).cpu().flatten()
        translated = torch.as_tensor(
            self.translated_mask, dtype=torch.bool
        ).cpu().flatten()
        source_indices = tuple(int(value) for value in self.source_indices)
        destination_indices = tuple(int(value) for value in self.destination_indices)
        require(
            self.role in {"QUERY", "REFERENCE"}
            and source.shape == translated.shape
            and len(source_indices) == len(destination_indices) == int(source.sum())
            and int(translated.sum()) == len(destination_indices)
            and len(set(source_indices)) == len(source_indices)
            and len(set(destination_indices)) == len(destination_indices)
            and all(source[index] for index in source_indices)
            and all(translated[index] for index in destination_indices),
            "connected translation bijection drift",
        )
        object.__setattr__(self, "source_mask", source)
        object.__setattr__(self, "translated_mask", translated)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "destination_indices", destination_indices)
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))


def _mask_cells(mask: torch.Tensor, grid: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    width = int(grid[1])
    value = torch.as_tensor(mask, dtype=torch.bool).cpu().flatten()
    require(value.shape == (math.prod(grid),), "translation mask/grid drift")
    return tuple(
        (int(index) // width, int(index) % width)
        for index in torch.nonzero(value, as_tuple=False).flatten().tolist()
    )


def _mask_from_cells(
    cells: Sequence[tuple[int, int]], grid: tuple[int, int]
) -> torch.Tensor:
    output = torch.zeros(math.prod(grid), dtype=torch.bool)
    width = int(grid[1])
    for row, column in cells:
        output[int(row) * width + int(column)] = True
    return output


def _border_distance(mask: torch.Tensor, grid: tuple[int, int]) -> int:
    cells = _mask_cells(mask, grid)
    require(bool(cells), "translation source mask is empty")
    height, width = grid
    return min(
        min(row, column, height - 1 - row, width - 1 - column)
        for row, column in cells
    )


def _connected_translation(
    source_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    grid: tuple[int, int],
    *,
    role: str,
    namespace: str,
    key: str,
) -> tuple[ConnectedTranslationV1 | None, dict[str, Any]]:
    """Enumerate once, hash-order once, and seal an exact translation map."""

    source = torch.as_tensor(source_mask, dtype=torch.bool).cpu().flatten()
    valid = torch.as_tensor(valid_mask, dtype=torch.bool).cpu().flatten()
    require(
        role in {"QUERY", "REFERENCE"}
        and source.shape == valid.shape == (math.prod(grid),)
        and bool(source.any())
        and not bool((source & ~valid).any()),
        "connected translation source/valid drift",
    )
    _, source_diagnostic = connected_components_4(source, grid)
    require(
        source_diagnostic.component_count == 1,
        "connected translation requires one source 4CC",
    )
    cells = _mask_cells(source, grid)
    rows = [row for row, _ in cells]
    columns = [column for _, column in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(columns), max(columns)
    source_border = _border_distance(source, grid)
    source_flat = tuple(row * grid[1] + column for row, column in cells)
    normalized_shape = sorted((row - r0, column - c0) for row, column in cells)
    normalized_shape_sha256 = _canonical_sha256(normalized_shape)
    population = []
    for delta_row in range(-r0, grid[0] - r1):
        for delta_column in range(-c0, grid[1] - c1):
            if delta_row == 0 and delta_column == 0:
                continue
            translated_cells = tuple(
                (row + delta_row, column + delta_column) for row, column in cells
            )
            translated = _mask_from_cells(translated_cells, grid)
            if bool((translated & ~valid).any()) or (
                _border_distance(translated, grid) != source_border
            ):
                continue
            destination_flat = tuple(
                row * grid[1] + column for row, column in translated_cells
            )
            source_to_destination = [
                {"source": int(src), "destination": int(dst)}
                for src, dst in zip(source_flat, destination_flat, strict=True)
            ]
            population.append(
                {
                    "delta_row": int(delta_row),
                    "delta_column": int(delta_column),
                    "translated_mask_sha256": tensor_sha256(
                        translated.to(torch.uint8)
                    ),
                    "selector_mask_sha256": v1_tensor_sha256(
                        translated.to(torch.uint8)
                    ),
                    "source_to_destination_sha256": _canonical_sha256(
                        source_to_destination
                    ),
                    "translated_mask": translated,
                    "destination_flat": destination_flat,
                }
            )
    population.sort(
        key=lambda row: hash_parts(
            namespace,
            key,
            row["selector_mask_sha256"],
        )
    )
    population_logical = [
        {
            item: row[item]
            for item in (
                "delta_row",
                "delta_column",
                "translated_mask_sha256",
                "selector_mask_sha256",
                "source_to_destination_sha256",
            )
        }
        for row in population
    ]
    base_receipt = {
        "role": role,
        "namespace": namespace,
        "key": key,
        "grid_shape": list(grid),
        "source_mask_sha256": tensor_sha256(source.to(torch.uint8)),
        "valid_mask_sha256": tensor_sha256(valid.to(torch.uint8)),
        "area": int(source.sum()),
        "source_component_count": int(source_diagnostic.component_count),
        "source_border_distance": source_border,
        "normalized_shape_sha256": normalized_shape_sha256,
        "legal_translation_count": len(population),
        "legal_translation_population_sha256": _canonical_sha256(
            population_logical
        ),
        "enumeration_result_blind": True,
        "model_response_retry_count": 0,
    }
    if not population:
        return None, {
            **base_receipt,
            "eligible": False,
            "selected_population_ordinal": None,
            "ineligibility_reason": f"{role}_CONNECTED_TRANSLATION_UNAVAILABLE",
        }
    chosen = population[0]
    selected = chosen["translated_mask"]
    _, translated_diagnostic = connected_components_4(selected, grid)
    destination_flat = tuple(int(value) for value in chosen["destination_flat"])
    source_to_destination = [
        {"source": int(src), "destination": int(dst)}
        for src, dst in zip(source_flat, destination_flat, strict=True)
    ]
    destination_to_source = [
        {"destination": int(dst), "source": int(src)}
        for src, dst in zip(source_flat, destination_flat, strict=True)
    ]
    receipt = {
        **base_receipt,
        "eligible": True,
        "selected_population_ordinal": 0,
        "ineligibility_reason": None,
        "delta_row": chosen["delta_row"],
        "delta_column": chosen["delta_column"],
        "translated_mask_sha256": chosen["translated_mask_sha256"],
        "translated_border_distance": _border_distance(selected, grid),
        "translated_component_count": int(translated_diagnostic.component_count),
        "source_to_destination": source_to_destination,
        "destination_to_source": destination_to_source,
        "source_to_destination_sha256": _canonical_sha256(
            source_to_destination
        ),
        "destination_to_source_sha256": _canonical_sha256(
            destination_to_source
        ),
        "non_identity_translation": True,
        "exact_area_preserved": int(selected.sum()) == int(source.sum()),
        "normalized_shape_preserved": True,
        "component_count_preserved": (
            translated_diagnostic.component_count
            == source_diagnostic.component_count
            == 1
        ),
        "border_distance_stratum_preserved": (
            _border_distance(selected, grid) == source_border
        ),
        "exact_translation_bijection": True,
    }
    return (
        ConnectedTranslationV1(
            role=role,
            source_mask=source,
            translated_mask=selected,
            source_indices=source_flat,
            destination_indices=destination_flat,
            receipt=receipt,
        ),
        receipt,
    )


def _pull_back_query_contributions(
    contributions: torch.Tensor,
    translation: ConnectedTranslationV1,
) -> torch.Tensor:
    require(translation.role == "QUERY", "query pullback received reference map")
    values = torch.as_tensor(contributions)
    require(
        values.ndim == 1 and values.numel() == translation.source_mask.numel(),
        "query pullback contribution shape drift",
    )
    source = torch.tensor(
        translation.source_indices, dtype=torch.long, device=values.device
    )
    destination = torch.tensor(
        translation.destination_indices, dtype=torch.long, device=values.device
    )
    translated_mask = translation.translated_mask.to(values.device)
    require(
        not bool(values.detach()[~translated_mask].ne(0).any()),
        "translated query evidence escaped translated mask",
    )
    output = torch.zeros_like(values)
    output[source] = values[destination]
    source_mask = translation.source_mask.to(values.device)
    require(
        not bool(output.detach()[~source_mask].ne(0).any()),
        "query pullback evidence escaped original root slot",
    )
    return output


def natural_candidate_lock_v2(
    candidate: CandidateReferenceFieldV1,
    direction_records: Mapping[str, Mapping[str, Any]],
) -> NaturalCandidatePLockV2:
    records = dict(direction_records)
    require(set(records) == set(FIXED_DIRECTIONS), "natural direction records absent")
    cores = {}
    for direction, record in records.items():
        validate_natural_p_lock_v2(record)
        cores[direction] = candidate_p_lock_v2_from_record(record["core_lock_record"])
    return NaturalCandidatePLockV2(candidate, records, cores)


def _root_patch_evidence(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    spec: RootDecodeSpecV1,
    *,
    streaming_chunk_size: int | None,
) -> torch.Tensor:
    qmask = torch.as_tensor(
        spec.query_mask, dtype=torch.bool, device=query.layers.device
    ).flatten()
    rmask = torch.as_tensor(
        spec.reference_mask,
        dtype=torch.bool,
        device=spec.candidate.layers.device,
    ).flatten()
    evidence = model.decode_candidate(
        query.layers,
        spec.candidate.layers,
        qmask.reshape(query.grid_shape),
        rmask.reshape(spec.candidate.grid_shape),
        query.grid_shape,
        spec.candidate.grid_shape,
        streaming_chunk_size=streaming_chunk_size,
    )
    exact_zero = torch.zeros_like(evidence.relational)
    pair = model.compare_relational(evidence.relational, exact_zero, qmask)
    return pair.contributions * qmask.to(pair.contributions.dtype)


def _fixed_denominator_receipt(
    query: QueryTokenFieldV1,
    natural: Mapping[str, Any],
    core: CandidatePLockV2,
) -> dict[str, Any]:
    """Recompute the selected complete-root denominator from Core P-V2.

    This path is independent of branch eligibility and decoded READY roots.  A
    reference-missing selected root therefore remains in coverage even when a
    mandatory control cannot be materialized and the branch returns exact H0.
    """

    query_valid = torch.as_tensor(query.valid_patch_mask, dtype=torch.bool).cpu()
    require(
        torch.equal(core.query_valid_mask, query_valid),
        "natural P-V2 query valid mask drift",
    )
    coverage = torch.zeros_like(core.query_valid_mask, dtype=torch.int64)
    denominator_roots = []
    for root in core.selected_roots:
        if root.structurally_mapped:
            coverage += root.query_mask.to(torch.int64)
            denominator_roots.append(int(root.root_ordinal))
    union = coverage > 0

    fixed = natural.get("fixed_denominator")
    require(isinstance(fixed, Mapping), "natural P-V2 fixed denominator absent")
    sealed_coverage = torch.as_tensor(fixed.get("coverage"), dtype=torch.int64)
    sealed_union = torch.as_tensor(fixed.get("query_union"), dtype=torch.bool)
    coverage_sha256 = tensor_sha256(coverage)
    union_sha256 = tensor_sha256(union)
    require(
        sealed_coverage.shape == coverage.shape
        and sealed_union.shape == union.shape
        and torch.equal(sealed_coverage, coverage)
        and torch.equal(sealed_union, union)
        and fixed.get("coverage_sha256") == coverage_sha256
        and fixed.get("query_union_sha256") == union_sha256
        and fixed.get("complete_root_coverage_used") is True
        and fixed.get("missing_root_contribution") == "EXACT_ZERO"
        and fixed.get("available_root_renormalization") is False
        and fixed.get("available_patch_renormalization") is False
        and fixed.get("outside_union_evidence") == "EXACT_ZERO"
        and fixed.get("score_sign_used_as_availability") is False,
        "natural/Core P-V2 fixed denominator receipt drift",
    )
    selected = tuple(int(root) for root in core.selected_root_ordinals)
    ready = tuple(
        int(root.root_ordinal)
        for root in core.selected_roots
        if root.status == ROOT_READY
    )
    reference_missing = tuple(
        int(root.root_ordinal)
        for root in core.selected_roots
        if root.status == ROOT_REFERENCE_MISSING
    )
    return {
        "natural_record_sha256": natural["record_sha256"],
        "core_lock_record_sha256": natural["core_lock_record"]["record_sha256"]
        if isinstance(natural.get("core_lock_record"), Mapping)
        else None,
        "complete_root_table_sha256": natural["complete_root_table_sha256"],
        "selected_root_ordinals": list(selected),
        "denominator_root_ordinals": denominator_roots,
        "ready_root_ordinals": list(ready),
        "reference_missing_root_ordinals": list(reference_missing),
        # Kept for consumers of the initial repair draft.  A proposal-ready
        # P-V2 selected row cannot contain QUERY_UNMAPPABLE roots.
        "missing_or_unmappable_root_ordinals": list(reference_missing),
        "root_status_by_ordinal": {
            str(root.root_ordinal): root.status for root in core.selected_roots
        },
        "selected_root_count": len(selected),
        "denominator_root_count": len(denominator_roots),
        "ready_root_count": len(ready),
        "reference_missing_root_count": len(reference_missing),
        "coverage_sha256": coverage_sha256,
        "query_union_sha256": union_sha256,
        "coverage_numel": int(coverage.numel()),
        "coverage_sum": int(coverage.sum()),
        "coverage_max": int(coverage.max()) if coverage.numel() else 0,
        "query_union_area": int(union.sum()),
        "complete_root_coverage_used": True,
        "missing_root_contribution": "EXACT_ZERO",
        "available_root_renormalization": False,
        "available_patch_renormalization": False,
        "outside_union_evidence": "EXACT_ZERO",
        "score_sign_used_as_availability": False,
    }


def _direction_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    natural: Mapping[str, Any],
    core: CandidatePLockV2,
    specs_by_root: Mapping[int, RootDecodeSpecV1],
    query_pullback_by_root: Mapping[int, ConnectedTranslationV1] | None = None,
    *,
    streaming_chunk_size: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    denominator_receipt = _fixed_denominator_receipt(query, natural, core)
    query_pullbacks = dict(query_pullback_by_root or {})
    ready_by_ordinal = {
        int(root.root_ordinal): root
        for root in core.selected_roots
        if root.status == ROOT_READY
    }
    require(
        set(specs_by_root) == set(ready_by_ordinal),
        "READY root decode population drift",
    )
    require(
        not query_pullbacks or set(query_pullbacks) == set(ready_by_ordinal),
        "query pullback root population drift",
    )
    zero = query.layers.new_zeros(query.valid_patch_mask.numel()) + _zero(model)
    evidence = {root.root_ordinal: zero for root in core.roots}
    decoded = []
    decoded_records = []
    for root in core.selected_roots:
        if root.status != ROOT_READY:
            continue
        require(root.root_ordinal in specs_by_root, "READY root decode spec absent")
        spec = specs_by_root[root.root_ordinal]
        spec_query_mask = torch.as_tensor(
            spec.query_mask, dtype=torch.bool
        ).cpu().flatten()
        pullback = query_pullbacks.get(root.root_ordinal)
        if pullback is None:
            require(
                torch.equal(spec_query_mask, root.query_mask),
                "READY root decode query binding drift",
            )
        else:
            require(
                pullback.role == "QUERY"
                and torch.equal(pullback.source_mask, root.query_mask)
                and torch.equal(pullback.translated_mask, spec_query_mask),
                "translated query/pullback binding drift",
            )
        require(
            spec.root_ordinal == root.root_ordinal,
            "READY root decode ordinal drift",
        )
        translated_patch = _root_patch_evidence(
            model,
            query,
            spec,
            streaming_chunk_size=streaming_chunk_size,
        )
        patch = (
            translated_patch
            if pullback is None
            else _pull_back_query_contributions(translated_patch, pullback)
        )
        evidence[root.root_ordinal] = patch
        decoded.append(root.root_ordinal)
        record = {
            "root_ordinal": int(root.root_ordinal),
            "source_reference_root_ordinal": int(
                spec.source_reference_root_ordinal
            ),
            "candidate_tokens_sha256": spec.candidate.tokens_sha256,
            "query_mask_sha256": tensor_sha256(
                spec_query_mask.to(torch.uint8)
            ),
            "original_query_mask_sha256": tensor_sha256(
                root.query_mask.to(torch.uint8)
            ),
            "reference_mask_sha256": tensor_sha256(
                spec.reference_mask.to(torch.uint8)
            ),
            "translated_patch_evidence_sha256": tensor_sha256(
                translated_patch
            ),
            "fixed_denominator_patch_evidence_sha256": tensor_sha256(patch),
            "query_pullback_applied": pullback is not None,
            "query_pullback_mapping_sha256": None
            if pullback is None
            else pullback.receipt["destination_to_source_sha256"],
            "query_pullback_before_fixed_denominator": pullback is not None,
        }
        decoded_records.append(record)
    aggregate = aggregate_fixed_denominator_v2(core, evidence)
    require(
        denominator_receipt["coverage_sha256"]
        == tensor_sha256(aggregate.structural_coverage)
        and denominator_receipt["query_union_sha256"]
        == tensor_sha256(aggregate.query_union_mask),
        "natural/Core P-V2 fixed denominator receipt drift",
    )
    return aggregate.scalar_score, {
        **denominator_receipt,
        "decoded_root_ordinals": decoded,
        "decoded_root_records": decoded_records,
        "decoded_root_count": len(decoded),
    }


def _energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    lock: NaturalCandidatePLockV2,
    specs_by_direction: Mapping[str, Mapping[int, RootDecodeSpecV1]],
    *,
    branch: str,
    eligible: bool,
    streaming_chunk_size: int | None,
    query_pullbacks_by_direction: Mapping[
        str, Mapping[int, ConnectedTranslationV1]
    ]
    | None = None,
    extra_receipt: Mapping[str, object] | None = None,
) -> CandidateGeometricEnergyV1:
    if not eligible:
        zero = _zero(model)
        denominator_receipts = {
            direction: {
                **_fixed_denominator_receipt(
                    query,
                    lock.natural_record_by_direction[direction],
                    lock.core_lock_by_direction[direction],
                ),
                "decoded_root_ordinals": [],
                "decoded_root_records": [],
                "decoded_root_count": 0,
            }
            for direction in FIXED_DIRECTIONS
        }
        return CandidateGeometricEnergyV1(
            branch,
            zero,
            False,
            (zero, zero),
            0,
            {
                "schema_version": SCHEMA_VERSION,
                "branch": branch,
                "eligible": False,
                "directions": denominator_receipts,
                **(extra_receipt or {}),
            },
        )
    direction_values = []
    receipts = {}
    count = 0
    pullbacks = dict(query_pullbacks_by_direction or {})
    require(
        not pullbacks or set(pullbacks) == set(FIXED_DIRECTIONS),
        "query pullback direction axis drift",
    )
    for direction in FIXED_DIRECTIONS:
        value, receipt = _direction_energy(
            model,
            query,
            lock.natural_record_by_direction[direction],
            lock.core_lock_by_direction[direction],
            specs_by_direction[direction],
            pullbacks.get(direction),
            streaming_chunk_size=streaming_chunk_size,
        )
        direction_values.append(value)
        receipts[direction] = receipt
        count += len(receipt["decoded_root_ordinals"])
    value = 0.5 * (direction_values[0] + direction_values[1])
    return CandidateGeometricEnergyV1(
        branch,
        value,
        True,
        (direction_values[0], direction_values[1]),
        count,
        {
            "schema_version": SCHEMA_VERSION,
            "branch": branch,
            "eligible": True,
            "directions": receipts,
            **(extra_receipt or {}),
        },
    )


def _real_specs(lock: NaturalCandidatePLockV2):
    return {
        direction: {
            root.root_ordinal: RootDecodeSpecV1(
                root.root_ordinal,
                root.query_mask,
                root.reference_mask,
                lock.candidate,
                root.root_ordinal,
            )
            for root in lock.core_lock_by_direction[direction].selected_roots
            if root.status == ROOT_READY
        }
        for direction in FIXED_DIRECTIONS
    }


def real_energy_pv2(
    model,
    query,
    lock: NaturalCandidatePLockV2,
    *,
    streaming_chunk_size: int | None = 64,
):
    return _energy(
        model,
        query,
        lock,
        _real_specs(lock),
        branch=REAL,
        eligible=True,
        streaming_chunk_size=streaming_chunk_size,
    )


def c_bind_energy_pv2(
    model,
    query,
    lock: NaturalCandidatePLockV2,
    binding: CDinoCandidateBindingV1,
    *,
    streaming_chunk_size: int | None = 64,
):
    specs = {}
    transports = []
    eligible = True
    for direction in FIXED_DIRECTIONS:
        specs[direction] = {}
        natural_sha = lock.natural_record_by_direction[direction]["record_sha256"]
        for root in lock.core_lock_by_direction[direction].selected_roots:
            if root.status != ROOT_READY:
                continue
            mask, receipt = transport_reference_mask_positive_overlap(
                root.reference_mask,
                lock.candidate.grid_shape,
                binding.candidate.grid_shape,
                binding.candidate.valid_patch_mask,
                p_lock_record_sha256=natural_sha,
            )
            transports.append(receipt.logical_record())
            if not receipt.decoder_legal:
                eligible = False
                continue
            specs[direction][root.root_ordinal] = RootDecodeSpecV1(
                root.root_ordinal,
                root.query_mask,
                mask,
                binding.candidate,
                root.root_ordinal,
            )
    return _energy(
        model,
        query,
        lock,
        specs,
        branch=C_BIND,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={"mask_transports": transports},
    )


def p_coord_energy_pv2(
    model,
    query,
    lock: NaturalCandidatePLockV2,
    *,
    namespace: str,
    query_id: str,
    streaming_chunk_size: int | None = 64,
):
    candidate, permutation = _global_candidate_permutation(
        lock.candidate, namespace=namespace, query_id=query_id
    )
    require(
        permutation.get("candidate_key") == lock.candidate.candidate_key
        and permutation.get("input_tokens_sha256") == lock.candidate.tokens_sha256
        and permutation.get("output_tokens_sha256") == candidate.tokens_sha256
        and permutation.get("fixed_point_free") is True
        and permutation.get("spatially_non_affine") is True
        and permutation.get("content_multiset_preserved") is True
        and permutation.get("candidate_axis_preserved") is True
        and permutation.get("same_permutation_all_roots_and_directions") is True
        and permutation.get("token_resize_or_interpolation_count") == 0,
        "global P_COORD constructor receipt drift",
    )
    application_by_direction = {
        direction: {
            "selected_root_ordinals": list(
                lock.core_lock_by_direction[direction].selected_root_ordinals
            ),
            "ready_root_ordinals": [
                int(root.root_ordinal)
                for root in lock.core_lock_by_direction[direction].selected_roots
                if root.status == ROOT_READY
            ],
            "controlled_candidate_tokens_sha256": candidate.tokens_sha256,
        }
        for direction in FIXED_DIRECTIONS
    }
    input_contract = {
        "schema_version": SCHEMA_VERSION,
        "namespace": namespace,
        "query_id": query_id,
        "candidate_key": lock.candidate.candidate_key,
        "candidate_physical_row": int(lock.candidate.physical_gallery_row),
        "candidate_source_image_sha256": lock.candidate.source_image_sha256,
        "grid_shape": list(lock.candidate.grid_shape),
        "valid_mask_sha256": permutation["valid_mask_sha256"],
        "valid_mask_p_v2_sha256": tensor_sha256(
            lock.candidate.valid_patch_mask.to(torch.uint8)
        ),
        "input_cache_payload_sha256": lock.candidate.cache_payload_sha256,
        "input_tokens_sha256": lock.candidate.tokens_sha256,
    }
    permutation = {
        **permutation,
        "namespace": namespace,
        "query_id": query_id,
        "input_contract": input_contract,
        "input_contract_sha256": _canonical_sha256(input_contract),
        "global_candidate_materialization_count": 1,
        "application_by_direction": application_by_direction,
        "application_receipt_sha256": _canonical_sha256(application_by_direction),
    }
    permutation["branch_input_logical_sha256"] = _canonical_sha256(
        {
            "input_contract": input_contract,
            "permutation_nonce": permutation["permutation_nonce"],
            "destination_to_source_sha256": permutation[
                "destination_to_source_sha256"
            ],
            "output_tokens_sha256": permutation["output_tokens_sha256"],
            "application_by_direction": application_by_direction,
        }
    )
    specs = {
        direction: {
            root.root_ordinal: RootDecodeSpecV1(
                root.root_ordinal,
                root.query_mask,
                root.reference_mask,
                candidate,
                root.root_ordinal,
            )
            for root in lock.core_lock_by_direction[direction].selected_roots
            if root.status == ROOT_READY
        }
        for direction in FIXED_DIRECTIONS
    }
    return _energy(
        model,
        query,
        lock,
        specs,
        branch=P_COORD,
        eligible=True,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={
            "global_candidate_permutation": permutation,
            "global_candidate_permutation_count": 1,
        },
    )


def topology_energy_pv2(
    model,
    query,
    lock: NaturalCandidatePLockV2,
    *,
    namespace: str,
    query_id: str,
    streaming_chunk_size: int | None = 64,
):
    specs = {}
    orders = {}
    geometry = {}
    eligible = True
    for direction in FIXED_DIRECTIONS:
        roots = tuple(
            root
            for root in lock.core_lock_by_direction[direction].selected_roots
            if root.status == ROOT_READY
        )
        order, rows = _matched_topology_derangement(
            roots,
            grid=lock.candidate.grid_shape,
            namespace=namespace,
            query_id=query_id,
            candidate_key=lock.candidate.candidate_key,
            direction=direction,
        )
        geometry[direction] = rows
        if order is None:
            eligible = False
            specs[direction] = {}
            continue
        orders[direction] = list(order)
        specs[direction] = {
            root.root_ordinal: RootDecodeSpecV1(
                root.root_ordinal,
                root.query_mask,
                roots[source].reference_mask,
                lock.candidate,
                roots[source].root_ordinal,
            )
            for root, source in zip(roots, order, strict=True)
        }
    return _energy(
        model,
        query,
        lock,
        specs,
        branch=T_SHAPE_MATCHED_ROOT_ASSIGNMENT,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={
            "matching_fields": [
                "area",
                "row_span",
                "col_span",
                "component_count",
                "border_touch",
            ],
            "root_destination_to_source": orders,
            "root_geometry_records": geometry,
            "per_destination_geometry_matched": eligible,
        },
    )


def _population_derangement(
    roots: Sequence[Any],
    *,
    namespace: str,
    query_id: str,
    candidate_key: str,
    direction: str,
) -> tuple[int, ...] | None:
    hashes = [tensor_sha256(root.reference_mask.to(torch.uint8)) for root in roots]
    donor_for: dict[int, int] = {}
    destination_for_donor: dict[int, int] = {}

    def assign(destination: int, seen: set[int]) -> bool:
        donors = sorted(
            (
                source
                for source in range(len(roots))
                if source != destination and hashes[source] != hashes[destination]
            ),
            key=lambda source: hashlib.sha256(
                (
                    f"{namespace}|{query_id}|{candidate_key}|{direction}|"
                    f"{destination}|{source}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        for source in donors:
            if source in seen:
                continue
            seen.add(source)
            prior = destination_for_donor.get(source)
            if prior is None or assign(prior, seen):
                destination_for_donor[source] = destination
                donor_for[destination] = source
                return True
        return False

    def degree(destination: int) -> int:
        return sum(
            source != destination and hashes[source] != hashes[destination]
            for source in range(len(roots))
        )

    destinations = sorted(
        range(len(roots)),
        key=lambda destination: (
            degree(destination),
            int(roots[destination].root_ordinal),
        ),
    )
    if not all(assign(destination, set()) for destination in destinations):
        return None
    result = tuple(donor_for[index] for index in range(len(roots)))
    require(
        sorted(result) == list(range(len(roots)))
        and all(index != source for index, source in enumerate(result))
        and all(hashes[index] != hashes[source] for index, source in enumerate(result)),
        "root-population assignment did not change every destination mask",
    )
    return result


def root_population_energy_pv2(
    model,
    query,
    lock: NaturalCandidatePLockV2,
    *,
    namespace: str,
    query_id: str,
    streaming_chunk_size: int | None = 64,
):
    specs = {}
    orders = {}
    ordinal_assignments = {}
    records = {}
    eligible = True
    for direction in FIXED_DIRECTIONS:
        roots = tuple(lock.core_lock_by_direction[direction].selected_roots)
        root_ordinals = [int(root.root_ordinal) for root in roots]
        root_statuses = [root.status for root in roots]
        input_hashes = [
            tensor_sha256(root.reference_mask.to(torch.uint8)) for root in roots
        ]
        bipartite_edges = [
            {
                "destination_position": int(destination),
                "destination_root_ordinal": int(roots[destination].root_ordinal),
                "source_position": int(source),
                "source_root_ordinal": int(roots[source].root_ordinal),
                "destination_mask_sha256": input_hashes[destination],
                "source_mask_sha256": input_hashes[source],
            }
            for destination in range(len(roots))
            for source in range(len(roots))
            if roots[destination].status == ROOT_READY
            and roots[source].status == ROOT_READY
            and source != destination
            and input_hashes[source] != input_hashes[destination]
        ]
        common = {
            "direction": direction,
            "namespace": namespace,
            "query_id": query_id,
            "candidate_key": lock.candidate.candidate_key,
            "candidate_physical_row": int(lock.candidate.physical_gallery_row),
            "selected_root_ordinals": root_ordinals,
            "selected_root_statuses": root_statuses,
            "complete_root_count": len(roots),
            "input_reference_mask_sha256": input_hashes,
            "input_mask_multiset_sha256": _canonical_sha256(
                sorted(input_hashes)
            ),
            "matching_fields": [],
            "shape_matching_required": False,
            "query_root_population_preserved": True,
            "p_lock_preserved": True,
            "bipartite_edges": bipartite_edges,
            "bipartite_graph_sha256": _canonical_sha256(bipartite_edges),
            "destination_order_policy": "ASCENDING_DEGREE_THEN_ROOT_ORDINAL",
            "edge_order_policy": "HASH_NAMESPACE_DESTINATION_SOURCE",
        }
        non_ready = [
            int(root.root_ordinal) for root in roots if root.status != ROOT_READY
        ]
        if non_ready:
            eligible = False
            specs[direction] = {}
            records[direction] = {
                **common,
                "eligible": False,
                "ineligibility_reason": "SELECTED_REFERENCE_ROOT_NOT_READY",
                "non_ready_root_ordinals": non_ready,
                "matched_root_count": 0,
                "destination_position_to_source_position": [],
                "destination_root_to_source_root": [],
                "output_reference_mask_sha256": [],
                "output_mask_multiset_sha256": None,
                "fixed_point_free": False,
                "all_destination_masks_changed": False,
                "root_population_preserved": False,
            }
            continue
        order = _population_derangement(
            roots,
            namespace=namespace,
            query_id=query_id,
            candidate_key=lock.candidate.candidate_key,
            direction=direction,
        )
        if order is None:
            eligible = False
            specs[direction] = {}
            records[direction] = {
                **common,
                "eligible": False,
                "ineligibility_reason": "NO_CHANGED_FIXED_POINT_FREE_ROOT_ASSIGNMENT",
                "non_ready_root_ordinals": [],
                "matched_root_count": 0,
                "destination_position_to_source_position": [],
                "destination_root_to_source_root": [],
                "output_reference_mask_sha256": [],
                "output_mask_multiset_sha256": None,
                "fixed_point_free": False,
                "all_destination_masks_changed": False,
                "root_population_preserved": False,
            }
            continue
        orders[direction] = list(order)
        output_hashes = [input_hashes[source] for source in order]
        assignment_rows = [
            {
                "destination_position": int(destination),
                "destination_root_ordinal": int(roots[destination].root_ordinal),
                "source_position": int(source),
                "source_root_ordinal": int(roots[source].root_ordinal),
                "input_reference_mask_sha256": input_hashes[destination],
                "output_reference_mask_sha256": input_hashes[source],
                "destination_mask_changed": (
                    input_hashes[destination] != input_hashes[source]
                ),
            }
            for destination, source in enumerate(order)
        ]
        ordinal_assignments[direction] = assignment_rows
        input_multiset_sha256 = _canonical_sha256(sorted(input_hashes))
        output_multiset_sha256 = _canonical_sha256(sorted(output_hashes))
        fixed_point_free = all(
            destination != source for destination, source in enumerate(order)
        )
        all_changed = all(
            input_hashes[index] != output_hashes[index]
            for index in range(len(roots))
        )
        records[direction] = {
            **common,
            "eligible": True,
            "ineligibility_reason": None,
            "non_ready_root_ordinals": [],
            "matched_root_count": len(order),
            "destination_position_to_source_position": list(order),
            "destination_root_to_source_root": assignment_rows,
            "output_reference_mask_sha256": output_hashes,
            "input_mask_multiset_sha256": input_multiset_sha256,
            "output_mask_multiset_sha256": output_multiset_sha256,
            "fixed_point_free": fixed_point_free,
            "all_destination_masks_changed": all_changed,
            "root_population_preserved": (
                input_multiset_sha256 == output_multiset_sha256
                and len(order) == len(roots)
            ),
        }
        require(
            fixed_point_free
            and all_changed
            and records[direction]["root_population_preserved"] is True,
            "root-population corruption receipt drift",
        )
        specs[direction] = {
            root.root_ordinal: RootDecodeSpecV1(
                root.root_ordinal,
                root.query_mask,
                roots[source].reference_mask,
                lock.candidate,
                roots[source].root_ordinal,
            )
            for root, source in zip(roots, order, strict=True)
        }
    return _energy(
        model,
        query,
        lock,
        specs,
        branch=T_ROOT_ASSIGNMENT_POPULATION,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={
            "claim": "COMPOSITE_ROOT_ASSIGNMENT_CORRUPTION_NOT_TOPOLOGY_ONLY",
            "root_destination_to_source": orders,
            "root_ordinal_assignments": ordinal_assignments,
            "population_records": records,
            "all_directions_receipted": set(records) == set(FIXED_DIRECTIONS),
            "branch_input_logical_sha256": _canonical_sha256(
                {
                    "schema_version": SCHEMA_VERSION,
                    "branch": T_ROOT_ASSIGNMENT_POPULATION,
                    "namespace": namespace,
                    "query_id": query_id,
                    "candidate_key": lock.candidate.candidate_key,
                    "candidate_physical_row": int(
                        lock.candidate.physical_gallery_row
                    ),
                    "population_records": records,
                }
            ),
        },
    )


def matched_null_energy_pv2(
    model,
    query,
    lock: NaturalCandidatePLockV2,
    *,
    namespace: str,
    query_id: str,
    streaming_chunk_size: int | None = 64,
):
    specs = {}
    pullbacks = {}
    translations = {}
    ineligibility_reasons = []
    eligible = True
    for direction in FIXED_DIRECTIONS:
        specs[direction] = {}
        pullbacks[direction] = {}
        translations[direction] = []
        for root in lock.core_lock_by_direction[direction].selected_roots:
            if root.status != ROOT_READY:
                translations[direction].append(
                    {
                        "root_ordinal": int(root.root_ordinal),
                        "root_status": root.status,
                        "evidence_bearing": False,
                        "eligible": True,
                        "ineligibility_reason": None,
                        "exact_zero_under_unchanged_denominator": True,
                        "query_translation": None,
                        "reference_translation": None,
                    }
                )
                continue
            query_namespace = f"{namespace}:QUERY"
            reference_namespace = f"{namespace}:REFERENCE"
            query_key = (
                f"{query_id}:{lock.candidate.candidate_key}:"
                f"{direction}:{root.root_ordinal}:QUERY"
            )
            reference_key = (
                f"{query_id}:{lock.candidate.candidate_key}:"
                f"{direction}:{root.root_ordinal}:REFERENCE"
            )
            query_translation, query_receipt = _connected_translation(
                root.query_mask,
                query.valid_patch_mask,
                query.grid_shape,
                role="QUERY",
                namespace=query_namespace,
                key=query_key,
            )
            reference_translation, reference_receipt = _connected_translation(
                root.reference_mask,
                lock.candidate.valid_patch_mask,
                lock.candidate.grid_shape,
                role="REFERENCE",
                namespace=reference_namespace,
                key=reference_key,
            )
            root_reason = None
            if query_translation is None:
                root_reason = "QUERY_CONNECTED_TRANSLATION_UNAVAILABLE"
            elif reference_translation is None:
                root_reason = "REFERENCE_CONNECTED_TRANSLATION_UNAVAILABLE"
            root_eligible = root_reason is None
            translations[direction].append(
                {
                    "root_ordinal": int(root.root_ordinal),
                    "root_status": root.status,
                    "evidence_bearing": True,
                    "eligible": root_eligible,
                    "ineligibility_reason": root_reason,
                    "exact_zero_under_unchanged_denominator": False,
                    "query_reference_control_keys_independent": (
                        query_namespace != reference_namespace
                        and query_key != reference_key
                    ),
                    "query_translation": query_receipt,
                    "reference_translation": reference_receipt,
                    "query_pullback_before_fixed_denominator": root_eligible,
                    "query_pullback_mapping_sha256": None
                    if query_translation is None
                    else query_translation.receipt[
                        "destination_to_source_sha256"
                    ],
                }
            )
            if not root_eligible:
                eligible = False
                ineligibility_reasons.append(
                    {
                        "direction": direction,
                        "root_ordinal": int(root.root_ordinal),
                        "reason": root_reason,
                    }
                )
                continue
            require(
                query_translation is not None and reference_translation is not None,
                "eligible N translation pair is absent",
            )
            pullbacks[direction][root.root_ordinal] = query_translation
            specs[direction][root.root_ordinal] = RootDecodeSpecV1(
                root.root_ordinal,
                query_translation.translated_mask,
                reference_translation.translated_mask,
                lock.candidate,
                root.root_ordinal,
            )
    translation_population_receipt = {
        direction: [
            {
                "root_ordinal": row["root_ordinal"],
                "root_status": row["root_status"],
                "evidence_bearing": row["evidence_bearing"],
                "query_population_sha256": None
                if row["query_translation"] is None
                else row["query_translation"][
                    "legal_translation_population_sha256"
                ],
                "reference_population_sha256": None
                if row["reference_translation"] is None
                else row["reference_translation"][
                    "legal_translation_population_sha256"
                ],
                "query_pullback_mapping_sha256": row.get(
                    "query_pullback_mapping_sha256"
                ),
            }
            for row in translations[direction]
        ]
        for direction in FIXED_DIRECTIONS
    }
    branch_input = {
        "schema_version": SCHEMA_VERSION,
        "branch": N_REGION_RESAMPLE,
        "namespace": namespace,
        "query_id": query_id,
        "candidate_key": lock.candidate.candidate_key,
        "candidate_physical_row": int(lock.candidate.physical_gallery_row),
        "translation_population_receipt": translation_population_receipt,
    }
    return _energy(
        model,
        query,
        lock,
        specs,
        branch=N_REGION_RESAMPLE,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        query_pullbacks_by_direction=pullbacks,
        extra_receipt={
            "claim": "COMPOSITE_CONNECTED_QUERY_REFERENCE_REGION_RESAMPLING",
            "translations_by_direction": translations,
            "translation_population_receipt": translation_population_receipt,
            "translation_population_receipt_sha256": _canonical_sha256(
                translation_population_receipt
            ),
            "branch_input_logical_sha256": _canonical_sha256(branch_input),
            "query_reference_translations_independently_namespaced": True,
            "query_pullback_before_p_v2_fixed_denominator": True,
            "ineligibility_reasons": ineligibility_reasons,
            "all_complete_roots_receipted": all(
                len(translations[direction])
                == len(lock.core_lock_by_direction[direction].selected_roots)
                for direction in FIXED_DIRECTIONS
            ),
        },
    )


__all__ = [
    "GXPV2Error",
    "NaturalCandidatePLockV2",
    "SCHEMA_VERSION",
    "c_bind_energy_pv2",
    "matched_null_energy_pv2",
    "natural_candidate_lock_v2",
    "p_coord_energy_pv2",
    "real_energy_pv2",
    "root_population_energy_pv2",
    "topology_energy_pv2",
]
