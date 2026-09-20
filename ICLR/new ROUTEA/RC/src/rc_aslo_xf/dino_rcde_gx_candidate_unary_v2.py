"""Candidate-unary connected geometric energies and matched null branches."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    QueryTokenFieldV1,
    token_tensor_sha256,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import LOCK_H0, ROOT_READY
from .dino_rcde_sr0_mt_v_runtime_v1 import CandidatePLockV1, hash_parts, tensor_sha256
from .dino_rcde_sr0_mt_controls_v1 import (
    CDinoCandidateBindingV1,
    _permute_selected_layers,
    random_connected_matched_shape,
    transport_reference_mask_positive_overlap,
)
from .r0_natural_binding_v1 import (
    _sattolo_permutation,
    spatial_permutation_is_affine,
)
from .geometry_hypothesis_v1 import connected_components_4


SCHEMA_VERSION = "rc_dino_rcde_gx_candidate_unary_v2_20260824"


REAL = "REAL"
C_BIND = "C_BIND"
P_COORD = "P_COORD"
T_SHAPE_MATCHED_ROOT_ASSIGNMENT = "T_SHAPE_MATCHED_ROOT_ASSIGNMENT"
T_ROOT_ASSIGNMENT_POPULATION = "T_ROOT_ASSIGNMENT_POPULATION"
N_REGION_RESAMPLE = "N_REGION_RESAMPLE"
BRANCHES = (
    REAL,
    C_BIND,
    P_COORD,
    T_SHAPE_MATCHED_ROOT_ASSIGNMENT,
    T_ROOT_ASSIGNMENT_POPULATION,
    N_REGION_RESAMPLE,
)


class GXUnaryError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise GXUnaryError(message)


@dataclass(frozen=True)
class RootDecodeSpecV1:
    root_ordinal: int
    query_mask: torch.Tensor
    reference_mask: torch.Tensor
    candidate: CandidateReferenceFieldV1
    source_reference_root_ordinal: int


@dataclass(frozen=True)
class CandidateGeometricEnergyV1:
    branch: str
    energy: torch.Tensor
    eligible: bool
    direction_energies: tuple[torch.Tensor, torch.Tensor]
    decoded_root_count: int
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        require(self.branch in BRANCHES, "unknown GX unary branch")
        require(
            self.energy.ndim == 0
            and self.energy.is_floating_point()
            and bool(torch.isfinite(self.energy)),
            "GX unary energy drift",
        )
        require(len(self.direction_energies) == 2, "GX unary direction count drift")


def _zero(model: torch.nn.Module) -> torch.Tensor:
    signed = getattr(model, "signed_head", None)
    require(
        isinstance(signed, torch.nn.Linear)
        and signed.out_features == 1
        and signed.bias is None,
        "GX unary requires the shared bias-free signed head",
    )
    return signed.weight.square().sum() * 0.0


def _decode_direction(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    specs: Sequence[RootDecodeSpecV1],
    denominator_query_masks: Sequence[torch.Tensor],
    *,
    streaming_chunk_size: int | None,
) -> tuple[torch.Tensor, int, list[dict[str, object]], dict[str, object]]:
    if not denominator_query_masks:
        return _zero(model), 0, [], {
            "complete_root_count": 0,
            "coverage_sha256": tensor_sha256(
                torch.zeros_like(query.valid_patch_mask, dtype=torch.int64)
            ),
            "query_union_sha256": tensor_sha256(
                torch.zeros_like(query.valid_patch_mask, dtype=torch.uint8)
            ),
            "missing_root_contribution": "EXACT_ZERO",
        }
    total = query.layers.new_zeros(query.valid_patch_mask.numel())
    coverage = torch.zeros_like(query.valid_patch_mask, dtype=torch.int64)
    for mask in denominator_query_masks:
        value = torch.as_tensor(mask, dtype=torch.bool).flatten()
        require(
            value.shape == query.valid_patch_mask.shape
            and not bool((value & ~query.valid_patch_mask).any()),
            "GX complete-root denominator query mask drift",
        )
        coverage += value.to(torch.int64)
    rows = []
    for spec in specs:
        qmask = torch.as_tensor(
            spec.query_mask, dtype=torch.bool, device=query.layers.device
        ).flatten()
        rmask = torch.as_tensor(
            spec.reference_mask,
            dtype=torch.bool,
            device=spec.candidate.layers.device,
        ).flatten()
        require(
            bool(qmask.any())
            and bool(rmask.any())
            and not bool((qmask.cpu() & ~query.valid_patch_mask.cpu()).any())
            and not bool(
                (rmask.cpu() & ~spec.candidate.valid_patch_mask.cpu()).any()
            ),
            "GX unary root mask drift",
        )
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
        unary_pair = model.compare_relational(
            evidence.relational, exact_zero, qmask
        )
        patch = unary_pair.contributions * qmask.to(unary_pair.contributions.dtype)
        total = total + patch
        rows.append(
            {
                "root_ordinal": spec.root_ordinal,
                "source_reference_root_ordinal": (
                    spec.source_reference_root_ordinal
                ),
                "query_mask_sha256": tensor_sha256(qmask.to(torch.uint8)),
                "reference_mask_sha256": tensor_sha256(rmask.to(torch.uint8)),
                "candidate_key": spec.candidate.candidate_key,
                "candidate_tokens_sha256": spec.candidate.tokens_sha256,
            }
        )
    covered = coverage > 0
    require(bool(covered.any()), "GX unary fixed denominator is empty")
    reciprocal = torch.zeros_like(total)
    reciprocal[covered.to(reciprocal.device)] = 1.0 / coverage[covered].to(
        device=reciprocal.device, dtype=reciprocal.dtype
    )
    patch = total * reciprocal
    union = covered.to(patch.device)
    energy = patch[union].mean()
    require(energy.ndim == 0 and bool(torch.isfinite(energy)), "GX unary decode nonfinite")
    return energy, len(specs), rows, {
        "complete_root_count": len(denominator_query_masks),
        "decoded_root_count": len(specs),
        "missing_root_count": len(denominator_query_masks) - len(specs),
        "coverage_sha256": tensor_sha256(coverage),
        "query_union_sha256": tensor_sha256(covered.to(torch.uint8)),
        "missing_root_contribution": "EXACT_ZERO",
        "available_root_renormalization": False,
        "available_patch_renormalization": False,
    }


def _candidate_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    specs_by_direction: Mapping[str, Sequence[RootDecodeSpecV1]],
    denominator_masks_by_direction: Mapping[str, Sequence[torch.Tensor]],
    *,
    branch: str,
    eligible: bool,
    streaming_chunk_size: int | None,
    extra_receipt: Mapping[str, object] | None = None,
) -> CandidateGeometricEnergyV1:
    if not eligible:
        zero = _zero(model)
        return CandidateGeometricEnergyV1(
            branch=branch,
            energy=zero,
            eligible=False,
            direction_energies=(zero, zero),
            decoded_root_count=0,
            receipt={
                "schema_version": SCHEMA_VERSION,
                "branch": branch,
                "eligible": False,
                "complete_root_count_by_direction": {
                    direction: len(denominator_masks_by_direction.get(direction, ()))
                    for direction in FIXED_DIRECTIONS
                },
                **(extra_receipt or {}),
            },
        )
    values = []
    count = 0
    rows = {}
    denominator_receipts = {}
    for direction in FIXED_DIRECTIONS:
        value, observed, detail, denominator = _decode_direction(
            model,
            query,
            specs_by_direction[direction],
            denominator_masks_by_direction[direction],
            streaming_chunk_size=streaming_chunk_size,
        )
        values.append(value)
        count += observed
        rows[direction] = detail
        denominator_receipts[direction] = denominator
    energy = 0.5 * (values[0] + values[1])
    return CandidateGeometricEnergyV1(
        branch=branch,
        energy=energy,
        eligible=True,
        direction_energies=(values[0], values[1]),
        decoded_root_count=count,
        receipt={
            "schema_version": SCHEMA_VERSION,
            "branch": branch,
            "eligible": True,
            "complete_root_count_by_direction": {
                direction: len(denominator_masks_by_direction[direction])
                for direction in FIXED_DIRECTIONS
            },
            "decoded_root_count_by_direction": {
                direction: len(specs_by_direction[direction])
                for direction in FIXED_DIRECTIONS
            },
            "missing_root_count_by_direction": {
                direction: len(denominator_masks_by_direction[direction])
                - len(specs_by_direction[direction])
                for direction in FIXED_DIRECTIONS
            },
            "directions": rows,
            "fixed_denominator_by_direction": denominator_receipts,
            **(extra_receipt or {}),
        },
    )


def _complete_selected_roots(lock: CandidatePLockV1, direction: str):
    sealed = lock.direction_locks[direction]
    if sealed.status == LOCK_H0:
        return None
    roots = tuple(sealed.selected_roots)
    require(
        len(roots) >= 2
        and all(root.binding_status == ROOT_READY for root in roots),
        "GX selected complete-root population drift",
    )
    return roots


def real_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    lock: CandidatePLockV1,
    *,
    streaming_chunk_size: int | None = 64,
) -> CandidateGeometricEnergyV1:
    specs = {}
    denominators = {}
    eligible = True
    for direction in FIXED_DIRECTIONS:
        roots = _complete_selected_roots(lock, direction)
        if roots is None:
            eligible = False
            specs[direction] = ()
            denominators[direction] = ()
            continue
        denominators[direction] = tuple(root.query_mask for root in roots)
        specs[direction] = tuple(
            RootDecodeSpecV1(
                root.root_ordinal,
                root.query_mask,
                root.reference_mask,
                lock.candidate,
                root.root_ordinal,
            )
            for root in roots
        )
    return _candidate_energy(
        model,
        query,
        specs,
        denominators,
        branch=REAL,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
    )


def c_bind_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    binding: CDinoCandidateBindingV1,
    *,
    streaming_chunk_size: int | None = 64,
) -> CandidateGeometricEnergyV1:
    specs = {}
    denominators = {}
    eligible = True
    transports = []
    for direction in FIXED_DIRECTIONS:
        roots = _complete_selected_roots(binding.destination_p_lock, direction)
        direction_specs = []
        if roots is None:
            eligible = False
            specs[direction] = ()
            denominators[direction] = ()
            continue
        denominators[direction] = tuple(root.query_mask for root in roots)
        for root in roots:
            mask, receipt = transport_reference_mask_positive_overlap(
                root.reference_mask,
                root.reference_grid_shape,
                binding.candidate.grid_shape,
                binding.candidate.valid_patch_mask,
                p_lock_record_sha256=(
                    binding.direction_locks[direction].p_lock_record_sha256
                ),
            )
            transports.append(receipt.logical_record())
            if not receipt.decoder_legal:
                eligible = False
                continue
            direction_specs.append(
                RootDecodeSpecV1(
                    root.root_ordinal,
                    root.query_mask,
                    mask,
                    binding.candidate,
                    root.root_ordinal,
                )
            )
        specs[direction] = tuple(direction_specs)
    return _candidate_energy(
        model,
        query,
        specs,
        denominators,
        branch=C_BIND,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={"mask_transports": transports},
    )


def _global_candidate_permutation(
    candidate: CandidateReferenceFieldV1,
    *,
    namespace: str,
    query_id: str,
) -> tuple[CandidateReferenceFieldV1, dict[str, object]]:
    mask = torch.as_tensor(candidate.valid_patch_mask, dtype=torch.bool).flatten()
    count = int(mask.sum())
    require(count >= 2, "global P_COORD candidate has fewer than two valid tokens")
    valid_mask_sha256 = tensor_sha256(mask.to(torch.uint8))
    base = (
        f"{namespace}|{query_id}|{candidate.candidate_key}|"
        f"{candidate.grid_shape[0]}|{candidate.grid_shape[1]}|"
        f"{candidate.cache_payload_sha256}|{candidate.tokens_sha256}|"
        f"{valid_mask_sha256}|GX_GLOBAL_P_COORD_V3"
    ).encode("utf-8")
    order_tensor = None
    selected_nonce = None
    full_order = None
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    for nonce in range(256):
        trial = _sattolo_permutation(
            count,
            hashlib.sha256(base + nonce.to_bytes(4, "big")).digest(),
        )
        full = torch.arange(mask.numel(), dtype=torch.long)
        full[selected] = selected[trial]
        if not spatial_permutation_is_affine(full, candidate.grid_shape):
            order_tensor = trial
            selected_nonce = nonce
            full_order = full
            break
    require(order_tensor is not None, "global P_COORD could not seal non-affine derangement")
    order = tuple(int(item) for item in order_tensor.tolist())
    layers = _permute_selected_layers(candidate.layers, mask, order)
    output_sha256 = token_tensor_sha256(layers)
    controlled = replace(
        candidate,
        layers=layers,
        cache_payload_sha256=hash_parts(
            namespace,
            candidate.cache_payload_sha256,
            query_id,
            output_sha256,
        ),
        tokens_sha256=output_sha256,
    )
    height, width = candidate.grid_shape
    valid_set = set(int(item) for item in selected.tolist())
    edges = []
    for index in sorted(valid_set):
        row, column = divmod(index, width)
        for neighbor in (
            index + 1 if column + 1 < width else -1,
            index + width if row + 1 < height else -1,
        ):
            if neighbor in valid_set:
                edges.append((index, neighbor))

    def adjacent(first: int, second: int) -> bool:
        r0, c0 = divmod(first, width)
        r1, c1 = divmod(second, width)
        return abs(r0 - r1) + abs(c0 - c1) == 1

    preserved_edges = sum(
        adjacent(int(full_order[first]), int(full_order[second]))
        for first, second in edges
    )
    destroyed_edges = len(edges) - preserved_edges
    require(destroyed_edges > 0, "global P_COORD preserved every valid-grid edge")
    outside = ~mask.to(candidate.layers.device)
    require(
        torch.equal(layers[:, outside, :], candidate.layers[:, outside, :]),
        "global P_COORD changed content outside valid mask",
    )
    return controlled, {
        "method": "ONE_GLOBAL_VALID_TOKEN_DERANGEMENT_PER_QUERY_CANDIDATE_V2",
        "candidate_key": candidate.candidate_key,
        "valid_token_count": count,
        "permutation_nonce": int(selected_nonce),
        "destination_to_source_sha256": hash_parts(
            "GX_GLOBAL_P_COORD_ORDER_V1", *order
        ),
        "input_tokens_sha256": candidate.tokens_sha256,
        "output_tokens_sha256": output_sha256,
        "valid_mask_sha256": valid_mask_sha256,
        "input_cache_payload_sha256": candidate.cache_payload_sha256,
        "permutation_seed_contract_sha256": hash_parts(
            "GX_GLOBAL_P_COORD_SEED_V3",
            namespace,
            query_id,
            candidate.candidate_key,
            candidate.grid_shape[0],
            candidate.grid_shape[1],
            candidate.cache_payload_sha256,
            candidate.tokens_sha256,
            valid_mask_sha256,
        ),
        "fixed_point_free": all(index != source for index, source in enumerate(order)),
        "spatially_non_affine": True,
        "original_valid_grid_edge_count": len(edges),
        "preserved_valid_grid_edge_count": preserved_edges,
        "destroyed_valid_grid_edge_count": destroyed_edges,
        "outside_valid_byte_equal": True,
        "token_resize_or_interpolation_count": 0,
        "content_multiset_preserved": True,
        "candidate_axis_preserved": True,
        "same_permutation_all_roots_and_directions": True,
    }


def p_coord_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    lock: CandidatePLockV1,
    *,
    namespace: str,
    query_id: str,
    streaming_chunk_size: int | None = 64,
) -> CandidateGeometricEnergyV1:
    specs = {}
    denominators = {}
    eligible = True
    try:
        controlled_candidate, global_permutation = _global_candidate_permutation(
            lock.candidate,
            namespace=namespace,
            query_id=query_id,
        )
    except Exception:
        controlled_candidate = lock.candidate
        global_permutation = {"eligible": False}
        eligible = False
    for direction in FIXED_DIRECTIONS:
        roots = _complete_selected_roots(lock, direction)
        direction_specs = []
        if roots is None:
            eligible = False
            specs[direction] = ()
            denominators[direction] = ()
            continue
        denominators[direction] = tuple(root.query_mask for root in roots)
        for root in roots:
            direction_specs.append(
                RootDecodeSpecV1(
                    root.root_ordinal,
                    root.query_mask,
                    root.reference_mask,
                    controlled_candidate,
                    root.root_ordinal,
                )
            )
        specs[direction] = tuple(direction_specs)
    return _candidate_energy(
        model,
        query,
        specs,
        denominators,
        branch=P_COORD,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={"global_candidate_permutation": global_permutation},
    )


def _root_geometry_signature(mask: torch.Tensor, grid: tuple[int, int]) -> tuple[object, ...]:
    value = torch.as_tensor(mask, dtype=torch.bool).flatten()
    indices = torch.nonzero(value, as_tuple=False).flatten()
    require(indices.numel() > 0, "topology root geometry is empty")
    width = int(grid[1])
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices % width
    _, diagnostic = connected_components_4(value, grid)
    return (
        int(indices.numel()),
        int(rows.max() - rows.min() + 1),
        int(columns.max() - columns.min() + 1),
        int(diagnostic.component_count),
        bool(
            (rows == 0).any()
            or (columns == 0).any()
            or (rows == grid[0] - 1).any()
            or (columns == grid[1] - 1).any()
        ),
    )


def _matched_topology_derangement(
    roots: Sequence[Any],
    *,
    grid: tuple[int, int],
    namespace: str,
    query_id: str,
    candidate_key: str,
    direction: str,
) -> tuple[tuple[int, ...] | None, list[dict[str, object]]]:
    signatures = [
        _root_geometry_signature(root.reference_mask, grid) for root in roots
    ]
    mask_hashes = [
        tensor_sha256(root.reference_mask.to(torch.uint8)) for root in roots
    ]
    donor_for: dict[int, int] = {}
    destination_for_donor: dict[int, int] = {}

    def assign(destination: int, seen: set[int]) -> bool:
        donors = sorted(
            (
                source
                for source in range(len(roots))
                if source != destination
                and signatures[source] == signatures[destination]
                and mask_hashes[source] != mask_hashes[destination]
            ),
            key=lambda source: hash_parts(
                namespace,
                query_id,
                candidate_key,
                direction,
                "MATCHED_TOPOLOGY",
                destination,
                source,
            ),
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

    order = sorted(
        range(len(roots)),
        key=lambda destination: hash_parts(
            namespace,
            query_id,
            candidate_key,
            direction,
            "MATCHED_TOPOLOGY_DESTINATION",
            destination,
        ),
    )
    if not all(assign(destination, set()) for destination in order):
        return None, [
            {
                "root_ordinal": int(root.root_ordinal),
                "geometry_signature": list(signatures[index]),
                "reference_mask_sha256": mask_hashes[index],
            }
            for index, root in enumerate(roots)
        ]
    result = tuple(donor_for[index] for index in range(len(roots)))
    require(
        sorted(result) == list(range(len(roots)))
        and all(index != source for index, source in enumerate(result)),
        "matched topology plan is not a derangement",
    )
    return result, [
        {
            "root_ordinal": int(root.root_ordinal),
            "geometry_signature": list(signatures[index]),
            "reference_mask_sha256": mask_hashes[index],
        }
        for index, root in enumerate(roots)
    ]


def topology_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    lock: CandidatePLockV1,
    *,
    namespace: str,
    query_id: str,
    streaming_chunk_size: int | None = 64,
) -> CandidateGeometricEnergyV1:
    specs = {}
    denominators = {}
    eligible = True
    orders = {}
    geometry_rows = {}
    for direction in FIXED_DIRECTIONS:
        roots = _complete_selected_roots(lock, direction)
        if roots is None or len(roots) < 2:
            eligible = False
            specs[direction] = ()
            denominators[direction] = ()
            continue
        denominators[direction] = tuple(root.query_mask for root in roots)
        order, rows = _matched_topology_derangement(
            roots,
            grid=lock.candidate.grid_shape,
            namespace=namespace,
            query_id=query_id,
            candidate_key=lock.candidate.candidate_key,
            direction=direction,
        )
        geometry_rows[direction] = rows
        if order is None:
            eligible = False
            specs[direction] = ()
            continue
        orders[direction] = list(order)
        specs[direction] = tuple(
            RootDecodeSpecV1(
                root.root_ordinal,
                root.query_mask,
                roots[source].reference_mask,
                lock.candidate,
                roots[source].root_ordinal,
            )
            for root, source in zip(roots, order, strict=True)
        )
    return _candidate_energy(
        model,
        query,
        specs,
        denominators,
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
            "root_geometry_records": geometry_rows,
            "per_destination_geometry_matched": eligible,
        },
    )


def matched_null_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    lock: CandidatePLockV1,
    *,
    namespace: str,
    query_id: str,
    streaming_chunk_size: int | None = 64,
) -> CandidateGeometricEnergyV1:
    specs = {}
    denominators = {}
    eligible = True
    for direction in FIXED_DIRECTIONS:
        roots = _complete_selected_roots(lock, direction)
        direction_specs = []
        if roots is None:
            eligible = False
            specs[direction] = ()
            denominators[direction] = ()
            continue
        denominators[direction] = tuple(root.query_mask for root in roots)
        for root in roots:
            qmask = random_connected_matched_shape(
                root.query_mask,
                query.valid_patch_mask,
                query.grid_shape,
                namespace=namespace,
                key=f"{query_id}:{lock.candidate.candidate_key}:{direction}:{root.root_ordinal}:Q",
            )
            rmask = random_connected_matched_shape(
                root.reference_mask,
                lock.candidate.valid_patch_mask,
                lock.candidate.grid_shape,
                namespace=namespace,
                key=f"{query_id}:{lock.candidate.candidate_key}:{direction}:{root.root_ordinal}:R",
            )
            if qmask is None or rmask is None:
                eligible = False
                continue
            direction_specs.append(
                RootDecodeSpecV1(
                    root.root_ordinal,
                    qmask,
                    rmask,
                    lock.candidate,
                    root.root_ordinal,
                )
            )
        specs[direction] = tuple(direction_specs)
    return _candidate_energy(
        model,
        query,
        specs,
        denominators,
        branch=N_REGION_RESAMPLE,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
    )


__all__ = [
    "BRANCHES",
    "C_BIND",
    "CandidateGeometricEnergyV1",
    "GXUnaryError",
    "N_REGION_RESAMPLE",
    "P_COORD",
    "REAL",
    "RootDecodeSpecV1",
    "T_SHAPE_MATCHED_ROOT_ASSIGNMENT",
    "T_ROOT_ASSIGNMENT_POPULATION",
    "c_bind_energy",
    "matched_null_energy",
    "p_coord_energy",
    "real_energy",
    "topology_energy",
]
