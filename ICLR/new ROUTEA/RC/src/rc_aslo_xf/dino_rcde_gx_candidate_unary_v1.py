"""Candidate-unary connected geometric energies and matched null branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_QUERY_LOCAL_COMPONENTS,
)
from .dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
    QueryTokenFieldV1,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import LOCK_H0, ROOT_READY
from .dino_rcde_sr0_mt_v_runtime_v1 import CandidatePLockV1, hash_parts, tensor_sha256
from .dino_rcde_sr0_mt_controls_v1 import (
    CDinoCandidateBindingV1,
    deterministic_derangement,
    permute_reference_content,
    random_connected_matched_shape,
    transport_reference_mask_positive_overlap,
)


REAL = "REAL"
C_BIND = "C_BIND"
P_COORD = "P_COORD"
T_TOPOLOGY = "T_TOPOLOGY"
N_MATCHED = "N_MATCHED"
BRANCHES = (REAL, C_BIND, P_COORD, T_TOPOLOGY, N_MATCHED)


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
    *,
    streaming_chunk_size: int | None,
) -> tuple[torch.Tensor, int, list[dict[str, object]]]:
    if not specs:
        return _zero(model), 0, []
    total = query.layers.new_zeros(query.valid_patch_mask.numel())
    coverage = torch.zeros_like(query.valid_patch_mask, dtype=torch.int64)
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
        coverage += qmask.detach().cpu().to(torch.int64)
        rows.append(
            {
                "root_ordinal": spec.root_ordinal,
                "source_reference_root_ordinal": (
                    spec.source_reference_root_ordinal
                ),
                "query_mask_sha256": tensor_sha256(qmask.to(torch.uint8)),
                "reference_mask_sha256": tensor_sha256(rmask.to(torch.uint8)),
                "candidate_key": spec.candidate.candidate_key,
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
    return energy, len(specs), rows


def _candidate_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    specs_by_direction: Mapping[str, Sequence[RootDecodeSpecV1]],
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
            receipt={"branch": branch, "eligible": False, **(extra_receipt or {})},
        )
    values = []
    count = 0
    rows = {}
    for direction in FIXED_DIRECTIONS:
        value, observed, detail = _decode_direction(
            model,
            query,
            specs_by_direction[direction],
            streaming_chunk_size=streaming_chunk_size,
        )
        values.append(value)
        count += observed
        rows[direction] = detail
    energy = 0.5 * (values[0] + values[1])
    return CandidateGeometricEnergyV1(
        branch=branch,
        energy=energy,
        eligible=True,
        direction_energies=(values[0], values[1]),
        decoded_root_count=count,
        receipt={
            "branch": branch,
            "eligible": True,
            "directions": rows,
            **(extra_receipt or {}),
        },
    )


def _ready_roots(lock: CandidatePLockV1, direction: str):
    sealed = lock.direction_locks[direction]
    if sealed.status == LOCK_H0:
        return None
    roots = [
        sealed.all_roots[root]
        for root in sealed.ordered_root_ordinals
        if sealed.all_roots[root].binding_status == ROOT_READY
    ]
    return roots if roots else None


def real_energy(
    model: torch.nn.Module,
    query: QueryTokenFieldV1,
    lock: CandidatePLockV1,
    *,
    streaming_chunk_size: int | None = 64,
) -> CandidateGeometricEnergyV1:
    specs = {}
    eligible = True
    for direction in FIXED_DIRECTIONS:
        roots = _ready_roots(lock, direction)
        if roots is None:
            eligible = False
            specs[direction] = ()
            continue
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
    eligible = True
    transports = []
    for direction in FIXED_DIRECTIONS:
        roots = _ready_roots(binding.destination_p_lock, direction)
        direction_specs = []
        if roots is None:
            eligible = False
            specs[direction] = ()
            continue
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
        branch=C_BIND,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={"mask_transports": transports},
    )


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
    eligible = True
    permutations = []
    for direction in FIXED_DIRECTIONS:
        roots = _ready_roots(lock, direction)
        direction_specs = []
        if roots is None:
            eligible = False
            specs[direction] = ()
            continue
        for root in roots:
            try:
                candidate, receipt = permute_reference_content(
                    lock.candidate,
                    root.reference_mask,
                    namespace=namespace,
                    query_id=query_id,
                    direction=direction,
                    arm_name=ARM_QUERY_LOCAL_COMPONENTS,
                )
            except Exception:
                eligible = False
                continue
            permutations.append(vars(receipt))
            direction_specs.append(
                RootDecodeSpecV1(
                    root.root_ordinal,
                    root.query_mask,
                    root.reference_mask,
                    candidate,
                    root.root_ordinal,
                )
            )
        specs[direction] = tuple(direction_specs)
    return _candidate_energy(
        model,
        query,
        specs,
        branch=P_COORD,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={"permutations": permutations},
    )


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
    eligible = True
    orders = {}
    for direction in FIXED_DIRECTIONS:
        roots = _ready_roots(lock, direction)
        if roots is None or len(roots) < 2:
            eligible = False
            specs[direction] = ()
            continue
        order = deterministic_derangement(
            len(roots),
            namespace,
            query_id,
            lock.candidate.candidate_key,
            direction,
        )
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
        branch=T_TOPOLOGY,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
        extra_receipt={"root_destination_to_source": orders},
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
    eligible = True
    for direction in FIXED_DIRECTIONS:
        roots = _ready_roots(lock, direction)
        direction_specs = []
        if roots is None:
            eligible = False
            specs[direction] = ()
            continue
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
        branch=N_MATCHED,
        eligible=eligible,
        streaming_chunk_size=streaming_chunk_size,
    )


__all__ = [
    "BRANCHES",
    "C_BIND",
    "CandidateGeometricEnergyV1",
    "GXUnaryError",
    "N_MATCHED",
    "P_COORD",
    "REAL",
    "RootDecodeSpecV1",
    "T_TOPOLOGY",
    "c_bind_energy",
    "matched_null_energy",
    "p_coord_energy",
    "real_energy",
    "topology_energy",
]
