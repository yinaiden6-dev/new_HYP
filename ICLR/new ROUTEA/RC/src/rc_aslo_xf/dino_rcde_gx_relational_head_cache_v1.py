"""Target-free pre-comparator relational head-cache schema and replay core.

This module performs no file I/O.  It stores frozen float32 relational fields
immediately before the live shared ``ell/rho/F`` comparator and replays the
true Natural/Core P-V2 fixed denominator.  T root-assignment payloads may be
retained as auxiliary diagnostics but are rejected by training replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch


SCHEMA_VERSION = "rc_dino_rcde_gx_relational_head_cache_v1_20260824"
FIXED_DIRECTIONS = ("a_to_b", "b_to_a")
TRAINING_BRANCHES = ("REAL", "C_BIND", "P_COORD", "N_REGION_RESAMPLE")
AUXILIARY_BRANCHES = ("T_ROOT_ASSIGNMENT_POPULATION",)
ALL_BRANCHES = (*TRAINING_BRANCHES, *AUXILIARY_BRANCHES)
ROOT_READY = "ROOT_READY"
ROOT_REFERENCE_MISSING = "ROOT_REFERENCE_MISSING"
ROOT_STATUSES = (ROOT_READY, ROOT_REFERENCE_MISSING)
TRAINABLE_HEAD_GROUPS = ("ell", "rho", "F")


class GXRelationalCacheError(ValueError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise GXRelationalCacheError(message)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes_hex": tensor.numpy().tobytes(order="C").hex(),
        }
    )


def require_sha256(value: object, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA256",
    )
    return str(value)


def canonical_mask(
    value: torch.Tensor, grid_shape: tuple[int, int], *, name: str
) -> torch.Tensor:
    grid = tuple(int(item) for item in grid_shape)
    require(len(grid) == 2 and min(grid) > 0, f"{name} grid drift")
    tensor = torch.as_tensor(value)
    require(
        tensor.device.type == "cpu"
        and tensor.dtype == torch.bool
        and tensor.numel() == math.prod(grid)
        and tensor.is_contiguous(),
        f"{name} must be contiguous CPU bool on its exact grid",
    )
    return tensor.detach().clone().flatten().contiguous()


def canonical_relational(value: torch.Tensor, *, query_numel: int) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    require(
        tensor.device.type == "cpu"
        and tensor.dtype == torch.float32
        and tensor.ndim == 2
        and tensor.shape[0] == query_numel
        and tensor.shape[1] > 0
        and tensor.is_contiguous()
        and not tensor.requires_grad
        and bool(torch.isfinite(tensor).all()),
        "cached relational payload must be finite contiguous detached CPU float32",
    )
    return tensor.detach().clone().contiguous()


@dataclass(frozen=True)
class QueryPullbackBijectionV1:
    source_original_indices: tuple[int, ...]
    destination_decode_indices: tuple[int, ...]
    source_to_destination_sha256: str
    destination_to_source_sha256: str
    logical_sha256: str

    def __post_init__(self) -> None:
        source = tuple(int(item) for item in self.source_original_indices)
        destination = tuple(int(item) for item in self.destination_decode_indices)
        require(
            bool(source)
            and len(source) == len(destination)
            and len(set(source)) == len(source)
            and len(set(destination)) == len(destination)
            and min((*source, *destination)) >= 0,
            "query pullback is not a complete active-slot bijection",
        )
        source_to_destination = [
            {"source": src, "destination": dst}
            for src, dst in zip(source, destination, strict=True)
        ]
        destination_to_source = [
            {"destination": dst, "source": src}
            for src, dst in zip(source, destination, strict=True)
        ]
        require(
            self.source_to_destination_sha256
            == canonical_sha256(source_to_destination)
            and self.destination_to_source_sha256
            == canonical_sha256(destination_to_source),
            "query pullback mapping hash drift",
        )
        payload = {
            "source_original_indices": list(source),
            "destination_decode_indices": list(destination),
            "source_to_destination_sha256": self.source_to_destination_sha256,
            "destination_to_source_sha256": self.destination_to_source_sha256,
        }
        require(
            self.logical_sha256 == canonical_sha256(payload),
            "query pullback logical hash drift",
        )
        object.__setattr__(self, "source_original_indices", source)
        object.__setattr__(self, "destination_decode_indices", destination)

    def payload(self) -> dict[str, object]:
        return {
            "source_original_indices": list(self.source_original_indices),
            "destination_decode_indices": list(self.destination_decode_indices),
            "source_to_destination_sha256": self.source_to_destination_sha256,
            "destination_to_source_sha256": self.destination_to_source_sha256,
            "logical_sha256": self.logical_sha256,
        }


def make_query_pullback(
    source_original_indices: Sequence[int],
    destination_decode_indices: Sequence[int],
) -> QueryPullbackBijectionV1:
    source = tuple(int(item) for item in source_original_indices)
    destination = tuple(int(item) for item in destination_decode_indices)
    source_to_destination = [
        {"source": src, "destination": dst}
        for src, dst in zip(source, destination, strict=True)
    ]
    destination_to_source = [
        {"destination": dst, "source": src}
        for src, dst in zip(source, destination, strict=True)
    ]
    payload = {
        "source_original_indices": list(source),
        "destination_decode_indices": list(destination),
        "source_to_destination_sha256": canonical_sha256(source_to_destination),
        "destination_to_source_sha256": canonical_sha256(destination_to_source),
    }
    return QueryPullbackBijectionV1(
        source_original_indices=source,
        destination_decode_indices=destination,
        source_to_destination_sha256=str(payload["source_to_destination_sha256"]),
        destination_to_source_sha256=str(payload["destination_to_source_sha256"]),
        logical_sha256=canonical_sha256(payload),
    )


@dataclass(frozen=True)
class FixedDenominatorCacheV1:
    outer_fold: int
    execution_ordinal: int
    query_id: str
    candidate_key: str
    candidate_physical_row: int
    branch: str
    direction: str
    query_grid_shape: tuple[int, int]
    raw_p_v2_record_sha256: str
    core_p_v2_record_sha256: str
    complete_root_table_sha256: str
    selected_root_ordinals: tuple[int, ...]
    root_status_by_ordinal: Mapping[int, str]
    original_query_masks_by_root: Mapping[int, torch.Tensor]
    coverage: torch.Tensor
    query_union: torch.Tensor
    coverage_sha256: str
    query_union_sha256: str
    root_mask_population_sha256: str
    logical_sha256: str

    def __post_init__(self) -> None:
        require(
            type(self.outer_fold) is int
            and 1 <= self.outer_fold <= 4
            and type(self.execution_ordinal) is int
            and self.execution_ordinal >= 0
            and isinstance(self.query_id, str)
            and bool(self.query_id)
            and isinstance(self.candidate_key, str)
            and bool(self.candidate_key)
            and type(self.candidate_physical_row) is int
            and self.candidate_physical_row >= 0
            and self.branch in ALL_BRANCHES
            and self.direction in FIXED_DIRECTIONS,
            "fixed denominator address drift",
        )
        for name, value in (
            ("raw P-V2 record", self.raw_p_v2_record_sha256),
            ("core P-V2 record", self.core_p_v2_record_sha256),
            ("complete root table", self.complete_root_table_sha256),
        ):
            require_sha256(value, name)
        selected = tuple(int(item) for item in self.selected_root_ordinals)
        require(
            bool(selected)
            and selected == tuple(sorted(set(selected)))
            and min(selected) >= 0,
            "fixed denominator selected-root population drift",
        )
        statuses = {int(key): str(value) for key, value in self.root_status_by_ordinal.items()}
        masks = {
            int(key): canonical_mask(
                value, self.query_grid_shape, name=f"root {key} original query"
            )
            for key, value in self.original_query_masks_by_root.items()
        }
        require(
            set(statuses) == set(masks) == set(selected)
            and all(status in ROOT_STATUSES for status in statuses.values()),
            "fixed denominator root status/mask axis drift",
        )
        expected = torch.zeros(math.prod(self.query_grid_shape), dtype=torch.int64)
        for root in selected:
            expected += masks[root].to(torch.int64)
        union = expected > 0
        coverage = torch.as_tensor(self.coverage)
        query_union = torch.as_tensor(self.query_union)
        require(
            coverage.device.type == "cpu"
            and coverage.dtype == torch.int64
            and coverage.is_contiguous()
            and coverage.shape == expected.shape
            and torch.equal(coverage, expected)
            and query_union.device.type == "cpu"
            and query_union.dtype == torch.bool
            and query_union.is_contiguous()
            and query_union.shape == union.shape
            and torch.equal(query_union, union)
            and bool(union.any()),
            "fixed denominator coverage/union drift",
        )
        coverage = coverage.detach().clone().contiguous()
        query_union = query_union.detach().clone().contiguous()
        root_rows = [
            {
                "root_ordinal": root,
                "status": statuses[root],
                "original_query_mask_sha256": tensor_sha256(masks[root].to(torch.uint8)),
            }
            for root in selected
        ]
        require(
            self.coverage_sha256 == tensor_sha256(coverage)
            and self.query_union_sha256 == tensor_sha256(query_union)
            and self.root_mask_population_sha256 == canonical_sha256(root_rows),
            "fixed denominator tensor/population hash drift",
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "outer_fold": self.outer_fold,
            "execution_ordinal": self.execution_ordinal,
            "query_id": self.query_id,
            "candidate_key": self.candidate_key,
            "candidate_physical_row": self.candidate_physical_row,
            "branch": self.branch,
            "direction": self.direction,
            "query_grid_shape": list(self.query_grid_shape),
            "raw_p_v2_record_sha256": self.raw_p_v2_record_sha256,
            "core_p_v2_record_sha256": self.core_p_v2_record_sha256,
            "complete_root_table_sha256": self.complete_root_table_sha256,
            "selected_root_ordinals": list(selected),
            "root_rows": root_rows,
            "coverage_shape": list(coverage.shape),
            "coverage_numel": int(coverage.numel()),
            "coverage_bytes": int(coverage.numel() * coverage.element_size()),
            "coverage_sha256": self.coverage_sha256,
            "query_union_shape": list(query_union.shape),
            "query_union_numel": int(query_union.numel()),
            "query_union_bytes": int(query_union.numel() * query_union.element_size()),
            "query_union_sha256": self.query_union_sha256,
            "root_mask_population_sha256": self.root_mask_population_sha256,
            "complete_root_coverage_used": True,
            "available_root_renormalization": False,
            "available_patch_renormalization": False,
        }
        require(
            self.logical_sha256 == canonical_sha256(payload),
            "fixed denominator logical hash drift",
        )
        object.__setattr__(self, "query_grid_shape", tuple(map(int, self.query_grid_shape)))
        object.__setattr__(self, "selected_root_ordinals", selected)
        object.__setattr__(self, "root_status_by_ordinal", MappingProxyType(statuses))
        object.__setattr__(self, "original_query_masks_by_root", MappingProxyType(masks))
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "query_union", query_union)

    @property
    def ready_root_ordinals(self) -> tuple[int, ...]:
        return tuple(
            root
            for root in self.selected_root_ordinals
            if self.root_status_by_ordinal[root] == ROOT_READY
        )

    @property
    def exact_zero_root_ordinals(self) -> tuple[int, ...]:
        return tuple(
            root
            for root in self.selected_root_ordinals
            if self.root_status_by_ordinal[root] == ROOT_REFERENCE_MISSING
        )

    def validate_bytes(self) -> None:
        expected_coverage = torch.zeros_like(self.coverage)
        root_rows = []
        for root in self.selected_root_ordinals:
            mask = self.original_query_masks_by_root[root]
            expected_coverage += mask.to(torch.int64)
            root_rows.append(
                {
                    "root_ordinal": root,
                    "status": self.root_status_by_ordinal[root],
                    "original_query_mask_sha256": tensor_sha256(
                        mask.to(torch.uint8)
                    ),
                }
            )
        expected_union = expected_coverage > 0
        require(
            self.coverage_sha256 == tensor_sha256(self.coverage)
            and self.query_union_sha256 == tensor_sha256(self.query_union)
            and torch.equal(self.coverage, expected_coverage)
            and torch.equal(self.query_union, expected_union)
            and self.root_mask_population_sha256 == canonical_sha256(root_rows)
            and all(
                mask.device.type == "cpu"
                and mask.dtype == torch.bool
                and mask.is_contiguous()
                for mask in self.original_query_masks_by_root.values()
            ),
            "fixed denominator cached bytes changed after sealing",
        )


def make_fixed_denominator_cache(
    *,
    outer_fold: int,
    execution_ordinal: int,
    query_id: str,
    candidate_key: str,
    candidate_physical_row: int,
    branch: str,
    direction: str,
    query_grid_shape: tuple[int, int],
    raw_p_v2_record_sha256: str,
    core_p_v2_record_sha256: str,
    complete_root_table_sha256: str,
    selected_root_ordinals: Sequence[int],
    root_status_by_ordinal: Mapping[int, str],
    original_query_masks_by_root: Mapping[int, torch.Tensor],
) -> FixedDenominatorCacheV1:
    selected = tuple(int(item) for item in selected_root_ordinals)
    masks = {
        int(root): canonical_mask(mask, query_grid_shape, name=f"root {root} query")
        for root, mask in original_query_masks_by_root.items()
    }
    coverage = torch.zeros(math.prod(query_grid_shape), dtype=torch.int64)
    for root in selected:
        coverage += masks[root].to(torch.int64)
    union = coverage > 0
    root_rows = [
        {
            "root_ordinal": root,
            "status": str(root_status_by_ordinal[root]),
            "original_query_mask_sha256": tensor_sha256(masks[root].to(torch.uint8)),
        }
        for root in selected
    ]
    common = {
        "schema_version": SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "execution_ordinal": execution_ordinal,
        "query_id": query_id,
        "candidate_key": candidate_key,
        "candidate_physical_row": candidate_physical_row,
        "branch": branch,
        "direction": direction,
        "query_grid_shape": list(query_grid_shape),
        "raw_p_v2_record_sha256": raw_p_v2_record_sha256,
        "core_p_v2_record_sha256": core_p_v2_record_sha256,
        "complete_root_table_sha256": complete_root_table_sha256,
        "selected_root_ordinals": list(selected),
        "root_rows": root_rows,
        "coverage_shape": list(coverage.shape),
        "coverage_numel": int(coverage.numel()),
        "coverage_bytes": int(coverage.numel() * coverage.element_size()),
        "coverage_sha256": tensor_sha256(coverage),
        "query_union_shape": list(union.shape),
        "query_union_numel": int(union.numel()),
        "query_union_bytes": int(union.numel() * union.element_size()),
        "query_union_sha256": tensor_sha256(union),
        "root_mask_population_sha256": canonical_sha256(root_rows),
        "complete_root_coverage_used": True,
        "available_root_renormalization": False,
        "available_patch_renormalization": False,
    }
    return FixedDenominatorCacheV1(
        outer_fold=outer_fold,
        execution_ordinal=execution_ordinal,
        query_id=query_id,
        candidate_key=candidate_key,
        candidate_physical_row=candidate_physical_row,
        branch=branch,
        direction=direction,
        query_grid_shape=query_grid_shape,
        raw_p_v2_record_sha256=raw_p_v2_record_sha256,
        core_p_v2_record_sha256=core_p_v2_record_sha256,
        complete_root_table_sha256=complete_root_table_sha256,
        selected_root_ordinals=selected,
        root_status_by_ordinal=root_status_by_ordinal,
        original_query_masks_by_root=masks,
        coverage=coverage,
        query_union=union,
        coverage_sha256=str(common["coverage_sha256"]),
        query_union_sha256=str(common["query_union_sha256"]),
        root_mask_population_sha256=str(common["root_mask_population_sha256"]),
        logical_sha256=canonical_sha256(common),
    )


@dataclass(frozen=True)
class RootRelationalPayloadV1:
    outer_fold: int
    execution_ordinal: int
    query_id: str
    query_source_image_sha256: str
    candidate_key: str
    candidate_physical_row: int
    candidate_content_binding_sha256: str
    branch: str
    direction: str
    root_ordinal: int
    source_reference_root_ordinal: int
    raw_p_v2_record_sha256: str
    core_p_v2_record_sha256: str
    complete_root_table_sha256: str
    branch_control_receipt_sha256: str
    direction_control_receipt_sha256: str
    root_control_receipt_sha256: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    original_reference_grid_shape: tuple[int, int]
    relational: torch.Tensor
    decode_query_mask: torch.Tensor
    decode_reference_mask: torch.Tensor
    original_root_query_mask: torch.Tensor
    original_root_reference_mask: torch.Tensor
    query_pullback: QueryPullbackBijectionV1 | None
    relational_sha256: str
    decode_query_mask_sha256: str
    decode_reference_mask_sha256: str
    original_root_query_mask_sha256: str
    original_root_reference_mask_sha256: str
    address_sha256: str
    logical_sha256: str

    def __post_init__(self) -> None:
        require(
            type(self.outer_fold) is int
            and 1 <= self.outer_fold <= 4
            and type(self.execution_ordinal) is int
            and self.execution_ordinal >= 0
            and isinstance(self.query_id, str)
            and bool(self.query_id)
            and isinstance(self.candidate_key, str)
            and bool(self.candidate_key)
            and type(self.candidate_physical_row) is int
            and self.candidate_physical_row >= 0
            and self.branch in ALL_BRANCHES
            and self.direction in FIXED_DIRECTIONS
            and type(self.root_ordinal) is int
            and self.root_ordinal >= 0
            and type(self.source_reference_root_ordinal) is int
            and self.source_reference_root_ordinal >= 0,
            "root relational cache address drift",
        )
        for name, value in (
            ("query source", self.query_source_image_sha256),
            ("candidate content binding", self.candidate_content_binding_sha256),
            ("raw P-V2", self.raw_p_v2_record_sha256),
            ("core P-V2", self.core_p_v2_record_sha256),
            ("complete root table", self.complete_root_table_sha256),
            ("branch control", self.branch_control_receipt_sha256),
            ("direction control", self.direction_control_receipt_sha256),
            ("root control", self.root_control_receipt_sha256),
        ):
            require_sha256(value, name)
        query_grid = tuple(map(int, self.query_grid_shape))
        reference_grid = tuple(map(int, self.reference_grid_shape))
        original_reference_grid = tuple(
            map(int, self.original_reference_grid_shape)
        )
        decode_query = canonical_mask(
            self.decode_query_mask, query_grid, name="decode query mask"
        )
        decode_reference = canonical_mask(
            self.decode_reference_mask, reference_grid, name="decode reference mask"
        )
        original_query = canonical_mask(
            self.original_root_query_mask, query_grid, name="original root query mask"
        )
        original_reference = canonical_mask(
            self.original_root_reference_mask,
            original_reference_grid,
            name="original root reference mask",
        )
        require(
            bool(decode_query.any())
            and bool(decode_reference.any())
            and bool(original_query.any())
            and bool(original_reference.any()),
            "READY root cache masks must be nonempty",
        )
        relational = canonical_relational(
            self.relational, query_numel=math.prod(query_grid)
        )
        if self.branch == "N_REGION_RESAMPLE":
            require(self.query_pullback is not None, "N cache lacks query pullback")
            source = tuple(torch.nonzero(original_query, as_tuple=False).flatten().tolist())
            destination = tuple(
                torch.nonzero(decode_query, as_tuple=False).flatten().tolist()
            )
            require(
                self.query_pullback.source_original_indices == source
                and self.query_pullback.destination_decode_indices == destination,
                "N query pullback/mask binding drift",
            )
        else:
            require(
                self.query_pullback is None and torch.equal(decode_query, original_query),
                "non-N cache changed query support or exposed a pullback",
            )
        require(
            self.relational_sha256 == tensor_sha256(relational)
            and self.decode_query_mask_sha256
            == tensor_sha256(decode_query.to(torch.uint8))
            and self.decode_reference_mask_sha256
            == tensor_sha256(decode_reference.to(torch.uint8))
            and self.original_root_query_mask_sha256
            == tensor_sha256(original_query.to(torch.uint8))
            and self.original_root_reference_mask_sha256
            == tensor_sha256(original_reference.to(torch.uint8)),
            "root relational/mask tensor hash drift",
        )
        address = {
            "outer_fold": self.outer_fold,
            "execution_ordinal": self.execution_ordinal,
            "query_id": self.query_id,
            "query_source_image_sha256": self.query_source_image_sha256,
            "candidate_key": self.candidate_key,
            "candidate_physical_row": self.candidate_physical_row,
            "candidate_content_binding_sha256": self.candidate_content_binding_sha256,
            "branch": self.branch,
            "direction": self.direction,
            "root_ordinal": self.root_ordinal,
            "source_reference_root_ordinal": self.source_reference_root_ordinal,
            "raw_p_v2_record_sha256": self.raw_p_v2_record_sha256,
            "core_p_v2_record_sha256": self.core_p_v2_record_sha256,
            "complete_root_table_sha256": self.complete_root_table_sha256,
            "branch_control_receipt_sha256": self.branch_control_receipt_sha256,
            "direction_control_receipt_sha256": self.direction_control_receipt_sha256,
            "root_control_receipt_sha256": self.root_control_receipt_sha256,
        }
        require(
            self.address_sha256 == canonical_sha256(address),
            "root cache address SHA drift",
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            **address,
            "address_sha256": self.address_sha256,
            "query_grid_shape": list(query_grid),
            "reference_grid_shape": list(reference_grid),
            "original_reference_grid_shape": list(original_reference_grid),
            "relational_dtype": "torch.float32",
            "relational_shape": list(relational.shape),
            "relational_numel": int(relational.numel()),
            "relational_bytes": int(relational.numel() * relational.element_size()),
            "relational_sha256": self.relational_sha256,
            "decode_query_mask_sha256": self.decode_query_mask_sha256,
            "decode_reference_mask_sha256": self.decode_reference_mask_sha256,
            "original_root_query_mask_sha256": self.original_root_query_mask_sha256,
            "original_root_reference_mask_sha256": self.original_root_reference_mask_sha256,
            "query_pullback": (
                None if self.query_pullback is None else self.query_pullback.payload()
            ),
            "pre_comparator": True,
            "target_free": True,
        }
        require(
            self.logical_sha256 == canonical_sha256(payload),
            "root relational cache logical SHA drift",
        )
        object.__setattr__(self, "query_grid_shape", query_grid)
        object.__setattr__(self, "reference_grid_shape", reference_grid)
        object.__setattr__(
            self, "original_reference_grid_shape", original_reference_grid
        )
        object.__setattr__(self, "relational", relational)
        object.__setattr__(self, "decode_query_mask", decode_query)
        object.__setattr__(self, "decode_reference_mask", decode_reference)
        object.__setattr__(self, "original_root_query_mask", original_query)
        object.__setattr__(self, "original_root_reference_mask", original_reference)

    def validate_bytes(self) -> None:
        require(
            self.relational.dtype == torch.float32
            and self.relational.device.type == "cpu"
            and self.relational.is_contiguous()
            and self.relational_sha256 == tensor_sha256(self.relational)
            and self.decode_query_mask_sha256
            == tensor_sha256(self.decode_query_mask.to(torch.uint8))
            and self.decode_reference_mask_sha256
            == tensor_sha256(self.decode_reference_mask.to(torch.uint8))
            and self.original_root_query_mask_sha256
            == tensor_sha256(self.original_root_query_mask.to(torch.uint8))
            and self.original_root_reference_mask_sha256
            == tensor_sha256(self.original_root_reference_mask.to(torch.uint8)),
            "root relational cache bytes changed after sealing",
        )


def make_root_relational_payload(
    *,
    outer_fold: int,
    execution_ordinal: int,
    query_id: str,
    query_source_image_sha256: str,
    candidate_key: str,
    candidate_physical_row: int,
    candidate_content_binding_sha256: str,
    branch: str,
    direction: str,
    root_ordinal: int,
    source_reference_root_ordinal: int,
    raw_p_v2_record_sha256: str,
    core_p_v2_record_sha256: str,
    complete_root_table_sha256: str,
    branch_control_receipt_sha256: str,
    direction_control_receipt_sha256: str,
    root_control_receipt_sha256: str,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    original_reference_grid_shape: tuple[int, int] | None = None,
    relational: torch.Tensor,
    decode_query_mask: torch.Tensor,
    decode_reference_mask: torch.Tensor,
    original_root_query_mask: torch.Tensor,
    original_root_reference_mask: torch.Tensor,
    query_pullback: QueryPullbackBijectionV1 | None = None,
) -> RootRelationalPayloadV1:
    relational_value = canonical_relational(
        relational, query_numel=math.prod(query_grid_shape)
    )
    decode_q = canonical_mask(decode_query_mask, query_grid_shape, name="decode query")
    decode_r = canonical_mask(
        decode_reference_mask, reference_grid_shape, name="decode reference"
    )
    original_q = canonical_mask(
        original_root_query_mask, query_grid_shape, name="original query"
    )
    original_reference_grid = (
        reference_grid_shape
        if original_reference_grid_shape is None
        else original_reference_grid_shape
    )
    original_r = canonical_mask(
        original_root_reference_mask,
        original_reference_grid,
        name="original reference",
    )
    address = {
        "outer_fold": outer_fold,
        "execution_ordinal": execution_ordinal,
        "query_id": query_id,
        "query_source_image_sha256": query_source_image_sha256,
        "candidate_key": candidate_key,
        "candidate_physical_row": candidate_physical_row,
        "candidate_content_binding_sha256": candidate_content_binding_sha256,
        "branch": branch,
        "direction": direction,
        "root_ordinal": root_ordinal,
        "source_reference_root_ordinal": source_reference_root_ordinal,
        "raw_p_v2_record_sha256": raw_p_v2_record_sha256,
        "core_p_v2_record_sha256": core_p_v2_record_sha256,
        "complete_root_table_sha256": complete_root_table_sha256,
        "branch_control_receipt_sha256": branch_control_receipt_sha256,
        "direction_control_receipt_sha256": direction_control_receipt_sha256,
        "root_control_receipt_sha256": root_control_receipt_sha256,
    }
    hashes = {
        "relational_sha256": tensor_sha256(relational_value),
        "decode_query_mask_sha256": tensor_sha256(decode_q.to(torch.uint8)),
        "decode_reference_mask_sha256": tensor_sha256(decode_r.to(torch.uint8)),
        "original_root_query_mask_sha256": tensor_sha256(original_q.to(torch.uint8)),
        "original_root_reference_mask_sha256": tensor_sha256(original_r.to(torch.uint8)),
    }
    address_sha = canonical_sha256(address)
    payload = {
        "schema_version": SCHEMA_VERSION,
        **address,
        "address_sha256": address_sha,
        "query_grid_shape": list(query_grid_shape),
        "reference_grid_shape": list(reference_grid_shape),
        "original_reference_grid_shape": list(original_reference_grid),
        "relational_dtype": "torch.float32",
        "relational_shape": list(relational_value.shape),
        "relational_numel": int(relational_value.numel()),
        "relational_bytes": int(
            relational_value.numel() * relational_value.element_size()
        ),
        **hashes,
        "query_pullback": None if query_pullback is None else query_pullback.payload(),
        "pre_comparator": True,
        "target_free": True,
    }
    return RootRelationalPayloadV1(
        outer_fold=outer_fold,
        execution_ordinal=execution_ordinal,
        query_id=query_id,
        query_source_image_sha256=query_source_image_sha256,
        candidate_key=candidate_key,
        candidate_physical_row=candidate_physical_row,
        candidate_content_binding_sha256=candidate_content_binding_sha256,
        branch=branch,
        direction=direction,
        root_ordinal=root_ordinal,
        source_reference_root_ordinal=source_reference_root_ordinal,
        raw_p_v2_record_sha256=raw_p_v2_record_sha256,
        core_p_v2_record_sha256=core_p_v2_record_sha256,
        complete_root_table_sha256=complete_root_table_sha256,
        branch_control_receipt_sha256=branch_control_receipt_sha256,
        direction_control_receipt_sha256=direction_control_receipt_sha256,
        root_control_receipt_sha256=root_control_receipt_sha256,
        query_grid_shape=query_grid_shape,
        reference_grid_shape=reference_grid_shape,
        original_reference_grid_shape=original_reference_grid,
        relational=relational_value,
        decode_query_mask=decode_q,
        decode_reference_mask=decode_r,
        original_root_query_mask=original_q,
        original_root_reference_mask=original_r,
        query_pullback=query_pullback,
        address_sha256=address_sha,
        logical_sha256=canonical_sha256(payload),
        **hashes,
    )


@dataclass(frozen=True)
class DirectionRelationalCacheV1:
    denominator: FixedDenominatorCacheV1
    root_payloads: tuple[RootRelationalPayloadV1, ...]
    direction_cache_sha256: str

    def __post_init__(self) -> None:
        denominator = self.denominator
        roots = tuple(self.root_payloads)
        ordinals = tuple(root.root_ordinal for root in roots)
        require(
            ordinals == denominator.ready_root_ordinals
            and len(roots) == len(denominator.ready_root_ordinals),
            "direction cache READY root population drift",
        )
        for root in roots:
            require(
                root.outer_fold == denominator.outer_fold
                and root.execution_ordinal == denominator.execution_ordinal
                and root.query_id == denominator.query_id
                and root.candidate_key == denominator.candidate_key
                and root.candidate_physical_row == denominator.candidate_physical_row
                and root.branch == denominator.branch
                and root.direction == denominator.direction
                and root.raw_p_v2_record_sha256
                == denominator.raw_p_v2_record_sha256
                and root.core_p_v2_record_sha256
                == denominator.core_p_v2_record_sha256
                and root.complete_root_table_sha256
                == denominator.complete_root_table_sha256
                and torch.equal(
                    root.original_root_query_mask,
                    denominator.original_query_masks_by_root[root.root_ordinal],
                ),
                "direction/root cache address or original-mask drift",
            )
        population = {
            "denominator_logical_sha256": denominator.logical_sha256,
            "root_logical_sha256": [root.logical_sha256 for root in roots],
        }
        require(
            self.direction_cache_sha256 == canonical_sha256(population),
            "direction cache population SHA drift",
        )
        object.__setattr__(self, "root_payloads", roots)

    def validate_bytes(self) -> None:
        self.denominator.validate_bytes()
        for root in self.root_payloads:
            root.validate_bytes()
        require(
            self.direction_cache_sha256
            == canonical_sha256(
                {
                    "denominator_logical_sha256": self.denominator.logical_sha256,
                    "root_logical_sha256": [
                        root.logical_sha256 for root in self.root_payloads
                    ],
                }
            ),
            "direction cache receipt changed after sealing",
        )


def make_direction_relational_cache(
    denominator: FixedDenominatorCacheV1,
    root_payloads: Sequence[RootRelationalPayloadV1],
) -> DirectionRelationalCacheV1:
    roots = tuple(sorted(root_payloads, key=lambda item: item.root_ordinal))
    population = {
        "denominator_logical_sha256": denominator.logical_sha256,
        "root_logical_sha256": [root.logical_sha256 for root in roots],
    }
    return DirectionRelationalCacheV1(
        denominator=denominator,
        root_payloads=roots,
        direction_cache_sha256=canonical_sha256(population),
    )


@dataclass(frozen=True)
class CandidateBranchRelationalCacheV1:
    outer_fold: int
    execution_ordinal: int
    query_id: str
    candidate_key: str
    candidate_physical_row: int
    branch: str
    branch_control_receipt_sha256: str
    directions: Mapping[str, DirectionRelationalCacheV1]
    candidate_cache_sha256: str

    def __post_init__(self) -> None:
        require(
            self.branch in ALL_BRANCHES
            and set(self.directions) == set(FIXED_DIRECTIONS),
            "candidate branch cache direction/branch drift",
        )
        require_sha256(self.branch_control_receipt_sha256, "branch control receipt")
        directions = dict(self.directions)
        for direction in FIXED_DIRECTIONS:
            denominator = directions[direction].denominator
            require(
                denominator.outer_fold == self.outer_fold
                and denominator.execution_ordinal == self.execution_ordinal
                and denominator.query_id == self.query_id
                and denominator.candidate_key == self.candidate_key
                and denominator.candidate_physical_row == self.candidate_physical_row
                and denominator.branch == self.branch
                and denominator.direction == direction
                and all(
                    root.branch_control_receipt_sha256
                    == self.branch_control_receipt_sha256
                    for root in directions[direction].root_payloads
                ),
                "candidate/direction cache binding drift",
            )
        all_roots = [
            root
            for direction in FIXED_DIRECTIONS
            for root in directions[direction].root_payloads
        ]
        require(
            len({root.query_source_image_sha256 for root in all_roots}) == 1
            and len({root.candidate_content_binding_sha256 for root in all_roots})
            == 1
            and len({root.relational.shape[1] for root in all_roots}) == 1,
            "candidate cache query/content/relational-dimension drift",
        )
        population = {
            "outer_fold": self.outer_fold,
            "execution_ordinal": self.execution_ordinal,
            "query_id": self.query_id,
            "candidate_key": self.candidate_key,
            "candidate_physical_row": self.candidate_physical_row,
            "branch": self.branch,
            "branch_control_receipt_sha256": self.branch_control_receipt_sha256,
            "direction_cache_sha256": {
                direction: directions[direction].direction_cache_sha256
                for direction in FIXED_DIRECTIONS
            },
        }
        require(
            self.candidate_cache_sha256 == canonical_sha256(population),
            "candidate branch cache population SHA drift",
        )
        object.__setattr__(self, "directions", MappingProxyType(directions))

    def validate_bytes(self) -> None:
        for direction in FIXED_DIRECTIONS:
            self.directions[direction].validate_bytes()
        require(
            self.candidate_cache_sha256
            == canonical_sha256(
                {
                    "outer_fold": self.outer_fold,
                    "execution_ordinal": self.execution_ordinal,
                    "query_id": self.query_id,
                    "candidate_key": self.candidate_key,
                    "candidate_physical_row": self.candidate_physical_row,
                    "branch": self.branch,
                    "branch_control_receipt_sha256": self.branch_control_receipt_sha256,
                    "direction_cache_sha256": {
                        direction: self.directions[direction].direction_cache_sha256
                        for direction in FIXED_DIRECTIONS
                    },
                }
            ),
            "candidate branch cache receipt changed after sealing",
        )


def make_candidate_branch_relational_cache(
    *,
    branch_control_receipt_sha256: str,
    directions: Mapping[str, DirectionRelationalCacheV1],
) -> CandidateBranchRelationalCacheV1:
    values = dict(directions)
    require(set(values) == set(FIXED_DIRECTIONS), "candidate cache directions absent")
    first = values[FIXED_DIRECTIONS[0]].denominator
    population = {
        "outer_fold": first.outer_fold,
        "execution_ordinal": first.execution_ordinal,
        "query_id": first.query_id,
        "candidate_key": first.candidate_key,
        "candidate_physical_row": first.candidate_physical_row,
        "branch": first.branch,
        "branch_control_receipt_sha256": branch_control_receipt_sha256,
        "direction_cache_sha256": {
            direction: values[direction].direction_cache_sha256
            for direction in FIXED_DIRECTIONS
        },
    }
    return CandidateBranchRelationalCacheV1(
        outer_fold=first.outer_fold,
        execution_ordinal=first.execution_ordinal,
        query_id=first.query_id,
        candidate_key=first.candidate_key,
        candidate_physical_row=first.candidate_physical_row,
        branch=first.branch,
        branch_control_receipt_sha256=branch_control_receipt_sha256,
        directions=values,
        candidate_cache_sha256=canonical_sha256(population),
    )


@dataclass(frozen=True)
class RootContributionReplayV1:
    root_ordinal: int
    contributions_in_original_query_slots: torch.Tensor
    query_pullback_applied: bool


@dataclass(frozen=True)
class DirectionEnergyReplayV1:
    energy: torch.Tensor
    patch_evidence: torch.Tensor
    root_contributions: Mapping[int, torch.Tensor]


@dataclass(frozen=True)
class CandidateEnergyReplayV1:
    branch: str
    energy: torch.Tensor
    direction_energies: tuple[torch.Tensor, torch.Tensor]


def require_live_head_comparator(model: torch.nn.Module, *, for_training: bool) -> None:
    require(callable(getattr(model, "compare_relational", None)), "live comparator absent")
    groups_method = getattr(model, "functional_parameter_groups", None)
    require(callable(groups_method), "live comparator parameter groups absent")
    groups = groups_method()
    require(
        isinstance(groups, Mapping)
        and set(TRAINABLE_HEAD_GROUPS).issubset(groups),
        "live ell/rho/F groups absent",
    )
    named = dict(model.named_parameters())
    head_names = [name for group in TRAINABLE_HEAD_GROUPS for name in groups[group]]
    require(set(head_names).issubset(named), "live ell/rho/F parameter absent")
    if for_training:
        require(
            all(named[name].requires_grad for name in head_names),
            "training replay received frozen ell/rho/F parameter",
        )


def replay_root_contribution(
    model: torch.nn.Module,
    payload: RootRelationalPayloadV1,
    *,
    for_training: bool = True,
) -> RootContributionReplayV1:
    require_live_head_comparator(model, for_training=for_training)
    payload.validate_bytes()
    parameter = next(model.parameters(), None)
    require(parameter is not None, "live comparator has no parameters")
    relational = payload.relational.to(device=parameter.device)
    decode_mask = payload.decode_query_mask.to(device=parameter.device)
    exact_zero = torch.zeros_like(relational)
    result = model.compare_relational(relational, exact_zero, decode_mask)
    contributions = getattr(result, "contributions", None)
    require(
        isinstance(contributions, torch.Tensor)
        and contributions.ndim == 1
        and contributions.numel() == payload.original_root_query_mask.numel()
        and contributions.dtype == torch.float32
        and bool(torch.isfinite(contributions).all()),
        "live comparator contribution drift",
    )
    if payload.query_pullback is None:
        output = contributions * payload.original_root_query_mask.to(
            device=contributions.device, dtype=contributions.dtype
        )
        pullback = False
    else:
        source = torch.tensor(
            payload.query_pullback.source_original_indices,
            dtype=torch.long,
            device=contributions.device,
        )
        destination = torch.tensor(
            payload.query_pullback.destination_decode_indices,
            dtype=torch.long,
            device=contributions.device,
        )
        output = torch.zeros_like(contributions)
        output[source] = contributions[destination]
        pullback = True
    original = payload.original_root_query_mask.to(output.device)
    require(
        not bool(output.detach()[~original].ne(0).any()),
        "replayed root contribution escaped original query slot",
    )
    return RootContributionReplayV1(
        root_ordinal=payload.root_ordinal,
        contributions_in_original_query_slots=output,
        query_pullback_applied=pullback,
    )


def replay_direction_energy(
    model: torch.nn.Module,
    cache: DirectionRelationalCacheV1,
    *,
    for_training: bool = True,
) -> DirectionEnergyReplayV1:
    cache.validate_bytes()
    denominator = cache.denominator
    root_outputs = {
        root.root_ordinal: replay_root_contribution(
            model, root, for_training=for_training
        ).contributions_in_original_query_slots
        for root in cache.root_payloads
    }
    exemplar = next(iter(root_outputs.values()))
    total = torch.zeros_like(exemplar)
    root_contributions: dict[int, torch.Tensor] = {}
    coverage = denominator.coverage.to(device=exemplar.device)
    union = denominator.query_union.to(device=exemplar.device)
    reciprocal = torch.zeros_like(exemplar)
    reciprocal[union] = coverage[union].to(exemplar.dtype).reciprocal()
    for root in denominator.selected_root_ordinals:
        if denominator.root_status_by_ordinal[root] == ROOT_READY:
            value = root_outputs[root]
            normalized = value * reciprocal
        else:
            normalized = torch.zeros_like(exemplar)
        root_contributions[root] = normalized
        total = total + normalized
    patch = total * union.to(total.dtype)
    require(
        not bool(patch.detach()[~union].ne(0).any()),
        "replayed direction evidence escaped fixed union",
    )
    energy = patch[union].mean()
    require(
        energy.ndim == 0
        and energy.dtype == torch.float32
        and bool(torch.isfinite(energy)),
        "replayed direction energy drift",
    )
    return DirectionEnergyReplayV1(
        energy=energy,
        patch_evidence=patch,
        root_contributions=MappingProxyType(root_contributions),
    )


def replay_candidate_energy(
    model: torch.nn.Module,
    cache: CandidateBranchRelationalCacheV1,
    *,
    for_training: bool = True,
) -> CandidateEnergyReplayV1:
    cache.validate_bytes()
    if for_training:
        require(
            cache.branch in TRAINING_BRANCHES,
            "auxiliary T relational cache cannot enter training replay",
        )
    values = tuple(
        replay_direction_energy(
            model, cache.directions[direction], for_training=for_training
        ).energy
        for direction in FIXED_DIRECTIONS
    )
    energy = 0.5 * (values[0] + values[1])
    return CandidateEnergyReplayV1(
        branch=cache.branch,
        energy=energy,
        direction_energies=values,  # type: ignore[arg-type]
    )


__all__ = [
    "ALL_BRANCHES",
    "AUXILIARY_BRANCHES",
    "CandidateBranchRelationalCacheV1",
    "CandidateEnergyReplayV1",
    "DirectionEnergyReplayV1",
    "DirectionRelationalCacheV1",
    "FIXED_DIRECTIONS",
    "FixedDenominatorCacheV1",
    "GXRelationalCacheError",
    "QueryPullbackBijectionV1",
    "ROOT_READY",
    "ROOT_REFERENCE_MISSING",
    "RootContributionReplayV1",
    "RootRelationalPayloadV1",
    "SCHEMA_VERSION",
    "TRAINING_BRANCHES",
    "canonical_sha256",
    "make_candidate_branch_relational_cache",
    "make_direction_relational_cache",
    "make_fixed_denominator_cache",
    "make_query_pullback",
    "make_root_relational_payload",
    "replay_candidate_energy",
    "replay_direction_energy",
    "replay_root_contribution",
    "tensor_sha256",
]
