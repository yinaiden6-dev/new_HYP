"""Additive compact execution primitives for the formal SR0-MT P path.

The objects in this module describe *addresses*, masks and candidate-local
features.  They deliberately do not contain episode roles.  In particular,
the feature callback receives one :class:`FactorizedCoordinate` and nothing
else.  A later metadata-only join may choose two natural-C128 members, but it
cannot alter either member's bytes.

This module performs no file I/O and loads no natural payload at import time.
It is an additive implementation of the compact/streaming repair contract;
the frozen dense implementation remains the scalar oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import torch

from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    FEATURE_DIM,
    FEATURE_SCHEMA_SHA256,
    CandidateActionFeatureTable,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    ROOT_H0,
    ROOT_READY,
    FormalPContractError,
    canonical_sha256,
    decode_mask_rle,
    encode_mask_rle,
    require_sha256,
    make_root_action_deployment,
    RootActionDeployment,
    tensor_sha256,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_compact_catalog_v1_20260815"
FACTORIZED_SOURCE_INDEX_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_factorized_source_index_v1_20260815"
)
ROLE_FREE_PAIR_ADDRESS_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_role_free_pair_address_v1_20260815"
)
ROLE_FREE_PAIR_MANIFEST_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_role_free_pair_manifest_v1_20260815"
)
ROLE_FREE_PAIR_FEATURE_CACHE_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_role_free_pair_feature_cache_v1_20260815"
)
VIRTUAL_SHARD_CATALOG_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_virtual_shard_catalog_v1_20260815"
)

NATURAL_CANDIDATE_COUNT = 128
SOURCE_INDEX_KIND = "FACTORIZED_SOURCE_INDEX"
PAIR_FEATURE_CACHE_KIND = "ROLE_FREE_PAIR_FEATURE_CACHE"

_ROLE_WORDS = frozenset(
    {
        "label",
        "target",
        "rival",
        "winner",
        "rank",
        "slot",
        "identity",
        "supergroup",
        "correctness",
        "outcome",
        "d1_gap",
        "d1_score",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalPContractError(message)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{name} must be an integer >= {minimum}",
    )
    return int(value)


def _grid(value: object, *, name: str) -> tuple[int, int]:
    _require(
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and min(value) > 0,
        f"{name} must be a positive [height,width]",
    )
    return int(value[0]), int(value[1])


def _tensor_bytes_sha256(value: torch.Tensor) -> str:
    """Use the frozen tensor hash for all dtype-visible physical tensors."""

    return tensor_sha256(torch.as_tensor(value).detach().cpu().contiguous())


def _assert_role_free(value: object, *, path: str = "root") -> None:
    """Reject role-bearing field names in pre-loss compact artifacts."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            _require(
                normalized not in _ROLE_WORDS,
                f"role-bearing key is forbidden at {path}.{key}",
            )
            _assert_role_free(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for ordinal, item in enumerate(value):
            _assert_role_free(item, path=f"{path}[{ordinal}]")


def _deployment_component_legal(mask: torch.Tensor, grid_shape: tuple[int, int]) -> bool:
    """Exact old deployment rule: >=4 patches, 2-D span and one 4CC."""

    value = torch.as_tensor(mask, dtype=torch.bool).reshape(grid_shape)
    points = torch.nonzero(value, as_tuple=False)
    if points.shape[0] < 4:
        return False
    if torch.unique(points[:, 0]).numel() < 2 or torch.unique(points[:, 1]).numel() < 2:
        return False
    active = {tuple(int(item) for item in point) for point in points.tolist()}
    first = next(iter(active))
    frontier = [first]
    visited = {first}
    while frontier:
        row, column = frontier.pop()
        for neighbour in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if neighbour in active and neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return visited == active


@dataclass(frozen=True)
class CompactMask:
    """One canonical mask stored once as RLE plus two independent hashes."""

    grid_shape: tuple[int, int]
    rle: tuple[tuple[int, int], ...]
    tensor_sha256: str
    logical_sha256: str

    def __post_init__(self) -> None:
        grid = _grid(self.grid_shape, name="compact mask grid")
        rle = tuple(tuple(run) for run in self.rle)
        mask = decode_mask_rle(
            [[int(start), int(length)] for start, length in rle],
            numel=grid[0] * grid[1],
        )
        require_sha256(self.tensor_sha256, name="compact mask tensor")
        require_sha256(self.logical_sha256, name="compact mask logical")
        _require(
            self.tensor_sha256 == _tensor_bytes_sha256(mask),
            "compact mask tensor hash drift",
        )
        _require(
            self.logical_sha256
            == canonical_sha256(
                {
                    "schema_version": SCHEMA_VERSION,
                    "grid_shape": list(grid),
                    "rle": [list(run) for run in rle],
                    "tensor_sha256": self.tensor_sha256,
                }
            ),
            "compact mask logical hash drift",
        )
        object.__setattr__(self, "grid_shape", grid)
        object.__setattr__(self, "rle", rle)

    @classmethod
    def from_tensor(
        cls, value: torch.Tensor, *, grid_shape: tuple[int, int]
    ) -> "CompactMask":
        grid = _grid(grid_shape, name="compact mask grid")
        mask = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous().flatten()
        _require(mask.shape == (grid[0] * grid[1],), "compact mask length drift")
        rle = tuple(tuple(run) for run in encode_mask_rle(mask))
        tensor_hash = _tensor_bytes_sha256(mask)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "grid_shape": list(grid),
            "rle": [list(run) for run in rle],
            "tensor_sha256": tensor_hash,
        }
        return cls(
            grid_shape=grid,
            rle=rle,
            tensor_sha256=tensor_hash,
            logical_sha256=canonical_sha256(payload),
        )

    def decode(self) -> torch.Tensor:
        return decode_mask_rle(
            [list(run) for run in self.rle],
            numel=self.grid_shape[0] * self.grid_shape[1],
        )

    def payload(self) -> dict[str, object]:
        return {
            "grid_shape": list(self.grid_shape),
            "rle": [list(run) for run in self.rle],
            "tensor_sha256": self.tensor_sha256,
            "logical_sha256": self.logical_sha256,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "CompactMask":
        _require(isinstance(value, Mapping), "compact mask payload is not a mapping")
        raw_rle = value.get("rle")
        _require(isinstance(raw_rle, list), "compact mask payload RLE absent")
        return cls(
            grid_shape=_grid(value.get("grid_shape"), name="compact mask payload grid"),
            rle=tuple(tuple(run) for run in raw_rle),  # type: ignore[arg-type]
            tensor_sha256=str(value.get("tensor_sha256", "")),
            logical_sha256=str(value.get("logical_sha256", "")),
        )


@dataclass(frozen=True)
class FactorizedCoordinate:
    """Decoded logical root/action coordinate passed to the feature oracle."""

    query_id: str
    execution_ordinal: int
    candidate_position: int
    candidate_key: str
    direction: str
    root_ordinal: int
    action_ordinal: int
    action_key: str
    eligible: bool
    binding_status: str
    colnomic_query_grid_shape: tuple[int, int]
    colnomic_reference_grid_shape: tuple[int, int]
    deployment_query_grid_shape: tuple[int, int]
    deployment_reference_grid_shape: tuple[int, int]
    query_geometry_sha256: str
    reference_geometry_sha256: str
    colnomic_query_action_mask: torch.Tensor
    colnomic_reference_action_mask: torch.Tensor
    deployment_query_mask: torch.Tensor
    deployment_reference_mask: torch.Tensor
    colnomic_query_action_mask_sha256: str
    colnomic_reference_action_mask_sha256: str
    deployment_query_mask_sha256: str
    deployment_reference_mask_sha256: str
    deployment_binding_sha256: str
    source_population_sha256: str
    coordinate_sha256: str


@dataclass(frozen=True)
class FactorizedDirectionalSourceIndex:
    """One complete candidate/direction population without Cartesian objects."""

    query_id: str
    historical_query_ordinal: int
    execution_ordinal: int
    query_source_image_sha256: str
    candidate_axis_sha256: str
    candidate_count_per_query: int
    candidate_position: int
    candidate_key: str
    candidate_physical_row: int
    candidate_reference_source_sha256: str
    direction: str
    query_token_sha256: str
    reference_token_sha256: str
    query_valid_mask_sha256: str
    reference_valid_mask_sha256: str
    query_geometry_sha256: str
    reference_geometry_sha256: str
    colnomic_query_root_masks: tuple[CompactMask, ...]
    colnomic_reference_mask_table: tuple[CompactMask, ...]
    colnomic_reference_mask_index: torch.Tensor
    deployment_query_root_masks: tuple[CompactMask, ...]
    deployment_reference_mask_table: tuple[CompactMask, ...]
    deployment_reference_mask_index: torch.Tensor
    action_keys: tuple[tuple[str, ...], ...]
    eligible: torch.Tensor
    feature_implementation_sha256: str
    numeric_policy_sha256: str
    population_sha256: str

    def __post_init__(self) -> None:
        _require(isinstance(self.query_id, str) and self.query_id, "query ID absent")
        for name, value in (
            ("historical query ordinal", self.historical_query_ordinal),
            ("execution ordinal", self.execution_ordinal),
            ("candidate position", self.candidate_position),
            ("candidate physical row", self.candidate_physical_row),
        ):
            _integer(value, name=name)
        _require(
            self.candidate_count_per_query == NATURAL_CANDIDATE_COUNT,
            "compact source must bind the complete natural C128 axis",
        )
        _require(
            self.candidate_position < self.candidate_count_per_query,
            "candidate position is outside natural C128",
        )
        _require(self.direction in P_DIRECTIONS, "compact P direction drift")
        for name, value in (
            ("query source image", self.query_source_image_sha256),
            ("candidate axis", self.candidate_axis_sha256),
            ("candidate key", self.candidate_key),
            ("candidate source image", self.candidate_reference_source_sha256),
            ("query token", self.query_token_sha256),
            ("reference token", self.reference_token_sha256),
            ("query valid mask", self.query_valid_mask_sha256),
            ("reference valid mask", self.reference_valid_mask_sha256),
            ("query geometry", self.query_geometry_sha256),
            ("reference geometry", self.reference_geometry_sha256),
            ("feature implementation", self.feature_implementation_sha256),
            ("numeric policy", self.numeric_policy_sha256),
            ("source population", self.population_sha256),
        ):
            require_sha256(value, name=name)

        colnomic_roots = tuple(self.colnomic_query_root_masks)
        colnomic_reference_table = tuple(self.colnomic_reference_mask_table)
        deployment_roots = tuple(self.deployment_query_root_masks)
        deployment_reference_table = tuple(self.deployment_reference_mask_table)
        _require(
            bool(colnomic_roots)
            and bool(colnomic_reference_table)
            and bool(deployment_roots)
            and bool(deployment_reference_table),
            "factorized mask axes are empty",
        )
        _require(
            len({item.logical_sha256 for item in colnomic_reference_table})
            == len(colnomic_reference_table),
            "ColNomic reference mask table is not deduplicated",
        )
        _require(
            len({item.logical_sha256 for item in deployment_reference_table})
            == len(deployment_reference_table),
            "deployment reference mask table is not deduplicated",
        )
        cq_grid = {item.grid_shape for item in colnomic_roots}
        cr_grid = {item.grid_shape for item in colnomic_reference_table}
        dq_grid = {item.grid_shape for item in deployment_roots}
        dr_grid = {item.grid_shape for item in deployment_reference_table}
        _require(
            len(cq_grid) == len(cr_grid) == len(dq_grid) == len(dr_grid) == 1,
            "factorized mask grid drift",
        )
        _require(
            len(colnomic_roots) == len(deployment_roots),
            "ColNomic/deployment root axis length drift",
        )
        keys = tuple(tuple(row) for row in self.action_keys)
        _require(
            len(keys) == len(colnomic_roots) and bool(keys[0]),
            "action-key matrix absent",
        )
        action_count = len(keys[0])
        _require(
            all(len(row) == action_count for row in keys),
            "action-key matrix is not Cartesian",
        )
        for row in keys:
            for value in row:
                require_sha256(value, name="coordinate action key")
        _require(
            len({value for row in keys for value in row})
            == len(colnomic_roots) * action_count,
            "coordinate action keys are duplicated or aliased",
        )
        colnomic_mask_index = torch.as_tensor(
            self.colnomic_reference_mask_index, dtype=torch.int64
        ).detach().cpu().contiguous()
        deployment_mask_index = torch.as_tensor(
            self.deployment_reference_mask_index, dtype=torch.int64
        ).detach().cpu().contiguous()
        eligibility = torch.as_tensor(
            self.eligible, dtype=torch.bool
        ).detach().cpu().contiguous()
        expected_shape = (len(colnomic_roots), action_count)
        _require(
            colnomic_mask_index.shape == expected_shape
            and deployment_mask_index.shape == expected_shape
            and eligibility.shape == expected_shape,
            "factorized Cartesian matrix shape drift",
        )
        _require(
            bool(colnomic_mask_index.ge(0).all())
            and bool(colnomic_mask_index.lt(len(colnomic_reference_table)).all()),
            "ColNomic reference mask-table binding is out of range",
        )
        _require(
            bool(deployment_mask_index.ge(0).all())
            and bool(
                deployment_mask_index.lt(len(deployment_reference_table)).all()
            ),
            "deployment reference mask-table binding is out of range",
        )
        # Decode each factorized axis exactly once.  The prior implementation
        # decoded the same four masks again for every eligible root/action
        # coordinate, turning validation back into an O(root*action*mask)
        # Cartesian object path.  The indexed boolean closure below proves the
        # identical per-coordinate nonempty predicate without changing the
        # logical matrices, hashes, errors, or on-demand decode API.
        colnomic_root_tensors = tuple(item.decode() for item in colnomic_roots)
        colnomic_reference_tensors = tuple(
            item.decode() for item in colnomic_reference_table
        )
        deployment_root_tensors = tuple(item.decode() for item in deployment_roots)
        deployment_reference_tensors = tuple(
            item.decode() for item in deployment_reference_table
        )
        colnomic_root_nonempty = torch.tensor(
            [bool(item.any()) for item in colnomic_root_tensors], dtype=torch.bool
        )
        colnomic_reference_nonempty = torch.tensor(
            [bool(item.any()) for item in colnomic_reference_tensors],
            dtype=torch.bool,
        )
        deployment_root_nonempty = torch.tensor(
            [bool(item.any()) for item in deployment_root_tensors], dtype=torch.bool
        )
        deployment_reference_nonempty = torch.tensor(
            [bool(item.any()) for item in deployment_reference_tensors],
            dtype=torch.bool,
        )
        coordinate_nonempty = (
            colnomic_root_nonempty[:, None]
            & colnomic_reference_nonempty[colnomic_mask_index]
            & deployment_root_nonempty[:, None]
            & deployment_reference_nonempty[deployment_mask_index]
        )
        _require(
            bool((~eligibility | coordinate_nonempty).all()),
            "eligible factorized coordinate has an empty feature/deployment mask",
        )

        ready_roots = torch.nonzero(
            eligibility.any(dim=1), as_tuple=False
        ).flatten().tolist()
        ready_reference_masks = torch.unique(
            deployment_mask_index[eligibility], sorted=True
        ).tolist()
        _require(
            all(
                _deployment_component_legal(
                    deployment_root_tensors[root],
                    deployment_roots[root].grid_shape,
                )
                for root in ready_roots
            ),
            "eligible deployment query mask violates >=4-patch 4CC 2-D span",
        )
        _require(
            all(
                _deployment_component_legal(
                    deployment_reference_tensors[ordinal],
                    deployment_reference_table[ordinal].grid_shape,
                )
                for ordinal in ready_reference_masks
            ),
            "eligible deployment reference mask violates >=4-patch 4CC 2-D span",
        )

        payload = self._population_payload(
            colnomic_roots=colnomic_roots,
            colnomic_reference_table=colnomic_reference_table,
            colnomic_mask_index=colnomic_mask_index,
            deployment_roots=deployment_roots,
            deployment_reference_table=deployment_reference_table,
            deployment_mask_index=deployment_mask_index,
            keys=keys,
            eligibility=eligibility,
        )
        _require(
            self.population_sha256 == canonical_sha256(payload),
            "factorized source population hash drift",
        )
        object.__setattr__(self, "colnomic_query_root_masks", colnomic_roots)
        object.__setattr__(
            self, "colnomic_reference_mask_table", colnomic_reference_table
        )
        object.__setattr__(
            self, "colnomic_reference_mask_index", colnomic_mask_index
        )
        object.__setattr__(self, "deployment_query_root_masks", deployment_roots)
        object.__setattr__(
            self, "deployment_reference_mask_table", deployment_reference_table
        )
        object.__setattr__(
            self, "deployment_reference_mask_index", deployment_mask_index
        )
        object.__setattr__(self, "action_keys", keys)
        object.__setattr__(self, "eligible", eligibility)

    def _population_payload(
        self,
        *,
        colnomic_roots: tuple[CompactMask, ...] | None = None,
        colnomic_reference_table: tuple[CompactMask, ...] | None = None,
        colnomic_mask_index: torch.Tensor | None = None,
        deployment_roots: tuple[CompactMask, ...] | None = None,
        deployment_reference_table: tuple[CompactMask, ...] | None = None,
        deployment_mask_index: torch.Tensor | None = None,
        keys: tuple[tuple[str, ...], ...] | None = None,
        eligibility: torch.Tensor | None = None,
    ) -> dict[str, object]:
        colnomic_roots = (
            self.colnomic_query_root_masks
            if colnomic_roots is None
            else colnomic_roots
        )
        colnomic_reference_table = (
            self.colnomic_reference_mask_table
            if colnomic_reference_table is None
            else colnomic_reference_table
        )
        colnomic_mask_index = (
            self.colnomic_reference_mask_index
            if colnomic_mask_index is None
            else colnomic_mask_index
        )
        deployment_roots = (
            self.deployment_query_root_masks
            if deployment_roots is None
            else deployment_roots
        )
        deployment_reference_table = (
            self.deployment_reference_mask_table
            if deployment_reference_table is None
            else deployment_reference_table
        )
        deployment_mask_index = (
            self.deployment_reference_mask_index
            if deployment_mask_index is None
            else deployment_mask_index
        )
        keys = self.action_keys if keys is None else keys
        eligibility = self.eligible if eligibility is None else eligibility
        root_count, action_count = eligibility.shape
        return {
            "schema_version": FACTORIZED_SOURCE_INDEX_SCHEMA,
            "query_id": self.query_id,
            "historical_query_ordinal": self.historical_query_ordinal,
            "execution_ordinal": self.execution_ordinal,
            "query_source_image_sha256": self.query_source_image_sha256,
            "candidate_axis_sha256": self.candidate_axis_sha256,
            "candidate_count_per_query": self.candidate_count_per_query,
            "candidate_position": self.candidate_position,
            "candidate_key": self.candidate_key,
            "candidate_physical_row": self.candidate_physical_row,
            "candidate_reference_source_sha256": self.candidate_reference_source_sha256,
            "direction": self.direction,
            "query_token_sha256": self.query_token_sha256,
            "reference_token_sha256": self.reference_token_sha256,
            "query_valid_mask_sha256": self.query_valid_mask_sha256,
            "reference_valid_mask_sha256": self.reference_valid_mask_sha256,
            "query_geometry_sha256": self.query_geometry_sha256,
            "reference_geometry_sha256": self.reference_geometry_sha256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "feature_implementation_sha256": self.feature_implementation_sha256,
            "numeric_policy_sha256": self.numeric_policy_sha256,
            "root_count": root_count,
            "action_count": action_count,
            "logical_coordinate_count": root_count * action_count,
            "ready_coordinate_count": int(eligibility.sum().item()),
            "h0_coordinate_count": int((~eligibility).sum().item()),
            "colnomic_query_root_mask_logical_sha256": [
                item.logical_sha256 for item in colnomic_roots
            ],
            "colnomic_reference_mask_table_logical_sha256": [
                item.logical_sha256 for item in colnomic_reference_table
            ],
            "colnomic_reference_mask_index_sha256": _tensor_bytes_sha256(
                colnomic_mask_index
            ),
            "deployment_query_root_mask_logical_sha256": [
                item.logical_sha256 for item in deployment_roots
            ],
            "deployment_reference_mask_table_logical_sha256": [
                item.logical_sha256 for item in deployment_reference_table
            ],
            "deployment_reference_mask_index_sha256": _tensor_bytes_sha256(
                deployment_mask_index
            ),
            "action_keys_sha256": canonical_sha256([list(row) for row in keys]),
            "eligibility_sha256": _tensor_bytes_sha256(eligibility),
        }

    @property
    def root_count(self) -> int:
        return int(self.eligible.shape[0])

    @property
    def action_count(self) -> int:
        return int(self.eligible.shape[1])

    @property
    def coordinate_count(self) -> int:
        return self.root_count * self.action_count

    def decode(self, root_ordinal: int, action_ordinal: int) -> FactorizedCoordinate:
        root = _integer(root_ordinal, name="root ordinal")
        action = _integer(action_ordinal, name="action ordinal")
        _require(root < self.root_count and action < self.action_count, "coordinate out of range")
        ready = bool(self.eligible[root, action])
        if ready:
            cq_mask = self.colnomic_query_root_masks[root].decode()
            colnomic_table_ordinal = int(
                self.colnomic_reference_mask_index[root, action]
            )
            cr_mask = self.colnomic_reference_mask_table[
                colnomic_table_ordinal
            ].decode()
            dq_mask = self.deployment_query_root_masks[root].decode()
            deployment_table_ordinal = int(
                self.deployment_reference_mask_index[root, action]
            )
            dr_mask = self.deployment_reference_mask_table[
                deployment_table_ordinal
            ].decode()
        else:
            cq_mask = torch.zeros_like(
                self.colnomic_query_root_masks[root].decode()
            )
            colnomic_table_ordinal = int(
                self.colnomic_reference_mask_index[root, action]
            )
            cr_mask = torch.zeros_like(
                self.colnomic_reference_mask_table[
                    colnomic_table_ordinal
                ].decode()
            )
            dq_mask = torch.zeros_like(
                self.deployment_query_root_masks[root].decode()
            )
            deployment_table_ordinal = int(
                self.deployment_reference_mask_index[root, action]
            )
            dr_mask = torch.zeros_like(
                self.deployment_reference_mask_table[
                    deployment_table_ordinal
                ].decode()
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_population_sha256": self.population_sha256,
            "query_id": self.query_id,
            "execution_ordinal": self.execution_ordinal,
            "candidate_position": self.candidate_position,
            "candidate_key": self.candidate_key,
            "direction": self.direction,
            "root_ordinal": root,
            "action_ordinal": action,
            "action_key": self.action_keys[root][action],
            "eligible": ready,
            "binding_status": ROOT_READY if ready else ROOT_H0,
            "colnomic_query_action_mask_sha256": _tensor_bytes_sha256(cq_mask),
            "colnomic_reference_action_mask_sha256": _tensor_bytes_sha256(cr_mask),
            "deployment_query_mask_sha256": _tensor_bytes_sha256(dq_mask),
            "deployment_reference_mask_sha256": _tensor_bytes_sha256(dr_mask),
        }
        deployment_binding_payload = {
            "action_key": self.action_keys[root][action],
            "eligible": ready,
            "query_mask_sha256": payload["deployment_query_mask_sha256"],
            "reference_mask_sha256": payload["deployment_reference_mask_sha256"],
            "query_grid_shape": list(
                self.deployment_query_root_masks[root].grid_shape
            ),
            "reference_grid_shape": list(
                self.deployment_reference_mask_table[
                    deployment_table_ordinal
                ].grid_shape
            ),
            "query_geometry_sha256": self.query_geometry_sha256,
            "reference_geometry_sha256": self.reference_geometry_sha256,
            "binding_status": ROOT_READY if ready else ROOT_H0,
        }
        deployment_binding_sha256 = canonical_sha256(deployment_binding_payload)
        payload["deployment_binding_sha256"] = deployment_binding_sha256
        return FactorizedCoordinate(
            query_id=self.query_id,
            execution_ordinal=self.execution_ordinal,
            candidate_position=self.candidate_position,
            candidate_key=self.candidate_key,
            direction=self.direction,
            root_ordinal=root,
            action_ordinal=action,
            action_key=self.action_keys[root][action],
            eligible=ready,
            binding_status=ROOT_READY if ready else ROOT_H0,
            colnomic_query_grid_shape=self.colnomic_query_root_masks[
                root
            ].grid_shape,
            colnomic_reference_grid_shape=self.colnomic_reference_mask_table[
                colnomic_table_ordinal
            ].grid_shape,
            deployment_query_grid_shape=self.deployment_query_root_masks[
                root
            ].grid_shape,
            deployment_reference_grid_shape=self.deployment_reference_mask_table[
                deployment_table_ordinal
            ].grid_shape,
            query_geometry_sha256=self.query_geometry_sha256,
            reference_geometry_sha256=self.reference_geometry_sha256,
            colnomic_query_action_mask=cq_mask,
            colnomic_reference_action_mask=cr_mask,
            deployment_query_mask=dq_mask,
            deployment_reference_mask=dr_mask,
            colnomic_query_action_mask_sha256=payload[
                "colnomic_query_action_mask_sha256"
            ],
            colnomic_reference_action_mask_sha256=payload[
                "colnomic_reference_action_mask_sha256"
            ],
            deployment_query_mask_sha256=payload["deployment_query_mask_sha256"],
            deployment_reference_mask_sha256=payload[
                "deployment_reference_mask_sha256"
            ],
            deployment_binding_sha256=deployment_binding_sha256,
            source_population_sha256=self.population_sha256,
            coordinate_sha256=canonical_sha256(payload),
        )

    def decode_deployment(
        self, root_ordinal: int, action_ordinal: int
    ) -> RootActionDeployment:
        """Decode only one selected deployment; never expand the Cartesian bank."""

        coordinate = self.decode(root_ordinal, action_ordinal)
        deployment = make_root_action_deployment(
            action_key=coordinate.action_key,
            eligible=coordinate.eligible,
            query_mask=coordinate.deployment_query_mask,
            reference_mask=coordinate.deployment_reference_mask,
            query_grid_shape=coordinate.deployment_query_grid_shape,
            reference_grid_shape=coordinate.deployment_reference_grid_shape,
            query_geometry_sha256=coordinate.query_geometry_sha256,
            reference_geometry_sha256=coordinate.reference_geometry_sha256,
        )
        _require(
            deployment.binding_sha256 == coordinate.deployment_binding_sha256,
            "selected compact deployment/legacy binding hash drift",
        )
        return deployment

    def payload(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": FACTORIZED_SOURCE_INDEX_SCHEMA,
            "target_free": True,
            "query_id": self.query_id,
            "historical_query_ordinal": self.historical_query_ordinal,
            "execution_ordinal": self.execution_ordinal,
            "query_source_image_sha256": self.query_source_image_sha256,
            "candidate_axis_sha256": self.candidate_axis_sha256,
            "candidate_count_per_query": self.candidate_count_per_query,
            "candidate_position": self.candidate_position,
            "candidate_key": self.candidate_key,
            "candidate_physical_row": self.candidate_physical_row,
            "candidate_reference_source_sha256": self.candidate_reference_source_sha256,
            "direction": self.direction,
            "query_token_sha256": self.query_token_sha256,
            "reference_token_sha256": self.reference_token_sha256,
            "query_valid_mask_sha256": self.query_valid_mask_sha256,
            "reference_valid_mask_sha256": self.reference_valid_mask_sha256,
            "query_geometry_sha256": self.query_geometry_sha256,
            "reference_geometry_sha256": self.reference_geometry_sha256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "feature_implementation_sha256": self.feature_implementation_sha256,
            "numeric_policy_sha256": self.numeric_policy_sha256,
            "colnomic_query_root_masks": [
                item.payload() for item in self.colnomic_query_root_masks
            ],
            "colnomic_reference_mask_table": [
                item.payload() for item in self.colnomic_reference_mask_table
            ],
            "colnomic_reference_mask_index": self.colnomic_reference_mask_index.clone(),
            "deployment_query_root_masks": [
                item.payload() for item in self.deployment_query_root_masks
            ],
            "deployment_reference_mask_table": [
                item.payload() for item in self.deployment_reference_mask_table
            ],
            "deployment_reference_mask_index": self.deployment_reference_mask_index.clone(),
            "action_keys": [list(row) for row in self.action_keys],
            "eligible": self.eligible.clone(),
            "population_receipt": {
                **self._population_payload(),
                "population_sha256": self.population_sha256,
            },
        }
        _assert_role_free(value)
        return value


def make_factorized_directional_source_index(
    *,
    query_id: str,
    historical_query_ordinal: int,
    execution_ordinal: int,
    query_source_image_sha256: str,
    candidate_axis_sha256: str,
    candidate_position: int,
    candidate_key: str,
    candidate_physical_row: int,
    candidate_reference_source_sha256: str,
    direction: str,
    query_token_sha256: str,
    reference_token_sha256: str,
    query_valid_mask_sha256: str,
    reference_valid_mask_sha256: str,
    query_geometry_sha256: str,
    reference_geometry_sha256: str,
    colnomic_query_root_masks: Sequence[torch.Tensor],
    colnomic_query_grid_shape: tuple[int, int],
    colnomic_reference_mask_table: Sequence[torch.Tensor],
    colnomic_reference_grid_shape: tuple[int, int],
    colnomic_reference_mask_index: torch.Tensor,
    deployment_query_root_masks: Sequence[torch.Tensor],
    deployment_query_grid_shape: tuple[int, int],
    deployment_reference_mask_table: Sequence[torch.Tensor],
    deployment_reference_grid_shape: tuple[int, int],
    deployment_reference_mask_index: torch.Tensor,
    action_keys: Sequence[Sequence[str]],
    eligible: torch.Tensor,
    feature_implementation_sha256: str,
    numeric_policy_sha256: str,
) -> FactorizedDirectionalSourceIndex:
    colnomic_roots = tuple(
        CompactMask.from_tensor(item, grid_shape=colnomic_query_grid_shape)
        for item in colnomic_query_root_masks
    )
    colnomic_table = tuple(
        CompactMask.from_tensor(item, grid_shape=colnomic_reference_grid_shape)
        for item in colnomic_reference_mask_table
    )
    deployment_roots = tuple(
        CompactMask.from_tensor(item, grid_shape=deployment_query_grid_shape)
        for item in deployment_query_root_masks
    )
    deployment_table = tuple(
        CompactMask.from_tensor(item, grid_shape=deployment_reference_grid_shape)
        for item in deployment_reference_mask_table
    )
    provisional = object.__new__(FactorizedDirectionalSourceIndex)
    values = {
        "query_id": query_id,
        "historical_query_ordinal": historical_query_ordinal,
        "execution_ordinal": execution_ordinal,
        "query_source_image_sha256": query_source_image_sha256,
        "candidate_axis_sha256": candidate_axis_sha256,
        "candidate_count_per_query": NATURAL_CANDIDATE_COUNT,
        "candidate_position": candidate_position,
        "candidate_key": candidate_key,
        "candidate_physical_row": candidate_physical_row,
        "candidate_reference_source_sha256": candidate_reference_source_sha256,
        "direction": direction,
        "query_token_sha256": query_token_sha256,
        "reference_token_sha256": reference_token_sha256,
        "query_valid_mask_sha256": query_valid_mask_sha256,
        "reference_valid_mask_sha256": reference_valid_mask_sha256,
        "query_geometry_sha256": query_geometry_sha256,
        "reference_geometry_sha256": reference_geometry_sha256,
        "colnomic_query_root_masks": colnomic_roots,
        "colnomic_reference_mask_table": colnomic_table,
        "colnomic_reference_mask_index": torch.as_tensor(
            colnomic_reference_mask_index, dtype=torch.int64
        )
        .detach()
        .cpu()
        .contiguous(),
        "deployment_query_root_masks": deployment_roots,
        "deployment_reference_mask_table": deployment_table,
        "deployment_reference_mask_index": torch.as_tensor(
            deployment_reference_mask_index, dtype=torch.int64
        )
        .detach()
        .cpu()
        .contiguous(),
        "action_keys": tuple(tuple(row) for row in action_keys),
        "eligible": torch.as_tensor(eligible, dtype=torch.bool)
        .detach()
        .cpu()
        .contiguous(),
        "feature_implementation_sha256": feature_implementation_sha256,
        "numeric_policy_sha256": numeric_policy_sha256,
    }
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    population = canonical_sha256(provisional._population_payload())
    return FactorizedDirectionalSourceIndex(**values, population_sha256=population)


def deserialize_factorized_directional_source_index(
    value: Mapping[str, object],
) -> FactorizedDirectionalSourceIndex:
    """Decode and fully hash-check one physical factorized source payload."""

    _require(
        isinstance(value, Mapping)
        and value.get("schema_version") == FACTORIZED_SOURCE_INDEX_SCHEMA
        and value.get("target_free") is True,
        "factorized source payload schema/role drift",
    )
    _assert_role_free(value)

    def masks(name: str) -> tuple[CompactMask, ...]:
        raw = value.get(name)
        _require(isinstance(raw, list) and bool(raw), f"{name} absent")
        _require(all(isinstance(item, Mapping) for item in raw), f"{name} row drift")
        return tuple(CompactMask.from_payload(item) for item in raw)  # type: ignore[arg-type]

    raw_keys = value.get("action_keys")
    _require(isinstance(raw_keys, list) and bool(raw_keys), "factorized action keys absent")
    receipt = value.get("population_receipt")
    _require(isinstance(receipt, Mapping), "factorized population receipt absent")
    population_sha = str(receipt.get("population_sha256", ""))
    return FactorizedDirectionalSourceIndex(
        query_id=str(value.get("query_id", "")),
        historical_query_ordinal=int(value.get("historical_query_ordinal", -1)),
        execution_ordinal=int(value.get("execution_ordinal", -1)),
        query_source_image_sha256=str(value.get("query_source_image_sha256", "")),
        candidate_axis_sha256=str(value.get("candidate_axis_sha256", "")),
        candidate_count_per_query=int(value.get("candidate_count_per_query", -1)),
        candidate_position=int(value.get("candidate_position", -1)),
        candidate_key=str(value.get("candidate_key", "")),
        candidate_physical_row=int(value.get("candidate_physical_row", -1)),
        candidate_reference_source_sha256=str(
            value.get("candidate_reference_source_sha256", "")
        ),
        direction=str(value.get("direction", "")),
        query_token_sha256=str(value.get("query_token_sha256", "")),
        reference_token_sha256=str(value.get("reference_token_sha256", "")),
        query_valid_mask_sha256=str(value.get("query_valid_mask_sha256", "")),
        reference_valid_mask_sha256=str(
            value.get("reference_valid_mask_sha256", "")
        ),
        query_geometry_sha256=str(value.get("query_geometry_sha256", "")),
        reference_geometry_sha256=str(value.get("reference_geometry_sha256", "")),
        colnomic_query_root_masks=masks("colnomic_query_root_masks"),
        colnomic_reference_mask_table=masks("colnomic_reference_mask_table"),
        colnomic_reference_mask_index=torch.as_tensor(
            value.get("colnomic_reference_mask_index"), dtype=torch.int64
        ),
        deployment_query_root_masks=masks("deployment_query_root_masks"),
        deployment_reference_mask_table=masks("deployment_reference_mask_table"),
        deployment_reference_mask_index=torch.as_tensor(
            value.get("deployment_reference_mask_index"), dtype=torch.int64
        ),
        action_keys=tuple(tuple(str(item) for item in row) for row in raw_keys),  # type: ignore[arg-type]
        eligible=torch.as_tensor(value.get("eligible"), dtype=torch.bool),
        feature_implementation_sha256=str(
            value.get("feature_implementation_sha256", "")
        ),
        numeric_policy_sha256=str(value.get("numeric_policy_sha256", "")),
        population_sha256=population_sha,
    )


@dataclass(frozen=True)
class RoleFreeCandidateAddress:
    query_id: str
    execution_ordinal: int
    query_source_image_sha256: str
    candidate_axis_sha256: str
    candidate_position: int
    candidate_key: str
    candidate_physical_row: int
    candidate_reference_source_sha256: str
    address_sha256: str

    def __post_init__(self) -> None:
        _require(isinstance(self.query_id, str) and self.query_id, "pair-address query absent")
        _integer(self.execution_ordinal, name="pair-address execution ordinal")
        _integer(self.candidate_position, name="pair-address candidate position")
        _integer(self.candidate_physical_row, name="pair-address physical row")
        _require(
            self.candidate_position < NATURAL_CANDIDATE_COUNT,
            "pair-address member is outside natural C128",
        )
        for name, value in (
            ("pair query source", self.query_source_image_sha256),
            ("pair candidate axis", self.candidate_axis_sha256),
            ("pair candidate key", self.candidate_key),
            ("pair candidate source", self.candidate_reference_source_sha256),
            ("pair member address", self.address_sha256),
        ):
            require_sha256(value, name=name)
        _require(
            self.address_sha256 == canonical_sha256(self.payload(include_hash=False)),
            "pair member address hash drift",
        )

    def payload(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "query_id": self.query_id,
            "execution_ordinal": self.execution_ordinal,
            "query_source_image_sha256": self.query_source_image_sha256,
            "candidate_axis_sha256": self.candidate_axis_sha256,
            "candidate_position": self.candidate_position,
            "candidate_key": self.candidate_key,
            "candidate_physical_row": self.candidate_physical_row,
            "candidate_reference_source_sha256": self.candidate_reference_source_sha256,
        }
        if include_hash:
            value["address_sha256"] = self.address_sha256
        return value


def make_role_free_candidate_address(
    *,
    query_id: str,
    execution_ordinal: int,
    query_source_image_sha256: str,
    candidate_axis_sha256: str,
    candidate_position: int,
    candidate_key: str,
    candidate_physical_row: int,
    candidate_reference_source_sha256: str,
) -> RoleFreeCandidateAddress:
    payload = {
        "query_id": query_id,
        "execution_ordinal": execution_ordinal,
        "query_source_image_sha256": query_source_image_sha256,
        "candidate_axis_sha256": candidate_axis_sha256,
        "candidate_position": candidate_position,
        "candidate_key": candidate_key,
        "candidate_physical_row": candidate_physical_row,
        "candidate_reference_source_sha256": candidate_reference_source_sha256,
    }
    return RoleFreeCandidateAddress(**payload, address_sha256=canonical_sha256(payload))


@dataclass(frozen=True)
class RoleFreePairAddress:
    query_id: str
    execution_ordinal: int
    query_source_image_sha256: str
    candidate_axis_sha256: str
    natural_axis_count: int
    members: tuple[RoleFreeCandidateAddress, RoleFreeCandidateAddress]
    pair_sha256: str

    def __post_init__(self) -> None:
        _require(self.natural_axis_count == NATURAL_CANDIDATE_COUNT, "pair did not join from C128")
        members = tuple(self.members)
        _require(len(members) == 2, "role-free pair must contain exactly two members")
        _require(
            members
            == tuple(sorted(members, key=lambda item: (item.candidate_key, item.candidate_physical_row))),
            "role-free pair members are not canonically ordered",
        )
        _require(
            len({item.candidate_key for item in members}) == 2
            and len({item.candidate_physical_row for item in members}) == 2,
            "role-free pair members alias",
        )
        for item in members:
            _require(
                item.query_id == self.query_id
                and item.execution_ordinal == self.execution_ordinal
                and item.query_source_image_sha256 == self.query_source_image_sha256
                and item.candidate_axis_sha256 == self.candidate_axis_sha256,
                "role-free pair member/query binding drift",
            )
        _require(
            self.pair_sha256 == canonical_sha256(self.payload(include_hash=False)),
            "role-free pair hash drift",
        )
        object.__setattr__(self, "members", members)

    def payload(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": ROLE_FREE_PAIR_ADDRESS_SCHEMA,
            "query_id": self.query_id,
            "execution_ordinal": self.execution_ordinal,
            "query_source_image_sha256": self.query_source_image_sha256,
            "candidate_axis_sha256": self.candidate_axis_sha256,
            "natural_axis_count": self.natural_axis_count,
            "members": [item.payload() for item in self.members],
        }
        if include_hash:
            value["pair_sha256"] = self.pair_sha256
        _assert_role_free(value)
        return value


def make_role_free_pair_address(
    natural_axis: Sequence[RoleFreeCandidateAddress],
    *,
    member_candidate_keys: Sequence[str],
) -> RoleFreePairAddress:
    axis = tuple(natural_axis)
    _require(len(axis) == NATURAL_CANDIDATE_COUNT, "pair join source axis is not C128")
    _require(
        [item.candidate_position for item in axis] == list(range(NATURAL_CANDIDATE_COUNT)),
        "natural C128 address order is incomplete or reordered",
    )
    _require(
        len({item.candidate_key for item in axis}) == NATURAL_CANDIDATE_COUNT
        and len({item.candidate_physical_row for item in axis}) == NATURAL_CANDIDATE_COUNT,
        "natural C128 address axis is duplicated",
    )
    query_bindings = {
        (
            item.query_id,
            item.execution_ordinal,
            item.query_source_image_sha256,
            item.candidate_axis_sha256,
        )
        for item in axis
    }
    _require(len(query_bindings) == 1, "natural C128 axis spans multiple queries")
    requested = tuple(member_candidate_keys)
    _require(len(requested) == 2 and len(set(requested)) == 2, "pair member keys must be two unique addresses")
    by_key = {item.candidate_key: item for item in axis}
    _require(all(item in by_key for item in requested), "pair member is absent from natural C128")
    members = tuple(sorted((by_key[item] for item in requested), key=lambda item: (item.candidate_key, item.candidate_physical_row)))
    query_id, execution, source_sha, axis_sha = next(iter(query_bindings))
    provisional = {
        "schema_version": ROLE_FREE_PAIR_ADDRESS_SCHEMA,
        "query_id": query_id,
        "execution_ordinal": execution,
        "query_source_image_sha256": source_sha,
        "candidate_axis_sha256": axis_sha,
        "natural_axis_count": NATURAL_CANDIDATE_COUNT,
        "members": [item.payload() for item in members],
    }
    return RoleFreePairAddress(
        query_id=query_id,
        execution_ordinal=execution,
        query_source_image_sha256=source_sha,
        candidate_axis_sha256=axis_sha,
        natural_axis_count=NATURAL_CANDIDATE_COUNT,
        members=members,  # type: ignore[arg-type]
        pair_sha256=canonical_sha256(provisional),
    )


@dataclass(frozen=True)
class RoleFreePairAddressManifest:
    records: tuple[RoleFreePairAddress, ...]
    expected_execution_ordinals: tuple[int, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        records = tuple(self.records)
        expected = tuple(_integer(item, name="expected execution ordinal") for item in self.expected_execution_ordinals)
        _require(expected == tuple(sorted(set(expected))), "expected pair query population is duplicated or unordered")
        _require(
            tuple(item.execution_ordinal for item in records) == expected,
            "pair-address records do not exactly cover expected query population",
        )
        _require(len({item.query_id for item in records}) == len(records), "pair-address query IDs duplicate")
        _require(
            self.manifest_sha256 == canonical_sha256(self.payload(include_hash=False)),
            "pair-address manifest hash drift",
        )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "expected_execution_ordinals", expected)

    def payload(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": ROLE_FREE_PAIR_MANIFEST_SCHEMA,
            "role_free": True,
            "natural_axis_count": NATURAL_CANDIDATE_COUNT,
            "query_count": len(self.records),
            "expected_execution_ordinals": list(self.expected_execution_ordinals),
            "records": [item.payload() for item in self.records],
        }
        if include_hash:
            value["manifest_sha256"] = self.manifest_sha256
        _assert_role_free(value)
        return value


def make_role_free_pair_address_manifest(
    records: Sequence[RoleFreePairAddress], *, expected_execution_ordinals: Sequence[int]
) -> RoleFreePairAddressManifest:
    ordered = tuple(sorted(records, key=lambda item: item.execution_ordinal))
    expected = tuple(expected_execution_ordinals)
    provisional: dict[str, object] = {
        "schema_version": ROLE_FREE_PAIR_MANIFEST_SCHEMA,
        "role_free": True,
        "natural_axis_count": NATURAL_CANDIDATE_COUNT,
        "query_count": len(ordered),
        "expected_execution_ordinals": list(expected),
        "records": [item.payload() for item in ordered],
    }
    return RoleFreePairAddressManifest(
        records=ordered,
        expected_execution_ordinals=expected,
        manifest_sha256=canonical_sha256(provisional),
    )


class CoordinateFeatureFunction(Protocol):
    """Role-free frozen feature oracle interface."""

    def __call__(self, coordinate: FactorizedCoordinate) -> torch.Tensor: ...


@dataclass(frozen=True)
class CandidateFeatureTensor:
    source_population_sha256: str
    features: torch.Tensor
    eligibility: torch.Tensor
    tensor_sha256: str
    eligibility_sha256: str
    logical_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.source_population_sha256, name="candidate feature source")
        require_sha256(self.tensor_sha256, name="candidate feature tensor")
        require_sha256(self.eligibility_sha256, name="candidate feature eligibility")
        require_sha256(self.logical_sha256, name="candidate feature logical")
        features = torch.as_tensor(self.features, dtype=torch.float64).detach().cpu().contiguous()
        eligible = torch.as_tensor(self.eligibility, dtype=torch.bool).detach().cpu().contiguous()
        _require(
            features.ndim == 3
            and features.shape[-1] == FEATURE_DIM
            and eligible.shape == features.shape[:2],
            "candidate feature tensor shape drift",
        )
        _require(bool(torch.isfinite(features).all()), "candidate feature tensor is nonfinite")
        _require(
            bool(features[~eligible].eq(0.0).all()),
            "H0 feature coordinates must be exact zero",
        )
        _require(self.tensor_sha256 == _tensor_bytes_sha256(features), "candidate feature tensor hash drift")
        _require(
            self.eligibility_sha256 == _tensor_bytes_sha256(eligible),
            "candidate feature eligibility hash drift",
        )
        _require(
            self.logical_sha256
            == canonical_sha256(
                {
                    "schema_version": ROLE_FREE_PAIR_FEATURE_CACHE_SCHEMA,
                    "source_population_sha256": self.source_population_sha256,
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    "tensor_sha256": self.tensor_sha256,
                    "eligibility_sha256": self.eligibility_sha256,
                    "shape": list(features.shape),
                }
            ),
            "candidate feature logical hash drift",
        )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "eligibility", eligible)


def extract_role_free_candidate_features(
    source_index: FactorizedDirectionalSourceIndex,
    coordinate_feature_fn: CoordinateFeatureFunction,
) -> CandidateFeatureTensor:
    """Evaluate every logical coordinate without exposing any episode role."""

    _require(
        isinstance(source_index, FactorizedDirectionalSourceIndex),
        "feature extraction requires a factorized source index",
    )
    rows = torch.zeros(
        (source_index.root_count, source_index.action_count, FEATURE_DIM),
        dtype=torch.float64,
    )
    for root in range(source_index.root_count):
        for action in range(source_index.action_count):
            coordinate = source_index.decode(root, action)
            if not coordinate.eligible:
                continue
            value = torch.as_tensor(coordinate_feature_fn(coordinate), dtype=torch.float64).detach().cpu().contiguous()
            _require(
                value.shape == (FEATURE_DIM,) and bool(torch.isfinite(value).all()),
                "coordinate feature callback must return finite [16]",
            )
            rows[root, action] = value
    tensor_hash = _tensor_bytes_sha256(rows)
    eligibility_hash = _tensor_bytes_sha256(source_index.eligible)
    logical_payload = {
        "schema_version": ROLE_FREE_PAIR_FEATURE_CACHE_SCHEMA,
        "source_population_sha256": source_index.population_sha256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "tensor_sha256": tensor_hash,
        "eligibility_sha256": eligibility_hash,
        "shape": list(rows.shape),
    }
    return CandidateFeatureTensor(
        source_population_sha256=source_index.population_sha256,
        features=rows,
        eligibility=source_index.eligible,
        tensor_sha256=tensor_hash,
        eligibility_sha256=eligibility_hash,
        logical_sha256=canonical_sha256(logical_payload),
    )


class CandidateTensorFunction(Protocol):
    """Production role-free batched feature interface: exactly one call/block."""

    def __call__(
        self, source_index: FactorizedDirectionalSourceIndex
    ) -> CandidateFeatureTensor: ...


def make_candidate_feature_tensor(
    source_index: FactorizedDirectionalSourceIndex,
    features: torch.Tensor,
) -> CandidateFeatureTensor:
    """Seal one already-batched Cartesian tensor to its role-free source."""

    _require(
        isinstance(source_index, FactorizedDirectionalSourceIndex),
        "candidate tensor source index type drift",
    )
    value = torch.as_tensor(features, dtype=torch.float64).detach().cpu().contiguous()
    _require(
        value.shape
        == (source_index.root_count, source_index.action_count, FEATURE_DIM),
        "candidate tensor factory Cartesian shape drift",
    )
    value = value.clone()
    value[~source_index.eligible] = 0.0
    tensor_hash = _tensor_bytes_sha256(value)
    eligibility_hash = _tensor_bytes_sha256(source_index.eligible)
    logical_payload = {
        "schema_version": ROLE_FREE_PAIR_FEATURE_CACHE_SCHEMA,
        "source_population_sha256": source_index.population_sha256,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "tensor_sha256": tensor_hash,
        "eligibility_sha256": eligibility_hash,
        "shape": list(value.shape),
    }
    sealed = CandidateFeatureTensor(
        source_population_sha256=source_index.population_sha256,
        features=value,
        eligibility=source_index.eligible,
        tensor_sha256=tensor_hash,
        eligibility_sha256=eligibility_hash,
        logical_sha256=canonical_sha256(logical_payload),
    )
    return validate_candidate_feature_tensor(source_index, sealed)


def validate_candidate_feature_tensor(
    source_index: FactorizedDirectionalSourceIndex,
    value: CandidateFeatureTensor,
) -> CandidateFeatureTensor:
    """Bind one externally batched tensor to the complete source population."""

    _require(
        isinstance(source_index, FactorizedDirectionalSourceIndex)
        and isinstance(value, CandidateFeatureTensor),
        "batched candidate feature object type drift",
    )
    _require(
        value.source_population_sha256 == source_index.population_sha256,
        "batched candidate feature source population drift",
    )
    _require(
        value.features.shape
        == (source_index.root_count, source_index.action_count, FEATURE_DIM),
        "batched candidate feature Cartesian shape drift",
    )
    _require(
        torch.equal(value.eligibility, source_index.eligible),
        "batched candidate feature eligibility differs from source index",
    )
    _require(
        bool(value.features[~source_index.eligible].eq(0.0).all()),
        "batched candidate feature H0 is not exact zero",
    )
    return value


def candidate_action_table_from_compact_tensor(
    source_index: FactorizedDirectionalSourceIndex,
    value: CandidateFeatureTensor,
) -> CandidateActionFeatureTable:
    """Build the old scoring table without constructing deployment objects."""

    sealed = validate_candidate_feature_tensor(source_index, value)
    return CandidateActionFeatureTable(
        candidate_key=source_index.candidate_key,
        root_ordinals=tuple(range(source_index.root_count)),
        action_keys_by_root=source_index.action_keys,
        features_by_root=tuple(
            sealed.features[root].clone() for root in range(source_index.root_count)
        ),
        eligible_by_root=tuple(
            sealed.eligibility[root].clone()
            for root in range(source_index.root_count)
        ),
    )


@dataclass(frozen=True)
class PairFeatureBlock:
    member_ordinal: int
    candidate_key: str
    direction: str
    source_population_sha256: str
    root_count: int
    action_count: int
    feature_offset: int
    coordinate_count: int
    block_sha256: str


@dataclass(frozen=True)
class RoleFreePairFeatureCache:
    pair_address_sha256: str
    blocks: tuple[PairFeatureBlock, ...]
    features: torch.Tensor
    eligibility: torch.Tensor
    cache_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.pair_address_sha256, name="pair cache address")
        require_sha256(self.cache_sha256, name="pair cache")
        blocks = tuple(self.blocks)
        _require(len(blocks) == 2 * len(P_DIRECTIONS), "pair cache must contain two members and both directions")
        expected_order = [
            (member, direction)
            for member in range(2)
            for direction in P_DIRECTIONS
        ]
        _require(
            [(item.member_ordinal, item.direction) for item in blocks] == expected_order,
            "pair feature blocks are not canonically ordered",
        )
        features = torch.as_tensor(self.features, dtype=torch.float64).detach().cpu().contiguous()
        eligibility = torch.as_tensor(self.eligibility, dtype=torch.bool).detach().cpu().contiguous()
        _require(
            features.ndim == 2
            and features.shape[1] == FEATURE_DIM
            and eligibility.shape == (features.shape[0],),
            "pair cache ragged tensor shape drift",
        )
        cursor = 0
        for block in blocks:
            for name, value in (
                ("member ordinal", block.member_ordinal),
                ("root count", block.root_count),
                ("action count", block.action_count),
                ("feature offset", block.feature_offset),
                ("coordinate count", block.coordinate_count),
            ):
                _integer(value, name=name)
            require_sha256(block.candidate_key, name="pair block candidate")
            require_sha256(block.source_population_sha256, name="pair block source")
            require_sha256(block.block_sha256, name="pair block")
            _require(
                block.feature_offset == cursor
                and block.coordinate_count == block.root_count * block.action_count,
                "pair cache block offsets are not contiguous",
            )
            stop = cursor + block.coordinate_count
            block_payload = {
                "member_ordinal": block.member_ordinal,
                "candidate_key": block.candidate_key,
                "direction": block.direction,
                "source_population_sha256": block.source_population_sha256,
                "root_count": block.root_count,
                "action_count": block.action_count,
                "feature_offset": block.feature_offset,
                "coordinate_count": block.coordinate_count,
                "feature_tensor_sha256": _tensor_bytes_sha256(features[cursor:stop]),
                "eligibility_sha256": _tensor_bytes_sha256(eligibility[cursor:stop]),
            }
            _require(block.block_sha256 == canonical_sha256(block_payload), "pair feature block hash drift")
            cursor = stop
        _require(cursor == features.shape[0], "pair cache has unaddressed tensor rows")
        _require(bool(features[~eligibility].eq(0.0).all()), "pair cache H0 row is nonzero")
        _require(
            self.cache_sha256 == canonical_sha256(self._hash_payload(blocks, features, eligibility)),
            "pair feature cache hash drift",
        )
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "eligibility", eligibility)

    def _hash_payload(
        self,
        blocks: tuple[PairFeatureBlock, ...] | None = None,
        features: torch.Tensor | None = None,
        eligibility: torch.Tensor | None = None,
    ) -> dict[str, object]:
        blocks = self.blocks if blocks is None else blocks
        features = self.features if features is None else features
        eligibility = self.eligibility if eligibility is None else eligibility
        return {
            "schema_version": ROLE_FREE_PAIR_FEATURE_CACHE_SCHEMA,
            "pair_address_sha256": self.pair_address_sha256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "blocks": [
                {
                    "member_ordinal": item.member_ordinal,
                    "candidate_key": item.candidate_key,
                    "direction": item.direction,
                    "source_population_sha256": item.source_population_sha256,
                    "root_count": item.root_count,
                    "action_count": item.action_count,
                    "feature_offset": item.feature_offset,
                    "coordinate_count": item.coordinate_count,
                    "block_sha256": item.block_sha256,
                }
                for item in blocks
            ],
            "features_sha256": _tensor_bytes_sha256(features),
            "eligibility_sha256": _tensor_bytes_sha256(eligibility),
        }

    def payload(self) -> dict[str, object]:
        value = {
            **self._hash_payload(),
            "role_free": True,
            "features": self.features.clone(),
            "eligibility": self.eligibility.clone(),
            "cache_sha256": self.cache_sha256,
        }
        _assert_role_free(value)
        return value

    def decode(
        self, *, member_ordinal: int, direction: str, root_ordinal: int, action_ordinal: int
    ) -> tuple[torch.Tensor, bool]:
        _integer(member_ordinal, name="pair cache member ordinal")
        _integer(root_ordinal, name="pair cache root ordinal")
        _integer(action_ordinal, name="pair cache action ordinal")
        matches = [
            item
            for item in self.blocks
            if item.member_ordinal == member_ordinal and item.direction == direction
        ]
        _require(len(matches) == 1, "pair cache block address missing or duplicated")
        block = matches[0]
        _require(root_ordinal < block.root_count and action_ordinal < block.action_count, "pair cache coordinate out of range")
        offset = block.feature_offset + root_ordinal * block.action_count + action_ordinal
        return self.features[offset].clone(), bool(self.eligibility[offset])


def make_role_free_pair_feature_cache(
    pair_address: RoleFreePairAddress,
    source_indexes: Sequence[FactorizedDirectionalSourceIndex],
    candidate_tensor_fn: CandidateTensorFunction,
) -> RoleFreePairFeatureCache:
    """Build four blocks with exactly one batched producer call per block."""

    _require(isinstance(pair_address, RoleFreePairAddress), "pair cache address type drift")
    by_address: dict[tuple[str, str], FactorizedDirectionalSourceIndex] = {}
    for item in source_indexes:
        key = (item.candidate_key, item.direction)
        _require(key not in by_address, "pair source index block duplicated")
        _require(
            item.query_id == pair_address.query_id
            and item.execution_ordinal == pair_address.execution_ordinal
            and item.candidate_axis_sha256 == pair_address.candidate_axis_sha256,
            "pair source index query binding drift",
        )
        by_address[key] = item
    expected = {
        (member.candidate_key, direction)
        for member in pair_address.members
        for direction in P_DIRECTIONS
    }
    _require(set(by_address) == expected, "pair source index population is not exact")
    blocks: list[PairFeatureBlock] = []
    feature_rows: list[torch.Tensor] = []
    eligibility_rows: list[torch.Tensor] = []
    cursor = 0
    for member_ordinal, member in enumerate(pair_address.members):
        for direction in P_DIRECTIONS:
            index = by_address[(member.candidate_key, direction)]
            candidate = validate_candidate_feature_tensor(
                index, candidate_tensor_fn(index)
            )
            flat = candidate.features.reshape(-1, FEATURE_DIM)
            eligible = candidate.eligibility.reshape(-1)
            block_payload = {
                "member_ordinal": member_ordinal,
                "candidate_key": member.candidate_key,
                "direction": direction,
                "source_population_sha256": index.population_sha256,
                "root_count": index.root_count,
                "action_count": index.action_count,
                "feature_offset": cursor,
                "coordinate_count": index.coordinate_count,
                "feature_tensor_sha256": _tensor_bytes_sha256(flat),
                "eligibility_sha256": _tensor_bytes_sha256(eligible),
            }
            blocks.append(
                PairFeatureBlock(
                    member_ordinal=member_ordinal,
                    candidate_key=member.candidate_key,
                    direction=direction,
                    source_population_sha256=index.population_sha256,
                    root_count=index.root_count,
                    action_count=index.action_count,
                    feature_offset=cursor,
                    coordinate_count=index.coordinate_count,
                    block_sha256=canonical_sha256(block_payload),
                )
            )
            feature_rows.append(flat)
            eligibility_rows.append(eligible)
            cursor += index.coordinate_count
    features = torch.cat(feature_rows, dim=0)
    eligibility = torch.cat(eligibility_rows, dim=0)
    provisional = object.__new__(RoleFreePairFeatureCache)
    object.__setattr__(provisional, "pair_address_sha256", pair_address.pair_sha256)
    object.__setattr__(provisional, "blocks", tuple(blocks))
    object.__setattr__(provisional, "features", features)
    object.__setattr__(provisional, "eligibility", eligibility)
    cache_hash = canonical_sha256(provisional._hash_payload())
    return RoleFreePairFeatureCache(
        pair_address_sha256=pair_address.pair_sha256,
        blocks=tuple(blocks),
        features=features,
        eligibility=eligibility,
        cache_sha256=cache_hash,
    )


def make_role_free_pair_feature_cache_scalar_oracle(
    pair_address: RoleFreePairAddress,
    source_indexes: Sequence[FactorizedDirectionalSourceIndex],
    coordinate_feature_fn: CoordinateFeatureFunction,
) -> RoleFreePairFeatureCache:
    """Independent slow oracle; forbidden as the formal production writer."""

    def produce(
        source_index: FactorizedDirectionalSourceIndex,
    ) -> CandidateFeatureTensor:
        return extract_role_free_candidate_features(
            source_index, coordinate_feature_fn
        )

    return make_role_free_pair_feature_cache(
        pair_address, source_indexes, produce
    )


def deserialize_role_free_pair_feature_cache(
    value: Mapping[str, object],
) -> RoleFreePairFeatureCache:
    """Decode the contiguous ragged tensor cache without deployment objects."""

    _require(
        isinstance(value, Mapping)
        and value.get("schema_version") == ROLE_FREE_PAIR_FEATURE_CACHE_SCHEMA
        and value.get("role_free") is True,
        "pair feature cache payload schema/role drift",
    )
    _assert_role_free(value)
    raw_blocks = value.get("blocks")
    _require(isinstance(raw_blocks, list), "pair feature cache blocks absent")
    blocks: list[PairFeatureBlock] = []
    for raw in raw_blocks:
        _require(isinstance(raw, Mapping), "pair feature cache block row drift")
        blocks.append(
            PairFeatureBlock(
                member_ordinal=int(raw.get("member_ordinal", -1)),
                candidate_key=str(raw.get("candidate_key", "")),
                direction=str(raw.get("direction", "")),
                source_population_sha256=str(
                    raw.get("source_population_sha256", "")
                ),
                root_count=int(raw.get("root_count", -1)),
                action_count=int(raw.get("action_count", -1)),
                feature_offset=int(raw.get("feature_offset", -1)),
                coordinate_count=int(raw.get("coordinate_count", -1)),
                block_sha256=str(raw.get("block_sha256", "")),
            )
        )
    return RoleFreePairFeatureCache(
        pair_address_sha256=str(value.get("pair_address_sha256", "")),
        blocks=tuple(blocks),
        features=torch.as_tensor(value.get("features"), dtype=torch.float64),
        eligibility=torch.as_tensor(value.get("eligibility"), dtype=torch.bool),
        cache_sha256=str(value.get("cache_sha256", "")),
    )


@dataclass(frozen=True)
class VirtualShardEntry:
    shard_ordinal: int
    execution_ordinals: tuple[int, ...]
    artifact_path: str
    artifact_file_sha256: str
    artifact_logical_sha256: str
    validation_path: str
    validation_file_sha256: str
    population_sha256: str
    logical_record_count: int
    logical_coordinate_count: int
    serialized_bytes: int

    def __post_init__(self) -> None:
        _integer(self.shard_ordinal, name="virtual shard ordinal")
        ordinals = tuple(_integer(item, name="virtual shard execution ordinal") for item in self.execution_ordinals)
        _require(ordinals == tuple(sorted(set(ordinals))) and bool(ordinals), "virtual shard ordinals are empty, duplicated, or unordered")
        _require(isinstance(self.artifact_path, str) and self.artifact_path, "virtual shard artifact path absent")
        _require(isinstance(self.validation_path, str) and self.validation_path, "virtual shard validation path absent")
        artifact = Path(self.artifact_path)
        validation = Path(self.validation_path)
        _require(
            not artifact.is_absolute() and ".." not in artifact.parts,
            "virtual shard artifact path must be RC-relative without parent traversal",
        )
        _require(
            not validation.is_absolute() and ".." not in validation.parts,
            "virtual shard validation path must be RC-relative without parent traversal",
        )
        for name, value in (
            ("virtual shard file", self.artifact_file_sha256),
            ("virtual shard logical", self.artifact_logical_sha256),
            ("virtual shard validation", self.validation_file_sha256),
            ("virtual shard population", self.population_sha256),
        ):
            require_sha256(value, name=name)
        _integer(self.logical_record_count, name="virtual shard logical records", minimum=1)
        _integer(
            self.logical_coordinate_count,
            name="virtual shard logical coordinates",
            minimum=1,
        )
        _require(
            self.logical_coordinate_count >= self.logical_record_count,
            "virtual shard coordinate count is smaller than its record count",
        )
        _integer(self.serialized_bytes, name="virtual shard serialized bytes", minimum=1)
        object.__setattr__(self, "execution_ordinals", ordinals)

    def payload(self) -> dict[str, object]:
        return {
            "shard_ordinal": self.shard_ordinal,
            "query_start": self.execution_ordinals[0],
            "query_stop": self.execution_ordinals[-1] + 1,
            "execution_ordinals": list(self.execution_ordinals),
            "execution_ordinals_sha256": canonical_sha256(list(self.execution_ordinals)),
            "artifact_path": self.artifact_path,
            "artifact_file_sha256": self.artifact_file_sha256,
            "artifact_logical_sha256": self.artifact_logical_sha256,
            "validation_path": self.validation_path,
            "validation_file_sha256": self.validation_file_sha256,
            "population_sha256": self.population_sha256,
            "logical_record_count": self.logical_record_count,
            "logical_coordinate_count": self.logical_coordinate_count,
            "serialized_bytes": self.serialized_bytes,
        }


@dataclass(frozen=True)
class VirtualShardCatalog:
    catalog_kind: str
    expected_execution_ordinals: tuple[int, ...]
    candidate_count_per_query: int
    direction_count: int
    shards: tuple[VirtualShardEntry, ...]
    catalog_sha256: str

    def __post_init__(self) -> None:
        _require(self.catalog_kind in {SOURCE_INDEX_KIND, PAIR_FEATURE_CACHE_KIND}, "virtual catalog kind drift")
        expected = tuple(_integer(item, name="catalog expected execution ordinal") for item in self.expected_execution_ordinals)
        _require(expected == tuple(sorted(set(expected))) and bool(expected), "virtual catalog expected population is empty, duplicated, or unordered")
        _require(self.candidate_count_per_query == NATURAL_CANDIDATE_COUNT, "virtual catalog lost natural C128 axis")
        _require(self.direction_count == len(P_DIRECTIONS), "virtual catalog direction count drift")
        shards = tuple(self.shards)
        _require(
            [item.shard_ordinal for item in shards] == list(range(len(shards))),
            "virtual shard ordinal order drift",
        )
        flattened = tuple(item for shard in shards for item in shard.execution_ordinals)
        _require(flattened == expected, "virtual shard catalog has a query gap, overlap, or reorder")
        records_per_query = (
            NATURAL_CANDIDATE_COUNT * len(P_DIRECTIONS)
            if self.catalog_kind == SOURCE_INDEX_KIND
            else 2 * len(P_DIRECTIONS)
        )
        for shard in shards:
            _require(
                shard.logical_record_count
                == len(shard.execution_ordinals) * records_per_query,
                "virtual shard logical record population drift for catalog kind",
            )
        _require(
            self.catalog_sha256 == canonical_sha256(self.payload(include_hash=False)),
            "virtual shard catalog hash drift",
        )
        object.__setattr__(self, "expected_execution_ordinals", expected)
        object.__setattr__(self, "shards", shards)

    def payload(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": VIRTUAL_SHARD_CATALOG_SCHEMA,
            "role_free": True,
            "target_free_source": self.catalog_kind == SOURCE_INDEX_KIND,
            "virtual_aggregate": True,
            "copies_feature_tensors": 0,
            "copies_deployment_masks": 0,
            "catalog_kind": self.catalog_kind,
            "expected_execution_ordinals": list(self.expected_execution_ordinals),
            "query_count": len(self.expected_execution_ordinals),
            "candidate_count_per_query": self.candidate_count_per_query,
            "direction_count": self.direction_count,
            "shards": [item.payload() for item in self.shards],
            "logical_record_count": sum(
                item.logical_record_count for item in self.shards
            ),
            "logical_coordinate_count": sum(
                item.logical_coordinate_count for item in self.shards
            ),
        }
        if include_hash:
            value["catalog_sha256"] = self.catalog_sha256
        _assert_role_free(value)
        return value


def make_virtual_shard_catalog(
    *,
    catalog_kind: str,
    expected_execution_ordinals: Sequence[int],
    shards: Sequence[VirtualShardEntry],
) -> VirtualShardCatalog:
    expected = tuple(expected_execution_ordinals)
    ordered = tuple(shards)
    provisional = {
        "schema_version": VIRTUAL_SHARD_CATALOG_SCHEMA,
        "role_free": True,
        "target_free_source": catalog_kind == SOURCE_INDEX_KIND,
        "virtual_aggregate": True,
        "copies_feature_tensors": 0,
        "copies_deployment_masks": 0,
        "catalog_kind": catalog_kind,
        "expected_execution_ordinals": list(expected),
        "query_count": len(expected),
        "candidate_count_per_query": NATURAL_CANDIDATE_COUNT,
        "direction_count": len(P_DIRECTIONS),
        "shards": [item.payload() for item in ordered],
        "logical_record_count": sum(
            item.logical_record_count for item in ordered
        ),
        "logical_coordinate_count": sum(
            item.logical_coordinate_count for item in ordered
        ),
    }
    return VirtualShardCatalog(
        catalog_kind=catalog_kind,
        expected_execution_ordinals=expected,
        candidate_count_per_query=NATURAL_CANDIDATE_COUNT,
        direction_count=len(P_DIRECTIONS),
        shards=ordered,
        catalog_sha256=canonical_sha256(provisional),
    )


def validate_virtual_shard_catalog(
    value: VirtualShardCatalog,
    *,
    expected_kind: str,
    expected_execution_ordinals: Sequence[int],
) -> dict[str, object]:
    """Independently recheck interval coverage without opening shard payloads."""

    _require(isinstance(value, VirtualShardCatalog), "virtual catalog type drift")
    _require(value.catalog_kind == expected_kind, "virtual catalog expected kind drift")
    expected = tuple(expected_execution_ordinals)
    _require(value.expected_execution_ordinals == expected, "virtual catalog expected population drift")
    seen: set[int] = set()
    previous: int | None = None
    for shard in value.shards:
        for execution in shard.execution_ordinals:
            _require(execution not in seen, "virtual catalog query overlap")
            if previous is not None:
                _require(execution > previous, "virtual catalog query reorder")
            seen.add(execution)
            previous = execution
    _require(seen == set(expected) and len(seen) == len(expected), "virtual catalog query gap")
    records_per_query = (
        NATURAL_CANDIDATE_COUNT * len(P_DIRECTIONS)
        if expected_kind == SOURCE_INDEX_KIND
        else 2 * len(P_DIRECTIONS)
    )
    _require(
        sum(item.logical_record_count for item in value.shards)
        == len(expected) * records_per_query,
        "virtual catalog logical record population drift",
    )
    payload = value.payload()
    _require(
        not any(isinstance(item, torch.Tensor) for item in _walk(payload)),
        "virtual catalog copied a tensor",
    )
    return {
        "schema_version": VIRTUAL_SHARD_CATALOG_SCHEMA,
        "validation_pass": True,
        "catalog_sha256": value.catalog_sha256,
        "query_count": len(expected),
        "shard_count": len(value.shards),
        "gap_count": 0,
        "overlap_count": 0,
        "copied_tensor_count": 0,
        "logical_record_count": sum(
            item.logical_record_count for item in value.shards
        ),
        "logical_coordinate_count": sum(
            item.logical_coordinate_count for item in value.shards
        ),
    }


def _walk(value: object):
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)
