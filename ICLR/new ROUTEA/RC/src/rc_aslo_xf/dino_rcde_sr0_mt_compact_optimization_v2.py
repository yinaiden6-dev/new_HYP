"""Exact-output performance helpers for compact source-index V2.

The frozen V1 implementations remain the semantic oracle.  This module only
removes repeated deterministic work:

* tensor digests are memoized by immutable storage identity and mutation
  version for the lifetime of one shard;
* query macro roots and their two directional masks are built once per query;
* reference action banks are memoized by their complete structural input; and
* canonical rasterization is memoized by source/destination geometry and mask.

No score, mask, action, binding, or population-hash formula is reimplemented.
In particular, every candidate/root/action still passes through the frozen
``_make_root_binding`` function.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Sequence

import torch

from . import dino_rcde_cw1_multitile_pseal_e1_v1 as pseal_v1
from . import dino_rcde_cw1_multitile_superregion_v2 as superregion_v2
from . import dino_rcde_sr0_mt_p_compact_catalog_v1 as catalog_v1
from . import dino_rcde_sr0_mt_p_natural_source_v1 as natural_v1
from .cw0_connected_region_v2 import enumerate_query_macro_seeds
from .dino_rcde_cw1_multitile_pseal_sources_v1 import (
    CandidateReferenceSourceV1,
    ColNomicTokenSourceV1,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    FormalPContractError,
    assert_no_forbidden_prejoin_keys,
)


@dataclass
class CompactOptimizationStatsV2:
    tensor_hash_requests: int = 0
    tensor_hash_hits: int = 0
    superregion_tensor_hash_requests: int = 0
    superregion_tensor_hash_hits: int = 0
    rasterization_requests: int = 0
    rasterization_hits: int = 0
    reference_bank_requests: int = 0
    reference_bank_hits: int = 0
    query_root_bank_builds: int = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalPContractError(message)


def _storage_key(value: torch.Tensor) -> tuple[object, ...]:
    """Address one immutable tensor view without reading its payload.

    The mutation version prevents reuse after an in-place torch operation.  A
    strong reference is retained in the cache, so allocator address reuse
    cannot alias an older entry during one shard.  Natural source tensors are
    CPU/contiguous; a non-contiguous or non-CPU input is normalized before this
    function is called.
    """

    storage = value.untyped_storage()
    return (
        value.device.type,
        value.device.index,
        str(value.dtype),
        tuple(value.shape),
        tuple(value.stride()),
        int(value.storage_offset()),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(getattr(value, "_version", 0)),
    )


class _TensorDigestMemo:
    def __init__(
        self,
        original: Callable[[torch.Tensor], str],
        *,
        stats: CompactOptimizationStatsV2,
        superregion: bool = False,
    ) -> None:
        self._original = original
        self._stats = stats
        self._superregion = superregion
        self._cache: dict[tuple[object, ...], tuple[torch.Tensor, str]] = {}

    def __call__(self, value: torch.Tensor) -> str:
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        key = _storage_key(tensor)
        if self._superregion:
            self._stats.superregion_tensor_hash_requests += 1
        else:
            self._stats.tensor_hash_requests += 1
        cached = self._cache.get(key)
        if cached is not None:
            if self._superregion:
                self._stats.superregion_tensor_hash_hits += 1
            else:
                self._stats.tensor_hash_hits += 1
            return cached[1]
        digest = self._original(tensor)
        self._cache[key] = (tensor, digest)
        return digest


class CompactOptimizationRuntimeV2:
    """Shard-local, exact deterministic memoization runtime."""

    def __init__(self) -> None:
        self.stats = CompactOptimizationStatsV2()
        self.tensor_sha256 = _TensorDigestMemo(
            catalog_v1.tensor_sha256, stats=self.stats
        )
        self.superregion_tensor_sha256 = _TensorDigestMemo(
            superregion_v2._tensor_sha,
            stats=self.stats,
            superregion=True,
        )
        self._reference_banks: dict[
            tuple[tuple[int, int], str, bytes], tuple[object, str]
        ] = {}
        self._rasters: dict[
            tuple[str, str, str], torch.Tensor
        ] = {}
        self._original_rasterize = superregion_v2.rasterize_by_canonical_overlap

    def reference_action_bank(
        self,
        *,
        grid_shape: tuple[int, int],
        valid_patch_mask: torch.Tensor,
        direction: str,
    ) -> tuple[object, str]:
        valid = (
            torch.as_tensor(valid_patch_mask, dtype=torch.bool)
            .detach()
            .cpu()
            .contiguous()
        )
        key = (
            tuple(int(item) for item in grid_shape),
            direction,
            valid.to(torch.uint8).numpy().tobytes(),
        )
        self.stats.reference_bank_requests += 1
        cached = self._reference_banks.get(key)
        if cached is not None:
            self.stats.reference_bank_hits += 1
            return cached
        value = pseal_v1._reference_action_bank(
            grid_shape=grid_shape,
            valid_patch_mask=valid,
            direction=direction,
        )
        self._reference_banks[key] = value
        return value

    def rasterize_by_canonical_overlap(
        self,
        mask: torch.Tensor,
        *,
        source_geometry: object,
        destination_geometry: object,
    ) -> torch.Tensor:
        value = (
            torch.as_tensor(mask, dtype=torch.bool)
            .detach()
            .cpu()
            .contiguous()
        )
        source_sha = str(getattr(source_geometry, "sha256"))
        destination_sha = str(getattr(destination_geometry, "sha256"))
        mask_sha = self.tensor_sha256(value)
        key = (source_sha, destination_sha, mask_sha)
        self.stats.rasterization_requests += 1
        cached = self._rasters.get(key)
        if cached is not None:
            self.stats.rasterization_hits += 1
            return cached.clone()
        mapped = self._original_rasterize(
            value,
            source_geometry=source_geometry,
            destination_geometry=destination_geometry,
        ).detach().cpu().contiguous()
        self._rasters[key] = mapped
        return mapped.clone()

    @contextmanager
    def patched_frozen_helpers(self) -> Iterator[None]:
        """Install exact memoized wrappers in the frozen functions' globals."""

        original_catalog_hash = catalog_v1.tensor_sha256
        original_superregion_hash = superregion_v2._tensor_sha
        original_rasterize = superregion_v2.rasterize_by_canonical_overlap
        catalog_v1.tensor_sha256 = self.tensor_sha256
        superregion_v2._tensor_sha = self.superregion_tensor_sha256
        superregion_v2.rasterize_by_canonical_overlap = (
            self.rasterize_by_canonical_overlap
        )
        try:
            yield
        finally:
            catalog_v1.tensor_sha256 = original_catalog_hash
            superregion_v2._tensor_sha = original_superregion_hash
            superregion_v2.rasterize_by_canonical_overlap = original_rasterize


def _query_directional_roots(
    query: ColNomicTokenSourceV1,
) -> dict[str, tuple[torch.Tensor, ...]]:
    roots = enumerate_query_macro_seeds(query.colnomic_geometry.grid_shape)
    full_roots = tuple(
        (
            root.window.mask & query.colnomic_geometry.valid_patch_mask
        ).detach().cpu().contiguous()
        for root in roots
    )
    return {
        direction: tuple(
            pseal_v1._direction_mask(
                root, query.colnomic_geometry.grid_shape, direction
            )
            for root in full_roots
        )
        for direction in P_DIRECTIONS
    }


def _candidate_directions_optimized(
    *,
    candidate_key: str,
    query: ColNomicTokenSourceV1,
    reference: ColNomicTokenSourceV1,
    directional_query_roots: Mapping[str, Sequence[torch.Tensor]],
    runtime: CompactOptimizationRuntimeV2,
) -> dict[str, list[dict[str, object]]]:
    """Exact V1 candidate directions with query-only work hoisted outward."""

    _require(query.kind == "query", "P natural query source kind drift")
    _require(reference.kind == "reference", "P natural reference source kind drift")
    output: dict[str, list[dict[str, object]]] = {}
    for direction in P_DIRECTIONS:
        action_bank_raw, action_bank_sha256 = runtime.reference_action_bank(
            grid_shape=reference.colnomic_geometry.grid_shape,
            valid_patch_mask=reference.colnomic_geometry.valid_patch_mask,
            direction=direction,
        )
        action_bank = tuple(action_bank_raw)  # type: ignore[arg-type]
        query_roots = tuple(directional_query_roots[direction])
        root_rows: list[dict[str, object]] = []
        for root_ordinal, directional_query_root in enumerate(query_roots):
            actions: list[dict[str, object]] = []
            if not action_bank:
                missing = superregion_v2._make_root_binding(
                    candidate_key=candidate_key,
                    direction=direction,
                    root_ordinal=root_ordinal,
                    source_reference_component=None,
                    colnomic_query_geometry=query.colnomic_geometry,
                    colnomic_reference_geometry=reference.colnomic_geometry,
                    dino_query_geometry=query.dino_geometry,
                    dino_reference_geometry=reference.dino_geometry,
                )
                actions.append(
                    {
                        "action_key": natural_v1._action_key(
                            candidate_key=candidate_key,
                            direction=direction,
                            root_ordinal=root_ordinal,
                            action_ordinal=0,
                            structural_action_sha256=action_bank_sha256,
                            query_geometry_sha256=query.dino_geometry.sha256,
                            reference_geometry_sha256=reference.dino_geometry.sha256,
                        ),
                        "colnomic_query_action_mask": torch.zeros_like(
                            directional_query_root
                        ),
                        "colnomic_reference_action_mask": torch.zeros_like(
                            reference.colnomic_geometry.valid_patch_mask
                        ),
                        "deployment_eligible": False,
                        "deployment_query_mask": torch.zeros_like(
                            missing.dino_query_tile_mask
                        ),
                        "deployment_reference_mask": torch.zeros_like(
                            missing.dino_reference_component_mask
                        ),
                    }
                )
            else:
                for action in action_bank:
                    binding = superregion_v2._make_root_binding(
                        candidate_key=candidate_key,
                        direction=direction,
                        root_ordinal=root_ordinal,
                        source_reference_component=action.component_mask,
                        colnomic_query_geometry=query.colnomic_geometry,
                        colnomic_reference_geometry=reference.colnomic_geometry,
                        dino_query_geometry=query.dino_geometry,
                        dino_reference_geometry=reference.dino_geometry,
                    )
                    deployment_eligible = binding.status == superregion_v2.ROOT_READY
                    actions.append(
                        {
                            "action_key": natural_v1._action_key(
                                candidate_key=candidate_key,
                                direction=direction,
                                root_ordinal=root_ordinal,
                                action_ordinal=action.action_ordinal,
                                structural_action_sha256=action.action_sha256,
                                query_geometry_sha256=query.dino_geometry.sha256,
                                reference_geometry_sha256=reference.dino_geometry.sha256,
                            ),
                            "colnomic_query_action_mask": directional_query_root,
                            "colnomic_reference_action_mask": action.selector_mask,
                            "deployment_eligible": deployment_eligible,
                            "deployment_query_mask": (
                                binding.dino_query_tile_mask
                                if deployment_eligible
                                else torch.zeros_like(binding.dino_query_tile_mask)
                            ),
                            "deployment_reference_mask": (
                                binding.dino_reference_component_mask
                                if deployment_eligible
                                else torch.zeros_like(
                                    binding.dino_reference_component_mask
                                )
                            ),
                        }
                    )
            root_rows.append(
                {
                    "root_ordinal": root_ordinal,
                    "reference_action_bank_sha256": action_bank_sha256,
                    "actions": actions,
                }
            )
        output[direction] = root_rows
    return output


def build_target_free_raw_query_optimized(
    *,
    query_id: str,
    historical_query_ordinal: int,
    execution_ordinal: int,
    heldout_fold: int,
    candidate_axis_sha256: str,
    query: ColNomicTokenSourceV1,
    candidates: Sequence[CandidateReferenceSourceV1],
    expected_candidate_count: int = natural_v1.EXPECTED_CANDIDATE_COUNT,
    runtime: CompactOptimizationRuntimeV2,
) -> dict[str, object]:
    """Build the byte-equivalent raw query while hoisting query-only work."""

    _require(isinstance(query_id, str) and bool(query_id), "query id absent")
    _require(
        heldout_fold in natural_v1.EXPECTED_OUTER_FOLDS,
        "target-free source fold is outside frozen outer folds",
    )
    _require(
        query.kind == "query" and query.native_key == historical_query_ordinal,
        "query source address drift",
    )
    ordered = tuple(candidates)
    _require(
        len(ordered) == expected_candidate_count
        and tuple(item.candidate_position for item in ordered)
        == tuple(range(expected_candidate_count))
        and len({item.physical_row for item in ordered}) == expected_candidate_count,
        "natural candidate axis is incomplete, reordered, or duplicated",
    )
    runtime.stats.query_root_bank_builds += 1
    directional_query_roots = _query_directional_roots(query)
    candidate_rows: list[dict[str, object]] = []
    for item in ordered:
        reference = item.source
        key = natural_v1.candidate_key_v1(
            physical_row=item.physical_row,
            source_image_sha256=reference.source_image_sha256,
        )
        candidate_rows.append(
            {
                "candidate_position": item.candidate_position,
                "candidate_key": key,
                "candidate_physical_row": item.physical_row,
                "candidate_reference_source_sha256": reference.source_image_sha256,
                "reference_tokens": reference.tokens,
                "reference_valid_patch_mask": (
                    reference.colnomic_geometry.valid_patch_mask
                ),
                "colnomic_reference_grid_shape": list(reference.grid_shape),
                "dino_reference_grid_shape": list(
                    reference.dino_geometry.grid_shape
                ),
                "reference_geometry_sha256": reference.dino_geometry.sha256,
                "directions": _candidate_directions_optimized(
                    candidate_key=key,
                    query=query,
                    reference=reference,
                    directional_query_roots=directional_query_roots,
                    runtime=runtime,
                ),
            }
        )
    result: dict[str, object] = {
        "query_id": query_id,
        "historical_query_ordinal": historical_query_ordinal,
        "execution_ordinal": execution_ordinal,
        "query_source_image_sha256": query.source_image_sha256,
        "source_fold": heldout_fold,
        "candidate_axis_sha256": candidate_axis_sha256,
        "query_tokens": query.tokens,
        "query_valid_patch_mask": query.colnomic_geometry.valid_patch_mask,
        "colnomic_query_grid_shape": list(query.grid_shape),
        "dino_query_grid_shape": list(query.dino_geometry.grid_shape),
        "query_geometry_sha256": query.dino_geometry.sha256,
        "candidates": candidate_rows,
    }
    assert_no_forbidden_prejoin_keys(result)
    return result


__all__ = [
    "CompactOptimizationRuntimeV2",
    "CompactOptimizationStatsV2",
    "build_target_free_raw_query_optimized",
]
