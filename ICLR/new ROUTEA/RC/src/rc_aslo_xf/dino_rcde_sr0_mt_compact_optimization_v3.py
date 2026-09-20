"""Exact separable binding optimization for compact source-index V3.

V2 already caches hashes, query directional masks, action banks and canonical
rasterization.  V3 additionally exploits the proven separability of the frozen
``_make_root_binding`` oracle:

* query deployment eligibility/mask depends only on query geometry and root;
* reference deployment eligibility/mask depends only on reference geometry and
  the unique reference action component; and
* the Cartesian coordinate is READY iff both independent components are READY.

The frozen oracle remains the only implementation of either component.  V3
does not reimplement rasterization, component legality, action keys, binding
rules, or any factorized population/hash formula.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from . import dino_rcde_cw1_multitile_superregion_v2 as superregion_v2
from . import dino_rcde_sr0_mt_compact_optimization_v2 as v2
from . import dino_rcde_sr0_mt_p_natural_source_v1 as natural_v1
from .cw0_connected_region_v2 import enumerate_query_macro_seeds
from .dino_rcde_cw1_multitile_pseal_sources_v1 import (
    CandidateReferenceSourceV1,
    ColNomicTokenSourceV1,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    assert_no_forbidden_prejoin_keys,
)


class CompactOptimizationRuntimeV3(v2.CompactOptimizationRuntimeV2):
    """V2 memoization plus exact reference-component oracle caching."""

    def __init__(self) -> None:
        super().__init__()
        self.stats.query_binding_oracle_calls = 0
        self.stats.reference_binding_requests = 0
        self.stats.reference_binding_hits = 0
        self._reference_deployments: dict[
            tuple[str, str, str], tuple[bool, torch.Tensor]
        ] = {}

    def reference_deployment(
        self,
        *,
        candidate_key: str,
        direction: str,
        root_ordinal: int,
        component: torch.Tensor,
        query: ColNomicTokenSourceV1,
        reference: ColNomicTokenSourceV1,
    ) -> tuple[bool, torch.Tensor]:
        component_value = (
            torch.as_tensor(component, dtype=torch.bool)
            .detach()
            .cpu()
            .contiguous()
        )
        key = (
            reference.colnomic_geometry.sha256,
            reference.dino_geometry.sha256,
            self.tensor_sha256(component_value),
        )
        self.stats.reference_binding_requests += 1
        cached = self._reference_deployments.get(key)
        if cached is not None:
            self.stats.reference_binding_hits += 1
            return cached[0], cached[1].clone()
        binding = superregion_v2._make_root_binding(
            candidate_key=candidate_key,
            direction=direction,
            root_ordinal=root_ordinal,
            source_reference_component=component_value,
            colnomic_query_geometry=query.colnomic_geometry,
            colnomic_reference_geometry=reference.colnomic_geometry,
            dino_query_geometry=query.dino_geometry,
            dino_reference_geometry=reference.dino_geometry,
        )
        ready = binding.status == superregion_v2.ROOT_READY
        v2._require(
            ready
            or str(binding.reason).startswith("REFERENCE_MAPPING_INELIGIBLE:"),
            "reference component oracle was confounded by query eligibility",
        )
        deployment = binding.dino_reference_component_mask.detach().cpu().contiguous()
        self._reference_deployments[key] = (ready, deployment)
        return ready, deployment.clone()


def _query_deployment_roots(
    *,
    query: ColNomicTokenSourceV1,
    representative_reference: ColNomicTokenSourceV1,
    representative_candidate_key: str,
    runtime: CompactOptimizationRuntimeV3,
) -> tuple[tuple[bool, torch.Tensor], ...]:
    """Ask the frozen oracle exactly once for every query macro root."""

    roots = enumerate_query_macro_seeds(query.colnomic_geometry.grid_shape)
    output: list[tuple[bool, torch.Tensor]] = []
    for root_ordinal in range(len(roots)):
        runtime.stats.query_binding_oracle_calls += 1
        binding = superregion_v2._make_root_binding(
            candidate_key=representative_candidate_key,
            direction=P_DIRECTIONS[0],
            root_ordinal=root_ordinal,
            source_reference_component=None,
            colnomic_query_geometry=query.colnomic_geometry,
            colnomic_reference_geometry=representative_reference.colnomic_geometry,
            dino_query_geometry=query.dino_geometry,
            dino_reference_geometry=representative_reference.dino_geometry,
        )
        ready = binding.reason == "NO_CANDIDATE_BOUND_COMPONENT"
        v2._require(
            binding.status == superregion_v2.ROOT_MISSING
            and (
                ready
                or str(binding.reason).startswith("QUERY_MAPPING_INELIGIBLE:")
            ),
            "query component oracle returned an unexpected state",
        )
        output.append(
            (ready, binding.dino_query_tile_mask.detach().cpu().contiguous())
        )
    return tuple(output)


def _candidate_directions_optimized(
    *,
    candidate_key: str,
    query: ColNomicTokenSourceV1,
    reference: ColNomicTokenSourceV1,
    directional_query_roots: Mapping[str, Sequence[torch.Tensor]],
    query_deployment_roots: Sequence[tuple[bool, torch.Tensor]],
    runtime: CompactOptimizationRuntimeV3,
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for direction in P_DIRECTIONS:
        action_bank_raw, action_bank_sha256 = runtime.reference_action_bank(
            grid_shape=reference.colnomic_geometry.grid_shape,
            valid_patch_mask=reference.colnomic_geometry.valid_patch_mask,
            direction=direction,
        )
        action_bank = tuple(action_bank_raw)  # type: ignore[arg-type]
        query_roots = tuple(directional_query_roots[direction])
        deployment_roots = tuple(query_deployment_roots)
        v2._require(
            len(query_roots) == len(deployment_roots),
            "query directional/deployment root axis drift",
        )
        first_ready_root = next(
            (
                ordinal
                for ordinal, (ready, _mask) in enumerate(deployment_roots)
                if ready
            ),
            None,
        )
        if first_ready_root is None:
            reference_deployments = [
                (
                    False,
                    torch.zeros_like(reference.dino_geometry.valid_patch_mask),
                )
                for _action in action_bank
            ]
        else:
            reference_deployments = [
                runtime.reference_deployment(
                    candidate_key=candidate_key,
                    direction=direction,
                    root_ordinal=first_ready_root,
                    component=action.component_mask,
                    query=query,
                    reference=reference,
                )
                for action in action_bank
            ]
        root_rows: list[dict[str, object]] = []
        for root_ordinal, directional_query_root in enumerate(query_roots):
            query_ready, deployment_query_root = deployment_roots[root_ordinal]
            actions: list[dict[str, object]] = []
            if not action_bank:
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
                            deployment_query_root
                        ),
                        "deployment_reference_mask": torch.zeros_like(
                            reference.dino_geometry.valid_patch_mask
                        ),
                    }
                )
            else:
                for action, (reference_ready, deployment_reference) in zip(
                    action_bank, reference_deployments
                ):
                    deployment_eligible = query_ready and reference_ready
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
                                deployment_query_root
                                if deployment_eligible
                                else torch.zeros_like(deployment_query_root)
                            ),
                            "deployment_reference_mask": (
                                deployment_reference
                                if deployment_eligible
                                else torch.zeros_like(deployment_reference)
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
    runtime: CompactOptimizationRuntimeV3,
) -> dict[str, object]:
    """Exact V1 raw query with separable frozen-binding oracle calls."""

    v2._require(isinstance(query_id, str) and bool(query_id), "query id absent")
    v2._require(
        heldout_fold in natural_v1.EXPECTED_OUTER_FOLDS,
        "target-free source fold is outside frozen outer folds",
    )
    v2._require(
        query.kind == "query" and query.native_key == historical_query_ordinal,
        "query source address drift",
    )
    ordered = tuple(candidates)
    v2._require(
        len(ordered) == expected_candidate_count
        and tuple(item.candidate_position for item in ordered)
        == tuple(range(expected_candidate_count))
        and len({item.physical_row for item in ordered}) == expected_candidate_count,
        "natural candidate axis is incomplete, reordered, or duplicated",
    )
    runtime.stats.query_root_bank_builds += 1
    directional_query_roots = v2._query_directional_roots(query)
    representative = ordered[0].source
    representative_key = natural_v1.candidate_key_v1(
        physical_row=ordered[0].physical_row,
        source_image_sha256=representative.source_image_sha256,
    )
    query_deployment_roots = _query_deployment_roots(
        query=query,
        representative_reference=representative,
        representative_candidate_key=representative_key,
        runtime=runtime,
    )
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
                "reference_valid_patch_mask": reference.colnomic_geometry.valid_patch_mask,
                "colnomic_reference_grid_shape": list(reference.grid_shape),
                "dino_reference_grid_shape": list(reference.dino_geometry.grid_shape),
                "reference_geometry_sha256": reference.dino_geometry.sha256,
                "directions": _candidate_directions_optimized(
                    candidate_key=key,
                    query=query,
                    reference=reference,
                    directional_query_roots=directional_query_roots,
                    query_deployment_roots=query_deployment_roots,
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
    "CompactOptimizationRuntimeV3",
    "build_target_free_raw_query_optimized",
]
