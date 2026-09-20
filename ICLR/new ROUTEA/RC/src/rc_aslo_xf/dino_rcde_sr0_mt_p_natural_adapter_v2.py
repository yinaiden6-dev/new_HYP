"""Natural compact-source adapter for the successor V2 P-lock.

The adapter is additive and never reads the V1 MAP/H0 decision.  It rebuilds
structural row legality from query masks, derives reference binding state from
the selected action, applies sign-independent row selection, and seals a
score-erased natural provenance wrapper around the qualified V2 core lock.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch

from .cw1_sr0_structure_v1 import (
    enumerate_superregion_bank,
    superregion_bank_sha256,
    superregion_sha256,
)
from .dino_rcde_sr0_mt_p_lock_v2 import (
    CANONICAL_ARM,
    LOCK_PROPOSAL_READY,
    LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE,
    PLockV2ContractError,
    candidate_p_lock_v2_from_record,
    candidate_p_lock_v2_to_record,
    canonical_sha256,
    make_candidate_p_lock_v2,
    mask_geometry_v2,
    tensor_sha256,
)
from .dino_rcde_sr0_mt_p_selector_v2 import (
    CompactProposalRowV2,
    CompactRootObservationV2,
    select_compact_p_proposal_v2,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_natural_lock_adapter_v2_20260818"
CLAIM_LEVEL = "ENGINEERING_NATURAL_COMPACT_ADAPTER_E1_ONLY_NOT_CONSUMABLE_P_LOCK"
ERASED_FIELDS = (
    "p_score",
    "utility",
    "confidence",
    "posterior",
    "target",
    "rival",
    "identity",
    "supergroup",
    "rank",
    "slot",
    "winner",
    "correctness",
    "d1_gap",
    "retrieval_outcome",
    "loss",
)


class NaturalAdapterV2Error(PLockV2ContractError):
    """A natural source/provenance or V2 adapter invariant failed."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise NaturalAdapterV2Error(message)


def _sha(value: Any, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{name} SHA drift",
    )
    return value


def _grid(value: Any, name: str) -> tuple[int, int]:
    require(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(item) is int and item > 0 for item in value),
        f"{name} grid drift",
    )
    return int(value[0]), int(value[1])


def _decode_mask(value: Mapping[str, Any], *, expected_grid: tuple[int, int], name: str) -> torch.Tensor:
    require(isinstance(value, Mapping) and set(value) >= {"grid_shape", "rle"}, f"{name} mask schema drift")
    grid = _grid(value["grid_shape"], name)
    require(grid == expected_grid, f"{name} grid mismatch")
    result = torch.zeros(math.prod(grid), dtype=torch.bool)
    previous_stop = 0
    rle = value["rle"]
    require(isinstance(rle, list), f"{name} RLE drift")
    for run in rle:
        require(
            isinstance(run, list)
            and len(run) == 2
            and all(type(item) is int for item in run),
            f"{name} RLE run drift",
        )
        start, length = run
        require(start >= previous_stop and length > 0 and start + length <= result.numel(), f"{name} RLE bounds/order drift")
        result[start : start + length] = True
        previous_stop = start + length
    return result


def _decode_eligibility(value: Mapping[str, Any], *, roots: int, actions: int) -> torch.Tensor:
    require(isinstance(value, Mapping) and value.get("shape") == [roots, actions], "eligibility shape drift")
    require(value.get("flatten_order") == "ROW_MAJOR", "eligibility flatten order drift")
    flat = torch.zeros(roots * actions, dtype=torch.bool)
    previous_stop = 0
    for run in value.get("true_runs", []):
        require(isinstance(run, list) and len(run) == 2 and all(type(item) is int for item in run), "eligibility RLE drift")
        start, length = run
        require(start >= previous_stop and length > 0 and start + length <= flat.numel(), "eligibility RLE bounds drift")
        flat[start : start + length] = True
        previous_stop = start + length
    return flat.reshape(roots, actions)


def _component_legal(mask: torch.Tensor, grid: tuple[int, int]) -> bool:
    image = torch.as_tensor(mask, dtype=torch.bool).reshape(grid)
    coordinates = torch.nonzero(image, as_tuple=False)
    if coordinates.shape[0] < 4:
        return False
    if int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1) < 2:
        return False
    if int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1) < 2:
        return False
    active = {(int(row), int(col)) for row, col in coordinates.tolist()}
    stack = [next(iter(active))]
    visited: set[tuple[int, int]] = set()
    while stack:
        row, col = stack.pop()
        if (row, col) in visited:
            continue
        visited.add((row, col))
        for item in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if item in active and item not in visited:
                stack.append(item)
    return visited == active


def _structural_rows(
    colnomic_query_grid: tuple[int, int],
    deployment_grid: tuple[int, int],
    query_masks: Sequence[torch.Tensor],
    row_scores: torch.Tensor,
) -> tuple[tuple[CompactProposalRowV2, ...], tuple[Any, ...]]:
    bank = tuple(enumerate_superregion_bank(colnomic_query_grid, include_r0_control=False))
    require(bank and int(bank[0].tile_weights.numel()) == len(query_masks), "CW1 bank/root population drift")
    scores = torch.as_tensor(row_scores)
    require(scores.ndim == 1 and scores.shape[0] == len(bank) and scores.is_floating_point() and bool(torch.isfinite(scores).all()), "row-score fixture shape/value drift")
    rows: list[CompactProposalRowV2] = []
    for ordinal, (region, score) in enumerate(zip(bank, scores, strict=True)):
        roots = tuple(int(value) for value in region.contributing_root_ordinals)
        masks = [query_masks[root] for root in roots]
        mapped = all(bool(mask.any()) for mask in masks)
        union = torch.stack(masks).any(dim=0) if mapped else torch.zeros(math.prod(deployment_grid), dtype=torch.bool)
        legal = mapped and _component_legal(union, deployment_grid) and int(union.sum()) > max(int(mask.sum()) for mask in masks)
        rows.append(
            CompactProposalRowV2(
                canonical_bank_ordinal=ordinal,
                structural_row_sha256=superregion_sha256(region),
                constituent_root_ordinals=roots,
                structurally_legal=bool(legal),
                structural_rejection_reason=None if legal else "QUERY_STRUCTURAL_ROW_ILLEGAL",
                score=score,
            )
        )
    return tuple(rows), bank


def build_natural_p_lock_v2(
    *,
    source_record: Mapping[str, Any],
    query_geometry_record: Mapping[str, Any],
    reference_geometry_record: Mapping[str, Any],
    fold_record: Mapping[str, Any],
    membership: Mapping[str, Any],
    row_scores: torch.Tensor,
    selected_action_indices: Sequence[int | None],
    fit_id: str,
    crossfit_role: str,
    p_checkpoint_sha256: str,
    p_training_manifest_sha256: str,
) -> dict[str, Any]:
    """Build one complete score-erased natural V2 adapter record."""

    require(source_record.get("query_id") == query_geometry_record.get("query_id") == fold_record.get("query_id"), "query ID join drift")
    execution = source_record.get("execution_ordinal")
    require(type(execution) is int and execution == query_geometry_record.get("execution_ordinal"), "execution ordinal join drift")
    require(source_record.get("query_source_image_sha256") == query_geometry_record.get("source_image_sha256") == fold_record.get("source_image_sha256"), "query source join drift")
    require(fit_id == membership.get("fit_id"), "fit ID drift")
    heldout = membership.get("heldout_addresses")
    require(isinstance(heldout, list) and any(item.get("execution_ordinal") == execution and item.get("query_id") == source_record.get("query_id") for item in heldout), "fixture is not heldout by membership")
    outer_fold = membership.get("outer_fold")
    require(type(outer_fold) is int and 1 <= outer_fold <= 4, "outer fold drift")
    if crossfit_role == "INNER_HELDOUT_DEPLOYMENT":
        require(
            fold_record.get("inner_fold") == membership.get("inner_heldout_fold")
            and membership.get("inner_heldout_fold") is not None
            and outer_fold != fold_record.get("inner_fold"),
            "inner fold/membership drift",
        )
    elif crossfit_role == "OUTER_HELDOUT_DEPLOYMENT":
        require(
            membership.get("inner_heldout_fold") is None
            and outer_fold == fold_record.get("inner_fold"),
            "outer fold/membership drift",
        )
    else:
        raise NaturalAdapterV2Error("formal crossfit role drift")
    _sha(p_checkpoint_sha256, "P checkpoint")
    _sha(p_training_manifest_sha256, "P training manifest")

    candidate_row = source_record.get("candidate_physical_row")
    require(type(candidate_row) is int and candidate_row == reference_geometry_record.get("physical_row"), "candidate physical-row join drift")
    require(source_record.get("candidate_reference_source_sha256") == reference_geometry_record.get("source_image_sha256"), "candidate source join drift")
    q_geometry = query_geometry_record.get("dino_geometry")
    r_geometry = reference_geometry_record.get("dino_geometry")
    require(isinstance(q_geometry, Mapping) and isinstance(r_geometry, Mapping), "DINO geometry absent")
    require(source_record.get("query_geometry_sha256") == q_geometry.get("geometry_sha256"), "query geometry hash drift")
    require(source_record.get("reference_geometry_sha256") == r_geometry.get("geometry_sha256"), "reference geometry hash drift")
    query_grid = _grid(source_record.get("deployment_query_grid_shape"), "deployment query")
    reference_grid = _grid(source_record.get("deployment_reference_grid_shape"), "deployment reference")
    require(query_grid == _grid(q_geometry.get("grid_shape"), "query geometry"), "query geometry grid drift")
    require(reference_grid == _grid(r_geometry.get("grid_shape"), "reference geometry"), "reference geometry grid drift")
    query_valid = torch.as_tensor(q_geometry.get("valid_patch_mask"), dtype=torch.bool).flatten()
    reference_valid = torch.as_tensor(r_geometry.get("valid_patch_mask"), dtype=torch.bool).flatten()
    require(query_valid.shape == (math.prod(query_grid),) and reference_valid.shape == (math.prod(reference_grid),), "valid-mask shape drift")

    root_count = source_record.get("root_count")
    action_count = source_record.get("action_count")
    require(type(root_count) is int and type(action_count) is int and root_count > 0 and action_count > 0, "root/action counts drift")
    query_mask_rows = source_record.get("deployment_query_root_masks")
    reference_table_rows = source_record.get("deployment_reference_mask_table")
    reference_indices = source_record.get("deployment_reference_mask_index")
    action_keys = source_record.get("action_keys")
    require(isinstance(query_mask_rows, list) and len(query_mask_rows) == root_count, "query-root mask table drift")
    require(isinstance(reference_table_rows, list) and reference_table_rows, "reference mask table absent")
    require(isinstance(reference_indices, list) and len(reference_indices) == root_count, "reference index matrix drift")
    require(isinstance(action_keys, list) and len(action_keys) == root_count, "action-key matrix drift")
    selected = tuple(selected_action_indices)
    require(len(selected) == root_count, "selected-action root axis drift")
    eligibility = _decode_eligibility(source_record.get("eligibility_rle"), roots=root_count, actions=action_count)
    query_masks = tuple(_decode_mask(item, expected_grid=query_grid, name=f"query root {root}") for root, item in enumerate(query_mask_rows))
    require(all(not bool((mask & ~query_valid).any()) for mask in query_masks), "query root escapes valid mask")

    observations: list[CompactRootObservationV2] = []
    for root in range(root_count):
        action = selected[root]
        qmask = query_masks[root]
        if action is None:
            rmask = torch.zeros(math.prod(reference_grid), dtype=torch.bool)
            action_key = None
        else:
            require(type(action) is int and 0 <= action < action_count and bool(eligibility[root, action]), "selected action is ineligible")
            require(len(reference_indices[root]) == action_count and len(action_keys[root]) == action_count, "selected action axis drift")
            table_ordinal = reference_indices[root][action]
            require(type(table_ordinal) is int and 0 <= table_ordinal < len(reference_table_rows), "reference table address drift")
            rmask = _decode_mask(reference_table_rows[table_ordinal], expected_grid=reference_grid, name=f"reference root {root}")
            require(not bool((rmask & ~reference_valid).any()), "reference component escapes valid mask")
            action_key = _sha(action_keys[root][action], f"action key {root}")
        observations.append(
            CompactRootObservationV2(
                root_ordinal=root,
                root_topology_sha256=canonical_sha256({"root": root, "query_mask_sha256": tensor_sha256(qmask)}),
                query_mask=qmask,
                reference_mask=rmask,
                query_grid_shape=query_grid,
                reference_grid_shape=reference_grid,
                reference_action_key_sha256=action_key,
            )
        )

    colnomic_grid = _grid(source_record.get("colnomic_query_grid_shape"), "ColNomic query")
    proposal_rows, bank = _structural_rows(colnomic_grid, query_grid, query_masks, row_scores)
    selection = select_compact_p_proposal_v2(proposal_rows, observations)
    core_lock = make_candidate_p_lock_v2(
        candidate_key=source_record.get("candidate_key"),
        candidate_physical_row=candidate_row,
        track=fold_record.get("track"),
        canonical_arm=CANONICAL_ARM,
        query_grid_shape=query_grid,
        reference_grid_shape=reference_grid,
        query_valid_mask=query_valid,
        reference_valid_mask=reference_valid,
        roots=tuple(item.to_lock_root() for item in observations),
        selected_root_ordinals=selection.selected_root_ordinals,
        lock_state=selection.lock_state,
        p_checkpoint_sha256=p_checkpoint_sha256,
        query_source_image_sha256=source_record.get("query_source_image_sha256"),
        reference_source_image_sha256=source_record.get("candidate_reference_source_sha256"),
        query_geometry_sha256=source_record.get("query_geometry_sha256"),
        reference_geometry_sha256=source_record.get("reference_geometry_sha256"),
    )
    coverage = torch.zeros(math.prod(query_grid), dtype=torch.int64)
    for root in core_lock.selected_roots:
        if root.structurally_mapped:
            coverage += root.query_mask.to(torch.int64)
    union = coverage > 0
    q_stats = mask_geometry_v2(query_valid, query_grid)
    r_stats = mask_geometry_v2(reference_valid, reference_grid)
    selected_radius = None if selection.selected_bank_ordinal is None else int(bank[selection.selected_bank_ordinal].radius)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_level": CLAIM_LEVEL,
        "query": {
            "query_id": source_record.get("query_id"),
            "historical_query_ordinal": fold_record.get("query_ordinal"),
            "execution_ordinal": execution,
            "query_source_image_sha256": source_record.get("query_source_image_sha256"),
            "outer_fold": outer_fold,
            "inner_heldout_fold": membership.get("inner_heldout_fold"),
            "fit_id": fit_id,
            "crossfit_role": crossfit_role,
            "track": fold_record.get("track"),
        },
        "candidate": {
            "candidate_position": source_record.get("candidate_position"),
            "candidate_key": source_record.get("candidate_key"),
            "candidate_physical_row": candidate_row,
            "candidate_native_content_sha256": source_record.get("cache_sha256"),
            "candidate_reference_source_sha256": source_record.get("candidate_reference_source_sha256"),
        },
        "direction": source_record.get("direction"),
        "canonical_arm": CANONICAL_ARM,
        "p_checkpoint_sha256": p_checkpoint_sha256,
        "p_training_manifest_sha256": p_training_manifest_sha256,
        "checkpoint_claim": "ENGINEERING_ONLY_NOT_CONSUMABLE",
        "complete_bank_sha256": superregion_bank_sha256(bank),
        "complete_root_table_sha256": canonical_sha256([item.to_lock_root().geometry_payload() for item in observations]),
        "selected": {
            "lock_state": selection.lock_state,
            "selected_bank_ordinal": selection.selected_bank_ordinal,
            "selected_bank_row_sha256": selection.selected_row_sha256,
            "selected_core_row_sha256": core_lock.geometry_signature.selected_row_sha256,
            "selected_radius": selected_radius,
            "selected_root_ordinals": list(selection.selected_root_ordinals),
        },
        "query_geometry": {
            "geometry_sha256": source_record.get("query_geometry_sha256"),
            "grid_shape": list(query_grid),
            "source_valid_mask_sha256": q_geometry.get("valid_patch_mask_sha256"),
            **q_stats.as_payload(),
        },
        "reference_geometry": {
            "geometry_sha256": source_record.get("reference_geometry_sha256"),
            "grid_shape": list(reference_grid),
            "source_valid_mask_sha256": r_geometry.get("valid_patch_mask_sha256"),
            **r_stats.as_payload(),
        },
        "core_lock_record": candidate_p_lock_v2_to_record(core_lock),
        "fixed_denominator": {
            "complete_root_coverage_used": True,
            "missing_root_contribution": "EXACT_ZERO",
            "available_root_renormalization": False,
            "available_patch_renormalization": False,
            "outside_union_evidence": "EXACT_ZERO",
            "score_sign_used_as_availability": False,
            "coverage": coverage.tolist(),
            "coverage_sha256": tensor_sha256(coverage),
            "query_union": union.tolist(),
            "query_union_sha256": tensor_sha256(union),
        },
        "row_count_by_radius": {str(radius): sum(item.radius == radius for item in bank) for radius in (1, 2, 3, 4)},
        "total_row_count": len(bank),
        "erased_fields": list(ERASED_FIELDS),
    }
    record["record_sha256"] = canonical_sha256(record)
    validate_natural_p_lock_v2(record)
    return record


_TOP = {
    "schema_version", "claim_level", "query", "candidate", "direction",
    "canonical_arm", "p_checkpoint_sha256", "p_training_manifest_sha256",
    "checkpoint_claim", "complete_bank_sha256", "complete_root_table_sha256",
    "selected", "query_geometry", "reference_geometry", "core_lock_record",
    "fixed_denominator", "row_count_by_radius", "total_row_count",
    "erased_fields", "record_sha256",
}


def validate_natural_p_lock_v2(value: Mapping[str, Any]) -> None:
    """Validate the exact score-erased natural wrapper and nested V2 core."""

    require(isinstance(value, Mapping) and set(value) == _TOP, "natural V2 lock field-set drift")
    require(value.get("schema_version") == SCHEMA_VERSION and value.get("claim_level") == CLAIM_LEVEL, "natural V2 lock schema/claim drift")
    unsigned = {key: copy.deepcopy(item) for key, item in value.items() if key != "record_sha256"}
    require(_sha(value.get("record_sha256"), "natural record") == canonical_sha256(unsigned), "natural V2 record hash drift")
    require(value.get("canonical_arm") == CANONICAL_ARM and value.get("checkpoint_claim") == "ENGINEERING_ONLY_NOT_CONSUMABLE", "natural V2 arm/checkpoint claim drift")
    _sha(value.get("p_checkpoint_sha256"), "P checkpoint")
    _sha(value.get("p_training_manifest_sha256"), "P training manifest")
    core = candidate_p_lock_v2_from_record(value.get("core_lock_record"))
    query = value.get("query")
    candidate = value.get("candidate")
    selected = value.get("selected")
    require(isinstance(query, Mapping) and isinstance(candidate, Mapping) and isinstance(selected, Mapping), "natural V2 nested address absent")
    require(
        query.get("query_id")
        and query.get("fit_id")
        and query.get("crossfit_role")
        in ("INNER_HELDOUT_DEPLOYMENT", "OUTER_HELDOUT_DEPLOYMENT"),
        "natural V2 query role drift",
    )
    require(candidate.get("candidate_key") == core.candidate_key and candidate.get("candidate_physical_row") == core.candidate_physical_row, "natural V2 candidate/core drift")
    require(selected.get("lock_state") == core.lock_state and tuple(selected.get("selected_root_ordinals", ())) == core.selected_root_ordinals, "natural V2 selected/core drift")
    require(_sha(selected.get("selected_bank_row_sha256"), "selected bank row") if core.lock_state == LOCK_PROPOSAL_READY else selected.get("selected_bank_row_sha256") is None, "natural V2 selected bank-row drift")
    require(selected.get("selected_core_row_sha256") == core.geometry_signature.selected_row_sha256, "natural V2 selected core-row drift")
    fixed = value.get("fixed_denominator")
    require(isinstance(fixed, Mapping), "natural V2 denominator absent")
    coverage = torch.as_tensor(fixed.get("coverage"), dtype=torch.int64)
    union = torch.as_tensor(fixed.get("query_union"), dtype=torch.bool)
    expected = torch.zeros_like(coverage)
    for root in core.selected_roots:
        if root.structurally_mapped:
            expected += root.query_mask.to(torch.int64)
    require(torch.equal(coverage, expected) and torch.equal(union, expected > 0), "natural V2 selected coverage/union drift")
    require(fixed.get("coverage_sha256") == tensor_sha256(coverage) and fixed.get("query_union_sha256") == tensor_sha256(union), "natural V2 denominator hash drift")
    require(
        fixed.get("complete_root_coverage_used") is True
        and fixed.get("missing_root_contribution") == "EXACT_ZERO"
        and fixed.get("available_root_renormalization") is False
        and fixed.get("available_patch_renormalization") is False
        and fixed.get("outside_union_evidence") == "EXACT_ZERO"
        and fixed.get("score_sign_used_as_availability") is False,
        "natural V2 denominator policy drift",
    )
    require(value.get("erased_fields") == list(ERASED_FIELDS), "natural V2 erased-field receipt drift")


__all__ = [
    "SCHEMA_VERSION",
    "CLAIM_LEVEL",
    "ERASED_FIELDS",
    "NaturalAdapterV2Error",
    "build_natural_p_lock_v2",
    "validate_natural_p_lock_v2",
]
