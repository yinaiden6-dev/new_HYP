"""Execution-only opaque donor-plan adapter for RoMa-RGH natural S0.

Corrected identities belong to the outer control planner.  The runtime receives
only a validated assertion that the frozen donor permutation is
identity-disjoint; neither identity strings nor target roles cross this module's
interface or enter its sealed plan.

The adapter deliberately mirrors ``candidate_binding_destroy_population``:
``proposal_only=True`` moves the complete donor proposal bundle while retaining
the destination verification reference, whereas ``False`` moves proposal and
verification-reference content together.  This module adds no model, score,
threshold, or scientific decision.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .roma_rgh_reference_first_v1 import (
    ReferenceFirstRoMaAtomBank,
    logical_sha256,
    seal_reference_first_bank,
    validate_reference_first_bank,
)


PLAN_SCHEMA = "roma_rgh_reference_first_s0_opaque_donor_plan_v1_20260910"
CONTROL_SCHEMA = "roma_rgh_reference_first_s0_opaque_binding_control_v1_20260910"
NULL_SHA256 = "0" * 64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: object, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be lowercase SHA256",
    )
    return value


def opaque_candidate_axis_sha256(candidate_keys: Sequence[str]) -> str:
    """Hash an ordered opaque candidate-key axis without reading identities."""

    keys = tuple(candidate_keys)
    _require(
        len(keys) >= 2
        and all(type(key) is str and bool(key) for key in keys)
        and len(set(keys)) == len(keys),
        "opaque candidate axis invalid",
    )
    return logical_sha256(list(keys))


@dataclass(frozen=True)
class FrozenOpaqueDonorPlan:
    """The complete identity-erased interface from planner to S0 runtime."""

    query_resource_key: str
    candidate_axis_sha256: str
    donor_indices: tuple[int, ...]
    fixed_point_free: bool
    corrected_identity_disjoint: bool
    planner_validation_sha256: str
    logical_sha256: str


def _plan_logical_sha256(value: FrozenOpaqueDonorPlan) -> str:
    return logical_sha256(
        {
            "schema": PLAN_SCHEMA,
            "query_resource_key": value.query_resource_key,
            "candidate_axis_sha256": value.candidate_axis_sha256,
            "donor_indices": list(value.donor_indices),
            "fixed_point_free": value.fixed_point_free,
            "corrected_identity_disjoint": value.corrected_identity_disjoint,
            "planner_validation_sha256": value.planner_validation_sha256,
        }
    )


def validate_frozen_opaque_donor_plan(
    value: FrozenOpaqueDonorPlan,
    *,
    expected_candidate_count: int | None = None,
) -> FrozenOpaqueDonorPlan:
    """Fail closed on any plan not carrying a complete validated derangement."""

    _require(type(value) is FrozenOpaqueDonorPlan, "opaque donor plan type drift")
    _require(
        type(value.query_resource_key) is str and bool(value.query_resource_key),
        "opaque donor plan query key invalid",
    )
    _require_sha256(value.candidate_axis_sha256, "candidate axis")
    _require_sha256(value.planner_validation_sha256, "planner validation")
    _require(
        value.planner_validation_sha256 != NULL_SHA256,
        "planner validation must bind a non-null validated artifact",
    )
    _require(type(value.donor_indices) is tuple, "donor indices must be a tuple")
    _require(
        len(value.donor_indices) >= 2
        and all(type(index) is int for index in value.donor_indices),
        "donor indices type/count drift",
    )
    count = len(value.donor_indices)
    if expected_candidate_count is not None:
        _require(
            type(expected_candidate_count) is int
            and expected_candidate_count == count,
            "donor plan candidate count drift",
        )
    _require(
        tuple(sorted(value.donor_indices)) == tuple(range(count)),
        "donor indices must be a complete permutation",
    )
    actual_fixed_point_free = all(
        destination != donor
        for destination, donor in enumerate(value.donor_indices)
    )
    _require(
        type(value.fixed_point_free) is bool
        and value.fixed_point_free
        and actual_fixed_point_free,
        "donor plan is not fixed-point-free",
    )
    _require(
        type(value.corrected_identity_disjoint) is bool
        and value.corrected_identity_disjoint,
        "identity-disjoint planner assertion absent",
    )
    _require_sha256(value.logical_sha256, "opaque donor plan logical hash")
    _require(
        value.logical_sha256 == _plan_logical_sha256(value),
        "opaque donor plan logical hash drift",
    )
    return value


def freeze_opaque_donor_plan(
    *,
    query_resource_key: str,
    candidate_axis_sha256: str,
    donor_indices: Sequence[int],
    fixed_point_free: bool,
    corrected_identity_disjoint: bool,
    planner_validation_sha256: str,
) -> FrozenOpaqueDonorPlan:
    """Seal an outer-planner product after identities have been erased."""

    _require(
        all(type(index) is int for index in donor_indices),
        "donor indices must contain exact integers",
    )
    staged = FrozenOpaqueDonorPlan(
        query_resource_key=query_resource_key,
        candidate_axis_sha256=candidate_axis_sha256,
        donor_indices=tuple(donor_indices),
        fixed_point_free=fixed_point_free,
        corrected_identity_disjoint=corrected_identity_disjoint,
        planner_validation_sha256=planner_validation_sha256,
        logical_sha256="PENDING",
    )
    sealed = replace(staged, logical_sha256=_plan_logical_sha256(staged))
    return validate_frozen_opaque_donor_plan(sealed)


def apply_frozen_opaque_donor_plan(
    banks: Sequence[ReferenceFirstRoMaAtomBank],
    *,
    donor_plan: FrozenOpaqueDonorPlan,
    namespace: str,
    proposal_only: bool,
) -> tuple[ReferenceFirstRoMaAtomBank, ...]:
    """Apply an identity-erased donor permutation before P materialization."""

    source = tuple(validate_reference_first_bank(bank) for bank in banks)
    count = len(source)
    plan = validate_frozen_opaque_donor_plan(
        donor_plan,
        expected_candidate_count=count,
    )
    _require(
        type(namespace) is str and bool(namespace) and namespace != "REAL",
        "candidate-binding namespace invalid",
    )
    _require(type(proposal_only) is bool, "proposal_only must be bool")
    _require(
        count >= 2
        and all(bank.control_namespace == "REAL" for bank in source)
        and {bank.query_resource_key for bank in source}
        == {plan.query_resource_key},
        "opaque donor population query/control drift",
    )
    destination_keys = tuple(bank.destination_candidate_key for bank in source)
    _require(
        len(set(destination_keys)) == count
        and opaque_candidate_axis_sha256(destination_keys)
        == plan.candidate_axis_sha256,
        "opaque donor candidate axis drift",
    )

    control_plan_sha256 = logical_sha256(
        {
            "schema": CONTROL_SCHEMA,
            "namespace": namespace,
            "proposal_only": proposal_only,
            "opaque_donor_plan_sha256": plan.logical_sha256,
            "destination_candidate_keys": list(destination_keys),
            "donor_source_reference_keys": [
                source[index].source_reference_resource_key
                for index in plan.donor_indices
            ],
        }
    )
    output: list[ReferenceFirstRoMaAtomBank] = []
    for destination, donor in enumerate(plan.donor_indices):
        destination_item = source[destination]
        donor_item = source[donor]
        controlled = replace(
            donor_item,
            destination_candidate_key=destination_item.destination_candidate_key,
            verification_reference_resource_key=(
                destination_item.verification_reference_resource_key
                if proposal_only
                else donor_item.verification_reference_resource_key
            ),
            control_namespace=namespace,
            parent_real_sha256=destination_item.logical_sha256,
            control_plan_sha256=control_plan_sha256,
            logical_sha256="PENDING",
        )
        output.append(seal_reference_first_bank(controlled))
    return tuple(output)


__all__ = [
    "PLAN_SCHEMA",
    "CONTROL_SCHEMA",
    "FrozenOpaqueDonorPlan",
    "opaque_candidate_axis_sha256",
    "freeze_opaque_donor_plan",
    "validate_frozen_opaque_donor_plan",
    "apply_frozen_opaque_donor_plan",
]
