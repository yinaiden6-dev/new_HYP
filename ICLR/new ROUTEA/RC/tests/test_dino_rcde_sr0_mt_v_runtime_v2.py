from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
sys.path.insert(0, str(RC_ROOT / "tests"))

import rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 as runtime_v1
import rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v2 as runtime_v2
from rc_aslo_xf.dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import PairEvidence

from test_dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (
    _SharedSyntheticV as _NaturalFixtureV,
    _head_fixture,
)
from test_dino_rcde_sr0_mt_three_arm_v1 import (
    _SharedSyntheticV,
    _locks,
)


def _assert_term_exact(first, second) -> None:
    assert first.status == second.status
    assert first.direction == second.direction
    assert first.owner_candidate_key == second.owner_candidate_key
    assert first.opponent_candidate_key == second.opponent_candidate_key
    assert torch.equal(first.query_mask, second.query_mask)
    assert torch.equal(first.patch_evidence, second.patch_evidence)
    assert torch.equal(first.scalar_contributions, second.scalar_contributions)
    assert torch.equal(first.logit, second.logit)
    assert first.decoded_root_ordinals == second.decoded_root_ordinals
    assert first.decode_receipts == second.decode_receipts


def test_v2_changes_only_full_decode_and_preserves_all_patch_and_local_exactly() -> None:
    query, locks = _locks()
    first_model = _SharedSyntheticV()
    second_model = _SharedSyntheticV()
    second_model.load_state_dict(first_model.state_dict())

    historical = runtime_v1.decode_pair(
        first_model, query, locks, "candidate-g", "candidate-c"
    )
    corrected = runtime_v2.decode_pair(
        second_model, query, locks, "candidate-g", "candidate-c"
    )

    for arm_name in (ARM_ALL_PATCH, ARM_QUERY_LOCAL_COMPONENTS):
        first_arm = historical.by_name()[arm_name]
        second_arm = corrected.by_name()[arm_name]
        assert torch.equal(first_arm.logit, second_arm.logit)
        assert torch.equal(
            first_arm.scalar_contributions, second_arm.scalar_contributions
        )
        for first_term, second_term in zip(
            (*first_arm.forward_by_direction, *first_arm.reverse_by_direction),
            (*second_arm.forward_by_direction, *second_arm.reverse_by_direction),
            strict=True,
        ):
            _assert_term_exact(first_term, second_term)

    old_full = historical.by_name()[ARM_QUERY_FULL_REFERENCE]
    new_full = corrected.by_name()[ARM_QUERY_FULL_REFERENCE]
    local = corrected.by_name()[ARM_QUERY_LOCAL_COMPONENTS]
    assert all(
        term.decoded_root_ordinals == (-1,)
        for term in (*old_full.forward_by_direction, *old_full.reverse_by_direction)
    )
    for full_term, local_term in zip(
        (*new_full.forward_by_direction, *new_full.reverse_by_direction),
        (*local.forward_by_direction, *local.reverse_by_direction),
        strict=True,
    ):
        assert len(full_term.decoded_root_ordinals) >= 2
        assert full_term.decoded_root_ordinals == local_term.decoded_root_ordinals
        assert torch.equal(full_term.query_mask, local_term.query_mask)
        assert [item.query_mask_sha256 for item in full_term.decode_receipts] == [
            item.query_mask_sha256 for item in local_term.decode_receipts
        ]
        assert all(
            item.owner_component_source_root_ordinal is None
            and item.opponent_component_source_root_ordinal is None
            for item in full_term.decode_receipts
        )


class _QueryGranularitySensitiveV(torch.nn.Module):
    """Fixture whose nonlinear mask normalization distinguishes union/root calls."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def decode_candidate(
        self,
        query_layers,
        reference_layers,
        query_mask,
        reference_mask,
        query_grid,
        reference_grid,
        **_,
    ):
        qmask = torch.as_tensor(
            query_mask, dtype=torch.bool, device=query_layers.device
        ).flatten()
        rmask = torch.as_tensor(
            reference_mask, dtype=torch.bool, device=reference_layers.device
        ).flatten()
        assert qmask.numel() == math.prod(query_grid)
        assert rmask.numel() == math.prod(reference_grid)
        reference_mean = reference_layers[0, rmask, 0].mean()
        # Consensus/normalization is deliberately nonlinear in the active
        # query extent, just as the real 4-D decoder is not union-distributive.
        normalization = qmask.sum().to(query_layers.dtype).square()
        relational = (
            query_layers[0, :, :1]
            * reference_mean
            / normalization
            * qmask[:, None]
            * self.scale
        )
        return SimpleNamespace(relational=relational)

    def compare_relational(self, relational_g, relational_c, query_mask):
        mask = torch.as_tensor(
            query_mask, dtype=torch.bool, device=relational_g.device
        ).flatten()
        contribution = (
            relational_g[:, 0] - relational_c[:, 0]
        ) * mask.to(relational_g.dtype)
        return PairEvidence(
            signed_support=contribution,
            modulation=mask.to(contribution.dtype),
            evidence=contribution,
            contributions=contribution,
            logit=contribution.sum() / mask.sum().to(contribution.dtype),
        )


class _ReferenceScopeInsensitiveV(torch.nn.Module):
    """Makes FULL/LOCAL outputs equal when mask scope carries no information."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def decode_candidate(
        self,
        query_layers,
        reference_layers,
        query_mask,
        reference_mask,
        query_grid,
        reference_grid,
        **_,
    ):
        qmask = torch.as_tensor(
            query_mask, dtype=torch.bool, device=query_layers.device
        ).flatten()
        rmask = torch.as_tensor(reference_mask, dtype=torch.bool).flatten()
        assert qmask.numel() == math.prod(query_grid)
        assert rmask.numel() == math.prod(reference_grid)
        # Candidate content remains distinct, but the supplied reference mask
        # is intentionally ignored so the shared aggregation path must close.
        candidate_value = reference_layers[0, :, 0].mean()
        relational = (
            query_layers[0, :, :1]
            * candidate_value
            * qmask[:, None]
            * self.scale
        )
        return SimpleNamespace(relational=relational)

    def compare_relational(self, relational_g, relational_c, query_mask):
        mask = torch.as_tensor(
            query_mask, dtype=torch.bool, device=relational_g.device
        ).flatten()
        contribution = (
            relational_g[:, 0] - relational_c[:, 0]
        ) * mask.to(relational_g.dtype)
        return PairEvidence(
            signed_support=contribution,
            modulation=mask.to(contribution.dtype),
            evidence=contribution,
            contributions=contribution,
            logit=contribution.sum() / mask.sum().to(contribution.dtype),
        )


def test_full_local_overlap_weights_and_union_denominator_are_identical() -> None:
    query, locks = _locks()
    result = runtime_v2.decode_pair(
        _ReferenceScopeInsensitiveV(),
        query,
        locks,
        "candidate-g",
        "candidate-c",
    )
    full = result.by_name()[ARM_QUERY_FULL_REFERENCE]
    local = result.by_name()[ARM_QUERY_LOCAL_COMPONENTS]
    assert torch.equal(full.logit, local.logit)
    assert torch.equal(full.scalar_contributions, local.scalar_contributions)
    for full_term, local_term in zip(
        (*full.forward_by_direction, *full.reverse_by_direction),
        (*local.forward_by_direction, *local.reverse_by_direction),
        strict=True,
    ):
        assert torch.equal(full_term.patch_evidence, local_term.patch_evidence)
        assert torch.equal(
            full_term.scalar_contributions, local_term.scalar_contributions
        )
        assert torch.equal(full_term.logit, local_term.logit)


def test_nonlinear_decoder_regression_rejects_union_once_full_semantics() -> None:
    query, locks = _locks()
    old_model = _QueryGranularitySensitiveV()
    new_model = _QueryGranularitySensitiveV()
    new_model.load_state_dict(old_model.state_dict())

    old = runtime_v1.decode_pair(
        old_model, query, locks, "candidate-g", "candidate-c"
    ).by_name()[ARM_QUERY_FULL_REFERENCE]
    new = runtime_v2.decode_pair(
        new_model, query, locks, "candidate-g", "candidate-c"
    ).by_name()[ARM_QUERY_FULL_REFERENCE]

    assert not torch.equal(old.logit, new.logit)
    assert all(
        term.decoded_root_ordinals != (-1,)
        for term in (*new.forward_by_direction, *new.reverse_by_direction)
    )


def test_natural_core_v2_projection_fixture_has_matched_rootwise_full_local_paths() -> None:
    _, _, query, _, projections = _head_fixture()
    locks = {key: value.runtime_lock for key, value in projections.items()}
    left, right = tuple(locks)
    result = runtime_v2.decode_pair(
        _NaturalFixtureV(), query, locks, left, right
    )
    full = result.by_name()[ARM_QUERY_FULL_REFERENCE]
    local = result.by_name()[ARM_QUERY_LOCAL_COMPONENTS]

    for full_term, local_term in zip(
        (*full.forward_by_direction, *full.reverse_by_direction),
        (*local.forward_by_direction, *local.reverse_by_direction),
        strict=True,
    ):
        assert full_term.status == local_term.status
        assert len(full_term.decoded_root_ordinals) >= 2
        assert full_term.decoded_root_ordinals == local_term.decoded_root_ordinals
        assert torch.equal(full_term.query_mask, local_term.query_mask)
        assert [item.query_mask_sha256 for item in full_term.decode_receipts] == [
            item.query_mask_sha256 for item in local_term.decode_receipts
        ]
        # A full reference hash is constant across roots for each candidate;
        # the natural-form local components retain root-specific scopes.
        assert len(
            {item.owner_reference_mask_sha256 for item in full_term.decode_receipts}
        ) == 1
        assert len(
            {item.opponent_reference_mask_sha256 for item in full_term.decode_receipts}
        ) == 1
        assert [item.owner_reference_mask_sha256 for item in full_term.decode_receipts] != [
            item.owner_reference_mask_sha256 for item in local_term.decode_receipts
        ]
        assert [
            item.opponent_reference_mask_sha256 for item in full_term.decode_receipts
        ] != [
            item.opponent_reference_mask_sha256 for item in local_term.decode_receipts
        ]
