from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))

from rc_aslo_xf.dino_rcde_v1_2_resource_core import (  # noqa: E402
    DINO_RCDE_V1_2,
    PairEvidence,
    SUMMARY_DIM,
)
from rc_aslo_xf.dino_rcde_colnomic_superregion_v1 import (  # noqa: E402
    STATUS_AMBIGUOUS,
    STATUS_H0,
    STATUS_READY,
    STATUS_SINGLE,
    SuperregionContractError,
    compare_regional_fields,
    decode_candidate_in_superregion,
    map_colnomic_hypotheses_to_dino_superregions,
    rasterize_by_canonical_overlap,
    reduce_crossfit_regional_pair,
    regular_grid_geometry,
    reorder_candidate_seals,
    resolve_actionable_component_state,
    validate_bounded_float_reduction,
    validate_conditioned_float_decomposition,
)
from rc_aslo_xf.lt_hyp_pvlock_v4 import (  # noqa: E402
    MAX_HYPOTHESES,
    SealedMultiPatchHypotheses,
)


QUERY_SHA = "0" * 64
REFERENCE_SHA = "3" * 64
COL_PROCESSOR_SHA = "1" * 64
DINO_PROCESSOR_SHA = "2" * 64


def _block(row: int, column: int) -> list[int]:
    return [
        (row + dy) * 8 + column + dx
        for dy in range(3)
        for dx in range(3)
    ]


def _hypotheses(
    blocks: list[list[int]], *, direction: str = "a_to_b"
) -> SealedMultiPatchHypotheses:
    capacity = 9
    seed_query = torch.full((MAX_HYPOTHESES,), -1)
    seed_reference = torch.full((MAX_HYPOTHESES,), -1)
    support_query = torch.full((MAX_HYPOTHESES, capacity), -1)
    support_reference = torch.full((MAX_HYPOTHESES, capacity), -1)
    weights = torch.zeros((MAX_HYPOTHESES, capacity), dtype=torch.float64)
    support_mask = torch.zeros((MAX_HYPOTHESES, capacity), dtype=torch.bool)
    legal = torch.zeros(MAX_HYPOTHESES, dtype=torch.bool)
    for index, block in enumerate(blocks):
        values = torch.tensor(block)
        seed_query[index] = values[0]
        seed_reference[index] = values[0]
        support_query[index] = values
        support_reference[index] = values
        weights[index] = 1.0 / capacity
        support_mask[index] = True
        legal[index] = True
    return SealedMultiPatchHypotheses(
        seed_query_indices=seed_query,
        seed_reference_indices=seed_reference,
        support_query_indices=support_query,
        support_reference_indices=support_reference,
        support_weights=weights,
        support_mask=support_mask,
        legal=legal,
        query_grid_shape=(8, 8),
        reference_grid_shape=(8, 8),
        direction=direction,
    )


def _geometries(*, dino_query_valid: torch.Tensor | None = None):
    col_query = regular_grid_geometry(
        source_image_sha256=QUERY_SHA,
        source_key="query:fixture",
        processor_config_sha256=COL_PROCESSOR_SHA,
        grid_shape=(8, 8),
    )
    col_reference = regular_grid_geometry(
        source_image_sha256=REFERENCE_SHA,
        source_key="physical-row:17",
        processor_config_sha256=COL_PROCESSOR_SHA,
        grid_shape=(8, 8),
    )
    dino_query = regular_grid_geometry(
        source_image_sha256=QUERY_SHA,
        source_key="query:fixture",
        processor_config_sha256=DINO_PROCESSOR_SHA,
        grid_shape=(8, 8),
        valid_patch_mask=dino_query_valid,
    )
    dino_reference = regular_grid_geometry(
        source_image_sha256=REFERENCE_SHA,
        source_key="physical-row:17",
        processor_config_sha256=DINO_PROCESSOR_SHA,
        grid_shape=(8, 8),
    )
    return col_query, col_reference, dino_query, dino_reference


def _seal(blocks: list[list[int]], *, candidate_key: str = "row:17"):
    col_query, col_reference, dino_query, dino_reference = _geometries()
    return map_colnomic_hypotheses_to_dino_superregions(
        _hypotheses(blocks),
        candidate_key=candidate_key,
        colnomic_query_geometry=col_query,
        colnomic_reference_geometry=col_reference,
        dino_query_geometry=dino_query,
        dino_reference_geometry=dino_reference,
    )


def test_candidate_multisupport_becomes_one_paired_connected_superregion() -> None:
    seal = _seal([_block(2, 2)])
    assert seal.status == STATUS_READY
    assert len(seal.regions) == 1
    region = seal.regions[0]
    assert region.source_slots == (0,)
    assert int(region.query_mask.sum()) >= 4
    assert int(region.reference_mask.sum()) >= 4
    assert region.query_mask_sha256 != region.reference_mask_sha256 or torch.equal(
        region.query_mask, region.reference_mask
    )


def test_touching_hypotheses_merge_but_disconnected_components_do_not() -> None:
    touching = _seal([_block(2, 1), _block(2, 4)])
    assert len(touching.regions) == 1
    assert touching.regions[0].source_slots == (0, 1)

    disconnected = _seal([_block(0, 0), _block(5, 5)])
    assert len(disconnected.regions) == 2
    assert resolve_actionable_component_state(
        [item.joint_sha256 for item in disconnected.regions]
    ) == STATUS_AMBIGUOUS


def test_mapping_never_repairs_a_padding_broken_region() -> None:
    valid = torch.ones(64, dtype=torch.bool)
    valid[_block(2, 2)[4]] = False
    col_query, col_reference, dino_query, dino_reference = _geometries(
        dino_query_valid=valid
    )
    seal = map_colnomic_hypotheses_to_dino_superregions(
        _hypotheses([_block(2, 2)]),
        candidate_key="row:17",
        colnomic_query_geometry=col_query,
        colnomic_reference_geometry=col_reference,
        dino_query_geometry=dino_query,
        dino_reference_geometry=dino_reference,
    )
    # Removing the centre of a 3x3 footprint leaves one ring, which is still a
    # connected 4CC; remove a full middle column to force two components.
    if seal.status == STATUS_READY:
        valid = torch.ones(64, dtype=torch.bool)
        valid[torch.tensor([19, 27, 35])] = False
        col_query, col_reference, dino_query, dino_reference = _geometries(
            dino_query_valid=valid
        )
        seal = map_colnomic_hypotheses_to_dino_superregions(
            _hypotheses([_block(2, 2)]),
            candidate_key="row:17",
            colnomic_query_geometry=col_query,
            colnomic_reference_geometry=col_reference,
            dino_query_geometry=dino_query,
            dino_reference_geometry=dino_reference,
        )
    assert seal.status == STATUS_H0
    assert seal.regions == ()


def test_canonical_overlap_requires_same_source_and_an_explicit_receipt() -> None:
    source, _, destination, _ = _geometries()
    mask = torch.zeros(64, dtype=torch.bool)
    mask[_block(2, 2)] = True
    mapped = rasterize_by_canonical_overlap(mask, source, destination)
    assert torch.equal(mapped, mask)

    foreign = regular_grid_geometry(
        source_image_sha256="f" * 64,
        source_key="query:fixture",
        processor_config_sha256=DINO_PROCESSOR_SHA,
        grid_shape=(8, 8),
    )
    with pytest.raises(SuperregionContractError, match="source-image mismatch"):
        rasterize_by_canonical_overlap(mask, source, foreign)


class _FakeDecoder:
    def decode_candidate(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        **_: object,
    ) -> SimpleNamespace:
        del query_grid, reference_grid
        reference_mean = reference_layers[0, reference_mask.flatten()].mean(dim=0)
        relational = query_layers[0, :, :SUMMARY_DIM] + reference_mean[:SUMMARY_DIM]
        relational = relational * query_mask.flatten()[:, None].to(relational.dtype)
        return SimpleNamespace(relational=relational)


def _regional_field(
    layers: torch.Tensor,
    reference: torch.Tensor,
    region,
    *,
    candidate_key: str,
):
    return decode_candidate_in_superregion(
        _FakeDecoder(),
        layers,
        reference,
        candidate_key=candidate_key,
        query_mask=region.query_mask,
        reference_mask=region.reference_mask,
        query_grid_shape=region.query_grid_shape,
        reference_grid_shape=region.reference_grid_shape,
        query_region_sha256=region.query_mask_sha256,
        reference_region_sha256=region.reference_mask_sha256,
    )


def test_region_only_comparator_has_zero_outside_value_and_gradient() -> None:
    torch.manual_seed(17)
    region = _seal([_block(2, 2)]).regions[0]
    query = torch.randn(4, 64, 32, requires_grad=True)
    left_reference = torch.randn(4, 64, 32, requires_grad=True)
    right_reference = torch.randn(4, 64, 32, requires_grad=True)
    left = _regional_field(query, left_reference, region, candidate_key="left")
    right = _regional_field(query, right_reference, region, candidate_key="right")
    comparator = DINO_RCDE_V1_2()
    evidence = compare_regional_fields(comparator, left, right)
    assert not bool(evidence.contributions[~region.query_mask].ne(0).any())
    evidence.logit.backward()
    assert not bool(query.grad[:, ~region.query_mask].ne(0).any())

    altered = query.detach().clone()
    altered[:, ~region.query_mask] += 1.0e6
    altered_left = _regional_field(
        altered, left_reference.detach(), region, candidate_key="left"
    )
    altered_right = _regional_field(
        altered, right_reference.detach(), region, candidate_key="right"
    )
    altered_evidence = compare_regional_fields(comparator, altered_left, altered_right)
    assert torch.equal(evidence.logit.detach(), altered_evidence.logit.detach())


def test_pair_swap_is_antisymmetric_and_same_reference_is_exact_zero() -> None:
    torch.manual_seed(23)
    region = _seal([_block(2, 2)]).regions[0]
    query = torch.randn(4, 64, 32)
    left = _regional_field(
        query, torch.randn(4, 64, 32), region, candidate_key="left"
    )
    right = _regional_field(
        query, torch.randn(4, 64, 32), region, candidate_key="right"
    )
    comparator = DINO_RCDE_V1_2()
    forward = compare_regional_fields(comparator, left, right)
    reverse = compare_regional_fields(comparator, right, left)
    same = compare_regional_fields(comparator, left, left)
    assert torch.allclose(forward.logit, -reverse.logit, atol=1.0e-7, rtol=0.0)
    assert torch.equal(same.logit, torch.zeros_like(same.logit))


def test_regional_contribution_closure_uses_bounded_ulp_not_unbounded_tolerance() -> None:
    region = _seal([_block(2, 2)]).regions[0]
    query = torch.zeros(4, 64, 32)
    left = _regional_field(
        query, torch.zeros(4, 64, 32), region, candidate_key="left"
    )
    right = _regional_field(
        query, torch.zeros(4, 64, 32), region, candidate_key="right"
    )

    class _ULPComparator:
        def __init__(self, steps: int):
            self.steps = steps

        def compare_relational(self, _left, _right, mask):
            contributions = mask.flatten().to(torch.float32)
            expected = contributions.sum() / mask.sum().to(torch.float32)
            observed = expected.clone()
            for _ in range(self.steps):
                observed = torch.nextafter(
                    observed, torch.full_like(observed, float("inf"))
                )
            return PairEvidence(
                signed_support=contributions,
                modulation=contributions,
                evidence=contributions,
                contributions=contributions,
                logit=observed,
            )

    expected = torch.tensor(1.0, dtype=torch.float32)
    four_ulp = expected.clone()
    thirty_two_ulp = expected.clone()
    for _ in range(4):
        four_ulp = torch.nextafter(
            four_ulp, torch.full_like(four_ulp, float("inf"))
        )
    for _ in range(32):
        thirty_two_ulp = torch.nextafter(
            thirty_two_ulp, torch.full_like(thirty_two_ulp, float("inf"))
        )
    validate_bounded_float_reduction(
        four_ulp, expected, name="test bounded ULP"
    )
    with pytest.raises(ValueError, match="bounded ULP failed"):
        validate_bounded_float_reduction(
            thirty_two_ulp, expected, name="test bounded ULP"
        )

    # A decomposition has a condition-aware backward-error bound; output ULPs
    # alone are invalid under cancellation.
    compare_regional_fields(_ULPComparator(32), left, right)

    class _GrossComparator(_ULPComparator):
        def compare_relational(self, _left, _right, mask):
            result = super().compare_relational(_left, _right, mask)
            return PairEvidence(
                signed_support=result.signed_support,
                modulation=result.modulation,
                evidence=result.evidence,
                contributions=result.contributions,
                logit=result.logit + 1.0e-3,
            )

    with pytest.raises(SuperregionContractError, match="closure failed"):
        compare_regional_fields(_GrossComparator(0), left, right)

    class _NonfiniteComparator:
        def compare_relational(self, _left, _right, mask):
            contributions = mask.flatten().to(torch.float32)
            active = int(torch.nonzero(mask.flatten(), as_tuple=False)[0])
            contributions[active] = float("nan")
            return PairEvidence(
                signed_support=contributions,
                modulation=contributions,
                evidence=contributions,
                contributions=contributions,
                logit=torch.tensor(float("nan")),
            )

    with pytest.raises(SuperregionContractError, match="nonfinite"):
        compare_regional_fields(_NonfiniteComparator(), left, right)

    with pytest.raises(ValueError, match="zero-mass"):
        validate_conditioned_float_decomposition(
            torch.tensor(5.0e-8, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            absolute_mass=0.0,
            operation_count=1,
            name="test zero-mass",
        )

    cancellation = torch.tensor(
        [1.0e5, 0.1, -1.0e5, 0.1], dtype=torch.float32
    )
    observed = cancellation.sum()
    reconstructed = cancellation.to(torch.float64).sum().to(torch.float32)
    assert abs(float(observed - reconstructed)) > 32 * float(
        torch.nextafter(
            reconstructed, torch.full_like(reconstructed, float("inf"))
        )
        - reconstructed
    )
    validate_conditioned_float_decomposition(
        observed,
        reconstructed,
        absolute_mass=cancellation.abs().to(torch.float64).sum(),
        operation_count=cancellation.numel() + 2,
        name="test cancellation",
    )
    with pytest.raises(ValueError, match="test missing term failed"):
        validate_conditioned_float_decomposition(
            observed + 1.0,
            reconstructed,
            absolute_mass=cancellation.abs().to(torch.float64).sum(),
            operation_count=cancellation.numel() + 2,
            name="test missing term",
        )

    with pytest.raises(ValueError, match="float64 bounded ULP failed"):
        validate_bounded_float_reduction(
            torch.tensor(1.0 + 1.0e-13, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            name="float64 bounded ULP",
        )


def _pair(value: float) -> PairEvidence:
    scalar = torch.tensor(value)
    return PairEvidence(
        signed_support=scalar.reshape(1),
        modulation=torch.ones(1),
        evidence=scalar.reshape(1),
        contributions=scalar.reshape(1),
        logit=scalar,
    )


def test_crossfit_reducer_has_fixed_four_term_denominator_and_exact_holds() -> None:
    decision = reduce_crossfit_regional_pair(
        [_pair(4.0), None], [None, None]
    )
    assert decision.status == STATUS_SINGLE
    assert float(decision.logit) == 1.0

    h0 = reduce_crossfit_regional_pair([None, None], [None, None])
    assert h0.status == STATUS_H0
    assert torch.equal(h0.logit, torch.zeros_like(h0.logit))

    ambiguity = reduce_crossfit_regional_pair(
        [_pair(4.0), _pair(3.0)], [_pair(-2.0), _pair(-1.0)], ambiguity=True
    )
    assert ambiguity.status == STATUS_AMBIGUOUS
    assert torch.equal(ambiguity.logit, torch.zeros_like(ambiguity.logit))


def test_candidate_reorder_changes_only_sequence_not_sealed_payload() -> None:
    first = _seal([_block(0, 0)], candidate_key="row:a")
    second = _seal([_block(5, 5)], candidate_key="row:b")
    reordered = reorder_candidate_seals((first, second), torch.tensor([1, 0]))
    assert [item.candidate_key for item in reordered] == ["row:b", "row:a"]
    assert reordered[0].regions[0].joint_sha256 == second.regions[0].joint_sha256
    assert reordered[1].regions[0].joint_sha256 == first.regions[0].joint_sha256


def test_public_mainline_apis_are_target_free() -> None:
    forbidden = {"target", "label", "truth", "winner", "rank", "d1", "gap"}
    for function in (
        map_colnomic_hypotheses_to_dino_superregions,
        decode_candidate_in_superregion,
        compare_regional_fields,
        reduce_crossfit_regional_pair,
    ):
        names = {name.lower() for name in inspect.signature(function).parameters}
        assert not names & forbidden
