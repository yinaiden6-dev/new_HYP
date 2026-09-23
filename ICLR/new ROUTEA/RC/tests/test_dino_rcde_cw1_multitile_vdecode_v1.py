from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from types import SimpleNamespace

import pytest
import torch

from rc_aslo_xf.cw0_connected_region_v2 import enumerate_query_macro_seeds
from rc_aslo_xf.dino_rcde_colnomic_superregion_v1 import regular_grid_geometry
from rc_aslo_xf.dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_ALL_PATCH,
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
    MANDATORY_ARMS,
    STATUS_READY,
    build_three_arm_family,
    seal_cw1_multitile_population,
)
from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (
    BINDING_C_LOCAL_COMPONENT,
    BINDING_JOINT_ROOT_COMPONENT_REORDER,
    BINDING_REAL,
    FIXED_DIRECTIONS,
    TERM_H0,
    TERM_READY,
    CW1MultiTileVDecodeError,
    CandidateReferenceFieldV1,
    DirectionalCandidateScopeV1,
    QueryTokenFieldV1,
    derange_local_component_binding_scope,
    decode_ordered_root_three_arm_pair,
    jointly_reorder_root_component_scope,
    registered_reducer_source_sha256,
    token_tensor_sha256,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import PairEvidence


QUERY_SHA = "a" * 64
COLNOMIC_PROCESSOR_SHA = "c" * 64
DINO_PROCESSOR_SHA = "d" * 64
MODEL_SHA = "e" * 64
COMPARATOR_SHA = "f" * 64
REDUCER_SHA = registered_reducer_source_sha256()
QUERY_CACHE_SHA = "3" * 64
REFERENCE_CACHE_SHA = {
    "candidate-g": "4" * 64,
    "candidate-c": "5" * 64,
}
PHYSICAL_ROWS = {"candidate-g": 17, "candidate-c": 18}


class _FixtureDecoder:
    """Small differentiable decoder that exposes the exact supplied masks."""

    def __init__(self, *, model_sha256: str = MODEL_SHA) -> None:
        self.model_checkpoint_sha256 = model_sha256
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

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
        qmask = torch.as_tensor(query_mask, dtype=torch.bool).flatten()
        rmask = torch.as_tensor(reference_mask, dtype=torch.bool).flatten()
        assert qmask.shape == (math.prod(query_grid),)
        assert rmask.shape == (math.prod(reference_grid),)
        self.calls.append((qmask.detach().clone(), rmask.detach().clone()))
        query_value = query_layers[0, :, 0]
        reference_value = reference_layers[0, :, 0]
        reference_mean = reference_value[rmask.to(reference_value.device)].mean()
        active = qmask.to(device=query_value.device, dtype=query_value.dtype)
        relational = torch.stack(
            (query_value * reference_mean, query_value.square() * reference_mean),
            dim=1,
        ) * active[:, None]
        return SimpleNamespace(relational=relational)


class _FixtureComparator:
    """Strictly antisymmetric comparator with exact per-query reconstruction."""

    def __init__(self, *, comparator_sha256: str = COMPARATOR_SHA) -> None:
        self.pair_comparator_sha256 = comparator_sha256

    def compare_relational(
        self,
        relational_g: torch.Tensor,
        relational_c: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> PairEvidence:
        mask = torch.as_tensor(query_mask, dtype=torch.bool).flatten().to(
            relational_g.device
        )
        active = mask.to(relational_g.dtype)
        signed = (relational_g[:, 0] - relational_c[:, 0]) * active
        modulation = active
        evidence = signed
        contributions = signed
        logit = contributions.sum() / mask.sum().to(contributions.dtype)
        return PairEvidence(
            signed_support=signed,
            modulation=modulation,
            evidence=evidence,
            contributions=contributions,
            logit=logit,
        )


def _block(shape: tuple[int, int], y: int, x: int) -> torch.Tensor:
    value = torch.zeros(shape, dtype=torch.bool)
    value[y : y + 2, x : x + 2] = True
    return value.flatten()


def _reference_sha(candidate_key: str) -> str:
    return ("b" if candidate_key == "candidate-g" else "2") * 64


def _geometries(candidate_key: str):
    reference_sha = _reference_sha(candidate_key)
    reference_key = f"reference-{candidate_key}"
    cq = regular_grid_geometry(
        source_image_sha256=QUERY_SHA,
        source_key="query-fixture",
        processor_config_sha256=COLNOMIC_PROCESSOR_SHA,
        grid_shape=(8, 8),
    )
    cr = regular_grid_geometry(
        source_image_sha256=reference_sha,
        source_key=reference_key,
        processor_config_sha256=COLNOMIC_PROCESSOR_SHA,
        grid_shape=(12, 12),
    )
    dq = regular_grid_geometry(
        source_image_sha256=QUERY_SHA,
        source_key="query-fixture",
        processor_config_sha256=DINO_PROCESSOR_SHA,
        grid_shape=(10, 12),
    )
    dr = regular_grid_geometry(
        source_image_sha256=reference_sha,
        source_key=reference_key,
        processor_config_sha256=DINO_PROCESSOR_SHA,
        grid_shape=(14, 16),
    )
    return cq, cr, dq, dr


def _components(candidate_key: str, ready_roots: set[int] | None = None):
    roots = enumerate_query_macro_seeds((8, 8))
    ready = set(range(len(roots))) if ready_roots is None else set(ready_roots)
    locations = (
        ((0, 0), (0, 9), (9, 0), (9, 9))
        if candidate_key == "candidate-g"
        else ((2, 2), (2, 7), (7, 2), (7, 7))
    )
    return {
        root: (_block((12, 12), *locations[root]) if root in ready else None)
        for root in range(len(roots))
    }


def _population(
    candidate_key: str,
    direction: str,
    *,
    ready_roots: set[int] | None = None,
):
    cq, cr, dq, dr = _geometries(candidate_key)
    return seal_cw1_multitile_population(
        candidate_key=candidate_key,
        direction=direction,
        reference_component_by_root=_components(candidate_key, ready_roots),
        colnomic_query_geometry=cq,
        colnomic_reference_geometry=cr,
        dino_query_geometry=dq,
        dino_reference_geometry=dr,
    )


def _scope(
    candidate_key: str,
    direction: str,
    *,
    query: QueryTokenFieldV1,
    candidate: CandidateReferenceFieldV1,
    bank_ordinal: int,
    ready_roots: set[int] | None = None,
    model_sha: str = MODEL_SHA,
    comparator_sha: str = COMPARATOR_SHA,
    reducer_sha: str = REDUCER_SHA,
) -> DirectionalCandidateScopeV1:
    population = _population(candidate_key, direction, ready_roots=ready_roots)
    _, _, dq, dr = _geometries(candidate_key)
    family = build_three_arm_family(
        population,
        bank_ordinal=bank_ordinal,
        dino_query_valid_patch_mask=dq.valid_patch_mask,
        dino_reference_valid_patch_mask=dr.valid_patch_mask,
        model_checkpoint_sha256=model_sha,
        pair_comparator_sha256=comparator_sha,
        reducer_sha256=reducer_sha,
    )
    return DirectionalCandidateScopeV1(
        population=population,
        family=family,
        query_source_image_sha256=query.source_image_sha256,
        query_source_key=query.source_key,
        query_cache_payload_sha256=query.cache_payload_sha256,
        query_geometry_record_sha256=query.geometry_record_sha256,
        query_tokens_sha256=query.tokens_sha256,
        reference_physical_gallery_row=candidate.physical_gallery_row,
        reference_source_image_sha256=candidate.source_image_sha256,
        reference_source_key=candidate.source_key,
        reference_cache_payload_sha256=candidate.cache_payload_sha256,
        reference_geometry_record_sha256=candidate.geometry_record_sha256,
        reference_tokens_sha256=candidate.tokens_sha256,
    )


def _first_joint_ready_ordinal() -> int:
    populations = [
        _population(candidate, direction)
        for candidate in ("candidate-g", "candidate-c")
        for direction in FIXED_DIRECTIONS
    ]
    for ordinal in range(len(populations[0].rows)):
        if all(item.rows[ordinal].status == STATUS_READY for item in populations):
            return ordinal
    raise AssertionError("fixture has no jointly ready structural row")


def _inputs(
    *,
    query_requires_grad: bool = False,
    ready_roots: dict[tuple[str, str], set[int] | None] | None = None,
    model_sha_by_scope: dict[tuple[str, str], str] | None = None,
    comparator_sha_by_scope: dict[tuple[str, str], str] | None = None,
    reducer_sha_by_scope: dict[tuple[str, str], str] | None = None,
    bank_ordinal: int | None = None,
):
    ordinal = _first_joint_ready_ordinal() if bank_ordinal is None else bank_ordinal
    _, _, dq, _ = _geometries("candidate-g")
    query_values = torch.linspace(
        0.2, 1.2, math.prod(dq.grid_shape), dtype=torch.float64
    ).reshape(1, -1, 1)
    query_values.requires_grad_(query_requires_grad)
    query = QueryTokenFieldV1(
        layers=query_values,
        grid_shape=dq.grid_shape,
        valid_patch_mask=dq.valid_patch_mask,
        source_image_sha256=QUERY_SHA,
        source_key=dq.source_key,
        cache_payload_sha256=QUERY_CACHE_SHA,
        geometry_record_sha256=dq.sha256,
        tokens_sha256=token_tensor_sha256(query_values),
    )
    candidates: dict[str, CandidateReferenceFieldV1] = {}
    for index, candidate_key in enumerate(("candidate-g", "candidate-c")):
        _, _, _, dr = _geometries(candidate_key)
        values = torch.linspace(
            0.1 + index,
            1.1 + 0.5 * index,
            math.prod(dr.grid_shape),
            dtype=torch.float64,
        ).reshape(1, -1, 1)
        candidates[candidate_key] = CandidateReferenceFieldV1(
            candidate_key=candidate_key,
            layers=values,
            grid_shape=dr.grid_shape,
            valid_patch_mask=dr.valid_patch_mask,
            physical_gallery_row=PHYSICAL_ROWS[candidate_key],
            source_image_sha256=_reference_sha(candidate_key),
            source_key=dr.source_key,
            cache_payload_sha256=REFERENCE_CACHE_SHA[candidate_key],
            geometry_record_sha256=dr.sha256,
            tokens_sha256=token_tensor_sha256(values),
        )
    ready_roots = ready_roots or {}
    model_sha_by_scope = model_sha_by_scope or {}
    comparator_sha_by_scope = comparator_sha_by_scope or {}
    reducer_sha_by_scope = reducer_sha_by_scope or {}
    scopes = {
        (candidate, direction): _scope(
            candidate,
            direction,
            query=query,
            candidate=candidates[candidate],
            bank_ordinal=ordinal,
            ready_roots=ready_roots.get((candidate, direction)),
            model_sha=model_sha_by_scope.get((candidate, direction), MODEL_SHA),
            comparator_sha=comparator_sha_by_scope.get(
                (candidate, direction), COMPARATOR_SHA
            ),
            reducer_sha=reducer_sha_by_scope.get(
                (candidate, direction), REDUCER_SHA
            ),
        )
        for candidate in ("candidate-g", "candidate-c")
        for direction in FIXED_DIRECTIONS
    }
    return query, candidates, scopes


def _decode(
    query: QueryTokenFieldV1,
    candidates: dict[str, CandidateReferenceFieldV1],
    scopes: dict[tuple[str, str], DirectionalCandidateScopeV1],
    *,
    left: str = "candidate-g",
    right: str = "candidate-c",
):
    decoder = _FixtureDecoder()
    result = decode_ordered_root_three_arm_pair(
        decoder,
        _FixtureComparator(),
        query,
        candidates,
        scopes,
        left_candidate_key=left,
        right_candidate_key=right,
    )
    return result, decoder


def _mask_sha(mask: torch.Tensor, shape: tuple[int, int]) -> str:
    flat = torch.as_tensor(mask, dtype=torch.uint8).flatten().contiguous()
    payload = (
        int(shape[0]).to_bytes(8, "little")
        + int(shape[1]).to_bytes(8, "little")
        + flat.numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def test_three_arms_use_one_model_and_local_decodes_ordered_components_not_union():
    query, candidates, scopes = _inputs()
    result, _ = _decode(query, candidates, scopes)
    assert tuple(result.by_name()) == MANDATORY_ARMS
    assert result.model_checkpoint_sha256 == MODEL_SHA

    full = result.by_name()[ARM_QUERY_FULL_REFERENCE]
    local = result.by_name()[ARM_QUERY_LOCAL_COMPONENTS]
    for full_term, local_term in zip(
        (*full.forward_by_direction, *full.reverse_by_direction),
        (*local.forward_by_direction, *local.reverse_by_direction),
        strict=True,
    ):
        assert torch.equal(full_term.query_mask, local_term.query_mask)
        assert full_term.status == local_term.status == TERM_READY

    for owner_key, terms in (
        ("candidate-g", local.forward_by_direction),
        ("candidate-c", local.reverse_by_direction),
    ):
        reference_shape = candidates[owner_key].grid_shape
        for term in terms:
            row = scopes[(owner_key, term.direction)].row
            exact_hashes = {
                binding.root_ordinal: _mask_sha(
                    binding.dino_reference_component_mask, reference_shape
                )
                for binding in row.root_bindings
            }
            union = torch.zeros(math.prod(reference_shape), dtype=torch.bool)
            for binding in row.root_bindings:
                union |= binding.dino_reference_component_mask
            union_hash = _mask_sha(union, reference_shape)
            assert all(
                receipt.owner_reference_mask_sha256
                == exact_hashes[receipt.root_ordinal]
                for receipt in term.decode_receipts
            )
            assert all(
                receipt.owner_reference_mask_sha256 != union_hash
                for receipt in term.decode_receipts
            )

    for owner_key, terms in (
        ("candidate-g", full.forward_by_direction),
        ("candidate-c", full.reverse_by_direction),
    ):
        full_hash = _mask_sha(
            candidates[owner_key].valid_patch_mask,
            candidates[owner_key].grid_shape,
        )
        assert all(
            receipt.owner_reference_mask_sha256 == full_hash
            for term in terms
            for receipt in term.decode_receipts
        )


def test_pair_swap_is_exactly_antisymmetric_for_every_arm():
    query, candidates, scopes = _inputs()
    forward, _ = _decode(query, candidates, scopes)
    swapped, _ = _decode(
        query, candidates, scopes, left="candidate-c", right="candidate-g"
    )
    for arm_name in MANDATORY_ARMS:
        left = forward.by_name()[arm_name]
        right = swapped.by_name()[arm_name]
        assert torch.allclose(left.logit, -right.logit, atol=1.0e-12, rtol=0.0)
        assert torch.allclose(
            left.scalar_contributions,
            -right.scalar_contributions,
            atol=1.0e-12,
            rtol=0.0,
        )


def test_candidate_container_reorder_is_invariant():
    query, candidates, scopes = _inputs()
    original, _ = _decode(query, candidates, scopes)
    reordered_candidates = dict(reversed(tuple(candidates.items())))
    reordered_scopes = dict(reversed(tuple(scopes.items())))
    reordered, _ = _decode(query, reordered_candidates, reordered_scopes)
    for arm_name in MANDATORY_ARMS:
        first = original.by_name()[arm_name]
        second = reordered.by_name()[arm_name]
        assert torch.equal(first.logit, second.logit)
        assert torch.equal(first.scalar_contributions, second.scalar_contributions)


def test_joint_root_component_reorder_is_exact_evidence_invariant():
    query, candidates, scopes = _inputs()
    clean, _ = _decode(query, candidates, scopes)
    reordered = {}
    for key, scope in scopes.items():
        roots = scope.row.structural_region.contributing_root_ordinals
        reordered[key] = jointly_reorder_root_component_scope(
            scope, tuple(reversed(roots))
        )
    observed, _ = _decode(query, candidates, reordered)
    assert observed.binding_mode == BINDING_JOINT_ROOT_COMPONENT_REORDER
    for arm_name in MANDATORY_ARMS:
        expected_arm = clean.by_name()[arm_name]
        observed_arm = observed.by_name()[arm_name]
        assert torch.equal(expected_arm.logit, observed_arm.logit)
        assert torch.equal(
            expected_arm.scalar_contributions,
            observed_arm.scalar_contributions,
        )
        assert tuple(
            term.decoded_root_ordinals
            for term in (*expected_arm.forward_by_direction, *expected_arm.reverse_by_direction)
        ) == tuple(
            term.decoded_root_ordinals
            for term in (*observed_arm.forward_by_direction, *observed_arm.reverse_by_direction)
        )


def test_component_only_derangement_executes_and_changes_only_local_addressing():
    query, candidates, scopes = _inputs()
    clean, _ = _decode(query, candidates, scopes)
    destroyed = {}
    for key, scope in scopes.items():
        roots = scope.row.structural_region.contributing_root_ordinals
        component_sources = roots[1:] + roots[:1]
        destroyed[key] = derange_local_component_binding_scope(
            scope, component_sources
        )
    control, _ = _decode(query, candidates, destroyed)
    assert control.binding_mode == BINDING_C_LOCAL_COMPONENT
    for arm_name in (ARM_ALL_PATCH, ARM_QUERY_FULL_REFERENCE):
        assert torch.equal(
            clean.by_name()[arm_name].logit,
            control.by_name()[arm_name].logit,
        )
        assert torch.equal(
            clean.by_name()[arm_name].scalar_contributions,
            control.by_name()[arm_name].scalar_contributions,
        )
    clean_local = clean.by_name()[ARM_QUERY_LOCAL_COMPONENTS]
    control_local = control.by_name()[ARM_QUERY_LOCAL_COMPONENTS]
    assert not torch.equal(clean_local.logit, control_local.logit)
    for clean_term, control_term in zip(
        (*clean_local.forward_by_direction, *clean_local.reverse_by_direction),
        (*control_local.forward_by_direction, *control_local.reverse_by_direction),
        strict=True,
    ):
        assert clean_term.query_mask.numpy().tobytes() == control_term.query_mask.numpy().tobytes()
        assert sorted(
            item.owner_reference_mask_sha256 for item in clean_term.decode_receipts
        ) == sorted(
            item.owner_reference_mask_sha256 for item in control_term.decode_receipts
        )
        assert all(
            item.owner_component_source_root_ordinal != item.root_ordinal
            and item.opponent_component_source_root_ordinal != item.root_ordinal
            and item.binding_mode == BINDING_C_LOCAL_COMPONENT
            for item in control_term.decode_receipts
        )


def test_local_arm_is_zero_and_has_zero_query_gradient_outside_shared_union():
    query, candidates, scopes = _inputs(query_requires_grad=True)
    result, _ = _decode(query, candidates, scopes)
    local = result.by_name()[ARM_QUERY_LOCAL_COMPONENTS]
    union = scopes[("candidate-g", "a_to_b")].row.dino_query_union_mask
    assert all(
        torch.equal(term.query_mask, union)
        for term in (*local.forward_by_direction, *local.reverse_by_direction)
    )
    assert torch.equal(
        local.scalar_contributions[~union],
        torch.zeros(int((~union).sum()), dtype=torch.float64),
    )
    local.logit.backward()
    gradient = query.layers.grad[0, :, 0]
    assert torch.equal(
        gradient[~union], torch.zeros(int((~union).sum()), dtype=torch.float64)
    )
    assert bool(gradient[union].ne(0).any())


def test_h0_is_exact_zero_but_fixed_four_term_denominator_is_not_renormalized():
    ready = {
        ("candidate-g", "a_to_b"): {0},
        ("candidate-c", "a_to_b"): {0},
    }
    query, candidates, scopes = _inputs(ready_roots=ready, bank_ordinal=0)
    result, _ = _decode(query, candidates, scopes)
    for arm_name in (ARM_QUERY_FULL_REFERENCE, ARM_QUERY_LOCAL_COMPONENTS):
        arm = result.by_name()[arm_name]
        assert arm.forward_by_direction[0].status == TERM_H0
        assert arm.reverse_by_direction[0].status == TERM_H0
        assert arm.forward_by_direction[0].logit.numpy().tobytes() == torch.zeros(
            (), dtype=arm.logit.dtype
        ).numpy().tobytes()
        assert len(arm.signed_term_logits) == 4
        expected = sum(
            arm.signed_term_logits, start=arm.logit.new_zeros(())
        ) / 4.0
        assert torch.equal(arm.logit, expected)
    assert result.by_name()[ARM_ALL_PATCH].forward_by_direction[0].status == TERM_READY


def test_mixed_checkpoint_hashes_fail_closed_before_decode():
    query, candidates, scopes = _inputs(
        model_sha_by_scope={("candidate-c", "b_to_a"): "3" * 64}
    )
    decoder = _FixtureDecoder()
    with pytest.raises(CW1MultiTileVDecodeError, match="lineage"):
        decode_ordered_root_three_arm_pair(
            decoder,
            _FixtureComparator(),
            query,
            candidates,
            scopes,
            left_candidate_key="candidate-g",
            right_candidate_key="candidate-c",
        )
    assert decoder.calls == []


def test_wrong_tokens_under_the_right_candidate_key_abort_before_forward():
    query, candidates, scopes = _inputs()
    wrong_layers = candidates["candidate-c"].layers.clone()
    with pytest.raises(CW1MultiTileVDecodeError, match="token tensor/cache"):
        replace(candidates["candidate-g"], layers=wrong_layers)

    decoder = _FixtureDecoder()
    bad_query = replace(query, cache_payload_sha256="6" * 64)
    with pytest.raises(CW1MultiTileVDecodeError, match="query token provenance"):
        decode_ordered_root_three_arm_pair(
            decoder,
            _FixtureComparator(),
            bad_query,
            candidates,
            scopes,
            left_candidate_key="candidate-g",
            right_candidate_key="candidate-c",
        )
    assert decoder.calls == []


@pytest.mark.parametrize("field", ("query", "reference"))
def test_in_place_token_mutation_aborts_before_forward(field: str):
    query, candidates, scopes = _inputs()
    if field == "query":
        query.layers.add_(1.0)
    else:
        candidates["candidate-g"].layers.add_(1.0)

    decoder = _FixtureDecoder()
    with pytest.raises(CW1MultiTileVDecodeError, match="mutated after provenance seal"):
        decode_ordered_root_three_arm_pair(
            decoder,
            _FixtureComparator(),
            query,
            candidates,
            scopes,
            left_candidate_key="candidate-g",
            right_candidate_key="candidate-c",
        )
    assert decoder.calls == []


def test_reference_row_source_cache_and_geometry_bindings_fail_closed():
    query, candidates, scopes = _inputs()
    decoder = _FixtureDecoder()
    bad_scopes = dict(scopes)
    bad_scopes[("candidate-g", "a_to_b")] = replace(
        scopes[("candidate-g", "a_to_b")],
        reference_physical_gallery_row=999,
    )
    with pytest.raises(CW1MultiTileVDecodeError, match="reference token provenance"):
        decode_ordered_root_three_arm_pair(
            decoder,
            _FixtureComparator(),
            query,
            candidates,
            bad_scopes,
            left_candidate_key="candidate-g",
            right_candidate_key="candidate-c",
        )
    assert decoder.calls == []

    with pytest.raises(CW1MultiTileVDecodeError, match="population geometry"):
        replace(
            scopes[("candidate-g", "a_to_b")],
            reference_geometry_record_sha256="7" * 64,
        )


@pytest.mark.parametrize("lineage", ("decoder", "comparator", "reducer"))
def test_actual_runtime_lineage_is_closed_before_forward(lineage: str):
    reducer_map = (
        {("candidate-g", "a_to_b"): "8" * 64}
        if lineage == "reducer"
        else None
    )
    query, candidates, scopes = _inputs(reducer_sha_by_scope=reducer_map)
    decoder = _FixtureDecoder(
        model_sha256="9" * 64 if lineage == "decoder" else MODEL_SHA
    )
    comparator = _FixtureComparator(
        comparator_sha256="0" * 64 if lineage == "comparator" else COMPARATOR_SHA
    )
    with pytest.raises(CW1MultiTileVDecodeError, match="lineage"):
        decode_ordered_root_three_arm_pair(
            decoder,
            comparator,
            query,
            candidates,
            scopes,
            left_candidate_key="candidate-g",
            right_candidate_key="candidate-c",
        )
    assert decoder.calls == []


def test_mixed_binding_modes_abort_before_forward():
    query, candidates, scopes = _inputs()
    roots = scopes[("candidate-g", "a_to_b")].row.structural_region.contributing_root_ordinals
    mixed = dict(scopes)
    mixed[("candidate-g", "a_to_b")] = jointly_reorder_root_component_scope(
        scopes[("candidate-g", "a_to_b")], tuple(reversed(roots))
    )
    decoder = _FixtureDecoder()
    with pytest.raises(CW1MultiTileVDecodeError, match="mix REAL"):
        decode_ordered_root_three_arm_pair(
            decoder,
            _FixtureComparator(),
            query,
            candidates,
            mixed,
            left_candidate_key="candidate-g",
            right_candidate_key="candidate-c",
        )
    assert decoder.calls == []
