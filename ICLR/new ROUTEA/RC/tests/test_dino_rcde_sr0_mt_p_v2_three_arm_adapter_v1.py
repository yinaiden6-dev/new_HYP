from __future__ import annotations

import copy
from dataclasses import fields
import math
from types import SimpleNamespace

import pytest
import torch

from rc_aslo_xf.dino_rcde_cw1_multitile_superregion_v2 import (
    ARM_QUERY_FULL_REFERENCE,
    ARM_QUERY_LOCAL_COMPONENTS,
)
from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (
    CandidateReferenceFieldV1,
    QueryTokenFieldV1,
    token_tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_adapter_v2 import build_natural_p_lock_v2
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_source_v1 import candidate_key_v1
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (
    DirectionProjectionReceiptV1,
    GeometryNamespaceBindingV1,
    NaturalPLockV2ThreeArmAdapterError,
    inner_oof_head_spec,
    outer_refit_head_spec,
    project_natural_v2_candidate_to_v1_runtime,
    select_natural_v2_head_records,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import decode_pair
from rc_aslo_xf.dino_rcde_v1_2_resource_core import PairEvidence
from rc_aslo_xf.cw1_sr0_structure_v1 import enumerate_superregion_bank


GRID = (8, 8)
QUERY_SOURCE = canonical_sha256({"fixture": "query-source"})
QUERY_CANONICAL_GEOMETRY = canonical_sha256({"fixture": "query-canonical-geometry"})
QUERY_CACHE_GEOMETRY = canonical_sha256({"fixture": "query-cache-geometry"})


def _sha(name: str) -> str:
    return canonical_sha256({"adapter-fixture": name})


def _rle(indices: tuple[int, ...]) -> dict[str, object]:
    runs: list[list[int]] = []
    for index in sorted(indices):
        if runs and runs[-1][0] + runs[-1][1] == index:
            runs[-1][1] += 1
        else:
            runs.append([index, 1])
    return {"grid_shape": list(GRID), "rle": runs}


ROOT_MASKS = (
    (0, 1, 8, 9),
    (2, 3, 10, 11),
    (16, 17, 24, 25),
    (18, 19, 26, 27),
)


def _record(
    *,
    candidate_position: int,
    physical_row: int,
    direction: str,
    spec,
    missing_root: int | None = None,
    reverse_scores: bool = False,
) -> dict[str, object]:
    reference_source = _sha(f"reference-source-{physical_row}")
    reference_canonical_geometry = _sha(f"reference-canonical-geometry-{physical_row}")
    candidate_key = candidate_key_v1(
        physical_row=physical_row, source_image_sha256=reference_source
    )
    source = {
        "query_id": "FIXTURE-Q",
        "execution_ordinal": 7,
        "query_source_image_sha256": QUERY_SOURCE,
        "candidate_physical_row": physical_row,
        "candidate_reference_source_sha256": reference_source,
        "query_geometry_sha256": QUERY_CANONICAL_GEOMETRY,
        "reference_geometry_sha256": reference_canonical_geometry,
        "candidate_position": candidate_position,
        "candidate_key": candidate_key,
        # The formal producer stores FactorizedDirectionalSourceIndex.population_sha256
        # in this legacy-named field.  That population hash includes direction and
        # therefore must differ across a_to_b / b_to_a for the same candidate.
        "cache_sha256": _sha(
            f"candidate-directional-population-{physical_row}-{direction}"
        ),
        "direction": direction,
        "colnomic_query_grid_shape": list(GRID),
        "deployment_query_grid_shape": list(GRID),
        "deployment_reference_grid_shape": list(GRID),
        "root_count": 4,
        "action_count": 1,
        "deployment_query_root_masks": [_rle(value) for value in ROOT_MASKS],
        "deployment_reference_mask_table": [_rle(value) for value in ROOT_MASKS],
        "deployment_reference_mask_index": [[index] for index in range(4)],
        "action_keys": [[_sha(f"action-{physical_row}-{index}")] for index in range(4)],
        "eligibility_rle": {
            "shape": [4, 1],
            "flatten_order": "ROW_MAJOR",
            "true_runs": [[0, 4]],
        },
    }
    valid = torch.ones(math.prod(GRID), dtype=torch.bool)
    query_geometry = {
        "query_id": "FIXTURE-Q",
        "execution_ordinal": 7,
        "source_image_sha256": QUERY_SOURCE,
        "dino_geometry": {
            "geometry_sha256": QUERY_CANONICAL_GEOMETRY,
            "grid_shape": list(GRID),
            "valid_patch_mask": valid,
            "valid_patch_mask_sha256": _sha("query-source-valid-mask"),
        },
    }
    reference_geometry = {
        "physical_row": physical_row,
        "source_image_sha256": reference_source,
        "dino_geometry": {
            "geometry_sha256": reference_canonical_geometry,
            "grid_shape": list(GRID),
            "valid_patch_mask": valid,
            "valid_patch_mask_sha256": _sha(f"reference-source-valid-mask-{physical_row}"),
        },
    }
    fold = {
        "query_id": "FIXTURE-Q",
        "query_ordinal": 70,
        "source_image_sha256": QUERY_SOURCE,
        "inner_fold": spec.source_fold,
        "track": "fixture-natural",
    }
    membership = {
        "fit_id": spec.fit_id,
        "outer_fold": spec.outer_fold,
        "inner_heldout_fold": spec.inner_heldout_fold,
        "heldout_addresses": [{"execution_ordinal": 7, "query_id": "FIXTURE-Q"}],
    }
    bank = enumerate_superregion_bank(GRID, include_r0_control=False)
    scores = torch.arange(len(bank), dtype=torch.float64)
    if not reverse_scores:
        scores = -scores
    selected_actions = [0, 0, 0, 0]
    if missing_root is not None:
        selected_actions[missing_root] = None
    return build_natural_p_lock_v2(
        source_record=source,
        query_geometry_record=query_geometry,
        reference_geometry_record=reference_geometry,
        fold_record=fold,
        membership=membership,
        row_scores=scores,
        selected_action_indices=selected_actions,
        fit_id=spec.fit_id,
        crossfit_role=spec.crossfit_role,
        p_checkpoint_sha256=_sha(f"checkpoint-{spec.fit_id}"),
        p_training_manifest_sha256=_sha(f"manifest-{spec.fit_id}"),
    )


def _token_fields(rows: tuple[int, ...] = (9, 10)):
    query_layers = torch.arange(64 * 4, dtype=torch.float64).reshape(1, 64, 4) / 100.0
    valid = torch.ones(64, dtype=torch.bool)
    query = QueryTokenFieldV1(
        layers=query_layers,
        grid_shape=GRID,
        valid_patch_mask=valid,
        source_image_sha256=QUERY_SOURCE,
        source_key="query_execution:7",
        cache_payload_sha256=_sha("query-cache-payload"),
        geometry_record_sha256=QUERY_CACHE_GEOMETRY,
        tokens_sha256=token_tensor_sha256(query_layers),
    )
    candidates = {}
    for index, row in enumerate(rows):
        source = _sha(f"reference-source-{row}")
        layers = torch.full((1, 64, 4), float(index + 1), dtype=torch.float64)
        field = CandidateReferenceFieldV1(
            candidate_key=candidate_key_v1(
                physical_row=row, source_image_sha256=source
            ),
            layers=layers,
            grid_shape=GRID,
            valid_patch_mask=valid,
            physical_gallery_row=row,
            source_image_sha256=source,
            source_key=f"reference_physical_row:{row}",
            cache_payload_sha256=_sha(f"reference-cache-payload-{row}"),
            geometry_record_sha256=_sha(f"reference-cache-geometry-{row}"),
            tokens_sha256=token_tensor_sha256(layers),
        )
        candidates[field.candidate_key] = field
    return query, candidates


def _bindings(query: QueryTokenFieldV1, candidate: CandidateReferenceFieldV1):
    return (
        GeometryNamespaceBindingV1(
            source_image_sha256=query.source_image_sha256,
            grid_shape=query.grid_shape,
            canonical_geometry_sha256=QUERY_CANONICAL_GEOMETRY,
            cache_geometry_record_sha256=query.geometry_record_sha256,
        ),
        GeometryNamespaceBindingV1(
            source_image_sha256=candidate.source_image_sha256,
            grid_shape=candidate.grid_shape,
            canonical_geometry_sha256=_sha(
                f"reference-canonical-geometry-{candidate.physical_gallery_row}"
            ),
            cache_geometry_record_sha256=candidate.geometry_record_sha256,
        ),
    )


def _head_fixture(spec=None):
    spec = outer_refit_head_spec(source_fold=1) if spec is None else spec
    records = [
        _record(
            candidate_position=position,
            physical_row=row,
            direction=direction,
            spec=spec,
            reverse_scores=(position == 1),
        )
        for position, row in enumerate((9, 10))
        for direction in ("a_to_b", "b_to_a")
    ]
    query, candidates = _token_fields()
    selected = select_natural_v2_head_records(
        records, spec=spec, expected_candidate_count=2
    )
    by_candidate: dict[str, dict[str, dict[str, object]]] = {}
    for record in selected:
        by_candidate.setdefault(record["candidate"]["candidate_key"], {})[
            record["direction"]
        ] = record
    projections = {}
    for key, candidate in candidates.items():
        query_binding, reference_binding = _bindings(query, candidate)
        projections[key] = project_natural_v2_candidate_to_v1_runtime(
            query=query,
            candidate=candidate,
            direction_records=by_candidate[key],
            spec=spec,
            query_geometry_binding=query_binding,
            reference_geometry_binding=reference_binding,
        )
    return spec, records, query, candidates, projections


class _SharedSyntheticV(torch.nn.Module):
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
        mean = reference_layers[0, rmask, 0].mean()
        relational = torch.stack((query_layers[0, :, 0] * mean, query_layers[0, :, 1] * mean), dim=1)
        return SimpleNamespace(relational=relational * qmask[:, None] * self.scale)

    def compare_relational(self, relational_g, relational_c, query_mask):
        mask = torch.as_tensor(
            query_mask, dtype=torch.bool, device=relational_g.device
        ).flatten()
        signed = (relational_g[:, 0] - relational_c[:, 0]) * mask
        return PairEvidence(
            signed_support=signed,
            modulation=mask.to(signed.dtype),
            evidence=signed,
            contributions=signed,
            logit=signed.sum() / mask.sum(),
        )


def test_projects_exact_runtime_view_and_keeps_hash_namespaces_distinct() -> None:
    _, records, query, _, projections = _head_fixture()
    for projection in projections.values():
        runtime = projection.runtime_lock
        for direction, sealed in runtime.direction_locks.items():
            source = next(
                record
                for record in records
                if record["candidate"]["candidate_key"] == runtime.candidate.candidate_key
                and record["direction"] == direction
            )
            assert sealed.p_lock_record_sha256 == source["record_sha256"]
            assert sealed.selected_row_sha256 == source["selected"]["selected_bank_row_sha256"]
            assert sealed.selected_row_sha256 != source["selected"]["selected_core_row_sha256"]
            assert torch.equal(
                sealed.query_union_mask,
                torch.as_tensor(source["fixed_denominator"]["query_union"]),
            )
            receipt = projection.direction_receipts[direction]
            assert receipt.query_geometry_namespace.canonical_geometry_sha256 == QUERY_CANONICAL_GEOMETRY
            assert receipt.query_geometry_namespace.cache_geometry_record_sha256 == query.geometry_record_sha256
            assert receipt.query_valid_hashes.v2_tensor_sha256 != receipt.query_valid_hashes.v1_p_tensor_sha256
            for root in sealed.all_roots.values():
                assert root.query_geometry_sha256 == query.geometry_record_sha256


def test_projection_receipt_has_one_exact_field_set_and_self_hashes() -> None:
    _, _, _, _, projections = _head_fixture()
    names = tuple(item.name for item in fields(DirectionProjectionReceiptV1))
    assert len(names) == len(set(names))
    assert names == (
        "direction",
        "natural_record_sha256",
        "core_record_sha256",
        "core_geometry_signature_sha256",
        "query_geometry_namespace",
        "reference_geometry_namespace",
        "query_valid_hashes",
        "reference_valid_hashes",
        "query_union_hashes",
        "root_mask_hashes",
        "logical_sha256",
    )
    receipt = next(iter(next(iter(projections.values())).direction_receipts.values()))
    payload = receipt.payload()
    assert set(payload) == {
        "schema_version",
        "direction",
        "natural_record_sha256",
        "core_record_sha256",
        "core_geometry_signature_sha256",
        "query_canonical_geometry_sha256",
        "query_cache_geometry_record_sha256",
        "reference_canonical_geometry_sha256",
        "reference_cache_geometry_record_sha256",
        "query_valid_hashes",
        "reference_valid_hashes",
        "query_union_hashes",
        "root_mask_hashes",
        "logical_sha256",
    }
    assert payload["logical_sha256"] == canonical_sha256(
        {key: value for key, value in payload.items() if key != "logical_sha256"}
    )


def test_adapted_view_runs_one_shared_three_arm_decoder_and_pair_swap() -> None:
    _, _, query, _, projections = _head_fixture()
    locks = {key: value.runtime_lock for key, value in projections.items()}
    left, right = tuple(locks)
    model = _SharedSyntheticV()
    real = decode_pair(model, query, locks, left, right)
    swapped = decode_pair(model, query, locks, right, left)
    for name, arm in real.by_name().items():
        assert torch.allclose(
            arm.logit, -swapped.by_name()[name].logit, atol=1e-12, rtol=0
        )
    full = real.by_name()[ARM_QUERY_FULL_REFERENCE]
    local = real.by_name()[ARM_QUERY_LOCAL_COMPONENTS]
    for full_term, local_term in zip(
        full.forward_by_direction, local.forward_by_direction, strict=True
    ):
        assert torch.equal(full_term.query_mask, local_term.query_mask)
        assert full_term.decode_receipts[0].owner_reference_mask_sha256


def test_reference_missing_is_never_silently_downgraded_to_v1_h0() -> None:
    spec, records, query, candidates, _ = _head_fixture()
    candidate = min(candidates.values(), key=lambda item: item.physical_gallery_row)
    missing = _record(
        candidate_position=0,
        physical_row=candidate.physical_gallery_row,
        direction="a_to_b",
        spec=spec,
        missing_root=0,
    )
    direction_records = {
        direction: next(
            record
            for record in records
            if record["candidate"]["candidate_key"] == candidate.candidate_key
            and record["direction"] == direction
        )
        for direction in ("a_to_b", "b_to_a")
    }
    direction_records["a_to_b"] = missing
    q_binding, r_binding = _bindings(query, candidate)
    with pytest.raises(
        NaturalPLockV2ThreeArmAdapterError, match="ROOT_REFERENCE_MISSING"
    ):
        project_natural_v2_candidate_to_v1_runtime(
            query=query,
            candidate=candidate,
            direction_records=direction_records,
            spec=spec,
            query_geometry_binding=q_binding,
            reference_geometry_binding=r_binding,
        )


def test_head_selection_is_explicit_for_inner_oof_and_outer_refit() -> None:
    outer_spec, outer_records, *_ = _head_fixture()
    inner_spec = inner_oof_head_spec(outer_fold=2, source_fold=1)
    inner_records = [
        _record(
            candidate_position=position,
            physical_row=row,
            direction=direction,
            spec=inner_spec,
        )
        for position, row in enumerate((9, 10))
        for direction in ("a_to_b", "b_to_a")
    ]
    population = [*reversed(inner_records), *outer_records]
    chosen = select_natural_v2_head_records(
        population, spec=inner_spec, expected_candidate_count=2
    )
    assert {item["query"]["fit_id"] for item in chosen} == {inner_spec.fit_id}
    assert {item["query"]["crossfit_role"] for item in chosen} == {
        "INNER_HELDOUT_DEPLOYMENT"
    }
    assert outer_spec.fit_id != inner_spec.fit_id
    with pytest.raises(
        NaturalPLockV2ThreeArmAdapterError, match="candidate_count x two directions"
    ):
        select_natural_v2_head_records(
            population[:-1], spec=outer_spec, expected_candidate_count=2
        )


def test_head_selection_separates_candidate_binding_from_directional_population() -> None:
    spec, records, *_ = _head_fixture()
    selected = select_natural_v2_head_records(
        records, spec=spec, expected_candidate_count=2
    )
    for position in (0, 1):
        hashes = {
            item["candidate"]["candidate_native_content_sha256"]
            for item in selected
            if item["candidate"]["candidate_position"] == position
        }
        assert len(hashes) == 2

    wrong_candidate = _record(
        candidate_position=0,
        physical_row=11,
        direction="b_to_a",
        spec=spec,
    )
    mismatched = [
        wrong_candidate
        if item["candidate"]["candidate_position"] == 0
        and item["direction"] == "b_to_a"
        else item
        for item in records
    ]
    with pytest.raises(
        NaturalPLockV2ThreeArmAdapterError,
        match="candidate binding differs across directions",
    ):
        select_natural_v2_head_records(
            mismatched, spec=spec, expected_candidate_count=2
        )

    duplicated = copy.deepcopy(records)
    for position in (0, 1):
        forward = next(
            item
            for item in duplicated
            if item["candidate"]["candidate_position"] == position
            and item["direction"] == "a_to_b"
        )
        reverse = next(
            item
            for item in duplicated
            if item["candidate"]["candidate_position"] == position
            and item["direction"] == "b_to_a"
        )
        reverse["candidate"]["candidate_native_content_sha256"] = forward[
            "candidate"
        ]["candidate_native_content_sha256"]
        reverse["record_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in reverse.items()
                if key != "record_sha256"
            }
        )
    with pytest.raises(
        NaturalPLockV2ThreeArmAdapterError,
        match="directional source populations are duplicated",
    ):
        select_natural_v2_head_records(
            duplicated, spec=spec, expected_candidate_count=2
        )


def test_nested_target_poison_and_candidate_axis_reorder_fail_closed() -> None:
    spec, records, *_ = _head_fixture()
    poisoned = copy.deepcopy(records)
    poisoned[0]["query"]["target"] = "forbidden"
    poisoned[0]["record_sha256"] = canonical_sha256(
        {key: value for key, value in poisoned[0].items() if key != "record_sha256"}
    )
    with pytest.raises(NaturalPLockV2ThreeArmAdapterError, match="query field-set"):
        select_natural_v2_head_records(
            poisoned, spec=spec, expected_candidate_count=2
        )

    reordered = copy.deepcopy(records)
    for item in reordered:
        item["candidate"]["candidate_position"] = 1 - item["candidate"]["candidate_position"]
        item["record_sha256"] = canonical_sha256(
            {key: value for key, value in item.items() if key != "record_sha256"}
        )
    with pytest.raises(NaturalPLockV2ThreeArmAdapterError, match="physical-row axis"):
        select_natural_v2_head_records(
            reordered, spec=spec, expected_candidate_count=2
        )


def test_geometry_and_mask_hash_namespace_poison_fail_closed() -> None:
    spec, records, query, candidates, _ = _head_fixture()
    candidate = min(candidates.values(), key=lambda item: item.physical_gallery_row)
    direction_records = {
        direction: next(
            record
            for record in records
            if record["candidate"]["candidate_key"] == candidate.candidate_key
            and record["direction"] == direction
        )
        for direction in ("a_to_b", "b_to_a")
    }
    _, r_binding = _bindings(query, candidate)
    wrong_query_binding = GeometryNamespaceBindingV1(
        source_image_sha256=query.source_image_sha256,
        grid_shape=query.grid_shape,
        canonical_geometry_sha256=QUERY_CANONICAL_GEOMETRY,
        cache_geometry_record_sha256=_sha("wrong-cache-geometry-namespace"),
    )
    with pytest.raises(
        NaturalPLockV2ThreeArmAdapterError, match="cache geometry namespace"
    ):
        project_natural_v2_candidate_to_v1_runtime(
            query=query,
            candidate=candidate,
            direction_records=direction_records,
            spec=spec,
            query_geometry_binding=wrong_query_binding,
            reference_geometry_binding=r_binding,
        )

    poisoned = copy.deepcopy(direction_records)
    poisoned["a_to_b"]["query_geometry"]["mask_sha256"] = _sha("wrong-v2-mask")
    poisoned["a_to_b"]["record_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in poisoned["a_to_b"].items()
            if key != "record_sha256"
        }
    )
    q_binding, r_binding = _bindings(query, candidate)
    with pytest.raises(
        NaturalPLockV2ThreeArmAdapterError, match="valid-mask V2 hash"
    ):
        project_natural_v2_candidate_to_v1_runtime(
            query=query,
            candidate=candidate,
            direction_records=poisoned,
            spec=spec,
            query_geometry_binding=q_binding,
            reference_geometry_binding=r_binding,
        )
