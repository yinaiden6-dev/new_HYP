from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))

from materialize_dino_rcde_sr0_mt_p_features_v1 import (  # noqa: E402
    materialize_source_payload,
    synthetic_source,
)
from dino_rcde_sr0_mt_c_col_p_independent_v1 import (  # noqa: E402
    IndependentCColPError,
    assert_observed_matches_independent,
    independent_bundle_fingerprint,
    independent_replay_c_col_p,
)
from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: E402
    CandidateReferenceFieldV1,
    QueryTokenFieldV1,
    token_tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_c_col_p_v1 import (  # noqa: E402
    NAMESPACE,
    build_control_feature_records,
    bundle_fingerprint,
    canonical_donor_positions,
    dino_axis_content_fingerprint,
    donor_receipt,
    replay_c_col_p,
    transport_reference_mask,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    P_DIRECTIONS,
    build_p_lock_record,
    canonical_sha256,
    feature_ledger_index,
    initialize_training,
    score_deployable_direction,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CandidatePLockV1,
    VQueryBundle,
    sealed_direction_lock_from_p_record,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identities(source) -> dict[int, str]:
    return {
        int(item["candidate_physical_row"]): f"IDENTITY-{item['candidate_physical_row']}"
        for item in source["queries"][0]["candidates"]
    }


def _multigrid_source():
    source = synthetic_source(query_count=1, candidate_count=128)
    shapes = ((6, 10), (7, 9), (8, 8), (9, 7))
    generator = torch.Generator().manual_seed(9917)
    for position, candidate in enumerate(source["queries"][0]["candidates"]):
        shape = shapes[position % len(shapes)]
        if shape == (8, 8):
            continue
        old_shape = (8, 8)
        candidate["reference_tokens"] = torch.randn(
            (shape[0] * shape[1], 12), generator=generator, dtype=torch.float32
        )
        candidate["reference_valid_patch_mask"] = torch.ones(
            shape[0] * shape[1], dtype=torch.bool
        )
        candidate["colnomic_reference_grid_shape"] = list(shape)
        candidate["dino_reference_grid_shape"] = list(shape)
        valid = torch.ones(shape[0] * shape[1], dtype=torch.bool)
        for roots in candidate["directions"].values():
            for root in roots:
                for action in root["actions"]:
                    action["colnomic_reference_action_mask"] = transport_reference_mask(
                        action["colnomic_reference_action_mask"],
                        old_shape,
                        shape,
                        valid,
                    )
                    action["deployment_reference_mask"] = transport_reference_mask(
                        action["deployment_reference_mask"],
                        old_shape,
                        shape,
                        valid,
                    )
    return source


def _real_bundle(source, model, *, candidate_count: int) -> VQueryBundle:
    ledger = materialize_source_payload(
        source,
        source_file_sha256=_sha("real-source"),
        source_role="I1_SYNTHETIC_ONLY",
    )
    index = feature_ledger_index(ledger)
    raw_query = source["queries"][0]
    query_grid = tuple(raw_query["dino_query_grid_shape"])
    generator = torch.Generator().manual_seed(901)
    query_layers = torch.randn(
        (1, query_grid[0] * query_grid[1], 3),
        generator=generator,
        dtype=torch.float64,
    )
    query = QueryTokenFieldV1(
        layers=query_layers,
        grid_shape=query_grid,
        valid_patch_mask=torch.ones(query_grid, dtype=torch.bool),
        source_image_sha256=raw_query["query_source_image_sha256"],
        source_key="synthetic-query-dino",
        cache_payload_sha256=_sha("query-cache"),
        geometry_record_sha256=raw_query["query_geometry_sha256"],
        tokens_sha256=token_tensor_sha256(query_layers),
    )
    by_position = {}
    for record in index.values():
        by_position.setdefault(record.candidate_position, {})[record.direction] = record
    locks = {}
    for raw_candidate in raw_query["candidates"]:
        position = raw_candidate["candidate_position"]
        reference_grid = tuple(raw_candidate["dino_reference_grid_shape"])
        reference_layers = torch.randn(
            (1, reference_grid[0] * reference_grid[1], 3),
            generator=generator,
            dtype=torch.float64,
        )
        candidate = CandidateReferenceFieldV1(
            candidate_key=raw_candidate["candidate_key"],
            layers=reference_layers,
            grid_shape=reference_grid,
            valid_patch_mask=torch.ones(reference_grid, dtype=torch.bool),
            physical_gallery_row=raw_candidate["candidate_physical_row"],
            source_image_sha256=raw_candidate[
                "candidate_reference_source_sha256"
            ],
            source_key=f"synthetic-reference-{position}",
            cache_payload_sha256=_sha(f"reference-cache-{position}"),
            geometry_record_sha256=raw_candidate["reference_geometry_sha256"],
            tokens_sha256=token_tensor_sha256(reference_layers),
        )
        directions = {}
        for direction in P_DIRECTIONS:
            raw_lock = build_p_lock_record(
                score_deployable_direction(model, by_position[position][direction]),
                crossfit_role="OUTER_HELDOUT_DEPLOYMENT",
                outer_fold=1,
                p_checkpoint_sha256=_sha("p-checkpoint"),
                p_training_manifest_sha256=_sha("p-training-manifest"),
            )
            directions[direction] = sealed_direction_lock_from_p_record(
                raw_lock, query=query, candidate=candidate
            )
        locks[candidate.candidate_key] = CandidatePLockV1(
            candidate=candidate, direction_locks=directions
        )
    assert len(locks) == candidate_count
    return VQueryBundle(
        query_id=raw_query["query_id"],
        execution_ordinal=raw_query["execution_ordinal"],
        outer_fold=1,
        query=query,
        locks=locks,
        candidate_axis_sha256=canonical_sha256(
            [item["candidate_physical_row"] for item in raw_query["candidates"]]
        ),
    )


def test_canonical_c128_donor_map_is_exact_fixed_point_free_cycle():
    source = synthetic_source(query_count=1, candidate_count=128)
    order = canonical_donor_positions(
        source["queries"][0], _identities(source), expected_count=128
    )
    assert order == tuple(range(1, 128)) + (0,)
    assert sorted(order) == list(range(128))
    assert all(position != donor for position, donor in enumerate(order))


def test_cyclic_offset_skips_same_corrected_identity_without_result_feedback():
    source = synthetic_source(query_count=1, candidate_count=4)
    rows = [
        int(item["candidate_physical_row"])
        for item in source["queries"][0]["candidates"]
    ]
    identities = dict(zip(rows, ("A", "A", "B", "B"), strict=True))
    assert canonical_donor_positions(
        source["queries"][0], identities, expected_count=4
    ) == (2, 3, 0, 1)


def test_source_receipt_binds_complete_c128_multiset_namespace_and_geometry():
    source = synthetic_source(query_count=1, candidate_count=128)
    receipt = donor_receipt(
        source["queries"][0], _identities(source), expected_count=128
    )
    assert receipt["namespace"] == NAMESPACE
    assert receipt["fixed_point_free"] is True
    assert receipt["complete_permutation"] is True
    assert receipt["content_multiset_preserved"] is True
    assert receipt["candidate_count"] == 128
    assert len(receipt["bindings"]) == 128
    assert receipt["donor_positions"] == list(range(1, 128)) + [0]
    assert receipt["response_dependent_retry_count"] == 0
    assert receipt["fallback_donor_count"] == 0


def test_complete_pre_p_replay_keeps_dino_axis_content_and_reruns_every_lock(
    monkeypatch,
):
    source = _multigrid_source()
    identities = _identities(source)
    model, _ = initialize_training()
    real = _real_bundle(source, model, candidate_count=128)
    result = replay_c_col_p(
        raw_source=source,
        query_id="SYNTH-0000",
        real_bundle=real,
        model=model,
        outer_fold=1,
        p_checkpoint_sha256=_sha("p-checkpoint"),
        p_training_manifest_sha256=_sha("p-training-manifest"),
        corrected_identity_by_physical_row=identities,
        expected_count=128,
    )
    assert result.receipt["fixed_point_free"] is True
    assert result.receipt["content_multiset_preserved"] is True
    assert result.receipt["corrected_identity_disjoint"] is True
    assert result.receipt["native_colnomic_token_resize_count"] == 0
    assert result.receipt["cross_grid_binding_count"] == 128
    assert result.receipt["complete_p_feature_head_resolver_lock_rerun"] is True
    assert result.receipt["rerun_p_lock_count"] == 256
    assert len(result.lock_records) == 256
    assert dino_axis_content_fingerprint(result.bundle) == dino_axis_content_fingerprint(real)
    assert bundle_fingerprint(result.bundle) == result.receipt[
        "control_bundle_fingerprint"
    ]
    for key in real.locks:
        assert result.bundle.locks[key].candidate is real.locks[key].candidate
    features = build_control_feature_records(
        source["queries"][0],
        corrected_identity_by_physical_row=identities,
        destination_dino_valid_by_candidate_key={
            key: real.locks[key].candidate.valid_patch_mask for key in real.locks
        },
        expected_count=128,
    )
    assert len(features) == 256
    assert all(record.candidate_key in real.locks for record in features)

    # The formal validator is a source-level second implementation.  Breaking
    # the production replay symbol must not affect its reconstruction.
    import rc_aslo_xf.dino_rcde_sr0_mt_c_col_p_v1 as production

    monkeypatch.setattr(
        production,
        "replay_c_col_p",
        lambda **_: (_ for _ in ()).throw(AssertionError("production replay used")),
    )
    rebuilt_bundle, rebuilt_receipt, rebuilt_records = independent_replay_c_col_p(
        raw_source=source,
        query_id="SYNTH-0000",
        real_bundle=real,
        model=model,
        outer_fold=1,
        p_checkpoint_sha256=_sha("p-checkpoint"),
        p_training_manifest_sha256=_sha("p-training-manifest"),
        corrected_identity_by_physical_row=identities,
        expected_count=128,
    )
    assert len(rebuilt_records) == 256
    assert dict(result.receipt) == rebuilt_receipt
    assert bundle_fingerprint(result.bundle) == independent_bundle_fingerprint(
        rebuilt_bundle
    )
    assert_observed_matches_independent(
        observed_bundle=result.bundle,
        observed_receipt=result.receipt,
        rebuilt_bundle=rebuilt_bundle,
        rebuilt_receipt=rebuilt_receipt,
    )

    tampered = dict(result.receipt)
    tampered["donor_positions"] = list(tampered["donor_positions"])
    tampered["donor_positions"][0] = 0
    with pytest.raises(IndependentCColPError, match="receipt differs"):
        assert_observed_matches_independent(
            observed_bundle=result.bundle,
            observed_receipt=tampered,
            rebuilt_bundle=rebuilt_bundle,
            rebuilt_receipt=rebuilt_receipt,
        )


def test_c_col_fails_closed_when_no_identity_disjoint_cyclic_matching_exists():
    source = synthetic_source(query_count=1, candidate_count=2)
    identities = {
        int(item["candidate_physical_row"]): "SAME-IDENTITY"
        for item in source["queries"][0]["candidates"]
    }
    with pytest.raises(Exception, match="identity-disjoint cyclic"):
        donor_receipt(source["queries"][0], identities, expected_count=2)


def test_positive_overlap_transport_handles_unequal_native_grids_without_resize():
    source = torch.tensor([True, False, False, False], dtype=torch.bool)
    result = transport_reference_mask(
        source,
        (2, 2),
        (3, 5),
        torch.ones(15, dtype=torch.bool),
    ).reshape(3, 5)
    # The top-left half-open source cell overlaps exactly rows 0..1, cols 0..2.
    expected = torch.zeros((3, 5), dtype=torch.bool)
    expected[:2, :3] = True
    assert torch.equal(result, expected)
