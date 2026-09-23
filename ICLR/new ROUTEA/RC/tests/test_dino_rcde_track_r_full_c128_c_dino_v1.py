from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

from rc_aslo_xf.dino_rcde_cw1_multitile_vdecode_v1 import (  # noqa: E402
    CandidateReferenceFieldV1,
    FIXED_DIRECTIONS,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (  # noqa: E402
    GeometryNamespaceBindingV1,
    outer_refit_head_spec,
    project_natural_v2_candidate_to_v1_runtime,
    select_natural_v2_head_records,
)
from rc_aslo_xf.dino_rcde_track_r_full_c128_c_dino_v1 import (  # noqa: E402
    FullC128CDinoContractError,
    FullC128CDinoPairPlanV1,
    FullC128CandidateAxisRecordV1,
    FullC128ControlAxisV1,
    corrected_identity_sha256,
    dino_content_binding_sha256_from_cache_index_record,
    full_c128_axis_from_selected_head_records,
    materialize_full_c128_c_dino_pair,
    move_full_c128_c_dino_pair_to_device,
    plan_full_c128_c_dino_pair,
)
from rc_aslo_xf.gallery_identity_repair import build_identity_map  # noqa: E402
from replay_dino_rcde_track_r_full_c128_c_dino_receipt_v1 import (  # noqa: E402
    IndependentReplayError,
    OUTPUT_STATUS,
    replay_receipt,
)

import dino_rcde_sr0_mt_v_input_common_v1 as v_input  # noqa: E402
import dino_rcde_track_r_input_common_v1 as track_input  # noqa: E402
from test_dino_rcde_sr0_mt_three_arm_v1 import _locks as synthetic_locks  # noqa: E402


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def synthetic_axis() -> tuple[FullC128ControlAxisV1, dict[str, object]]:
    _, pair_locks = synthetic_locks()
    by_row = {
        lock.candidate.physical_gallery_row: (key, lock)
        for key, lock in pair_locks.items()
    }
    records = []
    metadata: dict[str, object] = {"pair_locks": pair_locks}
    for position in range(128):
        if position in by_row:
            key, lock = by_row[position]
            source = lock.candidate.source_image_sha256
            p_locks = dict(lock.p_lock_record_sha256_by_direction)
            native = {
                direction: lock.candidate.tokens_sha256
                for direction in FIXED_DIRECTIONS
            }
        else:
            key = sha(f"synthetic-candidate-key-{position}")
            source = sha(f"synthetic-source-{position}")
            p_locks = {
                direction: sha(f"synthetic-p-{position}-{direction}")
                for direction in FIXED_DIRECTIONS
            }
            native = {
                direction: sha(f"synthetic-native-{position}-{direction}")
                for direction in FIXED_DIRECTIONS
            }
        records.append(
            FullC128CandidateAxisRecordV1(
                candidate_position=position,
                candidate_key=key,
                candidate_physical_row=position,
                candidate_reference_source_sha256=source,
                corrected_identity_sha256=corrected_identity_sha256(
                    f"synthetic-identity-{position}"
                ),
                p_lock_record_sha256_by_direction=p_locks,
                candidate_native_content_sha256_by_direction=native,
                dino_content_binding_sha256=sha(
                    f"synthetic-dino-content-binding-{position}"
                ),
            )
        )
    axis = FullC128ControlAxisV1(
        query_id="synthetic-target-free-q0",
        candidate_axis_sha256=canonical_sha256(list(range(128))),
        records=tuple(records),
    )
    return axis, metadata


def candidate_for_axis_record(
    record: FullC128CandidateAxisRecordV1,
    template: CandidateReferenceFieldV1,
) -> CandidateReferenceFieldV1:
    return CandidateReferenceFieldV1(
        candidate_key=record.candidate_key,
        physical_gallery_row=record.candidate_physical_row,
        layers=template.layers,
        grid_shape=template.grid_shape,
        valid_patch_mask=template.valid_patch_mask,
        source_image_sha256=record.candidate_reference_source_sha256,
        source_key=f"synthetic-reference:{record.candidate_physical_row}",
        cache_payload_sha256=sha(
            f"synthetic-cache-{record.candidate_physical_row}"
        ),
        geometry_record_sha256=sha(
            f"synthetic-geometry-{record.candidate_physical_row}"
        ),
        tokens_sha256=template.tokens_sha256,
    )


def test_synthetic_full_c128_plan_materializes_and_moves_only_pair_donors() -> None:
    axis, metadata = synthetic_axis()
    pair_locks = metadata["pair_locks"]
    assert isinstance(pair_locks, dict)
    left, right = tuple(pair_locks)
    plan = plan_full_c128_c_dino_pair(
        axis,
        left,
        right,
        namespace="TRACK_R_FULL_C128_SYNTHETIC_SEED17",
    )
    assert len(plan.destination_to_source) == 128
    assert sorted(plan.destination_to_source) == list(range(128))
    assert all(
        index != source
        for index, source in enumerate(plan.destination_to_source)
    )
    assert set(plan.donor_candidate_keys).isdisjoint({left, right})

    by_key = {item.candidate_key: item for item in axis.records}
    template = pair_locks[left].candidate
    donors = {
        key: candidate_for_axis_record(by_key[key], template)
        for key in plan.donor_candidate_keys
    }
    controlled, receipt = materialize_full_c128_c_dino_pair(
        plan,
        destination_pair_locks=pair_locks,
        donor_candidates=donors,
        donor_dino_content_binding_sha256_by_candidate_key={
            key: by_key[key].dino_content_binding_sha256
            for key in plan.donor_candidate_keys
        },
    )
    assert set(controlled) == {left, right}
    assert receipt["candidate_count"] == 128
    assert receipt["materialized_control_candidate_count"] == 2
    assert receipt["maximum_device_transfer_candidate_count"] == 2
    assert receipt["pair_member_donor_exclusion"] is True
    for key in (left, right):
        assert controlled[key].destination_p_lock is pair_locks[key]
        assert (
            controlled[key].p_lock_record_sha256_by_direction
            == pair_locks[key].p_lock_record_sha256_by_direction
        )
    moved = move_full_c128_c_dino_pair_to_device(
        controlled,
        "cpu",
        expected_pair_member_keys=(left, right),
    )
    assert len(moved) == 2
    assert all(binding.candidate.layers.device.type == "cpu" for binding in moved.values())
    with pytest.raises(FullC128CDinoContractError, match="exactly two"):
        move_full_c128_c_dino_pair_to_device(
            {f"accidental-full-axis-{index}": controlled[left] for index in range(128)},
            "cpu",
        )

    replay = replay_receipt(dict(receipt))
    assert replay["status"] == OUTPUT_STATUS
    assert replay["validated_candidate_count"] == 128
    assert replay["validated_pair_member_count"] == 2
    assert replay["target_or_rival_read_count"] == 0


def test_planner_rejects_cross_pair_donor_even_when_mapping_is_a_derangement() -> None:
    axis, metadata = synthetic_axis()
    pair_locks = metadata["pair_locks"]
    assert isinstance(pair_locks, dict)
    left, right = tuple(pair_locks)
    canonical = axis.canonical_matching_axis()
    index_by_key = {
        item.candidate_key: index for index, item in enumerate(canonical)
    }
    left_index = index_by_key[left]
    right_index = index_by_key[right]
    # A cyclic shift is a valid identity-disjoint C128 derangement, but choose
    # its direction so that the left destination receives the right member.
    shift = (right_index - left_index) % 128
    assert shift not in {0}
    order = tuple((index + shift) % 128 for index in range(128))
    assert order[left_index] == right_index
    with pytest.raises(
        FullC128CDinoContractError,
        match="inside the scored pair",
    ):
        FullC128CDinoPairPlanV1(
            axis=axis,
            namespace="TRACK_R_CROSS_PAIR_FORBIDDEN",
            pair_member_keys=(left, right),
            destination_to_source=order,
        )


def test_independent_replay_fails_closed_on_p_lock_tamper() -> None:
    axis, metadata = synthetic_axis()
    pair_locks = metadata["pair_locks"]
    assert isinstance(pair_locks, dict)
    left, right = tuple(pair_locks)
    plan = plan_full_c128_c_dino_pair(
        axis,
        left,
        right,
        namespace="TRACK_R_FULL_C128_TAMPER_SEED17",
    )
    by_key = {item.candidate_key: item for item in axis.records}
    donors = {
        key: candidate_for_axis_record(by_key[key], pair_locks[left].candidate)
        for key in plan.donor_candidate_keys
    }
    _, receipt = materialize_full_c128_c_dino_pair(
        plan,
        destination_pair_locks=pair_locks,
        donor_candidates=donors,
        donor_dino_content_binding_sha256_by_candidate_key={
            key: by_key[key].dino_content_binding_sha256
            for key in plan.donor_candidate_keys
        },
    )
    poisoned = copy.deepcopy(dict(receipt))
    poisoned["pair_bindings"][0][
        "destination_p_lock_output_sha256_by_direction"
    ]["a_to_b"] = sha("tampered-P-lock")
    poisoned["logical_sha256"] = canonical_sha256(
        {key: item for key, item in poisoned.items() if key != "logical_sha256"}
    )
    with pytest.raises(IndependentReplayError, match="P lock changed"):
        replay_receipt(poisoned)


def test_materializer_rejects_dino_content_binding_drift() -> None:
    axis, metadata = synthetic_axis()
    pair_locks = metadata["pair_locks"]
    assert isinstance(pair_locks, dict)
    left, right = tuple(pair_locks)
    plan = plan_full_c128_c_dino_pair(
        axis,
        left,
        right,
        namespace="TRACK_R_FULL_C128_DINO_BINDING_POISON_SEED17",
    )
    by_key = {item.candidate_key: item for item in axis.records}
    donors = {
        key: candidate_for_axis_record(by_key[key], pair_locks[left].candidate)
        for key in plan.donor_candidate_keys
    }
    bindings = {
        key: by_key[key].dino_content_binding_sha256
        for key in plan.donor_candidate_keys
    }
    bindings[plan.donor_candidate_keys[0]] = sha("wrong-dino-content-binding")
    with pytest.raises(
        FullC128CDinoContractError,
        match="DINO content/full-C128 binding drift",
    ):
        materialize_full_c128_c_dino_pair(
            plan,
            destination_pair_locks=pair_locks,
            donor_candidates=donors,
            donor_dino_content_binding_sha256_by_candidate_key=bindings,
        )


def actual_paths() -> tuple[Path, ...]:
    return (
        ROOT / "registry/current_authority_v121_20260821.json",
        ROOT / "results/dino_rcde_sr0_mt_p_v2_formal_lock_aggregate_v1/result.json",
        ROOT
        / "results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json",
    )


def test_actual_one_query_target_free_e0_closes_full_c128_and_pair_only_runtime() -> None:
    required = actual_paths()
    if not all(path.is_file() for path in required):
        pytest.skip("workspace-only target-free E0 inputs are absent")
    authority = json.loads(required[0].read_text(encoding="utf-8"))
    aggregate = json.loads(required[1].read_text(encoding="utf-8"))
    role_free = json.loads(required[2].read_text(encoding="utf-8"))
    execution = 0
    aggregate_row = next(
        item for item in aggregate["rows"] if item["execution_ordinal"] == execution
    )
    pair = next(
        item
        for item in role_free["pair_manifest"]["records"]
        if item["execution_ordinal"] == execution
    )
    artifact_path = ROOT / aggregate_row["lock_artifact_path"]
    if not artifact_path.is_file():
        pytest.skip("workspace-only full-C128 lock artifact is absent")
    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    assert artifact["target_free"] is True
    assert artifact["candidate_count"] == 128
    assert artifact["record_count"] == 1024
    spec = outer_refit_head_spec(source_fold=int(aggregate_row["source_fold"]))
    selected = select_natural_v2_head_records(
        artifact["records"],
        spec=spec,
        expected_candidate_count=128,
    )

    gallery_path = Path(authority["bindings"]["gallery_cache"]["path"])
    if not gallery_path.is_file():
        pytest.skip("workspace-only gallery identity input is absent")
    gallery = torch.load(
        gallery_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    identities = build_identity_map(tuple(map(str, gallery["setids"]))).labels
    cache_index, _, _, _, _ = v_input.load_schedule_cache_index(
        ROOT / authority["bindings"]["redacted_schedule"]["path"],
        ROOT / authority["bindings"]["redacted_cache_index"]["path"],
    )
    selected_rows = {
        int(item["candidate"]["candidate_physical_row"]) for item in selected
    }
    dino_content_by_row = {
        row: dino_content_binding_sha256_from_cache_index_record(
            cache_index[("reference", row)]
        )
        for row in selected_rows
    }
    axis = full_c128_axis_from_selected_head_records(
        selected,
        corrected_identity_by_physical_row={
            int(item["candidate"]["candidate_physical_row"]): identities[
                int(item["candidate"]["candidate_physical_row"])
            ]
            for item in selected
        },
        dino_content_binding_sha256_by_physical_row=dino_content_by_row,
        expected_candidate_axis_sha256=pair["candidate_axis_sha256"],
    )
    members = pair["members"]
    left, right = (str(item["candidate_key"]) for item in members)
    plan = plan_full_c128_c_dino_pair(
        axis,
        left,
        right,
        namespace="TRACK_R_OOF_C_DINO_FULL_C128_E0_V1_SEED17",
    )
    assert set(plan.donor_candidate_keys).isdisjoint({left, right})

    geometry_queries, geometry_references, _ = track_input.load_geometry_payload(
        ROOT / authority["bindings"]["geometry_payload"]["path"]
    )
    cache_root = ROOT / "runtime/dino_rcde_p0_v1_2/stable_fp16_cache"
    expected_model_sha = str(authority["cache_model_checkpoint_logical_sha256"])
    query = v_input.make_query_field(
        {
            "execution_ordinal": execution,
            "query_source_image_sha256": pair["query_source_image_sha256"],
        },
        cache_root=cache_root,
        index=cache_index,
        expected_model_sha256=expected_model_sha,
    )
    by_key: dict[str, dict[str, object]] = {}
    for record in selected:
        by_key.setdefault(str(record["candidate"]["candidate_key"]), {})[
            str(record["direction"])
        ] = record
    axis_by_key = {item.candidate_key: item for item in axis.records}
    needed = set((left, right, *plan.donor_candidate_keys))
    candidate_fields = {
        key: v_input.make_candidate_field(
            axis_by_key[key].candidate_physical_row,
            axis_by_key[key].candidate_reference_source_sha256,
            cache_root=cache_root,
            index=cache_index,
            expected_model_sha256=expected_model_sha,
        )
        for key in needed
    }
    q_geometry = geometry_queries[execution]
    destination_locks = {}
    for key in (left, right):
        candidate = candidate_fields[key]
        r_geometry = geometry_references[candidate.physical_gallery_row]
        destination_locks[key] = project_natural_v2_candidate_to_v1_runtime(
            query=query,
            candidate=candidate,
            direction_records=by_key[key],
            spec=spec,
            query_geometry_binding=GeometryNamespaceBindingV1(
                source_image_sha256=query.source_image_sha256,
                grid_shape=query.grid_shape,
                canonical_geometry_sha256=q_geometry["dino_geometry"][
                    "geometry_sha256"
                ],
                cache_geometry_record_sha256=query.geometry_record_sha256,
            ),
            reference_geometry_binding=GeometryNamespaceBindingV1(
                source_image_sha256=candidate.source_image_sha256,
                grid_shape=candidate.grid_shape,
                canonical_geometry_sha256=r_geometry["dino_geometry"][
                    "geometry_sha256"
                ],
                cache_geometry_record_sha256=candidate.geometry_record_sha256,
            ),
        ).runtime_lock
    controlled, receipt = materialize_full_c128_c_dino_pair(
        plan,
        destination_pair_locks=destination_locks,
        donor_candidates={
            key: candidate_fields[key] for key in plan.donor_candidate_keys
        },
        donor_dino_content_binding_sha256_by_candidate_key={
            key: axis_by_key[key].dino_content_binding_sha256
            for key in plan.donor_candidate_keys
        },
    )
    replay = replay_receipt(dict(receipt))
    moved = move_full_c128_c_dino_pair_to_device(
        controlled,
        "cpu",
        expected_pair_member_keys=(left, right),
    )
    assert replay["status"] == OUTPUT_STATUS
    assert replay["validated_candidate_count"] == 128
    assert len(controlled) == len(moved) == 2
    assert receipt["label_target_rival_read_count"] == 0
    assert receipt["scientific_reduction_count"] == 0
    assert receipt["model_forward_count"] == 0
    assert all(
        controlled[key].destination_p_lock is destination_locks[key]
        for key in (left, right)
    )


def test_independent_replay_source_does_not_import_production_control() -> None:
    source = (
        ROOT
        / "programs/replay_dino_rcde_track_r_full_c128_c_dino_receipt_v1.py"
    ).read_text(encoding="utf-8")
    assert "dino_rcde_track_r_full_c128_c_dino_v1" not in source
    assert "dino_rcde_sr0_mt_controls_v1" not in source
