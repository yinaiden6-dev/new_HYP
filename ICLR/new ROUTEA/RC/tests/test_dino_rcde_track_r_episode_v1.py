from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch

from rc_aslo_xf.cw1_sr0_structure_v1 import enumerate_superregion_bank
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_adapter_v2 import build_natural_p_lock_v2
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_source_v1 import candidate_key_v1
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_three_arm_adapter_v1 import (
    inner_oof_head_spec,
    outer_refit_head_spec,
)
from rc_aslo_xf.dino_rcde_track_r_episode_v1 import (
    TrackREpisodeError,
    ValidatedHeadRecordsV1,
    build_compact_episode,
    expected_head_specs,
    index_complete_artifact_records,
    validate_compact_episode,
)


GRID = (6, 6)
QUERY_SOURCE = canonical_sha256({"track-r-e1-test": "query-source"})
QUERY_GEOMETRY = canonical_sha256({"track-r-e1-test": "query-geometry"})
ROOT_MASKS = (
    (0, 1, 6, 7),
    (2, 3, 8, 9),
    (12, 13, 18, 19),
    (14, 15, 20, 21),
)


def _sha(value: object) -> str:
    return canonical_sha256({"track-r-e1-test": value})


def _rle(indices: tuple[int, ...]) -> dict[str, object]:
    runs: list[list[int]] = []
    for index in indices:
        if runs and runs[-1][0] + runs[-1][1] == index:
            runs[-1][1] += 1
        else:
            runs.append([index, 1])
    return {"grid_shape": list(GRID), "rle": runs}


def _record(*, spec, position: int, row: int, direction: str) -> dict[str, object]:
    reference_source = _sha(f"reference-source-{row}")
    reference_geometry = _sha(f"reference-geometry-{row}")
    candidate_key = candidate_key_v1(
        physical_row=row, source_image_sha256=reference_source
    )
    source = {
        "query_id": "TRACK-R-E1-FIXTURE",
        "execution_ordinal": 7,
        "query_source_image_sha256": QUERY_SOURCE,
        "candidate_physical_row": row,
        "candidate_reference_source_sha256": reference_source,
        "query_geometry_sha256": QUERY_GEOMETRY,
        "reference_geometry_sha256": reference_geometry,
        "candidate_position": position,
        "candidate_key": candidate_key,
        "cache_sha256": _sha(f"directional-population-{row}-{direction}"),
        "direction": direction,
        "colnomic_query_grid_shape": list(GRID),
        "deployment_query_grid_shape": list(GRID),
        "deployment_reference_grid_shape": list(GRID),
        "root_count": 4,
        "action_count": 1,
        "deployment_query_root_masks": [_rle(mask) for mask in ROOT_MASKS],
        "deployment_reference_mask_table": [_rle(mask) for mask in ROOT_MASKS],
        "deployment_reference_mask_index": [[index] for index in range(4)],
        "action_keys": [[_sha(f"action-{row}-{index}")] for index in range(4)],
        "eligibility_rle": {
            "shape": [4, 1],
            "flatten_order": "ROW_MAJOR",
            "true_runs": [[0, 4]],
        },
    }
    valid = torch.ones(math.prod(GRID), dtype=torch.bool)
    query_geometry = {
        "query_id": source["query_id"],
        "execution_ordinal": 7,
        "source_image_sha256": QUERY_SOURCE,
        "dino_geometry": {
            "geometry_sha256": QUERY_GEOMETRY,
            "grid_shape": list(GRID),
            "valid_patch_mask": valid,
            "valid_patch_mask_sha256": _sha("query-valid"),
        },
    }
    reference_geometry_record = {
        "physical_row": row,
        "source_image_sha256": reference_source,
        "dino_geometry": {
            "geometry_sha256": reference_geometry,
            "grid_shape": list(GRID),
            "valid_patch_mask": valid,
            "valid_patch_mask_sha256": _sha(f"reference-valid-{row}"),
        },
    }
    bank = enumerate_superregion_bank(GRID, include_r0_control=False)
    return build_natural_p_lock_v2(
        source_record=source,
        query_geometry_record=query_geometry,
        reference_geometry_record=reference_geometry_record,
        fold_record={
            "query_id": source["query_id"],
            "query_ordinal": 70,
            "source_image_sha256": QUERY_SOURCE,
            "inner_fold": 1,
            "track": "fixture",
        },
        membership={
            "fit_id": spec.fit_id,
            "outer_fold": spec.outer_fold,
            "inner_heldout_fold": spec.inner_heldout_fold,
            "heldout_addresses": [
                {"execution_ordinal": 7, "query_id": source["query_id"]}
            ],
        },
        row_scores=-torch.arange(len(bank), dtype=torch.float64),
        selected_action_indices=(0, 0, 0, 0),
        fit_id=spec.fit_id,
        crossfit_role=spec.crossfit_role,
        p_checkpoint_sha256=_sha(f"checkpoint-{spec.fit_id}"),
        p_training_manifest_sha256=_sha(f"manifest-{spec.fit_id}"),
    )


def _artifact(candidate_count: int = 2):
    specs = expected_head_specs(1)
    rows = tuple(range(9, 9 + candidate_count))
    records = [
        _record(spec=spec, position=position, row=row, direction=direction)
        for spec in specs
        for position, row in enumerate(rows)
        for direction in ("a_to_b", "b_to_a")
    ]
    return specs, rows, records


def test_four_head_index_and_compact_episode(monkeypatch):
    specs, rows, records = _artifact()
    monkeypatch.setattr(
        "rc_aslo_xf.dino_rcde_track_r_episode_v1.EXPECTED_CANDIDATE_COUNT", 2
    )
    monkeypatch.setattr(
        "rc_aslo_xf.dino_rcde_track_r_episode_v1.EXPECTED_RECORD_COUNT", 16
    )
    indexed = index_complete_artifact_records(
        records, source_fold=1, expected_candidate_count=2
    )
    assert tuple(indexed) == tuple(spec.fit_id for spec in specs)
    spec = inner_oof_head_spec(outer_fold=2, source_fold=1)
    target_key = indexed[spec.fit_id][0]["candidate"]["candidate_key"]
    rival_key = indexed[spec.fit_id][2]["candidate"]["candidate_key"]
    episode = build_compact_episode(
        selected_head_records=indexed[spec.fit_id],
        source_fold=1,
        outer_fold=2,
        query_id="TRACK-R-E1-FIXTURE",
        execution_ordinal=7,
        query_source_image_sha256=QUERY_SOURCE,
        track="fixture",
        target_candidate_key=target_key,
        strongest_rival_candidate_key=rival_key,
        loss_role_record_sha256=_sha("loss-role"),
        loss_join_sha256=_sha("loss-join"),
        lock_artifact_path="results/fixture/locks.pt",
        lock_artifact_sha256=_sha("lock-artifact"),
    )
    validate_compact_episode(episode)
    assert episode["query"]["fit_id"] == "P_OUTER2_INNER1_FIT"
    assert episode["target"]["candidate_physical_row"] == rows[0]
    assert episode["strongest_rival"]["candidate_physical_row"] == rows[1]
    assert set(episode["target"]["direction_records"]) == {"a_to_b", "b_to_a"}
    assert "identity_sha256" not in episode
    assert "supergroup_sha256" not in episode
    assert "score" not in episode


def test_candidate_axis_drift_and_wrong_role_fail(monkeypatch):
    _, _, records = _artifact()
    monkeypatch.setattr(
        "rc_aslo_xf.dino_rcde_track_r_episode_v1.EXPECTED_CANDIDATE_COUNT", 2
    )
    broken = copy.deepcopy(records)
    broken[-1]["candidate"]["candidate_physical_row"] = 99
    broken[-1]["record_sha256"] = canonical_sha256(
        {key: value for key, value in broken[-1].items() if key != "record_sha256"}
    )
    with pytest.raises(TrackREpisodeError):
        index_complete_artifact_records(
            broken, source_fold=1, expected_candidate_count=2
        )


def test_validated_head_fast_path_is_byte_equal_and_plain_input_revalidates(monkeypatch):
    _, _, records = _artifact()
    monkeypatch.setattr(
        "rc_aslo_xf.dino_rcde_track_r_episode_v1.EXPECTED_CANDIDATE_COUNT", 2
    )
    monkeypatch.setattr(
        "rc_aslo_xf.dino_rcde_track_r_episode_v1.EXPECTED_RECORD_COUNT", 16
    )
    indexed = index_complete_artifact_records(
        records, source_fold=1, expected_candidate_count=2
    )
    spec = inner_oof_head_spec(outer_fold=2, source_fold=1)
    validated = indexed[spec.fit_id]
    assert isinstance(validated, ValidatedHeadRecordsV1)
    target_key = validated[0]["candidate"]["candidate_key"]
    rival_key = validated[2]["candidate"]["candidate_key"]
    kwargs = dict(
        source_fold=1,
        outer_fold=2,
        query_id="TRACK-R-E1-FIXTURE",
        execution_ordinal=7,
        query_source_image_sha256=QUERY_SOURCE,
        track="fixture",
        target_candidate_key=target_key,
        strongest_rival_candidate_key=rival_key,
        loss_role_record_sha256=_sha("fast-loss-role"),
        loss_join_sha256=_sha("fast-loss-join"),
        lock_artifact_path="results/fixture/locks.pt",
        lock_artifact_sha256=_sha("fast-lock-artifact"),
    )
    slow = build_compact_episode(
        selected_head_records=list(validated), **kwargs
    )

    def forbidden_revalidation(*_args, **_kwargs):
        raise AssertionError("validated head was revalidated")

    monkeypatch.setattr(
        "rc_aslo_xf.dino_rcde_track_r_episode_v1.select_natural_v2_head_records",
        forbidden_revalidation,
    )
    fast = build_compact_episode(selected_head_records=validated, **kwargs)
    assert fast == slow
    with pytest.raises(TrackREpisodeError, match="episode-head validation failed"):
        build_compact_episode(selected_head_records=list(validated), **kwargs)


def test_descriptor_poison_fails(monkeypatch):
    specs, _, records = _artifact()
    monkeypatch.setattr(
        "rc_aslo_xf.dino_rcde_track_r_episode_v1.EXPECTED_CANDIDATE_COUNT", 2
    )
    indexed = index_complete_artifact_records(
        records, source_fold=1, expected_candidate_count=2
    )
    spec = inner_oof_head_spec(outer_fold=3, source_fold=1)
    target_key = indexed[spec.fit_id][0]["candidate"]["candidate_key"]
    rival_key = indexed[spec.fit_id][2]["candidate"]["candidate_key"]
    episode = build_compact_episode(
        selected_head_records=indexed[spec.fit_id],
        source_fold=1,
        outer_fold=3,
        query_id="TRACK-R-E1-FIXTURE",
        execution_ordinal=7,
        query_source_image_sha256=QUERY_SOURCE,
        track="fixture",
        target_candidate_key=target_key,
        strongest_rival_candidate_key=rival_key,
        loss_role_record_sha256=_sha("loss-role"),
        loss_join_sha256=_sha("loss-join"),
        lock_artifact_path="results/fixture/locks.pt",
        lock_artifact_sha256=_sha("lock-artifact"),
    )
    poisoned = copy.deepcopy(episode)
    poisoned["raw_score"] = 1.0
    with pytest.raises(TrackREpisodeError, match="field-set"):
        validate_compact_episode(poisoned)
    poisoned = copy.deepcopy(episode)
    poisoned["target"]["candidate_key"] = poisoned["strongest_rival"]["candidate_key"]
    poisoned["record_sha256"] = canonical_sha256(
        {key: value for key, value in poisoned.items() if key != "record_sha256"}
    )
    with pytest.raises(TrackREpisodeError, match="collision"):
        validate_compact_episode(poisoned)


def test_shard_programs_are_compact_and_fail_closed():
    root = Path(__file__).resolve().parents[1]
    producer = (root / "programs/materialize_dino_rcde_track_r_relative_e1_episode_shard_v1.py").read_text()
    validator = (root / "programs/validate_dino_rcde_track_r_relative_e1_episode_shard_v1.py").read_text()
    launcher = (root / "slurm/dino_rcde_track_r_relative_e1_episode_shards_v1.sbatch").read_text()
    assert "weights_only=True" in producer
    assert "mmap=True" in producer
    assert "weights_only=True" in validator
    assert "mmap=True" in validator
    assert "score_rank_winner_gap_outcome_read_count\": 0" in producer
    assert "dino_token_load_count\": 0" in producer
    assert "model_load_count\": 0" in producer
    assert "build_compact_episode" in validator
    assert "independent compact episode replay drift" in validator
    assert "fold_by_execution = dict(enumerate(fold_records))" in producer
    assert "fold_by_execution = dict(enumerate(fold_records))" in validator
    assert 'int(item["query_ordinal"]): item' not in producer
    assert 'int(item["query_ordinal"]): item' not in validator
    assert 'binding_path(authority, "redacted_schedule")' in producer
    assert 'binding_path(authority, "redacted_schedule")' in validator
    assert "WORKER_COUNT = 4" in producer
    assert "WORKER_COUNT = 4" in validator
    assert 'multiprocessing.get_context("spawn")' in producer
    assert 'multiprocessing.get_context("spawn")' in validator
    assert "initializer=_initialize_worker" in producer
    assert "initializer=_initialize_worker" in validator
    assert 'sorted(worker_results, key=lambda value: value["execution_ordinal"])' in producer
    assert 'sorted(worker_results, key=lambda value: value["execution_ordinal"])' in validator
    assert "#SBATCH --array=0-49%20" in launcher
    assert "#SBATCH --cpus-per-task=4" in launcher
    assert "#SBATCH --mem=16384" in launcher
    assert "#SBATCH --time=01:15:00" in launcher
    assert "rc_authority_validation=" in launcher
    assert "DINO_RCDE_TRACK_R_RELATIVE_E1_PARALLEL_AUTHORITY_V118_REVISION2_INDEPENDENT_VALIDATION_PASS" in launcher
    assert 'v.get("authority_sha256")==sha(ap)' in launcher
    assert 'v.get("authority_logical_sha256")==a.get("logical_sha256")' in launcher
    assert 'a.get("authority_revision")==2' in launcher
    for name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
        assert f"export {name}_NUM_THREADS=1" in launcher


def test_formal_fold_sequence_is_execution_axis_not_sparse_historical_ordinal():
    root = Path(__file__).resolve().parents[1]
    import json

    folds = json.loads(
        (root / "protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json").read_text()
    )["records"]
    aggregate = json.loads(
        (root / "results/dino_rcde_sr0_mt_p_v2_formal_lock_aggregate_v1/result.json").read_text()
    )["rows"]
    roles = json.loads(
        (root / "results/dino_rcde_sr0_mt_loss_join_v1/loss_join.json").read_text()
    )["episodes"]
    explicit = json.loads(
        (root / "results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_oof_schedule.json").read_text()
    )["records"]
    excluded = {25, 26, 101, 346, 354, 470}
    assert len(folds) == 600
    assert len(explicit) == 600
    assert [int(item["query_ordinal"]) for item in folds] != list(range(600))
    by_lock = {int(item["execution_ordinal"]): item for item in aggregate}
    by_role = {int(item["execution_ordinal"]): item for item in roles}
    assert set(by_lock) == set(by_role) == set(range(600)) - excluded
    for execution, fold in enumerate(folds):
        assert explicit[execution]["execution_ordinal"] == execution
        assert explicit[execution]["historical_query_ordinal"] == fold["query_ordinal"]
        assert explicit[execution]["query_id"] == fold["query_id"]
        assert explicit[execution]["source_image_sha256"] == fold["source_image_sha256"]
        assert explicit[execution]["heldout_fold"] == fold["inner_fold"]
        if execution in excluded:
            continue
        assert by_lock[execution]["query_id"] == by_role[execution]["query_id"] == fold["query_id"]
        assert by_lock[execution]["source_fold"] == fold["inner_fold"]
        assert by_role[execution]["query_source_image_sha256"] == fold["source_image_sha256"]


def test_downstream_reducer_and_vfit_program_boundaries():
    root = Path(__file__).resolve().parents[1]
    reducer = (root / "programs/reduce_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py").read_text()
    ledger_validator = (root / "programs/validate_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py").read_text()
    input_common = (root / "programs/dino_rcde_track_r_input_common_v1.py").read_text()
    fit = (root / "programs/run_dino_rcde_track_r_relative_v_fit_v1.py").read_text()
    fit_validator = (root / "programs/validate_dino_rcde_track_r_relative_v_fit_v1.py").read_text()
    assert "EXPECTED_PER_OUTER = {1: 445, 2: 446, 3: 445, 4: 446}" in reducer
    assert "episode_multiplicity_per_query\": 3" in reducer
    assert "weights_only=True" in reducer and "mmap=True" in reducer
    assert "set(query_occurrences.values()) == {3}" in ledger_validator
    assert "weights_only=True" in ledger_validator and "mmap=True" in ledger_validator
    assert "reference_cache" in input_common and "query_cache" in input_common
    assert "PAIRWISE_RELATIVE_ONLY" in fit
    assert '"donor_null_absolute_loss_count": 0' in fit
    assert "execution_manifest_argv" in fit
    assert "fresh_resume_model_bit_exact" in fit_validator
    assert "torch.equal(primary_state[key], resumed_state[key])" in fit_validator
