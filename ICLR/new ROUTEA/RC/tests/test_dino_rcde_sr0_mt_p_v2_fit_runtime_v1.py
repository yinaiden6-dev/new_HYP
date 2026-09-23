from __future__ import annotations

import copy

import pytest
import torch

from rc_aslo_xf import dino_rcde_sr0_mt_p_compact_catalog_v1 as C
from rc_aslo_xf.dino_rcde_cw1_multitile_sr0_p_v1 import FEATURE_SCHEMA_SHA256
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    canonical_sha256,
    initialize_training,
    tensor_sha256,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_v2_fit_runtime_v1 import (
    BoundShardPayloadV1,
    PFitV2RuntimeError,
    assert_compiled_role_free_exact_parity,
    build_and_seal_target_free_input_table,
    join_training_roles_after_seal,
    score_compiled_direction_v1,
    validate_v95_fit_manifest,
)


def _sha(value: object) -> str:
    return canonical_sha256(value)


def _mask(row_start: int, col_start: int) -> torch.Tensor:
    value = torch.zeros((8, 8), dtype=torch.bool)
    value[row_start : row_start + 4, col_start : col_start + 4] = True
    return value.flatten()


QUERY_MASKS = (
    _mask(0, 0),
    _mask(0, 2),
    _mask(2, 0),
    _mask(2, 2),
)
REFERENCE_MASKS = (_mask(0, 0), _mask(2, 2))


def _bool_rle(value: torch.Tensor) -> dict[str, object]:
    flat = value.flatten().tolist() + [False]
    runs = []
    start = None
    for index, item in enumerate(flat):
        if item and start is None:
            start = index
        elif not item and start is not None:
            runs.append([start, index - start])
            start = None
    return {
        "shape": list(value.shape),
        "flatten_order": "ROW_MAJOR",
        "true_runs": runs,
        "tensor_sha256": tensor_sha256(value),
    }


def _cache(*, geometry_h0: bool = False):
    blocks = []
    feature_rows = []
    eligibility_rows = []
    metadata = []
    cursor = 0
    execution = 8
    for member in range(2):
        candidate = _sha(["candidate", member])
        for direction_index, direction in enumerate(P_DIRECTIONS):
            features = torch.zeros((4, 2, 16), dtype=torch.float64)
            for root in range(4):
                features[root, 0, 0] = 1.0 + member + direction_index + root / 10
                features[root, 1, 0] = 1.5 + member + direction_index + root / 10
                features[root, :, 1] = torch.tensor([0.2, 0.3])
            eligible = torch.ones((4, 2), dtype=torch.bool)
            if geometry_h0 and member == 0 and direction == "a_to_b":
                eligible[0, 0] = False
                features[0, 0] = 0.0
            flat = features.reshape(-1, 16)
            flags = eligible.reshape(-1)
            base = {
                "member_ordinal": member,
                "candidate_key": candidate,
                "direction": direction,
                "source_population_sha256": _sha(["source", member, direction]),
                "root_count": 4,
                "action_count": 2,
                "feature_offset": cursor,
                "coordinate_count": 8,
            }
            payload = {
                **base,
                "feature_tensor_sha256": C._tensor_bytes_sha256(flat),
                "eligibility_sha256": C._tensor_bytes_sha256(flags),
            }
            block = C.PairFeatureBlock(
                **base, block_sha256=C.canonical_sha256(payload)
            )
            blocks.append(block)
            feature_rows.append(flat)
            eligibility_rows.append(flags)
            metadata.append((member, direction, candidate, features, eligible, block))
            cursor += 8
    features = torch.cat(feature_rows)
    eligibility = torch.cat(eligibility_rows)
    provisional = object.__new__(C.RoleFreePairFeatureCache)
    object.__setattr__(provisional, "pair_address_sha256", _sha("pair"))
    object.__setattr__(provisional, "blocks", tuple(blocks))
    object.__setattr__(provisional, "features", features)
    object.__setattr__(provisional, "eligibility", eligibility)
    cache = C.RoleFreePairFeatureCache(
        pair_address_sha256=_sha("pair"),
        blocks=tuple(blocks),
        features=features,
        eligibility=eligibility,
        cache_sha256=C.canonical_sha256(provisional._hash_payload()),
    )
    record = {
        "query_id": "Q-008",
        "execution_ordinal": execution,
        "pair_address_sha256": cache.pair_address_sha256,
        "pair_feature_cache": cache.payload(),
    }
    summary = {
        "query_id": "Q-008",
        "execution_ordinal": execution,
        "pair_address_sha256": cache.pair_address_sha256,
        "cache_sha256": cache.cache_sha256,
        "source_population_sha256s": [item.source_population_sha256 for item in cache.blocks],
        "block_count": 4,
        "coordinate_count": int(cache.features.shape[0]),
        "ready_coordinate_count": int(cache.eligibility.sum()),
        "h0_coordinate_count": int((~cache.eligibility).sum()),
    }
    shard = {
        "schema_version": "rc_dino_rcde_sr0_mt_role_free_pair_feature_cache_shard_v1_20260817",
        "status": "RCDE_SR0_MT_ROLE_FREE_PAIR_FEATURE_CACHE_SHARD_READY",
        "role_free": True,
        "target_free": True,
        "synthetic": False,
        "execution_start": 0,
        "execution_stop": 12,
        "pair_query_count": 1,
        "record_summaries": [summary],
        "record_population_sha256": _sha([summary]),
        "records": [record],
    }
    shard["logical_sha256"] = _sha(
        {key: item for key, item in shard.items() if key != "records"}
    )
    return cache, metadata, shard


def _sidecar_record(member, direction, candidate, features, eligible, block, cache):
    action_keys = [
        [_sha(["action", member, direction, root, action]) for action in range(2)]
        for root in range(4)
    ]
    query_masks = [
        C.CompactMask.from_tensor(item, grid_shape=(8, 8)).payload()
        for item in QUERY_MASKS
    ]
    reference_masks = [
        C.CompactMask.from_tensor(item, grid_shape=(8, 8)).payload()
        for item in REFERENCE_MASKS
    ]
    reference_index = [[0, 1] for _ in range(4)]
    value = {
        "schema_version": "fixture-sidecar-record-v1",
        "query_id": "Q-008",
        "query_source_image_sha256": _sha("query-source"),
        "execution_ordinal": 8,
        "pair_address_sha256": cache.pair_address_sha256,
        "member_ordinal": member,
        "candidate_axis_sha256": _sha("candidate-axis"),
        "candidate_count_per_query": 128,
        "candidate_position": member,
        "candidate_key": candidate,
        "candidate_physical_row": member,
        "candidate_reference_source_sha256": _sha(["reference", member]),
        "direction": direction,
        "source_population_sha256": block.source_population_sha256,
        "cache_sha256": cache.cache_sha256,
        "cache_block_sha256": block.block_sha256,
        "root_count": 4,
        "action_count": 2,
        "coordinate_count": 8,
        "ready_coordinate_count": int(eligible.sum()),
        "h0_coordinate_count": int((~eligible).sum()),
        "cache_feature_tensor_sha256": tensor_sha256(features),
        "cache_eligibility_sha256": tensor_sha256(eligible),
        "colnomic_query_grid_shape": [8, 8],
        "deployment_query_grid_shape": [8, 8],
        "deployment_reference_grid_shape": [8, 8],
        "query_geometry_sha256": _sha("query-geometry"),
        "reference_geometry_sha256": _sha(["reference-geometry", member]),
        "action_keys": action_keys,
        "action_keys_sha256": _sha(action_keys),
        "eligibility_rle": _bool_rle(eligible),
        "deployment_query_root_masks": query_masks,
        "deployment_query_root_masks_sha256": _sha(query_masks),
        "deployment_reference_mask_table": reference_masks,
        "deployment_reference_mask_table_sha256": _sha(reference_masks),
        "deployment_reference_mask_index": reference_index,
        "deployment_reference_mask_index_sha256": _sha(reference_index),
        "deployment_reference_mask_index_tensor_sha256": tensor_sha256(
            torch.tensor(reference_index)
        ),
    }
    value["record_sha256"] = _sha(value)
    return value


def _fixture(*, geometry_h0: bool = False, target_member: int = 0):
    cache, metadata, pair_shard = _cache(geometry_h0=geometry_h0)
    records = [
        _sidecar_record(*item, cache) for item in metadata
    ]
    sidecar = {
        "schema_version": "rc_dino_rcde_sr0_mt_training_consumer_e0_full594_structure_sidecar_shard_v1_20260818",
        "status": "RCDE_SR0_MT_TRAINING_CONSUMER_E0_FULL594_STRUCTURE_SIDECAR_SHARD_READY",
        "target_free": True,
        "execution_start": 0,
        "execution_stop": 12,
        "records": records,
        "record_population_sha256": _sha(records),
    }
    sidecar["logical_sha256"] = _sha(sidecar)
    episode = {
        "execution_ordinal": 8,
        "query_id": "Q-008",
        "query_source_image_sha256": _sha("query-source"),
        "pair_address_sha256": cache.pair_address_sha256,
        "target_member_ordinal": target_member,
        "target_candidate_key": _sha(["candidate", target_member]),
        "rival_member_ordinal": 1 - target_member,
        "rival_candidate_key": _sha(["candidate", 1 - target_member]),
    }
    pair_route = {
        "shard_ordinal": 0,
        "execution_start": 0,
        "execution_stop": 12,
        "selected_execution_ordinals": [8],
        "artifact_path": "cache/pair0.pt",
        "artifact_sha256": _sha("pair-file"),
        "validation_path": "results/pair0.validation.json",
        "validation_sha256": _sha("pair-validation"),
    }
    side_route = {
        **pair_route,
        "artifact_path": "results/side0.json",
        "artifact_sha256": _sha("side-file"),
        "validation_path": "results/side0.validation.json",
        "validation_sha256": _sha("side-validation"),
    }
    source_names = (
        "pair_aggregate_logical_sha256",
        "pair_validation_logical_sha256",
        "loss_join_logical_sha256",
        "loss_validation_logical_sha256",
        "membership_validation_logical_sha256",
        "sidecar_manifest_logical_sha256",
        "v94_adapter_validation_logical_sha256",
        "feature_implementation_sha256",
        "numeric_policy_sha256",
        "p_v2_adapter_sha256",
        "p_runtime_sha256",
        "role_free_scorer_sha256",
    )
    source = {name: _sha(name) for name in source_names}
    source["feature_schema_sha256"] = FEATURE_SCHEMA_SHA256
    recipe = {
        "seed": 17,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip_l2": 1.0,
        "updates_total": 2048,
        "episodes_per_update": 4,
        "warmup_updates": 128,
        "final_learning_rate": 0.00003,
        "checkpoint_interval_updates": 64,
        "early_stop": False,
        "auxiliary_loss": False,
        "direction_fusion_before_one_pair_loss": True,
        "loss": "softplus(-(U_target-U_rival))",
    }
    manifest = {
        "schema_version": "rc_dino_rcde_sr0_mt_p_v2_training_manifest_v1_20260818",
        "status": "RCDE_SR0_MT_P_V2_TRAINING_MANIFEST_READY",
        "claim_level": "FORMAL_P_V2_TRAINING_INPUT_MANIFEST_ONLY",
        "fit_id": "P_OUTER1_INNER2_FIT",
        "fit_role": "INNER_TRAIN_FIT",
        "outer_fold": 1,
        "inner_heldout_fold": 2,
        "identity_disjoint": True,
        "supergroup_disjoint": True,
        "train_query_count": 1,
        "eligible_episode_count": 1,
        "episodes": [episode],
        "episode_population_sha256": _sha([episode]),
        "membership_path": "results/memberships/P_OUTER1_INNER2_FIT/membership.json",
        "membership_file_sha256": _sha("membership-file"),
        "membership_logical_sha256": _sha("membership-logical"),
        "pair_feature_shards": [pair_route],
        "pair_feature_shards_sha256": _sha([pair_route]),
        "structure_sidecar_shards": [side_route],
        "structure_sidecar_shards_sha256": _sha([side_route]),
        "source_bindings": source,
        "recipe": recipe,
        "output_namespace": "results/dino_rcde_sr0_mt_p_v2_formal_fits_v1/P_OUTER1_INNER2_FIT",
        "fresh_resume_exact_required": True,
        "independent_fit_validation_required": True,
        "training_authorized": False,
        "consumable_checkpoint_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    manifest["logical_sha256"] = _sha(manifest)
    return manifest, pair_shard, sidecar


def _compile(manifest, pair, sidecar):
    validated = validate_v95_fit_manifest(manifest)
    table = build_and_seal_target_free_input_table(
        validated,
        pair_shards=(BoundShardPayloadV1("cache/pair0.pt", _sha("pair-file"), pair),),
        sidecar_shards=(BoundShardPayloadV1("results/side0.json", _sha("side-file"), sidecar),),
    )
    return validated, table


def test_v95_manifest_is_strict_and_recipe_bound() -> None:
    manifest, _, _ = _fixture()
    assert validate_v95_fit_manifest(manifest).fit_id == "P_OUTER1_INNER2_FIT"
    extra = copy.deepcopy(manifest)
    extra["legacy_execution_scope"] = "FORMAL_NATURAL"
    extra["logical_sha256"] = _sha(
        {key: item for key, item in extra.items() if key != "logical_sha256"}
    )
    with pytest.raises(PFitV2RuntimeError, match="field set"):
        validate_v95_fit_manifest(extra)
    changed = copy.deepcopy(manifest)
    changed["recipe"]["updates_total"] = 2047
    changed["logical_sha256"] = _sha(
        {key: item for key, item in changed.items() if key != "logical_sha256"}
    )
    with pytest.raises(PFitV2RuntimeError, match="recipe"):
        validate_v95_fit_manifest(changed)


def test_target_free_table_is_complete_and_roles_join_only_after_seal() -> None:
    manifest, pair, sidecar = _fixture(target_member=0)
    validated, table = _compile(manifest, pair, sidecar)
    assert len(table.inputs) == 4
    assert table.role_join_count == 0
    assert table.geometry_h0_coordinate_count == 0
    reversed_manifest, _, _ = _fixture(target_member=1)
    reversed_validated = validate_v95_fit_manifest(reversed_manifest)
    reversed_table = build_and_seal_target_free_input_table(
        reversed_validated,
        pair_shards=(BoundShardPayloadV1("cache/pair0.pt", _sha("pair-file"), pair),),
        sidecar_shards=(BoundShardPayloadV1("results/side0.json", _sha("side-file"), sidecar),),
    )
    assert reversed_table.table_sha256 == table.table_sha256
    joined = join_training_roles_after_seal(table, validated)
    reversed_joined = join_training_roles_after_seal(reversed_table, reversed_validated)
    assert joined.target_free_table_sha256 == reversed_joined.target_free_table_sha256
    assert joined.episodes[0].target[0].candidate_key != reversed_joined.episodes[0].target[0].candidate_key


def test_compiled_scorer_is_exact_and_uses_precomputed_rows() -> None:
    manifest, pair, sidecar = _fixture()
    _, table = _compile(manifest, pair, sidecar)
    model, _ = initialize_training()
    for value in table.inputs.values():
        assert value.row_ready
        assert len(value.row_ready) == len(value.row_weights) == len(value.regions)
        assert_compiled_role_free_exact_parity(model, value)
        observed = score_compiled_direction_v1(model, value)
        assert bool(observed.row_ready.any())
        assert torch.isfinite(observed.candidate_utility)


def test_geometry_h0_coordinate_fails_before_seal() -> None:
    manifest, pair, sidecar = _fixture(geometry_h0=True)
    validated = validate_v95_fit_manifest(manifest)
    with pytest.raises(PFitV2RuntimeError, match="geometry-H0"):
        build_and_seal_target_free_input_table(
            validated,
            pair_shards=(BoundShardPayloadV1("cache/pair0.pt", _sha("pair-file"), pair),),
            sidecar_shards=(BoundShardPayloadV1("results/side0.json", _sha("side-file"), sidecar),),
        )


def test_pair_sidecar_binding_and_shard_population_fail_closed() -> None:
    manifest, pair, sidecar = _fixture()
    validated = validate_v95_fit_manifest(manifest)
    bad = copy.deepcopy(sidecar)
    bad["records"][0]["candidate_key"] = _sha("wrong")
    bad["records"][0]["record_sha256"] = _sha(
        {key: item for key, item in bad["records"][0].items() if key != "record_sha256"}
    )
    bad["record_population_sha256"] = _sha(bad["records"])
    bad["logical_sha256"] = _sha(
        {key: item for key, item in bad.items() if key != "logical_sha256"}
    )
    with pytest.raises(PFitV2RuntimeError, match="address or population"):
        build_and_seal_target_free_input_table(
            validated,
            pair_shards=(BoundShardPayloadV1("cache/pair0.pt", _sha("pair-file"), pair),),
            sidecar_shards=(BoundShardPayloadV1("results/side0.json", _sha("side-file"), bad),),
        )
    with pytest.raises(PFitV2RuntimeError, match="shard path population"):
        build_and_seal_target_free_input_table(
            validated, pair_shards=(), sidecar_shards=()
        )
