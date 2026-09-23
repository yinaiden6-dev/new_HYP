from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    value = (ROOT / relative).read_text(encoding="utf-8")
    ast.parse(value)
    return value


def test_e1_shard_reduce_and_fit_chain_is_stage_separated():
    shard = _source("programs/materialize_dino_rcde_track_r_relative_e1_episode_shard_v1.py")
    shard_validator = _source("programs/validate_dino_rcde_track_r_relative_e1_episode_shard_v1.py")
    reducer = _source("programs/reduce_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py")
    ledger_validator = _source("programs/validate_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py")
    fit = _source("programs/run_dino_rcde_track_r_relative_v_fit_v1.py")
    fit_validator = _source("programs/validate_dino_rcde_track_r_relative_v_fit_v1.py")
    assert "score-free loss-role join" in shard
    assert '"dino_token_load_count": 0' in shard
    assert "build_compact_episode" in shard_validator
    assert "594/6 query closure" in reducer
    assert "EXPECTED_PER_OUTER = {1: 445, 2: 446, 3: 445, 4: 446}" in reducer
    assert "set(query_occurrences.values()) == {3}" in ledger_validator
    assert "execution_manifest_argv" in fit
    assert "PAIRWISE_RELATIVE_ONLY" in fit
    assert '"donor_null_absolute_loss_count": 0' in fit
    assert "fresh_resume_model_bit_exact" in fit_validator
    assert "torch.equal(primary_state[key], resumed_state[key])" in fit_validator


def test_outer_oof_prejoin_is_role_free_and_executes_controls():
    producer = _source("programs/materialize_dino_rcde_track_r_oof_prejoin_shard_v1.py")
    validator = _source("programs/validate_dino_rcde_track_r_oof_prejoin_shard_v1.py")
    assert "role_free_pair_manifest" in producer
    assert "loss_join" not in producer
    assert "label_target_rival_read_authorized" in producer
    assert "decode_pair(model, query_device, locks_device, left, right)" in producer
    assert "c_dino_v_derangement" in producer
    assert "permute_query_content" in producer
    assert "_p_reference_score" in producer
    assert "candidate_reorder_max_abs" in validator
    assert "pair_swap_max_abs" in validator
    assert "label_target_rival_read_count" in validator
    assert "weights_only=True" in producer and "mmap=True" in producer
    assert "weights_only=True" in validator and "mmap=True" in validator
    assert "fold_by_execution = dict(enumerate(fold_records))" in producer
    assert "fold_records = dict(enumerate(fold_sequence))" in validator
    assert 'int(item["query_ordinal"]): item' not in producer
    assert 'int(item["query_ordinal"]): item' not in validator


def test_p_utility_comparator_is_target_free_and_direction_fused():
    producer = _source("programs/materialize_dino_rcde_track_r_p_v2_utility_shard_v1.py")
    validator = _source("programs/validate_dino_rcde_track_r_p_v2_utility_shard_v1.py")
    reducer = _source("programs/reduce_dino_rcde_track_r_p_v2_utility_v1.py")
    assert "OUTER_TRAIN_REFIT" in producer
    assert "fuse_direction_utilities" in producer
    assert '"target_free": True' in producer
    assert '"target_rival_read_count": 0' in producer
    assert "candidate_count_per_query\": 128" in reducer
    assert "target_free_prejoin_score_ledger\": True" in reducer
    assert "P utility direction fusion drift" in validator
    assert "weights_only=True" in producer and "mmap=True" in producer
    assert "fold_by_execution = dict(enumerate(fold_records))" in producer
    assert 'int(item["query_ordinal"]): item' not in producer


def test_science_reducer_separates_A_B_and_forbids_postjoin_forward():
    reducer = _source("programs/reduce_dino_rcde_track_r_science_v1.py")
    validator = _source("programs/validate_dino_rcde_track_r_science_v1.py")
    assert "evaluate_track_r" in reducer
    assert '"decision_A"' in reducer
    assert '"decision_B"' in reducer
    assert '"postjoin_model_forward_count": 0' in reducer
    assert "p_v2_utility_comparator" in reducer
    assert "evaluate_track_r" in validator
    assert "independent Track-R statistical replay drift" in validator
    assert "postjoin_model_forward_count" in validator
    assert "weights_only=True" in reducer and "mmap=True" in reducer
    assert "weights_only=True" in validator and "mmap=True" in validator


def test_future_gpu_launchers_accept_only_approved_partition_aliases():
    fit = (
        ROOT / "slurm/dino_rcde_track_r_relative_v_fit_v1.sbatch"
    ).read_text(encoding="utf-8")
    assert 'test "${SLURM_JOB_PARTITION:?}" = accelerated' in fit
    assert "dev_accelerated-h100" not in fit
    assert 'test -n "${CUDA_VISIBLE_DEVICES:-}"' in fit

    oof = (
        ROOT / "slurm/dino_rcde_track_r_oof_prejoin_shards_v1.sbatch"
    ).read_text(encoding="utf-8")
    assert 'case "${SLURM_JOB_PARTITION:?}" in' in oof
    assert "accelerated|dev_accelerated-h100)" in oof
    assert 'test "${SLURM_JOB_PARTITION:?}" =' not in oof
    assert 'test -n "${CUDA_VISIBLE_DEVICES:-}"' in oof


def test_future_authority_parent_and_controller_bindings_match_stage_numbers():
    reducer_freezer = _source(
        "programs/freeze_dino_rcde_track_r_relative_e1_episode_reduction_authority_v119_r2.py"
    )
    fit_freezer = _source(
        "programs/freeze_dino_rcde_track_r_relative_v_fit_authority_v119.py"
    )
    oof_freezer = _source(
        "programs/freeze_dino_rcde_track_r_oof_prejoin_authority_v120.py"
    )
    assert '"parent_authority_v118"' in reducer_freezer
    assert "post_e1_controller_v3.sbatch" in reducer_freezer
    assert '"parent_authority_v119"' in fit_freezer
    assert '"parent_authority_v120"' in oof_freezer
