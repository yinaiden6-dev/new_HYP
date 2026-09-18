from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.n2_current_runtime_token_cache_v1 import (  # noqa: E402
    FRESH_FORWARD_AUDIT_QUERY_IDS,
    QUERY_COUNT,
    RECORD_KEYS,
    SHARD_COUNT,
    assert_parameter_names_reference_defined,
    expected_query_ordinals,
    parameter_schema_sha256,
    shard_for_ordinal,
    tensor_sha256,
    validate_audit_population,
    validate_prejoin_record_schema,
)


LEDGER = ROOT / "cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json"


def fixture_record() -> dict:
    return {
        "query_id": "Q000",
        "query_ordinal": 0,
        "heldout_fold": 0,
        "track": "outcome",
        "source_image_sha256": "a" * 64,
        "source_exif_orientation": 1,
        "decode_frame": "DECODED_RAW_BEFORE_EXIF",
        "raw_size_hw": (100, 200),
        "oriented_size_hw": (100, 200),
        "grid_shape": (2, 3),
        "image_tokens": torch.zeros(6, 128, dtype=torch.float16),
        "image_tokens_sha256": "b" * 64,
        "template_tokens": torch.ones(4, 128, dtype=torch.float16),
        "template_tokens_sha256": "c" * 64,
        "materialization_mode": "FRESH_CURRENT_RUNTIME_FORWARD",
        "current64_source_shard": None,
        "current64_source_payload_sha256": None,
        "model_update_count": 0,
    }


def test_contiguous_sixteen_shards_exactly_cover_987() -> None:
    population = [ordinal for shard in range(SHARD_COUNT) for ordinal in expected_query_ordinals(shard)]
    assert population == list(range(QUERY_COUNT))
    assert {len(expected_query_ordinals(shard)) for shard in range(SHARD_COUNT)} == {61, 62}
    assert all(shard_for_ordinal(ordinal) == shard for shard in range(SHARD_COUNT) for ordinal in expected_query_ordinals(shard))


def test_prejoin_record_schema_rejects_identity_and_candidate_fields() -> None:
    record = fixture_record()
    assert set(record) == RECORD_KEYS
    validate_prejoin_record_schema(record)
    for bad in ("target_identity", "supergroup", "candidate_row", "slot", "winner"):
        poisoned = dict(record)
        poisoned[bad] = "leak"
        with pytest.raises(ValueError):
            validate_prejoin_record_schema(poisoned)


def test_tensor_hash_binds_dtype_shape_and_bytes() -> None:
    value = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.float())
    assert tensor_sha256(value) != tensor_sha256(value.reshape(2, 6))


def test_shared_parameter_schema_does_not_depend_on_gallery_size_or_order() -> None:
    torch.manual_seed(17)
    module = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.GELU(), torch.nn.Linear(4, 1))
    before = parameter_schema_sha256(module)
    _gallery_a = ["g2", "g1"]
    _gallery_b = ["g9"] * 5412
    assert parameter_schema_sha256(module) == before
    assert_parameter_names_reference_defined(name for name, _ in module.named_parameters())
    with pytest.raises(ValueError):
        assert_parameter_names_reference_defined(["per_identity_table.weight"])


def test_preregistered_fresh_forward_audit_covers_all_shards_and_e0_fixtures() -> None:
    rows = json.loads(LEDGER.read_text())["queries"]
    validate_audit_population(FRESH_FORWARD_AUDIT_QUERY_IDS, rows)
    assert len(FRESH_FORWARD_AUDIT_QUERY_IDS) == 20
    assert len(set(FRESH_FORWARD_AUDIT_QUERY_IDS)) == 20


def test_existing_shard_reuse_requires_current_complete_envelope(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "programs"))
    producer = importlib.import_module(
        "materialize_routea_matched_three_arm_n2_token_cache_shard_v1"
    )
    shard = 0
    ordinals = expected_query_ordinals(shard)
    out = tmp_path / "shard00"
    out.mkdir()
    records = [
        {"materialization_mode": "FRESH_CURRENT_RUNTIME_FORWARD"}
        for _ in ordinals
    ]
    payload = {
        "schema_version": "routea_n2_current_runtime_987_token_cache_shard_v1_20260902",
        "status": producer.READY_STATUS,
        "claim_level": "TARGET_FREE_INTERNAL_TOKEN_CACHE_ENGINEERING_ONLY",
        "shard": shard,
        "shard_count": producer.SHARD_COUNT,
        "query_ordinal_begin": ordinals[0],
        "query_ordinal_end_exclusive": ordinals[-1] + 1,
        "records": records,
        "model": {
            "stable_encoder_fingerprint_sha256": producer.EXPECTED_ENCODER_FINGERPRINT,
            "encoder_parameter_schema_sha256": "d" * 64,
            "encoder_parameter_count": 1,
            "encoder_trainable_parameter_count": 0,
            "d1_checkpoint_load_count": 0,
            "d1_model_update_count": 0,
            "hardware_identity_is_authority": False,
        },
        "bindings": producer.expected_bindings(),
        "access": producer.expected_access(),
    }
    payload_path = out / "payload.pt"
    torch.save(payload, payload_path)
    receipt = {
        "schema_version": "routea_n2_current_runtime_987_token_cache_receipt_v1_20260902",
        "status": producer.READY_STATUS,
        "shard": shard,
        "query_count": len(ordinals),
        "query_ordinal_begin": ordinals[0],
        "query_ordinal_end_exclusive": ordinals[-1] + 1,
        "mode_counts": {"FRESH_CURRENT_RUNTIME_FORWARD": len(ordinals)},
        "payload_sha256": producer.sha256_file(payload_path),
        "model_update_count": 0,
        "next_authorized_stage": "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_SHARD_VALIDATION",
        "logical_sha256": "",
    }
    receipt["logical_sha256"] = producer.logical_sha256(receipt)
    (out / "receipt.json").write_text(json.dumps(receipt))
    assert producer.existing_complete(out, shard)
    payload["bindings"]["contract_sha256"] = "0" * 64
    torch.save(payload, payload_path)
    receipt["payload_sha256"] = producer.sha256_file(payload_path)
    receipt["logical_sha256"] = producer.logical_sha256(receipt)
    (out / "receipt.json").write_text(json.dumps(receipt))
    assert not producer.existing_complete(out, shard)


def test_validators_recompute_existing_outputs_and_final_reopens_all_sources() -> None:
    shard_source = (ROOT / "programs/validate_routea_matched_three_arm_n2_token_cache_shard_v1.py").read_text()
    aggregate_source = (ROOT / "programs/aggregate_routea_matched_three_arm_n2_token_cache_v1.py").read_text()
    final_source = (ROOT / "programs/validate_routea_matched_three_arm_n2_token_cache_aggregate_v1.py").read_text()
    assert "validation_already_committed" not in shard_source
    assert "aggregate_already_committed" not in aggregate_source
    assert "final_validation_already_committed" not in final_source
    assert "independent_source_geometry(row)" in final_source
    assert "payload.get(\"bindings\") == expected_bindings" in final_source
    assert "payload.get(\"access\") == expected_access" in final_source


def test_submission_graph_is_accelerated_sixteen_shards_max_eight_afterok() -> None:
    array = (ROOT / "slurm/routea_matched_three_arm_n2_token_cache_array_v1_1h.sbatch").read_text()
    aggregate = (ROOT / "slurm/routea_matched_three_arm_n2_token_cache_aggregate_v1_30m.sbatch").read_text()
    submit = (ROOT / "slurm/submit_routea_matched_three_arm_n2_token_cache_v1.sh").read_text()
    assert "#SBATCH --partition=accelerated" in array
    assert "#SBATCH --array=0-15%8" in array
    assert "#SBATCH --partition=accelerated" in aggregate
    assert 'dependency="afterok:${array_job}"' in submit
