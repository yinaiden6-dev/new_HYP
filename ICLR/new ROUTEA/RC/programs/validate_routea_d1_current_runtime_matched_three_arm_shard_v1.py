#!/usr/bin/env python3
"""Independent formula and lineage validation for one matched A/B/C shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_romav2_colnomic_visibility_xf_six_case_v1 as core  # noqa: E402
import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_V1_20260902.md"
SOURCE_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
SOURCE_VALIDATION = SOURCE_ROOT / "validation.json"
REFERENCE_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1"
REFERENCE_PAYLOAD = REFERENCE_ROOT / "payload.pt"
REFERENCE_VALIDATION = REFERENCE_ROOT / "independent_validation.json"
LABEL_JOIN_HANDOFF = ROOT / "results/routea_d1_current_runtime_64_label_join_v1/independent_validation.json"
RAW_C_ROOT = ROOT / "results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1"
RAW_C_VALIDATION = RAW_C_ROOT / "validation.json"
ROMA_WEIGHTS = WORKSPACE / "third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt"
EXPECTED_ROMA_CHECKPOINT_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
PROCESSOR_CONFIG = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b/preprocessor_config.json"
OUT_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_prejoin_v1"
SHARD_COUNT = 8
ARM_NAMES = ("A_ALL", "B_QUERY", "C_PAIRED")
PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "shard",
    "shard_count",
    "arm_names",
    "records",
    "bindings",
    "access",
}
RECORD_KEYS = {
    "role",
    "execution_ordinal",
    "query_id",
    "track",
    "heldout_fold",
    "raw_candidate_physical_rows",
    "d1_candidate_physical_rows",
    "union_physical_rows",
    "raw_cbind_destination_to_source_physical_rows",
    "d1_cbind_destination_to_source_physical_rows",
    "candidates",
    "target_role_read_count",
    "target_insertion_count",
    "model_update_count",
}
CANDIDATE_KEYS = {
    "union_ordinal",
    "physical_row",
    "raw_position",
    "d1_position",
    "reference_tokens_sha256",
    "reference_source_image_sha256",
    "reference_source_logical_sha256",
    "query_map",
    "reference_map",
    "query_map_sha256",
    "reference_map_sha256",
    "raw_arm_scores",
    "d1_arm_scores",
}
RECEIPT_KEYS = {
    "schema_version",
    "status",
    "shard",
    "query_count",
    "candidate_union_counts",
    "payload_sha256",
    "target_role_read_count",
    "model_update_count",
    "next_authorized_stage",
    "logical_sha256",
}


def map_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def independent_scores(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    query_map: torch.Tensor,
    reference_map: torch.Tensor,
) -> dict[str, float]:
    similarity = F.normalize(query_tokens.to(torch.float64), dim=1) @ F.normalize(
        reference_tokens.to(torch.float64), dim=1
    ).T
    q = query_map.to(torch.float64)
    r = reference_map.to(torch.float64)
    local_all = similarity.max(dim=1).values
    denominator = q.sum().clamp_min(1e-12)
    values = {
        "A_ALL": float(local_all.mean()),
        "B_QUERY": float(torch.sqrt(q.mean()) * (q * local_all).sum() / denominator),
        "C_PAIRED": float(
            torch.sqrt(q.mean() * r.mean())
            * (q * (similarity * r[None]).max(dim=1).values).sum()
            / denominator
        ),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("independent arm score became non-finite")
    return values


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shard not in range(SHARD_COUNT):
        raise ValueError("shard must be in 0..7")
    shard_dir = OUT_ROOT / f"shard{args.shard:02d}"
    payload_path = shard_dir / "payload.pt"
    receipt_path = shard_dir / "receipt.json"
    out = shard_dir / "validation.json"
    if out.exists():
        raise RuntimeError(f"immutable matched-three-arm validation exists: {out}")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    receipt = json.loads(receipt_path.read_text())
    source_path = SOURCE_ROOT / f"shard{args.shard:02d}/payload.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    source_by_execution = {
        int(record["execution_ordinal"]): record for record in source["records"]
    }
    references = torch.load(
        REFERENCE_PAYLOAD,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )["references"]
    raw_c_path = RAW_C_ROOT / f"shard{args.shard:02d}/result.json"
    raw_c = json.loads(raw_c_path.read_text())
    raw_c_by_execution = {
        int(record["execution_ordinal"]): record for record in raw_c["rows"]
    }
    records = payload.get("records", [])
    record_checks: list[bool] = []
    score_max_abs = 0.0
    live_reference_sha256: dict[int, str] = {}
    for record in records:
        if set(record) != RECORD_KEYS:
            raise RuntimeError("matched-three-arm record schema drift")
        execution = int(record["execution_ordinal"])
        source_record = source_by_execution[execution]
        raw_axis = list(map(int, source_record["raw_candidate_physical_rows"]))
        d1_axis = list(map(int, source_record["d1_candidate_physical_rows"]))
        union_axis = sorted(set(raw_axis).union(d1_axis))
        raw_positions = {row: index for index, row in enumerate(raw_axis)}
        d1_positions = {row: index for index, row in enumerate(d1_axis)}
        authority = raw_c_by_execution[execution]
        authority_axis = [
            int(item["physical_row"]) for item in authority["candidates"]
        ]
        authority_query_axis_ok = (
            authority.get("query_id") == source_record["query_id"]
            and authority.get("role") == source_record["role"]
            and authority.get("track") == source_record["track"]
            and authority_axis == raw_axis
        )
        authority_candidates = {
            int(item["physical_row"]): item for item in authority["candidates"]
        }
        candidate_checks: list[bool] = []
        for ordinal, candidate in enumerate(record["candidates"]):
            if set(candidate) != CANDIDATE_KEYS:
                raise RuntimeError("matched-three-arm candidate schema drift")
            physical_row = int(candidate["physical_row"])
            reference = references[physical_row]
            observed_reference_sha256 = live_reference_sha256.setdefault(
                physical_row, e0.sha256_file(Path(reference["source_path"]))
            )
            query_map = torch.as_tensor(candidate["query_map"])
            reference_map = torch.as_tensor(candidate["reference_map"])
            raw_scores = independent_scores(
                source_record["raw_image_tokens"],
                reference["tokens"],
                query_map,
                reference_map,
            )
            d1_scores = independent_scores(
                source_record["adapted_image_tokens"],
                reference["tokens"],
                query_map,
                reference_map,
            )
            for name in ARM_NAMES:
                score_max_abs = max(
                    score_max_abs,
                    abs(raw_scores[name] - float(candidate["raw_arm_scores"][name])),
                    abs(d1_scores[name] - float(candidate["d1_arm_scores"][name])),
                )
            raw_authority_ok = True
            if physical_row in raw_positions:
                old = authority_candidates[physical_row]
                raw_authority_ok = (
                    candidate["query_map_sha256"] == old["query_map_sha256"]
                    and candidate["reference_map_sha256"]
                    == old["reference_map_sha256"]
                    and abs(raw_scores["C_PAIRED"] - float(old["real_score"]))
                    <= 1e-12
                )
            candidate_checks.append(
                candidate["union_ordinal"] == ordinal
                and physical_row == union_axis[ordinal]
                and candidate["raw_position"] == raw_positions.get(physical_row)
                and candidate["d1_position"] == d1_positions.get(physical_row)
                and candidate["reference_tokens_sha256"]
                == reference["tokens_sha256"]
                and candidate["reference_source_image_sha256"]
                == reference["source_image_sha256"]
                == observed_reference_sha256
                and candidate["reference_source_logical_sha256"]
                == reference["source_logical_sha256"]
                and query_map.dtype == reference_map.dtype == torch.float64
                and query_map.ndim == reference_map.ndim == 1
                and query_map.shape
                == (source_record["raw_image_tokens"].shape[0],)
                and reference_map.shape == (reference["tokens"].shape[0],)
                and bool(torch.isfinite(query_map).all())
                and bool(torch.isfinite(reference_map).all())
                and float(query_map.min()) >= 0.0
                and float(query_map.max()) <= 1.0
                and float(reference_map.min()) >= 0.0
                and float(reference_map.max()) <= 1.0
                and candidate["query_map_sha256"] == map_sha256(query_map)
                and candidate["reference_map_sha256"] == map_sha256(reference_map)
                and set(candidate["raw_arm_scores"]) == set(ARM_NAMES)
                and set(candidate["d1_arm_scores"]) == set(ARM_NAMES)
                and all(
                    abs(raw_scores[name] - float(candidate["raw_arm_scores"][name]))
                    <= 1e-12
                    and abs(d1_scores[name] - float(candidate["d1_arm_scores"][name]))
                    <= 1e-12
                    for name in ARM_NAMES
                )
                and raw_authority_ok
            )
        record_checks.append(
            record["role"] == source_record["role"]
            and record["query_id"] == source_record["query_id"]
            and record["track"] == source_record["track"]
            and e0.sha256_file(Path(source_record["query_source_path"]))
            == source_record["query_source_sha256"]
            and record["heldout_fold"] == source_record["heldout_fold"]
            and record["raw_candidate_physical_rows"] == raw_axis
            and record["d1_candidate_physical_rows"] == d1_axis
            and record["union_physical_rows"] == union_axis
            and record["raw_cbind_destination_to_source_physical_rows"]
            == raw_axis[64:] + raw_axis[:64]
            and record["d1_cbind_destination_to_source_physical_rows"]
            == d1_axis[64:] + d1_axis[:64]
            and len(record["candidates"]) == len(union_axis)
            and authority_query_axis_ok
            and all(candidate_checks)
            and record["target_role_read_count"] == 0
            and record["target_insertion_count"] == 0
            and record["model_update_count"] == 0
        )

    source_validation = json.loads(SOURCE_VALIDATION.read_text())
    reference_validation = json.loads(REFERENCE_VALIDATION.read_text())
    label_join_handoff = json.loads(LABEL_JOIN_HANDOFF.read_text())
    raw_c_validation = json.loads(RAW_C_VALIDATION.read_text())
    source_seal = {
        int(item["shard"]): item for item in source_validation["shards"]
    }[args.shard]
    raw_c_seal = {
        int(item["shard"]): item for item in raw_c_validation["shards"]
    }[args.shard]
    expected_bindings = {
        "contract_sha256": e0.sha256_file(CONTRACT),
        "source_payload_sha256": e0.sha256_file(source_path),
        "source_validation_sha256": e0.sha256_file(SOURCE_VALIDATION),
        "reference_payload_sha256": e0.sha256_file(REFERENCE_PAYLOAD),
        "reference_validation_sha256": e0.sha256_file(REFERENCE_VALIDATION),
        "label_join_handoff_sha256": e0.sha256_file(LABEL_JOIN_HANDOFF),
        "raw_c_authority_sha256": e0.sha256_file(raw_c_path),
        "raw_c_validation_sha256": e0.sha256_file(RAW_C_VALIDATION),
        "roma_checkpoint_sha256": e0.sha256_file(ROMA_WEIGHTS),
        "processor_config_sha256": e0.sha256_file(PROCESSOR_CONFIG),
    }
    expected_access = {
        "target_identity_read_count": 0,
        "target_role_read_count": 0,
        "supergroup_read_count": 0,
        "sealed_read_count": 0,
        "target_insertion_count": 0,
        "model_update_count": 0,
    }
    checks = {
        "payload_envelope": set(payload) == PAYLOAD_KEYS
        and payload.get("schema_version")
        == "routea_d1_current_runtime_matched_three_arm_prejoin_shard_v1_20260902"
        and payload.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARD_READY"
        and payload.get("claim_level")
        == "TARGET_FREE_MATCHED_A_B_C_RAW_AND_D1_PORTABILITY_DIAGNOSTIC"
        and payload.get("shard") == args.shard
        and payload.get("shard_count") == SHARD_COUNT
        and tuple(payload.get("arm_names", ())) == ARM_NAMES,
        "input_authorities": source_validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and source_validation.get("logical_sha256")
        == e0.logical_sha256(source_validation)
        and all(source_validation.get("checks", {}).values())
        and source_seal.get("payload_sha256") == e0.sha256_file(source_path)
        and source_seal.get("receipt_sha256")
        == e0.sha256_file(SOURCE_ROOT / f"shard{args.shard:02d}/receipt.json")
        and source_seal.get("validation_sha256")
        == e0.sha256_file(SOURCE_ROOT / f"shard{args.shard:02d}/validation.json")
        and reference_validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
        and reference_validation.get("logical_sha256")
        == e0.logical_sha256(reference_validation)
        and all(reference_validation.get("checks", {}).values())
        and reference_validation.get("payload_sha256")
        == e0.sha256_file(REFERENCE_PAYLOAD)
        and label_join_handoff.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_64_LABEL_JOIN_INDEPENDENT_VALIDATION_PASS"
        and label_join_handoff.get("logical_sha256")
        == e0.logical_sha256(label_join_handoff)
        and all(label_join_handoff.get("checks", {}).values())
        and label_join_handoff.get("next_authorized_stage")
        == "D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_TARGET_FREE_MATERIALIZATION"
        and raw_c_validation.get("status")
        == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS"
        and all(raw_c_validation.get("checks", {}).values())
        and raw_c_validation.get("target_label_read_count") == 0
        and raw_c_seal.get("sha256") == e0.sha256_file(raw_c_path),
        "bindings": payload.get("bindings") == expected_bindings,
        "checkpoint_direct_hash": e0.sha256_file(ROMA_WEIGHTS)
        == EXPECTED_ROMA_CHECKPOINT_SHA256,
        "access": payload.get("access") == expected_access,
        "record_population": len(records) == len(source_by_execution) == 8
        and len({record["execution_ordinal"] for record in records}) == 8,
        "record_formula_and_raw_authority_replay": all(record_checks)
        and score_max_abs <= 1e-12,
        "receipt": set(receipt) == RECEIPT_KEYS
        and receipt.get("schema_version")
        == "routea_d1_current_runtime_matched_three_arm_prejoin_receipt_v1_20260902"
        and receipt.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARD_READY"
        and receipt.get("logical_sha256") == e0.logical_sha256(receipt)
        and receipt.get("shard") == args.shard
        and receipt.get("query_count") == 8
        and receipt.get("payload_sha256") == e0.sha256_file(payload_path)
        and receipt.get("candidate_union_counts")
        == [len(record["union_physical_rows"]) for record in records]
        and receipt.get("target_role_read_count") == 0
        and receipt.get("model_update_count") == 0,
        "receipt_next_stage": receipt.get("next_authorized_stage")
        == "MATCHED_THREE_ARM_SHARD_VALIDATION",
    }
    passed = all(checks.values())
    value = {
        "schema_version": "routea_d1_current_runtime_matched_three_arm_prejoin_shard_validation_v1_20260902",
        "status": (
            "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARD_VALIDATED"
            if passed
            else "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARD_VALIDATION_ABORT"
        ),
        "shard": args.shard,
        "checks": checks,
        "query_count": len(records),
        "formula_score_max_abs": score_max_abs,
        "payload_sha256": e0.sha256_file(payload_path),
        "receipt_sha256": e0.sha256_file(receipt_path),
        "target_role_read_count": 0,
        "model_update_count": 0,
        "next_authorized_stage": (
            "MATCHED_THREE_ARM_AGGREGATE_VALIDATION" if passed else None
        ),
        "logical_sha256": "",
    }
    value["logical_sha256"] = e0.logical_sha256(value)
    atomic_json(out, value)
    print(json.dumps({"status": value["status"], "checks": checks}, sort_keys=True))
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
