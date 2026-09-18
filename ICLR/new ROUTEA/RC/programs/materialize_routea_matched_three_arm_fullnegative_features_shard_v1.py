#!/usr/bin/env python3
"""Seal target-free common/native features for current RAW full negatives."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_romav2_colnomic_visibility_xf_six_case_v1 as core  # noqa: E402
import run_routea_d1_current_runtime_bridge_e0_v1 as e0  # noqa: E402
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as frozen  # noqa: E402


CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md"
ARM_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_prejoin_v1"
ARM_VALIDATION = ARM_ROOT / "validation.json"
BASE_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
BASE_VALIDATION = BASE_ROOT / "validation.json"
REFERENCE_ROOT = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1"
REFERENCE_PAYLOAD = REFERENCE_ROOT / "payload.pt"
REFERENCE_VALIDATION = REFERENCE_ROOT / "independent_validation.json"
RAW_C_ROOT = ROOT / "results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1"
RAW_C_VALIDATION = RAW_C_ROOT / "validation.json"
FROZEN_GATE_VALIDATION = ROOT / "results/romav2_colnomic_frozen_gate_definition_v1/validation.json"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_fullnegative_features_v1"
PRODUCER_FILE = Path(__file__).resolve()
VALIDATOR_FILE = ROOT / "programs/validate_routea_matched_three_arm_fullnegative_features_shard_v1.py"
AGGREGATE_FILE = ROOT / "programs/validate_routea_matched_three_arm_fullnegative_features_aggregate_v1.py"
CORE_FILE = ROOT / "programs/run_romav2_colnomic_visibility_xf_six_case_v1.py"
LAUNCHER_FILE = ROOT / "slurm/routea_matched_three_arm_fullnegative_features_v1_10m.sbatch"
SHARD_COUNT = 8
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")


def evidence_feature(
    raw_scores: list[float],
    evidence: dict[int, dict[str, float]],
    challenger: int,
    winner: int,
) -> torch.Tensor:
    return frozen.candidate_feature(raw_scores, evidence, challenger, winner)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.shard not in range(SHARD_COUNT):
        raise ValueError("shard must be in 0..7")
    out_dir = OUT_ROOT / f"shard{args.shard:02d}"
    if out_dir.exists():
        raise RuntimeError(f"immutable fullnegative feature shard exists: {out_dir}")
    arm_validation = json.loads(ARM_VALIDATION.read_text())
    base_validation = json.loads(BASE_VALIDATION.read_text())
    reference_validation = json.loads(REFERENCE_VALIDATION.read_text())
    raw_c_validation = json.loads(RAW_C_VALIDATION.read_text())
    frozen_validation = json.loads(FROZEN_GATE_VALIDATION.read_text())
    if not (
        arm_validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED"
        and arm_validation.get("logical_sha256") == e0.logical_sha256(arm_validation)
        and all(arm_validation.get("checks", {}).values())
        and base_validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and base_validation.get("logical_sha256") == e0.logical_sha256(base_validation)
        and reference_validation.get("status")
        == "ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED"
        and reference_validation.get("logical_sha256")
        == e0.logical_sha256(reference_validation)
        and raw_c_validation.get("status")
        == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS"
        and frozen_validation.get("status")
        == "ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS"
        and frozen_validation.get("feature_max_abs") == 0.0
        and frozen_validation.get("module_sha256")
        == e0.sha256_file(ROOT / "src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py")
    ):
        raise RuntimeError("fullnegative feature authority drift")
    arm_payload_path = ARM_ROOT / f"shard{args.shard:02d}/payload.pt"
    base_payload_path = BASE_ROOT / f"shard{args.shard:02d}/payload.pt"
    arm_seal = {int(x["shard"]): x for x in arm_validation["shards"]}[args.shard]
    base_seal = {int(x["shard"]): x for x in base_validation["shards"]}[args.shard]
    if not (
        arm_seal["payload_sha256"] == e0.sha256_file(arm_payload_path)
        and base_seal["payload_sha256"] == e0.sha256_file(base_payload_path)
        and reference_validation.get("payload_sha256") == e0.sha256_file(REFERENCE_PAYLOAD)
    ):
        raise RuntimeError("feature input payload seal drift")
    arm_payload = torch.load(arm_payload_path, map_location="cpu", weights_only=False, mmap=True)
    base_payload = torch.load(base_payload_path, map_location="cpu", weights_only=False, mmap=True)
    references = torch.load(REFERENCE_PAYLOAD, map_location="cpu", weights_only=False, mmap=True)["references"]
    raw_c_path = RAW_C_ROOT / f"shard{args.shard:02d}/result.json"
    raw_c_seal = {int(x["shard"]): x for x in raw_c_validation["shards"]}[args.shard]
    if raw_c_seal["sha256"] != e0.sha256_file(raw_c_path):
        raise RuntimeError("RAW+C evidence authority seal drift")
    raw_c_payload = json.loads(raw_c_path.read_text())
    raw_c_by_execution = {int(x["execution_ordinal"]): x for x in raw_c_payload["rows"]}
    base_by_execution = {int(x["execution_ordinal"]): x for x in base_payload["records"]}
    output_records: list[dict] = []
    c_feature_max_abs = 0.0
    for arm_record in arm_payload["records"]:
        execution = int(arm_record["execution_ordinal"])
        base_record = base_by_execution[execution]
        axis = list(map(int, arm_record["raw_candidate_physical_rows"]))
        candidate_by_row = {int(x["physical_row"]): x for x in arm_record["candidates"]}
        old_by_row = {
            int(x["physical_row"]): x
            for x in raw_c_by_execution[execution]["candidates"]
        }
        raw_scores = [float(base_record["raw_full_gallery_scores"][row]) for row in axis]
        winner = max(range(128), key=lambda i: (raw_scores[i], -axis[i]))
        evidence: dict[str, dict[int, dict[str, float]]] = {arm: {} for arm in ARMS}
        for position, row in enumerate(axis):
            candidate = candidate_by_row[row]
            qmap = torch.as_tensor(candidate["query_map"], dtype=torch.float64)
            rmap = torch.as_tensor(candidate["reference_map"], dtype=torch.float64)
            query_tokens = base_record["raw_image_tokens"]
            reference_tokens = references[row]["tokens"]
            ones_q = torch.ones_like(qmap)
            ones_r = torch.ones_like(rmap)
            a_real = float(candidate["raw_arm_scores"]["A_ALL"])
            a_replay = float(core.score(query_tokens, reference_tokens, ones_q, ones_r)[0])
            b_real = float(candidate["raw_arm_scores"]["B_QUERY"])
            b_replay_value, b_mass_value, _ = core.score(
                query_tokens, reference_tokens, qmap, ones_r
            )
            b_replay = float(b_replay_value)
            b_mass = float(b_mass_value)
            half_q = max(1, qmap.numel() // 2)
            b_qctrl = float(
                core.score(query_tokens, reference_tokens, qmap.roll(half_q), ones_r)[0]
            )
            old = old_by_row[row]
            c_real, c_mass, _ = core.score(
                query_tokens, reference_tokens, qmap, rmap
            )
            c_qctrl = float(
                core.score(
                    query_tokens,
                    reference_tokens,
                    qmap.roll(half_q),
                    rmap,
                )[0]
            )
            c_rctrl = float(
                core.score(
                    query_tokens,
                    reference_tokens,
                    qmap,
                    rmap.roll(max(1, rmap.numel() // 2)),
                )[0]
            )
            if not (
                abs(a_real - a_replay) <= 1e-12
                and abs(b_real - b_replay) <= 1e-12
                and candidate["query_map_sha256"] == old["query_map_sha256"]
                and candidate["reference_map_sha256"] == old["reference_map_sha256"]
                and abs(float(candidate["raw_arm_scores"]["C_PAIRED"]) - float(old["real_score"])) <= 1e-12
                and float(c_real) == float(old["real_score"])
                and float(c_mass) == float(old["visibility_mass"])
                and c_qctrl == float(old["query_control_score"])
                and c_rctrl == float(old["reference_control_score"])
            ):
                raise RuntimeError("fullnegative arm evidence replay drift")
            evidence["A_ALL"][position] = {
                "real_score": a_replay,
                "visibility_mass": 1.0,
                "query_control_score": a_replay,
                "reference_control_score": a_replay,
            }
            evidence["B_QUERY"][position] = {
                "real_score": b_replay,
                "visibility_mass": b_mass,
                "query_control_score": b_qctrl,
                "reference_control_score": b_replay,
            }
            evidence["C_PAIRED"][position] = {
                "real_score": float(c_real),
                "visibility_mass": float(c_mass),
                "query_control_score": c_qctrl,
                "reference_control_score": c_rctrl,
            }
        challengers = [position for position in range(128) if position != winner]
        real_common: dict[str, torch.Tensor] = {}
        real_native: dict[str, torch.Tensor] = {}
        cbind_common: dict[str, torch.Tensor] = {}
        cbind_native: dict[str, torch.Tensor] = {}
        cbind_source_positions = [(position + 64) % 128 for position in range(128)]
        for arm in ARMS:
            native = torch.stack(
                [evidence_feature(raw_scores, evidence[arm], c, winner) for c in challengers]
            )
            controlled_evidence = {
                destination: evidence[arm][source]
                for destination, source in enumerate(cbind_source_positions)
            }
            controlled = torch.stack(
                [
                    evidence_feature(raw_scores, controlled_evidence, c, winner)
                    for c in challengers
                ]
            )
            real_native[arm] = native
            real_common[arm] = native[:, :2].contiguous()
            cbind_native[arm] = controlled
            cbind_common[arm] = controlled[:, :2].contiguous()
        old_candidates = {
            int(x["candidate_position"]): x
            for x in raw_c_by_execution[execution]["candidates"]
        }
        old_raw_scores = [
            float(old_candidates[position]["raw_score"]) for position in range(128)
        ]
        if raw_scores != old_raw_scores:
            raise RuntimeError("current RAW vector no longer exactly replays frozen C")
        for row_index, challenger in enumerate(challengers):
            expected = frozen.candidate_feature(
                old_raw_scores, old_candidates, challenger, winner
            )
            c_feature_max_abs = max(
                c_feature_max_abs,
                float((real_native["C_PAIRED"][row_index] - expected).abs().max()),
            )
        output_records.append(
            {
                "data_split_role": arm_record["role"],
                "execution_ordinal": execution,
                "query_id": arm_record["query_id"],
                "track": arm_record["track"],
                "heldout_fold": int(arm_record["heldout_fold"]),
                "candidate_physical_rows": axis,
                "base_scores": torch.tensor(raw_scores, dtype=torch.float64),
                "base_winner_position": winner,
                "challenger_positions": challengers,
                "cbind_source_positions": cbind_source_positions,
                "evidence": evidence,
                "real_common_features": real_common,
                "real_native_features": real_native,
                "cbind_common_features": cbind_common,
                "cbind_native_features": cbind_native,
                "feature_sha256": {
                    "real_common": {
                        arm: e0.tensor_sha256(real_common[arm]) for arm in ARMS
                    },
                    "real_native": {
                        arm: e0.tensor_sha256(real_native[arm]) for arm in ARMS
                    },
                    "cbind_common": {
                        arm: e0.tensor_sha256(cbind_common[arm]) for arm in ARMS
                    },
                    "cbind_native": {
                        arm: e0.tensor_sha256(cbind_native[arm]) for arm in ARMS
                    },
                },
                "target_role_read_count": 0,
                "model_update_count": 0,
            }
        )
        print(json.dumps({"event":"fullnegative_feature_query_ready","shard":args.shard,"execution":execution},sort_keys=True),flush=True)
    if c_feature_max_abs != 0.0:
        raise RuntimeError("frozen C fullnegative feature replay is not bit-exact")
    value = {
        "schema_version": "routea_matched_three_arm_fullnegative_feature_shard_v1_20260902",
        "status": "ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_READY",
        "claim_level": "TARGET_FREE_RAW_FEATURE_LEDGER_ONLY",
        "shard": args.shard,
        "shard_count": SHARD_COUNT,
        "arms": list(ARMS),
        "records": output_records,
        "c_feature_max_abs": c_feature_max_abs,
        "bindings": {
            "contract_sha256": e0.sha256_file(CONTRACT),
            "arm_payload_sha256": e0.sha256_file(arm_payload_path),
            "arm_validation_sha256": e0.sha256_file(ARM_VALIDATION),
            "base_payload_sha256": e0.sha256_file(base_payload_path),
            "base_validation_sha256": e0.sha256_file(BASE_VALIDATION),
            "reference_payload_sha256": e0.sha256_file(REFERENCE_PAYLOAD),
            "reference_validation_sha256": e0.sha256_file(REFERENCE_VALIDATION),
            "raw_c_authority_sha256": e0.sha256_file(raw_c_path),
            "raw_c_validation_sha256": e0.sha256_file(RAW_C_VALIDATION),
            "frozen_gate_module_sha256": e0.sha256_file(ROOT / "src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py"),
            "frozen_gate_validation_sha256": e0.sha256_file(FROZEN_GATE_VALIDATION),
            "producer_sha256": e0.sha256_file(PRODUCER_FILE),
            "validator_sha256": e0.sha256_file(VALIDATOR_FILE),
            "aggregate_sha256": e0.sha256_file(AGGREGATE_FILE),
            "core_score_sha256": e0.sha256_file(CORE_FILE),
            "launcher_sha256": e0.sha256_file(LAUNCHER_FILE),
        },
        "access": {
            "target_identity_read_count": 0,
            "target_role_read_count": 0,
            "supergroup_read_count": 0,
            "sealed_read_count": 0,
            "model_update_count": 0,
        },
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".fullnegative-feature-{args.shard:02d}-", dir=OUT_ROOT))
    stage_payload = staging / "payload.pt"
    stage_receipt = staging / "receipt.json"
    torch.save(value, stage_payload)
    receipt = {
        "schema_version": "routea_matched_three_arm_fullnegative_feature_receipt_v1_20260902",
        "status": value["status"],
        "shard": args.shard,
        "query_count": len(output_records),
        "c_feature_max_abs": c_feature_max_abs,
        "payload_sha256": e0.sha256_file(stage_payload),
        "target_role_read_count": 0,
        "model_update_count": 0,
        "next_authorized_stage": "FULLNEGATIVE_FEATURE_SHARD_VALIDATION",
        "logical_sha256": "",
    }
    receipt["logical_sha256"] = e0.logical_sha256(receipt)
    try:
        stage_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.rename(staging, out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
