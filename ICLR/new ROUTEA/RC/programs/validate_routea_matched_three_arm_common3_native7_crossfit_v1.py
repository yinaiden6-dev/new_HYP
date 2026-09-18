#!/usr/bin/env python3
"""Independent full replay for the final matched A/B/C cross-fit.

This module deliberately does not import the training consumer.  It independently
validates inputs, joins labels on the current C128 axis, retrains six heads, and
replays REAL/C_BIND actions and all decision metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as frozen
from rc_aslo_xf.conditional_rep_sources import build_gallery_source

PAIR_ROOT = ROOT / "results/routea_matched_three_arm_pair64_training_features_v2"
PAIR = PAIR_ROOT / "payload.pt"
PAIR_RECEIPT = PAIR_ROOT / "receipt.json"
PAIR_LINEAGE = PAIR_ROOT / "post_full_lineage.json"
PAIR_VALID = PAIR_ROOT / "independent_validation.json"
PAIR_WRAPPER = ROOT / "programs/materialize_routea_matched_three_arm_pair64_training_features_v2.py"
PAIR_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_pair64_training_features_v2.py"
PAIR_LAUNCHER = ROOT / "slurm/routea_matched_three_arm_pair64_training_features_v2_10m.sbatch"
PAIR_V1_ROOT = ROOT / "results/routea_matched_three_arm_pair64_training_features_v1"
PAIR_V1_PAYLOAD = PAIR_V1_ROOT / "payload.pt"
PAIR_V1_VALID = PAIR_V1_ROOT / "independent_validation.json"
PAIR_V1_PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_pair64_training_features_v1.py"
PAIR_V1_VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_pair64_training_features_v1.py"
FULL = ROOT / "results/routea_matched_three_arm_fullnegative_features_v1"
FULL_VALID = FULL / "validation.json"
ROLES = ROOT / "results/cw0_rgh_xf_v2_p0_a0_manifest_v2"
ROLE_MANIFEST = ROLES / "role_manifest.json"
ROLE_VALID = ROLES / "independent_validation.json"
FROZEN_MODULE = ROOT / "src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py"
FROZEN_HEAD = ROOT / "results/romav2_colnomic_full_negative_action_gate_v1/result.json"
FROZEN_HEAD_VALID = ROOT / "results/romav2_colnomic_full_negative_action_gate_v1/independent_validation.json"
FROZEN_DEFINITION_VALID = ROOT / "results/romav2_colnomic_frozen_gate_definition_v1/validation.json"
FROZEN_RUNTIME = ROOT / "results/romav2_colnomic_current_runtime_frozen_gate_v1/result.json"
FROZEN_RUNTIME_VALID = ROOT / "results/romav2_colnomic_current_runtime_frozen_gate_v1/independent_validation.json"
PRIMARY = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md"
RUNTIME_ADDENDUM = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_CURRENT_RUNTIME_REGRESSION_ADDENDUM_V1_20260902.md"
PAIR_ADDENDUM = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_ADDENDUM_V1_20260902.md"
FINAL_ADDENDUM = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_FINAL_EXECUTION_AUDIT_ADDENDUM_V1_20260902.md"
TRAINER = ROOT / "programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py"
VALIDATOR = Path(__file__).resolve()
LAUNCHER = ROOT / "slurm/routea_matched_three_arm_common3_native7_crossfit_v1_10m.sbatch"
RESULT = ROOT / "results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json"
OUT = RESULT.parent / "independent_validation.json"

PRIMARY_SHA256 = "53b1d4f3b42a6f44b1e2242de13849b3e26bc4be3efb4e2c5ac45d1d11b4449e"
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
COMMON_NAMES = ("standardized_raw_gap", "symmetric_local_score")
NATIVE_NAMES = tuple(frozen.FEATURE_NAMES)
FAMILIES = {"COMMON3": ("real_common_features", "cbind_common_features", COMMON_NAMES),
            "NATIVE7": ("real_native_features", "cbind_native_features", NATIVE_NAMES)}
STEPS = 2000
FROZEN_EXPECTED = {"query_count": 32, "base_top1": 25, "final_top1": 27,
    "base_R@1": 25 / 32, "final_R@1": 27 / 32,
    "base_MRR": 0.8281994047619048, "final_MRR": 0.8779761904761905,
    "rescue": 2, "break": 0, "retained_correct": 25, "retained_wrong": 5,
    "wrong_to_wrong": 1, "switch_count": 3, "hold_count": 29}


class ValidationError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024): digest.update(block)
    return digest.hexdigest()


def logical(value: dict) -> str:
    body = {key: item for key, item in value.items() if key != "logical_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent); tmp = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally: tmp.unlink(missing_ok=True)


def keys(value: object, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected: raise ValidationError(f"{name} schema drift")


def finite_tensor(value: object, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or tuple(value.shape) != shape:
        raise ValidationError(f"{name} dtype/shape drift")
    if not bool(torch.isfinite(value).all()): raise ValidationError(f"{name} non-finite")
    return value


def finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{name} non-finite")
    return float(value)


def parameter_sha(weight: torch.Tensor, bias: float) -> str:
    body = {"weight": [float(x) for x in weight.detach().cpu().flatten()], "bias": float(bias)}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_logical(value: dict, status: str, name: str) -> None:
    if value.get("status") != status or value.get("logical_sha256") != logical(value):
        raise ValidationError(f"{name} envelope drift")


def validate_pair(fullv: dict) -> tuple[dict, dict, dict, dict]:
    pairv = json.loads(PAIR_VALID.read_text()); lineage = json.loads(PAIR_LINEAGE.read_text())
    receipt = json.loads(PAIR_RECEIPT.read_text()); v1v = json.loads(PAIR_V1_VALID.read_text())
    keys(pairv, {"checks", "full_validation_logical_sha256", "full_validation_sha256", "logical_sha256",
        "model_update_count", "next_authorized_stage", "schema_version", "status",
        "v1_core_validation_logical_sha256", "v2_lineage_logical_sha256", "v2_lineage_sha256",
        "v2_payload_sha256"}, "Pair64 V2 validation")
    valid_logical(pairv, "ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED", "Pair64 V2 validation")
    if not (set(pairv["checks"]) == {"bindings", "core", "full_predecessor", "lineage", "semantic_bit_exact", "v1_authority"}
        and all(x is True for x in pairv["checks"].values()) and pairv["model_update_count"] == 0
        and pairv["next_authorized_stage"] == "MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_TRAINING"
        and pairv["v2_payload_sha256"] == sha(PAIR) and pairv["v2_lineage_sha256"] == sha(PAIR_LINEAGE)
        and pairv["full_validation_sha256"] == sha(FULL_VALID)
        and pairv["full_validation_logical_sha256"] == fullv["logical_sha256"]):
        raise ValidationError("Pair64 V2 validator seal drift")
    keys(lineage, {"schema_version", "status", "v1_payload_sha256", "v1_validation_sha256",
        "v1_validation_logical_sha256", "v2_payload_sha256", "semantic_bit_exact", "full_validation_sha256",
        "full_validation_logical_sha256", "full_validation_mtime_ns", "v1_producer_sha256", "v1_validator_sha256",
        "v2_wrapper_sha256", "v2_validator_sha256", "v2_launcher_sha256", "primary_contract_sha256",
        "pair_addendum_sha256", "runtime_addendum_sha256", "slurm_job_id", "hostname", "gpu_name",
        "materialization_started_unix_ns", "materialization_finished_unix_ns", "model_update_count",
        "next_authorized_stage", "logical_sha256"}, "Pair64 V2 lineage")
    valid_logical(lineage, "ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_POST_FULL_READY", "Pair64 V2 lineage")
    if not (lineage["semantic_bit_exact"] is True and lineage["v2_payload_sha256"] == sha(PAIR)
        and lineage["v1_payload_sha256"] == sha(PAIR_V1_PAYLOAD)
        and lineage["v1_validation_sha256"] == sha(PAIR_V1_VALID)
        and lineage["v1_validation_logical_sha256"] == v1v["logical_sha256"]
        and lineage["v1_producer_sha256"] == sha(PAIR_V1_PRODUCER)
        and lineage["v1_validator_sha256"] == sha(PAIR_V1_VALIDATOR)
        and lineage["full_validation_sha256"] == sha(FULL_VALID)
        and lineage["full_validation_logical_sha256"] == fullv["logical_sha256"]
        and lineage["full_validation_mtime_ns"] == FULL_VALID.stat().st_mtime_ns
        and int(lineage["materialization_started_unix_ns"]) > FULL_VALID.stat().st_mtime_ns
        and int(lineage["materialization_finished_unix_ns"]) >= int(lineage["materialization_started_unix_ns"])
        and lineage["v2_wrapper_sha256"] == sha(PAIR_WRAPPER) and lineage["v2_validator_sha256"] == sha(PAIR_VALIDATOR)
        and lineage["v2_launcher_sha256"] == sha(PAIR_LAUNCHER) and lineage["primary_contract_sha256"] == sha(PRIMARY)
        and lineage["pair_addendum_sha256"] == sha(PAIR_ADDENDUM)
        and lineage["runtime_addendum_sha256"] == sha(RUNTIME_ADDENDUM)
        and lineage["model_update_count"] == 0 and lineage["next_authorized_stage"] == "PAIR64_V2_INDEPENDENT_VALIDATION"):
        raise ValidationError("Pair64 V2 lineage drift")
    keys(receipt, {"c_feature_max_abs", "claim_level", "eval_candidate_generation_authorized",
        "fixed_candidate_count", "logical_sha256", "model_update_count", "next_authorized_stage",
        "payload_sha256", "query_count", "schema_version", "status", "switch_label_count"}, "Pair64 receipt")
    valid_logical(receipt, "ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_READY", "Pair64 receipt")
    if not (receipt["payload_sha256"] == sha(PAIR) and receipt["query_count"] == receipt["switch_label_count"] == 64
        and receipt["fixed_candidate_count"] == 128 and receipt["c_feature_max_abs"] == 0.0
        and receipt["model_update_count"] == 0 and receipt["eval_candidate_generation_authorized"] is False):
        raise ValidationError("Pair64 receipt drift")
    payload = torch.load(PAIR, map_location="cpu", weights_only=False, mmap=True)
    keys(payload, {"schema_version", "status", "claim_level", "arms", "feature_names", "control_conventions",
        "records", "population", "c_exact_replay", "bindings", "usage_policy", "access"}, "Pair64 payload")
    if not (payload["status"] == "ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_READY"
        and payload["arms"] == list(ARMS) and len(payload["records"]) == 64
        and payload["population"]["positive_switch_label_count"] == payload["population"]["negative_switch_label_count"] == 32
        and payload["c_exact_replay"] == {"candidate_count": 128, "feature_count": 64, "feature_max_abs": 0.0,
            "map_hash_mismatch_count": 0, "scalar_mismatch_count": 0}
        and payload["access"]["eval_candidate_generation_count"] == payload["access"]["model_update_count"] == 0):
        raise ValidationError("Pair64 payload drift")
    record_keys = {"pair_cohort", "pair_row_ordinal", "execution_ordinal", "query_id", "inner_fold",
        "candidate_physical_rows", "base_scores", "base_winner_position", "challenger_positions", "switch_label",
        "evidence", "derived_evidence", "fixed_candidate_maps", "real_common_features", "real_native_features",
        "c_frozen_candidate_feature", "c_feature_exact", "pair_role_output_count", "target_identity_output_count",
        "eval_candidate_generation_count", "model_update_count"}
    executions = set()
    for record in payload["records"]:
        keys(record, record_keys, "Pair64 record")
        if not (isinstance(record["switch_label"], bool)
            and len(record["candidate_physical_rows"]) == len(set(record["candidate_physical_rows"])) == 128
            and len(record["challenger_positions"]) == 1
            and record["challenger_positions"][0] != record["base_winner_position"]
            and record["pair_role_output_count"] == record["target_identity_output_count"] == 0
            and record["eval_candidate_generation_count"] == record["model_update_count"] == 0
            and record["c_feature_exact"] is True): raise ValidationError("Pair64 record drift")
        finite_tensor(record["base_scores"], (128,), "Pair64 base")
        for family, (real_key, _, names) in FAMILIES.items():
            for arm in ARMS: finite_tensor(record[real_key][arm], (1, len(names)), f"Pair64 {family}/{arm}")
        if not torch.equal(record["real_native_features"]["C_PAIRED"], record["c_frozen_candidate_feature"]):
            raise ValidationError("Pair64 frozen feature drift")
        execution = int(record["execution_ordinal"])
        if execution in executions: raise ValidationError("Pair64 duplicate execution")
        executions.add(execution)
    return payload, pairv, lineage, receipt


def validate_frozen() -> tuple[dict, dict, dict]:
    definition = json.loads(FROZEN_DEFINITION_VALID.read_text()); runtime = json.loads(FROZEN_RUNTIME.read_text())
    runtimev = json.loads(FROZEN_RUNTIME_VALID.read_text()); head = json.loads(FROZEN_HEAD.read_text())
    headv = json.loads(FROZEN_HEAD_VALID.read_text())
    keys(definition, {"bias", "checks", "feature_comparison_count", "feature_max_abs", "feature_names",
        "head_sha256", "module_sha256", "parameter_count", "schema_version", "status", "switch_threshold", "weight"},
        "frozen definition")
    if not (definition["status"] == "ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS"
        and set(definition["checks"]) == {"all_features_bit_exact", "feature_names", "threshold", "weights"}
        and all(x is True for x in definition["checks"].values()) and definition["feature_comparison_count"] == 8128
        and definition["feature_max_abs"] == 0.0 and definition["module_sha256"] == sha(FROZEN_MODULE)
        and definition["head_sha256"] == sha(FROZEN_HEAD) and definition["feature_names"] == list(NATIVE_NAMES)
        and definition["weight"] == frozen.WEIGHT.tolist() and definition["bias"] == frozen.BIAS
        and definition["parameter_count"] == frozen.PARAMETER_COUNT == 7
        and definition["switch_threshold"] == frozen.SWITCH_THRESHOLD == 0.0): raise ValidationError("frozen definition drift")
    if not (head.get("logical_sha256") == logical(head) and head.get("parameter_count") == 7
        and head.get("head", {}).get("weight") == frozen.WEIGHT.tolist() and head.get("head", {}).get("bias") == frozen.BIAS
        and head.get("sealed_read_count") == 0): raise ValidationError("frozen source head drift")
    if not (headv.get("status") == "ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_INDEPENDENT_VALIDATION_PASS"
        and headv.get("logical_sha256") == logical(headv)
        and set(headv.get("checks", {})) == {"action_recompute", "boundary", "decision", "envelope", "fold_isolation", "parameter_ledger", "sources"}
        and all(x is True for x in headv["checks"].values()) and headv.get("result_sha256") == sha(FROZEN_HEAD)
        and headv.get("scientific_GO_or_NO_GO") is None and headv.get("sealed_data_read_authorized") is False):
        raise ValidationError("frozen source head validation drift")
    valid_logical(runtime, "ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_GO", "frozen runtime")
    if not (runtime["frozen_head_sha256"] == sha(FROZEN_HEAD) and runtime["frozen_head_validation_sha256"] == sha(FROZEN_HEAD_VALID)
        and runtime["parameter_count"] == 7 and runtime["model_update_count"] == runtime["sealed_read_count"] == 0
        and runtime["scientific_GO_or_NO_GO"] is None and len(runtime["actions"]) == 32): raise ValidationError("frozen runtime drift")
    if not (runtimev.get("status") == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_PASS"
        and set(runtimev.get("checks", {})) == {"actions_exact", "frozen_head", "gates_exact", "prejoin_validation",
            "result_envelope", "sealed_zero", "summary_exact"} and all(x is True for x in runtimev["checks"].values())
        and runtimev["result_sha256"] == sha(FROZEN_RUNTIME) and runtimev["sealed_read_count"] == 0):
        raise ValidationError("frozen runtime validation drift")
    finite_tensor(frozen.WEIGHT, (6,), "frozen weight"); finite_scalar(frozen.BIAS, "frozen bias")
    return definition, runtime, runtimev


def sym(a: float, b: float) -> float:
    return (float(a) - float(b)) / (abs(float(a)) + abs(float(b)) + 1e-12)


def reconstruct_feature(raw: list[float], evidence: dict[int, dict[str, float]], challenger: int, winner: int) -> torch.Tensor:
    mean = sum(map(float, raw)) / len(raw)
    std = (sum((float(value) - mean) ** 2 for value in raw) / len(raw)) ** 0.5
    c, w = evidence[challenger], evidence[winner]
    cn = float(c["real_score"]) / max(float(c["visibility_mass"]), 1e-12)
    wn = float(w["real_score"]) / max(float(w["visibility_mass"]), 1e-12)
    cq, wq = float(c["real_score"]) - float(c["query_control_score"]), float(w["real_score"]) - float(w["query_control_score"])
    cr, wr = float(c["real_score"]) - float(c["reference_control_score"]), float(w["real_score"]) - float(w["reference_control_score"])
    return torch.tensor([(float(raw[challenger]) - float(raw[winner])) / max(std, 1e-12),
        sym(c["real_score"], w["real_score"]), sym(c["visibility_mass"], w["visibility_mass"]),
        sym(cn, wn), sym(cq, wq), sym(cr, wr)], dtype=torch.float64)


def validate_full() -> tuple[list[dict], dict, list[dict]]:
    aggregate = json.loads(FULL_VALID.read_text())
    keys(aggregate, {"checks", "contract_sha256", "heldout_folds", "logical_sha256", "model_update_count",
        "next_authorized_stage", "query_count", "roles", "schema_version", "shards", "status",
        "target_role_read_count", "tracks"}, "full aggregate")
    valid_logical(aggregate, "ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_VALIDATED", "full aggregate")
    if not (set(aggregate["checks"]) == {"all_shards", "c_replay", "folds", "population", "roles", "target_free", "tracks"}
        and all(x is True for x in aggregate["checks"].values()) and aggregate["contract_sha256"] == PRIMARY_SHA256
        and aggregate["query_count"] == 64 and aggregate["roles"] == {"EVAL": 32, "TRAIN": 32}
        and aggregate["target_role_read_count"] == aggregate["model_update_count"] == 0): raise ValidationError("full aggregate drift")
    seals = {int(item["shard"]): item for item in aggregate["shards"]}
    if set(seals) != set(range(8)): raise ValidationError("full shard set drift")
    records, output_seals = [], []
    record_keys = {"data_split_role", "execution_ordinal", "query_id", "track", "heldout_fold",
        "candidate_physical_rows", "base_scores", "base_winner_position", "challenger_positions", "cbind_source_positions",
        "evidence", "real_common_features", "real_native_features", "cbind_common_features", "cbind_native_features",
        "feature_sha256", "target_role_read_count", "model_update_count"}
    for shard in range(8):
        directory = FULL / f"shard{shard:02d}"; pp, rp, vp = directory / "payload.pt", directory / "receipt.json", directory / "validation.json"
        seal = {"shard": shard, "payload_sha256": sha(pp), "receipt_sha256": sha(rp), "validation_sha256": sha(vp)}
        if seals[shard] != seal: raise ValidationError("full shard physical seal drift")
        receipt, validation = json.loads(rp.read_text()), json.loads(vp.read_text())
        valid_logical(receipt, "ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_READY", "full receipt")
        valid_logical(validation, "ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_VALIDATED", "full validation")
        if not (receipt["shard"] == validation["shard"] == shard and receipt["query_count"] == validation["query_count"] == 8
            and receipt["payload_sha256"] == validation["payload_sha256"] == seal["payload_sha256"]
            and validation["receipt_sha256"] == seal["receipt_sha256"]
            and set(validation["checks"]) == {"access", "authorities", "bindings", "envelope", "receipt", "records"}
            and all(x is True for x in validation["checks"].values())
            and receipt["target_role_read_count"] == validation["target_role_read_count"] == 0
            and receipt["model_update_count"] == validation["model_update_count"] == 0): raise ValidationError("full shard receipt drift")
        payload = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
        keys(payload, {"schema_version", "status", "claim_level", "shard", "shard_count", "arms", "records",
            "c_feature_max_abs", "bindings", "access"}, "full payload")
        if not (payload["status"] == "ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_READY"
            and payload["shard"] == shard and payload["shard_count"] == 8 and payload["arms"] == list(ARMS)
            and payload["c_feature_max_abs"] == 0.0 and len(payload["records"]) == 8
            and payload["access"] == {"target_identity_read_count": 0, "target_role_read_count": 0,
                "supergroup_read_count": 0, "sealed_read_count": 0, "model_update_count": 0}): raise ValidationError("full payload drift")
        for record in payload["records"]:
            keys(record, record_keys, "full record"); axis = [int(x) for x in record["candidate_physical_rows"]]
            base = finite_tensor(record["base_scores"], (128,), "full base"); winner = int(record["base_winner_position"])
            challengers = [int(x) for x in record["challenger_positions"]]
            if not (record["data_split_role"] in {"TRAIN", "EVAL"} and len(axis) == len(set(axis)) == 128
                and winner == max(range(128), key=lambda i: (float(base[i]), -axis[i]))
                and len(challengers) == len(set(challengers)) == 127 and set(challengers) == set(range(128)) - {winner}
                and record["cbind_source_positions"] == [(i + 64) % 128 for i in range(128)]
                and record["target_role_read_count"] == record["model_update_count"] == 0): raise ValidationError("full record axis drift")
            for family, (real_key, ctrl_key, names) in FAMILIES.items():
                for arm in ARMS:
                    real = finite_tensor(record[real_key][arm], (127, len(names)), f"full {family}/{arm} REAL")
                    ctrl = finite_tensor(record[ctrl_key][arm], (127, len(names)), f"full {family}/{arm} C_BIND")
                    if not torch.equal(real[:, 0], ctrl[:, 0]): raise ValidationError("C_BIND altered RAW gap")
                    prefix = "common" if family == "COMMON3" else "native"
                    if record["feature_sha256"][f"real_{prefix}"][arm] != tensor_sha(real): raise ValidationError("REAL hash drift")
                    if record["feature_sha256"][f"cbind_{prefix}"][arm] != tensor_sha(ctrl): raise ValidationError("C_BIND hash drift")
            raw = base.tolist()
            for arm in ARMS:
                evidence = record["evidence"][arm]
                expected_real_native = torch.stack([reconstruct_feature(raw, evidence, challenger, winner) for challenger in challengers])
                controlled = {destination: evidence[source] for destination, source in enumerate(record["cbind_source_positions"])}
                expected_cbind_native = torch.stack([reconstruct_feature(raw, controlled, challenger, winner) for challenger in challengers])
                if not (torch.equal(expected_real_native, record["real_native_features"][arm])
                    and torch.equal(expected_cbind_native, record["cbind_native_features"][arm])
                    and torch.equal(expected_real_native[:, :2].contiguous(), record["real_common_features"][arm])
                    and torch.equal(expected_cbind_native[:, :2].contiguous(), record["cbind_common_features"][arm])):
                    raise ValidationError("independent REAL/C_BIND feature formula replay drift")
            records.append(record)
        output_seals.append(seal)
    if len(records) != 64 or len({int(x["execution_ordinal"]) for x in records}) != 64: raise ValidationError("full population drift")
    return records, aggregate, output_seals


def gallery_seal(gallery) -> dict:
    return {"physical_row_count": len(gallery.raw_paths), "corrected_identity_count": len(set(gallery.corrected_identities)),
        "gallery_cache_sha256": gallery.gallery_cache_sha256, "raw_path_sequence_sha256": gallery.raw_path_sequence_sha256,
        "legacy_setid_sequence_sha256": gallery.legacy_setid_sequence_sha256,
        "corrected_mapping_sha256": gallery.corrected_mapping_sha256,
        "repair_contract_sha256": gallery.repair_contract_sha256,
        "repair_manifest_sha256": gallery.repair_manifest_sha256}


def late_join(records: list[dict]) -> tuple[list[dict], list[dict], dict, dict, list[dict], dict, dict]:
    manifest = json.loads(ROLE_MANIFEST.read_text()); validation = json.loads(ROLE_VALID.read_text())
    valid_logical(manifest, "RGH_P0_A0_ROLE_MANIFEST_READY", "role manifest")
    if not (validation.get("status") == "RGH_P0_A0_MANIFEST_V2_INDEPENDENT_VALIDATION_PASS"
        and validation.get("logical_sha256") == logical(validation)
        and validation.get("query_count") == validation.get("role_receipt_count") == 600
        and validation.get("P0_scientific_reduction_authorized") is False
        and manifest.get("role_shard_count") == 600): raise ValidationError("role authority drift")
    entries = {int(item["execution_ordinal"]): item for item in manifest["shards"]}
    if set(entries) != set(range(600)): raise ValidationError("role execution ledger drift")
    documents, seals = {}, []
    for execution in range(600):
        entry = entries[execution]; path = Path(entry["path"]); role = json.loads(path.read_text())
        if not (sha(path) == entry["sha256"] and role.get("logical_sha256") == entry.get("logical_sha256") == logical(role)
            and role.get("status") == "RGH_P0_A0_ROLE_SHARD_READY" and int(role.get("execution_ordinal", -1)) == execution
            and role.get("target_naturally_present") is entry.get("target_naturally_present")
            and role.get("target_insertion_count") == role.get("target_spatial_supervision_count") == role.get("raw_d1_field_count") == 0):
            raise ValidationError("role shard seal drift")
        documents[execution] = role
        seals.append({"execution_ordinal": execution, "path": str(path), "sha256": entry["sha256"],
                      "logical_sha256": entry["logical_sha256"]})
    gallery = build_gallery_source(verify_cache_file_sha256=True); labels = gallery.corrected_identities
    joined = []
    for record in records:
        role = documents[int(record["execution_ordinal"])]
        if role["query_id"] != record["query_id"] or role["track"] != record["track"]:
            raise ValidationError("current population role drift")
        # Intentionally ignore the legacy role target_candidate_position.  Rejoin the
        # identity on this record's current C128 axis using the corrected gallery.
        identity = str(role["identity"])
        axis_identities = [labels[int(physical)] for physical in record["candidate_physical_rows"]]
        if len(axis_identities) != 128 or len(set(axis_identities)) != 128:
            raise ValidationError("current C128 exact-label axis is not unique")
        positions = [i for i, candidate_identity in enumerate(axis_identities) if candidate_identity == identity]
        if len(positions) != 1: raise ValidationError("current-axis target join not unique")
        joined.append({**record, "target_position": positions[0], "target_identity": identity,
                       "supergroup": str(role["supergroup"])})
    train = sorted((row for row in joined if row["data_split_role"] == "TRAIN"), key=lambda row: int(row["execution_ordinal"]))
    evals = sorted((row for row in joined if row["data_split_role"] == "EVAL"), key=lambda row: int(row["execution_ordinal"]))
    disjoint = {"execution": not bool({x["execution_ordinal"] for x in train} & {x["execution_ordinal"] for x in evals}),
        "identity": not bool({x["target_identity"] for x in train} & {x["target_identity"] for x in evals}),
        "supergroup": not bool({x["supergroup"] for x in train} & {x["supergroup"] for x in evals}),
        "train_identity_count": len({x["target_identity"] for x in train}),
        "eval_identity_count": len({x["target_identity"] for x in evals}),
        "train_supergroup_count": len({x["supergroup"] for x in train}),
        "eval_supergroup_count": len({x["supergroup"] for x in evals})}
    if len(train) != 32 or len(evals) != 32 or not all(disjoint[name] for name in ("execution", "identity", "supergroup")):
        raise ValidationError("TRAIN/EVAL split drift")
    return train, evals, manifest, validation, seals, gallery_seal(gallery), disjoint


def feature_ledger(pair: dict, train: list[dict]) -> dict:
    output = {}
    for family, (real_key, _, names) in FAMILIES.items():
        output[family] = {}
        for arm in ARMS:
            matrix = torch.cat([row[real_key][arm] for row in pair["records"]]
                               + [row[real_key][arm] for row in train], dim=0).to(torch.float64)
            finite_tensor(matrix, (64 + 32 * 127, len(names)), f"{family}/{arm} design")
            zeros = [i for i in range(len(names)) if int(torch.count_nonzero(matrix[:, i])) == 0]
            duplicates = [[i, j] for i in range(len(names)) for j in range(i + 1, len(names))
                          if torch.equal(matrix[:, i], matrix[:, j])]
            with_bias = torch.cat([matrix, torch.ones((matrix.shape[0], 1), dtype=torch.float64)], dim=1)
            output[family][arm] = {"feature_names": list(names), "matrix_shape": list(matrix.shape),
                "pair_row_count": 64, "fullnegative_row_count": 32 * 127,
                "feature_rank": int(torch.linalg.matrix_rank(matrix)),
                "design_with_bias_rank": int(torch.linalg.matrix_rank(with_bias)),
                "exact_zero_column_indices": zeros, "exact_zero_column_names": [names[i] for i in zeros],
                "exact_duplicate_column_pairs": duplicates,
                "exact_duplicate_column_name_pairs": [[names[i], names[j]] for i, j in duplicates], "all_finite": True}
    return output


def retrain(pair: dict, train: list[dict], family: str, arm: str):
    real_key, _, names = FAMILIES[family]
    px = torch.cat([row[real_key][arm] for row in pair["records"]], dim=0).to(torch.float64)
    py = torch.tensor([float(row["switch_label"]) for row in pair["records"]], dtype=torch.float64)
    weights = torch.where(py.eq(0), torch.full_like(py, 4.0), torch.ones_like(py))
    finite_tensor(px, (64, len(names)), "pair design"); finite_tensor(py, (64,), "pair labels")
    torch.manual_seed(17); head = nn.Linear(len(names), 1, dtype=torch.float64)
    head.weight.data.zero_(); head.bias.data.zero_()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=1e-3)
    loss_steps = gradient_steps = 0; parameter_states = 1; final = (0.0, 0.0, 0.0)
    for _ in range(STEPS):
        optimizer.zero_grad(); pair_logits = head(px).squeeze(1)
        finite_tensor(pair_logits, (64,), "pair logits")
        pair_loss = (F.binary_cross_entropy_with_logits(pair_logits, py, reduction="none") * weights).mean()
        query_losses = []
        for row in train:
            values = head(row[real_key][arm]).squeeze(1); finite_tensor(values, (127,), "full-negative logits")
            winner, target = int(row["base_winner_position"]), int(row["target_position"])
            challengers = [int(x) for x in row["challenger_positions"]]
            if target == winner: query_losses.append(4 * F.softplus(values.max()))
            else:
                position = challengers.index(target); mask = torch.ones(127, dtype=torch.bool); mask[position] = False
                query_losses.append(F.softplus(-values[position]) + 4 * F.softplus(values[mask].max()))
        full_loss = torch.stack(query_losses).mean(); loss = pair_loss + full_loss
        if not bool(torch.isfinite(torch.stack([loss, pair_loss, full_loss])).all()): raise ValidationError("non-finite loss")
        loss_steps += 1; loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in head.parameters()):
            raise ValidationError("non-finite gradient")
        gradient_steps += 1; optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in head.parameters()): raise ValidationError("non-finite parameter")
        parameter_states += 1; final = (float(loss.detach()), float(pair_loss.detach()), float(full_loss.detach()))
    finite = {"input_finite": True, "finite_loss_step_count": loss_steps,
        "finite_gradient_step_count": gradient_steps, "finite_parameter_state_count": parameter_states,
        "all_finite": loss_steps == gradient_steps == STEPS and parameter_states == STEPS + 1}
    return head, final, finite


def actions(weight: torch.Tensor, bias: float, rows: list[dict], family: str, arm: str, control: bool = False) -> list[dict]:
    finite_tensor(weight, (len(FAMILIES[family][2]),), "action weight"); finite_scalar(bias, "action bias")
    real_key, ctrl_key, _ = FAMILIES[family]; key = ctrl_key if control else real_key; output = []
    for row in rows:
        axis = [int(x) for x in row["candidate_physical_rows"]]; base = row["base_scores"]
        winner = int(row["base_winner_position"]); challengers = [int(x) for x in row["challenger_positions"]]
        logits = row[key][arm] @ weight + float(bias); finite_tensor(logits, (127,), "action logits")
        index = max(range(127), key=lambda i: (float(logits[i]), -axis[challengers[i]]))
        challenger = challengers[index]; switch_logit = float(logits[index])
        decision = "SWITCH" if switch_logit > 0.0 else "HOLD"; final = challenger if decision == "SWITCH" else winner
        target = int(row["target_position"]); base_order = sorted(range(128), key=lambda i: (-float(base[i]), axis[i]))
        if base_order[0] != winner: raise ValidationError("base ranking tie drift")
        final_order = list(base_order)
        if decision == "SWITCH": final_order.remove(challenger); final_order.insert(0, challenger)
        base_correct, final_correct = winner == target, final == target
        output.append({"execution_ordinal": int(row["execution_ordinal"]), "query_id": row["query_id"],
            "track": row["track"], "heldout_fold": int(row["heldout_fold"]), "base_winner": winner,
            "base_winner_physical_row": axis[winner], "target_position": target, "target_physical_row": axis[target],
            "base_target_rank": base_order.index(target) + 1, "final_target_rank": final_order.index(target) + 1,
            "proposed_challenger": challenger, "proposed_challenger_physical_row": axis[challenger],
            "proposed_challenger_base_rank": base_order.index(challenger) + 1, "switch_logit": switch_logit,
            "decision": decision, "final_position": final, "final_physical_row": axis[final],
            "base_correct": base_correct, "final_correct": final_correct,
            "wrong_to_wrong": (not base_correct) and (not final_correct) and final != winner})
    return output


def summary(rows: list[dict]) -> dict:
    n = len(rows); base = sum(row["base_correct"] for row in rows); final = sum(row["final_correct"] for row in rows)
    return {"query_count": n, "base_top1": base, "final_top1": final, "base_R@1": base / n,
        "final_R@1": final / n, "base_MRR": sum(1 / row["base_target_rank"] for row in rows) / n,
        "final_MRR": sum(1 / row["final_target_rank"] for row in rows) / n,
        "rescue": sum((not row["base_correct"]) and row["final_correct"] for row in rows),
        "break": sum(row["base_correct"] and not row["final_correct"] for row in rows),
        "retained_correct": sum(row["base_correct"] and row["final_correct"] for row in rows),
        "retained_wrong": sum((not row["base_correct"]) and (not row["final_correct"]) for row in rows),
        "wrong_to_wrong": sum(row["wrong_to_wrong"] for row in rows),
        "switch_count": sum(row["decision"] == "SWITCH" for row in rows),
        "hold_count": sum(row["decision"] == "HOLD" for row in rows)}


def grouped(rows: list[dict], key: str) -> dict:
    return {str(value): summary([row for row in rows if row[key] == value]) for value in sorted({row[key] for row in rows})}


def retention(real: list[dict], control: list[dict]) -> dict:
    by_execution = {row["execution_ordinal"]: row for row in control}
    rescues = [row for row in real if (not row["base_correct"]) and row["final_correct"]]
    kept = sum(by_execution[row["execution_ordinal"]]["final_correct"] for row in rescues)
    return {"real_rescue_count": len(rescues), "cbind_retained_rescue_count": kept,
            "cbind_rescue_retention": kept / len(rescues) if rescues else 1.0}


def replay_runtime(current: list[dict], runtime: dict) -> float:
    historical = {int(row["execution_ordinal"]): row for row in runtime["actions"]}
    if set(historical) != {row["execution_ordinal"] for row in current}: raise ValidationError("runtime execution axis drift")
    fields = ("query_id", "base_winner", "target_position", "proposed_challenger", "decision", "final_position",
              "base_correct", "final_correct"); maximum = 0.0
    for row in current:
        old = historical[row["execution_ordinal"]]
        if any(row[field] != old[field] for field in fields): raise ValidationError("runtime action drift")
        maximum = max(maximum, abs(row["switch_logit"] - old["switch_logit"]))
    return maximum


def bindings(context: dict) -> dict:
    return {"primary_contract_sha256": sha(PRIMARY), "runtime_addendum_sha256": sha(RUNTIME_ADDENDUM),
        "pair_addendum_sha256": sha(PAIR_ADDENDUM), "final_execution_addendum_sha256": sha(FINAL_ADDENDUM),
        "pair_payload_sha256": sha(PAIR), "pair_receipt_sha256": sha(PAIR_RECEIPT),
        "pair_receipt_logical_sha256": context["pair_receipt"]["logical_sha256"],
        "pair_lineage_sha256": sha(PAIR_LINEAGE), "pair_lineage_logical_sha256": context["pair_lineage"]["logical_sha256"],
        "pair_validation_sha256": sha(PAIR_VALID), "pair_validation_logical_sha256": context["pair_validation"]["logical_sha256"],
        "pair_wrapper_sha256": sha(PAIR_WRAPPER), "pair_validator_sha256": sha(PAIR_VALIDATOR),
        "pair_launcher_sha256": sha(PAIR_LAUNCHER), "full_validation_sha256": sha(FULL_VALID),
        "full_validation_logical_sha256": context["full_validation"]["logical_sha256"], "full_shards": context["full_shards"],
        "role_manifest_sha256": sha(ROLE_MANIFEST), "role_manifest_logical_sha256": context["role_manifest"]["logical_sha256"],
        "role_validation_sha256": sha(ROLE_VALID), "role_validation_logical_sha256": context["role_validation"]["logical_sha256"],
        "role_shards": context["role_shards"], "gallery": context["gallery"], "frozen_module_sha256": sha(FROZEN_MODULE),
        "frozen_head_sha256": sha(FROZEN_HEAD), "frozen_head_validation_sha256": sha(FROZEN_HEAD_VALID),
        "frozen_definition_validation_sha256": sha(FROZEN_DEFINITION_VALID),
        "frozen_runtime_result_sha256": sha(FROZEN_RUNTIME),
        "frozen_runtime_result_logical_sha256": context["frozen_runtime"]["logical_sha256"],
        "frozen_runtime_validation_sha256": sha(FROZEN_RUNTIME_VALID), "trainer_sha256": sha(TRAINER),
        "validator_sha256": sha(VALIDATOR), "launcher_sha256": sha(LAUNCHER)}


def main() -> None:
    if OUT.exists(): raise RuntimeError(f"immutable independent validation exists: {OUT}")
    result = json.loads(RESULT.read_text()); checks: dict[str, bool] = {}; reason = None
    try:
        if sha(PRIMARY) != PRIMARY_SHA256: raise ValidationError("primary contract drift")
        records, fullv, full_shards = validate_full()
        pair, pairv, pair_lineage, pair_receipt = validate_pair(fullv)
        frozen_definition, frozen_runtime, frozen_runtimev = validate_frozen()
        train, evals, rolem, rolev, role_shards, gallery, disjoint = late_join(records)
        context = {"pair_validation": pairv, "pair_lineage": pair_lineage, "pair_receipt": pair_receipt,
            "full_validation": fullv, "full_shards": full_shards, "role_manifest": rolem, "role_validation": rolev,
            "role_shards": role_shards, "gallery": gallery, "disjoint": disjoint,
            "frozen_definition": frozen_definition, "frozen_runtime": frozen_runtime,
            "frozen_runtime_validation": frozen_runtimev}
        ledger = feature_ledger(pair, train)
        frozen_real = actions(frozen.WEIGHT, frozen.BIAS, evals, "NATIVE7", "C_PAIRED")
        frozen_cbind = actions(frozen.WEIGHT, frozen.BIAS, evals, "NATIVE7", "C_PAIRED", True)
        frozen_summary, frozen_cbind_summary = summary(frozen_real), summary(frozen_cbind)
        logit_max_abs = 0.0
        for row in evals:
            raw = row["base_scores"].tolist(); winner = int(row["base_winner_position"])
            evidence = row["evidence"]["C_PAIRED"]
            for index, challenger in enumerate(row["challenger_positions"]):
                reconstructed = float(row["real_native_features"]["C_PAIRED"][index] @ frozen.WEIGHT + frozen.BIAS)
                # Independent direct formula; no consumer or frozen.logit helper is called.
                direct_feature = reconstruct_feature(raw, evidence, int(challenger), winner)
                direct = float((frozen.WEIGHT * direct_feature).sum() + frozen.BIAS)
                finite_scalar(reconstructed, "frozen reconstructed logit"); finite_scalar(direct, "frozen direct logit")
                logit_max_abs = max(logit_max_abs, abs(reconstructed - direct))
        runtime_max_abs = replay_runtime(frozen_real, frozen_runtime)
        if not (frozen_summary == FROZEN_EXPECTED and logit_max_abs <= 1e-12 and runtime_max_abs <= 1e-12):
            raise ValidationError("frozen-C regression drift")
        frozen_block = {"pass": True, "expected_summary": FROZEN_EXPECTED, "summary": frozen_summary,
            "cbind_summary": frozen_cbind_summary, "cbind_rescue_retention": retention(frozen_real, frozen_cbind),
            "real_actions": frozen_real, "cbind_actions": frozen_cbind, "logit_max_abs": logit_max_abs,
            "runtime_action_max_abs": runtime_max_abs, "finite": True,
            "authority": {"feature_names": list(NATIVE_NAMES), "weight": frozen.WEIGHT.tolist(), "bias": frozen.BIAS,
                "parameter_count": frozen.PARAMETER_COUNT, "switch_threshold": frozen.SWITCH_THRESHOLD,
                "parameter_sha256": parameter_sha(frozen.WEIGHT, frozen.BIAS)}}
        heads, evaluations = {}, {}
        for family in FAMILIES:
            heads[family], evaluations[family] = {}, {}
            for arm in ARMS:
                head, losses, finite = retrain(pair, train, family, arm)
                weight, bias = head.weight.detach().flatten(), float(head.bias.detach())
                real, control = actions(weight, bias, evals, family, arm), actions(weight, bias, evals, family, arm, True)
                rs, cs, retain = summary(real), summary(control), retention(real, control)
                heads[family][arm] = {"weight": weight.tolist(), "bias": bias, "parameter_count": weight.numel() + 1,
                    "loss_total": losses[0], "loss_pair": losses[1], "loss_fullnegative": losses[2],
                    "parameter_sha256": parameter_sha(weight, bias), "finite_training": finite}
                evaluations[family][arm] = {"real": rs, "cbind": cs, "cbind_rescue_retention": retain,
                    "real_by_track": grouped(real, "track"), "real_by_heldout_fold": grouped(real, "heldout_fold"),
                    "actions": real, "cbind_actions": control, "finite_action_count": len(real),
                    "finite_cbind_action_count": len(control),
                    "actionable": rs["final_top1"] > 25 and rs["rescue"] > rs["break"] and rs["break"] <= 1,
                    "strict_external_candidate": (rs["final_top1"] > FROZEN_EXPECTED["final_top1"]
                        and rs["rescue"] > rs["break"] and rs["break"] <= 1
                        and rs["final_top1"] > cs["final_top1"] and retain["real_rescue_count"] > 0
                        and retain["cbind_rescue_retention"] < 1.0)}
        strict = [{"family": family, "arm": arm, "final_top1": evaluations[family][arm]["real"]["final_top1"]}
                  for family in FAMILIES for arm in ARMS if evaluations[family][arm]["strict_external_candidate"]]
        expected = {"schema_version": "routea_matched_three_arm_common3_native7_crossfit_v2_20260902",
            "status": "ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_COMPLETE",
            "claim_level": "INTERNAL_ARM_SELECTION_CROSSFIT_ONLY", "frozen_c_regression": frozen_block,
            "head_families": {"COMMON3": {"parameter_count": 3, "feature_names": list(COMMON_NAMES),
                    "effective_capacity": "STRICT_COMMON_NONDEGENERATE"},
                "NATIVE7": {"parameter_count": 7, "feature_names": list(NATIVE_NAMES),
                    "effective_capacity": "NOMINAL_MECHANISM_SPECIFIC_WITH_REGISTERED_DEGENERACIES"}},
            "feature_degeneracy_ledger": ledger, "heads": heads, "evaluations": evaluations,
            "train_eval_disjoint": disjoint,
            "action_authority": {"candidate_count": 128, "challenger_count": 127, "switch_threshold": 0.0,
                "base_tie_break": "score_descending_then_physical_row_ascending",
                "challenger_tie_break": "logit_descending_then_physical_row_ascending",
                "hold_transform": "preserve_full_base_order",
                "switch_transform": "move_challenger_to_rank1_shift_preceding_candidates_only",
                "cbind_preserves_destination_base_axis": True, "all_action_logits_finite": True},
            "arm_selection_decision": {"deployment_default": "FROZEN_C", "new_head_replacement_authorized": False,
                "provisional_external_candidates": strict,
                "external_confirmation_design_eligible_after_independent_validation": bool(strict),
                "candidate_binding_destruction_required": True, "tie_with_frozen_is_not_promotion": True},
            "input_finiteness": {"pair_real_features": True, "full_base_scores": True, "full_real_features": True,
                "full_cbind_features": True, "frozen_parameters": True, "all_finite": True},
            "bindings": bindings(context), "target_join_after_feature_seals": True,
            "target_join_authority": {"identity_and_supergroup_source": "HASH_BOUND_ROLE_LEDGER",
                "candidate_axis_source": "CURRENT_RUNTIME_FULLNEGATIVE_C128",
                "gallery_identity_source": "SIX_HASH_CORRECTED_GALLERY_SEAL",
                "legacy_role_candidate_axis_consumed": False,
                "legacy_role_target_candidate_position_consumed": False,
                "unique_current_axis_target_required": True},
            "role_ledger_is_not_scientific_p0_authority": True, "pair_training_query_count": 64,
            "fullnegative_train_query_count": 32, "eval_query_count": 32, "model_update_count": 12000,
            "sealed_read_count": 0, "scientific_GO_or_NO_GO": None,
            "next_authorized_stage": "MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION",
            "logical_sha256": ""}
        expected["logical_sha256"] = logical(expected)
        checks = {"result_exact_schema": set(result) == set(expected),
            "result_logical_hash": result.get("logical_sha256") == logical(result),
            "pair64_v2_post_full_lineage": True, "full_eight_shard_seals_and_formula_replay": True,
            "all_600_role_seals_and_current_axis_six_hash_join": True,
            "feature_rank_zero_duplicate_ledger": result.get("feature_degeneracy_ledger") == ledger,
            "frozen_real_cbind_runtime_and_source_head": result.get("frozen_c_regression") == frozen_block,
            "six_independent_2000_step_finite_retrains": result.get("heads") == heads,
            "real_cbind_actions_ranks_mrr_retention": result.get("evaluations") == evaluations,
            "bindings": result.get("bindings") == expected["bindings"],
            "claim_and_strict_promotion_boundary": (result.get("arm_selection_decision") == expected["arm_selection_decision"]
                and result.get("scientific_GO_or_NO_GO") is None and result.get("new_head_replacement_authorized") is None
                and result.get("next_authorized_stage") == "MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION"),
            "whole_result_exact": result == expected}
    except Exception as error:
        checks = {**checks, "independent_replay_exception_free": False}; passed = False
        reason = f"{type(error).__name__}: {error}"
    else:
        checks["independent_replay_exception_free"] = True; passed = all(value is True for value in checks.values())
    strict_ready = bool(result.get("arm_selection_decision", {}).get("external_confirmation_design_eligible_after_independent_validation")) if passed else False
    value = {"schema_version": "routea_matched_three_arm_common3_native7_crossfit_independent_validation_v2_20260902",
        "status": "ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION_PASS" if passed
                  else "ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION_ABORT",
        "checks": checks, "abort_reason": reason, "producer_result_sha256": sha(RESULT),
        "producer_result_logical_sha256": result.get("logical_sha256"), "model_update_replay_count": 12000 if passed else 0,
        "scientific_GO_or_NO_GO": None, "new_head_replacement_authorized": False,
        "external_confirmation_design_authorized": strict_ready,
        "next_authorized_stage": "MATCHED_THREE_ARM_EXTERNAL_CONFIRMATION_DESIGN" if strict_ready else None,
        "logical_sha256": ""}
    value["logical_sha256"] = logical(value); atomic(OUT, value)
    print(json.dumps({"status": value["status"], "checks": checks, "abort_reason": reason,
                      "next_authorized_stage": value["next_authorized_stage"]}, sort_keys=True))
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
