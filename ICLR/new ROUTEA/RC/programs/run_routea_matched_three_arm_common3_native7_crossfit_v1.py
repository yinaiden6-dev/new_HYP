#!/usr/bin/env python3
"""Final matched COMMON3/NATIVE7 A/B/C cross-fit consumer."""
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
sys.path.insert(0, str(ROOT / "programs"))
import run_routea_d1_current_runtime_bridge_e0_v1 as e0
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
FROZEN_VALID = ROOT / "results/romav2_colnomic_frozen_gate_definition_v1/validation.json"
FROZEN_RUNTIME = ROOT / "results/romav2_colnomic_current_runtime_frozen_gate_v1/result.json"
FROZEN_RUNTIME_VALID = ROOT / "results/romav2_colnomic_current_runtime_frozen_gate_v1/independent_validation.json"
PRIMARY = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md"
ADDENDUM = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_CURRENT_RUNTIME_REGRESSION_ADDENDUM_V1_20260902.md"
PAIR_ADDENDUM = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_ADDENDUM_V1_20260902.md"
EXECUTION_ADDENDUM = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_FINAL_EXECUTION_AUDIT_ADDENDUM_V1_20260902.md"
OUT = ROOT / "results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json"
VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_common3_native7_crossfit_v1.py"
LAUNCHER = ROOT / "slurm/routea_matched_three_arm_common3_native7_crossfit_v1_10m.sbatch"

EXPECTED_PRIMARY_SHA256 = "53b1d4f3b42a6f44b1e2242de13849b3e26bc4be3efb4e2c5ac45d1d11b4449e"
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
FAMILIES = {
    "COMMON3": ("real_common_features", "cbind_common_features", (
        "standardized_raw_gap", "symmetric_local_score")),
    "NATIVE7": ("real_native_features", "cbind_native_features", tuple(frozen.FEATURE_NAMES)),
}
STEPS = 2000
FROZEN_EXPECTED_SUMMARY = {
    "query_count": 32, "base_top1": 25, "final_top1": 27,
    "base_R@1": 25 / 32, "final_R@1": 27 / 32,
    "base_MRR": 0.8281994047619048, "final_MRR": 0.8779761904761905,
    "rescue": 2, "break": 0, "retained_correct": 25, "retained_wrong": 5,
    "wrong_to_wrong": 1, "switch_count": 3, "hold_count": 29,
}


class CrossfitContractError(RuntimeError):
    pass


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def exact_keys(value: dict, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise CrossfitContractError(f"{name} schema drift")


def finite_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise CrossfitContractError(f"{name} is not finite")


def finite_scalar(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise CrossfitContractError(f"{name} is not finite")


def parameter_sha(weight: torch.Tensor, bias: float) -> str:
    value = {"weight": [float(x) for x in weight.detach().cpu().flatten()], "bias": float(bias)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def gallery_seal(gallery) -> dict:
    return {
        "physical_row_count": len(gallery.raw_paths),
        "corrected_identity_count": len(set(gallery.corrected_identities)),
        "gallery_cache_sha256": gallery.gallery_cache_sha256,
        "raw_path_sequence_sha256": gallery.raw_path_sequence_sha256,
        "legacy_setid_sequence_sha256": gallery.legacy_setid_sequence_sha256,
        "corrected_mapping_sha256": gallery.corrected_mapping_sha256,
        "repair_contract_sha256": gallery.repair_contract_sha256,
        "repair_manifest_sha256": gallery.repair_manifest_sha256,
    }


def validate_pair_v2(fullv: dict) -> tuple[dict, dict, dict]:
    pairv = json.loads(PAIR_VALID.read_text())
    lineage = json.loads(PAIR_LINEAGE.read_text())
    receipt = json.loads(PAIR_RECEIPT.read_text())
    exact_keys(pairv, {
        "checks", "full_validation_logical_sha256", "full_validation_sha256",
        "logical_sha256", "model_update_count", "next_authorized_stage",
        "schema_version", "status", "v1_core_validation_logical_sha256",
        "v2_lineage_logical_sha256", "v2_lineage_sha256", "v2_payload_sha256",
    }, "Pair64 V2 validation")
    exact_keys(lineage, {
        "schema_version", "status", "v1_payload_sha256", "v1_validation_sha256",
        "v1_validation_logical_sha256", "v2_payload_sha256", "semantic_bit_exact",
        "full_validation_sha256", "full_validation_logical_sha256",
        "full_validation_mtime_ns", "v1_producer_sha256", "v1_validator_sha256",
        "v2_wrapper_sha256", "v2_validator_sha256", "v2_launcher_sha256",
        "primary_contract_sha256", "pair_addendum_sha256", "runtime_addendum_sha256",
        "slurm_job_id", "hostname", "gpu_name", "materialization_started_unix_ns",
        "materialization_finished_unix_ns", "model_update_count", "next_authorized_stage",
        "logical_sha256",
    }, "Pair64 V2 lineage")
    exact_keys(receipt, {
        "c_feature_max_abs", "claim_level", "eval_candidate_generation_authorized",
        "fixed_candidate_count", "logical_sha256", "model_update_count",
        "next_authorized_stage", "payload_sha256", "query_count", "schema_version",
        "status", "switch_label_count",
    }, "Pair64 V2 receipt")
    if not (
        pairv["status"] == "ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED"
        and pairv["logical_sha256"] == e0.logical_sha256(pairv)
        and set(pairv["checks"]) == {"bindings", "core", "full_predecessor", "lineage", "semantic_bit_exact", "v1_authority"}
        and all(x is True for x in pairv["checks"].values())
        and pairv["model_update_count"] == 0
        and pairv["next_authorized_stage"] == "MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_TRAINING"
        and pairv["v2_payload_sha256"] == e0.sha256_file(PAIR)
        and pairv["v2_lineage_sha256"] == e0.sha256_file(PAIR_LINEAGE)
        and pairv["v2_lineage_logical_sha256"] == lineage["logical_sha256"]
        and pairv["full_validation_sha256"] == e0.sha256_file(FULL_VALID)
        and pairv["full_validation_logical_sha256"] == fullv["logical_sha256"]
    ):
        raise CrossfitContractError("Pair64 V2 validation drift")
    if not (
        lineage["status"] == "ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_POST_FULL_READY"
        and lineage["logical_sha256"] == e0.logical_sha256(lineage)
        and lineage["semantic_bit_exact"] is True
        and lineage["v2_payload_sha256"] == e0.sha256_file(PAIR)
        and lineage["full_validation_sha256"] == e0.sha256_file(FULL_VALID)
        and lineage["full_validation_logical_sha256"] == fullv["logical_sha256"]
        and lineage["v2_wrapper_sha256"] == e0.sha256_file(PAIR_WRAPPER)
        and lineage["v2_validator_sha256"] == e0.sha256_file(PAIR_VALIDATOR)
        and lineage["v2_launcher_sha256"] == e0.sha256_file(PAIR_LAUNCHER)
        and lineage["v1_payload_sha256"] == e0.sha256_file(PAIR_V1_PAYLOAD)
        and lineage["v1_validation_sha256"] == e0.sha256_file(PAIR_V1_VALID)
        and lineage["v1_validation_logical_sha256"] == json.loads(PAIR_V1_VALID.read_text())["logical_sha256"]
        and lineage["v1_producer_sha256"] == e0.sha256_file(PAIR_V1_PRODUCER)
        and lineage["v1_validator_sha256"] == e0.sha256_file(PAIR_V1_VALIDATOR)
        and lineage["primary_contract_sha256"] == e0.sha256_file(PRIMARY)
        and lineage["pair_addendum_sha256"] == e0.sha256_file(PAIR_ADDENDUM)
        and lineage["runtime_addendum_sha256"] == e0.sha256_file(ADDENDUM)
        and lineage["model_update_count"] == 0
        and lineage["next_authorized_stage"] == "PAIR64_V2_INDEPENDENT_VALIDATION"
        and lineage["full_validation_mtime_ns"] == FULL_VALID.stat().st_mtime_ns
        and int(lineage["materialization_started_unix_ns"]) > FULL_VALID.stat().st_mtime_ns
        and int(lineage["materialization_finished_unix_ns"]) >= int(lineage["materialization_started_unix_ns"])
    ):
        raise CrossfitContractError("Pair64 V2 lineage drift")
    if not (
        receipt["status"] == "ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_READY"
        and receipt["logical_sha256"] == e0.logical_sha256(receipt)
        and receipt["payload_sha256"] == e0.sha256_file(PAIR)
        and receipt["query_count"] == receipt["switch_label_count"] == 64
        and receipt["fixed_candidate_count"] == 128
        and receipt["model_update_count"] == 0
        and receipt["eval_candidate_generation_authorized"] is False
    ):
        raise CrossfitContractError("Pair64 V2 receipt drift")
    return pairv, lineage, receipt


def validate_frozen_authority() -> tuple[dict, dict, dict]:
    definition = json.loads(FROZEN_VALID.read_text())
    runtime = json.loads(FROZEN_RUNTIME.read_text())
    runtimev = json.loads(FROZEN_RUNTIME_VALID.read_text())
    exact_keys(definition, {
        "bias", "checks", "feature_comparison_count", "feature_max_abs", "feature_names",
        "head_sha256", "module_sha256", "parameter_count", "schema_version", "status",
        "switch_threshold", "weight",
    }, "frozen definition")
    if not (
        definition["status"] == "ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS"
        and set(definition["checks"]) == {"all_features_bit_exact", "feature_names", "threshold", "weights"}
        and all(x is True for x in definition["checks"].values())
        and definition["feature_comparison_count"] == 8128
        and definition["feature_max_abs"] == 0.0
        and definition["module_sha256"] == e0.sha256_file(FROZEN_MODULE)
        and definition["head_sha256"] == e0.sha256_file(FROZEN_HEAD)
        and definition["feature_names"] == list(frozen.FEATURE_NAMES)
        and definition["weight"] == frozen.WEIGHT.tolist()
        and definition["bias"] == frozen.BIAS
        and definition["parameter_count"] == frozen.PARAMETER_COUNT == 7
        and definition["switch_threshold"] == frozen.SWITCH_THRESHOLD == 0.0
    ):
        raise CrossfitContractError("frozen definition authority drift")
    finite_tensor(frozen.WEIGHT, "frozen weight"); finite_scalar(frozen.BIAS, "frozen bias")
    if not (
        runtime.get("status") == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_GO"
        and runtime.get("logical_sha256") == e0.logical_sha256(runtime)
        and runtime.get("frozen_head_sha256") == e0.sha256_file(FROZEN_HEAD)
        and runtime.get("frozen_head_validation_sha256") == e0.sha256_file(FROZEN_HEAD_VALID)
        and runtime.get("parameter_count") == 7 and runtime.get("model_update_count") == 0
        and runtime.get("sealed_read_count") == 0 and runtime.get("scientific_GO_or_NO_GO") is None
        and len(runtime.get("actions", [])) == 32
    ):
        raise CrossfitContractError("frozen runtime action authority drift")
    if not (
        runtimev.get("status") == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_PASS"
        and set(runtimev.get("checks", {})) == {"actions_exact", "frozen_head", "gates_exact", "prejoin_validation", "result_envelope", "sealed_zero", "summary_exact"}
        and all(x is True for x in runtimev["checks"].values())
        and runtimev.get("result_sha256") == e0.sha256_file(FROZEN_RUNTIME)
        and runtimev.get("sealed_read_count") == 0
    ):
        raise CrossfitContractError("frozen runtime validation drift")
    headv = json.loads(FROZEN_HEAD_VALID.read_text())
    if not (
        headv.get("status") == "ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_INDEPENDENT_VALIDATION_PASS"
        and headv.get("logical_sha256") == e0.logical_sha256(headv)
        and set(headv.get("checks", {})) == {"action_recompute", "boundary", "decision", "envelope", "fold_isolation", "parameter_ledger", "sources"}
        and all(x is True for x in headv["checks"].values())
        and headv.get("result_sha256") == e0.sha256_file(FROZEN_HEAD)
        and headv.get("scientific_GO_or_NO_GO") is None
        and headv.get("sealed_data_read_authorized") is False
    ):
        raise CrossfitContractError("frozen source-head validation drift")
    return definition, runtime, runtimev


def validate_pair_record(record: dict) -> None:
    if not isinstance(record.get("switch_label"), bool):
        raise CrossfitContractError("Pair64 label drift")
    for family, (real_key, _, names) in FAMILIES.items():
        for arm in ARMS:
            value = record[real_key][arm]
            if not (isinstance(value, torch.Tensor) and tuple(value.shape) == (1, len(names)) and value.dtype == torch.float64):
                raise CrossfitContractError(f"Pair64 {family}/{arm} shape drift")
            finite_tensor(value, f"Pair64 {family}/{arm}")


def validate_full_record(record: dict) -> None:
    axis = [int(x) for x in record["candidate_physical_rows"]]
    base = record["base_scores"]
    winner = int(record["base_winner_position"])
    challengers = [int(x) for x in record["challenger_positions"]]
    if not (len(axis) == len(set(axis)) == 128 and isinstance(base, torch.Tensor) and tuple(base.shape) == (128,) and base.dtype == torch.float64):
        raise CrossfitContractError("full C128 axis drift")
    finite_tensor(base, "full base scores")
    expected_winner = max(range(128), key=lambda i: (float(base[i]), -axis[i]))
    if winner != expected_winner:
        raise CrossfitContractError("RAW winner drift")
    if not (len(challengers) == len(set(challengers)) == 127 and set(challengers) == set(range(128)) - {winner}):
        raise CrossfitContractError("challenger axis drift")
    source_positions = [int(x) for x in record["cbind_source_positions"]]
    if source_positions != [(position + 64) % 128 for position in range(128)]:
        raise CrossfitContractError("C_BIND fixed shift-64 ledger drift")
    for family, (real_key, control_key, names) in FAMILIES.items():
        for arm in ARMS:
            real, control = record[real_key][arm], record[control_key][arm]
            if not (isinstance(real, torch.Tensor) and isinstance(control, torch.Tensor)
                    and tuple(real.shape) == tuple(control.shape) == (127, len(names))
                    and real.dtype == control.dtype == torch.float64):
                raise CrossfitContractError(f"full {family}/{arm} shape drift")
            finite_tensor(real, f"full {family}/{arm} REAL"); finite_tensor(control, f"full {family}/{arm} C_BIND")
            if not torch.equal(real[:, 0], control[:, 0]):
                raise CrossfitContractError(f"{family}/{arm} C_BIND changed RAW gap")
            if e0.tensor_sha256(real) != record["feature_sha256"]["real_common" if family == "COMMON3" else "real_native"][arm]:
                raise CrossfitContractError(f"{family}/{arm} REAL feature hash drift")
            if e0.tensor_sha256(control) != record["feature_sha256"]["cbind_common" if family == "COMMON3" else "cbind_native"][arm]:
                raise CrossfitContractError(f"{family}/{arm} C_BIND feature hash drift")
    raw = base.tolist()
    for arm in ARMS:
        evidence = record["evidence"][arm]
        controlled = {destination: evidence[source] for destination, source in enumerate(source_positions)}
        expected_real = torch.stack([frozen.candidate_feature(raw, evidence, challenger, winner) for challenger in challengers])
        expected_control = torch.stack([frozen.candidate_feature(raw, controlled, challenger, winner) for challenger in challengers])
        if not torch.equal(expected_real, record["real_native_features"][arm]):
            raise CrossfitContractError(f"{arm} REAL formula replay drift")
        if not torch.equal(expected_control, record["cbind_native_features"][arm]):
            raise CrossfitContractError(f"{arm} C_BIND formula replay drift")
        if not torch.equal(expected_real[:, :2], record["real_common_features"][arm]):
            raise CrossfitContractError(f"{arm} REAL common projection drift")
        if not torch.equal(expected_control[:, :2], record["cbind_common_features"][arm]):
            raise CrossfitContractError(f"{arm} C_BIND common projection drift")


def load_inputs():
    if e0.sha256_file(PRIMARY) != EXPECTED_PRIMARY_SHA256:
        raise CrossfitContractError("primary contract drift")
    fullv = json.loads(FULL_VALID.read_text())
    if not (
        fullv.get("status") == "ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_VALIDATED"
        and fullv.get("logical_sha256") == e0.logical_sha256(fullv)
        and set(fullv.get("checks", {})) == {"all_shards", "c_replay", "folds", "population", "roles", "target_free", "tracks"}
        and all(x is True for x in fullv["checks"].values())
        and fullv.get("query_count") == 64 and fullv.get("target_role_read_count") == 0
        and fullv.get("model_update_count") == 0
    ):
        raise CrossfitContractError("full aggregate authority drift")
    pairv, lineage, receipt = validate_pair_v2(fullv)
    definition, runtime, runtimev = validate_frozen_authority()
    rolev = json.loads(ROLE_VALID.read_text()); rolem = json.loads(ROLE_MANIFEST.read_text())
    if not (
        rolev.get("status") == "RGH_P0_A0_MANIFEST_V2_INDEPENDENT_VALIDATION_PASS"
        and rolev.get("logical_sha256") == e0.logical_sha256(rolev)
        and rolev.get("query_count") == rolev.get("role_receipt_count") == 600
        and rolev.get("P0_scientific_reduction_authorized") is False
        and rolem.get("status") == "RGH_P0_A0_ROLE_MANIFEST_READY"
        and rolem.get("logical_sha256") == e0.logical_sha256(rolem)
        and rolem.get("role_shard_count") == 600
    ):
        raise CrossfitContractError("role authority drift")
    pair = torch.load(PAIR, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(pair, dict) or len(pair.get("records", [])) != 64:
        raise CrossfitContractError("Pair64 population drift")
    for record in pair["records"]: validate_pair_record(record)
    records, shard_seals = [], []
    seals = {int(x["shard"]): x for x in fullv["shards"]}
    if set(seals) != set(range(8)): raise CrossfitContractError("full shard ledger drift")
    for shard in range(8):
        payload_path = FULL / f"shard{shard:02d}/payload.pt"
        receipt_path = FULL / f"shard{shard:02d}/receipt.json"
        validation_path = FULL / f"shard{shard:02d}/validation.json"
        seal = {"shard": shard, "payload_sha256": e0.sha256_file(payload_path),
                "receipt_sha256": e0.sha256_file(receipt_path), "validation_sha256": e0.sha256_file(validation_path)}
        if seals[shard] != seal: raise CrossfitContractError("full shard seal drift")
        payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
        for record in payload["records"]: validate_full_record(record); records.append(record)
        shard_seals.append(seal)
    if len(records) != 64: raise CrossfitContractError("full population drift")
    roles = {int(x["execution_ordinal"]): x for x in rolem["shards"]}
    if set(roles) != set(range(600)):
        raise CrossfitContractError("role execution ledger drift")
    role_docs, role_seals = {}, []
    for execution in range(600):
        entry = roles[execution]; path = Path(entry["path"]); role = json.loads(path.read_text())
        if not (
            e0.sha256_file(path) == entry["sha256"]
            and role.get("logical_sha256") == entry.get("logical_sha256") == e0.logical_sha256(role)
            and role.get("status") == "RGH_P0_A0_ROLE_SHARD_READY"
            and int(role.get("execution_ordinal", -1)) == execution
            and role.get("target_naturally_present") is entry.get("target_naturally_present")
            and role.get("target_insertion_count") == 0
            and role.get("target_spatial_supervision_count") == 0
            and role.get("raw_d1_field_count") == 0
        ):
            raise CrossfitContractError("role shard seal drift")
        role_docs[execution] = role
        role_seals.append({"execution_ordinal": execution, "path": str(path), "sha256": entry["sha256"], "logical_sha256": entry["logical_sha256"]})
    gallery = build_gallery_source(verify_cache_file_sha256=True); labels = gallery.corrected_identities
    joined = []
    for record in records:
        execution = int(record["execution_ordinal"]); role = role_docs[execution]
        if not (role.get("query_id") == record["query_id"] and role.get("track") == record["track"]):
            raise CrossfitContractError("role join drift")
        identity = str(role["identity"])
        axis_identities = [labels[int(row)] for row in record["candidate_physical_rows"]]
        if len(axis_identities) != 128 or len(set(axis_identities)) != 128:
            raise CrossfitContractError("current C128 exact-label uniqueness drift")
        positions = [i for i, candidate_identity in enumerate(axis_identities) if candidate_identity == identity]
        if len(positions) != 1: raise CrossfitContractError("target not uniquely present")
        joined.append({**record, "target_position": positions[0], "target_identity": identity, "supergroup": str(role["supergroup"])})
    train = sorted((x for x in joined if x["data_split_role"] == "TRAIN"), key=lambda x: int(x["execution_ordinal"]))
    evals = sorted((x for x in joined if x["data_split_role"] == "EVAL"), key=lambda x: int(x["execution_ordinal"]))
    disjoint = {
        "execution": not bool({x["execution_ordinal"] for x in train} & {x["execution_ordinal"] for x in evals}),
        "identity": not bool({x["target_identity"] for x in train} & {x["target_identity"] for x in evals}),
        "supergroup": not bool({x["supergroup"] for x in train} & {x["supergroup"] for x in evals}),
        "train_identity_count": len({x["target_identity"] for x in train}), "eval_identity_count": len({x["target_identity"] for x in evals}),
        "train_supergroup_count": len({x["supergroup"] for x in train}), "eval_supergroup_count": len({x["supergroup"] for x in evals}),
    }
    if len(train) != 32 or len(evals) != 32 or not all(disjoint[k] for k in ("execution", "identity", "supergroup")):
        raise CrossfitContractError("TRAIN/EVAL disjointness drift")
    context = {"pair_validation": pairv, "pair_lineage": lineage, "pair_receipt": receipt,
               "full_validation": fullv, "role_validation": rolev, "role_manifest": rolem,
               "frozen_definition": definition, "frozen_runtime": runtime, "frozen_runtime_validation": runtimev,
               "full_shards": shard_seals, "role_shards": role_seals,
               "gallery": gallery_seal(gallery), "disjoint": disjoint}
    return pair, train, evals, context


def feature_ledger(pair: dict, train: list[dict]) -> dict:
    result = {}
    for family, (real_key, _, names) in FAMILIES.items():
        result[family] = {}
        for arm in ARMS:
            matrix = torch.cat([x[real_key][arm] for x in pair["records"]] + [x[real_key][arm] for x in train], dim=0).to(torch.float64)
            finite_tensor(matrix, f"{family}/{arm} design")
            zeros = [i for i in range(matrix.shape[1]) if int(torch.count_nonzero(matrix[:, i])) == 0]
            duplicates = [[i, j] for i in range(matrix.shape[1]) for j in range(i + 1, matrix.shape[1]) if torch.equal(matrix[:, i], matrix[:, j])]
            with_bias = torch.cat([matrix, torch.ones((matrix.shape[0], 1), dtype=torch.float64)], dim=1)
            result[family][arm] = {
                "feature_names": list(names), "matrix_shape": list(matrix.shape),
                "pair_row_count": 64, "fullnegative_row_count": 32 * 127,
                "feature_rank": int(torch.linalg.matrix_rank(matrix)),
                "design_with_bias_rank": int(torch.linalg.matrix_rank(with_bias)),
                "exact_zero_column_indices": zeros, "exact_zero_column_names": [names[i] for i in zeros],
                "exact_duplicate_column_pairs": duplicates,
                "exact_duplicate_column_name_pairs": [[names[i], names[j]] for i, j in duplicates],
                "all_finite": True,
            }
    return result


def train_head(pair: dict, train: list[dict], family: str, arm: str):
    real_key, _, names = FAMILIES[family]
    px = torch.cat([x[real_key][arm] for x in pair["records"]], dim=0).to(torch.float64)
    py = torch.tensor([float(x["switch_label"]) for x in pair["records"]], dtype=torch.float64)
    weights = torch.where(py.eq(0), torch.full_like(py, 4.0), torch.ones_like(py))
    finite_tensor(px, "pair train features"); finite_tensor(py, "pair labels")
    torch.manual_seed(17); head = nn.Linear(len(names), 1, dtype=torch.float64)
    head.weight.data.zero_(); head.bias.data.zero_()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=1e-3)
    loss_steps = grad_steps = 0; parameter_states = 1; final = (0.0, 0.0, 0.0)
    for _ in range(STEPS):
        optimizer.zero_grad(); pair_logits = head(px).squeeze(1); finite_tensor(pair_logits, "pair logits")
        pair_loss = (F.binary_cross_entropy_with_logits(pair_logits, py, reduction="none") * weights).mean(); query_losses = []
        for row in train:
            values = head(row[real_key][arm]).squeeze(1); finite_tensor(values, "full train logits")
            winner, target = int(row["base_winner_position"]), int(row["target_position"])
            challengers = [int(x) for x in row["challenger_positions"]]
            if target == winner: query_losses.append(4 * F.softplus(values.max()))
            else:
                position = challengers.index(target); mask = torch.ones(len(challengers), dtype=torch.bool); mask[position] = False
                query_losses.append(F.softplus(-values[position]) + 4 * F.softplus(values[mask].max()))
        full_loss = torch.stack(query_losses).mean(); loss = pair_loss + full_loss
        if not bool(torch.isfinite(torch.stack([loss, pair_loss, full_loss])).all()): raise CrossfitContractError("non-finite loss")
        loss_steps += 1; loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()): raise CrossfitContractError("non-finite gradient")
        grad_steps += 1; optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in head.parameters()): raise CrossfitContractError("non-finite parameter")
        parameter_states += 1; final = (float(loss.detach()), float(pair_loss.detach()), float(full_loss.detach()))
    finite = {"input_finite": True, "finite_loss_step_count": loss_steps, "finite_gradient_step_count": grad_steps,
              "finite_parameter_state_count": parameter_states,
              "all_finite": loss_steps == grad_steps == STEPS and parameter_states == STEPS + 1}
    return head, final, finite


def actions(weight: torch.Tensor, bias: float, rows: list[dict], family: str, arm: str, control: bool = False) -> list[dict]:
    finite_tensor(weight, "action weight"); finite_scalar(bias, "action bias")
    real_key, control_key, _ = FAMILIES[family]; key = control_key if control else real_key; output = []
    for row in rows:
        axis = [int(x) for x in row["candidate_physical_rows"]]; base = row["base_scores"]
        winner = int(row["base_winner_position"]); challengers = [int(x) for x in row["challenger_positions"]]
        logits = row[key][arm] @ weight + float(bias); finite_tensor(logits, "action logits")
        index = max(range(127), key=lambda i: (float(logits[i]), -axis[challengers[i]])); best = challengers[index]
        best_logit = float(logits[index]); decision = "SWITCH" if best_logit > 0.0 else "HOLD"; final = best if decision == "SWITCH" else winner
        target = int(row["target_position"]); base_order = sorted(range(128), key=lambda i: (-float(base[i]), axis[i]))
        if base_order[0] != winner: raise CrossfitContractError("action winner/tie drift")
        final_order = list(base_order)
        if decision == "SWITCH": final_order.remove(best); final_order.insert(0, best)
        base_correct, final_correct = winner == target, final == target
        output.append({
            "execution_ordinal": int(row["execution_ordinal"]), "query_id": row["query_id"], "track": row["track"],
            "heldout_fold": int(row["heldout_fold"]), "base_winner": winner, "base_winner_physical_row": axis[winner],
            "target_position": target, "target_physical_row": axis[target], "base_target_rank": base_order.index(target) + 1,
            "final_target_rank": final_order.index(target) + 1, "proposed_challenger": best,
            "proposed_challenger_physical_row": axis[best], "proposed_challenger_base_rank": base_order.index(best) + 1,
            "switch_logit": best_logit, "decision": decision, "final_position": final, "final_physical_row": axis[final],
            "base_correct": base_correct, "final_correct": final_correct,
            "wrong_to_wrong": (not base_correct) and (not final_correct) and final != winner,
        })
    return output


def summary(rows: list[dict]) -> dict:
    n = len(rows); base = sum(x["base_correct"] for x in rows); final = sum(x["final_correct"] for x in rows)
    return {"query_count": n, "base_top1": base, "final_top1": final, "base_R@1": base / n, "final_R@1": final / n,
            "base_MRR": sum(1 / x["base_target_rank"] for x in rows) / n,
            "final_MRR": sum(1 / x["final_target_rank"] for x in rows) / n,
            "rescue": sum((not x["base_correct"]) and x["final_correct"] for x in rows),
            "break": sum(x["base_correct"] and not x["final_correct"] for x in rows),
            "retained_correct": sum(x["base_correct"] and x["final_correct"] for x in rows),
            "retained_wrong": sum((not x["base_correct"]) and (not x["final_correct"]) for x in rows),
            "wrong_to_wrong": sum(x["wrong_to_wrong"] for x in rows),
            "switch_count": sum(x["decision"] == "SWITCH" for x in rows), "hold_count": sum(x["decision"] == "HOLD" for x in rows)}


def grouped(rows: list[dict], key: str) -> dict:
    return {str(group): summary([x for x in rows if x[key] == group]) for group in sorted({x[key] for x in rows})}


def retention(real: list[dict], control: list[dict]) -> dict:
    by_execution = {x["execution_ordinal"]: x for x in control}
    rescues = [x for x in real if (not x["base_correct"]) and x["final_correct"]]
    kept = sum(by_execution[x["execution_ordinal"]]["final_correct"] for x in rescues)
    return {"real_rescue_count": len(rescues), "cbind_retained_rescue_count": kept,
            "cbind_rescue_retention": kept / len(rescues) if rescues else 1.0}


def replay_runtime(current: list[dict], runtime: dict) -> float:
    historical = {int(x["execution_ordinal"]): x for x in runtime["actions"]}
    if set(historical) != {x["execution_ordinal"] for x in current}: raise CrossfitContractError("runtime execution axis drift")
    fields = ("query_id", "base_winner", "target_position", "proposed_challenger", "decision", "final_position", "base_correct", "final_correct")
    maximum = 0.0
    for row in current:
        old = historical[row["execution_ordinal"]]
        if any(row[field] != old[field] for field in fields): raise CrossfitContractError("runtime action replay drift")
        maximum = max(maximum, abs(row["switch_logit"] - old["switch_logit"]))
    return maximum


def bindings(context: dict) -> dict:
    return {
        "primary_contract_sha256": e0.sha256_file(PRIMARY), "runtime_addendum_sha256": e0.sha256_file(ADDENDUM),
        "pair_addendum_sha256": e0.sha256_file(PAIR_ADDENDUM), "final_execution_addendum_sha256": e0.sha256_file(EXECUTION_ADDENDUM),
        "pair_payload_sha256": e0.sha256_file(PAIR), "pair_receipt_sha256": e0.sha256_file(PAIR_RECEIPT),
        "pair_receipt_logical_sha256": context["pair_receipt"]["logical_sha256"], "pair_lineage_sha256": e0.sha256_file(PAIR_LINEAGE),
        "pair_lineage_logical_sha256": context["pair_lineage"]["logical_sha256"], "pair_validation_sha256": e0.sha256_file(PAIR_VALID),
        "pair_validation_logical_sha256": context["pair_validation"]["logical_sha256"], "pair_wrapper_sha256": e0.sha256_file(PAIR_WRAPPER),
        "pair_validator_sha256": e0.sha256_file(PAIR_VALIDATOR), "pair_launcher_sha256": e0.sha256_file(PAIR_LAUNCHER),
        "full_validation_sha256": e0.sha256_file(FULL_VALID), "full_validation_logical_sha256": context["full_validation"]["logical_sha256"],
        "full_shards": context["full_shards"], "role_manifest_sha256": e0.sha256_file(ROLE_MANIFEST),
        "role_manifest_logical_sha256": context["role_manifest"]["logical_sha256"], "role_validation_sha256": e0.sha256_file(ROLE_VALID),
        "role_validation_logical_sha256": context["role_validation"]["logical_sha256"], "role_shards": context["role_shards"],
        "gallery": context["gallery"], "frozen_module_sha256": e0.sha256_file(FROZEN_MODULE), "frozen_head_sha256": e0.sha256_file(FROZEN_HEAD),
        "frozen_head_validation_sha256": e0.sha256_file(FROZEN_HEAD_VALID),
        "frozen_definition_validation_sha256": e0.sha256_file(FROZEN_VALID), "frozen_runtime_result_sha256": e0.sha256_file(FROZEN_RUNTIME),
        "frozen_runtime_result_logical_sha256": context["frozen_runtime"]["logical_sha256"],
        "frozen_runtime_validation_sha256": e0.sha256_file(FROZEN_RUNTIME_VALID), "trainer_sha256": e0.sha256_file(Path(__file__).resolve()),
        "validator_sha256": e0.sha256_file(VALIDATOR), "launcher_sha256": e0.sha256_file(LAUNCHER),
    }


def zero_abort(status: str, phase: str, reason: str) -> None:
    value = {"schema_version": "routea_matched_three_arm_common3_native7_crossfit_abort_v2_20260902", "status": status,
             "claim_level": "ZERO_UPDATE_ENGINEERING_ABORT", "abort_phase": phase, "abort_reason": reason,
             "model_update_count": 0, "optimizer_constructed": False, "sealed_read_count": 0,
             "scientific_GO_or_NO_GO": None, "next_authorized_stage": None, "logical_sha256": ""}
    value["logical_sha256"] = e0.logical_sha256(value); atomic_json(OUT, value)


def frozen_regression(evals: list[dict], context: dict) -> tuple[dict, bool]:
    real_frozen = actions(frozen.WEIGHT, frozen.BIAS, evals, "NATIVE7", "C_PAIRED")
    cbind_frozen = actions(frozen.WEIGHT, frozen.BIAS, evals, "NATIVE7", "C_PAIRED", True)
    frozen_summary, frozen_cbind_summary = summary(real_frozen), summary(cbind_frozen)
    logit_max_abs = 0.0
    for row in evals:
        raw = row["base_scores"].tolist(); winner = int(row["base_winner_position"])
        evidence = row["evidence"]["C_PAIRED"]; features = row["real_native_features"]["C_PAIRED"]
        for index, challenger in enumerate(row["challenger_positions"]):
            reconstructed = float(features[index] @ frozen.WEIGHT + frozen.BIAS)
            direct = frozen.logit(raw, evidence, challenger, winner)
            finite_scalar(reconstructed, "reconstructed logit"); finite_scalar(direct, "direct logit")
            logit_max_abs = max(logit_max_abs, abs(reconstructed - direct))
    runtime_max_abs = replay_runtime(real_frozen, context["frozen_runtime"])
    passed = frozen_summary == FROZEN_EXPECTED_SUMMARY and logit_max_abs <= 1e-12 and runtime_max_abs <= 1e-12
    block = {
        "pass": passed, "expected_summary": FROZEN_EXPECTED_SUMMARY, "summary": frozen_summary,
        "cbind_summary": frozen_cbind_summary, "cbind_rescue_retention": retention(real_frozen, cbind_frozen),
        "real_actions": real_frozen, "cbind_actions": cbind_frozen, "logit_max_abs": logit_max_abs,
        "runtime_action_max_abs": runtime_max_abs, "finite": True,
        "authority": {"feature_names": list(frozen.FEATURE_NAMES), "weight": frozen.WEIGHT.tolist(), "bias": frozen.BIAS,
                      "parameter_count": frozen.PARAMETER_COUNT, "switch_threshold": frozen.SWITCH_THRESHOLD,
                      "parameter_sha256": parameter_sha(frozen.WEIGHT, frozen.BIAS)},
    }
    return block, passed


def main() -> None:
    if OUT.exists(): raise RuntimeError("immutable crossfit result exists")
    try:
        pair, train, evals, context = load_inputs(); ledger = feature_ledger(pair, train)
    except Exception as error:
        zero_abort("ROUTEA_MATCHED_THREE_ARM_INPUT_AUTHORITY_OR_FINITE_ABORT", "PRE_OPTIMIZER_INPUT", f"{type(error).__name__}: {error}")
        raise SystemExit(4)
    try:
        frozen_block, frozen_pass = frozen_regression(evals, context)
    except Exception as error:
        zero_abort("ROUTEA_MATCHED_THREE_ARM_C_REGRESSION_ABORT", "PRE_OPTIMIZER_FROZEN_C_REGRESSION",
                   f"{type(error).__name__}: {error}")
        raise SystemExit(4)
    if not frozen_pass:
        value = {"schema_version": "routea_matched_three_arm_common3_native7_crossfit_abort_v2_20260902",
                 "status": "ROUTEA_MATCHED_THREE_ARM_C_REGRESSION_ABORT", "claim_level": "FROZEN_C_REGRESSION_ONLY_ZERO_TRAINING",
                 "abort_phase": "PRE_OPTIMIZER_FROZEN_C_REGRESSION", "frozen_c_regression": frozen_block,
                 "feature_degeneracy_ledger": ledger, "model_update_count": 0, "optimizer_constructed": False,
                 "sealed_read_count": 0, "scientific_GO_or_NO_GO": None, "next_authorized_stage": None, "logical_sha256": ""}
        value["logical_sha256"] = e0.logical_sha256(value); atomic_json(OUT, value); raise SystemExit(4)
    heads, evaluations = {}, {}
    for family in FAMILIES:
        heads[family], evaluations[family] = {}, {}
        for arm in ARMS:
            head, losses, finite = train_head(pair, train, family, arm); weight = head.weight.detach().flatten(); bias = float(head.bias.detach())
            real = actions(weight, bias, evals, family, arm); control = actions(weight, bias, evals, family, arm, True)
            rs, cs, retain = summary(real), summary(control), retention(real, control)
            heads[family][arm] = {"weight": weight.tolist(), "bias": bias, "parameter_count": weight.numel() + 1,
                                  "loss_total": losses[0], "loss_pair": losses[1], "loss_fullnegative": losses[2],
                                  "parameter_sha256": parameter_sha(weight, bias), "finite_training": finite}
            evaluations[family][arm] = {
                "real": rs, "cbind": cs, "cbind_rescue_retention": retain, "real_by_track": grouped(real, "track"),
                "real_by_heldout_fold": grouped(real, "heldout_fold"), "actions": real, "cbind_actions": control,
                "finite_action_count": len(real), "finite_cbind_action_count": len(control),
                "actionable": rs["final_top1"] > 25 and rs["rescue"] > rs["break"] and rs["break"] <= 1,
                "strict_external_candidate": (rs["final_top1"] > FROZEN_EXPECTED_SUMMARY["final_top1"]
                    and rs["rescue"] > rs["break"] and rs["break"] <= 1
                    and rs["final_top1"] > cs["final_top1"] and retain["real_rescue_count"] > 0
                    and retain["cbind_rescue_retention"] < 1.0),
            }
    strict = [{"family": f, "arm": a, "final_top1": evaluations[f][a]["real"]["final_top1"]}
              for f in FAMILIES for a in ARMS if evaluations[f][a]["strict_external_candidate"]]
    result = {
        "schema_version": "routea_matched_three_arm_common3_native7_crossfit_v2_20260902",
        "status": "ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_COMPLETE",
        "claim_level": "INTERNAL_ARM_SELECTION_CROSSFIT_ONLY", "frozen_c_regression": frozen_block,
        "head_families": {
            "COMMON3": {"parameter_count": 3, "feature_names": list(FAMILIES["COMMON3"][2]), "effective_capacity": "STRICT_COMMON_NONDEGENERATE"},
            "NATIVE7": {"parameter_count": 7, "feature_names": list(FAMILIES["NATIVE7"][2]), "effective_capacity": "NOMINAL_MECHANISM_SPECIFIC_WITH_REGISTERED_DEGENERACIES"}},
        "feature_degeneracy_ledger": ledger, "heads": heads, "evaluations": evaluations, "train_eval_disjoint": context["disjoint"],
        "action_authority": {"candidate_count": 128, "challenger_count": 127, "switch_threshold": 0.0,
            "base_tie_break": "score_descending_then_physical_row_ascending",
            "challenger_tie_break": "logit_descending_then_physical_row_ascending",
            "hold_transform": "preserve_full_base_order",
            "switch_transform": "move_challenger_to_rank1_shift_preceding_candidates_only",
            "cbind_preserves_destination_base_axis": True, "all_action_logits_finite": True},
        "arm_selection_decision": {"deployment_default": "FROZEN_C", "new_head_replacement_authorized": False,
            "provisional_external_candidates": strict, "external_confirmation_design_eligible_after_independent_validation": bool(strict),
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
        "fullnegative_train_query_count": 32, "eval_query_count": 32, "model_update_count": 6 * STEPS,
        "sealed_read_count": 0, "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": "MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION", "logical_sha256": ""}
    result["logical_sha256"] = e0.logical_sha256(result); atomic_json(OUT, result)
    print(json.dumps({"status": result["status"], "strict_candidates": strict,
                      "evaluation": {f: {a: x["real"] for a, x in v.items()} for f, v in evaluations.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
