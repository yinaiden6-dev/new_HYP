#!/usr/bin/env python3
"""Independent replay of one target-free full-negative feature shard."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch
from torch.nn import functional as F


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
REFERENCE_PAYLOAD = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1/payload.pt"
REFERENCE_VALIDATION = ROOT / "results/routea_d1_current_runtime_matched_three_arm_reference_cache_v1/independent_validation.json"
RAW_C_ROOT = ROOT / "results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1"
RAW_C_VALIDATION = RAW_C_ROOT / "validation.json"
FROZEN_MODULE = ROOT / "src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py"
FROZEN_VALIDATION = ROOT / "results/romav2_colnomic_frozen_gate_definition_v1/validation.json"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_fullnegative_features_v1"
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
PAYLOAD_KEYS={"schema_version","status","claim_level","shard","shard_count","arms","records","c_feature_max_abs","bindings","access"}
RECORD_KEYS={"data_split_role","execution_ordinal","query_id","track","heldout_fold","candidate_physical_rows","base_scores","base_winner_position","challenger_positions","cbind_source_positions","evidence","real_common_features","real_native_features","cbind_common_features","cbind_native_features","feature_sha256","target_role_read_count","model_update_count"}
EVIDENCE_KEYS={"real_score","visibility_mass","query_control_score","reference_control_score"}
RECEIPT_KEYS={"schema_version","status","shard","query_count","c_feature_max_abs","payload_sha256","target_role_read_count","model_update_count","next_authorized_stage","logical_sha256"}
PRODUCER_FILE=ROOT/'programs/materialize_routea_matched_three_arm_fullnegative_features_shard_v1.py'
VALIDATOR_FILE=Path(__file__).resolve()
AGGREGATE_FILE=ROOT/'programs/validate_routea_matched_three_arm_fullnegative_features_aggregate_v1.py'
CORE_FILE=ROOT/'programs/run_romav2_colnomic_visibility_xf_six_case_v1.py'
LAUNCHER_FILE=ROOT/'slurm/routea_matched_three_arm_fullnegative_features_v1_10m.sbatch'


def score(query_tokens,reference_tokens,query_weights,reference_weights):
    q=F.normalize(torch.as_tensor(query_tokens).to(torch.float64),dim=1)
    r=F.normalize(torch.as_tensor(reference_tokens).to(torch.float64),dim=1)
    wq=torch.as_tensor(query_weights,dtype=torch.float64);wr=torch.as_tensor(reference_weights,dtype=torch.float64)
    similarity=q@r.T;local=(similarity*wr[None]).max(1).values
    mass=torch.sqrt(wq.mean()*wr.mean())
    value=mass*(wq*local).sum()/wq.sum().clamp_min(1e-12)
    return float(value),float(mass)


def sym(a: float, b: float) -> float:
    return (float(a) - float(b)) / (abs(float(a)) + abs(float(b)) + 1e-12)


def feature(raw: list[float], evidence: dict[int, dict], c: int, w: int) -> torch.Tensor:
    mean = sum(raw) / len(raw)
    std = (sum((value - mean) ** 2 for value in raw) / len(raw)) ** 0.5
    lc, lw = evidence[c], evidence[w]
    sc = lc["real_score"] / max(lc["visibility_mass"], 1e-12)
    sw = lw["real_score"] / max(lw["visibility_mass"], 1e-12)
    qc = lc["real_score"] - lc["query_control_score"]
    qw = lw["real_score"] - lw["query_control_score"]
    rc = lc["real_score"] - lc["reference_control_score"]
    rw = lw["real_score"] - lw["reference_control_score"]
    return torch.tensor(
        [
            (raw[c] - raw[w]) / max(std, 1e-12),
            sym(lc["real_score"], lw["real_score"]),
            sym(lc["visibility_mass"], lw["visibility_mass"]),
            sym(sc, sw),
            sym(qc, qw),
            sym(rc, rw),
        ],
        dtype=torch.float64,
    )


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--shard", type=int, required=True); args = parser.parse_args()
    if args.shard not in range(8): raise ValueError("shard must be in 0..7")
    shard_dir = OUT_ROOT / f"shard{args.shard:02d}"
    payload_path, receipt_path, out = shard_dir / "payload.pt", shard_dir / "receipt.json", shard_dir / "validation.json"
    if out.exists(): raise RuntimeError("immutable fullnegative feature validation exists")
    output = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    receipt = json.loads(receipt_path.read_text())
    arm_path = ARM_ROOT / f"shard{args.shard:02d}/payload.pt"; base_path = BASE_ROOT / f"shard{args.shard:02d}/payload.pt"
    arms = torch.load(arm_path, map_location="cpu", weights_only=False, mmap=True)
    bases = torch.load(base_path, map_location="cpu", weights_only=False, mmap=True)
    refs = torch.load(REFERENCE_PAYLOAD, map_location="cpu", weights_only=False, mmap=True)["references"]
    raw_c_path = RAW_C_ROOT / f"shard{args.shard:02d}/result.json"; raw_c = json.loads(raw_c_path.read_text())
    arm_by = {int(x["execution_ordinal"]): x for x in arms["records"]}; base_by = {int(x["execution_ordinal"]): x for x in bases["records"]}; old_by = {int(x["execution_ordinal"]): x for x in raw_c["rows"]}
    max_abs = 0.0; record_checks = []
    for record in output["records"]:
        if set(record)!=RECORD_KEYS:raise RuntimeError('fullnegative feature record schema drift')
        feature_families=("real_common_features","real_native_features","cbind_common_features","cbind_native_features")
        family_hash_names=("real_common","real_native","cbind_common","cbind_native")
        nested_schema_ok=(set(record['evidence'])==set(ARMS) and all(set(record[name])==set(ARMS) for name in feature_families) and set(record['feature_sha256'])==set(family_hash_names) and all(set(record['feature_sha256'][name])==set(ARMS) for name in family_hash_names))
        ex = int(record["execution_ordinal"]); ar = arm_by[ex]; br = base_by[ex]; axis = list(map(int, ar["raw_candidate_physical_rows"])); cands = {int(x["physical_row"]): x for x in ar["candidates"]}; old = {int(x["physical_row"]): x for x in old_by[ex]["candidates"]}
        raw = [float(br["raw_full_gallery_scores"][row]) for row in axis]; winner = max(range(128), key=lambda i: (raw[i], -axis[i])); expected_evidence = {arm: {} for arm in ARMS}
        for position, row in enumerate(axis):
            cand = cands[row]; q = torch.as_tensor(cand["query_map"], dtype=torch.float64); r = torch.as_tensor(cand["reference_map"], dtype=torch.float64); qt = br["raw_image_tokens"]; rt = refs[row]["tokens"]; oq, orf = torch.ones_like(q), torch.ones_like(r)
            a,_ = score(qt,rt,oq,orf);b,bmass=score(qt,rt,q,orf);bq,_=score(qt,rt,q.roll(max(1,q.numel()//2)),orf);oc=old[row];c,cmass=score(qt,rt,q,r);cq,_=score(qt,rt,q.roll(max(1,q.numel()//2)),r);cr,_=score(qt,rt,q,r.roll(max(1,r.numel()//2)))
            expected_evidence["A_ALL"][position] = {"real_score": a, "visibility_mass": 1.0, "query_control_score": a, "reference_control_score": a}
            expected_evidence["B_QUERY"][position] = {"real_score": b, "visibility_mass": bmass, "query_control_score": bq, "reference_control_score": b}
            expected_evidence["C_PAIRED"][position] = {"real_score": c, "visibility_mass": cmass, "query_control_score": cq, "reference_control_score": cr}
            if not(c==float(oc['real_score']) and cmass==float(oc['visibility_mass']) and cq==float(oc['query_control_score']) and cr==float(oc['reference_control_score'])):raise RuntimeError('independent C evidence replay drift')
        challengers = [i for i in range(128) if i != winner]; shift = [(i + 64) % 128 for i in range(128)]; checks = []
        for arm in ARMS:
            real = torch.stack([feature(raw, expected_evidence[arm], c, winner) for c in challengers]); ctrl_ev = {d: expected_evidence[arm][s] for d, s in enumerate(shift)}; ctrl = torch.stack([feature(raw, ctrl_ev, c, winner) for c in challengers])
            max_abs = max(max_abs, float((real - record["real_native_features"][arm]).abs().max()), float((ctrl - record["cbind_native_features"][arm]).abs().max()))
            hashes=record['feature_sha256'];stored_real_native=record["real_native_features"][arm];stored_real_common=record["real_common_features"][arm];stored_ctrl_native=record["cbind_native_features"][arm];stored_ctrl_common=record["cbind_common_features"][arm];checks.append(set(record["evidence"][arm])==set(range(128)) and all(set(x)==EVIDENCE_KEYS for x in record['evidence'][arm].values()) and record["evidence"][arm] == expected_evidence[arm] and stored_real_native.dtype==stored_real_common.dtype==stored_ctrl_native.dtype==stored_ctrl_common.dtype==torch.float64 and stored_real_native.shape==stored_ctrl_native.shape==(127,6) and stored_real_common.shape==stored_ctrl_common.shape==(127,2) and torch.equal(real,stored_real_native) and torch.equal(real[:,:2],stored_real_common) and torch.equal(ctrl,stored_ctrl_native) and torch.equal(ctrl[:,:2],stored_ctrl_common) and hashes['real_native'][arm]==e0.tensor_sha256(real)==e0.tensor_sha256(stored_real_native) and hashes['real_common'][arm]==e0.tensor_sha256(real[:,:2])==e0.tensor_sha256(stored_real_common) and hashes['cbind_native'][arm]==e0.tensor_sha256(ctrl)==e0.tensor_sha256(stored_ctrl_native) and hashes['cbind_common'][arm]==e0.tensor_sha256(ctrl[:,:2])==e0.tensor_sha256(stored_ctrl_common))
        old_candidates={int(x['candidate_position']):x for x in old_by[ex]['candidates']};old_raw=[float(old_candidates[i]['raw_score']) for i in range(128)];frozen_expected=torch.stack([frozen.candidate_feature(old_raw,old_candidates,c,winner) for c in challengers]);record_checks.append(nested_schema_ok and record['data_split_role']==ar['role'] and record['query_id']==ar['query_id'] and record['track']==ar['track'] and record['heldout_fold']==ar['heldout_fold'] and record["candidate_physical_rows"] == axis and raw==old_raw and torch.equal(record["base_scores"], torch.tensor(raw, dtype=torch.float64)) and record["base_winner_position"] == winner and record["challenger_positions"] == challengers and record["cbind_source_positions"] == shift and torch.equal(record['real_native_features']['C_PAIRED'],frozen_expected) and all(checks) and record["target_role_read_count"] == record["model_update_count"] == 0)
    arm_val=json.loads(ARM_VALIDATION.read_text());base_val=json.loads(BASE_VALIDATION.read_text());ref_val=json.loads(REFERENCE_VALIDATION.read_text());raw_val=json.loads(RAW_C_VALIDATION.read_text());frozen_val=json.loads(FROZEN_VALIDATION.read_text())
    expected_bindings={
        "contract_sha256":e0.sha256_file(CONTRACT),"arm_payload_sha256":e0.sha256_file(arm_path),"arm_validation_sha256":e0.sha256_file(ARM_VALIDATION),"base_payload_sha256":e0.sha256_file(base_path),"base_validation_sha256":e0.sha256_file(BASE_VALIDATION),"reference_payload_sha256":e0.sha256_file(REFERENCE_PAYLOAD),"reference_validation_sha256":e0.sha256_file(REFERENCE_VALIDATION),"raw_c_authority_sha256":e0.sha256_file(raw_c_path),"raw_c_validation_sha256":e0.sha256_file(RAW_C_VALIDATION),"frozen_gate_module_sha256":e0.sha256_file(FROZEN_MODULE),"frozen_gate_validation_sha256":e0.sha256_file(FROZEN_VALIDATION),"producer_sha256":e0.sha256_file(PRODUCER_FILE),"validator_sha256":e0.sha256_file(VALIDATOR_FILE),"aggregate_sha256":e0.sha256_file(AGGREGATE_FILE),"core_score_sha256":e0.sha256_file(CORE_FILE),"launcher_sha256":e0.sha256_file(LAUNCHER_FILE)
    }
    checks={
        "envelope":set(output)==PAYLOAD_KEYS and output.get('schema_version')=='routea_matched_three_arm_fullnegative_feature_shard_v1_20260902' and output.get("status")=="ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_READY" and output.get("shard")==args.shard and output.get('shard_count')==8 and tuple(output.get('arms',()))==ARMS and output.get("claim_level")=="TARGET_FREE_RAW_FEATURE_LEDGER_ONLY",
        "authorities":arm_val.get("status")=="ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED" and all(arm_val.get('checks',{}).values()) and base_val.get("status")=="ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED" and all(base_val.get('checks',{}).values()) and ref_val.get("status")=="ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_REFERENCE_CACHE_VALIDATED" and all(ref_val.get('checks',{}).values()) and raw_val.get("status")=="ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS" and frozen_val.get("status")=="ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS",
        "bindings":output["bindings"]==expected_bindings,
        "records":len(record_checks)==8 and all(record_checks) and max_abs==0.0 and output.get("c_feature_max_abs")==0.0,
        "access":output.get('access')=={'target_identity_read_count':0,'target_role_read_count':0,'supergroup_read_count':0,'sealed_read_count':0,'model_update_count':0},
        "receipt":set(receipt)==RECEIPT_KEYS and receipt.get('schema_version')=='routea_matched_three_arm_fullnegative_feature_receipt_v1_20260902' and receipt.get('status')=='ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_READY' and receipt.get('shard')==args.shard and receipt.get("logical_sha256")==e0.logical_sha256(receipt) and receipt.get("payload_sha256")==e0.sha256_file(payload_path) and receipt.get("query_count")==8 and receipt.get("c_feature_max_abs")==0.0 and receipt.get('target_role_read_count')==receipt.get('model_update_count')==0 and receipt.get('next_authorized_stage')=='FULLNEGATIVE_FEATURE_SHARD_VALIDATION'
    }
    passed=all(checks.values()); value={"schema_version":"routea_matched_three_arm_fullnegative_feature_validation_v1_20260902","status":"ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_VALIDATED" if passed else "ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_VALIDATION_ABORT","shard":args.shard,"checks":checks,"query_count":len(record_checks),"feature_max_abs":max_abs,"payload_sha256":e0.sha256_file(payload_path),"receipt_sha256":e0.sha256_file(receipt_path),"target_role_read_count":0,"model_update_count":0,"next_authorized_stage":"FULLNEGATIVE_FEATURE_AGGREGATE" if passed else None,"logical_sha256":""};value["logical_sha256"]=e0.logical_sha256(value);atomic_json(out,value);print(json.dumps({"status":value["status"],"checks":checks},sort_keys=True));raise SystemExit(0 if passed else 4)


if __name__ == "__main__": main()
