#!/usr/bin/env python3
"""Validate the N2/external design without reading any external endpoint."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_TO_UNTOUCHED_EXTERNAL_CONFIRMATION_DESIGN_V1_20260902.md"
N1_ROOT = ROOT / "results/routea_matched_three_arm_common3_native7_crossfit_v1"
N1_RESULT = N1_ROOT / "result.json"
N1_VALIDATION = N1_ROOT / "independent_validation.json"
CURRENT64_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
CURRENT64_VALIDATION = CURRENT64_ROOT / "validation.json"
SOURCES = {
    "d1_fold_runner": ROOT / "programs/run_a0_fold_d1.py",
    "d1_recipe_source": ROOT.parent / "programs/run_routea_v3_1_c6direct_m1.py",
    "d1_core": ROOT.parent / "route_a_core/route_a/o1_c6direct_m1_d1.py",
    "fold_authority": ROOT / "registry/upstream_inputs.json",
    "query987_ledger": ROOT / "cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json",
}
EXPECTED = {
    CONTRACT: "0b796b5a88481d7189ae026fc4e3464622f24f4e15b043e1e919a02f3df15ebd",
    N1_RESULT: "591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3",
    N1_VALIDATION: "1a7b08edfad7823db62f86c5f0b213106daead6f2ddba2d299102b69e546d8df",
    CURRENT64_VALIDATION: "6c031e5a82b8057a48b48794d53dd76f4888b76ed03f081a925425c64e9621cb",
    SOURCES["d1_fold_runner"]: "d8d58a0d96e1c3bc8457cf16f8b6aa0be46ceb7893a0cb6c0b75c50042eb0174",
    SOURCES["d1_recipe_source"]: "cdbac5833699c5904475cc308d1fbae6587a94765f9096d993093b2c1f621e41",
    SOURCES["d1_core"]: "a3973e45f97f2157883acb96b64c614d9fa93ec80187e6608f462fe841737274",
    SOURCES["fold_authority"]: "6f0399baa46a0c774a7214486467320eb0e778fe91cefd2b2010bc7512f75eac",
    SOURCES["query987_ledger"]: "df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec",
}
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_external_design_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def logical_sha256(value: dict) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "logical_sha256"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(not OUT_ROOT.exists(), f"immutable design validation exists: {OUT_ROOT}")
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha256_file(path) == expected, f"input hash drift: {path}")

    contract_text = CONTRACT.read_text()
    required_markers = (
        "design-only authority",
        "987-query current-runtime",
        "compact token cache",
        "obsolete `5404` legacy-label count",
        "`5412`",
        "corrected-identity mapping",
        "Pair64's historical `inner_fold`",
        "`DIFFICULT-0025`",
        "Single deployment model",
        "exactly `240`",
        "primary natural queries",
        "curator-only target-join manifest",
        "EXTERNAL_CANDIDATE_RECALL_NO_GO",
        "primary N3 one-shot enrollment",
        "Existing ISIC/IMA++ assets are opened",
        "exploratory architectural-generality result",
        "N2_CURRENT_RUNTIME_D1_C_E0_DESIGN_AND_LINEAGE_PREFLIGHT",
    )
    require(all(marker in contract_text for marker in required_markers), "design contract marker drift")

    result = json.loads(N1_RESULT.read_text())
    validation = json.loads(N1_VALIDATION.read_text())
    require(result.get("logical_sha256") == logical_sha256(result), "N1 result logical drift")
    require(validation.get("logical_sha256") == logical_sha256(validation), "N1 validation logical drift")
    require(
        validation.get("status") == "ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION_PASS"
        and len(validation.get("checks", {})) == 13
        and all(value is True for value in validation["checks"].values())
        and validation.get("next_authorized_stage") == "MATCHED_THREE_ARM_EXTERNAL_CONFIRMATION_DESIGN"
        and validation.get("scientific_GO_or_NO_GO") is None
        and validation.get("new_head_replacement_authorized") is False,
        "N1 validation boundary drift",
    )
    head = result["heads"]["NATIVE7"]["C_PAIRED"]
    real = result["evaluations"]["NATIVE7"]["C_PAIRED"]["real"]
    cbind = result["evaluations"]["NATIVE7"]["C_PAIRED"]["cbind"]
    retention = result["evaluations"]["NATIVE7"]["C_PAIRED"]["cbind_rescue_retention"]
    require(
        head["weight"] == [1.007810106300991, -3.7801241834284802, 5.716638282754007, 5.9125939065333615, -0.14872291353025213, -0.23872814933012354]
        and head["bias"] == -1.4778745133604176
        and head["parameter_sha256"] == "ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263"
        and real["base_top1"] == 25 and real["final_top1"] == 28
        and real["rescue"] == 3 and real["break"] == 0
        and cbind["final_top1"] == 20 and cbind["rescue"] == 0 and cbind["break"] == 5
        and retention == {"real_rescue_count": 3, "cbind_retained_rescue_count": 0, "cbind_rescue_retention": 0.0}
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("sealed_read_count") == 0
        and result["arm_selection_decision"]["new_head_replacement_authorized"] is False,
        "N1 selected-head metric or claim drift",
    )

    current = json.loads(CURRENT64_VALIDATION.read_text())
    require(
        current.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and current.get("logical_sha256") == logical_sha256(current)
        and current.get("query_count") == 64
        and len(current.get("shards", [])) == 8
        and all(value is True for value in current.get("checks", {}).values()),
        "current64 authority drift",
    )
    shard_seals = []
    for item in current["shards"]:
        shard = int(item["shard"])
        payload = CURRENT64_ROOT / f"shard{shard:02d}/payload.pt"
        receipt = CURRENT64_ROOT / f"shard{shard:02d}/receipt.json"
        shard_validation = CURRENT64_ROOT / f"shard{shard:02d}/validation.json"
        require(
            item == {
                "shard": shard,
                "payload_sha256": sha256_file(payload),
                "receipt_sha256": sha256_file(receipt),
                "validation_sha256": sha256_file(shard_validation),
            },
            "current64 shard seal drift",
        )
        shard_seals.append(item)

    ledger = json.loads(SOURCES["query987_ledger"].read_text())
    require(ledger.get("query_count") == 987 and ledger.get("logical_sha256"), "987 ledger envelope drift")
    checks = {
        "contract_physical_hash_and_required_clauses": True,
        "n1_result_and_independent_validation": True,
        "selected_native7_c_head_and_metrics": True,
        "d1_recipe_and_fold_source_hashes": True,
        "current64_validation_and_eight_shard_seals": True,
        "corrected_5412_and_canonical_fold_e0_requirements": True,
        "anonymous_external_prejoin_and_curator_join_boundary": True,
        "fixed_n240_bootstrap_candidate_recall_and_enrollment_rules": True,
        "external_query_target_and_sealed_access_zero": True,
    }
    value = {
        "schema_version": "routea_matched_three_arm_n2_external_design_validation_v1_20260902",
        "status": "ROUTEA_MATCHED_THREE_ARM_N2_EXTERNAL_DESIGN_VALIDATED",
        "claim_level": "DESIGN_ONLY_NO_N2_TRAINING_NO_EXTERNAL_ACCESS",
        "checks": checks,
        "bindings": {
            "contract_sha256": sha256_file(CONTRACT),
            "n1_result_sha256": sha256_file(N1_RESULT),
            "n1_result_logical_sha256": result["logical_sha256"],
            "n1_validation_sha256": sha256_file(N1_VALIDATION),
            "n1_validation_logical_sha256": validation["logical_sha256"],
            "current64_validation_sha256": sha256_file(CURRENT64_VALIDATION),
            "current64_validation_logical_sha256": current["logical_sha256"],
            "current64_shards": shard_seals,
            "d1_sources": {name: sha256_file(path) for name, path in SOURCES.items()},
            "validator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "access": {
            "internal_design_contract_read_count": 1,
            "internal_n1_result_read_count": 1,
            "internal_n1_validation_read_count": 1,
            "external_query_byte_read_count": 0,
            "external_target_read_count": 0,
            "external_outcome_read_count": 0,
            "sealed_query_read_count": 0,
            "model_update_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "external_execution_authorized": False,
        "n2_training_authorized": False,
        "next_authorized_stage": "N2_CURRENT_RUNTIME_D1_C_E0_DESIGN_AND_LINEAGE_PREFLIGHT",
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".n2-design-validation-", dir=OUT_ROOT.parent))
    try:
        output = staging / "independent_validation.json"
        with output.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.rename(staging, OUT_ROOT)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    print(json.dumps({"status": value["status"], "checks": checks, "next_authorized_stage": value["next_authorized_stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
