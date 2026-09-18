#!/usr/bin/env python3
"""Validate the formal current64 J1 execution authority without writes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_EXECUTION_AUTHORITY_V1_20260904.json"
PRODUCER = ROOT / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1.py"
J1_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1.py"
LAUNCHER = ROOT / "slurm/routea_n2_fresh_d1_matched_three_arm_current64_j1_v1_10m.sbatch"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1"

VERSION = "routea_n2_fresh_d1_matched_three_arm_current64_j1_execution_authority_v1_20260904"
STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_EXECUTION_AUTHORIZED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_EXECUTION"
CONTRACT_SHA256 = "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79"
PRODUCER_SHA256 = "3532a71264e1630b4fb37aa0d96796e8de5f36a5cc063b0551e546b2879bd5b3"
VALIDATOR_SHA256 = "a7bb32ba65221a4ecbb1f2ac89a9373c1697bdd3cf3a2b6b61caec8225855907"

EXPECTED_OUTPUTS = {
    "train32": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1/J1_TRAIN32_LABEL_LEDGER.json",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_TRAIN32_LABEL_LEDGER_READY",
    },
    "eval32": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1/J1_EVAL32_LABEL_LEDGER.json",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_EVAL32_LABEL_LEDGER_READY",
    },
    "result": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1/result.json",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_READY",
    },
    "independent_validation": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1/independent_validation.json",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_VALIDATED",
    },
}


class AuthorityError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AuthorityError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"bound file absent/nonregular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON absent/nonregular: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def verify_binding(binding: Any, *, logical: bool) -> bool:
    expected_keys = {"path", "sha256", "logical_sha256"} if logical else {"path", "sha256"}
    if not isinstance(binding, Mapping) or set(binding) != expected_keys:
        return False
    path = ROOT / str(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        return False
    if logical:
        value = read_json(path)
        return value.get("logical_sha256") == binding["logical_sha256"] == logical_sha256(value)
    return True


def validate_existing_outputs() -> str:
    if not OUT_ROOT.exists():
        return "FRESH_OUTPUT_ROOT_ABSENT"
    require(OUT_ROOT.is_dir() and not OUT_ROOT.is_symlink(), "formal output root is not a regular directory")
    expected_names = {Path(spec["path"]).name for spec in EXPECTED_OUTPUTS.values()}
    observed = {path.name for path in OUT_ROOT.iterdir()}
    require(observed == expected_names, "formal output root is partial or contains unexpected files")
    values = {
        role: read_json(ROOT / spec["path"]) for role, spec in EXPECTED_OUTPUTS.items()
    }
    for role, spec in EXPECTED_OUTPUTS.items():
        require(values[role].get("status") == spec["status"], f"formal {role} status drift")
        require(values[role].get("logical_sha256") == logical_sha256(values[role]), f"formal {role} logical drift")
    train, evaluation, result, validation = (
        values["train32"], values["eval32"], values["result"], values["independent_validation"]
    )
    require(
        result.get("train32_ledger", {}).get("sha256") == sha256_file(ROOT / EXPECTED_OUTPUTS["train32"]["path"])
        and result.get("eval32_ledger", {}).get("sha256") == sha256_file(ROOT / EXPECTED_OUTPUTS["eval32"]["path"])
        and validation.get("train32_ledger_sha256") == sha256_file(ROOT / EXPECTED_OUTPUTS["train32"]["path"])
        and validation.get("eval32_ledger_sha256") == sha256_file(ROOT / EXPECTED_OUTPUTS["eval32"]["path"])
        and validation.get("result_sha256") == sha256_file(ROOT / EXPECTED_OUTPUTS["result"]["path"])
        and validation.get("producer_sha256") == PRODUCER_SHA256
        and validation.get("validator_sha256") == VALIDATOR_SHA256
        and all(validation.get("checks", {}).values())
        and train.get("population") == {"query_count": 32, "identity_count": 12, "supergroup_count": 12, "target_winner_count": 27, "target_nonwinner_count": 5, "target_absent_count": 0}
        and evaluation.get("population") == {"query_count": 32, "identity_count": 11, "supergroup_count": 11, "target_winner_count": 24, "target_nonwinner_count": 8, "target_absent_count": 0}
        and all(count == 0 for block in result.get("three_population_overlap_counts", {}).values() for count in block.values())
        and result.get("model_update_count", result.get("access", {}).get("model_update_count")) == 0
        and validation.get("model_update_count") == 0,
        "formal J1 outputs do not form an exact independently validated closure",
    )
    return "EXISTING_OUTPUTS_EXACTLY_REVALIDATED"


def main() -> None:
    authority = read_json(AUTHORITY)
    require(
        set(authority) == {
            "version", "status", "claim_level", "contract", "inputs", "current64_shards",
            "programs", "authority_validator", "launcher", "outputs", "resource_contract",
            "hardware_identity_is_authority", "access_boundary", "scientific_GO_or_NO_GO",
            "ownership_GO_or_NO_GO", "automatic_stage_advance", "next_authorized_stage",
            "logical_sha256",
        }
        and authority.get("version") == VERSION
        and authority.get("status") == STATUS
        and authority.get("claim_level") == "ENGINEERING_CURRENT64_J1_POSTSEAL_JOIN_EXECUTION_ONLY"
        and authority.get("logical_sha256") == logical_sha256(authority),
        "authority envelope drift",
    )
    require(
        verify_binding(authority.get("contract"), logical=False)
        and authority["contract"]["sha256"] == CONTRACT_SHA256,
        "J1 contract binding drift",
    )
    expected_input_names = {
        "two_gate", "current64_aggregate", "current64_source", "current64_source_validation",
        "n2_label_result", "n2_label_validation", "n2_label_fold0", "n2_label_fold1",
        "n2_label_fold2", "n2_label_fold3", "n2_label_fold4", "pair64_fixed_pairs",
        "pair64_validation", "pair64_training_input_aggregate", "gallery_cache",
        "gallery_repair_registry", "gallery_repair_runtime", "gallery_repair_contract",
    }
    require(set(authority.get("inputs", {})) == expected_input_names, "direct input role set drift")
    for binding in authority.get("inputs", {}).values():
        require(verify_binding(binding, logical="logical_sha256" in binding), "direct input binding drift")
    aggregate = read_json(ROOT / authority["inputs"]["current64_aggregate"]["path"])
    seals = {int(item["shard"]): item for item in aggregate["shards"]}
    require(set(seals) == set(range(8)) and authority.get("current64_shards") == aggregate["shards"], "current64 shard seal list drift")
    for shard, seal in seals.items():
        base = ROOT / f"results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/shard{shard:02d}"
        require(
            seal["payload_sha256"] == sha256_file(base / "payload.pt")
            and seal["receipt_sha256"] == sha256_file(base / "receipt.json")
            and seal["validation_sha256"] == sha256_file(base / "validation.json"),
            f"current64 direct shard {shard} drift",
        )
    require(
        authority.get("programs") == {
            "producer": {"path": str(PRODUCER.relative_to(ROOT)), "sha256": PRODUCER_SHA256},
            "independent_validator": {"path": str(J1_VALIDATOR.relative_to(ROOT)), "sha256": VALIDATOR_SHA256},
        }
        and sha256_file(PRODUCER) == PRODUCER_SHA256
        and sha256_file(J1_VALIDATOR) == VALIDATOR_SHA256,
        "J1 program binding drift",
    )
    require(verify_binding(authority.get("authority_validator"), logical=False), "authority-validator self binding drift")
    require(verify_binding(authority.get("launcher"), logical=False), "launcher binding drift")
    require(authority.get("outputs") == EXPECTED_OUTPUTS, "formal output path/status contract drift")
    require(
        authority.get("resource_contract") == {
            "allowed_partitions": ["cpuonly", "dev_cpuonly"], "cpus_per_task": 4,
            "memory_gib": 32, "walltime_seconds": 600, "gpu_count": 0,
        }
        and authority.get("hardware_identity_is_authority") is False,
        "resource/hardware boundary drift",
    )
    launcher_text = LAUNCHER.read_text()
    authority_offset = launcher_text.find(Path(__file__).name)
    producer_offset = launcher_text.find(PRODUCER.name)
    validator_offset = launcher_text.find(J1_VALIDATOR.name)
    require(
        "#SBATCH --time=00:10:00" in launcher_text
        and "#SBATCH --cpus-per-task=4" in launcher_text
        and "#SBATCH --mem=32G" in launcher_text
        and "#SBATCH --export=NIL" in launcher_text
        and "${SLURM_TMPDIR:-/tmp}" in launcher_text
        and "export PATH=/usr/local/bin:/usr/bin:/bin" in launcher_text
        and 0 <= authority_offset < producer_offset < validator_offset,
        "launcher order/environment/resource contract drift",
    )
    if os.environ.get("SLURM_JOB_ID"):
        require(os.environ.get("SLURM_JOB_PARTITION") in {"cpuonly", "dev_cpuonly"}, "runtime partition not allowed")
        require(os.environ.get("SLURM_CPUS_PER_TASK") == "4", "runtime CPU count drift")
    require(
        authority.get("access_boundary") == {
            "postseal_target_join": True, "target_insertion_count": 0, "model_forward_count": 0,
            "model_update_count": 0, "external_read_count": 0, "sealed_read_count": 0,
            "home_file_modification_count": 0,
        }
        and authority.get("scientific_GO_or_NO_GO") is None
        and authority.get("ownership_GO_or_NO_GO") is None
        and authority.get("automatic_stage_advance") is False
        and authority.get("next_authorized_stage") == NEXT,
        "authority access/claim/next boundary drift",
    )
    output_mode = validate_existing_outputs()
    print(json.dumps({"status": STATUS, "authority_sha256": sha256_file(AUTHORITY), "output_mode": output_mode, "hardware_identity_is_authority": False}, sort_keys=True))


if __name__ == "__main__":
    main()
