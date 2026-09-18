#!/usr/bin/env python3
"""Fail-closed validation of the NATIVE7 T0 execution authority."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_EXECUTION_AUTHORITY_V1_20260904.json"
TRAINER = ROOT / "programs/run_routea_n2_fresh_d1_matched_three_arm_native7_training_v1.py"
TRAIN_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_matched_three_arm_native7_training_v1.py"
LAUNCHER = ROOT / "slurm/routea_n2_fresh_d1_matched_three_arm_native7_training_v1_10m.sbatch"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_native7_training_v1"
VERSION = "routea_n2_fresh_d1_matched_three_arm_native7_training_execution_authority_v1_20260904"
STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_EXECUTION_AUTHORIZED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_EXECUTION"
TRAIN_STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_COMPLETE"
VALID_STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_VALIDATED"
FORBIDDEN_EVAL = "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1/J1_EVAL32_LABEL_LEDGER.json"


class AuthorityError(RuntimeError):
    pass


def req(value: bool, message: str) -> None:
    if not value:
        raise AuthorityError(message)


def sha(path: Path) -> str:
    req(path.is_file() and not path.is_symlink(), f"missing/nonregular {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canon(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    copy = dict(value)
    copy.pop("logical_sha256", None)
    return canon(copy)


def read(path: Path) -> dict[str, Any]:
    req(path.is_file() and not path.is_symlink(), f"missing JSON {path}")
    value = json.loads(path.read_text())
    req(isinstance(value, dict), f"JSON root {path}")
    return value


def binding_ok(binding: Any) -> bool:
    req(isinstance(binding, Mapping), "binding object")
    expected = {"path", "sha256"} | ({"logical_sha256"} if "logical_sha256" in binding else set())
    req(set(binding) == expected, "binding keys")
    path = ROOT / str(binding["path"])
    req(sha(path) == binding["sha256"], f"binding physical {path}")
    if "logical_sha256" in binding:
        value = read(path)
        req(value.get("logical_sha256") == binding["logical_sha256"] == logical(value), f"binding logical {path}")
    return True


def output_mode(trainer_sha: str, validator_sha: str) -> str:
    if not OUT_ROOT.exists():
        return "FRESH_OUTPUT_ROOT_ABSENT"
    req(OUT_ROOT.is_dir() and not OUT_ROOT.is_symlink(), "output root type")
    req({p.name for p in OUT_ROOT.iterdir()} == {"heads.pt", "result.json", "independent_validation.json"}, "partial/unexpected formal output")
    result = read(OUT_ROOT / "result.json")
    validation = read(OUT_ROOT / "independent_validation.json")
    req(result.get("status") == TRAIN_STATUS and result.get("logical_sha256") == logical(result), "result seal")
    req(validation.get("status") == VALID_STATUS and validation.get("logical_sha256") == logical(validation), "validation seal")
    req(
        result.get("heads_payload_sha256") == sha(OUT_ROOT / "heads.pt")
        and validation.get("result_sha256") == sha(OUT_ROOT / "result.json")
        and validation.get("result_logical_sha256") == result["logical_sha256"]
        and validation.get("heads_payload_sha256") == sha(OUT_ROOT / "heads.pt")
        and validation.get("trainer_sha256") == trainer_sha
        and validation.get("validator_sha256") == validator_sha
        and all(validation.get("checks", {}).values()),
        "formal output closure",
    )
    return "EXISTING_OUTPUTS_EXACTLY_REVALIDATED"


def main() -> None:
    authority = read(AUTHORITY)
    req(authority.get("version") == VERSION and authority.get("status") == STATUS and authority.get("logical_sha256") == logical(authority), "authority envelope")
    req(authority.get("claim_level") == "ENGINEERING_TRAIN_ONLY_NATIVE7_EXECUTION_NO_EVAL_LABEL_OR_ACTION", "claim")
    for binding in authority.get("inputs", {}).values():
        binding_ok(binding)
    shard_specs = authority.get("input_shards", {})
    req(set(shard_specs) == {"current64", "pair64"}, "shard group set")
    aggregates = {"current64": read(ROOT / authority["inputs"]["current64_aggregate"]["path"]), "pair64": read(ROOT / authority["inputs"]["pair64_training_aggregate"]["path"])}
    for group, spec in shard_specs.items():
        items = aggregates[group].get("shards")
        req(spec == {"count": 8, "canonical_sha256": canon(items)} and isinstance(items, list) and len(items) == 8, "shard group")
        base = ("results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1" if group == "current64" else "results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1")
        for item in items:
            shard = int(item["shard"])
            req(shard in range(8), "shard ordinal")
            directory = ROOT / base / f"shard{shard:02d}"
            req(item["payload_sha256"] == sha(directory / "payload.pt") and item["receipt_sha256"] == sha(directory / "receipt.json") and item["validation_sha256"] == sha(directory / "validation.json"), "shard physical seal")
            validation = read(directory / "validation.json")
            req(item["validation_logical_sha256"] == validation.get("logical_sha256") == logical(validation), "shard logical seal")
    trainer_sha = sha(TRAINER)
    validator_sha = sha(TRAIN_VALIDATOR)
    req(authority.get("programs") == {
        "trainer": {"path": str(TRAINER.relative_to(ROOT)), "sha256": trainer_sha},
        "independent_validator": {"path": str(TRAIN_VALIDATOR.relative_to(ROOT)), "sha256": validator_sha},
    }, "program bindings")
    binding_ok(authority.get("authority_validator"))
    binding_ok(authority.get("launcher"))
    req(authority.get("forbidden_inputs") == {"current_eval_label_ledger": FORBIDDEN_EVAL}, "forbidden input declaration")
    forbidden_basename = Path(FORBIDDEN_EVAL).name
    req(forbidden_basename not in TRAINER.read_text() and forbidden_basename not in TRAIN_VALIDATOR.read_text(), "T0 source references EVAL ledger")
    req(authority.get("outputs") == {
        "heads": {"path": "results/routea_n2_fresh_d1_matched_three_arm_native7_training_v1/heads.pt"},
        "result": {"path": "results/routea_n2_fresh_d1_matched_three_arm_native7_training_v1/result.json", "status": TRAIN_STATUS},
        "independent_validation": {"path": "results/routea_n2_fresh_d1_matched_three_arm_native7_training_v1/independent_validation.json", "status": VALID_STATUS},
    }, "outputs")
    req(authority.get("resource_contract") == {"allowed_partitions": ["cpuonly", "dev_cpuonly"], "default_partition": "cpuonly", "cpus_per_task": 4, "memory_gib": 32, "walltime_seconds": 600, "gpu_count": 0}, "resources")
    text = LAUNCHER.read_text()
    first_authority = text.find('"$python_bin" "$authority_validator"')
    trainer_offset = text.find('"$python_bin" "$trainer"')
    train_validator_offset = text.find('"$python_bin" "$independent_validator"')
    final_authority = text.rfind('"$python_bin" "$authority_validator"')
    req(
        "#SBATCH --partition=cpuonly" in text and "#SBATCH --cpus-per-task=4" in text
        and "#SBATCH --mem=32G" in text and "#SBATCH --time=00:10:00" in text
        and "#SBATCH --export=NIL" in text and "${SLURM_TMPDIR:-/tmp}" in text
        and "export PATH=/usr/local/bin:/usr/bin:/bin" in text
        and 0 <= first_authority < trainer_offset < train_validator_offset < final_authority,
        "launcher contract/order",
    )
    if os.environ.get("SLURM_JOB_ID"):
        req(os.environ.get("SLURM_JOB_PARTITION") in {"cpuonly", "dev_cpuonly"}, "partition")
        req(os.environ.get("SLURM_CPUS_PER_TASK") == "4", "cpus")
    req(authority.get("access_boundary") == {
        "train_label_ledger_open_count": 1,
        "current_eval_label_file_open_count": 0,
        "train_target_identity_semantic_consumption_count": 0,
        "train_target_supergroup_semantic_consumption_count": 0,
        "model_update_count": 6000,
        "external_read_count": 0,
        "sealed_read_count": 0,
    }, "access boundary")
    req(authority.get("hardware_identity_is_authority") is False and authority.get("scientific_GO_or_NO_GO") is None and authority.get("automatic_stage_advance") is False and authority.get("next_authorized_stage") == NEXT, "claim/next")
    print(json.dumps({"status": STATUS, "authority_sha256": sha(AUTHORITY), "output_mode": output_mode(trainer_sha, validator_sha)}, sort_keys=True))


if __name__ == "__main__":
    main()
