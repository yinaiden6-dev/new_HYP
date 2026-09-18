#!/usr/bin/env python3
"""Fail-closed validator for the N2 postseal E0 execution authority."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_POSTSEAL_EVALUATION_EXECUTION_AUTHORITY_V1_20260904.json"
REDUCER = ROOT / "programs/reduce_routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1.py"
E0_VALIDATOR = ROOT / "programs/validate_routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1.py"
LAUNCHER = ROOT / "slurm/routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1_10m.sbatch"
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1"
VERSION = "routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_execution_authority_v1_20260904"
STATUS = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_POSTSEAL_EVALUATION_EXECUTION_AUTHORIZED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_POSTSEAL_EVALUATION_EXECUTION"
REDUCER_SHA = "c2006c14edc8a766a3f30cf3673b3edc519ed88dc4a0e6f333d56627ffe91245"
VALIDATOR_SHA = "f536573f740f8bae37c47a935d88c87f6a741dfdeefb6c26b80439f4f9d61889"
OUTPUTS = {
    "result": {"path": "results/routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1/result.json"},
    "independent_validation": {"path": "results/routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1/independent_validation.json", "status": "N2_FRESH_D1_MATCHED_THREE_ARM_POSTSEAL_EVALUATION_VALIDATED"},
}


class AuthorityError(RuntimeError):
    pass


def req(ok: bool, message: str) -> None:
    if not ok:
        raise AuthorityError(message)


def sha(path: Path) -> str:
    req(path.is_file() and not path.is_symlink(), f"missing/nonregular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canon(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    copy = dict(value)
    copy.pop("logical_sha256", None)
    return canon(copy)


def read(path: Path) -> dict[str, Any]:
    req(path.is_file() and not path.is_symlink(), f"missing JSON: {path}")
    value = json.loads(path.read_text())
    req(isinstance(value, dict), f"non-object JSON: {path}")
    return value


def bound(item: Any) -> bool:
    req(isinstance(item, Mapping) and set(item) in ({"path", "sha256"}, {"path", "sha256", "logical_sha256", "status"}), "binding schema")
    path = ROOT / str(item["path"])
    req(sha(path) == item["sha256"], f"physical drift: {path}")
    if "logical_sha256" in item:
        value = read(path)
        req(value.get("status") == item["status"], f"status drift: {path}")
        req(value.get("logical_sha256") == item["logical_sha256"] == logical(value), f"logical drift: {path}")
    return True


def output_mode() -> str:
    if not OUT_ROOT.exists():
        return "FRESH_OUTPUT_ROOT_ABSENT"
    req(OUT_ROOT.is_dir() and not OUT_ROOT.is_symlink(), "output root type")
    req({p.name for p in OUT_ROOT.iterdir()} == {"result.json", "independent_validation.json"}, "partial/unexpected output")
    result, validation = read(OUT_ROOT / "result.json"), read(OUT_ROOT / "independent_validation.json")
    req(result.get("logical_sha256") == logical(result), "result logical")
    req(result.get("status") in {"N2_FRESH_D1_CANDIDATE_BOUND_ACTION_CONDITIONAL_GO", "N2_FRESH_D1_RETRIEVAL_ONLY_NO_CANDIDATE_BINDING", "N2_FRESH_D1_ACTION_CONDITIONAL_NO_GO"}, "result terminal status")
    req(validation.get("status") == OUTPUTS["independent_validation"]["status"], "validation status")
    req(validation.get("logical_sha256") == logical(validation), "validation logical")
    req(validation.get("result_sha256") == sha(OUT_ROOT / "result.json") and validation.get("result_logical_sha256") == result["logical_sha256"], "validation result binding")
    req(validation.get("reducer_sha256") == REDUCER_SHA and validation.get("validator_sha256") == VALIDATOR_SHA and all(validation.get("checks", {}).values()), "independent closure")
    return "EXISTING_OUTPUTS_EXACTLY_REVALIDATED"


def main() -> None:
    authority = read(AUTHORITY)
    req(authority.get("version") == VERSION and authority.get("status") == STATUS and authority.get("logical_sha256") == logical(authority), "authority envelope")
    req(set(authority.get("inputs", {})) == {"contract", "a0_actions", "a0_validation", "j1_eval_ledger", "j1_public_validation", "raw_c_result", "raw_c_validation", "candidate_recall_result", "candidate_recall_validation", "gallery", "repair_registry", "repair_runtime", "repair_contract"}, "input roles")
    for item in authority["inputs"].values():
        bound(item)
    req(authority.get("programs") == {"reducer": {"path": str(REDUCER.relative_to(ROOT)), "sha256": REDUCER_SHA}, "independent_validator": {"path": str(E0_VALIDATOR.relative_to(ROOT)), "sha256": VALIDATOR_SHA}}, "program bindings")
    req(sha(REDUCER) == REDUCER_SHA and sha(E0_VALIDATOR) == VALIDATOR_SHA, "program physical hashes")
    bound(authority.get("authority_validator")); bound(authority.get("launcher"))
    req(authority.get("outputs") == OUTPUTS, "output contract")
    req(authority.get("resource_contract") == {"allowed_partitions": ["cpuonly", "dev_cpuonly"], "cpus_per_task": 4, "memory_gib": 32, "walltime_seconds": 600, "gpu_count": 0} and authority.get("hardware_identity_is_authority") is False, "resource boundary")
    text = LAUNCHER.read_text()
    first_authority = text.find('"$python_bin" "$authority_validator"')
    reducer_call = text.find('"$python_bin" "$reducer"')
    independent_call = text.find('"$python_bin" "$independent_validator"')
    second_authority = text.rfind('"$python_bin" "$authority_validator"')
    req("#SBATCH --time=00:10:00" in text and "#SBATCH --cpus-per-task=4" in text and "#SBATCH --mem=32G" in text and "#SBATCH --export=NIL" in text and "${SLURM_TMPDIR:-/tmp}" in text and "export PATH=/usr/local/bin:/usr/bin:/bin" in text and 0 <= first_authority < reducer_call < independent_call < second_authority, "launcher contract/order")
    if os.environ.get("SLURM_JOB_ID"):
        req(os.environ.get("SLURM_JOB_PARTITION") in {"cpuonly", "dev_cpuonly"}, "runtime partition")
        req(os.environ.get("SLURM_CPUS_PER_TASK") == "4", "runtime CPUs")
    access = authority.get("access_boundary", {})
    req(access == {"j1_eval_ledger_open_count": 1, "j1_train_ledger_open_count": 0, "pair_label_or_fixed_pair_open_count": 0, "optimizer_or_training_result_mutation_count": 0, "model_update_count": 0, "external_read_count": 0, "sealed_read_count": 0, "home_file_modification_count": 0}, "access boundary")
    req(authority.get("scientific_GO_or_NO_GO") is None and authority.get("ownership_GO_or_NO_GO") is None and authority.get("automatic_stage_advance") is False and authority.get("next_authorized_stage") == NEXT, "claim/next boundary")
    forbidden = "J1_TRAIN32_LABEL_LEDGER.json"
    req(forbidden not in REDUCER.read_text() and forbidden not in E0_VALIDATOR.read_text(), "forbidden TRAIN reference")
    req("native7_training_v1" not in REDUCER.read_text() and "native7_training_v1" not in E0_VALIDATOR.read_text(), "T0 direct reference forbidden")
    print(json.dumps({"status": STATUS, "authority_sha256": sha(AUTHORITY), "output_mode": output_mode()}, sort_keys=True))


if __name__ == "__main__":
    main()
