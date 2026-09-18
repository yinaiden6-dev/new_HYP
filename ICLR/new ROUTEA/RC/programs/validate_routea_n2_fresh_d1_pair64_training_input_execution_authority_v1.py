#!/usr/bin/env python3
"""Fail-closed skeleton for the Pair64 training-input execution authority.

This file intentionally cannot validate a final authority until M0, its
aggregate validator, and all launchers are frozen. Future P0/J0/M0 outputs
are bound by path/version/status only; each stage validator dynamically seals
the preceding stage's physical and logical hashes after it exists.
It writes nothing.  Hardware identity is never an authority input; only the
declared execution-only partition alternatives are checked at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_AUTHORITY_V1_20260904.json"
VERSION = "routea_n2_fresh_d1_matched_three_arm_pair64_training_input_execution_authority_v1_20260904"
AUTHORIZED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_AUTHORIZED"
SKELETON_WAIT = "ROUTEA_N2_FRESH_D1_PAIR64_AUTHORITY_VALIDATOR_SKELETON_WAITING_FINAL_HASHES"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_EXECUTION"

ACTIVE_CONTRACTS = {
    "active_contract": (
        "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUT_EXECUTION_CONTRACT_V1_20260904.md",
        "bc53ef74ec71283d79247a2117f25e2c708c2d30bb6fda1fad194b5668c97ade",
    ),
    "scope_correction": (
        "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_PAIR_SCOPE_CORRECTIVE_ADDENDUM_V1_20260904.md",
        "19fa1ca3e5b898adec22c9148c397d7765af48aa23a5057d155115649fdbe124",
    ),
    "label_correction": (
        "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_LABEL_AUTHORITY_CORRECTION_ADDENDUM_V1_20260904.md",
        "9c94f79cb718e7a111a983fa045814028cba1d737553a9c60daa46f118f39035",
    ),
    "highest_precedence_disposition": (
        "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_ACTIVE_CONTRACT_DISPOSITION_AND_EXECUTION_CLARIFICATION_V1_20260904.md",
        "502de768d26584613bb89f329e214e431d959e821fe6caa3ab5cd1cb5ad538b8",
    ),
}
SUPERSEDED_HASHES = {
    "07cff47995dfa838f6465c856b05aadbf787f3902ac09ccab1f17436fbd98ac8",
    "aacf113cdd31902a3a2d0d2cab0df7d91824a4e2b75a37626be6ac57ee20e759",
}

# None is deliberate for future code/launchers: the validator must remain
# unusable until each executable is immutable. Future result files do not
# belong here; their path/version/status contracts live in OUTPUT_PATHS.
ARTIFACT_SLOTS: dict[str, tuple[str | None, str | None]] = {
    "pair64_v2_payload": (
        "results/routea_matched_three_arm_pair64_training_features_v2/payload.pt",
        "d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3",
    ),
    "pair64_v2_receipt": (
        "results/routea_matched_three_arm_pair64_training_features_v2/receipt.json",
        "493f49f61801790ac5e99e0378c0b2e47972eb9919856e17047f543d439bd922",
    ),
    "pair64_v2_lineage": (
        "results/routea_matched_three_arm_pair64_training_features_v2/post_full_lineage.json",
        "76be3c0cfc839cbe2029e8f7b25f030be15fdb5c37c752ccf9e21e248da33d4e",
    ),
    "pair64_v2_validation": (
        "results/routea_matched_three_arm_pair64_training_features_v2/independent_validation.json",
        "657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b",
    ),
    "balanced32_v1": (
        "results/romav2_colnomic_visibility_xf_balanced32_v1/result.json",
        "77367e63ac835bb38340c1ae9c9d16018fde1c22b16a14a12f346d56201e404a",
    ),
    "balanced32_v2": (
        "results/romav2_colnomic_visibility_xf_balanced32_v2/result.json",
        "fd02c73f508f3f16acbafefb5b703dc864a7793ff7f688ff70efffdffa80f108",
    ),
    "rolefree_pair_address": (
        "results/dino_rcde_sr0_mt_role_free_pair_address_v1/role_free_pair_address_manifest.json",
        "6f9999053068a50b0b8f890a11c24987e77532ed31b5361c16c16956b9564e0b",
    ),
    "canonical_query_ledger": (
        "cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json",
        "df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec",
    ),
    "fresh_d1_oof_aggregate": (
        "results/routea_matched_three_arm_n2_current_runtime_d1_oof_prejoin_v1/independent_aggregate_validation.json",
        "fc5ac9422ae6f2fe6485b906525f7ddccd316ba9b77c481bb65f679ea706982f",
    ),
    "gallery_cache": (
        "../../../colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt",
        "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    ),
    "gallery_repair_registry": (
        "registry/gallery_identity_repair_v1.json",
        "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f",
    ),
    "gallery_repair_runtime": (
        "src/rc_aslo_xf/gallery_identity_repair.py",
        "995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d",
    ),
    "gallery_repair_contract": (
        "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json",
        "867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650",
    ),
    "n2_label_result": (
        "results/routea_matched_three_arm_n2_d1_label_join_authority_v1/result.json",
        "519cea43968abf8083f3d6c4b98e108d592f29976c470b01f464054d7c9a5861",
    ),
    "n2_label_validation": (
        "results/routea_matched_three_arm_n2_d1_label_join_authority_v1/independent_validation.json",
        "0b78e01d6940c19db40d606ce6012be190628b31d56e71851c22a97e83cfe3a8",
    ),
    "n2_label_fold0": ("results/routea_matched_three_arm_n2_d1_label_join_authority_v1/fold_0/payload.pt", "0a9c72582f0af05b9a83d28a6217ef62670a2a89b7bf433247e35e46d0e983f3"),
    "n2_label_fold1": ("results/routea_matched_three_arm_n2_d1_label_join_authority_v1/fold_1/payload.pt", "c411861a4ca6b8f41792893f747d5f66561af718c6681b3d4c4bbece6b40ab72"),
    "n2_label_fold2": ("results/routea_matched_three_arm_n2_d1_label_join_authority_v1/fold_2/payload.pt", "08b89f23d61012460da86191951202dbdb4e7aa4cffb75e2c673e5bb8c5f9a0f"),
    "n2_label_fold3": ("results/routea_matched_three_arm_n2_d1_label_join_authority_v1/fold_3/payload.pt", "a559a678fe056353de607ba92b2bda1e675470be7a477ccdc54513c1395afb94"),
    "n2_label_fold4": ("results/routea_matched_three_arm_n2_d1_label_join_authority_v1/fold_4/payload.pt", "d6014cb1c141efd7cb57a19eed9692b406efddf8ef0f52449690e06b1090c6b3"),
    "cw0_source_manifest": (
        "results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json",
        "e0be35125eddec391a92398e84103e77ca0a65f661452b9190c2b81f3fdbea23",
    ),
    "cw0_role_manifest": (
        "results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_manifest.json",
        "2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454",
    ),
    "cw0_result": (
        "results/cw0_rgh_xf_v2_p0_a0_manifest_v2/result.json",
        "1c3cd17e1349a84be7fb7af5d5b796874f3ec3d7a61539b3c6e535dddac04e08",
    ),
    "cw0_validation": (
        "results/cw0_rgh_xf_v2_p0_a0_manifest_v2/independent_validation.json",
        "ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac",
    ),
    "current64_source_manifest": (
        "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/manifest.json",
        "da7ff4c0ee24bb72d1fa026e3e103e01e3cf3f9cd489e8b33f26835b4065ab58",
    ),
    "current64_source_validation": (
        "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/independent_validation.json",
        "654453f5884fc2bc1848a54fe67cde37947f01cdc1b6493d3e764cda4cfb737e",
    ),
    "roma_weights": (
        "../../../third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt",
        "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7",
    ),
    "processor_config": (
        "../../../models/downloaded_models/colnomic-embed-multimodal-7b/preprocessor_config.json",
        "1a427e12a15406a5c4701273c91a0b319528de5c78523ae5ba5329d86e5d5557",
    ),
    "reference_resolver_source": (
        "src/rc_aslo_xf/cw1_sr0_s8_feature_runtime_v1.py",
        "5ca8bea5d07b07000401e25eda6f70178ad5952061988574f67f64f06300ee78",
    ),
    "current64_aggregate": (
        "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1/validation.json",
        "24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859",
    ),
    "p0_producer": (
        "programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py",
        "068c8297a0b773c4efefa093f2887393588213dce7ca347cb5d6517cc22e13b5",
    ),
    "p0_validator": (
        "programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py",
        "239f67385076b72b3f28c4223a5c4208f7ee64ba90c22008d618aaa627adf6db",
    ),
    "j0_producer": (
        "programs/materialize_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py",
        "5ef394b3cbc1688b8315b7ee7c2ad625a29f255b82d1018e48c4d4a3f5267fd4",
    ),
    "j0_validator": (
        "programs/validate_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py",
        "a536c5fdcc37834dd78a6cc8b18a7f13d76c14091db625533059291e88ad8c16",
    ),
    "m0_producer": (
        "programs/materialize_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py",
        "3e77fe479c9ccf1c4c7ca60a4c01e6b0947f7717609ba431755979d3e630fc2e",
    ),
    "m0_validator": (
        "programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py",
        "d8de67ccc9e5493157fa6aa64c57572927a72bda473186e2ef0eb620a8c6e415",
    ),
    "m0_aggregate_validator": (
        "programs/validate_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_aggregate_v1.py",
        "ef3596061e6929c02e919d214d53e9630a6d322105c068ff34dec79b9f4e01bf",
    ),
    "p0_j0_setup_launcher": (
        "slurm/routea_n2_fresh_d1_pair64_p0_j0_setup_v1_30m.sbatch",
        "24498a73952e6aa0bd3528fbf79c6f36bf2e86148cd85ceb0c1c3db7a76b8e1a",
    ),
    "m0_array_launcher": (
        "slurm/routea_n2_fresh_d1_pair64_training_features_array_v1_2h.sbatch",
        "396eda74f367a8c2960d2762651766e3ca0b682901a7361d8819170f9ad950c7",
    ),
    "aggregate_launcher": (
        "slurm/routea_n2_fresh_d1_pair64_training_features_aggregate_v1_30m.sbatch",
        "44eb979170e49fccb292b17ee995ea49d74fae03275ce7e83e98a30998808aad",
    ),
}

OUTPUT_PATHS = {
    "pair64_p0_preseal": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1/preseal.json",
        "version": "routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1_20260904",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_READY",
    },
    "pair64_p0_independent_validation": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1/independent_validation.json",
        "version": "routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1_20260904",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_BASE_PRESEAL_VALIDATED",
    },
    "pair64_j0_fixed_pairs": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1/fixed_pairs.json",
        "version": "routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1_20260904",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_READY",
    },
    "pair64_j0_independent_validation": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1/independent_validation.json",
        "version": "routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1_20260904",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_POSTSEAL_FIXED_TRAINING_PAIRS_VALIDATED",
    },
    "pair64_m0_shard_pattern": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1/shard{shard:02d}/{payload.pt,receipt.json,validation.json}",
        "version": "routea_n2_fresh_d1_matched_three_arm_pair64_training_feature_shard_v1_20260904",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURE_SHARD_VALIDATED",
    },
    "pair64_m0_aggregate_training_input_handoff": {
        "path": "results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1/independent_aggregate_validation.json",
        "version": "routea_n2_fresh_d1_matched_three_arm_pair64_training_features_aggregate_v1_20260904",
        "status": "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_PAIR64_TRAINING_INPUTS_VALIDATED",
    },
}

STAGE_TOPOLOGY = [
    "CURRENT64_TARGET_FREE_FULL_C128_AGGREGATE_VALIDATED",
    "PAIR64_P0_TARGET_FREE_BASE_PRESEAL",
    "PAIR64_P0_INDEPENDENT_VALIDATION",
    "PAIR64_J0_POSTSEAL_FIXED_PAIR_JOIN",
    "PAIR64_J0_INDEPENDENT_VALIDATION",
    "PAIR64_M0_POSTSEAL_TRAINING_ONLY_PAIR_FEATURES",
    "PAIR64_M0_INDEPENDENT_VALIDATION",
    "CURRENT64_AND_PAIR64_TRAINING_INPUT_AGGREGATE_VALIDATION",
]
RESOURCE_CONTRACT = {
    "p0_j0_setup": {
        "allowed_partitions": ["cpuonly", "dev_cpuonly"],
        "walltime_seconds": 600,
        "cpus_per_task": 4,
        "memory_gib": 32,
        "gpu_count": 0,
    },
    "m0_array": {
        "allowed_partitions": ["accelerated", "dev_accelerated"],
        "walltime_seconds": 600,
        "cpus_per_task": 8,
        "memory_gib": 96,
        "gpu_count": 1,
        "canonical_array_tasks": "0-7",
        "canonical_array_task_count": 8,
        "default_max_concurrency": 8,
        "dev_max_concurrency": 4,
    },
    "aggregate": {
        "allowed_partitions": ["cpuonly", "dev_cpuonly"],
        "walltime_seconds": 600,
        "cpus_per_task": 4,
        "memory_gib": 32,
        "gpu_count": 0,
    },
}


class AuthorityValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityValidationError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"bound file absent/non-regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("logical_sha256", None)
    return canonical_sha256(payload)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"JSON absent: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not object: {path}")
    return value


def verify_binding(binding: Any, relative_path: str, expected_sha256: str) -> bool:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        return False
    return (
        binding.get("path") == relative_path
        and binding.get("sha256") == expected_sha256
        and sha256_file(ROOT / relative_path) == expected_sha256
    )


def unfinished_slots() -> list[str]:
    artifacts = [
        name
        for name, (path, digest) in ARTIFACT_SLOTS.items()
        if path is None or digest is None
    ]
    outputs = [
        f"output:{name}"
        for name, spec in OUTPUT_PATHS.items()
        if any(spec.get(field) is None for field in ("path", "version", "status"))
    ]
    return sorted(artifacts + outputs)


def main() -> None:
    for _, (relative, expected) in ACTIVE_CONTRACTS.items():
        require(sha256_file(ROOT / relative) == expected, f"active contract drift: {relative}")
    missing = unfinished_slots()
    if missing:
        print(
            json.dumps(
                {
                    "status": SKELETON_WAIT,
                    "active_pair64_contract_count": 1,
                    "missing_final_hash_slots": missing,
                    "authority_json_generated": False,
                    "automatic_stage_advance": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(3)

    # No code below this point is reachable in the skeleton state.
    authority = read_json(AUTHORITY)
    expected_keys = {
        "version", "status", "claim_level", "active_pair64_contract_count", "active_contracts",
        "superseded_contract_hashes_rejected", "stage_topology", "artifacts",
        "authority_validator", "outputs", "resource_contract", "hardware_identity_is_authority",
        "access_topology", "scientific_GO_or_NO_GO", "ownership_GO_or_NO_GO",
        "automatic_stage_advance", "next_authorized_stage", "logical_sha256",
    }
    require(set(authority) == expected_keys, "authority top-level schema drift")
    require(
        authority.get("version") == VERSION
        and authority.get("status") == AUTHORIZED
        and authority.get("active_pair64_contract_count") == 1
        and authority.get("claim_level") == "ENGINEERING_PAIR64_TRAINING_INPUT_PREPARATION_ONLY_NO_HEAD_TRAINING"
        and authority.get("logical_sha256") == logical_sha256(authority),
        "authority envelope drift",
    )
    require(authority.get("active_contracts") == {
        name: {"path": path, "sha256": digest}
        for name, (path, digest) in ACTIVE_CONTRACTS.items()
    }, "active contract binding drift")
    require(
        set(authority.get("superseded_contract_hashes_rejected", [])) == SUPERSEDED_HASHES
        and not (SUPERSEDED_HASHES & {binding["sha256"] for binding in authority["active_contracts"].values()}),
        "superseded contract rejection failed",
    )
    require(authority.get("stage_topology") == STAGE_TOPOLOGY, "stage topology drift")
    require(
        verify_binding(
            authority.get("authority_validator"),
            str(Path(__file__).resolve().relative_to(ROOT)),
            sha256_file(Path(__file__).resolve()),
        ),
        "authority-validator self binding drift",
    )
    artifacts = authority.get("artifacts", {})
    require(set(artifacts) == set(ARTIFACT_SLOTS), "artifact role set drift")
    require(all(verify_binding(artifacts[name], path, digest) for name, (path, digest) in ARTIFACT_SLOTS.items()), "artifact binding drift")  # type: ignore[arg-type]
    require(authority.get("outputs") == OUTPUT_PATHS, "authority output path set drift")
    launcher_programs = {
        "p0_j0_setup_launcher": [
            "materialize_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py",
            "validate_routea_n2_fresh_d1_matched_three_arm_pair64_base_preseal_v1.py",
            "materialize_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py",
            "validate_routea_n2_fresh_d1_pair64_postseal_fixed_pairs_v1.py",
        ],
        "m0_array_launcher": [
            "materialize_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py",
            "validate_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_shard_v1.py",
        ],
        "aggregate_launcher": [
            "validate_routea_n2_fresh_d1_matched_three_arm_pair64_training_features_aggregate_v1.py",
        ],
    }
    for role, program_names in launcher_programs.items():
        launcher_path = ROOT / ARTIFACT_SLOTS[role][0]  # type: ignore[arg-type]
        text = launcher_path.read_text()
        authority_offset = text.find(Path(__file__).name)
        require(
            "#SBATCH --time=00:10:00" in text
            and "#SBATCH --export=NIL" in text
            and authority_offset >= 0
            and all(text.find(name) > authority_offset for name in program_names),
            f"launcher {role} does not validate authority before execution",
        )
    setup_text = (ROOT / ARTIFACT_SLOTS["p0_j0_setup_launcher"][0]).read_text()  # type: ignore[arg-type]
    m0_text = (ROOT / ARTIFACT_SLOTS["m0_array_launcher"][0]).read_text()  # type: ignore[arg-type]
    aggregate_text = (ROOT / ARTIFACT_SLOTS["aggregate_launcher"][0]).read_text()  # type: ignore[arg-type]
    require(
        "#SBATCH --cpus-per-task=4" in setup_text
        and "#SBATCH --mem=32G" in setup_text
        and "#SBATCH --cpus-per-task=8" in m0_text
        and "#SBATCH --mem=96G" in m0_text
        and "#SBATCH --gres=gpu:1" in m0_text
        and "#SBATCH --array=0-7%8" in m0_text
        and "#SBATCH --cpus-per-task=4" in aggregate_text
        and "#SBATCH --mem=32G" in aggregate_text,
        "launcher resources differ from frozen resource contract",
    )
    require(
        authority.get("resource_contract") == RESOURCE_CONTRACT
        and authority.get("hardware_identity_is_authority") is False,
        "resource/hardware authority drift",
    )
    forbidden_hardware_keys = {"gpu_name", "gpu_model", "node", "hostname", "cuda_device", "compute_capability"}
    require(not (forbidden_hardware_keys & set(authority)), "hardware identity entered authority")
    if os.environ.get("SLURM_JOB_ID"):
        stage = os.environ.get("ROUTEA_PAIR64_AUTHORITY_STAGE")
        require(stage in RESOURCE_CONTRACT, "runtime authority stage absent")
        resource = RESOURCE_CONTRACT[stage]
        require(os.environ.get("SLURM_JOB_PARTITION") in resource["allowed_partitions"], "runtime partition outside execution-only alternatives")
        require(int(os.environ.get("SLURM_CPUS_PER_TASK", "-1")) == resource["cpus_per_task"], "runtime CPU allocation drift")
        if stage == "m0_array":
            require(
                os.environ.get("SLURM_ARRAY_TASK_MIN") == "0"
                and os.environ.get("SLURM_ARRAY_TASK_MAX") == "7"
                and os.environ.get("SLURM_ARRAY_TASK_COUNT") == "8",
                "M0 runtime array is not canonical tasks 0-7",
            )
    require(
        authority.get("access_topology") == {
            "p0": {"target_read_count": 0, "model_update_count": 0},
            "j0": {"postseal_target_join": True, "target_insertion_count": 0, "model_update_count": 0},
            "m0": {
                "postseal_training_only": True,
                "semantic_endpoint_read_count": 128,
                "switch_label_read_count": 64,
                "target_identity_read_count": 0,
                "target_supergroup_read_count": 0,
                "cw0_read_count": 0,
                "model_update_count": 0,
            },
            "aggregate": {"head_training_authorized": False, "external_read_count": 0, "sealed_read_count": 0},
        },
        "P0/J0/M0 access topology drift",
    )
    require(
        authority.get("scientific_GO_or_NO_GO") is None
        and authority.get("ownership_GO_or_NO_GO") is None
        and authority.get("automatic_stage_advance") is False
        and authority.get("next_authorized_stage") == NEXT,
        "claim/transition boundary drift",
    )
    print(json.dumps({"status": AUTHORIZED, "authority_sha256": sha256_file(AUTHORITY), "hardware_identity_is_authority": False}, sort_keys=True))


if __name__ == "__main__":
    main()
