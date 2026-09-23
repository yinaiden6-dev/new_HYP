"""Fail-closed authority and source lineage for Track-R V122 utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_RELATIVE = "registry/current_authority_v122_20260821.json"
AUTHORITY_SCHEMA = "rc_current_authority_v122_20260821"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_SHARDS_AUTHORIZED"
EXCLUDED = (25, 26, 101, 346, 354, 470)
RUNTIME_ROOTS = (
    "programs/materialize_dino_rcde_track_r_p_v2_utility_shard_v1.py",
    "programs/validate_dino_rcde_track_r_p_v2_utility_shard_v1.py",
    "programs/reduce_dino_rcde_track_r_p_v2_utility_v1.py",
    "programs/validate_dino_rcde_track_r_p_v2_utility_comparator_v1.py",
    "programs/dino_rcde_track_r_p_v2_utility_runtime_v1.py",
    "programs/validate_dino_rcde_track_r_p_v2_utility_authority_v122.py",
    "src/rc_aslo_xf/__init__.py",
)


class V122RuntimeError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise V122RuntimeError(message)


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical({key: item for key, item in value.items() if key != "logical_sha256"})


def fsha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(relative: str) -> Path:
    raw = Path(relative)
    require(not raw.is_absolute() and ".." not in raw.parts, f"unsafe V122 path: {relative}")
    path = (ROOT / raw).resolve(strict=True)
    require(
        path.is_relative_to(ROOT)
        and path.is_file()
        and not path.is_symlink()
        and not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in path.parts
        ),
        f"unsafe/protected V122 file: {relative}",
    )
    return path


def verify_binding(binding: Any, expected: str | None = None) -> Path:
    require(isinstance(binding, Mapping), "V122 binding absent")
    relative = binding.get("path")
    require(
        isinstance(relative, str) and (expected is None or relative == expected),
        f"V122 binding path drift: {expected}",
    )
    path = safe(relative)
    require(
        binding.get("bytes") == path.stat().st_size
        and binding.get("sha256") == fsha(path),
        f"V122 binding hash drift: {relative}",
    )
    if "logical_sha256" in binding:
        value = json.loads(path.read_text(encoding="utf-8"))
        require(
            isinstance(value, dict)
            and value.get("logical_sha256") == logical(value) == binding["logical_sha256"],
            f"V122 logical binding drift: {relative}",
        )
    return path


def validate_authority(
    authority_path: Path,
    *,
    required_flag: str,
    runtime_binding: str,
    runtime_path: str,
) -> tuple[dict[str, Any], str]:
    path = authority_path.resolve(strict=True)
    require(path == (ROOT / AUTHORITY_RELATIVE).resolve(strict=True), "V122 authority path drift")
    require(path.is_file() and not path.is_symlink() and (path.stat().st_mode & 0o777) == 0o444, "V122 authority is not immutable")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        isinstance(value, dict)
        and value.get("schema_version") == AUTHORITY_SCHEMA
        and value.get("status") == AUTHORITY_STATUS
        and value.get("logical_sha256") == logical(value)
        and value.get(required_flag) is True
        and value.get("target_rival_read_authorized") is False
        and value.get("label_identity_supergroup_read_authorized") is False
        and value.get("dino_token_read_authorized") is False
        and value.get("query_count") == 600
        and value.get("eligible_query_count") == 594
        and value.get("excluded_execution_ordinals") == list(EXCLUDED)
        and value.get("shard_count") == 50
        and value.get("shard_size") == 12
        and value.get("p_model_backward_update_authorized") is False
        and value.get("scientific_reduction_authorized") is False
        and value.get("protected_access_authorized") is False
        and value.get("scientific_GO_or_NO_GO") is None,
        "V122 authority boundary drift",
    )
    bindings = value.get("bindings")
    require(isinstance(bindings, Mapping), "V122 bindings absent")
    verify_binding(bindings.get(runtime_binding), runtime_path)
    verify_binding(
        bindings.get("launcher"),
        "slurm/dino_rcde_track_r_p_v2_utility_shards_v1.sbatch",
    )
    closure = bindings.get("runtime_import_closure")
    require(
        isinstance(closure, Mapping)
        and closure.get("root_programs") == list(RUNTIME_ROOTS)
        and isinstance(closure.get("modules"), list)
        and closure.get("module_count") == len(closure["modules"])
        and closure.get("module_sequence_sha256")
        == logical({"rows": closure["modules"]}),
        "V122 runtime closure envelope drift",
    )
    for row in closure["modules"]:
        verify_binding(row)
    edges = closure.get("import_edges")
    require(
        isinstance(edges, list)
        and closure.get("import_edge_sequence_sha256") == logical({"rows": edges}),
        "V122 runtime import-edge drift",
    )
    for name in (
        "parent_authority_v121",
        "oof_prejoin_completion",
        "lock_manifest_index",
        "lock_manifest_validation",
        "fold_schedule",
        "full600_payload",
    ):
        verify_binding(bindings.get(name))
    source_manifests = bindings.get("source_shard_manifests")
    require(
        isinstance(source_manifests, Mapping)
        and source_manifests.get("count") == 50
        and isinstance(source_manifests.get("rows"), list)
        and len(source_manifests["rows"]) == 50
        and source_manifests.get("sequence_sha256")
        == logical({"rows": source_manifests["rows"]}),
        "V122 source-shard manifest binding drift",
    )
    for ordinal, row in enumerate(source_manifests["rows"]):
        require(row.get("shard_ordinal") == ordinal, "V122 source-shard order drift")
        verify_binding(row.get("execution_manifest"))
        verify_binding(row.get("source_descriptor"))
        artifact = row.get("source_artifact")
        require(isinstance(artifact, Mapping), "V122 source artifact receipt absent")
        artifact_path = safe(str(artifact.get("path")))
        require(
            artifact.get("bytes") == artifact_path.stat().st_size
            and isinstance(artifact.get("sha256"), str)
            and len(artifact["sha256"]) == 64
            and (artifact_path.stat().st_mode & 0o777) == 0o444,
            "V122 source artifact receipt drift",
        )
    for fold in (1, 2, 3, 4):
        verify_binding(bindings.get(f"p_outer_refit_checkpoint_fold{fold}"))
        verify_binding(bindings.get(f"p_outer_refit_manifest_fold{fold}"))
    resource = value.get("resource_contract")
    require(
        isinstance(resource, Mapping)
        and resource.get("partition") == "cpuonly"
        and resource.get("cpus_per_task") == 4
        and resource.get("memory_megabytes") == 65536
        and resource.get("walltime_seconds") == 14400
        and resource.get("launcher_header_walltime_seconds") == 21600
        and resource.get("scheduler_time_limit_override_required") is True
        and resource.get("scheduler_override_job_ids") == [5102240, 5102486]
        and resource.get("staged_arrays") == ["0", "1-49%8"],
        "V122 resource/staging contract drift",
    )
    require(
        value.get("authority_revision") == 5
        and value.get("resource_contract_revision") == 2
        and value.get("resource_repair_scope") == "SCHEDULER_TIME_LIMIT_ONLY_4H"
        and value.get("resource_validation_repair_scope")
        == "INDEPENDENT_RESOURCE_EVIDENCE_REPLAY"
        and value.get("resource_evidence_hardening_scope")
        == "FULL_PATH_AND_SCHEDULER_RECEIPT_REPLAY"
        and value.get("full_shard0_canary_job_id") == 5102372,
        "V122 resource revision envelope drift",
    )
    verify_binding(
        bindings.get("resource_addendum"),
        "plan/DINO_RCDE_TRACK_R_V122_4H_SCHEDULER_RESOURCE_ADDENDUM_V1_20260823.md",
    )
    verify_binding(
        bindings.get("resource_validation_repair_addendum"),
        "plan/DINO_RCDE_TRACK_R_V122_4H_INDEPENDENT_VALIDATION_REPAIR_ADDENDUM_V1_20260823.md",
    )
    verify_binding(
        bindings.get("resource_evidence_hardening_addendum"),
        "plan/DINO_RCDE_TRACK_R_V122_4H_FULL_EVIDENCE_REPLAY_ADDENDUM_V1_20260823.md",
    )
    prior_path = verify_binding(
        bindings.get("resource_parent_authority"),
        "registry/archive/current_authority_v122_revision2_before_4h_20260823.json",
    )
    prior_validation_path = verify_binding(
        bindings.get("resource_parent_authority_validation"),
        "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/archive/result_revision2_before_4h.json",
    )
    canary_path = verify_binding(
        bindings.get("full_shard0_canary_validation"),
        "results/dino_rcde_track_r_p_v2_full_shard0_canary_v1/validation_shard_000_012.json",
    )
    revision3_path = verify_binding(
        bindings.get("resource_authority_revision3"),
        "registry/archive/current_authority_v122_revision3_before_independent_4h_validation_20260823.json",
    )
    revision3_validation_path = verify_binding(
        bindings.get("resource_authority_revision3_validation"),
        "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/archive/result_revision3_before_independent_4h_validation.json",
    )
    revision4_path = verify_binding(
        bindings.get("resource_authority_revision4"),
        "registry/archive/current_authority_v122_revision4_before_full_resource_evidence_replay_20260823.json",
    )
    revision4_validation_path = verify_binding(
        bindings.get("resource_authority_revision4_validation"),
        "results/dino_rcde_track_r_p_v2_utility_authority_validation_v1/archive/result_revision4_before_full_resource_evidence_replay.json",
    )
    receipt_path = verify_binding(
        bindings.get("full_shard0_canary_scheduler_receipt"),
        "results/dino_rcde_track_r_p_v2_full_shard0_canary_v1/scheduler_receipt_job5102372.json",
    )
    stdout_path = verify_binding(
        bindings.get("full_shard0_canary_stdout"),
        "logs/rcde_tr_puf-5102372.out",
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_validation = json.loads(prior_validation_path.read_text(encoding="utf-8"))
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    revision3 = json.loads(revision3_path.read_text(encoding="utf-8"))
    revision3_validation = json.loads(
        revision3_validation_path.read_text(encoding="utf-8")
    )
    revision4 = json.loads(revision4_path.read_text(encoding="utf-8"))
    revision4_validation = json.loads(
        revision4_validation_path.read_text(encoding="utf-8")
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        prior.get("authority_revision") == 2
        and prior.get("resource_contract", {}).get("walltime_seconds") == 21600
        and prior.get("logical_sha256") == logical(prior)
        and value.get("resource_parent_authority_sha256") == fsha(prior_path)
        and prior_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS"
        and prior_validation.get("authority_sha256") == fsha(prior_path)
        and canary.get("status")
        == "V122_FULL_SHARD0_CANARY_INDEPENDENT_REPLAY_PASS"
        and canary.get("validation_pass") is True
        and canary.get("authority_sha256") == fsha(prior_path)
        and canary.get("query_count") == 12
        and canary.get("producer_model_forward_count") == 3072
        and canary.get("replay_model_forward_count") == 3072
        and canary.get("formal_write_count") == 0
        and float(canary.get("producer_elapsed_seconds"))
        + float(canary.get("replay_elapsed_seconds"))
        < 14400.0,
        "V122 independent four-hour evidence replay drift",
    )
    require(
        revision3.get("authority_revision") == 3
        and revision3.get("resource_contract", {}).get("walltime_seconds") == 14400
        and revision3.get("logical_sha256") == logical(revision3)
        and value.get("resource_validation_parent_authority_sha256")
        == fsha(revision3_path)
        and revision3_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS"
        and revision3_validation.get("authority_sha256") == fsha(revision3_path)
        and receipt.get("status")
        == "V122_FULL_SHARD0_CANARY_SCHEDULER_RECEIPT_CLOSED"
        and receipt.get("logical_sha256") == logical(receipt)
        and receipt.get("job_id") == 5102372
        and receipt.get("partition") == "dev_cpuonly"
        and receipt.get("state") == "COMPLETED"
        and receipt.get("exit_code") == "0:0"
        and receipt.get("time_limit_seconds") == 14400
        and 0 < receipt.get("elapsed_seconds") < 14400
        and receipt.get("stdout_sha256") == fsha(stdout_path)
        and receipt.get("validation_sha256") == fsha(canary_path),
        "V122 revision/scheduler evidence replay drift",
    )
    require(
        revision4.get("authority_revision") == 4
        and revision4.get("resource_contract", {}).get("walltime_seconds") == 14400
        and revision4.get("logical_sha256") == logical(revision4)
        and value.get("resource_evidence_parent_authority_sha256")
        == fsha(revision4_path)
        and revision4_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_AUTHORITY_INDEPENDENT_VALIDATION_PASS"
        and revision4_validation.get("authority_sha256") == fsha(revision4_path),
        "V122 revision-4 evidence-hardening lineage drift",
    )
    launcher = verify_binding(
        bindings.get("launcher"),
        "slurm/dino_rcde_track_r_p_v2_utility_shards_v1.sbatch",
    )
    launcher_text = launcher.read_text(encoding="utf-8")
    require(
        launcher_text.count("#SBATCH --time=06:00:00") == 1
        and "#SBATCH --time=04:00:00" not in launcher_text,
        "V122 retained submitted-spool launcher header drift",
    )
    return value, fsha(path)


def source_bindings(authority: Mapping[str, Any], authority_sha256: str) -> dict[str, Any]:
    bindings = authority["bindings"]
    return {
        "authority_sha256": authority_sha256,
        "authority_logical_sha256": authority["logical_sha256"],
        "parent_authority_v121_sha256": bindings["parent_authority_v121"]["sha256"],
        "oof_prejoin_completion_sha256": bindings["oof_prejoin_completion"]["sha256"],
        "lock_manifest_index_sha256": bindings["lock_manifest_index"]["sha256"],
        "lock_manifest_validation_sha256": bindings["lock_manifest_validation"]["sha256"],
        "fold_schedule_sha256": bindings["fold_schedule"]["sha256"],
        "full600_payload_sha256": bindings["full600_payload"]["sha256"],
        "source_shard_manifest_sequence_sha256": bindings[
            "source_shard_manifests"
        ]["sequence_sha256"],
        "p_outer_refit_checkpoint_sha256_by_fold": {
            str(fold): bindings[f"p_outer_refit_checkpoint_fold{fold}"]["sha256"]
            for fold in (1, 2, 3, 4)
        },
    }
