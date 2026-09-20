"""Validated dependency-reference resolution for SR0-MT execution nodes.

The final catalog freezes programs, stages, static arguments, dependency IDs,
and output roles.  It does not guess hashes of future artifacts.  Once an
upstream validator has emitted an immutable dependency receipt, this module
resolves only the explicitly declared ``path``/``sha256`` fields into a normal
scalar execution manifest.  It never submits or executes a node.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CATALOG_SCHEMA = "rc_dino_rcde_sr0_mt_execution_manifest_catalog_v1_20260815"
CATALOG_STATUS = "RCDE_SR0_MT_P_EXECUTION_MANIFESTS_READY"
CATALOG_VALIDATION_SCHEMA = (
    "rc_dino_rcde_sr0_mt_execution_manifest_validation_v1_20260815"
)
CATALOG_VALIDATION_STATUS = (
    "RCDE_SR0_MT_P_EXECUTION_MANIFESTS_INDEPENDENT_VALIDATION_PASS"
)
NODE_SCHEMA = "rc_dino_rcde_sr0_mt_p_execution_node_v1_20260815"
DEPENDENCY_MANIFEST_SCHEMA = (
    "rc_dino_rcde_sr0_mt_dependency_receipt_manifest_v1_20260815"
)
COMPLETION_MANIFEST_SCHEMA = (
    "rc_dino_rcde_sr0_mt_execution_completion_manifest_v1_20260815"
)
DEPENDENCY_RECEIPT_SCHEMA = "rc_dino_rcde_sr0_mt_dependency_receipt_v1_20260815"
DEPENDENCY_RECEIPT_STATUS = "RCDE_SR0_MT_DEPENDENCY_INDEPENDENT_VALIDATION_PASS"
RESOLVED_VALIDATION_SCHEMA = (
    "rc_dino_rcde_sr0_mt_resolved_execution_manifest_validation_v1_20260815"
)
RESOLVED_VALIDATION_STATUS = (
    "RCDE_SR0_MT_RESOLVED_EXECUTION_MANIFEST_INDEPENDENT_VALIDATION_PASS"
)
AUTHORITY_SENTINEL = "__LATEST_EXECUTION_AUTHORITY__"
AUTHORITY_SHA_SENTINEL = "__LATEST_EXECUTION_AUTHORITY_SHA256__"
PROTECTED = frozenset({"opened", "sealed", "c8", "s8"})
VALIDATION_CONTRACT_KEYS = frozenset(
    {
        "validator_program_field",
        "expected_validator_program",
        "schema_field",
        "expected_schema",
        "status_field",
        "expected_status",
        "pass_field",
        "node_id_field",
        "execution_manifest_sha256_field",
        "output_sha256_fields",
    }
)


class ExecutionResolutionError(RuntimeError):
    """One catalog/dependency-resolution invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionResolutionError(message)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe/absent file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: object, *, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} is not a lowercase SHA256",
    )
    return value


def read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionResolutionError(f"cannot read {name}: {path}") from exc
    require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def safe_file(root: Path, value: Path, *, name: str) -> Path:
    root = root.resolve(strict=True)
    require(not value.is_symlink(), f"{name} is symlinked")
    path = value.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ExecutionResolutionError(f"{name} escapes RC root") from exc
    require(path.is_file(), f"{name} is absent")
    require(not ({part.lower() for part in path.parts} & PROTECTED), f"{name} is protected")
    return path


def safe_declared_path(root: Path, value: object, *, name: str) -> Path:
    require(isinstance(value, str) and value, f"{name} path absent")
    raw = Path(value)
    path = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ExecutionResolutionError(f"{name} escapes RC root") from exc
    require(not ({part.lower() for part in path.parts} & PROTECTED), f"{name} is protected")
    return path


def logical_valid(value: Mapping[str, Any], field: str = "logical_sha256") -> bool:
    return value.get(field) == canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def validate_catalog_pair(
    *,
    root: Path,
    catalog_path: Path,
    catalog_sha256: str,
    validation_path: Path,
    validation_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    catalog_path = safe_file(root, catalog_path, name="execution catalog")
    validation_path = safe_file(root, validation_path, name="catalog validation")
    catalog_sha = require_sha(catalog_sha256, name="catalog")
    validation_sha = require_sha(validation_sha256, name="catalog validation")
    require(file_sha256(catalog_path) == catalog_sha, "catalog file hash drift")
    require(file_sha256(validation_path) == validation_sha, "catalog validation file hash drift")
    catalog = read_json(catalog_path, name="execution catalog")
    validation = read_json(validation_path, name="catalog validation")
    require(
        catalog.get("schema_version") == CATALOG_SCHEMA
        and catalog.get("status") == CATALOG_STATUS
        and logical_valid(catalog)
        and catalog.get("automatic_stage_advance") is False
        and catalog.get("next_authorized_stage") is None,
        "execution catalog closure drift",
    )
    nodes = catalog.get("nodes")
    require(isinstance(nodes, list) and catalog.get("node_count") == len(nodes), "catalog node population drift")
    require(
        validation.get("schema_version") == CATALOG_VALIDATION_SCHEMA
        and validation.get("status") == CATALOG_VALIDATION_STATUS
        and validation.get("validation_pass") is True
        and validation.get("catalog_file_sha256") == catalog_sha
        and validation.get("catalog_authority_sha256")
        == catalog.get("catalog_authority_sha256")
        and validation.get("i1_execution_graph_logical_sha256")
        == catalog.get("i1_execution_graph_logical_sha256")
        and validation.get("node_count") == len(nodes)
        and logical_valid(validation)
        and validation.get("automatic_stage_advance") is False
        and validation.get("next_authorized_stage") is None,
        "catalog validation closure drift",
    )
    return catalog, validation, catalog_path, validation_path


def node_template(
    catalog: Mapping[str, Any], node_id: str, *, root: Path
) -> dict[str, Any]:
    nodes = catalog.get("nodes")
    matches = [item for item in nodes if isinstance(item, dict) and item.get("node_id") == node_id]
    require(len(matches) == 1, "catalog node ID is absent/duplicate")
    node = matches[0]
    body = {key: item for key, item in node.items() if key != "node_template_sha256"}
    require(
        node.get("node_template_sha256") == canonical_sha256(body),
        "catalog node-template logical hash drift",
    )
    program = node.get("program")
    stage = node.get("stage")
    dependencies = node.get("dependencies")
    arguments = node.get("argument_template")
    output_roles = node.get("output_roles")
    validation_contract = node.get("validation_contract")
    require(isinstance(program, str) and program.endswith(".py"), "node program drift")
    require(isinstance(stage, str) and stage, "node stage absent")
    require(
        isinstance(dependencies, list)
        and len(dependencies) == len(set(dependencies))
        and all(isinstance(item, str) and item for item in dependencies),
        "node dependency IDs drift",
    )
    require(isinstance(arguments, dict) and arguments, "node argument template absent")
    require(isinstance(output_roles, dict) and output_roles, "node output roles absent")
    for role, path in output_roles.items():
        require(isinstance(role, str) and role and role.upper() == role, "output role name drift")
        safe_declared_path(root, path, name=f"output role {role}")
    require(
        isinstance(validation_contract, Mapping)
        and set(validation_contract) == VALIDATION_CONTRACT_KEYS,
        "node validation contract absent/drift",
    )
    for field in (
        "validator_program_field",
        "expected_validator_program",
        "schema_field",
        "expected_schema",
        "status_field",
        "expected_status",
        "pass_field",
        "node_id_field",
        "execution_manifest_sha256_field",
    ):
        require(
            isinstance(validation_contract[field], str)
            and bool(validation_contract[field]),
            f"node validation contract field drift: {field}",
        )
    output_sha_fields = validation_contract["output_sha256_fields"]
    require(
        isinstance(output_sha_fields, Mapping)
        and set(output_sha_fields) == set(output_roles)
        and all(isinstance(value, str) and value for value in output_sha_fields.values()),
        "node validation output-SHA field contract drift",
    )
    return node


def dependency_receipts(
    *, root: Path, manifest_path: Path, manifest_sha256: str, catalog_sha256: str
) -> dict[str, dict[str, Any]]:
    path = safe_file(root, manifest_path, name="dependency receipt manifest")
    require(file_sha256(path) == require_sha(manifest_sha256, name="dependency receipt manifest"), "dependency receipt manifest hash drift")
    manifest = read_json(path, name="dependency receipt manifest")
    require(
        manifest.get("schema_version") == DEPENDENCY_MANIFEST_SCHEMA
        and manifest.get("catalog_file_sha256") == catalog_sha256
        and logical_valid(manifest),
        "dependency receipt manifest closure drift",
    )
    entries = manifest.get("receipts")
    require(isinstance(entries, list), "dependency receipt entries absent")
    result: dict[str, dict[str, Any]] = {}
    for ordinal, entry in enumerate(entries):
        require(isinstance(entry, Mapping), f"dependency receipt entry {ordinal} drift")
        raw_path = entry.get("path")
        require(isinstance(raw_path, str) and raw_path, f"dependency receipt entry {ordinal} path absent")
        receipt_path = safe_file(root, Path(raw_path), name=f"dependency receipt {ordinal}")
        expected_sha = require_sha(entry.get("sha256"), name=f"dependency receipt {ordinal}")
        require(file_sha256(receipt_path) == expected_sha, f"dependency receipt {ordinal} hash drift")
        receipt = read_json(receipt_path, name=f"dependency receipt {ordinal}")
        require(
            receipt.get("schema_version") == DEPENDENCY_RECEIPT_SCHEMA
            and receipt.get("status") == DEPENDENCY_RECEIPT_STATUS
            and receipt.get("validation_pass") is True
            and receipt.get("catalog_file_sha256") == catalog_sha256
            and logical_valid(receipt)
            and receipt.get("automatic_stage_advance") is False
            and receipt.get("next_authorized_stage") is None,
            f"dependency receipt {ordinal} validation drift",
        )
        outputs = receipt.get("outputs")
        require(
            isinstance(outputs, Mapping) and outputs,
            f"dependency receipt {ordinal} outputs absent",
        )
        for role, output in outputs.items():
            require(
                isinstance(role, str)
                and role
                and isinstance(output, Mapping)
                and set(output) == {"path", "sha256"},
                f"dependency receipt {ordinal} output-role drift",
            )
            output_path = safe_file(
                root,
                Path(str(output.get("path"))),
                name=f"dependency receipt {ordinal} output {role}",
            )
            output_sha = require_sha(
                output.get("sha256"),
                name=f"dependency receipt {ordinal} output {role}",
            )
            require(
                file_sha256(output_path) == output_sha,
                f"dependency receipt {ordinal} output hash drift: {role}",
            )
        receipt_node_id = receipt.get("node_id")
        require(isinstance(receipt_node_id, str) and receipt_node_id not in result, "dependency receipt node duplicate")
        result[receipt_node_id] = receipt
    return result


def resolve_node_manifest(
    *,
    root: Path,
    catalog: Mapping[str, Any],
    catalog_sha256: str,
    node_id: str,
    receipts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, object], dict[str, object]]:
    node = node_template(catalog, node_id, root=root)
    dependencies = set(node["dependencies"])
    require(
        set(receipts) == dependencies,
        "dependency receipt set is not the exact catalog dependency set",
    )
    for dependency in sorted(dependencies):
        dependency_node = node_template(catalog, dependency, root=root)
        receipt = receipts[dependency]
        require(
            receipt.get("node_template_sha256")
            == dependency_node["node_template_sha256"],
            f"dependency receipt node-template substitution: {dependency}",
        )
        outputs = receipt.get("outputs")
        declared_outputs = dependency_node["output_roles"]
        require(
            isinstance(outputs, Mapping)
            and set(outputs) == set(declared_outputs),
            f"dependency receipt output-role set drift: {dependency}",
        )
        for role, declared_path in declared_outputs.items():
            expected_path = safe_declared_path(
                root,
                declared_path,
                name=f"catalog dependency output {dependency}.{role}",
            )
            actual_path = safe_file(
                root,
                Path(str(outputs[role]["path"])),
                name=f"dependency output {dependency}.{role}",
            )
            require(
                actual_path == expected_path,
                f"dependency output path substitution: {dependency}.{role}",
            )
    template = node["argument_template"]
    arguments: dict[str, object] = {}
    used_dependencies: set[str] = set()
    for key, raw in template.items():
        require(
            isinstance(key, str)
            and key
            and not key.startswith("-")
            and all(character in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in key),
            "unsafe argument-template key",
        )
        if isinstance(raw, (str, int, float, bool)) or raw is None:
            arguments[key] = raw
            continue
        require(isinstance(raw, Mapping) and set(raw) == {"dependency_node_id", "output_role", "attribute"}, f"argument reference drift: {key}")
        dependency = raw["dependency_node_id"]
        role = raw["output_role"]
        attribute = raw["attribute"]
        require(dependency in dependencies, f"cross-node dependency substitution: {key}")
        require(dependency in receipts, f"dependency receipt absent: {dependency}")
        receipt = receipts[str(dependency)]
        outputs = receipt.get("outputs")
        require(isinstance(outputs, Mapping) and role in outputs, f"dependency output role absent: {key}")
        output = outputs[role]
        require(isinstance(output, Mapping) and attribute in {"path", "sha256"}, f"dependency output attribute drift: {key}")
        value = output.get(attribute)
        if attribute == "path":
            safe_file(root, Path(str(value)), name=f"resolved dependency {key}")
            arguments[key] = str(Path(str(value)).resolve(strict=True))
        else:
            arguments[key] = require_sha(value, name=f"resolved dependency {key}")
        used_dependencies.add(str(dependency))
    require(
        used_dependencies <= dependencies,
        "argument reference uses an undeclared dependency",
    )
    if "authority" in arguments or "authority_sha256" in arguments:
        require(
            arguments.get("authority") == AUTHORITY_SENTINEL
            and arguments.get("authority_sha256") == AUTHORITY_SHA_SENTINEL,
            "resolved node authority sentinel pair drift",
        )
    manifest: dict[str, object] = {
        "schema_version": NODE_SCHEMA,
        "program": node["program"],
        "arguments": arguments,
    }
    manifest["logical_sha256"] = canonical_sha256(manifest)
    resolution = {
        "catalog_file_sha256": catalog_sha256,
        "node_id": node_id,
        "node_template_sha256": node["node_template_sha256"],
        "dependency_node_ids": sorted(dependencies),
        "dependency_receipt_logical_sha256s": {
            key: receipts[key]["logical_sha256"] for key in sorted(dependencies)
        },
    }
    return manifest, resolution


def validate_resolved_manifest(
    *,
    root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    expected: Mapping[str, Any],
) -> tuple[Path, str]:
    """Validate a physical resolved manifest against independent reconstruction."""

    path = safe_file(root, manifest_path, name="resolved execution manifest")
    actual_sha = file_sha256(path)
    require(
        actual_sha == require_sha(manifest_sha256, name="resolved execution manifest"),
        "resolved execution manifest physical hash drift",
    )
    require(
        read_json(path, name="resolved execution manifest") == dict(expected),
        "resolved execution manifest reconstruction drift",
    )
    return path, actual_sha


__all__ = [
    "AUTHORITY_SENTINEL",
    "AUTHORITY_SHA_SENTINEL",
    "CATALOG_SCHEMA",
    "CATALOG_STATUS",
    "CATALOG_VALIDATION_SCHEMA",
    "CATALOG_VALIDATION_STATUS",
    "COMPLETION_MANIFEST_SCHEMA",
    "DEPENDENCY_MANIFEST_SCHEMA",
    "DEPENDENCY_RECEIPT_SCHEMA",
    "DEPENDENCY_RECEIPT_STATUS",
    "ExecutionResolutionError",
    "NODE_SCHEMA",
    "RESOLVED_VALIDATION_SCHEMA",
    "RESOLVED_VALIDATION_STATUS",
    "VALIDATION_CONTRACT_KEYS",
    "canonical_sha256",
    "dependency_receipts",
    "file_sha256",
    "logical_valid",
    "node_template",
    "read_json",
    "require",
    "require_sha",
    "resolve_node_manifest",
    "safe_declared_path",
    "safe_file",
    "validate_catalog_pair",
    "validate_resolved_manifest",
]
