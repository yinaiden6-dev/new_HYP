#!/usr/bin/env python3
"""Materialize one natural, role-free SR0-MT pair-feature-cache shard.

The writer has one deliberately narrow data path::

    verified role-free pair addresses
      + one committed 12-query compact source-index shard
      + the frozen full600 ColNomic geometry/token payload
      -> two candidates x two directions per eligible query
      -> one ragged role-free pair feature cache per query

No loss-role file is an input.  Hidden target/rival roles are neither opened
nor reconstructed.  The monolithic compact shard is released before natural
token shards are loaded, and a conservative memory qualification fails closed
before ``torch.load``.

This is an additive writer only.  It does not authorize execution, train a
model, score heldout data, or advance a scientific stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from rc_aslo_xf.dino_rcde_cw1_multitile_pseal_sources_v1 import (
    CW1MultiTilePSealSourceLoaderV1,
    canonical_object_sha256,
)
from rc_aslo_xf.dino_rcde_cw1_multitile_sr0_p_v1 import FEATURE_SCHEMA_SHA256
from rc_aslo_xf.dino_rcde_sr0_mt_p_compact_catalog_v1 import (
    NATURAL_CANDIDATE_COUNT,
    ROLE_FREE_PAIR_ADDRESS_SCHEMA,
    ROLE_FREE_PAIR_MANIFEST_SCHEMA,
    FactorizedDirectionalSourceIndex,
    RoleFreeCandidateAddress,
    RoleFreePairAddress,
    RoleFreePairAddressManifest,
    deserialize_factorized_directional_source_index,
    deserialize_role_free_pair_feature_cache,
    make_candidate_feature_tensor,
    make_role_free_pair_feature_cache,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_compact_stream_v1 import (
    candidate_action_colnomic_feature_table_from_prepared,
    prepare_candidate_colnomic_features,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_natural_source_v1 import candidate_key_v1
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (
    P_DIRECTIONS,
    canonical_sha256,
    resolve_execution_authority_arguments,
    tensor_sha256,
)


RC_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_NAME = Path(__file__).name
QUERY_COUNT_PER_SHARD = 12
DIRECTION_COUNT = 2
FEATURE_DIM = 16
MINIMUM_AVAILABLE_MEMORY_BYTES = 16 * 1024**3
MAXIMUM_SOURCE_ARTIFACT_BYTES = 2 * 1024**3

SOURCE_SHARD_SCHEMA = "rc_dino_rcde_sr0_mt_compact_source_index_shard_v1_20260816"
SOURCE_SHARD_STATUS = "RCDE_SR0_MT_COMPACT_SOURCE_INDEX_SHARD_COMPLETE"
SOURCE_VALIDATION_STATUS = (
    "RCDE_SR0_MT_COMPACT_SOURCE_INDEX_INDEPENDENT_VALIDATION_PASS"
)
PAIR_ADDRESS_STATUS = "RCDE_SR0_MT_ROLE_FREE_PAIR_ADDRESS_READY"
OUTPUT_SCHEMA = (
    "rc_dino_rcde_sr0_mt_role_free_pair_feature_cache_shard_v1_20260817"
)
OUTPUT_STATUS = "RCDE_SR0_MT_PAIR_FEATURE_CACHE_E0_SHARD0_READY"
CLAIM_LEVEL = "ENGINEERING_NATURAL_ROLE_FREE_PAIR_FEATURE_CACHE_SHARD0_E0_ONLY"
EXECUTION_NODE_SCHEMA = (
    "rc_dino_rcde_sr0_mt_role_free_pair_feature_cache_execution_node_v1_20260817"
)
AUTHORITY_SCHEMA = "rc_current_authority_v52_20260817"
AUTHORITY_STATUS = (
    "RCDE_SR0_MT_ROLE_FREE_PAIR_FEATURE_CACHE_SHARD0_E0_MATERIALIZATION_AUTHORIZED"
)
AUTHORITY_STAGE = "SR0_MT_ROLE_FREE_PAIR_FEATURE_CACHE_SHARD0_E0_MATERIALIZATION"
AUTHORITY_SCOPE = "natural_role_free_pair_feature_cache_shard0_e0_materialization"
ARTIFACT_FILENAME = "pair_feature_cache.pt"
RESULT_FILENAME = "result.json"
COMMIT_FILENAME = "SHARD_COMMITTED.json"
RESULT_SCHEMA = (
    "rc_dino_rcde_sr0_mt_role_free_pair_feature_cache_publication_result_v1_20260817"
)
COMMIT_SCHEMA = (
    "rc_dino_rcde_sr0_mt_role_free_pair_feature_cache_commit_v1_20260817"
)
COMMIT_STATUS = "RCDE_SR0_MT_ROLE_FREE_PAIR_FEATURE_CACHE_SHARD_COMMITTED"
RESOURCE_ABORT_STATUS = "RCDE_SR0_MT_PAIR_FEATURE_CACHE_E0_RESOURCE_ABORT"
ENGINEERING_ABORT_STATUS = "RCDE_SR0_MT_PAIR_FEATURE_CACHE_E0_ENGINEERING_ABORT"
DECLARED_CPUS = 16
DECLARED_MEMORY_BYTES = 64 * 1024**3
DECLARED_WALL_SECONDS = 3600.0
DECLARED_WRITABLE_BYTES = 64 * 1024**3
RESOURCE_LIMIT_FRACTION = 0.70

_SHA256 = "0123456789abcdef"
_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "label",
        "target",
        "rival",
        "winner",
        "rank",
        "slot",
        "identity",
        "supergroup",
        "correctness",
        "outcome",
        "sign",
        "role",
        "gap",
        "score",
        "track",
        "fold",
        "loss_role",
        "loss_role_manifest",
        "target_member_ordinal",
        "rival_member_ordinal",
    }
)
_FORBIDDEN_PATH_FRAGMENTS = ("loss_role", "opened", "sealed", "c8", "s8")
_QUARANTINED_RELATIVE_PATH = (
    "results/dino_rcde_sr0_mt_role_free_pair_address_v1/loss_role_manifest.json"
)
_QUARANTINED_ACCESS_ATTEMPT_COUNT = 0
_QUARANTINE_AUDIT_HOOK_INSTALLED = False
_PROTECTED_ATTEMPT_COUNTS = {
    "loss_role": 0,
    "opened": 0,
    "sealed": 0,
    "C8": 0,
    "S8": 0,
}
_PROTECTED_SYSCALL_GUARDS_INSTALLED = False
_ORIGINAL_OS_STAT = os.stat
_ORIGINAL_OS_LSTAT = os.lstat

AUTHORITY_BINDING_KEYS = frozenset(
    {
        "parent_authority",
        "pair_address_validation_pass",
        "v49_validation",
        "virtual_aggregate",
        "role_free_pair_address",
        "source_index",
        "source_index_publication",
        "source_index_commit",
        "source_index_validation",
        "source_manifest",
        "source_manifest_validation",
        "geometry_payload",
        "geometry_receipt",
        "contract",
        "repair_addendum",
        "compact_catalog",
        "compact_stream",
        "compact_runtime",
        "natural_source",
        "natural_token_loader",
        "feature_schema",
        "colnomic_grid",
        "canonical_geometry",
        "producer",
        "execution_node",
        "freezer",
        "node_authorizer",
        "launcher",
        "authority_test",
        "writer_test",
        "output",
        "execution_interval",
    }
)
AUTHORITY_RESOURCE_CONTRACT = {
    "partition": "dev_cpuonly",
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": DECLARED_CPUS,
    "memory_megabytes": 65536,
    "declared_memory_bytes": DECLARED_MEMORY_BYTES,
    "declared_writable_bytes": DECLARED_WRITABLE_BYTES,
    "walltime_seconds": int(DECLARED_WALL_SECONDS),
    "qualification_fraction": RESOURCE_LIMIT_FRACTION,
    "maximum_peak_rss_bytes": 48103633715,
    "maximum_wall_seconds": 2520,
    "maximum_writable_bytes": 48103633715,
    "minimum_available_memory_bytes": max(
        MINIMUM_AVAILABLE_MEMORY_BYTES, 8 * 1644649495
    ),
}
AUTHORITY_TRUE_CAPABILITIES = frozenset(
    {
        "metadata_json_read_authorized",
        "source_index_artifact_read_authorized",
        "source_index_tensor_deserialization_authorized",
        "natural_token_payload_read_authorized",
        "natural_source_image_hash_read_authorized",
        "role_free_pair_feature_cache_materialization_authorized",
    }
)
AUTHORITY_FALSE_CAPABILITIES = frozenset(
    {
        "training_authorized",
        "natural_training_authorized",
        "natural_image_payload_read_authorized",
        "natural_image_decode_authorized",
        "checkpoint_deserialization_authorized",
        "model_load_authorized",
        "model_forward_authorized",
        "model_backward_authorized",
        "model_update_authorized",
        "C8_access_authorized",
        "S8_access_authorized",
        "opened_access_authorized",
        "sealed_access_authorized",
    }
)


class PairFeatureShardError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise PairFeatureShardError(message)


def require_sha256(value: object, *, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(item in _SHA256 for item in value),
        f"{name} is not a lowercase SHA256",
    )
    return str(value)


def guard_quarantined_path(path: str | os.PathLike[str]) -> None:
    """Count and reject exact role-side or protected-endpoint path access."""

    global _QUARANTINED_ACCESS_ATTEMPT_COUNT
    raw = Path(os.fspath(path))
    candidate = raw if raw.is_absolute() else RC_ROOT / raw
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    quarantined = Path(os.path.abspath(os.fspath(RC_ROOT / _QUARANTINED_RELATIVE_PATH)))
    protected: str | None = None
    if absolute == quarantined:
        protected = "loss_role"
    else:
        normalized_parts = {
            part.lower().replace("-", "_") for part in absolute.parts
        }
        for label, normalized in (
            ("opened", "opened"),
            ("sealed", "sealed"),
            ("C8", "c8"),
            ("S8", "s8"),
        ):
            if normalized in normalized_parts:
                protected = label
                break
    if protected is not None:
        _QUARANTINED_ACCESS_ATTEMPT_COUNT += 1
        _PROTECTED_ATTEMPT_COUNTS[protected] += 1
        raise PairFeatureShardError(f"protected path access attempted: {protected}")


def install_quarantine_audit_hook() -> None:
    """Reject Python-level attempts to open the quarantined role-side file.

    The explicit path firewall remains authoritative for stat/hash/path inputs;
    this audit hook independently catches an accidental direct ``open`` made by
    a dependency after the formal writer starts.
    """

    global _QUARANTINE_AUDIT_HOOK_INSTALLED, _PROTECTED_SYSCALL_GUARDS_INSTALLED
    if _QUARANTINE_AUDIT_HOOK_INSTALLED:
        return

    def _audit(event: str, arguments: tuple[object, ...]) -> None:
        if event != "open" or not arguments:
            return
        raw = arguments[0]
        if isinstance(raw, (str, bytes, os.PathLike)):
            guard_quarantined_path(os.fsdecode(raw))

    sys.addaudithook(_audit)
    _QUARANTINE_AUDIT_HOOK_INSTALLED = True
    if not _PROTECTED_SYSCALL_GUARDS_INSTALLED:

        def _guarded_stat(
            path: object, *args: object, **kwargs: object
        ) -> os.stat_result:
            if isinstance(path, (str, bytes, os.PathLike)):
                guard_quarantined_path(os.fsdecode(path))
            return _ORIGINAL_OS_STAT(path, *args, **kwargs)  # type: ignore[arg-type]

        def _guarded_lstat(
            path: object, *args: object, **kwargs: object
        ) -> os.stat_result:
            if isinstance(path, (str, bytes, os.PathLike)):
                guard_quarantined_path(os.fsdecode(path))
            return _ORIGINAL_OS_LSTAT(path, *args, **kwargs)  # type: ignore[arg-type]

        os.stat = _guarded_stat  # type: ignore[assignment]
        os.lstat = _guarded_lstat  # type: ignore[assignment]
        _PROTECTED_SYSCALL_GUARDS_INSTALLED = True


def file_sha256(path: Path) -> str:
    guard_quarantined_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def assert_role_free(value: object, *, path: str = "root") -> None:
    """Reject hidden-role fields without rejecting the literal ``role_free`` flag."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            require(
                normalized not in _FORBIDDEN_EXACT_KEYS,
                f"loss/role-bearing key is forbidden at {path}.{key}",
            )
            assert_role_free(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for ordinal, item in enumerate(value):
            assert_role_free(item, path=f"{path}[{ordinal}]")


def safe_input_path(root: Path, raw: object, *, name: str) -> Path:
    require(isinstance(raw, (str, Path)), f"{name} path absent")
    text = str(raw)
    guard_quarantined_path(text)
    normalized = text.lower().replace("-", "_")
    require(
        not any(fragment in normalized for fragment in _FORBIDDEN_PATH_FRAGMENTS),
        f"{name} path enters a forbidden role/evaluation namespace",
    )
    candidate = Path(text)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise PairFeatureShardError(f"{name} escapes RC root") from error
    require(path.is_file() and not path.is_symlink(), f"{name} is absent or unsafe")
    return path


def read_role_free_json(path: Path, *, name: str) -> dict[str, Any]:
    # The path firewall is checked before opening the file.
    require(
        "loss_role" not in path.as_posix().lower().replace("-", "_"),
        f"{name} path is a loss-role path",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairFeatureShardError(f"cannot read {name}") from error
    require(isinstance(value, dict), f"{name} is not a JSON object")
    assert_role_free(value)
    return value


def _candidate_address(value: object) -> RoleFreeCandidateAddress:
    require(isinstance(value, Mapping), "pair member is not a mapping")
    expected = {
        "query_id",
        "execution_ordinal",
        "query_source_image_sha256",
        "candidate_axis_sha256",
        "candidate_position",
        "candidate_key",
        "candidate_physical_row",
        "candidate_reference_source_sha256",
        "address_sha256",
    }
    require(set(value) == expected, "pair member field drift")
    try:
        return RoleFreeCandidateAddress(
            query_id=str(value["query_id"]),
            execution_ordinal=int(value["execution_ordinal"]),
            query_source_image_sha256=str(value["query_source_image_sha256"]),
            candidate_axis_sha256=str(value["candidate_axis_sha256"]),
            candidate_position=int(value["candidate_position"]),
            candidate_key=str(value["candidate_key"]),
            candidate_physical_row=int(value["candidate_physical_row"]),
            candidate_reference_source_sha256=str(
                value["candidate_reference_source_sha256"]
            ),
            address_sha256=str(value["address_sha256"]),
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise PairFeatureShardError("pair member validation failed") from error


def _pair_address(value: object) -> RoleFreePairAddress:
    require(isinstance(value, Mapping), "pair address is not a mapping")
    expected = {
        "schema_version",
        "query_id",
        "execution_ordinal",
        "query_source_image_sha256",
        "candidate_axis_sha256",
        "natural_axis_count",
        "members",
        "pair_sha256",
    }
    require(set(value) == expected, "pair address field drift")
    require(
        value.get("schema_version") == ROLE_FREE_PAIR_ADDRESS_SCHEMA,
        "pair address schema drift",
    )
    members = value.get("members")
    require(isinstance(members, list) and len(members) == 2, "pair member count drift")
    try:
        return RoleFreePairAddress(
            query_id=str(value["query_id"]),
            execution_ordinal=int(value["execution_ordinal"]),
            query_source_image_sha256=str(value["query_source_image_sha256"]),
            candidate_axis_sha256=str(value["candidate_axis_sha256"]),
            natural_axis_count=int(value["natural_axis_count"]),
            members=tuple(_candidate_address(item) for item in members),  # type: ignore[arg-type]
            pair_sha256=str(value["pair_sha256"]),
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise PairFeatureShardError("pair address validation failed") from error


def decode_role_free_pair_manifest(
    outer: Mapping[str, Any],
) -> RoleFreePairAddressManifest:
    """Decode only the role-free V51 artifact; no role-side receipt is accepted."""

    assert_role_free(outer)
    require(
        outer.get("status") == PAIR_ADDRESS_STATUS
        and outer.get("role_free") is True
        and outer.get("natural_axis_count") == NATURAL_CANDIDATE_COUNT
        and outer.get("query_count") == 594
        and outer.get("member_count") == 1188
        and outer.get("scientific_GO_or_NO_GO") is None
        and outer.get("automatic_stage_advance") is False
        and outer.get("next_authorized_stage") is None
        and outer.get("logical_sha256") == logical_sha256(outer),
        "role-free pair-address outer envelope drift",
    )
    raw = outer.get("pair_manifest")
    require(isinstance(raw, Mapping), "role-free pair manifest absent")
    assert_role_free(raw)
    expected_keys = {
        "schema_version",
        "role_free",
        "natural_axis_count",
        "query_count",
        "expected_execution_ordinals",
        "records",
        "manifest_sha256",
    }
    require(
        set(raw) == expected_keys
        and raw.get("schema_version") == ROLE_FREE_PAIR_MANIFEST_SCHEMA
        and raw.get("role_free") is True,
        "role-free pair manifest schema drift",
    )
    records = raw.get("records")
    executions = raw.get("expected_execution_ordinals")
    require(isinstance(records, list) and isinstance(executions, list), "pair population absent")
    try:
        return RoleFreePairAddressManifest(
            records=tuple(_pair_address(item) for item in records),
            expected_execution_ordinals=tuple(int(item) for item in executions),
            manifest_sha256=str(raw["manifest_sha256"]),
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise PairFeatureShardError("role-free pair manifest validation failed") from error


def pairs_for_shard(
    manifest: RoleFreePairAddressManifest, *, execution_start: int
) -> tuple[RoleFreePairAddress, ...]:
    require(
        isinstance(execution_start, int)
        and not isinstance(execution_start, bool)
        and 0 <= execution_start < 600
        and execution_start % QUERY_COUNT_PER_SHARD == 0,
        "execution start is not a frozen 12-query shard boundary",
    )
    stop = execution_start + QUERY_COUNT_PER_SHARD
    records = tuple(
        item
        for item in manifest.records
        if execution_start <= item.execution_ordinal < stop
    )
    expected = tuple(
        item
        for item in manifest.expected_execution_ordinals
        if execution_start <= item < stop
    )
    require(
        tuple(item.execution_ordinal for item in records) == expected,
        "role-free pair shard population drift",
    )
    return records


def effective_available_memory_bytes() -> int:
    """Return the conservative minimum of host and cgroup available memory."""

    values: list[int] = []
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            fields[key] = int(raw.strip().split()[0]) * 1024
        if "MemAvailable" in fields:
            values.append(fields["MemAvailable"])
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    for maximum_path, current_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        try:
            maximum_raw = maximum_path.read_text(encoding="ascii").strip()
            current = int(current_path.read_text(encoding="ascii").strip())
            if maximum_raw != "max":
                maximum = int(maximum_raw)
                # Ignore the common v1 sentinel meaning "unlimited".
                if 0 < maximum < (1 << 60):
                    values.append(maximum - current)
        except (OSError, UnicodeError, ValueError):
            continue
    require(bool(values), "available-memory qualification is unavailable")
    return max(0, min(values))


def qualify_source_artifact_resources(
    *, artifact_bytes: int, available_memory_bytes: int
) -> int:
    require(
        isinstance(artifact_bytes, int)
        and not isinstance(artifact_bytes, bool)
        and 0 < artifact_bytes <= MAXIMUM_SOURCE_ARTIFACT_BYTES,
        "compact source artifact size is outside the qualified envelope",
    )
    required = max(MINIMUM_AVAILABLE_MEMORY_BYTES, artifact_bytes * 8)
    require(
        available_memory_bytes >= required,
        f"insufficient memory for monolithic compact source load: need {required} bytes",
    )
    return required


def _source_key(value: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(value.get("execution_ordinal", -1)),
        str(value.get("candidate_key", "")),
        str(value.get("direction", "")),
    )


def select_pair_source_indexes(
    source_shard: Mapping[str, Any],
    pairs: Sequence[RoleFreePairAddress],
    *,
    execution_start: int,
    expected_artifact_logical_sha256: str,
    expected_record_population_sha256: str,
) -> dict[int, tuple[FactorizedDirectionalSourceIndex, ...]]:
    """Select and fully decode only the four authenticated records per pair."""

    stop = execution_start + QUERY_COUNT_PER_SHARD
    expected_artifact_logical_sha256 = require_sha256(
        expected_artifact_logical_sha256, name="source artifact logical"
    )
    expected_record_population_sha256 = require_sha256(
        expected_record_population_sha256, name="source record population"
    )
    records = source_shard.get("records")
    summaries = source_shard.get("record_summaries")
    require(
        source_shard.get("schema_version") == SOURCE_SHARD_SCHEMA
        and source_shard.get("status") == SOURCE_SHARD_STATUS
        and source_shard.get("target_free") is True
        and source_shard.get("execution_start") == execution_start
        and source_shard.get("execution_stop") == stop
        and source_shard.get("query_count") == QUERY_COUNT_PER_SHARD
        and source_shard.get("candidate_count_per_query") == NATURAL_CANDIDATE_COUNT
        and source_shard.get("direction_count") == DIRECTION_COUNT
        and source_shard.get("logical_record_count")
        == QUERY_COUNT_PER_SHARD * NATURAL_CANDIDATE_COUNT * DIRECTION_COUNT
        and source_shard.get("logical_sha256") == expected_artifact_logical_sha256
        and source_shard.get("record_population_sha256")
        == expected_record_population_sha256
        and isinstance(records, list)
        and isinstance(summaries, list)
        and len(records) == QUERY_COUNT_PER_SHARD * NATURAL_CANDIDATE_COUNT * DIRECTION_COUNT
        and len(summaries) == len(records),
        "compact source shard envelope drift",
    )
    logical = {
        key: item
        for key, item in source_shard.items()
        if key not in {"records", "logical_sha256"}
    }
    require(
        canonical_sha256(logical) == expected_artifact_logical_sha256
        and canonical_sha256(summaries) == expected_record_population_sha256,
        "compact source shard logical/population hash drift",
    )
    wanted = {
        (pair.execution_ordinal, member.candidate_key, direction)
        for pair in pairs
        for member in pair.members
        for direction in P_DIRECTIONS
    }
    require(len(wanted) == len(pairs) * 4, "wanted pair source population aliases")
    raw_by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for raw in records:
        require(isinstance(raw, Mapping), "compact source record is not a mapping")
        key = _source_key(raw)
        if key in wanted:
            require(key not in raw_by_key, "selected compact source record duplicated")
            raw_by_key[key] = raw
    summary_by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for raw in summaries:
        require(isinstance(raw, Mapping), "compact source summary is not a mapping")
        key = _source_key(raw)
        if key in wanted:
            require(key not in summary_by_key, "selected compact source summary duplicated")
            summary_by_key[key] = raw
    require(
        set(raw_by_key) == wanted and set(summary_by_key) == wanted,
        "selected two-candidate/two-direction source population is incomplete",
    )
    pair_by_execution = {item.execution_ordinal: item for item in pairs}
    output: dict[int, list[FactorizedDirectionalSourceIndex]] = {
        item.execution_ordinal: [] for item in pairs
    }
    for key in sorted(wanted):
        execution, candidate_key, direction = key
        pair = pair_by_execution[execution]
        member = next((item for item in pair.members if item.candidate_key == candidate_key), None)
        require(member is not None, "selected source member is absent from pair")
        try:
            index = deserialize_factorized_directional_source_index(raw_by_key[key])
        except (TypeError, ValueError, RuntimeError) as error:
            raise PairFeatureShardError("selected compact source record failed decoding") from error
        summary = summary_by_key[key]
        require(
            index.query_id == pair.query_id
            and index.execution_ordinal == execution
            and index.query_source_image_sha256 == pair.query_source_image_sha256
            and index.candidate_axis_sha256 == pair.candidate_axis_sha256
            and index.candidate_position == member.candidate_position
            and index.candidate_key == member.candidate_key
            and index.candidate_physical_row == member.candidate_physical_row
            and index.candidate_reference_source_sha256
            == member.candidate_reference_source_sha256
            and index.direction == direction
            and summary.get("candidate_position") == member.candidate_position
            and summary.get("population_sha256") == index.population_sha256,
            "selected source/pair/summary binding drift",
        )
        output[execution].append(index)
    result = {key: tuple(value) for key, value in output.items()}
    require(all(len(value) == 4 for value in result.values()), "pair source block count drift")
    return result


class BatchedNaturalPairFeatureProducer:
    """Compute each block once while sharing a candidate's full QxR similarity."""

    def __init__(
        self,
        token_resolver: Callable[
            [FactorizedDirectionalSourceIndex],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> None:
        self.token_resolver = token_resolver
        self._prepared: dict[tuple[int, str], object] = {}
        self.block_call_count = 0
        self.prepare_call_count = 0

    def __call__(self, source: FactorizedDirectionalSourceIndex):
        self.block_call_count += 1
        query, reference, q_valid, r_valid = self.token_resolver(source)
        query = torch.as_tensor(query).detach().cpu().contiguous()
        reference = torch.as_tensor(reference).detach().cpu().contiguous()
        q_valid = torch.as_tensor(q_valid, dtype=torch.bool).detach().cpu().contiguous()
        r_valid = torch.as_tensor(r_valid, dtype=torch.bool).detach().cpu().contiguous()
        require(
            tensor_sha256(query) == source.query_token_sha256
            and tensor_sha256(reference) == source.reference_token_sha256
            and tensor_sha256(q_valid) == source.query_valid_mask_sha256
            and tensor_sha256(r_valid) == source.reference_valid_mask_sha256,
            "natural token/source-index hash binding drift",
        )
        key = (source.execution_ordinal, source.candidate_key)
        if key not in self._prepared:
            self._prepared[key] = prepare_candidate_colnomic_features(
                query,
                reference,
                query_valid_patch_mask=q_valid,
                reference_valid_patch_mask=r_valid,
            )
            self.prepare_call_count += 1
        prepared = self._prepared[key]
        query_masks = torch.stack(
            [item.decode() for item in source.colnomic_query_root_masks]
        )
        reference_masks = torch.stack(
            [item.decode() for item in source.colnomic_reference_mask_table]
        )
        table = candidate_action_colnomic_feature_table_from_prepared(
            prepared,  # type: ignore[arg-type]
            query_action_masks=query_masks,
            reference_action_masks=reference_masks,
        )
        gathered = torch.empty(
            (source.root_count, source.action_count, FEATURE_DIM), dtype=torch.float64
        )
        for root in range(source.root_count):
            gathered[root] = table[root, source.colnomic_reference_mask_index[root]]
        return make_candidate_feature_tensor(source, gathered)


def build_pair_feature_shard(
    *,
    pairs: Sequence[RoleFreePairAddress],
    selected_indexes: Mapping[int, Sequence[FactorizedDirectionalSourceIndex]],
    token_resolver: Callable[
        [FactorizedDirectionalSourceIndex],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    execution_start: int,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable ragged caches after the big source shard is released."""

    assert_role_free(lineage, path="lineage")
    ordered = tuple(sorted(pairs, key=lambda item: item.execution_ordinal))
    require(
        tuple(item.execution_ordinal for item in ordered)
        == tuple(sorted(selected_indexes)),
        "pair/source execution population drift",
    )
    all_indexes = tuple(
        index
        for execution in sorted(selected_indexes)
        for index in selected_indexes[execution]
    )
    feature_implementations = {item.feature_implementation_sha256 for item in all_indexes}
    numeric_policies = {item.numeric_policy_sha256 for item in all_indexes}
    require(
        len(feature_implementations) == 1 and len(numeric_policies) == 1,
        "selected feature implementation/numeric policy drift",
    )
    excluded = tuple(
        execution
        for execution in range(execution_start, execution_start + QUERY_COUNT_PER_SHARD)
        if execution not in {item.execution_ordinal for item in ordered}
    )
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    block_call_count = 0
    prepare_call_count = 0
    for pair in ordered:
        # Query-local prepared similarities must not survive this iteration.
        producer = BatchedNaturalPairFeatureProducer(token_resolver)
        cache = make_role_free_pair_feature_cache(
            pair, tuple(selected_indexes[pair.execution_ordinal]), producer
        )
        require(
            producer.block_call_count == 4 and producer.prepare_call_count == 2,
            "per-query batched feature producer cardinality drift",
        )
        block_call_count += producer.block_call_count
        prepare_call_count += producer.prepare_call_count
        record = {
            "query_id": pair.query_id,
            "execution_ordinal": pair.execution_ordinal,
            "pair_address_sha256": pair.pair_sha256,
            "pair_feature_cache": cache.payload(),
        }
        records.append(record)
        summaries.append(
            {
                "query_id": pair.query_id,
                "execution_ordinal": pair.execution_ordinal,
                "pair_address_sha256": pair.pair_sha256,
                "cache_sha256": cache.cache_sha256,
                "source_population_sha256s": [
                    item.source_population_sha256 for item in cache.blocks
                ],
                "block_count": len(cache.blocks),
                "coordinate_count": int(cache.features.shape[0]),
                "ready_coordinate_count": int(cache.eligibility.sum().item()),
                "h0_coordinate_count": int((~cache.eligibility).sum().item()),
            }
        )
        producer._prepared.clear()
        release = getattr(token_resolver, "release_query", None)
        if callable(release):
            release(pair.execution_ordinal)
    require(
        block_call_count == len(ordered) * 4
        and prepare_call_count == len(ordered) * 2,
        "batched feature producer cardinality drift",
    )
    result: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "claim_level": CLAIM_LEVEL,
        "role_free": True,
        "target_free": True,
        "synthetic": False,
        "execution_start": execution_start,
        "execution_stop": execution_start + QUERY_COUNT_PER_SHARD,
        "query_address_count": QUERY_COUNT_PER_SHARD,
        "pair_query_count": len(ordered),
        "excluded_query_count": QUERY_COUNT_PER_SHARD - len(ordered),
        "excluded_execution_ordinals": list(excluded),
        "excluded_population_sha256": canonical_sha256(list(excluded)),
        "candidate_count_per_query": NATURAL_CANDIDATE_COUNT,
        "pair_member_count": len(ordered) * 2,
        "direction_count": DIRECTION_COUNT,
        "feature_dimension": FEATURE_DIM,
        "feature_implementation_sha256": next(iter(feature_implementations)),
        "numeric_policy_sha256": next(iter(numeric_policies)),
        "batched_block_call_count": block_call_count,
        "prepared_candidate_count": prepare_call_count,
        "full_similarity_call_count": prepare_call_count,
        "token_normalization_count": prepare_call_count * 2,
        "max_live_feature_query_count": 1 if ordered else 0,
        "record_summaries": summaries,
        "record_population_sha256": canonical_sha256(summaries),
        "records": records,
        "lineage": dict(lineage),
        "access_audit": {
            "forbidden_metadata_input_count": 0,
            "quarantined_file_attempted_access_count": 0,
            "natural_image_decode_count": 0,
            "model_load_count": 0,
            "model_forward_count": 0,
            "model_backward_count": 0,
            "model_update_count": 0,
            "training_count": 0,
            "opened_read_count": 0,
            "sealed_read_count": 0,
            "C8_read_count": 0,
            "S8_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    # Audit field names state zero access; they are not role values.  Exclude
    # the audit from the generic role-bearing-key check for this reason.
    assert_role_free({key: item for key, item in result.items() if key != "access_audit"})
    logical = {
        key: item
        for key, item in result.items()
        if key not in {"records", "logical_sha256"}
    }
    result["logical_sha256"] = canonical_sha256(logical)
    return result


class Full600PairTokenResolver:
    """Resolve only the V51-addressed query and two references per query."""

    def __init__(
        self,
        *,
        payload_path: Path,
        payload_sha256: str,
        pairs: Sequence[RoleFreePairAddress],
        rc_root: Path,
    ) -> None:
        self.loader = CW1MultiTilePSealSourceLoaderV1(
            payload_path=payload_path,
            rc_root=rc_root,
            expected_payload_file_sha256=payload_sha256,
        )
        payload = self.loader._load_payload()
        raw_queries = payload.get("query_records")
        raw_references = payload.get("reference_records")
        require(isinstance(raw_queries, list) and isinstance(raw_references, list), "full600 records absent")
        self.query_records = {
            int(item["execution_ordinal"]): item
            for item in raw_queries
            if isinstance(item, Mapping)
        }
        self.reference_records = {
            int(item["physical_row"]): item
            for item in raw_references
            if isinstance(item, Mapping)
        }
        self.pairs = {item.execution_ordinal: item for item in pairs}
        self.query_sources: dict[int, Any] = {}
        self.reference_sources: dict[int, Any] = {}
        self.max_live_query_count = 0
        self.max_live_reference_count = 0
        self.max_live_selected_entry_count = 0
        self.selected_query_entry_count = 0
        self.selected_reference_entry_count = 0

    def __call__(
        self, source: FactorizedDirectionalSourceIndex
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pairs.get(source.execution_ordinal)
        require(pair is not None, "token request is outside pair shard")
        member = next((item for item in pair.members if item.candidate_key == source.candidate_key), None)
        require(member is not None, "token request is outside role-free pair")
        query_record = self.query_records.get(source.execution_ordinal)
        reference_record = self.reference_records.get(member.candidate_physical_row)
        require(query_record is not None and reference_record is not None, "full600 pair source absent")
        axis = query_record.get("candidate_physical_rows")
        require(
            query_record.get("query_id") == pair.query_id
            and query_record.get("source_image_sha256") == pair.query_source_image_sha256
            and query_record.get("candidate_axis_sha256") == pair.candidate_axis_sha256
            and isinstance(axis, list)
            and len(axis) == NATURAL_CANDIDATE_COUNT
            and axis[member.candidate_position] == member.candidate_physical_row
            and reference_record.get("source_image_sha256")
            == member.candidate_reference_source_sha256,
            "full600 role-free pair address binding drift",
        )
        if source.execution_ordinal not in self.query_sources:
            self.query_sources[source.execution_ordinal] = self.loader._build_source(
                record=project_token_record(query_record, kind="query"),
                kind="query",
                native_key=source.historical_query_ordinal,
            )
            self.max_live_query_count = max(
                self.max_live_query_count, len(self.query_sources)
            )
            self.selected_query_entry_count += 1
        row = member.candidate_physical_row
        if row not in self.reference_sources:
            self.reference_sources[row] = self.loader._build_source(
                record=project_token_record(reference_record, kind="reference"),
                kind="reference",
                native_key=row,
            )
            self.selected_reference_entry_count += 1
            self.max_live_reference_count = max(
                self.max_live_reference_count, len(self.reference_sources)
            )
        self.max_live_selected_entry_count = max(
            self.max_live_selected_entry_count,
            len(self.query_sources) + len(self.reference_sources),
        )
        query = self.query_sources[source.execution_ordinal]
        reference = self.reference_sources[row]
        require(
            candidate_key_v1(
                physical_row=row, source_image_sha256=reference.source_image_sha256
            )
            == member.candidate_key
            and query.dino_geometry.sha256 == source.query_geometry_sha256
            and reference.dino_geometry.sha256 == source.reference_geometry_sha256,
            "full600 token/geometry source binding drift",
        )
        return (
            query.tokens,
            reference.tokens,
            query.colnomic_geometry.valid_patch_mask,
            reference.colnomic_geometry.valid_patch_mask,
        )

    def release_query(self, execution_ordinal: int) -> None:
        """Release this query and both selected references before the next pair."""

        self.query_sources.pop(execution_ordinal, None)
        pair = self.pairs.get(execution_ordinal)
        require(pair is not None, "release query is outside pair shard")
        for member in pair.members:
            self.reference_sources.pop(member.candidate_physical_row, None)
        require(
            not self.query_sources and not self.reference_sources,
            "selected token sources survived query release",
        )


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairFeatureShardError(f"cannot read {name}") from error
    require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def project_token_record(record: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """Verify the immutable parent then expose only token-loader fields.

    Fields such as ``heldout_fold`` and ``dino_cache`` remain covered by the
    parent record hash, but are never copied into or read from the projection.
    DINO *geometry* is retained only as a coordinate-system binding; no DINO
    token/cache payload crosses this adapter.
    """

    require(kind in {"query", "reference"}, "token projection kind drift")
    original_hash = record.get("record_logical_sha256")
    require(
        original_hash
        == canonical_object_sha256(
            {key: item for key, item in record.items() if key != "record_logical_sha256"}
        ),
        "full600 parent record logical hash drift",
    )
    allowed = {
        "source_path",
        "source_image_sha256",
        "colnomic_cache",
        "colnomic_geometry",
        "dino_geometry",
    }
    if kind == "query":
        allowed.add("query_id")
    require(allowed <= set(record), "token projection source fields absent")
    projected = {key: record[key] for key in sorted(allowed)}
    projected["record_logical_sha256"] = canonical_object_sha256(projected)
    require(
        "heldout_fold" not in projected
        and "dino_cache" not in projected
        and "checkpoint" not in projected,
        "forbidden full600 field crossed token projection",
    )
    return projected


def _authenticate_source_inputs(
    *,
    aggregate: Mapping[str, Any],
    aggregate_file_sha256: str,
    source_path: Path,
    source_file_sha256: str,
    source_validation: Mapping[str, Any],
    source_validation_file_sha256: str,
    source_manifest: Mapping[str, Any],
    source_manifest_file_sha256: str,
    execution_start: int,
) -> tuple[Mapping[str, Any], Path, str]:
    stop = execution_start + QUERY_COUNT_PER_SHARD
    require(
        aggregate.get("status") == "RCDE_SR0_MT_COMPACT_SOURCE_VIRTUAL_AGGREGATE_READY"
        and aggregate.get("virtual_aggregate") is True
        and aggregate.get("target_free") is True
        and aggregate.get("query_count") == 600
        and aggregate.get("shard_count") == 50
        and aggregate.get("logical_sha256") == logical_sha256(aggregate)
        and require_sha256(aggregate_file_sha256, name="aggregate file")
        == aggregate_file_sha256,
        "V49 virtual aggregate drift",
    )
    shards = aggregate.get("shards")
    require(isinstance(shards, list), "V49 shard ledger absent")
    matches = [
        item
        for item in shards
        if isinstance(item, Mapping)
        and item.get("execution_start") == execution_start
        and item.get("execution_stop") == stop
    ]
    require(len(matches) == 1, "V49 shard binding absent or duplicated")
    binding = matches[0]
    source_file_sha = require_sha256(source_file_sha256, name="source index file")
    require(
        source_file_sha == binding.get("artifact_file_sha256")
        and source_path.stat().st_size == binding.get("artifact_bytes")
        and source_validation_file_sha256 == binding.get("validation_file_sha256")
        and source_validation.get("status") == SOURCE_VALIDATION_STATUS
        and source_validation.get("validation_pass") is True
        and source_validation.get("execution_start") == execution_start
        and source_validation.get("execution_stop") == stop
        and source_validation.get("source_shard_file_sha256") == source_file_sha
        and source_validation.get("source_shard_artifact_logical_sha256")
        == binding.get("artifact_logical_sha256")
        and source_validation.get("record_population_sha256")
        == binding.get("record_population_sha256")
        and source_validation.get("source_manifest_sha256")
        == source_manifest_file_sha256
        and source_manifest.get("target_free") is True
        and source_manifest.get("execution_start") == execution_start
        and source_manifest.get("execution_stop") == stop
        and source_manifest.get("candidate_count_per_query")
        == NATURAL_CANDIDATE_COUNT
        and source_manifest.get("logical_sha256") == logical_sha256(source_manifest),
        "V49/source-validation/source-manifest lineage drift",
    )
    return binding, source_path, source_file_sha


def _serialized_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_torch(directory: Path, value: Mapping[str, Any]) -> Path:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{ARTIFACT_FILENAME}.", suffix=".staging", dir=directory
    )
    path = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
        return path
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        path.unlink(missing_ok=True)
        raise


def _stage_bytes(directory: Path, *, prefix: str, data: bytes) -> Path:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{prefix}.", suffix=".staging", dir=directory
    )
    path = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o444)
        return path
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        path.unlink(missing_ok=True)
        raise


def _file_bytes_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while True:
            a = lhs.read(8 * 1024 * 1024)
            b = rhs.read(8 * 1024 * 1024)
            if a != b:
                return False
            if not a:
                return True


def _pair_record_serialized_bytes(shard: Mapping[str, Any]) -> list[dict[str, int]]:
    output: list[dict[str, int]] = []
    records = shard.get("records")
    require(isinstance(records, list), "pair feature records absent")
    for record in records:
        require(isinstance(record, Mapping), "pair feature record drift")
        buffer = io.BytesIO()
        torch.save(dict(record), buffer)
        output.append(
            {
                "execution_ordinal": int(record["execution_ordinal"]),
                "serialized_bytes": len(buffer.getvalue()),
            }
        )
    return output


def validate_serialized_pair_feature_shard(value: Mapping[str, Any]) -> None:
    """Semantic postwrite replay of every pair cache, block and tensor hash."""

    records = value.get("records")
    summaries = value.get("record_summaries")
    require(
        value.get("schema_version") == OUTPUT_SCHEMA
        and value.get("status") == OUTPUT_STATUS
        and value.get("role_free") is True
        and value.get("target_free") is True
        and value.get("logical_sha256")
        == canonical_sha256(
            {
                key: item
                for key, item in value.items()
                if key not in {"records", "logical_sha256"}
            }
        )
        and isinstance(records, list)
        and isinstance(summaries, list)
        and len(records) == value.get("pair_query_count") == len(summaries)
        and value.get("record_population_sha256") == canonical_sha256(summaries),
        "serialized pair-feature shard envelope drift",
    )
    summary_by_execution = {
        int(item["execution_ordinal"]): item
        for item in summaries
        if isinstance(item, Mapping)
    }
    require(len(summary_by_execution) == len(summaries), "pair summary execution alias")
    observed: list[int] = []
    for record in records:
        require(
            isinstance(record, Mapping)
            and set(record)
            == {
                "query_id",
                "execution_ordinal",
                "pair_address_sha256",
                "pair_feature_cache",
            },
            "serialized pair-feature record field drift",
        )
        execution = int(record["execution_ordinal"])
        observed.append(execution)
        try:
            cache = deserialize_role_free_pair_feature_cache(
                record["pair_feature_cache"]  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise PairFeatureShardError("serialized pair cache semantic replay failed") from error
        summary = summary_by_execution.get(execution)
        require(
            summary is not None
            and record.get("query_id") == summary.get("query_id")
            and record.get("pair_address_sha256")
            == summary.get("pair_address_sha256")
            == cache.pair_address_sha256
            and summary.get("cache_sha256") == cache.cache_sha256
            and summary.get("block_count") == 4
            and summary.get("coordinate_count") == cache.features.shape[0]
            and summary.get("ready_coordinate_count")
            == int(cache.eligibility.sum().item())
            and summary.get("h0_coordinate_count")
            == int((~cache.eligibility).sum().item()),
            "serialized pair cache/summary binding drift",
        )
    require(observed == sorted(observed), "serialized pair execution order drift")


def _receipt_and_marker(
    *,
    shard: Mapping[str, Any],
    artifact_sha256: str,
    artifact_bytes: int,
    authority_sha256: str,
    execution_manifest_sha256: str,
    resource_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    receipt: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": OUTPUT_STATUS,
        "claim_level": CLAIM_LEVEL,
        "role_free": True,
        "target_free": True,
        "execution_start": shard["execution_start"],
        "execution_stop": shard["execution_stop"],
        "pair_query_count": shard["pair_query_count"],
        "artifact_filename": ARTIFACT_FILENAME,
        "artifact_file_sha256": artifact_sha256,
        "artifact_logical_sha256": shard["logical_sha256"],
        "artifact_bytes": artifact_bytes,
        "record_population_sha256": shard["record_population_sha256"],
        "authority_sha256": authority_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
        "resource_receipt": dict(resource_receipt),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    receipt["logical_sha256"] = logical_sha256(receipt)
    receipt_bytes = _serialized_json(receipt)
    marker: dict[str, Any] = {
        "schema_version": COMMIT_SCHEMA,
        "status": COMMIT_STATUS,
        "committed": True,
        "artifact_filename": ARTIFACT_FILENAME,
        "artifact_file_sha256": artifact_sha256,
        "artifact_logical_sha256": shard["logical_sha256"],
        "artifact_bytes": artifact_bytes,
        "receipt_filename": RESULT_FILENAME,
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_logical_sha256": receipt["logical_sha256"],
        "receipt_bytes": len(receipt_bytes),
        "execution_start": shard["execution_start"],
        "execution_stop": shard["execution_stop"],
        "execution_manifest_sha256": execution_manifest_sha256,
    }
    marker["logical_sha256"] = logical_sha256(marker)
    marker_bytes = _serialized_json(marker)
    return receipt, receipt_bytes, marker, marker_bytes


def publish_pair_feature_directory(
    directory: Path,
    shard: Mapping[str, Any],
    *,
    authority_sha256: str,
    execution_manifest_sha256: str,
    run_start: float,
    input_deserialization_seconds: float,
    feature_compute_seconds: float,
    source_index_hash_stream_bytes: int,
    source_index_mapped_file_bytes: int,
    geometry_payload_hash_stream_bytes: int,
    geometry_payload_mapped_file_bytes: int,
    geometry_payload_deserialization_count: int,
    colnomic_token_cache_file_deserialization_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish one complete immutable directory by one sibling rename.

    A pre-existing final path is never resumed, repaired, or overwritten.  All
    three files are first completed and fsynced inside a fresh sibling staging
    directory.  Only then is that directory atomically renamed into place.
    """

    parent = directory.parent
    require(parent.is_dir() and not parent.is_symlink(), "output parent absent or unsafe")
    require(
        not directory.exists() and not directory.is_symlink(),
        "pre-existing pair-feature output is forbidden",
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.", suffix=".staging", dir=parent
        )
    )
    renamed = False
    try:
        write_started = time.perf_counter()
        artifact_temporary = _stage_torch(staging, shard)
        artifact_path = staging / ARTIFACT_FILENAME
        os.rename(artifact_temporary, artifact_path)
        _fsync_directory(staging)
        write_seconds = time.perf_counter() - write_started

        read_started = time.perf_counter()
        artifact_sha = file_sha256(artifact_path)
        artifact_bytes = artifact_path.stat().st_size
        replay = torch.load(
            artifact_path, map_location="cpu", weights_only=True, mmap=True
        )
        require(
            isinstance(replay, Mapping)
            and replay.get("logical_sha256") == shard.get("logical_sha256"),
            "published pair-feature replay drift",
        )
        validate_serialized_pair_feature_shard(replay)
        del replay
        read_seconds = time.perf_counter() - read_started

        per_query_bytes = _pair_record_serialized_bytes(shard)
        wall_seconds = time.perf_counter() - run_start
        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        base_resource: dict[str, Any] = {
            "declared_cpu_count": DECLARED_CPUS,
            "declared_memory_bytes": DECLARED_MEMORY_BYTES,
            "declared_wall_seconds": DECLARED_WALL_SECONDS,
            "declared_writable_bytes": DECLARED_WRITABLE_BYTES,
            "qualification_fraction": RESOURCE_LIMIT_FRACTION,
            "maximum_peak_rss_bytes": int(
                DECLARED_MEMORY_BYTES * RESOURCE_LIMIT_FRACTION
            ),
            "maximum_wall_seconds": DECLARED_WALL_SECONDS * RESOURCE_LIMIT_FRACTION,
            "maximum_writable_bytes": int(
                DECLARED_WRITABLE_BYTES * RESOURCE_LIMIT_FRACTION
            ),
            "measured_peak_rss_bytes": peak_rss_bytes,
            "measured_wall_seconds": wall_seconds,
            "measured_writable_bytes": 0,
            "source_index_io_mode": "sha256_stream_plus_torch_mmap",
            "source_index_hash_stream_bytes": source_index_hash_stream_bytes,
            "source_index_mapped_file_bytes": source_index_mapped_file_bytes,
            "source_index_deserialization_count": 1,
            "geometry_payload_io_mode": "two_sha256_streams_plus_torch_mmap",
            "geometry_payload_hash_stream_bytes": geometry_payload_hash_stream_bytes,
            "geometry_payload_mapped_file_bytes": geometry_payload_mapped_file_bytes,
            "geometry_payload_deserialization_count": geometry_payload_deserialization_count,
            "colnomic_token_cache_file_deserialization_count": colnomic_token_cache_file_deserialization_count,
            "input_deserialization_seconds": input_deserialization_seconds,
            "feature_compute_seconds": feature_compute_seconds,
            "pair_cache_write_seconds": write_seconds,
            "pair_cache_read_seconds": read_seconds,
            "per_query_cache_bytes": per_query_bytes,
            "full_similarity_call_count": shard["full_similarity_call_count"],
            "token_normalization_count": shard["token_normalization_count"],
            "max_live_feature_query_count": shard["max_live_feature_query_count"],
            "max_live_selected_reference_entry_count": shard[
                "max_live_selected_reference_entry_count"
            ],
            "max_live_selected_entry_count": shard[
                "max_live_selected_entry_count"
            ],
            "selected_query_entry_count": shard["selected_query_entry_count"],
            "selected_reference_entry_count": shard[
                "selected_reference_entry_count"
            ],
            "token_cache_mapped_file_count": shard[
                "token_cache_mapped_file_count"
            ],
            "token_cache_mapped_file_bytes": shard[
                "token_cache_mapped_file_bytes"
            ],
            "natural_source_image_hash_file_count": shard[
                "natural_source_image_hash_file_count"
            ],
            "natural_source_image_hash_stream_bytes": shard[
                "natural_source_image_hash_stream_bytes"
            ],
        }
        total_bytes = artifact_bytes
        for _ in range(4):
            base_resource["measured_writable_bytes"] = total_bytes
            provisional = dict(base_resource)
            provisional["resource_gate_pass"] = True
            _, receipt_bytes, _, marker_bytes = _receipt_and_marker(
                shard=shard,
                artifact_sha256=artifact_sha,
                artifact_bytes=artifact_bytes,
                authority_sha256=authority_sha256,
                execution_manifest_sha256=execution_manifest_sha256,
                resource_receipt=provisional,
            )
            updated = artifact_bytes + len(receipt_bytes) + len(marker_bytes)
            if updated == total_bytes:
                break
            total_bytes = updated
        resource_pass = (
            peak_rss_bytes <= base_resource["maximum_peak_rss_bytes"]
            and wall_seconds <= base_resource["maximum_wall_seconds"]
            and total_bytes <= base_resource["maximum_writable_bytes"]
            and shard.get("max_live_feature_query_count") == 1
            and shard.get("max_live_selected_reference_entry_count") == 2
            and shard.get("max_live_selected_entry_count") == 3
        )
        base_resource["measured_writable_bytes"] = total_bytes
        base_resource["resource_gate_pass"] = resource_pass
        require(resource_pass, "pair-feature E0 resource qualification failed")

        receipt, receipt_bytes, marker, marker_bytes = _receipt_and_marker(
            shard=shard,
            artifact_sha256=artifact_sha,
            artifact_bytes=artifact_bytes,
            authority_sha256=authority_sha256,
            execution_manifest_sha256=execution_manifest_sha256,
            resource_receipt=base_resource,
        )
        receipt_temporary = _stage_bytes(
            staging, prefix=RESULT_FILENAME, data=receipt_bytes
        )
        os.rename(receipt_temporary, staging / RESULT_FILENAME)
        _fsync_directory(staging)
        marker_temporary = _stage_bytes(
            staging, prefix=COMMIT_FILENAME, data=marker_bytes
        )
        os.rename(marker_temporary, staging / COMMIT_FILENAME)
        _fsync_directory(staging)
        require(
            {item.name for item in staging.iterdir()}
            == {ARTIFACT_FILENAME, RESULT_FILENAME, COMMIT_FILENAME},
            "staging publication field drift",
        )
        for item in staging.iterdir():
            require(
                stat.S_IMODE(item.stat().st_mode) == 0o444,
                "staging publication file is mutable",
            )
        os.chmod(staging, 0o555)
        _fsync_directory(staging)
        require(
            not directory.exists() and not directory.is_symlink(),
            "pair-feature output appeared during staging",
        )
        os.rename(staging, directory)
        renamed = True
        _fsync_directory(parent)
        return receipt, marker
    finally:
        if not renamed and staging.exists():
            os.chmod(staging, 0o700)
            shutil.rmtree(staging)


def validate_complete_publication_readonly(
    directory: Path,
    shard: Mapping[str, Any],
    *,
    authority_sha256: str,
    execution_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an immutable complete final without modifying or resuming it."""

    require(directory.is_dir() and not directory.is_symlink(), "publication absent")
    require(
        {item.name for item in directory.iterdir()}
        == {ARTIFACT_FILENAME, RESULT_FILENAME, COMMIT_FILENAME},
        "complete publication field drift",
    )
    for item in directory.iterdir():
        require(stat.S_IMODE(item.stat().st_mode) == 0o444, "publication file is mutable")
    artifact_path = directory / ARTIFACT_FILENAME
    receipt_path = directory / RESULT_FILENAME
    marker_path = directory / COMMIT_FILENAME
    replay = torch.load(artifact_path, map_location="cpu", weights_only=True, mmap=True)
    require(isinstance(replay, Mapping), "publication artifact is not a mapping")
    validate_serialized_pair_feature_shard(replay)
    require(replay.get("logical_sha256") == shard.get("logical_sha256"), "publication logical drift")
    receipt = _read_json(receipt_path, name="publication receipt")
    marker = _read_json(marker_path, name="publication commit")
    artifact_sha = file_sha256(artifact_path)
    receipt_bytes = receipt_path.read_bytes()
    require(
        receipt.get("schema_version") == RESULT_SCHEMA
        and receipt.get("status") == OUTPUT_STATUS
        and receipt.get("artifact_file_sha256") == artifact_sha
        and receipt.get("artifact_logical_sha256") == shard.get("logical_sha256")
        and receipt.get("authority_sha256") == authority_sha256
        and receipt.get("execution_manifest_sha256") == execution_manifest_sha256
        and receipt.get("logical_sha256") == logical_sha256(receipt)
        and marker.get("schema_version") == COMMIT_SCHEMA
        and marker.get("status") == COMMIT_STATUS
        and marker.get("committed") is True
        and marker.get("artifact_file_sha256") == artifact_sha
        and marker.get("receipt_file_sha256")
        == hashlib.sha256(receipt_bytes).hexdigest()
        and marker.get("execution_manifest_sha256") == execution_manifest_sha256
        and marker.get("logical_sha256") == logical_sha256(marker),
        "complete publication binding drift",
    )
    return receipt, marker


def _execution_manifest_arguments(
    raw_argv: Sequence[str],
) -> tuple[dict[str, Any], Path, dict[str, Any], str]:
    require(
        len(raw_argv) == 2 and raw_argv[0] == "--execution-manifest",
        "formal writer accepts only --execution-manifest PATH",
    )
    path = safe_input_path(RC_ROOT.resolve(), raw_argv[1], name="execution manifest")
    node = read_role_free_json(path, name="execution manifest")
    require(
        set(node) == {"schema_version", "program", "arguments", "logical_sha256"}
        and node.get("schema_version") == EXECUTION_NODE_SCHEMA
        and node.get("program") == PROGRAM_NAME
        and node.get("logical_sha256") == logical_sha256(node)
        and isinstance(node.get("arguments"), Mapping),
        "pair-feature execution manifest drift",
    )
    arguments = resolve_execution_authority_arguments(node["arguments"])
    assert_role_free(arguments, path="execution.arguments")
    return dict(arguments), path, node, file_sha256(path)


def _arguments_to_argv(arguments: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in arguments.items():
        require(
            isinstance(key, str)
            and key
            and all(item in "abcdefghijklmnopqrstuvwxyz0123456789_" for item in key)
            and isinstance(value, (str, int, float))
            and not isinstance(value, bool),
            "execution argument is not one safe scalar",
        )
        result.extend(("--" + key.replace("_", "-"), str(value)))
    return result


def _validate_authority(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    require(path.name == "current_authority_v52_20260817.json", "authority is not exact V52")
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    versions = [
        (int(match.group(1)), candidate.resolve())
        for candidate in path.parent.glob("current_authority_v*_*.json")
        if (match := pattern.fullmatch(candidate.name))
    ]
    require(bool(versions), "versioned authority registry is empty")
    latest_version = max(item[0] for item in versions)
    latest_paths = [item_path for version, item_path in versions if version == latest_version]
    require(
        latest_version == 52
        and len(latest_paths) == 1
        and latest_paths[0] == path.resolve(),
        "V52 is not the unique latest execution authority",
    )
    actual = file_sha256(path)
    require(actual == require_sha256(expected_sha256, name="authority"), "authority file hash drift")
    value = read_role_free_json(path, name="V52 authority")
    require(
        value.get("schema_version") == AUTHORITY_SCHEMA
        and value.get("status") == AUTHORITY_STATUS
        and value.get("stage") == AUTHORITY_STAGE
        and value.get("claim_level") == CLAIM_LEVEL
        and value.get("authorized_scope") == {AUTHORITY_SCOPE: True}
        and value.get("automatic_stage_advance") is False
        and value.get("scientific_GO_or_NO_GO") is None
        and value.get("next_authorized_stage") is None
        and value.get("training_authorized") is False
        and value.get("model_load_authorized") is False
        and value.get("model_forward_authorized") is False
        and value.get("model_backward_authorized") is False
        and value.get("model_update_authorized") is False
        and all(value.get(key) is True for key in AUTHORITY_TRUE_CAPABILITIES)
        and all(value.get(key) is False for key in AUTHORITY_FALSE_CAPABILITIES)
        and value.get("logical_sha256") == logical_sha256(value),
        "V52 pair-feature authority drift",
    )
    return value, actual


_NODE_ARGUMENT_BINDINGS = {
    "contract": "contract",
    "role_free_pair_address": "role_free_pair_address",
    "source_virtual_aggregate": "virtual_aggregate",
    "source_virtual_aggregate_validation": "v49_validation",
    "source_index": "source_index",
    "source_index_publication": "source_index_publication",
    "source_index_commit": "source_index_commit",
    "source_index_validation": "source_index_validation",
    "source_manifest": "source_manifest",
    "source_manifest_validation": "source_manifest_validation",
    "geometry_payload": "geometry_payload",
    "geometry_receipt": "geometry_receipt",
}


def validate_authority_bindings(
    authority: Mapping[str, Any],
    *,
    root: Path,
    execution_manifest_path: Path,
    execution_manifest_sha256: str,
    arguments: Mapping[str, Any],
) -> dict[str, tuple[Path, str]]:
    """Authenticate the exact V52 byte graph before any tensor load."""

    bindings = authority.get("bindings")
    require(
        isinstance(bindings, Mapping)
        and set(bindings) == AUTHORITY_BINDING_KEYS,
        "V52 authority binding keyset drift",
    )
    output_relative = str(arguments.get("output"))
    require(
        bindings.get("output") == {"path": output_relative}
        and bindings.get("execution_interval")
        == {"execution_start": 0, "execution_stop": 12}
        and authority.get("output")
        == {
            "directory": output_relative,
            "artifact": f"{output_relative}/{ARTIFACT_FILENAME}",
            "result": f"{output_relative}/{RESULT_FILENAME}",
            "commit_marker": f"{output_relative}/{COMMIT_FILENAME}",
        }
        and authority.get("resource_contract") == AUTHORITY_RESOURCE_CONTRACT
        and authority.get("execution_start") == 0
        and authority.get("execution_stop") == 12
        and authority.get("expected_pair_query_count") == 12
        and authority.get("expected_pair_member_count") == 24,
        "V52 output/resource/interval contract drift",
    )
    verified: dict[str, tuple[Path, str]] = {}
    for name in sorted(AUTHORITY_BINDING_KEYS - {"output", "execution_interval"}):
        item = bindings.get(name)
        require(
            isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str),
            f"V52 binding envelope drift: {name}",
        )
        path = safe_input_path(root, item["path"], name=f"V52 binding {name}")
        observed = file_sha256(path)
        require(
            observed == require_sha256(item["sha256"], name=f"V52 binding {name}"),
            f"V52 binding byte drift: {name}",
        )
        verified[name] = (path, observed)
    node_path, node_sha = verified["execution_node"]
    require(
        node_path == execution_manifest_path.resolve()
        and node_sha == execution_manifest_sha256,
        "V52 execution-node binding drift",
    )
    for argument_name, binding_name in _NODE_ARGUMENT_BINDINGS.items():
        item = bindings[binding_name]
        require(
            arguments.get(argument_name) == item.get("path")
            and arguments.get(f"{argument_name}_sha256") == item.get("sha256"),
            f"execution node is not V52-bound: {argument_name}",
        )
    require(
        arguments.get("geometry_payload_logical_sha256")
        == bindings["geometry_payload"].get("payload_logical_sha256")
        and arguments.get("execution_start") == 0,
        "execution node logical/interval binding drift",
    )
    return verified


def main() -> None:
    install_quarantine_audit_hook()
    run_start = time.perf_counter()
    (
        raw_arguments,
        execution_manifest_path,
        execution_manifest,
        execution_manifest_sha,
    ) = _execution_manifest_arguments(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--role-free-pair-address", required=True, type=Path)
    parser.add_argument("--role-free-pair-address-sha256", required=True)
    parser.add_argument("--source-virtual-aggregate", required=True, type=Path)
    parser.add_argument("--source-virtual-aggregate-sha256", required=True)
    parser.add_argument("--source-virtual-aggregate-validation", required=True, type=Path)
    parser.add_argument("--source-virtual-aggregate-validation-sha256", required=True)
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--source-index-publication", required=True, type=Path)
    parser.add_argument("--source-index-publication-sha256", required=True)
    parser.add_argument("--source-index-commit", required=True, type=Path)
    parser.add_argument("--source-index-commit-sha256", required=True)
    parser.add_argument("--source-index-validation", required=True, type=Path)
    parser.add_argument("--source-index-validation-sha256", required=True)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--source-manifest-validation", required=True, type=Path)
    parser.add_argument("--source-manifest-validation-sha256", required=True)
    parser.add_argument("--geometry-payload", required=True, type=Path)
    parser.add_argument("--geometry-payload-sha256", required=True)
    parser.add_argument("--geometry-payload-logical-sha256", required=True)
    parser.add_argument("--geometry-receipt", required=True, type=Path)
    parser.add_argument("--geometry-receipt-sha256", required=True)
    parser.add_argument("--execution-start", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(_arguments_to_argv(raw_arguments))

    # There is intentionally no --loss-role-manifest argument.
    root = RC_ROOT.resolve()
    require(Path.cwd().resolve() == root, "formal writer cwd is not RC root")
    authority_path = safe_input_path(root, args.authority, name="authority")
    authority, authority_sha = _validate_authority(
        authority_path, args.authority_sha256
    )
    verified_bindings = validate_authority_bindings(
        authority,
        root=root,
        execution_manifest_path=execution_manifest_path,
        execution_manifest_sha256=execution_manifest_sha,
        arguments=raw_arguments,
    )
    require(args.execution_start == 0, "V52 E0 authority permits only shard [0,12)")
    contract_path = safe_input_path(root, args.contract, name="contract")
    contract_sha = verified_bindings["contract"][1]
    require(
        contract_sha == require_sha256(args.contract_sha256, name="contract"),
        "contract file hash drift",
    )
    pair_path = safe_input_path(root, args.role_free_pair_address, name="role-free pair address")
    aggregate_path = safe_input_path(root, args.source_virtual_aggregate, name="source aggregate")
    aggregate_validation_path = safe_input_path(
        root,
        args.source_virtual_aggregate_validation,
        name="source aggregate validation",
    )
    source_path = safe_input_path(root, args.source_index, name="source index")
    source_publication_path = safe_input_path(
        root, args.source_index_publication, name="source publication receipt"
    )
    source_commit_path = safe_input_path(
        root, args.source_index_commit, name="source commit marker"
    )
    validation_path = safe_input_path(root, args.source_index_validation, name="source validation")
    manifest_path = safe_input_path(root, args.source_manifest, name="source manifest")
    manifest_validation_path = safe_input_path(
        root, args.source_manifest_validation, name="source manifest validation"
    )
    geometry_path = safe_input_path(root, args.geometry_payload, name="geometry payload")
    geometry_receipt_path = safe_input_path(
        root, args.geometry_receipt, name="geometry receipt"
    )
    pair_file_sha = verified_bindings["role_free_pair_address"][1]
    aggregate_file_sha = verified_bindings["virtual_aggregate"][1]
    aggregate_validation_file_sha = verified_bindings["v49_validation"][1]
    source_file_sha_argument = require_sha256(args.source_index_sha256, name="source index")
    source_publication_sha = verified_bindings["source_index_publication"][1]
    source_commit_sha = verified_bindings["source_index_commit"][1]
    validation_file_sha = verified_bindings["source_index_validation"][1]
    manifest_file_sha = verified_bindings["source_manifest"][1]
    manifest_validation_file_sha = verified_bindings["source_manifest_validation"][1]
    geometry_sha = verified_bindings["geometry_payload"][1]
    geometry_receipt_sha = verified_bindings["geometry_receipt"][1]
    require(
        pair_file_sha == require_sha256(args.role_free_pair_address_sha256, name="pair file")
        and aggregate_file_sha
        == require_sha256(args.source_virtual_aggregate_sha256, name="aggregate file")
        and aggregate_validation_file_sha
        == require_sha256(
            args.source_virtual_aggregate_validation_sha256,
            name="aggregate validation file",
        )
        and source_publication_sha
        == require_sha256(
            args.source_index_publication_sha256, name="source publication"
        )
        and source_commit_sha
        == require_sha256(args.source_index_commit_sha256, name="source commit")
        and validation_file_sha
        == require_sha256(args.source_index_validation_sha256, name="validation file")
        and manifest_file_sha
        == require_sha256(args.source_manifest_sha256, name="source manifest file")
        and manifest_validation_file_sha
        == require_sha256(
            args.source_manifest_validation_sha256,
            name="source manifest validation file",
        )
        and geometry_sha
        == require_sha256(args.geometry_payload_sha256, name="geometry payload")
        and geometry_receipt_sha
        == require_sha256(args.geometry_receipt_sha256, name="geometry receipt"),
        "input file hash drift",
    )
    pair_outer = read_role_free_json(pair_path, name="role-free pair address")
    pair_manifest = decode_role_free_pair_manifest(pair_outer)
    pairs = pairs_for_shard(pair_manifest, execution_start=args.execution_start)
    require(len(pairs) == 12, "V52 E0 shard does not contain exactly 12 pairs")
    aggregate = read_role_free_json(aggregate_path, name="source aggregate")
    aggregate_validation = read_role_free_json(
        aggregate_validation_path, name="source aggregate validation"
    )
    require(
        aggregate_validation.get("status")
        == "RCDE_SR0_MT_COMPACT_SOURCE_VIRTUAL_AGGREGATE_INDEPENDENT_VALIDATION_PASS"
        and aggregate_validation.get("validation_pass") is True
        and aggregate_validation.get("aggregate_file_sha256") == aggregate_file_sha
        and aggregate_validation.get("aggregate_logical_sha256")
        == aggregate.get("logical_sha256")
        and aggregate_validation.get("query_count") == 600
        and aggregate_validation.get("shard_count") == 50,
        "V49 aggregate independent validation drift",
    )
    source_validation = read_role_free_json(validation_path, name="source validation")
    source_manifest = read_role_free_json(manifest_path, name="source manifest")
    source_manifest_validation = read_role_free_json(
        manifest_validation_path, name="source manifest validation"
    )
    require(
        source_manifest_validation.get("status")
        == "RCDE_SR0_MT_P_NATURAL_SOURCE_MANIFEST_INDEPENDENT_VALIDATION_PASS"
        and source_manifest_validation.get("validation_pass") is True
        and source_manifest_validation.get("manifest_file_sha256") == manifest_file_sha
        and source_manifest_validation.get("manifest_logical_sha256")
        == source_manifest.get("logical_sha256")
        and source_manifest_validation.get("execution_start") == 0
        and source_manifest_validation.get("execution_stop") == 12
        and source_manifest_validation.get("geometry_payload_sha256") == geometry_sha
        and source_manifest_validation.get("geometry_receipt_sha256")
        == geometry_receipt_sha,
        "natural source manifest independent validation drift",
    )
    binding, _, source_file_sha = _authenticate_source_inputs(
        aggregate=aggregate,
        aggregate_file_sha256=aggregate_file_sha,
        source_path=source_path,
        source_file_sha256=verified_bindings["source_index"][1],
        source_validation=source_validation,
        source_validation_file_sha256=validation_file_sha,
        source_manifest=source_manifest,
        source_manifest_file_sha256=manifest_file_sha,
        execution_start=args.execution_start,
    )
    require(source_file_sha == source_file_sha_argument, "source index file hash drift")
    derived_source_publication_path = safe_input_path(
        root, binding.get("publication_receipt_path"), name="source publication receipt"
    )
    derived_source_commit_path = safe_input_path(
        root, binding.get("commit_path"), name="source commit marker"
    )
    require(
        derived_source_publication_path == source_publication_path
        and derived_source_commit_path == source_commit_path
        and source_publication_sha == binding.get("publication_receipt_file_sha256")
        and source_commit_sha == binding.get("commit_file_sha256"),
        "source publication/commit file hash drift",
    )
    source_publication = read_role_free_json(
        source_publication_path, name="source publication receipt"
    )
    source_commit = read_role_free_json(source_commit_path, name="source commit marker")
    require(
        source_publication.get("artifact_file_sha256") == source_file_sha
        and source_publication.get("artifact_logical_sha256")
        == binding.get("artifact_logical_sha256")
        and source_commit.get("committed") is True
        and source_commit.get("artifact_file_sha256") == source_file_sha
        and source_commit.get("receipt_file_sha256") == source_publication_sha,
        "source publication/commit semantic drift",
    )
    required_memory = qualify_source_artifact_resources(
        artifact_bytes=int(binding["artifact_bytes"]),
        available_memory_bytes=effective_available_memory_bytes(),
    )
    source_load_started = time.perf_counter()
    try:
        raw_source = torch.load(
            source_path, map_location="cpu", weights_only=True, mmap=True
        )
    except Exception as error:
        raise PairFeatureShardError("cannot load qualified compact source shard") from error
    require(isinstance(raw_source, Mapping), "compact source shard is not a mapping")
    selected = select_pair_source_indexes(
        raw_source,
        pairs,
        execution_start=args.execution_start,
        expected_artifact_logical_sha256=str(binding["artifact_logical_sha256"]),
        expected_record_population_sha256=str(binding["record_population_sha256"]),
    )
    del raw_source
    gc.collect()
    source_load_seconds = time.perf_counter() - source_load_started

    derived_geometry_path = safe_input_path(
        root, source_manifest.get("geometry_payload"), name="geometry payload"
    )
    derived_geometry_receipt_path = safe_input_path(
        root, source_manifest.get("geometry_receipt"), name="geometry receipt"
    )
    require(
        derived_geometry_path == geometry_path
        and derived_geometry_receipt_path == geometry_receipt_path
        and geometry_sha == source_manifest.get("geometry_payload_sha256")
        and require_sha256(
            args.geometry_payload_logical_sha256, name="geometry payload logical"
        )
        == source_manifest.get("geometry_payload_logical_sha256")
        and geometry_receipt_sha == source_manifest.get("geometry_receipt_sha256"),
        "geometry receipt file hash drift",
    )
    geometry_load_started = time.perf_counter()
    resolver = Full600PairTokenResolver(
        payload_path=geometry_path,
        payload_sha256=geometry_sha,
        pairs=pairs,
        rc_root=root,
    )
    geometry_load_seconds = time.perf_counter() - geometry_load_started
    lineage = {
        "authority_sha256": authority_sha,
        "execution_manifest_sha256": execution_manifest_sha,
        "contract_file_sha256": contract_sha,
        "writer_file_sha256": verified_bindings["producer"][1],
        "compact_catalog_file_sha256": verified_bindings["compact_catalog"][1],
        "compact_stream_file_sha256": verified_bindings["compact_stream"][1],
        "compact_runtime_file_sha256": verified_bindings["compact_runtime"][1],
        "natural_source_file_sha256": verified_bindings["natural_source"][1],
        "natural_token_loader_file_sha256": verified_bindings[
            "natural_token_loader"
        ][1],
        "feature_schema_module_file_sha256": verified_bindings["feature_schema"][1],
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "role_free_pair_address_file_sha256": pair_file_sha,
        "role_free_pair_manifest_sha256": pair_manifest.manifest_sha256,
        "source_virtual_aggregate_file_sha256": aggregate_file_sha,
        "source_virtual_aggregate_validation_file_sha256": aggregate_validation_file_sha,
        "source_index_file_sha256": source_file_sha,
        "source_index_artifact_logical_sha256": binding["artifact_logical_sha256"],
        "source_record_population_sha256": binding["record_population_sha256"],
        "source_publication_receipt_file_sha256": source_publication_sha,
        "source_commit_file_sha256": source_commit_sha,
        "source_index_validation_file_sha256": validation_file_sha,
        "source_manifest_file_sha256": manifest_file_sha,
        "source_manifest_validation_file_sha256": manifest_validation_file_sha,
        "geometry_payload_file_sha256": geometry_sha,
        "geometry_payload_logical_sha256": require_sha256(
            args.geometry_payload_logical_sha256,
            name="geometry payload logical",
        ),
        "geometry_receipt_file_sha256": geometry_receipt_sha,
    }
    feature_started = time.perf_counter()
    output = build_pair_feature_shard(
        pairs=pairs,
        selected_indexes=selected,
        token_resolver=resolver,
        execution_start=args.execution_start,
        lineage=lineage,
    )
    feature_seconds = time.perf_counter() - feature_started
    token_cache_deserializations = sum(resolver.loader.shard_load_counts.values())
    token_cache_paths = tuple(resolver.loader._shard_load_counts)
    token_cache_mapped_bytes = sum(path.stat().st_size for path in token_cache_paths)
    source_image_paths = tuple(resolver.loader._source_file_hashes)
    source_image_hash_stream_bytes = sum(
        path.stat().st_size for path in source_image_paths
    )
    output.update(
        {
            "max_live_feature_query_count": resolver.max_live_query_count,
            "max_live_selected_reference_entry_count": resolver.max_live_reference_count,
            "max_live_selected_entry_count": resolver.max_live_selected_entry_count,
            "selected_query_entry_count": resolver.selected_query_entry_count,
            "selected_reference_entry_count": resolver.selected_reference_entry_count,
            "token_cache_mapped_file_count": len(token_cache_paths),
            "token_cache_mapped_file_bytes": token_cache_mapped_bytes,
            "natural_source_image_hash_file_count": len(source_image_paths),
            "natural_source_image_hash_stream_bytes": source_image_hash_stream_bytes,
        }
    )
    require(
        resolver.max_live_query_count == 1
        and resolver.max_live_reference_count == 2
        and resolver.max_live_selected_entry_count == 3
        and resolver.selected_query_entry_count == len(pairs)
        and resolver.selected_reference_entry_count == len(pairs) * 2
        and not resolver.query_sources
        and not resolver.reference_sources,
        "selected token live-set contract drift",
    )
    output["access_audit"].update(
        {
            "quarantined_file_attempted_access_count": _QUARANTINED_ACCESS_ATTEMPT_COUNT,
            "loss_role_attempted_access_count": _PROTECTED_ATTEMPT_COUNTS["loss_role"],
            "opened_attempted_access_count": _PROTECTED_ATTEMPT_COUNTS["opened"],
            "sealed_attempted_access_count": _PROTECTED_ATTEMPT_COUNTS["sealed"],
            "C8_attempted_access_count": _PROTECTED_ATTEMPT_COUNTS["C8"],
            "S8_attempted_access_count": _PROTECTED_ATTEMPT_COUNTS["S8"],
            "source_index_artifact_deserialization_count": 1,
            "geometry_payload_deserialization_count": 1,
            "colnomic_token_cache_file_deserialization_count": token_cache_deserializations,
            "colnomic_token_cache_mapped_file_count": len(token_cache_paths),
            "colnomic_token_cache_mapped_file_bytes": token_cache_mapped_bytes,
            "natural_token_payload_read_count": 1 + token_cache_deserializations,
            "natural_source_image_hash_file_count": len(source_image_paths),
            "natural_source_image_hash_read_count": len(source_image_paths),
            "natural_source_image_hash_stream_bytes": source_image_hash_stream_bytes,
            "natural_image_decode_count": 0,
            "dino_token_read_count": 0,
        }
    )
    require(
        _QUARANTINED_ACCESS_ATTEMPT_COUNT == 0,
        "quarantined metadata access count is nonzero",
    )
    output["logical_sha256"] = canonical_sha256(
        {
            key: item
            for key, item in output.items()
            if key not in {"records", "logical_sha256"}
        }
    )
    output_path = args.output.resolve() if args.output.is_absolute() else (root / args.output).resolve()
    try:
        output_path.relative_to(root)
    except ValueError as error:
        raise PairFeatureShardError("output escapes RC root") from error
    receipt, marker = publish_pair_feature_directory(
        output_path,
        output,
        authority_sha256=authority_sha,
        execution_manifest_sha256=execution_manifest_sha,
        run_start=run_start,
        input_deserialization_seconds=source_load_seconds + geometry_load_seconds,
        feature_compute_seconds=feature_seconds,
        source_index_hash_stream_bytes=int(binding["artifact_bytes"]),
        source_index_mapped_file_bytes=int(binding["artifact_bytes"]),
        geometry_payload_hash_stream_bytes=geometry_path.stat().st_size * 2,
        geometry_payload_mapped_file_bytes=geometry_path.stat().st_size,
        geometry_payload_deserialization_count=1,
        colnomic_token_cache_file_deserialization_count=sum(
            resolver.loader.shard_load_counts.values()
        ),
    )
    print(
        json.dumps(
            {
                "status": OUTPUT_STATUS,
                "execution_start": args.execution_start,
                "execution_stop": args.execution_start + QUERY_COUNT_PER_SHARD,
                "pair_query_count": len(pairs),
                "output_directory": output_path.relative_to(root).as_posix(),
                "logical_sha256": output["logical_sha256"],
                "publication_result": receipt,
                "commit_marker": marker,
                "scientific_GO_or_NO_GO": None,
                "automatic_stage_advance": False,
                "next_authorized_stage": None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except PairFeatureShardError as error:
        status = (
            RESOURCE_ABORT_STATUS
            if "resource qualification failed" in str(error)
            else ENGINEERING_ABORT_STATUS
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "claim_level": CLAIM_LEVEL,
                    "error": str(error),
                    "scientific_GO_or_NO_GO": None,
                    "automatic_stage_advance": False,
                    "next_authorized_stage": None,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
