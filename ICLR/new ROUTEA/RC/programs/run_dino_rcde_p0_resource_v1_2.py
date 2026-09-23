#!/usr/bin/env python3
"""Run the measured DINO-RCDE V1.2 P0 resource qualification.

This is a resource/engineering job, not E0 and not a scientific model run.  It
loads the frozen local DINOv2-with-registers checkpoint, encodes the frozen
worst-case cohort selected by the data preflight, executes the exact 65,125
parameter RCDE graph, and measures the work that the parent contract requires.
Any missing real measurement leaves ``benchmark_closed`` and ``resource_gate``
false.  Analytic estimates can never grant P0 readiness.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf.dino_rcde_hf_with_registers_v1_2 import (  # noqa: E402
    FP16TokenCache,
    audit_local_checkpoint,
    build_fp16_cache,
    canonical_sha256,
    extract_intermediate_patch_tokens,
    file_sha256,
    load_frozen_backbone,
    preprocess_canonical_image_path,
    tensor_sha256,
    validate_fp16_cache,
)
from rc_aslo_xf.dino_rcde_v1_2_resource_core import (  # noqa: E402
    DINO_RCDE_V1_2,
    EXPECTED_PARAMETER_COUNT,
    INIT_NAMESPACE,
    assignment_max_abs_error,
    assert_parameter_ledger,
    functional_group_norms,
)


SCHEMA = "rc_dino_rcde_p0_resource_benchmark_v1_2_20260812"
READY = "DINO_RCDE_P0_RESOURCE_READY"
BLOCKED = "DINO_RCDE_RESOURCE_BLOCKED"
SMOKE = "DINO_RCDE_RESOURCE_SMOKE_ONLY"
EXACT_FAST_PLAN = (
    ROOT
    / "plan/DINO_RCDE_P0_V1_2_ASYMMETRIC_EXACT_FAST_EXECUTION_ADDENDUM_20260813.md"
)
EXACT_FAST_VALIDATION_SCHEMA = "dino_rcde_p0_v1_2_exact_fast_independent_validation_v1"
EXACT_FAST_VALIDATION_STATUS = "P0_RESOURCE_EXECUTION_READY"
EXACT_FAST_CLAIM = "ENGINEERING_ONLY_NO_SCIENTIFIC_STAGE_AUTHORIZATION"
EXACT_FAST_TRAIN_QUERY_TILE_ROWS = 8
EXACT_FAST_TRAIN_REFERENCE_TILE_ROWS = 16
EXACT_FAST_C128_QUERY_TILE_ROWS = 8
EXACT_FAST_C128_REFERENCE_TILE_ROWS = 8
EXACT_FAST_PAIR_CHUNK = 256
PROTECTED_ZERO = {
    "formal_runner_C8_runtime_read_count": 0,
    "formal_runner_S8_runtime_read_count": 0,
    "formal_runner_opened_runtime_read_count": 0,
    "formal_runner_sealed_runtime_read_count": 0,
    "home_files_modified": 0,
    "target_free_runtime_legacy_query_carrier_read_count": 0,
    "project_agent_C8_role_metadata_untouched": False,
    "C8_image_token_score_or_result_read_count": 0,
}


class ResourceError(RuntimeError):
    pass


@contextmanager
def _exclusive_cache_guard(cache_root: Path) -> Iterable[None]:
    """Serialize writers to the stable shared cache without touching HOME."""

    cache_root.mkdir(parents=True, exist_ok=True)
    guard = cache_root / ".rcde_v1_2_writer.lock"
    with guard.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _inherited_colnomic_tensor_sha256(value: torch.Tensor) -> str:
    """Replay the tensor-hash contract used by the inherited V1.1 shards.

    V1.1 sealed ``image_token_sha256`` as the concatenation of the dtype
    string, JSON list shape, and contiguous tensor bytes.  RCDE V1.2 uses a
    newer JSON-object header for artifacts it creates itself.  The two hashes
    intentionally have different namespaces, so source-cache validation must
    replay the producer's V1.1 contract rather than the V1.2 artifact helper.
    """

    if not isinstance(value, torch.Tensor):
        raise ResourceError("inherited ColNomic tensor hash input is not a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ResourceError(f"immutable JSON drift: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = torch.load(path, map_location="cpu", weights_only=False)

        def equal(left: Any, right: Any) -> bool:
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                return torch.equal(left, right)
            if isinstance(left, dict) and isinstance(right, dict):
                return set(left) == set(right) and all(equal(left[key], right[key]) for key in left)
            if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
            return left == right

        if not equal(observed, value):
            raise ResourceError(f"immutable tensor artifact drift: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    torch.save(value, partial)
    try:
        os.link(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _resolve(binding: Mapping[str, Any]) -> Path:
    return (ROOT / str(binding["path"])).resolve()


def _load_exact_fast_qualification(path: Path) -> dict[str, Any]:
    """Bind the formal runner to the independently qualified exact-fast path."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise ResourceError(f"missing exact-fast independent qualification: {resolved}")
    qualification = json.loads(resolved.read_text(encoding="utf-8"))
    detached = dict(qualification)
    logical_sha256 = detached.pop("logical_sha256", None)
    plan = qualification.get("plan")
    gates = qualification.get("gates")
    projection = qualification.get("projection")
    execution_geometry = qualification.get("execution_geometry")
    plan_path = Path(str(plan.get("path", ""))) if isinstance(plan, dict) else Path()
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    if (
        qualification.get("schema_version") != EXACT_FAST_VALIDATION_SCHEMA
        or qualification.get("status") != EXACT_FAST_VALIDATION_STATUS
        or qualification.get("claim_level") != EXACT_FAST_CLAIM
        or qualification.get("resource_execution_ready") is not True
        or qualification.get("scientific_go") is not None
        or qualification.get("next_authorized_stage") is not None
        or qualification.get("formal_runner_integration_required_before_p0") is not True
        or logical_sha256 != _json_sha256(detached)
        or not isinstance(plan, dict)
        or plan_path.resolve() != EXACT_FAST_PLAN.resolve()
        or plan.get("sha256") != file_sha256(EXACT_FAST_PLAN)
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
        or not isinstance(projection, dict)
        or int(projection.get("learned_arm_count", -1)) != 6
        or int(projection.get("mandatory_control_count", -1)) != 5
        or int(projection.get("n1_inner_rcde_fit_count", -1)) != 24
        or projection.get("a1_excluded_from_total") is not True
        or float(projection.get("p0_through_n1_gpu_hours_with_25pct_safety", math.inf)) > 256.0
        or not isinstance(execution_geometry, dict)
        or qualification.get("training_query_tile_rows")
        != EXACT_FAST_TRAIN_QUERY_TILE_ROWS
        or qualification.get("training_reference_tile_rows")
        != EXACT_FAST_TRAIN_REFERENCE_TILE_ROWS
        or qualification.get("c128_query_tile_rows")
        != EXACT_FAST_C128_QUERY_TILE_ROWS
        or qualification.get("c128_reference_tile_rows")
        != EXACT_FAST_C128_REFERENCE_TILE_ROWS
        or execution_geometry.get("training_query_tile_rows")
        != EXACT_FAST_TRAIN_QUERY_TILE_ROWS
        or execution_geometry.get("training_reference_tile_rows")
        != EXACT_FAST_TRAIN_REFERENCE_TILE_ROWS
        or execution_geometry.get("c128_query_tile_rows")
        != EXACT_FAST_C128_QUERY_TILE_ROWS
        or execution_geometry.get("c128_reference_tile_rows")
        != EXACT_FAST_C128_REFERENCE_TILE_ROWS
    ):
        raise ResourceError("exact-fast independent qualification drift or failed gate")
    return {
        "qualification_receipt_path": str(resolved),
        "qualification_receipt_sha256": file_sha256(resolved),
        "qualification_validation_path": str(resolved),
        "qualification_validation_sha256": file_sha256(resolved),
        "plan_path": str(EXACT_FAST_PLAN.resolve()),
        "plan_sha256": file_sha256(EXACT_FAST_PLAN),
        "validation_logical_sha256": logical_sha256,
        "validation_schema_version": qualification["schema_version"],
        "validation_status": qualification["status"],
        "validation_claim_level": qualification["claim_level"],
        "resource_execution_ready": True,
        "scientific_go": None,
        "learned_arm_count": 6,
        "mandatory_control_count": 5,
        "n1_inner_rcde_fit_count": 24,
        "training_query_tile_rows": EXACT_FAST_TRAIN_QUERY_TILE_ROWS,
        "training_reference_tile_rows": EXACT_FAST_TRAIN_REFERENCE_TILE_ROWS,
        "c128_query_tile_rows": EXACT_FAST_C128_QUERY_TILE_ROWS,
        "c128_reference_tile_rows": EXACT_FAST_C128_REFERENCE_TILE_ROWS,
        "pair_chunk_size": EXACT_FAST_PAIR_CHUNK,
    }


def _assert_exact_fast_evidence(evidence: Any) -> None:
    if (
        evidence.assignment is not None
        or evidence.matrix_audit_enabled is not False
        or dict(evidence.matrix_sha256)
        or list(evidence.matrix_audit_slices)
        or evidence.analytic_null is not True
        or evidence.full_consensus_logits_resident is not True
        or int(evidence.resident_consensus_logit_elements) <= 0
        or int(evidence.logical_cost_volume_build_count) != 1
        or int(evidence.null_consensus_evaluation_count) != 0
        or tuple(evidence.null_pass_tile_counts) != (0, 0, 0)
        or int(evidence.real_consensus_evaluation_count)
        != int(evidence.consensus_tile_count_per_pass)
    ):
        raise ResourceError("exact-fast candidate execution semantic drift")


def _exact_fast_resident_receipt(
    evidence_rows: Sequence[Any], *, query_tile_rows: int, reference_tile_rows: int,
) -> dict[str, Any]:
    if not evidence_rows:
        raise ResourceError("exact-fast resident ledger is empty")
    for evidence in evidence_rows:
        _assert_exact_fast_evidence(evidence)
    elements = [int(evidence.resident_consensus_logit_elements) for evidence in evidence_rows]
    real_evaluations = [int(evidence.real_consensus_evaluation_count) for evidence in evidence_rows]
    return _exact_fast_resident_receipt_from_counts(
        elements, real_evaluations,
        query_tile_rows=query_tile_rows,
        reference_tile_rows=reference_tile_rows,
    )


def _exact_fast_resident_receipt_from_counts(
    elements: Sequence[int], real_evaluations: Sequence[int], *,
    query_tile_rows: int, reference_tile_rows: int,
) -> dict[str, Any]:
    elements = [int(value) for value in elements]
    real_evaluations = [int(value) for value in real_evaluations]
    if not elements or len(elements) != len(real_evaluations):
        raise ResourceError("exact-fast resident count ledger drift")
    return {
        "execution_path": "EXACT_FAST_REAL_TILE_REUSE_ANALYTIC_NULL",
        "query_tile_rows": int(query_tile_rows),
        "reference_tile_rows": int(reference_tile_rows),
        "analytic_null": True,
        "full_consensus_logits_resident": True,
        "matrix_audit_enabled": False,
        "full_cross_cost_materialized": False,
        "full_assignment_materialized": False,
        "resident_consensus_logit_elements_per_candidate": elements,
        "resident_consensus_logit_elements_total": sum(elements),
        "resident_consensus_logit_elements_max": max(elements),
        "resident_consensus_logit_bytes_total": 4 * sum(elements),
        "resident_consensus_logit_bytes_max": 4 * max(elements),
        "real_consensus_evaluations_per_candidate": real_evaluations,
        "real_consensus_evaluations_total": sum(real_evaluations),
        "null_consensus_evaluations_per_candidate": [0] * len(elements),
        "null_consensus_evaluations_total": 0,
    }


def _validate_inputs(
    protocol_path: Path, data_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "rc_dino_rcde_p0_protocol_v1_2_20260812":
        raise ResourceError("V1.2 protocol schema drift")
    if int(protocol.get("resource_limits", {}).get("required_parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise ResourceError("protocol parameter-count drift")
    if protocol.get("automatic_stage_advance") is not False:
        raise ResourceError("automatic stage advancement is not disabled")
    data_result = json.loads((data_dir / "result.json").read_text(encoding="utf-8"))
    data_validation = json.loads((data_dir / "data_validation.json").read_text(encoding="utf-8"))
    if data_result.get("status") != "DINO_RCDE_P0_DATA_STATISTICS_READY_FOR_RESOURCE":
        raise ResourceError("data/statistics P0 did not authorize resource measurement")
    if (
        data_validation.get("status") != "DINO_RCDE_P0_DATA_VALIDATION_PASS"
        or data_validation.get("next_required_component") != "P0_RESOURCE"
        or data_validation.get("result_sha256") != file_sha256(data_dir / "result.json")
    ):
        raise ResourceError("independent data validation did not authorize resource measurement")
    shape_path = data_dir / "resource_shape_ledger.json"
    if data_result.get("resource_shape_ledger_sha256") != file_sha256(shape_path):
        raise ResourceError("resource shape-ledger binding drift")
    shape = json.loads(shape_path.read_text(encoding="utf-8"))
    required = {
        "Q_NMAX", "Q_QUERY_P50", "Q_QUERY_P95", "Q_REFSUM_MAX", "Q_PAIR_MAX",
        "REFERENCE_P50", "REFERENCE_P95", "REFERENCE_MAX",
        "RAW_PIXEL_P50", "RAW_PIXEL_P95", "RAW_PIXEL_MAX",
    }
    if shape.get("schema_version") != "rcde_resource_shape_ledger_v1_2":
        raise ResourceError("resource shape-ledger schema drift")
    if set(shape.get("worst_case_cohort", {})) != required:
        raise ResourceError("resource cohort is not the frozen eleven-case cohort")
    if int(shape.get("query_count", -1)) != 600:
        raise ResourceError("resource query population drift")
    # The execution ordinal is dense 0..599 and is deliberately distinct from
    # the historical/global query ordinal.  Never use the latter as a tensor row.
    query_rows = [row for row in shape["rows"] if row.get("kind") == "query"]
    if sorted(int(row["execution_ordinal"]) for row in query_rows) != list(range(600)):
        raise ResourceError("dense execution ordinal mapping drift")
    data_role_path = data_dir / "rcde_data_role_receipt.json"
    if data_result.get("data_role_receipt_sha256") != file_sha256(data_role_path):
        raise ResourceError("data-role receipt binding drift")
    data_role = json.loads(data_role_path.read_text(encoding="utf-8"))
    matched = data_role.get("matched_colnomic_control")
    if not isinstance(matched, dict) or matched.get("eligible") is not True:
        raise ResourceError("matched ColNomic resource ledger is ineligible")
    if matched.get("matched_colnomic_resource_ledger_scope") != (
        "EXISTING_SOURCE_CACHE_READ_PLUS_FOUR_LAYER_ISOMETRIC_LIFT"
    ):
        raise ResourceError("matched ColNomic resource scope drift")
    return protocol, shape, matched


def _row_maps(shape: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    queries = {str(row["id"]): row for row in shape["rows"] if row.get("kind") == "query"}
    references = {
        int(row["physical_row"]): row for row in shape["rows"] if row.get("kind") == "reference"
    }
    if len(queries) != 600 or len(references) != int(shape["reference_union_count"]):
        raise ResourceError("resource shape row population drift")
    return queries, references


def _cache_logical_hash(payload: Mapping[str, Any]) -> str:
    return _json_sha256({
        "tokens": tensor_sha256(payload["tokens_fp16"]),
        "mask": tensor_sha256(payload["valid_patch_mask"]),
        "geometry": payload["geometry_receipt"],
        "model": payload["model_checkpoint_logical_sha256"],
        "source": payload["source_image_sha256"],
    })


_TIMING_AUTHORITY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _timing_authority(device: torch.device, model_checkpoint_logical_sha256: str) -> dict[str, Any]:
    """Bind reusable timing to the exact hardware, runtime and implementation."""

    device_name = torch.cuda.get_device_name(device)
    key = (device_name, model_checkpoint_logical_sha256)
    if key not in _TIMING_AUTHORITY_CACHE:
        wrapper = ROOT / "src/rc_aslo_xf/dino_rcde_hf_with_registers_v1_2.py"
        body = {
            "measurement_contract": "RCDE_FP16_CACHE_TIMING_AUTHORITY_V1_2",
            "device_name": device_name,
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": str(torch.version.cuda),
            "cudnn_version": int(torch.backends.cudnn.version() or 0),
            "resource_runner_sha256": file_sha256(Path(__file__).resolve()),
            "hf_wrapper_sha256": file_sha256(wrapper),
            "preprocess_authority_sha256": file_sha256(wrapper),
            "model_checkpoint_logical_sha256": model_checkpoint_logical_sha256,
        }
        body["logical_sha256"] = _json_sha256(body)
        _TIMING_AUTHORITY_CACHE[key] = body
    return dict(_TIMING_AUTHORITY_CACHE[key])


def _measurement_path(path: Path, authority: Mapping[str, Any]) -> Path:
    return path.with_name(f"{path.name}.measurement.{authority['logical_sha256'][:16]}.json")


def _validate_cache_measurement(
    measurement: Mapping[str, Any], *, path: Path, payload: Mapping[str, Any],
    authority: Mapping[str, Any], source_sha256: str,
) -> None:
    detached = dict(measurement)
    observed = detached.pop("logical_sha256", None)
    if (
        measurement.get("schema_version") != "rcde_fp16_cache_measurement_v1_2"
        or observed != _json_sha256(detached)
        or measurement.get("fresh_build_event") is not True
        or measurement.get("cache_artifact") != path.name
        or measurement.get("cache_logical_sha256") != payload["logical_sha256"]
        or measurement.get("source_image_sha256") != source_sha256
        or measurement.get("model_checkpoint_logical_sha256")
        != payload["model_checkpoint_logical_sha256"]
        or measurement.get("timing_authority") != authority
        or int(measurement.get("cache_artifact_bytes", -1)) != path.stat().st_size
        or int(measurement.get("source_read_operation_count", -1)) != 2
        or int(measurement.get("cache_artifact_write_operation_count", -1)) != 1
        or int(measurement.get("cache_artifact_validation_read_operation_count", -1)) != 1
        or int(measurement.get("fresh_build_source_io_bytes", -1))
        != 2 * int(measurement.get("source_file_bytes", -2))
        or int(measurement.get("fresh_build_artifact_io_bytes", -1))
        != 2 * int(measurement.get("timing_probe_artifact_bytes", -2))
        or int(measurement.get("fresh_build_logical_io_bytes", -1))
        != int(measurement.get("fresh_build_source_io_bytes", -2))
        + int(measurement.get("fresh_build_artifact_io_bytes", -3))
        or not all(
            math.isfinite(float(measurement.get(name, math.nan)))
            and float(measurement.get(name, 0.0)) >= 0.0
            for name in (
                "encode_gpu_wall_seconds", "encode_cpu_seconds",
                "encode_peak_cpu_rss_delta_bytes", "encode_peak_cuda_allocated_bytes",
                "encode_peak_cuda_reserved_bytes",
            )
        )
    ):
        raise ResourceError(f"cache timing authority or measurement drift: {path}")


def _load_cache(path: Path, expected_source_sha256: str, model_checkpoint_logical_sha256: str) -> tuple[dict[str, Any], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "tokens_fp16", "valid_patch_mask", "geometry_receipt", "cache_receipt",
        "model_checkpoint_logical_sha256", "source_image_sha256", "encode_seconds",
        "encode_cpu_seconds", "encode_peak_cpu_rss_delta_bytes",
        "encode_peak_cuda_allocated_bytes", "encode_peak_cuda_reserved_bytes",
        "logical_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ResourceError(f"token cache schema drift: {path}")
    if payload["source_image_sha256"] != expected_source_sha256:
        raise ResourceError(f"token cache source drift: {path}")
    if payload["model_checkpoint_logical_sha256"] != model_checkpoint_logical_sha256:
        raise ResourceError(f"token cache model drift: {path}")
    cache = FP16TokenCache(payload["tokens_fp16"], payload["valid_patch_mask"], payload["cache_receipt"])
    validate_fp16_cache(cache)
    logical = _cache_logical_hash(payload)
    if logical != payload["logical_sha256"]:
        raise ResourceError(f"token cache logical hash drift: {path}")
    return payload, logical


def _materialize_cache_unlocked(
    *,
    image_row: Mapping[str, Any],
    path: Path,
    backbone: torch.nn.Module,
    model_checkpoint_logical_sha256: str,
    device: torch.device,
) -> tuple[dict[str, Any], bool, float, int]:
    source = Path(str(image_row["source_image_path"]))
    source_sha = str(image_row["source_image_sha256"])
    authority = _timing_authority(device, model_checkpoint_logical_sha256)
    end_to_end_started = time.perf_counter()
    end_to_end_cpu_started = time.process_time()
    rss_before = _rss_bytes()
    if not source.is_file() or file_sha256(source) != source_sha:
        raise ResourceError(f"source image hash drift: {source}")
    existing_payload: dict[str, Any] | None = None
    measurement_path = _measurement_path(path, authority)
    if path.exists():
        existing_payload, _ = _load_cache(path, source_sha, model_checkpoint_logical_sha256)
        # A cache payload may be reused across nodes/runtimes, but its timing
        # may not.  Only a sidecar in the current authority namespace can close
        # the resource gate; otherwise execute the full timing path below.
        if measurement_path.is_file():
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            _validate_cache_measurement(
                measurement, path=path, payload=existing_payload,
                authority=authority, source_sha256=source_sha,
            )
            return existing_payload, True, float(existing_payload["encode_seconds"]), path.stat().st_size
    source_file_bytes = source.stat().st_size
    pixels, geometry, geometry_receipt = preprocess_canonical_image_path(source)
    if (
        list(geometry.grid_hw) != [int(image_row["grid_h"]), int(image_row["grid_w"])]
        or int(geometry.valid_patch_mask.sum()) != int(image_row["valid_tokens"])
    ):
        raise ResourceError("image preprocessing shape differs from frozen ledger")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cpu_started = time.process_time()
    gpu_rss_before = _rss_bytes()
    result = extract_intermediate_patch_tokens(
        backbone, pixels.to(device), geometry.valid_patch_mask.to(device)
    )
    _sync(device)
    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - cpu_started
    cache = build_fp16_cache(
        result,
        geometry_receipts=[geometry_receipt],
        model_receipt_sha256=model_checkpoint_logical_sha256,
    )
    payload: dict[str, Any] = {
        "tokens_fp16": cache.tokens_fp16,
        "valid_patch_mask": cache.valid_patch_mask,
        "geometry_receipt": geometry_receipt,
        "cache_receipt": dict(cache.receipt),
        "model_checkpoint_logical_sha256": model_checkpoint_logical_sha256,
        "source_image_sha256": source_sha,
        "encode_seconds": elapsed,
        "encode_cpu_seconds": cpu_elapsed,
        "encode_peak_cpu_rss_delta_bytes": max(0, _rss_bytes() - gpu_rss_before),
        "encode_peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "encode_peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    cache_receipt_without_logical = dict(payload["cache_receipt"])
    cache_receipt_without_logical.pop("logical_sha256", None)
    payload["cache_receipt"]["logical_sha256"] = canonical_sha256(cache_receipt_without_logical)
    payload["logical_sha256"] = _cache_logical_hash(payload)
    timing_artifact = path
    if existing_payload is None:
        _atomic_torch(path, payload)
    else:
        if existing_payload["logical_sha256"] != payload["logical_sha256"]:
            raise ResourceError("authority remeasurement changed scientific cache payload")
        timing_artifact = path.with_name(f".{path.name}.timing_probe.{os.getpid()}.pt")
        _atomic_torch(timing_artifact, payload)
    # A fresh build ends only after the artifact has been reopened and
    # semantically validated.  The source is read once for SHA256 and once by
    # image decode; the cache is written once and read once by validation.
    loaded, logical = _load_cache(timing_artifact, source_sha, model_checkpoint_logical_sha256)
    if logical != payload["logical_sha256"]:
        raise ResourceError("fresh token cache did not resume exactly")
    timing_artifact_bytes = timing_artifact.stat().st_size
    if existing_payload is not None:
        timing_artifact.unlink()
        loaded = existing_payload
    artifact_bytes = path.stat().st_size
    measurement = {
        "schema_version": "rcde_fp16_cache_measurement_v1_2",
        "fresh_build_event": True,
        "cache_artifact": path.name,
        "cache_logical_sha256": payload["logical_sha256"],
        "source_image_sha256": source_sha,
        "model_checkpoint_logical_sha256": model_checkpoint_logical_sha256,
        "timing_authority": authority,
        "end_to_end_cache_wall_seconds": time.perf_counter() - end_to_end_started,
        "end_to_end_cache_cpu_seconds": time.process_time() - end_to_end_cpu_started,
        "end_to_end_peak_cpu_rss_delta_bytes": max(0, _rss_bytes() - rss_before),
        "encode_gpu_wall_seconds": elapsed,
        "encode_cpu_seconds": cpu_elapsed,
        "encode_peak_cpu_rss_delta_bytes": int(payload["encode_peak_cpu_rss_delta_bytes"]),
        "encode_peak_cuda_allocated_bytes": int(payload["encode_peak_cuda_allocated_bytes"]),
        "encode_peak_cuda_reserved_bytes": int(payload["encode_peak_cuda_reserved_bytes"]),
        "source_file_bytes": source_file_bytes,
        "cache_artifact_bytes": artifact_bytes,
        "timing_probe_artifact_bytes": timing_artifact_bytes,
        "source_read_operation_count": 2,
        "cache_artifact_write_operation_count": 1,
        "cache_artifact_validation_read_operation_count": 1,
        "fresh_build_source_io_bytes": 2 * source_file_bytes,
        "fresh_build_artifact_io_bytes": 2 * timing_artifact_bytes,
        "fresh_build_logical_io_bytes": 2 * source_file_bytes + 2 * timing_artifact_bytes,
    }
    measurement["logical_sha256"] = _json_sha256(measurement)
    _write_json(measurement_path, measurement)
    return loaded, False, elapsed, artifact_bytes


def _materialize_cache(
    *,
    image_row: Mapping[str, Any],
    path: Path,
    backbone: torch.nn.Module,
    model_checkpoint_logical_sha256: str,
    device: torch.device,
) -> tuple[dict[str, Any], bool, float, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / f".{path.name}.writer.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            return _materialize_cache_unlocked(
                image_row=image_row,
                path=path,
                backbone=backbone,
                model_checkpoint_logical_sha256=model_checkpoint_logical_sha256,
                device=device,
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _core_inputs(payload: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    tokens = payload["tokens_fp16"]
    mask = payload["valid_patch_mask"]
    if tokens.shape[0] != 1 or mask.shape[0] != 1:
        raise ResourceError("resource benchmark requires one-image cache artifacts")
    tokens = tokens[0].to(device=device, dtype=torch.float32)
    mask = mask[0].to(device=device, dtype=torch.bool)
    grid = tuple(int(value) for value in mask.shape)
    if tuple(tokens.shape) != (4, math.prod(grid), 768):
        raise ResourceError("RCDE cache/core shape drift")
    return tokens, mask, grid


def _parameter_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {name: tensor_sha256(value.detach()) for name, value in model.named_parameters()}


def _gradient_update_audit(device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exercise the parent contract's tensor/group gradient and update gates."""

    model = DINO_RCDE_V1_2().to(device)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    generator = torch.Generator(device="cpu").manual_seed(17012026)
    fixtures = []
    for index, (qg, gg, cg) in enumerate((((4, 5), (5, 4), (4, 6)), ((5, 3), (3, 7), (6, 3)), ((3, 6), (5, 5), (4, 4)))):
        def draw(grid: tuple[int, int], shift: float) -> torch.Tensor:
            return (torch.randn((4, math.prod(grid), 768), generator=generator) + shift).to(device)
        qm = torch.ones(qg, dtype=torch.bool, device=device); qm[-1, -1] = False
        gm = torch.ones(gg, dtype=torch.bool, device=device); gm[-1, -1] = False
        cm = torch.ones(cg, dtype=torch.bool, device=device)
        fixtures.append((draw(qg, index * 0.1), draw(gg, 0.2), draw(cg, -0.2), qm, gm, cm, qg, gg, cg, 1.0 if index != 1 else -1.0))

    accumulated_square = {name: torch.zeros_like(value, dtype=torch.float64, device="cpu") for name, value in model.named_parameters()}
    none_counts = {name: 0 for name, _ in model.named_parameters()}
    nonfinite_counts = {name: 0 for name, _ in model.named_parameters()}
    first_scalar_counts: dict[str, int] | None = None
    # Each fixture is backpropagated after an independent zero_grad.  Squared
    # tensor norms are accumulated across the suite, so opposite directions
    # cannot cancel and no unsupported per-scalar nonzero gate is introduced.
    for fixture_index, fixture in enumerate(fixtures):
        model.zero_grad(set_to_none=True)
        q, g, c, qm, gm, cm, qg, gg, cg, direction = fixture
        _, _, pair = model.forward_pair(q, g, c, qm, gm, cm, qg, gg, cg)
        F.softplus(-direction * pair.logit).backward()
        scalar_nonfinite = scalar_inactive = scalar_total = 0
        for name, value in model.named_parameters():
            grad = value.grad
            if grad is None:
                none_counts[name] += 1
                scalar_inactive += value.numel(); scalar_total += value.numel(); continue
            detached = grad.detach()
            scalar_total += detached.numel()
            bad = int((~torch.isfinite(detached)).sum().item())
            nonfinite_counts[name] += bad
            scalar_nonfinite += bad
            scalar_inactive += int(detached.eq(0).sum().item())
            accumulated_square[name] += detached.to("cpu", torch.float64).square()
        if fixture_index == 0:
            first_scalar_counts = {
                "total": scalar_total,
                "inactive": scalar_inactive,
                "nonfinite": scalar_nonfinite,
            }

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    losses: list[float] = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        for fixture in fixtures:
            q, g, c, qm, gm, cm, qg, gg, cg, direction = fixture
            _, _, pair = model.forward_pair(q, g, c, qm, gm, cm, qg, gg, cg)
            loss = loss + F.softplus(-direction * pair.logit)
        loss = loss / len(fixtures)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        final_loss = torch.zeros((), device=device)
        for fixture in fixtures:
            q, g, c, qm, gm, cm, qg, gg, cg, direction = fixture
            _, _, pair = model.forward_pair(q, g, c, qm, gm, cm, qg, gg, cg)
            final_loss += F.softplus(-direction * pair.logit)
        final_loss = final_loss / len(fixtures)
    accumulated_inactive = sum(int(value.eq(0).sum()) for value in accumulated_square.values())
    accumulated_nonfinite = sum(int((~torch.isfinite(value)).sum()) for value in accumulated_square.values())
    updated_scalar_count = sum(
        int(value.detach().ne(initial[name]).sum().item()) for name, value in model.named_parameters()
    )
    delta_groups: dict[str, float] = {}
    named = dict(model.named_parameters())
    for group, names in model.functional_parameter_groups().items():
        total = sum(float((named[name].detach() - initial[name]).to(torch.float64).square().sum().cpu()) for name in names)
        delta_groups[group] = math.sqrt(total)
    tensor_gradient_contract = {
        name: {
            "fixture_grad_none_count": none_counts[name],
            "nonfinite_scalar_count": nonfinite_counts[name],
            "suite_squared_l2": float(accumulated_square[name].sum()),
            "suite_l2": math.sqrt(float(accumulated_square[name].sum())),
        }
        for name in accumulated_square
    }
    receipt = {
        "parameter_count": model.parameter_count(),
        "parameter_ledger": dict(assert_parameter_ledger(model)),
        "initialization_namespace": INIT_NAMESPACE,
        "initial_parameter_hashes": {name: tensor_sha256(value) for name, value in initial.items()},
        "first_backward_scalar_counts": first_scalar_counts,
        "suite_accumulated_inactive_scalar_count": accumulated_inactive,
        "suite_accumulated_nonfinite_scalar_count": accumulated_nonfinite,
        "fixture_count": len(fixtures),
        "per_fixture_zero_grad_before_backward": True,
        "tensor_gradient_contract": tensor_gradient_contract,
        "updated_scalar_count": updated_scalar_count,
        "functional_group_gradient_norms": {
            group: math.sqrt(sum(float(accumulated_square[name].sum()) for name in names))
            for group, names in model.functional_parameter_groups().items()
        },
        "functional_group_parameter_delta_norms": delta_groups,
        "optimizer": {"name": "AdamW", "lr": 3.0e-4, "weight_decay": 1.0e-4, "gradient_clip_l2": 1.0},
        "step_losses_before_update": losses,
        "loss_after_two_updates": float(final_loss.detach().cpu()),
    }
    receipt["gate"] = (
        receipt["parameter_count"] == EXPECTED_PARAMETER_COUNT
        and all(
            row["fixture_grad_none_count"] == 0
            and row["nonfinite_scalar_count"] == 0
            and row["suite_l2"] > 0.0
            and math.isfinite(row["suite_l2"])
            for row in tensor_gradient_contract.values()
        )
        and all(value > 0.0 and math.isfinite(value) for value in receipt["functional_group_gradient_norms"].values())
        and all(value > 0.0 and math.isfinite(value) for value in delta_groups.values())
        and float(final_loss) < losses[0]
    )
    return receipt, {name: tensor_sha256(value.detach()) for name, value in model.named_parameters()}


def _math_audit(device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model = DINO_RCDE_V1_2().to(device).eval()
    gen = torch.Generator().manual_seed(717)
    qg, gg, cg = (5, 6), (6, 5), (4, 7)
    q = torch.randn((4, math.prod(qg), 768), generator=gen).to(device)
    g = torch.randn((4, math.prod(gg), 768), generator=gen).to(device)
    c = torch.randn((4, math.prod(cg), 768), generator=gen).to(device)
    qm = torch.ones(qg, dtype=torch.bool, device=device); qm[-1, -1] = False
    gm = torch.ones(gg, dtype=torch.bool, device=device); gm[-1, -1] = False
    cm = torch.ones(cg, dtype=torch.bool, device=device)
    with torch.no_grad():
        eg, ec, pair = model.forward_pair(q, g, c, qm, gm, cm, qg, gg, cg)
        _, _, swapped = model.forward_pair(q, c, g, qm, cm, gm, qg, cg, gg)
        _, _, same = model.forward_pair(q, g, g, qm, gm, gm, qg, gg, gg)
        ng = model.decode_zero_cost_control(qm, gm, qg, gg)
        nc = model.decode_zero_cost_control(qm, cm, qg, cg)
        no_ref = model.compare_relational(ng.relational, nc.relational, qm)
        tiled = model.decode_candidate(
            q, g, qm, gm, qg, gg,
            streaming_chunk_size=7,
            consensus_tile_shape=(3, 4, 4, 3),
        )
        _, coverage = model.consensus.forward_tiled(
            eg.cost, qg, gg, qm, gm, tile_shape=(3, 4, 4, 3)
        )
        true_stream = model.decode_candidate_true_streaming(
            q, g, qm, gm, qg, gg,
            query_tile_rows=2, reference_tile_rows=2,
            materialize_assignment_for_audit=True,
        )
    assignment_error = assignment_max_abs_error(eg.assignment, tiled.assignment)
    if true_stream.assignment is None:
        raise ResourceError("small dense true-stream audit did not expose assignment")
    true_stream_error = assignment_max_abs_error(eg.assignment, true_stream.assignment)
    receipt = {
        "swap_contribution_max_abs_error": float((pair.contributions + swapped.contributions).abs().max().cpu()),
        "swap_logit_abs_error": float((pair.logit + swapped.logit).abs().cpu()),
        "same_ref_nonzero_count": int(torch.count_nonzero(same.evidence).cpu()),
        "no_ref_nonzero_count": int(torch.count_nonzero(no_ref.evidence).cpu()),
        "pair_mean_closure_abs_error": float((pair.logit - pair.contributions[qm.flatten()].mean()).abs().cpu()),
        "dense_tiled_logits_max_abs_error": float((eg.logits - tiled.logits).abs().max().cpu()),
        "dense_stream_assignment_max_abs_errors": assignment_error,
        "dense_tiled_relational_max_abs_error": float((eg.relational - tiled.relational).abs().max().cpu()),
        "dense_true_stream_assignment_max_abs_errors": true_stream_error,
        "dense_true_stream_relational_max_abs_error": float(
            (eg.relational - true_stream.relational).abs().max().cpu()
        ),
        "dense_true_stream_u_max_abs_error": float(
            (eg.assignment.u - true_stream.u).abs().max().cpu()
        ),
        "dense_true_stream_v_max_abs_error": float(
            (eg.assignment.v - true_stream.v).abs().max().cpu()
        ),
        "true_stream_real_pass_tile_counts": list(true_stream.real_pass_tile_counts),
        "true_stream_null_pass_tile_counts": list(true_stream.null_pass_tile_counts),
        "true_stream_matrix_sha256": dict(true_stream.matrix_sha256),
        "true_stream_matrix_audit_slices": list(true_stream.matrix_audit_slices),
        "true_stream_full_cost_materialized": False,
        "true_stream_full_assignment_materialized_in_deployment": False,
        "tile_coverage_min": int(coverage.min().cpu()),
        "tile_coverage_max": int(coverage.max().cpu()),
    }
    receipt["gate"] = (
        receipt["swap_contribution_max_abs_error"] <= 1.0e-7
        and receipt["swap_logit_abs_error"] <= 1.0e-7
        and receipt["same_ref_nonzero_count"] == 0
        and receipt["no_ref_nonzero_count"] == 0
        and receipt["pair_mean_closure_abs_error"] <= 1.0e-7
        and receipt["dense_tiled_logits_max_abs_error"] <= 1.0e-6
        and max(assignment_error.values()) <= 1.0e-6
        and receipt["dense_tiled_relational_max_abs_error"] <= 1.0e-6
        and max(true_stream_error.values()) <= 1.0e-6
        and receipt["dense_true_stream_relational_max_abs_error"] <= 1.0e-6
        and receipt["dense_true_stream_u_max_abs_error"] <= 1.0e-6
        and receipt["dense_true_stream_v_max_abs_error"] <= 1.0e-6
        and len(set(true_stream.real_pass_tile_counts)) == 1
        and len(set(true_stream.null_pass_tile_counts)) == 1
        and receipt["tile_coverage_min"] == receipt["tile_coverage_max"] == 1
    )
    tensors = {
        "real_contributions": pair.contributions.detach().cpu(),
        "swapped_contributions": swapped.contributions.detach().cpu(),
        "real_logit": pair.logit.detach().cpu(),
        "swapped_logit": swapped.logit.detach().cpu(),
        "same_evidence": same.evidence.detach().cpu(),
        "no_ref_evidence": no_ref.evidence.detach().cpu(),
        "dense_logits": eg.logits.detach().cpu(),
        "tiled_logits": tiled.logits.detach().cpu(),
        "tile_coverage": coverage.detach().cpu(),
        "query_valid_mask": qm.detach().cpu(),
        "dense_P": eg.assignment.P.detach().cpu(),
        "dense_R": eg.relational.detach().cpu(),
        "dense_u": eg.assignment.u.detach().cpu(),
        "dense_v": eg.assignment.v.detach().cpu(),
        "true_stream_P": true_stream.assignment.P.detach().cpu(),
        "true_stream_R": true_stream.relational.detach().cpu(),
        "true_stream_u": true_stream.u.detach().cpu(),
        "true_stream_v": true_stream.v.detach().cpu(),
    }
    return receipt, tensors


def _staged_query_episode_backward(
    model: DINO_RCDE_V1_2,
    query_payload: Mapping[str, Any],
    reference_payloads: Sequence[Mapping[str, Any]],
    *,
    global_query_count: int,
    device: torch.device,
) -> dict[str, Any]:
    """Backpropagate one exact query episode with at most two live unary graphs.

    The positive unary graph is retained once.  Its relational summary is
    exposed through a detached leaf while the eight negative unary graphs are
    constructed and backpropagated one at a time.  The accumulated leaf
    gradient is then applied to the retained positive graph as one VJP.  This
    is the chain rule for the unchanged mean pair loss; it changes only graph
    lifetime, not candidates, logits, loss weights, clipping or optimizer
    semantics.
    """

    if global_query_count < 1 or len(reference_payloads) < 2:
        raise ResourceError("staged query episode requires queries and at least one rival")
    rival_count = len(reference_payloads) - 1
    global_denominator = global_query_count * rival_count
    q, qm, qg = _core_inputs(query_payload, device)
    positive_tokens, positive_mask, positive_grid = _core_inputs(
        reference_payloads[0], device
    )
    positive = model.decode_candidate_true_streaming_fast(
        q, positive_tokens, qm, positive_mask, qg, positive_grid,
        query_tile_rows=EXACT_FAST_TRAIN_QUERY_TILE_ROWS,
        reference_tile_rows=EXACT_FAST_TRAIN_REFERENCE_TILE_ROWS,
    )
    _assert_exact_fast_evidence(positive)
    positive_proxy = positive.relational.detach().requires_grad_(True)
    rival_loss_tensors: list[torch.Tensor] = []
    post_rival_current_allocated_bytes: list[int] = []
    resident_elements = [int(positive.resident_consensus_logit_elements)]
    real_consensus_evaluations = [int(positive.real_consensus_evaluation_count)]
    for payload in reference_payloads[1:]:
        rival_tokens, rival_mask, rival_grid = _core_inputs(payload, device)
        rival = model.decode_candidate_true_streaming_fast(
            q, rival_tokens, qm, rival_mask, qg, rival_grid,
            query_tile_rows=EXACT_FAST_TRAIN_QUERY_TILE_ROWS,
            reference_tile_rows=EXACT_FAST_TRAIN_REFERENCE_TILE_ROWS,
        )
        _assert_exact_fast_evidence(rival)
        resident_elements.append(int(rival.resident_consensus_logit_elements))
        real_consensus_evaluations.append(int(rival.real_consensus_evaluation_count))
        pair = model.compare_relational(positive_proxy, rival.relational, qm)
        pair_loss = F.softplus(-pair.logit)
        rival_loss_tensors.append(pair_loss.detach())
        (pair_loss / global_denominator).backward()
        del pair_loss, pair, rival, rival_tokens, rival_mask
        post_rival_current_allocated_bytes.append(
            int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
        )
    positive_gradient = positive_proxy.grad
    if positive_gradient is None or not bool(torch.isfinite(positive_gradient).all()):
        raise ResourceError("staged positive-summary VJP gradient is absent or nonfinite")
    positive_gradient_l2 = float(
        torch.linalg.vector_norm(positive_gradient.detach().to(torch.float64)).cpu()
    )
    resident = _exact_fast_resident_receipt_from_counts(
        resident_elements, real_consensus_evaluations,
        query_tile_rows=EXACT_FAST_TRAIN_QUERY_TILE_ROWS,
        reference_tile_rows=EXACT_FAST_TRAIN_REFERENCE_TILE_ROWS,
    )
    positive.relational.backward(positive_gradient.detach())
    del positive, positive_proxy, positive_gradient
    rival_loss_values = torch.stack(rival_loss_tensors).detach().cpu().tolist()
    return {
        "episode_loss": sum(rival_loss_values) / rival_count,
        "rival_loss_values": rival_loss_values,
        "positive_summary_vjp_gradient_l2": positive_gradient_l2,
        "post_rival_current_allocated_bytes": post_rival_current_allocated_bytes,
        "rival_count": rival_count,
        "global_loss_denominator": global_denominator,
        "exact_fast_resident": resident,
    }


def _training_episode_benchmark(
    query_payloads: Sequence[Mapping[str, Any]],
    reference_payloads_by_query: Sequence[Sequence[Mapping[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    """Measure one real maximum four-query RCDE optimizer update."""

    if len(query_payloads) != 4 or len(reference_payloads_by_query) != 4:
        raise ResourceError("maximum training update requires four natural query episodes")
    if any(len(payloads) != 9 for payloads in reference_payloads_by_query):
        raise ResourceError("each maximum training episode requires one positive and eight negatives")
    model = DINO_RCDE_V1_2().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_bytes()
    started = time.perf_counter()
    cpu_started = time.process_time()
    model.reset_cost_counter()
    # The scientific update remains the mean of the same four query episodes
    # and their same eight natural rivals.  The staged VJP helper retains only
    # the positive and one current rival graph.  Holding all nine candidate
    # graphs caused the formal 37.57-GiB active-allocation OOM even before a
    # four-query monolithic backward could be reached.
    optimizer.zero_grad(set_to_none=True)
    episode_loss_values: list[float] = []
    positive_vjp_gradient_l2: list[float] = []
    post_rival_current_allocated_bytes: list[list[int]] = []
    resident_ledgers: list[Mapping[str, Any]] = []
    for query_payload, reference_payloads in zip(
        query_payloads, reference_payloads_by_query, strict=True
    ):
        staged = _staged_query_episode_backward(
            model, query_payload, reference_payloads,
            global_query_count=len(query_payloads), device=device,
        )
        episode_loss_values.append(float(staged["episode_loss"]))
        positive_vjp_gradient_l2.append(float(staged["positive_summary_vjp_gradient_l2"]))
        post_rival_current_allocated_bytes.append(
            list(staged["post_rival_current_allocated_bytes"])
        )
        resident_ledgers.append(dict(staged["exact_fast_resident"]))
    loss_value = sum(episode_loss_values) / len(episode_loss_values)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
    optimizer.step()
    _sync(device)
    wall = time.perf_counter() - started
    changed = sum(
        int(value.detach().ne(initial[name]).sum().item()) for name, value in model.named_parameters()
    )
    named = dict(model.named_parameters())
    delta_groups = {
        group: math.sqrt(sum(
            float((named[name].detach() - initial[name]).to(torch.float64).square().sum().cpu())
            for name in names
        ))
        for group, names in model.functional_parameter_groups().items()
    }
    resident_elements = [
        int(value)
        for ledger in resident_ledgers
        for value in ledger["resident_consensus_logit_elements_per_candidate"]
    ]
    real_evaluations = [
        int(value)
        for ledger in resident_ledgers
        for value in ledger["real_consensus_evaluations_per_candidate"]
    ]
    return {
        "query_episode_count": 4,
        "query_candidate_count": 9,
        "positive_count_per_query": 1,
        "negative_count_per_query": 8,
        "unary_cost_volume_count": int(model.cost_volume_build_count),
        "pair_comparison_count": 32,
        "optimizer_update_count": 1,
        "query_gradient_accumulation_steps": 4,
        "rival_gradient_accumulation_steps_per_query": 8,
        "pair_loss_backward_count": 32,
        "positive_summary_vjp_backward_count": 4,
        "maximum_live_query_graph_count": 1,
        "maximum_live_candidate_graph_count": 2,
        "gradient_accumulation_denominator": 32,
        "positive_summary_staged_vjp": True,
        "loss_reduction": "MEAN_4_QUERY_EPISODES_OF_MEAN_8_NATURAL_RIVALS",
        "episode_loss_values": episode_loss_values,
        "positive_summary_vjp_gradient_l2": positive_vjp_gradient_l2,
        "post_rival_current_cuda_allocated_bytes": post_rival_current_allocated_bytes,
        "scientific_episode_set_changed": False,
        "model_or_loss_changed_for_resource_fit": False,
        "execution_path": "EXACT_FAST_REAL_TILE_REUSE_ANALYTIC_NULL",
        "query_tile_rows": EXACT_FAST_TRAIN_QUERY_TILE_ROWS,
        "reference_tile_rows": EXACT_FAST_TRAIN_REFERENCE_TILE_ROWS,
        "analytic_null": True,
        "full_consensus_logits_resident": True,
        "matrix_audit_enabled": False,
        "resident_consensus_logit_elements_per_candidate": resident_elements,
        "resident_consensus_logit_elements_total": sum(resident_elements),
        "resident_consensus_logit_elements_max": max(resident_elements),
        "resident_consensus_logit_bytes_total": 4 * sum(resident_elements),
        "resident_consensus_logit_bytes_max": 4 * max(resident_elements),
        "real_consensus_evaluations_per_candidate": real_evaluations,
        "real_consensus_evaluations_total": sum(real_evaluations),
        "null_consensus_evaluations_per_candidate": [0] * len(resident_elements),
        "null_consensus_evaluations_total": 0,
        "training_diagnostic_d2h_transfer_count": len(query_payloads),
        "full_cross_cost_materialized": False,
        "full_assignment_materialized": False,
        "loss": loss_value,
        "gradient_norm_before_clip": grad_norm,
        "updated_scalar_count": changed,
        "functional_group_parameter_delta_norms": delta_groups,
        "wall_seconds": wall,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_cpu_rss_delta_bytes": max(0, _rss_bytes() - rss_before),
        "gate": int(model.cost_volume_build_count) == 36
        and all(value > 0.0 and math.isfinite(value) for value in delta_groups.values())
        and len(episode_loss_values) == 4
        and all(math.isfinite(value) for value in episode_loss_values)
        and all(value > 0.0 and math.isfinite(value) for value in positive_vjp_gradient_l2)
        and math.isfinite(loss_value) and math.isfinite(grad_norm) and grad_norm > 0.0,
    }


def _select_max_training_episode(
    data_dir: Path,
    queries: Mapping[str, Mapping[str, Any]],
    references: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    ledger = json.loads((data_dir / "fold_access_and_episode_receipt.json").read_text(encoding="utf-8"))
    eligible = []
    for episode in ledger.get("episodes", []):
        if episode.get("pair_loss_eligible") is not True or len(episode.get("selected_negatives", [])) != 8:
            continue
        rows = [int(episode["target_physical_row"]), *[int(item["physical_row"]) for item in episode["selected_negatives"]]]
        query = queries.get(str(episode["query_id"]))
        if query is None or any(row not in references for row in rows):
            continue
        shape_cost = int(query["valid_tokens"]) * sum(int(references[row]["valid_tokens"]) for row in rows)
        tie = hashlib.sha256(
            b"RCDE_RESOURCE_MAX_EPISODE_V1_2\0"
            + str(episode["heldout_fold"]).encode("ascii") + b"\0"
            + str(episode["query_id"]).encode("utf-8")
        ).hexdigest()
        raw_correct = episode.get("full_gallery_RAW_correct")
        if not isinstance(raw_correct, bool):
            raise ResourceError("postjoin episode receipt lacks full_gallery_RAW_correct")
        eligible.append((shape_cost, tie, episode, rows, query, raw_correct))
    legal_updates = []
    for heldout_fold in (1, 2, 3, 4):
        members = [item for item in eligible if int(item[2]["heldout_fold"]) == heldout_fold]
        pools = {}
        for correctness in (True, False):
            per_query: dict[str, tuple[Any, ...]] = {}
            for item in members:
                if item[5] is not correctness:
                    continue
                query_id = str(item[4]["id"])
                prior = per_query.get(query_id)
                if prior is None or item[0] > prior[0] or (item[0] == prior[0] and item[1] < prior[1]):
                    per_query[query_id] = item
            pools[correctness] = sorted(per_query.values(), key=lambda item: (-item[0], item[1]))
        if len(pools[True]) >= 2 and len(pools[False]) >= 2:
            selected_fold = pools[True][:2] + pools[False][:2]
            selection_tie = _json_sha256([
                {"query_id": item[4]["id"], "tie": item[1]} for item in selected_fold
            ])
            legal_updates.append((sum(int(item[0]) for item in selected_fold), selection_tie, heldout_fold, selected_fold))
    if not legal_updates:
        raise ResourceError("no legal same-fold 2 RAW-correct + 2 RAW-wrong maximum update")
    _, _, selected_fold, selected = sorted(legal_updates, key=lambda item: (-item[0], item[1]))[0]
    return {
        "shape_cost_sum": sum(int(item[0]) for item in selected),
        "heldout_fold": selected_fold,
        "raw_correct_count": sum(bool(item[5]) for item in selected),
        "raw_wrong_count": sum(not bool(item[5]) for item in selected),
        "same_outer_fold": len({int(item[2]["heldout_fold"]) for item in selected}) == 1,
        "selection_sha256": _json_sha256([
            {"query_id": item[4]["id"], "tie": item[1], "rows": item[3], "raw_correct": item[5]}
            for item in selected
        ]),
        "selected": [
            {"shape_cost": item[0], "tie_sha256": item[1], "episode": item[2],
             "physical_rows_target_then_611": item[3], "query": item[4],
             "full_gallery_RAW_correct": item[5]}
            for item in selected
        ],
        "eligible_episode_count": len(eligible),
        "fold_receipt_sha256": file_sha256(data_dir / "fold_access_and_episode_receipt.json"),
    }


def _pixel_intervention_benchmark(
    image_row: Mapping[str, Any], backbone: torch.nn.Module, device: torch.device
) -> dict[str, Any]:
    pixels, geometry, _ = preprocess_canonical_image_path(Path(str(image_row["source_image_path"])))
    valid_positions = torch.nonzero(geometry.valid_patch_mask.flatten(), as_tuple=False).flatten()
    if valid_positions.numel() < 16:
        raise ResourceError("worst query has fewer than 16 valid intervention seeds")
    indices = torch.linspace(0, valid_positions.numel() - 1, 16).round().to(torch.long)
    regions = [int(valid_positions[index]) for index in indices]
    batch = []
    # 1 clean + 16 fixed, result-blind one-patch connected corruptions.  P0 is
    # measuring encoder cost only; R1 later constructs K1-matched regions.
    for region in [None, *regions]:
        value = pixels.clone()
        if region is not None:
            row, column = divmod(region, geometry.grid_hw[1])
            value[..., row * 14 : (row + 1) * 14, column * 14 : (column + 1) * 14] = 0.0
        batch.append(value)
    batch_pixels = torch.cat(batch, dim=0)
    batch_valid = geometry.valid_patch_mask.unsqueeze(0).expand(17, -1, -1).contiguous()
    torch.cuda.reset_peak_memory_stats(device)
    rss_before = _rss_bytes()
    started = time.perf_counter()
    cpu_started = time.process_time()
    tokens = extract_intermediate_patch_tokens(
        backbone, batch_pixels.to(device), batch_valid.to(device)
    )
    _sync(device)
    hashes = [tensor_sha256(tokens.tokens_fp32[index].to(torch.float16)) for index in range(17)]
    return {
        "forward_count": 17,
        "corruption_count": 16,
        "region_patch_counts": [1] * 16,
        "region_seed_indices": regions,
        "distinct_output_hash_count": len(set(hashes)),
        "output_hashes": hashes,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_cpu_rss_delta_bytes": max(0, _rss_bytes() - rss_before),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "batch_shape": list(batch_pixels.shape),
        "single_batched_backbone_call": True,
        "gate": len(hashes) == 17 and len(regions) == len(set(regions)) == 16,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)]


def _cohort_members(
    cohort: Mapping[str, Any],
    queries: Mapping[str, Mapping[str, Any]],
    references: Mapping[int, Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Resolve every frozen, result-blind cohort member without substitution."""

    members: list[tuple[str, Mapping[str, Any]]] = []
    for name in ("Q_NMAX", "Q_QUERY_P50", "Q_QUERY_P95", "Q_REFSUM_MAX", "Q_PAIR_MAX"):
        item = cohort[name]
        query = queries.get(str(item["query_id"]))
        if query is None or int(query["execution_ordinal"]) != int(item["execution_ordinal"]):
            raise ResourceError(f"{name} query/execution mapping drift")
        members.append((name, query))
    for name in ("REFERENCE_P50", "REFERENCE_P95", "REFERENCE_MAX"):
        item = cohort[name]
        row = int(item.get("physical_row", item.get("reference", {}).get("physical_row", -1)))
        reference = references.get(row)
        if reference is None:
            raise ResourceError(f"{name} reference is outside frozen union")
        members.append((name, reference))
    return members


def _register_exact_fast_cohort_host(
    unique_hosts: dict[str, dict[str, Any]], *, case_name: str,
    host: Mapping[str, Any], candidates: Sequence[int],
) -> None:
    """Preserve case/C128 provenance while enforcing one case per host."""

    ordered_candidates = [int(value) for value in candidates]
    if len(ordered_candidates) != 128 or len(set(ordered_candidates)) != 128:
        raise ResourceError("cohort host query does not have unique natural C128")
    key = str(host["id"])
    if key in unique_hosts:
        prior = unique_hosts[key]
        if prior["candidate_physical_rows"] != ordered_candidates:
            raise ResourceError("same cohort host has divergent C128 memberships")
        raise ResourceError("frozen exact-fast cohort requires exactly one case per host")
    unique_hosts[key] = {
        "row": host,
        "case": str(case_name),
        "candidate_physical_rows": ordered_candidates,
        "candidate_count": len(ordered_candidates),
        "candidate_physical_rows_sha256": _json_sha256(ordered_candidates),
    }


def _cohort_audit_receipt(
    cohort: Mapping[str, Any],
    queries: Mapping[str, Mapping[str, Any]],
    references: Mapping[int, Mapping[str, Any]],
    cache_root: Path,
    backbone: torch.nn.Module,
    model_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    unique_hosts: dict[str, dict[str, Any]] = {}
    for name, source in _cohort_members(cohort, queries, references):
        item = cohort[name]
        if source["kind"] == "query":
            host = source
            candidates = [int(value) for value in item["candidate_physical_rows"]]
        else:
            host_item = item.get("host_query")
            if not isinstance(host_item, dict):
                raise ResourceError(f"{name} has no frozen natural-C128 host query")
            host = queries.get(str(host_item["query_id"]))
            if host is None or int(host["execution_ordinal"]) != int(host_item["execution_ordinal"]):
                raise ResourceError(f"{name} host query mapping drift")
            candidates = [int(value) for value in host_item["candidate_physical_rows"]]
            if int(source["physical_row"]) not in candidates:
                raise ResourceError(f"{name} is absent from its frozen natural-C128 host")
        _register_exact_fast_cohort_host(
            unique_hosts, case_name=name, host=host, candidates=candidates,
        )
        rows.append({
            "case": name, "kind": source["kind"],
            "physical_row": source.get("physical_row"),
            "host_query_id": host["id"],
            "host_execution_ordinal": int(host["execution_ordinal"]),
            "candidate_count": len(candidates),
            "candidate_physical_rows_sha256": _json_sha256(candidates),
        })

    workloads = []
    for query_id, host_item in sorted(unique_hosts.items()):
        host = host_item["row"]
        candidates = host_item["candidate_physical_rows"]
        case_name = str(host_item["case"])
        if len(candidates) != 128 or len(set(candidates)) != 128:
            raise ResourceError("cohort host query does not have natural C128")
        query_payload, _, _, _ = _materialize_cache(
            image_row=host,
            path=cache_root / f"query_{int(host['execution_ordinal']):04d}.pt",
            backbone=backbone,
            model_checkpoint_logical_sha256=model_hash,
            device=device,
        )
        reference_payloads = []
        for physical_row in candidates:
            payload, _, _, _ = _materialize_cache(
                image_row=references[physical_row],
                path=cache_root / f"reference_{physical_row:05d}.pt",
                backbone=backbone,
                model_checkpoint_logical_sha256=model_hash,
                device=device,
            )
            reference_payloads.append(payload)
        core = DINO_RCDE_V1_2().to(device).eval()
        q, qm, qg = _core_inputs(query_payload, device)
        core.reset_cost_counter()
        evidence_rows = []
        torch.cuda.reset_peak_memory_stats(device)
        rss_before = _rss_bytes()
        unary_started = time.perf_counter()
        unary_cpu_started = time.process_time()
        with torch.no_grad():
            for payload in reference_payloads:
                rt, rm, rg = _core_inputs(payload, device)
                evidence = core.decode_candidate_true_streaming_fast(
                    q, rt, qm, rm, qg, rg,
                    query_tile_rows=EXACT_FAST_C128_QUERY_TILE_ROWS,
                    reference_tile_rows=EXACT_FAST_C128_REFERENCE_TILE_ROWS,
                )
                _assert_exact_fast_evidence(evidence)
                evidence_rows.append(evidence)
        relational = torch.stack([evidence.relational for evidence in evidence_rows], dim=0)
        _sync(device)
        unary_wall = time.perf_counter() - unary_started
        unary_cpu = time.process_time() - unary_cpu_started
        pair_started = time.perf_counter()
        pair_cpu_started = time.process_time()
        with torch.no_grad():
            left_indices = torch.tensor(
                [left for left in range(128) for _ in range(left + 1, 128)],
                device=device, dtype=torch.long,
            )
            right_indices = torch.tensor(
                [right for left in range(128) for right in range(left + 1, 128)],
                device=device, dtype=torch.long,
            )
            pair_chunks = [
                core.compare_relational_batch(
                    relational,
                    left_indices[start:start + EXACT_FAST_PAIR_CHUNK],
                    right_indices[start:start + EXACT_FAST_PAIR_CHUNK],
                    qm,
                )
                for start in range(0, int(left_indices.numel()), EXACT_FAST_PAIR_CHUNK)
            ]
            pair_logits_cpu = torch.cat(pair_chunks, dim=0).detach().cpu().contiguous()
            pair_count = int(pair_logits_cpu.numel())
            digest = hashlib.sha256(pair_logits_cpu.numpy().tobytes())
        _sync(device)
        pair_wall = time.perf_counter() - pair_started
        pair_cpu = time.process_time() - pair_cpu_started
        relational_cpu = relational.detach().cpu().contiguous()
        # Reopen the stable token cache and instantiate a fresh model.  This is
        # outside the measured real path but binds every frozen host to exact
        # process-resume semantics.
        reopened_query, _ = _load_cache(
            cache_root / f"query_{int(host['execution_ordinal']):04d}.pt",
            str(host["source_image_sha256"]), model_hash,
        )
        rq, rqm, rqg = _core_inputs(reopened_query, device)
        replay_core = DINO_RCDE_V1_2().to(device).eval()
        replay_evidence_rows = []
        with torch.no_grad():
            for physical_row in candidates:
                reopened_reference, _ = _load_cache(
                    cache_root / f"reference_{physical_row:05d}.pt",
                    str(references[physical_row]["source_image_sha256"]), model_hash,
                )
                rt, rm, rg = _core_inputs(reopened_reference, device)
                replay_evidence = replay_core.decode_candidate_true_streaming_fast(
                    rq, rt, rqm, rm, rqg, rg,
                    query_tile_rows=EXACT_FAST_C128_QUERY_TILE_ROWS,
                    reference_tile_rows=EXACT_FAST_C128_REFERENCE_TILE_ROWS,
                )
                _assert_exact_fast_evidence(replay_evidence)
                replay_evidence_rows.append(replay_evidence)
            replay_relational = torch.stack(
                [evidence.relational for evidence in replay_evidence_rows], dim=0
            )
            replay_pair_chunks = [
                replay_core.compare_relational_batch(
                    replay_relational,
                    left_indices[start:start + EXACT_FAST_PAIR_CHUNK],
                    right_indices[start:start + EXACT_FAST_PAIR_CHUNK],
                    rqm,
                )
                for start in range(0, int(left_indices.numel()), EXACT_FAST_PAIR_CHUNK)
            ]
            replay_pair_cpu = torch.cat(replay_pair_chunks, dim=0).detach().cpu().contiguous()
        fresh_resume_relational_exact = tensor_sha256(relational_cpu) == tensor_sha256(
            replay_relational.detach().cpu().contiguous()
        )
        fresh_resume_pair_exact = tensor_sha256(pair_logits_cpu) == tensor_sha256(
            replay_pair_cpu
        )
        resident = _exact_fast_resident_receipt(
            evidence_rows,
            query_tile_rows=EXACT_FAST_C128_QUERY_TILE_ROWS,
            reference_tile_rows=EXACT_FAST_C128_REFERENCE_TILE_ROWS,
        )
        workloads.append({
            "case": case_name,
            "query_id": query_id,
            "execution_ordinal": int(host["execution_ordinal"]),
            "candidate_count": int(host_item["candidate_count"]),
            "candidate_physical_rows_sha256": str(
                host_item["candidate_physical_rows_sha256"]
            ),
            "unary_count": 128,
            "cost_volume_build_count": int(core.cost_volume_build_count),
            "pair_count": pair_count,
            "cost_volume_build_count_after_pairs": int(core.cost_volume_build_count),
            "pair_logit_stream_sha256": digest.hexdigest(),
            "candidate_order": "NATURAL_C128_LEFT_MAJOR",
            "pair_order": "(0,1),(0,2),...,(126,127)",
            "pair_chunk_size": EXACT_FAST_PAIR_CHUNK,
            "pair_device_to_host_transfer_count": 1,
            "fresh_resume_relational_exact": fresh_resume_relational_exact,
            "fresh_resume_pair_exact": fresh_resume_pair_exact,
            **resident,
            "unary_wall_seconds": unary_wall,
            "unary_cpu_seconds": unary_cpu,
            "pair_wall_seconds": pair_wall,
            "pair_cpu_seconds": pair_cpu,
            "wall_seconds": unary_wall + pair_wall,
            "cpu_seconds": unary_cpu + pair_cpu,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_cpu_rss_delta_bytes": max(0, _rss_bytes() - rss_before),
        })
    return {
        "required_case_count": 8,
        "executed_case_count": len(rows),
        "case_names": [row["case"] for row in rows],
        "rows": rows,
        "deduplicated_host_query_count": len(unique_hosts),
        "host_workloads": workloads,
        "gate": len(rows) == 8 and len({row["case"] for row in rows}) == 8
        and len(workloads) == len(unique_hosts)
        and {row["case"] for row in workloads} == {row["case"] for row in rows}
        and all(
            row["candidate_count"] == 128
            and len(row["candidate_physical_rows_sha256"]) == 64
            and row["unary_count"] == row["cost_volume_build_count"]
            == row["cost_volume_build_count_after_pairs"] == 128
            and row["pair_count"] == 8128
            and row["pair_chunk_size"] == EXACT_FAST_PAIR_CHUNK
            and row["pair_device_to_host_transfer_count"] == 1
            and row["analytic_null"] is True
            and row["full_consensus_logits_resident"] is True
            and row["matrix_audit_enabled"] is False
            and len(row["resident_consensus_logit_elements_per_candidate"]) == 128
            and row["fresh_resume_relational_exact"] is True
            and row["fresh_resume_pair_exact"] is True
            for row in workloads
        ),
    }


def _materialize_raw_pixel_cache_cohort(
    cohort: Mapping[str, Any],
    queries: Mapping[str, Mapping[str, Any]],
    references: Mapping[int, Mapping[str, Any]],
    cache_root: Path,
    backbone: torch.nn.Module,
    model_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    records = []
    for case_name in ("RAW_PIXEL_P50", "RAW_PIXEL_P95", "RAW_PIXEL_MAX"):
        case = cohort.get(case_name)
        if not isinstance(case, dict):
            raise ResourceError(f"missing frozen {case_name}")
        kind = str(case.get("kind"))
        if kind == "query":
            row = queries.get(str(case.get("query_id")))
            if row is None or int(row["execution_ordinal"]) != int(case["execution_ordinal"]):
                raise ResourceError(f"{case_name} query mapping drift")
            path = cache_root / f"query_{int(row['execution_ordinal']):04d}.pt"
        elif kind == "reference":
            row = references.get(int(case.get("physical_row", -1)))
            if row is None:
                raise ResourceError(f"{case_name} reference mapping drift")
            path = cache_root / f"reference_{int(row['physical_row']):05d}.pt"
        else:
            raise ResourceError(f"{case_name} kind drift")
        if int(case["raw_pixels"]) != int(row["raw_h"]) * int(row["raw_w"]):
            raise ResourceError(f"{case_name} raw-pixel count drift")
        payload, resumed, _, _ = _materialize_cache(
            image_row=row, path=path, backbone=backbone,
            model_checkpoint_logical_sha256=model_hash, device=device,
        )
        authority = _timing_authority(device, model_hash)
        measurement = json.loads(_measurement_path(path, authority).read_text(encoding="utf-8"))
        _validate_cache_measurement(
            measurement, path=path, payload=payload, authority=authority,
            source_sha256=str(row["source_image_sha256"]),
        )
        records.append({
            "case": case_name, "kind": kind, "raw_pixels": int(case["raw_pixels"]),
            "cache_artifact": path.name, "cache_logical_sha256": payload["logical_sha256"],
            "was_resumed": resumed,
            "end_to_end_cache_wall_seconds": measurement["end_to_end_cache_wall_seconds"],
            "end_to_end_cache_cpu_seconds": measurement["end_to_end_cache_cpu_seconds"],
            "end_to_end_peak_cpu_rss_delta_bytes": measurement["end_to_end_peak_cpu_rss_delta_bytes"],
            "fresh_build_logical_io_bytes": measurement["fresh_build_logical_io_bytes"],
            "source_read_operation_count": measurement["source_read_operation_count"],
            "cache_artifact_write_operation_count": measurement["cache_artifact_write_operation_count"],
            "cache_artifact_validation_read_operation_count": measurement["cache_artifact_validation_read_operation_count"],
        })
    return {
        "case_count": len(records), "records": records,
        "gate": len(records) == 3 and [row["case"] for row in records]
        == ["RAW_PIXEL_P50", "RAW_PIXEL_P95", "RAW_PIXEL_MAX"],
    }


def _cache_measurement_events(
    cache_root: Path, device: torch.device, model_checkpoint_logical_sha256: str,
) -> list[dict[str, Any]]:
    events = []
    authority = _timing_authority(device, model_checkpoint_logical_sha256)
    for path in sorted(cache_root.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "tokens_fp16" not in payload:
            continue
        tokens = payload["tokens_fp16"]
        mask = payload["valid_patch_mask"]
        if tokens.ndim != 4 or mask.ndim != 3:
            raise ResourceError(f"cache event tensor rank drift: {path}")
        total_tokens = int(mask.numel())
        measurement_path = _measurement_path(path, authority)
        if not measurement_path.is_file():
            raise ResourceError(f"cache event has no end-to-end measurement: {path}")
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        _validate_cache_measurement(
            measurement, path=path, payload=payload, authority=authority,
            source_sha256=str(payload["source_image_sha256"]),
        )
        events.append({
            "artifact": path.name,
            "total_tokens": total_tokens,
            "valid_tokens": int(mask.sum()),
            "artifact_bytes": path.stat().st_size,
            "encode_gpu_wall_seconds": float(measurement["encode_gpu_wall_seconds"]),
            "encode_cpu_seconds": float(measurement["encode_cpu_seconds"]),
            "encode_peak_cpu_rss_delta_bytes": int(measurement["encode_peak_cpu_rss_delta_bytes"]),
            "encode_peak_cuda_allocated_bytes": int(measurement["encode_peak_cuda_allocated_bytes"]),
            "encode_peak_cuda_reserved_bytes": int(measurement["encode_peak_cuda_reserved_bytes"]),
            "end_to_end_cache_wall_seconds": float(measurement["end_to_end_cache_wall_seconds"]),
            "end_to_end_cache_cpu_seconds": float(measurement["end_to_end_cache_cpu_seconds"]),
            "end_to_end_peak_cpu_rss_delta_bytes": int(measurement["end_to_end_peak_cpu_rss_delta_bytes"]),
            "source_file_bytes": int(measurement["source_file_bytes"]),
            "timing_probe_artifact_bytes": int(measurement["timing_probe_artifact_bytes"]),
            "source_read_operation_count": int(measurement["source_read_operation_count"]),
            "cache_artifact_write_operation_count": int(measurement["cache_artifact_write_operation_count"]),
            "cache_artifact_validation_read_operation_count": int(measurement["cache_artifact_validation_read_operation_count"]),
            "fresh_build_source_io_bytes": int(measurement["fresh_build_source_io_bytes"]),
            "fresh_build_artifact_io_bytes": int(measurement["fresh_build_artifact_io_bytes"]),
            "fresh_build_logical_io_bytes": int(measurement["fresh_build_logical_io_bytes"]),
            "fresh_build_event": True,
            "timing_authority": dict(authority),
            "measurement_logical_sha256": str(measurement["logical_sha256"]),
            "measurement_receipt": dict(measurement),
            "token_payload_bytes": int(tokens.numel() * tokens.element_size()),
        })
    if not events:
        raise ResourceError("no measured cache events")
    return events


def _conservative_rate_projection(
    events: Sequence[Mapping[str, Any]], numerator: str, denominator: str, total: int
) -> dict[str, float]:
    rates = [float(event[numerator]) / max(1.0, float(event[denominator])) for event in events]
    p95 = _quantile(rates, 0.95)
    worst = max(rates)
    return {
        "observed_p95_rate": p95,
        "observed_worst_rate": worst,
        "observed_p95_extrapolation": p95 * total,
        "measured_worst_extrapolation": worst * total,
        "selected_before_safety": max(p95 * total, worst * total),
    }


def _colnomic_isometry(layer: int) -> torch.Tensor:
    """Freeze the contract's 768x128 float64-QR isometric lift."""

    if layer not in (1, 2, 3, 4):
        raise ResourceError("matched ColNomic lift layer must be 1..4")
    count = 768 * 128
    values: list[float] = []
    counter = 0
    namespace = "COLNOMIC_RCDE_ISOMETRY_V1_1"
    while len(values) < count:
        digest = hashlib.sha256(
            f"{namespace}\0{layer}\0{counter}".encode("utf-8")
        ).digest()
        for offset in range(0, 32, 8):
            integer = int.from_bytes(digest[offset : offset + 8], "big")
            values.append(((integer >> 11) + 0.5) / float(1 << 53))
            if len(values) == count:
                break
        counter += 1
    uniform = torch.tensor(values, dtype=torch.float64).reshape(-1, 2)
    radius = torch.sqrt(-2.0 * torch.log(uniform[:, 0]))
    angle = 2.0 * math.pi * uniform[:, 1]
    gaussian = torch.stack(
        (radius * torch.cos(angle), radius * torch.sin(angle)), dim=1
    ).flatten()[:count].reshape(768, 128)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    return (q * signs.unsqueeze(0)).to(torch.float32).contiguous()


def _matched_colnomic_lift_benchmark(
    *,
    protocol: Mapping[str, Any],
    matched: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Measure the exact four-layer isometric-lift preprocessing path.

    The full population is projected from measured per-token events.  This
    benchmark deliberately uses the frozen maximum-token natural query and
    reference, never a synthetic tensor or a target-selected row.
    """

    if (
        matched.get("eligible") is not True
        or int(matched.get("query_dimension", -1)) != 128
        or int(matched.get("reference_dimension", -1)) != 128
        or int(matched.get("query_count", -1)) != 600
        or int(matched.get("reference_physical_row_count", -1)) != 5413
    ):
        raise ResourceError("matched ColNomic resource ledger contract drift")
    bindings = protocol["prejoin_bindings"]
    gallery_path = _resolve(bindings["gallery_token_cache"])
    manifest_path = _resolve(bindings["inherited_v1_1_redacted_manifest"])
    source_started = time.perf_counter()
    source_cpu_started = time.process_time()
    source_rss_before = _rss_bytes()
    if (
        gallery_path.stat().st_size != int(matched["gallery_token_cache_file_bytes"])
        or file_sha256(gallery_path) != str(matched["gallery_token_cache_sha256"])
        or file_sha256(manifest_path) != str(matched["redacted_query_manifest_sha256"])
    ):
        raise ResourceError("matched ColNomic source-cache physical binding drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest.get("rows", []))
    if len(rows) != 600:
        raise ResourceError("matched ColNomic query manifest population drift")
    # Choose the actual maximum image-token payload, with only prejoin hashes
    # and query IDs as deterministic ties.  Grid product is deliberately not
    # assumed to equal the token count.
    query_payloads = []
    for row in rows:
        payload = torch.load(
            Path(str(row["redacted_shard"])), map_location="cpu", weights_only=False
        )
        tokens = payload.get("image_tokens")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[1] != 128:
            raise ResourceError("matched ColNomic query shard token schema drift")
        query_payloads.append((int(tokens.shape[0]), str(row["image_token_sha256"]), str(row["query_id"]), row, payload))
    _, _, _, query_row, query_payload = sorted(
        query_payloads, key=lambda item: (-item[0], item[1], item[2])
    )[0]
    query_tokens = query_payload.get("image_tokens")
    query_shard_path = Path(str(query_row["redacted_shard"]))
    query_shard_sha256 = file_sha256(query_shard_path)
    query_tensor_legacy_sha256 = _inherited_colnomic_tensor_sha256(query_tokens)
    if (
        not isinstance(query_tokens, torch.Tensor)
        or tuple(query_tokens.shape) != (int(matched["query_image_token_max"]), 128)
        or query_tokens.dtype != torch.float16
        or query_shard_sha256 != str(query_row["redacted_shard_sha256"])
        or query_tensor_legacy_sha256 != str(query_row["image_token_sha256"])
    ):
        raise ResourceError("matched ColNomic maximum-query token drift")
    gallery = torch.load(gallery_path, map_location="cpu", weights_only=False, mmap=True)
    gallery_rows = gallery.get("passage_emb", [])
    if len(gallery_rows) != 5413:
        raise ResourceError("matched ColNomic gallery population drift")
    reference_row = sorted(
        range(len(gallery_rows)),
        key=lambda index: (-int(gallery_rows[index].shape[0]), index),
    )[0]
    reference_tokens = gallery_rows[reference_row]
    if (
        tuple(reference_tokens.shape) != (int(matched["reference_token_max"]), 128)
        or reference_tokens.dtype != torch.float16
    ):
        raise ResourceError("matched ColNomic maximum-reference token drift")
    # Force the selected mmap row to be read inside the source-cache event.
    reference_tokens = reference_tokens.contiguous()
    source_event = {
        "wall_seconds": time.perf_counter() - source_started,
        "cpu_seconds": time.process_time() - source_cpu_started,
        "peak_cpu_rss_delta_bytes": max(0, _rss_bytes() - source_rss_before),
        "measured_gallery_hash_read_bytes": int(matched["gallery_token_cache_file_bytes"]),
        "measured_max_query_shard_read_bytes": query_shard_path.stat().st_size,
        "measured_query_shard_population_read_bytes": int(matched["redacted_query_shard_total_file_bytes"]),
        "projected_population_source_read_bytes": int(matched["gallery_token_cache_file_bytes"])
        + int(matched["redacted_query_shard_total_file_bytes"]),
        "gallery_file_sha256_verified": True,
        "query_manifest_sha256_verified": True,
        "selected_query_id": str(query_row["query_id"]),
        "selected_query_ordinal": int(query_row["query_ordinal"]),
        "selected_query_token_count": int(query_tokens.shape[0]),
        "selected_query_tensor_shape": list(query_tokens.shape),
        "selected_query_tensor_dtype": str(query_tokens.dtype),
        "selected_query_shard_expected_sha256": str(query_row["redacted_shard_sha256"]),
        "selected_query_shard_observed_sha256": query_shard_sha256,
        "selected_query_tensor_expected_sha256": str(query_row["image_token_sha256"]),
        "selected_query_tensor_observed_sha256": query_tensor_legacy_sha256,
        "selected_query_tensor_hash_contract": "V1_1_DTYPE_STRING_JSON_LIST_SHAPE_CONTIGUOUS_BYTES",
    }

    generation_started = time.perf_counter()
    generation_cpu_started = time.process_time()
    isometries = [_colnomic_isometry(layer) for layer in (1, 2, 3, 4)]
    generation_wall = time.perf_counter() - generation_started
    generation_cpu = time.process_time() - generation_cpu_started
    isometry_hashes = [tensor_sha256(value) for value in isometries]
    events = []
    for kind, tokens in (("query", query_tokens), ("reference", reference_tokens)):
        torch.cuda.reset_peak_memory_stats(device)
        rss_before = _rss_bytes()
        started = time.perf_counter()
        cpu_started = time.process_time()
        source = tokens.to(device=device, dtype=torch.float32)
        matrices = [value.to(device=device) for value in isometries]
        lifted = torch.stack([source @ matrix.transpose(0, 1) for matrix in matrices])
        cached = lifted.to(torch.float16)
        _sync(device)
        events.append({
            "kind": kind,
            "source_token_count": int(tokens.shape[0]),
            "source_dimension": 128,
            "layer_count": 4,
            "lifted_dimension": 768,
            "lifted_fp16_bytes": int(cached.numel() * cached.element_size()),
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_cpu_rss_delta_bytes": max(0, _rss_bytes() - rss_before),
            "lifted_fp16_sha256": tensor_sha256(cached.detach().cpu()),
        })
        del source, matrices, lifted, cached
    return {
        "algorithm": "SHA256_COUNTER_GAUSSIAN_FLOAT64_QR_POSITIVE_R_DIAGONAL_CAST_FP32",
        "namespace": "COLNOMIC_RCDE_ISOMETRY_V1_1",
        "matrix_shape": [768, 128],
        "matrix_sha256": isometry_hashes,
        "orthonormality_max_abs_error_float64_replay": max(
            float((value.to(torch.float64).T @ value.to(torch.float64) - torch.eye(128, dtype=torch.float64)).abs().max())
            for value in isometries
        ),
        "generation_wall_seconds": generation_wall,
        "generation_cpu_seconds": generation_cpu,
        "events": events,
        "source_cache_event": source_event,
        "gate": len(set(isometry_hashes)) == 4
        and source_event["wall_seconds"] > 0.0
        and source_event["cpu_seconds"] > 0.0
        and source_event["projected_population_source_read_bytes"] > 0
        and {row["kind"]: row["source_token_count"] for row in events}
        == {
            "query": int(matched["query_image_token_max"]),
            "reference": int(matched["reference_token_max"]),
        }
        and max(
            float((value.to(torch.float64).T @ value.to(torch.float64) - torch.eye(128, dtype=torch.float64)).abs().max())
            for value in isometries
        ) <= 2.0e-6
        and all(
            row["source_token_count"] > 0 and row["lifted_fp16_bytes"]
            == row["source_token_count"] * 4 * 768 * 2
            for row in events
        ),
    }


def _build_projection(
    *,
    shape: Mapping[str, Any],
    cache_events: Sequence[Mapping[str, Any]],
    cohort_audit: Mapping[str, Any],
    training: Mapping[str, Any],
    pixel: Mapping[str, Any],
    main_measurement: Mapping[str, Any],
    matched_colnomic: Mapping[str, Any],
    matched_lift: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure, independently replayable resource projection."""

    safety = 1.25
    rows = list(shape["rows"])
    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    total_valid_tokens = sum(int(row["valid_tokens"]) for row in rows)
    image_count = len(rows)
    raw_payload_bytes = total_tokens * 4 * 768 * 2
    analytic_cache_bytes = raw_payload_bytes + total_tokens + image_count * 8192
    # A requested GPU is occupied for the whole cache-build event, including
    # source validation, preprocessing, atomic write and reopen validation.
    cache_gpu_rate = _conservative_rate_projection(
        cache_events, "end_to_end_cache_wall_seconds", "total_tokens", total_tokens
    )
    cache_cpu_rate = _conservative_rate_projection(
        cache_events, "end_to_end_cache_cpu_seconds", "total_tokens", total_tokens
    )
    cache_wall_rate = _conservative_rate_projection(
        cache_events, "end_to_end_cache_wall_seconds", "total_tokens", total_tokens
    )
    cache_artifact_byte_rate = _conservative_rate_projection(
        cache_events, "artifact_bytes", "total_tokens", total_tokens
    )
    cache_io_rate = _conservative_rate_projection(
        cache_events, "fresh_build_logical_io_bytes", "total_tokens", total_tokens
    )
    cache_bytes_before_safety = max(
        float(analytic_cache_bytes), cache_artifact_byte_rate["selected_before_safety"]
    )

    workloads = list(cohort_audit["host_workloads"])
    if not workloads:
        raise ResourceError("no full-C128 measured workload event")
    host_gpu = [float(row["wall_seconds"]) for row in workloads]
    host_cpu = [float(row["cpu_seconds"]) for row in workloads]
    host_gpu_base = max(_quantile(host_gpu, 0.95), max(host_gpu))
    host_cpu_base = max(_quantile(host_cpu, 0.95), max(host_cpu))
    unary_gpu_values = [float(row["unary_wall_seconds"]) for row in workloads]
    unary_cpu_values = [float(row["unary_cpu_seconds"]) for row in workloads]
    pair_gpu_values = [float(row["pair_wall_seconds"]) for row in workloads]
    pair_cpu_values = [float(row["pair_cpu_seconds"]) for row in workloads]
    unary_gpu_base = max(_quantile(unary_gpu_values, 0.95), max(unary_gpu_values)) / 128.0
    unary_cpu_base = max(_quantile(unary_cpu_values, 0.95), max(unary_cpu_values)) / 128.0
    pair_gpu_base = max(_quantile(pair_gpu_values, 0.95), max(pair_gpu_values)) / 8128.0
    pair_cpu_base = max(_quantile(pair_cpu_values, 0.95), max(pair_cpu_values)) / 8128.0
    max_fold_queries = 151
    training_gpu_per_update = float(training["wall_seconds"])
    training_cpu_per_update = float(training["cpu_seconds"])
    single_fold_gpu_seconds = 2048 * training_gpu_per_update + max_fold_queries * host_gpu_base
    single_fold_cpu_seconds = 2048 * training_cpu_per_update + max_fold_queries * host_cpu_base

    max_artifact_bytes = max(int(row["artifact_bytes"]) for row in cache_events)
    train_unary_count = 2048 * 4 * 9
    heldout_unary_count = max_fold_queries * 128
    heldout_pair_count = max_fold_queries * 8128
    # Pairs consume cached summaries, not DINO/cross-cost.  Count a 16-float
    # summary for both endpoints as their conservative logical I/O.
    per_pair_summary_bytes = 2 * max(int(row["total_tokens"]) for row in rows) * 16 * 4
    fold_io_bytes = (
        (train_unary_count + heldout_unary_count) * 2 * max_artifact_bytes
        + heldout_pair_count * per_pair_summary_bytes
    )
    fold_summary_scratch = (
        max_fold_queries * 128 * max(int(row["total_tokens"]) for row in rows) * 17 * 4
    )
    inference_gpu_seconds = max_fold_queries * host_gpu_base
    inference_cpu_seconds = max_fold_queries * host_cpu_base
    inference_io_bytes = heldout_unary_count * 2 * max_artifact_bytes + heldout_pair_count * per_pair_summary_bytes

    if matched_colnomic.get("eligible") is not True or matched_lift.get("gate") is not True:
        raise ResourceError("matched ColNomic lift resource qualification is not closed")
    matched_total_tokens = int(matched_colnomic["query_image_token_total"]) + int(
        matched_colnomic["reference_token_total"]
    )
    matched_lift_bytes = matched_total_tokens * 4 * 768 * 2
    lift_events = list(matched_lift["events"])
    if {row.get("kind") for row in lift_events} != {"query", "reference"}:
        raise ResourceError("matched ColNomic lift event population drift")
    lift_gpu_rate = max(float(row["wall_seconds"]) / int(row["source_token_count"]) for row in lift_events)
    lift_cpu_rate = max(float(row["cpu_seconds"]) / int(row["source_token_count"]) for row in lift_events)
    # QR generation and source-cache traversal are CPU work.  Do not inflate
    # GPU-hours with CPU walltime, but do include both in the CPU projection.
    matched_lift_gpu_seconds = lift_gpu_rate * matched_total_tokens
    matched_lift_cpu_seconds = (
        2.0 * float(matched_lift["source_cache_event"]["cpu_seconds"])
        + float(matched_lift["generation_cpu_seconds"])
        + lift_cpu_rate * matched_total_tokens
    )
    matched_gallery_bytes = int(matched_colnomic["gallery_token_cache_file_bytes"])
    matched_query_bytes = int(matched_colnomic["redacted_query_shard_total_file_bytes"])
    # Physical bindings hash every source byte, then the builder reads the
    # same cache population.  The lifted artifact is written once and reopened
    # once for validation/reuse.  Small JSON/spatial ledgers are separately
    # bound but omitted only as a documented negligible lower-order term.
    matched_source_bytes = 2 * (matched_gallery_bytes + matched_query_bytes)
    matched_lift_logical_io_bytes = matched_source_bytes + 2 * matched_lift_bytes

    arm_names = (
        "BAG", "CONTEXT", "REFERENCE_ONLY", "UNARY_FOREGROUND",
        "PAIR_DIRECTION_PERMUTED", "MATCHED_COLNOMIC",
    )
    stages: dict[str, dict[str, Any]] = {}
    for name in arm_names:
        stages[name] = {
            "fold_count": 4,
            "maximum_queries_per_fold": max_fold_queries,
            "training_updates_per_fold": 2048,
            "heldout_unary_per_fold": heldout_unary_count,
            "heldout_pair_per_fold": heldout_pair_count,
            "gpu_hours_per_fold_with_25pct_safety": safety * single_fold_gpu_seconds / 3600.0,
            "cpu_hours_per_fold_with_25pct_safety": safety * single_fold_cpu_seconds / 3600.0,
            "gpu_hours_four_fold_with_25pct_safety": safety * 4 * single_fold_gpu_seconds / 3600.0,
            "cpu_hours_four_fold_with_25pct_safety": safety * 4 * single_fold_cpu_seconds / 3600.0,
            "io_gib_four_fold_with_25pct_safety": safety * 4 * fold_io_bytes / (1024.0 ** 3),
            "incremental_scratch_gib_four_fold_with_25pct_safety": safety * 4 * fold_summary_scratch / (1024.0 ** 3),
        }
    matched_stage = stages["MATCHED_COLNOMIC"]
    matched_stage.update({
        "source_token_population": matched_total_tokens,
        "four_layer_isometric_lift_cache_bytes": matched_lift_bytes,
        "lift_materialized_once_and_reused_across_folds": True,
        "lift_gpu_hours_with_25pct_safety": safety * matched_lift_gpu_seconds / 3600.0,
        "lift_cpu_hours_with_25pct_safety": safety * matched_lift_cpu_seconds / 3600.0,
        "lift_source_write_validation_io_gib_with_25pct_safety": safety * matched_lift_logical_io_bytes / (1024.0 ** 3),
        "source_hash_read_operation_count": 1,
        "source_build_read_operation_count": 1,
        "lift_artifact_write_operation_count": 1,
        "lift_artifact_validation_read_operation_count": 1,
        "small_manifest_and_spatial_metadata_io_excluded_as_lower_order": True,
        "gpu_hours_four_fold_with_25pct_safety": matched_stage["gpu_hours_four_fold_with_25pct_safety"]
        + safety * matched_lift_gpu_seconds / 3600.0,
        "cpu_hours_four_fold_with_25pct_safety": matched_stage["cpu_hours_four_fold_with_25pct_safety"]
        + safety * matched_lift_cpu_seconds / 3600.0,
        "io_gib_four_fold_with_25pct_safety": matched_stage["io_gib_four_fold_with_25pct_safety"]
        + safety * matched_lift_logical_io_bytes / (1024.0 ** 3),
        "incremental_scratch_gib_four_fold_with_25pct_safety": matched_stage["incremental_scratch_gib_four_fold_with_25pct_safety"]
        + safety * matched_lift_bytes / (1024.0 ** 3),
    })

    e0_io_bytes = 256 * 4 * 9 * 2 * max_artifact_bytes
    stages["E0_256"] = {
        "training_updates": 256,
        "measured_update_query_count": 4,
        "measured_update_unary_count": 36,
        "gpu_hours_with_25pct_safety": safety * 256 * training_gpu_per_update / 3600.0,
        "cpu_hours_with_25pct_safety": safety * 256 * training_cpu_per_update / 3600.0,
        "io_gib_with_25pct_safety": safety * e0_io_bytes / (1024.0 ** 3),
        "incremental_scratch_gib_with_25pct_safety": safety * fold_summary_scratch / (1024.0 ** 3),
        "cost_basis": "MEASURED_REAL_FOUR_QUERY_36_UNARY_BACKWARD_OPTIMIZER_UPDATE_DOMINATES_E0_MINIBATCH",
    }

    full_control_gpu = safety * 4 * max_fold_queries * host_gpu_base / 3600.0
    full_control_cpu = safety * 4 * max_fold_queries * host_cpu_base / 3600.0
    full_control_io = safety * 4 * inference_io_bytes / (1024.0 ** 3)
    for name in ("R1_C_BIND", "R1_P_QUERY", "R1_P_REFERENCE"):
        stages[name] = {
            "fold_count": 4, "maximum_queries_per_fold": max_fold_queries,
            "unary_redecode_required": True, "pair_matrix_recomputed": True,
            "gpu_hours_four_fold_with_25pct_safety": full_control_gpu,
            "cpu_hours_four_fold_with_25pct_safety": full_control_cpu,
            "io_gib_four_fold_with_25pct_safety": full_control_io,
            "incremental_scratch_gib_four_fold_with_25pct_safety": safety * 4 * fold_summary_scratch / (1024.0 ** 3),
            "cost_basis": "MEASURED_FULL_C128_TRUE_STREAMING_REDECODE",
        }
    donor_gpu_seconds = max_fold_queries * 8128 * pair_gpu_base
    donor_cpu_seconds = max_fold_queries * 8128 * pair_cpu_base
    donor_io_bytes = heldout_pair_count * per_pair_summary_bytes
    stages["R1_RELIABILITY_DONOR"] = {
        "fold_count": 4, "maximum_queries_per_fold": max_fold_queries,
        "unary_redecode_required": False, "reuses_frozen_real_signed_support": True,
        "pair_matrix_recomputed_from_donor_reliability": True,
        "gpu_hours_four_fold_with_25pct_safety": safety * 4 * donor_gpu_seconds / 3600.0,
        "cpu_hours_four_fold_with_25pct_safety": safety * 4 * donor_cpu_seconds / 3600.0,
        "io_gib_four_fold_with_25pct_safety": safety * 4 * donor_io_bytes / (1024.0 ** 3),
        "incremental_scratch_gib_four_fold_with_25pct_safety": safety * 4 * fold_summary_scratch / (1024.0 ** 3),
        "cost_basis": "MEASURED_PAIR_ONLY_PATH_WITH_FROZEN_UNARY_SUMMARY_REUSE",
    }
    # RAW_DINO is a cheaper fixed cosine/mutual-match readout.  Charge one
    # measured full RCDE C128 replay as a strict, same-input dominating bound.
    stages["R1_RAW_DINO"] = {
        "fold_count": 4, "maximum_queries_per_fold": max_fold_queries,
        "learned_parameters": 0, "rcde_redecode_required": False,
        "gpu_hours_four_fold_with_25pct_safety": full_control_gpu,
        "cpu_hours_four_fold_with_25pct_safety": full_control_cpu,
        "io_gib_four_fold_with_25pct_safety": full_control_io,
        "incremental_scratch_gib_four_fold_with_25pct_safety": safety * 4 * fold_summary_scratch / (1024.0 ** 3),
        "cost_basis": "MEASURED_FULL_C128_RCDE_PATH_STRICTLY_DOMINATES_FIXED_RAW_DINO_READOUT",
    }
    for name in ("T1", "N1", "A1"):
        stages[name] = {
            "fold_count": 4,
            "gpu_hours_four_fold_with_25pct_safety": safety * 4 * inference_gpu_seconds / 3600.0,
            "cpu_hours_four_fold_with_25pct_safety": safety * 4 * inference_cpu_seconds / 3600.0,
            "io_gib_four_fold_with_25pct_safety": safety * 4 * inference_io_bytes / (1024.0 ** 3),
            "incremental_scratch_gib_four_fold_with_25pct_safety": safety * 4 * fold_summary_scratch / (1024.0 ** 3),
        }
    # N1 fits A/B/C/D plus two direction-permuted controls in every outer
    # fold, each with 512 tiny full-batch CPU updates.  A measured 65,125-
    # parameter natural RCDE update strictly dominates a linear <=3-feature
    # calibrator update, so charge that measured CPU and occupied CPU-node wall
    # as the conservative bound. The execution contract forbids requesting or
    # using a GPU for these fits; therefore they do not enter the GPU gate.
    n1_calibrator_count = 6
    n1_fit_count_per_calibrator_per_outer_fold = 4  # 3 inner + 1 outer-train final
    n1_calibrator_updates_per_fold = (
        n1_calibrator_count * n1_fit_count_per_calibrator_per_outer_fold * 512
    )
    n1_calibrator_cpu_seconds_four_fold = 4 * n1_calibrator_updates_per_fold * training_cpu_per_update
    n1_calibrator_node_wall_seconds_four_fold = 4 * n1_calibrator_updates_per_fold * training_gpu_per_update
    stages["N1"].update({
        "calibrator_count_per_fold": n1_calibrator_count,
        "fit_count_per_calibrator_per_outer_fold": n1_fit_count_per_calibrator_per_outer_fold,
        "fit_composition": "THREE_INNER_CROSSFIT_PLUS_ONE_OUTER_TRAIN_FINAL",
        "updates_per_calibrator": 512,
        "inner_crossfit_required": True,
        "calibrator_cpu_hours_four_fold_with_25pct_safety": safety * n1_calibrator_cpu_seconds_four_fold / 3600.0,
        "calibrator_occupied_node_hours_four_fold_with_25pct_safety": safety * n1_calibrator_node_wall_seconds_four_fold / 3600.0,
        "calibrator_execution_device": "CPU_ONLY_GPU_REQUEST_FORBIDDEN",
        "cpu_hours_four_fold_with_25pct_safety": stages["N1"]["cpu_hours_four_fold_with_25pct_safety"]
        + safety * n1_calibrator_cpu_seconds_four_fold / 3600.0,
        "calibrator_cost_basis": "MEASURED_FULL_RCDE_UPDATE_STRICTLY_DOMINATES_TINY_LINEAR_FULL_BATCH_CALIBRATOR_UPDATE",
    })
    n1_inner_arm_count = 2
    n1_outer_fold_count = 4
    n1_inner_fit_count = 3
    n1_total_rcde_fits = n1_inner_arm_count * n1_outer_fold_count * n1_inner_fit_count
    stages["N1_INNER_RCDE_CROSSFIT"] = {
        "arms": ["SELECTED_DINO", "MATCHED_COLNOMIC"],
        "outer_fold_count": n1_outer_fold_count,
        "inner_fit_count_per_arm_per_outer_fold": n1_inner_fit_count,
        "total_2048_update_rcde_fits": n1_total_rcde_fits,
        "training_updates_per_fit": 2048,
        "inner_validation_c128_queries_per_fit_conservative": max_fold_queries,
        "gpu_hours_with_25pct_safety": safety * n1_total_rcde_fits * single_fold_gpu_seconds / 3600.0,
        "cpu_hours_with_25pct_safety": safety * n1_total_rcde_fits * single_fold_cpu_seconds / 3600.0,
        "io_gib_with_25pct_safety": safety * n1_total_rcde_fits * fold_io_bytes / (1024.0 ** 3),
        "incremental_scratch_gib_with_25pct_safety": safety * fold_summary_scratch / (1024.0 ** 3),
        "matched_lift_reused_from_frozen_global_artifact": True,
        "cost_basis": "TWENTY_FOUR_INNER_RCDE_FIT_AND_C128_REPLAY_PATHS_NO_OUTER_IDENTITY_LEAKAGE",
    }
    pixel_rcde_gpu_seconds = 600 * (34 * unary_gpu_base + 17 * pair_gpu_base)
    pixel_rcde_cpu_seconds = 600 * (34 * unary_cpu_base + 17 * pair_cpu_base)
    pixel_rcde_io_bytes = 600 * 34 * 2 * max_artifact_bytes
    stages["PIXEL_17X600"] = {
        "batch_size": 17,
        "query_count": 600,
        "unary_rcde_per_query": 34, "pair_comparisons_per_query": 17,
        "gpu_hours_with_25pct_safety": safety * (float(pixel["wall_seconds"]) * 600 + pixel_rcde_gpu_seconds) / 3600.0,
        "cpu_hours_with_25pct_safety": safety * (float(pixel["cpu_seconds"]) * 600 + pixel_rcde_cpu_seconds) / 3600.0,
        "io_gib_with_25pct_safety": safety * pixel_rcde_io_bytes / (1024.0 ** 3),
        "incremental_scratch_gib_with_25pct_safety": 0.0,
        "cost_basis": "MEASURED_BATCH17_DINO_REENCODE_PLUS_MEASURED_34_UNARY_17_PAIR_RCDE_PER_QUERY",
    }
    cache_gpu_hours = safety * cache_gpu_rate["selected_before_safety"] / 3600.0
    cache_cpu_hours = safety * cache_cpu_rate["selected_before_safety"] / 3600.0
    cache_gib = safety * cache_bytes_before_safety / (1024.0 ** 3)
    arms_gpu = sum(stages[name]["gpu_hours_four_fold_with_25pct_safety"] for name in arm_names)
    arms_cpu = sum(stages[name]["cpu_hours_four_fold_with_25pct_safety"] for name in arm_names)
    control_names = (
        "R1_C_BIND", "R1_P_QUERY", "R1_P_REFERENCE",
        "R1_RELIABILITY_DONOR", "R1_RAW_DINO",
    )
    controls_gpu = sum(stages[name]["gpu_hours_four_fold_with_25pct_safety"] for name in control_names)
    controls_cpu = sum(stages[name]["cpu_hours_four_fold_with_25pct_safety"] for name in control_names)
    p0_n1_gpu = (
        cache_gpu_hours + stages["E0_256"]["gpu_hours_with_25pct_safety"]
        + arms_gpu + controls_gpu + stages["N1_INNER_RCDE_CROSSFIT"]["gpu_hours_with_25pct_safety"]
        + stages["PIXEL_17X600"]["gpu_hours_with_25pct_safety"]
        + stages["T1"]["gpu_hours_four_fold_with_25pct_safety"]
        + stages["N1"]["gpu_hours_four_fold_with_25pct_safety"]
    )
    p0_n1_cpu = (
        cache_cpu_hours + stages["E0_256"]["cpu_hours_with_25pct_safety"]
        + arms_cpu + controls_cpu + stages["N1_INNER_RCDE_CROSSFIT"]["cpu_hours_with_25pct_safety"]
        + stages["PIXEL_17X600"]["cpu_hours_with_25pct_safety"]
        + stages["T1"]["cpu_hours_four_fold_with_25pct_safety"]
        + stages["N1"]["cpu_hours_four_fold_with_25pct_safety"]
    )
    peak_candidates = [
        int(training["peak_cuda_reserved_bytes"]), int(pixel["peak_cuda_reserved_bytes"]),
        int(main_measurement["peak_cuda_reserved_bytes"]),
        *[int(row["peak_cuda_reserved_bytes"]) for row in workloads],
        *[int(row["encode_peak_cuda_reserved_bytes"]) for row in cache_events],
        *[int(row["peak_cuda_reserved_bytes"]) for row in lift_events],
    ]
    peak_cpu_candidates = [
        int(training["peak_cpu_rss_delta_bytes"]), int(pixel["peak_cpu_rss_delta_bytes"]),
        int(main_measurement["peak_cpu_rss_delta_bytes"]),
        *[int(row["peak_cpu_rss_delta_bytes"]) for row in workloads],
        *[int(row["end_to_end_peak_cpu_rss_delta_bytes"]) for row in cache_events],
        int(matched_lift["source_cache_event"]["peak_cpu_rss_delta_bytes"]),
        *[int(row["peak_cpu_rss_delta_bytes"]) for row in lift_events],
    ]
    return {
        "formula_version": "RCDE_RESOURCE_PROJECTION_EVENT_MAX_P95_WORST_V1_2",
        "safety_factor": safety,
        "population": {
            "image_count": image_count, "total_tokens": total_tokens,
            "total_valid_tokens": total_valid_tokens,
            "raw_fp16_payload_bytes_from_total_tokens": raw_payload_bytes,
            "analytic_cache_bytes_before_safety": analytic_cache_bytes,
        },
        "measurement_events": {
            "cache": list(cache_events), "full_c128": workloads,
            "maximum_training_episode": dict(training),
            "pixel_intervention_batch17": dict(pixel),
            "main_c128": dict(main_measurement),
            "matched_colnomic_lift": dict(matched_lift),
            "matched_colnomic_data_ledger": dict(matched_colnomic),
        },
        "conservative_bases": {
            "cache_gpu": cache_gpu_rate, "cache_cpu": cache_cpu_rate,
            "cache_end_to_end_wall": cache_wall_rate,
            "cache_artifact_bytes": cache_artifact_byte_rate,
            "cache_build_io_bytes": cache_io_rate,
            "full_c128_gpu_seconds": {
                "observed_values": host_gpu, "observed_p95": _quantile(host_gpu, 0.95),
                "measured_worst": max(host_gpu), "selected": host_gpu_base,
            },
            "full_c128_cpu_seconds": {
                "observed_values": host_cpu, "observed_p95": _quantile(host_cpu, 0.95),
                "measured_worst": max(host_cpu), "selected": host_cpu_base,
            },
            "full_c128_unary_gpu_seconds_per_candidate": unary_gpu_base,
            "full_c128_unary_cpu_seconds_per_candidate": unary_cpu_base,
            "full_c128_pair_gpu_seconds_per_pair": pair_gpu_base,
            "full_c128_pair_cpu_seconds_per_pair": pair_cpu_base,
            "matched_colnomic_lift_gpu_seconds_per_token_worst": lift_gpu_rate,
            "matched_colnomic_lift_cpu_seconds_per_token_worst": lift_cpu_rate,
        },
        "cache_gib_with_25pct_safety": cache_gib,
        "cache_build_gpu_hours_with_25pct_safety": cache_gpu_hours,
        "cache_build_cpu_hours_with_25pct_safety": cache_cpu_hours,
        "single_fold_single_arm_gpu_hours_with_25pct_safety": safety * single_fold_gpu_seconds / 3600.0,
        "single_fold_single_arm_cpu_hours_with_25pct_safety": safety * single_fold_cpu_seconds / 3600.0,
        "peak_gpu_gib": max(peak_candidates) / (1024.0 ** 3),
        "peak_gpu_stage_reserved_bytes": peak_candidates,
        "peak_cpu_rss_gib": max(peak_cpu_candidates) / (1024.0 ** 3),
        "peak_cpu_stage_rss_delta_bytes": peak_cpu_candidates,
        "stages": stages,
        "p0_through_n1_gpu_hours_with_25pct_safety": p0_n1_gpu,
        "p0_through_n1_cpu_hours_with_25pct_safety": p0_n1_cpu,
        "p0_through_n1_excludes_a1": True,
        "a1_listed_but_not_in_gate": True,
        "p0_through_n1_io_gib_with_25pct_safety": (
            safety * cache_io_rate["selected_before_safety"] / (1024.0 ** 3)
            + stages["E0_256"]["io_gib_with_25pct_safety"]
            + sum(stages[name]["io_gib_four_fold_with_25pct_safety"] for name in arm_names)
            + sum(stages[name]["io_gib_four_fold_with_25pct_safety"] for name in control_names)
            + stages["N1_INNER_RCDE_CROSSFIT"]["io_gib_with_25pct_safety"]
            + stages["PIXEL_17X600"]["io_gib_with_25pct_safety"]
            + stages["T1"]["io_gib_four_fold_with_25pct_safety"]
            + stages["N1"]["io_gib_four_fold_with_25pct_safety"]
        ),
        "maximum_concurrent_scratch_gib_with_25pct_safety": cache_gib + max(
            max(
                stages[name].get("incremental_scratch_gib_four_fold_with_25pct_safety", 0.0),
                stages[name].get("incremental_scratch_gib_with_25pct_safety", 0.0),
            ) for name in stages
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cross-node-validation", type=Path)
    parser.add_argument(
        "--exact-fast-qualification", type=Path, required=True,
        help="postjoin-only independent exact-fast engineering authority",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any((output / name).exists() for name in (
        "result.json", "rcde_resource_benchmark_v1_2.json", "math_audit.pt"
    )):
        raise ResourceError(f"job-specific resource output already contains immutable results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_total = time.perf_counter()
    try:
        protocol, shape, matched_colnomic = _validate_inputs(
            args.protocol.resolve(), args.data_result_dir.resolve()
        )
        # Resource execution is postjoin.  The qualification authority embeds
        # the fixed natural training/C128 engineering fixtures and must never
        # be loaded by the preceding prejoin data path.
        exact_fast_qualification = _load_exact_fast_qualification(
            args.exact_fast_qualification
        )
        if not torch.cuda.is_available():
            raise ResourceError("formal resource benchmark requires CUDA")
        device = torch.device("cuda")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
        checkpoint = audit_local_checkpoint()
        backbone_bundle = load_frozen_backbone(
            device=device, cache_root=output / "hf_offline_cache"
        )
        backbone = backbone_bundle.model
        # Cache binding must be stable across jobs/nodes; a runtime receipt
        # includes an ephemeral cache path, so bind only the frozen checkpoint
        # provenance and semantic extraction contract.
        model_receipt_sha = canonical_sha256({
            "model_id": checkpoint["model_id"],
            "revision": checkpoint["revision"],
            "file_sha256": checkpoint["file_sha256"],
            "layers_1_based": checkpoint["layers_1_based"],
            "hf_hidden_state_indices": checkpoint["hf_hidden_state_indices"],
        })
        queries, references = _row_maps(shape)
        cohort = shape["worst_case_cohort"]
        main_case = cohort["Q_REFSUM_MAX"]
        query_id = str(main_case["query_id"])
        execution_ordinal = int(main_case["execution_ordinal"])
        query_row = queries[query_id]
        if int(query_row["execution_ordinal"]) != execution_ordinal:
            raise ResourceError("query_id to dense execution_ordinal mapping drift")
        candidates = [int(value) for value in main_case["candidate_physical_rows"]]
        limit = 8 if args.mode == "smoke" else 128
        candidates = candidates[:limit]
        if len(candidates) != limit or len(set(candidates)) != limit:
            raise ResourceError("resource candidate cohort size/uniqueness drift")
        cache_root = args.cache_dir.resolve()
        if cache_root == output or output in cache_root.parents:
            raise ResourceError("stable cache directory must be outside job-specific result directory")
        cache_root.mkdir(parents=True, exist_ok=True)
        query_payload, query_resumed, query_seconds, query_bytes = _materialize_cache(
            image_row=query_row,
            path=cache_root / f"query_{execution_ordinal:04d}.pt",
            backbone=backbone,
            model_checkpoint_logical_sha256=model_receipt_sha,
            device=device,
        )
        # A second load is the required mid-query resume equivalence check.
        query_reloaded, query_logical = _load_cache(
            cache_root / f"query_{execution_ordinal:04d}.pt",
            str(query_row["source_image_sha256"]), model_receipt_sha,
        )
        query_resume_exact = query_logical == query_payload["logical_sha256"]
        reference_payloads: list[dict[str, Any]] = []
        encode_seconds: list[float] = [query_seconds] if query_seconds > 0 else []
        cache_bytes = query_bytes
        fresh = int(not query_resumed); resumed = int(query_resumed)
        split = min(64, limit)
        first_half_hashes: list[str] = []
        for ordinal, physical_row in enumerate(candidates):
            row = references.get(physical_row)
            if row is None:
                raise ResourceError(f"candidate outside frozen reference union: {physical_row}")
            payload, was_resumed, seconds, bytes_written = _materialize_cache(
                image_row=row,
                path=cache_root / f"reference_{physical_row:05d}.pt",
                backbone=backbone,
                model_checkpoint_logical_sha256=model_receipt_sha,
                device=device,
            )
            reference_payloads.append(payload)
            cache_bytes += bytes_written
            fresh += int(not was_resumed); resumed += int(was_resumed)
            if seconds > 0:
                encode_seconds.append(seconds)
            if ordinal < split:
                first_half_hashes.append(payload["logical_sha256"])
        # Re-open the first 64 candidate artifacts after the materialization
        # boundary. This catches nondeterministic/partial resume corruption.
        replayed_hashes = []
        for physical_row in candidates[:split]:
            payload, logical = _load_cache(
                cache_root / f"reference_{physical_row:05d}.pt",
                str(references[physical_row]["source_image_sha256"]), model_receipt_sha,
            )
            replayed_hashes.append(logical)
        reference_resume_exact = replayed_hashes == first_half_hashes

        cohort_audit = _cohort_audit_receipt(
            cohort, queries, references, cache_root, backbone,
            model_receipt_sha, device,
        )
        raw_pixel_cache_cohort = _materialize_raw_pixel_cache_cohort(
            cohort, queries, references, cache_root, backbone,
            model_receipt_sha, device,
        )

        math_receipt, math_tensors = _math_audit(device)
        _atomic_torch(output / "math_audit.pt", math_tensors)
        gradient_receipt, trained_hashes = _gradient_update_audit(device)

        frozen_training_episode = _select_max_training_episode(
            args.data_result_dir.resolve(), queries, references
        )
        train_query_payloads = []
        train_reference_payloads_by_query = []
        for selected_episode in frozen_training_episode["selected"]:
            train_query_row = selected_episode["query"]
            train_query_payload, _, _, _ = _materialize_cache(
                image_row=train_query_row,
                path=cache_root / f"query_{int(train_query_row['execution_ordinal']):04d}.pt",
                backbone=backbone,
                model_checkpoint_logical_sha256=model_receipt_sha,
                device=device,
            )
            train_query_payloads.append(train_query_payload)
            train_reference_payloads = []
            for physical_row in selected_episode["physical_rows_target_then_611"]:
                payload, _, _, _ = _materialize_cache(
                    image_row=references[physical_row],
                    path=cache_root / f"reference_{physical_row:05d}.pt",
                    backbone=backbone,
                    model_checkpoint_logical_sha256=model_receipt_sha,
                    device=device,
                )
                train_reference_payloads.append(payload)
            train_reference_payloads_by_query.append(train_reference_payloads)
        training_episode = _training_episode_benchmark(
            train_query_payloads, train_reference_payloads_by_query, device
        )
        training_episode["selection"] = {
            key: value for key, value in frozen_training_episode.items() if key != "selected"
        }
        training_episode["selected_queries"] = [
            {
                "query_id": item["query"]["id"],
                "shape_cost": item["shape_cost"],
                "tie_sha256": item["tie_sha256"],
                "physical_rows_target_then_611": item["physical_rows_target_then_611"],
            }
            for item in frozen_training_episode["selected"]
        ]

        core = DINO_RCDE_V1_2().to(device).eval()
        q_tokens, q_mask, q_grid = _core_inputs(query_reloaded, device)
        core.reset_cost_counter()
        torch.cuda.reset_peak_memory_stats(device)
        main_rss_before = _rss_bytes()
        unary_started = time.perf_counter()
        unary_cpu_started = time.process_time()
        evidence_rows = []
        for payload in reference_payloads:
            r_tokens, r_mask, r_grid = _core_inputs(payload, device)
            with torch.no_grad():
                evidence = core.decode_candidate_true_streaming_fast(
                    q_tokens, r_tokens, q_mask, r_mask, q_grid, r_grid,
                    query_tile_rows=EXACT_FAST_C128_QUERY_TILE_ROWS,
                    reference_tile_rows=EXACT_FAST_C128_REFERENCE_TILE_ROWS,
                )
            _assert_exact_fast_evidence(evidence)
            evidence_rows.append(evidence)
            del r_tokens, r_mask
        relational = torch.stack(
            [evidence.relational.detach() for evidence in evidence_rows], dim=0
        )
        _sync(device)
        unary_seconds = time.perf_counter() - unary_started
        unary_cpu_seconds = time.process_time() - unary_cpu_started
        cost_count = int(core.cost_volume_build_count)
        pair_started = time.perf_counter()
        pair_cpu_started = time.process_time()
        with torch.no_grad():
            left_indices = torch.tensor(
                [left for left in range(limit) for _ in range(left + 1, limit)],
                device=device, dtype=torch.long,
            )
            right_indices = torch.tensor(
                [right for left in range(limit) for right in range(left + 1, limit)],
                device=device, dtype=torch.long,
            )
            pair_chunks = [
                core.compare_relational_batch(
                    relational,
                    left_indices[start:start + EXACT_FAST_PAIR_CHUNK],
                    right_indices[start:start + EXACT_FAST_PAIR_CHUNK],
                    q_mask,
                )
                for start in range(0, int(left_indices.numel()), EXACT_FAST_PAIR_CHUNK)
            ]
            pair_logits_cpu = torch.cat(pair_chunks, dim=0).detach().cpu().contiguous()
            pair_count = int(pair_logits_cpu.numel())
            pair_digest = hashlib.sha256(pair_logits_cpu.numpy().tobytes())
        _sync(device)
        pair_seconds = time.perf_counter() - pair_started
        pair_cpu_seconds = time.process_time() - pair_cpu_started
        post_pair_cost_count = int(core.cost_volume_build_count)
        main_fast_resident = _exact_fast_resident_receipt(
            evidence_rows,
            query_tile_rows=EXACT_FAST_C128_QUERY_TILE_ROWS,
            reference_tile_rows=EXACT_FAST_C128_REFERENCE_TILE_ROWS,
        )
        main_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        main_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        main_peak_cpu_rss_delta = max(0, _rss_bytes() - main_rss_before)
        pre_resume_summary_hashes = [tensor_sha256(value.detach().cpu()) for value in relational]
        # Resume is not merely a token-file check: reopen the mid-query and
        # first 64 references, recompute RCDE summaries and a fixed output
        # stream, and require semantic equality.
        resume_core = DINO_RCDE_V1_2().to(device).eval()
        rq, rqm, rqg = _core_inputs(query_reloaded, device)
        resume_evidence_rows = []
        with torch.no_grad():
            for physical_row in candidates[:split]:
                reopened, _ = _load_cache(
                    cache_root / f"reference_{physical_row:05d}.pt",
                    str(references[physical_row]["source_image_sha256"]),
                    model_receipt_sha,
                )
                rt, rm, rg = _core_inputs(reopened, device)
                item = resume_core.decode_candidate_true_streaming_fast(
                    rq, rt, rqm, rm, rqg, rg,
                    query_tile_rows=EXACT_FAST_C128_QUERY_TILE_ROWS,
                    reference_tile_rows=EXACT_FAST_C128_REFERENCE_TILE_ROWS,
                )
                _assert_exact_fast_evidence(item)
                resume_evidence_rows.append(item)
            resume_relational = torch.stack(
                [item.relational.detach() for item in resume_evidence_rows], dim=0
            )
            resume_left = torch.tensor(
                [left for left in range(split) for _ in range(left + 1, split)],
                device=device, dtype=torch.long,
            )
            resume_right = torch.tensor(
                [right for left in range(split) for right in range(left + 1, split)],
                device=device, dtype=torch.long,
            )
            resume_pair_chunks = [
                resume_core.compare_relational_batch(
                    resume_relational,
                    resume_left[start:start + EXACT_FAST_PAIR_CHUNK],
                    resume_right[start:start + EXACT_FAST_PAIR_CHUNK],
                    rqm,
                )
                for start in range(0, int(resume_left.numel()), EXACT_FAST_PAIR_CHUNK)
            ]
            resume_output = torch.cat(resume_pair_chunks, dim=0).detach().cpu().contiguous()
        with torch.no_grad():
            original_pair_chunks = [
                core.compare_relational_batch(
                    relational[:split],
                    resume_left[start:start + EXACT_FAST_PAIR_CHUNK],
                    resume_right[start:start + EXACT_FAST_PAIR_CHUNK],
                    q_mask,
                )
                for start in range(0, int(resume_left.numel()), EXACT_FAST_PAIR_CHUNK)
            ]
            original_output = torch.cat(original_pair_chunks, dim=0).detach().cpu().contiguous()
        resume_summary_hashes = [
            tensor_sha256(item.relational.detach().cpu()) for item in resume_evidence_rows
        ]
        summary_resume_exact = resume_summary_hashes == pre_resume_summary_hashes[:split]
        output_resume_exact = tensor_sha256(resume_output) == tensor_sha256(original_output)

        # Measure the maximum-product pair through the actual deployment path:
        # projected-token -> haloed cross-cost tiles -> exact global three-pass
        # summaries.  Unlike the small math fixture above, this path is not
        # allowed to materialize full cost/Q/G/P.
        pair_case = cohort["Q_PAIR_MAX"]
        pair_query = queries[str(pair_case["query_id"])]
        pair_candidates = [int(value) for value in pair_case["candidate_physical_rows"]]
        frozen_reference = pair_case.get("reference")
        if not isinstance(frozen_reference, dict):
            raise ResourceError("Q_PAIR_MAX has no frozen reference member")
        pair_reference = int(frozen_reference["physical_row"])
        if pair_reference not in pair_candidates or pair_reference not in references:
            raise ResourceError("Q_PAIR_MAX frozen reference is outside its natural C128/union")
        if (
            int(frozen_reference["valid_tokens"]) != int(references[pair_reference]["valid_tokens"])
            or str(frozen_reference["source_image_sha256"]) != str(references[pair_reference]["source_image_sha256"])
            or int(pair_case["query_reference_token_product"])
            != int(pair_query["valid_tokens"]) * int(references[pair_reference]["valid_tokens"])
        ):
            raise ResourceError("Q_PAIR_MAX frozen argmax product/reference drift")
        if str(pair_query["id"]) == query_id:
            pq = query_payload
        else:
            pq, _, seconds, bytes_written = _materialize_cache(
                image_row=pair_query,
                path=cache_root / f"query_{int(pair_query['execution_ordinal']):04d}.pt",
                backbone=backbone,
                model_checkpoint_logical_sha256=model_receipt_sha,
                device=device,
            )
            if seconds > 0: encode_seconds.append(seconds)
            cache_bytes += bytes_written
        if pair_reference in candidates:
            pr = reference_payloads[candidates.index(pair_reference)]
        else:
            pr, _, seconds, bytes_written = _materialize_cache(
                image_row=references[pair_reference],
                path=cache_root / f"reference_{pair_reference:05d}.pt",
                backbone=backbone,
                model_checkpoint_logical_sha256=model_receipt_sha,
                device=device,
            )
            if seconds > 0: encode_seconds.append(seconds)
            cache_bytes += bytes_written
        pqt, pqm, pqg = _core_inputs(pq, device)
        prt, prm, prg = _core_inputs(pr, device)
        with torch.no_grad():
            streamed_worst = core.decode_candidate_true_streaming(
                pqt, prt, pqm, prm, pqg, prg,
                query_tile_rows=8, reference_tile_rows=8,
                materialize_assignment_for_audit=False,
            )
        expected_tile_count = math.ceil(pqg[0] / 8) * math.ceil(prg[0] / 8)
        worst_stream = {
            "query_id": pair_query["id"],
            "execution_ordinal": int(pair_query["execution_ordinal"]),
            "historical_query_ordinal": int(pair_query["query_ordinal"]),
            "reference_physical_row": pair_reference,
            "query_valid_tokens": int(pqm.sum()),
            "reference_valid_tokens": int(prm.sum()),
            "query_tile_rows": 8, "reference_tile_rows": 8, "halo": 2,
            "expected_tile_count_per_pass": expected_tile_count,
            "real_pass_tile_counts": list(streamed_worst.real_pass_tile_counts),
            "null_pass_tile_counts": list(streamed_worst.null_pass_tile_counts),
            "summary_pass_count": streamed_worst.summary_pass_count,
            "logical_cost_volume_build_count": streamed_worst.logical_cost_volume_build_count,
            "maximum_resident_cost_elements": streamed_worst.maximum_resident_cost_elements,
            "full_cost_element_count": int(pqt.shape[1] * prt.shape[1]),
            "full_cross_cost_materialized": False,
            "full_assignment_materialized": streamed_worst.assignment is not None,
            "matrix_sha256": dict(streamed_worst.matrix_sha256),
            "matrix_audit_slices": list(streamed_worst.matrix_audit_slices),
            "relational_sha256": tensor_sha256(streamed_worst.relational.detach().cpu()),
            "u_sha256": tensor_sha256(streamed_worst.u.detach().cpu()),
            "v_sha256": tensor_sha256(streamed_worst.v.detach().cpu()),
            "execution_path": "OLD_EXPLICIT_AUDIT_DECODER",
            "analytic_null": streamed_worst.analytic_null,
            "full_consensus_logits_resident": streamed_worst.full_consensus_logits_resident,
            "resident_consensus_logit_elements": streamed_worst.resident_consensus_logit_elements,
            "matrix_audit_enabled": streamed_worst.matrix_audit_enabled,
        }
        worst_stream["gate"] = (
            worst_stream["real_pass_tile_counts"] == [expected_tile_count] * 3
            and worst_stream["null_pass_tile_counts"] == [expected_tile_count] * 3
            and worst_stream["summary_pass_count"] == 3
            and worst_stream["logical_cost_volume_build_count"] == 1
            and worst_stream["maximum_resident_cost_elements"] < worst_stream["full_cost_element_count"]
            and worst_stream["full_cross_cost_materialized"] is False
            and worst_stream["full_assignment_materialized"] is False
            and set(worst_stream["matrix_sha256"]) == {"Q", "G", "P", "v"}
            and len(worst_stream["matrix_audit_slices"]) == 3
            and worst_stream["execution_path"] == "OLD_EXPLICIT_AUDIT_DECODER"
            and worst_stream["analytic_null"] is False
            and worst_stream["full_consensus_logits_resident"] is False
            and worst_stream["resident_consensus_logit_elements"] == 0
            and worst_stream["matrix_audit_enabled"] is True
        )

        pixel = _pixel_intervention_benchmark(pair_query, backbone, device)
        matched_lift = _matched_colnomic_lift_benchmark(
            protocol=protocol, matched=matched_colnomic, device=device
        )
        peak_allocated = max(
            int(training_episode["peak_cuda_allocated_bytes"]),
            int(pixel["peak_cuda_allocated_bytes"]), main_peak_allocated,
            *[int(row["peak_cuda_allocated_bytes"]) for row in cohort_audit["host_workloads"]],
            *[int(row["peak_cuda_allocated_bytes"]) for row in matched_lift["events"]],
        )
        peak_reserved = max(
            int(training_episode["peak_cuda_reserved_bytes"]),
            int(pixel["peak_cuda_reserved_bytes"]), main_peak_reserved,
            *[int(row["peak_cuda_reserved_bytes"]) for row in cohort_audit["host_workloads"]],
            *[int(row["peak_cuda_reserved_bytes"]) for row in matched_lift["events"]],
        )
        device_total = int(torch.cuda.get_device_properties(device).total_memory)

        cache_events = _cache_measurement_events(cache_root, device, model_receipt_sha)
        main_measurement = {
            "unary_wall_seconds": unary_seconds,
            "unary_cpu_seconds": unary_cpu_seconds,
            "pair_wall_seconds": pair_seconds,
            "pair_cpu_seconds": pair_cpu_seconds,
            "wall_seconds": unary_seconds + pair_seconds,
            "cpu_seconds": unary_cpu_seconds + pair_cpu_seconds,
            "peak_cuda_allocated_bytes": main_peak_allocated,
            "peak_cuda_reserved_bytes": main_peak_reserved,
            "peak_cpu_rss_delta_bytes": main_peak_cpu_rss_delta,
            "unary_count": limit, "pair_count": pair_count,
            "pair_chunk_size": EXACT_FAST_PAIR_CHUNK,
            "pair_device_to_host_transfer_count": 1,
            **main_fast_resident,
        }
        projection = _build_projection(
            shape=shape, cache_events=cache_events, cohort_audit=cohort_audit,
            training=training_episode, pixel=pixel, main_measurement=main_measurement,
            matched_colnomic=matched_colnomic, matched_lift=matched_lift,
        )
        limits = protocol["resource_limits"]
        limit_checks = {
            "cache_gib": projection["cache_gib_with_25pct_safety"] <= float(limits["cache_gib_max"]),
            "peak_gpu_gib": projection["peak_gpu_gib"] <= float(limits["peak_gpu_gib_max"]),
            "cache_build_gpu_hours": projection["cache_build_gpu_hours_with_25pct_safety"] <= float(limits["cache_build_gpu_hours_max"]),
            "single_fold_single_arm_gpu_hours": projection["single_fold_single_arm_gpu_hours_with_25pct_safety"] <= float(limits["single_fold_single_arm_gpu_hours_max"]),
            "p0_through_n1_gpu_hours": projection["p0_through_n1_gpu_hours_with_25pct_safety"] <= float(limits["p0_through_n1_gpu_hours_max"]),
        }
        exact_counts = (
            (args.mode == "formal" and limit == 128 and cost_count == 128 and post_pair_cost_count == 128 and pair_count == 8128)
            or (args.mode == "smoke" and cost_count == limit and post_pair_cost_count == limit and pair_count == limit * (limit - 1) // 2)
        )
        cross_node = None
        cross_node_gate = False
        if args.cross_node_validation is not None:
            cross_path = args.cross_node_validation.resolve()
            cross_node = json.loads(cross_path.read_text(encoding="utf-8"))
            cross_node_gate = (
                cross_node.get("schema_version") == "rc_dino_rcde_cross_node_reencode_reduction_v1_2_20260812"
                and cross_node.get("status") == "DINO_RCDE_CROSS_NODE_REENCODE_PASS"
                and cross_node.get("gate") is True
                and cross_node.get("distinct_eligible_nodes") is True
                and cross_node.get("protocol_sha256") == file_sha256(args.protocol.resolve())
                and cross_node.get("shape_ledger_sha256")
                == file_sha256(args.data_result_dir.resolve() / "resource_shape_ledger.json")
                and len(cross_node.get("node_receipt_sha256", [])) == 2
            )
        mandatory = {
            "frozen_checkpoint": checkpoint["revision"] == protocol["dino_freeze"]["revision"],
            "exact_core_parameter_count": gradient_receipt["parameter_count"] == EXPECTED_PARAMETER_COUNT,
            "math_contracts": bool(math_receipt["gate"]),
            "gradient_and_update": bool(gradient_receipt["gate"]),
            "real_max_training_episode": bool(training_episode["gate"]),
            "c128_unary_pair_reuse": bool(exact_counts),
            "exact_fast_execution": bool(
                exact_fast_qualification["resource_execution_ready"]
                and exact_fast_qualification["qualification_receipt_path"]
                == exact_fast_qualification["qualification_validation_path"]
                and exact_fast_qualification["qualification_receipt_sha256"]
                == exact_fast_qualification["qualification_validation_sha256"]
            ),
            "worst_dense_tiled_global_streaming": bool(worst_stream["gate"]),
            "mid_query_resume": bool(query_resume_exact and summary_resume_exact and output_resume_exact),
            "mid_reference_resume": bool(reference_resume_exact and summary_resume_exact and output_resume_exact),
            "pixel_intervention_17_forward": bool(pixel["gate"]),
            "all_frozen_cohort_cases_executed": bool(cohort_audit["gate"]),
            "raw_pixel_cache_cohort_executed": bool(raw_pixel_cache_cohort["gate"]),
            "matched_colnomic_lift_resource": bool(matched_lift["gate"]),
            # The parent contract requires independent re-encoding on two
            # eligible nodes. A single Slurm component can measure everything
            # else, but may not pretend this cross-node fact exists.
            "cross_node_reencode": cross_node_gate,
            "resource_limits": all(limit_checks.values()),
            "protected_access_zero": True,
        }
        benchmark_closed = args.mode == "formal" and all(mandatory.values())
        gate = benchmark_closed and bool(limits.get("analytic_estimate_can_close_gate") is False)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": READY if gate else (SMOKE if args.mode == "smoke" else BLOCKED),
            "mode": args.mode,
            "benchmark_closed": benchmark_closed,
            "resource_gate": gate,
            "claim_level": "P0_RESOURCE_ENGINEERING_ONLY_NO_E0_OR_SCIENTIFIC_SCORE",
            "next_authorized_stage": None,
            "protocol_sha256": file_sha256(args.protocol.resolve()),
            "data_result_sha256": file_sha256(args.data_result_dir.resolve() / "result.json"),
            "data_validation_sha256": file_sha256(args.data_result_dir.resolve() / "data_validation.json"),
            "resource_shape_ledger_sha256": file_sha256(args.data_result_dir.resolve() / "resource_shape_ledger.json"),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "core_sha256": file_sha256(ROOT / "src/rc_aslo_xf/dino_rcde_v1_2_resource_core.py"),
            "hf_wrapper_sha256": file_sha256(ROOT / "src/rc_aslo_xf/dino_rcde_hf_with_registers_v1_2.py"),
            "device": {
                "name": torch.cuda.get_device_name(device), "total_bytes": device_total,
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            },
            "backbone_receipt": dict(backbone_bundle.receipt),
            "exact_fast_qualification": exact_fast_qualification,
            "cross_node_reencode": {
                "provided": args.cross_node_validation is not None,
                "path": str(args.cross_node_validation.resolve()) if args.cross_node_validation else None,
                "sha256": file_sha256(args.cross_node_validation.resolve()) if args.cross_node_validation else None,
                "receipt": cross_node,
            },
            "parameter_and_gradient": gradient_receipt,
            "maximum_training_episode": training_episode,
            "trained_parameter_hashes_after_two_steps": trained_hashes,
            "math_contracts": math_receipt,
            "cohort_execution": {
                "case": "Q_REFSUM_MAX", "query_id": query_id,
                "execution_ordinal": execution_ordinal,
                "historical_query_ordinal": int(query_row["query_ordinal"]),
                "candidate_count": limit,
                "unary_count": limit, "cost_volume_build_count": cost_count,
                "cost_volume_build_count_after_pairs": post_pair_cost_count,
                "pair_count": pair_count, "pair_logit_stream_sha256": pair_digest.hexdigest(),
                "candidate_order": "NATURAL_C128_LEFT_MAJOR",
                "pair_order": "(0,1),(0,2),...,(126,127)",
                "pair_chunk_size": EXACT_FAST_PAIR_CHUNK,
                "pair_device_to_host_transfer_count": 1,
                **main_fast_resident,
                "unary_wall_seconds": unary_seconds, "pair_wall_seconds": pair_seconds,
                "unary_cpu_seconds": unary_cpu_seconds, "pair_cpu_seconds": pair_cpu_seconds,
                "peak_cuda_allocated_bytes": main_peak_allocated,
                "peak_cuda_reserved_bytes": main_peak_reserved,
                "peak_cpu_rss_delta_bytes": main_peak_cpu_rss_delta,
            },
            "worst_pair_dense_tiled_streaming": worst_stream,
            "frozen_worst_case_cohort": cohort_audit,
            "frozen_raw_pixel_cache_cohort": raw_pixel_cache_cohort,
            "cache_and_resume": {
                "fresh_artifact_count": fresh, "preexisting_resume_count": resumed,
                "mid_query_resume_hash_exact": query_resume_exact,
                "mid_reference_count": split,
                "mid_reference_resume_hash_exact": reference_resume_exact,
                "mid_summary_resume_hash_exact": summary_resume_exact,
                "mid_output_resume_hash_exact": output_resume_exact,
                "measured_cache_bytes": cache_bytes,
                "stable_cache_dir": str(cache_root),
                "stable_cache_outside_job_result": output not in cache_root.parents and cache_root != output,
                "cache_measurement_event_count": len(cache_events),
            },
            "pixel_intervention_resource": pixel,
            "matched_colnomic_lift_resource": matched_lift,
            "measurements": {
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "peak_cpu_rss_bytes": _rss_bytes(),
                "wall_seconds": time.perf_counter() - started_total,
            },
            "projections": projection,
            "limit_checks": limit_checks,
            "mandatory_checks": mandatory,
            "protected_access_counts": dict(PROTECTED_ZERO),
        }
        receipt["logical_sha256"] = _json_sha256(receipt)
        _write_json(output / "rcde_resource_benchmark_v1_2.json", receipt)
        _write_json(output / "result.json", {
            "schema_version": "rc_dino_rcde_p0_resource_result_v1_2_20260812",
            "status": receipt["status"], "benchmark_closed": benchmark_closed,
            "resource_gate": gate, "failure_domain": None if gate else "resource",
            "next_authorized_stage": None,
            "resource_receipt_sha256": file_sha256(output / "rcde_resource_benchmark_v1_2.json"),
            "protected_access_counts": dict(PROTECTED_ZERO),
        })
        print(json.dumps({"status": receipt["status"], "benchmark_closed": benchmark_closed, "resource_gate": gate}, sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "schema_version": "rc_dino_rcde_p0_resource_result_v1_2_20260812",
            "status": BLOCKED, "benchmark_closed": False, "resource_gate": False,
            "failure_domain": "resource_engineering", "next_authorized_stage": None,
            "error_type": type(error).__name__, "error": str(error),
            "protected_access_counts": dict(PROTECTED_ZERO),
        }
        _write_json(output / "result.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
