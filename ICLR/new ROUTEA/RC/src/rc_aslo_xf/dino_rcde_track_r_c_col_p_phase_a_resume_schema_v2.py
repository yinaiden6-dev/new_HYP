"""Append-only resumable Phase-A schemas for the V124 C_COL_P repair.

The schemas in this module are shared by the production and independent
implementations.  They intentionally contain no model or source-loading code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_DATE = "20260824"
CANDIDATE_COUNT = 128
DIRECTION_COUNT = 2
SOURCE_SHARD_SIZE = 16
CONTROL_SHARD_SIZE = 16
PROGRESS_INTERVAL = 8
NAMESPACE = "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_RESUME_V2"

QUERY_STAGE_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_query_stage_v2_20260824"
)
QUERY_STAGE_STATUS = "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_QUERY_STAGE_READY"
SOURCE_SHARD_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_source_shard_v2_20260824"
)
SOURCE_SHARD_STATUS = "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_SOURCE_SHARD_READY"
SOURCE_CHUNK_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_source_chunk_v2_20260824"
)
SOURCE_CHUNK_STATUS = "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_SOURCE_CHUNK_READY"
CONTEXT_STAGE_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_context_stage_v2_20260824"
)
CONTEXT_STAGE_STATUS = "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_CONTEXT_STAGE_READY"
CONTROL_SHARD_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_control_shard_v2_20260824"
)
CONTROL_SHARD_STATUS = "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_CONTROL_SHARD_READY"
CONTROL_CHUNK_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_control_chunk_v2_20260824"
)
CONTROL_CHUNK_STATUS = "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_CONTROL_CHUNK_READY"
CONTROL_AGGREGATE_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_control_aggregate_v2_20260824"
)
CONTROL_AGGREGATE_STATUS = (
    "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_CONTROL_AGGREGATE_READY"
)
PROGRESS_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_progress_v2_20260824"
)
PRODUCER_RESULT_SCHEMA = (
    "rc_dino_rcde_track_r_v124_c_col_p_phase_a_result_v2_20260824"
)
PRODUCER_RESULT_STATUS = "DINO_RCDE_TRACK_R_V124_C_COL_P_PHASE_A_V2_PASS"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_shard_ranges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, start + SOURCE_SHARD_SIZE)
        for start in range(0, CANDIDATE_COUNT, SOURCE_SHARD_SIZE)
    )


def control_shard_ranges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, start + CONTROL_SHARD_SIZE)
        for start in range(0, CANDIDATE_COUNT, CONTROL_SHARD_SIZE)
    )


def progress_chunk_ranges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, start + PROGRESS_INTERVAL)
        for start in range(0, CANDIDATE_COUNT, PROGRESS_INTERVAL)
    )


def shard_stem(start: int, stop: int) -> str:
    if (start, stop) not in source_shard_ranges():
        raise ValueError("Phase-A V2 shard range drift")
    return f"shard_{start:03d}_{stop:03d}"


__all__ = [
    "CANDIDATE_COUNT",
    "CONTROL_CHUNK_SCHEMA",
    "CONTROL_CHUNK_STATUS",
    "CONTROL_AGGREGATE_SCHEMA",
    "CONTROL_AGGREGATE_STATUS",
    "CONTROL_SHARD_SCHEMA",
    "CONTROL_SHARD_SIZE",
    "CONTROL_SHARD_STATUS",
    "CONTEXT_STAGE_SCHEMA",
    "CONTEXT_STAGE_STATUS",
    "DIRECTION_COUNT",
    "NAMESPACE",
    "PRODUCER_RESULT_SCHEMA",
    "PRODUCER_RESULT_STATUS",
    "PROGRESS_INTERVAL",
    "PROGRESS_SCHEMA",
    "QUERY_STAGE_SCHEMA",
    "QUERY_STAGE_STATUS",
    "SOURCE_SHARD_SCHEMA",
    "SOURCE_SHARD_SIZE",
    "SOURCE_SHARD_STATUS",
    "SOURCE_CHUNK_SCHEMA",
    "SOURCE_CHUNK_STATUS",
    "canonical_sha256",
    "control_shard_ranges",
    "file_sha256",
    "logical_sha256",
    "progress_chunk_ranges",
    "shard_stem",
    "source_shard_ranges",
]
