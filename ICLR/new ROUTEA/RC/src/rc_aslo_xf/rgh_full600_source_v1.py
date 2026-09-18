"""Public, target-free ColNomic source boundary for CW0-RGH-XF.

The immutable full600 geometry V2 payload is the sole query/C128 address
ledger.  This module exposes one execution at a time in physical-row-sorted
candidate order and delegates cache-entry, token, source-image, and ColNomic
geometry validation to the already qualified P-seal source loader.

No API in this module accepts or loads target roles, retrieval scores, ranks,
DINO tokens, a DINO/GX checkpoint, or a scientific result.  Exact-label role
joining and corrected-identity donor planning are separate pure functions so a
formal runner can keep the prejoin source phase physically isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from . import dino_rcde_cw1_multitile_pseal_sources_v1 as _qualified_source
from .dino_rcde_colnomic_superregion_v1 import CanonicalPatchGeometry


SCHEMA_VERSION = "rc_cw0_rgh_xf_v2_full600_source_v1_20260825"
SOURCE_LOGICAL_SCHEMA = "rc_cw0_rgh_xf_v2_execution_source_v1_20260825"
S0_SELECTION_SCHEMA = "rc_cw0_rgh_xf_v2_s0_selection_v1_20260825"
DONOR_PLAN_SCHEMA = "rc_cw0_rgh_xf_v2_corrected_identity_donor_plan_v1_20260825"
ROLE_JOIN_SCHEMA = "rc_cw0_rgh_xf_v2_exact_label_role_join_v1_20260825"

FULL600_GEOMETRY_FILE_SHA256 = (
    "fff5b980ffa88997ed9bb2a509686800a3448d18a93a714c4ef89007fb24329b"
)
FULL600_GEOMETRY_LOGICAL_SHA256 = (
    "d56db2cbae02629e861989285eb8cf53f8d4d5ff96a5eac0887e0176d5995a43"
)
FULL600_CANDIDATE_AXES_SHA256 = (
    "c0c6bbf6d02d4adcf8c9bc1e8d5dcba58d12986f3abd8b3274408ee0066d9ff2"
)
FULL600_REFERENCE_UNION_SHA256 = (
    "d3d905964a02c76fef40be368becab6e421e8af85bf553ccf3217d5c28519a96"
)
FULL600_QUERY_SEQUENCE_SHA256 = (
    "c632b32bebc0faf271ea6cb389a6d5c4add8a80c0bd8d8aa751c2ff6c353d508"
)
FULL600_REFERENCE_SEQUENCE_SHA256 = (
    "56e865b0068ffc4b636e5a759ee6f221bdfc43769475683154389fcb601187ea"
)
PREJOIN_FOLDS_FILE_SHA256 = (
    "44df51fe97a8db6de1a1101d0b310320da398d1eb88dd21b282c33cbcc6c6bc8"
)
PREJOIN_FOLDS_LOGICAL_SHA256 = (
    "325f5e7d3021c1d4db04ff8394f180471f3edd6fee707a6467b0d856fce2ca31"
)

EXPECTED_QUERY_COUNT = 600
EXPECTED_CANDIDATE_COUNT = 128
EXPECTED_REFERENCE_UNION_COUNT = 4_748
COLNOMIC_DIMENSION = 128

S0_SELECTION_NAMESPACE = "CW0_RGH_XF_V2_NATURAL_S0_QUERY_SELECTION_V1_20260825"
S0_TARGET_ABSENT_EXECUTIONS = frozenset({25, 26, 101, 346, 354, 470})
S0_OPENED_VISUAL_EXECUTIONS = frozenset({18, 89})

C_BIND_TRAIN_NAMESPACE = "CW0_RGH_XF_V2_C_BIND_TRAIN_FULL_C128_V1_20260825"
C_BIND_EVALUATION_NAMESPACE = (
    "CW0_RGH_XF_V2_C_BIND_EVALUATION_FULL_C128_V1_20260825"
)
_DONOR_NAMESPACES = frozenset({C_BIND_TRAIN_NAMESPACE, C_BIND_EVALUATION_NAMESPACE})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREJOIN_ALLOWED_FIELDS = frozenset(
    {"query_id", "query_ordinal", "inner_fold", "track", "source_image_sha256"}
)
_ROLE_ALLOWED_FIELDS = frozenset(
    {"query_id", "query_ordinal", "inner_fold", "track", "identity", "supergroup"}
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "target",
        "target_label",
        "target_identity",
        "rival",
        "raw_score",
        "d1_score",
        "rank",
        "winner",
        "outcome",
        "dino_tokens",
        "gx_score",
    }
)


class RGHFull600SourceError(RuntimeError):
    """Fail-closed source, selection, donor, or role-join violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RGHFull600SourceError(message)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, *, name: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{name} is not SHA-256")
    return str(value)


def _record_logical_sha256(value: Mapping[str, Any], *, self_key: str) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != self_key})


def _safe_json(path: Path, *, root: Path, expected_sha256: str) -> Mapping[str, Any]:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RGHFull600SourceError("JSON input escapes RC root") from error
    _require(resolved.is_file() and not resolved.is_symlink(), "JSON input is unsafe or absent")
    _require(_sha256_file(resolved) == expected_sha256, "JSON physical hash drift")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), "JSON input is not a mapping")
    return value


@dataclass(frozen=True)
class RGHColNomicSourceV1:
    kind: str
    native_key: int
    source_key: str
    source_path: str
    source_image_sha256: str
    cache_storage_kind: str
    cache_file_sha256: str
    cache_entry_index: int | None
    cache_entry_logical_sha256: str
    tokens: torch.Tensor
    tokens_sha256: str
    grid_shape: tuple[int, int]
    colnomic_geometry: CanonicalPatchGeometry
    source_logical_sha256: str

    def __post_init__(self) -> None:
        _require(self.kind in {"query", "reference"}, "source kind drift")
        _require(isinstance(self.native_key, int) and not isinstance(self.native_key, bool), "source native key drift")
        _sha256(self.source_image_sha256, name="source image")
        _sha256(self.cache_file_sha256, name="cache file")
        _sha256(self.cache_entry_logical_sha256, name="cache entry")
        _sha256(self.tokens_sha256, name="tokens")
        _sha256(self.source_logical_sha256, name="source logical")
        source_path = Path(self.source_path)
        _require(
            source_path.is_absolute()
            and source_path.is_file()
            and not source_path.is_symlink()
            and not (
                {part.lower() for part in source_path.parts}
                & _qualified_source._PROTECTED_PATH_PARTS
            ),
            "source path is unsafe, protected, or absent",
        )
        tokens = torch.as_tensor(self.tokens).detach().cpu().contiguous()
        _require(
            tokens.dtype == torch.float16
            and tokens.shape == (math.prod(self.grid_shape), COLNOMIC_DIMENSION)
            and bool(torch.isfinite(tokens).all())
            and _qualified_source.tensor_sha256(tokens) == self.tokens_sha256,
            "ColNomic token source drift",
        )
        _require(
            self.colnomic_geometry.grid_shape == self.grid_shape
            and self.colnomic_geometry.source_image_sha256 == self.source_image_sha256,
            "ColNomic token/geometry binding drift",
        )
        object.__setattr__(self, "tokens", tokens)


@dataclass(frozen=True)
class RGHCandidateReferenceV1:
    candidate_position: int
    physical_row: int
    source: RGHColNomicSourceV1

    def __post_init__(self) -> None:
        _require(self.candidate_position >= 0, "candidate position drift")
        _require(self.physical_row >= 0, "candidate physical row drift")
        _require(
            self.source.kind == "reference" and self.source.native_key == self.physical_row,
            "candidate/source binding drift",
        )


@dataclass(frozen=True)
class RGHExecutionSourcesV1:
    execution_ordinal: int
    historical_query_ordinal: int
    query_id: str
    inner_fold: int
    track: str
    source_image_sha256: str
    candidate_axis_sha256: str
    query: RGHColNomicSourceV1
    candidates: tuple[RGHCandidateReferenceV1, ...]
    source_logical_sha256: str

    def __post_init__(self) -> None:
        _require(0 <= self.execution_ordinal < EXPECTED_QUERY_COUNT, "execution ordinal drift")
        _require(self.inner_fold in {1, 2, 3, 4}, "inner fold drift")
        _sha256(self.source_image_sha256, name="query source image")
        _sha256(self.candidate_axis_sha256, name="candidate axis")
        _sha256(self.source_logical_sha256, name="execution source logical")
        _require(
            self.query.kind == "query"
            and self.query.native_key == self.historical_query_ordinal
            and self.query.source_image_sha256 == self.source_image_sha256,
            "query source binding drift",
        )
        rows = tuple(item.physical_row for item in self.candidates)
        _require(
            len(rows) == EXPECTED_CANDIDATE_COUNT
            and rows == tuple(sorted(rows))
            and len(set(rows)) == EXPECTED_CANDIDATE_COUNT
            and tuple(item.candidate_position for item in self.candidates)
            == tuple(range(EXPECTED_CANDIDATE_COUNT)),
            "physical-row-sorted C128 drift",
        )
        _require(_canonical_sha256(list(rows)) == self.candidate_axis_sha256, "candidate-axis hash drift")


def validate_full600_metadata(
    payload: Mapping[str, Any],
    schedule: Mapping[str, Any],
    *,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    expected_reference_count: int = EXPECTED_REFERENCE_UNION_COUNT,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[int, Mapping[str, Any]], tuple[Mapping[str, Any], ...]]:
    """Validate address/schedule metadata without opening any token payload."""

    queries = payload.get("query_records")
    references = payload.get("reference_records")
    cohort = payload.get("cohort")
    records = schedule.get("records")
    _require(
        isinstance(queries, list)
        and len(queries) == expected_query_count
        and isinstance(references, list)
        and len(references) == expected_reference_count
        and isinstance(cohort, Mapping)
        and isinstance(records, list)
        and len(records) == expected_query_count,
        "full600 metadata population drift",
    )
    _require(
        cohort.get("query_count") == expected_query_count
        and cohort.get("candidate_count_per_query") == expected_candidate_count
        and cohort.get("reference_union_count") == expected_reference_count,
        "full600 cohort count drift",
    )
    reference_by_row: dict[int, Mapping[str, Any]] = {}
    for record in references:
        _require(isinstance(record, Mapping), "reference record is not a mapping")
        row = record.get("physical_row")
        _require(isinstance(row, int) and not isinstance(row, bool) and row >= 0, "reference row drift")
        _require(row not in reference_by_row, "reference row duplicated")
        reference_by_row[row] = record

    historical: set[int] = set()
    query_ids: set[str] = set()
    for execution, (query, fold) in enumerate(zip(queries, records, strict=True)):
        _require(isinstance(query, Mapping) and isinstance(fold, Mapping), "query/schedule record drift")
        _require(set(fold) == _PREJOIN_ALLOWED_FIELDS, "prejoin schedule field drift")
        _require(not (set(query) & _FORBIDDEN_PUBLIC_KEYS), "query geometry contains a forbidden public field")
        _require(
            query.get("execution_ordinal") == execution
            and query.get("query_id") == fold.get("query_id")
            and query.get("historical_query_ordinal") == fold.get("query_ordinal")
            and query.get("heldout_fold") == fold.get("inner_fold")
            and query.get("source_image_sha256") == fold.get("source_image_sha256"),
            "query/schedule binding drift",
        )
        query_id = str(query["query_id"])
        history = int(query["historical_query_ordinal"])
        _require(query_id not in query_ids and history not in historical, "query identity/ordinal duplicated")
        query_ids.add(query_id)
        historical.add(history)
        axis = query.get("candidate_physical_rows")
        _require(
            isinstance(axis, list)
            and len(axis) == expected_candidate_count
            and all(isinstance(item, int) and not isinstance(item, bool) for item in axis)
            and axis == sorted(axis)
            and len(set(axis)) == expected_candidate_count
            and all(item in reference_by_row for item in axis)
            and query.get("candidate_axis_sha256") == _canonical_sha256(axis),
            "query C128 axis drift",
        )
    return tuple(queries), MappingProxyType(reference_by_row), tuple(records)


class RGHFull600SourceLoaderV1:
    """Load one target-free execution from immutable full600 ColNomic sources."""

    def __init__(
        self,
        *,
        rc_root: Path,
        geometry_payload_path: Path,
        prejoin_schedule_path: Path,
    ) -> None:
        self.rc_root = Path(rc_root).resolve()
        self.geometry_payload_path = Path(geometry_payload_path).resolve()
        self.prejoin_schedule_path = Path(prejoin_schedule_path).resolve()
        self._qualified_loader = _qualified_source.CW1MultiTilePSealSourceLoaderV1(
            payload_path=self.geometry_payload_path,
            rc_root=self.rc_root,
            expected_payload_file_sha256=FULL600_GEOMETRY_FILE_SHA256,
        )
        payload = self._qualified_loader._load_payload()
        _require(payload.get("logical_sha256") == FULL600_GEOMETRY_LOGICAL_SHA256, "geometry logical hash drift")
        _require(payload.get("query_record_sequence_sha256") == FULL600_QUERY_SEQUENCE_SHA256, "query sequence hash drift")
        _require(payload.get("reference_record_sequence_sha256") == FULL600_REFERENCE_SEQUENCE_SHA256, "reference sequence hash drift")
        cohort = payload.get("cohort")
        _require(
            isinstance(cohort, Mapping)
            and cohort.get("candidate_axes_sha256") == FULL600_CANDIDATE_AXES_SHA256
            and cohort.get("reference_union_sha256") == FULL600_REFERENCE_UNION_SHA256,
            "full600 candidate/reference population hash drift",
        )
        schedule = _safe_json(
            self.prejoin_schedule_path,
            root=self.rc_root,
            expected_sha256=PREJOIN_FOLDS_FILE_SHA256,
        )
        _require(
            schedule.get("schema_version") == "dino_rcde_prejoin_folds_600_v1_2_20260812"
            and schedule.get("status") == "DINO_RCDE_PREJOIN_FOLDS_600_PHYSICALLY_ISOLATED"
            and schedule.get("logical_sha256") == PREJOIN_FOLDS_LOGICAL_SHA256
            and _record_logical_sha256(schedule, self_key="logical_sha256")
            == PREJOIN_FOLDS_LOGICAL_SHA256,
            "prejoin schedule envelope/logical drift",
        )
        self._queries, self._references, self._schedule = validate_full600_metadata(payload, schedule)
        self._source_cache: dict[int, RGHColNomicSourceV1] = {}

    @property
    def execution_count(self) -> int:
        return len(self._queries)

    @property
    def shard_load_counts(self) -> Mapping[str, int]:
        return self._qualified_loader.shard_load_counts

    def _build_colnomic_source(
        self, record: Mapping[str, Any], *, kind: str, native_key: int
    ) -> RGHColNomicSourceV1:
        _require(kind in {"query", "reference"}, "source kind drift")
        _require(
            record.get("record_logical_sha256")
            == _qualified_source._logical_sha256(record, self_key="record_logical_sha256"),
            "geometry record logical hash drift",
        )
        source_sha = _sha256(record.get("source_image_sha256"), name="source image")
        query_id = record.get("query_id")
        source_key = (
            f"query:{native_key}:{query_id}" if kind == "query" else f"reference:{native_key}"
        )
        pointer = record.get("colnomic_cache")
        _require(isinstance(pointer, Mapping), "ColNomic pointer absent")
        entry, cache_file_sha = self._qualified_loader._entry_from_pointer(
            pointer=pointer, record=record, kind=kind, native_key=native_key
        )
        tokens, grid = self._qualified_loader._validate_tokens(entry, pointer)
        source_path = Path(str(record.get("source_path"))).resolve()
        _require(
            Path(str(entry.get("source_path"))).resolve() == source_path,
            "source path binding drift",
        )
        if kind == "query":
            _require(
                entry.get("query_id") == query_id
                and entry.get("source_image_tokens_sha256") == entry.get("tokens_sha256")
                and self._qualified_loader._source_file_sha256(source_path) == source_sha,
                "query source binding drift",
            )
        elif pointer.get("storage_kind") == _qualified_source.STORAGE_EXISTING:
            _require(entry.get("source_image_file_sha256") == source_sha, "reference source binding drift")
        else:
            _require(
                entry.get("source_image_sha256") == source_sha
                and self._qualified_loader._source_file_sha256(source_path) == source_sha,
                "supplemental reference source binding drift",
            )
        geometry = _qualified_source._geometry_from_record(
            record.get("colnomic_geometry"),
            expected_source_sha256=source_sha,
            expected_source_key=source_key,
        )
        _require(geometry.grid_shape == grid, "ColNomic token/geometry grid drift")
        source_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "native_key": native_key,
            "source_key": source_key,
            "source_image_sha256": source_sha,
            "cache_storage_kind": pointer["storage_kind"],
            "cache_file_sha256": cache_file_sha,
            "cache_entry_index": pointer.get("cache_entry_index"),
            "cache_entry_logical_sha256": pointer["entry_logical_sha256"],
            "tokens_sha256": pointer["tokens_sha256"],
            "grid_shape": list(grid),
            "colnomic_geometry_sha256": geometry.sha256,
        }
        return RGHColNomicSourceV1(
            kind=kind,
            native_key=native_key,
            source_key=source_key,
            source_path=str(source_path),
            source_image_sha256=source_sha,
            cache_storage_kind=str(pointer["storage_kind"]),
            cache_file_sha256=str(cache_file_sha),
            cache_entry_index=pointer.get("cache_entry_index"),
            cache_entry_logical_sha256=str(pointer["entry_logical_sha256"]),
            tokens=tokens,
            tokens_sha256=str(pointer["tokens_sha256"]),
            grid_shape=grid,
            colnomic_geometry=geometry,
            source_logical_sha256=_canonical_sha256(source_payload),
        )

    def load_execution(self, execution_ordinal: int) -> RGHExecutionSourcesV1:
        _require(
            isinstance(execution_ordinal, int)
            and not isinstance(execution_ordinal, bool)
            and 0 <= execution_ordinal < len(self._queries),
            "execution ordinal is out of range",
        )
        record = self._queries[execution_ordinal]
        schedule = self._schedule[execution_ordinal]
        historical = int(record["historical_query_ordinal"])
        query = self._build_colnomic_source(record, kind="query", native_key=historical)
        candidates: list[RGHCandidateReferenceV1] = []
        for position, physical_row in enumerate(record["candidate_physical_rows"]):
            row = int(physical_row)
            if row not in self._source_cache:
                self._source_cache[row] = self._build_colnomic_source(
                    self._references[row], kind="reference", native_key=row
                )
            candidates.append(
                RGHCandidateReferenceV1(
                    candidate_position=position,
                    physical_row=row,
                    source=self._source_cache[row],
                )
            )
        payload = {
            "schema_version": SOURCE_LOGICAL_SCHEMA,
            "execution_ordinal": execution_ordinal,
            "historical_query_ordinal": historical,
            "query_id": record["query_id"],
            "inner_fold": schedule["inner_fold"],
            "track": schedule["track"],
            "source_image_sha256": record["source_image_sha256"],
            "candidate_axis_sha256": record["candidate_axis_sha256"],
            "query_source_logical_sha256": query.source_logical_sha256,
            "candidate_source_logical_sha256": [item.source.source_logical_sha256 for item in candidates],
        }
        return RGHExecutionSourcesV1(
            execution_ordinal=execution_ordinal,
            historical_query_ordinal=historical,
            query_id=str(record["query_id"]),
            inner_fold=int(schedule["inner_fold"]),
            track=str(schedule["track"]),
            source_image_sha256=str(record["source_image_sha256"]),
            candidate_axis_sha256=str(record["candidate_axis_sha256"]),
            query=query,
            candidates=tuple(candidates),
            source_logical_sha256=_canonical_sha256(payload),
        )


@dataclass(frozen=True)
class RGHS0SelectedQueryV1:
    execution_ordinal: int
    selection_digest: str
    query_id: str
    query_ordinal: int
    source_image_sha256: str
    inner_fold: int
    track: str
    identity: str
    supergroup: str


@dataclass(frozen=True)
class RGHS0SelectionV1:
    namespace: str
    eligible_population_count: int
    eligible_population_sha256: str
    selected: tuple[RGHS0SelectedQueryV1, RGHS0SelectedQueryV1]
    logical_sha256: str


def select_result_blind_s0_queries(
    prejoin_records: Sequence[Mapping[str, Any]],
    role_records: Sequence[Mapping[str, Any]],
    *,
    namespace: str = S0_SELECTION_NAMESPACE,
    target_absent_executions: frozenset[int] = S0_TARGET_ABSENT_EXECUTIONS,
    opened_visual_executions: frozenset[int] = S0_OPENED_VISUAL_EXECUTIONS,
) -> RGHS0SelectionV1:
    """Select two plumbing queries without reading a score, rank, or outcome.

    Exact labels are used only after target-presence filtering and to enforce
    identity/supergroup diversity.  Thus the procedure is result-blind, not
    label-blind.
    """

    _require(namespace == S0_SELECTION_NAMESPACE, "S0 selection namespace drift")
    _require(len(prejoin_records) == len(role_records) == EXPECTED_QUERY_COUNT, "S0 input count drift")
    rows: list[tuple[str, int, Mapping[str, Any], Mapping[str, Any]]] = []
    for execution, (prejoin, role) in enumerate(zip(prejoin_records, role_records, strict=True)):
        _require(set(prejoin) == _PREJOIN_ALLOWED_FIELDS, "S0 prejoin field drift")
        _require(set(role) == _ROLE_ALLOWED_FIELDS, "S0 role field drift")
        _require(
            all(prejoin[key] == role[key] for key in ("query_id", "query_ordinal", "inner_fold", "track")),
            "S0 prejoin/role alignment drift",
        )
        if execution in target_absent_executions or execution in opened_visual_executions:
            continue
        digest_payload = {
            "namespace": namespace,
            "execution_ordinal": execution,
            "query_id": prejoin["query_id"],
            "query_ordinal": prejoin["query_ordinal"],
            "source_image_sha256": prejoin["source_image_sha256"],
            "inner_fold": prejoin["inner_fold"],
            "track": prejoin["track"],
        }
        rows.append((_canonical_sha256(digest_payload), execution, prejoin, role))
    rows.sort(key=lambda item: (item[0], item[1]))
    _require(rows, "S0 eligible population is empty")
    first = rows[0]
    second = next(
        (
            item
            for item in rows[1:]
            if item[3]["inner_fold"] != first[3]["inner_fold"]
            and item[3]["identity"] != first[3]["identity"]
            and item[3]["supergroup"] != first[3]["supergroup"]
            and item[3]["track"] != first[3]["track"]
        ),
        None,
    )
    _require(second is not None, "S0 has no result-blind diverse second query")

    def selected(item: tuple[str, int, Mapping[str, Any], Mapping[str, Any]]) -> RGHS0SelectedQueryV1:
        digest, execution, prejoin, role = item
        return RGHS0SelectedQueryV1(
            execution_ordinal=execution,
            selection_digest=digest,
            query_id=str(prejoin["query_id"]),
            query_ordinal=int(prejoin["query_ordinal"]),
            source_image_sha256=_sha256(prejoin["source_image_sha256"], name="S0 source image"),
            inner_fold=int(prejoin["inner_fold"]),
            track=str(prejoin["track"]),
            identity=str(role["identity"]),
            supergroup=str(role["supergroup"]),
        )

    chosen = (selected(first), selected(second))
    population_payload = [
        {"execution_ordinal": execution, "digest": digest}
        for digest, execution, _, _ in rows
    ]
    payload = {
        "schema_version": S0_SELECTION_SCHEMA,
        "namespace": namespace,
        "eligible_population_count": len(rows),
        "eligible_population_sha256": _canonical_sha256(population_payload),
        "selected": [item.__dict__ for item in chosen],
    }
    return RGHS0SelectionV1(
        namespace=namespace,
        eligible_population_count=len(rows),
        eligible_population_sha256=str(payload["eligible_population_sha256"]),
        selected=chosen,
        logical_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True)
class RGHCorrectedIdentityDonorPlanV1:
    namespace: str
    execution_ordinal: int
    candidate_axis_sha256: str
    corrected_identity_axis_sha256: str
    donor_positions: tuple[int, ...]
    donor_physical_rows: tuple[int, ...]
    logical_sha256: str


def _namespace_donor_offset(
    namespace: str,
    execution_ordinal: int,
    candidate_axis_sha256: str,
    legal_offsets: Sequence[int],
    *,
    exclude: frozenset[int] = frozenset(),
) -> int:
    available = tuple(offset for offset in legal_offsets if offset not in exclude)
    _require(bool(available), "donor namespace has no phase-distinct legal offset")
    return min(
        available,
        key=lambda offset: hashlib.sha256(
            (
                f"{namespace}|{execution_ordinal}|{candidate_axis_sha256}|{offset}"
            ).encode("utf-8")
        ).digest(),
    )


def corrected_identity_donor_derangement(
    *,
    namespace: str,
    execution_ordinal: int,
    candidate_physical_rows: Sequence[int],
    corrected_identity_by_physical_row: Mapping[int, str],
) -> RGHCorrectedIdentityDonorPlanV1:
    """Return a complete deterministic identity-disjoint cyclic donor plan."""

    _require(namespace in _DONOR_NAMESPACES, "donor namespace is not registered")
    rows = tuple(int(item) for item in candidate_physical_rows)
    _require(
        len(rows) == EXPECTED_CANDIDATE_COUNT
        and rows == tuple(sorted(rows))
        and len(set(rows)) == EXPECTED_CANDIDATE_COUNT,
        "donor candidate axis drift",
    )
    identities = tuple(corrected_identity_by_physical_row.get(row) for row in rows)
    _require(all(isinstance(item, str) and item for item in identities), "corrected identity absent")
    legal_offsets = [
        offset
        for offset in range(1, len(rows))
        if all(identities[position] != identities[(position + offset) % len(rows)] for position in range(len(rows)))
    ]
    _require(bool(legal_offsets), "candidate axis has no identity-disjoint cyclic derangement")
    axis_sha = _canonical_sha256(list(rows))
    train_offset = _namespace_donor_offset(
        C_BIND_TRAIN_NAMESPACE,
        execution_ordinal,
        axis_sha,
        legal_offsets,
    )
    offset = (
        train_offset
        if namespace == C_BIND_TRAIN_NAMESPACE
        else _namespace_donor_offset(
            C_BIND_EVALUATION_NAMESPACE,
            execution_ordinal,
            axis_sha,
            legal_offsets,
            exclude=frozenset({train_offset}),
        )
    )
    donor_positions = tuple((position + offset) % len(rows) for position in range(len(rows)))
    donor_rows = tuple(rows[position] for position in donor_positions)
    _require(
        tuple(sorted(donor_positions)) == tuple(range(len(rows)))
        and all(position != donor for position, donor in enumerate(donor_positions))
        and all(identities[position] != identities[donor] for position, donor in enumerate(donor_positions)),
        "donor plan is not a complete identity-disjoint derangement",
    )
    payload = {
        "schema_version": DONOR_PLAN_SCHEMA,
        "namespace": namespace,
        "execution_ordinal": execution_ordinal,
        "candidate_axis_sha256": axis_sha,
        "corrected_identity_axis_sha256": _canonical_sha256(list(identities)),
        "cyclic_offset": offset,
        "phase_distinct_from_train": (
            namespace == C_BIND_TRAIN_NAMESPACE or offset != train_offset
        ),
        "donor_positions": list(donor_positions),
        "donor_physical_rows": list(donor_rows),
        "response_dependent_retry_count": 0,
    }
    return RGHCorrectedIdentityDonorPlanV1(
        namespace=namespace,
        execution_ordinal=execution_ordinal,
        candidate_axis_sha256=str(payload["candidate_axis_sha256"]),
        corrected_identity_axis_sha256=str(payload["corrected_identity_axis_sha256"]),
        donor_positions=donor_positions,
        donor_physical_rows=donor_rows,
        logical_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True)
class RGHPrejoinCandidateAxisV1:
    execution_ordinal: int
    query_id: str
    query_ordinal: int
    source_image_sha256: str
    inner_fold: int
    track: str
    candidate_physical_rows: tuple[int, ...]
    candidate_axis_sha256: str
    population_seal_sha256: str


@dataclass(frozen=True)
class RGHExactLabelRoleJoinV1:
    execution_ordinal: int
    query_id: str
    target_corrected_identity: str
    target_present: bool
    target_candidate_position: int | None
    rival_candidate_positions: tuple[int, ...]
    role_record_sha256: str
    prejoin_population_seal_sha256: str
    logical_sha256: str


def validate_exact_label_role_join(
    prejoin: RGHPrejoinCandidateAxisV1,
    fold_record: Mapping[str, Any],
    role_record: Mapping[str, Any],
    corrected_identity_by_physical_row: Mapping[int, str],
) -> RGHExactLabelRoleJoinV1:
    """Join one exact training label only after an anonymous C128 seal exists."""

    _sha256(prejoin.population_seal_sha256, name="prejoin population seal")
    _require(set(fold_record) == _PREJOIN_ALLOWED_FIELDS, "role join fold schema drift")
    _require(set(role_record) == _ROLE_ALLOWED_FIELDS, "role join role schema drift")
    _require(
        prejoin.query_id == fold_record["query_id"] == role_record["query_id"]
        and prejoin.query_ordinal == fold_record["query_ordinal"] == role_record["query_ordinal"]
        and prejoin.inner_fold == fold_record["inner_fold"] == role_record["inner_fold"]
        and prejoin.track == fold_record["track"] == role_record["track"]
        and prejoin.source_image_sha256 == fold_record["source_image_sha256"],
        "role join query/fold/source drift",
    )
    rows = tuple(prejoin.candidate_physical_rows)
    _require(
        len(rows) == EXPECTED_CANDIDATE_COUNT
        and rows == tuple(sorted(rows))
        and len(set(rows)) == EXPECTED_CANDIDATE_COUNT
        and _canonical_sha256(list(rows)) == prejoin.candidate_axis_sha256,
        "role join C128 axis drift",
    )
    identities = tuple(corrected_identity_by_physical_row.get(row) for row in rows)
    _require(
        all(isinstance(item, str) and item for item in identities)
        and len(set(identities)) == EXPECTED_CANDIDATE_COUNT,
        "role join corrected identity axis is absent or duplicated",
    )
    target_identity = str(role_record["identity"])
    target_positions = tuple(index for index, identity in enumerate(identities) if identity == target_identity)
    _require(len(target_positions) <= 1, "target corrected identity is duplicated in C128")
    target_position = target_positions[0] if target_positions else None
    rivals = tuple(index for index, identity in enumerate(identities) if identity != target_identity)
    role_sha = _canonical_sha256(dict(role_record))
    payload = {
        "schema_version": ROLE_JOIN_SCHEMA,
        "execution_ordinal": prejoin.execution_ordinal,
        "query_id": prejoin.query_id,
        "target_corrected_identity": target_identity,
        "target_present": target_position is not None,
        "target_candidate_position": target_position,
        "rival_candidate_positions": list(rivals),
        "role_record_sha256": role_sha,
        "prejoin_population_seal_sha256": prejoin.population_seal_sha256,
    }
    return RGHExactLabelRoleJoinV1(
        execution_ordinal=prejoin.execution_ordinal,
        query_id=prejoin.query_id,
        target_corrected_identity=target_identity,
        target_present=target_position is not None,
        target_candidate_position=target_position,
        rival_candidate_positions=rivals,
        role_record_sha256=role_sha,
        prejoin_population_seal_sha256=prejoin.population_seal_sha256,
        logical_sha256=_canonical_sha256(payload),
    )


__all__ = [
    "SCHEMA_VERSION",
    "S0_SELECTION_NAMESPACE",
    "S0_TARGET_ABSENT_EXECUTIONS",
    "S0_OPENED_VISUAL_EXECUTIONS",
    "C_BIND_TRAIN_NAMESPACE",
    "C_BIND_EVALUATION_NAMESPACE",
    "RGHFull600SourceError",
    "RGHColNomicSourceV1",
    "RGHCandidateReferenceV1",
    "RGHExecutionSourcesV1",
    "RGHFull600SourceLoaderV1",
    "RGHS0SelectedQueryV1",
    "RGHS0SelectionV1",
    "RGHCorrectedIdentityDonorPlanV1",
    "RGHPrejoinCandidateAxisV1",
    "RGHExactLabelRoleJoinV1",
    "validate_full600_metadata",
    "select_result_blind_s0_queries",
    "corrected_identity_donor_derangement",
    "validate_exact_label_role_join",
]
