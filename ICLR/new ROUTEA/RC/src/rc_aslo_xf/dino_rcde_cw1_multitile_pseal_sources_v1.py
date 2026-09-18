"""Target-free natural sources for the CW1 multi-tile P-seal E1 stage.

The only public cohort selected here is the frozen historical current-eight
population, and every selected query retains its complete natural C128 physical
row axis.  This module is deliberately a source boundary: it does not read a
target, label, retrieval result, D1 quantity, protected endpoint, or trainable
model.  It validates immutable pointers from the independently qualified
full-600 geometry V2 payload and returns only ColNomic tokens plus canonical
ColNomic/DINO geometry.

Two ColNomic storage forms are supported because both occur in the qualified
payload:

* ``existing_spatial_cache`` addresses an exact entry in one frozen shard; and
* ``supplemental_current_payload`` addresses one of the 46 CPU-replayed gallery
  entries embedded in the geometry payload itself.

Every existing shard is file-hashed and deserialized at most once per loader.
Repeated candidate rows reuse one validated immutable source object.
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

from .colnomic_proposal_tokens import ColNomicLocalGrid, proposal_grid_sha256
from .dino_rcde_colnomic_superregion_v1 import CanonicalPatchGeometry


SCHEMA_VERSION = "rc_dino_rcde_cw1_multitile_pseal_sources_v1_20260815"
FULL600_GEOMETRY_SCHEMA = (
    "rc_dino_rcde_colnomic_sr_full600_geometry_payload_v2_20260815"
)
FULL600_GEOMETRY_STATUS = "RCDE_SR_FULL600_CANONICAL_GEOMETRY_V2_READY"
DEFAULT_GEOMETRY_PAYLOAD_SHA256 = (
    "fff5b980ffa88997ed9bb2a509686800a3448d18a93a714c4ef89007fb24329b"
)

CURRENT8_HISTORICAL_ORDINALS = (47, 127, 158, 168, 231, 472, 603, 915)
EXPECTED_QUERY_COUNT = 600
EXPECTED_CANDIDATE_COUNT = 128
EXPECTED_REFERENCE_COUNT = 4_748
COLNOMIC_DIMENSION = 128

STORAGE_EXISTING = "existing_spatial_cache"
STORAGE_SUPPLEMENTAL = "supplemental_current_payload"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROTECTED_PATH_PARTS = frozenset({"opened", "sealed", "c8", "s8"})


class CW1PSealSourceError(RuntimeError):
    """A payload, pointer, token, source, geometry, or cohort binding failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CW1PSealSourceError(message)


def _sha(value: Any, *, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA256",
    )
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash exact tensor dtype, shape, and contiguous bytes."""

    _require(isinstance(value, torch.Tensor), "token/hash value is not a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _canonical_view(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": tensor_sha256(tensor),
            }
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical_view(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_view(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_object_sha256(value: Any) -> str:
    rendered = json.dumps(
        _canonical_view(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _logical_sha256(value: Mapping[str, Any], *, self_key: str) -> str:
    return canonical_object_sha256(
        {key: item for key, item in value.items() if key != self_key}
    )


def _plain_canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _spatial_payload_logical_sha256(value: Mapping[str, Any]) -> str:
    entries = value.get("entries")
    _require(isinstance(entries, list), "spatial cache entries are absent")
    core = {
        key: item
        for key, item in value.items()
        if key not in {"entries", "logical_sha256"}
    }
    core["entries"] = [
        {key: item for key, item in entry.items() if key != "tokens"}
        if isinstance(entry, Mapping)
        else entry
        for entry in entries
    ]
    return _plain_canonical_sha256(core)


def _geometry_from_record(
    value: Any,
    *,
    expected_source_sha256: str,
    expected_source_key: str,
) -> CanonicalPatchGeometry:
    _require(isinstance(value, Mapping), "canonical geometry record is absent")
    required = {
        "source_image_sha256",
        "source_key",
        "processor_config_sha256",
        "raw_size_hw",
        "oriented_size_hw",
        "exif_orientation",
        "raw_to_oriented_affine",
        "raw_to_oriented_affine_sha256",
        "grid_shape",
        "valid_patch_mask",
        "valid_patch_mask_sha256",
        "cell_boxes_xyxy",
        "cell_boxes_xyxy_sha256",
        "coordinate_frame",
        "geometry_sha256",
        "logical_sha256",
    }
    _require(set(value) == required, "canonical geometry schema drift")
    _require(
        value.get("source_image_sha256") == expected_source_sha256
        and value.get("source_key") == expected_source_key,
        "canonical geometry source binding drift",
    )
    for tensor_name, hash_name in (
        ("raw_to_oriented_affine", "raw_to_oriented_affine_sha256"),
        ("valid_patch_mask", "valid_patch_mask_sha256"),
        ("cell_boxes_xyxy", "cell_boxes_xyxy_sha256"),
    ):
        _require(
            isinstance(value.get(tensor_name), torch.Tensor)
            and tensor_sha256(value[tensor_name]) == value.get(hash_name),
            f"canonical geometry {tensor_name} hash drift",
        )
    _require(
        _logical_sha256(value, self_key="logical_sha256")
        == value.get("logical_sha256"),
        "canonical geometry logical hash drift",
    )
    try:
        geometry = CanonicalPatchGeometry(
            source_image_sha256=str(value["source_image_sha256"]),
            source_key=str(value["source_key"]),
            processor_config_sha256=str(value["processor_config_sha256"]),
            raw_size_hw=tuple(int(item) for item in value["raw_size_hw"]),
            oriented_size_hw=tuple(int(item) for item in value["oriented_size_hw"]),
            exif_orientation=int(value["exif_orientation"]),
            raw_to_oriented_affine=torch.as_tensor(value["raw_to_oriented_affine"]),
            grid_shape=tuple(int(item) for item in value["grid_shape"]),
            valid_patch_mask=torch.as_tensor(value["valid_patch_mask"]),
            cell_boxes_xyxy=torch.as_tensor(value["cell_boxes_xyxy"]),
            coordinate_frame=str(value["coordinate_frame"]),
        )
    except (TypeError, ValueError) as error:
        raise CW1PSealSourceError("canonical geometry construction failed") from error
    _require(
        geometry.sha256 == value.get("geometry_sha256"),
        "canonical geometry object hash drift",
    )
    return geometry


@dataclass(frozen=True)
class ColNomicTokenSourceV1:
    """One validated anonymous image source on both native geometry rasters."""

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
    dino_geometry: CanonicalPatchGeometry
    source_logical_sha256: str

    def __post_init__(self) -> None:
        _require(self.kind in {"query", "reference"}, "source kind drift")
        _require(
            isinstance(self.native_key, int) and not isinstance(self.native_key, bool),
            "source native key drift",
        )
        _require(isinstance(self.source_key, str) and self.source_key, "source key absent")
        source_path = Path(self.source_path)
        _require(
            source_path.is_absolute()
            and source_path.is_file()
            and not source_path.is_symlink()
            and not ({part.lower() for part in source_path.parts} & _PROTECTED_PATH_PARTS),
            "public source image path is unsafe or absent",
        )
        _sha(self.source_image_sha256, name="source image")
        _require(
            self.cache_storage_kind in {STORAGE_EXISTING, STORAGE_SUPPLEMENTAL},
            "source storage kind drift",
        )
        _sha(self.cache_file_sha256, name="source cache file")
        _sha(self.cache_entry_logical_sha256, name="source cache entry")
        _sha(self.tokens_sha256, name="source tokens")
        _sha(self.source_logical_sha256, name="source logical")
        tokens = torch.as_tensor(self.tokens).detach().cpu().contiguous()
        _require(
            tokens.dtype == torch.float16
            and tokens.shape == (math.prod(self.grid_shape), COLNOMIC_DIMENSION)
            and bool(torch.isfinite(tokens).all())
            and tensor_sha256(tokens) == self.tokens_sha256,
            "public ColNomic token source drift",
        )
        _require(
            self.colnomic_geometry.grid_shape == self.grid_shape
            and self.colnomic_geometry.source_image_sha256 == self.source_image_sha256
            and self.dino_geometry.source_image_sha256 == self.source_image_sha256,
            "public source geometry binding drift",
        )
        object.__setattr__(self, "tokens", tokens)


@dataclass(frozen=True)
class CandidateReferenceSourceV1:
    candidate_position: int
    physical_row: int
    source: ColNomicTokenSourceV1

    def __post_init__(self) -> None:
        _require(
            isinstance(self.candidate_position, int)
            and not isinstance(self.candidate_position, bool)
            and self.candidate_position >= 0,
            "candidate position drift",
        )
        _require(
            isinstance(self.physical_row, int)
            and not isinstance(self.physical_row, bool)
            and self.physical_row >= 0,
            "candidate physical row drift",
        )
        _require(
            self.source.kind == "reference" and self.source.native_key == self.physical_row,
            "candidate/reference binding drift",
        )


@dataclass(frozen=True)
class Current8QuerySourceV1:
    historical_query_ordinal: int
    execution_ordinal: int
    query_id: str
    candidate_axis_sha256: str
    query: ColNomicTokenSourceV1
    candidates: tuple[CandidateReferenceSourceV1, ...]
    source_logical_sha256: str

    def __post_init__(self) -> None:
        _require(
            self.historical_query_ordinal in CURRENT8_HISTORICAL_ORDINALS,
            "query is outside frozen current-eight",
        )
        _require(
            isinstance(self.execution_ordinal, int)
            and not isinstance(self.execution_ordinal, bool)
            and self.execution_ordinal >= 0,
            "query execution ordinal drift",
        )
        _require(isinstance(self.query_id, str) and self.query_id, "query id absent")
        _sha(self.candidate_axis_sha256, name="candidate axis")
        _sha(self.source_logical_sha256, name="query source logical")
        _require(
            self.query.kind == "query"
            and self.query.native_key == self.historical_query_ordinal,
            "query source native-key drift",
        )
        _require(
            len(self.candidates) == EXPECTED_CANDIDATE_COUNT
            and tuple(item.candidate_position for item in self.candidates)
            == tuple(range(EXPECTED_CANDIDATE_COUNT))
            and len({item.physical_row for item in self.candidates})
            == EXPECTED_CANDIDATE_COUNT,
            "complete C128 candidate axis drift",
        )
        _require(
            _plain_canonical_sha256([item.physical_row for item in self.candidates])
            == self.candidate_axis_sha256,
            "public candidate-axis hash drift",
        )


@dataclass(frozen=True)
class Current8PSealSourcesV1:
    queries: tuple[Current8QuerySourceV1, ...]
    geometry_payload_file_sha256: str
    geometry_payload_logical_sha256: str
    unique_reference_count: int
    source_logical_sha256: str

    def __post_init__(self) -> None:
        _require(
            tuple(item.historical_query_ordinal for item in self.queries)
            == CURRENT8_HISTORICAL_ORDINALS,
            "current-eight query order/population drift",
        )
        _sha(self.geometry_payload_file_sha256, name="geometry payload file")
        _sha(self.geometry_payload_logical_sha256, name="geometry payload logical")
        _sha(self.source_logical_sha256, name="source cohort logical")
        unique = {
            item.physical_row
            for query in self.queries
            for item in query.candidates
        }
        _require(self.unique_reference_count == len(unique), "unique reference count drift")


class CW1MultiTilePSealSourceLoaderV1:
    """Validate and load the immutable result-blind current-eight source cohort."""

    def __init__(
        self,
        *,
        payload_path: Path,
        rc_root: Path | None = None,
        expected_payload_file_sha256: str = DEFAULT_GEOMETRY_PAYLOAD_SHA256,
    ) -> None:
        self.payload_path = Path(payload_path).resolve()
        self.rc_root = (
            Path(__file__).resolve().parents[2]
            if rc_root is None
            else Path(rc_root).resolve()
        )
        self.expected_payload_file_sha256 = _sha(
            expected_payload_file_sha256, name="expected geometry payload file"
        )
        self._payload: Mapping[str, Any] | None = None
        self._payload_file_sha256: str | None = None
        self._shard_payloads: dict[Path, Mapping[str, Any]] = {}
        self._file_hashes: dict[Path, str] = {}
        self._source_file_hashes: dict[Path, str] = {}
        self._reference_sources: dict[int, ColNomicTokenSourceV1] = {}
        self._shard_load_counts: dict[Path, int] = {}

    @property
    def shard_load_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {str(path): count for path, count in sorted(self._shard_load_counts.items())}
        )

    def _safe_cache_path(self, value: Any) -> Path:
        _require(isinstance(value, str) and value, "cache pointer path absent")
        raw = Path(value)
        path = raw.resolve() if raw.is_absolute() else (self.rc_root / raw).resolve()
        try:
            path.relative_to(self.rc_root)
        except ValueError as error:
            raise CW1PSealSourceError("cache pointer escapes RC root") from error
        _require(
            not ({part.lower() for part in path.parts} & _PROTECTED_PATH_PARTS),
            "cache pointer enters protected path",
        )
        _require(path.is_file() and not path.is_symlink(), "cache pointer is unsafe or absent")
        return path

    def _file_sha256(self, path: Path) -> str:
        if path not in self._file_hashes:
            self._file_hashes[path] = sha256_file(path)
        return self._file_hashes[path]

    def _source_file_sha256(self, path: Path) -> str:
        path = path.resolve()
        _require(path.is_file(), "source image is absent")
        _require(
            not ({part.lower() for part in path.parts} & _PROTECTED_PATH_PARTS),
            "source image enters protected path",
        )
        if path not in self._source_file_hashes:
            self._source_file_hashes[path] = sha256_file(path)
        return self._source_file_hashes[path]

    def _load_payload(self) -> Mapping[str, Any]:
        if self._payload is not None:
            return self._payload
        _require(
            self.payload_path.is_file() and not self.payload_path.is_symlink(),
            "full600 geometry payload is unsafe or absent",
        )
        try:
            self.payload_path.relative_to(self.rc_root)
        except ValueError as error:
            raise CW1PSealSourceError("geometry payload escapes RC root") from error
        file_sha = self._file_sha256(self.payload_path)
        _require(
            file_sha == self.expected_payload_file_sha256,
            "full600 geometry payload file hash drift",
        )
        value = torch.load(
            self.payload_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        _require(isinstance(value, Mapping), "full600 geometry payload schema drift")
        cohort = value.get("cohort")
        queries = value.get("query_records")
        references = value.get("reference_records")
        supplements = value.get("supplemental_gallery_entries")
        _require(
            value.get("schema_version") == FULL600_GEOMETRY_SCHEMA
            and value.get("status") == FULL600_GEOMETRY_STATUS
            and isinstance(cohort, Mapping)
            and cohort.get("query_count") == EXPECTED_QUERY_COUNT
            and cohort.get("candidate_count_per_query") == EXPECTED_CANDIDATE_COUNT
            and cohort.get("reference_union_count") == EXPECTED_REFERENCE_COUNT
            and isinstance(queries, list)
            and len(queries) == EXPECTED_QUERY_COUNT
            and isinstance(references, list)
            and len(references) == EXPECTED_REFERENCE_COUNT
            and isinstance(supplements, list)
            and _SHA256.fullmatch(str(value.get("logical_sha256", ""))) is not None,
            "full600 geometry payload envelope drift",
        )
        self._payload = value
        self._payload_file_sha256 = file_sha
        return value

    def _load_existing_shard(
        self, pointer: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Path]:
        path = self._safe_cache_path(pointer.get("cache_path"))
        expected_file_sha = _sha(
            pointer.get("cache_file_sha256"), name="spatial cache file"
        )
        _require(self._file_sha256(path) == expected_file_sha, "spatial cache file hash drift")
        if path not in self._shard_payloads:
            value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            _require(isinstance(value, Mapping), "spatial cache payload schema drift")
            _require(
                value.get("logical_sha256")
                == _sha(
                    pointer.get("cache_payload_logical_sha256"),
                    name="spatial cache payload",
                )
                and _spatial_payload_logical_sha256(value) == value.get("logical_sha256"),
                "spatial cache payload logical hash drift",
            )
            self._shard_payloads[path] = value
            self._shard_load_counts[path] = self._shard_load_counts.get(path, 0) + 1
        else:
            _require(
                self._shard_payloads[path].get("logical_sha256")
                == pointer.get("cache_payload_logical_sha256"),
                "shared spatial cache pointer logical drift",
            )
        return self._shard_payloads[path], path

    @staticmethod
    def _validate_tokens(
        entry: Mapping[str, Any], pointer: Mapping[str, Any]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        tokens = entry.get("tokens")
        grid = entry.get("grid_shape")
        _require(
            isinstance(grid, list)
            and len(grid) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in grid),
            "ColNomic token grid drift",
        )
        shape = (int(grid[0]), int(grid[1]))
        _require(
            isinstance(tokens, torch.Tensor)
            and tokens.dtype == torch.float16
            and tokens.shape == (math.prod(shape), COLNOMIC_DIMENSION)
            and bool(torch.isfinite(tokens).all()),
            "ColNomic token tensor drift",
        )
        digest = tensor_sha256(tokens)
        _require(
            digest == entry.get("tokens_sha256") == pointer.get("tokens_sha256")
            and list(shape) == pointer.get("grid_shape"),
            "ColNomic token/grid pointer hash drift",
        )
        return tokens.detach().cpu().contiguous(), shape

    @staticmethod
    def _existing_entry_logical(entry: Mapping[str, Any], *, kind: str) -> str:
        if kind == "query":
            return _plain_canonical_sha256(
                {
                    key: item
                    for key, item in entry.items()
                    if key not in {"tokens", "entry_logical_sha256"}
                }
            )
        indices = entry.get("image_token_indices")
        grid = entry.get("grid_shape")
        tokens = entry.get("tokens")
        _require(
            isinstance(indices, list)
            and isinstance(grid, list)
            and len(grid) == 2
            and isinstance(tokens, torch.Tensor),
            "gallery spatial index receipt drift",
        )
        try:
            local = ColNomicLocalGrid(
                tokens=tokens,
                grid_shape=(int(grid[0]), int(grid[1])),
                image_token_indices=torch.tensor(indices, dtype=torch.long),
            )
        except (TypeError, ValueError) as error:
            raise CW1PSealSourceError("gallery spatial grid reconstruction failed") from error
        return _plain_canonical_sha256(
            {
                **{
                    key: item
                    for key, item in entry.items()
                    if key not in {"tokens", "entry_logical_sha256"}
                },
                "proposal_grid_sha256": proposal_grid_sha256(local),
            }
        )

    def _entry_from_pointer(
        self,
        *,
        pointer: Mapping[str, Any],
        record: Mapping[str, Any],
        kind: str,
        native_key: int,
    ) -> tuple[Mapping[str, Any], str]:
        storage = pointer.get("storage_kind")
        _require(storage in {STORAGE_EXISTING, STORAGE_SUPPLEMENTAL}, "unknown cache storage kind")
        if storage == STORAGE_EXISTING:
            shard, path = self._load_existing_shard(pointer)
            entries = shard.get("entries")
            index = pointer.get("cache_entry_index")
            _require(
                isinstance(entries, list)
                and isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(entries)
                and isinstance(entries[index], Mapping),
                "spatial cache entry pointer drift",
            )
            entry = entries[index]
            expected_kind = "query" if kind == "query" else "gallery"
            intrinsic = entry.get("query_ordinal") if kind == "query" else entry.get("physical_row")
            _require(
                entry.get("kind") == expected_kind
                and intrinsic == native_key
                and entry.get("work_ordinal") == pointer.get("work_ordinal")
                and entry.get("entry_logical_sha256")
                == pointer.get("entry_logical_sha256")
                == self._existing_entry_logical(entry, kind=kind),
                "spatial cache entry binding/logical hash drift",
            )
            return entry, self._file_sha256(path)

        _require(kind == "reference", "supplemental query cache is forbidden")
        path = self._safe_cache_path(pointer.get("cache_path"))
        _require(path == self.payload_path, "supplemental pointer is not self-contained")
        payload = self._load_payload()
        entries = payload.get("supplemental_gallery_entries")
        matches = [
            item
            for item in entries
            if isinstance(item, Mapping) and item.get("physical_row") == native_key
        ] if isinstance(entries, list) else []
        _require(len(matches) == 1, "supplemental gallery entry is absent or duplicated")
        entry = matches[0]
        _require(
            entry.get("kind") == "gallery"
            and entry.get("entry_logical_sha256")
            == pointer.get("entry_logical_sha256")
            == _logical_sha256(entry, self_key="entry_logical_sha256"),
            "supplemental gallery entry logical hash drift",
        )
        return entry, str(self._payload_file_sha256)

    def _build_source(
        self,
        *,
        record: Mapping[str, Any],
        kind: str,
        native_key: int,
    ) -> ColNomicTokenSourceV1:
        source_sha = _sha(record.get("source_image_sha256"), name=f"{kind} source image")
        if kind == "query":
            query_id = record.get("query_id")
            _require(isinstance(query_id, str) and query_id, "query id absent")
            source_key = f"query:{native_key}:{query_id}"
        else:
            source_key = f"reference:{native_key}"
        _require(
            record.get("record_logical_sha256")
            == _logical_sha256(record, self_key="record_logical_sha256"),
            f"{kind} geometry record logical hash drift",
        )
        pointer = record.get("colnomic_cache")
        _require(isinstance(pointer, Mapping), "ColNomic cache pointer absent")
        entry, cache_file_sha = self._entry_from_pointer(
            pointer=pointer, record=record, kind=kind, native_key=native_key
        )
        tokens, grid = self._validate_tokens(entry, pointer)
        entry_source_path = entry.get("source_path")
        record_source_path = record.get("source_path")
        _require(
            isinstance(entry_source_path, str)
            and isinstance(record_source_path, str)
            and Path(entry_source_path).resolve() == Path(record_source_path).resolve(),
            f"{kind} source path binding drift",
        )
        if kind == "query":
            _require(
                entry.get("query_id") == record.get("query_id")
                and entry.get("source_image_tokens_sha256") == entry.get("tokens_sha256")
                and self._source_file_sha256(Path(record_source_path)) == source_sha,
                "query source/hash binding drift",
            )
        elif pointer.get("storage_kind") == STORAGE_EXISTING:
            _require(
                entry.get("source_image_file_sha256") == source_sha,
                "reference source hash binding drift",
            )
        else:
            _require(
                entry.get("source_image_sha256") == source_sha
                and self._source_file_sha256(Path(record_source_path)) == source_sha,
                "supplemental reference source/hash binding drift",
            )
        colnomic = _geometry_from_record(
            record.get("colnomic_geometry"),
            expected_source_sha256=source_sha,
            expected_source_key=source_key,
        )
        dino = _geometry_from_record(
            record.get("dino_geometry"),
            expected_source_sha256=source_sha,
            expected_source_key=source_key,
        )
        _require(colnomic.grid_shape == grid, "ColNomic token/geometry grid drift")
        source_payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "native_key": native_key,
            "source_key": source_key,
            "source_path": str(Path(record_source_path).resolve()),
            "source_image_sha256": source_sha,
            "cache_storage_kind": pointer["storage_kind"],
            "cache_file_sha256": cache_file_sha,
            "cache_entry_index": pointer.get("cache_entry_index"),
            "cache_entry_logical_sha256": pointer["entry_logical_sha256"],
            "tokens_sha256": pointer["tokens_sha256"],
            "grid_shape": list(grid),
            "colnomic_geometry_sha256": colnomic.sha256,
            "dino_geometry_sha256": dino.sha256,
        }
        return ColNomicTokenSourceV1(
            kind=kind,
            native_key=native_key,
            source_key=source_key,
            source_path=str(Path(record_source_path).resolve()),
            source_image_sha256=source_sha,
            cache_storage_kind=str(pointer["storage_kind"]),
            cache_file_sha256=cache_file_sha,
            cache_entry_index=pointer.get("cache_entry_index"),
            cache_entry_logical_sha256=str(pointer["entry_logical_sha256"]),
            tokens=tokens,
            tokens_sha256=str(pointer["tokens_sha256"]),
            grid_shape=grid,
            colnomic_geometry=colnomic,
            dino_geometry=dino,
            source_logical_sha256=_plain_canonical_sha256(source_payload),
        )

    def load_current8(self) -> Current8PSealSourcesV1:
        payload = self._load_payload()
        query_records = payload["query_records"]
        reference_records = payload["reference_records"]
        query_by_historical: dict[int, Mapping[str, Any]] = {}
        for record in query_records:
            _require(isinstance(record, Mapping), "query geometry record schema drift")
            historical = record.get("historical_query_ordinal")
            if historical in CURRENT8_HISTORICAL_ORDINALS:
                _require(historical not in query_by_historical, "current-eight query duplicated")
                query_by_historical[int(historical)] = record
        _require(
            set(query_by_historical) == set(CURRENT8_HISTORICAL_ORDINALS),
            "current-eight query population is incomplete",
        )
        reference_by_row: dict[int, Mapping[str, Any]] = {}
        for record in reference_records:
            _require(isinstance(record, Mapping), "reference geometry record schema drift")
            row = record.get("physical_row")
            _require(
                isinstance(row, int) and not isinstance(row, bool) and row >= 0,
                "reference physical row drift",
            )
            _require(row not in reference_by_row, "reference geometry row duplicated")
            reference_by_row[row] = record
        _require(
            len(reference_by_row) == EXPECTED_REFERENCE_COUNT,
            "reference geometry population drift",
        )

        output: list[Current8QuerySourceV1] = []
        for historical in CURRENT8_HISTORICAL_ORDINALS:
            record = query_by_historical[historical]
            axis = record.get("candidate_physical_rows")
            _require(
                isinstance(axis, list)
                and len(axis) == EXPECTED_CANDIDATE_COUNT
                and all(isinstance(item, int) and not isinstance(item, bool) for item in axis)
                and len(set(axis)) == EXPECTED_CANDIDATE_COUNT,
                "natural C128 candidate axis drift",
            )
            axis_sha = _sha(record.get("candidate_axis_sha256"), name="candidate axis")
            _require(
                _plain_canonical_sha256(axis) == axis_sha,
                "natural C128 candidate axis hash drift",
            )
            query_source = self._build_source(
                record=record, kind="query", native_key=historical
            )
            candidates: list[CandidateReferenceSourceV1] = []
            for position, physical_row in enumerate(axis):
                _require(
                    physical_row in reference_by_row,
                    "C128 candidate is absent from geometry reference union",
                )
                if physical_row not in self._reference_sources:
                    self._reference_sources[physical_row] = self._build_source(
                        record=reference_by_row[physical_row],
                        kind="reference",
                        native_key=physical_row,
                    )
                candidates.append(
                    CandidateReferenceSourceV1(
                        candidate_position=position,
                        physical_row=physical_row,
                        source=self._reference_sources[physical_row],
                    )
                )
            query_payload = {
                "schema_version": SCHEMA_VERSION,
                "historical_query_ordinal": historical,
                "execution_ordinal": int(record["execution_ordinal"]),
                "query_id": str(record["query_id"]),
                "candidate_axis_sha256": axis_sha,
                "query_source_logical_sha256": query_source.source_logical_sha256,
                "candidate_source_logical_sha256": [
                    item.source.source_logical_sha256 for item in candidates
                ],
            }
            output.append(
                Current8QuerySourceV1(
                    historical_query_ordinal=historical,
                    execution_ordinal=int(record["execution_ordinal"]),
                    query_id=str(record["query_id"]),
                    candidate_axis_sha256=axis_sha,
                    query=query_source,
                    candidates=tuple(candidates),
                    source_logical_sha256=_plain_canonical_sha256(query_payload),
                )
            )
        cohort_payload = {
            "schema_version": SCHEMA_VERSION,
            "geometry_payload_file_sha256": self._payload_file_sha256,
            "geometry_payload_logical_sha256": payload["logical_sha256"],
            "query_source_logical_sha256": [item.source_logical_sha256 for item in output],
            "unique_reference_count": len(self._reference_sources),
        }
        return Current8PSealSourcesV1(
            queries=tuple(output),
            geometry_payload_file_sha256=str(self._payload_file_sha256),
            geometry_payload_logical_sha256=_sha(
                payload.get("logical_sha256"), name="geometry payload logical"
            ),
            unique_reference_count=len(self._reference_sources),
            source_logical_sha256=_plain_canonical_sha256(cohort_payload),
        )


def load_current8_pseal_sources_v1(
    payload_path: Path,
    *,
    rc_root: Path | None = None,
    expected_payload_file_sha256: str = DEFAULT_GEOMETRY_PAYLOAD_SHA256,
) -> Current8PSealSourcesV1:
    """Convenience entry point for one complete, cached current-eight load."""

    return CW1MultiTilePSealSourceLoaderV1(
        payload_path=payload_path,
        rc_root=rc_root,
        expected_payload_file_sha256=expected_payload_file_sha256,
    ).load_current8()


__all__ = [
    "SCHEMA_VERSION",
    "FULL600_GEOMETRY_SCHEMA",
    "FULL600_GEOMETRY_STATUS",
    "DEFAULT_GEOMETRY_PAYLOAD_SHA256",
    "CURRENT8_HISTORICAL_ORDINALS",
    "EXPECTED_CANDIDATE_COUNT",
    "CW1PSealSourceError",
    "ColNomicTokenSourceV1",
    "CandidateReferenceSourceV1",
    "Current8QuerySourceV1",
    "Current8PSealSourcesV1",
    "CW1MultiTilePSealSourceLoaderV1",
    "load_current8_pseal_sources_v1",
    "sha256_file",
    "tensor_sha256",
]
