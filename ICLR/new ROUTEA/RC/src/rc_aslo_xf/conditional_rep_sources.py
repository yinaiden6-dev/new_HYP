"""Result-blind sources for the conditional-representation lineage.

This module is deliberately only a source/preflight boundary.  It neither
loads a trainable model nor reads any previous scientific result.  The two
public stages are kept separate:

``build_prejoin_sources``
    Replays the frozen fold-0 *optimization* roles from the sanitized query
    ledger and the already validated C0 natural-C128 receipts.  No query
    target identity or group is opened here.

``build_postjoin_sources``
    Opens the existing per-query loss records only after the prejoin records
    are frozen.  It marks natural-C128 hits/misses and assigns the hit records
    to deterministic identity-and-group-component-disjoint inner folds.

Gallery identities are intrinsic reference metadata.  They are rebuilt from
the immutable 5,413-row raw-image catalogue, checked against the legacy
ColNomic ``setids`` order, and then corrected with the existing frozen
identity-repair manifest.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from rc_aslo_xf.gallery_identity_repair import (
    CORRECTED_IDENTITY_COUNT,
    PHYSICAL_ROW_COUNT,
    SOURCE_CACHE_SHA256,
    build_identity_map,
)
from rc_aslo_xf.l0_natural_runtime import load_prejoin
from rc_aslo_xf.l0_targetfree_source import (
    GALLERY_CACHE,
    QUERY_LEDGER,
    fold_d1_checkpoint,
    load_query_ledger,
    sha256_file,
)


RC_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = RC_ROOT.parents[2]
RAW_GALLERY = (
    WORKSPACE
    / "dailymed"
    / "data"
    / "box_flat_20000_images"
    / "data"
    / "raw_images"
)
L0_CONTRACT = (
    RC_ROOT
    / "protocols"
    / "L0_NATURAL_HARDNEG_V_REPRESENTATION_QUALIFICATION_CONTRACT_V2_20260808.json"
)
C0_FOLD0_ROOT = (
    RC_ROOT
    / "results"
    / "l0_natural_hardneg_v_representation_v2_identityfix"
    / "c0"
    / "fold_0"
)
C0_FOLD0_VALIDATION = C0_FOLD0_ROOT / "validation_complete.json"
TARGET_JOIN_ROOT = (
    RC_ROOT
    / "results"
    / "l0_natural_hardneg_v_representation_v2_identityfix"
    / "target_join"
)
TARGET_JOIN_INDEX = TARGET_JOIN_ROOT / "query_target_join_index.json"

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
OUTER_FOLD = 0
INNER_FOLDS = 5
QUERY_COUNT = 987
OPTIMIZATION_ROLE_COUNT = 775
CONDITIONAL_HIT_COUNT = 751
CONDITIONAL_MISS_COUNT = 24
CANDIDATE_COUNT = 128
INNER_NAMESPACE = "RC_CONDITIONAL_REP_INNER_COMPONENT_FOLD_V1"

PREJOIN_VERSION = "rc_conditional_rep_fold0_prejoin_sources_v1"
POSTJOIN_VERSION = "rc_conditional_rep_fold0_postjoin_sources_v1"
TARGET_JOIN_INDEX_VERSION = (
    "l0_v2_identityfix_per_query_loss_and_postjoin_target_join_v1"
)
TARGET_JOIN_RECORD_VERSION = (
    "l0_v2_identityfix_per_query_loss_target_join_record_v1"
)


class ConditionalRepSourceError(RuntimeError):
    """A frozen source or role contract no longer matches the lineage."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _logical_sha(value: dict[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def _receipt(value: dict[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output["logical_sha256"] = canonical_sha256(output)
    return output


@dataclass(frozen=True)
class GallerySource:
    raw_paths: tuple[Path, ...]
    legacy_setids: tuple[str, ...]
    corrected_identities: tuple[str, ...]
    gallery_cache_sha256: str
    raw_path_sequence_sha256: str
    legacy_setid_sequence_sha256: str
    corrected_mapping_sha256: str
    repair_contract_sha256: str
    repair_manifest_sha256: str

    def identity_for_row(self, physical_row: int) -> str:
        if isinstance(physical_row, bool) or physical_row not in range(
            len(self.corrected_identities)
        ):
            raise ConditionalRepSourceError("gallery physical row is invalid")
        return self.corrected_identities[physical_row]


@dataclass(frozen=True)
class ConditionalPrejoinRecord:
    query_ordinal: int
    query_id: str
    track: str
    source_path: Path
    outer_heldout_fold: int
    c0_prejoin_logical_sha256: str
    candidate_physical_rows: tuple[int, ...]
    candidate_mapping_sha256: str
    record_logical_sha256: str


@dataclass(frozen=True)
class ConditionalPrejoinSources:
    records: tuple[ConditionalPrejoinRecord, ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class ConditionalJoinedRecord:
    prejoin: ConditionalPrejoinRecord
    target_identity: str
    group_id: str
    target_candidate_position: int | None
    inner_fold: int | None
    target_record_logical_sha256: str

    @property
    def target_present(self) -> bool:
        return self.target_candidate_position is not None


@dataclass(frozen=True)
class ConditionalPostjoinSources:
    records: tuple[ConditionalJoinedRecord, ...]
    eligible_records: tuple[ConditionalJoinedRecord, ...]
    miss_records: tuple[ConditionalJoinedRecord, ...]
    receipt: dict[str, Any]


def _raw_gallery_paths(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ConditionalRepSourceError("raw gallery root is missing")
    paths: list[Path] = []
    for directory, subdirectories, names in os.walk(root):
        subdirectories.sort()
        paths.extend(
            Path(directory) / name
            for name in sorted(names)
            if (Path(directory) / name).is_file()
            and Path(name).suffix.lower() in IMAGE_SUFFIXES
        )
    return tuple(paths)


def build_gallery_source(
    *,
    gallery_cache: Path = GALLERY_CACHE,
    raw_gallery_root: Path = RAW_GALLERY,
    verify_cache_file_sha256: bool = True,
) -> GallerySource:
    """Validate the physical-row catalogue without materializing its tokens."""

    if not gallery_cache.is_file():
        raise ConditionalRepSourceError("frozen ColNomic gallery cache is missing")
    cache_sha = sha256_file(gallery_cache) if verify_cache_file_sha256 else ""
    if verify_cache_file_sha256 and cache_sha != SOURCE_CACHE_SHA256:
        raise ConditionalRepSourceError("frozen ColNomic gallery hash drift")
    payload = torch.load(
        gallery_cache,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict) or set(payload) != {"passage_emb", "setids"}:
        raise ConditionalRepSourceError("frozen ColNomic gallery schema drift")
    references = payload["passage_emb"]
    legacy = tuple(map(str, payload["setids"]))
    if (
        not isinstance(references, list)
        or len(references) != PHYSICAL_ROW_COUNT
        or len(legacy) != PHYSICAL_ROW_COUNT
        or any(
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape[1] != 128
            or value.dtype != torch.float16
            for value in references
        )
    ):
        raise ConditionalRepSourceError("frozen ColNomic gallery tensor population drift")

    raw_paths = _raw_gallery_paths(raw_gallery_root)
    path_labels = tuple(path.stem.strip() for path in raw_paths)
    if len(raw_paths) != PHYSICAL_ROW_COUNT or path_labels != legacy:
        raise ConditionalRepSourceError(
            "raw gallery physical-row path order no longer matches legacy setids"
        )
    repaired = build_identity_map(legacy)
    if (
        len(repaired.labels) != PHYSICAL_ROW_COUNT
        or len(set(repaired.labels)) != CORRECTED_IDENTITY_COUNT
    ):
        raise ConditionalRepSourceError("corrected gallery identity population drift")
    relative_paths = [
        path.relative_to(raw_gallery_root).as_posix() for path in raw_paths
    ]
    return GallerySource(
        raw_paths=raw_paths,
        legacy_setids=legacy,
        corrected_identities=repaired.labels,
        gallery_cache_sha256=cache_sha or SOURCE_CACHE_SHA256,
        raw_path_sequence_sha256=canonical_sha256(relative_paths),
        legacy_setid_sequence_sha256=canonical_sha256(list(legacy)),
        corrected_mapping_sha256=repaired.corrected_row_identity_mapping_sha256,
        repair_contract_sha256=repaired.repair_contract_sha256,
        repair_manifest_sha256=repaired.manifest_sha256,
    )


def _load_c0_validation(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ConditionalRepSourceError("fold-0 C0 validation receipt is missing")
    value = json.loads(path.read_text())
    required_true_checks = {
        "coverage_source_segment_receipts_unchanged",
        "coverage_state_all_checks_passed",
        "immutable_pre_supplement_coverage_receipt",
        "raw_v_supplement_receipt_exact_when_required",
        "resolved_raw_v_can_expose_every_candidate_row",
        "target_free_no_label_join_or_identity_loader",
    }
    if (
        not isinstance(value, dict)
        or value.get("status")
        != "L0_V2_IDENTITYFIX_C0_FOLD_LOCAL_C128_READY"
        or value.get("fold") != OUTER_FOLD
        or value.get("range") != [0, QUERY_COUNT]
        or value.get("contract_sha256") != sha256_file(L0_CONTRACT)
        or value.get("target_join_performed") is not False
        or value.get("target_label_read") is not False
        or value.get("historical_identity_loader_calls") != 0
        or any(
            value.get(name) != 0
            for name in (
                "opened_runtime_read_count",
                "sealed_runtime_read_count",
                "a10_runtime_read_count",
                "home_files_modified",
            )
        )
        or set(value.get("checks", {})) != required_true_checks
        or not all(value["checks"].values())
        or value.get("logical_sha256") != _logical_sha(value)
    ):
        raise ConditionalRepSourceError("fold-0 C0 validation receipt drift")
    return value, sha256_file(path)


def _prejoin_record_payload(
    *,
    query_ordinal: int,
    query_id: str,
    track: str,
    source_path: Path,
    outer_heldout_fold: int,
    c0_logical_sha256: str,
    candidate_rows: Sequence[int],
    candidate_mapping_sha256: str,
) -> dict[str, Any]:
    return {
        "query_ordinal": query_ordinal,
        "query_id": query_id,
        "track": track,
        "source_path": str(source_path),
        "outer_heldout_fold": outer_heldout_fold,
        "c0_prejoin_logical_sha256": c0_logical_sha256,
        "candidate_physical_rows": list(candidate_rows),
        "candidate_mapping_sha256": candidate_mapping_sha256,
    }


def build_prejoin_sources(
    *,
    gallery: GallerySource | None = None,
    c0_root: Path = C0_FOLD0_ROOT,
    c0_validation_path: Path = C0_FOLD0_VALIDATION,
) -> ConditionalPrejoinSources:
    """Build the 775 result-blind fold-0 optimization C128 roles."""

    gallery = build_gallery_source() if gallery is None else gallery
    validation, validation_sha = _load_c0_validation(c0_validation_path)
    queries = load_query_ledger()
    ledger = json.loads(QUERY_LEDGER.read_text())
    checkpoint_path, checkpoint_sha = fold_d1_checkpoint(OUTER_FOLD)
    if (
        checkpoint_sha != validation.get("d1_checkpoint_sha256")
        or sha256_file(checkpoint_path) != checkpoint_sha
        or ledger.get("logical_sha256")
        != validation.get("source_query_ledger_logical_sha256")
    ):
        raise ConditionalRepSourceError("fold-0 source binding drift")
    prejoin_dir = c0_root / "prejoin"
    expected_files = [prejoin_dir / f"{ordinal:04d}.pt" for ordinal in range(QUERY_COUNT)]
    if not all(path.is_file() for path in expected_files):
        raise ConditionalRepSourceError("fold-0 C0 prejoin population is incomplete")

    optimization = [query for query in queries if query.heldout_fold != OUTER_FOLD]
    if len(optimization) != OPTIMIZATION_ROLE_COUNT:
        raise ConditionalRepSourceError("fold-0 optimization role population drift")
    records: list[ConditionalPrejoinRecord] = []
    record_payloads: list[dict[str, Any]] = []
    candidate_union: set[int] = set()
    for query in optimization:
        value = load_prejoin(
            prejoin_dir / f"{query.query_ordinal:04d}.pt",
            fold=OUTER_FOLD,
            ordinal=query.query_ordinal,
            contract_sha256=str(validation["contract_sha256"]),
            d1_checkpoint_sha256=checkpoint_sha,
            source_query_ledger_logical_sha256=str(ledger["logical_sha256"]),
        )
        if value["query_id"] != query.query_id:
            raise ConditionalRepSourceError("query ledger/C0 query binding drift")
        rows = tuple(int(item) for item in value["candidate_physical_rows_in_hash_permutation"])
        corrected = tuple(gallery.identity_for_row(row) for row in rows)
        if len(rows) != CANDIDATE_COUNT or len(set(corrected)) != CANDIDATE_COUNT:
            raise ConditionalRepSourceError("natural C128 corrected-identity uniqueness drift")
        candidate_union.update(rows)
        payload = _prejoin_record_payload(
            query_ordinal=query.query_ordinal,
            query_id=query.query_id,
            track=query.track,
            source_path=query.path,
            outer_heldout_fold=query.heldout_fold,
            c0_logical_sha256=str(value["logical_sha256"]),
            candidate_rows=rows,
            candidate_mapping_sha256=str(
                value["candidate_row_label_mapping_sha256_receipt_only"]
            ),
        )
        record_sha = canonical_sha256(payload)
        record_payloads.append({**payload, "record_logical_sha256": record_sha})
        records.append(
            ConditionalPrejoinRecord(
                query_ordinal=query.query_ordinal,
                query_id=query.query_id,
                track=query.track,
                source_path=query.path,
                outer_heldout_fold=query.heldout_fold,
                c0_prejoin_logical_sha256=str(value["logical_sha256"]),
                candidate_physical_rows=rows,
                candidate_mapping_sha256=str(
                    value["candidate_row_label_mapping_sha256_receipt_only"]
                ),
                record_logical_sha256=record_sha,
            )
        )
    if [record.query_ordinal for record in records] != sorted(
        record.query_ordinal for record in records
    ):
        raise ConditionalRepSourceError("optimization records are not canonical")
    track_counts = Counter(record.track for record in records)
    receipt = _receipt(
        {
            "version": PREJOIN_VERSION,
            "status": "RC_CONDITIONAL_REP_FOLD0_PREJOIN_READY",
            "claim_level": (
                "result_blind_source_and_role_preflight_only_no_label_join_"
                "training_target_determination_ownership_or_retrieval_claim"
            ),
            "data_role": "fold0_optimization_target_free_prejoin",
            "fold": OUTER_FOLD,
            "query_count": len(records),
            "outer_heldout_excluded_count": QUERY_COUNT - len(records),
            "candidate_count_per_query": CANDIDATE_COUNT,
            "track_counts": dict(sorted(track_counts.items())),
            "query_ordinal_sequence_sha256": canonical_sha256(
                [record.query_ordinal for record in records]
            ),
            "record_sequence_sha256": canonical_sha256(record_payloads),
            "candidate_union_row_count": len(candidate_union),
            "candidate_union_sha256": canonical_sha256(sorted(candidate_union)),
            "gallery_cache_sha256": gallery.gallery_cache_sha256,
            "gallery_raw_path_sequence_sha256": gallery.raw_path_sequence_sha256,
            "gallery_legacy_setid_sequence_sha256": gallery.legacy_setid_sequence_sha256,
            "gallery_corrected_mapping_sha256": gallery.corrected_mapping_sha256,
            "gallery_identity_repair_contract_sha256": gallery.repair_contract_sha256,
            "gallery_identity_repair_manifest_sha256": gallery.repair_manifest_sha256,
            "source_query_ledger_logical_sha256": str(ledger["logical_sha256"]),
            "c0_validation_sha256": validation_sha,
            "c0_validation_logical_sha256": str(validation["logical_sha256"]),
            "c0_prejoin_logical_sequence_sha256": canonical_sha256(
                [record.c0_prejoin_logical_sha256 for record in records]
            ),
            "target_join_performed": False,
            "target_label_read": False,
            "D1_score_rank_slot_winner_read": False,
            "opened_runtime_read_count": 0,
            "sealed_runtime_read_count": 0,
            "a10_runtime_read_count": 0,
            "home_files_modified": 0,
        }
    )
    return ConditionalPrejoinSources(records=tuple(records), receipt=receipt)


def _load_target_join_index(
    *, prejoin: ConditionalPrejoinSources, path: Path
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ConditionalRepSourceError("target-join index is missing")
    value = json.loads(path.read_text())
    records = value.get("records")
    expected_keys = {
        "version",
        "contract_sha256",
        "source_query_ledger_logical_sha256",
        "query_count",
        "records",
        "c0_validation_receipt_sha256_by_fold",
        "target_join_performed",
        "opened_runtime_read_count",
        "sealed_runtime_read_count",
        "a10_runtime_read_count",
        "home_files_modified",
        "logical_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("version") != TARGET_JOIN_INDEX_VERSION
        or value.get("contract_sha256") != sha256_file(L0_CONTRACT)
        or value.get("source_query_ledger_logical_sha256")
        != prejoin.receipt["source_query_ledger_logical_sha256"]
        or value.get("query_count") != QUERY_COUNT
        or not isinstance(records, list)
        or len(records) != QUERY_COUNT
        or value.get("c0_validation_receipt_sha256_by_fold", {}).get("0")
        != prejoin.receipt["c0_validation_sha256"]
        or value.get("target_join_performed") is not True
        or any(
            value.get(name) != 0
            for name in (
                "opened_runtime_read_count",
                "sealed_runtime_read_count",
                "a10_runtime_read_count",
                "home_files_modified",
            )
        )
        or value.get("logical_sha256") != _logical_sha(value)
    ):
        raise ConditionalRepSourceError("target-join index drift")
    for ordinal, item in enumerate(records):
        if (
            not isinstance(item, dict)
            or set(item)
            != {"query_ordinal", "query_id", "path", "record_logical_sha256"}
            or item.get("query_ordinal") != ordinal
            or item.get("path") != f"records/{ordinal:04d}.json"
            or not isinstance(item.get("query_id"), str)
            or not isinstance(item.get("record_logical_sha256"), str)
            or len(item["record_logical_sha256"]) != 64
        ):
            raise ConditionalRepSourceError("target-join index record drift")
    return value, sha256_file(path)


def _load_target_record(
    *,
    index_item: dict[str, Any],
    target_join_root: Path,
    contract_sha256: str,
    ledger_logical_sha256: str,
) -> dict[str, Any]:
    path = target_join_root / str(index_item["path"])
    if not path.is_file():
        raise ConditionalRepSourceError("target-join record is missing")
    value = json.loads(path.read_text())
    expected_keys = {
        "version",
        "contract_sha256",
        "source_query_ledger_logical_sha256",
        "query_ordinal",
        "query_id",
        "identity",
        "track",
        "group_id",
        "target_join_performed",
        "opened_runtime_read_count",
        "sealed_runtime_read_count",
        "a10_runtime_read_count",
        "home_files_modified",
        "logical_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("version") != TARGET_JOIN_RECORD_VERSION
        or value.get("contract_sha256") != contract_sha256
        or value.get("source_query_ledger_logical_sha256")
        != ledger_logical_sha256
        or value.get("query_ordinal") != index_item["query_ordinal"]
        or value.get("query_id") != index_item["query_id"]
        or not all(
            isinstance(value.get(name), str) and bool(value[name])
            for name in ("identity", "track", "group_id")
        )
        or value.get("target_join_performed") is not True
        or any(
            value.get(name) != 0
            for name in (
                "opened_runtime_read_count",
                "sealed_runtime_read_count",
                "a10_runtime_read_count",
                "home_files_modified",
            )
        )
        or value.get("logical_sha256") != index_item["record_logical_sha256"]
        or value.get("logical_sha256") != _logical_sha(value)
    ):
        raise ConditionalRepSourceError("target-join record drift")
    return value


def deterministic_inner_fold_assignments(
    records: Sequence[ConditionalJoinedRecord],
    *,
    folds: int = INNER_FOLDS,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Assign identity/group bipartite connected components to balanced folds."""

    if folds < 2 or not records or any(not record.target_present for record in records):
        raise ConditionalRepSourceError("inner-fold input population is invalid")
    by_ordinal = {record.prejoin.query_ordinal: record for record in records}
    if len(by_ordinal) != len(records):
        raise ConditionalRepSourceError("duplicate query ordinal in inner-fold input")
    parent = {ordinal: ordinal for ordinal in by_ordinal}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    identity_owner: dict[str, int] = {}
    group_owner: dict[str, int] = {}
    for record in sorted(records, key=lambda item: item.prejoin.query_ordinal):
        ordinal = record.prejoin.query_ordinal
        prior_identity = identity_owner.setdefault(record.target_identity, ordinal)
        prior_group = group_owner.setdefault(record.group_id, ordinal)
        union(ordinal, prior_identity)
        union(ordinal, prior_group)

    components: dict[int, list[ConditionalJoinedRecord]] = defaultdict(list)
    for record in records:
        components[find(record.prejoin.query_ordinal)].append(record)
    component_rows: list[dict[str, Any]] = []
    for values in components.values():
        values.sort(key=lambda item: item.prejoin.query_ordinal)
        payload = {
            "query_ordinals": [item.prejoin.query_ordinal for item in values],
            "identities": sorted({item.target_identity for item in values}),
            "groups": sorted({item.group_id for item in values}),
        }
        component_rows.append(
            {
                **payload,
                "query_count": len(values),
                "fingerprint": canonical_sha256(payload),
            }
        )
    component_rows.sort(key=lambda item: (-item["query_count"], item["fingerprint"]))

    fold_load = [0 for _ in range(folds)]
    fold_component_count = [0 for _ in range(folds)]
    assignments: dict[int, int] = {}
    assigned_components: list[dict[str, Any]] = []
    for component in component_rows:
        fold = min(
            range(folds),
            key=lambda candidate: (
                fold_load[candidate],
                fold_component_count[candidate],
                canonical_sha256(
                    [INNER_NAMESPACE, component["fingerprint"], candidate]
                ),
            ),
        )
        for ordinal in component["query_ordinals"]:
            assignments[int(ordinal)] = fold
        fold_load[fold] += int(component["query_count"])
        fold_component_count[fold] += 1
        assigned_components.append({**component, "inner_fold": fold})

    identity_folds: dict[str, set[int]] = defaultdict(set)
    group_folds: dict[str, set[int]] = defaultdict(set)
    for ordinal, fold in assignments.items():
        identity_folds[by_ordinal[ordinal].target_identity].add(fold)
        group_folds[by_ordinal[ordinal].group_id].add(fold)
    if (
        len(assignments) != len(records)
        or any(len(values) != 1 for values in identity_folds.values())
        or any(len(values) != 1 for values in group_folds.values())
        or any(load == 0 for load in fold_load)
    ):
        raise ConditionalRepSourceError("identity/group-disjoint inner-fold assignment failed")
    receipt = {
        "namespace": INNER_NAMESPACE,
        "fold_count": folds,
        "component_count": len(component_rows),
        "query_count_by_fold": {
            str(fold): fold_load[fold] for fold in range(folds)
        },
        "component_count_by_fold": {
            str(fold): fold_component_count[fold] for fold in range(folds)
        },
        "identity_count": len(identity_folds),
        "group_count": len(group_folds),
        "identity_disjoint": True,
        "group_disjoint": True,
        "component_sequence_sha256": canonical_sha256(assigned_components),
        "assignment_sha256": canonical_sha256(
            [[ordinal, assignments[ordinal]] for ordinal in sorted(assignments)]
        ),
    }
    return assignments, receipt


def build_postjoin_sources(
    *,
    prejoin: ConditionalPrejoinSources,
    gallery: GallerySource,
    target_join_index_path: Path = TARGET_JOIN_INDEX,
    target_join_root: Path = TARGET_JOIN_ROOT,
) -> ConditionalPostjoinSources:
    """Join training labels, freeze conditional hits, and build inner roles."""

    if (
        prejoin.receipt.get("version") != PREJOIN_VERSION
        or prejoin.receipt.get("target_join_performed") is not False
        or prejoin.receipt.get("logical_sha256") != _logical_sha(prejoin.receipt)
        or len(prejoin.records) != OPTIMIZATION_ROLE_COUNT
    ):
        raise ConditionalRepSourceError("conditional prejoin source is not frozen")
    index, index_sha = _load_target_join_index(
        prejoin=prejoin, path=target_join_index_path
    )
    index_by_ordinal = {
        int(item["query_ordinal"]): item for item in index["records"]
    }
    joined: list[ConditionalJoinedRecord] = []
    for source in prejoin.records:
        item = index_by_ordinal[source.query_ordinal]
        if item["query_id"] != source.query_id:
            raise ConditionalRepSourceError("prejoin/target-join query mismatch")
        target = _load_target_record(
            index_item=item,
            target_join_root=target_join_root,
            contract_sha256=str(index["contract_sha256"]),
            ledger_logical_sha256=str(index["source_query_ledger_logical_sha256"]),
        )
        if target["track"] != source.track:
            raise ConditionalRepSourceError("prejoin/target-join track mismatch")
        candidate_identities = tuple(
            gallery.identity_for_row(row) for row in source.candidate_physical_rows
        )
        target_identity = str(target["identity"])
        positions = [
            position
            for position, identity in enumerate(candidate_identities)
            if identity == target_identity
        ]
        if len(positions) > 1:
            raise ConditionalRepSourceError("target appears multiple times in corrected C128")
        joined.append(
            ConditionalJoinedRecord(
                prejoin=source,
                target_identity=target_identity,
                group_id=str(target["group_id"]),
                target_candidate_position=positions[0] if positions else None,
                inner_fold=None,
                target_record_logical_sha256=str(target["logical_sha256"]),
            )
        )
    hits = [record for record in joined if record.target_present]
    misses = [record for record in joined if not record.target_present]
    if len(hits) != CONDITIONAL_HIT_COUNT or len(misses) != CONDITIONAL_MISS_COUNT:
        raise ConditionalRepSourceError("fold-0 conditional hit/miss population drift")
    assignments, inner_receipt = deterministic_inner_fold_assignments(hits)
    joined = [
        replace(
            record,
            inner_fold=assignments[record.prejoin.query_ordinal]
            if record.target_present
            else None,
        )
        for record in joined
    ]
    hits = [record for record in joined if record.target_present]
    misses = [record for record in joined if not record.target_present]
    track_counts = Counter(record.prejoin.track for record in hits)
    identity_count = len({record.target_identity for record in hits})
    group_count = len({record.group_id for record in hits})
    if (
        identity_count != 62
        or group_count != 61
        or dict(track_counts)
        != {"difficult": 104, "new_difficult_train": 24, "outcome": 623}
    ):
        raise ConditionalRepSourceError("conditional eligible coverage drift")
    postjoin_rows = [
        {
            "query_ordinal": record.prejoin.query_ordinal,
            "query_id": record.prejoin.query_id,
            "target_present": record.target_present,
            "target_candidate_position": record.target_candidate_position,
            "inner_fold": record.inner_fold,
            "target_record_logical_sha256": record.target_record_logical_sha256,
        }
        for record in joined
    ]
    receipt = _receipt(
        {
            "version": POSTJOIN_VERSION,
            "status": "RC_CONDITIONAL_REP_FOLD0_POSTJOIN_INNER_ROLES_READY",
            "claim_level": (
                "postjoin_role_and_coverage_preflight_only_no_training_target_"
                "determination_ownership_or_retrieval_claim"
            ),
            "data_role": "fold0_optimization_postjoin_inner_crossfit",
            "fold": OUTER_FOLD,
            "prejoin_logical_sha256": prejoin.receipt["logical_sha256"],
            "target_join_index_sha256": index_sha,
            "target_join_index_logical_sha256": index["logical_sha256"],
            "optimization_role_count": len(joined),
            "conditional_hit_count": len(hits),
            "conditional_miss_count": len(misses),
            "eligible_identity_count": identity_count,
            "eligible_group_count": group_count,
            "eligible_track_counts": dict(sorted(track_counts.items())),
            "hit_query_ordinal_sequence_sha256": canonical_sha256(
                [record.prejoin.query_ordinal for record in hits]
            ),
            "miss_query_ordinal_sequence_sha256": canonical_sha256(
                [record.prejoin.query_ordinal for record in misses]
            ),
            "postjoin_record_sequence_sha256": canonical_sha256(postjoin_rows),
            "inner_folds": inner_receipt,
            "target_join_performed": True,
            "target_join_record_read_count": len(joined),
            "D1_score_rank_slot_winner_read": False,
            "opened_runtime_read_count": 0,
            "sealed_runtime_read_count": 0,
            "a10_runtime_read_count": 0,
            "home_files_modified": 0,
        }
    )
    return ConditionalPostjoinSources(
        records=tuple(joined),
        eligible_records=tuple(hits),
        miss_records=tuple(misses),
        receipt=receipt,
    )


def summarize_inner_fold_coverage(
    records: Iterable[ConditionalJoinedRecord],
) -> dict[str, dict[str, Any]]:
    """Return auditable per-fold population statistics for eligible records."""

    by_fold: dict[int, list[ConditionalJoinedRecord]] = defaultdict(list)
    for record in records:
        if not record.target_present or record.inner_fold not in range(INNER_FOLDS):
            raise ConditionalRepSourceError("inner-fold coverage received an ineligible record")
        by_fold[int(record.inner_fold)].append(record)
    if set(by_fold) != set(range(INNER_FOLDS)):
        raise ConditionalRepSourceError("inner-fold coverage is incomplete")
    return {
        str(fold): {
            "queries": len(values),
            "identities": len({value.target_identity for value in values}),
            "groups": len({value.group_id for value in values}),
            "tracks": dict(
                sorted(Counter(value.prejoin.track for value in values).items())
            ),
        }
        for fold, values in sorted(by_fold.items())
    }
