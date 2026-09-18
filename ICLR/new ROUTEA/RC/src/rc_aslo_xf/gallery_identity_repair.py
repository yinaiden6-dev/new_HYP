"""Immutable correction of generic filename collisions in the gallery.

The source ColNomic cache is preserved byte-for-byte.  Its legacy ``setids``
were generated from filename stems, which incorrectly merged unrelated files
such as ``animal/Carton.jpg`` and ``otc/Carton.jpg``.  This module supplies a
pre-registered physical-row-to-identity mapping for candidate reduction only.
It deliberately has no query interface and cannot perform a target join.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Sequence


RC_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = RC_ROOT / "registry" / "gallery_identity_repair_v1.json"
CONTRACT = RC_ROOT / "protocols" / "L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"

SOURCE_CACHE_SHA256 = "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc"
SOURCE_SETIDS_SHA256 = "59305a7b787fc1b61277c03fb99154c9edf5cbaec5a84366432b26cfae47dc80"
PHYSICAL_ROW_COUNT = 5413
CORRECTED_IDENTITY_COUNT = 5412


class GalleryIdentityRepairError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class GalleryIdentityMap:
    labels: tuple[str, ...]
    legacy_row_label_mapping_sha256: str
    corrected_row_identity_mapping_sha256: str
    repair_contract_sha256: str
    manifest_sha256: str


def _read_contract() -> dict[str, object]:
    value = json.loads(CONTRACT.read_text()) if CONTRACT.is_file() else {}
    expected = {
        "version",
        "status",
        "base_l0_contract",
        "source_gallery_cache_sha256",
        "source_legacy_setids_sha256",
        "identity_manifest",
        "identity_manifest_sha256",
        "physical_row_count",
        "legacy_filename_label_count",
        "corrected_identity_count",
        "required_rules",
        "forbidden",
        "claim_level",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("version") != "l0_c0_gallery_identity_repair_contract_v1_20260808"
        or value.get("status") != "FROZEN_BEFORE_L0_V2_IDENTITYFIX_C0"
        or value.get("source_gallery_cache_sha256") != SOURCE_CACHE_SHA256
        or value.get("source_legacy_setids_sha256") != SOURCE_SETIDS_SHA256
        or value.get("physical_row_count") != PHYSICAL_ROW_COUNT
        or value.get("legacy_filename_label_count") != 5404
        or value.get("corrected_identity_count") != CORRECTED_IDENTITY_COUNT
        or value.get("identity_manifest") != "../registry/gallery_identity_repair_v1.json"
        or not isinstance(value.get("identity_manifest_sha256"), str)
        or len(str(value["identity_manifest_sha256"])) != 64
    ):
        raise GalleryIdentityRepairError("gallery identity-repair contract drift")
    return value


def repair_contract_sha256() -> str:
    _read_contract()
    return sha256_file(CONTRACT)


def build_identity_map(legacy_labels: Sequence[str]) -> GalleryIdentityMap:
    """Return the corrected candidate identity for each physical gallery row."""

    contract = _read_contract()
    if len(legacy_labels) != PHYSICAL_ROW_COUNT:
        raise GalleryIdentityRepairError("gallery physical-row population drift")
    legacy = tuple(map(str, legacy_labels))
    if canonical_sha256(list(legacy)) != SOURCE_SETIDS_SHA256:
        raise GalleryIdentityRepairError("gallery legacy setid ordering drift")
    manifest = json.loads(REGISTRY.read_text()) if REGISTRY.is_file() else {}
    expected = {
        "version",
        "status",
        "purpose",
        "source_gallery_cache_sha256",
        "source_legacy_setids_sha256",
        "physical_row_count",
        "legacy_filename_label_count",
        "corrected_identity_count",
        "verified_byte_identical_duplicate_components",
        "filename_collision_overrides",
        "inference_rule",
    }
    manifest_sha = sha256_file(REGISTRY) if REGISTRY.is_file() else ""
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected
        or manifest.get("version") != "routea_gallery_identity_repair_v1_20260808"
        or manifest.get("status") != "FROZEN_PENDING_CONTRACT_HASH"
        or manifest.get("source_gallery_cache_sha256") != SOURCE_CACHE_SHA256
        or manifest.get("source_legacy_setids_sha256") != SOURCE_SETIDS_SHA256
        or manifest.get("physical_row_count") != PHYSICAL_ROW_COUNT
        or manifest.get("legacy_filename_label_count") != 5404
        or manifest.get("corrected_identity_count") != CORRECTED_IDENTITY_COUNT
        or manifest_sha != contract["identity_manifest_sha256"]
    ):
        raise GalleryIdentityRepairError("gallery identity-repair manifest drift")

    overrides = manifest.get("filename_collision_overrides")
    if not isinstance(overrides, list) or len(overrides) != 15:
        raise GalleryIdentityRepairError("gallery collision override population drift")
    corrected = list(legacy)
    seen_rows: set[int] = set()
    for item in overrides:
        if (
            not isinstance(item, dict)
            or set(item) != {"physical_row", "legacy_label", "corrected_identity"}
            or not isinstance(item.get("physical_row"), int)
            or not isinstance(item.get("legacy_label"), str)
            or not isinstance(item.get("corrected_identity"), str)
        ):
            raise GalleryIdentityRepairError("gallery collision override schema drift")
        row = int(item["physical_row"])
        if row < 0 or row >= PHYSICAL_ROW_COUNT or row in seen_rows or legacy[row] != item["legacy_label"]:
            raise GalleryIdentityRepairError("gallery collision override binding drift")
        seen_rows.add(row)
        corrected[row] = str(item["corrected_identity"])

    duplicate_components = manifest.get("verified_byte_identical_duplicate_components")
    if duplicate_components != [{
        "canonical_identity": "Biogen_21",
        "physical_rows": [714, 715],
        "relative_paths": ["homeopathic/Biogen_21 .jpg", "homeopathic/Biogen_21.jpg"],
        "sha256": "3972cbccd14da1942e6d64dfeb460322a8ad3e766f6dedb58a233f45ffca77ea",
    }]:
        raise GalleryIdentityRepairError("verified duplicate component contract drift")
    if corrected[714] != corrected[715] or corrected[714] != "Biogen_21":
        raise GalleryIdentityRepairError("byte-identical Biogen component no longer collapses")
    if len(set(corrected)) != CORRECTED_IDENTITY_COUNT:
        raise GalleryIdentityRepairError("corrected gallery identity population drift")
    by_identity: dict[str, list[int]] = defaultdict(list)
    for row, label in enumerate(corrected):
        by_identity[label].append(row)
    remaining_duplicates = {
        label: rows for label, rows in by_identity.items() if len(rows) > 1
    }
    if remaining_duplicates != {"Biogen_21": [714, 715]}:
        raise GalleryIdentityRepairError("unregistered gallery identity collision remains")
    return GalleryIdentityMap(
        labels=tuple(corrected),
        legacy_row_label_mapping_sha256=canonical_sha256(list(enumerate(legacy))),
        corrected_row_identity_mapping_sha256=canonical_sha256(list(enumerate(corrected))),
        repair_contract_sha256=repair_contract_sha256(),
        manifest_sha256=manifest_sha,
    )
