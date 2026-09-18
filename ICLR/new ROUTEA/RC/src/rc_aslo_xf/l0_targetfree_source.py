"""Sanitized source interfaces for L0-V2 target-free forward paths.

This module deliberately does *not* call the historical identity-bearing
``load_query_rows``, ``load_query_tokens`` or ``load_gallery`` helpers.  The
separate source-ledger compiler publishes only query paths, token paths,
grids, file hashes and an already-fixed outer-fold role.  C0 and target-free
heldout scoring can therefore neither inspect a query's target identity nor
reconstruct it from a runner ``QueryRow`` object.

Gallery exact labels remain available: they are intrinsic reference metadata
needed to reduce duplicate physical rows to one reference per exact gallery
label, and are never joined to a query target in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[2]
ROUTEA = RC_ROOT.parent
for search_path in (RC_ROOT / "programs", ROUTEA / "route_a_core"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_a0_fold_d1 as d1_runtime  # noqa: E402
from route_a import o1_c6direct_m1_runtime as m1_runtime  # noqa: E402
from rc_aslo_xf.gallery_identity_repair import build_identity_map  # noqa: E402


QUERY_LEDGER_ROOT = RC_ROOT / "cache" / "l0_natural_hardneg_v2_targetfree_inputs_v1"
QUERY_LEDGER = QUERY_LEDGER_ROOT / "query_ledger.json"
REDACTED_QUERY_TOKEN_CACHE_ROOT = RC_ROOT / "cache" / "l0_natural_hardneg_v2_redacted_query_tokens_v1"
GALLERY_CACHE = ROUTEA.parents[1] / "colnomic" / "difficult" / "raw_gallery_7b" / "cache" / "colnomic_gallery_emb_difficult.pt"
EXPECTED_GALLERY_SHA256 = "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc"
EXPECTED_GALLERY_SETIDS_SHA256 = "59305a7b787fc1b61277c03fb99154c9edf5cbaec5a84366432b26cfae47dc80"
D1_ROOT = RC_ROOT / "results" / "a0_optimization987_fivefold_oof_v1"
QUERY_LEDGER_VERSION = "l0_natural_hardneg_v2_targetfree_query_ledger_v1"


class L0TargetFreeSourceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TargetFreeQuerySpec:
    query_ordinal: int
    query_id: str
    track: str
    manifest_index: int
    path: Path
    shard: Path
    grid_h: int
    grid_w: int
    source_image_sha256: str
    source_shard_sha256: str
    heldout_fold: int
    oof_heldout_ordinal: int


@dataclass(frozen=True)
class TargetFreeQueryTokens:
    image_tokens: torch.Tensor
    template_tokens: torch.Tensor
    grid_h: int
    grid_w: int
    shard_sha256: str
    image_sha256: str


@dataclass(frozen=True)
class TargetFreeGallery:
    labels: tuple[str, ...]
    label_values: tuple[str, ...]
    reference_tokens_cpu: torch.Tensor
    reference_mask_cpu: torch.Tensor
    row_label_mapping_sha256: str
    legacy_row_label_mapping_sha256: str
    identity_repair_contract_sha256: str
    identity_repair_manifest_sha256: str


def _assert_no_target_schema(value: Mapping[str, Any]) -> None:
    forbidden = {"identity", "reference_label", "target", "target_label", "true_identity", "group_id", "supergroup"}
    keys = set(value)
    if keys.intersection(forbidden):
        raise L0TargetFreeSourceError("target-bearing field appeared in L0 target-free source ledger")


def load_query_ledger(path: Path = QUERY_LEDGER) -> list[TargetFreeQuerySpec]:
    if not path.is_file():
        raise L0TargetFreeSourceError("L0 V2 target-free query ledger is missing")
    value = json.loads(path.read_text())
    records = value.get("queries")
    expected = {
        "version",
        "query_count",
        "queries",
        "query_source_sha256",
        "target_label_read",
        "opened_runtime_read_count",
        "sealed_runtime_read_count",
        "home_files_modified",
        "logical_sha256",
    }
    if (
        set(value) != expected
        or value.get("version") != QUERY_LEDGER_VERSION
        or value.get("query_count") != 987
        or value.get("target_label_read") is not False
        or value.get("opened_runtime_read_count") != 0
        or value.get("sealed_runtime_read_count") != 0
        or value.get("home_files_modified") != 0
        or not isinstance(records, list)
        or len(records) != 987
        or value.get("logical_sha256")
        != canonical_sha256({key: value[key] for key in value if key != "logical_sha256"})
    ):
        raise L0TargetFreeSourceError("L0 V2 target-free query ledger receipt drift")
    output: list[TargetFreeQuerySpec] = []
    expected_keys = {
        "query_ordinal",
        "query_id",
        "track",
        "manifest_index",
        "path",
        "shard",
        "grid_h",
        "grid_w",
        "source_image_sha256",
        "source_shard_sha256",
        "heldout_fold",
        "oof_heldout_ordinal",
    }
    for ordinal, item in enumerate(records):
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise L0TargetFreeSourceError("L0 V2 target-free query ledger schema drift")
        _assert_no_target_schema(item)
        if (
            int(item["query_ordinal"]) != ordinal
            or not isinstance(item["query_id"], str)
            or item["track"] not in {"outcome", "difficult", "new_difficult_train"}
            or not isinstance(item["manifest_index"], int)
            or int(item["grid_h"]) < 1
            or int(item["grid_w"]) < 1
            or int(item["heldout_fold"]) not in range(5)
            or int(item["oof_heldout_ordinal"]) < 0
            or any(not isinstance(item[name], str) or len(item[name]) != 64 for name in ("source_image_sha256", "source_shard_sha256"))
        ):
            raise L0TargetFreeSourceError("L0 V2 target-free query ledger value drift")
        output.append(
            TargetFreeQuerySpec(
                query_ordinal=ordinal,
                query_id=str(item["query_id"]),
                track=str(item["track"]),
                manifest_index=int(item["manifest_index"]),
                path=Path(str(item["path"])),
                shard=Path(str(item["shard"])),
                grid_h=int(item["grid_h"]),
                grid_w=int(item["grid_w"]),
                source_image_sha256=str(item["source_image_sha256"]),
                source_shard_sha256=str(item["source_shard_sha256"]),
                heldout_fold=int(item["heldout_fold"]),
                oof_heldout_ordinal=int(item["oof_heldout_ordinal"]),
            )
        )
    if (
        [item.query_id for item in output] != sorted(item.query_id for item in output)
        or len({item.query_id for item in output}) != 987
        or {fold: sum(item.heldout_fold == fold for item in output) for fold in range(5)}
        != {0: 212, 1: 205, 2: 189, 3: 188, 4: 193}
    ):
        raise L0TargetFreeSourceError("L0 V2 target-free query role population drift")
    return output


def load_query_tokens(
    spec: TargetFreeQuerySpec,
    *,
    cache_root: Path | None = None,
) -> TargetFreeQueryTokens:
    """Load a query exclusively from the finalized redacted tensor cache.

    This is the target-free runtime boundary.  It never opens, hashes or
    deserializes a legacy query shard: legacy shards are accessible only to the
    separate one-way redaction materializer.  The cache index binds every
    artifact to the immutable sanitized ledger and its tensor hashes.
    """

    from rc_aslo_xf.l0_redacted_query_cache import (  # local import avoids a dataclass cycle
        L0RedactedQueryCacheError,
        load_redacted_query_artifact,
    )

    root = REDACTED_QUERY_TOKEN_CACHE_ROOT if cache_root is None else cache_root
    ledger = json.loads(QUERY_LEDGER.read_text())
    try:
        value = load_redacted_query_artifact(
            spec,
            specs=load_query_ledger(),
            cache_root=root,
            query_ledger_logical_sha256=str(ledger["logical_sha256"]),
            query_ledger_sha256=sha256_file(QUERY_LEDGER),
        )
    except L0RedactedQueryCacheError as exc:
        raise L0TargetFreeSourceError(str(exc)) from exc
    return TargetFreeQueryTokens(
        image_tokens=value["image_tokens"].detach().contiguous(),
        template_tokens=value["template_tokens"].detach().contiguous(),
        grid_h=spec.grid_h,
        grid_w=spec.grid_w,
        shard_sha256=spec.source_shard_sha256,
        image_sha256=spec.source_image_sha256,
    )


def load_target_free_gallery() -> TargetFreeGallery:
    """Load gallery reference metadata only; no optimization identity routing."""

    if not GALLERY_CACHE.is_file() or sha256_file(GALLERY_CACHE) != EXPECTED_GALLERY_SHA256:
        raise L0TargetFreeSourceError("L0 V2 frozen gallery file hash drift")
    payload = torch.load(GALLERY_CACHE, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise L0TargetFreeSourceError("L0 V2 gallery cache schema drift")
    legacy_labels = tuple(map(str, payload.get("setids", [])))
    references = list(payload.get("passage_emb", []))
    if (
        len(legacy_labels) != 5413
        or len(references) != 5413
        or len(set(legacy_labels)) != 5404
        or canonical_sha256(list(legacy_labels)) != EXPECTED_GALLERY_SETIDS_SHA256
    ):
        raise L0TargetFreeSourceError("L0 V2 gallery population or exact-label order drift")
    try:
        identity_map = build_identity_map(legacy_labels)
    except Exception as exc:
        raise L0TargetFreeSourceError("L0 gallery identity-repair binding drift") from exc
    labels = identity_map.labels
    try:
        padded, mask = m1_runtime.pad_reference_tokens(
            [reference.detach().to(torch.float16) for reference in references]
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise L0TargetFreeSourceError("L0 V2 gallery reference tensor schema drift") from exc
    return TargetFreeGallery(
        labels=labels,
        label_values=tuple(sorted(set(labels))),
        reference_tokens_cpu=padded,
        reference_mask_cpu=mask,
        row_label_mapping_sha256=identity_map.corrected_row_identity_mapping_sha256,
        legacy_row_label_mapping_sha256=identity_map.legacy_row_label_mapping_sha256,
        identity_repair_contract_sha256=identity_map.repair_contract_sha256,
        identity_repair_manifest_sha256=identity_map.manifest_sha256,
    )


def load_fold_local_d1_adapter(fold: int, *, device: torch.device) -> tuple[torch.nn.Module, str]:
    """Load only the assigned D1 checkpoint; no preflight/query role reads."""

    if fold not in range(5):
        raise L0TargetFreeSourceError("L0 V2 fold must be in 0..4")
    checkpoint_path, checkpoint_sha = fold_d1_checkpoint(fold)
    module = d1_runtime.import_runner()
    d1_runtime.configure_fold(module, fold)
    adapter = module.d1.build_fresh_d1_adapter().to(device).eval()
    adapter.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    return adapter, checkpoint_sha


def fold_d1_checkpoint(fold: int) -> tuple[Path, str]:
    """Return the frozen assigned D1 checkpoint without loading a query role."""

    if fold not in range(5):
        raise L0TargetFreeSourceError("L0 V2 fold must be in 0..4")
    result_path = D1_ROOT / f"fold_{fold}" / "result.json"
    checkpoint_path = D1_ROOT / f"fold_{fold}" / "d1_checkpoint.pt"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else ""
    if (
        result.get("status") != "A0_D1_FOLD_TRAINING_COMPLETE"
        or not checkpoint_path.is_file()
        or result.get("d1_checkpoint_sha256") != checkpoint_sha
    ):
        raise L0TargetFreeSourceError("L0 V2 fold-local D1 checkpoint receipt drift")
    return checkpoint_path, checkpoint_sha


def full_gallery_scores(
    adapter: torch.nn.Module,
    query: TargetFreeQueryTokens,
    gallery: TargetFreeGallery,
    *,
    device: torch.device,
    references: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact deployed D1 full-gallery score, with no query target input."""

    references = gallery.reference_tokens_cpu.float().to(device) if references is None else references
    reference_mask = gallery.reference_mask_cpu.to(device) if reference_mask is None else reference_mask
    with torch.inference_mode():
        image = adapter(query.image_tokens.float().to(device), query.grid_h, query.grid_w).float()
        return m1_runtime.chunked_sum_maxsim(
            torch.cat((image, query.template_tokens.float().to(device))),
            references,
            reference_mask,
            row_chunk_size=32,
        ).cpu()
