"""Target-free source and feature runtime for the natural S8 gate.

The runtime has two deliberately separate stages.

``materialize_target_free_ap_context``
    Loads one of the 987 identity-fixed fold-0 C128 prejoin records, obtains
    query image tokens exclusively from the validated redacted cache, resolves
    all 128 reference-local token grids, computes the frozen N0/O32 all-patch
    (AP) scores, and seals their hashes.  It cannot receive a target label.

``join_supervised_training_episode``
    May be called only on a valid sealed AP context.  It joins the natural
    exact identity, refuses target insertion, and freezes the AP-strongest
    non-target rival.  The same physical pair is then reused by REAL and both
    destruction controls.

Reference grids first reuse the independently validated 4,976-row
``conditional_colnomic_p_spatial_v1`` cache.  A physical row outside that
historical union is recovered without an encoder forward: the frozen CPU
processor reconstructs its image-token mask/grid and slices the already-frozen
5,413-row ColNomic embedding.  Recovered rows are cached for the process.

The public feature scorer consumes only a sealed target-free context and an
intrinsic physical row.  It has no label, rank, slot, winner, D1, mask, or
endpoint argument.  It materializes all-region BASE, V_C_BIND (P fixed) and
P_SPATIAL ledgers through the new S8 all-region core.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .colnomic_proposal_tokens import (
    ColNomicLocalGrid,
    proposal_grid_sha256,
    split_cached_spatial_tokens,
)
from .conditional_colnomic_ms_proposal import (
    SpatialTokenEntry,
    ValidatedSpatialCacheResolver,
    gallery_work_ordinal_by_physical_row,
)
from .conditional_rep_sources import (
    C0_FOLD0_ROOT,
    C0_FOLD0_VALIDATION,
    GallerySource,
    _load_c0_validation,
    build_gallery_source,
    build_prejoin_sources,
    canonical_sha256,
)
from .cw1_sr0_natural_s2_controls_v1 import (
    deterministic_derangement,
    resample_reference_content,
    role_preserving_query_derangement,
)
from .cw1_sr0_natural_superregion_s8_v1 import (
    CONTROL_BASE,
    CONTROL_V_C_BIND,
    NaturalAllRegionFeatureLedgerS8V1,
    materialize_all_natural_superregion_feature_ledger_s8_v1,
    verify_all_natural_superregions_s8_v1,
)
from .cw1_sr0_natural_superregion_v2 import (
    NaturalSuperregionPreparationV2,
    prepare_natural_superregion_candidate_v2,
)
from .l0_natural_runtime import load_prejoin
from .l0_targetfree_source import (
    GALLERY_CACHE,
    QUERY_LEDGER,
    TargetFreeQuerySpec,
    fold_d1_checkpoint,
    load_query_ledger,
    load_query_tokens,
    sha256_file,
)


SCHEMA_VERSION = "rc_cw1_sr0_s8_feature_runtime_v1"
AP_TEMPERATURE = 0.07
AP_FORMULA = (
    "mean_q[T*(logsumexp_r(cosine(q,r)/T)-log(R))],T=0.07;"
    "query=image_spatial_tokens_only;reference=all_spatial_image_tokens"
)
CANDIDATE_COUNT = 128
PHYSICAL_GALLERY_ROWS = 5413
CORRECTED_EXACT_IDENTITIES = 5412
LOCAL_DIMENSION = 128
LEGACY_SPATIAL_UNION_COUNT = 4976

RC_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = RC_ROOT.parents[2]
SPATIAL_CACHE_ROOT = RC_ROOT / "cache" / "conditional_colnomic_p_spatial_v1"
MODEL_DIR = (
    WORKSPACE / "models" / "downloaded_models" / "colnomic-embed-multimodal-7b"
)
PROCESSOR_CONFIG_FILES = (
    "adapter_config.json",
    "added_tokens.json",
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

C_BIND_NAMESPACE = "RC_CW1_SR0_S8_FEATURE_RUNTIME_V1_C_BIND"
P_SPATIAL_NAMESPACE = "RC_CW1_SR0_S8_FEATURE_RUNTIME_V1_P_SPATIAL"

PREJOIN_PROTECTED_ZERO: Mapping[str, int] = {
    "target_label_read_count": 0,
    "target_row_or_position_read_count": 0,
    "target_score_rank_or_outcome_read_count": 0,
    "D1_score_rank_slot_winner_gap_read_count": 0,
    "A10_A20_ownership_SWITCH_HOLD_read_count": 0,
    "human_or_model_spatial_annotation_read_count": 0,
    "opened_read_count": 0,
    "sealed_read_count": 0,
    "encoder_model_load_count": 0,
    "encoder_model_forward_count": 0,
    "home_write_count": 0,
    "model_training_update_count": 0,
}


class S8FeatureRuntimeError(RuntimeError):
    """A source, seal, or target-free boundary failed closed."""


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape and exact contiguous bytes."""

    if not isinstance(value, torch.Tensor):
        raise S8FeatureRuntimeError("hash input is not a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _identity_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def frozen_ap_all_patch_score(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
) -> torch.Tensor:
    """Replay the frozen N0/O32 AP statistic exactly.

    This is *not* sum-MaxSim.  Both local token banks are L2-normalized in
    float32, each query patch receives a temperature-0.07 log-mean-exp over
    every reference patch, and those query-patch values are averaged.
    """

    query = torch.as_tensor(query_tokens)
    reference = torch.as_tensor(reference_tokens)
    if (
        query.ndim != 2
        or reference.ndim != 2
        or query.shape[1] != LOCAL_DIMENSION
        or reference.shape[1] != LOCAL_DIMENSION
        or query.shape[0] < 1
        or reference.shape[0] < 1
        or not query.is_floating_point()
        or not reference.is_floating_point()
        or query.device != reference.device
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(reference).all())
    ):
        raise S8FeatureRuntimeError("AP requires finite local [patch,128] banks")
    query_unit = F.normalize(query.float(), dim=1, eps=1.0e-12)
    reference_unit = F.normalize(reference.float(), dim=1, eps=1.0e-12)
    similarity = query_unit @ reference_unit.T
    patch_support = AP_TEMPERATURE * (
        torch.logsumexp(similarity / AP_TEMPERATURE, dim=1)
        - math.log(float(reference.shape[0]))
    )
    score = patch_support.mean()
    if score.ndim != 0 or not bool(torch.isfinite(score)):
        raise RuntimeError("frozen AP statistic became non-finite")
    return score


@dataclass(frozen=True)
class FrozenSpatialReference:
    physical_row: int
    grid_shape: tuple[int, int]
    tokens: torch.Tensor
    tokens_sha256: str
    source_kind: str
    source_logical_sha256: str

    def __post_init__(self) -> None:
        tokens = torch.as_tensor(self.tokens).detach().cpu().contiguous()
        if (
            isinstance(self.physical_row, bool)
            or self.physical_row not in range(PHYSICAL_GALLERY_ROWS)
            or self.source_kind
            not in {"VALIDATED_4976_CACHE", "CPU_FROZEN_PROCESSOR_RECOVERY"}
            or len(self.source_logical_sha256) != 64
            or tokens.dtype != torch.float16
            or tokens.shape
            != (math.prod(self.grid_shape), LOCAL_DIMENSION)
            or not bool(torch.isfinite(tokens).all())
            or self.tokens_sha256 != tensor_sha256(tokens)
        ):
            raise S8FeatureRuntimeError("invalid frozen spatial reference")
        object.__setattr__(self, "tokens", tokens)


def _default_processor_factory() -> Any:
    # These flags prevent network fallback.  This path loads a processor only,
    # never the 7B encoder or any safetensors weight.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from colpali_engine.models import ColQwen2_5_Processor

    return ColQwen2_5_Processor.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )


def _default_processor_source_receipt() -> Mapping[str, Any]:
    files: dict[str, str] = {}
    for relative in PROCESSOR_CONFIG_FILES:
        path = MODEL_DIR / relative
        if not path.is_file():
            raise S8FeatureRuntimeError(
                f"frozen processor input is missing: {relative}"
            )
        files[relative] = sha256_file(path)
    try:
        colpali_version = metadata.version("colpali-engine")
    except metadata.PackageNotFoundError:
        colpali_version = "unknown"
    try:
        transformers_version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        transformers_version = "unknown"
    value = {
        "processor_class": "colpali_engine.models.ColQwen2_5_Processor",
        "model_directory": str(MODEL_DIR),
        "config_files_sha256": files,
        "weight_files_opened_or_hashed": False,
        "local_files_only": True,
        "colpali_engine_version": colpali_version,
        "transformers_version": transformers_version,
    }
    return {**value, "logical_sha256": canonical_sha256(value)}


def _default_gallery_embedding_loader() -> Sequence[torch.Tensor]:
    if not GALLERY_CACHE.is_file():
        raise S8FeatureRuntimeError("frozen gallery embedding is missing")
    payload = torch.load(
        GALLERY_CACHE, map_location="cpu", weights_only=False, mmap=True
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {"passage_emb", "setids"}
        or not isinstance(payload["passage_emb"], list)
        or len(payload["passage_emb"]) != PHYSICAL_GALLERY_ROWS
        or len(payload["setids"]) != PHYSICAL_GALLERY_ROWS
    ):
        raise S8FeatureRuntimeError("frozen gallery embedding schema drift")
    return payload["passage_emb"]


def _default_recovery_grid_builder(
    cached_valid_embedding: torch.Tensor, raw_path: Path, processor: Any
) -> ColNomicLocalGrid:
    from PIL import Image

    with Image.open(raw_path) as image:
        inputs = processor.process_images([image.convert("RGB")])
    if not isinstance(inputs, Mapping) or not {
        "input_ids",
        "attention_mask",
        "image_grid_thw",
    }.issubset(inputs):
        raise S8FeatureRuntimeError("frozen processor output schema drift")
    image_processor = getattr(processor, "image_processor", None)
    return split_cached_spatial_tokens(
        cached_valid_embedding,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_thw=inputs["image_grid_thw"],
        image_token_id=int(processor.image_token_id),
        merge_size=int(getattr(image_processor, "merge_size", 0)),
    )


class HybridSpatialReferenceResolver:
    """Validated 4,976-cache first, CPU frozen recovery second.

    All expensive resources are process-local and lazy.  A recovery processor
    is loaded at most once, gallery embeddings are mmap-loaded at most once,
    and every resolved physical row is cached in ``_resolved``.
    """

    def __init__(
        self,
        *,
        gallery_source: GallerySource | None = None,
        spatial_cache_resolver: ValidatedSpatialCacheResolver | None = None,
        legacy_candidate_union: Sequence[int] | None = None,
        processor_factory: Callable[[], Any] = _default_processor_factory,
        gallery_embedding_loader: Callable[[], Sequence[torch.Tensor]] = (
            _default_gallery_embedding_loader
        ),
        recovery_grid_builder: Callable[
            [torch.Tensor, Path, Any], ColNomicLocalGrid
        ] = _default_recovery_grid_builder,
        processor_source_receipt_factory: Callable[[], Mapping[str, Any]] = (
            _default_processor_source_receipt
        ),
        file_hasher: Callable[[Path], str] = sha256_file,
        spatial_cache_root: Path = SPATIAL_CACHE_ROOT,
    ) -> None:
        self._gallery_source = gallery_source
        self._spatial_cache_resolver = spatial_cache_resolver
        self._legacy_union = (
            None
            if legacy_candidate_union is None
            else tuple(int(item) for item in legacy_candidate_union)
        )
        self._legacy_work: dict[int, int] | None = None
        self._processor_factory = processor_factory
        self._gallery_embedding_loader = gallery_embedding_loader
        self._recovery_grid_builder = recovery_grid_builder
        self._processor_source_receipt_factory = processor_source_receipt_factory
        self._file_hasher = file_hasher
        self._spatial_cache_root = Path(spatial_cache_root)
        self._processor: Any | None = None
        self._processor_source_receipt: Mapping[str, Any] | None = None
        self._gallery_embeddings: Sequence[torch.Tensor] | None = None
        self._resolved: dict[int, FrozenSpatialReference] = {}

    @property
    def gallery_source(self) -> GallerySource:
        if self._gallery_source is None:
            self._gallery_source = build_gallery_source()
        return self._gallery_source

    def identity_for_row(self, physical_row: int) -> str:
        return self.gallery_source.identity_for_row(physical_row)

    def _ensure_legacy_work(self) -> dict[int, int]:
        if self._legacy_work is not None:
            return self._legacy_work
        if self._legacy_union is None:
            prejoin = build_prejoin_sources(gallery=self.gallery_source)
            self._legacy_union = tuple(
                sorted(
                    {
                        row
                        for record in prejoin.records
                        for row in record.candidate_physical_rows
                    }
                )
            )
        if (
            len(self._legacy_union) != LEGACY_SPATIAL_UNION_COUNT
            or tuple(sorted(set(self._legacy_union))) != self._legacy_union
        ):
            raise S8FeatureRuntimeError("legacy 4,976-row spatial union drift")
        self._legacy_work = gallery_work_ordinal_by_physical_row(
            self._legacy_union
        )
        return self._legacy_work

    def _cached(self, physical_row: int, work_ordinal: int) -> FrozenSpatialReference:
        if self._spatial_cache_resolver is None:
            self._spatial_cache_resolver = ValidatedSpatialCacheResolver(
                self._spatial_cache_root
            )
        entry: SpatialTokenEntry = self._spatial_cache_resolver.get(
            "gallery", work_ordinal
        )
        if entry.physical_row != physical_row:
            raise S8FeatureRuntimeError("validated spatial cache row binding drift")
        return FrozenSpatialReference(
            physical_row=physical_row,
            grid_shape=entry.grid_shape,
            tokens=entry.tokens,
            tokens_sha256=entry.tokens_sha256,
            source_kind="VALIDATED_4976_CACHE",
            source_logical_sha256=entry.entry_logical_sha256,
        )

    def _recovered(self, physical_row: int) -> FrozenSpatialReference:
        gallery = self.gallery_source
        if len(gallery.raw_paths) != PHYSICAL_GALLERY_ROWS:
            raise S8FeatureRuntimeError("raw gallery row population drift")
        if self._gallery_embeddings is None:
            self._gallery_embeddings = self._gallery_embedding_loader()
        if len(self._gallery_embeddings) != PHYSICAL_GALLERY_ROWS:
            raise S8FeatureRuntimeError("frozen gallery tensor population drift")
        if self._processor is None:
            self._processor = self._processor_factory()
            self._processor_source_receipt = dict(
                self._processor_source_receipt_factory()
            )
            if self._processor_source_receipt.get("logical_sha256") != canonical_sha256(
                {
                    key: item
                    for key, item in self._processor_source_receipt.items()
                    if key != "logical_sha256"
                }
            ):
                raise S8FeatureRuntimeError("frozen processor receipt drift")
        raw_path = gallery.raw_paths[physical_row]
        cached = (
            self._gallery_embeddings[physical_row]
            .detach()
            .cpu()
            .contiguous()
        )
        if (
            cached.dtype != torch.float16
            or cached.ndim != 2
            or cached.shape[1] != LOCAL_DIMENSION
            or not bool(torch.isfinite(cached).all())
        ):
            raise S8FeatureRuntimeError("frozen gallery row tensor drift")
        grid = self._recovery_grid_builder(
            cached, raw_path, self._processor
        )
        tokens = grid.tokens.detach().cpu().to(torch.float16).contiguous()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_kind": "CPU_FROZEN_PROCESSOR_RECOVERY",
            "physical_row": physical_row,
            "raw_path": str(raw_path),
            "raw_file_sha256": self._file_hasher(raw_path),
            "frozen_valid_embedding_sha256": tensor_sha256(cached),
            "grid_shape": list(grid.grid_shape),
            "image_token_indices": grid.image_token_indices.tolist(),
            "proposal_grid_sha256": proposal_grid_sha256(grid),
            "tokens_sha256": tensor_sha256(tokens),
            "processor_source_logical_sha256": self._processor_source_receipt[
                "logical_sha256"
            ],
            "encoder_model_load_count": 0,
            "encoder_model_forward_count": 0,
        }
        return FrozenSpatialReference(
            physical_row=physical_row,
            grid_shape=grid.grid_shape,
            tokens=tokens,
            tokens_sha256=payload["tokens_sha256"],
            source_kind="CPU_FROZEN_PROCESSOR_RECOVERY",
            source_logical_sha256=canonical_sha256(payload),
        )

    def resolve(self, physical_row: int) -> FrozenSpatialReference:
        if (
            isinstance(physical_row, bool)
            or not isinstance(physical_row, int)
            or physical_row not in range(PHYSICAL_GALLERY_ROWS)
        ):
            raise S8FeatureRuntimeError("invalid physical gallery row")
        if physical_row in self._resolved:
            return self._resolved[physical_row]
        work = self._ensure_legacy_work()
        output = (
            self._cached(physical_row, work[physical_row])
            if physical_row in work
            else self._recovered(physical_row)
        )
        self._resolved[physical_row] = output
        return output


@dataclass(frozen=True)
class IdentityFixedC128Source:
    query_ordinal: int
    query_id: str
    track: str
    query_grid_shape: tuple[int, int]
    query_tokens: torch.Tensor
    query_tokens_sha256: str
    c0_prejoin_logical_sha256: str
    candidate_mapping_sha256: str
    candidate_physical_rows: tuple[int, ...]


def load_identity_fixed_c128_source(
    query_ordinal: int,
    *,
    query_specs: Sequence[TargetFreeQuerySpec] | None = None,
    query_loader: Callable[[TargetFreeQuerySpec], Any] = load_query_tokens,
    c0_root: Path = C0_FOLD0_ROOT,
    c0_validation_path: Path = C0_FOLD0_VALIDATION,
) -> IdentityFixedC128Source:
    """Load any of the 987 identity-fixed fold-0 C128 records target-free."""

    specs = tuple(load_query_ledger() if query_specs is None else query_specs)
    if (
        isinstance(query_ordinal, bool)
        or not isinstance(query_ordinal, int)
        or query_ordinal not in range(len(specs))
    ):
        raise S8FeatureRuntimeError("query ordinal is outside the frozen ledger")
    spec = specs[query_ordinal]
    if spec.query_ordinal != query_ordinal:
        raise S8FeatureRuntimeError("query ledger ordinal drift")
    validation, _ = _load_c0_validation(c0_validation_path)
    ledger = json.loads(QUERY_LEDGER.read_text())
    _, checkpoint_sha = fold_d1_checkpoint(0)
    if (
        validation.get("d1_checkpoint_sha256") != checkpoint_sha
        or validation.get("source_query_ledger_logical_sha256")
        != ledger.get("logical_sha256")
    ):
        raise S8FeatureRuntimeError("identity-fixed C0 authority binding drift")
    c0 = load_prejoin(
        c0_root / "prejoin" / f"{query_ordinal:04d}.pt",
        fold=0,
        ordinal=query_ordinal,
        contract_sha256=str(validation["contract_sha256"]),
        d1_checkpoint_sha256=checkpoint_sha,
        source_query_ledger_logical_sha256=str(ledger["logical_sha256"]),
    )
    loaded = query_loader(spec)
    query = loaded.image_tokens.detach().cpu().contiguous()
    grid = (int(loaded.grid_h), int(loaded.grid_w))
    rows = tuple(
        int(item)
        for item in c0["candidate_physical_rows_in_hash_permutation"]
    )
    if (
        c0["query_id"] != spec.query_id
        or len(rows) != CANDIDATE_COUNT
        or len(set(rows)) != CANDIDATE_COUNT
        or query.dtype != torch.float16
        or query.shape != (math.prod(grid), LOCAL_DIMENSION)
        or grid != (spec.grid_h, spec.grid_w)
        or not bool(torch.isfinite(query).all())
    ):
        raise S8FeatureRuntimeError("identity-fixed C128/query source drift")
    return IdentityFixedC128Source(
        query_ordinal=query_ordinal,
        query_id=spec.query_id,
        track=spec.track,
        query_grid_shape=grid,
        query_tokens=query,
        query_tokens_sha256=tensor_sha256(query),
        c0_prejoin_logical_sha256=str(c0["logical_sha256"]),
        candidate_mapping_sha256=str(
            c0["candidate_row_label_mapping_sha256_receipt_only"]
        ),
        candidate_physical_rows=rows,
    )


@dataclass(frozen=True)
class TargetFreeAPContext:
    schema_version: str
    query_ordinal: int
    query_id: str
    track: str
    query_grid_shape: tuple[int, int]
    query_tokens: torch.Tensor
    query_tokens_sha256: str
    c0_prejoin_logical_sha256: str
    candidate_mapping_sha256: str
    candidate_physical_rows: tuple[int, ...]
    reference_source_logical_sha256: tuple[str, ...]
    reference_tokens_sha256: tuple[str, ...]
    candidate_binding_source_rows: tuple[int, ...]
    p_spatial_destination_to_source: torch.Tensor
    ap_scores: torch.Tensor
    ap_scores_sha256: str
    ap_formula: str
    ap_temperature: float
    protected_access_counts: Mapping[str, int]
    logical_sha256: str


def _prejoin_payload(context: TargetFreeAPContext) -> dict[str, Any]:
    return {
        "schema_version": context.schema_version,
        "query_ordinal": context.query_ordinal,
        "query_id": context.query_id,
        "track": context.track,
        "query_grid_shape": list(context.query_grid_shape),
        "query_tokens_sha256": context.query_tokens_sha256,
        "c0_prejoin_logical_sha256": context.c0_prejoin_logical_sha256,
        "candidate_mapping_sha256": context.candidate_mapping_sha256,
        "candidate_physical_rows": list(context.candidate_physical_rows),
        "reference_source_logical_sha256": list(
            context.reference_source_logical_sha256
        ),
        "reference_tokens_sha256": list(context.reference_tokens_sha256),
        "candidate_binding_source_rows": list(
            context.candidate_binding_source_rows
        ),
        "p_spatial_destination_to_source_sha256": tensor_sha256(
            context.p_spatial_destination_to_source
        ),
        "ap_scores_sha256": context.ap_scores_sha256,
        "ap_formula": context.ap_formula,
        "ap_temperature": context.ap_temperature,
        "target_join_performed": False,
        "protected_access_counts": dict(context.protected_access_counts),
    }


def validate_target_free_ap_context(context: TargetFreeAPContext) -> None:
    if not isinstance(context, TargetFreeAPContext):
        raise S8FeatureRuntimeError("expected a target-free AP context")
    query = torch.as_tensor(context.query_tokens).detach().cpu().contiguous()
    scores = torch.as_tensor(context.ap_scores).detach().cpu().contiguous()
    permutation = (
        torch.as_tensor(
            context.p_spatial_destination_to_source, dtype=torch.long
        )
        .detach()
        .cpu()
        .contiguous()
    )
    rows = context.candidate_physical_rows
    if (
        context.schema_version != SCHEMA_VERSION
        or context.ap_formula != AP_FORMULA
        or context.ap_temperature != AP_TEMPERATURE
        or context.protected_access_counts != PREJOIN_PROTECTED_ZERO
        or len(rows) != CANDIDATE_COUNT
        or len(set(rows)) != CANDIDATE_COUNT
        or len(context.reference_source_logical_sha256) != CANDIDATE_COUNT
        or len(context.reference_tokens_sha256) != CANDIDATE_COUNT
        or len(context.candidate_binding_source_rows) != CANDIDATE_COUNT
        or set(context.candidate_binding_source_rows) != set(rows)
        or any(
            left == right
            for left, right in zip(
                rows, context.candidate_binding_source_rows, strict=True
            )
        )
        or query.dtype != torch.float16
        or query.shape
        != (math.prod(context.query_grid_shape), LOCAL_DIMENSION)
        or tensor_sha256(query) != context.query_tokens_sha256
        or scores.shape != (CANDIDATE_COUNT,)
        or scores.dtype != torch.float32
        or not bool(torch.isfinite(scores).all())
        or tensor_sha256(scores) != context.ap_scores_sha256
        or permutation.shape != (query.shape[0],)
        or permutation.unique().numel() != query.shape[0]
        or bool(permutation.eq(torch.arange(query.shape[0])).any())
        or context.logical_sha256 != canonical_sha256(_prejoin_payload(context))
    ):
        raise S8FeatureRuntimeError("target-free AP context seal drift")
    expected_permutation = role_preserving_query_derangement(
        context.query_grid_shape,
        namespace=P_SPATIAL_NAMESPACE,
        key=context.query_id,
    )
    if not torch.equal(permutation, expected_permutation):
        raise S8FeatureRuntimeError("P_SPATIAL permutation receipt drift")


def _build_target_free_ap_context(
    source: IdentityFixedC128Source,
    *,
    resolver: HybridSpatialReferenceResolver,
    device: torch.device,
) -> TargetFreeAPContext:
    rows = source.candidate_physical_rows
    identities = tuple(resolver.identity_for_row(row) for row in rows)
    # Identity-fixed C0 is defined over exact identities.  Failing here is
    # safer than silently creating a same-identity candidate-binding control.
    if len(set(identities)) != CANDIDATE_COUNT:
        raise S8FeatureRuntimeError(
            "identity-fixed C128 contains duplicate corrected identities"
        )
    binding_positions = deterministic_derangement(
        CANDIDATE_COUNT,
        namespace=C_BIND_NAMESPACE,
        key=f"{source.query_id}:{canonical_sha256(list(rows))}",
    )
    binding_rows = tuple(rows[int(item)] for item in binding_positions)
    if any(
        identities[position] == identities[int(source_position)]
        for position, source_position in enumerate(binding_positions)
    ):
        raise S8FeatureRuntimeError("candidate-binding control is not identity-disjoint")
    p_spatial = role_preserving_query_derangement(
        source.query_grid_shape,
        namespace=P_SPATIAL_NAMESPACE,
        key=source.query_id,
    )
    query = source.query_tokens.to(device)
    scores: list[torch.Tensor] = []
    source_hashes: list[str] = []
    token_hashes: list[str] = []
    with torch.inference_mode():
        for row in rows:
            reference = resolver.resolve(row)
            scores.append(
                frozen_ap_all_patch_score(
                    query, reference.tokens.to(device)
                ).detach().cpu()
            )
            source_hashes.append(reference.source_logical_sha256)
            token_hashes.append(reference.tokens_sha256)
    ap_scores = torch.stack(scores).to(torch.float32).contiguous()
    provisional = TargetFreeAPContext(
        schema_version=SCHEMA_VERSION,
        query_ordinal=source.query_ordinal,
        query_id=source.query_id,
        track=source.track,
        query_grid_shape=source.query_grid_shape,
        query_tokens=source.query_tokens,
        query_tokens_sha256=source.query_tokens_sha256,
        c0_prejoin_logical_sha256=source.c0_prejoin_logical_sha256,
        candidate_mapping_sha256=source.candidate_mapping_sha256,
        candidate_physical_rows=rows,
        reference_source_logical_sha256=tuple(source_hashes),
        reference_tokens_sha256=tuple(token_hashes),
        candidate_binding_source_rows=binding_rows,
        p_spatial_destination_to_source=p_spatial,
        ap_scores=ap_scores,
        ap_scores_sha256=tensor_sha256(ap_scores),
        ap_formula=AP_FORMULA,
        ap_temperature=AP_TEMPERATURE,
        protected_access_counts=dict(PREJOIN_PROTECTED_ZERO),
        logical_sha256="",
    )
    output = TargetFreeAPContext(
        **{
            **provisional.__dict__,
            "logical_sha256": canonical_sha256(_prejoin_payload(provisional)),
        }
    )
    validate_target_free_ap_context(output)
    return output


def materialize_target_free_ap_context(
    query_ordinal: int,
    *,
    resolver: HybridSpatialReferenceResolver | None = None,
    device: torch.device | str = "cpu",
    source_loader: Callable[[int], IdentityFixedC128Source] = (
        load_identity_fixed_c128_source
    ),
) -> TargetFreeAPContext:
    """Materialize and seal AP for an anonymous natural C128 context."""

    runtime_resolver = (
        HybridSpatialReferenceResolver() if resolver is None else resolver
    )
    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise S8FeatureRuntimeError("CUDA was requested but is unavailable")
    source = source_loader(query_ordinal)
    if source.query_ordinal != query_ordinal:
        raise S8FeatureRuntimeError("C128 source loader changed the query ordinal")
    return _build_target_free_ap_context(
        source, resolver=runtime_resolver, device=run_device
    )


@dataclass(frozen=True)
class SupervisedTrainingEpisode:
    schema_version: str
    prejoin_logical_sha256: str
    query_ordinal: int
    query_id: str
    target_physical_row: int
    rival_physical_row: int
    target_ap_score: float
    rival_ap_score: float
    ap_margin: float
    target_identity_sha256: str
    target_join_performed: bool
    target_inserted: bool
    logical_sha256: str


def join_supervised_training_episode(
    context: TargetFreeAPContext,
    exact_identity: str,
    *,
    resolver: HybridSpatialReferenceResolver,
) -> SupervisedTrainingEpisode:
    """Join an exact identity only after AP/C128 hashes are immutable."""

    validate_target_free_ap_context(context)
    if not isinstance(exact_identity, str) or not exact_identity:
        raise S8FeatureRuntimeError("postjoin exact identity is invalid")
    identities = tuple(
        resolver.identity_for_row(row)
        for row in context.candidate_physical_rows
    )
    matches = [
        position
        for position, identity in enumerate(identities)
        if identity == exact_identity
    ]
    if len(matches) != 1:
        # No supplemental target is ever inserted.  A miss is ineligible.
        raise S8FeatureRuntimeError(
            "natural target is absent or non-unique in identity-fixed C128"
        )
    target_position = matches[0]
    eligible = [
        position
        for position, identity in enumerate(identities)
        if identity != exact_identity
    ]
    if not eligible:
        raise S8FeatureRuntimeError("postjoin AP rival population is empty")
    scores = context.ap_scores
    maximum = max(float(scores[position]) for position in eligible)
    tied = [
        position
        for position in eligible
        if float(scores[position]) == maximum
    ]
    rival_position = min(
        tied, key=lambda position: context.candidate_physical_rows[position]
    )
    target_row = context.candidate_physical_rows[target_position]
    rival_row = context.candidate_physical_rows[rival_position]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prejoin_logical_sha256": context.logical_sha256,
        "query_ordinal": context.query_ordinal,
        "query_id": context.query_id,
        "target_physical_row": target_row,
        "rival_physical_row": rival_row,
        "target_ap_score": float(scores[target_position]),
        "rival_ap_score": float(scores[rival_position]),
        "ap_margin": float(scores[target_position] - scores[rival_position]),
        "target_identity_sha256": _identity_hash(exact_identity),
        "target_join_performed": True,
        "target_inserted": False,
    }
    return SupervisedTrainingEpisode(
        **payload, logical_sha256=canonical_sha256(payload)
    )


@dataclass(frozen=True)
class CandidateFeaturePaths:
    schema_version: str
    prejoin_logical_sha256: str
    physical_row: int
    candidate_reference_source_sha256: str
    binding_reference_physical_row: int
    binding_reference_source_sha256: str
    base_preparation: NaturalSuperregionPreparationV2
    p_spatial_preparation: NaturalSuperregionPreparationV2
    base: NaturalAllRegionFeatureLedgerS8V1
    candidate_binding: NaturalAllRegionFeatureLedgerS8V1
    spatial: NaturalAllRegionFeatureLedgerS8V1
    base_directional_p_features: torch.Tensor
    base_directional_v_features: torch.Tensor
    candidate_binding_directional_p_features: torch.Tensor
    candidate_binding_directional_v_features: torch.Tensor
    spatial_directional_p_features: torch.Tensor
    spatial_directional_v_features: torch.Tensor
    logical_sha256: str


def _candidate_path_payload(value: CandidateFeaturePaths) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "prejoin_logical_sha256": value.prejoin_logical_sha256,
        "physical_row": value.physical_row,
        "candidate_reference_source_sha256": (
            value.candidate_reference_source_sha256
        ),
        "binding_reference_physical_row": value.binding_reference_physical_row,
        "binding_reference_source_sha256": (
            value.binding_reference_source_sha256
        ),
        "base_preparation_seal_sha256": value.base_preparation.seal_sha256,
        "p_spatial_preparation_seal_sha256": (
            value.p_spatial_preparation.seal_sha256
        ),
        "base_ledger_sha256": value.base.logical_sha256,
        "candidate_binding_ledger_sha256": value.candidate_binding.logical_sha256,
        "spatial_ledger_sha256": value.spatial.logical_sha256,
        "base_directional_p_features_sha256": tensor_sha256(
            value.base_directional_p_features
        ),
        "base_directional_v_features_sha256": tensor_sha256(
            value.base_directional_v_features
        ),
        "candidate_binding_directional_p_features_sha256": tensor_sha256(
            value.candidate_binding_directional_p_features
        ),
        "candidate_binding_directional_v_features_sha256": tensor_sha256(
            value.candidate_binding_directional_v_features
        ),
        "spatial_directional_p_features_sha256": tensor_sha256(
            value.spatial_directional_p_features
        ),
        "spatial_directional_v_features_sha256": tensor_sha256(
            value.spatial_directional_v_features
        ),
        "legacy_averaged_R12_features_use": (
            "regression_diagnostic_only_forbidden_for_formal_S8_model"
        ),
        "score_api_target_label_rank_slot_winner_fields": [],
    }


def materialize_candidate_feature_paths(
    context: TargetFreeAPContext,
    physical_row: int,
    *,
    resolver: HybridSpatialReferenceResolver,
    device: torch.device | str = "cpu",
) -> CandidateFeaturePaths:
    """Build BASE, fixed-P C_BIND and role-preserving P_SPATIAL ledgers.

    ``physical_row`` is intrinsic gallery metadata, not a candidate rank or
    slot.  The function locates its sealed C128 position internally only to
    read the result-blind candidate-binding derangement.
    """

    validate_target_free_ap_context(context)
    try:
        position = context.candidate_physical_rows.index(int(physical_row))
    except (ValueError, TypeError) as error:
        raise S8FeatureRuntimeError(
            "physical row is absent from the sealed natural C128"
        ) from error
    reference = resolver.resolve(int(physical_row))
    if (
        reference.source_logical_sha256
        != context.reference_source_logical_sha256[position]
        or reference.tokens_sha256 != context.reference_tokens_sha256[position]
    ):
        raise S8FeatureRuntimeError("reference source changed after AP sealing")
    binding_row = context.candidate_binding_source_rows[position]
    foreign = resolver.resolve(binding_row)
    binding_position = context.candidate_physical_rows.index(binding_row)
    if (
        foreign.source_logical_sha256
        != context.reference_source_logical_sha256[binding_position]
        or resolver.identity_for_row(binding_row)
        == resolver.identity_for_row(int(physical_row))
    ):
        raise S8FeatureRuntimeError("candidate-binding source drift")
    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise S8FeatureRuntimeError("CUDA was requested but is unavailable")
    query = context.query_tokens.to(run_device)
    reference_tokens = reference.tokens.to(run_device)
    foreign_tokens = resample_reference_content(
        foreign.tokens.to(run_device),
        foreign.grid_shape,
        reference.grid_shape,
    )
    spatial_query = query[
        context.p_spatial_destination_to_source.to(run_device)
    ].contiguous()
    with torch.inference_mode():
        base_preparation = prepare_natural_superregion_candidate_v2(
            query,
            reference_tokens,
            query_grid_shape=context.query_grid_shape,
            reference_grid_shape=reference.grid_shape,
        )
        base_verification = verify_all_natural_superregions_s8_v1(
            base_preparation,
            query,
            reference_tokens,
            control_mode=CONTROL_BASE,
        )
        binding_verification = verify_all_natural_superregions_s8_v1(
            base_preparation,
            query,
            foreign_tokens,
            control_mode=CONTROL_V_C_BIND,
        )
        spatial_preparation = prepare_natural_superregion_candidate_v2(
            spatial_query,
            reference_tokens,
            query_grid_shape=context.query_grid_shape,
            reference_grid_shape=reference.grid_shape,
        )
        spatial_verification = verify_all_natural_superregions_s8_v1(
            spatial_preparation,
            spatial_query,
            reference_tokens,
            control_mode=CONTROL_BASE,
        )
        base = materialize_all_natural_superregion_feature_ledger_s8_v1(
            base_preparation, base_verification
        )
        binding = materialize_all_natural_superregion_feature_ledger_s8_v1(
            base_preparation, binding_verification
        )
        spatial = materialize_all_natural_superregion_feature_ledger_s8_v1(
            spatial_preparation, spatial_verification
        )
    if (
        not torch.equal(base.features[:, :8], binding.features[:, :8])
        or base.region_keys != binding.region_keys
        or base.features.shape != spatial.features.shape
        or base.directional_p_features.ndim != 3
        or base.directional_p_features.shape[:2]
        != (2, base.features.shape[0])
        or base.directional_v_features.shape
        != (2, base.features.shape[0], 4)
        or binding.directional_p_features.shape
        != base.directional_p_features.shape
        or binding.directional_v_features.shape
        != base.directional_v_features.shape
        or spatial.directional_p_features.shape
        != base.directional_p_features.shape
        or spatial.directional_v_features.shape
        != base.directional_v_features.shape
        or not torch.equal(
            base.directional_p_features,
            binding.directional_p_features,
        )
        or not bool(base.features[:, 11].eq(1.0).all())
        or not bool(binding.features[:, 11].eq(1.0).all())
        or not bool(spatial.features[:, 11].eq(1.0).all())
        or not bool(base.directional_v_features[..., 3].eq(1.0).all())
        or not bool(binding.directional_v_features[..., 3].eq(1.0).all())
        or not bool(spatial.directional_v_features[..., 3].eq(1.0).all())
    ):
        raise S8FeatureRuntimeError("S8 all-region control ledger contract drift")
    provisional = CandidateFeaturePaths(
        schema_version=SCHEMA_VERSION,
        prejoin_logical_sha256=context.logical_sha256,
        physical_row=int(physical_row),
        candidate_reference_source_sha256=reference.source_logical_sha256,
        binding_reference_physical_row=binding_row,
        binding_reference_source_sha256=foreign.source_logical_sha256,
        base_preparation=base_preparation,
        p_spatial_preparation=spatial_preparation,
        base=base,
        candidate_binding=binding,
        spatial=spatial,
        base_directional_p_features=base.directional_p_features,
        base_directional_v_features=base.directional_v_features,
        candidate_binding_directional_p_features=(
            binding.directional_p_features
        ),
        candidate_binding_directional_v_features=(
            binding.directional_v_features
        ),
        spatial_directional_p_features=spatial.directional_p_features,
        spatial_directional_v_features=spatial.directional_v_features,
        logical_sha256="",
    )
    return CandidateFeaturePaths(
        **{
            **provisional.__dict__,
            "logical_sha256": canonical_sha256(
                _candidate_path_payload(provisional)
            ),
        }
    )


__all__ = [
    "SCHEMA_VERSION",
    "AP_TEMPERATURE",
    "AP_FORMULA",
    "PREJOIN_PROTECTED_ZERO",
    "S8FeatureRuntimeError",
    "FrozenSpatialReference",
    "HybridSpatialReferenceResolver",
    "IdentityFixedC128Source",
    "TargetFreeAPContext",
    "SupervisedTrainingEpisode",
    "CandidateFeaturePaths",
    "tensor_sha256",
    "frozen_ap_all_patch_score",
    "load_identity_fixed_c128_source",
    "validate_target_free_ap_context",
    "materialize_target_free_ap_context",
    "join_supervised_training_episode",
    "materialize_candidate_feature_paths",
]
