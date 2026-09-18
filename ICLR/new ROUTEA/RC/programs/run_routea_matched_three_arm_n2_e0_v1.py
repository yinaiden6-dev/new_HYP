#!/usr/bin/env python3
"""Run the N2 corrected-gallery/current-runtime design-lineage E0 gate."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
WORKSPACE = ROOT.parents[2]
for search_path in reversed(
    (ROOT / "src", ROOT / "programs", ROUTEA / "route_a_core", ROUTEA)
):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_a0_fold_d1 as d1_wrapper  # noqa: E402
from rc_aslo_xf.l0_targetfree_source import load_query_ledger  # noqa: E402
from rc_aslo_xf.gallery_identity_repair import (  # noqa: E402
    CONTRACT as IDENTITY_REPAIR_CONTRACT,
    REGISTRY as IDENTITY_REPAIR_REGISTRY,
)
from rc_aslo_xf.n2_corrected_d1_runtime_v1 import (  # noqa: E402
    N2CorrectedD1RuntimeError,
    corrected_labels_from_legacy,
    fold_assignment_sha256,
    join_query_ids_to_canonical_folds,
    natural_c128,
    reduce_corrected_full_gallery_scores,
    validate_canonical_987_ledger,
    validate_corrected_identity_axis,
)


CONTRACT = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_V1_20260902.md"
DESIGN = ROOT / "plan/ROUTEA_MATCHED_THREE_ARM_N2_TO_UNTOUCHED_EXTERNAL_CONFIRMATION_DESIGN_V1_20260902.md"
DESIGN_VALIDATION = ROOT / "results/routea_matched_three_arm_n2_external_design_v1/independent_validation.json"
QUERY_LEDGER = ROOT / "cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json"
UPSTREAMS = ROOT / "registry/upstream_inputs.json"
CURRENT64 = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
CURRENT64_VALIDATION = CURRENT64 / "validation.json"
CURRENT_SOURCE = ROOT / "results/romav2_colnomic_current_runtime_bridge_prejoin_v1"
PAIR64 = ROOT / "results/routea_matched_three_arm_pair64_training_features_v2/payload.pt"
PAIR64_VALIDATION = ROOT / "results/routea_matched_three_arm_pair64_training_features_v2/independent_validation.json"
RUNTIME = ROOT / "src/rc_aslo_xf/n2_corrected_d1_runtime_v1.py"
VALIDATOR = ROOT / "programs/validate_routea_matched_three_arm_n2_e0_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_e0_v1"
MODEL = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b"
GALLERY_CACHE = WORKSPACE / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
EXPECTED_PAIR64_SHA256 = "d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3"
EXPECTED_PAIR64_VALIDATION_SHA256 = "657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b"

EXPECTED_HASHES = {
    DESIGN: "0b796b5a88481d7189ae026fc4e3464622f24f4e15b043e1e919a02f3df15ebd",
    DESIGN_VALIDATION: "5681b3863c56bb8b7fa7ebd02f16c4253ba2412d4cc4ea3cd216bb90e420e0a5",
    QUERY_LEDGER: "df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec",
    UPSTREAMS: "6f0399baa46a0c774a7214486467320eb0e778fe91cefd2b2010bc7512f75eac",
    CURRENT64_VALIDATION: "6c031e5a82b8057a48b48794d53dd76f4888b76ed03f081a925425c64e9621cb",
    ROOT / "programs/run_a0_fold_d1.py": "d8d58a0d96e1c3bc8457cf16f8b6aa0be46ceb7893a0cb6c0b75c50042eb0174",
    ROUTEA / "programs/run_routea_v3_1_c6direct_m1.py": "cdbac5833699c5904475cc308d1fbae6587a94765f9096d993093b2c1f621e41",
    ROUTEA / "route_a_core/route_a/o1_c6direct_m1_d1.py": "a3973e45f97f2157883acb96b64c614d9fa93ec80187e6608f462fe841737274",
    GALLERY_CACHE: "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    PAIR64: EXPECTED_PAIR64_SHA256,
    PAIR64_VALIDATION: EXPECTED_PAIR64_VALIDATION_SHA256,
}

FIXTURES = (
    ("OUTCOME-0533", "outcome", (36, 20), "DECODED_RAW_BEFORE_EXIF", 1, True),
    ("DIFFICULT-0128", "difficult", (24, 32), "DECODED_RAW_BEFORE_EXIF", 6, True),
    ("NDV2-007-P01", "new_difficult_train", (32, 24), "EXIF_ORIENTED_BEFORE_RESIZE", 6, True),
    ("DIFFICULT-0025", "difficult", (25, 29), "DECODED_RAW_BEFORE_EXIF", 1, False),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "logical_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_predecessors() -> dict[str, Any]:
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file() and not path.is_symlink(), f"required source absent: {path}")
        require(sha256_file(path) == expected, f"required source hash drift: {path}")
    design_validation = json.loads(DESIGN_VALIDATION.read_text())
    require(
        design_validation.get("status") == "ROUTEA_MATCHED_THREE_ARM_N2_EXTERNAL_DESIGN_VALIDATED"
        and design_validation.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_D1_C_E0_DESIGN_AND_LINEAGE_PREFLIGHT"
        and design_validation.get("scientific_GO_or_NO_GO") is None
        and design_validation.get("n2_training_authorized") is False
        and design_validation.get("external_execution_authorized") is False
        and all(design_validation.get("checks", {}).values()),
        "N2 design predecessor is not a validated E0 authority",
    )
    return design_validation


def validate_d1_recipe() -> tuple[list[dict[str, Any]], int, bool]:
    recipes: list[dict[str, Any]] = []
    adapter_parameter_count = -1
    legacy_corrected_reducer_rejected = False
    for fold in range(5):
        module = d1_wrapper.import_runner()
        d1_wrapper.configure_fold(module, fold)
        recipe = module.d1.frozen_recipe()
        require(
            recipe.get("steps") == 800
            and recipe.get("batch_size") == 4
            and recipe.get("track_order")
            == ["outcome", "outcome", "difficult", "new_difficult_train"]
            and recipe.get("candidate_count") == 64
            and recipe.get("negative_count") == 63
            and recipe.get("optimizer")
            == {
                "kind": "AdamW",
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "gradient_clip_norm": 1.0,
            }
            and recipe.get("checkpoint_selection")
            == "final_step_only_no_holdout_selection"
            and recipe.get("seed") == 17
            and recipe.get("fold") == fold
            and recipe.get("effective_seed") == 17 + 1009 * fold,
            f"frozen D1 recipe drift at fold {fold}",
        )
        if fold == 0:
            adapter = module.d1.build_fresh_d1_adapter()
            adapter_parameter_count = sum(parameter.numel() for parameter in adapter.parameters())
            require(adapter_parameter_count == 49_792, "D1 adapter parameter count drift")
        recipes.append(recipe)
    require(module.d1.EXPECTED_FULL_GALLERY_LABELS == 5404, "legacy 5404 boundary drift")
    return recipes, adapter_parameter_count, legacy_corrected_reducer_rejected


def load_corrected_gallery_and_reducer_fixture() -> tuple[tuple[str, ...], dict[str, Any]]:
    gallery_payload = torch.load(GALLERY_CACHE, map_location="cpu", weights_only=False, mmap=True)
    legacy = tuple(map(str, gallery_payload.get("setids", ())))
    require(len(legacy) == 5413 and len(set(legacy)) == 5404, "legacy gallery input drift")
    corrected = corrected_labels_from_legacy(legacy)
    axis_receipt = validate_corrected_identity_axis(corrected)

    high = torch.zeros(5413, dtype=torch.float64)
    high[714], high[715] = 5.0, 6.0
    high_reduction = reduce_corrected_full_gallery_scores(high, corrected)
    high_slot = high_reduction.identity_values.index("Biogen_21")
    require(
        int(high_reduction.representative_physical_rows[high_slot]) == 715
        and float(high_reduction.identity_scores[high_slot]) == 6.0,
        "corrected max-row reducer fixture failed",
    )

    high[714] = 6.0
    tie_reduction = reduce_corrected_full_gallery_scores(high, corrected)
    tie_slot = tie_reduction.identity_values.index("Biogen_21")
    require(int(tie_reduction.representative_physical_rows[tie_slot]) == 714, "duplicate-row tie failed")
    all_tie = natural_c128(
        reduce_corrected_full_gallery_scores(torch.zeros(5413, dtype=torch.float64), corrected)
    )
    require(
        all_tie.representative_physical_rows == tuple(range(128))
        and len(set(all_tie.identities)) == 128,
        "all-identity tie/C128 reducer fixture failed",
    )

    legacy_axis_rejected = False
    try:
        reduce_corrected_full_gallery_scores(torch.zeros(5413), legacy)
    except N2CorrectedD1RuntimeError:
        legacy_axis_rejected = True
    require(legacy_axis_rejected, "legacy 5404 axis did not fail closed")

    # The historical full-gallery helper must remain unmodified and reject the
    # successor axis; N2 never monkey-patches its 5,404 constant.
    module = d1_wrapper.import_runner()
    historical_rejected = False
    try:
        module.d1.aggregate_full_gallery_exact_label_scores(
            torch.zeros(5413, dtype=torch.float64), corrected
        )
    except ValueError:
        historical_rejected = True
    require(historical_rejected, "legacy reducer unexpectedly accepted corrected 5412 axis")
    return corrected, {
        **axis_receipt,
        "max_row_representative": int(high_reduction.representative_physical_rows[high_slot]),
        "duplicate_tie_representative": int(tie_reduction.representative_physical_rows[tie_slot]),
        "all_identity_tie_c128_rows_sha256": hashlib.sha256(
            json.dumps(list(all_tie.representative_physical_rows), separators=(",", ":")).encode()
        ).hexdigest(),
        "legacy_5404_axis_rejected": legacy_axis_rejected,
        "historical_full_gallery_reducer_rejected_corrected_5412": historical_rejected,
        "identity_repair_contract_sha256": sha256_file(IDENTITY_REPAIR_CONTRACT),
        "identity_repair_registry_sha256": sha256_file(IDENTITY_REPAIR_REGISTRY),
    }


def load_current64_and_replay(
    corrected: tuple[str, ...], canonical_ledger: list[object]
) -> tuple[dict[str, dict], dict[str, Any], tuple[Any, ...]]:
    aggregate = json.loads(CURRENT64_VALIDATION.read_text())
    require(
        aggregate.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and aggregate.get("query_count") == 64
        and len(aggregate.get("shards", [])) == 8
        and all(aggregate.get("checks", {}).values()),
        "current64 aggregate authority drift",
    )
    sealed = {int(item["shard"]): item for item in aggregate["shards"]}
    require(set(sealed) == set(range(8)), "current64 shard seal population drift")
    source_validation_path = CURRENT_SOURCE / "validation.json"
    source_validation = json.loads(source_validation_path.read_text())
    source_seals = {
        int(item["shard"]): str(item["payload_sha256"])
        for item in source_validation.get("shards", [])
        if item.get("pass") is True
    }
    require(
        source_validation.get("status")
        == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_VALIDATION_PASS"
        and source_validation.get("target_label_read_count") == 0
        and all(source_validation.get("checks", {}).values())
        and set(source_seals) == set(range(8)),
        "current-runtime source validation drift",
    )
    records_by_query: dict[str, dict] = {}
    canonical_by_query = {str(spec.query_id): spec for spec in canonical_ledger}
    score_max_abs = 0.0
    shard_bindings = []
    source_join_projection: list[dict[str, Any]] = []
    poisoned_source_join_projection: list[dict[str, Any]] = []
    for shard in range(8):
        current_path = CURRENT64 / f"shard{shard:02d}/payload.pt"
        receipt_path = CURRENT64 / f"shard{shard:02d}/receipt.json"
        validation_path = CURRENT64 / f"shard{shard:02d}/validation.json"
        source_path = CURRENT_SOURCE / f"shard{shard:02d}/payload.pt"
        item = sealed[shard]
        require(
            item.get("payload_sha256") == sha256_file(current_path)
            and item.get("receipt_sha256") == sha256_file(receipt_path)
            and item.get("validation_sha256") == sha256_file(validation_path),
            f"current64 shard {shard} seal drift",
        )
        require(
            source_seals[shard] == sha256_file(source_path),
            f"current-runtime source shard {shard} seal drift",
        )
        current_payload = torch.load(current_path, map_location="cpu", weights_only=False, mmap=True)
        source_payload = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        source_by_query = {
            str(record["query_id"]): record for record in source_payload["records"]
        }
        require(len(current_payload["records"]) == len(source_by_query) == 8, "current64 source shard drift")
        poisoned_source_by_query = {
            query_id: {**record, "execution_ordinal": -(index + 1)}
            for index, (query_id, record) in enumerate(source_by_query.items())
        }
        require(set(poisoned_source_by_query) == set(source_by_query), "source execution poison changed query-id axis")
        for record in current_payload["records"]:
            query_id = str(record["query_id"])
            authority = source_by_query.get(query_id)
            canonical = canonical_by_query.get(query_id)
            require(authority is not None, "current64 query absent from source authority")
            require(canonical is not None, "current64 query absent from canonical ledger")
            require(query_id not in records_by_query, "current64 query duplicate")
            raw_scores = torch.as_tensor(record["raw_full_gallery_scores"])
            reduction = reduce_corrected_full_gallery_scores(raw_scores, corrected)
            c128 = natural_c128(reduction)
            authority_axis = tuple(map(int, authority["candidate_physical_rows"]))
            authority_ranked = tuple(map(int, authority["candidate_ranked_physical_rows"]))
            authority_scores = torch.as_tensor(authority["candidate_raw_scores"], dtype=torch.float64)
            observed_scores = torch.tensor(
                [float(raw_scores[row]) for row in authority_axis], dtype=torch.float64
            )
            delta = float((observed_scores - authority_scores).abs().max())
            score_max_abs = max(score_max_abs, delta)
            require(
                query_id == authority["query_id"]
                and record["track"] == authority["track"]
                == canonical.track
                and int(record["historical_query_ordinal"]) == canonical.query_ordinal
                and int(record["heldout_fold"]) == canonical.heldout_fold
                and tuple(record["query_grid_shape"]) == tuple(authority["query_grid_shape"])
                == (canonical.grid_h, canonical.grid_w)
                and Path(record["query_source_path"]).resolve() == canonical.path.resolve()
                and record["query_source_sha256"] == authority["query_source_sha256"]
                == canonical.source_image_sha256
                and torch.equal(record["raw_image_tokens"], authority["query_tokens"])
                and tensor_sha256(record["raw_image_tokens"]) == authority["query_tokens_sha256"]
                and tuple(record["raw_candidate_physical_rows"]) == authority_axis
                and tuple(record["raw_candidate_ranked_physical_rows"]) == authority_ranked
                and c128.representative_physical_rows == authority_ranked
                and tuple(sorted(c128.representative_physical_rows)) == authority_axis
                and delta <= 1e-6,
                f"current64 exact replay failed for {query_id}",
            )
            projection = {
                "query_id": query_id,
                "track": str(authority["track"]),
                "grid": list(map(int, authority["query_grid_shape"])),
                "query_tokens_sha256": str(authority["query_tokens_sha256"]),
                "candidate_physical_rows": list(map(int, authority["candidate_physical_rows"])),
                "candidate_ranked_physical_rows": list(map(int, authority["candidate_ranked_physical_rows"])),
            }
            poisoned_authority = poisoned_source_by_query[query_id]
            poisoned_projection = {
                "query_id": query_id,
                "track": str(poisoned_authority["track"]),
                "grid": list(map(int, poisoned_authority["query_grid_shape"])),
                "query_tokens_sha256": str(poisoned_authority["query_tokens_sha256"]),
                "candidate_physical_rows": list(map(int, poisoned_authority["candidate_physical_rows"])),
                "candidate_ranked_physical_rows": list(map(int, poisoned_authority["candidate_ranked_physical_rows"])),
            }
            require(projection == poisoned_projection, "source execution poison changed query-id join payload")
            source_join_projection.append(projection)
            poisoned_source_join_projection.append(poisoned_projection)
            records_by_query[query_id] = record
        shard_bindings.append(
            {
                "shard": shard,
                "payload_sha256": item["payload_sha256"],
                "receipt_sha256": item["receipt_sha256"],
                "validation_sha256": item["validation_sha256"],
                "source_payload_sha256": sha256_file(source_path),
            }
        )
    require(len(records_by_query) == 64, "current64 replay population drift")
    joined = join_query_ids_to_canonical_folds(tuple(records_by_query), canonical_ledger)
    source_join_sha256 = hashlib.sha256(
        json.dumps(source_join_projection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    poisoned_source_join_sha256 = hashlib.sha256(
        json.dumps(poisoned_source_join_projection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(source_join_sha256 == poisoned_source_join_sha256, "execution poison changed source join")
    return records_by_query, {
        "query_count": len(records_by_query),
        "token_byte_exact_count": len(records_by_query),
        "corrected_c128_exact_count": len(records_by_query),
        "candidate_score_max_abs": score_max_abs,
        "source_join_key": "query_id",
        "source_execution_ordinal_consumption_count": 0,
        "source_join_projection_sha256": source_join_sha256,
        "source_ignored_execution_mutation_replay_sha256": poisoned_source_join_sha256,
        "shards": shard_bindings,
        "canonical_fold_assignment_sha256": fold_assignment_sha256(joined),
    }, joined


def pair64_join(canonical_ledger: list[object]) -> tuple[dict[str, Any], tuple[Any, ...]]:
    validation = json.loads(PAIR64_VALIDATION.read_text())
    require(
        validation.get("status") == "ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED"
        and all(validation.get("checks", {}).values()),
        "Pair64 V2 validation drift",
    )
    require(
        sha256_file(PAIR64) == EXPECTED_PAIR64_SHA256
        and sha256_file(PAIR64_VALIDATION) == EXPECTED_PAIR64_VALIDATION_SHA256,
        "Pair64 frozen hash drift",
    )
    payload = torch.load(PAIR64, map_location="cpu", weights_only=False, mmap=True)
    records = payload.get("records", [])
    require(len(records) == 64, "Pair64 query population drift")
    # Deliberately extract one field only.  Neither inner_fold nor execution
    # ordinal is accepted by the canonical join API.
    query_ids = tuple(str(record["query_id"]) for record in records)
    joined = join_query_ids_to_canonical_folds(query_ids, canonical_ledger)
    poisoned_rows = [
        {"query_id": query_id, "inner_fold": 99, "execution_ordinal": -1}
        for query_id in query_ids
    ]
    poisoned = join_query_ids_to_canonical_folds(
        tuple(row["query_id"] for row in poisoned_rows), canonical_ledger
    )
    require(
        fold_assignment_sha256(joined) == fold_assignment_sha256(poisoned),
        "Pair64 ignored-field mutation changed canonical fold join",
    )
    return {
        "query_count": len(joined),
        "canonical_fold_assignment_sha256": fold_assignment_sha256(joined),
        "ignored_field_mutation_replay_sha256": fold_assignment_sha256(poisoned),
        "pair64_payload_deserialization_count": 1,
        "pair64_inner_fold_consumption_count": 0,
        "pair64_execution_ordinal_consumption_count": 0,
        "pair64_switch_label_consumption_count": 0,
        "pair64_target_identity_consumption_count": 0,
        "payload_sha256": sha256_file(PAIR64),
        "validation_sha256": sha256_file(PAIR64_VALIDATION),
    }, joined


def encode_fixture(
    spec: object,
    *,
    encoder: torch.nn.Module,
    processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(spec.path).resolve()
    require(sha256_file(path) == spec.source_image_sha256, "fixture source-image hash drift")
    with Image.open(path) as opened:
        orientation = int(opened.getexif().get(274, 1))
        if spec.track == "new_difficult_train":
            image = ImageOps.exif_transpose(opened).convert("RGB")
            decode_frame = "EXIF_ORIENTED_BEFORE_RESIZE"
        else:
            image = opened.convert("RGB")
            decode_frame = "DECODED_RAW_BEFORE_EXIF"
        inputs = processor.process_images([image]).to(device)
    with torch.inference_mode():
        encoded = encoder(**inputs)[0].float()
    image_mask = inputs["input_ids"][0] == processor.image_token_id
    require(image_mask.shape == encoded.shape[:1] and bool(image_mask.any()), "fixture image mask drift")
    image_tokens = encoded[image_mask].detach().half().cpu().contiguous()
    template_tokens = encoded[~image_mask].detach().half().cpu().contiguous()
    temporal, height, width = [int(item) for item in inputs["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    grid = (height // merge, width // merge)
    require(
        temporal == 1
        and grid == (int(spec.grid_h), int(spec.grid_w))
        and image_tokens.shape == (grid[0] * grid[1], 128)
        and template_tokens.ndim == 2
        and template_tokens.shape[1] == 128
        and bool(torch.isfinite(image_tokens).all())
        and bool(torch.isfinite(template_tokens).all()),
        "fixture current-runtime token/grid drift",
    )
    return {
        "query_id": spec.query_id,
        "query_ordinal": spec.query_ordinal,
        "heldout_fold": spec.heldout_fold,
        "track": spec.track,
        "source_image_sha256": spec.source_image_sha256,
        "source_exif_orientation": orientation,
        "decode_frame": decode_frame,
        "grid_shape": grid,
        "image_tokens": image_tokens,
        "image_tokens_sha256": tensor_sha256(image_tokens),
        "template_tokens_sha256": tensor_sha256(template_tokens),
    }


def fresh_fixture_replay(
    canonical_ledger: list[object], current_by_query: dict[str, dict]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("N2 E0 fresh current-runtime fixtures require CUDA")
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    device = torch.device("cuda")
    encoder = ColQwen2_5.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(
        str(MODEL), local_files_only=True
    )
    by_id = {spec.query_id: spec for spec in canonical_ledger}
    records: list[dict[str, Any]] = []
    forward_count = 0
    for query_id, track, expected_grid, decode_frame, expected_orientation, has_authority in FIXTURES:
        spec = by_id.get(query_id)
        require(spec is not None, f"fixture absent from canonical ledger: {query_id}")
        record = encode_fixture(spec, encoder=encoder, processor=processor, device=device)
        forward_count += 1
        require(
            record["track"] == track
            and tuple(record["grid_shape"]) == expected_grid
            and record["decode_frame"] == decode_frame
            and record["source_exif_orientation"] == expected_orientation,
            f"fixture scope drift: {query_id}",
        )
        if has_authority:
            authority = current_by_query.get(query_id)
            require(
                authority is not None
                and torch.equal(record["image_tokens"], authority["raw_image_tokens"])
                and record["image_tokens_sha256"] == tensor_sha256(authority["raw_image_tokens"])
                and record["template_tokens_sha256"] == tensor_sha256(authority["template_tokens"]),
                f"fresh/current64 token replay failed: {query_id}",
            )
            record["current64_token_byte_exact"] = True
            record["current64_template_token_byte_exact"] = True
        else:
            repeat = encode_fixture(spec, encoder=encoder, processor=processor, device=device)
            forward_count += 1
            require(
                torch.equal(record["image_tokens"], repeat["image_tokens"])
                and record["image_tokens_sha256"] == repeat["image_tokens_sha256"]
                and record["template_tokens_sha256"] == repeat["template_tokens_sha256"],
                "DIFFICULT-0025 repeated current-runtime forward is not byte-exact",
            )
            record["fresh_repeat_token_byte_exact"] = True
            record["fresh_repeat_image_tokens_sha256"] = repeat["image_tokens_sha256"]
            record["fresh_repeat_template_tokens_sha256"] = repeat["template_tokens_sha256"]
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "n2_e0_fixture_ready",
                    "query_id": query_id,
                    "track": track,
                    "grid": list(expected_grid),
                    "decode_frame": decode_frame,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    require(
        {tuple(record["grid_shape"]) for record in records}
        == {(36, 20), (24, 32), (32, 24), (25, 29)}
        and {record["track"] for record in records}
        == {"outcome", "difficult", "new_difficult_train"},
        "four-grid/three-track fixture closure failed",
    )
    return records, {
        "fixture_count": len(records),
        "model_load_count": 1,
        "model_forward_count": forward_count,
        "grids": sorted([list(item) for item in {tuple(r["grid_shape"]) for r in records}]),
        "tracks": dict(sorted(Counter(record["track"] for record in records).items())),
        "decode_frames": dict(sorted(Counter(record["decode_frame"] for record in records).items())),
    }


def _run_ready() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"immutable N2 E0 output exists: {OUT_ROOT}")
    validate_predecessors()
    canonical_ledger = load_query_ledger()
    ledger_receipt = validate_canonical_987_ledger(canonical_ledger)
    recipes, parameter_count, _ = validate_d1_recipe()
    corrected, reducer_receipt = load_corrected_gallery_and_reducer_fixture()
    current_by_query, current_receipt, current_join = load_current64_and_replay(
        corrected, canonical_ledger
    )
    pair_receipt, pair_join = pair64_join(canonical_ledger)
    fixture_records, fixture_receipt = fresh_fixture_replay(canonical_ledger, current_by_query)

    checks = {
        "validated_n2_design_predecessor": True,
        "frozen_d1_algorithm_recipe_and_five_seeds": True,
        "corrected_5413_to_5412_max_tie_c128_reducer": True,
        "legacy_5404_axis_and_reducer_fail_closed": True,
        "canonical_987_query_id_fold_ledger": True,
        "pair64_64_of_64_query_id_join_inner_fold_disabled": True,
        "current64_64_of_64_query_id_join": True,
        "current64_all_token_axis_score_replay": True,
        "four_grids_three_tracks_and_track_specific_exif": True,
        "difficult_0025_25x29_fresh_repeat": True,
        "target_external_sealed_and_training_access_zero": True,
    }
    payload_value = {
        "schema_version": "routea_matched_three_arm_n2_e0_payload_v1_20260902",
        "status": "ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_READY",
        "claim_level": "ENGINEERING_DESIGN_AND_LINEAGE_PREFLIGHT_ONLY_NO_TRAINING_NO_EXTERNAL_ACCESS",
        "checks": checks,
        "d1_recipe": {
            "fold_recipes": recipes,
            "adapter_parameter_count": parameter_count,
        },
        "corrected_gallery_reducer": reducer_receipt,
        "canonical_ledger": ledger_receipt,
        "pair64_join": pair_receipt,
        "current64_replay": current_receipt,
        "fixtures": fixture_records,
        "fixture_summary": fixture_receipt,
        "bindings": {
            "contract_sha256": sha256_file(CONTRACT),
            "design_sha256": sha256_file(DESIGN),
            "design_validation_sha256": sha256_file(DESIGN_VALIDATION),
            "query_ledger_sha256": sha256_file(QUERY_LEDGER),
            "upstream_fold_authority_sha256": sha256_file(UPSTREAMS),
            "current64_validation_sha256": sha256_file(CURRENT64_VALIDATION),
            "current64_source_validation_sha256": sha256_file(CURRENT_SOURCE / "validation.json"),
            "pair64_payload_sha256": sha256_file(PAIR64),
            "pair64_validation_sha256": sha256_file(PAIR64_VALIDATION),
            "gallery_cache_sha256": sha256_file(GALLERY_CACHE),
            "identity_repair_contract_sha256": sha256_file(IDENTITY_REPAIR_CONTRACT),
            "identity_repair_registry_sha256": sha256_file(IDENTITY_REPAIR_REGISTRY),
            "runtime_sha256": sha256_file(RUNTIME),
            "producer_sha256": sha256_file(Path(__file__).resolve()),
            "validator_sha256": sha256_file(VALIDATOR),
            "frozen_source_hashes": {
                str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): value
                for path, value in EXPECTED_HASHES.items()
            },
        },
        "access": {
            "internal_query_pixel_decode_count": 5,
            "current_runtime_encoder_load_count": 1,
            "current_runtime_encoder_forward_count": fixture_receipt["model_forward_count"],
            "d1_adapter_construction_count": 1,
            "d1_checkpoint_load_count": 0,
            "d1_model_update_count": 0,
            "action_head_model_update_count": 0,
            "pair64_training_artifact_read_count": 1,
            "pair64_switch_label_consumption_count": 0,
            "pair64_target_identity_consumption_count": 0,
            "target_label_join_count": 0,
            "external_query_read_count": 0,
            "external_target_read_count": 0,
            "external_outcome_read_count": 0,
            "sealed_read_count": 0,
        },
    }
    result_value = {
        "schema_version": "routea_matched_three_arm_n2_e0_result_v1_20260902",
        "status": payload_value["status"],
        "claim_level": payload_value["claim_level"],
        "checks": checks,
        "payload_sha256": "",
        "bindings": payload_value["bindings"],
        "access": payload_value["access"],
        "scientific_GO_or_NO_GO": None,
        "n2_training_authorized": False,
        "external_execution_authorized": False,
        "next_authorized_stage": "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT",
        "logical_sha256": "",
    }
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".n2-e0-v1-", dir=OUT_ROOT.parent))
    try:
        payload_path = staging / "payload.pt"
        result_path = staging / "result.json"
        torch.save(payload_value, payload_path)
        result_value["payload_sha256"] = sha256_file(payload_path)
        result_value["logical_sha256"] = logical_sha256(result_value)
        with result_path.open("w") as handle:
            json.dump(result_value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(staging, OUT_ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(
        json.dumps(
            {
                "status": result_value["status"],
                "checks": checks,
                "current64_replay": current_receipt["query_count"],
                "pair64_join": pair_receipt["query_count"],
                "fixtures": fixture_receipt,
                "next_authorized_stage": result_value["next_authorized_stage"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _publish_abort(error: Exception) -> None:
    if OUT_ROOT.exists():
        return
    value = {
        "schema_version": "routea_matched_three_arm_n2_e0_result_v1_20260902",
        "status": "ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_ABORT",
        "claim_level": "ENGINEERING_FAIL_CLOSED_NO_TRAINING_NO_EXTERNAL_ACCESS",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "protected_access": {
            "d1_checkpoint_load_count": 0,
            "d1_model_update_count": 0,
            "action_head_model_update_count": 0,
            "target_label_join_count": 0,
            "external_query_read_count": 0,
            "external_target_read_count": 0,
            "external_outcome_read_count": 0,
            "sealed_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "n2_training_authorized": False,
        "external_execution_authorized": False,
        "next_authorized_stage": None,
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".n2-e0-v1-abort-", dir=OUT_ROOT.parent))
    try:
        with (staging / "result.json").open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(staging, OUT_ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"status": value["status"], "error_type": value["error_type"], "next_authorized_stage": None}, sort_keys=True), flush=True)


def main() -> None:
    try:
        _run_ready()
    except Exception as error:
        _publish_abort(error)
        raise


if __name__ == "__main__":
    main()
