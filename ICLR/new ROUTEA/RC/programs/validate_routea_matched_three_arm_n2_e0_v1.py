#!/usr/bin/env python3
"""Independent validation of the N2 E0 corrected-gallery/current-runtime gate."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
for search_path in reversed(
    (ROOT / "src", ROOT / "programs", ROUTEA / "route_a_core", ROUTEA)
):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_a0_fold_d1 as d1_wrapper  # noqa: E402
from rc_aslo_xf.gallery_identity_repair import (  # noqa: E402
    CONTRACT as IDENTITY_REPAIR_CONTRACT,
    REGISTRY as IDENTITY_REPAIR_REGISTRY,
    build_identity_map,
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
PRODUCER = ROOT / "programs/run_routea_matched_three_arm_n2_e0_v1.py"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_n2_e0_v1"
PAYLOAD = OUT_ROOT / "payload.pt"
RESULT = OUT_ROOT / "result.json"
OUT = OUT_ROOT / "independent_validation.json"
WORKSPACE = ROOT.parents[2]
MODEL = WORKSPACE / "models/downloaded_models/colnomic-embed-multimodal-7b"
GALLERY_CACHE = WORKSPACE / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"

EXPECTED_DESIGN_SHA256 = "0b796b5a88481d7189ae026fc4e3464622f24f4e15b043e1e919a02f3df15ebd"
EXPECTED_DESIGN_VALIDATION_SHA256 = "5681b3863c56bb8b7fa7ebd02f16c4253ba2412d4cc4ea3cd216bb90e420e0a5"
EXPECTED_QUERY_LEDGER_SHA256 = "df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec"
EXPECTED_UPSTREAMS_SHA256 = "6f0399baa46a0c774a7214486467320eb0e778fe91cefd2b2010bc7512f75eac"
EXPECTED_CURRENT64_VALIDATION_SHA256 = "6c031e5a82b8057a48b48794d53dd76f4888b76ed03f081a925425c64e9621cb"
EXPECTED_SOURCE_HASHES = {
    ROOT / "programs/run_a0_fold_d1.py": "d8d58a0d96e1c3bc8457cf16f8b6aa0be46ceb7893a0cb6c0b75c50042eb0174",
    ROUTEA / "programs/run_routea_v3_1_c6direct_m1.py": "cdbac5833699c5904475cc308d1fbae6587a94765f9096d993093b2c1f621e41",
    ROUTEA / "route_a_core/route_a/o1_c6direct_m1_d1.py": "a3973e45f97f2157883acb96b64c614d9fa93ec80187e6608f462fe841737274",
    GALLERY_CACHE: "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    PAIR64: "d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3",
    PAIR64_VALIDATION: "657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b",
}
FIXTURE_SPECS = {
    "OUTCOME-0533": ("outcome", (36, 20), "DECODED_RAW_BEFORE_EXIF", 1, True),
    "DIFFICULT-0128": ("difficult", (24, 32), "DECODED_RAW_BEFORE_EXIF", 6, True),
    "NDV2-007-P01": ("new_difficult_train", (32, 24), "EXIF_ORIENTED_BEFORE_RESIZE", 6, True),
    "DIFFICULT-0025": ("difficult", (25, 29), "DECODED_RAW_BEFORE_EXIF", 1, False),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: dict[str, Any]) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "logical_sha256"})


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


def independent_reduce(
    scores: torch.Tensor, labels: Sequence[str]
) -> tuple[list[str], list[float], list[int]]:
    require(
        scores.shape == (5413,)
        and scores.is_floating_point()
        and bool(torch.isfinite(scores).all()),
        "independent reducer score schema drift",
    )
    positions: dict[str, list[int]] = defaultdict(list)
    for row, label in enumerate(labels):
        positions[str(label)].append(row)
    require(
        len(positions) == 5412
        and {label: rows for label, rows in positions.items() if len(rows) > 1}
        == {"Biogen_21": [714, 715]},
        "independent corrected identity population drift",
    )
    reduced = []
    for label, rows in positions.items():
        maximum = max(float(scores[row]) for row in rows)
        representative = min(row for row in rows if float(scores[row]) == maximum)
        reduced.append((label, maximum, representative))
    reduced.sort(key=lambda item: (-item[1], item[2]))
    return (
        [item[0] for item in reduced],
        [item[1] for item in reduced],
        [item[2] for item in reduced],
    )


def load_ledger() -> tuple[list[dict], dict[str, dict], dict[str, Any]]:
    document = json.loads(QUERY_LEDGER.read_text())
    rows = document.get("queries", [])
    require(
        document.get("version") == "l0_natural_hardneg_v2_targetfree_query_ledger_v1"
        and document.get("query_count") == 987
        and len(rows) == 987
        and document.get("target_label_read") is False
        and document.get("logical_sha256")
        == canonical_sha256({key: value for key, value in document.items() if key != "logical_sha256"}),
        "independent canonical ledger envelope drift",
    )
    require(
        all(int(row["query_ordinal"]) == ordinal for ordinal, row in enumerate(rows))
        and len({str(row["query_id"]) for row in rows}) == 987
        and [str(row["query_id"]) for row in rows]
        == sorted(str(row["query_id"]) for row in rows)
        and Counter(int(row["heldout_fold"]) for row in rows)
        == Counter({0: 212, 1: 205, 2: 189, 3: 188, 4: 193}),
        "independent canonical ledger population drift",
    )
    by_id = {str(row["query_id"]): row for row in rows}
    receipt = {
        "query_count": 987,
        "fold_counts": dict(sorted(Counter(int(row["heldout_fold"]) for row in rows).items())),
        "track_counts": dict(sorted(Counter(str(row["track"]) for row in rows).items())),
        "query_id_to_fold_sha256": canonical_sha256(
            [[row["query_id"], row["query_ordinal"], row["heldout_fold"]] for row in rows]
        ),
    }
    return rows, by_id, receipt


def assignment_hash(query_ids: Sequence[str], by_id: dict[str, dict]) -> str:
    require(len(query_ids) == len(set(query_ids)), "independent join input duplicate")
    output = []
    for query_id in query_ids:
        row = by_id.get(query_id)
        require(row is not None, "independent join member absent")
        output.append(
            {
                "query_id": query_id,
                "query_ordinal": int(row["query_ordinal"]),
                "heldout_fold": int(row["heldout_fold"]),
                "track": str(row["track"]),
                "grid_h": int(row["grid_h"]),
                "grid_w": int(row["grid_w"]),
                "source_image_sha256": str(row["source_image_sha256"]),
            }
        )
    return canonical_sha256(output)


def validate_recipe(payload: dict) -> bool:
    fold_recipes = payload.get("d1_recipe", {}).get("fold_recipes", [])
    require(len(fold_recipes) == 5, "producer D1 recipe population drift")
    for fold in range(5):
        module = d1_wrapper.import_runner()
        d1_wrapper.configure_fold(module, fold)
        expected = module.d1.frozen_recipe()
        require(fold_recipes[fold] == expected, f"independent D1 recipe replay failed fold {fold}")
        require(expected["effective_seed"] == 17 + 1009 * fold, "fold seed drift")
        if fold == 0:
            adapter = module.d1.build_fresh_d1_adapter()
            require(
                sum(parameter.numel() for parameter in adapter.parameters())
                == payload["d1_recipe"]["adapter_parameter_count"]
                == 49_792,
                "independent D1 parameter replay failed",
            )
    return True


def validate_reducer(payload: dict) -> tuple[tuple[str, ...], bool]:
    gallery = torch.load(GALLERY_CACHE, map_location="cpu", weights_only=False, mmap=True)
    legacy = tuple(map(str, gallery["setids"]))
    identity_map = build_identity_map(legacy)
    corrected = tuple(identity_map.labels)
    require(len(legacy) == 5413 and len(set(legacy)) == 5404, "independent legacy population drift")
    require(len(corrected) == 5413 and len(set(corrected)) == 5412, "independent corrected population drift")
    high = torch.zeros(5413, dtype=torch.float64)
    high[714], high[715] = 5.0, 6.0
    _, scores, rows = independent_reduce(high, corrected)
    biogen_rank = next(index for index, row in enumerate(rows) if row == 715)
    require(scores[biogen_rank] == 6.0, "independent max-row fixture failed")
    high[714] = 6.0
    _, _, tied_rows = independent_reduce(high, corrected)
    require(tied_rows[0] == 714, "independent duplicate tie fixture failed")
    _, _, all_tied_rows = independent_reduce(torch.zeros(5413, dtype=torch.float64), corrected)
    require(all_tied_rows[:128] == list(range(128)), "independent all-identity tie fixture failed")
    receipt = payload.get("corrected_gallery_reducer", {})
    require(
        receipt.get("physical_row_count") == 5413
        and receipt.get("corrected_identity_count") == 5412
        and receipt.get("max_row_representative") == 715
        and receipt.get("duplicate_tie_representative") == 714
        and receipt.get("legacy_5404_axis_rejected") is True
        and receipt.get("historical_full_gallery_reducer_rejected_corrected_5412") is True
        and receipt.get("identity_repair_contract_sha256") == sha256_file(IDENTITY_REPAIR_CONTRACT)
        and receipt.get("identity_repair_registry_sha256") == sha256_file(IDENTITY_REPAIR_REGISTRY),
        "producer corrected reducer receipt drift",
    )
    module = d1_wrapper.import_runner()
    historical_rejected = False
    try:
        module.d1.aggregate_full_gallery_exact_label_scores(torch.zeros(5413), corrected)
    except ValueError:
        historical_rejected = True
    require(historical_rejected, "independent legacy reducer did not reject corrected axis")
    return corrected, True


def validate_current64(
    payload: dict, corrected: tuple[str, ...], by_id: dict[str, dict]
) -> tuple[dict[str, dict], bool]:
    aggregate = json.loads(CURRENT64_VALIDATION.read_text())
    sealed = {int(item["shard"]): item for item in aggregate.get("shards", [])}
    source_validation = json.loads((CURRENT_SOURCE / "validation.json").read_text())
    source_seals = {
        int(item["shard"]): str(item["payload_sha256"])
        for item in source_validation.get("shards", [])
        if item.get("pass") is True
    }
    require(
        aggregate.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and len(sealed) == 8
        and all(aggregate.get("checks", {}).values()),
        "independent current64 aggregate drift",
    )
    require(
        source_validation.get("status")
        == "ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_VALIDATION_PASS"
        and source_validation.get("target_label_read_count") == 0
        and all(source_validation.get("checks", {}).values())
        and set(source_seals) == set(range(8)),
        "independent current-runtime source validation drift",
    )
    records_by_query: dict[str, dict] = {}
    max_abs = 0.0
    source_projection: list[dict[str, Any]] = []
    poisoned_source_projection: list[dict[str, Any]] = []
    for shard in range(8):
        current_path = CURRENT64 / f"shard{shard:02d}/payload.pt"
        receipt_path = CURRENT64 / f"shard{shard:02d}/receipt.json"
        validation_path = CURRENT64 / f"shard{shard:02d}/validation.json"
        source_path = CURRENT_SOURCE / f"shard{shard:02d}/payload.pt"
        require(
            sealed[shard]["payload_sha256"] == sha256_file(current_path)
            and sealed[shard]["receipt_sha256"] == sha256_file(receipt_path)
            and sealed[shard]["validation_sha256"] == sha256_file(validation_path),
            "independent current64 shard seal drift",
        )
        require(
            source_seals[shard] == sha256_file(source_path),
            "independent current-runtime source shard seal drift",
        )
        current = torch.load(current_path, map_location="cpu", weights_only=False, mmap=True)
        source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        source_by_query = {str(row["query_id"]): row for row in source["records"]}
        require(len(source_by_query) == 8, "independent source query-id uniqueness drift")
        poisoned_source_by_query = {
            query_id: {**row, "execution_ordinal": -(index + 1)}
            for index, (query_id, row) in enumerate(source_by_query.items())
        }
        for record in current["records"]:
            query_id = str(record["query_id"])
            authority = source_by_query.get(query_id)
            require(authority is not None, "independent source query-id join failed")
            require(query_id not in records_by_query and query_id in by_id, "current64 independent join drift")
            canonical = by_id[query_id]
            raw_scores = torch.as_tensor(record["raw_full_gallery_scores"])
            _, _, ranked_rows = independent_reduce(raw_scores, corrected)
            authority_axis = tuple(map(int, authority["candidate_physical_rows"]))
            authority_ranked = tuple(map(int, authority["candidate_ranked_physical_rows"]))
            observed = torch.tensor([float(raw_scores[row]) for row in authority_axis], dtype=torch.float64)
            expected = torch.as_tensor(authority["candidate_raw_scores"], dtype=torch.float64)
            delta = float((observed - expected).abs().max())
            max_abs = max(max_abs, delta)
            require(
                query_id == authority["query_id"]
                and record["track"] == authority["track"] == canonical["track"]
                and int(record["historical_query_ordinal"]) == int(canonical["query_ordinal"])
                and int(record["heldout_fold"]) == int(canonical["heldout_fold"])
                and tuple(record["query_grid_shape"]) == tuple(authority["query_grid_shape"])
                == (int(canonical["grid_h"]), int(canonical["grid_w"]))
                and Path(record["query_source_path"]).resolve() == Path(canonical["path"]).resolve()
                and record["query_source_sha256"] == authority["query_source_sha256"]
                == canonical["source_image_sha256"]
                and torch.equal(record["raw_image_tokens"], authority["query_tokens"])
                and tensor_sha256(record["raw_image_tokens"]) == authority["query_tokens_sha256"]
                and tuple(ranked_rows[:128]) == authority_ranked
                and tuple(sorted(ranked_rows[:128])) == authority_axis
                and delta <= 1e-6,
                f"independent current64 replay failed: {query_id}",
            )
            projection = {
                "query_id": query_id,
                "track": str(authority["track"]),
                "grid": list(map(int, authority["query_grid_shape"])),
                "query_tokens_sha256": str(authority["query_tokens_sha256"]),
                "candidate_physical_rows": list(map(int, authority["candidate_physical_rows"])),
                "candidate_ranked_physical_rows": list(map(int, authority["candidate_ranked_physical_rows"])),
            }
            poisoned = poisoned_source_by_query[query_id]
            poisoned_projection = {
                "query_id": query_id,
                "track": str(poisoned["track"]),
                "grid": list(map(int, poisoned["query_grid_shape"])),
                "query_tokens_sha256": str(poisoned["query_tokens_sha256"]),
                "candidate_physical_rows": list(map(int, poisoned["candidate_physical_rows"])),
                "candidate_ranked_physical_rows": list(map(int, poisoned["candidate_ranked_physical_rows"])),
            }
            require(projection == poisoned_projection, "independent source execution poison drift")
            source_projection.append(projection)
            poisoned_source_projection.append(poisoned_projection)
            records_by_query[query_id] = record
    qids = tuple(records_by_query)
    source_hash = canonical_sha256(source_projection)
    poison_hash = canonical_sha256(poisoned_source_projection)
    require(
        len(qids) == 64
        and payload["current64_replay"]["query_count"] == 64
        and payload["current64_replay"]["token_byte_exact_count"] == 64
        and payload["current64_replay"]["corrected_c128_exact_count"] == 64
        and payload["current64_replay"]["source_join_key"] == "query_id"
        and payload["current64_replay"]["source_execution_ordinal_consumption_count"] == 0
        and payload["current64_replay"]["source_join_projection_sha256"] == source_hash
        and payload["current64_replay"]["source_ignored_execution_mutation_replay_sha256"] == poison_hash
        and source_hash == poison_hash
        and abs(float(payload["current64_replay"]["candidate_score_max_abs"]) - max_abs) <= 1e-12
        and payload["current64_replay"]["canonical_fold_assignment_sha256"]
        == assignment_hash(qids, by_id),
        "producer current64 replay receipt drift",
    )
    return records_by_query, True


def validate_pair64(payload: dict, by_id: dict[str, dict]) -> bool:
    validation = json.loads(PAIR64_VALIDATION.read_text())
    pair = torch.load(PAIR64, map_location="cpu", weights_only=False, mmap=True)
    query_ids = tuple(str(record["query_id"]) for record in pair.get("records", []))
    expected_hash = assignment_hash(query_ids, by_id)
    receipt = payload.get("pair64_join", {})
    require(
        validation.get("status") == "ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED"
        and all(validation.get("checks", {}).values())
        and len(query_ids) == len(set(query_ids)) == 64
        and receipt.get("query_count") == 64
        and receipt.get("canonical_fold_assignment_sha256") == expected_hash
        and receipt.get("ignored_field_mutation_replay_sha256") == expected_hash
        and receipt.get("pair64_payload_deserialization_count") == 1
        and receipt.get("pair64_inner_fold_consumption_count") == 0
        and receipt.get("pair64_execution_ordinal_consumption_count") == 0
        and receipt.get("pair64_switch_label_consumption_count") == 0
        and receipt.get("pair64_target_identity_consumption_count") == 0
        and receipt.get("payload_sha256") == sha256_file(PAIR64)
        and receipt.get("validation_sha256") == sha256_file(PAIR64_VALIDATION),
        "independent Pair64 canonical join drift",
    )
    return True


def encode_fresh(row: dict, encoder: torch.nn.Module, processor: Any) -> tuple[torch.Tensor, torch.Tensor, str, tuple[int, int], int]:
    source_path = Path(row["path"]).resolve()
    require(sha256_file(source_path) == row["source_image_sha256"], "validator fixture source hash drift")
    with Image.open(source_path) as opened:
        orientation = int(opened.getexif().get(274, 1))
        if row["track"] == "new_difficult_train":
            image = ImageOps.exif_transpose(opened).convert("RGB")
            decode_frame = "EXIF_ORIENTED_BEFORE_RESIZE"
        else:
            image = opened.convert("RGB")
            decode_frame = "DECODED_RAW_BEFORE_EXIF"
        inputs = processor.process_images([image]).to("cuda")
    with torch.inference_mode():
        encoded = encoder(**inputs)[0].float()
    image_mask = inputs["input_ids"][0] == processor.image_token_id
    tokens = encoded[image_mask].detach().half().cpu().contiguous()
    template = encoded[~image_mask].detach().half().cpu().contiguous()
    _, height, width = [int(item) for item in inputs["image_grid_thw"][0].tolist()]
    merge = int(processor.image_processor.merge_size)
    return tokens, template, decode_frame, (height // merge, width // merge), orientation


def validate_fixtures(
    payload: dict, by_id: dict[str, dict], current_by_query: dict[str, dict]
) -> bool:
    records = payload.get("fixtures", [])
    by_query = {str(record.get("query_id")): record for record in records}
    require(set(by_query) == set(FIXTURE_SPECS), "fixture query population drift")
    require(torch.cuda.is_available(), "independent fixture validation requires CUDA")
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    encoder = ColQwen2_5.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, local_files_only=True
    ).cuda().eval()
    encoder.requires_grad_(False)
    processor = ColQwen2_5_Processor.from_pretrained(str(MODEL), local_files_only=True)
    validator_forward_count = 0
    for query_id, (track, grid, frame, expected_orientation, has_authority) in FIXTURE_SPECS.items():
        ledger_row = by_id[query_id]
        record = by_query[query_id]
        fresh, fresh_template, fresh_frame, fresh_grid, orientation = encode_fresh(ledger_row, encoder, processor)
        validator_forward_count += 1
        require(
            record["query_ordinal"] == ledger_row["query_ordinal"]
            and record["heldout_fold"] == ledger_row["heldout_fold"]
            and record["track"] == track == ledger_row["track"]
            and tuple(record["grid_shape"]) == fresh_grid == grid
            and record["decode_frame"] == fresh_frame == frame
            and record["source_exif_orientation"] == orientation == expected_orientation
            and record["source_image_sha256"] == ledger_row["source_image_sha256"]
            and torch.equal(record["image_tokens"], fresh)
            and record["image_tokens_sha256"] == tensor_sha256(fresh)
            and record["template_tokens_sha256"] == tensor_sha256(fresh_template),
            f"independent fresh fixture replay failed: {query_id}",
        )
        if has_authority:
            require(
                record.get("current64_token_byte_exact") is True
                and record.get("current64_template_token_byte_exact") is True
                and torch.equal(fresh, current_by_query[query_id]["raw_image_tokens"])
                and tensor_sha256(fresh_template) == tensor_sha256(current_by_query[query_id]["template_tokens"]),
                "fixture/current64 authority replay drift",
            )
        else:
            repeat, repeat_template, repeat_frame, repeat_grid, repeat_orientation = encode_fresh(
                ledger_row, encoder, processor
            )
            validator_forward_count += 1
            require(
                record.get("fresh_repeat_token_byte_exact") is True
                and torch.equal(fresh, repeat)
                and torch.equal(fresh_template, repeat_template)
                and fresh_frame == repeat_frame
                and fresh_grid == repeat_grid
                and orientation == repeat_orientation
                and record.get("fresh_repeat_image_tokens_sha256") == tensor_sha256(fresh)
                and record.get("fresh_repeat_template_tokens_sha256")
                == tensor_sha256(fresh_template),
                "DIFFICULT-0025 repeat receipt drift",
            )
    summary = payload.get("fixture_summary", {})
    require(
        summary.get("fixture_count") == 4
        and summary.get("model_load_count") == 1
        and summary.get("model_forward_count") == 5
        and {tuple(item) for item in summary.get("grids", [])}
        == {(36, 20), (24, 32), (32, 24), (25, 29)},
        "fixture summary drift",
    )
    require(validator_forward_count == 5, "independent fixture forward-count drift")
    return True


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable N2 E0 validation exists: {OUT}")
    result = json.loads(RESULT.read_text())
    payload = torch.load(PAYLOAD, map_location="cpu", weights_only=False, mmap=True)
    design_validation = json.loads(DESIGN_VALIDATION.read_text())
    checks: dict[str, bool] = {}
    checks["producer_envelope_and_payload_seal"] = (
        result.get("status")
        == payload.get("status")
        == "ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_READY"
        and result.get("logical_sha256") == logical_sha256(result)
        and result.get("payload_sha256") == sha256_file(PAYLOAD)
        and all(result.get("checks", {}).values())
        and all(payload.get("checks", {}).values())
    )
    checks["design_predecessor_and_static_lineage"] = (
        sha256_file(DESIGN) == EXPECTED_DESIGN_SHA256
        and sha256_file(DESIGN_VALIDATION) == EXPECTED_DESIGN_VALIDATION_SHA256
        and sha256_file(QUERY_LEDGER) == EXPECTED_QUERY_LEDGER_SHA256
        and sha256_file(UPSTREAMS) == EXPECTED_UPSTREAMS_SHA256
        and sha256_file(CURRENT64_VALIDATION) == EXPECTED_CURRENT64_VALIDATION_SHA256
        and all(sha256_file(path) == expected for path, expected in EXPECTED_SOURCE_HASHES.items())
        and design_validation.get("status") == "ROUTEA_MATCHED_THREE_ARM_N2_EXTERNAL_DESIGN_VALIDATED"
        and design_validation.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_D1_C_E0_DESIGN_AND_LINEAGE_PREFLIGHT"
    )
    bindings = payload.get("bindings", {})
    checks["producer_runtime_contract_and_validator_bindings"] = (
        bindings.get("contract_sha256") == sha256_file(CONTRACT)
        and bindings.get("design_sha256") == sha256_file(DESIGN)
        and bindings.get("design_validation_sha256") == sha256_file(DESIGN_VALIDATION)
        and bindings.get("query_ledger_sha256") == sha256_file(QUERY_LEDGER)
        and bindings.get("upstream_fold_authority_sha256") == sha256_file(UPSTREAMS)
        and bindings.get("current64_validation_sha256") == sha256_file(CURRENT64_VALIDATION)
        and bindings.get("current64_source_validation_sha256")
        == sha256_file(CURRENT_SOURCE / "validation.json")
        and bindings.get("pair64_payload_sha256") == sha256_file(PAIR64)
        and bindings.get("pair64_validation_sha256") == sha256_file(PAIR64_VALIDATION)
        and bindings.get("gallery_cache_sha256") == sha256_file(GALLERY_CACHE)
        and bindings.get("identity_repair_contract_sha256") == sha256_file(IDENTITY_REPAIR_CONTRACT)
        and bindings.get("identity_repair_registry_sha256") == sha256_file(IDENTITY_REPAIR_REGISTRY)
        and bindings.get("runtime_sha256") == sha256_file(RUNTIME)
        and bindings.get("producer_sha256") == sha256_file(PRODUCER)
        and bindings.get("validator_sha256") == sha256_file(Path(__file__).resolve())
        and result.get("bindings") == bindings
    )
    ledger_rows, by_id, ledger_receipt = load_ledger()
    checks["canonical_987_ledger_independent"] = payload.get("canonical_ledger") == ledger_receipt
    checks["frozen_d1_recipe_independent"] = validate_recipe(payload)
    corrected, checks["corrected_5412_reducer_independent"] = validate_reducer(payload)
    current_by_query, checks["current64_exact_replay_independent"] = validate_current64(
        payload, corrected, by_id
    )
    checks["pair64_query_id_only_join_independent"] = validate_pair64(payload, by_id)
    checks["four_grid_three_track_exif_fresh_replay_independent"] = validate_fixtures(
        payload, by_id, current_by_query
    )
    expected_access = {
        "internal_query_pixel_decode_count": 5,
        "current_runtime_encoder_load_count": 1,
        "current_runtime_encoder_forward_count": 5,
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
    }
    checks["protected_access_and_model_updates_zero"] = (
        payload.get("access") == result.get("access") == expected_access
        and result.get("scientific_GO_or_NO_GO") is None
        and result.get("n2_training_authorized") is False
        and result.get("external_execution_authorized") is False
        and result.get("next_authorized_stage")
        == "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT"
    )
    passed = all(checks.values())
    value = {
        "schema_version": "routea_matched_three_arm_n2_e0_independent_validation_v1_20260902",
        "status": (
            "ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_VALIDATED"
            if passed
            else "ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_VALIDATION_ABORT"
        ),
        "claim_level": "INDEPENDENT_ENGINEERING_PREFLIGHT_VALIDATION_ONLY",
        "checks": checks,
        "producer_result_sha256": sha256_file(RESULT),
        "producer_result_logical_sha256": result.get("logical_sha256"),
        "producer_payload_sha256": sha256_file(PAYLOAD),
        "validator_access": {
            "internal_query_pixel_decode_count": 5,
            "current_runtime_encoder_load_count": 1,
            "current_runtime_encoder_forward_count": 5,
            "d1_adapter_construction_count": 1,
            "d1_checkpoint_load_count": 0,
            "model_update_count": 0,
            "pair64_training_artifact_read_count": 1,
            "pair64_switch_label_consumption_count": 0,
            "pair64_target_identity_consumption_count": 0,
            "target_label_join_count": 0,
            "external_query_read_count": 0,
            "external_target_read_count": 0,
            "external_outcome_read_count": 0,
            "sealed_read_count": 0,
        },
        "scientific_GO_or_NO_GO": None,
        "n2_training_authorized": False,
        "external_execution_authorized": False,
        "next_authorized_stage": (
            "N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT"
            if passed
            else None
        ),
        "logical_sha256": "",
    }
    value["logical_sha256"] = logical_sha256(value)
    atomic_json(OUT, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "checks": checks,
                "next_authorized_stage": value["next_authorized_stage"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    raise SystemExit(0 if passed else 4)


def _publish_abort(error: Exception) -> None:
    if OUT.exists() or not OUT_ROOT.exists():
        return
    value = {
        "schema_version": "routea_matched_three_arm_n2_e0_independent_validation_v1_20260902",
        "status": "ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_VALIDATION_ABORT",
        "claim_level": "INDEPENDENT_ENGINEERING_PREFLIGHT_VALIDATION_FAIL_CLOSED",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "protected_access": {
            "d1_checkpoint_load_count": 0,
            "model_update_count": 0,
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
    atomic_json(OUT, value)
    print(json.dumps({"status": value["status"], "error_type": value["error_type"], "next_authorized_stage": None}, sort_keys=True), flush=True)


def main() -> None:
    try:
        _validate()
    except Exception as error:
        _publish_abort(error)
        raise


if __name__ == "__main__":
    main()
