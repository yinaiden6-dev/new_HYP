#!/usr/bin/env python3
"""Run DINO-RCDE V1.1 P0 without entering E0 or reading protected roles."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rc_dino_rcde_p0_result_v1_1_20260812"
READY = "DINO_RCDE_P0_READY"
PROTOCOL_ABORT = "DINO_RCDE_LOW_MARGIN_CONTRACT_UNFROZEN"
DATA_ABORT = "DINO_RCDE_DATA_ROLE_PREFLIGHT_ABORT"
NEGATIVE_ABORT = "DINO_RCDE_NEGATIVE_BAND_COVERAGE_INELIGIBLE"
HALL_ABORT = "N1_PERMUTATION_CONTROL_INELIGIBLE"
MATCHED_ABORT = "COLNOMIC_RCDE_MATCHED_CONTROL_INELIGIBLE"
STAT_ABORT = "DINO_RCDE_STATISTICALLY_INELIGIBLE"
RESOURCE_ABORT = "DINO_RCDE_RESOURCE_BLOCKED"
PROTECTED_ZERO = {
    "C8_runtime_read_count": 0,
    "opened_runtime_read_count": 0,
    "sealed_runtime_read_count": 0,
    "home_files_modified": 0,
}


class P0Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise P0Error(f"immutable artifact drift: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def exclusive_torch(path: Path, value: Any, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise P0Error(f"refusing to replace immutable tensor artifact: {path}")
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    torch.save(value, temporary)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_binding(binding: Mapping[str, Any]) -> Path:
    return (ROOT / str(binding["path"])).resolve()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    low = protocol.get("low_margin_contract", {})
    expected_forbidden = {
        "C128_construction", "negative_sampler", "loss", "model_feature",
        "eligibility", "HOLD", "A1_gate", "threshold_selection",
        "checkpoint_selection", "GO_primary_population",
    }
    checks = {
        "schema": protocol.get("schema_version") == "rc_dino_rcde_p0_protocol_v1_1_20260812",
        "status": protocol.get("status") == "FROZEN_BEFORE_NATURAL_RAW_SCORE_READ",
        "namespace": low.get("namespace") == "RAW_LOW_MARGIN_V1_1_Q25_MAD_C128",
        "median": low.get("median") == "stable_float64_sorted_average_indices_63_64",
        "mad": low.get("mad") == "stable_float64_sorted_average_indices_63_64",
        "quantile": low.get("quantile") == 0.25,
        "final_k": low.get("final_quantile_index") == 150,
        "forbidden": set(low.get("forbidden_uses", [])) == expected_forbidden,
        "pre_score": low.get("contract_must_validate_before_raw_score_read") is True,
        "no_advance": protocol.get("automatic_stage_advance") is False,
        "protected": protocol.get("protected_access_counts") == PROTECTED_ZERO,
    }
    if not all(checks.values()):
        raise P0Error(f"{PROTOCOL_ABORT}: {checks}")
    contract = resolve_binding({"path": protocol["execution_contract"]})
    if sha256_file(contract) != protocol.get("execution_contract_sha256"):
        raise P0Error(f"{PROTOCOL_ABORT}: execution contract hash drift")
    authority = protocol["parent_authority"]
    authority_path = resolve_binding(authority)
    if sha256_file(authority_path) != authority.get("sha256"):
        raise P0Error("parent authority hash drift")
    authority_value = json.loads(authority_path.read_text(encoding="utf-8"))
    if authority_value.get("core_addendum", {}).get("natural_training_authorized") is not False:
        raise P0Error("parent authority natural-training boundary drift")
    for name, binding in protocol.get("source_bindings", {}).items():
        source = resolve_binding(binding)
        if not source.is_file() or sha256_file(source) != binding.get("sha256"):
            raise P0Error(f"source binding drift: {name}")
    return protocol


def corrected_identities(setids: Sequence[str], repair: Mapping[str, Any]) -> list[str]:
    corrected = [str(value) for value in setids]
    for row in repair.get("filename_collision_overrides", []):
        corrected[int(row["physical_row"])] = str(row["corrected_identity"])
    for component in repair.get("verified_byte_identical_duplicate_components", []):
        identity = str(component["canonical_identity"])
        for physical_row in component["physical_rows"]:
            corrected[int(physical_row)] = identity
    return corrected


def reduce_one(scores: Any, corrected: Sequence[str]) -> tuple[list[int], list[float]]:
    winners: dict[str, tuple[float, int]] = {}
    for row, identity in enumerate(corrected):
        score = float(scores[row])
        current = winners.get(identity)
        if current is None or score > current[0] or (score == current[0] and row < current[1]):
            winners[identity] = (score, row)
    ordered = sorted(winners.items(), key=lambda item: (-item[1][0], item[0]))[:128]
    return [item[1][1] for item in ordered], [item[1][0] for item in ordered]


def robust_confidence(scores: Sequence[float]) -> float:
    if len(scores) != 128 or not all(math.isfinite(float(value)) for value in scores):
        raise P0Error("low-margin requires 128 finite scores")
    ordered = sorted(float(value) for value in scores)
    median = (ordered[63] + ordered[64]) * 0.5
    deviations = sorted(abs(value - median) for value in ordered)
    mad = (deviations[63] + deviations[64]) * 0.5
    sigma = 1.4826 * mad + 1.0e-6
    value = (float(scores[0]) - float(scores[1])) / sigma
    if not math.isfinite(value) or value < 0.0:
        raise P0Error("invalid RAW confidence")
    return value


def score_query(query: Any, gallery: Sequence[Any], device: Any, batch_size: int, torch: Any) -> Any:
    chunks = []
    query = query.float().to(device)
    for start in range(0, len(gallery), batch_size):
        refs = [value.float().to(device) for value in gallery[start:start + batch_size]]
        lengths = torch.tensor([value.shape[0] for value in refs], device=device)
        padded = torch.nn.utils.rnn.pad_sequence(refs, batch_first=True)
        valid = torch.arange(padded.shape[1], device=device)[None, :] < lengths[:, None]
        similarity = torch.einsum("pd,bqd->pbq", query, padded)
        similarity = similarity.masked_fill(~valid[None, :, :], float("-inf"))
        chunks.append(similarity.amax(dim=-1).sum(dim=0).cpu())
    result = torch.cat(chunks).float()
    if result.shape != (5413,) or not bool(torch.isfinite(result).all()):
        raise P0Error("invalid full-gallery RAW score vector")
    return result


def hall_feasible(groups: Sequence[str]) -> tuple[bool, dict[str, Any]]:
    counts = Counter(groups)
    total = sum(counts.values())
    maximum = max(counts.values(), default=0)
    ok = len(counts) >= 2 and maximum <= total - maximum
    return ok, {"records": total, "groups": len(counts), "maximum_group_records": maximum}


def sign_power(number: int) -> dict[str, Any]:
    if number <= 0:
        return {"groups": number, "minimum_rejection_positives": None, "minimum_p": None, "power80_p": None}
    rejection = number + 1
    for positive in range(number + 1):
        tail = sum(math.comb(number, k) for k in range(positive, number + 1)) / (2.0 ** number)
        if tail <= 0.05:
            rejection = positive
            break
    minimum_p = 2.0 ** (-number)
    p80 = None
    if rejection <= number:
        for step in range(5001, 10001):
            probability = step / 10000.0
            power = sum(
                math.comb(number, k) * probability ** k * (1.0 - probability) ** (number - k)
                for k in range(rejection, number + 1)
            )
            if power >= 0.8:
                p80 = probability
                break
    return {
        "groups": number,
        "minimum_rejection_positives": rejection if rejection <= number else None,
        "minimum_p": minimum_p,
        "power80_p": p80,
    }


def shape_for_image(path: Path) -> dict[str, int]:
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
        try:
            value = ImageOps.exif_transpose(image)
            width, height = value.size
            exif_metadata_fallback = 0
        except TypeError:
            if not isinstance(orientation, int) or orientation not in range(1, 9):
                raise P0Error(f"unparseable EXIF orientation: {path}")
            width, height = image.size
            if orientation in (5, 6, 7, 8):
                width, height = height, width
            exif_metadata_fallback = 1
    scale = 518.0 / max(height, width)
    resized_h = max(1, math.floor(height * scale + 0.5))
    resized_w = max(1, math.floor(width * scale + 0.5))
    padded_h = ((resized_h + 13) // 14) * 14
    padded_w = ((resized_w + 13) // 14) * 14
    return {
        "raw_h": height, "raw_w": width,
        "exif_orientation": int(orientation),
        "exif_metadata_fallback": exif_metadata_fallback,
        "resized_h": resized_h, "resized_w": resized_w,
        "grid_h": padded_h // 14, "grid_w": padded_w // 14,
        "valid_h": resized_h // 14, "valid_w": resized_w // 14,
        "valid_tokens": (resized_h // 14) * (resized_w // 14),
        "total_tokens": (padded_h // 14) * (padded_w // 14),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--resume-prejoin-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise P0Error(f"output already exists: {output}")
    output.mkdir(parents=True)
    started_ns = time.time_ns()
    try:
        protocol = load_protocol(args.protocol.resolve())
    except Exception as error:
        result = {
            "schema_version": SCHEMA, "status": PROTOCOL_ABORT,
            "failure_domain": "protocol", "next_authorized_stage": None,
            "RAW_score_read_count": 0, "error_type": type(error).__name__,
            "error": str(error), "protected_access_counts": dict(PROTECTED_ZERO),
        }
        exclusive_json(output / "result.json", result)
        return 2

    protocol_sha = sha256_file(args.protocol.resolve())
    contract_receipt = {
        "schema_version": "rc_dino_rcde_p0_machine_contract_v1_1_20260812",
        "status": "P0_MACHINE_CONTRACT_VALIDATED_BEFORE_RAW_SCORE_READ",
        "created_ns": time.time_ns(), "protocol_sha256": protocol_sha,
        "execution_contract_sha256": protocol["execution_contract_sha256"],
        "low_margin_contract": protocol["low_margin_contract"],
        "base_contract": protocol["base_contract"],
        "decision_precedence": protocol["decision_precedence"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "validator_path": "programs/validate_dino_rcde_p0_v1_1.py",
        "RAW_score_read_count": 0, "label_read_count": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    exclusive_json(output / "contract.json", contract_receipt)

    import torch
    if args.device != "cuda" or not torch.cuda.is_available() or args.score_batch_size < 1:
        raise P0Error("formal P0 RAW replay requires CUDA and positive score batch size")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    bindings = protocol["source_bindings"]
    runtime = json.loads(resolve_binding(bindings["target_free_runtime_manifest"]).read_text(encoding="utf-8"))
    target_free = sorted(
        [row for row in runtime["records"] if str(row["materialization_role"]).startswith("TRAIN_INNER_FOLD_")],
        key=lambda row: int(row["query_ordinal"]),
    )
    optimization = json.loads(resolve_binding(bindings["optimization_materialization"]).read_text(encoding="utf-8"))
    optimization_by_id = {str(row["query_id"]): row for row in optimization["rows"]}
    if len(target_free) != 600 or len({row["query_id"] for row in target_free}) != 600:
        raise P0Error("target-free 600-role drift")

    redacted_dir = output / "redacted_query_tokens"
    redacted_rows = []
    for ordinal, row in enumerate(target_free):
        query_id = str(row["query_id"])
        source = optimization_by_id.get(query_id)
        if source is None or source.get("source_image_sha256") != row.get("source_image_sha256"):
            raise P0Error(f"query source binding drift: {query_id}")
        source_path = Path(str(source["optimization_shard_path"]))
        if sha256_file(source_path) != source.get("optimization_shard_sha256"):
            raise P0Error(f"query shard hash drift: {query_id}")
        carrier = torch.load(source_path, map_location="cpu", weights_only=False)
        image_tokens = carrier.get("image_tokens")
        template_tokens = carrier.get("template_tokens")
        if (
            not isinstance(image_tokens, torch.Tensor) or not isinstance(template_tokens, torch.Tensor)
            or image_tokens.ndim != 2 or template_tokens.ndim != 2
            or image_tokens.shape[1] != 128 or template_tokens.shape[1] != 128
            or int(carrier.get("grid_h", -1)) * int(carrier.get("grid_w", -1)) != image_tokens.shape[0]
        ):
            raise P0Error(f"query token schema drift: {query_id}")
        redacted = {
            "schema_version": "rcde_redacted_colnomic_query_tokens_v1_1",
            "query_id": query_id,
            "query_ordinal": int(row["query_ordinal"]),
            "source_image_sha256": str(row["source_image_sha256"]),
            "image_tokens": image_tokens.detach().cpu().contiguous(),
            "template_tokens": template_tokens.detach().cpu().contiguous(),
            "grid_h": int(carrier["grid_h"]), "grid_w": int(carrier["grid_w"]),
        }
        shard = redacted_dir / f"query_{ordinal:04d}.pt"
        exclusive_torch(shard, redacted, torch)
        redacted_rows.append({
            "query_id": query_id, "query_ordinal": int(row["query_ordinal"]),
            "source_image_sha256": str(row["source_image_sha256"]),
            "source_carrier_sha256": str(source["optimization_shard_sha256"]),
            "redacted_shard": str(shard), "redacted_shard_sha256": sha256_file(shard),
            "image_token_sha256": tensor_sha256(image_tokens),
            "template_token_sha256": tensor_sha256(template_tokens),
            "grid_h": int(carrier["grid_h"]), "grid_w": int(carrier["grid_w"]),
            "source_carrier_forbidden_fields_present": sorted(
                set(carrier) & {"identity", "raw_full_gallery_rank", "raw_full_gallery_gt_score", "hard_negative_indices", "hard_negative_ids"}
            ),
        })
    redacted_manifest = {
        "schema_version": "rcde_redacted_colnomic_query_manifest_v1_1",
        "status": "RCDE_REDACTED_QUERY_TOKEN_CACHE_FROZEN",
        "query_count": len(redacted_rows),
        "source_carrier_payload_read_count": len(redacted_rows),
        "source_carrier_score_or_label_field_use_count": 0,
        "rows": redacted_rows,
        "logical_sha256": canonical_sha256(redacted_rows),
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    exclusive_json(output / "redacted_query_token_manifest.json", redacted_manifest)

    gallery_payload = torch.load(resolve_binding(bindings["gallery_token_cache"]), map_location="cpu", weights_only=False)
    gallery = list(gallery_payload["passage_emb"])
    setids = [str(value) for value in gallery_payload["setids"]]
    repair = json.loads(resolve_binding(bindings["gallery_identity_repair"]).read_text(encoding="utf-8"))
    corrected = corrected_identities(setids, repair)
    if len(gallery) != 5413 or len(set(corrected)) != 5412:
        raise P0Error("corrected gallery cardinality drift")

    resume_source = None
    if args.resume_prejoin_dir is not None:
        resume_source = args.resume_prejoin_dir.resolve()
        source_contract = json.loads((resume_source / "contract.json").read_text(encoding="utf-8"))
        source_receipt = json.loads((resume_source / "base_prejoin_receipt.json").read_text(encoding="utf-8"))
        source_prejoin_path = resume_source / "base_prejoin.pt"
        if (
            source_contract.get("protocol_sha256") != protocol_sha
            or source_receipt.get("base_prejoin_sha256") != sha256_file(source_prejoin_path)
        ):
            raise P0Error("resume prejoin binding drift")
        source_prejoin = torch.load(source_prejoin_path, map_location="cpu", weights_only=False)
        if source_prejoin.get("query_ids") != [row["query_id"] for row in redacted_rows]:
            raise P0Error("resume prejoin query sequence drift")
        all_scores = source_prejoin["physical_scores"].detach().clone()
        c128_rows = source_prejoin["c128_rows_by_raw_rank"].detach().clone()
        c128_scores = source_prejoin["c128_scores_by_raw_rank"].detach().clone()
        if (
            all_scores.shape != (600, 5413) or c128_rows.shape != (600, 128)
            or c128_scores.shape != (600, 128)
        ):
            raise P0Error("resume prejoin tensor shape drift")
        scoring_seconds = 0.0
        print(json.dumps({"stage": "raw_replay_exact_resume", "queries": 600,
                          "source": str(resume_source)}), flush=True)
    else:
        all_scores = torch.empty((600, 5413), dtype=torch.float32)
        c128_rows = torch.empty((600, 128), dtype=torch.int64)
        c128_scores = torch.empty((600, 128), dtype=torch.float64)
        scoring_started = time.perf_counter()
        for index, item in enumerate(redacted_rows):
            payload = torch.load(item["redacted_shard"], map_location="cpu", weights_only=False)
            query = torch.cat((payload["image_tokens"], payload["template_tokens"]), dim=0)
            scores = score_query(query, gallery, device, args.score_batch_size, torch)
            rows, values = reduce_one(scores.tolist(), corrected)
            all_scores[index] = scores
            c128_rows[index] = torch.tensor(rows, dtype=torch.int64)
            c128_scores[index] = torch.tensor(values, dtype=torch.float64)
            if (index + 1) % 10 == 0 or index + 1 == 600:
                print(json.dumps({"stage": "raw_replay", "done": index + 1, "total": 600}), flush=True)
        scoring_seconds = time.perf_counter() - scoring_started
    prejoin = {
        "schema_version": "rcde_colnomic_raw_c128_prejoin_v1_1",
        "query_ids": [row["query_id"] for row in redacted_rows],
        "query_ordinals": [row["query_ordinal"] for row in redacted_rows],
        "source_image_sha256": [row["source_image_sha256"] for row in redacted_rows],
        "physical_scores": all_scores,
        "c128_rows_by_raw_rank": c128_rows,
        "c128_scores_by_raw_rank": c128_scores,
        "corrected_identity_logical_sha256": canonical_sha256(corrected),
    }
    exclusive_torch(output / "base_prejoin.pt", prejoin, torch)
    prejoin_receipt = {
        "schema_version": "rcde_colnomic_raw_c128_prejoin_receipt_v1_1",
        "status": "RCDE_COLNOMIC_RAW_C128_PREJOIN_FROZEN",
        "created_ns": time.time_ns(), "query_count": 600,
        "physical_gallery_rows": 5413, "corrected_gallery_identities": 5412,
        "base_prejoin_sha256": sha256_file(output / "base_prejoin.pt"),
        "physical_score_tensor_sha256": tensor_sha256(all_scores),
        "c128_row_tensor_sha256": tensor_sha256(c128_rows),
        "c128_score_tensor_sha256": tensor_sha256(c128_scores),
        "scoring_seconds": scoring_seconds,
        "resume_prejoin_source": str(resume_source) if resume_source is not None else None,
        "resume_prejoin_sha256": sha256_file(resume_source / "base_prejoin.pt") if resume_source is not None else None,
        "RAW_score_read_count": 0 if resume_source is not None else 600,
        "validated_prejoin_hit_count": 600 if resume_source is not None else 0,
        "label_read_count": 0, "target_insertions": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    exclusive_json(output / "base_prejoin_receipt.json", prejoin_receipt)

    confidences = [robust_confidence(c128_scores[index].tolist()) for index in range(600)]
    folds = [int(str(row["materialization_role"]).rsplit("_", 1)[1]) for row in target_free]
    thresholds = {}
    flags = []
    for heldout in (1, 2, 3, 4):
        members = [index for index, fold in enumerate(folds) if fold != heldout]
        ordered = sorted(members, key=lambda index: (confidences[index], redacted_rows[index]["source_image_sha256"], redacted_rows[index]["query_id"]))
        k = math.ceil(0.25 * len(ordered))
        tau = confidences[ordered[k - 1]]
        thresholds[str(heldout)] = {
            "calibration_count": len(ordered), "k": k, "tau": tau,
            "membership_logical_sha256": canonical_sha256([redacted_rows[index]["query_id"] for index in ordered]),
        }
    final_order = sorted(range(600), key=lambda index: (confidences[index], redacted_rows[index]["source_image_sha256"], redacted_rows[index]["query_id"]))
    tau_final = confidences[final_order[149]]
    for index, item in enumerate(redacted_rows):
        flag = confidences[index] <= thresholds[str(folds[index])]["tau"]
        flags.append({
            "query_id": item["query_id"], "source_image_sha256": item["source_image_sha256"],
            "fold": folds[index], "u_RAW": confidences[index], "oof_low_margin": flag,
        })
    low_receipt = {
        "schema_version": "rcde_raw_low_margin_receipt_v1_1",
        "status": "RAW_LOW_MARGIN_V1_1_FROZEN_BEFORE_LABEL_JOIN",
        "namespace": "RAW_LOW_MARGIN_V1_1_Q25_MAD_C128",
        "protocol_sha256": protocol_sha,
        "base_prejoin_sha256": prejoin_receipt["base_prejoin_sha256"],
        "query_cache_manifest_sha256": sha256_file(output / "redacted_query_token_manifest.json"),
        "role_source_hashes": {
            "target_free_runtime_manifest": bindings["target_free_runtime_manifest"]["sha256"],
            "execution_roles": bindings["execution_roles"]["sha256"],
        },
        "thresholds": thresholds, "tau_final": tau_final,
        "rows": flags, "rows_logical_sha256": canonical_sha256(flags),
        "allowed_uses": protocol["low_margin_contract"]["allowed_uses"],
        "forbidden_uses": protocol["low_margin_contract"]["forbidden_uses"],
        "RAW_score_read_count": 600, "label_read_count": 0,
        "opened_read_count": 0, "sealed_read_count": 0, "C8_read_count": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    exclusive_json(output / "rcde_raw_low_margin_receipt.json", low_receipt)

    roles_value = json.loads(resolve_binding(bindings["execution_roles"]).read_text(encoding="utf-8"))
    role_rows = roles_value["training_role"]["records"]
    role_by_id = {str(row["query_id"]): row for row in role_rows}
    if set(role_by_id) != {row["query_id"] for row in redacted_rows}:
        raise P0Error("postjoin role membership drift")
    joined = []
    legacy_rank_matches = 0
    identity_to_groups: dict[str, set[str]] = defaultdict(set)
    for index, item in enumerate(redacted_rows):
        role = role_by_id[item["query_id"]]
        target = str(role["identity"])
        group = str(role["supergroup"])
        identity_to_groups[target].add(group)
        ranked_rows = c128_rows[index].tolist()
        ranked_ids = [corrected[row] for row in ranked_rows]
        target_hit = target in ranked_ids
        raw_correct = ranked_ids[0] == target
        source = optimization_by_id[item["query_id"]]
        carrier = torch.load(source["optimization_shard_path"], map_location="cpu", weights_only=False)
        carrier_identity = str(carrier.get("identity"))
        if carrier_identity in setids:
            target_row = max(row for row, identity in enumerate(setids) if identity == carrier_identity)
            physical_rank = int((all_scores[index] > all_scores[index, target_row]).sum().item())
            legacy_rank_matches += int(physical_rank == int(carrier.get("raw_full_gallery_rank", -1)))
        joined.append({
            "query_id": item["query_id"], "query_ordinal": item["query_ordinal"],
            "source_image_sha256": item["source_image_sha256"], "fold": int(role["inner_fold"]),
            "track": str(role["track"]), "identity": target, "supergroup": group,
            "target_hit": target_hit, "raw_correct": raw_correct,
            "raw_winner": ranked_ids[0], "oof_low_margin": flags[index]["oof_low_margin"],
            "candidate_identities_by_raw_rank": ranked_ids,
            "candidate_rows_by_raw_rank": ranked_rows,
        })

    expected = protocol["expected_population"]
    role_checks = {
        "queries_600": len(joined) == expected["training_queries"],
        "identities_50": len({row["identity"] for row in joined}) == expected["identities"],
        "supergroups_49": len({row["supergroup"] for row in joined}) == expected["supergroups"],
        "tracks_exact": Counter(row["track"] for row in joined) == Counter(expected["tracks"]),
        "fold_queries_exact": Counter(str(row["fold"]) for row in joined) == Counter(expected["fold_query_counts"]),
        "fold_groups_exact": {str(fold): len({row["supergroup"] for row in joined if row["fold"] == fold}) for fold in (1, 2, 3, 4)} == expected["fold_group_counts"],
        "legacy_deployed_physical_rank_replay_all": legacy_rank_matches == 600,
        "target_identity_present_in_corrected_gallery": all(row["identity"] in set(corrected) for row in joined),
    }

    negative_folds = {}
    negative_gate = True
    balanced_gate = True
    for heldout in (1, 2, 3, 4):
        train_rows = [row for row in joined if row["fold"] != heldout]
        allowed_ids = {row["identity"] for row in train_rows}
        allowed_groups = {row["supergroup"] for row in train_rows}
        eligible = [row for row in train_rows if row["target_hit"]]
        reports = []
        middle_ids: set[str] = set()
        middle_groups: set[str] = set()
        broad_ids: set[str] = set()
        broad_groups: set[str] = set()
        for row in eligible:
            legal = []
            for rank, identity in enumerate(row["candidate_identities_by_raw_rank"], start=1):
                groups = identity_to_groups.get(identity, set())
                if identity != row["identity"] and identity in allowed_ids and len(groups) == 1 and next(iter(groups)) in allowed_groups:
                    legal.append((rank, identity, next(iter(groups))))
            middle = [item for item in legal if 9 <= item[0] <= 32]
            broad = [item for item in legal if 33 <= item[0] <= 128]
            middle_ids.update(item[1] for item in middle); middle_groups.update(item[2] for item in middle)
            broad_ids.update(item[1] for item in broad); broad_groups.update(item[2] for item in broad)
            reports.append({"legal": len(legal), "middle": bool(middle), "broad": bool(broad)})
        denominator = len(reports)
        at6 = sum(item["legal"] >= 6 for item in reports) / denominator if denominator else 0.0
        both = sum(item["middle"] and item["broad"] for item in reports) / denominator if denominator else 0.0
        fold_ok = (
            at6 >= 0.8 and both >= 0.6 and len(middle_ids) >= 8 and len(middle_groups) >= 8
            and len(broad_ids) >= 8 and len(broad_groups) >= 8
        )
        negative_gate &= fold_ok
        correct_count = len({row["query_id"] for row in eligible if row["raw_correct"]})
        wrong_count = len({row["query_id"] for row in eligible if not row["raw_correct"]})
        balanced_gate &= correct_count >= 2 and wrong_count >= 2
        negative_folds[str(heldout)] = {
            "denominator": denominator, "at_least_6_rate": at6,
            "middle_and_broad_rate": both, "middle_identity_count": len(middle_ids),
            "middle_supergroup_count": len(middle_groups), "broad_identity_count": len(broad_ids),
            "broad_supergroup_count": len(broad_groups), "raw_correct_pool": correct_count,
            "raw_wrong_pool": wrong_count, "gate": fold_ok,
        }
    overall_denominator = sum(value["denominator"] for value in negative_folds.values())
    overall_at6 = sum(value["at_least_6_rate"] * value["denominator"] for value in negative_folds.values()) / overall_denominator if overall_denominator else 0.0
    overall_both = sum(value["middle_and_broad_rate"] * value["denominator"] for value in negative_folds.values()) / overall_denominator if overall_denominator else 0.0
    negative_gate &= overall_at6 >= 0.8 and overall_both >= 0.7

    hall_rows = {}
    hall_gate = True
    for fold in (1, 2, 3, 4):
        for correctness in (False, True):
            population = [row for row in joined if row["fold"] == fold and row["target_hit"] and row["raw_correct"] is correctness]
            ok, details = hall_feasible([row["supergroup"] for row in population])
            details["feasible"] = ok
            hall_rows[f"fold{fold}_{'correct' if correctness else 'wrong'}"] = details
            hall_gate &= ok

    slice_counts = {}
    for name, predicate in {
        "overall": lambda row: True,
        "raw_correct": lambda row: row["raw_correct"],
        "raw_wrong": lambda row: not row["raw_correct"],
        "low_margin": lambda row: row["oof_low_margin"],
    }.items():
        rows = [row for row in joined if predicate(row)]
        slice_counts[name] = {
            "queries": len(rows), "identities": len({row["identity"] for row in rows}),
            "supergroups": len({row["supergroup"] for row in rows}),
            "fold_supergroups": {str(fold): len({row["supergroup"] for row in rows if row["fold"] == fold}) for fold in (1, 2, 3, 4)},
            "maximum_supergroup_frame_share": max(Counter(row["supergroup"] for row in rows).values(), default=0) / len(rows) if rows else 0.0,
        }
    stat_gate = (
        slice_counts["overall"]["supergroups"] >= 40
        and all(value >= 8 for value in slice_counts["overall"]["fold_supergroups"].values())
        and slice_counts["raw_correct"]["supergroups"] >= 12
        and slice_counts["raw_wrong"]["supergroups"] >= 12
        and all(value >= 2 for value in slice_counts["raw_correct"]["fold_supergroups"].values())
        and all(value >= 2 for value in slice_counts["raw_wrong"]["fold_supergroups"].values())
    )
    power = {name: sign_power(value["supergroups"]) for name, value in slice_counts.items()}

    spatial = torch.load(resolve_binding(bindings["gallery_spatial_metadata"]), map_location="cpu", weights_only=False)
    spatial_rows = spatial["rows"]
    union_rows = sorted({int(value) for row in joined for value in row["candidate_rows_by_raw_rank"]})
    query_shapes = []
    for item in redacted_rows:
        source = optimization_by_id[item["query_id"]]
        query_shapes.append({"query_id": item["query_id"], **shape_for_image(Path(source["source_image_path"]))})
    reference_shapes = []
    for physical_row in union_rows:
        metadata = spatial_rows[physical_row]
        reference_shapes.append({"physical_row": physical_row, **shape_for_image(Path(metadata["path"]))})
    all_shapes = query_shapes + reference_shapes
    payload_bytes = sum(4 * row["valid_tokens"] * 768 * 2 for row in all_shapes)
    geometry_bytes = sum(row["total_tokens"] for row in all_shapes)
    index_bytes = len(all_shapes) * 512
    checksum_bytes = len(all_shapes) * 64
    filesystem_bytes = len(all_shapes) * 4096
    cache_bytes_with_safety = math.ceil((payload_bytes + geometry_bytes + index_bytes + checksum_bytes + filesystem_bytes) * 1.25)
    dino_root = (ROOT / protocol["dino_freeze_targets"]["source_checkout"]).resolve()
    dino_checkpoint = (ROOT / protocol["dino_freeze_targets"]["checkpoint"]).resolve()
    dino_assets = dino_root.is_dir() and dino_checkpoint.is_file()
    resource = {
        "status": "RESOURCE_BENCHMARK_NOT_REACHED" if not (all(role_checks.values()) and negative_gate and balanced_gate and hall_gate and stat_gate) else "RESOURCE_BENCHMARK_REQUIRED",
        "N_ref_union": len(union_rows), "query_images": len(query_shapes),
        "cache_payload_bytes": payload_bytes, "geometry_bytes": geometry_bytes,
        "index_bytes": index_bytes, "checksum_bytes": checksum_bytes,
        "filesystem_overhead_bytes": filesystem_bytes,
        "cache_bytes_with_25pct_safety": cache_bytes_with_safety,
        "cache_gib_with_25pct_safety": cache_bytes_with_safety / (1024.0 ** 3),
        "cache_limit_pass": cache_bytes_with_safety / (1024.0 ** 3) <= 80.0,
        "dino_source_present": dino_root.is_dir(), "dino_checkpoint_present": dino_checkpoint.is_file(),
        "dino_source_commit": None, "dino_checkpoint_sha256": sha256_file(dino_checkpoint) if dino_checkpoint.is_file() else None,
        "worst_query_valid_tokens": max(row["valid_tokens"] for row in query_shapes),
        "worst_reference_valid_tokens": max(row["valid_tokens"] for row in reference_shapes),
        "benchmark_closed": False,
        "resource_gate": False,
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    if dino_root.is_dir() and (dino_root / ".git").is_dir():
        head = dino_root / ".git" / "HEAD"
        if head.is_file():
            value = head.read_text(encoding="utf-8").strip()
            if value.startswith("ref: "):
                ref = dino_root / ".git" / value[5:]
                resource["dino_source_commit"] = ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
            else:
                resource["dino_source_commit"] = value
    exclusive_json(output / "rcde_resource_receipt.json", resource)

    matched_control = {
        "eligible": all(row["grid_h"] * row["grid_w"] > 0 for row in redacted_rows)
        and len(gallery) == 5413 and all(int(spatial_rows[index]["total_token_count"]) == int(gallery[index].shape[0]) for index in range(5413)),
        "query_dimension": 128, "reference_dimension": 128,
        "query_image_tokens_exclude_template": True,
        "reference_visual_indices_bound": True,
        "same_gallery_cache_as_RAW": True,
    }
    data_receipt = {
        "schema_version": "rcde_data_role_receipt_v1_1",
        "role_checks": role_checks, "target_hit_count": sum(row["target_hit"] for row in joined),
        "target_miss_count": sum(not row["target_hit"] for row in joined),
        "raw_correct_count": sum(row["raw_correct"] for row in joined),
        "raw_wrong_count": sum(not row["raw_correct"] for row in joined),
        "legacy_rank_match_count": legacy_rank_matches,
        "negative_folds": negative_folds, "negative_overall_at_least_6_rate": overall_at6,
        "negative_overall_middle_and_broad_rate": overall_both,
        "negative_gate": negative_gate, "balanced_pool_gate": balanced_gate,
        "n1_hall_strata": hall_rows, "n1_hall_gate": hall_gate,
        "matched_colnomic_control": matched_control,
        "joined_rows_logical_sha256": canonical_sha256([{k: v for k, v in row.items() if not k.startswith("candidate_")} for row in joined]),
        "label_read_count": 600, "target_insertions": 0,
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    exclusive_json(output / "rcde_data_role_receipt.json", data_receipt)
    statistics = {
        "schema_version": "rcde_statistics_preflight_v1_1",
        "slice_counts": slice_counts, "track_counts": dict(Counter(row["track"] for row in joined)),
        "power_discrete_resolution": power, "statistics_gate": stat_gate,
        "bootstrap_repetitions": 10000, "permutation_repetitions": 10000, "seed": 17,
        "low_margin_inference_eligible": slice_counts["low_margin"]["supergroups"] >= 8
        and sum(value > 0 for value in slice_counts["low_margin"]["fold_supergroups"].values()) >= 3,
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    exclusive_json(output / "rcde_statistics_receipt.json", statistics)

    if not all(role_checks.values()):
        status, failure = DATA_ABORT, "data/base"
    elif not negative_gate or not balanced_gate:
        status, failure = NEGATIVE_ABORT, "data"
    elif not hall_gate:
        status, failure = HALL_ABORT, "data/statistics"
    elif not matched_control["eligible"]:
        status, failure = MATCHED_ABORT, "data/base"
    elif not stat_gate:
        status, failure = STAT_ABORT, "data/statistics"
    elif not resource["resource_gate"]:
        status, failure = RESOURCE_ABORT, "resource"
    else:
        status, failure = READY, None
    result = {
        "schema_version": SCHEMA, "status": status, "failure_domain": failure,
        "claim_level": protocol["claim_level"],
        "next_authorized_stage": "E0" if status == READY else None,
        "automatic_stage_advance": False, "natural_training_authorized": False,
        "protocol_sha256": protocol_sha, "contract_sha256": sha256_file(output / "contract.json"),
        "base_prejoin_receipt_sha256": sha256_file(output / "base_prejoin_receipt.json"),
        "low_margin_receipt_sha256": sha256_file(output / "rcde_raw_low_margin_receipt.json"),
        "data_role_receipt_sha256": sha256_file(output / "rcde_data_role_receipt.json"),
        "statistics_receipt_sha256": sha256_file(output / "rcde_statistics_receipt.json"),
        "resource_receipt_sha256": sha256_file(output / "rcde_resource_receipt.json"),
        "RAW_score_read_count": 600, "label_read_count": 600,
        "protected_access_counts": dict(PROTECTED_ZERO),
        "elapsed_seconds": (time.time_ns() - started_ns) / 1.0e9,
    }
    exclusive_json(output / "result.json", result)
    artifacts = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema_version": "rcde_p0_artifact_manifest_v1_1",
        "status": status, "files": [{"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in artifacts],
        "redacted_query_token_manifest_sha256": sha256_file(output / "redacted_query_token_manifest.json"),
        "protected_access_counts": dict(PROTECTED_ZERO),
    }
    exclusive_json(output / "manifest.json", manifest)
    print(json.dumps({"status": status, "failure_domain": failure, "next_authorized_stage": result["next_authorized_stage"]}, sort_keys=True))
    return 0 if status == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
