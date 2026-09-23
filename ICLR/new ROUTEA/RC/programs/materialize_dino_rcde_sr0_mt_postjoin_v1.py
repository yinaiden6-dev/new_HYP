#!/usr/bin/env python3
"""Join sealed SR0-MT pair evidence to the frozen natural y/rival metadata.

This is a metadata-only operation after independent prejoin validation.  It
selects one already-computed unordered C128 pair per natural target hit and
orients its signed evidence as target-minus-rival.  It never invokes a model or
changes the candidate population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping

RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import argv_from_execution_manifest  # noqa: E402


SCHEMA = "rc_dino_rcde_sr0_mt_postjoin_ledger_v1_20260815"
STATUS = "RCDE_SR0_MT_POSTJOIN_LEDGER_SEALED"
PREJOIN_PASS = "RCDE_SR0_MT_PREJOIN_INDEPENDENT_VALIDATION_PASS"
I0_POSTJOIN_PASS = "RCDE_SR0_MT_I0_POSTJOIN_INDEPENDENT_VALIDATION_PASS"
TRAINING_CLOSURE_PASS = "RCDE_SR0_MT_FORMAL_TRAINING_CLOSURE_PASS"
HISTORICAL_PASS = "RCDE_SR0_MT_HISTORICAL_CONTROL_REPLAY_PASS"
ARMS = (
    "ALL_PATCH_SAME_MODEL",
    "QUERY_REGION_FULL_REFERENCE_SAME_MODEL",
    "PAIRED_QUERY_REFERENCE_REGION",
)
CONTROLS = (
    "C_DINO_V",
    "C_COL_P",
    "P_QUERY",
    "P_REFERENCE",
    "RANDOM_CONNECTED_MATCHED_SHAPE",
    "DISCONNECTED_MATCHED_AREA",
    "MAGNITUDE_ONLY",
    "SINGLE_SEED",
)


class PostjoinMaterializerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PostjoinMaterializerError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def f64(value: Any) -> float:
    require(isinstance(value, str) and len(value) == 16 and all(c in "0123456789abcdef" for c in value), "invalid float64 bits")
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    require(math.isfinite(number), "nonfinite margin")
    return number


def orient_arm(value: Mapping[str, Any], sign: float) -> dict[str, Any]:
    margin = sign * f64(value["margin_f64_bits"])
    positive = float(value["positive_mass"])
    negative = float(value["negative_mass"])
    require(math.isfinite(positive) and math.isfinite(negative) and positive >= 0.0 and negative <= 0.0, "signed mass convention drift")
    if sign > 0:
        oriented_positive = positive
        oriented_negative = negative
        concentration = float(value["max_positive_patch_fraction"])
    else:
        oriented_positive = -negative
        oriented_negative = -positive
        concentration = float(value["max_negative_patch_fraction"])
    require(math.isfinite(concentration) and 0.0 <= concentration <= 1.0, "concentration drift")
    return {
        "margin": margin,
        "dual_candidate_legal_h1": bool(value["dual_candidate_legal_h1"]),
        "legal_positive_mass": bool(value["dual_candidate_legal_h1"] and oriented_positive > 0.0),
        "positive_mass": oriented_positive,
        "negative_mass": oriented_negative,
        "cancellation": float(value["cancellation"]),
        "max_positive_patch_fraction": concentration,
        "geometry_contract_pass": bool(value["geometry_contract_pass"]),
        "query_mask_sha256": str(value["query_mask_sha256"]),
        "reference_scope_sha256": str(value["reference_scope_sha256"]),
        "signed_contribution_payload_sha256": str(value["signed_contribution_payload_sha256"]),
        "fixed_denominator": 4,
    }


def orient_control(value: Mapping[str, Any], sign: float) -> dict[str, Any]:
    if value.get("status") == "NOT_APPLICABLE_CONTRACT":
        return {"status": "NOT_APPLICABLE_CONTRACT", "margin": None}
    require(value.get("status") == "READY", "control status drift")
    return {
        "status": "READY",
        "margin": sign * f64(value["margin_f64_bits"]),
        "only_registered_variable_changed": bool(value["only_registered_variable_changed"]),
    }


def materialize(
    *,
    prejoin_ledger: Path,
    prejoin_validation_path: Path,
    i0_postjoin: Path,
    i0_postjoin_validation_path: Path,
    training_closure_path: Path,
    historical_control_path: Path,
) -> dict[str, Any]:
    pre_validation = read_json(prejoin_validation_path)
    require(pre_validation.get("status") == PREJOIN_PASS, "prejoin validation has not passed")
    require(pre_validation.get("prejoin_ledger_sha256") == file_sha(prejoin_ledger), "prejoin validation/ledger hash drift")
    i0_validation = read_json(i0_postjoin_validation_path)
    require(i0_validation.get("status") == I0_POSTJOIN_PASS, "I0 metadata postjoin validation has not passed")
    require(i0_validation.get("postjoin_ledger_sha256") == file_sha(i0_postjoin), "I0 postjoin hash drift")
    i0 = read_json(i0_postjoin)
    training = read_json(training_closure_path)
    historical = read_json(historical_control_path)
    require(training.get("status") == TRAINING_CLOSURE_PASS, "formal training closure has not passed")
    require(historical.get("status") == HISTORICAL_PASS, "historical control replay has not passed")
    historical_records = historical.get("records")
    require(isinstance(historical_records, list), "historical control records absent")
    historical_by_query = {str(item["query_id"]): item for item in historical_records}
    require(len(historical_by_query) == len(historical_records), "duplicate historical query")

    metadata_records = i0.get("records")
    require(isinstance(metadata_records, list) and len(metadata_records) == 600, "I0 postjoin population drift")
    metadata = {str(item["query_id"]): item for item in metadata_records}
    require(len(metadata) == 600, "duplicate I0 postjoin query")
    wanted: dict[str, frozenset[int]] = {}
    misses = []
    for query_id, item in metadata.items():
        if item.get("target_hit") is True and item.get("eligibility_reason") == "NATURAL_TARGET_AND_RIVAL_READY":
            target = int(item["target_physical_row"])
            rival = int(item["strongest_rival_physical_row"])
            require(target != rival, "target and rival physical row collide")
            wanted[query_id] = frozenset((target, rival))
        else:
            misses.append({
                "query_id": query_id,
                "execution_ordinal": int(item["execution_ordinal"]),
                "outer_fold": int(item["outer_fold"]),
                "target_miss_reason": item.get("target_miss_reason"),
                "eligibility_reason": item.get("eligibility_reason"),
            })
    require(len(wanted) == 594 and len(misses) == 6, "natural 594/6 closure drift")

    selected: dict[str, Mapping[str, Any]] = {}
    with prejoin_ledger.open("r", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        require(header.get("schema_version") == "rc_dino_rcde_sr0_mt_prejoin_ledger_v1_20260815", "prejoin header drift")
        for line in handle:
            record = json.loads(line)
            query_id = str(record.get("query_id"))
            if query_id in wanted and frozenset((int(record["left_physical_row"]), int(record["right_physical_row"]))) == wanted[query_id]:
                require(query_id not in selected, "multiple prejoin records match y/rival")
                selected[query_id] = record
    require(set(selected) == set(wanted), "one or more y/rival pairs absent from exhaustive prejoin")

    output_records = []
    for query_id in sorted(selected, key=lambda key: int(metadata[key]["execution_ordinal"])):
        source = selected[query_id]
        meta = metadata[query_id]
        target_row = int(meta["target_physical_row"])
        sign = 1.0 if int(source["left_physical_row"]) == target_row else -1.0
        require(int(source["right_physical_row"] if sign > 0 else source["left_physical_row"]) == int(meta["strongest_rival_physical_row"]), "selected pair orientation drift")
        arms = {name: orient_arm(source["arms"][name], sign) for name in ARMS}
        require(arms[ARMS[1]]["query_mask_sha256"] == arms[ARMS[2]]["query_mask_sha256"], "regional query mask drift after join")
        controls = {
            arm: {name: orient_control(source["controls"][arm][name], sign) for name in CONTROLS}
            for arm in ARMS
        }
        old = historical_by_query.get(query_id)
        require(isinstance(old, Mapping), f"historical control missing query: {query_id}")
        old_margin = float(old.get("margin"))
        require(math.isfinite(old_margin), "historical margin nonfinite")
        record = {
            "query_id": query_id,
            "execution_ordinal": int(meta["execution_ordinal"]),
            "historical_query_ordinal": int(meta["historical_query_ordinal"]),
            "source_image_sha256": str(meta["source_image_sha256"]),
            "outer_fold": int(meta["outer_fold"]),
            "identity": str(meta["identity"]),
            "supergroup": str(meta["supergroup"]),
            "track": str(meta["track"]),
            "target_corrected_identity": str(meta["target_corrected_identity"]),
            "target_physical_row": target_row,
            "strongest_rival_corrected_identity": str(meta["strongest_rival_corrected_identity"]),
            "strongest_rival_physical_row": int(meta["strongest_rival_physical_row"]),
            "prejoin_pair_record_sha256": str(source["record_sha256"]),
            "target_is_prejoin_left": sign > 0,
            "arms": arms,
            "controls": controls,
            "historical_controls": {"OLD_COLNOMIC_QWEN_MULTI_SUPPORT": {"margin": old_margin}},
        }
        record["record_sha256"] = logical_sha(record)
        output_records.append(record)

    closures = {
        "formal_source_hash_fold_prejoin_training_closure": bool(
            training.get("formal_source_hash_fold_prejoin_training_closure") is True
            and pre_validation.get("status") == PREJOIN_PASS
            and i0_validation.get("status") == I0_POSTJOIN_PASS
        ),
        "candidate_reorder_exact": bool(pre_validation.get("checks", {}).get("symmetry_counts")),
        "pair_swap_exact": bool(pre_validation.get("checks", {}).get("symmetry_counts")),
        "protected_access_zero": bool(
            training.get("protected_access_zero") is True
            and pre_validation.get("checks", {}).get("access_audit_zero") is True
        ),
        "mandatory_historical_control_replay": bool(
            historical.get("exact_population_replay") is True
            and len(historical_by_query) == 594
        ),
        "postjoin_model_forward_count": 0,
        "target_selected_pair_forward_count": 0,
        "target_insertion_count": 0,
        "candidate_mutation_count": 0,
    }
    result = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "prejoin_validation_status": PREJOIN_PASS,
        "population": {
            "query_count": 600,
            "natural_target_hit_count": 594,
            "natural_target_miss_count": 6,
            "eligible_record_count": 594,
        },
        "records": output_records,
        "misses": sorted(misses, key=lambda item: item["execution_ordinal"]),
        "closures": closures,
        "bindings": {
            "prejoin_ledger_sha256": file_sha(prejoin_ledger),
            "prejoin_validation_sha256": file_sha(prejoin_validation_path),
            "i0_postjoin_sha256": file_sha(i0_postjoin),
            "i0_postjoin_validation_sha256": file_sha(i0_postjoin_validation_path),
            "training_closure_sha256": file_sha(training_closure_path),
            "historical_control_sha256": file_sha(historical_control_path),
        },
        "record_sequence_sha256": logical_sha(output_records),
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "automatic_stage_advance": False,
    }
    result["logical_sha256"] = logical_sha(result)
    return result


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable postjoin exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(partial, path)
        os.chmod(path, 0o440)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prejoin-ledger", type=Path, required=True)
    parser.add_argument("--prejoin-validation", type=Path, required=True)
    parser.add_argument("--i0-postjoin", type=Path, required=True)
    parser.add_argument("--i0-postjoin-validation", type=Path, required=True)
    parser.add_argument("--training-closure", type=Path, required=True)
    parser.add_argument("--historical-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(
        argv_from_execution_manifest(sys.argv[1:], expected_program=Path(__file__).name)
    )
    result = materialize(
        prejoin_ledger=args.prejoin_ledger,
        prejoin_validation_path=args.prejoin_validation,
        i0_postjoin=args.i0_postjoin,
        i0_postjoin_validation_path=args.i0_postjoin_validation,
        training_closure_path=args.training_closure,
        historical_control_path=args.historical_control,
    )
    exclusive_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
