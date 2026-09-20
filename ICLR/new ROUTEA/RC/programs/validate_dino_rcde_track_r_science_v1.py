#!/usr/bin/env python3
"""Independently reconstruct Track-R scientific decisions A and B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping

import torch


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))

from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from dino_rcde_track_r_statistics_independent_v1 import (  # noqa: E402
    ARMS,
    PRIMARY_PERMUTATIONS,
    SEED,
    independent_evaluate_track_r,
)
from dino_rcde_track_r_v121_lineage_v1 import (  # noqa: E402
    read_authority as read_frozen_authority,
    validate_runtime_bindings,
)


AUTHORITY_PATH = "registry/current_authority_v123_20260822.json"
AUTHORITY_SCHEMA = "rc_current_authority_v123_20260822"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_AUTHORIZED"
RESULT_SCHEMA = "rc_dino_rcde_track_r_scientific_result_v1_20260821"
VALIDATION_SCHEMA = "rc_dino_rcde_track_r_scientific_validation_v1_20260821"
VALIDATION_STATUS = "DINO_RCDE_TRACK_R_SCIENTIFIC_INDEPENDENT_VALIDATION_PASS"
PREJOIN_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_READY"
PREJOIN_VALIDATION_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_INDEPENDENT_VALIDATION_PASS"
P_COMPARATOR_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_READY"
SHARD_COUNT = 50
SHARD_SIZE = 12
QUERY_COUNT = 600
VALID_QUERY_COUNT = 594


class TrackRScienceValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRScienceValidationError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "logical_sha256"})


def safe_path(path: Path, *, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(not {"c8", "s8", "opened", "sealed"}.intersection(part.lower() for part in value.parts), f"protected path requested: {value}")
    require((value.is_file() if file else value.is_dir()) and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object absent")
    return value


def binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(isinstance(item, Mapping), f"binding absent: {name}")
    path = safe_path(RC_ROOT / str(item["path"]))
    require(file_sha256(path) == item.get("sha256"), f"binding drift: {name}")
    return path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "immutable validation exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    path.chmod(0o444)


def _bits(value: str) -> float:
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    require(result == result and abs(result) != float("inf"), "nonfinite RAW score bits")
    return float(result)


def _logit(evidence: Mapping[str, Any], arm: str) -> float:
    value = evidence["arms"][arm]["logit"]
    require(isinstance(value, (int, float)), "arm logit absent")
    return float(value)


def validate(
    authority_path: Path,
    prejoin_root: Path,
    prejoin_validation_root: Path,
    i0_seal_path: Path,
    i0_seal_validation_path: Path,
    p_comparator_path: Path,
    p_comparator_validation_path: Path,
    result_path: Path,
    output: Path,
) -> dict[str, Any]:
    authority_path = safe_path(authority_path)
    authority, authority_sha256 = read_frozen_authority(
        authority_path,
        root=RC_ROOT,
        expected_relative=AUTHORITY_PATH,
        expected_schema=AUTHORITY_SCHEMA,
        expected_status=AUTHORITY_STATUS,
    )
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("scientific_validation_authorized") is True
        and authority.get("postjoin_model_forward_authorized") is False
        and authority.get("opened_sealed_access_authorized") is False,
        "V123 validation authority drift",
    )
    validate_runtime_bindings(
        authority,
        root=RC_ROOT,
        required_names=(
            "metadata_sealer",
            "metadata_validator",
            "statistics",
            "independent_statistics",
            "reducer",
            "validator",
            "runtime_entry_validator",
            "lineage_runtime",
            "package_initializer",
            "launcher",
            "post_p_controller",
        ),
    )
    result_path = safe_path(result_path)
    require(
        result_path
        == (RC_ROOT / str(authority["science_result_output"])).resolve(),
        "science result path drift",
    )
    result = read_json(result_path)
    require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_COMPLETE"
        and result.get("logical_sha256") == logical_sha256(result)
        and result.get("query_count") == VALID_QUERY_COUNT
        and result.get("postjoin_model_forward_count") == 0,
        "scientific result envelope drift",
    )
    require(
        result.get("source_bindings", {}).get("authority_sha256")
        == authority_sha256
        and result.get("source_bindings", {}).get("authority_logical_sha256")
        == authority.get("logical_sha256"),
        "scientific result authority lineage drift",
    )
    require(
        safe_path(i0_seal_path)
        == (RC_ROOT / authority["i0_seal_output"]).resolve()
        and safe_path(i0_seal_validation_path)
        == (RC_ROOT / authority["i0_seal_validation_output"]).resolve()
        and safe_path(p_comparator_path) == binding_path(authority, "p_comparator")
        and safe_path(p_comparator_validation_path)
        == binding_path(authority, "p_comparator_validation"),
        "independent V123 input path drift",
    )
    i0 = read_json(i0_seal_path)
    i0_validation = read_json(i0_seal_validation_path)
    p_comparator = read_json(p_comparator_path)
    p_validation = read_json(p_comparator_validation_path)
    require(
        i0.get("status") == "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_SEALED"
        and i0.get("logical_sha256")
        == authority.get("expected_i0_seal_logical_sha256")
        and i0_validation.get("status")
        == "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_INDEPENDENT_VALIDATION_PASS"
        and i0_validation.get("seal_sha256") == file_sha256(i0_seal_path)
        and i0_validation.get("seal_logical_sha256")
        == i0.get("logical_sha256")
        and i0_validation.get("logical_sha256") == logical_sha256(i0_validation)
        and p_comparator.get("status") == P_COMPARATOR_STATUS
        and p_comparator.get("logical_sha256") == logical_sha256(p_comparator)
        and p_comparator.get("target_free_prejoin_score_ledger") is True
        and p_validation.get("schema_version")
        == "rc_dino_rcde_track_r_p_v2_utility_comparator_validation_v1_20260822"
        and p_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_INDEPENDENT_VALIDATION_PASS"
        and p_validation.get("validation_pass") is True
        and p_validation.get("logical_sha256") == logical_sha256(p_validation)
        and p_validation.get("comparator_sha256") == file_sha256(p_comparator_path)
        and p_validation.get("comparator_logical_sha256")
        == p_comparator.get("logical_sha256")
        and p_validation.get("query_count") == VALID_QUERY_COUNT
        and p_validation.get("candidate_utility_count") == VALID_QUERY_COUNT * 128,
        "independent seal/comparator validation drift",
    )
    i0_by_execution = {int(item["execution_ordinal"]): item for item in i0["records"]}
    p_by_execution = {int(item["execution_ordinal"]): item for item in p_comparator["records"]}
    prejoin = safe_path(prejoin_root, file=False)
    validations = safe_path(prejoin_validation_root, file=False)
    require(
        prejoin == (RC_ROOT / authority["prejoin_root"]).resolve()
        and validations
        == (RC_ROOT / authority["prejoin_validation_root"]).resolve(),
        "independent prejoin root drift",
    )
    rows = []
    shard_receipts = []
    for shard_ordinal in range(SHARD_COUNT):
        start = shard_ordinal * SHARD_SIZE
        stop = min(QUERY_COUNT, start + SHARD_SIZE)
        shard_path = safe_path(prejoin / f"shard_{start:03d}_{stop:03d}.pt")
        validation_path = safe_path(validations / f"shard_{start:03d}_{stop:03d}.validation.json")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        shard_validation = read_json(validation_path)
        require(
            isinstance(shard, Mapping)
            and shard.get("status") == PREJOIN_STATUS
            and shard.get("logical_sha256") == logical_sha256(shard)
            and shard_validation.get("status") == PREJOIN_VALIDATION_STATUS
            and shard_validation.get("validation_pass") is True
            and shard_validation.get("producer_shard_sha256") == file_sha256(shard_path)
            and shard_validation.get("producer_shard_logical_sha256")
            == shard.get("logical_sha256")
            and shard.get("source_bindings", {}).get("authority_sha256")
            == authority.get("source_v121_authority_sha256")
            and shard_validation.get("source_v121_authority_sha256")
            == authority.get("source_v121_authority_sha256"),
            "validated prejoin shard drift",
        )
        for pre in shard["rows"]:
            execution = int(pre["execution_ordinal"])
            meta = i0_by_execution[execution]
            p_row = p_by_execution[execution]
            member_order = list(pre["member_order"])
            member_addresses = {
                str(item["candidate_key"]): item
                for item in pre.get("member_addresses", [])
            }
            target = str(meta["target_candidate_key"])
            rival = str(meta["rival_candidate_key"])
            require(
                set(member_order) == set(member_addresses) == {target, rival}
                and member_order[int(meta["target_member_ordinal"])] == target
                and member_order[int(meta["rival_member_ordinal"])] == rival
                and member_addresses[target]["candidate_physical_row"]
                == meta["target_physical_row"]
                and member_addresses[rival]["candidate_physical_row"]
                == meta["rival_physical_row"]
                and pre["query_id"] == meta["query_id"] == p_row["query_id"]
                and pre["query_source_image_sha256"]
                == meta["query_source_image_sha256"]
                == p_row["query_source_image_sha256"]
                and pre["outer_fold"] == meta["outer_fold"] == p_row["outer_fold"]
                and pre["pair_sha256"] == meta["pair_sha256"]
                and pre["candidate_axis_sha256"]
                == meta["candidate_axis_sha256"]
                == p_row["candidate_axis_sha256"],
                "independent role/pair metadata join drift",
            )
            sign = 1.0 if member_order[0] == target else -1.0
            p_candidates = {str(item["candidate_key"]): item for item in p_row["candidates"]}
            require(len(p_candidates) == 128 and target in p_candidates and rival in p_candidates, "independent P comparator join drift")
            arm_values = {}
            for arm in ARMS:
                arm_values[arm] = {
                    "REAL": sign * _logit(pre["real"], arm),
                    "C_DINO_V": sign * _logit(pre["C_DINO_V"], arm),
                    "P_QUERY": sign * _logit(pre["P_QUERY"], arm),
                    "P_REFERENCE": sign * float(pre["P_REFERENCE"][arm]["logit"]),
                }
            rows.append({
                "query_id": pre["query_id"],
                "execution_ordinal": execution,
                "group_sha256": meta["group_sha256"],
                "outer_fold": int(meta["outer_fold"]),
                "raw_margin": _bits(meta["target_raw_score_bits"]) - _bits(meta["rival_raw_score_bits"]),
                "p_v2_utility_margin": float(p_candidates[target]["utility"] - p_candidates[rival]["utility"]),
                "arms": arm_values,
            })
        shard_receipts.append(
            {
                "shard_ordinal": shard_ordinal,
                "producer_sha256": file_sha256(shard_path),
                "producer_logical_sha256": shard["logical_sha256"],
                "validation_sha256": file_sha256(validation_path),
                "validation_logical_sha256": shard_validation["logical_sha256"],
            }
        )
    rows.sort(key=lambda item: item["execution_ordinal"])
    require(len(rows) == VALID_QUERY_COUNT, "independent scientific population drift")
    require(
        len({row["execution_ordinal"] for row in rows}) == VALID_QUERY_COUNT
        and len({row["group_sha256"] for row in rows}) == 49,
        "independent 594/49 scientific population drift",
    )
    group_folds: dict[str, set[int]] = {}
    for row in rows:
        group_folds.setdefault(str(row["group_sha256"]), set()).add(
            int(row["outer_fold"])
        )
    require(
        all(len(value) == 1 for value in group_folds.values())
        and {
            fold: sum(next(iter(value)) == fold for value in group_folds.values())
            for fold in (1, 2, 3, 4)
        }
        == {1: 13, 2: 12, 3: 12, 4: 12},
        "independent group/fold isolation drift",
    )
    statistics = independent_evaluate_track_r(
        rows, repetitions=PRIMARY_PERMUTATIONS, seed=SEED
    )
    p_wrong = [row for row in rows if row["p_v2_utility_margin"] <= 0.0]
    p_diagnostics = {
        "query_count": len(rows),
        "P_wrong_query_count": len(p_wrong),
        "per_arm_P_wrong_direction": {
            arm: (
                None
                if not p_wrong
                else sum(row["arms"][arm]["REAL"] > 0.0 for row in p_wrong)
                / len(p_wrong)
            )
            for arm in ARMS
        },
        "per_arm_P_wrong_mean_margin": {
            arm: (
                None
                if not p_wrong
                else sum(row["arms"][arm]["REAL"] for row in p_wrong)
                / len(p_wrong)
            )
            for arm in ARMS
        },
    }
    require(
        result.get("statistics") == statistics
        and result.get("p_v2_utility_comparator_diagnostics") == p_diagnostics
        and result.get("decision_A") == statistics["decision_A"]
        and result.get("spatial_boundary") == statistics["spatial_boundary"]
        and result.get("decision_B") == statistics["decision_B"],
        "independent Track-R statistical replay drift",
    )
    require(
        result.get("claim_scoped_scientific_decisions")
        == {
            "specific_reference_evidence": statistics["decision_A"],
            "spatial_boundary": statistics["spatial_boundary"],
            "p_lock_regional_increment": statistics["decision_B"],
        }
        and result.get("multiplicity_scope")
        == {
            "maxT_family": "THREE_PRIMARY_ARM_DIRECTION_TESTS_ONLY",
            "destruction_and_regional_cross_family_adjustment": False,
            "preregistered_gate_semantics_preserved": True,
        }
        and result.get(
            "target_free_evidence_and_p_comparator_sealed_before_role_join"
        )
        is True
        and result.get("opened_sealed_access_count") == 0,
        "independent claim/multiplicity boundary drift",
    )
    require(
        result.get("shard_receipts") == shard_receipts
        and result.get("source_bindings", {}).get("i0_seal_sha256")
        == file_sha256(i0_seal_path)
        and result.get("source_bindings", {}).get(
            "i0_seal_validation_sha256"
        )
        == file_sha256(i0_seal_validation_path)
        and result.get("source_bindings", {}).get(
            "p_v2_utility_comparator_sha256"
        )
        == file_sha256(p_comparator_path)
        and result.get("source_bindings", {}).get(
            "p_v2_utility_comparator_validation_sha256"
        )
        == file_sha256(p_comparator_validation_path)
        and result.get("source_bindings", {}).get(
            "ordered_prejoin_shards_sha256"
        )
        == canonical_sha256(shard_receipts),
        "independent scientific source receipt drift",
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "result_sha256": file_sha256(result_path),
        "result_logical_sha256": result["logical_sha256"],
        "authority_sha256": authority_sha256,
        "authority_logical_sha256": authority["logical_sha256"],
        "query_count": VALID_QUERY_COUNT,
        "group_count": statistics["group_count"],
        "decision_A": statistics["decision_A"],
        "spatial_boundary": statistics["spatial_boundary"],
        "decision_B": statistics["decision_B"],
        "postjoin_model_forward_count": 0,
        "protected_access_count": 0,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    validation["logical_sha256"] = logical_sha256(validation)
    require(
        output.resolve()
        == (RC_ROOT / authority["science_validation_output"]).resolve(),
        "science validation output path drift",
    )
    atomic_json(output, validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--prejoin-root", type=Path, required=True)
    parser.add_argument("--prejoin-validation-root", type=Path, required=True)
    parser.add_argument("--i0-seal", type=Path, required=True)
    parser.add_argument("--i0-seal-validation", type=Path, required=True)
    parser.add_argument("--p-utility-comparator", type=Path, required=True)
    parser.add_argument("--p-utility-comparator-validation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = validate(args.authority, args.prejoin_root, args.prejoin_validation_root, args.i0_seal, args.i0_seal_validation, args.p_utility_comparator, args.p_utility_comparator_validation, args.result, args.output)
    print(json.dumps({"status": value["status"], "decision_A": value["decision_A"], "spatial_boundary": value["spatial_boundary"], "decision_B": value["decision_B"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
