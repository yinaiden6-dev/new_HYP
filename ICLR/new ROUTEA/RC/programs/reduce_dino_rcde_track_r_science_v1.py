#!/usr/bin/env python3
"""Join frozen Track-R OOF prejoin evidence and issue decisions A and B."""

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
from rc_aslo_xf.dino_rcde_track_r_statistics_v1 import (  # noqa: E402
    ARMS,
    PRIMARY_PERMUTATIONS,
    SEED,
    evaluate_track_r,
)
from dino_rcde_track_r_v121_lineage_v1 import (  # noqa: E402
    read_authority as read_frozen_authority,
    validate_runtime_bindings,
)


AUTHORITY_PATH = "registry/current_authority_v123_20260822.json"
AUTHORITY_SCHEMA = "rc_current_authority_v123_20260822"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_AUTHORIZED"
PREJOIN_SCHEMA = "rc_dino_rcde_track_r_oof_prejoin_shard_v1_20260821"
PREJOIN_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_READY"
PREJOIN_VALIDATION_STATUS = "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARD_INDEPENDENT_VALIDATION_PASS"
P_COMPARATOR_SCHEMA = "rc_dino_rcde_track_r_p_v2_utility_comparator_v1_20260821"
P_COMPARATOR_STATUS = "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_READY"
RESULT_SCHEMA = "rc_dino_rcde_track_r_scientific_result_v1_20260821"
SHARD_COUNT = 50
SHARD_SIZE = 12
QUERY_COUNT = 600
VALID_QUERY_COUNT = 594
EXCLUDED = (25, 26, 101, 346, 354, 470)


class TrackRScienceError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRScienceError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def safe_path(path: Path, *, file: bool = True) -> Path:
    value = path.resolve()
    require(RC_ROOT == value or RC_ROOT in value.parents, f"path escapes RC root: {value}")
    require(
        not {"c8", "s8", "opened", "sealed"}.intersection(
            part.lower() for part in value.parts
        ),
        f"protected path requested: {value}",
    )
    require((value.is_file() if file else value.is_dir()) and not value.is_symlink(), f"input absent/unsafe: {value}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(safe_path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(isinstance(item, Mapping), f"authority binding absent: {name}")
    path = safe_path(RC_ROOT / str(item["path"]))
    require(
        file_sha256(path) == item.get("sha256"),
        f"authority binding drift: {name}",
    )
    return path


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "immutable scientific result exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    path.chmod(0o444)


def _float64_bits(value: str) -> float:
    require(isinstance(value, str) and len(value) == 16, "float64 score bits drift")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    require(result == result and abs(result) != float("inf"), "score is nonfinite")
    return float(result)


def _group_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _arm_logit(evidence: Mapping[str, Any], arm: str) -> float:
    value = evidence.get("arms", {}).get(arm, {}).get("logit")
    require(isinstance(value, (int, float)), f"{arm} logit absent")
    return float(value)


def reduce(
    authority_path: Path,
    prejoin_root: Path,
    validation_root: Path,
    i0_seal_path: Path,
    i0_seal_validation_path: Path,
    p_comparator_path: Path,
    p_comparator_validation_path: Path,
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
        and authority.get("scientific_reduction_authorized") is True
        and authority.get("postjoin_model_forward_authorized") is False
        and authority.get("opened_sealed_access_authorized") is False
        and authority.get("automatic_stage_advance") is False,
        "V123 scientific authority drift",
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
    prejoin = safe_path(prejoin_root, file=False)
    validations = safe_path(validation_root, file=False)
    require(
        prejoin == (RC_ROOT / authority["prejoin_root"]).resolve()
        and validations == (RC_ROOT / authority["prejoin_validation_root"]).resolve(),
        "V123 prejoin root drift",
    )
    require(
        safe_path(i0_seal_path)
        == (RC_ROOT / authority["i0_seal_output"]).resolve()
        and safe_path(i0_seal_validation_path)
        == (RC_ROOT / authority["i0_seal_validation_output"]).resolve(),
        "V123 I0 seal CLI path drift",
    )
    for provided, name in (
        (p_comparator_path, "p_comparator"),
        (p_comparator_validation_path, "p_comparator_validation"),
    ):
        require(
            safe_path(provided) == binding_path(authority, name),
            f"V123 {name} CLI path drift",
        )
    i0 = read_json(i0_seal_path)
    i0_validation = read_json(i0_seal_validation_path)
    p_comparator = read_json(p_comparator_path)
    p_validation = read_json(p_comparator_validation_path)
    require(
        i0.get("status") == "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_SEALED"
        and i0.get("logical_sha256")
        == authority.get("expected_i0_seal_logical_sha256")
        and i0.get("query_count") == VALID_QUERY_COUNT
        and i0.get("group_count") == 49
        and i0.get("group_fold_isolation") is True
        and i0_validation.get("status")
        == "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_INDEPENDENT_VALIDATION_PASS"
        and i0_validation.get("validation_pass") is True
        and i0_validation.get("seal_sha256") == file_sha256(i0_seal_path)
        and i0_validation.get("seal_logical_sha256")
        == i0.get("logical_sha256")
        and i0_validation.get("logical_sha256") == logical_sha256(i0_validation)
        and i0.get("model_forward_count") == 0,
        "sealed I0 metadata drift",
    )
    require(
        p_comparator.get("schema_version") == P_COMPARATOR_SCHEMA
        and p_comparator.get("status") == P_COMPARATOR_STATUS
        and p_comparator.get("query_count") == VALID_QUERY_COUNT
        and p_comparator.get("logical_sha256") == logical_sha256(p_comparator)
        and p_comparator.get("target_free_prejoin_score_ledger") is True
        and p_comparator.get("scientific_GO_or_NO_GO") is None,
        "P-V2 utility comparator drift",
    )
    require(
        p_validation.get("schema_version")
        == "rc_dino_rcde_track_r_p_v2_utility_comparator_validation_v1_20260822"
        and p_validation.get("status")
        == "DINO_RCDE_TRACK_R_P_V2_UTILITY_COMPARATOR_INDEPENDENT_VALIDATION_PASS"
        and p_validation.get("validation_pass") is True
        and p_validation.get("logical_sha256") == logical_sha256(p_validation)
        and p_validation.get("comparator_sha256") == file_sha256(p_comparator_path)
        and p_validation.get("comparator_logical_sha256")
        == p_comparator.get("logical_sha256")
        and p_validation.get("query_count") == VALID_QUERY_COUNT
        and p_validation.get("candidate_utility_count") == VALID_QUERY_COUNT * 128
        and p_validation.get("target_rival_read_count") == 0
        and p_validation.get("dino_token_read_count") == 0
        and p_validation.get("scientific_reduction_count") == 0,
        "P-V2 comparator independent validation drift",
    )
    i0_by_execution = {
        int(item["execution_ordinal"]): item for item in i0["records"]
    }
    p_by_execution = {
        int(item["execution_ordinal"]): item for item in p_comparator["records"]
    }
    require(len(i0_by_execution) == len(p_by_execution) == VALID_QUERY_COUNT, "594-query join population drift")
    rows = []
    shard_receipts = []
    for shard_ordinal in range(SHARD_COUNT):
        start = shard_ordinal * SHARD_SIZE
        stop = min(QUERY_COUNT, start + SHARD_SIZE)
        shard_path = safe_path(prejoin / f"shard_{start:03d}_{stop:03d}.pt")
        validation_path = safe_path(
            validations / f"shard_{start:03d}_{stop:03d}.validation.json"
        )
        shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        validation = read_json(validation_path)
        require(
            isinstance(shard, Mapping)
            and shard.get("schema_version") == PREJOIN_SCHEMA
            and shard.get("status") == PREJOIN_STATUS
            and shard.get("logical_sha256") == logical_sha256(shard)
            and validation.get("status") == PREJOIN_VALIDATION_STATUS
            and validation.get("validation_pass") is True
            and validation.get("producer_shard_sha256") == file_sha256(shard_path)
            and validation.get("producer_shard_logical_sha256")
            == shard["logical_sha256"],
            "validated OOF prejoin shard drift",
        )
        require(
            shard.get("source_bindings", {}).get("authority_sha256")
            == authority.get("source_v121_authority_sha256")
            and validation.get("source_v121_authority_sha256")
            == authority.get("source_v121_authority_sha256"),
            "V121 prejoin source-authority drift",
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
            target_key = str(meta["target_candidate_key"])
            rival_key = str(meta["rival_candidate_key"])
            p_candidates = {
                str(item["candidate_key"]): item
                for item in p_row.get("candidates", [])
            }
            require(
                set(member_order) == {target_key, rival_key}
                and set(member_addresses) == {target_key, rival_key}
                and member_order[int(meta["target_member_ordinal"])] == target_key
                and member_order[int(meta["rival_member_ordinal"])] == rival_key
                and int(member_addresses[target_key]["candidate_physical_row"])
                == int(meta["target_physical_row"])
                and int(member_addresses[rival_key]["candidate_physical_row"])
                == int(meta["rival_physical_row"])
                and pre.get("query_id") == meta.get("query_id") == p_row.get("query_id")
                and pre.get("query_source_image_sha256")
                == meta.get("query_source_image_sha256")
                == p_row.get("query_source_image_sha256")
                and pre.get("outer_fold") == meta.get("outer_fold") == p_row.get("outer_fold")
                and pre.get("pair_sha256") == meta.get("pair_sha256")
                and pre.get("candidate_axis_sha256")
                == meta.get("candidate_axis_sha256")
                == p_row.get("candidate_axis_sha256")
                and p_row.get("target_free") is True
                and len(p_candidates) == 128
                and target_key in p_candidates
                and rival_key in p_candidates,
                "postjoin role/prejoin pair/P comparator drift",
            )
            sign = 1.0 if member_order[0] == target_key else -1.0
            raw_margin = _float64_bits(meta["target_raw_score_bits"]) - _float64_bits(
                meta["rival_raw_score_bits"]
            )
            p_margin = float(
                p_candidates[target_key]["utility"]
                - p_candidates[rival_key]["utility"]
            )
            arm_rows = {}
            for arm in ARMS:
                arm_rows[arm] = {
                    "REAL": sign * _arm_logit(pre["real"], arm),
                    "C_DINO_V": sign * _arm_logit(pre["C_DINO_V"], arm),
                    "P_QUERY": sign * _arm_logit(pre["P_QUERY"], arm),
                    "P_REFERENCE": sign
                    * float(pre["P_REFERENCE"][arm]["logit"]),
                }
            rows.append(
                {
                    "query_id": pre["query_id"],
                    "execution_ordinal": execution,
                    "group_sha256": str(meta["group_sha256"]),
                    "outer_fold": int(meta["outer_fold"]),
                    "raw_margin": raw_margin,
                    "p_v2_utility_margin": p_margin,
                    "arms": arm_rows,
                }
            )
        shard_receipts.append(
            {
                "shard_ordinal": shard_ordinal,
                "producer_sha256": file_sha256(shard_path),
                "producer_logical_sha256": shard["logical_sha256"],
                "validation_sha256": file_sha256(validation_path),
                "validation_logical_sha256": validation["logical_sha256"],
            }
        )
    rows.sort(key=lambda item: int(item["execution_ordinal"]))
    require(len(rows) == VALID_QUERY_COUNT and len({row["execution_ordinal"] for row in rows}) == VALID_QUERY_COUNT, "594-query scientific row closure drift")
    group_folds: dict[str, set[int]] = {}
    for row in rows:
        group_folds.setdefault(str(row["group_sha256"]), set()).add(
            int(row["outer_fold"])
        )
    require(
        len(group_folds) == 49
        and all(len(value) == 1 for value in group_folds.values())
        and {
            fold: sum(next(iter(value)) == fold for value in group_folds.values())
            for fold in (1, 2, 3, 4)
        }
        == {1: 13, 2: 12, 3: 12, 4: 12},
        "49-group fold isolation drift",
    )
    statistics = evaluate_track_r(
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
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_COMPLETE",
        "claim_level": "GIVEN_C128_SPECIFIC_REFERENCE_EVIDENCE_AND_P_LOCK_INCREMENT_ONLY",
        "query_count": VALID_QUERY_COUNT,
        "excluded_execution_ordinals": list(EXCLUDED),
        "statistics": statistics,
        "p_v2_utility_comparator_diagnostics": p_diagnostics,
        "decision_A": statistics["decision_A"],
        "spatial_boundary": statistics["spatial_boundary"],
        "decision_B": statistics["decision_B"],
        "claim_scoped_scientific_decisions": {
            "specific_reference_evidence": statistics["decision_A"],
            "spatial_boundary": statistics["spatial_boundary"],
            "p_lock_regional_increment": statistics["decision_B"],
        },
        "multiplicity_scope": {
            "maxT_family": "THREE_PRIMARY_ARM_DIRECTION_TESTS_ONLY",
            "destruction_and_regional_cross_family_adjustment": False,
            "preregistered_gate_semantics_preserved": True,
        },
        "target_free_evidence_and_p_comparator_sealed_before_role_join": True,
        "shard_receipts": shard_receipts,
        "source_bindings": {
            "authority_sha256": authority_sha256,
            "authority_logical_sha256": authority["logical_sha256"],
            "i0_seal_sha256": file_sha256(i0_seal_path),
            "i0_seal_validation_sha256": file_sha256(i0_seal_validation_path),
            "p_v2_utility_comparator_sha256": file_sha256(p_comparator_path),
            "p_v2_utility_comparator_validation_sha256": file_sha256(
                p_comparator_validation_path
            ),
            "ordered_prejoin_shards_sha256": canonical_sha256(shard_receipts),
        },
        "postjoin_model_forward_count": 0,
        "opened_sealed_access_count": 0,
        "full_gallery_retrieval_claim": False,
        "hold_switch_claim": False,
        "ownership_claim": False,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    require(
        output.resolve() == (RC_ROOT / authority["science_result_output"]).resolve(),
        "science result output path drift",
    )
    atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--prejoin-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--i0-seal", type=Path, required=True)
    parser.add_argument("--i0-seal-validation", type=Path, required=True)
    parser.add_argument("--p-utility-comparator", type=Path, required=True)
    parser.add_argument("--p-utility-comparator-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(
        args.authority,
        args.prejoin_root,
        args.validation_root,
        args.i0_seal,
        args.i0_seal_validation,
        args.p_utility_comparator,
        args.p_utility_comparator_validation,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision_A": result["decision_A"],
                "spatial_boundary": result["spatial_boundary"],
                "decision_B": result["decision_B"],
                "logical_sha256": result["logical_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
