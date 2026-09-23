#!/usr/bin/env python3
"""Independently replay the minimal Track-R I0 science metadata seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from dino_rcde_track_r_v121_lineage_v1 import (  # noqa: E402
    read_authority as read_frozen_authority,
    validate_runtime_bindings,
)


AUTHORITY = "registry/current_authority_v123_20260822.json"
AUTHORITY_SCHEMA = "rc_current_authority_v123_20260822"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_AUTHORIZED"
SEAL_SCHEMA = "rc_dino_rcde_track_r_i0_science_metadata_v1_20260822"
SEAL_STATUS = "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_SEALED"
VALIDATION_SCHEMA = "rc_dino_rcde_track_r_i0_science_metadata_validation_v1_20260822"
VALIDATION_STATUS = "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_INDEPENDENT_VALIDATION_PASS"
EXCLUDED = (25, 26, 101, 346, 354, 470)


class MetadataValidationError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise MetadataValidationError(message)


def fsha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def binding(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(isinstance(item, Mapping), f"binding absent: {name}")
    path = (ROOT / str(item["path"])).resolve(strict=True)
    require(
        path.is_relative_to(ROOT)
        and path.is_file()
        and not path.is_symlink()
        and fsha(path) == item.get("sha256"),
        f"binding drift: {name}",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authority_path = args.authority.resolve(strict=True)
    require(authority_path == (ROOT / AUTHORITY).resolve(strict=True), "authority path drift")
    authority, _ = read_frozen_authority(
        authority_path,
        root=ROOT,
        expected_relative=AUTHORITY,
        expected_schema=AUTHORITY_SCHEMA,
        expected_status=AUTHORITY_STATUS,
    )
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("metadata_seal_validation_authorized") is True,
        "metadata validation authority drift",
    )
    validate_runtime_bindings(
        authority,
        root=ROOT,
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
    seal_path = args.seal.resolve(strict=True)
    require(seal_path == (ROOT / str(authority["i0_seal_output"])).resolve(strict=True), "seal path drift")
    seal = read(seal_path)
    require(
        seal.get("schema_version") == SEAL_SCHEMA
        and seal.get("status") == SEAL_STATUS
        and seal.get("logical_sha256") == logical(seal)
        == authority.get("expected_i0_seal_logical_sha256")
        and seal.get("query_count") == 594
        and seal.get("group_count") == 49
        and seal.get("fold_group_count") == {"1": 13, "2": 12, "3": 12, "4": 12}
        and seal.get("group_fold_isolation") is True
        and seal.get("model_forward_count") == 0,
        "metadata seal envelope drift",
    )
    loss = read(binding(authority, "loss_join"))
    i0 = read(binding(authority, "i0_postjoin"))
    role = read(binding(authority, "role_free_pair"))["pair_manifest"]["records"]
    folds = read(binding(authority, "fold_schedule"))["records"]
    loss_by = {int(item["execution_ordinal"]): item for item in loss["episodes"]}
    i0_by = {int(item["execution_ordinal"]): item for item in i0["records"]}
    role_by = {int(item["execution_ordinal"]): item for item in role}
    seal_by = {int(item["execution_ordinal"]): item for item in seal["records"]}
    expected = set(range(600)) - set(EXCLUDED)
    require(set(loss_by) == set(role_by) == set(seal_by) == expected, "independent 594 closure drift")
    group_folds: dict[str, set[int]] = {}
    ordered_hashes = []
    for execution in sorted(expected):
        loss_row = loss_by[execution]
        meta = i0_by[execution]
        pair = role_by[execution]
        fold = folds[execution]
        observed = seal_by[execution]
        members = pair["members"]
        by_key = {str(item["candidate_key"]): item for item in members}
        target = str(loss_row["target_candidate_key"])
        rival = str(loss_row["rival_candidate_key"])
        target_ordinal = int(loss_row["target_member_ordinal"])
        rival_ordinal = int(loss_row["rival_member_ordinal"])
        group = hashlib.sha256(str(meta["supergroup"]).encode("utf-8")).hexdigest()
        require(
            loss_row["query_id"] == meta["query_id"] == pair["query_id"] == fold["query_id"] == observed["query_id"]
            and loss_row["query_source_image_sha256"] == meta["source_image_sha256"] == pair["query_source_image_sha256"] == fold["source_image_sha256"] == observed["query_source_image_sha256"]
            and int(meta["outer_fold"]) == int(fold["inner_fold"]) == int(observed["outer_fold"])
            and by_key[target]["candidate_physical_row"] == meta["target_physical_row"] == observed["target_physical_row"]
            and by_key[rival]["candidate_physical_row"] == meta["strongest_rival_physical_row"] == observed["rival_physical_row"]
            and {target_ordinal, rival_ordinal} == {0, 1}
            and members[target_ordinal]["candidate_key"] == target == observed["target_candidate_key"]
            and members[rival_ordinal]["candidate_key"] == rival == observed["rival_candidate_key"]
            and observed["target_member_ordinal"] == target_ordinal
            and observed["rival_member_ordinal"] == rival_ordinal
            and observed["group_sha256"] == group
            and observed["pair_sha256"] == pair["pair_sha256"]
            and observed["candidate_axis_sha256"] == pair["candidate_axis_sha256"]
            and observed["target_raw_score_bits"] == meta["target_raw_score_bits"]
            and observed["rival_raw_score_bits"] == meta["strongest_rival_raw_score_bits"],
            f"independent metadata record drift: {execution}",
        )
        for bits in (observed["target_raw_score_bits"], observed["rival_raw_score_bits"]):
            value = struct.unpack(">d", bytes.fromhex(bits))[0]
            require(value == value and abs(value) != float("inf"), "nonfinite score")
        require(observed["record_sha256"] == logical({key: item for key, item in observed.items() if key != "record_sha256"}), "record hash drift")
        ordered_hashes.append(observed["record_sha256"])
        group_folds.setdefault(group, set()).add(int(observed["outer_fold"]))
    require(
        len(group_folds) == 49
        and all(len(value) == 1 for value in group_folds.values())
        and seal["record_sequence_sha256"] == canonical_sha256(ordered_hashes),
        "independent group/sequence closure drift",
    )
    result: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": VALIDATION_STATUS,
        "validation_pass": True,
        "seal_sha256": fsha(seal_path),
        "seal_logical_sha256": seal["logical_sha256"],
        "query_count": 594,
        "group_count": 49,
        "fold_group_count": seal["fold_group_count"],
        "group_fold_isolation": True,
        "model_forward_count": 0,
        "opened_sealed_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical(result)
    output = args.output.resolve()
    require(output == (ROOT / str(authority["i0_seal_validation_output"])).resolve() and not output.exists(), "validation output drift")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    output.chmod(0o444)
    print(json.dumps({"status": result["status"], "logical_sha256": result["logical_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
