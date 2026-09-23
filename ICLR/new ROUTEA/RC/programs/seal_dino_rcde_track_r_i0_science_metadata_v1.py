#!/usr/bin/env python3
"""Seal the minimal target-bearing metadata used by Track-R science."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping


RC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC_ROOT / "src"))
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256  # noqa: E402
from dino_rcde_track_r_v121_lineage_v1 import (  # noqa: E402
    read_authority as read_frozen_authority,
    validate_runtime_bindings,
)


AUTHORITY_PATH = "registry/current_authority_v123_20260822.json"
AUTHORITY_SCHEMA = "rc_current_authority_v123_20260822"
AUTHORITY_STATUS = "DINO_RCDE_TRACK_R_SCIENTIFIC_REDUCTION_AUTHORIZED"
SCHEMA = "rc_dino_rcde_track_r_i0_science_metadata_v1_20260822"
STATUS = "DINO_RCDE_TRACK_R_I0_SCIENCE_METADATA_SEALED"
EXCLUDED = (25, 26, 101, 346, 354, 470)
QUERY_COUNT = 600
VALID_QUERY_COUNT = 594
EXPECTED_GROUPS = 49
EXPECTED_FOLD_GROUPS = {1: 13, 2: 12, 3: 12, 4: 12}


class TrackRMetadataSealError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise TrackRMetadataSealError(message)


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


def _group_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_score_bits(value: object) -> str:
    require(isinstance(value, str) and len(value) == 16, "RAW score bits drift")
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    require(number == number and abs(number) != float("inf"), "RAW score nonfinite")
    return value


def build_sealed_metadata(
    *,
    loss_join: Mapping[str, Any],
    i0_postjoin: Mapping[str, Any],
    role_free: Mapping[str, Any],
    fold_schedule: Mapping[str, Any],
    source_bindings: Mapping[str, str],
) -> dict[str, Any]:
    require(
        loss_join.get("status") == "RCDE_SR0_MT_LOSS_JOIN_READY"
        and loss_join.get("episode_count") == VALID_QUERY_COUNT
        and loss_join.get("excluded_execution_ordinals") == list(EXCLUDED)
        and loss_join.get("score_rank_outcome_read_count") == 0,
        "score-free loss join drift",
    )
    require(
        i0_postjoin.get("status") == "RCDE_SR0_MT_I0_POSTJOIN_LEDGER_SEALED"
        and i0_postjoin.get("query_count") == QUERY_COUNT
        and i0_postjoin.get("natural_target_hit_count") == VALID_QUERY_COUNT
        and i0_postjoin.get("natural_target_miss_count") == len(EXCLUDED),
        "I0 postjoin population drift",
    )
    pair_manifest = role_free.get("pair_manifest")
    require(
        role_free.get("status") == "RCDE_SR0_MT_ROLE_FREE_PAIR_ADDRESS_READY"
        and role_free.get("role_free") is True
        and role_free.get("query_count") == VALID_QUERY_COUNT
        and isinstance(pair_manifest, Mapping)
        and isinstance(pair_manifest.get("records"), list),
        "role-free pair manifest drift",
    )
    folds = fold_schedule.get("records")
    require(
        fold_schedule.get("status")
        == "DINO_RCDE_PREJOIN_FOLDS_600_PHYSICALLY_ISOLATED"
        and isinstance(folds, list)
        and len(folds) == QUERY_COUNT,
        "fold schedule drift",
    )
    loss_by_execution = {
        int(item["execution_ordinal"]): item for item in loss_join["episodes"]
    }
    i0_by_execution = {
        int(item["execution_ordinal"]): item for item in i0_postjoin["records"]
    }
    pair_by_execution = {
        int(item["execution_ordinal"]): item for item in pair_manifest["records"]
    }
    fold_by_execution = dict(enumerate(folds))
    expected = set(range(QUERY_COUNT)) - set(EXCLUDED)
    require(
        set(loss_by_execution) == set(pair_by_execution) == expected
        and set(i0_by_execution) == set(range(QUERY_COUNT))
        and set(fold_by_execution) == set(range(QUERY_COUNT)),
        "594/600 metadata execution closure drift",
    )

    records: list[dict[str, Any]] = []
    group_folds: dict[str, set[int]] = {}
    for execution in sorted(expected):
        loss = loss_by_execution[execution]
        meta = i0_by_execution[execution]
        pair = pair_by_execution[execution]
        fold = fold_by_execution[execution]
        members = list(pair["members"])
        by_key = {str(item["candidate_key"]): item for item in members}
        target_key = str(loss["target_candidate_key"])
        rival_key = str(loss["rival_candidate_key"])
        require(
            len(members) == 2
            and set(by_key) == {target_key, rival_key}
            and loss.get("query_id")
            == meta.get("query_id")
            == pair.get("query_id")
            == fold.get("query_id")
            and loss.get("query_source_image_sha256")
            == meta.get("source_image_sha256")
            == pair.get("query_source_image_sha256")
            == fold.get("source_image_sha256")
            and int(meta.get("outer_fold")) == int(fold.get("inner_fold"))
            and meta.get("target_hit") is True,
            f"metadata query/source/fold closure drift: {execution}",
        )
        target_member = by_key[target_key]
        rival_member = by_key[rival_key]
        target_ordinal = int(loss["target_member_ordinal"])
        rival_ordinal = int(loss["rival_member_ordinal"])
        require(
            {target_ordinal, rival_ordinal} == {0, 1}
            and
            int(target_member["candidate_physical_row"])
            == int(meta["target_physical_row"])
            and int(rival_member["candidate_physical_row"])
            == int(meta["strongest_rival_physical_row"])
            and members[target_ordinal]["candidate_key"]
            == target_key
            and members[rival_ordinal]["candidate_key"]
            == rival_key,
            f"metadata target/rival row/member closure drift: {execution}",
        )
        outer_fold = int(meta["outer_fold"])
        group_sha256 = _group_sha256(str(meta["supergroup"]))
        group_folds.setdefault(group_sha256, set()).add(outer_fold)
        record: dict[str, Any] = {
            "execution_ordinal": execution,
            "query_id": str(meta["query_id"]),
            "query_source_image_sha256": str(meta["source_image_sha256"]),
            "outer_fold": outer_fold,
            "group_sha256": group_sha256,
            "pair_sha256": str(pair["pair_sha256"]),
            "candidate_axis_sha256": str(pair["candidate_axis_sha256"]),
            "target_candidate_key": target_key,
            "rival_candidate_key": rival_key,
            "target_member_ordinal": target_ordinal,
            "rival_member_ordinal": rival_ordinal,
            "target_physical_row": int(meta["target_physical_row"]),
            "rival_physical_row": int(meta["strongest_rival_physical_row"]),
            "target_raw_score_bits": _require_score_bits(
                meta["target_raw_score_bits"]
            ),
            "rival_raw_score_bits": _require_score_bits(
                meta["strongest_rival_raw_score_bits"]
            ),
        }
        record["record_sha256"] = logical_sha256(record)
        records.append(record)
    require(
        len(group_folds) == EXPECTED_GROUPS
        and all(len(value) == 1 for value in group_folds.values()),
        "49-group fold isolation drift",
    )
    fold_groups = {
        fold: sum(next(iter(value)) == fold for value in group_folds.values())
        for fold in (1, 2, 3, 4)
    }
    require(fold_groups == EXPECTED_FOLD_GROUPS, "fold group count drift")
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_level": "MINIMAL_TARGET_ROLE_RAW_STRATUM_AND_GROUP_METADATA_ONLY",
        "query_count": VALID_QUERY_COUNT,
        "excluded_execution_ordinals": list(EXCLUDED),
        "group_count": EXPECTED_GROUPS,
        "fold_group_count": {str(key): item for key, item in fold_groups.items()},
        "group_fold_isolation": True,
        "records": records,
        "record_sequence_sha256": canonical_sha256(
            [item["record_sha256"] for item in records]
        ),
        "source_bindings": dict(source_bindings),
        "model_load_count": 0,
        "model_forward_count": 0,
        "model_backward_count": 0,
        "model_update_count": 0,
        "opened_sealed_access_count": 0,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {path}")
    return value


def _binding_path(authority: Mapping[str, Any], name: str) -> Path:
    item = authority.get("bindings", {}).get(name)
    require(isinstance(item, Mapping), f"authority binding absent: {name}")
    path = (RC_ROOT / str(item["path"])).resolve(strict=True)
    require(
        path.is_relative_to(RC_ROOT)
        and path.is_file()
        and not path.is_symlink()
        and file_sha256(path) == item.get("sha256"),
        f"authority binding drift: {name}",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authority_path = args.authority.resolve(strict=True)
    require(
        authority_path == (RC_ROOT / AUTHORITY_PATH).resolve(strict=True),
        "V123 authority path drift",
    )
    authority, _ = read_frozen_authority(
        authority_path,
        root=RC_ROOT,
        expected_relative=AUTHORITY_PATH,
        expected_schema=AUTHORITY_SCHEMA,
        expected_status=AUTHORITY_STATUS,
    )
    require(
        authority.get("schema_version") == AUTHORITY_SCHEMA
        and authority.get("status") == AUTHORITY_STATUS
        and authority.get("metadata_seal_authorized") is True
        and authority.get("postjoin_model_forward_authorized") is False,
        "V123 metadata seal authority drift",
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
    paths = {
        name: _binding_path(authority, name)
        for name in ("loss_join", "i0_postjoin", "role_free_pair", "fold_schedule")
    }
    source_bindings = {f"{name}_sha256": file_sha256(path) for name, path in paths.items()}
    value = build_sealed_metadata(
        loss_join=_read_json(paths["loss_join"]),
        i0_postjoin=_read_json(paths["i0_postjoin"]),
        role_free=_read_json(paths["role_free_pair"]),
        fold_schedule=_read_json(paths["fold_schedule"]),
        source_bindings=source_bindings,
    )
    require(
        value["logical_sha256"] == authority.get("expected_i0_seal_logical_sha256"),
        "V123 expected I0 seal logical SHA drift",
    )
    output = args.output.resolve()
    require(
        output == (RC_ROOT / str(authority["i0_seal_output"])).resolve()
        and not output.exists(),
        "I0 seal output path/existence drift",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    output.chmod(0o444)
    print(json.dumps({"status": value["status"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
