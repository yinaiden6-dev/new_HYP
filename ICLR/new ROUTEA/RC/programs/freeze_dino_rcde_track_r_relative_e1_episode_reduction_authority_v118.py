#!/usr/bin/env python3
"""Freeze V118 only after all 50 E1 shard validations PASS."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as H


ROOT = Path(__file__).resolve().parents[1]
AUTH = "registry/current_authority_v119_20260821.json"
PARENT = "registry/current_authority_v118_20260821.json"
STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_REDUCTION_AUTHORIZED"
PARENT_SCHEMA = "rc_current_authority_v118_20260821"
PARENT_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARDS_AUTHORIZED"
SHARD_SCHEMA = "rc_dino_rcde_track_r_relative_e1_episode_shard_v1_20260821"
SHARD_STATUS = "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_READY"


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_shard_source_authority(
    shard: Mapping[str, Any], expected_sha256: str, shard_ordinal: int
) -> None:
    source_bindings = shard.get("source_bindings")
    req(
        isinstance(source_bindings, Mapping)
        and source_bindings.get("authority_sha256") == expected_sha256,
        f"E1 shard{shard_ordinal} V118 source-authority SHA drift",
    )


def latest() -> tuple[int, Path]:
    pattern = re.compile(r"current_authority_v([1-9][0-9]*)_[0-9]{8}\.json")
    rows = [(int(match.group(1)), path.resolve()) for path in (ROOT / "registry").glob("current_authority_v*_*.json") if (match := pattern.fullmatch(path.name)) and path.is_file() and not path.is_symlink()]
    version = max(item[0] for item in rows)
    paths = [path for item_version, path in rows if item_version == version]
    req(len(paths) == 1, "latest authority alias")
    return version, paths[0]


def build() -> dict[str, Any]:
    version, current = latest()
    parent_path = (ROOT / PARENT).resolve(strict=True)
    self_path = (ROOT / AUTH).resolve(strict=False)
    req((version == 118 and current == parent_path) or (version == 119 and current == self_path and self_path.is_file()), "V118/V119 authority state drift")
    parent = json.loads(parent_path.read_text())
    req(
        parent.get("schema_version") == PARENT_SCHEMA
        and parent.get("status") == PARENT_STATUS
        and parent.get("authority_revision") == 2,
        "V118-r2 parent drift",
    )
    parent_sha256 = H.file_sha(parent_path)
    producer_root = ROOT / parent["producer_root"]
    validation_root = ROOT / parent["validation_root"]
    shard_bindings = []
    total_episodes = total_queries = total_excluded = 0
    for ordinal in range(50):
        start = ordinal * 12
        stop = start + 12
        producer = producer_root / f"shard_{start:03d}_{stop:03d}.pt"
        validation = validation_root / f"shard_{start:03d}_{stop:03d}.validation.json"
        req(producer.is_file() and validation.is_file(), f"E1 shard{ordinal} absent")
        shard = torch.load(
            producer, map_location="cpu", weights_only=True, mmap=True
        )
        req(
            isinstance(shard, Mapping)
            and shard.get("schema_version") == SHARD_SCHEMA
            and shard.get("status") == SHARD_STATUS
            and shard.get("shard_ordinal") == ordinal
            and shard.get("logical_sha256") == H.logical(shard),
            f"E1 shard{ordinal} producer envelope/logical drift",
        )
        require_shard_source_authority(shard, parent_sha256, ordinal)
        value = json.loads(validation.read_text())
        req(
            value.get("status") == "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_SHARD_INDEPENDENT_VALIDATION_PASS"
            and value.get("validation_pass") is True
            and value.get("shard_ordinal") == ordinal
            and value.get("producer_shard_sha256") == H.file_sha(producer)
            and value.get("producer_shard_logical_sha256")
            == shard["logical_sha256"],
            f"E1 shard{ordinal} validation drift",
        )
        total_episodes += int(value["episode_count"])
        total_queries += int(value["eligible_query_count"])
        total_excluded += int(value["excluded_query_count"])
        shard_bindings.append({
            "shard_ordinal": ordinal,
            "producer": H.bind(producer.relative_to(ROOT).as_posix(), immutable=True),
            "validation": H.bind(validation.relative_to(ROOT).as_posix(), with_logical=True, immutable=True),
        })
    req(total_queries == 594 and total_excluded == 6 and total_episodes == 1782, "E1 594/6/1782 closure drift")
    bindings = {
        "parent_authority_v118": H.bind(PARENT, with_logical=True, immutable=True),
        "reducer": H.bind("programs/reduce_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py"),
        "validator": H.bind("programs/validate_dino_rcde_track_r_relative_e1_episode_ledgers_v1.py"),
        "episode_core": H.bind("src/rc_aslo_xf/dino_rcde_track_r_episode_v1.py"),
        "test": H.bind("tests/test_dino_rcde_track_r_episode_v1.py"),
        "cross_revision_test": H.bind(
            "tests/test_dino_rcde_track_r_v119_cross_revision_guard_v1.py"
        ),
        "freezer": H.bind("programs/freeze_dino_rcde_track_r_relative_e1_episode_reduction_authority_v118.py"),
        "controller": H.bind("slurm/dino_rcde_track_r_post_e1_controller_v3.sbatch"),
        "orchestration_test": H.bind(
            "tests/test_dino_rcde_track_r_post_e1_controller_v3.py"
        ),
        "shards": {
            "count": 50,
            "logical_sha256": H.logical({"rows": shard_bindings}),
            "rows": shard_bindings,
        },
    }
    authority = {
        "schema_version": "rc_current_authority_v119_20260821",
        "status": STATUS,
        "stage": "TRACK_R_RELATIVE_E1_EPISODE_LEDGER_REDUCTION",
        "claim_level": "ENGINEERING_FOUR_COMPACT_INNER_OOF_EPISODE_LEDGERS_ONLY",
        "parent_v118_authority_revision": 2,
        "shard_source_authority_sha256": parent_sha256,
        "all_shard_source_authority_match": True,
        "query_count": 594,
        "excluded_query_count": 6,
        "total_episode_count": 1782,
        "expected_per_outer_fold": {"1": 445, "2": 446, "3": 445, "4": 446},
        "episode_reduction_authorized": True,
        "episode_reduction_validation_authorized": True,
        "token_load_authorized": False,
        "model_load_authorized": False,
        "training_authorized": False,
        "heldout_scoring_authorized": False,
        "protected_access_authorized": False,
        "automatic_stage_advance": False,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": None,
        "output_root": "results/dino_rcde_track_r_relative_e1_ledgers_v1",
        "validation_output": "results/dino_rcde_track_r_relative_e1_ledgers_validation_v1/result.json",
        "resource_contract": {"partition": "cpuonly", "cpus_per_task": 4, "memory_megabytes": 65536, "walltime_seconds": 3600},
        "bindings": bindings,
    }
    authority["logical_sha256"] = H.logical(authority)
    return authority


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / AUTH)
    args = parser.parse_args()
    req(args.output.resolve(strict=False) == (ROOT / AUTH).resolve(strict=False), "output drift")
    value = build()
    H.atomic(args.output, value)
    print(json.dumps({"status": value["status"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
