from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "programs") not in sys.path:
    sys.path.insert(0, str(ROOT / "programs"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import reduce_dino_rcde_track_r_relative_e1_episode_ledgers_v1 as producer
import validate_dino_rcde_track_r_relative_e1_episode_ledgers_v1 as validator
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256
from rc_aslo_xf.dino_rcde_track_r_episode_v1 import validate_compact_episode


def unsigned_record_sha256(episode: dict[str, object]) -> str:
    return canonical_sha256(
        {
            key: copy.deepcopy(value)
            for key, value in episode.items()
            if key != "record_sha256"
        }
    )


def changed_fields(
    source: dict[str, object], runtime: dict[str, object]
) -> set[str]:
    return {
        key
        for key in source
        if source[key] != runtime[key]
    }


def test_collision_fixture_is_losslessly_disambiguated_and_independently_replayed() -> None:
    shard_path = (
        ROOT
        / "results/dino_rcde_track_r_relative_e1_shards_v1/producer/"
        "shard_000_012.pt"
    )
    shard = torch.load(
        shard_path, map_location="cpu", weights_only=True, mmap=True
    )
    first = copy.deepcopy(shard["episodes"][0])
    second = copy.deepcopy(first)
    second_execution = int(first["query"]["execution_ordinal"]) + 600
    second["query"]["execution_ordinal"] = second_execution
    for role in ("target", "strongest_rival"):
        member = second[role]
        hashes = []
        for direction in ("a_to_b", "b_to_a"):
            record = member["direction_records"][direction]
            record["query"]["execution_ordinal"] = second_execution
            record["record_sha256"] = canonical_sha256(
                {
                    key: copy.deepcopy(value)
                    for key, value in record.items()
                    if key != "record_sha256"
                }
            )
            hashes.append(record["record_sha256"])
        member["direction_record_sha256"] = hashes
    second["record_sha256"] = unsigned_record_sha256(second)
    validate_compact_episode(first)
    validate_compact_episode(second)
    assert first["episode_id"] == second["episode_id"]

    runtime_first, map_first = producer.normalize_ledger_runtime_episode_address(
        first
    )
    runtime_second, map_second = producer.normalize_ledger_runtime_episode_address(
        second
    )
    independent_first, independent_map_first = (
        validator.independently_normalize_ledger_runtime_episode_address(first)
    )
    independent_second, independent_map_second = (
        validator.independently_normalize_ledger_runtime_episode_address(second)
    )
    assert runtime_first == independent_first
    assert runtime_second == independent_second
    assert map_first == independent_map_first
    assert map_second == independent_map_second
    assert runtime_first["episode_id"] != runtime_second["episode_id"]
    assert changed_fields(first, runtime_first) == {
        "episode_id",
        "record_sha256",
    }
    assert changed_fields(second, runtime_second) == {
        "episode_id",
        "record_sha256",
    }
    assert map_first["changed_fields"] == ["episode_id", "record_sha256"]
    assert map_first["source_episode_id"] == first["episode_id"]
    assert map_first["source_record_sha256"] == first["record_sha256"]
    assert map_first["runtime_episode_id"] == runtime_first["episode_id"]
    assert map_first["runtime_record_sha256"] == runtime_first["record_sha256"]


@pytest.mark.skipif(
    os.environ.get("ROUTEA_FULL_SOURCE_DIAGNOSTIC") != "1",
    reason="full 1782-episode GPFS replay is compute-node opt-in",
)
def test_full_1782_source_diagnostic_reproduces_collision_and_closes_runtime_axis() -> None:
    root = ROOT / "results/dino_rcde_track_r_relative_e1_shards_v1/producer"
    paths = sorted(root.glob("shard_*.pt"))
    assert len(paths) == 50
    file_state = {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    }
    source_ids: dict[int, list[str]] = {fold: [] for fold in (1, 2, 3, 4)}
    runtime_ids: dict[int, list[str]] = {fold: [] for fold in (1, 2, 3, 4)}
    count = 0
    for path in paths:
        shard = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
        for source in shard["episodes"]:
            runtime, address = producer.normalize_ledger_runtime_episode_address(
                source
            )
            independent, independent_address = (
                validator.independently_normalize_ledger_runtime_episode_address(
                    source
                )
            )
            assert runtime == independent
            assert address == independent_address
            assert changed_fields(source, runtime) == {
                "episode_id",
                "record_sha256",
            }
            fold = int(source["query"]["outer_fold"])
            source_ids[fold].append(source["episode_id"])
            runtime_ids[fold].append(runtime["episode_id"])
            count += 1
    assert count == 1782
    assert {fold: len(rows) for fold, rows in source_ids.items()} == {
        1: 445,
        2: 446,
        3: 445,
        4: 446,
    }
    assert {
        fold: len(rows) - len(set(rows))
        for fold, rows in source_ids.items()
    } == {1: 2, 2: 2, 3: 3, 4: 2}
    assert all(len(rows) == len(set(rows)) for rows in runtime_ids.values())
    all_runtime = [
        episode_id
        for fold in (1, 2, 3, 4)
        for episode_id in runtime_ids[fold]
    ]
    assert len(all_runtime) == len(set(all_runtime)) == 1782
    assert {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    } == file_state
