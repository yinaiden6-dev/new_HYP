from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))

import freeze_dino_rcde_track_r_relative_e1_episode_reduction_authority_v119_r2 as freezer  # noqa: E402
import reduce_dino_rcde_track_r_relative_e1_episode_ledgers_v1 as reducer  # noqa: E402
import validate_dino_rcde_track_r_relative_e1_episode_ledgers_v1 as validator  # noqa: E402


MODULES = (freezer, reducer, validator)


@pytest.mark.parametrize("module", MODULES)
def test_every_consumer_rejects_a_cross_revision_shard(module) -> None:
    expected = "a" * 64
    valid = {"source_bindings": {"authority_sha256": expected}}
    module.require_shard_source_authority(valid, expected, 17)

    for invalid in (
        {},
        {"source_bindings": {}},
        {"source_bindings": {"authority_sha256": "b" * 64}},
    ):
        with pytest.raises(RuntimeError, match="V118 source-authority SHA drift"):
            module.require_shard_source_authority(invalid, expected, 17)


@pytest.mark.parametrize("module", (reducer, validator))
def test_reduction_and_independent_validator_recheck_bound_current_v118_r2(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent_path = tmp_path / module.PARENT_AUTHORITY_PATH
    parent_path.parent.mkdir(parents=True)
    parent = {
        "schema_version": module.PARENT_AUTHORITY_SCHEMA,
        "status": module.PARENT_AUTHORITY_STATUS,
        "authority_revision": 2,
    }
    parent_path.write_text(json.dumps(parent, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "RC_ROOT", tmp_path)
    parent_sha256 = module.file_sha256(parent_path)
    authority = {
        "parent_v118_authority_revision": 2,
        "shard_source_authority_sha256": parent_sha256,
        "all_shard_source_authority_match": True,
        "bindings": {
            "parent_authority_v118": {
                "path": module.PARENT_AUTHORITY_PATH,
                "sha256": parent_sha256,
            }
        },
    }
    assert module.bound_parent_v118_sha256(authority) == parent_sha256

    parent["authority_revision"] = 1
    parent_path.write_text(json.dumps(parent, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="V118-r2 physical SHA drift"):
        module.bound_parent_v118_sha256(authority)


def test_v119_numbering_and_all_fifty_shard_guard_remain_frozen() -> None:
    source = (
        ROOT
        / "programs/freeze_dino_rcde_track_r_relative_e1_episode_reduction_authority_v119_r2.py"
    ).read_text(encoding="utf-8")
    assert 'AUTH = "registry/current_authority_v119_20260821.json"' in source
    assert 'PARENT = "registry/current_authority_v118_20260821.json"' in source
    assert "for ordinal in range(50):" in source
    assert "require_shard_source_authority(shard, parent_sha256, ordinal)" in source
    assert '"parent_v118_authority_revision": 2' in source
    assert '"authority_revision": 2' in source
    assert '"runtime_episode_address_repair"' in source
    assert '"all_shard_source_authority_match": True' in source
