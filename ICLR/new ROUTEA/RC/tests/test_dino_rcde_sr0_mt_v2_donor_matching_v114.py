from __future__ import annotations

import inspect

import freeze_dino_rcde_sr0_mt_v2_donor_matching_authority_v114 as F
import materialize_dino_rcde_sr0_mt_v2_donor_matching_v1 as P
import validate_dino_rcde_sr0_mt_v2_donor_matching_v1 as V


def test_v114_authority_adopts_scheduler_migration_without_training() -> None:
    authority = F.build()
    assert authority["scheduler_migration_adoption_authorized"] is True
    assert authority["projection_receipt_read_authorized"] is True
    assert authority["optimization_identity_supergroup_read_authorized"] is True
    assert authority["final_donor_matching_authorized"] is True
    assert authority["lock_payload_deserialization_authorized"] is False
    assert authority["model_load_authorized"] is False
    assert authority["training_authorized"] is False
    assert authority["V_training_authorized"] is False
    assert authority["scientific_GO_or_NO_GO"] is None
    assert authority["next_authorized_stage"] is None
    assert {key for key, value in authority.items() if value is True} == {
        "projection_receipt_read_authorized",
        "scheduler_migration_adoption_authorized",
        "optimization_identity_supergroup_read_authorized",
        "final_donor_matching_authorized",
    }
    assert "moved_launcher" in authority["bindings"]
    assert "scheduler_override_addendum" in authority["bindings"]
    assert "scheduler_migration_receipt" in authority["bindings"]


def test_frozen_graph_is_empty_and_reason_partition_is_exact() -> None:
    authority = F.build()
    result = P.build_result(authority, "0" * 64)
    assert result["status"] == P.NO_EDGE_STATUS
    assert result["scope_node_count"] == 2376
    assert result["direction_record_count"] == 4752
    assert result["static_edge_count"] == 225792
    assert result["geometry_edge_count"] == 0
    assert result["matched_node_count"] == 0
    assert result["null_static_zero_degree_count"] == 84
    assert result["null_geometry_zero_degree_count"] == 2292
    assert result["null_geometry_hall_unmatched_count"] == 0
    assert result["donor_ledger_eligible"] is False
    assert result["V_training_authorized"] is False
    V.validate_graph(authority, result, "0" * 64, replay_sacct=False)


def test_independent_validator_does_not_import_matching_producer() -> None:
    source = inspect.getsource(V)
    assert "materialize_dino_rcde_sr0_mt_v2_donor_matching_v1" not in source
    assert "hopcroft_like" in source
    assert 'result.get("geometry_edge_count")' in source
    assert "total_geometry == 0" in source
    assert "NO_GO_EXACT_GEOMETRY_EMPTY_GRAPH" in source


def test_canonical_matching_fixture_is_maximum_and_deterministic() -> None:
    adjacency = [[1, 2], [0, 2], [0, 1]]
    first = P.canonical_maximum_matching(adjacency)
    second = P.canonical_maximum_matching(adjacency)
    assert first == second
    assert len({item for item in first if item is not None}) == 3
    assert P.maximum_cardinality(adjacency) == 3
