from __future__ import annotations

import inspect
from pathlib import Path

import torch

from dino_rcde_bag_statistics_independent_v1 import (
    independent_evaluate_bag_completion,
)
from rc_aslo_xf.dino_rcde_bag_completion_v1 import (
    CANDIDATE_COUNT,
    deterministic_candidate_reorder,
    deterministic_identity_derangement,
    evaluate_bag_completion,
    permuted_delta,
)


RC_ROOT = Path(__file__).resolve().parents[1]


def _delta() -> torch.Tensor:
    base = torch.arange(CANDIDATE_COUNT, dtype=torch.float32)
    value = base[:, None] - base[None, :]
    value.diagonal().zero_()
    return value


def test_c_bind_is_complete_fixed_point_free_and_identity_disjoint() -> None:
    rows = tuple(range(1000, 1000 + CANDIDATE_COUNT))
    identities = tuple(f"id-{index}" for index in range(CANDIDATE_COUNT))
    order = deterministic_identity_derangement(rows, identities, query_id="query-a")
    assert sorted(order) == list(range(CANDIDATE_COUNT))
    assert all(index != source for index, source in enumerate(order))
    assert all(identities[index] != identities[source] for index, source in enumerate(order))
    assert order == deterministic_identity_derangement(rows, identities, query_id="query-a")


def test_candidate_reorder_has_exact_inverse_closure() -> None:
    rows = tuple(range(CANDIDATE_COUNT))
    order = deterministic_candidate_reorder(rows, query_id="query-b")
    inverse = [0] * CANDIDATE_COUNT
    for new_position, source_position in enumerate(order):
        inverse[source_position] = new_position
    delta = _delta()
    assert torch.equal(permuted_delta(permuted_delta(delta, order), inverse), delta)


def _rows() -> list[dict[str, object]]:
    fold_group_counts = (13, 12, 12, 12)
    groups: list[tuple[str, int]] = []
    for fold, count in enumerate(fold_group_counts, start=1):
        groups.extend((f"group-{fold}-{index}", fold) for index in range(count))
    rows = []
    for index in range(594):
        group, fold = groups[index % len(groups)]
        rows.append(
            {
                "query_id": f"query-{index}",
                "execution_ordinal": index,
                "group_sha256": group,
                "outer_fold": fold,
                "raw_margin": -1.0 if (index // len(groups)) % 2 == 0 else 1.0,
                "real_margin": 1.0,
                "c_bind_margin": -1.0,
            }
        )
    return rows


def test_frozen_statistics_can_issue_scoped_go() -> None:
    result = evaluate_bag_completion(_rows(), repetitions=199)
    assert result["bag_fixed_decoder_gate"] is True
    assert result["decision"] == "DINO_RCDE_BAG_FIXED_DECODER_CANDIDATE_BOUND_NONSPATIAL_EVIDENCE_GO"
    assert result["C_BIND"]["gate"] is True


def test_frozen_statistics_reject_this_decoder_only() -> None:
    rows = _rows()
    for row in rows:
        row["real_margin"] = -1.0
        row["c_bind_margin"] = -1.0
    result = evaluate_bag_completion(rows, repetitions=199)
    assert result["bag_fixed_decoder_gate"] is False
    assert result["decision"] == "DINO_RCDE_BAG_FIXED_DECODER_CANDIDATE_BOUND_EVIDENCE_NO_GO"


def test_primary_and_independent_statistics_are_exact() -> None:
    primary = evaluate_bag_completion(_rows(), repetitions=199)
    independent = independent_evaluate_bag_completion(_rows(), repetitions=199)
    assert primary == independent


def test_independent_statistics_does_not_import_primary_evaluator() -> None:
    source = (RC_ROOT / "programs/dino_rcde_bag_statistics_independent_v1.py").read_text(encoding="utf-8")
    assert "evaluate_bag_completion" not in source.replace("independent_evaluate_bag_completion", "")
    assert "from rc_aslo_xf.dino_rcde_bag_completion_v1" not in source


def test_contract_keeps_scientific_boundary() -> None:
    contract = (RC_ROOT / "plan/DINO_RCDE_BAG_FULL594_COMPLETION_CONTRACT_V1_20260827.md").read_text(encoding="utf-8")
    assert "does not validate a deployable dispersed multi-component model" in contract
    assert "rejects only this frozen BAG decoder" in contract
    source = inspect.getsource(evaluate_bag_completion)
    assert "FULL_GALLERY" not in source


def test_producer_binds_exact_historical_foldset_validation_status() -> None:
    source = (RC_ROOT / "programs/materialize_dino_rcde_bag_full594_shard_v1.py").read_text(encoding="utf-8")
    assert "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_VALIDATION_PASS" in source
