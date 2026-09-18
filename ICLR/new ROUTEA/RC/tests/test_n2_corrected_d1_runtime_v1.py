from __future__ import annotations

import pytest
import torch

from rc_aslo_xf.n2_corrected_d1_runtime_v1 import (
    CANONICAL_FOLD_COUNTS,
    N2CorrectedD1RuntimeError,
    fold_assignment_sha256,
    join_query_ids_to_canonical_folds,
    natural_c128,
    reduce_corrected_full_gallery_scores,
    validate_canonical_987_ledger,
    validate_corrected_identity_axis,
)


def corrected_labels() -> list[str]:
    labels = [f"identity-{row:04d}" for row in range(5413)]
    labels[714] = "Biogen_21"
    labels[715] = "Biogen_21"
    return labels


def canonical_ledger() -> list[dict]:
    folds = [fold for fold, count in CANONICAL_FOLD_COUNTS.items() for _ in range(count)]
    tracks = ("outcome", "difficult", "new_difficult_train")
    return [
        {
            "query_id": f"Q-{ordinal:04d}",
            "query_ordinal": ordinal,
            "heldout_fold": folds[ordinal],
            "track": tracks[ordinal % len(tracks)],
            "grid_h": 24,
            "grid_w": 32,
            "source_image_sha256": f"{ordinal:064x}"[-64:],
        }
        for ordinal in range(987)
    ]


def test_corrected_5412_reducer_max_tie_and_natural_c128() -> None:
    labels = corrected_labels()
    receipt = validate_corrected_identity_axis(labels)
    assert receipt["physical_row_count"] == 5413
    assert receipt["corrected_identity_count"] == 5412

    scores = torch.zeros(5413, dtype=torch.float64)
    scores[714] = 5.0
    scores[715] = 6.0
    reduced = reduce_corrected_full_gallery_scores(scores, labels)
    slot = reduced.identity_values.index("Biogen_21")
    assert float(reduced.identity_scores[slot]) == 6.0
    assert int(reduced.representative_physical_rows[slot]) == 715

    scores[714] = 6.0
    tied = reduce_corrected_full_gallery_scores(scores, labels)
    slot = tied.identity_values.index("Biogen_21")
    assert int(tied.representative_physical_rows[slot]) == 714

    all_tied = reduce_corrected_full_gallery_scores(torch.zeros_like(scores), labels)
    c128 = natural_c128(all_tied)
    assert len(c128.identities) == len(set(c128.identities)) == 128
    assert len(c128.representative_physical_rows) == len(set(c128.representative_physical_rows)) == 128
    assert c128.representative_physical_rows == tuple(range(128))


def test_legacy_5404_axis_fails_closed() -> None:
    labels = [f"legacy-{slot:04d}" for slot in range(5404)]
    labels.extend(labels[:9])
    assert len(labels) == 5413 and len(set(labels)) == 5404
    with pytest.raises(N2CorrectedD1RuntimeError, match="5,404"):
        validate_corrected_identity_axis(labels)
    with pytest.raises(N2CorrectedD1RuntimeError, match="5,404"):
        reduce_corrected_full_gallery_scores(torch.zeros(5413), labels)


def test_query_id_only_join_uses_canonical_987_fold() -> None:
    ledger = canonical_ledger()
    receipt = validate_canonical_987_ledger(ledger)
    assert receipt["query_count"] == 987
    assert receipt["fold_counts"] == CANONICAL_FOLD_COUNTS
    requested = ["Q-0001", "Q-0986", "Q-0500"]
    joined = join_query_ids_to_canonical_folds(requested, ledger)
    assert [item.query_id for item in joined] == requested
    assert [item.query_ordinal for item in joined] == [1, 986, 500]

    # Pair64-like metadata is deliberately outside the join interface.
    poisoned_pair_rows = [
        {"query_id": query_id, "inner_fold": 99, "execution_ordinal": -123}
        for query_id in requested
    ]
    replay = join_query_ids_to_canonical_folds(
        [row["query_id"] for row in poisoned_pair_rows], ledger
    )
    assert fold_assignment_sha256(replay) == fold_assignment_sha256(joined)


def test_join_rejects_duplicate_or_unknown_query_id() -> None:
    ledger = canonical_ledger()
    with pytest.raises(N2CorrectedD1RuntimeError, match="unique"):
        join_query_ids_to_canonical_folds(["Q-0001", "Q-0001"], ledger)
    with pytest.raises(N2CorrectedD1RuntimeError, match="absent"):
        join_query_ids_to_canonical_folds(["UNKNOWN"], ledger)
