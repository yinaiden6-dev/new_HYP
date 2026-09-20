"""Pure cross-fit helpers for the split SR0-MT I0 ledger chain.

The target-free prejoin and metadata-only postjoin are intentionally implemented
by separate executables with an independently validated file barrier.  This
module therefore exposes no combined builder and no alternate ranked-C128 join.
It only supplies canonical JSON hashing and the frozen fold topology used after
the Phase-B metadata gate has opened.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any, Mapping, Sequence


class SR0MTI0LedgerError(RuntimeError):
    """Raised when a cross-fit or canonical-ledger invariant is violated."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SR0MTI0LedgerError(message)


def build_nested_crossfit(role_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Use the frozen four group folds as three-way inner folds per outer split."""

    roles = sorted(role_records, key=lambda item: int(item["query_ordinal"]))
    _require(len(roles) == 600, "cross-fit role population drift")
    _require(
        len({str(item["query_id"]) for item in roles}) == 600,
        "cross-fit query IDs are not unique",
    )
    by_group: dict[str, set[int]] = defaultdict(set)
    by_identity: dict[str, set[int]] = defaultdict(set)
    for item in roles:
        fold = int(item["inner_fold"])
        _require(fold in (1, 2, 3, 4), "cross-fit fold outside 1..4")
        by_group[str(item["supergroup"])].add(fold)
        by_identity[str(item["identity"])].add(fold)
    _require(
        all(len(folds) == 1 for folds in by_group.values()),
        "supergroup leaks across outer folds",
    )
    _require(
        all(len(folds) == 1 for folds in by_identity.values()),
        "identity leaks across outer folds",
    )

    outer_payloads: list[dict[str, Any]] = []
    for outer in (1, 2, 3, 4):
        train = [item for item in roles if int(item["inner_fold"]) != outer]
        heldout = [item for item in roles if int(item["inner_fold"]) == outer]
        inner_payloads = []
        for inner in sorted({1, 2, 3, 4} - {outer}):
            inner_train = [item for item in train if int(item["inner_fold"]) != inner]
            inner_heldout = [item for item in train if int(item["inner_fold"]) == inner]
            train_groups = {str(item["supergroup"]) for item in inner_train}
            heldout_groups = {str(item["supergroup"]) for item in inner_heldout}
            train_ids = {str(item["identity"]) for item in inner_train}
            heldout_ids = {str(item["identity"]) for item in inner_heldout}
            _require(not (train_groups & heldout_groups), "inner supergroup leakage")
            _require(not (train_ids & heldout_ids), "inner identity leakage")
            inner_payloads.append(
                {
                    "inner_heldout_fold": inner,
                    "inner_train_source_folds": sorted(
                        {1, 2, 3, 4} - {outer, inner}
                    ),
                    "train_query_count": len(inner_train),
                    "heldout_query_count": len(inner_heldout),
                    "train_identity_count": len(train_ids),
                    "heldout_identity_count": len(heldout_ids),
                    "train_supergroup_count": len(train_groups),
                    "heldout_supergroup_count": len(heldout_groups),
                    "train_query_ids_sha256": canonical_sha256(
                        [str(item["query_id"]) for item in inner_train]
                    ),
                    "heldout_query_ids_sha256": canonical_sha256(
                        [str(item["query_id"]) for item in inner_heldout]
                    ),
                    "identity_disjoint": True,
                    "supergroup_disjoint": True,
                }
            )
        train_groups = {str(item["supergroup"]) for item in train}
        heldout_groups = {str(item["supergroup"]) for item in heldout}
        train_ids = {str(item["identity"]) for item in train}
        heldout_ids = {str(item["identity"]) for item in heldout}
        _require(not (train_groups & heldout_groups), "outer supergroup leakage")
        _require(not (train_ids & heldout_ids), "outer identity leakage")
        outer_payloads.append(
            {
                "outer_fold": outer,
                "outer_train_query_count": len(train),
                "outer_heldout_query_count": len(heldout),
                "outer_train_identity_count": len(train_ids),
                "outer_heldout_identity_count": len(heldout_ids),
                "outer_train_supergroup_count": len(train_groups),
                "outer_heldout_supergroup_count": len(heldout_groups),
                "outer_train_query_ids_sha256": canonical_sha256(
                    [str(item["query_id"]) for item in train]
                ),
                "outer_heldout_query_ids_sha256": canonical_sha256(
                    [str(item["query_id"]) for item in heldout]
                ),
                "p_inner_oof_folds": inner_payloads,
                "p_outer_refit_train_query_ids_sha256": canonical_sha256(
                    [str(item["query_id"]) for item in train]
                ),
                "v_train_input": "CONCATENATED_INNER_OOF_P_LOCKS_ONLY",
                "outer_identity_disjoint": True,
                "outer_supergroup_disjoint": True,
            }
        )
    value = {
        "schema_version": "rc_dino_rcde_sr0_mt_nested_crossfit_v1_20260815",
        "outer_fold_count": 4,
        "inner_fold_count_per_outer": 3,
        "source_fold_reuse_rule": "INNER_HELDOUT_EQUALS_ORIGINAL_FOLD_WITH_OUTER_FOLD_REMOVED",
        "outer_folds": outer_payloads,
    }
    value["logical_sha256"] = canonical_sha256(value)
    return value


__all__ = [
    "SR0MTI0LedgerError",
    "canonical_sha256",
    "build_nested_crossfit",
]
