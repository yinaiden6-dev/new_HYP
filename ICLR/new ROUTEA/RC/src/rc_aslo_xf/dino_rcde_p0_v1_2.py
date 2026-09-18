"""Pure contracts for the DINO-RCDE V1.2 P0 repair.

This module contains no model forward and reads no files.  It centralises the
parts of P0 that must be deterministic before E0 is authorised:

* corrected-identity score reduction;
* the model-visible canonical C128 order;
* fold-local inductive reference access;
* the frozen 6/1/1 negative sampler; and
* legacy-ledger availability accounting.

The formal validator intentionally reimplements the important calculations
instead of trusting this module or booleans emitted by the runner.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import struct
from typing import Any, Iterable, Mapping, Sequence


PHYSICAL_ROW_NAMESPACE = b"RCDE_PHYSICAL_ROW_V1_2\0"
MIDDLE_NAMESPACE = b"RCDE_MID_NEG_V1_2\0"
BROAD_NAMESPACE = b"RCDE_BROAD_NEG_V1_2\0"
EPISODE_ORDER_NAMESPACE = b"RCDE_EPISODE_ORDER_V1_2\0"


class RCDEP0V12Error(RuntimeError):
    """A fail-closed V1.2 P0 contract violation."""


def _utf8_field(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) >= 2**32:
        raise RCDEP0V12Error("hash field exceeds uint32 length")
    return struct.pack(">I", len(encoded)) + encoded


def physical_row_digest(row: int) -> bytes:
    """Return SHA256(namespace NUL || uint64_be(row))."""

    if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < 2**64:
        raise RCDEP0V12Error("physical row is outside uint64")
    return hashlib.sha256(PHYSICAL_ROW_NAMESPACE + struct.pack(">Q", row)).digest()


def selection_digest(
    namespace: bytes,
    *,
    query_sha256: str,
    corrected_identity: str,
    physical_row: int,
) -> bytes:
    """Hash a negative selection tuple with unambiguous field framing."""

    if namespace not in {MIDDLE_NAMESPACE, BROAD_NAMESPACE, EPISODE_ORDER_NAMESPACE}:
        raise RCDEP0V12Error("unregistered negative-selection namespace")
    try:
        query_digest = bytes.fromhex(query_sha256)
    except ValueError as exc:
        raise RCDEP0V12Error("query sha256 is not hexadecimal") from exc
    if len(query_digest) != 32:
        raise RCDEP0V12Error("query sha256 must contain 32 bytes")
    # The physical-row primitive itself consumes uint64 bytes.  The outer
    # selection tuple is deliberately text-framed: every variable-width field
    # is UTF-8 with a uint32 big-endian length, including digest hex strings.
    row_hash_hex = physical_row_digest(physical_row).hex()
    return hashlib.sha256(
        namespace
        + _utf8_field(query_sha256.lower())
        + _utf8_field(corrected_identity)
        + _utf8_field(row_hash_hex)
    ).digest()


def corrected_identities(setids: Sequence[str], repair: Mapping[str, Any]) -> list[str]:
    output = [str(value) for value in setids]
    for item in repair.get("filename_collision_overrides", []):
        output[int(item["physical_row"])] = str(item["corrected_identity"])
    for component in repair.get("verified_byte_identical_duplicate_components", []):
        identity = str(component["canonical_identity"])
        for physical_row in component["physical_rows"]:
            output[int(physical_row)] = identity
    return output


def reduce_scores(
    physical_scores: Sequence[float],
    corrected: Sequence[str],
    *,
    allowed_identities: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Max-reduce physical rows and rank identities by frozen RAW score.

    Exact physical-row ties select the smallest row.  Identity score ties are
    ordered by the corrected identity string.  Filtering happens *before* the
    identity reduction and rank assignment, which is the defining V1.2 repair
    for the fold-local sampler.
    """

    if len(physical_scores) != len(corrected):
        raise RCDEP0V12Error("score/identity cardinality mismatch")
    allowed = None if allowed_identities is None else set(map(str, allowed_identities))
    winners: dict[str, tuple[float, int]] = {}
    for physical_row, (raw_score, identity) in enumerate(zip(physical_scores, corrected, strict=True)):
        identity = str(identity)
        if allowed is not None and identity not in allowed:
            continue
        score = float(raw_score)
        if not math.isfinite(score):
            raise RCDEP0V12Error("RAW score is nonfinite")
        current = winners.get(identity)
        if current is None or score > current[0] or (score == current[0] and physical_row < current[1]):
            winners[identity] = (score, physical_row)
    if allowed is not None and set(winners) != allowed:
        missing = sorted(allowed - set(winners))
        raise RCDEP0V12Error(f"allowed identities are absent from gallery: {missing[:3]}")
    ordered = sorted(winners.items(), key=lambda item: (-item[1][0], item[0]))
    return [
        {
            "fold_local_rank": rank,
            "corrected_identity": identity,
            "physical_row": row,
            "raw_score": score,
        }
        for rank, (identity, (score, row)) in enumerate(ordered, start=1)
    ]


def natural_c128(
    physical_scores: Sequence[float], corrected: Sequence[str]
) -> list[dict[str, Any]]:
    ranked = reduce_scores(physical_scores, corrected)
    if len(ranked) < 128:
        raise RCDEP0V12Error("gallery has fewer than 128 corrected identities")
    output = []
    for raw_rank, item in enumerate(ranked[:128], start=1):
        output.append({**item, "raw_rank": raw_rank})
    return output


def canonical_c128(raw_c128: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove RAW rank/score from the only model-visible C128 payload."""

    if len(raw_c128) != 128:
        raise RCDEP0V12Error("canonical C128 requires exactly 128 identities")
    identities = [str(item["corrected_identity"]) for item in raw_c128]
    if len(set(identities)) != 128:
        raise RCDEP0V12Error("canonical C128 contains a duplicate identity")
    ordered = sorted(raw_c128, key=lambda item: str(item["corrected_identity"]))
    return [
        {
            "candidate_position": position,
            "physical_row": int(item["physical_row"]),
        }
        for position, item in enumerate(ordered)
    ]


def identity_group_map(role_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    groups: dict[str, set[str]] = defaultdict(set)
    for item in role_rows:
        groups[str(item["identity"])].add(str(item["supergroup"]))
    ambiguous = {identity: values for identity, values in groups.items() if len(values) != 1}
    if ambiguous:
        raise RCDEP0V12Error(f"identity maps to multiple supergroups: {sorted(ambiguous)[:3]}")
    return {identity: next(iter(values)) for identity, values in groups.items()}


def fold_access(
    role_rows: Sequence[Mapping[str, Any]],
    corrected: Sequence[str],
    *,
    heldout_fold: int,
) -> dict[str, Any]:
    """Construct the strict inductive reference allowlist for one fold."""

    if heldout_fold not in (1, 2, 3, 4):
        raise RCDEP0V12Error("heldout fold must be one of 1,2,3,4")
    mapping = identity_group_map(role_rows)
    train_rows = [item for item in role_rows if int(item["inner_fold"]) != heldout_fold]
    heldout_rows = [item for item in role_rows if int(item["inner_fold"]) == heldout_fold]
    train_identities = {str(item["identity"]) for item in train_rows}
    heldout_identities = {str(item["identity"]) for item in heldout_rows}
    train_groups = {str(item["supergroup"]) for item in train_rows}
    heldout_groups = {str(item["supergroup"]) for item in heldout_rows}
    if train_identities & heldout_identities:
        raise RCDEP0V12Error("identity leakage across inner fold")
    if train_groups & heldout_groups:
        raise RCDEP0V12Error("supergroup leakage across inner fold")
    allowed_rows = [row for row, identity in enumerate(corrected) if identity in train_identities]
    heldout_reference_rows = [row for row, identity in enumerate(corrected) if identity in heldout_identities]
    if set(allowed_rows) & set(heldout_reference_rows):
        raise RCDEP0V12Error("reference row leakage across inner fold")
    if not train_identities or not heldout_identities:
        raise RCDEP0V12Error("empty train or heldout identity population")
    if any(mapping[identity] not in train_groups for identity in train_identities):
        raise RCDEP0V12Error("train identity/group access mismatch")
    return {
        "heldout_fold": heldout_fold,
        "train_identities": sorted(train_identities),
        "train_supergroups": sorted(train_groups),
        "heldout_identities": sorted(heldout_identities),
        "heldout_supergroups": sorted(heldout_groups),
        "allowed_physical_rows": allowed_rows,
        "heldout_physical_rows": heldout_reference_rows,
    }


def _hash_pick(
    entries: Sequence[Mapping[str, Any]], namespace: bytes, query_sha256: str
) -> Mapping[str, Any] | None:
    if not entries:
        return None
    return min(
        entries,
        key=lambda item: (
            selection_digest(
                namespace,
                query_sha256=query_sha256,
                corrected_identity=str(item["corrected_identity"]),
                physical_row=int(item["physical_row"]),
            ),
            str(item["corrected_identity"]),
        ),
    )


def select_611_negatives(
    fold_local_ranking: Sequence[Mapping[str, Any]],
    *,
    target_identity: str,
    query_sha256: str,
) -> dict[str, Any]:
    """Select six hard, one middle and one broad fold-local negative.

    Rank bands are defined on the fold-local identity-reduced ranking before
    target exclusion: middle is rank 9--32 and broad is rank 33--last.  Missing
    band selections are filled, without duplication, from all remaining legal
    negatives ordered by fold-local rank, then the registered physical-row
    digest, then corrected identity.  Fewer than eight legal negatives uses all
    of them; fewer than two makes the episode training-ineligible.
    """

    identities = [str(item["corrected_identity"]) for item in fold_local_ranking]
    if len(identities) != len(set(identities)):
        raise RCDEP0V12Error("fold-local ranking contains duplicate identities")
    if target_identity not in identities:
        raise RCDEP0V12Error("target is outside the fold-local training allowlist")
    ranks = [int(item["fold_local_rank"]) for item in fold_local_ranking]
    if ranks != list(range(1, len(ranks) + 1)):
        raise RCDEP0V12Error("fold-local ranks are not contiguous")
    legal = [item for item in fold_local_ranking if str(item["corrected_identity"]) != target_identity]
    selected: list[tuple[Mapping[str, Any], str, str | None]] = []
    used: set[str] = set()

    for item in legal[:6]:
        identity = str(item["corrected_identity"])
        selected.append((item, "hard", None))
        used.add(identity)

    middle_pool = [
        item for item in legal
        if 9 <= int(item["fold_local_rank"]) <= 32
        and str(item["corrected_identity"]) not in used
    ]
    middle = _hash_pick(middle_pool, MIDDLE_NAMESPACE, query_sha256)
    if middle is not None:
        identity = str(middle["corrected_identity"])
        selected.append((middle, "middle", selection_digest(
            MIDDLE_NAMESPACE,
            query_sha256=query_sha256,
            corrected_identity=identity,
            physical_row=int(middle["physical_row"]),
        ).hex()))
        used.add(identity)

    broad_pool = [
        item for item in legal
        if int(item["fold_local_rank"]) >= 33
        and str(item["corrected_identity"]) not in used
    ]
    broad = _hash_pick(broad_pool, BROAD_NAMESPACE, query_sha256)
    if broad is not None:
        identity = str(broad["corrected_identity"])
        selected.append((broad, "broad", selection_digest(
            BROAD_NAMESPACE,
            query_sha256=query_sha256,
            corrected_identity=identity,
            physical_row=int(broad["physical_row"]),
        ).hex()))
        used.add(identity)

    requested = min(8, len(legal))
    fallback = sorted(
        (item for item in legal if str(item["corrected_identity"]) not in used),
        key=lambda item: (
            int(item["fold_local_rank"]),
            physical_row_digest(int(item["physical_row"])),
            str(item["corrected_identity"]),
        ),
    )
    for item in fallback:
        if len(selected) >= requested:
            break
        identity = str(item["corrected_identity"])
        selected.append((item, "fallback", physical_row_digest(int(item["physical_row"])).hex()))
        used.add(identity)

    records = [
        {
            "selection_ordinal": ordinal,
            "selection_role": role,
            "corrected_identity": str(item["corrected_identity"]),
            "physical_row": int(item["physical_row"]),
            "fold_local_rank": int(item["fold_local_rank"]),
            "selection_digest": digest,
        }
        for ordinal, (item, role, digest) in enumerate(selected)
    ]
    if len({item["corrected_identity"] for item in records}) != len(records):
        raise RCDEP0V12Error("negative sampler duplicated an identity")
    return {
        "available_negative_count": len(legal),
        "middle_bin_available": any(9 <= int(item["fold_local_rank"]) <= 32 for item in legal),
        "broad_bin_available": any(int(item["fold_local_rank"]) >= 33 for item in legal),
        "selected": records,
        "fallback_count": sum(item["selection_role"] == "fallback" for item in records),
        "training_eligible": len(legal) >= 2,
    }


def episode_model_order(
    *,
    query_sha256: str,
    target_identity: str,
    target_row: int,
    negatives: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a target-blind-looking, hash-fixed order for episode references."""

    entries = [{"corrected_identity": target_identity, "physical_row": target_row}]
    entries.extend(
        {"corrected_identity": str(item["corrected_identity"]), "physical_row": int(item["physical_row"])}
        for item in negatives
    )
    return sorted(
        entries,
        key=lambda item: (
            selection_digest(
                EPISODE_ORDER_NAMESPACE,
                query_sha256=query_sha256,
                corrected_identity=str(item["corrected_identity"]),
                physical_row=int(item["physical_row"]),
            ),
            str(item["corrected_identity"]),
        ),
    )


def legacy_replay_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, int | bool]:
    """Count only available historical ranks in the replay denominator."""

    available = sum(bool(item.get("available")) for item in records)
    matched = sum(bool(item.get("available")) and bool(item.get("matched")) for item in records)
    return {
        "population_count": len(records),
        "available_count": available,
        "unavailable_count": len(records) - available,
        "match_count": matched,
        "mismatch_count": available - matched,
        "all_available_match": available > 0 and available == matched,
    }
