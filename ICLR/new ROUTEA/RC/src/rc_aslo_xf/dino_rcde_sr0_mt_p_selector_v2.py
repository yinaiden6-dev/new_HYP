"""Sign-independent compact P proposal selection for the successor V2 lock.

This module is deliberately additive: it neither imports nor mutates the
frozen V1 lock/fanout implementation.  P is a *proposal ranker*.  Therefore a
legal structural row is selected even when every learned score is negative or
the maximum is exactly zero.  ``P_LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE`` is
reserved for an empty legal-row population.

The root adapter also keeps query geometry separate from candidate-reference
availability.  A query root with no legal reference action is serialized as
``ROOT_REFERENCE_MISSING`` with its query mask intact; an empty/unmappable
query root is ``ROOT_QUERY_UNMAPPABLE``.  There is no semantic no-match state
in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import torch

from .dino_rcde_sr0_mt_p_lock_v2 import (
    LOCK_PROPOSAL_READY,
    LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE,
    PLockV2ContractError,
    RELATIONAL_ACTIVE,
    RELATIONAL_EXACT_ZERO,
    ROOT_QUERY_UNMAPPABLE,
    ROOT_READY,
    ROOT_REFERENCE_MISSING,
    StructuralRootLockV2,
)


SCHEMA_VERSION = "rc_dino_rcde_sr0_mt_p_selector_v2_20260818"
QUERY_MAPPING_UNAVAILABLE = "CANONICAL_QUERY_MAPPING_UNAVAILABLE"
REFERENCE_ACTION_UNAVAILABLE = "NO_LEGAL_REFERENCE_COMPONENT"
REFERENCE_ACTION_READY = "REFERENCE_COMPONENT_LEGAL"

_SHA256 = re.compile(r"[0-9a-f]{64}")


class PSelectorV2ContractError(PLockV2ContractError):
    """A fail-closed successor-selector contract violation."""


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PSelectorV2ContractError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_scalar(value: torch.Tensor, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if (
        result.ndim != 0
        or not result.is_floating_point()
        or not bool(torch.isfinite(result))
    ):
        raise PSelectorV2ContractError(f"{name} must be one finite floating scalar")
    return result


@dataclass(frozen=True)
class CompactRootObservationV2:
    """Candidate-local observation from which a V2 root state is derived.

    The caller does not provide a status.  This prevents a missing reference
    action from erasing an otherwise valid structural query root.
    """

    root_ordinal: int
    root_topology_sha256: str
    query_mask: torch.Tensor
    reference_mask: torch.Tensor
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    reference_action_key_sha256: str | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.root_ordinal, bool)
            or not isinstance(self.root_ordinal, int)
            or self.root_ordinal < 0
        ):
            raise PSelectorV2ContractError("root ordinal is invalid")
        _require_sha256(self.root_topology_sha256, name="root topology")
        if self.reference_action_key_sha256 is not None:
            _require_sha256(
                self.reference_action_key_sha256, name="reference action key"
            )

    def to_lock_root(self) -> StructuralRootLockV2:
        """Derive READY, REFERENCE_MISSING, or QUERY_UNMAPPABLE exactly."""

        query = torch.as_tensor(self.query_mask, dtype=torch.bool).flatten()
        reference = torch.as_tensor(self.reference_mask, dtype=torch.bool).flatten()
        query_present = bool(query.any())
        reference_present = bool(reference.any())
        action_present = self.reference_action_key_sha256 is not None

        if not query_present:
            if reference_present or action_present:
                raise PSelectorV2ContractError(
                    "query-unmappable root cannot expose reference geometry/action"
                )
            status = ROOT_QUERY_UNMAPPABLE
            reason = QUERY_MAPPING_UNAVAILABLE
            evidence_status = RELATIONAL_EXACT_ZERO
        elif not action_present:
            if reference_present:
                raise PSelectorV2ContractError(
                    "reference-missing root cannot expose reference geometry"
                )
            status = ROOT_REFERENCE_MISSING
            reason = REFERENCE_ACTION_UNAVAILABLE
            evidence_status = RELATIONAL_EXACT_ZERO
        else:
            if not reference_present:
                raise PSelectorV2ContractError(
                    "READY reference action requires reference geometry"
                )
            status = ROOT_READY
            reason = REFERENCE_ACTION_READY
            evidence_status = RELATIONAL_ACTIVE

        return StructuralRootLockV2(
            root_ordinal=self.root_ordinal,
            root_topology_sha256=self.root_topology_sha256,
            status=status,
            state_reason=reason,
            action_key_sha256=self.reference_action_key_sha256,
            query_mask=query,
            reference_mask=reference,
            query_grid_shape=self.query_grid_shape,
            reference_grid_shape=self.reference_grid_shape,
            relational_evidence_status=evidence_status,
        )


@dataclass(frozen=True)
class CompactProposalRowV2:
    """One frozen bank row and its learned proposal-ranking scalar."""

    canonical_bank_ordinal: int
    structural_row_sha256: str
    constituent_root_ordinals: tuple[int, ...]
    structurally_legal: bool
    structural_rejection_reason: str | None
    score: torch.Tensor

    def __post_init__(self) -> None:
        if (
            isinstance(self.canonical_bank_ordinal, bool)
            or not isinstance(self.canonical_bank_ordinal, int)
            or self.canonical_bank_ordinal < 0
        ):
            raise PSelectorV2ContractError("canonical bank ordinal is invalid")
        _require_sha256(self.structural_row_sha256, name="structural row")
        roots = tuple(int(item) for item in self.constituent_root_ordinals)
        if (
            not roots
            or any(item < 0 for item in roots)
            or roots != tuple(sorted(roots))
            or len(set(roots)) != len(roots)
        ):
            raise PSelectorV2ContractError(
                "row root membership must be nonempty, unique, and canonical"
            )
        if not isinstance(self.structurally_legal, bool):
            raise PSelectorV2ContractError("row structural legality is not boolean")
        if self.structurally_legal:
            if self.structural_rejection_reason is not None:
                raise PSelectorV2ContractError(
                    "legal row cannot carry a structural rejection reason"
                )
        elif (
            not isinstance(self.structural_rejection_reason, str)
            or not self.structural_rejection_reason
        ):
            raise PSelectorV2ContractError(
                "illegal row requires a structural rejection reason"
            )
        object.__setattr__(self, "constituent_root_ordinals", roots)
        object.__setattr__(
            self, "score", _finite_scalar(self.score, name="proposal row score")
        )


@dataclass(frozen=True)
class CompactPProposalSelectionV2:
    """Result of sign-independent structural proposal selection."""

    lock_state: str
    selected_bank_ordinal: int | None
    selected_row_sha256: str | None
    selected_score: torch.Tensor
    selected_root_ordinals: tuple[int, ...]
    selected_roots: tuple[StructuralRootLockV2, ...]

    def __post_init__(self) -> None:
        score = _finite_scalar(self.selected_score, name="selected proposal score")
        object.__setattr__(self, "selected_score", score)
        if self.lock_state == LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE:
            if (
                self.selected_bank_ordinal is not None
                or self.selected_row_sha256 is not None
                or self.selected_root_ordinals
                or self.selected_roots
                or bool(score.detach().ne(0.0))
            ):
                raise PSelectorV2ContractError(
                    "unavailable proposal must have null membership and exact-zero score"
                )
            return
        if self.lock_state != LOCK_PROPOSAL_READY:
            raise PSelectorV2ContractError("unknown selector lock state")
        if (
            self.selected_bank_ordinal is None
            or self.selected_bank_ordinal < 0
            or self.selected_row_sha256 is None
        ):
            raise PSelectorV2ContractError("READY proposal address is incomplete")
        _require_sha256(self.selected_row_sha256, name="selected structural row")
        root_ordinals = tuple(root.root_ordinal for root in self.selected_roots)
        if root_ordinals != self.selected_root_ordinals:
            raise PSelectorV2ContractError("selected root membership drift")
        if any(root.status == ROOT_QUERY_UNMAPPABLE for root in self.selected_roots):
            raise PSelectorV2ContractError(
                "structurally legal selected row contains an unmappable query root"
            )


def select_compact_p_proposal_v2(
    rows: Sequence[CompactProposalRowV2],
    root_observations: Sequence[CompactRootObservationV2],
) -> CompactPProposalSelectionV2:
    """Select the exact maximum legal row, independent of absolute score sign.

    Input sequence order is irrelevant.  Exact score ties use the frozen
    result-blind key ``(structural_row_sha256, canonical_bank_ordinal)``.
    The selected root ledger contains the row's *complete* membership,
    including every ``ROOT_REFERENCE_MISSING`` root.
    """

    row_values = tuple(rows)
    root_values = tuple(item.to_lock_root() for item in root_observations)
    if not row_values or not root_values:
        raise PSelectorV2ContractError("selector population is empty")
    row_ordinals = tuple(item.canonical_bank_ordinal for item in row_values)
    if len(set(row_ordinals)) != len(row_ordinals):
        raise PSelectorV2ContractError("duplicate canonical bank ordinal")
    root_ordinals = tuple(item.root_ordinal for item in root_values)
    if len(set(root_ordinals)) != len(root_ordinals):
        raise PSelectorV2ContractError("duplicate canonical root ordinal")
    root_by_ordinal = {item.root_ordinal: item for item in root_values}

    for row in row_values:
        unknown = set(row.constituent_root_ordinals) - set(root_by_ordinal)
        if unknown:
            raise PSelectorV2ContractError("row references an unknown root ordinal")
        if row.structurally_legal and any(
            root_by_ordinal[ordinal].status == ROOT_QUERY_UNMAPPABLE
            for ordinal in row.constituent_root_ordinals
        ):
            raise PSelectorV2ContractError(
                "legal row cannot contain a query-unmappable root"
            )

    legal = tuple(item for item in row_values if item.structurally_legal)
    if not legal:
        zero = torch.stack(tuple(item.score for item in row_values)).square().sum() * 0.0
        return CompactPProposalSelectionV2(
            lock_state=LOCK_STRUCTURAL_PROPOSAL_UNAVAILABLE,
            selected_bank_ordinal=None,
            selected_row_sha256=None,
            selected_score=zero,
            selected_root_ordinals=(),
            selected_roots=(),
        )

    maximum = max(float(item.score.detach()) for item in legal)
    tied = tuple(item for item in legal if float(item.score.detach()) == maximum)
    selected = min(
        tied,
        key=lambda item: (
            item.structural_row_sha256,
            item.canonical_bank_ordinal,
        ),
    )
    membership = selected.constituent_root_ordinals
    selected_roots = tuple(root_by_ordinal[ordinal] for ordinal in membership)
    return CompactPProposalSelectionV2(
        lock_state=LOCK_PROPOSAL_READY,
        selected_bank_ordinal=selected.canonical_bank_ordinal,
        selected_row_sha256=selected.structural_row_sha256,
        selected_score=selected.score,
        selected_root_ordinals=membership,
        selected_roots=selected_roots,
    )


__all__ = [
    "SCHEMA_VERSION",
    "QUERY_MAPPING_UNAVAILABLE",
    "REFERENCE_ACTION_UNAVAILABLE",
    "REFERENCE_ACTION_READY",
    "PSelectorV2ContractError",
    "CompactRootObservationV2",
    "CompactProposalRowV2",
    "CompactPProposalSelectionV2",
    "select_compact_p_proposal_v2",
]
