"""Freeze matched-null-visible RGH atoms into connected superregions.

This module is deliberately downstream of
``rgh_reference_generated_atom_v2`` and upstream of every DINO/GX
verifier.  It changes neither the historical RGH atom generator nor any V
model.  Its only content-bearing inputs are pairs of already-frozen
``A_TO_B``/``B_TO_A`` projected atoms and their frozen matched-null contrasts.

An atom is visible iff both directional atoms are structurally legal and both
directional contrasts are strictly positive.  Only visible atoms enter the
joint compatibility graph.  Two atoms share an edge only when their reference
footprints and both directional query footprints four-touch or overlap and
their affine predictions agree under the already-registered two-query-cell-
diagonal scale.  Every graph component is sealed independently; there is no
top-k, winner selection, dilation, gap fill, morphology, or area cap.

The small mathematical helpers at the end implement the fixed two-direction
soft minimum and a normalized MIL marginal with an exact H0 branch and a
uniform atom prior.  They are independent of hard component freezing and keep
their autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import math
from typing import Any, Iterable, Sequence

import torch

from .cw0_connected_region_v2 import enumerate_reference_macro_seeds
from .cw0_connected_window_v1 import apply_affine
from .geometry_hypothesis_v1 import (
    connected_components_4,
    fixed_macro_bank_split,
    grid_cell_centres,
)
from .rgh_reference_generated_atom_v2 import (
    RGHAtomDirectionEvidence,
    RGHProjectedAtomV2,
)


RGH_FROZEN_SELECTED_SUPERREGION_SCHEMA_VERSION = (
    "rc_rgh_frozen_selected_superregion_v1"
)
RGH_FROZEN_SELECTION_STATE = "RGH_FROZEN_SELECTED_SUPERREGION"
RGH_H0_STATE = "RGH_H0"
RGH_DEFAULT_SOFTMIN_TEMPERATURE = 0.10
RGH_DEFAULT_MIL_TEMPERATURE = 0.10
RGH_FIXED_H0_PRIOR = 0.50


def _positive_grid(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] <= 0
        or value[1] <= 0
    ):
        raise ValueError(f"{name} must be a positive (height,width) tuple")
    return int(value[0]), int(value[1])


def _finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _frozen_scalar(value: torch.Tensor | float, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float64).detach().cpu().contiguous()
    if result.numel() != 1 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be one finite scalar")
    return result.reshape(())


def _mask(value: torch.Tensor, count: int, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous()
    if result.shape != (count,):
        raise ValueError(f"{name} shape drift")
    return result


def _provenance(
    value: torch.Tensor,
    query_count: int,
    reference_count: int,
    *,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.bool).detach().cpu().contiguous()
    if result.shape != (query_count, reference_count):
        raise ValueError(f"{name} shape drift")
    return result


def _sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA256")
    return value


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _update_tensor_digest(digest: Any, value: torch.Tensor) -> None:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))


def _update_canonical_digest(digest: Any, value: Any) -> None:
    """Hash every declared receipt field so upstream schema additions cannot alias."""

    if isinstance(value, torch.Tensor):
        digest.update(b"tensor\x00")
        _update_tensor_digest(digest, value)
    elif is_dataclass(value):
        digest.update(type(value).__qualname__.encode("ascii"))
        digest.update(b"\x00")
        for field in fields(value):
            digest.update(field.name.encode("ascii"))
            digest.update(b"\x00")
            _update_canonical_digest(digest, getattr(value, field.name))
    elif isinstance(value, bool):
        digest.update(b"bool\x001\x00" if value else b"bool\x000\x00")
    elif isinstance(value, int):
        digest.update(b"int\x00")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\x00")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evidence digest cannot encode non-finite floats")
        digest.update(b"float\x00")
        digest.update(value.hex().encode("ascii"))
        digest.update(b"\x00")
    elif isinstance(value, str):
        digest.update(b"str\x00")
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence\x00")
        digest.update(str(len(value)).encode("ascii"))
        digest.update(b"\x00")
        for item in value:
            _update_canonical_digest(digest, item)
    elif value is None:
        digest.update(b"none\x00")
    else:
        raise ValueError(f"unsupported evidence receipt field: {type(value).__name__}")


def direction_evidence_sha256(value: RGHAtomDirectionEvidence) -> str:
    """Content digest binding a frozen score to its complete hard receipt."""

    if not isinstance(value, RGHAtomDirectionEvidence):
        raise ValueError("direction evidence digest requires RGHAtomDirectionEvidence")
    digest = hashlib.sha256()
    digest.update(b"RGH_ATOM_DIRECTION_EVIDENCE_BINDING_V1\x00")
    _update_canonical_digest(digest, value)
    return digest.hexdigest()


def four_touch_or_overlap(
    first: torch.Tensor,
    second: torch.Tensor,
    grid_shape: tuple[int, int],
) -> bool:
    """Return exact four-neighbour touch/overlap without changing either mask."""

    shape = _positive_grid(grid_shape, name="touch grid")
    count = math.prod(shape)
    left = _mask(first, count, name="first touch mask").reshape(shape)
    right = _mask(second, count, name="second touch mask").reshape(shape)
    if bool((left & right).any()):
        return True
    horizontal = bool(
        (left[:, :-1] & right[:, 1:]).any()
        or (right[:, :-1] & left[:, 1:]).any()
    )
    vertical = bool(
        (left[:-1, :] & right[1:, :]).any()
        or (right[:-1, :] & left[1:, :]).any()
    )
    return horizontal or vertical


@dataclass(frozen=True)
class RGHSelectionBindingReceipt:
    """Candidate/source/checkpoint namespace shared by every atom in one seal."""

    candidate_key: str
    candidate_index: int
    query_source_sha256: str
    reference_source_sha256: str
    checkpoint_sha256: str
    assignment_implementation_sha256: str
    geometry_sha256: str
    direction_schedule_sha256: str
    real_namespace: str
    matched_null_namespace: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_key", _nonempty(self.candidate_key, name="candidate key")
        )
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("binding candidate index must be non-negative")
        for name in (
            "query_source_sha256",
            "reference_source_sha256",
            "checkpoint_sha256",
            "assignment_implementation_sha256",
            "geometry_sha256",
            "direction_schedule_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        real = _nonempty(self.real_namespace, name="REAL namespace")
        matched = _nonempty(self.matched_null_namespace, name="matched-null namespace")
        if real == matched:
            raise ValueError("REAL and matched-null namespaces must be distinct")
        object.__setattr__(self, "real_namespace", real)
        object.__setattr__(self, "matched_null_namespace", matched)


@dataclass(frozen=True)
class RGHFrozenAtomInput:
    """One strict V2 projected atom and its bound matched-null evidence."""

    a_to_b: RGHProjectedAtomV2
    b_to_a: RGHProjectedAtomV2
    matched_null_a_to_b: RGHAtomDirectionEvidence
    matched_null_b_to_a: RGHAtomDirectionEvidence
    binding: RGHSelectionBindingReceipt

    def __post_init__(self) -> None:
        first, second = self.a_to_b, self.b_to_a
        null_first, null_second = self.matched_null_a_to_b, self.matched_null_b_to_a
        if not isinstance(first, RGHProjectedAtomV2) or not isinstance(
            second, RGHProjectedAtomV2
        ):
            raise ValueError("frozen atom input requires two strict RGH V2 projected atoms")
        if not isinstance(null_first, RGHAtomDirectionEvidence) or not isinstance(
            null_second, RGHAtomDirectionEvidence
        ):
            raise ValueError("frozen atom input requires two matched-null evidence records")
        if not isinstance(self.binding, RGHSelectionBindingReceipt):
            raise ValueError("frozen atom input requires a source/checkpoint binding receipt")
        if self.binding.candidate_index != first.candidate_index:
            raise ValueError("binding receipt/candidate index drift")
        if first.direction != "A_TO_B" or second.direction != "B_TO_A":
            raise ValueError("frozen atom directions must be A_TO_B then B_TO_A")
        if null_first.direction != "A_TO_B" or null_second.direction != "B_TO_A":
            raise ValueError("matched-null directions must be A_TO_B then B_TO_A")
        if (
            first.candidate_index != second.candidate_index
            or first.reference_atom_ordinal != second.reference_atom_ordinal
            or first.query_grid_shape != second.query_grid_shape
            or first.reference_grid_shape != second.reference_grid_shape
            or not torch.equal(first.reference_footprint, second.reference_footprint)
        ):
            raise ValueError("directional RGH V2 records do not describe the same atom")
        canonical_atoms = enumerate_reference_macro_seeds(first.reference_grid_shape)
        if not 0 <= first.reference_atom_ordinal < len(canonical_atoms):
            raise ValueError("reference atom ordinal is outside the canonical axis")
        canonical_footprint = canonical_atoms[
            first.reference_atom_ordinal
        ].window.mask.detach().cpu()
        if not torch.equal(first.reference_footprint, canonical_footprint):
            raise ValueError("reference atom footprint is not its canonical macro atom")
        split = fixed_macro_bank_split(canonical_footprint, first.reference_grid_shape)
        expected_a = torch.nonzero(split.a, as_tuple=False).flatten()
        expected_b = torch.nonzero(split.b, as_tuple=False).flatten()
        if not (
            torch.equal(first.evidence.fit_reference_indices.cpu(), expected_a)
            and torch.equal(first.evidence.verify_reference_indices.cpu(), expected_b)
            and torch.equal(second.evidence.fit_reference_indices.cpu(), expected_b)
            and torch.equal(second.evidence.verify_reference_indices.cpu(), expected_a)
        ):
            raise ValueError("atom evidence endpoints do not match the canonical checkerboard")
        for projected, matched_null in (
            (first, null_first),
            (second, null_second),
        ):
            real = projected.evidence
            if (
                matched_null.candidate_index != real.candidate_index
                or matched_null.reference_atom_ordinal != real.reference_atom_ordinal
                or matched_null.direction != real.direction
                or not torch.equal(
                    matched_null.fit_reference_indices, real.fit_reference_indices
                )
                or not torch.equal(
                    matched_null.verify_reference_indices,
                    real.verify_reference_indices,
                )
            ):
                raise ValueError("REAL/matched-null atom evidence binding drift")
        if not (
            torch.equal(
                first.evidence.fit_reference_indices,
                second.evidence.verify_reference_indices,
            )
            and torch.equal(
                first.evidence.verify_reference_indices,
                second.evidence.fit_reference_indices,
            )
        ):
            raise ValueError("A_TO_B/B_TO_A fit and held-out banks are not complementary")

    @property
    def candidate_index(self) -> int:
        return int(self.a_to_b.candidate_index)

    @property
    def reference_atom_ordinal(self) -> int:
        return int(self.a_to_b.reference_atom_ordinal)

    @staticmethod
    def _structural_score(evidence: RGHAtomDirectionEvidence) -> torch.Tensor:
        score = torch.as_tensor(evidence.score, dtype=torch.float64).detach().cpu()
        return score if evidence.hard_eligible else score.abs() * 0.0

    @property
    def contrast_a_to_b(self) -> torch.Tensor:
        return _frozen_scalar(
            self._structural_score(self.a_to_b.evidence)
            - self._structural_score(self.matched_null_a_to_b),
            name="bound A_TO_B contrast",
        )

    @property
    def contrast_b_to_a(self) -> torch.Tensor:
        return _frozen_scalar(
            self._structural_score(self.b_to_a.evidence)
            - self._structural_score(self.matched_null_b_to_a),
            name="bound B_TO_A contrast",
        )

    @property
    def real_evidence_sha256(self) -> tuple[str, str]:
        return (
            direction_evidence_sha256(self.a_to_b.evidence),
            direction_evidence_sha256(self.b_to_a.evidence),
        )

    @property
    def matched_null_evidence_sha256(self) -> tuple[str, str]:
        return (
            direction_evidence_sha256(self.matched_null_a_to_b),
            direction_evidence_sha256(self.matched_null_b_to_a),
        )

    @property
    def structurally_legal(self) -> bool:
        return bool(
            self.a_to_b.legal
            and self.a_to_b.evidence.hard_eligible
            and self.b_to_a.legal
            and self.b_to_a.evidence.hard_eligible
        )

    @property
    def visible(self) -> bool:
        return bool(
            self.structurally_legal
            and float(self.contrast_a_to_b) > 0.0
            and float(self.contrast_b_to_a) > 0.0
        )


def _affine_disagreement_mse(
    first: RGHProjectedAtomV2,
    second: RGHProjectedAtomV2,
) -> float:
    if (
        first.query_grid_shape != second.query_grid_shape
        or first.reference_grid_shape != second.reference_grid_shape
    ):
        return float("inf")
    reference_union = first.reference_footprint | second.reference_footprint
    rows = torch.nonzero(reference_union, as_tuple=False).flatten()
    if rows.numel() == 0:
        return float("inf")
    coordinates = grid_cell_centres(first.reference_grid_shape, dtype=torch.float64)[rows]
    first_prediction = apply_affine(first.affine_fit.matrix.to(torch.float64), coordinates)
    second_prediction = apply_affine(second.affine_fit.matrix.to(torch.float64), coordinates)
    disagreement = (first_prediction - second_prediction).square().sum(dim=1).mean()
    return float(disagreement) if bool(torch.isfinite(disagreement)) else float("inf")


def affine_predictions_agree(
    first: RGHProjectedAtomV2,
    second: RGHProjectedAtomV2,
) -> bool:
    """Apply the frozen RMS < two query-cell-diagonals compatibility gate."""

    if first.query_grid_shape != second.query_grid_shape:
        return False
    height, width = first.query_grid_shape
    query_cell_diagonal_squared = (1.0 / height) ** 2 + (1.0 / width) ** 2
    threshold_mse = 4.0 * query_cell_diagonal_squared
    return _affine_disagreement_mse(first, second) < threshold_mse


def atoms_share_joint_edge(first: RGHFrozenAtomInput, second: RGHFrozenAtomInput) -> bool:
    """Test reference, both-query, and both-affine compatibility."""

    if (
        not first.visible
        or not second.visible
        or first.candidate_index != second.candidate_index
        or first.binding != second.binding
        or first.a_to_b.query_grid_shape != second.a_to_b.query_grid_shape
        or first.a_to_b.reference_grid_shape != second.a_to_b.reference_grid_shape
    ):
        return False
    query_shape = first.a_to_b.query_grid_shape
    reference_shape = first.a_to_b.reference_grid_shape
    return bool(
        four_touch_or_overlap(
            first.a_to_b.reference_footprint,
            second.a_to_b.reference_footprint,
            reference_shape,
        )
        and four_touch_or_overlap(
            first.a_to_b.query_footprint,
            second.a_to_b.query_footprint,
            query_shape,
        )
        and four_touch_or_overlap(
            first.b_to_a.query_footprint,
            second.b_to_a.query_footprint,
            query_shape,
        )
        and affine_predictions_agree(first.a_to_b, second.a_to_b)
        and affine_predictions_agree(first.b_to_a, second.b_to_a)
    )


@dataclass(frozen=True)
class RGHFrozenSelectedSuperregion:
    """One independently sealed connected component of visible atoms."""

    schema_version: str
    candidate_index: int
    component_ordinal: int
    binding: RGHSelectionBindingReceipt
    reference_atom_ordinals: tuple[int, ...]
    real_evidence_sha256: tuple[tuple[str, str], ...]
    matched_null_evidence_sha256: tuple[tuple[str, str], ...]
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    query_footprint_a_to_b: torch.Tensor
    query_footprint_b_to_a: torch.Tensor
    reference_footprint: torch.Tensor
    provenance_a_to_b: torch.Tensor
    provenance_b_to_a: torch.Tensor
    contrast_a_to_b: torch.Tensor
    contrast_b_to_a: torch.Tensor

    def __post_init__(self) -> None:
        if self.schema_version != RGH_FROZEN_SELECTED_SUPERREGION_SCHEMA_VERSION:
            raise ValueError("selected-superregion schema drift")
        if not isinstance(self.binding, RGHSelectionBindingReceipt):
            raise ValueError("selected superregion lost its source/checkpoint binding")
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
            or isinstance(self.component_ordinal, bool)
            or not isinstance(self.component_ordinal, int)
            or self.component_ordinal < 0
        ):
            raise ValueError("candidate/component ordinals must be non-negative")
        atoms = tuple(int(item) for item in self.reference_atom_ordinals)
        if not atoms or atoms != tuple(sorted(set(atoms))) or atoms[0] < 0:
            raise ValueError("component atom ordinals must be nonempty, sorted and unique")
        real_hashes = tuple(tuple(item) for item in self.real_evidence_sha256)
        null_hashes = tuple(tuple(item) for item in self.matched_null_evidence_sha256)
        if len(real_hashes) != len(atoms) or len(null_hashes) != len(atoms):
            raise ValueError("component evidence-hash axis drift")
        for family, name in (
            (real_hashes, "REAL evidence"),
            (null_hashes, "matched-null evidence"),
        ):
            if any(
                len(pair) != 2
                or any(_sha256(value, name=name) != value for value in pair)
                for pair in family
            ):
                raise ValueError(f"{name} hashes are malformed")
        q_shape = _positive_grid(self.query_grid_shape, name="selected query grid")
        r_shape = _positive_grid(self.reference_grid_shape, name="selected reference grid")
        q_count, r_count = math.prod(q_shape), math.prod(r_shape)
        qa = _mask(self.query_footprint_a_to_b, q_count, name="selected A_TO_B query")
        qb = _mask(self.query_footprint_b_to_a, q_count, name="selected B_TO_A query")
        reference = _mask(self.reference_footprint, r_count, name="selected reference")
        pa = _provenance(self.provenance_a_to_b, q_count, r_count, name="A_TO_B provenance")
        pb = _provenance(self.provenance_b_to_a, q_count, r_count, name="B_TO_A provenance")
        for name, value, shape in (
            ("A_TO_B query", qa, q_shape),
            ("B_TO_A query", qb, q_shape),
            ("reference", reference, r_shape),
        ):
            _, diagnostics = connected_components_4(value, shape)
            if not diagnostics.connected_valid or not diagnostics.has_2d_span:
                raise ValueError(f"selected {name} union must be one connected 2-D component")
        if (
            not torch.equal(pa.any(dim=1), qa)
            or not torch.equal(pb.any(dim=1), qb)
            or bool(pa[:, ~reference].any())
            or bool(pb[:, ~reference].any())
            or bool(pa[~qa].any())
            or bool(pb[~qb].any())
        ):
            raise ValueError("selected-superregion provenance does not close")
        ca = torch.as_tensor(self.contrast_a_to_b, dtype=torch.float64).detach().cpu().contiguous()
        cb = torch.as_tensor(self.contrast_b_to_a, dtype=torch.float64).detach().cpu().contiguous()
        if (
            ca.shape != (len(atoms),)
            or cb.shape != (len(atoms),)
            or not bool(torch.isfinite(ca).all() and torch.isfinite(cb).all())
            or bool(ca.le(0).any() or cb.le(0).any())
        ):
            raise ValueError("selected atoms require finite positive directional contrasts")
        object.__setattr__(self, "reference_atom_ordinals", atoms)
        object.__setattr__(self, "real_evidence_sha256", real_hashes)
        object.__setattr__(self, "matched_null_evidence_sha256", null_hashes)
        object.__setattr__(self, "query_grid_shape", q_shape)
        object.__setattr__(self, "reference_grid_shape", r_shape)
        object.__setattr__(self, "query_footprint_a_to_b", qa)
        object.__setattr__(self, "query_footprint_b_to_a", qb)
        object.__setattr__(self, "reference_footprint", reference)
        object.__setattr__(self, "provenance_a_to_b", pa)
        object.__setattr__(self, "provenance_b_to_a", pb)
        object.__setattr__(self, "contrast_a_to_b", ca)
        object.__setattr__(self, "contrast_b_to_a", cb)


@dataclass(frozen=True)
class RGHFrozenCandidateSelection:
    """All independently sealed components for one candidate, or exact H0."""

    schema_version: str
    candidate_index: int
    binding: RGHSelectionBindingReceipt
    state: str
    query_grid_shape: tuple[int, int]
    reference_grid_shape: tuple[int, int]
    visible_atom_ordinals: tuple[int, ...]
    components: tuple[RGHFrozenSelectedSuperregion, ...]
    h0_query_a_to_b: torch.Tensor
    h0_query_b_to_a: torch.Tensor
    h0_reference: torch.Tensor
    h0_provenance_a_to_b: torch.Tensor
    h0_provenance_b_to_a: torch.Tensor

    def __post_init__(self) -> None:
        if self.schema_version != RGH_FROZEN_SELECTED_SUPERREGION_SCHEMA_VERSION:
            raise ValueError("candidate-selection schema drift")
        if not isinstance(self.binding, RGHSelectionBindingReceipt):
            raise ValueError("candidate selection lost its source/checkpoint binding")
        q_shape = _positive_grid(self.query_grid_shape, name="selection query grid")
        r_shape = _positive_grid(self.reference_grid_shape, name="selection reference grid")
        q_count, r_count = math.prod(q_shape), math.prod(r_shape)
        h0_qa = _mask(self.h0_query_a_to_b, q_count, name="H0 A_TO_B query")
        h0_qb = _mask(self.h0_query_b_to_a, q_count, name="H0 B_TO_A query")
        h0_r = _mask(self.h0_reference, r_count, name="H0 reference")
        h0_pa = _provenance(
            self.h0_provenance_a_to_b,
            q_count,
            r_count,
            name="H0 A_TO_B provenance",
        )
        h0_pb = _provenance(
            self.h0_provenance_b_to_a,
            q_count,
            r_count,
            name="H0 B_TO_A provenance",
        )
        if bool(h0_qa.any() or h0_qb.any() or h0_r.any() or h0_pa.any() or h0_pb.any()):
            raise ValueError("H0 masks and provenance must be exact empty")
        visible = tuple(int(item) for item in self.visible_atom_ordinals)
        if visible != tuple(sorted(set(visible))):
            raise ValueError("visible atom ordinals must be sorted and unique")
        components = tuple(self.components)
        expected_state = RGH_FROZEN_SELECTION_STATE if components else RGH_H0_STATE
        if self.state != expected_state:
            raise ValueError("selection state/component count mismatch")
        component_atoms = tuple(
            sorted(atom for component in components for atom in component.reference_atom_ordinals)
        )
        if component_atoms != visible:
            raise ValueError("visible atoms and component partition disagree")
        if any(
            component.candidate_index != self.candidate_index
            or component.binding != self.binding
            or component.query_grid_shape != q_shape
            or component.reference_grid_shape != r_shape
            for component in components
        ):
            raise ValueError("component escaped candidate/grid selection")
        object.__setattr__(self, "query_grid_shape", q_shape)
        object.__setattr__(self, "reference_grid_shape", r_shape)
        object.__setattr__(self, "visible_atom_ordinals", visible)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "h0_query_a_to_b", h0_qa)
        object.__setattr__(self, "h0_query_b_to_a", h0_qb)
        object.__setattr__(self, "h0_reference", h0_r)
        object.__setattr__(self, "h0_provenance_a_to_b", h0_pa)
        object.__setattr__(self, "h0_provenance_b_to_a", h0_pb)


def _component_indices(atoms: Sequence[RGHFrozenAtomInput]) -> tuple[tuple[int, ...], ...]:
    count = len(atoms)
    adjacency = [set() for _ in range(count)]
    for first in range(count):
        for second in range(first + 1, count):
            if atoms_share_joint_edge(atoms[first], atoms[second]):
                adjacency[first].add(second)
                adjacency[second].add(first)
    remaining = set(range(count))
    components: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining, key=lambda index: atoms[index].reference_atom_ordinal)
        stack = [seed]
        members: set[int] = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(sorted(adjacency[current] - members, reverse=True))
        remaining.difference_update(members)
        components.append(
            tuple(sorted(members, key=lambda index: atoms[index].reference_atom_ordinal))
        )
    return tuple(
        sorted(components, key=lambda item: atoms[item[0]].reference_atom_ordinal)
    )


def _seal_component(
    atoms: Sequence[RGHFrozenAtomInput],
    indices: Sequence[int],
    *,
    component_ordinal: int,
) -> RGHFrozenSelectedSuperregion:
    selected = tuple(atoms[index] for index in indices)
    first = selected[0]
    qa = torch.stack([item.a_to_b.query_footprint for item in selected]).any(dim=0)
    qb = torch.stack([item.b_to_a.query_footprint for item in selected]).any(dim=0)
    reference = torch.stack([item.a_to_b.reference_footprint for item in selected]).any(dim=0)
    pa = torch.stack([item.a_to_b.query_reference_provenance for item in selected]).any(dim=0)
    pb = torch.stack([item.b_to_a.query_reference_provenance for item in selected]).any(dim=0)
    return RGHFrozenSelectedSuperregion(
        schema_version=RGH_FROZEN_SELECTED_SUPERREGION_SCHEMA_VERSION,
        candidate_index=first.candidate_index,
        component_ordinal=component_ordinal,
        binding=first.binding,
        reference_atom_ordinals=tuple(item.reference_atom_ordinal for item in selected),
        real_evidence_sha256=tuple(item.real_evidence_sha256 for item in selected),
        matched_null_evidence_sha256=tuple(
            item.matched_null_evidence_sha256 for item in selected
        ),
        query_grid_shape=first.a_to_b.query_grid_shape,
        reference_grid_shape=first.a_to_b.reference_grid_shape,
        query_footprint_a_to_b=qa,
        query_footprint_b_to_a=qb,
        reference_footprint=reference,
        provenance_a_to_b=pa,
        provenance_b_to_a=pb,
        contrast_a_to_b=torch.stack([item.contrast_a_to_b for item in selected]),
        contrast_b_to_a=torch.stack([item.contrast_b_to_a for item in selected]),
    )


def freeze_candidate_selected_superregions(
    atom_inputs: Sequence[RGHFrozenAtomInput],
) -> RGHFrozenCandidateSelection:
    """Freeze every visible joint component for exactly one candidate."""

    atoms = tuple(atom_inputs)
    if not atoms:
        raise ValueError("candidate selection requires the frozen atom axis")
    if not all(isinstance(item, RGHFrozenAtomInput) for item in atoms):
        raise ValueError("candidate selection received a non-atom input")
    candidate = atoms[0].candidate_index
    binding = atoms[0].binding
    q_shape = atoms[0].a_to_b.query_grid_shape
    r_shape = atoms[0].a_to_b.reference_grid_shape
    if any(
        item.candidate_index != candidate
        or item.binding != binding
        or item.a_to_b.query_grid_shape != q_shape
        or item.a_to_b.reference_grid_shape != r_shape
        for item in atoms
    ):
        raise ValueError("candidate selection input mixes candidates or grids")
    ordered = tuple(sorted(atoms, key=lambda item: item.reference_atom_ordinal))
    ordinals = tuple(item.reference_atom_ordinal for item in ordered)
    expected_ordinals = tuple(range(len(enumerate_reference_macro_seeds(r_shape))))
    if ordinals != expected_ordinals:
        raise ValueError("candidate selection must retain the complete canonical atom axis")
    visible = tuple(item for item in ordered if item.visible)
    components = tuple(
        _seal_component(visible, indices, component_ordinal=ordinal)
        for ordinal, indices in enumerate(_component_indices(visible))
    )
    q_count, r_count = math.prod(q_shape), math.prod(r_shape)
    return RGHFrozenCandidateSelection(
        schema_version=RGH_FROZEN_SELECTED_SUPERREGION_SCHEMA_VERSION,
        candidate_index=candidate,
        binding=binding,
        state=RGH_FROZEN_SELECTION_STATE if components else RGH_H0_STATE,
        query_grid_shape=q_shape,
        reference_grid_shape=r_shape,
        visible_atom_ordinals=tuple(item.reference_atom_ordinal for item in visible),
        components=components,
        h0_query_a_to_b=torch.zeros(q_count, dtype=torch.bool),
        h0_query_b_to_a=torch.zeros(q_count, dtype=torch.bool),
        h0_reference=torch.zeros(r_count, dtype=torch.bool),
        h0_provenance_a_to_b=torch.zeros((q_count, r_count), dtype=torch.bool),
        h0_provenance_b_to_a=torch.zeros((q_count, r_count), dtype=torch.bool),
    )


def freeze_selected_superregion_population(
    atom_inputs: Iterable[RGHFrozenAtomInput],
) -> tuple[RGHFrozenCandidateSelection, ...]:
    """Canonical candidate-order-invariant wrapper over the single-candidate core."""

    groups: dict[int, list[RGHFrozenAtomInput]] = {}
    for item in atom_inputs:
        if not isinstance(item, RGHFrozenAtomInput):
            raise ValueError("population selection received a non-atom input")
        groups.setdefault(item.candidate_index, []).append(item)
    if not groups:
        raise ValueError("population selection requires at least one candidate")
    return tuple(
        freeze_candidate_selected_superregions(groups[candidate])
        for candidate in sorted(groups)
    )


def joint_direction_softmin(
    first: torch.Tensor | float,
    second: torch.Tensor | float,
    *,
    temperature: float = RGH_DEFAULT_SOFTMIN_TEMPERATURE,
) -> torch.Tensor:
    """Smooth minimum of two directions, normalized to map (0,0) exactly to 0."""

    tau = _finite_positive(temperature, name="softmin temperature")
    left = torch.as_tensor(first)
    right = torch.as_tensor(second, dtype=left.dtype, device=left.device)
    if left.numel() != 1 or right.numel() != 1:
        raise ValueError("directional softmin requires two scalar values")
    left, right = left.reshape(()), right.reshape(())
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("directional softmin values must be finite")
    if bool(left.detach().eq(0.0) and right.detach().eq(0.0)):
        return (left.abs() + right.abs()) * 0.0
    values = torch.stack((-left / tau, -right / tau))
    return -tau * (torch.logsumexp(values, dim=0) - math.log(2.0))


def joint_direction_min(
    first: torch.Tensor | float,
    second: torch.Tensor | float,
) -> torch.Tensor:
    """Frozen deployment joint score: the exact weaker directional contrast."""

    left = torch.as_tensor(first)
    right = torch.as_tensor(second, dtype=left.dtype, device=left.device)
    if left.numel() != 1 or right.numel() != 1:
        raise ValueError("directional minimum requires two scalar values")
    left, right = left.reshape(()), right.reshape(())
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("directional minimum values must be finite")
    return torch.minimum(left, right)


def candidate_mil_h0_marginal(
    atom_scores: torch.Tensor,
    *,
    complete_atom_count: int,
    temperature: float = RGH_DEFAULT_MIL_TEMPERATURE,
) -> torch.Tensor:
    """H0(1/2) plus a fixed uniform prior over the complete atom axis."""

    tau = _finite_positive(temperature, name="MIL temperature")
    scores = torch.as_tensor(atom_scores)
    if (
        isinstance(complete_atom_count, bool)
        or not isinstance(complete_atom_count, int)
        or complete_atom_count <= 0
        or scores.ndim != 1
        or scores.numel() != complete_atom_count
        or not bool(torch.isfinite(scores).all())
    ):
        raise ValueError("MIL marginal requires one finite score per frozen atom")
    if bool(scores.detach().eq(0.0).all()):
        return scores.abs().sum() * 0.0
    atom_log_mean = torch.logsumexp(scores / tau, dim=0) - math.log(scores.numel())
    h0_log_prior = math.log(RGH_FIXED_H0_PRIOR)
    h1_log_prior = math.log(1.0 - RGH_FIXED_H0_PRIOR)
    return tau * torch.logaddexp(
        torch.as_tensor(h0_log_prior, dtype=scores.dtype, device=scores.device),
        torch.as_tensor(h1_log_prior, dtype=scores.dtype, device=scores.device)
        + atom_log_mean,
    )


def directional_candidate_mil_h0_marginal(
    a_to_b_scores: torch.Tensor,
    b_to_a_scores: torch.Tensor,
    *,
    complete_atom_count: int,
    mil_temperature: float = RGH_DEFAULT_MIL_TEMPERATURE,
) -> torch.Tensor:
    """Use the frozen exact direction minimum before the candidate H0 marginal."""

    first = torch.as_tensor(a_to_b_scores)
    second = torch.as_tensor(b_to_a_scores, dtype=first.dtype, device=first.device)
    if (
        isinstance(complete_atom_count, bool)
        or not isinstance(complete_atom_count, int)
        or complete_atom_count <= 0
        or first.ndim != 1
        or second.shape != first.shape
        or first.numel() != complete_atom_count
    ):
        raise ValueError("directional MIL requires two equally-sized atom vectors")
    scores = torch.stack(
        [
            joint_direction_min(left, right)
            for left, right in zip(first, second, strict=True)
        ]
    )
    return candidate_mil_h0_marginal(
        scores,
        complete_atom_count=complete_atom_count,
        temperature=mil_temperature,
    )


__all__ = [
    "RGH_DEFAULT_MIL_TEMPERATURE",
    "RGH_DEFAULT_SOFTMIN_TEMPERATURE",
    "RGH_FIXED_H0_PRIOR",
    "RGH_FROZEN_SELECTED_SUPERREGION_SCHEMA_VERSION",
    "RGH_FROZEN_SELECTION_STATE",
    "RGH_H0_STATE",
    "RGHFrozenAtomInput",
    "RGHFrozenCandidateSelection",
    "RGHFrozenSelectedSuperregion",
    "RGHSelectionBindingReceipt",
    "affine_predictions_agree",
    "atoms_share_joint_edge",
    "candidate_mil_h0_marginal",
    "directional_candidate_mil_h0_marginal",
    "direction_evidence_sha256",
    "four_touch_or_overlap",
    "freeze_candidate_selected_superregions",
    "freeze_selected_superregion_population",
    "joint_direction_min",
    "joint_direction_softmin",
]
