"""Target-free deterministic E1-MT proposal/seal plumbing.

This additive module fills exactly one gap between the E0 multi-raster
geometry core and a future learned, fold-local P model.  For each immutable
CW1 query root and one registered P direction it:

1. reads only anonymous ColNomic query/reference patch tokens;
2. evaluates a fixed, result-blind bank of canonical connected reference
   actions on the direction's P checkerboard;
3. selects one candidate-bound reference component by raw LME support, with
   exact ties resolved by the lowest canonical action ordinal;
4. discards every support scalar and seals only the selected component mask,
   hashes, and selection receipt;
5. delegates dual-raster mapping and complete r1--r4 population construction
   to :mod:`dino_rcde_cw1_multitile_superregion_v2`; and
6. emits all three mandatory matched scopes for one deterministic row.

The resolver is deliberately labelled plumbing-only.  It is neither a
trained P head nor identity-disjoint/OOF evidence.  There is no API for a
target, label, D1/rank, result, retrieval action, optimizer, or model update.
No DINO descriptor enters selection; DINO geometry is used only to map the
already sealed ColNomic masks to its distinct raster.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Sequence

import torch

from .cw0_connected_region_v2 import (
    enumerate_query_macro_seeds,
)
from .cw1_sr0_natural_superregion_v2 import enumerate_native_reference_actions_v2
from .dino_rcde_colnomic_superregion_v1 import (
    CanonicalPatchGeometry,
    P_DIRECTIONS,
)
from .dino_rcde_cw1_multitile_superregion_v2 import (
    MANDATORY_ARMS,
    ROOT_MISSING,
    ROOT_READY,
    STATUS_READY,
    CW1MultiTileContractError,
    CW1MultiTilePopulationV2,
    ThreeArmFamilyV2,
    build_three_arm_family,
    seal_cw1_multitile_population,
)
from .geometry_hypothesis_v1 import connected_components_4, fixed_macro_bank_split


SCHEMA_VERSION = "rc_dino_rcde_cw1_multitile_pseal_e1_v1"
RESOLVER_ID = "RAW_COLNOMIC_LME_FIXED_4CC_PLUMBING_V1"
RESOLVER_CLAIM = "DETERMINISTIC_PLUMBING_ONLY_NOT_TRAINED_OR_OOF_P"
SUPPORT_TEMPERATURE = 0.07
SUPPORT_TIE_ULPS = 8
ROW_TIE_ULPS = 16
PSEL_READY = "PSEL_CANDIDATE_COMPONENT_READY"
PSEL_MISSING = "PSEL_NO_LEGAL_CANDIDATE_COMPONENT"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class E1MTProposalContractError(ValueError):
    """The deterministic target-free P-seal contract was violated."""


def _sha(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise E1MTProposalContractError(f"{name} must be a lowercase SHA256")
    return value


def _payload_sha(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _canonical_tokens(
    value: torch.Tensor,
    *,
    patch_count: int,
    valid_patch_mask: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Detach one endpoint, retain device, and compute in frozen float32."""

    source = torch.as_tensor(value)
    if not source.is_floating_point() or source.ndim != 2 or source.shape[0] != patch_count:
        raise E1MTProposalContractError(
            f"{name} must be a floating [patch,dimension] tensor on its ColNomic raster"
        )
    if source.shape[1] <= 0:
        raise E1MTProposalContractError(f"{name} feature dimension must be positive")
    # Natural ColNomic payloads may be stored in a lower precision, but the
    # registered proposal calculation is float32.  Device is deliberately not
    # changed here; materializers may run this resolver on CPU or GPU without
    # a hidden host copy changing the numerical path.
    result = source.detach().to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise E1MTProposalContractError(f"{name} must be finite")
    valid = torch.as_tensor(
        valid_patch_mask, dtype=torch.bool, device=result.device
    ).detach().contiguous()
    if valid.shape != (patch_count,):
        raise E1MTProposalContractError(f"{name} valid-mask/raster mismatch")
    norm = torch.linalg.vector_norm(result, dim=1)
    if bool(norm[valid].le(0.0).any()):
        raise E1MTProposalContractError(f"{name} valid rows must have nonzero norm")
    normalized = torch.zeros_like(result)
    normalized[valid] = result[valid] / norm[valid, None]
    return normalized.contiguous()


def _direction_mask(mask: torch.Tensor, grid_shape: tuple[int, int], direction: str) -> torch.Tensor:
    if direction not in P_DIRECTIONS:
        raise E1MTProposalContractError("proposal direction drift")
    split = fixed_macro_bank_split(mask, grid_shape)
    return (split.a if direction == "a_to_b" else split.b).contiguous()


def _connected_action_or_none(
    mask: torch.Tensor, grid_shape: tuple[int, int]
) -> torch.Tensor | None:
    value = torch.as_tensor(mask, dtype=torch.bool).detach().cpu().contiguous()
    _, diagnostic = connected_components_4(value, grid_shape)
    if (
        diagnostic.component_count != 1
        or not diagnostic.connected_valid
        or not diagnostic.has_2d_span
        or diagnostic.active_count < 4
    ):
        return None
    return value


@dataclass(frozen=True)
class _ReferenceActionV1:
    """Ephemeral action-bank row; never stored in the public P seal."""

    action_ordinal: int
    source_action_ordinal: int
    source_root_ordinal: int
    transform: str
    component_mask: torch.Tensor
    selector_mask: torch.Tensor
    eligible: bool
    reason: str
    action_sha256: str


@lru_cache(maxsize=128)
def _cached_reference_action_bank(
    grid_shape: tuple[int, int],
    direction: str,
    valid_mask_bytes: bytes,
) -> tuple[tuple[_ReferenceActionV1, ...], str]:
    """Build the frozen one-fringe 4CC bank and deduplicate structurally.

    ``enumerate_native_reference_actions_v2`` is the registered result-blind
    I/ROT180 population.  It applies exactly one Chebyshev-distance-one
    (3x3/8-neighbour) fringe and deduplicates after that fringe.  This E1
    bridge additionally intersects the immutable geometry-valid mask and
    deduplicates once more, since padding can collapse otherwise distinct
    structural actions.  No descriptor participates in bank construction.
    """

    valid = torch.tensor(tuple(valid_mask_bytes), dtype=torch.uint8).to(torch.bool)
    if valid.shape != (math.prod(grid_shape),):
        raise E1MTProposalContractError("reference valid-mask/raster mismatch")
    source_actions = enumerate_native_reference_actions_v2(grid_shape)
    actions: list[_ReferenceActionV1] = []
    seen: dict[str, torch.Tensor] = {}
    for source in source_actions:
        raw = source.mask & valid
        component = _connected_action_or_none(raw, grid_shape)
        if component is None:
            continue
        component_sha = _tensor_sha(component)
        if component_sha in seen:
            if not torch.equal(seen[component_sha], component):
                raise RuntimeError("reference action hash collision")
            continue
        seen[component_sha] = component
        ordinal = len(actions)
        # The query endpoint is checkerboard split.  Per the frozen
        # query-heldout contract, both directions read the complete selected
        # candidate-reference action.
        selector = component
        eligible, reason = True, "READY"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "action_ordinal": ordinal,
            "source_action_ordinal": source.ordinal,
            "source_root_ordinal": source.source_root_ordinal,
            "transform": source.transform,
            "component_mask_sha256": _tensor_sha(component),
            "selector_mask_sha256": _tensor_sha(selector),
            "eligible": eligible,
            "reason": reason,
        }
        actions.append(
            _ReferenceActionV1(
                action_ordinal=ordinal,
                source_action_ordinal=source.ordinal,
                source_root_ordinal=source.source_root_ordinal,
                transform=source.transform,
                component_mask=component,
                selector_mask=selector,
                eligible=eligible,
                reason=reason,
                action_sha256=_payload_sha(payload),
            )
        )
    bank_sha = _payload_sha(
        {
            "schema_version": SCHEMA_VERSION,
            "grid_shape": list(grid_shape),
            "direction": direction,
            "action_sha256": [item.action_sha256 for item in actions],
        }
    )
    return tuple(actions), bank_sha


def _reference_action_bank(
    *,
    grid_shape: tuple[int, int],
    valid_patch_mask: torch.Tensor,
    direction: str,
) -> tuple[tuple[_ReferenceActionV1, ...], str]:
    valid = (
        torch.as_tensor(valid_patch_mask, dtype=torch.bool)
        .detach()
        .cpu()
        .contiguous()
    )
    return _cached_reference_action_bank(
        grid_shape,
        direction,
        valid.to(torch.uint8).numpy().tobytes(),
    )


def _clear_e1_structural_caches_for_tests() -> None:
    _cached_reference_action_bank.cache_clear()


def _raw_lme(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    reference_mask: torch.Tensor,
) -> float:
    if query_tokens.device != reference_tokens.device:
        raise E1MTProposalContractError("ColNomic query/reference devices differ")
    q = query_tokens[query_mask.to(device=query_tokens.device)]
    r = reference_tokens[reference_mask.to(device=reference_tokens.device)]
    if min(q.shape[0], r.shape[0]) <= 0:
        raise E1MTProposalContractError("raw LME requires nonempty direction banks")
    pair = q @ r.T
    per_query_patch = SUPPORT_TEMPERATURE * (
        torch.logsumexp(pair / SUPPORT_TEMPERATURE, dim=1)
        - math.log(pair.shape[1])
    )
    statistic = per_query_patch.mean()
    if not bool(torch.isfinite(statistic)):
        raise E1MTProposalContractError("raw LME produced a nonfinite value")
    return float(statistic)


def _comparison_tolerance(left: float, right: float, *, ulps: int) -> float:
    """Frozen bounded float32 comparison tolerance for deterministic ties."""

    if isinstance(ulps, bool) or not isinstance(ulps, int) or ulps <= 0:
        raise E1MTProposalContractError("comparison ULP budget must be positive")
    return (
        float(ulps)
        * float(torch.finfo(torch.float32).eps)
        * max(1.0, abs(float(left)), abs(float(right)))
    )


def _selection_payload(
    *,
    candidate_key: str,
    direction: str,
    root_ordinal: int,
    status: str,
    reason: str,
    selected_action_ordinal: int | None,
    selected_component_mask_sha256: str,
    query_root_mask_sha256: str,
    selector_query_mask_sha256: str,
    reference_action_bank_sha256: str,
    query_tokens_sha256: str,
    reference_tokens_sha256: str,
    geometry_sha256: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "resolver_id": RESOLVER_ID,
        "resolver_claim": RESOLVER_CLAIM,
        "candidate_key": candidate_key,
        "direction": direction,
        "root_ordinal": root_ordinal,
        "status": status,
        "reason": reason,
        "selected_action_ordinal": selected_action_ordinal,
        "selected_component_mask_sha256": selected_component_mask_sha256,
        "query_root_mask_sha256": query_root_mask_sha256,
        "selector_query_mask_sha256": selector_query_mask_sha256,
        "reference_action_bank_sha256": reference_action_bank_sha256,
        "query_tokens_sha256": query_tokens_sha256,
        "reference_tokens_sha256": reference_tokens_sha256,
        "geometry_sha256": list(geometry_sha256),
        "trained_p": False,
        "oof_p": False,
        "target_free": True,
        "model_update_count": 0,
    }


@dataclass(frozen=True)
class E1MTRootSelectionV1:
    """Persisted result for one fixed query root; contains no support scalar."""

    candidate_key: str
    direction: str
    root_ordinal: int
    reference_grid_shape: tuple[int, int]
    status: str
    reason: str
    selected_action_ordinal: int | None
    selected_component_mask: torch.Tensor
    selected_component_mask_sha256: str
    query_root_mask_sha256: str
    selector_query_mask_sha256: str
    reference_action_bank_sha256: str
    query_tokens_sha256: str
    reference_tokens_sha256: str
    geometry_sha256: tuple[str, str, str, str]
    selection_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_key, str) or not self.candidate_key:
            raise E1MTProposalContractError("candidate key must be nonempty")
        if self.direction not in P_DIRECTIONS:
            raise E1MTProposalContractError("proposal direction drift")
        if isinstance(self.root_ordinal, bool) or not isinstance(self.root_ordinal, int) or self.root_ordinal < 0:
            raise E1MTProposalContractError("root ordinal must be nonnegative")
        shape = tuple(int(item) for item in self.reference_grid_shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise E1MTProposalContractError("reference grid shape drift")
        mask = torch.as_tensor(
            self.selected_component_mask, dtype=torch.bool
        ).detach().cpu().contiguous()
        if mask.shape != (math.prod(shape),):
            raise E1MTProposalContractError("selected component mask/grid mismatch")
        if self.selected_component_mask_sha256 != _tensor_sha(mask):
            raise E1MTProposalContractError("selected component mask hash drift")
        if self.status == PSEL_READY:
            if (
                isinstance(self.selected_action_ordinal, bool)
                or not isinstance(self.selected_action_ordinal, int)
                or self.selected_action_ordinal < 0
                or self.reason != "READY"
                or _connected_action_or_none(mask, shape) is None
            ):
                raise E1MTProposalContractError("READY selection must seal one canonical 4CC action")
        elif self.status == PSEL_MISSING:
            if self.selected_action_ordinal is not None or bool(mask.any()) or not self.reason:
                raise E1MTProposalContractError("missing selection must be explicit zero/H0")
        else:
            raise E1MTProposalContractError("unknown root-selection status")
        hashes = (
            self.selected_component_mask_sha256,
            self.query_root_mask_sha256,
            self.selector_query_mask_sha256,
            self.reference_action_bank_sha256,
            self.query_tokens_sha256,
            self.reference_tokens_sha256,
            *self.geometry_sha256,
        )
        for index, value in enumerate(hashes):
            _sha(value, name=f"selection hash {index}")
        payload = _selection_payload(
            candidate_key=self.candidate_key,
            direction=self.direction,
            root_ordinal=self.root_ordinal,
            status=self.status,
            reason=self.reason,
            selected_action_ordinal=self.selected_action_ordinal,
            selected_component_mask_sha256=self.selected_component_mask_sha256,
            query_root_mask_sha256=self.query_root_mask_sha256,
            selector_query_mask_sha256=self.selector_query_mask_sha256,
            reference_action_bank_sha256=self.reference_action_bank_sha256,
            query_tokens_sha256=self.query_tokens_sha256,
            reference_tokens_sha256=self.reference_tokens_sha256,
            geometry_sha256=self.geometry_sha256,
        )
        if self.selection_receipt_sha256 != _payload_sha(payload):
            raise E1MTProposalContractError("root selection receipt drift")
        object.__setattr__(self, "reference_grid_shape", shape)
        object.__setattr__(self, "selected_component_mask", mask)


@dataclass(frozen=True)
class E1MTProposalSealV1:
    """Complete deterministic P plumbing output, population, and matched arms."""

    candidate_key: str
    direction: str
    resolver_id: str
    resolver_claim: str
    trained_p: bool
    oof_p: bool
    target_free: bool
    model_update_count: int
    root_selections: tuple[E1MTRootSelectionV1, ...]
    population: CW1MultiTilePopulationV2
    selected_bank_ordinal: int
    bank_selection_receipt_sha256: str
    three_arm_family: ThreeArmFamilyV2
    seal_sha256: str

    def __post_init__(self) -> None:
        if (
            self.resolver_id != RESOLVER_ID
            or self.resolver_claim != RESOLVER_CLAIM
            or self.trained_p
            or self.oof_p
            or not self.target_free
            or self.model_update_count != 0
        ):
            raise E1MTProposalContractError("plumbing resolver claim drift")
        selections = tuple(self.root_selections)
        population = self.population
        if (
            population.candidate_key != self.candidate_key
            or population.direction != self.direction
            or tuple(item.root_ordinal for item in selections) != tuple(range(len(selections)))
            or len(selections) != len(population.root_bindings)
        ):
            raise E1MTProposalContractError("selection/population root ledger drift")
        for selection, binding in zip(selections, population.root_bindings, strict=True):
            if (
                selection.candidate_key != self.candidate_key
                or selection.direction != self.direction
            ):
                raise E1MTProposalContractError("selected component candidate/direction drift")
            if selection.status == PSEL_MISSING:
                if binding.status != ROOT_MISSING or bool(
                    binding.colnomic_reference_component_mask.any()
                ):
                    raise E1MTProposalContractError("missing P selection became a mapped component")
            elif binding.status == ROOT_READY:
                if not torch.equal(
                    selection.selected_component_mask,
                    binding.colnomic_reference_component_mask,
                ):
                    raise E1MTProposalContractError(
                        "selected component changed before dual-raster seal"
                    )
            elif binding.status == ROOT_MISSING:
                # P selection may be lawful on the ColNomic raster while the
                # canonical cross-backbone map correctly yields H0.  The
                # selected ColNomic component remains available only in this
                # immutable P receipt; it cannot inject evidence downstream.
                if not (
                    binding.reason.startswith("QUERY_MAPPING_INELIGIBLE:")
                    or binding.reason.startswith("REFERENCE_MAPPING_INELIGIBLE:")
                ):
                    raise E1MTProposalContractError("mapped P component disappeared without reason")
            else:
                raise E1MTProposalContractError("unknown mapped root status")
        if not 0 <= self.selected_bank_ordinal < len(population.rows):
            raise E1MTProposalContractError("selected bank ordinal is outside r1--r4")
        family = self.three_arm_family
        if (
            family.population_sha256 != population.population_sha256
            or family.bank_ordinal != self.selected_bank_ordinal
            or tuple(item.name for item in family.arms) != MANDATORY_ARMS
        ):
            raise E1MTProposalContractError("three-arm family drifted from selected population row")
        bank_payload = {
            "schema_version": SCHEMA_VERSION,
            "resolver_id": RESOLVER_ID,
            "resolver_claim": RESOLVER_CLAIM,
            "population_sha256": population.population_sha256,
            "root_selection_receipt_sha256": [
                item.selection_receipt_sha256 for item in selections
            ],
            "selected_bank_ordinal": self.selected_bank_ordinal,
            "trained_p": False,
            "oof_p": False,
            "target_free": True,
            "model_update_count": 0,
        }
        if self.bank_selection_receipt_sha256 != _payload_sha(bank_payload):
            raise E1MTProposalContractError("bank selection receipt drift")
        seal_payload = {
            **bank_payload,
            "bank_selection_receipt_sha256": self.bank_selection_receipt_sha256,
            "three_arm_family_sha256": family.family_sha256,
        }
        if self.seal_sha256 != _payload_sha(seal_payload):
            raise E1MTProposalContractError("E1-MT P seal hash drift")
        object.__setattr__(self, "root_selections", selections)


def _make_root_selection(
    *,
    candidate_key: str,
    direction: str,
    root_ordinal: int,
    query_root_mask: torch.Tensor,
    selector_query_mask: torch.Tensor,
    actions: Sequence[_ReferenceActionV1],
    action_bank_sha256: str,
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    reference_tokens_sha256: str,
    reference_grid_shape: tuple[int, int],
    geometry_sha256: tuple[str, str, str, str],
) -> tuple[E1MTRootSelectionV1, float | None]:
    selected: _ReferenceActionV1 | None = None
    selected_value: float | None = None
    evaluated: list[tuple[_ReferenceActionV1, float]] = []
    if bool(selector_query_mask.any()):
        for action in actions:
            if not action.eligible:
                continue
            value = _raw_lme(
                query_tokens,
                reference_tokens,
                selector_query_mask,
                action.selector_mask,
            )
            evaluated.append((action, value))
    if evaluated:
        # Compare every action with the single global maximum.  This prevents
        # tolerance chaining from creating an unbounded effective tie band.
        maximum = max(value for _, value in evaluated)
        tied = tuple(
            (action, value)
            for action, value in evaluated
            if maximum - value
            <= _comparison_tolerance(maximum, value, ulps=SUPPORT_TIE_ULPS)
        )
        selected, selected_value = min(tied, key=lambda item: item[0].action_ordinal)
    return _seal_resolved_root_selection(
        candidate_key=candidate_key,
        direction=direction,
        root_ordinal=root_ordinal,
        query_root_mask=query_root_mask,
        selector_query_mask=selector_query_mask,
        selected=selected,
        selected_value=selected_value,
        action_bank_sha256=action_bank_sha256,
        query_tokens=query_tokens,
        reference_tokens_sha256=reference_tokens_sha256,
        reference_grid_shape=reference_grid_shape,
        geometry_sha256=geometry_sha256,
    )


def _seal_resolved_root_selection(
    *,
    candidate_key: str,
    direction: str,
    root_ordinal: int,
    query_root_mask: torch.Tensor,
    selector_query_mask: torch.Tensor,
    selected: _ReferenceActionV1 | None,
    selected_value: float | None,
    action_bank_sha256: str,
    query_tokens: torch.Tensor,
    reference_tokens_sha256: str,
    reference_grid_shape: tuple[int, int],
    geometry_sha256: tuple[str, str, str, str],
) -> tuple[E1MTRootSelectionV1, float | None]:
    """Persist a resolved root without ever persisting its scalar value."""

    if (selected is None) != (selected_value is None):
        raise E1MTProposalContractError("resolved action/value presence drift")
    if selected is None:
        status, reason, ordinal = (
            PSEL_MISSING,
            "EMPTY_QUERY_DIRECTION_BANK" if not bool(selector_query_mask.any()) else "NO_LEGAL_REFERENCE_ACTION",
            None,
        )
        component = torch.zeros(math.prod(reference_grid_shape), dtype=torch.bool)
    else:
        status, reason, ordinal = PSEL_READY, "READY", selected.action_ordinal
        component = selected.component_mask.clone()
    component_sha = _tensor_sha(component)
    # Bind exactly the P read-set.  A change confined to the complementary V
    # checkerboard must not change this P receipt.
    query_tokens_sha256 = _tensor_sha(
        query_tokens[selector_query_mask.to(device=query_tokens.device)]
    )
    payload = _selection_payload(
        candidate_key=candidate_key,
        direction=direction,
        root_ordinal=root_ordinal,
        status=status,
        reason=reason,
        selected_action_ordinal=ordinal,
        selected_component_mask_sha256=component_sha,
        query_root_mask_sha256=_tensor_sha(query_root_mask),
        selector_query_mask_sha256=_tensor_sha(selector_query_mask),
        reference_action_bank_sha256=action_bank_sha256,
        query_tokens_sha256=query_tokens_sha256,
        reference_tokens_sha256=reference_tokens_sha256,
        geometry_sha256=geometry_sha256,
    )
    return (
        E1MTRootSelectionV1(
            candidate_key=candidate_key,
            direction=direction,
            root_ordinal=root_ordinal,
            reference_grid_shape=reference_grid_shape,
            status=status,
            reason=reason,
            selected_action_ordinal=ordinal,
            selected_component_mask=component,
            selected_component_mask_sha256=component_sha,
            query_root_mask_sha256=_tensor_sha(query_root_mask),
            selector_query_mask_sha256=_tensor_sha(selector_query_mask),
            reference_action_bank_sha256=action_bank_sha256,
            query_tokens_sha256=query_tokens_sha256,
            reference_tokens_sha256=reference_tokens_sha256,
            geometry_sha256=geometry_sha256,
            selection_receipt_sha256=_payload_sha(payload),
        ),
        selected_value,
    )


def _batched_root_resolutions(
    *,
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    selector_query_masks: Sequence[torch.Tensor],
    actions: Sequence[_ReferenceActionV1],
) -> tuple[tuple[_ReferenceActionV1 | None, float | None], ...]:
    """Resolve all roots from one Q@R.T and one padded action reduction.

    The largest temporary is ``[query_patch, action, max_action_members]``;
    it avoids both a dense ``[Q,A,R]`` cube and repeated descriptor matmuls.
    """

    selectors = tuple(
        torch.as_tensor(item, dtype=torch.bool).detach().cpu().contiguous()
        for item in selector_query_masks
    )
    if any(item.shape != (query_tokens.shape[0],) for item in selectors):
        raise E1MTProposalContractError("batched root selector/raster mismatch")
    if not actions:
        return tuple((None, None) for _ in selectors)
    if any(not item.eligible for item in actions):
        raise E1MTProposalContractError("batched action bank contains an ineligible row")
    members = tuple(
        torch.nonzero(item.selector_mask, as_tuple=False).flatten() for item in actions
    )
    if any(item.numel() == 0 for item in members):
        raise E1MTProposalContractError("batched action bank contains an empty row")
    maximum_members = max(int(item.numel()) for item in members)
    action_count = len(actions)
    padded = torch.zeros((action_count, maximum_members), dtype=torch.long)
    membership = torch.zeros((action_count, maximum_members), dtype=torch.bool)
    counts = torch.empty(action_count, dtype=torch.long)
    for ordinal, indices in enumerate(members):
        count = int(indices.numel())
        padded[ordinal, :count] = indices
        membership[ordinal, :count] = True
        counts[ordinal] = count

    device = query_tokens.device
    pair = query_tokens @ reference_tokens.T
    gathered = pair.index_select(1, padded.to(device).reshape(-1)).reshape(
        query_tokens.shape[0], action_count, maximum_members
    )
    valid = membership.to(device)[None, :, :]
    gathered = gathered.masked_fill(~valid, -torch.inf)
    per_query_action = SUPPORT_TEMPERATURE * (
        torch.logsumexp(gathered / SUPPORT_TEMPERATURE, dim=2)
        - counts.to(device=device, dtype=pair.dtype).log()[None, :]
    )
    selector_matrix = torch.stack(selectors).to(device=device, dtype=pair.dtype)
    selector_counts = selector_matrix.sum(dim=1)
    nonempty = selector_counts.gt(0.0)
    weights = selector_matrix / selector_counts.clamp_min(1.0)[:, None]
    root_action = weights @ per_query_action

    maximum = root_action.max(dim=1).values
    scale = torch.maximum(
        torch.ones_like(root_action),
        torch.maximum(root_action.abs(), maximum.abs()[:, None]),
    )
    tolerance = (
        float(SUPPORT_TIE_ULPS)
        * float(torch.finfo(torch.float32).eps)
        * scale
    )
    tied = maximum[:, None] - root_action <= tolerance
    ordinals = torch.arange(action_count, device=device, dtype=torch.long)[None, :]
    sentinel = torch.full_like(ordinals.expand_as(tied), action_count)
    selected_ordinals = torch.where(tied, ordinals.expand_as(tied), sentinel).min(dim=1).values
    selected_values = root_action.gather(1, selected_ordinals[:, None]).squeeze(1)
    ordinal_list = selected_ordinals.detach().cpu().tolist()
    value_list = selected_values.detach().cpu().tolist()
    nonempty_list = nonempty.detach().cpu().tolist()
    return tuple(
        (actions[int(ordinal)], float(value)) if present else (None, None)
        for ordinal, value, present in zip(
            ordinal_list, value_list, nonempty_list, strict=True
        )
    )


def _select_population_row(
    population: CW1MultiTilePopulationV2,
    root_values: Sequence[float | None],
) -> int:
    ready_rows = tuple(row for row in population.rows if row.status == STATUS_READY)
    if not ready_rows:
        return 0
    evaluated: list[tuple[float, tuple[int, str, int], int]] = []
    for row in ready_rows:
        value = sum(
            float(row.structural_region.aggregation_weights[root])
            * float(root_values[root])
            for root in row.structural_region.contributing_root_ordinals
            if root_values[root] is not None
        )
        tie_key = (
            int(row.structural_region.radius),
            row.row_sha256,
            row.bank_ordinal,
        )
        evaluated.append((value, tie_key, row.bank_ordinal))
    maximum = max(value for value, _, _ in evaluated)
    tied = tuple(
        item
        for item in evaluated
        if maximum - item[0]
        <= _comparison_tolerance(maximum, item[0], ulps=ROW_TIE_ULPS)
    )
    return min(tied, key=lambda item: item[1])[2]


def _seal_deterministic_e1_mt_proposal_impl(
    *,
    batched: bool,
    candidate_key: str,
    direction: str,
    colnomic_query_tokens: torch.Tensor,
    colnomic_reference_tokens: torch.Tensor,
    colnomic_query_geometry: CanonicalPatchGeometry,
    colnomic_reference_geometry: CanonicalPatchGeometry,
    dino_query_geometry: CanonicalPatchGeometry,
    dino_reference_geometry: CanonicalPatchGeometry,
    model_checkpoint_sha256: str,
    pair_comparator_sha256: str,
    reducer_sha256: str,
) -> E1MTProposalSealV1:
    if not isinstance(batched, bool):
        raise E1MTProposalContractError("internal resolver backend flag must be boolean")

    if not isinstance(candidate_key, str) or not candidate_key:
        raise E1MTProposalContractError("candidate key must be nonempty")
    if direction not in P_DIRECTIONS:
        raise E1MTProposalContractError("proposal direction drift")
    for name, value in (
        ("model checkpoint", model_checkpoint_sha256),
        ("pair comparator", pair_comparator_sha256),
        ("reducer", reducer_sha256),
    ):
        _sha(value, name=name)
    q_count = math.prod(colnomic_query_geometry.grid_shape)
    r_count = math.prod(colnomic_reference_geometry.grid_shape)
    query_tokens = _canonical_tokens(
        colnomic_query_tokens,
        patch_count=q_count,
        valid_patch_mask=colnomic_query_geometry.valid_patch_mask,
        name="ColNomic query tokens",
    )
    reference_tokens = _canonical_tokens(
        colnomic_reference_tokens,
        patch_count=r_count,
        valid_patch_mask=colnomic_reference_geometry.valid_patch_mask,
        name="ColNomic reference tokens",
    )
    if query_tokens.shape[1] != reference_tokens.shape[1]:
        raise E1MTProposalContractError("ColNomic query/reference dimensions differ")
    if query_tokens.device != reference_tokens.device:
        raise E1MTProposalContractError("ColNomic query/reference devices differ")
    reference_sha = _tensor_sha(reference_tokens)
    geometry_sha = (
        colnomic_query_geometry.sha256,
        colnomic_reference_geometry.sha256,
        dino_query_geometry.sha256,
        dino_reference_geometry.sha256,
    )
    actions, action_bank_sha = _reference_action_bank(
        grid_shape=colnomic_reference_geometry.grid_shape,
        valid_patch_mask=colnomic_reference_geometry.valid_patch_mask,
        direction=direction,
    )
    roots = enumerate_query_macro_seeds(colnomic_query_geometry.grid_shape)
    root_masks: list[torch.Tensor] = []
    selector_masks: list[torch.Tensor] = []
    q_valid = colnomic_query_geometry.valid_patch_mask
    for root in roots:
        root_mask = root.window.mask & q_valid
        root_masks.append(root.window.mask)
        selector_masks.append(
            _direction_mask(root_mask, colnomic_query_geometry.grid_shape, direction)
        )
    selections: list[E1MTRootSelectionV1] = []
    root_values: list[float | None] = []
    resolved = (
        _batched_root_resolutions(
            query_tokens=query_tokens,
            reference_tokens=reference_tokens,
            selector_query_masks=selector_masks,
            actions=actions,
        )
        if batched
        else tuple((None, None) for _ in roots)
    )
    for root_ordinal, (root_mask, selector_mask) in enumerate(
        zip(root_masks, selector_masks, strict=True)
    ):
        if batched:
            selected, ephemeral_value = resolved[root_ordinal]
            selection, ephemeral_value = _seal_resolved_root_selection(
                candidate_key=candidate_key,
                direction=direction,
                root_ordinal=root_ordinal,
                query_root_mask=root_mask,
                selector_query_mask=selector_mask,
                selected=selected,
                selected_value=ephemeral_value,
                action_bank_sha256=action_bank_sha,
                query_tokens=query_tokens,
                reference_tokens_sha256=reference_sha,
                reference_grid_shape=colnomic_reference_geometry.grid_shape,
                geometry_sha256=geometry_sha,
            )
        else:
            selection, ephemeral_value = _make_root_selection(
                candidate_key=candidate_key,
                direction=direction,
                root_ordinal=root_ordinal,
                query_root_mask=root_mask,
                selector_query_mask=selector_mask,
                actions=actions,
                action_bank_sha256=action_bank_sha,
                query_tokens=query_tokens,
                reference_tokens=reference_tokens,
                reference_tokens_sha256=reference_sha,
                reference_grid_shape=colnomic_reference_geometry.grid_shape,
                geometry_sha256=geometry_sha,
            )
        selections.append(selection)
        root_values.append(ephemeral_value)
    component_ledger = {
        item.root_ordinal: (
            item.selected_component_mask if item.status == PSEL_READY else None
        )
        for item in selections
    }
    try:
        population = seal_cw1_multitile_population(
            candidate_key=candidate_key,
            direction=direction,
            reference_component_by_root=component_ledger,
            colnomic_query_geometry=colnomic_query_geometry,
            colnomic_reference_geometry=colnomic_reference_geometry,
            dino_query_geometry=dino_query_geometry,
            dino_reference_geometry=dino_reference_geometry,
        )
    except CW1MultiTileContractError as exc:
        raise E1MTProposalContractError(f"dual-raster population seal failed: {exc}") from exc
    mapped_root_values = [
        value if binding.status == ROOT_READY else None
        for value, binding in zip(root_values, population.root_bindings, strict=True)
    ]
    selected_bank_ordinal = _select_population_row(population, mapped_root_values)
    family = build_three_arm_family(
        population,
        bank_ordinal=selected_bank_ordinal,
        dino_query_valid_patch_mask=dino_query_geometry.valid_patch_mask,
        dino_reference_valid_patch_mask=dino_reference_geometry.valid_patch_mask,
        model_checkpoint_sha256=model_checkpoint_sha256,
        pair_comparator_sha256=pair_comparator_sha256,
        reducer_sha256=reducer_sha256,
    )
    bank_payload = {
        "schema_version": SCHEMA_VERSION,
        "resolver_id": RESOLVER_ID,
        "resolver_claim": RESOLVER_CLAIM,
        "population_sha256": population.population_sha256,
        "root_selection_receipt_sha256": [
            item.selection_receipt_sha256 for item in selections
        ],
        "selected_bank_ordinal": selected_bank_ordinal,
        "trained_p": False,
        "oof_p": False,
        "target_free": True,
        "model_update_count": 0,
    }
    bank_receipt = _payload_sha(bank_payload)
    seal_payload = {
        **bank_payload,
        "bank_selection_receipt_sha256": bank_receipt,
        "three_arm_family_sha256": family.family_sha256,
    }
    return E1MTProposalSealV1(
        candidate_key=candidate_key,
        direction=direction,
        resolver_id=RESOLVER_ID,
        resolver_claim=RESOLVER_CLAIM,
        trained_p=False,
        oof_p=False,
        target_free=True,
        model_update_count=0,
        root_selections=tuple(selections),
        population=population,
        selected_bank_ordinal=selected_bank_ordinal,
        bank_selection_receipt_sha256=bank_receipt,
        three_arm_family=family,
        seal_sha256=_payload_sha(seal_payload),
    )


def seal_deterministic_e1_mt_proposal(
    *,
    candidate_key: str,
    direction: str,
    colnomic_query_tokens: torch.Tensor,
    colnomic_reference_tokens: torch.Tensor,
    colnomic_query_geometry: CanonicalPatchGeometry,
    colnomic_reference_geometry: CanonicalPatchGeometry,
    dino_query_geometry: CanonicalPatchGeometry,
    dino_reference_geometry: CanonicalPatchGeometry,
    model_checkpoint_sha256: str,
    pair_comparator_sha256: str,
    reducer_sha256: str,
) -> E1MTProposalSealV1:
    """Resolve one E1-MT proposal with the vectorized production backend."""

    return _seal_deterministic_e1_mt_proposal_impl(
        batched=True,
        candidate_key=candidate_key,
        direction=direction,
        colnomic_query_tokens=colnomic_query_tokens,
        colnomic_reference_tokens=colnomic_reference_tokens,
        colnomic_query_geometry=colnomic_query_geometry,
        colnomic_reference_geometry=colnomic_reference_geometry,
        dino_query_geometry=dino_query_geometry,
        dino_reference_geometry=dino_reference_geometry,
        model_checkpoint_sha256=model_checkpoint_sha256,
        pair_comparator_sha256=pair_comparator_sha256,
        reducer_sha256=reducer_sha256,
    )


def _seal_deterministic_e1_mt_proposal_scalar_reference(
    *,
    candidate_key: str,
    direction: str,
    colnomic_query_tokens: torch.Tensor,
    colnomic_reference_tokens: torch.Tensor,
    colnomic_query_geometry: CanonicalPatchGeometry,
    colnomic_reference_geometry: CanonicalPatchGeometry,
    dino_query_geometry: CanonicalPatchGeometry,
    dino_reference_geometry: CanonicalPatchGeometry,
    model_checkpoint_sha256: str,
    pair_comparator_sha256: str,
    reducer_sha256: str,
) -> E1MTProposalSealV1:
    """Slow test oracle; not exported and never authorized for materialization."""

    return _seal_deterministic_e1_mt_proposal_impl(
        batched=False,
        candidate_key=candidate_key,
        direction=direction,
        colnomic_query_tokens=colnomic_query_tokens,
        colnomic_reference_tokens=colnomic_reference_tokens,
        colnomic_query_geometry=colnomic_query_geometry,
        colnomic_reference_geometry=colnomic_reference_geometry,
        dino_query_geometry=dino_query_geometry,
        dino_reference_geometry=dino_reference_geometry,
        model_checkpoint_sha256=model_checkpoint_sha256,
        pair_comparator_sha256=pair_comparator_sha256,
        reducer_sha256=reducer_sha256,
    )


__all__ = [
    "SCHEMA_VERSION",
    "RESOLVER_ID",
    "RESOLVER_CLAIM",
    "SUPPORT_TEMPERATURE",
    "SUPPORT_TIE_ULPS",
    "ROW_TIE_ULPS",
    "PSEL_READY",
    "PSEL_MISSING",
    "E1MTProposalContractError",
    "E1MTRootSelectionV1",
    "E1MTProposalSealV1",
    "seal_deterministic_e1_mt_proposal",
]
