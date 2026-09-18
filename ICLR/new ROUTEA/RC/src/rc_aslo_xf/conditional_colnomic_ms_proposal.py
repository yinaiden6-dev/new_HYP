"""Discrete ColNomic multi-support proposal artifacts for the conditional gate.

The proposal path is deliberately result blind.  It resolves only independently
validated spatial ColNomic cache shards and the frozen natural-C128 prejoin
records.  Similarity values are used transiently by ``lt_hyp_pvlock_v4`` to
construct a hypothesis, but no similarity, score, rank, target or label is
serialized.  The P->V interface contains only indices, masks, grid geometry and
cryptographic bindings.

Two fixed query checkerboards are used.  ``a_to_b`` proposes from parity-zero
query cells and ``b_to_a`` proposes from parity-one query cells.  Reference
proposal cells are never checkerboard-pruned: every valid reference image token
is available to both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from rc_aslo_xf.colnomic_proposal_tokens import (
    ColNomicLocalGrid,
    proposal_grid_sha256,
)
from rc_aslo_xf.conditional_rep_sources import canonical_sha256
from rc_aslo_xf.lt_hyp_pvlock_v4 import (
    DEFAULT_MAX_SUPPORT,
    MAX_HYPOTHESES,
    SealedMultiPatchHypotheses,
    propose_multisupport_p_only,
)
from rc_aslo_xf.l0_targetfree_source import sha256_file


SPATIAL_CACHE_VERSION = "rc_conditional_colnomic_p_spatial_segment_v1"
SPATIAL_VALIDATION_VERSION = (
    "rc_conditional_colnomic_p_spatial_segment_validator_v1"
)
PROPOSAL_VERSION = "rc_conditional_colnomic_ms_proposal_segment_v1"
PROPOSAL_ENTRY_VERSION = "rc_conditional_colnomic_ms_proposal_entry_v1"
PROPOSAL_STATUS = "RC_CONDITIONAL_COLNOMIC_MS_PROPOSAL_SEGMENT_READY"
PROPOSAL_RECEIPT_VERSION = (
    "rc_conditional_colnomic_ms_proposal_segment_receipt_v1"
)
PROPOSAL_VALIDATION_VERSION = (
    "rc_conditional_colnomic_ms_proposal_segment_validator_v1"
)
BANK_DIRECTIONS = ("a_to_b", "b_to_a")
BANK_PARITIES = (0, 1)
SUPPORT_CAPACITY = DEFAULT_MAX_SUPPORT
_SEGMENT_RE = re.compile(r"^(\d{4})_(\d{4})\.pt$")
PROPOSAL_TARGET_FREE_FLAGS = {
    "target_join_performed": False,
    "target_label_read": False,
    "D1_score_rank_slot_winner_read": False,
    "P_similarity_magnitude_serialized": False,
    "verifier_read_or_training_performed": False,
    "opened_runtime_read_count": 0,
    "sealed_runtime_read_count": 0,
    "a10_runtime_read_count": 0,
    "home_files_modified": 0,
}
PROPOSAL_BANK_CONTRACT = {
    "a_to_b_query_selection": "(x+y)%2 == 0",
    "b_to_a_query_selection": "(x+y)%2 == 1",
    "query_banks_complementary": True,
    "reference_selection_both_banks": "all_valid_spatial_image_tokens",
    "support_construction": "seed_local_mutual_one_to_one_matches",
    "seed_alone_is_target_evidence": False,
    "P_similarity_magnitude_serialized_or_visible_to_V": False,
    "V_consumes": "sealed_SealedMultiPatchHypotheses_only",
    "V_leaveout": "seed_and_halo",
    "old_opposite_sparse_reference_bank_required": False,
}


class ConditionalMSProposalError(RuntimeError):
    """A source or discrete proposal contract failed closed."""


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a tensor's exact dtype, shape and bytes."""

    if not isinstance(value, torch.Tensor):
        raise ConditionalMSProposalError("proposal value is not a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkerboard_query_mask(
    grid_shape: tuple[int, int], *, direction: str
) -> torch.Tensor:
    """Return one of two fixed, complementary result-blind query banks."""

    if direction not in BANK_DIRECTIONS:
        raise ConditionalMSProposalError("unknown checkerboard direction")
    if (
        not isinstance(grid_shape, tuple)
        or len(grid_shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in grid_shape)
        or min(grid_shape) <= 0
    ):
        raise ConditionalMSProposalError("invalid proposal grid")
    height, width = grid_shape
    index = torch.arange(height * width, dtype=torch.long)
    y = torch.div(index, width, rounding_mode="floor")
    x = index.remainder(width)
    parity = BANK_PARITIES[BANK_DIRECTIONS.index(direction)]
    return (x + y).remainder(2).eq(parity)


def _logical(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def _entry_logical_without_tokens(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"tokens", "entry_logical_sha256"}
        }
    )


@dataclass(frozen=True)
class SpatialTokenEntry:
    kind: str
    work_ordinal: int
    source_key: str
    query_ordinal: int | None
    query_id: str | None
    physical_row: int | None
    grid_shape: tuple[int, int]
    tokens: torch.Tensor
    tokens_sha256: str
    entry_logical_sha256: str
    cache_file: Path
    cache_file_sha256: str
    validation_file: Path
    validation_file_sha256: str


@dataclass(frozen=True)
class _SegmentDescriptor:
    kind: str
    start: int
    stop: int
    cache_file: Path
    receipt_file: Path
    validation_file: Path
    receipt: Mapping[str, Any]
    validation: Mapping[str, Any]


class ValidatedSpatialCacheResolver:
    """Resolve exact work ordinals from independently validated source shards.

    Shard metadata are checked before tensor payloads are opened.  A fully
    contained smoke shard may coexist with its later formal shard only when
    every duplicated entry has the same logical and token hashes.  Partial or
    disagreeing overlap remains an error; the resolver then keeps only the
    maximal validated shard, so every ordinal still has one source.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._descriptors: dict[str, list[_SegmentDescriptor]] = {
            "query": [],
            "gallery": [],
        }
        self._loaded: dict[Path, Mapping[str, Any]] = {}
        for kind in ("query", "gallery"):
            self._descriptors[kind] = self._scan(kind)

    def _scan(self, kind: str) -> list[_SegmentDescriptor]:
        directory = self.root / kind
        if not directory.is_dir():
            raise ConditionalMSProposalError(f"missing spatial {kind} cache directory")
        descriptors: list[_SegmentDescriptor] = []
        for cache_file in sorted(directory.glob("*.pt")):
            match = _SEGMENT_RE.fullmatch(cache_file.name)
            if match is None:
                raise ConditionalMSProposalError(
                    f"unregistered spatial cache filename: {cache_file.name}"
                )
            start, stop = map(int, match.groups())
            if not 0 <= start < stop:
                raise ConditionalMSProposalError("invalid spatial cache range")
            receipt_file = (
                self.root / "receipts" / f"{kind}_{start:04d}_{stop:04d}.json"
            )
            validation_file = (
                self.root / "validation" / f"{kind}_{start:04d}_{stop:04d}.json"
            )
            if not receipt_file.is_file() or not validation_file.is_file():
                raise ConditionalMSProposalError(
                    f"spatial source shard lacks receipt/validation: {cache_file}"
                )
            receipt = json.loads(receipt_file.read_text())
            validation = json.loads(validation_file.read_text())
            expected_range = [start, stop]
            required_checks = validation.get("checks")
            if (
                receipt.get("range") != expected_range
                or receipt.get("kind") != kind
                or receipt.get("status")
                != "RC_CONDITIONAL_COLNOMIC_P_SPATIAL_SEGMENT_READY"
                or receipt.get("logical_sha256") != _logical(receipt)
                or validation.get("version") != SPATIAL_VALIDATION_VERSION
                or validation.get("range") != expected_range
                or validation.get("kind") != kind
                or validation.get("status") != "PASS"
                or validation.get("logical_sha256") != _logical(validation)
                or not isinstance(required_checks, dict)
                or not required_checks
                or not all(value is True for value in required_checks.values())
                or validation.get("cache_file_sha256") != sha256_file(cache_file)
                or validation.get("materializer_receipt_sha256")
                != sha256_file(receipt_file)
                or validation.get("cache_logical_sha256")
                != receipt.get("cache_logical_sha256")
            ):
                raise ConditionalMSProposalError(
                    f"spatial source receipt failed closed: {cache_file}"
                )
            descriptors.append(
                _SegmentDescriptor(
                    kind=kind,
                    start=start,
                    stop=stop,
                    cache_file=cache_file,
                    receipt_file=receipt_file,
                    validation_file=validation_file,
                    receipt=receipt,
                    validation=validation,
                )
            )
        # Prefer maximal formal shards, but accept an earlier contained smoke
        # only after replaying the duplicated per-entry hashes.  This is not a
        # generic overlap relaxation: partial overlap and any content drift are
        # still fail-closed.
        maximal: list[_SegmentDescriptor] = []
        for candidate in sorted(
            descriptors,
            key=lambda item: (-(item.stop - item.start), item.start, item.stop),
        ):
            containers = [
                item
                for item in maximal
                if item.start <= candidate.start and candidate.stop <= item.stop
            ]
            partial = [
                item
                for item in maximal
                if max(item.start, candidate.start) < min(item.stop, candidate.stop)
                and item not in containers
            ]
            if partial:
                raise ConditionalMSProposalError(
                    f"partially overlapping validated {kind} cache shards are ambiguous"
                )
            if containers:
                container = containers[0]
                left = self._payload(candidate)["entries"]
                right = self._payload(container)["entries"]
                for ordinal in range(candidate.start, candidate.stop):
                    duplicate = left[ordinal - candidate.start]
                    formal = right[ordinal - container.start]
                    if (
                        duplicate.get("entry_logical_sha256")
                        != formal.get("entry_logical_sha256")
                        or duplicate.get("tokens_sha256")
                        != formal.get("tokens_sha256")
                    ):
                        raise ConditionalMSProposalError(
                            f"contained validated {kind} smoke differs from formal shard"
                        )
                continue
            maximal.append(candidate)
        maximal.sort(key=lambda item: (item.start, item.stop))
        for left, right in zip(maximal, maximal[1:], strict=False):
            if right.start < left.stop:
                raise ConditionalMSProposalError(
                    f"overlapping validated {kind} cache shards are ambiguous"
                )
        return maximal

    def available_ranges(self, kind: str) -> tuple[tuple[int, int], ...]:
        if kind not in self._descriptors:
            raise ConditionalMSProposalError("unknown spatial cache kind")
        return tuple((item.start, item.stop) for item in self._descriptors[kind])

    def _descriptor(self, kind: str, work_ordinal: int) -> _SegmentDescriptor:
        if kind not in self._descriptors or isinstance(work_ordinal, bool):
            raise ConditionalMSProposalError("invalid spatial source lookup")
        matches = [
            descriptor
            for descriptor in self._descriptors[kind]
            if descriptor.start <= int(work_ordinal) < descriptor.stop
        ]
        if len(matches) != 1:
            raise ConditionalMSProposalError(
                f"validated {kind} cache does not uniquely cover work ordinal "
                f"{work_ordinal}"
            )
        return matches[0]

    def _payload(self, descriptor: _SegmentDescriptor) -> Mapping[str, Any]:
        if descriptor.cache_file in self._loaded:
            return self._loaded[descriptor.cache_file]
        value = torch.load(
            descriptor.cache_file,
            map_location="cpu",
            weights_only=False,
        )
        entries = value.get("entries") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("version") != SPATIAL_CACHE_VERSION
            or value.get("kind") != descriptor.kind
            or value.get("range") != [descriptor.start, descriptor.stop]
            or not isinstance(entries, list)
            or len(entries) != descriptor.stop - descriptor.start
            or value.get("logical_sha256")
            != descriptor.receipt.get("cache_logical_sha256")
        ):
            raise ConditionalMSProposalError("spatial cache payload schema drift")
        self._loaded[descriptor.cache_file] = value
        return value

    def get(self, kind: str, work_ordinal: int) -> SpatialTokenEntry:
        descriptor = self._descriptor(kind, work_ordinal)
        payload = self._payload(descriptor)
        offset = int(work_ordinal) - descriptor.start
        raw = payload["entries"][offset]
        # ``work_ordinal`` indexes a target-free execution shard.  The frozen
        # source key instead binds the intrinsic query ordinal or physical
        # gallery row and these are not interchangeable.
        intrinsic_key = (
            raw.get("query_ordinal")
            if isinstance(raw, dict) and kind == "query"
            else raw.get("physical_row") if isinstance(raw, dict) else None
        )
        expected_key = f"{kind}:{intrinsic_key}"
        tokens = raw.get("tokens") if isinstance(raw, dict) else None
        grid = raw.get("grid_shape") if isinstance(raw, dict) else None
        expected_entry_logical: str | None = None
        if (
            isinstance(raw, dict)
            and isinstance(tokens, torch.Tensor)
            and isinstance(grid, list)
            and len(grid) == 2
        ):
            if kind == "query":
                expected_entry_logical = _entry_logical_without_tokens(raw)
            else:
                indices = raw.get("image_token_indices")
                if isinstance(indices, list):
                    try:
                        local_grid = ColNomicLocalGrid(
                            tokens=tokens,
                            grid_shape=(int(grid[0]), int(grid[1])),
                            image_token_indices=torch.tensor(indices, dtype=torch.long),
                        )
                    except (TypeError, ValueError):
                        expected_entry_logical = None
                    else:
                        expected_entry_logical = canonical_sha256(
                            {
                                **{
                                    key: item
                                    for key, item in raw.items()
                                    if key not in {"tokens", "entry_logical_sha256"}
                                },
                                "proposal_grid_sha256": proposal_grid_sha256(
                                    local_grid
                                ),
                            }
                        )
        if (
            not isinstance(raw, dict)
            or raw.get("kind") != kind
            or raw.get("work_ordinal") != int(work_ordinal)
            or isinstance(intrinsic_key, bool)
            or not isinstance(intrinsic_key, int)
            or raw.get("key") != expected_key
            or not isinstance(grid, list)
            or len(grid) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in grid)
            or min(grid) <= 0
            or not isinstance(tokens, torch.Tensor)
            or tokens.dtype != torch.float16
            or tokens.ndim != 2
            or tokens.shape != (math.prod(grid), 128)
            or not bool(torch.isfinite(tokens).all())
            or raw.get("tokens_sha256") != tensor_sha256(tokens)
            or raw.get("entry_logical_sha256")
            != expected_entry_logical
        ):
            raise ConditionalMSProposalError("spatial cache entry drift")
        query_ordinal: int | None = None
        query_id: str | None = None
        physical_row: int | None = None
        if kind == "query":
            query_ordinal = raw.get("query_ordinal")
            query_id = raw.get("query_id")
            if (
                isinstance(query_ordinal, bool)
                or not isinstance(query_ordinal, int)
                or not isinstance(query_id, str)
                or not query_id
            ):
                raise ConditionalMSProposalError("query spatial binding drift")
        else:
            physical_row = raw.get("physical_row")
            if isinstance(physical_row, bool) or not isinstance(physical_row, int):
                raise ConditionalMSProposalError("gallery spatial binding drift")
        return SpatialTokenEntry(
            kind=kind,
            work_ordinal=int(work_ordinal),
            source_key=expected_key,
            query_ordinal=query_ordinal,
            query_id=query_id,
            physical_row=physical_row,
            grid_shape=(int(grid[0]), int(grid[1])),
            tokens=tokens.detach().contiguous(),
            tokens_sha256=str(raw["tokens_sha256"]),
            entry_logical_sha256=str(raw["entry_logical_sha256"]),
            cache_file=descriptor.cache_file,
            cache_file_sha256=str(descriptor.validation["cache_file_sha256"]),
            validation_file=descriptor.validation_file,
            validation_file_sha256=sha256_file(descriptor.validation_file),
        )


def gallery_work_ordinal_by_physical_row(
    candidate_union: Sequence[int],
) -> dict[int, int]:
    """Bind physical rows to the sorted-union work order used by the P cache."""

    rows = tuple(int(item) for item in candidate_union)
    if rows != tuple(sorted(rows)) or len(set(rows)) != len(rows):
        raise ConditionalMSProposalError("candidate union is not a unique sorted order")
    return {row: ordinal for ordinal, row in enumerate(rows)}


def build_discrete_candidate_hypotheses(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    *,
    query_grid_shape: tuple[int, int],
    reference_grid_shape: tuple[int, int],
    device: torch.device | str = "cpu",
) -> tuple[SealedMultiPatchHypotheses, SealedMultiPatchHypotheses]:
    """Build the two fixed proposal banks; all P magnitudes die in this call."""

    run_device = torch.device(device)
    query = query_tokens.to(run_device)
    reference = reference_tokens.to(run_device)
    reference_mask = torch.ones(reference.shape[0], dtype=torch.bool, device=run_device)
    outputs = []
    for direction in BANK_DIRECTIONS:
        query_mask = checkerboard_query_mask(
            query_grid_shape, direction=direction
        ).to(run_device)
        output = propose_multisupport_p_only(
            query,
            reference,
            query_grid_shape=query_grid_shape,
            reference_grid_shape=reference_grid_shape,
            direction=direction,
            query_selection_mask=query_mask,
            reference_selection_mask=reference_mask,
        )
        outputs.append(output)
    return outputs[0], outputs[1]


def hypothesis_to_discrete(
    hypothesis: SealedMultiPatchHypotheses,
) -> dict[str, Any]:
    """Serialize only the discrete P->V interface (no weights or magnitudes)."""

    tensors = {
        "seed_query_indices": hypothesis.seed_query_indices.to(torch.int32),
        "seed_reference_indices": hypothesis.seed_reference_indices.to(torch.int32),
        "support_query_indices": hypothesis.support_query_indices.to(torch.int32),
        "support_reference_indices": hypothesis.support_reference_indices.to(torch.int32),
        "support_mask": hypothesis.support_mask.to(torch.bool),
        "legal": hypothesis.legal.to(torch.bool),
    }
    value: dict[str, Any] = {
        "direction": hypothesis.direction,
        "query_grid_shape": list(hypothesis.query_grid_shape),
        "reference_grid_shape": list(hypothesis.reference_grid_shape),
        **tensors,
        "tensor_sha256": {
            key: tensor_sha256(tensor) for key, tensor in tensors.items()
        },
    }
    value["logical_sha256"] = discrete_hypothesis_logical_sha256(value)
    return value


def discrete_hypothesis_logical_sha256(value: Mapping[str, Any]) -> str:
    """Hash metadata plus tensor receipts, never a transient P score."""

    return canonical_sha256(
        {
            "direction": value.get("direction"),
            "query_grid_shape": value.get("query_grid_shape"),
            "reference_grid_shape": value.get("reference_grid_shape"),
            "tensor_sha256": value.get("tensor_sha256"),
        }
    )


def hypothesis_from_discrete(value: Mapping[str, Any]) -> SealedMultiPatchHypotheses:
    """Reconstruct uniform weights and re-run every V4 structural invariant."""

    required_tensors = {
        "seed_query_indices": (torch.int32, (MAX_HYPOTHESES,)),
        "seed_reference_indices": (torch.int32, (MAX_HYPOTHESES,)),
        "support_query_indices": (
            torch.int32,
            (MAX_HYPOTHESES, SUPPORT_CAPACITY),
        ),
        "support_reference_indices": (
            torch.int32,
            (MAX_HYPOTHESES, SUPPORT_CAPACITY),
        ),
        "support_mask": (torch.bool, (MAX_HYPOTHESES, SUPPORT_CAPACITY)),
        "legal": (torch.bool, (MAX_HYPOTHESES,)),
    }
    receipts = value.get("tensor_sha256")
    if (
        value.get("direction") not in BANK_DIRECTIONS
        or not isinstance(receipts, dict)
        or set(receipts) != set(required_tensors)
        or value.get("logical_sha256") != discrete_hypothesis_logical_sha256(value)
    ):
        raise ConditionalMSProposalError("discrete hypothesis metadata drift")
    tensors: dict[str, torch.Tensor] = {}
    for key, (dtype, shape) in required_tensors.items():
        tensor = value.get(key)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or receipts.get(key) != tensor_sha256(tensor)
        ):
            raise ConditionalMSProposalError("discrete hypothesis tensor drift")
        tensors[key] = tensor.detach().cpu().contiguous()
    support_mask = tensors["support_mask"]
    counts = support_mask.sum(dim=1)
    weights = torch.zeros(support_mask.shape, dtype=torch.float64)
    for index in range(MAX_HYPOTHESES):
        if int(counts[index]) > 0:
            weights[index, support_mask[index]] = 1.0 / float(counts[index])
    query_grid = value.get("query_grid_shape")
    reference_grid = value.get("reference_grid_shape")
    if (
        not isinstance(query_grid, list)
        or not isinstance(reference_grid, list)
        or len(query_grid) != 2
        or len(reference_grid) != 2
    ):
        raise ConditionalMSProposalError("discrete hypothesis grid drift")
    return SealedMultiPatchHypotheses(
        seed_query_indices=tensors["seed_query_indices"].to(torch.long),
        seed_reference_indices=tensors["seed_reference_indices"].to(torch.long),
        support_query_indices=tensors["support_query_indices"].to(torch.long),
        support_reference_indices=tensors["support_reference_indices"].to(torch.long),
        support_weights=weights,
        support_mask=support_mask,
        legal=tensors["legal"],
        query_grid_shape=(int(query_grid[0]), int(query_grid[1])),
        reference_grid_shape=(int(reference_grid[0]), int(reference_grid[1])),
        direction=str(value["direction"]),
    )


def assert_discrete_only(value: Any) -> None:
    """Reject P magnitudes, target/D1 fields and floating tensors recursively."""

    forbidden_fragments = (
        "target",
        "label",
        "identity",
        "d1",
        "rank",
        "slot",
        "winner",
        "score",
        "similarity",
        "magnitude",
        "logit",
    )

    def visit(item: Any, path: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in forbidden_fragments):
                    raise ConditionalMSProposalError(
                        f"forbidden proposal field at {'.'.join((*path, str(key)))}"
                    )
                visit(child, (*path, str(key)))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, (*path, str(index)))
        elif isinstance(item, torch.Tensor) and item.is_floating_point():
            raise ConditionalMSProposalError(
                f"floating proposal tensor at {'.'.join(path)}"
            )

    visit(value, ())


def proposal_entry_logical_sha256(entry: Mapping[str, Any]) -> str:
    """Hash a proposal entry without serializing tensor bytes into JSON."""

    banks = entry.get("banks")
    if not isinstance(banks, list):
        raise ConditionalMSProposalError("proposal entry banks are missing")
    compact_banks: list[list[dict[str, Any]]] = []
    for bank in banks:
        if not isinstance(bank, list):
            raise ConditionalMSProposalError("proposal bank is not a list")
        compact_banks.append(
            [
                {
                    "direction": item.get("direction"),
                    "query_grid_shape": item.get("query_grid_shape"),
                    "reference_grid_shape": item.get("reference_grid_shape"),
                    "tensor_sha256": item.get("tensor_sha256"),
                    "logical_sha256": item.get("logical_sha256"),
                }
                for item in bank
            ]
        )
    compact = {
        key: item
        for key, item in entry.items()
        if key not in {"banks", "entry_logical_sha256"}
    }
    compact["banks"] = compact_banks
    return canonical_sha256(compact)


def proposal_payload_logical_sha256(payload: Mapping[str, Any]) -> str:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ConditionalMSProposalError("proposal payload entries are missing")
    compact = {
        key: item
        for key, item in payload.items()
        if key not in {"entries", "logical_sha256"}
    }
    compact["entries"] = [
        {
            key: item
            for key, item in entry.items()
            if key != "banks"
        }
        for entry in entries
    ]
    return canonical_sha256(compact)


def candidate_executability(
    candidate_legal_by_bank: torch.Tensor, *, target_position: int
) -> dict[str, Any]:
    """Compute the fixed rank-free same-bank target/non-target P gate."""

    legal = torch.as_tensor(candidate_legal_by_bank, dtype=torch.bool)
    if (
        legal.ndim != 2
        or legal.shape[0] != len(BANK_DIRECTIONS)
        or legal.shape[1] < 2
        or isinstance(target_position, bool)
        or int(target_position) not in range(legal.shape[1])
    ):
        raise ConditionalMSProposalError("candidate-legality gate input drift")
    target_position = int(target_position)
    non_target = torch.ones(legal.shape[1], dtype=torch.bool)
    non_target[target_position] = False
    target_by_bank = legal[:, target_position]
    non_target_count_by_bank = legal[:, non_target].sum(dim=1)
    same_bank = target_by_bank & non_target_count_by_bank.gt(0)
    any_bank = legal.any(dim=0)
    both_bank = legal.all(dim=0)
    return {
        "target_legal_by_bank": target_by_bank,
        "non_target_count_by_bank": non_target_count_by_bank,
        "same_bank_executable": same_bank,
        "non_target_union_count": int(any_bank[non_target].sum()),
        "non_target_both_bank_count": int(both_bank[non_target].sum()),
    }


class ValidatedMSProposalResolver:
    """Resolve target-free proposal entries only after independent validation."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        directory = self.root / "prejoin"
        if not directory.is_dir():
            raise ConditionalMSProposalError("proposal prejoin directory is missing")
        self._descriptors: list[_SegmentDescriptor] = []
        self._loaded: dict[Path, Mapping[str, Any]] = {}
        for result_file in sorted(directory.glob("*.pt")):
            match = _SEGMENT_RE.fullmatch(result_file.name)
            if match is None:
                raise ConditionalMSProposalError(
                    f"unregistered proposal filename: {result_file.name}"
                )
            start, stop = map(int, match.groups())
            receipt_file = self.root / "receipts" / f"{start:04d}_{stop:04d}.json"
            validation_file = (
                self.root / "validation" / f"{start:04d}_{stop:04d}.json"
            )
            if not receipt_file.is_file() or not validation_file.is_file():
                raise ConditionalMSProposalError(
                    "proposal segment lacks receipt or independent validation"
                )
            receipt = json.loads(receipt_file.read_text())
            validation = json.loads(validation_file.read_text())
            checks = validation.get("checks")
            if (
                not 0 <= start < stop
                or receipt.get("version") != PROPOSAL_RECEIPT_VERSION
                or receipt.get("status") != PROPOSAL_STATUS
                or receipt.get("range") != [start, stop]
                or receipt.get("logical_sha256") != _logical(receipt)
                or validation.get("version") != PROPOSAL_VALIDATION_VERSION
                or validation.get("status") != "PASS"
                or validation.get("range") != [start, stop]
                or validation.get("result") != str(result_file)
                or validation.get("result_file_sha256") != sha256_file(result_file)
                or validation.get("materializer_receipt_sha256")
                != sha256_file(receipt_file)
                or validation.get("logical_sha256") != _logical(validation)
                or not isinstance(checks, dict)
                or not checks
                or not all(value is True for value in checks.values())
            ):
                raise ConditionalMSProposalError(
                    f"proposal validation failed closed: {result_file}"
                )
            self._descriptors.append(
                _SegmentDescriptor(
                    kind="proposal",
                    start=start,
                    stop=stop,
                    cache_file=result_file,
                    receipt_file=receipt_file,
                    validation_file=validation_file,
                    receipt=receipt,
                    validation=validation,
                )
            )
        self._descriptors.sort(key=lambda item: (item.start, item.stop))
        for left, right in zip(self._descriptors, self._descriptors[1:], strict=False):
            if right.start < left.stop:
                raise ConditionalMSProposalError(
                    "overlapping validated proposal segments are ambiguous"
                )

    def ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple((item.start, item.stop) for item in self._descriptors)

    def covered_ordinals(self) -> tuple[int, ...]:
        return tuple(
            ordinal
            for item in self._descriptors
            for ordinal in range(item.start, item.stop)
        )

    def binding_receipt(self) -> dict[str, Any]:
        """Return an auditable hash-only binding for all validated segments."""

        rows = []
        for descriptor in self._descriptors:
            payload = self._payload(descriptor)
            rows.append(
                {
                    "range": [descriptor.start, descriptor.stop],
                    "result_file_sha256": sha256_file(descriptor.cache_file),
                    "result_logical_sha256": payload["logical_sha256"],
                    "receipt_file_sha256": sha256_file(descriptor.receipt_file),
                    "validation_file_sha256": sha256_file(
                        descriptor.validation_file
                    ),
                    "source_binding_logical_sha256": payload[
                        "source_binding"
                    ]["logical_sha256"],
                }
            )
        return {
            "segment_count": len(rows),
            "covered_ordinals": list(self.covered_ordinals()),
            "segment_sequence": rows,
            "segment_sequence_sha256": canonical_sha256(rows),
        }

    def _descriptor(self, work_ordinal: int) -> _SegmentDescriptor:
        matches = [
            item
            for item in self._descriptors
            if item.start <= int(work_ordinal) < item.stop
        ]
        if len(matches) != 1:
            raise ConditionalMSProposalError(
                f"proposal work ordinal {work_ordinal} is not uniquely validated"
            )
        return matches[0]

    def _payload(self, descriptor: _SegmentDescriptor) -> Mapping[str, Any]:
        if descriptor.cache_file in self._loaded:
            return self._loaded[descriptor.cache_file]
        value = torch.load(
            descriptor.cache_file, map_location="cpu", weights_only=False
        )
        entries = value.get("entries") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("version") != PROPOSAL_VERSION
            or value.get("status") != PROPOSAL_STATUS
            or value.get("range") != [descriptor.start, descriptor.stop]
            or value.get("banks") != list(BANK_DIRECTIONS)
            or value.get("proposal_bank_contract") != PROPOSAL_BANK_CONTRACT
            or value.get("flags") != PROPOSAL_TARGET_FREE_FLAGS
            or not isinstance(entries, list)
            or len(entries) != descriptor.stop - descriptor.start
            or value.get("logical_sha256")
            != proposal_payload_logical_sha256(value)
            or value.get("logical_sha256")
            != descriptor.validation.get("result_logical_sha256")
        ):
            raise ConditionalMSProposalError("validated proposal payload drift")
        self._loaded[descriptor.cache_file] = value
        return value

    def get(self, work_ordinal: int) -> Mapping[str, Any]:
        descriptor = self._descriptor(work_ordinal)
        payload = self._payload(descriptor)
        entry = payload["entries"][int(work_ordinal) - descriptor.start]
        banks = entry.get("banks") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or entry.get("version") != PROPOSAL_ENTRY_VERSION
            or entry.get("work_ordinal") != int(work_ordinal)
            or entry.get("entry_logical_sha256")
            != proposal_entry_logical_sha256(entry)
            or not isinstance(banks, list)
            or len(banks) != 2
            or any(not isinstance(bank, list) or len(bank) != 128 for bank in banks)
        ):
            raise ConditionalMSProposalError("validated proposal entry drift")
        for bank_index, direction in enumerate(BANK_DIRECTIONS):
            for discrete in banks[bank_index]:
                assert_discrete_only(discrete)
                hypothesis = hypothesis_from_discrete(discrete)
                if hypothesis.direction != direction:
                    raise ConditionalMSProposalError(
                        "proposal bank/direction binding drift"
                    )
        return entry
