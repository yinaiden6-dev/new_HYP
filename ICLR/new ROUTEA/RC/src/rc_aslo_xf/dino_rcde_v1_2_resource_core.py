"""DINO-RCDE V1.2 engineering core for E0 and resource qualification.

This module implements the complete 65,125-parameter architecture frozen by
the V1.1 design.  It deliberately does *not* implement data joins, retrieval
actions, target labels, or a scientific GO decision.  The resource runner can
therefore exercise the exact model graph without being able to silently grant
authority to train on natural roles.

The dense decoder consumes only a query/reference cross-cost volume plus the
two geometry-valid masks.  The pair comparator consumes only the two
cross-cost-derived 16-dimensional relational fields.  There is no query-only,
reference-only, rank, slot, identity, or score input to the forward API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


INPUT_DIM = 768
OUTPUT_DIM = 64
ADAPTER_RANK = 8
SUMMARY_DIM = 16
CONSENSUS_WIDTH = 8
COST_TEMPERATURE = 0.07
EXPECTED_PARAMETER_COUNT = 65_125
# V1.2 repairs execution only.  The scientific model, including its exact
# counter-mode initialization stream, remains the V1.1 object.
INIT_NAMESPACE = "RCDE_INIT_V1_1_SEED17"

SUMMARY_NAMES = (
    "matched_mass",
    "query_dustbin_mass",
    "max_assignment",
    "top1_top2_gap",
    "normalized_entropy",
    "reciprocal_assignment_mass",
    "assigned_logit_mean",
    "assigned_logit_variance",
    "cycle_return_mass",
    "reference_x_mean",
    "reference_y_mean",
    "reference_x_variance",
    "reference_y_variance",
    "reference_xy_covariance",
    "four_neighbour_displacement_agreement",
    "inverse_column_crowding_support",
)

# These two intentionally differ.  ``matched_mass`` is the one-way row
# non-dustbin probability 1-u.  ``reciprocal_assignment_mass`` is the mass of
# the product assignment sum_j Q_ij G_ij.  Keeping both is part of the frozen
# 16-field interface, not an accidental duplicate.
SUMMARY_FORMULAS = {
    "matched_mass": "1-u_i",
    "reciprocal_assignment_mass": "sum_j P_ij where P_ij=Q_ij*G_ij",
}


class RCDEContractError(ValueError):
    """Raised when a tensor or geometry violates the frozen RCDE interface."""


@dataclass(frozen=True)
class Assignment:
    """Dense dual-softmax partial assignment."""

    Q: torch.Tensor
    G: torch.Tensor
    P: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class CandidateEvidence:
    """All E0-auditable tensors for one query/reference candidate."""

    cost: torch.Tensor
    logits: torch.Tensor
    assignment: Assignment
    raw_summary: torch.Tensor
    null_summary: torch.Tensor
    relational: torch.Tensor


@dataclass(frozen=True)
class PairEvidence:
    """Strictly antisymmetric target-free evidence for an ordered pair."""

    signed_support: torch.Tensor
    modulation: torch.Tensor
    evidence: torch.Tensor
    contributions: torch.Tensor
    logit: torch.Tensor


@dataclass(frozen=True)
class StreamedCandidateEvidence:
    """Candidate evidence produced without a resident full cross-cost or P.

    ``assignment`` is populated only for a deliberately small dense-closure
    audit.  The deployed/worst-shape path leaves it ``None`` and exposes
    canonical row-major hashes and fixed slices instead.
    """

    raw_summary: torch.Tensor
    null_summary: torch.Tensor
    relational: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    matrix_sha256: Mapping[str, str]
    matrix_audit_slices: Sequence[Mapping[str, float | int]]
    assignment: Assignment | None
    logical_cost_volume_build_count: int
    consensus_tile_count_per_pass: int
    real_pass_tile_counts: tuple[int, int, int]
    null_pass_tile_counts: tuple[int, int, int]
    maximum_resident_cost_elements: int
    summary_pass_count: int
    real_consensus_evaluation_count: int
    null_consensus_evaluation_count: int
    full_consensus_logits_resident: bool
    resident_consensus_logit_elements: int
    matrix_audit_enabled: bool
    analytic_null: bool


def _canonical_mask(mask: torch.Tensor, grid: tuple[int, int], name: str) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool)
    if tuple(value.shape) != tuple(grid):
        raise RCDEContractError(f"{name} mask/grid mismatch: {tuple(value.shape)} != {grid}")
    if not bool(value.any()):
        raise RCDEContractError(f"{name} has no geometry-valid patch")
    return value.contiguous()


def _validate_layers(
    layers: torch.Tensor,
    mask: torch.Tensor,
    grid: tuple[int, int],
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.as_tensor(layers)
    valid = _canonical_mask(mask, grid, name)
    if (
        value.ndim != 3
        or tuple(value.shape[:1]) != (4,)
        or value.shape[1] != valid.numel()
        or value.shape[2] != INPUT_DIM
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise RCDEContractError(
            f"{name} layers must be finite floating [4,H*W,{INPUT_DIM}] tokens"
        )
    valid_rows = valid.flatten()
    input_norms = torch.linalg.vector_norm(value[:, valid_rows], dim=-1)
    if not bool(input_norms.gt(1.0e-12).all()):
        raise RCDEContractError(f"{name} has a geometry-valid DINO token with norm <=1e-12")
    return value, valid


def _hash_uniform(path: str, count: int) -> torch.Tensor:
    """Counter-mode SHA256 uniform values in the open interval (0, 1)."""

    values: list[float] = []
    counter = 0
    while len(values) < count:
        digest = hashlib.sha256(
            f"{INIT_NAMESPACE}\0{path}\0{counter}".encode("utf-8")
        ).digest()
        for offset in range(0, 32, 8):
            integer = int.from_bytes(digest[offset : offset + 8], "big")
            # Exactly representable 53-bit mantissa, excluding both endpoints.
            values.append(((integer >> 11) + 0.5) / float(1 << 53))
            if len(values) == count:
                break
        counter += 1
    return torch.tensor(values, dtype=torch.float64)


def _hash_gaussian(path: str, count: int) -> torch.Tensor:
    pairs = (count + 1) // 2
    uniform = _hash_uniform(path, pairs * 2).reshape(pairs, 2)
    radius = torch.sqrt(-2.0 * torch.log(uniform[:, 0]))
    angle = 2.0 * math.pi * uniform[:, 1]
    normal = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
    return normal.flatten()[:count]


def _copy_uniform(parameter: nn.Parameter, path: str, low: float, high: float) -> None:
    values = _hash_uniform(path, parameter.numel()).mul(high - low).add(low)
    parameter.data.copy_(values.reshape(parameter.shape).to(parameter.dtype))


def _copy_xavier(parameter: nn.Parameter, path: str) -> None:
    if parameter.ndim == 2:
        fan_out, fan_in = parameter.shape
    elif parameter.ndim == 4:
        fan_out = parameter.shape[0] * parameter.shape[2] * parameter.shape[3]
        fan_in = parameter.shape[1] * parameter.shape[2] * parameter.shape[3]
    else:  # pragma: no cover - all V1.2 xavier tensors are 2-D or 4-D
        raise RCDEContractError(f"unsupported Xavier tensor rank for {path}")
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    _copy_uniform(parameter, path, -bound, bound)


def _normalized_coordinates(
    grid: tuple[int, int],
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Coordinates normalized over the geometry-valid extent, not padding.

    The canonical mask is a top-left complete-footprint rectangle, but this
    implementation deliberately derives the occupied row/column extent from
    the mask so padded grid size cannot become candidate identity evidence.
    Coordinates outside that extent are immaterial because every downstream
    statistic is multiplied by the pair-valid mask.
    """

    height, width = grid
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    if tuple(valid.shape) != tuple(grid) or not bool(valid.any()):
        raise RCDEContractError("coordinate valid-mask/grid mismatch")
    occupied_y = torch.nonzero(valid.any(dim=1), as_tuple=False).flatten()
    occupied_x = torch.nonzero(valid.any(dim=0), as_tuple=False).flatten()
    y = torch.zeros(height, dtype=dtype, device=device)
    x = torch.zeros(width, dtype=dtype, device=device)
    if occupied_y.numel() > 1:
        first, last = int(occupied_y[0]), int(occupied_y[-1])
        y = 2.0 * (torch.arange(height, dtype=dtype, device=device) - first) / (last - first) - 1.0
    if occupied_x.numel() > 1:
        first, last = int(occupied_x[0]), int(occupied_x[-1])
        x = 2.0 * (torch.arange(width, dtype=dtype, device=device) - first) / (last - first) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=1)


def _masked_top2(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked = values.masked_fill(~valid[None, :], float("-inf"))
    k = min(2, int(valid.sum()))
    top = torch.topk(masked, k=k, dim=1).values
    first = top[:, 0]
    second = top[:, 1] if k == 2 else torch.zeros_like(first)
    return first, second


class Separable4DConsensus(nn.Module):
    """The exact bias-free 1->8->8->8->8->1 V1.1 consensus graph."""

    def __init__(self) -> None:
        super().__init__()
        self.ref1 = nn.Conv2d(1, CONSENSUS_WIDTH, 3, padding=1, bias=False)
        self.query1 = nn.Conv2d(CONSENSUS_WIDTH, CONSENSUS_WIDTH, 3, padding=1, bias=False)
        self.ref2 = nn.Conv2d(CONSENSUS_WIDTH, CONSENSUS_WIDTH, 3, padding=1, bias=False)
        self.query2 = nn.Conv2d(CONSENSUS_WIDTH, CONSENSUS_WIDTH, 3, padding=1, bias=False)
        self.out = nn.Conv2d(CONSENSUS_WIDTH, 1, 1, bias=False)

    @staticmethod
    def _ref_conv(value: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
        # [B,C,Hq,Wq,Hr,Wr] -> independent reference-grid convolutions.
        batch, channels, hq, wq, hr, wr = value.shape
        flat = value.permute(0, 2, 3, 1, 4, 5).reshape(batch * hq * wq, channels, hr, wr)
        result = layer(flat)
        out_channels = result.shape[1]
        return result.reshape(batch, hq, wq, out_channels, hr, wr).permute(0, 3, 1, 2, 4, 5)

    @staticmethod
    def _query_conv(value: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
        # [B,C,Hq,Wq,Hr,Wr] -> independent query-grid convolutions.
        batch, channels, hq, wq, hr, wr = value.shape
        flat = value.permute(0, 4, 5, 1, 2, 3).reshape(batch * hr * wr, channels, hq, wq)
        result = layer(flat)
        out_channels = result.shape[1]
        return result.reshape(batch, hr, wr, out_channels, hq, wq).permute(0, 3, 4, 5, 1, 2)

    def forward(
        self,
        cost: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        hq, wq = query_grid
        hr, wr = reference_grid
        if tuple(cost.shape) != (hq * wq, hr * wr):
            raise RCDEContractError("cost volume/grid mismatch")
        pair_mask = (query_mask.flatten()[:, None] & reference_mask.flatten()[None, :])
        mask6 = pair_mask.reshape(1, 1, hq, wq, hr, wr).to(device=cost.device, dtype=cost.dtype)
        base = cost.reshape(1, 1, hq, wq, hr, wr) * mask6
        value = F.gelu(self._ref_conv(base, self.ref1)) * mask6
        value = F.gelu(self._query_conv(value, self.query1)) * mask6
        value = F.gelu(self._ref_conv(value, self.ref2)) * mask6
        value = F.gelu(self._query_conv(value, self.query2)) * mask6
        value = self._ref_conv(value, self.out) * mask6
        return (value + base).reshape(hq * wq, hr * wr)

    def forward_tiled(
        self,
        cost: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        tile_shape: tuple[int, int, int, int],
        halo: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact halo-tiled execution of the frozen separable 4-D graph.

        The two query-grid and two reference-grid 3x3 convolutions give a
        receptive-field radius of two on every one of the four spatial axes.
        Each tile is therefore evaluated with a fixed two-cell halo and only
        its disjoint interior is written to the global output.  Global
        dual-softmax normalization is deliberately *not* performed here; the
        caller applies it after reassembly (densely or by global streaming).

        ``coverage`` is returned as an independently auditable integer tensor.
        Every global cost-volume cell must be written exactly once.
        """

        if halo != 2:
            raise RCDEContractError("V1.2 4D consensus halo must equal 2")
        if len(tile_shape) != 4 or any(int(value) <= 0 for value in tile_shape):
            raise RCDEContractError("4D consensus tile shape must contain four positive integers")
        hq, wq = query_grid
        hr, wr = reference_grid
        if tuple(cost.shape) != (hq * wq, hr * wr):
            raise RCDEContractError("tiled cost volume/grid mismatch")
        query_mask = _canonical_mask(query_mask, query_grid, "query")
        reference_mask = _canonical_mask(reference_mask, reference_grid, "reference")
        cost4 = cost.reshape(hq, wq, hr, wr)
        result4 = torch.zeros_like(cost4)
        coverage = torch.zeros((hq, wq, hr, wr), dtype=torch.int16, device=cost.device)
        tqy, tqx, try_, trx = (int(value) for value in tile_shape)

        def intervals(length: int, width: int) -> Iterable[tuple[int, int]]:
            for start in range(0, length, width):
                yield start, min(length, start + width)

        for qy0, qy1 in intervals(hq, tqy):
            for qx0, qx1 in intervals(wq, tqx):
                for ry0, ry1 in intervals(hr, try_):
                    for rx0, rx1 in intervals(wr, trx):
                        eqy0, eqy1 = max(0, qy0 - halo), min(hq, qy1 + halo)
                        eqx0, eqx1 = max(0, qx0 - halo), min(wq, qx1 + halo)
                        ery0, ery1 = max(0, ry0 - halo), min(hr, ry1 + halo)
                        erx0, erx1 = max(0, rx0 - halo), min(wr, rx1 + halo)
                        sub_q_grid = (eqy1 - eqy0, eqx1 - eqx0)
                        sub_r_grid = (ery1 - ery0, erx1 - erx0)
                        sub_q_mask = query_mask[eqy0:eqy1, eqx0:eqx1]
                        sub_r_mask = reference_mask[ery0:ery1, erx0:erx1]
                        sub_cost = cost4[
                            eqy0:eqy1, eqx0:eqx1, ery0:ery1, erx0:erx1
                        ].reshape(math.prod(sub_q_grid), math.prod(sub_r_grid))
                        sub_result = self.forward(
                            sub_cost,
                            sub_q_grid,
                            sub_r_grid,
                            sub_q_mask,
                            sub_r_mask,
                        ).reshape(*sub_q_grid, *sub_r_grid)
                        local_qy = slice(qy0 - eqy0, qy1 - eqy0)
                        local_qx = slice(qx0 - eqx0, qx1 - eqx0)
                        local_ry = slice(ry0 - ery0, ry1 - ery0)
                        local_rx = slice(rx0 - erx0, rx1 - erx0)
                        result4[qy0:qy1, qx0:qx1, ry0:ry1, rx0:rx1] = sub_result[
                            local_qy, local_qx, local_ry, local_rx
                        ]
                        coverage[qy0:qy1, qx0:qx1, ry0:ry1, rx0:rx1] += 1
        if not bool(coverage.eq(1).all()):
            raise RCDEContractError("4D consensus tiled interior coverage is not exactly one")
        return result4.reshape(hq * wq, hr * wr), coverage


class DINO_RCDE_V1_2(nn.Module):
    """Complete V1.1 RCDE graph, isolated for V1.2 resource/E0 work."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_mix_logits = nn.Parameter(torch.zeros(4))
        self.shared_projection = nn.Parameter(torch.empty(OUTPUT_DIM, INPUT_DIM))
        self.query_u = nn.Parameter(torch.empty(OUTPUT_DIM, ADAPTER_RANK))
        self.query_v = nn.Parameter(torch.empty(ADAPTER_RANK, INPUT_DIM))
        self.reference_u = nn.Parameter(torch.empty(OUTPUT_DIM, ADAPTER_RANK))
        self.reference_v = nn.Parameter(torch.empty(ADAPTER_RANK, INPUT_DIM))
        self.beta = nn.Parameter(torch.zeros(()))
        self.consensus = Separable4DConsensus()
        self.signed_head = nn.Linear(SUMMARY_DIM, 1, bias=False)
        self.modulation_fc1 = nn.Linear(SUMMARY_DIM * 3, 16, bias=False)
        self.modulation_fc2 = nn.Linear(16, 1, bias=False)
        self.pair_fc1 = nn.Linear(2, 16, bias=False)
        self.pair_fc2 = nn.Linear(16, 1, bias=False)
        self.cost_volume_build_count = 0
        self.reset_parameters_frozen()
        if self.parameter_count() != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("RCDE parameter-count contract drift")

    def reset_parameters_frozen(self) -> None:
        """Version-independent SHA-counter initialization from the V1.1 contract."""

        with torch.no_grad():
            gaussian = _hash_gaussian("shared_projection", INPUT_DIM * OUTPUT_DIM)
            matrix = gaussian.reshape(INPUT_DIM, OUTPUT_DIM)
            q, r = torch.linalg.qr(matrix, mode="reduced")
            signs = torch.where(torch.diag(r) < 0.0, -torch.ones(OUTPUT_DIM), torch.ones(OUTPUT_DIM))
            q = q * signs[None, :]
            self.shared_projection.copy_(q.T.to(self.shared_projection.dtype))
            self.layer_mix_logits.zero_()
            self.beta.zero_()
            _copy_uniform(self.query_u, "query_u", -1.0e-3, 1.0e-3)
            _copy_uniform(self.query_v, "query_v", -1.0e-3, 1.0e-3)
            _copy_uniform(self.reference_u, "reference_u", -1.0e-3, 1.0e-3)
            _copy_uniform(self.reference_v, "reference_v", -1.0e-3, 1.0e-3)
            for name, parameter in (
                ("consensus.ref1.weight", self.consensus.ref1.weight),
                ("consensus.query1.weight", self.consensus.query1.weight),
                ("consensus.ref2.weight", self.consensus.ref2.weight),
                ("consensus.query2.weight", self.consensus.query2.weight),
                ("consensus.out.weight", self.consensus.out.weight),
                ("signed_head.weight", self.signed_head.weight),
                ("modulation_fc1.weight", self.modulation_fc1.weight),
                ("modulation_fc2.weight", self.modulation_fc2.weight),
                ("pair_fc1.weight", self.pair_fc1.weight),
                ("pair_fc2.weight", self.pair_fc2.weight),
            ):
                _copy_xavier(parameter, name)

    def parameter_ledger(self) -> dict[str, int]:
        return {
            "projection_and_adapters": (
                self.shared_projection.numel()
                + self.query_u.numel()
                + self.query_v.numel()
                + self.reference_u.numel()
                + self.reference_v.numel()
            ),
            "layer_mix": self.layer_mix_logits.numel(),
            "beta": self.beta.numel(),
            "four_d_consensus": sum(parameter.numel() for parameter in self.consensus.parameters()),
            "signed_head": sum(parameter.numel() for parameter in self.signed_head.parameters()),
            "modulation_head": (
                sum(parameter.numel() for parameter in self.modulation_fc1.parameters())
                + sum(parameter.numel() for parameter in self.modulation_fc2.parameters())
            ),
            "pair_head": (
                sum(parameter.numel() for parameter in self.pair_fc1.parameters())
                + sum(parameter.numel() for parameter in self.pair_fc2.parameters())
            ),
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def functional_parameter_groups(self) -> dict[str, tuple[str, ...]]:
        return {
            "layer_mix": ("layer_mix_logits",),
            "W0": ("shared_projection",),
            "Q_lowrank": ("query_u", "query_v"),
            "R_lowrank": ("reference_u", "reference_v"),
            "beta": ("beta",),
            "4D_consensus": tuple(f"consensus.{name}" for name, _ in self.consensus.named_parameters()),
            "ell": ("signed_head.weight",),
            "rho": ("modulation_fc1.weight", "modulation_fc2.weight"),
            "F": ("pair_fc1.weight", "pair_fc2.weight"),
        }

    def reset_cost_counter(self) -> None:
        self.cost_volume_build_count = 0

    def _mix(self, layers: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_mix_logits, dim=0).to(layers.dtype)
        return torch.einsum("l,lnd->nd", weights, layers)

    def _project(self, layers: torch.Tensor, mask: torch.Tensor, role: str) -> torch.Tensor:
        mixed = self._mix(layers)
        if role == "query":
            weight = self.shared_projection + self.query_u @ self.query_v
        elif role == "reference":
            weight = self.shared_projection + self.reference_u @ self.reference_v
        else:  # pragma: no cover - private caller uses a literal role
            raise RCDEContractError("unknown RCDE projection role")
        projected = F.linear(mixed, weight)
        valid_norms = torch.linalg.vector_norm(projected[mask.flatten()], dim=-1)
        if (
            not bool(torch.isfinite(projected).all())
            or not bool(valid_norms.gt(1.0e-12).all())
        ):
            raise RCDEContractError(
                f"{role} has a geometry-valid projected token with nonfinite or norm <=1e-12"
            )
        return F.normalize(projected, dim=-1, eps=1.0e-12)

    def build_cost(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
    ) -> torch.Tensor:
        query_layers, query_mask = _validate_layers(query_layers, query_mask, query_grid, "query")
        reference_layers, reference_mask = _validate_layers(
            reference_layers, reference_mask, reference_grid, "reference"
        )
        query = self._project(query_layers, query_mask, "query")
        reference = self._project(reference_layers, reference_mask, "reference")
        cost = query @ reference.T / COST_TEMPERATURE
        pair_mask = query_mask.flatten()[:, None] & reference_mask.flatten()[None, :]
        self.cost_volume_build_count += 1
        return cost * pair_mask.to(cost.dtype)

    def dual_softmax_dense(
        self, logits: torch.Tensor, query_mask: torch.Tensor, reference_mask: torch.Tensor
    ) -> Assignment:
        query_valid = query_mask.flatten()
        reference_valid = reference_mask.flatten()
        nq = int(query_valid.sum())
        nr = int(reference_valid.sum())
        if tuple(logits.shape) != (query_valid.numel(), reference_valid.numel()):
            raise RCDEContractError("dual-softmax logit/mask mismatch")
        pair_valid = query_valid[:, None] & reference_valid[None, :]
        masked = logits.masked_fill(~pair_valid, float("-inf"))
        row_dust = self.beta.to(logits.dtype) + math.log(nr)
        col_dust = self.beta.to(logits.dtype) + math.log(nq)
        row_logz = torch.logaddexp(torch.logsumexp(masked, dim=1), row_dust.expand(masked.shape[0]))
        col_logz = torch.logaddexp(torch.logsumexp(masked, dim=0), col_dust.expand(masked.shape[1]))
        Q = torch.exp(masked - row_logz[:, None]).masked_fill(~pair_valid, 0.0)
        G = torch.exp(masked - col_logz[None, :]).masked_fill(~pair_valid, 0.0)
        u = torch.exp(row_dust - row_logz).masked_fill(~query_valid, 0.0)
        v = torch.exp(col_dust - col_logz).masked_fill(~reference_valid, 0.0)
        return Assignment(Q=Q, G=G, P=Q * G, u=u, v=v)

    def dual_softmax_streaming(
        self,
        logits: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        reference_chunk_size: int,
    ) -> Assignment:
        """Reference-axis global reduction used only by small closure tests.

        This helper accepts a resident logit matrix and intentionally retains
        Q/G/P.  Formal worst-shape execution uses
        :meth:`decode_candidate_true_streaming`, which starts from projected
        tokens, generates haloed cross-cost tiles, and never allocates the full
        cost, logits, Q, G, or P matrices.
        """

        if reference_chunk_size <= 0:
            raise RCDEContractError("reference chunk size must be positive")
        query_valid = query_mask.flatten()
        reference_valid = reference_mask.flatten()
        if tuple(logits.shape) != (query_valid.numel(), reference_valid.numel()):
            raise RCDEContractError("streaming dual-softmax logit/mask mismatch")
        nq = int(query_valid.sum())
        nr = int(reference_valid.sum())
        row_dust = self.beta.to(logits.dtype) + math.log(nr)
        col_dust = self.beta.to(logits.dtype) + math.log(nq)
        # Accumulate data-only chunk log-partitions first, then introduce the
        # dustbin exactly once.  The prior implementation seeded the running
        # accumulator with dust, so dust participated in one float32 logaddexp
        # per reference chunk whereas dense execution adds it only after the
        # complete row logsumexp.  This preserves the same mathematical
        # streaming formula while removing avoidable, shape-dependent rounding.
        row_data_logz = logits.new_full((logits.shape[0],), float("-inf"))
        col_logz = logits.new_full((logits.shape[1],), float("-inf"))
        for start in range(0, logits.shape[1], reference_chunk_size):
            stop = min(logits.shape[1], start + reference_chunk_size)
            chunk_valid = query_valid[:, None] & reference_valid[None, start:stop]
            chunk = logits[:, start:stop].masked_fill(~chunk_valid, float("-inf"))
            row_data_logz = torch.logaddexp(
                row_data_logz, torch.logsumexp(chunk, dim=1)
            )
            col_logz[start:stop] = torch.logaddexp(
                torch.logsumexp(chunk, dim=0), col_dust.expand(stop - start)
            )
        row_logz = torch.logaddexp(
            row_data_logz, row_dust.expand(logits.shape[0])
        )
        Q = torch.zeros_like(logits)
        G = torch.zeros_like(logits)
        for start in range(0, logits.shape[1], reference_chunk_size):
            stop = min(logits.shape[1], start + reference_chunk_size)
            chunk_valid = query_valid[:, None] & reference_valid[None, start:stop]
            chunk = logits[:, start:stop].masked_fill(~chunk_valid, float("-inf"))
            Q[:, start:stop] = torch.exp(chunk - row_logz[:, None]).masked_fill(~chunk_valid, 0.0)
            G[:, start:stop] = torch.exp(chunk - col_logz[None, start:stop]).masked_fill(~chunk_valid, 0.0)
        u = torch.exp(row_dust - row_logz).masked_fill(~query_valid, 0.0)
        v = torch.exp(col_dust - col_logz).masked_fill(~reference_valid, 0.0)
        return Assignment(Q=Q, G=G, P=Q * G, u=u, v=v)

    def _streamed_consensus_tiles(
        self,
        query_projected: torch.Tensor,
        reference_projected: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        *,
        query_tile_rows: int,
        reference_tile_rows: int,
        zero_cost: bool = False,
    ) -> Iterator[tuple[slice, slice, torch.Tensor, int]]:
        """Yield disjoint row-band interiors of the four-dimensional logits.

        Both x axes span their complete grids.  Each y band is expanded by the
        exact radius-two halo before the frozen consensus graph is evaluated.
        Therefore every yielded cell has the same receptive field as dense
        execution, while the largest resident cross-cost is only one expanded
        query band by one expanded reference band.
        """

        hq, wq = query_grid
        hr, wr = reference_grid
        if query_tile_rows <= 0 or reference_tile_rows <= 0:
            raise RCDEContractError("streaming row-band sizes must be positive")
        query4 = query_projected.reshape(hq, wq, OUTPUT_DIM)
        reference4 = reference_projected.reshape(hr, wr, OUTPUT_DIM)
        tile_ordinal = 0
        for qy0 in range(0, hq, query_tile_rows):
            qy1 = min(hq, qy0 + query_tile_rows)
            eqy0, eqy1 = max(0, qy0 - 2), min(hq, qy1 + 2)
            sub_q_grid = (eqy1 - eqy0, wq)
            sub_q_mask = query_mask[eqy0:eqy1, :]
            sub_q = query4[eqy0:eqy1, :, :].reshape(-1, OUTPUT_DIM)
            q_interior = slice((qy0 - eqy0) * wq, (qy1 - eqy0) * wq)
            q_global = slice(qy0 * wq, qy1 * wq)
            for ry0 in range(0, hr, reference_tile_rows):
                ry1 = min(hr, ry0 + reference_tile_rows)
                ery0, ery1 = max(0, ry0 - 2), min(hr, ry1 + 2)
                sub_r_grid = (ery1 - ery0, wr)
                sub_r_mask = reference_mask[ery0:ery1, :]
                sub_r = reference4[ery0:ery1, :, :].reshape(-1, OUTPUT_DIM)
                if zero_cost:
                    sub_cost = sub_q.new_zeros((sub_q.shape[0], sub_r.shape[0]))
                else:
                    sub_cost = sub_q @ sub_r.T / COST_TEMPERATURE
                sub_logits = self.consensus(
                    sub_cost,
                    sub_q_grid,
                    sub_r_grid,
                    sub_q_mask,
                    sub_r_mask,
                )
                r_interior = slice((ry0 - ery0) * wr, (ry1 - ery0) * wr)
                r_global = slice(ry0 * wr, ry1 * wr)
                yield q_global, r_global, sub_logits[q_interior, r_interior], int(sub_cost.numel())
                tile_ordinal += 1

    def _summarize_true_streaming(
        self,
        tile_factory: Callable[[], Iterator[tuple[slice, slice, torch.Tensor, int]]],
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        *,
        materialize_assignment_for_audit: bool,
        emit_matrix_audit: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Mapping[str, str],
        Sequence[Mapping[str, float | int]],
        Assignment | None,
        tuple[int, int, int],
        int,
    ]:
        """Exact global dual-softmax plus three-pass streamed summaries.

        Pass 1 computes global row/column log normalizers.  Pass 2 computes P
        mass, column load, first moments, and canonical row-major matrix hashes.
        Pass 3 computes centered second moments/covariance and inverse column
        crowding, which cannot be evaluated before the global column load is
        known.  Full Q/G/P are optional and forbidden on the formal worst path.
        """

        q_valid = query_mask.flatten()
        r_valid = reference_mask.flatten()
        nq_total, nr_total = q_valid.numel(), r_valid.numel()
        nq, nr = int(q_valid.sum()), int(r_valid.sum())
        dtype = self.beta.dtype
        device = self.beta.device
        row_dust = self.beta.to(dtype) + math.log(nr)
        col_dust = self.beta.to(dtype) + math.log(nq)
        max_resident = 0
        tile_count = 0

        # Pass 1: exact global log normalizers, never tile-local softmax.
        row_parts: dict[tuple[int, int], torch.Tensor] = {}
        col_parts: dict[tuple[int, int], torch.Tensor] = {}
        for qs, rs, logits, resident in tile_factory():
            tile_count += 1
            max_resident = max(max_resident, resident)
            valid = q_valid[qs, None] & r_valid[None, rs]
            masked = logits.masked_fill(~valid, float("-inf"))
            qkey = (int(qs.start), int(qs.stop))
            rkey = (int(rs.start), int(rs.stop))
            row_term = torch.logsumexp(masked, dim=1)
            col_term = torch.logsumexp(masked, dim=0)
            row_parts[qkey] = (
                row_term if qkey not in row_parts else torch.logaddexp(row_parts[qkey], row_term)
            )
            col_parts[rkey] = (
                col_term if rkey not in col_parts else torch.logaddexp(col_parts[rkey], col_term)
            )

        def ordered_cat(parts: Mapping[tuple[int, int], torch.Tensor], total: int) -> torch.Tensor:
            keys = sorted(parts)
            if not keys or keys[0][0] != 0 or keys[-1][1] != total or any(
                left[1] != right[0] for left, right in zip(keys, keys[1:])
            ):
                raise RCDEContractError("streamed band partition is not exact")
            return torch.cat([parts[key] for key in keys], dim=0)

        row_logz_raw = ordered_cat(row_parts, nq_total)
        col_logz_raw = ordered_cat(col_parts, nr_total)
        row_logz = torch.logaddexp(row_logz_raw, row_dust.expand_as(row_logz_raw))
        col_logz = torch.logaddexp(col_logz_raw, col_dust.expand_as(col_logz_raw))
        u = torch.exp(row_dust - row_logz).masked_fill(~q_valid, 0.0)
        v = torch.exp(col_dust - col_logz).masked_fill(~r_valid, 0.0)

        coords = _normalized_coordinates(reference_grid, reference_mask, device=device, dtype=dtype)
        if materialize_assignment_for_audit and not emit_matrix_audit:
            raise RCDEContractError("assignment materialization requires matrix audit")
        hashes = {name: hashlib.sha256() for name in ("Q", "G", "P")} if emit_matrix_audit else {}
        for name, digest in hashes.items():
            digest.update(
                f"RCDE_MATRIX_V1_2\0{name}\0{dtype}\0{nq_total}\0{nr_total}\0".encode("ascii")
            )
        audit_indices = (
            (0, (nq_total * nr_total) // 2, nq_total * nr_total - 1)
            if emit_matrix_audit else ()
        )
        audit_values: dict[int, dict[str, float | int]] = {}
        materialized = {
            name: torch.zeros((nq_total, nr_total), dtype=dtype, device=device)
            for name in ("Q", "G", "P")
        } if materialize_assignment_for_audit else None

        # Pass 2 is ordered by contiguous full-width query bands.  The CPU band
        # buffers make each SHA256 exactly canonical row-major and remain much
        # smaller than the full cross-product.
        current_qs: slice | None = None
        band_buffers: dict[str, torch.Tensor] | None = None

        def finish_band(qs: slice | None, buffers: dict[str, torch.Tensor] | None) -> None:
            if qs is None or buffers is None:
                return
            for name, value in buffers.items():
                contiguous = value.contiguous()
                hashes[name].update(contiguous.numpy().tobytes(order="C"))
                if materialized is not None:
                    materialized[name][qs] = contiguous.to(device)
            q_start, q_stop = int(qs.start), int(qs.stop)
            for flat_index in audit_indices:
                qi, ri = divmod(flat_index, nr_total)
                if q_start <= qi < q_stop:
                    local = qi - q_start
                    audit_values[flat_index] = {
                        "flat_index": flat_index,
                        "query_index": qi,
                        "reference_index": ri,
                        **{name: float(value[local, ri]) for name, value in buffers.items()},
                    }

        pass2_tile_count = 0
        q_accumulators: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        column_load_parts: dict[tuple[int, int], torch.Tensor] = {}
        for qs, rs, logits, resident in tile_factory():
            pass2_tile_count += 1
            max_resident = max(max_resident, resident)
            qkey = (int(qs.start), int(qs.stop))
            rkey = (int(rs.start), int(rs.stop))
            if emit_matrix_audit and (
                current_qs is None
                or (qs.start, qs.stop) != (current_qs.start, current_qs.stop)
            ):
                finish_band(current_qs, band_buffers)
                current_qs = qs
                rows = int(qs.stop) - int(qs.start)
                band_buffers = {
                    name: torch.zeros((rows, nr_total), dtype=dtype, device="cpu")
                    for name in ("Q", "G", "P")
                }
            valid = q_valid[qs, None] & r_valid[None, rs]
            masked = logits.masked_fill(~valid, float("-inf"))
            Q = torch.exp(masked - row_logz[qs, None]).masked_fill(~valid, 0.0)
            G = torch.exp(masked - col_logz[None, rs]).masked_fill(~valid, 0.0)
            P = Q * G
            if emit_matrix_audit:
                assert band_buffers is not None
                band_buffers["Q"][:, rs] = Q.detach().cpu()
                band_buffers["G"][:, rs] = G.detach().cpu()
                band_buffers["P"][:, rs] = P.detach().cpu()
            safe_logits = logits.masked_fill(~valid, 0.0)
            terms = {
                "reciprocal": P.sum(dim=1),
                "logit": (P * safe_logits).sum(dim=1),
                "x": (P * coords[rs, 0][None, :]).sum(dim=1),
                "y": (P * coords[rs, 1][None, :]).sum(dim=1),
                "cycle": (P * G).sum(dim=1),
                "entropy": -(
                Q.clamp_min(1.0e-30) * Q.clamp_min(1.0e-30).log()
                ).masked_fill(~valid, 0.0).sum(dim=1),
            }
            if qkey not in q_accumulators:
                q_accumulators[qkey] = terms
                prior_top = P.new_zeros((P.shape[0], 2))
            else:
                prior = q_accumulators[qkey]
                for name, term in terms.items():
                    prior[name] = prior[name] + term
                prior_top = torch.stack((prior["top1"], prior["top2"]), dim=1)
            column_term = P.sum(dim=0)
            column_load_parts[rkey] = (
                column_term if rkey not in column_load_parts
                else column_load_parts[rkey] + column_term
            )
            merged = torch.cat((prior_top, P), dim=1)
            best = torch.topk(merged, k=2, dim=1).values
            q_accumulators[qkey]["top1"] = best[:, 0]
            q_accumulators[qkey]["top2"] = best[:, 1]
        if emit_matrix_audit:
            finish_band(current_qs, band_buffers)
        if emit_matrix_audit and set(audit_values) != set(audit_indices):
            raise RCDEContractError("streamed matrix audit slices are incomplete")

        def cat_q_field(name: str) -> torch.Tensor:
            return ordered_cat({key: value[name] for key, value in q_accumulators.items()}, nq_total)

        reciprocal = cat_q_field("reciprocal")
        logit_numerator = cat_q_field("logit")
        x_numerator = cat_q_field("x")
        y_numerator = cat_q_field("y")
        cycle_numerator = cat_q_field("cycle")
        entropy_q = cat_q_field("entropy")
        top1 = cat_q_field("top1")
        top2 = cat_q_field("top2")
        column_load = ordered_cat(column_load_parts, nr_total)

        denominator = reciprocal.clamp_min(1.0e-12)
        assigned_mean_raw = logit_numerator / denominator
        mean_x = x_numerator / denominator
        mean_y = y_numerator / denominator

        # Pass 3: centered second moments and the column-load-dependent field.
        pass3_tile_count = 0
        q_second: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for qs, rs, logits, resident in tile_factory():
            pass3_tile_count += 1
            max_resident = max(max_resident, resident)
            qkey = (int(qs.start), int(qs.stop))
            valid = q_valid[qs, None] & r_valid[None, rs]
            masked = logits.masked_fill(~valid, float("-inf"))
            Q = torch.exp(masked - row_logz[qs, None]).masked_fill(~valid, 0.0)
            G = torch.exp(masked - col_logz[None, rs]).masked_fill(~valid, 0.0)
            P = Q * G
            weights = P / denominator[qs, None]
            safe_logits = logits.masked_fill(~valid, 0.0)
            dx = coords[rs, 0][None, :] - mean_x[qs, None]
            dy = coords[rs, 1][None, :] - mean_y[qs, None]
            terms = {
                "assigned_var": (
                    weights * (safe_logits - assigned_mean_raw[qs, None]).square()
                ).sum(dim=1),
                "var_x": (weights * dx.square()).sum(dim=1),
                "var_y": (weights * dy.square()).sum(dim=1),
                "cov_xy": (weights * dx * dy).sum(dim=1),
                "inverse_crowding": (
                    weights / (1.0 + column_load[rs][None, :])
                ).sum(dim=1),
            }
            if qkey not in q_second:
                q_second[qkey] = terms
            else:
                for name, term in terms.items():
                    q_second[qkey][name] = q_second[qkey][name] + term

        def cat_second(name: str) -> torch.Tensor:
            return ordered_cat({key: value[name] for key, value in q_second.items()}, nq_total)

        assigned_var_num = cat_second("assigned_var")
        var_x_num = cat_second("var_x")
        var_y_num = cat_second("var_y")
        cov_xy_num = cat_second("cov_xy")
        inverse_crowding = cat_second("inverse_crowding")

        entropy = entropy_q - u.clamp_min(1.0e-30) * u.clamp_min(1.0e-30).log()
        entropy = entropy / math.log(nr + 1)
        cycle_return = cycle_numerator / denominator
        assigned_mean = assigned_mean_raw / (1.0 + assigned_mean_raw.abs())
        assigned_var = assigned_var_num / (1.0 + assigned_var_num)
        hq, wq = query_grid
        q_coords = _normalized_coordinates(query_grid, query_mask, device=device, dtype=dtype)
        displacement = torch.stack((mean_x, mean_y), dim=1) - q_coords
        displacement_grid = displacement.reshape(hq, wq, 2)
        valid_grid = q_valid.reshape(hq, wq)
        yy, xx = torch.meshgrid(
            torch.arange(hq, device=device), torch.arange(wq, device=device), indexing="ij"
        )
        agreement_terms = []
        agreement_validity = []
        for dy_q, dx_q in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            inside = (yy + dy_q >= 0) & (yy + dy_q < hq) & (xx + dx_q >= 0) & (xx + dx_q < wq)
            neighbour_valid = torch.roll(valid_grid, shifts=(-dy_q, -dx_q), dims=(0, 1))
            both = valid_grid & neighbour_valid & inside
            neighbour = torch.roll(displacement_grid, shifts=(-dy_q, -dx_q), dims=(0, 1))
            score = torch.exp(-(displacement_grid - neighbour).square().sum(dim=-1))
            agreement_terms.append(score * both.to(dtype))
            agreement_validity.append(both.to(dtype))
        agreement_sum = torch.stack(agreement_terms, dim=0).sum(dim=0)
        agreement_count = torch.stack(agreement_validity, dim=0).sum(dim=0)
        agreement = (agreement_sum / agreement_count.clamp_min(1.0)).flatten()
        summary = torch.stack(
            (
                1.0 - u, u, top1, top1 - top2, entropy, reciprocal,
                assigned_mean, assigned_var, cycle_return, mean_x, mean_y,
                var_x_num, var_y_num, cov_xy_num, agreement, inverse_crowding,
            ),
            dim=1,
        ) * q_valid[:, None].to(dtype)
        assignment = None
        if materialized is not None:
            assignment = Assignment(
                Q=materialized["Q"], G=materialized["G"], P=materialized["P"], u=u, v=v
            )
        v_digest = hashlib.sha256() if emit_matrix_audit else None
        if v_digest is not None:
            v_digest.update(f"RCDE_VECTOR_V1_2\0v\0{dtype}\0{nr_total}\0".encode("ascii"))
            v_digest.update(v.detach().cpu().contiguous().numpy().tobytes(order="C"))
        finalized_hashes = {name: digest.hexdigest() for name, digest in hashes.items()}
        if v_digest is not None:
            finalized_hashes["v"] = v_digest.hexdigest()
        finalized_slices = []
        for index in audit_indices:
            item = dict(audit_values[index])
            item["v"] = float(v[item["reference_index"]].detach().cpu())
            finalized_slices.append(item)
        if not (tile_count == pass2_tile_count == pass3_tile_count):
            raise RCDEContractError("streamed summary pass tile-count mismatch")
        return (
            summary,
            u,
            v,
            finalized_hashes,
            tuple(finalized_slices),
            assignment,
            (tile_count, pass2_tile_count, pass3_tile_count),
            max_resident,
        )

    def _analytic_zero_cost_summary(
        self,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closed-form null summary for the bias-free zero-cost decoder.

        Every consensus convolution is bias-free and GELU(0)=0, so a zero
        cross-cost produces exactly zero logits for every parameter value.
        Running the full tiled consensus six times therefore cannot add a
        gradient to the consensus/projection parameters.  The null assignment
        still depends on trainable ``beta`` and geometry; this implementation
        preserves that dependency analytically without changing the model.
        """

        query_mask = _canonical_mask(query_mask, query_grid, "query")
        reference_mask = _canonical_mask(reference_mask, reference_grid, "reference")
        q_valid = query_mask.flatten()
        r_valid = reference_mask.flatten()
        nq_total, nr_total = q_valid.numel(), r_valid.numel()
        nq, nr = int(q_valid.sum()), int(r_valid.sum())
        dtype = self.beta.dtype
        device = self.beta.device

        # Use the same log-domain operations as dual_softmax_dense instead of
        # simplifying to sigmoid(beta).  This keeps FP32 values and beta
        # derivatives aligned with the audited decoder up to reduction-order
        # roundoff.
        zero = self.beta.to(dtype) * 0.0
        row_dust = self.beta.to(dtype) + math.log(nr)
        col_dust = self.beta.to(dtype) + math.log(nq)
        row_logz = torch.logaddexp(zero + math.log(nr), row_dust)
        col_logz = torch.logaddexp(zero + math.log(nq), col_dust)
        q_mass = torch.exp(-row_logz)
        g_mass = torch.exp(-col_logz)
        u_scalar = torch.exp(row_dust - row_logz)
        v_scalar = torch.exp(col_dust - col_logz)
        p_mass = q_mass * g_mass

        u = u_scalar.expand(nq_total).clone().masked_fill(~q_valid, 0.0)
        v = v_scalar.expand(nr_total).clone().masked_fill(~r_valid, 0.0)
        reciprocal_scalar = p_mass * nr
        top1_scalar = p_mass
        top2_scalar = p_mass if nr >= 2 else zero
        entropy_scalar = (
            -(nr * q_mass * q_mass.clamp_min(1.0e-30).log())
            - u_scalar * u_scalar.clamp_min(1.0e-30).log()
        ) / math.log(nr + 1)

        r_coords = _normalized_coordinates(
            reference_grid, reference_mask, device=device, dtype=dtype
        )
        valid_r_coords = r_coords[r_valid]
        mean_xy = valid_r_coords.mean(dim=0)
        centered = valid_r_coords - mean_xy[None, :]
        var_x = centered[:, 0].square().mean()
        var_y = centered[:, 1].square().mean()
        cov_xy = (centered[:, 0] * centered[:, 1]).mean()

        hq, wq = query_grid
        q_coords = _normalized_coordinates(
            query_grid, query_mask, device=device, dtype=dtype
        )
        displacement = mean_xy[None, :] - q_coords
        displacement_grid = displacement.reshape(hq, wq, 2)
        valid_grid = q_valid.reshape(hq, wq)
        yy, xx = torch.meshgrid(
            torch.arange(hq, device=device),
            torch.arange(wq, device=device),
            indexing="ij",
        )
        agreement_terms = []
        agreement_validity = []
        for dy_q, dx_q in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            inside = (
                (yy + dy_q >= 0) & (yy + dy_q < hq)
                & (xx + dx_q >= 0) & (xx + dx_q < wq)
            )
            neighbour_valid = torch.roll(valid_grid, shifts=(-dy_q, -dx_q), dims=(0, 1))
            both = valid_grid & neighbour_valid & inside
            neighbour = torch.roll(
                displacement_grid, shifts=(-dy_q, -dx_q), dims=(0, 1)
            )
            score = torch.exp(-(displacement_grid - neighbour).square().sum(dim=-1))
            agreement_terms.append(score * both.to(dtype))
            agreement_validity.append(both.to(dtype))
        agreement_sum = torch.stack(agreement_terms, dim=0).sum(dim=0)
        agreement_count = torch.stack(agreement_validity, dim=0).sum(dim=0)
        agreement = (agreement_sum / agreement_count.clamp_min(1.0)).flatten()

        column_load = p_mass * nq
        inverse_crowding = 1.0 / (1.0 + column_load)
        fields = (
            1.0 - u_scalar,
            u_scalar,
            top1_scalar,
            top1_scalar - top2_scalar,
            entropy_scalar,
            reciprocal_scalar,
            zero,
            zero,
            g_mass,
            mean_xy[0],
            mean_xy[1],
            var_x,
            var_y,
            cov_xy,
        )
        summary = torch.stack(
            tuple(value.expand(nq_total) for value in fields)
            + (agreement, inverse_crowding.expand(nq_total)),
            dim=1,
        )
        summary = summary * q_valid[:, None].to(dtype)
        return summary, u, v

    def decode_candidate_true_streaming_fast(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        *,
        query_tile_rows: int = 64,
        reference_tile_rows: int = 64,
    ) -> StreamedCandidateEvidence:
        """Execution-only fast path with unchanged candidate evidence.

        The real consensus tile graph is constructed once and reused by the
        three global summary reductions.  The analytic zero-cost control above
        replaces a mathematically constant consensus execution.  Training and
        bulk replay omit Q/G/P CPU copies and hashes because those values do
        not enter any score or loss; the original audited method remains the
        canonical path for sampled hash/slice qualification.
        """

        query_layers, query_mask = _validate_layers(
            query_layers, query_mask, query_grid, "query"
        )
        reference_layers, reference_mask = _validate_layers(
            reference_layers, reference_mask, reference_grid, "reference"
        )
        query = self._project(query_layers, query_mask, "query")
        reference = self._project(reference_layers, reference_mask, "reference")
        self.cost_volume_build_count += 1
        cached_tiles = tuple(
            self._streamed_consensus_tiles(
                query,
                reference,
                query_mask,
                reference_mask,
                query_grid,
                reference_grid,
                query_tile_rows=query_tile_rows,
                reference_tile_rows=reference_tile_rows,
            )
        )
        if not cached_tiles:
            raise RCDEContractError("fast streaming path produced no consensus tile")

        def real_tiles() -> Iterator[tuple[slice, slice, torch.Tensor, int]]:
            return iter(cached_tiles)

        raw, u, v, hashes, slices, assignment, pass_tile_counts, max_resident = (
            self._summarize_true_streaming(
                real_tiles,
                query_mask,
                reference_mask,
                query_grid,
                reference_grid,
                materialize_assignment_for_audit=False,
                emit_matrix_audit=False,
            )
        )
        null, _, _ = self._analytic_zero_cost_summary(
            query_mask, reference_mask, query_grid, reference_grid
        )
        if hashes or slices or assignment is not None:
            raise RCDEContractError("fast non-audit path leaked audit materialization")
        return StreamedCandidateEvidence(
            raw_summary=raw,
            null_summary=null,
            relational=raw - null,
            u=u,
            v=v,
            matrix_sha256=hashes,
            matrix_audit_slices=slices,
            assignment=assignment,
            logical_cost_volume_build_count=1,
            consensus_tile_count_per_pass=pass_tile_counts[0],
            real_pass_tile_counts=pass_tile_counts,
            null_pass_tile_counts=(0, 0, 0),
            maximum_resident_cost_elements=max_resident,
            summary_pass_count=3,
            real_consensus_evaluation_count=len(cached_tiles),
            null_consensus_evaluation_count=0,
            full_consensus_logits_resident=True,
            resident_consensus_logit_elements=sum(
                int(logits.numel()) for _, _, logits, _ in cached_tiles
            ),
            matrix_audit_enabled=False,
            analytic_null=True,
        )

    def decode_candidate_true_streaming(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        *,
        query_tile_rows: int = 8,
        reference_tile_rows: int = 8,
        materialize_assignment_for_audit: bool = False,
    ) -> StreamedCandidateEvidence:
        """End-to-end token-to-summary streaming for the formal worst path."""

        query_layers, query_mask = _validate_layers(query_layers, query_mask, query_grid, "query")
        reference_layers, reference_mask = _validate_layers(
            reference_layers, reference_mask, reference_grid, "reference"
        )
        query = self._project(query_layers, query_mask, "query")
        reference = self._project(reference_layers, reference_mask, "reference")
        self.cost_volume_build_count += 1

        def real_tiles() -> Iterator[tuple[slice, slice, torch.Tensor, int]]:
            return self._streamed_consensus_tiles(
                query, reference, query_mask, reference_mask, query_grid, reference_grid,
                query_tile_rows=query_tile_rows,
                reference_tile_rows=reference_tile_rows,
            )

        raw, u, v, hashes, slices, assignment, pass_tile_counts, max_resident = (
            self._summarize_true_streaming(
                real_tiles, query_mask, reference_mask, query_grid, reference_grid,
                materialize_assignment_for_audit=materialize_assignment_for_audit,
            )
        )

        def null_tiles() -> Iterator[tuple[slice, slice, torch.Tensor, int]]:
            return self._streamed_consensus_tiles(
                query, reference, query_mask, reference_mask, query_grid, reference_grid,
                query_tile_rows=query_tile_rows,
                reference_tile_rows=reference_tile_rows,
                zero_cost=True,
            )

        null, _, _, _, _, _, null_pass_tile_counts, null_max_resident = self._summarize_true_streaming(
            null_tiles, query_mask, reference_mask, query_grid, reference_grid,
            materialize_assignment_for_audit=False,
        )
        return StreamedCandidateEvidence(
            raw_summary=raw,
            null_summary=null,
            relational=raw - null,
            u=u,
            v=v,
            matrix_sha256=hashes,
            matrix_audit_slices=slices,
            assignment=assignment,
            logical_cost_volume_build_count=1,
            consensus_tile_count_per_pass=pass_tile_counts[0],
            real_pass_tile_counts=pass_tile_counts,
            null_pass_tile_counts=null_pass_tile_counts,
            maximum_resident_cost_elements=max(max_resident, null_max_resident),
            summary_pass_count=3,
            real_consensus_evaluation_count=sum(pass_tile_counts),
            null_consensus_evaluation_count=sum(null_pass_tile_counts),
            full_consensus_logits_resident=False,
            resident_consensus_logit_elements=0,
            matrix_audit_enabled=True,
            analytic_null=False,
        )

    @staticmethod
    def _summary(
        logits: torch.Tensor,
        assignment: Assignment,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
    ) -> torch.Tensor:
        q_valid = query_mask.flatten()
        r_valid = reference_mask.flatten()
        pair_valid = q_valid[:, None] & r_valid[None, :]
        Q, G, P, u = assignment.Q, assignment.G, assignment.P, assignment.u
        reciprocal = P.sum(dim=1)
        weights = P / reciprocal[:, None].clamp_min(1.0e-12)
        weights = weights.masked_fill(~pair_valid, 0.0)
        max_assignment, second_assignment = _masked_top2(P, r_valid)
        max_assignment = max_assignment.masked_fill(~q_valid, 0.0)
        second_assignment = second_assignment.masked_fill(~q_valid, 0.0)
        row_distribution = torch.cat((Q, u[:, None]), dim=1).clamp_min(1.0e-30)
        entropy = -(row_distribution * row_distribution.log()).sum(dim=1)
        entropy = entropy / math.log(int(r_valid.sum()) + 1)
        entropy = entropy.masked_fill(~q_valid, 0.0)

        assigned_mean = (weights * logits.masked_fill(~pair_valid, 0.0)).sum(dim=1)
        assigned_var = (
            weights * (logits.masked_fill(~pair_valid, 0.0) - assigned_mean[:, None]).square()
        ).sum(dim=1)
        # Map unbounded logit moments to stable, dimensionless fields while
        # preserving sign for the mean and zero for the null cost volume.
        assigned_mean = assigned_mean / (1.0 + assigned_mean.abs())
        assigned_var = assigned_var / (1.0 + assigned_var)

        coords = _normalized_coordinates(
            reference_grid, reference_mask, device=logits.device, dtype=logits.dtype
        )
        x = coords[:, 0]
        y = coords[:, 1]
        mean_x = (weights * x[None, :]).sum(dim=1)
        mean_y = (weights * y[None, :]).sum(dim=1)
        dx = x[None, :] - mean_x[:, None]
        dy = y[None, :] - mean_y[:, None]
        var_x = (weights * dx.square()).sum(dim=1)
        var_y = (weights * dy.square()).sum(dim=1)
        cov_xy = (weights * dx * dy).sum(dim=1)

        # A second reciprocal step: expectation of the reverse probability
        # under the normalized reciprocal assignment for this query patch.
        cycle_return = (weights * G).sum(dim=1)
        column_load = P.sum(dim=0)
        inverse_crowding = (weights / (1.0 + column_load[None, :])).sum(dim=1)

        hq, wq = query_grid
        q_coords = _normalized_coordinates(
            query_grid, query_mask, device=logits.device, dtype=logits.dtype
        )
        displacement = torch.stack((mean_x, mean_y), dim=1) - q_coords
        displacement_grid = displacement.reshape(hq, wq, 2)
        valid_grid = q_valid.reshape(hq, wq)
        agreement_sum = logits.new_zeros((hq, wq))
        agreement_count = logits.new_zeros((hq, wq))
        for dy_q, dx_q in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            source_y = slice(max(0, -dy_q), min(hq, hq - dy_q))
            source_x = slice(max(0, -dx_q), min(wq, wq - dx_q))
            target_y = slice(max(0, dy_q), min(hq, hq + dy_q))
            target_x = slice(max(0, dx_q), min(wq, wq + dx_q))
            both = valid_grid[source_y, source_x] & valid_grid[target_y, target_x]
            difference = displacement_grid[source_y, source_x] - displacement_grid[target_y, target_x]
            score = torch.exp(-difference.square().sum(dim=-1)) * both.to(logits.dtype)
            agreement_sum[source_y, source_x] += score
            agreement_count[source_y, source_x] += both.to(logits.dtype)
        agreement = (agreement_sum / agreement_count.clamp_min(1.0)).flatten()

        summary = torch.stack(
            (
                1.0 - u,
                u,
                max_assignment,
                max_assignment - second_assignment,
                entropy,
                reciprocal,
                assigned_mean,
                assigned_var,
                cycle_return,
                mean_x,
                mean_y,
                var_x,
                var_y,
                cov_xy,
                agreement,
                inverse_crowding,
            ),
            dim=1,
        )
        return summary * q_valid[:, None].to(summary.dtype)

    def decode_candidate(
        self,
        query_layers: torch.Tensor,
        reference_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        *,
        streaming_chunk_size: int | None = None,
        consensus_tile_shape: tuple[int, int, int, int] | None = None,
    ) -> CandidateEvidence:
        query_layers, query_mask = _validate_layers(query_layers, query_mask, query_grid, "query")
        reference_layers, reference_mask = _validate_layers(
            reference_layers, reference_mask, reference_grid, "reference"
        )
        cost = self.build_cost(
            query_layers,
            reference_layers,
            query_mask,
            reference_mask,
            query_grid,
            reference_grid,
        )
        if consensus_tile_shape is None:
            logits = self.consensus(cost, query_grid, reference_grid, query_mask, reference_mask)
        else:
            logits, _ = self.consensus.forward_tiled(
                cost,
                query_grid,
                reference_grid,
                query_mask,
                reference_mask,
                tile_shape=consensus_tile_shape,
            )
        if streaming_chunk_size is None:
            assignment = self.dual_softmax_dense(logits, query_mask, reference_mask)
        else:
            assignment = self.dual_softmax_streaming(
                logits,
                query_mask,
                reference_mask,
                reference_chunk_size=streaming_chunk_size,
            )
        raw = self._summary(
            logits, assignment, query_mask, reference_mask, query_grid, reference_grid
        )
        null_logits = torch.zeros_like(logits)
        if streaming_chunk_size is None:
            null_assignment = self.dual_softmax_dense(null_logits, query_mask, reference_mask)
        else:
            null_assignment = self.dual_softmax_streaming(
                null_logits,
                query_mask,
                reference_mask,
                reference_chunk_size=streaming_chunk_size,
            )
        null = self._summary(
            null_logits,
            null_assignment,
            query_mask,
            reference_mask,
            query_grid,
            reference_grid,
        )
        return CandidateEvidence(
            cost=cost,
            logits=logits,
            assignment=assignment,
            raw_summary=raw,
            null_summary=null,
            relational=raw - null,
        )

    def decode_zero_cost_control(
        self,
        query_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_grid: tuple[int, int],
        *,
        streaming_chunk_size: int | None = None,
        consensus_tile_shape: tuple[int, int, int, int] | None = None,
    ) -> CandidateEvidence:
        """Execute the explicit NO_REF control after the projection boundary.

        Normal model inputs are forbidden from containing zero-norm valid DINO
        or projected tokens.  Consequently NO_REF cannot be implemented by
        feeding synthetic zero descriptors through the deployed adapter.  This
        control injects the contractually intended zero *cross-cost* directly
        at the decoder boundary and proves that shape/geometry alone yields no
        relational evidence.
        """

        query_mask = _canonical_mask(query_mask, query_grid, "query")
        reference_mask = _canonical_mask(reference_mask, reference_grid, "reference")
        cost = self.beta.new_zeros((query_mask.numel(), reference_mask.numel()))
        if consensus_tile_shape is None:
            logits = self.consensus(cost, query_grid, reference_grid, query_mask, reference_mask)
        else:
            logits, _ = self.consensus.forward_tiled(
                cost,
                query_grid,
                reference_grid,
                query_mask,
                reference_mask,
                tile_shape=consensus_tile_shape,
            )
        if streaming_chunk_size is None:
            assignment = self.dual_softmax_dense(logits, query_mask, reference_mask)
        else:
            assignment = self.dual_softmax_streaming(
                logits, query_mask, reference_mask,
                reference_chunk_size=streaming_chunk_size,
            )
        raw = self._summary(
            logits, assignment, query_mask, reference_mask, query_grid, reference_grid
        )
        # The null is deliberately recomputed rather than aliased so validators
        # can detect a future shape-only shortcut.
        null_logits = torch.zeros_like(logits)
        if streaming_chunk_size is None:
            null_assignment = self.dual_softmax_dense(null_logits, query_mask, reference_mask)
        else:
            null_assignment = self.dual_softmax_streaming(
                null_logits, query_mask, reference_mask,
                reference_chunk_size=streaming_chunk_size,
            )
        null = self._summary(
            null_logits, null_assignment, query_mask, reference_mask, query_grid, reference_grid
        )
        return CandidateEvidence(
            cost=cost,
            logits=logits,
            assignment=assignment,
            raw_summary=raw,
            null_summary=null,
            relational=raw - null,
        )

    def compare_relational(
        self, relational_g: torch.Tensor, relational_c: torch.Tensor, query_mask: torch.Tensor
    ) -> PairEvidence:
        if (
            tuple(relational_g.shape) != tuple(relational_c.shape)
            or relational_g.ndim != 2
            or relational_g.shape[1] != SUMMARY_DIM
            or relational_g.shape[0] != query_mask.numel()
        ):
            raise RCDEContractError("pair relational-field mismatch")
        valid = query_mask.flatten()
        signed = self.signed_head(relational_g).squeeze(-1) - self.signed_head(relational_c).squeeze(-1)
        symmetric = torch.cat(
            (
                relational_g + relational_c,
                torch.abs(relational_g - relational_c),
                relational_g * relational_c,
            ),
            dim=1,
        )
        modulation = torch.sigmoid(
            self.modulation_fc2(F.gelu(self.modulation_fc1(symmetric))).squeeze(-1)
        )
        # Mask before the pair head, and return this exact tensor below.  The
        # previous post-hoc masked return value was not an ancestor of
        # ``logit``; consequently an audit retaining ``PairEvidence.evidence``
        # saw no gradient even though the unmasked precursor drove the head.
        evidence = modulation * signed * valid.to(signed.dtype)
        positive = self.pair_fc2(
            F.gelu(self.pair_fc1(torch.stack((evidence, modulation), dim=1)))
        ).squeeze(-1)
        negative = self.pair_fc2(
            F.gelu(self.pair_fc1(torch.stack((-evidence, modulation), dim=1)))
        ).squeeze(-1)
        contributions = (positive - negative) * valid.to(positive.dtype)
        logit = contributions.sum() / valid.sum().to(contributions.dtype)
        return PairEvidence(
            signed_support=signed * valid.to(signed.dtype),
            modulation=modulation * valid.to(modulation.dtype),
            evidence=evidence,
            contributions=contributions,
            logit=logit,
        )

    def compare_relational_batch(
        self,
        relational: torch.Tensor,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized, order-preserving pair logits for bulk C128 replay.

        This is the same comparator as :meth:`compare_relational`, with an
        explicit leading pair axis.  It returns only the deployed scalar
        logits, so bulk replay can transfer and hash one contiguous vector
        instead of synchronizing the GPU once for every pair.
        """

        if (
            relational.ndim != 3
            or relational.shape[2] != SUMMARY_DIM
            or relational.shape[1] != query_mask.numel()
        ):
            raise RCDEContractError("batched pair relational-field mismatch")
        left = torch.as_tensor(left_indices, dtype=torch.long, device=relational.device)
        right = torch.as_tensor(right_indices, dtype=torch.long, device=relational.device)
        if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
            raise RCDEContractError("batched pair indices must be equal-length vectors")
        if left.numel() == 0:
            return relational.new_empty((0,))
        if (
            int(left.min()) < 0
            or int(right.min()) < 0
            or int(left.max()) >= relational.shape[0]
            or int(right.max()) >= relational.shape[0]
        ):
            raise RCDEContractError("batched pair index is outside candidate axis")
        g = relational.index_select(0, left)
        c = relational.index_select(0, right)
        valid = query_mask.flatten()
        signed = self.signed_head(g).squeeze(-1) - self.signed_head(c).squeeze(-1)
        symmetric = torch.cat(
            (g + c, torch.abs(g - c), g * c), dim=2
        )
        modulation = torch.sigmoid(
            self.modulation_fc2(F.gelu(self.modulation_fc1(symmetric))).squeeze(-1)
        )
        evidence = modulation * signed * valid[None, :].to(signed.dtype)
        positive = self.pair_fc2(
            F.gelu(self.pair_fc1(torch.stack((evidence, modulation), dim=2)))
        ).squeeze(-1)
        negative = self.pair_fc2(
            F.gelu(self.pair_fc1(torch.stack((-evidence, modulation), dim=2)))
        ).squeeze(-1)
        contributions = (positive - negative) * valid[None, :].to(positive.dtype)
        return contributions.sum(dim=1) / valid.sum().to(contributions.dtype)

    def forward_pair(
        self,
        query_layers: torch.Tensor,
        reference_g_layers: torch.Tensor,
        reference_c_layers: torch.Tensor,
        query_mask: torch.Tensor,
        reference_g_mask: torch.Tensor,
        reference_c_mask: torch.Tensor,
        query_grid: tuple[int, int],
        reference_g_grid: tuple[int, int],
        reference_c_grid: tuple[int, int],
        *,
        streaming_chunk_size: int | None = None,
        consensus_tile_shape: tuple[int, int, int, int] | None = None,
    ) -> tuple[CandidateEvidence, CandidateEvidence, PairEvidence]:
        candidate_g = self.decode_candidate(
            query_layers,
            reference_g_layers,
            query_mask,
            reference_g_mask,
            query_grid,
            reference_g_grid,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        candidate_c = self.decode_candidate(
            query_layers,
            reference_c_layers,
            query_mask,
            reference_c_mask,
            query_grid,
            reference_c_grid,
            streaming_chunk_size=streaming_chunk_size,
            consensus_tile_shape=consensus_tile_shape,
        )
        pair = self.compare_relational(candidate_g.relational, candidate_c.relational, query_mask)
        return candidate_g, candidate_c, pair


def assignment_max_abs_error(left: Assignment, right: Assignment) -> dict[str, float]:
    return {
        name: float((getattr(left, name) - getattr(right, name)).abs().max().detach().cpu())
        for name in ("Q", "G", "P", "u", "v")
    }


def functional_group_norms(
    model: DINO_RCDE_V1_2, *, gradients: bool
) -> dict[str, float]:
    named = dict(model.named_parameters())
    result: dict[str, float] = {}
    for group, names in model.functional_parameter_groups().items():
        total = 0.0
        for name in names:
            value = named[name].grad if gradients else named[name]
            if value is not None:
                total += float(value.detach().to(torch.float64).square().sum().cpu())
        result[group] = math.sqrt(total)
    return result


def assert_parameter_ledger(model: DINO_RCDE_V1_2) -> Mapping[str, int]:
    ledger = model.parameter_ledger()
    expected = {
        "projection_and_adapters": 62_464,
        "layer_mix": 4,
        "beta": 1,
        "four_d_consensus": 1_808,
        "signed_head": 16,
        "modulation_head": 784,
        "pair_head": 48,
    }
    if ledger != expected or sum(ledger.values()) != EXPECTED_PARAMETER_COUNT:
        raise RCDEContractError(f"parameter ledger drift: {ledger}")
    return ledger
