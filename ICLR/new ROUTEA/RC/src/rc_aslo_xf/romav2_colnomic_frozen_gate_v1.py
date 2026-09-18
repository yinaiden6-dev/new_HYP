"""Frozen deployable six-feature, seven-parameter HOLD/SWITCH head."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import torch

FEATURE_NAMES = (
    "standardized_raw_gap",
    "symmetric_local_score",
    "symmetric_visibility_mass",
    "symmetric_visibility_normalized_local_similarity",
    "symmetric_query_spatial_robustness",
    "symmetric_reference_spatial_robustness",
)
WEIGHT = torch.tensor(
    [0.9812819097198987, -3.812628067996388, 5.826320131219255,
     5.860970045711693, -0.14878670951682949, -0.2442796885686483],
    dtype=torch.float64,
)
BIAS = -1.5548565799571785
PARAMETER_COUNT = 7
SWITCH_THRESHOLD = 0.0

def symmetric(a: float, b: float) -> float:
    return (float(a) - float(b)) / (abs(float(a)) + abs(float(b)) + 1e-12)

def candidate_feature(raw: Sequence[float], candidates: Mapping[int, Mapping[str, float]], challenger: int, winner: int) -> torch.Tensor:
    mean = sum(map(float, raw)) / len(raw)
    std = (sum((float(x) - mean) ** 2 for x in raw) / len(raw)) ** 0.5
    lc, lw = candidates[challenger], candidates[winner]
    sc = float(lc["real_score"]) / max(float(lc["visibility_mass"]), 1e-12)
    sw = float(lw["real_score"]) / max(float(lw["visibility_mass"]), 1e-12)
    qc, qw = float(lc["real_score"]) - float(lc["query_control_score"]), float(lw["real_score"]) - float(lw["query_control_score"])
    rc, rw = float(lc["real_score"]) - float(lc["reference_control_score"]), float(lw["real_score"]) - float(lw["reference_control_score"])
    return torch.tensor([
        (float(raw[challenger]) - float(raw[winner])) / max(std, 1e-12),
        symmetric(lc["real_score"], lw["real_score"]),
        symmetric(lc["visibility_mass"], lw["visibility_mass"]),
        symmetric(sc, sw), symmetric(qc, qw), symmetric(rc, rw),
    ], dtype=torch.float64)

def logit(raw: Sequence[float], candidates: Mapping[int, Mapping[str, float]], challenger: int, winner: int) -> float:
    return float((WEIGHT * candidate_feature(raw, candidates, challenger, winner)).sum() + BIAS)
