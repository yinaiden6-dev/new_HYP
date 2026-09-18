"""V7: fixed cycle-connected P families and a shared grouped residual metric.

No label/rank input, new encoder, global retrieval score or identity parameter.
Max pooling retains channel maxima, not every observation. Learned weights do
not certify semantic conflict. H0 is structural unavailability, not absence.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from typing import Any
import torch
from torch import nn

D = 128
FEATURE_D = 2 * D
ARM_REAL = "REAL"
ARM_ALLPATCH = "ALL_PATCH_NO_HYP"
ARM_QUERY_ONLY = "QUERY_ONLY_REGION"
ARM_NAMES = (ARM_REAL, ARM_ALLPATCH, ARM_QUERY_ONLY)
CONTROL_NAMES = ("REAL", "C_BIND", "P_COORD")
H0_SCORE = -256.0  # Below the mathematical nonempty score range [-255, 1].


def need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def logical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_sha(value):
    a = value.detach().cpu().contiguous()
    return hashlib.sha256(str(a.dtype).encode() + str(tuple(a.shape)).encode() + a.numpy().tobytes()).hexdigest()


def tensor_seal(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(value)}


def grouped_descriptor(residual_squared, reference_indices):
    """Each reference cell contributes its channelwise worst observed residual."""
    need(residual_squared.dtype == torch.float64 and residual_squared.ndim == 2 and residual_squared.shape[1] == D,
         "RESIDUAL_FP64_AXIS_INVALID")
    need(reference_indices.dtype == torch.long and reference_indices.shape == residual_squared.shape[:1],
         "REFERENCE_CELL_AXIS_INVALID")
    need(len(reference_indices) > 0 and bool((reference_indices >= 0).all()), "EMPTY_OR_INVALID_REFERENCE_GROUP")
    need(bool(torch.isfinite(residual_squared).all()) and bool((residual_squared >= 0).all()) and
         bool((residual_squared <= 4.0 + 1e-12).all()), "RESIDUAL_DOMAIN_INVALID")
    unique, inverse = torch.unique(reference_indices, sorted=True, return_inverse=True)
    maxima = torch.zeros((len(unique), D), dtype=torch.float64)
    maxima.scatter_reduce_(0, inverse[:, None].expand(-1, D), residual_squared, reduce="amax", include_self=True)
    return torch.cat((maxima.mean(dim=0), maxima.max(dim=0).values))


@lru_cache(maxsize=8)
def _four_axis(height, width, valid_tuple):
    valid = {i for i, flag in enumerate(valid_tuple) if flag}
    neighbours = {}
    for i in valid:
        row, col = divmod(i, width)
        neighbours[i] = tuple(j for r, c in ((row-1, col), (row+1, col), (row, col-1), (row, col+1))
                              if 0 <= r < height and 0 <= c < width for j in (r*width+c,) if j in valid)
    groups = {(i,) for i in valid}
    for _ in range(3):
        groups = {tuple(sorted((*g, j))) for g in groups for i in g for j in neighbours[i] if j not in g}
    return tuple(sorted(groups))


class ResidualMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(FEATURE_D, dtype=torch.float64))

    def weights(self):
        need(self.theta.dtype == torch.float64 and self.theta.device.type == "cpu" and
             bool(torch.isfinite(self.theta).all()), "HEAD_DTYPE_DEVICE_OR_FINITE_DRIFT")
        weights = D * torch.softmax(self.theta, dim=0)
        need(bool((weights > 0).all()) and bool(torch.isfinite(weights).all()), "METRIC_WEIGHT_UNDERFLOW_OR_NONFINITE")
        return weights

    def forward(self, features):
        need(features.dtype == torch.float64 and features.shape[-1] == FEATURE_D, "DESCRIPTOR_AXIS_DRIFT")
        return 1.0 - 0.5 * (features * self.weights()).sum(dim=-1)


class GroupedResidualHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.real = ResidualMetric()
        self.allpatch = ResidualMetric()
        self.query_only = ResidualMetric()

    def arms(self):
        return {ARM_REAL: self.real, ARM_ALLPATCH: self.allpatch, ARM_QUERY_ONLY: self.query_only}

    def optimizable_arms(self):
        return self.arms()


@dataclass
class ControlContext:
    name: str
    reference_keys: tuple[str, ...]
    residual_squared: torch.Tensor | None
    valid: torch.Tensor
    source_reference_indices: tuple[torch.Tensor, ...]
    reference_tokens: tuple[torch.Tensor, ...]
    reference_valid: tuple[torch.Tensor, ...]
    query_permutation: torch.Tensor
    families: tuple[tuple[dict, ...], ...]
    region_features: tuple[torch.Tensor, ...]
    allpatch_features: torch.Tensor
    allpatch_available: torch.Tensor
    query_axis_source: torch.Tensor
    query_selector_features: torch.Tensor


@dataclass
class EpisodeContext:
    query_resource_key: str
    candidate_keys: tuple[str, ...]
    query_tokens: torch.Tensor
    query_grid_shape: tuple[int, int]
    query_axis_current: torch.Tensor
    controls: dict[str, ControlContext]
    source_receipt: dict
    context_sha256: str


@dataclass
class Decision:
    query_resource_key: str
    candidate_resource_key: str
    reference_resource_key: str
    arm: str
    control: str
    candidate_score: torch.Tensor
    available: bool
    complete_seed_scores: torch.Tensor
    selected_H: dict | None
    map_seed_ordinal: int | None

    @property
    def h1(self):
        return self.available and self.arm != ARM_ALLPATCH

    @property
    def score(self):
        return self.candidate_score


def build_context(episode):
    keys0 = tuple(episode["candidate_keys"])
    keys = tuple(sorted(keys0))
    need(len(keys) == len(set(keys)) and len(keys) >= 2, "CANDIDATE_AXIS_INVALID")
    order = [keys0.index(k) for k in keys]
    q = episode["qtokens"].detach()
    grid = tuple(episode["query_grid_shape"])
    qvalid = episode["query_valid_axis"].to(torch.bool)
    need(q.dtype == torch.float64 and q.shape == (grid[0]*grid[1], D) and bool(torch.isfinite(q).all()),
         "QUERY_TOKEN_AXIS_INVALID")
    need(qvalid.shape == q.shape[:1] and bool(qvalid.all()), "V7_REQUIRES_GENUINE_DENSE_QUERY_GRID")
    need(bool(((q.square().sum(1)-1).abs() <= 2e-12).all()), "VALID_QUERY_TOKEN_NOT_UNIT_NORM")
    current_axis = torch.tensor(_four_axis(*grid, tuple(qvalid.tolist())), dtype=torch.long).reshape(-1, 4)
    need(len(current_axis) > 0, "QUERY_ONLY_FAMILY_EMPTY")
    variation = (q - q.mean(dim=0)).square()
    controls = {}
    selection_cache = {}
    feature_bindings = {}
    for name in CONTROL_NAMES:
        raw_axes = [episode["axes"][name][i] for i in order]
        permutation = raw_axes[0]["query_indices"].to(torch.long)
        need(torch.equal(torch.sort(permutation).values, torch.arange(len(q))) and
             all(torch.equal(a["query_indices"], permutation) for a in raw_axes),
             "CANDIDATE_DEPENDENT_OR_INCOMPLETE_QUERY_COORDINATE_PERMUTATION")
        residual = episode["residual_squared"][name][order].detach() if order != list(range(len(keys))) else episode["residual_squared"][name].detach()
        valid = episode["valid"][name][order].detach()
        need(residual.dtype == torch.float64 and residual.shape == (len(keys), len(q), D) and
             valid.dtype == torch.bool and valid.shape == residual.shape[:2], "CONTROL_RESIDUAL_AXIS_INVALID")
        references = tuple(episode["reference_tokens"][name][i] for i in order)
        reference_valid = tuple(episode["reference_valid"][name][i].to(torch.bool) for i in order)
        refkeys = tuple(a["reference_resource_key"] for a in raw_axes)
        refs = tuple(a["source_reference_indices"].to(torch.long) for a in raw_axes)
        families = tuple(tuple(episode["families"][name][i]["components"]) for i in order)
        region_features, ap_features, ap_available = [], [], []
        for i in range(len(keys)):
            rt, rv = references[i], reference_valid[i]
            need(rt.dtype == torch.float64 and rt.ndim == 2 and rt.shape[1] == D and
                 rv.shape == rt.shape[:1] and bool(rv.any()) and bool(torch.isfinite(rt).all()),
                 "EMPTY_OR_MALFORMED_REFERENCE_ASSET")
            need(bool(((rt[rv].square().sum(1)-1).abs() <= 2e-12).all()), "VALID_REFERENCE_TOKEN_NOT_UNIT_NORM")
            desc = []
            for component in families[i]:
                ids = torch.tensor(component["atom_indices"], dtype=torch.long)
                need(len(ids) >= 4 and bool(valid[i, ids].all()), "INVALID_GENERATED_H1_SUPPORT")
                desc.append(grouped_descriptor(residual[i, ids], refs[i][ids]))
            region_features.append(torch.stack(desc) if desc else torch.empty((0, FEATURE_D), dtype=torch.float64))
            rows = torch.nonzero(valid[i], as_tuple=False).flatten()
            ap_available.append(bool(len(rows)))
            ap_features.append(grouped_descriptor(residual[i, rows], refs[i][rows]) if len(rows) else torch.zeros(FEATURE_D, dtype=torch.float64))
        cache_key = tensor_sha(permutation)
        if cache_key not in selection_cache:
            source_axis = torch.argsort(permutation)[current_axis]
            v = variation[source_axis]
            selection_cache[cache_key] = (source_axis, torch.cat((v.mean(dim=1), v.max(dim=1).values), dim=1))
        source_axis, selector_features = selection_cache[cache_key]
        # Raw residuals are source-hashed by the adapter and fully reduced here.
        # Scoring needs descriptors and original reference tokens, not this large temporary.
        ctx = ControlContext(name, refkeys, None, valid, refs, references, reference_valid, permutation,
                             families, tuple(region_features), torch.stack(ap_features),
                             torch.tensor(ap_available, dtype=torch.bool), source_axis, selector_features)
        controls[name] = ctx
        feature_bindings[name] = {"region_features": [tensor_sha(f) for f in region_features],
                                 "allpatch_features": tensor_sha(ctx.allpatch_features),
                                 "query_selector_features": tensor_sha(selector_features),
                                 "query_source_axis": tensor_sha(source_axis), "query_permutation": cache_key}
    need(torch.equal(controls["REAL"].allpatch_features, controls["P_COORD"].allpatch_features) and
         torch.equal(controls["REAL"].allpatch_available, controls["P_COORD"].allpatch_available),
         "ALLPATCH_COORDINATE_INVARIANCE_FAILED")
    need(torch.equal(controls["REAL"].query_permutation, controls["C_BIND"].query_permutation),
         "C_BIND_CHANGED_QUERY_COORDINATES")
    source_receipt = dict(episode["source_receipt"])
    context_sha = logical({"query": episode["query_resource_key"], "candidate_keys": keys,
                           "source_receipt": source_receipt, "feature_bindings": feature_bindings})
    return EpisodeContext(episode["query_resource_key"], keys, q, grid, current_axis, controls, source_receipt, context_sha)


def _constant(head, count=1):
    return head.theta.sum() * 0.0 + torch.full((count,), H0_SCORE, dtype=torch.float64)


def _support(component):
    return {k: component[k] for k in ("atom_indices", "query_indices", "query_rc", "reference_indices", "reference_rc",
                                      "source_query_indices", "source_reference_indices")}


def _real_branch(context, control, head):
    scores, decisions = [], []
    available = []
    for i, key in enumerate(context.candidate_keys):
        vector = head(control.region_features[i])
        exists = bool(len(vector))
        if exists:
            score, ordinal_tensor = vector.max(dim=0)
            ordinal = int(ordinal_tensor.detach())
            support = _support(control.families[i][ordinal])
            need(bool(torch.isfinite(vector).all()) and bool((vector > H0_SCORE).all()), "NONEMPTY_SCORE_DOMAIN_DRIFT")
        else:
            score, ordinal, support = _constant(head)[0], None, None
        scores.append(score)
        available.append(exists)
        decisions.append(Decision(context.query_resource_key, key, control.reference_keys[i], ARM_REAL, control.name,
                                  score, exists, vector, support, ordinal))
    return {"scores": torch.stack(scores), "decisions": tuple(decisions),
            "available_mask": torch.tensor(available, dtype=torch.bool)}


def _allpatch_branch(context, control, head):
    values = head(control.allpatch_features)
    scores = torch.where(control.allpatch_available, values, _constant(head, len(context.candidate_keys)))
    decisions = []
    for i, key in enumerate(context.candidate_keys):
        available = bool(control.allpatch_available[i])
        ids = torch.nonzero(control.valid[i], as_tuple=False).flatten().tolist()
        support = {"atom_indices": ids, "source_query_indices": ids,
                   "source_reference_indices": control.source_reference_indices[i][ids].tolist(),
                   "claim": "ALL_PATCH_NO_HYP"} if available else None
        decisions.append(Decision(context.query_resource_key, key, control.reference_keys[i], ARM_ALLPATCH, control.name,
                                  scores[i], available, values[i:i+1] if available else values[:0], support, 0 if available else None))
    return {"scores": scores, "decisions": tuple(decisions), "available_mask": control.allpatch_available}


def appearance_assignment(query_four, reference, reference_valid, weights):
    """Fixed per-query-atom argmin on every valid reference token; not joint OT."""
    need(query_four.shape == (4, D) and reference.dtype == torch.float64 and bool(reference_valid.any()),
         "APPEARANCE_MATCH_INPUT_INVALID")
    # Assignment is discrete. All source features are frozen; selected residuals
    # remain independent of theta away from assignment boundaries.
    with torch.no_grad():
        residual = (query_four[:, None, :] - reference[None, :, :]).square()
        channel_weights = weights[:D].detach() + weights[D:].detach()
        costs = (residual * channel_weights).sum(dim=-1)
        costs[:, ~reference_valid] = torch.inf
        chosen = costs.argmin(dim=1)
        selected = residual[torch.arange(4), chosen]
    return selected, chosen


def _query_branch(context, control, head):
    weights = head.weights()
    selector_scores = (control.query_selector_features * weights).sum(dim=1)
    ordinal = int(selector_scores.detach().argmax())
    source_ids = control.query_axis_source[ordinal]
    current_ids = context.query_axis_current[ordinal]
    query_four = context.query_tokens[source_ids]
    descriptors, matches = [], []
    for ref, rv in zip(control.reference_tokens, control.reference_valid, strict=True):
        residual, ids = appearance_assignment(query_four, ref, rv, weights)
        descriptors.append(grouped_descriptor(residual, ids))
        matches.append(ids)
    descriptors = torch.stack(descriptors)
    scores = head(descriptors)
    need(bool(torch.isfinite(scores).all()) and bool((scores > H0_SCORE).all()), "QUERY_ONLY_SCORE_DOMAIN_DRIFT")
    decisions = []
    for i, key in enumerate(context.candidate_keys):
        support = {"atom_indices": source_ids.tolist(), "source_query_indices": source_ids.tolist(),
                   "query_indices": current_ids.tolist(),
                   "query_rc": [[int(k)//context.query_grid_shape[1], int(k)%context.query_grid_shape[1]] for k in current_ids],
                   "source_reference_indices": matches[i].tolist(), "assignment": "COMPLETE_REFERENCE_APPEARANCE_ARGMIN",
                   "query_selection_reads_candidate_content": False}
        decisions.append(Decision(context.query_resource_key, key, control.reference_keys[i], ARM_QUERY_ONLY, control.name,
                                  scores[i], True, scores[i:i+1], support, ordinal))
    return {"scores": scores, "decisions": tuple(decisions),
            "available_mask": torch.ones(len(context.candidate_keys), dtype=torch.bool),
            "query_selector_complete_scores": selector_scores,
            "query_selector_selected_atom_indices": source_ids.tolist(), "query_selector_selected_current_indices": current_ids.tolist(),
            "query_selector_map_ordinal": ordinal}


def score_episode(context, heads):
    branches = {}
    for name in CONTROL_NAMES:
        control = context.controls[name]
        branches[name] = {ARM_REAL: _real_branch(context, control, heads.real),
                          ARM_ALLPATCH: _allpatch_branch(context, control, heads.allpatch),
                          ARM_QUERY_ONLY: _query_branch(context, control, heads.query_only)}
    arms = {}
    for arm in ARM_NAMES:
        real, bind, coord = (branches[n][arm] for n in CONTROL_NAMES)
        arms[arm] = {"scores": real["scores"], "c_bind_scores": bind["scores"], "p_coord_scores": coord["scores"],
                     "decisions": real["decisions"], "c_bind_decisions": bind["decisions"], "p_coord_decisions": coord["decisions"],
                     "available_mask": real["available_mask"], "c_bind_available_mask": bind["available_mask"],
                     "p_coord_available_mask": coord["available_mask"]}
        if arm == ARM_QUERY_ONLY:
            for field in ("query_selector_complete_scores", "query_selector_selected_atom_indices",
                          "query_selector_selected_current_indices", "query_selector_map_ordinal"):
                arms[arm][field] = {n: branches[n][arm][field] for n in CONTROL_NAMES}
    need(torch.equal(arms[ARM_ALLPATCH]["scores"], arms[ARM_ALLPATCH]["p_coord_scores"]), "ALLPATCH_P_COORD_BITS_CHANGED")
    need(arms[ARM_QUERY_ONLY]["query_selector_selected_atom_indices"]["REAL"] ==
         arms[ARM_QUERY_ONLY]["query_selector_selected_atom_indices"]["C_BIND"], "C_BIND_CHANGED_QUERY_ONLY_PROPOSAL")
    return {"query_resource_key": context.query_resource_key, "candidate_keys": context.candidate_keys,
            "context_sha256": context.context_sha256, "arms": arms}


deploy_episode = score_episode


def serialize_decision(decision):
    return {"query_resource_key": decision.query_resource_key, "candidate_resource_key": decision.candidate_resource_key,
            "reference_resource_key": decision.reference_resource_key, "arm": decision.arm, "control": decision.control,
            "state": "H1" if decision.h1 else "ALL_PATCH" if decision.available else "H0",
            "h0_reason": None if decision.available else "STRUCTURALLY_EMPTY_AVAILABLE_FAMILY",
            "candidate_score_binary64": float(decision.candidate_score.detach()).hex(),
            "selected_H": decision.selected_H, "map_seed_ordinal": decision.map_seed_ordinal,
            "complete_seed_scores": tensor_seal(decision.complete_seed_scores)}


def serialize_episode_scores(result):
    output = {k: result[k] for k in ("query_resource_key", "candidate_keys", "context_sha256")}
    output["candidate_keys"] = list(output["candidate_keys"])
    output["arms"] = {}
    for arm in ARM_NAMES:
        original = result["arms"][arm]
        data = {}
        for field in ("scores", "c_bind_scores", "p_coord_scores"):
            data[field + "_binary64"] = [float(x).hex() for x in original[field].detach()]
        for field in ("decisions", "c_bind_decisions", "p_coord_decisions"):
            data[field] = [serialize_decision(d) for d in original[field]]
        for field in ("available_mask", "c_bind_available_mask", "p_coord_available_mask"):
            data[field] = original[field].tolist()
        if arm == ARM_QUERY_ONLY:
            data["query_selector_complete_scores"] = {name: tensor_seal(vector) for name, vector in original["query_selector_complete_scores"].items()}
            for field in ("query_selector_selected_atom_indices", "query_selector_selected_current_indices", "query_selector_map_ordinal"):
                data[field] = original[field]
        output["arms"][arm] = data
    return output


def context_summary(context):
    return {"query_resource_key": context.query_resource_key, "candidate_keys": list(context.candidate_keys),
            "context_sha256": context.context_sha256, "source_receipt_sha256": logical(context.source_receipt),
            "query_grid_shape": list(context.query_grid_shape),
            "query_only_complete_family_count": len(context.query_axis_current),
            "component_counts": {n: [len(x) for x in c.families] for n, c in context.controls.items()},
            "query_permutation_sha256": {n: tensor_sha(c.query_permutation) for n, c in context.controls.items()},
            "P_COORD_all_REAL_families_empty": all(not x for x in context.controls["P_COORD"].families)}
