"""DRAFT V8: weight complete atom residuals before grouped scalar maxima.

No label/rank input, new encoder, global retrieval score or identity parameter.
Every scalar maximum has one canonical physical-atom witness. This changes
scientific pooling order; it does not reinterpret the frozen V7 failure.
Worst-atom pooling is not a semantic-conflict certificate or ranking guarantee.
V8 also explicitly replaces linear evidence/sentinels with exp(-E/2) affinity
and structural H0=0. This changes margins and training; it is not cosmetic
normalization, a calibrated probability, or a proven Mercer kernel on regions. H0 is structural unavailability, not absence.
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
H0_SCORE = 0.0  # Bottom of [0, 1]: no available proposal affinity, not semantic absence.


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


def _canonical_max_index(values, source_query_indices):
    """Hard choice by scalar cost, then smallest original source query index."""
    with torch.no_grad():
        maximum = values.detach().max()
        tied = torch.nonzero(values.detach() == maximum, as_tuple=False).flatten()
        return int(tied[source_query_indices[tied].argmin()])


def coherent_energy(residual_squared, reference_indices, weights, source_query_indices=None):
    """Return nonnegative energy and physical witnesses; gradients use gathers."""
    need(residual_squared.dtype == torch.float64 and residual_squared.ndim == 2 and residual_squared.shape[1] == D,
         "RESIDUAL_FP64_AXIS_INVALID")
    need(reference_indices.dtype == torch.long and reference_indices.shape == residual_squared.shape[:1] and
         len(reference_indices) > 0 and bool((reference_indices >= 0).all()), "EMPTY_OR_INVALID_REFERENCE_GROUP")
    need(bool(torch.isfinite(residual_squared).all()) and bool((residual_squared >= 0).all()) and
         bool((residual_squared <= 4.0 + 1e-12).all()), "RESIDUAL_DOMAIN_INVALID")
    need(weights.dtype == torch.float64 and weights.shape == (FEATURE_D,) and
         bool(torch.isfinite(weights).all()) and bool((weights > 0).all()), "METRIC_WEIGHT_DOMAIN_INVALID")
    source_query_indices = (torch.arange(len(reference_indices), dtype=torch.long) if source_query_indices is None
                            else source_query_indices)
    need(source_query_indices.dtype == torch.long and source_query_indices.shape == reference_indices.shape and
         bool((source_query_indices >= 0).all()), "SOURCE_QUERY_AXIS_INVALID")
    mean_cost = (residual_squared * weights[:D]).sum(dim=1)
    max_cost = (residual_squared * weights[D:]).sum(dim=1)
    groups, inverse = torch.unique(reference_indices, sorted=True, return_inverse=True)
    with torch.no_grad():
        maxima = torch.zeros(len(groups), dtype=torch.float64)
        maxima.scatter_reduce_(0, inverse, mean_cost.detach(), reduce="amax", include_self=True)
        tied = mean_cost.detach() == maxima[inverse]
        sentinel = torch.iinfo(torch.long).max
        group_source = torch.full((len(groups),), sentinel, dtype=torch.long)
        group_source.scatter_reduce_(0, inverse[tied], source_query_indices[tied], reduce="amin", include_self=True)
        canonical = tied & (source_query_indices == group_source[inverse])
        positions = torch.arange(len(reference_indices), dtype=torch.long)
        chosen = torch.full((len(groups),), len(reference_indices), dtype=torch.long)
        # The final position tie-break only distinguishes exact duplicated atoms
        # with the same source ID; it does not create a new physical witness.
        chosen.scatter_reduce_(0, inverse[canonical], positions[canonical], reduce="amin", include_self=True)
    maximum_index = _canonical_max_index(max_cost, source_query_indices)
    # Index decisions are detached, but these gathers retain the chosen atom's
    # actual full-vector derivative. No per-channel synthetic residual is built.
    mean_term = mean_cost[chosen].mean()
    max_term = max_cost[maximum_index]
    energy = mean_term + max_term
    witnesses = {"pooling": "WEIGHT_CHANNELS_THEN_MAX_ACTUAL_ATOMS_V8",
                 "reference_group_indices": groups.tolist(),
                 "group_mean_witness_source_query_indices": source_query_indices[chosen].tolist(),
                 "global_max_witness_source_query_index": int(source_query_indices[maximum_index]),
                 "group_mean_cost_binary64": [float(x).hex() for x in mean_cost[chosen].detach()],
                 "global_max_cost_binary64": float(max_term.detach()).hex(),
                 "coherent_energy_binary64": float(energy.detach()).hex(),
                 "linear_audit_score_binary64": float((1.0 - 0.5 * energy).detach()).hex(),
                 "score_link": "FIXED_EXP_NEG_HALF_ENERGY_V8"}
    return energy, witnesses


def coherent_linear_audit_score(residual_squared, reference_indices, weights, source_query_indices=None):
    """Audit only: never used by candidate evidence, training, or deployment."""
    energy, witnesses = coherent_energy(residual_squared, reference_indices, weights, source_query_indices)
    return 1.0 - 0.5 * energy, witnesses


def coherent_score(residual_squared, reference_indices, weights, source_query_indices=None):
    """Fixed bounded affinity. No learned or scanned temperature or threshold."""
    energy, witnesses = coherent_energy(residual_squared, reference_indices, weights, source_query_indices)
    affinity = torch.exp(-0.5 * energy)
    need(bool(torch.isfinite(affinity)) and bool(affinity > 0.0) and bool(affinity <= 1.0),
         "NONEMPTY_AFFINITY_DOMAIN_INVALID")
    return affinity, witnesses


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

    def forward(self, residual_squared, reference_indices, source_query_indices=None, *, return_witness=False):
        score, witness = coherent_score(residual_squared, reference_indices, self.weights(), source_query_indices)
        return (score, witness) if return_witness else score


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
    residual_squared: torch.Tensor
    valid: torch.Tensor
    source_reference_indices: tuple[torch.Tensor, ...]
    reference_tokens: tuple[torch.Tensor, ...]
    reference_valid: tuple[torch.Tensor, ...]
    query_permutation: torch.Tensor
    families: tuple[tuple[dict, ...], ...]
    allpatch_available: torch.Tensor
    query_axis_source: torch.Tensor
    query_variation: torch.Tensor


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
    need(qvalid.shape == q.shape[:1] and bool(qvalid.all()), "V8_REQUIRES_GENUINE_DENSE_QUERY_GRID")
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
        ap_available = []
        for i in range(len(keys)):
            rt, rv = references[i], reference_valid[i]
            need(rt.dtype == torch.float64 and rt.ndim == 2 and rt.shape[1] == D and
                 rv.shape == rt.shape[:1] and bool(rv.any()) and bool(torch.isfinite(rt).all()),
                 "EMPTY_OR_MALFORMED_REFERENCE_ASSET")
            need(bool(((rt[rv].square().sum(1)-1).abs() <= 2e-12).all()), "VALID_REFERENCE_TOKEN_NOT_UNIT_NORM")
            for component in families[i]:
                ids = torch.tensor(component["atom_indices"], dtype=torch.long)
                need(len(ids) >= 4 and bool(valid[i, ids].all()), "INVALID_GENERATED_H1_SUPPORT")
            rows = torch.nonzero(valid[i], as_tuple=False).flatten()
            ap_available.append(bool(len(rows)))
            need(refs[i].shape == (len(q),), "SOURCE_REFERENCE_AXIS_INVALID")
            observed = residual[i, rows]
            need(bool(torch.isfinite(observed).all()) and bool((observed >= 0).all()) and
                 bool((observed <= 4.0 + 1e-12).all()), "RESIDUAL_DOMAIN_INVALID")
        cache_key = tensor_sha(permutation)
        if cache_key not in selection_cache:
            selection_cache[cache_key] = torch.argsort(permutation)[current_axis]
        source_axis = selection_cache[cache_key]
        # Scalar costs depend on learned weights, so raw FP64 residuals must
        # remain available. No old static 256-channel maxima descriptor is kept.
        ctx = ControlContext(name, refkeys, residual, valid, refs, references, reference_valid, permutation,
                             families, torch.tensor(ap_available, dtype=torch.bool), source_axis, variation)
        controls[name] = ctx
        feature_bindings[name] = {"residual_squared": tensor_sha(residual), "valid": tensor_sha(valid),
                                 "source_reference_indices": [tensor_sha(x) for x in refs],
                                 "query_variation": tensor_sha(variation), "families": logical(families),
                                 "query_source_axis": tensor_sha(source_axis), "query_permutation": cache_key,
                                 "pooling": "WEIGHT_CHANNELS_THEN_MAX_ACTUAL_ATOMS_V8"}
    need(torch.equal(controls["REAL"].residual_squared, controls["P_COORD"].residual_squared) and
         torch.equal(controls["REAL"].valid, controls["P_COORD"].valid) and
         all(torch.equal(a, b) for a, b in zip(controls["REAL"].source_reference_indices,
                                             controls["P_COORD"].source_reference_indices, strict=True)),
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
    scores, decisions, available = [], [], []
    for i, key in enumerate(context.candidate_keys):
        values, witnesses = [], []
        for component in control.families[i]:
            ids = torch.tensor(component["atom_indices"], dtype=torch.long)
            source_ids = torch.tensor(component["source_query_indices"], dtype=torch.long)
            value, witness = head(control.residual_squared[i, ids], control.source_reference_indices[i][ids],
                                  source_ids, return_witness=True)
            values.append(value); witnesses.append(witness)
        vector = torch.stack(values) if values else head.theta.sum() * 0.0 + torch.empty((0,), dtype=torch.float64)
        exists = bool(len(vector))
        if exists:
            score, ordinal_tensor = vector.max(dim=0)
            ordinal = int(ordinal_tensor.detach())
            support = dict(_support(control.families[i][ordinal]))
            support["pooling_witness"] = witnesses[ordinal]
            need(bool(torch.isfinite(vector).all()) and bool((vector > H0_SCORE).all()), "NONEMPTY_SCORE_DOMAIN_DRIFT")
        else:
            score, ordinal, support = _constant(head)[0], None, None
        scores.append(score); available.append(exists)
        decisions.append(Decision(context.query_resource_key, key, control.reference_keys[i], ARM_REAL, control.name,
                                  score, exists, vector, support, ordinal))
    return {"scores": torch.stack(scores), "decisions": tuple(decisions),
            "available_mask": torch.tensor(available, dtype=torch.bool)}


def _allpatch_branch(context, control, head):
    values, decisions = [], []
    for i, key in enumerate(context.candidate_keys):
        exists = bool(control.allpatch_available[i])
        ids = torch.nonzero(control.valid[i], as_tuple=False).flatten()
        if exists:
            score, witness = head(control.residual_squared[i, ids], control.source_reference_indices[i][ids], ids,
                                  return_witness=True)
            support = {"atom_indices": ids.tolist(), "source_query_indices": ids.tolist(),
                       "source_reference_indices": control.source_reference_indices[i][ids].tolist(),
                       "pooling_witness": witness, "claim": "ALL_PATCH_NO_HYP"}
            vector = score.reshape(1)
        else:
            score, support = _constant(head)[0], None
            vector = head.theta.sum() * 0.0 + torch.empty((0,), dtype=torch.float64)
        values.append(score)
        decisions.append(Decision(context.query_resource_key, key, control.reference_keys[i], ARM_ALLPATCH, control.name,
                                  score, exists, vector, support, 0 if exists else None))
    return {"scores": torch.stack(values), "decisions": tuple(decisions), "available_mask": control.allpatch_available}


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
    # The candidate-independent selector uses the same coherent operator order.
    mean_cost = (control.query_variation * weights[:D]).sum(dim=1)
    max_cost = (control.query_variation * weights[D:]).sum(dim=1)
    selector_mean = mean_cost[control.query_axis_source].mean(dim=1)
    # Sorting only the four source IDs makes scalar-max ties canonical.
    canonical_sources = control.query_axis_source.sort(dim=1).values
    candidate_max = max_cost[canonical_sources]
    with torch.no_grad():
        max_positions = candidate_max.detach().argmax(dim=1)
    selector_max = candidate_max[torch.arange(len(candidate_max)), max_positions]
    selector_scores = selector_mean + selector_max
    ordinal = int(selector_scores.detach().argmax())
    source_ids = control.query_axis_source[ordinal]
    current_ids = context.query_axis_current[ordinal]
    query_four = context.query_tokens[source_ids]
    values, matches, witnesses = [], [], []
    for ref, rv in zip(control.reference_tokens, control.reference_valid, strict=True):
        residual, ids = appearance_assignment(query_four, ref, rv, weights)
        value, witness = head(residual, ids, source_ids, return_witness=True)
        values.append(value); matches.append(ids); witnesses.append(witness)
    scores = torch.stack(values)
    need(bool(torch.isfinite(scores).all()) and bool((scores > H0_SCORE).all()), "QUERY_ONLY_SCORE_DOMAIN_DRIFT")
    decisions = []
    for i, key in enumerate(context.candidate_keys):
        support = {"atom_indices": source_ids.tolist(), "source_query_indices": source_ids.tolist(),
                   "query_indices": current_ids.tolist(),
                   "query_rc": [[int(k)//context.query_grid_shape[1], int(k)%context.query_grid_shape[1]] for k in current_ids],
                   "source_reference_indices": matches[i].tolist(), "assignment": "COMPLETE_REFERENCE_APPEARANCE_ARGMIN",
                   "query_selection_reads_candidate_content": False, "pooling_witness": witnesses[i],
                   "query_selector_max_witness_source_query_index": int(canonical_sources[ordinal, max_positions[ordinal]])}
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


CoherentResidualHeads = GroupedResidualHeads
