#!/usr/bin/env python3
"""Independent synthetic E0 for V7; no natural payload or encoder is opened.

Default execution prints a reviewable receipt but does not freeze it.  Use
--freeze only after the bound core and contract are final.  Numerical oracles
use NumPy/Python, independently of the production pooling/selection routines.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/rc_aslo_xf/reference_conditioned_grouped_residual_p_only_v7.py"
PLAN = ROOT / "plan/REFERENCE_CONDITIONED_GROUPED_RESIDUAL_P_ONLY_V7_20260908.md"


def need(value, message):
    if not bool(value):
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core():
    name = "independent_grouped_residual_e0_core_v7"
    spec = importlib.util.spec_from_file_location(name, CORE)
    need(spec is not None and spec.loader is not None, "CORE_IMPORT_UNAVAILABLE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rejects(call, expected):
    try:
        call()
    except (ValueError, AssertionError) as error:
        need(expected in str(error), f"WRONG_REJECTION:{error!s}")
    else:
        raise AssertionError("MISSING_REJECTION:" + expected)


def numpy_descriptor(residual, groups, np):
    keys = sorted(set(int(k) for k in groups))
    maxima = np.array([[max(float(residual[i, d]) for i in range(len(groups))
                            if int(groups[i]) == k) for d in range(128)] for k in keys])
    return np.concatenate((maxima.sum(axis=0) / len(keys), maxima.max(axis=0)))


def numpy_weights(theta, np):
    exponents = np.exp(theta - np.max(theta))
    return 128.0 * exponents / exponents.sum()


def numpy_appearance(query, reference, valid, weights, np):
    selected, residuals = [], []
    channel_weights = weights[:128] + weights[128:]
    for q in query:
        candidates = []
        for j, r in enumerate(reference):
            if bool(valid[j]):
                error = (q - r) ** 2
                cost = float(np.sum(error * channel_weights))
                candidates.append((cost, j, error))
        _, j, error = min(candidates, key=lambda item: (item[0], item[1]))
        selected.append(j)
        residuals.append(error)
    return np.array(residuals), np.array(selected)


def independent_four_axis(height, width):
    result = []
    for group in itertools.combinations(range(height * width), 4):
        remaining, reached = set(group[1:]), {group[0]}
        while True:
            additions = {b for b in remaining for a in reached
                         if abs(a // width - b // width) + abs(a % width - b % width) == 1}
            if not additions:
                break
            reached.update(additions)
            remaining.difference_update(additions)
        if not remaining:
            result.append(group)
    return result


def fixture(torch, np):
    rng = np.random.default_rng(1707)
    query = rng.normal(size=(16, 128))
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    references = []
    for _ in range(3):
        value = rng.normal(size=(11, 128))
        value /= np.linalg.norm(value, axis=1, keepdims=True)
        references.append(torch.from_numpy(value.copy()))
    query = torch.from_numpy(query.copy())
    keys = ("synthetic-candidate-b", "synthetic-candidate-a", "synthetic-candidate-c")
    sources = [((torch.arange(16) + 2 * i) % 11).long() for i in range(3)]
    permutation = torch.tensor([7, 2, 14, 1, 9, 0, 15, 4, 3, 12, 6, 10, 5, 13, 8, 11])
    episode = {"query_resource_key": "synthetic-v7-query", "candidate_keys": keys,
               "qtokens": query, "query_grid_shape": (4, 4),
               "query_valid_axis": torch.ones(16, dtype=torch.bool),
               "source_receipt": {"synthetic_only": True, "source": "v7-independent-E0"},
               "axes": {}, "families": {}, "residual_squared": {}, "valid": {},
               "reference_tokens": {}, "reference_valid": {}}
    for control in ("REAL", "C_BIND", "P_COORD"):
        axes, families, residual, refs, masks = [], [], [], [], []
        qperm = permutation if control == "P_COORD" else torch.arange(16)
        for i in range(3):
            donor = (i + 1) % 3 if control == "C_BIND" else i
            source = sources[donor]
            reference = references[donor]
            axes.append({"query_indices": qperm.clone(), "source_reference_indices": source.clone(),
                         "reference_resource_key": f"synthetic-reference-{donor}"})
            components = []
            for ids in ((0, 1, 4, 5), (10, 11, 14, 15)) if control != "P_COORD" else ():
                current = qperm[list(ids)].tolist()
                refids = source[list(ids)].tolist()
                components.append({"atom_indices": list(ids), "source_query_indices": list(ids),
                                   "query_indices": current, "query_rc": [[j // 4, j % 4] for j in current],
                                   "source_reference_indices": refids, "reference_indices": refids,
                                   "reference_rc": [[j // 4, j % 4] for j in refids]})
            families.append({"components": components})
            residual.append((query - reference[source]).square())
            refs.append(reference.clone())
            masks.append(torch.ones(len(reference), dtype=torch.bool))
        episode["axes"][control] = axes
        episode["families"][control] = families
        episode["residual_squared"][control] = torch.stack(residual)
        episode["valid"][control] = torch.ones((3, 16), dtype=torch.bool)
        episode["reference_tokens"][control] = refs
        episode["reference_valid"][control] = masks
    return episode


def reordered(episode, torch):
    output = copy.deepcopy(episode)
    order = [2, 0, 1]
    output["candidate_keys"] = tuple(episode["candidate_keys"][i] for i in order)
    for control in ("REAL", "C_BIND", "P_COORD"):
        for field in ("axes", "families", "reference_tokens", "reference_valid"):
            output[field][control] = [copy.deepcopy(episode[field][control][i]) for i in order]
        for field in ("residual_squared", "valid"):
            output[field][control] = episode[field][control][order].clone()
    return output


def full_vectors(result, core):
    output = {}
    for arm in core.ARM_NAMES:
        data = result["arms"][arm]
        for field in ("decisions", "c_bind_decisions", "p_coord_decisions"):
            for index, decision in enumerate(data[field]):
                output[f"{arm}/{field}/{index}"] = [float(x).hex() for x in decision.complete_seed_scores.detach()]
        if arm == core.ARM_QUERY_ONLY:
            for control, vector in data["query_selector_complete_scores"].items():
                output[f"{arm}/query_selector/{control}"] = [float(x).hex() for x in vector.detach()]
    return output


def run():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(17)
    initial_source_hashes = {path: sha(path) for path in (CORE, PLAN, Path(__file__))}
    core = load_core()
    checks, evidence = {}, {}

    def checked(name, call):
        call()
        checks[name] = True

    def pooling_oracle():
        residual = torch.zeros((6, 128), dtype=torch.float64)
        residual[:, :4] = torch.tensor([[0.5, 0.25, 1.0, 0.0], [1.0, 0.0, 0.25, 0.5],
                                       [0.25, 1.0, 0.0, 0.5], [0.5, 0.5, 0.5, 0.5],
                                       [0.0, 0.25, 1.0, 0.5], [0.25, 0.5, 0.0, 1.0]], dtype=torch.float64)
        groups = torch.tensor([8, 2, 8, 4, 2, 4])
        expected = numpy_descriptor(residual.numpy(), groups.numpy(), np)
        actual = core.grouped_descriptor(residual, groups)
        need(np.array_equal(actual.numpy(), expected), "GROUPED_DESCRIPTOR_ORACLE_MISMATCH")
        order = [5, 3, 1, 4, 0, 2]
        need(torch.equal(actual, core.grouped_descriptor(residual[order], groups[order])), "ATOM_ORDER_CHANGED_DESCRIPTOR")
        duplicate = core.grouped_descriptor(torch.cat((residual, residual[[0, 1, 0]])),
                                            torch.cat((groups, groups[[0, 1, 0]])))
        need(torch.equal(actual, duplicate), "EXACT_GROUP_DUPLICATE_NOT_IDEMPOTENT")
        dominated = torch.zeros((1, 128), dtype=torch.float64)
        need(torch.equal(actual, core.grouped_descriptor(torch.cat((residual, dominated)),
                         torch.cat((groups, torch.tensor([2]))))), "DOMINATED_OBSERVATION_INFORMATION_LOSS_COUNTEREXAMPLE_MISSING")
    checked("independent_grouping_oracle_order_and_duplicate_idempotence", pooling_oracle)

    def metric_domain():
        head = core.ResidualMetric()
        weights = head.weights()
        need(torch.equal(weights, torch.full((256,), 0.5, dtype=torch.float64)), "ZERO_HEAD_NOT_UNIFORM")
        lo = float(head(torch.full((256,), 4.0, dtype=torch.float64)).detach())
        hi = float(head(torch.zeros(256, dtype=torch.float64)).detach())
        need(lo == -255.0 and hi == 1.0 and core.H0_SCORE < lo, "SCORE_BOUNDS_OR_H0_ORDER_WRONG")
        q = torch.zeros(128, dtype=torch.float64); q[0] = 1.0
        r = torch.zeros_like(q); r[0], r[1] = 0.6, 0.8
        desc = core.grouped_descriptor((q-r).square()[None], torch.tensor([3]))
        need(abs(float(head(desc).detach()) - float(q @ r)) <= 2e-15, "SINGLE_ATOM_NOT_COSINE_AT_INITIALIZATION")
        for value in (-0.01, 4.1, float("nan")):
            rejects(lambda: core.grouped_descriptor(torch.full((1, 128), value, dtype=torch.float64), torch.tensor([0])),
                    "RESIDUAL_DOMAIN_INVALID")
        rejects(lambda: core.grouped_descriptor(torch.zeros((1, 128), dtype=torch.float32), torch.tensor([0])),
                "RESIDUAL_FP64_AXIS_INVALID")
        rejects(lambda: core.grouped_descriptor(torch.zeros((0, 128), dtype=torch.float64), torch.empty(0, dtype=torch.long)),
                "EMPTY_OR_INVALID_REFERENCE_GROUP")
        with torch.no_grad():
            head.theta[0] = 1000.0
        rejects(head.weights, "METRIC_WEIGHT_UNDERFLOW_OR_NONFINITE")
        evidence["mathematical_nonempty_range"] = [lo, hi]
    checked("fp64_metric_bounds_initial_cosine_and_invalid_domain_rejection", metric_domain)

    def limitations():
        head = core.ResidualMetric()
        residual = torch.zeros((2, 128), dtype=torch.float64)
        residual[:, 0] = torch.tensor([1.0, 0.5])
        before = core.grouped_descriptor(residual, torch.tensor([0, 1]))
        larger = residual.clone(); larger[1, 0] = 2.0
        after = core.grouped_descriptor(larger, torch.tensor([0, 1]))
        need(bool((after >= before).all()) and float(head(after).detach()) <= float(head(before).detach()),
             "LARGER_FIXED_GROUP_RESIDUAL_IMPROVED_SCORE")
        addition = core.grouped_descriptor(torch.cat((residual, torch.zeros((1, 128), dtype=torch.float64))), torch.tensor([0, 1, 2]))
        need(torch.equal(addition[128:], before[128:]) and float(head(addition).detach()) > float(head(before).detach()),
             "LOW_NEW_GROUP_DILUTION_COUNTEREXAMPLE_MISSING")
        original_score = float(head(before).detach())
        with torch.no_grad():
            head.theta[0] = -20.0; head.theta[128] = -20.0
        attenuated_score = float(head(before).detach())
        need(attenuated_score > original_score + 0.1, "LEARNED_WEIGHT_ATTENUATION_COUNTEREXAMPLE_MISSING")
        evidence["explicit_limitations"] = {"new_low_cost_group_can_improve_score": True,
            "learned_weights_can_attenuate_residual_contribution": True,
            "descriptor_preserves_channel_maxima_not_all_observations": True,
            "score_before_weight_attenuation": original_score, "score_after_weight_attenuation": attenuated_score}
    checked("fixed_weight_monotonicity_and_explicit_non_guarantee_counterexamples", limitations)

    def appearance_oracle():
        rng = np.random.default_rng(7021)
        q = rng.normal(size=(4, 128)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        r = rng.normal(size=(9, 128)); r /= np.linalg.norm(r, axis=1, keepdims=True)
        r[1] = q[0]; r[2] = q[0]; r[8] = q[1]
        valid = np.ones(9, dtype=bool); valid[8] = False
        theta = np.linspace(-0.2, 0.2, 256)
        head = core.ResidualMetric()
        with torch.no_grad(): head.theta.copy_(torch.from_numpy(theta))
        weights = head.weights()
        expected, indices = numpy_appearance(q, r, valid, numpy_weights(theta, np), np)
        selected, actual_indices = core.appearance_assignment(torch.from_numpy(q), torch.from_numpy(r), torch.from_numpy(valid), weights)
        need(np.array_equal(actual_indices.numpy(), indices) and int(actual_indices[0]) == 1,
             "FULL_REFERENCE_ARGMIN_OR_CANONICAL_TIE_MISMATCH")
        need(np.array_equal(selected.numpy(), expected), "APPEARANCE_SELECTED_RESIDUAL_MISMATCH")
        need(not selected.requires_grad, "DISCRETE_ASSIGNMENT_FEATURES_CARRY_GRADIENT")
    checked("independent_complete_reference_appearance_argmin_and_source_ties", appearance_oracle)

    episode = fixture(torch, np)
    context = core.build_context(episode)
    heads = core.GroupedResidualHeads()
    result = core.score_episode(context, heads)

    def context_oracle():
        expected_axis = independent_four_axis(4, 4)
        need(context.query_axis_current.tolist() == [list(g) for g in expected_axis], "COMPLETE_CONNECTED_FOUR_AXIS_MISMATCH")
        q = context.query_tokens.numpy()
        variation = (q - q.mean(axis=0)) ** 2
        weights = np.full(256, 0.5)
        for name in core.CONTROL_NAMES:
            control = context.controls[name]
            permutation = control.query_permutation.numpy()
            inverse = np.array([list(permutation).index(i) for i in range(16)])
            source_axis = inverse[np.array(expected_axis)]
            need(np.array_equal(source_axis, control.query_axis_source.numpy()), "P_COORD_INVERSE_SOURCE_AXIS_MISMATCH")
            features = np.concatenate((variation[source_axis].mean(axis=1), variation[source_axis].max(axis=1)), axis=1)
            need(np.allclose(features, control.query_selector_features.numpy(), rtol=0, atol=4e-17), "QUERY_VARIATION_DESCRIPTOR_MISMATCH")
            scores = np.sum(features * weights, axis=1)
            ordinal = int(np.argmax(scores))
            arm = result["arms"][core.ARM_QUERY_ONLY]
            need(arm["query_selector_map_ordinal"][name] == ordinal and
                 arm["query_selector_selected_atom_indices"][name] == source_axis[ordinal].tolist(),
                 "QUERY_ONLY_ORACLE_SELECTION_MISMATCH")
            for i in range(3):
                residual, groups = numpy_appearance(q[source_axis[ordinal]], control.reference_tokens[i].numpy(),
                    control.reference_valid[i].numpy(), weights, np)
                descriptor = numpy_descriptor(residual, groups, np)
                expected_score = 1.0 - 0.5 * np.sum(descriptor * weights)
                field = {"REAL": "scores", "C_BIND": "c_bind_scores", "P_COORD": "p_coord_scores"}[name]
                need(abs(float(arm[field][i].detach()) - float(expected_score)) <= 3e-15,
                     "QUERY_ONLY_COMPLETE_READOUT_ORACLE_MISMATCH")
        need(result["arms"][core.ARM_QUERY_ONLY]["query_selector_selected_atom_indices"]["REAL"] !=
             result["arms"][core.ARM_QUERY_ONLY]["query_selector_selected_atom_indices"]["P_COORD"],
             "FIXTURE_DOES_NOT_EXERCISE_CHANGED_P_COORD_SUPPORT")
        evidence["synthetic_complete_query_family_count"] = len(expected_axis)
    checked("complete_query_axis_variation_selection_and_p_coord_inverse_oracle", context_oracle)

    def invariances():
        reordered_context = core.build_context(reordered(episode, torch))
        need(core.serialize_episode_scores(result) == core.serialize_episode_scores(core.score_episode(reordered_context, heads)),
             "CANDIDATE_REORDER_CHANGED_CANONICAL_RESULT")
        need(full_vectors(result, core) == full_vectors(core.score_episode(reordered_context, heads), core),
             "CANDIDATE_REORDER_CHANGED_COMPLETE_VECTORS")
        need(torch.equal(result["arms"][core.ARM_ALLPATCH]["scores"], result["arms"][core.ARM_ALLPATCH]["p_coord_scores"]),
             "ALLPATCH_P_COORD_NOT_BIT_EXACT")
        for name in core.ARM_NAMES:
            data = result["arms"][name]
            need(bool(torch.isfinite(data["scores"]).all()), "NONFINITE_BRANCH_SCORE")
        real = result["arms"][core.ARM_REAL]
        need(any(float(d.score.detach()) < 0 and d.h1 for d in real["decisions"]), "NEGATIVE_H1_NOT_SELECTED")
        need(all(not d.h1 and not d.available and float(d.score.detach()) == -256.0 and
                 d.map_seed_ordinal is None and d.complete_seed_scores.numel() == 0 for d in real["p_coord_decisions"]),
             "STRUCTURAL_H0_SERIALIZATION_WRONG")
        need(float((real["p_coord_scores"][0] - real["p_coord_scores"][1]).detach()) == 0.0,
             "EMPTY_P_COORD_MARGIN_IS_NOT_ZERO")
        evidence["P_COORD_empty_family_margin_is_zero_not_independent_identity_evidence"] = True
    checked("whole_context_candidate_reorder_ap_p_coord_bits_negative_h1_and_structural_h0", invariances)

    def candidate_independence_and_rejections():
        changed = copy.deepcopy(episode)
        for name in core.CONTROL_NAMES:
            changed["reference_tokens"][name] = [-ref for ref in changed["reference_tokens"][name]]
            for i, ref in enumerate(changed["reference_tokens"][name]):
                indices = changed["axes"][name][i]["source_reference_indices"]
                changed["residual_squared"][name][i] = (changed["qtokens"]-ref[indices]).square()
        changed["source_receipt"]["content_intervention"] = "negated_unit_reference_vectors"
        altered = core.score_episode(core.build_context(changed), heads)
        a, b = result["arms"][core.ARM_QUERY_ONLY], altered["arms"][core.ARM_QUERY_ONLY]
        need(a["query_selector_selected_atom_indices"] == b["query_selector_selected_atom_indices"],
             "QUERY_SUPPORT_READ_CANDIDATE_CONTENT")
        need(not torch.equal(a["scores"], b["scores"]), "QUERY_ONLY_SCORES_IGNORE_REFERENCE_CONTENT")
        bad = copy.deepcopy(episode)
        bad["axes"]["P_COORD"][1]["query_indices"] = torch.arange(16)
        rejects(lambda: core.build_context(bad), "CANDIDATE_DEPENDENT_OR_INCOMPLETE_QUERY_COORDINATE_PERMUTATION")
        bad = copy.deepcopy(episode); bad["qtokens"][0] *= 2
        rejects(lambda: core.build_context(bad), "VALID_QUERY_TOKEN_NOT_UNIT_NORM")
        bad = copy.deepcopy(episode); bad["query_valid_axis"][0] = False
        rejects(lambda: core.build_context(bad), "V7_REQUIRES_GENUINE_DENSE_QUERY_GRID")
        bad = copy.deepcopy(episode); bad["reference_valid"]["REAL"][0].fill_(False)
        rejects(lambda: core.build_context(bad), "EMPTY_OR_MALFORMED_REFERENCE_ASSET")
    checked("query_proposal_candidate_independence_and_invalid_input_fail_closed", candidate_independence_and_rejections)

    def gradient_and_reload():
        need(sum(p.numel() for p in heads.parameters()) == 768, "THREE_EQUAL_256_PARAMETER_HEADS_REQUIRED")
        optimizers = {name: torch.optim.AdamW(head.parameters(), lr=0.03, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
                      for name, head in heads.arms().items()}
        scores = core.score_episode(context, heads)
        losses = []
        for index, name in enumerate(core.ARM_NAMES):
            arm = scores["arms"][name]
            losses.append((arm["scores"] * torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)).sum())
        sum(losses).backward()
        for name, head in heads.arms().items():
            need(head.theta.grad is not None and bool(torch.isfinite(head.theta.grad).all()) and
                 bool((head.theta.grad != 0).any()), "MISSING_NONZERO_FINITE_HEAD_GRADIENT:" + name)
            optimizers[name].step(); optimizers[name].zero_grad(set_to_none=True)
        before = {name: head.theta.detach().clone() for name, head in heads.arms().items()}
        sum(head.theta.sum() * 0.0 for head in heads.arms().values()).backward()
        for name, head in heads.arms().items():
            need(head.theta.grad is not None and torch.equal(head.theta.grad, torch.zeros_like(head.theta)),
                 "MASKED_EPISODE_GRADIENT_NOT_EXACT_ZERO")
            optimizers[name].step()
            need(not torch.equal(before[name], head.theta.detach()), "ZERO_CURRENT_GRADIENT_DID_NOT_EXERCISE_ADAM_MOMENTUM")
            need(float(optimizers[name].state[head.theta]["step"]) == 2.0, "MASKED_STEP_NOT_COUNTED")
        raw = core.deploy_episode(context, heads)
        serialized = core.serialize_episode_scores(raw)
        sealed_vectors = full_vectors(raw, core)
        need(json.loads(json.dumps(serialized, allow_nan=False)) == serialized, "SERIALIZATION_JSON_ROUNDTRIP_DRIFT")
        restored = core.GroupedResidualHeads()
        state = {name: [float(x).hex() for x in head.theta.detach()] for name, head in heads.arms().items()}
        state = json.loads(json.dumps(state))
        with torch.no_grad():
            for name, head in restored.arms().items():
                head.theta.copy_(torch.tensor([float.fromhex(x) for x in state[name]], dtype=torch.float64))
        replay = core.deploy_episode(context, restored)
        need(serialized == core.serialize_episode_scores(replay) and sealed_vectors == full_vectors(replay, core),
             "RELOADED_HEAD_DECISION_OR_COMPLETE_VECTOR_DRIFT")
        for arm in core.ARM_NAMES:
            for field in ("decisions", "c_bind_decisions", "p_coord_decisions"):
                for index, decision in enumerate(raw["arms"][arm][field]):
                    values = sealed_vectors[f"{arm}/{field}/{index}"]
                    tensor = torch.tensor([float.fromhex(x) for x in values], dtype=torch.float64)
                    need(core.tensor_seal(tensor) == core.tensor_seal(decision.complete_seed_scores),
                         "COMPLETE_VECTOR_BINARY64_REPLAY_DRIFT")
        evidence["head_parameter_counts"] = {name: head.theta.numel() for name, head in heads.arms().items()}
        evidence["masked_steps_have_zero_current_gradient_but_preserve_adam_momentum"] = True
        evidence["complete_vector_count"] = len(sealed_vectors)
        evidence["serialized_replay_sha256"] = hashlib.sha256(json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checked("three_arm_gradients_common_mask_zero_steps_and_binary64_deploy_reload", gradient_and_reload)
    need(all(sha(path) == digest for path, digest in initial_source_hashes.items()), "SOURCE_CHANGED_DURING_INDEPENDENT_E0")
    return {"schema_version": "rc_grouped_residual_e0_v7", "status": "RC_GROUPED_RESIDUAL_V7_INDEPENDENT_E0_PASS",
            "checks": checks, "check_count": len(checks), "evidence": evidence,
            "scope": "SYNTHETIC_ENGINEERING_ONLY_NO_NATURAL_TRAINING_OR_HYP_RESULT",
            "natural_payload_open_count": 0, "encoder_forward_count": 0, "formal392_consumed": False,
            "sources": {"core": {"path": str(CORE.relative_to(ROOT)), "sha256": sha(CORE)},
                        "validator": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(Path(__file__))},
                        "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": sha(PLAN)}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="Write the bound receipt after final source review")
    parser.add_argument("--out", type=Path, default=ROOT / "results/rc_grouped_residual_e0_v7/result.json")
    args = parser.parse_args()
    need(PLAN.is_file(), "V7_CONTRACT_NOT_YET_WRITTEN")
    result = run()
    if args.freeze:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out.exists():
            need(args.out.read_text() == payload, "EXISTING_E0_RECEIPT_DIFFERS_USE_NEW_OUTPUT_PATH")
        else:
            args.out.write_text(payload)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
