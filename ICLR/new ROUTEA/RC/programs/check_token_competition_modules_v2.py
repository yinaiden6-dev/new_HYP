#!/usr/bin/env python3
"""Short FP64 CPU contract checks; these are engineering, not research results."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import torch

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC / "src"))
from rc_aslo_xf.token_competition_v2 import (
    ARMS, TokenEvidence, TokenResidual, fixed_anchor_decision, new_model,
    parameter_report, shared_hit_miss_loss,
)


def same_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right))
    return left == right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=RC / "results/rc_token_competition_f128_v2/module_engineering.json")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(91)
    dtype = torch.float64
    b, c, n = 4, 7, 11
    q = torch.randn(b, c, n, 264, dtype=dtype)
    qm = torch.rand(b, c, n) > .2
    qm[0, 1] = False
    axis = torch.tensor([91, 14, 80, 51, 12, 18, 33])[None].expand(b, -1)
    ev = TokenEvidence(q, None, qm, None, axis)
    ev.validate()
    x = torch.randn(b, c, 18, dtype=dtype)
    anchor = torch.tensor([0, 2, 4, 1])
    truth = torch.zeros(b, c, dtype=torch.bool)
    truth[0, 3] = True
    truth[1, 2] = True
    truth[2, [1, 3]] = True
    # Row 3 is target absent; never assign a fabricated positive HOLD label.
    initial = torch.linspace(-.15, .15, 18, dtype=dtype)
    config = {"lr": .007, "weight_decay": .001, "pair_chunk_size": 3}
    models, optimizers = {}, {}
    for arm in ARMS:
        models[arm], optimizers[arm] = new_model(arm, 41, config)
        with torch.no_grad():
            models[arm].head.weight.copy_(initial)
    checks, details = {}, {}
    report = parameter_report(models)
    assert report["all_under_250k"] and report["reader_arms_within_10_percent_of_mean"]
    assert report["anchor_multi_identical_parameter_shapes"]
    assert same_tree(models[ARMS[1]].state_dict(), models[ARMS[2]].state_dict())
    checks["matched_capacity_and_identical_anchor_multi_initialization"] = True
    anchor_mask = torch.arange(c)[None] == anchor[:, None]
    base = torch.where(anchor_mask, 0.0, models[ARMS[0]].head(x).squeeze(-1))
    for arm, model in models.items():
        z, trace = model.forward_with_trace(x, ev, anchor)
        assert torch.equal(z, base) and torch.count_nonzero(trace["residual"]) == 0
        assert not model.head.weight.requires_grad
        optimizer_ids = {id(p) for group in optimizers[arm].param_groups for p in group["params"]}
        assert optimizer_ids == {id(p) for p in model.reader.parameters()}
        with torch.no_grad():
            fast_z, fast_trace = model.forward_with_trace(x, ev, anchor)
            assert torch.equal(fast_z, z)
            assert same_tree(fast_trace, trace)
    checks["zero_residual_exact_frozen_base_and_reader_only_optimizer"] = True
    checks["epoch_zero_no_grad_fast_path_matches_full_trace"] = True

    # Physical IDs, not slots, break exact frozen-base ties.
    tie_base = torch.zeros(1, c, dtype=dtype)
    tie_anchor = torch.tensor([0])
    rivals, valid = TokenResidual.select_rivals(tie_base, tie_anchor, axis[:1], True)
    assert rivals[0, 0, 1].item() == 4  # physical 12 is first nonanchor.
    assert rivals[0, 4, 1].item() == 1  # exclude self: physical 14 is next.
    for g in range(c):
        edges = rivals[0, g, valid[0, g]].tolist()
        assert g not in edges and len(edges) == len(set(edges))
        if g != 0:
            assert len(edges) == 2 and edges[0] == 0
    checks["multi_excludes_self_and_anchor_from_extra_rival_and_deduplicates"] = True
    checks["baseline_ties_use_ascending_physical_id"] = True
    staged, _ = new_model("TOKEN_QRR_MULTI", 19, config)
    with torch.no_grad():
        staged.head.weight[0, 0] = 1
        xa = torch.zeros_like(x)
        xa[:, 3, 0] = 5
        _, stage_a = staged.forward_with_trace(xa, ev, anchor)
        xa[:, 3, 0] = 0
        xa[:, 5, 0] = 5
        _, stage_b = staged.forward_with_trace(xa, ev, anchor)
        assert stage_a["rivals"][0, 0, 1].item() == 3
        assert stage_b["rivals"][0, 0, 1].item() == 5
    checks["rivals_use_current_forward_frozen_head_x0"] = True
    r2, v2 = TokenResidual.select_rivals(torch.zeros(1, 2), torch.tensor([0]),
                                       torch.tensor([[9, 3]]), True)
    assert v2.sum(-1).tolist() == [[1, 1]]
    checks["two_candidate_fallback"] = True

    for arm, model in models.items():
        optimizer = optimizers[arm]
        losses, output_grad, stem_grad = [], [], []
        observed_active = {name: False for name, _ in model.reader.named_parameters()}
        for _ in range(14):
            optimizer.zero_grad(set_to_none=True)
            loss = shared_hit_miss_loss(model(x, ev, anchor), truth, anchor)
            losses.append(float(loss.detach()))
            loss.backward()
            output_grad.append(float(model.reader.output.weight.grad.norm()))
            stem_grad.append(float(model.reader.encoder.query_stem[0].weight.grad.norm()))
            for name, p in model.reader.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), (arm, name)
                observed_active[name] |= bool(torch.count_nonzero(p.grad))
            assert model.head.weight.grad is None
            optimizer.step()
        assert output_grad[0] > 0 and max(stem_grad[1:]) > 0
        assert losses[-1] < losses[0], (arm, losses)
        assert all(observed_active.values()), (arm, observed_active)
        assert torch.equal(model.head.weight.detach().reshape(-1), initial)
        details[arm] = {"initial_loss": losses[0], "final_loss": losses[-1],
                        "first_output_gradient_norm": output_grad[0],
                        "max_stem_gradient_norm_after_first_update": max(stem_grad[1:]),
                        "all_reader_parameter_tensors_active": all(observed_active.values())}
        checks[arm + "_actual_gradients_and_learning"] = True
        with torch.no_grad():
            z, trace = model.forward_with_trace(x, ev, anchor)
            assert torch.count_nonzero(z[anchor_mask]) == 0
            assert torch.count_nonzero(trace["residual"][anchor_mask]) == 0
            torch.testing.assert_close(z, trace["base_logits"] + trace["residual"], rtol=0, atol=0)
            enc = model.reader.encode(ev)
            assert enc.query_tokens.shape == (b, c, n, 24)
            assert torch.count_nonzero(enc.query_tokens[0, 1]) == 0
            left = torch.arange(c)[None].expand(b, -1)
            right = left.roll(1, 1)
            ab = model.reader.pair_difference(enc, left, right)
            ba = model.reader.pair_difference(enc, right, left)
            assert torch.equal(ab, -ba)
            assert torch.count_nonzero(model.reader.pair_difference(enc, left, left)) == 0
            checks[arm + "_exact_antisymmetry_and_anchor_zero"] = True

            perm = torch.tensor([3, 0, 6, 2, 5, 1, 4])
            inv = torch.argsort(perm)
            evp = TokenEvidence(q[:, perm], None, qm[:, perm], None, axis[:, perm])
            zp = model(x[:, perm], evp, inv[anchor])
            torch.testing.assert_close(zp, z[:, perm], rtol=0, atol=2e-13)
            # Exact ties in frozen base preserve physical rival assignments too.
            zt, tt = model.forward_with_trace(torch.zeros_like(x), ev, anchor)
            ztp, ttp = model.forward_with_trace(torch.zeros_like(x), evp, inv[anchor])
            torch.testing.assert_close(ztp, zt[:, perm], rtol=0, atol=2e-13)
            original_ids = axis.gather(1, tt["rivals"].reshape(b, -1)).reshape(b, c, 2)
            permuted_ids = axis[:, perm].gather(1, ttp["rivals"].reshape(b, -1)).reshape(b, c, 2)
            assert torch.equal(permuted_ids, original_ids[:, perm])
            checks[arm + "_candidate_permutation_equivariance_with_ties"] = True

            qdirty = q.clone()
            qdirty[~qm] = torch.nan
            dirty = TokenEvidence(qdirty, torch.full((b, c, 1, 1), float("nan")),
                                  qm, None, axis)
            zd = model(x, dirty, anchor)
            torch.testing.assert_close(zd, z, rtol=0, atol=0)
            checks[arm + "_padding_and_unused_reference_do_not_participate"] = True
            for chunk in (1, 16, 32):
                torch.testing.assert_close(model(x, ev, anchor, chunk), z, rtol=0, atol=2e-13)
            checks[arm + "_chunk_size_invariance"] = True
            token_perm = torch.randperm(n)
            evt = TokenEvidence(q[:, :, token_perm], None, qm[:, :, token_perm], None, axis)
            torch.testing.assert_close(model(x, evt, anchor), z, rtol=0, atol=2e-13)
            checks[arm + "_joint_query_token_permutation_invariance"] = True
            if arm != "TOKEN_QR":
                scrambled = q.clone()
                scrambled[:, 3] = q[:, 3, token_perm]
                smask = qm.clone()
                smask[:, 3] = qm[:, 3, token_perm]
                broken_alignment = TokenEvidence(scrambled, None, smask, None, axis)
                zs = model(x, broken_alignment, anchor)
                assert not torch.allclose(zs, z, rtol=1e-8, atol=1e-10)
                checks[arm + "_sensitive_to_cross_candidate_token_alignment"] = True
            if arm == "TOKEN_QR":
                qalt = q.clone()
                qalt[:, 5] *= -3
                alt = model.reader.independent_score(model.reader.encode(
                    TokenEvidence(qalt, None, qm, None, axis)))
                original_d = model.reader.independent_score(enc)
                torch.testing.assert_close(alt[:, :5], original_d[:, :5], rtol=0, atol=0)
                checks["QR_independent_candidate_readout"] = True

        # Checkpoint the state AND Adam moments, then compare the next real step.
        stream = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=False)
        resumed, resumed_optimizer = new_model(arm, 777, config)
        resumed.load_state_dict(saved["model"])
        resumed_optimizer.load_state_dict(saved["optimizer"])
        for instance, opt in ((model, optimizer), (resumed, resumed_optimizer)):
            opt.zero_grad(set_to_none=True)
            shared_hit_miss_loss(instance(x, ev, anchor), truth, anchor).backward()
            opt.step()
        assert same_tree(model.state_dict(), resumed.state_dict())
        assert same_tree(optimizer.state_dict(), resumed_optimizer.state_dict())
        checks[arm + "_state_and_optimizer_exact_resume"] = True

        # Activation checkpointing changes storage only, not values or gradients.
        plain, _ = new_model(arm, 777, {**config, "checkpoint_chunks": False})
        plain.load_state_dict(model.state_dict())
        for instance in (model, plain):
            instance.zero_grad(set_to_none=True)
            shared_hit_miss_loss(instance(x, ev, anchor), truth, anchor).backward()
        for (name, p), (other_name, other) in zip(model.named_parameters(), plain.named_parameters()):
            assert name == other_name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, other.grad, rtol=0, atol=0)
        checks[arm + "_activation_checkpoint_exact_gradients"] = True

    # Every one of 127 challengers can win even with all evidence invalid. No
    # rival, support, confidence, or missing-evidence candidate mask is allowed.
    full_c = 128
    full_ev = TokenEvidence(torch.zeros(1, full_c, 2, 264, dtype=dtype), None,
                            torch.zeros(1, full_c, 2, dtype=torch.bool), None,
                            torch.arange(1000, 1000 + full_c)[None])
    full_anchor = torch.tensor([0])
    for arm in ARMS:
        model, _ = new_model(arm, 9, config)
        with torch.no_grad():
            model.head.weight[0, 0] = 1
            model.reader.output.weight.fill_(.05)  # exercise the full reader path
            for candidate in range(1, full_c):
                xf = torch.zeros(1, full_c, 18, dtype=dtype)
                xf[0, candidate, 0] = 2.0
                scores = model(xf, full_ev, full_anchor, 32)
                assert scores.shape == (1, 128) and torch.isfinite(scores).all()
                choice = fixed_anchor_decision(scores, full_anchor,
                                               candidate_keys=full_ev.physical_axis)
                assert choice.item() == candidate
        checks[arm + "_all_C128_candidates_scored_and_selectable_without_support_mask"] = True

    # Same loss and action implementation as V1, including identity probability
    # sum, miss softplus and strict HOLD on threshold/highest-challenger ties.
    hit_loss = shared_hit_miss_loss(base, truth, anchor, reduction="none")
    expected = -base[2].softmax(-1)[truth[2]].sum().log()
    torch.testing.assert_close(hit_loss[2], expected)
    expected_miss = torch.nn.functional.softplus(base[3, ~anchor_mask[3]]).mean()
    torch.testing.assert_close(hit_loss[3], expected_miss)
    for values in ([0., 2., 2.], [0., 0., -1.]):
        assert fixed_anchor_decision(torch.tensor([values]), torch.tensor([0])).item() == 0
    checks["shared_identity_hit_miss_loss_and_strict_fixed_anchor_action"] = True

    sources = [Path(__file__).resolve(), RC / "src/rc_aslo_xf/token_competition_v2.py"]
    payload = {"status": "TOKEN_COMPETITION_V2_ENGINEERING_PASS", "checks": checks,
               "parameter_report": report, "synthetic_fit": details,
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               "torch_version": torch.__version__, "dtype": "torch.float64",
               "elapsed_seconds": time.monotonic() - started,
               "evidence_level": "synthetic CPU engineering checks only; no scientific result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
