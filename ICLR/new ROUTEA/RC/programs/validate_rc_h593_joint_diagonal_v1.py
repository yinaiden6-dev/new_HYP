#!/usr/bin/env python3
"""Independent NumPy values, tie-aware gradients, logits and decisions."""
import numpy as np


def softplus_pair(v):
    # Match the legacy PyTorch default threshold=20, including its derivative.
    # Above 20 the implemented objective is exactly linear, not logaddexp.
    return (float(v), 1.) if v > 20. else (float(np.logaddexp(0., v)), float(np.exp(-np.logaddexp(0., -v))))


def basis(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 3 and x.shape[1:] == (127, 6) and np.isfinite(x).all()
    return np.concatenate((x, np.stack([x[..., i] ** 2 for i in range(6)], axis=-1)), axis=-1)


def values_gradient(theta, x, y, arm):
    phi = basis(x)
    theta = np.asarray(theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    assert theta.shape == (13,) and y.shape == (len(x),) and len(y)
    assert np.isin(y, np.arange(-1, 127)).all()
    z = np.sum(phi * theta[:-1], axis=-1) + theta[-1]
    dz = np.zeros_like(z)
    losses = []
    for i, (row, target) in enumerate(zip(z, y)):
        if arm == 'DIAG_CE13':
            allz = np.r_[0., row]
            peak = float(allz.max())
            ex = np.exp(allz - peak)
            losses.append(float(peak + np.log(ex.sum()) - allz[target + 1]))
            dz[i] = (ex / ex.sum())[1:]
            if target >= 0:
                dz[i, target] -= 1.
        else:
            assert arm == 'DIAG_COST113'
            mask = np.ones(127, dtype=bool)
            if target >= 0:
                mask[target] = False
            peak = float(row[mask].max())
            tied = mask & (row == peak)
            loss, derivative = softplus_pair(peak)
            dz[i, tied] = derivative / tied.sum()
            if target >= 0:
                value, derivative = softplus_pair(-row[target])
                loss += value
                dz[i, target] = -derivative
            losses.append(loss)
    dz /= len(y)
    gradient = np.r_[np.sum(phi * dz[..., None], axis=(0, 1)), dz.sum()]
    return float(np.mean(losses)), gradient


def logits_check(x, parameters, predictions, model):
    t = np.array([float.fromhex(v) for v in parameters])
    phi = basis(x) if len(t) == 13 else np.asarray(x, dtype=np.float64)
    z = np.sum(phi * t[:-1], axis=-1) + t[-1]
    saved = np.array([[float.fromhex(v) for v in r['models'][model]['logits_hex']] for r in predictions])
    error = float(np.max(abs(z - saved)))
    assert error < 2e-10
    for r, scores, wanted in zip(predictions, z, saved):
        top = int(scores.argmax())
        assert top == int(wanted.argmax()) and bool(scores[top] > 0) == bool(wanted[top] > 0)
        pos = r['challenger_positions'][top] if scores[top] > 0 else r['winner']
        assert r['candidate_physical_rows'][pos] == r['models'][model]['selected']
    return dict(logit_count=int(z.size), max_abs_error=error, decisions=len(predictions))


def self_test():
    import torch
    from rc_aslo_xf.h593_joint_diagonal_v1 import expand, objective
    # Compare the exact legacy objective programs before any natural fitting.
    from run_rc_six_cause_loss_binding_v1 import unit
    from run_rc_query_content_routing_oof4_v1 import objective as legacy_ce
    rng = np.random.default_rng(20260921)
    checks = 0
    for tied in (False, True):
        x = rng.normal(size=(4, 127, 6))
        y = np.array([-1, 0, 12, 126])
        t = np.zeros(13) if tied else rng.normal(size=13)
        xx = torch.tensor(x, dtype=torch.float64)
        yy = torch.tensor(y, dtype=torch.long)
        assert np.array_equal(expand(xx).numpy(), basis(x))
        for arm in ('DIAG_CE13', 'DIAG_COST113'):
            theta = torch.tensor(t, dtype=torch.float64, requires_grad=True)
            z = expand(xx) @ theta[:-1] + theta[-1]
            loss = objective(z, yy, arm)
            old = legacy_ce(z, yy + 1) if arm == 'DIAG_CE13' else unit(z, yy)
            assert torch.equal(loss, old)
            grad = torch.autograd.grad(loss, theta, retain_graph=True)[0]
            assert torch.equal(grad, torch.autograd.grad(old, theta)[0])
            independent, g = values_gradient(t, x, y, arm)
            assert abs(float(loss.detach()) - independent) < 2e-10
            assert np.max(abs(grad.numpy() - g)) < 2e-10
            checks += 4
    for value in (-100., 0., 20., 20.001, 100.):
        t = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        loss = torch.nn.functional.softplus(t)
        grad = torch.autograd.grad(loss, t)[0]
        v, g = softplus_pair(value)
        assert abs(float(loss.detach())-v) < 1e-14 and abs(float(grad)-g) < 1e-14
        checks += 1
    # HOLD zero wins all exact ties; first canonical challenger wins positive ties.
    p = [dict(winner=0, candidate_physical_rows=list(range(128)), challenger_positions=list(range(1,128)),
              models={'test':dict(logits_hex=[0.0.hex()]*127, selected=0)})]
    for bias, chosen in ((0., 0), (1., 1), (-1., 0)):
        t = [0.] * 12 + [bias]
        p[0]['models']['test'] = dict(logits_hex=[bias.hex()]*127, selected=chosen)
        logits_check(np.zeros((1,127,6)), [v.hex() for v in t], p, 'test')
        checks += 1
    return dict(status='JOINT_DIAGONAL_SYNTHETIC_PASS', checks=checks, natural_fits=0)
