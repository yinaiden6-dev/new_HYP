"""Fixed norm/phase interventions on RoMa's soft correspondence embedding."""
import torch

SEED = 20260924
DELTA = 0.125  # normalized [-1,1] coordinate displacement; frozen before outcomes
ARMS = ('NATIVE', 'ZERO', 'AMP_FLAT', 'AMP_PERMUTE',
        'GLOBAL_X_PLUS', 'GLOBAL_X_MINUS', 'LOCAL_X_PLUS', 'LOCAL_X_MINUS',
        'GLOBAL_Y_PLUS', 'GLOBAL_Y_MINUS', 'LOCAL_Y_PLUS', 'LOCAL_Y_MINUS')


def permutation(n, side):
    return torch.randperm(n, generator=torch.Generator().manual_seed(SEED + side))


def transform(p, omega, scale, arm, radius, side):
    assert arm in ARMS and p.shape[-1] == 2 * len(omega)
    if arm == 'NATIVE':
        return p, dict(kind='identity', norm_relative_error=0.)
    if arm == 'ZERO':
        return torch.zeros_like(p), dict(kind='zero')
    x = p.double()
    norm = x.norm(dim=-1, keepdim=True)
    assert float(norm.min()) > 0
    if arm.startswith('AMP_'):
        new_norm = torch.full_like(norm, radius) if arm == 'AMP_FLAT' else norm.reshape(-1)[permutation(norm.numel(), side).to(p.device)].reshape_as(norm)
        out = (x * new_norm / norm).to(p.dtype)
        direction_error = (out.double() / out.double().norm(dim=-1, keepdim=True) - x / norm).abs().max()
        assert float(direction_error) < 2e-6
        return out, dict(kind='amplitude', direction_max_error=float(direction_error),
                         native_norm_mean=float(norm.mean()), changed_norm_mean=float(new_norm.mean()))
    scope, axis, sign = arm.split('_')
    axis = {'X': 0, 'Y': 1}[axis]
    direction = 1. if sign == 'PLUS' else -1.
    shift = torch.ones(p.shape[:-1] + (1,), dtype=torch.float64, device=p.device) * DELTA * direction
    if scope == 'LOCAL':
        n = norm.numel()
        signs = torch.ones(n, dtype=torch.float64)
        signs[permutation(n, side)[:n // 2]] = -1.
        shift = shift * signs.reshape_as(shift).to(p.device)
    angle = shift * omega[:, axis].double().to(p.device) * float(scale)
    sin, cos = x.chunk(2, dim=-1)
    out = torch.cat((sin * angle.cos() + cos * angle.sin(),
                     cos * angle.cos() - sin * angle.sin()), dim=-1).to(p.dtype)
    osin, ocos = out.double().chunk(2, dim=-1)
    relative = (out.double().norm(dim=-1, keepdim=True) / norm - 1).abs().max()
    spectral = ((osin.square() + ocos.square()) - (sin.square() + cos.square())).abs().max()
    assert float(relative) < 2e-6 and float(spectral) < 2e-6
    return out, dict(kind='phase', norm_relative_error=float(relative),
                    spectral_power_max_abs_error=float(spectral),
                    perturbation_rms=float((out.double() - x).square().mean().sqrt()))


def self_test():
    import numpy as np
    g = torch.Generator().manual_seed(SEED)
    omega = torch.randn(16, 2, generator=g, dtype=torch.float64)
    coords = torch.randn(12, 2, generator=g, dtype=torch.float64)
    attention = torch.softmax(torch.randn(8, 12, generator=g, dtype=torch.float64), -1)
    phase = coords @ omega.T
    p = (attention @ torch.cat((phase.sin(), phase.cos()), -1)).reshape(1, 2, 4, 32)
    radius = float(p.norm(dim=-1).mean())
    outputs = {arm: transform(p, omega, 1., arm, radius, 0)[0] for arm in ARMS}
    errors = {}
    for axis in ('X', 'Y'):
        a = 0 if axis == 'X' else 1
        delta = np.zeros(2); delta[a] = DELTA
        shifted = (coords.numpy() + delta) @ omega.numpy().T
        expected = (attention.numpy() @ np.concatenate([np.sin(shifted), np.cos(shifted)], axis=-1)).reshape(p.shape)
        error = float(np.abs(expected - outputs[f'GLOBAL_{axis}_PLUS'].numpy()).max())
        assert error < 1e-12
        errors[axis + '_translation_equivalence'] = error
        glob = sum((outputs[f'GLOBAL_{axis}_{s}'] - p).square() for s in ('PLUS', 'MINUS'))
        local = sum((outputs[f'LOCAL_{axis}_{s}'] - p).square() for s in ('PLUS', 'MINUS'))
        assert torch.allclose(glob, local, atol=1e-12, rtol=0)
    assert torch.allclose(outputs['AMP_PERMUTE'].norm(dim=-1).flatten().sort().values,
                          p.norm(dim=-1).flatten().sort().values, atol=1e-12, rtol=0)
    assert torch.allclose(outputs['AMP_FLAT'].norm(dim=-1), torch.full(p.shape[:-1], radius, dtype=p.dtype), atol=1e-12, rtol=0)
    return dict(status='POSITION_FACTOR_ALGEBRA_PASS', independent_numpy_errors=errors,
                norm_and_spectral_power_preserved=True, antithetic_perturbation_dose_equal=True,
                amplitude_direction_preserved=True, delta=DELTA, arms=list(ARMS))
