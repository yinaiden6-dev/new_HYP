"""Conditional challenger-rank learning with fixed quadratic feature bases.

Both arms use only the original six binary64 inputs.  DIAG12 appends their
six squares; FULL_QUAD27 appends all 21 products x_i*x_j, i<=j, in
lexicographic (i,j) order.  There is no centering, normalization, clipping,
new evidence, common bias, or fitted feature transformation.

Labels are -1 for RAW-correct target-present rows, or a target index among
127 challengers.  Only target-challenger rows enter the rank numerator; the
denominator remains the count of ALL target-present training rows.  No
datasets, files, or evaluation labels are accessed by this module.
"""
import torch
from torch.nn import functional as F


FEATURES = 6
CHALLENGERS = 127
STEPS = 2000
LEARNING_RATE = 0.03
WEIGHT_DECAY = 0.001
MODES = ('DIAG12', 'FULL_QUAD27')
PAIRS = tuple((i, j) for i in range(FEATURES) for j in range(i, FEATURES))
DIMENSIONS = {'DIAG12': 12, 'FULL_QUAD27': 27}


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError('mode must be DIAG12 or FULL_QUAD27')


def expand(x, mode):
    """Expand a float64 tensor with final dimension six in a fixed order.

    Leading dimensions and the device are preserved.  The first six entries
    are exactly the supplied input entries.  Products are ordinary FP64
    multiplications; nonfinite inputs or overflowing products are rejected.
    """
    _check_mode(mode)
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float64:
        raise ValueError('x must be a torch.float64 tensor')
    if x.ndim < 1 or x.shape[-1] != FEATURES:
        raise ValueError('x must have final dimension six')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('x must be finite')
    if mode == 'DIAG12':
        products = x * x
    else:
        products = torch.stack([x[..., i] * x[..., j] for i, j in PAIRS], dim=-1)
    if not bool(torch.isfinite(products).all()):
        raise ValueError('quadratic product overflowed binary64')
    return torch.cat((x, products), dim=-1)


def _validate_labels(y, n, device):
    if not isinstance(y, torch.Tensor) or y.dtype != torch.long:
        raise ValueError('y must be a torch.long tensor')
    if y.shape != (n,) or y.device != device:
        raise ValueError('y must have one label per input row on the same device')
    if n == 0 or bool(((y < -1) | (y >= CHALLENGERS)).any()):
        raise ValueError('nonempty target-present rows require labels in [-1, 126]')


def conditional_loss(z, y):
    """Conditional 127-way CE numerator divided by all present rows."""
    if not isinstance(z, torch.Tensor) or z.dtype != torch.float64:
        raise ValueError('z must be a torch.float64 tensor')
    if z.ndim != 2 or z.shape[1] != CHALLENGERS or not bool(torch.isfinite(z).all()):
        raise ValueError('z must be a finite [N,127] tensor')
    _validate_labels(y, z.shape[0], z.device)
    positive = y >= 0
    if bool(positive.any()):
        return F.cross_entropy(z[positive], y[positive], reduction='sum') / z.shape[0]
    return (z * 0.0).sum()


def fit_rank(x, y, mode):
    """Fit a zero-initialized 12/27-weight rank head for exactly 2000 steps."""
    _check_mode(mode)
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float64:
        raise ValueError('x must be a torch.float64 tensor')
    if x.ndim != 3 or x.shape[1:] != (CHALLENGERS, FEATURES):
        raise ValueError('x must have shape [N,127,6]')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('x must be finite')
    _validate_labels(y, x.shape[0], x.device)
    theta = torch.zeros(DIMENSIONS[mode], dtype=torch.float64,
                        device=x.device, requires_grad=True)
    positive = y >= 0
    if not bool(positive.any()):
        return theta.detach()
    features = expand(x[positive].detach(), mode)
    targets = y[positive]
    denominator = x.shape[0]
    optimizer = torch.optim.AdamW([theta], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                                  betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
                                  foreach=False, fused=False)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(features @ theta, targets, reduction='sum') / denominator
        loss.backward()
        optimizer.step()
    if not bool(torch.isfinite(theta).all()):
        raise RuntimeError('nonfinite fitted quadratic rank weights')
    return theta.detach()


def self_test():
    """Only synthetic checks; no natural training or data access."""
    generator = torch.Generator().manual_seed(20260921)
    x = torch.randn((4, CHALLENGERS, FEATURES), generator=generator, dtype=torch.float64)
    diag, full = expand(x, 'DIAG12'), expand(x, 'FULL_QUAD27')
    assert diag.shape == (4, CHALLENGERS, 12) and full.shape == (4, CHALLENGERS, 27)
    assert torch.equal(diag[..., :6], x) and torch.equal(full[..., :6], x)
    for i in range(FEATURES):
        assert torch.equal(diag[..., 6 + i], x[..., i] * x[..., i])
    for k, (i, j) in enumerate(PAIRS):
        assert torch.equal(full[..., 6 + k], x[..., i] * x[..., j])
    assert PAIRS == tuple(sorted(PAIRS)) and len(PAIRS) == 21
    linear_theta = torch.randn(FEATURES, generator=generator, dtype=torch.float64)
    for basis, dimension in ((diag, 12), (full, 27)):
        embedded = torch.cat((linear_theta, torch.zeros(dimension - FEATURES, dtype=torch.float64)))
        # Different-width matrix products can use different reduction orders.
        torch.testing.assert_close(basis @ embedded, x @ linear_theta, rtol=1e-13, atol=1e-13)
    diag_theta = torch.randn(12, generator=generator, dtype=torch.float64)
    mapped = torch.zeros(27, dtype=torch.float64)
    mapped[:6] = diag_theta[:6]
    for i in range(FEATURES):
        mapped[6 + PAIRS.index((i, i))] = diag_theta[6 + i]
    torch.testing.assert_close(diag @ diag_theta, full @ mapped, rtol=1e-13, atol=1e-13)
    # Value, denominator, shift invariance, and gradient checks for both bases.
    y = torch.tensor([-1, 0, 126, -1], dtype=torch.long)
    for mode, basis in (('DIAG12', diag), ('FULL_QUAD27', full)):
        theta = torch.randn(DIMENSIONS[mode], generator=generator,
                            dtype=torch.float64, requires_grad=True)
        z = basis @ theta
        z.retain_grad()
        loss = conditional_loss(z, y)
        expected = F.cross_entropy(z[y >= 0], y[y >= 0], reduction='sum') / len(y)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        mean_positive = F.cross_entropy(z[y >= 0], y[y >= 0], reduction='mean')
        torch.testing.assert_close(loss, mean_positive / 2.0, rtol=0, atol=0)
        shifts = torch.tensor([3., -2., .5, 11.], dtype=torch.float64)[:, None]
        torch.testing.assert_close(loss, conditional_loss(z + shifts, y), rtol=1e-13, atol=1e-13)
        loss.backward()
        assert bool((z.grad[y == -1] == 0.0).all())
        probabilities = torch.softmax(z.detach()[y >= 0], dim=1)
        probabilities[torch.arange(2), y[y >= 0]] -= 1.0
        exact_grad = (basis[y >= 0] * probabilities[..., None]).sum(dim=(0, 1)) / len(y)
        torch.testing.assert_close(theta.grad, exact_grad, rtol=1e-12, atol=1e-13)
    # Two opposite corners of XOR are correct.  The original linear and
    # diagonal-square terms cannot jointly distinguish these two comparisons;
    # the x0*x1 interaction can do so with a shared positive coefficient.
    xor = torch.zeros((2, CHALLENGERS, FEATURES), dtype=torch.float64)
    xor[0, :, 0], xor[0, :, 1] = 1.0, -1.0
    xor[1, :, 0], xor[1, :, 1] = -1.0, 1.0
    xor[0, 0, 1], xor[1, 0, 1] = 1.0, -1.0
    xor_y = torch.zeros(2, dtype=torch.long)
    xor_theta = fit_rank(xor, xor_y, 'FULL_QUAD27')
    xor_scores = expand(xor, 'FULL_QUAD27') @ xor_theta
    assert torch.equal(xor_scores.argmax(dim=1), xor_y)
    assert xor_theta[6 + PAIRS.index((0, 1))] > 0
    assert conditional_loss(xor_scores, xor_y) < conditional_loss(torch.zeros_like(xor_scores), xor_y) * .1
    # The target at zero lies between equal numbers of +/-1 wrong points:
    # one negative square coefficient separates it, whereas a linear term
    # cannot rank it strictly above both extremes.
    bowl = torch.zeros((1, CHALLENGERS, FEATURES), dtype=torch.float64)
    bowl[0, 1:64, 0], bowl[0, 64:, 0] = 1.0, -1.0
    bowl_y = torch.zeros(1, dtype=torch.long)
    bowl_theta = fit_rank(bowl, bowl_y, 'DIAG12')
    bowl_scores = expand(bowl, 'DIAG12') @ bowl_theta
    assert bowl_scores.argmax(dim=1).item() == 0 and bowl_theta[6] < 0
    assert conditional_loss(bowl_scores, bowl_y) < conditional_loss(torch.zeros_like(bowl_scores), bowl_y) * .1
    for mode in MODES:
        empty = fit_rank(x, torch.full((4,), -1, dtype=torch.long), mode)
        assert not empty.requires_grad and torch.equal(empty, torch.zeros(DIMENSIONS[mode], dtype=torch.float64))
    print('H593_QUADRATIC_RANK_SYNTHETIC_SELFTEST_PASS', flush=True)


if __name__ == '__main__':
    self_test()
