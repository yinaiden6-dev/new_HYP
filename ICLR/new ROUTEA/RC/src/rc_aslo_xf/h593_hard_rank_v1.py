"""Six-weight conditional ranking against the strongest wrong challenger.

Each TRAIN row contains 127 challenger vectors, with -1 denoting a correct
RAW winner or [0,126] denoting the target challenger.  Target-absent rows must
already be excluded by the caller.  The numerator uses only target-challenger
rows, while its denominator remains ALL target-present training rows.

The only change from conditional-rank CE is its ranking objective:
softplus(max_wrong_logit - target_logit).  The max is torch.amax, whose
gradient is shared equally by tied wrong challengers.  The target is masked
out before that maximum.  No bias, feature transform, dataset access, output
writing, or evaluation takes place here; thread settings belong to the runner.
"""
import torch


FEATURES = 6
CHALLENGERS = 127
STEPS = 2000
LEARNING_RATE = 0.03
WEIGHT_DECAY = 0.001


def _validate_labels(y, n, device):
    if not isinstance(y, torch.Tensor) or y.dtype != torch.long:
        raise ValueError('y must be a torch.long tensor')
    if y.shape != (n,) or y.device != device:
        raise ValueError('y must have one label per input row on the same device')
    if n == 0 or bool(((y < -1) | (y >= CHALLENGERS)).any()):
        raise ValueError('nonempty target-present rows require labels in [-1, 126]')


def _target_mask(z, targets):
    mask = torch.zeros_like(z, dtype=torch.bool)
    return mask.scatter_(1, targets[:, None], True)


def _hard_numerator(z, targets, mask):
    target = z.gather(1, targets[:, None]).squeeze(1)
    # amax shares the backward gradient across ALL tied maxima; max(dim=...)
    # would send it only to the first tied candidate and change the experiment.
    strongest_wrong = z.masked_fill(mask, -torch.inf).amax(dim=1)
    margin = strongest_wrong - target
    return torch.logaddexp(torch.zeros_like(margin), margin).sum()


def hard_loss(z, y):
    """Sum strongest-wrong softplus losses divided by all present rows."""
    if not isinstance(z, torch.Tensor) or z.dtype != torch.float64:
        raise ValueError('z must be a torch.float64 tensor')
    if z.ndim != 2 or z.shape[1] != CHALLENGERS:
        raise ValueError('z must have shape [N,127]')
    _validate_labels(y, z.shape[0], z.device)
    if not bool(torch.isfinite(z).all()):
        raise ValueError('z must be finite')
    positive = y >= 0
    if not bool(positive.any()):
        return (z * 0.0).sum()
    selected, targets = z[positive], y[positive]
    return _hard_numerator(selected, targets, _target_mask(selected, targets)) / z.shape[0]


def fit_rank(x, y):
    """Fit six zero-initialized weights with 2000 fixed AdamW updates."""
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float64:
        raise ValueError('x must be a torch.float64 tensor')
    if x.ndim != 3 or x.shape[1:] != (CHALLENGERS, FEATURES):
        raise ValueError('x must have shape [N,127,6]')
    _validate_labels(y, x.shape[0], x.device)
    if not bool(torch.isfinite(x).all()):
        raise ValueError('x must be finite')
    theta = torch.zeros(FEATURES, dtype=torch.float64, device=x.device, requires_grad=True)
    positive = y >= 0
    if not bool(positive.any()):
        return theta.detach()
    features = x[positive].detach()
    targets = y[positive]
    denominator = x.shape[0]
    mask = _target_mask(features[..., 0], targets)
    optimizer = torch.optim.AdamW([theta], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                                  betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
                                  foreach=False, fused=False)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = _hard_numerator(features @ theta, targets, mask) / denominator
        loss.backward()
        optimizer.step()
    if not bool(torch.isfinite(theta).all()):
        raise RuntimeError('nonfinite fitted strongest-wrong rank weights')
    return theta.detach()


def self_test():
    """Synthetic objective, tie-gradient, masking, and learning checks only."""
    generator = torch.Generator().manual_seed(20260921)
    z = torch.randn((4, CHALLENGERS), generator=generator, dtype=torch.float64,
                    requires_grad=True)
    y = torch.tensor([-1, 0, 126, -1], dtype=torch.long)
    actual = hard_loss(z, y)
    # Independent explicit omission rather than reusing the target mask.
    terms = []
    for i in (1, 2):
        t = int(y[i])
        wrong = torch.cat((z[i, :t], z[i, t + 1:]))
        margin = wrong.amax() - z[i, t]
        terms.append(torch.logaddexp(torch.zeros_like(margin), margin))
    expected = sum(terms) / len(y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.stack(terms).mean() * 0.5, rtol=0, atol=0)
    shifts = torch.tensor([3., -2., .5, 11.], dtype=torch.float64)[:, None]
    torch.testing.assert_close(actual, hard_loss(z + shifts, y), rtol=1e-13, atol=1e-13)
    gradient = torch.autograd.grad(actual, z)[0]
    assert bool((gradient[y == -1] == 0.0).all())
    # All 126 wrongs tie at zero.  At margin=0 each receives (1/2)/126/N,
    # while the target receives -1/(2N), even if it is the first/last index.
    tied = torch.zeros((3, CHALLENGERS), dtype=torch.float64, requires_grad=True)
    tied_y = torch.tensor([0, 126, -1], dtype=torch.long)
    tie_gradient = torch.autograd.grad(hard_loss(tied, tied_y), tied)[0]
    intended = torch.full_like(tied, 0.5 / (126 * 3))
    intended[0, 0] = intended[1, 126] = -0.5 / 3
    intended[2] = 0.0
    torch.testing.assert_close(tie_gradient, intended, rtol=1e-14, atol=1e-17)
    # Omitting the target mask here would return softplus(0), not softplus(-4).
    clear = torch.ones((1, CHALLENGERS), dtype=torch.float64)
    clear[0, 17] = 5.0
    torch.testing.assert_close(hard_loss(clear, torch.tensor([17])),
                               torch.logaddexp(torch.tensor(0., dtype=torch.float64),
                                               torch.tensor(-4., dtype=torch.float64)),
                               rtol=0, atol=0)
    extreme = torch.full((3, CHALLENGERS), 1000., dtype=torch.float64)
    extreme[0, 0] = -1000.
    extreme[1] = -1000.
    extreme[1, 126] = 1000.
    extreme.requires_grad_(True)
    extreme_loss = hard_loss(extreme, tied_y)
    extreme_gradient = torch.autograd.grad(extreme_loss, extreme)[0]
    assert torch.isfinite(extreme_loss) and bool(torch.isfinite(extreme_gradient).all())
    torch.testing.assert_close(extreme_loss, torch.tensor(2000. / 3, dtype=torch.float64),
                               rtol=0, atol=0)
    # A small shared direction raises both target candidates above all wrongs.
    features = torch.zeros((3, CHALLENGERS, FEATURES), dtype=torch.float64)
    features[0, 0, 0] = features[1, 126, 0] = 1.0
    features[2] = 100.0
    learned = fit_rank(features, tied_y)
    scores = features @ learned
    assert learned.shape == (FEATURES,) and not learned.requires_grad
    assert learned[0] > 0 and bool((learned[1:] == 0.0).all())
    assert torch.equal(scores[:2].argmax(dim=1), tied_y[:2])
    assert hard_loss(scores, tied_y) < hard_loss(torch.zeros_like(scores), tied_y) * .1
    no_positive = fit_rank(features, torch.full((3,), -1, dtype=torch.long))
    assert torch.equal(no_positive, torch.zeros(FEATURES, dtype=torch.float64))
    zero_z = z.detach().clone().requires_grad_(True)
    zero_loss = hard_loss(zero_z, torch.full((4,), -1, dtype=torch.long))
    assert zero_loss.item() == 0 and bool((torch.autograd.grad(zero_loss, zero_z)[0] == 0).all())
    print('H593_HARD_RANK_SYNTHETIC_SELFTEST_PASS', flush=True)


if __name__ == '__main__':
    self_test()
