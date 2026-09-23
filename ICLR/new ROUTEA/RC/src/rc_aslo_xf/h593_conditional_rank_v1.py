"""Six-weight challenger-rank training with the HOLD action term removed.

The caller supplies only target-present TRAIN rows.  y=-1 identifies a correct
RAW winner; y in [0,126] identifies the target among 127 challengers.  Rank
training omits RAW-correct rows from its numerator but keeps the complete
target-present row count as denominator, matching the rank component of the
original 128-way joint cross-entropy.  No action head or common bias is fit.

This module performs no data loading, output writing, candidate selection, or
natural evaluation.  Thread configuration belongs to the runner.
"""
import torch
from torch.nn import functional as F


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


def loss_terms(z, y):
    """Return (joint128CE, conditional_rank, action), each scalar binary64.

    For A=logsumexp(z), action is softplus(A) for RAW-correct rows and
    softplus(-A) for target-challenger rows.  joint128CE is evaluated directly
    rather than assigned rank+action, so decomposition checks are independent
    of the expression used to compute the two components.
    """
    if not isinstance(z, torch.Tensor) or z.dtype != torch.float64:
        raise ValueError('z must be a torch.float64 tensor')
    if z.ndim != 2 or z.shape[1] != CHALLENGERS:
        raise ValueError('z must have shape [N, 127]')
    _validate_labels(y, z.shape[0], z.device)
    if not bool(torch.isfinite(z).all()):
        raise ValueError('z must be finite')
    positive = y >= 0
    n = z.shape[0]
    if bool(positive.any()):
        rank = F.cross_entropy(z[positive], y[positive], reduction='sum') / n
    else:
        # A zero connected to z supports a meaningful zero-gradient check.
        rank = (z * 0.0).sum()
    aggregate = torch.logsumexp(z, dim=1)
    # logaddexp(0, a) is the stable softplus identity.  PyTorch softplus's
    # default threshold=20 approximation is too coarse for binary64 checks
    # of the joint-loss decomposition near that threshold.
    signed = torch.where(positive, -aggregate, aggregate)
    action = torch.logaddexp(torch.zeros_like(signed), signed).mean()
    full = torch.cat((z.new_zeros((n, 1)), z), dim=1)
    joint = F.cross_entropy(full, y + 1, reduction='mean')
    return joint, rank, action


def fit_rank(x, y):
    """Fit six zero-initialized weights for exactly 2000 AdamW updates.

    Loss = sum conditional CE over target-challenger rows / all present rows.
    RAW-correct rows exert no rank gradient.  A dataset with no positives
    returns the initialized zero vector without optimization.
    """
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float64:
        raise ValueError('x must be a torch.float64 tensor')
    if x.ndim != 3 or x.shape[1:] != (CHALLENGERS, FEATURES):
        raise ValueError('x must have shape [N, 127, 6]')
    _validate_labels(y, x.shape[0], x.device)
    if not bool(torch.isfinite(x).all()):
        raise ValueError('x must be finite')
    theta = torch.zeros(FEATURES, dtype=torch.float64, device=x.device, requires_grad=True)
    positive = y >= 0
    if not bool(positive.any()):
        return theta.detach()
    # Input features are frozen.  This also avoids accumulating gradients into
    # an accidentally differentiable cache tensor supplied by the caller.
    features = x[positive].detach()
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
        raise RuntimeError('nonfinite fitted rank weights')
    return theta.detach()


def self_test():
    """Synthetic value/gradient checks and a tiny known learnable problem."""
    generator = torch.Generator().manual_seed(20260921)
    z = torch.randn((4, CHALLENGERS), generator=generator, dtype=torch.float64,
                    requires_grad=True)
    y = torch.tensor([-1, 0, 126, -1], dtype=torch.long)
    joint, rank, action = loss_terms(z, y)
    torch.testing.assert_close(joint, rank + action, rtol=1e-13, atol=1e-13)
    gj = torch.autograd.grad(joint, z, retain_graph=True)[0]
    gr = torch.autograd.grad(rank, z, retain_graph=True)[0]
    ga = torch.autograd.grad(action, z)[0]
    torch.testing.assert_close(gj, gr + ga, rtol=1e-12, atol=1e-14)
    assert bool((gr[y == -1] == 0.0).all())
    expected = F.cross_entropy(z[y >= 0], y[y >= 0], reduction='sum') / len(y)
    torch.testing.assert_close(rank, expected, rtol=0.0, atol=0.0)
    positive_mean = F.cross_entropy(z[y >= 0], y[y >= 0], reduction='mean')
    torch.testing.assert_close(rank, positive_mean * 0.5, rtol=0.0, atol=0.0)
    # Every row may have a different common logit shift; challenger ordering
    # and conditional rank CE remain unchanged, unlike the HOLD action term.
    shift = torch.tensor([3.0, -2.0, 0.5, 11.0], dtype=torch.float64)[:, None]
    _, shifted_rank, _ = loss_terms(z.detach() + shift, y)
    torch.testing.assert_close(rank, shifted_rank, rtol=1e-13, atol=1e-13)
    extreme = z.detach() + torch.tensor([16.0, -26.0, 1000.0, -1000.0],
                                       dtype=torch.float64)[:, None]
    ej, er, ea = loss_terms(extreme, y)
    torch.testing.assert_close(ej, er + ea, rtol=1e-13, atol=1e-13)
    all_hold = torch.full((4,), -1, dtype=torch.long)
    zero_z = z.detach().clone().requires_grad_(True)
    _, zero_rank, _ = loss_terms(zero_z, all_hold)
    zero_grad = torch.autograd.grad(zero_rank, zero_z)[0]
    assert zero_rank.item() == 0.0 and bool((zero_grad == 0.0).all())
    # Denominator includes a RAW-correct row even if that row has arbitrarily
    # different features; its features cannot affect the learned rank head.
    small = torch.zeros((3, CHALLENGERS, FEATURES), dtype=torch.float64)
    small_y = torch.tensor([0, 126, -1], dtype=torch.long)
    small[0, 0, 0] = 1.0
    small[1, 126, 0] = 1.0
    small[2] = 100.0
    theta = fit_rank(small, small_y)
    assert theta.shape == (FEATURES,) and not theta.requires_grad
    assert theta[0].item() > 0.0 and bool((theta[1:] == 0.0).all())
    assert torch.equal(torch.argmax(small[:2] @ theta, dim=1), small_y[:2])
    initial_rank = loss_terms(small @ torch.zeros_like(theta), small_y)[1]
    trained_rank = loss_terms(small @ theta, small_y)[1]
    assert trained_rank < initial_rank * 0.1
    no_positive = fit_rank(small, torch.full((3,), -1, dtype=torch.long))
    assert torch.equal(no_positive, torch.zeros(FEATURES, dtype=torch.float64))
    print('H593_CONDITIONAL_RANK_SYNTHETIC_SELFTEST_PASS', flush=True)


if __name__ == '__main__':
    self_test()
