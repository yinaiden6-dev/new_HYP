"""Full HOLD+127 decision learning with original six inputs and six squares.

No input rescaling, feature selection, new supervision, or post-fit gate.
The two losses and optimizer match the historical H593 FULL recipes.
"""
import torch
from torch.nn import functional as F

ARMS = ('DIAG_CE13', 'DIAG_COST113')


def expand(x):
    if x.dtype != torch.float64 or x.ndim != 3 or x.shape[1:] != (127, 6):
        raise ValueError('Expected FP64 [N,127,6]')
    phi = torch.cat((x, x * x), dim=-1)
    if not bool(torch.isfinite(phi).all()):
        raise ValueError('Nonfinite diagonal basis')
    return phi


def objective(z, y, arm):
    if arm not in ARMS or z.dtype != torch.float64 or z.shape != (len(y), 127):
        raise ValueError('Unknown objective or shape')
    if y.dtype != torch.long or not len(y) or bool(((y < -1) | (y > 126)).any()):
        raise ValueError('Targets must be -1 HOLD or a challenger index')
    if arm == 'DIAG_CE13':
        allz = torch.cat((z.new_zeros((len(z), 1)), z), dim=1)
        return (torch.logsumexp(allz, dim=1) - allz[torch.arange(len(y)), y + 1]).mean()
    raw = y == -1
    ii = torch.arange(len(y))
    idx = y.clamp_min(0)
    wrong = z.clone()
    wrong[ii[~raw], idx[~raw]] = -torch.inf
    return torch.where(raw, F.softplus(torch.amax(z, 1)),
        F.softplus(-z[ii, idx]) + F.softplus(torch.amax(wrong, 1))).mean()


def fit(x, y, arm):
    phi = expand(x.detach())
    theta = torch.nn.Parameter(torch.zeros(13, dtype=torch.float64))
    opt = torch.optim.AdamW([theta], lr=.03, weight_decay=.001)
    for _ in range(2000):
        opt.zero_grad()
        loss = objective(phi @ theta[:-1] + theta[-1], y, arm)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite fit loss')
        loss.backward()
        opt.step()
    if not bool(torch.isfinite(theta).all()):
        raise RuntimeError('Nonfinite fitted head')
    return theta.detach()
