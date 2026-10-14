"""RoMa-free reading of uncollapsed frozen ColNomic pair relations.

SPATIAL and SET have identical parameters. Only their message neighborhoods
differ. All tokens and all reference candidates remain in the calculation.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


TEMPERATURE = 0.1
WIDTH = 8
ARMS = ('BASE', 'COMPRESSED', 'SET', 'SPATIAL')


def relation(q, r):
    c = q @ r.T  # inputs are already normalized
    a = (c / TEMPERATURE).softmax(1)
    b = (c / TEMPERATURE).softmax(0)
    return c, a, b


def local_mean(x, grid, axis):
    """Mean over a 3x3 native patch neighborhood, including its center.

    Boundary normalization counts real cells, never wraps opposite borders.
    x is [query patches, reference patches, channels].
    """
    if axis == 1:
        return local_mean(x.transpose(0, 1), grid, 0).transpose(0, 1)
    h, w = map(int, grid)
    if h * w != x.shape[0]:
        raise ValueError('native grid must contain every image token')
    nr, channels = x.shape[1:]
    image = x.permute(1, 2, 0).reshape(nr, channels, h, w)
    pooled = F.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=False)
    return pooled.reshape(nr, channels, h * w).permute(2, 0, 1)


def uniform_mean(x, axis):
    return x.mean(axis, keepdim=True).expand_as(x)


def coordinates(grid, dtype, device):
    h, w = map(int, grid)
    yy, xx = torch.meshgrid((torch.arange(h, dtype=dtype, device=device) + .5) / h,
                           (torch.arange(w, dtype=dtype, device=device) + .5) / w,
                           indexing='ij')
    return torch.stack((xx.flatten(), yy.flatten()), -1)


class Reader(nn.Module):
    def __init__(self, arm, seed=17, width=WIDTH):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if arm == 'COMPRESSED':
                self.down = nn.Linear(520, 16)
                self.out = nn.Linear(16, 1)
            elif arm in ('SET', 'SPATIAL'):
                self.start = nn.Linear(4, width)
                self.mix = nn.ModuleList([nn.Linear(4 * width, width) for _ in range(2)])
                self.out = nn.Linear(width, 1)
            if arm != 'BASE':
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)

    def forward(self, q, r, qgrid, rgrid, intervention='native', trace=False):
        if self.arm == 'BASE':
            return q.new_zeros(()), {}
        c, a, b = relation(F.normalize(q, dim=1), F.normalize(r, dim=1))
        if self.arm == 'COMPRESSED':
            # Same 520-dimensional feature definition and tokenwise MLP as
            # the historical reader. New experiment shares a residual endpoint
            # across arms rather than reusing its old PRODUCT5 action.
            qz = F.layer_norm(q, (q.shape[1],))
            rz = F.layer_norm(r, (r.shape[1],))
            _, aa, bb = relation(F.normalize(qz, dim=1), F.normalize(rz, dim=1))
            xq, xr = coordinates(qgrid, q.dtype, q.device), coordinates(rgrid, r.dtype, r.device)
            def pack(z, ctx, xy, pos, soft, back):
                entropy = -(soft * soft.clamp_min(1e-30).log()).sum(1) / max(math.log(soft.shape[1]), 1.)
                return torch.cat((z, ctx, z-ctx, z*ctx, xy, pos, pos-xy,
                                  entropy[:, None], back[:, None]), 1)
            xx = pack(qz, aa @ rz, xq, aa @ xr, aa, (aa * bb).sum(1))
            yy = pack(rz, bb.T @ qz, xr, bb.T @ xq, bb.T, (aa * bb).sum(0))
            u = torch.sigmoid(self.out(torch.tanh(self.down(xx)))).mean()
            v = torch.sigmoid(self.out(torch.tanh(self.down(yy)))).mean()
            score = .5 * (u.log() + v.log()) + math.log(2.)
            return score, dict(query_summary=xx.detach(), reference_summary=yy.detach()) if trace else {}

        # log-relative probability avoids dependence on patch count alone.
        x = torch.stack((c, (a * len(r)).clamp_min(1e-30).log() * TEMPERATURE,
                         (b * len(q)).clamp_min(1e-30).log() * TEMPERATURE,
                         torch.sqrt(a * b)), -1)
        inverse = None
        if intervention == 'shuffle':
            g = torch.Generator(device='cpu').manual_seed(31013 + 31*len(q) + len(r))
            pq, pr = torch.randperm(len(q), generator=g).to(q.device), torch.randperm(len(r), generator=g).to(q.device)
            x = x[pq][:, pr]
            inverse = (torch.argsort(pq), torch.argsort(pr))
        elif intervention == 'rotate':
            # Rotate features and coordinates together by 180 degrees on each
            # grid: an exact neighborhood-preserving control, with no wrapping.
            x = x.flip((0, 1))
        elif intervention != 'native':
            raise ValueError(intervention)
        h = torch.tanh(self.start(x))
        traces = [h.detach()] if trace else []
        for layer in self.mix:
            if self.arm == 'SPATIAL':
                mq, mr = local_mean(h, qgrid, 0), local_mean(h, rgrid, 1)
                joint = local_mean(mq, rgrid, 1)
            else:
                mq, mr = uniform_mean(h, 0), uniform_mean(h, 1)
                joint = h.mean((0, 1), keepdim=True).expand_as(h)
            h = h + torch.tanh(layer(torch.cat((h, mq, mr, joint), -1)))
            if trace:
                traces.append(h.detach())
        edge = self.out(h).squeeze(-1)
        if inverse is not None:
            edge = edge[inverse[0]][:, inverse[1]]
        elif intervention == 'rotate':
            edge = edge.flip((0, 1))
        # Both directions retain all soft correspondences, no hard assignment.
        score = .5 * ((a * edge).sum(1).mean() + (b * edge).sum(0).mean())
        return score, dict(cosine=c.detach(), row_softmax=a.detach(), col_softmax=b.detach(),
                           states=traces, edge_evidence=edge.detach()) if trace else {}


def base_features(raw, winner, content):
    idx = [i for i in range(len(raw)) if i != winner]
    raw = torch.as_tensor(raw, dtype=torch.float64, device=content.device)
    content = content.to(torch.float64)
    gap = (raw[idx] - raw[winner]) / raw.std(unbiased=False).clamp_min(1e-12)
    sym = (content[idx] - content[winner]) / (content[idx].abs() + content[winner].abs() + 1e-12)
    return torch.stack((gap, sym, torch.ones_like(gap)), 1)


def logits(x, head, scores, winner):
    idx = [i for i in range(len(scores)) if i != winner]
    return x @ head + (scores[idx] - scores[winner]).to(torch.float64)


def cost(z, target):
    # -1 = target is RAW winner; -2 = target absent: no positive challenger.
    # Target-absent queries remain failures for accuracy at evaluation.
    if target < 0:
        return F.softplus(z.max())
    mask = torch.arange(len(z), device=z.device) != target
    return F.softplus(-z[target]) + F.softplus(z[mask].max())
