"""Feature pooling and equal-capacity output adapters, with differentiable scoring."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCES = ('COL_ONLY', 'COARSE', 'FINE', 'COARSE_FINE')
COMPONENTS = ('coarse_11', 'coarse_17', 'fine_lr_1', 'fine_lr_2', 'fine_lr_4',
              'fine_hr_1', 'fine_hr_2', 'fine_hr_4')
WIDTHS = dict(zip(COMPONENTS, (1024, 1024, 64, 128, 256, 64, 128, 256)))


def pool_cells(feature, geometry):
    """FP64 area pooling in oriented coordinates; original cell floor/ceil rule.

    Compute one integral volume per <=32 channels to bound GPU memory. These
    are new descriptor pools, not a bit-exact replacement of the old scorer.
    """
    assert feature.ndim == 4 and feature.shape[0] == 1
    a = feature[0].detach()
    h, w, c = a.shape
    boxes = torch.as_tensor(geometry.cell_boxes_xyxy, dtype=torch.float64, device=a.device)
    valid = torch.as_tensor(geometry.valid_patch_mask, dtype=torch.bool, device=a.device)
    wh = a.new_tensor([w, h], dtype=torch.float64)
    lo = torch.floor(boxes[:, :2] * wh).long()
    hi = torch.ceil(boxes[:, 2:] * wh).long()
    lo = torch.maximum(lo, torch.zeros_like(lo))
    lo = torch.minimum(lo, lo.new_tensor([w-1, h-1]))
    hi = torch.minimum(torch.maximum(hi, lo+1), hi.new_tensor([w, h]))
    x0, y0 = lo.T; x1, y1 = hi.T
    area = ((x1-x0)*(y1-y0)).double()[:, None]
    result = []
    for first in range(0, c, 32):
        block = a[..., first:first+32].double()
        sums = block.cumsum(0).cumsum(1)
        sums = F.pad(sums.permute(2, 0, 1), (1, 0, 1, 0)).permute(1, 2, 0)
        pooled = (sums[y1, x1]-sums[y0, x1]-sums[y1, x0]+sums[y0, x0])/area
        pooled[~valid] = 0
        result.append(pooled.cpu())
    values = torch.cat(result, dim=1)
    assert torch.isfinite(values).all()
    # Outcome-blind cells checked by an independent direct CPU reduction.
    errors = []
    for idx in sorted(set((0, len(values)//2, len(values)-1))):
        if not bool(valid[idx]):
            assert bool((values[idx] == 0).all()); continue
        direct = a[int(y0[idx]):int(y1[idx]), int(x0[idx]):int(x1[idx])].double().cpu().mean((0, 1))
        error = float((direct-values[idx]).abs().max()); errors.append(error)
        assert torch.allclose(direct, values[idx], atol=1e-8, rtol=1e-10), ('POOL_DIRECT', error)
    return values.float(), dict(feature_shape=list(feature.shape), pooled_shape=list(values.shape),
                               max_direct_mean_error=max(errors, default=0.), dtype='float32_after_FP64_pool')


def projected_input(tokens, components, source):
    """Seeded, label-free dimension control: every trainable adapter sees 128-D."""
    assert source in SOURCES
    names = (() if source == 'COL_ONLY' else
             COMPONENTS[:2] if source == 'COARSE' else
             COMPONENTS[2:] if source == 'FINE' else COMPONENTS)
    blocks = [tokens.double()] if not names else [components[k].double() for k in names]
    x = torch.cat([F.layer_norm(b, (b.shape[-1],)) for b in blocks], -1)
    generator = torch.Generator(device='cpu').manual_seed(20260922)
    matrix = (torch.randint(0, 2, (x.shape[1], 128), generator=generator).double()*2-1)/math.sqrt(x.shape[1])
    return F.layer_norm(x @ matrix.to(x.device), (128,))


class OutputAdapter(nn.Module):
    """Exactly 4096 trainable parameters for each feature source, shared q/r."""
    def __init__(self, seed=17):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.down = nn.Linear(128, 16, bias=False, dtype=torch.float64)
            self.up = nn.Linear(16, 128, bias=False, dtype=torch.float64)
            nn.init.zeros_(self.up.weight)

    def forward(self, tokens, aligned):
        # Extracted descriptors may be inference tensors. Clone outside the
        # extractor's inference context so linear backward may save them.
        gate = 0.1*torch.tanh(self.up(self.down(aligned.double().clone())))
        # Normalization occurs ONCE inside the scorer. Zero gate preserves the
        # original FP16 values cast to FP64, avoiding a second normalization.
        return tokens.double().clone()*(1+gate), gate


def scores_and_trace(q, r, u, v):
    q = F.normalize(q.double(), dim=1); r = F.normalize(r.double(), dim=1)
    sim = q @ r.T
    def one(a, b):
        local, index = (sim*b[None]).max(1)
        mass = (a.mean()*b.mean()).sqrt()
        contribution = mass*a*local/a.sum().clamp_min(1e-12)
        # Retain original operation order for the scalar, rather than summing
        # separately divided contributions.
        score = mass*(a*local).sum()/a.sum().clamp_min(1e-12)
        return score, mass, index, contribution
    score, mass, ix, contrib = one(u, v)
    qc = one(u.roll(max(1,len(u)//2)), v)[0]
    rc = one(u, v.roll(max(1,len(v)//2)))[0]
    free, fi = sim.max(1)
    return torch.stack((score, mass, qc, rc)), dict(
        free_maxsim_reference_token=fi, free_maxsim_similarity=free,
        weighted_maxsim_reference_token=ix, query_token_score_contribution=contrib)


def differentiable_features(raw, c4, winner):
    challengers = [i for i in range(len(raw)) if i != winner]
    # Raw is frozen and must retain the original Python mean/variance order.
    mean = sum(map(float, raw))/len(raw)
    std = (sum((float(v)-mean)**2 for v in raw)/len(raw))**0.5
    rawgap = c4.new_tensor([(float(raw[i])-float(raw[winner]))/max(std,1e-12) for i in challengers])
    s, m, q, r = c4.T
    columns = (s, m, s/m.clamp_min(1e-12), s-q, s-r)
    def contrast(x):
        a, b = x[challengers], x[winner]
        return (a-b)/(a.abs()+b.abs()+1e-12)
    return torch.stack((rawgap, *(contrast(x) for x in columns)), dim=1)


def self_test():
    from types import SimpleNamespace
    from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature
    gen = torch.Generator().manual_seed(72)
    geom = SimpleNamespace(cell_boxes_xyxy=torch.tensor([[0,0,.5,.5],[.4,.2,1,1],[0,0,1,1]],dtype=torch.float64),
                           valid_patch_mask=torch.tensor([True,True,False]))
    pool, check = pool_cells(torch.randn(1,9,7,65,generator=gen), geom)
    assert pool.shape==(3,65) and not pool[2].any()
    q=torch.randn(7,128,generator=gen).half();r=torch.randn(9,128,generator=gen).half()
    cq={k:torch.randn(7,d,generator=gen) for k,d in WIDTHS.items()}
    cr={k:torch.randn(9,d,generator=gen) for k,d in WIDTHS.items()}
    u=torch.rand(7,generator=gen,dtype=torch.float64);v=torch.rand(9,generator=gen,dtype=torch.float64)
    keys=('real_score','visibility_mass','query_control_score','reference_control_score')
    gradient_checks=[]
    for source in SOURCES:
        model=OutputAdapter(); assert sum(p.numel() for p in model.parameters())==4096
        aq=projected_input(q,cq,source);ar=projected_input(r,cr,source)
        qq,g=model(q,aq);rr,_=model(r,ar)
        assert torch.equal(qq,q.double()) and not g.any()
        for matched in (False,True):
            a,b=(u,v) if matched else (torch.ones_like(u),torch.ones_like(v))
            old,_=scores_and_trace(q,r,a,b);new,_=scores_and_trace(qq,rr,a,b)
            assert torch.equal(old,new)
        # Deliberate differentiable scalar, no natural query labels.
        opt=torch.optim.SGD(model.parameters(),lr=.5)
        for step in range(2):
            opt.zero_grad();qq,_=model(q,aq);rr,_=model(r,ar)
            loss=-scores_and_trace(qq,rr,u,v)[0][0];loss.backward()
            assert torch.isfinite(model.up.weight.grad).all() and model.up.weight.grad.abs().sum()>0
            if step:assert model.down.weight.grad.abs().sum()>0
            opt.step()
        gradient_checks.append(source)
    c4=torch.rand(8,4,generator=gen,dtype=torch.float64,requires_grad=True)
    raw=list(range(8));winner=7;x=differentiable_features(raw,c4,winner)
    evidence={i:dict(zip(keys,map(float,c4[i].detach()))) for i in range(8)}
    literal=torch.stack([candidate_feature(raw,evidence,i,winner) for i in range(7)])
    assert torch.equal(x.detach(),literal)
    x[:,1:].sum().backward();assert c4.grad is not None and torch.isfinite(c4.grad).all()
    with torch.inference_mode():
        frozen_tokens=q.double().clone();frozen_input=aq.clone()
    model=OutputAdapter();model(frozen_tokens,frozen_input)[0].square().sum().backward()
    assert model.up.weight.grad is not None and torch.isfinite(model.up.weight.grad).all()
    order=torch.tensor([6,0,4,1,7,5,3,2]);newwin=int((order==winner).nonzero()[0])
    xp=differentiable_features([raw[i] for i in order],c4.detach()[order],newwin)
    for row,oldidx in zip(xp,[int(i) for i in order if int(i)!=winner]):assert torch.equal(row,x[oldidx].detach())
    return dict(status='FEATURE_FUSION_CORE_PASS',pool_check=check,zero_adapter_exact=True,
                gradients=gradient_checks,parameter_count_per_source=4096,
                differentiable_feature_formula_exact=True,candidate_permutation_equivariant=True,
                inference_tensor_backward_pass=True)
