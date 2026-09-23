"""Token-level quality readout on frozen full pair relations; no encoders."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TEMPERATURE = .1
INPUT_DIM = 520


class Quality(nn.Module):
    def __init__(self, seed=17):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.down = nn.Linear(INPUT_DIM, 16, dtype=torch.float64)
            self.up = nn.Linear(16, 1, dtype=torch.float64)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return torch.sigmoid(self.up(torch.tanh(self.down(x))).squeeze(-1))


def relation(q, r, xyq, xyr, mode):
    """All active tokens, both directions, no hard correspondence/geometry gate."""
    if mode == 'PAIR':
        sim = F.normalize(q, dim=-1) @ F.normalize(r, dim=-1).T
        a = torch.softmax(sim / TEMPERATURE, dim=1)
        b = torch.softmax(sim.T / TEMPERATURE, dim=1)
        cq, cr = a @ r, b @ q
        pq, pr = a @ xyr, b @ xyq
        eq = -(a * a.clamp_min(1e-300).log()).sum(1) / max(math.log(len(r)), 1.)
        er = -(b * b.clamp_min(1e-300).log()).sum(1) / max(math.log(len(q)), 1.)
        backq, backr = (a * b.T).sum(1), (b * a.T).sum(1)
    elif mode == 'SINGLE':
        cq, cr, pq, pr = q, r, xyq, xyr
        eq, er = q.new_zeros(len(q)), r.new_zeros(len(r))
        backq, backr = eq + 1, er + 1
    else:
        raise ValueError(mode)
    def pack(z, c, xy, pos, entropy, back):
        return torch.cat((z, c, z-c, z*c, xy, pos, pos-xy,
                          entropy[:, None], back[:, None]), 1)
    return pack(q, cq, xyq, pq, eq, backq), pack(r, cr, xyr, pr, er, backr)


def pair(model, q, r, mode, trace=False):
    xq, xr = relation(q['z'], r['z'], q['xy'], r['xy'], mode)
    u, v = model(xq), model(xr)
    m = (u.mean()*v.mean()).sqrt()
    return m, dict(u=u, v=v, query_relation=xq, reference_relation=xr) if trace else None


def features(raw, winner, mass, content):
    idx = [i for i in range(len(raw)) if i != winner]
    mean = sum(map(float, raw))/len(raw)
    std = (sum((float(v)-mean)**2 for v in raw)/len(raw))**.5
    gap = mass.new_tensor([(float(raw[i])-float(raw[winner]))/max(std, 1e-12) for i in idx])
    def contrast(x):
        a, b = x[idx], x[winner]
        return (a-b)/(a.abs()+b.abs()+1e-12)
    return torch.stack((gap, contrast(mass*content), contrast(mass), contrast(content)), 1)


def cost(z, target):
    if target == -1:
        return F.softplus(z.amax())
    wrong = z.clone()
    wrong[target] = -torch.inf
    return F.softplus(-z[target]) + F.softplus(wrong.amax())


def synthetic_test():
    g = torch.Generator().manual_seed(53)
    q = dict(z=torch.randn(5, 128, generator=g, dtype=torch.float64),
             xy=torch.rand(5, 2, generator=g, dtype=torch.float64))
    r = dict(z=torch.randn(7, 128, generator=g, dtype=torch.float64),
             xy=torch.rand(7, 2, generator=g, dtype=torch.float64))
    model = Quality()
    assert sum(p.numel() for p in model.parameters()) == 8353
    with torch.no_grad(): model.up.weight.normal_(generator=g, std=.05)
    checks = []
    for mode in ('PAIR', 'SINGLE'):
        m, t = pair(model, q, r, mode, True)
        # Independent NumPy forward, including row/column softmax and MLP.
        a, b = q['z'].numpy(), r['z'].numpy()
        qa, rb = q['xy'].numpy(), r['xy'].numpy()
        if mode == 'PAIR':
            sim = (a/np.linalg.norm(a, axis=1)[:, None]) @ (b/np.linalg.norm(b, axis=1)[:, None]).T / TEMPERATURE
            def soft(x):
                v=np.exp(x-x.max(1, keepdims=True)); return v/v.sum(1, keepdims=True)
            ab, ba = soft(sim), soft(sim.T)
            ca, cb, pa, pb = ab@b, ba@a, ab@rb, ba@qa
            ea=-(ab*np.log(ab)).sum(1)/max(np.log(len(b)),1.)
            eb=-(ba*np.log(ba)).sum(1)/max(np.log(len(a)),1.)
            ra, rr = (ab*ba.T).sum(1), (ba*ab.T).sum(1)
        else:
            ca,cb,pa,pb=a,b,qa,rb
            ea,eb=np.zeros(len(a)),np.zeros(len(b));ra,rr=ea+1,eb+1
        for z,c,xy,p,e,back,key in [(a,ca,qa,pa,ea,ra,'query_relation'),(b,cb,rb,pb,eb,rr,'reference_relation')]:
            x=np.concatenate((z,c,z-c,z*c,xy,p,p-xy,e[:,None],back[:,None]),1)
            assert np.max(abs(x-t[key].detach().numpy())) < 2e-12
        def mlp(x):
            y=np.tanh(x@model.down.weight.detach().numpy().T+model.down.bias.detach().numpy())
            y=y@model.up.weight.detach().numpy().T+model.up.bias.detach().numpy()
            return 1/(1+np.exp(-y[:,0]))
        expected=np.sqrt(mlp(t['query_relation'].numpy()).mean()*mlp(t['reference_relation'].numpy()).mean())
        assert abs(expected-float(m.detach()))<1e-12
        # Joint token/coordinate permutation cannot alter the scalar.
        perm=torch.tensor([3,0,6,2,1,5,4])
        mr,_=pair(model,q,{k:v[perm] for k,v in r.items()},mode)
        assert torch.allclose(m,mr,atol=1e-12,rtol=1e-12)
        # Exact scalar-VJP and ordinary autograd must agree.
        for target in (-1,1):
            m1=copy.deepcopy(model);m2=copy.deepcopy(model)
            h1=nn.Parameter(torch.tensor([.1,.4,.3,.2,-.5],dtype=torch.float64));h2=nn.Parameter(h1.detach().clone())
            refs=[{k:v*(1+.1*i) for k,v in r.items()} for i in range(4)]
            mass=torch.stack([pair(m1,q,ref,mode)[0] for ref in refs])
            l=cost(features([.1,.3,.2,.0],1,mass,torch.tensor([.2,.3,.4,.1],dtype=torch.float64))@h1[:-1]+h1[-1],target);l.backward()
            with torch.no_grad(): detached=torch.stack([pair(m2,q,ref,mode)[0] for ref in refs])
            detached.requires_grad_()
            ll=cost(features([.1,.3,.2,.0],1,detached,torch.tensor([.2,.3,.4,.1],dtype=torch.float64))@h2[:-1]+h2[-1],target);ll.backward()
            for i,ref in enumerate(refs):
                if detached.grad[i] != 0: (pair(m2,q,ref,mode)[0]*detached.grad[i]).backward()
            error=max(float((a.grad-b.grad).abs().max()) for a,b in zip(m1.parameters(),m2.parameters()))
            assert error<1e-11 and torch.allclose(h1.grad,h2.grad,atol=1e-12,rtol=1e-12)
            checks.append(dict(mode=mode,target=target,gradient_error=error))
    # Quality truly depends on the other image only in PAIR.
    _,a=pair(model,q,r,'PAIR',True);_,b=pair(model,q,{**r,'z':-r['z']},'PAIR',True)
    assert not torch.allclose(a['u'],b['u'])
    _,a=pair(model,q,r,'SINGLE',True);_,b=pair(model,q,{**r,'z':-r['z']},'SINGLE',True)
    assert torch.equal(a['u'],b['u'])
    return dict(status='PAIR_QUALITY_NUMPY_GRADIENT_EQUIVARIANCE_PASS',checks=checks,
                parameters=8353,all_token_axis=True,real_label_reads=0)
