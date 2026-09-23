"""Exact all-candidate scoring and bounded-memory adapter gradients."""
import copy
import torch
from torch.nn import functional as TF
import rc_feature_fusion_core_v1 as F


def objective(z,target,kind):
    if kind=='CE':
        allz=torch.cat((z.new_zeros(1),z))
        return torch.logsumexp(allz,0)-allz[target+1]
    assert kind=='COST1'
    if target==-1:return TF.softplus(torch.amax(z,0))
    wrong=z.clone();wrong[target]=-torch.inf
    return TF.softplus(-z[target])+TF.softplus(torch.amax(wrong,0))


def encode(adapter,value,source,device):
    tokens=value['original_colnomic_tokens'].to(device,dtype=torch.float64)
    if adapter is None:return tokens,torch.zeros_like(tokens)
    return adapter(tokens,value['projected_inputs'][source].to(device))


def pair_forward(record,index,bank,adapter,source,mode,device):
    pair=record['pairs'][index]
    q,_=encode(adapter,bank(record['query_image_key']),source,device)
    r,_=encode(adapter,bank(pair['image_key']),source,device)
    u=pair['query_visibility'].to(device);v=pair['reference_visibility'].to(device)
    if mode=='FREE':u=torch.ones_like(u);v=torch.ones_like(v)
    return F.scores_and_trace(q,r,u,v)


def all_scores(record,bank,adapter,source,mode,device):
    return torch.stack([pair_forward(record,i,bank,adapter,source,mode,device)[0] for i in range(len(record['pairs']))])


def two_pass_backward(record,bank,adapter,theta,source,mode,kind,target):
    device=theta.device
    with torch.no_grad():c4=all_scores(record,bank,adapter,source,mode,device)
    c4=c4.detach().requires_grad_(adapter is not None)
    x=F.differentiable_features(record['candidate_raw_scores'],c4,record['winner'])
    z=x@theta[:-1]+theta[-1];loss=objective(z,target,kind)
    assert torch.isfinite(loss)
    loss.backward()
    active=0
    if adapter is not None:
        upstream=c4.grad.detach()
        for i,gradient in enumerate(upstream):
            # COST1 has mathematically zero derivatives for many candidates.
            # Every candidate was scored above; this is exact zero-VJP removal,
            # not a sampled candidate loss or top-K approximation.
            if not bool(gradient.any()):continue
            scores,_=pair_forward(record,i,bank,adapter,source,mode,device)
            torch.dot(scores,gradient).backward();active+=1
    return dict(loss=float(loss.detach()),active_vjp_candidates=active,candidates=len(record['pairs']))


def synthetic_test():
    gen=torch.Generator().manual_seed(301)
    values={}
    for i in range(7):
        n=4+i%3
        values[str(i)]=dict(original_colnomic_tokens=torch.randn(n,128,generator=gen).half(),
            projected_inputs={s:torch.randn(n,128,generator=gen,dtype=torch.float64) for s in F.SOURCES})
    pairs=[]
    for i in range(1,7):
        pairs.append(dict(image_key=str(i),query_visibility=torch.rand(4,generator=gen,dtype=torch.float64),
            reference_visibility=torch.rand(len(values[str(i)]['original_colnomic_tokens']),generator=gen,dtype=torch.float64)))
    record=dict(query_image_key='0',pairs=pairs,candidate_raw_scores=[float(i) for i in range(6)],winner=5)
    checks=[]
    for kind in ('COST1','CE'):
        for mode in ('FREE','ROMA_WEIGHTED'):
            for target in (-1,2):
                adapter=F.OutputAdapter();adapter.up.weight.data.normal_(generator=gen,std=.01)
                theta=torch.nn.Parameter(torch.linspace(-.3,.5,7,dtype=torch.float64))
                second=copy.deepcopy(adapter);th2=torch.nn.Parameter(theta.detach().clone())
                c4=all_scores(record,values.__getitem__,adapter,'COARSE_FINE',mode,'cpu')
                x=F.differentiable_features(record['candidate_raw_scores'],c4,record['winner'])
                loss=objective(x@theta[:-1]+theta[-1],target,kind);loss.backward()
                result=two_pass_backward(record,values.__getitem__,second,th2,'COARSE_FINE',mode,kind,target)
                err=max(float((a.grad-b.grad).abs().max()) for a,b in zip(adapter.parameters(),second.parameters()))
                assert abs(result['loss']-float(loss.detach()))<1e-13 and err<1e-10
                assert torch.allclose(theta.grad,th2.grad,atol=1e-12,rtol=1e-12)
                checks.append(dict(kind=kind,mode=mode,target=target,max_adapter_gradient_error=err))
    # Adam state roundtrip must preserve the next update exactly.
    a=F.OutputAdapter();t=torch.nn.Parameter(torch.ones(7,dtype=torch.float64)*.1)
    opt=torch.optim.AdamW([*a.parameters(),t],lr=.001)
    opt.zero_grad();two_pass_backward(record,values.__getitem__,a,t,'COL_ONLY','ROMA_WEIGHTED','CE',2);opt.step()
    b=copy.deepcopy(a);u=torch.nn.Parameter(t.detach().clone());opt2=torch.optim.AdamW([*b.parameters(),u],lr=.001)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    for model,head,optimizer in ((a,t,opt),(b,u,opt2)):
        optimizer.zero_grad();two_pass_backward(record,values.__getitem__,model,head,'COL_ONLY','ROMA_WEIGHTED','CE',2);optimizer.step()
    assert torch.equal(t,u) and all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    return dict(status='FUSION_TRAINING_GRADIENT_RESUME_PASS',checks=checks,optimizer_resume_next_update_exact=True)
