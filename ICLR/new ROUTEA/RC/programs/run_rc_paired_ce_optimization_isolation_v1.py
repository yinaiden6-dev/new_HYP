#!/usr/bin/env python3
"""Small fixed-class CE optimization; exact dyadic Fenchel certificate."""
import argparse
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
sys.dont_write_bytecode=True
import run_rc_paired_local_evidence_oof4_v1 as N
import rc_convex_loss_cause_core_v1 as K
OUT=ROOT/'results/rc_paired_ce_optimization_isolation_v1'
AUTH=ROOT/'registry/rc_paired_ce_optimization_isolation_authority_v1_20260912.json'
MODELS=('MEAN2','CURVE3','JOINT3')
DEN=1<<52
BOUND=64
need,read,write,write_bytes,bind,checked=N.need,N.read,N.write,N.write_bytes,N.bind,N.checked


def guard(stage,fold=None):
    a=read(AUTH)
    need(a['status']=='PAIRED_CE_OPTIMIZATION_ISOLATION_AUTHORIZED','AUTHORITY')
    for b in a['code_sources'].values():checked(b)
    need(a['code_sources']['program']==bind(__file__),'PROGRAM')
    checked(a['parent_authority'])
    if stage!='preflight':need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allowed={Path(b['path']).resolve() for b in a['public_sources'].values()}
    for bundle in a['cache_sources']:
        allowed.update(Path(b['path']).resolve() for b in bundle.values())
    if stage in ('fit','verify'):
        need(fold in range(4),'FOLD_RANGE')
        allowed.update(Path(b['path']).resolve() for k,b in a['fold_sources'][str(fold)].items() if k!='heldout_roles')
        allowed.update(Path(b['path']).resolve() for b in a['paired_fold_sources'][str(fold)].values())
    if stage=='join':
        for bundle in a['fold_sources'].values():allowed.update(Path(b['path']).resolve() for b in bundle.values())
        allowed.update(Path(b['path']).resolve() for b in a['join_sources'].values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(t in s for t in ('rc_opened_','/reports/','curator_roles','d1_mi','d1-mi','/grozi/','/target_join/')),'PROTECTED_READ:'+s)
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage!='join':
                name=p.relative_to(OUT).parts[0]
                preflight_file=name=='preflight.json' or (stage=='preflight' and name.startswith('.preflight.json.') and name.endswith('.tmp'))
                own=preflight_file or (fold is not None and name==f'fold{fold:02d}')
            need(own or p in allowed,'UNLISTED_RESULT_INPUT:'+s)
        if p.name=='heldout_roles.json':need(stage=='join','NO_HELD_LABELS_BEFORE_SEAL')
    sys.addaudithook(audit)
    for b in a['public_sources'].values():checked(b)
    if stage!='preflight':need(read(OUT/'preflight.json')['authority']==bind(AUTH),'PREFLIGHT_AUTHORITY')
    return a


def arrays(d,b):
    return np.concatenate((np.zeros((len(d),1,d.shape[-1])),d),1),np.concatenate((np.zeros((len(b),1)),b),1)


def value_gradient(theta,d,b,y):
    z=b+np.einsum('qck,k->qc',d,theta)
    lse=logsumexp(z,axis=1)
    probability=np.exp(z-lse[:,None]);r=np.arange(len(y))
    value=float(np.mean(lse-z[r,y]))
    grad=np.mean(np.einsum('qc,qck->qk',probability,d)-d[r,y],axis=0)
    return value,grad,probability


def exact_certificate(d,b,y,theta,counts=None):
    """Exact endpoint problem. The dyadic simplex witness supplies a valid cut."""
    K.iv.dps=50
    need(np.isfinite(d).all() and np.isfinite(b).all() and np.max(np.abs(theta))<=BOUND,'FINITE_BOX_POINT')
    if counts is None:
        _,_,p=value_gradient(theta,d,b,y)
        counts=np.floor(p*DEN).astype(np.int64)
        for i in range(len(counts)):counts[i,int(np.argmax(p[i]))]+=DEN-int(counts[i].sum())
    counts=np.asarray(counts,dtype=np.int64)
    need(counts.shape==b.shape and np.all(counts>=0) and np.all(counts.sum(1)==DEN),'EXACT_DYADIC_SIMPLEX')
    point=[K.rat(t) for t in theta]
    slope=[Fraction(0)]*len(theta);offset=Fraction(0);lo=hi=Fraction(0)
    for i in range(len(y)):
        bs=[K.rat(v) for v in b[i]];ds=[[K.rat(v) for v in row] for row in d[i]]
        logits=[bb+sum((x*t for x,t in zip(row,point)),Fraction(0)) for bb,row in zip(bs,ds)]
        maximum=max(logits)
        total=K.iv.mpf(0)
        for z in logits:total+=K.iv.exp(K._interval(z-maximum))
        ends=K._ends(K._interval(maximum-logits[int(y[i])])+K.iv.log(total))
        lo+=ends[0];hi+=ends[1]
        entropy=Fraction(0);q=[Fraction(int(v),DEN) for v in counts[i]]
        for p in q:
            if p:entropy+=K._ends(-K._interval(p)*K.iv.log(K._interval(p)))[0]
        offset+=sum((p*bb for p,bb in zip(q,bs)),Fraction(0))-bs[int(y[i])]+entropy
        for k in range(len(theta)):
            slope[k]+=sum((p*row[k] for p,row in zip(q,ds)),Fraction(0))-ds[int(y[i])][k]
    n=len(y);slope=[v/n for v in slope];offset/=n;lo/=n;hi/=n
    lower=max(Fraction(0),offset-BOUND*sum((abs(v) for v in slope),Fraction(0)))
    need(lower<=hi,'VALID_LOWER_UPPER_ORDER')
    gap=hi-lower
    return dict(probability_counts=counts.tolist(),probability_denominator=DEN,objective_lower=str(lo),objective_upper=str(hi),box_lower=str(lower),gap_upper=str(gap),gap_upper_float=K._up(gap),box_bound=BOUND,affine_slope=[str(v) for v in slope],affine_offset_lower=str(offset),certified_box_gap_le_1e_6=gap<=Fraction(1,1000000))


def load(fold):
    rows,train,held,mats,base,y,transforms,receipts=N.load_fold(fold)
    a=read(AUTH);sources=a['paired_fold_sources'][str(fold)]
    validation=read(checked(sources['validation']));previous=read(checked(sources['predictions']))
    need(validation['status']=='PAIRED_LOCAL_INDEPENDENT_NUMPY_FOLD_PASS' and validation['payload']==sources['predictions'] and previous['train_ordinals']==train and previous['heldout_ordinals']==held,'QUALIFIED_PREVIOUS_HEADS')
    need(previous['transforms']==transforms,'SAME_TRAIN_TRANSFORMS')
    old={m:np.asarray([float.fromhex(v) for v in previous['parameters'][m]['theta_binary64']]) for m in MODELS}
    for m in MODELS:need(np.max(abs(old[m]))<=BOUND,'OLD_HEAD_INSIDE_FIXED_BOX')
    return rows,train,held,mats,base,y,old,previous,transforms


def fit(fold):
    rows,train,held,mats,base,y,old,previous,transforms=load(fold)
    result={};allz={};yt=y.numpy()
    for m in MODELS:
        df,bf=arrays(mats[m][train].numpy(),base[train].numpy())
        opt=minimize(lambda t:value_gradient(t,df,bf,yt)[:2],old[m],jac=True,method='L-BFGS-B',bounds=[(-BOUND,BOUND)]*len(old[m]),options=dict(maxiter=2000,maxls=50,ftol=1e-14,gtol=1e-10))
        new=np.asarray(opt.x)
        before=exact_certificate(df,bf,yt,old[m]);after=exact_certificate(df,bf,yt,new)
        need(Fraction(after['objective_upper'])<=Fraction(before['objective_upper'])+Fraction(1,10**10),'NO_TRAIN_DATA_LOSS_INCREASE')
        oldloss=float(Fraction(before['objective_upper']));newloss=float(Fraction(after['objective_upper']))
        result[m]=dict(old_theta_hex=[float(v).hex() for v in old[m]],new_theta_hex=[float(v).hex() for v in new],old_certificate=before,new_certificate=after,old_loss=oldloss,new_loss=newloss,old_box_suboptimality_interval=[str(max(Fraction(0),Fraction(before['objective_lower'])-Fraction(after['objective_upper']))),str(Fraction(before['objective_upper'])-Fraction(after['box_lower']))],solver=dict(success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),evaluations=int(opt.nfev)),boundary_coordinates=[i for i,v in enumerate(new) if abs(v)==BOUND])
        for suffix,theta in [('ADAMW',old[m]),('CEOPT',new)]:allz[m+'_'+suffix]=base+mats[m]@torch.from_numpy(theta)
        print(dict(fold=fold,model=m,old_loss=oldloss,new_loss=newloss,remaining_gap=after['gap_upper_float'],solver_iterations=int(opt.nit)),flush=True)
    predictions=[]
    oldlookup={r['query_id']:r for r in previous['predictions']}
    for i in held:
        r=rows[i];models={}
        for m,z in allz.items():
            scores=z[i];j=int(torch.argmax(scores));pos=r['challenger_positions'][j] if scores[j]>0 else r['base_winner_position']
            models[m]=dict(logits_binary64=[float(v).hex() for v in scores],selected_physical_row=r['candidate_physical_rows'][pos])
            if m.endswith('_ADAMW'):need(models[m]==oldlookup[r['query_id']]['models'][m.removesuffix('_ADAMW')],'OLD_ACTION_LOGIT_BIT_PARITY')
        predictions.append(dict(query_id=r['query_id'],execution_ordinal=i,models=models))
    payload=dict(status='PAIRED_CE_OPT_FOLD_SEALED',authority=bind(AUTH),fold=fold,train_ordinals=train,heldout_ordinals=held,parameters=result,transforms=transforms,predictions=predictions,heldout_label_reads=0,EVAL_reads=0,base_training_updates=0)
    write(OUT/f'fold{fold:02d}'/'payload.json',payload)
    subprocess.run([sys.executable,__file__,'verify','--fold',str(fold)],check=True)


def verify(fold):
    rows,train,held,mats,base,y,old,previous,transforms=load(fold)
    folder=OUT/f'fold{fold:02d}';p=read(folder/'payload.json');need(p['authority']==bind(AUTH) and p['transforms']==transforms,'PAYLOAD_BINDING')
    err=0.;checks=0
    for m in MODELS:
        d,b=arrays(mats[m][train].numpy(),base[train].numpy())
        for kind in ('old','new'):
            theta=np.asarray([float.fromhex(v) for v in p['parameters'][m][kind+'_theta_hex']]);saved=p['parameters'][m][kind+'_certificate']
            rebuilt=exact_certificate(d,b,y.numpy(),theta,saved['probability_counts'])
            need(rebuilt==saved,'EXACT_FENCHEL_INTERVAL_REPLAY')
            t=torch.tensor(theta,requires_grad=True,dtype=torch.float64)
            z=base[train]+mats[m][train]@t
            loss=F.cross_entropy(torch.cat((z.new_zeros((len(z),1)),z),1),y);grad=torch.autograd.grad(loss,t)[0].numpy()
            val,ng,_=value_gradient(theta,d,b,y.numpy())
            need(abs(float(loss.detach())-val)<=2e-12 and np.allclose(grad,ng,atol=2e-12,rtol=0),'INDEPENDENT_TORCH_CE_GRADIENT')
            need(float(Fraction(saved['objective_lower']))-2e-12<=val<=float(Fraction(saved['objective_upper']))+2e-12,'NUMERIC_LOSS_INTERVAL_PARITY')
    for pred,i in zip(p['predictions'],held,strict=True):
        r=rows[i];need(pred['query_id']==r['query_id'],'PREDICTION_AXIS')
        for m in MODELS:
            for suffix,kind in [('ADAMW','old'),('CEOPT','new')]:
                theta=np.asarray([float.fromhex(v) for v in p['parameters'][m][kind+'_theta_hex']]);z=base[i].numpy()+np.sum(mats[m][i].numpy()*theta,axis=1)
                saved=pred['models'][m+'_'+suffix];want=np.asarray([float.fromhex(v) for v in saved['logits_binary64']]);diff=float(np.max(abs(z-want)));need(diff<=2e-10,'INDEPENDENT_ALL_LOGITS')
                j=int(np.argmax(z));pos=r['challenger_positions'][j] if z[j]>0 else r['base_winner_position'];need(saved['selected_physical_row']==r['candidate_physical_rows'][pos],'INDEPENDENT_ACTION')
                err=max(err,diff);checks+=127
    write(folder/'validation.json',dict(status='PAIRED_CE_OPT_EXACT_BOUND_AND_PREDICTION_REPLAY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),fold=fold,logit_checks=checks,max_abs_logit_error=err,heldout_label_reads=0,EVAL_reads=0))
    print(dict(stage='verify',fold=fold,checks=checks,max_abs_error=err),flush=True)


def preflight():
    rng=np.random.default_rng(17);d=rng.normal(size=(4,128,3));b=rng.normal(size=(4,128));d[:,0]=0;b[:,0]=0;y=np.array([0,1,64,127]);theta=np.array([.2,-.3,.1])
    loss,g,_=value_gradient(theta,d,b,y);t=torch.tensor(theta,requires_grad=True,dtype=torch.float64);z=torch.from_numpy(b)+torch.from_numpy(d)@t
    tl=F.cross_entropy(z,torch.from_numpy(y));tg=torch.autograd.grad(tl,t)[0].numpy()
    need(abs(float(tl.detach())-loss)<1e-12 and np.allclose(g,tg,atol=1e-12,rtol=0),'SYNTHETIC_FULL128_CE_GRADIENT')
    c=exact_certificate(d,b,y,theta);need(exact_certificate(d,b,y,theta,c['probability_counts'])==c,'CERTIFICATE_REPLAY')
    for t in rng.uniform(-BOUND,BOUND,size=(8,3)):
        f,_,_=value_gradient(t,d,b,y);need(f>=float(Fraction(c['box_lower']))-1e-12,'SYNTHETIC_GLOBAL_LOWER_BOUND')
    opt=minimize(lambda t:value_gradient(t,d,b,y)[:2],theta,jac=True,method='L-BFGS-B',bounds=[(-BOUND,BOUND)]*3,options=dict(maxiter=2000,ftol=1e-14,gtol=1e-10,maxls=50))
    cert=exact_certificate(d,b,y,opt.x);need(cert['gap_upper_float']<1e-6,'SYNTHETIC_NEAR_OPTIMUM_CERTIFIED')
    write(OUT/'preflight.json',dict(status='PAIRED_CE_OPT_SYNTHETIC_PASS',authority=bind(AUTH),natural_training_updates=0,checks=['full128_CE_gradient','exact_simplex','rational_interval_replay','global_box_lower_bound','near_optimum_certificate']))
    print('PAIRED_CE_OPT_SYNTHETIC_PASS',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','fit','verify']);ap.add_argument('--fold',type=int);args=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(args.stage,args.fold)
    if args.stage=='preflight':preflight()
    elif args.stage=='fit':fit(args.fold)
    else:verify(args.fold)
