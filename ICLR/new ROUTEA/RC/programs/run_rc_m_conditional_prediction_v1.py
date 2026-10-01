#!/usr/bin/env python3
"""Grouped, nested out-of-fold scalar prediction. No encoder or retrieval-head updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.interpolate import BSpline

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_m_conditional_prediction_v1'
ARMS = ['C_ONLY', 'C_RICH', 'C_PLUS_M', 'C_PLUS_RESIDUAL', 'RAW_C', 'RAW_C_M']
RIDGES = [0.0001, 0.001, 0.01, 0.1]

def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if p.exists():
        assert p.read_text() == s, ('IMMUTABLE_OUTPUT', str(p)); return
    tmp = p.with_name(p.name+f'.{os.getpid()}.tmp'); tmp.write_text(s); os.replace(tmp,p)
def emit(**x): print(json.dumps(x, allow_nan=False), flush=True)

def prepare():
    auth = RC/'registry/rc_h593_simple_explanations_authority_v1_20260924.json'
    a = read(auth)
    cache = RC/'results/rc_h593_simple_explanations_v1/cache.json'
    meta_path = RC/'results/rc_h593_quality_operator_eval_v1/result.json'
    meta = {x['query_id']: x for x in read(meta_path)['rows']}
    gallery = {x['physical_row']:x['identity'] for x in read(a['public']['gallery']['path'])['records']}
    splits = read(a['public']['split']['path'])['folds']
    rows=[]; missing=[]
    for r in read(cache)['rows']:
        z=meta[r['query_id']]; target=[i for i,g in enumerate(r['axis']) if gallery[g]==z['identity']]
        if not target: missing.append(r['query_id']); continue
        # Conditional candidate log likelihood is unambiguous in this gallery.
        assert len(target)==1, ('MULTI_REFERENCE_REQUIRES_PROTOCOL', r['query_id'], target)
        raw=np.zeros(128); raw[r['challengers']]=np.asarray(r['X'])[:,0]
        assert np.isfinite(r['free_content']).all() and np.all(np.asarray(r['mass'])>0)
        rows.append(dict(query_id=r['query_id'],component=z['component'],fold=z['fold'],
            axis=r['axis'],target=target[0],C=r['free_content'],logM=np.log(r['mass']).tolist(),RAW=raw.tolist()))
    assert len(rows)==570 and len(missing)==23
    write(OUT/'data.json',dict(rows=rows,candidate_missing=missing))
    sources=[auth,cache,meta_path,Path(a['public']['split']['path']),Path(a['public']['gallery']['path'])]
    write(OUT/'protocol.json',dict(status='CONDITIONAL_PREDICTION_FROZEN',sources=[bind(x) for x in sources],
        code=bind(__file__),data=bind(OUT/'data.json'),splits=splits,arms=ARMS,ridges=RIDGES,
        primary='C_PLUS_M minus C_ONLY group-equal held natural-C128 conditional NLL',
        sensitivity=['C_PLUS_M minus C_RICH (same number of spline coefficients)',
                     'C_PLUS_RESIDUAL minus C_ONLY','RAW_C_M minus RAW_C'],
        spline='SciPy BSpline, degree3 quantile knots;8 knots per channel,17 for C_RICH; drop final basis; linear extrapolation; fit on TRAIN only',
        residual='logM minus TRAIN-only spline least-squares prediction from C; removes this estimated mean only',
        inner='Original outer-TRAIN components assigned to two folds by SHA256(component); no outer-held selection',
        optimizer='Convex conditional multinomial log loss plus L2, L-BFGS with analytic gradient, <=2500 iterations; stationarity gate',
        weighting='Each component equal; each query equal within component;128 candidates are not independent observations',
        primary_evidence='Opened H593 grouped OOF diagnostic, not untouched confirmation or Shannon conditional MI estimator',
        counts=dict(total=593,target_present=570,candidate_missing=23),seed=20260930,
        pair_audit='All fixed0.005/0.01/0.02 C bands; residual signed C gaps and C-superior wrong subset; ties half',
        no_new_network=True,no_GPU=True,no_threshold_tuning=True))
    emit(stage='PREPARED',queries=570,arms=ARMS)

def guard():
    p=read(OUT/'protocol.json'); assert p['code']==bind(__file__) and p['data']==bind(OUT/'data.json')
    return p,read(OUT/'data.json')['rows']
def weights(rows):
    counts={}
    for r in rows: counts[r['component']]=counts.get(r['component'],0)+1
    return np.asarray([1/(len(counts)*counts[r['component']]) for r in rows])

def design(train,test,arm):
    c=np.asarray([r['C'] for r in train]); tc=np.asarray([r['C'] for r in test])
    m=np.asarray([r['logM'] for r in train]); tm=np.asarray([r['logM'] for r in test])
    state={}
    def spline(x,y,name,knots=8):
        base=np.quantile(x.ravel(),np.linspace(0,1,knots))
        assert np.all(np.diff(base)>0), ('REPEATED_SPLINE_KNOTS',name,base.tolist())
        full=np.r_[base[0]-np.arange(3,0,-1)*(base[1]-base[0]),base,base[-1]+np.arange(1,4)*(base[-1]-base[-2])]
        s=BSpline(full,np.eye(len(full)-4),3,extrapolate=True)
        def evaluate(values):
            f=values.ravel();clipped=np.clip(f,base[0],base[-1]);z=s(clipped)
            for edge,mask in [(base[0],f<base[0]),(base[-1],f>base[-1])]:
                z[mask]+=((f[mask]-edge)[:,None])*s.derivative()(edge)[None,:]
            return z[:,:-1]
        u=evaluate(x);v=evaluate(y)
        state[name]=dict(knots=full.tolist(),degree=3)
        return u.reshape(*x.shape,-1),v.reshape(*y.shape,-1)
    a,b=spline(c,tc,'C',17 if arm=='C_RICH' else 8)
    if arm=='C_PLUS_RESIDUAL':
        # Weight regression by query/component, so duplicate candidates do not create new groups.
        z=np.concatenate([np.ones((*a.shape[:2],1)),a],axis=-1)
        zz=np.concatenate([np.ones((*b.shape[:2],1)),b],axis=-1)
        w=np.repeat(np.sqrt(weights(train)/128),128)
        beta=np.linalg.lstsq(z.reshape(-1,z.shape[-1])*w[:,None],m.ravel()*w,rcond=1e-10)[0]
        m=m-z@beta; tm=tm-zz@beta; state['residual_beta']=beta.tolist()
    if arm in ['C_PLUS_M','C_PLUS_RESIDUAL','RAW_C_M']:
        u,v=spline(m,tm,'M'); a=np.concatenate([a,u],-1);b=np.concatenate([b,v],-1)
    if arm in ['RAW_C','RAW_C_M']:
        u,v=spline(np.asarray([r['RAW'] for r in train]),np.asarray([r['RAW'] for r in test]),'RAW')
        a=np.concatenate([a,u],-1);b=np.concatenate([b,v],-1)
    # Any query-common additive score cancels from softmax.
    a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    scale=np.maximum(np.sqrt(np.einsum('q,qkd->d',weights(train),a*a)/128),1e-6)
    a=np.ascontiguousarray(a/scale);b=np.ascontiguousarray(b/scale);state['feature_scale']=scale.tolist()
    return a,b,state

def objective(beta,x,y,w,ridge):
    score=x@beta; norm=logsumexp(score,axis=1); loss=norm-score[np.arange(len(y)),y]
    p=np.exp(score-norm[:,None]);p[np.arange(len(y)),y]-=1
    grad=x.reshape(-1,x.shape[-1]).T@(p*w[:,None]).ravel()+ridge*beta
    return float(w@loss+0.5*ridge*np.dot(beta,beta)),grad
def fit(x,rows,ridge):
    y=np.asarray([r['target'] for r in rows]);w=weights(rows);tick=time.monotonic()
    result=minimize(objective,np.zeros(x.shape[-1]),args=(x,y,w,ridge),jac=True,method='L-BFGS-B',
        options=dict(maxiter=2500,gtol=2e-7,ftol=1e-13,maxls=50))
    val,grad=objective(result.x,x,y,w,ridge)
    assert np.isfinite(result.x).all() and np.max(abs(grad))<2e-5, ('FIT_NOT_SOLVED',result.message,float(np.max(abs(grad))))
    assert val<=np.log(128)+1e-8
    return result.x,dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
        objective=val,max_abs_gradient=float(np.max(abs(grad))),seconds=time.monotonic()-tick)
def evaluate(x,beta,rows):
    s=x@beta;y=np.asarray([r['target'] for r in rows]);n=logsumexp(s,axis=1)-s[np.arange(len(y)),y]
    return s,n,float(weights(rows)@n)

def fold(index,budget):
    p,rows=guard(); split=next(x for x in p['splits'] if x['fold']==index)
    train=[r for r in rows if r['query_id'] in split['train_query_ids']]
    held=[r for r in rows if r['query_id'] in split['heldout_query_ids']]
    assert not {r['component'] for r in train}&{r['component'] for r in held}
    origin=time.monotonic();folder=OUT/f'fold{index}';folder.mkdir(parents=True,exist_ok=True)
    def bucket(r):return int(hashlib.sha256(str(r['component']).encode()).hexdigest()[:8],16)%2
    for arm in ARMS:
        dest=folder/f'{arm}.json'
        if dest.exists():assert read(dest)['protocol']==bind(OUT/'protocol.json');continue
        cv=[]
        for k in range(2):
            tr=[r for r in train if bucket(r)!=k];va=[r for r in train if bucket(r)==k]
            assert tr and va and not {r['component'] for r in tr}&{r['component'] for r in va}
            x,z,st=design(tr,va,arm)
            for ridge in RIDGES:
                cp=folder/'inner'/f'{arm}_{k}_{ridge}.json'
                if cp.exists():row=read(cp);assert row['protocol']==bind(OUT/'protocol.json')
                else:
                    if time.monotonic()-origin>budget:return 75
                    beta,audit=fit(x,tr,ridge);_,_,val=evaluate(z,beta,va)
                    row=dict(protocol=bind(OUT/'protocol.json'),inner=k,ridge=ridge,NLL=val,fit=audit,
                        validation_groups=len({r['component'] for r in va}),train_ids=[r['query_id'] for r in tr],validation_ids=[r['query_id'] for r in va])
                    write(cp,row);emit(stage='INNER',fold=index,arm=arm,inner=k,ridge=ridge,NLL=val)
                cv.append(row)
        best=min(RIDGES,key=lambda l:(sum(v['NLL']*v['validation_groups'] for v in cv if v['ridge']==l)/sum(v['validation_groups'] for v in cv if v['ridge']==l),-l))
        if time.monotonic()-origin>budget:return 75
        x,z,st=design(train,held,arm);beta,audit=fit(x,train,best);s,nll,avg=evaluate(z,beta,held)
        write(dest,dict(protocol=bind(OUT/'protocol.json'),fold=index,arm=arm,ridge=best,coefficients=beta.tolist(),design=st,fit=audit,
            train_ids=[r['query_id'] for r in train],held_ids=[r['query_id'] for r in held],
            rows=[dict(query_id=r['query_id'],component=r['component'],target=r['target'],axis=r['axis'],scores=s[i].tolist(),NLL=float(nll[i]),
                correct=bool(np.argmax(s[i])==r['target'])) for i,r in enumerate(held)],held_group_NLL=avg))
        emit(stage='OUTER_COMPLETE',fold=index,arm=arm,queries=len(held),NLL=avg)
    write(folder/'validation.json',dict(status='CONDITIONAL_FOLD_FITS_COMPLETE',protocol=bind(OUT/'protocol.json'),arms=[bind(folder/f'{a}.json') for a in ARMS]))
    return 0

def boot(pairs):
    groups={}
    for group,value in pairs:groups.setdefault(group,[]).append(value)
    v=np.asarray([np.mean(x) for x in groups.values()]);rng=np.random.default_rng(20260930)
    b=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return dict(queries=len(pairs),groups=len(v),group_equal_mean=float(v.mean()),exploratory_group95=np.quantile(b,[.025,.975]).tolist())
def join():
    p,data=guard();by={a:{} for a in ARMS};sources=[];error=0.
    for f in range(5):
        v=read(OUT/f'fold{f}/validation.json');assert v['protocol']==bind(OUT/'protocol.json')
        for b in v['arms']:
            assert bind(b['path'])==b;d=read(b['path']);sources.append(b)
            for r in d['rows']:
                s=np.array(r['scores']);ex=np.exp(s-s.max());loss=float(np.log(ex.sum())+s.max()-s[r['target']]);error=max(error,abs(loss-r['NLL']))
                assert r['query_id'] not in by[d['arm']];by[d['arm']][r['query_id']]=r
    assert error<1e-11 and all(len(v)==570 for v in by.values())
    contrasts={}
    for a,b in [('C_PLUS_M','C_ONLY'),('C_PLUS_M','C_RICH'),('C_PLUS_RESIDUAL','C_ONLY'),('RAW_C_M','RAW_C')]:
        contrasts[a+' minus '+b]=boot([(r['component'],r['NLL']-by[b][q]['NLL']) for q,r in by[a].items()])
    bands={}
    for width in [.005,.01,.02]:
        rr=[]
        for r in data:
            c=np.array(r['C']);m=np.array(r['logM']);t=r['target'];wrong=np.arange(128)!=t;ix=np.flatnonzero(wrong&(abs(c-c[t])<=width))
            if not len(ix):continue
            positive=ix[c[ix]>=c[t]]
            rr.append(dict(query_id=r['query_id'],component=r['component'],pairs=len(ix),mean_signed_C_gap=float(np.mean(c[t]-c[ix])),
                content_concordance=float(np.mean((c[t]>c[ix])+.5*(c[t]==c[ix]))),
                M_concordance=float(np.mean((m[t]>m[ix])+.5*(m[t]==m[ix]))),
                C_superior_wrong_M_concordance=None if not len(positive) else float(np.mean((m[t]>m[positive])+.5*(m[t]==m[positive])))))
        bands[str(width)]=dict(rows=rr,statistics={k:boot([(r['component'],r[k]) for r in rr if r[k] is not None]) for k in
            ['mean_signed_C_gap','content_concordance','M_concordance','C_superior_wrong_M_concordance']})
    result=dict(status='CONDITIONAL_PREDICTION_OOF_COMPLETE',protocol=bind(OUT/'protocol.json'),contrasts=contrasts,bands=bands,
        models={a:dict(NLL=boot([(r['component'],r['NLL']) for r in v.values()]),correct=sum(r['correct'] for r in v.values())) for a,v in by.items()},
        sources=sources,counts=p['counts'],boundary='Opened grouped OOF; fitting probes only. NLL benefit is not a direct Shannon CMI estimate, causal proof, or deployed model improvement.')
    write(OUT/'result.json',result);write(OUT/'validation.json',dict(status='CONDITIONAL_NLL_INDEPENDENT_ARITHMETIC_PASS',result=bind(OUT/'result.json'),max_error=error,OOF_queries=570))
    text=['# M conditional prediction: H593 grouped OOF','',result['boundary'],'','Natural ColNomic C128;570 target-present queries,23 candidate misses excluded and reported. Six scalar probes, original five group folds; inner selection sees outer TRAIN only.','',
        '| Contrast | Group mean held NLL difference | Exploratory group95% interval |','|---|---:|---|']
    for name,v in contrasts.items():text.append(f"| {name} | {v['group_equal_mean']:.8f} | {v['exploratory_group95']} |")
    text+=['','Negative differences favour M. Primary is C_PLUS_M vs C_ONLY; other contrasts are declared sensitivities, not independent confirmations. No operating threshold, encoder, adapter, or deployed head changed. All128 scores, fitting convergence and original split identities are retained.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')

def self_test():
    rng=np.random.default_rng(17);x=rng.normal(size=(3,5,4));y=np.array([0,2,4]);w=np.array([.2,.3,.5]);b=rng.normal(size=4);_,g=objective(b,x,y,w,.01)
    gn=[]
    for k in range(4):
        e=np.zeros(4);e[k]=1e-6;gn.append((objective(b+e,x,y,w,.01)[0]-objective(b-e,x,y,w,.01)[0])/2e-6)
    assert np.max(abs(g-np.array(gn)))<1e-8
    emit(status='CONDITIONAL_LOSS_GRADIENT_PASS',error=float(np.max(abs(g-np.array(gn)))))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','fold','join','test']);a.add_argument('--index',type=int,default=0);a.add_argument('--budget',type=int,default=420);x=a.parse_args()
    if x.stage=='prepare':prepare()
    elif x.stage=='fold':sys.exit(fold(x.index,x.budget))
    elif x.stage=='join':join()
    else:self_test()
