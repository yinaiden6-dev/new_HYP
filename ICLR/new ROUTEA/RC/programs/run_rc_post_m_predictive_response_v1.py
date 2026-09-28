#!/usr/bin/env python3
"""Prospective new-M response prediction of frozen H593 POST models.

Probe and prediction hashes must be sealed globally before endpoint execution.
No training/identity input to predictors; opened, label-selected pairs are an
explicit evaluation limitation. The old TRAIN16 transport result is preserved.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
for _thread_variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable]='4'
import numpy as np
import torch
from torch.nn import functional as F

RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
import run_rc_m_causal128_frozen_bridge_v1 as B

DEFAULT=RC/'results/rc_post_m_predictive_response_v1'
OLD=RC/'results/rc_post_m_level_gap_v1'
PLAN=RC/'plan/RC_POST_M_PREDICTIVE_RESPONSE_V1_20260927.md'
PROBES=np.asarray([-.02,-.01,0.,.01,.02],dtype=np.float64)
ENDPOINTS=np.asarray([-.08,-.04,.04,.08],dtype=np.float64)
MODELS=('CONSTANT','SHARED_LINEAR','SHARED_QUADRATIC','LOCAL_LINEAR',
        'LOCAL_QUADRATIC','PATCH_LINEAR_MAX','PATCH_QUADRATIC_MAX')
PATCH_MODELS=('PATCH_LINEAR_MAX','PATCH_QUADRATIC_MAX')
H=.02
TOL=2e-10
SIGN_EPS=1e-6
read,write,bind,checked=B.read,B.write,B.bind,B.checked


def arrays(binding):
    with np.load(checked(binding),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def immutable_npz(path,**values):
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)==set(values)
            for k,v in values.items():assert np.array_equal(z[k],v),('ORPHAN_NPZ_MISMATCH',str(path),k)
    else:B.npz_once(path,**values)


def prepare(root):
    assert not (root/'prediction_seal.json').exists()
    parent=read(OLD/'protocol.json');assert read(OLD/'validation.json')['status']=='POST_M_LEVEL_GAP_INDEPENDENT_PASS'
    assert 'PASS' in read(OLD/'second_arithmetic_validation.json')['status']
    oldworkers=read(checked(parent['workers']))['workers'];workers=[];labels=[];domain=[]
    for old in oldworkers:
        candidates=[]
        # Predictor-visible order is numerical position, not target/wrong role.
        for pos in sorted([old['target_position'],old['fixed_wrong_position']]):
            source=old['sources']['ORIGINAL_HR1'][str(pos)];m=float(source['M'])
            probe=m*np.exp(PROBES);probe[2]=m;end=m*np.exp(ENDPOINTS)
            assert np.isfinite(probe).all() and np.isfinite(end).all()
            assert ((probe>0)&(probe<=1)).all() and ((end>0)&(end<=1)).all()
            candidates.append(dict(position=pos,physical_row=old['axis'][pos],M0=m,
                probe_M=probe.tolist(),endpoint_M=end.tolist(),source=source))
            domain.extend([*probe,*end])
        workers.append(dict(index=old['index'],query_id=old['query_id'],component=old['component'],fold=old['fold'],
            axis=old['axis'],RAW_winner=old['RAW_winner'],hidden=old['hidden'],original_POST=old['original_POST'],
            candidates=candidates))
        labels.append(dict(index=old['index'],target_position=old['target_position'],fixed_wrong_position=old['fixed_wrong_position']))
    assert len(workers)==120 and len({r['component'] for r in workers})==45
    root.mkdir(parents=True,exist_ok=True)
    write(root/'workers.json',dict(workers=workers,labels_in_predictor_records=False),immutable=True)
    write(root/'evaluation_roles.json',dict(rows=labels,read_only_after_prediction_seal=True),immutable=True)
    history=RC/'results/rc_mass_information_transport_v1'
    history_sources=[bind(history/'protocol.json'),bind(history/'result.json'),
        bind(history/'independent_validation.json'),bind(RC/'programs/analyze_rc_mass_information_transport_v1.py')]
    protocol=dict(status='POST_M_PREDICTIVE_PROTOCOL_FROZEN',version=1,
        plan=bind(PLAN),program=bind(Path(__file__)),workers=bind(root/'workers.json'),evaluation_roles=bind(root/'evaluation_roles.json'),
        parent_bridge_protocol=parent['parent_bridge_protocol'],inputs=parent['inputs'],authority=parent['authority'],
        manifest=parent['manifest'],endpoints=parent['endpoints'],code_sources=parent['source_code'],
        old_level_gap_protocol=bind(OLD/'protocol.json'),old_level_gap_validation=bind(OLD/'validation.json'),
        historical_transport=history_sources,probe_shifts=PROBES.tolist(),endpoint_shifts=ENDPOINTS.tolist(),models=list(MODELS),
        query_count=120,candidate_count=240,components=45,threads=4,recommended_shards=8,budget_seconds=430,
        new_POST_candidate_forwards_upper_bound=2160,new_vision_RoMa_LLM_GPU_forwards=0,new_training=0,
        probe_domain_min=float(min(domain)),probe_domain_max=float(max(domain)),clipping=False,
        prediction='Given M, predict frozen POST score response. Original HR1 as baseline. New synthetic M shifts are unrevealed endpoints; the queries/groups are already opened.',
        shared_fit='Zero-intercept probe-response least squares; every component equal, then each query equal, then each of two candidates equal. No response endpoints or identity labels.',
        interpretation='Finite local computational response prediction, not M generation, new identity information, unique causality, or independent population confirmation.',
        endpoint_barrier='Every query probe and every prediction saved and hashed in global prediction_seal before any endpoint forward.',
        pair_scope='Inherited target-present fixed-pair diagnosis; intervention assignment uses ascending candidate positions. It is not target-free natural C128 retrieval evaluation.',
        full_axis_action='Other 126 candidate scores remain at original HR1; original frozen INTERNAL3 is recomputed. Report prediction agreement only, not new accuracy.',
        sign_epsilon=SIGN_EPS,arithmetic_tolerance=TOL,bootstrap_seed=20260927,bootstrap_draws=10000)
    write(root/'protocol.json',protocol,immutable=True)
    write(root/'preparation_validation.json',dict(status='POST_M_PREDICTIVE_PREPARATION_PASS',protocol=bind(root/'protocol.json'),
        queries=120,groups=45,candidates=240,new_forwards=0,domain=[min(domain),max(domain)]),immutable=True)
    print(json.dumps(dict(status=protocol['status'],protocol=bind(root/'protocol.json'),queries=120)),flush=True)


def guard(root):
    p=read(root/'protocol.json');assert p['status']=='POST_M_PREDICTIVE_PROTOCOL_FROZEN'
    for b in [p['program'],p['plan'],p['workers'],p['parent_bridge_protocol'],p['inputs'],p['authority'],p['manifest'],
              p['old_level_gap_protocol'],p['old_level_gap_validation'],*p['code_sources'],*p['historical_transport']]:checked(b)
    assert p['probe_shifts']==PROBES.tolist() and p['endpoint_shifts']==ENDPOINTS.tolist() and p['models']==list(MODELS)
    return p,bind(root/'protocol.json'),read(p['workers']['path'])['workers']


def location(root,index,pos,stage):return root/'queries'/f'{index:03d}'/f'p{pos:03d}'/stage


@torch.no_grad()
def actual_values(context,adapter,cache,reference,masses,baseline_argmax=None):
    hidden=cache['hidden'][cache['image_mask']];tokens=[];best=[];where=[];fixed=[]
    ref=reference.double()
    for mass in masses:
        x=B.projected(context.projection,cache,adapter(hidden,float(mass)))
        x=F.normalize(x.double(),dim=-1)
        similarities=x@ref.T;values,indices=similarities.max(1)
        tokens.append(x.cpu().numpy());best.append(values.cpu().numpy());where.append(indices.cpu().numpy().astype(np.int32))
        if baseline_argmax is not None:
            fixed.append(similarities[torch.arange(len(indices)),torch.as_tensor(baseline_argmax,dtype=torch.long)].cpu().numpy())
    return dict(tokens=np.stack(tokens),maxsim=np.stack(best),argmax=np.stack(where),L=np.stack(best).mean(1),
                references=ref.cpu().numpy(),fixed_maxsim=np.stack(fixed) if fixed else np.empty((0,)))


def worker(root,stage,shard,shards,budget,index=None):
    p,pb,workers=guard(root);assert 0<=shard<shards
    selection=[w for w in workers if w['index']==index] if index is not None else workers[shard::shards]
    assert selection
    seal=None
    if stage=='endpoint':
        seal=read(root/'prediction_seal.json');assert seal['status']=='POST_M_PREDICTIONS_SEALED' and seal['protocol']==pb
        assert seal['queries']==120 and seal['candidates']==240
        seal_binding=bind(root/'prediction_seal.json')
        prediction_bindings={r['index']:r for r in seal['rows']}
    start=time.monotonic();deadline=start+budget;context=None
    items={r['index']:r for r in read(p['inputs']['path'])['rows']}
    completed=0
    for w in selection:
        folder=root/'queries'/f"{w['index']:03d}";folder.mkdir(parents=True,exist_ok=True)
        with (folder/(stage+'.lock')).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            summary=folder/(stage+'.json')
            if summary.exists():
                r=read(summary);assert r['protocol']==pb
                for b in r['candidates']:checked(b)
                completed+=1;continue
            if time.monotonic()>deadline-25:return 75
            if context is None:context=B.Context(read(p['parent_bridge_protocol']['path']))
            item=items[w['index']];assert item['query_id']==w['query_id'] and item['post_row']['candidate_ids']==w['axis']
            for b in [w['hidden']['validation'],w['hidden']['payload'],w['original_POST']]:checked(b)
            adapter,head,cache,refs=context.load(item);receipts=[]
            if stage=='endpoint':
                sr=prediction_bindings[w['index']];assert sr['prediction']==bind(folder/'prediction.json')
                pred=read(checked(sr['prediction']));assert pred['protocol']==pb
            for c in w['candidates']:
                base=location(root,w['index'],c['position'],stage);receipt=base.with_suffix('.json');patch=base.with_suffix('.npz')
                if receipt.exists():
                    r=read(receipt);assert r['protocol']==pb and r['position']==c['position'];checked(r['patch']);receipts.append(bind(receipt));continue
                if time.monotonic()>deadline-20:return 75
                tick=time.monotonic()
                if stage=='probe':
                    values=actual_values(context,adapter,cache,refs[c['position']],c['probe_M'])
                    reference=values['references'];z0=values['tokens'][2]@reference.T
                    idx0=values['argmax'][2];rows=np.arange(len(idx0))
                    runner=z0.copy();runner[rows,idx0]=-np.inf;runneridx=np.argmax(runner,axis=1).astype(np.int32)
                    baseline_gap=z0[rows,idx0]-runner[rows,runneridx]
                    fixed=np.stack([(t@reference.T)[rows,idx0] for t in values['tokens']])
                    old=arrays(c['source']['patch']);oldrow=c['source']['row']
                    error=float(np.max(np.abs(values['maxsim'][2]-old['maxsim'][oldrow])))
                    assert error<TOL and np.array_equal(idx0,old['argmax'][oldrow]),('HR1_REPLAY_FAIL',w['index'],c['position'],error)
                    immutable_npz(patch,t=PROBES,M=np.asarray(c['probe_M']),tokens=values['tokens'],references=reference,
                        maxsim=values['maxsim'],argmax=values['argmax'],L=values['L'],fixed_maxsim=fixed,
                        baseline_top2_gap=baseline_gap,baseline_runnerup_argmax=runneridx)
                    extra=dict(baseline_replay_error=error)
                else:
                    probeb=read(location(root,w['index'],c['position'],'probe').with_suffix('.json'))['patch'];probe=arrays(probeb)
                    values=actual_values(context,adapter,cache,refs[c['position']],c['endpoint_M'],probe['argmax'][2])
                    assert np.array_equal(values['references'],probe['references'])
                    immutable_npz(patch,t=ENDPOINTS,M=np.asarray(c['endpoint_M']),tokens=values['tokens'],maxsim=values['maxsim'],
                        argmax=values['argmax'],L=values['L'],fixed_maxsim=values['fixed_maxsim'])
                    extra=dict(prediction_seal=seal_binding,query_prediction=prediction_bindings[w['index']]['prediction'],probe=probeb)
                r=dict(status='POST_M_'+stage.upper()+'_CANDIDATE_COMPLETE',protocol=pb,index=w['index'],query_id=w['query_id'],
                    position=c['position'],physical_row=c['physical_row'],head=head,patch=bind(patch),seconds=time.monotonic()-tick,**extra)
                write(receipt,r,immutable=True);receipts.append(bind(receipt))
            write(summary,dict(status='POST_M_'+stage.upper()+'_QUERY_COMPLETE',protocol=pb,index=w['index'],query_id=w['query_id'],
                component=w['component'],fold=w['fold'],candidates=receipts),immutable=True)
            completed+=1;print(json.dumps(dict(stage=stage,index=w['index'],completed=completed,assigned=len(selection))),flush=True)
    print(json.dumps(dict(stage=stage,shard=shard,complete=completed,seconds=time.monotonic()-start)),flush=True)
    return 0


def local_coefficients(probe):
    values=probe['L'];tokens=probe['tokens']
    return dict(L0=float(values[2]),slope=float((values[4]-values[0])/(2*H)),
        curvature=float((values[4]+values[0]-2*values[2])/H**2),
        half_slope=float((values[3]-values[1])/H),
        half_curvature=float((values[3]+values[1]-2*values[2])/(H/2)**2),
        token_slope=(tokens[4]-tokens[0])/(2*H),token_curvature=(tokens[4]+tokens[0]-2*tokens[2])/H**2)


def forecasts(probe,shared):
    c=local_coefficients(probe);x=ENDPOINTS;v=c['L0'];out=np.empty((len(MODELS),len(x)))
    out[0]=v;out[1]=v+shared['linear']*x
    out[2]=v+shared['quadratic'][0]*x+shared['quadratic'][1]*x*x
    out[3]=v+c['slope']*x;out[4]=v+c['slope']*x+.5*c['curvature']*x*x
    patch=[];idx=[];fixed=[];norms=[];stable=[];rows=np.arange(len(probe['tokens'][2]));baseidx=probe['argmax'][2]
    fixed_slope=np.einsum('nd,nd->n',c['token_slope'],probe['references'][baseidx])
    runner_slope=np.einsum('nd,nd->n',c['token_slope'],probe['references'][probe['baseline_runnerup_argmax']])
    tangent=np.einsum('nd,nd->n',c['token_slope'],probe['tokens'][2])
    for degree in (1,2):
        vv=[];ii=[];ff=[];nn=[];ss=[]
        for t in x:
            token=probe['tokens'][2]+t*c['token_slope']
            if degree==2:token=token+.5*t*t*c['token_curvature']
            sim=token@probe['references'].T;winner=sim.argmax(1)
            vv.append(sim[rows,winner]);ii.append(winner.astype(np.int32));ff.append(sim[rows,baseidx])
            norm=np.linalg.norm(token-probe['tokens'][2],axis=1);flag=probe['baseline_top2_gap']>2*norm
            assert np.all(winner[flag]==baseidx[flag]),'FORECAST_NORM_BOUND_FAILED'
            nn.append(norm);ss.append(flag)
        patch.append(np.stack(vv));idx.append(np.stack(ii));fixed.append(np.stack(ff));norms.append(np.stack(nn));stable.append(np.stack(ss))
    patch=np.stack(patch);out[5:]=patch.mean(2)
    assert np.isfinite(out).all()
    local={k:v for k,v in c.items() if not k.startswith('token_')}
    local.update(fixed_winner_alignment_mean_slope=float(fixed_slope.mean()),
        normalized_token_derivative_mean_norm=float(np.linalg.norm(c['token_slope'],axis=1).mean()),
        normalized_token_derivative_mean_tangent_error=float(np.abs(tangent).mean()),
        scalar_minus_fixed_winner_slope=float(c['slope']-fixed_slope.mean()))
    return dict(L=out,patch_maxsim=patch,patch_argmax=np.stack(idx),fixed_maxsim=np.stack(fixed),
        predicted_token_displacement_norm=np.stack(norms),predicted_bound_stable=np.stack(stable),
        fixed_winner_alignment_slope=fixed_slope,runnerup_alignment_slope=runner_slope,
        derivative_tangent_residual=tangent),local


def seal_predictions(root):
    p,pb,workers=guard(root)
    if (root/'prediction_seal.json').exists():
        seal=read(root/'prediction_seal.json');assert seal['protocol']==pb
        for b in seal['rows']:checked(b['probe']);checked(b['prediction'])
        print(json.dumps(dict(status=seal['status'],already_sealed=True)),flush=True);return
    assert not list((root/'queries').glob('*/endpoint.json')),'NEW_ENDPOINT_EXISTS_BEFORE_GLOBAL_SEAL'
    assert not list((root/'queries').glob('*/p*/endpoint.npz')),'NEW_ENDPOINT_ARRAY_EXISTS_BEFORE_GLOBAL_SEAL'
    counts={g:sum(w['component']==g for w in workers) for g in {w['component'] for w in workers}}
    xs=[];ys=[];ws=[];probes=[]
    for w in workers:
        qp=root/'queries'/f"{w['index']:03d}"/'probe.json';q=read(qp);assert q['protocol']==pb
        for c,b in zip(w['candidates'],q['candidates']):
            rec=read(checked(b));assert rec['position']==c['position'];z=arrays(rec['patch']);assert np.array_equal(z['t'],PROBES)
            xs.extend(PROBES);ys.extend(z['L']-z['L'][2]);ws.extend([1./(len(counts)*counts[w['component']]*2)]*len(PROBES))
        probes.append(dict(index=w['index'],probe=bind(qp)))
    x=np.asarray(xs);y=np.asarray(ys);weights=np.asarray(ws)
    linear=float(np.dot(weights*x,y)/np.dot(weights*x,x))
    design=np.stack([x,x*x],1);gram=design.T@(weights[:,None]*design);rhs=design.T@(weights*y)
    beta=np.linalg.solve(gram,rhs)
    shared=dict(linear=linear,quadratic=beta.tolist(),normal_equation_max_error=float(np.max(abs(gram@beta-rhs))),
        gram=gram.tolist(),rhs=rhs.tolist(),queries=120,groups=45,candidates=240,probes_per_candidate=5,
        labels_used=False,endpoint_values_used=False)
    write(root/'shared_probe_coefficients.json',shared,immutable=True)
    rows=[]
    for w,qb in zip(workers,probes):
        folder=root/'queries'/f"{w['index']:03d}";candidate_predictions=[]
        q=read(checked(qb['probe']))
        for c,b in zip(w['candidates'],q['candidates']):
            rec=read(checked(b));probe=arrays(rec['patch']);out,local=forecasts(probe,shared)
            pp=location(root,w['index'],c['position'],'prediction').with_suffix('.npz')
            immutable_npz(pp,t=ENDPOINTS,M=np.asarray(c['endpoint_M']),models=np.asarray(MODELS),
                patch_models=np.asarray(PATCH_MODELS),**out)
            candidate_predictions.append(dict(position=c['position'],physical_row=c['physical_row'],probe=rec['patch'],
                predictions=bind(pp),local=local,
                predicted_stable_fraction=(out['patch_argmax']==probe['argmax'][2][None,None,:]).mean(2).tolist()))
        path=folder/'prediction.json'
        write(path,dict(status='POST_M_QUERY_PREDICTIONS_BEFORE_ENDPOINT',protocol=pb,index=w['index'],query_id=w['query_id'],
            shared=bind(root/'shared_probe_coefficients.json'),probe=qb['probe'],candidates=candidate_predictions),immutable=True)
        rows.append(dict(**qb,prediction=bind(path)))
    seal=dict(status='POST_M_PREDICTIONS_SEALED',protocol=pb,shared=bind(root/'shared_probe_coefficients.json'),
        queries=120,candidates=240,rows=rows,endpoint_forwards_before_seal=0,labels_read_for_predictor=False)
    write(root/'prediction_seal.json',seal,immutable=True)
    print(json.dumps(dict(status=seal['status'],seal=bind(root/'prediction_seal.json'),shared=shared)),flush=True)


def sign(x):return int(x>SIGN_EPS)-int(x<-SIGN_EPS)


def actions(row,content,head):
    raw=np.asarray(row['raw_scores']);L=np.asarray(content);w=row['winner_index'];other=np.asarray(row['challenger_positions'])
    z=np.zeros(len(L));z[other]=np.stack([(raw[other]-raw[w])/max(raw.std(),1e-12),
        (L[other]-L[w])/(abs(L[other])+abs(L[w])+1e-12),np.ones(len(other))],1)@np.asarray(head)
    best=int(other[np.argmax(z[other])]);pick=best if z[best]>0 else w
    return z,pick


def grouped(values,key):
    groups={}
    for r in values:
        if r[key] is not None:groups.setdefault(r['component'],{}).setdefault(r['query_id'],[]).append(float(r[key]))
    vec=np.asarray([np.mean([np.mean(v) for v in qs.values()]) for qs in groups.values()])
    if not len(vec):return dict(groups=0,queries=0,mean=None,ci95=None)
    rng=np.random.default_rng(20260927);boot=vec[rng.integers(len(vec),size=(10000,len(vec)))].mean(1)
    return dict(groups=len(vec),queries=sum(len(v) for v in groups.values()),mean=float(vec.mean()),
        ci95=np.quantile(boot,[.025,.975]).tolist(),positive_groups=int((vec>0).sum()),negative_groups=int((vec<0).sum()))


def summarize(records,by,keys):
    out={}
    for r in records:
        name='/'.join(str(r[k]) for k in by);out.setdefault(name,[]).append(r)
    return {name:{k:grouped(rows,k) for k in keys} for name,rows in out.items()}


def evaluate(root):
    p,pb,workers=guard(root);seal=read(root/'prediction_seal.json');assert seal['protocol']==pb
    sb=bind(root/'prediction_seal.json');roles={r['index']:r for r in read(checked(p['evaluation_roles']))['rows']}
    items={r['index']:r for r in read(p['inputs']['path'])['rows']}
    records=[];pairs=[];matching=[];allrows=[];maxerr=0.;stage_seconds={'probe':0.,'endpoint':0.}
    for w,sr in zip(workers,seal['rows']):
        assert w['index']==sr['index'];folder=root/'queries'/f"{w['index']:03d}"
        pr=read(checked(sr['prediction']));ep=read(folder/'endpoint.json');assert pr['protocol']==ep['protocol']==pb
        item=items[w['index']];base=read(checked(w['original_POST']))['arms']['NATIVE']['L']
        cv=[];head=None
        for c,predrec,eb in zip(w['candidates'],pr['candidates'],ep['candidates']):
            e=read(checked(eb));assert e['prediction_seal']==sb and e['query_prediction']==sr['prediction']
            assert e['position']==c['position']==predrec['position'];head=e['head']
            probe=arrays(predrec['probe']);pred=arrays(predrec['predictions']);actual=arrays(e['patch'])
            assert np.array_equal(actual['t'],ENDPOINTS) and np.array_equal(pred['M'],actual['M'])
            err=float(np.max(abs(actual['maxsim'].mean(1)-actual['L'])));maxerr=max(maxerr,err)
            assert err<TOL and abs(base[c['position']]-probe['L'][2])<TOL
            cv.append(dict(position=c['position'],probe=probe,pred=pred,actual=actual))
            stage_seconds['endpoint']+=e['seconds']
            oldrec=read(location(root,w['index'],c['position'],'probe').with_suffix('.json'));stage_seconds['probe']+=oldrec['seconds']
            l0=probe['L'][2]
            for j,t in enumerate(ENDPOINTS):
                truth=float(actual['L'][j]-l0);constant_error=abs(truth);shared_error=abs(pred['L'][1,j]-actual['L'][j])
                for k,model in enumerate(MODELS):
                    value=float(pred['L'][k,j]-l0);error=value-truth
                    records.append(dict(index=w['index'],query_id=w['query_id'],component=w['component'],position=c['position'],
                        model=model,t=float(t),actual_change=truth,predicted_change=value,absolute_error=abs(error),squared_error=error*error,
                        MAE_minus_constant=abs(error)-constant_error,MAE_minus_shared=abs(error)-shared_error,
                        direction_agree=float(sign(value)==sign(truth)),actual_zero=float(sign(truth)==0)))
                actualswitch=actual['argmax'][j]!=probe['argmax'][2]
                for k,model in enumerate(PATCH_MODELS):
                    predicted=pred['patch_argmax'][k,j];switch=predicted!=probe['argmax'][2];errpatch=abs(pred['patch_maxsim'][k,j]-actual['maxsim'][j])
                    bound=pred['predicted_bound_stable'][k,j]
                    matching.append(dict(index=w['index'],query_id=w['query_id'],component=w['component'],position=c['position'],model=model,t=float(t),
                        winner_accuracy=float(np.mean(predicted==actual['argmax'][j])),actual_switch_fraction=float(actualswitch.mean()),
                        predicted_switch_fraction=float(switch.mean()),missed_switch_fraction=float((actualswitch&~switch).mean()),
                        spurious_switch_fraction=float((~actualswitch&switch).mean()),
                        predicted_stable_patch_MAE=float(errpatch[~switch].mean()) if (~switch).any() else None,
                        predicted_unstable_patch_MAE=float(errpatch[switch].mean()) if switch.any() else None,
                        predicted_bound_stable_fraction=float(bound.mean()),
                        bound_stable_actual_switch_rate=float(actualswitch[bound].mean()) if bound.any() else None,
                        bound_stable_patch_MAE=float(errpatch[bound].mean()) if bound.any() else None,
                        bound_unstable_patch_MAE=float(errpatch[~bound].mean()) if (~bound).any() else None,
                        rematching_extra=float(np.mean(actual['maxsim'][j]-actual['fixed_maxsim'][j]))))
        role=roles[w['index']];positions=[c['position'] for c in cv];orient=1 if positions[0]==role['target_position'] else -1
        assert set(positions)=={role['target_position'],role['fixed_wrong_position']}
        oldaction,oldpick=actions(item['post_row'],base,head)
        for radius in (.04,.08):
            for pattern,shifts in [('COMMON_PLUS',(radius,radius)),('COMMON_MINUS',(-radius,-radius)),
                                   ('OPPOSITE_PLUS',(radius,-radius)),('OPPOSITE_MINUS',(-radius,radius))]:
                js=[int(np.where(ENDPOINTS==t)[0][0]) for t in shifts]
                realL=np.asarray(base).copy()
                for c,j in zip(cv,js):realL[c['position']]=c['actual']['L'][j]
                realscore,realpick=actions(item['post_row'],realL,head)
                truegap=orient*((realL[positions[0]]-base[positions[0]])-(realL[positions[1]]-base[positions[1]]))
                trueaction=(realscore[role['target_position']]-realscore[role['fixed_wrong_position']])-(oldaction[role['target_position']]-oldaction[role['fixed_wrong_position']])
                sharedgap=orient*((cv[0]['pred']['L'][1,js[0]]-base[positions[0]])-(cv[1]['pred']['L'][1,js[1]]-base[positions[1]]))
                for k,model in enumerate(MODELS):
                    predL=np.asarray(base).copy()
                    for c,j in zip(cv,js):predL[c['position']]=c['pred']['L'][k,j]
                    predscore,predpick=actions(item['post_row'],predL,head)
                    predgap=orient*((predL[positions[0]]-base[positions[0]])-(predL[positions[1]]-base[positions[1]]))
                    predaction=(predscore[role['target_position']]-predscore[role['fixed_wrong_position']])-(oldaction[role['target_position']]-oldaction[role['fixed_wrong_position']])
                    pairs.append(dict(index=w['index'],query_id=w['query_id'],component=w['component'],radius=radius,pattern=pattern,model=model,
                        actual_gap_change=float(truegap),predicted_gap_change=float(predgap),gap_absolute_error=float(abs(predgap-truegap)),
                        gap_MAE_minus_constant=float(abs(predgap-truegap)-abs(truegap)),gap_MAE_minus_shared=float(abs(predgap-truegap)-abs(sharedgap-truegap)),
                        gap_direction_agree=float(sign(predgap)==sign(truegap)),actual_gap_zero=float(sign(truegap)==0),
                        actual_action_gap_change=float(trueaction),predicted_action_gap_change=float(predaction),
                        action_gap_absolute_error=float(abs(predaction-trueaction)),full_axis_choice_agree=float(predpick==realpick),
                        actual_changed_from_baseline=float(realpick!=oldpick)))
        allrows.append(dict(index=w['index'],query_id=w['query_id'],probe=sr['probe'],prediction=sr['prediction'],endpoint=bind(folder/'endpoint.json')))
    assert maxerr<TOL and len(allrows)==120
    summary=dict(candidate=summarize(records,('model','t'),('absolute_error','squared_error','MAE_minus_constant','MAE_minus_shared','direction_agree','actual_zero')),
        paired=summarize(pairs,('model','radius','pattern'),('gap_absolute_error','gap_MAE_minus_constant','gap_MAE_minus_shared','gap_direction_agree',
            'actual_gap_zero','action_gap_absolute_error','full_axis_choice_agree','actual_changed_from_baseline')),
        matching=summarize(matching,('model','t'),('winner_accuracy','actual_switch_fraction','predicted_switch_fraction',
            'missed_switch_fraction','spurious_switch_fraction','predicted_stable_patch_MAE','predicted_unstable_patch_MAE',
            'predicted_bound_stable_fraction','bound_stable_actual_switch_rate','bound_stable_patch_MAE','bound_unstable_patch_MAE','rematching_extra')))
    for row in summary['candidate'].values():
        mse=row['squared_error'];row['RMSE']=dict(mean=float(np.sqrt(mse['mean'])),ci95=np.sqrt(mse['ci95']).tolist(),groups=mse['groups'],queries=mse['queries'])
    result=dict(status='POST_M_PREDICTIVE_RESPONSE_COMPLETE',protocol=pb,prediction_seal=sb,queries=120,groups=45,
        summary=summary,rows=allrows,candidate_records=records,pair_records=pairs,matching_records=matching,
        measured_candidate_stage_seconds=stage_seconds,patch_mean_max_error=maxerr,
        boundary='Prospective synthetic log-M endpoints on already opened fixed pairs. Not unseen-query, full-natural-C128 accuracy, upstream M prediction, or unique causal proof.')
    write(root/'result.json',result,immutable=True)
    write(root/'validation.json',dict(status='POST_M_PREDICTIVE_JOIN_PASS',protocol=pb,result=bind(root/'result.json'),
        queries=120,candidates=240,groups=45,patch_mean_max_error=maxerr,global_seal_enforced=True),immutable=True)
    lines=['# Frozen POST new-M prediction','',result['boundary'],'',
        'All predictions were sealed before endpoint forwards. CONSTANT and SHARED_LINEAR are reported beside every local predictor. Existing TRAIN16 transport was an earlier positive result, not rediscovered here.','',
        '| Model | log-M shift | Content-change MAE | RMSE | MAE difference vs shared (95% exploratory group CI) |','|---|---:|---:|---:|---|']
    for key,row in summary['candidate'].items():
        model,t=key.split('/');d=row['MAE_minus_shared'];lines.append(f"| {model} | {t} | {row['absolute_error']['mean']:.8g} | {row['RMSE']['mean']:.8g} | {d['mean']:.8g} [{d['ci95'][0]:.8g}, {d['ci95'][1]:.8g}] |")
    lines+=['','Complete paired directions, action agreement, patch forecasts and error-by-predicted-stability tables are in result.json. There is no endpoint-based model selection.','',
        'Component-equal exploratory bootstrap intervals use 45 opened groups; 120 queries and their patches are not independent confirmatory samples.',
        f"Measured candidate work: {stage_seconds}; includes POST/projection, array saving and replay checks; excludes queueing, cache loading and prediction-seal work."]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],queries=120,groups=45,result=bind(root/'result.json'))),flush=True)


def check(root):
    """Separate NumPy reconstruction, saved arrays only; no adapter forward."""
    p,pb,workers=guard(root);result=read(root/'result.json');seal=read(root/'prediction_seal.json')
    assert result['protocol']==seal['protocol']==pb and result['prediction_seal']==bind(root/'prediction_seal.json')
    shared=read(checked(seal['shared']));maximum=0.;nvalues=0;prediction_error=0.;metric_error=0.;argmax_ties=0
    counts={g:sum(w['component']==g for w in workers) for g in {w['component'] for w in workers}}
    xx=[];yy=[];ww=[]
    rolemap={r['index']:r for r in read(checked(p['evaluation_roles']))['rows']}
    items={r['index']:r for r in read(p['inputs']['path'])['rows']}
    recmap={(r['index'],r['position'],r['model'],r['t']):r for r in result['candidate_records']}
    pairmap={(r['index'],r['model'],r['radius'],r['pattern']):r for r in result['pair_records']}
    assert len(recmap)==6720 and len(pairmap)==6720
    for w,sr in zip(workers,seal['rows']):
        pr=read(checked(sr['prediction']));ep=read(root/'queries'/f"{w['index']:03d}"/'endpoint.json')
        bypos={};head=None
        for c,predrec,eb in zip(w['candidates'],pr['candidates'],ep['candidates']):
            actualrec=read(checked(eb));assert actualrec['prediction_seal']==bind(root/'prediction_seal.json')
            probe=arrays(predrec['probe']);pred=arrays(predrec['predictions']);actual=arrays(actualrec['patch']);ref=probe['references']
            assert np.array_equal(probe['M'],c['probe_M']) and np.array_equal(actual['M'],c['endpoint_M'])
            xx.extend(PROBES);yy.extend(probe['L']-probe['L'][2]);ww.extend([1./(45*counts[w['component']]*2)]*5)
            bypos[c['position']]=(probe,pred,actual);head=np.asarray(actualrec['head'])
            for z in (probe,actual):
                for i,t in enumerate(z['tokens']):
                    sim=np.einsum('nd,rd->nr',t,ref,optimize=True);idx=sim.argmax(1);best=sim[np.arange(len(idx)),idx]
                    error=float(np.max(abs(best-z['maxsim'][i])));maximum=max(maximum,error)
                    assert error<TOL,('NUMPY_PATCH_MISMATCH',w['index'],c['position'],error)
                    stored=sim[np.arange(len(idx)),z['argmax'][i]]
                    assert np.max(abs(best-stored))<TOL,'STORED_INDEX_NOT_MAXIMAL'
                    argmax_ties+=int(np.count_nonzero(idx!=z['argmax'][i]))
                    nvalues+=1
            # Independently specified finite-difference formulas and forecasts.
            a,b,c0,d,e=probe['L'];s=(e-a)/.04;k=(e+a-2*c0)/.0004
            want=np.stack([np.repeat(c0,4),c0+ENDPOINTS*shared['linear'],
                c0+ENDPOINTS*shared['quadratic'][0]+ENDPOINTS**2*shared['quadratic'][1],
                c0+ENDPOINTS*s,c0+ENDPOINTS*s+.5*ENDPOINTS**2*k])
            prediction_error=max(prediction_error,float(np.max(abs(want-pred['L'][:5]))))
            token0=probe['tokens'][2];d1=(probe['tokens'][4]-probe['tokens'][0])/.04
            d2=(probe['tokens'][4]+probe['tokens'][0]-2*token0)/.0004
            for degree in (1,2):
                for j,t in enumerate(ENDPOINTS):
                    forecast=token0+t*d1+(.5*t*t*d2 if degree==2 else 0.)
                    sim=np.einsum('nd,rd->nr',forecast,ref,optimize=True);idx=sim.argmax(1);best=sim[np.arange(len(idx)),idx]
                    prediction_error=max(prediction_error,float(np.max(abs(best-pred['patch_maxsim'][degree-1,j]))))
                    stored=sim[np.arange(len(idx)),pred['patch_argmax'][degree-1,j]]
                    assert np.max(abs(best-stored))<TOL
                    argmax_ties+=int(np.count_nonzero(idx!=pred['patch_argmax'][degree-1,j]))
                    prediction_error=max(prediction_error,abs(float(best.mean())-pred['L'][4+degree,j]))
                    norm=np.sqrt(np.sum((forecast-token0)**2,axis=1));flag=probe['baseline_top2_gap']>2*norm
                    assert np.array_equal(flag,pred['predicted_bound_stable'][degree-1,j])
            for k,model in enumerate(MODELS):
                for j,t in enumerate(ENDPOINTS):
                    rr=recmap[(w['index'],c['position'],model,float(t))]
                    truth=actual['L'][j]-probe['L'][2];guess=pred['L'][k,j]-probe['L'][2]
                    want={'actual_change':truth,'predicted_change':guess,'absolute_error':abs(guess-truth),'squared_error':(guess-truth)**2,
                        'MAE_minus_constant':abs(guess-truth)-abs(truth),
                        'MAE_minus_shared':abs(guess-truth)-abs(pred['L'][1,j]-actual['L'][j])}
                    metric_error=max(metric_error,max(abs(rr[key]-v) for key,v in want.items()))
        item=items[w['index']];row=item['post_row'];base=np.asarray(read(checked(w['original_POST']))['arms']['NATIVE']['L'])
        raw=np.asarray(row['raw_scores']);winner=row['winner_index'];positions=sorted(bypos);roles=rolemap[w['index']]
        target,wrong=roles['target_position'],roles['fixed_wrong_position'];other=np.asarray(row['challenger_positions'])
        def independently_score(L):
            zz=head[0]*(raw-raw[winner])/np.std(raw)+head[1]*(L-L[winner])/(abs(L)+abs(L[winner])+1e-12)+head[2]
            zz[winner]=0.;k=int(other[np.argmax(zz[other])]);pick=k if zz[k]>0 else winner
            return zz,pick
        old,oldpick=independently_score(base)
        for radius in (.04,.08):
            for pattern,changes in [('COMMON_PLUS',(radius,radius)),('COMMON_MINUS',(-radius,-radius)),
                                    ('OPPOSITE_PLUS',(radius,-radius)),('OPPOSITE_MINUS',(-radius,radius))]:
                ji=[int(np.where(ENDPOINTS==x)[0][0]) for x in changes];real=base.copy()
                for pos,j in zip(positions,ji):real[pos]=bypos[pos][2]['L'][j]
                ra,rpick=independently_score(real);truegap=(real[target]-real[wrong])-(base[target]-base[wrong])
                truedelta=(ra[target]-ra[wrong])-(old[target]-old[wrong])
                for k,model in enumerate(MODELS):
                    guess=base.copy()
                    for pos,j in zip(positions,ji):guess[pos]=bypos[pos][1]['L'][k,j]
                    ga,gpick=independently_score(guess);guessgap=(guess[target]-guess[wrong])-(base[target]-base[wrong])
                    guessdelta=(ga[target]-ga[wrong])-(old[target]-old[wrong]);rr=pairmap[(w['index'],model,radius,pattern)]
                    want={'actual_gap_change':truegap,'predicted_gap_change':guessgap,'gap_absolute_error':abs(truegap-guessgap),
                        'actual_action_gap_change':truedelta,'predicted_action_gap_change':guessdelta,'action_gap_absolute_error':abs(truedelta-guessdelta),
                        'full_axis_choice_agree':float(rpick==gpick),'actual_changed_from_baseline':float(rpick!=oldpick)}
                    metric_error=max(metric_error,max(abs(rr[key]-v) for key,v in want.items()))
    x=np.asarray(xx);y=np.asarray(yy);weights=np.asarray(ww);design=np.column_stack([x,x*x])
    # Independent weighted least squares via SVD, distinct from seal's normal equations.
    beta=np.linalg.lstsq(design*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0]
    linear=np.linalg.lstsq(x[:,None]*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0][0]
    shared_error=max(float(np.max(abs(beta-shared['quadratic']))),abs(linear-shared['linear']))
    assert maximum<TOL and prediction_error<TOL and nvalues==2160
    assert metric_error<TOL and shared_error<TOL
    receipt=dict(status='POST_M_PREDICTIVE_SECOND_NUMPY_PASS',protocol=pb,result=bind(root/'result.json'),queries=120,candidates=240,
        actual_candidate_values=nvalues,numpy_patch_max_error=maximum,prediction_max_error=prediction_error,
        independently_recomputed_metric_rows=len(recmap)+len(pairmap),metric_max_error=metric_error,shared_SVD_fit_max_error=shared_error,
        floating_tie_equivalent_argmax_count=argmax_ties,
        frozen_prediction_seal=bind(root/'prediction_seal.json'),new_forwards=0)
    write(root/'second_arithmetic_validation.json',receipt,immutable=True)
    write(root/'independent_validation.json',receipt,immutable=True)
    print(json.dumps(receipt),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','worker','seal','join','check']);ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--stage',choices=['probe','endpoint'],default='probe');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=8)
    ap.add_argument('--budget',type=float,default=430);ap.add_argument('--index',type=int)
    a=ap.parse_args();B.configure()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='worker':return worker(a.root,a.stage,a.shard,a.shards,a.budget,a.index)
    elif a.command=='seal':seal_predictions(a.root)
    elif a.command=='join':evaluate(a.root)
    else:check(a.root)
    return 0

if __name__=='__main__':raise SystemExit(main())
