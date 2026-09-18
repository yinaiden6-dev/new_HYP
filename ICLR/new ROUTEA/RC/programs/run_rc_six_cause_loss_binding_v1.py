#!/usr/bin/env python3
"""Resolve cost ratio versus full-candidate objective; frozen CE binding control."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'programs'))
import run_rc_six_cause_coverage_v1 as C
H,N=C.H,C.N
read,write,need,bind,checked=C.read,C.write,C.need,C.bind,C.checked
OUT=ROOT/'results/rc_six_cause_isolation_v1/loss_binding'
AUTH=ROOT/'registry/rc_six_cause_loss_binding_authority_v1_20260913.json'

def guard(stage,fold=None):
    a=read(AUTH)
    for b in a['code_sources'].values():checked(b)
    if stage!='preflight':need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    for bs in a['features']:allow.update(Path(b['path']).resolve() for b in bs.values())
    if stage in ('fit','verify'):
        need(fold in range(5),'FOLD_RANGE');allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage=='join':
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for bs in a['fold_sources'].values():allow.update(Path(b['path']).resolve() for b in bs.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(t in s for t in ('/target_join/','d1-mi','d1_mi','/grozi/','/reports/','rc_opened_')),'PROTECTED_READ')
        if 'curator_roles' in s:need(stage=='join','HELD_LABELS_AFTER_SEALS')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage!='join':
                n=p.relative_to(OUT).parts[0];own=n=='preflight.json' or (stage=='preflight' and n.startswith('.preflight.json.') and n.endswith('.tmp')) or (fold is not None and n==f'fold{fold}')
            need(own or p in allow,'UNLISTED_RESULT:'+s)
    sys.addaudithook(audit)
    if stage!='preflight':need(read(OUT/'preflight.json')['authority']==bind(AUTH),'PREFLIGHT')
    return a

def unit(z,y):
    raw=y==-1;ii=torch.arange(len(y));idx=y.clamp_min(0);wrong=z.clone();wrong[ii[~raw],idx[~raw]]=-torch.inf
    return torch.where(raw,F.softplus(torch.amax(z,1)),F.softplus(-z[ii,idx])+F.softplus(torch.amax(wrong,1))).mean()

def train(x,y,kind):
    t=torch.nn.Parameter(torch.zeros(x.shape[-1]+1,dtype=torch.float64));opt=torch.optim.AdamW([t],lr=.03,weight_decay=.001)
    for step in range(2000):
        opt.zero_grad();z=x@t[:-1]+t[-1];l=unit(z,y) if kind=='COST1' else N.R.objective(z,y+1);need(torch.isfinite(l),'FINITE_LOSS');l.backward();opt.step()
    return t.detach()

def compute(fold):
    a=read(AUTH);b=a['fold_sources'][str(fold)];v=read(checked(b['ce_validation']));ce=read(checked(b['ce_payload']))
    need(v['status']=='SIX_CAUSE_COVERAGE_FRESH_FIT_NUMPY_PASS' and v['payload']==b['ce_payload'] and ce['config']==dict(fold=fold,seed=0,block='ALL'),'FROZEN_CE_QUALIFIED')
    roles=read(checked(b['train_roles']))['records'];roles={r['query_id']:r for r in roles};rows,_=H.features();fm=read(checked(a['public_sources']['split']))['folds'][fold]
    trainrows=[r for r in rows if r['query_id'] in set(fm['train_query_ids'])];held=[r for r in rows if r['query_id'] in set(fm['heldout_query_ids'])];need(set(roles)=={r['query_id'] for r in trainrows},'TRAIN_ROLES')
    labels,_=N.P.gallery_labels();data=H.batch(trainrows,roles,labels)
    params={'COST1':train(data['X'],data['y'],'COST1'),'RAW2_CE':train(data['X'][:,:,:1],data['y'],'CE'),'ALL_CE':torch.tensor([float.fromhex(x) for x in ce['models']['ALL_CE']['theta_hex']],dtype=torch.float64)}
    need([r['query_id'] for r in trainrows if H.target_position(r,roles[r['query_id']]['identity'],labels)>=-1]==ce['models']['ALL_CE']['query_ids'],'SAME_RECALL_PRESENT_TRAIN_ORDER')
    preds=[];old={p['query_id']:p for p in ce['predictions']};allz={}
    for name,t in params.items():
        for mode in ('REAL','CBIND') if name!='RAW2_CE' else ('REAL',):
            x=torch.stack([r['modes'][mode]['X'] for r in held]);x=x[:,:,:1] if name=='RAW2_CE' else x
            allz[name if mode=='REAL' else name+'_CBIND']=x@t[:-1]+t[-1]
    for j,r in enumerate(held):
        models={};need(torch.equal(r['modes']['REAL']['X'][:,:1],r['modes']['CBIND']['X'][:,:1]),'BIND_CONTROL_PRESERVES_RAW')
        for name,t in params.items():
            for mode in ('REAL','CBIND') if name!='RAW2_CE' else ('REAL',):
                m=name if mode=='REAL' else name+'_CBIND';z=allz[m][j];k=int(z.argmax());pos=r['challenger_positions'][k] if z[k]>0 else r['winner']
                models[m]=dict(logits_hex=[float(q).hex() for q in z],selected=r['candidate_physical_rows'][pos])
                if m=='ALL_CE':need(models[m]==old[r['query_id']]['models']['ALL_CE'],'FROZEN_CE_ALL_LOGITS_BIT_PARITY')
        preds.append(dict(query_id=r['query_id'],execution_ordinal=r['execution_ordinal'],models=models))
    return dict(status='SIX_CAUSE_LOSS_BINDING_FOLD_SEALED',authority=bind(AUTH),fold=fold,train_query_ids=[r['query_id'] for r in trainrows],parameters={m:[float(v).hex() for v in t] for m,t in params.items()},predictions=preds,heldout_label_reads=0,frozen_CE_training_updates=0)

def fit(fold,replay=False):
    p=compute(fold);folder=OUT/f'fold{fold}'
    if replay:
        need(p==read(folder/'payload.json'),'FRESH_EXACT_FIT_REPLAY');rows,_=H.features();rows={r['query_id']:r for r in rows};maxerr=0.;checks=0
        for pred in p['predictions']:
            r=rows[pred['query_id']]
            for m,v in pred['models'].items():
                name=m.removesuffix('_CBIND');mode='CBIND' if m.endswith('_CBIND') else 'REAL';x=r['modes'][mode]['X'].numpy();x=x[:,:1] if name=='RAW2_CE' else x;t=np.array([float.fromhex(z) for z in p['parameters'][name]]);z=np.sum(x*t[:-1],axis=1)+t[-1];err=float(abs(z-np.array([float.fromhex(q) for q in v['logits_hex']])).max());need(err<2e-10,'NUMPY_LOGITS');maxerr=max(maxerr,err);checks+=127;k=int(z.argmax());pos=r['challenger_positions'][k] if z[k]>0 else r['winner'];need(r['candidate_physical_rows'][pos]==v['selected'],'NUMPY_ACTION')
        write(folder/'validation.json',dict(status='SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),logit_checks=checks,max_abs_error=maxerr,heldout_label_reads=0))
    else:write(folder/'payload.json',p);subprocess.run([sys.executable,__file__,'verify','--fold',str(fold)],check=True)

def join():
    a=read(AUTH);ps=[];vs=[]
    for f in range(5):
        d=OUT/f'fold{f}';v=read(d/'validation.json');p=read(checked(v['payload']));need(v['status']=='SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS' and v['authority']==p['authority']==bind(AUTH),'ALL_FIVE_VALIDATED');ps.append(p);vs.append(bind(d/'validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=vs))
    roles=read(checked(a['join_sources']['curator']))['records'];roles={r['query_id']:r for r in roles};previous=read(checked(a['join_sources']['coverage_result']));pr={r['query_id']:r for r in previous['rows']};strong=read(checked(a['join_sources']['group_result']));sr={r['query_id']:r for r in strong['rows']};labels,_=N.P.gallery_labels();features,_=H.features();fr={r['query_id']:r for r in features};rows=[]
    for p in ps:
        old=read(checked(a['fold_sources'][str(p['fold'])]['ce_payload']));oldpred={r['query_id']:r for r in old['predictions']}
        train=read(checked(a['fold_sources'][str(p['fold'])]['train_roles']))['records'];ti={r['identity'] for r in train};tg={r['component'] for r in train}
        for pred in p['predictions']:
            q=pred['query_id'];role=roles[q];need(role['outer_fold']==p['fold'] and role['identity'] not in ti and role['component'] not in tg,'HELD_IDENTITY_COMPONENT_DISJOINT')
            correct={m:pr[q]['correct'][m] for m in ('RAW','ALL_COST4')};correct['GROUP_BASE']=sr[q]['correct']['GROUP_BASE']
            for m,v in pred['models'].items():correct[m]=labels[v['selected']]==role['identity']
            need(correct['ALL_CE']==pr[q]['correct']['ALL_CE'],'ALL593_CE_UNCHANGED')
            r=fr[q];z=np.array([float.fromhex(x) for x in oldpred[q]['models']['ALL_COST4']['logits_hex']]);j=int(z.argmax());old_best=r['candidate_physical_rows'][r['challenger_positions'][j]]
            rows.append(dict(query_id=q,original_query_id=role['original_query_id'],fold=p['fold'],component=role['component'],group=role['group'],correct=correct,target_in_C128=pr[q]['target_in_C128'],old_cost4_hold=bool(z[j]<=0),old_cost4_top_challenger_is_target=labels[old_best]==role['identity'],new_CE_selected=pred['models']['ALL_CE']['selected']))
    counts={m:sum(r['correct'][m] for r in rows) for m in rows[0]['correct']};pairs=[('ALL_COST4','COST1'),('COST1','ALL_CE'),('GROUP_BASE','ALL_CE'),('RAW','ALL_CE'),('RAW2_CE','ALL_CE'),('ALL_CE','ALL_CE_CBIND'),('COST1','COST1_CBIND')];comparisons={b+'__to__'+m:H.groupstats(rows,b,m) for b,m in pairs};resc=[r for r in rows if r['correct']['ALL_CE'] and not r['correct']['ALL_COST4']];losses=[r for r in rows if not r['correct']['ALL_CE'] and r['correct']['ALL_COST4']]
    action=dict(rescues=len(resc),rescues_old_hold_target_already_top_challenger=sum(r['old_cost4_hold'] and r['old_cost4_top_challenger_is_target'] for r in resc),rescues_other=len(resc)-sum(r['old_cost4_hold'] and r['old_cost4_top_challenger_is_target'] for r in resc),losses=len(losses),losses_from_correct_RAW_HOLD=sum(r['old_cost4_hold'] and r['correct']['RAW'] for r in losses),losses_from_previous_switch=sum(not r['old_cost4_hold'] for r in losses))
    result=dict(status='SIX_CAUSE_H593_COST_COMPETITION_BINDING_COMPLETE',authority=bind(AUTH),counts=counts,comparisons=comparisons,action_diagnosis=action,rows=rows,evidence_level='opened H593 grouped OOF diagnostic; followup declared after observing486; no external confirmation',frozen_CE_training_updates=0,HYP_GO_claimed=False)
    write(OUT/'result.json',result);need(len(rows)==593 and counts['ALL_CE']==486 and counts['ALL_COST4']==440 and counts['GROUP_BASE']==447 and counts['RAW']==426,'BASELINES_AND_CE_PARITY');write(OUT/'validation.json',dict(status='SIX_CAUSE_LOSS_BINDING_PARITY_COUNTS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),fold_validations=vs,all_CE_predictions_unchanged=True));print(dict(counts=counts,action=action,comparisons=comparisons),flush=True)

def preflight():
    z=torch.zeros((4,127),dtype=torch.float64,requires_grad=True);y=torch.tensor([-1,0,31,126]);a=unit(z,y);ls=[]
    for v,t in zip(z,y):ls.append(F.softplus(v.max()) if t==-1 else F.softplus(-v[t])+F.softplus(torch.cat((v[:t],v[t+1:])).max()))
    b=torch.stack(ls).mean();need(torch.equal(a,b) and torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z)[0]),'UNIT_COST_SCALAR_TIED_GRADIENT');write(OUT/'preflight.json',dict(status='SIX_CAUSE_LOSS_BINDING_SYNTHETIC_PASS',authority=bind(AUTH),natural_updates=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','fit','verify','join']);p.add_argument('--fold',type=int);args=p.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(args.stage,args.fold)
    if args.stage=='preflight':preflight()
    elif args.stage=='join':join()
    else:fit(args.fold,args.stage=='verify')
