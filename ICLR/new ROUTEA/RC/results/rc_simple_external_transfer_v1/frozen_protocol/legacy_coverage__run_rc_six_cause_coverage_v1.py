#!/usr/bin/env python3
"""H593 fixed-fold, matched-budget identity coverage and within-identity curves."""
import argparse,hashlib,json,os,subprocess,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'programs'))
import run_rc_new_hyp593_oof5_v1 as H
import run_rc_paired_local_evidence_oof4_v1 as N
read,write,need,bind,checked=N.read,N.write,N.need,N.bind,N.checked
OUT=ROOT/'results/rc_six_cause_isolation_v1/coverage'
AUTH=ROOT/'registry/rc_six_cause_coverage_authority_v1_20260913.json'
CONFIG=[dict(fold=f,seed=s,block=b) for f in range(5) for s in range(3) for b in ('COVERAGE','WITHIN')]+[dict(fold=f,seed=0,block='ALL') for f in range(5)]

def guard(stage,index=None):
    a=read(AUTH)
    for b in a['code_sources'].values():checked(b)
    if stage!='preflight':need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    for bs in a['features']:
        allow.update(Path(b['path']).resolve() for b in bs.values())
    if stage in ('fit','verify'):
        need(index in range(35),'CONFIG_INDEX');f=CONFIG[index]['fold']
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(f)].values())
    if stage=='join':
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for bs in a['fold_sources'].values():allow.update(Path(b['path']).resolve() for b in bs.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(t in s for t in ('/target_join/','d1_mi','d1-mi','/grozi/','/reports/','rc_opened_')),'PROTECTED_READ')
        if 'curator_roles' in s:need(stage=='join','NO_HELD_LABELS_BEFORE_SEALS')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage!='join':
                n=p.relative_to(OUT).parts[0];own=n=='preflight.json' or (stage=='preflight' and n.startswith('.preflight.json.') and n.endswith('.tmp')) or (index is not None and n==f'fit{index:02d}')
            need(own or p in allow,'UNLISTED_RESULT:'+s)
    sys.addaudithook(audit)
    if stage!='preflight':need(read(OUT/'preflight.json')['authority']==bind(AUTH),'PREFLIGHT')
    return a

def subsets(train,roles,seed,block):
    key=lambda s:hashlib.sha256((f'SIX_CAUSE_COVERAGE_V1|{seed}|'+s).encode()).hexdigest()
    pools=defaultdict(list)
    for r in train:pools[roles[r['query_id']]['identity']].append(r)
    for rs in pools.values():rs.sort(key=lambda r:key(r['query_id']))
    ids=sorted(pools,key=key)
    def roundrobin(selected,budget):
        result=[]
        for level in range(max(len(pools[i]) for i in selected)):
            for i in selected:
                if level<len(pools[i]):result.append(pools[i][level])
        need(len(result)>=budget,'EXACT_BUDGET_AVAILABLE')
        return sorted(result[:budget],key=lambda r:r['execution_ordinal'])
    if block=='COVERAGE':
        selected=ids[:max(1,len(ids)//2)]
        while sum(len(pools[i]) for i in selected)<128:selected.append(ids[len(selected)])
        need(len(ids)<=128 and len(selected)<len(ids),'BROAD_NARROW_NONTRIVIAL')
        return {'BROAD128':roundrobin(ids,128),'NARROW128':roundrobin(selected,128)}
    if block=='WITHIN':
        eligible=[i for i in ids if len(pools[i])>=4];need(len(eligible)>=2,'MULTIPLE_IDENTITIES_WITH_FOUR_IMAGES')
        return {f'PER_ID_{k}':sorted([r for i in eligible for r in pools[i][:k]],key=lambda r:r['execution_ordinal']) for k in (1,2,4)}
    return {'ALL':train}

def fit_theta(data,objective):
    if objective=='COST4':
        theta,_=H.fit_base(data);return theta
    theta=torch.nn.Parameter(torch.zeros(7,dtype=torch.float64));opt=torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
    for step in range(2000):
        opt.zero_grad();z=data['X']@theta[:6]+theta[6];l=N.R.objective(z,data['y']+1);need(torch.isfinite(l),'FINITE_CE');l.backward();opt.step()
    return theta.detach()

def compute(index):
    a=read(AUTH);c=CONFIG[index];f=c['fold'];rows,sources=H.features();fm=read(checked(a['public_sources']['split']))['folds'][f]
    roles=read(checked(a['fold_sources'][str(f)]['train_roles']))['records'];roles={r['query_id']:r for r in roles};labels,_=N.P.gallery_labels()
    train=[r for r in rows if r['query_id'] in set(fm['train_query_ids'])];held=[r for r in rows if r['query_id'] in set(fm['heldout_query_ids'])];need(set(roles)=={r['query_id'] for r in train},'TRAIN_ROLE_AXIS')
    keep=[r for r in train if H.target_position(r,roles[r['query_id']]['identity'],labels)>=-1]
    sets=subsets(keep,roles,c['seed'],c['block']);models={};pred=[]
    if c['block']=='ALL':
        b=a['fold_sources'][str(f)];v=read(checked(b['old_validation']));rr=read(checked(b['old_receipt']));need(v['status']=='H593_FOLD_RETRAIN_AND_PREDICTION_REPLAY_PASS' and v['payload']==rr['payload']==b['old_payload'],'OLD_BASE_QUALIFIED')
        old=torch.load(checked(b['old_payload']),weights_only=True,map_location='cpu')
    X=torch.stack([r['modes']['REAL']['X'] for r in held]);zs={}
    for name,rs in sets.items():
        data=H.batch(rs,roles,labels);need(data['target_absent_count']==0,'EXACT_RECALL_PRESENT_SAMPLE_BUDGET')
        for obj in ('COST4','CE'):
            m=name+'_'+obj
            theta=old['parameters']['BASE7_ALL']['theta'] if name=='ALL' and obj=='COST4' else fit_theta(data,obj)
            z=data['X']@theta[:6]+theta[6]
            models[m]=dict(theta_hex=[float(v).hex() for v in theta],query_ids=[r['query_id'] for r in rs],images=len(rs),identities=len({roles[r['query_id']]['identity'] for r in rs}),components=len({roles[r['query_id']]['component'] for r in rs}),train_CE=float(N.R.objective(z,data['y']+1)),train_cost4=float(H.loss(z,data['y'])),training_updates=0 if name=='ALL' and obj=='COST4' else 2000,original_full_train_count=len(train),excluded_target_absent=len(train)-len(keep))
            zs[m]=X@theta[:6]+theta[6]
            print(dict(index=index,model=m,images=len(rs),identities=models[m]['identities']),flush=True)
    if c['block']=='COVERAGE':need(models['BROAD128_CE']['images']==models['NARROW128_CE']['images']==128 and models['BROAD128_CE']['identities']>models['NARROW128_CE']['identities'],'SAME_BUDGET_DIFFERENT_COVERAGE')
    if c['block']=='WITHIN':
        ids=[{roles[q]['identity'] for q in models[f'PER_ID_{k}_CE']['query_ids']} for k in (1,2,4)];need(ids[0]==ids[1]==ids[2],'IDENTITY_SET_FIXED')
        for lo,hi in ((1,2),(2,4)):need(set(models[f'PER_ID_{lo}_CE']['query_ids'])<set(models[f'PER_ID_{hi}_CE']['query_ids']),'NESTED_IMAGES')
    for j,r in enumerate(held):
        ms={}
        for m,z in zs.items():
            v=z[j];k=int(v.argmax());p=r['challenger_positions'][k] if v[k]>0 else r['winner'];ms[m]=dict(selected=r['candidate_physical_rows'][p],logits_hex=[float(x).hex() for x in v])
        pred.append(dict(query_id=r['query_id'],execution_ordinal=r['execution_ordinal'],models=ms))
    return dict(status='SIX_CAUSE_COVERAGE_FIT_SEALED',authority=bind(AUTH),config=c,models=models,predictions=pred,heldout_label_reads=0)

def fit(index,replay=False):
    folder=OUT/f'fit{index:02d}';p=compute(index)
    if replay:
        saved=read(folder/'payload.json');need(saved==p,'FRESH_PARAMETERS_AND_PREDICTIONS_EXACT_REPLAY')
        rows,_=H.features();rows={r['query_id']:r for r in rows};err=0.;count=0
        for pred in p['predictions']:
            r=rows[pred['query_id']];X=r['modes']['REAL']['X'].numpy()
            for m,v in pred['models'].items():
                t=np.array([float.fromhex(x) for x in p['models'][m]['theta_hex']]);z=np.sum(X*t[:6],axis=1)+t[6];want=np.array([float.fromhex(x) for x in v['logits_hex']]);e=float(abs(z-want).max());need(e<2e-10,'INDEPENDENT_LOGITS');err=max(err,e);count+=127
                k=int(z.argmax());pos=r['challenger_positions'][k] if z[k]>0 else r['winner'];need(r['candidate_physical_rows'][pos]==v['selected'],'INDEPENDENT_ACTION')
        write(folder/'validation.json',dict(status='SIX_CAUSE_COVERAGE_FRESH_FIT_NUMPY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),logit_checks=count,max_abs_error=err,heldout_label_reads=0))
    else:
        write(folder/'payload.json',p);subprocess.run([sys.executable,__file__,'verify','--index',str(index)],check=True)

def join():
    a=read(AUTH);ps=[];vs=[]
    for i in range(35):
        f=OUT/f'fit{i:02d}';v=read(f/'validation.json');p=read(checked(v['payload']));need(v['status']=='SIX_CAUSE_COVERAGE_FRESH_FIT_NUMPY_PASS' and v['authority']==p['authority']==bind(AUTH),'ALL35_VALIDATED');ps.append(p);vs.append(bind(f/'validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=vs))
    roles=read(checked(a['join_sources']['curator']))['records'];roles={r['query_id']:r for r in roles};old=read(checked(a['join_sources']['old_result']));oldrows={r['query_id']:r for r in old['rows']};labels,_=N.P.gallery_labels();rows={}
    for q,r in roles.items():rows[q]=dict(query_id=q,original_query_id=r['original_query_id'],fold=r['outer_fold'],component=r['component'],group=r['group'],correct={k:oldrows[q]['correct'][k] for k in ('RAW','BASE7_ALL','BASE7_SMALL128')},target_in_C128=oldrows[q]['target_in_C128'])
    for p in ps:
        c=p['config'];train=read(checked(a['fold_sources'][str(c['fold'])]['train_roles']))['records'];trainids={r['identity'] for r in train};traincomps={r['component'] for r in train}
        for pred in p['predictions']:
            role=roles[pred['query_id']];need(role['outer_fold']==c['fold'] and role['identity'] not in trainids and role['component'] not in traincomps,'OUTER_IDENTITY_COMPONENT_DISJOINT')
            for m,z in pred['models'].items():
                key=m if c['block']=='ALL' else m+f"_S{c['seed']}";correct=labels[z['selected']]==role['identity'];rows[pred['query_id']]['correct'][key]=correct
                if m=='ALL_COST4':need(correct==oldrows[pred['query_id']]['correct']['BASE7_ALL'],'ORIGINAL_ALL_BASE_UNCHANGED')
    rs=list(rows.values());models=list(rs[0]['correct']);need(len(rs)==593 and all(set(r['correct'])==set(models) for r in rs),'WHOLE593_ALL_MODELS')
    counts={m:sum(r['correct'][m] for r in rs) for m in models};comparisons={}
    pairs=[('ALL_COST4','ALL_CE')]
    for s in range(3):
        for obj in ('COST4','CE'):
            pairs.extend([(f'NARROW128_{obj}_S{s}',f'BROAD128_{obj}_S{s}'),(f'PER_ID_1_{obj}_S{s}',f'PER_ID_2_{obj}_S{s}'),(f'PER_ID_2_{obj}_S{s}',f'PER_ID_4_{obj}_S{s}')])
        for subset in ('BROAD128','NARROW128','PER_ID_1','PER_ID_2','PER_ID_4'):pairs.append((f'{subset}_COST4_S{s}',f'{subset}_CE_S{s}'))
    for b,m in pairs:comparisons[b+'__to__'+m]=H.groupstats(rs,b,m)
    result=dict(status='SIX_CAUSE_H593_COVERAGE_PROTOCOL_OOF5_COMPLETE',authority=bind(AUTH),counts=counts,comparisons=comparisons,rows=rs,training_samples=[dict(config=p['config'],models=p['models']) for p in ps],recall_present=sum(r['target_in_C128'] for r in rs),candidate_source='RAW natural C128',action='all127 max>0 SWITCH else HOLD',evidence_level='already opened H593, fixed identity/source component fivefold diagnostic; three overlapping training sample draws',external_confirmation=False,HYP_GO_claimed=False,limits=['Seeds are overlapping resamples, not independent experiments','Within-identity curves apply to identities with at least4 recall-present training images','Broad/narrow compares coverage at128 recall-present images; it does not hold identity composition fixed','ALL_CE versus ALL_COST4 isolates objective under FULL protocol; not original PAIR plus FULL checkpoint'])
    write(OUT/'result.json',result);need(counts['RAW']==426 and counts['ALL_COST4']==counts['BASE7_ALL']==440,'HISTORICAL_BASELINE_REPLAY')
    write(OUT/'validation.json',dict(status='SIX_CAUSE_COVERAGE_COUNTS_ORIGINAL_BASE_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),fit_validations=vs,queries=593,recall_present=result['recall_present']))
    print(dict(counts=counts,recall_present=result['recall_present']),flush=True)

def preflight():
    z=torch.zeros((4,127),dtype=torch.float64,requires_grad=True);y=torch.tensor([-1,0,31,126]);a=H.loss(z,y);terms=[]
    for v,t in zip(z,y):
        if t==-1:terms.append(4*F.softplus(v.max()))
        else:terms.append(F.softplus(-v[t])+4*F.softplus(torch.cat((v[:t],v[t+1:])).max()))
    b=torch.stack(terms).mean();need(torch.equal(a,b),'COST4_LOSS_PARITY');need(torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z)[0]),'TIED_MAX_GRADIENT_PARITY')
    fake=[dict(query_id=f'q{i}_{j}',execution_ordinal=i*8+j) for i in range(40) for j in range(8)];roles={r['query_id']:dict(identity=r['query_id'].split('_')[0],component=r['query_id'].split('_')[0]) for r in fake}
    ss=subsets(fake,roles,0,'COVERAGE');need(len(ss['BROAD128'])==len(ss['NARROW128'])==128,'BUDGET_PREFLIGHT')
    ss=subsets(fake,roles,0,'WITHIN');need([len(ss[f'PER_ID_{k}']) for k in (1,2,4)]==[40,80,160],'FIXED_IDENTITY_PREFLIGHT')
    write(OUT/'preflight.json',dict(status='SIX_CAUSE_COVERAGE_SYNTHETIC_PASS',authority=bind(AUTH),natural_training_updates=0,checks=['original_cost4_loss_and_tied_gradient','equal128_coverage_budget','nested_fixed_identity_images']))
    print('SIX_CAUSE_COVERAGE_SYNTHETIC_PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','fit','verify','join']);p.add_argument('--index',type=int);args=p.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(args.stage,args.index)
    if args.stage=='preflight':preflight()
    elif args.stage=='join':join()
    else:fit(args.index,args.stage=='verify')
