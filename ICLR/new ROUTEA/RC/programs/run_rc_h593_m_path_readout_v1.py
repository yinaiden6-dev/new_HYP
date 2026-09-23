#!/usr/bin/env python3
"""CPU retrieval readout of sealed RoMa M-path interventions; no image inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_quality_operator_eval_v1 as E
import rc_roma_m_inside_v1 as I
OUT=ROOT/'results/rc_h593_m_path_readout_v1'
INSIDE=ROOT/'results/rc_h593_m_inside_v1'
POOL=ROOT/'cache/rc_h593_shared_pooling_v1'
AUTH=ROOT/'registry/rc_h593_m_path_readout_authority_v1_20260922.json'
LAUNCH=ROOT/'slurm/rc_h593_m_path_readout_v1.sbatch'
PLAN=ROOT/'plan/RC_H593_M_PATH_RETRIEVAL_READOUT_V1_20260922.md'
REPORT=ROOT/'reports/REPORT_H593_M_PATH_RETRIEVAL_READOUT_V1_20260922.md'
ARMS=tuple(b+'_'+s for b in I.BRANCHES for s in ('COARSE','HR1'))+('FREE',)
NATIVE='A1J1P1_HR1'
KINDS=('COST1','CE')
read,write,bind,checked=E.read,E.write,E.bind,E.checked
save=E.C.save


def bits(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def tensor_sha(t):
    return hashlib.sha256(str((str(t.dtype),tuple(t.shape))).encode()+t.contiguous().numpy().tobytes()).hexdigest()


def prepare():
    assert not AUTH.exists()
    parent=read(E.AUTH)
    for b in parent['code_sources'].values():checked(b)
    baseline={}
    for f in range(5):
        p=E.OUT/f'fold{f}';v=read(p/'validation.json');checked(v['payload'])
        assert v['status']=='QUALITY_OPERATOR_EVAL_FRESH_NUMPY_PASS' and v['authority']==bind(E.AUTH)
        baseline[str(f)]=dict(payload=v['payload'],validation=bind(p/'validation.json'))
    oldv=read(E.OUT/'validation.json');checked(oldv['result'])
    assert oldv['status']=='QUALITY_OPERATOR_EVAL_ALL_COUNTS_PASS' and oldv['authority']==bind(E.AUTH)
    sources=list(parent['code_sources'].values())+[bind(p) for p in (Path(__file__),Path(I.__file__),LAUNCH,PLAN)]
    write(AUTH,dict(status='M_PATH_READOUT_AUTHORIZED',parent=bind(E.AUTH),sources=sources,
        inside_authority=bind(ROOT/'registry/rc_h593_m_inside_authority_v1_20260922.json'),
        operator_authority=parent['operator_authority'],public_sources=parent['public_sources'],
        fold_sources=parent['fold_sources'],baseline_sources=baseline,join_sources=parent['join_sources'],
        operator_result=oldv['result'],operator_validation=bind(E.OUT/'validation.json'),arms=list(ARMS),
        training='Original retrieval-only 2000-step FP64 AdamW lr=.03 wd=.001; original folds and order',
        frozen='Per-fold sealed M1Q0R0 refit head',baseline_reuse='Reuse native simplified and FREE parameters after all feature/logit bits match',
        primary_contrasts=[['COST1_REFIT_A1J0P0_COARSE','COST1_REFIT_A1J1P1_COARSE'],['COST1_REFIT_NO_REFINER_DELTA_HR1','COST1_REFIT_A1J1P1_HR1']],
        primary_intervals='97.5 percent equal-component bootstrap intervals for two prespecified COST1 comparisons; secondary 95 percent',
        scope='Opened H593 OOF5 conditional predictor-path readout, original geometry fixed',
        new_roma_forwards=0,new_encoder_forwards=0,external_GO=False))
    x=torch.tensor([[.2,0.,.8],[.1,.4,0.]],dtype=torch.float64)
    q=assemble(x,torch.tensor([2.,3.,4.],dtype=torch.float64),torch.tensor([4.,4.,4.],dtype=torch.float64),[0,2],1,torch.tensor([-.2,.3],dtype=torch.float64))
    expected=numpy_x(x.numpy(),q['S'].numpy(),[0,2],1,np.array([-.2,.3]))
    assert np.max(np.abs(q['X'].numpy()-expected))<1e-13
    assert E.choose(np.array([0.,0.]),dict(candidate_physical_rows=[8,1,7],challenger_positions=[0,2],winner=1))==1
    write(OUT/'preflight.json',dict(status='M_PATH_READOUT_SYNTHETIC_PASS',authority=bind(AUTH),checks=['FP64_formula','zero_mass','NumPy_feature_formula','HOLD_zero'],natural_training=0))
    print('M_PATH_READOUT_PREPARED',flush=True)


def guard(stage,fold):
    a=read(AUTH)
    assert a['status']=='M_PATH_READOUT_AUTHORIZED'
    for b in [a['parent'],a['inside_authority'],a['operator_authority'],*a['sources']]:checked(b)
    assert os.environ.get('SLURM_JOB_ID')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    if stage in ('fit','verify'):
        assert fold in range(5)
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
        allow.update(Path(b['path']).resolve() for b in a['baseline_sources'][str(fold)].values())
    if stage in ('join','join-verify'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        allow.update(Path(a[k]['path']).resolve() for k in ('operator_result','operator_validation'))
        for bs in a['fold_sources'].values():allow.update(Path(b['path']).resolve() for b in bs.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        assert not any(k in s for k in ('d1-mi','d1_mi','formal392','/grozi/','/isic/','/target_join/','rc_opened_')),s
        if 'curator_roles' in s:assert stage in ('join','join-verify'),s
        if ROOT/'reports' in p.parents:assert stage=='join' and p==REPORT,s
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage in ('fit','verify'):
                first=p.relative_to(OUT).parts[0];own=first in ('inputs','ready.json','preflight.json','pilot.json',f'fold{fold}')
            source=(INSIDE in p.parents or E.C.OUT in p.parents) and p.relative_to(INSIDE if INSIDE in p.parents else E.C.OUT).parts[0].startswith('query')
            assert own or source or p in allow,s
    sys.addaudithook(audit)
    assert read(OUT/'preflight.json')['authority']==bind(AUTH)
    return a


def assemble(m,sums,counts,cs,w,rawgap):
    s=m*sums[None]/counts[None];l=s/m.clamp_min(1e-12)
    x=m.new_zeros((len(m),len(cs),6));x[:,:,0]=rawgap
    for k,z in ((1,s),(2,m),(3,l)):
        c=z[:,cs];v=z[:,w:w+1];x[:,:,k]=(c-v)/(c.abs()+v.abs()+1e-12)
    assert bool(torch.isfinite(x).all())
    return dict(M=m,S=s,L=l,X=x)


def numpy_x(m,s,cs,w,rawgap):
    x=np.zeros((len(m),len(cs),6),dtype=np.float64);x[:,:,0]=rawgap
    for k,z in ((1,s),(2,m),(3,s/np.maximum(m,1e-12))):
        c=z[:,cs];v=z[:,w:w+1];x[:,:,k]=(c-v)/(abs(c)+abs(v)+1e-12)
    return x


def make_query(a,index):
    path=OUT/'inputs'/f'query{index:03d}.pt';val=path.with_suffix('.json')
    if val.exists():
        v=read(val);assert v['authority']==bind(AUTH) and v['status']=='M_PATH_INPUT_PASS';checked(v['payload']);return
    d=INSIDE/f'query{index:03d}';iv=read(d/'inside_validation.json');v=read(d/'validation.json')
    im=read(checked(iv['payload']));p=read(checked(v['payload']))
    assert iv['status']=='M_INSIDE_PATHS_PASS' and v['status']=='M_VISUAL_ORIGIN_QUERY_PASS'
    assert iv['authority']==v['authority']==im['authority']==p['authority']==a['inside_authority']
    od=E.C.OUT/d.name;ov=read(od/'validation.json');op=read(checked(ov['payload']))
    assert ov['status']=='QUALITY_OPERATOR_QUERY_PASS' and ov['authority']==op['authority']==a['operator_authority']
    assert op['query_id']==p['query_id']==im['query_id'] and op['candidate_physical_rows']==p['candidate_physical_rows']
    assert op['source_image_sha256']==p['query_image']['sha256'] and op['winner']==p['raw_winner'] and op['execution_ordinal']==index
    ot=torch.load(checked(op['intermediates']),map_location='cpu',weights_only=True,mmap=True)
    assert ot['authority']==a['operator_authority'] and ot['query_id']==op['query_id']
    means=torch.empty((len(ARMS)-1,128,2),dtype=torch.float64);sums=[];counts=[]
    assert len(im['sources'])==len(im['records'])==len(ot['pairs'])==128
    for pos,(b,record,freepair) in enumerate(zip(im['sources'],im['records'],ot['pairs'])):
        assert record['candidate_position']==pos and record['physical_row']==freepair['physical_row']==op['candidate_physical_rows'][pos]
        z=torch.load(checked(b),map_location='cpu',weights_only=True,mmap=True)
        assert z['authority']==a['inside_authority'] and z['query_sha']==op['source_image_sha256'] and z['index']==index
        assert z['branches']==list(I.BRANCHES)
        for bi,branch in enumerate(I.BRANCHES):
            for si,stage in enumerate(('COARSE','HR1')):
                for side in (0,1):
                    item=z['sides'][side]['coarse'][branch] if stage=='COARSE' else z['sides'][side]['stages'][stage]['branches'][branch]
                    weights=item['weights'];assert weights.dtype==torch.float64
                    assert float(weights.mean()).hex()==float(item['mean_weight']).hex()
                    means[bi*2+si,pos,side]=item['mean_weight']
        assert freepair['candidate_position']==pos
        free=freepair['detail']['free_maxsim'];assert free.dtype==torch.float64 and free.ndim==1 and len(free)>0
        one=torch.ones_like(free)
        sums.append((one*free).sum());counts.append(one.sum().clamp_min(1e-12))
    m=torch.cat([torch.sqrt(means[:,:,0]*means[:,:,1]),torch.ones((1,128),dtype=torch.float64)])
    sums=torch.stack(sums);counts=torch.stack(counts);cs=op['challenger_positions'];w=op['winner']
    gap=torch.tensor(op['modes']['M0Q0R0']['X'],dtype=torch.float64)[:,0]
    values=assemble(m,sums,counts,cs,w,gap)
    for name,oldmode in ((NATIVE,'M1Q0R0'),('FREE','M0Q0R0')):
        ai=ARMS.index(name);old=op['modes'][oldmode]
        assert bits(values['X'][ai],torch.tensor(old['X'],dtype=torch.float64)),('BASELINE_X_BITS',index,name)
        assert bits(values['M'][ai],torch.tensor([r['visibility_mass'] for r in old['scores']],dtype=torch.float64)),('BASELINE_M_BITS',index,name)
        assert bits(values['S'][ai],torch.tensor([r['real_score'] for r in old['scores']],dtype=torch.float64)),('BASELINE_S_BITS',index,name)
    row={k:op[k] for k in ('query_id','execution_ordinal','source_image_sha256','candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')}
    value=dict(authority=bind(AUTH),row=row,arms=list(ARMS),**values,means=means,free_sums=sums,token_counts=counts,
        original_full_X=torch.tensor(op['modes']['NATIVE']['X'],dtype=torch.float64),
        sources=dict(inside_validation=bind(d/'inside_validation.json'),native_validation=bind(d/'validation.json'),operator_validation=bind(od/'validation.json'),free_vectors=op['intermediates']),
        label_reads=0,new_roma_forwards=0)
    if not path.exists():save(path,value)
    else:
        old=torch.load(path,map_location='cpu',weights_only=True);assert old['authority']==value['authority'] and old['row']==row and old['sources']==value['sources']
        for k in ('X','M','S','L','means','free_sums','token_counts','original_full_X'):assert bits(old[k],value[k]),k
    verify_input(path)


def verify_input(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    assert d['authority']==bind(AUTH) and d['arms']==list(ARMS) and d['label_reads']==0
    assert d['X'].shape==(25,127,6) and d['M'].shape==(25,128) and d['means'].shape==(24,128,2)
    for k in ('X','M','S','L','means','free_sums','token_counts','original_full_X'):assert d[k].dtype==torch.float64 and bool(torch.isfinite(d[k]).all()),k
    r=d['row'];x=numpy_x(d['M'].numpy(),d['S'].numpy(),r['challenger_positions'],r['winner'],d['X'][0,:,0].numpy())
    error=float(abs(x-d['X'].numpy()).max());assert error<2e-12
    ms=np.sqrt(d['means'].numpy()[:,:,0]*d['means'].numpy()[:,:,1]);assert np.max(abs(ms-d['M'][:-1].numpy()))<2e-12
    assert np.max(abs(d['M'].numpy()*d['free_sums'].numpy()/d['token_counts'].numpy()-d['S'].numpy()))<2e-12
    write(Path(path).with_suffix('.json'),dict(status='M_PATH_INPUT_PASS',authority=bind(AUTH),payload=bind(path),index=r['execution_ordinal'],arms=len(ARMS),candidates=128,max_numpy_error=error,label_reads=0))


def build(a,pilot=False):
    start=time.monotonic()
    for i in (range(1) if pilot else range(593)):
        if time.monotonic()-start>240:print(dict(status='INPUT_NORMAL_PARTIAL',next_index=i),flush=True);return
        make_query(a,i)
        if (i+1)%8==0:print(dict(event='INPUTS_READY',queries=i+1,seconds=time.monotonic()-start),flush=True)
    if pilot:
        subprocess.run([sys.executable,__file__,'pilot-verify'],check=True)
        write(OUT/'pilot.json',dict(status='M_PATH_FULL128_PILOT_PASS',authority=bind(AUTH),input_validation=bind(OUT/'inputs/query000.json'),seconds=time.monotonic()-start,natural_training=0))
        print('M_PATH_FULL128_PILOT_PASS',flush=True)
    else:
        seals=[]
        for i in range(593):
            p=OUT/'inputs'/f'query{i:03d}.json';v=read(p);assert v['status']=='M_PATH_INPUT_PASS' and v['index']==i and v['authority']==bind(AUTH);checked(v['payload']);seals.append(bind(p))
        write(OUT/'ready.json',dict(status='M_PATH_ALL593_INPUTS_PASS',authority=bind(AUTH),queries=593,arms=list(ARMS),validations=seals,label_reads=0))


def data():
    ready=read(OUT/'ready.json');assert ready['status']=='M_PATH_ALL593_INPUTS_PASS' and ready['authority']==bind(AUTH)
    values=[]
    for b in ready['validations']:
        v=read(checked(b));values.append(torch.load(checked(v['payload']),map_location='cpu',weights_only=True))
    assert [v['row']['execution_ordinal'] for v in values]==list(range(593))
    return values


def compute(a,fold):
    ds=data();fs=a['fold_sources'][str(fold)];bs=a['baseline_sources'][str(fold)]
    bv=read(checked(bs['validation']));old=read(checked(bs['payload']));assert bv['payload']==bs['payload'] and old['authority']==a['parent']
    roles={r['query_id']:r for r in read(checked(fs['train_roles']))['records']}
    split=read(checked(a['public_sources']['split']))['folds'][fold];trainset=set(split['train_query_ids']);heldset=set(split['heldout_query_ids'])
    assert set(roles)==trainset and not trainset&heldset
    train=[d for d in ds if d['row']['query_id'] in trainset];held=[d for d in ds if d['row']['query_id'] in heldset]
    assert [d['row']['query_id'] for d in train]==old['train_query_ids']
    assert not {d['row']['source_image_sha256'] for d in train}&{d['row']['source_image_sha256'] for d in held}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    ys=[E.H.target_position(d['row'],roles[d['row']['query_id']]['identity'],labels) for d in train];idx=[i for i,y in enumerate(ys) if y>=-1]
    effective=[train[i]['row']['query_id'] for i in idx];assert effective==old['effective_train_query_ids'];y=torch.tensor([ys[i] for i in idx])
    params={};zs={};memo={};timing=[];oldpred={p['query_id']:p for p in old['predictions']}
    for ai,arm in enumerate(ARMS):
        x=torch.stack([train[i]['X'][ai] for i in idx]);xx=torch.stack([d['X'][ai] for d in held]);digest=tensor_sha(x)
        for kind in KINDS:
            fixed=torch.tensor([float.fromhex(v) for v in old['parameters'][f'{kind}_REFIT_M1Q0R0']['theta_hex']],dtype=torch.float64)
            name=f'{kind}_FROZEN_{arm}';params[name]=dict(theta_hex=E.hx(fixed),arm=arm,method='FROZEN');zs[name]=xx@fixed[:-1]+fixed[-1]
            start=time.monotonic();name=f'{kind}_REFIT_{arm}';key=(kind,digest)
            if arm in (NATIVE,'FREE'):
                source=f'{kind}_REFIT_'+('M1Q0R0' if arm==NATIVE else 'M0Q0R0')
                t=torch.tensor([float.fromhex(v) for v in old['parameters'][source]['theta_hex']],dtype=torch.float64);reused=source
            elif key in memo:t,reused=memo[key]
            else:
                torch.manual_seed(17);t=E.L.train(x,y,kind);reused=None
            params[name]=dict(theta_hex=E.hx(t),arm=arm,method='REFIT',training_X_sha256=digest,reused_from=reused)
            zs[name]=xx@t[:-1]+t[-1];memo[key]=(t,name)
            if arm in (NATIVE,'FREE'):
                for j,d in enumerate(held):assert dict(logits_hex=E.hx(zs[name][j]),selected=E.choose(zs[name][j],d['row']))==oldpred[d['row']['query_id']]['models'][source],('BASELINE_LOGITS',fold,arm,kind)
            timing.append(dict(model=name,seconds=time.monotonic()-start,reused_from=reused));print(dict(event='READOUT_HEAD_DONE',fold=fold,**timing[-1]),flush=True)
    full=torch.stack([d['original_full_X'] for d in held])
    for kind in KINDS:
        t=torch.tensor([float.fromhex(v) for v in old['parameters'][f'{kind}_FROZEN_NATIVE']['theta_hex']],dtype=torch.float64);name=kind+'_ORIGINAL_FULL'
        params[name]=dict(theta_hex=E.hx(t),arm='ORIGINAL_FULL',method='SEALED_BASELINE');zs[name]=full@t[:-1]+t[-1]
        for j,d in enumerate(held):assert dict(logits_hex=E.hx(zs[name][j]),selected=E.choose(zs[name][j],d['row']))==oldpred[d['row']['query_id']]['models'][f'{kind}_FROZEN_NATIVE']
    preds=[dict(query_id=d['row']['query_id'],execution_ordinal=d['row']['execution_ordinal'],models={m:dict(logits_hex=E.hx(z[j]),selected=E.choose(z[j],d['row'])) for m,z in zs.items()}) for j,d in enumerate(held)]
    value=dict(status='M_PATH_READOUT_FOLD_SEALED',authority=bind(AUTH),fold=fold,train_query_ids=old['train_query_ids'],effective_train_query_ids=effective,parameters=params,predictions=preds,inputs=bind(OUT/'ready.json'),baseline=bs,heldout_label_reads=0)
    return value,{d['row']['query_id']:d for d in ds},timing


def fit(a,fold,replay=False):
    dest=OUT/f'fold{fold}'
    if not replay and (dest/'validation.json').exists():
        v=read(dest/'validation.json');assert v['authority']==bind(AUTH) and v['status']=='M_PATH_READOUT_FRESH_NUMPY_PASS';checked(v['payload']);return
    p,ds,timing=compute(a,fold)
    if not replay:
        write(dest/'payload.json',p)
        if not (dest/'runtime.json').exists():write(dest/'runtime.json',dict(job=os.environ['SLURM_JOB_ID'],fits=timing))
        subprocess.run([sys.executable,__file__,'verify','--fold',str(fold)],check=True);return
    assert p==read(dest/'payload.json'),'FRESH_READOUT_TRAINING_PARITY'
    error=0.;checks=0
    for pred in p['predictions']:
        d=ds[pred['query_id']]
        for name,par in p['parameters'].items():
            x=d['original_full_X'] if par['arm']=='ORIGINAL_FULL' else d['X'][ARMS.index(par['arm'])]
            t=np.array([float.fromhex(v) for v in par['theta_hex']]);z=np.sum(x.numpy()*t[:-1],axis=1)+t[-1]
            old=pred['models'][name];error=max(error,float(abs(z-np.array([float.fromhex(v) for v in old['logits_hex']])).max()));checks+=127
            assert error<2e-10 and E.choose(z,d['row'])==old['selected']
    write(dest/'validation.json',dict(status='M_PATH_READOUT_FRESH_NUMPY_PASS',authority=bind(AUTH),payload=bind(dest/'payload.json'),logit_checks=checks,max_abs_error=error,heldout_label_reads=0))


def join(a,replay=False):
    ps=[];seals=[]
    for f in range(5):
        path=OUT/f'fold{f}/validation.json';v=read(path);p=read(checked(v['payload']));assert v['status']=='M_PATH_READOUT_FRESH_NUMPY_PASS' and p['fold']==f and p['authority']==v['authority']==bind(AUTH);ps.append(p);seals.append(bind(path))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=seals))
    roles={r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    ds={d['row']['query_id']:d['row'] for d in data()};old={r['query_id']:r for r in read(checked(a['operator_result']))['rows']};rows=[]
    for p in ps:
        tr=read(checked(a['fold_sources'][str(p['fold'])]['train_roles']))['records'];ids={r['identity'] for r in tr};groups={r['component'] for r in tr}
        for pred in p['predictions']:
            r=ds[pred['query_id']];role=roles[pred['query_id']];assert role['outer_fold']==p['fold'] and role['identity'] not in ids and role['component'] not in groups
            selected=dict(RAW=r['candidate_physical_rows'][r['winner']],**{m:v['selected'] for m,v in pred['models'].items()});correct={m:labels[s]==role['identity'] for m,s in selected.items()}
            rank=r['raw_ranked_physical_rows'];target=next(i+1 for i,s in enumerate(rank) if labels[s]==role['identity']);ranks={m:1 if correct[m] else target+int(rank.index(s)+1>target) for m,s in selected.items()}
            for kind in KINDS:
                for new,prior in [(kind+'_ORIGINAL_FULL',kind+'_FROZEN_NATIVE'),(kind+'_REFIT_'+NATIVE,kind+'_REFIT_M1Q0R0'),(kind+'_REFIT_FREE',kind+'_REFIT_M0Q0R0')]:assert selected[new]==old[r['query_id']]['selected'][prior] and correct[new]==old[r['query_id']]['correct'][prior]
            rows.append(dict(query_id=r['query_id'],execution_ordinal=r['execution_ordinal'],fold=p['fold'],component=role['component'],target_in_C128=target<=128,selected=selected,correct=correct,ranks=ranks))
    rows.sort(key=lambda r:r['execution_ordinal']);assert len(rows)==593 and sum(r['target_in_C128'] for r in rows)==570
    summary={};comparisons={}
    for m in rows[0]['selected']:
        c=E.comparison(rows,'RAW',m);summary[m]=dict(correct=sum(r['correct'][m] for r in rows),total=593,MRR=float(np.mean([1/r['ranks'][m] for r in rows])),rescue_vs_RAW=c['rescue'],break_vs_RAW=c['loss'],net_vs_RAW=c['net'],by_fold={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)})
        if m!='RAW':comparisons[m]=E.comparison(rows,m.split('_')[0]+'_REFIT_'+NATIVE,m)
    primary=[]
    for base,new in a['primary_contrasts']:
        c=E.comparison(rows,base,new);gs={}
        for r in rows:gs.setdefault(r['component'],[]).append(int(r['correct'][new])-int(r['correct'][base]))
        d=np.array([np.mean(v) for _,v in sorted(gs.items())]);assert len(d)==64
        rng=np.random.default_rng(20260922);boot=d[rng.integers(0,64,size=(10000,64))].mean(1)
        c['family_adjusted_bootstrap97_5']=list(map(float,np.quantile(boot,[.0125,.9875])));primary.append(c)
    assert summary['RAW']['correct']==426 and summary['COST1_ORIGINAL_FULL']['correct']==481 and summary['CE_ORIGINAL_FULL']['correct']==486 and summary['COST1_REFIT_'+NATIVE]['correct']==481
    result=dict(status='M_PATH_READOUT_ALL593_COMPLETE',authority=bind(AUTH),summary=summary,comparisons=comparisons,primary=primary,rows=rows,candidate_recall=570,total=593,external_GO=False,scope=a['scope'])
    if replay:
        assert result==read(OUT/'result.json');write(OUT/'validation.json',dict(status='M_PATH_READOUT_COUNTS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),fold_validations=seals));return
    write(OUT/'result.json',result);subprocess.run([sys.executable,__file__,'join-verify'],check=True)
    lines=['# H593：M上游通路的实际检索读出','','原分组五折、自然RAW C128、593分母。COST1为主、CE辅助；原几何与采样固定。','', '|模型|正确/593|MRR|对RAW救回/损失|','|---|---:|---:|---:|']
    for m,v in summary.items():lines.append(f"|{m}|{v['correct']}|{v['MRR']:.6f}|{v['rescue_vs_RAW']}/{v['break_vs_RAW']}|")
    lines+=['','两项主要比较（97.5%组件区间）：','',json.dumps(primary,ensure_ascii=False,indent=2),'','固定头与重训需合看。J/P依赖于原始特征；置零改变分布。HR1的A-only仍包含原生配对细化，不能称无跨图信息。']
    REPORT.write_text('\n'.join(lines)+'\n');print('M_PATH_READOUT_ALL593_COMPLETE',flush=True)


def command(args):return subprocess.run(args,cwd=ROOT,check=True,text=True,capture_output=True,timeout=60).stdout


def submit(stage,dependency=None,array=None,key=None,partition=None):
    dest=OUT/'dispatch'/((key or stage)+'.json')
    if dest.exists():
        prior=read(dest);assert prior['authority']==bind(AUTH) and prior['stage']==stage and prior['dependency']==dependency and prior['array']==array
        return prior['job']
    assert stage in ('pilot','build','follow','fit','join')
    assert len(list((OUT/'dispatch').glob('*.json')))<512,'BOUNDED_CONTINUATION'
    args=['sbatch','--parsable','--hold','--time='+('00:05:00' if stage in ('pilot','build','follow') else '00:15:00')]
    if partition:
        assert partition in ('cpuonly','dev_cpuonly');args+=['--partition='+partition]
    if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    if array:args+=['--array='+array]
    args += [str(LAUNCH),stage]
    job=command(args).strip().split(';')[0];assert job.isdigit()
    try:
        spool=OUT/'dispatch'/f'spool_{job}.sh';spool.parent.mkdir(parents=True,exist_ok=True);command(['scontrol','write','batch_script',job,str(spool)]);assert spool.read_bytes()==LAUNCH.read_bytes()
        write(dest,dict(job=job,stage=stage,dependency=dependency,array=array,authority=bind(AUTH),spool=bind(spool),partition=partition or 'cpuonly'));command(['scontrol','release',job])
    except BaseException:command(['scancel',job]);raise
    print(dict(submitted=job,stage=stage,dependency=dependency,array=array),flush=True);return job


def follow(a):
    pilot=read(OUT/'pilot.json');assert pilot['status']=='M_PATH_FULL128_PILOT_PASS' and pilot['authority']==bind(AUTH)
    complete=sum((INSIDE/f'query{i:03d}/inside_validation.json').exists() and (INSIDE/f'query{i:03d}/validation.json').exists() for i in range(593))
    if complete<593:
        waves=[w for p in (POOL/'dispatch').glob('*.json') if (w:=read(p)).get('family')=='inside'];assert waves
        wave=max(waves,key=lambda w:int(w['job']));cb=wave['callback']
        states=command(['sacct','-X','-n','-P','-j',cb,'--format=JobID,State,ExitCode'])
        assert not any(s in states for s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')),states
        dest=OUT/'dispatch'/('after_inside_'+cb+'.json')
        if dest.exists():assert read(dest)['job']!=os.environ['SLURM_JOB_ID'],'UPSTREAM_CALLBACK_DID_NOT_ADVANCE'
        submit('follow',dependency=cb,key='after_inside_'+cb);print(dict(status='WAITING_SEALED_INSIDE_INPUTS',complete=complete,total=593,after=cb),flush=True);return
    if not (OUT/'ready.json').exists():
        key='build_after_'+os.environ['SLURM_JOB_ID'];job=submit('build',key=key);submit('follow',dependency=job,key='after_'+job);return
    if not (OUT/'validation.json').exists():
        fitjob=submit('fit',array='0-4%5',key='fits');submit('join',dependency=fitjob,key='join')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','pilot','pilot-verify','build','follow','fit','verify','join','join-verify'));p.add_argument('--fold',type=int);args=p.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage,args.fold)
        if args.stage in ('pilot','build'):build(a,args.stage=='pilot')
        elif args.stage=='pilot-verify':verify_input(OUT/'inputs/query000.pt')
        elif args.stage=='follow':follow(a)
        elif args.stage in ('fit','verify'):fit(a,args.fold,args.stage=='verify')
        else:join(a,args.stage=='join-verify')
