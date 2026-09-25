#!/usr/bin/env python3
"""CPU-only post-hoc readout on frozen V4 scores; never reads M as a feature."""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import run_rc_internal_m_condition_scale_v4 as V

ROOT=V.ROOT
PROBE=ROOT/'results/rc_internal_m_v4_probe_v1'
OUT=ROOT/'results/rc_internal_m_content_preserving_readout_v1'
AUTH=ROOT/'registry/rc_internal_m_content_preserving_readout_v1_authority_20260924.json'
PLAN=ROOT/'plan/RC_INTERNAL_M_V4_CONTENT_PRESERVING_READOUT_V1_20260924.md'
REPORT=ROOT/'reports/REPORT_INTERNAL_M_CONTENT_PRESERVING_READOUT_V1_20260924.md'
ARMS=('ORIGINAL3_CONTINUE','PRESERVE_REAL','PRESERVE_TRAIN_CONSTANT','PRESERVE_TRAIN_SHUFFLED')
MODES={'ORIGINAL3_CONTINUE':'REAL','PRESERVE_REAL':'REAL','PRESERVE_TRAIN_CONSTANT':'TRAIN_CONSTANT','PRESERVE_TRAIN_SHUFFLED':'TRAIN_SHUFFLED'}
read,write,bind,checked,need=V.read,V.write,V.bind,V.checked,V.need


def features(row,L0,L,original=False):
    raw=np.asarray(row['raw_scores'],dtype=np.float64); old=np.asarray(L0,dtype=np.float64); new=np.asarray(L,dtype=np.float64)
    need(all(x.shape==(128,) and np.isfinite(x).all() for x in (raw,old,new)), 'FULL_FINITE_C128')
    w=row['winner_index'];idx=[i for i in range(128) if i!=w]
    need(row['challenger_positions']==idx and w==int(raw.argmax()), 'ORIGINAL_CANDIDATE_AXIS')
    def sym(x):return (x[idx]-x[w])/(np.abs(x[idx])+abs(x[w])+1e-12)
    x=[(raw[idx]-raw[w])/max(float(raw.std()),1e-12),sym(old)]
    if not original:x.append(sym(new)-sym(old))
    return np.stack(x+[np.ones(127)],axis=1)


def batch_loss(z,rows):
    import torch
    from torch.nn import functional as F
    need(z.shape==(len(rows),127),'FULL_BATCH_LOGITS')
    target=torch.tensor([0 if r['target_positions'][0]==r['winner_index'] else r['challenger_positions'].index(r['target_positions'][0]) for r in rows],device=z.device)
    raw_right=torch.tensor([r['target_positions'][0]==r['winner_index'] for r in rows],device=z.device)
    indices=torch.arange(len(rows),device=z.device)
    masked=z.clone();masked[indices,target]=-torch.inf
    values=torch.where(raw_right,F.softplus(z.amax(dim=1)),F.softplus(-z[indices,target])+F.softplus(masked.amax(dim=1)))
    return values.mean()


def decision(row,x,theta):
    z=np.asarray(x)@np.asarray(theta)
    need(z.shape==(127,) and np.isfinite(z).all(),'FINITE_LOGITS')
    idx=row['challenger_positions'];pos=idx[int(z.argmax())] if float(z.max())>0 else row['winner_index']
    scores=np.zeros(128);scores[idx]=z
    return dict(prediction_position=pos,prediction_id=row['candidate_ids'][pos],prediction_identity=row['candidate_identities'][pos],
                switched=pos!=row['winner_index'],logits=z.tolist(),scores128=scores.tolist(),theta=list(theta),features=np.asarray(x).tolist())


def input_path(split,mode,q):
    if split=='probe':return PROBE/'predictions'/mode/(q+'.json')
    mapping={'REAL':('PRE_REAL','native'),'REAL_CONSTANT':('PRE_REAL','constant'),'REAL_SHUFFLED':('PRE_REAL','shuffled'),
             'TRAIN_SHUFFLED':('PRE_SHUFFLED','native')}
    if mode=='TRAIN_CONSTANT':return V.BASELINE/'PRE_CONSTANT/endpoints/0128/native'/(q+'.json')
    arm,intervention=mapping[mode]
    return V.OUT/arm/'endpoints/0128'/intervention/(q+'.json')


def inputs(m):
    payload={};bindings=[]
    for split,rows in [('train',m['train_rows']),('probe',m['probe_rows'])]:
        for r in rows:
            q=r['query_id'];p=V.V2/'encoder_cache'/q/'validation.json';cv=read(p)
            par=checked(cv['parity']);bindings+=[bind(p),cv['parity']]
            data={'L0':read(par)['fresh_L0']}
            for mode in ('REAL','REAL_CONSTANT','REAL_SHUFFLED','TRAIN_CONSTANT','TRAIN_SHUFFLED'):
                f=input_path(split,mode,q);d=read(f)
                need(d['query_id']==q and d['candidate_ids']==r['candidate_ids'],'CANDIDATE_SOURCE_PARITY')
                data[mode]=d['L'];bindings.append(bind(f))
            payload[q]=data
    return payload,bindings


def forbid_probe_labels():
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=str(Path(os.fsdecode(args[0])).resolve()).lower()
        need(not any(x in p for x in ('curator_roles','/target_join/','d1-mi','d1_mi','formal392')),'PROTECTED_LABEL_INPUT')
        need(p not in [str(PROBE/'result.json').lower(),str(V.V2/'result.json').lower()],'NO_OLD_PROBE_OUTCOME_READ')
    sys.addaudithook(audit)


def prepare():
    need(not AUTH.exists(),'AUTHORITY_ALREADY_FROZEN');forbid_probe_labels()
    pa=read(ROOT/'registry/rc_internal_m_v4_probe_v1_authority_20260924.json')
    m=read(checked(pa['manifest']))
    for lane in ('REAL','CONTROLS'):
        seal=read(PROBE/lane/'validation.json')
        for b in seal['predictions']:checked(b)
    for arm,base in [('PRE_REAL',V.OUT),('PRE_SHUFFLED',V.OUT),('PRE_CONSTANT',V.BASELINE)]:
        ep=read(base/arm/'endpoints/0128/validation.json')
        for b in ep['predictions']:checked(b)
    data,source=inputs(m)
    write(OUT/'input_data.json',dict(rows=data,held_labels_included=False))
    configs=dict(steps=2000,learning_rate=.03,weight_decay=.001,clip_norm=1.,dtype='float64',cpu_threads=2)
    sources=[Path(__file__),PLAN,ROOT/'tests/test_internal_m_content_preserving_readout_v1.py']
    write(AUTH,dict(status='POSTHOC_TRAIN_ONLY_CONTENT_PRESERVING_READOUT_FROZEN',manifest=pa['manifest'],
        parent_authority=bind(ROOT/'registry/rc_internal_m_v4_probe_v1_authority_20260924.json'),
        inputs=bind(OUT/'input_data.json'),input_sources=source,code_sources=[bind(p) for p in sources],
        warm_head=bind(V.OUT/'cpu/warmstart_INTERNAL3.json'),adapted_reference=bind(V.OUT/'readout_refits/GAIN_REAL.json'),
        external_reference=bind(V.OUT/'cpu/ADDITIVE4.json'),config=configs,arms=list(ARMS),
        feature_definition=['dRAW','sym_L0','sym_Ltheta_minus_sym_L0','bias'],direct_M_input=False,
        primary='PRESERVE_REAL versus ORIGINAL3_CONTINUE and frozen ADAPTED3; no probe selection',
        history='Motivated by opened PROBE8 failure audit; exploratory development, not untouched validation',
        retained_adapter_training_queries=[r['query_id'] for r in m['train_rows']],new_gpu_forwards=0))
    print('PREPARED',AUTH,flush=True)


def guard():
    a=read(AUTH)
    for b in a['code_sources']+a['input_sources']:checked(b)
    for k in ('parent_authority','warm_head','adapted_reference','external_reference'):checked(a[k])
    return a,read(checked(a['manifest'])),read(checked(a['inputs']))['rows']


def fit():
    import torch
    torch.set_num_threads(2);forbid_probe_labels();a,m,data=guard();rows=m['train_rows'];cfg=a['config']
    warm=read(checked(a['warm_head']))['theta'];all_predictions=[]
    for arm in ARMS:
        destination=OUT/'heads'/(arm+'.json')
        if destination.exists():
            old=read(destination);need(old['authority']==bind(AUTH) and old['steps']==2000,'FROZEN_COMPLETE_HEAD');continue
        original=arm=='ORIGINAL3_CONTINUE';mode=MODES[arm]
        X=np.stack([features(r,data[r['query_id']]['L0'],data[r['query_id']][mode],original) for r in rows])
        xt=torch.tensor(X,dtype=torch.float64)
        initial=warm if original else [warm[0],warm[1],0.,warm[2]]
        theta=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float64))
        opt=torch.optim.AdamW([theta],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);history=[];start=time.monotonic()
        for step in range(cfg['steps']):
            opt.zero_grad(set_to_none=True);loss=batch_loss(xt@theta,rows);loss.backward()
            need(bool(torch.isfinite(theta.grad).all()),'FINITE_GRADIENT')
            grad=float(torch.nn.utils.clip_grad_norm_([theta],cfg['clip_norm']));opt.step()
            if step==0 or (step+1)%100==0:
                with torch.no_grad():l=float(batch_loss(xt@theta,rows))
                history.append(dict(step=step+1,mean_cost1=l,gradient_norm=grad))
        params=theta.detach().tolist()
        write(destination,dict(authority=bind(AUTH),arm=arm,steps=cfg['steps'],initial_theta=initial,theta=params,
              fit_queries=[r['query_id'] for r in rows],history=history,seconds=time.monotonic()-start,
              held_labels_read=0,adapter_updates=0,direct_M_input=False))
        print('HEAD_FIT_COMPLETE',arm,params,history[-1]['mean_cost1'],flush=True)
    # All fitted heads fixed before any probe label file is opened.
    for split,rows in [('train',m['train_rows']),('probe',m['probe_rows'])]:
        for r in rows:
            q=r['query_id'];pred={}
            for arm in ARMS:
                head=read(OUT/'heads'/(arm+'.json'))
                x=features(r,data[q]['L0'],data[q][MODES[arm]],arm=='ORIGINAL3_CONTINUE')
                pred[arm]=decision(r,x,head['theta'])
                # Torch producer versus NumPy independent scoring.
                z=(torch.tensor(x,dtype=torch.float64)@torch.tensor(head['theta'],dtype=torch.float64)).tolist()
                need(float(np.max(np.abs(np.asarray(z)-pred[arm]['logits'])))<1e-10,'NUMPY_PARITY')
            rh=read(OUT/'heads/PRESERVE_REAL.json')['theta']
            for mode in ('REAL_CONSTANT','REAL_SHUFFLED'):
                pred['PRESERVE_REAL_INFER_'+mode]=decision(r,features(r,data[q]['L0'],data[q][mode]),rh)
            adapted=read(checked(a['adapted_reference']))['theta']
            pred['ADAPTED3_FROZEN_REFIT']=V.choose(r,data[q]['REAL'],adapted)
            external=read(checked(a['external_reference']))['theta']
            pred['EXTERNAL_ADDITIVE4']=V.choose(r,data[q]['L0'],external,'ADDITIVE4')
            f=OUT/'predictions'/split/(q+'.json')
            write(f,dict(authority=bind(AUTH),query_id=q,candidate_ids=r['candidate_ids'],predictions=pred,
                  held_labels_read=0,split=split));all_predictions.append(bind(f))
    write(OUT/'prelabel_validation.json',dict(status='ALL_HEADS_AND_PREDICTIONS_FROZEN_NUMPY_PASS',authority=bind(AUTH),
          heads=[bind(OUT/'heads'/(arm+'.json')) for arm in ARMS],predictions=all_predictions,held_label_reads=0))


def join():
    a,m,data=guard();seal=read(OUT/'prelabel_validation.json');need(seal['authority']==bind(AUTH),'PRELABEL_AUTHORITY')
    for b in seal['heads']+seal['predictions']:checked(b)
    # These labels were opened in previous development. They never enter fit().
    old=read(PROBE/'result.json');probe_labels={r['query_id']:r for r in old['rows']}
    rows=[]
    for split,inputs in [('TRAIN16',m['train_rows']),('PROBE8',m['probe_rows'])]:
        for r in inputs:
            q=r['query_id'];identity=r['target_id'] if split=='TRAIN16' else probe_labels[q]['identity']
            p=read(OUT/'predictions'/('train' if split=='TRAIN16' else 'probe')/(q+'.json'))
            selected={'RAW':r['candidate_identities'][r['winner_index']],**{k:v['prediction_identity'] for k,v in p['predictions'].items()}}
            rows.append(dict(query_id=q,split=split,identity=identity,component=r['component'] if split=='TRAIN16' else probe_labels[q]['component'],
                 target_present=identity in r['candidate_identities'],selected=selected,correct={k:v==identity for k,v in selected.items()}))
    summary={}
    for split in ('TRAIN16','PROBE8'):
        rs=[r for r in rows if r['split']==split];out={}
        for model in rs[0]['correct']:
            out[model]=dict(correct=sum(r['correct'][model] for r in rs),n=len(rs),
                  rescues=[r['query_id'] for r in rs if r['correct'][model] and not r['correct']['RAW']],
                  breaks=[r['query_id'] for r in rs if not r['correct'][model] and r['correct']['RAW']])
        summary[split]=out
    result=dict(status='POSTHOC_CONTENT_PRESERVING_READOUT_COMPLETE',authority=bind(AUTH),prelabel_seal=bind(OUT/'prelabel_validation.json'),
          label_source=bind(PROBE/'result.json'),rows=rows,summary=summary,probe_components=len({r['component'] for r in rows if r['split']=='PROBE8'}),
          new_encoder_roma_or_llm_forwards=0,adapter_updates=0,new_generalization_claim=False,protocol=a['history'])
    write(OUT/'result.json',result)
    lines=['# V4冻结表示：原始内容与内部变化分开读出','',a['history'],
           '只训练小头，固定TRAIN16/C128/2000步COST1；不直接读M，不改变适配器。PROBE8已打开，不能称独立确认。','',
           '| 路径 | TRAIN16 | PROBE8 | probe救回/误伤 |','|---|---:|---:|---:|']
    for k in summary['TRAIN16']:
        t=summary['TRAIN16'][k];p=summary['PROBE8'][k]
        lines.append(f"| {k} | {t['correct']}/16 | {p['correct']}/8 | {len(p['rescues'])}/{len(p['breaks'])} |")
    lines+=['','新增变化列只是保留原始和调制内容的独立自由度，不增加信息源；原内容列的存在不保证原排序或正确样本保留。',
            'ORIGINAL3_CONTINUE排除继续训练本身；两个TRAIN条件对照均为同四参数头；推理干预共用PRESERVE_REAL头。',
            '本轮源于旧probe失败分析；不按probe选择参数、阈值、checkpoint或宣布独立成功。','']
    REPORT.write_text('\n'.join(lines))
    write(OUT/'validation.json',dict(status='CONTENT_PRESERVING_READOUT_SOURCE_NUMPY_JOIN_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),report=bind(REPORT)))
    print('RESULT',summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit','join']);args=parser.parse_args()
    globals()[args.stage]()
