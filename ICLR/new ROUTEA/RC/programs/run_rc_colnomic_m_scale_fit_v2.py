#!/usr/bin/env python3
"""Fixed-budget TRAIN continuation with a relative AND absolute M objective."""
import argparse
import copy
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import run_rc_colnomic_m_train_fit_v1 as D

ROOT=D.ROOT
OUT=ROOT/'results/rc_colnomic_m_scale_fit_v2'
AUTH=ROOT/'registry/rc_colnomic_m_scale_fit_authority_v2_20260923.json'
PLAN=ROOT/'plan/RC_COLNOMIC_M_SCALE_FIT_V2_20260923.md'
LAUNCH=ROOT/'slurm/rc_colnomic_m_scale_fit_v2.sbatch'
ARMS=('FULL','BALANCED')
read,write,save,bind,checked=D.read,D.write,D.save,D.bind,D.checked


def objective(m,target,weight):
    e=torch.log(m+D.EPS)-torch.log(target+D.EPS)
    return weight*(e-e.mean()).square().mean()+e.mean().square()


def prepare():
    assert not AUTH.exists()
    previous=read(D.OUT/'validation.json');checked(previous['result'])
    panel=read(D.OUT/'panel.json')
    assert len(panel['records'])==4 and not panel['identity_labels_included']
    x=torch.tensor([.02,.01,.1],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([.01,.03,.09],dtype=torch.float64)
    assert torch.allclose(objective(x,y,1),D.objective(x,y,False),atol=1e-15,rtol=1e-15)
    torch.autograd.gradcheck(lambda z:objective(z,y,4),(x,))
    write(AUTH,dict(status='TRAIN_RELATIVE_AND_ABSOLUTE_M_FROZEN',
        sources=[bind(p) for p in (Path(__file__),Path(D.__file__),Path(D.C.__file__),Path(D.Q.__file__),PLAN,LAUNCH)],
        parent=bind(D.AUTH),panel=bind(D.OUT/'panel.json'),start_checkpoint=bind(D.OUT/'checkpoint.pt'),
        relations=[read(D.OUT/'relations'/f'query{i}.json')['payload'] for i in range(4)],
        initial_arm='CALIBRATED',start_step=256,final_step=2000,arms=list(ARMS),seed=17,
        lr=.001,weight_decay=.001,chunk_seconds=450,max_chunks=8,held_reads=0,
        candidate_count=128,train_queries=4,cross_group_start_only_after_fit_gate=True))
    write(OUT/'preflight.json',dict(status='LOSS_DECOMPOSITION_AND_GRADIENT_PASS',authority=bind(AUTH)))
    print('PREPARED',flush=True)


def gate(summary,rows):
    return dict(relative_error=summary['normalized_centered_mse']<=.25,
                order=summary['pair_order_agreement']>=.8,
                absolute_log_rmse=math.sqrt(summary['log_mse'])<=.3,
                every_query_log_bias=max(abs(r['metrics']['mean_log_error']) for r in rows)<=.1)


def run():
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    a=read(AUTH)
    for b in a['sources']+[a['parent'],a['panel'],a['start_checkpoint']]:checked(b)
    assert read(OUT/'preflight.json')['authority']==bind(AUTH)
    panel=read(a['panel']['path'])
    allow={Path(b['path']).resolve() for b in [a['panel'],a['start_checkpoint']]+a['relations']}
    def audit(event,args):
        if event=='socket.connect':assert not isinstance(args[1],tuple),'OFFLINE'
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if ROOT/'results' in p.parents:assert OUT in p.parents or p in allow,('UNAUTHORIZED_INPUT',str(p))
    sys.addaudithook(audit)
    lock=(OUT/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(8);start=time.monotonic()
    data=[torch.load(checked(b),weights_only=True,mmap=True) for b in a['relations']]
    for rec,d in zip(panel['records'],data):
        assert rec['query_id']==d['query_id'] and len(d['lengths'])==256 and d['authority']==a['parent']
    prior=torch.load(a['start_checkpoint']['path'],weights_only=True)
    assert prior['step']==256 and prior['authority']==a['parent']
    models,opts={},{}
    for arm in ARMS:
        m=D.C.Quality();m.load_state_dict(prior['models']['CALIBRATED']);models[arm]=m
        op=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
        op.load_state_dict(copy.deepcopy(prior['optimizers']['CALIBRATED']));opts[arm]=op
    # Continuation arms must not share optimizer moment storage.
    for p0,p1 in zip(models['FULL'].parameters(),models['BALANCED'].parameters()):
        assert opts['FULL'].state[p0]['exp_avg'].data_ptr()!=opts['BALANCED'].state[p1]['exp_avg'].data_ptr()
    step=256;history=[];cp=OUT/'checkpoint.pt'
    if cp.exists():
        p=torch.load(cp,weights_only=True);assert p['authority']==bind(AUTH)
        step,history=p['step'],p['history']
        for arm in ARMS:models[arm].load_state_dict(p['models'][arm]);opts[arm].load_state_dict(p['optimizers'][arm])
    def checkpoint():
        save(cp,dict(authority=bind(AUTH),step=step,history=history,
             models={k:m.state_dict() for k,m in models.items()},
             optimizers={k:op.state_dict() for k,op in opts.items()}),mutable=True)
    def receipt(phase):
        v=dict(status='NORMAL_CHUNK',phase=phase,step=step,seconds=time.monotonic()-start,authority=bind(AUTH))
        write(OUT/'chunks'/f"{os.environ['SLURM_JOB_ID']}.json",v);write(OUT/'status.json',v,mutable=True);print(v,flush=True)
    while True:
        if step in (500,1000,2000):
            for arm in ARMS:
                dest=OUT/'snapshots'/f'{arm}_{step}.json'
                if dest.exists():continue
                rows=[]
                for d in data:
                    with torch.no_grad():m,detail=D.packed_mass(models[arm],d,True)
                    rows.append(dict(query_id=d['query_id'],predicted_M=m.tolist(),metrics=D.metrics(m.numpy(),d['teacher'].numpy()),**detail))
                summary={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
                write(dest,dict(authority=bind(AUTH),arm=arm,step=step,rows=rows,mean_query_metrics=summary,gate=gate(summary,rows)))
                print(dict(event='SNAPSHOT',arm=arm,step=step,metrics=summary,gate=gate(summary,rows)),flush=True)
        if step==2000:break
        if time.monotonic()-start>a['chunk_seconds']:checkpoint();receipt('fit');return
        epoch,offset=divmod(step,4);g=torch.Generator().manual_seed(17+1000003*epoch)
        d=data[int(torch.randperm(4,generator=g)[offset])]
        for arm in ARMS:
            op=opts[arm];op.zero_grad();m,_=D.packed_mass(models[arm],d)
            loss=objective(m,d['teacher'],1 if arm=='FULL' else 4);loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in models[arm].parameters())
            op.step()
            if (step+1)%25==0:history.append(dict(step=step+1,arm=arm,loss=float(loss.detach()),query_id=d['query_id']))
        step+=1
        if step%25==0:checkpoint();print(dict(event='UPDATE',step=step,seconds=time.monotonic()-start),flush=True)
    checkpoint()
    final={arm:read(OUT/'snapshots'/f'{arm}_2000.json') for arm in ARMS}
    selected=next((arm for arm in ARMS if all(final[arm]['gate'].values())),None)
    # Independent NumPy recomputation of every final prediction's metrics.
    for v in final.values():
        for row,rec in zip(v['rows'],panel['records']):
            met=D.metrics(row['predicted_M'],rec['teacher_M'])
            assert max(abs(met[k]-row['metrics'][k]) for k in met)<1e-12
    write(OUT/'result.json',dict(status='SMALL_TRAIN_FIT_GATE_PASS' if selected else 'SMALL_TRAIN_FIT_GATE_NOT_MET',
        authority=bind(AUTH),selected_recipe=selected,arms=final,held_reads=0,
        boundary='Small TRAIN only; cross-group fitting must start fresh on original full TRAIN'))
    write(OUT/'validation.json',dict(status='TWO_ARMS_RELATIVE_ABSOLUTE_METRICS_PASS',result=bind(OUT/'result.json'),checkpoint=bind(cp),authority=bind(AUTH)))
    receipt('complete')


def advance():
    a=read(AUTH)
    for b in a['sources']:checked(b)
    jid=os.environ['SLURM_JOB_ID'];receipt=read(OUT/'chunks'/f'{jid}.json')
    assert receipt['status']=='NORMAL_CHUNK' and receipt['authority']==bind(AUTH)
    if (OUT/'validation.json').exists():print('SMALL_PANEL_COMPLETE',flush=True);return
    if len(list((OUT/'chunks').glob('*.json')))>=a['max_chunks']:
        write(OUT/'status.json',dict(status='STOPPED_CHUNK_BUDGET'),mutable=True);return
    dest=OUT/'continuations'/f'{jid}.json'
    if dest.exists():return
    args=['sbatch','--parsable','--dependency=afterok:'+jid,str(LAUNCH)]
    p=subprocess.run(args,text=True,capture_output=True,check=True,timeout=45)
    job=p.stdout.strip().split(';')[0];assert job.isdigit()
    write(dest,dict(job_id=job,previous=jid,authority=bind(AUTH)));print('CONTINUATION',job,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run','advance'));globals()[p.parse_args().stage]()
