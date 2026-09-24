#!/usr/bin/env python3
"""Nested TRAIN-only M fitting; cached relations; fixed group-disjoint diagnostic probe."""
import argparse, collections, fcntl, math, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
import run_rc_colnomic_m_train_fit_v1 as D
import run_rc_colnomic_m_scale_fit_v2 as S
import rc_m_logit_calibration_v3 as K
R=D.ROOT; OUT=R/'results/rc_m_small_scaling_v1'; AUTH=R/'registry/rc_m_small_scaling_authority_v1_20260924.json'
LAUNCH=R/'slurm/rc_m_small_scaling_v1.sbatch'; PLAN=R/'plan/RC_M_SMALL_SCALING_V1_20260924.md'
read,write,save,bind,checked=D.read,D.write,D.save,D.bind,D.checked
SIZES=(16,32,64); EPOCHS=(4,16,64,128,256,500)
def prepare():
    assert not AUTH.exists()
    old=read(D.PARENT);tr=read(checked(old['train_inputs']));roles={x['query_id']:x for x in read(checked(old['train_roles']))['records']}
    assert not tr['identity_labels_included'] and len(tr['records'])==457
    groups=collections.OrderedDict()
    for q in sorted(tr['records'],key=lambda x:x['execution_ordinal']):groups.setdefault(roles[q['query_id']]['component'],[]).append(q)
    probe_groups=list(groups)[-8:];probe=[groups[g][0] for g in probe_groups];training=[]
    for i in range(max(map(len,groups.values()))):
        for g,qs in groups.items():
            if g not in probe_groups and i<len(qs):training.append(qs[i])
    training=training[:64];assert len(training)==64
    prior=read(D.OUT/'panel.json');assert [x['query_id'] for x in training[:4]]==[x['query_id'] for x in prior['records']]
    assert not {roles[q['query_id']]['component'] for q in training}&set(probe_groups)
    ready=read(checked(old['ready']));keys={k for q in training+probe for k in [q['query_image_key']]+q['reference_keys']}
    panel=dict(training=training,probe=probe,features={k:ready['features'][k] for k in sorted(keys)},fit_components=[roles[q['query_id']]['component'] for q in training],probe_components=probe_groups,held_reads=0)
    write(OUT/'panel.json',panel)
    a=dict(sources=[bind(x) for x in [Path(__file__),LAUNCH,PLAN,Path(D.__file__),Path(S.__file__),Path(D.C.__file__),Path(D.Q.__file__),Path(K.__file__)]],panel=bind(OUT/'panel.json'),parent=bind(D.PARENT),train=old['train_inputs'],roles=old['train_roles'],old_four_result=bind(R/'results/rc_colnomic_m_scale_calibration_v3/result.json'),old_four_relations=[read(D.OUT/'relations'/f'query{i}.json')['payload'] for i in range(4)],sizes=list(SIZES),epochs=list(EPOCHS),chunk_seconds=450,max_chunks=4,held_reads=0,new_gpu_forwards=0)
    write(AUTH,a);print({'prepared':True,'fit':64,'probe':8,'fit_groups':len(set(panel['fit_components']))},flush=True)
def submit(stage,n,dep):
    args=['sbatch','--parsable','--dependency=afterok:'+dep,'--mem='+('24G' if stage=='cache' else {16:'32G',32:'56G',64:'96G'}[n]),str(LAUNCH),stage,str(n)]
    p=subprocess.run(args,capture_output=True,text=True,check=True,timeout=45);jid=p.stdout.strip().split(';')[0];assert jid.isdigit();return jid
def guard():
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    a=read(AUTH)
    for b in a['sources']+[a['panel']]:checked(b)
    return a,read(a['panel']['path'])
def guard_data(panel):
    allowed={Path(b['path']).resolve() for b in panel['features'].values()}
    allowed.update(Path(b['path']).resolve() for b in read(AUTH)['old_four_relations'])
    def audit(event,args):
        if event=='socket.connect':assert not isinstance(args[1],tuple),'OFFLINE'
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if R/'results' in p.parents:assert OUT in p.parents or p in allowed,('UNAUTHORIZED_INPUT',str(p))
    import sys;sys.addaudithook(audit)
def cache(a,p):
    bank=D.Q.Bank(p['features'],'COL_ONLY','cpu');start=time.monotonic();records=p['training']+p['probe']
    for i,rec in enumerate(records):
        meta=OUT/'relations'/f'{i:03d}.json'
        if meta.exists():continue
        if time.monotonic()-start>a['chunk_seconds']:break
        if i<4:
            b=a['old_four_relations'][i];checked(b)
        else:
            q=bank(rec['query_image_key']);rels=[]
            with torch.no_grad():
                for key in rec['reference_keys']:
                    r=bank(key);rels.extend(D.C.relation(q['z'],r['z'],q['xy'],r['xy'],'PAIR'))
            dest=meta.with_suffix('.pt');save(dest,dict(authority=bind(AUTH),query_id=rec['query_id'],x=torch.cat(rels),lengths=[len(x) for x in rels],teacher=torch.tensor(rec['teacher_M'],dtype=torch.float64)))
            del rels;b=bind(dest)
        write(meta,dict(payload=b,query_id=rec['query_id'],index=i,reused=i<4));print({'cached':i,'seconds':time.monotonic()-start},flush=True)
    complete=all((OUT/'relations'/f'{i:03d}.json').exists() for i in range(72))
    if complete:write(OUT/'cache_validation.json',dict(status='ALL72_RELATIONS_READY',authority=bind(AUTH),relations=[bind(OUT/'relations'/f'{i:03d}.json') for i in range(72)]))
    return dict(complete=complete,phase='cache',seconds=time.monotonic()-start)
def loadrel(i):
    v=read(OUT/'relations'/f'{i:03d}.json');b=v['payload'];checked(b);d=torch.load(b['path'],weights_only=True,mmap=True)
    assert d['query_id']==v['query_id'] and len(d['lengths'])==256
    return d
def evaluate(model,data,bias=None):
    rows=[];pred=[]
    for d in data:
        with torch.no_grad():m,detail=D.packed_mass(model,d,True)
        raw=m.numpy();pred.append(raw)
        rows.append(dict(query_id=d['query_id'],M_uncalibrated=raw.tolist(),activation=detail,uncalibrated_metrics=D.metrics(raw,d['teacher'].numpy())))
    targets=np.array([d['teacher'].numpy() for d in data]);bias=K.fit(np.array(pred),targets) if bias is None else bias
    for row,t in zip(rows,targets):
        m=K.apply(row['M_uncalibrated'],bias);row.update(M=m.tolist(),metrics=D.metrics(m,t))
    mean={k:float(np.mean([x['metrics'][k] for x in rows])) for k in rows[0]['metrics']}
    return dict(rows=rows,mean_query_metrics=mean,bias=bias,gate=S.gate(mean,rows))
def train(a,p,n):
    assert n in SIZES
    start=time.monotonic();dest=OUT/f'n{n}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():return dict(complete=True,phase='fit',seconds=0)
    # Load all selected relations once; no repeated relation forwards during updates.
    data=[loadrel(i) for i in range(n)];probe=[loadrel(i) for i in range(64,72)]
    model=D.C.Quality(seed=17);init=math.exp(float(np.log(np.array([q['teacher_M'] for q in p['training'][:n]])+D.EPS).mean()))-D.EPS
    with torch.no_grad():model.up.bias.fill_(math.log(init/(1-init)))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);step=0;history=[];cp=dest/'checkpoint.pt'
    if cp.exists():
        state=torch.load(cp,weights_only=True);assert state['authority']==bind(AUTH);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);step=state['step'];history=state['history']
    def checkpoint():save(cp,dict(authority=bind(AUTH),n=n,step=step,model=model.state_dict(),optimizer=opt.state_dict(),history=history),mutable=True)
    passed=False
    while True:
        if step and step%n==0 and step//n in EPOCHS:
            snap=dest/'snapshots'/f'epoch{step//n:03d}.json'
            if snap.exists():v=read(snap)
            else:
                checkpoint(); frozen=dest/'snapshot_models'/f'epoch{step//n:03d}.pt'
                save(frozen,dict(authority=bind(AUTH),model=model.state_dict(),step=step),mutable=True)
                fit=evaluate(model,data);test=evaluate(model,probe,fit['bias'])
                v=dict(authority=bind(AUTH),n=n,step=step,epochs=step/n,checkpoint=bind(frozen),fit=fit,probe=test,probe_used_for_stopping=False)
                write(snap,v);print({'snapshot':step//n,'n':n,'fit':fit['mean_query_metrics'],'probe':test['mean_query_metrics']},flush=True)
            passed=all(v['fit']['gate'].values())
            if passed or step==n*500:
                write(dest/'result.json',dict(status='FIT_GATE_PASS' if passed else 'FIT_GATE_NOT_MET_AT_500_EPOCHS',authority=bind(AUTH),snapshot=bind(snap),stop_uses='TRAIN_ONLY',n=n,epochs=step/n));break
        if time.monotonic()-start>a['chunk_seconds']:checkpoint();return dict(complete=False,phase='fit',step=step,n=n,seconds=time.monotonic()-start)
        epoch,offset=divmod(step,n);g=torch.Generator().manual_seed(17+1000003*epoch);d=data[int(torch.randperm(n,generator=g)[offset])]
        opt.zero_grad();m,_=D.packed_mass(model,d);loss=S.objective(m,d['teacher'],1);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in model.parameters())
        if step%max(n,25)==0:history.append(dict(step=step,query_id=d['query_id'],loss=float(loss.detach()),grad_norms={k:float(x.grad.norm()) for k,x in model.named_parameters()}))
        opt.step();step+=1
        if step%100==0:checkpoint();print({'n':n,'step':step,'epoch':step/n,'seconds':time.monotonic()-start},flush=True)
    checkpoint();return dict(complete=True,phase='fit',n=n,step=step,seconds=time.monotonic()-start)
def summary(a):
    v=dict(authority=bind(AUTH),four_historical=a['old_four_result'],arms={},boundary='TRAIN fitting plus eight group-disjoint TRAIN probe queries; not held119 or accuracy GO')
    for n in SIZES:
        d=OUT/f'n{n}';fs=sorted((d/'snapshots').glob('*.json'));v['arms'][str(n)]=dict(result=read(d/'result.json') if (d/'result.json').exists() else None,latest_snapshot=bind(fs[-1]) if fs else None,status=read(d/'status.json') if (d/'status.json').exists() else None)
    write(OUT/'summary.json',v,mutable=True)
def advance(stage,n):
    a=read(AUTH);dest=OUT/('cache' if stage=='cache' else f'n{n}');job=os.environ['SLURM_JOB_ID'];rec=read(dest/'chunks'/f'{job}.json');receipt=dest/'continuations'/f'{job}.json'
    if receipt.exists():return
    jobs=[];count=len(list((dest/'chunks').glob('*.json')))
    if rec['complete'] and stage=='cache':
        for size in SIZES:jobs.append(dict(n=size,job=submit('fit',size,job)))
    elif not rec['complete'] and count<a['max_chunks']:jobs.append(dict(n=n,job=submit(stage,n,job)))
    elif not rec['complete']:write(dest/'budget_stop.json',dict(status='BUDGET_EXHAUSTED_NOT_INFORMATION_ABSENCE',chunks=count,receipt=rec))
    write(receipt,dict(previous=job,jobs=jobs));summary(a);print({'continuations':jobs},flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','cache','fit','advance-cache','advance-fit']);ap.add_argument('n',type=int,nargs='?',default=0);z=ap.parse_args();torch.set_num_threads(8)
    if z.stage=='prepare':prepare()
    elif z.stage.startswith('advance'):advance(z.stage.split('-')[1],z.n)
    else:
        a,p=guard();dest=OUT/('cache' if z.stage=='cache' else f'n{z.n}');dest.mkdir(parents=True,exist_ok=True);lock=(dest/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard_data(p)
        receipt=cache(a,p) if z.stage=='cache' else train(a,p,z.n)
        receipt['authority']=bind(AUTH);write(dest/'chunks'/f"{os.environ['SLURM_JOB_ID']}.json",receipt);write(dest/'status.json',receipt,mutable=True)
