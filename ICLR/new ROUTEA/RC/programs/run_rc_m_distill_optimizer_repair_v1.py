#!/usr/bin/env python3
"""Same frozen relation features, same MLP and loss; bounded full-batch fit repair."""
import argparse,fcntl,math,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import run_rc_m_small_scaling_v1 as B
D,S,K=B.D,B.S,B.K
R=B.R;OUT=R/'results/rc_m_distill_optimizer_repair_v1';AUTH=R/'registry/rc_m_distill_optimizer_repair_authority_v1_20260924.json'
LAUNCH=R/'slurm/rc_m_distill_optimizer_repair_v1.sbatch';PLAN=R/'plan/RC_M_DISTILL_OPTIMIZER_REPAIR_V1_20260924.md'
read,write,save,bind,checked=B.read,B.write,B.save,B.bind,B.checked

def prepare():
    assert not AUTH.exists()
    old=read(B.AUTH);panel=read(checked(old['panel']));rels=[];warm={}
    for i in range(72):
        receipt=B.OUT/'relations'/f'{i:03d}.json';v=read(receipt)
        assert v['query_id']==(panel['training']+panel['probe'])[i]['query_id']
        rels.append(dict(receipt=bind(receipt),**v))
    for n in [16,32,64]:
        oldcp=B.OUT/f'n{n}/checkpoint.pt'
        # Atomic source checkpoints are copied once; later baseline updates cannot affect this run.
        state=torch.load(oldcp,map_location='cpu',weights_only=True)
        assert state['authority']==bind(B.AUTH) and state['n']==n
        dest=OUT/f'initial/n{n}.pt';save(dest,dict(model=state['model'],baseline_authority=state['authority'],baseline_step=state['step'],n=n))
        warm[str(n)]=bind(dest)
    a=dict(parent=bind(B.AUTH),panel=old['panel'],relations=rels,warm_starts=warm,sizes=[16,32,64],
           sources=[bind(p) for p in [Path(__file__),Path(B.__file__),Path(D.__file__),Path(S.__file__),Path(D.C.__file__),Path(D.Q.__file__),Path(K.__file__),PLAN,LAUNCH]],
           optimizer=dict(name='LBFGS',lr=1.,max_iter=1,max_eval=8,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12),
           max_outer_iterations=40,max_chunks=8,chunk_seconds=400,snapshots=[0,5,10,20,40],
           loss='Same unregularized query-balanced full log-M MSE as prior objective(weight=1)',
           changed='Optimizer only: deterministic full-query-average gradients; no AdamW decay; additional bounded optimization budget',
           unchanged='8353-parameter MLP, all tokens, all C128 candidates, FP64, frozen ColNomic relations, TRAIN-only calibration and original fit gates',
           probe_used_for_selection=False,full_train_policy='Original 457 TRAIN/2000 updates is short-budget control, never sufficient-fit proof; require measured TRAIN gate before interpreting held generalization',new_gpu_forwards=0)
    write(AUTH,a);print({'prepared':True,'warm_steps':{n:torch.load(b['path'],weights_only=True)['baseline_step'] for n,b in warm.items()}},flush=True)

def load_data(a,n):
    panel=read(checked(a['panel']));data=[]
    for i in list(range(n))+list(range(64,72)):
        rec=a['relations'][i];v=read(checked(rec['receipt']));assert v=={k:v for k,v in rec.items() if k!='receipt'}
        d=torch.load(checked(v['payload']),map_location='cpu',weights_only=True,mmap=True)
        expected=(panel['training']+panel['probe'])[i]
        assert d['query_id']==expected['query_id'] and len(d['lengths'])==256 and sum(d['lengths'])==d['x'].shape[0]
        assert d['x'].dtype==torch.float64 and torch.equal(d['teacher'],torch.tensor(expected['teacher_M'],dtype=torch.float64))
        data.append(d)
    return data[:n],data[n:]

def run(a,n):
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    torch.set_num_threads(8);dest=OUT/f'n{n}';dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();job=os.environ['SLURM_JOB_ID'];cp=dest/'checkpoint.pt'
    def receipt(phase,iteration):
        write(dest/'chunks'/f'{job}.json',dict(authority=bind(AUTH),phase=phase,iteration=iteration,seconds=time.monotonic()-start))
    if (dest/'result.json').exists():receipt('complete',read(dest/'result.json')['iteration']);return
    data,probe=load_data(a,n);model=D.C.Quality(seed=17)
    initial=torch.load(checked(a['warm_starts'][str(n)]),weights_only=True);model.load_state_dict(initial['model'])
    opts={k:v for k,v in a['optimizer'].items() if k!='name'};opt=torch.optim.LBFGS(model.parameters(),**opts)
    iteration=0;evaluations=0;history=[]
    if cp.exists():
        state=torch.load(cp,weights_only=True);assert state['authority']==bind(AUTH)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);iteration=state['iteration'];evaluations=state['evaluations'];history=state['history']
    def checkpoint():save(cp,dict(authority=bind(AUTH),model=model.state_dict(),optimizer=opt.state_dict(),iteration=iteration,evaluations=evaluations,history=history),mutable=True)
    while True:
        if iteration in a['snapshots']:
            path=dest/'snapshots'/f'iter{iteration:03d}.json'
            if path.exists():snapshot=read(path)
            else:
                fit=B.evaluate(model,data);test=B.evaluate(model,probe,fit['bias'])
                frozen=dest/'models'/f'iter{iteration:03d}.pt';save(frozen,dict(authority=bind(AUTH),model=model.state_dict(),iteration=iteration))
                snapshot=dict(authority=bind(AUTH),iteration=iteration,evaluations=evaluations,model=bind(frozen),fit=fit,probe=test,probe_used_for_selection=False)
                write(path,snapshot);print({'snapshot':iteration,'n':n,'fit':fit['mean_query_metrics'],'gate':fit['gate'],'probe':test['mean_query_metrics']},flush=True)
            if all(snapshot['fit']['gate'].values()) or iteration==a['max_outer_iterations']:
                status='FIT_GATE_PASS' if all(snapshot['fit']['gate'].values()) else 'FIT_GATE_NOT_MET_WITH_BOUNDED_OPTIMIZER_REPAIR'
                write(dest/'result.json',dict(status=status,authority=bind(AUTH),n=n,iteration=iteration,snapshot=bind(path),selection='TRAIN_ONLY',boundary='Does not establish information absence or held accuracy gain'))
                checkpoint();receipt('complete',iteration);return
        if time.monotonic()-start>a['chunk_seconds']:
            checkpoint();receipt('fit',iteration);return
        closure_values=[]
        def closure():
            nonlocal evaluations
            opt.zero_grad();loss_total=0.
            for d in data:
                m,_=D.packed_mass(model,d);loss=S.objective(m,d['teacher'],1)/n
                assert torch.isfinite(loss);loss.backward();loss_total+=float(loss.detach())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            evaluations+=1;closure_values.append(loss_total)
            print({'n':n,'iteration':iteration,'evaluation':evaluations,'train_log_mse':loss_total,'seconds':time.monotonic()-start},flush=True)
            return torch.tensor(loss_total,dtype=torch.float64)
        opt.step(closure);iteration+=1
        assert all(torch.isfinite(p).all() for p in model.parameters())
        history.append(dict(iteration=iteration,closure_losses=closure_values));checkpoint()

def advance(n):
    a=read(AUTH);dest=OUT/f'n{n}';job=os.environ['SLURM_JOB_ID'];receipt=dest/'dispatch'/f'{job}.json'
    if receipt.exists():return
    jobs=[]
    if not (dest/'result.json').exists():
        if len(list((dest/'chunks').glob('*.json')))<a['max_chunks']:
            cmd=['sbatch','--parsable','--dependency=afterok:'+job,'--mem='+{16:'32G',32:'56G',64:'96G'}[n],str(LAUNCH),'run',str(n)]
            p=subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=45);jobs.append(p.stdout.strip().split(';')[0])
        else:write(dest/'budget_stop.json',dict(status='BOUNDED_OPTIMIZER_BUDGET_EXHAUSTED',authority=bind(AUTH)))
    write(receipt,dict(previous=job,jobs=jobs));print({'continuations':jobs},flush=True)
    summary={}
    for size in a['sizes']:
        root=OUT/f'n{size}';ss=sorted((root/'snapshots').glob('*.json'))
        summary[str(size)]=dict(result=read(root/'result.json') if (root/'result.json').exists() else None,latest_snapshot=bind(ss[-1]) if ss else None)
    # Per-arm summary avoids competing workers replacing a shared snapshot.
    write(dest/'summary.json',dict(authority=bind(AUTH),arms=summary),mutable=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','advance']);p.add_argument('n',nargs='?',type=int);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        a=read(AUTH)
        for b in a['sources']:checked(b)
        if args.stage=='run':run(a,args.n)
        else:advance(args.n)
