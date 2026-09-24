#!/usr/bin/env python3
"""TRAIN-group validation of bounded M distillation; untouched by outer labels."""
import argparse,copy,fcntl,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import run_rc_m_distill_full_repair_v1 as F
R=F.R;D,S,K,B,T=F.D,F.S,F.K,F.B,F.T
OUT=R/'results/rc_m_distill_generalization_v1';AUTH=R/'registry/rc_m_distill_generalization_authority_v1_20260924.json'
LAUNCH=R/'slurm/rc_m_distill_generalization_v1.sbatch';PLAN=R/'plan/RC_M_DISTILL_GENERALIZATION_V1_20260924.md'
read,write,save,bind,checked=F.read,F.write,F.save,F.bind,F.checked

def prepare():
    assert not AUTH.exists();a=copy.deepcopy(read(F.AUTH));panel=read(checked(a['panel']))
    roles={x['query_id']:x['component'] for x in read(checked(a['train_roles']))['records']};train=read(checked(a['full_train_inputs']))['records']
    roles={x['query_id']:roles[x['query_id']] for x in train}
    vg=set(panel['probe_components']);fit=[x['query_id'] for x in train if roles[x['query_id']] not in vg];valid=[x['query_id'] for x in train if roles[x['query_id']] in vg]
    assert set(fit).isdisjoint(valid) and len(fit)+len(valid)==457 and all(x['query_id'] in fit for x in panel['training'])
    assert set(roles)==set(fit+valid)
    a.update(parent_full_repair=bind(F.AUTH),inner_fit_ids=fit,inner_valid_ids=valid,train_components=roles,inner_valid_groups=sorted(vg),
             regularization=[0.,.001],inner_iterations=[0,10,20,40,80],inner_max_iterations=80,
             inner_metric='Equal-component mean log-M MSE after bias calibrated ONLY on inner fit',
             selection='Minimum TRAIN-inner-group validation error; ties fewer iterations then larger ridge. Outer held labels and teacher forbidden.',
             initialization='Qualified n64 model excludes all inner validation groups; full457 old model MUST NOT initialize inner arms',
             generalization_scope='Opened development TRAIN-group selection plus original outer held119; not external confirmation',
             sources=[bind(x) for x in [Path(__file__),Path(F.__file__),Path(F.OLD.__file__),Path(B.__file__),Path(D.__file__),Path(S.__file__),Path(D.C.__file__),Path(D.Q.__file__),Path(K.__file__),Path(T.__file__),R/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',PLAN,LAUNCH]])
    write(AUTH,a);old=torch.load(checked(a['initial64']),weights_only=True);old['previous_authority']=old['authority'];old['authority']=bind(AUTH);save(OUT/'n64/checkpoint.pt',old)
    print({'prepared':True,'inner_fit':len(fit),'inner_validation':len(valid),'validation_groups':len(vg)},flush=True)

class Subset:
    def __init__(self,source,ids):self.source=source;self.ids=set(ids)
    def __len__(self):return len(self.ids)
    def __iter__(self):
        # Filter before opening relation caches, not after reconstructing unwanted queries.
        old=self.source.records;self.source.records=[x for x in old if x['query_id'] in self.ids]
        try:yield from self.source
        finally:self.source.records=old

def configure():F.OUT=OUT;F.AUTH=AUTH;F.LAUNCH=LAUNCH;T.OUT=OUT;T.AUTH=AUTH

def fit(a,stage,arm):
    configure();start=time.monotonic();base=read(checked(read(OUT/'n64/result.json')['snapshot']));assert all(base['fit']['gate'].values())
    d=OUT/('inner'+str(arm) if stage=='inner' else 'generalized/refit');d.mkdir(parents=True,exist_ok=True);lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (d/'result.json').exists():return
    source=F.RelationStream(a)
    if stage=='inner':ids=a['inner_fit_ids'];valid=a['inner_valid_ids'];ridge=a['regularization'][arm];limit=a['inner_max_iterations']
    else:
        sel=read(OUT/'inner_selection.json');ids=[x['query_id'] for x in source.records];valid=[];ridge=sel['ridge'];limit=sel['iteration']
    train=Subset(source,ids);validation=Subset(source,valid)
    model=D.C.Quality();state=torch.load(checked(base['model']),weights_only=True);model.load_state_dict(state['model'])
    opt=torch.optim.LBFGS(model.parameters(),**{k:v for k,v in a['optimizer'].items() if k!='name'});iteration=0;history=[];cp=d/'checkpoint.pt'
    if cp.exists():
        st=torch.load(cp,weights_only=True);assert st['authority']==bind(AUTH);model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);iteration=st['iteration'];history=st['history']
    def checkpoint():save(cp,dict(authority=bind(AUTH),model=model.state_dict(),optimizer=opt.state_dict(),iteration=iteration,history=history,ridge=ridge),mutable=True)
    while True:
        if (stage=='inner' and iteration in a['inner_iterations']) or (stage=='refit' and iteration==limit):
            sp=d/'snapshots'/f'iter{iteration:03d}.json'
            if not sp.exists():
                ft=F.evaluate(model,train);val=F.evaluate(model,validation,ft['bias']) if valid else None
                scores={}
                if val:
                    for row in val['rows']:scores.setdefault(a['train_components'][row['query_id']],[]).append(row['metrics']['log_mse'])
                metric=float(np.mean([np.mean(v) for v in scores.values()])) if scores else None
                mp=d/'models'/f'iter{iteration:03d}.pt';save(mp,dict(authority=bind(AUTH),model=model.state_dict(),iteration=iteration))
                write(sp,dict(authority=bind(AUTH),model=bind(mp),iteration=iteration,ridge=ridge,fit=ft,validation=val,group_validation_error=metric,outer_held_reads=0))
                print({'stage':stage,'arm':arm,'iteration':iteration,'fit_gate':ft['gate'],'validation_error':metric},flush=True)
            if iteration==limit:
                checkpoint();write(d/'result.json',dict(status='INNER_COMPLETE' if stage=='inner' else 'TRAIN_REFIT_COMPLETE',authority=bind(AUTH),snapshot=bind(sp),iteration=iteration,ridge=ridge));return
        if time.monotonic()-start>a['chunk_seconds']:checkpoint();return
        losses=[]
        def closure():
            opt.zero_grad();total=0.
            for datum in train:
                mass,_=D.packed_mass(model,datum);loss=S.objective(mass,datum['teacher'],1)/len(train);loss.backward();total+=float(loss.detach())
            weights=[p for name,p in model.named_parameters() if name.endswith('weight')]
            penalty=ridge*sum(p.square().sum() for p in weights)/sum(p.numel() for p in weights)
            penalty.backward();total+=float(penalty.detach())
            assert np.isfinite(total) and all(torch.isfinite(p.grad).all() for p in model.parameters());losses.append(total)
            return torch.tensor(total,dtype=torch.float64)
        opt.step(closure);iteration+=1;history.append(dict(iteration=iteration,losses=losses));checkpoint()
        print({'stage':stage,'arm':arm,'iteration':iteration,'losses':losses,'seconds':time.monotonic()-start},flush=True)

def select(a):
    choices=[]
    for arm in range(2):
        d=OUT/f'inner{arm}';assert read(d/'result.json')['status']=='INNER_COMPLETE'
        for p in sorted((d/'snapshots').glob('*.json')):
            x=read(p);assert x['authority']==bind(AUTH) and x['outer_held_reads']==0
            choices.append(dict(arm=arm,ridge=x['ridge'],iteration=x['iteration'],validation_error=x['group_validation_error'],snapshot=bind(p)))
    chosen=min(choices,key=lambda x:(x['validation_error'],x['iteration'],-x['ridge']))
    write(OUT/'inner_selection.json',dict(authority=bind(AUTH),**chosen,choices=choices,outer_held_reads=0))

def predict_general(a):
    configure();start=time.monotonic();gout=OUT/'generalized';result=read(gout/'refit/result.json');snap=read(checked(result['snapshot']));model=D.C.Quality();model.load_state_dict(torch.load(checked(snap['model']),weights_only=True)['model'])
    final=gout/'final_model.pt'
    if not final.exists():save(final,dict(authority=bind(AUTH),model=model.state_dict()))
    fitrows={x['query_id']:x for x in snap['fit']['rows']};train=read(checked(a['full_train_inputs']))['records'];held=read(checked(a['held_inputs']));assert not held['teacher_included'] and not held['identity_labels_included']
    bank=D.Q.Bank(read(checked(a['full_ready']))['features'],'COL_ONLY','cpu');bindings=[]
    for split,records in [('train',train),('held',held['records'])]:
        for rec in records:
            path=gout/'predictions'/split/f"query{rec['execution_ordinal']:03d}.pt"
            if not path.exists():
                if time.monotonic()-start>a['chunk_seconds']:return
                if split=='train':raw=torch.tensor(fitrows[rec['query_id']]['M_uncalibrated'],dtype=torch.float64);traces=[]
                else:
                    with torch.no_grad():raw,traces=T.estimate(rec,bank,model,True)
                mass=torch.from_numpy(K.apply(raw.numpy(),snap['fit']['bias']))
                save(path,dict(authority=bind(AUTH),model=bind(final),query_id=rec['query_id'],M=mass,M_uncalibrated=raw,quality_traces=traces,candidate_physical_rows=rec['candidate_physical_rows'],teacher_at_inference=False,held_identity_reads=0))
            p=torch.load(path,weights_only=True);assert p['authority']==bind(AUTH) and p['model']==bind(final) and p['query_id']==rec['query_id'];bindings.append(dict(split=split,query_id=rec['query_id'],prediction=bind(path)))
    write(gout/'prediction_seal.json',dict(authority=bind(AUTH),predictions=bindings,train=457,held=119,held_teacher_reads=0,held_identity_reads=0,train_fit_gate=snap['fit']['gate']))
    write(gout/'fit_validation.json',dict(status='GENERALIZATION_SELECTION_PREDICTIONS_SEALED',payload=bind(gout/'prediction_seal.json'),train_fit_gate=snap['fit']['gate'],boundary='Performance diagnostic; a failed TRAIN gate forbids information-absence inference'))

def submit(stage,n=0,dep=None,array=None):
    cmd=['sbatch','--parsable','--mem='+('96G' if stage=='run' and n==64 else '32G')]
    if dep:cmd+=['--dependency=afterok:'+dep]
    if array:cmd+=['--array='+array]
    cmd+=[str(LAUNCH),stage,str(n)];p=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=45);job=p.stdout.strip().split(';')[0];assert job.isdigit();return job

def advance(a,stage,n):
    job=os.environ['SLURM_JOB_ID'];dest=OUT/'dispatch'/f'{job}_{stage}_{n}.json'
    if dest.exists():return
    jobs={}
    if stage=='run':
        d=OUT/f'n{n}';rp=d/'result.json'
        if not rp.exists():
            if len(list((d/'chunks').glob('*.json')))<a['max_chunks']:jobs['continue']=submit('run',n,job)
            else:write(d/'budget_stop.json',dict(status='STOP_BUDGET',authority=bind(AUTH)))
        elif read(rp)['status']=='FIT_GATE_PASS':
            if n==64:
                jobs['cache']=submit('cache',dep=job,array='0-7%8');jobs['cache_join']=submit('cache-join',dep=jobs['cache'])
            else:jobs['predict']=submit('predict',dep=job)
    elif stage=='cache-join':
        jobs['full_fit_control']=submit('run',457,job)
        for arm in range(2):jobs['inner'+str(arm)]=submit('inner',arm,job)
    elif stage=='inner':
        if not (OUT/f'inner{n}/result.json').exists():jobs['continue']=bounded_submit(a,stage,n,job)
        elif all((OUT/f'inner{i}/result.json').exists() for i in range(2)):
            lock=(OUT/'selection_dispatch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
            rp=OUT/'selection_dispatch.json'
            if not rp.exists():jobs['select']=submit('select',dep=job);write(rp,dict(jobs=jobs))
    elif stage=='select':jobs['refit']=submit('refit',dep=job)
    elif stage=='refit':
        if (OUT/'generalized/refit/result.json').exists():jobs['predict-general']=submit('predict-general',dep=job)
        else:jobs['continue']=bounded_submit(a,stage,n,job)
    elif stage in ['predict','predict-general']:
        base=OUT if stage=='predict' else OUT/'generalized'
        if (base/'fit_validation.json').exists():jobs['join']=submit('join' if stage=='predict' else 'join-general',dep=job)
        else:jobs['continue']=bounded_submit(a,stage,n,job)
    write(dest,dict(previous=job,stage=stage,n=n,jobs=jobs));print({'submitted':jobs},flush=True)

def bounded_submit(a,stage,n,job):
    count=len(list((OUT/'stage_chunks'/f'{stage}_{n}').glob('*.json')))
    if count>=a['max_chunks']:
        write(OUT/f'{stage}_{n}_budget_stop.json',dict(status='STOP_BUDGET',authority=bind(AUTH)));return None
    return submit(stage,n,job)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('n',nargs='?',type=int,default=0);x=p.parse_args();torch.set_num_threads(8)
    if x.stage=='prepare':prepare()
    else:
        a=read(AUTH)
        for b in a['sources']:checked(b)
        assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available();configure()
        if x.stage.startswith('advance-'):advance(a,x.stage[len('advance-'):],x.n)
        elif x.stage=='run':F.run(a,x.n)
        elif x.stage=='cache':
            if not F.cache(a,int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))):raise SystemExit(75)
        elif x.stage=='cache-join':F.cache_join(a)
        elif x.stage in ['inner','refit']:fit(a,x.stage,x.n)
        elif x.stage=='select':select(a)
        elif x.stage=='predict':F.predict(a)
        elif x.stage=='predict-general':predict_general(a)
        elif x.stage=='join':
            assert read(OUT/'n457/result.json')['status']=='FIT_GATE_PASS';T.join(a)
        elif x.stage=='join-general':T.OUT=OUT/'generalized';T.join(a)
        else:raise ValueError(x.stage)
        if not x.stage.startswith('advance-'):write(OUT/'stage_chunks'/f'{x.stage}_{x.n}'/f"{os.environ['SLURM_JOB_ID']}.json",dict(authority=bind(AUTH),stage=x.stage,n=x.n))
