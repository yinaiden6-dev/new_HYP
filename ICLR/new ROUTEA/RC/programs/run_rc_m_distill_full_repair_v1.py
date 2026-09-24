#!/usr/bin/env python3
"""Same frozen relation features, same MLP and loss; bounded full-batch fit repair."""
import argparse,fcntl,math,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import run_rc_m_small_scaling_v1 as B
import run_rc_m_distill_optimizer_repair_v1 as OLD
import run_rc_fold0_mass_teacher_v1 as T
D,S,K=B.D,B.S,B.K
R=B.R;OUT=R/'results/rc_m_distill_full_repair_v1';AUTH=R/'registry/rc_m_distill_full_repair_authority_v1_20260924.json'
LAUNCH=R/'slurm/rc_m_distill_full_repair_v1.sbatch';PLAN=R/'plan/RC_M_DISTILL_FULL_REPAIR_V1_20260924.md'
read,write,save,bind,checked=B.read,B.write,B.save,B.bind,B.checked

def prepare():
    assert not AUTH.exists()
    small=read(OLD.AUTH);full_path=R/'registry/rc_fold0_m_scale_teacher_authority_v2_20260924.json';full=read(full_path)
    assert read(OLD.OUT/'n64/result.json')['status']=='FIT_GATE_NOT_MET_WITH_BOUNDED_OPTIMIZER_REPAIR'
    initial64=OLD.OUT/'n64/checkpoint.pt';state=torch.load(initial64,weights_only=True,map_location='cpu')
    assert state['authority']==bind(OLD.AUTH) and state['iteration']==40
    initial457=Path(full['train_inputs']['path']).parents[1]/'rc_fold0_m_scale_teacher_v2/final_model.pt'
    assert initial457.exists();model=torch.load(initial457,weights_only=True,map_location='cpu');assert model['authority']==bind(full_path) and model['step']==2000
    a={**full,**small};a.update(parent=bind(OLD.AUTH),full_parent=bind(full_path),initial64=bind(initial64),
        full_train_inputs=full['train_inputs'],full_ready=full['ready'],held_inputs=full['held_inputs'],
        sources=[bind(x) for x in [Path(__file__),Path(OLD.__file__),Path(B.__file__),Path(D.__file__),Path(S.__file__),Path(D.C.__file__),Path(D.Q.__file__),Path(K.__file__),Path(T.__file__),R/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',PLAN,LAUNCH]],
        sizes=[64,457],max_outer_iterations=160,snapshots=[0,40,60,80,120,160],max_chunks=16,chunk_seconds=360,full_max_iterations=80,
        cache_shards=8,training_acceptance='Measured original four TRAIN gates; stop or explicitly unqualified at fixed budget',
        full_train_policy='Release cached full457 optimization only after n64 FIT_GATE_PASS; release held119 only after full457 FIT_GATE_PASS',
        initial_full_updates=2000,initial_full_epochs=2000/457,new_gpu_forwards=0,
        full_cache_representation='FP64 soft contexts 128 plus last6 relation scalars; reconstruct z/difference/product/xy from pinned frozen feature bank with exact torch.equal checks')
    a['warm_starts']=dict(small['warm_starts']);a['warm_starts']['457']=bind(initial457)
    train=read(checked(a['full_train_inputs']))['records'];assert len(train)==457
    for rec in train:assert len(rec['teacher_M'])==128
    assert len(set(x['query_id'] for x in train))==457
    byid={x['query_id']:x for x in train};panel=read(checked(a['panel']))
    for rec in panel['training']+panel['probe']:
        for key in ['candidate_physical_rows','reference_keys','teacher_M']:
            assert rec[key]==byid[rec['query_id']][key],('REUSE_AXIS_MISMATCH',rec['query_id'],key)
    write(AUTH,a)
    state['previous_authority']=state['authority'];state['authority']=bind(AUTH)
    save(OUT/'n64/checkpoint.pt',state)
    write(OUT/'preflight.json',dict(status='CONTINUE64_AT40_AND_FULL457_AT2000_FROZEN',authority=bind(AUTH),train=457,held=119))
    print({'prepared':True,'continue64':40,'full_start':2000},flush=True)

def load_data(a,n):
    if n==64:return OLD.load_data(a,n)
    assert n==457
    cache=read(OUT/'cache/validation.json');assert cache['authority']==bind(AUTH) and cache['status']=='FULL457_RELATIONS_READY'
    return RelationStream(a),[]

class RelationStream:
    def __init__(self,a):
        self.a=a;self.records=read(checked(a['full_train_inputs']))['records'];self.bank=D.Q.Bank(read(checked(a['full_ready']))['features'],'COL_ONLY','cpu');self.seen=set()
    def __len__(self):return len(self.records)
    def __iter__(self):
        for rec in self.records:
            v=read(OUT/'cache/queries'/f"query{rec['execution_ordinal']:03d}.json");assert v['authority']==bind(AUTH) and v['query_id']==rec['query_id']
            b=v['payload'];path=checked(b) if rec['query_id'] not in self.seen else b['path'];self.seen.add(rec['query_id'])
            pack=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
            if v['kind']=='packed_reuse':
                assert pack['query_id']==rec['query_id'] and torch.equal(pack['teacher'],torch.tensor(rec['teacher_M'],dtype=torch.float64));yield pack
            else:
                assert pack['authority']==bind(AUTH) and pack['query_id']==rec['query_id'] and pack['candidate_physical_rows']==rec['candidate_physical_rows']
                q=self.bank(rec['query_image_key']);parts=pack['compact'].split(pack['lengths']);xs=[]
                for j,key in enumerate(rec['reference_keys']):
                    ref=self.bank(key)
                    for z,part in [(q,parts[2*j]),(ref,parts[2*j+1])]:xs.append(expand(z,part))
                yield dict(x=torch.cat(xs),lengths=pack['lengths'],teacher=torch.tensor(rec['teacher_M'],dtype=torch.float64),query_id=rec['query_id'])

def compact(x):return torch.cat([x[:,128:256],x[:,514:520]],dim=1)
def expand(z,p):
    context=p[:,:128]
    return torch.cat([z['z'],context,z['z']-context,z['z']*context,z['xy'],p[:,128:]],dim=1)

def cache(a,index):
    assert 0<=index<a['cache_shards'];start=time.monotonic();records=read(checked(a['full_train_inputs']))['records']
    reused={x['query_id']:x for x in a['relations']};bank=D.Q.Bank(read(checked(a['full_ready']))['features'],'COL_ONLY','cpu')
    d=OUT/'cache/shards'/str(index);d.mkdir(parents=True,exist_ok=True);lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);receipts=[]
    for pos,rec in enumerate(records):
        if pos%a['cache_shards']!=index:continue
        vp=OUT/'cache/queries'/f"query{rec['execution_ordinal']:03d}.json"
        if vp.exists():receipts.append(bind(vp));continue
        if time.monotonic()-start>a['chunk_seconds']:return False
        if rec['query_id'] in reused:
            b=reused[rec['query_id']]['payload'];old=torch.load(checked(b),weights_only=True,map_location='cpu',mmap=True)
            assert old['query_id']==rec['query_id'] and torch.equal(old['teacher'],torch.tensor(rec['teacher_M'],dtype=torch.float64));kind='packed_reuse'
        else:
            q=bank(rec['query_image_key']);parts=[];lengths=[]
            with torch.no_grad():
                for key in rec['reference_keys']:
                    ref=bank(key);xq,xr=D.C.relation(q['z'],ref['z'],q['xy'],ref['xy'],'PAIR')
                    for z,x in [(q,xq),(ref,xr)]:
                        c=compact(x);assert torch.equal(x,expand(z,c)),'COMPACT_RELATION_NOT_EXACT';parts.append(c);lengths.append(len(x))
            dest=vp.with_suffix('.pt');save(dest,dict(authority=bind(AUTH),query_id=rec['query_id'],candidate_physical_rows=rec['candidate_physical_rows'],compact=torch.cat(parts),lengths=lengths));b=bind(dest);kind='compact_new'
        write(vp,dict(authority=bind(AUTH),query_id=rec['query_id'],kind=kind,payload=b,reconstruction='torch.equal for each newly computed pair'))
        receipts.append(bind(vp));print({'cached':rec['execution_ordinal'],'kind':kind,'seconds':time.monotonic()-start},flush=True)
    write(d/'validation.json',dict(status='RELATION_CACHE_SHARD_PASS',authority=bind(AUTH),queries=receipts));return True

def cache_join(a):
    rows=[]
    for i in range(a['cache_shards']):
        v=read(OUT/'cache/shards'/str(i)/'validation.json');assert v['authority']==bind(AUTH)
        rows.extend(read(checked(b)) for b in v['queries'])
    records=read(checked(a['full_train_inputs']))['records'];assert sorted(x['query_id'] for x in rows)==sorted(x['query_id'] for x in records)
    write(OUT/'cache/validation.json',dict(status='FULL457_RELATIONS_READY',authority=bind(AUTH),count=len(rows),reused=sum(x['kind']=='packed_reuse' for x in rows),new=sum(x['kind']=='compact_new' for x in rows)))

def evaluate(model,data,bias=None):
    rows=[];pred=[];targets=[]
    for d in data:
        with torch.no_grad():m,detail=D.packed_mass(model,d,True)
        raw=m.numpy();target=d['teacher'].numpy();pred.append(raw);targets.append(target)
        rows.append(dict(query_id=d['query_id'],M_uncalibrated=raw.tolist(),activation=detail,uncalibrated_metrics=D.metrics(raw,target)))
    targets=np.asarray(targets);bias=K.fit(np.asarray(pred),targets) if bias is None else bias
    for row,target in zip(rows,targets):
        mass=K.apply(row['M_uncalibrated'],bias);row.update(M=mass.tolist(),metrics=D.metrics(mass,target))
    mean={k:float(np.mean([row['metrics'][k] for row in rows])) for k in rows[0]['metrics']}
    return dict(rows=rows,mean_query_metrics=mean,bias=bias,gate=S.gate(mean,rows))

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
        limit=a['full_max_iterations'] if n==457 else a['max_outer_iterations']
        if iteration in a['snapshots'] or iteration==limit:
            path=dest/'snapshots'/f'iter{iteration:03d}.json'
            if path.exists():snapshot=read(path)
            else:
                fit=evaluate(model,data);test=evaluate(model,probe,fit['bias']) if probe else dict(status='HELD_NOT_OPENED',rows=[])
                frozen=dest/'models'/f'iter{iteration:03d}.pt';save(frozen,dict(authority=bind(AUTH),model=model.state_dict(),iteration=iteration))
                snapshot=dict(authority=bind(AUTH),iteration=iteration,evaluations=evaluations,model=bind(frozen),fit=fit,probe=test,probe_used_for_selection=False)
                write(path,snapshot);print({'snapshot':iteration,'n':n,'fit':fit['mean_query_metrics'],'gate':fit['gate'],'probe':test['mean_query_metrics']},flush=True)
            if all(snapshot['fit']['gate'].values()) or iteration==limit:
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

def predict(a):
    start=time.monotonic();T.OUT=OUT;T.AUTH=AUTH
    result=read(OUT/'n457/result.json');assert result['status']=='FIT_GATE_PASS'
    snap=read(checked(result['snapshot']));model=D.C.Quality();state=torch.load(checked(snap['model']),weights_only=True);model.load_state_dict(state['model'])
    final=OUT/'final_model.pt'
    if not final.exists():save(final,dict(authority=bind(AUTH),model=model.state_dict(),iteration=snap['iteration']))
    cal=OUT/'mass_calibration.json';write(cal,dict(authority=bind(AUTH),model=bind(final),bias=snap['fit']['bias'],train=457,held_reads=0,train_snapshot=result['snapshot']))
    fit_rows={r['query_id']:r for r in snap['fit']['rows']};train=read(checked(a['full_train_inputs']))['records'];held=read(checked(a['held_inputs']));assert not held['teacher_included'] and not held['identity_labels_included']
    bank=D.Q.Bank(read(checked(a['full_ready']))['features'],'COL_ONLY','cpu');bindings=[]
    for split,records in [('train',train),('held',held['records'])]:
        for rec in records:
            p=OUT/'predictions'/split/f"query{rec['execution_ordinal']:03d}.pt"
            if not p.exists():
                if time.monotonic()-start>a['chunk_seconds']:return False
                if split=='train':
                    row=fit_rows[rec['query_id']];raw=torch.tensor(row['M_uncalibrated'],dtype=torch.float64);mass=torch.tensor(row['M'],dtype=torch.float64);traces=[]
                else:
                    with torch.no_grad():raw,traces=T.estimate(rec,bank,model,True)
                    mass=torch.from_numpy(K.apply(raw.numpy(),snap['fit']['bias']))
                save(p,dict(authority=bind(AUTH),model=bind(final),query_id=rec['query_id'],execution_ordinal=rec['execution_ordinal'],candidate_physical_rows=rec['candidate_physical_rows'],M=mass,M_uncalibrated=raw,calibration=bind(cal),quality_traces=traces,teacher_at_inference=False,held_identity_reads=0))
            v=torch.load(p,weights_only=True);assert v['authority']==bind(AUTH) and v['model']==bind(final) and v['query_id']==rec['query_id']
            bindings.append(dict(split=split,query_id=rec['query_id'],prediction=bind(p)))
    write(OUT/'prediction_seal.json',dict(status='M_TEACHER_PREDICTIONS_SEALED',authority=bind(AUTH),model=bind(final),predictions=bindings,train=457,held=119,held_teacher_reads=0,held_identity_reads=0))
    write(OUT/'fit_validation.json',dict(status='M_TEACHER_FIT_AND_PREDICTIONS_PASS',authority=bind(AUTH),payload=bind(OUT/'prediction_seal.json')));return True

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
            else:write(d/'budget_stop.json',dict(status='STOP_BUDGET_NOT_INFORMATION_ABSENCE',authority=bind(AUTH)))
        elif read(rp)['status']=='FIT_GATE_PASS':
            if n==64:
                jobs['cache']=submit('cache',dep=job,array='0-7%8');jobs['cache_join']=submit('cache-join',dep=jobs['cache'])
            else:jobs['predict']=submit('predict',dep=job)
    elif stage=='cache-join':jobs['full_fit']=submit('run',457,job)
    elif stage=='predict':
        if (OUT/'fit_validation.json').exists():jobs['join']=submit('join',dep=job)
        else:
            chunks=list((OUT/'predict_chunks').glob('*.json'))
            if len(chunks)<16:jobs['predict']=submit('predict',dep=job)
            else:write(OUT/'predict_budget_stop.json',dict(status='PREDICT_BUDGET_STOP',authority=bind(AUTH)))
    write(dest,dict(previous=job,stage=stage,n=n,jobs=jobs));print({'submitted':jobs},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('n',nargs='?',type=int,default=0);z=p.parse_args();torch.set_num_threads(8)
    if z.stage=='prepare':prepare()
    else:
        a=read(AUTH)
        for b in a['sources']:checked(b)
        assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
        if z.stage.startswith('advance-'):advance(a,z.stage[len('advance-'):],z.n)
        elif z.stage=='run':run(a,z.n)
        elif z.stage=='cache':
            if not cache(a,int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))):raise SystemExit(75)
        elif z.stage=='cache-join':cache_join(a)
        elif z.stage=='predict':
            complete=predict(a);write(OUT/'predict_chunks'/f"{os.environ['SLURM_JOB_ID']}.json",dict(complete=complete,authority=bind(AUTH)))
        elif z.stage=='join':
            assert read(OUT/'n457/result.json')['status']=='FIT_GATE_PASS';T.OUT=OUT;T.AUTH=AUTH;T.join(a)
        else:raise ValueError(z.stage)
