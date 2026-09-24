#!/usr/bin/env python3
"""ColQwen base-native C128 content and original RoMa global mass, grouped OOF5."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
OUT=ROOT/'results/rc_colpali_native_mass_head_v1'
AUTH=ROOT/'registry/rc_colpali_native_mass_head_authority_v1_20260923.json'
TOK=ROOT/'results/rc_colpali_h593_query_tokens_v4_legacy'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):
    assert sha(b['path'])==b['sha256'],b['path']
    return Path(b['path'])
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    s=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==s,('IMMUTABLE',str(p));return
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:f.write(s);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def prepare():
    assert not AUTH.exists()
    import copy
    parent_path=ROOT/'registry/rc_colpali_mass_transfer_authority_v1_20260923.json'
    parent=read(parent_path)
    quality=ROOT/'results/rc_colpali_native_quality_v1'
    validation=read(quality/'validation.json')
    assert validation['status']=='COLQWEN_BASE_NATIVE_ALL_QUALITY_PASS'
    workers=read(checked(validation['workers']));assert len(workers['records'])==593
    write(OUT/'workers.json',workers)
    labels={r['physical_row']:r['identity'] for r in read(checked(parent['gallery']))['records']}
    folds=copy.deepcopy(parent['folds']);byid={r['query_id']:r for r in workers['records']}
    for key,fold in folds.items():
        roles={r['query_id']:r for r in read(checked(fold['train_roles']))['records']}
        fold['effective_train_query_ids']=[q for q in fold['train_query_ids'] if any(labels[p]==roles[q]['identity'] for p in byid[q]['candidate_physical_rows'])]
    sources=[Path(__file__),ROOT/'programs/cache_colpali_h593_queries_v4_legacy.py',
        ROOT/'programs/run_rc_six_cause_loss_binding_v1.py',ROOT/'slurm/rc_colpali_native_mass_gpu_v1.sbatch',
        ROOT/'slurm/rc_colpali_native_mass_cpu_v1.sbatch',ROOT/'plan/RC_COLQWEN_BASE_NATIVE_MASS_HEAD_V1_20260923.md']
    a={k:v for k,v in parent.items() if k not in ('sources','workers','folds','candidate_source','scope','status')}
    a.update(status='COLQWEN_BASE_NATIVE_GLOBAL_MASS_HEAD_AUTHORIZED',sources=[bind(p) for p in sources],
        workers=bind(OUT/'workers.json'),folds=folds,parent_diagnostic=bind(parent_path),quality_validation=bind(quality/'validation.json'),
        candidate_source='ColQwen base FP64 MaxSim over full5413 reference tokens, gallery identity dedup, own natural C128',
        scope='Native ColQwen base candidate pipeline; original M quality and simplified MASS5, not full local seven-parameter transfer',
        evidence='Opened H593 grouped OOF5, all593 held-out predictions; new natural recall, no target insertion')
    write(AUTH,a)
    print({'status':'COLQWEN_BASE_NATIVE_HEAD_PREPARED','effective_train':{f:len(x['effective_train_query_ids']) for f,x in folds.items()}},flush=True)


def guard(stage,fold=None):
    a=read(AUTH)
    for b in a['sources']:checked(b)
    assert os.environ.get('SLURM_JOB_ID')
    allowed={Path(a['workers']['path']).resolve()}
    if stage=='fit':allowed.update(Path(a[k]['path']).resolve() for k in ('gallery','split'));allowed.add(Path(a['folds'][str(fold)]['train_roles']['path']).resolve())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if stage!='join' and p.suffix=='.json' and ROOT/'results' in p.parents:
            assert OUT in p.parents or TOK in p.parents or p in allowed,('UNAUTHORIZED_RESULT',str(p))
    sys.addaudithook(audit)
    return a,read(checked(a['workers']))['records']

def score(a,rows,shard,budget):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cache_colpali_h593_queries_v4_legacy as cache
    import fcntl
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    d=OUT/f'shards/{shard:02d}';d.mkdir(parents=True,exist_ok=True)
    lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if shard!=0:
        v=read(OUT/'shards/00/validation.json');assert v['authority']==bind(AUTH) and v['status']=='COLQWEN_BASE_SCORE_SHARD_PASS'
    if shard==0:checked(a['gallery_tokens'])
    refs=cache.gallery_memmap(a['gallery_tokens']['path'])
    selected=[r for r in rows if r['execution_ordinal']%50==shard];start=time.monotonic();seals=[]
    for row in selected:
        qd=OUT/f"queries/query{row['execution_ordinal']:03d}"
        if (qd/'validation.json').exists():
            v=read(qd/'validation.json');assert v['authority']==bind(AUTH);checked(v['payload']);checked(v['intermediate']);seals.append(bind(qd/'validation.json'));continue
        if time.monotonic()-start>budget:return 75
        q= torch.load(checked(row['query_tokens']),map_location='cpu',weights_only=False)
        assert q['query_id']==row['query_id'] and q['image_sha256']==row['image_sha256']
        assert q['tokens'].shape==(1030,128) and torch.equal(torch.where(q['image_token_mask'])[0],torch.arange(1024))
        query=q['tokens'].to('cuda',dtype=torch.float64);qi=F.normalize(query[:1024],dim=1)
        raw=[];stats=[];traces=[];error=0.
        with torch.inference_mode():
            for physical in row['candidate_physical_rows']:
                ref=torch.from_numpy(np.array(refs[physical],copy=True)).to('cuda',dtype=torch.float64)
                sim=query@ref.T;rv,ri=sim.max(1);raw.append(float(rv.sum()))
                local=qi@F.normalize(ref[:1024],dim=1).T
                forward=local.topk(2,dim=1);back=local.topk(2,dim=0).values
                st=torch.stack([forward.values[:,0].mean(),back[0].mean(),
                    (forward.values[:,0]-forward.values[:,1]).mean(),(back[0]-back[1]).mean(),local.mean()])
                stats.append(st.cpu().tolist())
                traces.append(dict(physical_row=physical,raw_max=rv.cpu(),raw_reference_token=ri.to(torch.int32).cpu(),
                    image_max=forward.values[:,0].cpu(),image_reference_token=forward.indices[:,0].to(torch.int32).cpu(),
                    image_reverse_max=back[0].cpu()))
                if physical==row['candidate_physical_rows'][0]:
                    # Independent CPU/NumPy scoring on the first pair of every query.
                    nq=q['tokens'].numpy().astype('float64');nr=np.asarray(refs[physical],dtype='float64')
                    rr=np.max(nq@nr.T,axis=1).sum();nqi=nq[:1024]/np.linalg.norm(nq[:1024],axis=1,keepdims=True)
                    nri=nr[:1024]/np.linalg.norm(nr[:1024],axis=1,keepdims=True);ns=nqi@nri.T
                    p=np.partition(ns,-2,axis=1)[:,-2:];b=np.partition(ns,-2,axis=0)[-2:]
                    vals=np.array([p.max(1).mean(),b.max(0).mean(),np.abs(p[:,1]-p[:,0]).mean(),np.abs(b[1]-b[0]).mean(),ns.mean()])
                    error=max(abs(rr-raw[-1]),float(abs(vals-np.array(stats[-1])).max()));assert error<2e-9,error
        axis=row['candidate_physical_rows'];winner=max(range(128),key=lambda i:(raw[i],-axis[i]));cs=[i for i in range(128) if i!=winner]
        qd.mkdir(parents=True,exist_ok=True);ip=qd/'intermediate.pt';tmp=qd/'intermediate.tmp'
        with tmp.open('wb') as f:torch.save(dict(authority=bind(AUTH),query_id=row['query_id'],pairs=traces,query_tokens=row['query_tokens'],gallery_tokens=a['gallery_tokens']),f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,ip)
        p=dict(authority=bind(AUTH),query_id=row['query_id'],execution_ordinal=row['execution_ordinal'],candidate_physical_rows=axis,
            raw_scores=raw,winner=winner,challenger_positions=cs,statistics=stats,mass=row['mass'],intermediate=bind(ip),label_reads=0)
        assert np.isfinite(np.array(stats)).all() and np.isfinite(raw).all()
        write(qd/'payload.json',p);write(qd/'validation.json',dict(status='COLQWEN_BASE_CONTENT_QUERY_PASS',authority=bind(AUTH),
            payload=bind(qd/'payload.json'),intermediate=bind(ip),independent_cpu_first_pair_error=error,candidates=128))
        seals.append(bind(qd/'validation.json'));print(dict(event='COLQWEN_BASE_QUERY_SCORED',index=row['execution_ordinal'],seconds=time.monotonic()-start),flush=True)
    write(d/'validation.json',dict(status='COLQWEN_BASE_SCORE_SHARD_PASS',authority=bind(AUTH),queries=len(selected),validations=seals))
    return 0

def load_rows(a):
    rows=[]
    for s in range(50):
        v=read(OUT/f'shards/{s:02d}/validation.json');assert v['status']=='COLQWEN_BASE_SCORE_SHARD_PASS' and v['authority']==bind(AUTH)
        for b in v['validations']:
            t=read(checked(b));assert t['authority']==bind(AUTH)
            rows.append(read(checked(t['payload'])))
    rows.sort(key=lambda r:r['execution_ordinal']);assert [r['execution_ordinal'] for r in rows]==list(range(593))
    return rows

def features(row,kind,shuffle=False):
    import numpy as np
    raw=np.array(row['raw_scores']);win=row['winner'];cs=row['challenger_positions']
    gap=(raw[cs]-raw[win])/max(float(raw.std()),1e-12)
    if kind=='CONTENT7':v=np.array(row['statistics'])
    else:
        m=np.array(row['mass']);m=np.roll(m,64) if shuffle else m
        l=np.array(row['statistics'])[:,0];v=np.stack([m*l,m,l],axis=1)
    rel=(v[cs]-v[win])/(abs(v[cs])+abs(v[win])+1e-12)
    return np.concatenate([gap[:,None],rel],axis=1)

def fit(a,fold,budget):
    import numpy as np
    import torch
    import run_rc_six_cause_loss_binding_v1 as L
    torch.set_num_threads(8);start=time.monotonic()
    rows=load_rows(a);b=a['folds'][str(fold)];roles={r['query_id']:r for r in read(checked(b['train_roles']))['records']}
    fm=read(checked(a['split']))['folds'][fold];labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    train=[r for r in rows if r['query_id'] in roles];held=[r for r in rows if r['query_id'] in set(fm['heldout_query_ids'])]
    assert [r['query_id'] for r in train]==b['train_query_ids'] and not set(roles)&set(fm['heldout_query_ids'])
    ys=[L.H.target_position(r,roles[r['query_id']]['identity'],labels) for r in train];keep=[i for i,y in enumerate(ys) if y>=-1]
    assert [train[i]['query_id'] for i in keep]==b['effective_train_query_ids']
    y=torch.tensor([ys[i] for i in keep]);d=OUT/f'fold{fold}';d.mkdir(parents=True,exist_ok=True)
    models={};parameters={};maxerr=0.
    for model in a['models']:
        kind,obj=model.rsplit('_',1);path=d/(model+'_parameters.json')
        if path.exists():
            v=read(path);assert v['authority']==bind(AUTH);theta=torch.tensor([float.fromhex(t) for t in v['theta_hex']],dtype=torch.float64)
        else:
            if time.monotonic()-start>budget:return 75
            x=torch.tensor(np.stack([features(train[i],kind) for i in keep]),dtype=torch.float64)
            torch.manual_seed(17);theta=L.train(x,y,obj)
            write(path,dict(authority=bind(AUTH),theta_hex=[float(t).hex() for t in theta],steps=2000))
        parameters[model]=[float(t).hex() for t in theta]
        for shuf in ((False,True) if kind=='MASS5' else (False,)):
            name=model+('_CBIND' if shuf else '');models[name]=[]
            for r in held:
                x=features(r,kind,shuf);z=x@theta[:-1].numpy()+float(theta[-1]);zt=(torch.from_numpy(x)@theta[:-1]+theta[-1]).numpy()
                err=float(abs(z-zt).max());maxerr=max(maxerr,err);assert err<2e-10
                j=int(z.argmax());pos=r['challenger_positions'][j] if z[j]>0 else r['winner']
                assert int(zt.argmax())==j and bool(zt.max()>0)==bool(z.max()>0)
                models[name].append(dict(selected=r['candidate_physical_rows'][pos],logits_hex=[float(t).hex() for t in z]))
        print(dict(event='COLQWEN_BASE_HEAD_FIT',fold=fold,model=model,seconds=time.monotonic()-start),flush=True)
    predictions=[dict(query_id=r['query_id'],models={m:v[i] for m,v in models.items()}) for i,r in enumerate(held)]
    write(d/'payload.json',dict(authority=bind(AUTH),fold=fold,parameters=parameters,predictions=predictions,heldout_label_reads=0,
        effective_train_query_ids=b['effective_train_query_ids']))
    write(d/'validation.json',dict(status='COLQWEN_BASE_HEAD_NUMPY_TORCH_PASS',authority=bind(AUTH),payload=bind(d/'payload.json'),max_error=maxerr))
    return 0

def join(a):
    import numpy as np
    rows=load_rows(a);preds={};seals=[]
    for f in range(5):
        v=read(OUT/f'fold{f}/validation.json');assert v['status']=='COLQWEN_BASE_HEAD_NUMPY_TORCH_PASS' and v['authority']==bind(AUTH)
        p=read(checked(v['payload']));assert p['fold']==f and p['authority']==bind(AUTH)
        fm=read(checked(a['split']))['folds'][f];assert {r['query_id'] for r in p['predictions']}==set(fm['heldout_query_ids'])
        assert not set(preds)&set(fm['heldout_query_ids']);preds.update({r['query_id']:dict(fold=f,**r) for r in p['predictions']});seals.append(bind(OUT/f'fold{f}/validation.json'))
    write(OUT/'predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=seals))
    roles={r['query_id']:r for r in read(checked(a['curator']))['records']};labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    result_rows=[]
    for r in rows:
        qid=r['query_id'];role=roles[qid];f=preds[qid]['fold'];assert role['outer_fold']==f
        train=read(checked(a['folds'][str(f)]['train_roles']))['records'];assert role['component'] not in {t['component'] for t in train} and role['identity'] not in {t['identity'] for t in train}
        axis=r['candidate_physical_rows'];target={i for i,v in enumerate(axis) if labels[v]==role['identity']};assert len(target)<=1
        order={'COLQWEN_BASE_RAW_C128':sorted(range(128),key=lambda i:(-r['raw_scores'][i],axis[i])),
               'COLQWEN_BASE_IMAGE_L':sorted(range(128),key=lambda i:(-r['statistics'][i][0],axis[i]))}
        for name,p in preds[qid]['models'].items():
            z=[0.]*128
            for i,t in zip(r['challenger_positions'],p['logits_hex']):z[i]=float.fromhex(t)
            order[name]=sorted(range(128),key=lambda i:(-z[i],i!=r['winner'],i));assert axis[order[name][0]]==p['selected']
        rr={m:next((1./(j+1) for j,i in enumerate(v) if i in target),0.) for m,v in order.items()}
        result_rows.append(dict(query_id=qid,fold=f,component=role['component'],target_in_C128=bool(target),
            selected={m:axis[v[0]] for m,v in order.items()},correct={m:v==1 for m,v in rr.items()},rr=rr))
    counts={m:sum(r['correct'][m] for r in result_rows) for m in result_rows[0]['correct']};comparisons={}
    for base in ('COLQWEN_BASE_RAW_C128','CONTENT7_COST1','MASS5_COST1_CBIND'):
        for model in counts:
            if base==model:continue
            groups=defaultdict(list)
            for r in result_rows:groups[r['component']].append(int(r['correct'][model])-int(r['correct'][base]))
            delta=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(20260923)
            boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
            comparisons[base+'__to__'+model]=dict(rescue=sum(r['correct'][model] and not r['correct'][base] for r in result_rows),
                breaks=sum(r['correct'][base] and not r['correct'][model] for r in result_rows),
                component_balanced_delta=float(delta.mean()),bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))))
    assert len(result_rows)==593
    value=dict(status='COLQWEN_BASE_NATIVE_GLOBAL_MASS_TRANSFER_COMPLETE',authority=bind(AUTH),counts=counts,denominator=593,target_in_C128=sum(r['target_in_C128'] for r in result_rows),
        comparisons=comparisons,MRR_C128={m:sum(r['rr'][m] for r in result_rows)/593 for m in counts},rows=result_rows,
        scope=a['scope'],candidate_source=a['candidate_source'],evidence=a['evidence'])
    write(OUT/'result.json',value);write(OUT/'validation.json',dict(status='COLQWEN_BASE_NATIVE_TRANSFER_FIVE_FOLDS_593_PASS',authority=bind(AUTH),result=bind(OUT/'result.json')))
    print(dict(status=value['status'],counts=counts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score','fit','join']);p.add_argument('--index',type=int,default=0);p.add_argument('--budget',type=int,default=700);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        a,rows=guard(args.stage,args.index)
        if args.stage=='score':sys.exit(score(a,rows,args.index,args.budget))
        elif args.stage=='fit':sys.exit(fit(a,args.index,args.budget))
        else:join(a)
