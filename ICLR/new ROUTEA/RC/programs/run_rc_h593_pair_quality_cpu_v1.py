#!/usr/bin/env python3
"""CPU execution of the unchanged pair-quality mechanism; no capacity probe."""
import argparse
from collections import OrderedDict, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import rc_pair_quality_core_v1 as C
OUT=ROOT/'results/rc_h593_pair_quality_cpu_v1'
AUTH=ROOT/'registry/rc_h593_pair_quality_cpu_authority_v1_20260923.json'
CACHE=ROOT/'results/rc_h593_feature_fusion_cache_v1'
FUSION=ROOT/'results/rc_h593_feature_fusion_train_v2'
PARENT=ROOT/'registry/rc_h593_feature_fusion_train_authority_v2_20260922.json'
PLAN=ROOT/'plan/RC_H593_PAIR_QUALITY_CPU_V1_20260923.md'
ARMS=('COL_ONLY_SINGLE','COL_ONLY_PAIR','COARSE_SINGLE','COARSE_PAIR','NONE','ROMA')
CONFIGS=[dict(fold=f,arm=arm,seed=17) for f in range(5) for arm in ARMS]
STEPS=2000


def need(x,s):
    if not x: raise RuntimeError(s)
def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest())
def checked(b):
    if bind(b['path'])!=b:
        from rc_h593_pair_quality_execution_source_compat_v1 import approved_execution_source_change
        need(approved_execution_source_change(b),'SHA_DRIFT:'+b['path'])
    return Path(b['path'])
def write(p,v,mutable=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    s=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists() and not mutable:need(p.read_text()==s,'IMMUTABLE:'+str(p));return
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');tmp.write_text(s);os.replace(tmp,p)
def save(p,v,mutable=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    need(mutable or not p.exists(),'IMMUTABLE:'+str(p))
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def prepare():
    need(not AUTH.exists(),'NEW_AUTHORITY')
    parent=read(PARENT);ready=read(CACHE/'ready.json');need(ready['status']=='FUSION_ALL593_FEATURE_CACHE_PASS','CACHE_READY')
    catalog=read(checked(ready['catalog']));need(len(catalog['queries'])==593,'593_QUERIES')
    sources=[Path(__file__),Path(C.__file__),PLAN,
             ROOT/'reports/REPORT_H593_PAIR_QUALITY_HISTORY_AND_OVERLAP_20260923.md',
             ROOT/'slurm/rc_h593_pair_quality_cpu_v1.sbatch',ROOT/'programs/dispatch_rc_h593_pair_quality_cpu_v1.py']
    warm={}
    for q in catalog['queries']:
        p=FUSION/'initial'/f"query{q['execution_ordinal']:03d}.pt"
        if p.exists():warm[q['query_id']]=bind(p)
    write(AUTH,dict(status='PAIR_QUALITY_AUTHORIZED',user_authorization='2026-09-23 user: skip tests, modify program, run directly on cpuonly',
        sources=[bind(p) for p in sources],parent=bind(PARENT),ready=bind(CACHE/'ready.json'),catalog=ready['catalog'],
        public_sources=parent['public_sources'],fold_sources=parent['fold_sources'],join_sources=parent['join_sources'],
        warm_content=warm,configs=CONFIGS,steps=STEPS,precision='FP64',new_encoder_forwards=0,
        roma_teacher_loss=False,primary_loss='COST1',max_parallel=4,max_chunks_per_config=128,execution_device='cpu',capacity_gate_required=False,original_gpu_authority=bind(ROOT/'registry/rc_h593_pair_quality_authority_v1_20260923.json'),
        target_absent_kept=23,evidence='Opened H593 original grouped OOF5; not external confirmation'))
    v=dict(status='PAIR_QUALITY_CPU_EXECUTION_READY',authority=bind(AUTH),additional_tests=0,user_instruction='Skip tests; directly run CPU training',existing_core_validation=bind(ROOT/'results/rc_h593_pair_quality_v1/preflight.json'));write(OUT/'preflight.json',v)
    print(dict(event='PAIR_QUALITY_PREPARED',authority=bind(AUTH),warm_content_queries=len(warm)),flush=True)


def guard(stage,index):
    a=read(AUTH);need(a['status']=='PAIR_QUALITY_AUTHORIZED','AUTHORITY')
    for b in a['sources']:checked(b)
    checked(a['ready']);checked(a['catalog']);checked(a['parent'])
    need(read(OUT/'preflight.json')['authority']==bind(AUTH),'PREFLIGHT')
    need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    if stage=='fit':
        need(index in range(len(CONFIGS)),'CONFIG');fold=CONFIGS[index]['fold']
        allow.add(Path(a['fold_sources'][str(fold)]['train_roles']['path']).resolve())
    if stage=='join':
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for fs in a['fold_sources'].values():allow.update(Path(b['path']).resolve() for b in fs.values())
    warm={Path(b['path']).resolve() for b in a['warm_content'].values()}
    pilot=ROOT/'results/rc_h593_feature_fusion_pilot_v1'
    def audit(event,args):
        if event=='socket.connect':need(not isinstance(args[1],tuple),'OFFLINE')
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(t in s for t in ('d1-mi','d1_mi','formal392','/grozi/','/isic/','/target_join/')),'PROTECTED')
        if 'curator_roles' in s:need(stage=='join','HELD_LABELS_AFTER_SEAL')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage=='fit':
                part=p.relative_to(OUT).parts[0];own=part in ('preflight.json','benchmark.json','benchmark_validation.json','content',f'fit{index:02d}')
            need(own or CACHE in p.parents or pilot in p.parents or p in allow or p in warm,'UNLISTED_RESULT:'+str(p))
    sys.addaudithook(audit)
    return a


class Bank:
    def __init__(self,bindings,source,device):
        self.bindings=bindings;self.source=source;self.device=device;self.values=OrderedDict();self.seen=set()
    def __call__(self,key):
        if key in self.values:self.values.move_to_end(key);return self.values[key]
        b=self.bindings[key]
        if key not in self.seen:checked(b);self.seen.add(key)
        p=torch.load(b['path'],map_location='cpu',weights_only=True,mmap=True)
        mask=p['valid_patch_mask'].bool();need(bool(mask.any()),'VALID_TOKENS')
        xy=(p['cell_boxes_xyxy'][:,:2]+p['cell_boxes_xyxy'][:,2:])/2
        col=p['original_colnomic_tokens'].to(self.device,dtype=torch.float64)
        quality_input=(F.layer_norm(col,(128,)) if self.source=='COL_ONLY'
                       else p['projected_inputs'][self.source].to(self.device,dtype=torch.float64))
        value=dict(z=quality_input[mask.to(self.device)],
                   xy=xy[mask].to(self.device,dtype=torch.float64),col=F.normalize(col,dim=1),
                   valid_indices=mask.nonzero().flatten(),source=b)
        self.values[key]=value
        if len(self.values)>192:self.values.popitem(last=False)
        return value


def record(meta):
    p=torch.load(checked(meta['payload']),map_location='cpu',weights_only=True)
    need(p['query_id']==meta['query_id'] and len(p['pairs'])==128,'QUERY_AXIS')
    # Store no native spatial maps in the learner's record.
    return dict(query_id=p['query_id'],execution_ordinal=p['execution_ordinal'],query_image_key=p['query_image_key'],
        source_image_sha256=p['source_image_sha256'],candidate_physical_rows=p['candidate_physical_rows'],
        candidate_raw_scores=p['candidate_raw_scores'],winner=p['winner'],challenger_positions=p['challenger_positions'],
        pairs=[dict(image_key=x['image_key'],native_mass=float(x['native_c4'][1])) for x in p['pairs']])


def content(a,rec,bank):
    qid=rec['query_id'];dest=OUT/'content'/f"query{rec['execution_ordinal']:03d}.pt"
    if qid in a['warm_content']:
        p=torch.load(checked(a['warm_content'][qid]),map_location='cpu',weights_only=True)
        need(p['query_id']==qid and p['authority']==a['parent'],'WARM_BINDING')
        return p['C4'][:,0].to(bank.device)
    if dest.exists():
        p=torch.load(dest,map_location='cpu',weights_only=True);need(p['authority']==bind(AUTH) and p['query_id']==qid,'CONTENT_BINDING')
        return p['L0'].to(bank.device)
    with torch.no_grad():
        q=bank(rec['query_image_key'])['col']
        l=torch.stack([(q@bank(p['image_key'])['col'].T).max(1).values.mean() for p in rec['pairs']])
    # Multiple fold workers may compute the same immutable value concurrently.
    import fcntl
    dest.parent.mkdir(exist_ok=True,parents=True)
    with dest.with_suffix('.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if not dest.exists():save(dest,dict(authority=bind(AUTH),query_id=qid,L0=l.cpu(),label_reads=0))
    return l


def masses(rec,bank,model,arm,trace=False):
    if arm=='NONE':return torch.ones(128,dtype=torch.float64,device=bank.device),None
    if arm=='ROMA':return torch.tensor([p['native_mass'] for p in rec['pairs']],dtype=torch.float64,device=bank.device),None
    mode=arm.rsplit('_',1)[1];q=bank(rec['query_image_key']);values=[];traces=[]
    for p in rec['pairs']:
        m,t=C.pair(model,q,bank(p['image_key']),mode,trace)
        values.append(m)
        if trace:
            # Full 520-D contexts are deterministic from frozen source tokens;
            # storing them for all 30 fits would need terabytes. Preserve the
            # actual quality outputs and compact relation diagnostics instead.
            traces.append(dict(u=t['u'].detach().cpu(),v=t['v'].detach().cpu(),
                query_relation_summary=t['query_relation'][:,-8:].mean(0).detach().cpu(),
                reference_relation_summary=t['reference_relation'][:,-8:].mean(0).detach().cpu()))
    return torch.stack(values),traces


def update(rec,l,bank,model,theta,arm,target):
    with torch.no_grad():m,_=masses(rec,bank,model,arm)
    m=m.detach().requires_grad_(model is not None)
    x=C.features(rec['candidate_raw_scores'],rec['winner'],m,l)
    loss=C.cost(x@theta[:-1]+theta[-1],target);need(bool(torch.isfinite(loss)),'FINITE_LOSS');loss.backward()
    active=0
    if model is not None:
        grad=m.grad.detach();q=bank(rec['query_image_key']);mode=arm.rsplit('_',1)[1]
        for i in range(128):
            if not bool(grad[i]):continue
            value,_=C.pair(model,q,bank(rec['pairs'][i]['image_key']),mode)
            (value*grad[i]).backward();active+=1
    return dict(loss=float(loss.detach()),active=active)




def load_fold(a,cfg):
    split=read(checked(a['public_sources']['split']))['folds'][cfg['fold']]
    roles={r['query_id']:r for r in read(checked(a['fold_sources'][str(cfg['fold'])]['train_roles']))['records']}
    need(set(roles)==set(split['train_query_ids']),'TRAIN_ROLES')
    labels={r['physical_row']:r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    train=[];held=[]
    for meta in read(checked(a['catalog']))['queries']:
        rec=record(meta);qid=rec['query_id']
        if qid in roles:
            target=[i for i,p in enumerate(rec['candidate_physical_rows']) if labels[p]==roles[qid]['identity']]
            need(len(target)<=1,'UNIQUE_TARGET')
            if target:
                t=target[0];train.append((rec,-1 if t==rec['winner'] else rec['challenger_positions'].index(t)))
        else:need(qid in split['heldout_query_ids'],'HELD_AXIS');held.append(rec)
    need(not {r['source_image_sha256'] for r,_ in train}&{r['source_image_sha256'] for r in held},'IMAGE_DISJOINT')
    return train,held


def fit(a,index):
    need(not torch.cuda.is_available(),'CPU_ONLY_VISIBLE');need(a['execution_device']=='cpu','CPU_EXECUTION_AUTHORITY')
    cfg=CONFIGS[index];arm=cfg['arm'];folder=OUT/f'fit{index:02d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'validation.json').exists():checked(read(folder/'validation.json')['payload']);return
    start=time.monotonic();train,held=load_fold(a,cfg);ready=read(checked(a['ready']))
    source='COARSE' if arm.startswith('COARSE') else 'COL_ONLY'
    bank=Bank(ready['features'],source,'cpu');model=None if arm in ('NONE','ROMA') else C.Quality().cpu()
    theta=torch.nn.Parameter(torch.zeros(5,dtype=torch.float64,device='cpu'))
    groups=[dict(params=[theta],lr=.003)]
    if model is not None:groups.append(dict(params=model.parameters(),lr=.001))
    opt=torch.optim.AdamW(groups,weight_decay=.001);history=[];step=0;cp=folder/'checkpoint.pt'
    if cp.exists():
        p=torch.load(cp,map_location='cpu',weights_only=True);need(p['authority']==bind(AUTH) and p['config']==cfg,'CHECKPOINT')
        if model is not None:model.load_state_dict(p['model'])
        with torch.no_grad():theta.copy_(p['theta'])
        opt.load_state_dict(p['optimizer']);step=p['step'];history=p['history']
    else:
        xs=[];ys=[]
        for rec,t in train:
            l=content(a,rec,bank)
            m=(torch.tensor([p['native_mass'] for p in rec['pairs']],dtype=torch.float64,device='cpu') if arm=='ROMA'
               else torch.full((128,),1. if arm=='NONE' else .5,dtype=torch.float64,device='cpu'))
            xs.append(C.features(rec['candidate_raw_scores'],rec['winner'],m,l));ys.append(t)
        x=torch.stack(xs);y=torch.tensor(ys,device='cpu');init=torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
        for _ in range(STEPS):
            init.zero_grad();z=x@theta[:-1]+theta[-1];raw=y==-1;ii=torch.arange(len(y),device='cpu');idx=y.clamp_min(0)
            wrong=z.clone();wrong[ii[~raw],idx[~raw]]=-torch.inf
            loss=torch.where(raw,F.softplus(z.amax(1)),F.softplus(-z[ii,idx])+F.softplus(wrong.amax(1))).mean()
            loss.backward();init.step()
        save(folder/'initial.pt',dict(authority=bind(AUTH),theta=theta.detach().cpu(),train_query_ids=[r['query_id'] for r,_ in train],loss=float(loss.detach())))
        del x,xs,init
    def checkpoint():
        save(cp,dict(authority=bind(AUTH),config=cfg,step=step,model=None if model is None else model.state_dict(),
            theta=theta.detach(),optimizer=opt.state_dict(),history=history),mutable=True)
    while step<STEPS:
        if time.monotonic()-start>450:checkpoint();print(dict(event='PAIR_QUALITY_CHUNK',index=index,step=step),flush=True);return
        epoch,offset=divmod(step,len(train));g=torch.Generator().manual_seed(17+1000003*epoch)
        rec,target=train[int(torch.randperm(len(train),generator=g)[offset])]
        l=content(a,rec,bank);opt.zero_grad();info=update(rec,l,bank,model,theta,arm,target)
        need(bool(torch.isfinite(theta.grad).all()) and (model is None or all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())),'FINITE_GRADIENT')
        opt.step();step+=1
        if step%25==0:
            history.append(dict(step=step,query_id=rec['query_id'],**info));checkpoint()
            print(dict(event='PAIR_QUALITY_UPDATE',index=index,step=step,**info),flush=True)
    checkpoint()
    final=folder/'final_model.pt'
    if not final.exists():save(final,dict(authority=bind(AUTH),config=cfg,theta=theta.detach().cpu(),model=None if model is None else {k:v.cpu() for k,v in model.state_dict().items()}))
    frozen=torch.load(final,map_location='cpu',weights_only=True)
    need(frozen['authority']==bind(AUTH) and frozen['config']==cfg and torch.equal(frozen['theta'],theta.detach().cpu()),'FINAL_MODEL_RESUME')
    if model is not None:need(all(torch.equal(v.cpu(),frozen['model'][k]) for k,v in model.state_dict().items()),'FINAL_QUALITY_RESUME')
    train_diagnostics=[]
    for rec,target in train:
        dest=folder/'train_predictions'/f"query{rec['execution_ordinal']:03d}.json"
        if dest.exists():
            p=read(dest);need(p['model']==bind(final),'TRAIN_DIAGNOSTIC_MODEL');train_diagnostics.append(p);continue
        if time.monotonic()-start>450:print(dict(event='PAIR_QUALITY_TRAIN_DIAGNOSTIC_CHUNK',index=index),flush=True);return
        with torch.no_grad():
            l=content(a,rec,bank);m,_=masses(rec,bank,model,arm)
            x=C.features(rec['candidate_raw_scores'],rec['winner'],m,l);z=x@theta[:-1]+theta[-1]
            pred=int(z.argmax()) if z.max()>0 else -1
            p=dict(model=bind(final),query_id=rec['query_id'],raw_correct=target==-1,correct=pred==target,
                target_challenger=target,selected_challenger=pred,loss=float(C.cost(z,target)),
                M=m.cpu().tolist(),L0=l.cpu().tolist(),logits=z.cpu().tolist())
        write(dest,p);train_diagnostics.append(p)
    predictions=[]
    for rec in held:
        dest=folder/'predictions'/f"query{rec['execution_ordinal']:03d}.pt"
        if dest.exists():predictions.append(bind(dest));continue
        if time.monotonic()-start>450:print(dict(event='PAIR_QUALITY_PREDICT_CHUNK',index=index),flush=True);return
        with torch.no_grad():
            l=content(a,rec,bank);m,traces=masses(rec,bank,model,arm,trace=True)
            x=C.features(rec['candidate_raw_scores'],rec['winner'],m,l);z=x@theta[:-1]+theta[-1]
            j=int(z.argmax());pos=rec['challenger_positions'][j] if z[j]>0 else rec['winner']
            # Candidate-binding destruction preserves L0 and RAW; only M moves.
            mc=m.roll(1);xc=C.features(rec['candidate_raw_scores'],rec['winner'],mc,l);zc=xc@theta[:-1]+theta[-1]
            jc=int(zc.argmax());pc=rec['challenger_positions'][jc] if zc[jc]>0 else rec['winner']
        save(dest,dict(authority=bind(AUTH),model=bind(final),query_id=rec['query_id'],execution_ordinal=rec['execution_ordinal'],
            candidate_physical_rows=rec['candidate_physical_rows'],winner=rec['winner'],challenger_positions=rec['challenger_positions'],
            raw=rec['candidate_raw_scores'],M=m.cpu(),L0=l.cpu(),X=x.cpu(),logits=z.cpu(),selected=rec['candidate_physical_rows'][pos],
            cbind_logits=zc.cpu(),cbind_selected=rec['candidate_physical_rows'][pc],quality_traces=traces,
            source_keys=[rec['query_image_key']]+[p['image_key'] for p in rec['pairs']],ready=a['ready'],heldout_label_reads=0))
        predictions.append(bind(dest))
    maxerr=0
    for b in predictions:
        p=torch.load(checked(b),map_location='cpu',weights_only=True)
        mass=p['M'].numpy();l=p['L0'].numpy();raw=p['raw'];w=p['winner'];ix=p['challenger_positions']
        mu=sum(raw)/128;sd=(sum((v-mu)**2 for v in raw)/128)**.5
        def sym(v):return (v[ix]-v[w])/(abs(v[ix])+abs(v[w])+1e-12)
        xn=np.stack(([(raw[i]-raw[w])/max(sd,1e-12) for i in ix],sym(mass*l),sym(mass),sym(l)),1)
        th=theta.detach().cpu().numpy();zn=(xn*th[:-1]).sum(1)+th[-1]
        err=float(abs(zn-p['logits'].numpy()).max());need(err<2e-10,'NUMPY_LOGITS');maxerr=max(maxerr,err)
        j=int(zn.argmax());pos=ix[j] if zn[j]>0 else w;need(p['selected']==p['candidate_physical_rows'][pos],'NUMPY_ACTION')
        if p['quality_traces'] is not None:
            rebuilt=np.array([np.sqrt(t['u'].numpy().mean()*t['v'].numpy().mean()) for t in p['quality_traces']])
            need(np.max(abs(rebuilt-mass))<2e-12,'NUMPY_QUALITY_MASS')
        cm=np.roll(mass,1)
        xcb=np.stack((xn[:,0],sym(cm*l),sym(cm),sym(l)),1)
        zcb=(xcb*th[:-1]).sum(1)+th[-1]
        need(np.max(abs(zcb-p['cbind_logits'].numpy()))<2e-10,'NUMPY_CBIND')
        j=int(zcb.argmax());pos=ix[j] if zcb[j]>0 else w;need(p['cbind_selected']==p['candidate_physical_rows'][pos],'NUMPY_CBIND_ACTION')
    write(folder/'payload.json',dict(authority=bind(AUTH),config=cfg,predictions=predictions,model=bind(final),
        train_query_ids=[r['query_id'] for r,_ in train],steps=step,heldout_label_reads=0,
        train_fit=dict(n=len(train_diagnostics),correct=sum(p['correct'] for p in train_diagnostics),
            raw_correct=sum(p['raw_correct'] for p in train_diagnostics),mean_loss=float(np.mean([p['loss'] for p in train_diagnostics])))))
    write(folder/'validation.json',dict(status='PAIR_QUALITY_FOLD_NUMPY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),max_logit_error=maxerr))


def join(a):
    fits=[];vals=[]
    for i,cfg in enumerate(CONFIGS):
        p=OUT/f'fit{i:02d}/validation.json';v=read(p);need(v['status']=='PAIR_QUALITY_FOLD_NUMPY_PASS' and v['authority']==bind(AUTH),'ALL_FOLDS')
        f=read(checked(v['payload']));need(f['config']==cfg,'CONFIG_BINDING');fits.append(f);vals.append(bind(p))
    write(OUT/'prediction_seal.json',dict(authority=bind(AUTH),validations=vals))
    roles={r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']};rows={}
    for fitrow in fits:
        cfg=fitrow['config'];arm=cfg['arm'];fs=a['fold_sources'][str(cfg['fold'])]
        tr=read(checked(fs['train_roles']))['records'];ti={r['identity'] for r in tr};tg={r['component'] for r in tr}
        old={r['query_id']:r for r in read(checked(fs['full_payload']))['predictions']}
        for b in fitrow['predictions']:
            p=torch.load(checked(b),map_location='cpu',weights_only=True);role=roles[p['query_id']]
            need(role['identity'] not in ti and role['component'] not in tg and role['outer_fold']==cfg['fold'],'GROUP_IDENTITY_DISJOINT')
            r=rows.setdefault(p['query_id'],dict(query_id=p['query_id'],fold=cfg['fold'],component=role['component'],correct={},selected={},mrr={}))
            if not r['correct']:
                r['correct']['RAW']=labels[p['candidate_physical_rows'][p['winner']]]==role['identity']
                r['correct']['ORIGINAL_COST1']=labels[old[p['query_id']]['models']['COST1']['selected']]==role['identity']
                r['target_in_C128']=any(labels[x]==role['identity'] for x in p['candidate_physical_rows'])
            for name,sel in [(arm,p['selected']),(arm+'_CBIND',p['cbind_selected'])]:r['correct'][name]=labels[sel]==role['identity'];r['selected'][name]=sel
            z=np.zeros(128);z[p['challenger_positions']]=p['logits'].numpy();order=sorted(range(128),key=lambda i:(-z[i],i!=p['winner'],p['candidate_physical_rows'][i]))
            hit=[k for k,i in enumerate(order) if labels[p['candidate_physical_rows'][i]]==role['identity']];r['mrr'][arm]=1/(hit[0]+1) if hit else 0.
    rows=sorted(rows.values(),key=lambda r:r['query_id']);need(len(rows)==593 and sum(r['target_in_C128'] for r in rows)==570,'593_DENOMINATOR')
    counts={k:sum(r['correct'][k] for r in rows) for k in rows[0]['correct']};need(counts['RAW']==426 and counts['ORIGINAL_COST1']==481,'BASELINE_PARITY')
    pairs=[('COL_ONLY_SINGLE','COL_ONLY_PAIR'),('COARSE_SINGLE','COARSE_PAIR'),('COL_ONLY_PAIR','COARSE_PAIR'),('NONE','COL_ONLY_PAIR'),('ROMA','COL_ONLY_PAIR')]
    comparisons={}
    from math import comb
    for b,m in pairs:
        rescue=sum(not r['correct'][b] and r['correct'][m] for r in rows);brk=sum(r['correct'][b] and not r['correct'][m] for r in rows)
        groups=defaultdict(list)
        for r in rows:groups[r['component']].append(int(r['correct'][m])-int(r['correct'][b]))
        d=np.array([np.mean(v) for _,v in sorted(groups.items())]);need(len(d)==64,'64_GROUPS');rng=np.random.default_rng(20260923)
        ci=np.quantile(d[rng.integers(0,64,(10000,64))].mean(1),[.025,.975])
        n=rescue+brk;pval=min(1.,2*sum(comb(n,j) for j in range(min(rescue,brk)+1))/2**n) if n else 1.
        comparisons[b+'__to__'+m]=dict(rescue=rescue,breaks=brk,net=rescue-brk,group_ci95=ci.tolist(),paired_p=pval)
    # Five planned comparisons, including the independent equal-budget RoMa comparator.
    running=0.
    for rank,(key,v) in enumerate(sorted(comparisons.items(),key=lambda kv:kv[1]['paired_p'])):
        running=max(running,min(1.,v['paired_p']*(len(comparisons)-rank)));v['holm_p']=running
    write(OUT/'result.json',dict(authority=bind(AUTH),status='PAIR_QUALITY_H593_OOF_COMPLETE',counts=counts,comparisons=comparisons,
        mrr={arm:float(np.mean([r['mrr'][arm] for r in rows])) for arm in ARMS},rows=rows,
        train_fit_by_fold=[dict(config=f['config'],**f['train_fit']) for f in fits],
        evidence='Opened H593, seed17, candidate recall570/593; fixed pooled projection; not universal representation impossibility'))
    write(OUT/'validation.json',dict(status='PAIR_QUALITY_JOIN_PASS',result=bind(OUT/'result.json'),queries=593,configs=30))
    print(dict(event='PAIR_QUALITY_COMPLETE',counts=counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','fit','join'));p.add_argument('--index',type=int,default=0);args=p.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage,args.index)
        if args.stage=='fit':fit(a,args.index)
        else:join(a)
