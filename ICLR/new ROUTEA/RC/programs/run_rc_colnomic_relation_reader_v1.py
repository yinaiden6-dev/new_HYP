#!/usr/bin/env python3
"""Prepared 64 TRAIN / 32 opened group-held pilot; cached ColNomic only.

All candidate outputs are written before the separate label join. The worker
has no access to teacher tensors or held identity files. Restart boundaries
are exact optimizer steps; no best-held endpoint selection.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
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

import rc_colnomic_relation_reader_v1 as C

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_colnomic_relation_reader_v1'
AUTH = ROOT / 'registry/rc_colnomic_relation_reader_authority_v1_20260924.json'
PARENT = ROOT / 'registry/rc_h593_pair_quality_authority_v1_20260923.json'
PLAN = ROOT / 'plan/RC_COLNOMIC_ROMA_FREE_RELATION_READOUT_EXECUTION_V1_20260924.md'


def need(value, message):
    if not value:
        raise ValueError(message)


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(p), sha256=h.hexdigest())


def checked(b):
    need(bind(b['path']) == b, 'SHA_DRIFT:' + b['path'])
    return Path(b['path'])


def write(p, value, mutable=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists() and not mutable:
        need(p.read_text() == text, 'IMMUTABLE:' + str(p)); return
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def save(p, value, mutable=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    need(mutable or not p.exists(), 'IMMUTABLE:' + str(p))
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def emit(event, **kw):
    print(json.dumps(dict(event=event, **kw), ensure_ascii=False, allow_nan=False), flush=True)


def prepare():
    if AUTH.exists():
        load_contract(); emit('ALREADY_PREPARED', authority=bind(AUTH)); return
    parent = read(PARENT)
    catalog = read(checked(parent['catalog']))
    split = read(checked(parent['public_sources']['split']))['folds'][0]
    train_roles = read(checked(parent['fold_sources']['0']['train_roles']))['records']
    roles = {r['query_id']: r for r in train_roles}
    gallery = read(checked(parent['public_sources']['gallery']))['records']
    identities = {x['physical_row']: x['identity'] for x in gallery}
    def choose(ids, n, role):
        return sorted(ids, key=lambda q:hashlib.sha256(('relation-v1/17/'+role+'/'+q).encode()).hexdigest())[:n]
    train_ids = choose(split['train_query_ids'], 64, 'train')
    held_ids = choose(split['heldout_query_ids'], 32, 'held')
    need(len(train_ids)==64 and len(held_ids)==32 and not set(train_ids)&set(held_ids), 'SPLIT')
    # This existing mixed summary is accessed only by intake. Export exclusively
    # free content values, not its teacher mass or native head features.
    content_path = ROOT / 'results/rc_h593_simple_explanations_v1/cache.json'
    source = read(content_path)
    free = {r['query_id']:r for r in source['rows']}
    metas = {r['query_id']:r for r in catalog['queries']}
    rows, targets, images, input_bindings = [], {}, {}, []
    for qid in train_ids + held_ids:
        meta = metas[qid]
        p = torch.load(checked(meta['payload']), weights_only=True, map_location='cpu')
        axis = list(p['candidate_physical_rows']); refs = [x['image_key'] for x in p['pairs']]
        need(axis==free[qid]['axis'] and len(axis)==len(set(axis))==128, 'FULL_C128_LINEAGE')
        need(p['winner']==axis.index(p['raw_ranked_physical_rows'][0]), 'RAW_WINNER')
        need(p['challenger_positions']==[i for i in range(128) if i!=p['winner']], 'CHALLENGER_AXIS')
        row = dict(query_id=qid, execution_ordinal=p['execution_ordinal'], source_image_sha256=p['source_image_sha256'],
                   query_image_key=p['query_image_key'], reference_image_keys=refs, axis=axis,
                   winner=p['winner'], raw=list(p['candidate_raw_scores']), L0=free[qid]['free_content'])
        need(len(row['L0'])==128 and np.isfinite(row['L0']).all(), 'CONTENT_AXIS')
        rows.append(row); input_bindings.append(meta['payload'])
        for key in [row['query_image_key']] + refs:
            image = catalog['images'][key]
            images[key] = dict(tokens=image['input'], grid=image['item']['grid'],
                               image_sha256=image['item']['image_sha256'], frame=image['item']['frame'])
        if qid in train_ids:
            positions = [i for i,v in enumerate(axis) if identities[v]==roles[qid]['identity']]
            need(len(positions)<=1, 'UNIQUE_TARGET')
            pos = positions[0] if positions else None
            challenger = -2 if pos is None else (-1 if pos==p['winner'] else p['challenger_positions'].index(pos))
            targets[qid] = dict(target_position=pos, target_challenger=challenger, component=roles[qid]['component'])
    train_hashes = {r['source_image_sha256'] for r in rows if r['query_id'] in train_ids}
    held_hashes = {r['source_image_sha256'] for r in rows if r['query_id'] in held_ids}
    need(not train_hashes&held_hashes, 'TRAIN_HELD_IMAGE_OVERLAP')
    manifest = dict(rows=rows, images=images, train_query_ids=train_ids, held_query_ids=held_ids,
                    original_fold=0, labels_included=False, teacher_fields_included=False)
    write(OUT/'manifest.json', manifest)
    write(OUT/'train_labels.json', targets)
    write(OUT/'intake.json', dict(parent=bind(PARENT), catalog=parent['catalog'],
        split=parent['public_sources']['split'], train_roles=parent['fold_sources']['0']['train_roles'],
        gallery=parent['public_sources']['gallery'], query_payloads=input_bindings,
        free_content_source=bind(content_path), mixed_teacher_fields_discarded=True,
        train=64, held=32, images=len(images), train_components=len({v['component'] for v in targets.values()}),
        train_target_absent=sum(v['target_position'] is None for v in targets.values()),
        selection='SHA256 relation-v1/17/role/query_id, no outcome or target-membership selection',
        full_C128=True, raw_images_read=0, held_label_reads=0))
    sys.path.insert(0, str(ROOT/'tests'))
    from test_colnomic_relation_reader_v1 import run
    qualification = run()
    write(OUT/'synthetic_validation.json', qualification)
    files = [Path(__file__), Path(C.__file__), ROOT/'tests/test_colnomic_relation_reader_v1.py',
             ROOT/'programs/dispatch_rc_colnomic_relation_reader_v1.py',
             ROOT/'slurm/rc_colnomic_relation_reader_v1.sbatch', PLAN]
    a = dict(status='ROMA_FREE_RELATION_READER_PILOT_AUTHORIZED', user_authorization='2026-09-24 做',
        manifest=bind(OUT/'manifest.json'), train_labels=bind(OUT/'train_labels.json'), intake=bind(OUT/'intake.json'),
        synthetic_validation=bind(OUT/'synthetic_validation.json'), code_sources=[bind(f) for f in files],
        join_sources=dict(curator=parent['join_sources']['curator'], gallery=parent['public_sources']['gallery'],
                          original_fold=parent['fold_sources']['0']['full_payload']),
        arms=list(C.ARMS), seed=17, passes=8, updates=512, head_warmstart_steps=1000,
        reader_lr=.001, head_lr=.003, warmstart_lr=.03, weight_decay=.001, clip_norm=1.,
        reader_dtype='float32', action_dtype='float64', relation_width=C.WIDTH, relation_layers=2,
        temperature=C.TEMPERATURE, loss='COST1', candidates=128, direct_teacher_inputs=False,
        train_target_absent_policy='all challenger logits suppressed; never count HOLD as correct',
        held_policy='Fixed final endpoint, already opened H593 fold0 development subset; no untouched claim',
        benchmark_max_seconds_per_update=90, benchmark_max_rss_gib=24,
        worker_budget_seconds=400, max_requeues=64, inference_interventions=['native','shuffle','rotate'])
    write(AUTH,a)
    emit('PREPARED',authority=bind(AUTH),train=64,held=32,images=len(images))


def load_contract():
    a = read(AUTH)
    for b in a['code_sources'] + [a['manifest'],a['intake'],a['synthetic_validation']]:
        checked(b)
    return a, read(checked(a['manifest']))


def read_guard(a, manifest, stage):
    allowed = {Path(b['tokens']['path']).resolve() for b in manifest['images'].values()}
    if stage=='fit':
        allowed.add(Path(a['train_labels']['path']).resolve())
    protected_labels = Path(a['train_labels']['path']).resolve()
    def audit(event,args):
        if event=='socket.connect' and len(args)>1 and isinstance(args[1],tuple):
            raise ValueError('WORKER_NETWORK_FORBIDDEN')
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=Path(os.fsdecode(args[0])).resolve(); lower=str(p).lower()
        need(not any(t in lower for t in ('formal392','d1-mi','d1_mi','curator_roles','/target_join/')), 'PROTECTED_READ')
        if p==protected_labels:
            need(stage=='fit', 'NO_TRAIN_LABELS_IN_CAPACITY')
        if ROOT/'results' in p.parents:
            need(OUT in p.parents or p in allowed, 'NON_COLNOMIC_RESULT_READ:'+str(p))
    sys.addaudithook(audit)


class Bank:
    def __init__(self, images):
        self.images=images; self.cache=OrderedDict(); self.seen=set()
    def get(self,key):
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        entry=self.images[key]
        if key not in self.seen:
            checked(entry['tokens']);self.seen.add(key)
        p=torch.load(entry['tokens']['path'],weights_only=True,map_location='cpu',mmap=True)
        tokens=p['tokens'].to(torch.float32)
        grid=tuple(entry['grid'])
        need(tokens.shape==(grid[0]*grid[1],128) and bool(torch.isfinite(tokens).all()),'FULL_IMAGE_PATCH_GRID')
        need(p['item']['grid']==list(grid) and p['item']['image_sha256']==entry['image_sha256'],'IMAGE_LINEAGE')
        value=dict(tokens=tokens,grid=grid)
        self.cache[key]=value
        if len(self.cache)>4096:
            self.cache.popitem(last=False)
        return value


def score_candidate(model,row,bank,pos,intervention='native',trace=False):
    q=bank.get(row['query_image_key']);r=bank.get(row['reference_image_keys'][pos])
    return model(q['tokens'],r['tokens'],q['grid'],r['grid'],intervention,trace)


def scores(model,row,bank,intervention='native'):
    if model.arm=='BASE':
        return torch.zeros(128)
    return torch.stack([score_candidate(model,row,bank,i,intervention)[0] for i in range(128)])


def features(row):
    return C.base_features(row['raw'],row['winner'],torch.tensor(row['L0'],dtype=torch.float64))


def update(model,head,opt,row,target,bank):
    start=time.monotonic(); opt.zero_grad()
    with torch.no_grad():
        s=scores(model,row,bank)
    s.requires_grad_(model.arm!='BASE')
    z=C.logits(features(row),head,s,row['winner']);loss=C.cost(z,target)
    need(bool(torch.isfinite(loss)),'FINITE_LOSS');loss.backward()
    active=0
    if model.arm!='BASE':
        for i in range(128):
            if s.grad[i]!=0:
                (score_candidate(model,row,bank,i)[0]*s.grad[i]).backward();active+=1
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'READER_GRADIENT')
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    else:
        norm=0.
    need(bool(torch.isfinite(head.grad).all()),'HEAD_GRADIENT')
    torch.nn.utils.clip_grad_norm_([head],1.)
    opt.step()
    return dict(loss=float(loss.detach()),reader_gradient_norm=norm,active_candidates=active,seconds=time.monotonic()-start)


def optimizer(a,model,head):
    groups=[dict(params=[head],lr=a['head_lr'])]
    if model.arm!='BASE':
        groups.append(dict(params=model.parameters(),lr=a['reader_lr']))
    return torch.optim.AdamW(groups,weight_decay=a['weight_decay'])


def capacity(a,m,budget):
    import resource
    row=next(r for r in m['rows'] if r['query_id']==m['train_query_ids'][0])
    bank=Bank(m['images']); folder=OUT/'capacity';folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    for arm in C.ARMS:
        dest=folder/(arm+'.json')
        if dest.exists():
            need(read(dest)['authority']==bind(AUTH),'CAPACITY_AUTHORITY');continue
        if time.monotonic()-start>budget:
            return 75
        model=C.Reader(arm);head=torch.nn.Parameter(torch.tensor([.1,.2,-.3],dtype=torch.float64))
        opt=optimizer(a,model,head)
        times=[]
        for _ in range(2):
            info=update(model,head,opt,row,0,bank);times.append(info)
            emit('CAPACITY_UPDATE',arm=arm,**info)
        checks=[]
        q=bank.get(row['query_image_key'])['tokens'].double()
        for i in (0,127):
            r=bank.get(row['reference_image_keys'][i])['tokens'].double()
            exact=float((F.normalize(q,dim=1)@F.normalize(r,dim=1).T).max(1).values.mean())
            err=abs(exact-row['L0'][i]);need(err<2e-10,'ORIGINAL_FREE_CONTENT_PARITY')
            checks.append(dict(candidate_position=i,error=err))
        value=dict(authority=bind(AUTH),arm=arm,times=times,content_checks=checks,
                   rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
                   parameters=sum(p.numel() for p in model.parameters()),real_label_reads=0,temporary_models_discarded=True)
        write(dest,value)
    records=[read(folder/(arm+'.json')) for arm in C.ARMS]
    need(all(max(t['seconds'] for t in r['times'])<a['benchmark_max_seconds_per_update'] for r in records),'CAPACITY_TIME_LIMIT')
    need(all(r['rss_gib']<a['benchmark_max_rss_gib'] for r in records),'CAPACITY_MEMORY_LIMIT')
    write(OUT/'capacity_validation.json',dict(status='FULL_C128_CPU_CAPACITY_PASS',authority=bind(AUTH),
        records=[bind(folder/(arm+'.json')) for arm in C.ARMS],real_label_reads=0,encoder_forwards=0,roma_forwards=0))
    emit('CAPACITY_PASS',records=records)
    return 0


def fit(a,m,index,budget):
    arm=C.ARMS[index]; folder=OUT/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'validation.json').exists():
        receipt=read(folder/'validation.json');need(receipt['authority']==bind(AUTH),'FIT_AUTHORITY')
        for b in receipt['predictions']+[receipt['model']]:checked(b)
        return 0
    need(read(OUT/'capacity_validation.json')['status']=='FULL_C128_CPU_CAPACITY_PASS','CAPACITY_FIRST')
    targets=read(checked(a['train_labels'])); rows={r['query_id']:r for r in m['rows']}
    bank=Bank(m['images']);model=C.Reader(arm);head=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64))
    opt=optimizer(a,model,head);step=0;history=[]; cp=folder/'checkpoint.pt';start=time.monotonic()
    if cp.exists():
        state=torch.load(cp,weights_only=True,map_location='cpu');need(state['authority']==bind(AUTH) and state['arm']==arm,'CHECKPOINT_BINDING')
        model.load_state_dict(state['reader']);head.data.copy_(state['head']);opt.load_state_dict(state['optimizer'])
        step=state['step'];history=state['history']
    else:
        xx=torch.stack([features(rows[q]) for q in m['train_query_ids']])
        yy=torch.tensor([targets[q]['target_challenger'] for q in m['train_query_ids']])
        warm=torch.optim.AdamW([head],lr=a['warmstart_lr'],weight_decay=a['weight_decay'])
        ii=torch.arange(len(yy));mask=yy>=0
        for _ in range(a['head_warmstart_steps']):
            warm.zero_grad();z=xx@head;wrong=z.clone();wrong[ii[mask],yy[mask]]=-torch.inf
            loss=torch.where(mask,F.softplus(-z[ii,yy.clamp_min(0)])+F.softplus(wrong.amax(1)),F.softplus(z.amax(1))).mean()
            loss.backward();warm.step()
        save(folder/'initial.pt',dict(authority=bind(AUTH),head=head.detach(),warmstart_loss=float(loss.detach())))
        emit('HEAD_WARMSTART',arm=arm,loss=float(loss.detach()))
    def checkpoint():
        save(cp,dict(authority=bind(AUTH),arm=arm,step=step,reader=model.state_dict(),head=head.detach(),
                     optimizer=opt.state_dict(),history=history),mutable=True)
    while step<a['updates']:
        if time.monotonic()-start>budget:
            checkpoint();emit('CHUNK',arm=arm,step=step,total=a['updates']);return 75
        epoch,offset=divmod(step,len(m['train_query_ids']))
        generator=torch.Generator().manual_seed(a['seed']+1000003*epoch)
        idx=int(torch.randperm(len(m['train_query_ids']),generator=generator)[offset])
        qid=m['train_query_ids'][idx]
        info=update(model,head,opt,rows[qid],targets[qid]['target_challenger'],bank)
        step+=1;history.append(dict(step=step,query_id=qid,**info));checkpoint()
        emit('FIT',arm=arm,step=step,total=a['updates'],**info)
    final=folder/'final_model.pt'
    if not final.exists():
        save(final,dict(authority=bind(AUTH),arm=arm,step=step,reader=model.state_dict(),head=head.detach()))
    frozen=torch.load(final,weights_only=True,map_location='cpu')
    need(torch.equal(frozen['head'],head.detach()) and all(torch.equal(v,frozen['reader'][k]) for k,v in model.state_dict().items()),'FINAL_ENDPOINT_DRIFT')
    model_binding=bind(final)
    predictions=[]
    for role,ids in (('train',m['train_query_ids']),('held',m['held_query_ids'])):
        for ordinal,qid in enumerate(ids):
            row=rows[qid]
            interventions=['native','shuffle','rotate'] if role=='held' and arm in ('SPATIAL','SET') else ['native']
            for intervention in interventions:
                dest=folder/'predictions'/f'{role}_{ordinal:03d}_{intervention}.json'
                if dest.exists():
                    p=read(dest);need(p['model']==model_binding,'PREDICTION_MODEL');predictions.append(bind(dest));continue
                if time.monotonic()-start>budget:
                    emit('PREDICTION_CHUNK',arm=arm,role=role,ordinal=ordinal,intervention=intervention);return 75
                with torch.no_grad():
                    s=scores(model,row,bank,intervention);x=features(row);z=C.logits(x,head,s,row['winner'])
                    idx=[i for i in range(128) if i!=row['winner']];j=int(z.argmax())
                    position=idx[j] if float(z[j])>0 else row['winner']
                p=dict(authority=bind(AUTH),model=model_binding,query_id=qid,role=role,intervention=intervention,
                    axis=row['axis'],winner=row['winner'],raw=row['raw'],L0=row['L0'],relation=s.tolist(),
                    logits=z.tolist(),selected_position=position,selected=row['axis'][position],held_labels_read=0)
                write(dest,p);predictions.append(bind(dest))
                emit('PREDICT',arm=arm,role=role,ordinal=ordinal,intervention=intervention)
            # Save rich relation states for a fixed subset/positions, not cases
            # chosen using errors, improvements or held identity.
            if role=='held' and ordinal<2 and arm!='BASE':
                tracepath=folder/'traces'/f'held_{ordinal:03d}.pt'
                if not tracepath.exists():
                    if time.monotonic()-start>budget:return 75
                    traces={}
                    with torch.no_grad():
                        for i in sorted({row['winner'],0}):
                            _,t=score_candidate(model,row,bank,i,trace=True)
                            traces[str(i)]=t
                    save(tracepath,dict(model=model_binding,query_id=qid,source_query_key=row['query_image_key'],
                        reference_keys=row['reference_image_keys'],candidate_states=traces))
    write(folder/'validation.json',dict(status='FIXED_ENDPOINT_PREDICTIONS_SEALED',authority=bind(AUTH),
        model=model_binding,predictions=predictions,train=64,held=32,updates=step,held_label_reads=0))
    emit('SEALED',arm=arm,train=64,held=32,updates=step)
    return 0


def join(a,m):
    # All arms must seal before this function opens any held identity label.
    receipts={arm:read(OUT/arm/'validation.json') for arm in C.ARMS}
    for arm,receipt in receipts.items():
        need(receipt['authority']==bind(AUTH) and receipt['status']=='FIXED_ENDPOINT_PREDICTIONS_SEALED','ALL_ARMS_SEALED')
        for b in [receipt['model']]+receipt['predictions']:checked(b)
    curator=read(checked(a['join_sources']['curator']))
    gallery=read(checked(a['join_sources']['gallery']))['records']
    identities={r['physical_row']:r['identity'] for r in gallery}
    # The curator format is qualified explicitly, rather than guessed per row.
    records=curator.get('records',curator.get('roles'))
    need(isinstance(records,list),'CURATOR_SCHEMA')
    roles={r['query_id']:r for r in records}
    train_components={roles[q]['component'] for q in m['train_query_ids']}
    held_components={roles[q]['component'] for q in m['held_query_ids']}
    need(not train_components&held_components,'GROUP_LEAKAGE')
    train_identities={roles[q]['identity'] for q in m['train_query_ids']}
    need(not train_identities&{roles[q]['identity'] for q in m['held_query_ids']},'IDENTITY_LEAKAGE')
    row_by_id={r['query_id']:r for r in m['rows']}
    tables=[];perquery={};error=0.
    for arm,receipt in receipts.items():
        model=torch.load(receipt['model']['path'],weights_only=True,map_location='cpu')
        head=np.array(model['head'],dtype=np.float64)
        by={}
        for b in receipt['predictions']:
            p=read(b['path']);qid=p['query_id'];n=len(p['axis']);w=p['winner'];idx=[i for i in range(n) if i!=w]
            source=row_by_id[qid]
            need(p['axis']==source['axis'] and p['raw']==source['raw'] and p['L0']==source['L0'] and w==source['winner'],'SEALED_INPUT_LINEAGE')
            need((qid in m['train_query_ids']) == (p['role']=='train'),'ROLE_ASSIGNMENT')
            raw=np.array(p['raw']);l=np.array(p['L0']);s=np.array(p['relation'],dtype=np.float64)
            xx=np.stack(((raw[idx]-raw[w])/max(raw.std(),1e-12),
                (l[idx]-l[w])/(abs(l[idx])+abs(l[w])+1e-12),np.ones(n-1)),1)
            # Worker subtracts FP32 relation values before casting to FP64.
            ds=(np.array(p['relation'],dtype=np.float32)[idx]-np.float32(p['relation'][w])).astype(np.float64)
            z=xx@head+ds
            err=float(np.max(abs(z-np.array(p['logits']))));error=max(error,err);need(err<1e-9,'INDEPENDENT_LOGIT_REPLAY')
            pos=idx[int(z.argmax())] if z.max()>0 else w;need(pos==p['selected_position'],'ACTION_REPLAY')
            role=roles[qid];target=role['identity'];rawok=identities[p['axis'][w]]==target;correct=identities[p['selected']]==target
            target_pos=[i for i,v in enumerate(p['axis']) if identities[v]==target]
            record=dict(query_id=qid,component=role['component'],raw_correct=rawok,correct=correct,
                rescue=correct and not rawok,breaks=rawok and not correct,target_in_C128=bool(target_pos),
                selected=p['selected'],winner=p['axis'][w],relation_target=s[target_pos[0]] if target_pos else None,
                source=b,role=p['role'],intervention=p['intervention'])
            by.setdefault((p['role'],p['intervention']),[]).append(record)
        for (role,intervention),rs in by.items():
            need(len(rs)==(64 if role=='train' else 32),'PREDICTION_DENOMINATOR')
            groups={}
            for r in rs:groups.setdefault(r['component'],[]).append(int(r['correct'])-int(r['raw_correct']))
            means=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(17)
            boots=rng.choice(means,(5000,len(means)),replace=True).mean(1)
            table=dict(arm=arm,role=role,intervention=intervention,n=len(rs),groups=len(groups),
                raw=sum(r['raw_correct'] for r in rs),correct=sum(r['correct'] for r in rs),
                rescue=sum(r['rescue'] for r in rs),breaks=sum(r['breaks'] for r in rs),
                target_absent=sum(not r['target_in_C128'] for r in rs),group_mean_net=float(means.mean()),
                exploratory_group_bootstrap95=np.quantile(boots,[.025,.975]).tolist())
            tables.append(table);perquery[arm+'/'+role+'/'+intervention]=rs
    old={r['query_id']:r for r in read(checked(a['join_sources']['original_fold']))['predictions']}
    reference=[]
    for qid in m['held_query_ids']:
        row=row_by_id[qid];p=old[qid]['models']['COST1'];z=np.array([float.fromhex(v) for v in p['logits_hex']])
        idx=[i for i in range(128) if i!=row['winner']];pos=idx[int(z.argmax())] if z.max()>0 else row['winner']
        need(row['axis'][pos]==p['selected'],'ORIGINAL_HEAD_ACTION_REPLAY')
        role=roles[qid];rawok=identities[row['axis'][row['winner']]]==role['identity'];ok=identities[p['selected']]==role['identity']
        reference.append(dict(query_id=qid,component=role['component'],correct=ok,raw_correct=rawok,
                              rescue=ok and not rawok,breaks=rawok and not ok,selected=p['selected']))
    perquery['ORIGINAL_COST1/held/native']=reference
    tables.append(dict(arm='ORIGINAL_COST1',role='held',intervention='native',n=32,groups=len(held_components),
                       correct=sum(r['correct'] for r in reference),raw=sum(r['raw_correct'] for r in reference),
                       rescue=sum(r['rescue'] for r in reference),breaks=sum(r['breaks'] for r in reference),
                       note='System reference trained on original full fold; different budget/architecture'))
    comparisons=[]
    spatial={r['query_id']:r for r in perquery['SPATIAL/held/native']}
    for other in ['BASE','COMPRESSED','SET','ORIGINAL_COST1']:
        group_delta={};rescues=breaks=0
        for r in perquery[other+'/held/native']:
            qid=r['query_id'];delta=int(spatial[qid]['correct'])-int(r['correct'])
            group_delta.setdefault(r['component'],[]).append(delta)
            rescues+=delta>0;breaks+=delta<0
        means=np.array([np.mean(v) for v in group_delta.values()]);rng=np.random.default_rng(17)
        samples=rng.choice(means,(5000,len(means)),replace=True).mean(1)
        comparisons.append(dict(primary=other=='SET',comparison='SPATIAL-'+other,rescue=rescues,breaks=breaks,
            net=rescues-breaks,group_mean=float(means.mean()),group_bootstrap95=np.quantile(samples,[.025,.975]).tolist()))
    result=dict(status='PILOT_INDEPENDENT_RECOUNT_PASS',authority=bind(AUTH),tables=tables,per_query=perquery,comparisons=comparisons,
                maximum_logit_error=error,scope='Opened H593 fold0 subset 64 TRAIN / 32 held; no external confirmation',
                teacher_inputs=False,encoder_forwards=0,roma_forwards=0)
    write(OUT/'result.json',result)
    lines=['# 无 RoMa 联合关系读出：64/32 开发试验','',
           '固定原 ColNomic 自然 C128，所有候选保留。仅训练身份标签；最终固定八遍端点。',
           'SET/SPATIAL 等参数、等样本与更新次数，计算时间另列；COMPRESSED 是历史算子的同末端适配，参数量不同。',
           '这是已打开的 H593 fold0 子集，不能称全593或外部确认。','',
           '|臂|角色|干预|正确/总数|RAW|救回|损失|组数|','|---|---|---|---|---|---|---|---|']
    for r in tables:
        lines.append(f"|{r['arm']}|{r['role']}|{r['intervention']}|{r['correct']}/{r['n']}|{r['raw']}|{r['rescue']}|{r['breaks']}|{r['groups']}|")
    lines+=['',f'独立 NumPy 候选logit核算最大误差：{error:.3g}。',
            '完整候选分数、关系证据、模型和优化器断点在 `results/rc_colnomic_relation_reader_v1/`。',
            '结果支持程度需结合训练拟合、SET/SPATIAL配对和坐标干预分析；单次负结果不证明token缺信息。']
    (ROOT/'reports/REPORT_COLNOMIC_ROMA_FREE_RELATION_READER_V1_20260924.md').write_text('\n'.join(lines)+'\n')
    emit('JOIN_COMPLETE',tables=tables,maximum_logit_error=error)
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','capacity','fit','join']);p.add_argument('--index',type=int,default=0)
    p.add_argument('--budget',type=float,default=400);args=p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    torch.set_num_interop_threads(1)
    if args.role=='prepare':prepare();return 0
    a,m=load_contract()
    if args.role!='join':read_guard(a,m,args.role)
    if args.role=='capacity':return capacity(a,m,args.budget)
    if args.role=='fit':return fit(a,m,args.index,args.budget)
    return join(a,m)


if __name__=='__main__':
    raise SystemExit(main())
