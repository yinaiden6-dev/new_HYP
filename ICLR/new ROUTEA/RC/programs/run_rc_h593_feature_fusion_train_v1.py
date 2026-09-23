#!/usr/bin/env python3
"""Retrieval-only output fusion: frozen folds, resumable all-C128 training."""
import argparse
from collections import OrderedDict,defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_feature_fusion_cache_v1 as K
import rc_feature_fusion_training_v1 as T
import run_rc_six_cause_loss_binding_v1 as L
F,P=K.F,K.P
read,write,save,bind,checked,need=K.read,K.write,K.save,K.bind,K.checked,K.need
OUT=ROOT/'results/rc_h593_feature_fusion_train_v1'
AUTH=ROOT/'registry/rc_h593_feature_fusion_train_authority_v1_20260922.json'
PLAN=ROOT/'plan/RC_H593_FEATURE_FUSION_TRAIN_V1_20260922.md'
LAUNCH=ROOT/'slurm/rc_h593_feature_fusion_train_v1.sbatch'
CPU_LAUNCH=ROOT/'slurm/rc_h593_feature_fusion_train_join_v1.sbatch'
PARENT=ROOT/'registry/rc_h593_quality_operator_eval_authority_v1_20260922.json'
CONFIGS=[dict(seed=seed,kind=kind,fold=fold,source=source,mode=mode)
         for seed in (17,29) for kind in ('COST1','CE') for fold in range(5)
         for source in (*F.SOURCES,'NO_ADAPTER') for mode in ('FREE','ROMA_WEIGHTED')]
STEPS=2000


def prepare():
    need(not AUTH.exists(),'NEW_AUTHORITY');parent=read(PARENT)
    write(AUTH,dict(status='FEATURE_FUSION_TRAIN_AUTHORIZED',cache_authority=bind(K.AUTH),parent=bind(PARENT),
        public_sources=parent['public_sources'],fold_sources=parent['fold_sources'],join_sources=parent['join_sources'],shared_sources=list(parent['code_sources'].values()),
        sources=[bind(p) for p in (Path(__file__),Path(T.__file__),Path(F.__file__),Path(K.__file__),Path(P.__file__),Path(L.__file__),PLAN,LAUNCH,CPU_LAUNCH)],
        configs=CONFIGS,steps=STEPS,epochs='Deterministic cyclic shuffle of recall-present TRAIN queries; update count fixed',
        precision='FP64 model, scores, gradients and optimizer; frozen original tokens preserved',
        batch_queries=1,all_candidates=128,adapter_lr=.001,head_lr=.003,weight_decay=.001,
        initialization='Zero output adapter; weighted head is original same-fold/same-loss head; FREE head matched full-TRAIN 2000-step warmup',
        primary='COST1',seeds=[17,29],selection='No best seed/arm/epoch selection; all configs reported',
        chunk_seconds=640,max_chunks_per_config=128,
        evidence='Opened H593 original grouped OOF5, natural RAW C128, 593 denominator; not external confirmation',
        user_authorization='2026-09-22 continue full fusion branch after qualified job5157186'))
    result=T.synthetic_test();result['authority']=bind(AUTH);write(OUT/'preflight.json',result)
    print(dict(event='FUSION_TRAIN_PREPARED',configs=len(CONFIGS),authority=bind(AUTH)),flush=True)


def guard(stage,index):
    a=read(AUTH);need(a['status']=='FEATURE_FUSION_TRAIN_AUTHORIZED','AUTHORITY')
    for b in a['sources']:checked(b)
    for b in a['shared_sources']:checked(b)
    checked(a['cache_authority']);checked(a['parent']);need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    need(read(OUT/'preflight.json')['authority']==bind(AUTH),'PREFLIGHT')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    if stage in ('fit','verify'):
        need(index in range(len(CONFIGS)),'CONFIG_RANGE')
        f=CONFIGS[index]['fold'];allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(f)].values())
    if stage=='join':
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for fs in a['fold_sources'].values():allow.update(Path(b['path']).resolve() for b in fs.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(k in s for k in ('d1-mi','d1_mi','formal392','/grozi/','/isic/','/target_join/','/reports/')),'PROTECTED')
        if 'curator_roles' in s:need(stage=='join','NO_HELD_LABELS_BEFORE_SEALS')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage in ('fit','verify'):
                name=p.relative_to(OUT).parts[0];own=name in ('preflight.json','benchmark.json','benchmark_validation.json','initial',f'fit{index:03d}')
            need(own or K.OUT in p.parents or P.OUT in p.parents or p in allow,'UNLISTED_RESULT:'+str(p))
    sys.addaudithook(audit)
    return a


def replace_checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


class Bank:
    def __init__(self,bindings,source):self.bindings=bindings;self.source=source;self.cache=OrderedDict();self.verified=set()
    def __call__(self,key):
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        b=self.bindings[key]
        if key not in self.verified:checked(b);self.verified.add(key)
        p=torch.load(b['path'],map_location='cpu',weights_only=True,mmap=True)
        value=dict(original_colnomic_tokens=p['original_colnomic_tokens'],projected_inputs={self.source:p['projected_inputs'][self.source]})
        self.cache[key]=value
        if len(self.cache)>512:self.cache.popitem(last=False)
        return value


def benchmark(a):
    need(torch.cuda.is_available(),'GPU_REQUIRED');torch.set_float32_matmul_precision('highest')
    pilot=torch.load(checked(read(P.OUT/'validation.json')['payload']),map_location='cpu',weights_only=True)
    manifest=read(P.OUT/'manifest.json');bindings={item['key']:read(P.feature_file(item).with_name('validation.json'))['payload'] for item in manifest['images']}
    record={k:pilot[k] for k in ('query_image_key','pairs','candidate_raw_scores','winner')}
    rows=[]
    for source in F.SOURCES:
        bank=Bank(bindings,source)
        for mode in ('FREE','ROMA_WEIGHTED'):
            for kind in ('COST1','CE'):
                adapter=F.OutputAdapter().cuda();head=torch.nn.Parameter(torch.linspace(-.2,.4,7,dtype=torch.float64,device='cuda'))
                optimizer=torch.optim.AdamW([{'params':adapter.parameters(),'lr':.001},{'params':[head],'lr':.003}],weight_decay=.001)
                torch.cuda.reset_peak_memory_stats();elapsed=[]
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
                    result=T.two_pass_backward(record,bank,adapter,head,source,mode,kind,0)
                    need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()),'FINITE_GRADIENTS')
                    need(bool(adapter.up.weight.grad.abs().sum()>0),'ADAPTER_GRADIENT')
                    optimizer.step();torch.cuda.synchronize();elapsed.append(time.monotonic()-start)
                row=dict(source=source,mode=mode,kind=kind,seconds=elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated(),**result)
                rows.append(row);print(dict(event='FULL_C128_TRAINING_BENCHMARK',**row),flush=True)
                del adapter,head,optimizer;torch.cuda.empty_cache()
    need(all(r['peak_gpu_bytes']<30*1024**3 and max(r['seconds'])<120 for r in rows),'RUNTIME_CAPACITY_GATE')
    write(OUT/'benchmark.json',dict(authority=bind(AUTH),status='FUSION_FULL128_BACKWARD_CAPACITY_PASS',rows=rows,
        synthetic_target_position=0,real_identity_label_reads=0,formal_training_updates=0,
        benchmark_weights_discarded=True,gpu=torch.cuda.get_device_name(0)))
    write(OUT/'benchmark_validation.json',dict(status='FUSION_TRAINING_CAPACITY_PASS',authority=bind(AUTH),payload=bind(OUT/'benchmark.json'),
        synthetic_gradient_preflight=bind(OUT/'preflight.json'),arms=16))


def load_scope(a,cfg):
    ready=read(K.OUT/'ready.json');need(ready['status']=='FUSION_ALL593_FEATURE_CACHE_PASS' and ready['authority']==a['cache_authority'],'ALL_FEATURES_READY')
    catalog=read(checked(ready['catalog']));sp=read(checked(a['public_sources']['split']))['folds'][cfg['fold']]
    fs=a['fold_sources'][str(cfg['fold'])];roles={r['query_id']:r for r in read(checked(fs['train_roles']))['records']}
    trids=set(sp['train_query_ids']);teids=set(sp['heldout_query_ids']);need(set(roles)==trids and not trids&teids,'TRAIN_HELD_DISJOINT')
    labels={r['physical_row']:r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    old=read(checked(fs['full_payload']));val=read(checked(fs['full_validation']));need(val['payload']==fs['full_payload'] and val['status']=='SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS','WARM_HEAD_VALIDATED')
    train=[];held=[];alltrain=[]
    for meta in catalog['queries']:
        rec=torch.load(checked(meta['payload']),map_location='cpu',weights_only=True)
        need(rec['authority']==a['cache_authority'] and rec['query_id']==meta['query_id'],'QUERY_RECORD_BINDING')
        if rec['query_id'] in trids:
            alltrain.append(rec);target=L.H.target_position(rec,roles[rec['query_id']]['identity'],labels)
            if target>=-1:train.append((rec,target))
        else:need(rec['query_id'] in teids,'HELD_AXIS');held.append(rec)
    need([q['query_id'] for q in alltrain]==old['train_query_ids'],'ORIGINAL_TRAIN_ORDER')
    need(not {q['source_image_sha256'] for q in alltrain}&{q['source_image_sha256'] for q in held},'IMAGE_DISJOINT')
    return ready,train,held,old,roles


def initial_features(record,mode,bank,device):
    if mode=='ROMA_WEIGHTED':
        c4=torch.stack([p['native_c4'] for p in record['pairs']])
        return F.differentiable_features(record['candidate_raw_scores'],c4,record['winner'])
    folder=OUT/'initial';folder.mkdir(exist_ok=True,parents=True);path=folder/f"query{record['execution_ordinal']:03d}.pt"
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            p=torch.load(path,map_location='cpu',weights_only=True)
            need(p['authority']==bind(AUTH) and p['query_id']==record['query_id'],'INITIAL_CACHE_BINDING');return p['X']
        with torch.no_grad():c4=T.all_scores(record,bank,None,'COL_ONLY','FREE',device).cpu()
        x=F.differentiable_features(record['candidate_raw_scores'],c4,record['winner'])
        save(path,dict(authority=bind(AUTH),query_id=record['query_id'],C4=c4,X=x,label_reads=0))
        return x


def config_order(n,seed,step):
    epoch,offset=divmod(step,n);gen=torch.Generator().manual_seed(seed+1000003*epoch)
    return int(torch.randperm(n,generator=gen)[offset])


def save_state(folder,index,adapter,theta,optimizer,step,ready,history,warm):
    replace_checkpoint(folder/'checkpoint.pt',dict(authority=bind(AUTH),config=CONFIGS[index],step=step,
        ready=bind(K.OUT/'ready.json'),adapter=None if adapter is None else adapter.state_dict(),theta=theta.detach(),
        optimizer=optimizer.state_dict(),history=history,warm_head_hex=warm))


def fit(a,index):
    need(torch.cuda.is_available(),'GPU_REQUIRED');torch.set_float32_matmul_precision('highest')
    benchmark=read(OUT/'benchmark_validation.json');need(benchmark['status']=='FUSION_TRAINING_CAPACITY_PASS' and benchmark['authority']==bind(AUTH),'TRAINING_BENCHMARK_REQUIRED')
    checked(benchmark['payload']);cfg=CONFIGS[index];folder=OUT/f'fit{index:03d}';folder.mkdir(exist_ok=True,parents=True)
    if (folder/'validation.json').exists():checked(read(folder/'validation.json')['payload']);return
    started=time.monotonic();ready,train,held,old,roles=load_scope(a,cfg)
    source=cfg['source'] if cfg['source']!='NO_ADAPTER' else 'COL_ONLY';bank=Bank(ready['features'],source)
    adapter=None if cfg['source']=='NO_ADAPTER' else F.OutputAdapter(cfg['seed']).cuda()
    theta=torch.nn.Parameter(torch.zeros(7,dtype=torch.float64,device='cuda'))
    groups=[dict(params=[theta],lr=.003)]
    if adapter is not None:groups.append(dict(params=adapter.parameters(),lr=.001))
    optimizer=torch.optim.AdamW(groups,weight_decay=.001)
    checkpoint=folder/'checkpoint.pt';history=[]
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cuda',weights_only=True)
        need(cp['authority']==bind(AUTH) and cp['config']==cfg and cp['ready']==bind(K.OUT/'ready.json'),'CHECKPOINT_BINDING')
        if adapter is not None:adapter.load_state_dict(cp['adapter'])
        with torch.no_grad():theta.copy_(cp['theta'])
        optimizer.load_state_dict(cp['optimizer']);step=cp['step'];history=cp['history'];warm=cp['warm_head_hex']
    else:
        if cfg['mode']=='ROMA_WEIGHTED':
            warm=old['parameters']['COST1' if cfg['kind']=='COST1' else 'ALL_CE']
        else:
            xs=[]
            for j,(rec,target) in enumerate(train):
                xs.append(initial_features(rec,'FREE',bank,'cuda'))
                if j%32==0:print(dict(event='FREE_HEAD_WARM_INPUTS',config=index,queries=j+1),flush=True)
            t=L.train(torch.stack(xs),torch.tensor([t for _,t in train]),cfg['kind']);warm=[float(x).hex() for x in t]
        with torch.no_grad():theta.copy_(torch.tensor([float.fromhex(v) for v in warm],dtype=torch.float64,device='cuda'))
        step=0;save_state(folder,index,adapter,theta,optimizer,step,ready,history,warm)
    while step<STEPS:
        if time.monotonic()-started>640:
            save_state(folder,index,adapter,theta,optimizer,step,ready,history,warm)
            print(dict(event='FUSION_TRAIN_CHUNK_SAVED',config=index,step=step,total=STEPS),flush=True);return
        rec,target=train[config_order(len(train),cfg['seed'],step)];optimizer.zero_grad(set_to_none=True)
        if adapter is None:
            x=initial_features(rec,cfg['mode'],bank,'cuda').cuda();loss=T.objective(x@theta[:-1]+theta[-1],target,cfg['kind']);loss.backward();info=dict(loss=float(loss.detach()),active_vjp_candidates=0)
        else:info=T.two_pass_backward(rec,bank,adapter,theta,source,cfg['mode'],cfg['kind'],target)
        need(bool(torch.isfinite(theta.grad).all()) and (adapter is None or all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.parameters())),'FINITE_GRADIENTS')
        optimizer.step();step+=1
        if step%50==0:
            history.append(dict(step=step,query_id=rec['query_id'],**info))
            save_state(folder,index,adapter,theta,optimizer,step,ready,history,warm)
            print(dict(event='FUSION_TRAIN_UPDATE',config=index,step=step,loss=info['loss'],seconds=time.monotonic()-started),flush=True)
    save_state(folder,index,adapter,theta,optimizer,step,ready,history,warm)
    final_path=folder/'final_model.pt'
    if not final_path.exists():
        save(final_path,dict(authority=bind(AUTH),config=cfg,step=step,theta=theta.detach().cpu(),
            adapter=None if adapter is None else {k:v.detach().cpu() for k,v in adapter.state_dict().items()},
            warm_head_hex=warm,ready=bind(K.OUT/'ready.json')))
    else:
        frozen=torch.load(final_path,map_location='cpu',weights_only=True)
        need(frozen['authority']==bind(AUTH) and frozen['config']==cfg and frozen['step']==STEPS and torch.equal(frozen['theta'],theta.detach().cpu()),'FINAL_MODEL_RESUME')
        if adapter is not None:need(all(torch.equal(frozen['adapter'][k],v.detach().cpu()) for k,v in adapter.state_dict().items()),'FINAL_ADAPTER_RESUME')
    model_binding=bind(final_path)
    def store_encoded(key):
        dest=folder/'encoded'/f'{key}.pt'
        if dest.exists():return
        with torch.no_grad():z,g=T.encode(adapter,bank(key),source,'cuda')
        save(dest,dict(source_features=ready['features'][key],model_checkpoint=model_binding,
            adapted_tokens=z.float().cpu(),gate=g.float().cpu(),dtype='FP32 analysis copy; scoring uses FP64',
            exact_reconstruction='Frozen original tokens, fixed projection, FP64 checkpoint and source-bound forward implementation'))
    for rec in held:
        ppath=folder/'predictions'/f"query{rec['execution_ordinal']:03d}.pt"
        if ppath.exists():
            pred=torch.load(ppath,map_location='cpu',weights_only=True);need(pred['model_checkpoint']==model_binding,'PREDICTION_CHECKPOINT');continue
        if time.monotonic()-started>640:print(dict(event='FUSION_PREDICT_CHUNK_SAVED',config=index),flush=True);return
        store_encoded(rec['query_image_key']);pairs=[]
        with torch.no_grad():
            for pos,pair in enumerate(rec['pairs']):
                store_encoded(pair['image_key'])
                scores,trace=T.pair_forward(rec,pos,bank,adapter,source,cfg['mode'],'cuda')
                pairs.append(dict(position=pos,physical_row=pair['physical_row'],image_key=pair['image_key'],c4=scores.cpu(),
                    trace={k:v.cpu() for k,v in trace.items()}))
            c4=torch.stack([p['c4'] for p in pairs]).cuda();x=F.differentiable_features(rec['candidate_raw_scores'],c4,rec['winner']);z=x@theta[:-1]+theta[-1]
            k=int(z.argmax());pos=rec['challenger_positions'][k] if float(z[k])>0 else rec['winner']
        save(ppath,dict(authority=bind(AUTH),config=cfg,query_id=rec['query_id'],execution_ordinal=rec['execution_ordinal'],
            model_checkpoint=model_binding,query_image_key=rec['query_image_key'],candidate_physical_rows=rec['candidate_physical_rows'],
            winner=rec['winner'],challenger_positions=rec['challenger_positions'],candidate_raw_scores=rec['candidate_raw_scores'],pairs=pairs,
            X=x.cpu(),logits=z.cpu(),selected=rec['candidate_physical_rows'][pos],heldout_label_reads=0))
        print(dict(event='FUSION_HELD_PREDICTED',config=index,ordinal=rec['execution_ordinal']),flush=True)
    predictions=[bind(folder/'predictions'/f"query{r['execution_ordinal']:03d}.pt") for r in held]
    write(folder/'payload.json',dict(status='FUSION_FOLD_PREDICTIONS_SEALED',authority=bind(AUTH),config=cfg,index=index,
        checkpoint=model_binding,steps=STEPS,train_query_ids=[r['query_id'] for r,_ in train],held_query_ids=[r['query_id'] for r in held],
        predictions=predictions,ready=bind(K.OUT/'ready.json'),heldout_label_reads=0))
    subprocess.run([sys.executable,__file__,'verify','--index',str(index)],check=True)


def verify(a,index):
    folder=OUT/f'fit{index:03d}';p=read(folder/'payload.json');need(p['authority']==bind(AUTH) and p['config']==CONFIGS[index] and p['steps']==STEPS,'FIT_SEAL')
    cp=torch.load(checked(p['checkpoint']),map_location='cpu',weights_only=True);theta=cp['theta'].numpy();maximum=0.;checks=0
    for b in p['predictions']:
        v=torch.load(checked(b),map_location='cpu',weights_only=True);need(v['model_checkpoint']==p['checkpoint'] and v['heldout_label_reads']==0,'PRED_SOURCE')
        x=v['X'].numpy();expected=v['logits'].numpy();z=(x*theta[:-1]).sum(1)+theta[-1]
        err=float(np.max(np.abs(z-expected)));need(err<2e-10,'NUMPY_ALL127_LOGITS');maximum=max(maximum,err);checks+=127
        k=int(z.argmax());pos=v['challenger_positions'][k] if z[k]>0 else v['winner'];need(v['candidate_physical_rows'][pos]==v['selected'],'NUMPY_ACTION')
        scores=torch.stack([pair['c4'] for pair in v['pairs']]);literal=F.differentiable_features(v['candidate_raw_scores'],scores,v['winner'])
        need(torch.allclose(literal,v['X'],atol=2e-12,rtol=2e-12) and scores.shape==(128,4),'FULL_C128_FEATURES')
        for pair in v['pairs']:
            contribution=pair['trace']['query_token_score_contribution'];need(abs(float(contribution.sum())-float(pair['c4'][0]))<2e-10,'CONTRIBUTION_SUM')
    write(folder/'validation.json',dict(status='FUSION_FOLD_NUMPY_READOUT_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),
        logit_checks=checks,max_numpy_error=maximum,heldout_label_reads=0,
        scope='All127 logits/actions, all128 feature/trace sums; gradients qualified against ordinary autograd separately'))
    print(dict(event='FUSION_FIT_VALIDATED',index=index,config=CONFIGS[index],held_queries=len(p['predictions'])),flush=True)


def join(a):
    validations=[];fits=[]
    for index in range(len(CONFIGS)):
        path=OUT/f'fit{index:03d}/validation.json';v=read(path);p=read(checked(v['payload']))
        need(v['status']=='FUSION_FOLD_NUMPY_READOUT_PASS' and v['authority']==p['authority']==bind(AUTH) and p['config']==CONFIGS[index],'ALL_CONFIGS_QUALIFIED')
        validations.append(bind(path));fits.append(p)
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=validations))
    roles={r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    rows={};model_names=[]
    for p in fits:
        cfg=p['config'];name=f"{cfg['source']}_{cfg['mode']}_{cfg['kind']}_s{cfg['seed']}"
        if name not in model_names:model_names.append(name)
        fs=a['fold_sources'][str(cfg['fold'])];trainroles=read(checked(fs['train_roles']))['records']
        trainids={x['identity'] for x in trainroles};components={x['component'] for x in trainroles}
        old=read(checked(fs['full_payload']));oldpred={r['query_id']:r for r in old['predictions']}
        for b in p['predictions']:
            pred=torch.load(checked(b),map_location='cpu',weights_only=True);qid=pred['query_id'];role=roles[qid]
            need(role['identity'] not in trainids and role['component'] not in components and role['outer_fold']==cfg['fold'],'HELD_IDENTITY_COMPONENT_DISJOINT')
            row=rows.setdefault(qid,dict(query_id=qid,component=role['component'],fold=cfg['fold'],correct={},selected={},mrr={}))
            if not row['correct']:
                raw=pred['candidate_physical_rows'][pred['winner']];row['selected']['RAW']=raw;row['correct']['RAW']=labels[raw]==role['identity']
                for kind,key in (('ORIGINAL_COST1','COST1'),('ORIGINAL_CE','ALL_CE')):
                    selected=oldpred[qid]['models'][key]['selected'];row['selected'][kind]=selected;row['correct'][kind]=labels[selected]==role['identity']
            row['selected'][name]=pred['selected'];row['correct'][name]=labels[pred['selected']]==role['identity']
            logits=np.zeros(128);logits[pred['challenger_positions']]=pred['logits'].numpy()
            axis=pred['candidate_physical_rows'];order=sorted(range(128),key=lambda i:(-logits[i],i!=pred['winner'],axis[i]))
            target=[i for i,pid in enumerate(axis) if labels[pid]==role['identity']];row['target_in_C128']=bool(target)
            row['mrr'][name]=0. if not target else 1./(order.index(target[0])+1)
    rows=sorted(rows.values(),key=lambda r:r['query_id']);need(len(rows)==593 and sum(r['target_in_C128'] for r in rows)==570,'ALL593_WITH23_MISSES')
    counts={m:sum(r['correct'][m] for r in rows) for m in ('RAW','ORIGINAL_COST1','ORIGINAL_CE',*model_names)}
    need([counts[m] for m in ('RAW','ORIGINAL_COST1','ORIGINAL_CE')]==[426,481,486],'HISTORICAL_BASELINE_PARITY')
    comparisons={}
    for name in model_names:
        loss='COST1' if '_COST1_' in name else 'CE';seed=name.rsplit('_s',1)[1];mode='ROMA_WEIGHTED' if '_ROMA_WEIGHTED_' in name else 'FREE'
        bases=['RAW','ORIGINAL_'+loss]
        if not name.startswith('NO_ADAPTER'):bases.append(f'NO_ADAPTER_{mode}_{loss}_s{seed}')
        for base in bases:
            gains=sum(not r['correct'][base] and r['correct'][name] for r in rows);breaks=sum(r['correct'][base] and not r['correct'][name] for r in rows)
            groups=defaultdict(list)
            for r in rows:groups[r['component']].append(int(r['correct'][name])-int(r['correct'][base]))
            d=np.array([np.mean(v) for _,v in sorted(groups.items())]);need(len(d)==64,'GROUP64');rng=np.random.default_rng(20260922)
            interval=np.quantile(d[rng.integers(0,64,size=(10000,64))].mean(1),[.025,.975])
            comparisons[base+'__to__'+name]=dict(rescue=gains,breaks=breaks,net=gains-breaks,group_mean=float(d.mean()),group_bootstrap95=interval.tolist())
    write(OUT/'result.json',dict(status='FUSION_H593_OOF5_COMPLETE',authority=bind(AUTH),counts=counts,comparisons=comparisons,
        MRR={m:float(np.mean([r['mrr'][m] for r in rows])) for m in model_names},rows=rows,
        evidence='Opened H593 grouped OOF, natural RAW C128; original baselines separately preserved; no external GO'))
    write(OUT/'validation.json',dict(status='FUSION_JOIN_COUNTS_AND_GROUPS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),fits=len(fits),queries=593))
    print(dict(event='FUSION_H593_OOF5_COMPLETE',counts=counts),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','benchmark','fit','verify','join'));ap.add_argument('--index',type=int);args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage,args.index)
        if args.stage=='benchmark':benchmark(a)
        elif args.stage=='fit':
            fit(a,args.index)
            folder=OUT/f'fit{args.index:03d}'
            cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            array=os.environ.get('SLURM_ARRAY_JOB_ID',os.environ['SLURM_JOB_ID'])
            write(folder/'chunks'/f'{array}_{args.index}.json',dict(status='FUSION_CHUNK_NORMAL_EXIT',authority=bind(AUTH),
                config_index=args.index,step=cp['step'],array_job=array,job_id=os.environ['SLURM_JOB_ID'],
                complete=(folder/'validation.json').exists(),checkpoint=bind(folder/'checkpoint.pt')))
        elif args.stage=='verify':verify(a,args.index)
        else:join(a)
