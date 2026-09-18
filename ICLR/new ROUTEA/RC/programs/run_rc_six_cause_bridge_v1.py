#!/usr/bin/env python3
"""Fixed-token compression probes, with fold-local retrieval supervision only."""
import argparse,hashlib,io,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_paired_local_evidence_oof4_v1 as N
P=N.P
read,write,write_bytes,need,bind,checked=N.read,N.write,N.write_bytes,N.need,N.bind,N.checked
OUT=ROOT/'results/rc_six_cause_isolation_v1/bridge'
AUTH=ROOT/'registry/rc_six_cause_bridge_authority_v1_20260913.json'
REPS=('S6','FREE','POOL','MATCH','CHANNEL','CHANNEL_PERM')
READERS=('LINEAR','QUADRATIC')
LOSSES=('CE','UNIT1')
STEPS=2000

def guard(stage,index=None):
    a=read(AUTH)
    for b in a['code_sources'].values():checked(b)
    need(a['code_sources']['program']==bind(__file__),'PROGRAM_PIN')
    if stage!='preflight':need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allowed={Path(b['path']).resolve() for b in a['public_sources'].values()}
    for bundle in a['old_cache_sources']:
        allowed.update(Path(b['path']).resolve() for b in bundle.values())
    fold=index//len(REPS) if index is not None and stage in ('fit','verify') else None
    if stage.startswith('cache'):
        need(index in range(16),'CACHE_INDEX')
        for k in ('raw_shards','roma_shards'):
            allowed.update(Path(a[k][index][x]['path']).resolve() for x in ('payload','receipt','validation'))
    if fold is not None:
        need(fold in range(4),'FOLD_INDEX')
        allowed.update(Path(b['path']).resolve() for k,b in a['fold_sources'][str(fold)].items() if k!='heldout_roles')
    if stage=='join':
        for bundle in a['fold_sources'].values():allowed.update(Path(b['path']).resolve() for b in bundle.values())
        allowed.update(Path(b['path']).resolve() for b in a['join_sources'].values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(t in s for t in ('/target_join/','d1-mi','d1_mi','/grozi/','curator_roles','/reports/','rc_opened_')),'PROTECTED_READ:'+s)
        if p.name=='heldout_roles.json':need(stage=='join','HELD_LABELS_AFTER_PREDICTION_SEALS')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage!='join':
                n=p.relative_to(OUT).parts[0]
                own=n=='preflight.json' or (stage=='preflight' and n.startswith('.preflight.json.') and n.endswith('.tmp')) or (stage.startswith('cache') and n==f'cache{index:02d}') or (stage in ('fit','verify') and (n.startswith('cache') or n==f'fit{index:02d}'))
            need(own or p in allowed,'UNLISTED_RESULT:'+s)
    sys.addaudithook(audit)
    if stage!='preflight':need(read(OUT/'preflight.json')['authority']==bind(AUTH),'PREFLIGHT')
    return a

def hats_numpy(values,weights):
    t=np.clip((values+1)*8,0,16).ravel();lo=np.minimum(t.astype(np.int64),15);f=t-lo
    w=np.broadcast_to(weights,values.shape).ravel()
    return np.bincount(lo,weights=w*(1-f),minlength=17)+np.bincount(lo+1,weights=w*f,minlength=17)

def hats_torch(values,weights):
    t=((values+1)*8).clamp(0,16).flatten();lo=t.to(torch.int64).clamp_max(15);f=t-lo
    w=weights.expand_as(values).flatten()
    return torch.bincount(lo,weights=w*(1-f),minlength=17)+torch.bincount(lo+1,weights=w*f,minlength=17)

def evidence(q,r,w,independent=False):
    if independent:
        q=np.asarray(q,dtype=np.float64);r=np.asarray(r,dtype=np.float64);w=np.asarray(w,dtype=np.float64)
        q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12);r=r/np.maximum(np.linalg.norm(r,axis=1,keepdims=True),1e-12)
        sim=q@r.T;j=np.argmax(sim,axis=1);p=sim[np.arange(len(q)),j];u=w/max(w.sum(),1e-12)
        channel=np.sum((q*r[j])*u[:,None],axis=0)
        return np.concatenate(([np.dot(p,u)],hats_numpy(p,u),hats_numpy(sim,u[:,None]/len(r)),channel))
    q=F.normalize(q.to(torch.float64),dim=1);r=F.normalize(r.to(torch.float64),dim=1);u=w/w.sum().clamp_min(1e-12)
    sim=q@r.T;p,j=sim.max(1);channel=((q*r[j])*u[:,None]).sum(0)
    return torch.cat(((p*u).sum().reshape(1),hats_torch(p,u),hats_torch(sim,u[:,None]/len(r)),channel)).numpy()

def cache(index,replay=False):
    a=read(AUTH);started=time.monotonic();data=[]
    for k in ('raw_shards','roma_shards'):
        b=a[k][index];v=read(checked(b['validation']));r=read(checked(b['receipt']))
        need(v['payload']==r['payload']==b['payload'] and v['receipt']==b['receipt'],'CACHE_SOURCE_CHAIN')
        data.append(torch.load(checked(b['payload']),weights_only=True,mmap=True,map_location='cpu'))
    raw,roma=data;refs={int(k):v for k,v in raw['references'].items()}
    need(roma['token_source']=={k:a['raw_shards'][index][k] for k in ('payload','receipt','validation')},'TOKEN_CHAIN')
    features=read(checked(a['public_sources']['native_features']))
    old=read(checked(a['old_cache_sources'][index]['receipt']))
    values=[];meta=[];maxsum=0.;maxfree=0.
    for off,(q,g) in enumerate(zip(raw['records'],roma['records'],strict=True)):
        i=index*8+off;f=features[i];axis=f['candidate_physical_rows']
        need(q['query_id']==g['query_id']==f['query_id'] and q['execution_ordinal']==g['execution_ordinal']==i,'QUERY_AXIS')
        need(q['query_source_sha256']==g['query_source_sha256']==f['source_image_sha256'],'IMAGE_BINDING')
        need(q['query_tokens_sha256']==g['query_tokens_sha256']==N.tsha(q['query_tokens']),'QUERY_TOKENS')
        need(list(q['candidate_physical_rows'])==list(g['candidate_physical_rows'])==axis and len(axis)==128,'NATURAL_C128')
        need(len(g['candidates'])==128,'ALL_CANDIDATES')
        arr=[]
        for p,c in enumerate(g['candidates']):
            r=refs[axis[p]];w=c['query_visibility']
            need(c['candidate_position']==p and c['physical_row']==axis[p] and c['reference_tokens_sha256']==r['tokens_sha256']==N.tsha(r['tokens']),'REF_BINDING')
            need(hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest()==c['query_map_sha256'],'WEIGHT_BINDING')
            v=evidence(q['query_tokens'],r['tokens'],w,replay)
            need(np.isfinite(v).all() and v.shape==(163,),'FEATURE_DOMAIN')
            err=abs(v[0]-v[35:].sum());need(err<2e-12,'CHANNEL_SUM_RECONSTRUCTS_FREE');maxsum=max(maxsum,err)
            arr.append(v)
        arr=np.stack(arr)
        with np.load(checked(old['payload']),allow_pickle=False) as z:
            oldf=np.sum(z[f'free_{i:03d}']*z[f'wq_{i:03d}'],axis=1)/np.maximum(z[f'wq_{i:03d}'].sum(axis=1),1e-12)
        err=float(abs(oldf-arr[:,0]).max());need(err<2e-12,'PREVIOUS_FREE_REPLAY');maxfree=max(maxfree,err)
        values.append(arr);meta.append(dict(execution_ordinal=i,query_id=q['query_id'],candidate_physical_rows=axis,source_image_sha256=f['source_image_sha256'],query_tokens_sha256=q['query_tokens_sha256']))
        print(dict(stage='cache-replay' if replay else 'cache',index=index,completed=off+1,seconds=time.monotonic()-started),flush=True)
    values=np.stack(values);folder=OUT/f'cache{index:02d}'
    if replay:
        receipt=read(folder/'receipt.json');need(receipt['records']==meta and receipt['authority']==bind(AUTH),'RECEIPT')
        with np.load(checked(receipt['payload']),allow_pickle=False) as z:err=float(abs(z['evidence']-values).max())
        need(err<2e-11,'INDEPENDENT_FULL_EVIDENCE_REPLAY')
        write(folder/'validation.json',dict(status='SIX_CAUSE_BRIDGE_CACHE_NUMPY_PASS',authority=bind(AUTH),receipt=bind(folder/'receipt.json'),payload=receipt['payload'],max_abs_error=err,channel_sum_error=maxsum,old_free_error=maxfree,label_reads=0))
    else:
        s=io.BytesIO();np.savez(s,evidence=values);write_bytes(folder/'payload.npz',s.getvalue())
        write(folder/'receipt.json',dict(status='SIX_CAUSE_BRIDGE_CACHE_SEALED',authority=bind(AUTH),payload=bind(folder/'payload.npz'),records=meta,channel_sum_error=maxsum,old_free_error=maxfree))
        subprocess.run([sys.executable,__file__,'cache-replay','--index',str(index)],check=True)

def representation(rows,rep):
    values=[]
    for s in range(16):
        d=OUT/f'cache{s:02d}';v=read(d/'validation.json');r=read(checked(v['receipt']))
        need(v['status']=='SIX_CAUSE_BRIDGE_CACHE_NUMPY_PASS' and v['payload']==r['payload'] and v['authority']==bind(AUTH),'CACHE_VALIDATED')
        with np.load(checked(v['payload']),allow_pickle=False) as z:values.append(z['evidence'].copy())
    v=np.concatenate(values);xs=[]
    for i,r in enumerate(rows):
        g=v[i].copy();w=r['base_winner_position'];cs=r['challenger_positions']
        d=g-g[w];x=r['X'].numpy()
        if rep=='FREE':extra=d[:,:1]
        elif rep=='POOL':extra=d[:,:18]
        elif rep=='MATCH':extra=np.concatenate((d[:,:1],d[:,18:35]),1)
        elif rep in ('CHANNEL','CHANNEL_PERM'):
            ch=g[:,35:]
            if rep=='CHANNEL_PERM':
                shift=1+int(hashlib.sha256(('SIX_CAUSE_CHANNEL_PERM|'+r['query_id']).encode()).hexdigest()[:8],16)%127
                ch=np.roll(ch,shift,axis=0)
            extra=np.concatenate((d[:,:1],ch-ch[w]),1)
        else:extra=np.empty((128,0))
        xs.append(np.concatenate((x,extra[cs]),axis=1))
    return np.stack(xs)

def design(x,train,reader):
    mean,scale=N.R.standardize_fit(x[train].reshape(-1,x.shape[-1]));x=(x-mean)/scale
    if reader=='QUADRATIC':
        square=x*x;sm,ss=N.R.standardize_fit(square[train].reshape(-1,x.shape[-1]));x=np.concatenate((x,(square-sm)/ss),2)
    else:sm,ss=np.empty(0),np.empty(0)
    x=np.concatenate((np.ones((*x.shape[:2],1)),x),2)
    return torch.from_numpy(x),dict(mean=mean.tolist(),scale=scale.tolist(),square_mean=sm.tolist(),square_scale=ss.tolist(),train_ordinals=train)

def loss(z,y,kind):
    if kind=='CE':return N.R.objective(z,y)
    allz=torch.cat((z.new_zeros((len(z),1)),z),1);r=torch.arange(len(z));t=allz[r,y]
    wrong=z.clone();positive=y>0;wrong[r[positive],y[positive]-1]=-torch.inf;maximum=wrong.max(1).values
    # Original FULL action surrogate with protection cost exactly 1.
    per=torch.where(y==0,F.softplus(maximum),F.softplus(-t)+F.softplus(maximum))
    return per.mean()

def optimize(d,b,y,kind):
    theta=torch.nn.Parameter(torch.zeros(d.shape[-1],dtype=torch.float64));opt=torch.optim.AdamW([theta],lr=.03,weight_decay=.001);trace=[]
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True);l=loss(b+d@theta,y,kind);l.backward();opt.step()
        if step in (0,499,999,1999):trace.append(dict(step=step+1,preupdate_loss=float(l.detach())))
    with torch.no_grad():z=b+d@theta
    return theta.detach(),dict(trace=trace,final_loss=float(loss(z,y,kind)),final_CE=float(loss(z,y,'CE')),train_correct=int((torch.cat((z.new_zeros((len(z),1)),z),1).argmax(1)==y).sum()),train_queries=len(y))

def fit(index,replay=False):
    fold,ri=divmod(index,len(REPS));rep=REPS[ri];rows,train,held,_,base,y,_,_=N.load_fold(fold)
    x=representation(rows,rep);folder=OUT/f'fit{index:02d}'
    if replay:
        saved=read(folder/'payload.json');need(saved['authority']==bind(AUTH) and saved['train_ordinals']==train and saved['heldout_ordinals']==held,'PAYLOAD')
    models={};predictions={};maxerr=0.;checks=0
    for reader in READERS:
        d,transform=design(x,train,reader)
        for objective in LOSSES:
            name=rep+'_'+reader+'_'+objective
            theta,info=optimize(d[train],base[train],y,objective)
            z=base+d@theta
            info.update(theta_hex=[float(v).hex() for v in theta],transform=transform,parameter_count=len(theta))
            if replay:
                need(saved['models'][name]==info,'FRESH_TRAIN_REPLAY')
                independent=base[held].numpy()+np.einsum('qck,k->qc',d[held].numpy(),theta.numpy())
                want=np.asarray([[float.fromhex(v) for v in p['models'][name]['logits_hex']] for p in saved['predictions']])
                err=float(abs(independent-want).max());need(err<2e-9,'INDEPENDENT_ALL_LOGITS');maxerr=max(maxerr,err);checks+=len(held)*127
                for j,i in enumerate(held):
                    sc=independent[j];k=int(sc.argmax());pos=rows[i]['challenger_positions'][k] if sc[k]>0 else rows[i]['base_winner_position']
                    need(rows[i]['candidate_physical_rows'][pos]==saved['predictions'][j]['models'][name]['selected'],'INDEPENDENT_ACTION')
            else:
                models[name]=info;predictions[name]=z[held].numpy()
            print(dict(stage='verify' if replay else 'fit',index=index,model=name,training=info['final_loss'],train_correct=info['train_correct']),flush=True)
    if replay:
        write(folder/'validation.json',dict(status='SIX_CAUSE_BRIDGE_FRESH_FIT_AND_NUMPY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),all_logit_checks=checks,max_abs_error=maxerr,heldout_label_reads=0))
    else:
        ps=[]
        for j,i in enumerate(held):
            r=rows[i];ms={}
            for m,z in predictions.items():
                scores=z[j];k=int(scores.argmax());pos=r['challenger_positions'][k] if scores[k]>0 else r['base_winner_position']
                ms[m]=dict(logits_hex=[float(v).hex() for v in scores],selected=r['candidate_physical_rows'][pos])
            ps.append(dict(query_id=r['query_id'],execution_ordinal=i,models=ms))
        write(folder/'payload.json',dict(status='SIX_CAUSE_BRIDGE_FOLD_REP_SEALED',authority=bind(AUTH),fold=fold,representation=rep,models=models,predictions=ps,train_ordinals=train,heldout_ordinals=held,heldout_label_reads=0))
        subprocess.run([sys.executable,__file__,'verify','--index',str(index)],check=True)

def join():
    import validate_rc_query_content_routing_oof4_v1 as V
    a=read(AUTH);payloads=[];validations=[]
    for i in range(24):
        f=OUT/f'fit{i:02d}';v=read(f/'validation.json');p=read(checked(v['payload']))
        need(v['status']=='SIX_CAUSE_BRIDGE_FRESH_FIT_AND_NUMPY_PASS' and v['authority']==p['authority']==bind(AUTH),'ALL_FITS_VALIDATED');payloads.append(p);validations.append(bind(f/'validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=validations,all_fits=24))
    old=read(checked(a['join_sources']['old_result']));labels,_=P.gallery_labels();features=read(checked(a['public_sources']['native_features']))
    rows={r['execution_ordinal']:dict(query_id=r['query_id'],original_query_id=r['original_query_id'],execution_ordinal=r['execution_ordinal'],fold=r['fold'],group=r['group'],identity=r['identity'],correct={k:r['correct'][k] for k in ('RAW','BASE7','GLOBAL7')},selected={k:r['selected_physical_rows'][k] for k in ('RAW','BASE7','GLOBAL7')},held_ce={}) for r in old['rows']}
    from scipy.special import logsumexp
    for p in payloads:
        fs=a['fold_sources'][str(p['fold'])];train=read(checked(fs['train_roles']))['records'];held=read(checked(fs['heldout_roles']))['records'];rm={r['query_id']:r for r in held}
        for key in ('identity','group','source_image_sha256','query_id'):need(not({r[key] for r in train}&{r[key] for r in held}),'FOLD_DISJOINT_'+key)
        for pred in p['predictions']:
            i=pred['execution_ordinal'];r=rows[i];f=features[i];role=rm[pred['query_id']];need(role['identity']==r['identity'] and role['group']==r['group'] and pred['query_id']==r['query_id'],'LABEL_JOIN')
            target=[k for k,g in enumerate(f['candidate_physical_rows']) if labels[g]==r['identity']];need(len(target)==1,'ONE_TARGET')
            y=0 if target[0]==f['base_winner_position'] else f['challenger_positions'].index(target[0])+1
            for m,z in pred['models'].items():
                r['selected'][m]=z['selected'];r['correct'][m]=labels[z['selected']]==r['identity'];sc=np.array([0.]+[float.fromhex(v) for v in z['logits_hex']]);r['held_ce'][m]=float(logsumexp(sc)-sc[y])
    rs=[rows[i] for i in range(128)];models=list(rs[0]['correct']);counts={m:sum(r['correct'][m] for r in rs) for m in models}
    primary='CHANNEL_LINEAR_CE';contrasts=[(primary,b) for b in ('BASE7','GLOBAL7','FREE_LINEAR_CE','CHANNEL_PERM_LINEAR_CE')]
    for rep in REPS:
        for reader in READERS:contrasts.append((rep+'_'+reader+'_UNIT1',rep+'_'+reader+'_CE'))
        for obj in LOSSES:contrasts.append((rep+'_QUADRATIC_'+obj,rep+'_LINEAR_'+obj))
    for rep in ('POOL','MATCH','CHANNEL'):
        for reader in READERS:
            for obj in LOSSES:contrasts.append((rep+'_'+reader+'_'+obj,'FREE_'+reader+'_'+obj))
    comparisons={b+'__to__'+m:V.compare(rs,m,b) for m,b in contrasts}
    screen=all(counts[primary]>counts[b] and comparisons[b+'__to__'+primary]['equal_group_difference']>0 for b in ('BASE7','GLOBAL7','FREE_LINEAR_CE','CHANNEL_PERM_LINEAR_CE'))
    result=dict(status='SIX_CAUSE_BRIDGE_TRAIN128_OOF4_COMPLETE',authority=bind(AUTH),counts=counts,rows=rs,comparisons=comparisons,fit_diagnostics=[dict(fold=p['fold'],representation=p['representation'],models=p['models']) for p in payloads],evidence_level='previously opened TRAIN128, four source/identity-group folds; mechanism diagnostic',candidate_source='frozen RAW natural C128',action='127 challenger; max>0 SWITCH else HOLD',primary=primary,primary_observed_net_screen=screen,HYP_GO_claimed=False,limits=['A negative finite reader does not prove absence of token identity information','No untouched evaluation; multiple diagnostic arms are not independent confirmations','Representation capacity differs and is reported','Unchanged scores can coexist with recoverable but unlearned information'])
    write(OUT/'result.json',result)
    need(all(counts[m]==int(np.asarray([r['correct'][m] for r in rs],dtype=int).sum()) for m in models),'INDEPENDENT_COUNTS')
    write(OUT/'validation.json',dict(status='SIX_CAUSE_BRIDGE_COUNTS_AND_FOLDS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),all_fits=24,prediction_validation_sources=validations,old_models_preserved=['RAW','BASE7','GLOBAL7'],EVAL_reads=0))
    print(dict(counts=counts,primary=primary,primary_observed_net_screen=result['primary_observed_net_screen']),flush=True)

def preflight():
    rng=np.random.default_rng(17);q=torch.from_numpy(rng.normal(size=(9,128)).astype(np.float16));r=torch.from_numpy(rng.normal(size=(13,128)).astype(np.float16));w=torch.from_numpy(rng.uniform(size=9))
    a=evidence(q,r,w);b=evidence(q,r,w,True);need(np.max(abs(a-b))<2e-12 and abs(a[0]-a[35:].sum())<2e-12,'EVIDENCE_BRIDGE_PREFLIGHT')
    need(abs(a[1:18].sum()-1)<2e-12 and abs(a[18:35].sum()-1)<2e-12,'HISTOGRAM_MASS')
    z=torch.tensor([[1.,-2.],[-1.,-2.],[-2.,1.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([1,0,2]);expected=torch.stack((F.softplus(-z[0,0])+F.softplus(z[0,1]),F.softplus(z[1].max()),F.softplus(-z[2,1])+F.softplus(z[2,0]))).mean()
    need(abs(float(loss(z,y,'UNIT1')-expected))<1e-12,'UNIT1_TARGET_RAW_RISK')
    loss(z,y,'UNIT1').backward();need(torch.isfinite(z.grad).all(),'GRADIENT_FINITE')
    write(OUT/'preflight.json',dict(status='SIX_CAUSE_BRIDGE_SYNTHETIC_PASS',authority=bind(AUTH),natural_updates=0,max_evidence_error=float(abs(a-b).max()),checks=['torch_numpy_all_features','channel_sum_scalar','continuous_histogram_mass','RAW_and_challenger_UNIT1_loss']))
    print('SIX_CAUSE_BRIDGE_SYNTHETIC_PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','cache','cache-replay','fit','verify','join']);p.add_argument('--index',type=int);args=p.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(args.stage,args.index)
    if args.stage=='preflight':preflight()
    elif args.stage.startswith('cache'):cache(args.index,args.stage=='cache-replay')
    elif args.stage in ('fit','verify'):fit(args.index,args.stage=='verify')
    else:join()
