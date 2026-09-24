#!/usr/bin/env python3
"""Frozen five-fold retrieval-only learning, fresh replay and separate outcome join."""
import argparse,hashlib,json,os,sys,subprocess,uuid,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'programs'))
import materialize_rc_new_hyp593_inputs_v1 as M
OUT=M.OUT;AUTH=ROOT/'registry/rc_new_hyp593_training_authority_v1_20260911.json'
MODELS=('BASE7_SMALL128','BASE7_ALL','CONSTANT1','CONDITIONAL4')
STEPS=2000

def guard(stage,fold):
 a=M.read(AUTH)
 for b in a['public_sources'].values():M.checked(b)
 if stage not in ('prepare','preflight'):
  M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
  pf=M.read(OUT/'training_preflight_v2.json');M.need(pf['status']=='H593_BATCH_LOSS_GRADIENT_ACTION_PREFLIGHT_PASS' and pf['authority']==M.bind(AUTH),'QUALIFIED_TRAINING_PREFLIGHT')
 M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_DEADLINE')
 own=(OUT/'roles'/f'fold{fold}.json').resolve()
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower()
  M.need(not any(x in s for x in ('rc_opened_','d1-mi','d1_mi','grozi','gisc_prerecall_universe','/target_join/')),'PROTECTED_READ')
  if stage in ('fit','replay'):
   M.need('curator_roles' not in s and '/reports/' not in s,'HELDOUT_LABEL_BARRIER')
   if OUT/'roles' in p.parents:M.need(p.resolve()==own,'ONLY_CURRENT_FOLD_TRAIN_ROLES')
 sys.addaudithook(audit)
 return a

def prepare():
 roles=M.read(OUT/'metadata/curator_roles.json')['records'];split=M.read(OUT/'metadata/split_manifest.json')
 for f in split['folds']:
  ids=set(f['train_query_ids']);rs=[{k:r[k] for k in ('query_id','identity','group','component')} for r in roles if r['query_id'] in ids]
  M.write(OUT/'roles'/f"fold{f['fold']}.json",dict(fold=f['fold'],records=rs,heldout_label_fields=False))
 M.write(OUT/'roles_manifest.json',dict(folds={str(f):M.bind(OUT/'roles'/f'fold{f}.json') for f in range(5)},source_curator=M.bind(OUT/'metadata/curator_roles.json')))

def features():
 result=[];sources={}
 for kind,count in [('reuse',32),('missing',43)]:
  for i in range(count):
   folder=OUT/'features'/kind/f'shard{i:02d}';val=M.read(folder/'validation.json');rec=M.read(folder/'receipt.json')
   M.need(val['status']=='H593_FEATURE_INDEPENDENT_INDEX_REPLAY_PASS' and val['payload']==rec['payload'] and val['receipt']==M.bind(folder/'receipt.json'),'ALL_FEATURE_SHARDS_VALIDATED')
   M.need(val['authority']==rec['authority']==M.bind(ROOT/'registry/rc_new_hyp593_feature_authority_v1_20260911.json'),'FEATURE_AUTHORITY')
   p=torch.load(M.checked(rec['payload']),map_location='cpu',weights_only=True);result+=p['records'];sources[f'{kind}_{i}']=dict(payload=rec['payload'],validation=M.bind(folder/'validation.json'))
 result.sort(key=lambda r:r['execution_ordinal']);workers=M.read(OUT/'metadata/worker_manifest.json')['records']
 M.need(len(result)==len(workers)==593,'COMPLETE593')
 for r,w in zip(result,workers):M.need(all(r[k]==w[k] for k in ('query_id','execution_ordinal','source_image_sha256')),'FEATURE_WORKER_JOIN')
 return result,sources

def target_position(row,identity,labels):
 pos=[i for i,p in enumerate(row['candidate_physical_rows']) if labels[p]==identity]
 M.need(len(pos)<=1,'DEDUP_IDENTITY_C128')
 if not pos:return -2
 return -1 if pos[0]==row['winner'] else row['challenger_positions'].index(pos[0])

def batch(rows,roles,labels):
 target=[target_position(r,roles[r['query_id']]['identity'],labels) for r in rows];keep=[i for i,v in enumerate(target) if v>=-1]
 M.need(keep,'NO_RECALL_PRESENT_TRAIN_ROWS')
 selected=[rows[i] for i in keep]
 return dict(X=torch.stack([r['modes']['REAL']['X'] for r in selected]),context=torch.stack([r['modes']['REAL']['context'] for r in selected]),dF=torch.stack([r['modes']['REAL']['dF'] for r in selected]),y=torch.tensor([target[i] for i in keep]),target_absent_count=len(rows)-len(keep))

def loss(z,y):
 israw=y.eq(-1);ii=torch.arange(len(y));idx=y.clamp_min(0);wrong=z.clone();wrong[ii[~israw],idx[~israw]]=-torch.inf
 return torch.where(israw,4*F.softplus(torch.amax(z,dim=1)),F.softplus(-z[ii,idx])+4*F.softplus(torch.amax(wrong,dim=1))).mean()

def fit_base(data,steps=STEPS):
 torch.manual_seed(17);theta=torch.nn.Parameter(torch.zeros(7,dtype=torch.float64));opt=torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
 for step in range(steps):
  opt.zero_grad();z=data['X']@theta[:6]+theta[6];l=loss(z,data['y']);M.need(bool(torch.isfinite(l)),'FINITE_BASE_LOSS');l.backward();M.need(bool(torch.isfinite(theta.grad).all()),'FINITE_BASE_GRAD');opt.step()
 return theta.detach(),float(l.detach())

def fit_correction(data,theta,count,steps=STEPS):
 beta=torch.nn.Parameter(torch.zeros(count,dtype=torch.float64));opt=torch.optim.AdamW([beta],lr=.03,weight_decay=.001);base=(data['X']@theta[:6]+theta[6]).detach()
 for step in range(steps):
  opt.zero_grad();z=base+F.softplus(data['context'][:,:,:count]@beta)*data['dF'];l=loss(z,data['y']);M.need(bool(torch.isfinite(l)),'FINITE_CORRECTION_LOSS');l.backward();M.need(bool(torch.isfinite(beta.grad).all()),'FINITE_CORRECTION_GRAD');opt.step()
 return beta.detach(),float(l.detach())

def small_subset(rows,roles):
 groups=defaultdict(list)
 for r in rows:groups[roles[r['query_id']]['component']].append(r)
 key=lambda s:hashlib.sha256(('H593_SMALL128_V1|'+s).encode()).hexdigest()
 for rs in groups.values():rs.sort(key=lambda r:key(r['query_id']))
 ordered=sorted(groups,key=key);out=[]
 for level in range(max(map(len,groups.values()))):
  for g in ordered:
   if level<len(groups[g]):out.append(groups[g][level])
 return out[:128]

def predict(params,rows):
 predictions=[]
 for r in rows:
  record={k:r[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')};record['models']={}
  for model,par in params.items():
   theta=par['theta'];record['models'][model]={}
   for mode in ('REAL','CBIND'):
    d=r['modes'][mode];z=d['X']@theta[:6]+theta[6]
    if 'beta' in par:z=z+F.softplus(d['context'][:,:len(par['beta'])]@par['beta'])*d['dF']
    j=int(torch.argmax(z));winner=r['challenger_positions'][j] if float(z[j])>0 else r['winner'];record['models'][model][mode]=dict(selected_position=winner,logits=z)
  predictions.append(record)
 return predictions

def compute(fold):
 rows,src=features();fm=M.read(OUT/'metadata/split_manifest.json')['folds'][fold];role_index=M.read(OUT/'roles_manifest.json');roles=M.read(M.checked(role_index['folds'][str(fold)]))['records'];roles={r['query_id']:r for r in roles}
 train=[r for r in rows if r['query_id'] in set(fm['train_query_ids'])];test=[r for r in rows if r['query_id'] in set(fm['heldout_query_ids'])];M.need(set(roles)=={r['query_id'] for r in train},'EXACT_FOLD_TRAIN_LABELS')
 sys.path.insert(0,str(ROOT/'src'))
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities
 small=small_subset(train,roles);all_data=batch(train,roles,labels);small_data=batch(small,roles,labels)
 small_theta,small_loss=fit_base(small_data);theta,base_loss=fit_base(all_data);params={'BASE7_SMALL128':dict(theta=small_theta),'BASE7_ALL':dict(theta=theta)};losses={'BASE7_SMALL128':small_loss,'BASE7_ALL':base_loss}
 for name,n in [('CONSTANT1',1),('CONDITIONAL4',4)]:
  beta,l=fit_correction(all_data,theta,n);params[name]=dict(theta=theta,beta=beta);losses[name]=l
 return dict(fold=fold,parameters=params,predictions=predict(params,test),feature_sources=src,train_role=M.bind(OUT/'roles'/f'fold{fold}.json'),small_query_ids=[r['query_id'] for r in small],train_count=len(train),heldout_count=len(test),target_absent_train=all_data['target_absent_count'],target_absent_small=small_data['target_absent_count'],last_preupdate_losses=losses,steps=STEPS,authority=M.bind(AUTH))

def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
 if type(a)!=type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def fit(fold,replay,nonce):
 folder=OUT/'fits'/f'fold{fold}';value=compute(fold)
 if replay:
  M.need(nonce==os.environ.get('H593_FIT_NONCE'),'FRESH_FIT_REPLAY');receipt=M.read(folder/'receipt.json');old=torch.load(M.checked(receipt['payload']),map_location='cpu',weights_only=True)
  M.need(same(old,value),'ALL_PARAMETERS_AND_HELDOUT_LOGITS_BIT_REPLAY')
  M.write(folder/'validation.json',dict(status='H593_FOLD_RETRAIN_AND_PREDICTION_REPLAY_PASS',payload=receipt['payload'],receipt=M.bind(folder/'receipt.json'),fresh_nonce=nonce,heldout_label_reads=0))
 else:
  folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt';M.need(not p.exists(),'IMMUTABLE_FOLD_OUTPUT');torch.save(value,p);M.write(folder/'receipt.json',dict(payload=M.bind(p),authority=M.bind(AUTH),heldout_label_reads=0))
  nonce=uuid.uuid4().hex;subprocess.run([sys.executable,__file__,'replay','--fold',str(fold),'--nonce',nonce],env=dict(os.environ,H593_FIT_NONCE=nonce),check=True)

def groupstats(rows,base,new):
 groups=defaultdict(list)
 for r in rows:groups[r['component']].append(int(r['correct'][new])-int(r['correct'][base]))
 diffs=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(20260911);boot=np.mean(diffs[rng.integers(0,len(diffs),size=(10000,len(diffs)))],axis=1)
 return dict(rescue=sum(not r['correct'][base] and r['correct'][new] for r in rows),loss=sum(r['correct'][base] and not r['correct'][new] for r in rows),equal_component_difference=float(diffs.mean()),component_bootstrap95=[float(x) for x in np.quantile(boot,[.025,.975])],positive_components=int((diffs>0).sum()),negative_components=int((diffs<0).sum()),components=len(diffs))

def join():
 folds=[];bindings=[]
 # Verify EVERY prediction seal and fresh replay before opening query identities.
 for f in range(5):
  folder=OUT/'fits'/f'fold{f}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='H593_FOLD_RETRAIN_AND_PREDICTION_REPLAY_PASS' and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json'),'ALL_FOLD_SEALS_FIRST');folds.append(torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True));bindings.append(M.bind(folder/'validation.json'))
 roles={r['query_id']:r for r in M.read(OUT/'metadata/curator_roles.json')['records']};sys.path.insert(0,str(ROOT/'src'))
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;rows=[]
 for fold in folds:
  for p in fold['predictions']:
   role=roles[p['query_id']];M.need(role['outer_fold']==fold['fold'],'TRUE_HELDOUT_MEMBERSHIP');axis=p['candidate_physical_rows'];identity=role['identity'];rank=next((i+1 for i,physical in enumerate(p['raw_ranked_physical_rows']) if labels[physical]==identity),None);M.need(rank is not None,'GALLERY_IDENTITY_EXISTS')
   correct={'RAW':labels[axis[p['winner']]]==identity};ranks={'RAW':rank}
   for model in MODELS:
    for mode in ('REAL','CBIND'):
     key=model if mode=='REAL' else model+'_CBIND';pos=p['models'][model][mode]['selected_position'];chosen=axis[pos];correct[key]=labels[chosen]==identity
     selected_rank=p['raw_ranked_physical_rows'].index(chosen)+1
     ranks[key]=1 if correct[key] else rank+int(selected_rank>rank)
   rows.append(dict(query_id=p['query_id'],original_query_id=role['original_query_id'],fold=fold['fold'],component=role['component'],group=role['group'],target_in_C128=rank<=128,correct=correct,ranks=ranks))
 M.need(len(rows)==len({r['query_id'] for r in rows})==593,'EXACT593_JOIN')
 scores={m:dict(correct=sum(r['correct'][m] for r in rows),MRR=sum(1/r['ranks'][m] for r in rows)/593) for m in rows[0]['correct']}
 comparisons={f'{b}__to__{n}':groupstats(rows,b,n) for b,n in [('BASE7_SMALL128','BASE7_ALL'),('BASE7_ALL','CONDITIONAL4'),('CONSTANT1','CONDITIONAL4'),('RAW','BASE7_ALL')]}
 result=dict(status='H593_NEW_DEVELOPMENT_OOF5_COMPLETE',population=593,folds=5,identity_count=68,component_count=64,recall_C128=sum(r['target_in_C128'] for r in rows),scores=scores,comparisons=comparisons,rows=rows,fold_validation_sources=bindings,old_EVAL128_99_comparable=False,external_confirmation=False,automatic_deployment_change=False)
 M.write(OUT/'result.json',result);print(json.dumps({k:result[k] for k in ('status','recall_C128','scores','comparisons')},indent=2))

def preflight():
 # Batched original loss equals scalar per-query loss and gradients, including HOLD rows.
 torch.manual_seed(17);z=torch.randn(4,127,dtype=torch.float64,requires_grad=True);y=torch.tensor([-1,0,31,126]);a=loss(z,y)
 ls=[]
 for v,t in zip(z,y):
  if t==-1:ls.append(4*F.softplus(v.max()))
  else:
   mask=torch.ones(127,dtype=torch.bool);mask[t]=False;ls.append(F.softplus(-v[t])+4*F.softplus(v[mask].max()))
 b=torch.stack(ls).mean();M.need(torch.equal(a,b),'BATCH_LOSS_PARITY');ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0];M.need(torch.equal(ga,gb),'BATCH_GRADIENT_PARITY')
 # Scalar max distributes gradients across ties; torch.max(dim) would not.
 for fill in (0.,1.):
  z=torch.full((4,127),fill,dtype=torch.float64,requires_grad=True);a=loss(z,y);ls=[]
  for v,targ in zip(z,y):
   if targ==-1:ls.append(4*F.softplus(v.max()))
   else:
    mask=torch.ones(127,dtype=torch.bool);mask[targ]=False;ls.append(F.softplus(-v[targ])+4*F.softplus(v[mask].max()))
  b=torch.stack(ls).mean();M.need(torch.equal(a,b),'TIED_LOSS_PARITY');ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0];M.need(torch.equal(ga,gb),'TIED_MAX_GRADIENT_PARITY')
 # Physical-order tie keeps first challenger; nonpositive logit preserves RAW.
 r=dict(query_id='synthetic',execution_ordinal=0,candidate_physical_rows=[0,1,2],raw_ranked_physical_rows=[1,0,2],winner=1,challenger_positions=[0,2],modes={mode:dict(X=torch.zeros((2,6),dtype=torch.float64),context=torch.zeros((2,4),dtype=torch.float64),dF=torch.zeros(2,dtype=torch.float64)) for mode in ('REAL','CBIND')})
 t=torch.zeros(7,dtype=torch.float64);M.need(predict({'x':dict(theta=t)},[r])[0]['models']['x']['REAL']['selected_position']==1,'HOLD_ZERO');t[-1]=1;M.need(predict({'x':dict(theta=t)},[r])[0]['models']['x']['REAL']['selected_position']==0,'PHYSICAL_TIE')
 M.need(target_position(r,'absent',['a','b','c'])==-2,'MISSING_TARGET_NOT_INSERTED');M.need(target_position(r,'b',['a','b','c'])==-1,'RAW_TARGET_POSITION')
 M.write(OUT/'training_preflight_v2.json',dict(status='H593_BATCH_LOSS_GRADIENT_ACTION_PREFLIGHT_PASS',natural_training_steps=0,heldout_label_reads=0,authority=M.bind(AUTH)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','fit','replay','join']);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--nonce');a=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(a.stage,a.fold)
 if a.stage=='prepare':prepare()
 elif a.stage=='preflight':preflight()
 elif a.stage=='join':join()
 else:fit(a.fold,a.stage=='replay',a.nonce)
