#!/usr/bin/env python3
"""Matched-parameter endpoint-sign readout on frozen fold-local BASE7_ALL."""
import argparse,json,os,sys,subprocess,uuid,hashlib
from pathlib import Path
from collections import defaultdict
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC
OUT=ROOT/'results/rc_h593_endpoint_competition_v1'
AUTH=ROOT/'registry/rc_h593_endpoint_competition_training_authority_v1_20260911.json'
CACHE_AUTH=ROOT/'registry/rc_h593_endpoint_competition_cache_authority_v1_20260911.json'
MODELS=('BASE7','RELATIVE1','RELATIVE2','ENDPOINT2');MODES=('REAL','J_BIND','CBIND');STEPS=2000

def guard(stage,fold):
 a=M.read(AUTH)
 for b in a['public_sources'].values():M.checked(b)
 M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_DEADLINE')
 if stage!='preflight':
  M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');pf=M.read(OUT/'training_preflight.json');M.need(pf['status']=='ENDPOINT_SIGN_CONTROL_AND_ACTION_PREFLIGHT_PASS' and pf['authority']==M.bind(AUTH),'PREFLIGHT_BINDING')
 ownrole=(P.OUT/'roles'/f'fold{fold}.json').resolve()
 def audit(e,args):
  if e!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower();M.need(not any(x in s for x in ('rc_opened_','d1-mi','d1_mi','grozi','gisc_prerecall_universe','target_join')),'DIAGNOSTIC_AND_PROTECTED_SOURCE_BARRIER')
  M.need(not (p.parent==P.OUT and p.name in ('result.json','result_validation.json')),'PARENT_OOF_OUTCOMES_NOT_USED')
  if stage in ('fit','replay','preflight'):
   M.need('curator_roles' not in s and '/reports/' not in s,'HELDOUT_LABEL_BARRIER')
   if P.OUT/'roles' in p.parents:M.need(p.resolve()==ownrole,'CURRENT_FOLD_TRAIN_LABELS_ONLY')
   if P.OUT/'fits' in p.parents:M.need(P.OUT/'fits'/f'fold{fold}' in p.parents,'CURRENT_FOLD_PARENT_BASE_ONLY')
   if OUT/'fits' in p.parents:M.need(OUT/'fits'/f'fold{fold}' in p.parents,'CURRENT_FOLD_OUTPUT_ONLY')
 sys.addaudithook(audit)

def representation(j,winner,challengers,mode):
 donor=list(range(len(j))) if mode=='REAL' else [(i+len(j)//2)%len(j) for i in range(len(j))]
 relative=[];state=[]
 for c in challengers:
  a,b=float(j[donor[c]]),float(j[donor[winner]]);den=abs(a)+abs(b)+1e-12;relative.append((a-b)/den);state.append((a+b)/den)
 r=torch.tensor(relative,dtype=torch.float64);s=torch.tensor(state,dtype=torch.float64)
 return dict(RELATIVE1=r[:,None],RELATIVE2=torch.stack((r,torch.ones_like(r)),1),ENDPOINT2=torch.stack((r,s),1))

def load_features():
 base,base_sources=P.features();jrows={};sources={}
 for kind,n in [('reuse',32),('missing',43)]:
  for s in range(n):
   folder=OUT/'cache'/kind/f'shard{s:02d}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='H593_J_UNLABELLED_NUMPY_POSITION_REPLAY_PASS' and v['authority']==r['authority']==M.bind(CACHE_AUTH) and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json'),'J_CACHE_QUALIFIED')
   payload=torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True,mmap=True)
   for q in payload['records']:M.need(q['query_id'] not in jrows,'UNIQUE_J_QUERY');jrows[q['query_id']]=q
   sources[f'{kind}_{s}']=dict(payload=r['payload'],validation=M.bind(folder/'validation.json'))
 M.need(len(jrows)==len(base)==593,'COMPLETE593_J_JOIN')
 for r in base:
  j=jrows.pop(r['query_id']);M.need(all(r[k]==j[k] for k in ('query_id','execution_ordinal','source_image_sha256','candidate_physical_rows')),'J_ORIGINAL_FEATURE_AXIS')
  free=j['F'];df=torch.tensor([FC.symmetric(free[c],free[r['winner']]) for c in r['challenger_positions']],dtype=torch.float64);M.need(torch.equal(df,r['modes']['REAL']['dF']),'FREE_CONTENT_ORIGINAL_BITS')
  r['J']=j['J'];r['endpoint_features']={mode:representation(r['J'],r['winner'],r['challenger_positions'],mode) for mode in MODES}
 return base,dict(original=base_sources,competition=sources)

def parent_base(fold):
 folder=P.OUT/'fits'/f'fold{fold}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='H593_FOLD_RETRAIN_AND_PREDICTION_REPLAY_PASS' and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json'),'PARENT_FOLD_VALIDATION')
 payload=torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True);M.need(payload['fold']==fold and payload['authority']==M.bind(P.AUTH),'PARENT_FOLD_AUTHORITY')
 return payload,dict(payload=r['payload'],validation=M.bind(folder/'validation.json'))

def fit_gamma(base,x,y,steps=STEPS):
 gamma=torch.nn.Parameter(torch.zeros(x.shape[-1],dtype=torch.float64));opt=torch.optim.AdamW([gamma],lr=.03,weight_decay=.001);nonzero=0
 for _ in range(steps):
  opt.zero_grad();z=base+x@gamma;loss=P.loss(z,y);M.need(bool(torch.isfinite(loss)),'FINITE_LOSS');loss.backward();M.need(gamma.grad is not None and bool(torch.isfinite(gamma.grad).all()),'FINITE_GRADIENT');nonzero+=int(bool(torch.count_nonzero(gamma.grad)));opt.step();M.need(bool(torch.isfinite(gamma).all()),'FINITE_PARAMETERS')
 return dict(gamma=gamma.detach(),last_preupdate_loss=float(loss.detach()),nonzero_gradient_steps=nonzero)

def predict(theta,parameters,rows):
 out=[]
 for r in rows:
  p={k:r[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')};p['models']={}
  for model in MODELS:
   p['models'][model]={}
   for mode in MODES:
    source='CBIND' if mode=='CBIND' else 'REAL';z=r['modes'][source]['X']@theta[:6]+theta[6]
    if model!='BASE7':z=z+r['endpoint_features'][mode][model]@parameters[model]['gamma']
    k=int(torch.argmax(z));selected=r['challenger_positions'][k] if float(z[k])>0 else r['winner'];p['models'][model][mode]=dict(logits=z,selected_position=selected)
  M.need(torch.equal(p['models']['BASE7']['REAL']['logits'],p['models']['BASE7']['J_BIND']['logits']),'BASE_INVARIANT_TO_J_BIND');out.append(p)
 return out

def compute(fold):
 rows,sources=load_features();parent,pb=parent_base(fold);theta=parent['parameters']['BASE7_ALL']['theta'].clone();before=theta.clone()
 split=M.read(P.OUT/'metadata/split_manifest.json')['folds'][fold];rolemap=M.read(P.OUT/'roles_manifest.json');rolefile=M.checked(rolemap['folds'][str(fold)]);roles={r['query_id']:r for r in M.read(rolefile)['records']}
 train=[r for r in rows if r['query_id'] in set(split['train_query_ids'])];heldout=[r for r in rows if r['query_id'] in set(split['heldout_query_ids'])];M.need(set(roles)=={r['query_id'] for r in train},'EXACT_TRAIN_SCOPE')
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;targets=[P.target_position(r,roles[r['query_id']]['identity'],labels) for r in train];kept=[(r,t) for r,t in zip(train,targets) if t>=-1];M.need(kept,'NONEMPTY_TRAIN')
 base=torch.stack([r['modes']['REAL']['X'] for r,t in kept])@theta[:6]+theta[6];y=torch.tensor([t for r,t in kept]);params={}
 for model in MODELS[1:]:
  x=torch.stack([r['endpoint_features']['REAL'][model] for r,t in kept]);params[model]=fit_gamma(base.detach(),x,y)
 M.need(torch.equal(theta,before),'BASE_NEVER_UPDATED');predictions=predict(theta,params,heldout);old={r['query_id']:r for r in parent['predictions']}
 for r in predictions:
  for mode in ('REAL','CBIND'):M.need(P.same(r['models']['BASE7'][mode],old[r['query_id']]['models']['BASE7_ALL'][mode]),'BASE_PARENT_PREDICTION_BIT_PARITY')
 return dict(fold=fold,base_theta=theta,parameters=params,predictions=predictions,source_parent=pb,feature_sources=sources,training_roles=M.bind(rolefile),train_images=len(train),heldout_images=len(heldout),absent_target_train=len(train)-len(kept),steps=STEPS,authority=M.bind(AUTH))

def run_fold(fold,replay,nonce):
 folder=OUT/'fits'/f'fold{fold}';value=compute(fold)
 if replay:
  M.need(nonce==os.environ.get('ENDPOINT_FIT_NONCE'),'FRESH_FIT_REPLAY');r=M.read(folder/'receipt.json');old=torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True);M.need(P.same(old,value),'ALL_PARAMETERS_PREDICTIONS_RETRAIN_REPLAY')
  M.write(folder/'validation.json',dict(status='ENDPOINT_FOLD_RETRAIN_PREDICTION_REPLAY_PASS',payload=r['payload'],receipt=M.bind(folder/'receipt.json'),authority=M.bind(AUTH),fresh_nonce=nonce,parent_base_predictions_bit_exact=True,heldout_label_reads=0))
 else:
  folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt';M.need(not p.exists(),'IMMUTABLE_FOLD')
  with p.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
  M.write(folder/'receipt.json',dict(payload=M.bind(p),authority=M.bind(AUTH),heldout_label_reads=0))
  nonce=uuid.uuid4().hex;subprocess.run([sys.executable,__file__,'replay','--fold',str(fold),'--nonce',nonce],env=dict(os.environ,ENDPOINT_FIT_NONCE=nonce),check=True)

def join():
 payloads=[];bindings=[]
 for f in range(5):
  folder=OUT/'fits'/f'fold{f}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='ENDPOINT_FOLD_RETRAIN_PREDICTION_REPLAY_PASS' and v['authority']==r['authority']==M.bind(AUTH) and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json'),'ALL_FOLDS_SEALED_FIRST');payloads.append(torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True));bindings.append(M.bind(folder/'validation.json'))
 roles={r['query_id']:r for r in M.read(P.OUT/'metadata/curator_roles.json')['records']}
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;rows=[]
 for payload in payloads:
  for p in payload['predictions']:
   role=roles[p['query_id']];M.need(role['outer_fold']==payload['fold'],'HELDOUT_FOLD');axis=p['candidate_physical_rows'];rank=next(i+1 for i,x in enumerate(p['raw_ranked_physical_rows']) if labels[x]==role['identity']);correct={'RAW':labels[axis[p['winner']]]==role['identity']};ranks={'RAW':rank}
   for model in MODELS:
    for mode in MODES:
     key=model if mode=='REAL' else model+'_'+mode;physical=axis[p['models'][model][mode]['selected_position']];correct[key]=labels[physical]==role['identity'];srank=p['raw_ranked_physical_rows'].index(physical)+1;ranks[key]=1 if correct[key] else rank+int(srank>rank)
   rows.append(dict(query_id=p['query_id'],original_query_id=role['original_query_id'],fold=payload['fold'],group=role['group'],component=role['component'],target_in_C128=rank<=128,correct=correct,ranks=ranks))
 M.need(len(rows)==len({r['query_id'] for r in rows})==593,'FULL593_DENOMINATOR')
 scores={m:dict(correct=sum(r['correct'][m] for r in rows),MRR=sum(1/r['ranks'][m] for r in rows)/593) for m in rows[0]['correct']}
 comparisons={base+'__to__'+new:P.groupstats(rows,base,new) for base,new in [('BASE7','RELATIVE1'),('BASE7','RELATIVE2'),('BASE7','ENDPOINT2'),('RELATIVE1','ENDPOINT2'),('RELATIVE2','ENDPOINT2')]}
 rescue=[r for r in rows if r['correct']['ENDPOINT2'] and not r['correct']['BASE7']];binding=dict(real_incremental_rescues=len(rescue),J_BIND_incremental_rescues_retained=sum(r['correct']['ENDPOINT2_J_BIND'] for r in rescue),REAL_correct=scores['ENDPOINT2']['correct'],J_BIND_correct=scores['ENDPOINT2_J_BIND']['correct'])
 support=all(scores['ENDPOINT2']['correct']>scores[b]['correct'] and comparisons[b+'__to__ENDPOINT2']['equal_component_difference']>0 for b in ('BASE7','RELATIVE2'));intervals=all(comparisons[b+'__to__ENDPOINT2']['component_bootstrap95'][0]>0 for b in ('BASE7','RELATIVE2'))
 result=dict(status='H593_ENDPOINT_SIGN_OOF5_COMPLETE',recall_C128=sum(r['target_in_C128'] for r in rows),scores=scores,comparisons=comparisons,binding=binding,internal_count_and_group_support=support,group_interval_support=intervals,scientific_GO_claimed=False,external_confirmation=False,old_EVAL128_99_comparable=False,rows=rows,fold_validation_sources=bindings,authority=M.bind(AUTH),no_parent_outcome_selection=True)
 M.write(OUT/'result.json',result);print(json.dumps({k:result[k] for k in ('status','scores','comparisons','binding','internal_count_and_group_support','group_interval_support')},indent=2))

def preflight():
 a=torch.tensor([.02,.01],dtype=torch.float64);b=torch.tensor([-.01,-.02],dtype=torch.float64);x=representation(a,1,[0],'REAL');y=representation(b,1,[0],'REAL');M.need(torch.equal(x['RELATIVE2'],y['RELATIVE2']) and float(x['ENDPOINT2'][0,1])>0>float(y['ENDPOINT2'][0,1]),'SIGN_COLLAPSE_AND_RECOVERY')
 zero=representation(torch.zeros(128,dtype=torch.float64),10,[i for i in range(128) if i!=10],'REAL');M.need(torch.equal(zero['ENDPOINT2'],torch.zeros_like(zero['ENDPOINT2'])),'NULL_J_HAS_ZERO_ENDPOINT_CORRECTION')
 # Verify functional candidate permutation: keep the same physical winner and challengers.
 j=torch.tensor([.02,-.04,.01,.07],dtype=torch.float64);perm=[2,0,3,1];old=representation(j,1,[0,2,3],'REAL')['ENDPOINT2'];new=representation(j[perm],perm.index(1),[perm.index(i) for i in [0,2,3]],'REAL')['ENDPOINT2'];M.need(torch.equal(old,new),'CANDIDATE_PERMUTATION_EQUIVARIANCE')
 # Extra parameters alone must not change a frozen baseline at zero initialization.
 torch.manual_seed(17);base=torch.randn(3,127,dtype=torch.float64);features=torch.randn(3,127,2,dtype=torch.float64);M.need(torch.equal(base,base+features@torch.zeros(2,dtype=torch.float64)),'ZERO_RESIDUAL_BASE_PARITY');g=fit_gamma(base,features,torch.tensor([-1,0,126]),steps=4);M.need(g['nonzero_gradient_steps']==4,'RETRIEVAL_LABEL_GRADIENTS_REACH_ONLY_RESIDUAL')
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'training_synthetic.pt'
 with p.open('xb') as f:torch.save(dict(gamma=g['gamma'],base=base),f)
 r=torch.load(p,map_location='cpu',weights_only=True,mmap=True);M.need(torch.equal(r['gamma'],g['gamma']),'ACTUAL_SERIALIZATION')
 M.write(OUT/'training_preflight.json',dict(status='ENDPOINT_SIGN_CONTROL_AND_ACTION_PREFLIGHT_PASS',authority=M.bind(AUTH),natural_training_steps=0,synthetic_training_steps=4,matched_new_parameters=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','replay','join','preflight']);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--nonce');a=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(a.stage,a.fold)
 if a.stage=='preflight':preflight()
 elif a.stage=='join':join()
 else:run_fold(a.fold,a.stage=='replay',a.nonce)
