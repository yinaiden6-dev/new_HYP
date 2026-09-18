#!/usr/bin/env python3
"""Fresh same-training-scope risk comparison on the original EVAL32/EVAL128 panels."""
import argparse,json,os,sys,subprocess,uuid
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P
import run_rc_h593_group_risk_strong_base_v1 as G
OUT=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
AUTH=ROOT/'registry/rc_fixed_panels_train269_group_risk_authority_v1_20260911.json'
SEAL=ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
EXCLUSION=ROOT/'registry/rc_eval128_identity_group_exclusion_v1_20260910.json'
OLD128=ROOT/'results/rc_original7_expanded_eval128_manifest_v1/curator_roles.json'
MODEL_NAMES=('ORIGINAL7','IMAGE269','GROUP269')

def prepare():
 c=M.read(P.OUT/'metadata/curator_roles.json')['records'];ex=M.read(EXCLUSION);old128=M.read(OLD128)['records']
 bysha={r['source_image_sha256']:r for r in c};byid={r['original_query_id']:r for r in c};M.need(len(bysha)==len(c)==593,'UNIQUE_IMAGES593')
 ti=set(ex['training_union_identities']);tg=set(ex['training_union_groups']);train=[r for r in c if r['identity'] in ti and r['group'] in tg]
 e32=[byid[q] for q in ex['split_sets']['FULL_EVAL32']['query_ids']];e128=[bysha[r['source_image_sha256']] for r in old128]
 M.need(len(train)==269 and len({r['identity'] for r in train})==32 and len({r['group'] for r in train})==32,'TRAIN269_ORIGINAL32_IDENTITIES')
 M.need(len(e32)==32 and len(e128)==128,'EXACT_OLD_PANELS')
 for a,b in zip(e128,old128):M.need(all(a[k]==b[k] for k in ('source_image_sha256','identity','group')),'OLD128_SHA_IDENTITY_GROUP_JOIN')
 panels={'EVAL32':e32,'EVAL128':e128};checks={}
 for n,rs in panels.items():
  checks[n]={}
  for k in ('query_id','source_image_sha256','identity','group','component'):
   count=len({r[k] for r in train}&{r[k] for r in rs});M.need(count==0,'TRAIN_TEST_DISJOINT_'+n+'_'+k);checks[n][k]=count
 M.need(not ({r['query_id'] for r in e32}&{r['query_id'] for r in e128}),'PANELS_SEPARATE')
 safe=('query_id','original_query_id','execution_ordinal','source_image_sha256','identity','group','component')
 M.write(OUT/'train_roles.json',dict(records=[{k:r[k] for k in safe} for r in train],evaluation_labels_included=False))
 evalrows=[{k:r[k] for k in safe} for rs in panels.values() for r in rs]
 M.write(OUT/'eval_curator_roles.json',dict(records=evalrows,historical_panels_already_opened=True))
 pm=dict(train_query_ids=[r['query_id'] for r in train],panels={n:[r['query_id'] for r in rs] for n,rs in panels.items()},selection_used_outcomes=False)
 M.write(OUT/'panel_manifest.json',pm)
 M.write(OUT/'metadata_validation.json',dict(status='FIXED269_TRAIN_TEST_IMAGE_IDENTITY_GROUP_EXCLUSION_PASS',counts=dict(train=269,train_identities=32,train_groups=32,EVAL32=32,EVAL128=128),overlaps=checks,sources={p.name:M.bind(p) for p in (EXCLUSION,OLD128,P.OUT/'metadata/curator_roles.json')},train_role=M.bind(OUT/'train_roles.json'),eval_role=M.bind(OUT/'eval_curator_roles.json'),panels=M.bind(OUT/'panel_manifest.json'),join128='source_image_sha256 then verify identity and group'))
 print('FIXED269_TRAIN_TEST_IMAGE_IDENTITY_GROUP_EXCLUSION_PASS',flush=True)

def guard(stage):
 M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_CUTOFF')
 if stage=='prepare':return
 a=M.read(AUTH)
 for b in a['sources'].values():M.checked(b)
 if stage=='join':
  for b in a['evaluation_sources'].values():M.checked(b)
 M.need(M.read(OUT/'metadata_validation.json')['status']=='FIXED269_TRAIN_TEST_IMAGE_IDENTITY_GROUP_EXCLUSION_PASS','METADATA_VALIDATION')
 if stage not in ('preflight',):
  M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');pf=M.read(OUT/'preflight.json');M.need(pf['authority']==M.bind(AUTH) and pf['status']=='FIXED_PANEL_GROUP_RISK_PREFLIGHT_PASS','PREFLIGHT')
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower()
  M.need(not any(v in s for v in ('rc_opened_','d1-mi','d1_mi','grozi','gisc_prerecall_universe','/target_join/')),'PROTECTED_INPUT')
  if stage in ('fit','replay','preflight'):
   M.need('curator_roles' not in s and '/reports/' not in s,'NO_EVAL_LABELS_OR_REPORTS')
   for root in (P.OUT/'fits',P.OUT/'roles',G.OUT):M.need(root not in p.parents,'NO_593_TRAINED_MODELS_OR_FOLD_LABELS')
   M.need(p.name!='result.json','NO_OUTCOME_RESULTS')
 sys.addaudithook(audit)

def original_theta():
 seal=M.read(SEAL);M.need(seal['parameter_sha256']=='ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263','ORIGINAL_EC7_ONLY')
 return torch.tensor([float.fromhex(x) for x in seal['weight_binary64']]+[float.fromhex(seal['bias_binary64'])],dtype=torch.float64)

def compute():
 rows,sources=P.features();split=M.read(OUT/'panel_manifest.json');roles={r['query_id']:r for r in M.read(OUT/'train_roles.json')['records']}
 train=[r for r in rows if r['query_id'] in set(split['train_query_ids'])];testids={q for ids in split['panels'].values() for q in ids};test=[r for r in rows if r['query_id'] in testids]
 M.need(set(roles)=={r['query_id'] for r in train} and len(train)==269 and len(test)==160,'EXACT269_160')
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;data=P.batch(train,roles,labels);eligible=[r for r in train if P.target_position(r,roles[r['query_id']]['identity'],labels)>=-1]
 weights,counts=G.group_weights([roles[r['query_id']]['component'] for r in eligible]);M.need(len(weights)==len(data['y']),'TRAIN_AXIS')
 params={'ORIGINAL7':dict(theta=original_theta())}
 theta,loss=P.fit_base(data);params['IMAGE269']=dict(theta=theta,last_preupdate_loss=loss)
 params['GROUP269']=G.fit_group_base(data,weights)
 return dict(parameters=params,predictions=P.predict(params,test),feature_sources=sources,train_role=M.bind(OUT/'train_roles.json'),panels=M.bind(OUT/'panel_manifest.json'),train_images=len(train),loss_eligible_images=len(eligible),target_absent_train=data['target_absent_count'],effective_groups=len(counts),effective_group_image_counts=counts,group_weights=weights,steps=2000,seed=17,from_zero=True,prior_593_model_reads=0,heldout_label_reads=0,authority=M.bind(AUTH))

def fit(replay,nonce):
 value=compute();folder=OUT/'fits'
 if replay:
  M.need(nonce==os.environ.get('FIXED_PANEL_REPLAY_NONCE'),'FRESH_PROCESS_REPLAY');r=M.read(folder/'receipt.json');old=torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True);M.need(P.same(old,value),'ALL_TRAINING_AND_PREJOIN_PREDICTIONS_BIT_REPLAY')
  M.write(folder/'validation.json',dict(status='FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS',payload=r['payload'],receipt=M.bind(folder/'receipt.json'),authority=M.bind(AUTH),heldout_label_reads=0,prior_593_model_reads=0,nonce=nonce));print('FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS',flush=True)
 else:
  folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt'
  with p.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
  M.write(folder/'receipt.json',dict(payload=M.bind(p),authority=M.bind(AUTH),heldout_label_reads=0))
  nonce=uuid.uuid4().hex;subprocess.run([sys.executable,__file__,'replay','--nonce',nonce],env=dict(os.environ,FIXED_PANEL_REPLAY_NONCE=nonce),check=True)

def join():
 folder=OUT/'fits';v=M.read(folder/'validation.json');receipt=M.read(folder/'receipt.json');M.need(v['status']=='FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS' and v['payload']==receipt['payload'] and v['receipt']==M.bind(folder/'receipt.json') and v['authority']==M.bind(AUTH),'PREDICTION_SEAL_BEFORE_LABELS')
 payload=torch.load(M.checked(receipt['payload']),map_location='cpu',weights_only=True)
 roles={r['query_id']:r for r in M.read(OUT/'eval_curator_roles.json')['records']};split=M.read(OUT/'panel_manifest.json');pred={r['query_id']:r for r in payload['predictions']}
 M.need(set(roles)==set(pred),'EXACT_EVAL_JOIN')
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;panels={}
 for name,qids in split['panels'].items():
  rows=[]
  for q in qids:
   r=pred[q];role=roles[q];axis=r['candidate_physical_rows'];rank=next(i+1 for i,p in enumerate(r['raw_ranked_physical_rows']) if labels[p]==role['identity']);chosen={'RAW':axis[r['winner']]}
   for model in MODEL_NAMES:
    for mode in ('REAL','CBIND'):chosen[model if mode=='REAL' else model+'_CBIND']=axis[r['models'][model][mode]['selected_position']]
   correct={k:labels[p]==role['identity'] for k,p in chosen.items()};ranks={k:(rank if k=='RAW' else 1 if correct[k] else rank+int(r['raw_ranked_physical_rows'].index(p)+1>rank)) for k,p in chosen.items()}
   rows.append(dict(query_id=q,original_query_id=role['original_query_id'],component=role['component'],group=role['group'],target_in_C128=rank<=128,correct=correct,ranks=ranks,selected_physical_rows=chosen))
  scores={k:dict(correct=sum(r['correct'][k] for r in rows),MRR=sum(1/r['ranks'][k] for r in rows)/len(rows)) for k in rows[0]['correct']}
  comparisons={b+'__to__'+n:P.groupstats(rows,b,n) for b,n in [('RAW','ORIGINAL7'),('ORIGINAL7','IMAGE269'),('ORIGINAL7','GROUP269'),('IMAGE269','GROUP269')]}
  panels[name]=dict(population=len(rows),recall_C128=sum(r['target_in_C128'] for r in rows),scores=scores,comparisons=comparisons,rows=rows)
 result=dict(status='FIXED_PANELS_TRAIN269_GROUP_RISK_DEVELOPMENT_COMPLETE',panels=panels,train_images=269,train_groups=32,loss_eligible_images=payload['loss_eligible_images'],target_absent_train=payload['target_absent_train'],authority=M.bind(AUTH),fit_validation=M.bind(folder/'validation.json'),historically_opened_development=True,external_confirmation=False,deployment_changed=False,primary='GROUP269 versus ORIGINAL7 on each fixed panel',matched_training_control='GROUP269 versus IMAGE269',old_ec7_training_objective='original PAIR64 plus FULL32, not identical to the two new full-only objectives')
 M.write(OUT/'result.json',result);print(json.dumps({n:{'scores':p['scores'],'comparisons':p['comparisons']} for n,p in panels.items()},indent=2),flush=True)

def preflight():
 old=M.read(P.OUT/'training_preflight_v2.json');M.need(old['status']=='H593_BATCH_LOSS_GRADIENT_ACTION_PREFLIGHT_PASS','REUSED_LOSS_ACTION_PREFLIGHT')
 torch.manual_seed(17);z=torch.randn(4,127,dtype=torch.float64,requires_grad=True);y=torch.tensor([-1,0,31,126]);w,_=G.group_weights(['a','a','b','b']);a=P.loss(z,y);b=(w*G.per_query_loss(z,y)).sum();M.need(torch.equal(a,b),'EQUAL_GROUPS_ORIGINAL_LOSS');M.need(torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z)[0]),'EQUAL_GROUPS_ORIGINAL_GRADIENT')
 t=original_theta();M.need(t.shape==(7,) and bool(torch.isfinite(t).all()),'EC7')
 p=OUT/'serialization_preflight.pt'
 with p.open('xb') as f:torch.save(dict(theta=t),f)
 M.need(torch.equal(torch.load(p,map_location='cpu',weights_only=True)['theta'],t),'ACTUAL_FILE_STREAM_SERIALIZATION')
 M.write(OUT/'preflight.json',dict(status='FIXED_PANEL_GROUP_RISK_PREFLIGHT_PASS',authority=M.bind(AUTH),natural_training_updates=0,reused_loss_gradient_action_preflight=M.bind(P.OUT/'training_preflight_v2.json')))
 print('FIXED_PANEL_GROUP_RISK_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','fit','replay','join']);ap.add_argument('--nonce');a=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(a.stage)
 if a.stage=='prepare':prepare()
 elif a.stage=='preflight':preflight()
 elif a.stage=='join':join()
 else:fit(a.stage=='replay',a.nonce)
