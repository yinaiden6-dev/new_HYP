#!/usr/bin/env python3
"""Only within-pool group weighting changes; frozen ec7 and its predictions are reused."""
import argparse,ast,hashlib,json,os,sys,subprocess,uuid
from pathlib import Path
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P
OUT=ROOT/'results/rc_original_mixed96_group_risk_v1'
PREV=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
AUTH=ROOT/'registry/rc_original_mixed96_group_risk_authority_v1_20260911.json'
LEGACY=ROOT/'programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py'
HEAD=ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
CLOSURE=ROOT/'results/rc_native7_training_objective_factorial_v1/input_closure.json'
SPLIT=ROOT/'registry/rc_shared_query_target_prior_split_qualification_v1_20260909.json'
PAIR=ROOT/'results/routea_matched_three_arm_pair64_training_features_v2'
FULL=ROOT/'results/routea_matched_three_arm_fullnegative_features_v1'

def encode(d):return json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def tensor_sha(t):
 x=t.detach().cpu().contiguous();return hashlib.sha256(str(x.dtype).encode()+encode(list(x.shape))+x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def dump_tensor(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
def groupweights(groups):
 counts=Counter(groups);return torch.tensor([1/(len(counts)*counts[g]) for g in groups],dtype=torch.float64),dict(counts)

def prepare():
 M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_CUTOFF')
 qualified=M.read(SPLIT);M.need(qualified['status']=='RC_SHARED_QUERY_TARGET_PRIOR_SPLIT_QUALIFICATION_V1_PASS','EXISTING_SPLIT_PASS')
 pv=M.read(PAIR/'independent_validation.json');M.need(pv['status']=='ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED' and pv['v2_payload_sha256']==M.sha(PAIR/'payload.pt'),'PAIR_SEAL')
 pairs=torch.load(PAIR/'payload.pt',weights_only=True,map_location='cpu',mmap=True)['records'];fv=M.read(FULL/'validation.json');M.need(fv['status']=='ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_VALIDATED' and all(fv['checks'].values()),'FULL_SEAL')
 full=[];sources={}
 for s in fv['shards']:
  p=FULL/f"shard{int(s['shard']):02d}/payload.pt";M.need(M.sha(p)==s['payload_sha256'],'FULL_SOURCE_SHA');sources[str(s['shard'])]=M.bind(p)
  full += [r for r in torch.load(p,weights_only=True,map_location='cpu',mmap=True)['records'] if r['data_split_role']=='TRAIN']
 full.sort(key=lambda r:r['execution_ordinal']);cl=M.read(CLOSURE)
 M.need([[r['pair_cohort'],r['pair_row_ordinal'],r['execution_ordinal']] for r in pairs]==cl['pair_order'],'ORIGINAL_PAIR_ORDER')
 M.need([r['execution_ordinal'] for r in full]==cl['training_order'],'ORIGINAL_FULL_ORDER')
 design=torch.cat([r['real_native_features']['C_PAIRED'] for r in pairs+full]);M.need(tensor_sha(design)==cl['scales_and_design']['NATIVE7']['feature_sha256'],'ORIGINAL_4128X6_TRAIN_FEATURE_BITS')
 oldroles={Path(s['path']).name:s for s in qualified['role_sources']};modern={r['original_query_id']:r for r in M.read(PREV/'train_roles.json')['records']}
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;roles=[];role_sources=[];pairdata=[];fulldata=[]
 for pool,rows in [('PAIR',pairs),('FULL',full)]:
  for r in rows:
   seal=oldroles[f"role_exec{r['execution_ordinal']:03d}.json"];p=ROOT/seal['path'];M.need(M.sha(p)==seal['sha256'],'TRAIN_ROLE_SHA');role=M.read(p)
   logical=hashlib.sha256(encode({k:v for k,v in role.items() if k!='logical_sha256'})).hexdigest();M.need(logical==role['logical_sha256']==seal['logical_sha256'],'TRAIN_ROLE_LOGICAL_SHA')
   m=modern[r['query_id']];M.need(role['query_id']==r['query_id'] and role['execution_ordinal']==r['execution_ordinal'] and role['identity']==m['identity'] and role['supergroup']==m['group'],'OLD_AND_MODERN_TRAIN_METADATA')
   M.need(role['raw_d1_field_count']==role['target_spatial_supervision_count']==role['target_insertion_count']==0,'RETRIEVAL_ONLY_ROLE')
   target=[i for i,p in enumerate(r['candidate_physical_rows']) if labels[p]==role['identity']];M.need(len(target)==1,'NATURAL_TRAIN_TARGET')
   item={k:r[k] for k in ('query_id','execution_ordinal','base_winner_position','challenger_positions','candidate_physical_rows')};item['real_native_features']={'C_PAIRED':r['real_native_features']['C_PAIRED'].clone()};item['target_position']=target[0]
   if pool=='PAIR':
    item['switch_label']=r['switch_label'];M.need(bool(r['switch_label'])==(target[0]!=r['base_winner_position']) and (not r['switch_label'] or target[0]==r['challenger_positions'][0]),'ORIGINAL_PAIR_IDENTITY_LABEL');pairdata.append(item)
   else:fulldata.append(item)
   roles.append({**m,'pool':pool,'legacy_execution_ordinal':r['execution_ordinal'],'supergroup':role['supergroup']});role_sources.append(M.bind(p))
 M.need(len(pairdata)==64 and len(fulldata)==32 and len({r['query_id'] for r in roles})==96,'EXACT_ORIGINAL96')
 weights={};counts={}
 for pool in ('PAIR','FULL'):weights[pool],counts[pool]=groupweights([r['supergroup'] for r in roles if r['pool']==pool])
 M.need(len(counts['PAIR'])==20 and len(counts['FULL'])==12 and not(set(counts['PAIR'])&set(counts['FULL'])),'ORIGINAL20_PLUS12_GROUPS')
 dump_tensor(OUT/'train_inputs.pt',dict(pair={'records':pairdata},full=fulldata,weights=weights))
 M.write(OUT/'train_roles.json',dict(records=roles,evaluation_labels_included=False))
 M.write(OUT/'train_input_seal.json',dict(status='ORIGINAL_MIXED96_INPUTS_FROZEN',payload=M.bind(OUT/'train_inputs.pt'),train_roles=M.bind(OUT/'train_roles.json'),counts=dict(PAIR=64,FULL=32,unique_images=96,identities=32),group_counts={k:len(v) for k,v in counts.items()},group_image_counts=counts,feature_sha256=tensor_sha(design),original_training_order=cl['training_order'],original_pair_order=cl['pair_order'],sources=dict(pair=M.bind(PAIR/'payload.pt'),pair_validation=M.bind(PAIR/'independent_validation.json'),full_validation=M.bind(FULL/'validation.json'),split=M.bind(SPLIT),closure=M.bind(CLOSURE),modern_train=M.bind(PREV/'train_roles.json')),full_shards=sources,training_role_sources=role_sources,original_head_training_updates=0))
 print('ORIGINAL_MIXED96_INPUTS_FROZEN',flush=True)

def guard(stage):
 M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_CUTOFF')
 if stage=='prepare':return
 a=M.read(AUTH)
 for b in a['sources'].values():M.checked(b)
 if stage!='preflight':M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');v=M.read(OUT/'preflight.json');M.need(v['status']=='ORIGINAL_MIXED96_GROUP_PREFLIGHT_PASS' and v['authority']==M.bind(AUTH),'PREFLIGHT')
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower();M.need(not any(v in s for v in ('rc_opened_','d1-mi','d1_mi','grozi','gisc_prerecall_universe','/target_join/')),'PROTECTED_INPUT')
  if stage in ('fit','replay','preflight'):
   M.need('curator_roles' not in s and '/reports/' not in s and '/role_shards/' not in s,'ONLY_STAGED_TRAIN_LABELS')
   M.need(p.name not in ('result.json','result_validation.json'),'NO_OUTCOMES_IN_FIT');M.need(P.OUT/'fits' not in p.parents and P.OUT/'roles' not in p.parents,'NO593_FIT_LABELS')
 sys.addaudithook(audit)

def changed_function(weights):
 text=LEGACY.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='train_head');old=ast.get_source_segment(text,node)
 before=['pair_loss = (F.binary_cross_entropy_with_logits(pair_logits, py, reduction="none") * weights).mean()', 'full_loss = torch.stack(query_losses).mean()']
 after=['pair_loss = ((F.binary_cross_entropy_with_logits(pair_logits, py, reduction="none") * weights) * pair_group_weights).sum()', 'full_loss = (torch.stack(query_losses) * full_group_weights).sum()']
 body=old
 for x,y in zip(before,after):M.need(body.count(x)==1,'EXACT_TWO_MEAN_EDITS');body=body.replace(x,y)
 M.need('loss = pair_loss + full_loss' in body,'KEEP_ONE_TO_ONE_POOL_MIXTURE')
 def finite_tensor(x,name):M.need(bool(torch.isfinite(x).all()),'FINITE_'+name)
 ns=dict(torch=torch,nn=nn,F=F,STEPS=2000,FAMILIES={'NATIVE7':('real_native_features','cbind_native_features',list(range(6)))},finite_tensor=finite_tensor,CrossfitContractError=RuntimeError,pair_group_weights=weights['PAIR'],full_group_weights=weights['FULL'])
 exec(compile(body,str(LEGACY)+'#WITHIN_POOL_GROUP_MEANS','exec'),ns)
 return ns['train_head'],dict(original_function_sha256=hashlib.sha256(old.encode()).hexdigest(),changed_function_sha256=hashlib.sha256(body.encode()).hexdigest(),replacement_count=2,PAIR_FULL_mixture=[1,1])

def previous():
 a=M.read(AUTH);receipt=M.read(M.checked(a['sources']['previous_receipt']));v=M.read(M.checked(a['sources']['previous_fit_validation']))
 M.need(v['status']=='FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS' and v['payload']==receipt['payload'] and v['receipt']==a['sources']['previous_receipt'],'PREVIOUS_SEALED_PREDICTIONS')
 M.need(receipt['payload']==a['sources']['previous_payload'],'PREVIOUS_PAYLOAD_BINDING');return torch.load(M.checked(receipt['payload']),map_location='cpu',weights_only=True),receipt['payload']

def compute():
 seal=M.read(OUT/'train_input_seal.json');data=torch.load(M.checked(seal['payload']),weights_only=True,map_location='cpu');fn,change=changed_function(data['weights'])
 head,loss,finite=fn(data['pair'],data['full'],'NATIVE7','C_PAIRED');theta=torch.cat([head.weight.detach().flatten(),head.bias.detach()]);M.need(finite['all_finite'],'FINITE_NEW_TRAINING')
 old,oldbind=previous();oh=M.read(HEAD);expected=torch.tensor([float.fromhex(x) for x in oh['weight_binary64']]+[float.fromhex(oh['bias_binary64'])],dtype=torch.float64);M.need(torch.equal(old['parameters']['ORIGINAL7']['theta'],expected),'FROZEN_EC7_REUSED')
 oldpred={r['query_id']:r for r in old['predictions']};rows,featuresources=P.features();test=[r for r in rows if r['query_id'] in oldpred];pred=P.predict({'GROUP_MIXED96':dict(theta=theta)},test)
 for r in pred:r['models']['ORIGINAL7']=oldpred[r['query_id']]['models']['ORIGINAL7']
 return dict(parameters={'ORIGINAL7':dict(theta=expected),'GROUP_MIXED96':dict(theta=theta)},predictions=pred,previous_source_payload=oldbind,train_input_seal=M.bind(OUT/'train_input_seal.json'),train_roles=M.bind(OUT/'train_roles.json'),feature_sources=featuresources,authority=M.bind(AUTH),function_change=change,last_preupdate_losses=loss,finite_training=finite,original_head_training_updates=0,old_predictions_copied=True,new_head_training_updates=2000,heldout_label_reads=0)

def fit(replay,nonce):
 value=compute();folder=OUT/'fits'
 if replay:
  M.need(nonce==os.environ.get('MIXED96_NONCE'),'FRESH_REPLAY');r=M.read(folder/'receipt.json');old=torch.load(M.checked(r['payload']),weights_only=True,map_location='cpu');M.need(P.same(old,value),'NEW_HEAD_AND_PREDICTIONS_BIT_REPLAY')
  M.write(folder/'validation.json',dict(status='ORIGINAL_MIXED96_GROUP_FRESH_REPLAY_PASS',payload=r['payload'],receipt=M.bind(folder/'receipt.json'),authority=M.bind(AUTH),nonce=nonce,original_head_training_updates=0,heldout_label_reads=0))
 else:
  dump_tensor(folder/'payload.pt',value);M.write(folder/'receipt.json',dict(payload=M.bind(folder/'payload.pt'),authority=M.bind(AUTH)));nonce=uuid.uuid4().hex
  subprocess.run([sys.executable,__file__,'replay','--nonce',nonce],env=dict(os.environ,MIXED96_NONCE=nonce),check=True);print('ORIGINAL_MIXED96_GROUP_FRESH_REPLAY_PASS',flush=True)

def join():
 r=M.read(OUT/'fits/receipt.json');v=M.read(OUT/'fits/validation.json');M.need(v['status']=='ORIGINAL_MIXED96_GROUP_FRESH_REPLAY_PASS' and v['payload']==r['payload'] and v['receipt']==M.bind(OUT/'fits/receipt.json'),'SEAL_BEFORE_LABELS');pay=torch.load(M.checked(r['payload']),weights_only=True,map_location='cpu')
 a=M.read(AUTH)
 for b in a['evaluation_sources'].values():M.checked(b)
 oldval=M.read(PREV/'result_validation.json');M.need(oldval['status']=='FIXED_PANELS_INDEPENDENT_ACTION_RANK_COUNTS_PASS' and oldval['result']==M.bind(PREV/'result.json'),'PREVIOUS_VALIDATED_BASELINE');old=M.read(PREV/'result.json');roles={r['query_id']:r for r in M.read(PREV/'eval_curator_roles.json')['records']};split=M.read(PREV/'panel_manifest.json');pred={r['query_id']:r for r in pay['predictions']}
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;panels={}
 for panel,qids in split['panels'].items():
  rows=[];prior={r['query_id']:r for r in old['panels'][panel]['rows']}
  for q in qids:
   p=pred[q];role=roles[q];axis=p['candidate_physical_rows'];order=p['raw_ranked_physical_rows'];target=next(i for i in order if labels[i]==role['identity']);rank=order.index(target)+1;chosen={'RAW':axis[p['winner']]}
   for model in ('ORIGINAL7','GROUP_MIXED96'):
    for mode in ('REAL','CBIND'):chosen[model if mode=='REAL' else model+'_CBIND']=axis[p['models'][model][mode]['selected_position']]
   correct={k:t==target for k,t in chosen.items()};ranks={k:rank if k=='RAW' else 1 if correct[k] else rank+int(order.index(t)+1>rank) for k,t in chosen.items()}
   M.need(all(chosen[k]==prior[q]['selected_physical_rows'][k] and correct[k]==prior[q]['correct'][k] and ranks[k]==prior[q]['ranks'][k] for k in ('RAW','ORIGINAL7','ORIGINAL7_CBIND')),'REUSED_BASELINE_PERQUERY_PARITY')
   rows.append(dict(query_id=q,original_query_id=role['original_query_id'],component=role['component'],group=role['group'],target_in_C128=target in axis,correct=correct,ranks=ranks,selected_physical_rows=chosen))
  scores={k:dict(correct=sum(r['correct'][k] for r in rows),MRR=sum(1/r['ranks'][k] for r in rows)/len(rows)) for k in rows[0]['correct']}
  panels[panel]=dict(population=len(rows),recall_C128=sum(r['target_in_C128'] for r in rows),scores=scores,comparisons={'ORIGINAL7__to__GROUP_MIXED96':P.groupstats(rows,'ORIGINAL7','GROUP_MIXED96')},rows=rows)
 result=dict(status='ORIGINAL_MIXED96_GROUP_FIXED_PANEL_DEVELOPMENT_COMPLETE',panels=panels,authority=M.bind(AUTH),fit_validation=M.bind(OUT/'fits/validation.json'),previous_result_validation=M.bind(PREV/'result_validation.json'),original_head_training_updates=0,old_predictions_copied=True,new_head_training_updates=2000,only_factor='group weights inside each ORIGINAL training pool; PAIR64+FULL32 mixture remains1:1',historically_opened_development=True,external_confirmation=False,deployment_changed=False)
 M.write(OUT/'result.json',result);print(json.dumps({n:{'scores':p['scores'],'comparisons':p['comparisons']} for n,p in panels.items()},indent=2),flush=True)

def preflight():
 seal=M.read(OUT/'train_input_seal.json');M.need(seal['status']=='ORIGINAL_MIXED96_INPUTS_FROZEN','STAGED_INPUTS');data=torch.load(M.checked(seal['payload']),weights_only=True,map_location='cpu');fn,change=changed_function(data['weights'])
 for pool,n in [('PAIR',64),('FULL',32)]:M.need(len(data['weights'][pool])==n and abs(float(data['weights'][pool].sum())-1)<1e-14,'POOL_WEIGHT_NORMALIZATION')
 # Uniform group sizes must preserve the scalar loss and its derivatives, without training any head.
 z=torch.tensor([-1.,0.,1.,2.],dtype=torch.float64,requires_grad=True);labels=torch.tensor([0.,1.,0.,1.],dtype=torch.float64);cost=torch.where(labels==0,4.,1.);loss=F.binary_cross_entropy_with_logits(z,labels,reduction='none')*cost;w,_=groupweights(['a','a','b','b']);u=loss.mean();g=(loss*w).sum();M.need(torch.equal(u,g) and torch.equal(torch.autograd.grad(u,z,retain_graph=True)[0],torch.autograd.grad(g,z)[0]),'UNCHANGED_UNIFORM_POOL_LOSS_GRADIENT')
 M.write(OUT/'preflight.json',dict(status='ORIGINAL_MIXED96_GROUP_PREFLIGHT_PASS',authority=M.bind(AUTH),function_change=change,original_head_training_updates=0,natural_training_updates=0,original_train_feature_sha256=seal['feature_sha256']))
 print('ORIGINAL_MIXED96_GROUP_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','fit','replay','join']);ap.add_argument('--nonce');a=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(a.stage)
 if a.stage=='prepare':prepare()
 elif a.stage=='preflight':preflight()
 elif a.stage=='join':join()
 else:fit(a.stage=='replay',a.nonce)
