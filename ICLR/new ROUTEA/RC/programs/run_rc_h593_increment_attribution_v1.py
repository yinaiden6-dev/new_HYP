#!/usr/bin/env python3
"""One global-bias control and frozen increment-only binding intervention."""
import argparse,json,os,sys,subprocess,uuid
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P
import run_rc_h593_endpoint_competition_v1 as E
OUT=ROOT/'results/rc_h593_increment_attribution_v1';AUTH=ROOT/'registry/rc_h593_increment_attribution_authority_v1_20260911.json'
MODELS=('BASE7','BIAS1','CONSTANT1','CONDITIONAL4');MODES=('REAL','INCREMENT_BIND','CBIND')

def guard(stage,fold):
 a=M.read(AUTH)
 for b in a['sources'].values():M.checked(b)
 M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_DEADLINE')
 if stage!='preflight':
  M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');pf=M.read(OUT/'preflight.json');M.need(pf['status']=='BIAS_AND_INCREMENT_INTERVENTION_PREFLIGHT_PASS' and pf['authority']==M.bind(AUTH),'PREFLIGHT')
 own=(P.OUT/'roles'/f'fold{fold}.json').resolve()
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower();M.need(not any(x in s for x in ('rc_opened_','d1-mi','d1_mi','grozi','gisc_prerecall_universe','target_join')),'PROTECTED_READ')
  M.need(not (p.parent==P.OUT and p.name in ('result.json','result_validation.json')),'PARENT_OUTCOME_NOT_FIT_INPUT')
  M.need('rc_h593_endpoint_competition_v1/result' not in s,'ENDPOINT_OUTCOME_NOT_FIT_INPUT')
  if stage in ('fit','replay','preflight'):
   M.need('curator_roles' not in s and '/reports/' not in s,'NO_HELDOUT_LABELS')
   if P.OUT/'roles' in p.parents:M.need(p.resolve()==own,'CURRENT_FOLD_TRAIN_LABELS_ONLY')
   if P.OUT/'fits' in p.parents:M.need(P.OUT/'fits'/f'fold{fold}' in p.parents,'CURRENT_FOLD_PARENT_ONLY')
 sys.addaudithook(audit)

def predict(theta,oldparams,bias,rows):
 predictions=[]
 for r in rows:
  d={k:r[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')};d['models']={}
  for model in MODELS:
   d['models'][model]={}
   for mode in MODES:
    base_source='CBIND' if mode=='CBIND' else 'REAL';inc_source='REAL' if mode=='REAL' else 'CBIND';z=r['modes'][base_source]['X']@theta[:6]+theta[6]
    if model=='BIAS1':z=z+bias[0]
    elif model in ('CONSTANT1','CONDITIONAL4'):
     beta=oldparams[model]['beta'];q=r['modes'][inc_source];z=z+torch.nn.functional.softplus(q['context'][:,:len(beta)]@beta)*q['dF']
    k=int(torch.argmax(z));pos=r['challenger_positions'][k] if float(z[k])>0 else r['winner'];d['models'][model][mode]=dict(selected_position=pos,logits=z)
  for model in ('BASE7','BIAS1'):M.need(torch.equal(d['models'][model]['REAL']['logits'],d['models'][model]['INCREMENT_BIND']['logits']),'NON_EVIDENCE_MODELS_BIND_INVARIANT')
  predictions.append(d)
 return predictions

def parent(fold):
 folder=P.OUT/'fits'/f'fold{fold}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='H593_FOLD_RETRAIN_AND_PREDICTION_REPLAY_PASS' and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json'),'PARENT_VALIDATION');p=torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True);M.need(p['authority']==M.bind(P.AUTH) and p['fold']==fold,'PARENT_FOLD');return p,dict(payload=r['payload'],validation=M.bind(folder/'validation.json'))

def compute(fold):
 rows,sources=P.features();p,pb=parent(fold);theta=p['parameters']['BASE7_ALL']['theta'];split=M.read(P.OUT/'metadata/split_manifest.json')['folds'][fold];rolemap=M.read(P.OUT/'roles_manifest.json');rolepath=M.checked(rolemap['folds'][str(fold)]);roles={r['query_id']:r for r in M.read(rolepath)['records']};train=[r for r in rows if r['query_id'] in set(split['train_query_ids'])];test=[r for r in rows if r['query_id'] in set(split['heldout_query_ids'])];M.need(set(roles)=={r['query_id'] for r in train},'FOLD_TRAIN_SCOPE')
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;data=P.batch(train,roles,labels);base=data['X']@theta[:6]+theta[6];x=torch.ones((*base.shape,1),dtype=torch.float64);fit=E.fit_gamma(base.detach(),x,data['y'],steps=2000);predictions=predict(theta,p['parameters'],fit['gamma'],test);old={r['query_id']:r for r in p['predictions']}
 for r in predictions:
  for m,oldm in [('BASE7','BASE7_ALL'),('CONSTANT1','CONSTANT1'),('CONDITIONAL4','CONDITIONAL4')]:
   for mode in ('REAL','CBIND'):M.need(P.same(r['models'][m][mode],old[r['query_id']]['models'][oldm][mode]),'FROZEN_PARENT_PREDICTION_BITS')
 return dict(fold=fold,bias_fit=fit,predictions=predictions,source_parent=pb,feature_sources=sources,train_role=M.bind(rolepath),train_count=len(train),heldout_count=len(test),target_absent_train=data['target_absent_count'],authority=M.bind(AUTH),existing_model_parameters_updated=False)

def run_fold(fold,replay,nonce):
 folder=OUT/'fits'/f'fold{fold}';value=compute(fold)
 if replay:
  M.need(nonce==os.environ.get('INCREMENT_REPLAY_NONCE'),'FRESH_REPLAY');r=M.read(folder/'receipt.json');old=torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True);M.need(P.same(old,value),'BIAS_TRAINING_AND_ALL_PREDICTION_BITS')
  M.write(folder/'validation.json',dict(status='INCREMENT_FOLD_RETRAIN_PREDICTION_REPLAY_PASS',payload=r['payload'],receipt=M.bind(folder/'receipt.json'),fresh_nonce=nonce,authority=M.bind(AUTH),heldout_label_reads=0))
 else:
  folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt';M.need(not p.exists(),'IMMUTABLE_FOLD')
  with p.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
  M.write(folder/'receipt.json',dict(payload=M.bind(p),authority=M.bind(AUTH)))
  nonce=uuid.uuid4().hex;subprocess.run([sys.executable,__file__,'replay','--fold',str(fold),'--nonce',nonce],env=dict(os.environ,INCREMENT_REPLAY_NONCE=nonce),check=True)

def join():
 ps=[];sources=[]
 for f in range(5):
  folder=OUT/'fits'/f'fold{f}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='INCREMENT_FOLD_RETRAIN_PREDICTION_REPLAY_PASS' and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json') and v['authority']==M.bind(AUTH),'FIVE_PREDICTION_SEALS_BEFORE_LABELS');ps.append(torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True));sources.append(M.bind(folder/'validation.json'))
 roles={r['query_id']:r for r in M.read(P.OUT/'metadata/curator_roles.json')['records']}
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;rows=[]
 for p in ps:
  for r in p['predictions']:
   role=roles[r['query_id']];M.need(role['outer_fold']==p['fold'],'HELDOUT_FOLD');axis=r['candidate_physical_rows'];rank=next(i+1 for i,x in enumerate(r['raw_ranked_physical_rows']) if labels[x]==role['identity']);correct={'RAW':labels[axis[r['winner']]]==role['identity']};ranks={'RAW':rank}
   for m in MODELS:
    for mode in MODES:
     key=m if mode=='REAL' else m+'_'+mode;physical=axis[r['models'][m][mode]['selected_position']];correct[key]=labels[physical]==role['identity'];sr=r['raw_ranked_physical_rows'].index(physical)+1;ranks[key]=1 if correct[key] else rank+int(sr>rank)
   rows.append(dict(query_id=r['query_id'],original_query_id=role['original_query_id'],fold=p['fold'],component=role['component'],group=role['group'],target_in_C128=rank<=128,correct=correct,ranks=ranks))
 M.need(len(rows)==len({r['query_id'] for r in rows})==593,'COMPLETE593')
 scores={m:dict(correct=sum(r['correct'][m] for r in rows),MRR=sum(1/r['ranks'][m] for r in rows)/593) for m in rows[0]['correct']};comparisons={b+'__to__'+n:P.groupstats(rows,b,n) for b,n in [('BASE7','BIAS1'),('BIAS1','CONSTANT1'),('BIAS1','CONDITIONAL4'),('BASE7','CONDITIONAL4')]};binding={}
 for m in ('CONSTANT1','CONDITIONAL4'):
  rescue=[r for r in rows if r['correct'][m] and not r['correct']['BASE7']];binding[m]=dict(REAL_rescues=len(rescue),INCREMENT_BIND_rescues_retained=sum(r['correct'][m+'_INCREMENT_BIND'] for r in rescue),REAL_correct=scores[m]['correct'],INCREMENT_BIND_correct=scores[m+'_INCREMENT_BIND']['correct'])
 result=dict(status='H593_INCREMENT_ATTRIBUTION_FOLLOWUP_COMPLETE',recall_C128=sum(r['target_in_C128'] for r in rows),scores=scores,comparisons=comparisons,binding=binding,rows=rows,fold_validation_sources=sources,authority=M.bind(AUTH),post_result_followup=True,scientific_GO_claimed=False,old_EVAL128_99_comparable=False)
 M.write(OUT/'result.json',result);print(json.dumps({k:result[k] for k in ('status','scores','comparisons','binding')},indent=2))

def preflight():
 # A scalar global bias equals a change to the frozen bias term; no identity signal is added.
 theta=torch.arange(7,dtype=torch.float64)/8;xx=torch.eye(6,dtype=torch.float64);b=torch.tensor([.25],dtype=torch.float64);M.need(torch.equal(xx@theta[:6]+theta[6]+b[0],xx@theta[:6]+(theta[6]+b[0])),'BIAS_FORMULA')
 row=dict(query_id='synthetic',execution_ordinal=0,candidate_physical_rows=[0,1,2],raw_ranked_physical_rows=[1,0,2],winner=1,challenger_positions=[0,2],modes={m:dict(X=torch.zeros((2,6),dtype=torch.float64),context=torch.ones((2,4),dtype=torch.float64),dF=torch.tensor([.2,-.2] if m=='REAL' else [-.2,.2],dtype=torch.float64)) for m in ('REAL','CBIND')})
 params=dict(CONSTANT1=dict(beta=torch.zeros(1,dtype=torch.float64)),CONDITIONAL4=dict(beta=torch.zeros(4,dtype=torch.float64)));out=predict(torch.zeros(7,dtype=torch.float64),params,torch.zeros(1,dtype=torch.float64),[row])[0];M.need(out['models']['CONDITIONAL4']['REAL']['selected_position']==0 and out['models']['CONDITIONAL4']['INCREMENT_BIND']['selected_position']==2,'ONLY_INCREMENT_BIND_CHANGES_CONTENT_DIRECTION')
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'synthetic.pt'
 with p.open('xb') as f:torch.save(out,f)
 M.need(P.same(torch.load(p,map_location='cpu',weights_only=True),out),'PAYLOAD_ROUNDTRIP')
 M.write(OUT/'preflight.json',dict(status='BIAS_AND_INCREMENT_INTERVENTION_PREFLIGHT_PASS',authority=M.bind(AUTH),natural_training_steps=0))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','replay','join','preflight']);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--nonce');a=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(17);guard(a.stage,a.fold)
 if a.stage=='preflight':preflight()
 elif a.stage=='join':join()
 else:run_fold(a.fold,a.stage=='replay',a.nonce)
