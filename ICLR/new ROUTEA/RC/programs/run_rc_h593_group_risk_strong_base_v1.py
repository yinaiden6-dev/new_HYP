#!/usr/bin/env python3
"""Same-data group-risk intervention and completed SMALL-base compensation controls."""
import argparse,ast,copy,json,os,sys,subprocess,uuid
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P
OUT=ROOT/'results/rc_h593_group_risk_strong_base_v1';AUTH=ROOT/'registry/rc_h593_group_risk_strong_base_authority_v1_20260911.json'
MODELS=('ALL_BASE','ALL_CONST','ALL_COND','SMALL_BASE','SMALL_CONST','SMALL_COND','GROUP_BASE','GROUP_CONST','GROUP_COND');MODES=('REAL','INCREMENT_BIND','CBIND');LOSS_VECTOR=None

def guard(stage,fold):
 a=M.read(AUTH)
 for b in a['sources'].values():M.checked(b)
 M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_DEADLINE')
 if stage!='preflight':
  M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');pf=M.read(OUT/'preflight.json');M.need(pf['status']=='GROUP_RISK_AND_STRONG_BASE_PREFLIGHT_PASS' and pf['authority']==M.bind(AUTH),'PREFLIGHT')
 own=(P.OUT/'roles'/f'fold{fold}.json').resolve()
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower();M.need(not any(x in s for x in ('rc_opened_','d1-mi','d1_mi','grozi','gisc_prerecall_universe','target_join')),'PROTECTED_SOURCE')
  M.need(not (p.parent==P.OUT and p.name in ('result.json','result_validation.json')),'PARENT_OUTCOMES_NOT_FIT_INPUTS')
  if stage in ('fit','replay','preflight'):
   M.need('curator_roles' not in s and '/reports/' not in s,'NO_HELDOUT_LABELS')
   if P.OUT/'roles' in p.parents:M.need(p.resolve()==own,'CURRENT_FOLD_LABELS_ONLY')
   if P.OUT/'fits' in p.parents:M.need(P.OUT/'fits'/f'fold{fold}' in p.parents,'CURRENT_FOLD_PARENT_ONLY')
 sys.addaudithook(audit)

def per_query_loss(z,y):
 global LOSS_VECTOR
 if LOSS_VECTOR is None:
  tree=ast.parse(Path(P.__file__).read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='loss'));ret=node.body[-1];M.need(isinstance(ret,ast.Return) and isinstance(ret.value,ast.Call) and isinstance(ret.value.func,ast.Attribute) and ret.value.func.attr=='mean','ONLY_FINAL_MEAN_REMOVED');ret.value=ret.value.func.value;node.name='loss_vector';ns={'torch':torch,'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(P.__file__),'exec'),ns);LOSS_VECTOR=ns['loss_vector']
 return LOSS_VECTOR(z,y)

def group_weights(groups):
 counts=Counter(groups);g=len(counts);M.need(g>0,'NONEMPTY_EFFECTIVE_GROUPS');w=torch.tensor([1/(g*counts[x]) for x in groups],dtype=torch.float64);M.need(abs(float(w.sum())-1)<1e-12,'UNIT_DATA_LOSS_WEIGHT')
 for name in counts:M.need(abs(float(w[torch.tensor([x==name for x in groups])].sum())-1/g)<1e-12,'EQUAL_GROUP_TOTAL_WEIGHT')
 return w,dict(sorted(counts.items()))

def fit_group_base(data,weight):
 theta=torch.nn.Parameter(torch.zeros(7,dtype=torch.float64));optimizer=torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
 for _ in range(2000):
  optimizer.zero_grad();z=data['X']@theta[:6]+theta[6];loss=(weight*per_query_loss(z,data['y'])).sum();M.need(bool(torch.isfinite(loss)),'FINITE_GROUP_BASE_LOSS');loss.backward();M.need(bool(torch.isfinite(theta.grad).all()),'FINITE_GROUP_BASE_GRAD');optimizer.step()
 return dict(theta=theta.detach(),last_preupdate_loss=float(loss.detach()))

def fit_group_correction(data,theta,count,weight):
 beta=torch.nn.Parameter(torch.zeros(count,dtype=torch.float64));optimizer=torch.optim.AdamW([beta],lr=.03,weight_decay=.001);base=(data['X']@theta[:6]+theta[6]).detach()
 for _ in range(2000):
  optimizer.zero_grad();z=base+F.softplus(data['context'][:,:,:count]@beta)*data['dF'];loss=(weight*per_query_loss(z,data['y'])).sum();M.need(bool(torch.isfinite(loss)),'FINITE_GROUP_CORRECTION_LOSS');loss.backward();M.need(bool(torch.isfinite(beta.grad).all()),'FINITE_GROUP_CORRECTION_GRAD');optimizer.step()
 return dict(theta=theta,beta=beta.detach(),last_preupdate_loss=float(loss.detach()))

def parent(fold):
 folder=P.OUT/'fits'/f'fold{fold}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='H593_FOLD_RETRAIN_AND_PREDICTION_REPLAY_PASS' and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json'),'PARENT_VALIDATION');p=torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True);M.need(p['authority']==M.bind(P.AUTH) and p['fold']==fold,'PARENT_FOLD');return p,dict(payload=r['payload'],validation=M.bind(folder/'validation.json'))

def predict(params,rows):
 out=[]
 for r in rows:
  row={k:r[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')};row['models']={}
  for name,par in params.items():
   row['models'][name]={};theta=par['theta']
   for mode in MODES:
    bmode='CBIND' if mode=='CBIND' else 'REAL';imode='REAL' if mode=='REAL' else 'CBIND';z=r['modes'][bmode]['X']@theta[:6]+theta[6]
    if 'beta' in par:
     beta=par['beta'];d=r['modes'][imode];z=z+F.softplus(d['context'][:,:len(beta)]@beta)*d['dF']
    index=int(torch.argmax(z));pos=r['challenger_positions'][index] if float(z[index])>0 else r['winner'];row['models'][name][mode]=dict(selected_position=pos,logits=z)
  out.append(row)
 return out

def compute(fold):
 rows,sources=P.features();p,pb=parent(fold);split=M.read(P.OUT/'metadata/split_manifest.json')['folds'][fold];rm=M.read(P.OUT/'roles_manifest.json');rp=M.checked(rm['folds'][str(fold)]);roles={r['query_id']:r for r in M.read(rp)['records']};train=[r for r in rows if r['query_id'] in set(split['train_query_ids'])];test=[r for r in rows if r['query_id'] in set(split['heldout_query_ids'])];M.need(set(roles)=={r['query_id'] for r in train},'TRAIN_SCOPE')
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;data=P.batch(train,roles,labels);eligible=[r for r in train if P.target_position(r,roles[r['query_id']]['identity'],labels)>=-1];M.need(len(eligible)==len(data['y']),'LOSS_ELIGIBLE_AXIS');weight,counts=group_weights([roles[r['query_id']]['component'] for r in eligible])
 params={new:p['parameters'][old] for new,old in [('ALL_BASE','BASE7_ALL'),('ALL_CONST','CONSTANT1'),('ALL_COND','CONDITIONAL4'),('SMALL_BASE','BASE7_SMALL128')]}
 small_theta=params['SMALL_BASE']['theta']
 for name,n in [('SMALL_CONST',1),('SMALL_COND',4)]:
  beta,loss=P.fit_correction(data,small_theta,n);params[name]=dict(theta=small_theta,beta=beta,last_preupdate_loss=loss)
 params['GROUP_BASE']=fit_group_base(data,weight);gt=params['GROUP_BASE']['theta']
 params['GROUP_CONST']=fit_group_correction(data,gt,1,weight);params['GROUP_COND']=fit_group_correction(data,gt,4,weight)
 predictions=predict(params,test);old={r['query_id']:r for r in p['predictions']}
 for r in predictions:
  for new,previous in [('ALL_BASE','BASE7_ALL'),('ALL_CONST','CONSTANT1'),('ALL_COND','CONDITIONAL4'),('SMALL_BASE','BASE7_SMALL128')]:
   for mode in ('REAL','CBIND'):M.need(P.same(r['models'][new][mode],old[r['query_id']]['models'][previous][mode]),'EXISTING_MODEL_PREDICTION_BIT_PARITY')
 small=set(p['small_query_ids']);small_kept=[r for r in eligible if r['query_id'] in small];sc=Counter(roles[r['query_id']]['component'] for r in small_kept);keys=set(sc)|set(counts);tv=.5*sum(abs(sc.get(k,0)/len(small_kept)-counts.get(k,0)/len(eligible)) for k in keys)
 return dict(fold=fold,parameters=params,predictions=predictions,feature_sources=sources,source_parent=pb,train_role=M.bind(rp),train_images=len(train),heldout_images=len(test),loss_eligible_images=len(eligible),effective_groups=len(counts),effective_group_image_counts=counts,group_weights=weight,old_small_effective_group_counts=dict(sc),old_small_vs_ALL_effective_group_weight_TV=tv,authority=M.bind(AUTH),same_ALL_rows_for_weight_intervention=True)

def run_fold(fold,replay,nonce):
 folder=OUT/'fits'/f'fold{fold}';value=compute(fold)
 if replay:
  M.need(nonce==os.environ.get('GROUP_RISK_NONCE'),'FRESH_REPLAY');r=M.read(folder/'receipt.json');old=torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True);M.need(P.same(old,value),'ALL_NEW_FITS_AND_PREDICTIONS_BIT_REPLAY');M.write(folder/'validation.json',dict(status='GROUP_RISK_FOLD_RETRAIN_PREDICTION_REPLAY_PASS',payload=r['payload'],receipt=M.bind(folder/'receipt.json'),authority=M.bind(AUTH),fresh_nonce=nonce,heldout_label_reads=0))
 else:
  folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt';M.need(not p.exists(),'IMMUTABLE_FOLD')
  with p.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
  M.write(folder/'receipt.json',dict(payload=M.bind(p),authority=M.bind(AUTH)))
  nonce=uuid.uuid4().hex;subprocess.run([sys.executable,__file__,'replay','--fold',str(fold),'--nonce',nonce],env=dict(os.environ,GROUP_RISK_NONCE=nonce),check=True)

def join():
 payloads=[];seals=[]
 for f in range(5):
  folder=OUT/'fits'/f'fold{f}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='GROUP_RISK_FOLD_RETRAIN_PREDICTION_REPLAY_PASS' and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json') and v['authority']==M.bind(AUTH),'ALL_FOLD_SEALS');payloads.append(torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True));seals.append(M.bind(folder/'validation.json'))
 roles={r['query_id']:r for r in M.read(P.OUT/'metadata/curator_roles.json')['records']}
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities;rows=[]
 for p in payloads:
  for r in p['predictions']:
   role=roles[r['query_id']];M.need(role['outer_fold']==p['fold'],'HELDOUT_GROUP');axis=r['candidate_physical_rows'];rank=next(i+1 for i,x in enumerate(r['raw_ranked_physical_rows']) if labels[x]==role['identity']);correct={'RAW':labels[axis[r['winner']]]==role['identity']};ranks={'RAW':rank}
   for name in MODELS:
    for mode in MODES:
     key=name if mode=='REAL' else name+'_'+mode;physical=axis[r['models'][name][mode]['selected_position']];correct[key]=labels[physical]==role['identity'];sr=r['raw_ranked_physical_rows'].index(physical)+1;ranks[key]=1 if correct[key] else rank+int(sr>rank)
   rows.append(dict(query_id=r['query_id'],original_query_id=role['original_query_id'],fold=p['fold'],component=role['component'],group=role['group'],target_in_C128=rank<=128,correct=correct,ranks=ranks))
 M.need(len(rows)==len({r['query_id'] for r in rows})==593,'FULL593')
 pairs=[('ALL_COND','GROUP_COND'),('SMALL_COND','GROUP_COND'),('ALL_BASE','GROUP_BASE'),('ALL_CONST','GROUP_CONST'),('SMALL_BASE','SMALL_CONST'),('SMALL_BASE','SMALL_COND'),('SMALL_CONST','SMALL_COND'),('GROUP_CONST','GROUP_COND')]
 scores={m:dict(correct=sum(r['correct'][m] for r in rows),MRR=sum(1/r['ranks'][m] for r in rows)/593) for m in rows[0]['correct']};comparisons={b+'__to__'+n:P.groupstats(rows,b,n) for b,n in pairs};result=dict(status='H593_GROUP_RISK_STRONG_BASE_OOF5_COMPLETE',scores=scores,comparisons=comparisons,rows=rows,recall_C128=sum(r['target_in_C128'] for r in rows),fold_validation_sources=seals,authority=M.bind(AUTH),scientific_GO_claimed=False,old_EVAL128_99_comparable=False,group_weight_is_experimental_target_not_assumed_true_distribution=True)
 M.write(OUT/'result.json',result);print(json.dumps({k:result[k] for k in ('status','scores','comparisons')},indent=2))

def preflight():
 y=torch.tensor([-1,0,31,126]);z=torch.arange(4*127,dtype=torch.float64).reshape(4,127)/128;M.need(torch.equal(per_query_loss(z,y).mean(),P.loss(z,y)),'ORIGINAL_PER_QUERY_LOSS_UNCHANGED')
 groups=['a','a','b','b'];w,_=group_weights(groups);M.need(torch.equal(w,torch.full((4,),.25,dtype=torch.float64)),'UNIFORM_EQUAL_SIZE_GROUPS')
 theta=torch.tensor(.5,dtype=torch.float64,requires_grad=True);x=torch.tensor([1.,1.,-1.,-1.],dtype=torch.float64);labels=torch.full((4,),-1);scores=(x*theta)[:,None].expand(-1,127);risk=(w*per_query_loss(scores,labels)).sum();gradient=torch.autograd.grad(risk,theta)[0]
 dup=[0,1,0,1,2,3];wd,_=group_weights([groups[i] for i in dup]);td=torch.tensor(.5,dtype=torch.float64,requires_grad=True);sd=(x[dup]*td)[:,None].expand(-1,127);rd=(wd*per_query_loss(sd,labels[dup])).sum();gd=torch.autograd.grad(rd,td)[0];M.need(torch.allclose(risk,rd,atol=1e-13,rtol=0) and torch.allclose(gradient,gd,atol=1e-13,rtol=0),'WHOLE_GROUP_REPLICATION_RISK_AND_GRADIENT_INVARIANCE');M.need(not torch.allclose(P.loss(scores,labels),P.loss(sd,labels[dup]),atol=1e-13,rtol=0),'IMAGE_RISK_CHANGES_GROUP_PRIOR')
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'synthetic.pt'
 with p.open('xb') as f:torch.save(dict(weights=w,risk=risk.detach(),gradient=gradient),f)
 r=torch.load(p,map_location='cpu',weights_only=True);M.need(torch.equal(r['weights'],w),'SERIALIZATION')
 M.write(OUT/'preflight.json',dict(status='GROUP_RISK_AND_STRONG_BASE_PREFLIGHT_PASS',authority=M.bind(AUTH),natural_training_steps=0,group_replication_gradient_check=True))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','replay','join','preflight']);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--nonce');a=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(17);guard(a.stage,a.fold)
 if a.stage=='preflight':preflight()
 elif a.stage=='join':join()
 else:run_fold(a.fold,a.stage=='replay',a.nonce)
