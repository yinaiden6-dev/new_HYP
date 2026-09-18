#!/usr/bin/env python3
"""Read-only selected-head/endpoint replay; no optimization, no EVAL opens."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib,json,math,os,sys
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/rc_train128_protected_projection_oof4_v1'
OUT=ROOT/'results/rc_train128_protected_projection_result_review_v1'
INPUT=ROOT/'results/rc_original7_train128_inputs_v1'
PARENT=ROOT/'results/rc_train128_disagreement_oof4_v1'
SOURCES={}; BLOCKED=[]
MODELS=('BASE7','PROTECTED7','REPAIR_ONLY7')
NAMES=('RAW','S','M','L','Q','R','bias')
def guard(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
 if any(x in s for x in ('/rc_opened_','/rc_original7_eval','/grozi/','d1_mi','d1-mi','/target_join/','/role_shards/')) or p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.pt','.npz'):
  BLOCKED.append(str(p));raise PermissionError('OUTSIDE_READONLY_TRAIN_REVIEW')
sys.addaudithook(guard)
sys.path.insert(0,str(ROOT/'src'))
from rc_aslo_xf.gallery_identity_repair import build_identity_map
torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_flush_denormal(False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,expected=None):
 p=Path(p).resolve(); h=sha(p)
 if expected is not None: assert h==expected,(p,h,expected)
 SOURCES[str(p)]={'path':str(p),'sha256':h};return json.loads(p.read_text())
def checked(b):
 p=Path(b['path']);assert sha(p)==b['sha256'],p
 SOURCES[str(p)]={'path':str(p),'sha256':b['sha256']};return p
def frac(x): return Fraction.from_float(float(x))
def params(x): return [float.fromhex(v) for v in x['weight_binary64']]+[float.fromhex(x['bias_binary64'])]
def rank(c):
 s=c['training_score'];return (-Fraction(s['equal_group_accuracy_fraction']),-s['correct'],Fraction(s['L1_from_BASE_fraction']),c['source_error_execution_ordinal'])
def feature(row):
 raw=list(map(float.fromhex,row['candidate_raw_scores_binary64']));c4={int(k):{n:float.fromhex(v) for n,v in d.items()} for k,d in row['C4_binary64'].items()}
 mean=sum(raw)/128;sd=(sum((v-mean)**2 for v in raw)/128)**.5; w=row['base_winner_position']
 def sym(a,b): return (a-b)/(abs(a)+abs(b)+1e-12)
 out=[]
 for c in row['challenger_positions']:
  a,b=c4[c],c4[w];s,t=a['real_score'],b['real_score'];m,n=a['visibility_mass'],b['visibility_mass']
  out.append([(raw[c]-raw[w])/max(sd,1e-12),sym(s,t),sym(m,n),sym(s/max(m,1e-12),t/max(n,1e-12)),sym(s-a['query_control_score'],t-b['query_control_score']),sym(s-a['reference_control_score'],t-b['reference_control_score'])])
 assert [[float(v).hex() for v in a] for a in out]==row['features_binary64']['REAL']
 return torch.tensor(out,dtype=torch.float64)
def predict(row,theta):
 z=row['X']@torch.tensor(theta[:6],dtype=torch.float64)+theta[6];cs=row['challenger_positions'];axis=row['candidate_physical_rows']
 k=max(range(127),key=lambda j:(float(z[j]),-axis[cs[j]]));final=cs[k] if float(z[k])>0 else row['base_winner_position']
 return {'all127_logits_binary64':[float(v).hex() for v in z], 'best_challenger_position':cs[k],'best_logit_binary64':float(z[k]).hex(),'final_position':final,'final_physical_row':axis[final],'action':'SWITCH' if float(z[k])>0 else 'HOLD'}
def score(rows,theta,theta0):
 good=[r['query_id'] for r in rows if predict(r,theta)['final_position']==r['target_position']]
 groups=sorted({r['group'] for r in rows});means=[Fraction(sum(r['query_id'] in good for r in rows if r['group']==g),sum(r['group']==g for r in rows)) for g in groups]
 return {'correct':len(good),'correct_query_ids':good,'training_image_count':len(rows),'training_group_count':len(groups),'equal_group_accuracy_fraction':str(sum(means,Fraction(0))/len(groups)),'L1_from_BASE_fraction':str(sum((abs(frac(a)-frac(b)) for a,b in zip(theta,theta0)),Fraction(0)))}
def minimum_margin(rows,theta):
 p=list(map(frac,theta));result=None
 for r in rows:
  cs=r['challenger_positions']; vectors={c:list(map(frac,r['X'][j]))+[Fraction(1)] for j,c in enumerate(cs)};t=r['target_position']
  if t==r['base_winner_position']: constraints=[[-x for x in vectors[c]] for c in cs]
  else: constraints=[vectors[t]]+[[x-y for x,y in zip(vectors[t],vectors[c])] for c in cs if c!=t]
  for a in constraints:
   z=sum((x*y for x,y in zip(a,p)),Fraction(0));result=z if result is None or z<result else result
 return result
result=read(SOURCE/'result.json','5f5cc6e1a0aba87c22b3c0fea7040d9b54304d38e7db27fce42a9147ae398042')
validation=read(SOURCE/'validation.json','f852f1dc4f4fd93c2688331b989259353713cb834dc38583f87e7dfb8faefc46')
checked(validation['result']);checked(validation['program']);checked(validation['authority'])
inputval=read(INPUT/'validation.json');rows=read(checked(inputval['feature_records']));roles={r['query_id']:r for r in result['rows']}
manifest=read(PARENT/'fold_manifest.json');assert len(rows)==len(roles)==128
images=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images';paths=[]
for directory,subdirs,filenames in os.walk(images):
 subdirs.sort();paths.extend(Path(directory)/n for n in sorted(filenames) if (Path(directory)/n).is_file() and Path(n).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'})
assert len(paths)==5413
mapping=build_identity_map([p.stem.strip() for p in paths]);assert mapping.corrected_row_identity_mapping_sha256==result['gallery_mapping_sha256']
read(ROOT/'registry/gallery_identity_repair_v1.json');read(ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json')
for row in rows:
 role=roles[row['query_id']];row.update(group=role['group'],fold=role['fold'],original_query_id=role['original_query_id']);row['X']=feature(row)
 targets=[p for p,a in enumerate(row['candidate_physical_rows']) if mapping.labels[a]==role['identity']];assert len(targets)==1;row['target_position']=targets[0]
 assert (row['target_position']==row['base_winner_position'])==role['RAW_correct']
folds=[];losses=[];gains=[];heldout_checks=0;allpred={}; training_score_replays=0
for f in range(4):
 directory=SOURCE/f'fold{f:02d}';v=read(checked(result['fold_validations'][str(f)]));seal=read(checked(v['seal']));pars=read(checked(seal['parameters']));ps=read(checked(seal['predictions']))
 prior=read(checked(seal['closure']['prior_parameters']));assert prior['BASE7']==pars['BASE7']
 for key in ['prior_fold_validation','prior_fold_seal','prior_predictions','train_roles','fold_manifest']:checked(seal['closure'][key])
 assert v['heldout_label_reads']==v['EVAL_reads']==v['forbidden_read_attempts']==0 and v['all_candidate_selection_and_parameter_bits_replayed']
 train=[r for r in rows if r['fold']!=f];held=[r for r in rows if r['fold']==f];assert set(r['group'] for r in train).isdisjoint(r['group'] for r in held)
 assert [r['execution_ordinal'] for r in train]==seal['closure']['training_ordinals'];assert [r['execution_ordinal'] for r in held]==seal['closure']['heldout_ordinals']
 base=params(pars['BASE7']);base_score=score(train,base,base);protected=[r for r in train if r['query_id'] in base_score['correct_query_ids']];delta=minimum_margin(protected,base)
 frec={'fold':f,'training_count':len(train),'heldout_count':len(held),'protected_training_count':len(protected),'BASE_training_correct':base_score['correct'],'delta':float(delta),'delta_exact_fraction':str(delta),'models':{}}
 for m in MODELS[1:]:
  a=pars[m];theta=params(a);s=score(train,theta,base);training_score_replays+=1;assert s==a['selected_training_score'];assert delta==Fraction(a['delta_exact_fraction'])
  assert set(a['protected_training_query_ids'])==set(base_score['correct_query_ids']);assert min(a['candidate_pool'],key=rank)['candidate_id']==a['selected_candidate_id']
  assert min(a['candidate_pool'],key=rank)['parameters']['parameter_sha256']==a['parameter_sha256']
  for cand in a['candidate_pool']:assert score(train,params(cand['parameters']),base)==cand['training_score'];training_score_replays+=1
  bset=set(base_score['correct_query_ids']);sset=set(s['correct_query_ids']);assert bset-sset==set(a['selected_training_BASE_loss_query_ids']);assert sset-bset==set(a['selected_training_rescue_query_ids'])
  required=protected+[r for r in train if r['query_id']==a['selected_candidate_id']] if m=='PROTECTED7' else [r for r in train if r['query_id']==a['selected_candidate_id']]
  minimum=minimum_margin(required,theta);entry=next(e for e in a['candidate_ledger'] if e['error_query_id']==a['selected_candidate_id']);assert minimum==Fraction(entry['actual_min_required_margin_fraction']) and minimum>0
  if m=='PROTECTED7':assert not bset-sset
  frec['models'][m]={'parameters':theta,'BASE_parameters':base,'delta_parameters':[x-y for x,y in zip(theta,base)],'L1':float(Fraction(s['L1_from_BASE_fraction'])),'L1_fraction':s['L1_from_BASE_fraction'],'training_correct':s['correct'],'training_rescues':len(sset-bset),'training_losses':len(bset-sset),'equal_group_accuracy_fraction':s['equal_group_accuracy_fraction'],'selected_source_error':roles[a['selected_candidate_id']]['original_query_id'],'required_minimum_margin':float(minimum),'required_minimum_margin_fraction':str(minimum),'exact_delta_floor_met':minimum>=delta,'LP_calls':a['LP_call_count'],'accepted_candidates_including_BASE':a['accepted_candidate_count_including_BASE'],'solver_success':entry['solver_success'],'solver_status':entry['solver_status'],'fallback_reason':a['fallback_reason']}
 for r,p in zip(held,ps,strict=True):
  assert r['query_id']==p['query_id'];allpred[r['query_id']]=p
  for m in MODELS:
   z=predict(r,params(pars[m]));assert z==p['predictions'][m];heldout_checks+=127
   assert (z['final_position']==r['target_position'])==roles[r['query_id']]['correct'][m]
  changed=roles[r['query_id']]['correct'];kind='loss' if changed['BASE7'] and not changed['PROTECTED7'] else 'gain' if not changed['BASE7'] and changed['PROTECTED7'] else None
  if kind:
   z=p['predictions']['PROTECTED7'];c=z['best_challenger_position'];j=r['challenger_positions'].index(c);x=list(map(float,r['X'][j]))+[1.];a=params(pars['PROTECTED7']);oldz=float.fromhex(p['predictions']['BASE7']['all127_logits_binary64'][j]);newz=float.fromhex(z['all127_logits_binary64'][j]);contrib=[(v-b)*xx for v,b,xx in zip(a,base,x)]
   detail={k:roles[r['query_id']][k] for k in ['query_id','original_query_id','execution_ordinal','fold','group','identity','RAW_correct']};detail.update({'kind':kind,'predictions':{m:{k:(float.fromhex(v) if k=='best_logit_binary64' else v) for k,v in p['predictions'][m].items() if k!='all127_logits_binary64'} for m in MODELS},'selected_PROTECTED_candidate_base_logit':oldz,'selected_PROTECTED_candidate_new_logit':newz,'selected_PROTECTED_candidate_feature':dict(zip(NAMES,x)),'logit_delta_parameter_contributions':dict(zip(NAMES,contrib)),'contribution_rounding_residual':newz-oldz-math.fsum(contrib),'strongest_challenger_changed':p['predictions']['BASE7']['best_challenger_position']!=c,'target_position':r['target_position']})
   (losses if kind=='loss' else gains).append(detail)
 folds.append(frec)
assert heldout_checks==48768 and len(losses)==4 and len(gains)==6
assert all(d['RAW_correct'] and d['predictions']['BASE7']['action']=='HOLD' and d['predictions']['PROTECTED7']['action']=='SWITCH' for d in losses)
metrics={m:{k:result['scores'][m][k] for k in ['correct','rescue_vs_RAW','break_vs_RAW','equal_group_accuracy_fraction']} for m in MODELS}
review={'status':'TRAIN128_PROTECTED_PROJECTION_SELECTED_HEAD_READONLY_REVIEW_PASS','scope':'post-hoc TRAIN128 group OOF development only; no EVAL inference','source_result':SOURCES[str(SOURCE/'result.json')],'source_validation':SOURCES[str(SOURCE/'validation.json')],'sources':list(SOURCES.values()),'feature_scalar_bit_checks':128*127*6,'heldout_logit_bit_checks':heldout_checks,'candidate_pool_training_scores_replayed':training_score_replays,'selected_training_protection_verified_all_folds':True,'EVAL_reads':0,'image_pixel_reads':0,'optimizer_updates':0,'LP_calls':0,'forbidden_read_attempts':len(BLOCKED),'metrics':metrics,'folds':folds,'PROTECTED_losses':losses,'PROTECTED_gains':gains,'OOF_gate_pass':False,'failure':'all four additional losses are heldout RAW-correct HOLD to incorrect SWITCH; empirical TRAIN protection does not extend to unseen groups','limits':['readout replays saved heads and finite candidate selection, not a fresh LP optimization','large L1 movement is measured, but a smaller step alone is not proven sufficient','no claim that new HYP or all seven-parameter model classes fail','TRAIN128 repeatedly reused; OOF development is not untouched external confirmation']}
OUT.mkdir(parents=True,exist_ok=True);target=OUT/'review.json';data=(json.dumps(review,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
if target.exists():assert target.read_bytes()==data
else:target.write_bytes(data);target.chmod(0o444)
print(json.dumps({'review':str(target),'sha256':sha(target),'metrics':metrics,'folds':[{k:f[k] for k in ['fold','protected_training_count','delta']} for f in folds],'losses':[{k:d[k] for k in ['original_query_id','fold','logit_delta_parameter_contributions']} for d in losses]},indent=2))
