#!/usr/bin/env python3
"""Bounded TRAIN-risk search and separate label-aware rescue-cone diagnostics."""
import argparse,json,os,sys,subprocess,uuid
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import rc_convex_loss_cause_core_v1 as K
TRAIN=ROOT/'results/rc_convex_train_loss_cause_v1'
DIAG=ROOT/'results/rc_opened_convex_loss_rescue_cones_v1'
PREV=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
AUTH=ROOT/'registry/rc_convex_cause_isolation_authority_v1_20260912.json'
BOUND=64;TOL=1e-5;CUTS=256;SECONDS=420

def sourcecheck(stage):
 a=M.read(AUTH)
 for b in a['sources'].values():M.checked(b)
 if stage!='preflight':M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).absolute();s=str(p).lower();M.need(not any(x in s for x in ('d1-mi','d1_mi','grozi','gisc_prerecall_universe','/target_join/')),'PROTECTED_INPUT')
  if stage in ('preflight','train','verify_train','predict'):
   M.need('rc_opened_' not in s and 'curator_roles' not in s and '/reports/' not in s,'TRAIN_LABEL_ISOLATION')
   M.need(p.name not in ('result.json','result_validation.json') or TRAIN in p.parents,'NO_EVAL_RESULT_READ')
   parent=ROOT/'results/rc_new_hyp593_oof5_v1'
   M.need(parent/'fits' not in p.parents and parent/'roles' not in p.parents,'NO_OTHER_FOLD_PARAMS_OR_LABELS')
 sys.addaudithook(audit)
 if stage!='preflight':
  pf=M.read(TRAIN/'preflight.json');M.need(pf['authority']==M.bind(AUTH) and pf['status']=='CONVEX_CAUSE_PIPELINE_PREFLIGHT_PASS','QUALIFIED_PREFLIGHT')
 return a

def problem():
 p=M.read(TRAIN/'train_problem.json');return p,K.decode_terms(p['terms'])
def folder(stage,index):return TRAIN/'fit' if stage in ('train','verify_train') else DIAG/'solves'/f'cone{index:02d}'
def cone(index,a):
 manifest=M.read(M.checked(a['diagnostic_sources']['manifest']));entry=manifest['cones'][index]
 if 'binding' in entry:entry=entry['binding']
 elif 'input' in entry:entry=entry['input']
 data=M.read(M.checked(entry));return data,[[Fraction(x) for x in row] for row in data['A_exact']]
def compute(stage,index):
 p,terms=problem();old=[float.fromhex(v) for v in p['old_theta_binary64']];M.need(max(map(abs,old))<=BOUND,'OLD_HEAD_INSIDE_BOX');baseline=K.objective_interval(terms,old);a=M.read(AUTH)
 args=dict(max_cuts=CUTS,max_seconds=SECONDS,bound=BOUND,tolerance=TOL,progress=lambda v:print(json.dumps(v),flush=True));extra={}
 if stage=='diagnose':
  c,A=cone(index,a);args.update(cone_float=np.asarray(A,dtype=np.float64),cone_exact=A,strict_witness=[Fraction(v) for v in c['strict_witness']],conflict_threshold=baseline['upper_rational']);extra=dict(cone_id=c['cone_id'],cone_index=index,cone_input=M.bind(Path(a['diagnostic_sources']['manifest']['path']).parent/f'cone{index:02d}.json'),label_aware=True,not_a_trainable_model=True)
 result=K.solve(terms,old,**args)
 components={'original':{pool:K.objective_interval(tt,old) for pool,tt in [('PAIR',terms[:64]),('FULL',terms[64:])]}}
 if result['best_theta_exact'] is not None:components['solution']={pool:K.objective_interval(tt,result['best_theta_exact']) for pool,tt in [('PAIR',terms[:64]),('FULL',terms[64:])]}
 return dict(status='CONVEX_CAUSE_SEARCH_COMPLETED',solver=result,baseline_objective=baseline,train_problem=M.bind(TRAIN/'train_problem.json'),authority=M.bind(AUTH),stage=stage,original_head_training_updates=0,new_Adam_updates=0,loss_components=components,scope='REAL_ARITHMETIC_LOGGED_SURROGATE_WITHIN_FIXED_PARAMETER_BOX',AdamW_decay_not_reinterpreted_as_L2=True,**extra)

def solve(stage,index):
 out=folder(stage,index);out.mkdir(parents=True,exist_ok=True);result=compute(stage,index);M.write(out/'search.json',result);nonce=uuid.uuid4().hex
 verify='verify_train' if stage=='train' else 'verify_cone'
 subprocess.run([sys.executable,__file__,verify,'--index',str(index),'--nonce',nonce],env=dict(os.environ,RC_CONVEX_VERIFY_NONCE=nonce),check=True)
 print(json.dumps(dict(event='SEARCH_AND_BOUND_REPLAY_COMPLETE',stage=stage,index=index,status=result['solver']['status'],lower=result['solver']['lower_bound'],upper=result['solver']['upper_bound'],gap=result['solver']['certified_gap_upper'])),flush=True)

def verify(stage,index,nonce):
 M.need(nonce==os.environ.get('RC_CONVEX_VERIFY_NONCE'),'FRESH_BOUND_VALIDATOR');out=folder(stage,index);s=M.read(out/'search.json');M.need(s['authority']==M.bind(AUTH) and s['train_problem']==M.bind(TRAIN/'train_problem.json'),'SEARCH_BINDINGS');p,terms=problem();kwargs={}
 if stage=='verify_cone':c,A=cone(index,M.read(AUTH));kwargs['cone_exact']=A
 v=K.validate_certificate(terms,s['solver'],bound=BOUND,**kwargs)
 baseline=K.objective_interval(terms,[float.fromhex(x) for x in p['old_theta_binary64']]);M.need(baseline==s['baseline_objective'],'SAVED_HEAD_LOSS_REPLAY')
 M.need(max(abs(float.fromhex(x)) for x in p['old_theta_binary64'])<=BOUND,'OLD_HEAD_FEASIBLE_BOX')
 if stage=='verify_cone':M.need(s['solver']['conflict_threshold_rational']==baseline['upper_rational'],'CONFLICT_THRESHOLD_IS_CERTIFIED_ORIGINAL_FEASIBLE_UPPER')
 for name,theta in [('original',[float.fromhex(x) for x in p['old_theta_binary64']]),('solution',s['solver']['best_theta_exact'])]:
  if theta is not None:
   for pool,tt in [('PAIR',terms[:64]),('FULL',terms[64:])]:M.need(K.objective_interval(tt,theta)==s['loss_components'][name][pool],'POOL_LOSS_COMPONENT_REPLAY')
 M.write(out/'validation.json',dict(status='CONVEX_CAUSE_EXACT_BOUNDS_REPLAY_PASS',search=M.bind(out/'search.json'),authority=M.bind(AUTH),core=M.bind(K.__file__),bounds=v,fresh_nonce=nonce,solver_calls=0,original_head_training_updates=0))

def predict():
 import torch
 import run_rc_new_hyp593_oof5_v1 as P
 v=M.read(TRAIN/'fit/validation.json');M.need(v['status']=='CONVEX_CAUSE_EXACT_BOUNDS_REPLAY_PASS' and v['search']==M.bind(TRAIN/'fit/search.json'),'VALIDATED_TRAIN_SEARCH')
 s=M.read(TRAIN/'fit/search.json')['solver'];M.need(s['theta_float_is_exact_primal'] and not s['diagnostic_only'],'TRAIN_ONLY_BINARY64_SOLUTION')
 theta=torch.tensor([float.fromhex(x) for x in s['theta_binary64']],dtype=torch.float64)
 prev=M.read(PREV/'fits/receipt.json');pv=M.read(PREV/'fits/validation.json');M.need(pv['payload']==prev['payload'] and pv['receipt']==M.bind(PREV/'fits/receipt.json') and pv['status']=='FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS','PREVIOUS_PREDICTION_SEAL')
 old=torch.load(M.checked(prev['payload']),weights_only=True,map_location='cpu');oldpred={r['query_id']:r for r in old['predictions']};rows,src=P.features();test=[r for r in rows if r['query_id'] in oldpred];pred=P.predict({'TRAIN_CONVEX7':dict(theta=theta)},test)
 for r in pred:
  M.need(all(r[k]==oldpred[r['query_id']][k] for k in ('candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')),'UNCHANGED_ACTION_AXES');r['models']['ORIGINAL7']=oldpred[r['query_id']]['models']['ORIGINAL7']
 value=dict(parameters={'TRAIN_CONVEX7':dict(theta=theta),'ORIGINAL7':old['parameters']['ORIGINAL7']},predictions=pred,feature_sources=src,train_search=M.bind(TRAIN/'fit/search.json'),train_search_validation=M.bind(TRAIN/'fit/validation.json'),previous_payload=prev['payload'],authority=M.bind(AUTH),heldout_label_reads=0,original_head_training_updates=0)
 path=TRAIN/'predictions.pt'
 with path.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
 M.write(TRAIN/'prediction_seal.json',dict(status='CONVEX_TRAIN_ONLY_ALL160_PREDICTIONS_SEALED',payload=M.bind(path),train_search=M.bind(TRAIN/'fit/search.json'),train_search_validation=M.bind(TRAIN/'fit/validation.json'),authority=M.bind(AUTH),heldout_label_reads=0,original_head_training_updates=0));print('CONVEX_TRAIN_ONLY_ALL160_PREDICTIONS_SEALED',flush=True)

def preflight():
 p,terms=problem();M.need(len(p['old_theta_binary64'])==7 and len(terms)>96,'MIXED96_TERMS');interval=K.objective_interval(terms,[float.fromhex(v) for v in p['old_theta_binary64']]);M.need(interval['lower']>0 and interval['upper']>=interval['lower'],'FINITE_ORIGINAL_LOSS')
 import torch
 import torch.nn.functional as F
 seal=M.read(TRAIN/'input_seal.json')
 for binding in seal['sources'].values():M.checked(binding)
 pack=torch.load(M.checked(seal['sources']['original_train_pack']),weights_only=True,map_location='cpu');theta=torch.tensor([float.fromhex(v) for v in p['old_theta_binary64']],dtype=torch.float64)
 px=torch.cat([r['real_native_features']['C_PAIRED'] for r in pack['pair']['records']]);py=torch.tensor([float(r['switch_label']) for r in pack['pair']['records']],dtype=torch.float64);pair_loss=(F.binary_cross_entropy_with_logits(px@theta[:6]+theta[6],py,reduction='none')*torch.where(py==0,4.,1.)).mean();ql=[]
 for row in pack['full']:
  z=row['real_native_features']['C_PAIRED']@theta[:6]+theta[6]
  if row['target_position']==row['base_winner_position']:ql.append(4*F.softplus(z.max()))
  else:
   t=row['challenger_positions'].index(row['target_position']);mask=torch.ones(len(z),dtype=torch.bool);mask[t]=False;ql.append(F.softplus(-z[t])+4*F.softplus(z[mask].max()))
 original_float=float(pair_loss+torch.stack(ql).mean());M.need(abs(original_float-interval['upper'])<1e-12,'ORIGINAL_SCALAR_LOSS_AND_CONVEX_TERMS_MATCH')
 M.write(TRAIN/'preflight.json',dict(status='CONVEX_CAUSE_PIPELINE_PREFLIGHT_PASS',authority=M.bind(AUTH),saved_ec7_objective=interval,original_scalar_fp64_loss=original_float,objective_equivalence_tolerance=1e-12,original_head_training_updates=0,optimizer_calls=0,box=BOUND,steps=CUTS,time_per_search=SECONDS,core_qualification=M.read(AUTH)['sources']['core_qualification']))
 print(json.dumps(dict(status='CONVEX_CAUSE_PIPELINE_PREFLIGHT_PASS',original_loss=interval['upper'])),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','train','diagnose','verify_train','verify_cone','predict']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--nonce');a=ap.parse_args();sourcecheck(a.stage)
 if a.stage=='preflight':preflight()
 elif a.stage=='predict':predict()
 elif a.stage.startswith('verify_'):verify(a.stage,a.index,a.nonce)
 else:solve(a.stage,a.index)
