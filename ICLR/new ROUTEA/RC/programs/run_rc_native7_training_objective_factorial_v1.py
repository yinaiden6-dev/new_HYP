#!/usr/bin/env python3
"""One fixed NATIVE7 2x2 loss factorial, retrieval supervision only."""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
NAME='rc_native7_training_objective_factorial_v1'
OUT=ROOT/'results'/NAME
AUTH=ROOT/'registry/rc_native7_training_objective_factorial_authority_v1_20260909.json'
PLAN=ROOT/'plan/RC_NATIVE7_TRAINING_OBJECTIVE_FACTORIAL_V1_20260909.md'
LAUNCH=ROOT/'slurm/rc_native7_training_objective_factorial_v1_dev_cpuonly_59m.sbatch'
SOURCE=ROOT/'programs/run_rc_absolute_evidence_scale_calibration_v1.py'
PARENT_AUTH=ROOT/'registry/rc_absolute_evidence_scale_calibration_authority_v1_20260909.json'
CONSTRAINTS=ROOT/'results/rc_train_action_constraint_feasibility_v1'
CONFIG={'PAIR_SIGN':(True,False),'PAIR_RANK':(True,True),'FULL_SIGN':(False,False),'FULL_RANK':(False,True)}
MODES=('REAL','CBIND')
U=None;H=None

def need(x,m):
 if not bool(x):raise RuntimeError(m)
def setup():
 global U,H
 original=json.loads(PARENT_AUTH.read_text())
 need(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original['sources']['program']['sha256'],'SOURCE_WORKER_PIN')
 spec=importlib.util.spec_from_file_location('fixed_retrieval_inputs_for_loss_factorial',SOURCE)
 U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)
 U.helper();H=U.H;U.deadline();need(U.sources()==original['sources'],'PARENT_INPUT_AUTHORITY')
 return original

def sources():
 v=H.read(CONSTRAINTS/'independent_validation.json')
 need(v['status']=='TRAIN_CONSTRAINT_INDEPENDENT_EXACT_VALIDATION_PASS' and v['result_sha256']==H.sha(CONSTRAINTS/'result.json'),'CONSTRAINT_CERTIFICATES')
 return {'program':H.binding(__file__),'plan':H.binding(PLAN),'launcher':H.binding(LAUNCH),
         'source_worker':H.binding(SOURCE),'parent_authority':H.binding(PARENT_AUTH),
         'constraints':H.binding(CONSTRAINTS/'result.json'),'constraint_validation':H.binding(CONSTRAINTS/'independent_validation.json'),
         'continuation':H.binding(U.EXT)}

def train_function(pure,name):
 keep_pair,rank=CONFIG[name]
 original=ROOT/H.PINS['old_runner'][0]
 text=original.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='train_head')
 body=ast.get_source_segment(text,node)
 old='query_losses.append(F.softplus(-values[position]) + 4 * F.softplus(values[mask].max()))'
 new='query_losses.append(F.softplus(-values[position]) + 4 * F.softplus(values[mask].max() - values[position]))'
 need(body.count(old)==1 and body.count('loss = pair_loss + full_loss')==1,'ONLY_TWO_FROZEN_LOSS_EDITS')
 if rank:body=body.replace(old,new)
 if not keep_pair:body=body.replace('loss = pair_loss + full_loss','loss = full_loss')
 digest=hashlib.sha256(body.encode()).hexdigest()
 if name=='PAIR_SIGN':return pure['train_head'],digest
 namespace=dict(pure['train_head'].__globals__)
 exec(compile('from __future__ import annotations\n'+body,str(original)+'#'+name,'exec'),namespace)
 return namespace['train_head'],digest

def e0():
 import torch
 t=torch.tensor(2.,dtype=torch.float64,requires_grad=True);u=torch.tensor(1.,dtype=torch.float64)
 sp=torch.nn.functional.softplus
 sign=sp(-t)+4*sp(u);rank=sp(-t)+4*sp(u-t)
 need(float(rank)<float(sign),'CORRECT_ACTION_NEED_NOT_MAKE_WRONG_NEGATIVE')
 need(float(sp(u-t))==float(sp((u+3)-(t+3))),'RELATIVE_COMPETITION_TRANSLATION')
 sign_grad=torch.autograd.grad(sign,t,retain_graph=True)[0];rank_grad=torch.autograd.grad(rank,t)[0]
 need(float(rank_grad)<float(sign_grad),'RELATIVE_MARGIN_PUSHES_TARGET')
 return {'status':'NATIVE7_LOSS_FACTORIAL_E0_PASS','SIGN':float(sign.detach()),'RANK':float(rank.detach()),
         'checks':{'correct_positive_target_can_beat_positive_wrong':True,'relative_term_translation_invariant':True,
                   'target_gradient_includes_competitor_margin':True}}

def fit(pure,pair,train):
 params={};started=time.monotonic();expected=H.read(U.NATIVE)
 for name in CONFIG:
  fn,digest=train_function(pure,name);head,loss,finite=fn(pair,train,'NATIVE7','C_PAIRED')
  w=head.weight.detach().flatten();bias=float(head.bias.detach())
  p={'weight_binary64':[H.hx(v) for v in w],'bias_binary64':H.hx(bias),'parameter_sha256':pure['parameter_sha'](w,bias),
     'parameter_count':7,'loss_total_last_recorded':loss[0],'PAIR_loss_last_recorded':loss[1],'FULL_loss_last_recorded':loss[2],
     'finite_training':finite,'training_function_sha256':digest,'PAIR_in_objective':CONFIG[name][0],'relative_wrong_term':CONFIG[name][1]}
  if name=='PAIR_SIGN':need(p['weight_binary64']==expected['weight_binary64'] and p['bias_binary64']==expected['bias_binary64'],'BASELINE_PARAMETER_REGRESSION_ABORT')
  params[name]=p
  print(json.dumps({'event':'LOSS_FACTORIAL_HEAD_FIT','condition':name,'seconds':time.monotonic()-started,'parameter_sha256':p['parameter_sha256']}),flush=True)
 return params

def predict(row,p,mode):return U.predict(row,p,'NATIVE7',mode)
def params_tensor(p):
 import torch
 return torch.tensor([float.fromhex(v) for v in p['weight_binary64']],dtype=torch.float64),float.fromhex(p['bias_binary64'])

def pair_with_groups(new,base,group_by_id):
 data=U.paired(new,base);groups={}
 for n,b in zip(new,base):
  key=group_by_id[n['query_id']];g=groups.setdefault(key,{'queries':0,'new_correct':0,'base_correct':0})
  g['queries']+=1;g['new_correct']+=int(n['final_correct']);g['base_correct']+=int(b['final_correct'])
 for g in groups.values():g['net']=g['new_correct']-g['base_correct']
 data['supergroups']=groups;data['group_balanced_accuracy_difference']=sum(g['net']/g['queries'] for g in groups.values())/len(groups)
 return data

def summarize(params,prejoin,train,evals,entries,labels,pure,pair):
 import torch
 seal=H.read(OUT/'eval_prejoin_seal.json')
 need(seal['parameters_sha256']==H.sha(OUT/'parameters.json') and seal['eval_prejoin_sha256']==H.sha(OUT/'eval_prejoin.json') and H.BARRIER.blocked==0,'FULL_PREJOIN_REQUIRED')
 H.BARRIER.release();joined=H.role_join(evals,entries,labels)
 need(all(not ({r[k] for r in train}&{r[k] for r in joined}) for k in ('query_id','target_identity','supergroup')),'TRAIN_EVAL_OVERLAP')
 acts={};metrics={};comparisons={};pair_metrics={}
 for role,rows in [('TRAIN',train),('EVAL',joined)]:
  need(len(rows)==32,'WHOLE32')
  acts[role]={};metrics[role]={};mapping={r['query_id']:r['supergroup'] for r in rows}
  for name in CONFIG:
   w,b=params_tensor(params[name]);acts[role][name]={mode:pure['actions'](w,b,rows,'NATIVE7','C_PAIRED',control=mode=='CBIND') for mode in MODES}
   metrics[role][name]={mode:pure['summary'](v) for mode,v in acts[role][name].items()}
  comparisons[role]={}
  for name,base in [('PAIR_RANK','PAIR_SIGN'),('FULL_SIGN','PAIR_SIGN'),('FULL_RANK','PAIR_RANK'),('FULL_RANK','FULL_SIGN'),('FULL_RANK','PAIR_SIGN')]:
   comparisons[role][name+'_vs_'+base]=pair_with_groups(acts[role][name]['REAL'],acts[role][base]['REAL'],mapping)
  for r in rows:
   for name in CONFIG:
    for mode in MODES:
     need(next(x for x in prejoin if x['query_id']==r['query_id'])['predictions'][name][mode]==predict(r,params[name],mode),'PREJOIN_REPLAY')
 old=H.read(ROOT/H.PINS['old_result'][0])
 for mode,key in [('REAL','actions'),('CBIND','cbind_actions')]:
  need(H.encode(acts['EVAL']['PAIR_SIGN'][mode])==H.encode(old['evaluations']['NATIVE7']['C_PAIRED'][key]),'OLD_NATIVE7_ACTION_REGRESSION_ABORT')
 for name in CONFIG:
  w,b=params_tensor(params[name]);correct=0
  for r in pair['records']:
   z=float((r['real_native_features']['C_PAIRED']@w+b)[0]);correct+=int((z>0)==bool(r['switch_label']))
  pair_metrics[name]={'query_count':64,'correct_pair_decisions':correct,'supervision_used_for_optimization':CONFIG[name][0]}
 primary=comparisons['EVAL']['PAIR_RANK_vs_PAIR_SIGN'];gain=primary['net']>0;no_break=primary['break']==0
 status='NATIVE7_OBJECTIVE_PRIMARY_NO_BREAK_IMPROVEMENT' if gain and no_break else 'NATIVE7_OBJECTIVE_PRIMARY_NET_GAIN_WITH_BREAKS' if gain else 'NATIVE7_OBJECTIVE_PRIMARY_NO_NET_GAIN'
 retention={name:pure['retention'](acts['EVAL'][name]['REAL'],acts['EVAL'][name]['CBIND']) for name in CONFIG}
 return {'status':status,'authority_sha256':H.sha(AUTH),'parameters_sha256':H.sha(OUT/'parameters.json'),
         'eval_prejoin_seal_sha256':H.sha(OUT/'eval_prejoin_seal.json'),'metrics':metrics,'actions':acts,'comparisons':comparisons,
         'PAIR_pool_description':pair_metrics,'C_BIND_rescue_retention':retention,
         'primary_comparison':'EVAL PAIR_RANK versus PAIR_SIGN','primary_internal_candidate':gain and no_break,
         'baseline_parameter_and_action_regression_pass':True,'head_parameter_count':7,'candidate_count':128,
         'TRAIN_query_count':32,'EVAL_query_count':32,'EVAL_supergroup_count':len({r['supergroup'] for r in joined}),
         'new_encoder_or_RoMa_forwards':0,'task_supervision':'retrieval labels only; no spatial annotation',
         'evidence_level':'Previously opened internal EVAL32, not external confirmation','HYP_GO_claimed':False,'deployment_changed':False,
         'limits':['PAIR_SIGN is the exact old seven-parameter model; other cells change only the two frozen loss factors.',
                   'FULL cells remove PAIR loss from gradients but retain reporting on that already-labelled training pool.',
                   'Secondary cells do not replace the fixed primary comparison; no seed/checkpoint/threshold selection.',
                   'TRAIN feasibility certificates do not prove EVAL improvement or require a perfect32 scientific gate.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','freeze','run','validate'],required=True);phase=p.parse_args().phase
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);setup();src=sources()
 pre=OUT.parent/(NAME+'_preflight')/'result.json'
 if phase=='freeze':
  need(not AUTH.exists() and not OUT.exists(),'APPEND_ONLY_AUTHORITY')
  v=H.read(pre);need(v['sources']==src,'PREFLIGHT_SOURCE_DRIFT')
  H.atomic(AUTH,{'status':'NATIVE7_OBJECTIVE_FACTORIAL_AUTHORIZED','sources':src,'configuration':{k:list(v) for k,v in CONFIG.items()},
               'primary':'PAIR_RANK_vs_PAIR_SIGN','steps':2000,'seed':17,'cutoff_UTC':H.read(U.EXT)['cutoff_UTC'],'preflight':H.binding(pre)})
  print(json.dumps({'authority_sha256':H.sha(AUTH)}),flush=True);return
 frozen,pure,pair,train,evals,entries,labels,closure=U.prepare()
 need(closure==H.read(ROOT/'results/rc_absolute_evidence_scale_calibration_v1/input_closure.json'),'ORIGINAL_SOURCE_REBUILD')
 if phase=='preflight':
  fns={name:train_function(pure,name)[1] for name in CONFIG}
  H.atomic(pre,{'status':'NATIVE7_OBJECTIVE_FACTORIAL_PREFLIGHT_PASS','sources':src,'function_shas':fns,'e0':e0(),
               'training_updates':0,'native_design_sha256':closure['scales_and_design']['NATIVE7']['feature_sha256'],
               'TRAIN_count':32,'EVAL_count':32,'runtime_EVAL_target_reads':0})
  print(json.dumps({'status':'PREFLIGHT_PASS','function_shas':fns,'e0':e0()}),flush=True);return
 need(H.read(AUTH)['sources']==src,'AUTHORITY_DRIFT');validate=phase=='validate'
 need(OUT.exists() if validate else not OUT.exists(),'OUTPUT_STATE')
 if validate:need(H.read(OUT/'input_closure.json')==closure,'INPUT_CLOSURE_REPLAY')
 else:H.atomic(OUT/'input_closure.json',closure)
 params=fit(pure,pair,train)
 if validate:need(H.read(OUT/'parameters.json')==params,'INDEPENDENT_PARAMETER_RETRAIN')
 else:H.atomic(OUT/'parameters.json',params)
 # Also seal TRAIN predictions, for complete independent prediction replay.
 all_rows=sorted(train+evals,key=lambda r:r['execution_ordinal'])
 preds=[{'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],
         'predictions':{name:{mode:predict(r,params[name],mode) for mode in MODES} for name in CONFIG}} for r in all_rows]
 if validate:need(H.read(OUT/'eval_prejoin.json')==preds,'INDEPENDENT_ALL_PREDICTIONS_REPLAY')
 else:
  H.atomic(OUT/'eval_prejoin.json',preds);H.atomic(OUT/'eval_prejoin_seal.json',{'parameters_sha256':H.sha(OUT/'parameters.json'),
    'eval_prejoin_sha256':H.sha(OUT/'eval_prejoin.json'),'EVAL_count':32,'TRAIN_count':32,'candidate_count':128,'runtime_EVAL_target_reads':0})
 value=summarize(params,preds,train,evals,entries,labels,pure,pair)
 if validate:
  need(H.read(OUT/'result.json')==value,'INDEPENDENT_RESULT_REPLAY')
  H.atomic(OUT/'independent_validation.json',{'status':'NATIVE7_OBJECTIVE_FACTORIAL_INDEPENDENT_REEXECUTION_PASS',
    'result_sha256':H.sha(OUT/'result.json'),'checks':{'original_source_features_rebuilt':True,'all_four_heads_retrained_exactly':True,
    'old_NATIVE7_parameters_actions_replayed':True,'all_full_C128_predictions_metrics_replayed':True},
    'scope':'New process same frozen algorithm replay, no claim of a separate optimizer or new evaluation population.'})
 else:H.atomic(OUT/'result.json',value)
 print(json.dumps({'status':value['status'],'validation':validate,'EVAL':value['metrics']['EVAL'],'comparisons':value['comparisons']['EVAL']}),flush=True)
if __name__=='__main__':main()
