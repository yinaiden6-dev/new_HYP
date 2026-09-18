#!/usr/bin/env python3
"""One retrieval-trained shared projection angle, against exact fixed PROJECTION8."""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,math,os,sys,time,types
from pathlib import Path
import torch
from torch import nn
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
NAME='rc_shared_projection_direction9_v1'
OUT=ROOT/'results'/NAME
AUTH=ROOT/'registry/rc_shared_projection_direction9_authority_v1_20260910.json'
PLAN=ROOT/'plan/RC_SHARED_PROJECTION_DIRECTION9_TRAIN_ONLY_V1_20260910.md'
LAUNCH=ROOT/'slurm/rc_shared_projection_direction9_v1_dev_cpuonly_30m.sbatch'
BASE_PROGRAM=ROOT/'programs/run_rc_reference_support_maxmin_readout_v1.py'
BASE_SHA='a0fa2490b98350c37c58974f9edd61d1848c254532cbcfc2703561e557188f6b'
BASE_AUTH=ROOT/'registry/rc_reference_support_maxmin_readout_authority_v1_20260910.json'
BASE=ROOT/'results/rc_reference_support_maxmin_readout_v1'
BASE_OUTCOMES=('parameters.json','eval_prejoin.json','eval_prejoin_seal.json','result.json','independent_validation.json')
MODELS=('FIXED_PROJECTION8','DIRECTION9');MODES=('REAL','CBIND','EXTRA_BIND');ARM='C_PAIRED'
R=S=U=H=None
RELEASED=False;HASH_DEPTH=0;BLOCKED=[]
REAL_LINEAR=nn.Linear
CONTRACT={'theory_name':'new HYP','models':list(MODELS),'modes':list(MODES),'input':'original_native6_plus_raw_Jc_Tc_Jw_Tw',
 'distance':'abs(cos(theta)*(J+T)+sin(theta)*(T-J))/math.sqrt(2.0)',
 'contrast':'(distance_c-distance_w)/(distance_c+distance_w+1e-12)',
 'shared_theta':'ONE_GLOBAL_SCALAR_FOR_ALL_QUERIES_CANDIDATES_AND_MODES','theta_initial_binary64':0.0.hex(),
 'theta_wrap_clip_or_restart':False,'fixed_theta_is_buffer':True,'learned_theta_is_parameter':True,
 'parameter_counts':{'FIXED_PROJECTION8':8,'DIRECTION9':9},'seed':17,'updates':2000,'dtype':'float64',
 'optimizer':'original_AdamW_lr0.03_weight_decay0.001_including_theta',
 'loss_and_training_order':'EXACT_ORIGINAL_train_head_CODE_PRIVATE_nn_CONSTRUCTOR_PROXY_ONLY',
 'linear_readout':'construct_seven_features_then_original_Linear_7_1_no_split_dot',
 'evaluation_inference':'frozen_theta_seven_features_then_original_matrix_at_weight_plus_bias',
 'new_encoders_or_LP_calls':0,'EVAL_targets_before_all_prediction_seals':0,'baseline_outcomes_before_all_prediction_seals':0,
 'capacity_oracle_inputs':False,'external_confirmation':False,'deployment_changed':False}

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower();bad=None
 if any(x in s for x in ('/rc_opened_eval_strict_','/rc_train_origin_linear_projection_capacity_review_v1/','/rc_reference_support_train_direction_capacity_v1/','projection_capacity','angle_capacity','direction_capacity')):bad='ANGLE_OR_EVAL_CAPACITY_ORACLE_FORBIDDEN'
 if '/rc/reports/' in s:bad='REPORT_INPUT_FORBIDDEN'
 if p.parent==BASE and p.name in BASE_OUTCOMES and not RELEASED and not HASH_DEPTH:bad='5139151_OUTCOME_BEFORE_DIRECTION_PREJOIN_SEAL'
 if bad:BLOCKED.append(str(p));raise RuntimeError(bad)
sys.addaudithook(audit)

def setup():
 global R,S,U,H
 need(hashlib.sha256(BASE_PROGRAM.read_bytes()).hexdigest()==BASE_SHA,'FROZEN_FOUR_HEAD_PROGRAM')
 spec=importlib.util.spec_from_file_location('qualified_fixed_projection_readout',BASE_PROGRAM);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
 R.setup();S=R.S;U=R.U;H=R.H
 a=H.read(BASE_AUTH);need(a['status']=='MAXMIN_READOUT_AUTHORIZED' and a['sources']==R.sources(True),'FOUR_HEAD_SOURCE_AUTHORITY')

def hash_binding(path):
 global HASH_DEPTH
 HASH_DEPTH+=1
 try:return H.binding(path)
 finally:HASH_DEPTH-=1

def sources():
 return {'program':H.binding(PROGRAM),'plan':H.binding(PLAN),'launcher':H.binding(LAUNCH),
  'four_head_program':H.binding(BASE_PROGRAM),'four_head_authority':H.binding(BASE_AUTH),
  'four_head_sources':R.sources(True),'four_head_outputs_hash_only':{n:hash_binding(BASE/n) for n in BASE_OUTCOMES},
  'continuation':H.binding(U.EXT),'operator_contract':CONTRACT}

def packed_key(mode):return 'direction9_packed_'+mode
def feature_key(name,mode):return 'direction9_frozen_features_'+name+'_'+mode

def projection_features(raw,theta):
 need(raw.dtype==torch.float64 and raw.ndim==2 and raw.shape[1]==10,'PACKED_FEATURE_DOMAIN')
 sc=raw[:,6]+raw[:,7];dc=raw[:,7]-raw[:,6]
 sw=raw[:,8]+raw[:,9];dw=raw[:,9]-raw[:,8]
 co,si=torch.cos(theta),torch.sin(theta)
 c=torch.abs(co*sc+si*dc)/math.sqrt(2.0)
 w=torch.abs(co*sw+si*dw)/math.sqrt(2.0)
 phi=(c-w)/(c+w+1e-12)
 return torch.cat([raw[:,:6],phi.unsqueeze(1)],dim=1)

class ProjectionHead(nn.Module):
 def __init__(self,learned):
  super().__init__();self.linear=REAL_LINEAR(7,1,dtype=torch.float64);self.learned=bool(learned)
  if learned:self.theta=nn.Parameter(torch.zeros((),dtype=torch.float64))
  else:self.register_buffer('theta',torch.zeros((),dtype=torch.float64))
  self.gradient_trace={'theta_gradient_calls':0,'nonzero_theta_gradient_calls':0,'first_theta_gradient_binary64':None,'max_abs_theta_gradient_binary64':0.0.hex()}
  if learned:self.theta.register_hook(self.record_gradient)
 @property
 def weight(self):return self.linear.weight
 @property
 def bias(self):return self.linear.bias
 def parameters(self,recurse=True):
  yield from self.linear.parameters(recurse=recurse)
  if self.learned:yield self.theta
 def forward(self,raw):return self.linear(projection_features(raw,self.theta))
 def record_gradient(self,gradient):
  need(bool(torch.isfinite(gradient).all()),'FINITE_THETA_GRADIENT')
  value=float(gradient.detach());tr=self.gradient_trace
  if tr['theta_gradient_calls']==0:tr['first_theta_gradient_binary64']=value.hex()
  tr['theta_gradient_calls']+=1;tr['nonzero_theta_gradient_calls']+=int(value!=0.)
  tr['max_abs_theta_gradient_binary64']=max(float.fromhex(tr['max_abs_theta_gradient_binary64']),abs(value)).hex()
  return gradient

def prepare():
 pure,pair,train,evals,entries,labels,parent=R.prepare(True)
 need(parent==H.read(BASE/'input_closure.json'),'FOUR_HEAD_INPUT_CLOSURE_EXACT')
 manifest=H.read(R.CACHE/'manifest.json');cache={}
 for entry in manifest['records']:
  record=H.read(R.resolve_binding(entry,True));key=(record['kind'],record['query_id'],record['execution_ordinal']);need(key not in cache,'UNIQUE_CACHE_KEY');cache[key]=record
 ledger=[];counts={'FULL':0,'PAIR':0};replay_count=0
 def extend(row,kind):
  nonlocal replay_count
  key=(kind,row['query_id'],row['execution_ordinal']);record=cache.pop(key)
  need(record['candidate_physical_rows']==row['candidate_physical_rows'] and record['base_winner_position']==row['base_winner_position'] and record['challenger_positions']==row['challenger_positions'],'PACKED_CANDIDATE_AXIS')
  j={int(k):float.fromhex(v) for k,v in record['fixed_J_binary64'].items()};t={int(k):float.fromhex(v) for k,v in record['T_binary64'].items()}
  need(set(j)==set(t),'MATCHED_J_T_POSITIONS');out=dict(row);mode_hashes={}
  for mode in (('REAL',) if kind=='PAIR' else MODES):
   source_mode='REAL' if mode=='EXTRA_BIND' else mode
   native=row[R.feature_key('ORIGINAL7',source_mode)][ARM]
   donors={p:p for p in j} if mode=='REAL' else {p:int(row['cbind_source_positions'][p]) for p in j}
   need(set(donors.values())==set(j),'WHOLE_EVIDENCE_DONOR_MAPPING')
   winner=donors[int(row['base_winner_position'])]
   xy=torch.tensor([[j[donors[int(c)]],t[donors[int(c)]],j[winner],t[winner]] for c in row['challenger_positions']],dtype=torch.float64)
   raw=torch.cat([native,xy],dim=1);need(bool(torch.isfinite(raw).all()),'FINITE_PACKED_INPUT')
   out[packed_key(mode)]={ARM:raw}
   fixed=projection_features(raw,torch.zeros((),dtype=torch.float64));expected=row[R.feature_key('PROJECTION8',mode)][ARM]
   need(H.tensor_sha(fixed)==H.tensor_sha(expected),'FIXED_THETA_ZERO_FEATURE_BITS:'+mode)
   need(H.tensor_sha(raw[:,:6])==H.tensor_sha(native),'ORIGINAL_NATIVE6_UNCHANGED')
   mode_hashes[mode]={'packed10':H.tensor_sha(raw),'theta0_seven':H.tensor_sha(fixed)};replay_count+=len(raw)
  counts[kind]+=1;ledger.append({'kind':kind,'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'modes':mode_hashes});return out
 pair={**pair,'records':[extend(r,'PAIR') for r in pair['records']]}
 train=[extend(r,'FULL') for r in train];evals=[extend(r,'FULL') for r in evals]
 need(not cache and counts=={'FULL':64,'PAIR':64},'FULL_INPUT_COVERAGE')
 closure={'parent_four_head_input_closure':parent,'packed_feature_ledger':ledger,'query_counts':counts,
  'fixed_theta0_feature_rows_bit_exact':replay_count,'old_native6_and_donor_axis_unchanged':True,
  'TRAIN_PAIR_order_unchanged':True,'runtime_EVAL_role_reads':0,'baseline_outcome_reads':0,'new_training_updates':0}
 return pure,pair,train,evals,entries,labels,closure

def synthetic(pure):
 raw=torch.tensor([[0.,0.,0.,0.,0.,0.,.1,.3,-.2,.1],[0.,0.,0.,0.,0.,0.,0.,.2,.4,0.],
  [1.,2.,3.,4.,5.,6.,-.8,.2,-.4,.1],[0.,0.,0.,0.,0.,0.,-.25,.25,.25,-.25]],dtype=torch.float64)
 theta=torch.zeros((),dtype=torch.float64);features=projection_features(raw,theta)
 expected=[]
 for row in raw:
  jc,tc,jw,tw=map(float,row[6:]);c=abs(jc+tc)/math.sqrt(2.0);w=abs(jw+tw)/math.sqrt(2.0)
  expected.append(list(map(float,row[:6]))+[(c-w)/(c+w+1e-12)])
 need(torch.equal(features.view(torch.int64),torch.tensor(expected,dtype=torch.float64).view(torch.int64)),'SYNTHETIC_FIXED0_LITERAL_FEATURE_BITS')
 fixed=ProjectionHead(False);learned=ProjectionHead(True)
 need(sum(p.numel() for p in fixed.parameters())==8 and sum(p.numel() for p in learned.parameters())==9,'PARAMETER_COUNTS')
 need(not isinstance(fixed.theta,nn.Parameter) and isinstance(learned.theta,nn.Parameter),'BUFFER_VS_PARAMETER')
 learned.weight.data.zero_();learned.bias.data.zero_();y=torch.tensor([[1.],[0.]],dtype=torch.float64)
 loss=torch.nn.functional.binary_cross_entropy_with_logits(learned(raw[:2]),y);loss.backward()
 need(learned.theta.grad is not None and float(learned.theta.grad)==0.,'FIRST_ZERO_WEIGHT_THETA_GRAD_CONNECTED_ZERO')
 first_projection_grad=float(learned.weight.grad[0,6]);need(math.isfinite(first_projection_grad) and first_projection_grad!=0.,'FIRST_PROJECTION_COEFFICIENT_GRADIENT_NONZERO')
 learned.zero_grad();learned.weight.data[0,6]=.25
 torch.nn.functional.binary_cross_entropy_with_logits(learned(raw[:2]),y).backward()
 need(learned.theta.grad is not None and bool(torch.isfinite(learned.theta.grad)) and float(learned.theta.grad)!=0.,'NONZERO_PROJECTION_WEIGHT_CONNECTS_THETA_GRADIENT')
 need(nn.Linear is REAL_LINEAR,'GLOBAL_TORCH_NN_UNCHANGED')
 fn=private_train_function(pure,True);need(fn.__globals__['STEPS']==pure['train_head'].__globals__['STEPS']==2000,'ORIGINAL_STEP_COUNT')
 fn.__globals__['STEPS']=0
 dummy={'records':[{packed_key('REAL'):{ARM:raw[:1]},'switch_label':True}]}
 factory_head,_,factory_finite=fn(dummy,[],'PACKED_DIRECTION',ARM)
 need(sum(p.numel() for p in factory_head.parameters())==9 and factory_finite['all_finite'] and pure['train_head'].__globals__['STEPS']==2000,'PRIVATE_CONSTRUCTOR_ZERO_UPDATE_PROBE')
 return {'status':'SHARED_DIRECTION9_SYNTHETIC_H0_PASS','fixed0_seven_features_binary64_exact':True,'negative_sum_and_zero_abs_cases':True,
  'parameter_counts':{'FIXED_PROJECTION8':8,'DIRECTION9':9},'first_theta_grad_connected_zero':True,'nonzero_theta_grad_with_nonzero_projection_weight':True,
  'gradient_trace':learned.gradient_trace,'first_projection_coefficient_gradient_binary64':first_projection_grad.hex(),
  'private_constructor_original_loop_zero_update_probe':True,'natural_training_updates':0,'optimizer_updates':0,'global_torch_nn_unchanged':True}

def train_loop_source():
 text=(ROOT/H.PINS['old_runner'][0]).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='train_head')
 return hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()

def private_train_function(pure,learned):
 original=pure['train_head'];namespace=dict(original.__globals__);families=dict(namespace['FAMILIES'])
 names=tuple(H.load_modules()[0].FEATURE_NAMES)+('J_challenger','T_challenger','J_winner','T_winner')
 families['PACKED_DIRECTION']=(packed_key('REAL'),packed_key('CBIND'),names);namespace['FAMILIES']=families
 def factory(in_features,out_features,**kwargs):
  need(in_features==10 and out_features==1 and kwargs=={'dtype':torch.float64},'ONLY_EXPECTED_ORIGINAL_CONSTRUCTOR_REPLACED')
  return ProjectionHead(learned)
 namespace['nn']=types.SimpleNamespace(Linear=factory)
 fn=types.FunctionType(original.__code__,namespace,name=original.__name__,argdefs=original.__defaults__,closure=original.__closure__)
 need(fn.__code__ is original.__code__ and nn.Linear is REAL_LINEAR,'ORIGINAL_LOOP_CODE_AND_GLOBAL_NN_PRESERVED')
 return fn

def fit(pure,pair,train):
 params={};digest=train_loop_source()
 for name in MODELS:
  started=time.monotonic();fn=private_train_function(pure,name=='DIRECTION9');head,loss,finite=fn(pair,train,'PACKED_DIRECTION',ARM)
  need(nn.Linear is REAL_LINEAR,'NO_GLOBAL_NN_MONKEYPATCH')
  w=head.weight.detach().flatten();b=float(head.bias.detach());theta=float(head.theta.detach())
  need(bool(torch.isfinite(w).all()) and math.isfinite(b) and math.isfinite(theta),'FINITE_FINAL_MODEL')
  trace=dict(head.gradient_trace)
  if name=='DIRECTION9':
   need(trace['theta_gradient_calls']==2000 and float.fromhex(trace['first_theta_gradient_binary64'])==0.,'THETA_CONNECTED_ALL_STEPS_FIRST_ZERO')
  else:need(theta==0. and trace['theta_gradient_calls']==0,'FIXED_THETA_BUFFER_UNCHANGED')
  state={'weight_binary64':[float(v).hex() for v in w],'bias_binary64':b.hex(),'theta_binary64':theta.hex()}
  linear_sha=pure['parameter_sha'](w,b)
  full_sha=linear_sha if name=='FIXED_PROJECTION8' else hashlib.sha256(H.encode(state)).hexdigest()
  params[name]={**state,'parameter_sha256':full_sha,'linear_parameter_sha256':linear_sha,
   'parameter_count':sum(p.numel() for p in head.parameters()),'loss_total_last_recorded':loss[0],
   'PAIR_loss_last_recorded':loss[1],'FULL_loss_last_recorded':loss[2],'finite_training':finite,
   'original_train_loop_source_sha256':digest,'private_constructor_proxy_only':True,
   'theta_gradient_trace':trace,'theta_learned':name=='DIRECTION9','projection_coefficient_binary64':float(w[6]).hex(),
   'axis_radians_mod_pi_for_description_only':(math.pi/4+theta)%math.pi}
  print(json.dumps({'event':'SHARED_DIRECTION_HEAD_FIT','model':name,'seconds':time.monotonic()-started,
   'parameter_sha256':full_sha,'theta_binary64':theta.hex(),'nonzero_theta_gradient_calls':trace['nonzero_theta_gradient_calls']}),flush=True)
 return params

def linear_parameters(p):
 return torch.tensor([float.fromhex(x) for x in p['weight_binary64']],dtype=torch.float64),float.fromhex(p['bias_binary64'])

def materialize_frozen_features(pure,params,pair,train,evals):
 families=pure['actions'].__globals__['FAMILIES'];ledger=[]
 for name in MODELS:
  families[name]=(feature_key(name,'REAL'),feature_key(name,'CBIND'),tuple(H.load_modules()[0].FEATURE_NAMES)+('shared_projection_contrast',))
 def extend(row,kind):
  out=dict(row);item={'kind':kind,'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'features':{}}
  for name in MODELS:
   theta=torch.tensor(float.fromhex(params[name]['theta_binary64']),dtype=torch.float64);item['features'][name]={}
   for mode in (('REAL',) if kind=='PAIR' else MODES):
    with torch.no_grad():value=projection_features(row[packed_key(mode)][ARM],theta)
    need(bool(torch.isfinite(value).all()) and value.shape[1]==7,'FINITE_FROZEN_SEVEN_FEATURES')
    if name=='FIXED_PROJECTION8':need(H.tensor_sha(value)==H.tensor_sha(row[R.feature_key('PROJECTION8',mode)][ARM]),'FINAL_FIXED0_FEATURE_REPLAY')
    out[feature_key(name,mode)]={ARM:value};item['features'][name][mode]=H.tensor_sha(value)
  ledger.append(item);return out
 pair={**pair,'records':[extend(r,'PAIR') for r in pair['records']]}
 train=[extend(r,'FULL') for r in train];evals=[extend(r,'FULL') for r in evals]
 return pair,train,evals,ledger

def predict(row,param,name,mode):
 w,b=linear_parameters(param);values=row[feature_key(name,mode)][ARM]@w+b
 need(bool(values.isfinite().all()) and len(values)==127,'FINITE_ALL127_LOGITS')
 axis=row['candidate_physical_rows'];cs=row['challenger_positions'];winner=row['base_winner_position']
 i=max(range(127),key=lambda k:(float(values[k]),-axis[cs[k]]));switch=float(values[i])>0
 return {'final_position':int(cs[i]) if switch else int(winner),'decision':'SWITCH' if switch else 'HOLD',
  'all127_logits_binary64':[float(x).hex() for x in values]}

def summarize(params,predictions,pure,pair,train,evals,entries,labels):
 global RELEASED
 seal=H.read(OUT/'eval_prejoin_seal.json')
 need(seal['parameters_sha256']==H.sha(OUT/'parameters.json') and seal['eval_prejoin_sha256']==H.sha(OUT/'eval_prejoin.json')
  and seal['frozen_features_sha256']==H.sha(OUT/'frozen_feature_ledger.json'),'NEW_PARAMETERS_ALL_PREDICTIONS_SEALED')
 need(not BLOCKED and H.BARRIER.blocked==0 and R.PRIOR_BLOCKED==0 and R.ORACLE_DENIED==0,'NO_PRESEAL_FORBIDDEN_READS')
 H.BARRIER.release();RELEASED=True
 joined=H.role_join(evals,entries,labels)
 need(all(not({r[k] for r in train}&{r[k] for r in joined}) for k in ('query_id','target_identity','supergroup')),'TRAIN_EVAL_DISJOINT')
 old_validation=H.read(BASE/'independent_validation.json');old_result=H.read(BASE/'result.json');old_params=H.read(BASE/'parameters.json')
 old_predictions=H.read(BASE/'eval_prejoin.json');old_seal=H.read(BASE/'eval_prejoin_seal.json')
 need(old_validation['status']=='MAXMIN_READOUT_INDEPENDENT_REEXECUTION_PASS' and old_validation['result_sha256']==H.sha(BASE/'result.json')
  and all(v is True for v in old_validation['checks'].values()),'FOUR_HEAD_PREDECESSOR_QUALIFIED')
 need(old_seal['parameters_sha256']==old_result['parameters_sha256']==H.sha(BASE/'parameters.json')
  and old_seal['eval_prejoin_sha256']==H.sha(BASE/'eval_prejoin.json') and old_result['eval_prejoin_seal_sha256']==H.sha(BASE/'eval_prejoin_seal.json'),'FOUR_HEAD_PREDECESSOR_SEALS')
 old_fixed=old_params['PROJECTION8'];fixed=params['FIXED_PROJECTION8']
 for k in ('weight_binary64','bias_binary64','parameter_sha256','parameter_count','loss_total_last_recorded','PAIR_loss_last_recorded','FULL_loss_last_recorded','finite_training'):
  need(fixed[k]==old_fixed[k],'FIXED8_EXACT_TRAINING_REPLAY:'+k)
 need(fixed['theta_binary64']==0.0.hex() and fixed['original_train_loop_source_sha256']==old_fixed['unchanged_training_function_sha256'],'FIXED8_CONSTRUCTOR_ONLY_CHANGE')
 previous={r['query_id']:r for r in old_predictions};current={r['query_id']:r for r in predictions}
 need(set(previous)==set(current),'SAME_FULL64_QUERY_UNIVERSE')
 for q in current:need(current[q]['predictions']['FIXED_PROJECTION8']==previous[q]['predictions']['PROJECTION8'],'FIXED8_ALL_MODES_PREDICTIONS_EXACT')
 acts={};metrics={};comparisons={};controls={};raw_losses={}
 for role,rows in (('TRAIN',train),('EVAL',joined)):
  acts[role]={};metrics[role]={};controls[role]={};raw_losses[role]={};need(len(rows)==32,'COMPLETE32_ROLE')
  for name in MODELS:
   w,b=linear_parameters(params[name]);acts[role][name]={}
   for mode in MODES:
    rr=rows if mode!='EXTRA_BIND' else [{**r,feature_key(name,'REAL'):r[feature_key(name,mode)]} for r in rows]
    aa=pure['actions'](w,b,rr,name,ARM,control=mode=='CBIND');acts[role][name][mode]=aa
    for row,action in zip(rows,aa):
     pred=current[row['query_id']]['predictions'][name][mode]
     need(pred==predict(row,params[name],name,mode),'SEALED_PREDICTIONS_REPLAY')
     need((pred['final_position'],pred['decision'])==(action['final_position'],action['decision']),'SEALED_ACTION_REPLAY')
     selected=row['challenger_positions'].index(action['proposed_challenger'])
     need(float(action['switch_logit']).hex()==pred['all127_logits_binary64'][selected],'SEALED_SELECTED_LOGIT')
   metrics[role][name]={mode:pure['summary'](aa) for mode,aa in acts[role][name].items()}
   controls[role][name]={mode:S.retention_diagnostic(acts[role][name]['REAL'],acts[role][name][mode],old_result['actions'][role]['ORIGINAL7']['REAL'],rows) for mode in ('CBIND','EXTRA_BIND')}
   raw_losses[role][name]={mode:[a['query_id'] for a in aa if a['base_correct'] and not a['final_correct']] for mode,aa in acts[role][name].items()}
  need(acts[role]['FIXED_PROJECTION8']==old_result['actions'][role]['PROJECTION8'],'FIXED8_ALL_ACTIONS_EXACT')
  comparisons[role]={
   'DIRECTION9_vs_ORIGINAL7':S.paired(acts[role]['DIRECTION9']['REAL'],old_result['actions'][role]['ORIGINAL7']['REAL'],rows),
   'DIRECTION9_vs_FIXED_PROJECTION8':S.paired(acts[role]['DIRECTION9']['REAL'],acts[role]['FIXED_PROJECTION8']['REAL'],rows),
   'FIXED_PROJECTION8_vs_ORIGINAL7':S.paired(acts[role]['FIXED_PROJECTION8']['REAL'],old_result['actions'][role]['ORIGINAL7']['REAL'],rows)}
 primary=comparisons['EVAL']['DIRECTION9_vs_ORIGINAL7'];net=primary['net']>0 and primary['group_balanced_accuracy_difference']>0
 goal=net and primary['break']==0
 need(old_result['metrics']['EVAL']['ORIGINAL7']['REAL']['final_top1']==28 and old_result['metrics']['EVAL']['ORIGINAL7']['REAL']['base_top1']==25,'ORIGINAL28_RAW25_LINEAGE')
 need(not goal or metrics['EVAL']['DIRECTION9']['REAL']['final_top1']>=29,'USER_GAIN_AND_PRESERVATION')
 pair_metrics={}
 for name in MODELS:
  w,b=linear_parameters(params[name]);correct=sum(int((float((r[feature_key(name,'REAL')][ARM]@w+b)[0])>0)==bool(r['switch_label'])) for r in pair['records'])
  pair_metrics[name]={'query_count':64,'correct_pair_decisions':correct,'scope':'original_labelled_optimization_pool'}
 status='DIRECTION9_INTERNAL_GAIN_AND_PRESERVATION_CANDIDATE' if goal else 'DIRECTION9_POSITIVE_NET_WITH_LOSSES_ONLY' if net else 'DIRECTION9_NO_INTERNAL_GAIN_AND_PRESERVATION'
 return {'status':status,'theory_name':'new HYP','authority_sha256':H.sha(AUTH),'parameters_sha256':H.sha(OUT/'parameters.json'),
  'eval_prejoin_seal_sha256':H.sha(OUT/'eval_prejoin_seal.json'),'input_closure_sha256':H.sha(OUT/'input_closure.json'),
  'metrics':metrics,'actions':acts,'comparisons':comparisons,'control_diagnostics':controls,'RAW_original_correct_losses':raw_losses,
  'original7_baseline_metrics':old_result['metrics']['EVAL']['ORIGINAL7']['REAL'],'PAIR_training_pool_description':pair_metrics,
  'meets_user_gain_and_preservation':goal,'overall_positive_net_group_direction':net,
  'FIXED_PROJECTION8_parameters_predictions_actions_exact':True,'parameter_counts':{'FIXED_PROJECTION8':8,'DIRECTION9':9},
  'learned_theta_binary64':params['DIRECTION9']['theta_binary64'],'theta_gradient_trace':params['DIRECTION9']['theta_gradient_trace'],
  'projection_coefficient_binary64':params['DIRECTION9']['projection_coefficient_binary64'],
  'candidate_source':'original_RAW_C128_all127_challenger_HOLD_SWITCH','task_training':'original_PAIR64_and_FULL_TRAIN32_retrieval_identity_supervision',
  'EVAL_scope':'previously_opened_internal32_11groups_not_external_confirmation','extra_parameter_not_new_raw_information_claim':True,
  'forbidden_read_attempts':len(BLOCKED),'new_encoder_or_LP_calls':0,'HYP_GO_claimed':False,'deployment_changed':False}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=('preflight','freeze','run','validate'));args=p.parse_args()
 torch.set_num_threads(8);torch.set_num_interop_threads(1);setup();U.deadline();src=sources()
 pre=OUT.parent/(NAME+'_preflight')/(hashlib.sha256(H.encode(src)).hexdigest()+'.json')
 if args.phase=='freeze':
  need(not AUTH.exists() and not OUT.exists(),'APPEND_ONLY_AUTHORITY');value=H.read(pre)
  need(value['status']=='SHARED_DIRECTION9_PREFLIGHT_PASS' and value['sources']==src and value['contract']==CONTRACT,'FROZEN_PREFLIGHT_REQUIRED')
  H.atomic(AUTH,{'status':'SHARED_DIRECTION9_AUTHORIZED','sources':src,'contract':CONTRACT,'preflight':H.binding(pre),
   'cutoff_UTC':H.read(U.EXT)['cutoff_UTC'],'primary':'DIRECTION9_vs_ORIGINAL7','comparison':'DIRECTION9_vs_FIXED_PROJECTION8'})
  print(json.dumps({'status':'SHARED_DIRECTION9_AUTHORIZED','authority_sha256':H.sha(AUTH)}),flush=True);return
 pure,pair,train,evals,entries,labels,closure=prepare()
 if args.phase=='preflight':
  h0=synthetic(pure);need(not BLOCKED and H.BARRIER.blocked==0,'PREFLIGHT_ROLE_AND_OUTCOME_GUARDS')
  H.atomic(pre,{'status':'SHARED_DIRECTION9_PREFLIGHT_PASS','sources':src,'contract':CONTRACT,'input_closure':closure,
   'synthetic':h0,'natural_training_updates':0,'runtime_EVAL_role_reads':0,'baseline_outcome_reads':0})
  print(json.dumps({'status':'SHARED_DIRECTION9_PREFLIGHT_PASS','fixed0_replayed_feature_rows':closure['fixed_theta0_feature_rows_bit_exact'],'preflight':H.binding(pre)}),flush=True);return
 need(H.read(AUTH)['sources']==src and H.read(AUTH)['contract']==CONTRACT,'AUTHORITY_DRIFT');validate=args.phase=='validate'
 need(OUT.exists() if validate else not OUT.exists(),'APPEND_ONLY_RUN_STATE')
 if validate:need(H.read(OUT/'input_closure.json')==closure,'INDEPENDENT_INPUT_REBUILD')
 else:H.atomic(OUT/'input_closure.json',closure)
 params=fit(pure,pair,train)
 if validate:need(H.read(OUT/'parameters.json')==params,'INDEPENDENT_TWO_HEAD_PARAMETER_RETRAIN')
 else:H.atomic(OUT/'parameters.json',params)
 pair,train,evals,feature_ledger=materialize_frozen_features(pure,params,pair,train,evals)
 if validate:need(H.read(OUT/'frozen_feature_ledger.json')==feature_ledger,'INDEPENDENT_FROZEN_THETA_FEATURES')
 else:H.atomic(OUT/'frozen_feature_ledger.json',feature_ledger)
 rows=sorted(train+evals,key=lambda r:r['execution_ordinal'])
 predictions=[{'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],
  'predictions':{name:{mode:predict(r,params[name],name,mode) for mode in MODES} for name in MODELS}} for r in rows]
 if validate:need(H.read(OUT/'eval_prejoin.json')==predictions,'INDEPENDENT_ALL_MODE_LOGITS')
 else:
  H.atomic(OUT/'eval_prejoin.json',predictions)
  H.atomic(OUT/'eval_prejoin_seal.json',{'parameters_sha256':H.sha(OUT/'parameters.json'),'eval_prejoin_sha256':H.sha(OUT/'eval_prejoin.json'),
   'frozen_features_sha256':H.sha(OUT/'frozen_feature_ledger.json'),'models':list(MODELS),'modes':list(MODES),'TRAIN_count':32,'EVAL_count':32,
   'candidate_count':128,'total_FULL64_challenger_logits':64*2*3*127,'runtime_EVAL_target_reads':0,'baseline_outcome_reads':0,'forbidden_read_attempts':len(BLOCKED)})
 result=summarize(params,predictions,pure,pair,train,evals,entries,labels)
 if validate:
  need(H.read(OUT/'result.json')==result,'INDEPENDENT_RESULT_REPLAY')
  H.atomic(OUT/'independent_validation.json',{'status':'SHARED_DIRECTION9_INDEPENDENT_REEXECUTION_PASS','result_sha256':H.sha(OUT/'result.json'),
   'checks':{'both_heads_retrained_exactly':True,'theta_in_parameter_digest':True,'fixed8_previous_parameters_predictions_actions_exact':True,
    'all_FULL64_modes_frozen_theta_features_predictions_replayed':True,'shared_direction_no_query_angles':True,'outcome_and_oracle_guards':True}})
 else:H.atomic(OUT/'result.json',result)
 print(json.dumps({'status':result['status'],'validation':validate,'EVAL_REAL':{n:v['REAL'] for n,v in result['metrics']['EVAL'].items()},'comparisons':result['comparisons']['EVAL'],
  'theta_binary64':result['learned_theta_binary64'],'theta_gradient_trace':result['theta_gradient_trace']}),flush=True)
if __name__=='__main__':main()
