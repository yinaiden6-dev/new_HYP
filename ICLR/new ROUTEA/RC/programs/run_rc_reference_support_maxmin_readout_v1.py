#!/usr/bin/env python3
"""Qualified max–min support readout with fixed controls and a user-requested projection-distance secondary head."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
from fractions import Fraction
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
NAME='rc_reference_support_maxmin_readout_v1'
OUT=ROOT/'results'/NAME
AUTH=ROOT/'registry/rc_reference_support_maxmin_readout_authority_v1_20260910.json'
PLAN=ROOT/'plan/RC_REFERENCE_SUPPORT_MAXMIN_READOUT_V1_20260910.md'
ADDENDUM=ROOT/'plan/RC_REFERENCE_SUPPORT_PROJECTION_DISTANCE_ADDENDUM_V2_20260910.md'
LAUNCH=ROOT/'slurm/rc_reference_support_maxmin_readout_v1_dev_cpuonly_15m.sbatch'
TEMPLATE=ROOT/'programs/run_rc_same_support_specificity_v1.py'
TEMPLATE_SHA='8d9c59f042cd37e7511114436357ab774a0ec72a8919599a8dcc8a30c80ee835'
PARENT_AUTH=ROOT/'registry/rc_same_support_specificity_authority_v1_20260910.json'
PARENT_AUTH_SHA='3106eecbbab2154c49133435b5e0866192e38d2218fc6384208d4ac4ade65418'
PRIOR=ROOT/'results/rc_same_support_specificity_v1'
CACHE=ROOT/'results/rc_reference_support_maxmin_cache_v1'
CACHE_AUTH=ROOT/'registry/rc_reference_support_maxmin_cache_authority_v1_20260910.json'
PRIOR_FILES=('parameters.json','eval_prejoin.json','eval_prejoin_seal.json','result.json','independent_validation.json')
MODELS={'ORIGINAL7':'NATIVE7','FIXED8':'FIXED8','MAXMIN8':'MAXMIN8','PROJECTION8':'PROJECTION8'}
MODES=('REAL','CBIND','EXTRA_BIND');ARM='C_PAIRED'
S=None;U=None;H=None
HASH_DEPTH=0;PRIOR_RELEASED=False;PRIOR_BLOCKED=0;ORACLE_DENIED=0

def need(value,message):
 if not bool(value):raise RuntimeError(message)

def outcome_audit(event,args):
 global PRIOR_BLOCKED,ORACLE_DENIED
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 path=Path(os.fsdecode(args[0])).resolve()
 if '/rc_opened_eval_strict_' in str(path):
  ORACLE_DENIED+=1;raise PermissionError('CAPACITY_ORACLE_DIRECTORY_FORBIDDEN')
 if path.parent==PRIOR and path.name in PRIOR_FILES and not PRIOR_RELEASED and not HASH_DEPTH:
  PRIOR_BLOCKED+=1;raise PermissionError('PRIOR_5139024_OUTCOME_READ_BEFORE_NEW_PREJOIN_SEAL')
sys.addaudithook(outcome_audit)

def hash_binding(path):
 global HASH_DEPTH
 HASH_DEPTH+=1
 try:return H.binding(path)
 finally:HASH_DEPTH-=1

def setup():
 global S,U,H
 need(hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()==TEMPLATE_SHA,'QUALIFIED_TEMPLATE_PIN')
 need(hashlib.sha256(PARENT_AUTH.read_bytes()).hexdigest()==PARENT_AUTH_SHA,'PARENT_AUTHORITY_PIN')
 spec=importlib.util.spec_from_file_location('qualified_fixed_support_source',TEMPLATE)
 S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);S.setup();U=S.U;H=S.H
 need(S.sources()==H.read(PARENT_AUTH)['sources'],'QUALIFIED_PARENT_SOURCE_CLOSURE')

def resolve_binding(value,cache_only=False):
 p=Path(value['path'])
 if not p.is_absolute():
  p=(ROOT/p) if (ROOT/p).exists() else (CACHE/p)
 p=p.resolve()
 if cache_only:need(p.is_relative_to(CACHE),'CACHE_BINDING_OUTSIDE_OUTPUT')
 need(H.sha(p)==value['sha256'],'BINDING_DRIFT:'+str(p))
 return p

def cache_sources():
 paths={k:CACHE/k for k in ('manifest.json','result.json','validation.json')}
 need(all(p.is_file() for p in paths.values()),'CACHE_NOT_COMPLETE_READOUT_CLOSED')
 manifest=H.read(paths['manifest.json']);validation=H.read(paths['validation.json'])
 need(manifest['status']=='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_COMPLETE' and manifest['all8320_gap_qualified'] is True,'CACHE_COMPLETE_ALL8320_REQUIRED')
 need(validation['status']=='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_INDEPENDENT_VALIDATION_PASS' and validation['all8320_gap_qualified'] is True,'INDEPENDENT_ALL8320_REQUIRED')
 for name,key in (('manifest.json','manifest'),('result.json','result')):
  need(resolve_binding(validation[key],True)==paths[name].resolve(),'VALIDATOR_OUTPUT_BINDING:'+key)
 need(resolve_binding(manifest['sources']['plan'])==PLAN.resolve(),'CACHE_PLAN_BINDING')
 need(validation['sources']==manifest['sources'],'CACHE_VALIDATOR_SOURCE_AGREEMENT')
 need(resolve_binding(manifest['authority'])==CACHE_AUTH.resolve(),'CACHE_AUTHORITY_BINDING')
 authority=H.read(CACHE_AUTH)
 need(authority['status']=='RC_REFERENCE_SUPPORT_MAXMIN_CACHE_V1_AUTHORIZED' and authority['sources']==manifest['sources'] and authority['contract']==manifest['contract'] and Path(authority['output']).resolve()==CACHE.resolve(),'CACHE_AUTHORITY_SOURCE_CLOSURE')
 contract=manifest['contract']
 need(contract['games']==8320 and contract['opponents_per_game']==127 and contract['exact_math']=='ORIGINAL_BINARY64_ENDPOINTS_NOT_ROUNDED_D','FROZEN_GAME_OPERATOR_CONTRACT')
 need(contract['qualification_gap']=={'numerator':'1','denominator':'100000000'} and contract['T']=='float(Fraction(exact_lower_L)).hex();nearest_binary64;retain_negative','FROZEN_CERTIFICATE_AND_ROUNDING_CONTRACT')
 for source_name in ('program','validator','launcher','pilot_operator','pilot_independent_math'):
  need(source_name in manifest['sources'],'CACHE_QUALIFICATION_SOURCE_REQUIRED:'+source_name)
 for value in manifest['sources'].values():
  if not isinstance(value,dict) or not {'path','sha256'}.issubset(value):continue
  path=Path(value['path']);path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
  if path.is_relative_to(ROOT):need(H.sha(path)==value['sha256'],'CACHE_SOURCE_DRIFT')
  else:
   need(path==S.GALLERY.resolve() and value['sha256']==S.GALLERY_SHA,'ONLY_QUALIFIED_GALLERY_EXTERNAL_SOURCE')
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
   need(h.hexdigest()==value['sha256'],'ORIGINAL_GALLERY_SOURCE_DRIFT')
 return {'qualified_cache':{k:H.binding(p) for k,p in paths.items()},'cache_authority':H.binding(CACHE_AUTH),'cache_sources':manifest['sources']}

def sources(require_cache):
 source={'program':H.binding(__file__),'plan':H.binding(PLAN),'user_PROJECTION8_addendum':H.binding(ADDENDUM),'launcher':H.binding(LAUNCH),
  'qualified_template':H.binding(TEMPLATE),'parent_authority':H.binding(PARENT_AUTH),
  'prior_5139024_outputs_hash_only':{k:hash_binding(PRIOR/k) for k in PRIOR_FILES},
  'continuation':H.binding(U.EXT),'split_qualification':H.binding(U.SPLIT)}
 if require_cache:source.update(cache_sources())
 return source

def feature_key(name,mode):
 if name=='ORIGINAL7':return S.feature_key('ORIGINAL7',mode)
 if name=='FIXED8':return S.feature_key('SPECIFIC8',mode)
 return 'maxmin_readout_'+name+'_'+mode

def prepare(require_cache):
 import torch
 pure,pair,train,evals,entries,labels,parent_closure=S.prepare()
 need(parent_closure==H.read(PRIOR/'input_closure.json'),'QUALIFIED_FIXED_SUPPORT_INPUT_REPLAY')
 families=pure['train_head'].__globals__['FAMILIES']
 need(families is pure['actions'].__globals__['FAMILIES'],'SHARED_FEATURE_DISPATCH')
 families['FIXED8']=families['SPECIFIC8']
 if not require_cache:
  return pure,pair,train,evals,entries,labels,{'status':'OLD_INPUTS_QUALIFIED_MAXMIN_CACHE_NOT_CONSUMED',
   'parent_input_closure':parent_closure,'TRAIN32_PAIR64_EVAL32_inputs_replayed':True,'cache_loaded':False,
   'new_training_updates':0,'runtime_EVAL_role_reads':0,'prior_5139024_outcome_reads':0,
   'qualification_limit':'Staged old-input check only; final preflight and training require all8320 independently qualified games.'}
 frozen=H.load_modules()[0]
 manifest=H.read(CACHE/'manifest.json');records={};record_bindings=[];certificate_bindings=[];T_certificate_checks=0
 for entry in manifest['records']:
  path=resolve_binding(entry,True);record=H.read(path)
  key=(record['kind'],record['query_id'],record['execution_ordinal'])
  need(key==(entry['kind'],entry['query_id'],entry['execution_ordinal']) and key not in records,'UNIQUE_CACHE_RECORD_KEY')
  need(record['all_game_gap_qualified'] is True,'EVERY_GAME_IN_RECORD_QUALIFIED')
  cert=resolve_binding(record['certificates'],True);cert_doc=H.read(cert)
  need((cert_doc['kind'],cert_doc['query_id'],cert_doc['execution_ordinal'])==key and cert_doc['all_game_gap_qualified'] is True,'CERTIFICATE_QUERY_BINDING')
  games=cert_doc['games'];positions=[g['candidate_position'] for g in games]
  need(positions==record['score_positions'] and len(set(positions))==len(positions),'COMPLETE_ORDERED_CERTIFICATE_AXIS')
  for game in games:
   certificate=game['certificate'];need(certificate is not None and certificate['gap_qualified'] is True,'CERTIFICATE_GAP_FLAG')
   lo=certificate['lower'];hi=certificate['upper']
   lower=Fraction(int(lo['numerator']),int(lo['denominator']));upper=Fraction(int(hi['numerator']),int(hi['denominator']))
   need(lower<=upper and upper-lower<=Fraction(1,100000000),'EXACT_CERTIFICATE_GAP_CLOSED')
   need(record['T_binary64'][str(game['candidate_position'])]==float(lower).hex(),'T_EXACT_LOWER_NEAREST_BINARY64_NO_CLAMP')
   T_certificate_checks+=1
  records[key]=record;record_bindings.append(H.binding(path));certificate_bindings.append(H.binding(cert))
 need(len(records)==128 and T_certificate_checks==8320,'CACHE_FULL64_PAIR64_ALL8320_T_CERTIFICATES')
 prior_inputs=H.read(S.INPUT/'result.json')['records']
 prior_by_key={(r['kind'],r['query_id'],r['execution_ordinal']):r for r in prior_inputs}
 ledger=[];counts={'FULL':0,'PAIR':0};positions_count={'FULL':0,'PAIR':0}
 def extend(row,kind):
  key=(kind,row['query_id'],row['execution_ordinal']);record=records.pop(key);old=prior_by_key.pop(key)
  for field in ('candidate_physical_rows','base_winner_position','challenger_positions'):
   need(record[field]==row[field],'CACHE_ORIGINAL_AXIS:'+field)
  if kind=='PAIR':
   need(record['pair_cohort']==row['pair_cohort'] and record['switch_label']==row['switch_label'],'PAIR_ORIGINAL_LABEL_COHORT')
  axis=row['candidate_physical_rows'];need(len(axis)==128 and len(set(axis))==128 and not {714,715}.issubset(set(axis)),'ORIGINAL_COMPLETE_C128')
  expected_positions=set(range(128)) if kind=='FULL' else set(row['challenger_positions'])|{row['base_winner_position']}
  need(set(record['score_positions'])==expected_positions and len(record['score_positions'])==len(expected_positions),'FULL128_PAIR2_SUPPORT_AXIS')
  t={int(k):float.fromhex(v) for k,v in record['T_binary64'].items()}
  need(set(t)==expected_positions and all(math.isfinite(v) for v in t.values()),'SIGNED_FINITE_CERTIFIED_T_AXIS')
  need(record['fixed_J_binary64']==old['specificity_margin_binary64'],'FIXED_J_EXACT_PREDECESSOR')
  fixed_j={int(k):float.fromhex(v) for k,v in record['fixed_J_binary64'].items()}
  need(set(fixed_j)==set(t),'FIXED_AND_OPTIMIZED_SUPPORT_AXIS')
  summed={position:fixed_j[position]+t[position] for position in t}
  need(all(math.isfinite(value) for value in summed.values()),'FINITE_FIXED_PLUS_OPTIMIZED_CUE')
  projected={position:abs(value)/math.sqrt(2.0) for position,value in summed.items()}
  need(all(math.isfinite(value) and value>=0 for value in projected.values()),'FINITE_PROJECTION_FOOT_DISTANCE')
  out=dict(row);row_ledger={'kind':kind,'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],
   'PROJECTION8_signed_sum_binary64':{str(position):H.hx(value) for position,value in summed.items()},
   'PROJECTION8_distance_binary64':{str(position):H.hx(value) for position,value in projected.items()},'features':{}}
  for mode in (('REAL',) if kind=='PAIR' else ('REAL','CBIND')):
   native=row[feature_key('ORIGINAL7',mode)][ARM];fixed=row[feature_key('FIXED8',mode)][ARM]
   donors={p:p for p in t} if mode=='REAL' else {p:int(row['cbind_source_positions'][p]) for p in t}
   need(set(donors.values())==set(t),'WHOLE_EVIDENCE_DONOR_PERMUTATION')
   def contrast(values):return torch.tensor([frozen.symmetric(values[donors[int(c)]],values[donors[int(row['base_winner_position'])]]) for c in row['challenger_positions']],dtype=torch.float64).unsqueeze(1)
   rebuilt_fixed=torch.cat([native,contrast(fixed_j)],dim=1)
   need(H.tensor_sha(rebuilt_fixed)==H.tensor_sha(fixed),'FIXED8_EXACT_SPECIFIC8_REBUILD')
   for name,cue in (('MAXMIN8',t),('PROJECTION8',projected)):
    value=torch.cat([native,contrast(cue)],dim=1)
    need(H.tensor_sha(value[:,:6])==H.tensor_sha(fixed[:,:6])==H.tensor_sha(native),'ONLY_SEVENTH_COLUMN_CHANGE')
    out[feature_key(name,mode)]={ARM:value}
   row_ledger['features'][mode]={name:H.tensor_sha(out[feature_key(name,mode)][ARM]) for name in MODELS}
  if kind=='FULL':
   for name in ('MAXMIN8','PROJECTION8'):
    real=out[feature_key(name,'REAL')][ARM];control=out[feature_key(name,'CBIND')][ARM]
    extra=torch.cat([real[:,:6],control[:,6:]],dim=1)
    out[feature_key(name,'EXTRA_BIND')]={ARM:extra}
    need(H.tensor_sha(extra[:,:6])==H.tensor_sha(row['real_native_features'][ARM]),'EXTRA_BIND_REAL_NATIVE6')
   row_ledger['features']['EXTRA_BIND']={name:H.tensor_sha(out[feature_key(name,'EXTRA_BIND')][ARM]) for name in MODELS}
  ledger.append(row_ledger);counts[kind]+=1;positions_count[kind]+=len(t)
  return out
 pair={**pair,'records':[extend(r,'PAIR') for r in pair['records']]}
 train=[extend(r,'FULL') for r in train];evals=[extend(r,'FULL') for r in evals]
 need(not records and not prior_by_key and counts=={'FULL':64,'PAIR':64} and positions_count=={'FULL':8192,'PAIR':128},'ALL8320_CERTIFIED_VALUES_USED')
 for name,label in (('MAXMIN8','symmetric_certified_maxmin_support_value'),('PROJECTION8','symmetric_diagonal_projection_foot_distance')):
  families[name]=(feature_key(name,'REAL'),feature_key(name,'CBIND'),tuple(frozen.FEATURE_NAMES)+(label,))
 designs={}
 for name in MODELS:
  x=torch.cat([r[feature_key(name,'REAL')][ARM] for r in pair['records']+train])
  need(x.shape==(4128,6 if name=='ORIGINAL7' else 7) and bool(torch.isfinite(x).all()),'TRAIN_PAIR_DESIGN_AXIS_FINITE')
  zeros=torch.nonzero((x==0).all(dim=0)).flatten().tolist()
  if name!='ORIGINAL7':need(6 not in zeros,'DEGENERATE_APPENDED_COLUMN')
  xb=torch.cat([x,torch.ones((len(x),1),dtype=torch.float64)],dim=1)
  designs[name]={'shape':list(x.shape),'sha256':H.tensor_sha(x),'design_rank_with_bias':int(torch.linalg.matrix_rank(xb)),
   'zero_columns':zeros,'nonzero_counts':torch.count_nonzero(x,dim=0).tolist(),'parameter_count':x.shape[1]+1,'all_finite':True}
 closure={'parent_input_closure':parent_closure,'cache_manifest':H.binding(CACHE/'manifest.json'),'cache_validation':H.binding(CACHE/'validation.json'),
  'cache_record_bindings':record_bindings,'certificate_bindings':certificate_bindings,'T_nearest_exact_lower_bound_checks':T_certificate_checks,'cache_query_counts':counts,'cache_game_counts':positions_count,
  'overlay_feature_ledger':ledger,'TRAIN_PAIR_design_ledger':designs,'new_scalar':'binary64 nearest rounding of exact certified lower bound L; signed, never clamped',
  'ORIGINAL6_and_FIXED8_exact_predecessor':True,'MAXMIN8_and_PROJECTION8_change_only_seventh_column':True,'PROJECTION8_cue':'abs(binary64_add(original J, binary64 rounded exact L))/math.sqrt(2.0), before original symmetric contrast','EXTRA_BIND_preserves_REAL_native6':True,
  'runtime_EVAL_role_reads':0,'prior_5139024_outcome_reads':0,'oracle_read_attempts':ORACLE_DENIED,'new_encoder_or_RoMa_forwards':0}
 return pure,pair,train,evals,entries,labels,closure

def fit(pure,pair,train):
 params={};expected=H.read(U.NATIVE);start=time.monotonic()
 code=(ROOT/H.PINS['old_runner'][0]).read_text();node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='train_head')
 digest=hashlib.sha256(ast.get_source_segment(code,node).encode()).hexdigest()
 for name,family in MODELS.items():
  head,loss,finite=pure['train_head'](pair,train,family,ARM);w=head.weight.detach().flatten();b=float(head.bias.detach())
  p={'weight_binary64':[H.hx(v) for v in w],'bias_binary64':H.hx(b),'parameter_sha256':pure['parameter_sha'](w,b),
   'parameter_count':len(w)+1,'loss_total_last_recorded':loss[0],'PAIR_loss_last_recorded':loss[1],'FULL_loss_last_recorded':loss[2],
   'finite_training':finite,'unchanged_training_function_sha256':digest}
  if name=='ORIGINAL7':need(p['weight_binary64']==expected['weight_binary64'] and p['bias_binary64']==expected['bias_binary64'],'ORIGINAL7_PARAMETER_REGRESSION_ABORT')
  params[name]=p
  print(json.dumps({'event':'MAXMIN_READOUT_HEAD_FIT','model':name,'seconds':time.monotonic()-start,'parameter_sha256':p['parameter_sha256']}),flush=True)
 return params

def tensors(p):return S.tensors(p)

def predict(row,param,name,mode):
 w,b=tensors(param);values=row[feature_key(name,mode)][ARM]@w+b
 need(bool(values.isfinite().all()) and len(values)==127,'FINITE_ALL127_PREDICTION')
 axis=row['candidate_physical_rows'];cs=row['challenger_positions'];winner=row['base_winner_position']
 i=max(range(127),key=lambda k:(float(values[k]),-axis[cs[k]]));switch=float(values[i])>0
 return {'final_position':int(cs[i]) if switch else int(winner),'decision':'SWITCH' if switch else 'HOLD',
  'all127_logits_binary64':[H.hx(x) for x in values]}

def summarize(params,predictions,pure,pair,train,evals,entries,labels):
 global PRIOR_RELEASED
 seal=H.read(OUT/'eval_prejoin_seal.json')
 need(seal['parameters_sha256']==H.sha(OUT/'parameters.json') and seal['eval_prejoin_sha256']==H.sha(OUT/'eval_prejoin.json'),'PREJOIN_SEAL_BINDING')
 need(H.BARRIER.blocked==0 and PRIOR_BLOCKED==0 and ORACLE_DENIED==0,'NO_PRESEAL_OUTCOME_READS')
 H.BARRIER.release();PRIOR_RELEASED=True;joined=H.role_join(evals,entries,labels)
 need(all(not ({r[k] for r in train}&{r[k] for r in joined}) for k in ('query_id','target_identity','supergroup')),'TRAIN_EVAL_DISJOINT')
 old_params=H.read(PRIOR/'parameters.json');old_preds=H.read(PRIOR/'eval_prejoin.json');old_result=H.read(PRIOR/'result.json');old_validation=H.read(PRIOR/'independent_validation.json')
 need(old_validation['status']=='SAME_SUPPORT_SPECIFICITY_INDEPENDENT_REEXECUTION_PASS' and old_validation['result_sha256']==H.sha(PRIOR/'result.json'),'PREVIOUS_5139024_VALIDATED')
 old_seal=H.read(PRIOR/'eval_prejoin_seal.json')
 need(old_seal['parameters_sha256']==old_result['parameters_sha256']==H.sha(PRIOR/'parameters.json') and old_seal['eval_prejoin_sha256']==H.sha(PRIOR/'eval_prejoin.json') and old_result['eval_prejoin_seal_sha256']==H.sha(PRIOR/'eval_prejoin_seal.json'),'PREVIOUS_PARAMETER_AND_PREDICTION_SEALS')
 prior_map={r['query_id']:r for r in old_preds};byid={r['query_id']:r for r in predictions}
 for name,oldname in (('ORIGINAL7','ORIGINAL7'),('FIXED8','SPECIFIC8')):
  need(params[name]==old_params[oldname],'EXACT_PREDECESSOR_HEAD_PARAMETERS:'+name)
  need(set(prior_map)==set(byid),'EXACT_PREDECESSOR_FULL64_AXIS')
  for q in byid:need(byid[q]['predictions'][name]==prior_map[q]['predictions'][oldname],'EXACT_PREDECESSOR_ALL3MODE_PREDICTIONS:'+name)
 acts={};metrics={};comparisons={};diagnostics={};raw_breaks={}
 for role,rows in [('TRAIN',train),('EVAL',joined)]:
  need(len(rows)==32,'WHOLE32');acts[role]={};metrics[role]={};diagnostics[role]={};raw_breaks[role]={}
  for name,family in MODELS.items():
   w,b=tensors(params[name]);acts[role][name]={}
   for mode in MODES:
    use_rows=rows if mode!='EXTRA_BIND' else [{**r,feature_key(name,'REAL'):r[feature_key(name,mode)]} for r in rows]
    aa=pure['actions'](w,b,use_rows,family,ARM,control=mode=='CBIND');acts[role][name][mode]=aa
    for row,a in zip(rows,aa):
     pred=byid[row['query_id']]['predictions'][name][mode]
     need(pred==predict(row,params[name],name,mode),'SEALED_ALL_LOGIT_REPLAY')
     need((pred['final_position'],pred['decision'])==(a['final_position'],a['decision']),'SEALED_ACTION_REPLAY')
     selected=row['challenger_positions'].index(a['proposed_challenger'])
     need(H.hx(a['switch_logit'])==pred['all127_logits_binary64'][selected],'SEALED_SELECTED_LOGIT_REPLAY')
   metrics[role][name]={mode:pure['summary'](a) for mode,a in acts[role][name].items()}
   raw_breaks[role][name]={mode:[a['query_id'] for a in aa if a['base_correct'] and not a['final_correct']] for mode,aa in acts[role][name].items()}
  need(acts[role]['ORIGINAL7']['REAL']==acts[role]['ORIGINAL7']['EXTRA_BIND'],'ORIGINAL7_EXTRA_BIND_IDENTITY')
  for name,oldname in (('ORIGINAL7','ORIGINAL7'),('FIXED8','SPECIFIC8')):
   need(acts[role][name]==old_result['actions'][role][oldname],'EXACT_PREDECESSOR_ACTIONS_ALL_MODES:'+name)
  comparisons[role]={n+'_vs_'+b:S.paired(acts[role][n]['REAL'],acts[role][b]['REAL'],rows) for n,b in [('MAXMIN8','ORIGINAL7'),('MAXMIN8','FIXED8'),('FIXED8','ORIGINAL7'),('PROJECTION8','ORIGINAL7'),('PROJECTION8','FIXED8'),('PROJECTION8','MAXMIN8')]}
  for name in MODELS:
   diagnostics[role][name]={mode:S.retention_diagnostic(acts[role][name]['REAL'],acts[role][name][mode],acts[role]['ORIGINAL7']['REAL'],rows) for mode in ('CBIND','EXTRA_BIND')}
 old_native=H.read(ROOT/H.PINS['old_result'][0])
 for mode,key in [('REAL','actions'),('CBIND','cbind_actions')]:need(H.encode(acts['EVAL']['ORIGINAL7'][mode])==H.encode(old_native['evaluations']['NATIVE7'][ARM][key]),'ORIGINAL_NATIVE7_ACTIONS_EXACT')
 pair_metrics={}
 for name in MODELS:
  w,b=tensors(params[name]);correct=sum(int((float((r[feature_key(name,'REAL')][ARM]@w+b)[0])>0)==bool(r['switch_label'])) for r in pair['records'])
  pair_metrics[name]={'query_count':64,'correct_pair_decisions':correct,'evidence_level':'labelled optimization pool, not independent evaluation'}
 primary=comparisons['EVAL']['MAXMIN8_vs_ORIGINAL7'];net_gain=primary['net']>0 and primary['group_balanced_accuracy_difference']>0
 goal=net_gain and primary['break']==0
 secondary=comparisons['EVAL']['PROJECTION8_vs_ORIGINAL7']
 projection_net=secondary['net']>0 and secondary['group_balanced_accuracy_difference']>0
 projection_goal=projection_net and secondary['break']==0
 need(metrics['EVAL']['ORIGINAL7']['REAL']['final_top1']==28 and metrics['EVAL']['ORIGINAL7']['REAL']['base_top1']==25,'QUALIFIED_BASELINE_28_RAW25')
 need(not goal or metrics['EVAL']['MAXMIN8']['REAL']['final_top1']>=29,'AT_LEAST29_FOR_USER_GOAL')
 need(not projection_goal or metrics['EVAL']['PROJECTION8']['REAL']['final_top1']>=29,'AT_LEAST29_FOR_PROJECTION_USER_GOAL')
 status='MAXMIN8_INTERNAL_GAIN_AND_PRESERVATION_CANDIDATE' if goal else 'PROJECTION8_SECONDARY_GAIN_AND_PRESERVATION_ONLY' if projection_goal else 'SUPPORT_READOUT_POSITIVE_NET_WITH_LOSSES_ONLY' if net_gain or projection_net else 'SUPPORT_READOUT_NO_INTERNAL_GAIN_AND_PRESERVATION'
 return {'status':status,'theory_name':'new HYP','authority_sha256':H.sha(AUTH),'input_closure_sha256':H.sha(OUT/'input_closure.json'),
  'parameters_sha256':H.sha(OUT/'parameters.json'),'eval_prejoin_seal_sha256':H.sha(OUT/'eval_prejoin_seal.json'),
  'metrics':metrics,'actions':acts,'comparisons':comparisons,'control_diagnostics':diagnostics,'RAW_original_correct_losses':raw_breaks,
  'PAIR_training_pool_description':pair_metrics,'primary_internal_candidate':goal,'meets_user_gain_and_preservation':{'MAXMIN8':goal,'PROJECTION8':projection_goal},
  'PROJECTION8_explicit_user_requested_secondary_candidate':projection_goal,
  'overall_net_gain_flags':{'MAXMIN8':net_gain,'PROJECTION8':projection_net},'MAXMIN8_outperforms_FIXED8_net':comparisons['EVAL']['MAXMIN8_vs_FIXED8']['net']>0,
  'ORIGINAL7_and_FIXED8_parameters_predictions_actions_exact_predecessor':True,'original7_EXTRABIND_identity':True,
  'candidate_source':'frozen original RAW full-gallery C128; full127-challenger HOLD/SWITCH',
  'model_parameter_counts':{n:p['parameter_count'] for n,p in params.items()},'candidate_count':128,'TRAIN_query_count':32,'EVAL_query_count':32,
  'EVAL_supergroup_count':len({r['supergroup'] for r in joined}),'new_encoder_or_RoMa_forwards':0,
  'task_supervision':'original retrieval identity and positive-negative pair labels only; no task spatial annotation',
  'evidence_level':'previously opened internal EVAL32, not external confirmation','HYP_GO_claimed':False,'deployment_changed':False,
  'oracle_read_attempts':ORACLE_DENIED,'prior_preseal_outcome_read_attempts':PRIOR_BLOCKED,
  'limits':['MAXMIN8 substitutes the appended margin contrast; PROJECTION8 is the user-requested secondary Euclidean foot distance abs(J+T)/sqrt(2), not the signed coordinate, a replacement primary or a selected angle.',
   'The optimized scalar is binary64 rounding of a certified feasible lower bound, with all8320 exact certificate gaps qualified.',
   'Positive support separability is not identity or user-intent proof; numerous wrong candidates may have positive optimized margins.',
   'Max-min and primal-dual algebra are standard matrix-game methods, not claimed as novel theory.',
   'Controls do not establish pixel causality or ownership; no external confirmation or deployment advance is implied.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['staged-preflight','preflight','freeze','run','validate'],required=True);phase=p.parse_args().phase
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);setup()
 staged=phase=='staged-preflight';src=sources(not staged)
 staged_path=OUT.parent/(NAME+'_staged_preflight')/H.sha(__file__)/'result.json';pre=OUT.parent/(NAME+'_preflight')/'result.json'
 if phase=='freeze':
  need(not AUTH.exists() and not OUT.exists(),'APPEND_ONLY_AUTHORITY')
  value=H.read(pre);need(value['sources']==src and value['status']=='MAXMIN_READOUT_FINAL_PREFLIGHT_PASS','FINAL_CACHE_QUALIFIED_PREFLIGHT_REQUIRED')
  H.atomic(AUTH,{'status':'MAXMIN_READOUT_AUTHORIZED','sources':src,'primary':'MAXMIN8_vs_ORIGINAL7',
   'secondary':['MAXMIN8_vs_FIXED8','FIXED8_vs_ORIGINAL7','PROJECTION8_vs_ORIGINAL7','PROJECTION8_vs_FIXED8','PROJECTION8_vs_MAXMIN8'],'models':MODELS,'modes':MODES,'steps':2000,'seed':17,
   'loss':'unchanged original PAIR_SIGN','cutoff_UTC':H.read(U.EXT)['cutoff_UTC'],'preflight':H.binding(pre),
   'user_goal':'REAL >=29, paired rescue >=1 and break0 versus ORIGINAL28, group mean direction positive'})
  print(json.dumps({'authority_sha256':H.sha(AUTH)}),flush=True);return
 pure,pair,train,evals,entries,labels,closure=prepare(not staged)
 if staged:
  H.atomic(staged_path,{'status':'MAXMIN_READOUT_STAGED_OLD_INPUTS_PASS_CACHE_PENDING','sources':src,'input_closure':closure,
   'training_updates':0,'runtime_EVAL_role_reads':0,'final_preflight_or_training_authorized':False})
  print(json.dumps({'status':'MAXMIN_READOUT_STAGED_OLD_INPUTS_PASS_CACHE_PENDING','training_updates':0}),flush=True);return
 if phase=='preflight':
  H.atomic(pre,{'status':'MAXMIN_READOUT_FINAL_PREFLIGHT_PASS','sources':src,'input_closure':closure,'training_updates':0,
   'checks':{'all8320_qualified_cache_games_required':True,'FIXED8_exact_previous_SPECIFIC8_columns':True,
    'MAXMIN8_and_PROJECTION8_only_seventh_column_changed':True,'EXTRA_BIND_only_appended_column':True,'runtime_EVAL_role_reads':0,
    'prior_5139024_parameters_outcomes_unread':True,'oracle_directory_guard_active':True}})
  print(json.dumps({'status':'MAXMIN_READOUT_FINAL_PREFLIGHT_PASS','design':closure['TRAIN_PAIR_design_ledger']}),flush=True);return
 need(H.read(AUTH)['sources']==src,'AUTHORITY_DRIFT');validate=phase=='validate'
 need(OUT.exists() if validate else not OUT.exists(),'APPEND_ONLY_OUTPUT_STATE')
 if validate:need(H.read(OUT/'input_closure.json')==closure,'INDEPENDENT_INPUT_REBUILD')
 else:H.atomic(OUT/'input_closure.json',closure)
 params=fit(pure,pair,train)
 if validate:need(H.read(OUT/'parameters.json')==params,'INDEPENDENT_PARAMETER_RETRAIN')
 else:H.atomic(OUT/'parameters.json',params)
 rows=sorted(train+evals,key=lambda r:r['execution_ordinal'])
 preds=[{'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],
  'predictions':{name:{mode:predict(r,params[name],name,mode) for mode in MODES} for name in MODELS}} for r in rows]
 if validate:need(H.read(OUT/'eval_prejoin.json')==preds,'INDEPENDENT_ALL_LOGITS_REPLAY')
 else:
  H.atomic(OUT/'eval_prejoin.json',preds)
  H.atomic(OUT/'eval_prejoin_seal.json',{'parameters_sha256':H.sha(OUT/'parameters.json'),'eval_prejoin_sha256':H.sha(OUT/'eval_prejoin.json'),
   'EVAL_count':32,'TRAIN_count':32,'candidate_count':128,'models':list(MODELS),'modes':list(MODES),
   'runtime_EVAL_target_reads':0,'prior_5139024_outcome_reads':0,'oracle_read_attempts':ORACLE_DENIED,
   'total_FULL64_mode_actions':64*len(MODELS)*len(MODES),'total_FULL64_challenger_logits':64*len(MODELS)*len(MODES)*127})
 result=summarize(params,preds,pure,pair,train,evals,entries,labels)
 if validate:
  need(H.read(OUT/'result.json')==result,'INDEPENDENT_RESULT_REPLAY')
  H.atomic(OUT/'independent_validation.json',{'status':'MAXMIN_READOUT_INDEPENDENT_REEXECUTION_PASS','result_sha256':H.sha(OUT/'result.json'),
   'checks':{'qualified_cache_and_feature_overlays_rebuilt':True,'all_four_heads_retrained_exactly':True,
    'ORIGINAL7_and_FIXED8_predecessor_parameters_predictions_actions_exact':True,'all_FULL64_three_modes_predictions_metrics_replayed':True,
    'EXTRA_BIND_only_appended_column':True,'prior_outcome_and_oracle_guards_active':True},
   'scope':'Fresh-process readout reexecution; separately qualified all8320 exact-support certificates bound by cache validation.'})
 else:H.atomic(OUT/'result.json',result)
 print(json.dumps({'status':result['status'],'validation':validate,'EVAL_REAL':{n:v['REAL'] for n,v in result['metrics']['EVAL'].items()},'comparisons':result['comparisons']['EVAL']}),flush=True)

if __name__=='__main__':main()
