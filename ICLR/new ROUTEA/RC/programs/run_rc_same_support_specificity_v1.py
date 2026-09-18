#!/usr/bin/env python3
"""Same-support reference competition versus free-content pooling, with the original readout retained."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
NAME = 'rc_same_support_specificity_v1'
OUT = ROOT/'results'/NAME
AUTH = ROOT/'registry/rc_same_support_specificity_authority_v1_20260910.json'
PLAN = ROOT/'plan/RC_SAME_SUPPORT_REFERENCE_SPECIFICITY_V1_20260910.md'
TEMPLATE = ROOT/'programs/run_rc_evidence_dispersion_vs_curve_v1.py'
TEMPLATE_SHA = '9e168c46e4e79328687d1b0d73480de91ca60267e54177d0bfd03635a706f886'
GALLERY = ROOT.parents[2]/'colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt'
GALLERY_SHA = '11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc'
LAUNCH = ROOT/'slurm/rc_same_support_specificity_v1_dev_cpuonly_59m.sbatch'
SOURCE = ROOT/'programs/run_rc_absolute_evidence_scale_calibration_v1.py'
PARENT_AUTH = ROOT/'registry/rc_absolute_evidence_scale_calibration_authority_v1_20260909.json'
INPUT = ROOT/'results/rc_same_support_specificity_inputs_v1'
INPUT_PINS = {'manifest.json':'378dbf79e540653f6bb1b555d3e1079c7153cd350ed7d689fc3b147cd728c574', 'result.json':'13ec8fa9f79282f2cd94e9b451660c99b432c0989ead279cf96c4bfdf3f94bc2', 'validation.json':'545b10d603064e2df9d746e9e16fdebdb64de2dc56048aced04c44bc76e48a75'}
PRODUCER_SHA = '4bd77d8e7aaab3f51155f2721dbba553e899b1b7e28f7dde138b586509320627'
MODELS = {'ORIGINAL7':'NATIVE7', 'SPECIFIC8':'SPECIFIC8', 'FREE8':'FREE8'}
INPUT_NAMES = {'ORIGINAL7':'ORIGINAL_C', 'SPECIFIC8':'SPECIFICITY', 'FREE8':'FREE'}
MODES = ('REAL','CBIND','EXTRA_BIND')
ARM = 'C_PAIRED'
U = None
H = None
ORACLE_DENIED = 0

def need(value, message):
 if not bool(value): raise RuntimeError(message)

def oracle_audit(event, args):
 global ORACLE_DENIED
 if event != 'open' or not args: return
 p = args[0]
 if isinstance(p, (str, bytes, os.PathLike)):
  name = os.path.abspath(os.fsdecode(p))
  if '/rc_opened_eval_strict_' in name:
   ORACLE_DENIED += 1
   raise PermissionError('EVAL_CAPACITY_ORACLE_INPUT_FORBIDDEN')

# This guard remains active before and after the outcome barrier, for every phase.
sys.addaudithook(oracle_audit)

def setup():
 global U,H
 parent = json.loads(PARENT_AUTH.read_text())
 need(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == parent['sources']['program']['sha256'], 'SOURCE_WORKER_PIN')
 spec = importlib.util.spec_from_file_location('qualified_inputs_for_specificity_trial', SOURCE)
 U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)
 U.helper(); H=U.H; U.deadline()
 need(U.sources() == parent['sources'], 'PARENT_INPUT_AUTHORITY')

def sources():
 for name,digest in INPUT_PINS.items(): need(H.sha(INPUT/name)==digest, 'SPECIFICITY_INPUT_PIN:'+name)
 manifest=H.read(INPUT/'manifest.json'); validation=H.read(INPUT/'validation.json')
 need(validation['status']=='RC_SAME_SUPPORT_SPECIFICITY_INPUTS_V1_INDEPENDENT_VALIDATION_PASS','INPUT_VALIDATION_STATUS')
 need(validation['fresh_explicit_subprocess'] and validation['independent_literal_pool_old_scalar_and_feature_arithmetic'],'INDEPENDENT_INPUT_VALIDATION')
 need(validation['PAIR_all128_free_vectors_recomputed_from_original_tokens'] and validation['FULL_all128_free_vectors_reuse_two_qualified_cache_hashes'],'COMPLETE_REFERENCE_AXIS_VALIDATION')
 for name,key in (('result.json','result'),('manifest.json','manifest')):
  value=validation[key]
  need(Path(value['path']).resolve()==(INPUT/name).resolve() and value['sha256']==H.sha(INPUT/name),'INPUT_VALIDATION_BINDING:'+key)
 need(manifest['sources']==validation['sources'],'INPUT_SOURCE_AGREEMENT')
 for value in manifest['sources'].values():
  source=Path(value['path']).resolve()
  if source.is_relative_to(ROOT): actual=H.sha(source)
  else:
   need(source==GALLERY.resolve() and value['sha256']==GALLERY_SHA,'ONLY_EXPLICIT_ORIGINAL_GALLERY_OUTSIDE_RC')
   h=hashlib.sha256()
   with source.open('rb') as f:
    for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
   actual=h.hexdigest()
  need(actual==value['sha256'],'INPUT_SOURCE_DRIFT:'+str(source))
 need(Path(manifest['sources']['plan']['path']).resolve()==PLAN.resolve() and manifest['sources']['plan']['sha256']==H.sha(PLAN),'INPUT_PLAN_BINDING')
 need(manifest['sources']['program']['sha256']==PRODUCER_SHA,'PRODUCER_PIN')
 need(H.sha(TEMPLATE)==TEMPLATE_SHA,'FROZEN_RUNNER_TEMPLATE_PIN')
 return {'program':H.binding(__file__),'plan':H.binding(PLAN),'launcher':H.binding(LAUNCH),
  'source_worker':H.binding(SOURCE),'parent_authority':H.binding(PARENT_AUTH),'qualified_template':H.binding(TEMPLATE),
  'qualified_inputs':{k:H.binding(INPUT/k) for k in INPUT_PINS},'input_producer':manifest['sources']['program'],
  'continuation':H.binding(U.EXT),'split_qualification':H.binding(U.SPLIT)}

def feature_key(name, mode):
 if name=='ORIGINAL7': return 'cbind_native_features' if mode=='CBIND' else 'real_native_features'
 return 'specificity_trial_'+name+'_'+mode

def matrix(hex_rows):
 import torch
 return torch.tensor([[float.fromhex(x) for x in row] for row in hex_rows],dtype=torch.float64)

def e0():
 import torch
 # Independent literal same-support pooling, retaining all128 reference columns.
 def pool(a,w,g,axis):
  denom=torch.sum(w).clamp_min(1e-12)
  values=torch.stack([torch.sum(w*a[h])/denom for h in range(128)])
  other=[h for h in range(128) if h!=g]
  need(len(other)==127 and g not in other,'E0_SELF_EXCLUSION_FULL127')
  score=max(float(values[h]) for h in other)
  best=min((h for h in other if float(values[h])==score),key=lambda h:axis[h])
  return float(values[g]),float(values[g])-score,best,values
 a=torch.zeros((128,2),dtype=torch.float64)
 a[0]=torch.tensor([.75,.75],dtype=torch.float64)
 a[1]=torch.tensor([1.,0.],dtype=torch.float64)
 a[2]=torch.tensor([0.,1.],dtype=torch.float64)
 w=torch.ones(2,dtype=torch.float64); axis=list(range(128))
 free,specific,strong,pooled=pool(a,w,0,axis)
 envelope=float(torch.sum(w*a[1:].max(dim=0).values)/w.sum())
 need((free,specific,strong)==(.75,.25,1) and free-envelope==-.25,'E0_POOL_BEFORE_MAX_LITERAL_PLAN')
 scaled=pool(a,w*3,0,axis)
 need(scaled[:3]==(free,specific,strong) and torch.equal(scaled[3],pooled),'E0_SAME_SUPPORT_NORMALIZATION')
 pair_a=torch.zeros_like(a); pair_a[5]=.75; pair_a[9]=.5; pair_a[127]=.875
 p5=pool(pair_a,w,5,axis); p9=pool(pair_a,w,9,axis)
 need(p5[:3]==(.75,-.125,127) and p9[:3]==(.5,-.375,127),'E0_PAIR_SUPPORT_POSITIONS_DISTINCT_FROM_LOCAL_ROWS')
 zero=pool(pair_a,torch.zeros_like(w),5,axis)
 need(zero[:2]==(0.,0.) and torch.equal(zero[3],torch.zeros(128,dtype=torch.float64)),'E0_ZERO_SUPPORT_RETAINED')
 return {'status':'E0_SAME_SUPPORT_SELF_EXCLUSION_AND_POOL_ORDER_PASS','C128':128,'other_reference_count':127,
  'free_binary64':free.hex(),'specificity_binary64':specific.hex(),'strongest_other_position':strong,
  'pool_then_max_binary64':float(pooled[strong]).hex(),'max_then_pool_wrong_margin_binary64':(free-envelope).hex(),
  'uniform_weight_scale_invariant':True,'PAIR_support_positions':[5,9],'PAIR_strongest_other_positions':[p5[2],p9[2]],
  'zero_support_retained_with_F_J_zero':True,'correct_identity_or_accuracy_claimed':False}

def prepare():
 import torch
 frozen,pure,pair,train,evals,entries,labels,parent_closure=U.prepare()
 need(parent_closure==H.read(ROOT/'results/rc_absolute_evidence_scale_calibration_v1/input_closure.json'),'ORIGINAL_SOURCE_REBUILD')
 input_result=H.read(INPUT/'result.json'); records=input_result['records']
 literal_e0=e0(); need(input_result['E0']==literal_e0 and H.read(INPUT/'validation.json')['E0']==literal_e0,'INDEPENDENT_LITERAL_OPERATOR_E0')
 index={(r['kind'],r['query_id'],r['execution_ordinal']):r for r in records}
 need(len(index)==len(records)==128,'FULL64_PAIR64_UNIQUE')
 feature_ledger=[]
 def extend(row,kind):
  s=index.pop((kind,row['query_id'],row['execution_ordinal']))
  for key in ('candidate_physical_rows','challenger_positions','base_winner_position'):
   need(s[key]==row[key],'AXIS_UNCHANGED:'+key)
  need(s['base_scores_binary64']==[H.hx(x) for x in row['base_scores']],'RAW_UNCHANGED')
  if kind=='PAIR':
   need(s['pair_cohort']==row['pair_cohort'] and s['pair_row_ordinal']==row['pair_row_ordinal'] and s['switch_label']==row['switch_label'],'PAIR_ORDER_AND_LABEL_UNCHANGED')
  else: need(s['cbind_source_positions']==row['cbind_source_positions'],'CBIND_DONOR_UNCHANGED')
  ev=row['evidence'][ARM]
  need(s['evidence_binary64']=={str(p):{k:H.hx(v) for k,v in e.items()} for p,e in ev.items()},'ORIGINAL_C4_EVIDENCE_REPLAY')
  axis=row['candidate_physical_rows']
  need(len(axis)==128 and len(set(axis))==128 and not {714,715}.issubset(set(axis)),'UNIQUE_C128_NO_EQUIVALENT_PAIR')
  refs=s['all128_reference_sources']
  need(len(refs)==128 and [r['candidate_position'] for r in refs]==list(range(128)) and [r['physical_row'] for r in refs]==axis,'FULL_AND_PAIR_COMPLETE_REFERENCE_AXIS')
  need(all(r['image_token_count']>0 and len(r['image_tokens_sha256'])==64 for r in refs),'ORIGINAL_REFERENCE_IMAGE_SOURCE_SEALS')
  ff={int(p):float.fromhex(v) for p,v in s['free_pool_mean_binary64'].items()}
  jj={int(p):float.fromhex(v) for p,v in s['specificity_margin_binary64'].items()}
  strong={int(p):int(v) for p,v in s['strongest_other_position'].items()}
  need(set(ff)==set(jj)==set(strong)==set(ev),'SOURCE_SUPPORT_POSITION_AXIS')
  need(all(math.isfinite(v) for v in list(ff.values())+list(jj.values())),'FINITE_SAME_SUPPORT_CUES')
  need(len(s['candidates'])==len(ev),'SUPPORT_CANDIDATE_METADATA')
  for candidate in s['candidates']:
   pos=candidate['candidate_position']; competitor=strong[pos]
   need(0<=competitor<128 and competitor!=pos and candidate['strongest_other_position']==competitor and candidate['strongest_other_physical_row']==axis[competitor],'SELF_EXCLUDED_OTHER_REFERENCE')
   need(candidate['free_pool_mean_binary64']==s['free_pool_mean_binary64'][str(pos)] and candidate['specificity_margin_binary64']==s['specificity_margin_binary64'][str(pos)],'CUE_METADATA_AGREEMENT')
   need(H.hx(ff[pos]-float.fromhex(candidate['strongest_other_pool_binary64']))==H.hx(jj[pos]),'SAME_POOL_DIAGONAL_MINUS_OTHER_MAX')
  out=dict(row); ledger={'kind':kind,'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'features':{}}
  for mode in (('REAL',) if kind=='PAIR' else ('REAL','CBIND')):
   native=row[feature_key('ORIGINAL7',mode)][ARM]
   donor={p:p for p in ev} if mode=='REAL' else {p:int(s['cbind_source_positions'][p]) for p in ev}
   transformed={p:ev[donor[p]] for p in ev}
   rebuilt=torch.stack([frozen.candidate_feature(row['base_scores'].tolist(),transformed,int(c),int(row['base_winner_position'])) for c in row['challenger_positions']])
   need(H.tensor_sha(rebuilt)==H.tensor_sha(native)==H.tensor_sha(matrix(s['feature_binary64'][mode]['ORIGINAL_C'])),'ORIGINAL_NATIVE6_REPLAY')
   def contrast(values):
    return torch.tensor([frozen.symmetric(values[donor[int(c)]],values[donor[int(row['base_winner_position'])]]) for c in row['challenger_positions']],dtype=torch.float64).unsqueeze(1)
   for name,extra in (('SPECIFIC8',contrast(jj)),('FREE8',contrast(ff))):
    value=torch.cat([native,extra],dim=1)
    need(H.tensor_sha(value)==H.tensor_sha(matrix(s['feature_binary64'][mode][INPUT_NAMES[name]])),'APPENDED_FEATURE_REBUILD:'+name+':'+mode)
    need(H.tensor_sha(value[:,:6])==H.tensor_sha(native),'NATIVE6_UNCHANGED')
    out[feature_key(name,mode)]={ARM:value}
   ledger['features'][mode]={name:H.tensor_sha(out[feature_key(name,mode)][ARM]) for name in MODELS}
  if kind=='FULL':
   for name in ('SPECIFIC8','FREE8'):
    real=out[feature_key(name,'REAL')][ARM]; control=out[feature_key(name,'CBIND')][ARM]
    extra=torch.cat([real[:,:6],control[:,6:]],dim=1)
    need(H.tensor_sha(extra[:,:6])==H.tensor_sha(real[:,:6]) and H.tensor_sha(extra[:,6:])==H.tensor_sha(control[:,6:]),'EXTRA_BIND_ONLY_COLUMN7')
    out[feature_key(name,'EXTRA_BIND')]={ARM:extra}
   ledger['features']['EXTRA_BIND']={name:H.tensor_sha(out[feature_key(name,'EXTRA_BIND')][ARM]) for name in MODELS}
  feature_ledger.append(ledger)
  return out
 pair={**pair,'records':[extend(r,'PAIR') for r in pair['records']]}
 train=[extend(r,'FULL') for r in train]; evals=[extend(r,'FULL') for r in evals]
 need(not index,'ALL_INPUT_RECORDS_CONSUMED')
 families=pure['train_head'].__globals__['FAMILIES']
 need(families is pure['actions'].__globals__['FAMILIES'],'SHARED_FEATURE_DISPATCH')
 for name,extra_name in (('SPECIFIC8','symmetric_same_support_reference_specificity'),('FREE8','symmetric_same_support_free_content')):
  families[name]=(feature_key(name,'REAL'),feature_key(name,'CBIND'),tuple(frozen.FEATURE_NAMES)+(extra_name,))
 designs={}
 for name in MODELS:
  x=torch.cat([r[feature_key(name,'REAL')][ARM] for r in pair['records']+train])
  need(x.shape==(4128,6 if name=='ORIGINAL7' else 7) and bool(torch.isfinite(x).all()),'TRAIN_DESIGN_AXIS_FINITE')
  zeros=torch.nonzero((x==0).all(dim=0)).flatten().tolist()
  if name!='ORIGINAL7': need(6 not in zeros,'DEGENERATE_ZERO_APPENDED_FEATURE')
  xb=torch.cat([x,torch.ones((len(x),1),dtype=torch.float64)],dim=1)
  designs[name]={'shape':list(x.shape),'sha256':H.tensor_sha(x),'design_rank_with_bias':int(torch.linalg.matrix_rank(xb)),
   'zero_columns':zeros,'nonzero_counts':torch.count_nonzero(x,dim=0).tolist(),'all_finite':True,'parameter_count':int(x.shape[1]+1)}
 closure={'parent_input_closure':parent_closure,'qualified_input_bindings':{k:H.binding(INPUT/k) for k in INPUT_PINS},
  'overlay_feature_ledger':feature_ledger,'TRAIN_PAIR_design_ledger':designs,'literal_plan_E0':literal_e0,'all_FULL_PAIR_reference_pool_axes_are_original128':True,
  'original_C4_and_native6_rebuilt_bit_exact':True,'original_PAIR_shift_definitions_retained':True,
  'EXTRA_BIND_preserves_REAL_native6':True,'runtime_EVAL_role_reads':0,'new_encoder_or_RoMa_forwards':0,
  'oracle_directory_guard_active_all_phases':True,'oracle_read_attempts':ORACLE_DENIED}
 return pure,pair,train,evals,entries,labels,closure

def fit(pure,pair,train):
 expected=H.read(U.NATIVE); params={}; start=time.monotonic()
 code=(ROOT/H.PINS['old_runner'][0]).read_text()
 node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='train_head')
 digest=hashlib.sha256(ast.get_source_segment(code,node).encode()).hexdigest()
 for name,family in MODELS.items():
  head,loss,finite=pure['train_head'](pair,train,family,ARM)
  w=head.weight.detach().flatten(); b=float(head.bias.detach())
  p={'weight_binary64':[H.hx(v) for v in w],'bias_binary64':H.hx(b),'parameter_sha256':pure['parameter_sha'](w,b),
   'parameter_count':len(w)+1,'loss_total_last_recorded':loss[0],'PAIR_loss_last_recorded':loss[1],'FULL_loss_last_recorded':loss[2],
   'finite_training':finite,'unchanged_training_function_sha256':digest}
  if name=='ORIGINAL7': need(p['weight_binary64']==expected['weight_binary64'] and p['bias_binary64']==expected['bias_binary64'],'ORIGINAL7_PARAMETER_REGRESSION_ABORT')
  params[name]=p
  print(json.dumps({'event':'SAME_SUPPORT_HEAD_FIT','model':name,'seconds':time.monotonic()-start,'parameter_sha256':p['parameter_sha256']}),flush=True)
 return params

def tensors(p):
 import torch
 return torch.tensor([float.fromhex(x) for x in p['weight_binary64']],dtype=torch.float64),float.fromhex(p['bias_binary64'])

def predict(row,p,name,mode):
 w,b=tensors(p); values=row[feature_key(name,mode)][ARM]@w+b
 need(bool(values.isfinite().all()) and len(values)==127,'FINITE_ALL127_PREDICTION')
 axis=row['candidate_physical_rows']; cs=row['challenger_positions']; winner=row['base_winner_position']
 i=max(range(127),key=lambda k:(float(values[k]),-axis[cs[k]])); switch=float(values[i])>0
 return {'final_position':int(cs[i]) if switch else int(winner),'decision':'SWITCH' if switch else 'HOLD',
  'all127_logits_binary64':[H.hx(x) for x in values]}

def paired(new,base,rows):
 value=U.paired(new,base); group_by_id={r['query_id']:r['supergroup'] for r in rows}; groups={}
 for n,b in zip(new,base):
  g=groups.setdefault(group_by_id[n['query_id']],{'queries':0,'new_correct':0,'base_correct':0})
  g['queries']+=1; g['new_correct']+=int(n['final_correct']); g['base_correct']+=int(b['final_correct'])
 for g in groups.values(): g['net']=g['new_correct']-g['base_correct']
 value['supergroups']=groups
 value['group_balanced_accuracy_difference']=sum(g['net']/g['queries'] for g in groups.values())/len(groups)
 value['MRR_difference']=sum(1/n['final_target_rank']-1/b['final_target_rank'] for n,b in zip(new,base))/len(new)
 return value

def retention_diagnostic(real,control,baseline,rows):
 byid={r['query_id']:r for r in control}
 raw_rescues=[r['query_id'] for r in real if r['final_correct'] and not r['base_correct']]
 added=U.paired(real,baseline)['rescue_query_ids']
 return {'control_vs_REAL':paired(control,real,rows),
  'RAW_rescue_query_ids':raw_rescues,'RAW_rescues_retained_query_ids':[q for q in raw_rescues if byid[q]['final_correct']],
  'RAW_rescues_lost_query_ids':[q for q in raw_rescues if not byid[q]['final_correct']],
  'new_rescues_vs_ORIGINAL7_query_ids':added,'new_rescues_retained_query_ids':[q for q in added if byid[q]['final_correct']],
  'new_rescues_lost_query_ids':[q for q in added if not byid[q]['final_correct']]}

def summarize(params,predictions,pure,pair,train,evals,entries,labels):
 seal=H.read(OUT/'eval_prejoin_seal.json')
 need(seal['parameters_sha256']==H.sha(OUT/'parameters.json') and seal['eval_prejoin_sha256']==H.sha(OUT/'eval_prejoin.json') and H.BARRIER.blocked==0 and ORACLE_DENIED==0,'PREJOIN_SEAL_REQUIRED')
 H.BARRIER.release(); joined=H.role_join(evals,entries,labels)
 need(all(not ({r[k] for r in train}&{r[k] for r in joined}) for k in ('query_id','target_identity','supergroup')),'TRAIN_EVAL_DISJOINT')
 acts={}; metrics={}; comparisons={}; diagnostics={}; raw_breaks={}
 pred_by_id={r['query_id']:r for r in predictions}
 for role,rows in [('TRAIN',train),('EVAL',joined)]:
  need(len(rows)==32,'WHOLE32'); acts[role]={}; metrics[role]={}; diagnostics[role]={}; raw_breaks[role]={}
  for name,family in MODELS.items():
   w,b=tensors(params[name]); acts[role][name]={}
   for mode in MODES:
    use_rows=rows
    if mode=='EXTRA_BIND':
     realkey=feature_key(name,'REAL'); extra=feature_key(name,mode)
     use_rows=[{**r,realkey:r[extra]} for r in rows]
    aa=pure['actions'](w,b,use_rows,family,ARM,control=mode=='CBIND'); acts[role][name][mode]=aa
    for row,a in zip(rows,aa):
     pred=pred_by_id[row['query_id']]['predictions'][name][mode]
     need(pred==predict(row,params[name],name,mode),'SEALED_PREDICTION_REPLAY')
     need(pred['final_position']==a['final_position'] and pred['decision']==a['decision'],'SEALED_ACTION_REPLAY')
     selected_index=row['challenger_positions'].index(a['proposed_challenger'])
     need(H.hx(a['switch_logit'])==pred['all127_logits_binary64'][selected_index],'SEALED_SELECTED_LOGIT_REPLAY')
   metrics[role][name]={mode:pure['summary'](a) for mode,a in acts[role][name].items()}
   raw_breaks[role][name]={mode:[a['query_id'] for a in aa if a['base_correct'] and not a['final_correct']] for mode,aa in acts[role][name].items()}
  need(acts[role]['ORIGINAL7']['REAL']==acts[role]['ORIGINAL7']['EXTRA_BIND'],'ORIGINAL7_EXTRA_BIND_IDENTITY')
  comparisons[role]={n+'_vs_'+b:paired(acts[role][n]['REAL'],acts[role][b]['REAL'],rows) for n,b in [('SPECIFIC8','ORIGINAL7'),('SPECIFIC8','FREE8'),('FREE8','ORIGINAL7')]}
  for name in MODELS:
   diagnostics[role][name]={mode:retention_diagnostic(acts[role][name]['REAL'],acts[role][name][mode],acts[role]['ORIGINAL7']['REAL'],rows) for mode in ('CBIND','EXTRA_BIND')}
 old=H.read(ROOT/H.PINS['old_result'][0])
 for mode,key in [('REAL','actions'),('CBIND','cbind_actions')]:
  need(H.encode(acts['EVAL']['ORIGINAL7'][mode])==H.encode(old['evaluations']['NATIVE7'][ARM][key]),'ORIGINAL7_ACTION_REGRESSION_ABORT')
 pair_metrics={}
 for name in MODELS:
  w,b=tensors(params[name]); correct=sum(int((float((r[feature_key(name,'REAL')][ARM]@w+b)[0])>0)==bool(r['switch_label'])) for r in pair['records'])
  pair_metrics[name]={'query_count':64,'correct_pair_decisions':correct,'evidence_level':'labelled optimization pool, not independent evaluation'}
 primary=comparisons['EVAL']['SPECIFIC8_vs_ORIGINAL7']; free=comparisons['EVAL']['FREE8_vs_ORIGINAL7']
 primary_net=primary['net']>0 and primary['group_balanced_accuracy_difference']>0
 free_net=free['net']>0 and free['group_balanced_accuracy_difference']>0
 primary_pass=primary_net and primary['break']==0
 free_pass=free_net and free['break']==0
 status='SPECIFIC8_INTERNAL_GAIN_AND_PRESERVATION_CANDIDATE' if primary_pass else 'FREE8_SECONDARY_GAIN_AND_PRESERVATION_ONLY' if free_pass else 'SAME_SUPPORT_POSITIVE_NET_WITH_LOSSES_ONLY' if primary_net or free_net else 'SAME_SUPPORT_NO_INTERNAL_GAIN_AND_PRESERVATION'
 need(ORACLE_DENIED==0,'NO_ORACLE_READ_ATTEMPTS')
 return {'status':status,'theory_name':'new HYP','authority_sha256':H.sha(AUTH),'input_closure_sha256':H.sha(OUT/'input_closure.json'),
  'parameters_sha256':H.sha(OUT/'parameters.json'),'eval_prejoin_seal_sha256':H.sha(OUT/'eval_prejoin_seal.json'),
  'metrics':metrics,'actions':acts,'comparisons':comparisons,'control_diagnostics':diagnostics,'RAW_original_correct_losses':raw_breaks,
  'PAIR_training_pool_description':pair_metrics,'primary_internal_candidate':primary_pass,
  'primary_no_break_improvement':primary_pass,'FREE8_secondary_internal_candidate':free_pass,
  'secondary_overall_net_gain_flags':{'SPECIFIC8':primary_net,'FREE8':free_net},
  'meets_user_gain_and_preservation':{'SPECIFIC8':primary_pass,'FREE8':free_pass},
  'user_goal_plan_sha256':H.sha(PLAN),
  'SPECIFIC8_outperforms_FREE8_net':comparisons['EVAL']['SPECIFIC8_vs_FREE8']['net']>0,
  'baseline_parameter_and_action_regression_pass':True,'original7_EXTRABIND_identity':True,
  'candidate_source':'frozen original RAW full-gallery C128; full127-challenger HOLD/SWITCH',
  'model_parameter_counts':{n:p['parameter_count'] for n,p in params.items()},'candidate_count':128,'TRAIN_query_count':32,'EVAL_query_count':32,
  'EVAL_supergroup_count':len({r['supergroup'] for r in joined}),'new_encoder_or_RoMa_forwards':0,
  'task_supervision':'retrieval identity and original positive-negative pair labels only; no task spatial annotation',
  'evidence_level':'previously opened internal EVAL32, not external confirmation','HYP_GO_claimed':False,'deployment_changed':False,
  'oracle_read_attempts':ORACLE_DENIED,'limits':['SPECIFIC8 adds full128 same-support strongest-other-reference competition; FREE8 adds the same-pool diagonal without competition.',
   'Equal parameter counts do not establish equal function classes or isolate effective capacity.',
   'EXTRA_BIND tests the appended feature candidate association; it is not a pixel intervention or spatial ownership test.',
   'The primary user goal requires gain and preservation of all ORIGINAL7 correct queries; positive net with losses is a separately reported tradeoff.',
   'No evaluation-driven seed, epoch, threshold, loss or feature selection; all three predefined heads retained.']}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['preflight','freeze','run','validate'],required=True); phase=p.parse_args().phase
 import torch
 torch.set_num_threads(8); torch.set_num_interop_threads(1); setup(); src=sources(); pre=OUT.parent/(NAME+'_preflight')/'result.json'
 if phase=='freeze':
  need(not AUTH.exists() and not OUT.exists(),'APPEND_ONLY_AUTHORITY')
  v=H.read(pre); need(v['sources']==src and v['status']=='SAME_SUPPORT_SPECIFICITY_PREFLIGHT_PASS','PREFLIGHT_REQUIRED')
  H.atomic(AUTH,{'status':'SAME_SUPPORT_SPECIFICITY_TRIAL_AUTHORIZED','sources':src,'primary':'SPECIFIC8_vs_ORIGINAL7',
   'secondary':['SPECIFIC8_vs_FREE8','FREE8_vs_ORIGINAL7'],'models':MODELS,'modes':MODES,'steps':2000,'seed':17,
   'loss':'unchanged original PAIR_SIGN','cutoff_UTC':H.read(U.EXT)['cutoff_UTC'],'preflight':H.binding(pre),
   'primary_internal_rule':'paired net > 0, break == 0 versus ORIGINAL7, and equal-supergroup accuracy difference > 0',
   'current_user_goal_rule':'paired net > 0, no paired breaks versus ORIGINAL7, and equal-supergroup accuracy difference > 0'})
  print(json.dumps({'authority_sha256':H.sha(AUTH)}),flush=True); return
 pure,pair,train,evals,entries,labels,closure=prepare()
 if phase=='preflight':
  H.atomic(pre,{'status':'SAME_SUPPORT_SPECIFICITY_PREFLIGHT_PASS','sources':src,'input_closure':closure,'training_updates':0,
   'runtime_EVAL_target_reads':0,'checks':{'original_source_rebuilt':True,'both_appended_columns_rebuilt':True,'all_original_native6_unchanged':True,
    'EXTRA_BIND_only_appended_column':True,'literal_plan_E0_pass':True,'no_degenerate_appended_column':True,'oracle_directory_guard_active':True}})
  print(json.dumps({'status':'SAME_SUPPORT_SPECIFICITY_PREFLIGHT_PASS','design':closure['TRAIN_PAIR_design_ledger']}),flush=True); return
 need(H.read(AUTH)['sources']==src,'AUTHORITY_DRIFT'); validate=phase=='validate'
 need(OUT.exists() if validate else not OUT.exists(),'APPEND_ONLY_OUTPUT_STATE')
 if validate: need(H.read(OUT/'input_closure.json')==closure,'INDEPENDENT_INPUT_REBUILD')
 else: H.atomic(OUT/'input_closure.json',closure)
 params=fit(pure,pair,train)
 if validate: need(H.read(OUT/'parameters.json')==params,'INDEPENDENT_PARAMETER_RETRAIN')
 else: H.atomic(OUT/'parameters.json',params)
 rows=sorted(train+evals,key=lambda r:r['execution_ordinal'])
 preds=[{'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],
  'predictions':{name:{mode:predict(r,params[name],name,mode) for mode in MODES} for name in MODELS}} for r in rows]
 if validate: need(H.read(OUT/'eval_prejoin.json')==preds,'INDEPENDENT_ALL_LOGITS_REPLAY')
 else:
  H.atomic(OUT/'eval_prejoin.json',preds)
  H.atomic(OUT/'eval_prejoin_seal.json',{'parameters_sha256':H.sha(OUT/'parameters.json'),'eval_prejoin_sha256':H.sha(OUT/'eval_prejoin.json'),
   'EVAL_count':32,'TRAIN_count':32,'candidate_count':128,'models':list(MODELS),'modes':list(MODES),'runtime_EVAL_target_reads':0,'oracle_read_attempts':ORACLE_DENIED})
 result=summarize(params,preds,pure,pair,train,evals,entries,labels)
 if validate:
  need(H.read(OUT/'result.json')==result,'INDEPENDENT_RESULT_REPLAY')
  H.atomic(OUT/'independent_validation.json',{'status':'SAME_SUPPORT_SPECIFICITY_INDEPENDENT_REEXECUTION_PASS','result_sha256':H.sha(OUT/'result.json'),
   'checks':{'qualified_inputs_and_both_feature_overlays_rebuilt':True,'all_three_heads_retrained_exactly':True,
    'original_NATIVE7_parameters_REAL_CBIND_actions_exact':True,'all_FULL64_three_modes_predictions_metrics_replayed':True,
    'EXTRA_BIND_only_appended_column':True,'oracle_directory_guard_active':True},
   'scope':'fresh-process exact reexecution; full128 same-support cue arithmetic and old scalar replay independently qualified before fitting'})
 else: H.atomic(OUT/'result.json',result)
 print(json.dumps({'status':result['status'],'validation':validate,'EVAL_REAL':{n:v['REAL'] for n,v in result['metrics']['EVAL'].items()},'comparisons':result['comparisons']['EVAL']}),flush=True)

if __name__=='__main__': main()
