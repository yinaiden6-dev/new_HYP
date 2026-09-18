#!/usr/bin/env python3
"""Small post-result review of the fixed TRAIN96/TRAIN128 readouts; never fits.

Requires completed producer validation. Reconstructs every prediction before
opening EVAL roles or result, then separately reviews both panels and baselines.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve()
OUT = ROOT/'results/rc_original7_train128_readout_v1'
AUTH = ROOT/'registry/rc_original7_train128_readout_authority_v1_20260910.json'
READOUT = ROOT/'programs/run_rc_original7_train128_readout_v1.py'
READOUT_SHA = '88b981c1394e2948c19d4ff2728eee7976654eb646be38bc6a862a962fcb7af7'
PLAN = ROOT/'plan/RC_ORIGINAL7_TRAIN96_TRAIN128_FIXED_RECIPE_READOUT_V1_20260910.md'
PLAN_SHA = '716e590d220799b8af12f0e0b104bf0a29b612b149f13d940b65021cfb3b055c'
HELPER = ROOT/'programs/review_rc_original7_eval128_full_evidence_v1.py'
HELPER_SHA = '4ed80c0d155dced8a0d9717a9bdde800ba667ff2b93bd9908623107e00232dff'
SHARED = ROOT/'results/rc_shared_query_target_prior_cache_v1'
E128 = ROOT/'results/rc_original7_eval128_full_evidence_v1'
E128_META = ROOT/'results/rc_original7_expanded_eval128_manifest_v1'
OLD_ROLES = ROOT/'results/cw0_rgh_xf_v2_p0_a0_manifest_v2'
REPORT = ROOT/'reports/REPORT_RC_ORIGINAL7_TRAIN128_READOUT_INDEPENDENT_REVIEW_V1_20260910.md'
MODELS = ('ORIGINAL7','FULL96','FULL128')
MODES = ('REAL','CBIND')
COUNTS = {'EVAL32':32,'EVAL128':128}
BASE_SHA = 'ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'
TRAIN_FN_SHA = '45921b88203ef7065d824bdaab8e5468fc6d617dd3cb6d3fee213dfd2ccc7403'
C4 = ('real_score','visibility_mass','query_control_score','reference_control_score')
RELEASED=False; HASH_DEPTH=0; BLOCKED=[]

def need(v,m):
 if not bool(v): raise RuntimeError(m)
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
 if any(t in s for t in ('d1_mi','d1-mi','grozi','gisc_prerecall_universe','/rc_opened_eval_strict_','direction_capacity','angle_capacity')):
  BLOCKED.append(str(p));raise PermissionError('PROTECTED_INPUT_FORBIDDEN')
 if not HASH_DEPTH and not RELEASED and (p==OUT/'result.json' or p.name=='curator_roles.json' or '/role_shards/' in s or '/target_join/' in s or p in {OLD_ROLES/'role_manifest.json',OLD_ROLES/'independent_validation.json'} or (p.name in ('result.json','independent_review.json') and p.is_relative_to(ROOT/'results') and not p.is_relative_to(ROOT/'results/rc_original7_train128_readout_v1_preflight'))):
  BLOCKED.append(str(p));raise PermissionError('ROLE_OR_OUTCOME_BEFORE_INDEPENDENT_PREDICTION_CLOSURE')
sys.addaudithook(audit)
def sha(p):
 global HASH_DEPTH
 HASH_DEPTH+=1
 try:
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for block in iter(lambda:f.read(8<<20),b''):h.update(block)
  return h.hexdigest()
 finally: HASH_DEPTH-=1
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else ROOT/p
 need(sha(p)==v['sha256'],'SOURCE_SHA:'+str(p));return p.resolve()
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def hx(v):return float(v).hex()
def tsha(v):
 import torch
 v=v.detach().cpu().contiguous();return hashlib.sha256(str(v.dtype).encode()+json.dumps(list(v.shape),separators=(',',':')).encode()+v.view(torch.uint8).numpy().tobytes()).hexdigest()
def append(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():need(p.read_bytes()==data,'APPEND_ONLY_REVIEW_DRIFT');return
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o444)

def math_helpers():
 need(sha(HELPER)==HELPER_SHA,'INDEPENDENT_MATH_SOURCE_PIN')
 text=HELPER.read_text();tree=ast.parse(text)
 names={'gallery_labels','independent_feature_matrix','independent_prediction','metrics','compare','signflip_independent','eq'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 need({n.name for n in nodes}==names,'PURE_REVIEW_HELPERS')
 action=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='action')
 body=ast.get_source_segment(text,action)
 old='len(rank_labels)==len(set(rank_labels))==5412'
 need(body.count(old)==1,'ONE_EXPLICIT_RANK_LENGTH_ADAPTATION')
 # Old32 supplies its complete natural C128 ranking; all targets are present.
 # EVAL128 supplies the full5412 ranking. Caller enforces each exact size.
 body=body.replace(old,'len(rank_labels)==len(set(rank_labels))==len(raw_rank)')
 nodes.append(ast.parse(body).body[0])
 space={'ROOT':ROOT,'sys':sys,'os':os,'Path':Path,'math':math,'Fraction':Fraction,
        'defaultdict':defaultdict,'need':need,'hx':hx,'C4':C4}
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(HELPER)+'#TRAIN128_REVIEW', 'exec'),space)
 return space

def source_and_seals():
 # No parameters, features, role payload or result is opened before this gate.
 v=read(OUT/'validation.json')
 need(v['status']=='ORIGINAL7_TRAIN128_READOUT_FRESH_RESULT_REPLAY_PASS' and v['explicit_fresh_subprocess'] and v['all_actions_metrics_correct_sets_and_controls_replayed'] and v['new_training_updates']==0,'COMPLETED_PRODUCER_VALIDATION_REQUIRED')
 need(checked(v['result'])==(OUT/'result.json').resolve() and checked(v['authority'])==AUTH.resolve(),'COMPLETED_RESULT_BINDINGS')
 need(sha(READOUT)==READOUT_SHA and sha(PLAN)==PLAN_SHA,'FROZEN_READOUT_PLAN')
 a=read(AUTH);need(a['status']=='ORIGINAL7_TRAIN128_READOUT_AUTHORIZED','READOUT_AUTHORITY')
 contract=a['contract'];need(contract['models']==list(MODELS) and contract['primary_model']=='FULL128' and contract['FULL_sets']=={'ORIGINAL7':32,'FULL96':96,'FULL128':128} and contract['seed']==17 and contract['steps']==2000 and contract['training_function_sha256']==TRAIN_FN_SHA,'PREDECLARED_MODELS_AND_RECIPE')
 need(checked(a['source_bindings']['program'])==READOUT.resolve() and checked(a['source_bindings']['plan'])==PLAN.resolve(),'AUTHORITY_PROGRAM_PLAN')
 for value in a['source_bindings'].values():checked(value)
 for value in a['postjoin_source_bindings'].values():checked(value)  # Hashes only before release.
 fs=read(OUT/'parameter_seal.json');fv=read(OUT/'fit_validation.json');ps=read(OUT/'prediction_seal.json');pv=read(OUT/'prediction_validation.json')
 need(fs['status']=='TRAIN128_ALL_THREE_PARAMETERS_SEALED_BEFORE_EVAL_FEATURES' and fv['status']=='TRAIN128_FRESH_INDEPENDENT_REFIT_PASS','PARAMETER_AND_REFIT_QUALIFICATION')
 need(fs['authority']==fv['authority']==ps['authority']==pv['authority']==bind(AUTH),'ALL_SEALS_SAME_AUTHORITY')
 need(fs['models']==list(MODELS) and fs['primary_model']=='FULL128' and fv['all_three_heads_bit_exact'] and fv['explicit_fresh_subprocess'] and fv['original7_parameter_sha256']==BASE_SHA and fv['training_updates']==6000,'FRESH_ALL_THREE_REFIT')
 need(fv['parameter_seal']==bind(OUT/'parameter_seal.json') and fv['parameters']==fs['parameters']==v['parameters']==bind(OUT/'parameters.json'),'PARAMETER_DIGEST_CHAIN')
 need(fv['training_input_closure']==fs['training_input_closure']==bind(OUT/'training_input_closure.json'),'TRAIN_CLOSURE_CHAIN')
 need(ps['status']=='TRAIN128_ALL160_ALLTHREE_REAL_CBIND_PREDICTIONS_SEALED' and pv['status']=='TRAIN128_FRESH_ALL160_PREDICTION_REPLAY_PASS','PREDICTIONS_AND_FRESH_REPLAY')
 need(ps['models']==list(MODELS) and ps['modes']==list(MODES) and ps['panels']==COUNTS and ps['all127_logit_count']==pv['all127_logit_count']==121920 and pv['explicit_fresh_subprocess'],'ALL160_THREE_HEADS_TWO_MODES')
 for name in ('parameters','parameter_seal','fit_validation','predictions'):need(ps[name]==bind(OUT/(name+'.json')),'PREDICTION_SEAL_COMPONENT:'+name)
 need(pv['prediction_seal']==bind(OUT/'prediction_seal.json') and pv['predictions']==ps['predictions'] and v['fit_validation']==bind(OUT/'fit_validation.json') and v['prediction_validation']==bind(OUT/'prediction_validation.json'),'FRESH_REPLAY_DIGEST_CHAIN')
 need(fs['EVAL_features_roles_or_outcomes_read']==fv['EVAL_features_roles_or_outcomes_read']==ps['EVAL_role_outcome_reads']==pv['EVAL_role_outcome_reads']==0 and all(z['forbidden_read_attempts']==0 for z in (fs,fv,ps,pv)),'SOURCE_BARRIERS_CLOSED')
 return a,v

def old32_bank(authority):
 import numpy as np
 m=read(checked(authority['source_bindings']['shared_manifest']));v=read(checked(authority['source_bindings']['shared_validation']))
 need(m['status']=='RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_COMPLETE' and v['manifest_sha256']==sha(SHARED/'manifest.json') and all(x is True for x in v['checks'].values()),'OLD_SHARED_INPUT_QUALIFIED')
 entries=sorted((x for x in m['records'] if x['kind']=='FULL' and x['role']=='EVAL'),key=lambda x:x['execution_ordinal']);need(len(entries)==32,'OLD_EVAL32')
 rows=[]
 for entry in entries:
  path=checked(entry);meta=read(path);ap=checked(meta['arrays']);prefix='shared_EVAL_'+str(entry['execution_ordinal'])
  need(checked(authority['source_bindings'][prefix+'_metadata'])==path and checked(authority['source_bindings'][prefix+'_arrays'])==ap,'EVAL32_AUTHORITY_ARRAYS')
  with np.load(ap,allow_pickle=False) as arrays:raw=arrays['raw_scores'].tolist()
  axis=meta['axis'];order=sorted(range(128),key=lambda i:(-raw[i],axis[i]));winner=order[0]
  need(len(axis)==128 and winner==meta['winner'] and meta['kind']=='FULL' and meta['role']=='EVAL' and meta['original_C_scalars_bit_exact'],'OLD32_NATIVE_C128')
  rows.append({'query_id':meta['query_id'],'execution_ordinal':meta['execution_ordinal'],'track':'OLD_PANEL',
   'candidate_physical_rows':axis,'candidate_raw_scores_binary64':[hx(x) for x in raw],
   'base_winner_position':winner,'challenger_positions':[i for i in range(128) if i!=winner],
   'C4_binary64':{str(c['candidate_position']):c['old_scalars_binary64'] for c in meta['candidates']},
   'raw_ranked_physical_rows':[axis[i] for i in order], 'source_image_sha256':meta['query_source_image_sha256']})
 return rows

def replay_predictions(a,h):
 import torch,numpy as np
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 pref=read(checked(a['preflight']));need(pref['status']=='TRAIN128_READOUT_PREFLIGHT_READY' and pref['runtime']['torch']==str(torch.__version__) and pref['runtime']['numpy']==np.__version__,'NUMERIC_RUNTIME')
 params=read(OUT/'parameters.json');need(set(params)==set(MODELS),'EXACT_THREE_PARAMETERS')
 tensors={}
 for name,p in params.items():
  w=torch.tensor([float.fromhex(x) for x in p['weight_binary64']],dtype=torch.float64);b=float.fromhex(p['bias_binary64'])
  need(w.shape==(6,) and bool(torch.isfinite(w).all()) and math.isfinite(b) and p['parameter_count']==7 and p['training_function_sha256']==TRAIN_FN_SHA,'PARAMETER_DOMAIN_AND_RECIPE')
  need(hashlib.sha256(encode({'weight':list(map(float,w)),'bias':b})).hexdigest()==p['parameter_sha256'],'PARAMETER_SHA')
  finite=p['finite_training'];need(finite['all_finite'] and finite['finite_loss_step_count']==finite['finite_gradient_step_count']==2000 and finite['finite_parameter_state_count']==2001,'FULL2000STEP_FIT')
  tensors[name]=(w,b)
 need(params['ORIGINAL7']['parameter_sha256']==BASE_SHA,'ORIGINAL7_EC7_REGRESSION')
 small=old32_bank(a);large=read(checked(a['source_bindings']['EVAL128_prejoin_records']))
 banks={'EVAL32':small,'EVAL128':large};preds=read(OUT/'predictions.json');need(set(preds)==set(COUNTS),'TWO_PANELS')
 logits=actions=0
 for panel,rows in banks.items():
  need(len(rows)==len(preds[panel])==COUNTS[panel] and len({r['query_id'] for r in rows})==COUNTS[panel],'COMPLETE_PANEL')
  for row,sealed in zip(rows,preds[panel],strict=True):
   need(row['query_id']==sealed['query_id'] and row['execution_ordinal']==sealed['execution_ordinal'],'PANEL_QUERY_ORDER')
   need(len(row['raw_ranked_physical_rows'])==(128 if panel=='EVAL32' else 5412),'PANEL_NATIVE_RANK_LENGTH')
   need(set(sealed['predictions'])==set(MODELS) and all(set(x)==set(MODES) for x in sealed['predictions'].values()),'ALL_MODELS_MODES')
   for mode in MODES:
    x=h['independent_feature_matrix'](row,mode);need(tsha(x)==sealed['feature_sha256'][mode],'INDEPENDENT_SIX_COLUMN_BITS')
    for model,(w,b) in tensors.items():
     expected=h['independent_prediction'](row,x@w+b)
     need(expected==sealed['predictions'][model][mode],'INDEPENDENT_121920_LOGITS_960_ACTIONS')
     if panel=='EVAL128' and model=='ORIGINAL7':need(expected==row['predictions'][mode],'OLD128_ORIGINAL7_ALL_LOGITS')
     logits+=127;actions+=1
 need(logits==121920 and actions==960 and not BLOCKED,'ALL_PREDICTIONS_CLOSED_BEFORE_ROLES')
 return banks,preds,params

def group_stats(actions,expected,h,baseline=None):
 import numpy as np
 if baseline is not None:need([x['query_id'] for x in actions]==[x['query_id'] for x in baseline],'PAIRED_GROUP_ORDER')
 grouped={}
 for i,x in enumerate(actions):
  g=grouped.setdefault(x['group'],{'group':x['group'],'query_count':0,'RAW_correct':0,'REAL_correct':0})
  g['query_count']+=1;g['RAW_correct']+=int(x['base_correct'] if baseline is None else baseline[i]['final_correct']);g['REAL_correct']+=int(x['final_correct'])
 groups=[grouped[k] for k in sorted(grouped)]
 for g in groups:g['net']=g['REAL_correct']-g['RAW_correct'];g['accuracy_difference']=g['net']/g['query_count']
 mean=float(sum((Fraction(g['net'],g['query_count']) for g in groups),Fraction())/len(groups))
 signflip=h['signflip_independent'](groups)
 for key,value in signflip.items():need(value==expected['two_sided_group_signflip'][key],'INDEPENDENT_SUBSET_SUM_SIGNFLIP:'+key)
 values=np.array([g['accuracy_difference'] for g in groups],dtype=np.float64);rng=np.random.default_rng(20260910)
 samples=values[rng.integers(0,len(groups),size=(10000,len(groups)))].mean(axis=1);ci=np.quantile(samples,[.025,.975],method='linear')
 boot={'seed':20260910,'draws':10000,'sampling':'source groups with replacement, all group means equally weighted','interval_level':.95,'percentile_method':'numpy.quantile linear','lower':float(ci[0]),'upper':float(ci[1]),'numpy_version':np.__version__,'bit_generator':type(rng.bit_generator).__name__,'samples_binary64_sha256':hashlib.sha256(samples.tobytes()).hexdigest()}
 if baseline is not None:
  for g in groups:g['baseline_correct']=g.pop('RAW_correct');g['model_correct']=g.pop('REAL_correct')
  need(expected['baseline']=='ORIGINAL7','GROUP_BASELINE_LABEL')
 need(groups==expected['groups'] and len(groups)==expected['group_count'] and mean==expected['equal_group_mean_accuracy_difference'] and boot==expected['group_bootstrap'],'GROUP_COUNTS_MEAN_BOOTSTRAP_BITS')
 return {'group_count':len(groups),'equal_group_mean_accuracy_difference':mean,'group_bootstrap':boot,'independent_subset_sum_signflip':signflip}

def join_and_compare(a,banks,preds,params,h):
 global RELEASED
 need(not RELEASED and not BLOCKED,'NO_ROLE_ACCESS_BEFORE_REPLAY');RELEASED=True
 post=a['postjoin_source_bindings'];rm=read(checked(post['EVAL32_role_manifest']));role_entries={e['execution_ordinal']:e for e in rm['shards']}
 labels,mapping=h['gallery_labels']();roles={'EVAL32':{},'EVAL128':{}}
 for row in banks['EVAL32']:
  role=read(checked(role_entries[row['execution_ordinal']]));need(role['query_id']==row['query_id'],'OLD32_ROLE_QUERY_JOIN')
  roles['EVAL32'][row['query_id']]={'identity':role['identity'],'group':role['supergroup'],'original_query_id':role['query_id'],'track':role['track']}
 curator=read(checked(post['EVAL128_curator']));roles['EVAL128']={r['query_id']:r for r in curator['records']}
 result=read(OUT/'result.json');need(result['status']=='ORIGINAL7_TRAIN128_FIXED_RECIPE_READOUT_COMPLETE' and result['authority']==bind(AUTH) and result['program']==bind(READOUT) and result['plan']==bind(PLAN),'VALIDATED_RESULT_PROVENANCE')
 need(result['models']==list(MODELS) and result['primary_model']=='FULL128' and result['all127_logit_count']==121920 and result['parameter_count_per_head']==7 and result['feature_count']==6,'FIXED_RESULT_SCOPE')
 closure=read(OUT/'training_input_closure.json');need(result['training_input_closure']==closure and closure['old_FULL32_bit_exact'] and closure['PAIR64_feature_label_order_unchanged'],'TRAIN_SCOPE_CLOSURE')
 for name,n in [('ORIGINAL7',32),('FULL96',96),('FULL128',128)]:
  c=closure['counts'][name];need(c['selected_FULL_images']==n and c['effective_FULL_loss_rows']+c['recall_miss_count']==n and c['PAIR_auxiliary_rows']==64 and c['effective_FULL_loss_rows']==params[name]['effective_FULL_loss_rows'],'SELECTED_VERSUS_EFFECTIVE_FULL_DOMAIN')
  need(c['effective_execution_order']==sorted(c['effective_execution_order']) and all(0<=i<n for i in c['effective_execution_order']) and len(c['effective_execution_order'])==c['effective_FULL_loss_rows'],'NESTED_TRAIN_ORDER')
 need(set(closure['counts']['ORIGINAL7']['effective_execution_order'])<=set(closure['counts']['FULL96']['effective_execution_order'])<=set(closure['counts']['FULL128']['effective_execution_order']),'NESTED_EFFECTIVE_TRAIN_SETS')
 all_actions={};all_metrics={};all_preservation={};statistics={}
 for panel,rows in banks.items():
  need(set(roles[panel])=={r['query_id'] for r in rows},'ALL_PANEL_ROLES_NO_FILTER')
  acts={name:{mode:[] for mode in MODES} for name in MODELS}
  for row,pred in zip(rows,preds[panel],strict=True):
   role=roles[panel][row['query_id']];report_row={**row,'track':role.get('track',row['track'])}
   if panel=='EVAL128':need(role['execution_ordinal']==row['execution_ordinal'] and role['source_image_sha256']==row['source_image_sha256'],'EVAL128_IMAGE_ROLE_JOIN')
   for name in MODELS:
    for mode in MODES:acts[name][mode].append(h['action'](report_row,pred['predictions'][name][mode],role,labels))
  need(encode(acts)==encode(result['actions'][panel]),'ALL_ACTION_FIELDS_EXACT:'+panel)
  metrics={name:{mode:h['metrics'](aa) for mode,aa in modes.items()} for name,modes in acts.items()};h['eq'](metrics,result['metrics'][panel],'METRICS/'+panel)
  base=acts['ORIGINAL7']['REAL'];base_good={x['query_id'] for x in base if x['final_correct']};raw_good={x['query_id'] for x in base if x['base_correct']}
  need(len(base_good)==(28 if panel=='EVAL32' else 99) and len(raw_good)==(25 if panel=='EVAL32' else 88),'ORIGINAL_AND_RAW_BASELINE_COUNTS')
  if panel=='EVAL32':need(all(x['candidate_recall'] for x in base),'OLD32_ALL_TARGETS_PRESENT')
  retention={};statistics[panel]={}
  for name in MODELS:
   aa=acts[name]['REAL'];good={x['query_id'] for x in aa if x['final_correct']}
   retention[name]={'original7_correct_count':len(base_good),'retained_original7_correct':len(good&base_good),'lost_original7_correct_query_ids':sorted(base_good-good),'added_vs_original7_query_ids':sorted(good-base_good),'RAW_correct_count':len(raw_good),'retained_RAW_correct':len(good&raw_good),'lost_RAW_correct_query_ids':sorted(raw_good-good)}
   statistics[panel][name]={'vs_RAW':group_stats(aa,result['group_results_vs_RAW'][panel][name],h)}
   need(statistics[panel][name]['vs_RAW']['group_count']==(11 if panel=='EVAL32' else 21),'EXPECTED_GROUP_UNITS')
   if name!='ORIGINAL7':statistics[panel][name]['vs_ORIGINAL7']=group_stats(aa,result['group_results_vs_ORIGINAL7'][panel][name],h,base)
   h['eq'](h['compare'](acts[name]['CBIND'],aa),result['CBIND_vs_REAL'][panel][name],'CBIND/'+panel+'/'+name)
   for mode in MODES:
    for x in acts[name][mode]:
     if not x['candidate_recall']:need(not x['base_correct'] and not x['final_correct'] and x['target_challenger_logit'] is None and x['final_target_rank_full_gallery']==x['raw_target_rank_full_gallery'],'TARGET_ABSENT_RETAINED')
  need(retention==result['original7_and_RAW_correct_preservation'][panel],'SEPARATE_RAW_ORIGINAL_CORRECT_SETS')
  comparisons={name+'_vs_ORIGINAL7':h['compare'](acts[name]['REAL'],base) for name in ('FULL96','FULL128')}
  comparisons['FULL128_vs_FULL96']=h['compare'](acts['FULL128']['REAL'],acts['FULL96']['REAL']);h['eq'](comparisons,result['paired_comparisons'][panel],'PAIRED/'+panel)
  all_actions[panel]=acts;all_metrics[panel]=metrics;all_preservation[panel]=retention
 # Compare complete baseline correct sets to previously qualified outcomes only after replay.
 old32=read(checked(post['EVAL32_original7_result']));old128=read(checked(post['EVAL128_result']))
 for panel,prior in [('EVAL32',old32['actions']['EVAL']['ORIGINAL7']),('EVAL128',old128['actions'])]:
  for mode in MODES:
   old={x['query_id']:x for x in prior[mode]};new=all_actions[panel]['ORIGINAL7'][mode]
   need(set(old)=={x['query_id'] for x in new},'BASELINE_ALL_QUERY_IDS')
   for x in new:
    for key in ('final_correct','base_correct','final_physical_row','decision'):need(x[key]==old[x['query_id']][key],'BASELINE_FIELD:'+key)
    need(hx(x['switch_logit'])==hx(old[x['query_id']]['switch_logit']),'BASELINE_LOGIT_BITS')
 need(result['source_gallery_mapping_sha256']==mapping,'GALLERY_MAPPING')
 need(result['target_insertions']==result['EVAL_target_absent_queries_dropped']==result['threshold_changes']==result['new_features']==result['new_encoder_RoMa_LP_calls']==0 and result['deployment_changed'] is False,'NO_SCOPE_EXPANSION')
 return all_metrics,all_preservation,statistics

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review-validated-result',action='store_true',required=True);parser.parse_args()
 authority,validation=source_and_seals();h=math_helpers();banks,preds,params=replay_predictions(authority,h)
 metrics,preservation,statistics=join_and_compare(authority,banks,preds,params,h)
 value={'status':'ORIGINAL7_TRAIN128_EXTERNAL_POST_RESULT_REVIEW_PASS','reviewer':bind(PROGRAM),'source_readout':bind(READOUT),'pure_math_source':bind(HELPER),
  'authority':bind(AUTH),'source_validation':bind(OUT/'validation.json'),'source_result':bind(OUT/'result.json'),
  'parameter_seal':bind(OUT/'parameter_seal.json'),'fit_validation':bind(OUT/'fit_validation.json'),'prediction_seal':bind(OUT/'prediction_seal.json'),'prediction_validation':bind(OUT/'prediction_validation.json'),
  'models':list(MODELS),'primary_model':'FULL128','panels':COUNTS,'independent_logit_bit_checks':121920,'independent_action_checks':960,
  'metrics':metrics,'original7_and_RAW_correct_preservation':preservation,'group_uncertainty':statistics,
  'all_roles_and_results_opened_after_own_prediction_replay':True,'new_training_updates':0,'third_fit_performed':False,'new_encoder_RoMa_LP_calls':0,
  'rank_scopes':{'EVAL32':'original natural C128; every target present; no fabricated5412 axis','EVAL128':'full5412 identity rank; C128-absent targets retained'},
  'group_p_algorithm':'independent integer subset-sum signflip','bootstrap':'fixed seed20260910/10000-draw NumPy replay, not external replication',
  'scientific_GO_or_NO_GO':None,'deployment_changed':False,'limits':['No third training; fresh refit qualification is source-bound and reused.','Source barriers and seals support execution lineage; this review cannot retroactively prove wall-clock chronology.','Both EVAL panels are opened development data; no external-confirmation claim.']}
 append(OUT/'review.json',encode(value)+b'\n')
 lines=['# TRAIN96/TRAIN128固定配方读出：外部复核','',
  '独立重放121,920个logits、960个动作；没有第三次训练。先完成预测重放，再读取角色及结果。',
  '旧EVAL32与新EVAL128分开核对；分别核对原28/99正确集合及RAW25/88正确集合的保持情况。','',
  '| 面板 | 模型 | 正确 | MRR | 对RAW救/损 | 相对ORIGINAL新增/损失 |','| --- | --- | ---: | ---: | --- | --- |']
 for panel in COUNTS:
  for name in MODELS:
   m=metrics[panel][name]['REAL'];p=preservation[panel][name]
   lines.append(f"| {panel} | {name} | {m['final_top1']}/{COUNTS[panel]} | {m['final_full_gallery_MRR']:.9f} | {m['rescue']}/{m['break']} | {len(p['added_vs_original7_query_ids'])}/{len(p['lost_original7_correct_query_ids'])} |")
 for panel in COUNTS:
  st=statistics[panel]['FULL128']['vs_ORIGINAL7'];ci=st['group_bootstrap'];test=st['independent_subset_sum_signflip']
  lines.append(f"{panel} FULL128对ORIGINAL7组等权差 {st['equal_group_mean_accuracy_difference']:+.9f}；95%组bootstrap [{ci['lower']:.9f}, {ci['upper']:.9f}]；精确sign-flip p={test['two_sided_p']:.9g}。")
 lines+=['','旧32按原C128排名、target全部在C128；新128按完整5412排名逐字面move-to-front核对MRR，缺席不丢。',
  '11/21组均值、固定bootstrap区间及独立subset-sum精确sign-flip复核一致。FULL128保持预定主条件，FULL96仅作已预定诊断；不选择新主条件，不自动部署。',
  '完整统计与来源绑定见同目录review.json；这是已有开发面板的结果重放，不能宣称未触碰外部确认。',f"复核SHA：{sha(OUT/'review.json')}。",'']
 append(REPORT,'\n'.join(lines).encode());print(json.dumps({'status':value['status'],'review':bind(OUT/'review.json'),'report':bind(REPORT)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
