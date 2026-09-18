#!/usr/bin/env python3
"""After validated DIR9 fitting: independent theta/feature/logit/action review, no fitting."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_shared_projection_direction9_v1'
BASE=ROOT/'results/rc_reference_support_maxmin_readout_v1'
AUTH=ROOT/'registry/rc_shared_projection_direction9_authority_v1_20260910.json'
MAIN=ROOT/'programs/run_rc_shared_projection_direction9_v1.py'
MAIN_SHA='d091d798d9884f2de040cc58771100e0267fed307ff5a58ad4fd82a605d07c03'
REVIEW_HELPER=ROOT/'programs/review_rc_reference_support_maxmin_readout_v1.py'
REVIEW_HELPER_SHA='fd0860d8bbf58b032fc1f7f9ffaadce18679ccaf5c96ae4621653fa32f1023df'
REPORT=ROOT/'reports/REPORT_NEW_HYP_DIRECTION9_POST_RESULT_REVIEW_V1_20260910.md'
MODELS=('FIXED_PROJECTION8','DIRECTION9');MODES=('REAL','CBIND','EXTRA_BIND')
PARAMETERS={'FIXED_PROJECTION8':8,'DIRECTION9':9}
B=None

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 name=str(Path(os.fsdecode(args[0])).resolve()).lower()
 if any(x in name for x in ('/rc_opened_eval_strict_','projection_capacity','angle_capacity','direction_capacity','d1_mi','d1-mi','/grozi/','gisc_prerecall_universe')):
  raise PermissionError('CAPACITY_OR_PROTECTED_INPUT_FORBIDDEN')
sys.addaudithook(audit)

def load_review_math():
 global B
 need(sha(REVIEW_HELPER)==REVIEW_HELPER_SHA,'QUALIFIED_REVIEW_MATH_PIN')
 spec=importlib.util.spec_from_file_location('qualified_standalone_action_review_math',REVIEW_HELPER)
 B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

def tensor_sha(t):
 import torch
 t=t.detach().cpu().contiguous()
 return hashlib.sha256(str(t.dtype).encode()+B.encode(list(t.shape))+t.view(torch.uint8).numpy().tobytes()).hexdigest()

def parameter_review(params,result,old_params):
 import torch
 axes={}
 need(set(params)==set(MODELS) and result['parameter_counts']==PARAMETERS,'TWO_HEAD_PARAMETER_SCHEMA')
 for name,p in params.items():
  w=[float.fromhex(x) for x in p['weight_binary64']];b=float.fromhex(p['bias_binary64']);theta=float.fromhex(p['theta_binary64'])
  learned=name=='DIRECTION9'
  need(len(w)==7 and p['parameter_count']==PARAMETERS[name] and all(math.isfinite(x) for x in w+[b,theta]),'FINITE_EIGHT_OR_NINE_PARAMETERS')
  linear_sha=hashlib.sha256(B.encode({'weight':w,'bias':b})).hexdigest()
  state={k:p[k] for k in ('weight_binary64','bias_binary64','theta_binary64')}
  full_sha=hashlib.sha256(B.encode(state)).hexdigest() if learned else linear_sha
  need(p['linear_parameter_sha256']==linear_sha and p['parameter_sha256']==full_sha,'FULL_THETA_PARAMETER_DIGEST')
  need(p['theta_learned'] is learned and p['private_constructor_proxy_only'] is True and p['finite_training']['all_finite'] is True,'TRAINING_CONSTRUCTOR_AND_FINITE_TRACE')
  need(p['projection_coefficient_binary64']==p['weight_binary64'][6],'PROJECTION_COEFFICIENT_BINDING')
  trace=p['theta_gradient_trace'];maximum=float.fromhex(trace['max_abs_theta_gradient_binary64'])
  need(math.isfinite(maximum) and maximum>=0,'FINITE_THETA_GRADIENT_MAX')
  if learned:
   need(trace['theta_gradient_calls']==2000 and float.fromhex(trace['first_theta_gradient_binary64'])==0.,'ALL2000_THETA_GRADIENT_CALLS_FIRST_ZERO')
   need(0<=trace['nonzero_theta_gradient_calls']<=1999 and ((trace['nonzero_theta_gradient_calls']==0)==(maximum==0.)),'THETA_GRADIENT_AGGREGATE_CONSISTENCY')
  else:
   need(p['theta_binary64']==0.0.hex() and trace['theta_gradient_calls']==trace['nonzero_theta_gradient_calls']==0 and trace['first_theta_gradient_binary64'] is None and maximum==0.,'FIXED_THETA_BUFFER')
  tt=torch.tensor(theta,dtype=torch.float64);co=float(torch.cos(tt));si=float(torch.sin(tt));den=math.sqrt(2.)
  a=(co-si)/den;bb=(co+si)/den;angle=(math.pi/4+theta)%math.pi
  need(abs(a*a+bb*bb-1.)<=2e-15,'UNIT_PROJECTION_AXIS')
  need(float(p['axis_radians_mod_pi_for_description_only']).hex()==angle.hex(),'REPORTED_ANGLE_FROM_SEALED_THETA')
  axes[name]={'theta_binary64':p['theta_binary64'],'theta_radians':theta,'cos_binary64':co.hex(),'sin_binary64':si.hex(),
   'J_axis_coefficient':a,'T_axis_coefficient':bb,'J_axis_coefficient_binary64':a.hex(),'T_axis_coefficient_binary64':bb.hex(),
   'axis_radians_mod_pi':angle,'axis_degrees_mod_180':angle*180/math.pi,'theta_gradient_trace':trace,
   'projection_coefficient_binary64':p['projection_coefficient_binary64'],
   'interpretation':'Algebraic axis description only; frozen inference keeps sum/difference operation order and abs, without angle wrapping.'}
 old=old_params['PROJECTION8'];fixed=params['FIXED_PROJECTION8']
 for k in ('weight_binary64','bias_binary64','parameter_sha256','parameter_count','loss_total_last_recorded','PAIR_loss_last_recorded','FULL_loss_last_recorded','finite_training'):
  need(fixed[k]==old[k],'EXACT_PRIOR_PROJECTION8_TRAINING:'+k)
 for name in MODELS:need(params[name]['original_train_loop_source_sha256']==old['unchanged_training_function_sha256'],'ORIGINAL_LOOP_SOURCE_DIGEST')
 need(result['learned_theta_binary64']==params['DIRECTION9']['theta_binary64'] and result['theta_gradient_trace']==params['DIRECTION9']['theta_gradient_trace'] and result['projection_coefficient_binary64']==params['DIRECTION9']['projection_coefficient_binary64'],'RESULT_THETA_STATE_BINDINGS')
 return axes

def independent_features(row,cached,params,mode):
 import torch
 native_mode='CBIND' if mode=='CBIND' else 'REAL'
 native=torch.tensor([[float.fromhex(x) for x in values] for values in row['feature_binary64'][native_mode]['ORIGINAL_C']],dtype=torch.float64)
 j={int(k):float.fromhex(v) for k,v in cached['fixed_J_binary64'].items()};t={int(k):float.fromhex(v) for k,v in cached['T_binary64'].items()}
 need(set(j)==set(t)==set(range(128)) and cached['fixed_J_binary64']==row['specificity_margin_binary64'],'QUALIFIED_FULL128_J_T')
 donors=list(range(128)) if mode=='REAL' else row['cbind_source_positions'];winner=donors[row['base_winner_position']]
 jc=torch.tensor([j[donors[c]] for c in row['challenger_positions']],dtype=torch.float64)
 tc=torch.tensor([t[donors[c]] for c in row['challenger_positions']],dtype=torch.float64)
 jw=torch.full_like(jc,j[winner]);tw=torch.full_like(tc,t[winner])
 theta=torch.tensor(float.fromhex(params['theta_binary64']),dtype=torch.float64)
 # Same declared FP64 arithmetic order, independently expressed without importing the trained runner.
 co=torch.cos(theta);si=torch.sin(theta)
 dc=torch.abs(co*(jc+tc)+si*(tc-jc))/math.sqrt(2.0)
 dw=torch.abs(co*(jw+tw)+si*(tw-jw))/math.sqrt(2.0)
 extra=(dc-dw)/(dc+dw+1e-12)
 return torch.cat([native,extra.unsqueeze(1)],dim=1)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review-validated-result',action='store_true',required=True);p.parse_args()
 vp=OUT/'independent_validation.json';need(vp.is_file(),'DIR9_INDEPENDENT_VALIDATION_NOT_READY')
 val=read(vp)
 need(val['status']=='SHARED_DIRECTION9_INDEPENDENT_REEXECUTION_PASS' and all(x is True for x in val['checks'].values()),'DIR9_REFIT_VALIDATION_PASS_REQUIRED')
 need(val['result_sha256']==sha(OUT/'result.json'),'VALIDATION_RESULT_BINDING')
 load_review_math();need(sha(MAIN)==MAIN_SHA,'REVIEWED_DIR9_RUNNER_PIN')
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 result=read(OUT/'result.json');auth=read(AUTH);params=read(OUT/'parameters.json');pred=read(OUT/'eval_prejoin.json');seal=read(OUT/'eval_prejoin_seal.json');closure=read(OUT/'input_closure.json');ledger=read(OUT/'frozen_feature_ledger.json')
 need(auth['status']=='SHARED_DIRECTION9_AUTHORIZED' and B.source(auth['sources']['program'])==MAIN.resolve() and auth['primary']=='DIRECTION9_vs_ORIGINAL7','AUTHORITY_PRIMARY_BINDING')
 need(auth['contract']['models']==list(MODELS) and auth['contract']['modes']==list(MODES) and auth['contract']['parameter_counts']==PARAMETERS and auth['contract']['updates']==2000,'FROZEN_TWO_HEAD_CONTRACT')
 need(result['authority_sha256']==sha(AUTH) and result['parameters_sha256']==seal['parameters_sha256']==sha(OUT/'parameters.json'),'PARAMETER_SEALS')
 need(result['eval_prejoin_seal_sha256']==sha(OUT/'eval_prejoin_seal.json') and seal['eval_prejoin_sha256']==sha(OUT/'eval_prejoin.json') and result['input_closure_sha256']==sha(OUT/'input_closure.json') and seal['frozen_features_sha256']==sha(OUT/'frozen_feature_ledger.json'),'INPUT_FEATURE_AND_PREDICTION_SEALS')
 need(seal['models']==list(MODELS) and seal['modes']==list(MODES) and seal['total_FULL64_challenger_logits']==48768 and seal['TRAIN_count']==seal['EVAL_count']==32 and seal['candidate_count']==128,'TWO_HEAD_THREE_MODE_FULL64_PRESEAL')
 need(seal['runtime_EVAL_target_reads']==seal['baseline_outcome_reads']==seal['forbidden_read_attempts']==0,'PRESEAL_READ_COUNTERS')
 for name,value in auth['sources']['four_head_outputs_hash_only'].items():need(B.source(value)==(BASE/name).resolve(),'FOUR_HEAD_OUTCOME_SOURCE_BINDING')
 old=read(BASE/'result.json');oldp=read(BASE/'parameters.json');oldpred=read(BASE/'eval_prejoin.json');olds=read(BASE/'eval_prejoin_seal.json');oldv=read(BASE/'independent_validation.json');oldclosure=read(BASE/'input_closure.json')
 need(oldv['status']=='MAXMIN_READOUT_INDEPENDENT_REEXECUTION_PASS' and oldv['result_sha256']==sha(BASE/'result.json'),'QUALIFIED_FOUR_HEAD_PREDECESSOR')
 need(old['parameters_sha256']==olds['parameters_sha256']==sha(BASE/'parameters.json') and olds['eval_prejoin_sha256']==sha(BASE/'eval_prejoin.json') and old['eval_prejoin_seal_sha256']==sha(BASE/'eval_prejoin_seal.json'),'PREDECESSOR_HEAD_SEALS')
 need(old['input_closure_sha256']==sha(BASE/'input_closure.json') and closure['parent_four_head_input_closure']==oldclosure,'EXACT_PRIOR_INPUT_CLOSURE')
 axes=parameter_review(params,result,oldp)
 need(sha(B.INPUT)==B.INPUT_SHA and sha(B.ROLE_MANIFEST)==B.ROLE_SHA and sha(B.HELPER)==B.HELPER_SHA,'INDEPENDENT_AXIS_ROLE_HELPER_PINS')
 rows=[r for r in read(B.INPUT)['records'] if r['kind']=='FULL'];byq={r['query_id']:r for r in rows};by_pred={r['query_id']:r for r in pred};by_old={r['query_id']:r for r in oldpred}
 need(len(byq)==len(rows)==len(by_pred)==len(pred)==len(by_old)==64 and set(byq)==set(by_pred)==set(by_old),'FULL64_QUERY_UNIVERSE')
 feature_by_key={(r['kind'],r['query_id'],r['execution_ordinal']):r for r in ledger}
 need(len(feature_by_key)==len(ledger)==128 and sum(k[0]=='FULL' for k in feature_by_key)==sum(k[0]=='PAIR' for k in feature_by_key)==64,'FULL64_PAIR64_FROZEN_FEATURE_LEDGER')
 for key,item in feature_by_key.items():
  need(set(item['features'])==set(MODELS),'LEDGER_TWO_HEADS')
  for name in MODELS:need(set(item['features'][name])==({'REAL'} if key[0]=='PAIR' else set(MODES)),'LEDGER_MODES')
 cache_manifest_path=B.source(auth['sources']['four_head_sources']['qualified_cache']['manifest.json']);manifest=read(cache_manifest_path)
 need(manifest['all8320_gap_qualified'] is True,'ALL8320_QUALIFIED_SOURCE')
 cache_entries={r['query_id']:r for r in manifest['records'] if r['kind']=='FULL'};need(set(cache_entries)==set(byq),'FULL_CACHE_AXIS')
 feature_checks=0;logit_checks=0;cache_bindings=[]
 for q,row in byq.items():
  entry=cache_entries[q];rp=B.source(entry);cached=read(rp);cache_bindings.append(B.bind(rp))
  need(cached['candidate_physical_rows']==row['candidate_physical_rows'] and cached['base_winner_position']==row['base_winner_position'] and cached['challenger_positions']==row['challenger_positions'] and cached['cbind_source_positions']==row['cbind_source_positions'],'ALL_MODE_CANDIDATE_DONOR_AXIS')
  need(by_pred[q]['execution_ordinal']==by_old[q]['execution_ordinal']==row['execution_ordinal'] and set(by_pred[q]['predictions'])==set(MODELS),'PREDICTION_QUERY_AND_HEAD_AXIS')
  need(by_pred[q]['predictions']['FIXED_PROJECTION8']==by_old[q]['predictions']['PROJECTION8'],'FIXED_PRIOR_ALL_MODE_PREDICTIONS')
  item=feature_by_key[('FULL',q,row['execution_ordinal'])]
  for name in MODELS:
   need(set(by_pred[q]['predictions'][name])==set(MODES),'PREDICTION_THREE_MODES')
   w=torch.tensor([float.fromhex(x) for x in params[name]['weight_binary64']],dtype=torch.float64);bias=float.fromhex(params[name]['bias_binary64'])
   for mode in MODES:
    features=independent_features(row,cached,params[name],mode)
    need(features.shape==(127,7) and bool(torch.isfinite(features).all()) and tensor_sha(features)==item['features'][name][mode],'INDEPENDENT_THETA_FEATURE_LEDGER_BITS')
    logits=features@w+bias
    need([float(x).hex() for x in logits]==by_pred[q]['predictions'][name][mode]['all127_logits_binary64'],'INDEPENDENT_ALL48768_LOGIT_BITS')
    feature_checks+=1;logit_checks+=len(logits)
 need(feature_checks==384 and logit_checks==48768,'EXACT_FEATURE_LOGIT_COUNTS')
 order_source=oldclosure['parent_input_closure']['parent_input_closure'];orders={'TRAIN':order_source['training_order'],'EVAL':order_source['eval_order']}
 need(all(len(x)==32 for x in orders.values()) and not(set(orders['TRAIN'])&set(orders['EVAL'])),'QUALIFIED_SPLIT_ORDERS')
 roles={int(r['execution_ordinal']):r for r in read(B.ROLE_MANIFEST)['shards']};groups={};contexts={};role_sources=[]
 need(sha(B.IDENTITY_AUTH)==B.IDENTITY_AUTH_SHA,'IDENTITY_AUTHORITY_PIN')
 ia=read(B.IDENTITY_AUTH);ic=B.source(ia['sources']['identity_core']);contract=B.source(ia['sources']['identity_contract'])
 need(sha(ic)==B.IDENTITY_CORE_SHA and sha(contract)==B.IDENTITY_CONTRACT_SHA,'IDENTITY_IMPLEMENTATION_PINS')
 sys.path.insert(0,str(ROOT/'src'));spec=importlib.util.spec_from_file_location('direction_review_identity_catalogue',B.HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 need(sha(ROOT/helper.PINS['identity_manifest'][0])==helper.PINS['identity_manifest'][1],'CORRECTED_IDENTITY_MANIFEST');labels=helper.corrected_labels()
 original_actions={a['query_id']:a for role in ('TRAIN','EVAL') for a in old['actions'][role]['ORIGINAL7']['REAL']}
 for q,row in byq.items():
  path=B.source(roles[row['execution_ordinal']]);role=read(path);role_sources.append(B.bind(path))
  need(role['query_id']==q and role['target_insertion_count']==role['raw_d1_field_count']==role['target_spatial_supervision_count']==0,'QUALIFIED_ROLE')
  target=[i for i,p in enumerate(row['candidate_physical_rows']) if labels[p]==role['identity']]
  need(len(target)==1 and target[0]==original_actions[q]['target_position'],'INDEPENDENT_TARGET_BINDING')
  groups[q]=role['supergroup'];contexts[q]=(target[0],role['track'],original_actions[q]['heldout_fold'])
 actions={};metrics={};comparisons={};controls={};losses={}
 for section in ('actions','metrics','comparisons','control_diagnostics','RAW_original_correct_losses'):need(set(result[section])=={'TRAIN','EVAL'},'RESULT_SPLIT_SCHEMA')
 for role in ('TRAIN','EVAL'):
  rr=sorted((r for r in rows if r['execution_ordinal'] in orders[role]),key=lambda r:r['execution_ordinal'])
  need([r['execution_ordinal'] for r in rr]==orders[role],'ORDERED_WHOLE32')
  actions[role]={};metrics[role]={};controls[role]={};losses[role]={}
  for section in ('actions','metrics','RAW_original_correct_losses'):need(set(result[section][role])==set(MODELS),'RESULT_TWO_HEADS')
  for name in MODELS:
   need(set(result['actions'][role][name])==set(MODES),'RESULT_THREE_CONTROLS')
   actions[role][name]={mode:[B.replay(r,by_pred[r['query_id']]['predictions'][name][mode],*contexts[r['query_id']]) for r in rr] for mode in MODES}
   need(B.encode(actions[role][name])==B.encode(result['actions'][role][name]),'EXACT_INDEPENDENT_ACTION_REPLAY')
   metrics[role][name]={m:B.metrics(a) for m,a in actions[role][name].items()};B.close(metrics[role][name],result['metrics'][role][name],'METRICS/'+role+'/'+name)
   losses[role][name]={m:[a['query_id'] for a in aa if a['base_correct'] and not a['final_correct']] for m,aa in actions[role][name].items()}
   need(losses[role][name]==result['RAW_original_correct_losses'][role][name],'RAW_LOSS_SETS')
   controls[role][name]={mode:B.retention(actions[role][name]['REAL'],actions[role][name][mode],old['actions'][role]['ORIGINAL7']['REAL'],groups) for mode in ('CBIND','EXTRA_BIND')}
  need(B.encode(actions[role]['FIXED_PROJECTION8'])==B.encode(old['actions'][role]['PROJECTION8']),'FIXED_PRIOR_ALL_MODE_ACTIONS_EXACT')
  comparisons[role]={
   'DIRECTION9_vs_ORIGINAL7':B.paired(actions[role]['DIRECTION9']['REAL'],old['actions'][role]['ORIGINAL7']['REAL'],groups),
   'DIRECTION9_vs_FIXED_PROJECTION8':B.paired(actions[role]['DIRECTION9']['REAL'],actions[role]['FIXED_PROJECTION8']['REAL'],groups),
   'FIXED_PROJECTION8_vs_ORIGINAL7':B.paired(actions[role]['FIXED_PROJECTION8']['REAL'],old['actions'][role]['ORIGINAL7']['REAL'],groups)}
  B.close(comparisons[role],result['comparisons'][role],'COMPARISONS/'+role);B.close(controls[role],result['control_diagnostics'][role],'CONTROLS/'+role)
 original=old['metrics']['EVAL']['ORIGINAL7']['REAL'];need(original['base_top1']==25 and original['final_top1']==28 and result['original7_baseline_metrics']==original,'ORIGINAL28_RAW25_BASELINE')
 need(len({groups[r['query_id']] for r in rows if r['execution_ordinal'] in orders['EVAL']})==11,'EVAL11_GROUPS')
 primary=comparisons['EVAL']['DIRECTION9_vs_ORIGINAL7'];net=primary['net']>0 and primary['group_balanced_accuracy_difference']>0;goal=net and primary['break']==0
 need(result['meets_user_gain_and_preservation']==goal and result['overall_positive_net_group_direction']==net,'USER_GAIN_AND_PRESERVATION_FLAGS')
 need(not goal or metrics['EVAL']['DIRECTION9']['REAL']['final_top1']>=29,'GOAL_AT_LEAST29')
 status='DIRECTION9_INTERNAL_GAIN_AND_PRESERVATION_CANDIDATE' if goal else 'DIRECTION9_POSITIVE_NET_WITH_LOSSES_ONLY' if net else 'DIRECTION9_NO_INTERNAL_GAIN_AND_PRESERVATION'
 need(result['status']==status and result['FIXED_PROJECTION8_parameters_predictions_actions_exact'] is True,'RESULT_STATUS_AND_REPLAY_FLAG')
 need(result['forbidden_read_attempts']==result['new_encoder_or_LP_calls']==0 and result['HYP_GO_claimed'] is False and result['deployment_changed'] is False,'SCIENCE_BOUNDARY')
 review={'status':'SHARED_DIRECTION9_POST_RESULT_INDEPENDENT_REVIEW_PASS','reviewer':B.bind(Path(__file__).resolve()),'review_math':B.bind(REVIEW_HELPER),
  'sources':{p.name:B.bind(p) for p in (AUTH,MAIN,OUT/'result.json',vp,OUT/'parameters.json',OUT/'eval_prejoin.json',OUT/'eval_prejoin_seal.json',OUT/'input_closure.json',OUT/'frozen_feature_ledger.json')},
  'prior_result':B.bind(BASE/'result.json'),'prior_validation':B.bind(BASE/'independent_validation.json'),'axis_source':B.bind(B.INPUT),
  'cache_manifest':B.bind(cache_manifest_path),'cache_full64_records':cache_bindings,'role_manifest':B.bind(B.ROLE_MANIFEST),'role_sources':role_sources,
  'identity_authority':B.bind(B.IDENTITY_AUTH),'identity_core':B.bind(ic),'identity_contract':B.bind(contract),
  'head_count':2,'mode_count':3,'FULL_query_count':64,'independently_rebuilt_theta_feature_matrices':feature_checks,
  'independently_recomputed_binary64_logits':logit_checks,'independently_replayed_actions':384,'fixed_prior_parameter_loss_prediction_action_replay_exact':True,
  'theta_full_parameter_digest_checked':True,'gradient_trace_scope':'Aggregate counters and finite final state checked; exact training/gradient reproduction belongs to the completed independent refit validator.',
  'axes_from_sealed_theta':axes,'metrics':metrics,'comparisons':comparisons,'control_diagnostics':controls,'RAW_original_correct_losses':losses,
  'meets_user_gain_and_preservation':goal,'primary':'DIRECTION9_vs_ORIGINAL7','new_training_updates':0,'new_LP_calls':0,'new_encoder_forwards':0,
  'aggregate_float_comparison_absolute_tolerance':B.FLOAT_TOLERANCE,'MRR_and_group_means_computed_as_exact_rationals':True,
  'chronology_limit':'Bound seals, pinned code and zero read counters do not alone retrospectively prove wall-clock read chronology.',
  'external_confirmation_claimed':False,'deployment_changed':False,'report_path':str(REPORT)}
 B.append(OUT/'independent_review.json',B.encode(review)+b'\n')
 lines=['# new HYP：DIR9 独立结果复核','',
  '原 RAW C128、已打开 matched EVAL32（11组）；RAW 25/32，前序 ORIGINAL7 28/32。','',
  '| 模型 | 参数数 | TRAIN /32 | EVAL /32 | 对 RAW 救/损 | 对 ORIGINAL7 救/损 |',
  '| --- | ---: | ---: | ---: | --- | --- |']
 for name in MODELS:
  m=metrics['EVAL'][name]['REAL'];c=comparisons['EVAL'][name+'_vs_ORIGINAL7']
  lines.append(f"| {name} | {PARAMETERS[name]} | {metrics['TRAIN'][name]['REAL']['final_top1']} | {m['final_top1']} | {m['rescue']}/{m['break']} | {c['rescue']}/{c['break']} |")
 learned=axes['DIRECTION9'];tr=learned['theta_gradient_trace']
 lines+=['',f"DIRECTION9 保留原28并新增至少1个：{goal}。",
  f"共享 theta={learned['theta_radians']:.12g} rad；对应轴角度（仅展示，模180°）={learned['axis_degrees_mod_180']:.9f}°。",
  f"实际系数轴 (J,T)=({learned['J_axis_coefficient']:.12g}, {learned['T_axis_coefficient']:.12g})；推理仍保留原 sum/difference 运算顺序。",
  f"theta 梯度调用 {tr['theta_gradient_calls']} 次；首步 {tr['first_theta_gradient_binary64']}；非零 {tr['nonzero_theta_gradient_calls']} 次。",
  f"相对 ORIGINAL7 新增 {primary['rescue_query_ids']}；丢失 {primary['break_query_ids']}；group 平均差 {primary['group_balanced_accuracy_difference']:.9f}。",'']
 for mode in ('REAL','CBIND','EXTRA_BIND'):lines.append(f"- {mode}：FIXED_PROJECTION8 {metrics['EVAL']['FIXED_PROJECTION8'][mode]['final_top1']}/32；DIRECTION9 {metrics['EVAL']['DIRECTION9'][mode]['final_top1']}/32。")
 for mode in ('CBIND','EXTRA_BIND'):
  d=controls['EVAL']['DIRECTION9'][mode];lines.append(f"- {mode} 对新增救回的保留：{d['new_rescues_retained_query_ids']}；丢失：{d['new_rescues_lost_query_ids']}。")
 lines+=['','独立复算384份冻结特征矩阵、48,768个FP64 logits及384个完整动作；固定头参数、损失、预测和动作精确回放前序PROJECTION8。',
  '本复核没有重训、LP求解、方向搜索或encoder执行。首步梯度为0符合全零投影系数初始化；不能把这一点误报为梯度断开。',
  '结果属于内部开发；共享方向增加一个参数，不自动证明新信息、外部泛化、ownership或部署资格。',
  f"复核SHA：`{sha(OUT/'independent_review.json')}`。",'']
 B.append(REPORT,'\n'.join(lines).encode())
 print(json.dumps({'status':review['status'],'review':B.bind(OUT/'independent_review.json'),'report':B.bind(REPORT),
  'EVAL_REAL':{n:metrics['EVAL'][n]['REAL']['final_top1'] for n in MODELS},'user_goal':goal,'theta':learned['theta_radians'],'axis_degrees':learned['axis_degrees_mod_180']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
