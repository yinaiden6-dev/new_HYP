#!/usr/bin/env python3
"""Post-validation independent sealed-logit/action review. No refit, LP, or scheduler."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, sys
from fractions import Fraction
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_reference_support_maxmin_readout_v1'
PRIOR=ROOT/'results/rc_same_support_specificity_v1'
AUTH=ROOT/'registry/rc_reference_support_maxmin_readout_authority_v1_20260910.json'
MAIN=ROOT/'programs/run_rc_reference_support_maxmin_readout_v1.py'
MAIN_SHA='a0fa2490b98350c37c58974f9edd61d1848c254532cbcfc2703561e557188f6b'
HELPER=ROOT/'programs/run_rc_full_mass_free_identity_development_v1.py'
HELPER_SHA='4f9bebeb75b0aea8070f6cd8d4ec8f2b71c4e680d33e6154899a998f1d742cb2'
INPUT=ROOT/'results/rc_same_support_specificity_inputs_v1/result.json'
INPUT_SHA='13ec8fa9f79282f2cd94e9b451660c99b432c0989ead279cf96c4bfdf3f94bc2'
ROLE_MANIFEST=ROOT/'results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_manifest.json'
ROLE_SHA='2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454'
REPORT=ROOT/'reports/REPORT_NEW_HYP_MAXMIN_READOUT_POST_RESULT_REVIEW_V1_20260910.md'
IDENTITY_AUTH=ROOT/'registry/rc_full_mass_free_identity_development_authority_v1_20260909.json'
IDENTITY_AUTH_SHA='db1230861616ee8c3686ad35ddd9720d91ba51024057581998fa1aeb4ad07961'
IDENTITY_CORE_SHA='995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d'
IDENTITY_CONTRACT_SHA='867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650'
MODELS=('ORIGINAL7','FIXED8','MAXMIN8','PROJECTION8')
MODES=('REAL','CBIND','EXTRA_BIND')
PARAMETERS={'ORIGINAL7':7,'FIXED8':8,'MAXMIN8':8,'PROJECTION8':8}
PAIRS=(('MAXMIN8','ORIGINAL7'),('MAXMIN8','FIXED8'),('FIXED8','ORIGINAL7'),
       ('PROJECTION8','ORIGINAL7'),('PROJECTION8','FIXED8'),('PROJECTION8','MAXMIN8'))
FLOAT_TOLERANCE=2e-15

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def source(value):
 p=Path(value['path']);p=p if p.is_absolute() else ROOT/p
 need(sha(p)==value['sha256'],'SOURCE_BINDING_DRIFT:'+str(p));return p.resolve()
def append(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():need(p.read_bytes()==data,'APPEND_ONLY_REVIEW_DRIFT:'+str(p));return
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o444)
def close(a,b,label):
 if isinstance(a,dict):
  need(isinstance(b,dict) and set(a)==set(b),'DICTIONARY_SCHEMA:'+label)
  for k in a:close(a[k],b[k],label+'/'+str(k))
 elif isinstance(a,list):
  need(isinstance(b,list) and len(a)==len(b),'LIST_SCHEMA:'+label)
  for i,(x,y) in enumerate(zip(a,b)):close(x,y,label+'/'+str(i))
 elif isinstance(a,float):
  need(isinstance(b,(float,int)) and math.isfinite(a) and math.isfinite(float(b)) and abs(a-float(b))<=FLOAT_TOLERANCE,'FLOAT_METRIC_DRIFT:'+label)
 else:need(type(a)==type(b) and a==b,'EXACT_VALUE_DRIFT:'+label)

def no_oracle(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=str(Path(os.fsdecode(args[0])).resolve()).lower()
 if any(s in p for s in ('/rc_opened_eval_strict_','d1_mi','d1-mi','/grozi/','gisc_prerecall_universe')):
  raise PermissionError('PROTECTED_OR_ORACLE_INPUT_FORBIDDEN')
sys.addaudithook(no_oracle)

def metrics(rows):
 n=len(rows);base=sum(x['base_correct'] for x in rows);final=sum(x['final_correct'] for x in rows)
 return {'query_count':n,'base_top1':base,'final_top1':final,'base_R@1':base/n,'final_R@1':final/n,
  'base_MRR':float(sum((Fraction(1,x['base_target_rank']) for x in rows),Fraction())/n),
  'final_MRR':float(sum((Fraction(1,x['final_target_rank']) for x in rows),Fraction())/n),
  'rescue':sum(not x['base_correct'] and x['final_correct'] for x in rows),
  'break':sum(x['base_correct'] and not x['final_correct'] for x in rows),
  'retained_correct':sum(x['base_correct'] and x['final_correct'] for x in rows),
  'retained_wrong':sum(not x['base_correct'] and not x['final_correct'] for x in rows),
  'wrong_to_wrong':sum(x['wrong_to_wrong'] for x in rows),'switch_count':sum(x['decision']=='SWITCH' for x in rows),
  'hold_count':sum(x['decision']=='HOLD' for x in rows)}

def paired(new,base,groups):
 need([r['query_id'] for r in new]==[r['query_id'] for r in base],'PAIRED_QUERY_ORDER')
 rescue=[n['query_id'] for n,b in zip(new,base) if n['final_correct'] and not b['final_correct']]
 broken=[n['query_id'] for n,b in zip(new,base) if b['final_correct'] and not n['final_correct']]
 stats={}
 for n,b in zip(new,base):
  v=stats.setdefault(groups[n['query_id']],{'queries':0,'new_correct':0,'base_correct':0})
  v['queries']+=1;v['new_correct']+=int(n['final_correct']);v['base_correct']+=int(b['final_correct'])
 for v in stats.values():v['net']=v['new_correct']-v['base_correct']
 group_difference=sum((Fraction(v['net'],v['queries']) for v in stats.values()),Fraction())/len(stats)
 mrr=sum((Fraction(1,n['final_target_rank'])-Fraction(1,b['final_target_rank']) for n,b in zip(new,base)),Fraction())/len(new)
 return {'rescue':len(rescue),'break':len(broken),'net':len(rescue)-len(broken),'rescue_query_ids':rescue,'break_query_ids':broken,
  'supergroups':stats,'group_balanced_accuracy_difference':float(group_difference),'MRR_difference':float(mrr)}

def retention(real,control,baseline,groups):
 indexed={a['query_id']:a for a in control}
 raw=[a['query_id'] for a in real if a['final_correct'] and not a['base_correct']]
 added=paired(real,baseline,groups)['rescue_query_ids']
 return {'control_vs_REAL':paired(control,real,groups),'RAW_rescue_query_ids':raw,
  'RAW_rescues_retained_query_ids':[q for q in raw if indexed[q]['final_correct']],
  'RAW_rescues_lost_query_ids':[q for q in raw if not indexed[q]['final_correct']],
  'new_rescues_vs_ORIGINAL7_query_ids':added,'new_rescues_retained_query_ids':[q for q in added if indexed[q]['final_correct']],
  'new_rescues_lost_query_ids':[q for q in added if not indexed[q]['final_correct']]}

def replay(row,pred,target,track,fold):
 axis=row['candidate_physical_rows'];cs=row['challenger_positions'];winner=int(row['base_winner_position'])
 need(len(axis)==128 and len(set(axis))==128 and cs==[p for p in range(128) if p!=winner],'FULL_C128_CHALLENGER_AXIS')
 raw=[float.fromhex(x) for x in row['base_scores_binary64']]
 values=[float.fromhex(x) for x in pred['all127_logits_binary64']]
 need(len(values)==127 and all(math.isfinite(v) for v in raw+values),'ALL127_FINITE_LOGITS')
 # Sorting independently reproduces the maximum-logit, minimum-physical-row tie rule.
 ordering=sorted(range(127),key=lambda j:(-values[j],axis[cs[j]]));j=ordering[0];best=cs[j]
 decision='SWITCH' if values[j]>0 else 'HOLD';final=best if decision=='SWITCH' else winner
 need(pred['decision']==decision and pred['final_position']==final,'SEALED_TARGET_FREE_DECISION')
 base_order=sorted(range(128),key=lambda k:(-raw[k],axis[k]));need(base_order[0]==winner,'RAW_WINNER_AND_PHYSICAL_TIE')
 final_order=base_order.copy()
 if decision=='SWITCH':final_order.remove(best);final_order.insert(0,best)
 bc=winner==target;fc=final==target
 return {'execution_ordinal':row['execution_ordinal'],'query_id':row['query_id'],'track':track,'heldout_fold':fold,
  'base_winner':winner,'base_winner_physical_row':axis[winner],'target_position':target,'target_physical_row':axis[target],
  'base_target_rank':base_order.index(target)+1,'final_target_rank':final_order.index(target)+1,
  'proposed_challenger':best,'proposed_challenger_physical_row':axis[best],'proposed_challenger_base_rank':base_order.index(best)+1,
  'switch_logit':values[j],'decision':decision,'final_position':final,'final_physical_row':axis[final],
  'base_correct':bc,'final_correct':fc,'wrong_to_wrong':not bc and not fc and final!=winner}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review-validated-result',action='store_true',required=True);p.parse_args()
 # No result, role, or prediction parsing is allowed before the completed refit validator is present and bound.
 vpath=OUT/'independent_validation.json';need(vpath.is_file(),'HEAD_INDEPENDENT_VALIDATION_NOT_READY')
 validation=read(vpath)
 need(validation['status']=='MAXMIN_READOUT_INDEPENDENT_REEXECUTION_PASS' and all(x is True for x in validation['checks'].values()),'REFIT_VALIDATION_PASS_REQUIRED')
 need(validation['result_sha256']==sha(OUT/'result.json'),'VALIDATOR_RESULT_BINDING')
 need(sha(MAIN)==MAIN_SHA,'FROZEN_REVIEWED_RUNNER_PIN')
 result=read(OUT/'result.json');auth=read(AUTH);params=read(OUT/'parameters.json');prejoin=read(OUT/'eval_prejoin.json');seal=read(OUT/'eval_prejoin_seal.json');closure=read(OUT/'input_closure.json')
 need(auth['status']=='MAXMIN_READOUT_AUTHORIZED' and source(auth['sources']['program'])==MAIN.resolve(),'AUTHORITY_RUNNER_BINDING')
 need(result['authority_sha256']==sha(AUTH) and result['parameters_sha256']==seal['parameters_sha256']==sha(OUT/'parameters.json'),'PARAMETER_AUTHORITY_SEALS')
 need(result['input_closure_sha256']==sha(OUT/'input_closure.json') and result['eval_prejoin_seal_sha256']==sha(OUT/'eval_prejoin_seal.json') and seal['eval_prejoin_sha256']==sha(OUT/'eval_prejoin.json'),'PREJOIN_AND_INPUT_CLOSURE')
 need(seal['models']==list(MODELS) and seal['modes']==list(MODES) and seal['total_FULL64_mode_actions']==768 and seal['total_FULL64_challenger_logits']==97536,'FOUR_HEAD_THREE_MODE_PRESEAL_COUNTS')
 need(seal['EVAL_count']==seal['TRAIN_count']==32 and seal['candidate_count']==128,'PRESEAL_FULL64_AXIS')
 need(seal['runtime_EVAL_target_reads']==seal['prior_5139024_outcome_reads']==seal['oracle_read_attempts']==0,'DECLARED_PRESEAL_READ_BOUNDARY')
 need(auth['primary']=='MAXMIN8_vs_ORIGINAL7' and auth['modes']==list(MODES) and set(auth['models'])==set(MODELS),'PRIMARY_AND_CONTROL_CONTRACT')
 need(result['model_parameter_counts']==PARAMETERS and set(params)==set(MODELS),'HEAD_PARAMETER_COUNTS')
 need(result['candidate_count']==128 and result['TRAIN_query_count']==result['EVAL_query_count']==32,'RESULT_FULL64_C128_COUNTS')
 for section in ('actions','metrics','comparisons','control_diagnostics','RAW_original_correct_losses'):
  need(set(result[section])=={'TRAIN','EVAL'},'RESULT_SPLIT_SCHEMA:'+section)
 for role in ('TRAIN','EVAL'):
  for section in ('actions','metrics','RAW_original_correct_losses'):
   need(set(result[section][role])==set(MODELS),'RESULT_FOUR_HEADS:'+section)
   for name in MODELS:need(set(result[section][role][name])==set(MODES),'RESULT_THREE_MODES:'+section)
 for name,par in params.items():
  weight=[float.fromhex(x) for x in par['weight_binary64']];bias=float.fromhex(par['bias_binary64'])
  need(len(weight)+1==PARAMETERS[name]==par['parameter_count'] and all(math.isfinite(x) for x in weight+[bias]),'FINITE_PARAMETERS:'+name)
  need(hashlib.sha256(encode({'weight':weight,'bias':bias})).hexdigest()==par['parameter_sha256'],'PARAMETER_SHA:'+name)
 for name,value in auth['sources']['prior_5139024_outputs_hash_only'].items():need(source(value)==(PRIOR/name).resolve(),'PRIOR_FROZEN_OUTPUT_BINDING')
 old=read(PRIOR/'result.json');oldp=read(PRIOR/'parameters.json');oldpred=read(PRIOR/'eval_prejoin.json');olds=read(PRIOR/'eval_prejoin_seal.json');oldv=read(PRIOR/'independent_validation.json');oldclosure=read(PRIOR/'input_closure.json')
 need(oldv['status']=='SAME_SUPPORT_SPECIFICITY_INDEPENDENT_REEXECUTION_PASS' and oldv['result_sha256']==sha(PRIOR/'result.json'),'PREDECESSOR_INDEPENDENT_VALIDATION')
 need(old['parameters_sha256']==olds['parameters_sha256']==sha(PRIOR/'parameters.json') and olds['eval_prejoin_sha256']==sha(PRIOR/'eval_prejoin.json') and old['eval_prejoin_seal_sha256']==sha(PRIOR/'eval_prejoin_seal.json'),'PREDECESSOR_SEALED_HEADS_AND_PREDICTIONS')
 need(old['input_closure_sha256']==sha(PRIOR/'input_closure.json') and closure['parent_input_closure']==oldclosure,'PREDECESSOR_INPUT_CLOSURE')
 need(sha(INPUT)==INPUT_SHA and sha(ROLE_MANIFEST)==ROLE_SHA and sha(HELPER)==HELPER_SHA,'QUALIFIED_AXIS_AND_ROLE_SOURCES')
 rows=[r for r in read(INPUT)['records'] if r['kind']=='FULL'];byq={r['query_id']:r for r in rows}
 predictions={r['query_id']:r for r in prejoin};old_predictions={r['query_id']:r for r in oldpred}
 need(len(rows)==len(byq)==len(predictions)==len(old_predictions)==64 and set(byq)==set(predictions)==set(old_predictions),'COMPLETE_FULL64_QUERY_AXIS')
 orders=oldclosure['parent_input_closure'];split_orders={'TRAIN':orders['training_order'],'EVAL':orders['eval_order']}
 need(all(len(v)==32 for v in split_orders.values()) and not(set(split_orders['TRAIN'])&set(split_orders['EVAL'])),'QUALIFIED_DISJOINT_TRAIN_EVAL_ORDERS')
 roles={int(r['execution_ordinal']):r for r in read(ROLE_MANIFEST)['shards']}
 sys.path.insert(0,str(ROOT/'src'))
 spec=importlib.util.spec_from_file_location('read_only_frozen_identity_catalogue',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 need(sha(IDENTITY_AUTH)==IDENTITY_AUTH_SHA,'IDENTITY_AUTHORITY_CONTAINER_PIN')
 identity_authority=read(IDENTITY_AUTH)
 identity_core=source(identity_authority['sources']['identity_core']);identity_contract=source(identity_authority['sources']['identity_contract'])
 need(sha(identity_core)==IDENTITY_CORE_SHA and sha(identity_contract)==IDENTITY_CONTRACT_SHA,'IDENTITY_CORE_AND_CONTRACT_PINS')
 need(sha(ROOT/helper.PINS['identity_manifest'][0])==helper.PINS['identity_manifest'][1],'CORRECTED_IDENTITY_MANIFEST_PIN')
 labels=helper.corrected_labels();groups={};contexts={};role_sources=[]
 old_actions={a['query_id']:a for role in ('TRAIN','EVAL') for a in old['actions'][role]['ORIGINAL7']['REAL']}
 for q,row in byq.items():
  ex=row['execution_ordinal'];entry=roles[ex];rp=source(entry);role=read(rp);role_sources.append(bind(rp))
  need(role['query_id']==q and role['target_insertion_count']==role['raw_d1_field_count']==role['target_spatial_supervision_count']==0,'INDEPENDENT_ROLE_SOURCE')
  targets=[i for i,physical in enumerate(row['candidate_physical_rows']) if labels[physical]==role['identity']]
  need(len(targets)==1 and targets[0]==old_actions[q]['target_position'],'INDEPENDENT_CORRECTED_TARGET_BINDING')
  groups[q]=role['supergroup'];contexts[q]=(targets[0],role['track'],old_actions[q]['heldout_fold'])
  need(predictions[q]['execution_ordinal']==old_predictions[q]['execution_ordinal']==ex,'PREJOIN_EXECUTION_ORDER')
  need(set(predictions[q]['predictions'])==set(MODELS),'EVERY_HEAD_PREDICTION_PRESENT')
  for model in MODELS:need(set(predictions[q]['predictions'][model])==set(MODES),'EVERY_CONTROL_PREDICTION_PRESENT')
  for name,prior_name in (('ORIGINAL7','ORIGINAL7'),('FIXED8','SPECIFIC8')):
   need(params[name]==oldp[prior_name] and predictions[q]['predictions'][name]==old_predictions[q]['predictions'][prior_name],'EXACT_PREDECESSOR_HEAD_AND_ALL_LOGITS:'+name)
 acts={};stats={};comparisons={};control_checks={};losses={}
 for role in ('TRAIN','EVAL'):
  order=split_orders[role];role_rows=sorted((r for r in rows if r['execution_ordinal'] in order),key=lambda r:r['execution_ordinal'])
  need([r['execution_ordinal'] for r in role_rows]==order,'ORDERED_WHOLE_SPLIT')
  acts[role]={};stats[role]={};losses[role]={};control_checks[role]={}
  for name in MODELS:
   acts[role][name]={mode:[replay(r,predictions[r['query_id']]['predictions'][name][mode],*contexts[r['query_id']]) for r in role_rows] for mode in MODES}
   # Every action field is exact, including the binary64 selected logit.
   need(encode(acts[role][name])==encode(result['actions'][role][name]),'EXACT_INDEPENDENT_ACTION_REPLAY:'+role+':'+name)
   stats[role][name]={mode:metrics(a) for mode,a in acts[role][name].items()}
   close(stats[role][name],result['metrics'][role][name],'METRICS/'+role+'/'+name)
   losses[role][name]={mode:[a['query_id'] for a in aa if a['base_correct'] and not a['final_correct']] for mode,aa in acts[role][name].items()}
   need(losses[role][name]==result['RAW_original_correct_losses'][role][name],'RAW_LOSS_SETS')
  need(acts[role]['ORIGINAL7']['EXTRA_BIND']==acts[role]['ORIGINAL7']['REAL'],'ORIGINAL_EXTRA_BIND_IDENTITY')
  for name,prior_name in (('ORIGINAL7','ORIGINAL7'),('FIXED8','SPECIFIC8')):need(encode(acts[role][name])==encode(old['actions'][role][prior_name]),'EXACT_PREDECESSOR_ACTIONS')
  comparisons[role]={a+'_vs_'+b:paired(acts[role][a]['REAL'],acts[role][b]['REAL'],groups) for a,b in PAIRS}
  close(comparisons[role],result['comparisons'][role],'COMPARISONS/'+role)
  for name in MODELS:
   control_checks[role][name]={mode:retention(acts[role][name]['REAL'],acts[role][name][mode],acts[role]['ORIGINAL7']['REAL'],groups) for mode in ('CBIND','EXTRA_BIND')}
  close(control_checks[role],result['control_diagnostics'][role],'CONTROL_DIAGNOSTICS/'+role)
 eval_stats=stats['EVAL'];need(eval_stats['ORIGINAL7']['REAL']['base_top1']==25 and eval_stats['ORIGINAL7']['REAL']['final_top1']==28,'RAW25_NATIVE28_BASELINE')
 need(len({groups[r['query_id']] for r in rows if r['execution_ordinal'] in split_orders['EVAL']})==result['EVAL_supergroup_count']==11,'EVAL11_GROUPS')
 net={};goal={}
 for name in ('MAXMIN8','PROJECTION8'):
  c=comparisons['EVAL'][name+'_vs_ORIGINAL7'];net[name]=c['net']>0 and c['group_balanced_accuracy_difference']>0
  goal[name]=net[name] and c['break']==0
  need(not goal[name] or eval_stats[name]['REAL']['final_top1']>=29,'GOAL_REQUIRES_AT_LEAST29')
 need(result['meets_user_gain_and_preservation']==goal and result['overall_net_gain_flags']==net,'PRIMARY_SECONDARY_GOAL_FLAGS')
 need(result['primary_internal_candidate']==goal['MAXMIN8'] and result['PROJECTION8_explicit_user_requested_secondary_candidate']==goal['PROJECTION8'],'PRIMARY_NOT_REPLACED_BY_SECONDARY')
 expected_status='MAXMIN8_INTERNAL_GAIN_AND_PRESERVATION_CANDIDATE' if goal['MAXMIN8'] else 'PROJECTION8_SECONDARY_GAIN_AND_PRESERVATION_ONLY' if goal['PROJECTION8'] else 'SUPPORT_READOUT_POSITIVE_NET_WITH_LOSSES_ONLY' if any(net.values()) else 'SUPPORT_READOUT_NO_INTERNAL_GAIN_AND_PRESERVATION'
 need(result['status']==expected_status,'RESULT_STATUS')
 need(result['HYP_GO_claimed'] is False and result['deployment_changed'] is False and result['oracle_read_attempts']==result['prior_preseal_outcome_read_attempts']==0,'SCIENTIFIC_BOUNDARY')
 review={'status':'MAXMIN_READOUT_POST_RESULT_INDEPENDENT_ACTION_REVIEW_PASS','reviewer':bind(Path(__file__).resolve()),
  'sources':{p.name:bind(p) for p in (AUTH,MAIN,OUT/'result.json',vpath,OUT/'parameters.json',OUT/'eval_prejoin.json',OUT/'eval_prejoin_seal.json',OUT/'input_closure.json')},
  'axis_source':bind(INPUT),'role_manifest':bind(ROLE_MANIFEST),'role_sources':role_sources,'identity_helper':bind(HELPER),
  'identity_authority':bind(IDENTITY_AUTH),'identity_core':bind(identity_core),'identity_contract':bind(identity_contract),
  'prior_result':bind(PRIOR/'result.json'),'prior_validation':bind(PRIOR/'independent_validation.json'),
  'head_count':4,'mode_count':3,'FULL_query_count':64,'candidate_count':128,'replayed_challenger_logits':97536,'replayed_actions':768,
  'action_replay_exact':True,'ORIGINAL7_FIXED8_exact_predecessor':True,'aggregate_float_comparison_absolute_tolerance':FLOAT_TOLERANCE,
  'aggregate_MRR_and_group_means_recomputed_with_exact_rationals':True,'metrics':stats,'comparisons':comparisons,
  'RAW_original_correct_losses':losses,'control_diagnostics':control_checks,'meets_user_gain_and_preservation':goal,
  'primary':'MAXMIN8_vs_ORIGINAL7','projection_is_explicit_secondary':True,'new_training_updates':0,'new_LP_calls':0,
  'scope':'Independent sealed-logit action and metric replay after completed fresh-process four-head refit validation; no additional model training.',
  'chronology_limit':'Checks source code, bound prejoin seals and reported read counters; does not retrospectively prove wall-clock read chronology.',
  'external_confirmation_claimed':False,'deployment_changed':False,'report_path':str(REPORT)}
 append(OUT/'independent_review.json',encode(review)+b'\n')
 lines=['# new HYP 四头读出：独立动作复核','',
  '独立重训验证已完成，本复核重新读取封存 logits，独立执行全部 C128 动作及评价算术。',
  '范围：原 RAW C128、已打开 matched EVAL32（11组）；原 RAW 25/32、ORIGINAL7 28/32。','',
  '| 模型 | TRAIN /32 | EVAL /32 | 对 RAW 救/损 | 对 ORIGINAL7 救/损 | 保留原28并新增 |',
  '| --- | ---: | ---: | --- | --- | --- |']
 for name in MODELS:
  m=eval_stats[name]['REAL'];pairing=paired(acts['EVAL'][name]['REAL'],acts['EVAL']['ORIGINAL7']['REAL'],groups)
  flag=str(goal[name]) if name in goal else '对照'
  lines.append(f"| {name} | {stats['TRAIN'][name]['REAL']['final_top1']} | {m['final_top1']} | {m['rescue']}/{m['break']} | {pairing['rescue']}/{pairing['break']} | {flag} |")
 lines+=['',f"主比较 MAXMIN8 达成用户目标：{goal['MAXMIN8']}；用户提出的次比较 PROJECTION8：{goal['PROJECTION8']}。",
  '两者分别报告，不用次比较替换主比较。','',
  '复核覆盖97,536个 challenger logits、768个动作；按 physical row 确定平局，logit>0才 SWITCH。',
  'ORIGINAL7/FIXED8 的参数、全部预测和动作与前序完全一致；RAW 原正确损失名单、救回保留和 group/MRR 均已复算。','']
 for name in ('MAXMIN8','PROJECTION8'):
  c=comparisons['EVAL'][name+'_vs_ORIGINAL7'];lines.append(f"- {name}：新增 {c['rescue_query_ids']}；丢失 {c['break_query_ids']}；group 平均准确率差 {c['group_balanced_accuracy_difference']:.8f}。")
  for mode in ('CBIND','EXTRA_BIND'):
   d=control_checks['EVAL'][name][mode];lines.append(f"  {mode} 下新增救回保留 {d['new_rescues_retained_query_ids']}；丢失 {d['new_rescues_lost_query_ids']}。")
 lines+=['','这是内部开发结果，不是未触碰外部确认、严格空间 ownership 或自动部署。',
  '本复核没有重训、重新求解 LP 或读取容量 oracle；参数/特征的独立重训由前置验证承担。',
  f"复核产物 SHA：`{sha(OUT/'independent_review.json')}`。",'']
 append(REPORT,'\n'.join(lines).encode())
 print(json.dumps({'status':review['status'],'review':bind(OUT/'independent_review.json'),'report':bind(REPORT),
  'EVAL_REAL':{n:eval_stats[n]['REAL']['final_top1'] for n in MODELS},'user_goal':goal},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
