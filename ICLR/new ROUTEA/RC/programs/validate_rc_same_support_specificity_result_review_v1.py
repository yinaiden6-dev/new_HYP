#!/usr/bin/env python3
"""Post-validation independent readout review. Never trains or inspects Slurm.

Run only after the parent reviewer releases completed artifacts. New result data
is not read unless independent_validation.json first declares exact reexecution
PASS. All models/modes are retained; no selection or deployment is performed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
MODELS=('ORIGINAL7','SPECIFIC8','FREE8')
MODES=('REAL','CBIND','EXTRA_BIND')
INPUT_NAMES={'ORIGINAL7':'ORIGINAL_C','SPECIFIC8':'SPECIFICITY','FREE8':'FREE'}
PINNED={
 'program':('programs/run_rc_same_support_specificity_v1.py','8d9c59f042cd37e7511114436357ab774a0ec72a8919599a8dcc8a30c80ee835'),
 'input':('results/rc_same_support_specificity_inputs_v1/result.json','13ec8fa9f79282f2cd94e9b451660c99b432c0989ead279cf96c4bfdf3f94bc2'),
 'input_validation':('results/rc_same_support_specificity_inputs_v1/validation.json','545b10d603064e2df9d746e9e16fdebdb64de2dc56048aced04c44bc76e48a75'),
 'old_result':('results/rc_absolute_evidence_scale_calibration_v1/result.json','595f1ca3b1b6151af1303c1f6d68cc350af37ef823c204b3a339d55e84e29f6c'),
 'old_validation':('results/rc_absolute_evidence_scale_calibration_v1/independent_validation.json','6cb51164d4e0ec5adc4ba9e2e26ab14ced61c558170ae19a7230a8005d72c0b9'),
 'native_parameters':('registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json','42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b'),
 'roles':('results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_manifest.json','2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454'),
}

def require(x,label):
 if not bool(x):raise RuntimeError(label)
def safe(path):
 p=Path(path).resolve();require(p.is_relative_to(ROOT),'PATH_OUTSIDE_PROJECT')
 require(not any(x in str(p).lower() for x in ('d1_mi','d1-mi','d1_minimal_intervention','grozi','gisc_prerecall_universe','rc_opened_eval_strict_')),'PROTECTED_OR_ORACLE_PATH')
 return p
def sha(path):
 h=hashlib.sha256()
 with safe(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(safe(path).read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def binding(path):return {'path':str(safe(path).relative_to(ROOT)),'sha256':sha(path)}
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else ROOT/p
 require(sha(p)==v['sha256'],'SOURCE_SHA:'+str(p));return p

def atomic(path,data):
 path=safe(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():require(path.read_bytes()==data,'EXISTING_REVIEW_DIFFERS');return
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 path.chmod(0o444)
def symmetric(a,b):return (a-b)/(abs(a)+abs(b)+1e-12)
def hexes(x):return [float(v).hex() for v in x]

def rebuild_features(row):
 raw=[float.fromhex(x) for x in row['base_scores_binary64']]
 ev={int(k):{j:float.fromhex(v) for j,v in e.items()} for k,e in row['evidence_binary64'].items()}
 free={int(k):float.fromhex(v) for k,v in row['free_pool_mean_binary64'].items()}
 specific={int(k):float.fromhex(v) for k,v in row['specificity_margin_binary64'].items()}
 require(len(raw)==len(ev)==len(free)==len(specific)==128 and set(ev)==set(free)==set(specific),'FULL_C128')
 require(all(math.isfinite(v) for v in list(free.values())+list(specific.values())),'FINITE_FREE_AND_SPECIFICITY')
 axis=row['candidate_physical_rows']
 require(len(axis)==len(set(axis))==128,'UNCHANGED_PHYSICAL_AXIS')
 for c in row['candidates']:
  own=c['candidate_position'];other=c['strongest_other_position']
  require(0<=other<128 and other!=own and c['strongest_other_physical_row']==axis[other],'SELF_EXCLUSION_AND_DIAGNOSTIC_AXIS')
  require((free[own]-float.fromhex(c['strongest_other_pool_binary64'])).hex()==specific[own].hex(),'DIAGONAL_MINUS_STRONGEST_OTHER')
 mu=sum(raw)/len(raw);std=(sum((v-mu)**2 for v in raw)/len(raw))**.5
 winner=row['base_winner_position'];challengers=row['challenger_positions'];features={}
 for mode in ('REAL','CBIND'):
  donors={p:p for p in ev} if mode=='REAL' else {p:int(row['cbind_source_positions'][p]) for p in ev}
  require(set(donors.values())==set(ev),'DONOR_PERMUTATION')
  def expanded(p):
   e=ev[donors[p]];s=e['real_score'];m=e['visibility_mass']
   return [s,m,s/max(m,1e-12),s-e['query_control_score'],s-e['reference_control_score']]
  base=expanded(winner);out={name:[] for name in MODELS}
  for c in challengers:
   current=expanded(c);old=[(raw[c]-raw[winner])/max(std,1e-12)]+[symmetric(a,b) for a,b in zip(current,base)]
   out['ORIGINAL7'].append(old)
   out['SPECIFIC8'].append(old+[symmetric(specific[donors[c]],specific[donors[winner]])])
   out['FREE8'].append(old+[symmetric(free[donors[c]],free[donors[winner]])])
  for name in MODELS:
   require([hexes(x) for x in out[name]]==row['feature_binary64'][mode][INPUT_NAMES[name]],'SEALED_INPUT_FEATURE_REPLAY:'+mode+':'+name)
  features[mode]=out
 features['EXTRA_BIND']={'ORIGINAL7':features['REAL']['ORIGINAL7']}
 for name in ('SPECIFIC8','FREE8'):
  features['EXTRA_BIND'][name]=[a[:6]+b[6:] for a,b in zip(features['REAL'][name],features['CBIND'][name])]
 return raw,features

def score_action(row,old,raw,x,params):
 import torch
 w=torch.tensor([float.fromhex(v) for v in params['weight_binary64']],dtype=torch.float64)
 b=float.fromhex(params['bias_binary64']);values=torch.tensor(x,dtype=torch.float64)@w+b
 require(len(values)==127 and bool(values.isfinite().all()),'FINITE_ALL127_LOGITS')
 axis=row['candidate_physical_rows'];cs=row['challenger_positions'];winner=row['base_winner_position'];target=old['target_position']
 require(axis[target]==old['target_physical_row'] and winner==old['base_winner'] and axis[winner]==old['base_winner_physical_row'],'OLD_TARGET_AND_RAW_AXIS')
 pos=max(range(len(cs)),key=lambda i:(float(values[i]),-axis[cs[i]]));best=cs[pos];logit=float(values[pos]);switched=logit>0.0
 final=best if switched else winner
 order=sorted(range(128),key=lambda i:(-raw[i],axis[i]));require(order[0]==winner,'RAW_WINNER')
 after=[final]+[p for p in order if p!=final] if switched else list(order)
 a={'execution_ordinal':int(row['execution_ordinal']),'query_id':row['query_id'],'track':old['track'],'heldout_fold':int(old['heldout_fold']),
  'base_winner':winner,'base_winner_physical_row':axis[winner],'target_position':target,'target_physical_row':axis[target],
  'base_target_rank':order.index(target)+1,'final_target_rank':after.index(target)+1,
  'proposed_challenger':best,'proposed_challenger_physical_row':axis[best],'proposed_challenger_base_rank':order.index(best)+1,
  'switch_logit':logit,'decision':'SWITCH' if switched else 'HOLD','final_position':final,'final_physical_row':axis[final],
  'base_correct':winner==target,'final_correct':final==target,'wrong_to_wrong':winner!=target and final!=target and final!=winner}
 pred={'final_position':final,'decision':a['decision'],'all127_logits_binary64':hexes(values)}
 return a,pred

def summarize(rows):
 n=len(rows);base=sum(r['base_correct'] for r in rows);final=sum(r['final_correct'] for r in rows)
 return {'query_count':n,'base_top1':base,'final_top1':final,'base_R@1':base/n,'final_R@1':final/n,
  'base_MRR':sum(1/r['base_target_rank'] for r in rows)/n,'final_MRR':sum(1/r['final_target_rank'] for r in rows)/n,
  'rescue':sum(not r['base_correct'] and r['final_correct'] for r in rows),'break':sum(r['base_correct'] and not r['final_correct'] for r in rows),
  'retained_correct':sum(r['base_correct'] and r['final_correct'] for r in rows),'retained_wrong':sum(not r['base_correct'] and not r['final_correct'] for r in rows),
  'wrong_to_wrong':sum(r['wrong_to_wrong'] for r in rows),'switch_count':sum(r['decision']=='SWITCH' for r in rows),'hold_count':sum(r['decision']=='HOLD' for r in rows)}

def paired(new,base,groups):
 require([r['query_id'] for r in new]==[r['query_id'] for r in base],'PAIRED_AXIS')
 resc=[n['query_id'] for n,b in zip(new,base) if n['final_correct'] and not b['final_correct']]
 lost=[n['query_id'] for n,b in zip(new,base) if b['final_correct'] and not n['final_correct']]
 bygroup={}
 for n,b in zip(new,base):
  g=bygroup.setdefault(groups[n['query_id']],{'queries':0,'new_correct':0,'base_correct':0})
  g['queries']+=1;g['new_correct']+=int(n['final_correct']);g['base_correct']+=int(b['final_correct'])
 for g in bygroup.values():g['net']=g['new_correct']-g['base_correct']
 return {'rescue':len(resc),'break':len(lost),'net':len(resc)-len(lost),'rescue_query_ids':resc,'break_query_ids':lost,
  'supergroups':bygroup,'group_balanced_accuracy_difference':sum(g['net']/g['queries'] for g in bygroup.values())/len(bygroup),
  'MRR_difference':sum(1/n['final_target_rank']-1/b['final_target_rank'] for n,b in zip(new,base))/len(new)}

def controls(real,ctrl,original,groups):
 control={r['query_id']:r for r in ctrl};raw=[r['query_id'] for r in real if r['final_correct'] and not r['base_correct']]
 added=paired(real,original,groups)['rescue_query_ids']
 return {'control_vs_REAL':paired(ctrl,real,groups),'RAW_rescue_query_ids':raw,
  'RAW_rescues_retained_query_ids':[q for q in raw if control[q]['final_correct']],
  'RAW_rescues_lost_query_ids':[q for q in raw if not control[q]['final_correct']],
  'new_rescues_vs_ORIGINAL7_query_ids':added,'new_rescues_retained_query_ids':[q for q in added if control[q]['final_correct']],
  'new_rescues_lost_query_ids':[q for q in added if not control[q]['final_correct']]}

def review(out):
 import torch
 # This is intentionally the first new experiment artifact opened.
 require(sha(out/'independent_validation.json')=='6cabaebb90e5453d607d33faf7c6c11c1f02278a63b57a64f891dd51096211c2','RELEASED_REEXECUTION_SHA')
 val=read(out/'independent_validation.json')
 require(val['status']=='SAME_SUPPORT_SPECIFICITY_INDEPENDENT_REEXECUTION_PASS' and all(v is True for v in val['checks'].values()),'REEXECUTION_PASS_REQUIRED')
 require(sha(out/'result.json')==val['result_sha256']=='174195a1d5c5bb99602ea822d6702b2317740a0a7f265b2d19227059803207d1','VALIDATED_RELEASED_RESULT_SHA')
 pinned={}
 for key,(rel,digest) in PINNED.items():
  p=ROOT/rel;require(sha(p)==digest,'PIN:'+key);pinned[key]=binding(p)
 result=read(out/'result.json');params=read(out/'parameters.json');seal=read(out/'eval_prejoin_seal.json');preds=read(out/'eval_prejoin.json')
 authority=read(ROOT/'registry/rc_same_support_specificity_authority_v1_20260910.json')
 require(sha(ROOT/'registry/rc_same_support_specificity_authority_v1_20260910.json')==result['authority_sha256'],'AUTHORITY_BINDING')
 for item in authority['sources'].values():
  if isinstance(item,dict) and 'path' in item:checked(item)
  elif isinstance(item,dict):
   for child in item.values():checked(child)
 require(result['parameters_sha256']==seal['parameters_sha256']==sha(out/'parameters.json'),'PARAMETER_SEAL')
 require(seal['eval_prejoin_sha256']==sha(out/'eval_prejoin.json') and result['eval_prejoin_seal_sha256']==sha(out/'eval_prejoin_seal.json'),'PREDICTION_SEAL')
 require(result['input_closure_sha256']==sha(out/'input_closure.json'),'INPUT_CLOSURE_SEAL')
 require(seal['runtime_EVAL_target_reads']==seal['oracle_read_attempts']==0,'PREJOIN_ZERO_READS')
 require(set(params)==set(MODELS) and set(seal['models'])==set(MODELS) and tuple(seal['modes'])==MODES,'ALL_PREDECLARED_HEADS_MODES')
 native=read(ROOT/PINNED['native_parameters'][0]);old=read(ROOT/PINNED['old_result'][0]);oldv=read(ROOT/PINNED['old_validation'][0])
 require(oldv['result_sha256']==PINNED['old_result'][1] and all(x is True for x in oldv['checks'].values()),'OLD_BASELINE_VALIDATED')
 for key in ('weight_binary64','bias_binary64'):require(params['ORIGINAL7'][key]==native[key],'BASELINE_PARAMETER_BITS')
 for name,p in params.items():
  w=[float.fromhex(v) for v in p['weight_binary64']];b=float.fromhex(p['bias_binary64'])
  require(all(math.isfinite(v) for v in w+[b]) and len(w)+1==p['parameter_count']==(7 if name=='ORIGINAL7' else 8),'FINITE_HEAD_DIMENSIONS')
  require(hashlib.sha256(canonical({'weight':w,'bias':b})).hexdigest()==p['parameter_sha256'],'PARAMETER_CONTENT_SHA')
 inputs=read(ROOT/PINNED['input'][0]);iv=read(ROOT/PINNED['input_validation'][0])
 require(iv['status']=='RC_SAME_SUPPORT_SPECIFICITY_INPUTS_V1_INDEPENDENT_VALIDATION_PASS' and iv['result']['sha256']==PINNED['input'][1],'QUALIFIED_FEATURE_INPUTS')
 full={r['query_id']:r for r in inputs['records'] if r['kind']=='FULL'};require(len(full)==64,'FULL64')
 predmap={p['query_id']:p for p in preds};require(set(predmap)==set(full),'PREDICTION_QUERY_AXIS')
 entries={int(e['execution_ordinal']):e for e in read(ROOT/PINNED['roles'][0])['shards']}
 groups={};role_bindings=[];actions={};metrics={};comparisons={};diagnostics={};raw_losses={};checks=0
 for role in ('TRAIN','EVAL'):
  original=old['actions'][role]['NATIVE7']['REAL'];require(len(original)==32,'ROLE32')
  actions[role]={n:{m:[] for m in MODES} for n in MODELS}
  for previous in original:
   qid=previous['query_id'];r=full[qid];require(r['execution_ordinal']==previous['execution_ordinal'],'EXECUTION_AXIS')
   entry=entries[int(r['execution_ordinal'])];rp=checked(entry);rr=read(rp)
   require(rr['query_id']==qid and rr['track']==previous['track'] and rr['target_insertion_count']==rr['raw_d1_field_count']==rr['target_spatial_supervision_count']==0,'ORIGINAL_ROLE_BINDING')
   groups[qid]=rr['supergroup'];role_bindings.append(binding(rp))
   raw,features=rebuild_features(r)
   for name in MODELS:
    for mode in MODES:
     a,p=score_action(r,previous,raw,features[mode][name],params[name]);actions[role][name][mode].append(a)
     require(p==predmap[qid]['predictions'][name][mode],'ALL_LOGITS_PREDICTIONS_REPLAY:'+qid+':'+name+':'+mode);checks+=127
  metrics[role]={n:{m:summarize(a) for m,a in modes.items()} for n,modes in actions[role].items()}
  require(actions[role]['ORIGINAL7']['REAL']==original and actions[role]['ORIGINAL7']['CBIND']==old['actions'][role]['NATIVE7']['CBIND'],'ORIGINAL7_FULL64_ACTION_REPLAY')
  require(actions[role]['ORIGINAL7']['EXTRA_BIND']==original,'ORIGINAL_EXTRA_BIND_IDENTITY')
  comparisons[role]={n+'_vs_'+b:paired(actions[role][n]['REAL'],actions[role][b]['REAL'],groups) for n,b in (('SPECIFIC8','ORIGINAL7'),('SPECIFIC8','FREE8'),('FREE8','ORIGINAL7'))}
  diagnostics[role]={n:{m:controls(actions[role][n]['REAL'],actions[role][n][m],actions[role]['ORIGINAL7']['REAL'],groups) for m in ('CBIND','EXTRA_BIND')} for n in MODELS}
  raw_losses[role]={n:{m:[a['query_id'] for a in aa if a['base_correct'] and not a['final_correct']] for m,aa in modes.items()} for n,modes in actions[role].items()}
 for key,recomputed in [('actions',actions),('metrics',metrics),('comparisons',comparisons),('control_diagnostics',diagnostics),('RAW_original_correct_losses',raw_losses)]:
  require(result[key]==recomputed,'RESULT_REVIEW:'+key)
 require(len({groups[a['query_id']] for a in actions['EVAL']['ORIGINAL7']['REAL']})==11,'EVAL_GROUP11')
 original_correct={a['query_id'] for a in actions['EVAL']['ORIGINAL7']['REAL'] if a['final_correct']}
 raw_correct={a['query_id'] for a in actions['EVAL']['ORIGINAL7']['REAL'] if a['base_correct']}
 require(len(original_correct)==28 and len(raw_correct)==25 and raw_correct<=original_correct,'BASELINE28_PRESERVES_RAW25')
 goals={};overall={}
 for name in ('SPECIFIC8','FREE8'):
  cur={a['query_id'] for a in actions['EVAL'][name]['REAL'] if a['final_correct']};cmp=comparisons['EVAL'][name+'_vs_ORIGINAL7']
  overall[name]=cmp['net']>0 and cmp['group_balanced_accuracy_difference']>0
  goals[name]=len(cur)>=29 and original_correct<=cur and raw_correct<=cur and cmp['group_balanced_accuracy_difference']>0
  require(goals[name]==(overall[name] and cmp['break']==0),'SET_VS_COUNT_GOAL_EQUIVALENCE')
 require(goals==result['meets_user_gain_and_preservation'],'STRICT_USER_GOAL_FLAGS')
 require(goals['SPECIFIC8']==result['primary_internal_candidate'] and goals['FREE8']==result['FREE8_secondary_internal_candidate'],'PRIMARY_STRICT_GOAL_FLAGS')
 require(overall==result['secondary_overall_net_gain_flags'],'SECONDARY_OVERALL_GAIN_FLAGS')
 require(goals['SPECIFIC8']==result['primary_no_break_improvement'],'NO_BREAK_FLAG')
 expected_status='SPECIFIC8_INTERNAL_GAIN_AND_PRESERVATION_CANDIDATE' if goals['SPECIFIC8'] else 'FREE8_SECONDARY_GAIN_AND_PRESERVATION_ONLY' if goals['FREE8'] else 'SAME_SUPPORT_POSITIVE_NET_WITH_LOSSES_ONLY' if overall['SPECIFIC8'] or overall['FREE8'] else 'SAME_SUPPORT_NO_INTERNAL_GAIN_AND_PRESERVATION'
 require(result['status']==expected_status,'PREDECLARED_RESULT_STATUS')
 require(result['SPECIFIC8_outperforms_FREE8_net']==(comparisons['EVAL']['SPECIFIC8_vs_FREE8']['net']>0),'SPECIFIC_VS_FREE_FLAG')
 require(result['HYP_GO_claimed'] is False and result['deployment_changed'] is False and result['oracle_read_attempts']==0,'NO_AUTOMATIC_CLAIM_OR_DEPLOYMENT')
 return {'status':'SAME_SUPPORT_SPECIFICITY_POST_RESULT_INDEPENDENT_REVIEW_PASS','reviewer_program':binding(__file__),
  'result':binding(out/'result.json'),'independent_reexecution':binding(out/'independent_validation.json'),
  'source_bindings':pinned,'role_bindings':role_bindings,'counts':{'FULL_queries':64,'models':3,'modes':3,'logits_replayed':checks,'actions_replayed':64*3*3},
  'EVAL_REAL':{n:metrics['EVAL'][n]['REAL'] for n in MODELS},'EVAL_comparisons':comparisons['EVAL'],
  'EVAL_controls':diagnostics['EVAL'],'EVAL_RAW_losses':raw_losses['EVAL'],'meets_user_gain_and_preservation':goals,
  'overall_internal_gain':overall,'original_correct_set_size':28,'RAW_correct_set_size':25,
  'cue_pool_arithmetic': 'Reuses independently qualified literal FULL128/PAIR128 F/J pools; this review rebuilds contrasts and complete head/action readout.',
  'additional_training_updates':0,'new_EVAL_target_source':'none; targets from independently validated original FULL64 baseline, same bound axes',
  'evidence_level':'previously opened internal EVAL32 only','HYP_GO_claimed':False,'deployment_changed':False}

def render(v):
 lines=['# new HYP：同支持reference竞争结果独立复核','',v['status'],'',
  '本复核先要求正式新进程完整重训验证PASS，再从封存输入hex特征和参数独立重算；没有新增训练。',
  '目标沿用已独立验证原FULL64 baseline和同一候选轴；group来自绑定的原role ledger。','',
  '| Head | EVAL32正确 | 对原28救/损/净增 | group平均差 | 增益且保持原正确 |',
  '| --- | --- | --- | --- | --- |']
 for n in MODELS:
  m=v['EVAL_REAL'][n]
  if n=='ORIGINAL7':tail='0/0/0 | 0 | 基线'
  else:
   c=v['EVAL_comparisons'][n+'_vs_ORIGINAL7'];tail=f"{c['rescue']}/{c['break']}/{c['net']:+d} | {c['group_balanced_accuracy_difference']:.9f} | {v['meets_user_gain_and_preservation'][n]}"
  lines.append(f"| {n} | {m['final_top1']}/32 | {tail} |")
 for n in ('SPECIFIC8','FREE8'):
  c=v['EVAL_comparisons'][n+'_vs_ORIGINAL7'];lines.append(f"{n}相对原28：新增{c['rescue_query_ids']}；损失{c['break_query_ids']}。")
 lines+=['','完整重算73,152个logits及576个动作，核对全部REAL/C_BIND/EXTRA_BIND指标、',
  '原NATIVE7参数和动作、RAW25保持集合、相对原28正确的集合包含关系及所有救回控制保留。','',
  'SPECIFIC8与FREE8的预定比较：'+json.dumps(v['EVAL_comparisons']['SPECIFIC8_vs_FREE8'],ensure_ascii=False,sort_keys=True),'',
  '新增救回在EXTRA_BIND下的保留（仅特征candidate对应诊断，不是空间ownership）：','']
 for n in ('SPECIFIC8','FREE8'):
  c=v['EVAL_controls'][n]['EXTRA_BIND'];lines.append(f"- {n}: 新增{c['new_rescues_vs_ORIGINAL7_query_ids']}；保留{c['new_rescues_retained_query_ids']}；丢失{c['new_rescues_lost_query_ids']}。")
 lines+=['','以上是已打开内部EVAL32、原RAW C128、全127 challenger action结果。',
  ('本次有内部候选达到增益且保持原正确的条件。' if any(v['meets_user_gain_and_preservation'].values()) else '本次固定统计、训练与内部评价没有达到增益且保持原正确的目标。'),
  '这个结果不能推出所有reference竞争证据普遍无效，也不能据此自动改换阈值或筛选样本。',
  '不构成未触碰外部确认，不自动部署，也不自动宣布普遍new HYP理论成立。','',
  '结果SHA：'+v['result']['sha256'],'独立重训验证SHA：'+v['independent_reexecution']['sha256'],
  '本复核程序SHA：'+v['reviewer_program']['sha256'],'']
 return '\n'.join(lines).encode()

def main():
 p=argparse.ArgumentParser();p.add_argument('--result-dir',required=True);p.add_argument('--output');p.add_argument('--report');a=p.parse_args()
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 out=safe(a.result_dir);v=review(out);dest=safe(a.output) if a.output else out/'post_result_independent_review.json'
 atomic(dest,canonical(v)+b'\n')
 if a.report:atomic(safe(a.report),render(v))
 print(json.dumps({'status':v['status'],'output':binding(dest),'EVAL_REAL':v['EVAL_REAL'],'meets_user_gain_and_preservation':v['meets_user_gain_and_preservation']},sort_keys=True))
if __name__=='__main__':main()
