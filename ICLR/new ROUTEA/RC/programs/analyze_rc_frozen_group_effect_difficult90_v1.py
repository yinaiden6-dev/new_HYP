#!/usr/bin/env python3
"""Frozen difficult90 S / QR group neutralization; no fitting or scheduler calls."""
from __future__ import annotations
import argparse,builtins,datetime,hashlib,importlib.util,io,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rc_frozen_group_effect_difficult90_v1'
PLAN=ROOT/'plan/RC_FROZEN_GROUP_EFFECT_DECOMPOSITION_V1_20260909.md'
HELPER=ROOT/'programs/explain_rc_frozen_roma_difficult90_action_terms_v1.py'
HELPER_SHA='f5f4033d81aded6511fa49a99a005c240a6fd0c0ca7897a55dcfb256d3757f38'
EXT=ROOT/'registry/rc_retrieval_only_research_extension_authority_v1_20260909.json'
EXT_SHA='b9bdcd4b1c847fc22755a2e5565f30a4e77f3bda4f92e4273769601caf10ceda'
CONT=ROOT/'registry/rc_retrieval_only_post_submission_continuation_addendum_v1_20260909.json'
CONT_SHA='822135bc29e5270fe4222b57be3643abfea61668688eb2f8587e9d461a4c0a7d'
CONDITIONS={'ORIGINAL':(), 'DROP_S':(1,), 'DROP_QR':(4,5), 'DROP_S_QR':(1,4,5)}
BOPEN=builtins.open;IOPEN=io.open

def need(x,m):
 if not x:raise RuntimeError(m)
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with BOPEN(p,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def logical(x):return hashlib.sha256(encode({k:v for k,v in x.items() if k!='logical_sha256'})).hexdigest()
def atomic(p,x):
 need(not p.exists(),'APPEND_ONLY:'+str(p));p.write_bytes(encode(x)+b'\n')
def bind(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def hx(x):return float(x).hex()
def ident(x):return 714 if int(x) in (714,715) else int(x)
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

class Barrier:
 def __init__(self,paths):
  self.paths={str(Path(p).resolve()) for p in paths};self.released=False;self.blocked=0
  builtins.open=self.open_builtin;io.open=self.open_io
 def check(self,file,mode):
  if isinstance(file,(str,bytes,Path)) and ('r' in mode or '+' in mode) and str(Path(file).resolve()) in self.paths and not self.released:
   self.blocked+=1;raise RuntimeError('TARGET_OR_OUTCOME_READ_BEFORE_PREJOIN_SEAL')
 def open_builtin(self,file,mode='r',*a,**k):self.check(file,mode);return BOPEN(file,mode,*a,**k)
 def open_io(self,file,mode='r',*a,**k):self.check(file,mode);return IOPEN(file,mode,*a,**k)

def prepare():
 import torch
 need(sha(HELPER)==HELPER_SHA and sha(EXT)==EXT_SHA and sha(CONT)==CONT_SHA,'ENTRY_SOURCE_PIN')
 authority=read(EXT);need(datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(authority['cutoff_UTC']),'RESEARCH_CUTOFF')
 helper=load(HELPER,'frozen_difficult90_group_helper');sources={k:{'path':v[0],'sha256':v[1]} for k,v in helper.PINS.items()}
 sources.update({'helper':bind(HELPER),'program':bind(Path(__file__).resolve()),'plan':bind(PLAN),'continuation':bind(EXT),'post_submission_continuation':bind(CONT)})
 for x in sources.values():need(sha(ROOT/x['path'])==x['sha256'],'SOURCE_PIN:'+x['path'])
 barrier=Barrier([ROOT/helper.PINS[k][0] for k in ('target_join','original_result','original_validation')])
 repair=read(ROOT/helper.PINS['identity_repair'][0]);dups=repair['verified_byte_identical_duplicate_components']
 need(repair['physical_row_count']==5413 and repair['corrected_identity_count']==5412 and len(dups)==1 and sorted(dups[0]['physical_rows'])==[714,715],'IDENTITY_EQUIVALENCE')
 v=read(ROOT/helper.PINS['real_prejoin_validation'][0]);need(v['status']=='ROMAV2_COLNOMIC_DIFFICULT90_REAL_PREJOIN_VALIDATION_PASS' and v['query_count']==90 and all(v['checks'].values()),'PREJOIN_SOURCE_VALIDATION')
 rows={}
 for shard in v['shards']:
  path=ROOT/f"results/romav2_colnomic_difficult90_real_prejoin_v1/shard{shard['shard']:02d}/result.json"
  need(shard['pass'] and all(shard['checks'].values()) and sha(path)==shard['sha256'],'SHARD_BINDING');sources['real_shard_'+str(shard['shard'])]=bind(path);d=read(path)
  need(d['logical_sha256']==logical(d) and d['target_role_read_count']==0 and d['target_insertion_count']==d['model_update_count']==0,'SHARD_TARGET_FREE')
  for row in d['rows']:
   n=int(row['query_ordinal']);need(n not in rows and row['target_role_read_count']==row['target_insertion_count']==0,'ROW_TARGET_FREE_UNIQUE');rows[n]=row
 need(sorted(rows)==list(range(90)),'ALL90_ROWS')
 gate=load(ROOT/helper.PINS['frozen_gate'][0],'frozen_group_effect_gate')
 need(gate.PARAMETER_COUNT==7 and gate.WEIGHT.dtype==torch.float64 and gate.SWITCH_THRESHOLD==0.0 and tuple(gate.FEATURE_NAMES)==helper.FEATURE_NAMES,'GATE_SCHEMA')
 return helper,gate,sources,barrier,rows

def literal_feature(raw,candidates,c,w):
 import torch
 mean=sum(float(v) for v in raw)/len(raw);std=(sum((float(v)-mean)**2 for v in raw)/len(raw))**.5
 def rel(a,b):return (float(a)-float(b))/(abs(float(a))+abs(float(b))+1e-12)
 a,b=candidates[c],candidates[w];sa,sb=float(a['real_score']),float(b['real_score']);ma,mb=float(a['visibility_mass']),float(b['visibility_mass'])
 return torch.tensor([(float(raw[c])-float(raw[w]))/max(std,1e-12),rel(sa,sb),rel(ma,mb),rel(sa/max(ma,1e-12),sb/max(mb,1e-12)),
  rel(sa-float(a['query_control_score']),sb-float(b['query_control_score'])),rel(sa-float(a['reference_control_score']),sb-float(b['reference_control_score']))],dtype=torch.float64)

def predict_all(rows,gate,validate):
 import torch
 preds=[]
 for ordinal,row in sorted(rows.items()):
  candidates={int(x['candidate_position']):x for x in row['candidates']};need(sorted(candidates)==list(range(128)),'FULL_C128')
  axis=[int(candidates[i]['physical_row']) for i in range(128)];need(len(set(axis))==128 and all(0<=x<5413 for x in axis),'PHYSICAL_AXIS')
  raw=[float(candidates[i]['raw_score']) for i in range(128)];winner=max(range(128),key=lambda i:(raw[i],-axis[i]));cs=[i for i in range(128) if i!=winner];values={k:[] for k in CONDITIONS};features=[]
  for c in cs:
   x=literal_feature(raw,candidates,c,winner) if validate else gate.candidate_feature(raw,candidates,c,winner)
   need(bool(torch.isfinite(x).all()) and x.shape==(6,),'FEATURE_FINITE');features.append([hx(t) for t in x])
   for name,cols in CONDITIONS.items():
    modified=x.clone()
    for col in cols:modified[col]=0.0;need(hx(modified[col])=='0x0.0p+0','LITERAL_POSITIVE_ZERO')
    for col in set(range(6))-set(cols):need(hx(modified[col])==hx(x[col]),'RETAINED_FEATURE_BITS')
    z=float((gate.WEIGHT*modified).sum()+gate.BIAS);need(math.isfinite(z),'FINITE_LOGIT');values[name].append(hx(z))
    if not cols:need(hx(z)==hx(gate.logit(raw,candidates,c,winner)),'ORIGINAL_LOGIT_BITS')
  conditions={}
  for name,zs in values.items():
   zz=[float.fromhex(x) for x in zs];i=max(range(127),key=lambda i:(zz[i],-axis[cs[i]]));best=cs[i];raw_switch=zz[i]>0;effective=raw_switch and ident(axis[best])!=ident(axis[winner]);final=best if effective else winner
   conditions[name]={'all127_logits_binary64':zs,'proposed_challenger_position':best,'switch_logit_binary64':zs[i],'raw_action':'SWITCH' if raw_switch else 'HOLD','effective_action':'SWITCH' if effective else 'HOLD','final_prediction_position':final,'final_prediction_identity_representative':ident(axis[final])}
  preds.append({'query_ordinal':ordinal,'query_id':row['query_id'],'opened_split':row['opened_split'],'candidate_physical_rows':axis,'raw_scores_binary64':[hx(x) for x in raw],
   'base_winner_position':winner,'challenger_positions':cs,'all127_original_features_binary64':features,'conditions':conditions})
 return preds

def independent_action(pred,name,target):
 axis=pred['candidate_physical_rows'];cs=pred['challenger_positions'];p=pred['conditions'][name];zz={c:float.fromhex(x) for c,x in zip(cs,p['all127_logits_binary64'])};winner=pred['base_winner_position'];best=max(cs,key=lambda c:(zz[c],-axis[c]));raw_switch=zz[best]>0;effective=raw_switch and ident(axis[best])!=ident(axis[winner]);final=best if effective else winner
 positions=[i for i in range(128) if ident(axis[i])==ident(target)];t=positions[0] if positions else None;tz=None if t is None or t==winner else zz[t]
 return {'target_present_c128':t is not None,'target_position':t,'base_winner_position':winner,'base_winner_physical_row':axis[winner],
 'proposed_challenger_position':best,'proposed_challenger_physical_row':axis[best],'switch_logit':zz[best],'switch_logit_binary64':hx(zz[best]),
 'target_switch_logit':tz,'target_switch_logit_binary64':None if tz is None else hx(tz),'raw_action':'SWITCH' if raw_switch else 'HOLD','effective_action':'SWITCH' if effective else 'HOLD',
 'final_prediction_position':final,'final_prediction_physical_row':axis[final],'final_prediction_identity_representative':ident(axis[final]),'base_correct':ident(axis[winner])==ident(target),
 'final_correct':ident(axis[final])==ident(target),'wrong_to_wrong':ident(axis[winner])!=ident(target) and ident(axis[final])!=ident(target) and effective}

def stats(actions):
 rescue=[a['query_ordinal'] for a in actions if not a['base_correct'] and a['final_correct']];broken=[a['query_ordinal'] for a in actions if a['base_correct'] and not a['final_correct']]
 return {'query_count':len(actions),'base_top1':sum(a['base_correct'] for a in actions),'final_top1':sum(a['final_correct'] for a in actions),'target_present_c128_count':sum(a['target_present_c128'] for a in actions),
 'rescue':len(rescue),'break':len(broken),'rescue_ordinals':rescue,'break_ordinals':broken,'switch_count':sum(a['effective_action']=='SWITCH' for a in actions),'wrong_to_wrong':sum(a['wrong_to_wrong'] for a in actions)}

def evaluate(preds,helper,gate,sources,barrier,rows,validate):
 seal=read(OUT/'prejoin_seal.json');need(seal['predictions_sha256']==sha(OUT/'predictions_prejoin.json') and seal['execution_manifest_sha256']==sha(OUT/'execution_manifest.json') and barrier.blocked==0,'PREJOIN_SEAL_REQUIRED')
 barrier.released=True;join=read(ROOT/helper.PINS['target_join'][0]);old=read(ROOT/helper.PINS['original_result'][0]);val=read(ROOT/helper.PINS['original_validation'][0]);auth=read(ROOT/helper.PINS['original_reduction_authority'][0])
 need(val['result_sha256']==helper.PINS['original_result'][1] and val['status']=='ROMAV2_COLNOMIC_DIFFICULT90_REGRESSION_INDEPENDENT_VALIDATION_PASS' and all(val['checks'].values()),'OLD_VALIDATION')
 need(old['authority_sha256']==helper.PINS['original_reduction_authority'][1] and old['logical_sha256']==logical(old),'OLD_LOGICAL_AUTHORITY')
 need(auth['manifest']['target_join_sha256']==helper.PINS['target_join'][1] and auth['source_sha256'][helper.PINS['frozen_gate'][0]]==helper.PINS['frozen_gate'][1],'OLD_JOIN_GATE_AUTHORITY')
 need(join['logical_sha256']==logical(join) and join['status']=='ROMAV2_COLNOMIC_DIFFICULT90_TARGET_JOIN_BOUND_REDUCER_ONLY','JOIN_LOGICAL')
 targets={int(x['query_ordinal']):x for x in join['rows']};original={int(x['query_ordinal']):x for x in old['actions']['REAL']};need(sorted(targets)==sorted(original)==list(range(90)),'ALL90_JOIN')
 acts={k:[] for k in CONDITIONS}
 for pred in preds:
  n=pred['query_ordinal'];target=targets[n];previous=original[n];need(pred['query_id']==target['query_id']==previous['query_id'] and pred['opened_split']==target['opened_split']==previous['opened_split'],'QUERY_SPLIT_JOIN')
  for name in CONDITIONS:
   if validate:a=independent_action(pred,name,target['target_gallery_physical_row'])
   else:
    cs=pred['challenger_positions'];zz={c:float.fromhex(x) for c,x in zip(cs,pred['conditions'][name]['all127_logits_binary64'])};raw=[float.fromhex(x) for x in pred['raw_scores_binary64']]
    a=helper.score_action(None,raw,pred['candidate_physical_rows'],pred['base_winner_position'],zz,target['target_gallery_physical_row'],gate)
   for k in ('proposed_challenger_position','switch_logit_binary64','raw_action','effective_action','final_prediction_position','final_prediction_identity_representative'):need(a[k]==pred['conditions'][name][k],'PREJOIN_ACTION_REPLAY')
   a.update({k:pred[k] for k in ('query_ordinal','query_id','opened_split')});acts[name].append(a)
   if name=='ORIGINAL':
    for k in helper.FIELDS_REPLAYED:need(a[k]==previous[k],'OLD_ACTION:'+str(n)+':'+k)
    for k in ('switch_logit','target_switch_logit'):need(helper.same_float(a[k],previous[k]),'OLD_ACTION_FLOAT:'+str(n)+':'+k)
    implicit=previous['proposed_challenger_position'] if previous['effective_action']=='SWITCH' else previous['base_winner_position']
    need(a['final_prediction_position']==implicit and a['base_correct']==(previous['base_rank']==1) and a['final_correct']==(previous['final_rank']==1),'OLD_FINAL_IDENTITY')
 summary={k:(stats(v) if validate else helper.r1_summary(v)) for k,v in acts.items()};baseline=summary['ORIGINAL'];need(all(baseline[k]==old['summaries']['REAL'][k] for k in baseline),'BASELINE_SUMMARY')
 need((baseline['base_top1'],baseline['final_top1'],baseline['rescue'],baseline['break'])==(61,69,8,0),'ORIGINAL_61_69_8_0')
 baseline_success={a['query_ordinal'] for a in acts['ORIGINAL'] if a['final_correct']};rescues=set(baseline['rescue_ordinals']);paired={}
 for name,a in acts.items():
  success={x['query_ordinal'] for x in a if x['final_correct']};gained=sorted(success-baseline_success);lost=sorted(baseline_success-success)
  if validate:
   gained2=[b['query_ordinal'] for b,n in zip(acts['ORIGINAL'],a) if not b['final_correct'] and n['final_correct']];lost2=[b['query_ordinal'] for b,n in zip(acts['ORIGINAL'],a) if b['final_correct'] and not n['final_correct']];need(gained==gained2 and lost==lost2,'INDEPENDENT_PAIRED_COUNT')
  paired[name]={'new_correct_vs_original':len(gained),'lost_correct_vs_original':len(lost),'net_vs_original':len(gained)-len(lost),'new_correct_ordinals':gained,'lost_correct_ordinals':lost,
   'original_rescues_retained':len(rescues&success),'original_rescues_lost':len(rescues-success),'retained_original_rescue_ordinals':sorted(rescues&success),'lost_original_rescue_ordinals':sorted(rescues-success),
   'final_identity_changed_count':sum(b['final_prediction_identity_representative']!=n['final_prediction_identity_representative'] for b,n in zip(acts['ORIGINAL'],a))}
 for x in sources.values():need(sha(ROOT/x['path'])==x['sha256'],'SOURCE_CHANGED_DURING_RUN')
 return {'status':'FROZEN_GROUP_EFFECT_DIFFICULT90_EXACT_REPLAY_COMPLETE','theory_name':'new HYP','head':'original FROZEN_C seven-parameter head, distinct from NATIVE7',
 'candidate_source':'original RAW full-gallery natural C128','scoring_action':'RoMa soft visibility times ColNomic full-reference image-token MaxSim; frozen rowwise FP64 six-feature weighted sum; all127 challenger zero-threshold effective HOLD/SWITCH',
 'evidence_level':'already opened old difficult90 fixed-head feature sensitivity, not untouched confirmation','conditions':{k:list(v) for k,v in CONDITIONS.items()},'sources':sources,
 'prejoin_seal_sha256':sha(OUT/'prejoin_seal.json'),'execution_manifest_sha256':sha(OUT/'execution_manifest.json'),'summaries':summary,'paired_vs_original':paired,'actions':acts,
 'original_population':old['population'],'baseline_all90_actions_and_float_bits_exact':True,'query_count':90,'candidate_count':128,'logit_count':45720,'runtime_target_or_outcome_reads_before_seal':0,
 'counterfactual_selection_uses_all127':True,'frozen_duplicate_identity_component':[714,715],'model_update_count':0,'new_encoder_or_matcher_forwards':0,'threshold_scan_count':0,
 'HYP_GO_claimed':False,'deployment_changed':False,'MRR_recomputed':False,'limits':['These are frozen derived-feature interventions, not pixel or correspondence interventions.',
 'Setting S or Q/R head contrasts to zero does not erase all content or geometry information from retained features.',
 'The old FROZEN_C difficult90 head and current NATIVE7 EVAL32 have different parameters and populations; counts must not be pooled.',
 'All four predeclared conditions are reported. An improved counterfactual is not adopted as a model or selected as a new threshold.',
 'Recomputed labels are used only after all target-free predictions are sealed; the population itself was already opened.']}

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--produce',action='store_true');g.add_argument('--validate',action='store_true');args=p.parse_args();validate=args.validate
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);need(OUT.exists() if validate else not OUT.exists(),'APPEND_ONLY_PHASE')
 helper,gate,sources,barrier,rows=prepare();manifest={'sources':sources,'conditions':{k:list(v) for k,v in CONDITIONS.items()},'query_count':90,'candidate_count':128,
 'weights_binary64':[hx(x) for x in gate.WEIGHT],'bias_binary64':hx(gate.BIAS),'arithmetic':'CLONE_6FEATURE_SET_SELECTED_FP64_POSITIVE_ZERO_THEN_ROWWISE_WEIGHT_TIMES_FEATURE_SUM_PLUS_BIAS',
 'retrieval_only':True,'model_update_count':0,'runtime_target_or_outcome_reads_before_seal':0,'CPU_threads':1}
 if validate:need(read(OUT/'execution_manifest.json')==manifest,'MANIFEST_REBUILD')
 else:OUT.mkdir();atomic(OUT/'execution_manifest.json',manifest)
 preds=predict_all(rows,gate,validate)
 if validate:need(read(OUT/'predictions_prejoin.json')==preds,'INDEPENDENT_RAW_SCALAR_FEATURE_AND_PREDICTION_REBUILD')
 else:
  atomic(OUT/'predictions_prejoin.json',preds);atomic(OUT/'prejoin_seal.json',{'execution_manifest_sha256':sha(OUT/'execution_manifest.json'),'predictions_sha256':sha(OUT/'predictions_prejoin.json'),
   'runtime_target_or_outcome_reads_before_seal':0,'query_count':90,'condition_count':4,'logit_count':45720})
 result=evaluate(preds,helper,gate,sources,barrier,rows,validate)
 if validate:
  need(read(OUT/'result.json')==result,'INDEPENDENT_RESULT_REPLAY')
  atomic(OUT/'independent_validation.json',{'status':'FROZEN_GROUP_EFFECT_DIFFICULT90_INDEPENDENT_REEXECUTION_PASS','result_sha256':sha(OUT/'result.json'),'checks':{
   'sources_and_prejoin_seals_exact':True,'raw_scalar_six_features_independently_rebuilt':True,'all45720_rowwise_FP64_logits_exact':True,'all360_actions_identity_rules_independently_replayed':True,
   'all_summary_paired_and_rescue_retention_counts_recomputed':True,'original_all90_actions_and_float_bits_exact':True,'no_training_or_new_model_or_threshold_selection':True}})
 else:atomic(OUT/'result.json',result)
 print(json.dumps({'status':result['status'],'validation':validate,'summaries':result['summaries'],'paired_vs_original':result['paired_vs_original'],'result_sha256':sha(OUT/'result.json')},sort_keys=True),flush=True)
if __name__=='__main__':main()
