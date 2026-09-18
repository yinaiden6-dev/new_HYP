#!/usr/bin/env python3
"""Same old FROZEN_C QR neutralization on original matched32 runtime; no fit."""
from __future__ import annotations
import argparse,ast,datetime,hashlib,importlib.util,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rc_frozen_qr_removal_matched32_v1'
PLAN=ROOT/'plan/RC_FROZEN_QR_REMOVAL_MATCHED32_BRIDGE_V1_20260909.md'
HELPER=ROOT/'programs/explain_rc_frozen_roma_action_terms_v1.py';HELPER_SHA='e51fc957fe0c780173ce931e5f79c11fd9c6263c1faf5329b9a6fd1e15f25476'
INFRA=ROOT/'programs/analyze_rc_frozen_group_effect_difficult90_v1.py';INFRA_SHA='5b27188e3d15ce59a8ebd47b91eb01555fe2d59e32cab1e89cc2512c77f2d13e'
OLD90=ROOT/'results/rc_frozen_group_effect_difficult90_v1'
OLD90_PINS={'result.json':'c27f7c3aa543ba73130be059e1047415d86a35dc2fb17ffe5c00febfbbdb3356','independent_validation.json':'13ce0b1d7d0bce7a0264f03f010466230b6def55236198bee5375166f7349b76','execution_manifest.json':'0371da5e546df29bdef1093707bcfb7278c9f7cded4b51a4da360c0a14abd7ef'}
CONDITIONS={'ORIGINAL':(), 'DROP_QR':(4,5)}
def need(x,m):
 if not x:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def prepare():
 import torch
 need(sha(HELPER)==HELPER_SHA and sha(INFRA)==INFRA_SHA,'HELPER_PINS');h=load(HELPER,'matched32_group_original_helper');u=load(INFRA,'matched32_group_infra')
 sources={k:{'path':x[0],'sha256':x[1]} for k,x in h.PINS.items()};sources.update({'helper':u.bind(HELPER),'infra':u.bind(INFRA),'plan':u.bind(PLAN),'program':u.bind(Path(__file__).resolve()),'continuation':u.bind(u.EXT),'post_submission_continuation':u.bind(u.CONT)})
 need(u.sha(u.EXT)==u.EXT_SHA and u.sha(u.CONT)==u.CONT_SHA,'CONTINUATION_PIN');need(datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(u.read(u.EXT)['cutoff_UTC']),'DEADLINE')
 for name,digest in OLD90_PINS.items():need(u.sha(OLD90/name)==digest,'DIFFICULT90_BRIDGE_PIN');sources['difficult90_'+name]=u.bind(OLD90/name)
 for x in sources.values():need(u.sha(ROOT/x['path'])==x['sha256'],'SOURCE_PIN:'+x['path'])
 barrier=u.Barrier([ROOT/h.PINS[k][0] for k in ('old_result','old_validation')]);v=u.read(ROOT/h.PINS['prejoin_validation'][0])
 need(v['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS' and all(v['checks'].values()) and v['train_count']==v['eval_count']==32,'PREJOIN_VALIDATION')
 rows=[];allq=[]
 for s in range(8):
  ent=next(x for x in v['shards'] if x['shard']==s);path=ROOT/h.PRE/f'shard{s:02d}/result.json';need(ent['pass'] and all(ent['checks'].values()) and u.sha(path)==ent['sha256'],'PREJOIN_SHARD');sources['prejoin_shard_'+str(s)]=u.bind(path);doc=u.read(path)
  need(doc['logical_sha256']==u.logical(doc) and doc['shard']==s and doc['target_role_read_count']==doc['target_insertion_count']==0,'SHARD_ENVELOPE')
  allq.extend(doc['rows']);rows.extend(x for x in doc['rows'] if x['role']=='EVAL')
 need(len(allq)==len({x['query_id'] for x in allq})==64 and len(rows)==32 and sum(x['role']=='TRAIN' for x in allq)==32,'ORIGINAL_FULL64_EVAL32_AXIS')
 for row in rows:
  path=ROOT/h.ROLE_ROOT/f"role_exec{int(row['execution_ordinal']):03d}.json";barrier.paths.add(str(path.resolve()));sources['role_exec_'+str(row['execution_ordinal'])]=u.bind(path)
 head=u.read(ROOT/h.PINS['head'][0]);hv=u.read(ROOT/h.PINS['head_validation'][0]);need(hv['result_sha256']==h.PINS['head'][1] and all(hv['checks'].values()) and hv['status']=='ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_INDEPENDENT_VALIDATION_PASS','FROZEN_HEAD_VALIDATION')
 gate=load(ROOT/h.PINS['frozen_gate'][0],'matched32_group_frozen_gate');w=torch.tensor(head['head']['weight'],dtype=torch.float64);b=float(head['head']['bias'])
 need(w.shape==(6,) and head['parameter_count']==7 and [u.hx(x) for x in w]==[u.hx(x) for x in gate.WEIGHT] and u.hx(b)==u.hx(gate.BIAS),'SAME_OLD_FROZEN_GATE_WEIGHTS')
 old90=u.read(OLD90/'execution_manifest.json');need(old90['weights_binary64']==[u.hx(x) for x in w] and old90['bias_binary64']==u.hx(b),'SAME_HEAD_AS_DIFFICULT90')
 tree=ast.parse((ROOT/h.PINS['feature_program'][0]).read_text());funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('sym','candidate_feature')];need(len(funcs)==2,'PURE_FEATURE_EXTRACTION');ns={'torch':torch};exec(compile(ast.Module(body=funcs,type_ignores=[]),h.PINS['feature_program'][0],'exec'),ns)
 repair=u.read(ROOT/h.PINS['identity_repair'][0]);dups=repair['verified_byte_identical_duplicate_components'];need(repair['physical_row_count']==5413 and repair['corrected_identity_count']==5412 and len(dups)==1 and sorted(dups[0]['physical_rows'])==[714,715],'IDENTITY_REPAIR')
 source=u.read(ROOT/h.PINS['old_source_manifest'][0]);need(source['logical_sha256']==u.logical(source),'SOURCE_MANIFEST_LOGICAL')
 return h,u,sources,barrier,rows,w,b,ns['candidate_feature'],source

def predictions(rows,w,b,feature,u,validate):
 import torch
 preds=[]
 for row in rows:
  candidates={int(x['candidate_position']):x for x in row['candidates']};need(sorted(candidates)==list(range(128)) and len(row['candidates'])==128,'C128_COMPLETE')
  axis=[int(candidates[i]['physical_row']) for i in range(128)];need(len(set(axis))==128 and all(0<=x<5413 for x in axis),'PHYSICAL_ROWS');raw=[float(candidates[i]['raw_score']) for i in range(128)];winner=max(range(128),key=lambda i:(raw[i],-axis[i]));cs=[i for i in range(128) if i!=winner];ff=[];zz={k:[] for k in CONDITIONS}
  for c in cs:
   x=u.literal_feature(raw,candidates,c,winner) if validate else feature(raw,candidates,c,winner);need(x.dtype==torch.float64 and x.shape==(6,) and bool(torch.isfinite(x).all()),'FEATURE_SCHEMA');ff.append([u.hx(t) for t in x])
   for name,cols in CONDITIONS.items():
    modified=x.clone()
    for col in cols:modified[col]=0.0;need(u.hx(modified[col])=='0x0.0p+0','POSITIVE_ZERO')
    for col in set(range(6))-set(cols):need(u.hx(modified[col])==u.hx(x[col]),'RETAINED_FEATURE_BITS')
    z=float((w*modified).sum()+b);need(math.isfinite(z),'FINITE_LOGIT');zz[name].append(u.hx(z))
  pp={}
  for name,hx in zz.items():
   logits=[float.fromhex(z) for z in hx];i=max(range(127),key=lambda i:(logits[i],-axis[cs[i]]));best=cs[i];switch=logits[i]>0
   pp[name]={'all127_logits_binary64':hx,'proposed_challenger':best,'switch_logit_binary64':hx[i],'decision':'SWITCH' if switch else 'HOLD','final_position':best if switch else winner}
  preds.append({'execution_ordinal':int(row['execution_ordinal']),'query_id':row['query_id'],'role':'EVAL','candidate_physical_rows':axis,'raw_scores_binary64':[u.hx(t) for t in raw],
   'base_winner':winner,'challenger_positions':cs,'all127_features_binary64':ff,'conditions':pp})
 return preds

def independent_action(p,name,targets):
 axis=p['candidate_physical_rows'];cs=p['challenger_positions'];z=[float.fromhex(x) for x in p['conditions'][name]['all127_logits_binary64']];index=max(range(127),key=lambda i:(z[i],-axis[cs[i]]));best=cs[index];winner=p['base_winner'];switch=z[index]>0;final=best if switch else winner;positions=[i for i in range(128) if axis[i] in targets]
 return {'execution_ordinal':p['execution_ordinal'],'query_id':p['query_id'],'target_present':bool(positions),'target_position':positions[0] if positions else None,'base_winner':winner,
 'proposed_challenger':best,'switch_logit':z[index],'decision':'SWITCH' if switch else 'HOLD','final_position':final,'base_correct':axis[winner] in targets,'final_correct':axis[final] in targets}

def stats(actions):
 eligible=[a for a in actions if a['target_present']]
 return {'query_count':len(actions),'eligible_target_count':len(eligible),'target_absent_count':len(actions)-len(eligible),'base_top1':sum(a['base_correct'] for a in eligible),'final_top1':sum(a['final_correct'] for a in eligible),
 'rescue':sum(not a['base_correct'] and a['final_correct'] for a in eligible),'break':sum(a['base_correct'] and not a['final_correct'] for a in eligible),'switch_count':sum(a['decision']=='SWITCH' for a in eligible)}

def evaluate(preds,h,u,sources,barrier,source,validate):
 seal=u.read(OUT/'prejoin_seal.json');need(seal['predictions_sha256']==u.sha(OUT/'predictions_prejoin.json') and seal['execution_manifest_sha256']==u.sha(OUT/'execution_manifest.json') and barrier.blocked==0,'PREJOIN_SEAL_REQUIRED');barrier.released=True
 prior={int(x['execution_ordinal']):x for x in source['records']};acts={k:[] for k in CONDITIONS};meta={}
 for p in preds:
  e=p['execution_ordinal'];role=u.read(ROOT/h.ROLE_ROOT/f'role_exec{e:03d}.json');r=prior[e]
  need(role['logical_sha256']==u.logical(role) and role['target_naturally_present'] is True and role['query_id']==r['query_id']==p['query_id'] and role['candidate_axis_sha256']==r['candidate_axis_sha256'] and role['source_manifest_logical_sha256']==source['logical_sha256'] and role['prejoin_source_record_logical_sha256']==r['record_logical_sha256'],'ROLE_SOURCE_JOIN')
  need(role['target_insertion_count']==role['target_spatial_supervision_count']==role['raw_d1_field_count']==0,'RETRIEVAL_ONLY_ROLE')
  t=int(r['candidate_physical_rows'][int(role['target_candidate_position'])]);targets={714,715} if t in (714,715) else {t}
  if t in (714,715):need(role['identity']=='Biogen_21','DUPLICATE_IDENTITY')
  need(sum(x in targets for x in p['candidate_physical_rows'])<=1,'UNIQUE_CURRENT_TARGET');meta[e]={'supergroup':role['supergroup'],'target_identity':role['identity'],'target_equivalent_physical_rows':sorted(targets)}
  for name in CONDITIONS:
   if validate:a=independent_action(p,name,targets)
   else:
    q={'axis':p['candidate_physical_rows'],'winner':p['base_winner'],'challengers':p['challenger_positions'],'query_id':p['query_id'],'execution_ordinal':e};a=h.action(q,[float.fromhex(x) for x in p['conditions'][name]['all127_logits_binary64']],targets)
   for k in ('proposed_challenger','decision','final_position'):need(a[k]==p['conditions'][name][k],'PREJOIN_ACTION_REPLAY')
   need(u.hx(a['switch_logit'])==p['conditions'][name]['switch_logit_binary64'],'PREJOIN_LOGIT_BITS');acts[name].append(a)
 old=u.read(ROOT/h.PINS['old_result'][0]);v=u.read(ROOT/h.PINS['old_validation'][0]);need(v['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_PASS' and v['result_sha256']==h.PINS['old_result'][1] and all(v['checks'].values()),'ORIGINAL_VALIDATION')
 summaries={name:stats(a) if validate else h.summary(a) for name,a in acts.items()};s=summaries['ORIGINAL'];gates={'eligible_target_count_ge_30':s['eligible_target_count']>=30,'strict_top1_gain':s['final_top1']>s['base_top1'],'rescue_gt_break':s['rescue']>s['break'],'break_at_most_one':s['break']<=1};passed=all(gates.values())
 replay={'schema_version':'rc_romav2_colnomic_current_runtime_frozen_gate_v1_20260901','status':'ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_GO' if passed else 'ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_NO_GO',
 'claim_level':'ADAPTIVE_INTERNAL_OOF_DEPLOYMENT_COMPATIBILITY_ONLY','frozen_head_sha256':h.PINS['head'][1],'frozen_head_validation_sha256':h.PINS['head_validation'][1],'parameter_count':7,'model_update_count':0,
 'evaluation_summary':s,'gates':gates,'actions':acts['ORIGINAL'],'target_join_after_all_prejoin_shards':True,'sealed_read_count':0,'scientific_GO_or_NO_GO':None,
 'next_authorized_stage':'NEW_DIFFICULT_SEALED_QUERY_TOKENIZATION_E0' if passed else None,'logical_sha256':''};replay['logical_sha256']=u.logical(replay)
 need(replay==old and u.encode(replay)==u.encode(old),'COMPLETE_ORIGINAL_RESULT_EXACT')
 for a,b in zip(acts['ORIGINAL'],old['actions']):need(u.hx(a['switch_logit'])==u.hx(b['switch_logit']),'ORIGINAL_ACTION_FLOAT_BITS')
 need((s['eligible_target_count'],s['base_top1'],s['final_top1'],s['rescue'],s['break'])==(32,25,27,2,0),'ORIGINAL_32_25_27_2_0')
 oldmap={a['query_id']:a for a in acts['ORIGINAL']};newmap={a['query_id']:a for a in acts['DROP_QR']};oldsuccess={k for k,a in oldmap.items() if a['final_correct']};newsuccess={k for k,a in newmap.items() if a['final_correct']};oldrescue={k for k,a in oldmap.items() if not a['base_correct'] and a['final_correct']};gained=sorted(newsuccess-oldsuccess);lost=sorted(oldsuccess-newsuccess)
 if validate:need(gained==sorted(k for k in oldmap if not oldmap[k]['final_correct'] and newmap[k]['final_correct']) and lost==sorted(k for k in oldmap if oldmap[k]['final_correct'] and not newmap[k]['final_correct']),'PAIRED_COUNT_REPLAY')
 comparison={'new_correct_vs_original':len(gained),'lost_correct_vs_original':len(lost),'net_vs_original':len(gained)-len(lost),'new_correct_query_ids':gained,'lost_correct_query_ids':lost,
 'original_two_rescues_retained':len(oldrescue&newsuccess),'original_two_rescues_lost':len(oldrescue-newsuccess),'retained_original_rescue_query_ids':sorted(oldrescue&newsuccess),'lost_original_rescue_query_ids':sorted(oldrescue-newsuccess),
 'prediction_changed_query_ids':[k for k in oldmap if oldmap[k]['final_position']!=newmap[k]['final_position']],'correct_query_ids_original':sorted(oldsuccess),'correct_query_ids_drop_qr':sorted(newsuccess)}
 old90=u.read(OLD90/'result.json');v90=u.read(OLD90/'independent_validation.json');need(v90['result_sha256']==u.sha(OLD90/'result.json') and all(v90['checks'].values()),'DIFFICULT90_INDEPENDENT_VALIDATION')
 bridge={'same_old_FROZEN_C_weights_bias_verified':True,'matched32_original':27,'matched32_drop_qr':summaries['DROP_QR']['final_top1'],'difficult90_original':old90['summaries']['ORIGINAL']['final_top1'],'difficult90_drop_qr':old90['summaries']['DROP_QR']['final_top1'],
 'matched32_preserves_all_old_correct':not lost,'difficult90_preserves_all_old_correct':old90['paired_vs_original']['DROP_QR']['lost_correct_vs_original']==0,'descriptive_internal_two_bundle_simplification_candidate':not lost and old90['paired_vs_original']['DROP_QR']['lost_correct_vs_original']==0,
 'denominators_pooled':False,'new_external_confirmation':False,'deployment_changed':False}
 for x in sources.values():need(u.sha(ROOT/x['path'])==x['sha256'],'SOURCE_CHANGED')
 return {'status':'FROZEN_QR_REMOVAL_MATCHED32_EXACT_REPLAY_COMPLETE','theory_name':'new HYP','head':'old FROZEN_C seven-parameter head; not NATIVE7',
 'candidate_source':'original current-runtime RAW full-gallery C128','scoring_action':'RoMa soft visibility times ColNomic full-reference image-token MaxSim; original rowwise FP64 weighted sum and max127 zero-threshold HOLD/SWITCH',
 'runtime_identity_rule':'corrected target physical-row membership with714/715equivalence; preserve original matched32 raw SWITCH rule without difficult90 effective-HOLD filter',
 'evidence_level':'previously opened original matched EVAL32 compatibility bundle, no new population confirmation','sources':sources,'conditions':{k:list(v) for k,v in CONDITIONS.items()},
 'execution_manifest_sha256':u.sha(OUT/'execution_manifest.json'),'prejoin_seal_sha256':u.sha(OUT/'prejoin_seal.json'),'summaries':summaries,'actions':acts,'paired_DROP_QR_vs_ORIGINAL':comparison,'same_head_two_bundle_bridge':bridge,
 'query_metadata_after_join':meta,'old_complete_result_canonical_payload_exact':True,'original32_action_logit_bits_exact':True,'query_count':32,'candidate_count':128,'logit_count':8128,
 'runtime_EVAL_target_or_old_outcome_reads_before_seal':0,'model_update_count':0,'new_encoder_or_matcher_forward_count':0,'scheduler_calls':0,'HYP_GO_claimed':False,'deployment_changed':False,
 'limits':['Only the same old frozen head is compared across two opened bundles; NATIVE7 28/32 is a separate head.',
 'Derived Q/R features set to zero, not removal of RoMa geometry, visibility, pixels, content, or all relation information.',
 'Every challenger is recomputed and reselected; the original matched runtime differs from difficult90 in duplicate effective-action bookkeeping.',
 'No training, threshold scan, new candidate scores or automatic adoption. Two opened denominators are not pooled.']}

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--produce',action='store_true');g.add_argument('--validate',action='store_true');validate=p.parse_args().validate
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);need(OUT.exists() if validate else not OUT.exists(),'APPEND_ONLY_PHASE');h,u,sources,barrier,rows,w,b,feature,source=prepare()
 manifest={'sources':sources,'conditions':{k:list(v) for k,v in CONDITIONS.items()},'query_count':32,'candidate_count':128,'weight_binary64':[u.hx(t) for t in w],'bias_binary64':u.hx(b),
 'arithmetic':'FP64_ROW_WISE_(WEIGHT*FEATURES).SUM_PLUS_BIAS','all127_reselected':True,'runtime_rule':'original matched32 raw SWITCH plus corrected identity membership; no difficult90 effective-HOLD substitution','parameter_updates':0}
 if validate:need(u.read(OUT/'execution_manifest.json')==manifest,'MANIFEST_REBUILD')
 else:OUT.mkdir();u.atomic(OUT/'execution_manifest.json',manifest)
 preds=predictions(rows,w,b,feature,u,validate)
 if validate:need(u.read(OUT/'predictions_prejoin.json')==preds,'INDEPENDENT_FEATURE_PREDICTION_REBUILD')
 else:
  u.atomic(OUT/'predictions_prejoin.json',preds);u.atomic(OUT/'prejoin_seal.json',{'execution_manifest_sha256':u.sha(OUT/'execution_manifest.json'),'predictions_sha256':u.sha(OUT/'predictions_prejoin.json'),
  'query_count':32,'conditions':2,'logit_count':8128,'runtime_EVAL_target_or_old_outcome_reads_before_seal':0})
 result=evaluate(preds,h,u,sources,barrier,source,validate)
 if validate:
  need(u.read(OUT/'result.json')==result,'INDEPENDENT_RESULT_REPLAY');u.atomic(OUT/'independent_validation.json',{'status':'FROZEN_QR_REMOVAL_MATCHED32_INDEPENDENT_REEXECUTION_PASS','result_sha256':u.sha(OUT/'result.json'),
  'checks':{'same_old_FROZEN_C_parameters_as_difficult90':True,'all_raw_scalar_features_independently_rebuilt':True,'all8128_rowwise_FP64_logits_exact':True,'all64_actions_and_identity_membership_independent':True,
  'complete_original27_result_and_switch_logit_bits_exact':True,'paired_counts_and_old_rescue_retention_independent':True,'source_and_prejoin_seals_exact':True,'no_training_or_scheduler_calls':True}})
 else:u.atomic(OUT/'result.json',result)
 print(json.dumps({'status':result['status'],'validation':validate,'summaries':result['summaries'],'paired':result['paired_DROP_QR_vs_ORIGINAL'],'same_head_two_bundle_bridge':result['same_head_two_bundle_bridge'],'result_sha256':u.sha(OUT/'result.json')},sort_keys=True),flush=True)
if __name__=='__main__':main()
