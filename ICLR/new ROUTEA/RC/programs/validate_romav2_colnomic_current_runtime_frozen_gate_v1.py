#!/usr/bin/env python3
import hashlib,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import run_romav2_colnomic_no_regret_action_gate_v1 as feature
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
SRC=ROOT/'results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1';PRE=SRC/'validation.json';RESULT=ROOT/'results/romav2_colnomic_current_runtime_frozen_gate_v1/result.json';HEAD=ROOT/'results/romav2_colnomic_full_negative_action_gate_v1/result.json';ROLES=ROOT/'results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_shards';OUT=ROOT/'results/romav2_colnomic_current_runtime_frozen_gate_v1/independent_validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable independent validation exists')
 pre=json.load(open(PRE));result=json.load(open(RESULT));head=json.load(open(HEAD));weights=torch.tensor(head['head']['weight'],dtype=torch.float64);bias=float(head['head']['bias']);gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities;records=[]
 for s in range(8):records.extend(r for r in json.load(open(SRC/f'shard{s:02d}/result.json'))['rows'] if r['role']=='EVAL')
 actions=[]
 for r in records:
  role=json.load(open(ROLES/f"role_exec{int(r['execution_ordinal']):03d}.json"));identity=str(role['identity']);cands={int(z['candidate_position']):z for z in r['candidates']};axis=[int(cands[i]['physical_row']) for i in range(128)];raw=[float(cands[i]['raw_score']) for i in range(128)];winner=max(range(128),key=lambda i:(raw[i],-axis[i]));tp=[i for i,row in enumerate(axis) if labels[row]==identity];target=tp[0] if tp else None;logits=[]
  for c in range(128):
   if c!=winner:logits.append((float((weights*feature.candidate_feature(raw,cands,c,winner)).sum()+bias),c))
  z,best=max(logits,key=lambda x:(x[0],-axis[x[1]]));final=best if z>0 else winner;actions.append({'execution_ordinal':int(r['execution_ordinal']),'query_id':r['query_id'],'target_present':target is not None,'target_position':target,'base_winner':winner,'proposed_challenger':best,'switch_logit':z,'decision':'SWITCH' if z>0 else 'HOLD','final_position':final,'base_correct':labels[axis[winner]]==identity,'final_correct':labels[axis[final]]==identity})
 e=[x for x in actions if x['target_present']];summary={'query_count':len(actions),'eligible_target_count':len(e),'target_absent_count':len(actions)-len(e),'base_top1':sum(x['base_correct'] for x in e),'final_top1':sum(x['final_correct'] for x in e),'rescue':sum((not x['base_correct']) and x['final_correct'] for x in e),'break':sum(x['base_correct'] and not x['final_correct'] for x in e),'switch_count':sum(x['decision']=='SWITCH' for x in e)};checks={'prejoin_validation':pre['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS','result_envelope':result['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_GO' and result['logical_sha256']==logical(result),'frozen_head':result['frozen_head_sha256']==sha(HEAD) and result['parameter_count']==7 and result['model_update_count']==0,'summary_exact':summary==result['evaluation_summary'],'actions_exact':actions==result['actions'],'gates_exact':result['gates']=={'eligible_target_count_ge_30':True,'strict_top1_gain':True,'rescue_gt_break':True,'break_at_most_one':True},'sealed_zero':result['sealed_read_count']==0};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_current_runtime_frozen_gate_independent_validation_v1_20260901','status':'ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_FAIL','checks':checks,'recomputed_summary':summary,'result_sha256':sha(RESULT),'sealed_read_count':0,'sealed_query_tokenization_e0_authorized':passed,'sealed_full_scoring_authorized':False,'next_authorized_stage':'NEW_DIFFICULT_SEALED_QUERY_TOKENIZATION_E0' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks,'summary':summary},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
