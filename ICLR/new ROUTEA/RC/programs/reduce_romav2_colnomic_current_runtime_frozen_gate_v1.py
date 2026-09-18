#!/usr/bin/env python3
import hashlib,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import run_romav2_colnomic_no_regret_action_gate_v1 as feature
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
SRC=ROOT/'results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1';VALID=SRC/'validation.json';HEAD=ROOT/'results/romav2_colnomic_full_negative_action_gate_v1/result.json';HEAD_VALID=ROOT/'results/romav2_colnomic_full_negative_action_gate_v1/independent_validation.json';ROLES=ROOT/'results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_shards';OUT=ROOT/'results/romav2_colnomic_current_runtime_frozen_gate_v1/result.json'
EXPECTED_HEAD='b296e946ff84ef574605ca5a6f6c6d6892b06dc8f38002b33b913a5afc6f059f';EXPECTED_VALID='2f112c8daa7b0c199e7e7ca2933116bdaf261b547277140950c809cc3bd264de'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable frozen-gate bridge result exists')
 validation=json.load(open(VALID));assert validation['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS';assert sha(HEAD)==EXPECTED_HEAD and sha(HEAD_VALID)==EXPECTED_VALID
 head=json.load(open(HEAD));weights=torch.tensor(head['head']['weight'],dtype=torch.float64);bias=float(head['head']['bias']);assert weights.shape==(6,) and head['parameter_count']==7
 gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities;rows=[]
 for s in range(8):
  doc=json.load(open(SRC/f'shard{s:02d}/result.json'))
  rows.extend(r for r in doc['rows'] if r['role']=='EVAL')
 assert len(rows)==32
 actions=[]
 for r in rows:
  role=json.load(open(ROLES/f"role_exec{int(r['execution_ordinal']):03d}.json"));identity=str(role['identity']);cands={int(z['candidate_position']):z for z in r['candidates']};axis=[int(cands[i]['physical_row']) for i in range(128)];raw=[float(cands[i]['raw_score']) for i in range(128)];winner=max(range(128),key=lambda i:(raw[i],-axis[i]));target_positions=[i for i,row in enumerate(axis) if labels[row]==identity];target=target_positions[0] if target_positions else None;logits=[]
  for c in range(128):
   if c==winner:continue
   z=float((weights*feature.candidate_feature(raw,cands,c,winner)).sum()+bias);logits.append((z,c))
  best_logit,best=max(logits,key=lambda x:(x[0],-axis[x[1]]));final=best if best_logit>0 else winner;base_correct=labels[axis[winner]]==identity;final_correct=labels[axis[final]]==identity;actions.append({'execution_ordinal':int(r['execution_ordinal']),'query_id':r['query_id'],'target_present':target is not None,'target_position':target,'base_winner':winner,'proposed_challenger':best,'switch_logit':best_logit,'decision':'SWITCH' if best_logit>0 else 'HOLD','final_position':final,'base_correct':base_correct,'final_correct':final_correct})
 eligible=[x for x in actions if x['target_present']];base_top=sum(x['base_correct'] for x in eligible);final_top=sum(x['final_correct'] for x in eligible);rescue=sum((not x['base_correct']) and x['final_correct'] for x in eligible);brk=sum(x['base_correct'] and not x['final_correct'] for x in eligible);switch=sum(x['decision']=='SWITCH' for x in eligible);gates={'eligible_target_count_ge_30':len(eligible)>=30,'strict_top1_gain':final_top>base_top,'rescue_gt_break':rescue>brk,'break_at_most_one':brk<=1};passed=all(gates.values());v={'schema_version':'rc_romav2_colnomic_current_runtime_frozen_gate_v1_20260901','status':'ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_GO' if passed else 'ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_NO_GO','claim_level':'ADAPTIVE_INTERNAL_OOF_DEPLOYMENT_COMPATIBILITY_ONLY','frozen_head_sha256':sha(HEAD),'frozen_head_validation_sha256':sha(HEAD_VALID),'parameter_count':7,'model_update_count':0,'evaluation_summary':{'query_count':len(actions),'eligible_target_count':len(eligible),'target_absent_count':len(actions)-len(eligible),'base_top1':base_top,'final_top1':final_top,'rescue':rescue,'break':brk,'switch_count':switch},'gates':gates,'actions':actions,'target_join_after_all_prejoin_shards':True,'sealed_read_count':0,'scientific_GO_or_NO_GO':None,'next_authorized_stage':'NEW_DIFFICULT_SEALED_QUERY_TOKENIZATION_E0' if passed else None,'logical_sha256':''};v['logical_sha256']=logical(v);OUT.parent.mkdir(parents=True,exist_ok=False);OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'evaluation':v['evaluation_summary'],'gates':gates},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
