#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1';PARENT=ROOT/'results/romav2_colnomic_current_runtime_bridge_prejoin_v1';OUT=SRC/'validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable RoMa validation exists')
 executions=[];roles=[];shards=[]
 for s in range(8):
  p=SRC/f'shard{s:02d}/result.json';x=json.load(open(p));parent=PARENT/f'shard{s:02d}/payload.pt';checks={'envelope':x['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_SHARD_READY' and x['shard']==s and x['logical_sha256']==logical(x),'source':x['bindings']['source_payload_sha256']==sha(parent),'zero_access':x['target_role_read_count']==x['target_insertion_count']==x['sealed_read_count']==x['model_update_count']==0,'rows':len(x['rows'])==8};ok=True
  for r in x['rows']:
   executions.append(int(r['execution_ordinal']));roles.append(r['role']);c=r['candidates'];axis=[int(z['physical_row']) for z in c];ok &= len(c)==128 and [z['candidate_position'] for z in c]==list(range(128)) and axis==sorted(axis) and len(set(axis))==128 and r['target_role_read_count']==r['target_insertion_count']==0
   for z in c:ok &= all(math.isfinite(float(z[k])) for k in ('raw_score','real_score','query_control_score','reference_control_score','visibility_mass')) and float(z['visibility_mass'])>=0 and len(z['query_map_sha256'])==len(z['reference_map_sha256'])==64
  checks['candidate_payloads']=ok;passed=all(checks.values());shards.append({'shard':s,'sha256':sha(p),'checks':checks,'pass':passed})
 checks={'all_shards':all(z['pass'] for z in shards),'query_count':len(executions)==64,'unique_executions':len(set(executions))==64,'role_balance':roles.count('TRAIN')==roles.count('EVAL')==32};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_current_runtime_bridge_roma_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(executions),'train_count':roles.count('TRAIN'),'eval_count':roles.count('EVAL'),'target_label_read_count':0,'sealed_read_count':0,'next_authorized_stage':'CURRENT_RUNTIME_FROZEN_GATE_OOF_REDUCTION' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
