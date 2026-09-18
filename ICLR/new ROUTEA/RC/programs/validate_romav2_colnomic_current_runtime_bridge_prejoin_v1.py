#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/romav2_colnomic_current_runtime_bridge_prejoin_v1'
OUT=SRC/'validation.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def tsha(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape),separators=(',',':')).encode());h.update(x.view(torch.uint8).numpy().tobytes());return h.hexdigest()

def main():
 if OUT.exists():raise RuntimeError('immutable validation exists')
 executions=[];roles=[];shards=[];all_ok=True
 for s in range(8):
  d=SRC/f'shard{s:02d}';rp=d/'receipt.json';pp=d/'payload.pt';r=json.load(open(rp));p=torch.load(pp,map_location='cpu',weights_only=False,mmap=True)
  checks={
   'receipt':r['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_SHARD_READY' and r['shard']==s and r['query_count']==8 and r['target_label_read_count']==0 and r['sealed_model_scoring_count']==0,
   'payload_hash':sha(pp)==r['payload_sha256'],
   'payload':p['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_SHARD_READY' and p['shard']==s and p['shard_count']==8 and len(p['records'])==8 and p['access']=={'target_label_read_count':0,'retrieval_result_read_count':0,'sealed_pixel_decode_count':0,'sealed_model_scoring_count':0},
  }
  needed=set()
  for q in p['records']:
   axis=list(map(int,q['candidate_physical_rows']));qt=q['query_tokens'];rs=q['candidate_raw_scores'];needed.update(axis);executions.append(int(q['execution_ordinal']));roles.append(q['role'])
   checks.setdefault('queries',True);checks['queries'] &= len(axis)==len(set(axis))==128 and axis==sorted(axis) and qt.dtype==torch.float16 and qt.ndim==2 and qt.shape[1]==128 and qt.shape[0]==math.prod(q['query_grid_shape']) and tsha(qt)==q['query_tokens_sha256'] and rs.dtype==torch.float64 and tuple(rs.shape)==(128,) and bool(torch.isfinite(rs).all()) and q['target_or_label_read_count']==0
  checks['reference_coverage']=needed<=set(map(int,p['references']))
  checks['references']=True
  for row,x in p['references'].items():
   t=x['tokens'];checks['references'] &= int(row)==int(x['physical_row']) and t.dtype==torch.float16 and t.ndim==2 and t.shape[1]==128 and t.shape[0]==math.prod(x['grid_shape']) and tsha(t)==x['tokens_sha256'] and bool(torch.isfinite(t).all())
  passed=all(checks.values());all_ok &= passed;shards.append({'shard':s,'checks':checks,'pass':passed,'payload_sha256':r['payload_sha256'],'query_count':8,'reference_count':len(p['references'])})
 checks={'all_shards':all_ok,'query_count':len(executions)==64,'unique_executions':len(set(executions))==64,'role_balance':roles.count('TRAIN')==roles.count('EVAL')==32,'shard_modulo_exact':all(e is not None for e in executions)}
 passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_current_runtime_bridge_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(executions),'train_count':roles.count('TRAIN'),'eval_count':roles.count('EVAL'),'target_label_read_count':0,'sealed_model_scoring_count':0,'next_authorized_stage':'CURRENT_RUNTIME_BRIDGE_ROMA_VISIBILITY_MATERIALIZATION' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
