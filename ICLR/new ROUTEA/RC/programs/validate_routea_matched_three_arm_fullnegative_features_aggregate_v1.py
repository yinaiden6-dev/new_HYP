#!/usr/bin/env python3
from collections import Counter
import json,os,sys,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'programs'))
import run_routea_d1_current_runtime_bridge_e0_v1 as e0
CONTRACT=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md';SRC=ROOT/'results/routea_matched_three_arm_fullnegative_features_v1';OUT=SRC/'validation.json'
VALIDATION_KEYS={'schema_version','status','shard','checks','query_count','feature_max_abs','payload_sha256','receipt_sha256','target_role_read_count','model_update_count','next_authorized_stage','logical_sha256'}
CHECK_KEYS={'envelope','authorities','bindings','records','access','receipt'}
def atomic(path,value):
 fd,name=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.partial',dir=path.parent);tmp=Path(name)
 try:
  with os.fdopen(fd,'w') as h:json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
  os.replace(tmp,path)
 finally:tmp.unlink(missing_ok=True)
def main():
 if OUT.exists():raise RuntimeError('immutable feature aggregate exists')
 records=[];shards=[];ok=[]
 for s in range(8):
  d=SRC/f'shard{s:02d}';p=d/'payload.pt';r=d/'receipt.json';v=d/'validation.json';x=torch.load(p,map_location='cpu',weights_only=False,mmap=True);z=json.loads(v.read_text());q=json.loads(r.read_text());ok.append(set(z)==VALIDATION_KEYS and set(z.get('checks',{}))==CHECK_KEYS and all(type(value) is bool and value for value in z.get('checks',{}).values()) and z.get('schema_version')=='routea_matched_three_arm_fullnegative_feature_validation_v1_20260902' and z.get('status')=='ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_VALIDATED' and z.get('shard')==s and z.get('query_count')==8 and z.get('next_authorized_stage')=='FULLNEGATIVE_FEATURE_AGGREGATE' and z.get('target_role_read_count')==z.get('model_update_count')==0 and z.get('logical_sha256')==e0.logical_sha256(z) and z.get('payload_sha256')==e0.sha256_file(p) and z.get('receipt_sha256')==e0.sha256_file(r) and z.get('feature_max_abs')==0.0 and q.get('payload_sha256')==e0.sha256_file(p));records+=x['records'];shards.append({'shard':s,'payload_sha256':e0.sha256_file(p),'receipt_sha256':e0.sha256_file(r),'validation_sha256':e0.sha256_file(v)})
 roles=Counter(x['data_split_role'] for x in records);tracks=Counter(x['track'] for x in records);folds=Counter(int(x['heldout_fold']) for x in records);checks={'all_shards':all(ok),'population':len(records)==len({x['query_id'] for x in records})==64,'roles':roles==Counter({'TRAIN':32,'EVAL':32}),'tracks':tracks==Counter({'outcome':53,'difficult':9,'new_difficult_train':2}),'folds':folds==Counter({1:16,2:19,3:12,4:17}),'target_free':all(x['target_role_read_count']==x['model_update_count']==0 for x in records),'c_replay':all(torch.isfinite(x['real_native_features']['C_PAIRED']).all() for x in records)};passed=all(checks.values());value={'schema_version':'routea_matched_three_arm_fullnegative_feature_aggregate_v1_20260902','status':'ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_VALIDATED' if passed else 'ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_ABORT','checks':checks,'query_count':len(records),'roles':dict(roles),'tracks':dict(tracks),'heldout_folds':{str(k):v for k,v in sorted(folds.items())},'shards':shards,'contract_sha256':e0.sha256_file(CONTRACT),'target_role_read_count':0,'model_update_count':0,'next_authorized_stage':'MATCHED_THREE_ARM_CROSSFIT_AFTER_PAIR64' if passed else None,'logical_sha256':''};value['logical_sha256']=e0.logical_sha256(value);atomic(OUT,value);print(json.dumps({'status':value['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
