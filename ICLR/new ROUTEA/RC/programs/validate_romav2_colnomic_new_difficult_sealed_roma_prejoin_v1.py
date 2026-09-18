#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/romav2_colnomic_new_difficult_sealed_roma_prejoin_v1';TOK=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_prejoin_v1';VALID=TOK/'validation.json';OUT=SRC/'validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable sealed RoMa validation exists')
 source_validation=json.load(open(VALID));assert source_validation['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_PREJOIN_VALIDATION_PASS';ordinals=[];total_scores=0;shards=[]
 for s in range(8):
  p=SRC/f'shard{s:02d}/result.json';x=json.load(open(p));source=TOK/f'shard{s:02d}/payload.pt';checks={'envelope':x['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_ROMA_PREJOIN_SHARD_READY' and x['shard']==s and x['logical_sha256']==logical(x),'source':x['bindings']['source_payload_sha256']==sha(source),'zero_target':x['target_role_read_count']==x['target_insertion_count']==x['model_update_count']==0,'score_count':x['sealed_roma_candidate_score_count']==128*len(x['rows'])};ok=True
  for r in x['rows']:
   ordinals.append(int(r['query_ordinal']));c=r['candidates'];axis=[int(z['physical_row']) for z in c];ok &= len(c)==128 and [z['candidate_position'] for z in c]==list(range(128)) and axis==sorted(axis) and len(set(axis))==128 and r['target_role_read_count']==r['target_insertion_count']==0
   for z in c:ok &= all(math.isfinite(float(z[k])) for k in ('raw_score','real_score','query_control_score','reference_control_score','visibility_mass')) and float(z['visibility_mass'])>=0 and len(z['query_map_sha256'])==len(z['reference_map_sha256'])==64
  checks['rows']=ok;total_scores+=x['sealed_roma_candidate_score_count'];passed=all(checks.values());shards.append({'shard':s,'sha256':sha(p),'checks':checks,'pass':passed,'query_count':len(x['rows'])})
 checks={'all_shards':all(x['pass'] for x in shards),'all_31_ordinals':sorted(ordinals)==list(range(31)),'unique_queries':len(set(ordinals))==31,'score_count_31x128':total_scores==31*128};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_roma_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_ROMA_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_ROMA_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(ordinals),'sealed_roma_candidate_score_count':total_scores,'target_join_manifest_read_count':0,'target_label_read_count':0,'sealed_result_reduction_authorized':False,'sealed_fullrank_prejoin_authorized':passed,'next_authorized_stage':'NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
