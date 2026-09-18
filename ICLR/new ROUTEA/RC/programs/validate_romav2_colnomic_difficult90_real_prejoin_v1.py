#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/romav2_colnomic_difficult90_real_prejoin_v1';TOK=ROOT/'results/romav2_colnomic_difficult90_token_fullrank_prejoin_v1';AUTH=TOK/'validation.json';OUT=SRC/'validation.json';PROGRAM=ROOT/'programs/materialize_romav2_colnomic_difficult90_real_shard_v1.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable difficult90 REAL validation exists')
 assert json.load(open(AUTH))['status']=='ROMAV2_COLNOMIC_DIFFICULT90_TOKEN_FULLRANK_PREJOIN_VALIDATION_PASS';ordinals=[];shards=[]
 for s in range(10):
  p=SRC/f'shard{s:02d}/result.json';x=json.load(open(p));source=TOK/f'shard{s:02d}/payload.pt';t=torch.load(source,map_location='cpu',weights_only=False,mmap=True);tm={int(r['query_ordinal']):r for r in t['records']};checks={'envelope':x['status']=='ROMAV2_COLNOMIC_DIFFICULT90_REAL_PREJOIN_SHARD_READY' and x['shard']==s and x['logical_sha256']==logical(x),'source':x['bindings']['source_payload_sha256']==sha(source) and x['bindings']['source_validation_sha256']==sha(AUTH),'zero_target':x['target_role_read_count']==x['target_insertion_count']==x['model_update_count']==0,'row_count':len(x['rows'])==9};ok=True
  for r in x['rows']:
   o=int(r['query_ordinal']);ordinals.append(o);q=tm[o];axis=list(map(int,q['candidate_physical_rows']));raw={row:float(v) for row,v in zip(axis,q['candidate_raw_scores'].tolist(),strict=True)};c=r['candidates'];ok &= r['query_id']==q['query_id'] and r['opened_split']==q['opened_split'] and len(c)==128 and [z['candidate_position'] for z in c]==list(range(128)) and [int(z['physical_row']) for z in c]==axis and r['target_role_read_count']==r['target_insertion_count']==0
   for z in c:ok &= float(z['raw_score'])==raw[int(z['physical_row'])] and all(math.isfinite(float(z[k])) for k in ('real_score','query_control_score','reference_control_score','visibility_mass')) and float(z['visibility_mass'])>=0 and len(z['query_map_sha256'])==len(z['reference_map_sha256'])==64
  checks['records']=ok;passed=all(checks.values());shards.append({'shard':s,'sha256':sha(p),'checks':checks,'pass':passed,'query_count':9})
 checks={'all_shards':all(x['pass'] for x in shards),'all_90_ordinals':sorted(ordinals)==list(range(90)),'unique_queries':len(set(ordinals))==90};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_difficult90_real_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_REAL_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_DIFFICULT90_REAL_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(ordinals),'real_program_sha256':sha(PROGRAM),'target_label_read_count':0,'cbind_reindex_authorized':passed,'next_authorized_stage':'DIFFICULT90_CBIND_REINDEX' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
