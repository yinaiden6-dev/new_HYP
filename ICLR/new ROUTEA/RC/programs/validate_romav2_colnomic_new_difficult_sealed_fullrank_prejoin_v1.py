#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1';TOK=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_prejoin_v1';ROMA=ROOT/'results/romav2_colnomic_new_difficult_sealed_roma_prejoin_v1/validation.json';OUT=SRC/'validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable fullrank validation exists')
 roma=json.load(open(ROMA));assert roma['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_ROMA_PREJOIN_VALIDATION_PASS';token={}
 for s in range(8):
  p=torch.load(TOK/f'shard{s:02d}/payload.pt',map_location='cpu',weights_only=False,mmap=True)
  for r in p['records']:token[int(r['query_ordinal'])]=r
 ordinals=[];shards=[];max_diff=0.0
 for s in range(8):
  p=SRC/f'shard{s:02d}/result.json';x=json.load(open(p));checks={'envelope':x['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_SHARD_READY' and x['shard']==s and x['logical_sha256']==logical(x),'zero_target':x['target_or_label_read_count']==x['target_join_manifest_read_count']==0,'shape':x['exact_label_count']==5412 and x['query_count']==len(x['rows'])};ok=True
  for r in x['rows']:
   o=int(r['query_ordinal']);ordinals.append(o);rank=list(map(int,r['ranked_physical_rows']));score=list(map(float,r['ranked_score_values']));t=token[o];expected=list(map(int,t['candidate_ranked_physical_rows']));score_map={int(row):float(v) for row,v in zip(t['candidate_physical_rows'],t['candidate_raw_scores'].tolist(),strict=True)};ok &= len(rank)==len(set(rank))==len(score)==5412 and all(math.isfinite(v) for v in score) and all(score[i]>=score[i+1] for i in range(len(score)-1)) and rank[:128]==expected and r['target_or_label_read_count']==0
   d=max(abs(score[i]-score_map[rank[i]]) for i in range(128));max_diff=max(max_diff,d);ok &= d==0.0
  checks['rows_and_c128_exact']=ok;passed=all(checks.values());shards.append({'shard':s,'sha256':sha(p),'checks':checks,'pass':passed,'query_count':len(x['rows'])})
 checks={'all_shards':all(x['pass'] for x in shards),'all_31_ordinals':sorted(ordinals)==list(range(31)),'unique_queries':len(set(ordinals))==31,'token_population':len(token)==31,'top128_score_max_abs_zero':max_diff==0.0};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_fullrank_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(ordinals),'exact_label_count':5412,'top128_score_max_abs':max_diff,'target_join_manifest_read_count':0,'target_label_read_count':0,'candidate_binding_prejoin_authorized':passed,'sealed_result_reduction_authorized':False,'next_authorized_stage':'NEW_DIFFICULT_SEALED_CANDIDATE_BINDING_PREJOIN' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
