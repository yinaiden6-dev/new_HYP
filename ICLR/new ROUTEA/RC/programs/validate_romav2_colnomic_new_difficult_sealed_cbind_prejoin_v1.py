#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/romav2_colnomic_new_difficult_sealed_cbind_prejoin_v1';TOK=ROOT/'results/romav2_colnomic_new_difficult_sealed_token_prejoin_v1';AUTH=ROOT/'results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1/validation.json';OUT=SRC/'validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable C_BIND validation exists')
 assert json.load(open(AUTH))['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_VALIDATION_PASS';token={}
 for s in range(8):
  p=torch.load(TOK/f'shard{s:02d}/payload.pt',map_location='cpu',weights_only=False,mmap=True)
  for r in p['records']:token[int(r['query_ordinal'])]=r
 ordinals=[];total=0;shards=[]
 for s in range(8):
  p=SRC/f'shard{s:02d}/result.json';x=json.load(open(p));checks={'envelope':x['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_CBIND_PREJOIN_SHARD_READY' and x['shard']==s and x['logical_sha256']==logical(x),'permutation':x['permutation']=={'candidate_count':128,'source_position_for_destination':'(destination+64)%128','shift':64,'fixed_point_count':0},'source':x['bindings']['source_payload_sha256']==sha(TOK/f'shard{s:02d}/payload.pt'),'zero_target':x['target_role_read_count']==x['target_insertion_count']==x['model_update_count']==0};ok=True
  for r in x['rows']:
   o=int(r['query_ordinal']);ordinals.append(o);t=token[o];axis=list(map(int,t['candidate_physical_rows']));raw={row:float(v) for row,v in zip(axis,t['candidate_raw_scores'].tolist(),strict=True)};c=r['candidates'];ok &= r['query_id']==t['query_id'] and r['binding_shift']==64 and r['binding_fixed_point_count']==0 and len(c)==128
   for dest,z in enumerate(c):
    src=(dest+64)%128;ok &= z['candidate_position']==dest and int(z['physical_row'])==axis[dest] and z['binding_source_position']==src and int(z['binding_source_physical_row'])==axis[src] and src!=dest and float(z['raw_score'])==raw[axis[dest]] and all(math.isfinite(float(z[k])) for k in ('real_score','query_control_score','reference_control_score','visibility_mass')) and len(z['query_map_sha256'])==len(z['reference_map_sha256'])==64
  checks['rows_exact']=ok;total+=x['sealed_cbind_candidate_score_count'];checks['score_count']=x['sealed_cbind_candidate_score_count']==128*len(x['rows']);passed=all(checks.values());shards.append({'shard':s,'sha256':sha(p),'checks':checks,'pass':passed,'query_count':len(x['rows'])})
 checks={'all_shards':all(x['pass'] for x in shards),'all_31_ordinals':sorted(ordinals)==list(range(31)),'unique_queries':len(set(ordinals))==31,'score_count_31x128':total==31*128,'token_population':len(token)==31};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_cbind_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_CBIND_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_CBIND_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(ordinals),'candidate_score_count':total,'target_join_manifest_read_count':0,'target_label_read_count':0,'top_level_prejoin_seal_authorized':passed,'sealed_result_reduction_authorized':False,'next_authorized_stage':'NEW_DIFFICULT_SEALED_TOP_LEVEL_PREJOIN_SEAL' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
