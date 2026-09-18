#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REAL=ROOT/'results/romav2_colnomic_difficult90_real_prejoin_v1';SRC=ROOT/'results/romav2_colnomic_difficult90_cbind_prejoin_v1';OUT=SRC/'validation.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable difficult90 C_BIND validation exists')
 ordinals=[];shards=[]
 for s in range(10):
  rp=REAL/f'shard{s:02d}/result.json';real=json.load(open(rp));p=SRC/f'shard{s:02d}/result.json';x=json.load(open(p));rm={int(r['query_ordinal']):r for r in real['rows']};checks={'envelope':x['status']=='ROMAV2_COLNOMIC_DIFFICULT90_CBIND_PREJOIN_SHARD_READY' and x['shard']==s and x['logical_sha256']==logical(x),'source':x['source_real_sha256']==sha(rp),'zero_model_target':x['model_forward_count']==x['target_role_read_count']==x['target_insertion_count']==0,'rows':len(x['rows'])==9};ok=True
  for r in x['rows']:
   o=int(r['query_ordinal']);ordinals.append(o);rr=rm[o];c=r['candidates'];ok &= r['query_id']==rr['query_id'] and r['opened_split']==rr['opened_split'] and len(c)==128 and r['binding_shift']==64 and r['binding_fixed_point_count']==0
   for dest,z in enumerate(c):
    src=(dest+64)%128;a=rr['candidates'][dest];b=rr['candidates'][src];ok &= z['candidate_position']==dest and z['physical_row']==a['physical_row'] and z['raw_score']==a['raw_score'] and z['binding_source_position']==src and z['binding_source_physical_row']==b['physical_row'] and src!=dest and all(z[k]==b[k] for k in ('real_score','query_control_score','reference_control_score','visibility_mass','query_map_sha256','reference_map_sha256'))
  checks['exact_reindex']=ok;passed=all(checks.values());shards.append({'shard':s,'checks':checks,'pass':passed,'sha256':sha(p)})
 checks={'all_shards':all(x['pass'] for x in shards),'all_90_ordinals':sorted(ordinals)==list(range(90)),'unique_queries':len(set(ordinals))==90};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_difficult90_cbind_prejoin_validation_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_CBIND_PREJOIN_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_DIFFICULT90_CBIND_PREJOIN_VALIDATION_FAIL','checks':checks,'shards':shards,'query_count':len(ordinals),'model_forward_count':0,'target_label_read_count':0,'reduction_authorized':passed,'next_authorized_stage':'DIFFICULT90_FROZEN_REDUCTION' if passed else None};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
