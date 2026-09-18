#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'programs'))
import run_romav2_colnomic_visibility_xf_six_case_v1 as core
SRC=ROOT/'results/romav2_colnomic_difficult90_real_prejoin_v1';AUTH=SRC/'validation.json';OUT=ROOT/'results/romav2_colnomic_difficult90_cbind_prejoin_v1'
def main():
 if OUT.exists():raise RuntimeError('immutable difficult90 C_BIND root exists')
 assert json.load(open(AUTH))['status']=='ROMAV2_COLNOMIC_DIFFICULT90_REAL_PREJOIN_VALIDATION_PASS';OUT.mkdir(parents=True,exist_ok=False)
 for s in range(10):
  source=SRC/f'shard{s:02d}/result.json';x=json.load(open(source));rows=[]
  for r in x['rows']:
   real=r['candidates'];c=[]
   for dest in range(128):
    src=(dest+64)%128;a=real[dest];b=real[src];c.append({'candidate_position':dest,'physical_row':a['physical_row'],'raw_score':a['raw_score'],'binding_source_position':src,'binding_source_physical_row':b['physical_row'],'real_score':b['real_score'],'query_control_score':b['query_control_score'],'reference_control_score':b['reference_control_score'],'visibility_mass':b['visibility_mass'],'query_map_sha256':b['query_map_sha256'],'reference_map_sha256':b['reference_map_sha256']})
   rows.append({'query_ordinal':r['query_ordinal'],'query_id':r['query_id'],'opened_split':r['opened_split'],'candidate_count':128,'binding_shift':64,'binding_fixed_point_count':0,'candidates':c,'target_role_read_count':0,'target_insertion_count':0})
  value={'schema_version':'rc_romav2_colnomic_difficult90_cbind_prejoin_shard_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_CBIND_PREJOIN_SHARD_READY','claim_level':'OPENED_REGRESSION_EXACT_REINDEX_OF_REAL_REFERENCE_PAYLOAD','shard':s,'rows':rows,'source_real_sha256':core.sha(source),'target_role_read_count':0,'target_insertion_count':0,'model_forward_count':0,'logical_sha256':''};value['logical_sha256']=core.logical(value);core.atomic(OUT/f'shard{s:02d}/result.json',value)
 print(json.dumps({'status':'ROMAV2_COLNOMIC_DIFFICULT90_CBIND_PREJOIN_COMPLETE','shard_count':10,'query_count':90,'model_forward_count':0},sort_keys=True))
if __name__=='__main__':main()
