#!/usr/bin/env python3
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'registry/romav2_colnomic_new_difficult_sealed_reduction_authority_v1_20260901.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(rel,status):
 p=ROOT/rel;x=json.load(open(p));
 if x.get('status')!=status:raise RuntimeError(f'bad status {rel}')
 return p,x
def main():
 if OUT.exists():raise RuntimeError('immutable sealed reduction authority exists')
 required={
  'model_lineage':('results/romav2_colnomic_new_difficult_model_lineage_v1/result.json','ROMAV2_COLNOMIC_NEW_DIFFICULT_MODEL_LINEAGE_FROZEN'),
  'frozen_gate':('results/romav2_colnomic_frozen_gate_definition_v1/validation.json','ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS'),
  'current_runtime_gate':('results/romav2_colnomic_current_runtime_frozen_gate_v1/independent_validation.json','ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_PASS'),
  'token_prejoin':('results/romav2_colnomic_new_difficult_sealed_token_prejoin_v1/validation.json','ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_TOKEN_PREJOIN_VALIDATION_PASS'),
  'real_prejoin':('results/romav2_colnomic_new_difficult_sealed_roma_prejoin_v1/validation.json','ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_ROMA_PREJOIN_VALIDATION_PASS'),
  'fullrank_prejoin':('results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1/validation.json','ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_VALIDATION_PASS'),
  'fullrank_prejoin_v2':('results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1/validation_v2.json','ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_FULLRANK_PREJOIN_VALIDATION_V2_PASS'),
  'cbind_prejoin':('results/romav2_colnomic_new_difficult_sealed_cbind_prejoin_v1/validation.json','ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_CBIND_PREJOIN_VALIDATION_PASS'),
 }
 bindings={};seal_paths=[]
 for name,(rel,status) in required.items():
  p,x=read(rel,status);bindings[name]={'path':rel,'sha256':sha(p),'status':status};seal_paths.append(p)
 manifest=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/binding_receipt.json';m=json.load(open(manifest));
 if m['status']!='NEW_DIFFICULT_SEALED_MANIFEST_BINDING_PASS':raise RuntimeError('manifest binding drift')
 bindings['manifest_binding']={'path':str(manifest.relative_to(ROOT)),'sha256':sha(manifest),'status':m['status']};seal_paths.append(manifest)
 contracts=['plan/ROMAV2_COLNOMIC_FULL_NEGATIVE_EXTERNAL_CONFIRMATION_CONTRACT_V1_20260831.md','plan/ROMAV2_COLNOMIC_NEW_DIFFICULT_C_BIND_CONTROL_V1_20260901.md','plan/ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUDIT_ADDENDUM_V1_20260901.md'];sources=['src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py','programs/reduce_romav2_colnomic_new_difficult_sealed_v1.py','programs/validate_romav2_colnomic_new_difficult_sealed_v1.py','programs/materialize_romav2_colnomic_new_difficult_sealed_token_shard_v1.py','programs/materialize_romav2_colnomic_new_difficult_sealed_roma_shard_v1.py','programs/materialize_romav2_colnomic_new_difficult_sealed_fullrank_shard_v1.py','programs/materialize_romav2_colnomic_new_difficult_sealed_cbind_shard_v1.py'];source_hashes={rel:sha(ROOT/rel) for rel in contracts+sources}
 artifacts={}
 for family,filename in [('token','payload.pt'),('real','result.json'),('fullrank','result.json'),('cbind','result.json')]:
  root={'token':ROOT/'results/romav2_colnomic_new_difficult_sealed_token_prejoin_v1','real':ROOT/'results/romav2_colnomic_new_difficult_sealed_roma_prejoin_v1','fullrank':ROOT/'results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1','cbind':ROOT/'results/romav2_colnomic_new_difficult_sealed_cbind_prejoin_v1'}[family];items=[]
  for s in range(8):
   p=root/f'shard{s:02d}'/filename;items.append({'shard':s,'path':str(p.relative_to(ROOT)),'sha256':sha(p)});seal_paths.append(p)
  artifacts[family]=items
 prejoin=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/prejoin_manifest.json';target_join=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/target_join_manifest.json';
 if sha(prejoin)!=m['outputs']['prejoin_manifest_sha256'] or sha(target_join)!=m['outputs']['target_join_manifest_sha256']:raise RuntimeError('manifest output hash drift')
 seal_paths.extend([prejoin,target_join]);
 for p in seal_paths:p.chmod(0o444)
 value={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_reduction_authority_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUTHORIZED','claim_level':'ONE_SHOT_DIRECTIONAL_REDUCTION_AUTHORITY_NOT_RESULT','bindings':bindings,'prejoin_artifacts':artifacts,'source_sha256':source_hashes,'manifest':{'prejoin_path':str(prejoin.relative_to(ROOT)),'prejoin_sha256':sha(prejoin),'target_join_path':str(target_join.relative_to(ROOT)),'target_join_sha256':sha(target_join),'query_count':31},'permissions':{'target_join_attempt_count':1,'authorized_target_join_reader_process_count':2,'reducer_target_join_read_count':1,'independent_validator_target_join_read_count':1,'sealed_model_forward_authorized':False,'sealed_roma_forward_authorized':False,'model_or_threshold_change_authorized':False,'all_31_denominator_required':True},'ranking_semantics':'MOVE_SELECTED_CHALLENGER_EXACT_LABEL_TO_FRONT_STABLE_REMAINDER','control_semantics':{'CBIND':'INDEPENDENT_SHIFT64_REFERENCE_PAYLOAD','Q':'(S10,S10,S01)_DIAGNOSTIC','R':'(S01,S10,S01)_DIAGNOSTIC','strict_spatial_causal_claim_authorized':False},'result_path':'results/romav2_colnomic_new_difficult_sealed_directional_v1/result.json','validation_path':'results/romav2_colnomic_new_difficult_sealed_directional_v1/independent_validation.json','logical_sha256':''};value['logical_sha256']=logical(value);OUT.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');OUT.chmod(0o444);print(json.dumps({'status':value['status'],'artifact_counts':{k:len(v) for k,v in artifacts.items()}},sort_keys=True))
if __name__=='__main__':main()
