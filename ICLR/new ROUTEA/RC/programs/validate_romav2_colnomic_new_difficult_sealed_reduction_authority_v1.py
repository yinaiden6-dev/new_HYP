#!/usr/bin/env python3
import hashlib,json,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];AUTH=ROOT/'registry/romav2_colnomic_new_difficult_sealed_reduction_authority_v1_20260901.json';OUT=ROOT/'results/romav2_colnomic_new_difficult_sealed_reduction_authority_v1/validation.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists():raise RuntimeError('immutable authority validation exists')
 a=json.load(open(AUTH));artifact_ok=True
 for items in a['prejoin_artifacts'].values():
  for x in items:
   p=ROOT/x['path'];artifact_ok &= p.is_file() and sha(p)==x['sha256'] and stat.S_IMODE(p.stat().st_mode)==0o444
 source_ok=all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in a['source_sha256'].items());binding_ok=True
 for x in a['bindings'].values():binding_ok &= (ROOT/x['path']).is_file() and sha(ROOT/x['path'])==x['sha256'] and json.load(open(ROOT/x['path']))['status']==x['status']
 m=a['manifest'];manifest_ok=sha(ROOT/m['prejoin_path'])==m['prejoin_sha256'] and sha(ROOT/m['target_join_path'])==m['target_join_sha256'] and stat.S_IMODE((ROOT/m['target_join_path']).stat().st_mode)==0o444
 checks={'envelope':a['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUTHORIZED' and a['logical_sha256']==logical(a),'artifacts':artifact_ok,'sources':source_ok,'bindings':binding_ok,'manifest':manifest_ok,'permissions':a['permissions']=={'target_join_attempt_count':1,'authorized_target_join_reader_process_count':2,'reducer_target_join_read_count':1,'independent_validator_target_join_read_count':1,'sealed_model_forward_authorized':False,'sealed_roma_forward_authorized':False,'model_or_threshold_change_authorized':False,'all_31_denominator_required':True},'result_absent':not (ROOT/a['result_path']).exists() and not (ROOT/a['validation_path']).exists()};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_reduction_authority_validation_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUTHORITY_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUTHORITY_VALIDATION_FAIL','checks':checks,'authority_sha256':sha(AUTH),'target_join_read_count':0,'sealed_reduction_authorized':passed};OUT.parent.mkdir(parents=True,exist_ok=False);OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
