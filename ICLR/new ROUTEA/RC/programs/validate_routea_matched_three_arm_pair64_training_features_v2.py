#!/usr/bin/env python3
"""Independent validation of the fresh post-full Pair64 V2 envelope."""
from __future__ import annotations
import hashlib,json,os,shutil,struct,sys,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'programs'))
import materialize_routea_matched_three_arm_pair64_training_features_v1 as p1
import validate_routea_matched_three_arm_pair64_training_features_v1 as v1
V2=ROOT/'results/routea_matched_three_arm_pair64_training_features_v2';V1=ROOT/'results/routea_matched_three_arm_pair64_training_features_v1';FULLV=ROOT/'results/routea_matched_three_arm_fullnegative_features_v1/validation.json';V1V=V1/'independent_validation.json';OUT=V2/'independent_validation.json';LINEAGE=V2/'post_full_lineage.json';P2=ROOT/'programs/materialize_routea_matched_three_arm_pair64_training_features_v2.py';L2=ROOT/'slurm/routea_matched_three_arm_pair64_training_features_v2_10m.sbatch';PRIMARY=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md';PAIR_ADDENDUM=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_ADDENDUM_V1_20260902.md';RUNTIME_ADDENDUM=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_CURRENT_RUNTIME_REGRESSION_ADDENDUM_V1_20260902.md'
LINEAGE_KEYS={'schema_version','status','v1_payload_sha256','v1_validation_sha256','v1_validation_logical_sha256','v2_payload_sha256','semantic_bit_exact','full_validation_sha256','full_validation_logical_sha256','full_validation_mtime_ns','v1_producer_sha256','v1_validator_sha256','v2_wrapper_sha256','v2_validator_sha256','v2_launcher_sha256','primary_contract_sha256','pair_addendum_sha256','runtime_addendum_sha256','slurm_job_id','hostname','gpu_name','materialization_started_unix_ns','materialization_finished_unix_ns','model_update_count','next_authorized_stage','logical_sha256'}
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as h:
  while block:=h.read(8*1024*1024):d.update(block)
 return d.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def logical(value):return canonical({k:v for k,v in value.items() if k!='logical_sha256'})
def tensor_sha(value):return hashlib.sha256(torch.as_tensor(value).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def equal(a,b):
 if type(a) is not type(b):return False
 if isinstance(a,torch.Tensor):return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and tensor_sha(a)==tensor_sha(b)
 if isinstance(a,dict):return list(a)==list(b) and all(type(x) is type(y) for x,y in zip(a,b)) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,float):return struct.pack('!d',a)==struct.pack('!d',b)
 return a==b
def atomic(path,value):
 fd,name=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.partial',dir=path.parent);tmp=Path(name)
 try:
  with os.fdopen(fd,'w') as h:json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
  os.replace(tmp,path)
 finally:tmp.unlink(missing_ok=True)
def main():
 if OUT.exists():raise RuntimeError('immutable Pair64 V2 validation exists')
 temp_root=Path(tempfile.mkdtemp(prefix='.pair64-v2-validation-',dir=V2.parent));core_path=temp_root/'v1_core.json'
 try:
  v1.OUT_ROOT=V2;v1.PAYLOAD_PATH=V2/'payload.pt';v1.RECEIPT_PATH=V2/'receipt.json';v1.OUT=core_path;v1.main()
  core=json.loads(core_path.read_text());lineage=json.loads(LINEAGE.read_text());full=json.loads(FULLV.read_text());oldv=json.loads(V1V.read_text());new=torch.load(V2/'payload.pt',map_location='cpu',weights_only=False);old=torch.load(V1/'payload.pt',map_location='cpu',weights_only=False);same=equal(new,old)
  checks={
  'core':core.get('status')=='ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_INDEPENDENT_VALIDATION_PASS' and core.get('logical_sha256')==logical(core) and len(core.get('checks',{}))==18 and all(x is True for x in core.get('checks',{}).values()) and core.get('producer_payload_sha256')==sha(V2/'payload.pt'),
  'full_predecessor':full.get('status')=='ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_VALIDATED' and full.get('logical_sha256')==logical(full) and len(full.get('checks',{}))==7 and all(x is True for x in full.get('checks',{}).values()) and full.get('next_authorized_stage')=='MATCHED_THREE_ARM_CROSSFIT_AFTER_PAIR64' and lineage.get('full_validation_sha256')==sha(FULLV) and lineage.get('full_validation_logical_sha256')==full.get('logical_sha256') and lineage.get('full_validation_mtime_ns')==FULLV.stat().st_mtime_ns and int(lineage.get('materialization_started_unix_ns',0))>FULLV.stat().st_mtime_ns,
  'v1_authority':oldv.get('status')=='ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_INDEPENDENT_VALIDATION_PASS' and oldv.get('logical_sha256')==logical(oldv) and len(oldv.get('checks',{}))==18 and all(x is True for x in oldv.get('checks',{}).values()) and oldv.get('producer_payload_sha256')==sha(V1/'payload.pt') and lineage.get('v1_payload_sha256')==sha(V1/'payload.pt') and lineage.get('v1_validation_sha256')==sha(V1V) and lineage.get('v1_validation_logical_sha256')==oldv.get('logical_sha256'),
  'lineage':set(lineage)==LINEAGE_KEYS and lineage.get('schema_version')=='routea_matched_three_arm_pair64_training_features_v2_post_full_lineage_20260902' and lineage.get('status')=='ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_POST_FULL_READY' and lineage.get('logical_sha256')==logical(lineage) and isinstance(lineage.get('slurm_job_id'),int) and isinstance(lineage.get('hostname'),str) and bool(lineage.get('hostname')) and isinstance(lineage.get('gpu_name'),str) and bool(lineage.get('gpu_name')) and int(lineage.get('materialization_finished_unix_ns',0))>=int(lineage.get('materialization_started_unix_ns',0)) and lineage.get('model_update_count')==0 and lineage.get('next_authorized_stage')=='PAIR64_V2_INDEPENDENT_VALIDATION',
  'semantic_bit_exact':same and lineage.get('semantic_bit_exact') is True and lineage.get('v2_payload_sha256')==sha(V2/'payload.pt'),
  'bindings':lineage.get('v1_producer_sha256')==sha(ROOT/'programs/materialize_routea_matched_three_arm_pair64_training_features_v1.py') and lineage.get('v1_validator_sha256')==sha(ROOT/'programs/validate_routea_matched_three_arm_pair64_training_features_v1.py') and lineage.get('v2_wrapper_sha256')==sha(P2) and lineage.get('v2_validator_sha256')==sha(Path(__file__).resolve()) and lineage.get('v2_launcher_sha256')==sha(L2) and lineage.get('primary_contract_sha256')==sha(PRIMARY) and lineage.get('pair_addendum_sha256')==sha(PAIR_ADDENDUM) and lineage.get('runtime_addendum_sha256')==sha(RUNTIME_ADDENDUM),
  }
  passed=all(checks.values());value={'schema_version':'routea_matched_three_arm_pair64_training_features_v2_validation_20260902','status':'ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED' if passed else 'ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATION_ABORT','checks':checks,'v2_payload_sha256':sha(V2/'payload.pt'),'v1_core_validation_logical_sha256':core.get('logical_sha256'),'v2_lineage_sha256':sha(LINEAGE),'v2_lineage_logical_sha256':lineage.get('logical_sha256'),'full_validation_sha256':sha(FULLV),'full_validation_logical_sha256':full.get('logical_sha256'),'model_update_count':0,'next_authorized_stage':'MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_TRAINING' if passed else None,'logical_sha256':''};value['logical_sha256']=logical(value);atomic(OUT,value);print(json.dumps({'status':value['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
 finally:shutil.rmtree(temp_root,ignore_errors=True)
if __name__=='__main__':main()
