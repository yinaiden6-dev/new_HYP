#!/usr/bin/env python3
"""Fresh post-full Pair64 materialization using the byte-frozen V1 producer."""
import json,os,shutil,socket,struct,sys,tempfile,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'programs'))
import materialize_routea_matched_three_arm_pair64_training_features_v1 as v1
OUT_ROOT=ROOT/'results/routea_matched_three_arm_pair64_training_features_v2';V1_ROOT=ROOT/'results/routea_matched_three_arm_pair64_training_features_v1';FULLV=ROOT/'results/routea_matched_three_arm_fullnegative_features_v1/validation.json';LINEAGE=OUT_ROOT/'post_full_lineage.json';V1V=V1_ROOT/'independent_validation.json';V2_VALIDATOR=ROOT/'programs/validate_routea_matched_three_arm_pair64_training_features_v2.py';V2_LAUNCHER=ROOT/'slurm/routea_matched_three_arm_pair64_training_features_v2_10m.sbatch';PRIMARY=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md';PAIR_ADDENDUM=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_ADDENDUM_V1_20260902.md';RUNTIME_ADDENDUM=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_CURRENT_RUNTIME_REGRESSION_ADDENDUM_V1_20260902.md'
def equal(a,b):
 if type(a) is not type(b):return False
 if isinstance(a,torch.Tensor):return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and v1.tensor_sha256(a)==v1.tensor_sha256(b)
 if isinstance(a,dict):return list(a)==list(b) and all(type(x) is type(y) for x,y in zip(a,b)) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,float):return struct.pack('!d',a)==struct.pack('!d',b)
 return a==b
def atomic_json(path,value):
 fd,name=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.partial',dir=path.parent);tmp=Path(name)
 try:
  with os.fdopen(fd,'w') as h:json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
  os.replace(tmp,path)
 finally:tmp.unlink(missing_ok=True)
def main():
 if OUT_ROOT.exists():raise RuntimeError('immutable Pair64 V2 exists')
 started=time.time_ns();full=json.loads(FULLV.read_text());oldv=json.loads(V1V.read_text())
 if not(full.get('status')=='ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_VALIDATED' and full.get('logical_sha256')==v1.logical_sha256(full) and len(full.get('checks',{}))==7 and all(x is True for x in full.get('checks',{}).values()) and full.get('query_count')==64 and full.get('target_role_read_count')==0 and full.get('model_update_count')==0 and full.get('next_authorized_stage')=='MATCHED_THREE_ARM_CROSSFIT_AFTER_PAIR64'):raise RuntimeError('full feature aggregate not validated before Pair64 V2')
 if not(oldv.get('status')=='ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_INDEPENDENT_VALIDATION_PASS' and oldv.get('logical_sha256')==v1.logical_sha256(oldv) and len(oldv.get('checks',{}))==18 and all(x is True for x in oldv.get('checks',{}).values()) and oldv.get('producer_payload_sha256')==v1.sha256_file(V1_ROOT/'payload.pt')):raise RuntimeError('Pair64 V1 authority drift')
 container=Path(tempfile.mkdtemp(prefix='.pair64-v2-publication-',dir=OUT_ROOT.parent));stage_root=container/'materialized'
 try:
  v1.OUT_ROOT=stage_root;sys.argv=[sys.argv[0]];v1.main();new=torch.load(stage_root/'payload.pt',map_location='cpu',weights_only=False);old=torch.load(V1_ROOT/'payload.pt',map_location='cpu',weights_only=False);same=equal(new,old)
  value={'schema_version':'routea_matched_three_arm_pair64_training_features_v2_post_full_lineage_20260902','status':'ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_POST_FULL_READY' if same else 'ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_POST_FULL_ABORT','v1_payload_sha256':v1.sha256_file(V1_ROOT/'payload.pt'),'v1_validation_sha256':v1.sha256_file(V1V),'v1_validation_logical_sha256':oldv['logical_sha256'],'v2_payload_sha256':v1.sha256_file(stage_root/'payload.pt'),'semantic_bit_exact':same,'full_validation_sha256':v1.sha256_file(FULLV),'full_validation_logical_sha256':full['logical_sha256'],'full_validation_mtime_ns':FULLV.stat().st_mtime_ns,'v1_producer_sha256':v1.sha256_file(ROOT/'programs/materialize_routea_matched_three_arm_pair64_training_features_v1.py'),'v1_validator_sha256':v1.sha256_file(ROOT/'programs/validate_routea_matched_three_arm_pair64_training_features_v1.py'),'v2_wrapper_sha256':v1.sha256_file(Path(__file__).resolve()),'v2_validator_sha256':v1.sha256_file(V2_VALIDATOR),'v2_launcher_sha256':v1.sha256_file(V2_LAUNCHER),'primary_contract_sha256':v1.sha256_file(PRIMARY),'pair_addendum_sha256':v1.sha256_file(PAIR_ADDENDUM),'runtime_addendum_sha256':v1.sha256_file(RUNTIME_ADDENDUM),'slurm_job_id':int(os.environ['SLURM_JOB_ID']) if os.environ.get('SLURM_JOB_ID') else None,'hostname':socket.gethostname(),'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'materialization_started_unix_ns':started,'materialization_finished_unix_ns':time.time_ns(),'model_update_count':0,'next_authorized_stage':'PAIR64_V2_INDEPENDENT_VALIDATION' if same else None,'logical_sha256':''};value['logical_sha256']=v1.logical_sha256(value);atomic_json(stage_root/'post_full_lineage.json',value);os.rename(stage_root,OUT_ROOT);print(json.dumps({'status':value['status'],'semantic_bit_exact':same,'gpu_name':value['gpu_name']},sort_keys=True));raise SystemExit(0 if same else 4)
 finally:
  if container.exists():shutil.rmtree(container)
if __name__=='__main__':main()
