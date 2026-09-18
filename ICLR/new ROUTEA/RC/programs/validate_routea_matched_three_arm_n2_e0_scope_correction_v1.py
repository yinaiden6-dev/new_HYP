#!/usr/bin/env python3
"""Validate the append-only N2 E0 scope correction."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
ADDENDUM=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_ADDENDUM_V1_20260902.md'
E0_CONTRACT=ROOT/'plan/ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_V1_20260902.md'
E0=ROOT/'results/routea_matched_three_arm_n2_e0_v1'
PAYLOAD=E0/'payload.pt';RESULT=E0/'result.json';VALIDATION=E0/'independent_validation.json'
SOURCE=ROOT/'results/romav2_colnomic_current_runtime_bridge_prejoin_v1'
OUT=ROOT/'results/routea_matched_three_arm_n2_e0_scope_correction_v1'
EXPECTED={ADDENDUM:'dd3b667907ecebce34aec4d37cbfb5520c4e432ef71a630b74b9fcffd8b00264',E0_CONTRACT:'9c136e001334e82a885ee9aa8f247dd9b70a05a936edacce73db4d2cfcc61dcd',PAYLOAD:'9313e9290f4337c28e90fcbb2adcb8dc30ff2ef970c9202d9a1715e44ae0bca7',RESULT:'91b5685e2d3906dd86fac87705cefcb134d863248b2f9c989086feff6d1c4797',VALIDATION:'6e2975fa61642d07654b876d44d24bb803d46d085b7f1cd236acc08aaa00f1b1'}
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as h:
  while block:=h.read(8*1024*1024):d.update(block)
 return d.hexdigest()
def logical(value):return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def req(x,msg):
 if not x:raise RuntimeError(msg)
def main():
 req(not OUT.exists(),'immutable correction validation exists')
 for path,expected in EXPECTED.items():req(path.is_file() and sha(path)==expected,f'hash drift: {path}')
 addendum=ADDENDUM.read_text();markers=('ENGINEERING_PASS_WITH_SCOPE_CORRECTION','all 64 current-runtime queries replay raw image-token bytes','three fresh current64 fixtures','fresh-generate and seal both image','per-identity','per-reference trainable table','N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT')
 req(all(x in addendum for x in markers),'correction marker drift')
 result=json.loads(RESULT.read_text());validation=json.loads(VALIDATION.read_text())
 req(result['logical_sha256']==logical(result) and validation['logical_sha256']==logical(validation),'E0 logical drift')
 req(validation['status']=='ROUTEA_MATCHED_THREE_ARM_N2_E0_DESIGN_AND_LINEAGE_PREFLIGHT_VALIDATED' and len(validation['checks'])==10 and all(x is True for x in validation['checks'].values()) and validation['next_authorized_stage']=='N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT' and validation['scientific_GO_or_NO_GO'] is None and validation['n2_training_authorized'] is False and validation['external_execution_authorized'] is False,'E0 validation boundary drift')
 payload=torch.load(PAYLOAD,map_location='cpu',weights_only=False,mmap=True)
 current=payload['current64_replay'];fixtures=payload['fixtures']
 req(current['query_count']==current['token_byte_exact_count']==current['corrected_c128_exact_count']==64 and current['candidate_score_max_abs']==0.0,'all64 image/C128/score scope drift')
 by={x['query_id']:x for x in fixtures};req(set(by)=={'OUTCOME-0533','DIFFICULT-0128','NDV2-007-P01','DIFFICULT-0025'},'fixture population drift')
 for q in ('OUTCOME-0533','DIFFICULT-0128','NDV2-007-P01'):
  req(by[q]['current64_token_byte_exact'] is True and by[q]['current64_template_token_byte_exact'] is True,'fresh current64 image/template replay drift')
 d=by['DIFFICULT-0025'];req(d['fresh_repeat_token_byte_exact'] is True and d['image_tokens_sha256']==d['fresh_repeat_image_tokens_sha256'] and d['template_tokens_sha256']==d['fresh_repeat_template_tokens_sha256'],'25x29 repeat drift')
 source_count=0
 for shard in range(8):
  source=torch.load(SOURCE/f'shard{shard:02d}/payload.pt',map_location='cpu',weights_only=False,mmap=True)
  for record in source['records']:
   req('template_tokens' not in record and 'query_tokens' in record,'source bridge schema does not support corrected scope')
   source_count+=1
 req(source_count==64,'source bridge population drift')
 checks={'e0_artifact_hashes_and_logical':True,'e0_independent_validation_ten_checks':True,'all64_image_c128_score_scope':True,'three_current64_image_template_fixtures':True,'difficult0025_image_template_repeat':True,'source_bridge_template_field_absent':True,'reference_defined_no_identity_parameter_contract':True,'protected_access_and_updates_zero':True}
 value={'schema_version':'routea_matched_three_arm_n2_e0_scope_correction_validation_v1_20260902','status':'ROUTEA_MATCHED_THREE_ARM_N2_E0_SCOPE_CORRECTION_VALIDATED','claim_level':'ENGINEERING_PASS_WITH_SCOPE_CORRECTION_NO_TRAINING_NO_EXTERNAL_ACCESS','checks':checks,'bindings':{'addendum_sha256':sha(ADDENDUM),'e0_contract_sha256':sha(E0_CONTRACT),'e0_payload_sha256':sha(PAYLOAD),'e0_result_sha256':sha(RESULT),'e0_result_logical_sha256':result['logical_sha256'],'e0_validation_sha256':sha(VALIDATION),'e0_validation_logical_sha256':validation['logical_sha256'],'validator_sha256':sha(Path(__file__).resolve())},'evidence_scope':{'all64_image_token_exact_count':64,'all64_corrected_c128_exact_count':64,'all64_score_max_abs':0.0,'fresh_current64_image_template_exact_count':3,'fresh_25x29_repeat_image_template_exact_count':1,'unsupported_all64_template_bridge_claim':False},'access':{'internal_source_bridge_record_read_count':64,'external_query_read_count':0,'external_target_read_count':0,'sealed_read_count':0,'model_update_count':0},'n2_training_authorized':False,'external_execution_authorized':False,'scientific_GO_or_NO_GO':None,'next_authorized_stage':'N2_CURRENT_RUNTIME_987QUERY_TOKEN_CACHE_MATERIALIZATION_CONTRACT','logical_sha256':''};value['logical_sha256']=logical(value)
 OUT.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.n2-e0-scope-',dir=OUT.parent))
 try:
  p=stage/'independent_validation.json'
  with p.open('w') as h:json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
  os.rename(stage,OUT)
 finally:
  if stage.exists():shutil.rmtree(stage)
 print(json.dumps({'status':value['status'],'checks':checks,'next_authorized_stage':value['next_authorized_stage']},sort_keys=True))
if __name__=='__main__':main()
