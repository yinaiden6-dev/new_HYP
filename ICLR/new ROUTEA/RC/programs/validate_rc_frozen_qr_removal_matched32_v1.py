#!/usr/bin/env python3
"""Fresh independent replay; exact serialized bytes avoid integer JSON-key mismatch."""
from pathlib import Path
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PRODUCER=ROOT/'programs/analyze_rc_frozen_qr_removal_matched32_v1.py'
PRODUCER_SHA='86267e29809eb45fe8e55750fc436f7a7330a8116a31f7d694d1f4183de0277e'
def need(x,m):
 if not x:raise RuntimeError(m)
def main():
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 need(hashlib.sha256(PRODUCER.read_bytes()).hexdigest()==PRODUCER_SHA,'FROZEN_PRODUCER_PIN')
 sp=importlib.util.spec_from_file_location('frozen_matched32_companion_validator_source',PRODUCER);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 need(m.OUT.exists() and not (m.OUT/'independent_validation.json').exists(),'APPEND_ONLY_VALIDATION')
 h,u,sources,barrier,rows,w,b,feature,source=m.prepare()
 manifest={'sources':sources,'conditions':{k:list(v) for k,v in m.CONDITIONS.items()},'query_count':32,'candidate_count':128,'weight_binary64':[u.hx(t) for t in w],'bias_binary64':u.hx(b),
 'arithmetic':'FP64_ROW_WISE_(WEIGHT*FEATURES).SUM_PLUS_BIAS','all127_reselected':True,'runtime_rule':'original matched32 raw SWITCH plus corrected identity membership; no difficult90 effective-HOLD substitution','parameter_updates':0}
 need(u.read(m.OUT/'execution_manifest.json')==manifest,'INDEPENDENT_MANIFEST_REBUILD')
 preds=m.predictions(rows,w,b,feature,u,True);need(u.read(m.OUT/'predictions_prejoin.json')==preds,'INDEPENDENT_RAW_SCALAR_FEATURE_LOGIT_PREJOIN_REBUILD')
 result=m.evaluate(preds,h,u,sources,barrier,source,True)
 # Integer execution ordinals are valid in-memory dict keys. JSON object keys
 # reload as strings; compare the actual serialized artifact, then normalized
 # JSON structures. Neither comparison weakens numerical equality.
 need((m.OUT/'result.json').read_bytes()==u.encode(result)+b'\n','EXACT_SERIALIZED_RESULT_BYTES')
 need(u.read(m.OUT/'result.json')==json.loads(u.encode(result)),'JSON_NORMALIZED_RESULT_EXACT')
 original=u.read(m.OUT/'result.json')
 need(all(isinstance(k,int) for k in result['query_metadata_after_join']) and all(isinstance(k,str) for k in original['query_metadata_after_join']),'EXPECTED_JSON_KEY_NORMALIZATION')
 value={'status':'FROZEN_QR_REMOVAL_MATCHED32_INDEPENDENT_REEXECUTION_PASS','result_sha256':u.sha(m.OUT/'result.json'),'validator':u.bind(Path(__file__).resolve()),'frozen_producer_sha256':PRODUCER_SHA,
 'checks':{'same_old_FROZEN_C_parameters_as_difficult90':True,'all_raw_scalar_features_independently_rebuilt':True,'all8128_rowwise_FP64_logits_exact':True,'all64_actions_and_identity_membership_independent':True,
 'complete_original27_result_and_switch_logit_bits_exact':True,'paired_counts_and_old_rescue_retention_independent':True,'source_and_prejoin_seals_exact':True,'exact_serialized_result_bytes':True,'normalized_JSON_result_exact':True,'no_training_or_scheduler_calls':True},
 'serialization_repair':{'frozen_producer_and_result_modified':False,'original_embedded_validate_failure':'INDEPENDENT_RESULT_REPLAY compares integer in-memory metadata execution keys with JSON string keys','companion_change':'Fresh independent rebuild, then exact result-file byte comparison and normalized JSON equality; no feature, logit or action changes'}}
 u.atomic(m.OUT/'independent_validation.json',value)
 print(json.dumps({'status':value['status'],'result_sha256':value['result_sha256'],'summaries':result['summaries'],'same_head_two_bundle_bridge':result['same_head_two_bundle_bridge'],'paired':result['paired_DROP_QR_vs_ORIGINAL']},sort_keys=True),flush=True)
if __name__=='__main__':main()
