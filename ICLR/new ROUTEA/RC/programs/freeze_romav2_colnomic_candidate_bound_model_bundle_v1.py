#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'isolated/romav2_colnomic_candidate_bound_no_regret_v1_20260901';REG=ROOT/'registry/romav2_colnomic_candidate_bound_no_regret_model_card_v1_20260901.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if OUT.exists() or REG.exists():raise RuntimeError('immutable model bundle exists')
 required={
  'model_lineage':('results/romav2_colnomic_new_difficult_model_lineage_v1/result.json','ROMAV2_COLNOMIC_NEW_DIFFICULT_MODEL_LINEAGE_FROZEN'),
  'gate_definition':('results/romav2_colnomic_frozen_gate_definition_v1/validation.json','ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS'),
  'current_runtime_oof':('results/romav2_colnomic_current_runtime_frozen_gate_v1/independent_validation.json','ROMAV2_COLNOMIC_CURRENT_RUNTIME_FROZEN_GATE_INDEPENDENT_VALIDATION_PASS'),
  'difficult90':('results/romav2_colnomic_difficult90_frozen_regression_v1/independent_validation.json','ROMAV2_COLNOMIC_DIFFICULT90_REGRESSION_INDEPENDENT_VALIDATION_PASS'),
  'sealed31':('results/romav2_colnomic_new_difficult_sealed_directional_v1/independent_validation.json','ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_DIRECTIONAL_INDEPENDENT_VALIDATION_PASS'),
 }
 evidence={}
 for name,(rel,status) in required.items():
  x=json.load(open(ROOT/rel));assert x['status']==status;evidence[name]={'path':rel,'sha256':sha(ROOT/rel),'status':status}
 sources=['src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py','src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py','programs/run_romav2_colnomic_visibility_xf_six_case_v1.py','programs/materialize_romav2_colnomic_current_runtime_bridge_roma_shard_v1.py','programs/reduce_romav2_colnomic_new_difficult_sealed_v1.py','plan/ROMAV2_COLNOMIC_NEW_DIFFICULT_C_BIND_CONTROL_V1_20260901.md','plan/ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUDIT_ADDENDUM_V1_20260901.md'];source_sha={x:sha(ROOT/x) for x in sources};lineage=json.load(open(ROOT/required['model_lineage'][0]));gate=json.load(open(ROOT/required['gate_definition'][0]));value={'schema_version':'rc_romav2_colnomic_candidate_bound_no_regret_model_card_v1_20260901','status':'ROMAV2_COLNOMIC_CANDIDATE_BOUND_NO_REGRET_MODEL_FROZEN','model_name':'RoMaV2-ColNomic-CandidateBound-NoRegret-V1','inference':{'gallery_physical_rows':5413,'corrected_exact_labels':5412,'natural_candidate_width':128,'feature_names':gate['feature_names'],'weight':gate['weight'],'bias':gate['bias'],'parameter_count':gate['parameter_count'],'switch_threshold':gate['switch_threshold'],'maximum_switches_per_query':1,'ranking_update':'MOVE_SELECTED_CHALLENGER_EXACT_LABEL_TO_FRONT_STABLE_REMAINDER','abstention':'HOLD_WHEN_MAX_LOGIT_LE_ZERO'},'models':{'romav2':lineage['romav2'],'dinov3':lineage['dinov3'],'colnomic':lineage['colnomic'],'gallery':lineage['gallery']},'source_sha256':source_sha,'evidence':evidence,'supported_claims':['target_free_full_C128_no_regret_correction_on_opened_difficult90','candidate_binding_is_required_for_observed_rescues','exact_instance_reference_conditioned_retrieval_correction'],'unsupported_claims':['strict_spatial_coordinate_causality','spatial_ownership','paper_strength_untouched_external_gain','identity_specific_or_target_mask_supervision'],'frozen_prohibitions':['no_threshold_tuning','no_candidate_width_tuning','no_pooling_or_feature_change','no_identity_specific_parameters','no_target_insertion','no_human_mask_box_point_polygon','no_reuse_of_difficult_or_new_difficult_for_selection'],'next_authorized_stage':'UNTOUCHED_EXTERNAL_DATASET_MANIFEST_AND_ONE_SHOT_CONFIRMATION','query_or_target_read_count':0,'logical_sha256':''};value['logical_sha256']=logical(value);OUT.mkdir(parents=True,exist_ok=False);(OUT/'bundle_manifest.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');readme=f"""# {value['model_name']}\n\nImmutable manifest: `bundle_manifest.json`.\n\nFrozen claim: target-free, candidate-bound, no-regret exact-instance retrieval correction.\nStrict spatial ownership and untouched external gain are not yet claimed.\n\nEvidence: current-runtime OOF 25/32 to 27/32; opened difficult90 61/90 to 69/90 with 8 rescues and 0 breaks; sealed new_difficult was ceiling-saturated at 31/31 and remained 31/31.\n\nNext: one untouched external dataset under the exact frozen manifest.\n""";(OUT/'README.md').write_text(readme);REG.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');
 for p in (OUT/'bundle_manifest.json',OUT/'README.md',REG):p.chmod(0o444)
 print(json.dumps({'status':value['status'],'bundle':str(OUT),'logical_sha256':value['logical_sha256']},sort_keys=True))
if __name__=='__main__':main()
