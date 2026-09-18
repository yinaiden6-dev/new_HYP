#!/usr/bin/env python3
"""Freeze a fixed-panel development experiment before new natural fits."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
def bind(s):
 p=ROOT/s;return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
public={
 'program':'programs/run_rc_fixed_panels_train269_group_risk_v1.py',
 'validator':'programs/validate_rc_fixed_panels_train269_group_risk_v1.py',
 'historic_input_validator':'programs/validate_rc_fixed269_historic_inputs_v1.py',
 'freezer':'programs/freeze_rc_fixed_panels_train269_group_risk_v1.py',
 'reporter':'programs/report_rc_fixed_panels_train269_group_risk_v1.py',
 'input_helper':'programs/materialize_rc_new_hyp593_inputs_v1.py',
 'feature_and_image_fit_helper':'programs/run_rc_new_hyp593_oof5_v1.py',
 'group_fit_helper':'programs/run_rc_h593_group_risk_strong_base_v1.py',
 'original_parameter_seal':'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json',
 'feature_authority':'registry/rc_new_hyp593_feature_authority_v1_20260911.json',
 'panel_manifest':'results/rc_fixed_panels_train269_group_risk_v1/panel_manifest.json',
 'train_labels':'results/rc_fixed_panels_train269_group_risk_v1/train_roles.json',
 'metadata_validation':'results/rc_fixed_panels_train269_group_risk_v1/metadata_validation.json',
 'plan':'plan/RC_FIXED_PANELS_TRAIN269_GROUP_RISK_V1_20260911.md',
 'launcher':'slurm/rc_fixed_panels_train269_group_risk_v1.sbatch',
}
evaluation={
 'eval_labels':'results/rc_fixed_panels_train269_group_risk_v1/eval_curator_roles.json',
 'original128_curator':'results/rc_original7_expanded_eval128_manifest_v1/curator_roles.json',
 'h593_curator':'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json',
 'original_scope_exclusions':'registry/rc_eval128_identity_group_exclusion_v1_20260910.json',
 'original32_result':'results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json',
 'original128_result':'results/rc_original7_eval128_full_evidence_v1/result.json',
}
value=dict(status='FIXED_OLD_PANELS_FRESH269_GROUP_RISK_AUTHORIZED',recorded_utc=datetime.now(timezone.utc).isoformat(),user_objective='RAW to original model to theory-guided optimization on identical EVAL32 and EVAL128 panels',historically_opened_development=True,primary_model='GROUP269',primary_comparator='ORIGINAL7 ec7',mechanism_control='IMAGE269 same rows architecture and update budget',train_images=269,train_identities=32,train_groups=32,models=['ORIGINAL7','IMAGE269','GROUP269'],new_fit_initialization='zero',new_fit_steps=2000,training_risk='original full-C128 SIGN with wrong term coefficient4; only image versus group mean changes',prior_593_fold_weights_allowed=False,full_eval_image_identity_group_exclusion=True,acceptance='observed paired net improvement per fixed panel; zero-loss not required; report statistical uncertainty separately',cutoff_UTC='2026-09-11T16:00:00+00:00',deployment_change=False,sources={k:bind(v) for k,v in public.items()},evaluation_sources={k:bind(v) for k,v in evaluation.items()})
p=ROOT/'registry/rc_fixed_panels_train269_group_risk_authority_v1_20260911.json'
with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(dict(authority=bind(str(p.relative_to(ROOT))),train_images=269,panels={'EVAL32':32,'EVAL128':128})))
