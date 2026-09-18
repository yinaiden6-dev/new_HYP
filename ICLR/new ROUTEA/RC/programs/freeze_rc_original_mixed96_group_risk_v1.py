#!/usr/bin/env python3
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs')]
import materialize_rc_new_hyp593_inputs_v1 as M
OUT=ROOT/'results/rc_original_mixed96_group_risk_v1';PREV=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
public={
 'program':ROOT/'programs/run_rc_original_mixed96_group_risk_v1.py',
 'validator':ROOT/'programs/validate_rc_original_mixed96_group_risk_v1.py',
 'reporter':ROOT/'programs/report_rc_original_mixed96_group_risk_v1.py',
 'freezer':Path(__file__).resolve(),
 'input_helper':ROOT/'programs/materialize_rc_new_hyp593_inputs_v1.py',
 'feature_helper':ROOT/'programs/run_rc_new_hyp593_oof5_v1.py',
 'legacy_runner':ROOT/'programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py',
 'original_seal':ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json',
 'feature_authority':ROOT/'registry/rc_new_hyp593_feature_authority_v1_20260911.json',
 'previous_authority':ROOT/'registry/rc_fixed_panels_train269_group_risk_authority_v1_20260911.json',
 'previous_payload':PREV/'fits/payload.pt',
 'previous_receipt':PREV/'fits/receipt.json',
 'previous_fit_validation':PREV/'fits/validation.json',
 'panel_manifest':PREV/'panel_manifest.json',
 'train_roles':OUT/'train_roles.json',
 'train_inputs':OUT/'train_inputs.pt',
 'train_input_seal':OUT/'train_input_seal.json',
 'plan':ROOT/'plan/RC_ORIGINAL_MIXED96_GROUP_RISK_V1_20260911.md',
 'launcher':ROOT/'slurm/rc_original_mixed96_group_risk_v1.sbatch',
}
evaluation={'eval_labels':PREV/'eval_curator_roles.json','previous_result':PREV/'result.json','previous_result_validation':PREV/'result_validation.json'}
M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_CUTOFF');M.need(M.read(OUT/'train_input_seal.json')['status']=='ORIGINAL_MIXED96_INPUTS_FROZEN','INPUT_SEAL')
a=dict(status='ORIGINAL_MIXED96_GROUP_RISK_AUTHORIZED',recorded_utc=M.datetime.now(M.timezone.utc).isoformat(),sources={k:M.bind(p) for k,p in public.items()},evaluation_sources={k:M.bind(p) for k,p in evaluation.items()},old_head_training_updates=0,old_predictions_reused=True,new_head_count=1,new_training_steps=2000,sole_factor='within-pool group weights; original PAIR64+FULL32 mixture1:1',primary='GROUP_MIXED96 versus frozen ORIGINAL7 on each old panel',acceptance='observed rescue minus loss >0 per panel, zero-loss not required',historically_opened_development=True,cutoff_UTC=M.DEADLINE.isoformat(),deployment_changed=False)
p=ROOT/'registry/rc_original_mixed96_group_risk_authority_v1_20260911.json';M.write(p,a);print(json.dumps(M.bind(p)),flush=True)
