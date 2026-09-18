#!/usr/bin/env python3
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs')]
import materialize_rc_new_hyp593_inputs_v1 as M
TRAIN=ROOT/'results/rc_convex_train_loss_cause_v1';UNIT=TRAIN/'unit_cost96';PREV=ROOT/'results/rc_fixed_panels_train269_group_risk_v1';CAUSE=ROOT/'results/rc_opened_convex_cause_readout_v1'
sources={
 'program':ROOT/'programs/run_rc_full_candidate_identity_loss_v1.py',
 'validator':ROOT/'programs/validate_rc_full_candidate_identity_loss_v1.py',
 'freezer':Path(__file__).resolve(),
 'input_helper':ROOT/'programs/materialize_rc_new_hyp593_inputs_v1.py',
 'unit_training_helper':ROOT/'programs/run_rc_unit_cost96_cause_v1.py',
 'legacy_runner':ROOT/'programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py',
 'prediction_helper':ROOT/'programs/run_rc_new_hyp593_oof5_v1.py',
 'feature_authority':ROOT/'registry/rc_new_hyp593_feature_authority_v1_20260911.json',
 'input_seal':TRAIN/'input_seal.json',
 'original_training_pack':ROOT/'results/rc_original_mixed96_group_risk_v1/train_inputs.pt',
 'original_training_seal':ROOT/'results/rc_original_mixed96_group_risk_v1/train_input_seal.json',
 'original_training_roles':ROOT/'results/rc_original_mixed96_group_risk_v1/train_roles.json',
 'original_head_seal':ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json',
 'previous_unit_fit':UNIT/'fit.json',
 'previous_unit_fit_validation':UNIT/'fit_validation.json',
 'previous_prediction_seal':UNIT/'prediction_seal.json',
 'previous_payload':UNIT/'predictions.pt',
 'previous_authority':ROOT/'registry/rc_convex_cause_isolation_authority_v1_20260912.json',
 'proposal':ROOT/'plan/RC_FULL_CANDIDATE_IDENTITY_LOSS_NEXT_STEP_V1_20260912.md',
 'execution_plan':ROOT/'plan/RC_FULL_CANDIDATE_IDENTITY_LOSS_EXECUTION_V1_20260912.md',
 'launcher':ROOT/'slurm/rc_full_candidate_identity_loss_v1.sbatch',
}
evaluation_sources={'previous_result':PREV/'result.json','previous_result_validation':PREV/'result_validation.json','eval_labels':PREV/'eval_curator_roles.json','panel_manifest':PREV/'panel_manifest.json','previous_cause_result':CAUSE/'result.json','previous_cause_validation':CAUSE/'independent_validation.json'}
v=M.read(UNIT/'fit_validation.json');M.need(v['status']=='UNIT_COST96_FRESH_PARAMETER_REPLAY_PASS' and v['fit']==M.bind(UNIT/'fit.json'),'EXISTING_UNIT_COST_CONTROL_VALID')
a=dict(status='FULL_CANDIDATE_IDENTITY_LOSS_SINGLE_FACTOR_AUTHORIZED',recorded_utc=M.datetime.now(M.timezone.utc).isoformat(),user_authorization='2026-09-12 continue/go on to execute fixed LISTWISE_UNIT1 proposal',sources={k:M.bind(p) for k,p in sources.items()},evaluation_sources={k:M.bind(p) for k,p in evaluation_sources.items()},primary_single_factor='LISTWISE_UNIT1 vs TRAIN_UNIT_COST7, FULL loss only',strong_comparator='ORIGINAL7 ec7 fixed old EVAL32=28 and EVAL128=99',models=['ORIGINAL7','TRAIN_UNIT_COST7','LISTWISE_UNIT1'],new_head_count=1,old_head_retraining=False,train_population={'PAIR':64,'FULL':32},steps=2000,temperature=1,PAIR_FULL_mixture=[1,1],candidate_count=128,acceptance='observed net rescue minus loss per fixed panel; no zero-loss requirement',historically_opened_development=True,automatic_deployment=False)
p=ROOT/'registry/rc_full_candidate_identity_loss_authority_v1_20260912.json';M.write(p,a);print(json.dumps(M.bind(p)),flush=True)
