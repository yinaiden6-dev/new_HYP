#!/usr/bin/env python3
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs')]
import materialize_rc_new_hyp593_inputs_v1 as M
TRAIN=ROOT/'results/rc_convex_train_loss_cause_v1';DIAG=ROOT/'results/rc_opened_convex_loss_rescue_cones_v1';PREV=ROOT/'results/rc_fixed_panels_train269_group_risk_v1'
sources={
 'program':ROOT/'programs/run_rc_convex_cause_isolation_v1.py',
 'core':ROOT/'programs/rc_convex_loss_cause_core_v1.py',
 'unit_cost_program':ROOT/'programs/run_rc_unit_cost96_cause_v1.py',
 'collector':ROOT/'programs/collect_rc_convex_cause_isolation_v1.py',
 'prepare':ROOT/'programs/prepare_rc_convex_cause_inputs_v1.py',
 'freezer':Path(__file__).resolve(),
 'input_helper':ROOT/'programs/materialize_rc_new_hyp593_inputs_v1.py',
 'prediction_helper':ROOT/'programs/run_rc_new_hyp593_oof5_v1.py',
 'core_qualification':ROOT/'results/rc_convex_cause_core_preflight_v1/result.json',
 'train_problem':TRAIN/'train_problem.json',
 'input_seal':TRAIN/'input_seal.json',
 'original_seal':ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json',
 'previous_payload':PREV/'fits/payload.pt',
 'previous_receipt':PREV/'fits/receipt.json',
 'previous_fit_validation':PREV/'fits/validation.json',
 'previous_authority':ROOT/'registry/rc_fixed_panels_train269_group_risk_authority_v1_20260911.json',
 'plan':ROOT/'plan/RC_CONVEX_CAUSE_ISOLATION_V1_20260912.md',
 **{k+'_launcher':ROOT/'slurm'/f'rc_convex_cause_{k}_v1.sbatch' for k in ('train','cones','collect')},
}
eval_sources={'result':PREV/'result.json','result_validation':PREV/'result_validation.json','eval_labels':PREV/'eval_curator_roles.json','panel_manifest':PREV/'panel_manifest.json'}
qual=M.read(sources['core_qualification']);M.need('PASS' in qual['status'],'CORE_QUALIFIED');M.need(len(M.read(DIAG/'inputs/manifest.json')['cones'])==6,'SIX_FIXED_DIAGNOSTIC_CONES')
a=dict(status='CONVEX_CAUSE_ISOLATION_BOUNDED_CONTINUATION_AUTHORIZED',recorded_utc=M.datetime.now(M.timezone.utc).isoformat(),user_authorization='2026-09-12 user explicit go continuing annotated cause-isolation and minimum necessary experiment',scope='one TRAIN-only logged-loss optimization, one original96 unit-cost control, six quarantined analytical cones; no other research restarted',sources={k:M.bind(p) for k,p in sources.items()},diagnostic_sources={'manifest':M.bind(DIAG/'inputs/manifest.json')},evaluation_sources={k:M.bind(p) for k,p in eval_sources.items()},train_only_conditions=2,unit_cost_control=1,parameter_box=64,max_cuts=256,max_search_seconds=420,gap_tolerance=1e-5,old_head_retraining=False,explicit_L2_added=False,diagnostic_coefficients_allowed_in_training=False,automatic_deployment=False,original_experiment_deadline_not_globally_extended=True)
p=ROOT/'registry/rc_convex_cause_isolation_authority_v1_20260912.json';M.write(p,a);print(json.dumps(M.bind(p)),flush=True)
