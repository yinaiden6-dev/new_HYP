#!/usr/bin/env python3
"""Build a reviewable V7 authority draft; never authorizes or submits execution."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--contract',default='plan/REFERENCE_CONDITIONED_GROUPED_RESIDUAL_P_ONLY_V7_20260908.md')
    p.add_argument('--e0',default='results/rc_grouped_residual_e0_v7/result.json')
    p.add_argument('--e0-status',default='RC_GROUPED_RESIDUAL_V7_INDEPENDENT_E0_PASS')
    p.add_argument('--input-manifest-sha256',default='7dbed0e914a0c176625ed5109ff5b2900ecdd5c0b8f4564d4d190f041d4caf89')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    spec=importlib.util.spec_from_file_location('gr7_authority_runner',root/'programs/run_rc_grouped_residual_development_v7.py')
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    def bind(rel):
        path=root/rel;runner.need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),'SOURCE_NOT_READY:'+rel)
        return {'path':rel,'sha256':runner.sha(path)}
    a={'schema':'rc_grouped_residual_development_authority_v7_20260908','status':'DRAFT_REQUIRES_ROOT_REVIEW_AND_FREEZE',
       'contract':bind(args.contract),'core':bind('src/rc_aslo_xf/reference_conditioned_grouped_residual_p_only_v7.py'),
       'adapter':bind('programs/prepare_rc_grouped_residual_inputs_v7.py'),
       'runner':bind('programs/run_rc_grouped_residual_development_v7.py'),
       'launcher':bind('slurm/rc_grouped_residual_development_v7_dev_cpuonly_59m.sbatch'),
       'gate':bind('src/rc_aslo_xf/reference_conditioned_latent_proposal_natural_gate_v2.py'),
       'e0':{**bind(args.e0),'expected_status':args.e0_status},
       'input_manifest_sha256':args.input_manifest_sha256,
       'postjoin':{'authority':bind('registry/rc_lth_p_only_natural_optimization_postjoin_authority_v1_20260907.json'),
                   'validation':bind('registry/rc_lth_p_only_natural_optimization_postjoin_authority_v1_20260907.independent_validation.json')},
       'old_v6':{'result':bind('results/rc_competitive_witness_development_v6/result.json'),
                 'statistics':bind('results/rc_competitive_witness_development_v6/statistics.json'),
                 'validation':bind('results/rc_competitive_witness_development_v6_validation_json_schema_repair_v1/result.json'),
                 'fresh_state':bind('results/rc_competitive_witness_development_work_v6/fresh_state.json')},
       'training':runner.TRAINING,'new_method_variant_count':1,
       'matched_arms':['REAL','ALL_PATCH_NO_HYP','QUERY_ONLY_REGION'],
       'gate_requirements':{'structural_coverage_at_least':26,'rank_success_at_least':21,'supergroup_margin_positive':True,
          'paired_net_vs_each_matched_control_at_least':4,'c_bind_positive_drops_at_least':21,'p_coord_positive_drops_at_least':21,
          'control_drop_supergroup_means_positive':True,'paired_net_vs_frozen_v6_query_only_at_least':4},
       'coordinate_control_limitation':'All PC proposals are structurally H0; PC drop alone is not identity spatial causality.',
       'work_rel':'results/rc_grouped_residual_development_work_v7','output_rel':'results/rc_grouped_residual_development_v7',
       'validation_rel':'results/rc_grouped_residual_development_v7_validation',
       'resource_bound':{'partition':'dev_cpuonly','cpus_per_worker':4,'memory_gib_per_worker':64,'minutes_per_allocation':59,
                         'gpu_count':0,'independent_training_lanes':2,'continuations_same_checkpoint_only':True},
       'automatic_stage_advance':False,'formal_outcomes_before_development_go_authorized':False,'scientific_GO_or_NO_GO':None}
    sources=[]
    for key in ('contract','core','adapter','runner','launcher','gate','e0'):sources.append({k:v for k,v in a[key].items() if k in ('path','sha256')})
    sources.extend(a['postjoin'].values());sources.extend(a['old_v6'].values())
    sources.append(bind('src/rc_aslo_xf/reference_conditioned_latent_proposal_natural_gate_v1.py'))
    sources.append(bind('programs/prepare_rc_grouped_residual_authority_v7.py'))
    a['sources']=sorted(sources,key=lambda x:x['path']);a['logical_sha256']=runner.logical(a)
    runner.need(not args.out.exists(),'APPEND_ONLY_DRAFT_EXISTS');runner.atomic_json(args.out,a)
    print(json.dumps({'status':a['status'],'path':str(args.out),'sha256':runner.sha(args.out),'sources':len(a['sources'])}))
if __name__=='__main__':main()
