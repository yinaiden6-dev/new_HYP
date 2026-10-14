"""Guarded frozen replay of the accepted core plus fresh-reference controls.

The existing ten-module core is unchanged. Quick mode limits only its query
axis; the supplemental intervention branch always checks all 290 matrices.
This entry neither reencodes images nor infers edited-image identity labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

from scripts.colnomic_mechanism import reproduce_full_audit as core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only',action='store_true')
    mode.add_argument('--reproduce',action='store_true')
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--quick',action='store_true',help='Core8 queries/all5405 references/all8 heads; supplemental controls fully replayed')
    parser.add_argument('--plan-only',action='store_true',help='Write a plan without numerical acceptance')
    args=parser.parse_args()
    workspace=args.workspace.resolve();common=core.initialize(workspace)
    torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ['',None]:
        raise ValueError('Use CUDA_VISIBLE_DEVICES empty: replay is CPU-only')
    base=workspace/core.RELATIVE_BASE
    output=core.fresh_output(args.output or base/'reproduction_support/final_runs'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid())),workspace)
    scratch=output/'runtime_temp';scratch.mkdir()
    tempfile.tempdir=str(scratch);os.environ['TMPDIR']=str(scratch)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(scratch/'torchinductor')
    os.environ['COLNOMIC_WORKSPACE']=str(workspace)
    core.install_guard(workspace,output)
    plan=dict(mode='reproduce' if args.reproduce else 'verify-only',quick=args.quick,GPU=False,encoder=False,
        core='Unchanged accepted reproduce_full_audit.py: complete baseline/relation axes and ten modules; quick excludes long modules',
        supplement='All twelve Query identities/sixty arms, twenty-eight freshly encoded references, forty encoding records; all145 fresh+145 frozen matrices and all traces',
        output=str(output),workspace=str(workspace),
        limits=['Quick results do not renew all ten module verifications',
                'Fresh-reference intervention is preselected candidate response, not full-gallery accuracy',
                'Historical gallery encoder provenance remains unresolved; same ordinal Token axis is not physical-grid equivalence'])
    common.write(output/'plan.json',plan)
    if args.plan_only:
        common.write(output/'validation.json',dict(status='planned_not_numerically_verified',plan=plan))
        print('Plan saved without numerical acceptance',output,flush=True);return
    frozen=common.read(base/'reproduction_support/final_full_verification.json')
    assert frozen['status']=='passed' and frozen['scope']=='full_frozen_numerical_verification'
    assert common.sha(Path(core.__file__))==frozen['entry_sha256'],'Accepted core code changed; independently renew full validation'
    assert common.sha(Path(common.__file__))==frozen['workspace_path_implementation_sha256']
    for record in frozen['module_verification']:
        assert record['status']=='passed'
        filename=next(filename for name,filename,_ in core.MODULES if name==record['module'])
        assert common.sha(workspace/'scripts/colnomic_mechanism'/(filename+'.py'))==record['verifier_sha256']
    command=[sys.executable,'-B','-m','scripts.colnomic_mechanism.reproduce_full_audit',
        '--workspace',str(workspace),'--output',str(output/'core'),'--reproduce' if args.reproduce else '--verify-only']
    if args.quick:command.append('--quick')
    started=time.monotonic()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
    with (output/'core.log').open('w') as log:
        subprocess.run(command,cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    core_result=common.read(output/'core/validation.json')
    assert core_result['status']=='passed'
    print('Accepted core replay passed',core_result['scope'],flush=True)
    analyzer=importlib.import_module('scripts.colnomic_mechanism.fresh_reference_interventions')
    verifier=importlib.import_module('scripts.colnomic_mechanism.verify_fresh_reference_interventions')
    for module in [analyzer,verifier]:
        module.ROOT=workspace;module.BASE=base;module.OLD=base/'detail_interventions';module.FOLDER=base/'fresh_reference_interventions'
    analysis=base/'fresh_reference_interventions/analysis'
    if args.reproduce:
        analysis=output/'fresh_reference_reproduced'
        analyzer.analyze(analysis)
    supplemental=verifier.verify(analysis,output/'fresh_reference_independent')
    assert supplemental['status']=='passed' and supplemental['full_pointwise_matrices']==290
    feature_bridge=importlib.import_module('scripts.colnomic_mechanism.verify_fresh_feature_bridge')
    feature_bridge.ROOT=workspace;feature_bridge.BASE=base
    feature_bridge.OLD=base/'detail_interventions';feature_bridge.FOLDER=base/'fresh_reference_interventions'
    feature_result=feature_bridge.verify(output/'fresh_reference_feature_bridge')
    assert feature_result['status']=='passed' and feature_result['replayed_feature_token_rows']==63575
    descriptive=importlib.import_module('scripts.colnomic_mechanism.summarize_fresh_reference_interventions')
    descriptive.ROOT=workspace;descriptive.FOLDER=base/'fresh_reference_interventions'
    descriptive.summarize(output/'fresh_reference_summary_independent',descriptive.FOLDER/'interpretation')
    summary_result=common.read(output/'fresh_reference_summary_independent/validation.json')
    assert summary_result['status']=='passed'
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',scope='core_quick_plus_complete_fresh_reference_verification' if args.quick else 'full_core_plus_fresh_reference_numerical_verification',
        plan=plan,core_scope=core_result['scope'],core_validation_SHA256=common.sha(output/'core/validation.json'),
        previously_full_core_validation_SHA256=common.sha(base/'reproduction_support/final_full_verification.json'),
        fresh_reference=supplemental,fresh_reference_feature_bridge=feature_result,
        fresh_reference_descriptive_summary=summary_result,
        dispatcher_SHA256=common.sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,
        overall_goal_acceptance=False,causal_complete=False)
    common.write(output/'validation.json',result)
    print('Final frozen dispatcher passed',result['scope'],output,flush=True)


if __name__=='__main__':main()
