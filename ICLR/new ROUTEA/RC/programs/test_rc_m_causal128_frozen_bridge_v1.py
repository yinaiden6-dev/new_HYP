#!/usr/bin/env python3
"""Independent algebra and lineage checks; no model forward or new data selection."""
import json
from pathlib import Path
import tempfile
import numpy as np
import run_rc_m_causal128_frozen_bridge_v1 as M


def run():
    prior=M.read(M.PHASE/'inputs.json')['rows']
    maximum=0.;count=0;hold_cases=0
    head=[.17,.73,-.09]
    for item in prior:
        row=item['post_row'];winner=row['winner_index'];ext=item['external']
        # Use all already-sealed masses, including normalized controls.
        for masses in item['masses'].values():
            masses=np.asarray(masses,np.float64)
            content=np.asarray(ext['free_content'])+np.linspace(-.03,.02,128)
            external=M.external(ext,masses);decision=M.decision(row,content,head)
            post=np.zeros(128);post[row['challenger_positions']]=decision['logits']
            comparisons=[(winner,(winner+1)%128),((winner+1)%128,winner),((winner+2)%128,(winner+3)%128)]
            for target,wrong in comparisons:
                actual=M.independent_gap(item,masses,content,head,target,wrong)
                expected=dict(M_gap=float(masses[target]-masses[wrong]),
                    logM_gap=float(np.log(masses[target])-np.log(masses[wrong])),
                    POST_content_gap=float(content[target]-content[wrong]),
                    POST_action_gap=float(post[target]-post[wrong]),
                    NATIVE7_action_gap=external['NATIVE7']['logits128'][target]-external['NATIVE7']['logits128'][wrong],
                    M_FREE_action_gap=external['M_FREE']['logits128'][target]-external['M_FREE']['logits128'][wrong])
                error=max(abs(actual[k]-expected[k]) for k in expected)
                assert error<M.LIMIT,(item['index'],target,wrong,error)
                maximum=max(maximum,error);count+=1;hold_cases+=target==winner or wrong==winner
    effects=M.property_contrasts({a:{'metric':x} for a,x in zip(M.ARMS,[11.,5.,3.,1.])},'metric')
    assert effects==dict(P_with_J=6.,P_without_J=2.,J_with_P=8.,J_without_P=4.,J_P_interaction=4.)
    groups=M.group_summary([dict(component='a',v=1),dict(component='a',v=3),dict(component='b',v=-2)],'v')
    assert groups['mean']==0. and groups['groups']==2 and groups['positive_groups']==groups['negative_groups']==1
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'value.json';M.write(path,{'x':1},immutable=True);M.write(path,{'x':1},immutable=True)
        try:M.write(path,{'x':2},immutable=True)
        except AssertionError:pass
        else:raise AssertionError('IMMUTABLE_DRIFT_ACCEPTED')
        b=M.bind(path);path.write_text('tamper')
        try:M.checked(b)
        except AssertionError:pass
        else:raise AssertionError('SOURCE_TAMPER_ACCEPTED')
    return dict(status='CAUSAL128_FROZEN_BRIDGE_ALGEBRA_AND_RESUME_TEST_PASS',
        prior_phase_queries=len(prior),independent_pair_world_checks=count,RAW_hold_zero_cases=hold_cases,
        independent_formula_max_error=maximum,property_factor_algebra=True,group_equal_aggregation=True,
        immutable_drift_rejected=True,source_tamper_rejected=True,
        training=False,upstream_forwards=0,model_forwards=0)

if __name__=='__main__':print(json.dumps(run(),indent=2))
