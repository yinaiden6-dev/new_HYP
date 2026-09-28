#!/usr/bin/env python3
"""Read-only scientific interpretation of sealed new-M response predictions.

No model forwards or fits. Primary metrics come from the frozen experiment;
additional geometry/rare-switch summaries are explicitly descriptive.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

RC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RC/'programs'))
import run_rc_post_m_predictive_response_v1 as P


def main():
    root=P.DEFAULT;p,pb,workers=P.guard(root)
    validation=P.read(root/'independent_validation.json')
    assert validation['status']=='POST_M_PREDICTIVE_SECOND_NUMPY_PASS'
    r=P.read(P.checked(validation['result']));seal=P.read(P.checked(r['prediction_seal']))
    assert r['protocol']==seal['protocol']==pb
    endpoints=[];geometry=[];switch=[];fixedpairs=[];chronology=[]
    rolemap={z['index']:z for z in P.read(P.checked(p['evaluation_roles']))['rows']}
    for w,sr in zip(workers,seal['rows']):
        pred=P.read(P.checked(sr['prediction']));folder=root/'queries'/f"{w['index']:03d}"
        er=P.read(folder/'endpoint.json');bypos={}
        for c,pr,eb in zip(w['candidates'],pred['candidates'],er['candidates']):
            actualrec=P.read(P.checked(eb));probe=P.arrays(pr['probe']);forecast=P.arrays(pr['predictions']);actual=P.arrays(actualrec['patch'])
            assert actualrec['prediction_seal']==r['prediction_seal']
            chronology.append(Path(actualrec['patch']['path']).stat().st_mtime_ns-Path(r['prediction_seal']['path']).stat().st_mtime_ns)
            position=c['position'];local=pr['local'];npatch=len(probe['argmax'][2]);bypos[position]=(probe,forecast,actual,local)
            radiusprobe=(probe['argmax'][0]==probe['argmax'][2])&(probe['argmax'][4]==probe['argmax'][2])
            geometry.append(dict(index=w['index'],query_id=w['query_id'],component=w['component'],position=position,
                scalar_slope=local['slope'],fixed_winner_alignment_slope=local['fixed_winner_alignment_mean_slope'],
                scalar_minus_fixed_abs=abs(local['scalar_minus_fixed_winner_slope']),
                half_vs_full_slope_abs=abs(local['half_slope']-local['slope']),
                token_derivative_norm=local['normalized_token_derivative_mean_norm'],
                tangent_error=local['normalized_token_derivative_mean_tangent_error'],
                baseline_winner_retained_at_both_outer_probes=float(radiusprobe.mean()),
                baseline_top2_gap_mean=float(probe['baseline_top2_gap'].mean())))
            for j,t in enumerate(P.ENDPOINTS):
                truechange=float(actual['L'][j]-probe['L'][2]);changed=actual['argmax'][j]!=probe['argmax'][2]
                for k,model in enumerate(P.PATCH_MODELS):
                    predicted=forecast['patch_argmax'][k,j];predchange=predicted!=probe['argmax'][2]
                    truepositive=int((predchange&changed).sum());falsepositive=int((predchange&~changed).sum());falsenegative=int((~predchange&changed).sum())
                    winner_correct=int((predicted==actual['argmax'][j]).sum());bound=forecast['predicted_bound_stable'][k,j]
                    switch.append(dict(index=w['index'],query_id=w['query_id'],component=w['component'],position=position,model=model,t=float(t),
                        patches=npatch,actual_switched=int(changed.sum()),predicted_switched=int(predchange.sum()),
                        switch_true_positive=truepositive,switch_false_positive=falsepositive,switch_false_negative=falsenegative,
                        switched_identity_correct=int(((predicted==actual['argmax'][j])&changed).sum()),
                        winner_accuracy=float(winner_correct/npatch),baseline_no_switch_winner_accuracy=float((~changed).mean()),
                        winner_accuracy_minus_no_switch=float(winner_correct/npatch-(~changed).mean()),
                        bound_stable_patches=int(bound.sum()),bound_stable_actual_switched=int((bound&changed).sum()),
                        bound_stable_fraction=float(bound.mean()),bound_violation_fraction=float((bound&changed).mean())))
                endpoints.append(dict(index=w['index'],query_id=w['query_id'],component=w['component'],position=position,t=float(t),
                    absolute_actual_change=abs(truechange),actual_zero=float(P.sign(truechange)==0),
                    fixed_baseline_winner_change=float(actual['fixed_maxsim'][j].mean()-probe['L'][2]),
                    actual_rematching_extra=float(actual['L'][j]-actual['fixed_maxsim'][j].mean())))
        role=rolemap[w['index']];tp,wp=role['target_position'],role['fixed_wrong_position']
        pt,pw=bypos[tp],bypos[wp]
        for j,t in enumerate(P.ENDPOINTS):
            truegap=float((pt[2]['L'][j]-pt[0]['L'][2])-(pw[2]['L'][j]-pw[0]['L'][2]))
            fixedguess=float(t*(pt[3]['fixed_winner_alignment_mean_slope']-pw[3]['fixed_winner_alignment_mean_slope']))
            localguess=float(t*(pt[3]['slope']-pw[3]['slope']))
            fixedpairs.append(dict(index=w['index'],query_id=w['query_id'],component=w['component'],t=float(t),
                actual_gap_change=truegap,fixed_reference_alignment_prediction=fixedguess,local_score_prediction=localguess,
                absolute_actual_gap_change=abs(truegap),fixed_alignment_absolute_error=abs(fixedguess-truegap),
                local_score_absolute_error=abs(localguess-truegap),fixed_alignment_MAE_minus_shared_zero=abs(fixedguess-truegap)-abs(truegap),
                alignment_vs_local_prediction_abs=abs(fixedguess-localguess),
                fixed_alignment_direction_agreement=float(P.sign(fixedguess)==P.sign(truegap)),actual_zero=float(P.sign(truegap)==0)))
    assert min(chronology)>0,'ENDPOINT_MTIME_NOT_AFTER_SEAL'
    def summarize(rows,by,keys):return P.summarize(rows,by,keys)
    out=dict(status='POST_M_PREDICTIVE_INTERPRETATION_COMPLETE',protocol=pb,result=validation['result'],
        source_script=P.bind(Path(__file__)),independent_validation=P.bind(root/'independent_validation.json'),
        primary_results=r['summary'],scope=r['boundary'],
        chronology=dict(all_endpoint_arrays_bound_to_global_seal=True,endpoint_array_count=len(chronology),
            earliest_endpoint_array_minus_seal_seconds=min(chronology)/1e9,
            timestamp_caveat='Filesystem mtime is corroboration, not independent trusted time; execution barriers and bound program/prediction hashes supply protocol closure.'),
        descriptive_geometry={key:P.grouped(geometry,key) for key in ('scalar_slope','fixed_winner_alignment_slope','scalar_minus_fixed_abs',
            'half_vs_full_slope_abs','token_derivative_norm','tangent_error','baseline_winner_retained_at_both_outer_probes','baseline_top2_gap_mean')},
        endpoint_scale=summarize(endpoints,('t',),('absolute_actual_change','actual_zero','fixed_baseline_winner_change','actual_rematching_extra')),
        geometry_common_shift=summarize(fixedpairs,('t',),('absolute_actual_gap_change','fixed_alignment_absolute_error','local_score_absolute_error',
            'fixed_alignment_MAE_minus_shared_zero','alignment_vs_local_prediction_abs','fixed_alignment_direction_agreement','actual_zero')),
        patch_switch_summary=summarize(switch,('model','t'),('winner_accuracy','baseline_no_switch_winner_accuracy','winner_accuracy_minus_no_switch',
            'bound_stable_fraction','bound_violation_fraction')),
        geometry_rows=geometry,endpoint_rows=endpoints,common_shift_rows=fixedpairs,patch_switch_rows=switch,
        caveats=['Probe slope equals derivative/reference alignment only within finite-difference and assignment constraints; that algebra itself is not prospective evidence.',
            'Useful evidence is its performance on new M endpoints sealed after predictions, compared to shared and constant predictors.',
            'Forecast no-switch flags are not actual-model guarantees; miss/switch counts must accompany overall winner accuracy.',
            'This predicts the frozen response given M. It does not explain why upstream M contains identity evidence or remove dependence on RoMa.',
            'Coordinates, opened 120/45 population, two selected candidates and local 4/8-percent log shifts limit the conclusion.',
            'Additional geometry summaries are descriptive; no model is selected by these endpoint results.'])
    P.write(root/'scientific_interpretation.json',out,immutable=True)
    lines=['# Prospective POST response: scientific interpretation','',r['boundary'],'',
        'The prior TRAIN16/PROBE8 scalar transport experiment already established predictive response structure. This experiment tests new local conditions and matching-location forecasts, with a global prediction barrier.','',
        'For a fixed matching index, content-score response is the normalized query-token movement projected onto the selected reference token. With free MaxSim, switching reference indices adds another contribution. Scalar M alone does not set the response sign: token motion and reference content jointly do so. The following endpoint tests determine how well local measurements predict that response.','',
        '## Candidate prediction errors','',
        '| model | shift | MAE | MAE minus shared slope [95% group CI] |','|---|---:|---:|---|']
    for key,v in r['summary']['candidate'].items():
        model,t=key.split('/');d=v['MAE_minus_shared'];lines.append(f"| {model} | {t} | {v['absolute_error']['mean']:.8g} | {d['mean']:.8g} [{d['ci95'][0]:.8g}, {d['ci95'][1]:.8g}] |")
    lines+=['','## Candidate-specific response under a common M change','',
        'A shared linear slope predicts zero paired-gap change for the same shift applied to both candidates. Candidate-local reference-alignment slopes predict a nonzero gap when their responses differ. This tests a measurable content-dependent condition rather than relabeling M as quality.','',
        '| shift | actual absolute gap change | alignment forecast MAE | MAE minus shared-zero [95% group CI] |','|---:|---:|---:|---|']
    for key,v in out['geometry_common_shift'].items():
        d=v['fixed_alignment_MAE_minus_shared_zero'];lines.append(f"| {key} | {v['absolute_actual_gap_change']['mean']:.8g} | {v['fixed_alignment_absolute_error']['mean']:.8g} | {d['mean']:.8g} [{d['ci95'][0]:.8g}, {d['ci95'][1]:.8g}] |")
    lines+=['','## Matching forecast, with no-switch baseline','',
        '| model / shift | winner accuracy | no-switch accuracy | difference [95% group CI] | forecast bound-stable fraction |','|---|---:|---:|---|---:|']
    for key,v in out['patch_switch_summary'].items():
        d=v['winner_accuracy_minus_no_switch'];lines.append(f"| {key} | {v['winner_accuracy']['mean']:.6f} | {v['baseline_no_switch_winner_accuracy']['mean']:.6f} | {d['mean']:.6f} [{d['ci95'][0]:.6f}, {d['ci95'][1]:.6f}] | {v['bound_stable_fraction']['mean']:.6f} |")
    lines+=['','## Evidence boundary','']+[f'- {s}' for s in out['caveats']]
    (root/'scientific_interpretation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=out['status'],output=str(root/'scientific_interpretation.json'),minimum_endpoint_after_seal_seconds=min(chronology)/1e9)),flush=True)

if __name__=='__main__':main()
