#!/usr/bin/env python3
"""Locate the source of M contrasts; no downstream heads or accuracy counts.

Read the completed, predeclared outcome-independent eight-query norm/phase
panel. The three selected rescue cases are deliberately not substituted for
it. Fixed wrong competitors are selected by unchanged ColNomic content, never M.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from analyze_rc_mass_information_transport_v1 import ROOT, read, write, grouped, SOURCES
from rc_roma_position_factor_v1 import ARMS

OUT = ROOT/'results/rc_mass_discrimination_source_v1'
PHASE = ROOT/'results/rc_roma_position_factor_v1'


def checked(b):
    p=Path(b['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == b['sha256'], str(p)
    return p


def contrasts(mass,target,wrong,content):
    lm=np.log(np.maximum(np.asarray(mass),1e-12))
    nearest=int(wrong[np.argmin(abs(content[wrong]-content[target]))])
    strongest=int(wrong[np.argmax(content[wrong])])
    band=wrong[abs(content[wrong]-content[target])<=.005]
    return dict(nearest_content_logM_gap=float(lm[target]-lm[nearest]),
        strongest_content_logM_gap=float(lm[target]-lm[strongest]),
        near_band_logM_gap=float(np.mean(lm[target]-lm[band])) if len(band) else None,
        mean_logM=float(lm.mean()),
        wrong_mean_logM=float(lm[wrong].mean()),target_logM=float(lm[target]),
        centered_logM_spread=float(lm.std()), nearest_position=nearest,
        strongest_position=strongest,near_band_count=len(band))


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'result.json').exists()
    manifest=read(PHASE/'manifest.json')
    panel=manifest['outcome_independent_panel']
    write(OUT/'protocol.json',dict(primary_panel=panel,selected_rescues_excluded=manifest['selected_rescue_audit'],
        analysis='COARSE M itself, not action accuracy; strongest unchanged-content wrong comparator primary; nearest and 0.005 band secondary.',
        main_effects=['AMP_FLAT minus NATIVE','AMP_PERMUTE minus NATIVE','mean LOCAL phase minus mean GLOBAL phase'],
        common_scale_control='Within-query logM target/wrong differences remove a shared multiplicative mass scale.',
        no_forward=True,no_retraining=True,no_accuracy=True))
    oldrows=read(ROOT/'results/rc_pair_relation_localization_v1/inside_paths/rows41.json')
    byindex={r['index']:r for r in oldrows}
    cache={r['query_id']:r for r in read(ROOT/'results/rc_h593_simple_explanations_v1/cache.json')['rows']}
    samples=[]; maxparity=0.; invariantmax=0.; paired=0
    for index in panel:
        old=byindex[index]; q=old['query_id']; row=cache[q]
        validation=read(PHASE/f'query{index:03d}'/'gpu_validation.json')
        assert validation['status']=='POSITION_FACTOR_ALL128_PASS' and validation['query_id']==q
        assert validation['candidate_physical_rows']==row['axis'] and len(validation['pairs'])==128
        original=read(checked(old['inside_payload']))
        masses={arm:[] for arm in ARMS}
        perpair=[]
        for pos,b in enumerate(validation['pairs']):
            p=torch.load(checked(b),map_location='cpu',weights_only=True,mmap=True)
            assert p['position']==pos and p['physical_row']==row['axis'][pos]
            for arm in ARMS:
                means=[float(side['weights'].double().mean()) for side in p['branches'][arm]]
                value=math.sqrt(means[0]*means[1]); masses[arm].append(value)
                for side in p['branches'][arm]:
                    invariantmax=max(invariantmax,side['invariants'].get('norm_relative_error',0.))
                if arm in ('NATIVE','ZERO'):
                    branch='A1J1P1' if arm=='NATIVE' else 'A1J1P0'
                    error=abs(value-original['records'][pos]['branches'][branch]['COARSE'])
                    maxparity=max(maxparity,error); assert error<2e-10
            paired+=1
        save=dict(index=index,query_id=q,component=old['component'],target_present=old['target_in_C128'],
                  axis=row['axis'],mass_by_arm=masses,source_validation=str(PHASE/f'query{index:03d}'/'gpu_validation.json'))
        if old['target_in_C128']:
            target=row['axis'].index(old['target_physical_row'])
            wrong=np.array([i for i in range(128) if i!=target]); content=np.array(row['free_content'])
            save['target_position']=target
            save['arm_metrics']={arm:contrasts(masses[arm],target,wrong,content) for arm in ARMS}
            metrics=['strongest_content_logM_gap','nearest_content_logM_gap','near_band_logM_gap',
                     'target_logM','wrong_mean_logM','centered_logM_spread']
            save['contrasts']={}
            for metric in metrics:
                x={arm:save['arm_metrics'][arm][metric] for arm in ARMS}
                if x['NATIVE'] is None: continue
                loc=np.mean([x[k] for k in ARMS if k.startswith('LOCAL')]); glob=np.mean([x[k] for k in ARMS if k.startswith('GLOBAL')])
                save['contrasts'][metric]=dict(AMP_FLAT_minus_NATIVE=x['AMP_FLAT']-x['NATIVE'],
                    AMP_PERMUTE_minus_NATIVE=x['AMP_PERMUTE']-x['NATIVE'],
                    GLOBAL_minus_NATIVE=float(glob-x['NATIVE']),LOCAL_minus_NATIVE=float(loc-x['NATIVE']),
                    LOCAL_minus_GLOBAL=float(loc-glob),ZERO_minus_NATIVE=x['ZERO']-x['NATIVE'])
        samples.append(save)
        print(json.dumps(dict(stage='query_M_source_complete',index=index,pairs=128)),flush=True)
    assert invariantmax<2e-6
    present=[r for r in samples if r['target_present']]
    summary={}
    for metric in sorted({k for r in present for k in r['contrasts']}):
        summary[metric]={}
        contrast_keys=sorted({k for r in present for k in r['contrasts'].get(metric,{})})
        for contrast in contrast_keys:
            rr=[dict(component=r['component'],v=r['contrasts'].get(metric,{}).get(contrast)) for r in present]
            summary[metric][contrast]=grouped(rr,'v')
    # Existing factorial intervention evidence on precisely the same fixed
    # content opponents; no head or M-dependent opponent re-selection.
    factorial=[]
    for old in oldrows:
        if not old['target_in_C128']: continue
        r=cache[old['query_id']]; oldm=read(checked(old['inside_payload']))
        t=r['axis'].index(old['target_physical_row']); wrong=np.array([i for i in range(128) if i!=t]); content=np.array(r['free_content'])
        vals={arm:contrasts([p['branches'][arm]['COARSE'] for p in oldm['records']],t,wrong,content)
              for arm in ['A1J1P1','A1J1P0','A1J0P1','A1J0P0','A0J1P1']}
        key='strongest_content_logM_gap'; v={a:x[key] for a,x in vals.items()}
        factorial.append(dict(query_id=r['query_id'],component=old['component'],values=vals,
            J_given_P=v['A1J1P1']-v['A1J0P1'],P_given_J=v['A1J1P1']-v['A1J1P0'],
            P_without_J=v['A1J0P1']-v['A1J0P0'],J_without_P=v['A1J1P0']-v['A1J0P0'],
            JP=v['A1J1P1']-v['A1J1P0']-v['A1J0P1']+v['A1J0P0'],
            A_given_JP=v['A1J1P1']-v['A0J1P1']))
    for p in [Path(__file__),ROOT/'programs/rc_roma_position_factor_v1.py',
              ROOT/'programs/analyze_rc_mass_information_transport_v1.py']:
        SOURCES.append(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    output=dict(status='M_DISCRIMINATION_SOURCE_PANEL8_COMPLETE',endpoint='COARSE',
        queries=len(samples),target_present=len(present),pairs=paired,source_head_training=False,
        max_native_zero_M_parity_error=maxparity,max_phase_norm_relative_error=invariantmax,
        phase_contrasts=summary,phase_rows=samples,
        common41_factorial={k:grouped(factorial,k) for k in ['J_given_P','P_given_J','P_without_J','J_without_P','JP','A_given_JP']},
        factorial_rows=factorial,sources=SOURCES,
        limitation='Small predeclared eight-group panel. Phase shifts hold norm and frequency power but can violate learned coordinate conventions and content-position alignment. All interventions condition on native J/A; they do not identify independent sufficient inputs or a universal matching law.')
    write(OUT/'result.json',output)
    print(json.dumps(dict(status=output['status'],queries=len(samples),target_present=len(present),
        phase=summary['strongest_content_logM_gap'],factorial=output['common41_factorial']),indent=2))


if __name__=='__main__': main()
