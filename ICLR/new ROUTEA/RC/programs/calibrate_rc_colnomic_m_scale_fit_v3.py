#!/usr/bin/env python3
"""Freeze a global M calibration on the completed four-query TRAIN panel."""
import json
from pathlib import Path
import numpy as np
import rc_m_logit_calibration_v3 as K
import run_rc_colnomic_m_scale_fit_v2 as S

ROOT=S.ROOT
OUT=ROOT/'results/rc_colnomic_m_scale_calibration_v3'
AUTH=ROOT/'registry/rc_colnomic_m_scale_calibration_authority_v3_20260924.json'
PLAN=ROOT/'plan/RC_COLNOMIC_M_GLOBAL_CALIBRATION_V3_20260924.md'


def main():
    assert not AUTH.exists()
    previous=S.read(S.OUT/'validation.json');result=S.read(S.checked(previous['result']))
    parent=S.read(S.AUTH);panel=S.read(S.checked(parent['panel']))
    S.write(AUTH,dict(status='TRAIN_SHARED_M_SCALE_CALIBRATION_FROZEN',
        sources=[S.bind(p) for p in (Path(__file__),Path(K.__file__),Path(S.__file__),PLAN)],
        previous_result=previous['result'],previous_validation=S.bind(S.OUT/'validation.json'),
        panel=parent['panel'],thresholds_changed=False,per_query_parameters=False,identity_labels_used=False,
        selection='FULL then BALANCED; unchanged four fitting gates',held_reads=0))
    target=np.asarray([r['teacher_M'] for r in panel['records']])
    arms={}
    for arm in S.ARMS:
        pred=np.asarray([r['predicted_M'] for r in result['arms'][arm]['rows']])
        bias=K.fit(pred,target);changed=K.apply(pred,bias);rows=[]
        for i,rec in enumerate(panel['records']):
            # A strictly monotone global scalar preserves candidate ordering.
            assert np.array_equal(np.argsort(pred[i],kind='stable'),np.argsort(changed[i],kind='stable'))
            rows.append(dict(query_id=rec['query_id'],M_uncalibrated=pred[i].tolist(),predicted_M=changed[i].tolist(),
                             metrics=S.D.metrics(changed[i],target[i])))
        summary={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
        arms[arm]=dict(arm=arm,network_step=2000,calibration_bias=bias,rows=rows,
                      mean_query_metrics=summary,gate=S.gate(summary,rows))
    selected=next((arm for arm in S.ARMS if all(arms[arm]['gate'].values())),None)
    S.write(OUT/'result.json',dict(status='SMALL_TRAIN_FIT_GATE_PASS' if selected else 'SMALL_TRAIN_FIT_GATE_NOT_MET',
        authority=S.bind(AUTH),arms=arms,selected_recipe=selected,calibration='GLOBAL_LOGIT_M_BIAS',
        held_reads=0,identity_labels_used=False,network_updates=0,
        boundary='Four TRAIN queries only; extra scalar fitted to teacher M, not retrieval labels'))
    S.write(OUT/'validation.json',dict(status='GLOBAL_M_CALIBRATION_MONOTONIC_AND_MEAN_PASS',authority=S.bind(AUTH),
        result=S.bind(OUT/'result.json'),network_updates=0,held_reads=0))
    print(json.dumps(dict(status='COMPLETE',selected=selected,arms={k:dict(gate=v['gate'],bias=v['calibration_bias'],
        metrics=v['mean_query_metrics']) for k,v in arms.items()})),flush=True)


if __name__=='__main__':main()
