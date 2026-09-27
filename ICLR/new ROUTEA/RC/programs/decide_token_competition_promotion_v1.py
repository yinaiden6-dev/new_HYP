#!/usr/bin/env python3
"""Apply the pre-result promotion policy, without tuning it on held outcomes."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

RC = Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text())
def bind(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def checked(binding):
    if bind(binding['path'])!=binding: raise ValueError('source binding changed: '+binding['path'])
    return Path(binding['path'])


def qualifies(row, policy):
    values=[float(row[k]) for k in ('net','identity_cluster_gain_ci_low','identity_cluster_gain_ci_high')]
    if not all(math.isfinite(v) for v in values): raise ValueError('nonfinite promotion evidence')
    if values[1]>values[2]: raise ValueError('invalid interval')
    c=policy['criteria']
    return values[0]>c['net_greater_than'] and (
        c.get('identity_cluster_gain_ci_low_greater_than') is None or
        values[1]>c['identity_cluster_gain_ci_low_greater_than'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=RC/'results/rc_token_competition_f128_v2')
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        p={'criteria':{'net_greater_than':0,'identity_cluster_gain_ci_low_greater_than':0}}
        r={'net':'1','identity_cluster_gain_ci_low':'0.001','identity_cluster_gain_ci_high':'0.02'}
        assert qualifies(r,p)
        assert not qualifies({**r,'net':'0'},p)
        assert not qualifies({**r,'identity_cluster_gain_ci_low':'0'},p)
        assert not qualifies({**r,'identity_cluster_gain_ci_low':'-0.01'},p)
        print('PROMOTION_BOUNDARY_CHECK_PASS');return
    root=args.root.resolve(); policy_path=root/'promotion_policy.json';policy=read(policy_path)
    validation_path=root/'validation.json'
    if not validation_path.exists():raise SystemExit('WAITING_FOR_INDEPENDENT_FINAL_VALIDATION')
    val=read(validation_path)
    if val.get('status')!=policy['scientific_validation_required'] or val.get('fits_complete')!=15:
        raise ValueError('all 15 independently validated fits are required')
    if val['protocol']!=policy['original_protocol']:raise ValueError('promotion population/protocol changed')
    checked(val['protocol'])
    required_validation = {'metrics.csv', 'paired_vs_B_CAL.csv', 'paired_structural.csv',
        'joined_predictions.json', 'candidate_predictions.json', 'trace_manifest.json', 'REPORT_TOKEN_COMPETITION_V2.md'}
    if not required_validation <= set(val.get('artifacts', {})):
        raise ValueError('complete independently bound result artifacts are required')
    for name, b in val['artifacts'].items():
        if Path(b['path']).resolve() != (root/name).resolve():
            raise ValueError('unexpected independently validated artifact path: '+name)
        checked(b)
    analysis_path=root/'analysis/analysis_summary.json';analysis=read(analysis_path)
    if analysis.get('status')!='TOKEN_COMPETITION_ANALYSIS_PASS':raise ValueError('completed analysis required')
    if analysis.get('validation')!=bind(validation_path):raise ValueError('analysis/final result mismatch')
    if analysis.get('protocol')!=val['protocol']:raise ValueError('analysis protocol mismatch')
    expected_analysis = {
        'report': root/'analysis/REPORT_TOKEN_COMPETITION_F128_V2_ANALYSIS.md',
        'per_fold': root/'analysis/per_fold.csv',
        'per_query': root/'analysis/per_query_diagnostics.csv'}
    if set(analysis.get('artifacts', {})) != set(expected_analysis):
        raise ValueError('complete analysis artifacts are required')
    for name, path in expected_analysis.items():
        if analysis['artifacts'][name] != bind(path):
            raise ValueError('analysis artifact binding mismatch: '+name)
    with (root/'paired_vs_B_CAL.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['operating_point']==policy['operating_point']]
    arms=policy['candidate_arms']
    if len(rows)!=3 or {r['model'] for r in rows}!=set(arms):raise ValueError('primary comparison coverage changed')
    for row in rows:
        if row['baseline']!=policy['baseline'] or int(row['seed'])!=0 or int(row['queries'])!=128:
            raise ValueError('not the registered baseline, seed, or population')
    decisions=[{'arm':r['model'],'passed':qualifies(r,policy),'comparison':r} for r in rows]
    winners=[x['arm'] for x in decisions if x['passed']]
    result={'status':'TOKEN_COMPETITION_F128_SEED_CONFIRMATION_GATE_'+('PASS' if winners else 'NO_GO'),
        'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'origin_protocol':val['protocol'],
        'origin_validation':bind(validation_path),'analysis_gate':bind(analysis_path),
        'gate_specification':bind(policy_path),'qualifying_arms':winners,'decisions':decisions,
        'next_round':policy['next_round'],'publish_current_results':True,
        'interpretation':'Development replication trigger only; no external confirmation or unique-cause claim.',
        'decision_source':bind(__file__)}
    path=root/'analysis/promotion_decision.json'
    if path.exists():
        old=read(path)
        for key in ('status','origin_protocol','origin_validation','analysis_gate','gate_specification','qualifying_arms','decisions'):
            if old.get(key)!=result[key]:raise ValueError('completed promotion decision changed: '+key)
        result=old
    else:path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'qualifying_arms':winners,'decision':str(path)}))


if __name__=='__main__':main()
