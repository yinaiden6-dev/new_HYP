#!/usr/bin/env python3
"""All-endpoint and matched-domain coordinate comparison for the fixed M audit.

No model execution or model selection. Published after complete main validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
RC=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RC/'programs'),str(RC/'src')]
import run_rc_m_causal128_frozen_bridge_v1 as B
from run_rc_post_m_level_gap_v1 import COORDINATES, EFFECTS
DEFAULT=RC/'results/rc_post_m_level_gap_v1'
METRICS=('logM_gap','POST_content_gap','POST_action_gap','NATIVE7_action_gap','M_FREE_action_gap')


def calculate(root):
    val=B.read(root/'validation.json');assert val['status']=='POST_M_LEVEL_GAP_INDEPENDENT_PASS'
    result=B.read(B.checked(val['result']));rows=result['rows'];indexed={};matched={};totals={};max_error=0.
    for row in rows:
        for comp in row['comparisons']:
            key=(row['index'],comp['panel'],comp['contrast'],comp['replicate']);indexed.setdefault(key,{})[comp['coordinate']]=(row,comp)
    for key,coords in indexed.items():
        assert set(coords)==set(COORDINATES)
        row=coords['log_M'][0];scope='/'.join(key[1:3])
        cs=[coords[c][1] for c in COORDINATES]
        # Coordinate changes must never change the two factual endpoints.
        for cell in ('LL','GG'):
            for metric in METRICS:max_error=max(max_error,abs(cs[0]['cells'][cell]['gaps'][metric]-cs[1]['cells'][cell]['gaps'][metric]))
        for metric in METRICS:
            total=cs[0]['cells']['GG']['gaps'][metric]-cs[0]['cells']['LL']['gaps'][metric]
            totals.setdefault(scope,{}).setdefault(metric,[]).append(dict(component=row['component'],index=row['index'],total=total))
        if not all(c['complete_legal_grid'] for c in cs):continue
        for coordinate,comp in zip(COORDINATES,cs):
            for metric in METRICS:
                matched.setdefault(scope,{}).setdefault(coordinate,{}).setdefault(metric,[]).append(dict(component=row['component'],index=row['index'],**comp['effects'][metric]))
    assert max_error<2e-10
    totals_summary={scope:{metric:B.group_summary(entries,'total') for metric,entries in metrics.items()} for scope,metrics in totals.items()}
    matched_summary={scope:{coord:{metric:{e:B.group_summary(entries,e) for e in EFFECTS} for metric,entries in metrics.items()} for coord,metrics in coords.items()} for scope,coords in matched.items()}
    coverage={scope:dict(all_endpoint_queries=sorted({r['index'] for r in metrics['POST_content_gap']}),
        common_legal_queries=sorted({r['index'] for r in matched[scope]['log_M']['POST_content_gap']})) for scope,metrics in totals.items()}
    signs={}
    for scope,coordinates in matched_summary.items():
        signs[scope]={metric:{effect:dict(log_M_mean=coordinates['log_M'][metric][effect]['mean'],log_odds_M_mean=coordinates['log_odds_M'][metric][effect]['mean'],
            same_mean_sign=bool(np.sign(coordinates['log_M'][metric][effect]['mean'])==np.sign(coordinates['log_odds_M'][metric][effect]['mean'])))
            for effect in EFFECTS} for metric in METRICS}
    out=dict(status='POST_M_LEVEL_GAP_MATCHED_COORDINATE_ANALYSIS_PASS',main_validation=B.bind(root/'validation.json'),
        source_program=B.bind(Path(__file__)),all_endpoint_total_summary=totals_summary,matched_domain_coordinate_summary=matched_summary,
        matched_mean_sign_comparison=signs,coverage=coverage,endpoint_coordinate_parity_max_error=max_error,
        labels='Post-hoc descriptive conditional finite differences; not independent confirmation or unique causal attribution.',
        sample_policy='All endpoint totals include all120 J/P pairs. Coordinate contribution comparisons use exactly the common legal sample set (99 for J_with_P; all120 other J/P edges; all9 phase groups). The main report separately preserves full120 log-odds results.')
    B.write(root/'matched_coordinate_analysis.json',out,immutable=True)
    lines=['# Common level / relative gap: all endpoints and matched-domain coordinate comparison','',out['labels'],'',out['sample_policy'],'',
        '| Panel / contrast | Endpoint | All-endpoint total (GG−LL) | Queries | Groups |','|---|---|---|---|---|']
    def fmt(v):return f"{v['mean']:+.9f}; CI={v['exploratory_bootstrap95']}; {v['positive_groups']}/{v['groups']} positive"
    for scope,metrics in totals_summary.items():
        for metric,d in metrics.items():lines.append(f"| {scope} | {metric} | {fmt(d)} | {len(coverage[scope]['all_endpoint_queries'])} | {d['groups']} |")
    lines+=['','The next matrix uses the same samples across coordinates. L/G indicate endpoint origin, not lower/higher M or better/worse results.','',
        '| Scope | Coordinate | Endpoint | δ at μL | δ at μG | μ at δL | μ at δG | Interaction | Symmetric μ | Symmetric δ |','|---|---|---|---|---|---|---|---|---|---|']
    for scope,coords in matched_summary.items():
        for coordinate,metrics in coords.items():
            for metric in ('POST_content_gap','POST_action_gap','NATIVE7_action_gap','M_FREE_action_gap'):
                d=metrics[metric];order=('delta_at_mu_L','delta_at_mu_G','mu_at_delta_L','mu_at_delta_G','interaction','symmetric_mu','symmetric_delta')
                lines.append('| '+scope+' | '+coordinate+' | '+metric+' | '+' | '.join(fmt(d[e]) for e in order)+' |')
    lines+=['','Refer also to condition_range_audit.md/json: mass-domain legality is not membership in the original TRAIN distribution. The analysis never removes valid observations based on these diagnostics.',
        'The phase9 and broader J/P panels overlap and are not pooled. Four axis/sign repetitions are grouped before bootstrap. Every interval is exploratory.','']
    path=root/'matched_coordinate_analysis.md';text='\n'.join(lines)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print(json.dumps(dict(status=out['status'],endpoint_coordinate_parity_max_error=max_error,panels=len(matched_summary))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT);args=ap.parse_args();calculate(args.root)
if __name__=='__main__':main()
