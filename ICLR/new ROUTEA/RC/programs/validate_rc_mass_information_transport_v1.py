#!/usr/bin/env python3
"""Independent scalar/count check and critical-pair limits of the diagnostic."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_mass_information_transport_v1'


def read(p):
    return json.loads(Path(p).read_text())


def main():
    result = read(OUT/'result.json')
    for source in result['sources']:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    cache = read(ROOT/'results/rc_h593_simple_explanations_v1/cache.json')['rows']
    meta = {r['query_id']:r for r in read(ROOT/'results/rc_h593_quality_operator_eval_v1/result.json')['rows']}
    auth = read(ROOT/'registry/rc_h593_simple_explanations_authority_v1_20260924.json')
    identities = {r['physical_row']:r['identity'] for r in read(auth['public']['gallery']['path'])['records']}
    grouped = {}; n = 0
    for row in cache:
        m = meta[row['query_id']]
        target_scores = []
        for t, physical in enumerate(row['axis']):
            if identities[physical] != m['identity']:
                continue
            comparisons = []
            for j, other in enumerate(row['axis']):
                if identities[other] == m['identity']:
                    continue
                if abs(row['free_content'][t]-row['free_content'][j]) <= .005:
                    comparisons.append(1. if row['mass'][t] > row['mass'][j] else (.5 if row['mass'][t] == row['mass'][j] else 0.))
            if comparisons:
                target_scores.append(sum(comparisons)/len(comparisons))
        if target_scores:
            n += 1
            grouped.setdefault(m['component'],[]).append(sum(target_scores)/len(target_scores))
    independently_computed = sum(sum(v)/len(v) for v in grouped.values())/len(grouped)
    saved = result['external']['cohorts']['all_target_present']['Lband_0.005_M_concordance']
    assert n == saved['queries'] and len(grouped) == saved['groups']
    assert abs(independently_computed-saved['group_equal_mean']) < 1e-12

    a = read(ROOT/'registry/rc_postllm_m_v1_authority_20260924.json')
    manifest = read(a['manifest']['path']); norm = manifest['mass_normalization']
    xy = []; records = []
    for idx,row in enumerate(manifest['train_rows']+manifest['probe_rows']):
        d = read(ROOT/'results/rc_postllm_m_signal_decomposition_v2/queries'/row['query_id']/'result.json')
        z = [(math.log(max(m,norm['epsilon']))-norm['log_mean'])/norm['log_std'] for m in d['M']]
        c = d['arms']['CONSTANT']['L']; f = d['arms']['NATIVE_FIXED_ARGMAX']['L']
        delta = [v-u for u,v in zip(c,f)]
        if idx < 16:
            xy.extend(zip(z,delta))
        else:
            records.append((row,d,z,c,f,delta))
    # Closed-form slope rather than reusing the least-squares implementation.
    slope = math.fsum(x*y for x,y in xy)/math.fsum(x*x for x,y in xy)
    saved_slope = result['internal']['results'][0]['coefficients'][0]
    assert abs(slope-saved_slope) < 1e-14
    critical = []
    for row,d,z,c,f,delta in records:
        targets = row.get('target_positions')
        if targets is None:
            targets = [i for i,identity in enumerate(row['candidate_identities']) if identity == meta[row['query_id']]['identity']]
        if not targets:
            continue
        # Fix one target without using the new scores. All pilot identities
        # have one reference; fail closed if this assumption changes.
        assert len(targets) == 1
        t = targets[0]; wrong = [j for j in range(128) if j not in targets]
        for label,w in [('RAW_winner',d['winner_index']),('strongest_constant_content_wrong',max(wrong,key=lambda j:c[j]))]:
            if w == t:
                continue
            exact = delta[t]-delta[w]
            predicted = slope*(z[t]-z[w])
            critical.append(dict(query_id=row['query_id'],opponent=label,
                target_position=t,wrong_position=w,
                original_content_gap=c[t]-c[w], actual_fixed_content_gap=f[t]-f[w],
                actual_change_of_gap=exact, M_scalar_predicted_change=predicted,
                signed_residual=exact-predicted,
                same_change_sign=bool(exact*predicted>0),
                target_M=d['M'][t],wrong_M=d['M'][w]))
    output = dict(status='INDEPENDENT_INFORMATION_TRANSPORT_RECOUNT_PASS',
        source_hashes_verified=len(result['sources']),
        content_matched_M=dict(queries=n,groups=len(grouped),group_equal_concordance=independently_computed),
        independently_fitted_logM_slope=slope,
        critical_pair_rows=critical,
        critical_pair_max_abs_residual=max(abs(r['signed_residual']) for r in critical),
        interpretation='Scalar transport fit is an explanatory approximation, not a replacement-model performance result; critical close pairs can retain important residuals. No identity accuracy computed.',
        new_model_forwards=0)
    (OUT/'independent_validation.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
