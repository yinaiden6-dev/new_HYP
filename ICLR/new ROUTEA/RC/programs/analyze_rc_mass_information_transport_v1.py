#!/usr/bin/env python3
"""Describe information complementarity and score transport, not retrieval gains.

Read-only opened H593 / POST24 sources. No encoder, RoMa, adapter, head or
threshold training. Regressions are explanatory probes fitted on TRAIN only;
they are not proposed retrieval models or causal interventions.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_mass_information_transport_v1'
SOURCES = []


def read(p):
    p = Path(p)
    b = p.read_bytes()
    SOURCES.append(dict(path=str(p.resolve()), sha256=hashlib.sha256(b).hexdigest()))
    return json.loads(b)


def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def grouped(rows, key):
    groups = {}
    for r in rows:
        if r.get(key) is not None:
            groups.setdefault(r['component'], []).append(r[key])
    v = np.array([np.mean(x) for x in groups.values()])
    if not len(v):
        return dict(queries=0, groups=0)
    rng = np.random.default_rng(20260924)
    draws = v[rng.integers(len(v), size=(5000, len(v)))].mean(1)
    return dict(queries=sum(len(x) for x in groups.values()), groups=len(v),
                group_equal_mean=float(v.mean()),
                exploratory_group_bootstrap95=np.quantile(draws, [.025, .975]).tolist())


def win(a, b):
    return float(np.mean((a > b) + .5 * (a == b)))


def external():
    a = read(ROOT / 'registry/rc_h593_simple_explanations_authority_v1_20260924.json')
    cache = read(ROOT / 'results/rc_h593_simple_explanations_v1/cache.json')
    old = read(ROOT / 'results/rc_h593_quality_operator_eval_v1/result.json')
    gallery = read(a['public']['gallery']['path'])['records']
    split = read(a['public']['split']['path'])['folds']
    gid = {x['physical_row']: x['identity'] for x in gallery}
    meta = {x['query_id']: x for x in old['rows']}
    rows = cache['rows']
    assert len(rows) == len(meta) == 593
    # Fit log M using only outcome-free summaries already available to content
    # retrieval: standardized RAW gap and within-query centered free MaxSim.
    # A cubic is a declared limited explanatory family, NOT all token readouts.
    inputs, response = {}, {}
    for r in rows:
        raw = np.zeros(128)
        raw[r['challengers']] = np.array(r['X'])[:, 0]
        l = np.array(r['free_content'])
        inputs[r['query_id']] = np.stack([raw, l-l.mean()], 1)
        z = np.log(np.maximum(r['mass'], 1e-12))
        response[r['query_id']] = z-z.mean()

    def basis(x):
        u, v = x.T
        return np.stack([np.ones(len(x)), u, v, u*u, u*v, v*v,
                         u**3, u*u*v, u*v*v, v**3], 1)

    residual, foldstats = {}, []
    for f in split:
        train, held = f['train_query_ids'], f['heldout_query_ids']
        assert not set(train) & set(held)
        assert not {meta[q]['component'] for q in train} & {meta[q]['component'] for q in held}
        x = np.concatenate([inputs[q] for q in train])
        y = np.concatenate([response[q] for q in train])
        mean, std = x.mean(0), np.maximum(x.std(0), 1e-12)
        design = basis((x-mean)/std)
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        e, yy = [], []
        for q in held:
            pred = basis((inputs[q]-mean)/std) @ beta
            pred -= pred.mean()
            residual[q] = response[q]-pred
            e.extend(residual[q]); yy.extend(response[q])
        e, yy = np.array(e), np.array(yy)
        foldstats.append(dict(fold=f['fold'], train_queries=len(train), held_queries=len(held),
            held_query_centered_R2=float(1-np.dot(e,e)/np.dot(yy,yy)),
            coefficients=beta.tolist(), input_mean=mean.tolist(), input_std=std.tolist()))
    details = []
    for r in rows:
        q = r['query_id']; m = meta[q]
        y = np.array([gid[g] == m['identity'] for g in r['axis']])
        assert bool(y.any()) == m['target_in_C128']
        assert bool(y[r['winner']]) == m['correct']['RAW']
        if not y.any():
            continue
        # Physical-reference ties belonging to the same identity are retained;
        # all target-vs-wrong comparisons are averaged before group aggregation.
        targets, wrong = np.where(y)[0], np.where(~y)[0]
        l, lm = np.array(r['free_content']), np.log(np.maximum(r['mass'], 1e-12))
        lmres = residual[q]
        row = dict(query_id=q, component=m['component'], raw_correct=m['correct']['RAW'],
                   targets=len(targets), fold=m['fold'])
        row['all_wrong_M_concordance'] = float(np.mean([win(lm[t],lm[wrong]) for t in targets]))
        row['all_wrong_content_concordance'] = float(np.mean([win(l[t],l[wrong]) for t in targets]))
        nearest = [int(wrong[np.argmin(abs(l[wrong]-l[t]))]) for t in targets]
        row['nearest_content_M_concordance'] = float(np.mean([win(lm[t],lm[w]) for t,w in zip(targets,nearest)]))
        row['nearest_content_logM_gap'] = float(np.mean([lm[t]-lm[w] for t,w in zip(targets,nearest)]))
        row['nearest_content_abs_L_gap'] = float(np.mean([abs(l[t]-l[w]) for t,w in zip(targets,nearest)]))
        row['nearest_content_residual_M_concordance'] = float(np.mean([win(lmres[t],lmres[w]) for t,w in zip(targets,nearest)]))
        for width in [.005,.01,.02]:
            vals, resvals, counts = [], [], []
            for t in targets:
                matched = wrong[abs(l[wrong]-l[t]) <= width]
                if len(matched):
                    vals.append(win(lm[t],lm[matched])); resvals.append(win(lmres[t],lmres[matched])); counts.append(len(matched))
            name = 'Lband_' + str(width)
            row[name+'_M_concordance'] = float(np.mean(vals)) if vals else None
            row[name+'_residual_M_concordance'] = float(np.mean(resvals)) if vals else None
            row[name+'_wrong_count'] = float(np.mean(counts)) if vals else None
        # Conflicts with stronger content impostors are described as failures
        # as well as agreements; no selection on COST1 correction outcomes.
        vals = [win(lm[t],lm[wrong[l[wrong]>l[t]]]) for t in targets if np.any(l[wrong]>l[t])]
        row['content_superior_wrong_M_concordance'] = float(np.mean(vals)) if vals else None
        details.append(row)
    keys = [k for k in details[0] if k not in ('query_id','component','raw_correct','targets','fold')]
    cohorts = {'all_target_present': details,
               'RAW_correct': [r for r in details if r['raw_correct']],
               'RAW_wrong_target_present': [r for r in details if not r['raw_correct']]}
    result = dict(scope='Opened H593, original five groups folds and ColNomic natural C128. No outcome-based rescue selection.',
        total_queries=593, target_present=len(details), candidate_missing=593-len(details),
        cohorts={c:{k:grouped(v,k) for k in keys} for c,v in cohorts.items()},
        content_predicts_logM_folds=foldstats,
        limitation='Matching on free L is approximate and does not balance image difficulty or all content features. Residualization rejects only a cubic of two scalar summaries; neither conditional independence nor token information absence is proved.')
    write(OUT/'external_per_query.json',details)
    return result


def internal():
    authority = read(ROOT / 'registry/rc_postllm_m_v1_authority_20260924.json')
    manifest = read(authority['manifest']['path'])
    norm = manifest['mass_normalization']
    components = {r['query_id']: r['component'] for r in read(
        ROOT/'results/rc_h593_quality_operator_eval_v1/result.json')['rows']}
    rows = manifest['train_rows']+manifest['probe_rows']
    arrays, records = [], []
    for row in rows:
        d = read(ROOT/'results/rc_postllm_m_signal_decomposition_v2/queries'/row['query_id']/'result.json')
        assert d['candidate_ids'] == row['candidate_ids']
        assert np.max(abs(np.array(d['M'])-row['M'])) == 0
        z = (np.log(np.maximum(d['M'], norm['epsilon']))-norm['log_mean'])/norm['log_std']
        const = np.array(d['arms']['CONSTANT']['L'])
        delta = np.array(d['arms']['NATIVE_FIXED_ARGMAX']['L'])-const
        native = np.array(d['arms']['NATIVE']['L'])-const
        arrays.append((z,const,delta,native))
        records.append(dict(query_id=row['query_id'],component=components[row['query_id']],split=row['split']))
    nt = len(manifest['train_rows']); assert nt == 16 and len(rows) == 24
    assert not {components[r['query_id']] for r in manifest['train_rows']} & {components[r['query_id']] for r in manifest['probe_rows']}
    train = np.concatenate([np.stack(x[:3],1) for x in arrays[:nt]])
    lmean = float(train[:,1].mean())

    def design(z,l,degree,interaction):
        columns = [z**k for k in range(1,degree+1)]
        if interaction:
            columns += [(l-lmean)*z**k for k in range(1,degree+1)]
        return np.stack(columns,1)

    results = []
    for degree, interaction in [(1,False),(2,False),(3,False),(3,True)]:
        beta = np.linalg.lstsq(design(train[:,0],train[:,1],degree,interaction),train[:,2],rcond=None)[0]
        detail = []
        for rec, (z,l,delta,native) in zip(records,arrays):
            pred = design(z,l,degree,interaction) @ beta
            centered = delta-delta.mean(); err = delta-pred
            centerr = centered-(pred-pred.mean())
            detail.append(dict(**rec, score_change_RMSE=float(np.sqrt(np.mean(err**2))),
                candidate_contrast_R2=float(1-np.dot(centerr,centerr)/max(np.dot(centered,centered),1e-30)),
                candidate_contrast_RMSE=float(np.sqrt(np.mean(centerr**2))),
                actual_change_range=float(np.ptp(delta)),
                rematching_extra_RMS=float(np.sqrt(np.mean((native-delta)**2)))))
        held = detail[nt:]
        results.append(dict(model=f'M_polynomial_degree{degree}'+('_with_content_interaction' if interaction else ''),
            coefficients=beta.tolist(), fit_queries=nt, fit_labels_used=False,
            zero_response_at_standardized_logM_zero=True,
            probe_group_equal={k:grouped(held,k) for k in ('candidate_contrast_R2','candidate_contrast_RMSE','score_change_RMSE','rematching_extra_RMS')},
            per_query=detail))
    return dict(scope='Sealed POST_REAL128; TRAIN16 explanatory fit and previously opened PROBE8; frozen original adapter/projector/INTERNAL3. No new adapter or head fit.',
        target_response='Saved fixed-reference-hit change in free-content score between real and constant internal M.',
        fit_family='No-intercept polynomial of standardized logM, degree1/2/3 declared together; degree3 with interactions with constant-condition content is a diagnostic comparator.',
        results=results,
        limitation='A predictive scalar surrogate is not a causal intervention and is not equivalence of representations. No claim of internal/external head identity or population invariance; 8 probes are already opened.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'result.json').exists(), 'Output already exists; do not overwrite analysis results'
    spec = dict(status='DECLARED_BEFORE_COMPUTE', purpose='Information complementarity and implementation-independent score transport, not top1 improvement',
        external='All target-present H593; group-equal target-vs-content-matched wrong M concordance, including RAW-correct and RAW-wrong. Nearest and all three fixed bands reported.',
        internal='TRAIN-only scalar surrogate of existing POST effect, report every declared degree and content interaction on opened PROBE8. No endpoint selection.',
        statistical_unit='Query averages, then component averages. 5000 group bootstrap draws, exploratory, uncorrected.',
        no_model_or_threshold_training=True, no_gpu_or_roma_or_llm_forward=True)
    write(OUT/'protocol.json',spec)
    ext = external(); post = internal()
    source = Path(__file__)
    SOURCES.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    result = dict(status='INFORMATION_AND_TRANSPORT_DIAGNOSTIC_COMPLETE', external=ext, internal=post, sources=SOURCES)
    write(OUT/'result.json', result)
    print(json.dumps(dict(status=result['status'], output=str(OUT),
        M_matched=ext['cohorts']['all_target_present']['Lband_0.005_M_concordance'],
        M_matched_raw_wrong=ext['cohorts']['RAW_wrong_target_present']['Lband_0.005_M_concordance'],
        internal=[dict(model=x['model'],contrast=x['probe_group_equal']['candidate_contrast_R2']) for x in post['results']]),indent=2))


if __name__ == '__main__':
    main()
