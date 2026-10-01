#!/usr/bin/env python3
"""Replay saved OOF spline coefficients and independently audit group separation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline

RC = Path(__file__).resolve().parents[1]
ROOT = RC / 'results/rc_m_conditional_prediction_v1'


def read(p):
    return json.loads(p.read_text())


def binding(p):
    return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def basis(values, spec):
    knots = np.array(spec['knots'])
    spline = BSpline(knots, np.eye(len(knots)-4), 3)
    low, high = knots[3], knots[-4]
    raw = np.asarray(values)
    x = raw.ravel()
    result = spline(np.clip(x, low, high))
    for edge, selected in [(low, x < low), (high, x > high)]:
        result[selected] += (x[selected]-edge)[:, None] * spline.derivative()(edge)
    return result[:, :-1].reshape(*raw.shape, -1)


def main():
    protocol = read(ROOT / 'protocol.json')
    data = {r['query_id']: r for r in read(ROOT / 'data.json')['rows']}
    maxima = dict(score=0., NLL=0., feature_scale=0.)
    seen = {a: set() for a in protocol['arms']}
    sources = []
    for fold in range(5):
        split = next(s for s in protocol['splits'] if s['fold'] == fold)
        for arm in protocol['arms']:
            path = ROOT / f'fold{fold}/{arm}.json'
            model = read(path); spec = model['design']
            train = [data[k] for k in model['train_ids']]
            held = [data[k] for k in model['held_ids']]
            assert set(model['train_ids']) == set(split['train_query_ids']) & data.keys()
            assert set(model['held_ids']) == set(split['heldout_query_ids']) & data.keys()
            assert not {r['component'] for r in train} & {r['component'] for r in held}
            inner = [read(ROOT / f'fold{fold}/inner/{arm}_{k}_{ridge}.json')
                     for k in range(2) for ridge in protocol['ridges']]
            assert len(inner) == 8
            for fit in inner:
                assert set(fit['train_ids']) | set(fit['validation_ids']) == set(model['train_ids'])
                assert not {data[k]['component'] for k in fit['train_ids']} & {data[k]['component'] for k in fit['validation_ids']}
                assert fit['fit']['max_abs_gradient'] < 2e-5
            chosen = min(protocol['ridges'], key=lambda x: (
                sum(v['NLL']*v['validation_groups'] for v in inner if v['ridge']==x) /
                sum(v['validation_groups'] for v in inner if v['ridge']==x), -x))
            assert model['ridge'] == chosen and model['fit']['max_abs_gradient'] < 2e-5
            def design(rows):
                c = np.asarray([r['C'] for r in rows]); m = np.asarray([r['logM'] for r in rows])
                cb = basis(c, spec['C']); parts = [cb]
                if 'residual_beta' in spec:
                    z = np.concatenate([np.ones((*cb.shape[:2],1)), cb], axis=-1)
                    m = m-z@np.array(spec['residual_beta'])
                if 'M' in spec:
                    parts.append(basis(m, spec['M']))
                if 'RAW' in spec:
                    parts.append(basis([r['RAW'] for r in rows], spec['RAW']))
                features = np.concatenate(parts, axis=-1)
                return features-features.mean(axis=1, keepdims=True)
            x = design(train)
            counts = {r['component']: sum(v['component']==r['component'] for v in train) for r in train}
            weights = np.array([1/(len(counts)*counts[r['component']]) for r in train])
            scale = np.maximum(np.sqrt(np.sum(weights[:,None,None]*x*x, axis=(0,1))/128), 1e-6)
            maxima['feature_scale'] = max(maxima['feature_scale'], float(np.max(abs(scale-np.array(spec['feature_scale'])))))
            for channel in ['C','RAW']:
                if channel not in spec: continue
                values = np.asarray([r[channel] for r in train])
                actual = np.array(spec[channel]['knots'])[3:-3]
                assert np.allclose(actual,np.quantile(values,np.linspace(0,1,len(actual))),rtol=0,atol=1e-12)
            scores = design(held)/np.array(spec['feature_scale']) @ np.array(model['coefficients'])
            for r,s in zip(model['rows'],scores):
                assert r['query_id'] not in seen[arm]; seen[arm].add(r['query_id'])
                original = data[r['query_id']]
                assert r['axis']==original['axis'] and r['target']==original['target']
                maxima['score'] = max(maxima['score'],float(np.max(abs(s-np.asarray(r['scores'])))))
                nll = float(np.log(np.exp(s-s.max()).sum())+s.max()-s[r['target']])
                maxima['NLL'] = max(maxima['NLL'],abs(nll-r['NLL']))
                assert bool(np.argmax(s)==r['target']) == r['correct']
            sources.append(binding(path))
    assert all(v==set(data) for v in seen.values()) and max(maxima.values())<1e-9
    result = dict(status='INDEPENDENT_SAVED_MODEL_REPLAY_AND_SPLIT_AUDIT_PASS',max_errors=maxima,
        queries=570,models=30,sources=sources,auditor=binding(Path(__file__)),
        boundary='Opened grouped OOF; fixed-family predictive increment, not Shannon CMI or unique causal attribution.')
    output = ROOT / 'independent_model_replay.json'
    encoded = json.dumps(result,indent=2,sort_keys=True)+'\n'
    if output.exists(): assert output.read_text()==encoded
    else: output.write_text(encoded)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}))


if __name__ == '__main__':
    main()
