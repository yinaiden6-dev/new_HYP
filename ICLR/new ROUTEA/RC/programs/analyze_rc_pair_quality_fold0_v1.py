#!/usr/bin/env python3
"""Sealed first-fold mechanism analysis; no training or encoder forward."""
import argparse
from collections import defaultdict
import csv
import datetime
import hashlib
import json
from math import comb
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_h593_pair_quality_cpu_v1'
OUT = ROOT / 'results/rc_pair_quality_fold0_analysis_v1'
AUTH = ROOT / 'registry/rc_pair_quality_fold0_analysis_v1_20260923.json'
PARENT = ROOT / 'registry/rc_h593_pair_quality_cpu_authority_v1_20260923.json'
ARMS = ('COL_ONLY_SINGLE', 'COL_ONLY_PAIR', 'COARSE_SINGLE', 'COARSE_PAIR', 'NONE', 'ROMA')
PAIRS = [('COL_ONLY_SINGLE', 'COL_ONLY_PAIR'), ('COARSE_SINGLE', 'COARSE_PAIR'),
         ('COL_ONLY_PAIR', 'COARSE_PAIR'), ('NONE', 'COL_ONLY_PAIR'), ('ROMA', 'COL_ONLY_PAIR')]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(binding):
    assert bind(binding['path']) == binding, ('SHA_DRIFT', binding['path'])
    return Path(binding['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == text, ('IMMUTABLE', str(path))
        return
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def prepare():
    parent = read(PARENT)
    seal_path = SOURCE / 'fold0_priority_prediction_seal.json'
    seal = read(seal_path)
    assert seal['status'] == 'FIRST_FOLD_SIX_ARMS_SEALED' and seal['authority'] == bind(PARENT)
    assert seal['folds'] == [0] and len(seal['fits']) == 6 and seal['heldout_label_reads'] == 0
    for i, fit in enumerate(seal['fits']):
        assert fit['index'] == i and fit['config'] == parent['configs'][i]
        for key in ('validation', 'payload', 'model'):
            checked(fit[key])
        assert len(fit['predictions']) == 119
    write(AUTH, dict(status='FIRST_FOLD_MECHANISM_ANALYSIS_AUTHORIZED',
                     time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     user_scope='Prioritize first fold; explain mechanism rather than seek new optimum',
                     parent=bind(PARENT), seal=bind(seal_path), source=bind(__file__),
                     fold=0, arms=list(ARMS), expected_queries=119, comparisons=PAIRS,
                     analysis='All six arms; RAW and original seven-parameter COST1; rescue overlap, candidate mass, margins and CBIND',
                     no_training=True, held_labels_read=False, group_bootstrap_seed=20260923,
                     group_bootstrap_samples=10000,
                     interpretation='Opened first-fold development evidence, simplified five-parameter action; not H593 OOF5 or external confirmation'))
    print('FIRST_FOLD_ANALYSIS_SCOPE_FROZEN', flush=True)


def action(candidate_rows, winner, challengers, logits):
    j = int(np.argmax(logits))
    return candidate_rows[challengers[j] if logits[j] > 0 else winner]


def pair_stats(rows, baseline, method, seed):
    rescue = sum(not r['models'][baseline]['correct'] and r['models'][method]['correct'] for r in rows)
    breaks = sum(r['models'][baseline]['correct'] and not r['models'][method]['correct'] for r in rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['models'][method]['correct']) - int(r['models'][baseline]['correct']))
    means = np.array([np.mean(values) for _, values in sorted(groups.items())])
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(0, len(means), (10000, len(means)))].mean(1)
    n = rescue + breaks
    p = min(1., 2 * sum(comb(n, j) for j in range(min(rescue, breaks) + 1)) / 2 ** n) if n else 1.
    return dict(rescue=rescue, breaks=breaks, net=rescue-breaks,
                group_equal_mean=float(means.mean()), group_ci95=np.quantile(sampled, [.025, .975]).tolist(),
                group_count=len(means), positive_groups=int((means > 0).sum()), negative_groups=int((means < 0).sum()),
                paired_query_p=p, query_p_is_descriptive=True,
                leave_one_group_out_min=float(min((means.sum()-v)/(len(means)-1) for v in means)) if len(means)>1 else None)


def run():
    a = read(AUTH)
    checked(a['source'])
    parent = read(checked(a['parent']))
    seal = read(checked(a['seal']))
    fits = []
    # Validate all six frozen payloads before opening held identities.
    for i, item in enumerate(seal['fits']):
        v = read(checked(item['validation']))
        p = read(checked(item['payload']))
        assert v['status'] == 'PAIR_QUALITY_FOLD_NUMPY_PASS'
        assert v['payload'] == item['payload'] and p['authority'] == a['parent']
        assert p['config'] == parent['configs'][i] and p['steps'] == 2000 and len(p['predictions']) == 119
        checked(p['model'])
        for b in p['predictions']:
            checked(b)
        fits.append(p)
    write(OUT / 'input_seal.json', dict(analysis_authority=bind(AUTH), parent=a['parent'],
                                       prediction_seal=a['seal'], heldout_label_reads=0))
    roles = {r['query_id']: r for r in read(checked(parent['join_sources']['curator']))['records']}
    labels = {r['physical_row']: r['identity'] for r in read(checked(parent['public_sources']['gallery']))['records']}
    fs = parent['fold_sources']['0']
    tr = read(checked(fs['train_roles']))['records']
    train_ids = {r['identity'] for r in tr}
    train_groups = {r['component'] for r in tr}
    old = {r['query_id']: r for r in read(checked(fs['full_payload']))['predictions']}
    rows = {}
    max_error = 0.
    checked_scores = 0
    for arm, fit in zip(ARMS, fits):
        model = torch.load(checked(fit['model']), map_location='cpu', weights_only=True)
        theta = model['theta'].numpy()
        assert theta.shape == (5,)
        for binding in fit['predictions']:
            p = torch.load(binding['path'], map_location='cpu', weights_only=True)
            qid = p['query_id']; role = roles[qid]
            assert role['outer_fold'] == 0 and role['component'] not in train_groups and role['identity'] not in train_ids
            cand = p['candidate_physical_rows']; w = p['winner']; ix = p['challenger_positions']
            assert len(cand) == 128 and len(set(cand)) == 128 and len(ix) == 127
            target = [i for i, physical in enumerate(cand) if labels[physical] == role['identity']]
            assert len(target) <= 1
            m = p['M'].numpy(); l = p['L0'].numpy(); raw = np.array(p['raw'])
            def sym(v):
                return (v[ix]-v[w])/(abs(v[ix])+abs(v[w])+1e-12)
            gap = (raw[ix]-raw[w])/max(float(raw.std()), 1e-12)
            xn = np.stack([gap, sym(m*l), sym(m), sym(l)], axis=1)
            z = (xn*theta[:-1]).sum(1)+theta[-1]
            mc = np.roll(m, 1)
            xc = np.stack([gap, sym(mc*l), sym(mc), sym(l)], axis=1)
            zc = (xc*theta[:-1]).sum(1)+theta[-1]
            error = max(float(abs(z-p['logits'].numpy()).max()), float(abs(zc-p['cbind_logits'].numpy()).max()))
            assert error < 2e-10
            max_error = max(max_error, error); checked_scores += 254
            assert action(cand,w,ix,z) == p['selected'] and action(cand,w,ix,zc) == p['cbind_selected']
            if p['quality_traces'] is not None:
                rebuilt = np.array([np.sqrt(t['u'].numpy().mean()*t['v'].numpy().mean()) for t in p['quality_traces']])
                assert np.max(abs(rebuilt-m)) < 2e-12
            if qid not in rows:
                oz = np.array([float.fromhex(x) for x in old[qid]['models']['COST1']['logits_hex']])
                osel = old[qid]['models']['COST1']['selected']
                assert action(cand,w,ix,oz) == osel
                rows[qid] = dict(query_id=qid, execution_ordinal=p['execution_ordinal'], component=role['component'],
                                 fold=0, target_in_C128=bool(target), target_position=target[0] if target else None,
                                 candidate_physical_rows=cand, winner=w, target_identity=role['identity'],
                                 original_logits=oz.tolist(), models={
                                     'RAW':dict(selected=cand[w],correct=labels[cand[w]]==role['identity']),
                                     'ORIGINAL_COST1':dict(selected=osel,correct=labels[osel]==role['identity'])})
            r = rows[qid]
            assert r['candidate_physical_rows'] == cand and r['winner'] == w
            scores = np.zeros(128); scores[ix] = z
            entry = dict(selected=p['selected'], correct=labels[p['selected']]==role['identity'],
                         changed_from_original=p['selected']!=r['models']['ORIGINAL_COST1']['selected'],
                         mass_mean=float(m.mean()), mass_std=float(m.std()), mass_range=float(np.ptp(m)),
                         source=binding, logits=z.tolist(), M=m.tolist(), L0=l.tolist())
            if target:
                t = target[0]; wrong = np.arange(128)!=t
                maxwrong = float(m[wrong].max())
                entry.update(target_margin=float(scores[t]-scores[wrong].max()),
                             target_quality=float(m[t]), strongest_wrong_quality=maxwrong,
                             quality_separation=float((m[t]-maxwrong)/(abs(m[t])+abs(maxwrong)+1e-12)),
                             quality_rank_min=int((m>m[t]).sum()+1),
                             quality_rank_max=int((m>=m[t]).sum()),
                             target_score=float(scores[t]))
            r['models'][arm] = entry
            r['models'][arm+'_CBIND'] = dict(selected=p['cbind_selected'],correct=labels[p['cbind_selected']]==role['identity'])
        print('ANALYZED',arm,119,flush=True)
    rows = sorted(rows.values(),key=lambda r:r['query_id'])
    assert len(rows)==119 and all(set(r['models'])==set(rows[0]['models']) for r in rows)
    original_rescues = [r for r in rows if not r['models']['RAW']['correct'] and r['models']['ORIGINAL_COST1']['correct']]
    original_breaks = [r for r in rows if r['models']['RAW']['correct'] and not r['models']['ORIGINAL_COST1']['correct']]
    summary = {}
    for arm in rows[0]['models']:
        summary[arm] = dict(correct=sum(r['models'][arm]['correct'] for r in rows), n=119,
                            versus_RAW=pair_stats(rows,'RAW',arm,a['group_bootstrap_seed']),
                            versus_original=pair_stats(rows,'ORIGINAL_COST1',arm,a['group_bootstrap_seed']),
                            retained_original_rescues=sum(r['models'][arm]['correct'] for r in original_rescues),
                            reproduced_original_breaks=sum(not r['models'][arm]['correct'] for r in original_breaks),
                            identical_original_decisions=sum(r['models'][arm]['selected']==r['models']['ORIGINAL_COST1']['selected'] for r in rows))
        if arm in ARMS:
            present=[r['models'][arm] for r in rows if r['target_in_C128']]
            summary[arm].update(mean_target_margin=float(np.mean([p['target_margin'] for p in present])),
                                mean_quality_separation=float(np.mean([p['quality_separation'] for p in present])),
                                mean_candidate_mass_std=float(np.mean([r['models'][arm]['mass_std'] for r in rows])),
                                quality_target_strict_first=sum(p['quality_rank_max']==1 for p in present))
    comparisons={b+'__to__'+m:pair_stats(rows,b,m,a['group_bootstrap_seed']) for b,m in PAIRS}
    maximum=0.
    for rank,(key,v) in enumerate(sorted(comparisons.items(),key=lambda item:item[1]['paired_query_p'])):
        maximum=max(maximum,min(1.,v['paired_query_p']*(len(comparisons)-rank)));v['holm_query_p']=maximum
    result=dict(status='FIRST_FOLD_SIX_ARM_MECHANISM_ANALYSIS_COMPLETE',authority=bind(AUTH),
                n=119,components=len({r['component'] for r in rows}),identities=len({r['target_identity'] for r in rows}),
                train_roles=len(tr),train_valid=len(fits[0]['train_query_ids']),target_recall=sum(r['target_in_C128'] for r in rows),
                original_rescues=[r['query_id'] for r in original_rescues],original_breaks=[r['query_id'] for r in original_breaks],
                summary=summary,comparisons=comparisons,train_fit=[dict(config=f['config'],**f['train_fit']) for f in fits],
                theta={arm:torch.load(checked(f['model']),map_location='cpu',weights_only=True)['theta'].tolist() for arm,f in zip(ARMS,fits)},
                rows=rows,evidence='Opened first-fold held panel; same fixed training split, seed17; no universal information absence claim')
    write(OUT/'result.json',result)
    # Recount selections using physical identity, independent of stored correct flags.
    for arm,s in summary.items():
        assert s['correct']==sum(labels[r['models'][arm]['selected']]==r['target_identity'] for r in rows)
        assert s['correct']==summary['RAW']['correct']+s['versus_RAW']['net']
    write(OUT/'validation.json',dict(status='FIRST_FOLD_NUMPY_SCORES_AND_IDENTITY_COUNTS_PASS',
                                     result=bind(OUT/'result.json'),n=119,arms=6,
                                     checked_scores=checked_scores,max_logit_error=max_error))
    with (OUT/'summary.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['arm','correct','n','rescue_vs_RAW','break_vs_RAW','original_rescues_retained','identical_original_decisions'])
        for arm,s in summary.items():w.writerow([arm,s['correct'],119,s['versus_RAW']['rescue'],s['versus_RAW']['breaks'],s['retained_original_rescues'],s['identical_original_decisions']])
    print(json.dumps({arm:{k:v for k,v in s.items() if k not in ('versus_RAW','versus_original')} for arm,s in summary.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    prepare() if args.stage=='prepare' else run()
