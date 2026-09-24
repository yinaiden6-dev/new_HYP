#!/usr/bin/env python3
"""Cached, grouped OOF comparisons of joint calibration and simpler explanations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import run_rc_six_cause_loss_binding_v1 as LOSS

OUT = ROOT / 'results/rc_h593_simple_explanations_v1'
AUTH = ROOT / 'registry/rc_h593_simple_explanations_authority_v1_20260924.json'
PLAN = ROOT / 'plan/RC_H593_SIMPLE_EXPLANATIONS_V1_20260924.md'
LAUNCH = ROOT / 'slurm/rc_h593_simple_explanations_v1.sbatch'
PARENT = ROOT / 'registry/rc_h593_quality_operator_eval_authority_v1_20260922.json'
SOURCE = ROOT / 'results/rc_h593_quality_operator_v1'
REPORT = ROOT / 'reports/REPORT_H593_SIMPLE_EXPLANATIONS_V1_20260924.md'
ARMS = ('RAW2', 'QUALITY2', 'CONTENT2', 'RAW_QUALITY3', 'RAW_CONTENT3',
        'ADDITIVE4', 'PRODUCT5', 'ADD_M2_5', 'ADD_L2_5')
BUDGETS = {'b0': 0., 'b1': .01, 'b3': .03, 'b5': .05, 'net': 1.}


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(b):
    need(bind(b['path']) == b, 'SHA:' + b['path'])
    return Path(b['path'])


def sym(a, b):
    return (a - b) / (np.abs(a) + np.abs(b) + 1e-12)


def design(x, arm):
    cols = {'RAW2': [0], 'QUALITY2': [2], 'CONTENT2': [3],
            'RAW_QUALITY3': [0, 2], 'RAW_CONTENT3': [0, 3],
            'ADDITIVE4': [0, 2, 3], 'PRODUCT5': [0, 1, 2, 3]}
    if arm in cols:
        return x[..., cols[arm]].copy()
    base = x[..., [0, 2, 3]]
    col = 2 if arm == 'ADD_M2_5' else 3
    need(arm in ('ADD_M2_5', 'ADD_L2_5'), 'ARM')
    return np.concatenate([base, x[..., col:col + 1] ** 2], axis=-1)


def threshold_value(x):
    return {'-inf': -np.inf, '+inf': np.inf}.get(x, x) if isinstance(x, str) else float(x)


def threshold_json(x):
    return '+inf' if x == np.inf else '-inf' if x == -np.inf else float(x)


def grid(v, count, zero=True):
    vals = list(np.quantile(v, np.linspace(0, 1, count))) + [-np.inf, np.inf]
    if zero:
        vals.append(0.)
    return sorted(set(vals))


def action(z, threshold=0., gate=None, winner_mass=None):
    arg = np.argmax(z, axis=1)
    switch = np.max(z, axis=1) > threshold
    if gate is not None:
        switch &= winner_mass < gate
    return np.where(switch, arg + 1, 0).astype(int)


def stats(pred, target):
    correct = pred == target
    raw = target == 0
    return dict(correct=int(correct.sum()), rescue=int((correct & ~raw).sum()),
                breaks=int((~correct & raw).sum()), switches=int((pred != 0).sum()),
                raw_correct=int(raw.sum()), count=len(target))


def select_thresholds(zfit, zval, yval, mfit=None, mval=None):
    gates = grid(mfit, 31, False) if mfit is not None else [np.inf]
    thresholds = grid(zfit.max(1), 31 if mfit is not None else 101)
    best = {}; keys = {}
    for gate in gates:
        for tau in thresholds:
            pred = action(zval, tau, gate if mfit is not None else None, mval)
            st = stats(pred, yval)
            key = (st['correct'], -st['breaks'], -st['switches'], tau, -gate)
            for name, budget in BUDGETS.items():
                if st['breaks'] <= budget * st['raw_correct'] + 1e-12:
                    if name not in keys or key > keys[name]:
                        keys[name] = key
                        best[name] = dict(threshold=threshold_json(tau),
                            gate=threshold_json(gate) if mfit is not None else None,
                            validation=st, requested_validation_break_rate=budget)
    need(set(best) == set(BUDGETS), 'ALL_HOLD_FEASIBLE')
    return best


def prepare():
    need(not AUTH.exists(), 'FRESH_AUTHORITY')
    p = read(PARENT)
    sources = []
    for i in range(593):
        vpath = SOURCE / f'query{i:03d}/validation.json'
        v = read(vpath)
        need(v['status'] == 'QUALITY_OPERATOR_QUERY_PASS', 'SOURCE_VALIDATED')
        sources.append(dict(validation=bind(vpath), payload=v['payload']))
    oldfolds = []
    for f in range(5):
        d = ROOT / f'results/rc_h593_quality_operator_eval_v1/fold{f}'
        v = read(d / 'validation.json')
        need(v['status'] == 'QUALITY_OPERATOR_EVAL_FRESH_NUMPY_PASS', 'PARENT_FOLD')
        oldfolds.append(dict(validation=bind(d / 'validation.json'), payload=v['payload']))
    codes = dict(p['code_sources'])
    codes.update(simple_program=bind(__file__), simple_plan=bind(PLAN), simple_launcher=bind(LAUNCH))
    a = dict(status='SIMPLE_EXPLANATIONS_AUTHORIZED', parent=bind(PARENT), codes=codes,
             sources=sources, parent_folds=oldfolds, public=p['public_sources'],
             folds=p['fold_sources'], join=p['join_sources'], arms=list(ARMS),
             budgets=BUDGETS, losses=['COST1', 'CE'], primary='COST1_PRODUCT5@zero minus COST1_ADDITIVE4@zero',
             updates=2000, seed=17, dtype='float64', evidence='Opened H593 grouped OOF5, not fresh external confirmation',
             encoders_run=0, roma_run=0)
    write(AUTH, a)
    print(dict(status='PREPARED', authority=bind(AUTH)), flush=True)


def guard(stage, fold):
    a = read(AUTH)
    for b in a['codes'].values():
        checked(b)
    checked(a['parent'])
    if stage not in ('preflight',):
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allowed = {Path(b['path']).resolve() for b in a['public'].values()}
    if stage == 'cache':
        for b in a['sources'] + a['parent_folds']:
            allowed.update(Path(x['path']).resolve() for x in b.values())
        allowed.update(Path(x['full_payload']['path']).resolve() for x in a['folds'].values())
    if stage in ('fit', 'verify'):
        need(fold in range(5), 'FOLD')
        allowed.add(Path(a['folds'][str(fold)]['train_roles']['path']).resolve())
    if stage in ('join', 'publish'):
        allowed.update(Path(x['path']).resolve() for x in a['join'].values())
        allowed.update(Path(x['train_roles']['path']).resolve() for x in a['folds'].values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve(); lower = str(path).lower()
        need(not any(s in lower for s in ('d1-mi', 'd1_mi', 'formal392', '/grozi/', '/target_join/')), 'PROTECTED')
        if 'curator_roles' in lower:
            need(stage == 'join' and (OUT / 'all_predictions_prelabel_seal.json').exists(), 'NO_HELD_LABELS')
        if ROOT / 'reports' in path.parents:
            need(stage == 'publish' and path == REPORT, 'NO_REPORT_READ')
        if ROOT / 'results' in path.parents:
            own = OUT in path.parents
            if own and stage in ('fit', 'verify'):
                first = path.relative_to(OUT).parts[0]
                own = first in ('cache.json', 'cache_validation.json', 'preflight.json', f'fold{fold}')
            need(own or path in allowed, 'UNLISTED_RESULT:' + str(path))
    sys.addaudithook(audit)
    if stage != 'preflight':
        need(read(OUT / 'preflight.json')['authority'] == bind(AUTH), 'PREFLIGHT')
    return a


def cache(a):
    rows = []; maxerr = 0.
    for i, source in enumerate(a['sources']):
        v = read(checked(source['validation'])); p = read(checked(source['payload']))
        need(v['payload'] == source['payload'] and p['execution_ordinal'] == i, 'SOURCE_AXIS')
        mode = p['modes']['M1Q0R0']; x = np.asarray(mode['X'], dtype=np.float64)
        m = np.array([r['visibility_mass'] for r in mode['scores']])
        l = np.array([r['real_score'] for r in p['modes']['M0Q0R0']['scores']])
        s = np.array([r['real_score'] for r in mode['scores']])
        w = p['winner']; c = p['challenger_positions']
        need(x.shape == (127, 6) and len(m) == 128 and sorted(c + [w]) == list(range(128)), 'C128')
        need(np.all(m >= 0) and np.isfinite(x).all(), 'FINITE')
        expected = np.stack([sym(m[c]*l[c], m[w]*l[w]), sym(m[c], m[w]),
                             sym(l[c], l[w])], axis=1)
        err = max(float(np.abs(expected - x[:, 1:4]).max()), float(np.abs(s - m*l).max()))
        maxerr = max(maxerr, err); need(err < 2e-10, 'FREE_CONTENT_FORMULA')
        need(np.max(np.abs(x[:, 4:])) < 2e-10, 'NO_LOCAL_RESPONSES')
        rows.append(dict(query_id=p['query_id'], execution_ordinal=i,
            source_image_sha256=p['source_image_sha256'], axis=p['candidate_physical_rows'],
            raw_ranked=p['raw_ranked_physical_rows'], winner=w, challengers=c,
            X=x[:, :4].tolist(), native_X=p['modes']['NATIVE']['X'], mass=m.tolist(), free_content=l.tolist()))
    native = []; prior_product = []
    for f in range(5):
        old = read(checked(a['folds'][str(f)]['full_payload']))
        native.append({k: old['parameters'][k] for k in ('COST1', 'ALL_CE')})
        oldop = read(checked(a['parent_folds'][f]['payload']))
        prior_product.append({k: oldop['parameters'][f'{k}_REFIT_M1Q0R0']['theta_hex'] for k in ('COST1', 'CE')})
    payload = dict(authority=bind(AUTH), rows=rows, native_parameters=native,
                   prior_product_parameters=prior_product, labels_included=False)
    write(OUT / 'cache.json', payload)
    write(OUT / 'cache_validation.json', dict(status='CACHE_PASS', authority=bind(AUTH),
          payload=bind(OUT / 'cache.json'), sources=a['sources'], max_formula_error=maxerr,
          queries=593, new_encoder_forwards=0, new_roma_forwards=0))
    print(dict(status='CACHE_PASS', max_formula_error=maxerr), flush=True)


def load_cache():
    v = read(OUT / 'cache_validation.json')
    need(v['status'] == 'CACHE_PASS' and v['authority'] == bind(AUTH), 'CACHE_SEAL')
    return read(checked(v['payload']))


def split_data(a, payload, fold):
    rows = payload['rows']; split = read(checked(a['public']['split']))['folds'][fold]
    roles = {r['query_id']: r for r in read(checked(a['folds'][str(fold)]['train_roles']))['records']}
    gallery = {r['physical_row']: r['identity'] for r in read(checked(a['public']['gallery']))['records']}
    trainids = set(split['train_query_ids']); heldids = set(split['heldout_query_ids'])
    need(set(roles) == trainids and not trainids & heldids, 'TRAIN_ONLY_ROLES')
    train = np.array([i for i, r in enumerate(rows) if r['query_id'] in trainids])
    held = np.array([i for i, r in enumerate(rows) if r['query_id'] in heldids])
    need(not {rows[i]['source_image_sha256'] for i in train} & {rows[i]['source_image_sha256'] for i in held}, 'IMAGE_DISJOINT')
    key = lambda c: hashlib.sha256(('SIMPLE_EXPLANATIONS_V1|' + c).encode()).hexdigest()
    components = sorted({r['component'] for r in roles.values()}, key=key)
    vc = set(components[:math.ceil(.2 * len(components))])
    inner_val = np.array([i for i in train if roles[rows[i]['query_id']]['component'] in vc])
    inner_fit = np.array([i for i in train if roles[rows[i]['query_id']]['component'] not in vc])
    identity = lambda inds: {roles[rows[i]['query_id']]['identity'] for i in inds}
    need(not identity(inner_fit) & identity(inner_val), 'INNER_IDENTITY_DISJOINT')
    targets = np.full(len(rows), -99, dtype=int)
    for i in train:
        r = rows[i]; order = [r['winner']] + r['challengers']; target = roles[r['query_id']]['identity']
        targets[i] = next((j for j, pos in enumerate(order) if gallery[r['axis'][pos]] == target), -1)
    return train, held, inner_fit, inner_val, targets, roles


def train(x, y):
    raise RuntimeError('use train_kind')


def train_kind(x, y, kind):
    need(len(y) > 0 and np.min(y) >= 0, 'EFFECTIVE_TRAIN')
    torch.manual_seed(17)
    return LOSS.train(torch.from_numpy(x), torch.from_numpy(y - 1), kind).numpy()


def predict(x, theta):
    return (torch.from_numpy(x) @ torch.from_numpy(theta[:-1]) + theta[-1]).numpy()


def fit(a, fold):
    start = time.monotonic(); p = load_cache(); rows = p['rows']
    tr, held, inner, val, y, roles = split_data(a, p, fold)
    full = np.asarray([r['X'] for r in rows], dtype=np.float64)
    inner_eff = inner[y[inner] >= 0]; train_eff = tr[y[tr] >= 0]
    directory = OUT / f'fold{fold}'; directory.mkdir(parents=True, exist_ok=True)
    split = dict(train=[rows[i]['query_id'] for i in tr], held=[rows[i]['query_id'] for i in held],
                 inner_fit=[rows[i]['query_id'] for i in inner], inner_validation=[rows[i]['query_id'] for i in val],
                 effective_train=[rows[i]['query_id'] for i in train_eff],
                 inner_fit_components=sorted({roles[rows[i]['query_id']]['component'] for i in inner}),
                 inner_validation_components=sorted({roles[rows[i]['query_id']]['component'] for i in val}))
    write(directory / 'split.json', split)
    for arm in ARMS:
        x = design(full, arm)
        for kind in ('COST1', 'CE'):
            name = kind + '_' + arm; path = directory / (name + '.json')
            if path.exists():
                need(read(path)['authority'] == bind(AUTH), 'RESUME_AUTHORITY'); continue
            if time.monotonic() - start > 380:
                return 75
            t0 = time.monotonic()
            ti = train_kind(x[inner_eff], y[inner_eff], kind)
            zfit = predict(x[inner], ti); zv = predict(x[val], ti)
            thresholds = select_thresholds(zfit, zv, y[val])
            theta = train_kind(x[train_eff], y[train_eff], kind)
            zh = predict(x[held], theta)
            parity = None
            if arm == 'PRODUCT5':
                old = np.array([float.fromhex(v) for v in p['prior_product_parameters'][fold][kind]])
                oldz = predict(full[held], np.r_[old[:4], old[-1]])
                parity = float(np.abs(zh - oldz).max())
                need(parity < 2e-7 and np.array_equal(action(zh), action(oldz)), 'PARENT_SIMPLIFIED_PARITY')
            write(path, dict(authority=bind(AUTH), model=name, fold=fold,
                 theta_hex=[float(v).hex() for v in theta], inner_theta_hex=[float(v).hex() for v in ti],
                 inner_fit_logits=zfit.tolist(), inner_validation_logits=zv.tolist(), held_logits=zh.tolist(),
                 thresholds=thresholds, prior_product_max_error=parity, seconds=time.monotonic()-t0,
                 heldout_label_reads=0, loss=kind, arm=arm))
            print(dict(event='MODEL_SEALED', fold=fold, model=name, seconds=time.monotonic()-t0), flush=True)
    wm = np.array([r['mass'][r['winner']] for r in rows])
    gate = select_thresholds(full[inner, :, 1], full[val, :, 1], y[val], wm[inner], wm[val])
    write(directory / 'gate.json', dict(authority=bind(AUTH), thresholds=gate,
          inner_fit_logits=full[inner, :, 1].tolist(), inner_validation_logits=full[val, :, 1].tolist(),
          held_logits=full[held, :, 1].tolist(), heldout_label_reads=0))
    verify(a, fold)
    return 0


def verify(a, fold):
    p = load_cache(); rows = p['rows']; tr, held, inner, val, y, roles = split_data(a, p, fold)
    full = np.asarray([r['X'] for r in rows]); directory = OUT / f'fold{fold}'
    expected_split = read(directory / 'split.json')
    need(expected_split['held'] == [rows[i]['query_id'] for i in held], 'HELD_AXIS')
    predictions = {}; bindings = []; maximum = 0.; checks = 0
    for arm in ARMS:
        x = design(full, arm)
        for kind in ('COST1', 'CE'):
            name = kind + '_' + arm; path = directory / (name + '.json'); q = read(path)
            need(q['authority'] == bind(AUTH) and q['fold'] == fold and q['model'] == name, 'MODEL_SEAL')
            for ids, par, field in ((inner, 'inner_theta_hex', 'inner_fit_logits'),
                                     (val, 'inner_theta_hex', 'inner_validation_logits'),
                                     (held, 'theta_hex', 'held_logits')):
                t = np.array([float.fromhex(v) for v in q[par]])
                z = np.sum(x[ids] * t[:-1], axis=-1) + t[-1]
                sealed = np.array(q[field]); err = float(np.abs(z-sealed).max()); maximum=max(maximum,err)
                need(err < 2e-10 and np.array_equal(action(z), action(sealed)), 'NUMPY_LOGITS_ACTIONS')
                checks += z.size
            need(select_thresholds(np.array(q['inner_fit_logits']), np.array(q['inner_validation_logits']), y[val]) == q['thresholds'], 'THRESHOLD_REPLAY')
            zh = np.array(q['held_logits']); predictions[name + '@zero'] = action(zh).tolist()
            for budget, t in q['thresholds'].items():
                predictions[name + '@' + budget] = action(zh, threshold_value(t['threshold'])).tolist()
            bindings.append(bind(path))
    wm = np.array([r['mass'][r['winner']] for r in rows]); g = read(directory / 'gate.json')
    need(g['authority'] == bind(AUTH), 'GATE_AUTHORITY')
    need(select_thresholds(full[inner, :, 1], full[val, :, 1], y[val], wm[inner], wm[val]) == g['thresholds'], 'GATE_REPLAY')
    for budget, t in g['thresholds'].items():
        predictions['GATE@' + budget] = action(full[held, :, 1], threshold_value(t['threshold']), threshold_value(t['gate']), wm[held]).tolist()
    nx = np.asarray([rows[i]['native_X'] for i in held])
    for kind, key in (('COST1', 'COST1'), ('CE', 'ALL_CE')):
        theta = np.array([float.fromhex(v) for v in p['native_parameters'][fold][key]])
        predictions['NATIVE_' + kind] = action(np.sum(nx*theta[:-1], axis=2)+theta[-1]).tolist()
    predictions['RAW'] = [0] * len(held)
    write(directory / 'predictions.json', dict(authority=bind(AUTH), fold=fold,
          query_ids=[rows[i]['query_id'] for i in held], predictions=predictions, heldout_label_reads=0))
    write(directory / 'validation.json', dict(status='FOLD_NUMPY_AND_SELECTION_PASS', authority=bind(AUTH),
          predictions=bind(directory / 'predictions.json'), split=bind(directory / 'split.json'),
          models=bindings, gate=bind(directory / 'gate.json'), max_numpy_error=maximum, logit_checks=checks))
    print(dict(status='FOLD_NUMPY_AND_SELECTION_PASS', fold=fold, models=len(bindings), maxerr=maximum), flush=True)


def compare(rows, baseline, new):
    components = sorted({r['component'] for r in rows})
    d = np.array([np.mean([int(r['correct'][new])-int(r['correct'][baseline]) for r in rows if r['component']==c]) for c in components])
    rng = np.random.default_rng(20260924)
    boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    null = (rng.choice([-1, 1], size=(20000, len(d))) * d).mean(1)
    gain = [r['query_id'] for r in rows if r['correct'][new] and not r['correct'][baseline]]
    loss = [r['query_id'] for r in rows if r['correct'][baseline] and not r['correct'][new]]
    return dict(baseline=baseline, new=new, rescue=len(gain), breaks=len(loss), net=len(gain)-len(loss),
                gained=gain, lost=loss, equal_component_delta=float(d.mean()),
                bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                sign_flip_p=float((1+(np.abs(null)>=abs(d.mean())-1e-15).sum())/20001))


def join(a):
    ps = []; seals = []
    for f in range(5):
        d = OUT / f'fold{f}'; v = read(d / 'validation.json')
        need(v['status']=='FOLD_NUMPY_AND_SELECTION_PASS' and v['authority']==bind(AUTH), 'ALL5_VALIDATED')
        for b in v['models'] + [v['gate'], v['split']]:
            checked(b)
        ps.append(read(checked(v['predictions']))); seals.append(bind(d/'validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json', dict(authority=bind(AUTH), folds=seals))
    roles = {r['query_id']: r for r in read(checked(a['join']['curator']))['records']}
    gallery = {r['physical_row']: r['identity'] for r in read(checked(a['public']['gallery']))['records']}
    cache = load_cache(); byid = {r['query_id']:r for r in cache['rows']}; rows=[]
    for fold, p in enumerate(ps):
        train = read(checked(a['folds'][str(fold)]['train_roles']))['records']
        train_ids={r['identity'] for r in train}; train_components={r['component'] for r in train}
        for j,q in enumerate(p['query_ids']):
            r=byid[q]; role=roles[q]; need(role['outer_fold']==fold and role['identity'] not in train_ids and role['component'] not in train_components, 'OUTER_DISJOINT')
            axis=[r['axis'][r['winner']]]+[r['axis'][i] for i in r['challengers']]
            selected={m:axis[v[j]] for m,v in p['predictions'].items()}
            correct={m:gallery[s]==role['identity'] for m,s in selected.items()}
            rows.append(dict(query_id=q, fold=fold, component=role['component'], execution_ordinal=r['execution_ordinal'],
                             target_in_C128=any(gallery[s]==role['identity'] for s in axis), selected=selected, correct=correct))
    rows.sort(key=lambda r:r['execution_ordinal'])
    need([r['execution_ordinal'] for r in rows]==list(range(593)), 'ALL593_ONCE')
    summary={}
    for m in rows[0]['correct']:
        summary[m]=dict(correct=sum(r['correct'][m] for r in rows), total=593,
                        rescue=sum(r['correct'][m] and not r['correct']['RAW'] for r in rows),
                        breaks=sum(not r['correct'][m] and r['correct']['RAW'] for r in rows),
                        switches=sum(r['selected'][m]!=r['selected']['RAW'] for r in rows),
                        by_fold={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)})
        summary[m]['raw_correct_loss_rate']=summary[m]['breaks']/426
    need(summary['RAW']['correct']==426 and summary['NATIVE_COST1']['correct']==481 and summary['NATIVE_CE']['correct']==486, 'NATIVE_PARITY')
    need(sum(r['target_in_C128'] for r in rows)==570, 'RECALL570')
    pairs=[('COST1_ADDITIVE4@zero','COST1_PRODUCT5@zero'),('COST1_ADD_M2_5@zero','COST1_PRODUCT5@zero'),
           ('COST1_ADD_L2_5@zero','COST1_PRODUCT5@zero'),('GATE@net','COST1_PRODUCT5@net')]
    comparisons=[compare(rows,b,n) for b,n in pairs]
    running=0.
    for rank,i in enumerate(sorted(range(4),key=lambda i:comparisons[i]['sign_flip_p'])):
        running=max(running,min(1.,(4-rank)*comparisons[i]['sign_flip_p']));comparisons[i]['holm_p']=running
    result=dict(status='SIMPLE_EXPLANATIONS_COMPLETE',authority=bind(AUTH), summary=summary,
                planned_comparisons=comparisons, rows=rows, candidate_recall=570, evidence=a['evidence'], external_GO=False)
    write(OUT/'result.json',result)
    # Independent reconstruction from sealed integer actions, not stored correctness.
    recomputed={m:0 for m in summary}
    for p in ps:
        for j,q in enumerate(p['query_ids']):
            r=byid[q]; role=roles[q]
            for m, decisions in p['predictions'].items():
                k=decisions[j]; pos=r['winner'] if k==0 else r['challengers'][k-1]
                recomputed[m]+=int(gallery[r['axis'][pos]]==role['identity'])
    need(recomputed=={m:s['correct'] for m,s in summary.items()}, 'INDEPENDENT_COUNTS')
    write(OUT/'validation.json',dict(status='ALL593_SEALS_ACTIONS_COUNTS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),folds=seals,independent_counts=recomputed))
    print(dict(status=result['status'], primary=comparisons[0]),flush=True)


def publish(a):
    v=read(OUT/'validation.json');need(v['status']=='ALL593_SEALS_ACTIONS_COUNTS_PASS','VALIDATED')
    p=read(checked(v['result']))
    lines=['# H593：简单解释对照结果','',
           '原ColNomic自然C128、593张、64组件五折；召回570/593。当前为已打开总体的探索性OOF机制分析，不是新外部确认。',
           'M为原整体质量，L0为自由内容；全部臂复用相同缓存。COST1主分析，CE敏感性对照。',
           'b0/b1/b3/b5仅表示内部验证的误伤预算，绝不是外折误伤保证；net无误伤预算。zero固定零阈值。',
           '', '|模型|正确/593|救回|误伤|切换|RAW正确误伤率|','|---|---:|---:|---:|---:|---:|']
    for m,s in p['summary'].items():
        lines.append(f"|{m}|{s['correct']}|{s['rescue']}|{s['breaks']}|{s['switches']}|{s['raw_correct_loss_rate']:.4f}|")
    lines+=['','## 预定比较','', '|新模型|基线|救/损|组件等权差95%区间|Holm p|','|---|---|---:|---|---:|']
    for c in p['planned_comparisons']:
        lines.append(f"|{c['new']}|{c['baseline']}|{c['rescue']}/{c['breaks']}|{c['bootstrap95']}|{c['holm_p']:.6f}|")
    lines+=['','## 解释边界','',
            '乘积对比项是已有质量/内容的确定性变换，增益若存在是读出形式的增量，不是新增底层信息。相同参数量的平方项对照不能穷尽所有加性模型。',
            '简单门控不是ELViS或To Match or Not to Match官方复现。原始强Qwen3重排522/593结果仍保留，不能据本实验声称普遍优于重排。',
            '各臂完整参数、内层选择、逐候选分数、外折动作和来源哈希位于results/rc_h593_simple_explanations_v1。']
    REPORT.write_text('\n'.join(lines)+'\n')


def preflight():
    # Missing targets must never turn a HOLD into a correct decision.
    z=np.array([[0.,0.],[2.,2.],[-2.,-1.]])
    need(action(z).tolist()==[0,1,0], 'HOLD_AND_TIE')
    y=np.array([-1,1,0]);need(stats(action(z),y)['correct']==2,'MISSING_TARGET')
    chosen=select_thresholds(z,z,y)
    need(chosen['b0']['validation']['breaks']==0,'ZERO_BUDGET')
    need(action(z,np.inf).tolist()==[0,0,0],'ALL_HOLD')
    need(action(z,-np.inf,-np.inf,np.array([0.,1.,2.])).tolist()==[0,0,0],'GATE_HOLD')
    m=np.array([.2,.7]);l=np.array([.8,.3]);ss=sym(m*l,(m*l)[0])
    need(not np.allclose(ss,sym(m,m[0])*sym(l,l[0])),'PRODUCT_NOT_CONTRAST_PRODUCT')
    tx=torch.tensor([[.2,-.5],[.7,-.4],[-1.,-.3]],dtype=torch.float64);ty=torch.tensor([-1,0,1])
    scalar=np.mean([np.logaddexp(0,.2),np.logaddexp(0,-.7)+np.logaddexp(0,-.4),np.logaddexp(0,.3)+np.logaddexp(0,-1.)])
    need(abs(float(LOSS.unit(tx,ty))-scalar)<1e-14,'COST1_SCALAR')
    write(OUT/'preflight.json',dict(status='SYNTHETIC_PASS',authority=bind(AUTH),checks=['HOLD_tie','missing_target','budget_selection','gate','product_contrast','COST1_scalar'],natural_training_updates=0))
    print('SYNTHETIC_PASS',flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','cache','fit','verify','join','publish']);ap.add_argument('--fold',type=int)
    args=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage,args.fold)
        if args.stage=='preflight':preflight()
        elif args.stage=='cache':cache(a)
        elif args.stage=='fit':sys.exit(fit(a,args.fold))
        elif args.stage=='verify':verify(a,args.fold)
        elif args.stage=='join':join(a)
        elif args.stage=='publish':publish(a)
