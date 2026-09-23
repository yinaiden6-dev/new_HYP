#!/usr/bin/env python3
"""CPU cached-feature pilot: 71 queries, original folds, 50 matched configurations."""
import argparse
from collections import OrderedDict, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'programs'))
import rc_feature_fusion_core_v1 as F
import rc_feature_fusion_training_v1 as T

OUT = ROOT / 'results/rc_h71_feature_fusion_pilot_v1'
AUTH = ROOT / 'registry/rc_h71_feature_fusion_pilot_authority_v1_20260923.json'
PARENT = ROOT / 'registry/rc_h593_feature_fusion_train_authority_v2_20260922.json'
CACHE = ROOT / 'results/rc_h593_feature_fusion_cache_v1'
OLD = ROOT / 'results/rc_h593_feature_fusion_train_v2'
PLAN = ROOT / 'plan/RC_H71_FEATURE_FUSION_PILOT_V1_20260923.md'
LAUNCH = ROOT / 'slurm/rc_h71_feature_fusion_pilot_v1.sbatch'
FOLLOW = ROOT / 'programs/dispatch_rc_h71_feature_fusion_pilot_v1.py'
CONFIGS = [dict(seed=17, kind='COST1', fold=f, source=s, mode=m)
           for f in range(5) for s in (*F.SOURCES, 'NO_ADAPTER')
           for m in ('FREE', 'ROMA_WEIGHTED')]
STEPS = 240
SECONDS = 450


def need(x, why):
    if not x:
        raise RuntimeError(why)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(b):
    need(bind(b['path']) == b, 'SHA_DRIFT:' + b['path'])
    return Path(b['path'])


def write(path, value, mutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and not mutable:
        need(path.read_text() == text, 'IMMUTABLE:' + str(path))
        return
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def save(path, value, mutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need(mutable or not path.exists(), 'IMMUTABLE:' + str(path))
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def prepare():
    need(not AUTH.exists(), 'NEW_AUTHORITY')
    parent = read(PARENT)
    core_paths = {str(Path(F.__file__).resolve()), str(Path(T.__file__).resolve())}
    core_sources = [b for b in parent['sources'] if b['path'] in core_paths]
    need(len(core_sources) == 2, 'ORIGINAL_CORE_BINDINGS')
    for b in core_sources:
        checked(b)
    gradient = read(OLD / 'preflight.json')
    need(gradient['status'] == 'FUSION_TRAINING_GRADIENT_RESUME_PASS'
         and gradient['authority'] == bind(PARENT)
         and gradient['optimizer_resume_next_update_exact'], 'ORIGINAL_GRADIENT_QUALIFICATION')
    ready = read(CACHE / 'ready.json')
    need(ready['status'] == 'FUSION_ALL593_FEATURE_CACHE_PASS', 'READY')
    need(ready['authority'] == parent['cache_authority'], 'CACHE_LINEAGE')
    catalog = read(checked(ready['catalog']))
    panel = [r for r in catalog['queries'] if 0 <= r['execution_ordinal'] <= 70]
    panel.sort(key=lambda r: r['execution_ordinal'])
    need([r['execution_ordinal'] for r in panel] == list(range(71)), 'PANEL71')
    ids = {r['query_id'] for r in panel}
    split = read(checked(parent['public_sources']['split']))
    folds = []
    for f in split['folds']:
        folds.append(dict(fold=f['fold'],
                          train_query_ids=[q for q in f['train_query_ids'] if q in ids],
                          heldout_query_ids=[q for q in f['heldout_query_ids'] if q in ids]))
    need([len(f['heldout_query_ids']) for f in folds] == [21, 15, 15, 5, 15], 'FOLD_COUNTS')
    warm = {}
    for fold in range(5):
        for mode, offset in [('FREE', 8), ('ROMA_WEIGHTED', 9)]:
            folder = OLD / f'fit{fold * 10 + offset:03d}'
            v = read(folder / 'validation.json')
            p = read(checked(v['payload']))
            need(v['status'] == 'FUSION_FOLD_NUMPY_READOUT_PASS', 'VALID_WARM_SOURCE')
            need(v['authority'] == p['authority'] == bind(PARENT), 'WARM_AUTHORITY')
            need(p['config'] == dict(seed=17, kind='COST1', fold=fold, source='NO_ADAPTER', mode=mode), 'WARM_CONFIG')
            cp = torch.load(checked(p['checkpoint']), map_location='cpu', weights_only=True)
            warm[f'{fold}/{mode}'] = dict(theta_hex=cp['warm_head_hex'], checkpoint=p['checkpoint'],
                                         validation=bind(folder / 'validation.json'))
    initial = {}
    for r in panel:
        path = OLD / 'initial' / f"query{r['execution_ordinal']:03d}.pt"
        if path.exists():
            initial[r['query_id']] = bind(path)
    snapshot = dict(records=panel, folds=folds, warm=warm, cached_initial=initial,
                    selection='Fixed execution ordinals0..70 before pilot outcomes; opened panel',
                    warm_training='Original full TRAIN fold, not only71; no same-fold held labels')
    write(OUT / 'snapshot.json', snapshot)
    sources = [Path(__file__), Path(F.__file__), Path(T.__file__), PLAN, LAUNCH, FOLLOW, OLD / 'preflight.json',
               ROOT / 'slurm/rc_h71_feature_fusion_follow_v1.sbatch']
    authority = dict(status='H71_FUSION_PILOT_AUTHORIZED', sources=[bind(p) for p in sources],
                     parent=bind(PARENT), ready=bind(CACHE / 'ready.json'), snapshot=bind(OUT / 'snapshot.json'),
                     public_sources=parent['public_sources'], fold_sources=parent['fold_sources'],
                     join_sources=parent['join_sources'], configs=CONFIGS, steps=STEPS,
                     execution_device='cpu', new_encoder_forwards=0, new_gpu_jobs=0,
                     max_parallel=50, max_waves=32, chunk_seconds=SECONDS,
                     user_instruction='First test whether the reduced COST1 single-seed fusion design gains',
                     primary_comparison='Matched-mode COL_ONLY and NO_ADAPTER, all arms reported',
                     precision='FP64 scoring and training; FP32 encoded analysis copies',
                     evidence='Opened71-query grouped pilot with original full-TRAIN warm heads; not external GO')
    write(AUTH, authority)
    write(OUT / 'preflight.json', dict(status='H71_PILOT_READY', authority=bind(AUTH),
                                     queries=71, configurations=50, cached_initial_queries=len(initial)))
    print(json.dumps(dict(event='H71_PREPARED',queries=71,configs=50,steps=STEPS)), flush=True)


def guard(stage, index):
    a = read(AUTH)
    need(a['status'] == 'H71_FUSION_PILOT_AUTHORIZED', 'AUTHORITY')
    for b in [*a['sources'], a['ready'], a['snapshot'], a['parent']]:
        checked(b)
    need(read(OUT / 'preflight.json')['authority'] == bind(AUTH), 'PREFLIGHT')
    need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    snap = read(a['snapshot']['path'])
    allowed = {Path(b['path']).resolve() for b in a['public_sources'].values()}
    allowed |= {Path(b['path']).resolve() for b in snap['cached_initial'].values()}
    if stage in ('fit', 'verify'):
        need(index in range(len(CONFIGS)), 'CONFIG')
        allowed.add(Path(a['fold_sources'][str(CONFIGS[index]['fold'])]['train_roles']['path']).resolve())
    if stage == 'join':
        allowed |= {Path(b['path']).resolve() for b in a['join_sources'].values()}
        for fs in a['fold_sources'].values():
            allowed |= {Path(b['path']).resolve() for b in fs.values()}
    def audit(event, args):
        if event == 'socket.connect':
            need(not isinstance(args[1], tuple), 'OFFLINE')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        need(not any(t in str(p).lower() for t in ('d1-mi', 'd1_mi', 'formal392', '/grozi/', '/isic/', '/target_join/')), 'PROTECTED')
        if 'curator_roles' in str(p):
            need(stage == 'join', 'NO_HELD_LABELS_BEFORE_SEAL')
        if ROOT / 'results' in p.parents:
            own = OUT in p.parents
            if own and stage in ('fit', 'verify'):
                own = p.relative_to(OUT).parts[0] in ('preflight.json', 'snapshot.json', 'initial', f'fit{index:03d}')
            need(own or CACHE in p.parents or ROOT/'results/rc_h593_feature_fusion_pilot_v1' in p.parents or p in allowed,
                 'UNLISTED_RESULT:' + str(p))
    sys.addaudithook(audit)
    return a, snap


class Bank:
    def __init__(self, bindings, source):
        self.bindings, self.source = bindings, source
        self.cache, self.verified = OrderedDict(), set()

    def __call__(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        b = self.bindings[key]
        if key not in self.verified:
            checked(b)
            self.verified.add(key)
        p = torch.load(b['path'], map_location='cpu', weights_only=True, mmap=True)
        value = dict(original_colnomic_tokens=p['original_colnomic_tokens'],
                     projected_inputs={self.source: p['projected_inputs'][self.source]})
        self.cache[key] = value
        if len(self.cache) > 512:
            self.cache.popitem(last=False)
        return value


def target_position(record, identity, labels):
    pos = [i for i, p in enumerate(record['candidate_physical_rows']) if labels[p] == identity]
    need(len(pos) <= 1, 'DEDUP_C128')
    if not pos:
        return -2
    return -1 if pos[0] == record['winner'] else record['challenger_positions'].index(pos[0])


def load_scope(a, snap, cfg):
    ready = read(a['ready']['path'])
    f = snap['folds'][cfg['fold']]
    trainids, heldids = set(f['train_query_ids']), set(f['heldout_query_ids'])
    need(not trainids & heldids, 'FOLD_DISJOINT')
    fs = a['fold_sources'][str(cfg['fold'])]
    roles = {r['query_id']: r for r in read(checked(fs['train_roles']))['records']}
    labels = {r['physical_row']: r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    train, held, alltrain = [], [], []
    for meta in snap['records']:
        rec = torch.load(checked(meta['payload']), map_location='cpu', weights_only=True)
        need(rec['query_id'] == meta['query_id'] and len(rec['pairs']) == 128, 'QUERY_C128')
        if rec['query_id'] in trainids:
            alltrain.append(rec)
            target = target_position(rec, roles[rec['query_id']]['identity'], labels)
            if target >= -1:
                train.append((rec, target))
        else:
            need(rec['query_id'] in heldids, 'HELD_AXIS')
            held.append(rec)
    need(train and len(alltrain) + len(held) == 71, 'SPLIT_COVERAGE')
    need(not {r['source_image_sha256'] for r in alltrain} & {r['source_image_sha256'] for r in held}, 'IMAGE_DISJOINT')
    return ready, train, held


def initial_features(a, snap, rec, mode, bank):
    if mode == 'ROMA_WEIGHTED':
        c4 = torch.stack([p['native_c4'] for p in rec['pairs']])
        return F.differentiable_features(rec['candidate_raw_scores'], c4, rec['winner'])
    old = snap['cached_initial'].get(rec['query_id'])
    if old:
        p = torch.load(checked(old), map_location='cpu', weights_only=True)
        need(p['authority'] == a['parent'] and p['query_id'] == rec['query_id'], 'INITIAL_SOURCE')
        return p['X']
    path = OUT / 'initial' / f"query{rec['execution_ordinal']:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            p = torch.load(path, map_location='cpu', weights_only=True)
            need(p['authority'] == bind(AUTH), 'INITIAL_AUTHORITY')
            return p['X']
        with torch.no_grad():
            c4 = T.all_scores(rec, bank, None, 'COL_ONLY', 'FREE', 'cpu')
        x = F.differentiable_features(rec['candidate_raw_scores'], c4, rec['winner'])
        save(path, dict(authority=bind(AUTH), query_id=rec['query_id'], C4=c4, X=x, label_reads=0))
        return x


def config_order(n, seed, step):
    epoch, offset = divmod(step, n)
    gen = torch.Generator().manual_seed(seed + 1000003 * epoch)
    return int(torch.randperm(n, generator=gen)[offset])


def fit(a, snap, index):
    cfg = CONFIGS[index]
    folder = OUT / f'fit{index:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'fit.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'validation.json').exists():
        checked(read(folder / 'validation.json')['payload'])
        return
    started = time.monotonic()
    ready, train, held = load_scope(a, snap, cfg)
    source = cfg['source'] if cfg['source'] != 'NO_ADAPTER' else 'COL_ONLY'
    bank = Bank(ready['features'], source)
    adapter = None if cfg['source'] == 'NO_ADAPTER' else F.OutputAdapter(cfg['seed'])
    theta = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))
    groups = [dict(params=[theta], lr=.003)]
    if adapter is not None:
        groups.append(dict(params=adapter.parameters(), lr=.001))
    optimizer = torch.optim.AdamW(groups, weight_decay=.001)
    warm = snap['warm'][f"{cfg['fold']}/{cfg['mode']}"]
    initial = dict(authority=bind(AUTH), config=cfg, warm=warm, zero_adapter=True,
                   train_query_ids=[r['query_id'] for r, _ in train], held_query_ids=[r['query_id'] for r in held])
    write(folder / 'initial.json', initial)
    checkpoint = folder / 'checkpoint.pt'
    history, step = [], 0
    if checkpoint.exists():
        cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
        need(cp['authority'] == bind(AUTH) and cp['config'] == cfg and cp['ready'] == a['ready'], 'RESUME_BINDING')
        if adapter is not None:
            adapter.load_state_dict(cp['adapter'])
        with torch.no_grad():
            theta.copy_(cp['theta'])
        optimizer.load_state_dict(cp['optimizer'])
        step, history = cp['step'], cp['history']
    else:
        with torch.no_grad():
            theta.copy_(torch.tensor([float.fromhex(v) for v in warm['theta_hex']], dtype=torch.float64))
    def save_state():
        save(checkpoint, dict(authority=bind(AUTH), config=cfg, step=step, ready=a['ready'],
                             adapter=None if adapter is None else adapter.state_dict(), theta=theta.detach(),
                             optimizer=optimizer.state_dict(), history=history, warm=warm), mutable=True)
    save_state()
    print(dict(event='H71_FIT_RESUMED', index=index, config=cfg, step=step, train=len(train), held=len(held)), flush=True)
    while step < STEPS:
        if time.monotonic() - started > SECONDS:
            save_state()
            print(dict(event='H71_CHUNK_SAVED', index=index, step=step), flush=True)
            return
        rec, target = train[config_order(len(train), cfg['seed'], step)]
        optimizer.zero_grad(set_to_none=True)
        tick = time.monotonic()
        if adapter is None:
            x = initial_features(a, snap, rec, cfg['mode'], bank)
            loss = T.objective(x @ theta[:-1] + theta[-1], target, 'COST1')
            loss.backward()
            info = dict(loss=float(loss.detach()), active_vjp_candidates=0, candidates=128)
        else:
            info = T.two_pass_backward(rec, bank, adapter, theta, source, cfg['mode'], 'COST1', target)
        need(bool(torch.isfinite(theta.grad).all()), 'FINITE_HEAD_GRADIENT')
        if adapter is not None:
            need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in adapter.parameters()), 'FINITE_ADAPTER_GRADIENT')
        optimizer.step()
        step += 1
        history.append(dict(step=step, query_id=rec['query_id'], seconds=time.monotonic()-tick, **info))
        if step == 1 or step % 10 == 0:
            save_state()
            print(dict(event='H71_UPDATE', index=index, step=step, seconds=time.monotonic()-started, loss=info['loss']), flush=True)
    save_state()
    final = folder / 'final_model.pt'
    value = dict(authority=bind(AUTH), config=cfg, step=step, theta=theta.detach(),
                 adapter=None if adapter is None else adapter.state_dict(), warm=warm, ready=a['ready'])
    if not final.exists():
        save(final, value)
    else:
        prior = torch.load(final, map_location='cpu', weights_only=True)
        need(prior['authority'] == value['authority'] and torch.equal(prior['theta'], value['theta']), 'FINAL_IMMUTABLE')
        if adapter is not None:
            need(all(torch.equal(v, prior['adapter'][k]) for k, v in adapter.state_dict().items()), 'FINAL_ADAPTER_IMMUTABLE')
    model = bind(final)
    def store_encoded(key):
        dest = folder / 'encoded' / f'{key}.pt'
        if dest.exists():
            return
        with torch.no_grad():
            z, g = T.encode(adapter, bank(key), source, 'cpu')
        save(dest, dict(source_features=ready['features'][key], model_checkpoint=model,
                        adapted_tokens=z.float(), gate=g.float(), dtype='FP32 analysis copy; scoring FP64',
                        exact_reconstruction='Original bound tokens, projected input, FP64 final model and sealed code'))
    for rec in held:
        dest = folder / 'predictions' / f"query{rec['execution_ordinal']:03d}.pt"
        if dest.exists():
            need(torch.load(dest, map_location='cpu', weights_only=True)['model_checkpoint'] == model, 'PREDICTION_RESUME')
            continue
        if time.monotonic() - started > SECONDS:
            return
        store_encoded(rec['query_image_key'])
        pairs = []
        with torch.no_grad():
            for pos, pair in enumerate(rec['pairs']):
                store_encoded(pair['image_key'])
                scores, trace = T.pair_forward(rec, pos, bank, adapter, source, cfg['mode'], 'cpu')
                pairs.append(dict(position=pos, physical_row=pair['physical_row'], image_key=pair['image_key'], c4=scores, trace=trace))
            c4 = torch.stack([p['c4'] for p in pairs])
            x = F.differentiable_features(rec['candidate_raw_scores'], c4, rec['winner'])
            z = x @ theta[:-1] + theta[-1]
            k = int(z.argmax())
            pos = rec['challenger_positions'][k] if float(z[k]) > 0 else rec['winner']
        save(dest, dict(authority=bind(AUTH), config=cfg, query_id=rec['query_id'], execution_ordinal=rec['execution_ordinal'],
                        model_checkpoint=model, candidate_physical_rows=rec['candidate_physical_rows'],
                        winner=rec['winner'], challenger_positions=rec['challenger_positions'], candidate_raw_scores=rec['candidate_raw_scores'],
                        pairs=pairs, X=x, logits=z, selected=rec['candidate_physical_rows'][pos], heldout_label_reads=0))
        print(dict(event='H71_PREDICTED', index=index, ordinal=rec['execution_ordinal']), flush=True)
    write(folder / 'payload.json', dict(status='H71_FOLD_PREDICTIONS_SEALED', authority=bind(AUTH), config=cfg,
                                       checkpoint=model, steps=STEPS, train_query_ids=[r['query_id'] for r, _ in train],
                                       held_query_ids=[r['query_id'] for r in held],
                                       predictions=[bind(folder/'predictions'/f"query{r['execution_ordinal']:03d}.pt") for r in held],
                                       heldout_label_reads=0))
    subprocess.run([sys.executable, __file__, 'verify', '--index', str(index)], check=True)


def verify(a, snap, index):
    folder = OUT / f'fit{index:03d}'
    p = read(folder / 'payload.json')
    need(p['authority'] == bind(AUTH) and p['config'] == CONFIGS[index] and p['steps'] == STEPS, 'FIT_SEAL')
    expected = set(snap['folds'][CONFIGS[index]['fold']]['heldout_query_ids'])
    cp = torch.load(checked(p['checkpoint']), map_location='cpu', weights_only=True)
    theta = cp['theta'].numpy()
    maximum, checks, seen = 0., 0, set()
    for b in p['predictions']:
        v = torch.load(checked(b), map_location='cpu', weights_only=True)
        need(v['authority'] == bind(AUTH) and v['model_checkpoint'] == p['checkpoint'] and v['heldout_label_reads'] == 0, 'PRED_SOURCE')
        need(v['query_id'] not in seen, 'UNIQUE_PRED')
        seen.add(v['query_id'])
        x, expected_logits = v['X'].numpy(), v['logits'].numpy()
        z = (x * theta[:-1]).sum(1) + theta[-1]
        err = float(np.max(np.abs(z - expected_logits)))
        need(err < 2e-10 and x.shape == (127, 6), 'NUMPY127_LOGITS')
        maximum, checks = max(maximum, err), checks + 127
        k = int(z.argmax())
        pos = v['challenger_positions'][k] if z[k] > 0 else v['winner']
        need(v['candidate_physical_rows'][pos] == v['selected'], 'NUMPY_ACTION')
        scores = torch.stack([pair['c4'] for pair in v['pairs']])
        literal = F.differentiable_features(v['candidate_raw_scores'], scores, v['winner'])
        need(scores.shape == (128, 4) and torch.allclose(literal, v['X'], atol=2e-12, rtol=2e-12), 'C128_FEATURES')
        for pair in v['pairs']:
            need(abs(float(pair['trace']['query_token_score_contribution'].sum()) - float(pair['c4'][0])) < 2e-10, 'TRACE_SUM')
    need(seen == expected, 'ALL_EXPECTED_HELD_QUERIES')
    write(folder / 'validation.json', dict(status='H71_FOLD_NUMPY_PASS', authority=bind(AUTH),
                                          payload=bind(folder / 'payload.json'), logit_checks=checks, max_numpy_error=maximum,
                                          scope='NumPy127-logit/action recount; C128 feature and contribution sums; original gradient core reused'))
    print(dict(event='H71_FIT_VALIDATED', index=index, queries=len(seen)), flush=True)


def compare(rows, base, name):
    groups = defaultdict(list)
    for row in rows:
        groups[row['component']].append(int(row['correct'][name]) - int(row['correct'][base]))
    d = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(20260923)
    interval = np.quantile(d[rng.integers(0, len(d), size=(20000, len(d)))].mean(1), [.025, .975])
    loo = [(d.sum() - v)/(len(d)-1) for v in d]
    rescue = sum(r['correct'][name] and not r['correct'][base] for r in rows)
    breaks = sum(r['correct'][base] and not r['correct'][name] for r in rows)
    return dict(base=base, arm=name, rescue=rescue, breaks=breaks, net=rescue-breaks, groups=len(d),
                group_mean=float(d.mean()), group_bootstrap95=interval.tolist(), leave_one_group_out_range=[min(loo), max(loo)],
                fold_net={str(f):sum(int(r['correct'][name])-int(r['correct'][base]) for r in rows if r['fold']==f) for f in range(5)})


def join(a, snap):
    fits, validations = [], []
    for i, cfg in enumerate(CONFIGS):
        path = OUT / f'fit{i:03d}/validation.json'
        v = read(path)
        p = read(checked(v['payload']))
        need(v['status']=='H71_FOLD_NUMPY_PASS' and v['authority']==p['authority']==bind(AUTH) and p['config']==cfg, 'ALL50_VALIDATED')
        fits.append(p)
        validations.append(bind(path))
    write(OUT / 'all_predictions_prelabel_seal.json', dict(authority=bind(AUTH), validations=validations))
    roles = {r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']}
    labels = {r['physical_row']:r['identity'] for r in read(checked(a['public_sources']['gallery']))['records']}
    rows, model_names = {}, []
    for p in fits:
        cfg = p['config']
        name = cfg['source'] + '/' + cfg['mode']
        if name not in model_names:
            model_names.append(name)
        fs = a['fold_sources'][str(cfg['fold'])]
        tr = read(checked(fs['train_roles']))['records']
        trainids, components = {r['identity'] for r in tr}, {r['component'] for r in tr}
        old = {r['query_id']:r for r in read(checked(fs['full_payload']))['predictions']}
        for b in p['predictions']:
            pred = torch.load(checked(b), map_location='cpu', weights_only=True)
            qid = pred['query_id']
            role = roles[qid]
            need(role['identity'] not in trainids and role['component'] not in components and role['outer_fold']==cfg['fold'], 'IDENTITY_COMPONENT_DISJOINT')
            row = rows.setdefault(qid,dict(query_id=qid,component=role['component'],fold=cfg['fold'],correct={},selected={},mrr={},margin={}))
            axis = pred['candidate_physical_rows']
            if not row['correct']:
                raw = axis[pred['winner']]
                row['selected']['RAW'] = raw
                row['correct']['RAW'] = labels[raw] == role['identity']
                oldsel = old[qid]['models']['COST1']['selected']
                row['selected']['ORIGINAL_COST1'] = oldsel
                row['correct']['ORIGINAL_COST1'] = labels[oldsel] == role['identity']
            row['selected'][name] = pred['selected']
            row['correct'][name] = labels[pred['selected']] == role['identity']
            logits = np.zeros(128)
            logits[pred['challenger_positions']] = pred['logits'].numpy()
            order = sorted(range(128), key=lambda j:(-logits[j],j!=pred['winner'],axis[j]))
            target = [j for j,pid in enumerate(axis) if labels[pid]==role['identity']]
            need(len(target)<=1, 'UNIQUE_IDENTITY')
            row['target_in_C128'] = bool(target)
            row['mrr'][name] = 0. if not target else 1./(order.index(target[0])+1)
            row['margin'][name] = None if not target else float(logits[target[0]]-max(logits[j] for j in range(128) if j!=target[0]))
    rows = sorted(rows.values(), key=lambda r:r['query_id'])
    need({r['query_id'] for r in rows}=={r['query_id'] for r in snap['records']}, 'PANEL_COVERAGE')
    need(len({r['component'] for r in rows})==37 and len(rows)==71, 'PANEL71_GROUP37')
    names = ['RAW','ORIGINAL_COST1',*model_names]
    counts = {m:sum(r['correct'][m] for r in rows) for m in names}
    comparisons = {}
    for name in model_names:
        source, mode = name.split('/')
        bases = ['RAW','ORIGINAL_COST1']
        if source!='NO_ADAPTER': bases.append('NO_ADAPTER/'+mode)
        if source not in ('NO_ADAPTER','COL_ONLY'): bases.append('COL_ONLY/'+mode)
        if mode=='ROMA_WEIGHTED': bases.append(source+'/FREE')
        for base in bases:
            comparisons[base+'__to__'+name] = compare(rows,base,name)
    result = dict(status='H71_FUSION_PILOT_COMPLETE', authority=bind(AUTH), queries=71, groups=37,
                  target_absent=sum(not r['target_in_C128'] for r in rows), counts=counts, comparisons=comparisons,
                  MRR={m:float(np.mean([r['mrr'][m] for r in rows])) for m in model_names},rows=rows,
                  evidence=a['evidence'], limitations='Single-seed reduced240-step opened pilot; no winner selection, no external GO; no gain does not prove absent information')
    write(OUT/'result.json',result)
    write(OUT/'validation.json',dict(status='H71_JOIN_COUNTS_AND_GROUPS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),fits=50,queries=71,groups=37))
    lines=['# 71张融合收益先导','', '71 queries / 37 groups; COST1 seed17; 原五折; 同折原全量TRAIN初始化后，在71张子集微调240步。','', '| 方法 | 正确/71 | MRR |','|---|---:|---:|']
    for name in names:
        lines.append(f"| {name} | {counts[name]} | {result['MRR'].get(name,'—')} |")
    lines += ['', '全部配对比较、救回/损失、每折净增、分组区间、逐query分差见result.json；所有预测保存完整C128中间量。区间为描述性边际区间，未据此选择最佳臂，不能将多臂中最好的一个当成独立确认。', '', a['evidence']]
    write(OUT/'report.json',dict(markdown='\n'.join(lines)+'\n'))
    (OUT/'summary_zh.md').write_text('\n'.join(lines)+'\n')
    print(dict(event='H71_COMPLETE',counts=counts),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('prepare','fit','verify','join'))
    ap.add_argument('--index',type=int)
    args=ap.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if args.stage=='prepare':
        prepare()
        return
    a,snap=guard(args.stage,args.index)
    if args.stage=='fit':
        fit(a,snap,args.index)
        folder=OUT/f'fit{args.index:03d}'
        cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        root=os.environ.get('SLURM_ARRAY_JOB_ID',os.environ['SLURM_JOB_ID'])
        write(folder/'chunks'/f'{root}_{args.index}.json',dict(status='H71_CHUNK_NORMAL_EXIT',authority=bind(AUTH),
              index=args.index,step=cp['step'],job_id=os.environ['SLURM_JOB_ID'],array_job=root,
              complete=(folder/'validation.json').exists(),checkpoint=bind(folder/'checkpoint.pt')))
    elif args.stage=='verify':verify(a,snap,args.index)
    else:join(a,snap)


if __name__=='__main__':main()
