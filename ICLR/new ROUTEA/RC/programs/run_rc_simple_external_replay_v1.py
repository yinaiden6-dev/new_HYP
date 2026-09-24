#!/usr/bin/env python3
"""Frozen simplified H593 heads on existing external tokens; CPU only.

Preflight verifies all source shards before any replay. Every external prediction
is sealed before either dataset's query identities are opened. No fitting occurs
in this program. Query checkpoints allow short Slurm jobs to resume safely.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature

OUT = ROOT / 'results/rc_simple_external_transfer_v1'
AUTH = ROOT / 'registry/rc_simple_external_transfer_authority_v1_20260924.json'
BUNDLE = OUT / 'source/bundle.json'
MODELS = ('MASS5', 'ADDITIVE4', 'NATIVE7')
DATA = {
    'grozi': dict(root=ROOT / 'results/rc_new_hyp_grozi120_external_v1',
        auth=ROOT / 'registry/rc_new_hyp_grozi120_inference_authority_v1_20260913.json',
        tasks=60, queries=480, groups=27, prefix='GROZI'),
    'isic': dict(root=ROOT / 'results/rc_new_hyp_isic_transfer_v1',
        auth=ROOT / 'registry/rc_new_hyp_isic_inference_authority_v1_20260914.json',
        tasks=68, queries=537, groups=346, prefix='ISIC'),
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).absolute()), sha256=sha(path))


def checked(binding):
    path = Path(binding['path'])
    need(sha(path) == binding['sha256'], 'SOURCE_DRIFT:' + str(path))
    return path


def write(path, value, replace=False):
    """Atomic publication; authoritative JSON files are immutable and resumable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        need(read(path) == value, 'IMMUTABLE_OUTPUT:' + str(path))
        return
    tmp = path.parent / ('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    if replace:
        os.replace(tmp, path)
    else:
        try:
            os.link(tmp, path)
        except FileExistsError:
            need(read(path) == value, 'CONCURRENT_OUTPUT_MISMATCH')
        finally:
            tmp.unlink()


def tensor_sha(value):
    t = value.detach().cpu().contiguous()
    return hashlib.sha256(str(t.dtype).encode('ascii') +
        json.dumps(list(t.shape), separators=(',', ':')).encode('ascii') +
        t.view(torch.uint8).numpy().tobytes()).hexdigest()


def task_info(task):
    need(task is not None and 0 <= task < 128, 'TASK_RANGE_0_127')
    name, shard = ('grozi', task) if task < 60 else ('isic', task - 60)
    return name, shard, DATA[name]


def guard(stage):
    need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    a = read(AUTH)
    need(a['status'] == 'SIMPLE_EXTERNAL_TRANSFER_AUTHORIZED', 'AUTHORITY')
    for b in a['code_sources'].values():
        checked(b)
    def bindings(value):
        if isinstance(value, dict):
            if set(('path', 'sha256')) <= set(value):
                yield value
            else:
                for v in value.values(): yield from bindings(v)
        elif isinstance(value, list):
            for v in value: yield from bindings(v)
    external = {str(Path(b['path']).absolute()): b for b in bindings(a['external_inputs'])}
    for d in DATA.values():
        need(str(d['auth']) in external, 'EXTERNAL_AUTHORITY_NOT_PINNED')
        checked(external[str(d['auth'])])
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p).lower()
        need(not any(x in s for x in ('d1-mi', 'd1_mi', 'formal392')), 'PROTECTED_DATA')
        if 'curator_roles' in s:
            need(stage in ('join', 'verify') and (OUT / 'all_predictions_prelabel_seal.json').exists(),
                 'NO_EXTERNAL_LABELS_BEFORE_ALL_SEALS')
        if p.name in ('result.json', 'result_validation.json') and any(d['root'] in p.parents for d in DATA.values()):
            need(stage in ('join', 'verify'), 'NO_OLD_RESULTS_DURING_PREDICTION')
    sys.addaudithook(audit)
    return a


def source_envelope(name, shard, load=True):
    d = DATA[name]
    oldab = bind(d['auth'])
    a = read(d['auth'])
    loaded, sources = {}, {}
    for kind in ('raw', 'roma'):
        folder = d['root'] / kind / f'shard{shard:02d}'
        receipt = read(folder / 'receipt.json')
        validation = read(folder / 'validation.json')
        need(receipt['authority'] == validation['authority'] == oldab, 'OLD_AUTHORITY')
        need(receipt['payload'] == validation['payload'] and
             validation['status'] == d['prefix'] + '_' + kind.upper() + '_CPU_PASS', 'OLD_VALIDATION')
        p = checked(receipt['payload'])
        sources[kind] = dict(payload=receipt['payload'], receipt=bind(folder / 'receipt.json'),
                             validation=bind(folder / 'validation.json'))
        if load:
            loaded[kind] = torch.load(p, map_location='cpu', mmap=True, weights_only=True)
    wp = checked(a['sources']['worker'])
    workers = read(wp)['records'][8 * shard:8 * (shard + 1)]
    sources.update(old_authority=oldab, worker=a['sources']['worker'])
    return loaded, sources, workers


def axes_and_integrity(name, shard, raw, roma, sources, workers):
    d = DATA[name]
    n = 1 if name == 'isic' and shard == 67 else 8
    need(len(raw['records']) == len(roma['records']) == len(workers) == n, 'SHARD_QUERY_COUNT')
    need(raw['authority'] == roma['authority'] == sources['old_authority'] and
         roma['raw'] == sources['raw']['payload'], 'RAW_ROMA_BINDING')
    reference_checked = set()
    axes = []
    for q, r, w in zip(raw['records'], roma['records'], workers):
        need(q['query_id'] == r['query_id'] == w['query_id'], 'QUERY_AXIS')
        need(q['query_source_sha256'] == r['query_source_sha256'] and
             tensor_sha(q['query_tokens']) == q['query_tokens_sha256'] == r['query_tokens_sha256'], 'QUERY_HASH')
        axis = q['candidate_physical_rows']; order = q['raw_ranked_physical_rows']
        need(len(axis) == len(set(axis)) == 128 and axis == sorted(order[:128]) and
             order[:128] == q['candidate_ranked_physical_rows'] and axis == r['candidate_physical_rows'], 'NATURAL_C128')
        need(torch.equal(q['candidate_raw_scores'], q['raw_physical_scores'][axis]), 'RAW_SCORE_AXIS')
        need(len(r['candidates']) == 128, 'ALL128_CANDIDATES')
        for pos, c in enumerate(r['candidates']):
            physical = axis[pos]; ref = raw['references'][physical]
            need(c['candidate_position'] == pos and c['physical_row'] == physical == ref['physical_row'] and
                 c['reference_tokens_sha256'] == ref['tokens_sha256'], 'REFERENCE_AXIS')
            if physical not in reference_checked:
                need(tensor_sha(ref['tokens']) == ref['tokens_sha256'], 'REFERENCE_TOKEN_HASH')
                reference_checked.add(physical)
            for side, count in (('query', len(q['query_tokens'])), ('reference', len(ref['tokens']))):
                value = c[side + '_visibility']
                need(value.dtype == torch.float64 and value.shape == (count,) and torch.isfinite(value).all() and
                     bool(((value >= 0) & (value <= 1)).all()), 'VISIBILITY_DOMAIN')
                digest = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                need(digest == c[side + '_map_sha256'], 'VISIBILITY_HASH')
            mass = float(torch.sqrt(c['query_visibility'].mean() * c['reference_visibility'].mean()))
            need(mass.hex() == float(c['old_scores']['visibility_mass']).hex(), 'MASS_EXACT')
        axes.append(dict(query_id=q['query_id'], source_image_sha256=q['query_source_sha256'],
            axis=axis, raw_winner=order[0], query_tokens_sha256=q['query_tokens_sha256']))
    return axes


def preflight(task):
    name, shard, _ = task_info(task)
    path = OUT / 'preflight' / f'task{task:03d}.json'
    if path.exists():
        old = read(path)
        need(old['authority'] == bind(AUTH) and old['status'] == 'EXTERNAL_CACHE_SHARD_PASS', 'PREFLIGHT_RESUME')
        for kind in ('raw', 'roma'):
            for b in old['sources'][kind].values(): checked(b)
        return old
    loaded, sources, workers = source_envelope(name, shard)
    axes = axes_and_integrity(name, shard, loaded['raw'], loaded['roma'], sources, workers)
    p = dict(status='EXTERNAL_CACHE_SHARD_PASS', authority=bind(AUTH), task=task, dataset=name,
        shard=shard, sources=sources, axes=axes, target_reads=0, new_gpu_forwards=0)
    write(path, p)
    print(json.dumps(dict(event='CACHE_PREFLIGHT_PASS', task=task, queries=len(axes))), flush=True)
    return p


def preflight_join():
    seals, ids = [], []
    for task in range(128):
        p = OUT / 'preflight' / f'task{task:03d}.json'; value = read(p)
        need(value['status'] == 'EXTERNAL_CACHE_SHARD_PASS' and value['authority'] == bind(AUTH), 'ALL128_PREFLIGHT')
        ids.extend(x['query_id'] for x in value['axes']); seals.append(bind(p))
    need(len(ids) == len(set(ids)) == 1017, 'ALL1017_UNIQUE')
    # Gallery identities are reference metadata, not external query targets.
    sa = read(ROOT / 'registry/rc_h593_simple_explanations_authority_v1_20260924.json')
    gallery = sa['public']['gallery']; checked(gallery)
    write(OUT / 'preflight.json', dict(status='ALL_EXTERNAL_CACHES_PASS', authority=bind(AUTH),
        tasks=128, queries=1017, validations=seals, legacy_gallery=gallery, labels_opened=0))


def heads():
    p = read(BUNDLE)
    need(p['status'] == 'SIMPLE_SOURCE_HEADS_FROZEN_PASS' and p['primary'] == 'MASS5' and
         p['external_queries_read'] == 0 and set(p['heads']) == set(MODELS) and
         p['authority'] == bind(AUTH) and p['native_theta_and_source_logits_exact_legacy_parity'], 'FROZEN_SOURCE_BUNDLE')
    checked(p['program'])
    checked(p['input_manifest'])
    values = {}
    for model, bindings in p['heads'].items():
        v = read(checked(bindings['validation']))
        h = read(checked(bindings['head']))
        need(v['status'] == 'SOURCE_HEAD_NUMPY_ACTIONS_PASS' and v['head'] == bindings['head'] and
             v['authority'] == h['authority'] == p['authority'] and
             v['input_manifest'] == h['input_manifest'] == p['input_manifest'], 'SOURCE_HEAD_VALIDATION')
        need(h['model'] == model and h['program'] == p['program'] and h['optimizer'] == p['protocol'] and
             h['loss'] == 'COST1' and h['external_queries_read'] == v['external_queries_read'] == 0 and
             h['inference']['threshold'] == 0 and h['inference']['challengers'] == 127, 'SOURCE_HEAD_PROTOCOL')
        values[model] = torch.tensor([float.fromhex(v) for v in h['theta_hex']], dtype=torch.float64)
    need([len(values[m]) for m in MODELS] == [5, 4, 7], 'HEAD_DIMENSIONS')
    return values, bind(BUNDLE)


def free_pair(qnorm, reference, candidate):
    rnorm = F.normalize(reference.to(torch.float64), dim=1)
    free = (qnorm @ rnorm.T).max(dim=1).values
    total = free.sum(); n = len(free)
    mass = torch.sqrt(candidate['query_visibility'].mean() * candidate['reference_visibility'].mean())
    # Preserve M1Q0R0 operator order; this is not old weighted real_score/M.
    s = mass * total / n
    l0 = total / n
    return dict(mass=float(mass), free_content=float(l0), score=float(s),
        feature_content=float(s) / max(float(mass), 1e-12))


def prediction(q, r, refs, hs, source_bindings, bundle_binding, name, task, ordinal):
    folder = OUT / name / 'queries' / q['query_id']
    payload_path = folder / 'payload.json'; valid_path = folder / 'validation.json'
    if valid_path.exists():
        v = read(valid_path); checked(v['payload'])
        need(v['authority'] == bind(AUTH) and v['bundle'] == bundle_binding and
             v['sources'] == source_bindings, 'QUERY_RESUME')
        return bind(valid_path)
    checkpoint = folder / 'partial.json'
    pairs = []
    if checkpoint.exists():
        cp = read(checkpoint)
        need(cp['authority'] == bind(AUTH) and cp['sources'] == source_bindings and
             cp['query_id'] == q['query_id'], 'PARTIAL_SOURCE')
        pairs = cp['pairs']
    axis = q['candidate_physical_rows']
    qnorm = F.normalize(q['query_tokens'].to(torch.float64), dim=1)
    start = time.monotonic()
    for pos in range(len(pairs), 128):
        c = r['candidates'][pos]
        one = free_pair(qnorm, refs[axis[pos]]['tokens'], c)
        if pos == 0:
            # Independent FP64 NumPy oracle at one preregistered candidate per query.
            nq = q['query_tokens'].numpy().astype(np.float64); nr = refs[axis[pos]]['tokens'].numpy().astype(np.float64)
            nq /= np.maximum(np.linalg.norm(nq, axis=1, keepdims=True), 1e-12)
            nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-12)
            oracle = float(np.max(nq @ nr.T, axis=1).mean())
            one['numpy_free_content_error'] = abs(one['free_content'] - oracle)
            need(one['numpy_free_content_error'] < 2e-10, 'NUMPY_FREE_MAXSIM')
        pairs.append(one)
        if len(pairs) % 16 == 0:
            write(checkpoint, dict(authority=bind(AUTH), sources=source_bindings,
                  query_id=q['query_id'], pairs=pairs), replace=True)
    raw = list(map(float, q['candidate_raw_scores']))
    winner = axis.index(q['raw_ranked_physical_rows'][0]); challengers = [i for i in range(128) if i != winner]
    evidence = {i: dict(real_score=p['score'], visibility_mass=p['mass'],
        query_control_score=p['score'], reference_control_score=p['score']) for i, p in enumerate(pairs)}
    sx = torch.stack([candidate_feature(raw, evidence, c, winner) for c in challengers])[:, :4]
    nx = torch.stack([candidate_feature(raw, {i: c['old_scores'] for i, c in enumerate(r['candidates'])}, c, winner)
                      for c in challengers])
    features = {'MASS5': sx, 'ADDITIVE4': sx[:, [0, 2, 3]], 'NATIVE7': nx}
    models = {}; maxerr = 0.
    for model in MODELS:
        x = features[model]; theta = hs[model]
        z = x @ theta[:-1] + theta[-1]
        independent = np.sum(x.numpy() * theta[:-1].numpy(), axis=1) + float(theta[-1])
        error = float(np.max(np.abs(z.numpy() - independent))); maxerr = max(maxerr, error)
        need(error < 2e-10 and np.isfinite(independent).all(), 'NUMPY_HEAD_LOGITS')
        k = int(z.argmax()); j = int(np.argmax(independent))
        selected = axis[challengers[k]] if float(z[k]) > 0 else axis[winner]
        check = axis[challengers[j]] if independent[j] > 0 else axis[winner]
        need(selected == check, 'NUMPY_HEAD_ACTION')
        models[model] = dict(selected=selected, logits_hex=[float(v).hex() for v in z])
    p = dict(authority=bind(AUTH), bundle=bundle_binding, sources=source_bindings, dataset=name,
        task=task, ordinal=ordinal, query_id=q['query_id'], source_image_sha256=q['query_source_sha256'],
        axis=axis, raw_ranked_physical_rows=q['raw_ranked_physical_rows'], raw_scores=raw,
        winner=winner, challenger_positions=challengers, raw_selected=axis[winner],
        pairs=pairs, simplified_X=sx.tolist(), native_X=nx.tolist(), models=models,
        target_reads=0, training_updates=0, new_encoder_forwards=0, new_RoMa_forwards=0)
    write(payload_path, p)
    write(valid_path, dict(status='EXTERNAL_SIMPLE_QUERY_NUMPY_PASS', authority=bind(AUTH),
        bundle=bundle_binding, sources=source_bindings, payload=bind(payload_path), max_logit_error=maxerr,
        independent_free_maxsim_pairs=1, zero_mass_pairs=sum(v['mass'] == 0 for v in pairs),
        clamped_mass_pairs=sum(v['mass'] < 1e-12 for v in pairs), target_reads=0))
    print(json.dumps(dict(event='EXTERNAL_QUERY_SEALED', query_id=q['query_id'], task=task,
        seconds=time.monotonic()-start)), flush=True)
    return bind(valid_path)


def replay(task):
    name, shard, d = task_info(task)
    pf = read(OUT / 'preflight.json')
    need(pf['status'] == 'ALL_EXTERNAL_CACHES_PASS' and pf['authority'] == bind(AUTH), 'ALL_CACHES_PREFLIGHT_FIRST')
    hs, bb = heads()
    path = OUT / name / f'shard{shard:02d}/validation.json'
    if path.exists():
        v = read(path)
        need(v['authority'] == bind(AUTH) and v['bundle'] == bb and v['status'] == 'EXTERNAL_SIMPLE_SHARD_PASS', 'SHARD_RESUME')
        for b in v['queries']: checked(b)
        return
    oldpf = read(checked(pf['validations'][task]))
    loaded, sources, workers = source_envelope(name, shard)
    need(sources == oldpf['sources'], 'PREFLIGHT_SOURCE_PIN')
    axes = axes_and_integrity(name, shard, loaded['raw'], loaded['roma'], sources, workers)
    need(axes == oldpf['axes'], 'PREFLIGHT_AXIS_PIN')
    seals = []
    for i, (q, r) in enumerate(zip(loaded['raw']['records'], loaded['roma']['records'])):
        seals.append(prediction(q, r, loaded['raw']['references'], hs, sources, bb, name, task, 8*shard+i))
    # Exact legacy head/action replay is outcome-blind and uses sealed old logits.
    oldval = read(d['root'] / 'predictions' / f'shard{shard:02d}/validation.json')
    need(oldval['authority'] == sources['old_authority'] and
         oldval['status'] == d['prefix'] + '_FROZEN_HEAD_NUMPY_ACTION_PASS', 'LEGACY_PREDICTION_VALIDATION')
    oldpayload = read(checked(oldval['payload']))
    need(oldpayload['authority'] == sources['old_authority'] and oldpayload['raw'] == sources['raw']['payload'] and
         oldpayload['roma'] == sources['roma']['payload'], 'LEGACY_PREDICTION_INPUT_BINDING')
    oldrecords = oldpayload['records']
    need(len(oldrecords) == len(seals) and oldpayload['target_reads'] == 0, 'LEGACY_PREDICTION_COUNT')
    maximum = 0.
    for b, old in zip(seals, oldrecords):
        new = read(checked(read(checked(b))['payload']))
        need(new['query_id'] == old['query_id'] and new['raw_selected'] == old['raw_selected'], 'OLD_PREDICTION_AXIS')
        chosen = new['models']['NATIVE7']; reference = old['models']['COST1']
        need(chosen['selected'] == reference['selected'], 'NATIVE_OLD_ACTION_EXACT')
        nz = np.array([float.fromhex(x) for x in chosen['logits_hex']])
        oz = np.array([float.fromhex(x) for x in reference['logits_hex']])
        err = float(np.max(np.abs(nz-oz))); maximum = max(maximum, err)
        need(err < 2e-10, 'NATIVE_OLD_LOGITS')
    write(path, dict(status='EXTERNAL_SIMPLE_SHARD_PASS', authority=bind(AUTH), bundle=bb,
        dataset=name, shard=shard, task=task, queries=seals, sources=sources,
        legacy_predictions=oldval['payload'], native_old_max_error=maximum, target_reads=0))


def all_predictions():
    hs, bb = heads()
    records = collections.defaultdict(list); validations = []
    for task in range(128):
        name, shard, _ = task_info(task)
        p = OUT / name / f'shard{shard:02d}/validation.json'; v = read(p)
        need(v['status'] == 'EXTERNAL_SIMPLE_SHARD_PASS' and v['authority'] == bind(AUTH) and v['bundle'] == bb,
             'ALL_PREDICTIONS_QUALIFIED')
        validations.append(bind(p))
        for b in v['queries']:
            qv = read(checked(b)); q = read(checked(qv['payload']))
            need(q['target_reads'] == q['training_updates'] == 0 and q['bundle'] == bb, 'PREDICTION_BOUNDARY')
            records[name].append(q)
    need(sum(map(len, records.values())) == 1017, 'ALL1017_PREDICTIONS')
    for name, values in records.items():
        d = DATA[name]; a = read(d['auth']); workers = read(checked(a['sources']['worker']))['records']
        need([q['query_id'] for q in values] == [q['query_id'] for q in workers], 'ALL_DATASET_QUERY_ORDER')
    seal = dict(status='ALL1017_PREDICTIONS_SEALED', authority=bind(AUTH), bundle=bb,
        validations=validations, target_reads=0, primary='MASS5', secondary='ADDITIVE4')
    write(OUT / 'all_predictions_prelabel_seal.json', seal)
    return records, hs


def compare(rows, baseline, model):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['correct'][model]) - int(r['correct'][baseline]))
    delta = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(20260924)
    boot = np.concatenate([delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1)
                           for _ in range(50)])
    ci = np.quantile(boot, [.025, .975]).tolist()
    rescue = sum(not r['correct'][baseline] and r['correct'][model] for r in rows)
    loss = sum(r['correct'][baseline] and not r['correct'][model] for r in rows)
    return dict(rescue=rescue, loss=loss, net=rescue-loss, query_difference=(rescue-loss)/len(rows),
        group_count=len(groups), equal_group_difference=float(delta.mean()), group_bootstrap95=ci,
        positive_group_interval=bool(ci[0] > 0), changed_choices=sum(r['selected'][model] != r['selected'][baseline] for r in rows))


def joined_result(name, predictions, hs):
    d = DATA[name]; olda = read(d['auth'])
    roles = {r['query_id']: r for r in read(checked(olda['curator_after_all_seals']))['records']}
    if name == 'grozi':
        gallery = read(checked(read(OUT / 'preflight.json')['legacy_gallery']))['records']
        need([g['physical_row'] for g in gallery] == list(range(5413)), 'LEGACY_GALLERY_AXIS')
        appended = read(checked(olda['sources']['gallery_append']))['records']
        need([g['physical_row'] for g in appended] == list(range(5413, 5533)), 'GROZI_APPENDED_AXIS')
        labels = [g['identity'] for g in gallery] + [g['identity'] for g in appended]
    else:
        gallery = read(checked(olda['sources']['gallery']))['records']
        need([g['physical_row'] for g in gallery] == list(range(390)), 'ISIC_GALLERY_AXIS')
        labels = [g['identity'] for g in gallery]
    rows = []; independent = collections.Counter()
    for q in predictions:
        role = roles[q['query_id']]; order = q['raw_ranked_physical_rows']; target = role['identity']
        rank = next(i+1 for i, p in enumerate(order) if labels[p] == target)
        chosen = {'RAW': q['raw_selected']}
        for model in MODELS:
            p = q['models'][model]; z = np.array([float.fromhex(v) for v in p['logits_hex']])
            x = np.array(q['native_X'] if model == 'NATIVE7' else q['simplified_X'])
            if model == 'ADDITIVE4': x = x[:, [0, 2, 3]]
            theta = hs[model].numpy(); nz = np.sum(x*theta[:-1], axis=1)+theta[-1]
            need(np.max(np.abs(z-nz)) < 2e-10, 'JOIN_NUMPY_LOGITS')
            k = int(np.argmax(nz)); physical = q['axis'][q['challenger_positions'][k]] if nz[k] > 0 else q['raw_selected']
            need(physical == p['selected'], 'JOIN_ACTION')
            chosen[model] = physical
        correct = {m: labels[v] == target for m, v in chosen.items()}
        ranks = {m: next(i+1 for i, p in enumerate([v]+[p for p in order if p != v]) if labels[p] == target)
                 for m, v in chosen.items()}
        for m, v in chosen.items(): independent[m] += labels[v] == target
        rows.append(dict(query_id=q['query_id'], identity=target, component=role['component'],
            target_in_C128=rank<=128, correct=correct, ranks=ranks, selected=chosen))
    need(len(rows) == d['queries'] and len({r['component'] for r in rows}) == d['groups'], 'COHORT_DENOMINATORS')
    counts = {m: sum(r['correct'][m] for r in rows) for m in ('RAW',)+MODELS}
    need(dict(independent) == counts, 'INDEPENDENT_COUNTS')
    oldv = read(d['root'] / 'result_validation.json'); checked(oldv['result'])
    oldresult = read(oldv['result']['path']); oldrows = {r['query_id']: r for r in oldresult['rows']}
    need(counts['RAW'] == oldresult['counts']['RAW'] and counts['NATIVE7'] == oldresult['counts']['COST1'], 'LEGACY_COUNTS')
    for r in rows:
        old = oldrows[r['query_id']]
        need(r['selected']['RAW'] == old['selected']['RAW'] and r['selected']['NATIVE7'] == old['selected']['COST1'] and
             r['correct']['NATIVE7'] == old['correct']['COST1'] and r['component'] == old['component'] and
             r['target_in_C128'] == old['target_in_C128'], 'LEGACY_PER_QUERY_PARITY')
    comparisons = {b+'__to__'+m: compare(rows, b, m) for b, m in
        [('RAW','NATIVE7'),('RAW','MASS5'),('RAW','ADDITIVE4'),('NATIVE7','MASS5'),('NATIVE7','ADDITIVE4'),('MASS5','ADDITIVE4')]}
    return dict(status='FIXED_SIMPLE_EXTERNAL_DATASET_COMPLETE', dataset=name, authority=bind(AUTH),
        bundle=bind(BUNDLE), predictions_seal=bind(OUT/'all_predictions_prelabel_seal.json'),
        query_count=len(rows), groups=d['groups'], group_unit='source_video' if name=='grozi' else 'known_patient',
        source_fit='single all-H593 COST1 fit, 570 target-present training queries; no external retraining or calibration',
        primary='MASS5', secondary='ADDITIVE4', counts=counts,
        MRR={m: math.fsum(1/r['ranks'][m] for r in rows)/len(rows) for m in counts},
        target_recall_C128=sum(r['target_in_C128'] for r in rows), comparisons=comparisons, rows=rows,
        native_all_choices_match_legacy=True, external_training_updates=0,
        candidate_source='original native ColNomic full-gallery natural C128; all127 challengers; HOLD=0',
        external_scope='fixed-head transfer recheck on previously used external panel', untouched_confirmation_claimed=False,
        clinical_diagnosis_claimed=False, ownership_claimed=False,
        bootstrap=dict(draws=100000,seed=20260924,estimand='equally weighted group mean paired accuracy difference'),
        legacy_result=oldv['result'])


def join(verify=False):
    predictions, hs = all_predictions()
    bindings = {}
    for name in DATA:
        result = joined_result(name, predictions[name], hs)
        path = OUT / name / 'result.json'
        if verify:
            need(read(path) == result, 'INDEPENDENT_JOIN_REPLAY')
        else:
            write(path, result)
        bindings[name] = bind(path)
        print(json.dumps(dict(dataset=name, counts=result['counts'], recall=result['target_recall_C128'],
            comparisons=result['comparisons'])), flush=True)
    summary = dict(status='FIXED_SIMPLE_EXTERNAL_TRANSFER_COMPLETE', authority=bind(AUTH), bundle=bind(BUNDLE),
        datasets=bindings, query_count=1017, primary='MASS5', secondary='ADDITIVE4',
        untouched_confirmation_claimed=False, predictions_seal=bind(OUT/'all_predictions_prelabel_seal.json'))
    if verify:
        need(read(OUT/'result.json') == summary, 'SUMMARY_PARITY')
        write(OUT/'result_validation.json', dict(status='ALL1017_SEALS_COUNTS_ACTIONS_GROUPS_PASS',
            authority=bind(AUTH), result=bind(OUT/'result.json'), datasets=bindings,
            join_replays=2, independent_numpy_head_actions=3051, native_choice_parity_queries=1017))
    else:
        write(OUT/'result.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('preflight','preflight-all','preflight-join','replay','join','verify'))
    parser.add_argument('--task', type=int)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    torch.set_num_interop_threads(1)
    guard(args.stage)
    if args.stage == 'preflight': preflight(args.task)
    elif args.stage == 'preflight-all':
        for task in range(128): preflight(task)
        preflight_join()
    elif args.stage == 'preflight-join': preflight_join()
    elif args.stage == 'replay': replay(args.task)
    else: join(args.stage == 'verify')


if __name__ == '__main__':
    main()
