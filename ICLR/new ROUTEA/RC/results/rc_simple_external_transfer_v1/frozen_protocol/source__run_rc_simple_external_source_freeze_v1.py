#!/usr/bin/env python3
"""Freeze all-source MASS5/ADDITIVE4 heads without opening external data.

The legacy native COST1 fit is independently repeated and must match the old
seven parameters and all source logits exactly.  This is not an OOF evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import run_rc_six_cause_loss_binding_v1 as L
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import FEATURE_NAMES

OUT = ROOT / 'results/rc_simple_external_transfer_v1/source'
AUTH = ROOT / 'registry/rc_simple_external_transfer_authority_v1_20260924.json'
LEGACY_AUTH = ROOT / 'registry/rc_new_hyp_external_head_freeze_authority_v1_20260913.json'
LEGACY_HEAD = ROOT / 'results/rc_new_hyp_external_head_freeze_v1/COST1/head.json'
LEGACY_VALIDATION = LEGACY_HEAD.with_name('validation.json')
SIMPLE_AUTH = ROOT / 'registry/rc_h593_simple_explanations_authority_v1_20260924.json'
CACHE_VALIDATION = ROOT / 'results/rc_h593_simple_explanations_v1/cache_validation.json'
MODELS = ('NATIVE7', 'MASS5', 'ADDITIVE4')
COLS = {'MASS5': [0, 1, 2, 3], 'ADDITIVE4': [0, 2, 3]}
NAMES = ['standardized_raw_gap', 'symmetric_mass_times_free_content',
         'symmetric_visibility_mass', 'symmetric_free_content']
PROTOCOL = dict(loss='COST1', optimizer='AdamW', lr=.03, weight_decay=.001,
                steps=2000, initialization='zero', checkpoint='final',
                dtype='float64', threads=8, source_queries=593,
                eligible_target_present=570, excluded_target_absent=23,
                threshold=0., source_recipe='legacy all-H593 external COST1 freeze')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(item):
    need(bind(item['path']) == item, 'SOURCE_HASH:' + item['path'])
    return Path(item['path'])


def tensor_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def prepare():
    """Pin only source-domain inputs. No external payload is opened here."""
    old = read(LEGACY_AUTH)
    simple = read(SIMPLE_AUTH)
    cachev = read(CACHE_VALIDATION)
    inputs = dict(legacy_authority=bind(LEGACY_AUTH), legacy_head=bind(LEGACY_HEAD),
                  legacy_validation=bind(LEGACY_VALIDATION),
                  simple_authority=bind(SIMPLE_AUTH), cache_validation=bind(CACHE_VALIDATION),
                  cache=cachev['payload'], gallery=simple['public']['gallery'])
    for name, b in old['public_sources'].items():
        inputs['legacy_public_' + name] = b
    for i, family in enumerate(old['features']):
        for name, b in family.items():
            inputs[f'legacy_feature_{i:02d}_{name}'] = b
    inputs['feature_authority'] = simple['public']['feature_authority']
    code = dict(old['code_sources'])
    code['source_freeze_program'] = bind(__file__)
    manifest = dict(status='SOURCE_INPUTS_PINNED', inputs=inputs, codes=code,
                    protocol=PROTOCOL, program=bind(__file__), external_queries_read=0)
    # Re-running preparation is only allowed if the entire manifest is unchanged.
    path = OUT / 'input_manifest.json'
    if path.exists():
        need(read(path) == manifest, 'SOURCE_MANIFEST_ALREADY_FROZEN_DIFFERENT')
    else:
        write(path, manifest)
    print(json.dumps(dict(status='SOURCE_INPUTS_PINNED', manifest=bind(path))), flush=True)


def guard(stage):
    manifest = read(OUT / 'input_manifest.json')
    need(manifest['protocol'] == PROTOCOL and manifest['program'] == bind(__file__), 'SOURCE_PROGRAM_PROTOCOL')
    for b in manifest['codes'].values():
        checked(b)
    for b in manifest['inputs'].values():
        checked(b)
    if stage not in ('preflight',):
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
        a = read(AUTH)
        need(bind(OUT / 'input_manifest.json') in a['source_inputs'].values(), 'AUTHORITY_SOURCE_MANIFEST')
        for family in ('code_sources', 'source_inputs'):
            for b in a[family].values():
                checked(b)
        # external_inputs, if present, are metadata only and are never checked/opened.
    allowed = {Path(b['path']).resolve() for b in manifest['inputs'].values()}
    allowed.update(Path(b['path']).resolve() for b in manifest['codes'].values())
    allowed.update((AUTH.resolve(), (OUT / 'input_manifest.json').resolve()))

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        lower = str(path).lower()
        need(not any(t in lower for t in ('d1-mi', 'd1_mi', 'formal392', '/target_join/',
                                         'grozi', 'isic_transfer', 'rpc_external')), 'NO_EXTERNAL_OR_PROTECTED_READ')
        need(ROOT / 'reports' not in path.parents, 'NO_REPORT_READ_IN_SOURCE_FIT')
        if ROOT / 'results' in path.parents:
            need(OUT in path.parents or path in allowed, 'UNLISTED_SOURCE_RESULT:' + str(path))
    sys.addaudithook(audit)
    return manifest


def load_data(manifest):
    ins = manifest['inputs']
    cachev = read(checked(ins['cache_validation']))
    need(cachev['status'] == 'CACHE_PASS' and cachev['payload'] == ins['cache'], 'CACHE_VALIDATION')
    cache = read(checked(ins['cache']))
    need(cache['labels_included'] is False, 'SOURCE_CACHE_LABEL_FREE')
    need(cachev['authority'] == ins['simple_authority'] == cache['authority'], 'SIMPLE_CACHE_LINEAGE')
    old = read(checked(ins['legacy_head']))
    oldv = read(checked(ins['legacy_validation']))
    need(oldv['status'] == 'FINAL_HEAD_FRESH_TRAIN_NUMPY_PASS' and oldv['head'] == ins['legacy_head'], 'LEGACY_HEAD_VALIDATION')
    need(old['authority'] == oldv['authority'] == ins['legacy_authority'], 'LEGACY_LINEAGE')
    need(old['optimizer'] == dict(name='AdamW', lr=.03, weight_decay=.001, steps=2000,
                                 initialization='zero', checkpoint='final'), 'EXACT_LEGACY_OPTIMIZER_RECIPE')
    need(old['runtime']['threads'] == 8 and old['inference']['dtype'] == 'float64'
         and old['inference']['threshold'] == 0., 'EXACT_LEGACY_NUMERICAL_PROTOCOL')
    roles = {r['query_id']: r for r in read(checked(ins['legacy_public_curator']))['records']}
    gallery = read(checked(ins['gallery']))
    need(gallery['corrected_mapping_sha256'] == old['gallery_mapping'], 'SAME_GALLERY_IDENTITY_MAPPING')
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    rows, _ = L.H.features()
    cr = cache['rows']
    need(len(rows) == len(cr) == len(roles) == 593, 'COMPLETE593_SOURCE')
    need([r['query_id'] for r in rows] == old['development_query_ids'], 'OLD_SOURCE_QUERY_ORDER')
    native = torch.stack([r['modes']['REAL']['X'] for r in rows])
    full = np.asarray([r['X'] for r in cr], dtype=np.float64)
    targets = []
    maximum = 0.
    for i, (r, c) in enumerate(zip(rows, cr)):
        need(r['query_id'] == c['query_id'] and r['execution_ordinal'] == c['execution_ordinal'] == i, 'CACHE_QUERY_ORDER')
        need(r['source_image_sha256'] == c['source_image_sha256'], 'SAME_SOURCE_IMAGE')
        need(r['candidate_physical_rows'] == c['axis'] and r['winner'] == c['winner'] and
             r['challenger_positions'] == c['challengers'], 'IDENTICAL_C128_ACTION_AXIS')
        need(sorted(c['challengers'] + [c['winner']]) == list(range(128)), 'PERMUTED_FULL_C128')
        err = float(np.abs(native[i].numpy() - np.asarray(c['native_X'])).max())
        maximum = max(maximum, err)
        need(err < 2e-10, 'SOURCE_NATIVE_CACHE_NUMERIC_PARITY')
        need(np.array_equal(native[i, :, 0].numpy(), full[i, :, 0]), 'EXACT_SOURCE_RAW_GAP')
        m, l = np.asarray(c['mass']), np.asarray(c['free_content'])
        w, ch = c['winner'], c['challengers']
        sym = lambda a, b: (a - b) / (np.abs(a) + np.abs(b) + 1e-12)
        expected = np.stack([sym(m[ch] * l[ch], m[w] * l[w]), sym(m[ch], m[w]), sym(l[ch], l[w])], axis=1)
        need(np.max(np.abs(expected - full[i, :, 1:])) < 2e-10, 'SIMPLIFIED_SOURCE_FORMULA')
        targets.append(L.H.target_position(r, roles[r['query_id']]['identity'], labels))
    targets = np.array(targets, dtype=np.int64)
    eligible = np.flatnonzero(targets >= -1)
    absent = np.flatnonzero(targets == -2)
    need(len(eligible) == 570 and len(absent) == 23, 'SAME570_TARGET_PRESENT_23_ABSENT')
    trainids = [rows[i]['query_id'] for i in eligible]
    absentids = [rows[i]['query_id'] for i in absent]
    need(trainids == old['train_query_ids'] and absentids == old['absent_query_ids'], 'EXACT_LEGACY_TRAIN_AND_ABSENT_ORDER')
    data = dict(rows=rows, full=full, native=native, y=torch.from_numpy(targets[eligible]),
                eligible=eligible, old=old, train_query_ids=trainids, absent_query_ids=absentids,
                native_cache_max_error=maximum,
                effective_train_components=len({roles[q]['component'] for q in trainids}),
                gallery_mapping=gallery['corrected_mapping_sha256'])
    return data


def design(data, model):
    return data['native'] if model == 'NATIVE7' else torch.from_numpy(data['full'][..., COLS[model]].copy())


def selected(rows, z):
    return [r['challenger_positions'][int(v.argmax())] if float(v.max()) > 0 else r['winner']
            for r, v in zip(rows, z)]


def head_payload(data, model, theta):
    x = design(data, model)
    z = x @ theta[:-1] + theta[-1]
    theta_hex = [float(t).hex() for t in theta]
    parity = None
    if model == 'NATIVE7':
        need(theta_hex == data['old']['theta_hex'], 'LEGACY_NATIVE7_EXACT_THETA_PARITY')
        parity = {}
        for mode in ('REAL', 'CBIND'):
            mx = torch.stack([r['modes'][mode]['X'] for r in data['rows']])
            mz = mx @ theta[:-1] + theta[-1]
            need(tensor_hash(mz) == data['old']['engineering_logit_sha256'][mode], 'LEGACY_NATIVE7_EXACT_LOGITS_' + mode)
            need(selected(data['rows'], mz) == data['old']['engineering_selected_positions'][mode], 'LEGACY_NATIVE7_EXACT_ACTION_' + mode)
            parity[mode] = dict(logit_sha256=tensor_hash(mz), exact_actions=True)
    return dict(status='SOURCE_HEAD_FROZEN_NOT_EXTERNAL_RESULT', authority=bind(AUTH),
                input_manifest=bind(OUT / 'input_manifest.json'), program=bind(__file__),
                model=model, loss='COST1', theta_hex=theta_hex,
                feature_names=list(FEATURE_NAMES) if model == 'NATIVE7' else [NAMES[j] for j in COLS[model]],
                train_query_ids=data['train_query_ids'], absent_query_ids=data['absent_query_ids'],
                development_query_ids=[r['query_id'] for r in data['rows']],
                effective_train_components=data['effective_train_components'], gallery_mapping=data['gallery_mapping'],
                training_features_sha256=tensor_hash(x[data['eligible']]),
                training_targets_sha256=hashlib.sha256(data['y'].numpy().astype('<i8').tobytes()).hexdigest(),
                source_logit_sha256=tensor_hash(z), source_selected_positions=selected(data['rows'], z),
                native_legacy_exact_parity=parity, native_cache_max_error=data['native_cache_max_error'],
                inference=dict(dtype='float64', threshold=0., candidates=128, challengers=127,
                               tie_rule='first_challenger_in_frozen_order',
                               decision='SWITCH_IF_MAX_LOGIT_GT_ZERO_ELSE_HOLD'),
                optimizer=PROTOCOL, legacy_native_head=bind(LEGACY_HEAD), external_queries_read=0,
                evidence='All opened H593 source fit; not grouped OOF accuracy',
                runtime=dict(torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads()))


def verify(data, model, fresh_train=False):
    path = OUT / model / 'head.json'
    p = read(path)
    theta = torch.tensor([float.fromhex(v) for v in p['theta_hex']], dtype=torch.float64)
    need(p == head_payload(data, model, theta), 'SOURCE_HEAD_SEMANTIC_REPLAY')
    x = design(data, model)
    if fresh_train:
        replay = L.train(x[data['eligible']], data['y'], 'COST1')
        need(torch.equal(replay, theta), 'SOURCE_FRESH_TRAIN_EXACT_PARITY')
    z = x @ theta[:-1] + theta[-1]
    independent = np.sum(x.numpy() * theta[:-1].numpy(), axis=-1) + float(theta[-1])
    error = float(np.abs(independent - z.numpy()).max())
    need(error < 2e-10, 'SOURCE_NUMPY_LOGITS')
    need(selected(data['rows'], torch.from_numpy(independent)) == p['source_selected_positions'], 'SOURCE_NUMPY_ACTIONS')
    v = dict(status='SOURCE_HEAD_NUMPY_ACTIONS_PASS', authority=bind(AUTH),
             head=bind(path), input_manifest=bind(OUT / 'input_manifest.json'),
             maximum_logit_error=error, checked_logits=593 * 127,
             legacy_native_exact_parity=(model == 'NATIVE7'),
             fresh_train_replay=fresh_train, external_queries_read=0)
    # Preserve a previous stronger verification instead of downgrading it.
    vp = path.with_name('validation.json')
    if vp.exists():
        oldv = read(vp)
        if oldv.get('head') == v['head'] and oldv.get('fresh_train_replay'):
            v['fresh_train_replay'] = True
    write(vp, v)
    print(json.dumps(dict(status=v['status'], model=model, maximum_logit_error=error)), flush=True)


def fit(data, models):
    for model in models:
        path = OUT / model / 'head.json'
        if path.exists():
            verify(data, model)
            print(json.dumps(dict(event='SOURCE_FROZEN_HEAD_REUSED', model=model)), flush=True)
            continue
        start = time.monotonic()
        x = design(data, model)
        theta = L.train(x[data['eligible']], data['y'], 'COST1')
        need(torch.isfinite(theta).all(), 'FINITE_SOURCE_HEAD')
        payload = head_payload(data, model, theta)
        write(path, payload)
        write(path.with_name('timing.json'), dict(model=model, seconds=time.monotonic() - start,
              slurm_job_id=os.environ['SLURM_JOB_ID'], completed_updates=2000))
        verify(data, model)


def bundle(data):
    heads = {}
    for model in MODELS:
        verify(data, model)
        heads[model] = dict(head=bind(OUT / model / 'head.json'), validation=bind(OUT / model / 'validation.json'))
    write(OUT / 'bundle.json', dict(status='SIMPLE_SOURCE_HEADS_FROZEN_PASS', authority=bind(AUTH),
          primary='MASS5', control='ADDITIVE4', native_reference='NATIVE7', heads=heads,
          program=bind(__file__), input_manifest=bind(OUT / 'input_manifest.json'), protocol=PROTOCOL,
          native_theta_and_source_logits_exact_legacy_parity=True, external_queries_read=0,
          frozen_before_external_predictions=True, source_fit_not_OOF=True))
    print(json.dumps(dict(status='SIMPLE_SOURCE_HEADS_FROZEN_PASS', bundle=bind(OUT / 'bundle.json'))), flush=True)


def preflight(data):
    need(design(data, 'MASS5').shape == (593, 127, 4), 'MASS5_DESIGN')
    need(design(data, 'ADDITIVE4').shape == (593, 127, 3), 'ADDITIVE4_DESIGN')
    for model in MODELS:
        x = design(data, model)
        zero = torch.zeros(x.shape[-1] + 1, dtype=torch.float64)
        need(selected(data['rows'], x @ zero[:-1] + zero[-1]) == [r['winner'] for r in data['rows']], 'ZERO_HOLD')
    write(OUT / 'preflight.json', dict(status='SOURCE_PROTOCOL_SCOPE_ACTION_PREFLIGHT_PASS',
          input_manifest=bind(OUT / 'input_manifest.json'), source_queries=593,
          train_queries=570, absent_queries=23, legacy_source_order_exact=True,
          native_cache_max_error=data['native_cache_max_error'], natural_training_updates=0,
          external_queries_read=0))
    print(json.dumps(read(OUT / 'preflight.json')), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'preflight', 'fit', 'verify', 'bundle'])
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--fresh-train', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if args.stage == 'prepare':
        prepare()
        return
    manifest = guard(args.stage)
    data = load_data(manifest)
    if args.stage == 'preflight':
        preflight(data)
    elif args.stage == 'fit':
        fit(data, (args.model,) if args.model else MODELS)
        if args.model is None:
            bundle(data)
    elif args.stage == 'verify':
        for model in (args.model,) if args.model else MODELS:
            verify(data, model, args.fresh_train)
    else:
        bundle(data)


if __name__ == '__main__':
    main()
