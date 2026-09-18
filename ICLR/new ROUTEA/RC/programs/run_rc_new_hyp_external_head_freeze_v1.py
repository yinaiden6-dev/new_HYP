#!/usr/bin/env python3
"""Fit fixed full-development recipes before any external query evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import run_rc_six_cause_loss_binding_v1 as L
import run_rc_h593_group_risk_strong_base_v1 as G
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import FEATURE_NAMES
H = L.H
M = H.M
read, write, bind, checked, need = M.read, M.write, M.bind, M.checked, M.need
OUT = ROOT / 'results/rc_new_hyp_external_head_freeze_v1'
AUTH = ROOT / 'registry/rc_new_hyp_external_head_freeze_authority_v1_20260913.json'
MODELS = ('COST1', 'CE', 'COST4', 'GROUP_COST4', 'RAW2_CE')


def guard(stage):
    a = read(AUTH)
    allowed = {AUTH.resolve()}
    for family in ('code_sources', 'public_sources'):
        for b in a[family].values():
            allowed.add(Path(checked(b)).resolve())
    for shard in a['features']:
        for b in shard.values():
            allowed.add(Path(checked(b)).resolve())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p).lower()
        need(not any(t in s for t in ('/target_join/', 'd1-mi', 'd1_mi', 'grozi', '/reports/', 'rc_opened_')), 'PROTECTED_READ')
        if ROOT / 'results' in p.parents:
            need(OUT in p.parents or p in allowed, 'UNLISTED_RESULT:' + s)
    sys.addaudithook(audit)
    if stage != 'preflight':
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
        need(read(OUT / 'preflight.json')['authority'] == bind(AUTH), 'PREFLIGHT')
    return a


def action(x, theta, row):
    z = x @ theta[:-1] + theta[-1]
    k = int(z.argmax())
    return z, row['challenger_positions'][k] if z[k] > 0 else row['winner']


def compute(model):
    a = read(AUTH)
    rows, _ = H.features()
    roles = {r['query_id']: r for r in read(checked(a['public_sources']['curator']))['records']}
    need(set(roles) == {r['query_id'] for r in rows}, 'ALL_OPENED_DEVELOPMENT_ROLES')
    labels, mapping = L.N.P.gallery_labels()
    data = H.batch(rows, roles, labels)
    eligible = [r for r in rows if H.target_position(r, roles[r['query_id']]['identity'], labels) >= -1]
    absent = [r['query_id'] for r in rows if H.target_position(r, roles[r['query_id']]['identity'], labels) == -2]
    need(len(eligible) == 570 and len(absent) == 23, 'FROZEN_TRAIN_SCOPE')
    x, y = data['X'], data['y']
    if model in ('COST1', 'CE'):
        theta = L.train(x, y, model)
    elif model == 'COST4':
        theta = H.fit_base(data)[0]
    elif model == 'GROUP_COST4':
        weight, _ = G.group_weights([roles[r['query_id']]['component'] for r in eligible])
        theta = G.fit_group_base(data, weight)['theta']
    else:
        theta = L.train(x[:, :, :1], y, 'CE')
    need(torch.isfinite(theta).all(), 'FINITE_PARAMETERS')
    digests, selected = {}, {}
    for mode in ('REAL', 'CBIND'):
        allx = torch.stack([r['modes'][mode]['X'] for r in rows])
        if model == 'RAW2_CE':
            allx = allx[:, :, :1]
        z = allx @ theta[:-1] + theta[-1]
        digests[mode] = hashlib.sha256(z.numpy().astype('<f8', copy=False).tobytes()).hexdigest()
        positions = []
        for r, v in zip(rows, z):
            k = int(v.argmax())
            positions.append(r['challenger_positions'][k] if v[k] > 0 else r['winner'])
        selected[mode] = positions
    return dict(status='FULL_DEVELOPMENT_HEAD_SEALED_NOT_EXTERNAL_RESULT', authority=bind(AUTH),
                model=model, theta_hex=[float(t).hex() for t in theta],
                feature_names=list(FEATURE_NAMES[:1] if model == 'RAW2_CE' else FEATURE_NAMES),
                train_query_ids=[r['query_id'] for r in eligible], absent_query_ids=absent,
                development_query_ids=[r['query_id'] for r in rows],
                effective_train_components=len({roles[r['query_id']]['component'] for r in eligible}),
                gallery_mapping=mapping, engineering_logit_sha256=digests,
                engineering_selected_positions=selected, external_queries_read=0,
                inference=dict(dtype='float64', threshold=0.0, candidates=128, challengers=127,
                               tie_rule='first_challenger_in_frozen_order',
                               decision='SWITCH_IF_MAX_LOGIT_GT_ZERO_ELSE_HOLD'),
                optimizer=dict(name='AdamW', lr=.03, weight_decay=.001, steps=2000,
                               initialization='zero', checkpoint='final'),
                runtime=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version,
                             threads=torch.get_num_threads()), HYP_GO_claimed=False)


def fit(model, verify=False):
    folder = OUT / model
    payload = compute(model)
    if not verify:
        write(folder / 'head.json', payload)
        subprocess.run([sys.executable, __file__, 'verify', '--model', model], check=True)
        print(json.dumps(dict(model=model, status='HEAD_FRESH_REPLAY_PASS')), flush=True)
        return
    need(payload == read(folder / 'head.json'), 'EXACT_FRESH_TRAIN_AND_LOGITS_REPLAY')
    rows, _ = H.features()
    t = np.array([float.fromhex(h) for h in payload['theta_hex']], dtype=np.float64)
    max_error = 0.0
    for mode in ('REAL', 'CBIND'):
        for i, row in enumerate(rows):
            x = row['modes'][mode]['X']
            if model == 'RAW2_CE':
                x = x[:, :1]
            z = np.sum(x.numpy() * t[:-1], axis=1) + t[-1]
            reference = (x @ torch.from_numpy(t[:-1]) + t[-1]).numpy()
            error = float(np.abs(z - reference).max())
            need(error < 2e-10, 'NUMPY_LOGITS')
            max_error = max(max_error, error)
            k = int(np.argmax(z))
            pos = row['challenger_positions'][k] if z[k] > 0 else row['winner']
            need(pos == payload['engineering_selected_positions'][mode][i], 'NUMPY_ACTION')
    write(folder / 'validation.json', dict(status='FINAL_HEAD_FRESH_TRAIN_NUMPY_PASS',
          authority=bind(AUTH), head=bind(folder / 'head.json'), logit_checks=593 * 127 * 2,
          max_abs_error=max_error, external_queries_read=0))


def join():
    heads = {}
    for m in MODELS:
        v = read(OUT / m / 'validation.json')
        need(v['status'] == 'FINAL_HEAD_FRESH_TRAIN_NUMPY_PASS' and v['authority'] == bind(AUTH), 'ALL_FIVE_VALIDATED')
        p = read(checked(v['head']))
        need(p['model'] == m and p['authority'] == bind(AUTH), 'MODEL_LINEAGE')
        heads[m] = dict(head=v['head'], validation=bind(OUT / m / 'validation.json'))
    write(OUT / 'bundle.json', dict(status='EXTERNAL_HEADS_READY_DATA_AND_INFERENCE_PENDING',
          authority=bind(AUTH), primary='COST1', secondary='CE', heads=heads,
          sources=read(AUTH)['public_sources'], external_queries_scored=0,
          OOF_scores_not_final_head_scores=True, automatic_deployment_change=False,
          HYP_GO_claimed=False))
    print(json.dumps(dict(status='EXTERNAL_HEADS_READY_DATA_AND_INFERENCE_PENDING', models=list(heads))), flush=True)


def preflight():
    row = dict(challenger_positions=list(range(1, 128)), winner=0)
    x = torch.zeros((127, 6), dtype=torch.float64)
    t = torch.zeros(7, dtype=torch.float64)
    need(action(x, t, row)[1] == 0, 'ZERO_HOLD')
    t[-1] = 1
    need(action(x, t, row)[1] == 1, 'FIRST_TIED_CHALLENGER')
    z = torch.zeros((4, 127), dtype=torch.float64, requires_grad=True)
    y = torch.tensor([-1, 0, 31, 126])
    need(torch.equal(G.per_query_loss(z, y).mean(), H.loss(z, y)), 'GROUP_ORIGINAL_LOSS')
    weights, _ = G.group_weights(['a', 'a', 'b', 'c'])
    need(torch.allclose(weights, torch.tensor([1/6, 1/6, 1/3, 1/3], dtype=torch.float64)), 'GROUP_WEIGHTS')
    write(OUT / 'preflight.json', dict(status='FINAL_HEAD_RECIPE_ACTION_PREFLIGHT_PASS', authority=bind(AUTH), natural_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preflight', 'fit', 'verify', 'join'])
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    guard(args.stage)
    if args.stage == 'preflight': preflight()
    elif args.stage == 'join': join()
    else:
        need(args.model in MODELS, 'MODEL_REQUIRED')
        fit(args.model, args.stage == 'verify')
