#!/usr/bin/env python3
"""Prelabel CPU controls for an independently addressed internal-M pilot.

run_controls(authority_path_or_binding, manifest_dict_or_None, pilot_out)
must run before the pilot's label join. It never opens held labels.
summarize_controls(join_result_dict_or_path, returned_seal_or_controls_dir)
uses only the caller's already audited, post-seal label rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('PRE_REAL', 'PRE_CONSTANT', 'POST_REAL', 'POST_CONSTANT')
TOL = 1e-10
CONFIG = dict(optimizer='AdamW', learning_rate=3e-4, weight_decay=1e-3,
              seed=17, updates=16, clip_norm=1., delta_dtype='float32',
              scoring_dtype='float64', device='cpu', trainable_parameters=5,
              selection='Fixed terminal update16; only additional update budget matched')


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(binding):
    need(bind(binding['path']) == binding, 'CONTROL_SHA_DRIFT:' + binding['path'])
    return Path(binding['path'])


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def vector(value, size, name):
    value = np.asarray(value, dtype=np.float64)
    need(value.shape == (size,) and np.isfinite(value).all(), 'CONTROL_VECTOR:' + name)
    return value


def compare(a, b, name):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'CONTROL_SHAPE:' + name)
    error = float(np.max(np.abs(a - b))) if a.size else 0.
    need(error <= TOL, f'CONTROL_PARITY:{name}:{error}')
    return error


def features(row, content):
    raw = vector(row['raw_scores'], 128, 'RAW')
    mass = vector(row['M'], 128, 'M')
    content = vector(content, 128, 'L')
    axis = row['candidate_ids']; winner = row['winner_index']
    need(len(axis) == len(set(axis)) == 128 and axis == sorted(axis), 'CONTROL_AXIS')
    need(len(row['candidate_identities']) == len(set(row['candidate_identities'])) == 128, 'CONTROL_IDENTITY_AXIS')
    need(winner == int(np.argmax(raw)), 'CONTROL_RAW_WINNER')
    idx = np.asarray([i for i in range(128) if i != winner])
    need(idx.tolist() == row['challenger_positions'], 'CONTROL_CHALLENGER_AXIS')
    def sym(v):
        return (v[idx] - v[winner]) / (np.abs(v[idx]) + abs(v[winner]) + 1e-12)
    return np.column_stack(((raw[idx] - raw[winner]) / max(float(raw.std(ddof=0)), 1e-12),
                            sym(mass * content), sym(mass), sym(content), np.ones(127)))


def head_prediction(row, content, theta, additive=False):
    x = features(row, content)
    if additive:
        x = x[:, [0, 2, 3, 4]]
    theta = vector(theta, x.shape[1], 'THETA')
    z = x @ theta
    scores = np.zeros(128, dtype=np.float64)
    scores[row['challenger_positions']] = z
    best = int(np.argmax(z))
    pos = row['challenger_positions'][best] if z[best] > 0. else row['winner_index']
    return dict(L=list(map(float, content)), features=x.tolist(), logits=z.tolist(), scores128=scores.tolist(),
                challenger_positions=row['challenger_positions'], prediction_position=pos,
                prediction_id=row['candidate_ids'][pos], prediction_identity=row['candidate_identities'][pos],
                switched=pos != row['winner_index'], threshold=0., hold_score=0.)


def content_prediction(row, content):
    content = vector(content, 128, 'PURE_L')
    order = np.argsort(-content, kind='stable').tolist(); pos = order[0]
    return dict(L=content.tolist(), scores128=content.tolist(), ranked_positions=order,
                prediction_position=pos, prediction_id=row['candidate_ids'][pos],
                prediction_identity=row['candidate_identities'][pos], switched=pos != row['winner_index'],
                explicit_external_M=False, internally_conditioned_M_may_remain=True)


def cost1(z, row):
    import torch
    from torch.nn import functional as F
    need(len(row['target_positions']) == 1, 'CONTROL_SINGLE_TRAIN_TARGET')
    target = row['target_positions'][0]
    if target == row['winner_index']:
        return F.softplus(z.amax())
    pos = row['challenger_positions'].index(target)
    wrong = z.clone(); wrong[pos] = -torch.inf
    return F.softplus(-z[pos]) + F.softplus(wrong.amax())


def fit_score_update(train_rows, theta0):
    """Only five delta parameters receive gradient or weight decay."""
    import torch
    need(len(train_rows) == 16, 'CONTROL_TRAIN16')
    torch.manual_seed(CONFIG['seed'])
    base = torch.tensor(theta0, dtype=torch.float64, device='cpu')
    delta = torch.nn.Parameter(torch.zeros(5, dtype=torch.float32, device='cpu'))
    opt = torch.optim.AdamW([delta], lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    history = []
    for step, row in enumerate(train_rows):
        x = torch.tensor(features(row, row['L0']), dtype=torch.float64, device='cpu')
        opt.zero_grad(set_to_none=True)
        z = x @ (base + delta.double())
        if step == 0:
            compare(z.detach().numpy(), head_prediction(row, row['L0'], theta0)['logits'], 'ZERO_DELTA')
        loss = cost1(z, row); loss.backward()
        need(delta.grad is not None and torch.isfinite(delta.grad).all(), 'CONTROL_FINITE_GRAD')
        norm = float(torch.nn.utils.clip_grad_norm_([delta], CONFIG['clip_norm']))
        need(norm > 0., 'CONTROL_NONZERO_GRAD')
        opt.step()
        need(base.grad is None and torch.isfinite(delta).all(), 'CONTROL_ONLY_DELTA_TRAINED')
        history.append(dict(step=step + 1, query_id=row['query_id'], loss=float(loss.detach()),
                            gradient_norm=norm, delta_theta=delta.detach().double().tolist()))
    final = (base + delta.detach().double()).tolist()
    return dict(config=CONFIG, theta0=list(map(float, theta0)), theta=final,
                theta_hex=[float(v).hex() for v in final], delta_theta=delta.detach().double().tolist(),
                history=history, held_label_reads=0, original_full_TRAIN_head_preserved=True,
                scope='Extra TRAIN16 updates on a full-TRAIN head; not a system trained with only 16 labels')


def load_additive(manifest, simple_root):
    """Verify full-fold theta, exact feature order, and original held logits."""
    simple_root = Path(simple_root)
    fv_path = simple_root / 'fold0/validation.json'; fv = read(fv_path)
    need(fv['status'] == 'FOLD_NUMPY_AND_SELECTION_PASS', 'CONTROL_ADDITIVE_VALIDATION')
    apath = simple_root / 'fold0/COST1_ADDITIVE4.json'; abind = bind(apath)
    need(abind in fv['models'], 'CONTROL_ADDITIVE_SEALED_MODEL')
    model = read(checked(abind)); authority = read(checked(model['authority']))
    need(model['authority'] == fv['authority'] and model['model'] == 'COST1_ADDITIVE4'
         and model['fold'] == 0 and model['heldout_label_reads'] == 0, 'CONTROL_ADDITIVE_FOLD')
    need(authority['public']['split'] == manifest['provenance']['split'], 'CONTROL_IDENTICAL_OUTER_SPLIT')
    split = read(checked(fv['split']))
    original_split = read(checked(manifest['provenance']['split']))['folds'][0]
    need(set(split['train']) == set(original_split['train_query_ids'])
         and set(split['held']) == set(original_split['heldout_query_ids']), 'CONTROL_FULL_TRAIN_HEAD')
    need(set(r['query_id'] for r in manifest['train_rows']) <= set(split['train'])
         and set(r['query_id'] for r in manifest['probe_rows']) <= set(split['held']), 'CONTROL_PANEL_SPLITS')
    cvpath = simple_root / 'cache_validation.json'; cv = read(cvpath)
    need(cv['status'] == 'CACHE_PASS' and cv['authority'] == model['authority'], 'CONTROL_SIMPLE_CACHE')
    cache = read(checked(cv['payload']))
    need(cache['labels_included'] is False and cache['authority'] == model['authority'], 'CONTROL_CACHE_LABEL_FREE')
    rows = {r['query_id']: r for r in cache['rows']}
    theta = [float.fromhex(v) for v in model['theta_hex']]
    need(len(theta) == 4, 'CONTROL_ADDITIVE_FULL_THETA')
    all_x = np.asarray([rows[q]['X'] for q in split['held']], dtype=np.float64)
    replay = all_x[:, :, [0, 2, 3]] @ np.asarray(theta[:3]) + theta[3]
    err = compare(replay, model['held_logits'], 'ADDITIVE_ORIGINAL_HELD_LOGITS')
    for row in manifest['train_rows'] + manifest['probe_rows']:
        old = rows[row['query_id']]
        need(old['axis'] == row['candidate_ids'] and old['winner'] == row['winner_index']
             and old['challengers'] == row['challenger_positions'], 'CONTROL_SIMPLE_CANDIDATE_BINDING')
        err = max(err, compare(old['mass'], row['M'], 'SIMPLE_M'),
                  compare(old['free_content'], row['L0'], 'SIMPLE_L0'),
                  compare(old['X'], features(row, row['L0'])[:, :4], 'SIMPLE_FEATURES'))
    return theta, dict(model=abind, validation=bind(fv_path), authority=model['authority'],
                       split=fv['split'], cache=cv['payload'], cache_validation=bind(cvpath),
                       parameter_field='theta_hex (full TRAIN refit, not inner_theta_hex)',
                       theta_hex=model['theta_hex'], theta=theta,
                       feature_order=['standardized_raw_gap', 'symmetric_M_gap', 'symmetric_L_gap', 'bias'],
                       max_replay_error=err, full_train_count=len(split['train']),
                       effective_train_count=len(split['effective_train']),
                       note='Full outer TRAIN fitting excludes queries without a target in C128, as in the original experiment')


def _run_controls(authority, manifest, out, simple_root):
    authority_binding = authority if isinstance(authority, dict) else bind(authority)
    a = read(checked(authority_binding)); m = read(checked(a['manifest']))
    if manifest is not None:
        need(m == manifest, 'CONTROL_MANIFEST_ARGUMENT')
    out = Path(out).resolve(); destination = out / 'cpu_controls'
    need(len(m['train_rows']) == 16 and len(m['probe_rows']) == 8 and tuple(a['arms']) == ARMS,
         'CONTROL_PILOT_16_8_FOUR_ARMS')
    need(not any(k.startswith('target') or k in ('identity', 'component', 'group', 'raw_correct')
                 for row in m['probe_rows'] for k in row), 'CONTROL_LABEL_FREE_PROBE')
    primary_seal_binding = bind(out / 'all_predictions_prelabel_seal.json')
    primary_seal = read(checked(primary_seal_binding))
    need(primary_seal['authority'] == authority_binding and primary_seal['held_label_reads'] == 0,
         'CONTROL_PRIMARY_PRELABEL_SEAL')
    need(primary_seal['seals'] == [bind(out / arm / 'prediction_seal.json') for arm in ARMS],
         'CONTROL_FOUR_SEALS')
    own = destination / 'all_predictions_prelabel_seal.json'
    if own.exists():
        prior = read(own)
        need(prior['authority'] == authority_binding and prior['primary_seal'] == primary_seal_binding
             and prior['program'] == bind(__file__), 'CONTROL_RESUME_BINDING')
        checked(prior['parameters'])
        for item in prior['predictions']:
            checked(item)
        return prior
    need(not (out / 'result.json').exists(), 'CONTROL_CANNOT_FIRST_FIT_AFTER_LABEL_JOIN')
    theta0 = [float.fromhex(v) for v in m['frozen_head']['theta_hex']]
    compare(theta0, m['frozen_head']['theta'], 'FROZEN_HEAD_BITS')
    parent = read(checked(m['frozen_head']['parent']))
    old = parent['parameters']['COST1_REFIT_M1Q0R0']['theta_hex']
    need([old[i] for i in (0, 1, 2, 3, 6)] == m['frozen_head']['theta_hex']
         and parent['fold'] == 0 and parent['heldout_label_reads'] == 0, 'CONTROL_OLD_FULL_TRAIN_HEAD')
    additive_theta, additive_provenance = load_additive(m, simple_root)
    allowed = {}
    arm_seals = {}
    for arm in ARMS:
        seal = read(checked(bind(out / arm / 'prediction_seal.json')))
        need(seal['authority'] == authority_binding and seal['held_label_reads'] == 0, 'CONTROL_ARM_PRELABEL')
        checked(seal['adapter']); arm_seals[arm] = seal
        allowed[arm] = {item['path']: item for item in seal['predictions']}
    fit = fit_score_update(m['train_rows'], theta0)
    parameters = dict(authority=authority_binding, manifest=a['manifest'], program=bind(__file__),
                      frozen_head=m['frozen_head'], additive=additive_provenance, score_update=fit,
                      labels_read='TRAIN manifest target_positions only; no held labels', held_label_reads=0)
    write(destination / 'parameters.json', parameters)
    parameter_binding = bind(destination / 'parameters.json')
    predictions = []
    for panel, rows in (('train', m['train_rows']), ('probe', m['probe_rows'])):
        for row in rows:
            q = row['query_id']; cv = read(out / 'encoder_cache' / q / 'validation.json')
            need(cv['authority'] == authority_binding and cv['status'] == 'QUERY_ENCODER_CACHE_PASS'
                 and cv['held_labels_read'] is False, 'CONTROL_ENCODER_CACHE')
            parity = read(checked(cv['parity']))
            need(parity['query_id'] == q, 'CONTROL_PARITY_QUERY')
            compare(parity['original_L0'], row['L0'], 'ORIGINAL_L0')
            contents = dict(SOURCE_ZERO=row['L0'], FRESH_ZERO=parity['fresh_L0'])
            source_bindings = [bind(out / 'encoder_cache' / q / 'validation.json'), cv['parity']]
            for arm in ARMS:
                for intervention in (('native', 'shuffled') if arm.endswith('REAL') else ('native',)):
                    path = out / arm / 'predictions' / panel / (q + '_' + intervention + '.json')
                    b = bind(path); need(allowed[arm].get(b['path']) == b, 'CONTROL_SEALED_PREDICTION')
                    p = read(checked(b))
                    need(p['authority'] == authority_binding and p['adapter'] == arm_seals[arm]['adapter']
                         and p['query_id'] == q and p['arm'] == arm and p['intervention'] == intervention
                         and p['held_label_reads'] == 0 and p['candidate_ids'] == row['candidate_ids'],
                         'CONTROL_PREDICTION_IDENTITY')
                    compare(p['M'], row['M'], 'PREDICTION_M'); compare(p['raw_scores'], row['raw_scores'], 'PREDICTION_RAW')
                    rebuilt = head_prediction(row, p['L'], theta0)
                    compare(rebuilt['logits'], p['decision']['logits'], 'PRIMARY_HEAD')
                    need(rebuilt['prediction_id'] == p['decision']['prediction_id'], 'PRIMARY_ACTION')
                    name = arm + ('_SHUFFLED' if intervention == 'shuffled' else '')
                    contents[name] = p['L']; source_bindings.append(b)
            methods = {}
            for name, content in contents.items():
                methods['PRODUCT5__' + name] = head_prediction(row, content, theta0)
                methods['ADDITIVE4__' + name] = head_prediction(row, content, additive_theta, additive=True)
                methods['PURE_L__' + name] = content_prediction(row, content)
            methods['SCORE_UPDATE5_ZERO'] = head_prediction(row, row['L0'], theta0)
            methods['SCORE_UPDATE5'] = head_prediction(row, row['L0'], fit['theta'])
            path = destination / 'predictions' / panel / (q + '.json')
            write(path, dict(authority=authority_binding, parameters=parameter_binding, query_id=q, split=panel,
                             candidate_ids=row['candidate_ids'], candidate_identities=row['candidate_identities'],
                             raw_winner_position=row['winner_index'], M=row['M'], raw_scores=row['raw_scores'],
                             methods=methods, source_bindings=source_bindings, held_label_reads=0))
            predictions.append(bind(path))
    seal = dict(status='INTERNAL_M_CPU_CONTROLS_PRELABEL_SEALED', authority=authority_binding,
                manifest=a['manifest'], program=bind(__file__), primary_seal=primary_seal_binding,
                output_dir=str(destination), parameters=parameter_binding, predictions=predictions,
                train_queries=16, probe_queries=8, held_label_reads=0,
                new_encoder_forwards=0, new_roma_forwards=0, score_update_steps=16)
    write(own, seal)
    return seal


def run_controls(authority, manifest, out, simple_root=None):
    """Create all supplemental predictions before any held-label join."""
    active = [True]
    def deny_labels(event, args):
        if active[0] and event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0]).lower()
            need('curator_roles' not in path and '/target_join/' not in path, 'CONTROL_HELD_LABEL_READ_FORBIDDEN')
    sys.addaudithook(deny_labels)
    try:
        return _run_controls(authority, manifest, out,
                             simple_root or ROOT / 'results/rc_h593_simple_explanations_v1')
    finally:
        active[0] = False


def summarize_controls(join_result, controls_output, report_path=None):
    """No label-file reads: consume the original join's audited identity rows."""
    result = read(join_result) if isinstance(join_result, (str, os.PathLike)) else join_result
    seal = (read(Path(controls_output) / 'all_predictions_prelabel_seal.json')
            if isinstance(controls_output, (str, os.PathLike)) else controls_output)
    need(result['status'] == 'INTERNAL_M_PILOT_POSTSEAL_JOIN_COMPLETE'
         and result['authority'] == seal['authority'], 'CONTROL_POSTSEAL_JOIN_AUTHORITY')
    destination = Path(seal['output_dir']); checked(seal['parameters'])
    labels = {r['query_id']: r for r in result['rows']}
    rows = []
    for binding in seal['predictions']:
        p = read(checked(binding)); r = labels[p['query_id']]
        need(r['split'] == p['split'] and p['authority'] == seal['authority'], 'CONTROL_JOIN_QUERY')
        correct = {k: v['prediction_identity'] == r['identity'] for k, v in p['methods'].items()}
        correct['RAW'] = r['correct']['RAW']
        ids = {k: v['prediction_id'] for k, v in p['methods'].items()}
        ids['RAW'] = r['selected_physical_row']['RAW']
        content_metrics = {}
        for name, prediction in p['methods'].items():
            if not name.startswith('PURE_L__'):
                continue
            targets = r['target_positions']
            need(len(targets) <= 1, 'CONTROL_JOIN_SINGLE_TARGET')
            target = targets[0] if targets else None
            l = np.asarray(prediction['L'])
            content_metrics[name] = dict(target_in_C128=bool(targets),
                target_rank=None if target is None else prediction['ranked_positions'].index(target) + 1,
                target_margin=None if target is None else float(l[target] - np.delete(l, target).max()))
        rows.append(dict(query_id=r['query_id'], split=r['split'], component=r['component'],
                         target_in_C128=r['target_in_C128'], correct=correct,
                         selected_physical_row=ids, pure_content=content_metrics))
    need(len(rows) == len(labels) == 24 and len({r['query_id'] for r in rows}) == 24, 'CONTROL_EXACT_JOIN_ROWS')
    def effect(part, method, baseline):
        rescue = sum(r['correct'][method] and not r['correct'][baseline] for r in part)
        harm = sum(r['correct'][baseline] and not r['correct'][method] for r in part)
        return dict(rescue=rescue, harm=harm, net=rescue - harm,
                    changed=sum(r['selected_physical_row'][method] != r['selected_physical_row'][baseline] for r in part))
    summary = {}
    for panel, split in (('TRAIN16', 'train'), ('PROBE8', 'probe')):
        part = [r for r in rows if r['split'] == split]; raw_correct = sum(r['correct']['RAW'] for r in part)
        models = {}
        for method in rows[0]['correct']:
            baseline = ('ADDITIVE4__FRESH_ZERO' if method.startswith('ADDITIVE4__') else
                        'PURE_L__FRESH_ZERO' if method.startswith('PURE_L__') else
                        'SCORE_UPDATE5_ZERO' if method.startswith('SCORE_UPDATE5') else 'PRODUCT5__FRESH_ZERO')
            if method == 'RAW':
                baseline = 'RAW'
            raw = effect(part, method, 'RAW')
            models[method] = dict(correct=sum(r['correct'][method] for r in part), count=len(part),
                                  vs_RAW=raw, RAW_correct_loss_rate=raw['harm'] / raw_correct if raw_correct else None,
                                  baseline=baseline, vs_baseline=effect(part, method, baseline))
        interactions = {}
        for prefix in ('PRODUCT5__', 'ADDITIVE4__', 'PURE_L__'):
            pre = effect(part, prefix + 'PRE_REAL', prefix + 'PRE_CONSTANT')
            post = effect(part, prefix + 'POST_REAL', prefix + 'POST_CONSTANT')
            interactions[prefix.rstrip('_')] = dict(pre_real_minus_constant=pre, post_real_minus_constant=post,
                position_difference_in_net_correct=pre['net'] - post['net'],
                position_difference_in_accuracy=(pre['net'] - post['net']) / len(part))
        summary[panel] = dict(count=len(part), components=len({r['component'] for r in part}),
                              natural_candidate_recall=sum(r['target_in_C128'] for r in part),
                              RAW_correct=raw_correct, models=models, position_interactions=interactions)
    output = dict(status='INTERNAL_M_CPU_CONTROLS_POSTSEAL_SUMMARY', authority=seal['authority'],
                  prelabel_seal=bind(destination / 'all_predictions_prelabel_seal.json'),
                  summary=summary, rows=rows, external_GO=False, generalization_confirmed=False,
                  limitations=['TRAIN16 is stratified fitting data; PROBE8 is opened-data development only.',
                    'SCORE_UPDATE5 matches additional 16 updates, not optimization quality or parameter count.',
                    'ADDITIVE4 is a frozen full-TRAIN readout diagnostic; adapters were trained under PRODUCT5.',
                    'Pure L can encode internal M; a gain is not proof of semantic or spatial ownership improvement.'])
    write(destination / 'summary.json', output)
    lines = ['# ColNomic 内部 M：CPU 补充对照', '',
             '原 fold0 TRAIN16 / PROBE8、原自然 RAW C128。新编码和 RoMa 次数均为零。',
             'ADDITIVE4 为完整 TRAIN 旧参数；SCORE_UPDATE5 从同一完整 TRAIN 原头出发，仅增加固定16步训练。', '',
             '|面板|方法|正确|相对 RAW 救回/误伤|相对同读出零模块净增|', '|---|---|---:|---:|---:|']
    for panel, info in summary.items():
        for method, values in info['models'].items():
            raw = values['vs_RAW']
            lines.append(f"|{panel}|{method}|{values['correct']}/{values['count']}|{raw['rescue']}/{raw['harm']}|{values['vs_baseline']['net']}|")
    lines.extend(['', '位置比较使用 REAL−CONSTANT 增量之差；完整逐图配对与纯 L 目标名次/margin 在 summary.json。',
                  '8张探针不能确认泛化；纯 L 改善仍可能来自质量写入相似度，不等于语义或空间归属证明。'])
    path = Path(report_path) if report_path else destination / 'report.md'
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text('\n'.join(lines) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--authority', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--simple-root')
    args = parser.parse_args()
    result = run_controls(args.authority, None, args.out, args.simple_root)
    print(json.dumps(dict(status=result['status'], output_dir=result['output_dir'], predictions=len(result['predictions']))))
