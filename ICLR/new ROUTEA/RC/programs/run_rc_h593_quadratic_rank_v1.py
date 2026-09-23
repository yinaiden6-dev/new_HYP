#!/usr/bin/env python3
"""Two fixed quadratic challenger-rank arms; no HOLD/SWITCH action."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import run_rc_h593_gap_curve_v1 as U
import run_rc_h593_conditional_rank_v1 as C

ROOT = U.ROOT
sys.path.insert(0, str(ROOT / 'src'))
read, write, bind, checked, need, hx = U.read, U.write, U.bind, U.checked, U.need, U.hx
OUT = ROOT / 'results/rc_h593_quadratic_rank_v1'
AUTH = ROOT / 'registry/rc_h593_quadratic_rank_authority_v1_20260921.json'
PLAN = ROOT / 'plan/RC_H593_QUADRATIC_RANK_V1_20260921.md'
REPORT = ROOT / 'reports/REPORT_H593_QUADRATIC_RANK_V1_20260921.md'
BASE = ROOT / 'results/rc_h593_conditional_rank_v1'
PARENT = BASE
CONTROLS = ('COST1_FULL', 'CE_FULL', 'RANK_ONLY6')
ARMS = ('DIAG12', 'FULL_QUAD27')
MODELS = CONTROLS + ARMS
PRIMARY = 'FULL_QUAD27'
QUANTILES = (0., .01, .1, .25, .5, .75, .9, .99, 1.)
INPUT_NAMES = ('RAW', 'S', 'M', 'L', 'Q', 'R')


def prepare():
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    prior_path = ROOT / 'registry/rc_h593_conditional_rank_authority_v1_20260921.json'
    prior = read(prior_path)
    for entry in prior['code_sources'].values():
        checked(entry)
    validated = read(BASE / 'validation.json')
    need(validated['status'] == 'CONDITIONAL_RANK_ALL_COUNTS_PASS' and
         validated['authority'] == bind(prior_path), 'PARENT_COMPLETE')
    checked(validated['result'])
    codes = dict(program=Path(__file__), shared_io_loader=Path(U.__file__),
                 conditional_helper=Path(C.__file__), plan=PLAN,
                 core=ROOT / 'src/rc_aslo_xf/h593_quadratic_rank_v1.py',
                 independent=ROOT / 'programs/validate_rc_h593_quadratic_rank_v1.py',
                 conditional_core=ROOT / 'src/rc_aslo_xf/h593_conditional_rank_v1.py',
                 conditional_independent=ROOT / 'programs/validate_rc_h593_conditional_rank_v1.py',
                 feature_formula=ROOT / 'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',
                 launcher=ROOT / 'slurm/rc_h593_quadratic_rank_v1.sbatch')
    folds = {}
    for f in range(5):
        v = read(BASE / f'fold{f}/validation.json')
        need(v['status'] == 'CONDITIONAL_RANK_FRESH_REPLAY_NUMPY_PASS' and
             v['authority'] == bind(prior_path), 'BASE_VALIDATED')
        p = read(checked(v['payload']))
        need(p['authority'] == bind(prior_path) and p['fold'] == f, 'PARENT_FOLD_SEAL')
        folds[str(f)] = dict(base_payload=v['payload'],
                            base_validation=bind(BASE / f'fold{f}/validation.json'),
                            train_roles=prior['fold_sources'][str(f)]['train_roles'])
    write(AUTH, dict(status='H593_QUADRATIC_RANK_AUTHORIZED', parent_authority=bind(prior_path),
          code_sources={k: bind(v) for k, v in codes.items()},
          public_sources=prior['public_sources'], features=prior['features'], fold_sources=folds,
          join_sources=dict(parent_result=validated['result'], parent_validation=bind(BASE / 'validation.json'),
                            curator=prior['join_sources']['curator']),
          arms=list(ARMS), primary=PRIMARY, primary_control='RANK_ONLY6', strong_control='CE_FULL',
          secondary_arm='DIAG12', historical_control='COST1_FULL', arm_selection=False,
          parameter_counts=dict(DIAG12=12, FULL_QUAD27=27), new_common_bias=0, precision='float64',
          feature_expansion=dict(original_inputs=list(INPUT_NAMES), diagonal='original6 then x_i*x_i i=0..5',
              full='original6 then x_i*x_j in lexicographic i=0..5,j=i..5',
              standardized=False, centered=False, clipped=False, new_evidence=False),
          optimizer=dict(name='AdamW', lr=.03, weight_decay=.001, steps=2000, initialization='zero',
                         betas=[.9,.999], eps=1e-8, foreach=False, fused=False),
          objective='sum conditional127 CE on RAWwrong TRAIN divided by ALL recall-present TRAIN count',
          primary_signal_rule='FULL_QUAD27 target-top count >104 and positive equal-component oracle difference vs RANK_ONLY6 and CE_FULL',
          evidence_level='Opened H593 grouped fivefold development ranking-only experiment',
          action_calibration=False, deployment_change=False, encoder_forwards=0))
    print(json.dumps(dict(status='PREPARED', authority=bind(AUTH))), flush=True)


def guard(stage, fold=None):
    a = read(AUTH)
    checked(a['parent_authority'])
    for b in a['code_sources'].values():
        checked(b)
    need(a['code_sources']['program'] == bind(__file__), 'PROGRAM_SHA')
    if stage not in ('preflight', 'publish'):
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allow = {Path(b['path']).resolve() for b in a['public_sources'].values()}
    for sources in a['features']:
        allow.update(Path(b['path']).resolve() for b in sources.values())
    if stage in ('fit', 'verify'):
        need(fold in range(5), 'FOLD_RANGE')
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join', 'join-verify', 'publish'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for sources in a['fold_sources'].values():
            allow.update(Path(b['path']).resolve() for b in sources.values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve(); s = str(path).lower()
        need(not any(t in s for t in ('rc_opened_', 'd1-mi', 'd1_mi', 'formal392', '/grozi/', '/target_join/')), 'PROTECTED_READ')
        if 'curator_roles' in s:
            need(stage in ('join', 'join-verify', 'publish'), 'NO_OUTER_LABELS_BEFORE_SEALS')
        if ROOT / 'reports' in path.parents:
            need(stage == 'publish' and path == REPORT, 'NO_REPORT_ACCESS')
        if ROOT / 'results' in path.parents:
            own = OUT in path.parents
            if own and stage in ('fit', 'verify', 'preflight'):
                part = path.relative_to(OUT).parts[0]
                own = part in ('preflight.json', '.preflight.json.tmp', f'fold{fold}')
            need(own or path in allow, 'UNLISTED_RESULT:' + s)
    sys.addaudithook(audit)
    if stage != 'preflight':
        p = read(OUT / 'preflight.json')
        need(p['authority'] == bind(AUTH) and p['status'] == 'QUADRATIC_RANK_SYNTHETIC_PASS', 'PREFLIGHT')
    return a


def features(a):
    return C.features(a)


def target(r, identity, labels):
    return C.target(r, identity, labels)


def train_inputs(a, fold):
    import torch
    rows, labels = features(a)
    split = read(checked(a['public_sources']['split']))['folds'][fold]
    roles = {r['query_id']: r for r in read(checked(a['fold_sources'][str(fold)]['train_roles']))['records']}
    train_ids, held_ids = set(split['train_query_ids']), set(split['heldout_query_ids'])
    need(set(roles) == train_ids and not train_ids & held_ids, 'TRAIN_ONLY_ROLES')
    train = [r for r in rows if r['query_id'] in train_ids]
    held = [r for r in rows if r['query_id'] in held_ids]
    need(not {r['source_image_sha256'] for r in train} & {r['source_image_sha256'] for r in held}, 'OUTER_SOURCE_DISJOINT')
    keep = [r for r in train if target(r, roles[r['query_id']]['identity'], labels) >= -1]
    y = torch.tensor([target(r, roles[r['query_id']]['identity'], labels) for r in keep], dtype=torch.long)
    x = torch.stack([r['modes']['REAL']['X'] for r in keep])
    sources = a['fold_sources'][str(fold)]
    old = read(checked(sources['base_payload']))
    validation = read(checked(sources['base_validation']))
    need(validation['payload'] == sources['base_payload'] and validation['status'] == 'CONDITIONAL_RANK_FRESH_REPLAY_NUMPY_PASS', 'BASE_SEALS')
    need(validation['authority'] == old['authority'] == a['parent_authority'] and
         old['fold'] == fold and old['status'] == 'CONDITIONAL_RANK_FOLD_SEALED', 'PARENT_AUTHORITY_FOLD')
    need(set(old['train_query_ids']) == train_ids, 'SAME_TRAIN_POPULATION')
    return train, held, keep, x, y, roles, labels, old


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    need(values.size > 0 and np.isfinite(values).all(), 'FINITE_NONEMPTY_DIAGNOSTIC')
    return list(map(float, np.quantile(values, QUANTILES)))


def magnitude_table(design, names):
    values = design.detach().cpu().numpy().reshape(-1, len(names))
    need(np.isfinite(values).all(), 'FINITE_TRAIN_BASIS')
    return dict(row_count=len(values), dimension=len(names), finite=True,
                quantile_probabilities=list(QUANTILES),
                columns=[dict(name=name, signed_quantiles=quantiles(values[:, j]),
                              absolute_quantiles=quantiles(abs(values[:, j])),
                              zero_fraction=float(np.mean(values[:, j] == 0.0)))
                         for j, name in enumerate(names)])


def training_magnitudes(x):
    from rc_aslo_xf.h593_quadratic_rank_v1 import expand, PAIRS
    tables = dict(original6=magnitude_table(x, INPUT_NAMES))
    for mode in ARMS:
        pairs = [(i, i) for i in range(6)] if mode == 'DIAG12' else PAIRS
        names = list(INPUT_NAMES) + [f'{INPUT_NAMES[i]}*{INPUT_NAMES[j]}' for i, j in pairs]
        tables[mode] = magnitude_table(expand(x, mode), names)
    return dict(scope='All recall-present TRAIN rows, all 127 challengers; no heldout data',
                train_present_queries=len(x), fitted_scaling=False, standardized=False,
                centered=False, clipped=False, tables=tables)


def scores(x, parameters, mode):
    import torch
    from rc_aslo_xf.h593_quadratic_rank_v1 import expand
    theta = torch.tensor([float.fromhex(v) for v in parameters], dtype=torch.float64)
    if mode in ARMS:
        return expand(x, mode) @ theta
    return x @ theta[:6] + (theta[6] if len(theta) == 7 else 0.0)


def train_metrics(z, y):
    import torch
    from rc_aslo_xf.h593_conditional_rank_v1 import loss_terms
    joint, rank, action = loss_terms(z, y)
    positive = y >= 0
    result = dict(joint128_CE=float(joint), conditional_rank_loss_total_N=float(rank),
                  action_loss=float(action), target_top=int(((z.argmax(1) == y) & positive).sum()),
                  ranking_positive_count=int(positive.sum()))
    if bool(positive.any()):
        zz, yy = z[positive], y[positive]
        target_score = zz.gather(1, yy[:, None]).squeeze(1)
        mask = torch.zeros_like(zz, dtype=torch.bool).scatter_(1, yy[:, None], True)
        margin = target_score - zz.masked_fill(mask, -torch.inf).amax(1)
        indices = torch.arange(127)[None, :]
        ranks = (((zz > target_score[:, None]) |
                  ((zz == target_score[:, None]) & (indices < yy[:, None]))).sum(1) + 1)
        result.update(challenger_MRR=float((1.0 / ranks.to(torch.float64)).mean()),
                      target_margin_quantiles=quantiles(margin.numpy()),
                      hard_softplus_loss_total_N=float(torch.logaddexp(torch.zeros_like(margin), -margin).sum() / len(y)))
    else:
        result.update(challenger_MRR=None, target_margin_quantiles=None, hard_softplus_loss_total_N=0.0)
    return result


def compute(a, fold):
    import torch
    from rc_aslo_xf.h593_quadratic_rank_v1 import fit_rank
    train, held, keep, x, y, roles, labels, old = train_inputs(a, fold)
    need(set(old['parameters']) == set(CONTROLS), 'THREE_PARENT_CONTROLS')
    positive_ids = [r['query_id'] for r, value in zip(keep, y) if int(value) >= 0]
    need(old['effective_train_query_ids'] == [r['query_id'] for r in keep] and
         old['ranking_positive_query_ids'] == positive_ids, 'SAME_EFFECTIVE_TRAIN_AND_DENOMINATOR')
    params = {name: old['parameters'][name] for name in CONTROLS}
    magnitudes = training_magnitudes(x)
    for mode in ARMS:
        params[mode] = hx(fit_rank(x, y, mode))
    metrics = {name: train_metrics(scores(x, par, name), y) for name, par in params.items()}
    for mode in CONTROLS:
        need(all(metrics[mode][key] == value for key, value in old['train_metrics'][mode].items()),
             'PARENT_TRAIN_METRICS_EXACT')
    xx = torch.stack([r['modes']['REAL']['X'] for r in held])
    oldp = {r['query_id']: r for r in old['predictions']}
    need(set(oldp) == {r['query_id'] for r in held}, 'SAME_HELDOUT_AXIS')
    zs = {name: scores(xx, par, name) for name, par in params.items()}
    predictions = []
    for i, r in enumerate(held):
        for key in ('execution_ordinal', 'winner', 'candidate_physical_rows', 'challenger_positions'):
            need(r[key] == oldp[r['query_id']][key], 'PARENT_CANDIDATE_AXIS_PARITY')
        models = {}
        for name, z in zs.items():
            values, top = hx(z[i]), int(z[i].argmax())
            model = dict(logits_hex=values, top_index=top,
                         top_physical=r['candidate_physical_rows'][r['challenger_positions'][top]])
            if name in CONTROLS:
                need(model == oldp[r['query_id']]['models'][name], 'ALL_CONTROL_LOGITS_BIT_PARITY')
            models[name] = model
        predictions.append(dict(query_id=r['query_id'], execution_ordinal=r['execution_ordinal'],
                           winner=r['winner'], candidate_physical_rows=r['candidate_physical_rows'],
                           challenger_positions=r['challenger_positions'], models=models))
    return dict(status='QUADRATIC_RANK_FOLD_SEALED', authority=bind(AUTH), fold=fold,
                parent_payload=a['fold_sources'][str(fold)]['base_payload'], parameters=params,
                train_query_ids=[r['query_id'] for r in train],
                effective_train_query_ids=[r['query_id'] for r in keep],
                ranking_positive_query_ids=positive_ids, train_counts=dict(all=len(train), present=len(keep),
                ranking_positive=len(positive_ids), raw_correct=len(keep)-len(positive_ids),
                excluded_target_absent=len(train)-len(keep),
                positive_identities=len({roles[q]['identity'] for q in positive_ids}),
                positive_components=len({roles[q]['component'] for q in positive_ids})),
                train_metrics=metrics, train_feature_magnitudes=magnitudes, predictions=predictions,
                heldout_label_reads=0, training_updates_per_arm={mode: 2000 if positive_ids else 0 for mode in ARMS},
                empty_positive_training=not bool(positive_ids), control_training_updates=0,
                action_trained=False, new_accuracy_result=False)


def fit(a, fold, replay=False):
    import torch
    from rc_aslo_xf.h593_quadratic_rank_v1 import expand, conditional_loss
    from validate_rc_h593_quadratic_rank_v1 import basis, independent_loss_gradient, validate_logits
    from validate_rc_h593_conditional_rank_v1 import validate_logits as linear_logits
    folder = OUT / f'fold{fold}'
    if not replay and (folder / 'validation.json').exists():
        v = read(folder / 'validation.json')
        need(v['status'] == 'QUADRATIC_RANK_FRESH_REPLAY_NUMPY_PASS' and
             v['authority'] == bind(AUTH) and v['payload'] == bind(folder / 'payload.json'), 'RESUME_VALIDATED_FOLD')
        return
    started = time.monotonic()
    payload = compute(a, fold)
    if not replay:
        write(folder / 'payload.json', payload)
        write(folder / 'runtime.json', dict(job_id=os.environ['SLURM_JOB_ID'],
              fit_and_prediction_seconds=time.monotonic()-started, natural_new_fits=len(ARMS),
              fitted_arms=list(ARMS)))
        subprocess.run([sys.executable, __file__, 'verify', '--fold', str(fold)], check=True)
        print(json.dumps(dict(event='FOLD_VALIDATED', fold=fold, train=payload['train_counts'])), flush=True)
        return
    need(payload == read(folder / 'payload.json'), 'FRESH_FIT_PARAMETERS_PREDICTIONS_REPLAY')
    train, held, keep, x, y, roles, labels, old = train_inputs(a, fold)
    xx = torch.stack([r['modes']['REAL']['X'] for r in held]).numpy()
    checks, gradients, bases = {}, {}, {}
    for name in MODELS:
        params = payload['parameters'][name]
        saved = [r['models'][name]['logits_hex'] for r in payload['predictions']]
        if name in ARMS:
            checks[name] = validate_logits(xx, params, saved, name)
        elif len(params) == 6:
            checks[name] = linear_logits(xx, params, saved)
        else:
            t = np.array([float.fromhex(v) for v in params])
            z = np.sum(xx*t[:6], axis=2) + t[6]
            want = np.array([[float.fromhex(v) for v in row] for row in saved])
            error = float(np.max(abs(z-want)))
            need(error < 2e-10 and np.array_equal(z.argmax(1), want.argmax(1)), 'CONTROL_NUMPY_LOGITS')
            checks[name] = dict(logit_count=int(z.size), max_abs_error=error)
    for mode in ARMS:
        design = expand(x, mode)
        need(np.array_equal(design.numpy(), basis(x.numpy(), mode)), 'TRAIN_BASIS_NUMPY_EXACT')
        need(np.array_equal(expand(torch.from_numpy(xx), mode).numpy(), basis(xx, mode)), 'HELDOUT_BASIS_NUMPY_EXACT')
        bases[mode] = dict(train_basis_exact=True, heldout_basis_exact=True, original6_exact=True,
                           feature_dimension=int(design.shape[-1]))
        theta = np.array([float.fromhex(v) for v in payload['parameters'][mode]])
        loss, gradient = independent_loss_gradient(theta, x.numpy(), y.numpy(), mode)
        t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        rank = conditional_loss(design @ t, y)
        tg = torch.autograd.grad(rank, t)[0].detach().numpy()
        loss_error, gradient_error = abs(float(rank.detach())-loss), float(np.max(abs(tg-gradient)))
        need(loss_error < 2e-10 and gradient_error < 2e-10, 'INDEPENDENT_LOSS_GRADIENT')
        gradients[mode] = dict(rank_loss_error=loss_error, rank_gradient_max_error=gradient_error)
    write(folder / 'validation.json', dict(status='QUADRATIC_RANK_FRESH_REPLAY_NUMPY_PASS',
          authority=bind(AUTH), payload=bind(folder / 'payload.json'), heldout_label_reads=0,
          independent_logit_checks=checks, independent_basis_checks=bases,
          independent_loss_gradient_checks=gradients, both_arms_fresh_fit_exact=True))


def row_losses(z, target_index):
    if target_index == -2:
        return dict(conditional_CE=None, hard_softplus=None, target_margin=None, action=None, joint128_CE=None)
    values = np.asarray(z, dtype=np.float64)
    maximum = float(values.max())
    logmass = float(np.log(np.exp(values-maximum).sum()))
    aggregate = maximum + logmass
    if target_index == -1:
        action = float(np.logaddexp(0., aggregate))
        return dict(conditional_CE=0., hard_softplus=0., target_margin=None, action=action, joint128_CE=action)
    wrong = max(v for i, v in enumerate(z) if i != target_index)
    margin = float(z[target_index] - wrong)
    rank = float(logmass - (z[target_index]-maximum))
    action = float(np.logaddexp(0., -aggregate))
    return dict(conditional_CE=rank, hard_softplus=float(np.logaddexp(0., -margin)),
                target_margin=margin, action=action, joint128_CE=rank+action)


def compare(rows, baseline, new):
    eligible = [r for r in rows if r['ranking_eligible']]
    groups = defaultdict(list)
    for row in rows:
        groups[row['component']].append(int(row['raw_or_top_oracle'][new])-int(row['raw_or_top_oracle'][baseline]))
    need(len(groups) == 64, '64_COMPONENTS')
    differences = np.array([np.mean(values) for _, values in sorted(groups.items())])
    rng = np.random.default_rng(20260920)
    boot = differences[rng.integers(0, 64, size=(10000, 64))].mean(1)
    gained = [r['query_id'] for r in eligible if r['target_is_top'][new] and not r['target_is_top'][baseline]]
    lost = [r['query_id'] for r in eligible if not r['target_is_top'][new] and r['target_is_top'][baseline]]
    return dict(baseline=baseline, new=new, ranking_gained=len(gained), ranking_lost=len(lost),
                ranking_net=len(gained)-len(lost), gained_query_ids=gained, lost_query_ids=lost,
                equal_component_oracle_difference=float(differences.mean()),
                bootstrap95=list(map(float, np.quantile(boot, [.025,.975]))))


def join(a, replay=False):
    payloads, seals = [], []
    for fold in range(5):
        folder = OUT / f'fold{fold}'
        validation = read(folder / 'validation.json')
        need(validation['status'] == 'QUADRATIC_RANK_FRESH_REPLAY_NUMPY_PASS' and
             validation['authority'] == bind(AUTH) and validation['both_arms_fresh_fit_exact'], 'ALL_FOLDS_VALIDATED')
        payload = read(checked(validation['payload']))
        need(payload['fold'] == fold and payload['authority'] == bind(AUTH), 'FOLD_BINDING')
        payloads.append(payload); seals.append(bind(folder / 'validation.json'))
    write(OUT / 'all_predictions_prelabel_seal.json', dict(authority=bind(AUTH), validations=seals))
    roles = {r['query_id']: r for r in read(checked(a['join_sources']['curator']))['records']}
    prior = read(checked(a['join_sources']['parent_result']))
    need(read(checked(a['join_sources']['parent_validation']))['result'] == a['join_sources']['parent_result'], 'PARENT_JOIN_SEAL')
    oldrows = {r['query_id']: r for r in prior['rows']}
    worker = {r['query_id']: r for r in read(checked(a['public_sources']['worker']))['records']}
    need(len(worker) == 593 and all('source_image_sha256' in r for r in worker.values()), 'WORKER_IMAGE_SHA_SCHEMA')
    gallery = read(checked(a['public_sources']['gallery']))
    labels = {r['physical_row']: r['identity'] for r in gallery['records']}
    rows = []
    for payload in payloads:
        training = read(checked(a['fold_sources'][str(payload['fold'])]['train_roles']))['records']
        need(set(payload['train_query_ids']) == {r['query_id'] for r in training}, 'JOIN_TRAIN_QUERY_AXIS')
        train_image_hashes = {worker[q]['source_image_sha256'] for q in payload['train_query_ids']}
        for prediction in payload['predictions']:
            q = prediction['query_id']; role = roles[q]
            need(role['outer_fold'] == payload['fold'], 'OUTER_FOLD')
            for key in ('identity', 'component'):
                need(role[key] not in {r[key] for r in training}, 'OUTER_GROUP_IMAGE_DISJOINT')
            need(worker[q]['source_image_sha256'] not in train_image_hashes, 'OUTER_SOURCE_IMAGE_DISJOINT')
            axis, identity = prediction['candidate_physical_rows'], role['identity']
            raw = labels[axis[prediction['winner']]] == identity
            present = any(labels[v] == identity for v in axis)
            need(raw == oldrows[q]['RAW_correct'] and present == oldrows[q]['target_present'], 'BASE_MEMBERSHIP_PARITY')
            top, ranks, selected, losses = {}, {}, {}, {}
            target_index = target(prediction, identity, labels)
            for name, model in prediction['models'].items():
                z = [float.fromhex(v) for v in model['logits_hex']]
                need(len(z) == 127 and all(np.isfinite(z)), 'FINITE_ALL127')
                order = sorted(range(127), key=lambda j: (-z[j], j))
                physical = axis[prediction['challenger_positions'][order[0]]]
                need(physical == model['top_physical'] and order[0] == model['top_index'], 'TOP_RECOMPUTED')
                selected[name], top[name] = physical, labels[physical] == identity
                ranks[name] = None if target_index < 0 else order.index(target_index)+1
                losses[name] = row_losses(z, target_index)
                if name in CONTROLS:
                    need(top[name] == oldrows[q]['target_is_top'][name] and
                         ranks[name] == oldrows[q]['target_challenger_rank'][name] and
                         selected[name] == oldrows[q]['top_physical'][name], 'PARENT_SORTING_PARITY')
            rows.append(dict(query_id=q, original_query_id=role['original_query_id'], component=role['component'],
                             fold=payload['fold'], RAW_correct=raw, target_present=present,
                             ranking_eligible=not raw and present, target_is_top=top,
                             target_challenger_rank=ranks, top_physical=selected, ranking_losses=losses,
                             raw_or_top_oracle={m: raw or top[m] for m in MODELS}))
    rows.sort(key=lambda r: r['query_id'])
    need(len(rows) == len({r['query_id'] for r in rows}) == 593, 'ALL593_JOIN')
    need([sum(r['fold'] == f for r in rows) for f in range(5)] == [119,118,119,119,118], 'FOLD_SIZES')
    eligible = [r for r in rows if r['ranking_eligible']]
    present_rows = [r for r in rows if r['target_present']]
    need(len(eligible) == 144 and sum(r['RAW_correct'] for r in rows) == 426 and len(present_rows) == 570, 'ALL_POPULATIONS')
    summary = {}
    for mode in MODELS:
        margins = [r['ranking_losses'][mode]['target_margin'] for r in eligible]
        summary[mode] = dict(target_top_among144=sum(r['target_is_top'][mode] for r in eligible),
             raw_or_top_oracle_among593=sum(r['raw_or_top_oracle'][mode] for r in rows),
             by_fold={str(f): sum(r['target_is_top'][mode] for r in eligible if r['fold'] == f) for f in range(5)},
             conditional_CE_per_present570=sum(r['ranking_losses'][mode]['conditional_CE'] for r in present_rows)/570,
             conditional_CE_per_positive144=sum(r['ranking_losses'][mode]['conditional_CE'] for r in eligible)/144,
             hard_loss_per_present570=sum(r['ranking_losses'][mode]['hard_softplus'] for r in present_rows)/570,
             joint128_CE_per_present570=sum(r['ranking_losses'][mode]['joint128_CE'] for r in present_rows)/570,
             action_loss_per_present570=sum(r['ranking_losses'][mode]['action'] for r in present_rows)/570,
             challenger_MRR_among144=float(np.mean([1/r['target_challenger_rank'][mode] for r in eligible])),
             target_margin_quantiles_among144=quantiles(margins), target_margin_mean_among144=float(np.mean(margins)))
    for mode, count in [('COST1_FULL',104), ('CE_FULL',104), ('RANK_ONLY6',103)]:
        need(summary[mode]['target_top_among144'] == count and
             summary[mode]['raw_or_top_oracle_among593'] == count+426, 'FROZEN_SORTING_BASELINES')
    comparisons = {mode: {baseline: compare(rows, baseline, mode) for baseline in CONTROLS} for mode in ARMS}
    interaction_vs_diagonal = compare(rows, 'DIAG12', PRIMARY)
    old40 = [r for r in eligible if not r['target_is_top']['COST1_FULL']]
    need(len(old40) == 40, 'ORIGINAL40_SORTING_FAILURES')
    original40 = {mode: dict(rescued=sum(r['target_is_top'][mode] for r in old40),
                            still_wrong=sum(not r['target_is_top'][mode] for r in old40),
                            lost_original104=sum(not r['target_is_top'][mode] for r in eligible if r['target_is_top']['COST1_FULL']))
                  for mode in ARMS}
    signal = summary[PRIMARY]['target_top_among144'] > 104 and all(
        comparisons[PRIMARY][baseline]['equal_component_oracle_difference'] > 0
        for baseline in ('RANK_ONLY6','CE_FULL'))
    output = dict(status='H593_QUADRATIC_RANK_ALL593_COMPLETE', authority=bind(AUTH), primary=PRIMARY,
                  primary_control='RANK_ONLY6', secondary_diagnostic='DIAG12', best_arm_selected=False,
                  population=593, target_present=570, target_absent=23, ranking_eligible=144,
                  quantile_probabilities=list(QUANTILES), summary=summary, comparisons=comparisons,
                  fixed_interaction_vs_diagonal=interaction_vs_diagonal,
                  original40_vs_COST1=original40, positive_internal_ranking_signal=bool(signal),
                  train_counts={str(p['fold']):p['train_counts'] for p in payloads},
                  train_metrics={str(p['fold']):p['train_metrics'] for p in payloads},
                  train_feature_magnitudes={str(p['fold']):p['train_feature_magnitudes'] for p in payloads},
                  rows=rows, source=a['join_sources']['parent_result'], evidence_level=a['evidence_level'],
                  new_deployed_accuracy=None, action_trained=False, external_GO=False,
                  limits=['RAW-or-top is oracle availability, not deployed prediction accuracy.',
                          'No best-arm selection: FULL_QUAD27 is the predetermined primary; DIAG12 is a fixed secondary diagnostic.',
                          'All quadratic products are unnormalized; fixed AdamW regularization therefore depends on feature units.',
                          'Feature magnitudes describe TRAIN only and do not select scaling, clipping or hyperparameters.',
                          'Family nesting does not guarantee ordering after finite AdamW training.',
                          'Component bootstrap fixes current predictions and omits shared-training and adaptive-development uncertainty.',
                          'Existing full593 accuracy remains unchanged until a separately trained and evaluated action is available.'])
    if replay:
        need(output == read(OUT / 'result.json'), 'FRESH_JOIN_EXACT_REPLAY')
        write(OUT / 'validation.json', dict(status='QUADRATIC_RANK_ALL_COUNTS_PASS', authority=bind(AUTH),
              result=bind(OUT / 'result.json'), fold_validations=seals))
    else:
        write(OUT / 'result.json', output)
        subprocess.run([sys.executable, __file__, 'join-verify'], check=True)
        print(json.dumps(dict(status='COMPLETE', summary=summary, comparisons=comparisons,
                             positive_internal_ranking_signal=signal)), flush=True)


def publish():
    validation = read(OUT / 'validation.json')
    need(validation['status'] == 'QUADRATIC_RANK_ALL_COUNTS_PASS', 'RESULT_VALIDATED')
    result = read(checked(validation['result']))
    lines = ['# H593：固定二次特征条件排序实验', '',
             '同六维原始证据、自然RAW C128、五折分组OOF、条件127候选CE。FULL_QUAD27为预定主臂，DIAG12为固定次要诊断；不按结果选择最佳臂。无新HOLD/SWITCH决策。', '',
             '| 排序头 | 144例中target第一 | 候选MRR／144 | 条件CE／570 | RAW或最高挑战者可覆盖／593（oracle） |',
             '|---|---:|---:|---:|---:|']
    for mode, stat in result['summary'].items():
        lines.append(f"| {mode} | {stat['target_top_among144']} | {stat['challenger_MRR_among144']:.6f} | {stat['conditional_CE_per_present570']:.6f} | {stat['raw_or_top_oracle_among593']} |")
    lines += ['', '| 新臂／对照 | 排序救回 | 排序损失 | 净增 | 组件等权差 | 95%区间 |', '|---|---:|---:|---:|---:|---|']
    for mode, entries in result['comparisons'].items():
        for baseline, stat in entries.items():
            lines.append(f"| {mode} / {baseline} | {stat['ranking_gained']} | {stat['ranking_lost']} | {stat['ranking_net']} | {stat['equal_component_oracle_difference']:.6f} | {stat['bootstrap95']} |")
    stat = result['fixed_interaction_vs_diagonal']
    lines.append(f"| FULL_QUAD27 / DIAG12（固定交互诊断） | {stat['ranking_gained']} | {stat['ranking_lost']} | {stat['ranking_net']} | {stat['equal_component_oracle_difference']:.6f} | {stat['bootstrap95']} |")
    lines += ['', f"预定主臂内部排序信号：{result['positive_internal_ranking_signal']}。要求FULL_QUAD27超过104，并对RANK_ONLY6及CE都有正的等组件方向。", '',
              '原COST1的40例排序失败完整核算如下；这些错例没有用于选择训练集或手设规则。']
    for mode, stat in result['original40_vs_COST1'].items():
        lines.append(f"- {mode}：40例中救回{stat['rescued']}，仍错{stat['still_wrong']}；原104例中损失{stat['lost_original104']}。")
    lines += ['', 'TRAIN原六维及扩展坐标的逐维有符号／绝对值分位数均保存在result.json。未标准化、裁剪或拟合尺度；单位可能影响固定正则化和优化，不把函数族嵌套等同于有限步AdamW的最优性。', '',
              '原最终准确率COST1=481/593、CE=486/593、GAP_BIAS2=492/593保留。oracle列不是新准确率；23例候选缺失保留在593分母。分组区间未计入反复开发选择或共享训练不确定性。', '',
              '结果：`results/rc_h593_quadratic_rank_v1/result.json`；独立核算及重放：各fold/validation.json与汇总validation.json。', '']
    REPORT.write_text('\n'.join(lines))
    print(REPORT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','preflight','fit-all','fit','verify','join','join-verify','publish'))
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'fit-all':
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
        for fold in range(5):
            subprocess.run([sys.executable, __file__, 'fit', '--fold', str(fold)], check=True)
        subprocess.run([sys.executable, __file__, 'join'], check=True)
    else:
        authority = guard(args.stage, args.fold)
        if args.stage == 'publish':
            publish()
        else:
            import torch
            torch.set_num_threads(8); torch.set_num_interop_threads(1)
            if args.stage == 'preflight':
                from rc_aslo_xf.h593_quadratic_rank_v1 import self_test
                from validate_rc_h593_quadratic_rank_v1 import self_test as independent_test
                write(OUT / 'preflight.json', dict(status='QUADRATIC_RANK_SYNTHETIC_PASS',
                      authority=bind(AUTH), core=self_test(), independent=independent_test(), natural_updates=0))
                print('QUADRATIC_RANK_SYNTHETIC_PASS', flush=True)
            elif args.stage in ('fit','verify'):
                fit(authority, args.fold, args.stage == 'verify')
            else:
                join(authority, args.stage == 'join-verify')
