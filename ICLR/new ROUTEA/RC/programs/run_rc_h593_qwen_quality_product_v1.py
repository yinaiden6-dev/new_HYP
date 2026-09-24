#!/usr/bin/env python3
"""One extra M*L feature on the sealed QWEN_ML5 grouped calibration.

Only the ten new six-parameter fits are trained. Parent heads remain immutable.
The product is formed in the candidate domain before taking a symmetric gap.
"""
import argparse
import copy
import fcntl
import io
import os
from pathlib import Path
import sys
import time

import run_rc_h593_qwen_quality_joint_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_qwen_quality_product_v1'
AUTH = ROOT / 'registry/rc_h593_qwen_quality_product_authority_v1_20260924.json'
PLAN = ROOT / 'plan/RC_H593_QWEN_QUALITY_PRODUCT_V1_20260924.md'
SLURM = ROOT / 'slurm/rc_h593_qwen_quality_product_v1.sbatch'
PARENT_AUTH = parent.AUTH
PARENT_OUT = parent.OUT
ARM = 'QWEN_ML_PRODUCT6'
LOSSES = ('CE', 'COST1')
COLUMNS = ['raw_standardized_delta', 'qwen_standardized_delta',
           'sym_mass_delta', 'sym_free_content_delta', 'sym_mass_times_free_content_delta']
read, sha, bind, checked, write = parent.read, parent.sha, parent.bind, parent.checked, parent.write


def product_column(row):
    import numpy as np
    m = np.asarray(row['mass'], dtype=np.float64)
    l = np.asarray(row['free_content'], dtype=np.float64)
    c, w = row['challenger_positions'], row['winner']
    assert m.shape == l.shape == (128,)
    assert np.isfinite(m).all() and np.isfinite(l).all()
    assert (m >= 0).all() and (m <= 1).all()
    return parent.sym(m[c]*l[c], m[w]*l[w])


def matrix(rows):
    import numpy as np
    result = []
    for row in rows:
        x = np.asarray(row['X'], dtype=np.float64)
        assert x.shape == (127, 4) and np.isfinite(x).all()
        result.append(np.column_stack((x, product_column(row))))
    return np.ascontiguousarray(result, dtype=np.float64)


def order_from_logits(row, logits):
    import numpy as np
    z = np.asarray(logits, dtype=np.float64)
    assert z.shape == (127,) and np.isfinite(z).all()
    full = np.zeros(128, dtype=np.float64)
    full[row['challenger_positions']] = z
    winner = row['winner']
    return sorted(range(128), key=lambda i: (-full[i], i != winner, i))


def explicit_logits(x, theta):
    import numpy as np
    return np.asarray([sum(float(a)*float(b) for a, b in zip(row, theta[:-1]))
                       + float(theta[-1]) for row in x], dtype=np.float64)


def prepare():
    import numpy as np
    import run_rc_h593_qwen3_rerank_v2_layout as q
    previous = read(PARENT_AUTH)
    for source in previous['sources']:
        checked(source)
    pv_path = PARENT_OUT/'validation.json'
    pv = read(pv_path)
    assert pv['status'] == 'QWEN_QUALITY_593_AXES_BASELINE_PARITY_SEALS_COUNTS_PASS'
    assert pv['authority'] == bind(PARENT_AUTH)
    checked(pv['result'])  # Hash only: held correctness is not opened here.
    cache = read(checked(previous['cache']))
    assert cache['labels_included'] is False and cache['columns'] == COLUMNS[:4]
    assert len(cache['rows']) == 593
    seen = set()
    feature_error = 0.
    for row in cache['rows']:
        assert row['query_id'] not in seen
        seen.add(row['query_id'])
        axis, w, c = row['candidate_physical_rows'], row['winner'], row['challenger_positions']
        assert len(set(axis)) == 128 and c == [i for i in range(128) if i != w]
        order = sorted(range(128), key=lambda i: (-row['raw_scores'][i], axis[i]))
        assert order[0] == w
        m, l = np.asarray(row['mass']), np.asarray(row['free_content'])
        x = np.asarray(row['X'], dtype=np.float64)
        err = float(np.max(np.abs(x[:, 2:4]-np.column_stack(
            (parent.sym(m[c], m[w]), parent.sym(l[c], l[w]))))))
        assert err < 2e-10
        feature_error = max(feature_error, err)
        augmented = matrix([row])[0]
        assert np.array_equal(augmented[:, :4], x)
        independent = np.asarray([(float(m[i])*float(l[i])-float(m[w])*float(l[w])) /
                                  (abs(float(m[i])*float(l[i]))+abs(float(m[w])*float(l[w]))+1e-12)
                                  for i in c])
        assert np.array_equal(augmented[:, 4], independent)
    folds = {}
    for fold in range(5):
        validation_path = PARENT_OUT/f'fold{fold}/validation.json'
        v = read(validation_path)
        assert v['status'] == 'QWEN_QUALITY_FOLD_BASELINE_AND_NUMPY_TORCH_PASS'
        assert v['authority'] == bind(PARENT_AUTH) and v['fold'] == fold
        checked(v['payload'])
        old = previous['folds'][str(fold)]
        folds[str(fold)] = {key: old[key] for key in ('train_roles', 'train_query_ids', 'effective_train_query_ids')}
        folds[str(fold)].update(parent_validation=bind(validation_path), parent_payload=v['payload'])
    a = dict(status='H593_QWEN_QUALITY_PRODUCT6_PREDECLARED',
             parent_authority=bind(PARENT_AUTH), parent_validation=bind(pv_path),
             parent_result=pv['result'],
             sources=[bind(Path(__file__)), bind(PLAN), bind(SLURM),
                      bind(Path(parent.__file__)), bind(Path(q.__file__))],
             parent_source_bindings=previous['sources'], cache=previous['cache'],
             split=previous['split'], gallery=previous['gallery'], curator=previous['curator'],
             folds=folds, arm=ARM, columns=COLUMNS, parameter_count=6,
             product_definition='sym(M_g*L_g, M_w*L_w); not sym(M_g,M_w)*sym(L_g,L_w)',
             losses=list(LOSSES), primary='CE_'+ARM, primary_comparator='CE_QWEN_ML5',
             secondary='COST1_'+ARM, secondary_comparator='COST1_QWEN_ML5',
             optimizer=previous['optimizer'], action=previous['action'],
             baseline_arms=parent.ARMS, baseline_retraining=False, new_fits=10,
             evidence='Previously opened H593, same original grouped OOF5 and natural ColNomic C128; exploratory extension, not fresh external confirmation',
             no_new_gpu_forwards=True, no_held_threshold_selection=True,
             no_other_fold_trained_logits_in_features=True)
    assert a['optimizer'] == dict(name='AdamW', lr=.03, weight_decay=.001, steps=2000,
                                 zero_initialization=True, dtype='float64', threads=8, seed=17)
    write(AUTH, a)
    write(OUT/'preflight.json', dict(status='QWEN_PRODUCT6_FEATURES_AND_PARENT_SEALS_PASS',
                                   authority=bind(AUTH), cache=a['cache'], queries=593,
                                   candidates=128, additional_columns=1, new_fits=10,
                                   parent_columns_unchanged=True, product_formula_exact=True,
                                   max_parent_feature_error=feature_error, labels_read=0, gpu_forwards=0))
    print(dict(status='PREPARED_ONLY_NOT_SUBMITTED', queries=593, fits=10), flush=True)


def guard(stage, fold=None):
    a = read(AUTH)
    for source in a['sources'] + a['parent_source_bindings']:
        checked(source)
    checked(a['parent_authority'])
    preflight = read(OUT/'preflight.json')
    assert preflight['authority'] == bind(AUTH)
    assert preflight['status'] == 'QWEN_PRODUCT6_FEATURES_AND_PARENT_SEALS_PASS'
    assert os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED'
    if stage == 'fit':
        b = a['folds'][str(fold)]
        allowed = {Path(a[key]['path']).resolve() for key in ('split', 'gallery', 'cache')}
        allowed.update(Path(b[key]['path']).resolve() for key in ('train_roles', 'parent_payload', 'parent_validation'))
        allowed.add((OUT/'preflight.json').resolve())
        current_fold = (OUT/f'fold{fold}').resolve()
        curator = Path(a['curator']['path']).resolve()

        def audit(event, args):
            if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            p = Path(os.fsdecode(args[0])).resolve()
            assert p != curator and ROOT/'reports' not in p.parents, ('HELD_LABEL_OR_REPORT_READ', str(p))
            if ROOT/'results' in p.parents:
                assert p in allowed or current_fold in p.parents, ('UNAUTHORIZED_RESULT_READ', str(p))
        sys.addaudithook(audit)
    cache = read(checked(a['cache']))
    assert cache['labels_included'] is False and cache['columns'] == COLUMNS[:4]
    return a, cache['rows']


def baseline_parity(a, held, fold):
    """Read only this fold's prior heads and label-free predictions; never refit."""
    import numpy as np
    b = a['folds'][str(fold)]
    v = read(checked(b['parent_validation']))
    assert v['status'] == 'QWEN_QUALITY_FOLD_BASELINE_AND_NUMPY_TORCH_PASS'
    assert v['authority'] == a['parent_authority'] and v['payload'] == b['parent_payload']
    old = read(checked(b['parent_payload']))
    assert old['authority'] == a['parent_authority'] and old['fold'] == fold
    assert old['heldout_label_reads'] == 0 and old['train_query_ids'] == b['train_query_ids']
    assert old['effective_train_query_ids'] == b['effective_train_query_ids']
    predictions = {r['query_id']: r['models'] for r in old['predictions']}
    assert len(predictions) == len(held) and set(predictions) == {r['query_id'] for r in held}
    checks = {}
    for arm, indices in parent.ARMS.items():
        for loss_name in LOSSES:
            name = loss_name+'_'+arm
            theta = np.asarray([float.fromhex(x) for x in old['parameters'][name]])
            assert theta.shape == (len(indices)+1,) and np.isfinite(theta).all()
            maximum = 0.
            for row in held:
                x = np.asarray(row['X'], dtype=np.float64)[:, indices]
                z = explicit_logits(x, theta)
                saved = np.asarray([float.fromhex(x) for x in predictions[row['query_id']][name]['logits_hex']])
                assert saved.shape == z.shape == (127,)
                maximum = max(maximum, float(np.max(np.abs(z-saved))))
                old_choice = predictions[row['query_id']][name]['selected']
                assert row['candidate_physical_rows'][order_from_logits(row, z)[0]] == old_choice
                assert row['candidate_physical_rows'][order_from_logits(row, saved)[0]] == old_choice
            assert maximum < 2e-10, ('BASELINE_LOGIT_DRIFT', fold, name, maximum)
            checks[name] = dict(max_explicit_dot_error=maximum, exact_choices=len(held),
                                checked_logits=len(held)*127, retrained=False)
    return checks, old


def save_checkpoint(path, value):
    import torch
    temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def fit(a, rows, fold, budget):
    import numpy as np
    import torch
    import run_rc_h593_qwen3_rerank_v2_layout as q
    torch.set_num_threads(8)
    torch.manual_seed(17)
    start = time.monotonic()
    authority = bind(AUTH)
    d = OUT/f'fold{fold}'
    d.mkdir(parents=True, exist_ok=True)
    lock = (d/'fit.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    b = a['folds'][str(fold)]
    roles = {r['query_id']: r for r in read(checked(b['train_roles']))['records']}
    split = read(checked(a['split']))['folds'][fold]
    assert set(roles) == set(split['train_query_ids'])
    assert not set(roles) & set(split['heldout_query_ids'])
    labels = {r['physical_row']: r['identity'] for r in read(checked(a['gallery']))['records']}
    train = [r for r in rows if r['query_id'] in roles]
    held = [r for r in rows if r['query_id'] in set(split['heldout_query_ids'])]
    assert [r['query_id'] for r in train] == b['train_query_ids']
    checks, _ = baseline_parity(a, held, fold)
    print(dict(event='ALL_PARENT_HEADS_PARITY_PASS', fold=fold, arms=len(checks), held=len(held)), flush=True)
    kept, targets = [], []
    for row in train:
        positive = [i for i, p in enumerate(row['candidate_physical_rows'])
                    if labels[p] == roles[row['query_id']]['identity']]
        assert len(positive) <= 1
        if positive:
            kept.append(row)
            targets.append(-1 if positive[0] == row['winner'] else row['challenger_positions'].index(positive[0]))
    assert [r['query_id'] for r in kept] == b['effective_train_query_ids']
    x = torch.tensor(matrix(kept), dtype=torch.float64)
    h = matrix(held)
    y = torch.tensor(targets, dtype=torch.long)
    params, max_error = {}, 0.
    for loss_name in LOSSES:
        name = loss_name+'_'+ARM
        path = d/(name+'_parameters.json')
        if path.exists():
            p = read(path)
            assert p['authority'] == authority and p['steps'] == 2000 and p['model'] == name
            assert p['train_query_ids'] == b['effective_train_query_ids']
            theta_values = np.asarray([float.fromhex(v) for v in p['theta_hex']])
        else:
            cp = d/(name+'_checkpoint.pt')
            theta = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
            opt = torch.optim.AdamW([theta], lr=.03, weight_decay=.001)
            step = 0
            if cp.exists():
                state = torch.load(cp, weights_only=True, map_location='cpu')
                assert state['authority'] == authority and state['model'] == name and state['fold'] == fold
                assert state['columns'] == COLUMNS and 0 <= state['step'] <= 2000
                with torch.no_grad():
                    theta.copy_(state['theta'])
                opt.load_state_dict(state['optimizer'])
                step = state['step']
            while step < 2000:
                opt.zero_grad()
                value = q.loss(x@theta[:-1]+theta[-1], y, loss_name)
                assert torch.isfinite(value)
                value.backward()
                assert torch.isfinite(theta.grad).all()
                opt.step()
                step += 1
                if step % 100 == 0 or step == 2000:
                    save_checkpoint(cp, dict(authority=authority, model=name, fold=fold,
                                             columns=COLUMNS, theta=theta.detach(),
                                             optimizer=opt.state_dict(), step=step))
                    print(dict(event='FIT_PROGRESS', fold=fold, model=name, step=step), flush=True)
                    if time.monotonic()-start > budget and step < 2000:
                        return 75
            theta_values = theta.detach().numpy()
            with torch.no_grad():
                z = x@theta[:-1]+theta[-1]
                choices = z.argmax(1)
                choices[z.max(1).values <= 0] = -1
                final_loss = float(q.loss(z, y, loss_name))
                correct = int((choices == y).sum())
            write(path, dict(authority=authority, model=name, steps=2000, columns=COLUMNS,
                             theta_hex=[float(v).hex() for v in theta_values],
                             effective_train=len(kept), train_correct=correct,
                             final_training_loss=final_loss,
                             train_query_ids=[r['query_id'] for r in kept], heldout_label_reads=0))
        assert theta_values.shape == (6,) and np.isfinite(theta_values).all()
        params[name] = [float(v).hex() for v in theta_values]
        z = h@theta_values[:-1]+theta_values[-1]
        independent = (torch.from_numpy(h)@torch.from_numpy(theta_values[:-1])+theta_values[-1]).numpy()
        error = float(np.max(np.abs(z-independent)))
        assert error < 2e-10
        max_error = max(max_error, error)
        if time.monotonic()-start > budget and loss_name != LOSSES[-1]:
            return 75
    predictions = []
    for row, xx in zip(held, h):
        models = {}
        for name, values in params.items():
            t = np.asarray([float.fromhex(v) for v in values])
            z = xx@t[:-1]+t[-1]
            independently = explicit_logits(xx, t)
            error = float(np.max(np.abs(z-independently)))
            assert error < 2e-10
            max_error = max(max_error, error)
            selected = row['candidate_physical_rows'][order_from_logits(row, z)[0]]
            assert selected == row['candidate_physical_rows'][order_from_logits(row, independently)[0]]
            models[name] = dict(selected=selected, logits_hex=[float(v).hex() for v in z])
        predictions.append(dict(query_id=row['query_id'], models=models))
    write(d/'payload.json', dict(authority=authority, fold=fold, parameters=params,
                                predictions=predictions, train_query_ids=b['train_query_ids'],
                                effective_train_query_ids=b['effective_train_query_ids'],
                                parent_payload=b['parent_payload'], baseline_parity=checks,
                                heldout_label_reads=0, baseline_fits=0, new_fits=2))
    write(d/'validation.json', dict(status='QWEN_PRODUCT6_FOLD_BASELINES_AND_LOGITS_PASS',
                                   authority=authority, payload=bind(d/'payload.json'), fold=fold,
                                   max_independent_logit_error=max_error, baseline_parity=checks))
    return 0


def join(a, rows):
    import numpy as np
    predictions, parameters, seals, parent_predictions, parent_parameters = {}, {}, [], {}, {}
    split_all = read(checked(a['split']))['folds']
    for fold in range(5):
        v = read(OUT/f'fold{fold}/validation.json')
        assert v['status'] == 'QWEN_PRODUCT6_FOLD_BASELINES_AND_LOGITS_PASS' and v['fold'] == fold
        assert v['authority'] == bind(AUTH)
        payload = read(checked(v['payload']))
        assert payload['authority'] == bind(AUTH) and payload['fold'] == fold
        assert payload['heldout_label_reads'] == 0 and payload['new_fits'] == 2 and payload['baseline_fits'] == 0
        expected = set(split_all[fold]['heldout_query_ids'])
        assert {r['query_id'] for r in payload['predictions']} == expected
        assert not set(predictions) & expected
        predictions.update({r['query_id']: dict(r, fold=fold) for r in payload['predictions']})
        parameters[fold] = payload['parameters']
        seals.append(bind(OUT/f'fold{fold}/validation.json'))
        held = [r for r in rows if r['query_id'] in expected]
        _, old = baseline_parity(a, held, fold)
        parent_predictions.update({r['query_id']: r['models'] for r in old['predictions']})
        parent_parameters[fold] = old['parameters']
    assert len(predictions) == len(parent_predictions) == len(rows) == 593
    write(OUT/'predictions_prelabel_seal.json', dict(authority=bind(AUTH), fold_validations=seals,
                                                   cache=a['cache'], predictions=593,
                                                   held_labels_opened_after_this_seal=True))
    roles = {r['query_id']: r for r in read(checked(a['curator']))['records']}
    labels = {r['physical_row']: r['identity'] for r in read(checked(a['gallery']))['records']}
    pv = read(checked(a['parent_validation']))
    assert pv['authority'] == a['parent_authority'] and pv['result'] == a['parent_result']
    previous_result = read(checked(a['parent_result']))
    previous = {r['query_id']: r for r in previous_result['rows']}
    assert set(previous) == set(predictions)
    new_names = [loss+'_'+ARM for loss in LOSSES]
    old_names = [loss+'_'+arm for arm in parent.ARMS for loss in LOSSES]
    train_identities, train_components = {}, {}
    for fold in range(5):
        train_roles = read(checked(a['folds'][str(fold)]['train_roles']))['records']
        train_identities[fold] = {r['identity'] for r in train_roles}
        train_components[fold] = {r['component'] for r in train_roles}
    result_rows, maximum = [], 0.
    for row in rows:
        qid = row['query_id']
        role = roles[qid]
        fold = role['outer_fold']
        axis, winner = row['candidate_physical_rows'], row['winner']
        assert fold == predictions[qid]['fold'] == previous[qid]['fold']
        assert role['component'] == previous[qid]['component']
        assert role['identity'] not in train_identities[fold]
        assert role['component'] not in train_components[fold]
        raw_order = sorted(range(128), key=lambda i: (-row['raw_scores'][i], axis[i]))
        raw_rank = {i: k for k, i in enumerate(raw_order)}
        direct = sorted(range(128), key=lambda i: (-row['qwen_logit'][i], raw_rank[i], axis[i]))
        orders = {'RAW': raw_order, 'QWEN3_DIRECT': direct}
        for name in old_names+new_names:
            if name in new_names:
                entry = predictions[qid]['models'][name]
                theta = np.asarray([float.fromhex(v) for v in parameters[fold][name]])
                # Reconstruct the extra column with an independent scalar formula.
                w = row['winner']
                m, l = row['mass'], row['free_content']
                product = [(float(m[i])*float(l[i])-float(m[w])*float(l[w])) /
                           (abs(float(m[i])*float(l[i]))+abs(float(m[w])*float(l[w]))+1e-12)
                           for i in row['challenger_positions']]
                xx = np.column_stack((np.asarray(row['X'], dtype=np.float64), product))
            else:
                entry = parent_predictions[qid][name]
                theta = np.asarray([float.fromhex(v) for v in parent_parameters[fold][name]])
                xx = np.asarray(row['X'], dtype=np.float64)[:, parent.ARMS[name.split('_', 1)[1]]]
            z = explicit_logits(xx, theta)
            saved = np.asarray([float.fromhex(v) for v in entry['logits_hex']])
            error = float(np.max(np.abs(z-saved)))
            assert error < 2e-10
            maximum = max(maximum, error)
            order = order_from_logits(row, saved)
            assert axis[order[0]] == entry['selected']
            assert order_from_logits(row, z)[0] == order[0]
            orders[name] = order
        target = {i for i, physical in enumerate(axis) if labels[physical] == role['identity']}
        assert len(target) <= 1 and bool(target) == previous[qid]['target_in_C128']
        rr = {name: next((1./(j+1) for j, i in enumerate(order) if i in target), 0.)
              for name, order in orders.items()}
        chosen = {name: axis[order[0]] for name, order in orders.items()}
        correct = {name: value == 1 for name, value in rr.items()}
        for name in previous[qid]['correct']:
            assert chosen[name] == previous[qid]['selected'][name]
            assert correct[name] == previous[qid]['correct'][name]
            assert rr[name] == previous[qid]['reciprocal_rank_C128'][name]
        result_rows.append(dict(query_id=qid, original_query_id=role['original_query_id'],
                                component=role['component'], fold=fold, target_in_C128=bool(target),
                                selected=chosen, correct=correct, reciprocal_rank_C128=rr))
    counts = {name: sum(r['correct'][name] for r in result_rows) for name in result_rows[0]['correct']}
    assert all(counts[name] == value for name, value in previous_result['counts'].items())
    assert sum(r['target_in_C128'] for r in result_rows) == 570
    comparisons = {}
    for loss_name in LOSSES:
        model = loss_name+'_'+ARM
        for base in ('RAW', 'QWEN3_DIRECT', loss_name+'_QWEN3', loss_name+'_QWEN_M4', loss_name+'_QWEN_ML5'):
            comparisons[base+'__to__'+model] = parent.comparison(result_rows, base, model)
    primary = comparisons[a['primary_comparator']+'__to__'+a['primary']]
    secondary = comparisons[a['secondary_comparator']+'__to__'+a['secondary']]
    fold_comparisons = {}
    for fold in range(5):
        fold_rows = [r for r in result_rows if r['fold'] == fold]
        fold_comparisons[str(fold)] = {
            loss: parent.comparison(fold_rows, loss+'_QWEN_ML5', loss+'_'+ARM) for loss in LOSSES}
    result = dict(status='H593_QWEN_PRODUCT6_GROUPED_OOF_COMPLETE', authority=bind(AUTH),
                  counts=counts, denominator=593, target_in_C128=570, absent_from_C128=23,
                  folds={str(k): {m: sum(r['correct'][m] for r in result_rows if r['fold'] == k)
                                  for m in counts} for k in range(5)},
                  MRR_C128={m: sum(r['reciprocal_rank_C128'][m] for r in result_rows)/593 for m in counts},
                  comparisons=comparisons, per_fold_primary_and_secondary_comparisons=fold_comparisons,
                  primary=a['primary'], primary_comparator=a['primary_comparator'],
                  secondary=a['secondary'], secondary_comparator=a['secondary_comparator'],
                  primary_comparison=primary, secondary_comparison=secondary,
                  parent_result=a['parent_result'], rows=result_rows, evidence=a['evidence'],
                  new_fits=10, baseline_retraining=False, inference_threshold_tuning=False,
                  parameter_count=6, added_parameter_count=1, max_independent_dot_error=maximum,
                  parameter_matched_nonlinearity_control=False,
                  limitation='A gain would support this added product feature under this protocol, not prove interaction is uniquely necessary or superior to all one-parameter alternatives.')
    write(OUT/'result.json', result)
    lines = ['# H593：Qwen 加性质量头新增 M×L 乘积项', '',
             '原 ColNomic 自然 C128、原五折身份/来源组隔离、同一 Qwen logit 与 RoMa M/自由内容 L；593 张均计入，目标在 C128 内 570 张、缺失 23 张。仅新训六参数头的 2 种损失 × 5 折，旧头原样复用。', '',
             'QWEN_ML5 = RAW差分 + Qwen差分 + sym(M) + sym(L) + 偏置（五参数）。',
             'QWEN_ML_PRODUCT6 额外加入 sym(M_g L_g, M_w L_w)，并重新拟合六参数；不是 sym(M)×sym(L)。', '',
             '| 损失 | 旧加性 ML5 | 新乘积 PRODUCT6 | 相对旧头救回 / 误伤 | 净增 |',
             '|---|---:|---:|---:|---:|']
    for loss_name in LOSSES:
        comparator, model = loss_name+'_QWEN_ML5', loss_name+'_'+ARM
        c = comparisons[comparator+'__to__'+model]
        lines.append(f"| {loss_name} | {counts[comparator]}/593 | {counts[model]}/593 | {c['rescue']} / {c['breaks']} | {c['net']:+d} |")
    lines += ['', f"CE 为预声明主检验：component 均衡 95% 区间 {primary['component_balanced_bootstrap95']}；query 加权分组 bootstrap 区间 {primary['query_weighted_cluster_bootstrap95']}。",
              f"COST1 为次检验：component 均衡 95% 区间 {secondary['component_balanced_bootstrap95']}；query 加权分组 bootstrap 区间 {secondary['query_weighted_cluster_bootstrap95']}。", '',
              '所有旧臂逐折、逐候选 logit 和决策已核对；新臂全部 127 个挑战者分数与 HOLD=0、参数、训练/优化器断点均保留。五折预测分别封存后才读取汇总标签，未按 held 结果挑阈值或改主臂。', '',
              '这是已多次查看的 H593 上开发性增量检验。新增一参数，未加入同参数量其它非线性对照；正结果只能支持本协议下该乘积特征的增量，不能证明所有收益唯一来自乘法，也不等于新外部验证。', '',
              '| 折 | CE 救回/误伤/净增 | COST1 救回/误伤/净增 |', '|---|---:|---:|']
    for fold in range(5):
        cells = []
        for loss_name in LOSSES:
            c = fold_comparisons[str(fold)][loss_name]
            cells.append(f"{c['rescue']}/{c['breaks']}/{c['net']:+d}")
        lines.append(f"| {fold} | {' | '.join(cells)} |")
    report = OUT/'report_zh.md'
    text = '\n'.join(lines)+'\n'
    if report.exists():
        assert report.read_text() == text
    else:
        temporary = report.with_name('.'+report.name+f'.{os.getpid()}.tmp')
        temporary.write_text(text)
        os.replace(temporary, report)
    write(OUT/'validation.json', dict(status='QWEN_PRODUCT6_593_PARENT_PARITY_SEALS_COUNTS_PASS',
                                     authority=bind(AUTH), result=bind(OUT/'result.json'), report=bind(report),
                                     prelabel_seal=bind(OUT/'predictions_prelabel_seal.json'),
                                     fold_validations=seals, max_independent_dot_error=maximum))
    print(dict(status=result['status'], counts=counts, primary=primary, secondary=secondary), flush=True)


def self_test():
    import numpy as np
    import torch
    import run_rc_h593_qwen3_rerank_v2_layout as q
    q.self_test()
    row = dict(mass=[.2]+[.8]*127, free_content=[.7]+[.3]*127,
               winner=0, challenger_positions=list(range(1, 128)),
               candidate_physical_rows=list(range(128)), X=[[.5, -.25, .6, -.4]]*127)
    product = product_column(row)
    expected = (.8*.3-.2*.7)/(.8*.3+.2*.7+1e-12)
    assert np.all(product == expected)
    wrong = parent.sym(.8, .2)*parent.sym(.3, .7)
    assert expected > 0 and wrong < 0 and abs(expected-wrong) > .4
    x = matrix([row])[0]
    assert x.shape == (127, 5) and x.flags.c_contiguous
    old_theta = np.array([.5, -.25, .125, -.5, .25])
    new_theta = np.concatenate((old_theta[:-1], [0.], old_theta[-1:]))
    np.testing.assert_array_equal(explicit_logits(x, new_theta), explicit_logits(np.asarray(row['X']), old_theta))
    assert order_from_logits(row, np.zeros(127))[0] == 0
    assert order_from_logits(row, np.ones(127))[0] == 1
    zero_row = dict(row, mass=[0.]*128, free_content=[0.]*128)
    assert np.array_equal(product_column(zero_row), np.zeros(127))
    assert np.isfinite(product_column(dict(row, free_content=[-.7]+[-.3]*127))).all()
    tx = torch.tensor(np.stack((x[:3], -x[:3])), dtype=torch.float64)
    target = torch.tensor([-1, 1])
    for loss_name in LOSSES:
        theta = torch.nn.Parameter(torch.zeros(6, dtype=torch.float64))
        optimizer = torch.optim.AdamW([theta], lr=.03, weight_decay=.001)
        for _ in range(2):
            optimizer.zero_grad()
            q.loss(tx@theta[:-1]+theta[-1], target, loss_name).backward()
            optimizer.step()
        buffer = io.BytesIO()
        torch.save(dict(theta=theta.detach().clone(), optimizer=copy.deepcopy(optimizer.state_dict())), buffer)
        for _ in range(2):
            optimizer.zero_grad()
            q.loss(tx@theta[:-1]+theta[-1], target, loss_name).backward()
            optimizer.step()
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        resumed = torch.nn.Parameter(saved['theta'].clone())
        resumed_optimizer = torch.optim.AdamW([resumed], lr=.03, weight_decay=.001)
        resumed_optimizer.load_state_dict(saved['optimizer'])
        for _ in range(2):
            resumed_optimizer.zero_grad()
            q.loss(tx@resumed[:-1]+resumed[-1], target, loss_name).backward()
            resumed_optimizer.step()
        assert torch.equal(theta.detach(), resumed.detach()), ('RESUME_DRIFT', loss_name)
    print(dict(status='QWEN_PRODUCT6_FORMULA_EMBEDDING_ACTION_OPTIMIZER_RESUME_PASS',
               product_not_product_of_deltas=True, zero_product_embeds_additive=True,
               losses=list(LOSSES)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'fit', 'join', 'self-test'))
    parser.add_argument('--fold', type=int)
    parser.add_argument('--budget', type=int, default=450)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'self-test':
        self_test()
    else:
        assert args.stage != 'fit' or args.fold in range(5)
        authority, data = guard(args.stage, args.fold)
        if args.stage == 'fit':
            sys.exit(fit(authority, data, args.fold, args.budget))
        join(authority, data)
