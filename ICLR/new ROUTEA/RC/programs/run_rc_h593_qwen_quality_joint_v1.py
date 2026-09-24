#!/usr/bin/env python3
"""CPU-only grouped Qwen + M/free-L calibration, with exact baseline choices.

Task 3 execution is separately released after the fixed external transfer task.
Fit processes may read only the current TRAIN identities, never held labels.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_h593_qwen_quality_joint_v1'
AUTH = ROOT / 'registry/rc_h593_qwen_quality_joint_authority_v1_20260924.json'
PARENT = ROOT / 'registry/rc_h593_qwen3_rerank_authority_v2_layout_20260923.json'
QOUT = ROOT / 'results/rc_h593_qwen3_rerank_v2_layout'
SIMPLE = ROOT / 'results/rc_h593_simple_explanations_v1'
PLAN = ROOT / 'plan/RC_H593_QWEN_QUALITY_JOINT_V1_20260924.md'
ARMS = {'QWEN3': [0, 1], 'QWEN_L4': [0, 1, 3],
        'QWEN_M4': [0, 1, 2], 'QWEN_ML5': [0, 1, 2, 3]}
LOSSES = ('CE', 'COST1')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}


def checked(b):
    assert sha(b['path']) == b['sha256'], ('SHA_DRIFT', b['path'])
    return Path(b['path'])


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists():
        assert p.read_text() == value, ('IMMUTABLE', str(p))
        return
    tmp = p.with_name('.' + p.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def sym(a, b):
    return (a-b)/(abs(a)+abs(b)+1e-12)


def prepare():
    """Only label-free score/feature reads; model choices already frozen in PLAN."""
    import numpy as np
    import run_rc_h593_qwen3_rerank_v2_layout as q
    parent = read(PARENT)
    workers = read(checked(parent['workers']))
    assert workers['labels_included'] is False
    score = q.all_scores(parent, workers)
    cv = read(SIMPLE/'cache_validation.json')
    assert cv['status'] == 'CACHE_PASS'
    cache = read(checked(cv['payload']))
    assert cache['labels_included'] is False
    simple = {r['query_id']: r for r in cache['rows']}
    assert len(simple) == len(workers['records']) == 593
    rows = []
    maximum = 0.
    for row in workers['records']:
        r = simple[row['query_id']]
        assert r['execution_ordinal'] == row['execution_ordinal']
        assert r['source_image_sha256'] == row['image_sha256']
        assert r['axis'] == row['candidate_physical_rows']
        assert r['winner'] == row['winner'] and r['challengers'] == row['challenger_positions']
        axis = r['axis']; w = r['winner']; c = r['challengers']
        assert len(set(axis)) == 128 and c == [i for i in range(128) if i != w]
        raw_order = sorted(range(128), key=lambda i: (-row['raw_scores'][i], axis[i]))
        assert r['raw_ranked'][:128] == [axis[i] for i in raw_order] and raw_order[0] == w
        qx = q.make_x(row, score[row['query_id']])
        sx = np.asarray(r['X'], dtype=np.float64)
        raw_error = float(np.max(np.abs(qx[:, 0]-sx[:, 0])))
        assert raw_error < 2e-10
        maximum = max(maximum, raw_error)
        m = np.asarray(r['mass'], dtype=np.float64)
        l = np.asarray(r['free_content'], dtype=np.float64)
        assert m.shape == l.shape == (128,)
        assert np.isfinite(m).all() and np.isfinite(l).all()
        assert np.min(m) >= 0 and np.max(m) <= 1
        ml = np.column_stack((sym(m[c], m[w]), sym(l[c], l[w])))
        error = float(np.max(np.abs(ml-sx[:, 2:4])))
        assert error < 2e-10
        maximum = max(maximum, error)
        x = np.column_stack((qx, sx[:, 2:4]))
        assert np.isfinite(x).all() and x.shape == (127, 4)
        rows.append(dict(row, X=x.tolist(), mass=r['mass'], free_content=r['free_content'],
                         qwen_logit=[p['logit'] for p in score[row['query_id']]['pairs']]))
    assert set(simple) == {r['query_id'] for r in rows}
    write(OUT/'cache.json', dict(rows=rows, labels_included=False,
                               columns=['raw_standardized_delta', 'qwen_standardized_delta',
                                        'sym_mass_delta', 'sym_free_content_delta']))
    fold_bindings = {}
    for k in range(5):
        v = read(QOUT/f'fold{k}/validation.json')
        assert v['status'] == 'QWEN3_CALIBRATION_NUMPY_TORCH_PASS'
        checked(v['payload'])
        fold_bindings[str(k)] = dict(parent['folds'][str(k)],
                                     qwen_payload=v['payload'],
                                     qwen_validation=bind(QOUT/f'fold{k}/validation.json'))
    a = dict(status='H593_QWEN_QUALITY_JOINT_PREDECLARED', parent=bind(PARENT),
             sources=[bind(Path(__file__)), bind(PLAN), bind(Path(q.__file__)),
                      bind(ROOT/'slurm/rc_h593_qwen_quality_joint_v1.sbatch')],
             cache=bind(OUT/'cache.json'), cache_parent=bind(SIMPLE/'cache_validation.json'),
             qwen_result=bind(QOUT/'result.json'), workers=parent['workers'],
             split=parent['split'], gallery=parent['gallery'], curator=parent['curator'],
             folds=fold_bindings, arms=ARMS, losses=list(LOSSES), primary='CE_QWEN_ML5',
             primary_comparator='CE_QWEN3', quality_added_comparator='CE_QWEN_L4',
             optimizer=dict(name='AdamW', lr=.03, weight_decay=.001, steps=2000,
                            zero_initialization=True, dtype='float64', threads=8, seed=17),
             action='RAW winner HOLD score 0, max challenger > 0 only; physical-axis first tie',
             evidence='Previously opened H593, original grouped OOF5, no new external confirmation',
             no_new_gpu_forwards=True, no_held_threshold_selection=True,
             no_other_fold_trained_logits_in_features=True)
    write(AUTH, a)
    write(OUT/'preflight.json', dict(status='QWEN_QUALITY_AXES_FEATURES_593_PASS',
                                   authority=bind(AUTH), cache=bind(OUT/'cache.json'),
                                   queries=593, candidates=128, max_formula_error=maximum,
                                   labels_read=0, gpu_forwards=0))
    print(dict(status='PREPARED_ONLY_NOT_SUBMITTED', queries=593, fits=40), flush=True)


def guard(stage, fold=None):
    a = read(AUTH)
    for b in a['sources']:
        checked(b)
    assert read(OUT/'preflight.json')['authority'] == bind(AUTH)
    if stage in ('fit', 'join'):
        assert os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED'
    if stage == 'fit':
        allowed = {Path(a[k]['path']).resolve() for k in ('split', 'gallery')}
        allowed.update(Path(a['folds'][str(fold)][k]['path']).resolve()
                       for k in ('train_roles', 'qwen_payload', 'qwen_validation'))
        allowed.update((OUT/'cache.json', OUT/'preflight.json'))
        current_fold = OUT/f'fold{fold}'
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
    assert cache['labels_included'] is False
    return a, cache['rows']


def matrix(rows, arm):
    import numpy as np
    return np.ascontiguousarray(np.asarray([r['X'] for r in rows], dtype=np.float64)[:, :, ARMS[arm]])


def fit(a, rows, fold, budget):
    import numpy as np
    import torch
    import run_rc_h593_qwen3_rerank_v2_layout as q
    torch.set_num_threads(8)
    torch.manual_seed(17)
    start = time.monotonic()
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
    kept, targets = [], []
    for r in train:
        pos = [i for i, p in enumerate(r['candidate_physical_rows']) if labels[p] == roles[r['query_id']]['identity']]
        assert len(pos) <= 1
        if pos:
            kept.append(r)
            targets.append(-1 if pos[0] == r['winner'] else r['challenger_positions'].index(pos[0]))
    assert [r['query_id'] for r in kept] == b['effective_train_query_ids']
    y = torch.tensor(targets)
    old = read(checked(b['qwen_payload']))
    assert old['fold'] == fold and old['heldout_label_reads'] == 0
    assert old['train_query_ids'] == b['train_query_ids']
    old_predictions = {r['query_id']: r for r in old['predictions']}
    baseline_parity = {}
    params = {}
    max_error = 0.
    # QWEN3 comes first. Both original losses must reproduce before new arms.
    for arm in ARMS:
        for loss_name in LOSSES:
            if arm != 'QWEN3':
                assert set(baseline_parity) == set(LOSSES), 'BASELINE_PARITY_FIRST'
            name = loss_name+'_'+arm
            path = d/(name+'_parameters.json')
            x = torch.tensor(matrix(kept, arm), dtype=torch.float64)
            h = matrix(held, arm)
            if path.exists():
                p = read(path)
                assert p['authority'] == bind(AUTH) and p['steps'] == 2000
                theta_values = np.array([float.fromhex(v) for v in p['theta_hex']])
            else:
                cp = d/(name+'_checkpoint.pt')
                theta = torch.nn.Parameter(torch.zeros(x.shape[-1]+1, dtype=torch.float64))
                opt = torch.optim.AdamW([theta], lr=.03, weight_decay=.001)
                step = 0
                if cp.exists():
                    s = torch.load(cp, weights_only=True, map_location='cpu')
                    assert s['authority'] == bind(AUTH) and s['model'] == name
                    with torch.no_grad():
                        theta.copy_(s['theta'])
                    opt.load_state_dict(s['optimizer'])
                    step = s['step']
                while step < 2000:
                    opt.zero_grad()
                    v = q.loss(x@theta[:-1]+theta[-1], y, loss_name)
                    assert torch.isfinite(v)
                    v.backward()
                    opt.step()
                    step += 1
                    if step % 100 == 0 or step == 2000:
                        tmp = cp.with_suffix('.tmp')
                        with tmp.open('wb') as f:
                            torch.save(dict(authority=bind(AUTH), model=name, theta=theta.detach(),
                                            optimizer=opt.state_dict(), step=step), f)
                            f.flush(); os.fsync(f.fileno())
                        os.replace(tmp, cp)
                        print(dict(event='FIT_PROGRESS', fold=fold, model=name, step=step), flush=True)
                        if time.monotonic()-start > budget and step < 2000:
                            return 75
                theta_values = theta.detach().numpy()
                with torch.no_grad():
                    z = x@theta[:-1]+theta[-1]
                    chosen = z.argmax(1)
                    chosen[z.max(1).values <= 0] = -1
                    train_correct = int((chosen == y).sum())
                    final_loss = float(q.loss(z, y, loss_name))
                write(path, dict(authority=bind(AUTH), model=name, steps=2000,
                                 theta_hex=[float(v).hex() for v in theta_values],
                                 effective_train=len(kept), train_correct=train_correct,
                                 final_training_loss=final_loss,
                                 train_query_ids=[r['query_id'] for r in kept], heldout_label_reads=0))
            params[name] = [float(v).hex() for v in theta_values]
            z = h@theta_values[:-1]+theta_values[-1]
            independent = (torch.from_numpy(h)@torch.from_numpy(theta_values[:-1])+theta_values[-1]).numpy()
            err = float(np.max(np.abs(z-independent)))
            assert err < 2e-10
            max_error = max(max_error, err)
            if arm == 'QWEN3':
                old_theta = np.array([float.fromhex(v) for v in old['parameters'][loss_name]])
                t_err = float(np.max(np.abs(theta_values-old_theta)))
                assert t_err < 2e-10, ('BASELINE_PARAMETER_DRIFT', fold, loss_name, t_err)
                logit_error = 0.
                for i, r in enumerate(held):
                    previous = old_predictions[r['query_id']]['models'][loss_name]
                    oldz = np.array([float.fromhex(v) for v in previous['logits_hex']])
                    logit_error = max(logit_error, float(np.max(np.abs(z[i]-oldz))))
                    j = int(z[i].argmax())
                    pos = r['challenger_positions'][j] if z[i,j] > 0 else r['winner']
                    assert r['candidate_physical_rows'][pos] == previous['selected'], ('BASELINE_CHOICE_DRIFT', r['query_id'])
                assert logit_error < 2e-9, ('BASELINE_LOGIT_DRIFT', fold, loss_name, logit_error)
                baseline_parity[loss_name] = dict(parameter_max_error=t_err, logit_max_error=logit_error,
                                                exact_choices=len(held), theta_hex_exact=params[name] == old['parameters'][loss_name])
                print(dict(event='BASELINE_PARITY_PASS', fold=fold, loss=loss_name,
                           **baseline_parity[loss_name]), flush=True)
            if time.monotonic()-start > budget:
                return 75
    predictions = []
    for r in held:
        models = {}
        fullx = np.asarray(r['X'], dtype=np.float64)
        for name, values in params.items():
            arm = name.split('_', 1)[1]
            t = np.array([float.fromhex(v) for v in values])
            z = fullx[:, ARMS[arm]]@t[:-1]+t[-1]
            j = int(z.argmax())
            pos = r['challenger_positions'][j] if z[j] > 0 else r['winner']
            models[name] = dict(selected=r['candidate_physical_rows'][pos],
                                logits_hex=[float(v).hex() for v in z])
        predictions.append(dict(query_id=r['query_id'], models=models))
    write(d/'payload.json', dict(authority=bind(AUTH), fold=fold, parameters=params,
                                predictions=predictions, train_query_ids=b['train_query_ids'],
                                effective_train_query_ids=b['effective_train_query_ids'],
                                baseline_parity=baseline_parity, heldout_label_reads=0))
    write(d/'validation.json', dict(status='QWEN_QUALITY_FOLD_BASELINE_AND_NUMPY_TORCH_PASS',
                                   authority=bind(AUTH), payload=bind(d/'payload.json'), fold=fold,
                                   max_numpy_torch_error=max_error, baseline_parity=baseline_parity))
    return 0


def comparison(rows, base, model):
    import numpy as np
    groups = defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['correct'][model])-int(r['correct'][base]))
    values = [v for _, v in sorted(groups.items())]
    total = np.array([sum(v) for v in values], dtype=float)
    size = np.array([len(v) for v in values], dtype=float)
    mean = total/size
    rng = np.random.default_rng(20260924)
    sample = rng.integers(0, len(values), size=(10000, len(values)))
    macro = mean[sample].mean(1)
    query = total[sample].sum(1)/size[sample].sum(1)
    rescue = sum(r['correct'][model] and not r['correct'][base] for r in rows)
    breaks = sum(r['correct'][base] and not r['correct'][model] for r in rows)
    return dict(rescue=rescue, breaks=breaks, net=rescue-breaks,
                changed=sum(r['selected'][model] != r['selected'][base] for r in rows),
                baseline_correct=sum(r['correct'][base] for r in rows),
                baseline_correct_loss_rate=breaks/max(sum(r['correct'][base] for r in rows), 1),
                components=len(values), component_balanced_delta=float(mean.mean()),
                component_balanced_bootstrap95=list(map(float, np.quantile(macro, [.025, .975]))),
                query_weighted_delta=float(total.sum()/size.sum()),
                query_weighted_cluster_bootstrap95=list(map(float, np.quantile(query, [.025, .975]))))


def join(a, rows):
    import numpy as np
    preds, params, seals = {}, {}, []
    for fold in range(5):
        v = read(OUT/f'fold{fold}/validation.json')
        assert v['status'] == 'QWEN_QUALITY_FOLD_BASELINE_AND_NUMPY_TORCH_PASS'
        assert v['authority'] == bind(AUTH)
        p = read(checked(v['payload']))
        assert p['authority'] == bind(AUTH) and p['fold'] == fold and p['heldout_label_reads'] == 0
        split = read(checked(a['split']))['folds'][fold]
        assert {r['query_id'] for r in p['predictions']} == set(split['heldout_query_ids'])
        assert not set(preds) & set(split['heldout_query_ids'])
        preds.update({r['query_id']: dict(r, fold=fold) for r in p['predictions']})
        params[fold] = p['parameters']
        seals.append(bind(OUT/f'fold{fold}/validation.json'))
    assert len(preds) == 593
    write(OUT/'predictions_prelabel_seal.json', dict(authority=bind(AUTH), fold_validations=seals,
                                                   cache=a['cache'], predictions=593,
                                                   held_labels_opened_after_this_seal=True))
    # Curator is opened only after all five independently sealed prediction files.
    roles = {r['query_id']: r for r in read(checked(a['curator']))['records']}
    labels = {r['physical_row']: r['identity'] for r in read(checked(a['gallery']))['records']}
    previous_result = read(checked(a['qwen_result']))
    previous = {r['query_id']: r for r in previous_result['rows']}
    assert len(previous) == 593
    model_names = [loss+'_'+arm for arm in ARMS for loss in LOSSES]
    result_rows = []
    error_max = 0.
    for r in rows:
        qid = r['query_id']; role = roles[qid]; fold = role['outer_fold']
        axis = r['candidate_physical_rows']; winner = r['winner']
        assert preds[qid]['fold'] == fold == previous[qid]['fold']
        assert role['component'] == previous[qid]['component']
        train_roles = read(checked(a['folds'][str(fold)]['train_roles']))['records']
        assert role['identity'] not in {t['identity'] for t in train_roles}
        assert role['component'] not in {t['component'] for t in train_roles}
        raw_order = sorted(range(128), key=lambda i: (-r['raw_scores'][i], axis[i]))
        raw_rank = {i: k for k, i in enumerate(raw_order)}
        direct_order = sorted(range(128), key=lambda i: (-r['qwen_logit'][i], raw_rank[i], axis[i]))
        orders = {'RAW': raw_order, 'QWEN3_DIRECT': direct_order}
        for name in model_names:
            arm = name.split('_', 1)[1]
            entry = preds[qid]['models'][name]
            t = np.array([float.fromhex(v) for v in params[fold][name]])
            x = np.asarray(r['X'], dtype=np.float64)[:, ARMS[arm]]
            # Independent explicit dot avoids sharing the fit-time matrix product.
            z = np.asarray([sum(float(u)*float(v) for u,v in zip(xx,t[:-1]))+float(t[-1]) for xx in x])
            saved = np.asarray([float.fromhex(v) for v in entry['logits_hex']])
            err = float(np.max(np.abs(z-saved)))
            assert err < 2e-10
            error_max = max(error_max, err)
            full = np.zeros(128)
            full[r['challenger_positions']] = saved
            order = sorted(range(128), key=lambda i: (-full[i], i != winner, i))
            assert axis[order[0]] == entry['selected']
            independent_full = np.zeros(128)
            independent_full[r['challenger_positions']] = z
            independent_order = sorted(range(128), key=lambda i: (-independent_full[i], i != winner, i))
            assert independent_order[0] == order[0]
            orders[name] = order
        target = {i for i,p in enumerate(axis) if labels[p] == role['identity']}
        assert len(target) <= 1 and bool(target) == previous[qid]['target_in_C128']
        rr = {m: next((1./(j+1) for j,i in enumerate(order) if i in target), 0.) for m,order in orders.items()}
        chosen = {m: axis[order[0]] for m,order in orders.items()}
        correct = {m: value == 1 for m,value in rr.items()}
        for old_name, new_name in [('RAW','RAW'), ('QWEN3_DIRECT','QWEN3_DIRECT'),
                                   ('QWEN3_CAL_CE','CE_QWEN3'), ('QWEN3_CAL_COST1','COST1_QWEN3')]:
            assert chosen[new_name] == previous[qid]['selected'][old_name]
            assert correct[new_name] == previous[qid]['correct'][old_name]
        result_rows.append(dict(query_id=qid, original_query_id=role['original_query_id'],
                                component=role['component'], fold=fold, target_in_C128=bool(target),
                                selected=chosen, correct=correct, reciprocal_rank_C128=rr))
    counts = {m: sum(r['correct'][m] for r in result_rows) for m in result_rows[0]['correct']}
    assert {k: counts[k] for k in ('RAW','QWEN3_DIRECT','CE_QWEN3','COST1_QWEN3')} == {
        'RAW':426, 'QWEN3_DIRECT':522, 'CE_QWEN3':523, 'COST1_QWEN3':477}
    assert sum(r['target_in_C128'] for r in result_rows) == 570
    bases = ('RAW','QWEN3_DIRECT','CE_QWEN3','COST1_QWEN3','CE_QWEN_L4','COST1_QWEN_L4')
    comparisons = {base+'__to__'+m: comparison(result_rows,base,m)
                   for base in bases for m in counts if m != base}
    result = dict(status='H593_QWEN_QUALITY_JOINT_GROUPED_OOF_COMPLETE', authority=bind(AUTH),
                  counts=counts, denominator=593, target_in_C128=570,
                  folds={str(k): {m:sum(r['correct'][m] for r in result_rows if r['fold']==k) for m in counts} for k in range(5)},
                  MRR_C128={m:sum(r['reciprocal_rank_C128'][m] for r in result_rows)/593 for m in counts},
                  comparisons=comparisons, primary=a['primary'], primary_comparator=a['primary_comparator'],
                  quality_added_comparator=a['quality_added_comparator'], rows=result_rows,
                  evidence=a['evidence'], inference_threshold_tuning=False,
                  end_to_end_speed_claim=False, max_independent_dot_error=error_max)
    write(OUT/'result.json', result)
    write(OUT/'validation.json', dict(status='QWEN_QUALITY_593_AXES_BASELINE_PARITY_SEALS_COUNTS_PASS',
                                     authority=bind(AUTH), result=bind(OUT/'result.json'),
                                     prelabel_seal=bind(OUT/'predictions_prelabel_seal.json'),
                                     fold_validations=seals, max_independent_dot_error=error_max))
    lines = ['# H593：强Qwen重排与质量联合校准', '',
             '原ColNomic自然C128，原五折身份和来源组隔离，593张均计入；目标召回570/593。冻结原Qwen全部logit，新增计算仅CPU头训练。此前已查看H593，本轮为开发性分组检验。', '',
             '| 模型 | 正确/593 | 对RAW救回/误伤 | 对Qwen CE救回/误伤 | MRR@C128 |',
             '|---|---:|---:|---:|---:|']
    for m,n in counts.items():
        rc = comparisons.get('RAW__to__'+m, {'rescue':0,'breaks':0})
        qc = comparisons.get('CE_QWEN3__to__'+m, {'rescue':0,'breaks':0})
        lines.append(f"| {m} | {n}/593 | {rc['rescue']}/{rc['breaks']} | {qc['rescue']}/{qc['breaks']} | {result['MRR_C128'][m]:.6f} |")
    primary = comparisons['CE_QWEN3__to__CE_QWEN_ML5']
    added = comparisons['CE_QWEN_L4__to__CE_QWEN_ML5']
    lines += ['', '主臂预先固定为CE_QWEN_ML5，未按结果改选。原三参数两种损失逐折参数/全候选logit核对通过，593个基线决策与封存旧结果完全一致。', '',
              f"主臂相对Qwen CE：{primary['rescue']}救/{primary['breaks']}损，净增{primary['net']}；component均衡净增95%区间 {primary['component_balanced_bootstrap95']}；query加权cluster-bootstrap区间 {primary['query_weighted_cluster_bootstrap95']}。", '',
              f"主臂相对仅加自由内容的CE_QWEN_L4：{added['rescue']}救/{added['breaks']}损，净增{added['net']}。", '',
              '此结果只评价预声明的联合校准方式。未做同硬件完整pipeline成本实验，不能由小头耗时断言比Qwen更便宜；也不将标签oracle并集作为系统成绩。全部折头、127挑战者分数和HOLD=0、逐图动作、原logit/M/L/X及哈希均保留。']
    text = '\n'.join(lines)+'\n'
    report = OUT/'report_zh.md'
    if report.exists():
        assert report.read_text() == text
    else:
        report.write_text(text)
    print(dict(status=result['status'], counts=counts, primary=primary), flush=True)


def self_test():
    import numpy as np
    import run_rc_h593_qwen3_rerank_v2_layout as q
    q.self_test()
    rows = [{'X': [[1.,2.,3.,4.],[-1.,-2.,-3.,-4.]]}]
    for arm, indexes in ARMS.items():
        x = matrix(rows, arm)
        assert x.flags.c_contiguous
        assert np.array_equal(x, np.array([r['X'] for r in rows])[:,:,indexes])
    assert sym(0.,0.) == 0 and sym(1.,0.) > 0
    print(dict(status='JOINT_DESIGN_AND_ORIGINAL_LOSS_PREFLIGHT_PASS'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('prepare','fit','join','self-test'))
    p.add_argument('--fold', type=int)
    p.add_argument('--budget', type=int, default=450)
    args = p.parse_args()
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
