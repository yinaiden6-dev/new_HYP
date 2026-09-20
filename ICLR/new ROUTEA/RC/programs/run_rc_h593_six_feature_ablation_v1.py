#!/usr/bin/env python3
"""Original H593 OOF5: frozen-coordinate zeroing and matched leave-one-out refits."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import run_rc_six_cause_loss_binding_v1 as L

H, N = L.H, L.N
read, write, need, bind, checked = L.read, L.write, L.need, L.bind, L.checked
OUT = ROOT / 'results/rc_h593_six_feature_ablation_v1'
AUTH = ROOT / 'registry/rc_h593_six_feature_ablation_authority_v1_20260920.json'
PLAN = ROOT / 'plan/RC_H593_SIX_FEATURE_ABLATION_V1_20260920.md'
LAUNCH = ROOT / 'slurm/rc_h593_six_feature_ablation_v1.sbatch'
FEATURES = ('RAW', 'S', 'M', 'L', 'Q', 'R')
KINDS = ('COST1', 'CE')


def prepare():
    parent = read(L.AUTH)
    for source in parent['code_sources'].values():
        checked(source)
    sources = dict(parent['code_sources'])
    sources.update(program=bind(__file__), launcher=bind(LAUNCH), plan=bind(PLAN),
                   original_loss_binding=bind(L.__file__),
                   identity_core=bind(ROOT / 'src/rc_aslo_xf/gallery_identity_repair.py'))
    folds = {}
    for fold in range(5):
        p = L.OUT / f'fold{fold}/payload.json'
        v = L.OUT / f'fold{fold}/validation.json'
        need(read(v)['payload'] == bind(p), 'OLD_FOLD_VALIDATION')
        folds[str(fold)] = dict(parent['fold_sources'][str(fold)],
                               full_payload=bind(p), full_validation=bind(v))
    public = dict(parent['public_sources'])
    public['feature_authority'] = bind(ROOT / 'registry/rc_new_hyp593_feature_authority_v1_20260911.json')
    write(AUTH, dict(status='H593_SIX_FEATURE_ABLATION_AUTHORIZED', user_authorization_date='2026-09-20',
                     parent=bind(L.AUTH), code_sources=sources, public_sources=public,
                     features=parent['features'], fold_sources=folds,
                     join_sources=dict(curator=parent['join_sources']['curator'],
                                       old_result=bind(L.OUT / 'result.json'), old_validation=bind(L.OUT / 'validation.json')),
                     features_order=list(FEATURES), losses=list(KINDS), primary='COST1',
                     methods=['FULL', 'ZERO_EACH', 'REFIT_EACH'], steps=2000, seed=17,
                     precision='float64', optimizer=dict(name='AdamW', lr=.03, weight_decay=.001),
                     fold_count=5, new_refit_heads=60, old_full_replays=10,
                     new_encoder_forwards=0, automatic_deployment_change=False,
                     evidence_level='Post-hoc opened H593 grouped OOF5 development ablation'))
    print(json.dumps(dict(status='PREPARED', authority=bind(AUTH))), flush=True)


def guard(stage, fold):
    a = read(AUTH)
    need(a['status'] == 'H593_SIX_FEATURE_ABLATION_AUTHORIZED', 'AUTHORITY')
    for b in a['code_sources'].values():
        checked(b)
    need(a['code_sources']['program'] == bind(__file__), 'PROGRAM_PIN')
    if stage != 'preflight':
        need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allow = {Path(b['path']).resolve() for b in a['public_sources'].values()}
    for sources in a['features']:
        allow.update(Path(b['path']).resolve() for b in sources.values())
    if stage in ('fit', 'verify'):
        need(fold in range(5), 'FOLD')
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join', 'join-verify'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for sources in a['fold_sources'].values():
            allow.update(Path(b['path']).resolve() for b in sources.values())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        text = str(p).lower()
        need(not any(t in text for t in ('d1-mi', 'd1_mi', 'formal392', '/grozi/', '/target_join/',
                                         'gisc_prerecall_universe', '/reports/', 'rc_opened_')), 'PROTECTED_READ')
        if 'curator_roles' in text:
            need(stage in ('join', 'join-verify'), 'HELDOUT_LABELS_AFTER_ALL_SEALS')
        if ROOT / 'results' in p.parents:
            own = OUT in p.parents
            if own and stage in ('fit', 'verify'):
                first = p.relative_to(OUT).parts[0]
                own = first == 'preflight.json' or first == f'fold{fold}'
            need(own or p in allow, 'UNLISTED_RESULT:' + text)
    sys.addaudithook(audit)
    if stage != 'preflight':
        pf = read(OUT / 'preflight.json')
        need(pf['authority'] == bind(AUTH) and pf['status'] == 'ABLATION_PREFLIGHT_PASS', 'PREFLIGHT')
    return a


def hx(values):
    return [float(x).hex() for x in values]


def score(x, theta, keep):
    return x[..., keep] @ theta[:-1] + theta[-1]


def choose(z, row):
    j = int(z.argmax())
    pos = row['challenger_positions'][j] if float(z[j]) > 0 else row['winner']
    return row['candidate_physical_rows'][pos]


def compute(a, fold):
    b = a['fold_sources'][str(fold)]
    old = read(checked(b['full_payload']))
    val = read(checked(b['full_validation']))
    need(val['payload'] == b['full_payload'] and val['status'] == 'SIX_CAUSE_LOSS_BINDING_FRESH_NUMPY_PASS', 'FULL_QUALIFIED')
    roles = {r['query_id']: r for r in read(checked(b['train_roles']))['records']}
    rows, _ = H.features()
    split = read(checked(a['public_sources']['split']))['folds'][fold]
    train = [r for r in rows if r['query_id'] in set(split['train_query_ids'])]
    held = [r for r in rows if r['query_id'] in set(split['heldout_query_ids'])]
    need(set(roles) == {r['query_id'] for r in train}, 'TRAIN_LABEL_AXIS')
    need(not set(split['train_query_ids']) & set(split['heldout_query_ids']), 'DISJOINT_QUERIES')
    labels, label_hash = N.P.gallery_labels()
    data = H.batch(train, roles, labels)
    effective_ids = [r['query_id'] for r in train if H.target_position(r, roles[r['query_id']]['identity'], labels) >= -1]
    ce = read(checked(b['ce_payload']))
    need(effective_ids == ce['models']['ALL_CE']['query_ids'], 'EXACT_OLD_TRAIN_ORDER')
    need([r['query_id'] for r in train] == old['train_query_ids'], 'EXACT_OLD_TRAIN_POOL')
    x = torch.stack([r['modes']['REAL']['X'] for r in held])
    need(x.dtype == data['X'].dtype == torch.float64 and x.shape[1:] == (127, 6), 'INPUT_SCHEMA')
    params, logits, timing = {}, {}, []
    for kind in KINDS:
        old_kind = 'ALL_CE' if kind == 'CE' else kind
        torch.manual_seed(17)
        start = time.perf_counter()
        full = L.train(data['X'], data['y'], kind)
        need(hx(full) == old['parameters'][old_kind], 'FULL_PARAMETERS_BIT_EXACT:' + kind)
        name = kind + '_FULL'
        params[name] = dict(theta_hex=hx(full), retained_indices=list(range(6)), method='FULL', dropped=None)
        logits[name] = x @ full[:-1] + full[-1]
        timing.append(dict(model=name, seconds=time.perf_counter()-start))
        for i, feature in enumerate(FEATURES):
            name = f'{kind}_ZERO_{feature}'
            theta = full.clone(); theta[i] = 0.
            params[name] = dict(theta_hex=hx(theta), retained_indices=list(range(6)), method='ZERO', dropped=feature)
            logits[name] = x @ theta[:-1] + theta[-1]
            keep = [j for j in range(6) if j != i]
            name = f'{kind}_REFIT_{feature}'
            torch.manual_seed(17)
            start = time.perf_counter()
            theta = L.train(data['X'][..., keep], data['y'], kind)
            params[name] = dict(theta_hex=hx(theta), retained_indices=keep, method='REFIT', dropped=feature)
            logits[name] = score(x, theta, keep)
            timing.append(dict(model=name, seconds=time.perf_counter()-start))
            print(json.dumps(dict(event='REFIT_DONE', fold=fold, model=name, seconds=timing[-1]['seconds'])), flush=True)
    old_predictions = {p['query_id']: p for p in old['predictions']}
    predictions = []
    for j, row in enumerate(held):
        values = {m: dict(logits_hex=hx(z[j]), selected=choose(z[j], row)) for m, z in logits.items()}
        for kind in KINDS:
            need(values[kind+'_FULL'] == old_predictions[row['query_id']]['models']['ALL_CE' if kind == 'CE' else kind], 'FULL_LOGITS_AND_ACTION_BITS')
        predictions.append(dict(query_id=row['query_id'], execution_ordinal=row['execution_ordinal'],
                                candidate_physical_rows=row['candidate_physical_rows'],
                                challenger_positions=row['challenger_positions'], winner=row['winner'], models=values))
    payload = dict(status='H593_ABLATION_FOLD_SEALED', authority=bind(AUTH), fold=fold,
                   train_query_ids=[r['query_id'] for r in train], effective_train_query_ids=effective_ids,
                   target_absent_train=data['target_absent_count'], train_roles=b['train_roles'],
                   gallery_mapping_sha256=label_hash, parameters=params, predictions=predictions,
                   heldout_label_reads=0, full_reference_parity=True, new_training_updates=14*2000)
    return payload, rows, timing


def fit(a, fold, replay=False):
    folder = OUT / f'fold{fold}'
    if not replay and (folder / 'payload.json').exists():
        if (folder / 'validation.json').exists():
            v = read(folder / 'validation.json')
            need(v['authority'] == bind(AUTH) and v['payload'] == bind(folder / 'payload.json')
                 and v['status'] == 'H593_ABLATION_FRESH_REPLAY_NUMPY_PASS', 'RESUME_VALIDATED_FOLD')
            print(json.dumps(dict(event='ALREADY_VALIDATED', fold=fold)), flush=True)
        else:
            subprocess.run([sys.executable, __file__, 'verify', '--fold', str(fold)], check=True)
        return
    payload, rows, timing = compute(a, fold)
    if not replay:
        write(folder / 'payload.json', payload)
        write(folder / 'runtime.json', dict(job_id=os.environ['SLURM_JOB_ID'], fits=timing, threads=torch.get_num_threads()))
        subprocess.run([sys.executable, __file__, 'verify', '--fold', str(fold)], check=True)
        return
    need(payload == read(folder / 'payload.json'), 'FRESH_PROCESS_FIT_AND_LOGIT_BITS')
    by_id = {r['query_id']: r for r in rows}
    maximum, checks = 0., 0
    for prediction in payload['predictions']:
        row = by_id[prediction['query_id']]
        x = row['modes']['REAL']['X'].numpy()
        for name, par in payload['parameters'].items():
            theta = np.array([float.fromhex(v) for v in par['theta_hex']])
            z = np.sum(x[:, par['retained_indices']] * theta[:-1], axis=1) + theta[-1]
            expected = prediction['models'][name]
            error = float(np.max(np.abs(z - np.array([float.fromhex(v) for v in expected['logits_hex']]))))
            need(error < 2e-10, 'NUMPY_LOGITS')
            need(choose(z, row) == expected['selected'], 'NUMPY_ACTION')
            maximum = max(maximum, error); checks += len(z)
    write(folder / 'validation.json', dict(status='H593_ABLATION_FRESH_REPLAY_NUMPY_PASS', authority=bind(AUTH),
                                           payload=bind(folder / 'payload.json'), max_abs_error=maximum,
                                           logit_checks=checks, heldout_label_reads=0, full_reference_parity=True))


def group_comparison(rows, full, ablated):
    groups = defaultdict(list)
    for r in rows:
        groups[r['component']].append(int(r['correct'][full]) - int(r['correct'][ablated]))
    delta = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(20260920)
    boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
    flips = np.random.default_rng(20260920).choice([-1., 1.], size=(9999, len(delta)))
    p = (1 + np.count_nonzero((flips * delta).mean(1) >= delta.mean()-1e-15)) / 10000
    return dict(equal_component_full_minus_ablation=float(delta.mean()), bootstrap95=list(map(float, np.quantile(boot, [.025, .975]))),
                signflip_one_sided_p=float(p), components=len(delta))


def seal_and_roles(a):
    folds, validations = [], []
    for f in range(5):
        v = read(OUT / f'fold{f}/validation.json')
        need(v['status'] == 'H593_ABLATION_FRESH_REPLAY_NUMPY_PASS' and v['authority'] == bind(AUTH), 'ALL_FOLDS_VALIDATED')
        p = read(checked(v['payload']))
        need(p['fold'] == f and p['authority'] == bind(AUTH), 'FOLD_BINDING')
        folds.append(p); validations.append(bind(OUT / f'fold{f}/validation.json'))
    write(OUT / 'all_predictions_prelabel_seal.json', dict(authority=bind(AUTH), fold_validations=validations))
    roles = {r['query_id']: r for r in read(checked(a['join_sources']['curator']))['records']}
    return folds, roles


def join(a):
    folds, roles = seal_and_roles(a)
    labels, _ = N.P.gallery_labels()
    features, _ = H.features(); feature = {r['query_id']: r for r in features}
    prior = read(checked(a['join_sources']['old_result']))
    old = {r['query_id']: r for r in prior['rows']}
    rows = []
    for p in folds:
        train = read(checked(a['fold_sources'][str(p['fold'])]['train_roles']))['records']
        identities = {r['identity'] for r in train}; components = {r['component'] for r in train}
        for pred in p['predictions']:
            q = pred['query_id']; role = roles[q]; row = feature[q]
            need(role['outer_fold'] == p['fold'] and role['identity'] not in identities and role['component'] not in components, 'GROUP_IDENTITY_DISJOINT')
            raw_selected = row['candidate_physical_rows'][row['winner']]
            selected = dict(RAW=raw_selected, **{m:v['selected'] for m,v in pred['models'].items()})
            correct = {m:labels[s] == role['identity'] for m,s in selected.items()}
            ranked = row['raw_ranked_physical_rows']
            target_rank = next(i+1 for i,s in enumerate(ranked) if labels[s] == role['identity'])
            ranks = {m:1 if correct[m] else target_rank + int(ranked.index(s)+1 > target_rank) for m,s in selected.items()}
            for kind in KINDS:
                need(correct[kind+'_FULL'] == old[q]['correct']['ALL_CE' if kind == 'CE' else kind], 'FULL593_DECISIONS')
            need(correct['RAW'] == old[q]['correct']['RAW'], 'RAW_PARITY')
            rows.append(dict(query_id=q, original_query_id=role['original_query_id'], fold=p['fold'],
                             identity=role['identity'], component=role['component'], target_in_C128=target_rank<=128,
                             selected=selected, correct=correct, ranks=ranks))
    rows.sort(key=lambda r:r['query_id'])
    need(len(rows) == len({r['query_id'] for r in rows}) == 593, 'FULL593')
    need(sum(r['target_in_C128'] for r in rows) == 570, 'ABSENT23_RETAINED')
    need(len({r['component'] for r in rows}) == 64 and len({r['identity'] for r in rows}) == 68, 'GROUPS_IDENTITIES')
    summary = {}
    for m in rows[0]['correct']:
        full = m.split('_')[0]+'_FULL' if m != 'RAW' else 'RAW'
        stat = dict(correct=sum(r['correct'][m] for r in rows),
                    MRR=sum(1/r['ranks'][m] for r in rows)/593,
                    rescue_vs_RAW=sum(not r['correct']['RAW'] and r['correct'][m] for r in rows),
                    break_vs_RAW=sum(r['correct']['RAW'] and not r['correct'][m] for r in rows),
                    rescue_vs_FULL=sum(not r['correct'][full] and r['correct'][m] for r in rows),
                    break_vs_FULL=sum(r['correct'][full] and not r['correct'][m] for r in rows),
                    changed_vs_FULL=sum(r['selected'][full] != r['selected'][m] for r in rows),
                    switches=sum(r['selected']['RAW'] != r['selected'][m] for r in rows),
                    fold_correct={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)})
        if m not in ('RAW', 'COST1_FULL', 'CE_FULL'):
            stat.update(group_comparison(rows, full, m))
        summary[m] = stat
    for kind in KINDS:
        for method in ('ZERO', 'REFIT'):
            names = sorted((f'{kind}_{method}_{f}' for f in FEATURES), key=lambda n:summary[n]['signflip_one_sided_p'])
            running = 0.
            for i,m in enumerate(names):
                running = max(running, min(1., (6-i)*summary[m]['signflip_one_sided_p']))
                summary[m]['holm_adjusted_p_within_six'] = running
    need([summary[m]['correct'] for m in ('RAW', 'COST1_FULL', 'CE_FULL')] == [426,481,486], 'BASELINE_COUNTS')
    result = dict(status='H593_SIX_FEATURE_ABLATION_COMPLETE', authority=bind(AUTH), population=593,
                  recall_present=570, target_absent=23, folds=5, components=64, identities=68,
                  primary='COST1', features=list(FEATURES), summary=summary, rows=rows,
                  evidence_level=a['evidence_level'], external_confirmation=False,
                  automatic_deployment_change=False, new_encoder_forwards=0,
                  limits=['Frozen zeroing is computational dependence, not physical cue deletion.',
                          'Refits measure this fixed learner and shared OOF development population.',
                          'Coupled features can substitute information; no universal necessity or minimality claim.',
                          'Holm families are six features per loss and intervention; inference is descriptive after development.'])
    write(OUT / 'result.json', result)
    subprocess.run([sys.executable, __file__, 'join-verify'], check=True)
    report = ['# H593 六项证据逐项消融', '',
              '原 H593 grouped OOF5、RAW 自然 C128、原六统计、FP64/2000 updates；COST1 为主、CE 为对照。全部593图纳入，包含23个候选缺失。', '',
              'FULL参数和原始127分数逐位复现封存结果；每个REFIT在独立新进程重训重放，NumPy核算通过。ZERO只置零该权重且保留bias；REFIT删除该输入列后重训其余五权重和bias。', '',
              '| 模型 | 正确/593 | 对RAW救/损 | 对FULL救/损 | MRR |', '|---|---:|---:|---:|---:|']
    for kind in KINDS:
        for name in [kind+'_FULL']+[f'{kind}_{method}_{f}' for method in ('ZERO','REFIT') for f in FEATURES]:
            s=summary[name]
            report.append(f"| {name} | {s['correct']} | {s['rescue_vs_RAW']}/{s['break_vs_RAW']} | {s['rescue_vs_FULL']}/{s['break_vs_FULL']} | {s['MRR']:.6f} |")
    report += ['', 'RAW：426/593。对FULL救/损以相应COST1或CE完整头为基准。', '',
               '## 配对分组结果', '', '| 模型 | FULL−消融：等组差pp | bootstrap95% pp | Holm p（六项） |', '|---|---:|---|---:|']
    for kind in KINDS:
        for method in ('ZERO','REFIT'):
            for f in FEATURES:
                name=f'{kind}_{method}_{f}';s=summary[name];lo,hi=s['bootstrap95']
                report.append(f"| {name} | {100*s['equal_component_full_minus_ablation']:.3f} | [{100*lo:.3f}, {100*hi:.3f}] | {s['holm_adjusted_p_within_six']:.4f} |")
    report += ['', '正差表示完整头较好；区间为64 component等权差，不是图片微平均差。H593已经开发；本表不作为新外部确认，也不按结果替换部署模型。', '',
               '删项不消除该特征在其它耦合统计中的全部信息。固定头下降不能证明重新训练无法补偿；重训无下降也不证明底层视觉线索无用。', '',
               '原始参数及全部127分数：fold0/ 到 fold4/ 的 payload.json。完整逐查询结果和配对统计：result.json。复核：validation.json 和各fold/validation.json。']
    N.P.write_bytes(OUT / 'report_zh.md', ('\n'.join(report)+'\n').encode())
    print(json.dumps({m:s['correct'] for m,s in summary.items()}), flush=True)


def verify_join(a):
    folds, roles = seal_and_roles(a)
    labels, _ = N.P.gallery_labels()
    result = read(OUT / 'result.json')
    independent, decisions = defaultdict(int), {}
    for fold in folds:
        for pred in fold['predictions']:
            role = roles[pred['query_id']]
            axis = pred['candidate_physical_rows']; raw = axis[pred['winner']]
            chosen = {'RAW':raw}
            for name,v in pred['models'].items():
                scores = [float.fromhex(h) for h in v['logits_hex']]
                need(len(scores)==127 and all(np.isfinite(scores)), 'FINITE_ALL127')
                k=max(range(127), key=lambda j:scores[j])
                chosen[name] = axis[pred['challenger_positions'][k]] if scores[k]>0 else raw
                need(chosen[name]==v['selected'], 'INDEPENDENT_FULL_AXIS_ACTION')
            correct={m:labels[s]==role['identity'] for m,s in chosen.items()}
            decisions[pred['query_id']] = correct
            for m,v in correct.items(): independent[m] += int(v)
    need(len(decisions)==593, 'INDEPENDENT593')
    for row in result['rows']:
        need(row['correct']==decisions[row['query_id']], 'INDEPENDENT_LABEL_JOIN')
    for m,s in result['summary'].items():
        need(independent[m]==s['correct'], 'INDEPENDENT_COUNTS')
        full=m.split('_')[0]+'_FULL' if m!='RAW' else 'RAW'
        for ref,tag in [('RAW','RAW'),(full,'FULL')]:
            rescue=sum(v[m] and not v[ref] for v in decisions.values())
            broken=sum(v[ref] and not v[m] for v in decisions.values())
            need((rescue,broken)==(s['rescue_vs_'+tag],s['break_vs_'+tag]), 'INDEPENDENT_TRANSITIONS')
        if m not in ('RAW','COST1_FULL','CE_FULL'):
            check=group_comparison(result['rows'],full,m)
            need(all(s[k]==v for k,v in check.items()), 'STATISTICS_REPLAY')
    write(OUT/'validation.json',dict(status='H593_ABLATION_INDEPENDENT_ACTION_COUNTS_PASS',authority=bind(AUTH),
                                    result=bind(OUT/'result.json'), queries=593, model_conditions=26,
                                    challenger_logit_checks=593*26*127, all_labels_and_counts_reconstructed=True,
                                    baseline_counts=dict(RAW=426,COST1=481,CE=486)))


def preflight():
    torch.manual_seed(17)
    for tied in (True, False):
        z = (torch.zeros(4,127,dtype=torch.float64) if tied else torch.randn(4,127,dtype=torch.float64)).requires_grad_()
        y=torch.tensor([-1,0,31,126]); batch=L.unit(z,y)
        scalar=torch.stack([F.softplus(v.max()) if t==-1 else F.softplus(-v[t])+F.softplus(torch.cat((v[:t],v[t+1:])).max()) for v,t in zip(z,y)]).mean()
        need(torch.allclose(batch,scalar,rtol=0,atol=1e-14), 'SCALAR_COST1')
        need(torch.allclose(torch.autograd.grad(batch,z,retain_graph=True)[0],torch.autograd.grad(scalar,z)[0],rtol=0,atol=1e-14),'TIE_GRADIENT')
        ce=L.N.R.objective(z,y+1)
        independent=F.cross_entropy(torch.cat([z.new_zeros(4,1),z],1),y+1)
        need(torch.allclose(ce,independent,rtol=0,atol=1e-14),'CE_HOLD_CLASS')
    x=torch.randn(3,127,6,dtype=torch.float64)
    for i in range(6):
        keep=[j for j in range(6) if j!=i];theta=torch.randn(6,dtype=torch.float64)
        modified=x.clone();modified[...,i]+=1000
        need(torch.equal(score(x,theta,keep),score(modified,theta,keep)), 'DROPPED_INPUT_INVARIANCE')
    row=dict(candidate_physical_rows=[3,9,12],challenger_positions=[0,2],winner=1)
    need(choose(np.array([0.,0.]),row)==9 and choose(np.array([1.,1.]),row)==3, 'HOLD_AND_TIE')
    write(OUT/'preflight.json',dict(status='ABLATION_PREFLIGHT_PASS',authority=bind(AUTH),natural_training_updates=0,
                                   checks=['scalar_cost1_and_tie_gradients','CE_HOLD_class','six_dropped_input_invariance','HOLD_zero_and_axis_tie']))
    print('ABLATION_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','preflight','fit','verify','join','join-verify']);parser.add_argument('--fold',type=int)
    args=parser.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        authority=guard(args.stage,args.fold)
        if args.stage=='preflight':preflight()
        elif args.stage=='join':join(authority)
        elif args.stage=='join-verify':verify_join(authority)
        else:fit(authority,args.fold,args.stage=='verify')
