#!/usr/bin/env python3
"""Independent NumPy action audit and post-seal label join for the internal-M pilot.

No model inference, fitting, checkpoint selection, or job submission occurs here.
Every saved C128 content vector and 127-challenger action is verified before the
held curator file is opened. Correctness compares gallery identities, never rows.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_prellm_m_adapter_v1'
AUTH = ROOT / 'registry/rc_prellm_m_adapter_authority_v1_20260924.json'
REPORT = ROOT / 'reports/REPORT_COLNOMIC_INTERNAL_M_PILOT_V1_20260924.md'
ARMS = ('PRE_REAL', 'PRE_CONSTANT', 'POST_REAL', 'POST_CONSTANT')
PRIMARY = ('RAW', 'OLD_EXTERNAL_M', 'FRESH_ZERO') + ARMS
DIAGNOSTIC = ('PRE_REAL_SHUFFLED', 'POST_REAL_SHUFFLED')
TOL = 1e-10


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).absolute()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(binding):
    need(bind(binding['path']) == binding, 'JOIN_SHA_DRIFT:' + binding['path'])
    return Path(binding['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    need(result.shape == (size,) and np.isfinite(result).all(), 'FINITE_SHAPE:' + name)
    return result


def compare(left, right, name):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    need(left.shape == right.shape and np.isfinite(left).all()
         and np.isfinite(right).all(), 'COMPARISON_SHAPE:' + name)
    error = float(np.max(np.abs(left - right))) if left.size else 0.
    need(error <= TOL, f'NUMPY_PARITY:{name}:{error}')
    return error


def recompute(row, content, theta):
    """Rebuild all 127 features/logits independently of the Torch producer."""
    raw = vector(row['raw_scores'], 128, 'RAW')
    mass = vector(row['M'], 128, 'M')
    content = vector(content, 128, 'L')
    theta = vector(theta, 5, 'HEAD')
    axis = row['candidate_ids']
    need(len(axis) == len(set(axis)) == 128 and axis == sorted(axis), 'ORIGINAL_PHYSICAL_AXIS')
    need(len(row['candidate_identities']) == len(set(row['candidate_identities'])) == 128,
         'IDENTITY_DEDUPLICATED_C128')
    winner = int(row['winner_index'])
    need(0 <= winner < 128 and winner == int(np.argmax(raw)), 'RAW_WINNER_STABLE_TIE')
    idx = np.asarray([i for i in range(128) if i != winner], dtype=np.int64)
    need(row['challenger_positions'] == idx.tolist(), 'ORIGINAL_CHALLENGER_AXIS')
    def sym(values):
        return (values[idx] - values[winner]) / (np.abs(values[idx]) + abs(values[winner]) + 1e-12)
    x = np.column_stack(((raw[idx] - raw[winner]) / max(float(raw.std(ddof=0)), 1e-12),
                         sym(mass * content), sym(mass), sym(content)))
    z = x @ theta[:4] + theta[4]
    need(np.isfinite(x).all() and np.isfinite(z).all(), 'FINITE_HEAD_ARITHMETIC')
    best = int(np.argmax(z))
    position = int(idx[best]) if z[best] > 0 else winner
    return dict(L=content.tolist(), features=x.tolist(), logits=z.tolist(),
                challenger_positions=idx.tolist(), prediction_position=position,
                prediction_id=axis[position], prediction_identity=row['candidate_identities'][position],
                switched=position != winner, max_challenger_logit=float(z[best]))


def verify_decision(saved, fresh, name):
    error = compare(vector(saved['logits'], 127, name), fresh['logits'], name)
    for key in ('challenger_positions', 'prediction_position', 'prediction_id', 'switched'):
        need(saved[key] == fresh[key], 'DECISION_PARITY:' + name + ':' + key)
    return error


def prelabel_audit(a, m):
    """Fail closed on any missing/stale prediction, before opening held labels."""
    authority = bind(AUTH)
    need(read(AUTH) == a and read(checked(a['manifest'])) == m, 'AUTHORITY_MANIFEST_CURRENT')
    need(tuple(a['arms']) == ARMS, 'EXACT_FOUR_ARMS')
    for source in a['code_sources']:
        checked(source)
    inputs = m['train_rows'] + m['probe_rows']
    query_ids = [r['query_id'] for r in inputs]
    need(len(m['train_rows']) == a['train_count'] == 16
         and len(m['probe_rows']) == a['probe_count'] == 8
         and len(set(query_ids)) == 24, 'SIXTEEN_TRAIN_EIGHT_PROBE')
    need(not any(k.startswith('target') or k in ('identity', 'group', 'component')
                 for row in m['probe_rows'] for k in row), 'PROBE_MANIFEST_LABEL_FREE')
    seal_binding = bind(OUT / 'all_predictions_prelabel_seal.json')
    seal = read(checked(seal_binding))
    need(seal['authority'] == authority and seal['held_label_reads'] == 0, 'PRELABEL_SEAL')
    need(seal['seals'] == [bind(OUT / arm / 'prediction_seal.json') for arm in ARMS],
         'ALL_FOUR_ARM_SEALS')
    head = read(checked(m['frozen_head']['parent']))
    head_validation = read(checked(m['frozen_head']['validation']))
    need(head_validation['payload'] == m['frozen_head']['parent']
         and head_validation['status'] == 'QUALITY_OPERATOR_EVAL_FRESH_NUMPY_PASS', 'OLD_HEAD_VALIDATED')
    old_theta = head['parameters']['COST1_REFIT_M1Q0R0']['theta_hex']
    need([old_theta[i] for i in (0, 1, 2, 3, 6)] == m['frozen_head']['theta_hex']
         and float.fromhex(old_theta[4]) == float.fromhex(old_theta[5]) == 0., 'FROZEN_HEAD_BITS')
    theta = [float.fromhex(x) for x in m['frozen_head']['theta_hex']]
    need(theta == m['frozen_head']['theta'] and head['fold'] == 0
         and head['heldout_label_reads'] == 0, 'ORIGINAL_FOLD0_TRAIN_HEAD')
    prior_predictions = {r['query_id']: r for r in head['predictions']}
    gallery = read(checked(m['provenance']['gallery']))['records']
    labels = {r['physical_row']: r['identity'] for r in gallery}
    need(len(labels) == len(gallery), 'GALLERY_UNIQUE_PHYSICAL_ROWS')
    all_cache = read(OUT / 'encoder_cache/validation.json')
    need(all_cache['authority'] == authority and all_cache['status'] == 'ENCODER_CACHE_PASS', 'CACHE_COMPLETE')
    need(all_cache['queries'] == [bind(OUT/'encoder_cache'/q/'validation.json') for q in query_ids],
         'EXACT_QUERY_CACHES')
    audited = {}
    errors = []
    sources = [seal_binding, m['frozen_head']['parent'], m['frozen_head']['validation'],
               m['provenance']['gallery'], bind(OUT/'encoder_cache/validation.json')]
    for row in inputs:
        q = row['query_id']
        need([labels[x] for x in row['candidate_ids']] == row['candidate_identities'],
             'GALLERY_CANDIDATE_IDENTITIES')
        cache_dir = OUT / 'encoder_cache' / q
        cv = read(cache_dir / 'validation.json')
        need(cv['authority'] == authority and cv['status'] == 'QUERY_ENCODER_CACHE_PASS'
             and cv['held_labels_read'] is False, 'QUERY_CACHE_PRELABEL')
        checked(cv['payload'])
        parity = read(checked(cv['parity']))
        need(parity['query_id'] == q, 'QUERY_PARITY_ID')
        source = recompute(row, row['L0'], theta)
        fresh = recompute(row, parity['fresh_L0'], theta)
        errors.append(compare(parity['original_L0'], row['L0'], q + ':ORIGINAL_L0'))
        errors.append(verify_decision(parity['original'], source, q + ':ORIGINAL'))
        errors.append(verify_decision(parity['fresh'], fresh, q + ':FRESH'))
        need(parity['cache_forward_error'] <= a['cache_parity_atol']
             and parity['token_mean_cosine'] >= a['source_parity']['mean_cosine_min']
             and max(abs(x-y) for x,y in zip(fresh['L'], source['L'])) <= a['source_parity']['max_content_error']
             and source['prediction_id'] == fresh['prediction_id'], 'FRESH_SOURCE_GATE')
        need(len(cv['zero_adapter']) == 8
             and {v['arm'] for v in cv['zero_adapter']} == set(ARMS)
             and all(v['error'] <= a['cache_parity_atol'] for v in cv['zero_adapter']), 'ALL_ZERO_ADAPTER_GATES')
        if row in m['probe_rows']:
            old = prior_predictions[q]['models']['COST1_REFIT_M1Q0R0']
            errors.append(compare([float.fromhex(v) for v in old['logits_hex']], source['logits'], q + ':OLD_PUBLISHED'))
            need(old['selected'] == source['prediction_id'], 'PUBLISHED_EXTERNAL_BASELINE_ACTION')
        audited[q] = dict(OLD_EXTERNAL_M=source, FRESH_ZERO=fresh)
        sources += [bind(cache_dir/'validation.json'), cv['payload'], cv['parity']]
    for arm, arm_binding in zip(ARMS, seal['seals']):
        arm_seal = read(checked(arm_binding))
        fit = read(OUT / arm / 'fit_validation.json')
        need(fit['authority'] == authority and fit['status'] == 'FIXED_TRAIN_UPDATES_COMPLETE'
             and fit['steps'] == a['updates'] == 16 and fit['backbone_trainable'] == 0
             and fit['head_trainable'] == 0 and fit['held_label_reads'] == 0, 'FIXED_FINAL_ADAPTER')
        checked(fit['adapter'])
        checked(fit['checkpoint'])
        need(arm_seal['authority'] == authority and arm_seal['status'] == 'PREDICTIONS_SEALED'
             and arm_seal['adapter'] == fit['adapter'] and arm_seal['held_label_reads'] == 0, 'ARM_SEAL_VALID')
        expected = [(split, row, intervention)
                    for split, rows in (('train', m['train_rows']), ('probe', m['probe_rows']))
                    for row in rows
                    for intervention in (('native', 'shuffled') if arm.endswith('REAL') else ('native',))]
        paths = [OUT / arm / 'predictions' / split / (row['query_id'] + '_' + intervention + '.json')
                 for split, row, intervention in expected]
        need(arm_seal['predictions'] == [bind(path) for path in paths], 'EXACT_COMPLETE_ARM_PREDICTIONS')
        for (_, row, intervention), binding in zip(expected, arm_seal['predictions']):
            pred = read(checked(binding))
            q = row['query_id']
            need(pred['authority'] == authority and pred['adapter'] == fit['adapter']
                 and pred['query_id'] == q and pred['arm'] == arm
                 and pred['intervention'] == intervention and pred['held_label_reads'] == 0,
                 'PREDICTION_LINEAGE')
            for field in ('M', 'raw_scores', 'candidate_ids'):
                need(pred[field] == row[field], 'PREDICTION_ORIGINAL_AXIS:' + field)
            key = arm if intervention == 'native' else arm + '_SHUFFLED'
            fresh = recompute(row, pred['L'], theta)
            errors.append(verify_decision(pred['decision'], fresh, q + ':' + key))
            errors.append(verify_decision(pred['source_baseline'], audited[q]['OLD_EXTERNAL_M'], q + ':SOURCE'))
            errors.append(verify_decision(pred['fresh_baseline'], audited[q]['FRESH_ZERO'], q + ':ZERO'))
            audited[q][key] = fresh
            sources.append(binding)
        sources += [arm_binding, bind(OUT/arm/'fit_validation.json'), fit['adapter'], fit['checkpoint']]
    audit_files = []
    for row in inputs:
        q = row['query_id']
        need(set(audited[q]) == set(PRIMARY[1:] + DIAGNOSTIC), 'EVERY_NATIVE_AND_SHUFFLED_MODE')
        path = OUT / 'independent_numpy' / (q + '.json')
        write(path, dict(status='FULL_C128_AND_127_LOGITS_NUMPY_PASS', authority=authority,
                         query_id=q, candidate_ids=row['candidate_ids'],
                         candidate_identities=row['candidate_identities'], M=row['M'],
                         raw_scores=row['raw_scores'], winner_index=row['winner_index'],
                         theta=theta, models=audited[q], held_label_reads=0))
        audit_files.append(bind(path))
    integrity = dict(status='ALL_PREDICTIONS_INDEPENDENTLY_VERIFIED_BEFORE_HELD_LABELS',
                     authority=authority, prelabel_seal=seal_binding, sources=sources,
                     query_audits=audit_files, queries=24, native_predictions=96,
                     shuffled_inference_diagnostics=48, all128_L=True, all127_logits=True,
                     max_numpy_logit_or_content_error=max(errors, default=0.), tolerance=TOL,
                     held_label_reads=0)
    write(OUT / 'independent_prelabel_validation.json', integrity)
    return audited, integrity, labels


def effect(rows, model, baseline):
    gained = [r['query_id'] for r in rows if r['correct'][model] and not r['correct'][baseline]]
    lost = [r['query_id'] for r in rows if r['correct'][baseline] and not r['correct'][model]]
    base_correct = sum(r['correct'][baseline] for r in rows)
    return dict(baseline=baseline, rescued=len(gained), breaks=len(lost), net=len(gained)-len(lost),
                baseline_correct=base_correct,
                baseline_correct_loss_rate=len(lost)/base_correct if base_correct else None,
                changed_decisions=sum(r['selected_identity'][model] != r['selected_identity'][baseline] for r in rows),
                rescued_query_ids=gained, broken_query_ids=lost)


def summarize(rows, models):
    return dict(n=len(rows), components=len({r['component'] for r in rows}),
                groups=len({r['group'] for r in rows if r['group'] is not None}),
                target_in_C128=sum(r['target_in_C128'] for r in rows),
                models={model: dict(correct=sum(r['correct'][model] for r in rows),
                    accuracy=sum(r['correct'][model] for r in rows)/len(rows) if rows else None,
                    versus_RAW=effect(rows, model, 'RAW'),
                    versus_OLD_EXTERNAL_M=effect(rows, model, 'OLD_EXTERNAL_M')) for model in models})


def join(a, m):
    # No held role file may be read above this call's successful return.
    audited, integrity, gallery = prelabel_audit(a, m)
    authority = bind(AUTH)
    parent = read(checked(m['provenance']['parent']))
    split = read(checked(m['provenance']['split']))['folds'][0]
    train_binding = m['provenance']['train_roles']
    train_roles = {r['query_id']: r for r in read(checked(train_binding))['records']}
    need(set(train_roles) == set(split['train_query_ids']), 'FULL_FOLD0_TRAIN_ROLES')
    need(set(r['query_id'] for r in m['train_rows']) <= set(train_roles)
         and set(r['query_id'] for r in m['probe_rows']) <= set(split['heldout_query_ids'])
         and not set(split['train_query_ids']) & set(split['heldout_query_ids']), 'OUTER_SPLIT_MEMBERSHIP')
    curator_binding = parent['join_sources']['curator']
    curator = {r['query_id']: r for r in read(checked(curator_binding))['records']}
    train_components = {r['component'] for r in train_roles.values()}
    train_identities = {r['identity'] for r in train_roles.values()}
    train_groups = {r['group'] for r in train_roles.values() if r.get('group') is not None}
    rows = []
    for split_name, inputs in (('train', m['train_rows']), ('probe', m['probe_rows'])):
        for row in inputs:
            q = row['query_id']
            role = train_roles[q] if split_name == 'train' else curator[q]
            if split_name == 'probe':
                need(role['outer_fold'] == 0 and role['component'] not in train_components
                     and role['identity'] not in train_identities, 'OUTER_GROUP_COMPONENT_IDENTITY_DISJOINT')
                if role.get('group') is not None:
                    need(role['group'] not in train_groups, 'OUTER_GROUP_DISJOINT')
            else:
                need(row['component'] == role['component'] and row['target_id'] == role['identity']
                     and row.get('group') == role.get('group'), 'TRAIN_ROLE_IDENTITY_PARITY')
            selected = dict(RAW=row['candidate_ids'][row['winner_index']],
                            **{key: value['prediction_id'] for key, value in audited[q].items()})
            selected_identity = {key: gallery[value] for key, value in selected.items()}
            correct = {key: identity == role['identity'] for key, identity in selected_identity.items()}
            targets = [i for i, identity in enumerate(row['candidate_identities']) if identity == role['identity']]
            need(len(targets) <= 1, 'IDENTITY_TARGET_DEDUP')
            if split_name == 'train':
                need(targets == row['target_positions'] and correct['RAW'] == row['raw_correct'], 'TRAIN_TARGET_PARITY')
            rows.append(dict(query_id=q, split=split_name, fold=0, identity=role['identity'],
                             component=role['component'], group=role.get('group'),
                             target_in_C128=bool(targets), target_positions=targets,
                             selected_physical_row=selected, selected_identity=selected_identity, correct=correct,
                             intermediate_127_scores=bind(OUT/'independent_numpy'/(q+'.json'))))
    tr = [r for r in rows if r['split'] == 'train']
    pr = [r for r in rows if r['split'] == 'probe']
    need(sum(r['correct']['RAW'] for r in tr) == 8, 'EIGHT_CORRECT_EIGHT_WRONG_TRAIN')
    grouping = dict(full_fold_train_components=len(train_components),
                    selected_train_components=len({r['component'] for r in tr}),
                    probe_components=len({r['component'] for r in pr}),
                    probe_groups=len({r['group'] for r in pr if r['group'] is not None}),
                    probe_group_metadata_count=sum(r['group'] is not None for r in pr),
                    group_disjoint_checked_where_available=True, component_identity_disjoint=True)
    summary = {name: summarize(part, PRIMARY) for name, part in (('TRAIN16', tr), ('PROBE8', pr))}
    diagnostic = {name: dict(summary=summarize(part, DIAGNOSTIC),
                            native_comparison={key: effect(part, key, key.removesuffix('_SHUFFLED')) for key in DIAGNOSTIC})
                  for name, part in (('TRAIN16', tr), ('PROBE8', pr))}
    result = dict(status='INTERNAL_M_PILOT_POSTSEAL_JOIN_COMPLETE', authority=authority,
                  manifest=a['manifest'], frozen_head=m['frozen_head'],
                  independent_prelabel_validation=bind(OUT/'independent_prelabel_validation.json'),
                  label_sources=dict(train=train_binding, held_curator=curator_binding),
                  lineage=dict(model=a['model'], fold=0, dataset='Opened H593 development panel',
                               candidates='Original natural RAW C128, immutable physical-row and identity axes',
                               scoring='COST1_REFIT_M1Q0R0 fixed; true external M in every arm; internal query encoder adapter',
                               baselines=['RAW', 'OLD_EXTERNAL_M', 'FRESH_ZERO'],
                               selection='Fixed terminal update16, no probe-based checkpoint selection'),
                  grouping=grouping, rows=rows, summary=summary,
                  internal_M_misbinding_inference_diagnostic=diagnostic,
                  external_GO=False, generalization_confirmed=False,
                  limitations=['TRAIN16 is a fitting/engineering pilot stratified to eight RAW-correct and eight RAW-wrong queries.',
                               'PROBE8 is outcome-blind outer-held selection on an already opened development dataset; actual group counts are reported.',
                               'Eight probe images do not establish generalization; no checkpoint, hyperparameter, or arm is selected from these outcomes.',
                               'Shuffled internal M is an inference-only diagnostic; the external scoring M is unchanged and no shuffled adapter is fitted.',
                               'NumPy verifies saved C128 content-to-head arithmetic and identity decisions; it does not independently reproduce encoder outputs.'])
    write(OUT/'result.json', result)
    validation = dict(status='INTERNAL_M_PILOT_NUMPY_IDENTITY_JOIN_PASS', authority=authority,
                      result=bind(OUT/'result.json'), prelabel_validation=bind(OUT/'independent_prelabel_validation.json'),
                      queries=24, train=16, probe=8, grouping=grouping,
                      max_numpy_error=integrity['max_numpy_logit_or_content_error'], tolerance=TOL,
                      all_predictions_sealed_and_verified_before_held_labels=True,
                      all128_L_and_all127_logits_saved=True, identity_correctness=True,
                      frozen_head=True, fixed_terminal_checkpoint=True,
                      external_GO=False, generalization_confirmed=False)
    write(OUT/'validation.json', validation)
    publish(result, validation)
    print(json.dumps(dict(status=result['status'], result=bind(OUT/'result.json'),
                          validation=bind(OUT/'validation.json')), ensure_ascii=False), flush=True)
    return result


def publish(result, validation):
    lines = ['# ColNomic internal-M pilot v1 — 2026-09-24', '',
             '固定模型：ColNomic 7B（原检索 LoRA 保留，主干冻结）。数据：已打开的 H593 开发集，原始 fold0；'
             '16 张 TRAIN（8 张 RAW 正确、8 张 RAW 错误）及按预冻结 SHA 顺序选出的 8 张 outer-held 探针。', '',
             '候选：原始自然 RAW C128。评分：冻结 COST1_REFIT_M1Q0R0，所有实验臂的输出评分都继续使用真实外部 M。'
             'PRE 在 LLM 前调制视觉 token，POST 在 LLM 后、检索投影前调制；REAL 与 CONSTANT 使用相同参数量和训练顺序。', '',
             '四臂固定训练到第 16 次更新；所有预测和 127 个 challenger 分数先封存并经 NumPy 独立重算，再读取 held 标签。'
             '没有按探针结果选 checkpoint。', '',
             f"独立重算最大误差：{validation['max_numpy_error']:.3g}（阈值 {TOL:g}）。正确性按 gallery identity 判断。", '',
             f"实际组数：TRAIN {result['grouping']['selected_train_components']} 个 component；"
             f"PROBE {result['grouping']['probe_components']} 个 component / {result['grouping']['probe_groups']} 个 group。"
             '探针与完整 fold0 TRAIN 的 component、identity 分离，存在 group 元数据时同时核验 group 分离。', '']
    for panel, summary in result['summary'].items():
        lines += [f'## {panel}', '',
                  '| 路径 | 正确/总数 | 对 RAW 救回/破坏 | 对旧外部 M 救回/破坏 | 对旧外部 M 改动决策 |',
                  '|---|---:|---:|---:|---:|']
        for key, item in summary['models'].items():
            raw, old = item['versus_RAW'], item['versus_OLD_EXTERNAL_M']
            lines.append(f"| {key} | {item['correct']}/{summary['n']} | {raw['rescued']}/{raw['breaks']} | "
                         f"{old['rescued']}/{old['breaks']} | {old['changed_decisions']} |")
        lines += ['', f"C128 含目标：{summary['target_in_C128']}/{summary['n']}。"
                  '每条救回/破坏的 query ID、原正确样本损失率保存在 result.json。', '']
    lines += ['## 内部 M 错绑诊断', '',
              '只在推理时把内部 M 按候选轴循环错移一位；外部评分的 M 保持真实绑定。没有训练错绑模型，因此只解释推理敏感性。', '',
              '| 子集 | 诊断 | 正确/总数 | 相对对应 native 救回/破坏 | 决策改动 |', '|---|---|---:|---:|---:|']
    for panel, entry in result['internal_M_misbinding_inference_diagnostic'].items():
        for key, item in entry['summary']['models'].items():
            effect_ = entry['native_comparison'][key]
            lines.append(f"| {panel} | {key} | {item['correct']}/{entry['summary']['n']} | "
                         f"{effect_['rescued']}/{effect_['breaks']} | {effect_['changed_decisions']} |")
    lines += ['', '## 证据边界', '',
              '这是拟合可行性与已打开开发集的小样本探针。8 张图片及其实际组数不能支持泛化确认，'
              '也不据此自动扩大实验、选择超参数或宣称优于完整五折 COST1。', '',
              '结果：[result.json](../results/rc_prellm_m_adapter_v1/result.json)；'
              '验证：[validation.json](../results/rc_prellm_m_adapter_v1/validation.json)；'
              '每张图完整 C128 L、127×4 特征和 127 logits：'
              '[independent_numpy](../results/rc_prellm_m_adapter_v1/independent_numpy)。', '']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temp = REPORT.with_name('.' + REPORT.name + f'.{os.getpid()}.tmp')
    temp.write_text('\n'.join(lines))
    os.replace(temp, REPORT)
