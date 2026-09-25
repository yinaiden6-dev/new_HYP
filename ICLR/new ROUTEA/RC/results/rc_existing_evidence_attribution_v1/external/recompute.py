#!/usr/bin/env python3
"""Read-only arithmetic attribution of existing original seven-parameter COST1.

Writes only this new sidecar directory. No models are trained or run, no new
labels are joined, and no original sealed source is changed. Rescue membership
comes from the already-opened original H593 operator-evaluation result.
"""
import ast
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RESULT = ROOT/'results/rc_h593_quality_operator_eval_v1/result.json'
CACHE = ROOT/'results/rc_h593_simple_explanations_v1/cache.json'
FEATURE_SOURCE = ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py'
OPERATOR_SOURCE = ROOT/'programs/rc_quality_content_operator_v1.py'
EPS = 1e-12


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(binding):
    assert bind(binding['path']) == binding, binding['path']
    return Path(binding['path'])


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sym(a, b):
    return (a-b)/(abs(a)+abs(b)+EPS)


def feature_names():
    tree = ast.parse(FEATURE_SOURCE.read_text())
    return next(list(ast.literal_eval(n.value)) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FEATURE_NAMES' for t in n.targets))


def recompute():
    old, cache = read(RESULT), read(CACHE)
    assert old['total'] == len(old['rows']) == len(cache['rows']) == 593
    cache_by_id = {r['query_id']: r for r in cache['rows']}
    folds = {}; predictions = {}; sources = [bind(RESULT), bind(CACHE), bind(FEATURE_SOURCE), bind(OPERATOR_SOURCE), bind(__file__)]
    for fold in range(5):
        directory = ROOT/f'results/rc_h593_quality_operator_eval_v1/fold{fold}'
        validation = read(directory/'validation.json')
        assert validation['status'] == 'QUALITY_OPERATOR_EVAL_FRESH_NUMPY_PASS'
        payload = read(checked(validation['payload']))
        assert payload['fold'] == fold and payload['native_exact']
        folds[fold] = payload
        for p in payload['predictions']:
            assert p['query_id'] not in predictions
            predictions[p['query_id']] = (fold, p)
        sources += [bind(directory/'validation.json'), validation['payload']]
    names = feature_names(); assert len(names) == 6
    maximum = 0.; count = 0; rescues = []; rows = []
    formula_error = 0.; native_source_feature_error = 0.
    for row in old['rows']:
        q = row['query_id']; x = cache_by_id[q]; fold, sealed = predictions[q]
        assert fold == row['fold']
        par = folds[fold]['parameters']['COST1_FROZEN_NATIVE']
        assert par['method'] == 'FROZEN' and par['level'] == 'NATIVE'
        assert par['theta_hex'] == cache['native_parameters'][fold]['COST1']
        theta = list(map(float.fromhex, par['theta_hex']))
        native_X = x['native_X']; assert len(native_X) == 127
        logits = [sum(v*w for v,w in zip(f, theta[:-1]))+theta[-1] for f in native_X]
        sealed_logits = list(map(float.fromhex, sealed['models']['COST1_FROZEN_NATIVE']['logits_hex']))
        maximum = max(maximum, max(abs(a-b) for a,b in zip(logits,sealed_logits)))
        count += len(logits)
        chosen_k = max(range(127), key=lambda i: logits[i])
        chosen_position = x['challengers'][chosen_k] if logits[chosen_k] > 0 else x['winner']
        selected = x['axis'][chosen_position]
        assert selected == sealed['models']['COST1_FROZEN_NATIVE']['selected'] == row['selected']['COST1_FROZEN_NATIVE']
        rows.append(dict(query_id=q, fold=fold, original_native_correct=row['correct']['COST1_FROZEN_NATIVE'],
                         raw_correct=row['correct']['RAW'], selected_physical_row=selected,
                         original_127_logits_replayed=True))
        if not (row['correct']['COST1_FROZEN_NATIVE'] and not row['correct']['RAW']):
            continue
        # The opened result establishes that this selected identity is correct;
        # this is a descriptive post-outcome subset, never a selection for training.
        target = chosen_position; winner = x['winner']; k = x['challengers'].index(target)
        features = native_X[k]
        terms = [v*w for v,w in zip(features, theta[:-1])] + [theta[-1]]
        validation_binding = old['operator_validations'][row['execution_ordinal']]
        validation = read(checked(validation_binding))
        op = read(checked(validation['payload']))
        assert op['query_id'] == q and op['candidate_physical_rows'] == x['axis']
        native_source_feature_error = max(native_source_feature_error,
                                         max(abs(v-w) for v,w in zip(op['modes']['NATIVE']['X'][k], features)))
        scores = op['modes']['NATIVE']['scores']
        st, sw = scores[target], scores[winner]
        assert len(scores) == 128
        def scalars(s):
            return [s['real_score'], s['visibility_mass'],
                    s['real_score']/max(s['visibility_mass'], EPS),
                    s['real_score']-s['query_control_score'],
                    s['real_score']-s['reference_control_score']]
        target_values, winner_values = scalars(st), scalars(sw)
        rebuilt = [features[0]] + [sym(t,w) for t,w in zip(target_values,winner_values)]
        formula_error = max(formula_error, max(abs(v-w) for v,w in zip(rebuilt,features)))
        all_scores = [0.]*128
        for position, logit in zip(x['challengers'], logits):
            all_scores[position] = logit
        wrong = max((j for j in range(128) if j != target), key=lambda j: all_scores[j])
        target_logit = logits[k]
        margin = target_logit-all_scores[wrong]
        sealed_margin = row['operator_attribution']['COST1']['target_vs_strongest_wrong_margin']['NATIVE']
        assert abs(margin-sealed_margin) < 2e-10
        rescues.append(dict(query_id=q, original_query_id=row['original_query_id'], fold=fold,
            execution_ordinal=row['execution_ordinal'], identity=row['identity'],
            target_position=target, target_physical_row=x['axis'][target],
            raw_winner_position=winner, raw_winner_physical_row=x['axis'][winner],
            free_content_target=x['free_content'][target], free_content_raw_winner=x['free_content'][winner],
            free_content_difference=x['free_content'][target]-x['free_content'][winner],
            free_content_symmetric_difference=sym(x['free_content'][target], x['free_content'][winner]),
            M_target=x['mass'][target], M_raw_winner=x['mass'][winner],
            native_target_scores=st, native_raw_winner_scores=sw,
            original_fold_theta_hex=par['theta_hex'], original_fold_theta=theta,
            target_vs_RAW_features=dict(zip(names,features)),
            target_vs_HOLD_logit_terms=dict(zip(names+['bias'],terms)),
            term_sum=sum(terms), original_sealed_target_logit=sealed_logits[k],
            term_sum_absolute_error=abs(sum(terms)-sealed_logits[k]),
            strongest_wrong_physical_row=x['axis'][wrong], strongest_wrong_is_RAW_winner=wrong==winner,
            strongest_wrong_logit=all_scores[wrong], target_vs_strongest_wrong_margin=margin,
            original_operator_margin_shapley=row['operator_attribution']['COST1']['operator_shapley'],
            operator_source_validation=validation_binding, operator_source_payload=validation['payload']))
        sources += [validation_binding, validation['payload']]
    assert len(rescues) == 58 and maximum < 2e-10 and formula_error < 2e-10 and native_source_feature_error == 0
    all_terms = names+['bias']
    summary = dict(status='EXISTING_H593_ORIGINAL_COST1_ARITHMETIC_ATTRIBUTION_PASS',
        head='Original held-fold COST1 Linear(6,1), seven parameters; COST1_FROZEN_NATIVE, no refit',
        data='H593 original grouped five-fold OOF, natural ColNomic C128; opened development data',
        original_baseline=dict(raw_correct=426, native_correct=481, native_rescues=58, native_breaks=3,
                               denominator=593, candidate_recall=570),
        selected_subset='All58 original native-correct RAW-wrong queries; selected after original outcomes',
        source_files=sources, feature_order=names, term_order=all_terms,
        formula=dict(symmetric='sym(a,b)=(a-b)/(abs(a)+abs(b)+1e-12)',
            raw_gap='(RAW_g-RAW_w)/max(population_std(RAW_C128),1e-12); cached original feature retained',
            native_score='S=M*sum_i[u_i*max_j(sim_ij*v_j)]/max(sum_i u_i,1e-12)',
            mass='M=sqrt(mean(u)*mean(v))',
            normalized_local_similarity='S/max(M,1e-12); still contains query u and reference v weights; NOT free ColNomic content',
            query_control='S_qctrl uses u rolled by max(1,len(u)//2), keeping reference v and score definition',
            reference_control='S_rctrl uses v rolled by max(1,len(v)//2), keeping query u and score definition',
            query_robustness_feature='sym(S_g-S_qctrl_g,S_w-S_qctrl_w)',
            reference_robustness_feature='sym(S_g-S_rctrl_g,S_w-S_rctrl_w)',
            decision='z_g=sum_k(theta_k*X_gk)+bias; max among127 challengers switches iff >0; otherwise HOLD RAW winner'),
        subset_statistics=dict(count=len(rescues),
            positive_M_contrast=sum(r['target_vs_RAW_features'][names[2]]>0 for r in rescues),
            positive_native_normalized_local_contrast=sum(r['target_vs_RAW_features'][names[3]]>0 for r in rescues),
            negative_free_content_contrast=sum(r['free_content_difference']<0 for r in rescues),
            zero_free_content_contrast=sum(r['free_content_difference']==0 for r in rescues),
            mean_target_vs_HOLD_terms={name:sum(r['target_vs_HOLD_logit_terms'][name] for r in rescues)/len(rescues) for name in all_terms},
            mean_target_logit=sum(r['term_sum'] for r in rescues)/len(rescues),
            minimum_target_logit=min(r['term_sum'] for r in rescues),
            mean_original_operator_margin_shapley={name:sum(r['original_operator_margin_shapley'][name] for r in rescues)/len(rescues)
                for name in ('MASS','QUERY','REFERENCE')}),
        checks=dict(replayed_queries=len(rows), original_logits_recomputed=count,
                    maximum_original_logit_error=maximum,
                    maximum_rescue_term_sum_error=max(r['term_sum_absolute_error']for r in rescues),
                    maximum_rebuilt_feature_error=formula_error,
                    maximum_native_source_feature_error=native_source_feature_error,
                    original_actions_all_equal=True),
        limits=['57/58 is descriptive in an outcome-selected rescue subset, not a causal proportion.',
            'Feature contributions are exact logit arithmetic, not separate interventions or causal effects.',
            'Changing M also changes native score and both rolled-control contrast features; its causal path is not its single coefficient.',
            'Q/R robustness columns are score differences under rolled weights, not Q/R operator Shapley contributions or spatial ownership.',
            'No reference-main-effect claim for this original seven-parameter head is imported from simplified PRODUCT5.',
            'No claim that negative free-content contrast means all frozen ColNomic tokens lack identifying information.'])
    retention = {}
    for mode in ('M0Q0R0','M0Q0R1','M0Q1R0','M0Q1R1','M1Q0R0','M1Q0R1','M1Q1R0','FIXED_FREE_ARGMAX'):
        key = 'COST1_FROZEN_'+mode
        selected_ids = {r['query_id'] for r in rescues}
        retention[mode] = dict(original58_rescues_retained=sum(r['correct'][key] for r in old['rows'] if r['query_id'] in selected_ids),
            identity_changes_vs_native=sum(r['selected'][key]!=r['selected']['COST1_FROZEN_NATIVE'] for r in old['rows']),
            identity_changes_vs_RAW=sum(r['selected'][key]!=r['selected']['RAW'] for r in old['rows']))
    summary['original_frozen_operator_rescue_retention'] = retention
    write(OUT/'rescues58.json',rescues); write(OUT/'all593_replay.json',rows); write(OUT/'summary.json',summary)
    report = ['# 原H593七参数COST1：已有证据的独立算术归因', '',
        '自然ColNomic C128，原五折held参数；RAW426/593，原生481/593（58救回/3误伤）。不使用简化PRODUCT5头代替原头。', '',
        '原58例救回中：58例M相对RAW winner更高；58例原生质量归一化局部分数更高；57例自由ColNomic内容分数反而更低。',
        '这是按原结果事后选出的救回组的描述统计，57/58不是因果比例。', '',
        '| 原头输入项 | 对target相对HOLD logit的平均贡献 |', '|---|---:|']
    report += [f'| {name} | {summary["subset_statistics"]["mean_target_vs_HOLD_terms"][name]:+.9f} |' for name in all_terms]
    report += ['', '这些是算术贡献。原生S/M仍含u加权和reference v加权，不能称纯内容；Q/R robustness是滚动权重控制的评分差，不是独立算子贡献。', '',
        f'独立重算{count}个原始logit，最大误差{maximum:.3e}；全部593个动作与原封存预测一致。58例原生特征另从原算子S/M及控制分数重建，最大误差{formula_error:.3e}。', '',
        '逐例分差、7项贡献、完整参数与原始来源见rescues58.json；源文件SHA、公式和边界见summary.json；复核入口recompute.py。']
    (OUT/'report_zh.md').write_text('\n'.join(report)+'\n')
    write(OUT/'validation.json',dict(status=summary['status'],script=bind(__file__),sources=sources,
        artifacts=[bind(OUT/name)for name in ('summary.json','rescues58.json','all593_replay.json','report_zh.md')],checks=summary['checks']))
    print(json.dumps({'status':summary['status'],'checks':summary['checks'],'subset_statistics':summary['subset_statistics']}))


if __name__ == '__main__':
    recompute()
