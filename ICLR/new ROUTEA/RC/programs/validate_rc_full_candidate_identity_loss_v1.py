#!/usr/bin/env python3
"""Independently replay sealed all160 actions before joining any evaluation labels.

This collector does no fitting. Only the two already opened fixed panels are
scored; the other rows returned by the frozen feature loader are never scored.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P

TRAIN = ROOT / 'results/rc_convex_train_loss_cause_v1'
UNIT = TRAIN / 'unit_cost96'
PREV = ROOT / 'results/rc_fixed_panels_train269_group_risk_v1'
CAUSE = ROOT / 'results/rc_opened_convex_cause_readout_v1'
OUT = ROOT / 'results/rc_full_candidate_identity_loss_v1'
AUTH = ROOT / 'registry/rc_full_candidate_identity_loss_authority_v1_20260912.json'
REPORT = ROOT / 'reports/REPORT_FULL_CANDIDATE_IDENTITY_LOSS_V1_20260912.md'
MODELS = ('ORIGINAL7', 'TRAIN_UNIT_COST7', 'LISTWISE_UNIT1')
BASELINES = MODELS[:2]
HEX7 = ('0x1.01ffd7d7241b3p+0', '-0x1.e3db1bf7526f8p+1',
        '0x1.6ddd66d0df5d3p+2', '0x1.7a67f045c5bd2p+2',
        '-0x1.3095a38e39e50p-3', '-0x1.e8ea4dd00f544p-3',
        '-0x1.7a55fbee7993ap+0')
EC7 = 'ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'


def need(value, label):
    if not value:
        raise RuntimeError(label)


def path_of(binding):
    path = Path(binding['path'])
    return path if path.is_absolute() else ROOT / path


def checked(binding):
    path = path_of(binding)
    need(M.sha(path) == binding['sha256'], 'SOURCE_DRIFT:' + str(path))
    return path


def equal_binding(left, right):
    return (path_of(left).resolve() == path_of(right).resolve()
            and left['sha256'] == right['sha256'])


def indexed(rows, key='query_id'):
    result = {row[key]: row for row in rows}
    need(len(result) == len(rows), 'DUPLICATE_' + key)
    return result


def tensor_bits_equal(left, right):
    return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype and left.shape == right.shape
            and left.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            == right.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def action(logits, challengers, winner):
    need(len(logits) == len(challengers) and len(logits) > 0, 'ACTION_DIMENSIONS')
    best = max(range(len(logits)), key=lambda i: float(logits[i]))
    return challengers[best] if float(logits[best]) > 0 else winner


def moved_rank(raw, selected, target):
    order = [selected] + [row for row in raw if row != selected]
    need(len(order) == len(raw) == len(set(order)) and set(order) == set(raw),
         'MOVE_TO_FRONT_PERMUTATION')
    return order.index(target) + 1


def summarize(rows):
    scores = {}
    for name in rows[0]['correct']:
        correct = sum(row['correct'][name] for row in rows)
        mrr = sum((Fraction(1, row['ranks'][name]) for row in rows), Fraction()) / len(rows)
        scores[name] = dict(correct=correct, queries=len(rows), accuracy=correct / len(rows),
                            MRR=float(mrr), MRR_rational=str(mrr),
                            switch_count=sum(row['selected_physical_rows'][name]
                                             != row['selected_physical_rows']['RAW'] for row in rows))
    comparisons = {}
    pairs = [('TRAIN_UNIT_COST7', 'LISTWISE_UNIT1'), ('ORIGINAL7', 'LISTWISE_UNIT1')]
    pairs += [('RAW', name) for name in MODELS]
    pairs += [(base + '_CBIND', 'LISTWISE_UNIT1_CBIND') for base in BASELINES]
    pairs += [(name + '_CBIND', name) for name in MODELS]
    for base, new in pairs:
        rescue = [r['query_id'] for r in rows if not r['correct'][base] and r['correct'][new]]
        loss = [r['query_id'] for r in rows if r['correct'][base] and not r['correct'][new]]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['component']].append(int(row['correct'][new]) - int(row['correct'][base]))
        means = {key: Fraction(sum(values), len(values)) for key, values in grouped.items()}
        comparisons[base + '__to__' + new] = dict(
            rescue=len(rescue), loss=len(loss), net=len(rescue) - len(loss),
            rescue_query_ids=rescue, loss_query_ids=loss,
            positive_net=len(rescue) > len(loss),
            equal_component_difference_rational=str(sum(means.values(), Fraction()) / len(means)),
            components=len(means), positive_components=sum(v > 0 for v in means.values()),
            negative_components=sum(v < 0 for v in means.values()))
    binding = {}
    for name in MODELS:
        rescued = [r for r in rows if not r['correct']['RAW'] and r['correct'][name]]
        retained = sum(r['correct'][name + '_CBIND'] for r in rescued)
        binding[name] = dict(REAL_rescues_over_RAW=len(rescued), CBIND_retained_REAL_rescues=retained,
                             CBIND_removed_REAL_rescues=len(rescued) - retained,
                             REAL_minus_CBIND_correct=scores[name]['correct'] - scores[name + '_CBIND']['correct'])
    return dict(population=len(rows), recall_C128=sum(r['target_in_C128'] for r in rows),
                scores=scores, comparisons=comparisons, C_BIND_rescue_retention=binding, rows=rows)


def write_report(result):
    lines = ['# Full candidate identity loss：固定开发面板比较', '',
             '证据级别：已打开的 EVAL32／EVAL128 开发结果；未产生新 593 准确率、外部确认或 HYP GO。', '',
             '候选来源为冻结 RAW full-gallery C128。三个头均为相同 6 个特征加 bias；'
             'REAL／CBIND 均运行完整 127 challenger HOLD/SWITCH，最大 logit ≤ 0 时 HOLD，正值并列取首个 challenger。', '',
             '仅 LISTWISE_UNIT1 新拟合：原 mixed96（PAIR64 + FULL32），原顺序、FP64、零初始化、'
             'seed 17、2000 AdamW 更新。PAIR 单位成本保持不变；FULL 改为 logsumexp([RAW=0, 127 logits]) − target logit。'
             'ORIGINAL7 与 TRAIN_UNIT_COST7 的旧参数和预测逐 bit 保留。', '',
             '| 面板 | 头／路径 | 正确数 | Accuracy | MRR |', '|---|---|---:|---:|---:|']
    for panel in ('EVAL32', 'EVAL128'):
        for name, score in result['panels'][panel]['scores'].items():
            lines.append(f"| {panel} | {name} | {score['correct']}/{score['queries']} | "
                         f"{score['accuracy']:.6f} | {score['MRR']:.9f} |")
    lines += ['', '| 面板 | REAL 比较 | Rescue | Loss | Net |', '|---|---|---:|---:|---:|']
    for panel in ('EVAL32', 'EVAL128'):
        for base in ('TRAIN_UNIT_COST7', 'ORIGINAL7'):
            c = result['panels'][panel]['comparisons'][base + '__to__LISTWISE_UNIT1']
            lines.append(f"| {panel} | {base} → LISTWISE_UNIT1 | {c['rescue']} | {c['loss']} | {c['net']:+d} |")
    lines += ['', 'TRAIN_UNIT_COST7 → LISTWISE_UNIT1 是 FULL loss 的单因素比较；'
              'ORIGINAL7 → LISTWISE_UNIT1 是相对原强基线的比较，同时包含成本与 FULL loss 差异。'
              '逐面板报告净增，不将零损失设置为成功前提。', '',
              'MRR 由完整 gallery 实际 move-to-front 后的秩计算，并保存精确 Fraction。'
              '独立 NumPy 重放了 121920 个 logits，随后才打开评估标签及历史结果；'
              '训练与两面板的 query、图像 SHA、identity、group、component 均无交集。', '',
              f"结果：`{(OUT / 'result.json').relative_to(ROOT)}`。", '',
              f"独立验证：`{(OUT / 'independent_validation.json').relative_to(ROOT)}`。", '']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main():
    need(bool(os.environ.get('SLURM_JOB_ID')), 'SLURM_REQUIRED')
    need(not (OUT / 'result.json').exists() and not (OUT / 'independent_validation.json').exists()
         and not REPORT.exists(), 'IMMUTABLE_NEW_READOUT')
    authority = M.read(AUTH)
    ab = M.bind(AUTH)
    expected_eval = dict(previous_result=PREV / 'result.json',
                         previous_result_validation=PREV / 'result_validation.json',
                         eval_labels=PREV / 'eval_curator_roles.json',
                         panel_manifest=PREV / 'panel_manifest.json',
                         previous_cause_result=CAUSE / 'result.json',
                         previous_cause_validation=CAUSE / 'independent_validation.json')
    need(set(authority['evaluation_sources']) == set(expected_eval), 'EXACT_EVALUATION_SOURCE_KEYS')
    for key, path in expected_eval.items():
        need(path_of(authority['evaluation_sources'][key]).resolve() == path.resolve(), 'EVAL_PATH_' + key)
    labels_open = False
    evaluation_paths = {path.resolve() for path in expected_eval.values()}

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        text = str(path).lower()
        need(not any(token in text for token in ('d1-mi', 'd1_mi', 'grozi',
                  'gisc_prerecall_universe', '/target_join/', 'formal_private')), 'PROTECTED_INPUT')
        need(labels_open or (path not in evaluation_paths and 'curator_roles' not in text
                             and 'rc_opened_' not in text and '/reports/' not in text),
             'PREDICTION_SEAL_AND_REPLAY_BEFORE_LABELS')

    sys.addaudithook(audit)
    for binding in authority['sources'].values():
        checked(binding)
    need(any(equal_binding(binding, M.bind(__file__)) for binding in authority['sources'].values()),
         'INDEPENDENT_VALIDATOR_AUTHORITY_PIN')
    fit = M.read(OUT / 'fit.json')
    fitval = M.read(OUT / 'fit_validation.json')
    need(fit['status'] == 'LISTWISE_UNIT1_MATCHED_TRAINING_COMPLETE'
         and fit['train_count'] == dict(PAIR=64, FULL=32, total=96), 'NEW_FIT_SCOPE')
    need(fitval['status'] == 'LISTWISE_UNIT1_FRESH_PARAMETER_REPLAY_PASS', 'FRESH_PARAMETER_REPLAY')
    need(equal_binding(fitval['fit'], M.bind(OUT / 'fit.json')), 'FIT_VALIDATION_HASH')
    need(isinstance(fitval.get('fresh_nonce'), str) and len(fitval['fresh_nonce']) == 32, 'FRESH_REPLAY_NONCE')
    for value in (fit, fitval):
        need(equal_binding(value['authority'], ab)
             and equal_binding(value['input_seal'], M.bind(TRAIN / 'input_seal.json')), 'FIT_SOURCE_BINDINGS')
        need(value['original_head_training_updates'] == value['baseline_head_training_updates'] == 0
             and value['new_head_training_updates'] == 2000 and value['heldout_label_reads'] == 0,
             'ONLY_ONE_NEW_HEAD2000_UPDATES')
    need(fit['finite_training']['all_finite'] is True, 'FINITE_LISTWISE_TRAINING')
    need(fit['optimizer'] == dict(name='AdamW', lr=.03, weight_decay=.001, steps=2000,
                                seed=17, initialization='zero', dtype='float64'), 'UNCHANGED_OPTIMIZER')
    preflight = M.read(OUT / 'preflight.json')
    need(preflight['status'] == 'LISTWISE_UNIT1_OBJECTIVE_PREFLIGHT_PASS'
         and equal_binding(preflight['authority'], ab)
         and equal_binding(preflight['input_seal'], fit['input_seal'])
         and preflight['function_change'] == fit['function_change'], 'QUALIFIED_FROZEN_OBJECTIVE')
    change = fit['function_change']
    need(change['replacement_count'] == 1 and change['changed_component'] == 'FULL_LOSS_ONLY'
         and change['PAIR_loss'] == 'cost1_BCE_mean'
         and change['FULL_loss'] == 'mean_logsumexp_RAW0_plus_all_challengers_minus_target'
         and change['PAIR_FULL_mixture'] == [1, 1] and change['temperature'] == 1
         and change['new_features'] == 0 and change['parameter_count'] == 7, 'ONLY_FULL_LOSS_CHANGE')
    inputseal = M.read(TRAIN / 'input_seal.json')
    need(inputseal['status'] == 'TRAIN_INPUTS_SEALED_BEFORE_EVAL_CONES'
         and inputseal['evaluation_labels_included'] is False, 'ORIGINAL_TRAIN_ONLY_SEAL')
    need(inputseal['counts']['PAIR'] == 64 and inputseal['counts']['FULL'] == 32
         and inputseal['counts']['unique_images'] == 96 and inputseal['counts']['parameters'] == 7,
         'EXACT_ORIGINAL_MIXED96')
    originalseal = M.read(checked(inputseal['sources']['original_train_input_seal']))
    need(equal_binding(originalseal['payload'], inputseal['sources']['original_train_pack']), 'PACK_SEAL_CHAIN')
    pack = torch.load(checked(originalseal['payload']), weights_only=True, map_location='cpu')
    trainrole = M.read(checked(originalseal['train_roles']))
    need(trainrole['evaluation_labels_included'] is False, 'TRAIN_ROLE_ONLY')
    train = indexed(trainrole['records'])
    need(len(train) == 96 and len({r['identity'] for r in train.values()}) == 32, 'TRAIN96_IDENTITIES32')
    pair, full = pack['pair']['records'], pack['full']
    need(len(pair) == 64 and len(full) == 32 and len({r['query_id'] for r in pair + full}) == 96,
         'ORIGINAL_PACK96')
    need([r['execution_ordinal'] for r in pair] == [r[2] for r in originalseal['original_pair_order']]
         and [r['execution_ordinal'] for r in full] == originalseal['original_training_order'], 'EXACT_TRAIN_ORDER')
    trainold = indexed(train.values(), 'original_query_id')
    need(set(trainold) == {r['query_id'] for r in pair + full}, 'PACK_TRAIN_ROLE_QUERY_JOIN')
    for row in pair + full:
        need(trainold[row['query_id']]['legacy_execution_ordinal'] == row['execution_ordinal'], 'TRAIN_LEGACY_ORDINAL')

    seal = M.read(OUT / 'prediction_seal.json')
    need(seal['status'] == 'LISTWISE_UNIT1_ALL160_PREDICTIONS_SEALED', 'ALL160_SEALED_BEFORE_JOIN')
    need(path_of(seal['payload']).resolve() == (OUT / 'predictions.pt').resolve(), 'EXACT_NEW_PAYLOAD')
    payload = torch.load(checked(seal['payload']), weights_only=True, map_location='cpu')
    for value in (seal, payload):
        for key, path in dict(fit=OUT / 'fit.json', fit_validation=OUT / 'fit_validation.json',
                              authority=AUTH, input_seal=TRAIN / 'input_seal.json').items():
            need(equal_binding(value[key], M.bind(path)), 'PREDICTION_BINDING_' + key)
        need(value['original_head_training_updates'] == value['baseline_head_training_updates']
             == value['heldout_label_reads'] == 0, 'NO_BASELINE_RETRAIN_OR_LABEL_READ')
    need(equal_binding(payload['previous_payload'], M.bind(UNIT / 'predictions.pt')), 'EXACT_BASELINE_PAYLOAD')
    oldseal = M.read(UNIT / 'prediction_seal.json')
    oldfit = M.read(UNIT / 'fit.json')
    oldfitval = M.read(UNIT / 'fit_validation.json')
    need(oldseal['status'] == 'UNIT_COST96_TRAIN_ONLY_ALL160_PREDICTIONS_SEALED'
         and oldfitval['status'] == 'UNIT_COST96_FRESH_PARAMETER_REPLAY_PASS', 'BASELINE_PREJOIN_VALIDATION')
    for key, path in dict(unit_cost_fit=UNIT / 'fit.json',
                          unit_cost_fit_validation=UNIT / 'fit_validation.json', payload=UNIT / 'predictions.pt').items():
        need(equal_binding(oldseal[key], M.bind(path)), 'BASELINE_SEAL_' + key)
    need(equal_binding(oldfitval['fit'], M.bind(UNIT / 'fit.json'))
         and equal_binding(oldfit['authority'], oldfitval['authority'])
         and equal_binding(oldfit['authority'], oldseal['authority'])
         and equal_binding(oldfit['input_seal'], fit['input_seal'])
         and equal_binding(oldfitval['input_seal'], fit['input_seal']), 'BASELINE_TRAIN_SOURCE_CHAIN')
    need(oldfit['optimizer'] == fit['optimizer']
         and oldfit['train_count'] == fit['train_count']
         and change['unit_cost_function_sha256'] == oldfit['function_change']['changed_function_sha256'],
         'SINGLE_FACTOR_UNIT_COST_TRAINING_LINEAGE')
    old = torch.load(checked(oldseal['payload']), weights_only=True, map_location='cpu')
    need(set(payload['parameters']) == set(MODELS) and set(old['parameters']) == set(BASELINES), 'EXACT_MODEL_SETS')
    theta = {}
    for model in MODELS:
        parameters = payload['parameters'][model]
        need(set(parameters) == {'theta'}, 'SEVEN_PARAMETER_ARCHITECTURE')
        value = parameters['theta']
        need(isinstance(value, torch.Tensor) and value.dtype == torch.float64 and tuple(value.shape) == (7,)
             and bool(torch.isfinite(value).all()), 'FINITE_BINARY64_THETA7')
        theta[model] = value.numpy()
        if model in BASELINES:
            need(tensor_bits_equal(value, old['parameters'][model]['theta']), 'BASELINE_PARAMETER_BITS_' + model)
    need(tuple(float(v).hex() for v in theta['ORIGINAL7']) == HEX7, 'ORIGINAL_EC7_BITS')
    need(tuple(float(v).hex() for v in theta['TRAIN_UNIT_COST7']) == tuple(oldfit['theta_binary64']), 'UNIT_COST_FIT_BITS')
    need(tuple(float(v).hex() for v in theta['LISTWISE_UNIT1']) == tuple(fit['theta_binary64']), 'LISTWISE_FIT_BITS')
    features, feature_sources = P.features()
    need(payload['feature_sources'] == old['feature_sources'] == feature_sources, 'FROZEN_FEATURE_SOURCES')
    fi = indexed(features)
    predictions, oldpred = indexed(payload['predictions']), indexed(old['predictions'])
    need(len(predictions) == len(oldpred) == 160 and set(predictions) == set(oldpred), 'EXACT_OLD160')
    chosen, maximum_error, logit_count = {}, 0.0, 0
    for q, prediction in predictions.items():
        feature, prior = fi[q], oldpred[q]
        for key in ('query_id', 'execution_ordinal', 'candidate_physical_rows', 'raw_ranked_physical_rows',
                    'winner', 'challenger_positions'):
            need(prediction[key] == feature[key] == prior[key], 'FROZEN_AXES_' + key)
        axis, raw = feature['candidate_physical_rows'], feature['raw_ranked_physical_rows']
        challengers = feature['challenger_positions']
        need(len(axis) == 128 and axis == sorted(set(axis)) and set(axis) == set(raw[:128]), 'NATURAL_RAW_C128')
        need(len(raw) == len(set(raw)) and axis[feature['winner']] == raw[0], 'FULL_GALLERY_RAW_WINNER')
        need(challengers == [j for j in range(128) if j != feature['winner']], 'ALL127_CHALLENGERS')
        need(set(prediction['models']) == set(MODELS), 'THREE_MODELS_EACH_QUERY')
        selected = {'RAW': raw[0]}
        for model in MODELS:
            need(set(prediction['models'][model]) == {'REAL', 'CBIND'}, 'REAL_CBIND_MODES')
            for mode in ('REAL', 'CBIND'):
                report = prediction['models'][model][mode]
                x, observed = feature['modes'][mode]['X'], report['logits']
                need(x.dtype == torch.float64 and tuple(x.shape) == (127, 6), 'FEATURE127X6')
                need(observed.dtype == torch.float64 and tuple(observed.shape) == (127,), 'LOGITS127')
                z = np.asarray(x.numpy(), dtype=np.float64) @ theta[model][:6] + theta[model][6]
                need(np.isfinite(z).all() and np.isfinite(observed.numpy()).all(), 'FINITE_LOGITS')
                error = float(np.max(np.abs(z - observed.numpy())))
                maximum_error = max(maximum_error, error)
                logit_count += len(z)
                need(error <= 1e-12, 'INDEPENDENT_NUMPY_LOGIT_REPLAY')
                position = action(z, challengers, feature['winner'])
                need(position == report['selected_position'], 'PYTHON_HOLD_SWITCH_REPLAY')
                if model in BASELINES:
                    oldreport = prior['models'][model][mode]
                    need(position == oldreport['selected_position']
                         and tensor_bits_equal(observed, oldreport['logits']), 'BASELINE_PREDICTION_BITS')
                selected[model if mode == 'REAL' else model + '_CBIND'] = axis[position]
        chosen[q] = selected
    need(logit_count == 121920, 'ALL121920_LOGITS_REPLAYED')

    # All evaluation targets and historic outcomes remain unread until this point.
    labels_open = True
    for binding in authority['evaluation_sources'].values():
        checked(binding)
    previous = M.read(expected_eval['previous_result'])
    previousval = M.read(expected_eval['previous_result_validation'])
    cause = M.read(expected_eval['previous_cause_result'])
    causeval = M.read(expected_eval['previous_cause_validation'])
    need(previousval['status'] == 'FIXED_PANELS_INDEPENDENT_ACTION_RANK_COUNTS_PASS'
         and equal_binding(previousval['result'], authority['evaluation_sources']['previous_result']), 'PREVIOUS_RESULT_VALIDATED')
    need(previousval['original7_parameter_sha256'] == EC7 and tuple(previousval['original7_binary64']) == HEX7,
         'PREVIOUS_ORIGINAL7_IDENTITY')
    need(causeval['status'] == 'CONVEX_CAUSE_INDEPENDENT_ACTION_RANK_AND_BOUND_COMPARISON_PASS'
         and equal_binding(causeval['result'], authority['evaluation_sources']['previous_cause_result'])
         and equal_binding(causeval['unit_cost_payload'], oldseal['payload'])
         and equal_binding(causeval['unit_cost_prediction_seal'], M.bind(UNIT / 'prediction_seal.json')),
         'PREVIOUS_UNIT_COST_OUTCOME_VALIDATED')
    split = M.read(expected_eval['panel_manifest'])
    roles = indexed(M.read(expected_eval['eval_labels'])['records'])
    need(set(split['panels']) == {'EVAL32', 'EVAL128'} and split['selection_used_outcomes'] is False, 'FIXED_PANEL_SELECTION')
    ids = [q for panel in ('EVAL32', 'EVAL128') for q in split['panels'][panel]]
    need(len(ids) == len(set(ids)) == 160 and set(ids) == set(predictions) == set(roles), 'EXACT_EVAL160')
    overlaps = {}
    for panel, qs in split['panels'].items():
        need(len(qs) == {'EVAL32': 32, 'EVAL128': 128}[panel], 'PANEL_SIZE')
        overlaps[panel] = {}
        for key in ('query_id', 'original_query_id', 'source_image_sha256', 'identity', 'group', 'component'):
            overlaps[panel][key] = len({r[key] for r in train.values()} & {roles[q][key] for q in qs})
            need(overlaps[panel][key] == 0, 'TRAIN_EVAL_DISJOINT_' + panel + '_' + key)
    for key in ('query_id', 'source_image_sha256', 'identity', 'group', 'component'):
        need(not ({roles[q][key] for q in split['panels']['EVAL32']}
                  & {roles[q][key] for q in split['panels']['EVAL128']}), 'EVAL_PANELS_DISJOINT_' + key)
    from rc_aslo_xf.conditional_rep_sources import build_gallery_source
    labels = build_gallery_source(verify_cache_file_sha256=True).corrected_identities
    panels = {}
    for panel, qs in split['panels'].items():
        previous_rows = indexed(previous['panels'][panel]['rows'])
        cause_rows = indexed(cause['panels'][panel]['rows'])
        need(set(previous_rows) == set(cause_rows) == set(qs), 'HISTORICAL_PANEL_QUERY_PARITY')
        rows = []
        for q in qs:
            f, role, selected = fi[q], roles[q], chosen[q]
            need(f['execution_ordinal'] == role['execution_ordinal']
                 and f['source_image_sha256'] == role['source_image_sha256'], 'FEATURE_ROLE_IMAGE_BINDING')
            raw = f['raw_ranked_physical_rows']
            targets = [physical for physical in raw if labels[physical] == role['identity']]
            need(len(targets) == 1, 'UNIQUE_FULL_GALLERY_TARGET')
            target = targets[0]
            row = dict(query_id=q, original_query_id=role['original_query_id'], group=role['group'],
                       component=role['component'], target_in_C128=target in f['candidate_physical_rows'],
                       selected_physical_rows=selected,
                       correct={name: physical == target for name, physical in selected.items()},
                       ranks={name: moved_rank(raw, physical, target) for name, physical in selected.items()})
            for historic, names in ((previous_rows[q], ('RAW', 'ORIGINAL7', 'ORIGINAL7_CBIND')),
                                    (cause_rows[q], ('RAW', 'ORIGINAL7', 'ORIGINAL7_CBIND',
                                                     'TRAIN_UNIT_COST7', 'TRAIN_UNIT_COST7_CBIND'))):
                for key in ('original_query_id', 'group', 'component', 'target_in_C128'):
                    need(row[key] == historic[key], 'HISTORICAL_ROW_LINEAGE_' + key)
                for key in ('correct', 'ranks', 'selected_physical_rows'):
                    need(all(row[key][name] == historic[key][name] for name in names), 'BASELINE_PER_QUERY_OUTCOME_PARITY')
            rows.append(row)
        panels[panel] = summarize(rows)
        for name in ('RAW', 'ORIGINAL7', 'ORIGINAL7_CBIND', 'TRAIN_UNIT_COST7', 'TRAIN_UNIT_COST7_CBIND'):
            need(panels[panel]['scores'][name] == cause['panels'][panel]['scores'][name], 'BASELINE_EXACT_SCORE_PARITY')
    result = dict(status='LISTWISE_UNIT1_FIXED_OPENED_PANELS_DEVELOPMENT_READOUT_COMPLETE',
                  authority=ab, program=M.bind(__file__), fit=M.bind(OUT / 'fit.json'),
                  fit_validation=M.bind(OUT / 'fit_validation.json'), input_seal=fit['input_seal'],
                  prediction_seal=M.bind(OUT / 'prediction_seal.json'), payload=seal['payload'],
                  previous_payload=payload['previous_payload'], evaluation_sources=authority['evaluation_sources'],
                  head_class='Frozen native six features plus one bias; 7 FP64 parameters',
                  candidate_source='Frozen natural RAW full-gallery C128, all 127 challengers',
                  training_population=dict(PAIR=64, FULL=32, unique_images=96, identities=32),
                  function_change=fit['function_change'], optimizer=fit['optimizer'],
                  original_head_training_updates=0, baseline_head_training_updates=0,
                  new_head_training_updates=2000, collector_optimizer_calls=0,
                  baseline_parameters_and_predictions_copied_bit_exact=True,
                  baseline_outcomes_match_previous_validated_results=True,
                  original7_parameter_sha256=EC7, original7_binary64=list(HEX7),
                  independently_checked_logits=logit_count, maximum_absolute_numpy_logit_error=maximum_error,
                  train_eval_overlaps=overlaps, panels=panels,
                  primary_comparison='TRAIN_UNIT_COST7__to__LISTWISE_UNIT1',
                  strong_baseline_comparison='ORIGINAL7__to__LISTWISE_UNIT1',
                  success_criterion='Report rescue, loss and net separately on each fixed panel; no zero-loss gate',
                  primary_positive_net_by_panel={p: panels[p]['comparisons']['TRAIN_UNIT_COST7__to__LISTWISE_UNIT1']['net'] > 0 for p in panels},
                  strong_baseline_positive_net_by_panel={p: panels[p]['comparisons']['ORIGINAL7__to__LISTWISE_UNIT1']['net'] > 0 for p in panels},
                  evaluation_labels_opened_after_prediction_seal_and_independent_replay=True,
                  historically_opened_development=True, external_confirmation=False,
                  new_593_accuracy_computed=False, HYP_GO_claimed=False, deployment_changed=False,
                  slurm_job_id=os.environ['SLURM_JOB_ID'])
    M.write(OUT / 'result.json', result)
    validation = dict(status='LISTWISE_UNIT1_INDEPENDENT_ACTION_RANK_COMPARISON_PASS',
                      result=M.bind(OUT / 'result.json'), authority=ab, program=M.bind(__file__),
                      fit_validation=result['fit_validation'], prediction_seal=result['prediction_seal'],
                      payload=seal['payload'], previous_payload=payload['previous_payload'],
                      independently_checked_logits=logit_count, maximum_absolute_logit_error=maximum_error,
                      logit_tolerance=1e-12, optimizer_calls=0, baseline_parameter_and_prediction_bits_equal=True,
                      baseline_outcome_parity_on_both_panels=True, train_eval_overlaps=overlaps,
                      labels_opened_after_seal_and_independent_prediction_replay=True,
                      MRR_method='Literal full-gallery move-to-front followed by exact Fraction reciprocal ranks',
                      action_rule='Earliest challenger on equal maxima; HOLD when maximum <= 0',
                      new_593_accuracy_computed=False, slurm_job_id=os.environ['SLURM_JOB_ID'])
    M.write(OUT / 'independent_validation.json', validation)
    write_report(result)
    print(json.dumps(dict(status=validation['status'], result=M.bind(OUT / 'result.json'),
                          independently_checked_logits=logit_count)), flush=True)


def synthetic_test():
    need(action([0., 0.], [2, 3], 1) == 1, 'HOLD_ZERO')
    need(action([-1., -2.], [2, 3], 1) == 1, 'HOLD_NEGATIVE')
    need(action([1., 1.], [2, 3], 1) == 2, 'FIRST_TIE')
    need(moved_rank([4, 2, 9], 9, 2) == 3 and moved_rank([4, 2, 9], 2, 2) == 1, 'LITERAL_FULL_RANK')
    need(not tensor_bits_equal(torch.tensor([0.], dtype=torch.float64),
                               torch.tensor([-0.], dtype=torch.float64)), 'BITWISE_NOT_NUMERIC_PARITY')
    rows = []
    for i, (unit, new) in enumerate(((False, True), (True, False), (False, True))):
        correct = dict(RAW=False, ORIGINAL7=unit, TRAIN_UNIT_COST7=unit, LISTWISE_UNIT1=new)
        correct.update({name + '_CBIND': False for name in MODELS})
        rows.append(dict(query_id=str(i), component=str(i), target_in_C128=True, correct=correct,
                         ranks={name: 1 if yes else 2 for name, yes in correct.items()},
                         selected_physical_rows={name: int(yes) for name, yes in correct.items()}))
    result = summarize(rows)
    cmp = result['comparisons']['TRAIN_UNIT_COST7__to__LISTWISE_UNIT1']
    need((cmp['rescue'], cmp['loss'], cmp['net'], cmp['positive_net']) == (2, 1, 1, True), 'NET_WITH_NONZERO_LOSS')
    need(result['scores']['LISTWISE_UNIT1']['MRR_rational'] == '5/6', 'FRACTION_MRR')
    print('LISTWISE_UNIT1_INDEPENDENT_SYNTHETIC_ACTION_RANK_SUMMARY_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-test', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if args.synthetic_test:
        synthetic_test()
    else:
        synthetic_test()
        main()
