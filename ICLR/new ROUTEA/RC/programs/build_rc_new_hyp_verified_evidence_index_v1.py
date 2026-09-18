#!/usr/bin/env python3
"""Nine explicitly mapped validated trials: provenance index, no predictions or fitting."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_new_hyp_verified_evidence_index_v1'
SPECS = [{'trial': 'rc_absolute_evidence_scale_calibration_v1',
  'layout': 'standard',
  'heads': ['NATIVE7', 'ABS12', 'REL12'],
  'trial_baseline': 'NATIVE7',
  'baseline_replays': ['NATIVE7'],
  'title': '绝对证据尺度扩展',
  'result_status': 'ABSOLUTE_SCALE_NO_INTERNAL_NET_GAIN',
  'validation_status': 'ABSOLUTE_SCALE_INDEPENDENT_REEXECUTION_VALIDATION_PASS',
  'pins': {'result.json': '595f1ca3b1b6151af1303c1f6d68cc350af37ef823c204b3a339d55e84e29f6c',
           'independent_validation.json': '6cb51164d4e0ec5adc4ba9e2e26ab14ced61c558170ae19a7230a8005d72c0b9',
           'parameters.json': '1b163130c59d7eec64ab67a48dfa3bf369525b79159e03519a632b032eade490'},
  'parameter_file': 'parameters.json'},
 {'trial': 'rc_native7_training_objective_factorial_v1',
  'layout': 'standard',
  'heads': ['PAIR_SIGN', 'PAIR_RANK', 'FULL_SIGN', 'FULL_RANK'],
  'trial_baseline': 'PAIR_SIGN',
  'baseline_replays': ['PAIR_SIGN'],
  'title': '固定训练目标对照',
  'result_status': 'NATIVE7_OBJECTIVE_PRIMARY_NO_NET_GAIN',
  'validation_status': 'NATIVE7_OBJECTIVE_FACTORIAL_INDEPENDENT_REEXECUTION_PASS',
  'pins': {'result.json': '51d4e9a80d18d73c7f37730ff1c840186c279c3e59ed5b54a0d6a029522e9c17',
           'independent_validation.json': '932a9aa825950607b75c1dd72267fbd570b8f96102ce5a9595bfcb0ad84c0e07',
           'parameters.json': '3405a316d456df62d2a0276808f5d746124b91ba1fdc581bb9b23acbb9daa316'},
  'parameter_file': 'parameters.json'},
 {'trial': 'rc_pair_control_scale_harmonization_v1',
  'layout': 'standard',
  'heads': ['ORIGINAL7', 'HARMONIZED7'],
  'trial_baseline': 'ORIGINAL7',
  'baseline_replays': ['ORIGINAL7'],
  'title': 'PAIR控制尺度统一',
  'result_status': 'PAIR_CONTROL_DEFINITION_NO_INTERNAL_NET_GAIN',
  'validation_status': 'PAIR_CONTROL_DEFINITION_INDEPENDENT_REEXECUTION_PASS',
  'pins': {'result.json': 'ea7244d674c9a2e6f60a1771685fba812b9b118505c478636213c2600063af5a',
           'independent_validation.json': '198f10cc5dbec9c133c532260f63c489015e6132d48e6eb16513baec4e3efd43',
           'parameters.json': 'c3a55c7c869a7c3a60dc718f8b567800526a5248aaeac355e8926562b98d76a8'},
  'parameter_file': 'parameters.json'},
 {'trial': 'rc_full_reference_content_bridge_v1',
  'layout': 'standard',
  'heads': ['ORIGINAL_C', 'D_IMAGE', 'D_FULL'],
  'trial_baseline': 'ORIGINAL_C',
  'baseline_replays': ['ORIGINAL_C', 'D_IMAGE'],
  'title': '完整reference内容桥接',
  'result_status': 'FULL_REFERENCE_BRIDGE_NO_INTERNAL_GAIN_OVER_C',
  'validation_status': 'FULL_REFERENCE_CONTENT_BRIDGE_INDEPENDENT_REEXECUTION_PASS',
  'pins': {'result.json': '45ed29c7c3d1d6288fa18e588eeef31df7365bd54c1f23846680bf86351b0ecf',
           'independent_validation.json': '2add69653a5cc9d5c87c3f162efd12b44302406e4095a8a99b4cab7b4ffdaa46',
           'parameters.json': '8d366d4358b24c124c13ebd29f06c5982f0126827975ee162c83e68610cca575'},
  'parameter_file': 'parameters.json'},
 {'trial': 'rc_retrained_evidence_sufficiency_v1',
  'layout': 'standard',
  'heads': ['RAW2', 'RAW_PLUS_M3', 'RAW_PLUS_L3', 'JOINT4', 'ORIGINAL7'],
  'trial_baseline': 'ORIGINAL7',
  'baseline_replays': ['ORIGINAL7'],
  'title': '重新训练证据子集',
  'result_status': 'RETRAINED_JOINT_M_L_INTERNAL_SIMPLIFICATION_NOT_ESTABLISHED',
  'validation_status': 'RETRAINED_EVIDENCE_SUFFICIENCY_INDEPENDENT_REEXECUTION_PASS',
  'pins': {'result.json': '20102901242c734c2b5c0404e0e5571999ab1dc25ceefa5cee2f37f884a12601',
           'independent_validation.json': 'ebac4a7d704e121d972f8999dbcbe767e588681e3f9a7352974f78d9e809e195',
           'parameters.json': '8d1781e0abaec6bcd2cdacb6f0637140398764ca1f533452f9bb1cf956aee20b'},
  'parameter_file': 'parameters.json'},
 {'trial': 'rc_product_response_factorial_v1',
  'layout': 'standard',
  'heads': ['JOINT4', 'PRODUCT5', 'RESPONSE6', 'ORIGINAL7'],
  'trial_baseline': 'ORIGINAL7',
  'baseline_replays': ['JOINT4', 'ORIGINAL7'],
  'title': 'S与Q/R固定2×2对照',
  'result_status': 'PRODUCT_RESPONSE_FACTORIAL_INTERNAL_SIMPLIFICATION_NOT_ESTABLISHED',
  'validation_status': 'PRODUCT_RESPONSE_FACTORIAL_INDEPENDENT_REEXECUTION_PASS',
  'pins': {'result.json': 'e9dae1e5837e72d3f38ba3e4473d0ff555b6ee90fadfbe3d3dbf64376759c0bf',
           'independent_validation.json': 'f91c01caac1824ceb8cde358a9c22f95b94fe0d846965b35815c66cc778ed3bb',
           'parameters.json': '019cfeac14575b97371f66f81e5ad6ec55ec321af502a0b9110e4cc207c79b43'},
  'parameter_file': 'parameters.json'},
 {'trial': 'rc_frozen_group_effect_native64_v1',
  'layout': 'native_fixed',
  'heads': ['ORIGINAL7', 'DROP_S', 'DROP_QR', 'DROP_S_QR', 'REFIT_RESPONSE6', 'REFIT_PRODUCT5', 'REFIT_JOINT4'],
  'trial_baseline': 'ORIGINAL7',
  'baseline_replays': ['ORIGINAL7', 'REFIT_RESPONSE6', 'REFIT_PRODUCT5', 'REFIT_JOINT4'],
  'title': '固定NATIVE7组效应',
  'result_status': 'FROZEN_NATIVE64_GROUP_EFFECT_DECOMPOSITION_COMPLETE',
  'validation_status': 'FROZEN_NATIVE64_GROUP_EFFECT_FRESH_REEXECUTION_PASS',
  'pins': {'result.json': 'eebbd6e9b15be6181ed375f04f6c8cbc7a985f5f871073c41b29c1f03a5975fe',
           'independent_validation.json': 'c0d4fd6e729d3e23e36fc3b2533cb92da823c33931569a740c88539f1e2445cd',
           'frozen_parameters.json': '019cfeac14575b97371f66f81e5ad6ec55ec321af502a0b9110e4cc207c79b43'},
  'parameter_file': 'frozen_parameters.json'},
 {'trial': 'rc_frozen_group_effect_difficult90_v1',
  'layout': 'legacy90',
  'heads': ['ORIGINAL', 'DROP_S', 'DROP_QR', 'DROP_S_QR'],
  'trial_baseline': 'ORIGINAL',
  'baseline_replays': ['ORIGINAL'],
  'title': '旧FROZEN_C difficult90组效应',
  'result_status': 'FROZEN_GROUP_EFFECT_DIFFICULT90_EXACT_REPLAY_COMPLETE',
  'validation_status': 'FROZEN_GROUP_EFFECT_DIFFICULT90_INDEPENDENT_REEXECUTION_PASS',
  'pins': {'result.json': 'c27f7c3aa543ba73130be059e1047415d86a35dc2fb17ffe5c00febfbbdb3356',
           'independent_validation.json': '13ce0b1d7d0bce7a0264f03f010466230b6def55236198bee5375166f7349b76'}},
 {'trial': 'rc_frozen_qr_removal_matched32_v1',
  'layout': 'legacy32',
  'heads': ['ORIGINAL', 'DROP_QR'],
  'trial_baseline': 'ORIGINAL',
  'baseline_replays': ['ORIGINAL'],
  'title': '旧FROZEN_C matched32同头桥接',
  'result_status': 'FROZEN_QR_REMOVAL_MATCHED32_EXACT_REPLAY_COMPLETE',
  'validation_status': 'FROZEN_QR_REMOVAL_MATCHED32_INDEPENDENT_REEXECUTION_PASS',
  'pins': {'result.json': 'e06a01ed4d820df8e15776a1fbc130d359814fa1f07f1683656e8fe272553e2e',
           'independent_validation.json': '3fb7fda90c00f874fb27a9fe08eca8d29e3a42b1274de1f6678fde7b5fb7ae79'}}]

LEGACY_PARAMETER_PINS = {
    'results/romav2_colnomic_full_negative_action_gate_v1/result.json': 'b296e946ff84ef574605ca5a6f6c6d6892b06dc8f38002b33b913a5afc6f059f',
    'results/romav2_colnomic_full_negative_action_gate_v1/independent_validation.json': '2f112c8daa7b0c199e7e7ca2933116bdaf261b547277140950c809cc3bd264de',
    'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py': '96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7',
}
NATIVE_DATASET = 'original_RAW_C128_matched_FULL64_native_action_bundle'
LEGACY32_DATASET = 'old_FROZEN_C_current_runtime_bridge_matched_EVAL32'
LEGACY90_DATASET = 'old_FROZEN_C_difficult90_opened_val38_test52'
METRIC_NAMES = ('query_count', 'base_top1', 'final_top1', 'rescue', 'break')
FIELDS = (
    'row_id', 'source_trial', 'source_head_key', 'head_lineage', 'condition', 'parameter_count',
    'dataset', 'split', 'evidence_mode', 'candidate_source', 'candidate_count', 'action_arithmetic', 'scoring_path',
    'baseline_name', 'query_count', 'baseline_top1', 'final_top1', 'rescue_vs_RAW', 'break_vs_RAW', 'net_vs_RAW',
    'trial_comparator', 'comparator_top1', 'paired_rescue_vs_comparator', 'paired_break_vs_comparator', 'paired_net_vs_comparator',
    'evidence_level', 'operation_kind', 'baseline_replay_duplicate', 'replay_note', 'model_adopted_by_this_record',
    'result_status', 'result_path', 'result_sha256', 'metric_json_pointer', 'actions_json_pointer', 'action_list_sha256',
    'validation_status', 'validation_path', 'validation_sha256', 'parameter_source_path', 'parameter_source_sha256',
)


def need(value, reason):
    if not bool(value): raise RuntimeError(reason)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pin(relative, digest):
    path = (ROOT / relative).resolve()
    need(path.is_relative_to(ROOT) and 'jackknife' not in relative, 'EXPLICIT_INDEX_SCOPE')
    need(sha(path) == digest, 'INPUT_HASH_DRIFT:' + relative)
    return json.loads(path.read_text())


def source_binding(relative, digest): return {'path': relative, 'sha256': digest}


def load_sources():
    loaded = {}; bindings = {}
    for spec in SPECS:
        trial = spec['trial']; source = {}
        for filename, digest in spec['pins'].items():
            relative = 'results/' + trial + '/' + filename
            source[filename] = read_pin(relative, digest); bindings[relative] = source_binding(relative, digest)
        result, validation = source['result.json'], source['independent_validation.json']
        need(result['status'] == spec['result_status'], 'EXACT_RESULT_STATUS')
        need(validation['status'] == spec['validation_status'] and validation['status'].endswith('_PASS'), 'EXACT_PASS_VALIDATION')
        need(validation['result_sha256'] == spec['pins']['result.json'], 'VALIDATION_RESULT_BINDING')
        need(validation['checks'] and all(v is True for v in validation['checks'].values()), 'ALL_SOURCE_CHECKS_TRUE')
        if spec['layout'] == 'standard':
            need(result['parameters_sha256'] == spec['pins'][spec['parameter_file']], 'RESULT_PARAMETER_FILE_BOUND')
        loaded[trial] = source
    old_head_relative, old_val_relative = tuple(LEGACY_PARAMETER_PINS)[:2]
    old_head = read_pin(old_head_relative, LEGACY_PARAMETER_PINS[old_head_relative])
    old_val = read_pin(old_val_relative, LEGACY_PARAMETER_PINS[old_val_relative])
    need(old_val['status'] == 'ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_INDEPENDENT_VALIDATION_PASS' and
         old_val['result_sha256'] == LEGACY_PARAMETER_PINS[old_head_relative] and
         all(v is True for v in old_val['checks'].values()), 'OLD_HEAD_PARAMETER_VALIDATION')
    need(old_head['parameter_count'] == len(old_head['head']['weight']) + 1 == 7 and
         all(math.isfinite(float(v)) for v in old_head['head']['weight'] + [old_head['head']['bias']]), 'FROZEN_C_SEVEN_PARAMETERS')
    for relative, digest in LEGACY_PARAMETER_PINS.items():
        need(sha(ROOT / relative) == digest, 'OLD_PARAMETER_SOURCE_PIN')
        bindings[relative] = source_binding(relative, digest)
    for trial in ('rc_frozen_group_effect_difficult90_v1', 'rc_frozen_qr_removal_matched32_v1'):
        gate = loaded[trial]['result.json']['sources']['frozen_gate']
        need(gate['sha256'] == LEGACY_PARAMETER_PINS[gate['path']], 'SAME_OLD_FROZEN_GATE')
    bridge = loaded['rc_frozen_qr_removal_matched32_v1']['result.json']
    need(bridge['sources']['head']['sha256'] == LEGACY_PARAMETER_PINS[old_head_relative] and
         bridge['same_head_two_bundle_bridge']['same_old_FROZEN_C_weights_bias_verified'] is True, 'LEGACY_SAME_WEIGHT_BRIDGE')
    return loaded, {'schema': 'new_hyp_verified_evidence_sources_v1', 'source_specs': SPECS,
        'program': source_binding(str(Path(__file__).relative_to(ROOT)), sha(__file__)),
        'source_files': bindings, 'trial_count': 9, 'pending_results_or_scheduler_read': False,
        'scope': 'nine explicit completed independently validated trials only; no automatic latest-file selection'}


def action_counts(actions, layout):
    need(actions and len({a['query_id'] for a in actions}) == len(actions), 'UNIQUE_ACTION_QUERY_AXIS')
    for action in actions:
        need(type(action['base_correct']) is bool and type(action['final_correct']) is bool, 'BOOLEAN_ACTION_CORRECTNESS')
        if layout in ('standard', 'native_fixed'):
            need(action['base_correct'] == (action['base_winner'] == action['target_position']) and
                 action['final_correct'] == (action['final_position'] == action['target_position']), 'NATIVE_POSITION_CORRECTNESS')
        elif layout == 'legacy32':
            need(action['target_present'] is True and action['base_correct'] == (action['base_winner'] == action['target_position']) and
                 action['final_correct'] == (action['final_position'] == action['target_position']), 'MATCHED32_TARGET_MEMBERSHIP')
        elif layout == 'legacy90':
            need(not action['final_correct'] or action['target_present_c128'], 'LEGACY90_TARGET_PRESENCE')
            final = action['proposed_challenger_position'] if action['effective_action'] == 'SWITCH' else action['base_winner_position']
            need(final == action['final_prediction_position'], 'LEGACY90_EFFECTIVE_ACTION')
        else: raise RuntimeError('UNMAPPED_ACTION_SCHEMA')
    correct_base = {a['query_id'] for a in actions if a['base_correct']}
    correct_final = {a['query_id'] for a in actions if a['final_correct']}
    return {'query_count': len(actions), 'base_top1': len(correct_base), 'final_top1': len(correct_final),
        'rescue': len(correct_final - correct_base), 'break': len(correct_base - correct_final)}


def parameters(spec, source, key):
    if spec['layout'] in ('legacy90', 'legacy32'):
        relative = tuple(LEGACY_PARAMETER_PINS)[0]
        return 7, relative, LEGACY_PARAMETER_PINS[relative]
    params = source[spec['parameter_file']]
    if spec['layout'] == 'native_fixed':
        parameter_key = {'ORIGINAL7': 'ORIGINAL7', 'DROP_S': 'ORIGINAL7', 'DROP_QR': 'ORIGINAL7', 'DROP_S_QR': 'ORIGINAL7',
                         'REFIT_RESPONSE6': 'RESPONSE6', 'REFIT_PRODUCT5': 'PRODUCT5', 'REFIT_JOINT4': 'JOINT4'}[key]
    else: parameter_key = key
    value = params[parameter_key]
    count = len(value['weight_binary64']) + 1
    need(count == value['parameter_count'], 'VERIFIED_PARAMETER_COUNT')
    need(all(math.isfinite(float.fromhex(v)) for v in value['weight_binary64'] + [value['bias_binary64']]), 'FINITE_PARAMETER_BYTES')
    relative = 'results/' + spec['trial'] + '/' + spec['parameter_file']
    return count, relative, spec['pins'][spec['parameter_file']]


def score_description(spec, key):
    trial = spec['trial']
    common = 'RoMa soft visibility x ColNomic full-reference image-token weighted MaxSim; RAW C128 all127 HOLD/SWITCH'
    if trial == 'rc_full_reference_content_bridge_v1':
        return {'ORIGINAL_C': common, 'D_IMAGE': 'original C quality mass and query weights; free reference image-token MaxSim; native action',
                'D_FULL': 'original D_IMAGE path with full original valid reference passage-token MaxSim axis; native action'}[key]
    if trial == 'rc_absolute_evidence_scale_calibration_v1':
        return common + {'NATIVE7': '; original native6', 'ABS12': '; native6 plus five TRAIN-RMS absolute differences',
                         'REL12': '; native6 plus five TRAIN-RMS signed-square relative features'}[key]
    if trial == 'rc_native7_training_objective_factorial_v1': return common + '; original native6; training objective=' + key
    if trial == 'rc_pair_control_scale_harmonization_v1':
        return common + ('; original mixed PAIR control definitions' if key == 'ORIGINAL7' else '; PAIR V1 Q/R controls harmonized to half-axis')
    if spec['layout'] == 'native_fixed': return common + '; fixed group intervention or already fitted subset replay=' + key
    if spec['layout'] in ('legacy90', 'legacy32'): return common + '; old FROZEN_C fixed features=' + key
    return common + '; original native feature subset=' + key


def slices(spec, result):
    if spec['layout'] == 'standard':
        for role in ('TRAIN', 'EVAL'):
            need(set(result['metrics'][role]) == set(spec['heads']), 'EXACT_MAPPED_HEADS')
            for mode in ('REAL', 'CBIND'):
                for key in spec['heads']:
                    yield role, mode, key, result['metrics'][role][key][mode], result['actions'][role][key][mode], result['actions'][role][spec['trial_baseline']][mode], '/metrics/' + role + '/' + key + '/' + mode, '/actions/' + role + '/' + key + '/' + mode
    elif spec['layout'] == 'native_fixed':
        for role in ('TRAIN', 'EVAL'):
            for mode in ('REAL', 'CBIND'):
                need(set(result['metrics'][role][mode]) == set(spec['heads']), 'EXACT_MAPPED_FIXED_CONDITIONS')
                for key in spec['heads']:
                    yield role, mode, key, result['metrics'][role][mode][key], result['actions'][role][mode][key], result['actions'][role][mode][spec['trial_baseline']], '/metrics/' + role + '/' + mode + '/' + key, '/actions/' + role + '/' + mode + '/' + key
    elif spec['layout'] in ('legacy90', 'legacy32'):
        need(set(result['summaries']) == set(spec['heads']), 'EXACT_LEGACY_CONDITIONS')
        role = 'OPENED_VAL38_TEST52' if spec['layout'] == 'legacy90' else 'EVAL'
        for key in spec['heads']:
            yield role, 'REAL', key, result['summaries'][key], result['actions'][key], result['actions'][spec['trial_baseline']], '/summaries/' + key, '/actions/' + key
    else: raise RuntimeError('UNMAPPED_RESULT_LAYOUT')


def build_rows(loaded):
    rows = []
    for spec in SPECS:
        trial = spec['trial']; source = loaded[trial]; result = source['result.json']; legacy = spec['layout'] in ('legacy90', 'legacy32')
        dataset = LEGACY90_DATASET if spec['layout'] == 'legacy90' else LEGACY32_DATASET if spec['layout'] == 'legacy32' else NATIVE_DATASET
        for role, mode, key, summary, actions, baseline_actions, metric_pointer, action_pointer in slices(spec, result):
            counts = action_counts(actions, spec['layout'])
            need(all(counts[k] == summary[k] for k in METRIC_NAMES), 'SUMMARY_ACTION_COUNT_MISMATCH:' + trial + ':' + key)
            expected_n = 90 if spec['layout'] == 'legacy90' else 32
            expected_raw = 61 if spec['layout'] == 'legacy90' else 27 if role == 'TRAIN' else 25
            need(counts['query_count'] == expected_n and counts['base_top1'] == expected_raw, 'DATASET_DENOMINATOR_AND_RAW_BASE')
            need(counts['final_top1'] == counts['base_top1'] + counts['rescue'] - counts['break'], 'NET_ARITHMETIC')
            need([a['query_id'] for a in actions] == [a['query_id'] for a in baseline_actions], 'COMPARATOR_QUERY_AXIS')
            correct = {a['query_id'] for a in actions if a['final_correct']}; old_correct = {a['query_id'] for a in baseline_actions if a['final_correct']}
            count, parameter_path, parameter_sha = parameters(spec, source, key)
            duplicate = key in spec['baseline_replays']
            counterfactual = key in ('DROP_S', 'DROP_QR', 'DROP_S_QR')
            operation = 'frozen_computation_counterfactual' if counterfactual else 'validated_existing_head_replay' if duplicate else 'fixed_protocol_retrained_ablation'
            evidence = 'TRAIN_DESCRIPTIVE' if role == 'TRAIN' else 'OPENED_INTERNAL_EVAL'
            if counterfactual: evidence += '_FROZEN_COUNTERFACTUAL'
            if mode == 'CBIND': evidence += '_CBIND_CONTROL'
            if legacy: lineage = 'FROZEN_C_original_seven_parameter_head'
            elif spec['layout'] == 'native_fixed': lineage = 'NATIVE7_original_fixed_head' if not key.startswith('REFIT_') else 'native_feature_subset_' + key.removeprefix('REFIT_')
            else: lineage = 'native_feature_action_' + key
            row = {'row_id': trial + '|' + role + '|' + mode + '|' + key, 'source_trial': trial, 'source_head_key': key,
                'head_lineage': lineage, 'condition': key, 'parameter_count': count, 'dataset': dataset, 'split': role,
                'evidence_mode': mode, 'candidate_source': 'original RAW full-gallery C128; no target insertion', 'candidate_count': 128,
                'action_arithmetic': 'FP64_rowwise_(weight*feature).sum()+bias' if legacy else 'FP64_complete_matrix_X@weight+bias',
                'scoring_path': score_description(spec, key), 'baseline_name': 'RAW_without_action',
                'query_count': counts['query_count'], 'baseline_top1': counts['base_top1'], 'final_top1': counts['final_top1'],
                'rescue_vs_RAW': counts['rescue'], 'break_vs_RAW': counts['break'], 'net_vs_RAW': counts['rescue'] - counts['break'],
                'trial_comparator': spec['trial_baseline'] + '/' + mode, 'comparator_top1': len(old_correct),
                'paired_rescue_vs_comparator': len(correct - old_correct), 'paired_break_vs_comparator': len(old_correct - correct),
                'paired_net_vs_comparator': len(correct) - len(old_correct), 'evidence_level': evidence, 'operation_kind': operation,
                'baseline_replay_duplicate': duplicate, 'replay_note': 'replay of an existing validated head on this bundle; not a new independent experiment' if duplicate else '',
                'model_adopted_by_this_record': False, 'result_status': spec['result_status'], 'result_path': 'results/' + trial + '/result.json',
                'result_sha256': spec['pins']['result.json'], 'metric_json_pointer': metric_pointer, 'actions_json_pointer': action_pointer,
                'action_list_sha256': hashlib.sha256(encode(actions)).hexdigest(), 'validation_status': spec['validation_status'],
                'validation_path': 'results/' + trial + '/independent_validation.json', 'validation_sha256': spec['pins']['independent_validation.json'],
                'parameter_source_path': parameter_path, 'parameter_source_sha256': parameter_sha}
            need(set(row) == set(FIELDS), 'EXACT_CSV_SCHEMA'); rows.append(row)
    need(len(rows) == len({r['row_id'] for r in rows}) == 118, 'EXACT_118_ROWS_NINE_TRIALS')
    return rows


def cell(value):
    if type(value) is bool: return 'true' if value else 'false'
    if value is None: return ''
    return str(value)


def csv_bytes(rows):
    buffer = io.StringIO(newline=''); writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader(); writer.writerows({k: cell(row[k]) for k in FIELDS} for row in rows)
    data = buffer.getvalue().encode()
    parsed = list(csv.DictReader(io.StringIO(data.decode(), newline='')))
    need(len(parsed) == len(rows), 'CSV_LINE_COUNT')
    for raw, row in zip(parsed, rows, strict=True): need(raw == {k: cell(row[k]) for k in FIELDS}, 'CSV_ALL_CELL_ROUNDTRIP')
    return data


def concise_markdown(rows):
    text = '# new HYP 已验证证据索引\n\n'
    text += '九个明确指定的已验证试验，共118条记录。每条CSV/JSON记录都绑定原结果、独立验证、参数来源与算术路径；全部正确数和救/损从原动作重算。没有新增预测或训练。\n\n'
    text += '| 试验 | 数据与头范围 | 已打开 REAL 正确数 | RAW基线 |\n|---|---|---|---:|\n'
    for spec in SPECS:
        selected = [r for r in rows if r['source_trial'] == spec['trial'] and r['split'] != 'TRAIN' and r['evidence_mode'] == 'REAL']
        values = '；'.join(r['source_head_key'] + '=' + str(r['final_top1']) + '/' + str(r['query_count']) for r in selected)
        scope = '旧FROZEN_C difficult90' if spec['layout'] == 'legacy90' else '旧FROZEN_C matched EVAL32' if spec['layout'] == 'legacy32' else 'native系列 matched EVAL32'
        text += '| ' + spec['title'] + ' | ' + scope + ' | ' + values + ' | ' + str(selected[0]['baseline_top1']) + ' |\n'
    text += '\n旧FROZEN_C的DROP_QR在difficult90为70/90、9救0损；同一旧头在matched32为27/32、3救1损。后者与原27/32同分，但正确集合有变化，因此不能称两个bundle都无损。NATIVE7的28/32属于另一个头，不能与旧69/70合并。\n\n'
    text += '反复出现的原模型和已训练子集回放标为 `baseline_replay_duplicate=true`，不算新增独立实验。固定特征置零都标为计算反事实；索引没有采用任何新模型。TRAIN、EVAL、REAL与C_BIND分别保存，分母不合并。\n\n'
    text += '完整数据：[CSV](evidence.csv)、[JSON](evidence.json)、[来源清单](sources_manifest.json)、[独立重建验证](independent_validation.json)。\n'
    return text.encode()


def exclusive(path, data):
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    path.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('build', 'validate'), required=True)
    validate = parser.parse_args().phase == 'validate'
    need(OUT.exists() if validate else not OUT.exists(), 'APPEND_ONLY_EVIDENCE_INDEX')
    loaded, manifest = load_sources(); rows = build_rows(loaded)
    payload = {'schema': 'new_hyp_verified_evidence_index_v1', 'theory_name': 'new HYP', 'trial_count': 9,
        'row_count': 118, 'columns': list(FIELDS), 'rows': rows, 'model_adoption_changed': False,
        'new_predictions': 0, 'training_updates': 0, 'scheduler_calls': 0,
        'limits': ['An evidence index does not add independent scientific samples or establish universal HYP.',
                   'Counterfactual feature-zero outcomes are not deployed or selected models.',
                   'Old FROZEN_C and native heads, separate datasets, and repeated baseline replays remain separate.']}
    artifacts = {'sources_manifest.json': encode(manifest) + b'\n', 'evidence.json': encode(payload) + b'\n',
                 'evidence.csv': csv_bytes(rows), 'README.md': concise_markdown(rows)}
    if validate:
        for filename, data in artifacts.items(): need((OUT / filename).read_bytes() == data, 'FRESH_SOURCE_REBUILD:' + filename)
        independent = {'status': 'NEW_HYP_EVIDENCE_INDEX_INDEPENDENT_REBUILD_PASS',
            'artifacts': {name: sha(OUT / name) for name in artifacts}, 'row_count': 118, 'trial_count': 9,
            'checks': {'all_nine_results_match_explicit_validation_PASS_and_hash': True,
                'all118_action_count_summaries_recomputed': True, 'all118_same_trial_comparator_counts_recomputed': True,
                'all_parameter_counts_bound_to_verified_frozen_files': True, 'CSV_every_cell_roundtrips_and_matches_JSON': True,
                'old_FROZEN_C_and_native_bundles_separate': True, 'no_pending_job_inputs_predictions_training_or_scheduler_reads': True}}
        path = OUT / 'independent_validation.json'
        if path.exists(): need(path.read_bytes() == encode(independent) + b'\n', 'VALIDATION_REPLAY')
        else: exclusive(path, encode(independent) + b'\n')
    else:
        OUT.mkdir()
        for filename, data in artifacts.items(): exclusive(OUT / filename, data)
    print(json.dumps({'status': 'NEW_HYP_EVIDENCE_INDEX_INDEPENDENT_REBUILD_PASS' if validate else 'NEW_HYP_EVIDENCE_INDEX_BUILT',
        'row_count': 118, 'trial_count': 9, 'evidence_sha256': sha(OUT / 'evidence.json')}), flush=True)


if __name__ == '__main__': main()
