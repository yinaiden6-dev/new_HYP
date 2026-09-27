#!/usr/bin/env python3
"""Independent F128, same-population B_CAL then frozen residual verification.

Reuses independent mathematical helpers, never trainer-generated correctness.
Baseline stage does not read held labels. All stage first seals/replays 60 fits,
then opens curator labels for reporting. Per-fit hash receipts make replay resume
bounded and safe; changing any input, artifact or verifier invalidates the receipt.
"""
import argparse
import fcntl
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from join_rebut_qr_qrr_v1 import (bind, checked, decision, held_loss, keyed,
    read, scale_for, write_csv, write_json, calibrate)
from join_rebut_qr_qrr_frozenbase_v2 import (need, choose_epoch, baseline_binding,
    record_stats, head_equal, replay, collect_predictions)
from rebut_qr_qrr_f128_common_v1 import F128Inputs, verify_panel, verify_sources, verify_evidence_gate
from rc_aslo_xf.rebut_qr_qrr_v1 import RebutScorer

RC = Path(__file__).resolve().parents[1]
ARMS = ('QR', 'QR_VEC', 'QRR')
FIELDS = ('train_query_ids', 'inner_fit_query_ids', 'inner_val_query_ids', 'heldout_query_ids')


def exact_state(a, b, description):
    need(set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), description)


def common_artifacts(folder, residual=False):
    names = ['result.json', 'model.pt', 'checkpoint.pt', 'inner_validation.json']
    if residual:
        names += ['main_inner_model.pt', 'inner_best_ce_model.pt']
        names += [str(p.relative_to(folder)) for p in sorted((folder / 'inner').glob('epoch*.json'))]
    return {name: bind(folder / name) for name in names}


def validate_baseline(protocol_path, protocol, fold, seed, folder, data):
    common, evidence = data.common, data.evidence
    split, roles, gallery = protocol['folds'][str(fold)], data.roles, data.identity
    result, inner = read(folder / 'result.json'), read(folder / 'inner_validation.json')
    saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    expected = {'protocol': bind(protocol_path),
                'runner_sha256': bind(RC / 'programs/run_rebut_qr_qrr_f128_v1.py')['sha256'],
                'module_sha256': bind(RC / 'src/rc_aslo_xf/rebut_qr_qrr_v1.py')['sha256'],
                'fold': fold, 'seed': seed, 'arm': 'B_CAL'}
    need(result['status'] == 'TRAINING_AND_LABEL_FREE_PREDICTIONS_COMPLETE', 'BASELINE_STATUS')
    need(all(v['binding'] == expected for v in (result, inner, saved, checkpoint)), 'BASELINE_BINDING')
    need(checkpoint['stage'] == 'done' and result['outer_labels_opened'] is False, 'BASELINE_SEAL')
    need(result['panel'].startswith('F128') and result['population_queries'] == 128, 'BASELINE_PANEL')
    need(result['evidence_bindings'] == {}, 'BASELINE_READS_NO_F')
    need(result['model_sha256'] == bind(folder / 'model.pt')['sha256'], 'BASELINE_MODEL_SHA')
    need(set(saved['model']) == {'head.weight'} and saved['model']['head.weight'].shape == (1, 18), 'BASELINE_HEAD18')
    need(result['trainable_parameters'] == 18, 'BASELINE_PARAMETER_COUNT')
    for field in FIELDS:
        need(result[field] == split[field], 'BASELINE_SPLIT:' + field)
    exact_state(saved['model'], checkpoint['model'], 'BASELINE_CHECKPOINT_MODEL')
    inner_scale, outer_scale = scale_for(common, split['inner_fit_query_ids']), scale_for(common, split['train_query_ids'])
    need(torch.equal(inner_scale, checkpoint['inner_normalization']), 'BASELINE_INNER_SCALE')
    need(torch.equal(outer_scale, saved['normalization']), 'BASELINE_OUTER_SCALE')
    need(torch.equal(outer_scale, checkpoint['normalization']), 'BASELINE_CHECKPOINT_SCALE')
    need(np.array_equal(inner_scale.numpy(), inner['normalization_rms']), 'BASELINE_INNER_JSON_SCALE')
    need(np.array_equal(outer_scale.numpy(), result['normalization_rms']), 'BASELINE_JSON_SCALE')
    ih = [r for r in result['history'] if r['stage'] == 'inner']
    rh = [r for r in result['history'] if r['stage'] == 'refit']
    need(result['history'] == checkpoint['history'], 'BASELINE_HISTORY')
    need(ih and [r['epoch'] for r in ih] == list(range(1, len(ih) + 1)), 'BASELINE_INNER_EPOCHS')
    chosen = min(ih, key=lambda r: (r['validation_loss'], r['epoch']))['epoch']
    need(chosen == result['best_epoch'] == inner['best_epoch'] == checkpoint['best_epoch'], 'BASELINE_EPOCH_SELECTION')
    need([r['epoch'] for r in rh] == list(range(1, chosen + 1)), 'BASELINE_REFIT_EPOCHS')
    patience, best = 0, math.inf
    for row in ih:
        need(patience < protocol['optimizer']['patience'], 'BASELINE_TRAINED_AFTER_STOP')
        if row['validation_loss'] < best:
            best, patience = row['validation_loss'], 0
        else:
            patience += 1
    need(len(ih) <= protocol['optimizer']['max_epochs'] and
         (len(ih) == protocol['optimizer']['max_epochs'] or patience >= protocol['optimizer']['patience']), 'BASELINE_STOP_RULE')
    need(checkpoint['updates'] == len(ih)*len(split['inner_fit_query_ids']) + chosen*len(split['train_query_ids']), 'BASELINE_UPDATES')
    inner_saved = {'model': checkpoint['best_model'], 'normalization': inner_scale}
    maximum = max(replay('B_CAL', inner_saved, inner['records'], common, evidence, split['inner_val_query_ids'], 'baseline_inner'),
                  replay('B_CAL', saved, result['predictions'], common, evidence, split['heldout_query_ids'], 'baseline_held'))
    stats = record_stats(inner['records'], common, split['inner_val_query_ids'], gallery, roles)
    need(abs(stats['validation_loss'] - best) < 2e-10, 'BASELINE_BEST_CE')
    tau, rescues, breaks = calibrate(inner['records'], common, gallery, roles)
    need(result['calibration'] == inner['calibration'] == checkpoint['calibration'], 'BASELINE_CALIBRATION')
    need(tau == result['calibration']['threshold'] and tau.hex() == result['calibration']['threshold_hex'], 'BASELINE_TAU')
    need(rescues == result['calibration']['rescues'] and breaks == result['calibration']['breaks'], 'BASELINE_CALIBRATION_COUNTS')
    for row in result['predictions']:
        zero, pos = decision(row['logits'], row['winner'], 0.), decision(row['logits'], row['winner'], tau)
        need(row['selected_position'] == pos and row['selected_physical_row'] == row['axis'][pos], 'BASELINE_DECISION')
        need(row['selected_fixed_zero_position'] == zero and row['selected_fixed_zero_physical_row'] == row['axis'][zero], 'BASELINE_ZERO_DECISION')
    return {'fold': fold, 'seed': seed, 'arm': 'B_CAL', 'selected_epoch': chosen,
            'selected_tau': tau, 'max_replay_error': maximum}


def validate_residual(protocol_path, protocol, fold, seed, arm, folder, data):
    common, evidence, evidence_bindings = data.common, data.evidence, data.bindings
    split, roles, gallery = protocol['folds'][str(fold)], data.roles, data.identity
    source = baseline_binding(protocol, fold, seed)
    sr = read(checked(source['source_result']))
    sm = torch.load(checked(source['source_model']), map_location='cpu', weights_only=False)
    sc = torch.load(checked(source['source_checkpoint']), map_location='cpu', weights_only=False)
    need(sr['binding']['protocol'] == protocol['source_protocol'], 'SAME_F128_BASELINE_PROTOCOL')
    need(sr['binding']['fold'] == fold and sr['binding']['seed'] == seed and sr['binding']['arm'] == 'B_CAL', 'SAME_F128_BASELINE_ID')
    for field in FIELDS:
        need(sr[field] == split[field], 'SAME_F128_BASELINE_SPLIT')
    protocol_bind = bind(protocol_path)
    manifest_path = Path(protocol['evidence_manifest'])
    module_sha = bind(RC / 'src/rc_aslo_xf/rebut_qr_qrr_v1.py')['sha256']
    runner_sha = bind(RC / 'programs/run_rebut_qr_qrr_f128_v1.py')['sha256']
    maximum, receipts, predictions = 0.0, [], {}
    result = read(folder / 'result.json')
    saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    inner = read(folder / 'inner_validation.json')
    main_saved = torch.load(folder / 'main_inner_model.pt', map_location='cpu', weights_only=False)
    best_ce_saved = torch.load(folder / 'inner_best_ce_model.pt', map_location='cpu', weights_only=False)
    expected = {'protocol': protocol_bind, 'runner_sha256': runner_sha, 'module_sha256': module_sha,
                'fold': fold, 'seed': seed, 'arm': arm, 'evidence_manifest': bind(manifest_path)}
    # The phase-specific source binding may additionally be embedded in binding.
    for k, v in expected.items():
        need(result['binding'].get(k) == v, 'RESULT_BINDING:' + k)
    for value in (saved, checkpoint, inner, main_saved, best_ce_saved):
        need(value['binding'] == result['binding'], 'ARTIFACT_BINDING')
    for binding in result['artifacts'].values():
        checked(binding)
    need(result['binding']['baseline_bindings'] == source, 'EMBEDDED_BASELINE_BINDING')
    need(result['binding']['v1_runner_sha256'] == protocol['dependency_bindings']['runner_v1']['sha256'], 'SOURCE_RUNNER_BINDING')
    checked(result['binding']['baseline_inner_predictions'])
    need(result['baseline_bindings'] == source, 'SOURCE_BINDINGS')
    need(result['outer_labels_opened'] is False and checkpoint['stage'] == 'done', 'FINAL_SEAL')
    need(result['head_exact_unchanged'] is True, 'FROZEN_HEAD_RECEIPT')
    need(result['evidence_bindings'] == evidence_bindings, 'EVIDENCE_BINDINGS')
    need(result['model_sha256'] == bind(folder / 'model.pt')['sha256'], 'MODEL_SHA')
    for field in ('train_query_ids', 'inner_fit_query_ids', 'inner_val_query_ids', 'heldout_query_ids'):
        need(result[field] == split[field], 'RESULT_POPULATION')
    head_equal(saved['model'], sm['model'])
    head_equal(checkpoint['model'], sm['model'])
    head_equal(main_saved['model'], sc['best_model'])
    head_equal(best_ce_saved['model'], sc['best_model'])
    need(set(saved['model']) == set(checkpoint['model']) and all(
        torch.equal(saved['model'][k], checkpoint['model'][k]) for k in saved['model']), 'FINAL_CHECKPOINT_MODEL')
    need(torch.equal(saved['normalization'], sm['normalization']), 'OUTER_PHASE_SCALE')
    need(torch.equal(main_saved['normalization'], sc['inner_normalization']), 'INNER_PHASE_SCALE')
    need(torch.equal(best_ce_saved['normalization'], sc['inner_normalization']), 'INNER_CE_PHASE_SCALE')
    need(np.array_equal(inner['normalization_rms'], sc['inner_normalization'].numpy()), 'INNER_JSON_SCALE')
    need(np.array_equal(result['normalization_rms'], sm['normalization'].numpy()), 'RESULT_SCALE')
    tau = float(sr['calibration']['threshold'])
    need(float(result['calibration']['threshold']) == float(inner['threshold']) == 0.0, 'MAIN_THRESHOLD_NOT_ZERO')
    need(tau == float(result['calibration']['inherited_tau_secondary']) == float(inner['inherited_tau_secondary']), 'TAU_WAS_RETRAINED')
    need(result['calibration']['new_threshold_fitting'] is False, 'THRESHOLD_REFIT')
    need(result['input_F_standardized'] is False, 'UNDECLARED_F_NORMALIZATION')
    epoch_paths = sorted((folder / 'inner').glob('epoch*.json'))
    need(bool(epoch_paths), 'INNER_EPOCH_RECORDS_REQUIRED')
    epoch_records, stats = {}, []
    for ep in epoch_paths:
        value = read(ep)
        if 'binding' in value:
            need(value['binding'] == result['binding'], 'EPOCH_BINDING')
        recomputed = record_stats(value['records'], common, split['inner_val_query_ids'], gallery, roles)
        need(abs(recomputed['validation_loss'] - value['validation_loss']) < 2e-10, 'EPOCH_LOSS')
        need(recomputed['fixed0_correct'] == value['fixed0_correct'], 'EPOCH_CORRECT')
        need(value['inherited_tau'] == tau, 'EPOCH_INHERITED_TAU')
        stats.append({'epoch': value['epoch'], **recomputed})
        epoch_records[value['epoch']] = value['records']
    selected, best_ce, base_correct = choose_epoch(stats)
    selection = result['selection']
    need(selected == result['best_epoch'] == selection['main_epoch'], 'SELECTED_EPOCH')
    need(best_ce == selection['best_ce_epoch'], 'BEST_CE_EPOCH')
    need(base_correct == selection['baseline_fixed0_correct'], 'BASELINE_INNER_CORRECT')
    need((selected == 0) == selection['epoch0_fallback'], 'FALLBACK_FLAG')
    need(result['reader_enabled'] == saved['reader_enabled'] == (selected > 0), 'READER_ENABLE_FLAG')
    need(result['selection'] == inner['selection'] == checkpoint['selection'], 'SELECTION_BINDING')
    need(main_saved['epoch'] == selected and best_ce_saved['epoch'] == best_ce, 'INNER_MODEL_EPOCH')
    need(inner['records'] == epoch_records[selected], 'INNER_SELECTED_RECORDS')
    need(inner['best_ce_records'] == epoch_records[best_ce], 'INNER_CE_RECORDS')
    need(inner['baseline_records'] == epoch_records[0], 'INNER_BASELINE_RECORDS')
    need(set(main_saved['model']) == set(checkpoint['main_inner_model']) and all(
        torch.equal(main_saved['model'][k], checkpoint['main_inner_model'][k])
        for k in main_saved['model']), 'SELECTED_INNER_CHECKPOINT')
    need(set(best_ce_saved['model']) == set(checkpoint['best_ce_model']) and all(
        torch.equal(best_ce_saved['model'][k], checkpoint['best_ce_model'][k])
        for k in best_ce_saved['model']), 'BEST_CE_CHECKPOINT')
    inner_base = {'model': sc['best_model'], 'normalization': sc['inner_normalization']}
    maximum = max(maximum, replay('B_CAL', inner_base, epoch_records[0], common, evidence,
                                  split['inner_val_query_ids'], 'baseline_inner_epoch0'))
    maximum = max(maximum, replay(arm, main_saved, inner['records'], common, evidence,
                                  split['inner_val_query_ids'], 'selected_inner'))
    maximum = max(maximum, replay(arm, best_ce_saved, inner['best_ce_records'], common, evidence,
                                  split['inner_val_query_ids'], 'best_ce_inner_diagnostic'))
    maximum = max(maximum, replay(arm, saved, result['predictions'], common, evidence,
                                  split['heldout_query_ids'], 'outer_refit'))
    history = result['history']
    ih = [r for r in history if r['stage'] == 'inner']
    rh = [r for r in history if r['stage'] == 'refit']
    need([r['epoch'] for r in rh] == list(range(1, selected + 1)), 'REFIT_EPOCH_BUDGET')
    need([r['epoch'] for r in ih] == list(range(len(stats))), 'INNER_HISTORY_COVERAGE')
    for h in ih:
        need(abs(h['validation_loss'] - stats[h['epoch']]['validation_loss']) < 2e-10, 'HISTORY_LOSS')
        need(h['fixed_zero_correct'] == stats[h['epoch']]['fixed0_correct'], 'HISTORY_CORRECT')
    # Verify early stopping includes baseline epoch zero; no extra epochs are allowed.
    patience, best_loss = 0, stats[0]['validation_loss']
    maximum_epochs = int(protocol['optimizer']['max_epochs'])
    limit = int(protocol['optimizer']['patience'])
    need(len(stats) - 1 <= maximum_epochs, 'MAX_EPOCH_BUDGET')
    for s in stats[1:]:
        need(patience < limit, 'TRAINED_AFTER_EARLY_STOP')
        if s['validation_loss'] < best_loss:
            best_loss, patience = s['validation_loss'], 0
        else:
            patience += 1
    need(patience >= limit or len(stats) - 1 == maximum_epochs, 'UNJUSTIFIED_EARLY_STOP')
    need(checkpoint['patience_used'] == patience, 'PATIENCE_ACCOUNTING')
    inner_updates = (len(stats) - 1) * len(split['inner_fit_query_ids'])
    refit_updates = selected * len(split['train_query_ids'])
    need(checkpoint['updates'] == inner_updates + refit_updates, 'TRAINING_UPDATE_ACCOUNTING')
    for stage, expected_updates in (('inner', inner_updates), ('refit', refit_updates)):
        need(result['gradient_summary'].get(stage, {}).get('updates', 0) == expected_updates, 'GRADIENT_UPDATE_ACCOUNTING')
    shape_model = RebutScorer(arm, 18, 72, 72, head_bias=False)
    need(result['trainable_parameters'] == sum(p.numel() for p in shape_model.reader.parameters()), 'TRAINABLE_READER_PARAMETER_COUNT')
    need(result['frozen_head_parameters'] == 18, 'FROZEN_HEAD_PARAMETER_COUNT')
    if selected == 0:
        need(torch.count_nonzero(saved['model']['reader.output.weight']).item() == 0, 'FALLBACK_NONZERO_RESIDUAL')
        baseline_rows = keyed(sr['predictions'])
        for r in result['predictions']:
            need(r['logits'] == baseline_rows[r['query_id']]['logits'], 'FALLBACK_NOT_EXACT_BASELINE')
        need(result['gradient_summary'].get('refit', {}).get('updates', 0) == 0, 'FALLBACK_REFIT_UPDATES')
    for r in result['predictions']:
        zero, secondary = decision(r['logits'], r['winner'], 0.0), decision(r['logits'], r['winner'], tau)
        need(r['selected_position'] == r['selected_fixed_zero_position'] == zero, 'MAIN_DECISION')
        need(r['selected_physical_row'] == r['selected_fixed_zero_physical_row'] == r['axis'][zero], 'MAIN_PHYSICAL_DECISION')
        need(r['inherited_tau_selected_position'] == secondary and r['inherited_tau_selected_physical_row'] == r['axis'][secondary], 'SECONDARY_DECISION')
        need(r['action'] == ('HOLD' if zero == r['winner'] else 'SWITCH'), 'MAIN_ACTION')
    collect_predictions(predictions, arm, seed, fold, result['predictions'], tau)
    receipts.append({'fold': fold, 'seed': seed, 'arm': arm, 'result': bind(folder / 'result.json'),
                     'model': bind(folder / 'model.pt'), 'checkpoint': bind(folder / 'checkpoint.pt'),
                     'inner_validation': bind(folder / 'inner_validation.json'),
                     'main_inner_model': bind(folder / 'main_inner_model.pt'),
                     'epoch_records': [bind(p) for p in epoch_paths], 'baseline_bindings': source,
                     'selected_epoch': selected, 'best_ce_epoch': best_ce,
                     'epoch0_fallback': selected == 0, 'inherited_tau': tau})
    need(result['panel'].startswith('F128') and result['population_queries'] == 128, 'RESIDUAL_PANEL')
    need(result['history'] == checkpoint['history'], 'RESIDUAL_CHECKPOINT_HISTORY')
    return {**receipts[0], 'max_replay_error': maximum}


def wait_for_results(root, protocol, residual):
    arms = ARMS if residual else ('B_CAL',)
    total = 45 if residual else 15
    status = 'FROZEN_BASE_RESIDUAL_LABEL_FREE_COMPLETE' if residual else 'TRAINING_AND_LABEL_FREE_PREDICTIONS_COMPLETE'
    missing = []
    for fold in range(5):
        for seed in protocol['seeds']:
            for arm in arms:
                folder = root / f'fold{fold}/seed{seed}/{arm}'
                p = folder / 'result.json'
                if not p.exists() or read(p).get('status') != status:
                    missing.append({'fold': fold, 'seed': seed, 'arm': arm})
    if missing:
        print(json.dumps({'status': 'F128_WAITING_FOR_RESULTS', 'phase': 'residual' if residual else 'baseline',
                          'complete': total-len(missing), 'total': total, 'missing': missing,
                          'held_labels_opened': False}), flush=True)
        raise SystemExit(75)


def verify_phase(root, protocol_path, residual, deadline):
    protocol = read(protocol_path)
    verify_panel(protocol)
    verify_sources(protocol)
    evidence_gate = verify_evidence_gate(protocol)
    out = Path(protocol['output_root'])
    wait_for_results(out, protocol, residual)
    if residual:
        source = read(checked(protocol['source_validation']))
        need(source['status'] == 'F128_BASELINE15_INDEPENDENT_PASS' and source['protocol'] == protocol['source_protocol'], 'F128_BASELINE_GATE')
    arms = ARMS if residual else ('B_CAL',)
    manifest_binding = bind(protocol['evidence_manifest'])
    manifest = read(protocol['evidence_manifest'])
    need(manifest['status'] == 'POOLED_F128_COMPLETE' and len(manifest['records']) == 128 and
         set(r['query_id'] for r in manifest['records']) == set(protocol['panel_query_ids']), 'F128_MANIFEST')
    evidence_inputs = {r['query_id']: {'path': str(checked(r['payload'])), 'sha256': r['payload']['sha256']}
                       for r in manifest['records']}
    checked(protocol['common_features']); checked(protocol['gallery'])
    for split in protocol['folds'].values():
        checked(split['train_roles'])
    receipts, maximum, data = [], 0.0, None
    receipt_root = root / 'independent_receipts' / ('residual' if residual else 'baseline')
    receipt_root.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        data = None
        for seed in protocol['seeds']:
            for arm in arms:
                if time.monotonic() >= deadline - 20:
                    print(json.dumps({'status': 'F128_REPLAY_RESUMABLE', 'fits_verified_this_phase': len(receipts), 'held_labels_opened': False}), flush=True)
                    raise SystemExit(75)
                folder = out / f'fold{fold}/seed{seed}/{arm}'
                receipt_path = receipt_root / f'fold{fold}_seed{seed}_{arm}.json'
                inputs = {'protocol': bind(protocol_path), 'verifier': bind(__file__),
                          'data_adapter': bind(RC / 'programs/rebut_qr_qrr_f128_common_v1.py'),
                          'math_v1': bind(RC / 'programs/join_rebut_qr_qrr_v1.py'),
                          'math_v2': bind(RC / 'programs/join_rebut_qr_qrr_frozenbase_v2.py'),
                          'common': protocol['common_features'], 'gallery': protocol['gallery'],
                          'train_roles': protocol['folds'][str(fold)]['train_roles'],
                          'evidence_manifest': manifest_binding,
                          'evidence_validation': evidence_gate,
                          'artifacts': common_artifacts(folder, residual)}
                inputs['evidence'] = evidence_inputs
                if residual:
                    inputs['baseline'] = baseline_binding(protocol, fold, seed)
                    for value in inputs['baseline'].values():
                        checked(value)
                    inputs['baseline_inner'] = bind(Path(inputs['baseline']['source_result']['path']).parent / 'inner_validation.json')
                    inputs['baseline_validation'] = protocol['source_validation']
                prior = read(receipt_path) if receipt_path.exists() else None
                if prior and prior.get('status') == 'F128_FIT_INDEPENDENT_PASS' and prior.get('inputs') == inputs:
                    receipt = prior
                else:
                    if data is None:
                        data = F128Inputs(protocol, protocol['folds'][str(fold)], arm)
                    details = (validate_residual(protocol_path, protocol, fold, seed, arm, folder, data) if residual else
                               validate_baseline(protocol_path, protocol, fold, seed, folder, data))
                    receipt = {'status': 'F128_FIT_INDEPENDENT_PASS', 'inputs': inputs,
                               'held_labels_opened': False, **details}
                    write_json(receipt_path, receipt)
                maximum = max(maximum, receipt['max_replay_error'])
                receipts.append({'receipt': bind(receipt_path), 'fold': fold, 'seed': seed, 'arm': arm,
                                 'selected_epoch': receipt['selected_epoch'], 'max_replay_error': receipt['max_replay_error']})
                print(json.dumps({'stage': 'F128_independent_replay', 'phase': 'residual' if residual else 'baseline',
                                  'complete': len(receipts), 'total': 45 if residual else 15,
                                  'fold': fold, 'seed': seed, 'arm': arm,
                                  'cached': bool(prior and prior.get('inputs') == inputs)}), flush=True)
        del data
    need(bind(protocol['evidence_manifest']) == manifest_binding, 'MANIFEST_CHANGED_DURING_REPLAY')
    for value in evidence_inputs.values():
        checked(value)
    return protocol, receipts, maximum


def stable_write(path, value):
    if path.exists():
        need(read(path) == value, 'EXISTING_SEAL_CHANGED:' + str(path))
    else:
        write_json(path, value)


def seal_baselines(root, root_protocol, baseline_path, baseline_protocol, receipts, maximum):
    validation = {'status': 'F128_BASELINE15_INDEPENDENT_PASS', 'protocol': bind(baseline_path),
                  'root_protocol': bind(root / 'protocol.json'), 'fits_complete': 15, 'panel_queries': 128,
                  'held_predictions_replayed': 128*3, 'held_scores_replayed': 128*3*128,
                  'held_labels_opened': False, 'maximum_model_replay_error': maximum,
                  'receipts': receipts, 'verifier': bind(__file__)}
    validation_path = root / 'baseline_validation.json'
    stable_write(validation_path, validation)
    bindings = {}
    for fold in range(5):
        bindings[str(fold)] = {}
        for seed in baseline_protocol['seeds']:
            folder = Path(baseline_protocol['output_root']) / f'fold{fold}/seed{seed}/B_CAL'
            bindings[str(fold)][str(seed)] = {key: bind(folder / name) for key, name in (
                ('source_result', 'result.json'), ('source_model', 'model.pt'), ('source_checkpoint', 'checkpoint.pt'))}
    rp = {**root_protocol, 'arms': list(ARMS), 'output_root': str(root / 'residual'),
          'baseline_bindings': bindings, 'source_protocol': bind(baseline_path),
          'selection': root_protocol['residual_selection'], 'threshold': root_protocol['residual_threshold'],
          'source_validation': bind(validation_path),
          'dependency_bindings': {'runner_v1': bind(RC / 'programs/run_rebut_qr_qrr_v1.py'),
                                  'runner_v2': bind(RC / 'programs/run_rebut_qr_qrr_frozenbase_v2.py')}}
    stable_write(root / 'residual_protocol.json', rp)
    return validation


def write_final(root, root_protocol, baseline_protocol, residual_protocol, receipts, maximum):
    # This is the ONLY place held labels are opened, after all 60 replay receipts.
    need(len(receipts) == 60, 'ALL60_REQUIRED_BEFORE_LABELS')
    curator_path = RC / 'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json'
    curator = keyed(read(curator_path)['records'])
    common = keyed(torch.load(checked(root_protocol['common_features']), map_location='cpu', weights_only=False)['records'])
    gallery = {int(r['physical_row']): r['identity'] for r in read(checked(root_protocol['gallery']))['records']}
    panel = sorted(root_protocol['panel_query_ids'])
    need(set(panel) <= set(curator), 'CURATOR_PANEL')
    predictions = {}
    for phase_protocol, arms in ((baseline_protocol, ('B_CAL',)), (residual_protocol, ARMS)):
        for fold in range(5):
            for seed in phase_protocol['seeds']:
                for arm in arms:
                    result = read(Path(phase_protocol['output_root']) / f'fold{fold}/seed{seed}/{arm}/result.json')
                    tau = result['calibration']['threshold'] if arm == 'B_CAL' else result['calibration']['inherited_tau_secondary']
                    collect_predictions(predictions, arm, seed, fold, result['predictions'], tau)
    need(len(predictions) == 24 and all(set(rows) == set(panel) for rows in predictions.values()), 'ALL60_EXACT_OOF128')
    raw = {q: int(common[q]['axis'][common[q]['winner']]) for q in panel}
    for rows in predictions.values():
        for q, row in rows.items():
            need(row['fold'] == curator[q]['outer_fold'], 'CURATOR_FOLD_MISMATCH')
    predictions[('RAW', None, 'untrained')] = {q: {'selected': raw[q], 'fold': curator[q]['outer_fold'], 'threshold': None} for q in panel}
    raw_ok = np.array([gallery[raw[q]] == curator[q]['identity'] for q in panel])
    main, vectors, joined = [], {}, []
    for (model, seed, point), rows in predictions.items():
        ok = np.array([gallery[rows[q]['selected']] == curator[q]['identity'] for q in panel])
        vectors[(model, seed, point)] = ok
        main.append({'model': model, 'seed': '' if seed is None else seed, 'operating_point': point,
                     'queries': 128, 'correct': int(ok.sum()), 'accuracy': float(ok.mean()),
                     'rescues_vs_RAW': int((ok & ~raw_ok).sum()), 'breaks_vs_RAW': int((~ok & raw_ok).sum()),
                     'net_vs_RAW': int(ok.sum() - raw_ok.sum())})
        for q in panel:
            joined.append({'model': model, 'seed': seed, 'operating_point': point, 'query_id': q,
                           **rows[q], 'correct': gallery[rows[q]['selected']] == curator[q]['identity']})
    pairs = []
    identities = [curator[q]['identity'] for q in panel]
    clusters = sorted(set(identities))
    for arm in ARMS:
        for seed in root_protocol['seeds']:
            for point in ('zero', 'inherited_tau'):
                new, old = vectors[(arm, seed, point)], vectors[('B_CAL', seed, point)]
                diff = new.astype(int)-old.astype(int)
                cluster_sums = np.array([diff[np.array([x == ident for x in identities])].sum() for ident in clusters])
                cluster_counts = np.array([identities.count(ident) for ident in clusters])
                rng = np.random.default_rng(20260927)
                sample = rng.integers(0, len(clusters), size=(10000, len(clusters)))
                boot = cluster_sums[sample].sum(1)/cluster_counts[sample].sum(1)
                pairs.append({'model': arm, 'seed': seed, 'operating_point': point, 'baseline': 'same_F128_B_CAL',
                              'queries': 128, 'rescues': int((new & ~old).sum()), 'breaks': int((~new & old).sum()),
                              'net': int(diff.sum()), 'identity_cluster_gain_ci_low': float(np.quantile(boot, .025)),
                              'identity_cluster_gain_ci_high': float(np.quantile(boot, .975))})
    write_csv(root / 'metrics.csv', main)
    write_csv(root / 'paired_vs_B_CAL.csv', pairs)
    write_json(root / 'joined_predictions.json', {'panel': 'F128', 'records': joined})
    report = ['# F128 same-population B_CAL and frozen residual readers', '',
              'Development panel: execution ordinals 0–127, original grouped five folds, natural ColNomic C128. '
              'B_CAL is retrained on this same panel; each residual freezes its phase-matched B_CAL. '
              'Three seeds are separate repeats, not 384 independent images.', '',
              'Main operating point is fixed zero. Inherited B_CAL threshold is secondary. '
              'An inner-validation epoch must improve fixed-zero correctness to enable the residual; '
              'otherwise epoch zero preserves its baseline exactly. Best-CE diagnostics are not extra held models.', '',
              '| Model | Seed | Operating point | Correct /128 | RAW rescue / break |',
              '|---|---:|---|---:|---:|']
    report += [f"| {r['model']} | {r['seed']} | {r['operating_point']} | {r['correct']}/128 | {r['rescues_vs_RAW']}/{r['breaks_vs_RAW']} |" for r in main]
    report += ['', 'This is a development analysis, not full-H593 or external confirmation. '
               'All 60 saved models were independently replayed before held labels were opened. '
               'No head, epoch, threshold, arm or seed is selected from these held results.', '',
               '[Paired comparisons](paired_vs_B_CAL.csv) · [All predictions](joined_predictions.json) · [Validation](validation.json)', '']
    (root / 'REPORT_F128_RESULTS.md').write_text('\n'.join(report))
    receipt = {'status': 'REBUT_QR_QRR_F128_ALL60_INDEPENDENT_JOIN_PASS',
               'protocol': bind(root / 'protocol.json'), 'baseline_protocol': bind(root / 'baseline_protocol.json'),
               'residual_protocol': bind(root / 'residual_protocol.json'), 'fits_complete': 60, 'panel_queries': 128,
               'held_predictions_replayed': 128*3*4, 'held_scores_replayed': 128*3*4*128,
               'maximum_model_replay_error': maximum, 'held_labels_read_only_after_all60_checks': True,
               'seeds_reported_separately': True, 'identity_cluster_bootstrap_resamples': 10000,
               'external_GO': False, 'formal_H593_replacement': False, 'receipts': receipts,
               'sources': {'curator': bind(curator_path), 'verifier': bind(__file__)},
               'artifacts': {p.name: bind(p) for p in [root / 'metrics.csv', root / 'paired_vs_B_CAL.csv',
                                                    root / 'joined_predictions.json', root / 'REPORT_F128_RESULTS.md']}}
    write_json(root / 'validation.json', receipt)
    print(json.dumps({'status': receipt['status'], 'fits_complete': 60, 'panel_queries': 128}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['baseline', 'all'])
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--wall-seconds', type=float, default=480.)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root_protocol = read(args.protocol)
    verify_panel(root_protocol); verify_sources(root_protocol)
    root = args.protocol.resolve().parent
    lock = (root / 'join.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + args.wall_seconds
    bp_path = root / 'baseline_protocol.json'
    bp, br, bm = verify_phase(root, bp_path, False, deadline)
    seal_baselines(root, root_protocol, bp_path, bp, br, bm)
    if args.stage == 'baseline':
        print(json.dumps({'status': 'F128_BASELINE15_INDEPENDENT_PASS', 'held_labels_opened': False,
                          'residual_protocol': str(root / 'residual_protocol.json')}), flush=True)
        return
    rp, rr, rm = verify_phase(root, root / 'residual_protocol.json', True, deadline)
    write_final(root, root_protocol, bp, rp, br+rr, max(bm, rm))


if __name__ == '__main__':
    main()
