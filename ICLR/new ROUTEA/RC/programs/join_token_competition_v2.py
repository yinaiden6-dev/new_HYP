#!/usr/bin/env python3
"""Resume-aware independent V2 replay; held labels open only after all 15 fits.

Stage-specific frozen heads, RMS, epoch0 selection, full-C128 CE, early stopping,
refit budget and logits are checked independently using existing audited math.
Per-query replay receipts additionally bound long native-token forward passes.
"""
import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

RC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RC / 'src'), str(RC / 'programs')]
from join_rebut_qr_qrr_v1 import bind, decision, keyed, read, write_csv, write_json
from join_rebut_qr_qrr_frozenbase_v2 import (need, choose_epoch, baseline_binding,
    record_stats, head_equal, collect_predictions)
from join_rebut_qr_qrr_f128_v1 import common_artifacts
from run_rebut_qr_qrr_v1 import atomic_torch, atomic_json, checked
from token_competition_data_v2 import TokenInputs, ARMS, verify_panel, verify_sources, verify_evidence_gate
from rc_aslo_xf.token_competition_v2 import new_model
from rc_aslo_xf.rebut_qr_qrr_v1 import RebutScorer

FIELDS = ('train_query_ids', 'inner_fit_query_ids', 'inner_val_query_ids', 'heldout_query_ids')
REPLAY_CONTEXT = {}


def state_digest(saved):
    h = hashlib.sha256()
    for name, tensor in sorted({**saved['model'], '_normalization': saved['normalization']}.items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def replay(arm, saved, rows, common, evidence, ids, phase):
    """Numerical replay without labels, optionally retaining full held traces."""
    if arm == 'B_CAL':
        model = RebutScorer('B_CAL', 18, 72, 72, head_bias=False).double().eval()
    else:
        model, _ = new_model(arm, 0, REPLAY_CONTEXT['optimizer'])
        model.eval()
    model.load_state_dict(saved['model'], strict=True)
    byid = keyed(rows)
    need(set(byid) == set(ids), 'REPLAY_POPULATION:' + phase)
    fingerprint = state_digest(saved)
    cache = REPLAY_CONTEXT['folder'] / 'independent_replay' / phase
    cache.mkdir(parents=True, exist_ok=True)
    maximum = 0.0
    with torch.no_grad():
        for qid in ids:
            row, original = byid[qid], common[qid]
            need(row['axis'] == original['axis'] and row['winner'] == original['winner'], 'REPLAY_AXIS')
            inputs = {'model_state_sha256': fingerprint, 'query_id': qid, 'arm': arm,
                      'phase': phase, 'scores': row['logits'], 'axis': row['axis'],
                      'winner': row['winner'], 'sources': REPLAY_CONTEXT['sources']}
            receipt_path = cache / (qid + '.json')
            prior = read(receipt_path) if receipt_path.exists() else None
            if prior and prior.get('status') == 'TOKEN_QUERY_REPLAY_PASS' and prior.get('inputs') == inputs:
                if prior.get('trace'):
                    checked(prior['trace'])
                maximum = max(maximum, prior['maximum_error'])
                continue
            if time.monotonic() >= REPLAY_CONTEXT['deadline'] - 10:
                print(json.dumps({'status': 'TOKEN_REPLAY_RESUMABLE', 'phase': phase,
                                  'query_id': qid, 'held_labels_opened': False}), flush=True)
                raise SystemExit(75)
            x = original['X0'].double()[None] / saved['normalization']
            anchor = torch.tensor([original['winner']])
            if phase == 'outer_refit':
                logits, trace = model.forward_with_trace(x, evidence[qid], anchor)
            else:
                logits = model(x, evidence[qid] if arm != 'B_CAL' else None, anchor)
                trace = None
            z = logits[0].numpy()
            stored = np.asarray(row['logits'], dtype=np.float64)
            need(stored.shape == (128,) and np.isfinite(stored).all(), 'REPLAY_FINITE')
            need(stored[original['winner']] == 0.0, 'HOLD_NOT_ZERO')
            error = float(np.max(np.abs(z - stored)))
            need(error < 2e-10, f'REPLAY_MISMATCH:{phase}:{qid}:{error}')
            if 'logits_hex' in row:
                need(np.array_equal(stored, [float.fromhex(v) for v in row['logits_hex']]), 'LOGITS_HEX')
            trace_binding = None
            if trace is not None:
                trace_path = cache / (qid + '.pt')
                atomic_torch(trace_path, {'query_id': qid, 'axis': row['axis'], 'winner': row['winner'],
                    'inputs': inputs, 'logits': logits.detach().cpu(),
                    'trace': {key: value.detach().cpu() if torch.is_tensor(value) else value
                              for key, value in trace.items()}})
                trace_binding = bind(trace_path)
            atomic_json(receipt_path, {'status': 'TOKEN_QUERY_REPLAY_PASS', 'inputs': inputs,
                        'maximum_error': error, 'held_labels_opened': False, 'trace': trace_binding})
            maximum = max(maximum, error)
    return maximum


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
    manifest_path = checked(protocol['evidence_manifest'])
    module_sha = bind(RC / 'src/rc_aslo_xf/token_competition_v2.py')['sha256']
    runner_sha = bind(RC / 'programs/run_token_competition_v2.py')['sha256']
    maximum, receipts, predictions = 0.0, [], {}
    result = read(folder / 'result.json')
    saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    inner = read(folder / 'inner_validation.json')
    main_saved = torch.load(folder / 'main_inner_model.pt', map_location='cpu', weights_only=False)
    best_ce_saved = torch.load(folder / 'inner_best_ce_model.pt', map_location='cpu', weights_only=False)
    expected = {'protocol': protocol_bind, 'runner_sha256': runner_sha, 'module_sha256': module_sha,
                'fold': fold, 'seed': seed, 'arm': arm, 'evidence_manifest': bind(manifest_path),
                'data_adapter_sha256': bind(RC / 'programs/token_competition_data_v2.py')['sha256'],
                'source_runner_sha256': bind(RC / 'programs/run_rebut_qr_qrr_frozenbase_v2.py')['sha256']}
    # The phase-specific source binding may additionally be embedded in binding.
    for k, v in expected.items():
        need(result['binding'].get(k) == v, 'RESULT_BINDING:' + k)
    for value in (saved, checkpoint, inner, main_saved, best_ce_saved):
        need(value['binding'] == result['binding'], 'ARTIFACT_BINDING')
    for binding in result['artifacts'].values():
        checked(binding)
    need(result['binding']['baseline_bindings'] == source, 'EMBEDDED_BASELINE_BINDING')
    need(result['binding']['v1_runner_sha256'] == bind(RC / 'programs/run_rebut_qr_qrr_v1.py')['sha256'], 'SOURCE_RUNNER_BINDING')
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
    shape_model, _ = new_model(arm, seed, protocol['optimizer'])
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



def wait_for_results(protocol):
    output = Path(protocol['output_root'])
    missing = []
    for fold in range(5):
        for arm in ARMS:
            path = output / f'fold{fold}/seed0/{arm}/result.json'
            if not path.exists() or read(path).get('status') != 'FROZEN_BASE_RESIDUAL_LABEL_FREE_COMPLETE':
                missing.append({'fold': fold, 'seed': 0, 'arm': arm})
    if missing:
        print(json.dumps({'status': 'TOKEN_WAITING_FOR_ALL15_RESULTS', 'complete': 15-len(missing),
                          'total': 15, 'missing': missing, 'held_labels_opened': False}), flush=True)
        raise SystemExit(75)


def verify_all(protocol_path, deadline):
    protocol = read(protocol_path)
    verify_panel(protocol)
    verify_sources(protocol)
    gate = verify_evidence_gate(protocol)
    wait_for_results(protocol)
    source = read(checked(protocol['source_validation']))
    need(source['status'] == 'F128_BASELINE15_INDEPENDENT_PASS' and
         source['protocol'] == protocol['source_protocol'], 'SAME_F128_BASELINE_SEAL')
    checked(protocol['source_protocol'])
    root, output = protocol_path.resolve().parent, Path(protocol['output_root'])
    receipt_root = root / 'independent_receipts'
    receipt_root.mkdir(parents=True, exist_ok=True)
    manifest = read(checked(protocol['evidence_manifest']))
    evidence_bindings = {r['query_id']: {'path': str(checked(r['payload'])), 'sha256': r['payload']['sha256']}
                         for r in manifest['records']}
    sources = {'protocol': bind(protocol_path), 'verifier': bind(__file__),
               'module': bind(RC / 'src/rc_aslo_xf/token_competition_v2.py'),
               'data': bind(RC / 'programs/token_competition_data_v2.py'),
               'runner': bind(RC / 'programs/run_token_competition_v2.py'),
               'math_v1': bind(RC / 'programs/join_rebut_qr_qrr_v1.py'),
               'math_v2': bind(RC / 'programs/join_rebut_qr_qrr_frozenbase_v2.py'),
               'f128_join': bind(RC / 'programs/join_rebut_qr_qrr_f128_v1.py'),
               'common_features': protocol['common_features'], 'gallery': protocol['gallery'],
               'evidence_manifest': bind(checked(protocol['evidence_manifest'])),
               'evidence_validation': gate}
    receipts, maximum = [], 0.0
    for fold in range(5):
        data = None
        for arm in ARMS:
            folder = output / f'fold{fold}/seed0/{arm}'
            receipt_path = receipt_root / f'fold{fold}_seed0_{arm}.json'
            baseline = baseline_binding(protocol, fold, 0)
            for value in baseline.values():
                checked(value)
            roles = protocol['folds'][str(fold)]['train_roles']
            checked(roles)
            inputs = {'sources': sources, 'fold': fold, 'seed': 0, 'arm': arm,
                      'artifacts': common_artifacts(folder, True), 'evidence': evidence_bindings,
                      'train_roles': roles, 'baseline': baseline, 'baseline_validation': protocol['source_validation'],
                      'baseline_inner': bind(Path(baseline['source_result']['path']).parent / 'inner_validation.json')}
            prior = read(receipt_path) if receipt_path.exists() else None
            cached = prior is not None and prior.get('status') == 'TOKEN_FIT_INDEPENDENT_PASS' and prior.get('inputs') == inputs
            if cached:
                for value in prior['trace_artifacts'].values():
                    checked(value)
                receipt = prior
            else:
                if time.monotonic() >= deadline - 10:
                    print(json.dumps({'status': 'TOKEN_REPLAY_RESUMABLE', 'fits_verified': len(receipts),
                                      'held_labels_opened': False}), flush=True)
                    raise SystemExit(75)
                if data is None:
                    data = TokenInputs(protocol, protocol['folds'][str(fold)], arm)
                REPLAY_CONTEXT.clear()
                REPLAY_CONTEXT.update({'folder': folder, 'deadline': deadline,
                    'optimizer': protocol['optimizer'], 'sources': {**sources, 'evidence': evidence_bindings}})
                details = validate_residual(protocol_path, protocol, fold, 0, arm, folder, data)
                traces = {q: bind(folder / 'independent_replay/outer_refit' / (q + '.pt'))
                          for q in protocol['folds'][str(fold)]['heldout_query_ids']}
                receipt = {'status': 'TOKEN_FIT_INDEPENDENT_PASS', 'inputs': inputs,
                           'held_labels_opened': False, 'trace_artifacts': traces, **details}
                atomic_json(receipt_path, receipt)
            maximum = max(maximum, receipt['max_replay_error'])
            receipts.append({'receipt': bind(receipt_path), 'fold': fold, 'seed': 0, 'arm': arm,
                'selected_epoch': receipt['selected_epoch'], 'max_replay_error': receipt['max_replay_error']})
            print(json.dumps({'stage': 'TOKEN_independent_replay', 'complete': len(receipts), 'total': 15,
                              'fold': fold, 'seed': 0, 'arm': arm, 'cached': cached}), flush=True)
        del data
    need(verify_evidence_gate(protocol) == gate, 'EVIDENCE_CHANGED_DURING_REPLAY')
    for value in evidence_bindings.values():
        checked(value)
    return protocol, receipts, maximum


def write_final(protocol_path, protocol, receipts, maximum):
    # Sole held-label boundary: all 15 saved residual fits have passed independent
    # replay and the old F128 baseline lineage was sealed before this function.
    need(len(receipts) == 15, 'ALL15_REQUIRED_BEFORE_HELD_LABELS')
    root, output = protocol_path.resolve().parent, Path(protocol['output_root'])
    curator_path = RC / 'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json'
    curator = keyed(read(curator_path)['records'])
    common = keyed(torch.load(checked(protocol['common_features']), map_location='cpu', weights_only=False)['records'])
    gallery = {int(r['physical_row']): r['identity'] for r in read(checked(protocol['gallery']))['records']}
    panel = sorted(protocol['panel_query_ids'])
    need(set(panel) <= set(curator), 'CURATOR_PANEL')
    predictions, candidate_rows, traces = {}, [], []
    for fold in range(5):
        baseline = read(checked(baseline_binding(protocol, fold, 0)['source_result']))
        collect_predictions(predictions, 'B_CAL', 0, fold, baseline['predictions'], baseline['calibration']['threshold'])
        for arm in ARMS:
            folder = output / f'fold{fold}/seed0/{arm}'
            result = read(folder / 'result.json')
            collect_predictions(predictions, arm, 0, fold, result['predictions'], result['calibration']['inherited_tau_secondary'])
            for row in result['predictions']:
                qid = row['query_id']
                trace_path = folder / 'independent_replay/outer_refit' / (qid + '.pt')
                saved = torch.load(trace_path, map_location='cpu', weights_only=False)
                trace = saved['trace']
                need(saved['axis'] == row['axis'] and saved['winner'] == row['winner'], 'TRACE_AXIS')
                need(torch.equal(saved['logits'][0], torch.tensor(row['logits'], dtype=torch.float64)), 'TRACE_LOGITS')
                base, residual = trace['base_logits'][0], trace['residual'][0]
                D = trace['D'][0]
                need(base.shape == residual.shape == D.shape == (128,), 'TRACE_SCORE_DIMENSIONS')
                for pos, physical in enumerate(row['axis']):
                    edges = []
                    for j, rival in enumerate(trace['rivals'][0, pos].tolist()):
                        if bool(trace['rival_valid'][0, pos, j]):
                            edges.append({'rival_position': rival, 'rival_physical_row': row['axis'][rival],
                                          'pair_difference': float(trace['pair_differences'][0, pos, j])})
                    candidate_rows.append({'model': arm, 'seed': 0, 'fold': fold, 'query_id': qid,
                        'position': pos, 'physical_row': physical, 'raw_anchor': pos == row['winner'],
                        'zero_base': float(base[pos]), 'D': float(D[pos]), 'residual': float(residual[pos]),
                        'logit': row['logits'][pos], 'logit_hex': row['logits_hex'][pos], 'edges': edges})
                traces.append({'model': arm, 'fold': fold, 'seed': 0, 'query_id': qid, 'trace': bind(trace_path)})
    need(len(predictions) == 8 and all(set(rows) == set(panel) for rows in predictions.values()), 'EXACT_OOF128_COVERAGE')
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
            'net_vs_RAW': int(ok.sum()-raw_ok.sum())})
        for q in panel:
            joined.append({'model': model, 'seed': seed, 'operating_point': point, 'query_id': q,
                           **rows[q], 'correct': gallery[rows[q]['selected']] == curator[q]['identity']})
    pairs = []
    identities = [curator[q]['identity'] for q in panel]
    clusters = sorted(set(identities))
    for arm, baseline_model in [(a, 'B_CAL') for a in ARMS] + [('TOKEN_QRR_MULTI', 'TOKEN_QRR_ANCHOR')]:
        for point in ('zero', 'inherited_tau'):
            new, old = vectors[(arm, 0, point)], vectors[(baseline_model, 0, point)]
            diff = new.astype(int)-old.astype(int)
            cluster_sums = np.array([diff[np.array([x == ident for x in identities])].sum() for ident in clusters])
            cluster_counts = np.array([identities.count(ident) for ident in clusters])
            rng = np.random.default_rng(20260927)
            sampled = rng.integers(0, len(clusters), size=(10000, len(clusters)))
            boot = cluster_sums[sampled].sum(1)/cluster_counts[sampled].sum(1)
            changed = sum(predictions[(arm, 0, point)][q]['selected'] != predictions[(baseline_model, 0, point)][q]['selected'] for q in panel)
            pairs.append({'model': arm, 'seed': 0, 'operating_point': point, 'baseline': 'same_F128_' + baseline_model + '_seed0',
                'queries': 128, 'rescues': int((new & ~old).sum()), 'breaks': int((~new & old).sum()),
                'net': int(diff.sum()), 'baseline_correct': int(old.sum()),
                'baseline_correct_loss_rate': float((~new & old).sum()/old.sum()) if old.any() else None,
                'changed_decisions': changed,
                'identity_cluster_gain_ci_low': float(np.quantile(boot, .025)),
                'identity_cluster_gain_ci_high': float(np.quantile(boot, .975))})
    write_csv(root / 'metrics.csv', main)
    write_csv(root / 'paired_vs_B_CAL.csv', [r for r in pairs if r['baseline'] == 'same_F128_B_CAL_seed0'])
    write_csv(root / 'paired_structural.csv', [r for r in pairs if r['baseline'] == 'same_F128_TOKEN_QRR_ANCHOR_seed0'])
    write_json(root / 'joined_predictions.json', {'panel': 'F128 development', 'seed': 0, 'records': joined})
    write_json(root / 'candidate_predictions.json', {'panel': 'F128 development', 'seed': 0,
        'candidate_source': 'frozen natural ColNomic C128', 'records': candidate_rows})
    write_json(root / 'trace_manifest.json', {'records': traces})
    report = ['# Native token competition V2: F128 development, seed0', '',
        'Population: original execution ordinals 0–127, original grouped five folds, frozen natural ColNomic C128. '
        'Each residual freezes the existing same-F128, seed0 B_CAL head and normalization separately for inner fitting and outer refit. '
        'Native query tokens remain unpooled. Full C128 hit/miss loss and the original AdamW budget are preserved.', '',
        'The main operating point is fixed zero; inherited B_CAL thresholds are secondary. '
        'Inner-validation correctness must strictly improve over frozen B_CAL to enable an epoch; '
        'otherwise epoch zero exactly preserves B_CAL. All 15 residual fits passed independent replay before held labels were opened.', '',
        '| Model | Seed | Operating point | Correct /128 | RAW rescue / break |',
        '|---|---:|---|---:|---:|']
    report += [f"| {r['model']} | {r['seed']} | {r['operating_point']} | {r['correct']}/128 | {r['rescues_vs_RAW']}/{r['breaks_vs_RAW']} |" for r in main]
    report += ['', 'This is a seed0 development analysis. It does not establish full-H593 performance or external confirmation. '
        'Held results do not select an arm, epoch, threshold or seed.', '',
        '[Paired comparisons](paired_vs_B_CAL.csv) · [MULTI versus ANCHOR](paired_structural.csv) · '
        '[Candidate scores and edges](candidate_predictions.json) · '
        '[Trace tensors](trace_manifest.json) · [Validation](validation.json)', '']
    (root / 'REPORT_TOKEN_COMPETITION_V2.md').write_text('\n'.join(report))
    files = ['metrics.csv', 'paired_vs_B_CAL.csv', 'paired_structural.csv', 'joined_predictions.json', 'candidate_predictions.json',
             'trace_manifest.json', 'REPORT_TOKEN_COMPETITION_V2.md']
    validation = {'status': 'TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS',
        'protocol': bind(protocol_path), 'fits_complete': 15, 'panel_queries': 128, 'seed': 0,
        'held_predictions_replayed': 128*3, 'held_scores_replayed': 128*3*128,
        'maximum_model_replay_error': maximum, 'held_labels_read_only_after_all15_checks': True,
        'identity_cluster_bootstrap_resamples': 10000, 'development_analysis': True,
        'external_GO': False, 'formal_H593_replacement': False, 'receipts': receipts,
        'sources': {'curator': bind(curator_path), 'verifier': bind(__file__),
                    'baseline_validation': protocol['source_validation']},
        'artifacts': {name: bind(root / name) for name in files}}
    atomic_json(root / 'validation.json', validation)
    print(json.dumps({'status': validation['status'], 'fits_complete': 15, 'panel_queries': 128}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True, type=Path)
    ap.add_argument('--threads', default=2, type=int)
    ap.add_argument('--wall-seconds', default=480., type=float)
    args = ap.parse_args()
    if args.threads < 1 or args.wall_seconds <= 0:
        raise ValueError('positive thread count and wall budget required')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    lock = (args.protocol.resolve().parent / 'join.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, receipts, maximum = verify_all(args.protocol, time.monotonic()+args.wall_seconds)
    write_final(args.protocol, protocol, receipts, maximum)


if __name__ == '__main__':
    main()
