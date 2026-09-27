#!/usr/bin/env python3
"""Independent, fail-closed join for F71 frozen-baseline residual readers.

All 45 predictions must be sealed before outer labels are opened. Reuses the
v1 read-only numerical helpers, but independently checks v2 phase-specific heads,
normalizations, epoch-zero fallback, fixed-zero selection and inherited tau.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

RC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RC / 'programs'), str(RC / 'src')]
from join_rebut_qr_qrr_v1 import (bind, checked, decision, held_loss, keyed,
                                  read, scale_for, write_csv, write_json)
from rc_aslo_xf.rebut_qr_qrr_v1 import EvidenceBatch, RebutScorer


ARMS = ('QR', 'QR_VEC', 'QRR')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def choose_epoch(rows):
    """Independent fixed-zero improvement criterion; epoch zero is fallback."""
    need(rows and rows[0]['epoch'] == 0, 'EPOCH_ZERO_REQUIRED')
    need([r['epoch'] for r in rows] == list(range(len(rows))), 'EPOCHS_NOT_CONTIGUOUS')
    base_correct = rows[0]['fixed0_correct']
    eligible = [r for r in rows[1:] if r['fixed0_correct'] > base_correct]
    selected = min(eligible, key=lambda r: (r['validation_loss'], r['epoch'])) if eligible else rows[0]
    ce = min(rows, key=lambda r: (r['validation_loss'], r['epoch']))
    return selected['epoch'], ce['epoch'], base_correct


def baseline_binding(protocol, fold, seed):
    values = protocol['baseline_bindings']
    if str(fold) in values and str(seed) in values[str(fold)]:
        return values[str(fold)][str(seed)]
    for key in (f'fold{fold}/seed{seed}', f'{fold}:{seed}', f'fold{fold}_seed{seed}'):
        if key in values:
            return values[key]
    raise ValueError(f'BASELINE_BINDING_MISSING:{fold}:{seed}')


def train_truth(common, qid, gallery, roles):
    need(qid in roles, 'INNER_LABEL_NOT_IN_TRAIN')
    return [gallery[int(p)] == roles[qid]['identity'] for p in common[qid]['axis']]


def record_stats(rows, common, ids, gallery, roles):
    rows = keyed(rows)
    need(set(rows) == set(ids), 'INNER_RECORD_POPULATION')
    losses, correct = [], 0
    for q in ids:
        r, original = rows[q], common[q]
        need(r['axis'] == original['axis'] and r['winner'] == original['winner'], 'INNER_RECORD_AXIS')
        scores = r['logits']
        chosen = decision(scores, original['winner'], 0.0)
        truth = train_truth(common, q, gallery, roles)
        correct += int(truth[chosen])
        losses.append(held_loss(scores, truth, original['winner']))
    return {'validation_loss': float(np.mean(losses)), 'fixed0_correct': correct}


def head_equal(state, baseline):
    actual = {k: v for k, v in state.items() if k.startswith('head.')}
    expected = {k: v for k, v in baseline.items() if k.startswith('head.')}
    need(set(actual) == set(expected) == {'head.weight'}, 'FROZEN_HEAD_KEYS')
    need(all(torch.equal(actual[k], expected[k]) for k in expected), 'FROZEN_HEAD_CHANGED')


def replay(arm, saved, rows, common, evidence, ids, phase):
    model = RebutScorer(arm, 18, 72, 72, head_bias=False).double().eval()
    model.load_state_dict(saved['model'], strict=True)
    byid = keyed(rows)
    need(set(byid) == set(ids), 'REPLAY_POPULATION:' + phase)
    maximum = 0.0
    with torch.no_grad():
        for q in ids:
            r, original = byid[q], common[q]
            need(r['axis'] == original['axis'] and r['winner'] == original['winner'], 'REPLAY_AXIS:' + phase)
            x = original['X0'].double()[None] / saved['normalization']
            z = model(x, evidence[q] if arm != 'B_CAL' else None,
                      torch.tensor([original['winner']]))[0].numpy()
            stored = np.asarray(r['logits'], dtype=np.float64)
            need(stored.shape == (128,) and np.isfinite(stored).all(), 'REPLAY_FINITE:' + phase)
            need(stored[original['winner']] == 0.0, 'HOLD_NOT_ZERO:' + phase)
            error = float(np.max(np.abs(z - stored)))
            maximum = max(maximum, error)
            need(error < 2e-10, f'REPLAY_MISMATCH:{phase}:{q}:{error}')
            if 'logits_hex' in r:
                need(np.array_equal(stored, [float.fromhex(v) for v in r['logits_hex']]), 'LOGITS_HEX')
    return maximum


def collect_predictions(target, name, seed, fold, rows, tau):
    for r in rows:
        for point, threshold in (('zero', 0.0), ('inherited_tau', tau)):
            key = (name, seed, point)
            records = target.setdefault(key, {})
            q = r['query_id']
            need(q not in records, 'DUPLICATE_HELD_QUERY')
            pos = decision(r['logits'], r['winner'], threshold)
            records[q] = {'selected': int(r['axis'][pos]), 'fold': fold, 'threshold': threshold}


def independent_checks(protocol_path, threads=2):
    protocol = read(protocol_path)
    out = Path(protocol['output_root'])
    panel = set(protocol['panel_query_ids'])
    need(len(panel) == 71 and protocol['seeds'] == [0, 1, 2], 'FIXED_F71_SEEDS')
    need(tuple(protocol['arms']) == ARMS, 'EXPECTED_THREE_RESIDUAL_ARMS')
    jobs = [(fold, seed, arm, out / f'fold{fold}/seed{seed}/{arm}')
            for fold in range(5) for seed in protocol['seeds'] for arm in ARMS]
    missing = []
    for fold, seed, arm, folder in jobs:
        path = folder / 'result.json'
        if not path.exists() or read(path).get('status') != 'FROZEN_BASE_RESIDUAL_LABEL_FREE_COMPLETE':
            missing.append({'fold': fold, 'seed': seed, 'arm': arm})
    if missing:
        print(json.dumps({'status': 'WAITING_FOR_ALL45_RESULTS', 'complete': 45 - len(missing),
                          'missing': missing, 'held_labels_opened': False}), flush=True)
        raise SystemExit(75)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for dep in protocol['dependency_bindings'].values():
        checked(dep)
    source_validation = read(checked(protocol['source_validation']))
    need(source_validation['status'] == 'REBUT_QR_QRR_ALL60_F71_INDEPENDENT_JOIN_PASS', 'SOURCE_VALIDATION')
    need(source_validation['protocol'] == protocol['source_protocol'], 'SOURCE_PROTOCOL_VALIDATION')
    checked(protocol['source_protocol'])
    common = keyed(torch.load(checked(protocol['common_features']), map_location='cpu', weights_only=False)['records'])
    gallery = {int(r['physical_row']): r['identity'] for r in read(checked(protocol['gallery']))['records']}
    mb = protocol['evidence_manifest']
    manifest_path = checked(mb) if isinstance(mb, dict) else Path(mb)
    manifest = read(manifest_path)
    need(manifest['status'] == 'POOLED_F71_COMPLETE', 'EVIDENCE_NOT_COMPLETE')
    entries = keyed(manifest['records'])
    need(set(entries) == panel, 'EVIDENCE_PANEL')
    evidence, evidence_bindings = {}, {}
    for q, entry in entries.items():
        p = checked(entry['payload'])
        data = torch.load(p, map_location='cpu', weights_only=False)
        need(data['query_id'] == q and data['axis'] == common[q]['axis'] and
             data['winner'] == common[q]['winner'], 'EVIDENCE_AXIS')
        ev = EvidenceBatch(data['query'].double()[None], data['reference'].double()[None],
                           data['query_valid'].bool()[None], data['reference_valid'].bool()[None])
        ev.validate()
        need(ev.query.shape == ev.reference.shape == (1, 128, 64, 72), 'EVIDENCE_SHAPE')
        evidence[q], evidence_bindings[q] = ev, bind(p)
    module_path = RC / 'src/rc_aslo_xf/rebut_qr_qrr_v1.py'
    runner_path = RC / 'programs/run_rebut_qr_qrr_frozenbase_v2.py'
    protocol_bind = bind(protocol_path)
    module_sha, runner_sha = bind(module_path)['sha256'], bind(runner_path)['sha256']
    predictions, base_cache, receipts = {}, {}, []
    maximum = 0.0
    for index, (fold, seed, arm, folder) in enumerate(jobs):
        split = protocol['folds'][str(fold)]
        train, held = set(split['train_query_ids']), set(split['heldout_query_ids'])
        inner_fit, inner_val = set(split['inner_fit_query_ids']), set(split['inner_val_query_ids'])
        need(not train & held and train | held == panel, 'OUTER_SPLIT')
        need(not inner_fit & inner_val and inner_fit | inner_val == train, 'INNER_SPLIT')
        roles = keyed(read(checked(split['train_roles']))['records'])
        need(train <= set(roles) and not held & set(roles), 'ROLE_LABEL_BOUNDARY')
        for grouping in ('identity', 'component'):
            need(not {roles[q][grouping] for q in inner_fit} & {roles[q][grouping] for q in inner_val}, 'INNER_GROUP_LEAK')
        source = baseline_binding(protocol, fold, seed)
        for b in source.values():
            if isinstance(b, dict) and 'path' in b and 'sha256' in b:
                checked(b)
        if (fold, seed) not in base_cache:
            sr = read(checked(source['source_result']))
            sm = torch.load(checked(source['source_model']), map_location='cpu', weights_only=False)
            sc = torch.load(checked(source['source_checkpoint']), map_location='cpu', weights_only=False)
            need(sr['binding']['arm'] == 'B_CAL' and sr['binding']['fold'] == fold and sr['binding']['seed'] == seed, 'BASELINE_ID')
            need(sr['binding']['protocol'] == protocol['source_protocol'], 'BASELINE_PROTOCOL')
            need(sm['binding'] == sc['binding'] == sr['binding'], 'BASELINE_ARTIFACT_BINDING')
            need(sc['stage'] == 'done' and sr['outer_labels_opened'] is False, 'BASELINE_NOT_SEALED')
            need(sr['model_sha256'] == source['source_model']['sha256'], 'BASELINE_MODEL_SHA')
            for field in ('train_query_ids', 'inner_fit_query_ids', 'inner_val_query_ids', 'heldout_query_ids'):
                need(sr[field] == split[field], 'BASELINE_POPULATION')
            need(torch.equal(sm['normalization'], scale_for(common, split['train_query_ids'])), 'OUTER_BASE_SCALE')
            need(torch.equal(sc['inner_normalization'], scale_for(common, split['inner_fit_query_ids'])), 'INNER_BASE_SCALE')
            maximum = max(maximum, replay('B_CAL', sm, sr['predictions'], common, evidence,
                                          split['heldout_query_ids'], 'baseline_held'))
            collect_predictions(predictions, 'B_CAL', seed, fold, sr['predictions'], sr['calibration']['threshold'])
            base_cache[(fold, seed)] = (sr, sm, sc)
        sr, sm, sc = base_cache[(fold, seed)]
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
        print(json.dumps({'stage': 'independent_replay', 'completed': index + 1, 'total': 45,
                          'fold': fold, 'seed': seed, 'arm': arm, 'selected_epoch': selected}), flush=True)
    need(len(predictions) == 24 and all(set(r) == panel for r in predictions.values()), 'COMPLETE_HELD_PANEL')
    return protocol, common, gallery, predictions, receipts, maximum


def write_report(protocol_path, protocol, common, gallery, predictions, receipts, maximum):
    """Outer labels are first opened here, after every sealed prediction passed."""
    out = Path(protocol['output_root'])
    panel = sorted(protocol['panel_query_ids'])
    curator_path = RC / 'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json'
    curator = keyed(read(curator_path)['records'])
    raw = {q: int(common[q]['axis'][common[q]['winner']]) for q in panel}
    predictions[('RAW', None, 'untrained')] = {
        q: {'selected': raw[q], 'fold': curator[q]['outer_fold'], 'threshold': None} for q in panel}
    # v1 is a prior development recipe, not a same-training causal comparison.
    previous = RC / 'results/rc_rebut_qr_qrr_v1'
    pv = read(previous / 'validation.json')
    checked(pv['artifacts']['joined_predictions.json'])
    for r in read(previous / 'joined_predictions.json')['records']:
        if r['model'] not in ARMS:
            continue
        point = 'zero' if r['operating_point'] == 'zero' else 'v1_selected_tau'
        predictions.setdefault(('V1_' + r['model'], r['seed'], point), {})[r['query_id']] = {
            k: r[k] for k in ('selected', 'fold', 'threshold')}
    raw_ok = np.array([gallery[raw[q]] == curator[q]['identity'] for q in panel])
    vectors, main = {}, []
    for key, values in predictions.items():
        need(set(values) == set(panel), 'REPORT_PANEL')
        model, seed, point = key
        correct = np.array([gallery[values[q]['selected']] == curator[q]['identity'] for q in panel])
        changed = np.array([values[q]['selected'] != raw[q] for q in panel])
        vectors[key] = correct
        row = {'model': model, 'seed': '' if seed is None else seed, 'operating_point': point,
               'queries': 71, 'correct': int(correct.sum()), 'accuracy': float(correct.mean()),
               'rescues_vs_RAW': int((correct & ~raw_ok).sum()), 'breaks_vs_RAW': int((~correct & raw_ok).sum()),
               'net_vs_RAW': int(correct.sum() - raw_ok.sum()), 'switches': int(changed.sum()),
               'wrong_to_wrong_switches': int((changed & ~correct & ~raw_ok).sum()),
               'scope': ('previous_F71_development_recipe' if model.startswith('V1_') else
                         'untrained' if model == 'RAW' else 'F71_frozen_phase_matched_B_CAL')}
        for f in range(5):
            mask = np.array([curator[q]['outer_fold'] == f for q in panel])
            row[f'fold{f}_queries'], row[f'fold{f}_correct'] = int(mask.sum()), int(correct[mask].sum())
        main.append(row)
    matrices = {}
    for grouping in ('identity', 'component'):
        groups = sorted({curator[q][grouping] for q in panel})
        members = [np.array([curator[q][grouping] == g for q in panel]) for g in groups]
        draws = np.random.default_rng(20260927).integers(0, len(groups), (2000, len(groups)))
        matrices[grouping] = members, draws
    paired = []

    def compare(newkey, basekey, role):
        new, old = vectors[newkey], vectors[basekey]
        delta = new.astype(float) - old.astype(float)
        row = {'new_model': newkey[0], 'baseline': basekey[0], 'seed': newkey[1],
               'operating_point': newkey[2], 'baseline_operating_point': basekey[2],
               'comparison_role': role, 'queries': 71, 'rescues': int((new & ~old).sum()),
               'breaks': int((~new & old).sum()), 'net_correct': int(new.sum() - old.sum()),
               'accuracy_delta': float(delta.mean())}
        for grouping, (members, draws) in matrices.items():
            total = np.array([delta[m].sum() for m in members])
            counts = np.array([m.sum() for m in members])
            samples = total[draws].sum(1) / counts[draws].sum(1)
            eq = (total / counts)[draws].mean(1)
            lo, hi = np.quantile(samples, [.025, .975])
            elo, ehi = np.quantile(eq, [.025, .975])
            row.update({f'{grouping}_groups': len(members),
                        f'{grouping}_cluster_bootstrap_lo': float(lo), f'{grouping}_cluster_bootstrap_hi': float(hi),
                        f'{grouping}_equal_mean_delta': float((total / counts).mean()),
                        f'{grouping}_equal_bootstrap_lo': float(elo), f'{grouping}_equal_bootstrap_hi': float(ehi)})
        paired.append(row)

    for seed in protocol['seeds']:
        for point in ('zero', 'inherited_tau'):
            for arm in ARMS:
                key = arm, seed, point
                compare(key, ('B_CAL', seed, point), 'frozen_matched_baseline')
                compare(key, ('RAW', None, 'untrained'), 'raw_context')
                if arm == 'QRR':
                    compare(key, ('QR_VEC', seed, point), 'primary_early_vs_late_interaction')
                if point == 'zero':
                    compare(key, ('V1_' + arm, seed, 'zero'), 'previous_recipe_context_not_single_factor')
    write_csv(out / 'main_results.csv', main)
    write_csv(out / 'paired_comparisons.csv', paired)
    write_json(out / 'joined_predictions.json', {'panel': 'F71', 'records': [
        {'model': k[0], 'seed': k[1], 'operating_point': k[2], 'query_id': q,
         'display_id': curator[q]['original_query_id'], 'identity': curator[q]['identity'],
         'component': curator[q]['component'], **r,
         'correct': gallery[r['selected']] == curator[q]['identity']}
        for k, values in predictions.items() for q, r in sorted(values.items())]})
    fallback = sum(r['epoch0_fallback'] for r in receipts)
    report = ['# F71 frozen-baseline QR / QR-vec / QRR residuals', '',
              'All45 fits completed and were independently replayed before this join opened outer-held labels. '
              'This is the same previously used71-query grouped development panel, not full H593 or external confirmation.', '',
              'The common18 B_CAL head and its normalization are frozen. Inner selection uses the original inner-fit B_CAL; '
              'outer refit uses its separately fitted outer-TRAIN B_CAL. Reader inputs retain the predeclared pooled/projected F71 interface.', '',
              'Primary operating point: fixed threshold0. An epoch is eligible only if inner fixed0 correct count strictly exceeds '
              'the baseline; minimum validation CE chooses among eligible epochs (earliest tie). Otherwise the reader remains at epoch0. '
              'This TRAIN selection does not guarantee no breaks on held groups. Secondary thresholds are inherited from B_CAL, never refit.', '',
              f'Epoch-zero fallback occurred in{fallback}/45 fits.', '',
              '| Model | Seed | Operating point | Correct /71 | RAW rescue / break |',
              '|---|---:|---|---:|---:|']
    for r in main:
        if not r['model'].startswith('V1_'):
            report.append(f"| {r['model']} | {r['seed']} | {r['operating_point']} | {r['correct']}/71 | {r['rescues_vs_RAW']}/{r['breaks_vs_RAW']} |")
    report += ['', 'QRR versus QR_VEC is the structural contrast; each reader versus phase-matched B_CAL tests incremental utility. '
               'Seeds are reported separately, never as213 independent observations. Previous v1 results are contextual because the training recipe changed.', '',
               'Identity-cluster paired bootstrap uses2000 resamples, with component sensitivity and equal-group estimates. '
               'These intervals condition on the fixed predictions and do not account for repeated development or overlapping training samples. '
               'Held reference identities may appear as TRAIN negatives in this fixed-gallery protocol.', '',
               'A negative result limits this pooled input and training recipe; it does not establish M sufficiency, uniqueness, '
               'or absence of information from the full RoMa representation.', '',
               '- [Main results](main_results.csv)', '- [Paired comparisons](paired_comparisons.csv)',
               '- [Joined predictions](joined_predictions.json)', '- [Independent validation](validation.json)', '']
    (out / 'report.md').write_text('\n'.join(report))
    receipt = {'status': 'REBUT_QR_QRR_FROZENBASE_V2_ALL45_F71_INDEPENDENT_JOIN_PASS',
               'protocol': bind(protocol_path), 'fits_complete': 45, 'panel_queries': 71,
               'seeds_reported_separately': protocol['seeds'], 'epoch0_fallback_fits': fallback,
               'new_model_held_predictions_replayed': 71 * 3 * 3, 'new_held_scores_replayed': 71 * 3 * 3 * 128,
               'maximum_model_replay_error': maximum, 'phase_specific_frozen_heads_exact': True,
               'inner_all_epoch_logits_recounted': True, 'fixed0_selection_independently_recomputed': True,
               'inherited_tau_unchanged': True, 'epoch0_fallback_exact': True,
               'held_labels_read_only_after_all45_checks': True, 'new_training_updates': 0,
               'external_GO': False, 'formal_H593_replacement': False,
               'sources': {'curator': bind(curator_path), 'v1_validation': bind(previous / 'validation.json'),
                           'v1_join_helpers': bind(RC / 'programs/join_rebut_qr_qrr_v1.py')},
               'fits': receipts, 'verifier': bind(__file__), 'issues': [],
               'artifacts': {p: bind(out / p) for p in ('main_results.csv', 'paired_comparisons.csv',
                                                       'joined_predictions.json', 'report.md')}}
    write_json(out / 'validation.json', receipt)
    print(json.dumps({'status': receipt['status'], 'fits': 45, 'queries': 71,
                      'epoch0_fallback_fits': fallback, 'maximum_replay_error': maximum}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, default=RC / 'results/rc_rebut_qr_qrr_frozenbase_v2/protocol.json')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    protocol, common, gallery, predictions, receipts, maximum = independent_checks(args.protocol, args.threads)
    write_report(args.protocol, protocol, common, gallery, predictions, receipts, maximum)


if __name__ == '__main__':
    main()
