#!/usr/bin/env python3
"""Fail-closed, independent F71 collection after all60 predeclared fits finish.

Replays stored models and calibrations before opening outer held labels. Historical
heads are projected to the same71 only; their larger training population is kept
explicit. Three seeds are reported separately, never as213 independent queries.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

RC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RC / 'src'))
from rc_aslo_xf.rebut_qr_qrr_v1 import EvidenceBatch, RebutScorer


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest()}


def checked(value):
    assert bind(value['path'])['sha256'] == value['sha256'], value['path']
    return Path(value['path'])


def keyed(rows):
    answer = {r['query_id']: r for r in rows}
    assert len(answer) == len(rows)
    return answer


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.join.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def decision(scores, anchor, tau):
    assert len(scores) == 128 and scores[anchor] == 0 and all(math.isfinite(v) for v in scores)
    contenders = [j for j in range(128) if j != anchor]
    best = max(contenders, key=lambda j: scores[j])
    ties = sum(scores[j] == scores[best] for j in contenders)
    return best if scores[best] > tau and ties == 1 else anchor


def scale_for(common, ids):
    arrays = [common[q]['X0'][torch.arange(128) != common[q]['winner']] for q in ids]
    rms = torch.cat(arrays).square().mean(0).sqrt()
    return torch.where(rms > 1e-12, rms, torch.ones_like(rms))


def held_loss(scores, truth, anchor):
    z = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    if truth.any():
        top = z.max()
        correct = z[truth]
        peak = correct.max()
        return float(top + np.log(np.exp(z - top).sum()) - peak - np.log(np.exp(correct - peak).sum()))
    challengers = np.delete(z, anchor)
    # Match torch softplus's declared default threshold20 exactly.
    return float(np.where(challengers > 20, challengers, np.logaddexp(0, challengers)).mean())


def calibrate(records, common, identities, roles):
    maxima = sorted(set(max(v for j, v in enumerate(r['logits']) if j != r['winner']) for r in records))
    taus = [math.nextafter(maxima[0], -math.inf)] + maxima
    best = None
    for tau in taus:
        rescued = broken = 0
        for r in records:
            q, axis, anchor = r['query_id'], r['axis'], r['winner']
            chosen = decision(r['logits'], anchor, tau)
            identity = roles[q]['identity']
            raw = identities[axis[anchor]] == identity
            new = identities[axis[chosen]] == identity
            rescued += int(new and not raw)
            broken += int(raw and not new)
        key = (rescued - broken, tau)
        if best is None or key > best[0]:
            best = (key, tau, rescued, broken)
    return best[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, default=RC / 'results/rc_rebut_qr_qrr_v1/protocol.json')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    protocol = read(args.protocol)
    out = Path(protocol['output_root'])
    bindings = bind(args.protocol)
    panel = set(protocol['panel_query_ids'])
    assert len(panel) == 71
    jobs = [(fold, seed, arm, out / f'fold{fold}/seed{seed}/{arm}')
            for fold in range(5) for seed in protocol['seeds'] for arm in protocol['arms']]
    assert len(jobs) == 60
    missing = []
    for fold, seed, arm, folder in jobs:
        path = folder / 'result.json'
        if not path.exists():
            missing.append({'fold': fold, 'seed': seed, 'arm': arm, 'reason': 'RESULT_MISSING'})
        elif read(path).get('status') != 'TRAINING_AND_LABEL_FREE_PREDICTIONS_COMPLETE':
            missing.append({'fold': fold, 'seed': seed, 'arm': arm, 'reason': 'NOT_COMPLETE'})
    if missing:
        print(json.dumps({'status': 'WAITING_FOR_ALL60_RESULTS', 'complete': 60 - len(missing), 'missing': missing,
                          'held_labels_opened': False, 'scientific_metrics_written': False}), flush=True)
        raise SystemExit(75)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    common = keyed(torch.load(checked(protocol['common_features']), map_location='cpu', weights_only=False)['records'])
    assert panel <= set(common)
    gallery = {r['physical_row']: r['identity'] for r in read(checked(protocol['gallery']))['records']}
    manifest_path = Path(protocol['evidence_manifest'])
    manifest = read(manifest_path)
    assert manifest['status'] == 'POOLED_F71_COMPLETE'
    entries = keyed(manifest['records'])
    assert set(entries) == panel
    manifest_binding = bind(manifest_path)
    module_path = RC / 'src/rc_aslo_xf/rebut_qr_qrr_v1.py'
    runner_path = RC / 'programs/run_rebut_qr_qrr_v1.py'
    module_sha = bind(module_path)['sha256']
    runner_sha = bind(runner_path)['sha256']
    evidence = {}
    evidence_bindings = {}
    for q, entry in entries.items():
        binding = entry.get('evidence', entry.get('payload', entry))
        path = checked(binding)
        value = torch.load(path, map_location='cpu', weights_only=False)
        assert value['query_id'] == q and value['axis'] == common[q]['axis'] and value['winner'] == common[q]['winner']
        ev = EvidenceBatch(value['query'].to(torch.float64)[None], value['reference'].to(torch.float64)[None],
                           value['query_valid'].bool()[None], value['reference_valid'].bool()[None])
        ev.validate()
        assert ev.query.shape == ev.reference.shape == (1, 128, 64, 72)
        evidence[q] = ev
        evidence_bindings[q] = bind(path)
    # No curator/outer-held identity file has been opened before every result,
    # candidate axis, checkpoint, model score, inner epoch and tau is checked.
    predictions = {}
    fit_receipts = []
    replay_max = 0.0
    for job_index, (fold, seed, arm, folder) in enumerate(jobs):
        result = read(folder / 'result.json')
        split = protocol['folds'][str(fold)]
        expected_binding = {'protocol': bindings, 'runner_sha256': runner_sha, 'module_sha256': module_sha,
                            'fold': fold, 'seed': seed, 'arm': arm}
        if arm != 'B_CAL':
            expected_binding['evidence_manifest'] = manifest_binding
        assert result['binding'] == expected_binding
        assert result['outer_labels_opened'] is False
        assert result['evidence_bindings'] == ({} if arm == 'B_CAL' else evidence_bindings)
        for field in ['train_query_ids', 'inner_fit_query_ids', 'inner_val_query_ids', 'heldout_query_ids']:
            assert result[field] == split[field]
        train = set(split['train_query_ids'])
        held = set(split['heldout_query_ids'])
        assert not train & held and train | held == panel
        roles = keyed(read(checked(split['train_roles']))['records'])
        assert train <= set(roles) and not held & set(roles)
        assert bind(folder / 'model.pt')['sha256'] == result['model_sha256']
        saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
        assert saved['binding'] == checkpoint['binding'] == expected_binding
        assert checkpoint['stage'] == 'done'
        assert 'head.bias' not in saved['model']
        scale = scale_for(common, split['train_query_ids'])
        assert torch.equal(scale, saved['normalization'])
        assert np.array_equal(scale.numpy(), result['normalization_rms'])
        model = RebutScorer(arm, 18, 72, 72, head_bias=False).to(torch.float64).eval()
        model.load_state_dict(saved['model'], strict=True)
        assert result['trainable_parameters'] == sum(p.numel() for p in model.parameters() if p.requires_grad)
        inner = read(folder / 'inner_validation.json')
        assert inner['binding'] == expected_binding
        assert inner['best_epoch'] == result['best_epoch'] == checkpoint['best_epoch']
        assert inner['calibration'] == result['calibration'] == checkpoint['calibration']
        history = result['history']
        inner_history = [r for r in history if r['stage'] == 'inner']
        refit_history = [r for r in history if r['stage'] == 'refit']
        assert result['best_epoch'] == min(inner_history, key=lambda x: x['validation_loss'])['epoch']
        assert len(refit_history) == result['best_epoch']
        assert [r['epoch'] for r in refit_history] == list(range(1, result['best_epoch'] + 1))
        inner_scale = scale_for(common, split['inner_fit_query_ids'])
        assert np.array_equal(inner_scale.numpy(), inner['normalization_rms'])
        selected_tau = result['calibration']['threshold']
        assert math.isfinite(selected_tau) and selected_tau.hex() == result['calibration']['threshold_hex']
        held_rows = keyed(result['predictions'])
        validation_rows = keyed(inner['records'])
        assert set(held_rows) == held and set(validation_rows) == set(split['inner_val_query_ids'])
        for phase, rows, normalized in [('held', held_rows, scale), ('inner_validation', validation_rows, inner_scale)]:
            if phase == 'inner_validation':
                model.load_state_dict(checkpoint['best_model'], strict=True)
            with torch.no_grad():
                for q, row in rows.items():
                    original = common[q]
                    axis = original['axis']
                    anchor = original['winner']
                    assert len(axis) == len(set(axis)) == 128
                    assert row['axis'] == axis and row['winner'] == anchor
                    x = original['X0'][None] / normalized
                    z = model(x, None if arm == 'B_CAL' else evidence[q], torch.tensor([anchor]))[0].numpy()
                    stored = np.asarray(row['logits'], dtype=np.float64)
                    assert stored.shape == (128,) and np.isfinite(stored).all() and stored[anchor] == 0
                    error = float(np.max(np.abs(z - stored)))
                    replay_max = max(replay_max, error)
                    assert error < 2e-10, (fold, seed, arm, q, phase, error)
                    if phase == 'held':
                        assert np.array_equal(stored, [float.fromhex(h) for h in row['logits_hex']])
                        pos = decision(row['logits'], anchor, selected_tau)
                        zero = decision(row['logits'], anchor, 0.0)
                        assert row['selected_position'] == pos and row['selected_physical_row'] == axis[pos]
                        assert row['selected_fixed_zero_position'] == zero and row['selected_fixed_zero_physical_row'] == axis[zero]
                        assert row['action'] == ('HOLD' if pos == anchor else 'SWITCH')
                        for operating_point, physical in [('selected_tau', axis[pos]), ('zero', axis[zero])]:
                            key = (arm, seed, operating_point)
                            records = predictions.setdefault(key, {})
                            assert q not in records
                            records[q] = {'selected': physical, 'fold': fold, 'threshold': selected_tau if operating_point == 'selected_tau' else 0.0}
        tau, rescues, breaks = calibrate(list(validation_rows.values()), common, gallery, roles)
        assert tau == selected_tau
        assert rescues == result['calibration']['rescues'] and breaks == result['calibration']['breaks']
        losses = []
        for q, row in validation_rows.items():
            truth = [gallery[p] == roles[q]['identity'] for p in row['axis']]
            losses.append(held_loss(row['logits'], truth, row['winner']))
        validation_loss = sum(losses) / len(losses)
        assert abs(validation_loss - min(r['validation_loss'] for r in inner_history)) < 2e-10
        fit_receipts.append({'fold': fold, 'seed': seed, 'arm': arm, 'result': bind(folder / 'result.json'),
                             'model': bind(folder / 'model.pt'), 'checkpoint': bind(folder / 'checkpoint.pt'),
                             'inner_validation': bind(folder / 'inner_validation.json'), 'best_epoch': result['best_epoch'],
                             'selected_tau': tau})
        print(json.dumps({'stage': 'independent_replay', 'completed': job_index + 1, 'total': 60, 'fold': fold, 'seed': seed, 'arm': arm}), flush=True)
    assert len(predictions) == 24 and all(set(p) == panel for p in predictions.values())
    # All formal scores are now sealed and checked. Outer labels are joined only
    # for final reporting; no epoch, threshold, arm or seed is selected from them.
    curator_path = RC / 'results/rc_new_hyp593_oof5_v1/metadata/curator_roles.json'
    curator = keyed(read(curator_path)['records'])
    assert panel <= set(curator)
    historical_path = RC / 'results/rc_h593_s_bias_competition_v1/result.json'
    historical = keyed(read(historical_path)['rows'])
    h496_path = out / 'historical496_predictions.json'
    h496 = read(h496_path)['records']
    raw = {q: common[q]['axis'][common[q]['winner']] for q in panel}
    for name in ['RAW', 'COST1_481', 'GAP_BIAS2_492', 'CONTENT18_496']:
        records = {}
        for q in panel:
            selected = raw[q] if name == 'RAW' else h496[q]['selected'] if name == 'CONTENT18_496' else historical[q]['selected']['COST1_FULL' if name == 'COST1_481' else 'GAP_BIAS2']
            assert selected in common[q]['axis']
            records[q] = {'selected': selected, 'fold': curator[q]['outer_fold'], 'threshold': None}
        predictions[(name, None, 'historical')] = records
    assert sum(r['correct']['COST1_FULL'] for r in historical.values()) == 481
    assert sum(r['correct']['GAP_BIAS2'] for r in historical.values()) == 492
    assert sum(gallery[r['selected']] == curator[q]['identity'] for q, r in h496.items()) == 496
    ordered = sorted(panel)
    raw_correct = np.array([gallery[raw[q]] == curator[q]['identity'] for q in ordered])
    accuracy_vectors = {}
    main_rows = []
    for key, values in predictions.items():
        name, seed, operating_point = key
        correct = np.array([gallery[values[q]['selected']] == curator[q]['identity'] for q in ordered])
        selected = np.array([values[q]['selected'] for q in ordered])
        changed = selected != np.array([raw[q] for q in ordered])
        accuracy_vectors[key] = correct
        row = {'model': name, 'seed': '' if seed is None else seed, 'operating_point': operating_point,
               'queries': 71, 'correct': int(correct.sum()), 'accuracy': float(correct.mean()),
               'rescues_vs_RAW': int((correct & ~raw_correct).sum()), 'breaks_vs_RAW': int((~correct & raw_correct).sum()),
               'net_vs_RAW': int(correct.sum() - raw_correct.sum()), 'switches': int(changed.sum()),
               'wrong_to_wrong_switches': int((changed & ~correct & ~raw_correct).sum()),
               'training_population': 'F71 original outer TRAIN intersection' if seed is not None else 'frozen historical original H593 outer TRAIN; RAW is untrained'}
        for fold in range(5):
            mask = np.array([curator[q]['outer_fold'] == fold for q in ordered])
            row[f'fold{fold}_queries'] = int(mask.sum())
            row[f'fold{fold}_correct'] = int(correct[mask].sum())
        main_rows.append(row)
    bootstrap_matrices = {}
    for grouping in ['identity', 'component']:
        groups = sorted({curator[q][grouping] for q in ordered})
        members = [np.array([curator[q][grouping] == g for q in ordered]) for g in groups]
        draws = np.random.default_rng(20260927).integers(0, len(groups), size=(2000, len(groups)))
        bootstrap_matrices[grouping] = (members, draws)

    def comparison(newkey, basekey, role):
        new, old = accuracy_vectors[newkey], accuracy_vectors[basekey]
        delta = new.astype(float) - old.astype(float)
        row = {'new_model': newkey[0], 'baseline': basekey[0], 'seed': newkey[1], 'operating_point': newkey[2],
               'comparison_role': role, 'queries': 71, 'rescues': int((new & ~old).sum()), 'breaks': int((~new & old).sum()),
               'net_correct': int(new.sum() - old.sum()), 'accuracy_delta': float(delta.mean())}
        for grouping, (members, draws) in bootstrap_matrices.items():
            totals = np.array([delta[m].sum() for m in members])
            counts = np.array([m.sum() for m in members])
            sampled = totals[draws].sum(1) / counts[draws].sum(1)
            equal = (totals / counts)[draws].mean(1)
            lo, hi = np.quantile(sampled, [.025, .975])
            elo, ehi = np.quantile(equal, [.025, .975])
            row.update({f'{grouping}_groups': len(members), f'{grouping}_cluster_bootstrap_lo': float(lo),
                        f'{grouping}_cluster_bootstrap_hi': float(hi), f'{grouping}_equal_mean_delta': float((totals / counts).mean()),
                        f'{grouping}_equal_bootstrap_lo': float(elo), f'{grouping}_equal_bootstrap_hi': float(ehi)})
        return row

    comparisons = []
    for key in sorted(k for k in predictions if k[1] is not None):
        arm, seed, point = key
        for base in ['RAW', 'COST1_481', 'GAP_BIAS2_492', 'CONTENT18_496']:
            comparisons.append(comparison(key, (base, None, 'historical'), 'historical_same71_context_different_training_population'))
        if arm != 'B_CAL':
            comparisons.append(comparison(key, ('B_CAL', seed, point), 'secondary_QR_vs_B_CAL' if arm == 'QR' else 'matched_budget_baseline'))
        if arm == 'QRR':
            comparisons.append(comparison(key, ('QR_VEC', seed, point), 'primary_QRR_vs_QR_VEC'))
    write_csv(out / 'main_results.csv', main_rows)
    write_csv(out / 'paired_comparisons.csv', comparisons)
    write_json(out / 'joined_predictions.json', {'panel': 'F71', 'records': [
        {'model': key[0], 'seed': key[1], 'operating_point': key[2], 'query_id': q,
         'display_id': curator[q]['original_query_id'], 'identity': curator[q]['identity'],
         'component': curator[q]['component'], **row,
         'correct': gallery[row['selected']] == curator[q]['identity']}
        for key, values in predictions.items() for q, row in sorted(values.items())]})
    report = ['# QR / QR-vec / QRR: F71 exploratory grouped development result', '',
              'This is the frozen existing ordinal0–70 panel (71 queries), not a new593-query result or an external GO. All60 predeclared arm/seed/fold fits completed before outer labels were joined.', '',
              'The original query identity/component folds remain unchanged. Gallery references of held identities may occur as training negatives: this is a fixed-gallery evaluation, not unseen-reference-gallery generalization. See fold_exposure_audit.json.', '',
              'Each seed is reported separately;71 queries are not counted as213 independent observations. Historical481/492/496 heads are evaluated on these same71 queries but were trained on larger original H593 outer-TRAIN populations.', '',
              '| Model | Seed | Operating point | Correct /71 | RAW rescue / break | Wrong-to-wrong switches |',
              '|---|---:|---|---:|---:|---:|']
    for r in main_rows:
        report.append(f"| {r['model']} | {r['seed']} | {r['operating_point']} | {r['correct']}/71 | {r['rescues_vs_RAW']}/{r['breaks_vs_RAW']} | {r['wrong_to_wrong_switches']} |")
    report += ['', 'Primary structural comparison: QRR versus QR_VEC with the same seed and operating point. Secondary: QR versus B_CAL. Both zero-threshold and inner-validation-selected-threshold results are retained; no held-based operating-point or seed selection is performed.', '',
               'Paired intervals use2000 identity-cluster bootstrap resamples; component-cluster sensitivity and equal-group estimates are included. Intervals are descriptive conditional on these fixed predictions and do not account for repeated H593 development or overlapping training sets.', '',
               'A negative pilot cannot establish absence of useful information in richer evidence or establish M sufficiency. A positive pilot remains developmental and does not replace the original full-population results. No external GO is declared.', '',
               '- [Main table](main_results.csv)', '- [Paired comparisons](paired_comparisons.csv)', '- [Independent validation](validation.json)', '- [Full joined predictions](joined_predictions.json)', '']
    (out / 'report.md').write_text('\n'.join(report))
    receipt = {'status': 'REBUT_QR_QRR_ALL60_F71_INDEPENDENT_JOIN_PASS', 'protocol': bindings,
               'fits_complete': 60, 'panel_queries': 71, 'seeds_reported_separately': protocol['seeds'],
               'new_model_held_predictions_replayed': 71 * 4 * 3, 'held_scores_replayed': 71 * 4 * 3 * 128,
               'maximum_model_replay_error': replay_max, 'inner_epoch_and_threshold_replayed': True,
               'held_labels_read_only_after_all60_checks': True,
               'identity_cluster_bootstrap_resamples': 2000, 'component_sensitivity': True,
               'new_training_updates': 0, 'external_GO': False, 'formal_H593_replacement': False,
               'sources': {'curator': bind(curator_path), 'historical481492': bind(historical_path),
                           'historical496': bind(h496_path), 'evidence_manifest': manifest_binding,
                           'common_features': protocol['common_features'], 'module': bind(module_path), 'runner': bind(runner_path)},
               'fits': fit_receipts, 'verifier': bind(__file__),
               'artifacts': {p: bind(out / p) for p in ['main_results.csv', 'paired_comparisons.csv', 'joined_predictions.json', 'report.md']},
               'issues': []}
    write_json(out / 'validation.json', receipt)
    print(json.dumps({'status': receipt['status'], 'fits': 60, 'queries': 71, 'maximum_replay_error': replay_max}), flush=True)


if __name__ == '__main__':
    main()
