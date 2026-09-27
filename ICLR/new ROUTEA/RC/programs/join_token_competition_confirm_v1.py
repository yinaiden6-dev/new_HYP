#!/usr/bin/env python3
"""Independently replay all 30 fits before opening labels; report every seed."""
import argparse
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import torch
import join_token_competition_v2 as core
from token_competition_confirm_common_v1 import (
    ARMS, FINAL_PASS, RC, bind, checked, read, verify_confirmation)
from token_competition_confirm_data_v1 import TokenInputs, verify_evidence_gate
from join_rebut_qr_qrr_v1 import keyed, write_csv, write_json
from join_rebut_qr_qrr_frozenbase_v2 import baseline_binding, collect_predictions, need
from join_rebut_qr_qrr_f128_v1 import common_artifacts
from run_rebut_qr_qrr_v1 import atomic_json


def require_all_results(protocol):
    missing = []
    for fold in range(5):
        for seed in (1, 2):
            for arm in ARMS:
                folder = Path(protocol['output_root']) / f'fold{fold}/seed{seed}/{arm}'
                result = folder / 'result.json'
                adapter = folder / 'execution_adapter.json'
                if not result.exists() or not adapter.exists() or read(result).get('status') != 'FROZEN_BASE_RESIDUAL_LABEL_FREE_COMPLETE':
                    missing.append({'fold': fold, 'seed': seed, 'arm': arm})
    if missing:
        print(json.dumps({'status': 'TOKEN_CONFIRMATION_WAITING_FOR_ALL30_RESULTS',
            'complete': 30 - len(missing), 'total': 30, 'missing': missing,
            'held_labels_opened': False}), flush=True)
        raise SystemExit(75)


def verify_adapter(folder, protocol_path, protocol, fold, seed, arm):
    path = folder / 'execution_adapter.json'
    value = read(path)
    need(value.get('status') == 'TOKEN_CONFIRMATION_EXECUTION_ADAPTER_PASS', 'ADAPTER_STATUS')
    need(value.get('protocol') == bind(protocol_path) and value.get('result') == bind(folder / 'result.json'), 'ADAPTER_INPUT_BINDINGS')
    need((value['fold'], value['seed'], value['arm']) == (fold, seed, arm), 'ADAPTER_FIT_ID')
    need(value.get('held_labels_opened') is False, 'ADAPTER_LABEL_BOUNDARY')
    need(value.get('origin_evidence_validation') == protocol['origin_evidence_validation'], 'ADAPTER_ORIGIN_EVIDENCE')
    names = ('programs/run_token_competition_confirm_v1.py', 'programs/token_competition_confirm_data_v1.py',
             'programs/run_token_competition_v2.py', 'programs/token_competition_data_v2.py')
    need(value.get('sources') == {n: protocol['code_sources'][n] for n in names}, 'ADAPTER_SOURCE_PINS')
    for source in value['sources'].values():
        checked(source)
    return bind(path)


def verify_all(protocol_path, deadline):
    protocol = read(protocol_path)
    verify_confirmation(protocol)
    gate = verify_evidence_gate(protocol)
    require_all_results(protocol)  # Before even constructing TRAIN-only data.
    source = read(checked(protocol['source_validation']))
    need(source['status'] == 'F128_BASELINE15_INDEPENDENT_PASS' and source['protocol'] == protocol['source_protocol'], 'BASELINE_SEAL')
    checked(protocol['source_protocol'])
    root = Path(protocol['output_root'])
    receipts_dir = root / 'independent_receipts'
    receipts_dir.mkdir(exist_ok=True)
    manifest = read(protocol['evidence_manifest'])
    evidence_bindings = {r['query_id']: {'path': str(checked(r['payload'])), 'sha256': r['payload']['sha256']}
                         for r in manifest['records']}
    sources = {'protocol': bind(protocol_path), 'verifier': bind(__file__),
        'v2_verifier': bind(RC / 'programs/join_token_competition_v2.py'),
        'source_code': protocol['code_sources'], 'common_features': protocol['common_features'],
        'gallery': protocol['gallery'], 'evidence_manifest': bind(protocol['evidence_manifest']),
        'evidence_validation': gate, 'origin_protocol': protocol['origin_protocol']}
    receipts, maximum = [], 0.0
    for fold in range(5):
        data = None
        for seed in (1, 2):
            for arm in ARMS:
                folder = root / f'fold{fold}/seed{seed}/{arm}'
                adapter = verify_adapter(folder, protocol_path, protocol, fold, seed, arm)
                baseline = baseline_binding(protocol, fold, seed)
                for value in baseline.values():
                    checked(value)
                roles = protocol['folds'][str(fold)]['train_roles']
                checked(roles)
                inputs = {'sources': sources, 'fold': fold, 'seed': seed, 'arm': arm,
                    'artifacts': common_artifacts(folder, True), 'execution_adapter': adapter,
                    'evidence': evidence_bindings, 'train_roles': roles, 'baseline': baseline,
                    'baseline_validation': protocol['source_validation'],
                    'baseline_inner': bind(Path(baseline['source_result']['path']).parent / 'inner_validation.json')}
                receipt_path = receipts_dir / f'fold{fold}_seed{seed}_{arm}.json'
                prior = read(receipt_path) if receipt_path.exists() else None
                cached = prior is not None and prior.get('status') == 'TOKEN_CONFIRMATION_FIT_INDEPENDENT_PASS' and prior.get('inputs') == inputs
                if cached:
                    for value in prior['trace_artifacts'].values():
                        checked(value)
                    receipt = prior
                else:
                    if time.monotonic() >= deadline - 10:
                        print(json.dumps({'status': 'TOKEN_CONFIRMATION_REPLAY_RESUMABLE',
                            'fits_verified': len(receipts), 'held_labels_opened': False}), flush=True)
                        raise SystemExit(75)
                    if data is None:
                        data = TokenInputs(protocol, protocol['folds'][str(fold)], arm)
                    core.REPLAY_CONTEXT.clear()
                    core.REPLAY_CONTEXT.update({'folder': folder, 'deadline': deadline,
                        'optimizer': protocol['optimizer'], 'sources': {**sources, 'evidence': evidence_bindings}})
                    details = core.validate_residual(protocol_path, protocol, fold, seed, arm, folder, data)
                    traces = {q: bind(folder / 'independent_replay/outer_refit' / (q + '.pt'))
                              for q in protocol['folds'][str(fold)]['heldout_query_ids']}
                    receipt = {'status': 'TOKEN_CONFIRMATION_FIT_INDEPENDENT_PASS', 'inputs': inputs,
                        'held_labels_opened': False, 'trace_artifacts': traces, **details}
                    atomic_json(receipt_path, receipt)
                maximum = max(maximum, receipt['max_replay_error'])
                receipts.append({'receipt': bind(receipt_path), 'fold': fold, 'seed': seed, 'arm': arm,
                    'selected_epoch': receipt['selected_epoch'], 'max_replay_error': receipt['max_replay_error']})
                print(json.dumps({'stage': 'TOKEN_CONFIRMATION_independent_replay',
                    'complete': len(receipts), 'total': 30, 'fold': fold, 'seed': seed, 'arm': arm, 'cached': cached}), flush=True)
        del data
    need(verify_evidence_gate(protocol) == gate, 'EVIDENCE_CHANGED_DURING_REPLAY')
    for value in evidence_bindings.values():
        checked(value)
    return protocol, receipts, maximum


def summarize(predictions, panel, curator, gallery, raw):
    """Pure aggregation, also exercised on tiny synthetic decision fixtures."""
    vectors, metrics, perfold, joined, pairs = {}, [], [], [], []
    raw_ok = np.array([gallery[raw[q]] == curator[q]['identity'] for q in panel])
    for (model, seed, point), rows in predictions.items():
        need(set(rows) == set(panel), 'EXACT_OOF_PANEL_COVERAGE')
        ok = np.array([gallery[rows[q]['selected']] == curator[q]['identity'] for q in panel])
        vectors[(model, seed, point)] = ok
        for fold in [None, 0, 1, 2, 3, 4]:
            mask = np.array([fold is None or curator[q]['outer_fold'] == fold for q in panel])
            if not mask.any():
                continue
            value = {'model': model, 'seed': '' if seed is None else seed,
                'operating_point': point, 'queries': int(mask.sum()), 'correct': int(ok[mask].sum()),
                'accuracy': float(ok[mask].mean()),
                'rescues_vs_RAW': int((ok & ~raw_ok & mask).sum()),
                'breaks_vs_RAW': int((~ok & raw_ok & mask).sum()),
                'net_vs_RAW': int(ok[mask].sum() - raw_ok[mask].sum())}
            (metrics if fold is None else perfold).append(value if fold is None else {'fold': fold, **value})
        for q, good in zip(panel, ok):
            need(rows[q]['fold'] == curator[q]['outer_fold'], 'CURATOR_FOLD_MISMATCH')
            joined.append({'model': model, 'seed': seed, 'operating_point': point,
                           'query_id': q, **rows[q], 'correct': bool(good)})
    identities = np.array([curator[q]['identity'] for q in panel])
    clusters = sorted(set(identities))
    counts = np.array([(identities == ident).sum() for ident in clusters])
    rng = np.random.default_rng(20260927)
    sampled = rng.integers(0, len(clusters), size=(10000, len(clusters)))
    for arm, base in [(a, 'B_CAL') for a in ARMS] + [('TOKEN_QRR_MULTI', 'TOKEN_QRR_ANCHOR')]:
        for point in ('zero', 'inherited_tau'):
            for seeds, label in [([0], '0'), ([1], '1'), ([2], '2'), ([1, 2], '1,2'), ([0, 1, 2], '0,1,2')]:
                new = np.stack([vectors[(arm, seed, point)] for seed in seeds])
                old = np.stack([vectors[(base, seed, point)] for seed in seeds])
                diff = new.astype(int) - old.astype(int)
                # Resample identities jointly across every included seed. Seeds
                # are repeated measurements of the same queries, not new data.
                cluster_sums = np.array([diff[:, identities == ident].sum() for ident in clusters])
                boot = cluster_sums[sampled].sum(1) / (len(seeds) * counts[sampled].sum(1))
                changed = sum(predictions[(arm, seed, point)][q]['selected'] != predictions[(base, seed, point)][q]['selected']
                              for seed in seeds for q in panel)
                pairs.append({'model': arm, 'seed': label, 'operating_point': point, 'baseline': 'same_F128_' + base,
                    'queries': len(panel), 'query_seed_observations': len(panel) * len(seeds),
                    'rescues': int((new & ~old).sum()), 'breaks': int((~new & old).sum()),
                    'net': int(diff.sum()), 'mean_net_per_seed': float(diff.sum() / len(seeds)),
                    'baseline_correct': int(old.sum()),
                    'baseline_correct_loss_rate': float((~new & old).sum() / old.sum()) if old.any() else None,
                    'changed_decisions': changed,
                    'identity_cluster_gain_ci_low': float(np.quantile(boot, .025)),
                    'identity_cluster_gain_ci_high': float(np.quantile(boot, .975)),
                    'aggregation': 'single_seed' if len(seeds) == 1 else 'all_declared_seeds_no_selection'})
    summaries = []
    for model in ('B_CAL',) + ARMS:
        for point in ('zero', 'inherited_tau'):
            for seeds, label in [([1, 2], '1,2'), ([0, 1, 2], '0,1,2')]:
                correct = [int(vectors[(model, seed, point)].sum()) for seed in seeds]
                summaries.append({'model': model, 'operating_point': point, 'seeds': label,
                    'unique_queries': len(panel), 'query_seed_observations': len(seeds) * len(panel),
                    'mean_correct_per_seed': float(np.mean(correct)),
                    'mean_accuracy': float(np.mean(correct) / len(panel)),
                    'seed_correct_counts': json.dumps(correct), 'seed_selection': False})
    return metrics, perfold, pairs, summaries, joined


def write_final(protocol_path, protocol, receipts, maximum):
    need(len(receipts) == 30 and {(r['fold'], r['seed'], r['arm']) for r in receipts} ==
         {(f, s, a) for f in range(5) for s in (1, 2) for a in ARMS}, 'ALL30_REQUIRED_BEFORE_HELD_LABELS')
    for item in receipts:
        value = read(checked(item['receipt']))
        need(value['status'] == 'TOKEN_CONFIRMATION_FIT_INDEPENDENT_PASS', 'FINAL_RECEIPT_STATUS')
    origin_validation = read(checked(protocol['origin_validation']))
    curator_path = checked(origin_validation['sources']['curator'])
    # Sole held-label boundary, reached only after all30 independent replays.
    curator = keyed(read(curator_path)['records'])
    common = keyed(torch.load(checked(protocol['common_features']), map_location='cpu', weights_only=False)['records'])
    gallery = {int(r['physical_row']): r['identity'] for r in read(checked(protocol['gallery']))['records']}
    panel = sorted(protocol['panel_query_ids'])
    root = Path(protocol['output_root'])
    predictions, traces = {}, []
    seed0_binding = origin_validation['artifacts']['joined_predictions.json']
    for row in read(checked(seed0_binding))['records']:
        if row['model'] == 'RAW':
            continue
        need(row['seed'] == 0, 'ORIGIN_SEED0_ONLY')
        key = (row['model'], 0, row['operating_point'])
        target = predictions.setdefault(key, {})
        need(row['query_id'] not in target, 'DUPLICATE_ORIGIN_PREDICTION')
        target[row['query_id']] = {k: row[k] for k in ('selected', 'fold', 'threshold')}
    need(len(predictions) == 8, 'ALL_ORIGIN_ARMS_AND_OPERATING_POINTS')
    for fold in range(5):
        for seed in (1, 2):
            base = read(checked(baseline_binding(protocol, fold, seed)['source_result']))
            collect_predictions(predictions, 'B_CAL', seed, fold, base['predictions'], base['calibration']['threshold'])
            for arm in ARMS:
                folder = root / f'fold{fold}/seed{seed}/{arm}'
                result = read(folder / 'result.json')
                collect_predictions(predictions, arm, seed, fold, result['predictions'], result['calibration']['inherited_tau_secondary'])
                for row in result['predictions']:
                    path = folder / 'independent_replay/outer_refit' / (row['query_id'] + '.pt')
                    traces.append({'model': arm, 'fold': fold, 'seed': seed, 'query_id': row['query_id'], 'trace': bind(path)})
    need(len(predictions) == 24, 'ALL_THREE_SEEDS_REQUIRED')
    raw = {q: int(common[q]['axis'][common[q]['winner']]) for q in panel}
    predictions[('RAW', None, 'untrained')] = {q: {'selected': raw[q], 'fold': curator[q]['outer_fold'], 'threshold': None} for q in panel}
    metrics, perfold, pairs, summary, joined = summarize(predictions, panel, curator, gallery, raw)
    write_csv(root / 'metrics.csv', metrics)
    write_csv(root / 'perfold_metrics.csv', perfold)
    write_csv(root / 'paired_vs_B_CAL.csv', [r for r in pairs if r['baseline'] == 'same_F128_B_CAL'])
    write_csv(root / 'paired_structural.csv', [r for r in pairs if r['baseline'] == 'same_F128_TOKEN_QRR_ANCHOR'])
    write_csv(root / 'seed_summary.csv', summary)
    write_json(root / 'joined_predictions.json', {'panel': 'F128 opened development', 'seeds': [0, 1, 2],
        'seed0_reference': seed0_binding, 'seed_selection': False, 'records': joined})
    write_json(root / 'trace_manifest.json', {'records': traces,
        'seed0_reference': origin_validation['artifacts']['trace_manifest.json']})
    scope = {'head': '18-parameter frozen SAME F128 B_CAL plus native-token residual; phase-specific frozen head and RMS',
        'model': list(ARMS), 'dataset': 'Original F128 execution ordinals0..127; grouped fivefold',
        'candidate_source': 'Original natural ColNomic C128; all128 candidates remain eligible',
        'action': 'Fixed zero primary; original same-seed B_CAL inherited threshold secondary',
        'baseline': 'Same F128, same seed B_CAL; MULTI vs ANCHOR structural comparison',
        'evidence_level': 'Additional seeds on the same opened development panel, not independent queries or external validation',
        'origin_protocol': protocol['origin_protocol'], 'new_seeds': [1, 2], 'reference_seed': 0,
        'best_seed_selected': False, 'new_backbone_forwards': 0, 'external_GO': False,
        'formal_H593_replacement': False, 'additional_round_automatically_authorized': False}
    write_json(root / 'scope.json', scope)
    report = ['# Native token competition: F128 seed confirmation', '',
        'All30 additional fits (all three arms, seeds1 and2, five original folds) passed independent V2 numerical replay before this join opened held labels. '
        'Seed0 is the original frozen V2 reference. These are repeated seeds on the same128 opened development queries; no best seed is selected.', '',
        'The native ColNomic C128 cache, architecture, full-C128 hit/miss loss, AdamW budget, nested folds and fixed-zero action are unchanged. '
        'Each arm freezes its matching same-seed B_CAL head and RMS separately for inner fitting and outer refit. '
        'Inherited B_CAL thresholds remain secondary; epoch0 remains the exact baseline fallback.', '',
        '| Model | Seed | Point | Correct /128 | RAW rescue / break |',
        '|---|---:|---|---:|---:|']
    report += [f"| {r['model']} | {r['seed']} | {r['operating_point']} | {r['correct']}/128 | {r['rescues_vs_RAW']}/{r['breaks_vs_RAW']} |" for r in metrics]
    report += ['', 'Paired tables include every seed separately, seeds1+2, and all three seeds. '
        'Bootstrap resampling keeps all repeated seeds of each identity together. '
        'This experiment does not establish full-H593 performance or external confirmation. '
        'No further experiment is automatically authorized by this report.', '',
        '[Per-fold metrics](perfold_metrics.csv) · [Paired versus B_CAL](paired_vs_B_CAL.csv) · '
        '[MULTI versus ANCHOR](paired_structural.csv) · [Seed summaries](seed_summary.csv) · '
        '[Scope](scope.json) · [Validation](validation.json)', '']
    (root / 'REPORT_TOKEN_COMPETITION_CONFIRM_V1.md').write_text('\n'.join(report))
    files = ['metrics.csv', 'perfold_metrics.csv', 'paired_vs_B_CAL.csv', 'paired_structural.csv',
        'seed_summary.csv', 'joined_predictions.json', 'trace_manifest.json', 'scope.json',
        'REPORT_TOKEN_COMPETITION_CONFIRM_V1.md']
    validation = {'status': FINAL_PASS, 'protocol': bind(protocol_path),
        'fits_complete': 30, 'reference_fits': 15, 'panel_queries': 128,
        'new_seeds': [1, 2], 'reported_seeds': [0, 1, 2], 'seed_selection': False,
        'held_predictions_replayed': 128 * 3 * 2, 'held_scores_replayed': 128 * 3 * 2 * 128,
        'maximum_model_replay_error': maximum, 'held_labels_read_only_after_all30_checks': True,
        'identity_cluster_bootstrap_resamples': 10000, 'seed_clustered_with_query': True,
        'development_analysis': True, 'external_GO': False, 'formal_H593_replacement': False,
        'receipts': receipts, 'sources': {'curator': bind(curator_path), 'verifier': bind(__file__),
            'v2_verifier': bind(RC / 'programs/join_token_competition_v2.py'),
            'origin_validation': protocol['origin_validation'], 'baseline_validation': protocol['source_validation']},
        'artifacts': {name: bind(root / name) for name in files}}
    atomic_json(root / 'validation.json', validation)
    print(json.dumps({'status': FINAL_PASS, 'fits_complete': 30, 'panel_queries': 128}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--wall-seconds', type=float, default=480)
    args = parser.parse_args()
    if args.threads < 1 or args.wall_seconds <= 0:
        raise ValueError('Positive thread count and wall budget required')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    lock = (args.protocol.resolve().parent / 'join.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, receipts, maximum = verify_all(args.protocol, time.monotonic() + args.wall_seconds)
    write_final(args.protocol, protocol, receipts, maximum)


if __name__ == '__main__':
    main()
