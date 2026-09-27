#!/usr/bin/env python3
"""Publish common analysis metadata only after the independent all30 join."""
import argparse
import csv
import json
from pathlib import Path
from token_competition_confirm_common_v1 import OUT, FINAL_PASS, bind, checked, read, verify_confirmation


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=OUT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path, validation_path = root / 'protocol.json', root / 'validation.json'
    protocol, validation = read(protocol_path), read(validation_path)
    verify_confirmation(protocol)
    if root != Path(protocol['output_root']).resolve():
        raise ValueError('Analysis root differs from the isolated confirmation protocol')
    if validation.get('status') != FINAL_PASS or validation.get('protocol') != bind(protocol_path):
        raise ValueError('Independent confirmation all30 validation required')
    for binding in validation['artifacts'].values():
        checked(binding)
    metrics = {(r['model'], r['seed'], r['operating_point']): r for r in rows(root / 'metrics.csv')}
    records = read(root / 'joined_predictions.json')['records']
    indexed = {(r['model'], str(r['seed']), r['operating_point'], r['query_id']): r for r in records}
    comparisons = []
    for source in ('paired_vs_B_CAL.csv', 'paired_structural.csv'):
        for pair in rows(root / source):
            if pair['seed'] not in ('0', '1', '2'):
                continue
            seed, model, point = pair['seed'], pair['model'], pair['operating_point']
            baseline = pair['baseline'].removeprefix('same_F128_')
            model_metric = metrics[(model, seed, point)]
            fold_nets = []
            for fold in range(5):
                subset = [r for r in records if r['model'] == model and str(r['seed']) == seed and
                          r['operating_point'] == point and r['fold'] == fold]
                fold_nets.append(sum(int(r['correct']) - int(indexed[(baseline, seed, point, r['query_id'])]['correct']) for r in subset))
            comparisons.append({'model': model, 'baseline': baseline, 'seed': int(seed),
                'operating_point': point, 'queries': int(pair['queries']),
                'correct': int(model_metric['correct']), 'accuracy': float(model_metric['accuracy']),
                'baseline_correct': int(pair['baseline_correct']), 'rescues': int(pair['rescues']),
                'breaks': int(pair['breaks']), 'net': int(pair['net']),
                'gain_percentage_points': 100 * int(pair['net']) / int(pair['queries']),
                'baseline_correct_loss_rate': float(pair['baseline_correct_loss_rate']) if pair['baseline_correct_loss_rate'] else None,
                'changed_decisions': int(pair['changed_decisions']),
                'identity_cluster_gain_ci_low': float(pair['identity_cluster_gain_ci_low']),
                'identity_cluster_gain_ci_high': float(pair['identity_cluster_gain_ci_high']),
                'fold_nets': fold_nets, 'positive_folds': sum(x > 0 for x in fold_nets),
                'negative_folds': sum(x < 0 for x in fold_nets), 'unchanged_folds': sum(x == 0 for x in fold_nets)})
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    mapping = {'report': ('REPORT_TOKEN_COMPETITION_CONFIRM_V1.md', 'REPORT_TOKEN_COMPETITION_CONFIRM_V1.md'),
               'per_fold': ('perfold_metrics.csv', 'per_fold.csv'),
               'per_query': ('joined_predictions.json', 'per_query.json')}
    artifacts = {}
    for name, (original, filename) in mapping.items():
        target = output / filename
        target.write_bytes((root / original).read_bytes())
        artifacts[name] = bind(target)
    value = {'status': 'TOKEN_COMPETITION_ANALYSIS_PASS', 'validation': bind(validation_path),
        'protocol': bind(protocol_path), 'artifacts': artifacts,
        'queries': 128, 'population': 'F128',
        'primary_operating_point': 'zero', 'secondary_operating_point': 'inherited_tau',
        'seeds': [0, 1, 2], 'new_seeds': [1, 2], 'reference_seed': 0,
        'seed_selection': False, 'comparisons': comparisons,
        'aggregate_seed_summaries': validation['artifacts']['seed_summary.csv'],
        'scope': validation['artifacts']['scope.json'], 'analyzer': bind(__file__),
        'inputs': {'analysis_source': bind(__file__)},
        'external_GO': False, 'formal_H593_replacement': False,
        'additional_round_automatically_authorized': False}
    target = output / 'analysis_summary.json'
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': value['status'], 'analysis_summary': bind(target)}), flush=True)


if __name__ == '__main__':
    main()
