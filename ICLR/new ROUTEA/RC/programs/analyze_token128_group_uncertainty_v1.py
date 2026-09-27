#!/usr/bin/env python3
"""Describe identity grouping of the frozen F128 effects; no new model selection."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

RC = Path(__file__).resolve().parents[1]


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=RC/'results/rc_token_competition_f128_v2')
    args = parser.parse_args()
    root = args.root.resolve()
    validation_path = root/'validation.json'
    validation = json.loads(validation_path.read_text())
    assert validation['status'] == 'TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS'
    summary_path = root/'analysis/analysis_summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'TOKEN_COMPETITION_ANALYSIS_PASS'
    original_root = Path(validation['protocol']['path']).parent
    assert summary['validation']['path'] == str(original_root/'validation.json')
    assert summary['validation']['sha256'] == bind(validation_path)['sha256']
    table_path = root/'analysis/per_query_diagnostics.csv'
    assert summary['artifacts']['per_query']['path'] == str(original_root/'analysis/per_query_diagnostics.csv')
    assert summary['artifacts']['per_query']['sha256'] == bind(table_path)['sha256']
    with table_path.open() as f:
        rows = list(csv.DictReader(f))
    output_rows, models = [], []
    for model in ('TOKEN_QR', 'TOKEN_QRR_ANCHOR', 'TOKEN_QRR_MULTI'):
        groups = defaultdict(lambda: {'queries': 0, 'rescues': 0, 'breaks': 0, 'net': 0})
        selected = [r for r in rows if r['model'] == model and r['operating_point'] == 'zero']
        assert len(selected) == len({r['query_id'] for r in selected}) == 128
        for row in selected:
            group = groups[row['identity']]
            group['queries'] += 1
            group['rescues'] += row['rescue'] == 'True'
            group['breaks'] += row['break'] == 'True'
            group['net'] += int(row['net'])
        for identity, value in sorted(groups.items()):
            assert value['net'] == value['rescues'] - value['breaks']
            output_rows.append({'model': model, 'identity': identity, **value})
        models.append({'model': model, 'queries': 128, 'identity_groups': len(groups),
            'max_queries_per_identity': max(v['queries'] for v in groups.values()),
            'groups_with_any_changed_correctness': sum(bool(v['rescues'] or v['breaks']) for v in groups.values()),
            'positive_net_groups': sum(v['net'] > 0 for v in groups.values()),
            'negative_net_groups': sum(v['net'] < 0 for v in groups.values()),
            'zero_net_groups': sum(v['net'] == 0 for v in groups.values()),
            'rescues': sum(v['rescues'] for v in groups.values()),
            'breaks': sum(v['breaks'] for v in groups.values()),
            'net': sum(v['net'] for v in groups.values())})
    out = root/'cause_analysis/uncertainty'
    out.mkdir(parents=True, exist_ok=True)
    table = out/'identity_group_effects.csv'
    with table.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    result = {'status': 'TOKEN128_GROUP_EFFECT_ACCOUNTING_PASS',
        'validation': bind(validation_path), 'summary': bind(summary_path), 'input': bind(table_path),
        'source': bind(__file__), 'models': models, 'artifacts': {'groups': bind(table)},
        'interpretation': '128 images form 48 identity groups. The original cluster interval treats images of one identity jointly. These counts describe why a positive point estimate is not the same as a strictly positive lower confidence endpoint; they do not prove that more data alone would resolve the training/mechanism issue.',
        'new_training': False, 'gate_changed': False}
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(models, ensure_ascii=False))


if __name__ == '__main__': main()
