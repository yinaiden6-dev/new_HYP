#!/usr/bin/env python3
"""Export all validated TRAIN pilot margins; no inference or parameter fitting."""
from fractions import Fraction
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rc-maxmin-pilot-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_reference_support_maxmin_train_pilot_v1'
DEST = ROOT / 'reports/figures/new_hyp_train_support_capacity_v1'


def bind(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fraction(item):
    return Fraction(int(item['numerator']), int(item['denominator']))


def main():
    result = json.loads((SOURCE / 'result.json').read_text())
    valid = json.loads((SOURCE / 'validation.json').read_text())
    assert valid['result'] == bind(SOURCE / 'result.json')
    assert valid['status'] == 'TRAIN_MAXMIN_PILOT_INDEPENDENT_EXACT_CERTIFICATE_VALIDATION_PASS'
    assert result['summary']['all512_gap_qualified']
    assert bind(SOURCE / 'prejoin_seal.json') == result['prejoin_seal']
    seal = json.loads((SOURCE / 'prejoin_seal.json').read_text())
    targets = {r['execution_ordinal']: r for r in result['TRAIN_postjoin']['targets']}
    rows = []
    for r in seal['records']:
        lo, hi = (fraction(r['certificate'][k]) for k in ('lower', 'upper'))
        b = fraction(r['original_exact_support_margin'])
        rows.append({'query_id': r['query_id'], 'execution_ordinal': r['execution_ordinal'],
                     'candidate_position': r['candidate_position'], 'physical_row': r['physical_row'],
                     'target': r['candidate_position'] == targets[r['execution_ordinal']]['target_position'],
                     'fixed_support_margin': float(b), 'optimized_lower': float(lo),
                     'optimized_upper': float(hi), 'lower_numerator': str(lo.numerator),
                     'lower_denominator': str(lo.denominator),
                     'fixed_numerator': str(b.numerator), 'fixed_denominator': str(b.denominator)})
    assert len(rows) == 512 and sum(r['target'] for r in rows) == 4
    DEST.mkdir(parents=True, exist_ok=False)
    with (DEST / 'margins.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    low = min(min(r['fixed_support_margin'], r['optimized_lower']) for r in rows) - .02
    high = max(max(r['fixed_support_margin'], r['optimized_upper']) for r in rows) + .02
    for ax, ex in zip(axes.flat, targets):
        subset = [r for r in rows if r['execution_ordinal'] == ex]
        wrong = [r for r in subset if not r['target']]; own = next(r for r in subset if r['target'])
        ax.plot([low, high], [low, high], color='#999999', lw=.8, ls='--', label='Equal margin')
        ax.axhline(0, color='#dddddd', lw=.7); ax.axvline(0, color='#dddddd', lw=.7)
        ax.scatter([r['fixed_support_margin'] for r in wrong], [r['optimized_lower'] for r in wrong],
                   s=14, alpha=.65, color='#546e7a', label='Wrong reference')
        ax.scatter([own['fixed_support_margin']], [own['optimized_lower']], s=95,
                   marker='*', color='#c62828', label='Target reference', zorder=5)
        ax.set_title(own['query_id']); ax.set_xlim(low, high); ax.set_ylim(low, high)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(loc='upper left', fontsize=8, frameon=False)
    fig.supxlabel('Fixed-support margin (exact endpoint pooling)')
    fig.supylabel('Optimized-support certified lower bound')
    fig.suptitle('TRAIN support capacity: useful evidence is not sufficient for identity', fontsize=12)
    fig.text(.5, .925, '4 TRAIN queries / 3 groups; positive: 4/4 targets and 502/508 wrong candidates',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(.02, .02, 1, .92))
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(DEST / ('support_capacity.' + suffix), dpi=180)
    plt.close(fig)
    manifest = {'sources': {name: bind(SOURCE / name) for name in ('result.json', 'validation.json', 'prejoin_seal.json')},
                'program': bind(Path(__file__).resolve()), 'rows': 512,
                'outputs': [bind(p) for p in sorted(DEST.iterdir())],
                'scope': 'TRAIN_CAPACITY_DIAGNOSTIC_NOT_RETRIEVAL_ACCURACY', 'EVAL_reads': 0}
    with (DEST / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    for p in DEST.iterdir(): p.chmod(0o444)
    print(json.dumps(bind(DEST / 'manifest.json')))


if __name__ == '__main__':
    main()
