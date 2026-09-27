#!/usr/bin/env python3
"""Run one F128 fold's independent seed/arm fits with bounded CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_rebut_qr_qrr_f128_v1'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('baseline', 'residual'), required=True)
    p.add_argument('--fold', required=True, type=int, choices=range(5))
    p.add_argument('--wall-seconds', type=int, default=480)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if min(a.wall_seconds, a.workers, a.threads) <= 0:
        p.error('budgets and thread counts must be positive')
    protocol = OUT / f'{a.phase}_protocol.json'
    config = json.loads(protocol.read_text())
    expected = ['B_CAL'] if a.phase == 'baseline' else ['QR', 'QR_VEC', 'QRR']
    if (config['arms'] != expected or config['seeds'] != [0, 1, 2]
            or len(config['panel_query_ids']) != 128):
        raise ValueError('F128 frozen phase/panel differs from submitted design')
    start = time.monotonic()

    def one(item):
        seed, arm = item
        remaining = a.wall_seconds - (time.monotonic() - start) - 15
        if remaining < 20:
            return {'seed': seed, 'arm': arm, 'exit_code': 75}
        cmd = [sys.executable, '-u', str(RC / 'programs/run_rebut_qr_qrr_f128_v1.py'),
               '--phase', a.phase, '--protocol', str(protocol), '--fold', str(a.fold),
               '--seed', str(seed), '--arm', arm, '--threads', str(a.threads),
               '--wall-seconds', str(int(remaining))]
        if a.dry_run:
            return {'command': cmd, 'exit_code': 0}
        try:
            result = subprocess.run(cmd, check=False, timeout=max(1, remaining + 35))
        except subprocess.TimeoutExpired:
            return {'seed': seed, 'arm': arm, 'exit_code': 75,
                    'reason': 'child budget expired; resume sealed checkpoint'}
        return {'seed': seed, 'arm': arm, 'exit_code': result.returncode}

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(one, (seed, arm))
                   for seed in config['seeds'] for arm in config['arms']]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({'stage': 'packed_task', 'phase': a.phase,
                              'fold': a.fold, **result}), flush=True)
    if any(x['exit_code'] not in (0, 75) for x in results):
        return 1
    return 75 if any(x['exit_code'] == 75 for x in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
