#!/usr/bin/env python3
"""Pack twelve independent arm/seed fits into one fold's CPU allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', required=True, type=int, choices=range(5))
    p.add_argument('--wall-seconds', type=int, default=480)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    protocol = ROOT/'results/rc_rebut_qr_qrr_v1/protocol.json'
    config = json.loads(protocol.read_text())
    started = time.monotonic()
    jobs = [(seed, arm) for seed in config['seeds'] for arm in config['arms']]

    def one(item):
        seed, arm = item
        remaining = a.wall_seconds - (time.monotonic() - started) - 15
        if remaining < 20:
            return dict(seed=seed, arm=arm, exit_code=75, reason='allocation budget')
        command = [sys.executable, '-u', str(ROOT/'programs/run_rebut_qr_qrr_v1.py'),
                   '--protocol', str(protocol), '--fold', str(a.fold), '--seed', str(seed),
                   '--arm', arm, '--threads', str(a.threads), '--wall-seconds', str(int(remaining))]
        if a.dry_run:
            return dict(command=command, exit_code=0)
        result = subprocess.run(command, check=False)
        return dict(seed=seed, arm=arm, exit_code=result.returncode)

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for task in as_completed([pool.submit(one, item) for item in jobs]):
            result = task.result(); results.append(result)
            print(json.dumps(dict(stage='packed_task', fold=a.fold, **result)), flush=True)
    bad = [x for x in results if x['exit_code'] not in (0, 75)]
    if bad:
        print(json.dumps(dict(stage='FAILED', tasks=bad)), flush=True)
        return 1
    return 75 if any(x['exit_code'] == 75 for x in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
