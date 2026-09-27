#!/usr/bin/env python3
"""Run nine independent residual fits within one resume-aware CPU allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

RC = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', required=True, type=int, choices=range(5))
    p.add_argument('--wall-seconds', type=int, default=480)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    protocol = RC / 'results/rc_rebut_qr_qrr_frozenbase_v2/protocol.json'
    config = json.loads(protocol.read_text())
    prep = json.loads((protocol.parent / 'baseline_preparation_validation.json').read_text())
    assert prep['status'] == 'FROZENBASE_V2_PHASE_SPECIFIC_BASELINES_PASS'
    assert prep['protocol']['sha256'] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    if not a.dry_run:
        checks = json.loads((protocol.parent / 'engineering_checks.json').read_text())
        assert checks['status'] == 'FROZENBASE_V2_ENGINEERING_PASS'
        for binding in checks['sources'].values():
            assert hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest() == binding['sha256']
    start = time.monotonic()

    def one(item):
        seed, arm = item
        remaining = a.wall_seconds - (time.monotonic() - start) - 15
        if remaining < 20:
            return {'seed': seed, 'arm': arm, 'exit_code': 75}
        cmd = [sys.executable, '-u', str(RC / 'programs/run_rebut_qr_qrr_frozenbase_v2.py'),
               '--protocol', str(protocol), '--fold', str(a.fold), '--seed', str(seed),
               '--arm', arm, '--threads', str(a.threads), '--wall-seconds', str(int(remaining))]
        if a.dry_run:
            return {'command': cmd, 'exit_code': 0}
        result = subprocess.run(cmd, check=False)
        return {'seed': seed, 'arm': arm, 'exit_code': result.returncode}

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = [(seed, arm) for seed in config['seeds'] for arm in config['arms']]
        for future in as_completed([pool.submit(one, job) for job in jobs]):
            result = future.result(); results.append(result)
            print(json.dumps({'stage': 'packed_task', 'fold': a.fold, **result}), flush=True)
    if any(x['exit_code'] not in (0, 75) for x in results):
        return 1
    return 75 if any(x['exit_code'] == 75 for x in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
