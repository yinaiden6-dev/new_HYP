#!/usr/bin/env python3
"""Scheduler-neutral bounded driver for 30 confirmation fits and their join."""
import argparse
import subprocess
import sys
from pathlib import Path
from token_competition_confirm_common_v1 import OUT, RC, read, verify_confirmation


def task(index):
    if not 0 <= index < 30:
        raise ValueError('Confirmation task index must be 0..29')
    return index // 6, 1 + (index % 6) // 3, ('TOKEN_QR', 'TOKEN_QRR_ANCHOR', 'TOKEN_QRR_MULTI')[index % 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'join'])
    parser.add_argument('--protocol', type=Path, default=OUT / 'protocol.json')
    parser.add_argument('--index', type=int)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--wall-seconds', type=float, default=480)
    args = parser.parse_args()
    if args.threads < 1 or args.wall_seconds <= 10:
        raise ValueError('Positive thread count and more than10 seconds wall budget required')
    verify_confirmation(read(args.protocol))
    program = 'run_token_competition_confirm_v1.py' if args.stage == 'train' else 'join_token_competition_confirm_v1.py'
    command = [sys.executable, '-u', str(RC / 'programs' / program), '--protocol', str(args.protocol),
               '--threads', str(args.threads), '--wall-seconds', str(args.wall_seconds - 5)]
    if args.stage == 'train':
        if args.index is None:
            raise ValueError('Training requires an explicit --index0..29')
        fold, seed, arm = task(args.index)
        command += ['--fold', str(fold), '--seed', str(seed), '--arm', arm]
    return subprocess.run(command).returncode


if __name__ == '__main__':
    raise SystemExit(main())
