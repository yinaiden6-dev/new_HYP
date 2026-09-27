#!/usr/bin/env python3
"""Idempotent CPU-only all30 fit -> independent join DAG, gate PASS required."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import re
import subprocess
from pathlib import Path
from token_competition_confirm_common_v1 import OUT, RC, bind, checked, read, verify_confirmation

SCRIPT = RC / 'slurm/token_competition_confirm_v1.sbatch'


def write(path, value):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def command(stage, predecessor=None):
    partition = 'cpuonly'
    cmd = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--partition=' + partition,
           '--job-name=token128c_' + stage, '--comment=TOKEN_COMPETITION_F128_CONFIRM_V1:' + stage]
    if predecessor:
        cmd.append('--dependency=afterok:' + predecessor)
    if stage == 'train':
        cmd.append('--array=0-29%30')
    elif stage != 'join':
        raise ValueError('Exactly one train array and one join are authorized')
    return cmd + [str(SCRIPT), stage]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocol_path = OUT / 'protocol.json'
    protocol = read(protocol_path)
    verify_confirmation(protocol)
    prep = read(OUT / 'preparation_validation.json')
    if prep.get('status') != 'TOKEN_COMPETITION_F128_CONFIRM_PREPARATION_PASS' or prep.get('protocol') != bind(protocol_path):
        raise ValueError('Confirmation preparation is not sealed')
    launcher = read(OUT / 'launcher_engineering.json')
    if launcher.get('status') != 'TOKEN_COMPETITION_CONFIRM_LAUNCHER_PASS' or launcher.get('source') != bind(SCRIPT):
        raise ValueError('Actual confirmation launcher branches must be verified before submission')
    checked(launcher['checker'])
    source_validation = read(OUT / 'source_engineering.json')
    if (source_validation.get('status') != 'TOKEN_COMPETITION_CONFIRM_SOURCES_PASS' or
            source_validation.get('protocol') != bind(protocol_path)):
        raise ValueError('Confirmation wrapper imports and source checks must pass before submission')
    with (OUT / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = OUT / 'submission.json'
        state = read(path) if path.exists() else {'status': 'PREPARED', 'protocol': bind(protocol_path),
            'script': bind(SCRIPT), 'stages': {}, 'population': 128, 'candidate_count': 128,
            'seeds': [1, 2], 'baseline_fits_reused': 10, 'new_reader_fits': 30,
            'new_backbone_forwards': 0, 'gpus': 0, 'authorized_confirmation_rounds': 1}
        if state['protocol'] != bind(protocol_path) or state['script'] != bind(SCRIPT):
            raise ValueError('Existing submission differs; refuse duplicate submission')
        jobs = {}
        for stage in ('train', 'join'):
            if stage in state['stages']:
                jobs[stage] = state['stages'][stage]['job_id']
                continue
            predecessor = jobs.get('train') if stage == 'join' else None
            cmd = command(stage, predecessor)
            if args.dry_run:
                print(json.dumps({'stage': stage, 'command': cmd}), flush=True)
                jobs[stage] = 'DRY_' + stage
                continue
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode:
                state['status'] = 'PARTIAL_SUBMISSION'
                state['last_error'] = {'stage': stage, 'stderr': result.stderr}
                write(path, state)
                raise RuntimeError(result.stderr)
            job = result.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+', job):
                raise ValueError('Unexpected sbatch response: ' + result.stdout)
            jobs[stage] = job
            state['stages'][stage] = {'job_id': job, 'predecessor': predecessor,
                'array': '0-29%30' if stage == 'train' else None,
                'partition': 'cpuonly',
                'command': cmd, 'submitted_utc': datetime.now(timezone.utc).isoformat()}
            state['status'] = 'SUBMITTING'
            write(path, state)
            print(json.dumps({'stage': stage, 'job_id': job, 'afterok': predecessor}), flush=True)
        if not args.dry_run:
            state['status'] = 'COMPLETE_CONFIRMATION_CHAIN_SUBMITTED'
            write(path, state)


if __name__ == '__main__':
    main()
