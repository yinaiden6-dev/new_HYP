#!/usr/bin/env python3
"""Submit the isolated F128 CPU chain after the existing unified128 join."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_rebut_qr_qrr_f128_v1'
SCRIPT = RC / 'slurm/rebut_qr_qrr_f128_v1.sbatch'


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def stages():
    return [('evidence', 'unified128', None), ('verify', 'evidence', None), ('baseline', 'verify', '0-4%5'),
            ('seal', 'baseline', None), ('residual', 'seal', '0-4%5'),
            ('join', 'residual', None)]


def command(stage, predecessor, array):
    cmd = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes',
           '--partition=accelerated', '--job-name=rebut128_' + stage,
           '--comment=REBUT_QR_QRR_F128_V1:' + stage,
           '--dependency=afterok:' + predecessor]
    if array:
        cmd.append('--array=' + array)
    return cmd + [str(SCRIPT), stage]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--unified128-join', default='5166240')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'\d+', a.unified128_join):
        raise ValueError('invalid Slurm prerequisite ID')
    protocol = OUT / 'protocol.json'
    config = read(protocol)
    if len(config['panel_query_ids']) != 128:
        raise ValueError('not the frozen F128 panel')
    prep = read(OUT / 'preparation_validation.json')
    if (prep['status'] != 'F128_DATA_PREPARATION_PASS_NOT_EVIDENCE_COMPLETION'
            or prep['protocol'] != binding(protocol)
            or prep['baseline_protocol'] != binding(OUT / 'baseline_protocol.json')
            or prep['evidence_inputs'] != binding(OUT / 'evidence_inputs.json')):
        raise ValueError('F128 preparation is not sealed to this protocol')
    def check_sources(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                if binding(value['path'])['sha256'] != value['sha256']:
                    raise ValueError('prepared source changed: ' + value['path'])
            else:
                for v in value.values():
                    check_sources(v)
        elif isinstance(value, list):
            for v in value:
                check_sources(v)
    check_sources(config['sources'])
    check_sources(config['code_sources'])
    data_checks = read(OUT / 'data_engineering.json')
    if data_checks['status'] != 'F128_DATA_ENGINEERING_PASS':
        raise ValueError('data engineering verification missing')
    training_checks = read(OUT / 'training_engineering.json')
    if training_checks['status'] != 'F128_TRAINING_ENGINEERING_PASS':
        raise ValueError('training engineering verification missing')
    check_sources(training_checks['sources'])
    launcher = read(OUT / 'launcher_engineering.json')
    if launcher['status'] != 'F128_LAUNCHER_ENGINEERING_PASS':
        raise ValueError('launcher verification missing')
    check_sources(launcher['sources'])
    # Execution code is sealed separately so a change cannot silently cross a
    # submitted checkpoint or invalidate another conversation's frozen inputs.
    launch_sources = [binding(RC / p) for p in [
        'programs/run_rebut_qr_qrr_f128_fold_v1.py',
        'programs/submit_rebut_qr_qrr_f128_v1.py',
        'slurm/rebut_qr_qrr_f128_v1.sbatch']]
    path = OUT / 'submission.json'
    with (OUT / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(path) if path.exists() else {
            'status': 'PREPARED', 'protocol': binding(protocol),
            'launch_sources': launch_sources, 'unified128_join': a.unified128_join,
            'new_backbone_gpu_forwards': 0, 'gpus_per_allocation': 1,
            'gpu_used_for_computation': False, 'new_baseline_fits': 15, 'new_reader_fits': 45,
            'query_count': 128, 'partition': 'accelerated', 'stages': {}}
        if (state['protocol'] != binding(protocol) or state['launch_sources'] != launch_sources
                or state['unified128_join'] != a.unified128_join):
            raise ValueError('refuse changing a submitted experiment or dependency')
        jobs = {'unified128': a.unified128_join}
        for name, dependency, array in stages():
            if name in state['stages']:
                jobs[name] = state['stages'][name]['job_id']
                continue
            cmd = command(name, jobs[dependency], array)
            if a.dry_run:
                print(json.dumps({'stage': name, 'command': cmd}))
                jobs[name] = 'DRY_' + name
                continue
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                state['status'] = 'PARTIAL_SUBMISSION'
                state['last_error'] = {'stage': name, 'stderr': result.stderr,
                                       'returncode': result.returncode}
                write(path, state)
                raise RuntimeError(result.stderr)
            job = result.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+', job):
                raise ValueError('unexpected sbatch response: ' + result.stdout)
            jobs[name] = job
            state['stages'][name] = {
                'job_id': job, 'predecessor': jobs[dependency], 'command': cmd,
                'submitted_utc': datetime.now(timezone.utc).isoformat()}
            state['status'] = 'SUBMITTING'
            write(path, state)
            print(json.dumps({'stage': name, 'job_id': job, 'afterok': jobs[dependency]}), flush=True)
        if not a.dry_run:
            state['status'] = 'SUBMITTED_WAITING_F128_VALIDATION'
            write(path, state)


if __name__ == '__main__':
    main()
