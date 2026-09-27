#!/usr/bin/env python3
"""Bounded, serial stage runner; only the scheduler determines concurrency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_token_competition_f128_v2'


def run(cmd, deadline):
    remain = deadline-time.monotonic()
    if remain < 20:
        return 75
    return subprocess.run(cmd + ['--wall-seconds', str(max(1., remain-5))]).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['pilot','evidence','verify','train','join'])
    ap.add_argument('--index', type=int)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--wall-seconds', type=float, default=480)
    a = ap.parse_args()
    deadline = time.monotonic()+a.wall_seconds
    path = OUT/'protocol.json'
    p = json.loads(path.read_text())
    protocol_binding={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for b in p['code_sources'].values():
        if hashlib.sha256(Path(b['path']).read_bytes()).hexdigest() != b['sha256']:
            raise ValueError('Submitted source changed: '+b['path'])
    def command(file):
        return [sys.executable,'-u',str(RC/'programs'/file),'--protocol',str(path),
                '--threads',str(a.threads)]
    if a.stage == 'pilot':
        # Evidence and training programs enforce the original inner-fit scope.
        rc = run(command('build_token_competition_evidence_v2.py')+
                 ['--index',str(p['pilot_execution_ordinal'])], deadline)
        if rc: return rc
        rc = run(command('verify_token_competition_evidence_v2.py')+['--pilot'],deadline)
        if rc: return rc
        for arm in p['arms']:
            receipt = OUT/'pilot'/f'{arm}.json'
            if receipt.exists():
                saved=json.loads(receipt.read_text())
                if (saved.get('status') == 'FROZEN_BASE_REAL_QUERY_ENGINEERING_PILOT_PASS'
                        and saved.get('binding',{}).get('protocol') == protocol_binding
                        and saved.get('binding',{}).get('arm') == arm):
                    continue
            receipt.parent.mkdir(exist_ok=True)
            rc=run(command('run_token_competition_v2.py')+['--fold','0','--seed','0','--arm',arm,
                   '--pilot','--pilot-output',str(receipt)],deadline)
            if rc:return rc
        value={'status':'TOKEN_COMPETITION_REAL_PILOT_ALL3_PASS','scientific_result':False,
               'protocol':protocol_binding,
               'query_id':p['pilot_query_id'],'arms':p['arms'],
               'receipts':{arm:{'path':str(OUT/'pilot'/f'{arm}.json'),
                    'sha256':hashlib.sha256((OUT/'pilot'/f'{arm}.json').read_bytes()).hexdigest()}
                    for arm in p['arms']}}
        temp=OUT/'pilot_validation.json.tmp'
        temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,OUT/'pilot_validation.json')
        print(json.dumps(value),flush=True)
        return 0
    if a.stage == 'evidence':
        if a.index is None or not 0<=a.index<16:raise ValueError('evidence shard0..15')
        for index in range(a.index*8, (a.index+1)*8):
            rc=run(command('build_token_competition_evidence_v2.py')+['--index',str(index)],deadline)
            if rc:return rc
        return 0
    if a.stage == 'verify':
        return run(command('verify_token_competition_evidence_v2.py'), deadline)
    if a.stage == 'train':
        if a.index is None or not 0<=a.index<15:raise ValueError('training task0..14')
        fold,arm=a.index//3,p['arms'][a.index%3]
        return run(command('run_token_competition_v2.py')+['--fold',str(fold),'--seed','0','--arm',arm],deadline)
    return run(command('join_token_competition_v2.py'),deadline)


if __name__=='__main__':
    raise SystemExit(main())
