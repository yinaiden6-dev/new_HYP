#!/usr/bin/env python3
"""Sequential dev queue for unchanged original shards; preserve scientific code."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

RC = Path(__file__).resolve().parents[1]
ROOT = RC/'results/rc_m_structure_binding_isolation_v2'


def main(dry=False):
    selection=json.loads((ROOT/'dev_cpu_selection_20261001.json').read_text())
    pp=ROOT/'protocol.json';protocol=json.loads(pp.read_text())
    pb=dict(path=str(pp.resolve()),sha256=hashlib.sha256(pp.read_bytes()).hexdigest())
    inputs=json.loads((ROOT/'inputs.json').read_text())
    start=time.monotonic()
    for index in selection['indices']:
        workers=inputs['workers'][index::protocol['shards']]
        missing=[]
        for w in workers:
            p=ROOT/'pairs'/f"{w['ordinal']:03d}"/'validation.json'
            if p.exists():
                v=json.loads(p.read_text())
                assert v['status']=='ISOLATION_PAIR_COMPLETE' and v['protocol']==pb
            else:missing.append(w['ordinal'])
        print(json.dumps(dict(original_shard=index,missing_pairs=missing)),flush=True)
        if dry or not missing:continue
        remaining=int(420-(time.monotonic()-start))
        if remaining<45:return 75
        cmd=[sys.executable,'-u',str(RC/'programs/run_rc_m_structure_binding_isolation_v2.py'),
             'work','--index',str(index),'--budget',str(remaining)]
        code=subprocess.run(cmd).returncode
        if code:return code
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true')
    sys.exit(main(p.parse_args().dry_run))
