#!/usr/bin/env python3
"""Pack CPU exports to stay below the project submit limit; resumable by query."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h128_unified_resume_v1'

if __name__=='__main__':
    started=time.monotonic()
    a=json.loads((OUT/'authority.json').read_text())
    execution=json.loads((OUT/'export_pack_authority.json').read_text())
    for b in execution['sources']:
        assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
    index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    assert index in range(execution['packs'])
    for query in a['indices'][index::execution['packs']]:
        if time.monotonic()-started>420:sys.exit(75)
        cmd=['/usr/bin/python3',str(ROOT/'programs/run_rc_unified128_resume_v1.py'),'export','--index',str(query)]
        p=subprocess.Popen(cmd,start_new_session=True)
        try:code=p.wait(timeout=max(1,515-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            sys.exit(75)
        if code:sys.exit(code)
        print(dict(event='CPU_EXPORT_QUERY_DONE',pack=index,index=query),flush=True)
