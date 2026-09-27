#!/usr/bin/env python3
"""Keep per-restart receipts distinct; all original scientific code is reused."""
import os
from pathlib import Path
import sys
import run_rc_h593_unified_acquisition_v1 as original

original_write=original.write
def write(p,value,*args,**kwargs):
    p=Path(p)
    if p.parent.name=='chunks' and p.name==os.environ['SLURM_JOB_ID']+'.json':
        attempt=int(os.environ.get('SLURM_RESTART_COUNT','0'))
        p=p.with_name(f"{os.environ['SLURM_JOB_ID']}_restart{attempt:02d}.json")
        value=dict(value,slurm_job_id=os.environ['SLURM_JOB_ID'],restart=attempt)
    return original_write(p,value,*args,**kwargs)

if __name__=='__main__':
    original.write=write
    original.collect(int(sys.argv[1]))
