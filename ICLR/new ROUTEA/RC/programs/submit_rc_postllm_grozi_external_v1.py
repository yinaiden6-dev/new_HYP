#!/usr/bin/env python3
"""Submit only the authorized final GroZi transfer, with exact dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import cache_rc_postllm_h593_v1 as C

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_postllm_grozi_external_v1'
AUTH=ROOT/'registry/rc_postllm_grozi_external_authority_v1_20260925.json'
LAUNCH=ROOT/'slurm/rc_postllm_grozi_external_v1.sbatch'


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot-partition',default='dev_accelerated')
    p.add_argument('--partition',default='accelerated');p.add_argument('--join-partition',default='cpuonly,dev_cpuonly')
    args=p.parse_args()
    a=C.read(AUTH)
    for b in a['code_sources']:C.checked(b)
    receipt=OUT/'jobs.json';jobs=C.read(receipt) if receipt.exists() else {'authority':C.bind(AUTH),'jobs':{}}
    C.need(jobs['authority']==C.bind(AUTH),'RESUME_SUBMISSION_AUTHORITY')
    def submit(role,options):
        if role in jobs['jobs']:return jobs['jobs'][role]['job_id']
        cmd=['sbatch','--parsable',*options,str(LAUNCH),role]
        result=subprocess.run(cmd,check=True,text=True,capture_output=True)
        jid=result.stdout.strip().split(';')[0];C.need(jid.isdigit(),'SBATCH_JOB_ID')
        jobs['jobs'][role]={'job_id':jid,'command':cmd,'stderr':result.stderr}
        C.write(receipt,jobs);print(role,jid,flush=True);return jid
    first=submit('pilot',['--partition='+args.pilot_partition])
    second=submit('worker',['--partition='+args.partition,'--array=0-11%12','--dependency=afterok:'+first])
    submit('join',['--partition='+args.join_partition,'--gres=none','--mem=16G',
        '--dependency=afterok:'+first+':'+second])


if __name__=='__main__':main()
