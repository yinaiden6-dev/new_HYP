#!/usr/bin/env python3
"""Bounded execution-only continuation of the existing, validated acquisition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h128_unified_resume_v1'
AUTH=OUT/'authority.json'
CACHE=ROOT/'cache/rc_h593_unified_acquisition_v1'
OLD_AUTH=ROOT/'registry/rc_h593_unified_acquisition_authority_v1_20260923.json'
FAMILIES={'inside':'rc_h593_m_inside_v1','visual':'rc_h593_m_visual_origin_v1','coordinate':'rc_h593_roma_coordinate_precision_v2'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,indent=2)+'\n'
    if p.exists():assert p.read_text()==text;return
    t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');t.write_text(text);os.replace(t,p)
def complete(i):
    return all((ROOT/'results'/name/f'query{i:03d}/validation.json').exists() for name in FAMILIES.values())

def prepare():
    assert not AUTH.exists()
    a=read(OLD_AUTH)
    for b in [*a['sources'],a['workers'],a['profile']]:assert sha(b['path'])==b['sha256']
    v=read(CACHE/'query070/export_validation.json')
    assert v['status']=='UNIFIED_CPU_EXPORT_ORIGINAL_VALIDATORS_PASS'
    for b in v['validations'].values():assert sha(b['path'])==b['sha256']
    done=[i for i in range(128) if complete(i)]
    indices=[i for i in range(128) if i not in done]
    sources=[Path(__file__),ROOT/'programs/run_rc_unified128_collect_chunk_v1.py',ROOT/'slurm/rc_unified128_resume_gpu_v1.sbatch',ROOT/'slurm/rc_unified128_resume_cpu_v1.sbatch']
    write(AUTH,dict(status='UNIFIED128_RESUME_AUTHORIZED',user_request='Resume unified acquisition and held fusion; retain prior 128-query acquisition cap',
        original_authority=bind(OLD_AUTH),pilot_export=bind(CACHE/'query070/export_validation.json'),
        sources=[bind(p) for p in sources],indices=indices,already_all_families_complete=done,
        scientific_changes=0,query_cap=128,max_parallel=46,max_restarts=31,
        reuse='Original per-family parts and per-pair unified capsules; no completed native forwards repeated'))
    print(dict(status='PREPARED',additional_queries=len(indices),indices=indices),flush=True)

def main(stage,index):
    a=read(AUTH)
    assert os.environ.get('SLURM_JOB_ID')
    for b in [a['original_authority'],*a['sources']]:assert sha(b['path'])==b['sha256']
    if stage=='join':
        assert all(complete(i) for i in range(128))
        validations={f:[bind(ROOT/'results'/name/f'query{i:03d}/validation.json') for i in range(128)] for f,name in FAMILIES.items()}
        for vs in validations.values():
            for b in vs:
                v=read(b['path']);assert v['status'].endswith('PASS');assert sha(v['payload']['path'])==v['payload']['sha256']
        write(OUT/'validation.json',dict(status='UNIFIED128_THREE_FAMILIES_COMPLETE',authority=bind(AUTH),queries=128,validations=validations))
        return 0
    assert index in a['indices']
    if complete(index):return 0
    if stage=='collect' and (CACHE/f'query{index:03d}/ready.json').exists():return 0
    cmd=([str(ROOT.parents[2]/'.venv-romav2/bin/python'),str(ROOT/'programs/run_rc_unified128_collect_chunk_v1.py'),str(index)]
         if stage=='collect' else
         [str(ROOT.parents[2]/'.venv-romav2/bin/python'),str(ROOT/'programs/run_rc_h593_unified_acquisition_v1.py'),stage,'--index',str(index)])
    result=subprocess.run(cmd)
    if result.returncode:return result.returncode
    if stage=='collect':return 0 if (CACHE/f'query{index:03d}/ready.json').exists() else 75
    assert complete(index)
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect','export','join']);p.add_argument('--index',type=int);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:sys.exit(main(args.stage,args.index))
