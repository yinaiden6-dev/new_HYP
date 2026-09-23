#!/usr/bin/env python3
"""Bounded CPU arrays with dependency callbacks; no polling daemon or GPU jobs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_h71_feature_fusion_pilot_v1'
AUTH=ROOT/'registry/rc_h71_feature_fusion_pilot_authority_v1_20260923.json'
LAUNCH=ROOT/'slurm/rc_h71_feature_fusion_pilot_v1.sbatch'
CALLBACK=ROOT/'slurm/rc_h71_feature_fusion_follow_v1.sbatch'
DISPATCH=OUT/'dispatch'


def read(path):return json.loads(Path(path).read_text())
def bind(path):
    path=Path(path).resolve()
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return dict(path=str(path),sha256=h.hexdigest())
def checked(b):
    assert bind(b['path'])==b,('SHA_DRIFT',b['path'])
    return Path(b['path'])
def write(path,value,mutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and not mutable:
        assert path.read_text()==text,('IMMUTABLE',str(path));return
    tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp');tmp.write_text(text);os.replace(tmp,path)
def command(args):
    p=subprocess.run(list(map(str,args)),cwd=ROOT,text=True,capture_output=True,timeout=45)
    if p.returncode:raise RuntimeError(str(args)+' '+p.stderr)
    return p.stdout.strip()
def info(job):
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)',command(['scontrol','show','job','-o',job])))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def status(state,**kw):write(DISPATCH/'status.json',dict(status=state,time_utc=now(),**kw),mutable=True)


def submit(script,args,extra=()):
    argv=['sbatch','--parsable','--hold','--partition=cpuonly,dev_cpuonly',*extra,str(script),*map(str,args)]
    job=command(argv).split(';')[0];assert job.isdigit(),job
    # Preserve a receipt immediately, even if a later resource/spool check fails.
    receipt=DISPATCH/'submissions'/f'{job}.json'
    write(receipt,dict(job=job,argv=argv,time_utc=now(),submitted_held=True))
    spool=DISPATCH/'spools'/f'{job}.sh';spool.parent.mkdir(parents=True,exist_ok=True)
    command(['scontrol','write','batch_script',job,str(spool)])
    assert spool.read_bytes()==script.read_bytes(),'SUBMITTED_SPOOL_MISMATCH'
    d=info(job)
    assert d['UserId'].startswith('ap7811(') and 'gres/gpu' not in d['ReqTRES'],'CPU_ONLY'
    assert set(d['Partition'].split(','))=={'cpuonly','dev_cpuonly'},d
    expected=('8','32G','00:10:00') if script==LAUNCH else ('1','2G','00:05:00')
    assert d['CPUs/Task']==expected[0] and d['MinMemoryNode']==expected[1] and d['TimeLimit']==expected[2],d
    return dict(job=job,args=argv,resources=d,spool=bind(spool))


def release(j):
    d=info(j['job'])
    assert d['JobState']=='PENDING' and d['Reason']=='JobHeldUser',d
    command(['scontrol','release',j['job']])


def enqueue(a,wave,indices):
    assert 0<=wave<a['max_waves'] and len(indices)<=50 and len(indices)==len(set(indices))
    j=submit(LAUNCH,['fit'],['--array='+','.join(map(str,indices))+'%50'])
    record=dict(wave=wave,indices=indices,training=j,authority=bind(AUTH))
    write(DISPATCH/f'wave{wave:03d}.json',record)
    callback=submit(CALLBACK,['follow','--wave',wave],['--dependency=afterany:'+j['job']])
    write(DISPATCH/f'wave{wave:03d}_callback.json',callback)
    release(callback)
    release(j)
    status('CPU_TRAINING_QUEUED',wave=wave,configs=len(indices),training_job=j['job'],callback_job=callback['job'],indices=indices)
    print(json.dumps(dict(training=j['job'],follow=callback['job'],configs=len(indices),wave=wave)),flush=True)


def complete(a):
    done=[]
    for i,cfg in enumerate(a['configs']):
        path=OUT/f'fit{i:03d}/validation.json'
        if not path.exists():continue
        v=read(path);p=read(checked(v['payload']))
        assert v['status']=='H71_FOLD_NUMPY_PASS' and v['authority']==p['authority']==bind(AUTH) and p['config']==cfg
        done.append(i)
    return done


def follow(a,wave):
    w=read(DISPATCH/f'wave{wave:03d}.json');job=w['training']['job']
    assert w['authority']==bind(AUTH)
    receipts=[]
    for i in w['indices']:
        path=OUT/f'fit{i:03d}/chunks/{job}_{i}.json'
        assert path.exists(),('MISSING_NORMAL_CHUNK_RECEIPT',job,i)
        p=read(path)
        assert p['status']=='H71_CHUNK_NORMAL_EXIT' and p['authority']==bind(AUTH) and p['index']==i and p['array_job']==job
        receipts.append(bind(path))
    done=complete(a)
    write(DISPATCH/f'wave{wave:03d}_closed.json',dict(receipts=receipts,completed=done))
    if len(done)==len(a['configs']):
        j=submit(LAUNCH,['join'])
        write(DISPATCH/'join.json',j)
        cb=submit(CALLBACK,['finish'],['--dependency=afterany:'+j['job']])
        write(DISPATCH/'join_callback.json',cb)
        release(cb);release(j)
        status('ALL50_READY_JOIN_QUEUED',job=j['job'],callback=cb['job'])
    else:enqueue(a,wave+1,[i for i in range(len(a['configs'])) if i not in done])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('start','follow','finish'));ap.add_argument('--wave',type=int)
    args=ap.parse_args();DISPATCH.mkdir(parents=True,exist_ok=True)
    with (DISPATCH/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a=read(AUTH)
        assert a['execution_device']=='cpu' and a['new_gpu_jobs']==0 and a['steps']==240
        for b in a['sources']:checked(b)
        try:
            if args.stage=='start':
                assert not list(DISPATCH.glob('wave[0-9][0-9][0-9].json')),'ALREADY_SUBMITTED'
                enqueue(a,0,list(range(50)))
            elif args.stage=='follow':follow(a,args.wave)
            else:
                v=read(OUT/'validation.json')
                assert v['status']=='H71_JOIN_COUNTS_AND_GROUPS_PASS' and v['authority']==bind(AUTH)
                checked(v['result']);status('COMPLETE',validation=bind(OUT/'validation.json'),queries=71,configs=50)
        except Exception as e:
            status('STOPPED_NEEDS_REVIEW',stage=args.stage,wave=args.wave,error=repr(e));raise


if __name__=='__main__':main()
