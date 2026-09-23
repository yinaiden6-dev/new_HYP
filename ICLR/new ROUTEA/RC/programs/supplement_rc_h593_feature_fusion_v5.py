#!/usr/bin/env python3
"""Cooperative rolling top-up for the unchanged fusion v3 training ledger."""
import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion_rolling_io',ROOT/'programs/run_rc_h593_packed_gpu_v2.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
OLD=ROOT/'results/rc_h593_feature_fusion_dispatch_v3'
OUT=ROOT/'results/rc_h593_feature_fusion_rolling_v5'
TRAIN=ROOT/'results/rc_h593_feature_fusion_train_v2'
AUTH=ROOT/'registry/rc_h593_feature_fusion_rolling_authority_v5_20260923.json'
TRAIN_AUTH=ROOT/'registry/rc_h593_feature_fusion_train_authority_v2_20260922.json'
LAUNCH=ROOT/'slurm/rc_h593_feature_fusion_train_v2.sbatch'
read,write,bind,checked,command=P.read,P.write,P.bind,P.checked,P.command
LIMIT=50

def status(state,**kw):
    write(OUT/'status.json',dict(status=state,time_utc=P.now(),pid=os.getpid(),host=socket.gethostname(),**kw),replace=True)

def choose(complete,active,attempts):
    assert len(active)<=LIMIT
    return sorted((i for i in range(200) if i not in complete|active),key=lambda i:(attempts[i],i))[:LIMIT-len(active)]

def prepare():
    assert not AUTH.exists()
    assert choose(set(range(14)),set(range(47,79)),Counter())[0]==14
    assert len(choose(set(),set(range(32)),Counter()))==18
    assert not choose(set(),set(range(LIMIT)),Counter())
    for n in (0,1,46,49,50):
        active=set(range(n));done=set(range(190,200));todo=choose(done,active,Counter())
        assert len(todo)==LIMIT-n and not (set(todo)&(done|active))
    assert choose(set(range(199)),set(),Counter())==[199]
    sources=[Path(__file__),Path(P.__file__),ROOT/'programs/dispatch_rc_h593_feature_fusion_v3.py',
             ROOT/'registry/rc_h593_feature_fusion_dispatch_authority_v3_20260922.json',TRAIN_AUTH,LAUNCH,
             ROOT/'plan/RC_H593_FEATURE_FUSION_ROLLING_V5_20260923.md']
    a=read(TRAIN_AUTH);assert len(a['configs'])==200
    write(AUTH,dict(status='FUSION_ROLLING_TOPUP_AUTHORIZED',sources=[bind(p) for p in sources],train_authority=bind(TRAIN_AUTH),
         maximum_active=LIMIT,config_count=200,max_chunks=128,maximum_hours=48,minutes=13,
         user_instruction='Raise 5158421 and its rolling fusion continuations to 50; assess shorter limits using measured runtimes',
         scientific_change=False,ledger=str(OLD),original_dispatcher='Continues unchanged; recognizes supplemented waves through shared ledger',
         coordination='Temporarily hold one verified pending recorded task during submission; its existence prevents the legacy empty-wave branch. Release only this hold afterward. Running tasks untouched.',
         no_pending_anchor='Do not race the legacy dispatcher: wait for its next wave when no recorded pending task can protect the transaction'))
    write(OUT/'preflight.json',dict(status='ROLLING_CAP_AND_EXCLUSION_PASS',authority=bind(AUTH)))

def waves():return [(p,read(p)) for p in sorted(OLD.glob('wave[0-9][0-9][0-9][0-9].json'))]

def accounting(jobs):
    if not jobs:return {}
    text=command(['sacct','-X','-n','-P','-j',','.join(sorted(jobs)),'--format=JobID,State,ExitCode']).stdout
    return {v[0]:dict(state=v[1],exit=v[2]) for line in text.splitlines() if len(v:=line.split('|'))>=3}

def inspect(rows):
    queued={r['job']:r for r in rows};active=set();attempts=Counter();unknown=[];complete=set();owned={}
    for i in range(200):
        p=TRAIN/f'fit{i:03d}/validation.json'
        if p.exists():
            v=read(p);assert v['status']=='FUSION_FOLD_NUMPY_READOUT_PASS' and v['authority']==bind(TRAIN_AUTH)
            checked(v['payload']);complete.add(i)
    for path,w in waves():
        for i in w['indices']:
            attempts[i]+=1;key=f"{w['job_id']}_{i}";owned[key]=i
            if key in queued:
                assert i not in active,('DUPLICATE_ACTIVE_CONFIG',i)
                active.add(i)
            elif not (OUT/'retired'/f'{key}.json').exists():unknown.append((key,i,w['job_id']))
    acc=accounting({j for key,i,j in unknown})
    for key,i,job in unknown:
        item=acc.get(key)
        if not item or item['state'] in ('RUNNING','PENDING','COMPLETING'):
            active.add(i);continue
        assert item['state']=='COMPLETED' and item['exit']=='0:0',('FAILED_TRAIN_CHUNK',key,item)
        p=TRAIN/f'fit{i:03d}/chunks/{job}_{i}.json'
        if not p.exists():active.add(i);continue
        rec=read(p);assert rec['status']=='FUSION_CHUNK_NORMAL_EXIT' and rec['authority']==bind(TRAIN_AUTH) and rec['config_index']==i
        # Checkpoint files advance on resume; retain old receipt bindings without
        # falsely requiring its historical checkpoint hash to remain current.
        write(OUT/'retired'/f'{key}.json',dict(accounting=item,receipt=bind(p),index=i,step=rec['step'],complete=rec['complete']))
    assert len(active)<=LIMIT
    return complete,active,attempts,owned

def ensure_throttles(rows,owned):
    # Includes any future waves submitted by the unchanged legacy dispatcher.
    roots={r['job'].split('_')[0]:r['job'] for r in rows if r['job'] in owned}
    for root,example in roots.items():
        receipt=OUT/'throttles'/f'{root}.json'
        if receipt.exists():continue
        before=P.jobinfo(example)
        assert before['JobName']=='h593_ftrain' and before['UserId'].startswith('ap7811(')
        assert before['ArrayJobId']==root
        if before.get('ArrayTaskThrottle')!=str(LIMIT):
            command(['scontrol','update',f'JobId={root}',f'ArrayTaskThrottle={LIMIT}'])
        after=P.jobinfo(example)
        assert after['ArrayTaskThrottle']==str(LIMIT)
        for k in ('TimeLimit','ReqTRES','Partition','Dependency','CPUs/Task'):
            assert before.get(k)==after.get(k),(root,k)
        write(receipt,dict(array_job=root,before=before,after=after,time_utc=P.now()))

def release_anchor(path):
    x=read(path);job=x['job'];state=P.jobinfo(job)
    if state['JobState']=='PENDING' and state.get('Reason')=='JobHeldUser':
        assert state['JobName']=='h593_ftrain' and state['UserId'].startswith('ap7811(')
        command(['scontrol','release',job]);after=P.jobinfo(job);assert after.get('Reason')!='JobHeldUser'
        for k in ('TimeLimit','ReqTRES','Partition'):assert after.get(k)==x['before'].get(k)
    write(path.with_name(path.stem+'_released.json'),dict(job=job,time_utc=P.now()))

def recover():
    for p in (OUT/'anchors').glob('*.json'):
        if p.stem.endswith('_released') or p.with_name(p.stem+'_released.json').exists():continue
        release_anchor(p)

def tick():
    if (TRAIN/'validation.json').exists():
        v=read(TRAIN/'validation.json');assert v['status']=='FUSION_JOIN_COUNTS_AND_GROUPS_PASS';checked(v['result'])
        status('COMPLETE',validation=bind(TRAIN/'validation.json'));return True
    rows=P.queue();complete,active,attempts,owned=inspect(rows)
    ensure_throttles(rows,owned)
    if len(complete)==200:
        p=OLD/'join_submitted.json'
        if not p.exists():
            # Legacy controller owns this join while its heartbeat is current.
            legacy=read(OLD/'status.json')
            status('ALL200_READY_WAITING_LEGACY_JOIN',complete_configs=200,legacy=legacy);return False
        status('FINAL_JOIN_QUEUED',complete_configs=200,job=read(p)['job_id']);return False
    todo=choose(complete,active,attempts)
    assert all(attempts[i]<128 for i in todo)
    candidates=[r for r in rows if r['job'] in owned and r['name']=='h593_ftrain' and r['state']=='PENDING' and r['partition']=='accelerated' and r['reason'] not in ('JobHeldUser','JobHeldAdmin','Dependency','DependencyNeverSatisfied')]
    if not todo or not candidates:
        status('ROLLING_ACTIVE' if not todo else 'WAITING_SAFE_SUBMISSION_WINDOW',complete_configs=len(complete),active_configs=len(active),capacity=LIMIT-len(active));return False
    # The old dispatcher only submits when *no* recorded wave remains queued.
    # Hold one already-pending task for the short transaction, so that predicate
    # cannot become true while we append a new wave atomically.
    anchor=max(candidates,key=lambda r:tuple(map(int,r['job'].split('_'))))['job'];before=P.jobinfo(anchor)
    if before['JobState']!='PENDING':return False
    assert before['UserId'].startswith('ap7811(') and before['JobName']=='h593_ftrain'
    ap=OUT/'anchors'/f'{time.time_ns()}.json';write(ap,dict(job=anchor,before=before))
    command(['scontrol','hold',anchor])
    try:
        state=P.jobinfo(anchor);assert state['JobState']=='PENDING' and state.get('Reason')=='JobHeldUser'
        # Refresh after the protected window begins; don't use a stale capacity.
        complete,active,attempts,owned=inspect(P.queue());todo=choose(complete,active,attempts)
        if not todo:return False
        assert all(attempts[i]<128 for i in todo)
        args=['--hold','--partition=accelerated','--time=00:13:00','--array='+','.join(map(str,todo))+f'%{LIMIT}',str(LAUNCH),'fit']
        job=command(['sbatch','--parsable',*args]).stdout.strip().split(';')[0];assert job.isdigit()
        spool=OUT/'spools'/f'{job}.sh';spool.parent.mkdir(parents=True,exist_ok=True)
        command(['scontrol','write','batch_script',job,str(spool)]);assert spool.read_bytes()==LAUNCH.read_bytes()
        info=P.jobinfo(job);assert info['Partition']=='accelerated' and info['TimeLimit']=='00:13:00' and info['CPUs/Task']=='8'
        assert 'gres/gpu=1' in info['ReqTRES'] and 'mem=64G' in info['ReqTRES']
        assert info['ArrayTaskThrottle']==str(LIMIT)
        number=max(int(p.stem[4:]) for p,_ in waves())+1
        value=dict(job_id=job,args=args,authority=bind(AUTH),stage='FUSION_ROLLING_RESUME',indices=todo,
                   spool=bind(spool),verified_resources=info,time_utc=P.now(),temporary_anchor=anchor)
        write(OLD/f'wave{number:04d}.json',value);write(OUT/'waves'/f'{number:04d}.json',value)
        command(['scontrol','release',job]);assert P.jobinfo(job).get('Reason')!='JobHeldUser'
        status('ROLLING_TOPUP_SUBMITTED',complete_configs=len(complete),previous_active=len(active),submitted=job,indices=todo,maximum_active=LIMIT)
        print(dict(submitted=job,indices=todo,active_before=len(active)),flush=True)
    finally:release_anchor(ap)
    return False

def watch():
    a=read(AUTH)
    for b in a['sources']:checked(b)
    lock=(OUT/'watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);recover()
    deadline=time.monotonic()+a['maximum_hours']*3600;errors=0
    while time.monotonic()<deadline:
        try:
            if tick():return
            errors=0
        except AssertionError as e:
            status('STOPPED_VALIDATION_FAILURE',error=repr(e));raise
        except Exception as e:
            errors+=1;status('TRANSIENT_ERROR',error=repr(e),consecutive=errors)
            if errors>=5:raise
        time.sleep(30)
    status('STOPPED_48H_LIMIT',queued_training_preserved=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','watch'));a=ap.parse_args()
    prepare() if a.stage=='prepare' else watch()
