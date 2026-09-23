#!/usr/bin/env python3
"""Continue the same M acquisitions with validated native-stage reuse."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import socket
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('old_m_dispatch',ROOT/'programs/run_rc_h593_m_priority_dispatch_v1.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
P=M.P
OLD=M.OUT
OUT=ROOT/'results/rc_h593_m_stage_reuse_dispatch_v1'
AUTH=ROOT/'registry/rc_h593_m_stage_reuse_dispatch_authority_v1_20260923.json'
LAUNCH=ROOT/'slurm/rc_h593_shared_stages_v1.sbatch'
QUAL=ROOT/'cache/rc_h593_shared_stages_v1/qualification.json'
read,write,bind,checked=P.read,P.write,P.bind,P.checked

def choose(done,active,excluded,capacity,attempts):
    pools={'inside':[i for i in range(593) if i not in done['inside'] and ('inside',i) not in active|excluded],
           'visual':[i for i in range(593) if i in done['inside'] and i not in done['visual'] and ('visual',i) not in active|excluded]}
    answer=[]
    while len(answer)<capacity and any(pools.values()):
        for f in ('inside','inside','visual'):
            if pools[f] and len(answer)<capacity:
                i=pools[f].pop(0);assert attempts[f,i]<32;answer.append((f,i))
    assert len(answer)==len(set(answer)) and not set(answer)&(active|excluded)
    return answer

def prepare():
    assert not AUTH.exists()
    sources=[Path(__file__),Path(M.__file__),LAUNCH,ROOT/'registry/rc_h593_m_priority_dispatch_authority_v1_20260923.json',
             ROOT/'registry/rc_h593_shared_stages_authority_v1_20260923.json']
    for b in read(sources[-2])['sources']:checked(b)
    for b in read(sources[-1])['sources']:checked(b)
    from collections import Counter
    done={'inside':{0,1,2},'visual':{0}}
    x=choose(done,{('visual',1)},set(),46,Counter())
    assert ('visual',2) in x and not any(f=='visual' and i not in done['inside'] for f,i in x)
    write(AUTH,dict(status='M_PRIORITY_SHORT_ARRAYS_AUTHORIZED',sources=[bind(p) for p in sources],maximum_days=14,
          families=['inside','visual'],maximum_active_gpus=46,minutes=12,
          user_instruction='Collect missing GPU stage data once; downstream CPU reuse; prioritize reuse of complete results',
          visual_prerequisite='All128 inside native stages sealed; exact replay qualification required',
          old_active_arrays='Allow to finish unchanged, no cancellation or duplicate query writers',
          qualification_path=str(QUAL),scientific_protocol_unchanged=True))
    write(OUT/'preflight.json',dict(status='NATIVE_REUSE_DEPENDENCY_SELECTION_PASS',authority=bind(AUTH)))

def handoff():
    proc=read(OLD/'supervisor_process.json')
    assert proc['host']==socket.gethostname()
    path=Path(f"/proc/{proc['pid']}/cmdline")
    if path.exists() and path.read_bytes():
        argv=[x.decode() for x in path.read_bytes().split(b'\0') if x]
        assert argv==proc['command'] and path.stat().st_uid==os.getuid(),argv
        os.kill(proc['pid'],signal.SIGTERM)
        for _ in range(50):
            if not path.exists() or not path.read_bytes():break
            time.sleep(.1)
        assert not path.exists() or not path.read_bytes()
    lock=(OLD/'watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('waves','retired'):
        for p in (OLD/name).glob('*.json'):write(OUT/name/p.name,read(p))
    write(OUT/'takeover.json',read(OLD/'takeover.json'))
    for p in [*OLD.glob('*_join.json'),*OLD.glob('readout_released.json')]:write(OUT/p.name,read(p))
    write(OUT/'handoff.json',dict(time_utc=P.now(),stopped=proc,qualification=bind(QUAL),preserved_active_arrays=True))
    return lock

def watch():
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'start.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a=read(AUTH)
    for b in a['sources']:checked(b)
    pilot=read(QUAL.with_name('qualification_submitted.json'))['job']
    deadline=time.monotonic()+24*3600
    while not QUAL.exists():
        write(OUT/'status.json',dict(status='WAITING_FOR_NATIVE_STAGE_REPLAY_QUALIFICATION',time_utc=P.now(),
                                   pid=os.getpid(),host=socket.gethostname(),pilot=pilot),replace=True)
        acc=M.accounting([pilot]).get(pilot)
        if acc and acc['state'] in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL'):
            write(OUT/'status.json',dict(status='STOPPED_QUALIFICATION_FAILURE',accounting=acc),replace=True);return
        assert time.monotonic()<deadline,'QUALIFICATION_WAIT_24H_LIMIT'
        time.sleep(20)
    q=read(QUAL);assert q['status']=='SHARED_NATIVE_STAGES_FULL128_AND_GPU_PASS'
    assert q['authority']==bind(ROOT/'registry/rc_h593_shared_stages_authority_v1_20260923.json')
    acc=M.accounting([pilot]).get(pilot)
    while acc and acc['state'] in ('RUNNING','COMPLETING'):
        time.sleep(10);acc=M.accounting([pilot]).get(pilot)
    assert acc and acc['state']=='COMPLETED' and acc['exit']=='0:0',acc
    old_lock=handoff()
    M.OUT,M.AUTH,M.LAUNCH,M.choose=OUT,AUTH,LAUNCH,choose
    old_event=M.event
    def event(state,**kw):
        kw.pop('deferred',None)
        old_event(state,**kw,native_reuse=True,visual_requires_inside=True,fusion_resumed_separately=True)
    M.event=event
    M.watch()
    assert old_lock

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','watch'));args=ap.parse_args()
    prepare() if args.stage=='prepare' else watch()
