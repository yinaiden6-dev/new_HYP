#!/usr/bin/env python3
"""Scheduler-only continuation for the frozen coarse-exit experiment."""
from collections import Counter
import fcntl,hashlib,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'results/rc_h593_coarse_exit_v1'
OUT=ROOT/'results/rc_h593_coarse_exit_dispatch_v1'
AUTH=ROOT/'registry/rc_h593_coarse_exit_dispatch_authority_v1_20260923.json'
CONTROL=ROOT/'slurm/rc_h593_coarse_exit_control_v1.sbatch'
GPU=ROOT/'slurm/rc_h593_coarse_exit_gpu_v1.sbatch'
CPU=ROOT/'slurm/rc_h593_coarse_exit_cpu_v1.sbatch'
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(x,indent=2,ensure_ascii=False)+'\n'
 if p.exists():assert p.read_text()==s;return
 t=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');t.write_text(s);os.link(t,p);t.unlink()
def cmd(args):return subprocess.run(list(map(str,args)),cwd=ROOT,text=True,capture_output=True,check=True)
def info(job):return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=(\S*)',cmd(['scontrol','show','job','-o',job]).stdout))
def submit(script,stage,indices=None,dependency=None):
 args=['sbatch','--parsable','--hold','--partition='+('accelerated' if script==GPU else 'cpuonly')]
 if indices is not None:args+=['--array='+','.join(map(str,indices))+'%50']
 if dependency:args+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
 args += [str(script),stage];job=cmd(args).stdout.strip().split(';')[0];assert job.isdigit()
 spool=OUT/'spools'/f'{job}.sh';spool.parent.mkdir(parents=True,exist_ok=True);cmd(['scontrol','write','batch_script',job,spool]);assert spool.read_bytes()==script.read_bytes()
 z=info(job);assert z['TimeLimit']=='00:10:00' and z['CPUs/Task']=='8'
 if script==GPU:assert 'gres/gpu=1' in z['ReqTRES'] and 'mem=64G' in z['ReqTRES']
 else:assert 'gres/gpu' not in z['ReqTRES'] and z['Partition']=='cpuonly'
 return dict(job_id=job,args=args,spool=bind(spool),verified=z,stage=stage,indices=indices or [])
def control(previous):
 a=read(AUTH)
 for b in a['sources']:assert bind(b['path'])==b
 assert os.environ.get('SLURM_JOB_ID')
 lock=(OUT/'control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (EXP/'validation.json').exists():return
 acc=cmd(['sacct','-X','-n','-P','-j',previous,'--format=State,ExitCode']).stdout.splitlines()
 assert acc and all(x.split('|')[:2]==['COMPLETED','0:0'] for x in acc),acc
 waves=[read(p) for p in sorted((OUT/'waves').glob('*.json'))]
 counts=Counter(i for w in waves for i in w['indices'])
 if not (EXP/'probe_validation.json').exists():
  assert sum(w['stage']=='probe' for w in waves)<12
  nextjob=submit(GPU,'probe')
 else:
  p=read(EXP/'probe_validation.json');assert p['status']=='COARSE_EARLY_EXIT_BITS_AND_ZERO_REFINER_PASS' and bind(p['payload']['path'])==p['payload']
  complete=set()
  for i in range(70):
   q=EXP/f'query{i:03d}/validation.json'
   if q.exists():
    v=read(q);assert v['status']=='COARSE_FIXED_COST1_CPU_PASS' and bind(v['payload']['path'])==v['payload'];complete.add(i)
  if len(complete)==70:
   nextjob=submit(CPU,'join');write(OUT/'join_submitted.json',nextjob);cmd(['scontrol','release',nextjob['job_id']]);return
  indices=[0] if 0 not in complete else sorted(set(range(70))-complete,key=lambda i:(counts[i],i))[:50]
  assert all(counts[i]<12 for i in indices)
  nextjob=submit(CPU,'worker',indices)
 callback=submit(CONTROL,nextjob['job_id'],dependency=nextjob['job_id'])
 nextjob['callback']=callback;nextjob['authority']=bind(AUTH)
 write(OUT/'waves'/f'{len(waves):04d}.json',nextjob)
 for job in (callback['job_id'],nextjob['job_id']):cmd(['scontrol','release',job]);assert info(job).get('Reason')!='JobHeldUser'
 print(json.dumps(nextjob))
if __name__=='__main__':control(sys.argv[1])
