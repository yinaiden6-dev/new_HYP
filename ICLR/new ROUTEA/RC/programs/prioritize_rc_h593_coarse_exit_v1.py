#!/usr/bin/env python3
"""Temporarily prioritize only the requested early-exit chain, then restore holds."""
import argparse,fcntl,importlib.util,os,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('priority_io',ROOT/'programs/run_rc_h593_packed_gpu_v2.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
read,write,bind,checked,command=P.read,P.write,P.bind,P.checked,P.command
OUT=ROOT/'results/rc_h593_coarse_exit_priority_v1'
EXP=ROOT/'results/rc_h593_coarse_exit_v1'
CHAIN=ROOT/'results/rc_h593_coarse_exit_dispatch_v1'
FUSION=ROOT/'results/rc_h593_feature_fusion_dispatch_v3'
AUTH=ROOT/'registry/rc_h593_coarse_exit_priority_authority_v1_20260923.json'
FIELDS=('ReqTRES','CPUs/Task','TimeLimit','Dependency','ArrayTaskThrottle')
def status(name,**kw):write(OUT/'status.json',dict(status=name,pid=os.getpid(),host=socket.gethostname(),time_utc=P.now(),**kw),replace=True)
def log(value):write(OUT/'events'/f'{time.time_ns()}.json',dict(time_utc=P.now(),**value))
def same(before,after):
 for k in FIELDS:assert before.get(k)==after.get(k),(k,before.get(k),after.get(k))
def move(job,partition):
 before=P.jobinfo(job);assert before['JobState']=='PENDING' and before['UserId'].startswith('ap7811(')
 command(['scontrol','update','JobId='+job,'Partition='+partition]);after=P.jobinfo(job)
 assert after['Partition']==partition;same(before,after);log(dict(event='PARTITION_VERIFIED',job=job,before=before,after=after))
def targets():
 answer={}
 for path in sorted((CHAIN/'waves').glob('*.json')):
  w=read(path);answer[str(w['job_id'])]=w['stage']
  if 'callback' in w:answer[str(w['callback']['job_id'])]='controller'
 p=CHAIN/'join_submitted.json'
 if p.exists():answer[str(read(p)['job_id'])]='join'
 return answer
def apply():
 for b in read(AUTH)['sources']:checked(b)
 roots={str(read(p)['job_id']) for p in FUSION.glob('wave*.json')}
 rows=P.queue();scope=[r for r in rows if r['state']=='PENDING' and r['reason'] not in ('JobHeldUser','JobHeldAdmin','Dependency','DependencyNeverSatisfied') and ((r['job'].split('_')[0] in roots and r['name']=='h593_ftrain') or (r['job']=='5158898' and r['name']=='h593_uqual'))]
 for row in scope:
  job=row['job'];before=P.jobinfo(job)
  if before['JobState']!='PENDING':continue
  assert before['UserId'].startswith('ap7811(') and before.get('Reason') not in ('JobHeldUser','JobHeldAdmin')
  record=OUT/'holds'/f'{job}.json';write(record,dict(job=job,before=before))
  command(['scontrol','hold',job]);after=P.jobinfo(job);assert after['JobState']=='PENDING' and after['Reason']=='JobHeldUser';same(before,after)
  # Pending held dev jobs still consume submission slots; move only these to normal.
  if 'dev_' in after['Partition']:move(job,'accelerated')
 log(dict(event='REQUESTED_CHAIN_PRIORITIZED',held=[r['job'] for r in scope]))
 before=P.jobinfo('5158972')
 if before['JobState']=='PENDING':move('5158972','cpuonly,dev_cpuonly')
 status('EARLY_CHAIN_PRIORITIZED',held=len(scope),probe='5158971',callback='5158972')
def restore(reason):
 restored=[]
 for p in sorted((OUT/'holds').glob('*.json')):
  done=OUT/'restored'/p.name
  if done.exists():continue
  x=read(p);job=x['job'];res=command(['scontrol','show','job','-o',job],check=False)
  if res.returncode:
   write(done,dict(job=job,skipped='Job no longer in controller',reason=reason));continue
  info=P.fields(res.stdout)
  if info['JobState']=='PENDING' and info.get('Reason')=='JobHeldUser':
   if info['Partition']!=x['before']['Partition']:move(job,x['before']['Partition'])
   command(['scontrol','release',job]);after=P.jobinfo(job);same(x['before'],after);assert after.get('Reason')!='JobHeldUser'
   restored.append(job)
  write(done,dict(job=job,reason=reason,time_utc=P.now()))
 log(dict(event='TEMPORARY_HOLDS_RESTORED',reason=reason,restored=restored));return restored

def tick():
 p=EXP/'validation.json'
 if p.exists():
  v=read(p);assert v['status']=='COARSE_CACHED70_RECOUNT_PASS' and v['all70_queries_validated'];checked(v['result'])
  restored=restore('EARLY_CHAIN_COMPLETE');status('COMPLETE_RESTORED',result=v['result'],restored=restored);return True
 owned=targets();rows=P.queue();scope=[r for r in rows if r['job'].split('_')[0] in owned]
 # Detect actual failure, not ordinary dependency/resource waits.
 acct=command(['sacct','-X','-n','-P','-j',','.join(sorted(owned)),'--format=JobID,State,ExitCode']).stdout
 bad=[]
 for line in acct.splitlines():
  f=line.split('|')
  if len(f)>=3 and f[1].split()[0] in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE'):bad.append(f[:3])
 if bad:
  restored=restore('EARLY_CHAIN_FAILED');status('STOPPED_EARLY_CHAIN_FAILURE',failures=bad,restored=restored);return True
 dev=[r for r in rows if 'dev_' in r['partition']]
 eligible=[r for r in scope if r['state']=='PENDING' and r['reason'] not in ('JobHeldUser','JobHeldAdmin','Dependency','DependencyNeverSatisfied') and 'dev_' not in r['partition']]
 eligible.sort(key=lambda r:({'probe':0,'controller':1,'join':1,'worker':2}[owned[r['job'].split('_')[0]]],r['job']))
 for row in eligible:
  if len(dev)>=4:break
  cpu=owned[row['job'].split('_')[0]]!='probe'
  if any(('dev_cpuonly' if cpu else 'dev_accelerated') in r['partition'] and r['job'].split('_')[0] in owned for r in dev):continue
  move(row['job'],'cpuonly,dev_cpuonly' if cpu else 'accelerated,dev_accelerated')
  break
 status('PRIORITY_CHAIN_ACTIVE',probe='5158971',target_jobs=scope,held_tasks=len(list((OUT/'holds').glob('*.json'))));return False

def watch():
 for b in read(AUTH)['sources']:checked(b)
 lock=(OUT/'watch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 deadline=time.monotonic()+48*3600;errors=0
 while time.monotonic()<deadline:
  try:
   if tick():return
   errors=0
  except Exception as e:
   errors+=1;status('RETRYING_PRIORITY_OPERATION',error=repr(e),consecutive=errors)
   if errors>=5:
    restore('PRIORITY_SUPERVISOR_ERROR');raise
  time.sleep(20)
 restore('PRIORITY_48H_LIMIT');status('PRIORITY_48H_LIMIT_RESTORED')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('apply','watch','restore'));args=parser.parse_args()
 if args.stage=='apply':apply()
 elif args.stage=='watch':watch()
 else:restore('EXPLICIT_RESTORE')
