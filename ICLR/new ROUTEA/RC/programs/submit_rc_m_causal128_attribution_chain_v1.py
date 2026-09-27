#!/usr/bin/env python3
"""Scoped CPU-only attribution pipeline; immutable sources and idempotent submission."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

RC = Path(__file__).resolve().parents[1]
OUT = RC/'results/rc_m_causal128_attribution_chain_v1'
PYTHON = RC.parents[2]/'.venv-colpali/bin/python'
SCRIPT = RC/'slurm/rc_m_causal128_attribution_v1.sbatch'
MODULES = {
    'paths': RC/'programs/run_rc_m_causal128_path_ablation_v1.py',
    'bridge': RC/'programs/run_rc_m_causal128_frozen_bridge_v1.py',
    'phase': RC/'programs/run_rc_m_phase_newgroups_cpu_v1.py',
}
SUBROOTS = {'paths': OUT/'paths', 'bridge': OUT/'bridge', 'phase': OUT/'phase_replication'}


def read(path): return json.loads(Path(path).read_text())

def put(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def bind(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return {'path':str(path),'sha256':h.hexdigest()}

def checked(binding):
    assert bind(binding['path'])=={k:binding[k] for k in ['path','sha256']},'SOURCE_CHANGED: '+binding['path']
    return Path(binding['path'])

def definitions():
    return [
      dict(name='path_extract',deps=[],array='0-15%16',cpu=4,mem='16G',restarts=16),
      dict(name='path_validate',deps=['path_extract'],cpu=4,mem='16G',restarts=4),
      dict(name='pair_structure',deps=['path_validate'],cpu=2,mem='8G',restarts=4),
      dict(name='frozen_prepare',deps=['path_validate'],cpu=4,mem='16G',restarts=4),
      dict(name='frozen_bridge',deps=['frozen_prepare'],array='0-15%16',cpu=4,mem='32G',restarts=32),
      dict(name='frozen_join',deps=['frozen_bridge'],cpu=4,mem='16G',restarts=8),
      dict(name='joint_pair_bridge',deps=['frozen_prepare'],cpu=4,mem='32G',restarts=16),
      dict(name='compensation_fit',deps=['path_validate'],array='0-4%5',cpu=4,mem='16G',restarts=16),
      dict(name='compensation_join',deps=['compensation_fit'],cpu=4,mem='16G',restarts=8),
      dict(name='phase_pilot',deps=[],cpu=4,mem='32G',restarts=8),
      dict(name='phase_replication',deps=['phase_pilot'],array='0-8%9',cpu=4,mem='32G',restarts=8),
      dict(name='phase_join',deps=['phase_replication'],cpu=4,mem='16G',restarts=8),
      dict(name='final_summary',deps=['pair_structure','frozen_join','joint_pair_bridge','compensation_join','phase_join'],cpu=4,mem='16G',restarts=4),
      dict(name='health',deps=['path_extract','path_validate','pair_structure','frozen_prepare','frozen_bridge','frozen_join',
           'joint_pair_bridge','compensation_fit','compensation_join','phase_pilot','phase_replication','phase_join','final_summary'],
           dependency='afterany',cpu=2,mem='8G',restarts=2),
    ]

def verify():
    p=read(OUT/'protocol.json')
    for item in p['sources']:checked(item)
    return p

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    # All module preparation is label/outcome-blind except already frozen diagnostic target/wrong identity selection.
    for key in ['paths','phase']:
        subprocess.run([str(PYTHON),str(MODULES[key]),'prepare','--root',str(SUBROOTS[key])],check=True)
    sources=[Path(__file__),SCRIPT,*MODULES.values(),
             RC/'programs/test_rc_m_causal128_chain_v1.py',
             RC/'programs/summarize_rc_m_causal128_attribution_v1.py',
             RC/'programs/analyze_rc_m_causal128_pair_structure_v1.py',
             RC/'programs/dev_refill_rc_m_causal128_v1.py',
             RC/'plan/RC_M_CAUSAL128_ATTRIBUTION_CHAIN_V1_20260927.md',
             SUBROOTS['paths']/'protocol.json',SUBROOTS['phase']/'protocol.json']
    p=dict(schema='M_CAUSAL128_ATTRIBUTION_CHAIN_V1',sources=[bind(x) for x in sources],
           stages=definitions(),queries=128,roots={k:str(v) for k,v in SUBROOTS.items()},
           models='Existing frozen external and POST; original PRODUCT5 only for matched compensation',
           relation_reader_training=False,new_gpu_jobs=0,selection_on_outcomes=False,
           scope='Attribution, not retrieval-model optimization',
           properties_and_path_deletions_not_equated=True)
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==p,'REFUSE_MUTATE_FROZEN_CHAIN'
    else:put(path,p)
    put(OUT/'preparation_validation.json',dict(status='CAUSAL128_CHAIN_PREPARED',protocol=bind(path)))
    return p

def argv_for(stage,jobs):
    args=['sbatch','--parsable','--partition=cpuonly','--kill-on-invalid-dep=yes',
          '--cpus-per-task='+str(stage['cpu']),'--mem='+stage['mem'],
          '--comment=M_CAUSAL128:'+stage['name']]
    if stage['deps']:
        args+=['--dependency='+stage.get('dependency','afterok')+':'+':'.join(jobs[x] for x in stage['deps'])]
    if stage.get('array'):args+=['--array='+stage['array']]
    return args+[str(SCRIPT),stage['name'],str(stage['restarts'])]

def submit(dry_run=False):
    p=verify();path=OUT/'submission.json'
    with (OUT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(path) if path.exists() else dict(protocol=bind(OUT/'protocol.json'),stages={},status='PREPARED')
        assert state['protocol']==bind(OUT/'protocol.json')
        jobs={}
        for i,stage in enumerate(p['stages']):
            name=stage['name'];prior=state['stages'].get(name)
            if prior:
                assert prior.get('job_id'),'UNCERTAIN_SUBMISSION_REQUIRES_RECONCILIATION: '+name
                jobs[name]=prior['job_id'];continue
            args=argv_for(stage,jobs)
            if dry_run:
                jobs[name]=str(9100000+i);print(json.dumps({'stage':name,'argv':args}));continue
            state['stages'][name]=dict(status='SUBMITTING',argv=args)
            put(path,state)
            try:
                r=subprocess.run(args,capture_output=True,text=True,timeout=40)
                assert r.returncode==0,'SBATCH_REJECTED: '+r.stderr.strip()
                m=re.fullmatch(r'(\d+)(?:;[^\s]+)?',r.stdout.strip());assert m,'SBATCH_RESPONSE_UNCERTAIN'
                job=m.group(1)
            except Exception as error:
                state['stages'][name].update(status='SUBMISSION_NEEDS_RECONCILIATION',error=str(error));put(path,state);raise
            jobs[name]=job
            state['stages'][name].update(status='SUBMITTED',job_id=job,dependencies={x:jobs[x] for x in stage['deps']})
            state['status']='PARTIALLY_SUBMITTED';put(path,state)
            print(json.dumps({'stage':name,'job_id':job}),flush=True)
        if not dry_run:state['status']='ALL_STAGES_SUBMITTED';put(path,state)


def call(key,action,*extra):
    return subprocess.call([str(PYTHON),str(MODULES[key]),action,'--root',str(SUBROOTS[key]),*map(str,extra)])

def health():
    p=verify();s=read(OUT/'submission.json');jobs=[x['job_id'] for x in s['stages'].values() if x.get('job_id')]
    r=subprocess.run(['sacct','-X','-n','-P','-j',','.join(jobs),'--format=JobID,JobName,Partition,State,ExitCode'],capture_output=True,text=True,timeout=30)
    val={}
    for key,path in [('paths',SUBROOTS['paths']/'cache_validation.json'),('compensation',SUBROOTS['paths']/'validation.json'),
                     ('bridge',SUBROOTS['bridge']/'validation.json'),('joint_pairs',SUBROOTS['bridge']/'joint_pairs/validation.json'),
                     ('phase',SUBROOTS['phase']/'validation.json'),
                     ('pair_structure',OUT/'pair_structure/validation.json'),('final',OUT/'final_summary/validation.json')]:
        val[key]=dict(present=path.exists())
        if path.exists():val[key].update(status=read(path).get('status'),source=bind(path))
    result=dict(status='CAUSAL128_CHAIN_HEALTH_SNAPSHOT',time_utc=datetime.now(timezone.utc).isoformat(),
                protocol=bind(OUT/'protocol.json'),accounting=r.stdout.splitlines(),accounting_error=r.stderr,
                validations=val,no_scientific_success_inferred_from_exit_codes=True)
    put(OUT/'health.json',result);print(json.dumps(result));return 0

def run_stage(name):
    verify();index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    if name=='path_extract':return call('paths','extract','--shard',index,'--shards',16,'--budget',430)
    if name=='path_validate':return call('paths','validate')
    if name=='pair_structure':return subprocess.call([str(PYTHON),str(RC/'programs/analyze_rc_m_causal128_pair_structure_v1.py'),'--root',str(OUT)])
    if name=='compensation_fit':return call('paths','fit','--fold',index,'--budget',430)
    if name=='compensation_join':return call('paths','join')
    if name=='frozen_prepare':return call('bridge','prepare','--source-root',SUBROOTS['paths'])
    if name=='frozen_bridge':return call('bridge','worker','--shard',index,'--shards',16,'--budget',430)
    if name=='frozen_join':return call('bridge','join')
    if name=='joint_pair_bridge':return call('bridge','joint-pairs','--budget',430)
    if name=='phase_pilot':return call('phase','pilot','--budget',430)
    if name=='phase_replication':
        ordinal=read(SUBROOTS['phase']/'protocol.json')['workers'][index]['index']
        return call('phase','worker','--index',ordinal,'--budget',430)
    if name=='phase_join':return call('phase','join')
    if name=='final_summary':return subprocess.call([str(PYTHON),str(RC/'programs/summarize_rc_m_causal128_attribution_v1.py'),'--root',str(OUT)])
    if name=='health':return health()
    raise ValueError(name)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','dry-run','run']);p.add_argument('stage',nargs='?');a=p.parse_args()
    if a.action=='prepare':prepare();print('CAUSAL128_CHAIN_PREPARED');return 0
    if a.action in ['submit','dry-run']:submit(a.action=='dry-run');return 0
    try:return run_stage(a.stage)
    except Exception as e:
        put(OUT/'stage_errors'/f"{a.stage}_{os.environ.get('SLURM_ARRAY_TASK_ID','scalar')}.json",dict(
            status='ATTRIBUTION_ENGINEERING_ERROR',error=repr(e),scientific_result=False,
            time_utc=datetime.now(timezone.utc).isoformat()))
        raise

if __name__=='__main__':raise SystemExit(main())
