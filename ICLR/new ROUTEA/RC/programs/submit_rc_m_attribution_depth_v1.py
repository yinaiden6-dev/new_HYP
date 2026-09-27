#!/usr/bin/env python3
"""Submit two separately sealed, CPU-only attribution branches and their joins."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RC = Path(__file__).resolve().parents[1]
ROOT = RC / 'results/rc_m_attribution_depth_v1'
LAUNCHER = RC / 'slurm/rc_m_attribution_depth_v1.sbatch'
UP = RC / 'results/rc_m_context_binding_v1'
DOWN = RC / 'results/rc_post_m_level_gap_v1'
UP_CODE = RC / 'programs/rc_m_context_binding_v1.py'
DOWN_CODE = RC / 'programs/run_rc_post_m_level_gap_v1.py'
DOWN_CHECK = RC / 'programs/check_rc_post_m_level_gap_v1.py'
STAGES = {
    'upstream': dict(program=UP_CODE, args=['worker','--pair-index','{index}','--budget','430'], array='0-17%18', count=18, dependencies=[]),
    'downstream': dict(program=DOWN_CODE, args=['worker','--shard','{index}','--shards','8','--budget','430'], array='0-7%8', count=8, dependencies=[]),
    'upstream_join': dict(program=UP_CODE, args=['join'], count=1, dependencies=['upstream']),
    'upstream_bridge': dict(program=UP_CODE, args=['bridge-worker','--index','{index}','--budget','430'], array='0-8%9', count=9, dependencies=['upstream_join']),
    'upstream_bridge_join': dict(program=UP_CODE, args=['bridge-join'], count=1, dependencies=['upstream_bridge']),
    'downstream_join': dict(program=DOWN_CODE, args=['join'], count=1, dependencies=['downstream']),
    'downstream_audit': dict(program=DOWN_CHECK, args=[], count=1, dependencies=['downstream_join']),
    'closure': dict(program=None, args=[], count=1, dependencies=['upstream_bridge_join','downstream_audit']),
}


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p=Path(p).resolve()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def save(p,obj,immutable=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists() and immutable:
        assert p.read_text()==content,('IMMUTABLE_CHANGED',str(p));return
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(content);os.replace(tmp,p)


def guard():
    p=read(ROOT/'protocol.json')
    assert p['status']=='M_ATTRIBUTION_DEPTH_ORCHESTRATION_FROZEN'
    for b in p['sources']:
        assert bind(b['path'])==b,('SEALED_SOURCE_CHANGED',b['path'])
    return p


def prepare(up_status,down_status):
    assert up_status and down_status
    sources=[bind(x) for x in [Path(__file__),LAUNCHER,UP_CODE,DOWN_CODE,DOWN_CHECK,UP/'protocol.json',DOWN/'protocol.json']]
    p=dict(status='M_ATTRIBUTION_DEPTH_ORCHESTRATION_FROZEN',
        scope='Separate P-context alignment and downstream common-level/gap causal diagnostics; no QRR or accuracy search.',
        sources=sources,required_statuses={str(UP/'validation.json'):up_status,str(DOWN/'validation.json'):down_status,
            str(DOWN/'second_arithmetic_validation.json'):'POST_M_LEVEL_GAP_SECOND_ARITHMETIC_PASS'},
        stages={k:{**v,'program':str(v['program']) if v['program'] else None} for k,v in STAGES.items()},
        resources=dict(partition='accelerated',cpu=4,memory='32G',gpu_for_scheduling=1,gpu_computation=False,
            time_limit='00:10:00',worker_budget_seconds=430,max_same_id_requeues=6),
        population='Opened historical mechanism panels; post-hoc diagnostics, not untouched external confirmation.')
    save(ROOT/'protocol.json',p,immutable=True)
    print(json.dumps(dict(status=p['status'],protocol=bind(ROOT/'protocol.json'))))


def submit():
    p=guard();path=ROOT/'submission.json'
    record=read(path) if path.exists() else dict(status='SUBMITTING',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=bind(ROOT/'protocol.json'),stages={})
    assert record['protocol']==bind(ROOT/'protocol.json')
    for name,spec in STAGES.items():
        if name in record['stages']:continue
        argv=['sbatch','--parsable','--job-name=M_'+name]
        if spec.get('array'):argv+=['--array='+spec['array']]
        dependencies=[record['stages'][x]['job_id'] for x in spec['dependencies']]
        if dependencies:argv+=['--dependency=afterok:'+':'.join(dependencies)]
        argv += [str(LAUNCHER),name]
        proc=subprocess.run(argv,capture_output=True,text=True)
        if proc.returncode:
            record['last_error']=dict(stage=name,stderr=proc.stderr,argv=argv)
            save(path,record);print(proc.stderr,file=sys.stderr);return proc.returncode
        job=proc.stdout.strip().split(';')[0];assert job.isdigit(),proc.stdout
        record['stages'][name]=dict(job_id=job,argv=argv,dependencies=dependencies,array=spec.get('array'),tasks=spec['count'])
        save(path,record);print(name,job,flush=True)
    record['status']='M_ATTRIBUTION_DEPTH_CHAIN_SUBMITTED';save(path,record)
    return verify()


def verify():
    guard();record=read(ROOT/'submission.json');checks=[]
    for name,s in record['stages'].items():
        job=s['job_id']
        value=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'Partition=accelerated' in value and 'TimeLimit=00:10:00' in value and 'gres/gpu=1' in value
        assert 'cpu=4' in value and 'mem=32G' in value
        for dep in s['dependencies']:assert dep in value,('DEPENDENCY_NOT_VISIBLE',name,dep)
        spool=ROOT/('submitted_'+job+'.sbatch')
        subprocess.run(['scontrol','write','batch_script',job,str(spool)],check=True,capture_output=True)
        assert spool.read_bytes()==LAUNCHER.read_bytes()
        checks.append(dict(stage=name,job_id=job,scheduler=value.strip(),spool_matches=True))
    save(ROOT/'scheduler_validation.json',dict(status='M_ATTRIBUTION_DEPTH_SCHEDULER_VERIFIED',jobs=checks,protocol=bind(ROOT/'protocol.json')))
    print('SCHEDULER_AND_SPOOL_PASS');return 0


def close():
    p=guard();receipts=[]
    for name,expected in p['required_statuses'].items():
        file=Path(name)
        if not file.exists():return 75
        value=read(file);assert value['status']==expected,(name,value['status'])
        receipts.append(dict(status=value['status'],**bind(file)))
    save(ROOT/'validation.json',dict(status='M_ATTRIBUTION_DEPTH_BOTH_BRANCHES_PASS',protocol=bind(ROOT/'protocol.json'),
        receipts=receipts,unique_cause_claim=False,independent_external_confirmation_claim=False),immutable=True)
    print('M_ATTRIBUTION_DEPTH_BOTH_BRANCHES_PASS');return 0


def run(stage):
    guard();assert stage in STAGES
    if stage=='closure':return close()
    spec=STAGES[stage];index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));assert 0<=index<spec['count']
    args=[x.format(index=index) for x in spec['args']]
    return subprocess.run([sys.executable,'-u',str(spec['program']),*args]).returncode


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','submit','verify','run','close'])
    ap.add_argument('stage',nargs='?');ap.add_argument('--up-status');ap.add_argument('--down-status');a=ap.parse_args()
    if a.mode=='prepare':prepare(a.up_status,a.down_status);return 0
    if a.mode=='submit':return submit()
    if a.mode=='verify':return verify()
    if a.mode=='run':return run(a.stage)
    return close()


if __name__=='__main__':
    raise SystemExit(main())
