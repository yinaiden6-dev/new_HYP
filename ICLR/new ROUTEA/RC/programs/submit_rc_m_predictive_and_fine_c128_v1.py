#!/usr/bin/env python3
"""Sealed CPU task chain for prospective POST probes and fine C128 attribution."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

RC = Path(__file__).resolve().parents[1]
ROOT = RC / 'results/rc_m_predictive_and_fine_c128_chain_v1'
PRED = RC / 'results/rc_post_m_predictive_response_v1'
FINE = RC / 'results/rc_m_fine_c128_attribution_v1'
PRED_CODE = RC / 'programs/run_rc_post_m_predictive_response_v1.py'
FINE_CODE = RC / 'programs/run_rc_m_fine_c128_attribution_v1.py'
LAUNCHER = RC / 'slurm/rc_m_predictive_and_fine_c128_v1.sbatch'
PLAN = RC / 'plan/RC_M_PREDICTIVE_AND_FINE_C128_CHAIN_V1_20260927.md'
STAGES = {
    'predict_probe': (PRED_CODE, ['worker','--stage','probe','--shard','{index}','--shards','8','--budget','430'], 8, []),
    'predict_seal': (PRED_CODE, ['seal'], 1, ['predict_probe']),
    'predict_endpoint': (PRED_CODE, ['worker','--stage','endpoint','--shard','{index}','--shards','8','--budget','430'], 8, ['predict_seal']),
    'predict_join': (PRED_CODE, ['join'], 1, ['predict_endpoint']),
    'predict_check': (PRED_CODE, ['check'], 1, ['predict_join']),
    'phase_existing': (FINE_CODE, ['phase-worker','--index','{index}','--budget','430'], 11, []),
    'phase_join': (FINE_CODE, ['phase-join'], 1, ['phase_existing']),
    'context_pilot': (FINE_CODE, ['worker','--shard','0','--shards','50','--budget','430','--limit-pairs','1'], 1, []),
    'context': (FINE_CODE, ['worker','--shard','{index}','--shards','50','--budget','430'], 50, ['context_pilot']),
    'context_join': (FINE_CODE, ['context-join'], 1, ['context']),
    'fine_bridge': (FINE_CODE, ['bridge-worker','--index','{index}','--budget','430'], 11, ['context_join']),
    'fine_join': (FINE_CODE, ['join'], 1, ['fine_bridge','phase_join']),
    'closure': (None, [], 1, ['predict_check','fine_join']),
}


def read(path): return json.loads(Path(path).read_text())
def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def save(path, data, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and immutable:
        assert path.read_text() == data, ('IMMUTABLE_CHANGED', str(path)); return
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(data); os.replace(temp, path)
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def specification():
    return {name: dict(program=str(code) if code else None, args=args, count=count,
                       dependencies=dependencies)
            for name,(code,args,count,dependencies) in STAGES.items()}


def prepare():
    pp, fp = read(PRED/'protocol.json'), read(FINE/'protocol.json')
    assert 'FROZEN' in pp['status'] and 'FROZEN' in fp['status']
    sources = [bind(path) for path in [Path(__file__), LAUNCHER, PLAN, PRED_CODE, FINE_CODE,
                                      PRED/'protocol.json', FINE/'protocol.json']]
    p = dict(status='M_PREDICTIVE_FINE_CHAIN_FROZEN', sources=sources, stages=specification(),
             roots=dict(predictive=str(PRED), fine=str(FINE)),
             resources=dict(partition='cpuonly', dev_partition='dev_cpuonly,cpuonly',
                            CPUs=4, memory='32G', time_limit='00:10:00', budget_seconds=430,
                            same_id_requeue_limit=32, GPUs=0),
             branch_scope=dict(predictive='Opened120 queries; new M intervention endpoints predicted before forwarding; not untouched-query confirmation.',
                               fine='Eleven complete natural C128 axes; outcome-independent8 and selected rescue3 reported separately. Not all128/593 fine-property attribution.'),
             continuation='Only budget75 or timeout124 requeues; scientific or I/O failures stop dependent chain.',
             expected_fine_status='FINE_C128_THREE_PROPERTY_ACCOUNTING_PASS',
             no_new_training=True, no_encoder_matcher_or_LLM_forward=True)
    save(ROOT/'protocol.json', p, immutable=True)
    print(json.dumps(dict(status=p['status'], protocol=bind(ROOT/'protocol.json'))), flush=True)


def guard():
    p = read(ROOT/'protocol.json')
    assert p['status'] == 'M_PREDICTIVE_FINE_CHAIN_FROZEN' and p['stages'] == specification()
    for source in p['sources']: assert bind(source['path']) == source, ('SEALED_SOURCE_CHANGED', source['path'])
    return p


def submit():
    guard(); record_path = ROOT/'submission.json'
    record = read(record_path) if record_path.exists() else dict(status='SUBMITTING', utc=utc(), protocol=bind(ROOT/'protocol.json'), stages={})
    assert record['protocol'] == bind(ROOT/'protocol.json')
    for name,(code,args,count,deps) in STAGES.items():
        if name in record['stages']: continue
        argv = ['sbatch','--parsable','--job-name=M2_'+name]
        if count > 1: argv += ['--array=0-'+str(count-1)+'%'+str(min(50,count))]
        dependency_ids = [record['stages'][d]['job_id'] for d in deps]
        if dependency_ids: argv += ['--dependency=afterok:'+':'.join(dependency_ids)]
        argv += [str(LAUNCHER),name]
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode:
            record['last_error'] = dict(stage=name, argv=argv, stderr=result.stderr)
            save(record_path, record); print(result.stderr, file=sys.stderr); return result.returncode
        job = result.stdout.strip().split(';')[0]; assert job.isdigit(), result.stdout
        record['stages'][name] = dict(job_id=job, dependencies=dependency_ids, tasks=count, argv=argv)
        save(record_path, record); print(name,job,flush=True)
    record['status'] = 'M_PREDICTIVE_FINE_CHAIN_SUBMITTED'; save(record_path,record)
    return verify()


def verify():
    guard(); record=read(ROOT/'submission.json'); checks=[]
    for name,stage in record['stages'].items():
        job=stage['job_id']; output=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'TimeLimit=00:10:00' in output and 'cpu=4' in output and 'mem=32G' in output, output
        assert 'gres/gpu' not in output, output
        for dep in stage['dependencies']: assert dep in output, (name,dep,output)
        spool=ROOT/('submitted_'+job+'.sbatch')
        subprocess.run(['scontrol','write','batch_script',job,str(spool)],check=True,capture_output=True)
        assert spool.read_bytes()==LAUNCHER.read_bytes()
        checks.append(dict(stage=name,job_id=job,scheduler=output.strip(),spool_matches=True))
    save(ROOT/'scheduler_validation.json',dict(status='M_PREDICTIVE_FINE_SCHEDULER_PASS',utc=utc(),jobs=checks))
    print('M_PREDICTIVE_FINE_SCHEDULER_PASS',flush=True); return 0


def close():
    p=guard(); paths=[PRED/'validation.json',PRED/'independent_validation.json',FINE/'validation.json']
    receipts=[]
    for path in paths:
        if not path.exists(): return 75
        value=read(path); assert 'PASS' in value['status'],(path,value['status'])
        if path.parent==FINE: assert value['status']==p['expected_fine_status']
        receipts.append(dict(status=value['status'],**bind(path)))
    save(ROOT/'validation.json',dict(status='M_PREDICTIVE_FINE_BOTH_BRANCHES_PASS',protocol=bind(ROOT/'protocol.json'),
        receipts=receipts,full128_fine_attribution_complete=False,untouched_query_confirmation=False,
        unique_cause_claim=False),immutable=True)
    print('M_PREDICTIVE_FINE_BOTH_BRANCHES_PASS',flush=True); return 0


def run(stage):
    guard()
    if stage=='closure': return close()
    code,args,count,_=STAGES[stage]
    index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')); assert 0<=index<count
    return subprocess.run([sys.executable,'-u',str(code),*[a.format(index=index) for a in args]]).returncode


def refill():
    """Offer only ready pending jobs in this recorded chain to the dev CPU lane."""
    guard(); submission=read(ROOT/'submission.json'); permitted={s['job_id'] for s in submission['stages'].values()}
    output=subprocess.check_output(['squeue','-u',os.environ.get('USER','ap7811'),'-h','-r','-o','%i|%P|%T|%R'],text=True)
    rows=[line.split('|',3) for line in output.splitlines() if line.strip()]
    active_dev=sum('dev_' in partition for _,partition,_,_ in rows)
    slots=max(0,4-active_dev); changes=[]
    stage_by_id={s['job_id']:name for name,s in submission['stages'].items()}
    priority={'context_pilot':0,'predict_probe':1,'predict_seal':1,'predict_endpoint':1,'predict_join':1,'predict_check':1,
              'phase_existing':2,'phase_join':2,'context':3,'context_join':0,'fine_bridge':1,'fine_join':0,'closure':0}
    eligible=[]
    for job,partition,state,reason in rows:
        parent=job.split('_')[0]
        if parent not in permitted or state!='PENDING' or 'dev_' in partition: continue
        if reason not in ('Priority','Resources','None'): continue
        assert re.fullmatch(r'\d+(?:_\d+)?',job),job
        eligible.append((priority[stage_by_id[parent]],job,partition,reason))
    for _,job,partition,reason in sorted(eligible)[:slots]:
        before=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'JobState=PENDING' in before and 'TimeLimit=00:10:00' in before and 'gres/gpu' not in before
        changed=subprocess.run(['scontrol','update','JobId='+job,'Partition=dev_cpuonly,cpuonly'],capture_output=True,text=True)
        if changed.returncode:
            changes.append(dict(job=job,status='UPDATE_REJECTED',stderr=changed.stderr)); break
        after=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'Partition=dev_cpuonly,cpuonly' in after and 'cpu=4' in after and 'mem=32G' in after
        changes.append(dict(job=job,status='VERIFIED',before=before.strip(),after=after.strip()))
    record=dict(utc=utc(),dev_tasks_before=active_dev,changes=changes)
    path=ROOT/'dev_refill.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle: handle.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True); return 0


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','submit','verify','run','refill','close']);parser.add_argument('stage',nargs='?')
    args=parser.parse_args()
    if args.mode=='prepare': prepare(); return 0
    if args.mode=='run': return run(args.stage)
    return {'submit':submit,'verify':verify,'refill':refill,'close':close}[args.mode]()


if __name__=='__main__': raise SystemExit(main())
