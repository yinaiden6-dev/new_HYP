#!/usr/bin/env python3
"""Submit one frozen upstream-only expansion DAG, safely resumable submission."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import run_rc_m_upstream_all593_v1 as U
ROOT,RC=U.OUT,U.RC
LAUNCHER=RC/'slurm/rc_m_upstream_all593_v1.sbatch'
STAGES=[
 ('prepare',None,[],'dev_cpuonly,cpuonly'),
 ('pilot',None,['prepare'],'dev_cpuonly,cpuonly'),
 ('worker','0-49%50',['pilot'],'cpuonly'),
 ('join',None,['worker'],'cpuonly'),
 ('property-fit','0-4%5',['join'],'cpuonly'),
 ('property-join',None,['property-fit'],'cpuonly'),
 ('cached',None,[],'cpuonly'),
 ('path-prepare',None,[],'cpuonly'),
 ('path-extract','0-49%50',['path-prepare'],'cpuonly'),
 ('path-validate',None,['path-extract'],'cpuonly'),
 ('path-fit','0-4%5',['path-validate'],'cpuonly'),
 ('path-join',None,['path-fit'],'cpuonly'),
 ('separability',None,['path-validate'],'cpuonly'),
 ('final',None,['property-join','cached','path-join','separability'],'cpuonly')]


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),Path(U.__file__),U.PLAN,LAUNCHER,
       RC/'programs/analyze_rc_m_upstream_all593_v1.py',RC/'programs/run_rc_m_upstream593_path_compensation_v1.py',
       RC/'programs/analyze_rc_m_upstream593_pair_structure_v1.py']
    for name in ['run_rc_m_phase_newgroups_cpu_v1.py','probe_rc_roma_property_restore_cpu_v3.py',
                 'rc_roma_property_restore_v2.py','rc_roma_property_restore_v1.py','rc_roma_position_factor_v1.py',
                 'rc_m_context_binding_v1.py','run_rc_m_structure_binding_isolation_v2.py',
                 'rc_roma_shared_native_cache_v1.py','run_rc_m593_cached_replication_fix_v1.py',
                 'run_rc_m_causal128_property_compensation_v1.py','run_rc_m_causal128_path_ablation_v1.py',
                 'run_rc_six_cause_loss_binding_v1.py']:
        paths.append(RC/'programs'/name)
    legacy=U.read(RC/'results/rc_m_causal128_attribution_chain_v1/phase_replication/protocol.json')
    paths.extend(Path(b['path']) for b in legacy['code_sources'])
    authority=dict(status='ALL593_UPSTREAM_EXPANSION_AUTHORIZED',user_request='把咱们做的上游归因实验，都扩展到593',
        date='2026-10-09',code_sources=[U.bind(p) for p in dict.fromkeys(paths)],stages=STAGES,
        new_GPU=False,query_universe=593,candidates=128,worlds=60,original_results_immutable=True,
        reuse_complete=['inside593','visual593','coordinate593','path_external_replication593'],
        test_receipt=U.bind(ROOT/'self_test.json'))
    path=ROOT/'authority.json'
    if path.exists():assert U.read(path)==json.loads(json.dumps(authority)),'FROZEN_AUTHORITY_CHANGED'
    else:U.write(path,authority)


def submit():
    U.guard();statepath=ROOT/'submission.json'
    with (ROOT/'submission.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=U.read(statepath) if statepath.exists() else dict(status='SUBMITTING',authority=U.bind(ROOT/'authority.json'),stages={})
        assert state['authority']==U.bind(ROOT/'authority.json')
        for name,array,deps,partition in STAGES:
            if name in state['stages']:continue
            args=['sbatch','--parsable','--hold','--partition='+partition,'--job-name=M593_'+name]
            if array:args+=['--array='+array]
            if deps:args+=['--dependency=afterok:'+':'.join(state['stages'][k]['job_id'] for k in deps)]
            args += [str(LAUNCHER),name]
            # Persist intent before submission. An uncertain response is never
            # retried automatically, avoiding duplicate experiment arrays.
            if state.get('pending_intent'):raise RuntimeError('RECONCILE_UNCERTAIN_SUBMISSION_FIRST')
            state['pending_intent']=dict(name=name,args=args);U.write(statepath,state)
            run=subprocess.run(args,text=True,capture_output=True)
            if run.returncode:
                state['last_rejection']=dict(name=name,stderr=run.stderr);state.pop('pending_intent');U.write(statepath,state)
                raise RuntimeError(run.stderr)
            job=run.stdout.strip().split(';')[0];assert job.isdigit(),run.stdout
            state['stages'][name]=dict(job_id=job,args=args,array=array,dependencies=deps,partition=partition)
            state.pop('pending_intent');U.write(statepath,state);print(name,job,flush=True)
        for name,s in state['stages'].items():
            if s.get('released'):continue
            job=s['job_id'];fields=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
            assert 'TimeLimit=00:10:00' in fields and 'gres/gpu' not in fields,fields
            assert 'cpu=4' in fields and 'mem=32G' in fields,fields
            spool=ROOT/f'submitted_{job}.sbatch'
            if not spool.exists():subprocess.run(['scontrol','write','batch_script',job,str(spool)],check=True,capture_output=True)
            assert spool.read_bytes()==LAUNCHER.read_bytes(),'SPOOL_CHANGED'
            s.update(scheduler_before_release=fields.strip(),spool=U.bind(spool));U.write(statepath,state)
        for name,s in state['stages'].items():
            if s.get('released'):continue
            subprocess.run(['scontrol','release',s['job_id']],check=True)
            s['released']=True;s['scheduler_after_release']=subprocess.check_output(['scontrol','show','job','-o',s['job_id']],text=True).strip()
            U.write(statepath,state)
        state['status']='SUBMITTED_VERIFIED_RELEASED';U.write(statepath,state)
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','submit']);a=ap.parse_args()
    freeze() if a.mode=='freeze' else submit()
