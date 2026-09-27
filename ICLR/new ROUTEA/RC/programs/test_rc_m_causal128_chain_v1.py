#!/usr/bin/env python3
"""Exercise submission idempotence and actual shell requeue semantics without Slurm mutations."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain',HERE/'submit_rc_m_causal128_attribution_chain_v1.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def main():
    with tempfile.TemporaryDirectory(prefix='causal128-chain-test-') as td:
        root=Path(td);c.OUT=root/'chain';c.OUT.mkdir();c.put(c.OUT/'protocol.json',{'stages':c.definitions()})
        c.verify=lambda:c.read(c.OUT/'protocol.json')
        real_run=c.subprocess.run;calls=[]
        def fake(argv,**kw):
            assert argv[0]=='sbatch' and '--kill-on-invalid-dep=yes' in argv
            calls.append(argv);return SimpleNamespace(returncode=0,stdout=str(9200000+len(calls))+'\n',stderr='')
        c.subprocess.run=fake
        c.submit();count=len(calls);assert count==len(c.definitions())
        c.submit();assert len(calls)==count
        s=c.read(c.OUT/'submission.json');jobs={k:v['job_id'] for k,v in s['stages'].items()}
        for stage,argv in zip(c.definitions(),calls):
            if stage['deps']:
                dep='--dependency='+stage.get('dependency','afterok')+':'+':'.join(jobs[x] for x in stage['deps'])
                assert dep in argv
        before=(c.OUT/'submission.json').read_bytes();c.submit(dry_run=True)
        assert len(calls)==count and (c.OUT/'submission.json').read_bytes()==before
        c.OUT=root/'rejection';c.OUT.mkdir();c.put(c.OUT/'protocol.json',{'stages':c.definitions()})
        def reject(argv,**kw):return SimpleNamespace(returncode=1,stdout='',stderr='test rejection')
        c.subprocess.run=reject
        try:c.submit()
        except AssertionError:pass
        else:raise AssertionError('REJECTION_ACCEPTED')
        assert c.read(c.OUT/'submission.json')['stages']['path_extract']['status']=='SUBMISSION_NEEDS_RECONCILIATION'
        c.subprocess.run=fake
        try:c.submit()
        except AssertionError:pass
        else:raise AssertionError('UNCERTAIN_RESUBMITTED')
        assert len(calls)==count
        c.subprocess.run=real_run
        envfile=root/'functions.sh'
        envfile.write_text('timeout() { return "$TEST_TIMEOUT_STATUS"; }\nscontrol() { printf "%s\\n" "$*" >> "$TEST_REQUEUE_LOG"; }\n')
        script=HERE.parent/'slurm/rc_m_causal128_attribution_v1.sbatch'
        cases=[(0,0,0,False),(1,0,1,False),(75,0,0,True),(124,0,0,True),(75,16,124,False)]
        for k,(status,restarts,expected,queued) in enumerate(cases):
            log=root/f'requeue{k}.txt'
            env={**os.environ,'BASH_ENV':str(envfile),'TEST_TIMEOUT_STATUS':str(status),'TEST_REQUEUE_LOG':str(log),
                 'SLURM_JOB_ID':'9200010','SLURM_ARRAY_JOB_ID':'9200001','SLURM_ARRAY_TASK_ID':'2','SLURM_RESTART_COUNT':str(restarts)}
            r=subprocess.run(['bash',str(script),'path_extract','16'],env=env,capture_output=True,text=True)
            assert r.returncode==expected,(k,r.returncode,r.stderr)
            assert log.exists()==queued
            if queued:assert log.read_text().strip()=='requeue 9200001_2'
        result={'status':'CAUSAL128_CHAIN_EXECUTION_TEST_PASS','stages':count,'checks':['exact_dependencies','idempotent_submission',
            'dry_run_no_changes','rejection_persisted','uncertain_not_resubmitted','actual_shell_success_failure_timeout_requeue_limit'],
            'sources':[c.bind(HERE/'submit_rc_m_causal128_attribution_chain_v1.py'),c.bind(script),c.bind(__file__)]}
        target=HERE.parent/'results/rc_m_causal128_attribution_chain_v1/chain_tests.json';c.put(target,result)
        print(json.dumps(result))
if __name__=='__main__':main()
