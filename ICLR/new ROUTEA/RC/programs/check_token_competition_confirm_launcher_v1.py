#!/usr/bin/env python3
"""Execute the actual launcher shell with mocked worker/controller commands."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

RC=Path(__file__).resolve().parents[1]
SCRIPT=RC/'slurm/token_competition_confirm_v1.sbatch'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=RC/'results/rc_token_competition_f128_confirm_v1/launcher_engineering.json')
    a=p.parse_args()
    source=SCRIPT.read_text()
    subprocess.run(['bash','-n',str(SCRIPT)],check=True)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='token-launcher-check-') as d:
        root=Path(d); bins=root/'bin';bins.mkdir()
        worker=bins/'worker'
        worker.write_text('#!/usr/bin/python3\nimport json,os,sys\n'
            'with open(os.environ["TOKEN_TEST_TRACE"],"a") as f:f.write(json.dumps({"kind":"worker","argv":sys.argv[1:],"cuda":os.getenv("CUDA_VISIBLE_DEVICES")})+"\\n")\n'
            'sys.exit(int(os.environ.get("TOKEN_TEST_EXIT","0")))\n')
        worker.chmod(0o755)
        ctl=bins/'scontrol'
        ctl.write_text('#!/usr/bin/python3\nimport json,os,sys\n'
            'with open(os.environ["TOKEN_TEST_TRACE"],"a") as f:f.write(json.dumps({"kind":"scontrol","argv":sys.argv[1:]})+"\\n")\n')
        ctl.chmod(0o755)
        # Only replace command locations: case branches, arrays, quoting,
        # timeout and status handling remain exactly the submitted source.
        edited=source.replace('export PATH=/usr/local/bin:/usr/bin:/bin',
                  'export PATH='+shlex.quote(str(bins)+':/usr/local/bin:/usr/bin:/bin'))
        old="token_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'"
        if edited.count(old)!=1:raise ValueError('launcher interpreter assignment drift')
        edited=edited.replace(old,'token_python='+shlex.quote(str(worker)))
        launch=root/'launcher.sh';launch.write_text(edited)
        cases=[('join',[],0,0,0,None),('train',[],0,0,0,'29'),
               ('train',['3'],0,0,0,'3'),('join',[],75,0,0,None),
               ('train',[],75,0,0,'29'),('join',[],124,0,0,None),
               ('join',[],75,96,124,None),('train',[],1,0,1,'29'),
               ('unknown',[],0,0,2,None)]
        for n,(stage,extra,status,restarts,expected,index) in enumerate(cases):
            trace=root/f'trace{n}.jsonl'
            env=dict(os.environ,TOKEN_TEST_TRACE=str(trace),TOKEN_TEST_EXIT=str(status),
                     SLURM_JOB_ID='999901',SLURM_RESTART_COUNT=str(restarts))
            for k in ('SLURM_ARRAY_JOB_ID','SLURM_ARRAY_TASK_ID'):env.pop(k,None)
            if stage in ('train',):
                env.update(SLURM_ARRAY_JOB_ID='999900',SLURM_ARRAY_TASK_ID='29')
            result=subprocess.run(['bash',str(launch),stage]+extra,env=env,capture_output=True,text=True)
            if result.returncode!=expected:raise AssertionError((stage,status,restarts,result.returncode,result.stderr))
            events=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
            workers=[e for e in events if e['kind']=='worker'];requeues=[e for e in events if e['kind']=='scontrol']
            if stage!='unknown':
                if len(workers)!=1 or workers[0]['cuda']!='':raise AssertionError('worker/CUDA routing')
                argv=workers[0]['argv']
                if argv[:3]!=['-u','ICLR/new ROUTEA/RC/programs/run_token_competition_confirm_stage_v1.py',stage]:
                    raise AssertionError(('worker argv',argv))
                if index is not None and argv[-2:]!=['--index',index]:raise AssertionError(('index',argv))
            expect_requeue=status in (75,124) and restarts<96 and stage!='unknown'
            if bool(requeues)!=expect_requeue:raise AssertionError('requeue condition')
            if expect_requeue:
                job='999900_'+env['SLURM_ARRAY_TASK_ID'] if 'SLURM_ARRAY_JOB_ID' in env else '999901'
                if requeues!=[{'kind':'scontrol','argv':['requeue',job]}]:raise AssertionError('requeue wrong job')
            checks.append({'stage':stage,'worker_status':status,'restarts':restarts,'exit':result.returncode,'passed':True})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    v={'status':'TOKEN_COMPETITION_CONFIRM_LAUNCHER_PASS','checks':checks,
       'source':{'path':str(SCRIPT),'sha256':hashlib.sha256(SCRIPT.read_bytes()).hexdigest()},
       'checker':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
       'executed_shell_branches':True,'submitted_real_jobs':False}
    a.output.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)


if __name__=='__main__':main()
