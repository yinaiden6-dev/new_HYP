#!/usr/bin/env python3
"""Submit and verify an isolated, resume-safe OCR support pilot DAG."""
import fcntl
import subprocess
import run_rc_ocr_support_pilot128_v1 as R

STAGES=[('smoke',None,[],'dev_cpuonly,cpuonly'),('ocr','0-31%32',['smoke'],'cpuonly'),
        ('index',None,['ocr'],'cpuonly'),('features','0-15%16',['index'],'cpuonly'),
        ('fit','0-4%5',['features'],'cpuonly'),('join',None,['fit'],'cpuonly')]

def main():
    R.protocol();root=R.OUT;path=root/'submission.json';launcher=R.RC/'slurm/rc_ocr_support_pilot128_v1.sbatch'
    assert R.read(root/'self_test.json')['status']=='OCR_TOKEN_POOL_FREE_MATCH_AND_LOSS_PASS'
    assert R.read(root/'smoke_validation.json')['status']=='PADDLE_CPU_SOURCE_AND_BOX_SCHEMA_PASS'
    assert R.read(root/'real_pair_validation.json')['status']=='REAL_PADDLE_ROMA_TEXT_INTERFACE_PASS'
    with (root/'submission.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        d=R.read(path) if path.exists() else dict(status='SUBMITTING',protocol=R.bind(root/'protocol.json'),stages={})
        assert d['protocol']==R.bind(root/'protocol.json')
        for name,array,deps,partition in STAGES:
            if name in d['stages']:continue
            assert not d.get('intent'),'RECONCILE_UNCERTAIN_SUBMISSION'
            args=['sbatch','--parsable','--hold','--partition='+partition,'--job-name=OCRsup_'+name]
            if array:args+=['--array='+array]
            if deps:args+=['--dependency=afterok:'+':'.join(d['stages'][k]['job_id'] for k in deps)]
            args += [str(launcher),name];d['intent']=dict(stage=name,args=args);R.write(path,d)
            r=subprocess.run(args,text=True,capture_output=True)
            if r.returncode:
                d.pop('intent');d['rejection']=r.stderr;R.write(path,d);raise RuntimeError(r.stderr)
            job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
            d['stages'][name]=dict(job_id=job,args=args,dependencies=deps);d.pop('intent');R.write(path,d);print(name,job,flush=True)
        for name,s in d['stages'].items():
            if s.get('released'):continue
            fields=subprocess.check_output(['scontrol','show','job','-o',s['job_id']],text=True)
            assert 'TimeLimit=00:10:00' in fields and 'cpu=4' in fields and 'mem=24G' in fields and 'gres/gpu' not in fields
            spool=root/f"submitted_{s['job_id']}.sbatch"
            if not spool.exists():subprocess.run(['scontrol','write','batch_script',s['job_id'],str(spool)],check=True,capture_output=True)
            assert spool.read_bytes()==launcher.read_bytes();s['spool']=R.bind(spool);s['before_release']=fields;R.write(path,d)
        for name,s in d['stages'].items():
            if s.get('released'):continue
            subprocess.run(['scontrol','release',s['job_id']],check=True)
            s['released']=True;s['after_release']=subprocess.check_output(['scontrol','show','job','-o',s['job_id']],text=True);R.write(path,d)
        d['status']='SUBMITTED_VERIFIED_RELEASED';R.write(path,d)

if __name__=='__main__':main()
