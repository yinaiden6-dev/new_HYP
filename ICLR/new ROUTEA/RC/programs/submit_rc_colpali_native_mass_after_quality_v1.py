#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import run_rc_colpali_native_mass_head_v1 as H
assert not (H.OUT/'submission.json').exists()
H.prepare()
def run(args):return subprocess.run(args,text=True,capture_output=True,check=True).stdout.strip()
gpu=H.ROOT/'slurm/rc_colpali_native_mass_gpu_v1.sbatch';cpu=H.ROOT/'slurm/rc_colpali_native_mass_cpu_v1.sbatch';jobs={}
try:
    jobs['score_pilot']=run(['sbatch','--parsable','--hold','--array=0',str(gpu)]).split(';')[0]
    jobs['score_remaining']=run(['sbatch','--parsable','--hold','--array=1-49%50','--dependency=afterok:'+jobs['score_pilot'],str(gpu)]).split(';')[0]
    jobs['folds']=run(['sbatch','--parsable','--hold','--array=0-4%5','--dependency=afterok:'+jobs['score_remaining'],str(cpu),'fit']).split(';')[0]
    jobs['summary']=run(['sbatch','--parsable','--hold','--dependency=afterok:'+jobs['folds'],str(cpu),'join']).split(';')[0]
    for name,jid in jobs.items():
        dest=H.OUT/('submitted_'+jid+'.sbatch');run(['scontrol','write','batch_script',jid,str(dest)])
        assert dest.read_bytes()==(gpu if name.startswith('score') else cpu).read_bytes()
    for jid in jobs.values():run(['scontrol','release',jid])
finally:
    H.write(H.OUT/'submission.json',jobs)
print(json.dumps(jobs),flush=True)
