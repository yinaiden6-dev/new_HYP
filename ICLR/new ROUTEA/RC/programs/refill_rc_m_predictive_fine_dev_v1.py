#!/usr/bin/env python3
"""Scheduling-only refill accepting Slurm %R's parenthesized pending reasons."""
import json
import os
import re
import subprocess
import submit_rc_m_predictive_and_fine_c128_v1 as C


def main():
    C.guard()
    submission=C.read(C.ROOT/'submission.json')
    allowed={value['job_id']:name for name,value in submission['stages'].items()}
    data=subprocess.check_output(['squeue','-u',os.environ.get('USER','ap7811'),'-h','-r','-o','%i|%P|%T|%R'],text=True)
    rows=[x.split('|',3) for x in data.splitlines() if x.strip()]
    slots=max(0,4-sum('dev_' in partition for _,partition,_,_ in rows))
    priorities={'context_pilot':0,'predict_seal':0,'predict_join':0,'predict_check':0,'context_join':0,'fine_join':0,'closure':0,
                'predict_probe':1,'predict_endpoint':1,'fine_bridge':1,'phase_existing':2,'phase_join':0,'context':3}
    candidates=[]
    for job,partition,state,reason in rows:
        parent=job.split('_')[0]
        if parent not in allowed or state!='PENDING' or 'dev_' in partition:continue
        if reason.strip().strip('()') not in {'Priority','Resources','None'}:continue
        assert re.fullmatch(r'\d+(?:_\d+)?',job)
        candidates.append((priorities[allowed[parent]],job))
    changes=[]
    for _,job in sorted(candidates)[:slots]:
        before=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        if 'JobState=PENDING' not in before:continue
        assert 'TimeLimit=00:10:00' in before and 'cpu=4' in before and 'mem=32G' in before and 'gres/gpu' not in before
        changed=subprocess.run(['scontrol','update','JobId='+job,'Partition=dev_cpuonly,cpuonly'],capture_output=True,text=True)
        if changed.returncode:
            changes.append(dict(job=job,status='REJECTED',stderr=changed.stderr));break
        after=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        assert 'Partition=dev_cpuonly,cpuonly' in after and 'cpu=4' in after and 'mem=32G' in after
        changes.append(dict(job=job,status='VERIFIED',before=before.strip(),after=after.strip()))
    record=dict(utc=C.utc(),helper=C.bind(__file__),available_slots=slots,changes=changes,
                scheduling_only=True,scientific_sources_unchanged=True)
    with (C.ROOT/'dev_refill_format_repair.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(dict(utc=record['utc'],available_slots=slots,changes=[{k:v for k,v in x.items() if k not in ['before','after']} for x in changes])),flush=True)


if __name__=='__main__':main()
