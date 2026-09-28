#!/usr/bin/env python3
"""Replace only pending context shards by execution-identical node bundles."""
import re
import subprocess
import sys
import submit_rc_m_predictive_and_fine_c128_v1 as C


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def main():
    C.guard()
    target = C.ROOT/'packed_execution_submission.json'
    assert not target.exists(), 'Already submitted: inspect recorded jobs instead.'
    submission = C.read(C.ROOT/'submission.json')
    parent = submission['stages']['context']['job_id']
    join = submission['stages']['context_join']['job_id']
    snapshot = output(['squeue','-j',parent,'-h','-r','-o','%i|%T'])
    held=[];retained=[];before=[]
    for line in snapshot.splitlines():
        job,state=line.split('|');assert re.fullmatch(parent+r'_\d+',job)
        live=output(['scontrol','show','job',job,'-o'])
        if 'JobState=PENDING' in live:
            subprocess.run(['scontrol','hold',job],check=True)
            after=output(['scontrol','show','job',job,'-o'])
            assert 'JobState=PENDING' in after and 'Reason=JobHeldUser' in after, after
            held.append(job);before.append(dict(job=job,before=live,held=after))
        else:
            retained.append(job)
    assert held, 'No pending context work to bundle.'
    indices=sorted(int(j.split('_')[1]) for j in held)
    groups=[indices[i:i+13] for i in range(0,len(indices),13)]
    assert len(indices)==len(set(indices)) and not set(held)&set(retained)
    launcher=C.RC/'slurm/rc_m_predictive_fine_packed_v1.sbatch'
    plan=dict(status='PACKED_EXECUTION_ONLY',scientific_protocol=C.bind(C.ROOT/'protocol.json'),
        wrapper=C.bind(C.RC/'programs/run_rc_m_predictive_fine_packed_v1.py'),
        launcher=C.bind(launcher),groups=groups,child_threads=4,CPUs=52,memory='128G',
        time_limit='00:10:00',retained_original_jobs=retained,replaced_original_jobs=held,
        native_pilot_peak_memory_GB=5.05,original_scientific_shards=50,
        evidence='cpuonly allocates a whole 152-CPU node to each 4-CPU shard. Bundle up to13 independent existing shard processes; scientific calls, inputs, seeds, outputs and 4-thread math remain unchanged.')
    C.save(C.ROOT/'packed_execution_plan.json',plan,immutable=True)
    record=dict(status='HELD_FOR_REPLACEMENT',utc=C.utc(),plan=C.bind(C.ROOT/'packed_execution_plan.json'),original_states=before)
    C.save(target,record)
    argv=['sbatch','--parsable','--array=0-'+str(len(groups)-1)+'%'+str(len(groups)),str(launcher)]
    job=output(argv).split(';')[0];assert job.isdigit()
    record.update(status='SUBMITTED',job_id=job,argv=argv);C.save(target,record)
    live=output(['scontrol','show','job',job,'-o'])
    assert 'cpu=52' in live and 'mem=128G' in live and 'TimeLimit=00:10:00' in live and 'gres/gpu' not in live
    spool=C.ROOT/('submitted_packed_'+job+'.sbatch')
    subprocess.run(['scontrol','write','batch_script',job,str(spool)],check=True,capture_output=True)
    assert spool.read_bytes()==launcher.read_bytes()
    dependency='afterok:'+':'.join([job]+retained)
    subprocess.run(['scontrol','update','JobId='+join,'Dependency='+dependency],check=True)
    joined=output(['scontrol','show','job',join,'-o'])
    for prerequisite in [job]+retained:
        assert prerequisite in joined,(prerequisite,joined)
    record.update(status='DEPENDENCY_REWIRED',new_dependency=dependency,join_scheduler=joined,packed_scheduler=live,spool_matches=True)
    C.save(target,record)
    for old in held:
        liveold=output(['scontrol','show','job',old,'-o'])
        assert 'JobState=PENDING' in liveold and 'Reason=JobHeldUser' in liveold
        subprocess.run(['scancel',old],check=True)
    record.update(status='PACKED_EXECUTION_MIGRATION_PASS',replaced_cancelled_jobs=held)
    C.save(target,record)
    print(C.utc(),job,len(groups),'bundles;',len(held),'replaced;',retained,'retained',flush=True)


if __name__=='__main__':
    main()
