"""Launch the sealed inference shards from a relocated transfer directory."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs='+',type=int,default=[0])
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or min(args.gpus)<0:parser.error('GPU IDs must be unique nonnegative integers')
    active=json.loads((ROOT/'cache/content_box_ce_polish18/deployment/active.json').read_text())
    deployment=Path(active['path'])
    if not deployment.is_absolute():deployment=ROOT/'cache/content_box_ce_polish18/deployment'/deployment
    query_count=len(json.loads((deployment/'manifest.json').read_text())['queries'])
    timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs=deployment/'logs'/('resume_'+timestamp)
    records=[]
    for index,gpu in enumerate(args.gpus):
        command=[args.python,'-B',str(ROOT/'scripts/run_content_box_ce_polish18.py'),'run','--deployment',str(deployment),
            '--workers',str(len(args.gpus)),'--worker-index',str(index),'--device','cuda:0','--cpu-threads',str(args.cpu_threads)]
        records.append(dict(worker=index,gpu=gpu,command=command,log=str(logs/f'gpu{gpu}.log')))
    if args.dry_run:
        print(json.dumps(records,ensure_ascii=False,indent=2));return
    logs.mkdir(parents=True,exist_ok=False)
    children=[]
    try:
        for record in records:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(record['gpu']),OPENBLAS_NUM_THREADS=str(args.cpu_threads),
                OMP_NUM_THREADS=str(args.cpu_threads),PYTHONDONTWRITEBYTECODE='1')
            with Path(record['log']).open('w') as stream:
                child=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            children.append(child);record['pid']=child.pid
        (logs/'processes.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
        print('Workers started. Logs: '+str(logs),flush=True)
        last=0.
        while any(child.poll() is None for child in children):
            if time.monotonic()-last>=60:
                completed=sum(1 for p in (deployment/'queries').glob('*/prediction.json'))
                print(json.dumps(dict(completed_queries=completed,total_queries=query_count,
                    running_workers=sum(child.poll() is None for child in children))),flush=True);last=time.monotonic()
            time.sleep(1)
        failures=[record['worker'] for record,child in zip(records,children) if child.returncode!=0]
        if failures:raise RuntimeError('Workers failed: '+str(failures)+'; inspect '+str(logs))
        for stage in ['verify','report']:
            subprocess.run([args.python,'-B',str(ROOT/'scripts/run_content_box_ce_polish18.py'),stage,
                '--deployment',str(deployment),'--cpu-threads',str(args.cpu_threads)],cwd=ROOT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS=str(args.cpu_threads),OMP_NUM_THREADS=str(args.cpu_threads)),check=True)
        print('Completed. Results: '+str(deployment),flush=True)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':
    main()
