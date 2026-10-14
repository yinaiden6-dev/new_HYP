from evaluation.paths import project_path
import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

from .common import ROOT, WORK, claim, key, read, save, sha, signature


def ready(job, inventories=None):
    inventories = {} if inventories is None else inventories
    if any(not project_path(p).exists() for p in job.get('requires_files', [])):
        return False
    for dataset, splits in job.get('requires', {}).items():
        path = WORK/dataset/'manifest.json'
        if not path.exists():
            return False
        if dataset not in inventories:
            inventories[dataset] = (read(path)['rows'], {p.stem for p in (WORK/dataset/'features').glob('*.pt')})
        rows, available = inventories[dataset]
        rows = [r for r in rows if r['split'] in splits]
        start, stop = job.get('query_range', [0, len(rows)])
        for row in rows[start:stop]:
            if key(row) not in available:
                return False
    return True


def free(gpu):
    if gpu.startswith('cpu'):
        return True
    result = subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],
                            text=True,capture_output=True)
    return result.returncode == 0 and int(result.stdout.strip()) < 200


def fill_roma(gpu, environment, lease):
    dataset = WORK/'forb32'
    if not (dataset/'manifest.json').exists() or not free(gpu):
        return
    available = {p.stem for p in (dataset/'roma512').glob('*.pt')}
    if all(key(r) in available for r in read(dataset/'manifest.json')['rows']):
        return
    with claim(lease) as leased:
        if not leased or not free(gpu):
            return
        command = ['bash',str(ROOT/'scripts/forb_python.sh'),'-m','drugpack.method.roma',
                   '--precision','fp16','--datasets','forb32','--splits','transfer','--max-queries','1']
        log = WORK/'intermediate/logs'/f'idle_roma_{socket.gethostname()}_{gpu}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--jobs', default=str(ROOT/'configs/method/jobs.json'))
    parser.add_argument('--idle-roma', action='store_true')
    args = parser.parse_args()
    assert not args.idle_roma or not args.gpu.startswith('cpu')
    host = socket.gethostname()
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES='' if args.gpu.startswith('cpu') else args.gpu)
    lease = WORK/'intermediate/leases'/f'{host}_{args.gpu}.lock'
    while True:
        pending = False
        trained = False
        readiness = {}
        inventories = {}
        for job in sorted(read(args.jobs),key=lambda j:j.get('priority',0)):
            status = WORK/'intermediate/jobs'/(job['id']+'.json')
            old = read(status) if status.exists() else {}
            if old.get('status') in ('complete','failed'):
                continue
            pending = True
            dependency = signature([job.get('requires',{}),job.get('requires_files',[]),job.get('query_range')])
            if dependency not in readiness:
                readiness[dependency] = ready(job, inventories)
            if not readiness[dependency] or not free(args.gpu):
                continue
            with claim(status.with_suffix('.lock')) as held, claim(lease) as leased:
                if not held or not leased:
                    continue
                old = read(status) if status.exists() else {}
                if old.get('status') in ('complete','failed') or not free(args.gpu):
                    continue
                log = WORK/'intermediate/logs'/(job['id']+'.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                info = dict(status='running',job=job,host=host,gpu=args.gpu,start=time.time(),
                            sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'drugpack/method').glob('*.py')})
                command = ['bash',str(ROOT/'scripts/forb_python.sh')]+[a.replace('rebut.train.', 'drugpack.method.') for a in job['arguments']]
                with log.open('a') as stream:
                    process = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                    info['pid'] = process.pid
                    save(status,info)
                    code = process.wait()
                info.update(status='complete' if code == 0 else 'failed',returncode=code,end=time.time())
                save(status,info)
                print(job['id'],info['status'],flush=True)
                trained = True
                break
        if not pending:
            break
        if args.idle_roma and not trained:
            fill_roma(args.gpu,environment,lease)
        time.sleep(15)


if __name__ == '__main__':
    main()
