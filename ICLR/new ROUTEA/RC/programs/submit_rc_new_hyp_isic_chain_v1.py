#!/usr/bin/env python3
"""Submit resumably: reference -> bootstrap RAW -> RoMa pilot -> full -> join."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_new_hyp_isic_transfer_v1/submission'


def read(p): return json.loads(Path(p).read_text())


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2); f.write('\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dry-run', action='store_true')
    dry = parser.parse_args().dry_run
    ref = read(OUT/'reference_submission_v1.json')['job_id']
    ids, records = {}, []
    stages = ['raw_pilot', 'roma_pilot', 'raw_rest', 'roma_rest', 'join']
    for i, stage in enumerate(stages):
        if stage == 'raw_pilot': script, array, dep, limit, arguments = 'bootstrap', '0', ref, '00:10:00', []
        elif stage == 'roma_pilot': script, array, dep, limit, arguments = 'inference', '0', ids['raw_pilot'], '00:15:00', ['roma']
        elif stage == 'raw_rest': script, array, dep, limit, arguments = 'inference', '1-67%46', ids['roma_pilot'], '00:10:00', ['raw']
        elif stage == 'roma_rest': script, array, dep, limit, arguments = 'inference', '1-67%46', ids['raw_rest'], '00:15:00', ['roma']
        else: script, array, dep, limit, arguments = 'join', None, ids['roma_pilot']+':'+ids['roma_rest'], '00:10:00', []
        launcher = ROOT / f'slurm/rc_new_hyp_isic_{script}_v1.sbatch'
        argv = ['sbatch', '--parsable', '--partition=dev_accelerated,accelerated', '--time='+limit,
                '--dependency=afterok:'+dep, '--kill-on-invalid-dep=yes']
        if array is not None: argv += ['--array='+array]
        argv += [str(launcher), *arguments]
        receipt = OUT / ('chain_'+stage+'_v1.json')
        if dry:
            job = str(9900000+i)
            value = dict(stage=stage, synthetic_job_id=job, argv=argv, dry_run=True)
        elif receipt.exists():
            value = read(receipt); job = value['job_id']
            assert value['argv'] == argv and value['launcher_sha256'] == sha(launcher), 'SUBMITTED_CHAIN_DRIFT'
        else:
            p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
            job = p.stdout.strip().split(';')[0]; assert job.isdigit()
            value = dict(stage=stage, job_id=job, argv=argv, launcher_sha256=sha(launcher),
                         submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), stderr=p.stderr)
            write(receipt, value)
        if not dry:
            spool = OUT / f'{stage}_job{job}.sbatch'
            if not spool.exists():
                subprocess.run(['scontrol','write','batch_script',job,str(spool.relative_to(ROOT))], cwd=ROOT, check=True, capture_output=True)
            assert spool.read_bytes() == launcher.read_bytes(), 'SPOOL_DRIFT'
        ids[stage] = job; records.append(value); print(json.dumps(value), flush=True)
    if dry: return
    final = OUT / 'inference_chain_receipt_v1.json'
    if not final.exists():
        write(final, dict(reference_job=ref, jobs=ids, records=records, query_count=537, reference_count=390,
            shards=68, last_shard_size=1, concurrency_per_stage=46, spool_byte_identical=True,
            authority_creation='bootstrap first checks390-reference validation, freezes inference authority, then encodes queries',
            scientific_result_available=False))


if __name__ == '__main__': main()
