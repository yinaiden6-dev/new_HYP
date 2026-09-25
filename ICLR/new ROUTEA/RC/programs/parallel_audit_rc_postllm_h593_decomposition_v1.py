#!/usr/bin/env python3
"""Schedule the unchanged independent audit over disjoint query outputs.

Original serial join must wait for this worker. No scoring, model inference,
training, or label reads occur here. Each process owns one audit_rows file.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import time
import json

import audit_rc_postllm_h593_decomposition_v1 as A

CONTEXT = None


def work(index):
    protocol, pb, manifest, started, budget = CONTEXT
    if time.monotonic()-started > budget-30:
        return None
    record = protocol['rows'][index]
    audit = A.audit_query(record, manifest['rows'][index], protocol, pb)
    return dict(index=index, source=audit['source'], max_logit_error=audit['max_logit_error'])


def main():
    global CONTEXT
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--budget',type=float,default=440)
    args = ap.parse_args()
    started = time.monotonic()
    protocol = A.read(A.OUT/'protocol.json')
    pb = A.bind(A.OUT/'protocol.json')
    for b in protocol['code_sources']:
        A.checked(b)
    manifest = A.read(A.checked(protocol['manifest']))
    CONTEXT = protocol,pb,manifest,started,args.budget
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('fork')) as pool:
        for item in pool.map(work,range(593),chunksize=1):
            if item is not None:
                completed.append(item)
                if len(completed)%32 == 0:
                    print(json.dumps(dict(audited=len(completed),total=593,elapsed=time.monotonic()-started)),flush=True)
    if len(completed) != 593:
        print(json.dumps(dict(status='PARTIAL_PARALLEL_AUDIT',audited=len(completed))),flush=True)
        sys.exit(75)
    A.write(A.OUT/'parallel_audit_validation.json',dict(status='ALL593_PARALLEL_INDEPENDENT_AUDIT_PASS',
        protocol=pb,wrapper=A.bind(Path(__file__)),unchanged_validator=A.bind(Path(A.__file__)),
        processes=args.workers,elapsed_seconds=time.monotonic()-started,
        rows=[A.bind(A.OUT/'audit_rows'/f'{i:04d}.json') for i in range(593)],held_label_reads=0))
    print(json.dumps(dict(status='ALL593_PARALLEL_INDEPENDENT_AUDIT_PASS',elapsed_seconds=time.monotonic()-started)),flush=True)


if __name__ == '__main__':
    main()
