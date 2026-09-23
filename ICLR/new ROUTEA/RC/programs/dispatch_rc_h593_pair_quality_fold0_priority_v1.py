#!/usr/bin/env python3
"""Execution-only fold-0 continuation of the unchanged pair-quality worker."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import dispatch_rc_h593_pair_quality_cpu_v1 as D
from rc_h593_pair_quality_execution_source_compat_v1 import approved_execution_source_change

ROOT = D.ROOT
OUT = D.OUT
EXEC_AUTH = ROOT / 'registry/rc_h593_pair_quality_fold0_priority_execution_v1_20260923.json'
INDICES = tuple(range(6))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(binding):
    if bind(binding['path']) != binding:
        raise RuntimeError('SHA_DRIFT:' + binding['path'])
    return Path(binding['path'])


def validated(index, authority):
    path = OUT / f'fit{index:02d}/validation.json'
    if not path.exists():
        return None
    value = D.read(path)
    if value['status'] != 'PAIR_QUALITY_FOLD_NUMPY_PASS' or value['authority'] != bind(D.AUTH):
        raise RuntimeError('VALIDATION_BINDING:' + str(index))
    payload = D.read(checked(value['payload']))
    if payload['authority'] != bind(D.AUTH) or payload['config'] != authority['configs'][index]:
        raise RuntimeError('CONFIG_BINDING:' + str(index))
    if payload['steps'] != authority['steps'] or len(payload['predictions']) != 119:
        raise RuntimeError('INCOMPLETE_FOLD:' + str(index))
    checked(payload['model'])
    if payload['heldout_label_reads'] != 0:
        raise RuntimeError('UNEXPECTED_LABEL_READS')
    return dict(index=index, config=payload['config'], validation=bind(path),
                payload=value['payload'], model=payload['model'],
                predictions=payload['predictions'])


def main():
    lock = (OUT / 'dispatcher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    execution = D.read(EXEC_AUTH)
    if execution['status'] != 'FOLD0_PRIORITY_EXECUTION_AUTHORIZED' or execution['indices'] != list(INDICES):
        raise RuntimeError('EXECUTION_SCOPE')
    checked(execution['scientific_authority'])
    for binding in execution['sources']:
        checked(binding)
    authority = D.read(D.AUTH)
    for binding in authority['sources']:
        if bind(binding['path']) != binding and not approved_execution_source_change(binding):
            raise RuntimeError('SCIENTIFIC_SOURCE_DRIFT:' + binding['path'])
    if any(authority['configs'][i]['fold'] != 0 for i in INDICES):
        raise RuntimeError('NOT_FIRST_FOLD')
    ledger = OUT / 'dispatch.json'
    state = D.read(ledger)
    if any(job['index'] not in INDICES for job in state['active']):
        raise RuntimeError('LATER_FOLD_NOT_PAUSED')
    if (len({job['index'] for job in state['active']}) != len(state['active'])
            or len({job['job_id'] for job in state['active']}) != len(state['active'])):
        raise RuntimeError('DUPLICATE_ACTIVE_FIT')
    state['priority_indices'] = list(INDICES)
    state['partition'] = 'dev_cpuonly'
    start = time.monotonic()
    while time.monotonic() - start < 3 * 86400:
        try:
            active = []
            for job in state['active']:
                status = D.state(job['job_id'])
                if status == 'COMPLETED':
                    if validated(job['index'], authority) is not None:
                        state['complete'] = sorted(set(state['complete']) | {job['index']})
                elif status in D.WAITING:
                    active.append(job)
                else:
                    raise RuntimeError('FIT_FAILED:' + str(job) + ':' + status)
            state['active'] = active
            done = [i for i in INDICES if validated(i, authority) is not None]
            state['complete'] = sorted(set(state['complete']) | set(done))
            D.write(ledger, state)
            # At most the two remaining first-fold fits; no later-fold submissions.
            while len(state['active']) < 2:
                occupied = {job['index'] for job in state['active']} | set(done)
                available = [i for i in INDICES if i not in occupied]
                if not available:
                    break
                index = available[0]
                attempts = state['chunks'].get(str(index), 0)
                if attempts >= authority['max_chunks_per_config']:
                    raise RuntimeError('CHUNK_BUDGET:' + str(index))
                job = D.submit('fit', index)
                state['active'].append(job)
                state['chunks'][str(index)] = attempts + 1
                D.write(ledger, state)
            status = dict(status='FIRST_FOLD_PRIORITY_RUNNING', time_utc=now(),
                          pid=os.getpid(), host=socket.gethostname(),
                          active=state['active'], complete=len(state['complete']), total=30,
                          priority_complete=len(done), priority_total=6,
                          priority_indices=list(INDICES), partition='dev_cpuonly',
                          later_folds_paused=True, execution_authority=bind(EXEC_AUTH))
            if len(done) == 6 and not state['active']:
                seal = OUT / 'fold0_priority_prediction_seal.json'
                sealed_fits = [validated(i, authority) for i in INDICES]
                for fit_record in sealed_fits:
                    for prediction in fit_record['predictions']:
                        checked(prediction)
                D.write(seal, dict(status='FIRST_FOLD_SIX_ARMS_SEALED',
                                  authority=bind(D.AUTH), execution_authority=bind(EXEC_AUTH),
                                  folds=[0], queries_per_arm=119,
                                  fits=sealed_fits,
                                  heldout_label_reads=0,
                                  scope='Fixed original first fold; not H593 OOF5 join'))
                status.update(status='FIRST_FOLD_COMPLETE_AWAITING_ANALYSIS', seal=bind(seal))
                D.write(OUT / 'dispatch_status.json', status)
                D.write(OUT / 'fold0_priority_status.json', status)
                return
            D.write(OUT / 'dispatch_status.json', status)
            D.write(OUT / 'fold0_priority_status.json', status)
        except Exception as error:
            message = str(error)
            retry = any(x in message for x in ('QOSMaxSubmitJobPerUserLimit', 'temporarily unavailable', 'Socket timed out'))
            status = dict(status='RETRY_SCHEDULER' if retry else 'STOPPED_ERROR',
                          time_utc=now(), error=message, pid=os.getpid(), host=socket.gethostname(),
                          partition='dev_cpuonly', execution_authority=bind(EXEC_AUTH))
            D.write(OUT / 'fold0_priority_status.json', status)
            D.write(OUT / 'dispatch_status.json', status)
            if not retry:
                raise
        time.sleep(30)
    D.write(OUT / 'fold0_priority_status.json', dict(status='STOPPED_TIME_BUDGET', time_utc=now()))


if __name__ == '__main__':
    main()
