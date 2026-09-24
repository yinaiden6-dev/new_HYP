#!/usr/bin/env python3
"""Four-TRAIN-query, full-C128 factorial diagnosis of a constant M student."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import rc_pair_quality_core_v1 as C
import run_rc_h593_pair_quality_cpu_v1 as Q

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/rc_colnomic_m_train_fit_v1'
AUTH = ROOT/'registry/rc_colnomic_m_train_fit_authority_v1_20260923.json'
PARENT = ROOT/'registry/rc_fold0_mass_teacher_authority_v1_20260923.json'
PLAN = ROOT/'plan/RC_COLNOMIC_M_TRAIN_FIT_DIAGNOSTIC_V1_20260923.md'
LAUNCH = ROOT/'slurm/rc_colnomic_m_train_fit_v1.sbatch'
ARMS = ('ORIGINAL', 'CALIBRATED', 'CENTERED', 'CALIBRATED_CENTERED')
SNAPSHOTS = (0, 16, 64, 128, 256)
EPS = 1e-8
read, write, save, bind = Q.read, Q.write, Q.save, Q.bind


def checked(b):
    assert bind(b['path']) == b, ('SHA_DRIFT', b['path'])
    return Path(b['path'])


def objective(pred, teacher, centered):
    residual = torch.log(pred+EPS)-torch.log(teacher+EPS)
    if centered:
        residual = residual-residual.mean()
    return residual.square().mean()


def packed_mass(model, packed, detail=False):
    x = packed['x']
    hidden = torch.tanh(model.down(x))
    y = torch.sigmoid(model.up(hidden)).flatten()
    means = torch.stack([v.mean() for v in y.split(packed['lengths'])]).reshape(-1, 2)
    mass = means.prod(1).sqrt()
    d = {}
    if detail:
        with torch.no_grad():
            deriv = 1-hidden.square()
            d = dict(tanh_saturated_fraction=float((deriv < .01).double().mean()),
                     tanh_derivative_mean=float(deriv.mean()),
                     quality_min=float(y.min()), quality_max=float(y.max()),
                     hidden_unit_derivative_mean=deriv.mean(0).tolist())
    return mass, d


def metrics(pred, teacher):
    pred, teacher = np.asarray(pred, dtype=np.float64), np.asarray(teacher, dtype=np.float64)
    lp, lt = np.log(pred+EPS), np.log(teacher+EPS)
    err = lp-lt
    centered = float(np.var(err))
    target_var = float(np.var(lt))
    i, j = np.triu_indices(len(pred), 1)
    dt, dp = lt[i]-lt[j], lp[i]-lp[j]
    mask = abs(dt) > 1e-12
    sign = np.sign(dt[mask])*np.sign(dp[mask])
    agreement = float(np.where(abs(dp[mask]) <= 1e-12, .5, sign > 0).mean())
    return dict(log_mse=float(np.mean(err**2)), centered_log_mse=centered,
                teacher_centered_log_variance=target_var, normalized_centered_mse=centered/target_var,
                mean_log_error=float(err.mean()), pair_order_agreement=agreement,
                predicted_M_std=float(pred.std()), teacher_M_std=float(teacher.std()),
                predicted_logM_std=float(lp.std()), teacher_logM_std=float(lt.std()))


def synthetic():
    # Batch packing, centered loss, and gradients versus ordinary pair graphs.
    gen = torch.Generator().manual_seed(91)
    q = dict(z=torch.randn(5, 128, generator=gen, dtype=torch.float64),
             xy=torch.rand(5, 2, generator=gen, dtype=torch.float64))
    refs = [dict(z=torch.randn(7+i, 128, generator=gen, dtype=torch.float64),
                 xy=torch.rand(7+i, 2, generator=gen, dtype=torch.float64)) for i in range(3)]
    rels = [z for r in refs for z in C.relation(q['z'], r['z'], q['xy'], r['xy'], 'PAIR')]
    packed = dict(x=torch.cat(rels), lengths=[len(x) for x in rels])
    targets = torch.tensor([.001, .01, .1], dtype=torch.float64)
    errs = []
    for centered in (False, True):
        m = C.Quality()
        with torch.no_grad():
            m.up.weight.normal_(generator=gen, std=.1)
        n = copy.deepcopy(m)
        y, _ = packed_mass(m, packed)
        z = torch.stack([C.pair(n, q, r, 'PAIR')[0] for r in refs])
        objective(y, targets, centered).backward()
        objective(z, targets, centered).backward()
        e = max(float((a.grad-b.grad).abs().max()) for a, b in zip(m.parameters(), n.parameters()))
        assert e < 2e-12 and float((y-z).abs().max()) < 2e-12
        met = metrics(y.detach().numpy(), targets.numpy())
        assert abs(met['log_mse']-met['centered_log_mse']-met['mean_log_error']**2) < 2e-12
        errs.append(e)
    ideal = metrics(targets.numpy(), targets.numpy())
    assert ideal['normalized_centered_mse'] == 0 and ideal['pair_order_agreement'] == 1
    return dict(status='PACKED_FORWARD_GRADIENT_AND_NUMPY_METRICS_PASS', gradient_errors=errs)


def prepare():
    assert not AUTH.exists(), 'FROZEN_AUTHORITY_EXISTS'
    old = read(PARENT)
    train = read(checked(old['train_inputs']))
    assert train['teacher_included'] and not train['identity_labels_included']
    roles = {x['query_id']: x['component'] for x in read(checked(old['train_roles']))['records']}
    seen, records = set(), []
    for rec in sorted(train['records'], key=lambda x: x['execution_ordinal']):
        component = roles[rec['query_id']]
        if component in seen:
            continue
        seen.add(component)
        records.append({**{k: rec[k] for k in ('query_id', 'execution_ordinal', 'query_image_key',
                                             'candidate_physical_rows', 'reference_keys', 'teacher_M')},
                        'component': component})
        if len(records) == 4:
            break
    assert len(records) == 4 and len(seen) == 4
    ready = read(checked(old['ready']))
    keys = {key for rec in records for key in [rec['query_image_key']]+rec['reference_keys']}
    features = {key: ready['features'][key] for key in sorted(keys)}
    prior = Path(old['train_inputs']['path']).parent
    replay = [bind(prior/'predictions/train'/f"query{rec['execution_ordinal']:03d}.pt") for rec in records]
    value = math.exp(float(np.log(np.asarray([rec['teacher_M'] for rec in records])+EPS).mean()))-EPS
    assert 0 < value < 1
    panel = dict(records=records, features=features, selection='first four distinct TRAIN components by execution_ordinal',
                 teacher_included=True, identity_labels_included=False, held_reads=0)
    write(OUT/'panel.json', panel)
    test = synthetic()
    write(AUTH, dict(status='M_TRAIN_FIT_DIAGNOSTIC_FROZEN', sources=[bind(p) for p in
                     (Path(__file__), Path(C.__file__), Path(Q.__file__), PLAN, LAUNCH)],
                     parent=bind(PARENT), train_source=old['train_inputs'], roles_source=old['train_roles'],
                     ready_source=old['ready'], panel=bind(OUT/'panel.json'),
                     previous_model=bind(prior/'final_model.pt'), previous_train_predictions=replay,
                     arms=list(ARMS), calibrated_mass=value, calibrated_bias=math.log(value/(1-value)),
                     steps=256, snapshots=list(SNAPSHOTS), lr=.001, weight_decay=.001, seed=17,
                     chunk_seconds=450, max_chunks=16, new_encoder_forwards=0, new_RoMa_forwards=0,
                     held_reads=0, training_queries=4, candidates_per_query=128, precision='FP64'))
    write(OUT/'preflight.json', dict(**test, authority=bind(AUTH)))
    print(json.dumps(dict(status='PREPARED', query_ordinals=[r['execution_ordinal'] for r in records],
                         calibrated_mass=value, features=len(features), preflight=test)), flush=True)


def guard():
    a = read(AUTH)
    for b in a['sources']+[a['panel'], a['previous_model']]+a['previous_train_predictions']:
        checked(b)
    assert read(OUT/'preflight.json')['authority'] == bind(AUTH)
    panel = read(a['panel']['path'])
    allow = {Path(b['path']).resolve() for b in panel['features'].values()}
    allow.update(Path(b['path']).resolve() for b in [a['previous_model']]+a['previous_train_predictions'])
    def audit(event, args):
        if event == 'socket.connect':
            assert not isinstance(args[1], tuple), 'OFFLINE'
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if ROOT/'results' in path.parents:
            assert OUT in path.parents or path in allow, ('UNAUTHORIZED_DATA', str(path))
    sys.addaudithook(audit)
    return a, panel


def checkpoint(path, models, opts, step, history, a):
    save(path, dict(authority=bind(AUTH), step=step, history=history,
                   models={k: m.state_dict() for k, m in models.items()},
                   optimizers={k: op.state_dict() for k, op in opts.items()}), mutable=True)


def run():
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available(), 'CPU_SLURM_ONLY'
    a, panel = guard()
    lock = (OUT/'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    start = time.monotonic()
    def receipt(phase, step=0):
        v = dict(status='NORMAL_CHUNK', phase=phase, step=step, seconds=time.monotonic()-start, authority=bind(AUTH))
        write(OUT/'chunks'/f"{os.environ['SLURM_JOB_ID']}.json", v)
        write(OUT/'status.json', v, mutable=True)
        print(json.dumps(v), flush=True)
    # Only selected token files can be read by the bank.
    Q.checked = checked
    bank = Q.Bank(panel['features'], 'COL_ONLY', 'cpu')
    for idx, rec in enumerate(panel['records']):
        dest = OUT/'relations'/f'query{idx}.pt'
        meta = dest.with_suffix('.json')
        if meta.exists():
            checked(read(meta)['payload'])
            continue
        if time.monotonic()-start > a['chunk_seconds']:
            receipt('cache'); return
        rels = []
        q = bank(rec['query_image_key'])
        with torch.no_grad():
            for key in rec['reference_keys']:
                r = bank(key)
                rels.extend(C.relation(q['z'], r['z'], q['xy'], r['xy'], 'PAIR'))
        assert len(rels) == 256
        packed = dict(authority=bind(AUTH), query_id=rec['query_id'], x=torch.cat(rels),
                      lengths=[len(x) for x in rels], teacher=torch.tensor(rec['teacher_M'], dtype=torch.float64))
        save(dest, packed, mutable=True)
        write(meta, dict(payload=bind(dest), query_id=rec['query_id'], tokens=len(packed['x']),
                         bytes=dest.stat().st_size, precision='FP64', candidate_count=128))
        print(json.dumps(dict(event='RELATIONS_SAVED', index=idx, seconds=time.monotonic()-start)), flush=True)
        del packed, rels
    del bank
    data = [torch.load(OUT/'relations'/f'query{i}.pt', weights_only=True, mmap=True) for i in range(4)]
    for d, r in zip(data, panel['records']):
        assert d['authority'] == bind(AUTH) and d['query_id'] == r['query_id'] and len(d['lengths']) == 256
    if not (OUT/'old_replay.json').exists():
        model = C.Quality()
        old = torch.load(a['previous_model']['path'], weights_only=True)
        assert old['authority'] == a['parent'] and old['step'] == 2000
        model.load_state_dict(old['model'])
        rows = []
        for d, b in zip(data, a['previous_train_predictions']):
            with torch.no_grad():
                m, det = packed_mass(model, d, True)
            original = torch.load(b['path'], weights_only=True)
            assert original['query_id'] == d['query_id'] and original['model'] == a['previous_model']
            error = float((m-original['M']).abs().max())
            assert error < 2e-12, ('OLD_REPLAY_DRIFT', error)
            rows.append(dict(query_id=d['query_id'], replay_max_error=error, **det,
                             metrics=metrics(m.numpy(), d['teacher'].numpy()), predicted_M=m.tolist()))
        write(OUT/'old_replay.json', dict(status='OLD_TRAIN_PREDICTION_PARITY_PASS', rows=rows,
                                        authority=bind(AUTH)))
        del model
    models, opts = {}, {}
    for arm in ARMS:
        model = C.Quality(seed=a['seed'])
        if arm.startswith('CALIBRATED'):
            with torch.no_grad(): model.up.bias.fill_(a['calibrated_bias'])
        models[arm] = model
        opts[arm] = torch.optim.AdamW(model.parameters(), lr=a['lr'], weight_decay=a['weight_decay'])
    cp = OUT/'checkpoint.pt'
    step, history = 0, []
    if cp.exists():
        p = torch.load(cp, weights_only=True)
        assert p['authority'] == bind(AUTH)
        step, history = p['step'], p['history']
        for arm in ARMS:
            models[arm].load_state_dict(p['models'][arm]); opts[arm].load_state_dict(p['optimizers'][arm])
    while True:
        if step in SNAPSHOTS:
            for arm in ARMS:
                dest = OUT/'snapshots'/f'{arm}_{step:03d}.json'
                if dest.exists(): continue
                if time.monotonic()-start > a['chunk_seconds']:
                    checkpoint(cp, models, opts, step, history, a); receipt('snapshot', step); return
                rows = []
                for d in data:
                    with torch.no_grad(): m, det = packed_mass(models[arm], d, True)
                    rows.append(dict(query_id=d['query_id'], predicted_M=m.tolist(), **det,
                                     metrics=metrics(m.numpy(), d['teacher'].numpy())))
                summary = {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
                write(dest, dict(authority=bind(AUTH), arm=arm, step=step, rows=rows, mean_query_metrics=summary))
                print(json.dumps(dict(event='SNAPSHOT', arm=arm, step=step, **summary)), flush=True)
        if step == a['steps']: break
        if time.monotonic()-start > a['chunk_seconds']:
            checkpoint(cp, models, opts, step, history, a); receipt('fit', step); return
        epoch, offset = divmod(step, 4)
        gen = torch.Generator().manual_seed(a['seed']+1000003*epoch)
        qi = int(torch.randperm(4, generator=gen)[offset])
        d = data[qi]
        for arm in ARMS:
            model, opt = models[arm], opts[arm]
            opt.zero_grad()
            m, _ = packed_mass(model, d)
            loss = objective(m, d['teacher'], 'CENTERED' in arm)
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
            grad = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
            opt.step()
            history.append(dict(step=step+1, query_id=d['query_id'], arm=arm, loss=float(loss.detach()), gradient_norms=grad))
        step += 1
        if step % 4 == 0:
            checkpoint(cp, models, opts, step, history, a)
            print(json.dumps(dict(event='UPDATE', step=step, seconds=time.monotonic()-start)), flush=True)
    checkpoint(cp, models, opts, step, history, a)
    # Engineering positive control: free log-mass parameter per TRAIN pair.
    target = torch.stack([d['teacher'] for d in data])
    logtarget = torch.log(target+EPS)
    free = torch.nn.Parameter(torch.full_like(logtarget, float(logtarget.mean())))
    opt = torch.optim.Adam([free], lr=.05)
    for _ in range(2000):
        opt.zero_grad(); loss = (free-logtarget).square().mean(); loss.backward(); opt.step()
    free_pred = (free.detach().exp()-EPS).numpy()
    lookup = [metrics(p, t.numpy()) for p, t in zip(free_pred, target)]
    assert max(x['normalized_centered_mse'] for x in lookup) < 1e-6
    write(OUT/'lookup_control.json', dict(label='PAIR_LOOKUP_ENGINEERING_CONTROL_NOT_TOKEN_INFORMATION_EVIDENCE',
                                         steps=2000, rows=lookup, predicted_M=free_pred.tolist()))
    final = {arm: read(OUT/'snapshots'/f'{arm}_256.json') for arm in ARMS}
    result = dict(status='TRAIN_FIT_DIAGNOSTIC_COMPLETE', authority=bind(AUTH), train_queries=4,
                  candidates=128, held_reads=0, old_replay=read(OUT/'old_replay.json'),
                  arms=final, lookup_control=bind(OUT/'lookup_control.json'),
                  small_panel_fit_gate={arm: v['mean_query_metrics']['normalized_centered_mse'] <= .25
                                        and v['mean_query_metrics']['pair_order_agreement'] >= .8 for arm, v in final.items()},
                  boundary='TRAIN-only teacher diagnostic; neither held retrieval gain nor token information absence established')
    write(OUT/'result.json', result)
    write(OUT/'validation.json', dict(status='TRAIN_ONLY_FULL128_REPLAY_AND_OPTIMIZER_PASS', authority=bind(AUTH), result=bind(OUT/'result.json')))
    receipt('complete', step)


def advance():
    a = read(AUTH)
    for b in a['sources']: checked(b)
    jid = os.environ['SLURM_JOB_ID']
    chunk = read(OUT/'chunks'/f'{jid}.json')
    assert chunk['status'] == 'NORMAL_CHUNK' and chunk['authority'] == bind(AUTH)
    if (OUT/'validation.json').exists():
        checked(read(OUT/'validation.json')['result'])
        print('COMPLETE_NO_FURTHER_SUBMISSION', flush=True); return
    lock = (OUT/'submit.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = OUT/'continuations'/f'{jid}.json'
    if receipt.exists(): return
    count = len(list((OUT/'chunks').glob('*.json')))
    if count >= a['max_chunks']:
        write(OUT/'status.json', dict(status='STOPPED_AT_CHUNK_BUDGET', chunks=count, authority=bind(AUTH)), mutable=True)
        return
    args = ['sbatch', '--parsable', '--dependency=afterok:'+jid, str(LAUNCH)]
    p = subprocess.run(args, capture_output=True, text=True, timeout=45, check=True)
    job = p.stdout.strip().split(';')[0]
    assert job.isdigit(), p.stdout
    write(receipt, dict(previous=jid, job_id=job, args=args, authority=bind(AUTH)))
    print(json.dumps(dict(event='CONTINUATION_SUBMITTED', job_id=job)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run', 'advance'))
    globals()[parser.parse_args().stage]()
