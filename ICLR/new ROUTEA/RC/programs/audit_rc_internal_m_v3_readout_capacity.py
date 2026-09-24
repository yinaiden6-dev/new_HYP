#!/usr/bin/env python3
"""Retrospective TRAIN16 scalar audit; never an encoder or probe evaluation.

Run with .venv-colpali/bin/python. Only existing JSON scalars are read. LP
witnesses, the previously inspected bias=0.01 witness, and CPU readout fits are
post hoc diagnostics, not frozen model results or evidence of useful real M.
"""
import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.optimize import LinearConstraint, brentq, linprog, minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_internal_m_learned_use_v3'
OUTPUT = ROOT / 'results/rc_internal_m_v3_fit_diagnostics/readout_capacity_audit.json'
BINDINGS = {}


def read(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    BINDINGS[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def load_endpoint(directory, train_ids):
    directory = Path(directory)
    validation = read(directory.parent / 'validation.json')
    expected = {Path(b['path']).name: b['sha256'] for b in validation['predictions']
                if Path(b['path']).parent == directory.resolve()}
    assert set(expected) == {q + '.json' for q in train_ids}
    rows = []
    for q in train_ids:
        path = directory / (q + '.json')
        d = read(path)
        assert BINDINGS[str(path.resolve())] == expected[path.name]
        assert d['query_id'] == q and d['held_label_reads'] == 0
        assert d['direct_M_in_head'] is False
        rows.append(d)
    return rows


def matrix(row):
    dec = row['decision']
    raw, content = np.asarray(row['raw_scores']), np.asarray(row['L'])
    indices = np.asarray(dec['challenger_positions'])
    winner = next(i for i in range(128) if i not in indices)
    target = dec['target_position']
    assert raw.shape == content.shape == (128,) and len(indices) == 127
    x = np.zeros((128, 3))
    x[indices, 0] = (raw[indices] - raw[winner]) / max(raw.std(), 1e-12)
    x[indices, 1] = ((content[indices] - content[winner]) /
                     (abs(content[indices]) + abs(content[winner]) + 1e-12))
    x[indices, 2] = 1
    assert np.max(abs(x @ dec['theta'] - dec['scores128'])) < 1e-12
    return x, winner, target, indices


def evaluate(rows, theta):
    losses, rescues, breaks, changed = [], [], [], []
    for row in rows:
        x, winner, target, indices = matrix(row)
        scores = x @ theta
        wrong = indices[indices != target]
        value = np.logaddexp(0, scores[wrong].max())
        if target != winner:
            value += np.logaddexp(0, -scores[target])
        pred = int(indices[np.argmax(scores[indices])]) if scores[indices].max() > 0 else winner
        q = row['query_id']
        if pred == target and target != winner:
            rescues.append(q)
        if pred != target and target == winner:
            breaks.append(q)
        if pred != winner:
            changed.append(q)
        losses.append(float(value))
    return dict(mean_cost1=float(np.mean(losses)), correct=8 + len(rescues) - len(breaks),
                rescues_vs_raw=rescues, breaks_vs_raw=breaks,
                raw_correct_loss_rate=len(breaks) / 8, changed_vs_raw=len(changed))


def linear_capacity(rows):
    good, bad, constraints = [], [], {}
    for row in rows:
        x, w, t, _ = matrix(row)
        q = row['query_id']
        (good if t == w else bad).append(q)
        constraints[q] = x[t] - x[np.arange(128) != t]
    assert len(good) == len(bad) == 8

    def solve(queries):
        a = np.concatenate([constraints[q] for q in queries])
        result = linprog(np.zeros(3), A_ub=-a, b_ub=-np.ones(len(a)),
                         bounds=[(None, None)] * 3, method='highs')
        if result.success:
            assert (a @ result.x).min() > .99999
            return result.x.tolist()
        assert result.status == 2, result.message
        return None

    individual = [q for q in bad if solve(good + [q]) is not None]
    best, witness, trials = [], None, 0
    for count in range(len(individual), 0, -1):
        for subset in itertools.combinations(individual, count):
            trials += 1
            candidate = solve(good + list(subset))
            if candidate is not None:
                best, witness = list(subset), candidate
                break
        if best:
            break
    return dict(method='Exhaustive descending subsets; 127 strict inequalities per required query',
                constraint='Preserve all eight RAW-correct identities; arbitrary common three weights',
                margin='Unit strict margin; unconstrained coefficient scale',
                all16_strictly_feasible=solve(good + bad) is not None,
                individually_feasible_rescue_preserving_raw8=individual,
                maximum_strict_correct_while_preserving_raw8=8 + len(best),
                one_maximum_rescue_subset=best, one_witness_theta=witness,
                subset_trials=trials, status='RETROSPECTIVE_CAPACITY_ONLY_NOT_MODEL_PERFORMANCE')


def convex_readout(rows):
    """Convex epigraph of unchanged COST1, without AdamW weight decay."""
    theta = np.asarray(rows[0]['decision']['theta'])
    mats = [matrix(r) for r in rows]
    scale = np.array([1., 25., 1.])
    constraints = []
    for j, (x, w, t, idx) in enumerate(mats):
        for i in idx[idx != t]:
            c = np.zeros(19)
            c[:3], c[3+j] = x[i] * scale, -1
            constraints.append(c)
    a = np.asarray(constraints)

    def objective(v):
        value = np.logaddexp(0, v[3:]).sum()
        gradient = np.zeros(19)
        gradient[3:] = expit(v[3:])
        for x, w, t, idx in mats:
            if t != w:
                f = x[t] * scale
                z = f @ v[:3]
                value += np.logaddexp(0, -z)
                gradient[:3] -= expit(-z) * f
        return value / 16, gradient / 16

    start = np.r_[theta / scale, [(x @ theta)[idx[idx != t]].max() for x, w, t, idx in mats]]
    result = minimize(objective, start, jac=True, method='SLSQP',
                      constraints=[LinearConstraint(a, -np.inf, 0)],
                      options=dict(maxiter=300, ftol=1e-11))
    stationarity = float(abs(objective(result.x)[1] + a.T @ result.multipliers).max())
    violation = float((a @ result.x).max())
    assert result.success and stationarity < 1e-4 and violation < 1e-7
    fitted = result.x[:3] * scale
    return dict(theta=fitted.tolist(), iterations=result.nit,
                status='RETROSPECTIVE_NUMERICAL_CONVEX_OPTIMUM_NOT_MODEL_PERFORMANCE',
                exact_certified_lower_bound=False, weight_decay_in_objective=False,
                kkt_stationarity_inf=stationarity, dual_min=float(result.multipliers.min()),
                complementarity_max=float(abs(result.multipliers * (a @ result.x)).max()),
                constraint_max_violation=violation, **evaluate(rows, fitted))


def cpu_readout(rows, initial, recipe):
    """Same full-batch AdamW COST1 recipe, common warm head for every endpoint."""
    import torch
    torch.set_num_threads(1)
    start = time.monotonic()
    mats = [matrix(r) for r in rows]
    x = torch.tensor(np.stack([v[0] for v in mats]), dtype=torch.float64)
    wrong = torch.ones((16, 128), dtype=torch.bool)
    targets = torch.tensor([v[2] for v in mats])
    bad = torch.tensor([v[1] != v[2] for v in mats])
    for i, (_, w, t, _) in enumerate(mats):
        wrong[i, w] = wrong[i, t] = False
    theta = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
    opt = torch.optim.AdamW([theta], lr=recipe['head_lr'], weight_decay=recipe['weight_decay'])
    for _ in range(2000):
        opt.zero_grad(set_to_none=True)
        z = x @ theta
        cost = torch.nn.functional.softplus(z.masked_fill(~wrong, -torch.inf).amax(dim=1))
        cost = cost + torch.nn.functional.softplus(-z[torch.arange(16), targets]) * bad
        cost.mean().backward()
        torch.nn.utils.clip_grad_norm_([theta], recipe['clip_norm'])
        opt.step()
    fitted = theta.detach().numpy()
    assert np.isfinite(fitted).all()
    return dict(status='RETROSPECTIVE_FIXED_ENDPOINT_CPU_REFIT_NOT_MODEL_PERFORMANCE',
                initial_theta=list(initial), theta=fitted.tolist(), updates=2000,
                update_unit='Full TRAIN16 mean COST1', optimizer='AdamW',
                learning_rate=recipe['head_lr'], weight_decay=recipe['weight_decay'],
                clip_norm=recipe['clip_norm'], elapsed_seconds=time.monotonic()-start,
                **evaluate(rows, fitted))


def bias_witness(rows):
    theta = np.asarray(rows[0]['decision']['theta'])

    def derivative(bias):
        th = theta.copy()
        th[2] = bias
        value = 0.
        for row in rows:
            x, w, t, idx = matrix(row)
            z = x @ th
            value += expit(z[idx[idx != t]].max())
            if t != w:
                value -= expit(-z[t])
        return value / 16

    optimum = brentq(derivative, -10, 10)
    fitted = theta.copy()
    fitted[2] = optimum
    witness = theta.copy()
    witness[2] = .01
    return dict(status='PREVIOUSLY_INSPECTED_POST_HOC_WITNESS_NO_NEW_BIAS_SCAN',
                fixed_witness_bias=.01, fixed_witness=evaluate(rows, witness),
                scalar_cost1_optimal_bias=float(optimum), scalar_optimum=evaluate(rows, fitted))


def intervention(native, control):
    rows = []
    for left, right in zip(native, control):
        assert left['query_id'] == right['query_id']
        _, w, t, _ = matrix(left)
        delta = np.asarray(left['L']) - np.asarray(right['L'])
        ld, rd = left['decision'], right['decision']
        rows.append(dict(query_id=left['query_id'], raw_correct=t == w,
                         L_delta_abs_max=float(abs(delta).max()), L_delta_abs_mean=float(abs(delta).mean()),
                         target_minus_raw_winner_L_delta=float(delta[t] - delta[w]),
                         cost1_native_minus_control=ld['cost1'] - rd['cost1'],
                         target_margin_native_minus_control=ld['target_vs_strongest_wrong_margin'] - rd['target_vs_strongest_wrong_margin']))
    return dict(rows=rows, L_delta_abs_max=max(r['L_delta_abs_max'] for r in rows),
                L_delta_abs_mean=float(np.mean([r['L_delta_abs_mean'] for r in rows])),
                positive_target_vs_winner_delta_on_raw_wrong=sum(r['target_minus_raw_winner_L_delta'] > 0 for r in rows if not r['raw_correct']),
                negative_target_vs_winner_delta_on_raw_wrong=sum(r['target_minus_raw_winner_L_delta'] < 0 for r in rows if not r['raw_correct']),
                mean_cost_native_minus_control=float(np.mean([r['cost1_native_minus_control'] for r in rows])),
                mean_margin_native_minus_control=float(np.mean([r['target_margin_native_minus_control'] for r in rows])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--extra-endpoint', action='append', default=[], metavar='NAME=DIRECTORY',
                        help='Optional sealed TRAIN16 endpoint; same fixed diagnostic recipe')
    args = parser.parse_args()
    warm = read(SOURCE / 'cpu/warmstart_INTERNAL3.json')
    authority = read(warm['authority']['path'])
    assert BINDINGS[str(Path(warm['authority']['path']).resolve())] == warm['authority']['sha256']
    train_ids = warm['fit_queries']
    assert len(train_ids) == len(set(train_ids)) == 16
    definitions = [('PRE_REAL128', 'PRE_REAL', 128, 'native'),
                   ('PRE_REAL128_CONSTANT', 'PRE_REAL', 128, 'constant'),
                   ('PRE_REAL128_SHUFFLED', 'PRE_REAL', 128, 'shuffled'),
                   ('PRE_CONSTANT128', 'PRE_CONSTANT', 128, 'native'),
                   ('PRE_REAL16', 'PRE_REAL', 16, 'native')]
    sets = {name: load_endpoint(SOURCE / arm / 'endpoints' / f'{step:04d}' / mode, train_ids)
            for name, arm, step, mode in definitions}
    zero = []
    for row in sets['PRE_REAL128']:
        q = row['query_id']
        parity = read(ROOT / 'results/rc_prellm_m_adapter_v2/encoder_cache' / q / 'parity.json')
        zero.append(dict(row, L=parity['fresh_L0'], decision=warm['predictions'][q]))
    sets['FRESH_ZERO_WARMSTART'] = zero
    for extra in args.extra_endpoint:
        name, directory = extra.split('=', 1)
        assert name not in sets
        sets[name] = load_endpoint(Path(directory).resolve(), train_ids)
    output = dict(status='TRAIN16_RETROSPECTIVE_READOUT_CAPACITY_DIAGNOSTIC_COMPLETE',
                  evidence_level='Opened TRAIN16 post hoc diagnosis; no held/probe evaluation',
                  no_generalization_claim=True, model_performance_claim=False,
                  probe_label_reads=0, encoder_forwards=0, roma_forwards=0,
                  input_features=['raw_standardized_delta', 'sym_L', 'bias'],
                  explicit_M_in_readout=False, candidates_per_query=128, raw_correct=8,
                  train_ids=train_ids,
                  prior_frozen_recipe=dict(internal_updates=128, internal_update_unit='One query full C128',
                                           internal_train_passes=8, common_warmstart_full_batch_updates=2000,
                                           external_full_batch_updates=2000,
                                           budgets_equal=False),
                  retrospective_recipe='Common V3 warm INTERNAL3 initial theta; fixed endpoint content; 2000 full TRAIN16 AdamW COST1 updates',
                  limitations=['LP is unconstrained strict-margin capacity, not a fitted model.',
                               'Convex optimum is numerical and unregularized, not a certified exact lower bound.',
                               'Bias 0.01 was inspected post hoc before this script; it is a witness, not a selected model.',
                               'CPU fits and controls are diagnostics; none establishes real-M specificity.'], sets={})
    for name, rows in sets.items():
        observed = evaluate(rows, rows[0]['decision']['theta'])
        assert abs(observed['mean_cost1'] - np.mean([r['decision']['cost1'] for r in rows])) < 1e-12
        convex = convex_readout(rows)
        cpu = cpu_readout(rows, warm['theta'], authority)
        cpu['cost1_above_numerical_convex_optimum'] = cpu['mean_cost1'] - convex['mean_cost1']
        output['sets'][name] = dict(observed=observed,
                                   observed_theta=rows[0]['decision']['theta'],
                                   pure_content_top1=sum(r['decision']['free_content_target_rank'] == 1 for r in rows),
                                   linear_capacity=linear_capacity(rows),
                                   convex_cost1_readout=convex, fixed_2000_step_cpu_readout=cpu)
        if name in ('PRE_REAL128', 'PRE_REAL128_CONSTANT', 'PRE_REAL128_SHUFFLED', 'PRE_CONSTANT128'):
            output['sets'][name]['bias_diagnostic'] = bias_witness(rows)
    output['interventions'] = {mode: intervention(sets['PRE_REAL128'], sets[key])
                               for mode, key in [('constant', 'PRE_REAL128_CONSTANT'),
                                                 ('shuffled', 'PRE_REAL128_SHUFFLED')]}
    script = Path(__file__).resolve()
    BINDINGS[str(script)] = hashlib.sha256(script.read_bytes()).hexdigest()
    output['input_bindings'] = [dict(path=p, sha256=s) for p, s in sorted(BINDINGS.items())]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), status=output['status'],
                          cpu={k: dict(correct=v['fixed_2000_step_cpu_readout']['correct'],
                                       cost1=v['fixed_2000_step_cpu_readout']['mean_cost1'],
                                       gap=v['fixed_2000_step_cpu_readout']['cost1_above_numerical_convex_optimum'])
                               for k, v in output['sets'].items()})))


if __name__ == '__main__':
    main()
