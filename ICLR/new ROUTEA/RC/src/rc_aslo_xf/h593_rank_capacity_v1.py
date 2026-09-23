"""Exact certificates for the linear rank capacity of frozen six-feature rows.

This is a target-aware *oracle diagnostic*, never a training procedure.  A
query supplies 127 challenger feature vectors and the target's challenger
index.  Bias cancels from target-versus-wrong rank constraints.  HOLD itself
is deliberately absent: the certificates concern challenger rank only.

Float64 linear programming proposes witnesses; rational arithmetic alone
decides the certified status.  An unresolved numerical solve is not evidence
of infeasibility.  No data loading, model fitting, or filesystem writes occur
in this module.  Its command-line entry point runs synthetic self-tests only.
"""
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog


DIMENSION = 6
CHALLENGERS = 127
SOLVER_TIME_LIMIT_SECONDS = 10.0


def _features(item):
    """Validate the public input schema without mutating the supplied item."""
    if not isinstance(item['query_id'], str) or not item['query_id']:
        raise ValueError('query_id must be a nonempty string')
    x = np.asarray(item['x'], dtype=np.float64)
    if x.shape != (CHALLENGERS, DIMENSION) or not np.isfinite(x).all():
        raise ValueError('x must be a finite 127 by 6 binary64 array')
    target = item['target_index']
    if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)):
        raise ValueError('target_index must be an integer')
    target = int(target)
    if not 0 <= target < CHALLENGERS:
        raise ValueError('target_index must be in [0, 127)')
    return x, target


def make_constraints(items):
    """Return exact target-minus-wrong rows and their query/candidate mapping.

    Conversion precedes subtraction: these are differences of exact cached
    binary64 inputs, not rationals made from already-rounded differences.
    The order is input query order, then ascending wrong challenger index.
    """
    rows, metadata = [], []
    seen = set()
    for item in items:
        x, target = _features(item)
        q = item['query_id']
        if q in seen:
            raise ValueError('duplicate query_id in shared constraint system')
        seen.add(q)
        t = [Fraction.from_float(float(v)) for v in x[target]]
        for j in range(CHALLENGERS):
            if j == target:
                continue
            rows.append([t[k] - Fraction.from_float(float(x[j, k]))
                         for k in range(DIMENSION)])
            metadata.append(dict(query_id=q, wrong_index=j))
    return rows, metadata


def rational_solve(matrix, rhs):
    """Solve an exact rational linear system; free coordinates default to zero.

    A sparse LP support normally has at most seven entries.  This routine
    nevertheless permits a rectangular/rank-deficient system and returns
    None for inconsistency.  The caller must check nonnegativity separately.
    """
    if len(matrix) != len(rhs) or not matrix:
        raise ValueError('nonempty matrix and matching right hand side required')
    n = len(matrix[0])
    if any(len(r) != n for r in matrix):
        raise ValueError('ragged rational matrix')
    a = [[Fraction(v) for v in r] + [Fraction(b)]
         for r, b in zip(matrix, rhs)]
    row, pivots = 0, []
    for c in range(n):
        pivot = next((i for i in range(row, len(a)) if a[i][c] != 0), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        scale = a[row][c]
        a[row] = [v / scale for v in a[row]]
        for i in range(len(a)):
            if i != row and a[i][c] != 0:
                factor = a[i][c]
                a[i] = [v - factor * u for v, u in zip(a[i], a[row])]
        pivots.append((row, c))
        row += 1
        if row == len(a):
            break
    if any(all(v == 0 for v in r[:n]) and r[n] != 0 for r in a):
        return None
    solution = [Fraction(0) for _ in range(n)]
    for i, c in pivots:
        solution[c] = a[i][n]
    return solution


def _exact_margin(rows, theta):
    values = [Fraction.from_float(float(v)) for v in theta]
    return min(sum((x * y for x, y in zip(row, values)), Fraction(0))
               for row in rows)


def _dual_certificate(rows, indices):
    """Reconstruct and fully verify a proposed sparse Farkas support."""
    if not indices:
        return None
    selected = [rows[i] for i in indices]
    matrix = [[row[k] for row in selected] for k in range(DIMENSION)]
    matrix.append([Fraction(1)] * len(indices))
    weights = rational_solve(matrix, [Fraction(0)] * DIMENSION + [Fraction(1)])
    if weights is None or any(w < 0 for w in weights):
        return None
    support = [(i, w) for i, w in zip(indices, weights) if w > 0]
    if not support or sum((w for _, w in support), Fraction(0)) != 1:
        return None
    if any(sum((w * rows[i][k] for i, w in support), Fraction(0)) != 0
           for k in range(DIMENSION)):
        return None
    return dict(status='EXACT_INFEASIBLE',
                support_indices=[i for i, _ in support],
                weights=[dict(num=str(w.numerator), den=str(w.denominator))
                         for _, w in support])


def _solver_info(result):
    return dict(status=int(result.status), message=str(result.message),
                success=bool(result.success))


def solve_constraints(exact_rows, metadata):
    """Certify strict linear separability or return UNRESOLVED.

    Primal: min ||theta||_1 subject to A theta >= 1, via theta+ - theta-.
    Dual: lambda >= 0, sum(lambda) = 1, and A.T lambda = 0.
    Each LP has its own 10-second solver time limit.  Neither solver status
    nor approximate feasibility is used as a scientific certificate.
    """
    if len(exact_rows) != len(metadata):
        raise ValueError('constraint rows and metadata lengths differ')
    if not exact_rows:
        return dict(status='UNRESOLVED', reason='empty constraint system',
                    model_fits=0, oracle_diagnostic=True)
    rows = []
    for row in exact_rows:
        if len(row) != DIMENSION or any(not isinstance(v, Fraction) for v in row):
            raise ValueError('exact_rows must contain six Fraction coordinates per row')
        rows.append(list(row))
    for meta in metadata:
        if not isinstance(meta.get('query_id'), str) or not isinstance(meta.get('wrong_index'), int):
            raise ValueError('invalid constraint metadata')
    info = dict(constraint_count=len(rows), parameter_count=DIMENSION,
                solver_time_limit_seconds=SOLVER_TIME_LIMIT_SECONDS,
                model_fits=0, oracle_diagnostic=True,
                scope='Target-aware strict challenger-rank feasibility; no HOLD or generalization claim')
    # A zero row immediately proves that strict domination is impossible.
    zero = next((i for i, row in enumerate(rows) if not any(row)), None)
    if zero is not None:
        certificate = _dual_certificate(rows, [zero])
        assert certificate is not None
        return dict(info, **certificate, exact_constraint_validation=True)
    try:
        a = np.asarray([[float(v) for v in row] for row in rows], dtype=np.float64)
    except (OverflowError, ValueError) as exc:
        return dict(info, status='UNRESOLVED', reason='float conversion failed: ' + str(exc))
    if not np.isfinite(a).all():
        return dict(info, status='UNRESOLVED', reason='nonfinite float proposal matrix')
    options = dict(presolve=True, time_limit=SOLVER_TIME_LIMIT_SECONDS,
                   primal_feasibility_tolerance=1e-9,
                   dual_feasibility_tolerance=1e-9)
    try:
        primal = linprog(np.ones(2 * DIMENSION),
                         A_ub=-np.column_stack((a, -a)), b_ub=-np.ones(len(rows)),
                         bounds=(0.0, None), method='highs', options=options)
    except Exception as exc:
        return dict(info, status='UNRESOLVED', reason='primal solver exception: ' + repr(exc))
    info['primal_solver'] = _solver_info(primal)
    # Even a time-limited solver may have a valid witness; exact checking
    # determines whether it is usable.  Numeric success can also fail here.
    if primal.x is not None and len(primal.x) == 2 * DIMENSION:
        theta = primal.x[:DIMENSION] - primal.x[DIMENSION:]
        if np.isfinite(theta).all():
            margin = _exact_margin(rows, theta)
            if margin > 0:
                return dict(info, status='STRICT_FEASIBLE',
                            theta_hex=[float(v).hex() for v in theta],
                            exact_min_margin=str(margin), exact_constraint_validation=True)
            info['rejected_primal_exact_min_margin'] = str(margin)
    try:
        dual = linprog(np.zeros(len(rows)),
                       A_eq=np.vstack((a.T, np.ones(len(rows)))),
                       b_eq=np.r_[np.zeros(DIMENSION), 1.0],
                       bounds=(0.0, None), method='highs', options=options)
    except Exception as exc:
        return dict(info, status='UNRESOLVED', reason='dual solver exception: ' + repr(exc))
    info['dual_solver'] = _solver_info(dual)
    if dual.x is not None and len(dual.x) == len(rows) and np.isfinite(dual.x).all():
        indices = np.flatnonzero(dual.x > 0.0).tolist()
        certificate = _dual_certificate(rows, indices)
        if certificate is not None:
            return dict(info, **certificate, exact_constraint_validation=True)
        info['uncertified_numeric_dual_support'] = indices
    return dict(info, status='UNRESOLVED',
                reason='neither an exact strictly positive margin nor exact nonnegative dual witness was obtained',
                exact_constraint_validation=False)


def query_certificate(item):
    """Return a single-query rank certificate using candidate index support."""
    x, target = _features(item)
    common = dict(query_id=item['query_id'], target_index=target,
                  model_fits=0, oracle_diagnostic=True)
    collision = next((i for i in range(CHALLENGERS)
                      if i != target and np.array_equal(x[i], x[target])), None)
    if collision is not None:
        return dict(common, status='EXACT_FEATURE_COLLISION', wrong_index=collision,
                    exact_constraint_validation=True)
    rows, meta = make_constraints([item])
    result = solve_constraints(rows, meta)
    answer = dict(result, **common)
    if result['status'] == 'STRICT_FEASIBLE':
        answer['status'] = 'LINEAR_SEPARABLE'
    elif result['status'] == 'EXACT_INFEASIBLE':
        answer['status'] = 'EXACT_CONVEX_HULL'
        answer['constraint_support_indices'] = list(result['support_indices'])
        answer['support_indices'] = [meta[i]['wrong_index'] for i in result['support_indices']]
    return answer


def _selftest():
    # A strict exposed target, surrounded target, and exact collision.
    exposed = np.zeros((CHALLENGERS, DIMENSION), dtype=np.float64)
    exposed[0, 0] = 1.0
    a = dict(query_id='synthetic-exposed', x=exposed, target_index=0)
    assert query_certificate(a)['status'] == 'LINEAR_SEPARABLE'
    inside = np.zeros_like(exposed)
    inside[1:, 0] = 1.0
    inside[2, 0] = -1.0
    b = dict(query_id='synthetic-hull', x=inside, target_index=0)
    hull = query_certificate(b)
    assert hull['status'] == 'EXACT_CONVEX_HULL'
    weights = [Fraction(int(v['num']), int(v['den'])) for v in hull['weights']]
    assert sum(weights) == 1
    for k in range(DIMENSION):
        assert sum((w * Fraction.from_float(float(inside[i, k]))
                    for i, w in zip(hull['support_indices'], weights)), Fraction(0)) == 0
    collision = exposed.copy()
    collision[1] = collision[0]
    assert query_certificate(dict(query_id='synthetic-collision', x=collision,
                                  target_index=0))['status'] == 'EXACT_FEATURE_COLLISION'
    # Each query is individually separable, but one shared theta cannot
    # simultaneously prefer +e1 and -e1 to zero.
    reversed_x = -exposed
    reverse = dict(query_id='synthetic-reverse', x=reversed_x, target_index=0)
    assert query_certificate(reverse)['status'] == 'LINEAR_SEPARABLE'
    rows, meta = make_constraints([a, reverse])
    shared = solve_constraints(rows, meta)
    assert shared['status'] == 'EXACT_INFEASIBLE'
    assert len(rows) == len(meta) == 252
    # Ensure subtraction really occurs in rationals, before rounding.
    large = np.zeros_like(exposed)
    large[0, 0], large[1:, 0] = 1.0, 2.0 ** -54
    exact, _ = make_constraints([dict(query_id='synthetic-exact', x=large, target_index=0)])
    assert exact[0][0] == Fraction(1) - Fraction(1, 2 ** 54)
    assert exact[0][0] != Fraction.from_float(float(large[0, 0] - large[1, 0]))
    print('H593_RANK_CAPACITY_SYNTHETIC_SELFTEST_PASS', flush=True)


if __name__ == '__main__':
    _selftest()
