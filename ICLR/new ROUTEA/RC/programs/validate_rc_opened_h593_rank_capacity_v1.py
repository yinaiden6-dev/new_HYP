#!/usr/bin/env python3
"""Pure-stdlib exact validation of per-query, six-feature ranking certificates.

The original binary64 feature endpoints and separating coefficients are read
as exact rationals. No optimizer, cached labels, model trainer or natural-data
loader is imported. A strict linear capacity certificate is not a learned
model, and a convex-hull certificate does not exclude target-winning ties.
"""
from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(value, target_index=None):
    _require(isinstance(value, int) and not isinstance(value, bool), 'candidate index must be an integer')
    _require(0 <= value < 127, 'candidate index outside 127 challengers')
    if target_index is not None:
        _require(value != target_index, 'target cannot be its own wrong candidate')
    return value


def _rational(value):
    _require(not isinstance(value, bool), 'boolean is not a feature endpoint')
    number = float(value)
    _require(math.isfinite(number), 'nonfinite binary64 endpoint')
    return Fraction.from_float(number)


def _fraction_payload(value):
    return dict(num=str(value.numerator), den=str(value.denominator))


def _weight(value):
    _require(isinstance(value, dict) and set(value) == {'num', 'den'}, 'invalid rational weight schema')
    _require(isinstance(value['num'], str) and isinstance(value['den'], str), 'rational fields must be strings')
    try:
        numerator, denominator = int(value['num']), int(value['den'])
    except ValueError as error:
        raise ValueError('rational fields must be integers') from error
    _require(denominator > 0, 'rational denominator must be positive')
    _require(numerator >= 0, 'convex weight must be nonnegative')
    return Fraction(numerator, denominator)


def verify_query(x, target_index, certificate):
    """Validate one certificate against exactly 127 original six-dimensional rows.

    Invalid certificates raise ValueError. UNRESOLVED returns proved=False;
    accepting its schema does not certify separability or inseparability.
    Equality treats +0 and -0 as the same real-valued feature coordinate.
    """
    _require(isinstance(x, (list, tuple)) and len(x) == 127, 'expected 127 challenger rows')
    rows = []
    for row in x:
        _require(isinstance(row, (list, tuple)) and len(row) == 6, 'expected six feature coordinates')
        rows.append([_rational(value) for value in row])
    target_index = _index(target_index)
    _require(isinstance(certificate, dict), 'certificate must be an object')
    status = certificate.get('status')
    target = rows[target_index]
    result = dict(status=status, target_index=target_index, challenger_count=127,
                  wrong_challenger_count=126, feature_dimension=6,
                  endpoint_interpretation='original binary64 endpoints treated as exact real rationals',
                  learned_model_or_accuracy_result=False)
    if status == 'LINEAR_SEPARABLE':
        encoded = certificate.get('theta_hex')
        _require(isinstance(encoded, (list, tuple)) and len(encoded) == 6, 'expected six separating coefficients')
        theta = []
        for value in encoded:
            _require(isinstance(value, str), 'separating coefficient must be a hex string')
            try:
                number = float.fromhex(value)
            except ValueError as error:
                raise ValueError('invalid separating coefficient hex') from error
            theta.append(_rational(number))
        margins = [sum(((target[j] - row[j]) * theta[j] for j in range(6)), Fraction(0))
                   for i, row in enumerate(rows) if i != target_index]
        minimum = min(margins)
        _require(minimum > 0, 'separating witness lacks an exact strictly positive margin against every wrong challenger')
        result.update(proved=True, exact_min_margin=_fraction_payload(minimum),
                      claim='a query-specific real linear score strictly ranks target above all 126 wrong challengers',
                      shared_head_or_generalization_proved=False)
        return result
    if status == 'EXACT_CONVEX_HULL':
        support, encoded = certificate.get('support_indices'), certificate.get('weights')
        _require(isinstance(support, (list, tuple)) and 1 <= len(support) <= 126, 'invalid convex support length')
        _require(isinstance(encoded, (list, tuple)) and len(encoded) == len(support), 'convex support/weight count mismatch')
        support = [_index(i, target_index) for i in support]
        _require(len(set(support)) == len(support), 'convex support indices must be distinct')
        weights = [_weight(value) for value in encoded]
        _require(sum(weights, Fraction(0)) == 1, 'convex weights must sum exactly to one')
        reconstructed = [sum((weight * rows[i][j] for i, weight in zip(support, weights)), Fraction(0))
                         for j in range(6)]
        _require(reconstructed == target, 'weighted wrong endpoints do not exactly reconstruct target')
        result.update(proved=True, support_size=len(support), positive_support_size=sum(w > 0 for w in weights),
                      exact_reconstruction=True,
                      claim='no real linear score can make target the unique highest-scoring challenger',
                      target_can_still_win_a_tie=True, nonlinear_information_loss_proved=False)
        return result
    if status == 'EXACT_FEATURE_COLLISION':
        wrong = _index(certificate.get('wrong_index'), target_index)
        _require(rows[wrong] == target, 'target and specified wrong six-vectors differ exactly')
        result.update(proved=True, wrong_index=wrong,
                      claim='a deterministic pointwise function of only these six real-valued coordinates must give this pair equal scores',
                      strict_distinction_impossible=True, target_can_still_win_a_tie=True,
                      candidate_context_or_additional_inputs_covered=False)
        return result
    if status == 'UNRESOLVED':
        result.update(proved=False, claim='no mathematical capacity conclusion supplied')
        return result
    raise ValueError('unknown certificate status')


def verify_system(items, certificate):
    """Verify shared linear ranking constraints in the declared item order.

    Each item contributes 126 rows x_target - x_wrong in increasing wrong
    candidate index. These are label-aware ranking-only oracle constraints;
    they contain neither a bias nor any HOLD/action requirement.
    """
    _require(isinstance(items, (list, tuple)) and len(items) > 0, 'expected a nonempty query system')
    exact_rows = []
    query_ids = []
    for item in items:
        _require(isinstance(item, dict), 'query system item must be an object')
        query = item.get('query_id')
        _require(isinstance(query, str) and bool(query), 'query_id must be a nonempty string')
        _require(query not in query_ids, 'duplicate query_id in shared system')
        query_ids.append(query)
        x = item.get('x')
        _require(isinstance(x, (list, tuple)) and len(x) == 127, 'expected 127 challenger rows')
        rows = []
        for row in x:
            _require(isinstance(row, (list, tuple)) and len(row) == 6, 'expected six feature coordinates')
            rows.append([_rational(value) for value in row])
        target = _index(item.get('target_index'))
        exact_rows.extend([[rows[target][j] - row[j] for j in range(6)]
                           for wrong, row in enumerate(rows) if wrong != target])
    _require(isinstance(certificate, dict), 'certificate must be an object')
    status = certificate.get('status')
    result = dict(status=status, query_count=len(items), constraint_count=len(exact_rows),
                  parameter_count=6, query_ids=query_ids,
                  ordering='item order, then increasing wrong index excluding target',
                  scope='shared real linear challenger ranking only; no bias or HOLD/action constraint',
                  learned_model_or_accuracy_result=False)
    if status == 'STRICT_FEASIBLE':
        encoded = certificate.get('theta_hex')
        _require(isinstance(encoded, (list, tuple)) and len(encoded) == 6, 'expected six separating coefficients')
        theta = []
        for value in encoded:
            _require(isinstance(value, str), 'separating coefficient must be a hex string')
            try:
                theta.append(_rational(float.fromhex(value)))
            except ValueError as error:
                raise ValueError('invalid separating coefficient') from error
        minimum = min(sum((a*b for a, b in zip(row, theta)), Fraction(0)) for row in exact_rows)
        _require(minimum > 0, 'shared witness lacks a positive exact margin on every constraint')
        result.update(proved=True, exact_min_margin=_fraction_payload(minimum),
                      claim='one shared real linear score strictly ranks each included target above every wrong challenger',
                      trainability_or_generalization_proved=False)
        return result
    if status == 'EXACT_INFEASIBLE':
        support, encoded = certificate.get('support_indices'), certificate.get('weights')
        _require(isinstance(support, (list, tuple)) and 1 <= len(support) <= len(exact_rows), 'invalid shared dual support length')
        _require(isinstance(encoded, (list, tuple)) and len(encoded) == len(support), 'shared dual support/weight mismatch')
        _require(all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(exact_rows)
                     for i in support), 'invalid shared constraint index')
        _require(len(set(support)) == len(support), 'shared dual support must not repeat a constraint')
        weights = [_weight(value) for value in encoded]
        _require(sum(weights, Fraction(0)) == 1, 'shared dual weights must sum exactly to one')
        zero = [sum((weight * exact_rows[i][j] for i, weight in zip(support, weights)), Fraction(0))
                for j in range(6)]
        _require(all(v == 0 for v in zero), 'shared dual weighted exact constraints do not cancel')
        result.update(proved=True, support_size=len(support), positive_support_size=sum(w > 0 for w in weights),
                      claim='no shared real linear score simultaneously gives every included target a strictly positive ranking margin',
                      ties_excluded_from_impossibility=True, pointwise_information_loss_proved=False)
        return result
    if status == 'UNRESOLVED':
        result.update(proved=False, claim='no shared-capacity conclusion supplied')
        return result
    raise ValueError('unknown shared certificate status')


def self_test():
    count = 0

    def accept(rows, certificate, expected, target=0):
        nonlocal count
        result = verify_query(rows, target, certificate)
        _require(result['status'] == expected, 'synthetic accepted status mismatch')
        count += 1
        return result

    def reject(rows, certificate, target=0):
        nonlocal count
        try:
            verify_query(rows, target, certificate)
        except ValueError:
            count += 1
            return
        raise AssertionError('invalid synthetic certificate was accepted')

    zero = [[0.0] * 6 for _ in range(127)]
    separated = [list(row) for row in zero]
    separated[0][0] = .1
    for row in separated[1:]:
        row[0] = -.2
    linear = dict(status='LINEAR_SEPARABLE', theta_hex=[1.0.hex()] + [0.0.hex()] * 5)
    answer = accept(separated, linear, 'LINEAR_SEPARABLE')
    expected = Fraction.from_float(.1) - Fraction.from_float(-.2)
    _require(answer['exact_min_margin'] == _fraction_payload(expected), 'endpoint subtraction was rounded before exact validation')
    _require(expected != Fraction.from_float(.1 - (-.2)), 'synthetic unrounded subtraction guard')
    reject(zero, linear)
    reject(separated, dict(linear, theta_hex=[(-1.0).hex()] + [0.0.hex()] * 5))
    hull_rows = [list(row) for row in zero]
    hull_rows[0][:2] = [.25, .75]
    hull_rows[1][0] = 1.0
    hull_rows[2][1] = 1.0
    hull = dict(status='EXACT_CONVEX_HULL', support_indices=[1, 2],
                weights=[dict(num='1', den='4'), dict(num='3', den='4')])
    accept(hull_rows, hull, 'EXACT_CONVEX_HULL')
    reject(hull_rows, dict(hull, support_indices=[1, 1]))
    reject(hull_rows, dict(hull, support_indices=[0, 2]))
    reject(hull_rows, dict(hull, weights=[dict(num='-1', den='4'), dict(num='5', den='4')]))
    reject(hull_rows, dict(hull, weights=[dict(num='1', den='4'), dict(num='2', den='4')]))
    shifted = [list(row) for row in hull_rows]
    shifted[0][0] = math.nextafter(.25, math.inf)
    reject(shifted, hull)
    collision = dict(status='EXACT_FEATURE_COLLISION', wrong_index=1)
    accept(zero, collision, 'EXACT_FEATURE_COLLISION')
    signed_zero = [list(row) for row in zero]
    signed_zero[1][0] = -0.0
    accept(signed_zero, collision, 'EXACT_FEATURE_COLLISION')
    reject(separated, collision)
    reject(zero, dict(collision, wrong_index=0))
    reject(zero, dict(collision, wrong_index=True))
    unknown = accept(zero, dict(status='UNRESOLVED', reason='synthetic'), 'UNRESOLVED')
    _require(unknown['proved'] is False, 'unresolved must not be a proof')
    broken = [list(row) for row in zero]
    broken[0][0] = math.nan
    reject(broken, dict(status='UNRESOLVED'))
    reject(zero[:126], collision)
    reject(zero, dict(status='UNKNOWN'))
    forward = [list(row) for row in zero]
    reverse = [list(row) for row in zero]
    forward[0][0] = 1.0
    reverse[0][0] = -1.0
    first = dict(query_id='first', x=forward, target_index=0)
    second = dict(query_id='second', x=reverse, target_index=0)
    feasible = dict(status='STRICT_FEASIBLE', theta_hex=[1.0.hex()] + [0.0.hex()] * 5)
    _require(verify_system([first], feasible)['proved'], 'shared one-query witness')
    count += 1
    dual = dict(status='EXACT_INFEASIBLE', support_indices=[0, 126],
                weights=[dict(num='1', den='2'), dict(num='1', den='2')])
    _require(verify_system([first, second], dual)['proved'], 'two-query opposite constraint certificate')
    count += 1
    for bad_items, bad_certificate in [([first, second], feasible),
                                        ([first, second], dict(dual, support_indices=[0, 1])),
                                        ([first, first], dual)]:
        try:
            verify_system(bad_items, bad_certificate)
        except ValueError:
            count += 1
        else:
            raise AssertionError('invalid shared synthetic certificate accepted')
    _require(verify_system([first, second], dict(status='UNRESOLVED'))['proved'] is False,
             'shared unresolved is not a proof')
    count += 1
    return dict(status='H593_RANK_CAPACITY_INDEPENDENT_SYNTHETIC_PASS', checks=count,
                natural_data_reads=0, optimizer_calls=0, stdlib_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), sort_keys=True), flush=True)
