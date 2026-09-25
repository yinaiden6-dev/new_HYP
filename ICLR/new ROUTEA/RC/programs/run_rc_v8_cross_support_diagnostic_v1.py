#!/usr/bin/env python3
"""Frozen V8 support cross-comparison; diagnostic only, no ranking or fitting."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'rc_v8_cross_support_diagnostic_v1'
AUTH = ROOT / ('registry/' + PREFIX + '_authority_20260908.json')
PLAN = ROOT / 'plan/RC_V8_CROSS_SUPPORT_DIAGNOSTIC_V1_20260908.md'
LAUNCH = ROOT / ('slurm/' + PREFIX + '_dev_cpuonly_59m.sbatch')
OUT = ROOT / ('results/' + PREFIX)
PARENT = ROOT / 'registry/rc_coherent_residual_development_authority_v8_20260908.json'
PARENT_OUT = ROOT / 'results/rc_coherent_residual_development_v8'
PINS = {
    'parent': '4a6d24014f7e2e8b13b3de9b41faf76e1d1ad7a8c625ee47db442f6755f34380',
    'core': '19806530ba94c4fbdfac050cf437326e9790b446633f2b5d4b001dbb371d6fd2',
    'runner': '2e2bc48c07dd866aaf836a6c364f6d9343b5718b00954d2c7ce1e7f065918836',
    'adapter': 'e933814472a3bca65afa9cead70242a93ad816668d12e7a5d8a124b914ee3d99',
    'checkpoint': '8ccf6b9815025ffe494ed9d9db1a2c504ea14d267496d59f9707c847b22283ef',
    'parent_result': '09374961f1c4d62f381bc0e519e73851c04ad78dec346fa1dd0314471c7f734e',
    'parent_validation': 'b83ebcc18f76eaf6327ec5aef6b6206574d06accc6e86cd54dafad65521f6f21',
}
STOP = False


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def logical(value):
    return hashlib.sha256(encode(value)).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {'path': str(Path(path).relative_to(ROOT)), 'sha256': sha(path)}


def checked(value):
    path = ROOT / value['path']
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT), 'UNSAFE_SOURCE')
    need(not any(x in str(path).lower() for x in ('d1_mi', 'd1-mi', 'grozi', 'gisc_prerecall_universe')), 'PROTECTED_SOURCE')
    need(sha(path) == value['sha256'], 'SOURCE_HASH_DRIFT:' + value['path'])
    return path


def module(value, name):
    spec = importlib.util.spec_from_file_location(name, checked(value))
    need(spec is not None and spec.loader is not None, 'IMPORT_FAILED')
    item = importlib.util.module_from_spec(spec)
    sys.modules[name] = item
    spec.loader.exec_module(item)
    return item


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encode(value) + b'\n'
    if path.exists():
        need(path.read_bytes() == data, 'IMMUTABLE_REPLAY_DRIFT:' + str(path))
        return
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def hx(value):
    value = float(value)
    need(math.isfinite(value), 'NONFINITE')
    return value.hex()


def number(value):
    result = float.fromhex(value)
    need(math.isfinite(result) and result.hex() == value, 'NONCANONICAL_BINARY64')
    return result


def rational(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def prepare_cross_axis(sources, valid):
    """Cache candidate-local source lookup once; never changes residual arithmetic."""
    import torch
    need(sources.dtype == torch.long and valid.dtype == torch.bool and sources.shape == valid.shape, 'SOURCE_AXIS_DOMAIN')
    src = sources.tolist()
    need(len(set(src)) == len(src) and min(src, default=0) >= 0, 'SOURCE_QUERY_AXIS_NOT_UNIQUE')
    observed = valid.tolist()
    return {source: index for index, source in enumerate(src) if observed[index]}


def cross_score(core, residuals, refs, sources, valid, support_sources, weights, prepared_axis=None):
    """Same ordered source-query domain; one missing atom makes the cell UNKNOWN."""
    import torch
    need(sources.dtype == refs.dtype == torch.long and valid.dtype == torch.bool, 'ATOM_AXIS_DTYPE')
    need(sources.shape == refs.shape == valid.shape == residuals.shape[:1], 'ATOM_AXIS_SHAPE')
    need(support_sources and len(set(support_sources)) == len(support_sources), 'EMPTY_OR_DUPLICATE_SUPPORT')
    lookup = prepare_cross_axis(sources, valid) if prepared_axis is None else prepared_axis
    missing = [source for source in support_sources if source not in lookup]
    if missing:
        return {'score_binary64': None, 'energy_binary64': None, 'missing_count': len(missing), 'witness': None,
                'reference_group_count': None, 'mean_term_binary64': None, 'max_term_binary64': None}
    ids = torch.tensor([lookup[source] for source in support_sources], dtype=torch.long)
    need(bool((refs[ids] >= 0).all()), 'VALID_ASSIGNMENT_NEGATIVE_REFERENCE')
    score, witness = core.coherent_score(residuals[ids], refs[ids], weights, torch.tensor(support_sources, dtype=torch.long))
    mean_term = torch.tensor([float.fromhex(x) for x in witness['group_mean_cost_binary64']], dtype=torch.float64).mean()
    return {'score_binary64': hx(score), 'energy_binary64': witness['coherent_energy_binary64'], 'missing_count': 0, 'witness': witness,
            'reference_group_count': len(witness['reference_group_indices']), 'mean_term_binary64': hx(mean_term),
            'max_term_binary64': witness['global_max_cost_binary64']}


def pair_summary(query, targets):
    keys = query['candidate_keys']
    target_set = set(targets)
    need(target_set and target_set.issubset(keys) and len(target_set) < len(keys), 'TARGET_AXIS_INVALID')
    scores = [number(x) for x in query['original_scores_binary64']]
    ti = max((i for i, key in enumerate(keys) if key in target_set), key=lambda i: scores[i])
    wi = max((i for i, key in enumerate(keys) if key not in target_set), key=lambda i: scores[i])
    tr, wr = query['owners'][ti], query['owners'][wi]
    cells = {'KT_Ht': tr['scores_binary64'][ti], 'KW_Hw': wr['scores_binary64'][wi],
             'KW_Ht': tr['scores_binary64'][wi], 'KT_Hw': wr['scores_binary64'][ti]}
    energies = {'ET_Ht': tr['energies_binary64'][ti], 'EW_Hw': wr['energies_binary64'][wi],
                'EW_Ht': tr['energies_binary64'][wi], 'ET_Hw': wr['energies_binary64'][ti]}
    missing = {'KT_Ht': tr['missing_counts'][ti], 'KW_Hw': wr['missing_counts'][wi],
               'KW_Ht': tr['missing_counts'][wi], 'KT_Hw': wr['missing_counts'][ti]}
    original = Fraction.from_float(scores[ti]) - Fraction.from_float(scores[wi])
    complete = all(value is not None for value in cells.values())
    terms, float_terms, category = None, None, 'UNKNOWN_OR_H0'
    if complete:
        f = {key: Fraction.from_float(number(value)) for key, value in cells.items()}
        m = f['KT_Ht'] - f['KW_Hw']
        a, b = f['KT_Ht'] - f['KW_Ht'], f['KT_Hw'] - f['KW_Hw']
        sw, st = f['KW_Hw'] - f['KW_Ht'], f['KT_Ht'] - f['KT_Hw']
        need(m == original and m == a - sw == b + st, 'EXACT_ACCOUNTING_FAILED')
        terms = {key: rational(value) for key, value in {'m': m, 'a': a, 'b': b, 'sw': sw, 'st': st}.items()}
        float_terms = {key: hx(value) for key, value in {'m': m, 'a': a, 'b': b, 'sw': sw, 'st': st}.items()}
        if original > 0:
            category = 'ORIGINAL_SUCCESS_COMPLETE'
        elif a > 0 and b > 0:
            category = 'DIAGONAL_DOMAIN_REVERSAL'
        elif a <= 0 and b <= 0:
            category = 'FROZEN_READOUT_NONPOSITIVE_ON_BOTH_SUPPORTS'
        else:
            category = 'SUPPORT_DEPENDENT_MIXED'
    blockers, unknown = [], []
    fixed_status = 'UNAVAILABLE_TARGET_H0'
    if cells['KT_Ht'] is not None:
        target_score = number(cells['KT_Ht'])
        for i, key in enumerate(keys):
            if key in target_set:
                continue
            value = tr['scores_binary64'][i]
            if value is None:
                unknown.append(key)
            elif number(value) >= target_score:
                blockers.append(key)
        fixed_status = ('REFUTED_BY_KNOWN_BLOCKER' if blockers else 'UNDETERMINED_WITH_UNKNOWN'
                        if unknown else 'CONDITIONAL_DOMINANCE_ON_FIXED_TARGET_SUPPORT')
    return {'query_resource_key': query['query_resource_key'], 'fixed_target_candidate_resource_key': keys[ti],
            'fixed_wrong_candidate_resource_key': keys[wi], 'target_support_sha256': tr['source_H_sha256'],
            'wrong_support_sha256': wr['source_H_sha256'], 'target_atom_count': tr['atom_count'],
            'wrong_atom_count': wr['atom_count'], 'target_H0': tr['support_state'] == 'H0',
            'wrong_H0': wr['support_state'] == 'H0', 'original_margin_binary64': hx(scores[ti] - scores[wi]),
            'original_margin_exact': rational(original), 'scores_binary64': cells, 'energies_binary64': energies,
            'missing_counts': missing, 'complete_four_cell_comparison': complete, 'exact_terms': terms, 'terms_binary64': float_terms,
            'accounting_identity': 'm = a - sw = b + st', 'category': category,
            'fixed_target_support_full_C128': {'status': fixed_status, 'known_blocker_keys': blockers,
                'unknown_wrong_candidate_keys': unknown, 'known_blocker_count': len(blockers),
                'unknown_wrong_candidate_count': len(unknown),
                'comparison_wrong_candidate_count': len(keys) - len(target_set)},
            'interpretation': 'CROSS_SUPPORTS_NEED_NOT_BE_LEGAL_CANDIDATE_PROPOSALS_NO_RERANKING_OR_RESCUE_CLAIM'}


def synthetic_e0():
    import torch
    torch.set_num_threads(1)
    core = module({'path': 'src/rc_aslo_xf/reference_conditioned_coherent_residual_p_only_v8.py', 'sha256': PINS['core']}, 'rc_v8_cross_support_e0_core')
    e = torch.zeros((4, 128), dtype=torch.float64)
    e[0, 0] = 2; e[1, 0] = 2; e[2, 1] = 1; e[3, 1] = 1
    refs, sources = torch.tensor([0, 0, 1, 1]), torch.tensor([8, 2, 7, 3])
    valid, weights = torch.ones(4, dtype=torch.bool), torch.ones(256, dtype=torch.float64)
    original = cross_score(core, e, refs, sources, valid, [8, 2, 7, 3], weights)
    assert cross_score(core, e, refs, sources, valid, [8, 2, 7, 3], weights, prepare_cross_axis(sources, valid)) == original
    assert original['witness']['group_mean_witness_source_query_indices'] == [2, 3]
    assert original['witness']['global_max_witness_source_query_index'] == 2
    perm = torch.tensor([3, 1, 0, 2])
    assert cross_score(core, e[perm], refs[perm], sources[perm], valid[perm], [8, 2, 7, 3], weights) == original
    invalid = valid.clone(); invalid[1] = False
    missing = cross_score(core, e, refs, sources, invalid, [8, 2], weights)
    assert missing['score_binary64'] is None and missing['energy_binary64'] is None and missing['missing_count'] == 1
    assert cross_score(core, e, refs, sources, valid, [8, 999], weights)['missing_count'] == 1
    # Test original pair stays fixed even if a third candidate scores highest on Ht.
    def owner(values):
        return {'scores_binary64': [None if x is None else hx(x) for x in values],
                'energies_binary64': [None if x is None else hx(1) for x in values],
                'missing_counts': [1 if x is None else 0 for x in values], 'source_H_sha256': '0' * 64,
                'support_state': 'H1', 'atom_count': 4}
    q = {'query_resource_key': 'q', 'candidate_keys': ['c0', 'c1', 'c2'], 'original_scores_binary64': [hx(.5), hx(.75), hx(.25)],
         'owners': [owner([.5, .25, .625]), owner([.875, .75, .5]), owner([.5, .5, .25])]}
    pair = pair_summary(q, ['c0'])
    assert pair['category'] == 'DIAGONAL_DOMAIN_REVERSAL' and pair['fixed_wrong_candidate_resource_key'] == 'c1'
    assert pair['fixed_target_support_full_C128']['known_blocker_keys'] == ['c2']
    q['owners'][0] = owner([.5, .25, None])
    assert pair_summary(q, ['c0'])['fixed_target_support_full_C128']['status'] == 'UNDETERMINED_WITH_UNKNOWN'
    q['owners'][0] = owner([.5, .25, .375])
    assert pair_summary(q, ['c0'])['fixed_target_support_full_C128']['status'] == 'CONDITIONAL_DOMINANCE_ON_FIXED_TARGET_SUPPORT'
    q['owners'][0] = owner([.5, None, .375])
    assert pair_summary(q, ['c0'])['category'] == 'UNKNOWN_OR_H0'
    q['owners'][0] = owner([None, None, None]); q['owners'][0]['support_state'] = 'H0'
    assert pair_summary(q, ['c0'])['fixed_target_support_full_C128']['status'] == 'UNAVAILABLE_TARGET_H0'
    verify_exact_terms(pair)
    corrupted = json.loads(encode(pair)); corrupted['exact_terms']['a']['numerator'] = '999'
    try:
        verify_exact_terms(corrupted)
    except RuntimeError:
        pass
    else:
        raise AssertionError('tampered_exact_accounting_accepted')
    with tempfile.TemporaryDirectory(prefix='.rc-v8-cross-e0-', dir=ROOT / 'results') as temporary:
        artifact = Path(temporary) / 'artifact.json'
        atomic(artifact, {'value': 1}); source = binding(artifact)
        atomic(artifact, {'value': 1})
        try:
            atomic(artifact, {'value': 2})
        except RuntimeError:
            pass
        else:
            raise AssertionError('nonidentical_artifact_overwrite_accepted')
        artifact.chmod(0o600); artifact.write_bytes(encode({'value': 2}) + b'\n')
        try:
            checked(source)
        except RuntimeError:
            pass
        else:
            raise AssertionError('tampered_artifact_hash_accepted')
    return {'status': 'RC_V8_CROSS_SUPPORT_SYNTHETIC_E0_PASS', 'checks': {
        'physical_witness_ties_use_smallest_source_query': True, 'candidate_row_permutation_preserves_source_domain': True,
        'cached_source_lookup_preserves_complete_score_and_witness': True,
        'single_invalid_assignment_makes_entire_cross_score_unknown': True, 'missing_source_never_intersected_or_imputed': True,
        'exact_binary64_rational_dual_decomposition': True, 'original_wrong_never_reselected': True,
        'known_blocker_refutes_conditional_dominance': True, 'unknown_prevents_full_C128_winning_claim': True,
        'fully_observed_fixed_support_is_conditional_only': True, 'H0_explicitly_unavailable': True,
        'same_observation_diagonal_reversal_constructed': True, 'artifact_hash_and_append_only_tamper_rejected': True,
        'independent_exact_arithmetic_tamper_rejected': True},
        'natural_value_reads': 0, 'scientific_GO_or_NO_GO': None}


def freeze():
    need(not AUTH.exists(), 'AUTHORITY_EXISTS')
    need(sha(PARENT) == PINS['parent'], 'PARENT_AUTHORITY_DRIFT')
    parent = read(PARENT)
    need(parent['status'] == 'RC_COHERENT_RESIDUAL_V8_DEVELOPMENT_AUTHORITY_READY', 'PARENT_NOT_FROZEN')
    files = {'program': Path(__file__), 'plan': PLAN, 'launcher': LAUNCH, 'parent': PARENT,
             'checkpoint': PARENT_OUT / 'final_checkpoint.json', 'saved_scores': PARENT_OUT / 'score_records.jsonl',
             'parent_result': PARENT_OUT / 'result.json', 'parent_validation': ROOT / parent['validation_rel'] / 'result.json'}
    for name in ('core', 'runner', 'adapter'):
        files[name] = checked(parent[name])
    for name, pin in PINS.items():
        need(sha(files[name]) == pin, 'ORIGINAL_V8_PIN_DRIFT:' + name)
    e0 = synthetic_e0()
    e0_path = ROOT / ('results/' + PREFIX + '_e0/result.json')
    atomic(e0_path, e0)
    files['e0'] = e0_path
    a = {'status': 'RC_V8_CROSS_SUPPORT_DIAGNOSTIC_AUTHORIZED', 'sources': {key: binding(path) for key, path in files.items()},
         'postjoin': parent['postjoin'], 'input_manifest_sha256': parent['input_manifest_sha256'],
         'parent_contract_sha256': parent['contract']['sha256'], 'order_sha256': parent['training']['order_sha256'],
         'query_count': 32, 'candidate_count': 128, 'expected_nonempty_owner_support_count': 1672,
         'output_rel': str(OUT.relative_to(ROOT)), 'cross_supports_required_to_be_candidate_legal': False,
         'missing_policy': 'ANY_MISSING_ASSIGNMENT_IS_UNKNOWN_NO_INTERSECTION_OR_IMPUTATION',
         'candidate_or_support_reselection_authorized': False, 'training_authorized': False,
         'formal_panel_authorized': False, 'automatic_stage_advance': False, 'scientific_GO_or_NO_GO': None}
    atomic(AUTH, a)
    print(json.dumps({'status': a['status'], 'authority_sha256': sha(AUTH)}), flush=True)


def load_authority():
    a = read(AUTH)
    need(a['status'] == 'RC_V8_CROSS_SUPPORT_DIAGNOSTIC_AUTHORIZED', 'AUTHORITY_INVALID')
    need(a['output_rel'] == str(OUT.relative_to(ROOT)), 'OUTPUT_DRIFT')
    for name, pin in PINS.items():
        need(a['sources'][name]['sha256'] == pin, 'FROZEN_PARENT_PIN_DRIFT')
    for value in a['sources'].values():
        checked(value)
    parent = read(checked(a['sources']['parent']))
    for name in ('core', 'runner', 'adapter'):
        need(a['sources'][name] == parent[name], 'PARENT_SOURCE_BINDING_DRIFT')
    need(a['postjoin'] == parent['postjoin'] and a['input_manifest_sha256'] == parent['input_manifest_sha256'], 'PARENT_SCOPE_DRIFT')
    return a


def run():
    global STOP
    began = time.monotonic()
    a, ash = load_authority(), sha(AUTH)
    import torch
    torch.set_num_threads(1)
    adapter = module(a['sources']['adapter'], 'rc_v8_cross_support_adapter')
    core = module(a['sources']['core'], 'rc_v8_cross_support_core')
    runner = module(a['sources']['runner'], 'rc_v8_cross_support_checkpoint_reader')
    bundle = adapter.InputBundle(ROOT, expected_manifest_sha256=a['input_manifest_sha256'])
    cp = read(checked(a['sources']['checkpoint']))
    need(cp['schema'] == runner.SCHEMA and cp['update_index'] == 512 and cp['contract_sha256'] == a['parent_contract_sha256']
         and cp['order_sha256'] == a['order_sha256'], 'FINAL_CHECKPOINT_INVALID')
    heads = core.GroupedResidualHeads()
    heads.load_state_dict(runner.unpack(cp['model'], torch), strict=True)
    heads.eval()
    weights = heads.real.weights().detach()
    saved_list = [json.loads(line) for line in checked(a['sources']['saved_scores']).read_text().splitlines()]
    saved = {row['query_resource_key']: row for row in saved_list}
    need(len(saved_list) == len(saved) == 32, 'SAVED_QUERY_AXIS')
    records = []
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    with torch.no_grad():
        for ep in bundle.iter_queries():
            if STOP or time.monotonic() - began > 3000:
                print(json.dumps({'status': 'INCOMPLETE_TIME_LIMIT_OR_SIGNAL', 'sealed_queries': len(records),
                                  'automatic_resume_authorized': False}), flush=True)
                return 75
            key, keys = ep['query_resource_key'], list(ep['candidate_keys'])
            old = saved[key]
            need(old['candidate_keys'] == keys and keys == sorted(set(keys)) and len(keys) == 128, 'SAVED_C128_AXIS_DRIFT')
            banks = ep['banks_by_control']['REAL']
            need(all(bank.reference_resource_key == keys[i] for i, bank in enumerate(banks)), 'REAL_REFERENCE_AXIS_DRIFT')
            prepared_axes = [prepare_cross_axis(bank.source_query_indices, ep['valid']['REAL'][i]) for i, bank in enumerate(banks)]
            owners = []
            for ci, candidate in enumerate(keys):
                decision = old['arms']['REAL']['decisions'][ci]
                score = old['arms']['REAL']['scores_binary64'][ci]
                need(decision['candidate_resource_key'] == candidate and decision['query_resource_key'] == key
                     and decision['arm'] == 'REAL' and decision['control'] == 'REAL'
                     and decision['candidate_score_binary64'] == score, 'SAVED_DECISION_AXIS')
                support = decision['selected_H']
                families = ep['families']['REAL'][ci]['components']
                row = {'owner_candidate_resource_key': candidate, 'source_H_sha256': logical(support),
                       'support_state': decision['state'], 'source_query_indices': [], 'atom_count': 0,
                       'saved_map_seed_ordinal': decision['map_seed_ordinal'], 'diagonal_witness': None,
                       'scores_binary64': [None] * 128, 'energies_binary64': [None] * 128,
                       'missing_counts': [None] * 128, 'reference_group_counts': [None] * 128,
                       'mean_terms_binary64': [None] * 128, 'max_terms_binary64': [None] * 128,
                       'diagonal_score_replayed': True,
                       'diagonal_physical_witness_replayed': support is not None}
                if support is None:
                    need(not families and decision['state'] == 'H0' and decision['map_seed_ordinal'] is None and number(score) == 0, 'H0_DRIFT')
                    row['unavailable_reason'] = 'OWNER_STRUCTURAL_H0_NO_OBSERVATION_DOMAIN'
                else:
                    ordinal = decision['map_seed_ordinal']
                    need(decision['state'] == 'H1' and type(ordinal) is int and 0 <= ordinal < len(families), 'H1_FAMILY_AXIS')
                    expected = core._support(families[ordinal])
                    need({k: v for k, v in support.items() if k != 'pooling_witness'} == expected, 'SAVED_SUPPORT_GENERATION_DRIFT')
                    domain = support['source_query_indices']
                    need([prepared_axes[ci].get(source) for source in domain] == support['atom_indices'], 'DIAGONAL_SOURCE_TO_ATOM_SUPPORT_DRIFT')
                    need(banks[ci].source_reference_indices[torch.tensor(support['atom_indices'], dtype=torch.long)].tolist()
                         == support['source_reference_indices'], 'DIAGONAL_SOURCE_REFERENCE_SUPPORT_DRIFT')
                    row.update(source_query_indices=domain, atom_count=len(domain))
                    for sj in range(128):
                        bank = banks[sj]
                        cross = cross_score(core, ep['residual_squared']['REAL'][sj], bank.source_reference_indices,
                                            bank.source_query_indices, ep['valid']['REAL'][sj], domain, weights, prepared_axes[sj])
                        row['scores_binary64'][sj] = cross['score_binary64']
                        row['energies_binary64'][sj] = cross['energy_binary64']
                        row['missing_counts'][sj] = cross['missing_count']
                        row['reference_group_counts'][sj] = cross['reference_group_count']
                        row['mean_terms_binary64'][sj] = cross['mean_term_binary64']
                        row['max_terms_binary64'][sj] = cross['max_term_binary64']
                        if sj == ci:
                            need(cross['score_binary64'] == score and cross['witness'] == support['pooling_witness'], 'DIAGONAL_SCORE_OR_WITNESS_REPLAY_DRIFT')
                            row['diagonal_witness'] = cross['witness']
                owners.append(row)
            query = {'status': 'RC_V8_CROSS_SUPPORT_QUERY_PREJOIN_SEALED', 'query_resource_key': key, 'candidate_keys': keys,
                     'authority_sha256': ash, 'source_receipt_sha256': logical(ep['source_receipt']),
                     'weight_sha256': core.tensor_sha(weights), 'saved_query_sha256': logical(old),
                     'original_scores_binary64': old['arms']['REAL']['scores_binary64'], 'owners': owners,
                     'candidate_or_support_reselection_count': 0, 'target_reads': 0,
                     'cross_support_candidate_legality_asserted': False}
            dest = OUT / 'prejoin' / (key + '.json')
            atomic(dest, query)
            records.append({'query_resource_key': key, 'path': str(dest.relative_to(OUT)), 'sha256': sha(dest)})
            print(json.dumps({'event': 'CROSS_SUPPORT_QUERY_SEALED', 'query_count': len(records), 'query': key}), flush=True)
    closure = bundle.prejoin_receipt()
    need(len(records) == 32 and {row['query_resource_key'] for row in records} == set(saved), 'FULL32_PREJOIN_NOT_CLOSED')
    atomic(OUT / 'input_closure.json', closure)
    seal = {'status': 'RC_V8_CROSS_SUPPORT_FULL32_C128_PREJOIN_CLOSED', 'authority_sha256': ash,
            'records': sorted(records, key=lambda x: x['query_resource_key']), 'input_closure_sha256': sha(OUT / 'input_closure.json'),
            'label_read_attempts_before_closure': bundle.barrier.blocked_read_attempts}
    atomic(OUT / 'prejoin_seal.json', seal)
    result = summarize(a, seal, saved_list)
    atomic(OUT / 'result.json', result)
    print(json.dumps({'status': result['status'], 'output': str(OUT), 'scientific_GO_or_NO_GO': None}), flush=True)
    return 0


def summarize(a, seal, saved_list):
    joined = read(checked(a['postjoin']['authority']))
    validation = read(checked(a['postjoin']['validation']))
    need(joined['record_count'] == 32 and joined['target_insertion_count'] == 0 and joined['fold_id'] == 1, 'POSTJOIN_SCOPE')
    need(validation['authority_sha256'] == a['postjoin']['authority']['sha256'] and validation['checks']
         and all(value is True for value in validation['checks'].values()), 'POSTJOIN_VALIDATION')
    roles = {row['query_resource_key']: row for row in joined['records']}
    need(len(roles) == 32 and set(roles) == {row['query_resource_key'] for row in seal['records']}, 'ROLE_QUERY_AXIS')
    pairs, counts = [], Counter()
    for item in seal['records']:
        path = OUT / item['path']
        need(sha(path) == item['sha256'], 'PREJOIN_ARTIFACT_DRIFT')
        query = read(path)
        role = roles[item['query_resource_key']]
        pairs.append({**pair_summary(query, role['target_candidate_resource_keys']), 'supergroup_hash': role['supergroup_hash']})
        for row in query['owners']:
            counts['owner_count'] += 1
            counts['nonempty_owner_count' if row['support_state'] == 'H1' else 'H0_owner_count'] += 1
            if row['support_state'] == 'H1':
                counts['known_cross_scores'] += sum(value is not None for value in row['scores_binary64'])
                counts['unknown_cross_scores'] += sum(value is None for value in row['scores_binary64'])
    need(counts['owner_count'] == 4096 and counts['nonempty_owner_count'] == a['expected_nonempty_owner_support_count'], 'OWNER_POPULATION_DRIFT')
    parent = read(checked(a['sources']['parent_result']))
    pv = read(checked(a['sources']['parent_validation']))
    need(parent['score_records_sha256'] == logical(saved_list) and parent['final_checkpoint_sha256'] == PINS['checkpoint'], 'PARENT_SCORE_BINDING')
    need(pv['status'] == 'RC_COHERENT_RESIDUAL_V8_DEVELOPMENT_INDEPENDENT_VALIDATION_PASS' and pv['result_sha256'] == PINS['parent_result']
         and pv['checkpoint_sha256'] == PINS['checkpoint'] and all(value is True for value in pv['checks'].values()), 'PARENT_NOT_INDEPENDENTLY_VALIDATED')
    categories = Counter(pair['category'] for pair in pairs)
    failures = [pair for pair in pairs if number(pair['original_margin_binary64']) <= 0]
    return {'status': 'RC_V8_CROSS_SUPPORT_DIAGNOSTIC_COMPLETE', 'claim_level': 'POSTHOC_FIXED_SUPPORT_CROSS_COMPARISON_ONLY',
            'authority_sha256': sha(AUTH), 'prejoin_seal_sha256': sha(OUT / 'prejoin_seal.json'), 'query_count': 32, 'candidate_count': 128,
            'counts': dict(counts), 'category_counts': {key: categories[key] for key in ('ORIGINAL_SUCCESS_COMPLETE',
                'DIAGONAL_DOMAIN_REVERSAL', 'FROZEN_READOUT_NONPOSITIVE_ON_BOTH_SUPPORTS', 'SUPPORT_DEPENDENT_MIXED', 'UNKNOWN_OR_H0')},
            'denominators': {'all_queries': 32, 'original_failures': len(failures),
                'all_queries_with_complete_four_cells': sum(pair['complete_four_cell_comparison'] for pair in pairs),
                'original_failures_with_complete_four_cells': sum(pair['complete_four_cell_comparison'] for pair in failures)},
            'records': pairs, 'original_v8_gate_go': parent['gate_go'], 'training_updates': 0,
            'candidate_or_support_reselection_count': 0, 'formal_panel_consumed': False,
            'automatic_stage_advance': False, 'scientific_GO_or_NO_GO': None,
            'limitations': ['Cross-support scores need not be legal candidate proposals.',
                'The query observation cells are shared, but reference-group measures can differ across candidates.',
                'sw and st can be negative; neither is claimed to be nonnegative MAP regret.',
                'Full C128 fixed-support dominance is conditional and is not a model ranking or rescue.',
                'Missing assignments remain UNKNOWN; H0 has no cross-comparison observation domain.']}


def verify_exact_terms(pair):
    """Independent algebraic check directly from saved four cells, not summary code."""
    if not pair['complete_four_cell_comparison']:
        need(pair['exact_terms'] is None and pair['terms_binary64'] is None, 'UNAVAILABLE_PAIR_HAS_EXACT_TERMS')
        return
    cells = {key: Fraction.from_float(number(value)) for key, value in pair['scores_binary64'].items()}
    terms = {key: Fraction(int(value['numerator']), int(value['denominator'])) for key, value in pair['exact_terms'].items()}
    expected = {'m': cells['KT_Ht'] - cells['KW_Hw'], 'a': cells['KT_Ht'] - cells['KW_Ht'],
                'b': cells['KT_Hw'] - cells['KW_Hw'], 'sw': cells['KW_Hw'] - cells['KW_Ht'],
                'st': cells['KT_Ht'] - cells['KT_Hw']}
    need(terms == expected and terms['a'] - terms['sw'] == terms['b'] + terms['st'] == terms['m'], 'INDEPENDENT_EXACT_TERMS_INVALID')
    need(pair['terms_binary64'] == {key: hx(value) for key, value in expected.items()}, 'INDEPENDENT_FLOAT_TERMS_INVALID')


def validate():
    """Separate process: hashes, source decisions and saved arithmetic; no raw replay."""
    a = load_authority()
    import torch
    torch.set_num_threads(1)
    core = module(a['sources']['core'], 'rc_v8_cross_support_validator_core')
    runner = module(a['sources']['runner'], 'rc_v8_cross_support_validator_checkpoint_reader')
    cp = read(checked(a['sources']['checkpoint']))
    need(cp['schema'] == runner.SCHEMA and cp['update_index'] == 512
         and cp['contract_sha256'] == a['parent_contract_sha256'] and cp['order_sha256'] == a['order_sha256'], 'VALIDATOR_FINAL_CHECKPOINT_INVALID')
    heads = core.GroupedResidualHeads()
    heads.load_state_dict(runner.unpack(cp['model'], torch), strict=True)
    heads.eval()
    expected_weight_sha256 = core.tensor_sha(heads.real.weights().detach())
    seal = read(OUT / 'prejoin_seal.json')
    need(seal['status'] == 'RC_V8_CROSS_SUPPORT_FULL32_C128_PREJOIN_CLOSED' and seal['authority_sha256'] == sha(AUTH)
         and seal['label_read_attempts_before_closure'] == 0 and len(seal['records']) == 32, 'PREJOIN_SEAL_INVALID')
    need(sha(OUT / 'input_closure.json') == seal['input_closure_sha256'], 'INPUT_CLOSURE_DRIFT')
    closure = read(OUT / 'input_closure.json')
    need(closure['input_manifest_sha256'] == a['input_manifest_sha256'] and closure['prejoin_label_reads'] == 0
         and len(closure['records']) == 32, 'INPUT_CLOSURE_INVALID')
    source_receipts = {row['query_resource_key']: logical(row) for row in closure['records']}
    saved_list = [json.loads(line) for line in checked(a['sources']['saved_scores']).read_text().splitlines()]
    saved = {row['query_resource_key']: row for row in saved_list}
    need(len(saved_list) == len(saved) == 32, 'SAVED_QUERY_AXIS')
    seen, weight_hashes = set(), set()
    for item in seal['records']:
        key = item['query_resource_key']
        need(key not in seen and item['path'] == 'prejoin/' + key + '.json', 'PREJOIN_QUERY_AXIS_OR_PATH')
        seen.add(key)
        path = OUT / item['path']
        need(sha(path) == item['sha256'], 'QUERY_HASH_DRIFT')
        query, old = read(path), saved[key]
        need(query['query_resource_key'] == key and query['candidate_keys'] == old['candidate_keys'] == sorted(set(old['candidate_keys']))
             and len(query['candidate_keys']) == len(query['owners']) == 128, 'C128_AXIS')
        need(query['authority_sha256'] == sha(AUTH) and query['saved_query_sha256'] == logical(old)
             and query['source_receipt_sha256'] == source_receipts[key], 'QUERY_LINEAGE')
        need(query['original_scores_binary64'] == old['arms']['REAL']['scores_binary64'] and query['target_reads'] == 0
             and query['candidate_or_support_reselection_count'] == 0 and query['cross_support_candidate_legality_asserted'] is False, 'QUERY_SCOPE')
        weight_hashes.add(query['weight_sha256'])
        for i, row in enumerate(query['owners']):
            decision = old['arms']['REAL']['decisions'][i]
            support = decision['selected_H']
            need(row['owner_candidate_resource_key'] == query['candidate_keys'][i] == decision['candidate_resource_key']
                 and row['source_H_sha256'] == logical(support) and row['saved_map_seed_ordinal'] == decision['map_seed_ordinal']
                 and row['support_state'] == decision['state'], 'OWNER_LINEAGE')
            fields = ('scores_binary64', 'energies_binary64', 'missing_counts', 'reference_group_counts', 'mean_terms_binary64', 'max_terms_binary64')
            need(all(len(row[field]) == 128 for field in fields), 'CROSS_AXIS')
            if support is None:
                need(row['source_query_indices'] == [] and row['atom_count'] == 0 and row['diagonal_witness'] is None
                     and row['unavailable_reason'] == 'OWNER_STRUCTURAL_H0_NO_OBSERVATION_DOMAIN'
                     and all(value is None for field in fields for value in row[field]), 'H0_NOT_UNAVAILABLE')
            else:
                need(row['source_query_indices'] == support['source_query_indices'] and row['atom_count'] == len(support['source_query_indices'])
                     and len(set(row['source_query_indices'])) == row['atom_count'] and row['diagonal_witness'] == support['pooling_witness']
                     and row['scores_binary64'][i] == decision['candidate_score_binary64'] and row['missing_counts'][i] == 0
                     and row['energies_binary64'][i] == support['pooling_witness']['coherent_energy_binary64'], 'DIAGONAL_OR_SUPPORT_DRIFT')
                for score, energy, missing, groups, mean_term, max_term in zip(*(row[field] for field in fields), strict=True):
                    need(type(missing) is int and 0 <= missing <= row['atom_count'], 'MISSING_COUNT_INVALID')
                    if missing:
                        need(score is None and energy is None and groups is None and mean_term is None and max_term is None, 'MISSING_ASSIGNMENT_SCORED')
                    else:
                        need(score is not None and energy is not None and 0 < number(score) <= 1 and number(energy) >= 0, 'COMPLETE_SCORE_INVALID')
                        need(type(groups) is int and 1 <= groups <= row['atom_count'] and number(mean_term) >= 0 and number(max_term) >= 0
                             and hx(number(mean_term) + number(max_term)) == energy, 'ENERGY_REFERENCE_GROUP_ACCOUNTING')
    need(seen == set(saved) == set(source_receipts) and weight_hashes == {expected_weight_sha256}, 'FULL32_OR_PINNED_CHECKPOINT_WEIGHT_AXIS')
    expected = summarize(a, seal, saved_list)
    need(read(OUT / 'result.json') == expected, 'POSTJOIN_ARITHMETIC_OR_SUMMARY_DRIFT')
    for pair in read(OUT / 'result.json')['records']:
        verify_exact_terms(pair)
    receipt = {'status': 'RC_V8_CROSS_SUPPORT_INDEPENDENT_ARTIFACT_VALIDATION_PASS',
               'claim_level': 'SAVED_ARTIFACT_HASH_AXIS_DIAGONAL_AND_EXACT_ARITHMETIC_VALIDATION_NO_RAW_CROSS_SCORE_REPLAY',
               'authority_sha256': sha(AUTH), 'result_sha256': sha(OUT / 'result.json'),
               'checks': {'full32_C128_source_axes_and_hashes': True, 'full128_by128_saved_matrix_axes': True,
                   'diagonal_matches_original_score_and_physical_witness': True, 'H0_and_any_missing_remain_unavailable': True,
                   'source_domains_and_owner_support_hashes': True, 'exact_rational_dual_accounting_recomputed': True,
                   'weight_hash_matches_pinned_final_checkpoint_REAL_head': True,
                   'fixed_original_pair_and_fullC128_known_unknown_accounting': True, 'original_NO_GO_preserved': expected['original_v8_gate_go'] is False},
               'raw_input_cross_score_replay_performed': False, 'scientific_GO_or_NO_GO': None, 'automatic_stage_advance': False}
    atomic(OUT / 'artifact_validation.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('e0', 'freeze', 'run', 'validate'), required=True)
    args = parser.parse_args()
    if args.phase == 'e0':
        print(json.dumps(synthetic_e0(), sort_keys=True))
    elif args.phase == 'freeze':
        freeze()
    elif args.phase == 'validate':
        validate()
    else:
        raise SystemExit(run())
