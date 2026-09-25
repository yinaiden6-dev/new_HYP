#!/usr/bin/env python3
"""Exact scalar completion bounds on all frozen V8 legal REAL regions; audit only."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import signal
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'rc_v8_full_family_completion_certificate_v1'
AUTH = ROOT / ('registry/' + PREFIX + '_authority_20260909.json')
PLAN = ROOT / 'plan/RC_V8_FULL_FAMILY_COMPLETION_CERTIFICATE_V1_20260909.md'
LAUNCH = ROOT / ('slurm/' + PREFIX + '_dev_cpuonly_59m.sbatch')
OUT = ROOT / ('results/' + PREFIX)
BASE_PATH = ROOT / 'programs/run_rc_v8_cross_support_diagnostic_v1.py'
BASE_SHA = '8c6deb905cea73128efdbde206ab53a702d3548775b2ee209ad0d27f1e7fe3e3'
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA:
    raise RuntimeError('PINNED_PARENT_PROGRAM_DRIFT')
spec = importlib.util.spec_from_file_location('rc_v8_completion_parent', BASE_PATH)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)
need, read, sha, logical, checked, atomic = b.need, b.read, b.sha, b.logical, b.checked, b.atomic
binding, module, hx, number, rational = b.binding, b.module, b.hx, b.number, b.rational
EXTRA = {
    'cross_program': (BASE_PATH, BASE_SHA),
    'cross_result': (b.OUT / 'result.json', '7a62985feaefcb0e102ab3229ab78d8fb76d661531ce86c394b8b24c7d9f054a'),
    'cross_validation': (b.OUT / 'artifact_validation.json', '0e2dfda678217feed8a7bba36f0cb9b35350a8abff48868c72a66e9fed6b155c'),
}
STOP = False


def frac(encoded):
    value = Fraction(int(encoded['numerator']), int(encoded['denominator']))
    need(rational(value) == encoded, 'NONCANONICAL_FRACTION')
    return value


def exact_bound(group_costs, maximum, missing):
    """Ideal exact scalar expression of the saved, already FP64-rounded atom costs."""
    need(type(missing) is int and missing >= 0, 'MISSING_DOMAIN')
    vals = [Fraction.from_float(number(x)) for x in group_costs]
    mx = Fraction.from_float(number(maximum))
    need(all(v >= 0 for v in vals) and mx >= 0, 'NEGATIVE_COST')
    if not vals:
        need(missing > 0 and mx == 0, 'EMPTY_OBSERVED_DOMAIN')
        return Fraction(0)
    return sum(vals, Fraction(0)) / (len(vals) + missing) + mx


def classify(cells, owner):
    own = cells[owner]
    need(own['missing_count'] == 0, 'OWNER_NOT_COMPLETE')
    energy = frac(own['exact_energy_or_lower_bound'])
    differences = [(i, frac(c['exact_energy_or_lower_bound']) - energy) for i, c in enumerate(cells) if i != owner]
    known = [i for i, delta in differences if delta <= 0 and cells[i]['missing_count'] == 0]
    unknown = [i for i, delta in differences if delta <= 0 and cells[i]['missing_count'] > 0]
    state = 'REFUTED' if known else 'UNRESOLVED' if unknown else 'CERTIFIED'
    return {'status': state, 'owner_exact_energy': rational(energy),
            'minimum_other_bound_minus_owner_exact': rational(min(delta for _, delta in differences)),
            'known_complete_refuter_candidate_indices': known,
            'unknown_noncertifying_candidate_indices': unknown,
            'all_other_candidate_count': len(cells) - 1}


def prepared_bank(bank, valid):
    import torch
    sources = bank.source_query_indices.tolist()
    need(len(set(sources)) == len(sources), 'NONUNIQUE_SOURCE_AXIS')
    qok = bank.query_valid_axis[bank.source_query_indices]
    inside = bank.source_forward_reference_xy.abs().lt(1.0).all(dim=1)
    rok = bank.reference_valid_axis[bank.source_reference_indices]
    need(torch.equal(qok & inside & rok, valid) and torch.equal(valid, bank.valid_mask), 'ORIGINAL_VALIDITY_FORMULA_DRIFT')
    return {'all': dict(zip(sources, range(len(sources)))), 'valid': b.prepare_cross_axis(bank.source_query_indices, valid),
            'causes': {s: {'query_invalid': not bool(qok[i]), 'forward_out_of_bounds': not bool(inside[i]),
                           'assigned_reference_invalid': not bool(rok[i])} for i, s in enumerate(sources)}}


def certificate_cell(core, residuals, refs, sources, domain, weights, axis):
    """Never evaluate an invalid assignment; UNKNOWN has only a hypothetical bound."""
    import torch
    known = [s for s in domain if s in axis['valid']]
    missing = [s for s in domain if s not in axis['valid']]
    causes = Counter()
    for source in missing:
        need(source in axis['all'], 'SOURCE_QUERY_AXIS_NOT_COMPLETE')
        reason = axis['causes'][source]
        need(any(reason.values()), 'MISSING_WITHOUT_ORIGINAL_VALIDITY_CAUSE')
        causes.update(k for k, v in reason.items() if v)
    witness, score, energy = None, None, None
    if known:
        ids = torch.tensor([axis['valid'][s] for s in known], dtype=torch.long)
        src = torch.tensor(known, dtype=torch.long)
        if missing:
            _, witness = core.coherent_energy(residuals[ids], refs[ids], weights, src)
        else:
            affinity, witness = core.coherent_score(residuals[ids], refs[ids], weights, src)
            score, energy = hx(affinity), witness['coherent_energy_binary64']
    group_costs = [] if witness is None else witness['group_mean_cost_binary64']
    maximum = hx(0.0) if witness is None else witness['global_max_cost_binary64']
    exact = exact_bound(group_costs, maximum, len(missing))
    cell = {'observation_status': 'UNKNOWN_PARTIAL' if missing else 'COMPLETE',
            'missing_count': len(missing), 'observed_count': len(known), 'missing_source_query_indices': missing,
            'missing_cause_counts_nonexclusive': dict(causes),
            'observed_reference_group_indices': [] if witness is None else witness['reference_group_indices'],
            'observed_group_max_a_binary64': group_costs,
            'observed_group_witness_source_query_indices': [] if witness is None else witness['group_mean_witness_source_query_indices'],
            'observed_max_b_binary64': maximum,
            'observed_max_b_witness_source_query_index': None if witness is None else witness['global_max_witness_source_query_index'],
            'observed_group_count': len(group_costs), 'bound_denominator': len(group_costs) + len(missing),
            'exact_energy_or_lower_bound': rational(exact),
            'original_score_binary64': score, 'original_energy_binary64': energy,
            'exact_energy_minus_original_fp64_energy': None if missing else rational(exact-Fraction.from_float(number(energy))),
            'bound_is_observed_score': False}
    return cell, witness


def verify_cell(cell, domain):
    missing, observed, groups = cell['missing_count'], cell['observed_count'], cell['observed_group_count']
    need(type(missing) is int and 0 <= missing <= len(domain) and observed + missing == len(domain), 'OBSERVATION_COUNTS')
    need(len(cell['missing_source_query_indices']) == len(set(cell['missing_source_query_indices'])) == missing
         and set(cell['missing_source_query_indices']).issubset(domain), 'MISSING_SOURCE_AXIS')
    remaining = set(domain) - set(cell['missing_source_query_indices'])
    refs = cell['observed_reference_group_indices']
    need(refs == sorted(set(refs)) and len(refs) == groups and 0 <= groups <= observed
         and (groups > 0) == (observed > 0) and all(type(x) is int and x >= 0 for x in refs), 'REFERENCE_GROUP_AXIS')
    need(len(cell['observed_group_max_a_binary64']) == len(cell['observed_group_witness_source_query_indices']) == groups
         and set(cell['observed_group_witness_source_query_indices']).issubset(remaining), 'GROUP_WITNESS_AXIS')
    need(cell['bound_denominator'] == groups + missing and cell['bound_is_observed_score'] is False, 'BOUND_DOMAIN')
    expected = exact_bound(cell['observed_group_max_a_binary64'], cell['observed_max_b_binary64'], missing)
    need(frac(cell['exact_energy_or_lower_bound']) == expected, 'EXACT_BOUND_REPLAY_DRIFT')
    causes = cell['missing_cause_counts_nonexclusive']
    need(set(causes).issubset({'query_invalid', 'forward_out_of_bounds', 'assigned_reference_invalid'})
         and all(type(v) is int and 0 < v <= missing for v in causes.values())
         and sum(causes.values()) >= missing, 'MISSING_CAUSE_ACCOUNTING')
    if missing:
        need(cell['observation_status'] == 'UNKNOWN_PARTIAL' and cell['original_score_binary64'] is None
             and cell['original_energy_binary64'] is None and cell['exact_energy_minus_original_fp64_energy'] is None, 'UNKNOWN_WAS_SCORED')
    else:
        need(cell['observation_status'] == 'COMPLETE' and 0 < number(cell['original_score_binary64']) <= 1
             and number(cell['original_energy_binary64']) >= 0, 'COMPLETE_ORIGINAL_SCORE')
        import torch
        original_mean = torch.tensor([number(v) for v in cell['observed_group_max_a_binary64']], dtype=torch.float64).mean()
        original_energy = original_mean + number(cell['observed_max_b_binary64'])
        need(hx(original_energy) == cell['original_energy_binary64']
             and hx(torch.exp(-0.5 * original_energy)) == cell['original_score_binary64'], 'ORIGINAL_FP64_LINK_REPLAY')
        need(frac(cell['exact_energy_minus_original_fp64_energy']) == expected-Fraction.from_float(number(cell['original_energy_binary64'])),
             'EXACT_IDEAL_VS_ORIGINAL_FP64_DIFFERENCE_DRIFT')
    need((cell['observed_max_b_witness_source_query_index'] in remaining) if observed
         else cell['observed_max_b_witness_source_query_index'] is None, 'GLOBAL_WITNESS_AXIS')


def synthetic_e0():
    import torch
    torch.set_num_threads(1)
    core = module({'path': 'src/rc_aslo_xf/reference_conditioned_coherent_residual_p_only_v8.py', 'sha256': b.PINS['core']}, 'rc_completion_e0_core')
    def scalar_energy(rows):
        groups = {}
        for group, a, _ in rows:
            groups[group] = max(groups.get(group, Fraction(0)), Fraction(a))
        return sum(groups.values(), Fraction(0)) / len(groups) + max(Fraction(v[2]) for v in rows)
    cases = 0
    # Exhaust all 0/1/2 scalar costs and assignments to old or new groups.
    for observed in ([], [(0, 1, 2)], [(0, 2, 1), (1, 1, 2)], [(0, 1, 1), (0, 1, 1)]):
        for m in (1, 2):
            grouped = {}
            for group, aa, _ in observed:
                grouped[group] = max(grouped.get(group, 0), aa)
            lower = exact_bound([hx(v) for v in grouped.values()], hx(max((v[2] for v in observed), default=0)), m)
            for assignments in itertools.product(range(4), repeat=m):
                for costs in itertools.product(range(3), repeat=2 * m):
                    full = observed + [(assignments[i], costs[2*i], costs[2*i+1]) for i in range(m)]
                    need(scalar_energy(full) >= lower, 'EXHAUSTIVE_COMPLETION_BOUND_FAILED')
                    cases += 1
    residual = torch.zeros((4, 128), dtype=torch.float64)
    residual[:, 0] = torch.tensor([1., 1., 2., 3.], dtype=torch.float64)
    sources = torch.tensor([8, 2, 7, 3]); refs = torch.tensor([0, 0, 1, 1])
    weights = torch.ones(256, dtype=torch.float64)
    axis = {'all': dict(zip(sources.tolist(), range(4))), 'valid': {8:0, 2:1, 7:2, 3:3},
            'causes': {s: {'forward_out_of_bounds': False} for s in sources.tolist()}}
    full, witness = certificate_cell(core, residual, refs, sources, [8,2,7,3], weights, axis)
    verify_cell(full, [8,2,7,3])
    direct = b.cross_score(core, residual, refs, sources, torch.ones(4,dtype=torch.bool), [8,2,7,3], weights)
    need(full['original_score_binary64'] == direct['score_binary64'] and witness == direct['witness'], 'COMPLETE_PARENT_PARITY')
    need(witness['group_mean_witness_source_query_indices'] == [2,3], 'PHYSICAL_TIE_WITNESS')
    partial_axis = {'all': axis['all'], 'valid': {8:0,2:1},
                    'causes': {s: {'forward_out_of_bounds': s in (7,3)} for s in sources.tolist()}}
    partial, _ = certificate_cell(core, residual, refs, sources, [8,2,7,3], weights, partial_axis)
    verify_cell(partial, [8,2,7,3])
    poisoned=residual.clone(); poisoned[2:]=float('nan')
    poison_partial,_=certificate_cell(core,poisoned,refs,sources,[8,2,7,3],weights,partial_axis)
    need(poison_partial==partial,'INVALID_PLACEHOLDER_RESIDUAL_WAS_READ')
    from types import SimpleNamespace
    native=SimpleNamespace(source_query_indices=torch.arange(4),source_reference_indices=torch.tensor([0,1,0,1]),
        query_valid_axis=torch.tensor([True,True,False,True]),reference_valid_axis=torch.tensor([True,False]),
        source_forward_reference_xy=torch.tensor([[0.,0.],[1.,0.],[0.,0.],[-1.,0.]],dtype=torch.float64),
        valid_mask=torch.tensor([True,False,False,False]))
    native_axis=prepared_bank(native,native.valid_mask)
    need(native_axis['valid']=={0:0} and native_axis['causes'][1]['forward_out_of_bounds']
         and native_axis['causes'][1]['assigned_reference_invalid'] and native_axis['causes'][2]['query_invalid']
         and native_axis['causes'][3]['forward_out_of_bounds'],'NATIVE_VALIDITY_AND_BOUNDARY_CAUSES')
    missing_axis = {'all':axis['all'], 'valid':{}, 'causes': {s:{'forward_out_of_bounds':True} for s in sources.tolist()}}
    absent, _ = certificate_cell(core,residual,refs,sources,[8,2,7,3],weights,missing_axis)
    verify_cell(absent,[8,2,7,3]); need(frac(absent['exact_energy_or_lower_bound']) == 0,'ALL_MISSING_BOUND')
    need(classify([full, full],0)['status'] == 'REFUTED','TIE_NOT_REFUTER')
    need(classify([full, absent],0)['status'] == 'UNRESOLVED','UNKNOWN_DISCARDED')
    high = json.loads(b.encode(full)); high['exact_energy_or_lower_bound'] = rational(frac(full['exact_energy_or_lower_bound']) + 1)
    need(classify([full, high],0)['status'] == 'CERTIFIED','STRICT_CERTIFICATE')
    perm = torch.tensor([3,1,0,2]); pa = {'all':dict(zip(sources[perm].tolist(),range(4))),
        'valid':dict(zip(sources[perm].tolist(),range(4))), 'causes':axis['causes']}
    permuted, pw = certificate_cell(core,residual[perm],refs[perm],sources[perm],[8,2,7,3],weights,pa)
    need((permuted,pw) == (full,witness),'SOURCE_PERMUTATION_CHANGED_SCORE')
    try:
        verify_cell(high,[8,2,7,3])
    except RuntimeError:
        pass
    else:
        raise AssertionError('TAMPERED_BOUND_ACCEPTED')
    with tempfile.TemporaryDirectory(prefix='.rc-v8-completion-e0-', dir=ROOT/'results') as tmp:
        artifact = Path(tmp)/'a.json'; atomic(artifact,{'x':1}); original = binding(artifact)
        atomic(artifact,{'x':1})
        try:
            atomic(artifact,{'x':2})
        except RuntimeError:
            pass
        else:
            raise AssertionError('IMMUTABLE_OVERWRITE_ACCEPTED')
        artifact.chmod(0o600); artifact.write_text('{}')
        try:
            checked(original)
        except RuntimeError:
            pass
        else:
            raise AssertionError('HASH_TAMPER_ACCEPTED')
    return {'status':'RC_V8_FULL_FAMILY_COMPLETION_CERTIFICATE_SYNTHETIC_E0_PASS',
            'exhaustive_completion_cases':cases, 'checks':{
                'nonnegative_scalar_cost_completion_bound':True, 'old_group_merges_new_groups_and_ties':True,
                'all_missing_zero_is_bound_only':True, 'unknown_never_observed_or_imputed':True,
                'invalid_placeholder_nan_residual_never_read':True,'original_native_validity_strict_boundary_and_overlapping_causes':True,
                'complete_original_score_witness_bit_exact':True, 'source_axis_permutation':True,
                'strict_certificate_known_refuter_unknown_distinct':True, 'exact_bound_tamper_rejected':True,
                'immutable_hash_tamper_rejected':True}, 'natural_value_reads':0, 'scientific_GO_or_NO_GO':None}


def freeze():
    need(not AUTH.exists(), 'AUTHORITY_EXISTS')
    parent = b.load_authority()
    files = dict(parent['sources'])
    for name, (path,pin) in EXTRA.items():
        need(sha(path) == pin,'EXTRA_SOURCE_DRIFT:'+name); files[name] = binding(path)
    files.update(program=binding(Path(__file__)), plan=binding(PLAN), launcher=binding(LAUNCH),
        cross_authority=binding(b.AUTH), cross_seal=binding(b.OUT/'prejoin_seal.json'),
        generation_seal=binding(ROOT/'results/rc_cycle_closed_region_generation_v1/prejoin_seal.json'))
    e0_path = ROOT/('results/'+PREFIX+'_e0/result.json'); atomic(e0_path,synthetic_e0()); files['e0']=binding(e0_path)
    cv = read(checked(files['cross_validation']))
    need(cv['status'] == 'RC_V8_CROSS_SUPPORT_INDEPENDENT_ARTIFACT_VALIDATION_PASS'
         and cv['result_sha256'] == EXTRA['cross_result'][1] and all(cv['checks'].values()), 'CROSS_NOT_VALIDATED')
    authority = {'status':'RC_V8_FULL_FAMILY_COMPLETION_CERTIFICATE_AUTHORIZED', 'sources':files,
        'postjoin':parent['postjoin'], 'input_manifest_sha256':parent['input_manifest_sha256'],
        'parent_contract_sha256':parent['parent_contract_sha256'], 'order_sha256':parent['order_sha256'],
        'output_rel':str(OUT.relative_to(ROOT)), 'query_count':32,'candidate_count':128,
        'expected_component_count':2691,'expected_nonempty_owner_count':1672,
        'denominator_policy':'OBSERVED_REFERENCE_GROUP_COUNT_PLUS_MISSING_QUERY_ATOM_COUNT',
        'comparison_arithmetic':'EXACT_RATIONAL_ON_SAVED_FP64_ATOM_COSTS_NOT_FP64_COMPLETION_MODEL',
        'unknown_policy':'RETAIN_UNKNOWN_HYPOTHETICAL_NONNEGATIVE_COST_BOUND_ONLY',
        'training_authorized':False,'candidate_ranking_authorized':False,'formal_panel_authorized':False,
        'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}
    atomic(AUTH,authority)
    print(json.dumps({'status':authority['status'],'authority_sha256':sha(AUTH)}),flush=True)


def load_authority():
    a = read(AUTH)
    need(a['status']=='RC_V8_FULL_FAMILY_COMPLETION_CERTIFICATE_AUTHORIZED'
         and a['output_rel']==str(OUT.relative_to(ROOT)),'AUTHORITY_INVALID')
    for value in a['sources'].values(): checked(value)
    for name,pin in b.PINS.items(): need(a['sources'][name]['sha256']==pin,'ORIGINAL_V8_PIN_DRIFT')
    for name,(_,pin) in EXTRA.items(): need(a['sources'][name]['sha256']==pin,'CROSS_PIN_DRIFT')
    parent = read(checked(a['sources']['cross_authority']))
    for name in ('postjoin','input_manifest_sha256','parent_contract_sha256','order_sha256'):
        need(a[name]==parent[name],'PARENT_SCOPE_DRIFT:'+name)
    need(a['sources']['program'] == binding(Path(__file__)) and a['sources']['plan']==binding(PLAN)
         and a['sources']['launcher']==binding(LAUNCH),'CURRENT_PROGRAM_BINDING')
    return a


def load_head(a):
    import torch
    torch.set_num_threads(1)
    core = module(a['sources']['core'],'rc_completion_core')
    runner = module(a['sources']['runner'],'rc_completion_runner')
    cp = read(checked(a['sources']['checkpoint']))
    need(cp['schema']==runner.SCHEMA and cp['update_index']==512
         and cp['contract_sha256']==a['parent_contract_sha256'] and cp['order_sha256']==a['order_sha256'],'FINAL_CHECKPOINT_INVALID')
    heads=core.GroupedResidualHeads(); heads.load_state_dict(runner.unpack(cp['model'],torch),strict=True); heads.eval()
    return core, heads.real.weights().detach()


def saved_inputs(a):
    old = [json.loads(line) for line in checked(a['sources']['saved_scores']).read_text().splitlines()]
    need(len(old)==32 and len({r['query_resource_key'] for r in old})==32,'ORIGINAL_FULL32_AXIS')
    seal=read(checked(a['sources']['cross_seal']))
    need(len(seal['records'])==32 and seal['authority_sha256']==a['sources']['cross_authority']['sha256'],'CROSS_SEAL')
    cross={}
    for item in seal['records']:
        key=item['query_resource_key']; need(item['path']=='prejoin/'+key+'.json','CROSS_PATH')
        path=b.OUT/item['path']; need(sha(path)==item['sha256'],'CROSS_QUERY_HASH')
        cross[key]=path
    need(set(cross)=={r['query_resource_key'] for r in old},'CROSS_FULL32_AXIS')
    return {r['query_resource_key']:r for r in old},cross


def replay_selected(row, old_owner, selected):
    need(row['support']=={k:v for k,v in selected.items() if k!='pooling_witness'}
         and logical(selected)==old_owner['source_H_sha256'],'ORIGINAL_SELECTED_SUPPORT_DRIFT')
    need(row['owner_witness']==selected['pooling_witness']==old_owner['diagonal_witness'],'ORIGINAL_SELECTED_WITNESS_DRIFT')
    for j,c in enumerate(row['cells']):
        need(c['missing_count']==old_owner['missing_counts'][j]
             and c['original_score_binary64']==old_owner['scores_binary64'][j]
             and c['original_energy_binary64']==old_owner['energies_binary64'][j],'ORIGINAL_SELECTED_FULL_C128_REPLAY_DRIFT')
        if c['missing_count']==0:
            import torch
            mean = torch.tensor([number(x) for x in c['observed_group_max_a_binary64']],dtype=torch.float64).mean()
            need(hx(mean)==old_owner['mean_terms_binary64'][j]
                 and c['observed_max_b_binary64']==old_owner['max_terms_binary64'][j]
                 and c['observed_group_count']==old_owner['reference_group_counts'][j],'ORIGINAL_SELECTED_COST_REPLAY_DRIFT')


def run():
    global STOP
    import torch
    began=time.monotonic(); a=load_authority(); core,weights=load_head(a)
    need(not OUT.exists(),'OUTPUT_EXISTS_NO_RESUME_AUTHORIZED')
    adapter=module(a['sources']['adapter'],'rc_completion_adapter')
    bundle=adapter.InputBundle(ROOT,expected_manifest_sha256=a['input_manifest_sha256'])
    saved,cross_paths=saved_inputs(a); records=[]; component_count=0; replay_count=0
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGUSR1,stop); signal.signal(signal.SIGTERM,stop)
    def guard():
        if STOP or time.monotonic()-began>3000:
            raise TimeoutError('INTERNAL_3000S_OR_SIGNAL')
    try:
        with torch.no_grad():
            for ep in bundle.iter_queries():
                guard(); key=ep['query_resource_key']; keys=list(ep['candidate_keys']); old=saved[key]; cross=read(cross_paths[key])
                need(keys==old['candidate_keys']==cross['candidate_keys']==sorted(set(keys)) and len(keys)==128,'FULL_C128_AXIS')
                banks=ep['banks_by_control']['REAL']
                need(all(bank.reference_resource_key==keys[i] for i,bank in enumerate(banks)),'REAL_REFERENCE_BINDING')
                axes=[prepared_bank(bank,ep['valid']['REAL'][i]) for i,bank in enumerate(banks)]
                owners=[]
                for ci,candidate in enumerate(keys):
                    decision=old['arms']['REAL']['decisions'][ci]; components=ep['families']['REAL'][ci]['components']
                    need(decision['candidate_resource_key']==candidate and decision['arm']=='REAL' and decision['control']=='REAL','DECISION_AXIS')
                    selected=decision['selected_H']; need(bool(components)==(selected is not None),'H0_FAMILY_DRIFT')
                    rows=[]
                    for ordinal,component in enumerate(components):
                        guard(); support=core._support(component); domain=support['source_query_indices']
                        need(len(domain)==len(set(domain))>=4 and [axes[ci]['valid'].get(s) for s in domain]==support['atom_indices'],'LEGAL_OWNER_SOURCE_AXIS')
                        cells=[]; own_witness=None
                        for cj,bank in enumerate(banks):
                            if cj%16==0: guard()
                            cell,witness=certificate_cell(core,ep['residual_squared']['REAL'][cj],bank.source_reference_indices,
                                bank.source_query_indices,domain,weights,axes[cj])
                            cells.append(cell)
                            if cj==ci: own_witness=witness
                        row={'component_ordinal':ordinal,'component_sha256':logical(component),'support':support,
                            'support_sha256':logical(support),'owner_witness':own_witness,'cells':cells,
                            'original_selected_H':ordinal==decision['map_seed_ordinal'],'certificate':classify(cells,ci)}
                        if row['original_selected_H']:
                            need(cells[ci]['original_score_binary64']==decision['candidate_score_binary64'],'SELECTED_OWNER_SCORE')
                            replay_selected(row,cross['owners'][ci],selected); replay_count+=1
                        rows.append(row); component_count+=1
                    owners.append({'candidate_resource_key':candidate,'original_state':decision['state'],
                        'original_map_seed_ordinal':decision['map_seed_ordinal'],'component_count':len(rows),'components':rows})
                query={'status':'RC_V8_FULL_FAMILY_COMPLETION_QUERY_PREJOIN_SEALED','query_resource_key':key,'candidate_keys':keys,
                    'authority_sha256':sha(AUTH),'source_receipt_sha256':logical(ep['source_receipt']),
                    'weight_sha256':core.tensor_sha(weights),'original_query_sha256':logical(old),'cross_query_sha256':sha(cross_paths[key]),
                    'original_scores_binary64':old['arms']['REAL']['scores_binary64'],'owners':owners,'target_reads':0,
                    'family_axis_sha256':logical(ep['families']),'training_updates':0,'candidate_ranking_produced':False}
                dest=OUT/'prejoin'/(key+'.json'); atomic(dest,query)
                records.append({'query_resource_key':key,'path':'prejoin/'+key+'.json','sha256':sha(dest)})
                print(json.dumps({'event':'FULL_FAMILY_QUERY_SEALED','query_count':len(records),'component_count':component_count}),flush=True)
        need(len(records)==32 and component_count==2691 and replay_count==1672,'FULL_POPULATION_NOT_CLOSED')
        closure=bundle.prejoin_receipt(); atomic(OUT/'input_closure.json',closure)
        seal={'status':'RC_V8_FULL_FAMILY_COMPLETION_FULL32_C128_PREJOIN_CLOSED','authority_sha256':sha(AUTH),
            'records':sorted(records,key=lambda r:r['query_resource_key']),'component_count':component_count,
            'selected_full_C128_rows_replayed':replay_count,'input_closure_sha256':sha(OUT/'input_closure.json'),
            'label_read_attempts_before_closure':bundle.barrier.blocked_read_attempts}
        atomic(OUT/'prejoin_seal.json',seal)
        atomic(OUT/'result.json',summarize(a,seal))
        print(json.dumps({'status':'RC_V8_FULL_FAMILY_COMPLETION_CERTIFICATE_COMPLETE','scientific_GO_or_NO_GO':None}),flush=True)
        return 0
    except TimeoutError:
        receipt={'status':'INCOMPLETE_TIME_LIMIT_OR_SIGNAL','sealed_queries':len(records),'processed_component_count':component_count,
            'authority_sha256':sha(AUTH),'resumable':False,'scientific_GO_or_NO_GO':None}
        atomic(OUT/'incomplete.json',receipt); print(json.dumps(receipt),flush=True); return 75


def summarize(a,seal):
    need(sha(OUT/'prejoin_seal.json') and len(seal['records'])==32,'POSTJOIN_BEFORE_FULL32_SEAL')
    joined=read(checked(a['postjoin']['authority'])); validation=read(checked(a['postjoin']['validation']))
    need(joined['record_count']==32 and joined['target_insertion_count']==0 and joined['fold_id']==1,'POSTJOIN_SCOPE')
    need(validation['authority_sha256']==a['postjoin']['authority']['sha256'] and validation['checks']
         and all(v is True for v in validation['checks'].values()),'POSTJOIN_VALIDATION')
    roles={r['query_resource_key']:r for r in joined['records']}; records=[]; counts=Counter()
    need(set(roles)=={r['query_resource_key'] for r in seal['records']},'POSTJOIN_QUERY_AXIS')
    for item in seal['records']:
        path=OUT/item['path']; need(sha(path)==item['sha256'],'PREJOIN_HASH_DRIFT'); q=read(path)
        keys=q['candidate_keys']; targets=set(roles[item['query_resource_key']]['target_candidate_resource_keys'])
        need(targets and targets.issubset(keys),'TARGET_AXIS')
        certified=[]; not_refuted=[]; target_details=[]; selected_target_states=[]
        for owner in q['owners']:
            states=[c['certificate']['status'] for c in owner['components']]; ck=owner['candidate_resource_key']
            counts.update(states); counts['owner_count']+=1; counts['H0_owner_count']+=not states
            if 'CERTIFIED' in states: certified.append(ck)
            if any(s!='REFUTED' for s in states): not_refuted.append(ck)
            for row in owner['components']:
                for cell in row['cells']:
                    counts['complete_cross_cells' if cell['missing_count']==0 else 'unknown_cross_cells']+=1
            if ck in targets:
                selected=[row for row in owner['components'] if row['original_selected_H']]
                selected_state=selected[0]['certificate']['status'] if selected else 'H0'
                selected_target_states.append(selected_state)
                target_details.append({'candidate_resource_key':ck,'H0':not states,'component_statuses':states,
                    'original_selected_H_certificate_status':selected_state,'original_map_seed_ordinal':owner['original_map_seed_ordinal']})
        target_cert=bool(targets.intersection(certified)); target_possible=bool(targets.intersection(not_refuted))
        wrong_cert=sorted(set(certified)-targets)
        wrong_not_refuted=sorted(set(not_refuted)-targets)
        scores=[number(x) for x in q['original_scores_binary64']]
        ti=max((i for i,k in enumerate(keys) if k in targets),key=lambda i:scores[i])
        wi=max((i for i,k in enumerate(keys) if k not in targets),key=lambda i:scores[i])
        counts['queries_target_any_certified_H']+=target_cert; counts['queries_target_any_not_refuted_H']+=target_possible
        counts['queries_target_H0']+=all(x['H0'] for x in target_details)
        counts['queries_any_wrong_candidate_certified']+=bool(wrong_cert)
        counts['queries_target_certified_and_no_wrong_candidate_certified']+=target_cert and not wrong_cert
        counts['queries_unique_certified_candidate_is_target']+=len(certified)==1 and target_cert
        counts['queries_strict_unique_target_all_wrong_regions_refuted_or_H0']+=target_cert and not wrong_not_refuted
        counts['queries_original_selected_target_H_certified']+='CERTIFIED' in selected_target_states
        counts['original_V8_successes']+=scores[ti]>scores[wi]
        records.append({'query_resource_key':q['query_resource_key'],'supergroup_hash':roles[q['query_resource_key']]['supergroup_hash'],
            'target_candidates':target_details,'target_any_certified_H':target_cert,'target_any_not_refuted_H':target_possible,
            'certified_candidate_keys':certified,'wrong_certified_candidate_keys':wrong_cert,
            'wrong_candidate_keys_with_any_not_refuted_region':wrong_not_refuted,
            'unique_certified_candidate_is_target':len(certified)==1 and target_cert,
            'strict_unique_target_all_wrong_regions_refuted_or_H0':target_cert and not wrong_not_refuted,
            'original_v8_margin_binary64':hx(scores[ti]-scores[wi]),
            'original_v8_margin_exact':rational(Fraction.from_float(scores[ti])-Fraction.from_float(scores[wi]))})
    need(counts['owner_count']==4096 and counts['H0_owner_count']==2424
         and sum(counts[s] for s in ('CERTIFIED','REFUTED','UNRESOLVED'))==2691
         and counts['complete_cross_cells']+counts['unknown_cross_cells']==2691*128
         and counts['original_V8_successes']==9,'POPULATION_OR_ORIGINAL_OUTCOME_DRIFT')
    return {'status':'RC_V8_FULL_FAMILY_COMPLETION_CERTIFICATE_COMPLETE','claim_level':'POSTHOC_FIXED_WEIGHT_FULL_LEGAL_FAMILY_ORACLE_CAPACITY_AUDIT',
        'authority_sha256':sha(AUTH),'prejoin_seal_sha256':sha(OUT/'prejoin_seal.json'),'query_count':32,'candidate_count':128,
        'counts':dict(counts),'records':records,'original_v8_gate_go':False,'training_updates':0,
        'candidate_ranking_produced':False,'formal_panel_consumed':False,'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None,
        'interpretation':['Exact scalar arithmetic treats saved FP64 atom costs as exact nonnegative numbers.',
            'Certificates bound hypothetical valid-reference completions; missing assignments remain UNKNOWN.',
            'The bound is not a bound on every possible FP64 reduction rounding path.',
            'Cross-comparison supports need not be legal proposals for competing references.',
            'Existence of a certified target region is oracle capacity at fixed weights, not accuracy or a target-free selection policy.',
            'Reference grouping remains candidate dependent; the query source-cell domain is shared.',
            'Neither a count at least 26 nor improvement on selected-support diagnostics authorizes automatic training or formal evaluation.']}


def validate():
    """Separate artifact process; explicit NO independent raw cross-score recomputation."""
    a=load_authority(); core,weights=load_head(a); expected_weight=core.tensor_sha(weights)
    saved,cross_paths=saved_inputs(a); seal=read(OUT/'prejoin_seal.json'); closure=read(OUT/'input_closure.json')
    need(seal['status']=='RC_V8_FULL_FAMILY_COMPLETION_FULL32_C128_PREJOIN_CLOSED' and seal['authority_sha256']==sha(AUTH)
         and seal['label_read_attempts_before_closure']==0 and len(seal['records'])==32
         and seal['component_count']==2691 and seal['selected_full_C128_rows_replayed']==1672,'FULL32_SEAL_INVALID')
    need(sha(OUT/'input_closure.json')==seal['input_closure_sha256'] and closure['prejoin_label_reads']==0
         and closure['input_manifest_sha256']==a['input_manifest_sha256'] and len(closure['records'])==32,'INPUT_CLOSURE_INVALID')
    receipts={r['query_resource_key']:r for r in closure['records']}
    gseal=read(checked(a['sources']['generation_seal'])); generation={r['query_resource_key']:r for r in gseal['records']}
    seen=set(); total=0; replays=0
    for item in seal['records']:
        key=item['query_resource_key']; need(key not in seen and item['path']=='prejoin/'+key+'.json','QUERY_AXIS'); seen.add(key)
        path=OUT/item['path']; need(sha(path)==item['sha256'],'QUERY_HASH_DRIFT'); q=read(path); old=saved[key]; cross=read(cross_paths[key])
        gi=generation[key]; need(gi['file']=='prejoin/'+key+'.json','GENERATION_QUERY_PATH')
        gp=ROOT/'results/rc_cycle_closed_region_generation_v1'/gi['file']; need(sha(gp)==gi['sha256'],'GENERATION_HASH_DRIFT')
        gen=read(gp); families={r['candidate_resource_key']:r['components'] for r in gen['families']['REAL']}
        need(q['candidate_keys']==old['candidate_keys']==cross['candidate_keys'] and len(q['owners'])==128
             and set(families)==set(q['candidate_keys']),'C128_AXIS')
        need(q['authority_sha256']==sha(AUTH) and q['source_receipt_sha256']==logical(receipts[key])
             and q['original_query_sha256']==logical(old) and q['cross_query_sha256']==sha(cross_paths[key])
             and q['weight_sha256']==expected_weight and q['family_axis_sha256']==receipts[key]['family_axis_sha256'],'QUERY_LINEAGE')
        need(q['original_scores_binary64']==old['arms']['REAL']['scores_binary64'] and q['target_reads']==0
             and q['training_updates']==0 and q['candidate_ranking_produced'] is False,'QUERY_SCOPE')
        for ci,owner in enumerate(q['owners']):
            keyc=q['candidate_keys'][ci]; decision=old['arms']['REAL']['decisions'][ci]; comps=families[keyc]
            need(owner['candidate_resource_key']==keyc and owner['original_state']==decision['state']
                 and owner['original_map_seed_ordinal']==decision['map_seed_ordinal']
                 and owner['component_count']==len(owner['components'])==len(comps),'OWNER_AXIS')
            for ordinal,row in enumerate(owner['components']):
                support=core._support(comps[ordinal]); domain=support['source_query_indices']
                need(row['component_ordinal']==ordinal and row['component_sha256']==logical(comps[ordinal])
                     and row['support']==support and row['support_sha256']==logical(support)
                     and len(row['cells'])==128 and row['original_selected_H']==(ordinal==decision['map_seed_ordinal']),'COMPLETE_FAMILY_AXIS')
                for cell in row['cells']: verify_cell(cell,domain)
                need(row['certificate']==classify(row['cells'],ci),'CERTIFICATE_EXACT_ARITHMETIC_DRIFT')
                own=row['cells'][ci]; witness=row['owner_witness']
                need(own['missing_count']==0 and witness['group_mean_cost_binary64']==own['observed_group_max_a_binary64']
                     and witness['global_max_cost_binary64']==own['observed_max_b_binary64']
                     and witness['reference_group_indices']==own['observed_reference_group_indices']
                     and witness['group_mean_witness_source_query_indices']==own['observed_group_witness_source_query_indices']
                     and witness['global_max_witness_source_query_index']==own['observed_max_b_witness_source_query_index']
                     and witness['coherent_energy_binary64']==own['original_energy_binary64'],'OWNER_WITNESS_COST_DRIFT')
                if row['original_selected_H']:
                    replay_selected(row,cross['owners'][ci],decision['selected_H']); replays+=1
                total+=1
    need(seen==set(saved)==set(receipts)==set(generation) and total==2691 and replays==1672,'FULL_FAMILY_CLOSURE')
    expected=summarize(a,seal); need(read(OUT/'result.json')==expected,'SUMMARY_REPLAY_DRIFT')
    receipt={'status':'RC_V8_FULL_FAMILY_COMPLETION_CERTIFICATE_ARTIFACT_VALIDATION_PASS',
        'claim_level':'SOURCE_HASH_FULL_FAMILY_AXIS_SAVED_COST_EXACT_BOUND_AND_PRIOR_SELECTED_ROW_REPLAY_NO_RAW_INDEPENDENT_REPLAY',
        'authority_sha256':sha(AUTH),'result_sha256':sha(OUT/'result.json'),'checks':{
            'all32_C128_original_2691_legal_components':True,'all344448_cross_cells_bound_arithmetic':True,
            'original1672_selected_rows_fullC128_score_energy_cost_replay':True,
            'original_selected_diagonal_physical_witness_replay':True,'frozen_checkpoint_weight_hash':True,
            'missing_remains_unknown_never_actual_zero_imputation':True,'postjoin_only_after_full32_seal':True,
            'summary_and_original_nine_successes_reproduced':True},
        'raw_input_cross_score_replay_performed':False,'scientific_GO_or_NO_GO':None,'automatic_stage_advance':False}
    atomic(OUT/'artifact_validation.json',receipt); print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('e0','freeze','run','validate'))
    args=parser.parse_args()
    if args.phase=='e0': print(json.dumps(synthetic_e0(),sort_keys=True))
    elif args.phase=='freeze': freeze()
    elif args.phase=='validate': validate()
    else: raise SystemExit(run())
