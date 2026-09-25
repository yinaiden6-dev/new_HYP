#!/usr/bin/env python3
"""One frozen V8 common-query-measure diagnostic; no fitting, rank, or GO."""
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
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'rc_v8_common_query_measure_diagnostic_v1'
AUTH = ROOT / ('registry/' + PREFIX + '_authority_20260909.json')
PLAN = ROOT / 'plan/RC_V8_COMMON_QUERY_MEASURE_DIAGNOSTIC_V1_20260909.md'
LAUNCH = ROOT / ('slurm/' + PREFIX + '_dev_cpuonly_59m.sbatch')
OUT = ROOT / ('results/' + PREFIX)
PARENT_PROGRAM = ROOT / 'programs/run_rc_v8_full_family_completion_certificate_v1.py'
PARENT_PROGRAM_SHA = 'bb295dd9ab1347127cf5ade2095f6cb9e00938a22e2ea21e1d9997be55accce3'
if hashlib.sha256(PARENT_PROGRAM.read_bytes()).hexdigest() != PARENT_PROGRAM_SHA:
    raise RuntimeError('PINNED_FULL_FAMILY_PROGRAM_DRIFT')
spec = importlib.util.spec_from_file_location('rc_v8_common_measure_parent', PARENT_PROGRAM)
f = importlib.util.module_from_spec(spec); sys.modules[spec.name] = f; spec.loader.exec_module(f)
b = f.b
need, read, sha, logical, checked, atomic = f.need, f.read, f.sha, f.logical, f.checked, f.atomic
binding, module, hx, number, rational, frac = f.binding, f.module, f.hx, f.number, f.rational, f.frac
EXTRA = {
    'full_family_program': (PARENT_PROGRAM, PARENT_PROGRAM_SHA),
    'full_family_result': (f.OUT/'result.json', '47668e24854a608974bc7c585b979ae980e361a8d70338b9d2efadd40efc35d6'),
    'full_family_validation': (f.OUT/'artifact_validation.json', 'a9895dea6a645ddbfbf84bbb8d210e06cd723a5714870183d38eda996b78daea'),
}
STOP = False


def query_bound(a_costs, b_costs, n):
    """Exact scalar bound on saved FP64 costs; UNKNOWN values are not imputed."""
    need(type(n) is int and n > 0 and len(a_costs) == len(b_costs) <= n, 'COMMON_MEASURE_AXIS')
    aa = [Fraction.from_float(number(x)) for x in a_costs]
    bb = [Fraction.from_float(number(x)) for x in b_costs]
    need(all(x >= 0 for x in aa + bb), 'NONNEGATIVE_ATOM_COST')
    return sum(aa, Fraction(0))/n + max(bb, default=Fraction(0))


def common_cell(core, residual, refs, sources, domain, weights, axis):
    import torch
    grouped, witness = f.certificate_cell(core, residual, refs, sources, domain, weights, axis)
    known = [s for s in domain if s in axis['valid']]
    positions = torch.tensor([axis['valid'][s] for s in known], dtype=torch.long)
    # Evaluate only originally valid rows; invalid placeholders remain unread.
    rows = residual[positions]
    aa = (rows * weights[:core.D]).sum(dim=1)
    bb = (rows * weights[core.D:]).sum(dim=1)
    ahex, bhex = [hx(x) for x in aa], [hx(x) for x in bb]
    exact = query_bound(ahex, bhex, len(domain))
    query = {'observation_status': grouped['observation_status'], 'missing_count': grouped['missing_count'],
        'observed_count': len(known), 'bound_denominator': len(domain),
        'exact_energy_or_lower_bound': rational(exact), 'bound_is_observed_score': False,
        'arithmetic': 'EXACT_SCALAR_ON_SAVED_FP64_ATOM_COSTS', 'counterfactual_score_produced': False}
    return {'grouped': grouped, 'observed_source_query_indices': known,
        'observed_source_reference_indices': refs[positions].tolist(),
        'observed_a_binary64': ahex, 'observed_b_binary64': bhex, 'common_query': query}, witness


def verify_cell(cell, domain):
    """Replay both measures and canonical physical witnesses from saved scalars."""
    grouped = cell['grouped']; f.verify_cell(grouped, domain)
    known = [s for s in domain if s not in grouped['missing_source_query_indices']]
    need(cell['observed_source_query_indices'] == known and len(set(known)) == len(known), 'SAVED_SOURCE_DOMAIN')
    aa, bb = cell['observed_a_binary64'], cell['observed_b_binary64']
    refs = cell['observed_source_reference_indices']
    need(len(aa) == len(bb) == len(refs) == len(known)
         and all(type(x) is int and x >= 0 for x in refs), 'SAVED_COST_AXIS')
    groups = sorted(set(refs)); costs = []; witnesses = []
    for group in groups:
        indices = [i for i, ref in enumerate(refs) if ref == group]
        winner = max(indices, key=lambda i: (number(aa[i]), -known[i]))
        costs.append(aa[winner]); witnesses.append(known[winner])
    mx = max(range(len(known)), key=lambda i: (number(bb[i]), -known[i])) if known else None
    need(groups == grouped['observed_reference_group_indices']
         and costs == grouped['observed_group_max_a_binary64']
         and witnesses == grouped['observed_group_witness_source_query_indices']
         and (bb[mx] if mx is not None else hx(0)) == grouped['observed_max_b_binary64']
         and (known[mx] if mx is not None else None) == grouped['observed_max_b_witness_source_query_index'],
         'GROUPED_POOLING_FROM_ATOM_COST_DRIFT')
    expected = {'observation_status': grouped['observation_status'], 'missing_count': grouped['missing_count'],
        'observed_count': len(known), 'bound_denominator': len(domain),
        'exact_energy_or_lower_bound': rational(query_bound(aa, bb, len(domain))),
        'bound_is_observed_score': False, 'arithmetic': 'EXACT_SCALAR_ON_SAVED_FP64_ATOM_COSTS',
        'counterfactual_score_produced': False}
    need(cell['common_query'] == expected, 'COMMON_QUERY_EXACT_BOUND_DRIFT')


def synthetic_e0():
    import torch
    torch.set_num_threads(1)
    inherited = f.synthetic_e0()
    cases = pareto = 0
    for n in (1, 2, 3):
        for m in range(n + 1):
            for observed in itertools.product(range(3), repeat=2*(n-m)):
                aa, bb = observed[:n-m], observed[n-m:]
                lower = query_bound([hx(x) for x in aa], [hx(x) for x in bb], n)
                for missing in itertools.product(range(3), repeat=2*m):
                    full_a, full_b = aa + missing[:m], bb + missing[m:]
                    need(Fraction(sum(full_a), n) + max(full_b) >= lower, 'QUERY_COMPLETION_BOUND_FAILED')
                    cases += 1
        for ta, tb, da, db in itertools.product(itertools.product(range(2), repeat=n), repeat=4):
            wa = tuple(x+y for x,y in zip(ta,da)); wb = tuple(x+y for x,y in zip(tb,db))
            need(Fraction(sum(ta), n)+max(tb) <= Fraction(sum(wa), n)+max(wb), 'PARETO_MONOTONICITY_FAILED')
            pareto += 1
    ta = [Fraction(1,10)]*3+[Fraction(9,10)]; wa = [Fraction(3,25)]*3+[Fraction(19,20)]
    grouped_t = (ta[0]+ta[-1])/2 + max(ta)
    grouped_w = sum(wa)/4 + max(wa)
    query_t = sum(ta)/4 + max(ta)
    need(all(x<y for x,y in zip(ta,wa)) and grouped_t > grouped_w and query_t < grouped_w,
         'DIFFERENT_REFERENCE_PARTITIONS_COUNTEREXAMPLE')
    # A strict improvement confined to a nonmaximal b coordinate can leave E equal.
    need(query_bound([hx(1),hx(1)],[hx(.1),hx(.9)],2)
         ==query_bound([hx(1),hx(1)],[hx(.2),hx(.9)],2),
         'NONMAX_B_STRICT_COORDINATE_MUST_NOT_IMPLY_STRICT_ENERGY')
    core = module({'path':'src/rc_aslo_xf/reference_conditioned_coherent_residual_p_only_v8.py',
                   'sha256':b.PINS['core']}, 'rc_common_query_e0_core')
    residual = torch.zeros((4,128),dtype=torch.float64); residual[:,0] = torch.tensor([1.,1.,2.,3.])
    sources = torch.tensor([8,2,7,3]); refs = torch.tensor([0,0,1,1]); weights = torch.ones(256,dtype=torch.float64)
    axis = {'all':dict(zip(sources.tolist(),range(4))), 'valid':dict(zip(sources.tolist(),range(4))),
            'causes':{s:{'forward_out_of_bounds':False} for s in sources.tolist()}}
    domain = sources.tolist(); full, witness = common_cell(core,residual,refs,sources,domain,weights,axis)
    verify_cell(full,domain)
    perm = torch.tensor([3,1,0,2]); perm_axis = {'all':dict(zip(sources[perm].tolist(),range(4))),
        'valid':dict(zip(sources[perm].tolist(),range(4))), 'causes':axis['causes']}
    need(common_cell(core,residual[perm],refs[perm],sources[perm],domain,weights,perm_axis)==(full,witness), 'SOURCE_PERMUTATION')
    partial_axis = {'all':axis['all'], 'valid':{8:0,2:1},
        'causes':{s:{'forward_out_of_bounds':s in (7,3)} for s in domain}}
    partial,_ = common_cell(core,residual,refs,sources,domain,weights,partial_axis); verify_cell(partial,domain)
    poisoned = residual.clone(); poisoned[2:]=float('nan')
    need(common_cell(core,poisoned,refs,sources,domain,weights,partial_axis)[0]==partial,'INVALID_NAN_PLACEHOLDER_READ')
    empty_axis = {'all':axis['all'],'valid':{},'causes':{s:{'forward_out_of_bounds':True} for s in domain}}
    absent,_ = common_cell(core,poisoned,refs,sources,domain,weights,empty_axis); verify_cell(absent,domain)
    need(absent['common_query']['observation_status']=='UNKNOWN_PARTIAL'
         and frac(absent['common_query']['exact_energy_or_lower_bound'])==0,'UNKNOWN_NOT_ZERO_OBSERVATION')
    need(f.classify([full['common_query'],absent['common_query']],0)['status']=='UNRESOLVED', 'UNKNOWN_DISCARDED')
    need(f.classify([full['common_query'],full['common_query']],0)['status']=='REFUTED', 'TIE_IS_NOT_STRICT_CERTIFICATE')
    try: f.classify([absent['common_query'],full['common_query']],0)
    except RuntimeError: pass
    else: raise AssertionError('UNKNOWN_OWNER_WRONGLY_TREATED_AS_AVAILABLE_HYPOTHESIS')
    for field in ('observed_a_binary64','observed_source_query_indices'):
        bad=json.loads(b.encode(full)); bad[field][0] = hx(9) if field=='observed_a_binary64' else 999
        try: verify_cell(bad,domain)
        except RuntimeError: pass
        else: raise AssertionError('TAMPER_ACCEPTED:'+field)
    bad=json.loads(b.encode(full)); bad['common_query']['exact_energy_or_lower_bound']=rational(Fraction(99))
    try: verify_cell(bad,domain)
    except RuntimeError: pass
    else: raise AssertionError('COMMON_QUERY_BOUND_TAMPER_ACCEPTED')
    return {'status':'RC_V8_COMMON_QUERY_MEASURE_SYNTHETIC_E0_PASS',
        'query_completion_cases':cases,'pareto_cases':pareto,'parent_e0':inherited,
        'counterexample':{'target_grouped':rational(grouped_t),'wrong_grouped':rational(grouped_w),
                          'target_common_query':rational(query_t),'wrong_common_query':rational(grouped_w)},
        'checks':{'componentwise_pareto_consistency':True,'candidate_group_partition_reversal':True,
            'exhaustive_fixed_query_denominator_missing_bounds':True,'source_permutation':True,
            'canonical_physical_atom_ties':True,'nan_invalid_placeholders_unread':True,
            'unknown_not_imputed_or_removed':True,'saved_atom_cost_and_axis_tamper_rejected':True,
            'strict_nonmax_b_coordinate_can_leave_equal_energy':True,
            'unknown_owner_not_available_H1_and_empty_family_has_no_certificate':True,
            'common_query_bound_tamper_rejected':True},
        'natural_value_reads':0,'scientific_GO_or_NO_GO':None}


def freeze():
    need(not AUTH.exists(),'AUTHORITY_EXISTS')
    parent=f.load_authority(); files=dict(parent['sources'])
    for name,(path,pin) in EXTRA.items():
        need(sha(path)==pin,'FULL_FAMILY_SOURCE_DRIFT:'+name); files[name]=binding(path)
    files.update(program=binding(Path(__file__)),plan=binding(PLAN),launcher=binding(LAUNCH),
        full_family_authority=binding(f.AUTH),full_family_seal=binding(f.OUT/'prejoin_seal.json'))
    receipt=read(checked(files['full_family_validation']))
    need(receipt['status']=='RC_V8_FULL_FAMILY_COMPLETION_CERTIFICATE_ARTIFACT_VALIDATION_PASS'
         and receipt['result_sha256']==EXTRA['full_family_result'][1] and all(receipt['checks'].values()),'PARENT_NOT_VALIDATED')
    e0=ROOT/('results/'+PREFIX+'_e0/result.json'); atomic(e0,synthetic_e0()); files['e0']=binding(e0)
    a={'status':'RC_V8_COMMON_QUERY_MEASURE_DIAGNOSTIC_AUTHORIZED','sources':files,
       **{k:parent[k] for k in ('postjoin','input_manifest_sha256','parent_contract_sha256','order_sha256')},
       'output_rel':str(OUT.relative_to(ROOT)),'query_count':32,'candidate_count':128,'expected_component_count':2691,
       'measure':'SUM_OBSERVED_A_OVER_FIXED_FULL_QUERY_SUPPORT_SIZE_PLUS_MAX_OBSERVED_B',
       'unknown_policy':'NONNEGATIVE_HYPOTHETICAL_COMPLETION_BOUND_NOT_ZERO_IMPUTATION',
       'comparison_arithmetic':'EXACT_SCALAR_ON_SAVED_FP64_ATOM_COSTS_NOT_RUNTIME_FP64_MODEL',
       'training_authorized':False,'candidate_ranking_authorized':False,'formal_panel_authorized':False,
       'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}
    atomic(AUTH,a); print(json.dumps({'status':a['status'],'authority_sha256':sha(AUTH)}),flush=True)


def load_authority():
    a=read(AUTH); need(a['status']=='RC_V8_COMMON_QUERY_MEASURE_DIAGNOSTIC_AUTHORIZED'
                       and a['output_rel']==str(OUT.relative_to(ROOT)),'AUTHORITY_SCOPE')
    for v in a['sources'].values(): checked(v)
    for name,pin in b.PINS.items(): need(a['sources'][name]['sha256']==pin,'ORIGINAL_V8_PIN_DRIFT')
    for name,(_,pin) in EXTRA.items(): need(a['sources'][name]['sha256']==pin,'FULL_FAMILY_PIN_DRIFT')
    parent=read(checked(a['sources']['full_family_authority']))
    for k in ('postjoin','input_manifest_sha256','parent_contract_sha256','order_sha256'):
        need(a[k]==parent[k],'PARENT_SCOPE_DRIFT:'+k)
    need(a['sources']['program']==binding(Path(__file__)) and a['sources']['plan']==binding(PLAN)
         and a['sources']['launcher']==binding(LAUNCH),'PROGRAM_BINDING')
    need(a['query_count']==32 and a['candidate_count']==128 and a['expected_component_count']==2691
         and a['training_authorized'] is False and a['candidate_ranking_authorized'] is False
         and a['formal_panel_authorized'] is False and a['automatic_stage_advance'] is False
         and a['scientific_GO_or_NO_GO'] is None,'NO_AUTOMATIC_STAGE_AUTHORIZATION')
    return a


def parent_queries(a):
    seal=read(checked(a['sources']['full_family_seal']))
    need(seal['authority_sha256']==a['sources']['full_family_authority']['sha256']
         and len(seal['records'])==32 and seal['component_count']==2691,'PARENT_SEAL')
    paths={}
    for item in seal['records']:
        key=item['query_resource_key']; need(key not in paths and item['path']=='prejoin/'+key+'.json','PARENT_QUERY_AXIS')
        path=f.OUT/item['path']; need(sha(path)==item['sha256'],'PARENT_QUERY_HASH'); paths[key]=path
    return paths


def run():
    global STOP
    import torch
    began=time.monotonic(); a=load_authority(); core,weights=f.load_head(a)
    need(not OUT.exists(),'OUTPUT_EXISTS_NO_RESUME')
    adapter=module(a['sources']['adapter'],'rc_common_query_adapter')
    bundle=adapter.InputBundle(ROOT,expected_manifest_sha256=a['input_manifest_sha256'])
    parents=parent_queries(a); records=[]; total=0; cell_total=0
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop)
    def guard():
        if STOP or time.monotonic()-began>3000: raise TimeoutError('INTERNAL_3000S_OR_SIGNAL')
    try:
        with torch.no_grad():
            for ep in bundle.iter_queries():
                guard(); key=ep['query_resource_key']; parent=read(parents[key]); keys=list(ep['candidate_keys'])
                need(keys==parent['candidate_keys']==sorted(set(keys)) and len(keys)==128,'FULL_C128_AXIS')
                banks=ep['banks_by_control']['REAL']
                need(all(bank.reference_resource_key==keys[i] for i,bank in enumerate(banks)),'REAL_REFERENCE_BINDING')
                axes=[f.prepared_bank(bank,ep['valid']['REAL'][i]) for i,bank in enumerate(banks)]
                owners=[]
                for ci,old_owner in enumerate(parent['owners']):
                    components=ep['families']['REAL'][ci]['components']; rows=[]
                    need(old_owner['candidate_resource_key']==keys[ci] and len(components)==old_owner['component_count'],'OWNER_AXIS')
                    for ordinal,component in enumerate(components):
                        guard(); old_row=old_owner['components'][ordinal]; support=core._support(component)
                        need(support==old_row['support'] and logical(component)==old_row['component_sha256'],'ALL_LEGAL_REGION_AXIS')
                        domain=support['source_query_indices']
                        need(len(domain)==len(set(domain))>=4 and [axes[ci]['valid'].get(s) for s in domain]==support['atom_indices'],'LEGAL_OWNER_SOURCE_AXIS')
                        cells=[]; own_witness=None
                        for cj,bank in enumerate(banks):
                            if cj%16==0: guard()
                            cell,witness=common_cell(core,ep['residual_squared']['REAL'][cj],bank.source_reference_indices,
                                bank.source_query_indices,domain,weights,axes[cj])
                            need(cell['grouped']==old_row['cells'][cj],'ALL344448_PARENT_CELL_BIT_REPLAY_DRIFT')
                            cells.append(cell); cell_total+=1
                            if cj==ci: own_witness=witness
                        need(own_witness==old_row['owner_witness'],'ORIGINAL_PHYSICAL_OWNER_WITNESS_DRIFT')
                        need(f.classify([c['grouped'] for c in cells],ci)==old_row['certificate'],'PARENT_CERTIFICATE_DRIFT')
                        row={**old_row,'cells':cells,'common_query_certificate':f.classify([c['common_query'] for c in cells],ci)}
                        rows.append(row); total+=1
                    owners.append({**old_owner,'components':rows})
                q={'status':'RC_V8_COMMON_QUERY_MEASURE_QUERY_PREJOIN_SEALED','query_resource_key':key,'candidate_keys':keys,
                    'authority_sha256':sha(AUTH),'parent_query_sha256':sha(parents[key]),
                    'source_receipt_sha256':logical(ep['source_receipt']),'weight_sha256':core.tensor_sha(weights),
                    'family_axis_sha256':logical(ep['families']),'owners':owners,
                    'target_reads':0,'training_updates':0,'candidate_ranking_produced':False}
                dest=OUT/'prejoin'/(key+'.json'); atomic(dest,q)
                records.append({'query_resource_key':key,'path':'prejoin/'+key+'.json','sha256':sha(dest)})
                print(json.dumps({'event':'COMMON_QUERY_MEASURE_QUERY_SEALED','queries':len(records),'components':total}),flush=True)
        need(len(records)==32 and total==2691 and cell_total==344448,'FULL_AXIS_NOT_CLOSED')
        atomic(OUT/'input_closure.json',bundle.prejoin_receipt())
        seal={'status':'RC_V8_COMMON_QUERY_MEASURE_FULL32_C128_PREJOIN_CLOSED','authority_sha256':sha(AUTH),
            'records':sorted(records,key=lambda x:x['query_resource_key']),'component_count':total,'cross_cell_count':cell_total,
            'input_closure_sha256':sha(OUT/'input_closure.json'),
            'label_read_attempts_before_closure':bundle.barrier.blocked_read_attempts}
        atomic(OUT/'prejoin_seal.json',seal); atomic(OUT/'result.json',summarize(a,seal))
        print(json.dumps({'status':'RC_V8_COMMON_QUERY_MEASURE_DIAGNOSTIC_COMPLETE','scientific_GO_or_NO_GO':None}),flush=True)
        return 0
    except TimeoutError:
        receipt={'status':'INCOMPLETE_TIME_LIMIT_OR_SIGNAL','sealed_queries':len(records),'processed_component_count':total,
            'authority_sha256':sha(AUTH),'resumable':False,'scientific_GO_or_NO_GO':None}
        atomic(OUT/'incomplete.json',receipt); print(json.dumps(receipt),flush=True); return 75


def pair_accounting(target,wrong,domain):
    need(target['grouped']['missing_count']==wrong['grouped']['missing_count']==0,'ORIGINAL_WRONG_CERT_TARGET_NOT_COMPLETE')
    need(target['observed_source_query_indices']==wrong['observed_source_query_indices']==domain,'PAIR_COMMON_SOURCE_AXIS')
    ta,tb,wa,wb=([Fraction.from_float(number(x)) for x in c[k]]
        for c,k in ((target,'observed_a_binary64'),(target,'observed_b_binary64'),
                    (wrong,'observed_a_binary64'),(wrong,'observed_b_binary64')))
    n=len(domain); ca=sum(x<=y for x,y in zip(ta,wa)); cb=sum(x<=y for x,y in zip(tb,wb))
    target_pareto=ca==cb==n; strict=any(x<y for x,y in zip(ta+tb,wa+wb))
    return {'support_size':n,'target_mean_a_exact':rational(sum(ta)/n),'wrong_mean_a_exact':rational(sum(wa)/n),
        'target_max_b_exact':rational(max(tb)),'wrong_max_b_exact':rational(max(wb)),
        'target_a_le_wrong_a_count':ca,'target_b_le_wrong_b_count':cb,
        'target_joint_coordinatewise_pareto_le':target_pareto,'target_joint_pareto_with_any_strict_coordinate':target_pareto and strict,
        'wrong_joint_coordinatewise_pareto_le':all(x<=y for x,y in zip(wa+wb,ta+tb)),
        'target_common_query_energy_exact':target['common_query']['exact_energy_or_lower_bound'],
        'wrong_common_query_energy_exact':wrong['common_query']['exact_energy_or_lower_bound'],
        'wrong_common_query_energy_strictly_lower_than_actual_target':
            frac(wrong['common_query']['exact_energy_or_lower_bound'])<frac(target['common_query']['exact_energy_or_lower_bound']),
        'target_group_count':target['grouped']['observed_group_count'],'wrong_group_count':wrong['grouped']['observed_group_count']}


def summarize(a,seal):
    need(len(seal['records'])==32 and seal['component_count']==2691 and seal['cross_cell_count']==344448
         and seal['label_read_attempts_before_closure']==0 and read(OUT/'prejoin_seal.json')==seal,'POSTJOIN_BEFORE_FULL_SEAL')
    joined=read(checked(a['postjoin']['authority'])); validation=read(checked(a['postjoin']['validation']))
    need(joined['record_count']==32 and joined['target_insertion_count']==0 and joined['fold_id']==1,'POSTJOIN_SCOPE')
    need(validation['authority_sha256']==a['postjoin']['authority']['sha256']
         and validation['checks'] and all(v is True for v in validation['checks'].values()),'POSTJOIN_VALIDATION')
    roles={r['query_resource_key']:r for r in joined['records']}
    need(set(roles)=={r['query_resource_key'] for r in seal['records']},'POSTJOIN_FULL32')
    counters={'grouped':Counter(),'common_query':Counter()}; records=[]; transitions=[]; target_transitions=[]
    for item in seal['records']:
        path=OUT/item['path']; need(sha(path)==item['sha256'],'QUERY_HASH'); q=read(path); keys=q['candidate_keys']
        targets=set(roles[q['query_resource_key']]['target_candidate_resource_keys'])
        need(len(targets)==1 and targets.issubset(keys),'OPENED_SINGLE_TARGET_AXIS'); ti=keys.index(next(iter(targets)))
        modes={}
        for mode,field in (('grouped','certificate'),('common_query','common_query_certificate')):
            counts=counters[mode]; certified=[]; possible=[]; target_states=[]
            for owner in q['owners']:
                ck=owner['candidate_resource_key']; states=[r[field]['status'] for r in owner['components']]
                counts.update(states); counts['owner_count']+=1; counts['H0_owner_count']+=not states
                if 'CERTIFIED' in states: certified.append(ck)
                if any(s!='REFUTED' for s in states): possible.append(ck)
                if ck in targets: target_states=states
            target_cert=bool(targets.intersection(certified)); wrong=sorted(set(certified)-targets); other_possible=sorted(set(possible)-targets)
            counts['queries_target_any_certified_H']+=target_cert
            counts['queries_target_any_not_refuted_H']+=bool(targets.intersection(possible))
            counts['queries_any_wrong_candidate_certified']+=bool(wrong)
            counts['queries_unique_certified_candidate_is_target']+=target_cert and len(certified)==1
            counts['queries_strict_unique_target_all_wrong_regions_refuted_or_H0']+=target_cert and not other_possible
            counts['queries_target_H0']+=not target_states
            modes[mode]={'target_component_statuses':target_states,'target_any_certified_H':target_cert,
                'certified_candidate_keys':certified,'wrong_certified_candidate_keys':wrong,
                'wrong_candidate_keys_with_any_not_refuted_region':other_possible,
                'strict_unique_target_all_wrong_regions_refuted_or_H0':target_cert and not other_possible}
        for ci,owner in enumerate(q['owners']):
            for row in owner['components']:
                if row['certificate']['status']!='CERTIFIED': continue
                record={'query_resource_key':q['query_resource_key'],'owner_candidate_resource_key':keys[ci],
                    'component_ordinal':row['component_ordinal'],'support_sha256':row['support_sha256'],
                    'original_grouped_status':'CERTIFIED','common_query_status':row['common_query_certificate']['status'],
                    'common_query_certificate':row['common_query_certificate']}
                if keys[ci] in targets: target_transitions.append(record)
                else:
                    record['actual_target_candidate_resource_key']=keys[ti]
                    record['atom_cost_accounting']=pair_accounting(row['cells'][ti],row['cells'][ci],row['support']['source_query_indices'])
                    transitions.append(record)
        records.append({'query_resource_key':q['query_resource_key'],'supergroup_hash':roles[q['query_resource_key']]['supergroup_hash'],
                        'target_candidate_keys':sorted(targets),'measures':modes})
    for counter in counters.values():
        need(counter['owner_count']==4096 and counter['H0_owner_count']==2424
             and sum(counter[s] for s in ('CERTIFIED','REFUTED','UNRESOLVED'))==2691,'ALL_REGION_COUNT')
    parent=read(checked(a['sources']['full_family_result']))
    for key,value in counters['grouped'].items(): need(parent['counts'][key]==value,'ORIGINAL_GROUPED_COUNTS_DRIFT:'+key)
    need(len(transitions)==21 and len(target_transitions)==22,'ORIGINAL43_CERTIFICATE_POPULATION')
    return {'status':'RC_V8_COMMON_QUERY_MEASURE_DIAGNOSTIC_COMPLETE',
        'claim_level':'POSTHOC_FROZEN_WEIGHT_COMMON_MEASURE_ORACLE_CERTIFICATE_AND_ATOM_ACCOUNTING_DIAGNOSTIC',
        'authority_sha256':sha(AUTH),'prejoin_seal_sha256':sha(OUT/'prejoin_seal.json'),
        'query_count':32,'candidate_count':128,'component_count':2691,'cross_cell_count':344448,
        'counts':{k:dict(v) for k,v in counters.items()},'records':records,
        'original_21_wrong_certified_region_transitions':transitions,
        'original_22_target_certified_region_transitions':target_transitions,
        'wrong_certificate_transition_counts':dict(Counter(r['common_query_status'] for r in transitions)),
        'target_certificate_transition_counts':dict(Counter(r['common_query_status'] for r in target_transitions)),
        'training_updates':0,'candidate_ranking_produced':False,'formal_panel_consumed':False,
        'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None,
        'interpretation':['Every original legal REAL region and all C128 candidates are retained.',
            'The counterfactual changes only the measure of observed scalar a costs to the uniform source query-cell measure.',
            'Exact fractions apply to saved FP64 atom costs, not arbitrary runtime FP64 completion reductions.',
            'Missing assignments remain UNKNOWN; their nonnegative-cost lower bound is not an observed or imputed score.',
            'A certified target region is oracle evidence capacity, not accuracy, a rescue, or a target-free selection rule.',
            'No AP, Q, CB, or PC controls are computed; no scientific GO can be issued.',
            'If wrong certificates retain lower atom costs, subsequent mechanism work must address identity necessary extent or readout; no automatic trial follows.']}


def validate():
    """Separate-process saved artifact replay; no independent raw recomputation claim."""
    a=load_authority(); core,weights=f.load_head(a); expected_weight=core.tensor_sha(weights); parents=parent_queries(a)
    seal=read(OUT/'prejoin_seal.json'); closure=read(OUT/'input_closure.json')
    need(seal['status']=='RC_V8_COMMON_QUERY_MEASURE_FULL32_C128_PREJOIN_CLOSED' and seal['authority_sha256']==sha(AUTH)
         and len(seal['records'])==32 and seal['component_count']==2691 and seal['cross_cell_count']==344448
         and seal['label_read_attempts_before_closure']==0,'SEALED_FULL_AXIS')
    need(sha(OUT/'input_closure.json')==seal['input_closure_sha256'] and closure['prejoin_label_reads']==0
         and closure['input_manifest_sha256']==a['input_manifest_sha256'] and len(closure['records'])==32,'INPUT_CLOSURE')
    receipts={r['query_resource_key']:r for r in closure['records']}; seen=set(); total=cells=0
    for item in seal['records']:
        key=item['query_resource_key']; need(key not in seen and item['path']=='prejoin/'+key+'.json','QUERY_AXIS'); seen.add(key)
        path=OUT/item['path']; need(sha(path)==item['sha256'],'QUERY_HASH'); q=read(path); parent=read(parents[key])
        need(q['authority_sha256']==sha(AUTH) and q['parent_query_sha256']==sha(parents[key])
             and q['query_resource_key']==key and q['candidate_keys']==parent['candidate_keys']
             and q['source_receipt_sha256']==logical(receipts[key])
             and q['family_axis_sha256']==receipts[key]['family_axis_sha256']
             and q['weight_sha256']==expected_weight and q['target_reads']==0
             and q['training_updates']==0 and q['candidate_ranking_produced'] is False,'QUERY_LINEAGE_SCOPE')
        need(len(q['owners'])==len(parent['owners'])==128,'OWNER_COUNT')
        for ci,(owner,old_owner) in enumerate(zip(q['owners'],parent['owners'])):
            need({k:v for k,v in owner.items() if k!='components'}=={k:v for k,v in old_owner.items() if k!='components'}
                 and len(owner['components'])==len(old_owner['components']),'OWNER_H0_COMPLETE_AXIS')
            for row,old_row in zip(owner['components'],old_owner['components']):
                need({k:v for k,v in row.items() if k not in ('cells','common_query_certificate')}
                     =={k:v for k,v in old_row.items() if k!='cells'} and len(row['cells'])==128,'REGION_IMMUTABLE_AXIS')
                domain=row['support']['source_query_indices']
                for cell,old_cell in zip(row['cells'],old_row['cells']):
                    verify_cell(cell,domain); need(cell['grouped']==old_cell,'ALL_PARENT_CELL_REPLAY'); cells+=1
                need(row['certificate']==f.classify([c['grouped'] for c in row['cells']],ci)
                     and row['common_query_certificate']==f.classify([c['common_query'] for c in row['cells']],ci),'CERTIFICATE_REPLAY')
                total+=1
    need(seen==set(parents)==set(receipts) and total==2691 and cells==344448,'FULL_ARTIFACT_CLOSURE')
    need(read(OUT/'result.json')==summarize(a,seal),'POSTJOIN_SUMMARY_REPLAY')
    receipt={'status':'RC_V8_COMMON_QUERY_MEASURE_ARTIFACT_VALIDATION_PASS',
        'claim_level':'SOURCE_HASH_FULL_AXIS_ALL_SAVED_ATOM_COST_MEASURE_AND_PARENT_CELL_REPLAY_NO_RAW_INDEPENDENT_REPLAY',
        'authority_sha256':sha(AUTH),'result_sha256':sha(OUT/'result.json'),
        'checks':{'full32_C128_all2691_legal_regions':True,'all344448_original_grouped_cells_replayed':True,
            'saved_atom_costs_grouped_pooling_and_canonical_witness_replayed':True,
            'common_query_exact_sum_mean_max_and_missing_bound_replayed':True,
            'full_axis_certificate_states_replayed':True,'all21_wrong_and22_target_certificate_transitions':True,
            'atom_cost_pareto_and_actual_target_pair_accounting':True,'postjoin_after_full32_seal_only':True},
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
