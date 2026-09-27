#!/usr/bin/env python3
"""Frozen V8 assignment intervention on original selected supports; diagnostic only."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os
from fractions import Fraction
from pathlib import Path
import signal, sys, tempfile, time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PREFIX='rc_v8_assignment_intervention_v1'
AUTH=ROOT/('registry/'+PREFIX+'_authority_20260909.json')
PLAN=ROOT/'plan/RC_V8_ASSIGNMENT_INTERVENTION_V1_20260909.md'
LAUNCH=ROOT/('slurm/'+PREFIX+'_dev_cpuonly_59m.sbatch')
OUT=ROOT/('results/'+PREFIX)
PARENT_PROGRAM=ROOT/'programs/run_rc_v8_cross_support_diagnostic_v1.py'
PARENT_PROGRAM_SHA='8c6deb905cea73128efdbde206ab53a702d3548775b2ee209ad0d27f1e7fe3e3'
if hashlib.sha256(PARENT_PROGRAM.read_bytes()).hexdigest()!=PARENT_PROGRAM_SHA:raise RuntimeError('PINNED_HELPER_DRIFT')
spec=importlib.util.spec_from_file_location('rc_assignment_v8_helpers',PARENT_PROGRAM)
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
need,encode,logical,sha,read,binding,checked,module,atomic,hx,number,rational=(getattr(b,n) for n in ('need','encode','logical','sha','read','binding','checked','module','atomic','hx','number','rational'))
CONDITIONS=('HARD','WITHIN_R','ALL_R')
CONTRACT={'conditions':list(CONDITIONS),'support':'ORIGINAL_V8_REAL_SELECTED_H_NO_RESELECTION',
 'within_reference_axis':'SORTED_UNIQUE_ORIGINAL_SELECTED_H_SOURCE_REFERENCE_INDICES',
 'all_reference_axis':'SORTED_ALL_VALID_ORIGINAL_REFERENCE_INDICES',
 'assignment':'PER_QUERY_ARGMIN_SUM_CHANNELS_EXPLICIT_SQUARED_RESIDUAL_TIMES_W_MU_PLUS_W_MAX',
 'tie':'LOWEST_ORIGINAL_SOURCE_REFERENCE_INDEX','query_chunk':32,'reference_chunk':1024,
 'primary':'ORIGINAL_FROZEN_V8_COHERENT_GROUPED_ENERGY_AND_EXP_NEG_HALF_ENERGY',
 'secondary':'EXACT_FRACTION_SUM_SAVED_FP64_A_OVER_FIXED_QUERY_COUNT_PLUS_MAX_SAVED_FP64_B',
 'H0':'REMAIN_STRUCTURAL_H0_NO_APPEARANCE_FILL','training_updates':0,
 'region_reselection_count':0,'formal_panel_authorized':False,'automatic_stage_advance':False,
 'rank_claim':'FROZEN_SUPPORT_COUNTERFACTUAL_ONLY_NOT_A_NEW_RETRIEVAL_METHOD_OR_HYP_GO'}
STOP=False

def tensor_record(t):
    a=t.detach().cpu().contiguous()
    return {'dtype':str(a.dtype),'shape':list(a.shape),'sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest()}

def frac(r):return Fraction(int(r['numerator']),int(r['denominator']))

def exact_query_energy(aa,bb):
    need(len(aa)==len(bb)>0,'COMMON_QUERY_AXIS')
    return sum((Fraction.from_float(number(x)) for x in aa),Fraction())/len(aa)+max(Fraction.from_float(number(x)) for x in bb)

def admissible_assignment(q,reference,admissible,weights,*,query_chunk=32,reference_chunk=1024):
    """One fixed argmin; full scalar matrix retained, no masked/invalid token read."""
    import torch
    need(q.dtype==reference.dtype==weights.dtype==torch.float64 and q.ndim==reference.ndim==2
         and q.shape[1]==reference.shape[1]==128 and weights.shape==(256,),'ASSIGNMENT_DTYPE_AXIS')
    need(admissible.dtype==torch.long and admissible.ndim==1 and len(admissible)>0
         and torch.equal(admissible,torch.unique(admissible,sorted=True))
         and bool((admissible>=0).all()) and bool((admissible<len(reference)).all()),'ADMISSIBLE_REFERENCE_AXIS')
    need(bool(torch.isfinite(q).all()) and bool(torch.isfinite(reference[admissible]).all()),'VALID_TOKEN_NONFINITE')
    need(bool(torch.isfinite(weights).all()) and bool((weights>0).all()),'WEIGHTS_INVALID')
    need(query_chunk>0 and reference_chunk>0 and query_chunk*reference_chunk*128*8<=64*1024*1024,'PAIR_WORKSPACE_BOUND')
    costs=torch.empty((len(q),len(admissible)),dtype=torch.float64)
    channels=weights[:128]+weights[128:]
    for qi in range(0,len(q),query_chunk):
        for ri in range(0,len(admissible),reference_chunk):
            # No expanded dot-product identity: preserve the original explicit residual arithmetic.
            work=q[qi:qi+query_chunk,None,:]-reference[admissible[ri:ri+reference_chunk]][None,:,:]
            work.square_();work.mul_(channels)
            costs[qi:qi+query_chunk,ri:ri+reference_chunk]=work.sum(dim=-1)
    need(bool(torch.isfinite(costs).all()),'PAIR_COST_NONFINITE')
    chosen=admissible[costs.argmin(dim=1)]
    selected=(q-reference[chosen]).square()
    return selected,chosen,costs

def score_cell(core,residual,refs,sources,weights):
    score,witness=core.coherent_score(residual,refs,weights,sources)
    aa=(residual*weights[:128]).sum(1);bb=(residual*weights[128:]).sum(1)
    ahex,bhex=[hx(x) for x in aa],[hx(x) for x in bb]
    common=exact_query_energy(ahex,bhex)
    grouped=sum((Fraction.from_float(number(x)) for x in witness['group_mean_cost_binary64']),Fraction())/len(witness['group_mean_cost_binary64'])+Fraction.from_float(number(witness['global_max_cost_binary64']))
    return {'state':'H1','score_binary64':hx(score),'energy_binary64':witness['coherent_energy_binary64'],
            'pooling_witness':witness,'chosen_source_reference_indices':refs.tolist(),
            'a_binary64':ahex,'b_binary64':bhex,'common_query_energy_exact':rational(common),
            'grouped_energy_exact_from_saved_atom_costs':rational(grouped),
            'grouping_excess_over_common_query_exact':rational(grouped-common),
            'selected_residual':tensor_record(residual)}

def h0_cell():return {'state':'H0','score_binary64':hx(0),'energy_binary64':None,'pooling_witness':None,
    'chosen_source_reference_indices':[],'a_binary64':[],'b_binary64':[],'common_query_energy_exact':None,
    'grouped_energy_exact_from_saved_atom_costs':None,'grouping_excess_over_common_query_exact':None,'selected_residual':None,
    'admissible_source_reference_indices':[],'pair_cost_matrix':None,'argmin_column_vector':None}

def validate_cell(core,cell,residual,costs,domain,weights):
    import torch
    if cell['state']=='H0':
        need(cell==h0_cell() and residual is None and costs is None and not domain,'H0_APPEARANCE_FILL');return
    refs=torch.tensor(cell['chosen_source_reference_indices'],dtype=torch.long)
    sources=torch.tensor(domain,dtype=torch.long)
    expected=score_cell(core,residual,refs,sources,weights)
    for key,value in expected.items():need(cell[key]==value,'SELECTED_RESIDUAL_SCORE_REPLAY:'+key)
    allowed=cell['admissible_source_reference_indices']
    need(allowed==sorted(set(allowed)) and len(allowed)>0 and set(refs.tolist()).issubset(allowed),'SAVED_REFERENCE_DOMAIN')
    if costs is None:
        need(cell['pair_cost_matrix'] is None and cell['argmin_column_vector'] is None,'HARD_MATRIX_UNEXPECTED')
    else:
        need(costs.dtype==torch.float64 and costs.shape==(len(domain),len(allowed))
             and bool(torch.isfinite(costs).all()) and bool((costs>=0).all()),'SAVED_COST_MATRIX_AXIS')
        need(cell['pair_cost_matrix']==tensor_record(costs),'SAVED_COST_MATRIX_HASH')
        columns=costs.argmin(1)
        need(cell['argmin_column_vector']==tensor_record(columns)
             and torch.equal(refs,torch.tensor(allowed,dtype=torch.long)[columns]),'FULL_MATRIX_ARGMIN_REPLAY')
        channels=weights[:128]+weights[128:]
        need(torch.equal((residual*channels).sum(1),costs[torch.arange(len(domain)),columns]),'ARGMIN_SELECTED_RESIDUAL_COST_DRIFT')

def load_head(a):
    import torch
    core=module(a['sources']['core'],'rc_assignment_v8_core')
    runner=module(a['sources']['runner'],'rc_assignment_v8_checkpoint')
    cp=read(checked(a['sources']['checkpoint']))
    need(cp['schema']==runner.SCHEMA and cp['update_index']==512 and cp['contract_sha256']==a['parent_contract_sha256']
         and cp['order_sha256']==a['order_sha256'],'FROZEN_CHECKPOINT_SCOPE')
    heads=core.GroupedResidualHeads();heads.load_state_dict(runner.unpack(cp['model'],torch),strict=True);heads.eval()
    return core,heads.real.weights().detach()

def freeze():
    need(not AUTH.exists(),'AUTHORITY_EXISTS')
    need(sha(b.PARENT)==b.PINS['parent'],'PARENT_AUTHORITY_DRIFT');parent=read(b.PARENT)
    paths={'program':Path(__file__),'plan':PLAN,'launcher':LAUNCH,'helper':PARENT_PROGRAM,'parent':b.PARENT,
      'checkpoint':b.PARENT_OUT/'final_checkpoint.json','saved_scores':b.PARENT_OUT/'score_records.jsonl',
      'parent_result':b.PARENT_OUT/'result.json','parent_validation':ROOT/parent['validation_rel']/'result.json'}
    for name in ('core','runner','adapter'):paths[name]=checked(parent[name])
    for name,pin in b.PINS.items():need(sha(paths[name])==pin,'ORIGINAL_V8_PIN_DRIFT:'+name)
    validation=read(paths['parent_validation'])
    need(validation['status']=='RC_COHERENT_RESIDUAL_V8_DEVELOPMENT_INDEPENDENT_VALIDATION_PASS'
         and validation['result_sha256']==b.PINS['parent_result'] and validation['gate_go'] is False
         and validation['checks'] and all(x is True for x in validation['checks'].values()),'V8_PARENT_VALIDATION')
    e0path=ROOT/('results/'+PREFIX+'_e0/result.json');atomic(e0path,synthetic_e0());paths['e0']=e0path
    authority={'status':'RC_V8_ASSIGNMENT_INTERVENTION_AUTHORIZED','sources':{k:binding(v) for k,v in paths.items()},
      'postjoin':parent['postjoin'],'input_manifest_sha256':parent['input_manifest_sha256'],
      'parent_contract_sha256':parent['contract']['sha256'],'order_sha256':parent['training']['order_sha256'],
      'output_rel':str(OUT.relative_to(ROOT)),'query_count':32,'candidate_count':128,'expected_H1':1672,'expected_H0':2424,
      'contract':CONTRACT,'scientific_GO_or_NO_GO':None}
    atomic(AUTH,authority);print(json.dumps({'status':authority['status'],'authority_sha256':sha(AUTH)}),flush=True)

def load_authority():
    a=read(AUTH)
    need(a['status']=='RC_V8_ASSIGNMENT_INTERVENTION_AUTHORIZED' and a['contract']==CONTRACT
         and a['output_rel']==str(OUT.relative_to(ROOT)) and a['query_count']==32 and a['candidate_count']==128
         and a['expected_H1']==1672 and a['expected_H0']==2424 and a['scientific_GO_or_NO_GO'] is None,'AUTHORITY_SCOPE')
    for v in a['sources'].values():checked(v)
    for k,v in b.PINS.items():need(a['sources'][k]['sha256']==v,'FROZEN_V8_PIN_DRIFT')
    need(a['sources']['helper']['sha256']==PARENT_PROGRAM_SHA and a['sources']['program']==binding(Path(__file__))
         and a['sources']['plan']==binding(PLAN) and a['sources']['launcher']==binding(LAUNCH),'PROGRAM_BINDING')
    parent=read(checked(a['sources']['parent']))
    for k in ('postjoin','input_manifest_sha256'):need(a[k]==parent[k],'PARENT_SCOPE_DRIFT:'+k)
    need(a['parent_contract_sha256']==parent['contract']['sha256']
         and a['order_sha256']==parent['training']['order_sha256'],'PARENT_TRAINING_SCOPE')
    e0=read(checked(a['sources']['e0']))
    need(e0['status']=='RC_V8_ASSIGNMENT_INTERVENTION_SYNTHETIC_E0_PASS' and e0['checks']
         and all(v is True for v in e0['checks'].values()),'E0_NOT_CLOSED')
    return a

def saved_queries(a):
    rows=[json.loads(x) for x in checked(a['sources']['saved_scores']).read_text().splitlines()]
    result=read(checked(a['sources']['parent_result']))
    need(len(rows)==32 and len({r['query_resource_key'] for r in rows})==32
         and logical(rows)==result['score_records_sha256'],'FROZEN_SCORE_BINDING')
    return {r['query_resource_key']:r for r in rows}

def save_arrays(path,arrays):
    import numpy as np
    path.parent.mkdir(parents=True,exist_ok=True);need(not path.exists(),'ARRAY_OUTPUT_EXISTS')
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name,delete=False) as stream:
        temporary=Path(stream.name);np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    try:os.chmod(temporary,0o444);os.link(temporary,path)
    finally:temporary.unlink()

def run():
    global STOP
    import torch
    began=time.monotonic();a=load_authority();core,weights=load_head(a);saved=saved_queries(a)
    need(not OUT.exists(),'OUTPUT_EXISTS_NO_RESUME')
    adapter=module(a['sources']['adapter'],'rc_assignment_v8_inputs')
    bundle=adapter.InputBundle(ROOT,expected_manifest_sha256=a['input_manifest_sha256'])
    records=[];h1=h0=0
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    def guard():
        if STOP or time.monotonic()-began>3000:raise TimeoutError('INTERNAL_3000S_OR_SIGNAL')
    try:
        with torch.no_grad():
            for ep in bundle.iter_queries():
                guard();key=ep['query_resource_key'];keys=list(ep['candidate_keys']);old=saved[key]
                need(keys==old['candidate_keys']==sorted(set(keys)) and len(keys)==128,'FULL_C128_AXIS')
                owners=[];arrays={};vectors={k:[] for k in CONDITIONS}
                for ci,candidate in enumerate(keys):
                    guard();decision=old['arms']['REAL']['decisions'][ci];support=decision['selected_H'];bank=ep['banks_by_control']['REAL'][ci]
                    need(decision['candidate_resource_key']==candidate and decision['reference_resource_key']==candidate
                         and bank.reference_resource_key==candidate and decision['candidate_score_binary64']==old['arms']['REAL']['scores_binary64'][ci],
                         'FROZEN_CANDIDATE_REFERENCE_BINDING')
                    owner={'candidate_resource_key':candidate,'reference_resource_key':candidate,'source_H':support,
                           'source_H_sha256':logical(support),'original_map_seed_ordinal':decision['map_seed_ordinal'],
                           'source_query_indices':[],'conditions':{}}
                    components=ep['families']['REAL'][ci]['components']
                    if support is None:
                        need(not components and decision['state']=='H0' and decision['map_seed_ordinal'] is None
                             and number(decision['candidate_score_binary64'])==0,'H0_DRIFT')
                        owner['conditions']={mode:h0_cell() for mode in CONDITIONS};h0+=1
                    else:
                        ordinal=decision['map_seed_ordinal'];need(type(ordinal) is int and 0<=ordinal<len(components),'MAP_ORDINAL')
                        need({k:v for k,v in support.items() if k!='pooling_witness'}==core._support(components[ordinal]),'OLD_H_GENERATION_DRIFT')
                        domain=support['source_query_indices'];ids=torch.tensor(support['atom_indices'],dtype=torch.long)
                        need(len(domain)==len(set(domain))>=4 and bool(ep['valid']['REAL'][ci,ids].all())
                             and bank.source_query_indices[ids].tolist()==domain,'ORIGINAL_QUERY_SUPPORT_AXIS')
                        refs=bank.source_reference_indices[ids];need(refs.tolist()==support['source_reference_indices'],'OLD_REFERENCE_SUPPORT_AXIS')
                        q=ep['qtokens'][torch.tensor(domain,dtype=torch.long)];reference=ep['reference_tokens']['REAL'][ci]
                        valid=ep['reference_valid']['REAL'][ci];need(bool(valid[refs].all()),'OLD_REFERENCE_INVALID')
                        allowed={'HARD':torch.unique(refs,sorted=True),'WITHIN_R':torch.unique(refs,sorted=True),
                                 'ALL_R':torch.nonzero(valid,as_tuple=False).flatten()}
                        need(set(allowed['WITHIN_R'].tolist()).issubset(allowed['ALL_R'].tolist()),'NESTED_REFERENCE_SETS')
                        hard=ep['residual_squared']['REAL'][ci,ids]
                        need(torch.equal(hard,(q-reference[refs]).square()),'ORIGINAL_RAW_SELECTED_RESIDUAL_BIT_REPLAY')
                        owner['source_query_indices']=domain;owner['original_source_reference_indices']=refs.tolist()
                        owner['valid_reference_indices']=allowed['ALL_R'].tolist();owner['query_tokens']=tensor_record(q)
                        owner['valid_reference_tokens']=tensor_record(reference[allowed['ALL_R']])
                        for mode in CONDITIONS:
                            if mode=='HARD':residual,chosen,costs=hard,refs,None
                            else:residual,chosen,costs=admissible_assignment(q,reference,allowed[mode],weights)
                            cell=score_cell(core,residual,chosen,torch.tensor(domain,dtype=torch.long),weights)
                            cell.update(admissible_source_reference_indices=allowed[mode].tolist(),
                                pair_cost_matrix=tensor_record(costs) if costs is not None else None,
                                argmin_column_vector=tensor_record(costs.argmin(1)) if costs is not None else None)
                            validate_cell(core,cell,residual,costs,domain,weights)
                            if mode=='HARD':need(cell['score_binary64']==decision['candidate_score_binary64']
                                and cell['pooling_witness']==support['pooling_witness'],'HARD_K_ENERGY_WITNESS_BIT_DRIFT')
                            arrays[f'c{ci}_{mode}_residual']=residual.detach().numpy()
                            if costs is not None:arrays[f'c{ci}_{mode}_costs']=costs.detach().numpy()
                            owner['conditions'][mode]=cell
                        h1+=1
                    for mode in CONDITIONS:vectors[mode].append(owner['conditions'][mode]['score_binary64'])
                    owners.append(owner)
                need(vectors['HARD']==old['arms']['REAL']['scores_binary64'],'FULL_C128_HARD_VECTOR_DRIFT')
                array_path=OUT/'arrays'/(key+'.npz');save_arrays(array_path,arrays)
                query={'status':'RC_V8_ASSIGNMENT_INTERVENTION_QUERY_PREJOIN_SEALED','query_resource_key':key,'candidate_keys':keys,
                    'authority_sha256':sha(AUTH),'saved_query_sha256':logical(old),'source_receipt_sha256':logical(ep['source_receipt']),
                    'weights':tensor_record(weights),'weights_binary64':[hx(x) for x in weights],
                    'arrays':{'path':str(array_path.relative_to(OUT)),'sha256':sha(array_path)},
                    'owners':owners,'full_C128_scores_binary64':vectors,'score_vector_sha256':logical(vectors),
                    'target_reads':0,'training_updates':0,'support_reselection_count':0,
                    'reassigned_pairs_required_to_satisfy_original_cycle_or_graph':False}
                dest=OUT/'prejoin'/(key+'.json');atomic(dest,query)
                records.append({'query_resource_key':key,'path':str(dest.relative_to(OUT)),'sha256':sha(dest)})
                print(json.dumps({'event':'ASSIGNMENT_QUERY_SEALED','query_count':len(records),'H1':h1,'H0':h0,'elapsed_seconds':time.monotonic()-began}),flush=True)
        need(len(records)==32 and {x['query_resource_key'] for x in records}==set(saved) and h1==1672 and h0==2424,'FULL_PREJOIN_AXIS')
        atomic(OUT/'input_closure.json',bundle.prejoin_receipt())
        seal={'status':'RC_V8_ASSIGNMENT_INTERVENTION_FULL32_C128_PREJOIN_CLOSED','authority_sha256':sha(AUTH),
          'records':sorted(records,key=lambda r:r['query_resource_key']),'H1_count':h1,'H0_count':h0,
          'input_closure_sha256':sha(OUT/'input_closure.json'),'label_read_attempts_before_closure':bundle.barrier.blocked_read_attempts}
        atomic(OUT/'prejoin_seal.json',seal);atomic(OUT/'result.json',summarize(a,seal))
        print(json.dumps({'status':'RC_V8_ASSIGNMENT_INTERVENTION_COMPLETE','scientific_GO_or_NO_GO':None}),flush=True);return 0
    except TimeoutError:
        atomic(OUT/'incomplete.json',{'status':'INCOMPLETE_TIME_LIMIT_OR_SIGNAL','sealed_queries':len(records),
            'authority_sha256':sha(AUTH),'resumable':False,'scientific_GO_or_NO_GO':None});return 75

def sealed_query(item):
    path=OUT/item['path'];need(item['path']=='prejoin/'+item['query_resource_key']+'.json' and sha(path)==item['sha256'],'SEALED_QUERY_HASH_PATH')
    return read(path)

def summarize(a,seal):
    need(read(OUT/'prejoin_seal.json')==seal and seal['authority_sha256']==sha(AUTH) and len(seal['records'])==32
         and seal['H1_count']==1672 and seal['H0_count']==2424 and seal['label_read_attempts_before_closure']==0
         and sha(OUT/'input_closure.json')==seal['input_closure_sha256'],'POSTJOIN_BEFORE_FULL_SEAL')
    joined=read(checked(a['postjoin']['authority']));valid=read(checked(a['postjoin']['validation']))
    need(joined['record_count']==32 and joined['fold_id']==1 and joined['target_insertion_count']==0
         and valid['authority_sha256']==a['postjoin']['authority']['sha256'] and valid['checks']
         and all(v is True for v in valid['checks'].values()),'POSTJOIN_VALIDATION')
    roles={r['query_resource_key']:r for r in joined['records']}
    need(set(roles)=={x['query_resource_key'] for x in seal['records']},'FULL32_ROLE_AXIS')
    records=[]
    for item in seal['records']:
        q=sealed_query(item);keys=q['candidate_keys'];targetset=set(roles[q['query_resource_key']]['target_candidate_resource_keys'])
        need(targetset and targetset.issubset(keys),'TARGET_AXIS');ti=[i for i,k in enumerate(keys) if k in targetset];wi=[i for i,k in enumerate(keys) if k not in targetset]
        hard=[number(x) for x in q['full_C128_scores_binary64']['HARD']]
        fixed_t=max(ti,key=lambda i:hard[i]);fixed_w=max(wi,key=lambda i:hard[i]);conditions={}
        for mode in CONDITIONS:
            scores=[number(x) for x in q['full_C128_scores_binary64'][mode]]
            t=max(ti,key=lambda i:scores[i]);w=max(wi,key=lambda i:scores[i]);margin=scores[t]-scores[w]
            energies=[frac(o['conditions'][mode]['common_query_energy_exact']) if o['conditions'][mode]['state']=='H1' else None for o in q['owners']]
            et=min((energies[i] for i in ti if energies[i] is not None),default=None)
            ew=min((energies[i] for i in wi if energies[i] is not None),default=None)
            secondary_success=et is not None and (ew is None or et<ew)
            accounting={}
            for label,index in (('fixed_original_target',fixed_t),('fixed_original_wrong',fixed_w)):
                cell=q['owners'][index]['conditions'][mode];baseline=q['owners'][index]['conditions']['HARD']
                if cell['state']=='H0':accounting[label]={'state':'H0'};continue
                dg=frac(cell['grouped_energy_exact_from_saved_atom_costs'])-frac(baseline['grouped_energy_exact_from_saved_atom_costs'])
                dq=frac(cell['common_query_energy_exact'])-frac(baseline['common_query_energy_exact'])
                dgap=frac(cell['grouping_excess_over_common_query_exact'])-frac(baseline['grouping_excess_over_common_query_exact'])
                need(dg==dq+dgap,'EXACT_GROUPING_MEDIATION_ACCOUNTING')
                accounting[label]={'state':'H1','delta_grouped_exact':rational(dg),'delta_common_query_exact':rational(dq),
                    'delta_grouping_excess_exact':rational(dgap),'identity':'delta_grouped = delta_common_query + delta_grouping_excess'}
            conditions[mode]={'rank_success':margin>0,'margin_binary64':hx(margin),
              'fixed_original_pair_energy_accounting':accounting,
              'selected_target_key':keys[t],'strongest_wrong_key':keys[w],
              'fixed_original_target_wrong_margin_binary64':hx(scores[fixed_t]-scores[fixed_w]),
              'fixed_original_pair_crossing_is_accuracy_claim':False,
              'common_query_secondary_rank_success':secondary_success,
              'common_query_target_energy_exact':rational(et) if et is not None else None,
              'common_query_wrong_energy_exact':rational(ew) if ew is not None else None}
        records.append({'query_resource_key':q['query_resource_key'],'supergroup_hash':roles[q['query_resource_key']]['supergroup_hash'],
            'target_candidate_keys':sorted(targetset),'fixed_original_target_key':keys[fixed_t],'fixed_original_wrong_key':keys[fixed_w],
            'conditions':conditions})
    counts={}
    for mode in CONDITIONS:
        hard_flags=[r['conditions']['HARD']['rank_success'] for r in records];flags=[r['conditions'][mode]['rank_success'] for r in records]
        secondary=[r['conditions'][mode]['common_query_secondary_rank_success'] for r in records]
        secondary_hard=[r['conditions']['HARD']['common_query_secondary_rank_success'] for r in records]
        rescue=sum(x and not h for x,h in zip(flags,hard_flags));breaks=sum(h and not x for x,h in zip(flags,hard_flags))
        sr=sum(x and not h for x,h in zip(secondary,secondary_hard));sb=sum(h and not x for x,h in zip(secondary,secondary_hard))
        counts[mode]={'primary_success':sum(flags),'primary_rescue_vs_HARD':rescue,'primary_break_vs_HARD':breaks,'primary_paired_net_vs_HARD':rescue-breaks,
          'secondary_common_query_success':sum(secondary),'secondary_rescue_vs_HARD':sr,'secondary_break_vs_HARD':sb,'secondary_paired_net_vs_HARD':sr-sb,
          'fixed_original_pair_negative_to_positive_count':sum(number(r['conditions']['HARD']['fixed_original_target_wrong_margin_binary64'])<=0
              and number(r['conditions'][mode]['fixed_original_target_wrong_margin_binary64'])>0 for r in records)}
    need(counts['HARD']['primary_success']==9,'FROZEN_HARD_9_OF_32_RANK_REPLAY')
    return {'status':'RC_V8_ASSIGNMENT_INTERVENTION_COMPLETE','claim_level':'POSTHOC_FROZEN_SUPPORT_COUNTERFACTUAL_RANK_DIAGNOSTIC',
        'authority_sha256':sha(AUTH),'prejoin_seal_sha256':sha(OUT/'prejoin_seal.json'),'query_count':32,'candidate_count':128,
        'H1_count':1672,'H0_count':2424,'counts':counts,'records':records,'contract':CONTRACT,
        'interpretation_limits':['Fixed original selected query supports remain candidate dependent.',
          'Reassigned pairs may violate original cycle and paired connectivity; this is a controlled diagnostic.',
          'WITHIN_R isolates reassignment within fixed admissible R extent; ALL_R additionally expands extent.',
          'Per-atom assignment cost is minimized; neither grouped nor common-query pooled energy is guaranteed to improve.',
          'No reproduction claim for historical 27/32 or 69/90 whole-system outcomes; no formal HYP GO.'],
        'training_updates':0,'region_reselection_count':0,'formal_panel_consumed':False,'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}

def validate():
    import numpy as np
    import torch
    a=load_authority();core,weights=load_head(a);saved=saved_queries(a);seal=read(OUT/'prejoin_seal.json');h1=h0=0
    for item in seal['records']:
        q=sealed_query(item);key=q['query_resource_key'];old=saved[key]
        need(q['authority_sha256']==sha(AUTH) and q['saved_query_sha256']==logical(old) and q['candidate_keys']==old['candidate_keys']
             and q['weights']==tensor_record(weights) and q['weights_binary64']==[hx(x) for x in weights],'QUERY_BINDING')
        array=q['arrays'];need(array['path']=='arrays/'+key+'.npz' and sha(OUT/array['path'])==array['sha256'],'ARRAY_HASH_PATH')
        arrays=np.load(OUT/array['path'],allow_pickle=False);expected_keys=set();vectors={m:[] for m in CONDITIONS}
        for ci,owner in enumerate(q['owners']):
            original=old['arms']['REAL']['decisions'][ci];support=original['selected_H'];domain=owner['source_query_indices']
            need(owner['candidate_resource_key']==q['candidate_keys'][ci] and owner['source_H']==support
                 and owner['source_H_sha256']==logical(support) and owner['original_map_seed_ordinal']==original['map_seed_ordinal'],'OLD_H_TAMPER')
            if support is None:need(not domain,'H0_QUERY_SUPPORT');h0+=1
            else:
                need(domain==support['source_query_indices'] and owner['original_source_reference_indices']==support['source_reference_indices'],'H_SOURCE_AXIS')
                h1+=1
            for mode in CONDITIONS:
                cell=owner['conditions'][mode]
                if support is None:residual=costs=None
                else:
                    rkey=f'c{ci}_{mode}_residual';expected_keys.add(rkey);residual=torch.from_numpy(arrays[rkey].copy())
                    ckey=f'c{ci}_{mode}_costs';costs=None
                    if mode!='HARD':expected_keys.add(ckey);costs=torch.from_numpy(arrays[ckey].copy())
                    allowed=sorted(set(support['source_reference_indices'])) if mode!='ALL_R' else owner['valid_reference_indices']
                    need(cell['admissible_source_reference_indices']==allowed,'ADMISSIBLE_DOMAIN_TAMPER')
                    if mode=='HARD':need(cell['chosen_source_reference_indices']==support['source_reference_indices']
                        and cell['pooling_witness']==support['pooling_witness'],'HARD_WITNESS_TAMPER')
                validate_cell(core,cell,residual,costs,domain,weights);vectors[mode].append(cell['score_binary64'])
            need(owner['conditions']['HARD']['score_binary64']==original['candidate_score_binary64'],'HARD_SCORE_TAMPER')
        need(set(arrays.files)==expected_keys,'NPZ_ARRAY_AXIS');arrays.close()
        need(vectors==q['full_C128_scores_binary64'] and logical(vectors)==q['score_vector_sha256']
             and vectors['HARD']==old['arms']['REAL']['scores_binary64'],'FULL_C128_VECTOR_REPLAY')
    need(h1==1672 and h0==2424,'VALIDATED_OWNER_POPULATION')
    need(read(OUT/'result.json')==summarize(a,seal),'COUNTERFACTUAL_FULL_C128_RANK_REDUCTION_DRIFT')
    receipt={'status':'RC_V8_ASSIGNMENT_INTERVENTION_INDEPENDENT_ARTIFACT_VALIDATION_PASS',
        'authority_sha256':sha(AUTH),'result_sha256':sha(OUT/'result.json'),'query_count':32,'H1_count':h1,'H0_count':h0,
        'checks':{'immutable_parent_and_source_chain':True,'all_original_selected_H_and_H0_preserved':True,
          'full_cost_matrix_argmin_and_lowest_sourceR_ties_replayed':True,'selected_residual_pair_cost_replayed':True,
          'original_core_K_energy_physical_witness_from_selected_residuals_replayed':True,
          'secondary_exact_common_query_energy_replayed':True,'all_3_full_C128_vectors_replayed':True,
          'all_32_actual_rank_and_rescue_break_counts_reduced_after_seal':True,'HARD_nine_of_32_replayed':True},
        'independent_raw_query_reference_tokens_to_full_cost_matrix_recomputed':False,
        'validation_boundary':'Saved full cost matrices and selected FP64 residuals replayed; source-token-to-cost production is pinned but not independently repeated.',
        'scientific_GO_or_NO_GO':None,'automatic_stage_advance':False}
    atomic(OUT/'artifact_validation.json',receipt);print(json.dumps(receipt,sort_keys=True),flush=True)

def synthetic_e0():
    import torch
    core=module({'path':'src/rc_aslo_xf/reference_conditioned_coherent_residual_p_only_v8.py','sha256':b.PINS['core']},'rc_assignment_e0_core')
    q=torch.zeros((5,128),dtype=torch.float64);q[:,0]=torch.tensor([0.,.25,.5,.75,1.])
    reference=torch.zeros((7,128),dtype=torch.float64);reference[:,0]=torch.tensor([0.,.5,.5,.75,1.,1.5,0.]);reference[6]=float('nan')
    weights=torch.linspace(.1,1.1,256,dtype=torch.float64);allowed=torch.tensor([0,1,2,3,4,5])
    r,j,costs=admissible_assignment(q,reference,allowed,weights,query_chunk=2,reference_chunk=2)
    r2,j2,c2=admissible_assignment(q,reference,allowed,weights,query_chunk=5,reference_chunk=6)
    need(torch.equal(r,r2) and torch.equal(j,j2) and torch.equal(costs,c2),'CHUNK_BIT_PARITY')
    need(j.tolist()==[0,0,1,3,4],'LOWEST_SOURCE_R_TIE')
    expected=((q[:,None]-reference[allowed][None]).square()*(weights[:128]+weights[128:])).sum(-1)
    need(torch.equal(costs,expected),'EXPLICIT_ORIGINAL_COST_ARITHMETIC')
    oldrefs=torch.tensor([1,1,2,3,3]);within=torch.unique(oldrefs,sorted=True)
    wr,wj,wc=admissible_assignment(q,reference,within,weights)
    need(set(wj.tolist()).issubset(within.tolist()) and bool((costs.min(1).values<=wc.min(1).values).all()),'REFERENCE_NESTING_ARGMIN_BOUND')
    domain=[8,2,7,3,11];cell=score_cell(core,r,j,torch.tensor(domain),weights)
    cell.update(admissible_source_reference_indices=allowed.tolist(),pair_cost_matrix=tensor_record(costs),argmin_column_vector=tensor_record(costs.argmin(1)))
    validate_cell(core,cell,r,costs,domain,weights)
    bad=json.loads(encode(cell));bad['chosen_source_reference_indices'][2]=2
    try:validate_cell(core,bad,r,costs,domain,weights)
    except (RuntimeError,ValueError):pass
    else:raise AssertionError('ARGMIN_TAMPER_ACCEPTED')
    validate_cell(core,h0_cell(),None,None,[],weights)
    bad=h0_cell();bad['score_binary64']=hx(.5)
    try:validate_cell(core,bad,None,None,[],weights)
    except RuntimeError:pass
    else:raise AssertionError('H0_FILL_ACCEPTED')
    # Candidate order is external to each assignment and keyed scores remain unchanged.
    inputs={'c0':q,'c1':q.flip(0)}
    forward={k:tensor_record(admissible_assignment(v,reference,allowed,weights)[1]) for k,v in inputs.items()}
    reverse={k:tensor_record(admissible_assignment(inputs[k],reference,allowed,weights)[1]) for k in reversed(tuple(inputs))}
    need(forward==reverse,'CANDIDATE_PERMUTATION_DRIFT')
    with tempfile.TemporaryDirectory(prefix='rc-assignment-e0-') as temp:
        path=Path(temp)/'immutable.json';atomic(path,{'x':1});atomic(path,{'x':1})
        try:atomic(path,{'x':2})
        except RuntimeError:pass
        else:raise AssertionError('IMMUTABLE_OVERWRITE_ACCEPTED')
    return {'status':'RC_V8_ASSIGNMENT_INTERVENTION_SYNTHETIC_E0_PASS','checks':{
      'explicit_squared_residual_original_cost_arithmetic':True,'query_and_reference_chunk_bit_parity':True,
      'lowest_original_source_reference_index_ties':True,'within_admissible_set_and_full_nested_set':True,
      'invalid_nan_reference_tokens_unread':True,'H0_no_appearance_fill':True,'candidate_permutation':True,
      'selected_residual_coherent_score_energy_witness_replay':True,'argmin_tamper_rejected':True,
      'immutable_nonidentical_overwrite_rejected':True},'natural_value_reads':0,'scientific_GO_or_NO_GO':None}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=('e0','freeze','run','validate','run-and-validate'));args=parser.parse_args()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.phase=='e0':print(json.dumps(synthetic_e0(),sort_keys=True));return 0
    if args.phase=='freeze':freeze();return 0
    if args.phase in ('run','run-and-validate'):
        code=run()
        if code or args.phase=='run':return code
    validate();return 0
if __name__=='__main__':raise SystemExit(main())
