#!/usr/bin/env python3
"""One frozen formula on V7's saved supports and final head; never rerank."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,signal,statistics,sys,tempfile,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PREFIX='rc_v7_residual_envelope_diagnostic_v1'
AUTH=ROOT/('registry/'+PREFIX+'_authority_20260908.json')
PLAN=ROOT/'plan/RC_V7_FIXED_SUPPORT_RESIDUAL_ENVELOPE_DIAGNOSTIC_V1_20260908.md'
LAUNCH=ROOT/'slurm/rc_v7_residual_envelope_diagnostic_v1_dev_cpuonly_59m.sbatch'
OUT=ROOT/('results/'+PREFIX)
V7AUTH=ROOT/'registry/rc_grouped_residual_development_authority_v7_20260908.json'
ADAPTER='programs/prepare_rc_grouped_residual_inputs_v7.py'
CORE='src/rc_aslo_xf/reference_conditioned_grouped_residual_p_only_v7.py'
RUNNER='programs/run_rc_grouped_residual_development_v7.py'
V7OUT='results/rc_grouped_residual_development_v7'
STOP=False

def need(v,c):
    if not bool(v):raise RuntimeError(c)
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def logical(x):return hashlib.sha256(encode(x)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def binding(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def checked(b):
    p=ROOT/b['path'];need(not p.is_symlink() and p.resolve().is_relative_to(ROOT),'UNSAFE_SOURCE')
    need(not any(x in str(p).lower() for x in ('d1_mi','d1-mi','grozi','gisc_prerecall_universe')),'PROTECTED_SOURCE')
    need(sha(p)==b['sha256'],'SOURCE_HASH_DRIFT:'+b['path']);return p
def module(b,name):
    p=checked(b);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=encode(x)+b'\n'
    if p.exists():need(p.read_bytes()==data,'IMMUTABLE_REPLAY_DRIFT');return
    fd,n=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.chmod(n,0o444);os.link(n,p)
    finally:os.unlink(n)
def hx(x):
    x=float(x);need(math.isfinite(x),'NONFINITE');return x.hex()

def fixed_support(e,refs,source_q,weights,*,saved_old_score=None):
    """Fixed support only. Channel-envelope certificate and actual-atom score."""
    import torch
    need(e.dtype==weights.dtype==torch.float64 and e.ndim==2 and e.shape[1]==128 and weights.shape==(256,), 'FP64_RESIDUAL_WEIGHT_AXIS')
    need(bool(torch.isfinite(e).all()) and bool((e>=0).all()) and bool(torch.isfinite(weights).all()) and bool((weights>0).all()),'POSITIVE_FINITE_DOMAIN')
    need(refs.dtype==source_q.dtype==torch.long and refs.shape==source_q.shape==e.shape[:1] and len(set(source_q.tolist()))==len(e),'SOURCE_ATOM_AXIS')
    if not len(e):
        need(saved_old_score is None or float(saved_old_score)==-256.0,'EMPTY_H0_SCORE_DRIFT')
        return {'empty_H1':True,'atom_count':0,'reference_cell_count':0,'old_score':-256.0,'coherent_fixed_support_score':-256.0,
                'score_gap':0.0,'excess_penalty':0.0,'reference_groups':[],'selected_global_worst_source_query_index':None,
                'positive_weight_componentwise_certificate':True,'numeric_coherent_ge_old':True,'roundoff_limited_negative_gap':False,
                'score_comparison_roundoff_bound':0.0,'terms':None}
    order=torch.argsort(source_q);e=e[order];refs=refs[order];source_q=source_q[order]
    unique=torch.unique(refs,sorted=True);wmu,wM=weights[:128],weights[128:]
    maxima=[];coherent_mu=[];group_records=[]
    for ref in unique:
        local=torch.nonzero(refs==ref,as_tuple=False).flatten();values=e[local];envelope=values.max(0).values
        dots=(values*wmu).sum(1);win=int(dots.argmax());atom=values[win]
        need(bool((envelope>=atom).all()),'REFERENCE_ENVELOPE_CERTIFICATE_FAILED')
        maxima.append(envelope);coherent_mu.append(dots[win])
        envelope_dot=(envelope*wmu).sum()
        group_records.append({'source_reference_index':int(ref),'atom_count':len(local),
          'selected_worst_source_query_index':int(source_q[local[win]]),
          'old_envelope_weighted_term_binary64':hx(envelope_dot),'coherent_worst_actual_term_binary64':hx(dots[win]),
          'term_gap_binary64':hx(envelope_dot-dots[win]),'channelwise_dominance_certificate':True})
    means=torch.stack(maxima).mean(0);global_envelope=e.max(0).values
    global_dots=(e*wM).sum(1);global_win=int(global_dots.argmax())
    need(bool((global_envelope>=e[global_win]).all()),'GLOBAL_ENVELOPE_CERTIFICATE_FAILED')
    old_mu=(means*wmu).sum();old_M=(global_envelope*wM).sum();old_concat=(torch.cat((means,global_envelope))*weights).sum()
    # This expression exactly matches the frozen head, preserving its concat reduction.
    old=1.0-.5*old_concat
    if saved_old_score is not None:need(hx(old)==hx(saved_old_score),'SAVED_OLD_SELECTED_SCORE_NOT_REPLAYED')
    new_mu=torch.stack(coherent_mu).mean();new_M=global_dots[global_win];coherent=1.0-.5*(new_mu+new_M)
    gap=coherent-old;unit=2.0**-53;ops=16*(len(e)+128)+128;gamma=(ops*unit)/(1-ops*unit)
    bound=gamma*(1+float(weights.sum())*float(e.max()))
    need(float(gap)>=-bound,'COHERENT_ENVELOPE_INEQUALITY_EXCEEDS_ROUNDOFF')
    return {'empty_H1':False,'atom_count':len(e),'reference_cell_count':len(unique),'old_score':float(old),
      'coherent_fixed_support_score':float(coherent),'score_gap':float(gap),'excess_penalty':float(gap),
      'reference_groups':group_records,'selected_global_worst_source_query_index':int(source_q[global_win]),
      'positive_weight_componentwise_certificate':True,'numeric_coherent_ge_old':float(gap)>=0,
      'roundoff_limited_negative_gap':float(gap)<0,'score_comparison_roundoff_bound':bound,
      'terms':{'old_mu':float(old_mu),'old_global':float(old_M),'coherent_mu':float(new_mu),'coherent_global':float(new_M),
        'mu_excess_half':float(.5*(old_mu-new_mu)),'global_excess_half':float(.5*(old_M-new_M)),
        'old_concat_penalty':float(old_concat),'old_split_minus_concat':float(old_mu+old_M-old_concat),
        'score_gap_minus_split_excess':float(gap-.5*((old_mu-new_mu)+(old_M-new_M)))}}

def synthetic_e0():
    import torch
    w=torch.ones(256,dtype=torch.float64)
    e=torch.zeros((4,128),dtype=torch.float64);e[0,0]=4;e[1,1]=4;e[2,0]=1;e[3,1]=1
    r=torch.tensor([0,0,1,1]);q=torch.tensor([8,2,7,3]);out=fixed_support(e,r,q,w)
    assert out['score_gap']==3.25 and out['terms']['old_mu']==5 and out['terms']['coherent_mu']==2.5
    assert out['terms']['old_global']==8 and out['terms']['coherent_global']==4
    assert out['reference_groups'][0]['selected_worst_source_query_index']==2 and out['selected_global_worst_source_query_index']==2
    perm=torch.tensor([3,1,0,2]);assert fixed_support(e[perm],r[perm],q[perm],w)==out
    e=torch.ones((4,128),dtype=torch.float64);equal=fixed_support(e,r,q,w);assert equal['score_gap']==0
    empty=fixed_support(e[:0],r[:0],q[:0],w,saved_old_score=-256);assert empty['empty_H1'] and empty['score_gap']==0
    try:fixed_support(e,r,q,-w)
    except RuntimeError:pass
    else:raise AssertionError('negativeweightsaccepted')
    anchored_rows=[{'candidate_resource_key':f'c{i}','old_score':old,'coherent_fixed_support_score':coh,'excess_penalty':coh-old,'selected_H_sha256':str(i),'empty_H1':False} for i,(old,coh) in enumerate(((3.,3.),(2.,2.),(1.,100.)))]
    anchor=anchored_summary(anchored_rows,['c0']);assert anchor['fixed_wrong_candidate_resource_key']=='c1' and anchor['coherent_fixed_pair_margin']==1.
    return {'status':'RC_V7_RESIDUAL_ENVELOPE_SYNTHETIC_E0_PASS','checks':{
      'disjoint_channel_maxima_have_exact_positive_excess':True,'coherent_envelope_equality_when_same_actual_atom':True,
      'source_query_tie_break_and_row_permutation_invariant':True,'empty_support_keeps_H0':True,'nonpositive_weights_rejected':True,'fixed_pair_never_reselects_coherent_winner':True},
      'natural_value_reads':0,'scientific_GO_or_NO_GO':None}

def freeze():
    need(not AUTH.exists(),'AUTHORITY_EXISTS');v7=read(V7AUTH)
    need(v7['status']=='RC_GROUPED_RESIDUAL_V7_DEVELOPMENT_AUTHORITY_READY','PARENT_NOT_FROZEN')
    e0=synthetic_e0();e0path=ROOT/('results/'+PREFIX+'_e0/result.json');atomic(e0path,e0)
    # Binding bytes is not an outcome-dependent interpretation of their values.
    files={'program':Path(__file__),'plan':PLAN,'launcher':LAUNCH,'e0':e0path,'v7_authority':V7AUTH,
      'adapter':ROOT/ADAPTER,'core':ROOT/CORE,'runner':ROOT/RUNNER,
      'checkpoint':ROOT/V7OUT/'final_checkpoint.json','saved_scores':ROOT/V7OUT/'score_records.jsonl','v7_result':ROOT/V7OUT/'result.json'}
    for name in ('adapter','core'):
        need(binding(files[name])=={k:v7[name][k] for k in ('path','sha256')},'PARENT_CODE_DRIFT')
    validation=ROOT/v7['validation_rel']/'result.json'
    a={'status':'RC_V7_RESIDUAL_ENVELOPE_ONE_FORMULA_DIAGNOSTIC_AUTHORIZED','sources':{n:binding(p) for n,p in files.items()},
      'postjoin':v7['postjoin'],'input_manifest_sha256':v7['input_manifest_sha256'],
      'contract_sha256':v7['contract']['sha256'],'query_count':32,'candidate_count':128,'output_rel':str(OUT.relative_to(ROOT)),
      'v7_independent_validation':binding(validation) if validation.is_file() else None,
      'candidate_or_support_reselection_authorized':False,'training_authorized':False,'formal_panel_authorized':False,
      'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}
    atomic(AUTH,a);print(json.dumps({'status':a['status'],'authority_sha256':sha(AUTH)}),flush=True)

def anchored_summary(candidate_rows,targets):
    keys=[x['candidate_resource_key'] for x in candidate_rows];targetset=set(targets)
    need(targetset and targetset.issubset(keys) and len(targetset)<len(keys),'POSTJOIN_TARGET_AXIS_INVALID')
    ti=max((i for i,k in enumerate(keys) if k in targetset),key=lambda i:candidate_rows[i]['old_score'])
    wi=max((i for i,k in enumerate(keys) if k not in targetset),key=lambda i:candidate_rows[i]['old_score'])
    t,w=candidate_rows[ti],candidate_rows[wi];old=t['old_score']-w['old_score'];new=t['coherent_fixed_support_score']-w['coherent_fixed_support_score']
    return {'fixed_target_candidate_resource_key':keys[ti],'fixed_wrong_candidate_resource_key':keys[wi],
      'target_selected_H_sha256':t['selected_H_sha256'],'wrong_selected_H_sha256':w['selected_H_sha256'],
      'old_fixed_pair_margin':old,'coherent_fixed_pair_margin':new,'target_excess_penalty':t['excess_penalty'],
      'wrong_excess_penalty':w['excess_penalty'],'target_minus_wrong_excess_penalty':t['excess_penalty']-w['excess_penalty'],
      'fixed_pair_nonpositive_to_positive':old<=0<new,'fixed_pair_positive_to_nonpositive':new<=0<old,
      'target_empty_H1':t['empty_H1'],'wrong_empty_H1':w['empty_H1'],
      'interpretation':'FIXED_ORIGINAL_CANDIDATE_AND_SUPPORT_PAIR_NOT_A_RANKING_OR_RESCUE_RESULT'}

def run():
    global STOP
    began=time.monotonic();a=read(AUTH);ash=sha(AUTH)
    need(a['status']=='RC_V7_RESIDUAL_ENVELOPE_ONE_FORMULA_DIAGNOSTIC_AUTHORIZED','AUTHORITY_INVALID')
    for name,b in a['sources'].items():
        if name!='v7_result':checked(b)
    import torch
    torch.set_num_threads(1)
    adapter=module(a['sources']['adapter'],'rc_v7_envelope_adapter');core=module(a['sources']['core'],'rc_v7_envelope_core')
    runner=module(a['sources']['runner'],'rc_v7_envelope_checkpoint_reader')
    bundle=adapter.InputBundle(ROOT,expected_manifest_sha256=a['input_manifest_sha256'])
    checkpoint=read(checked(a['sources']['checkpoint']))
    need(checkpoint['schema']==runner.SCHEMA and checkpoint['update_index']==512 and checkpoint['contract_sha256']==a['contract_sha256'],'FINAL_CHECKPOINT_INVALID')
    heads=core.GroupedResidualHeads();heads.load_state_dict(runner.unpack(checkpoint['model'],torch),strict=True);heads.eval();weights=heads.real.weights().detach()
    saved_list=[json.loads(line) for line in checked(a['sources']['saved_scores']).read_text().splitlines()]
    need(len(saved_list)==32,'SAVED_SCORE_QUERY_COUNT');saved={x['query_resource_key']:x for x in saved_list};need(len(saved)==32,'DUPLICATE_SAVED_QUERY')
    OUT.mkdir(parents=True,exist_ok=True);qdir=OUT/'prejoin';qdir.mkdir(exist_ok=True);records=[]
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    with torch.no_grad():
        for ep in bundle.iter_queries():
            if STOP or time.monotonic()-began>3000:
                print(json.dumps({'status':'DIAGNOSTIC_RESUMABLE','sealed_queries':len(records)}),flush=True);return 75
            key=ep['query_resource_key'];old=saved[key];keys=list(ep['candidate_keys']);need(old['candidate_keys']==keys,'SAVED_C128_AXIS_DRIFT')
            dest=qdir/(key+'.json');candidates=[]
            for ci,k in enumerate(keys):
                decision=old['arms']['REAL']['decisions'][ci]
                need(decision['candidate_resource_key']==k and decision['query_resource_key']==key and decision['control']=='REAL','SAVED_DECISION_AXIS')
                bank=ep['banks_by_control']['REAL'][ci];families=ep['families']['REAL'][ci]['components']
                all_features=[]
                for family in families:
                    ids=torch.tensor(family['atom_indices'],dtype=torch.long)
                    all_features.append(core.grouped_descriptor(ep['residual_squared']['REAL'][ci,ids],bank.source_reference_indices[ids]))
                vector=heads.real(torch.stack(all_features)) if all_features else torch.empty(0,dtype=torch.float64)
                need(core.tensor_seal(vector)==decision['complete_seed_scores'],'FROZEN_OLD_COMPLETE_VECTOR_REPLAY_DRIFT')
                score=float.fromhex(decision['candidate_score_binary64']);support=decision['selected_H']
                need(hx(score)==old['arms']['REAL']['scores_binary64'][ci],'SAVED_SCORE_VECTOR_DECISION_DRIFT')
                if families:
                    ordinal=decision['map_seed_ordinal'];need(type(ordinal)is int and 0<=ordinal<len(families),'SAVED_MAP_ORDINAL_INVALID')
                    need(int(vector.argmax())==ordinal and hx(vector[ordinal])==hx(score),'SAVED_OLD_MAP_REPLAY_DRIFT')
                    need(support==core._support(families[ordinal]) and decision['state']=='H1','SAVED_SELECTED_H_DRIFT')
                    ids=torch.tensor(support['atom_indices'],dtype=torch.long);need(bool(ep['valid']['REAL'][ci,ids].all()),'SELECTED_INVALID_ATOM')
                else:
                    need(support is None and decision['map_seed_ordinal'] is None and decision['state']=='H0','EMPTY_FAMILY_SAVED_H1')
                    ids=torch.empty(0,dtype=torch.long)
                d=fixed_support(ep['residual_squared']['REAL'][ci,ids],bank.source_reference_indices[ids],bank.source_query_indices[ids],weights,saved_old_score=score)
                candidates.append({'candidate_resource_key':k,'reference_resource_key':bank.reference_resource_key,
                  'saved_map_seed_ordinal':decision['map_seed_ordinal'],'selected_H_sha256':logical(support),**d})
            out={'status':'FIXED_SUPPORT_ENVELOPE_QUERY_PREJOIN_SEALED','query_resource_key':key,'candidate_keys':keys,
              'authority_sha256':ash,'source_receipt_sha256':logical(ep['source_receipt']),'weight_sha256':adapter.tensor_sha(weights),
              'saved_query_sha256':logical(old),'candidates':candidates,'candidate_or_support_reselection_count':0,'target_reads':0}
            atomic(dest,out);records.append({'query_resource_key':key,'path':str(dest.relative_to(OUT)),'sha256':sha(dest)})
            print(json.dumps({'event':'FIXED_SUPPORT_DIAGNOSTIC_QUERY_SEALED','count':len(records),'query':key}),flush=True)
            del ep
    closure=bundle.prejoin_receipt();need(len(records)==32 and {x['query_resource_key'] for x in records}==set(saved),'PREJOIN32_NOT_CLOSED')
    seal={'status':'FIXED_SUPPORT_ENVELOPE_FULL32_C128_PREJOIN_CLOSED','authority_sha256':ash,'records':sorted(records,key=lambda x:x['query_resource_key']),
      'input_closure_sha256':logical(closure),'label_read_attempts_before_closure':bundle.barrier.blocked_read_attempts}
    atomic(OUT/'prejoin_seal.json',seal)
    # First private role-bearing reads occur after every candidate diagnostic closes.
    joined=read(checked(a['postjoin']['authority']));validation=read(checked(a['postjoin']['validation']))
    need(validation['authority_sha256']==a['postjoin']['authority']['sha256'] and all(validation['checks'].values()),'POSTJOIN_NOT_VALIDATED')
    need(joined['record_count']==32 and joined['target_insertion_count']==0,'POSTJOIN_POPULATION_INVALID')
    roles={r['query_resource_key']:r for r in joined['records']};need(set(roles)==set(saved),'ROLE_QUERY_AXIS_DRIFT')
    pairs=[];all_gaps=[];empty=0;limited=0
    for item in seal['records']:
        value=read(OUT/item['path']);need(sha(OUT/item['path'])==item['sha256'],'PREJOIN_ARTIFACT_DRIFT')
        role=roles[item['query_resource_key']];pair=anchored_summary(value['candidates'],role['target_candidate_resource_keys'])
        pairs.append({'query_resource_key':item['query_resource_key'],'supergroup_hash':role['supergroup_hash'],**pair})
        all_gaps.extend(c['excess_penalty'] for c in value['candidates']);empty+=sum(c['empty_H1'] for c in value['candidates']);limited+=sum(c['roundoff_limited_negative_gap'] for c in value['candidates'])
    original=read(checked(a['sources']['v7_result']))
    need(original['final_checkpoint_sha256']==a['sources']['checkpoint']['sha256'] and original['score_records_sha256']==logical(saved_list),'V7_RESULT_SOURCE_BINDING_DRIFT')
    independent={'status':'NOT_BOUND_AT_DIAGNOSTIC_FREEZE','independent_validation_confirmed':False}
    if a['v7_independent_validation'] is not None:
        v=read(checked(a['v7_independent_validation']));need(v['result_sha256']==a['sources']['v7_result']['sha256'],'V7_VALIDATOR_SOURCE_DRIFT')
        independent={'status':v['status'],'independent_validation_confirmed':v['status']=='RC_GROUPED_RESIDUAL_V7_DEVELOPMENT_INDEPENDENT_VALIDATION_PASS' and all(v['checks'].values())}
    def stats(field,subset=None):
        values=[p[field] for p in (pairs if subset is None else subset)]
        return {'mean':math.fsum(values)/len(values),'median':statistics.median(values)} if values else {'mean':None,'median':None}
    strata={
      'old_fixed_pair_nonpositive':[p for p in pairs if p['old_fixed_pair_margin']<=0],
      'old_fixed_pair_positive':[p for p in pairs if p['old_fixed_pair_margin']>0],
      'both_saved_H_nonempty':[p for p in pairs if not p['target_empty_H1'] and not p['wrong_empty_H1']],
      'at_least_one_saved_H_empty':[p for p in pairs if p['target_empty_H1'] or p['wrong_empty_H1']]}
    result={'status':'RC_V7_FIXED_SUPPORT_RESIDUAL_ENVELOPE_DIAGNOSTIC_COMPLETE','claim_level':'POSTHOC_FIXED_PAIR_ACCOUNTING_NOT_RETRIEVAL_RESULT',
      'authority_sha256':ash,'prejoin_seal_sha256':sha(OUT/'prejoin_seal.json'),'query_count':32,'candidate_count':128,'candidate_support_count':4096,
      'empty_H1_candidate_count':empty,'roundoff_limited_negative_gap_count':limited,'all_positive_weight_componentwise_certificates_pass':True,
      'fixed_pair_nonpositive_to_positive_count':sum(p['fixed_pair_nonpositive_to_positive'] for p in pairs),
      'fixed_pair_positive_to_nonpositive_count':sum(p['fixed_pair_positive_to_nonpositive'] for p in pairs),
      'excess_penalty_summaries':{field:stats(field) for field in ('target_excess_penalty','wrong_excess_penalty','target_minus_wrong_excess_penalty')},
      'fixed_pair_strata':{name:{'query_count':len(subset),'excess_penalty_summaries':{field:stats(field,subset) for field in ('target_excess_penalty','wrong_excess_penalty','target_minus_wrong_excess_penalty')}} for name,subset in strata.items()},
      'all_candidate_excess_penalty':{'mean':math.fsum(all_gaps)/len(all_gaps),'median':statistics.median(all_gaps)},
      'records':pairs,'original_v7_gate_go':original['gate_go'],'original_v7_independent_validation':independent,
      'candidate_or_support_reselection_count':0,'training_updates':0,'formal_panel_consumed':False,'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}
    atomic(OUT/'result.json',result);print(json.dumps({'status':result['status'],'out':str(OUT)}),flush=True);return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('e0','freeze','run'),required=True);args=p.parse_args()
    if args.phase=='e0':print(json.dumps(synthetic_e0(),sort_keys=True))
    elif args.phase=='freeze':freeze()
    else:raise SystemExit(run())
