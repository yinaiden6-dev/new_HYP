#!/usr/bin/env python3
"""One absolute-scale calibration test with a matched relative-only control."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
NAME='rc_absolute_evidence_scale_calibration_v1'
OUT=ROOT/'results'/NAME
PREFLIGHT=OUT.parent/(NAME+'_preflight_v2')
AUTH=ROOT/'registry/rc_absolute_evidence_scale_calibration_authority_v1_20260909.json'
PLAN=ROOT/'plan/RC_ABSOLUTE_EVIDENCE_SCALE_CALIBRATION_V1_20260909.md'
LAUNCH=ROOT/'slurm/rc_absolute_evidence_scale_calibration_v1_dev_cpuonly_59m.sbatch'
HELPER=ROOT/'programs/run_rc_full_mass_free_identity_development_v1.py'
NATIVE=ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
EXT=ROOT/'registry/rc_retrieval_only_research_extension_authority_v1_20260909.json'
SPLIT=ROOT/'registry/rc_shared_query_target_prior_split_qualification_v1_20260909.json'
PINS={HELPER:'4f9bebeb75b0aea8070f6cd8d4ec8f2b71c4e680d33e6154899a998f1d742cb2',
      NATIVE:'42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b',
      EXT:'b9bdcd4b1c847fc22755a2e5565f30a4e77f3bda4f92e4273769601caf10ceda',
      SPLIT:'4f1e242170681e2509f1ed70c0b3a0dacbd0df3bd575f3fe624f39b04d74f642'}
MODELS=('NATIVE7','ABS12','REL12')
MODES=('REAL','CBIND')
ARM='C_PAIRED'
ADDITIONAL=('delta_S','delta_M','delta_L','delta_Q_response','delta_R_response')
KEYS={'NATIVE7':('real_native_features','cbind_native_features'),
      'ABS12':('real_absolute_features','cbind_absolute_features'),
      'REL12':('real_relative_square_features','cbind_relative_square_features')}
H=None


def need(x,message):
    if not bool(x): raise RuntimeError(message)


def helper():
    global H
    need(hashlib.sha256(HELPER.read_bytes()).hexdigest()==PINS[HELPER],'INPUT_HELPER_DRIFT')
    spec=importlib.util.spec_from_file_location('frozen_scalar_training_inputs',HELPER)
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    return H


def deadline():
    value=H.read(EXT)
    cutoff=datetime.fromisoformat(value['cutoff_UTC'])
    need(datetime.now(timezone.utc)<cutoff,'USER_RESEARCH_DEADLINE_REACHED')


def sources():
    import torch
    values={}
    for p,digest in PINS.items():
        need(H.sha(p)==digest,'SOURCE_PIN:'+p.name);values[p.name]=H.binding(p)
    for key,(relative,digest) in H.PINS.items():
        need(H.sha(ROOT/relative)==digest,'LEGACY_PIN:'+key);values['legacy_'+key]={'path':relative,'sha256':digest}
    values.update(program=H.binding(__file__),plan=H.binding(PLAN),launcher=H.binding(LAUNCH))
    values['runtime']={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),
                       'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}
    return values


def atoms(e):
    s=float(e['real_score']);m=float(e['visibility_mass'])
    return (s,m,s/max(m,1e-12),s-float(e['query_control_score']),s-float(e['reference_control_score']))


def raw_extra(row,mode):
    import torch
    e=H.transform_evidence(row['evidence'][ARM],mode,row.get('cbind_source_positions'))
    winner=int(row['base_winner_position']);w=atoms(e[winner])
    return torch.tensor([[a-b for a,b in zip(atoms(e[int(c)]),w)] for c in row['challenger_positions']],dtype=torch.float64)


def native_key(mode):return 'real_native_features' if mode=='REAL' else 'cbind_native_features'


def e0(frozen):
    import torch
    e={0:{'real_score':.30,'visibility_mass':.75,'query_control_score':.24,'reference_control_score':.26},
       1:{'real_score':.25,'visibility_mass':.50,'query_control_score':.22,'reference_control_score':.20}}
    raw=[2.,1.]
    f=frozen.candidate_feature(raw,e,1,0)
    scaled={i:{k:v*.5 for k,v in row.items()} for i,row in e.items()}
    fs=frozen.candidate_feature(raw,scaled,1,0)
    delta=max(abs(float(x)) for x in fs-f)
    need(delta<1e-9,'EXPECTED_RELATIVE_SCALE_NEAR_INVARIANCE')
    a=torch.tensor([x-y for x,y in zip(atoms(e[1]),atoms(e[0]))],dtype=torch.float64)
    b=torch.tensor([x-y for x,y in zip(atoms(scaled[1]),atoms(scaled[0]))],dtype=torch.float64)
    need(not torch.equal(a,b) and torch.equal(a[2],b[2]),'ABSOLUTE_QUALITY_INFORMATION_TEST')
    need(bool(torch.isfinite(f).all()),'FINITE_FEATURES')
    return {'status':'ABSOLUTE_SCALE_FEATURE_E0_PASS','relative_max_difference_under_common_quality_scale':delta,
            'absolute_difference_max_change':float((a-b).abs().max()),
            'checks':{'relative_features_nearly_ignore_common_quality_scale':True,'absolute_scale_changes_remain_observable':True,
                      'normalized_content_held_fixed':True,'finite_original_features':True},
            'interpretation':'Feature-information property only; no claim about natural accuracy or root cause.'}


def prepare():
    import torch
    frozen,unused,pure=H.load_modules()
    fullv=H.read(ROOT/H.PINS['full_validation'][0]);pairv=H.read(ROOT/H.PINS['pair_validation'][0])
    need(all(v is True for v in fullv['checks'].values()) and all(v is True for v in pairv['checks'].values()),'INPUT_VALIDATION')
    full=[];shards=[]
    for entry in fullv['shards']:
        p=H.FULL/f"shard{int(entry['shard']):02d}/payload.pt"
        need(H.sha(p)==entry['payload_sha256'],'FULL_SHARD_SHA')
        data=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
        full.extend(data['records']);shards.append(H.binding(p))
    train0=sorted((r for r in full if r['data_split_role']=='TRAIN'),key=lambda r:r['execution_ordinal'])
    eval0=sorted((r for r in full if r['data_split_role']=='EVAL'),key=lambda r:r['execution_ordinal'])
    need(len(train0)==len(eval0)==32 and len({r['query_id'] for r in full})==64,'FULL64_POPULATION')
    rolem=H.read(ROOT/H.PINS['role_manifest'][0]);entries={e['execution_ordinal']:e for e in rolem['shards']}
    H.BARRIER=H.ReadBarrier([entries[r['execution_ordinal']]['path'] for r in train0],
        [entries[r['execution_ordinal']]['path'] for r in eval0],
        [ROOT/H.PINS[k][0] for k in ('old_result','old_validation')])
    sys.addaudithook(H.BARRIER.hook)
    split=H.read(SPLIT)
    need(all(v==0 for row in split['overlap_counts'].values() for v in row.values()),'PAIR_TRAIN_EVAL_SPLIT')
    pair=torch.load(H.checked({'path':H.PINS['pair_payload'][0],'sha256':H.PINS['pair_payload'][1]}),map_location='cpu',mmap=True,weights_only=True)
    need(len(pair['records'])==64,'PAIR64')
    all_training=pair['records']+train0
    extras={
        'ABS12':torch.cat([raw_extra(r,'REAL') for r in all_training]),
        'REL12':torch.cat([r['real_native_features'][ARM][:,1:]*r['real_native_features'][ARM][:,1:].abs() for r in all_training])}
    rms={};ledger={}
    for name,x in extras.items():
        need(x.shape==(4128,5) and x.dtype==torch.float64 and bool(torch.isfinite(x).all()),'TRAIN_ONLY_SCALE_AXIS')
        value=x.square().mean(dim=0).sqrt();rms[name]=torch.where(value>0,value,torch.ones_like(value))
        ledger[name]={'raw_RMS_binary64':[H.hx(v) for v in value], 'divisor_binary64':[H.hx(v) for v in rms[name]],
                      'zero_RMS_columns':torch.nonzero(value==0).flatten().tolist(),'training_row_count':4128}
    def extend(row,pair_row=False):
        out=dict(row)
        for mode in (('REAL',) if pair_row else MODES):
            nk=native_key(mode);native=row[nk][ARM]
            evidence=H.transform_evidence(row['evidence'][ARM],mode,row.get('cbind_source_positions'))
            rebuilt=torch.stack([frozen.candidate_feature(row['base_scores'].tolist(),evidence,int(c),int(row['base_winner_position'])) for c in row['challenger_positions']])
            need(H.tensor_sha(native)==H.tensor_sha(rebuilt),'ORIGINAL_NATIVE_FEATURE_BITS')
            for name in ('ABS12','REL12'):
                x=raw_extra(row,mode) if name=='ABS12' else native[:,1:]*native[:,1:].abs()
                key=KEYS[name][0 if mode=='REAL' else 1]
                out[key]={ARM:torch.cat([native,x/rms[name]],dim=1)}
        return out
    pair={**pair,'records':[extend(r,True) for r in pair['records']]}
    train0=[extend(r) for r in train0];evals=[extend(r) for r in eval0]
    labels=H.corrected_labels();train=H.role_join(train0,entries,labels)
    family=pure['train_head'].__globals__['FAMILIES']
    for name in ('ABS12','REL12'):
        family[name]=(KEYS[name][0],KEYS[name][1],tuple(frozen.FEATURE_NAMES)+tuple(name+'_'+x for x in ADDITIONAL))
    for name in MODELS:
        key=KEYS[name][0];design=torch.cat([r[key][ARM] for r in pair['records']+train])
        with_bias=torch.cat([design,torch.ones((len(design),1),dtype=torch.float64)],dim=1)
        ledger.setdefault(name,{})['design_rank_with_bias']=int(torch.linalg.matrix_rank(with_bias))
        ledger[name]['feature_sha256']=H.tensor_sha(design)
        ledger[name]['parameter_count']=design.shape[1]+1
    closure={'source_shards':shards,'training_order':[r['execution_ordinal'] for r in train],
             'pair_order':[[r['pair_cohort'],r['pair_row_ordinal'],r['execution_ordinal']] for r in pair['records']],
             'eval_order':[r['execution_ordinal'] for r in evals],'scales_and_design':ledger,
             'TRAIN_role_reads':32,'EVAL_role_reads':0,'new_forward_count':0,
             'eval_feature_shas':{str(r['execution_ordinal']):{name:{mode:H.tensor_sha(r[KEYS[name][0 if mode=='REAL' else 1]][ARM]) for mode in MODES} for name in MODELS} for r in evals}}
    return frozen,pure,pair,train,evals,entries,labels,closure


def fit(pure,pair,train):
    params={};start=time.monotonic()
    expected=H.read(NATIVE)
    for name in MODELS:
        head,loss,finite=pure['train_head'](pair,train,name,ARM)
        weight=head.weight.detach().flatten();bias=float(head.bias.detach())
        params[name]={'weight_binary64':[H.hx(v) for v in weight],'bias_binary64':H.hx(bias),
                      'parameter_sha256':pure['parameter_sha'](weight,bias),'parameter_count':len(weight)+1,
                      'loss_total':loss[0],'loss_pair':loss[1],'loss_fullnegative':loss[2],'finite_training':finite}
        if name=='NATIVE7':
            need(params[name]['weight_binary64']==expected['weight_binary64'] and params[name]['bias_binary64']==expected['bias_binary64'],'NATIVE7_PARAMETER_REGRESSION_ABORT')
        print(json.dumps({'event':'CALIBRATION_HEAD_FIT_COMPLETE','model':name,'seconds':time.monotonic()-start,'parameter_sha256':params[name]['parameter_sha256']}),flush=True)
    return params


def predict(row,param,name,mode):
    import torch
    w=torch.tensor([float.fromhex(x) for x in param['weight_binary64']],dtype=torch.float64);b=float.fromhex(param['bias_binary64'])
    values=row[KEYS[name][0 if mode=='REAL' else 1]][ARM]@w+b
    axis=row['candidate_physical_rows'];cs=row['challenger_positions'];winner=row['base_winner_position']
    i=max(range(len(cs)),key=lambda i:(float(values[i]),-axis[cs[i]]));switch=float(values[i])>0
    return {'final_position':int(cs[i]) if switch else int(winner),'decision':'SWITCH' if switch else 'HOLD',
            'all127_logits_binary64':[H.hx(x) for x in values]}


def paired(new,base):
    need([r['query_id'] for r in new]==[r['query_id'] for r in base],'PAIRED_QUERY_ORDER')
    rescue=[n['query_id'] for n,b in zip(new,base) if n['final_correct'] and not b['final_correct']]
    breaks=[n['query_id'] for n,b in zip(new,base) if b['final_correct'] and not n['final_correct']]
    return {'rescue':len(rescue),'break':len(breaks),'net':len(rescue)-len(breaks),'rescue_query_ids':rescue,'break_query_ids':breaks}


def result(params,prejoin,train,evals,entries,labels,pure,closure):
    import torch
    seal=H.read(OUT/'eval_prejoin_seal.json')
    need(seal['parameters_sha256']==H.sha(OUT/'parameters.json') and seal['eval_prejoin_sha256']==H.sha(OUT/'eval_prejoin.json') and H.BARRIER.blocked==0,'POSTJOIN_SEAL')
    H.BARRIER.release();joined=H.role_join(evals,entries,labels)
    need(all(not ({r[k] for r in train}&{r[k] for r in joined}) for k in ('query_id','target_identity','supergroup')),'TRAIN_EVAL_OVERLAP')
    old=H.read(ROOT/H.PINS['old_result'][0]);all_actions={};metrics={}
    for role,rows in (('TRAIN',train),('EVAL',joined)):
        all_actions[role]={};metrics[role]={}
        for name in MODELS:
            p=params[name];w=torch.tensor([float.fromhex(x) for x in p['weight_binary64']],dtype=torch.float64);b=float.fromhex(p['bias_binary64'])
            all_actions[role][name]={mode:pure['actions'](w,b,rows,name,ARM,control=mode=='CBIND') for mode in MODES}
            metrics[role][name]={mode:pure['summary'](acts) for mode,acts in all_actions[role][name].items()}
    for mode,field in (('REAL','actions'),('CBIND','cbind_actions')):
        need(H.encode(all_actions['EVAL']['NATIVE7'][mode])==H.encode(old['evaluations']['NATIVE7'][ARM][field]),'BASELINE_ACTION_REGRESSION_ABORT')
    by_ex={r['execution_ordinal']:r for r in prejoin}
    for row in joined:
        for name in MODELS:
            for mode in MODES:
                pred=by_ex[row['execution_ordinal']]['predictions'][name][mode]
                need(pred==predict(row,params[name],name,mode),'PREJOIN_POSTJOIN_DRIFT')
    comparisons={role:{name:paired(all_actions[role][name]['REAL'],all_actions[role]['NATIVE7']['REAL']) for name in ('ABS12','REL12')} for role in ('TRAIN','EVAL')}
    vs_relative={role:paired(all_actions[role]['ABS12']['REAL'],all_actions[role]['REL12']['REAL']) for role in ('TRAIN','EVAL')}
    new_ids=set(comparisons['EVAL']['ABS12']['rescue_query_ids'])
    preserved=[r['query_id'] for r in all_actions['EVAL']['ABS12']['CBIND'] if r['query_id'] in new_ids and r['final_correct']]
    improvement=comparisons['EVAL']['ABS12']['net']>0 and comparisons['EVAL']['ABS12']['break']==0
    attribution=improvement and vs_relative['EVAL']['net']>0 and not preserved
    status=('ABSOLUTE_SCALE_INTERNAL_NO_BREAK_IMPROVEMENT' if improvement else
            'ABSOLUTE_SCALE_INTERNAL_NET_GAIN_WITH_BREAKS' if comparisons['EVAL']['ABS12']['net']>0 else
            'ABSOLUTE_SCALE_NO_INTERNAL_NET_GAIN')
    return {'status':status,
            'authority_sha256':H.sha(AUTH),'input_closure_sha256':H.sha(OUT/'input_closure.json'),
            'parameters_sha256':H.sha(OUT/'parameters.json'),'eval_prejoin_seal_sha256':H.sha(OUT/'eval_prejoin_seal.json'),
            'baseline_parameter_and_action_regression_pass':True,'metrics':metrics,'actions':all_actions,
            'paired_vs_NATIVE7':comparisons,'ABS12_vs_REL12':vs_relative,
            'ABS12_new_rescues_retained_under_CBIND':preserved,'internal_improvement_candidate':improvement,
            'absolute_information_specific_internal_support':attribution,'scales_and_design':closure['scales_and_design'],
            'query_count':32,'TRAIN_query_count':32,'PAIR_query_count':64,'candidate_count':128,
            'EVAL_supergroup_count':len({r['supergroup'] for r in joined}),
            'evidence_level':'Previously opened internal RAW EVAL32 only; no external confirmation or old P-only/HYP GO',
            'supervised_training':'Retrieval PAIR labels and FULL TRAIN exact identity only; no task spatial annotation',
            'new_encoder_or_RoMa_forward_count':0,'deployment_changed':False,'HYP_GO_claimed':False,
            'limits':['ABS12 is the fixed primary; REL12 is matched parameter count, not guaranteed identical effective capacity.',
                      'Existing relative feature scale near-invariance is not proof that absolute scale is useful.',
                      'New head results may not be combined with historical FROZEN_C difficult90 results.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('preflight','freeze','run','validate'),required=True);phase=p.parse_args().phase
    import torch
    torch.set_num_threads(8);torch.set_num_interop_threads(1);helper();deadline();src=sources()
    if phase=='freeze':
        need(not AUTH.exists() and not OUT.exists(),'APPEND_ONLY_AUTHORITY')
        proof=H.read(PREFLIGHT/'result.json')
        need(proof['sources']==src,'PREFLIGHT_SOURCE_DRIFT')
        H.atomic(AUTH,{'status':'ABSOLUTE_SCALE_CALIBRATION_AUTHORIZED','sources':src,'models':list(MODELS),'primary':'ABS12',
                      'updates_per_head':2000,'cutoff_UTC':H.read(EXT)['cutoff_UTC'],'preflight':H.binding(PREFLIGHT/'result.json')})
        print(json.dumps({'authority_sha256':H.sha(AUTH)}),flush=True);return
    frozen,pure,pair,train,evals,entries,labels,closure=prepare()
    if phase=='preflight':
        H.atomic(PREFLIGHT/'result.json',{'status':'ABSOLUTE_SCALE_PREFLIGHT_PASS','sources':src,'e0':e0(frozen),'input_closure':closure,'training_updates':0})
        print(json.dumps({'status':'PREFLIGHT_PASS','e0':e0(frozen),'design':closure['scales_and_design']}),flush=True);return
    need(H.read(AUTH)['sources']==src,'AUTHORITY_SOURCE_DRIFT')
    validating=phase=='validate';need(OUT.exists() if validating else not OUT.exists(),'OUTPUT_STATE')
    if validating:need(H.read(OUT/'input_closure.json')==closure,'INPUT_REBUILD_DRIFT')
    else:H.atomic(OUT/'input_closure.json',closure)
    params=fit(pure,pair,train)
    if validating:need(H.read(OUT/'parameters.json')==params,'INDEPENDENT_RETRAIN_PARAMETER_DRIFT')
    else:H.atomic(OUT/'parameters.json',params)
    prejoin=[{'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],
              'predictions':{name:{mode:predict(r,params[name],name,mode) for mode in MODES} for name in MODELS}} for r in evals]
    if validating:need(H.read(OUT/'eval_prejoin.json')==prejoin,'INDEPENDENT_PREJOIN_REPLAY')
    else:
        H.atomic(OUT/'eval_prejoin.json',prejoin)
        H.atomic(OUT/'eval_prejoin_seal.json',{'parameters_sha256':H.sha(OUT/'parameters.json'),'eval_prejoin_sha256':H.sha(OUT/'eval_prejoin.json'),
                 'query_count':32,'candidate_count':128,'runtime_EVAL_role_reads_before_seal':0,'models':list(MODELS)})
    value=result(params,prejoin,train,evals,entries,labels,pure,closure)
    if validating:
        need(H.read(OUT/'result.json')==value,'INDEPENDENT_RESULT_REPLAY')
        H.atomic(OUT/'independent_validation.json',{'status':'ABSOLUTE_SCALE_INDEPENDENT_REEXECUTION_VALIDATION_PASS',
                 'result_sha256':H.sha(OUT/'result.json'),'checks':{'source_features_RMS_rebuilt':True,'all_three_heads_retrained_exactly':True,
                 'all_EVAL_predictions_and_metrics_replayed':True,'original_NATIVE7_regression':True},
                 'scope':'New process full source/training/prediction replay, using the same frozen training algorithm; not a distinct learning method.'})
    else:H.atomic(OUT/'result.json',value)
    print(json.dumps({'status':value['status'],'validation':validating,'EVAL':value['metrics']['EVAL'],
                      'paired':value['paired_vs_NATIVE7']['EVAL'],'vs_relative':value['ABS12_vs_REL12']['EVAL']}),flush=True)


if __name__=='__main__':main()
