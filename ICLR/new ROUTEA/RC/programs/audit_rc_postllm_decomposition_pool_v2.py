#!/usr/bin/env python3
"""Independent saved-patch evidence -> score -> fixed-head decision replay.

Does not import any producer/backend/adapter or decision function.  It never
trains or runs a model.  Torch only deserializes the sealed three-parameter head.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DECOMP = ROOT/'results/rc_postllm_m_signal_decomposition_v2'
POOL = ROOT/'results/rc_postllm_train_pool_common_v2'
POST = ROOT/'results/rc_postllm_m_v1'
OUT = DECOMP/'independent_decomposition_pool_audit'
CEF = 'H593-90acde9b0567a472f4232120'
MEMO = {}


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p=Path(p).resolve()
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return {'path':str(p),'sha256':h.hexdigest()}


def checked(b):
    key=(b['path'],b['sha256'])
    if key not in MEMO:
        need(bind(b['path'])==b,'SHA:'+b['path'])
        MEMO[key]=True
    return Path(b['path'])


def write(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    os.replace(tmp,p)


def arrays(binding):
    with np.load(checked(binding),allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def mean(v):
    return math.fsum(float(x) for x in v)/len(v)


def close(a,b,limit,msg):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    need(error<=limit,msg+': '+str(error))
    return error


def scores_to_action(row,L,theta):
    raw=row['raw_scores']; w=row['winner_index'];cs=row['challenger_positions']
    need(cs==[i for i in range(128) if i!=w],'CHALLENGER_AXIS')
    rawmean=mean(raw)
    sd=math.sqrt(mean([(x-rawmean)**2 for x in raw]))
    logits=[]
    for c in cs:
        value=math.fsum([theta[0]*(raw[c]-raw[w])/max(sd,1e-12),
            theta[1]*(L[c]-L[w])/(abs(L[c])+abs(L[w])+1e-12),theta[2]])
        logits.append(value)
    j=max(range(127),key=lambda i:logits[i])
    picked=cs[j] if logits[j]>0 else w
    return {'logits':logits,'prediction_position':picked,
        'prediction_identity':row['candidate_identities'][picked],
        'switched':picked!=w,'best_challenger_position':cs[j],
        'best_challenger_logit':logits[j]}


def actioncheck(row, L, theta, saved):
    actual=scores_to_action(row,L,theta)
    error=close(actual['logits'],saved['logits'],2e-10,'LOGITS')
    for k in ['prediction_position','prediction_identity','switched','best_challenger_position']:
        need(actual[k]==saved[k],'ACTION:'+k)
    close(theta,saved['theta'],0.,'HEAD_EXACT')
    return actual,error


def row_summary(rows, label):
    names=list(rows[0]['arms']) if rows else []
    output={}
    for arm in names:
        correct=[];rescues=[];breaks=[]
        for r in rows:
            q=r['query_id'];yes=r['arms'][arm]['action']['prediction_identity']==label[q]['identity']
            if yes:correct.append(q)
            if yes and not label[q]['correct']['RAW']:rescues.append(q)
            if not yes and label[q]['correct']['RAW']:breaks.append(q)
        output[arm]={'n':len(rows),'correct':len(correct),'correct_query_ids':correct,
            'rescues':rescues,'breaks':breaks}
    return output


def main(decomposition_only=False):
    global OUT
    if decomposition_only:
        OUT=DECOMP/'independent_decomposition_audit'
    start=time.monotonic()
    ds=read(DECOMP/'summary.json');ps=read(POOL/'summary.json')
    need(ds['status']=='ALL24_COMPLETE' and ds['completed_queries']==24,'DECOMPOSITION_NOT_COMPLETE')
    if not decomposition_only:
        need(ps['completed_probe_queries']==8,'POOL_NOT_COMPLETE')
    dp=read(DECOMP/'protocol.json');pp=read(POOL/'protocol.json')
    for p in [dp,pp]:
        for s in p['sources']:checked(s)
    need(not (OUT/'validation.json').exists(),'NEW_AUDIT_REQUIRED')
    manifest=read(checked(next(s for s in dp['sources'] if s['path'].endswith('/input_manifest.json'))))
    allrows=manifest['train_rows']+manifest['probe_rows']
    original={r['query_id']:r for r in allrows}
    need(dp['row_order']==[r['query_id'] for r in allrows],'DECOMP_MANIFEST_AXIS')
    need(pp['pool_query_ids']==[r['query_id'] for r in manifest['train_rows']],'POOL_TRAIN16')
    need(pp['probe_query_ids']==[r['query_id'] for r in manifest['probe_rows']],'POOL_PROBE8')
    need(not set(pp['pool_query_ids']) & set(pp['probe_query_ids']),'NO_POOL_PROBE_OVERLAP')
    cp=torch.load(checked(next(s for s in dp['sources'] if s['path'].endswith('/0128.pt'))),
        map_location='cpu',weights_only=True)
    need(cp['arm']=='POST_REAL' and cp['step']==128,'POST_REAL128')
    theta=cp['head'].double().tolist()
    labels={r['query_id']:r for r in read(checked(ds['opened_label_source']))['rows']}
    need(ds['opened_label_source']==ps['opened_label_source'],'SAME_LABEL_SOURCE')
    drecords={r['query_id']:r for r in [read(checked(s)) for s in ds['source_results']]}
    precords={} if decomposition_only else {r['query_id']:r for r in [read(checked(s)) for s in ps['result_bindings']]}
    need(set(drecords)==set(original),'ALL24_EXACT')
    if not decomposition_only:
        need(set(pp['probe_query_ids'])<=set(precords),'ALL8_POOL')
    maxima={'candidate_L':0.,'logit':0.,'NPZ_recomposed_patch':0.,'recomposed_L':0.,'recomposed_logit':0.,
        'energy':0.,'CPU_GPU_L':0.,'CPU_GPU_logit':0.,'argmax_fraction':0.,'pool_direction_curve':0.}
    diagnostics=[];pool_diagnostics=[];candidate_count=0;logit_count=0
    decomposition_by_candidate={}
    for q in dp['row_order']:
        row=original[q];r=drecords[q]
        need(r['protocol']==bind(DECOMP/'protocol.json'),'DROW_PROTOCOL')
        need(r['candidate_ids']==row['candidate_ids'] and r['M']==row['M'] and r['raw_scores']==row['raw_scores'],'DROW_SOURCE')
        checked(r['cache_binding'])
        need(len(r['candidate_results'])==128,'C128_COMPLETE')
        byarm={a:[] for a in dp['score_arms']}
        candidates=[]
        for i,binding in enumerate(r['candidate_results']):
            c=read(checked(binding));candidate_count+=1
            need(c['protocol']==r['protocol'] and c['query_id']==q and c['candidate_position']==i,'CANDIDATE_BINDING')
            need(c['candidate_id']==row['candidate_ids'][i] and c['M']==row['M'][i],'CANDIDATE_SOURCE')
            need(c['reference_binding']==row['reference_tokens'][i],'REFERENCE_SOURCE_BINDING')
            checked(c['reference_binding'])
            z=arrays(c['patch_evidence']);arms=z['token_arms'].tolist()
            need(arms==dp['token_arms'],'TOKEN_ARM_AXIS')
            best=z['maxsim_by_patch'];locations=z['best_reference_token'];constantloc=locations[0]
            need(best.shape==(5,r['query_patch_count']),'PATCH_AXIS')
            score={a:mean(best[j]) for j,a in enumerate(arms)}
            score['NATIVE_FIXED_ARGMAX']=mean(z['fixed_native_by_patch'])
            score['COMMON_FIXED_ARGMAX']=mean(z['fixed_common_by_patch'])
            for arm,L in score.items():
                maxima['candidate_L']=max(maxima['candidate_L'],close(L,c['scores'][arm],1e-12,'PATCH_MEAN_L'))
                byarm[arm].append(L)
            changes={a:float(np.mean(locations[j]!=constantloc)) for j,a in enumerate(arms)}
            for arm,change in changes.items():
                maxima['argmax_fraction']=max(maxima['argmax_fraction'],close(change,c['token_and_matching_stats'][arm]['argmax_changed_fraction_vs_constant'],1e-14,'ARGMAX_FRACTION'))
            native_gain=best[1]-z['fixed_native_by_patch'];common_gain=best[2]-z['fixed_common_by_patch']
            need(min(native_gain.min(),common_gain.min())>=-1e-12,'MAXSIM_DOMINATES_OLD_ARGMAX')
            close(mean(native_gain),c['assignment_gain']['native_mean'],1e-12,'NATIVE_ASSIGNMENT_GAIN')
            close(mean(common_gain),c['assignment_gain']['common_mean'],1e-12,'COMMON_ASSIGNMENT_GAIN')
            total=mean(z['M_specific_hidden_delta_norm']**2)
            spatial=mean(z['M_specific_hidden_spatial_norm']**2)
            common=float(np.square(z['common_hidden_vector'].astype(np.float64)).sum())
            rd=c['residual_decomposition']
            for value,key in [(total,'actual_BF16_delta_energy'),(spatial,'spatial_energy'),(common,'common_energy')]:
                maxima['energy']=max(maxima['energy'],close(value,rd[key],max(1e-9,abs(value)*1e-12),'RESIDUAL_ENERGY'))
            need(abs(total-spatial-common)<max(1e-9,total*1e-5),'ORTHOGONAL_ENERGY_CLOSURE')
            need(rd['recomposed_projection_independently_evaluated'],'RECOMPOSED_NOT_DECLARED_INDEPENDENT')
            need(rd['BF16_recomposed_max_abs_error']<=dp['recomposition_limits']['hidden_max_abs'],'RECOMPOSED_HIDDEN_RECORDED')
            need(rd['recomposed_token_max_abs_error']<=dp['recomposition_limits']['token_max_abs'],'RECOMPOSED_TOKEN_RECORDED')
            patch_err=float(np.max(np.abs(best[4]-best[1])))
            maxima['NPZ_recomposed_patch']=max(maxima['NPZ_recomposed_patch'],patch_err)
            maxima['recomposed_L']=max(maxima['recomposed_L'],close(score['RECOMPOSED'],score['NATIVE'],dp['recomposition_limits']['L_max_abs'],'RECOMPOSED_L'))
            detail={'candidate_position':i,'candidate_id':c['candidate_id'],'scores':score,
                'M':c['M'],'argmax_changed_fraction':changes,
                'native_free_minus_fixed':mean(native_gain),'common_free_minus_fixed':mean(common_gain),
                'common_energy_share':common/total if total>0 else None,
                'recomposed_patch_score_max_error':patch_err}
            candidates.append(detail)
            decomposition_by_candidate[(q,i)]={'constant_best_reference_token':constantloc,
                'common_hidden_vector':z['common_hidden_vector'],'candidate':c,'detail':detail}
        actions={}
        for arm,L in byarm.items():
            maxima['candidate_L']=max(maxima['candidate_L'],close(L,r['arms'][arm]['L'],1e-12,'ROW_L_FROM_PATCHES'))
            action,err=actioncheck(row,L,theta,r['arms'][arm]['decision'])
            maxima['logit']=max(maxima['logit'],err);logit_count+=127
            actions[arm]={'L':L,'action':action}
        ne=close(actions['RECOMPOSED']['action']['logits'],actions['NATIVE']['action']['logits'],dp['recomposition_limits']['logit_max_abs'],'RECOMPOSED_LOGITS')
        maxima['recomposed_logit']=max(maxima['recomposed_logit'],ne)
        need(actions['RECOMPOSED']['action']['prediction_position']==actions['NATIVE']['action']['prediction_position'],'RECOMPOSED_ACTION')
        anchors={}
        for arm,ab in r['cpu_gpu_anchors'].items():
            gpu=read(checked(ab['source']))
            errL=float(np.max(np.abs(np.asarray(actions[arm]['L'])-gpu['L'])))
            errz=float(np.max(np.abs(np.asarray(actions[arm]['action']['logits'])-gpu['decision']['logits'])))
            need(actions[arm]['action']['prediction_position']==gpu['decision']['prediction_position'],'CPU_GPU_ACTION')
            maxima['CPU_GPU_L']=max(maxima['CPU_GPU_L'],errL);maxima['CPU_GPU_logit']=max(maxima['CPU_GPU_logit'],errz)
            close(errL,ab['max_L_abs_error'],2e-12,'CPU_GPU_ANCHOR_L')
            close(errz,ab['max_logit_abs_error'],2e-10,'CPU_GPU_ANCHOR_Z')
            anchors[arm]={'L_max_error':errL,'logit_max_error':errz,'same_position':True}
        diagnostics.append({'query_id':q,'split':r['split'],'arms':actions,'candidates':candidates,
            'CPU_GPU_anchors':anchors})
    poolval=read(POOL/'train_pool_validation.json')
    need(poolval['held_query_count']==0 and poolval['query_ids']==pp['pool_query_ids'],'POOL_NO_HELD')
    poolz=arrays(poolval['payload'])
    need(poolz['query_ids'].tolist()==pp['pool_query_ids'],'POOL_NPZ_QUERY_AXIS')
    need(np.array_equal(np.diff(poolz['offsets']),poolz['patch_counts']),'POOL_PATCH_OFFSETS')
    close(poolz['query_pool_weights'],np.ones(16)/16,0.,'EQUAL_QUERY_POOL')
    for source in poolval['source_cache_bindings']:checked(source)
    direct=read(POOL/'pool_direct_validation.json')
    need(direct['status']=='ALL16_ACTUAL_ADAPTER_UP_VS_NONLINEAR_LATENT_POOL_PASS','DIRECT_POOL_PRODUCER_CHECK')
    need([r['query_id'] for r in direct['query_checks']]==pp['pool_query_ids'],'DIRECT_POOL16')
    need(direct['held_queries']==0 and direct['pool_max_abs_error']<=1e-5,'DIRECT_POOL_ERROR')
    decomp_byq={x['query_id']:x for x in diagnostics}
    for q,r in precords.items():
        row=original[q]
        checked(r['source_decomposition']);checked(r['cache_binding']);checked(r['pool_binding'])
        need(r['protocol']==bind(POOL/'protocol.json'),'POOL_PROTOCOL')
        need(r['candidate_ids']==row['candidate_ids'] and r['M']==row['M'],'POOL_ROW_SOURCE')
        curves=arrays(r['curve_binding'])
        close(curves['M'],row['M'],0.,'CURVE_M')
        byarm={a:[] for a in pp['arms']};candidates=[]
        for i,binding in enumerate(r['candidate_results']):
            c=read(checked(binding));candidate_count+=1
            need(c['query_id']==q and c['candidate_position']==i and c['M']==row['M'][i],'POOL_CANDIDATE')
            checked(c['source_candidate'])
            z=arrays(c['patch_evidence']);old=decomposition_by_candidate[(q,i)]
            need(np.array_equal(z['constant_best_reference_token'],old['constant_best_reference_token']),'SAME_OLD_ARGMAX')
            close(z['actual_BF16_common_direction'],old['common_hidden_vector'],0.,'QUANTIZATION_BRIDGE_VECTOR')
            for key,curve in [('pool_direction','pool_direction'),('ideal_self_direction','self_direction'),
                              ('pool_latent','pool_latent_response'),('ideal_self_latent','self_latent_response')]:
                maxima['pool_direction_curve']=max(maxima['pool_direction_curve'],close(z[key],curves[curve][i],0.,'CURVE_CANDIDATE:'+key))
            arms=z['arms'].tolist();need(arms==pp['arms'][:2],'POOL_ARM_AXIS')
            score={};changes={}
            for j,arm in enumerate(arms):
                score[arm]=mean(z['maxsim_by_patch'][j])
                score[arm+'_FIXED_ARGMAX']=mean(z['fixed_argmax_by_patch'][j])
                need((z['maxsim_by_patch'][j]-z['fixed_argmax_by_patch'][j]).min()>=-1e-12,'POOL_MAXSIM_DOMINATES')
                changes[arm]=float(np.mean(z['best_reference_token'][j]!=z['constant_best_reference_token']))
                close(changes[arm],c['stats'][arm]['argmax_changed_fraction_vs_constant'],1e-14,'POOL_ARGMAX')
            for arm,L in score.items():
                maxima['candidate_L']=max(maxima['candidate_L'],close(L,c['scores'][arm],1e-12,'POOL_PATCH_MEAN'))
                byarm[arm].append(L)
            candidates.append({'candidate_position':i,'candidate_id':c['candidate_id'],
                'M':c['M'],'scores':score,'argmax_changed_fraction':changes,
                'pool_free_minus_fixed':score['TRAIN_POOL_COMMON']-score['TRAIN_POOL_COMMON_FIXED_ARGMAX'],
                'self_free_minus_fixed':score['IDEAL_SELF_COMMON']-score['IDEAL_SELF_COMMON_FIXED_ARGMAX'],
                'quantization_bridge':c['quantization_bridge']})
        aliases={'ACTUAL_COMMON_BRIDGE':'COMMON_ONLY','CPU_NATIVE_REFERENCE':'NATIVE','CPU_CONSTANT_REFERENCE':'CONSTANT'}
        for alias,key in aliases.items():byarm[alias]=decomp_byq[q]['arms'][key]['L']
        actions={}
        for arm,L in byarm.items():
            maxima['candidate_L']=max(maxima['candidate_L'],close(L,r['arms'][arm]['L'],1e-12,'POOL_ROW_L'))
            action,err=actioncheck(row,L,theta,r['arms'][arm]['decision']);logit_count+=127
            maxima['logit']=max(maxima['logit'],err);actions[arm]={'L':L,'action':action}
        pool_diagnostics.append({'query_id':q,'split':r['split'],'arms':actions,'candidates':candidates})
    dsumm={sp:row_summary([r for r in diagnostics if r['split']==sp],labels) for sp in ['train','probe']}
    psumm={sp:row_summary([r for r in pool_diagnostics if r['split']==sp],labels) for sp in ['probe','train_engineering']}
    for actual,saved in [(dsumm,ds['summary']),(psumm,ps['summary'])]:
        for sp,sub in actual.items():
            if not sub:continue
            for arm,v in sub.items():
                for k in ['n','correct']:need(v[k]==saved[sp][arm][k],'SUMMARY_COUNT')
                for k in ['rescues','breaks']:need(set(v[k])==set(saved[sp][arm][k]),'SUMMARY_CASES')
    cefrow=original[CEF];w=cefrow['winner_index'];targetpos=[i for i,x in enumerate(cefrow['candidate_identities']) if x==labels[CEF]['identity']]
    need(len(targetpos)==1,'CEFDINIR_UNIQUE_TARGET');t=targetpos[0];j=cefrow['challenger_positions'].index(t)
    cef={}
    for name,records in [('decomposition',diagnostics),('pool',pool_diagnostics)]:
        if not records:
            continue
        r=next(x for x in records if x['query_id']==CEF)
        details={}
        for arm,values in r['arms'].items():
            L=values['L'];details[arm]={'target_L':L[t],'RAWwinner_L':L[w],'target_minus_RAWwinner_L':L[t]-L[w],
                'target_logit':values['action']['logits'][j],'selected_identity':values['action']['prediction_identity'],
                'correct':values['action']['prediction_identity']==labels[CEF]['identity']}
        cef[name]={'target_position':t,'RAWwinner_position':w,'arms':details,
            'target_candidate':r['candidates'][t],'RAWwinner_candidate':r['candidates'][w]}
        base=details['CONSTANT' if name=='decomposition' else 'CPU_CONSTANT_REFERENCE']
        pairs=[('NATIVE','NATIVE_FIXED_ARGMAX'),('COMMON_ONLY','COMMON_FIXED_ARGMAX')] if name=='decomposition' else [
            ('TRAIN_POOL_COMMON','TRAIN_POOL_COMMON_FIXED_ARGMAX'),('IDEAL_SELF_COMMON','IDEAL_SELF_COMMON_FIXED_ARGMAX')]
        cef[name]['sequential_readout_contributions']={}
        for free,fixed in pairs:
            before=details[fixed];after=details[free]
            cef[name]['sequential_readout_contributions'][free]={
                'constant_L_difference':base['target_minus_RAWwinner_L'],
                'old_assignment_content_change_L_difference':before['target_minus_RAWwinner_L']-base['target_minus_RAWwinner_L'],
                'rematching_L_difference_increment':after['target_minus_RAWwinner_L']-before['target_minus_RAWwinner_L'],
                'constant_target_logit':base['target_logit'],
                'old_assignment_content_change_target_logit':before['target_logit']-base['target_logit'],
                'rematching_target_logit_increment':after['target_logit']-before['target_logit'],
                'final_target_logit':after['target_logit'],
                'note':'Sequential conditional contrasts; not order-independent causal contribution percentages.'}
    result={'status':'INDEPENDENT_POST_PATCH_SCORE_ACTION_DECOMPOSITION_ONLY_PASS' if decomposition_only else 'INDEPENDENT_POST_PATCH_SCORE_ACTION_DECOMPOSITION_POOL_PASS',
        'decomposition_only':decomposition_only,
        'program':bind(__file__),'decomposition_summary':bind(DECOMP/'summary.json'),
        'pool_summary':bind(POOL/'summary.json'),'source_bindings_checked':len(MEMO),
        'candidate_payloads_checked':candidate_count,'logits_recomputed':logit_count,
        'maximum_errors':maxima,'decomposition_counts':dsumm,'pool_counts':psumm,
        'cefdinir':cef,'decomposition_rows':diagnostics,'pool_rows':pool_diagnostics,
        'head':theta,'new_training':False,'new_model_forwards':0,
        'limits':['Independent replay starts at saved per-patch MaxSim/fixed-assignment outputs; it does not rerun projection or MaxSim matrices.',
                  'RECOMPOSED saved patch outputs, candidate scores and decisions are independently compared; hidden/token error bounds are checked producer measurements, not fresh independent full tensor recomputation.',
                  'POOL source membership and preserved nonlinear latent/candidate outputs are checked; the original 16-query adapter-hook calculation is a producer validation, not rerun here.',
                  'TRAIN16 and repeatedly opened PROBE8; descriptive interventions, not new independent accuracy or optimal model selection.',
                  'A shared hidden direction can yield patch-dependent rotations after normalization; shared direction is not a uniform final score or proof of better semantic attention.'],
        'seconds':time.monotonic()-start}
    write(OUT/'result.json',result)
    lines=['# POST残差与TRAIN共享方向：独立逐patch输出/决策核算','',
        f"核对 {candidate_count} 个完整候选证据文件、重算 {logit_count} 个logit；最大L误差 {maxima['candidate_L']:.3g}，最大logit误差 {maxima['logit']:.3g}。",'',
        '|分支|臂|正确/n|救回|损失|','|---|---|---:|---:|---:|']
    for group,summ in [('拆分',dsumm),('共享TRAIN方向',psumm)]:
        for sp,sub in summ.items():
            for arm,v in sub.items():lines.append(f"|{group}/{sp}|{arm}|{v['correct']}/{v['n']}|{len(v['rescues'])}|{len(v['breaks'])}|")
    lines+=['','## cefdinir-fig2','', '|分支|臂|target−RAWwinner L|target logit|正确|','|---|---|---:|---:|---|']
    for name,v in cef.items():
        for arm,c in v['arms'].items():lines.append(f"|{name}|{arm}|{c['target_minus_RAWwinner_L']:.9f}|{c['target_logit']:.9f}|{c['correct']}|")
    lines+=['','## 核算范围','']+[f'- {x}' for x in result['limits']]
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'validation.json',{'status':result['status'],'result':bind(OUT/'result.json'),
        'report':bind(OUT/'report_zh.md'),'seconds':time.monotonic()-start})
    print(json.dumps({'status':result['status'],'maximum_errors':maxima,
        'decomposition_counts':dsumm,'pool_counts':psumm,'cefdinir':{k:v['arms'] for k,v in cef.items()},
        'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--decomposition-only',action='store_true')
    args=parser.parse_args()
    main(args.decomposition_only)
