#!/usr/bin/env python3
"""Matched POST versus sealed V4 PRE; frozen hidden cache, no LLM forward."""
from __future__ import annotations
import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import run_rc_internal_m_condition_scale_v4 as V
import run_rc_internal_m_v4_probe_v1 as P

ROOT = V.ROOT
OUT = ROOT / 'results/rc_postllm_m_v1'
AUTH = ROOT / 'registry/rc_postllm_m_v1_authority_20260924.json'
PLAN = ROOT / 'plan/RC_POSTLLM_M_V1_EXECUTION_20260924.md'
LAUNCH = ROOT / 'slurm/rc_postllm_m_v1.sbatch'
REPORT = ROOT / 'reports/REPORT_POSTLLM_M_V1_20260924.md'
ARMS = ('POST_REAL', 'POST_CONSTANT', 'POST_SHUFFLED')
read, write, bind, checked, need, save, emit = V.read, V.write, V.bind, V.checked, V.need, V.save, V.emit
OLD = V.OLD
FEATURES = V.FEATURES
state_cpu, tree_equal = V.state_cpu, V.tree_equal
cpu_optimizer_state, make_optimizer = V.cpu_optimizer_state, V.make_optimizer
choose, summary, validate_pending = V.choose, V.summary, V.validate_pending
joint_derivatives = V.joint_derivatives
cached_row = V.cached_row


def no_probe_labels():
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = str(Path(os.fsdecode(args[0])).resolve()).lower()
        need(not any(s in p for s in ('curator_roles', '/target_join/', 'd1-mi', 'd1_mi', 'formal392')), 'PROTECTED_LABEL_FILE')
        need(p not in (str(P.OUT/'result.json').lower(), str(V.V2/'result.json').lower()), 'NO_OPENED_PROBE_LABELS_IN_WORKER')
    sys.addaudithook(audit)


def prepare():
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    no_probe_labels()
    parent = read(V.AUTH)
    V.check_sources(parent)
    m = read(checked(parent['manifest']))
    need(len(m['train_rows']) == 16 and len(m['probe_rows']) == 8, 'SAME16_8')
    need(not any(k in r for r in m['probe_rows'] for k in ('target_id', 'identity', 'target_positions', 'component')), 'LABEL_FREE_PROBE')
    caches = []
    for row in m['train_rows'] + m['probe_rows']:
        d = V.V2/'encoder_cache'/row['query_id']
        seal = read(d/'validation.json')
        need(seal['authority'] == bind(V.V2_AUTH), 'SEALED_ENCODER_SOURCE')
        checked(seal['payload']); checked(seal['parity'])
        caches += [bind(d/'validation.json'), seal['payload'], seal['parity']]
    a = {k: parent[k] for k in V.CONFIG}
    a.update(status='POSTLLM_FIXED_LOCATION_COMPARISON_AUTHORIZED',
        user_authorization='2026-09-24 现在做后注入', parent=bind(V.AUTH),
        manifest=parent['manifest'], model=parent['model'], baseline_initial=parent['baseline_initial'],
        model_source_validation=parent['model_source_validation'], cache_bindings=caches,
        warm_head=bind(V.OUT/'cpu/warmstart_INTERNAL3.json'),
        external_heads={k: bind(V.OUT/'cpu'/f'{k}.json') for k in ('ADDITIVE4', 'PRODUCT5')},
        probe_label_source=read(P.OUT/'validation.json')['result'],
        pre_train_result=bind(V.OUT/'result.json'), pre_probe_validation=bind(P.OUT/'validation.json'),
        changed_factor='Adapter moves from pre-LLM visual merger tokens to post-LLM/pre-retrieval-projection image hidden states',
        arms=list(ARMS), direct_M_in_head=False, vision_or_llm_or_roma_forwards=0,
        probe_policy='Fixed128 then opened PROBE8, no checkpoint or hyperparameter selection; post-hoc development only',
        worker_budget=400, max_requeues=48, initial_partition='dev_accelerated,accelerated',
        projection_parity='Exact BF16 original projector including frozen LoRA; fail closed on native-token or full-C128 zero-adapter mismatch',
        projection_cpu_conformance=bind(ROOT/'results/rc_internal_m_v4_probe_v1/adapter_cpu_audit/post_backend_cpu_validation.json'),
        readout_refit='Same warm INTERNAL3; frozen128 endpoint;2000 fullTRAIN16 updates; fixed and secondary',
        code_sources=[bind(p) for p in (Path(__file__), PLAN, LAUNCH,
            ROOT/'programs/rc_postllm_m_backend_v1.py', ROOT/'tests/test_postllm_m_v1.py')])
    write(AUTH, a)
    warm = read(checked(a['warm_head']))
    write(OUT/'cpu/warmstart_INTERNAL3.json', warm)
    write(OUT/'cpu/validation.json', dict(status='TRAIN16_COMMON_WARM_AND_EXTERNAL_HEADS_PASS',
        authority=bind(AUTH), heads=[bind(OUT/'cpu/warmstart_INTERNAL3.json')], warm_source=a['warm_head'], new_training=0))
    emit(stage='prepare', status=a['status'], authority=bind(AUTH))


def guard():
    a = read(AUTH)
    for b in a['code_sources'] + a['cache_bindings']:
        checked(b)
    parent = read(checked(a['parent']))
    V.check_sources(parent)
    for k in ('warm_head', 'baseline_initial', 'pre_train_result', 'pre_probe_validation', 'projection_cpu_conformance'):
        checked(a[k])
    for b in a['external_heads'].values():
        checked(b)
    m = read(checked(a['manifest']))
    need({r['query_id'] for r in m['train_rows']}.isdisjoint(r['query_id'] for r in m['probe_rows']), 'DISJOINT_QUERY_SPLIT')
    return a, m


def load_model(a):
    from rc_postllm_m_backend_v1 import load_projection
    model = load_projection(a, device='cuda')
    write(OUT/'loading'/f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json",
        dict(authority=bind(AUTH), report=model.report))
    return model


def create_adapter(a, m, arm):
    import torch
    from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter
    torch.manual_seed(a['seed'])
    small = ScaledQualityResidualAdapter(hidden_size=3584, bottleneck=a['bottleneck'],
        conditioning='constant' if arm == 'POST_CONSTANT' else 'real', condition_gain=a['condition_gain'],
        mass_log_mean=m['mass_normalization']['log_mean'], mass_log_std=m['mass_normalization']['log_std'],
        mass_epsilon=m['mass_normalization']['epsilon'], residual_scale=a['residual_scale'])
    initial = torch.load(checked(a['baseline_initial']), map_location='cpu', weights_only=True)
    need(tree_equal(initial['adapter'], state_cpu(small)), 'EXACT_PRE_POST_INITIAL_STATE')
    return small.to('cuda')


def predict(model, cache, small, mass):
    from rc_postllm_m_backend_v1 import predict_projection
    return predict_projection(model, cache, small, mass)


def projection_preflight(a, m):
    """All24 zero-adapter token parity before any optimization."""
    import torch
    model = load_model(a)
    small = create_adapter(a, m, 'POST_REAL')
    records = []
    for row in m['train_rows'] + m['probe_rows']:
        cache = cached_row(row, 'cuda')
        native = cache['native_tokens'][cache['image_mask']]
        with torch.no_grad():
            for mass in (min(row['M']), max(row['M'])):
                new = predict(model, cache, small, mass)
                error = float((new-native).abs().max())
                need(error <= 1e-6, 'POST_ZERO_NATIVE_TOKEN_PARITY')
                records.append(dict(query_id=row['query_id'], mass=mass, token_error=error, patch_count=len(new)))
        del cache
    write(OUT/'projection_validation.json', dict(status='ALL24_POST_PROJECTION_ZERO_ADAPTER_PASS',
        authority=bind(AUTH), checks=records, model_report=model.report, held_label_reads=0,
        llm_forward_calls=0, vision_forward_calls=0, roma_forward_calls=0))
    del model, small
    torch.cuda.empty_cache()


def probe(a, m, arm, budget):
    import torch
    started = time.monotonic()
    sealpath = OUT/arm/'probe_validation.json'
    if sealpath.exists():
        seal = read(sealpath); need(seal['authority'] == bind(AUTH), 'PROBE_AUTHORITY')
        for b in seal['predictions']: checked(b)
        return True
    fit = read(OUT/arm/'fit_validation.json')
    need(fit['steps'] == 128 and fit['authority'] == bind(AUTH), 'FROZEN128_BEFORE_PROBE')
    snapshot = bind(OUT/arm/'snapshots/0128.pt'); need(snapshot in fit['snapshots'], 'TERMINAL_SNAPSHOT')
    saved = torch.load(checked(snapshot), map_location='cpu', weights_only=True)
    model = load_model(a); small = create_adapter(a, m, arm)
    small.load_state_dict(saved['adapter']); small.eval()
    modes = ['native', 'constant', 'shuffled'] if arm == 'POST_REAL' else ['native']
    records = []
    for row in m['probe_rows']:
        cache = refs = None
        for mode in modes:
            p = OUT/arm/'probe'/mode/(row['query_id']+'.json')
            if p.exists():
                old = read(p)
                need(old['snapshot'] == snapshot and old['authority'] == bind(AUTH)
                     and old['candidate_ids'] == row['candidate_ids'], 'PROBE_PREDICTION_RESUME')
                records.append(bind(p)); continue
            if time.monotonic()-started > budget-25: return False
            if cache is None:
                cache=cached_row(row,'cuda');refs=OLD.references(row)
            partial=p.with_suffix('.partial.json'); values=[]
            if partial.exists():
                old=read(partial)
                need(old['snapshot']==snapshot and old['authority']==bind(AUTH) and old['query_id']==row['query_id']
                     and old['intervention']==mode, 'PROBE_PARTIAL')
                values=old['L'];validate_pending(values)
            def persist(vals):
                write(partial,dict(authority=bind(AUTH),snapshot=snapshot,query_id=row['query_id'],intervention=mode,L=vals))
            if not score_all(model,cache,small,row,refs,values,started,budget,persist,mode): return False
            d=choose(row,values,saved['head'])
            write(p,dict(authority=bind(AUTH),snapshot=snapshot,query_id=row['query_id'],arm=arm,intervention=mode,
                candidate_ids=row['candidate_ids'],raw_scores=row['raw_scores'],M=row['M'],L=values,decision=d,
                direct_M_in_head=False,training_updates=0,held_label_reads=0))
            records.append(bind(p));emit(stage='probe_prediction',arm=arm,intervention=mode,query_id=row['query_id'])
        del cache,refs;torch.cuda.empty_cache()
    write(sealpath,dict(status='POST_LABEL_FREE_PROBE_COMPLETE',authority=bind(AUTH),snapshot=snapshot,predictions=records))
    return True


def refit(a, m, budget):
    import torch
    started=time.monotonic()
    allpred=[];maximum_error=0.
    warm=read(checked(a['warm_head']))['theta']
    for arm in ARMS:
        fitseal=read(OUT/arm/'fit_validation.json'); probeseal=read(OUT/arm/'probe_validation.json')
        need(fitseal['authority']==probeseal['authority']==bind(AUTH), 'SEALED_ARMS')
        for b in fitseal['endpoints']:
            for p in read(checked(b))['predictions']:checked(p)
        for b in probeseal['predictions']:checked(b)
        destination=OUT/'readout_refits'/(arm+'.json')
        if destination.exists():
            need(read(destination)['authority']==bind(AUTH),'REFIT_AUTHORITY')
        else:
            if time.monotonic()-started>budget-60:return False
            values={r['query_id']:torch.tensor(read(OUT/arm/'endpoints/0128/native'/(r['query_id']+'.json'))['L'],dtype=torch.float64) for r in m['train_rows']}
            fitted=V.fit_cpu_head(m['train_rows'],values,warm,'INTERNAL3',2000,a)
            fitted.update(authority=bind(AUTH),kind='INTERNAL3',direct_M_in_head=False)
            write(destination,fitted)
    # Seal independent NumPy scores for train and probe before opening labels.
    import numpy as np
    for split in ('train','probe'):
        for row in m[split+'_rows']:
            decisions={}
            for arm in ARMS:
                modes=['native','constant','shuffled'] if arm=='POST_REAL' else ['native']
                for mode in modes:
                    folder=OUT/arm/('endpoints/0128' if split=='train' else 'probe')/mode
                    rec=read(folder/(row['query_id']+'.json'))
                    name=arm if mode=='native' else arm+'_'+mode.upper()
                    native=P.numpy_decision(row,rec['L'],rec['decision']['theta'])
                    error=float(np.max(np.abs(np.asarray(native['logits'])-rec['decision']['logits'])))
                    maximum_error=max(maximum_error,error);need(error<1e-10,'INDEPENDENT_NUMPY_LOGITS')
                    need(native['prediction_identity']==rec['decision']['prediction_identity'] and native['switched']==rec['decision']['switched'],'NUMPY_ACTION')
                    decisions[name]=native
                    head=read(OUT/'readout_refits'/(arm+'.json'))
                    decisions[name+'_REFIT']=P.numpy_decision(row,rec['L'],head['theta'])
            L0=read(V.V2/'encoder_cache'/row['query_id']/'parity.json')['fresh_L0']
            for kind,b in a['external_heads'].items():
                decisions['EXTERNAL_'+kind]=P.numpy_decision(row,L0,read(checked(b))['theta'],kind)
            dest=OUT/'audited_decisions'/split/(row['query_id']+'.json')
            write(dest,dict(authority=bind(AUTH),query_id=row['query_id'],decisions=decisions,held_label_reads=0))
            allpred.append(bind(dest))
    write(OUT/'prelabel_validation.json',dict(status='POST_ALL_PREDICTIONS_NUMPY_REPLAY_PASS',authority=bind(AUTH),
        decisions=allpred,refits=[bind(OUT/'readout_refits'/(arm+'.json')) for arm in ARMS],maximum_logit_error=maximum_error,held_label_reads=0))
    return True


def join(a, m):
    seal=read(OUT/'prelabel_validation.json');need(seal['authority']==bind(AUTH),'JOIN_AUTHORITY')
    for b in seal['decisions']+seal['refits']:checked(b)
    labels={r['query_id']:r for r in read(checked(a['probe_label_source']))['rows']}
    rows=[]
    for split in ('train','probe'):
        for r in m[split+'_rows']:
            q=r['query_id']; identity=r['target_id'] if split=='train' else labels[q]['identity']
            decisions=read(OUT/'audited_decisions'/split/(q+'.json'))['decisions']
            selected={'RAW':r['candidate_identities'][r['winner_index']],**{k:v['prediction_identity'] for k,v in decisions.items()}}
            rows.append(dict(query_id=q,split=split,identity=identity,component=r['component'] if split=='train' else labels[q]['component'],
                target_in_C128=identity in r['candidate_identities'],selected=selected,correct={k:v==identity for k,v in selected.items()}))
    totals={}
    for split in ('train','probe'):
        rr=[r for r in rows if r['split']==split]; totals[split]={}
        for k in rr[0]['correct']:
            totals[split][k]=dict(correct=sum(r['correct'][k] for r in rr),n=len(rr),
                rescues=[r['query_id'] for r in rr if r['correct'][k] and not r['correct']['RAW']],
                breaks=[r['query_id'] for r in rr if not r['correct'][k] and r['correct']['RAW']])
    pre=read(checked(a['pre_train_result']));preprobe=read(checked(a['probe_label_source']))
    comparisons=[]
    for post,pretrain,preheld in (
        ('POST_REAL',pre['internal']['PRE_REAL']['endpoints']['128']['native'],'REAL'),
        ('POST_CONSTANT',pre['reused_gain1_and_constant']['PRE_CONSTANT']['endpoints']['128']['native'],'TRAIN_CONSTANT'),
        ('POST_SHUFFLED',pre['internal']['PRE_SHUFFLED']['endpoints']['128']['native'],'TRAIN_SHUFFLED'),
        ('POST_REAL_CONSTANT',pre['internal']['PRE_REAL']['endpoints']['128']['constant'],'REAL_CONSTANT'),
        ('POST_REAL_SHUFFLED',pre['internal']['PRE_REAL']['endpoints']['128']['shuffled'],'REAL_SHUFFLED'),
        ('POST_REAL_REFIT',pre['fixed_endpoint_readout_refits']['GAIN_REAL']['summary'],'REAL_REFIT'),
        ('POST_CONSTANT_REFIT',pre['fixed_endpoint_readout_refits']['V3_CONSTANT']['summary'],'TRAIN_CONSTANT_REFIT'),
        ('POST_SHUFFLED_REFIT',pre['fixed_endpoint_readout_refits']['GAIN_TRAIN_SHUFFLED']['summary'],'TRAIN_SHUFFLED_REFIT')):
        comparisons.append(dict(post=post,pre_held_arm=preheld,pre_train=pretrain['correct'],post_train=totals['train'][post]['correct'],
            pre_probe=preprobe['summary'][preheld]['correct'],post_probe=totals['probe'][post]['correct']))
    result=dict(status='MATCHED_POSTLLM_LOCATION_EXPERIMENT_COMPLETE',authority=bind(AUTH),rows=rows,summary=totals,
        position_comparison=comparisons,
        pre_train_reference=pre['internal'],pre_probe_reference=preprobe['summary'],
        pre_constant_reference=pre['reused_gain1_and_constant']['PRE_CONSTANT'],
        pre_train_refits=pre['fixed_endpoint_readout_refits'],
        pre_post_refit_mapping={'POST_REAL_REFIT':'REAL_REFIT',
            'POST_REAL_CONSTANT_REFIT':'REAL_CONSTANT_REAL_REFIT_HEAD',
            'POST_REAL_SHUFFLED_REFIT':'REAL_SHUFFLED_REAL_REFIT_HEAD',
            'POST_CONSTANT_REFIT':'TRAIN_CONSTANT_REFIT','POST_SHUFFLED_REFIT':'TRAIN_SHUFFLED_REFIT'},
        evidence=a['probe_policy'],label_source=a['probe_label_source'],max_numpy_error=seal['maximum_logit_error'],
        llm_or_roma_forwards=0,prelabel_validation=bind(OUT/'prelabel_validation.json'))
    write(OUT/'result.json',result)
    lines=['# LLM后、检索投影前的M注入：固定位置对照','',a['probe_policy'],
        '同一TRAIN16、自然ColNomic C128、初始化、128次逐query完整C128 COST1；仅移动适配器位置。末端不直接读取M。',
        '全部LLM隐藏特征复用已有24图缓存，投影保留原冻结LoRA和BF16计算。', '',
        '| 路径 | TRAIN16 | PROBE8 | probe救回/误伤 |','|---|---:|---:|---:|']
    for k in totals['train']:
        t,p=totals['train'][k],totals['probe'][k]
        lines.append(f"| {k} | {t['correct']}/16 | {p['correct']}/8 | {len(p['rescues'])}/{len(p['breaks'])} |")
    lines+=['','## 与封存PRE的同协议位置比较','','| 配置 | PRE TRAIN | POST TRAIN | PRE probe | POST probe |',
        '|---|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {c['post']} | {c['pre_train']}/16 | {c['post_train']}/16 | {c['pre_probe']}/8 | {c['post_probe']}/8 |")
    lines+=['','PRE V4完整对照引用封存结果；不得把早期V2的外部M通路6/8当作内部POST成功。',
        '固定终点refit仅使用TRAIN标签，是预定次级诊断；未按probe选择头或阈值。',
        '本结果只回答这套适配器、训练预算和已打开开发probe上的位置差异，不证明普遍最优注入点。','']
    REPORT.write_text('\n'.join(lines))
    write(OUT/'validation.json',dict(status='POST_SOURCE_NUMPY_IDENTITY_JOIN_PASS',authority=bind(AUTH),
        result=bind(OUT/'result.json'),report=bind(REPORT)))
    emit(stage='join',status=result['status'],summary=totals)


def submit_job(stage, arm='', dependencies=()):
    cmd=['sbatch','--parsable']
    if stage in ('refit','join'):
        cmd+=['--partition=cpuonly,dev_cpuonly','--gres=none','--mem=16G']
    if dependencies:cmd+=['--dependency=afterok:'+':'.join(dependencies)]
    cmd += [str(LAUNCH),stage,arm]
    result=subprocess.run(cmd,check=True,capture_output=True,text=True)
    return result.stdout.strip().split(';')[0],cmd


def submit():
    a,m=guard();d=OUT/'dispatch';d.mkdir(parents=True,exist_ok=True)
    lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    path=d/'pilot.json'
    if path.exists():print(read(path));return
    job,cmd=submit_job('pilot','POST_REAL')
    write(path,dict(job_id=job,command=cmd,authority=bind(AUTH)));print(job,flush=True)


def advance():
    a,m=guard();need(read(OUT/'pilot_validation.json')['status']=='ACTUAL_MODEL_JOINT_UPDATE_RESUME_PASS','PILOT_FIRST')
    d=OUT/'dispatch';lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    jobs={}
    for arm in ARMS:
        p=d/(arm+'.json')
        if not p.exists():
            job,cmd=submit_job('worker',arm);write(p,dict(job_id=job,command=cmd,authority=bind(AUTH)))
        jobs[arm]=read(p)['job_id']
    p=d/'refit.json'
    if not p.exists():
        job,cmd=submit_job('refit',dependencies=tuple(jobs.values()));write(p,dict(job_id=job,command=cmd,authority=bind(AUTH)))
    p2=d/'join.json'
    if not p2.exists():
        job,cmd=submit_job('join',dependencies=(read(p)['job_id'],));write(p2,dict(job_id=job,command=cmd,authority=bind(AUTH)))
    print('POST_SUCCESSORS',jobs,'refit',read(p)['job_id'],'join',read(p2)['job_id'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','submit','advance','pilot','worker','refit','join'])
    parser.add_argument('--arm',default='POST_REAL');parser.add_argument('--budget',type=float,default=400)
    args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='submit':submit();return
    if args.stage=='advance':advance();return
    started=time.monotonic()
    if args.stage!='join':no_probe_labels()
    a,m=guard();need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');V.setup(a)
    if args.stage in ('pilot','worker'):
        if not (OUT/'projection_validation.json').exists():projection_preflight(a,m)
        need(read(OUT/'projection_validation.json')['authority']==bind(AUTH),'PROJECTION_AUTHORITY')
        mm=V.shuffled_manifest(m,a['condition_shift']) if args.arm=='POST_SHUFFLED' else m
        # The fixed permutation also applies to label-free probe candidate M.
        if args.arm=='POST_SHUFFLED':
            for row in mm['probe_rows']:row['M']=row['M'][1:]+row['M'][:1]
        remaining=args.budget-(time.monotonic()-started)
        if remaining<60:raise SystemExit(75)
        done=train(a,mm,args.arm,remaining,pilot=args.stage=='pilot')
        if done and args.stage=='worker':
            remaining=args.budget-(time.monotonic()-started)
            if remaining<40:raise SystemExit(75)
            done=probe(a,mm,args.arm,remaining)
    elif args.stage=='refit':done=refit(a,m,args.budget-(time.monotonic()-started))
    else:join(a,m);done=True
    if not done:raise SystemExit(75)


# The four lifecycle functions below are copied from the sealed V4 runner.
# Tests verify that only PRE/POST names, source tensor and trace field differ.


def score_all(model, cache, small, row, refs, values, started, budget, persist, intervention='native'):
    import torch
    need(intervention in ('native','constant','shuffled'), 'INTERVENTION')
    validate_pending(values)
    need(len(refs) == len(row['M']) == 128, 'SCORE_FULL_C128_AXIS')
    masses = row['M'][1:] + row['M'][:1] if intervention == 'shuffled' else row['M']
    original = small.conditioning
    constant = original == 'constant' or intervention == 'constant'
    if intervention == 'constant':
        small.conditioning = 'constant'
    try:
        with torch.no_grad():
            common = predict(model, cache, small, masses[0]) if constant and len(values) < 128 else None
            for i in range(len(values),128):
                if time.monotonic()-started > budget:
                    persist(values); return False
                output = common if constant else predict(model, cache, small, masses[i])
                values.append(float(OLD.content_score(output, refs[i])))
                if (i+1) % 16 == 0:
                    persist(values)
        persist(values)
        return True
    finally:
        small.conditioning = original


def evaluate_endpoint(a, m, arm, step, model, small, theta, started, budget, snapshot):
    """Checkpoint fixed; per-query complete and partial predictions bind its SHA."""
    import torch
    d = OUT / arm / 'endpoints' / f'{step:04d}'
    interventions = ['native']
    if arm == 'POST_REAL' and step in a['diagnostic_updates']:
        interventions += a['diagnostic_interventions']
    seal = d / 'validation.json'
    if seal.exists():
        v = read(seal); need(v['snapshot'] == snapshot and v['authority'] == bind(AUTH), 'ENDPOINT_RESUME')
        for b in v['predictions']:
            checked(b)
        return True
    all_predictions = []
    for row in m['train_rows']:
        cache = None; refs = None
        for intervention in interventions:
            p = d / intervention / (row['query_id'] + '.json')
            if p.exists():
                record = read(p)
                need(record['snapshot'] == snapshot and record['authority'] == bind(AUTH), 'PREDICTION_RESUME')
                all_predictions.append(bind(p)); continue
            if time.monotonic()-started > budget-20:
                return False
            if cache is None:
                cache = cached_row(row, 'cuda'); refs = OLD.references(row)
            partial = p.with_suffix('.partial.json')
            values = []
            if partial.exists():
                old = read(partial)
                need(old['snapshot'] == snapshot and old['authority'] == bind(AUTH)
                     and old['query_id'] == row['query_id'] and old['intervention'] == intervention,
                     'PARTIAL_SNAPSHOT_AXIS')
                values = old['L']; validate_pending(values)
            def persist(v):
                write(partial,dict(authority=bind(AUTH),snapshot=snapshot,query_id=row['query_id'],intervention=intervention,L=v))
            if not score_all(model,cache,small,row,refs,values,started,budget,persist,intervention):
                return False
            prediction = choose(row, values, theta.detach().cpu())
            write(p,dict(authority=bind(AUTH),snapshot=snapshot,step=step,arm=arm,intervention=intervention,
                         query_id=row['query_id'],candidate_ids=row['candidate_ids'],
                         L=values,M=row['M'],raw_scores=row['raw_scores'],decision=prediction,
                         direct_M_in_head=False,held_label_reads=0))
            all_predictions.append(bind(p))
            emit(stage='train_endpoint', arm=arm, step=step, intervention=intervention,
                 query_id=row['query_id'],loss=prediction['cost1'],correct=prediction['correct'])
        del refs,cache; torch.cuda.empty_cache()
    aggregate = {}
    for intervention in interventions:
        predictions = {r['query_id']: read(d/intervention/(r['query_id']+'.json'))['decision'] for r in m['train_rows']}
        aggregate[intervention] = summary(m['train_rows'],predictions)
    write(seal,dict(status='FULL_TRAIN16_ENDPOINT_PASS',authority=bind(AUTH),snapshot=snapshot,
                   step=step,arm=arm,predictions=all_predictions,summaries=aggregate,held_label_reads=0))
    emit(stage='train_endpoint_complete',arm=arm,step=step,summaries=aggregate)
    return True


def finish_pilot(a, arm, checkpoint_path, small, theta, optimizer, elapsed_seconds):
    """Idempotently seal a committed first update, including after preemption."""
    import torch
    cp = Path(checkpoint_path); d = cp.parent
    loaded = torch.load(cp, map_location='cpu', weights_only=True)
    need(loaded['authority'] == bind(AUTH) and loaded['arm'] == arm, 'PILOT_CHECKPOINT_LINEAGE')
    need(loaded['step'] == 1 and not loaded['pending'], 'ATOMIC_STEP_COMMIT')
    need(torch.equal(loaded['head'], theta.detach().cpu()), 'HEAD_RELOAD')
    for name, tensor in small.state_dict().items():
        need(torch.equal(loaded['adapter'][name], tensor.detach().cpu()), 'ADAPTER_RELOAD')
    need(tree_equal(loaded['optimizer'], cpu_optimizer_state(optimizer)), 'PILOT_OPTIMIZER_RELOAD')
    receipt = read(d / 'steps/0001.json')
    need(receipt['adapter_parameter_change_max'] > 0 and receipt['head_parameter_change_max'] > 0,
         'PILOT_BOTH_MODULES_CHANGED')
    need(receipt['gradient_forward_error'] < 2e-8, 'PILOT_FORWARD_PARITY')
    need(receipt['token_changed_fraction'] > 0, 'PILOT_TOKEN_CHANGE_VISIBLE')
    result = dict(status='ACTUAL_MODEL_JOINT_UPDATE_RESUME_PASS',
                  authority=bind(AUTH), arm=arm, step=1, checkpoint=bind(cp),
                  step_record=bind(d/'steps/0001.json'), full_candidates=128,
                  head_updates=True, adapter_updates=True, head_M_input=False,
                  resume_parameters_exact=True, resume_optimizer_exact=True,
                  opened_probe_evaluations=0, measured_seconds=elapsed_seconds)
    write(OUT / 'pilot_validation.json', result)
    return result


def train(a,m,arm,budget,pilot=False):
    import torch
    started = time.monotonic()
    d = OUT / arm; d.mkdir(parents=True,exist_ok=True)
    lock = (d/'worker.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    need(arm in ARMS and (not pilot or arm == 'POST_REAL'),'TRAIN_ARM')
    if (d/'fit_validation.json').exists():
        need(not pilot,'PILOT_AFTER_TRAIN')
        return True
    cpu_receipt = read(OUT/'cpu/validation.json')
    need(cpu_receipt['status']=='TRAIN16_COMMON_WARM_AND_EXTERNAL_HEADS_PASS'
         and cpu_receipt['authority']==bind(AUTH),'CPU_FIRST')
    for binding in cpu_receipt['heads']:
        checked(binding)
    if not pilot:
        need(read(OUT/'pilot_validation.json')['status']=='ACTUAL_MODEL_JOINT_UPDATE_RESUME_PASS','PILOT_REQUIRED')
    model = load_model(a); small = create_adapter(a,m,arm)
    warm_binding = bind(OUT/'cpu/warmstart_INTERNAL3.json'); warm = read(warm_binding['path'])
    need(warm_binding in cpu_receipt['heads'], 'SEALED_COMMON_WARM_HEAD')
    theta = torch.nn.Parameter(torch.tensor(warm['theta'],dtype=torch.float64,device='cuda'))
    optimizer = make_optimizer(small,theta,a)
    cp = d/'checkpoint.pt'; step=0; pending=[]; history=[]
    def checkpoint(values):
        validate_pending(values)
        save(cp,dict(authority=bind(AUTH),arm=arm,adapter=state_cpu(small),head=theta.detach().cpu().clone(),
                     optimizer=cpu_optimizer_state(optimizer),step=step,history=history,pending=list(values),
                     pending_query_id=m['train_rows'][step % 16]['query_id'] if step < a['updates'] else None,
                     warmstart=warm_binding,candidate_count=128,conditioning=small.conditioning))
    if cp.exists():
        saved = torch.load(cp,map_location='cpu',weights_only=True)
        need(saved['authority']==bind(AUTH) and saved['arm']==arm and saved['warmstart']==warm_binding,'CHECKPOINT_BINDING')
        small.load_state_dict(saved['adapter']); theta.data.copy_(saved['head'].to('cuda'))
        optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];pending=saved['pending'];history=saved['history']
        validate_pending(pending)
        need(0 <= step <= a['updates'] and len(history) == step, 'CHECKPOINT_STEP_HISTORY')
        expected_query = m['train_rows'][step % 16]['query_id'] if step < a['updates'] else None
        need(saved['pending_query_id'] == expected_query and saved['conditioning'] == small.conditioning,
             'CHECKPOINT_PENDING_QUERY_CONDITION')
        need(tree_equal(saved['optimizer'], cpu_optimizer_state(optimizer)), 'OPTIMIZER_RELOAD_EXACT')
    else:
        checkpoint([])
        save(d/'initial.pt',dict(authority=bind(AUTH),adapter=state_cpu(small),head=theta.detach().cpu().clone(),
                                warmstart=warm_binding,step=0,arm=arm))
    if pilot and step == 1:
        # A preemption can occur after the atomic update commit and before its
        # receipt. Recover that receipt without repeating or skipping an update.
        finish_pilot(a,arm,cp,small,theta,optimizer,time.monotonic()-started)
    target = 1 if pilot else a['updates']
    while True:
        if step in a['checkpoint_updates']:
            snap = d/'snapshots'/f'{step:04d}.pt'
            if not snap.exists():
                save(snap,dict(authority=bind(AUTH),adapter=state_cpu(small),head=theta.detach().cpu().clone(),
                               arm=arm,step=step,warmstart=warm_binding))
            frozen = torch.load(snap,map_location='cpu',weights_only=True)
            need(frozen['authority']==bind(AUTH) and frozen['arm']==arm and frozen['step']==step,
                 'SNAPSHOT_LINEAGE')
            need(tree_equal(frozen['adapter'],state_cpu(small))
                 and torch.equal(frozen['head'],theta.detach().cpu()),'SNAPSHOT_STATE_PARITY')
            if step in a['endpoint_updates'] and not evaluate_endpoint(a,m,arm,step,model,small,theta,started,budget,bind(snap)):
                return False
        if step >= target:
            break
        if time.monotonic()-started > budget-30:
            return False
        row=m['train_rows'][step % len(m['train_rows'])]
        cache=cached_row(row,'cuda');refs=OLD.references(row)
        score_started=time.monotonic(); resumed_candidates=len(pending)
        if step==0 and not pending:
            # Actual engineering parity: the zero residual starts at the exact
            # cached content and the same common head on both conditions.
            with torch.no_grad():
                encoded=predict(model,cache,small,row['M'][0])
                native=cache['native_tokens'][cache['image_mask']]
                need(float((encoded-native).abs().max())<=1e-6,'ZERO_ADAPTER_NATIVE_PARITY')
        if not score_all(model,cache,small,row,refs,pending,started,budget,checkpoint):
            return False
        scoring_seconds = time.monotonic()-score_started
        if step == 0:
            zero_content_error = float((torch.tensor(pending,dtype=torch.float64)-cache['fresh_L0'].cpu()).abs().max())
            need(zero_content_error < 2e-8,'ZERO_FULL128_FRESH_CONTENT_PARITY')
        content=torch.tensor(pending,dtype=torch.float64,device='cuda',requires_grad=True)
        value,derivative,head_gradient=joint_derivatives(row,content,theta)
        active=derivative.nonzero().flatten().tolist()
        need(active and torch.isfinite(derivative).all() and torch.isfinite(head_gradient).all(),'JOINT_GRAD_FINITE')
        if time.monotonic()-started > budget-35:
            checkpoint(pending);return False
        optimizer.zero_grad(set_to_none=True);theta.grad=head_gradient.clone()
        grad_started=time.monotonic();replay_error=0.;last_output=None
        if arm=='POST_CONSTANT':
            # One identical query representation contributes to all active
            # candidates; summing all terms preserves the full C128 gradient.
            last_output=predict(model,cache,small,row['M'][0])
            terms=[]
            for i in active:
                score=OLD.content_score(last_output,refs[i])
                replay_error=max(replay_error,abs(float(score.detach())-pending[i]))
                terms.append(score*derivative[i])
            torch.stack(terms).sum().backward()
        else:
            for i in active:
                last_output=predict(model,cache,small,row['M'][i])
                score=OLD.content_score(last_output,refs[i])
                replay_error=max(replay_error,abs(float(score.detach())-pending[i]))
                (score*derivative[i]).backward()
        need(replay_error < 2e-8,'NO_GRAD_VJP_FORWARD_PARITY')
        gradients=[p.grad for p in small.parameters() if p.grad is not None]
        need(gradients and all(torch.isfinite(g).all() for g in gradients),'ADAPTER_GRAD_FINITE')
        adapter_norm=float(torch.nn.utils.clip_grad_norm_(small.parameters(),a['clip_norm']))
        head_norm=float(torch.nn.utils.clip_grad_norm_([theta],a['clip_norm']))
        need(adapter_norm > 0,'ADAPTER_GRAD_NONZERO')
        need(not any(p.grad is not None for p in model.parameters()),'FROZEN_BACKBONE_GRAD')
        before=state_cpu(small);head_before=theta.detach().cpu().clone()
        prior_tokens=last_output.detach().clone()
        optimizer.step()
        need(bool(torch.isfinite(theta).all()) and all(bool(torch.isfinite(p).all()) for p in small.parameters()),
             'FINITE_UPDATED_PARAMETERS')
        adapter_change=max(float((small.state_dict()[k].detach().cpu()-v).abs().max()) for k,v in before.items())
        head_change=float((theta.detach().cpu()-head_before).abs().max())
        need(adapter_change > 0 and head_change > 0,'BOTH_MODULES_UPDATE')
        record=dict(step=step+1,query_id=row['query_id'],arm=arm,loss_before_update=float(value.detach()),
                    content=pending,derivative=derivative.cpu().tolist(),head_gradient=head_gradient.cpu().tolist(),
                    decision_before_update=choose(row,pending,head_before),head_before=head_before.tolist(),
                    head_after=theta.detach().cpu().tolist(),active_candidates=active,
                    adapter_gradient_norm=adapter_norm,head_gradient_norm=head_norm,
                    adapter_parameter_change_max=adapter_change,head_parameter_change_max=head_change,
                    gradient_forward_error=replay_error,scoring_seconds=scoring_seconds,
                    backward_seconds=time.monotonic()-grad_started,resumed_candidates=resumed_candidates,
                    backbone_trainable=0,direct_M_in_head=False,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if step==0 or step+1 in a['checkpoint_updates']:
            with torch.no_grad():
                updated=predict(model,cache,small,row['M'][active[-1]])
                delta=updated.float()-prior_tokens.float()
                record['token_max_change']=float(delta.abs().max())
                record['token_changed_fraction']=float((delta!=0).float().mean())
                source=cache['hidden'][cache['image_mask']];modulated=small(source,row['M'][active[-1]])
                record['postllm_residual_max']=float((modulated.float()-source.float()).abs().max())
                save(d/'traces'/f'{step+1:04d}.pt',dict(query_id=row['query_id'],candidate_position=active[-1],
                     M=row['M'][active[-1]],source=source.cpu(),modulated_source=modulated.cpu(),
                     before_tokens=prior_tokens.cpu(),after_tokens=updated.cpu(),image_mask=cache['image_mask'].cpu()))
        write(d/'steps'/f'{step+1:04d}.json',record)
        history.append({k:v for k,v in record.items() if k not in ('content','derivative','decision_before_update','head_gradient')})
        step+=1;pending=[];checkpoint([])
        if step==1 and pilot:
            finish_pilot(a,arm,cp,small,theta,optimizer,time.monotonic()-started)
        emit(stage='fit',arm=arm,step=step,total=a['updates'],loss=record['loss_before_update'],
             scoring_seconds=record['scoring_seconds'],adapter_gradient_norm=adapter_norm,head_gradient_norm=head_norm)
        del refs,cache,last_output,content,value,prior_tokens;torch.cuda.empty_cache()
    if not pilot:
        endpoint_bindings=[bind(d/'endpoints'/f'{u:04d}'/'validation.json') for u in a['endpoint_updates']]
        write(d/'fit_validation.json',dict(status='FIXED128_UPDATES_TRAIN_ENDPOINTS_COMPLETE',authority=bind(AUTH),
              arm=arm,steps=step,checkpoint=bind(cp),endpoints=endpoint_bindings,
              snapshots=[bind(d/'snapshots'/f'{u:04d}.pt') for u in a['checkpoint_updates']],
              head_trainable_parameters=3,adapter_parameters=sum(p.numel() for p in small.parameters()),
              head_features=FEATURES['INTERNAL3'],direct_M_in_head=False,held_label_reads=0,
              selection='Fixed terminal128; no model or epoch selection'))
    return True


if __name__ == '__main__':
    main()
