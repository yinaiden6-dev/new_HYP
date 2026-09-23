#!/usr/bin/env python3
"""Nested, grouped action calibration for two fixed challenger rankers."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import run_rc_h593_quadratic_rank_v1 as Q

ROOT = Q.ROOT
sys.path.insert(0, str(ROOT / 'src'))
read, write, bind, checked, need, hx = Q.read, Q.write, Q.bind, Q.checked, Q.need, Q.hx
OUT = ROOT / 'results/rc_h593_rank_action_v1'
AUTH = ROOT / 'registry/rc_h593_rank_action_authority_v1_20260921.json'
PLAN = ROOT / 'plan/RC_H593_RANK_ACTION_V1_20260921.md'
REPORT = ROOT / 'reports/REPORT_H593_RANK_ACTION_V1_20260921.md'
BASE = ROOT / 'results/rc_h593_quadratic_rank_v1'
GAP = ROOT / 'results/rc_h593_gap_curve_v1'
ARMS = ('RANK_ONLY6', 'DIAG12')
ACTION = {'RANK_ONLY6': 'LINEAR_ACTION3', 'DIAG12': 'DIAG_ACTION3'}
CONTROLS = ('COST1_FULL', 'CE_FULL', 'GAP_BIAS2')
MODELS = ('RAW',) + CONTROLS + tuple(ACTION.values())


def prepare():
    need(not AUTH.exists(), 'AUTHORITY_ALREADY_FROZEN')
    parent = ROOT / 'registry/rc_h593_quadratic_rank_authority_v1_20260921.json'
    prior = read(parent)
    for source in prior['code_sources'].values():
        checked(source)
    v = read(BASE / 'validation.json'); gv = read(GAP / 'validation.json')
    need(v['status'] == 'QUADRATIC_RANK_ALL_COUNTS_PASS' and v['authority'] == bind(parent), 'PARENT_VALIDATED')
    need(gv['status'] == 'GAP_CURVE_ALL_COUNTS_PASS', 'GAP_VALIDATED')
    checked(v['result']); checked(gv['result'])
    codes = dict(parent_helper=Path(Q.__file__), shared_io=Path(Q.U.__file__), conditional_helper=Path(Q.C.__file__),
        program=Path(__file__), plan=PLAN,
        core=ROOT/'src/rc_aslo_xf/h593_rank_action_v1.py', independent=ROOT/'programs/validate_rc_h593_rank_action_v1.py',
        rank_core=ROOT/'src/rc_aslo_xf/h593_conditional_rank_v1.py',
        quadratic_core=ROOT/'src/rc_aslo_xf/h593_quadratic_rank_v1.py',
        linear_independent=ROOT/'programs/validate_rc_h593_conditional_rank_v1.py',
        quadratic_independent=ROOT/'programs/validate_rc_h593_quadratic_rank_v1.py',
        feature_formula=ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',
        fit_launcher=ROOT/'slurm/rc_h593_rank_action_v1.sbatch',
        join_launcher=ROOT/'slurm/rc_h593_rank_action_join_v1.sbatch')
    folds = {}
    for f in range(5):
        pv = read(BASE/f'fold{f}/validation.json'); pg = read(GAP/f'fold{f}/validation.json')
        need(pv['status'] == 'QUADRATIC_RANK_FRESH_REPLAY_NUMPY_PASS' and pv['authority']==bind(parent), 'PARENT_FOLD')
        need(pg['status'] == 'GAP_CURVE_FRESH_REPLAY_PASS', 'GAP_FOLD')
        checked(pv['payload']); checked(pg['payload'])
        folds[str(f)] = dict(rank_payload=pv['payload'], rank_validation=bind(BASE/f'fold{f}/validation.json'),
            gap_payload=pg['payload'], gap_validation=bind(GAP/f'fold{f}/validation.json'),
            train_roles=prior['fold_sources'][str(f)]['train_roles'])
    write(AUTH, dict(status='H593_RANK_ACTION_AUTHORIZED', parent_authority=bind(parent),
        code_sources={k:bind(p) for k,p in codes.items()}, public_sources=prior['public_sources'],
        features=prior['features'], fold_sources=folds,
        join_sources=dict(rank_result=v['result'], rank_validation=bind(BASE/'validation.json'),
            gap_result=gv['result'], gap_validation=bind(GAP/'validation.json'), curator=prior['join_sources']['curator']),
        arms=list(ARMS), primary='DIAG_ACTION3', primary_control='LINEAR_ACTION3', strong_control='GAP_BIAS2',
        inner_folds=4, new_inner_rank_fits=40, fresh_refit_checks=40, outer_rank_refits=0,
        rank_optimizer=prior['optimizer'], rank_steps=2000,
        action=dict(features=['top_minus_second', 'X_top_RAW'], parameters=3, rms='all inner OOF rows per arm',
            objective='sum nonzero-delta logaddexp(0,-delta*g)/ALL calibration N + .001/2 theta_squared',
            optimizer='L-BFGS-B', initialization='zero', maxiter=2000, ftol=1e-12, gtol=1e-8,
            coefficients='unconstrained', bias_selection=False, switch='g>0', original_switch_locked=False),
        signal='DIAG_ACTION3 beats LINEAR_ACTION3 and GAP_BIAS2 in full593 count and component mean',
        evidence_level='Opened H593 exploratory grouped nested OOF; inspired by prior secondary DIAG result',
        prior_primary_rewritten=False, deployment_change=False, encoder_forwards=0))
    print(json.dumps(dict(status='PREPARED', authority=bind(AUTH))), flush=True)


def guard(stage, fold=None):
    a=read(AUTH); checked(a['parent_authority'])
    for source in a['code_sources'].values(): checked(source)
    need(a['code_sources']['program']==bind(__file__), 'PROGRAM_SHA')
    if stage not in ('preflight','publish'): need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    allow={Path(b['path']).resolve() for b in a['public_sources'].values()}
    for sources in a['features']: allow.update(Path(b['path']).resolve() for b in sources.values())
    if stage in ('fit','verify'):
        need(fold in range(5), 'FOLD_RANGE')
        allow.update(Path(b['path']).resolve() for b in a['fold_sources'][str(fold)].values())
    if stage in ('join','join-verify','publish'):
        allow.update(Path(b['path']).resolve() for b in a['join_sources'].values())
        for sources in a['fold_sources'].values(): allow.update(Path(b['path']).resolve() for b in sources.values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
        p=Path(os.fsdecode(args[0])).resolve(); s=str(p).lower()
        need(not any(t in s for t in ('rc_opened_','d1-mi','d1_mi','formal392','/grozi/','/target_join/')), 'PROTECTED_READ')
        if 'curator_roles' in s: need(stage in ('join','join-verify','publish'), 'NO_OUTER_LABELS')
        if ROOT/'reports' in p.parents: need(stage=='publish' and p==REPORT, 'NO_REPORT_READ')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own and stage in ('fit','verify','preflight'):
                own=p.relative_to(OUT).parts[0] in ('preflight.json','.preflight.json.tmp',f'fold{fold}')
            need(own or p in allow, 'UNLISTED_RESULT:'+s)
    sys.addaudithook(audit)
    if stage!='preflight':
        p=read(OUT/'preflight.json'); need(p['status']=='RANK_ACTION_SYNTHETIC_PASS' and p['authority']==bind(AUTH),'PREFLIGHT')
    return a


def data(a,fold):
    rows,labels=Q.features(a); splits=read(checked(a['public_sources']['split']))['folds']
    source=a['fold_sources'][str(fold)]
    roles={r['query_id']:r for r in read(checked(source['train_roles']))['records']}
    train_ids=set(splits[fold]['train_query_ids']); held_ids=set(splits[fold]['heldout_query_ids'])
    need(set(roles)==train_ids and not train_ids & held_ids, 'OUTER_TRAIN_ONLY')
    train=[r for r in rows if r['query_id'] in train_ids]; held=[r for r in rows if r['query_id'] in held_ids]
    need(not {r['source_image_sha256'] for r in train}&{r['source_image_sha256'] for r in held}, 'OUTER_IMAGE_DISJOINT')
    old=read(checked(source['rank_payload'])); ov=read(checked(source['rank_validation']))
    gap=read(checked(source['gap_payload'])); gv=read(checked(source['gap_validation']))
    need(ov['payload']==source['rank_payload'] and ov['authority']==a['parent_authority'] and old['fold']==fold,'RANK_SOURCE_SEAL')
    need(gv['payload']==source['gap_payload'] and gap['fold']==fold,'GAP_SOURCE_SEAL')
    need(set(old['train_query_ids'])==set(gap['train_query_ids'])==train_ids,'SAME_TRAIN')
    return rows,labels,splits,roles,train,held,old,gap


def rank_fit(x,y,arm):
    if arm=='RANK_ONLY6':
        from rc_aslo_xf.h593_conditional_rank_v1 import fit_rank
        return fit_rank(x,y)
    from rc_aslo_xf.h593_quadratic_rank_v1 import fit_rank
    return fit_rank(x,y,'DIAG12')


def selected(record,top,switch):
    pos=record['challenger_positions'][top] if switch else record['winner']
    return record['candidate_physical_rows'][pos]


def compute(a,fold):
    import torch
    from rc_aslo_xf.h593_rank_action_v1 import action_features,fit_action,action_scores
    rows,labels,splits,roles,train,held,old,gap=data(a,fold)
    byid={r['query_id']:r for r in rows}; train_ids={r['query_id'] for r in train}
    calibration={m:[] for m in ARMS}; inner_heads=[]
    for inner in range(5):
        if inner==fold: continue
        inner_ids=set(splits[inner]['heldout_query_ids']); need(inner_ids<=train_ids,'INNER_WITHIN_TRAIN')
        fitting=[r for r in train if r['query_id'] not in inner_ids]
        validating=[r for r in train if r['query_id'] in inner_ids]
        for key in ('identity','component'):
            need(not {roles[r['query_id']][key] for r in fitting}&{roles[r['query_id']][key] for r in validating},'INNER_GROUP_DISJOINT')
        need(not {r['source_image_sha256'] for r in fitting}&{r['source_image_sha256'] for r in validating},'INNER_IMAGE_DISJOINT')
        effective=[r for r in fitting if Q.target(r,roles[r['query_id']]['identity'],labels)>=-1]
        x=torch.stack([r['modes']['REAL']['X'] for r in effective])
        y=torch.tensor([Q.target(r,roles[r['query_id']]['identity'],labels) for r in effective],dtype=torch.long)
        xx=torch.stack([r['modes']['REAL']['X'] for r in validating])
        heads={}
        for arm in ARMS:
            heads[arm]=hx(rank_fit(x,y,arm)); z=Q.scores(xx,heads[arm],arm)
            for row,zz in zip(validating,z):
                q=row['query_id']; top,features=action_features(zz.numpy(),row['modes']['REAL']['X'].numpy())
                # Labels are used here only AFTER the held-inner prediction exists.
                raw=selected(row,top,False); rival=selected(row,top,True); identity=roles[q]['identity']
                calibration[arm].append(dict(query_id=q,inner_fold=inner,logits_hex=hx(zz),top_index=top,
                    features_hex=hx(features),delta=int(labels[rival]==identity)-int(labels[raw]==identity),
                    raw_correct=labels[raw]==identity,top_correct=labels[rival]==identity))
        inner_heads.append(dict(inner_fold=inner,parameters=heads,train_query_ids=[r['query_id'] for r in fitting],
            effective_train_query_ids=[r['query_id'] for r in effective],heldout_query_ids=[r['query_id'] for r in validating]))
        print(json.dumps(dict(event='INNER_RANKS_DONE',outer_fold=fold,inner_fold=inner)),flush=True)
    actions={}
    for arm,records in calibration.items():
        records.sort(key=lambda r:byid[r['query_id']]['execution_ordinal'])
        need(len(records)==len(train) and {r['query_id'] for r in records}==train_ids,'INNER_COVERAGE')
        design=np.array([[float.fromhex(v) for v in r['features_hex']] for r in records])
        delta=np.array([r['delta'] for r in records])
        actions[arm]=fit_action(design,delta)
    xx=torch.stack([r['modes']['REAL']['X'] for r in held])
    allz={m:Q.scores(xx,old['parameters'][m],m) for m in ARMS}
    oldp={p['query_id']:p for p in old['predictions']}; gaps={p['query_id']:p for p in gap['predictions']}
    predictions=[]
    for i,row in enumerate(held):
        q=row['query_id']; op=oldp[q]; gp=gaps[q]
        for key in ('candidate_physical_rows','challenger_positions','winner','execution_ordinal'):
            need(row[key]==op[key]==gp[key],'FROZEN_OUTER_AXIS')
        models={m:gp['models'][m] for m in CONTROLS}; rank_models={}
        for m in ('COST1_FULL','CE_FULL'):
            need(models[m]['logits_hex']==op['models'][m]['logits_hex'],'OLD_CONTROL_LOGITS_PARITY')
            z=[float.fromhex(v) for v in models[m]['logits_hex']]; top=int(np.argmax(z))
            need(selected(row,top,z[top]>0)==models[m]['selected'],'OLD_CONTROL_ACTION')
        for arm,zs in allz.items():
            z=zs[i]; need(hx(z)==op['models'][arm]['logits_hex'],'OUTER_RANK_LOGITS_BIT_PARITY')
            top,features=action_features(z.numpy(),row['modes']['REAL']['X'].numpy())
            score=float(action_scores(features[None,:],actions[arm])[0])
            need(top==op['models'][arm]['top_index'],'OUTER_RANK_TOP_PARITY')
            rank_models[arm]=op['models'][arm]
            models[ACTION[arm]]=dict(selected=selected(row,top,score>0),top_index=top,
                features_hex=hx(features),score_hex=score.hex(),switch=score>0)
        predictions.append(dict(query_id=q,execution_ordinal=row['execution_ordinal'],winner=row['winner'],
            candidate_physical_rows=row['candidate_physical_rows'],challenger_positions=row['challenger_positions'],
            rank_models=rank_models,models=models))
    return dict(status='RANK_ACTION_FOLD_SEALED',authority=bind(AUTH),fold=fold,
        train_query_ids=[r['query_id'] for r in train],inner_heads=inner_heads,calibration=calibration,
        rank_parameters={m:old['parameters'][m] for m in ARMS},action_parameters=actions,predictions=predictions,
        heldout_label_reads=0,rank_outer_refits=0,unique_inner_rank_fits=8,action_fits=2,
        parent_rank_payload=a['fold_sources'][str(fold)]['rank_payload'])


def fit(a,fold,replay=False):
    from validate_rc_h593_rank_action_v1 import independent_action_features,independent_scales,independent_loss_gradient,independent_action_scores,validate_training,validate_actions
    from validate_rc_h593_conditional_rank_v1 import validate_logits as linear_validate
    from validate_rc_h593_quadratic_rank_v1 import validate_logits as diag_validate
    folder=OUT/f'fold{fold}'
    if not replay and (folder/'validation.json').exists():
        v=read(folder/'validation.json')
        need(v['status']=='RANK_ACTION_NESTED_FRESH_NUMPY_PASS' and v['authority']==bind(AUTH) and v['payload']==bind(folder/'payload.json'),'RESUME_SEAL')
        return
    start=time.monotonic(); p=compute(a,fold)
    if not replay:
        write(folder/'payload.json',p)
        write(folder/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,unique_inner_rank_fits=8))
        subprocess.run([sys.executable,__file__,'verify','--fold',str(fold)],check=True)
        print(json.dumps(dict(event='FOLD_VALIDATED',fold=fold)),flush=True); return
    need(p==read(folder/'payload.json'),'FRESH_NESTED_PARAMETERS_AND_PREDICTIONS_EXACT')
    rows,labels,splits,roles,train,held,old,gap=data(a,fold); byid={r['query_id']:r for r in rows}
    logit_count=0; max_error=0.; action_checks={}
    def verify_rank(records,parameters,arm):
        nonlocal logit_count,max_error
        xx=np.stack([byid[r['query_id']]['modes']['REAL']['X'].numpy() for r in records])
        saved=[r['logits_hex'] for r in records]
        v=linear_validate(xx,parameters,saved) if arm=='RANK_ONLY6' else diag_validate(xx,parameters,saved,'DIAG12')
        logit_count+=v['logit_count']; max_error=max(max_error,v['max_abs_error'])
    for h in p['inner_heads']:
        for arm in ARMS:
            records=[r for r in p['calibration'][arm] if r['inner_fold']==h['inner_fold']]
            need({r['query_id'] for r in records}==set(h['heldout_query_ids']),'INNER_RECORD_COVERAGE')
            verify_rank(records,h['parameters'][arm],arm)
    for arm in ARMS:
        records=p['calibration'][arm]; fs=[]; ds=[]
        for record in records:
            row=byid[record['query_id']]; zz=np.array([float.fromhex(v) for v in record['logits_hex']])
            top,features=independent_action_features(zz,row['modes']['REAL']['X'].numpy())
            need(top==record['top_index'] and hx(features)==record['features_hex'],'INDEPENDENT_INNER_ACTION_FEATURES')
            identity=roles[row['query_id']]['identity']
            delta=int(labels[selected(row,top,True)]==identity)-int(labels[selected(row,top,False)]==identity)
            need(delta==record['delta'],'TRAIN_DELTA_FROM_IDENTITY')
            fs.append(features); ds.append(delta)
        design=np.array(fs); delta=np.array(ds); head=p['action_parameters'][arm]
        theta=np.array([float.fromhex(v) for v in head['theta_hex']]); scales=np.array([float.fromhex(v) for v in head['scale_hex']])
        independent=independent_scales(design)
        need(np.allclose(independent,scales,rtol=1e-13,atol=1e-13),'ALL_INNER_RMS')
        loss,gradient=independent_loss_gradient(theta,design,delta,scales)
        training_check=validate_training(design,delta,head)
        need(np.isfinite(loss) and np.isfinite(gradient).all() and float(np.max(abs(gradient)))<1e-5,'CONVEX_ACTION_STATIONARITY')
        outer_records=[dict(query_id=r['query_id'],logits_hex=r['rank_models'][arm]['logits_hex']) for r in p['predictions']]
        verify_rank(outer_records,p['rank_parameters'][arm],arm)
        outer_features=[]; tops=[]
        for pred in p['predictions']:
            row=byid[pred['query_id']]; zz=np.array([float.fromhex(v) for v in pred['rank_models'][arm]['logits_hex']])
            top,features=independent_action_features(zz,row['modes']['REAL']['X'].numpy())
            saved=pred['models'][ACTION[arm]]
            need(top==saved['top_index'] and hx(features)==saved['features_hex'],'OUTER_ACTION_FEATURES')
            outer_features.append(features); tops.append(top)
        scores=independent_action_scores(np.array(outer_features),head)
        for pred,top,score in zip(p['predictions'],tops,scores):
            saved=pred['models'][ACTION[arm]]; wanted=float.fromhex(saved['score_hex'])
            need(abs(float(score)-wanted)<2e-10 and bool(score>0)==saved['switch'],'INDEPENDENT_OUTER_SCORE_SIGN')
            need(selected(pred,top,score>0)==saved['selected'],'INDEPENDENT_ACTION_SELECTED')
        outer_check=validate_actions(
            np.array([[float.fromhex(v) for v in r['rank_models'][arm]['logits_hex']] for r in p['predictions']]),
            np.stack([byid[r['query_id']]['modes']['REAL']['X'].numpy() for r in p['predictions']]),head,
            [r['winner'] for r in p['predictions']],[r['challenger_positions'] for r in p['predictions']],
            [r['candidate_physical_rows'] for r in p['predictions']],
            [r['models'][ACTION[arm]]['score_hex'] for r in p['predictions']],
            [r['models'][ACTION[arm]]['selected'] for r in p['predictions']],
            saved_top_indices=[r['models'][ACTION[arm]]['top_index'] for r in p['predictions']],
            saved_features_hex=[r['models'][ACTION[arm]]['features_hex'] for r in p['predictions']])
        action_checks[arm]=dict(loss=float(loss),gradient_max_abs=float(np.max(abs(gradient))),
            inner_records=len(records),outer_actions=len(p['predictions']),rms_verified=True,training=training_check,outer=outer_check)
    write(folder/'validation.json',dict(status='RANK_ACTION_NESTED_FRESH_NUMPY_PASS',authority=bind(AUTH),
        payload=bind(folder/'payload.json'),independent_rank_logit_count=logit_count,rank_max_abs_error=max_error,
        action_checks=action_checks,heldout_label_reads=0,fresh_inner_refits=8,fresh_action_refits=2))


def compare(rows,base,new):
    groups=defaultdict(list)
    for r in rows: groups[r['component']].append(int(r['correct'][new])-int(r['correct'][base]))
    need(len(groups)==64,'COMPONENT64')
    d=np.array([np.mean(v) for _,v in sorted(groups.items())]); rng=np.random.default_rng(20260920)
    boot=d[rng.integers(0,64,size=(10000,64))].mean(1)
    gained=[r['query_id'] for r in rows if r['correct'][new] and not r['correct'][base]]
    lost=[r['query_id'] for r in rows if not r['correct'][new] and r['correct'][base]]
    return dict(baseline=base,new=new,rescue=len(gained),loss=len(lost),net=len(gained)-len(lost),
        changed=sum(r['selected'][base]!=r['selected'][new] for r in rows),gained_query_ids=gained,lost_query_ids=lost,
        equal_component_difference=float(d.mean()),bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))))


def join(a,replay=False):
    payloads=[]; seals=[]
    for f in range(5):
        v=read(OUT/f'fold{f}/validation.json')
        need(v['status']=='RANK_ACTION_NESTED_FRESH_NUMPY_PASS' and v['authority']==bind(AUTH),'ALL_FOLDS_SEALED')
        p=read(checked(v['payload'])); need(p['fold']==f and p['authority']==bind(AUTH),'FOLD_BOUND')
        payloads.append(p); seals.append(bind(OUT/f'fold{f}/validation.json'))
    write(OUT/'all_predictions_prelabel_seal.json',dict(authority=bind(AUTH),validations=seals))
    roles={r['query_id']:r for r in read(checked(a['join_sources']['curator']))['records']}
    worker={r['query_id']:r for r in read(checked(a['public_sources']['worker']))['records']}
    gallery=read(checked(a['public_sources']['gallery'])); labels={r['physical_row']:r['identity'] for r in gallery['records']}
    prior=read(checked(a['join_sources']['gap_result'])); rank=read(checked(a['join_sources']['rank_result']))
    for name in ('gap','rank'):
        need(read(checked(a['join_sources'][name+'_validation']))['result']==a['join_sources'][name+'_result'],'PARENT_JOIN_SEAL')
    gp={r['query_id']:r for r in prior['rows']}; rp={r['query_id']:r for r in rank['rows']}
    rows=[]
    for p in payloads:
        training=read(checked(a['fold_sources'][str(p['fold'])]['train_roles']))['records']
        train_ids={r['query_id'] for r in training}; need(train_ids==set(p['train_query_ids']),'TRAIN_AXIS')
        train_hashes={worker[q]['source_image_sha256'] for q in train_ids}
        for pred in p['predictions']:
            q=pred['query_id']; role=roles[q]
            need(role['outer_fold']==p['fold'] and worker[q]['source_image_sha256'] not in train_hashes,'OUTER_FOLD_IMAGE')
            for k in ('identity','component'): need(role[k] not in {r[k] for r in training},'OUTER_GROUP_DISJOINT')
            axis=pred['candidate_physical_rows']; raw=axis[pred['winner']]
            chosen=dict(RAW=raw,**{m:pred['models'][m]['selected'] for m in CONTROLS+tuple(ACTION.values())})
            correct={m:labels[v]==role['identity'] for m,v in chosen.items()}
            present=any(labels[v]==role['identity'] for v in axis)
            for m in ('RAW',)+CONTROLS:
                need(chosen[m]==gp[q]['selected'][m] and correct[m]==gp[q]['correct'][m],'OLD_FINAL_DECISION_PARITY')
            top={}
            for arm in ARMS:
                model=pred['models'][ACTION[arm]]; top[arm]=selected(pred,model['top_index'],True)
                score=float.fromhex(model['score_hex'])
                need(model['switch']==(score>0) and chosen[ACTION[arm]]==selected(pred,model['top_index'],score>0),'JOIN_ACTION_REPLAY')
                need(top[arm]==rp[q]['top_physical'][arm],'PARENT_RANK_CANDIDATE')
            need(present==gp[q]['target_in_C128']==rp[q]['target_present'],'CANDIDATE_MEMBERSHIP')
            rows.append(dict(query_id=q,original_query_id=role['original_query_id'],component=role['component'],fold=p['fold'],
                target_in_C128=present,selected=chosen,correct=correct,top_physical=top,
                original40_ranking_blocked=(not correct['RAW'] and present and not rp[q]['target_is_top']['COST1_FULL'])))
    rows.sort(key=lambda r:r['query_id'])
    need(len(rows)==len({r['query_id'] for r in rows})==593 and sum(r['target_in_C128'] for r in rows)==570,'ALL593_RECALL570')
    summary={}
    for m in MODELS:
        correct=sum(r['correct'][m] for r in rows); rescues=sum(r['correct'][m] and not r['correct']['RAW'] for r in rows)
        breaks=sum(not r['correct'][m] and r['correct']['RAW'] for r in rows)
        summary[m]=dict(correct=correct,total=593,accuracy=correct/593,rescue_vs_RAW=rescues,break_vs_RAW=breaks,
            net_vs_RAW=rescues-breaks,raw_correct_loss_rate=breaks/426,switches=sum(r['selected'][m]!=r['selected']['RAW'] for r in rows),
            by_fold={str(f):sum(r['correct'][m] for r in rows if r['fold']==f) for f in range(5)},
            old40_correct=sum(r['correct'][m] for r in rows if r['original40_ranking_blocked']))
    for m,n in [('RAW',426),('COST1_FULL',481),('CE_FULL',486),('GAP_BIAS2',492)]: need(summary[m]['correct']==n,'FROZEN_FINAL_BASELINES')
    comparisons={m:{b:compare(rows,b,m) for b in ('RAW',)+CONTROLS} for m in ACTION.values()}
    paired=compare(rows,'LINEAR_ACTION3','DIAG_ACTION3')
    strong=comparisons['DIAG_ACTION3']['GAP_BIAS2']
    signal=paired['net']>0 and strong['net']>0 and paired['equal_component_difference']>0 and strong['equal_component_difference']>0
    output=dict(status='H593_RANK_ACTION_ALL593_COMPLETE',authority=bind(AUTH),summary=summary,comparisons=comparisons,
        matched_ranker_comparison=paired,positive_internal_performance_signal=signal,
        action_parameters={str(p['fold']):p['action_parameters'] for p in payloads},rows=rows,
        evidence_level=a['evidence_level'],all593_decisions_evaluated=True,action_trained=True,
        automatic_deployment_change=False,external_GO=False,prior_primary_rewritten=False,
        limits=['This is adaptive development on opened H593, not untouched confirmation.',
            'Component intervals condition on predictions and do not cover repeated model selection or shared training.',
            'The new recipe changes challenger ranking and calibrates all queries; no previous SWITCH is locked.',
            'Logistic calibration is fixed in advance; no outer threshold or sample-specific adjustment.'])
    if replay:
        need(output==read(OUT/'result.json'),'FRESH_JOIN_EXACT')
        write(OUT/'validation.json',dict(status='RANK_ACTION_ALL_COUNTS_PASS',authority=bind(AUTH),result=bind(OUT/'result.json'),fold_validations=seals))
    else:
        write(OUT/'result.json',output); subprocess.run([sys.executable,__file__,'join-verify'],check=True)
        print(json.dumps(dict(status='COMPLETE',summary=summary,matched=paired,internal_signal=signal)),flush=True)


def publish():
    v=read(OUT/'validation.json'); need(v['status']=='RANK_ACTION_ALL_COUNTS_PASS','VALIDATED'); r=read(checked(v['result']))
    lines=['# H593：平方排序与线性排序的同配方行动验证','',
        '完整593张、自然RAW C128、五折分组OOF。每个外层TRAIN有四个inner留出组，新训练两类rank头生成校准证据；动作只使用top间隔与RAW内容间隔。DIAG_ACTION3为本轮探索主臂，不回写上一轮FULL_QUAD27的主实验结论。','',
        '| 模型 | 正确／593 | 相对RAW救回 | 相对RAW损失 | 切换数 |','|---|---:|---:|---:|---:|']
    for m,s in r['summary'].items(): lines.append(f"| {m} | {s['correct']} | {s['rescue_vs_RAW']} | {s['break_vs_RAW']} | {s['switches']} |")
    lines+=['','| 对照→DIAG_ACTION3 | 救回 | 损失 | 净增 | 组件等权差 | 95%区间 |','|---|---:|---:|---:|---:|---|']
    for b,s in dict(r['comparisons']['DIAG_ACTION3'],LINEAR_ACTION3=r['matched_ranker_comparison']).items():
        lines.append(f"| {b} | {s['rescue']} | {s['loss']} | {s['net']} | {s['equal_component_difference']:.6f} | {s['bootstrap95']} |")
    lines+=['',f"本轮预定内部性能信号：{r['positive_internal_performance_signal']}。要求相对共同门的线性对照与原492均为正净增和正组件方向。",'',
        '23张候选缺失保留在593分母。这里只报告实际HOLD/SWITCH选中的身份，不用RAW-or-top oracle代替准确率。旧模型保留，不自动替换论文主模型。','',
        'gate三参数无符号约束，RMS仅来自本臂TRAIN内层OOF全部记录，neutral对损失分子贡献0但保留分母。学习bias后直接使用0阈值，无外层调参、无后续阈值扫描、无旧正SWITCH锁。','',
        '全部inner头和action在新进程重拟合重放，NumPy独立核算分数、特征、RMS、凸目标梯度与动作。反复开发后的分组区间不等于外部确认。','',
        '结果和逐query见`results/rc_h593_rank_action_v1/result.json`；各折及汇总验证见对应validation.json。','']
    REPORT.write_text('\n'.join(lines)); print(REPORT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('prepare','preflight','fit','verify','join','join-verify','publish')); parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.stage=='prepare': prepare()
    else:
        a=guard(args.stage,args.fold)
        if args.stage=='publish': publish()
        else:
            import torch
            torch.set_num_threads(8); torch.set_num_interop_threads(1)
            if args.stage=='preflight':
                from rc_aslo_xf.h593_rank_action_v1 import self_test
                from validate_rc_h593_rank_action_v1 import self_test as independent_test
                write(OUT/'preflight.json',dict(status='RANK_ACTION_SYNTHETIC_PASS',authority=bind(AUTH),core=self_test(),independent=independent_test(),natural_updates=0))
                print('RANK_ACTION_SYNTHETIC_PASS',flush=True)
            elif args.stage in ('fit','verify'): fit(a,args.fold,args.stage=='verify')
            else: join(a,args.stage=='join-verify')
