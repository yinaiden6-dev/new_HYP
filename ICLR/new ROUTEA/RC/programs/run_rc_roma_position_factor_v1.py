#!/usr/bin/env python3
"""Small, source-pinned norm/phase experiment; no training or fine refinement."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'programs'), str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
import rc_roma_position_factor_v1 as P
from rc_roma_shared_native_cache_v1 import ExactPool
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature

OUT = ROOT/'results/rc_roma_position_factor_v1'
AUTH = ROOT/'registry/rc_roma_position_factor_authority_v1_20260924.json'
PLAN = ROOT/'plan/RC_ROMA_POSITION_FACTOR_V1_20260924.md'
SCRIPT = ROOT/'slurm/rc_roma_position_factor_v1.sbatch'
INSIDE = ROOT/'results/rc_h593_m_inside_v1'
SUBSET = ROOT/'results/rc_h593_subset41_frozen_paths_v1'
read, write, save, bind, checked = C.read, C.write, C.save, C.bind, C.checked


def emit(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def prepare():
    assert not AUTH.exists()
    algebra = P.self_test()
    source = ROOT/'results/rc_pair_relation_localization_v1/inside_paths/rows41.json'
    rows = read(source)
    panel, groups = [], set()
    for r in rows:
        if r['component'] not in groups:
            panel.append(r['index']); groups.add(r['component'])
        if len(panel) == 8:
            break
    cases = [r['index'] for r in rows if r['original_rescue']]
    assert panel == [0,1,2,4,5,6,7,8] and cases == [9,21,32]
    indices = sorted(set(panel + cases))
    workers = read(C.WORKERS)['records']
    manifest = dict(indices=indices, outcome_independent_panel=panel, selected_rescue_audit=cases,
                    workers=[workers[i] for i in indices],
                    cohort_source=bind(source),
                    radius=[math.sqrt(1024*np.mean([v['side_rms'][s]['matching_position']**2
                          for r in rows for v in r['component_diagnostics']['candidates']])) for s in (0,1)])
    write(OUT/'manifest.json', manifest)
    parent = read(ROOT/'registry/rc_h593_subset41_frozen_paths_authority_v1_20260923.json')
    code = [Path(__file__), Path(P.__file__), PLAN, SCRIPT, Path(C.__file__), Path(C.M.__file__),
            ROOT/'programs/rc_roma_shared_native_cache_v1.py',
            ROOT/'programs/run_rc_h593_subset41_frozen_paths_v1.py',
            ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',
            ROOT.parents[2]/'third_party/RoMaV2/src/romav2/matcher.py',
            ROOT.parents[2]/'third_party/RoMaV2/src/romav2/dpt.py']
    write(AUTH, dict(status='ROMA_POSITION_FACTOR_AUTHORIZED', sources=[bind(p) for p in code],
                    manifest=bind(OUT/'manifest.json'), profile=bind(C.PROFILE), heads=bind(SUBSET/'heads.json'),
                    gallery=parent['gallery'], old_result=parent['old_result'],
                    branches=list(P.ARMS), query_count=len(indices), candidates=128,
                    delta=P.DELTA, seed=P.SEED, stage='COARSE', budget_seconds=400,
                    user_authorization='Continue norm versus position-arrangement isolation; retain intermediates; small existing opened panel.',
                    training=False, external_GO=False, protected_data_read=False))
    write(OUT/'preflight.json', dict(**algebra, authority=bind(AUTH)))
    emit(status='PREPARED', indices=indices, radius=manifest['radius'], arms=len(P.ARMS))


def guard():
    a = read(AUTH)
    assert a['status'] == 'ROMA_POSITION_FACTOR_AUTHORIZED'
    for b in [*a['sources'], a['manifest'], a['profile'], a['heads']]:
        checked(b)
    assert read(OUT/'preflight.json')['authority'] == bind(AUTH)
    return a, read(a['manifest']['path'])


def cpu_copy(t):
    x = t.detach().cpu().clone()
    if x.dtype == torch.float32 and torch.equal(x.to(torch.bfloat16).float(), x):
        return dict(tensor=x.to(torch.bfloat16), original_dtype='float32', lossless_bfloat16=True)
    return dict(tensor=x, original_dtype=str(x.dtype).split('.')[-1], lossless_bfloat16=False)


def restore(x, device):
    return x['tensor'].to(device=device, dtype=getattr(torch, x['original_dtype']))


def ensure_head(model):
    path = OUT/'head_replay_state.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            save(path, dict(authority=bind(AUTH), head={k:v.detach().cpu().clone() for k,v in model.matcher.head.state_dict().items()},
                            omega=model.matcher.omega.detach().cpu().clone(), scale=model.matcher.scale.detach().cpu().clone(),
                            config=dict(dim_in=1024,out_dim=3,patch_size=16,features=256,
                                        out_channels=[256,512,1024,1024],pos_embed=False,feature_only=False,
                                        down_ratio=4,align_corners=True)))


def descriptor(model, core, item, stats):
    sha = item['sha256']
    receipt = OUT/'images'/f'{sha}.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            d = read(receipt); assert d['profile'] == bind(C.PROFILE)
            p = torch.load(checked(d['payload']), map_location='cpu', weights_only=True, mmap=True)
            stats['new_cache_descriptor_reads'] += 1
            return [t.to('cuda').clone() for t in p['features']], d['payload']
        old_receipt = ROOT/'cache/rc_h593_shared_native_v1/images'/f'{sha}.json'
        if old_receipt.exists():
            d = read(old_receipt)
            assert d['status'] == 'NATIVE_COARSE_IMAGE_PASS' and d['image']['sha256'] == sha
            p = torch.load(checked(d['payload']), map_location='cpu', weights_only=True, mmap=True)
            assert p['profile'] == bind(C.PROFILE) and p['input_key'] == d['input_key']
            write(receipt, dict(profile=bind(C.PROFILE), payload=d['payload'], source=bind(old_receipt), image=item))
            stats['historical_descriptor_reads'] += 1
            return [t.to('cuda').clone() for t in p['features']], d['payload']
        assert bind(item['path'])['sha256'] == sha
        image = model._load_image(core.oriented(Path(item['path'])))
        lr = F.interpolate(image, size=(800,800), mode='bicubic', align_corners=False, antialias=True)
        values = model.f(lr)
        path = OUT/'images'/f'{sha}.pt'
        save(path, dict(profile=bind(C.PROFILE), image=item, features=[t.detach().cpu().clone() for t in values]))
        b = bind(path); write(receipt, dict(profile=bind(C.PROFILE), payload=b, image=item))
        stats['new_descriptor_forwards'] += 1
        return values, b


@torch.inference_mode()
def capture(model, qf, rf, index, pos, old, qbinding, rbinding, query_sha, reference_sha, path, stats):
    if path.exists():
        p = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert p['authority'] == bind(AUTH) and p['query_sha'] == query_sha and p['reference_sha'] == reference_sha
        stats['pair_capture_reused'] += 1
        return p
    sides = []
    if index == pos == 0 and old['full_first_pair_components']:
        for s, fs in enumerate((qf, rf)):
            comp = old['sides'][s]['first_pair_components']
            assert torch.equal(fs[-1].cpu(), comp['appearance'])
            sides.append(dict(J=cpu_copy(comp['joint_context']), P=cpu_copy(comp['matching_position'])))
        stats['historical_complete_pair_reused'] += 1
    else:
        import romav2.matcher as module
        original = module._compute_head_preds
        def observe(**kwargs):
            side = dict(J=cpu_copy(kwargs['f_mv_A']), P=cpu_copy(kwargs['match_emb_AB']))
            native = original(**kwargs)
            side.update(native_warp=native[0].detach().cpu().clone(), native_confidence=native[1].detach().cpu().clone())
            sides.append(side)
            return native
        module._compute_head_preds = observe
        try:
            model.matcher([x.clone() for x in qf], [x.clone() for x in rf], img_A=None, img_B=None, bidirectional=True)
        finally:
            module._compute_head_preds = original
        stats['new_cross_image_forwards'] += 1
    assert len(sides) == 2
    p = dict(authority=bind(AUTH), query_sha=query_sha, reference_sha=reference_sha,
             descriptor_sources=[qbinding, rbinding], sides=sides)
    save(path, p)
    return p


@torch.inference_mode()
def gpu(index, pilot=False):
    a, manifest = guard()
    assert os.environ.get('SLURM_JOB_ID') and index in manifest['indices']
    if pilot and (OUT/'pilot_validation.json').exists():
        assert read(OUT/'pilot_validation.json')['status'] == 'POSITION_FACTOR_FIRST_PAIR_PASS'
        return 0
    if not pilot:
        assert read(OUT/'pilot_validation.json')['status'] == 'POSITION_FACTOR_FIRST_PAIR_PASS'
    d = OUT/f'query{index:03d}'; d.mkdir(parents=True, exist_ok=True)
    with (d/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (d/'gpu_validation.json').exists():
            return 0
        start = time.monotonic()
        w = next(w for w in manifest['workers'] if w['execution_ordinal'] == index)
        q, prior, refs = C.load_input(w)
        im = read(INSIDE/f'query{index:03d}/inside_manifest.json')
        assert im['query_id'] == w['query_id']
        profile = read(a['profile']['path'])
        core = C.M.legacy_core(profile); core.cell_means = ExactPool()
        model = C.M.gpu_model(profile)
        assert model.H_lr == model.W_lr == 800 and model.bidirectional
        ensure_head(model)
        stats = {k:0 for k in ('new_cache_descriptor_reads','historical_descriptor_reads','new_descriptor_forwards',
                               'pair_capture_reused','historical_complete_pair_reused','new_cross_image_forwards')}
        qp = Path(q['query_source_path'])
        qgeom, _ = C.M.geometry(qp,q['query_source_sha256'],'phase-query',q['query_grid_shape'],q['processor_input_frame'],profile['sources']['processor']['sha256'])
        qf, qb = descriptor(model,core,dict(path=str(qp),sha256=q['query_source_sha256']),stats)
        import romav2.matcher as module
        count = 0
        for pos, physical in enumerate(q['candidate_physical_rows']):
            if pilot and pos > 0:
                break
            dst = d/f'pair{pos:03d}.pt'
            if dst.exists():
                count += 1; continue
            if time.monotonic() - start > a['budget_seconds']:
                emit(status='NORMAL_BUDGET_CHECKPOINT',index=index,pairs=count,stats=stats)
                return 75
            ref = refs[physical]
            old = torch.load(checked(im['sources'][pos]),map_location='cpu',weights_only=True)
            assert old['reference_sha'] == ref['source_image_sha256'] and old['query_sha'] == q['query_source_sha256']
            rf, rb = descriptor(model,core,dict(path=ref['source_path'],sha256=ref['source_image_sha256']),stats)
            rg, _ = C.M.geometry(Path(ref['source_path']),ref['source_image_sha256'],'phase-reference',ref['grid_shape'],'DECODED_RAW_BEFORE_EXIF',profile['sources']['processor']['sha256'])
            cap_path = d/f'capture{pos:03d}.pt'
            cap = capture(model,qf,rf,index,pos,old,qb,rb,q['query_source_sha256'],ref['source_image_sha256'],cap_path,stats)
            branches = {arm:[] for arm in P.ARMS}
            for side, (fs, geom) in enumerate(((qf,qgeom),(rf,rg))):
                values = cap['sides'][side]
                j, p = restore(values['J'],'cuda'), restore(values['P'],'cuda')
                for arm in P.ARMS:
                    changed, invariants = P.transform(p,model.matcher.omega,model.matcher.scale,arm,manifest['radius'][side],side)
                    warp, confidence = module._compute_head_preds(f_list_A=[x.clone() for x in fs],
                        match_emb_AB=changed,f_mv_A=j,img_A=None,img_B=None,head=model.matcher.head)
                    weights = core.cell_means(confidence[0,...,0].sigmoid().cpu(),geom)
                    if arm in ('NATIVE','ZERO'):
                        branch = 'A1J1P1' if arm == 'NATIVE' else 'A1J1P0'
                        expected = old['sides'][side]['coarse'][branch]['weights']
                        assert torch.equal(weights,expected), ('ORIGINAL_COARSE_BITS',index,pos,side,arm,float((weights-expected).abs().max()))
                        if arm == 'NATIVE' and 'native_confidence' in values:
                            assert torch.equal(confidence.cpu(),values['native_confidence'])
                    record = dict(weights=weights,confidence=confidence.detach().cpu().clone(),invariants=invariants)
                    if arm == 'NATIVE':
                        record['native_warp'] = warp.detach().cpu().clone()
                    branches[arm].append(record)
            save(dst, dict(authority=bind(AUTH),index=index,position=pos,physical_row=physical,
                           capture=bind(cap_path),old_inside_source=im['sources'][pos],branches=branches))
            count += 1
            emit(status='PAIR_SAVED',index=index,pairs=count,seconds=time.monotonic()-start,stats=stats)
        if pilot:
            write(OUT/'pilot_validation.json',dict(status='POSITION_FACTOR_FIRST_PAIR_PASS',authority=bind(AUTH),
                  pair=bind(d/'pair000.pt'),native_and_zero_maps_bit_exact=True,arms=len(P.ARMS),
                  original_feature_reuse=stats,head_state=bind(OUT/'head_replay_state.pt')))
            return 0
        assert count == 128
        write(d/'gpu_validation.json',dict(status='POSITION_FACTOR_ALL128_PASS',authority=bind(AUTH),index=index,
                  query_id=w['query_id'],pairs=[bind(d/f'pair{pos:03d}.pt') for pos in range(128)],
                  candidate_physical_rows=q['candidate_physical_rows'],stats_last_chunk=stats,
                  native_and_zero_maps_bit_exact=True,refiner_forwards=0,encoder_or_llm_forwards=0))
        return 0


def replay_query(index, a, manifest):
    import run_rc_h593_subset41_frozen_paths_v1 as R
    import run_rc_h593_quality_operator_eval_v1 as E
    d = OUT/f'query{index:03d}'
    if (d/'cpu_validation.json').exists():
        return
    validation = read(d/'gpu_validation.json')
    assert validation['status'] == 'POSITION_FACTOR_ALL128_PASS' and validation['authority'] == bind(AUTH)
    w = next(w for w in manifest['workers'] if w['execution_ordinal'] == index)
    q, old, refs = C.load_input(w)
    head = read(a['heads']['path'])['heads'][w['query_id']]
    theta = np.array([float.fromhex(x) for x in head['theta_hex']])
    prior = read(SUBSET/f'query{index:03d}/payload.json')
    row = prior['row']
    assert row['candidate_physical_rows'] == q['candidate_physical_rows'] == validation['candidate_physical_rows']
    scores = {mode+'/'+arm:{} for mode in ('M_ONLY','FULL_UV') for arm in P.ARMS}
    maximum = 0.
    for pos, b in enumerate(validation['pairs']):
        pair = torch.load(checked(b),map_location='cpu',weights_only=True)
        assert pair['position'] == pos and pair['physical_row'] == q['candidate_physical_rows'][pos]
        ref = refs[pair['physical_row']]
        sim = F.normalize(q['query_tokens'].double(),dim=1) @ F.normalize(ref['tokens'].double(),dim=1).T
        nu,nv = old['candidates'][pos]['query_visibility'],old['candidates'][pos]['reference_visibility']
        local = R.locals_torch(sim,nv)
        for arm in P.ARMS:
            u,v = [x['weights'] for x in pair['branches'][arm]]
            for mode in ('M_ONLY','FULL_UV'):
                s = R.pair_scores(sim,nu,nv,u,v,mode,local)
                independent = R.independent(sim.numpy(),nu.numpy(),nv.numpy(),u.numpy(),v.numpy(),mode)
                error = max(abs(s[k]-independent[k]) for k in s)
                maximum=max(maximum,error);assert error<2e-10
                scores[mode+'/'+arm][pos]=s
    raw = q['candidate_raw_scores']; predictions = {}
    for name,evidence in scores.items():
        x = torch.stack([candidate_feature(raw,evidence,c,row['winner']) for c in row['challenger_positions']])
        z = x @ torch.tensor(theta[:-1]) + float(theta[-1])
        nz = (x.numpy()*theta[:-1]).sum(1)+theta[-1]
        error=float(np.max(np.abs(nz-z.numpy())));maximum=max(maximum,error);assert error<2e-10
        selected=E.choose(z,row);assert selected==E.choose(nz,row)
        if name.endswith('/NATIVE') or name.endswith('/ZERO'):
            mode,arm=name.split('/')
            oldname=mode+('/native/COARSE' if arm=='NATIVE' else '/inside/A1J1P0/COARSE')
            expected=np.array([float.fromhex(t)for t in prior['models'][oldname]['logits_hex']])
            assert np.max(abs(z.numpy()-expected))<2e-10 and selected==prior['models'][oldname]['selected']
        predictions[name]=dict(logits=z.tolist(),selected=selected,X=x.tolist())
    payload=dict(authority=bind(AUTH),index=index,query_id=w['query_id'],row=row,fold=head['fold'],
                 theta=theta.tolist(),models=predictions,scores=scores,maximum_independent_error=maximum,
                 gpu_source=bind(d/'gpu_validation.json'))
    write(d/'cpu_payload.json',payload)
    write(d/'cpu_validation.json',dict(status='POSITION_FACTOR_CPU128_PASS',payload=bind(d/'cpu_payload.json'),
                                     logits=len(predictions)*127,maximum_independent_error=maximum))
    emit(status='CPU_QUERY_PASS',index=index,logits=len(predictions)*127)


def join():
    a,m=guard();start=time.monotonic()
    for i in m['indices']:
        if time.monotonic()-start>400:
            return 75
        replay_query(i,a,m)
    # Read opened identity labels only after all label-free predictions are sealed.
    old={r['query_id']:r for r in read(checked(a['old_result']))['rows']}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    rows=[]
    for i in m['indices']:
        p=read(checked(read(OUT/f'query{i:03d}/cpu_validation.json')['payload']))
        r=old[p['query_id']];axis=p['row']['candidate_physical_rows']
        target=next((n for n,x in enumerate(axis)if labels[x]==r['identity']),None)
        record=dict(index=i,query_id=p['query_id'],original_query_id=r['original_query_id'],component=r['component'],
                    cohort='OUTCOME_INDEPENDENT8' if i in m['outcome_independent_panel'] else 'SELECTED_RESCUES3',
                    raw_correct=r['correct']['RAW'],target_in_C128=target is not None,models={})
        for name,pred in p['models'].items():
            z=np.zeros(128);z[p['row']['challenger_positions']]=pred['logits']
            item=dict(correct=labels[pred['selected']]==r['identity'],selected=pred['selected'])
            if target is not None:
                wrong=max(z[j]for j in range(128)if j!=target)
                item.update(target_logit=float(z[target]),margin=float(z[target]-wrong))
                mode=name.split('/')[0]
                native_z=np.zeros(128);native_z[p['row']['challenger_positions']]=p['models'][mode+'/NATIVE']['logits']
                fixed_wrong=max((j for j in range(128)if j!=target),key=lambda j:native_z[j])
                item['fixed_native_wrong_margin']=float(z[target]-z[fixed_wrong])
                evidence=p['scores'][name]
                masses=[evidence[str(j)]['visibility_mass'] for j in range(128)]
                native_m=[p['scores'][mode+'/NATIVE'][str(j)]['visibility_mass'] for j in range(128)]
                fixed_m_wrong=max((j for j in range(128)if j!=target),key=lambda j:native_m[j])
                item['logM_fixed_native_wrong_margin']=math.log(max(masses[target],1e-300)/max(masses[fixed_m_wrong],1e-300))
            record['models'][name]=item
        if target is not None:
            record['contrasts']={}
            for mode in ('M_ONLY','FULL_UV'):
                for metric in ('fixed_native_wrong_margin','margin','logM_fixed_native_wrong_margin'):
                    global_value=np.mean([record['models'][mode+'/GLOBAL_'+axis+'_'+sign][metric]for axis in ('X','Y')for sign in ('PLUS','MINUS')])
                    local_value=np.mean([record['models'][mode+'/LOCAL_'+axis+'_'+sign][metric]for axis in ('X','Y')for sign in ('PLUS','MINUS')])
                    native_value=record['models'][mode+'/NATIVE'][metric]
                    record['contrasts'][mode+'/'+metric]=dict(local_minus_global=float(local_value-global_value),
                        global_minus_native=float(global_value-native_value),
                        amplitude_flat_minus_native=record['models'][mode+'/AMP_FLAT'][metric]-native_value,
                        amplitude_permute_minus_native=record['models'][mode+'/AMP_PERMUTE'][metric]-native_value)
        rows.append(record)
    summaries={}
    for cohort in ('OUTCOME_INDEPENDENT8','SELECTED_RESCUES3'):
        rs=[r for r in rows if r['cohort']==cohort]
        summaries[cohort]={name:dict(correct=sum(r['models'][name]['correct']for r in rs),n=len(rs),
            rescues=sum(not r['raw_correct'] and r['models'][name]['correct']for r in rs),
            breaks=sum(r['raw_correct'] and not r['models'][name]['correct']for r in rs))for name in rows[0]['models']}
    contrast_summary={}
    for cohort in ('OUTCOME_INDEPENDENT8','SELECTED_RESCUES3'):
        rs=[r for r in rows if r['cohort']==cohort and r['target_in_C128']]
        contrast_summary[cohort]={}
        for key in rs[0]['contrasts'] if rs else []:
            contrast_summary[cohort][key]={}
            for contrast in rs[0]['contrasts'][key]:
                values=[r['contrasts'][key][contrast]for r in rs]
                groups=sorted({r['component']for r in rs})
                group_means=np.array([np.mean([v for r,v in zip(rs,values)if r['component']==g])for g in groups])
                boot=np.random.default_rng(P.SEED).choice(group_means,(20000,len(groups)),replace=True).mean(1)
                contrast_summary[cohort][key][contrast]=dict(queries=len(rs),groups=len(groups),
                    group_mean=float(group_means.mean()),group_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    negative_queries=sum(v<0 for v in values),positive_queries=sum(v>0 for v in values))
    write(OUT/'result.json',dict(status='POSITION_FACTOR_11_QUERY_COMPLETE',authority=bind(AUTH),rows=rows,summary=summaries,
          contrast_summary=contrast_summary,
          scope='Opened small-panel COARSE fixed-head interventions; no universal or external claim'))
    write(OUT/'validation.json',dict(status='POSITION_FACTOR_INDEPENDENT_CPU_AND_PARITY_PASS',authority=bind(AUTH),
          result=bind(OUT/'result.json'),queries=len(rows),logits_recomputed=len(rows)*24*127,
          native_and_zero_anchors_reproduced=True))
    lines=['# RoMa norm/phase isolation: completed', '', 'COARSE only; original fold COST1, natural C128, fixed HOLD=0. No training. Eight outcome-independent group representatives and three selected original rescues are reported separately.', '', '| Arm | Panel8 correct | Selected rescues3 correct |', '|---|---:|---:|']
    for arm in summaries['OUTCOME_INDEPENDENT8']:
        lines.append(f"| {arm} | {summaries['OUTCOME_INDEPENDENT8'][arm]['correct']}/8 | {summaries['SELECTED_RESCUES3'][arm]['correct']}/3 |")
    lines+=['','Norm/phase manipulation is a conditional head-input intervention; see the frozen plan for off-manifold and coordinate-frame limitations. Full per-candidate predictions and tensor inputs are retained.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    emit(status='POSITION_FACTOR_COMPLETE',summary=summaries)
    return 0


def dispatch():
    a,m=guard()
    assert read(OUT/'pilot_validation.json')['status']=='POSITION_FACTOR_FIRST_PAIR_PASS'
    p=OUT/'dispatch/train.json';p.parent.mkdir(parents=True,exist_ok=True)
    with (p.parent/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if p.exists():
            job=read(p)['job_id']
        else:
            cmd=['sbatch','--parsable','--array='+','.join(map(str,m['indices']))+'%11',str(SCRIPT),'worker']
            job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            assert job.isdigit();write(p,dict(job_id=job,command=cmd,authority=bind(AUTH)))
        j=p.parent/'join.json'
        if not j.exists():
            cmd=['sbatch','--parsable','--partition=dev_cpuonly,cpuonly','--gres=none','--mem=16G',
                 '--dependency=afterok:'+job,str(SCRIPT),'join']
            jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            assert jid.isdigit();write(j,dict(job_id=jid,command=cmd,authority=bind(AUTH)))
        emit(status='POSITION_FACTOR_ARRAY_SUBMITTED',array=job,join=read(j)['job_id'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','self-test','pilot','worker','dispatch','join'])
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare();return 0
    if args.stage=='self-test':emit(**P.self_test());return 0
    if args.stage=='pilot':return gpu(0,True)
    if args.stage=='worker':return gpu(args.index)
    if args.stage=='dispatch':dispatch();return 0
    return join()


if __name__=='__main__':
    sys.exit(main())
