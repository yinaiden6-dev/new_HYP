#!/usr/bin/env python3
"""Label-blind H593 RoMa coordinate-precision intervention, resumable by 16 pairs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_original7_train128_roma_v1 as M
from rc_roma_coordinate_precision_v1 import LEVELS,CoordinateHook,quantize,self_test
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature
import rc_roma_coordinate_intermediates_v2 as I
OUT=ROOT/'results/rc_h593_roma_coordinate_precision_v2'
AUTH=ROOT/'registry/rc_h593_roma_coordinate_precision_authority_v2_20260922.json'
PLAN=ROOT/'plan/RC_H593_ROMA_COORDINATE_PRECISION_V2_20260922.md'
WORKERS=OUT/'workers.json'
PROFILE=M.PROFILE


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def read(p):return json.loads(Path(p).read_text())


def bind(p):
    p=Path(p).absolute();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest())


def checked(b):
    need(bind(b['path'])==b,'SOURCE_DRIFT:'+b['path']);return Path(b['path'])


def check_profile(profile):
    # Historical exports are provenance, not inputs to this new experiment.
    for k,b in profile['sources'].items():
        if k not in ('old_roma','old_train56_export'):checked(b)
    for tree in profile['source_trees'].values():
        for b in tree['python_files']:checked(b)


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists():need(p.read_text()==text,'IMMUTABLE:'+str(p));return
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');tmp.write_text(text);os.link(tmp,p);tmp.unlink()


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);need(not p.exists(),'IMMUTABLE_PART')
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink()


def prepare():
    need(not AUTH.exists(),'NEW_AUTHORITY')
    source=ROOT/'results/rc_crisp_manual_baseline_v1/H593_workers.json';workers=read(source)['records']
    need(len(workers)==593 and [w['execution_ordinal'] for w in workers]==list(range(593)),'ALL593_ORDER')
    keep=('execution_ordinal','query_id','source_index','source_query_id','source_image_sha256','raw','roma')
    workers=[{k:w[k] for k in keep} for w in workers]
    write(WORKERS,dict(records=workers,source_manifest=bind(source),label_reads=0))
    profile=read(PROFILE)
    check_profile(profile)
    codes=dict(program=Path(__file__),hook=ROOT/'programs/rc_roma_coordinate_precision_v1.py',
        intermediate_code=Path(I.__file__),roma_adapter=Path(M.__file__),feature_formula=ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',plan=PLAN,
        launcher=ROOT/'slurm/rc_h593_roma_coordinate_precision_v2.sbatch')
    write(AUTH,dict(status='H593_ROMA_COORDINATE_PRECISION_AUTHORIZED',user_authorization='2026-09-22: RoMa matching coordinate precision, not visibility-map granularity',
        code_sources={k:bind(v) for k,v in codes.items()},workers=bind(WORKERS),profile=bind(PROFILE),
        levels=LEVELS,pilot_index=0,total_queries=593,candidates_per_query=128,pairs_per_part=16,
        intervention='Round prev_warp independently in x,y before EACH RoMa refiner; both AB and BA, both LR and HR passes; no clipping',
        quantizer='Q_B(z)=round(z*B/2)*(2/B), B=256,64,16; normalized per-axis displacement <=1/B plus float rounding',
        native='Unmodified precise setting: LR800, HR1280, bidirectional. Every native pair must reproduce sealed overlap-derived token maps and four scores bit exactly.',
        terminal_control='Quantize FINAL warps only: overlap unchanged, scorer does not read warp; negative dependency control.',
        scope='Coordinate sampling precision inside refinement, not match error against ground truth and not switching RoMa inference presets',
        full_workers_gate='Pilot index0 all128 pairs + four levels must validate before indices1..592',
        evaluation='Same H593 grouped fivefold: frozen COST1 and CE heads, plus matched same-loss refits. No 496-head selection. All593 denominator, no identity annotations beyond original retrieval labels.',
        forbidden=['D1-MI','formal392','GroZi','ownership','SAM','new encoder training'],external_GO=False,intermediate_retention='Per candidate, all levels: token-aligned final coordinate centers/means/covariances, 64x64 warp samples, overlap weights, MaxSim argmax and token contributions; coarse matcher warp/confidence. Frozen image/token source bindings preserved. Full1280 dense tensors not archived; no full-resolution replay claim.'))
    print(dict(status='PREPARED',authority=bind(AUTH)),flush=True)


def guard(stage,index):
    a=read(AUTH);need(a['status']=='H593_ROMA_COORDINATE_PRECISION_AUTHORIZED','AUTHORITY')
    for b in a['code_sources'].values():checked(b)
    need(a['code_sources']['program']==bind(__file__),'PROGRAM_SHA');checked(a['profile'])
    check_profile(read(a['profile']['path']))
    workers=read(checked(a['workers']))['records']
    allowed={Path(a['workers']['path']).resolve(),Path(a['profile']['path']).resolve()}
    if stage in ('worker','verify'):
        need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');need(index in range(593),'QUERY_INDEX')
        for kind in ('raw','roma'):allowed.update(Path(b['path']).resolve() for b in workers[index][kind].values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(k in s for k in ('curator_roles','/target_join/','d1-mi','d1_mi','formal392','/grozi/','rc_opened_','/reports/')),'LABEL_OR_PROTECTED_READ')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own:
                first=p.relative_to(OUT).parts[0]
                own=first in ('workers.json','preflight.json',f'.preflight.json.{os.getpid()}.tmp',f'query{index:03d}' if index is not None else '', 'query000')
            need(own or p in allowed,'UNLISTED_RESULT:'+str(p))
    sys.addaudithook(audit)
    if stage!='preflight':
        p=read(OUT/'preflight.json');need(p['status']=='ROMA_COORDINATE_PRECISION_SYNTHETIC_PASS' and p['authority']==bind(AUTH),'PREFLIGHT')
        if index!=0:
            p=read(OUT/'query000/validation.json');need(p['status']=='ROMA_COORDINATE_QUERY_PASS' and p['authority']==bind(AUTH),'PILOT_MUST_PASS')
    return a,workers


def load_input(w):
    payloads=[]
    for kind in ('raw','roma'):
        b=w[kind];r=read(checked(b['receipt']));v=read(checked(b['validation']))
        need(r['payload']==v['payload']==b['payload'],'ENVELOPE_PAYLOAD')
        need(v.get('receipt',b['receipt'])==b['receipt'],'ENVELOPE_RECEIPT')
        need(v['status'].endswith(('CPU_REPLAY_PASS','_CPU_PASS')),'VALIDATED_SOURCE')
        payloads.append(torch.load(checked(b['payload']),map_location='cpu',weights_only=True,mmap=True))
    raw,roma=payloads;i=w['source_index'];q=raw['records'][i];g=roma['records'][i]
    need(q['query_id']==g['query_id']==w['source_query_id'],'SOURCE_QUERY')
    need(q['query_source_sha256']==g['query_source_sha256']==w['source_image_sha256'],'SOURCE_IMAGE')
    need(q['candidate_physical_rows']==g['candidate_physical_rows'] and len(g['candidates'])==128,'C128')
    need(q['query_tokens_sha256']==g['query_tokens_sha256']==M.token_sha(q['query_tokens']),'QUERY_TOKEN_SHA')
    return q,g,raw['references']


def c4(core,q,r,wq,wr):
    real,mass,_=core.score(q,r,wq,wr)
    qc=core.score(q,r,wq.roll(max(1,len(wq)//2)),wr)[0]
    rc=core.score(q,r,wq,wr.roll(max(1,len(wr)//2)))[0]
    return dict(zip(M.SCORE_KEYS,map(float,(real,mass,qc,rc))))


def worker(a,w,index):
    folder=OUT/f'query{index:03d}'
    if (folder/'validation.json').exists():
        v=read(folder/'validation.json');need(v['authority']==bind(AUTH) and v['payload']==bind(folder/'payload.json'),'RESUME_SEAL');return
    started=time.monotonic();q,old,refs=load_input(w);profile=read(PROFILE)
    core=M.legacy_core(profile);model=M.gpu_model(profile);hook=CoordinateHook(model)
    captured={}
    def capture(module,args,output):
        captured.clear();captured.update({k:output[k].detach().cpu().clone() for k in ('warp_AB','warp_BA','confidence_AB','confidence_BA')})
    capture_handle=model.matcher.register_forward_hook(capture)
    need(model.H_lr==model.W_lr==800 and model.H_hr==model.W_hr==1280 and model.bidirectional,'UNCHANGED_PRECISE')
    qpath=Path(q['query_source_path']);need(bind(qpath)['sha256']==q['query_source_sha256'],'QUERY_IMAGE_BYTES')
    qgeom,_=M.geometry(qpath,q['query_source_sha256'],'coordinate-query',q['query_grid_shape'],q['processor_input_frame'],profile['sources']['processor']['sha256'])
    qimage=core.oriented(qpath);parts=[]
    for part in range(8):
        dest=folder/f'part{part:02d}.pt'
        if dest.exists():
            existing=torch.load(dest,map_location='cpu',weights_only=True)
            need(existing['authority']==bind(AUTH) and existing['query_id']==w['query_id'] and existing['part']==part,'PART_RESUME')
            parts.append(bind(dest));continue
        pairs=[]
        for pos in range(part*16,(part+1)*16):
            physical=q['candidate_physical_rows'][pos];r=refs[int(physical)];prior=old['candidates'][pos]
            need(physical==prior['physical_row'] and r['tokens_sha256']==prior['reference_tokens_sha256']==M.token_sha(r['tokens']),'REFERENCE_BINDING')
            rp=Path(r['source_path']);need(bind(rp)['sha256']==r['source_image_sha256'],'REF_IMAGE_BYTES')
            rg,_=M.geometry(rp,r['source_image_sha256'],'coordinate-reference',r['grid_shape'],'DECODED_RAW_BEFORE_EXIF',profile['sources']['processor']['sha256'])
            ri=core.oriented(rp);arms={};native=None;coarse=None
            for name in LEVELS:
                hook.select(name);pred=model.match(qimage,ri)
                if coarse is None:coarse=dict(captured)
                else:need(all(torch.equal(coarse[k],captured[k]) for k in coarse),'UNCHANGED_COARSE_MATCHER')
                u=core.cell_means(pred['overlap_AB'][0,...,0].cpu(),qgeom)
                v=core.cell_means(pred['overlap_BA'][0,...,0].cpu(),rg)
                scalars=c4(core,q['query_tokens'],r['tokens'],u,v)
                for weights in (u,v):need(weights.dtype==torch.float64 and bool(torch.isfinite(weights).all()) and bool(((weights>=0)&(weights<=1)).all()),'VALID_MAP')
                if name=='NATIVE':
                    need(M.bit_equal(u,prior['query_visibility']) and M.bit_equal(v,prior['reference_visibility']),'NATIVE_MAP_DRIFT')
                    need(all(float(scalars[k]).hex()==float(prior['old_scores'][k]).hex() for k in M.SCORE_KEYS),'NATIVE_C4_DRIFT')
                    native=pred
                    terminal_changed=sum(int(torch.count_nonzero(quantize(pred[k],16)-pred[k])) for k in ('warp_AB','warp_BA'))
                    need(terminal_changed>0,'NONTRIVIAL_TERMINAL_CONTROL')
                    terminal=dict(pred,warp_AB=quantize(pred['warp_AB'],16),warp_BA=quantize(pred['warp_BA'],16))
                    need(terminal['overlap_AB'] is pred['overlap_AB'] and terminal['overlap_BA'] is pred['overlap_BA'],'TERMINAL_OVERLAP_UNCHANGED')
                    diagnostics=dict(terminal_coordinate_elements_changed=terminal_changed,terminal_score_dependency=False)
                else:
                    need(len(hook.calls)==12 and sum(c['changed'] for c in hook.calls)>0,'ALL12_BIDIRECTIONAL_REFINER_INTERVENTIONS')
                    diagnostics=dict(calls=list(hook.calls),query_map_l1=float((u-arms['NATIVE']['query_visibility']).abs().mean()),
                        reference_map_l1=float((v-arms['NATIVE']['reference_visibility']).abs().mean()),
                        final_warp_RMS_normalized={k:float(((pred[k]-native[k]).double()**2).mean().sqrt()) for k in ('warp_AB','warp_BA')})
                intermediate=dict(query_coordinates=I.coordinate_record(pred['warp_AB'],qgeom),reference_coordinates=I.coordinate_record(pred['warp_BA'],rg),content=I.content_record(q['query_tokens'],r['tokens'],u,v))
                need(abs(float(intermediate['content']['query_token_score_contribution'].sum())-scalars[M.SCORE_KEYS[0]])<2e-12,'TOKEN_SCORE_DECOMPOSITION')
                arms[name]=dict(query_visibility=u,reference_visibility=v,scores=scalars,diagnostics=diagnostics,intermediate=intermediate)
                if name!='NATIVE':del pred
            pairs.append(dict(candidate_position=pos,physical_row=int(physical),reference_tokens_sha256=r['tokens_sha256'],arms=arms,coarse_matcher=coarse))
            del native,terminal
        value=dict(authority=bind(AUTH),query_id=w['query_id'],part=part,pairs=pairs)
        save(dest,value);parts.append(bind(dest));print(dict(event='COORDINATE_PART_READY',index=index,part=part,pairs=(part+1)*16,seconds=time.monotonic()-started),flush=True)
    hook.close();capture_handle.remove();del model
    records=[]
    for b in parts:records+=torch.load(checked(b),map_location='cpu',weights_only=True)['pairs']
    raw=list(map(float,q['candidate_raw_scores']));ranked=sorted(range(128),key=lambda i:(-raw[i],q['candidate_physical_rows'][i]));win=ranked[0];challengers=[i for i in range(128) if i!=win]
    modes={}
    for name in LEVELS:
        evidence={p['candidate_position']:p['arms'][name]['scores'] for p in records}
        x=torch.stack([candidate_feature(raw,evidence,c,win) for c in challengers])
        modes[name]=dict(X=x.tolist(),scores=[evidence[i] for i in range(128)])
    write(folder/'payload.json',dict(authority=bind(AUTH),query_id=w['query_id'],execution_ordinal=index,source_image_sha256=w['source_image_sha256'],
        candidate_physical_rows=q['candidate_physical_rows'],raw_ranked_physical_rows=q['raw_ranked_physical_rows'],winner=win,challenger_positions=challengers,
        parts=parts,modes=modes,terminal_coordinate_only_control='NATIVE identical by unchanged overlaps and scorer dependency',label_reads=0,model_updates=0))
    write(folder/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(0)))
    subprocess.run([sys.executable,__file__,'verify','--index',str(index)],check=True)


def verify(a,w,index):
    folder=OUT/f'query{index:03d}';p=read(folder/'payload.json');need(p['authority']==bind(AUTH),'OWN_AUTHORITY')
    q,old,refs=load_input(w);pairs=[];checks=0
    for b in p['parts']:
        chunk=torch.load(checked(b),map_location='cpu',weights_only=True);need(chunk['authority']==bind(AUTH) and chunk['query_id']==w['query_id'],'PART_BINDING');pairs+=chunk['pairs']
    need([v['candidate_position'] for v in pairs]==list(range(128)) and [v['physical_row'] for v in pairs]==q['candidate_physical_rows'],'COMPLETE128')
    for pair in pairs:
        pos=pair['candidate_position'];r=refs[pair['physical_row']]
        for name,arm in pair['arms'].items():
            scores=M.replay_c4(q['query_tokens'],r['tokens'],arm['query_visibility'],arm['reference_visibility'])
            need(all(float(scores[k]).hex()==float(arm['scores'][k]).hex()==float(p['modes'][name]['scores'][pos][k]).hex() for k in M.SCORE_KEYS),'INDEPENDENT_C4_BITS');checks+=4
            need(arm['intermediate']['query_coordinates']['center_xy'].shape==(len(arm['query_visibility']),2),'QUERY_COORDINATE_AXIS')
            need(arm['intermediate']['reference_coordinates']['center_xy'].shape==(len(arm['reference_visibility']),2),'REFERENCE_COORDINATE_AXIS')
            replay=I.content_record(q['query_tokens'],r['tokens'],arm['query_visibility'],arm['reference_visibility'])
            need(all(torch.equal(v,arm['intermediate']['content'][k]) for k,v in replay.items()),'MAXSIM_INTERMEDIATE_REPLAY')
            if name=='NATIVE':need(all(float(scores[k]).hex()==float(old['candidates'][pos]['old_scores'][k]).hex() for k in M.SCORE_KEYS),'OLD_NATIVE_BITS')
    raw=list(map(float,q['candidate_raw_scores']));winner=p['winner'];cs=p['challenger_positions']
    need(winner==max(range(128),key=lambda i:(raw[i],-q['candidate_physical_rows'][i])) and cs==[i for i in range(128) if i!=winner],'RAW_ACTION_AXIS')
    for name in LEVELS:
        evidence={i:s for i,s in enumerate(p['modes'][name]['scores'])}
        x=torch.stack([candidate_feature(raw,evidence,c,winner) for c in cs])
        need(torch.equal(x,torch.tensor(p['modes'][name]['X'],dtype=torch.float64)),'FEATURE_REPLAY')
    write(folder/'validation.json',dict(status='ROMA_COORDINATE_QUERY_PASS',authority=bind(AUTH),payload=bind(folder/'payload.json'),query_id=w['query_id'],index=index,
        candidates=128,levels=list(LEVELS),independent_C4_scalars=checks,native_maps_and_scores_bit_exact=True,label_reads=0,model_updates=0))
    print(dict(event='QUERY_VALIDATED',index=index,checks=checks),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','preflight','worker','verify'));ap.add_argument('--index',type=int);args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a,ws=guard(args.stage,args.index)
        if args.stage=='preflight':
            p=self_test();p['intermediates']=I.self_test();p['authority']=bind(AUTH);write(OUT/'preflight.json',p);print(p)
        elif args.stage=='worker':worker(a,ws[args.index],args.index)
        else:verify(a,ws[args.index],args.index)
