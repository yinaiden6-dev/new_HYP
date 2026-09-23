#!/usr/bin/env python3
"""Label-blind full-C128 qualification of standalone features and fusion inputs."""
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
from torch.nn import functional as TF

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'), str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
import rc_feature_fusion_core_v1 as F
M=C.M
read,write,save,bind,checked,need=C.read,C.write,C.save,C.bind,C.checked,C.need
OUT=ROOT/'results/rc_h593_feature_fusion_pilot_v1'
AUTH=ROOT/'registry/rc_h593_feature_fusion_pilot_authority_v1_20260922.json'
PLAN=ROOT/'plan/RC_H593_FEATURE_FUSION_V1_20260922.md'
LAUNCH=ROOT/'slurm/rc_h593_feature_fusion_pilot_v1.sbatch'


def image_item(path, sha, grid, frame, tokens_sha):
    item=dict(path=str(path),image_sha256=sha,grid=list(grid),frame=frame,tokens_sha256=tokens_sha)
    item['key']=hashlib.sha256(json.dumps({k:v for k,v in item.items() if k!='path'},sort_keys=True).encode()).hexdigest()
    return item


def prepare():
    need(not AUTH.exists(),'NEW_AUTHORITY')
    worker=read(C.WORKERS)['records'][0]
    q,old,refs=C.load_input(worker)
    qi=image_item(q['query_source_path'],q['query_source_sha256'],q['query_grid_shape'],q['processor_input_frame'],q['query_tokens_sha256'])
    images={qi['key']:qi};candidates=[]
    for pos,physical in enumerate(q['candidate_physical_rows']):
        r=refs[physical]
        item=image_item(r['source_path'],r['source_image_sha256'],r['grid_shape'],'DECODED_RAW_BEFORE_EXIF',r['tokens_sha256'])
        images[item['key']]=item
        candidates.append(dict(position=pos,physical_row=physical,image_key=item['key']))
    write(OUT/'manifest.json',dict(worker=worker,source_workers=bind(C.WORKERS),query_image_key=qi['key'],
        images=list(images.values()),candidates=candidates,selection='Original execution_ordinal 0 and every natural C128 candidate; no target labels',label_reads=0))
    sources=[Path(__file__),Path(F.__file__),Path(C.__file__),Path(M.__file__),
        ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',
        ROOT/'src/rc_aslo_xf/colnomic_dino_canonical_geometry_v1.py',
        ROOT/'src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py',PLAN,LAUNCH]
    C.check_profile(read(M.PROFILE))
    write(AUTH,dict(status='FEATURE_FUSION_PILOT_AUTHORIZED',user_authorization='2026-09-22 user: 开始; start four feature sources by matcher evidence experiment',
        code_sources=[bind(p) for p in sources],manifest=bind(OUT/'manifest.json'),profile=bind(M.PROFILE),
        feature_sources=list(F.SOURCES),components=F.WIDTHS,arms=[s+'_'+m for s in F.SOURCES for m in ('FREE','ROMA_WEIGHTED')],
        selection='execution ordinal0, all128 candidates; first candidate only for full RoMa feature hook qualification',
        trainable_adapter_parameters=4096,source_projection_seed=20260922,initialization_seed=17,
        training_updates=0,label_reads=0,scope='Engineering extraction, zero-adapter replay and gradients; not natural-data training or accuracy',
        full_training_requires='Separate frozen training execution contract after engineering timing; original grouped fivefold and retrieval labels only',
        prohibited=['D1-MI','formal392','GroZi','ISIC','heldout labels','ownership','SAM'],external_GO=False))
    print(dict(event='PREPARED',images=len(images),candidates=128,authority=bind(AUTH)),flush=True)


def guard(stage):
    a=read(AUTH);need(a['status']=='FEATURE_FUSION_PILOT_AUTHORIZED','AUTHORITY')
    for b in a['code_sources']:checked(b)
    checked(a['profile']);manifest=read(checked(a['manifest']))
    C.check_profile(read(M.PROFILE))
    if stage in ('worker','verify'):need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allowed={Path(a['manifest']['path']).resolve(),Path(a['profile']['path']).resolve()}
    for kind in ('raw','roma'):
        allowed.update(Path(b['path']).resolve() for b in manifest['worker'][kind].values())
    allowed.update(Path(i['path']).resolve() for i in manifest['images'])
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(k in s for k in ('curator_roles','/target_join/','d1-mi','d1_mi','formal392','/grozi/','/isic/','/reports/')),'LABEL_OR_PROTECTED_READ')
        if ROOT/'results' in p.parents:need(OUT in p.parents or p in allowed,'UNLISTED_RESULT:'+str(p))
    sys.addaudithook(audit)
    if stage!='preflight':
        v=read(OUT/'preflight.json');need(v['authority']==bind(AUTH) and v['status']=='FEATURE_FUSION_CORE_PASS','PREFLIGHT')
    return a,manifest


def geometry(item,profile):
    need(bind(item['path'])['sha256']==item['image_sha256'],'IMAGE_SHA')
    return M.geometry(Path(item['path']),item['image_sha256'],item['key'],item['grid'],item['frame'],profile['sources']['processor']['sha256'])


def image_arrays(model,image):
    t=model._load_image(image)
    return [TF.interpolate(t,size=(side,side),mode='bicubic',align_corners=False,antialias=True) for side in (800,1280)]


def feature_file(item):return OUT/'images'/item['key']/'payload.pt'


def load_features(item):
    path=feature_file(item);v=read(path.with_name('validation.json'))
    need(v['status']=='FUSION_IMAGE_FEATURES_PASS' and v['authority']==bind(AUTH),'IMAGE_SEAL')
    p=torch.load(checked(v['payload']),map_location='cpu',weights_only=True)
    need(p['item']==item and set(p['components'])==set(F.COMPONENTS),'FEATURE_ITEM')
    return p


@torch.inference_mode()
def extract(model,image,item,tokens,geom,geom_meta):
    path=feature_file(item)
    if path.with_name('validation.json').exists():return load_features(item)
    need(not path.exists(),'UNSEALED_IMAGE_PAYLOAD_REQUIRES_REVIEW')
    lr,hr=image_arrays(model,image);features={};checks={}
    coarse=model.f(lr)
    need(len(coarse)==2,'COARSE_LAYERS')
    for name,feature in zip(F.COMPONENTS[:2],coarse):features[name],checks[name]=F.pool_cells(feature,geom)
    del coarse
    for stage,tensor in (('lr',lr),('hr',hr)):
        fine=model.refiner_features(tensor)
        need(set(fine)=={1,2,4},'FINE_SCALES')
        for scale,feature in fine.items():
            name=f'fine_{stage}_{scale}';features[name],checks[name]=F.pool_cells(feature,geom)
        del fine
    for name,width in F.WIDTHS.items():need(features[name].shape==(len(tokens),width),'FEATURE_WIDTH')
    projected={source:F.projected_input(tokens,features,source) for source in F.SOURCES}
    payload=dict(authority=bind(AUTH),item=item,geometry=geom_meta,
        cell_boxes_xyxy=geom.cell_boxes_xyxy,valid_patch_mask=geom.valid_patch_mask,
        original_colnomic_tokens=tokens.clone(),components=features,projected_inputs=projected,pooling_checks=checks,
        full_resolution_activations_saved=False,model_updates=0,label_reads=0)
    save(path,payload)
    write(path.with_name('validation.json'),dict(status='FUSION_IMAGE_FEATURES_PASS',authority=bind(AUTH),payload=bind(path),
        key=item['key'],token_count=len(tokens),component_widths=F.WIDTHS,
        max_direct_mean_error=max(c['max_direct_mean_error'] for c in checks.values())))
    return payload


@torch.inference_mode()
def qualify_hook(model,images,geoms,stored,old,core):
    counts=dict(coarse=0,fine=0);comparisons=[]
    def coarse_hook(module,args,output):
        side=counts['coarse'];need(side<2,'COARSE_CALL_COUNT');counts['coarse']+=1
        for name,feature in zip(F.COMPONENTS[:2],output):
            pooled,_=F.pool_cells(feature,geoms[side])
            need(M.bit_equal(pooled,stored[side]['components'][name]),'STANDALONE_COARSE_PARITY')
            comparisons.append(dict(side=side,component=name,bit_exact=True))
    def fine_hook(module,args,output):
        k=counts['fine'];counts['fine']+=1;need(k<4,'FINE_CALL_COUNT')
        side=k%2;stage='lr' if k<2 else 'hr'
        for scale,feature in output.items():
            name=f'fine_{stage}_{scale}';pooled,_=F.pool_cells(feature,geoms[side])
            need(M.bit_equal(pooled,stored[side]['components'][name]),'STANDALONE_FINE_PARITY')
            comparisons.append(dict(side=side,component=name,bit_exact=True))
    handles=[model.f.register_forward_hook(coarse_hook),model.refiner_features.register_forward_hook(fine_hook)]
    try:pred=model.match(*images)
    finally:
        for h in handles:h.remove()
    u=core.cell_means(pred['overlap_AB'][0,...,0].cpu(),geoms[0])
    v=core.cell_means(pred['overlap_BA'][0,...,0].cpu(),geoms[1])
    need(M.bit_equal(u,old['query_visibility']) and M.bit_equal(v,old['reference_visibility']),'NATIVE_OVERLAP_BITS')
    need(counts==dict(coarse=2,fine=4) and len(comparisons)==16,'ALL_COMPONENT_PARITY')
    return dict(status='STANDALONE_MATCHER_FEATURE_POOL_PARITY_PASS',comparisons=comparisons,
                overlap_bit_exact=True,full_match_calls=1,labels_used=0,
                scope='All pooled feature elements match; not a saved full dense activation comparison')


def worker(a,manifest):
    if (OUT/'payload.pt').exists():
        verify(a,manifest);return
    started=time.monotonic();q,old,refs=C.load_input(manifest['worker']);profile=read(M.PROFILE)
    core=M.legacy_core(profile);model=M.gpu_model(profile)
    need(model.H_lr==800 and model.H_hr==1280 and model.bidirectional,'PRECISE_SETTING')
    items={r['key']:r for r in manifest['images']}
    qi=items[manifest['query_image_key']];qgeom,qmeta=geometry(qi,profile);qimage=core.oriented(qi['path'])
    qp=extract(model,qimage,qi,q['query_tokens'],qgeom,qmeta)
    first=manifest['candidates'][0];ri=items[first['image_key']];r=refs[first['physical_row']]
    rgeom,rmeta=geometry(ri,profile);rimage=core.oriented(ri['path'])
    rp=extract(model,rimage,ri,r['tokens'],rgeom,rmeta)
    hook=qualify_hook(model,(qimage,rimage),(qgeom,rgeom),(qp,rp),old['candidates'][0],core)
    write(OUT/'hook_qualification.json',dict(authority=bind(AUTH),**hook))
    # Standalone extraction must never accidentally execute the matcher or refiners.
    def forbidden(module,args):raise RuntimeError('MATCHER_OR_REFINER_USED_DURING_STANDALONE_ENCODING')
    handles=[model.matcher.register_forward_pre_hook(forbidden),model.refiners.register_forward_pre_hook(forbidden)]
    handles.extend(module.register_forward_pre_hook(forbidden) for module in model.refiners.values())
    for pos,candidate in enumerate(manifest['candidates']):
        r=refs[candidate['physical_row']];item=items[candidate['image_key']]
        geom,meta=geometry(item,profile)
        extract(model,core.oriented(item['path']),item,r['tokens'],geom,meta)
        if pos%8==0:print(dict(event='STANDALONE_FEATURES',candidates=pos+1,total=128,seconds=time.monotonic()-started),flush=True)
    for h in handles:h.remove()
    del model;torch.cuda.empty_cache()
    pairs=[];gradients=[]
    adapters={s:F.OutputAdapter() for s in F.SOURCES}
    for candidate in manifest['candidates']:
        pos=candidate['position'];r=refs[candidate['physical_row']];rp=load_features(items[candidate['image_key']]);prior=old['candidates'][pos]
        need(prior['physical_row']==candidate['physical_row'] and M.token_sha(r['tokens'])==rp['item']['tokens_sha256'],'REFERENCE_AXIS')
        u=prior['query_visibility'];v=prior['reference_visibility'];arms={}
        with torch.no_grad():
            for source in F.SOURCES:
                zq,gq=adapters[source](q['query_tokens'],qp['projected_inputs'][source])
                zr,gr=adapters[source](r['tokens'],rp['projected_inputs'][source])
                need(torch.equal(zq,q['query_tokens'].double()) and torch.equal(zr,r['tokens'].double()) and not gq.any() and not gr.any(),'ZERO_ADAPTER_EXACT')
            for mode in ('FREE','ROMA_WEIGHTED'):
                uu,vv=(u,v) if mode=='ROMA_WEIGHTED' else (torch.ones_like(u),torch.ones_like(v))
                scores,trace=F.scores_and_trace(zq,zr,uu,vv)
                if mode=='ROMA_WEIGHTED':need(all(float(scores[i]).hex()==float(prior['old_scores'][key]).hex() for i,key in enumerate(M.SCORE_KEYS)),'ALL128_NATIVE_SCORE_BITS')
                arms[mode]=dict(scores=scores.detach(),trace={k:v.detach() for k,v in trace.items()})
        if pos==0:
            for source in F.SOURCES:
                for mode in ('FREE','ROMA_WEIGHTED'):
                    model=adapters[source];model.zero_grad(set_to_none=True)
                    zq,_=model(q['query_tokens'],qp['projected_inputs'][source]);zr,_=model(r['tokens'],rp['projected_inputs'][source])
                    uu,vv=(u,v) if mode=='ROMA_WEIGHTED' else (torch.ones_like(u),torch.ones_like(v))
                    score=F.scores_and_trace(zq,zr,uu,vv)[0][0];score.backward()
                    norm=float(model.up.weight.grad.norm())
                    need(math.isfinite(norm) and norm>0,'NATURAL_UNLABELLED_GRADIENT')
                    gradients.append(dict(source=source,mode=mode,up_gradient_norm=norm,optimizer_steps=0,label_reads=0))
        pairs.append(dict(position=pos,physical_row=candidate['physical_row'],image_key=candidate['image_key'],
            query_visibility=u,reference_visibility=v,arms=arms))
        if pos%16==0:print(dict(event='ZERO_ADAPTER_REPLAY',candidates=pos+1,total=128,seconds=time.monotonic()-started),flush=True)
    raw=list(map(float,q['candidate_raw_scores']));winner=max(range(128),key=lambda i:(raw[i],-q['candidate_physical_rows'][i]))
    modes={}
    for source in F.SOURCES:
        for mode in ('FREE','ROMA_WEIGHTED'):
            scores=torch.stack([p['arms'][mode]['scores'] for p in pairs])
            modes[source+'_'+mode]=dict(c4=scores,X=F.differentiable_features(raw,scores,winner),
                trace_reference=mode,zero_gate=True)
    save(OUT/'payload.pt',dict(authority=bind(AUTH),manifest=bind(OUT/'manifest.json'),query_id=q['query_id'],
        query_image_key=qi['key'],candidate_physical_rows=q['candidate_physical_rows'],candidate_raw_scores=raw,
        winner=winner,challenger_positions=[i for i in range(128) if i!=winner],pairs=pairs,modes=modes,
        zero_adapter_state_dicts={s:{k:v.detach() for k,v in m.state_dict().items()} for s,m in adapters.items()},
        natural_unlabelled_gradient_checks=gradients,training_updates=0,label_reads=0))
    write(OUT/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started,
        unique_image_geometry_count=len(items),full_match_calls=1,gpu=torch.cuda.get_device_name(0),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),images_per_second=len(items)/(time.monotonic()-started)))
    subprocess.run([sys.executable,__file__,'verify'],check=True)


def verify(a,manifest):
    q,old,refs=C.load_input(manifest['worker']);p=torch.load(OUT/'payload.pt',map_location='cpu',weights_only=True)
    need(p['authority']==bind(AUTH) and p['manifest']==bind(OUT/'manifest.json'),'PAYLOAD_SOURCE')
    need(p['candidate_physical_rows']==q['candidate_physical_rows'] and len(p['pairs'])==128,'FULL_C128')
    images={i['key']:i for i in manifest['images']};image_seals=[]
    for item in images.values():
        value=load_features(item);need(M.token_sha(value['original_colnomic_tokens'])==item['tokens_sha256'],'COLNOMIC_TOKEN_BINDING')
        for name,width in F.WIDTHS.items():
            x=value['components'][name];need(x.shape==(math.prod(item['grid']),width) and bool(torch.isfinite(x).all()),'FINITE_ALIGNED_FEATURES')
        for source in F.SOURCES:
            direct=F.projected_input(value['original_colnomic_tokens'],value['components'],source)
            need(torch.equal(direct,value['projected_inputs'][source]),'INDEPENDENT_PROJECTION_REPLAY')
        image_seals.append(bind(feature_file(item).with_name('validation.json')))
    scalar_checks=0
    for pair in p['pairs']:
        pos=pair['position'];need(pair['physical_row']==q['candidate_physical_rows'][pos],'CANDIDATE_AXIS')
        r=refs[pair['physical_row']];u=pair['query_visibility'];v=pair['reference_visibility']
        need(M.bit_equal(u,old['candidates'][pos]['query_visibility']) and M.bit_equal(v,old['candidates'][pos]['reference_visibility']),'CACHED_WEIGHTS')
        for mode in ('FREE','ROMA_WEIGHTED'):
            uu,vv=(u,v) if mode=='ROMA_WEIGHTED' else (torch.ones_like(u),torch.ones_like(v))
            literal=M.replay_c4(q['query_tokens'],r['tokens'],uu,vv)
            score=pair['arms'][mode]['scores']
            need(all(float(score[i]).hex()==float(literal[k]).hex() for i,k in enumerate(M.SCORE_KEYS)),'INDEPENDENT_SCORE_REPLAY')
            trace=pair['arms'][mode]['trace']
            need(abs(float(trace['query_token_score_contribution'].sum())-float(score[0]))<2e-12,'CONTRIBUTION_SUM')
            # Reconstruct trace separately from saved tokens and maps.
            sim=TF.normalize(q['query_tokens'].double(),dim=1)@TF.normalize(r['tokens'].double(),dim=1).T
            fv,fi=sim.max(1);wv,wi=(sim*vv[None]).max(1)
            need(torch.equal(fi,trace['free_maxsim_reference_token']) and torch.equal(wi,trace['weighted_maxsim_reference_token']),'MATCH_INDEX_REPLAY')
            need(torch.equal(fv,trace['free_maxsim_similarity']),'SIMILARITY_REPLAY')
            contrib=(uu.mean()*vv.mean()).sqrt()*uu*wv/uu.sum().clamp_min(1e-12)
            need(torch.equal(contrib,trace['query_token_score_contribution']),'CONTRIBUTION_REPLAY');scalar_checks+=4
    raw=p['candidate_raw_scores'];winner=p['winner'];need(winner==max(range(128),key=lambda i:(raw[i],-p['candidate_physical_rows'][i])),'WINNER')
    from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature
    for source in F.SOURCES:
        model=F.OutputAdapter();model.load_state_dict(p['zero_adapter_state_dicts'][source]);need(not model.up.weight.any(),'ZERO_GATE_STATE')
        for mode in ('FREE','ROMA_WEIGHTED'):
            rec=p['modes'][source+'_'+mode]
            c4=torch.stack([pair['arms'][mode]['scores'] for pair in p['pairs']]);need(torch.equal(c4,rec['c4']),'C4_AXIS')
            evidence={i:dict(zip(M.SCORE_KEYS,map(float,c4[i]))) for i in range(128)}
            x=torch.stack([candidate_feature(raw,evidence,i,winner) for i in p['challenger_positions']])
            need(torch.equal(x,rec['X']) and x.shape==(127,6),'ORIGINAL_FORMULA_REPLAY')
    hook=read(OUT/'hook_qualification.json');need(hook['authority']==bind(AUTH) and hook['status']=='STANDALONE_MATCHER_FEATURE_POOL_PARITY_PASS','HOOK_QUALIFICATION')
    need(len(p['natural_unlabelled_gradient_checks'])==8 and all(v['up_gradient_norm']>0 for v in p['natural_unlabelled_gradient_checks']),'EIGHT_GRADIENT_ARMS')
    write(OUT/'validation.json',dict(status='FEATURE_FUSION_PILOT_FULL128_PASS',authority=bind(AUTH),payload=bind(OUT/'payload.pt'),
        image_validations=image_seals,hook_qualification=bind(OUT/'hook_qualification.json'),candidates=128,arms=list(p['modes']),
        independent_score_scalar_checks=scalar_checks,all_native_scores_bit_exact=True,train_updates=0,label_reads=0,
        scientific_result=False,next_stage='Freeze full extraction and fold-local training protocol using pilot timing; no accuracy has been evaluated'))
    print(dict(event='FEATURE_FUSION_PILOT_FULL128_PASS',images=len(images),candidates=128,arms=8,scalar_checks=scalar_checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','preflight','worker','verify'));args=parser.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a,m=guard(args.stage)
        if args.stage=='preflight':
            result=F.self_test();result['authority']=bind(AUTH);write(OUT/'preflight.json',result);print(result,flush=True)
        elif args.stage=='worker':worker(a,m)
        else:verify(a,m)
