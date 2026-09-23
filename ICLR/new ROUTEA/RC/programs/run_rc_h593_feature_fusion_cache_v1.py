#!/usr/bin/env python3
"""Deduplicated label-blind H593 feature extraction, reusing the qualified pilot."""
import argparse
from collections import defaultdict
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
import run_rc_h593_feature_fusion_pilot_v1 as P
F,C,M=P.F,P.C,P.M
read,write,save,bind,checked,need=P.read,P.write,P.save,P.bind,P.checked,P.need
OUT=ROOT/'results/rc_h593_feature_fusion_cache_v1'
AUTH=ROOT/'registry/rc_h593_feature_fusion_cache_authority_v1_20260922.json'
PLAN=ROOT/'plan/RC_H593_FEATURE_FUSION_EXECUTION_V1_20260922.md'
LAUNCH=ROOT/'slurm/rc_h593_feature_fusion_cache_v1.sbatch'
CPU_LAUNCH=ROOT/'slurm/rc_h593_feature_fusion_catalog_v1.sbatch'
BATCH=256


def prepare():
    need(not AUTH.exists(),'NEW_AUTHORITY')
    v=read(P.OUT/'validation.json');need(v['status']=='FEATURE_FUSION_PILOT_FULL128_PASS','PILOT_PASS');checked(v['payload'])
    ws=read(C.WORKERS)['records'];need(len(ws)==593,'ALL593')
    sources=[Path(__file__),Path(P.__file__),Path(F.__file__),Path(C.__file__),Path(M.__file__),PLAN,LAUNCH,CPU_LAUNCH]
    write(AUTH,dict(status='FEATURE_FUSION_CACHE_AUTHORIZED',code_sources=[bind(p) for p in sources],
        pilot_authority=bind(P.AUTH),pilot_validation=bind(P.OUT/'validation.json'),pilot_manifest=bind(P.OUT/'manifest.json'),
        profile=bind(M.PROFILE),workers=bind(C.WORKERS),batch_images=BATCH,total_queries=593,candidates_per_query=128,
        train_label_reads=0,heldout_label_reads=0,scope='Single-image frozen descriptors; no trainable weights and no RoMa matcher beyond existing pilot',
        user_authorization='2026-09-22 continue tasks following job5157186',automatic_followup='All unique feature shards and independent completeness join'))
    print(dict(status='PREPARED',authority=bind(AUTH)),flush=True)


def guard(stage):
    a=read(AUTH);need(a['status']=='FEATURE_FUSION_CACHE_AUTHORIZED','AUTHORITY')
    for b in a['code_sources']:checked(b)
    for k in ('pilot_authority','pilot_validation','pilot_manifest','profile','workers'):checked(a[k])
    need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    allowed={Path(a['workers']['path']).resolve()}
    if stage=='catalog':
        for w in read(a['workers']['path'])['records']:
            for k in ('raw','roma'):allowed.update(Path(b['path']).resolve() for b in w[k].values())
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(t in s for t in ('curator_roles','/target_join/','d1-mi','d1_mi','formal392','/grozi/','/isic/','/reports/')),'PROTECTED_OR_LABEL_READ')
        if ROOT/'results' in p.parents:need(OUT in p.parents or P.OUT in p.parents or p in allowed,'UNLISTED_RESULT:'+str(p))
    sys.addaudithook(audit)
    return a


def catalog(a):
    if (OUT/'catalog_validation.json').exists():return submit_extraction()
    workers=read(a['workers']['path'])['records'];groups=defaultdict(list)
    for w in workers:groups[(w['raw']['payload']['path'],w['roma']['payload']['path'])].append(w)
    pilot=read(a['pilot_manifest']['path']);reuse={i['key']:i for i in pilot['images']}
    images={};queries=[];source_records=[]
    def register(item,tokens):
        key=item['key']
        if key in images:return key
        need(M.token_sha(tokens)==item['tokens_sha256'],'INPUT_TOKEN_SHA')
        path=OUT/'inputs'/f'{key}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            need(saved['item']==item and M.bit_equal(saved['tokens'],tokens),'INPUT_RESUME')
        else:save(path,dict(item=item,tokens=tokens.clone(),authority=bind(AUTH)))
        entry=dict(item=item,input=bind(path),pilot_validation=None)
        if key in reuse:
            old=P.load_features(reuse[key]);need(M.bit_equal(old['original_colnomic_tokens'],tokens),'PILOT_REUSE_TOKEN_BITS')
            entry['pilot_validation']=bind(P.feature_file(reuse[key]).with_name('validation.json'))
        images[key]=entry
        return key
    for index,ws in enumerate(groups.values()):
        first=ws[0];payloads=[]
        for kind in ('raw','roma'):
            b=first[kind];v=read(checked(b['validation']));r=read(checked(b['receipt']))
            need(v['payload']==r['payload']==b['payload'] and v['status'].endswith(('CPU_REPLAY_PASS','_CPU_PASS')),'SOURCE_QUALIFIED')
            payloads.append(torch.load(checked(b['payload']),map_location='cpu',weights_only=True,mmap=True))
            source_records.append(b)
        raw,roma=payloads
        for w in ws:
            q,g=raw['records'][w['source_index']],roma['records'][w['source_index']]
            need(q['query_id']==g['query_id']==w['source_query_id'],'QUERY_SOURCE')
            need(q['query_source_sha256']==g['query_source_sha256']==w['source_image_sha256'],'QUERY_IMAGE')
            need(q['candidate_physical_rows']==g['candidate_physical_rows'] and len(g['candidates'])==128,'FULL_C128')
            qi=P.image_item(q['query_source_path'],q['query_source_sha256'],q['query_grid_shape'],q['processor_input_frame'],q['query_tokens_sha256'])
            qkey=register(qi,q['query_tokens']);pairs=[]
            for pos,physical in enumerate(q['candidate_physical_rows']):
                ref=raw['references'][physical];old=g['candidates'][pos]
                need(old['physical_row']==physical and old['reference_tokens_sha256']==ref['tokens_sha256'],'REF_BINDING')
                ri=P.image_item(ref['source_path'],ref['source_image_sha256'],ref['grid_shape'],'DECODED_RAW_BEFORE_EXIF',ref['tokens_sha256'])
                key=register(ri,ref['tokens'])
                pairs.append(dict(position=pos,physical_row=physical,image_key=key,query_visibility=old['query_visibility'].clone(),
                    reference_visibility=old['reference_visibility'].clone(),native_c4=torch.tensor([old['old_scores'][k] for k in M.SCORE_KEYS],dtype=torch.float64)))
            raw_scores=list(map(float,q['candidate_raw_scores']));winner=max(range(128),key=lambda i:(raw_scores[i],-q['candidate_physical_rows'][i]))
            record=dict(authority=bind(AUTH),query_id=w['query_id'],execution_ordinal=w['execution_ordinal'],
                source_image_sha256=w['source_image_sha256'],query_image_key=qkey,candidate_physical_rows=q['candidate_physical_rows'],
                raw_ranked_physical_rows=q['raw_ranked_physical_rows'],candidate_raw_scores=raw_scores,winner=winner,
                challenger_positions=[i for i in range(128) if i!=winner],pairs=pairs,label_reads=0)
            path=OUT/'queries'/f"query{w['execution_ordinal']:03d}.pt"
            if not path.exists():save(path,record)
            else:
                old=torch.load(path,map_location='cpu',weights_only=True)
                need(old['authority']==bind(AUTH) and old['query_id']==w['query_id'] and old['candidate_physical_rows']==record['candidate_physical_rows'],'QUERY_RESUME')
            queries.append(dict(query_id=w['query_id'],execution_ordinal=w['execution_ordinal'],query_image_key=qkey,payload=bind(path)))
        print(dict(event='SOURCE_GROUP_CATALOGUED',groups=index+1,total=len(groups),queries=len(queries),unique_images=len(images)),flush=True)
        del raw,roma,payloads
    queries.sort(key=lambda r:r['execution_ordinal']);need([q['execution_ordinal'] for q in queries]==list(range(593)),'ALL593_AXIS')
    images=dict(sorted(images.items()));new=[k for k,v in images.items() if v['pilot_validation'] is None]
    shards=[new[i:i+BATCH] for i in range(0,len(new),BATCH)];need(0<len(shards)<=46,'ONE_ARRAY_CAPACITY')
    write(OUT/'catalog.json',dict(authority=bind(AUTH),images=images,queries=queries,shards=shards,
        reused_pilot_images=len(images)-len(new),new_images=len(new),source_groups=source_records,label_reads=0))
    write(OUT/'catalog_validation.json',dict(status='FUSION_ALL593_CATALOG_PASS',authority=bind(AUTH),catalog=bind(OUT/'catalog.json'),
        queries=593,unique_image_geometries=len(images),new_images=len(new),shards=len(shards),label_reads=0))
    submit_extraction()


def submit_extraction():
    v=read(OUT/'catalog_validation.json');cat=read(checked(v['catalog']))
    need(v['authority']==bind(AUTH) and v['status']=='FUSION_ALL593_CATALOG_PASS','CATALOG_SEAL')
    def submit(path,args):
        if path.exists():return read(path)['job_id']
        p=subprocess.run(['/usr/bin/sbatch','--parsable',*args],cwd=ROOT,capture_output=True,text=True)
        need(p.returncode==0,'SBATCH:'+p.stderr);job=p.stdout.strip().split(';')[0];need(job.isdigit(),'JOB_ID')
        write(path,dict(authority=bind(AUTH),job_id=job,arguments=args));return job
    array=submit(OUT/'extract_submitted.json',[f"--array=0-{len(cat['shards'])-1}%46",str(LAUNCH)])
    join=submit(OUT/'join_submitted.json',['--dependency=afterok:'+array,'--kill-on-invalid-dep=yes',str(CPU_LAUNCH),'join'])
    print(dict(event='FULL_FEATURE_EXTRACTION_SUBMITTED',array=array,join=join,shards=len(cat['shards']),new_images=cat['new_images']),flush=True)


def feature_path(key):return OUT/'images'/key/'payload.pt'


def feature_binding(entry):
    if entry['pilot_validation']:
        v=read(checked(entry['pilot_validation']));need(v['status']=='FUSION_IMAGE_FEATURES_PASS','PILOT_IMAGE_PASS')
        return v['payload']
    v=read(feature_path(entry['item']['key']).with_name('validation.json'))
    need(v['status']=='FUSION_FULL_IMAGE_PASS' and v['authority']==bind(AUTH),'FULL_IMAGE_PASS')
    return v['payload']


@torch.inference_mode()
def extract(a,index):
    cv=read(OUT/'catalog_validation.json');cat=read(checked(cv['catalog']));need(cv['authority']==bind(AUTH),'CATALOG_AUTH')
    need(index in range(len(cat['shards'])),'SHARD_RANGE')
    folder=OUT/f'shard{index:02d}'
    if (folder/'validation.json').exists():return verify_shard(cat,index)
    profile=read(M.PROFILE);C.check_profile(profile);model=M.gpu_model(profile);core=M.legacy_core(profile);start=time.monotonic()
    def forbidden(module,args):raise RuntimeError('UNEXPECTED_PAIRWISE_ROMA_FORWARD')
    handles=[model.matcher.register_forward_pre_hook(forbidden),model.refiners.register_forward_pre_hook(forbidden)]
    handles.extend(m.register_forward_pre_hook(forbidden) for m in model.refiners.values())
    for num,key in enumerate(cat['shards'][index]):
        entry=cat['images'][key];item=entry['item'];path=feature_path(key)
        if path.with_name('validation.json').exists():checked(feature_binding(entry));continue
        original=torch.load(checked(entry['input']),map_location='cpu',weights_only=True)
        tokens=original['tokens'];need(M.token_sha(tokens)==item['tokens_sha256'],'TOKEN_SHA')
        geom,meta=P.geometry(item,profile);lr,hr=P.image_arrays(model,core.oriented(item['path']))
        components={};checks={};coarse=model.f(lr)
        for name,x in zip(F.COMPONENTS[:2],coarse):components[name],checks[name]=F.pool_cells(x,geom)
        del coarse
        for stage,t in (('lr',lr),('hr',hr)):
            fine=model.refiner_features(t)
            for scale,x in fine.items():
                name=f'fine_{stage}_{scale}';components[name],checks[name]=F.pool_cells(x,geom)
            del fine
        need(set(components)==set(F.COMPONENTS),'EIGHT_COMPONENTS')
        projections={s:F.projected_input(tokens,components,s) for s in F.SOURCES}
        save(path,dict(authority=bind(AUTH),item=item,input=entry['input'],original_colnomic_tokens=tokens.clone(),
            geometry=meta,cell_boxes_xyxy=geom.cell_boxes_xyxy,valid_patch_mask=geom.valid_patch_mask,
            components=components,projected_inputs=projections,pooling_checks=checks,full_resolution_activations_saved=False,
            model_updates=0,label_reads=0))
        write(path.with_name('validation.json'),dict(status='FUSION_FULL_IMAGE_PASS',authority=bind(AUTH),payload=bind(path),
            key=key,max_direct_mean_error=max(c['max_direct_mean_error'] for c in checks.values())))
        if num%16==0:print(dict(event='UNIQUE_IMAGE_ENCODED',shard=index,done=num+1,total=len(cat['shards'][index]),seconds=time.monotonic()-start),flush=True)
    for h in handles:h.remove()
    write(folder/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,images=len(cat['shards'][index]),gpu=torch.cuda.get_device_name(0)))
    verify_shard(cat,index)


def verify_shard(cat,index):
    validations=[]
    for key in cat['shards'][index]:
        entry=cat['images'][key];b=feature_binding(entry);p=torch.load(checked(b),map_location='cpu',weights_only=True)
        need(p['item']==entry['item'] and M.token_sha(p['original_colnomic_tokens'])==entry['item']['tokens_sha256'],'IMAGE_TOKEN_BINDING')
        for s in F.SOURCES:need(torch.equal(F.projected_input(p['original_colnomic_tokens'],p['components'],s),p['projected_inputs'][s]),'PROJECTION_REPLAY')
        validations.append(bind(feature_path(key).with_name('validation.json')))
    write(OUT/f'shard{index:02d}/validation.json',dict(status='FUSION_FEATURE_SHARD_PASS',authority=bind(AUTH),shard=index,
        catalog=bind(OUT/'catalog.json'),image_validations=validations,images=len(validations),label_reads=0))
    print(dict(event='FUSION_FEATURE_SHARD_PASS',shard=index,images=len(validations)),flush=True)


def join(a):
    cv=read(OUT/'catalog_validation.json');cat=read(checked(cv['catalog']));seen=set();seals=[]
    for i,keys in enumerate(cat['shards']):
        p=OUT/f'shard{i:02d}/validation.json';v=read(p)
        need(v['status']=='FUSION_FEATURE_SHARD_PASS' and v['authority']==bind(AUTH) and v['images']==len(keys),'ALL_SHARDS_PASS')
        for b in v['image_validations']:
            iv=read(checked(b));checked(iv['payload']);need(iv['key'] not in seen,'UNIQUE_IMAGES');seen.add(iv['key'])
        seals.append(bind(p))
    features={}
    for key,entry in cat['images'].items():
        if entry['pilot_validation']:checked(feature_binding(entry));seen.add(key)
        features[key]=feature_binding(entry)
    need(seen==set(cat['images']) and len(cat['queries'])==593,'COMPLETE_IMAGE_AND_QUERY_AXES')
    write(OUT/'ready.json',dict(status='FUSION_ALL593_FEATURE_CACHE_PASS',authority=bind(AUTH),catalog=bind(OUT/'catalog.json'),
        shard_validations=seals,features=features,queries=593,unique_images=len(features),reused_pilot_images=cat['reused_pilot_images'],
        label_reads=0,training_updates=0))
    print(dict(event='FUSION_ALL593_FEATURE_CACHE_PASS',queries=593,unique_images=len(features)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','catalog','extract','join'));ap.add_argument('--index',type=int);args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a=guard(args.stage)
        if args.stage=='catalog':catalog(a)
        elif args.stage=='extract':extract(a,args.index)
        else:join(a)
