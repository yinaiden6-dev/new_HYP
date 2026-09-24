#!/usr/bin/env python3
"""Complete only missing RoMa pairs for ColQwen base-native candidates."""
import argparse
import ast
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import run_rc_colpali_mass_transfer_v1 as P

ROOT=P.ROOT;WORK=ROOT.parents[2]
OUT=ROOT/'results/rc_colpali_native_quality_v1'
AUTH=ROOT/'registry/rc_colpali_native_quality_authority_v1_20260923.json'
NATIVE=ROOT/'results/rc_colpali_native_c128_v1'
PROFILE=ROOT/'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json'
read,write,bind,checked=P.read,P.write,P.bind,P.checked


def prepare():
    from PIL import Image
    assert not AUTH.exists()
    parent=read(P.AUTH);nv=read(NATIVE/'validation.json');checked(nv['result'])
    g=read(checked(parent['gallery']))['records'];labels=[r['identity'] for r in g]
    catalog_path=ROOT/'results/rc_h593_feature_fusion_cache_v1/catalog.json';catalog=read(catalog_path)
    items={v['item']['path']:v['item'] for v in catalog['images'].values()}
    qitems={v['query_id']:catalog['images'][v['query_image_key']]['item'] for v in catalog['queries']}
    native=[read(NATIVE/f'queries/query{i:03d}/payload.json') for i in range(593)]
    old={r['query_id']:r for r in read(checked(parent['workers']))['records']}
    source=WORK/'.venv-colpali/lib/python3.13/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='smart_resize')
    ns={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    processor=read(read(PROFILE)['sources']['processor']['path']);records=[];refs={};grid_checks=[];dedup_changed=[]
    for row in native:
        # Public gallery-identity dedup, matching original RAW candidate protocol.
        seen=set();rank=[]
        for physical in row['ranked_physical_rows']:
            if labels[physical] not in seen:seen.add(labels[physical]);rank.append(physical)
        axis=sorted(rank[:128]);winner=axis.index(rank[0]);o=old[row['query_id']]
        if axis!=row['candidate_physical_rows']:dedup_changed.append(row['execution_ordinal'])
        masses=dict(zip(o['candidate_physical_rows'],o['mass']))
        qi=qitems[row['query_id']];assert qi['image_sha256']==row['image_sha256']
        for physical in axis:
            if str(physical) in refs:continue
            path=g[physical]['image_path']
            if path in items:item=dict(items[path])
            else:
                with Image.open(path) as im:h,w=im.height,im.width
                hh,ww=ns['smart_resize'](h,w,factor=28,min_pixels=processor['min_pixels'],max_pixels=processor['max_pixels'])
                item=dict(path=path,image_sha256=bind(path)['sha256'],grid=[hh//28,ww//28],frame='DECODED_RAW_BEFORE_EXIF')
            refs[str(physical)]=item
        records.append(dict(query_id=row['query_id'],execution_ordinal=row['execution_ordinal'],query_item=qi,
            candidate_physical_rows=axis,raw_scores=[row['gallery_scores'][p] for p in axis],winner=winner,
            challenger_positions=[p for p in range(128) if p!=winner],query_tokens=o['query_tokens'],query_receipt=o['query_receipt'],
            image_sha256=row['image_sha256'],reuse_mass={str(p):masses[p] for p in axis if p in masses},
            missing=[p for p in axis if p not in masses],reuse_source=o['operator_payload']))
    # Deterministic first64 known reference shapes independently validate recovery formula.
    for physical in sorted(map(int,refs)):
        t=refs[str(physical)]
        if t['path'] not in items:continue
        with Image.open(t['path']) as im:h,w=im.height,im.width
        hh,ww=ns['smart_resize'](h,w,factor=28,min_pixels=processor['min_pixels'],max_pixels=processor['max_pixels'])
        assert [hh//28,ww//28]==t['grid'],('GEOMETRY_RECOVERY',physical)
        grid_checks.append(physical)
        if len(grid_checks)==64:break
    write(OUT/'workers.json',dict(records=records,references=refs,labels_included=False))
    sources=[Path(__file__),Path(P.__file__),source,ROOT/'programs/materialize_rc_original7_train128_roma_v1.py',
        ROOT/'programs/rc_roma_exact_feature_cache_v2.py',ROOT/'programs/rc_roma_shared_native_cache_v1.py',
        ROOT/'slurm/rc_colpali_native_quality_v1.sbatch',ROOT/'slurm/rc_colpali_native_quality_join_v1.sbatch',
        ROOT/'plan/RC_COLQWEN_BASE_NATIVE_QUALITY_V1_20260923.md']
    write(AUTH,dict(status='COLQWEN_BASE_NATIVE_MISSING_QUALITY_AUTHORIZED',sources=[bind(p) for p in sources],
        workers=bind(OUT/'workers.json'),parent=bind(P.AUTH),native=bind(NATIVE/'validation.json'),profile=bind(PROFILE),
        catalog=bind(catalog_path),pairs=593*128,reused_pairs=sum(len(r['reuse_mass']) for r in records),
        missing_pairs=sum(len(r['missing']) for r in records),grid_recovery_checks=grid_checks,
        identity_dedup_changed_queries=dedup_changed,shards=50,
        quality='Original RoMa overlap pooled by the exact original geometry, sqrt(mean(u)*mean(v)); frozen sampling grid is part of quality definition',
        content_encoder='ColQwen base only; no ColNomic embeddings or ranks enter scoring',
        cache_policy='Reuse exact image pairs; native missing pairs only; query feature memoization preserves original forward',
        precision='Original RoMa profile and FP64 pooling; save u/v and source provenance per missing pair',
        downstream='Native C128 content7 and simplified mass5, same group folds and training budget; not full local seven-parameter transfer'))
    print({'status':'PREPARED','missing':sum(len(r['missing']) for r in records),'dedup_changed':dedup_changed},flush=True)


def load(stage):
    a=read(AUTH)
    for b in a['sources']:checked(b)
    checked(a['profile']);w=read(checked(a['workers']))
    assert os.environ.get('SLURM_JOB_ID')
    if stage=='worker':
        def audit(event,args):
            if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0])).resolve()
            if p.suffix=='.json' and ROOT/'results' in p.parents:
                assert OUT in p.parents,('UNAUTHORIZED_RESULT_READ',str(p))
        sys.addaudithook(audit)
    return a,w


def save(path,value):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def worker(a,w,index,budget):
    import torch
    import materialize_rc_original7_train128_roma_v1 as M
    from rc_roma_exact_feature_cache_v2 import ExactFeatureCache
    from rc_roma_shared_native_cache_v1 import ExactPool
    torch.set_num_threads(8);d=OUT/f'shards/{index:02d}';d.mkdir(parents=True,exist_ok=True)
    lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if index:
        v=read(OUT/'shards/00/validation.json');assert v['status']=='COLQWEN_BASE_NATIVE_QUALITY_SHARD_PASS' and v['authority']==bind(AUTH)
    profile=read(PROFILE)
    for name,b in profile['sources'].items():
        if name not in ('old_roma','old_train56_export'):checked(b)
    for tree in profile['source_trees'].values():
        for b in tree['python_files']:checked(b)
    core=M.legacy_core(profile);model=M.gpu_model(profile);cache=ExactFeatureCache(model);pool=ExactPool()
    start=time.monotonic();seals=[]
    def geom(item):
        assert bind(item['path'])['sha256']==item['image_sha256']
        return M.geometry(Path(item['path']),item['image_sha256'],'native-quality',item['grid'],item['frame'],profile['sources']['processor']['sha256'])[0]
    with torch.inference_mode():
        for row in [r for r in w['records'] if r['execution_ordinal']%50==index]:
            qd=OUT/f"queries/query{row['execution_ordinal']:03d}";vp=qd/'validation.json'
            if vp.exists():v=read(vp);assert v['authority']==bind(AUTH);checked(v['payload']);seals.append(bind(vp));continue
            if time.monotonic()-start>budget:return 75
            qi=core.oriented(Path(row['query_item']['path']));qg=geom(row['query_item']);qd.mkdir(parents=True,exist_ok=True)
            checkpath=qd/'existing_pair_parity.json'
            if not checkpath.exists() and row['reuse_mass']:
                physical=int(next(iter(row['reuse_mass'])));ref=w['references'][str(physical)];rg=geom(ref)
                pred=model.match(qi,core.oriented(Path(ref['path'])))
                u=pool(pred['overlap_AB'][0,...,0].cpu(),qg);v=pool(pred['overlap_BA'][0,...,0].cpu(),rg)
                assert torch.equal(u,core.cell_means(pred['overlap_AB'][0,...,0].cpu(),qg))
                assert torch.equal(v,core.cell_means(pred['overlap_BA'][0,...,0].cpu(),rg))
                mass=float(torch.sqrt(u.mean()*v.mean()));expected=row['reuse_mass'][str(physical)]
                assert abs(mass-expected)<1e-10,('EXISTING_MASS_PARITY',row['query_id'],physical,mass,expected)
                write(checkpath,dict(authority=bind(AUTH),physical_row=physical,mass=mass,expected=expected,error=abs(mass-expected)))
                del pred
            receipts=[]
            for part,begin in enumerate(range(0,len(row['missing']),8)):
                rp=qd/f'part{part:02d}.json'
                if rp.exists():v=read(rp);assert v['authority']==bind(AUTH);checked(v['payload']);receipts.append(bind(rp));continue
                pairs=[]
                for physical in row['missing'][begin:begin+8]:
                    ref=w['references'][str(physical)];rg=geom(ref)
                    pred=model.match(qi,core.oriented(Path(ref['path'])))
                    u=pool(pred['overlap_AB'][0,...,0].cpu(),qg);v=pool(pred['overlap_BA'][0,...,0].cpu(),rg)
                    pairs.append(dict(physical_row=physical,query_visibility=u,reference_visibility=v,
                        visibility_mass=float(torch.sqrt(u.mean()*v.mean())),reference_image_sha256=ref['image_sha256'],
                        query_geometry=qg.sha256,reference_geometry=rg.sha256))
                    del pred
                payload=qd/f'part{part:02d}.pt';save(payload,dict(authority=bind(AUTH),query_id=row['query_id'],pairs=pairs))
                write(rp,dict(authority=bind(AUTH),payload=bind(payload),physical_rows=[p['physical_row'] for p in pairs]));receipts.append(bind(rp))
                print({'query':row['execution_ordinal'],'part':part,'new_pairs':min(begin+8,len(row['missing'])),'total_missing':len(row['missing']),'seconds':time.monotonic()-start},flush=True)
                if time.monotonic()-start>budget:return 75
            mass=dict(row['reuse_mass'])
            for b in receipts:
                p=torch.load(checked(read(checked(b))['payload']),map_location='cpu',weights_only=True)
                for v in p['pairs']:assert str(v['physical_row']) not in mass;mass[str(v['physical_row'])]=v['visibility_mass']
            assert set(map(int,mass))==set(row['candidate_physical_rows'])
            p=dict(authority=bind(AUTH),query_id=row['query_id'],mass=[mass[str(i)] for i in row['candidate_physical_rows']],parts=receipts,reused=row['reuse_source'])
            write(qd/'payload.json',p);write(vp,dict(status='COLQWEN_BASE_NATIVE_QUALITY_QUERY_PASS',authority=bind(AUTH),payload=bind(qd/'payload.json')));seals.append(bind(vp))
    write(d/'validation.json',dict(status='COLQWEN_BASE_NATIVE_QUALITY_SHARD_PASS',authority=bind(AUTH),queries=seals))
    return 0


def join(a,w):
    records=[]
    for row in w['records']:
        v=read(OUT/f"queries/query{row['execution_ordinal']:03d}/validation.json");assert v['status']=='COLQWEN_BASE_NATIVE_QUALITY_QUERY_PASS' and v['authority']==bind(AUTH)
        p=read(checked(v['payload']));r={k:row[k] for k in ('query_id','execution_ordinal','candidate_physical_rows','query_tokens','query_receipt','image_sha256')}
        r['mass']=p['mass'];r['quality_source']=v['payload'];records.append(r)
    write(OUT/'ready_workers.json',dict(records=records,labels_included=False))
    write(OUT/'validation.json',dict(status='COLQWEN_BASE_NATIVE_ALL_QUALITY_PASS',authority=bind(AUTH),workers=bind(OUT/'ready_workers.json'),reused=a['reused_pairs'],new=a['missing_pairs']))
    print({'status':'COLQWEN_BASE_NATIVE_ALL_QUALITY_PASS','queries':len(records)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','join']);p.add_argument('--index',type=int,default=0);p.add_argument('--budget',type=int,default=660);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        a,w=load(args.stage)
        if args.stage=='worker':sys.exit(worker(a,w,args.index,args.budget))
        else:join(a,w)
