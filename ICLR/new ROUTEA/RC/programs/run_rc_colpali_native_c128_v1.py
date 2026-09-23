#!/usr/bin/env python3
"""ColPali full-gallery retrieval; natural C128 and exact-pair reuse inventory."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import run_rc_colpali_mass_transfer_v1 as P

ROOT=P.ROOT
OUT=ROOT/'results/rc_colpali_native_c128_v1'
AUTH=ROOT/'registry/rc_colpali_native_c128_authority_v1_20260923.json'
read,write,bind,checked=P.read,P.write,P.bind,P.checked


def prepare():
    assert not AUTH.exists()
    parent=read(P.AUTH)
    for key in ('token_validation','token_manifest','gallery_tokens','workers'):
        checked(parent[key])
    manifest=read(parent['token_manifest']['path'])
    queries=read(parent['workers']['path'])['records']
    images={r['query_id']:r for r in manifest['records']}
    records=[]
    for r in queries:
        im=images[r['query_id']]
        assert im['image_sha256']==r['image_sha256']
        records.append(dict(query_id=r['query_id'],execution_ordinal=r['execution_ordinal'],
            image_path=im['image_path'],image_sha256=im['image_sha256'],
            query_tokens=r['query_tokens'],query_receipt=r['query_receipt']))
    gallery=read(checked(parent['gallery']))['records']
    assert [r['physical_row'] for r in gallery]==list(range(5413))
    write(OUT/'workers.json',dict(records=records,references=[dict(physical_row=r['physical_row'],image_path=r['image_path']) for r in gallery],labels_included=False))
    sources=[Path(__file__),Path(P.__file__),ROOT/'programs/cache_colpali_h593_queries_v4_legacy.py',
        ROOT/'slurm/rc_colpali_native_c128_v1.sbatch',ROOT/'slurm/rc_colpali_native_c128_join_v1.sbatch',
        ROOT/'plan/RC_COLPALI_NATIVE_C128_V1_20260923.md']
    write(AUTH,dict(status='COLPALI_OWN_NATURAL_C128_AUTHORIZED',sources=[bind(p) for p in sources],
        workers=bind(OUT/'workers.json'),parent_diagnostic=bind(P.AUTH),
        gallery_tokens=parent['gallery_tokens'],token_validation=parent['token_validation'],
        gallery=parent['gallery'],curator=parent['curator'],split=parent['split'],folds=parent['folds'],
        queries=593,gallery_size=5413,candidates=128,shards=50,
        score='ColPali MaxSim sum over all 1030 cached tokens, FP64, no renormalization; identical to prior diagnostic scoring',
        ranking='Descending score, ascending physical row for exact ties; no target insertion',
        feature_boundary='Existing mass-transfer result is auxiliary fixed-ColNomic-C128 evidence only',
        reuse='Only identical query image and reference physical row with matching source provenance',
        downstream='Refit on original group folds; recompute effective TRAIN from native ColPali recall, never reuse ColNomic effective TRAIN IDs',
        scope='H593 opened development panel; ColPali natural retrieval, not untouched external confirmation'))
    print({'status':'COLPALI_NATIVE_PREPARED','queries':593,'gallery':5413},flush=True)


def load(stage):
    a=read(AUTH)
    for b in a['sources']:checked(b)
    rows=read(checked(a['workers']))['records']
    assert os.environ.get('SLURM_JOB_ID')
    if stage=='score':
        allowed={Path(a['workers']['path']).resolve()}
        def audit(event,args):
            if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0])).resolve()
            if p.suffix=='.json' and ROOT/'results' in p.parents:
                assert OUT in p.parents or P.TOK in p.parents or p in allowed,('LABEL_OR_FOREIGN_RESULT_READ',str(p))
        sys.addaudithook(audit)
    return a,rows


def score(a,rows,index,budget):
    import numpy as np
    import torch
    import cache_colpali_h593_queries_v4_legacy as C
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    d=OUT/f'shards/{index:02d}';d.mkdir(parents=True,exist_ok=True)
    lock=(d/'lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if index:
        v=read(OUT/'shards/00/validation.json');assert v['status']=='COLPALI_NATIVE_SCORE_SHARD_PASS' and v['authority']==bind(AUTH)
    if index==0:checked(a['gallery_tokens'])
    gallery=C.gallery_memmap(a['gallery_tokens']['path']);assert gallery.shape==(5413,1030,128)
    refs=torch.from_numpy(np.array(gallery,copy=True)).to('cuda',dtype=torch.float64)
    start=time.monotonic();seals=[]
    for row in [r for r in rows if r['execution_ordinal']%50==index]:
        qd=OUT/f"queries/query{row['execution_ordinal']:03d}";vp=qd/'validation.json'
        if vp.exists():
            v=read(vp);assert v['authority']==bind(AUTH);checked(v['payload']);seals.append(bind(vp));continue
        if time.monotonic()-start>budget:return 75
        q=torch.load(checked(row['query_tokens']),map_location='cpu',weights_only=False)
        assert q['query_id']==row['query_id'] and q['image_sha256']==row['image_sha256'] and q['tokens'].shape==(1030,128)
        query=q['tokens'].to('cuda',dtype=torch.float64)
        qd.mkdir(parents=True,exist_ok=True);partial=qd/'partial.json'
        raw=[]
        if partial.exists():
            p=read(partial);assert p['authority']==bind(AUTH) and p['query_id']==row['query_id'];raw=p['scores']
        begun=time.monotonic()
        with torch.inference_mode():
            for pos in range(len(raw),5413,8):
                sim=torch.einsum('nd,csd->cns',query,refs[pos:pos+8])
                vals=sim.max(dim=2).values.sum(dim=1)
                raw.extend(vals.cpu().tolist())
                if (pos//8)%32==31 or len(raw)==5413 or time.monotonic()-start>budget:
                    obj=dict(authority=bind(AUTH),query_id=row['query_id'],scores=raw)
                    tmp=partial.with_suffix('.tmp');tmp.write_text(json.dumps(obj,allow_nan=False));os.replace(tmp,partial)
                if len(raw)<5413 and time.monotonic()-start>budget:return 75
        assert len(raw)==5413 and np.isfinite(raw).all()
        order=sorted(range(5413),key=lambda i:(-raw[i],i));axis=sorted(order[:128]);winner=axis.index(order[0])
        # Independent NumPy FP64 checks on a fixed row, the winner, and boundary ranks.
        errors={};nq=q['tokens'].numpy().astype(np.float64)
        for physical in sorted({0,order[0],order[127],order[128]}):
            expected=float((nq@np.asarray(gallery[physical],dtype=np.float64).T).max(axis=1).sum())
            errors[str(physical)]=abs(expected-raw[physical]);assert errors[str(physical)]<2e-9
        payload=dict(authority=bind(AUTH),query_id=row['query_id'],execution_ordinal=row['execution_ordinal'],
            candidate_source='ColPali natural top128 of full5413 gallery',
            gallery_scores=raw,ranked_physical_rows=order,candidate_physical_rows=axis,
            raw_scores=[raw[p] for p in axis],winner=winner,challenger_positions=[i for i in range(128) if i!=winner],
            query_tokens=row['query_tokens'],image_sha256=row['image_sha256'],label_reads=0)
        write(qd/'payload.json',payload);write(vp,dict(status='COLPALI_NATIVE_5413_TO128_PASS',authority=bind(AUTH),
            payload=bind(qd/'payload.json'),independent_numpy_errors=errors))
        seals.append(bind(vp));print({'query':row['execution_ordinal'],'full_gallery':5413,'seconds':time.monotonic()-begun},flush=True)
    write(d/'validation.json',dict(status='COLPALI_NATIVE_SCORE_SHARD_PASS',authority=bind(AUTH),validations=seals))
    return 0


def collect(a):
    rows=[];seals=[]
    for s in range(50):
        v=read(OUT/f'shards/{s:02d}/validation.json');assert v['authority']==bind(AUTH) and v['status']=='COLPALI_NATIVE_SCORE_SHARD_PASS'
        for b in v['validations']:
            q=read(checked(b));assert q['authority']==bind(AUTH)
            rows.append(read(checked(q['payload'])));seals.append(b)
    rows.sort(key=lambda r:r['execution_ordinal']);assert [r['execution_ordinal'] for r in rows]==list(range(593))
    write(OUT/'prelabel_seal.json',dict(authority=bind(AUTH),queries=seals))
    return rows


def join(a):
    rows=collect(a)
    parent=read(checked(a['parent_diagnostic']));old={r['query_id']:r for r in read(checked(parent['workers']))['records']}
    jobs=[];reused=0;missing=0
    for r in rows:
        previous=old[r['query_id']];assert r['image_sha256']==previous['image_sha256']
        checked(previous['operator_payload']);m=dict(zip(previous['candidate_physical_rows'],previous['mass']))
        hits=[p for p in r['candidate_physical_rows'] if p in m];needs=[p for p in r['candidate_physical_rows'] if p not in m]
        reused+=len(hits);missing+=len(needs)
        jobs.append(dict(query_id=r['query_id'],execution_ordinal=r['execution_ordinal'],
            native_payload=bind(OUT/f"queries/query{r['execution_ordinal']:03d}/payload.json"),
            candidate_physical_rows=r['candidate_physical_rows'],reuse_operator=previous['operator_payload'],
            reusable_original_grid_mass={str(p):m[p] for p in hits},missing_physical_rows=needs))
    # No labels enter the missing-pair work manifest.
    write(OUT/'pair_reuse_inventory.json',dict(authority=bind(AUTH),pairs=593*128,reusable_pairs=reused,
        missing_pairs=missing,records=jobs,labels_included=False,
        caution='Reuse scalar M only under the identical original pooling definition; ColPali32x32 local weights require a separate mapping qualification'))
    roles={r['query_id']:r for r in read(checked(a['curator']))['records']}
    labels={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    results=[]
    for r in rows:
        target=roles[r['query_id']]['identity'];ranks=[i+1 for i,p in enumerate(r['ranked_physical_rows']) if labels[p]==target]
        rank=min(ranks) if ranks else None
        results.append(dict(query_id=r['query_id'],component=roles[r['query_id']]['component'],
            full_gallery_rank=rank,correct=rank==1,target_in_C128=rank is not None and rank<=128,
            colpali_winner=r['ranked_physical_rows'][0],candidate_overlap=len(set(r['candidate_physical_rows'])&set(old[r['query_id']]['candidate_physical_rows']))))
    folds={}
    byid={r['query_id']:r for r in results}
    for f,b in a['folds'].items():
        folds[f]=dict(train_query_ids=b['train_query_ids'],effective_train_query_ids=[q for q in b['train_query_ids'] if byid[q]['target_in_C128']],
            heldout_query_ids=read(checked(a['split']))['folds'][int(f)]['heldout_query_ids'])
    write(OUT/'native_effective_train_manifest.json',dict(authority=bind(AUTH),folds=folds,rule='Natural ColPali recall only; original group split unchanged'))
    result=dict(status='COLPALI_NATIVE_RAW_AND_REUSE_INVENTORY_COMPLETE',authority=bind(AUTH),denominator=593,gallery_size=5413,
        raw_correct=sum(r['correct'] for r in results),recall_C128=sum(r['target_in_C128'] for r in results),
        R5=sum(r['full_gallery_rank'] is not None and r['full_gallery_rank']<=5 for r in results)/593,
        MRR=sum(1/r['full_gallery_rank'] if r['full_gallery_rank'] else 0 for r in results)/593,
        reusable_pairs=reused,missing_pairs=missing,rows=results,quality_head_result_available=False)
    write(OUT/'result.json',result);write(OUT/'validation.json',dict(status='COLPALI_NATIVE_RAW_593_PASS',authority=bind(AUTH),result=bind(OUT/'result.json')))
    print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score','join']);p.add_argument('--index',type=int,default=0);p.add_argument('--budget',type=int,default=735);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        a,rows=load(args.stage)
        if args.stage=='score':sys.exit(score(a,rows,args.index,args.budget))
        else:join(a)
