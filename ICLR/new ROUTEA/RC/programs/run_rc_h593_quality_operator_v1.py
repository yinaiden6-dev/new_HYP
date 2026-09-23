#!/usr/bin/env python3
"""CPU-only operator interventions on existing sealed RoMa and ColNomic caches."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v1 as C
import rc_quality_content_operator_v1 as K
from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature
read,write,bind,checked,need,save=C.read,C.write,C.bind,C.checked,C.need,C.save
M=C.M;LEVELS=K.LEVELS
OUT=ROOT/'results/rc_h593_quality_operator_v1';WORKERS=OUT/'workers.json'
AUTH=ROOT/'registry/rc_h593_quality_operator_authority_v1_20260922.json'
PLAN=ROOT/'plan/RC_H593_QUALITY_OPERATOR_V1_20260922.md'
LAUNCH=ROOT/'slurm/rc_h593_quality_operator_v1.sbatch'


def prepare():
    need(not AUTH.exists(),'NEW_AUTHORITY');source=C.OUT/'workers.json';records=read(source)['records'];groups=defaultdict(list)
    for r in records:groups[r['raw']['payload']['path']].append(r)
    shards=[sorted(groups[k],key=lambda r:r['execution_ordinal']) for k in sorted(groups)]
    need(len(shards)==75 and sum(map(len,shards))==593 and max(map(len,shards))<=8,'SHARD75')
    write(WORKERS,dict(shards=shards,source=bind(source),labels_included=False))
    codes={k:bind(v) for k,v in dict(program=Path(__file__),operators=Path(K.__file__),shared_io=Path(C.__file__),original_replay=Path(M.__file__),feature_formula=ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',plan=PLAN,launcher=LAUNCH).items()}
    profile=read(C.PROFILE);checked(profile['sources']['core'])
    write(AUTH,dict(status='QUALITY_OPERATOR_AUTHORIZED',user_authorization='2026-09-22 continue mechanism analysis and experiments; retain intermediate evidence',workers=bind(WORKERS),code_sources=codes,profile=bind(C.PROFILE),native_scorer=profile['sources']['core'],levels=list(LEVELS),factors=K.FACTORS,total=593,candidates=128,pilot_shard=0,
        new_encoder_forwards=0,new_RoMa_forwards=0,device='CPU FP64 8 threads',evaluation='Frozen COST1/CE plus original matched refits; exact native baseline, original grouped folds; no heldout labels before predictions sealed',
        scope='Functional intervention into query pooling, reference weighting and overall quality mass; fixed-free-argmax separates reference value attenuation from match reselection. Not six input-column zeroing.'))
    print(dict(status='PREPARED',authority=bind(AUTH)),flush=True)


def guard(stage,shard):
    a=read(AUTH);need(a['status']=='QUALITY_OPERATOR_AUTHORIZED','AUTHORITY')
    for b in a['code_sources'].values():checked(b)
    checked(a['profile']);checked(a['native_scorer']);ws=read(checked(a['workers']))['shards']
    allowed={Path(a['workers']['path']).resolve()}
    if stage!='preflight':
        need(os.environ.get('SLURM_JOB_ID') and shard in range(75),'SLURM_AND_SHARD')
        for r in ws[shard]:
            for kind in ('raw','roma'):allowed.update(Path(b['path']).resolve() for b in r[kind].values())
    ownids={r['execution_ordinal'] for r in ws[shard]} if shard is not None else set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        need(not any(k in s for k in ('curator_roles','/reports/','d1-mi','d1_mi','formal392','/target_join/','/grozi/','rc_opened_')),'LABEL_OR_PROTECTED_READ')
        if ROOT/'results' in p.parents:
            own=OUT in p.parents
            if own:
                first=p.relative_to(OUT).parts[0];own=first in ('workers.json','preflight.json',f'.preflight.json.{os.getpid()}.tmp','shard00',f'shard{shard:02d}' if shard is not None else '') or first in {f'query{i:03d}' for i in ownids}
            need(own or p in allowed,'UNLISTED_RESULT:'+s)
    sys.addaudithook(audit)
    if stage!='preflight':
        pf=read(OUT/'preflight.json');need(pf['status']=='QUALITY_OPERATOR_SYNTHETIC_PASS' and pf['authority']==bind(AUTH),'PREFLIGHT')
        if shard!=0:
            v=read(OUT/'shard00/validation.json');need(v['status']=='QUALITY_OPERATOR_SHARD_PASS' and v['authority']==bind(AUTH),'PILOT_REQUIRED')
    return a,ws


def inputs(rows):
    memo={}
    for w in rows:
        loaded=[]
        for kind in ('raw','roma'):
            b=w[kind];key=b['payload']['path']
            if key not in memo:
                rec=read(checked(b['receipt']));val=read(checked(b['validation']))
                need(rec['payload']==val['payload']==b['payload'] and val.get('receipt',b['receipt'])==b['receipt'],'SOURCE_ENVELOPE')
                need(val['status'].endswith(('CPU_REPLAY_PASS','_CPU_PASS')),'SOURCE_QUALIFIED')
                memo[key]=torch.load(checked(b['payload']),map_location='cpu',weights_only=True,mmap=True)
            loaded.append(memo[key])
        raw,roma=loaded;q=raw['records'][w['source_index']];g=roma['records'][w['source_index']]
        need(q['query_id']==g['query_id']==w['source_query_id'] and q['query_source_sha256']==g['query_source_sha256']==w['source_image_sha256'],'QUERY_SOURCE')
        need(q['candidate_physical_rows']==g['candidate_physical_rows'] and len(g['candidates'])==128,'SOURCE_C128')
        need(M.token_sha(q['query_tokens'])==q['query_tokens_sha256']==g['query_tokens_sha256'],'QUERY_TOKENS')
        yield w,q,g,raw['references']


def worker(a,rows,shard,replay=False):
    if not replay and (OUT/f'shard{shard:02d}/validation.json').exists():
        v=read(OUT/f'shard{shard:02d}/validation.json');need(v['status']=='QUALITY_OPERATOR_SHARD_PASS' and v['authority']==bind(AUTH),'SHARD_RESUME')
        for b in v['queries']:checked(b)
        print(dict(event='SHARD_ALREADY_VALIDATED',shard=shard),flush=True);return
    start=time.monotonic();native=M.legacy_core(read(C.PROFILE));seals=[];maximum=0.;checks=0
    for w,q,g,refs in inputs(rows):
        d=OUT/f"query{w['execution_ordinal']:03d}";pairs=[]
        if not replay and (d/'validation.json').exists():
            v=read(d/'validation.json');need(v['authority']==bind(AUTH) and v['status']=='QUALITY_OPERATOR_QUERY_PASS','RESUME');checked(v['payload']);seals.append(bind(d/'validation.json'));continue
        old=torch.load(d/'intermediates.pt',map_location='cpu',weights_only=True) if replay else None
        if replay:need(old['authority']==bind(AUTH) and old['query_id']==w['query_id'],'INTERMEDIATE_SOURCE')
        for pos,physical in enumerate(q['candidate_physical_rows']):
            r=refs[physical];prior=g['candidates'][pos];need(physical==prior['physical_row'] and M.token_sha(r['tokens'])==r['tokens_sha256']==prior['reference_tokens_sha256'],'REF_TOKENS')
            u=prior['query_visibility'];v=prior['reference_visibility'];scores,detail=K.compute(q['query_tokens'],r['tokens'],u,v)
            need(all(float(scores['NATIVE'][k]).hex()==float(prior['old_scores'][k]).hex() for k in K.SCORE_KEYS),'NATIVE_SCORE_BITS')
            if replay:
                record=old['pairs'][pos];need(record['physical_row']==physical and record['scores']==scores,'PAIR_REPLAY')
                need(all(torch.equal(t,record['detail'][k]) for k,t in detail.items()),'TOKEN_DETAILS_REPLAY')
                independent=K.independent(q['query_tokens'].numpy(),r['tokens'].numpy(),u.numpy(),v.numpy())
                err=max(abs(scores[m][k]-independent[m][k]) for m in LEVELS for k in K.SCORE_KEYS);maximum=max(maximum,err);need(err<2e-10,'NUMPY_OPERATOR_SCORES');checks+=len(LEVELS)*4
            pairs.append(dict(candidate_position=pos,physical_row=int(physical),reference_tokens_sha256=r['tokens_sha256'],scores=scores,detail=detail))
        raw=list(map(float,q['candidate_raw_scores']));axis=q['candidate_physical_rows'];winner=max(range(128),key=lambda i:(raw[i],-axis[i]));cs=[i for i in range(128) if i!=winner]
        modes={}
        for name in LEVELS:
            e={p['candidate_position']:p['scores'][name] for p in pairs};modes[name]=dict(X=torch.stack([candidate_feature(raw,e,c,winner) for c in cs]).tolist(),scores=[e[i] for i in range(128)])
        value=dict(authority=bind(AUTH),query_id=w['query_id'],execution_ordinal=w['execution_ordinal'],source_image_sha256=w['source_image_sha256'],candidate_physical_rows=axis,raw_ranked_physical_rows=q['raw_ranked_physical_rows'],winner=winner,challenger_positions=cs,modes=modes,label_reads=0,new_encoder_forwards=0,new_RoMa_forwards=0)
        if not replay:
            if not (d/'intermediates.pt').exists():save(d/'intermediates.pt',dict(authority=bind(AUTH),query_id=w['query_id'],raw_source=w['raw'],roma_source=w['roma'],pairs=pairs))
            value['intermediates']=bind(d/'intermediates.pt');write(d/'payload.json',value)
        else:
            value['intermediates']=bind(d/'intermediates.pt');need(value==read(d/'payload.json'),'FEATURE_REPLAY')
            write(d/'validation.json',dict(status='QUALITY_OPERATOR_QUERY_PASS',authority=bind(AUTH),payload=bind(d/'payload.json'),intermediates=value['intermediates'],native_exact=True,max_numpy_error=maximum,label_reads=0))
            seals.append(bind(d/'validation.json'))
        print(dict(event='OPERATOR_QUERY_CHECKED' if replay else 'OPERATOR_QUERY_READY',shard=shard,index=w['execution_ordinal'],seconds=time.monotonic()-start),flush=True)
    if replay:write(OUT/f'shard{shard:02d}/validation.json',dict(status='QUALITY_OPERATOR_SHARD_PASS',authority=bind(AUTH),queries=seals,query_count=len(rows),checks=checks,max_numpy_error=maximum,label_reads=0));return
    runtime=OUT/f'shard{shard:02d}/runtime.json'
    if not runtime.exists():write(runtime,dict(job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,device='cpu',queries=len(rows)))
    subprocess.run([sys.executable,__file__,'verify','--shard',str(shard)],check=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','preflight','worker','verify'));ap.add_argument('--shard',type=int);args=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    else:
        a,ws=guard(args.stage,args.shard)
        if args.stage=='preflight':p=K.self_test();p['authority']=bind(AUTH);write(OUT/'preflight.json',p);print(p)
        else:worker(a,ws[args.shard],args.shard,args.stage=='verify')
