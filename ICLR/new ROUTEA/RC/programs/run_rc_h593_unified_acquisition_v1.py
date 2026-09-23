#!/usr/bin/env python3
"""Shared GPU acquisition with immutable pair capsules and CPU-only export."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_roma_coordinate_precision_v2 as C
import rc_roma_visual_origin_v1 as V
import rc_roma_shared_native_cache_v1 as S
import rc_h593_unified_acquisition_core_v1 as G
from rc_roma_exact_feature_cache_v2 import equal_tree,clone_tree
read,write,save,bind,checked=C.read,C.write,C.save,C.bind,C.checked
OUT=ROOT/'cache/rc_h593_unified_acquisition_v1'
AUTH=ROOT/'registry/rc_h593_unified_acquisition_authority_v1_20260923.json'
DIRS={f:ROOT/'results'/n for f,n in dict(inside='rc_h593_m_inside_v1',visual='rc_h593_m_visual_origin_v1',coordinate='rc_h593_roma_coordinate_precision_v2').items()}
AUTHS={f:ROOT/'registry'/n for f,n in dict(inside='rc_h593_m_inside_authority_v1_20260922.json',visual='rc_h593_m_visual_origin_authority_v1_20260922.json',coordinate='rc_h593_roma_coordinate_precision_authority_v2_20260922.json').items()}
SCRIPTS={f:ROOT/'programs'/n for f,n in dict(inside='run_rc_h593_m_inside_v1.py',visual='run_rc_h593_m_visual_origin_v1.py',coordinate='run_rc_h593_roma_coordinate_precision_v2.py').items()}

def prepare():
    assert not AUTH.exists()
    sources=[Path(__file__),Path(G.__file__),Path(C.__file__),Path(V.__file__),Path(G.I.__file__),Path(S.__file__),
             ROOT/'slurm/rc_h593_unified_acquisition_v1.sbatch',ROOT/'slurm/rc_h593_unified_export_v1.sbatch',
             ROOT/'plan/RC_H593_UNIFIED_ACQUISITION_V1_20260923.md',*AUTHS.values(),*SCRIPTS.values()]
    for a in AUTHS.values():
        for b in read(a)['sources'] if 'sources' in read(a) else read(a)['code_sources'].values():checked(b)
    write(AUTH,dict(status='UNIFIED_GPU_ACQUISITION_AUTHORIZED',sources=[bind(p) for p in dict.fromkeys(sources)],
        profile=bind(C.PROFILE),workers=bind(C.WORKERS),queries=593,candidates=128,
        families=list(DIRS),chunk_seconds=640,max_chunks=32,precision='Original native tensors and FP64 statistics; no conversion',
        scope='One native forward with joint observers, shared unchanged single-image descriptors and matcher; registered interventions only',
        outputs='Compact immutable pair capsules; existing scientific artifacts/validators via CPU export',
        user_instruction='Combine required fields in one GPU acquisition, reuse existing data, CPU downstream'))

def guard(index,stage):
    a=read(AUTH)
    for b in [*a['sources'],a['profile'],a['workers']]:checked(b)
    assert os.environ.get('SLURM_JOB_ID') and index in range(593)
    w=read(C.WORKERS)['records'][index]
    allowed={Path(b['path']).resolve() for k in ('raw','roma') for b in w[k].values()}
    allowed.add(C.WORKERS.resolve())
    op=ROOT/f'results/rc_h593_quality_operator_v1/query{index:03d}'
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        assert not any(x in s for x in ('curator_roles','/target_join/','d1-mi','d1_mi','formal392','/grozi/','/isic/'))
        assert '/reports/' not in s
        if ROOT/'results' in p.parents:
            assert p in allowed or op in p.parents or any(d/f'query{index:03d}' in p.parents or d/'inputs' in p.parents for d in DIRS.values()),s
    # Source hashes were verified above; source plan text is not read afterwards.
    sys.addaudithook(audit)
    return a,w

def previous(index):
    rows={f:{} for f in DIRS};bindings={f:[] for f in DIRS}
    for f,d in DIRS.items():
        for p in sorted((d/f'query{index:03d}').glob('part*.pt')):
            value=torch.load(p,map_location='cpu',weights_only=True)
            assert value['authority']==bind(AUTHS[f])
            bindings[f].append(bind(p))
            for pair in value['pairs']:
                pos=pair['candidate_position'];assert pos not in rows[f];rows[f][pos]=pair
    return rows,bindings

def context(index,w):
    q,old,refs=C.load_input(w);profile=read(C.PROFILE);C.check_profile(profile)
    qp=Path(q['query_source_path']);assert bind(qp)['sha256']==q['query_source_sha256']
    qgeom,qmeta=C.M.geometry(qp,q['query_source_sha256'],'m-origin-query',q['query_grid_shape'],q['processor_input_frame'],profile['sources']['processor']['sha256'])
    opv=read(ROOT/f'results/rc_h593_quality_operator_v1/query{index:03d}/validation.json');operator=read(checked(opv['payload']))
    assert operator['query_id']==w['query_id']
    return q,old,refs,profile,qgeom,qmeta,operator,opv

def collect(index,pilot=False):
    a,w=guard(index,'pilot' if pilot else 'collect')
    assert torch.cuda.is_available()
    if not pilot:
        qual=read(OUT/'qualification.json');assert qual['status']=='UNIFIED_FIRST8_REGISTERED_FIELDS_BIT_EXACT' and qual['authority']==bind(AUTH)
    folder=OUT/('pilot' if pilot else f'query{index:03d}');folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'collect.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'ready.json').exists():checked(read(folder/'ready.json')['manifest']);return
    previous_rows,previous_sources=previous(index)
    q,old,refs,profile,qgeom,qmeta,operator,opv=context(index,w)
    core=C.M.legacy_core(profile);core.cell_means=S.ExactPool()
    model=C.M.gpu_model(profile);qi=core.oriented(Path(q['query_source_path']))
    input_bindings={};images={q['query_source_sha256']:dict(path=q['query_source_path'],sha256=q['query_source_sha256'])}
    def input_sink(sha,tag,lr,hr):
        key=sha+'_'+tag
        if key in input_bindings:return
        path=OUT/'inputs'/f'{key}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a+') as lk:
            fcntl.flock(lk,fcntl.LOCK_EX)
            if not path.exists():save(path,dict(authority=bind(AUTH),image_sha256=sha,transform=tag,original_image=images[sha],
                low_resolution_shape=list(lr.shape),high_resolution_shape=list(hr.shape),
                low_resolution_thumbnail64=torch.nn.functional.interpolate(lr,size=(64,64),mode='area').cpu(),
                high_resolution_thumbnail64=torch.nn.functional.interpolate(hr,size=(64,64),mode='area').cpu()))
        input_bindings[key]=bind(path)
    ob=G.JointObserver(model,core,input_sink);hook=C.CoordinateHook(model)
    start=time.monotonic();counts=Counter();sealed=[]
    try:
        for pos in range(8 if pilot else 128):
            dest=folder/f'pair{pos:03d}.pt'
            if dest.exists():sealed.append(bind(dest));continue
            need={f:pilot or pos not in previous_rows[f] for f in DIRS}
            if not any(need.values()):continue
            if not pilot and time.monotonic()-start>640:break
            physical=q['candidate_physical_rows'][pos];ref=refs[physical]
            assert ref['tokens_sha256']==old['candidates'][pos]['reference_tokens_sha256']
            rp=Path(ref['source_path']);assert bind(rp)['sha256']==ref['source_image_sha256']
            images[ref['source_image_sha256']]=dict(path=str(rp),sha256=ref['source_image_sha256'])
            rg,rm=C.M.geometry(rp,ref['source_image_sha256'],'m-origin-reference',ref['grid_shape'],'DECODED_RAW_BEFORE_EXIF',profile['sources']['processor']['sha256'])
            oldrows={f:previous_rows[f].get(pos) for f in DIRS}
            rows,n=G.acquire_pair(model,core,ob,hook,qi,core.oriented(rp),q,ref,qgeom,rg,qmeta,rm,pos,
                float(operator['modes']['M0Q0R0']['scores'][pos]['real_score']),need,oldrows,index)
            # Native token maps/C4 always preserve the historical contract.
            for f in ('inside','visual'):
                if f in rows:
                    st=rows[f]['arms']['NATIVE']['stages']['HR1']
                    assert C.M.bit_equal(st['AB']['weights'],old['candidates'][pos]['query_visibility'])
                    assert C.M.bit_equal(st['BA']['weights'],old['candidates'][pos]['reference_visibility'])
                    rows[f]['inputs']={k:v for k,v in input_bindings.items() if k.split('_',1)[0] in (q['query_source_sha256'],ref['source_image_sha256'])}
            if pilot:compare_pair(index,pos,rows,previous_rows)
            save(dest,dict(authority=bind(AUTH),query_id=w['query_id'],index=index,candidate_position=pos,physical_row=physical,
                query_image=images[q['query_source_sha256']],query_geometry=qmeta,candidate_physical_rows=q['candidate_physical_rows'],
                operator_source=opv['payload'],rows=rows,computed=need,counts=n))
            sealed.append(bind(dest));counts.update(n)
            print(dict(event='UNIFIED_PAIR_SAVED',index=index,candidate=pos,counts=n,seconds=time.monotonic()-start),flush=True)
    finally:hook.close();ob.close()
    if pilot:
        assert len(sealed)==8 and counts['native_forwards']==8 and counts['coordinate_changed_forwards']==24 and counts['visual_changed_forwards']==48
        write(OUT/'qualification.json',dict(status='UNIFIED_FIRST8_REGISTERED_FIELDS_BIT_EXACT',authority=bind(AUTH),pairs=8,
            counts=dict(counts),capsules=sealed,seconds=time.monotonic()-start,labels_read=0,
            scope='Natural first8 paired fields against three independently sealed original protocols; exporter verified separately'))
        return
    covered={f:set(previous_rows[f]) for f in DIRS}
    for p in folder.glob('pair*.pt'):
        z=torch.load(p,map_location='cpu',weights_only=True);assert z['authority']==bind(AUTH)
        for f in DIRS:
            if f in z['rows']:covered[f].add(z['candidate_position'])
    done=all(v==set(range(128)) for v in covered.values())
    if done:
        manifest=dict(status='UNIFIED_ALL128_GPU_DATA_READY',authority=bind(AUTH),index=index,query_id=w['query_id'],
              capsules=[bind(p) for p in sorted(folder.glob('pair*.pt'))],previous_sources=previous_sources,labels_read=0)
        write(folder/'manifest.json',manifest);write(folder/'ready.json',dict(status=manifest['status'],manifest=bind(folder/'manifest.json')))
    write(folder/'chunks'/f"{os.environ['SLURM_JOB_ID']}.json",dict(status='UNIFIED_NORMAL_EXIT',authority=bind(AUTH),
        index=index,complete=done,coverage={k:len(v) for k,v in covered.items()},seconds=time.monotonic()-start,counts=dict(counts)))

def compare_pair(index,pos,new,old):
    for f in ('inside','visual'):
        for arm in new[f]['arms']:
            assert equal_tree(new[f]['arms'][arm]['stages'],old[f][pos]['arms'][arm]['stages']),(f,pos,arm,'STAGE_BITS')
        for k in ('candidate_position','physical_row','reference_image','reference_geometry','free_content'):
            assert equal_tree(new[f][k],old[f][pos][k]),(f,pos,k)
    assert equal_tree(new['coordinate'],old['coordinate'][pos]),('COORDINATE_BITS',pos)
    refsha=new['inside_paths']['reference_sha']
    p=DIRS['inside']/f'query{index:03d}'/'inside'/(refsha+'.pt')
    value=torch.load(p,map_location='cpu',weights_only=True)
    for k,v in new['inside_paths'].items():assert equal_tree(v,value[k]),('INSIDE_PATH_BITS',pos,k)

def export(index):
    a,w=guard(index,'export');assert not torch.cuda.is_available(),'CPU_EXPORT_ONLY'
    folder=OUT/f'query{index:03d}';v=read(folder/'ready.json');m=read(checked(v['manifest']))
    assert m['authority']==bind(AUTH) and m['index']==index
    previous_rows,bindings=previous(index);rows={f:dict(previous_rows[f]) for f in DIRS}
    for bs in m['previous_sources'].values():
        for b in bs:checked(b)
    for b in m['capsules']:
        z=torch.load(checked(b),map_location='cpu',weights_only=True);assert z['authority']==bind(AUTH) and z['index']==index
        pos=z['candidate_position']
        for f in DIRS:
            if f in z['rows']:
                if pos in rows[f]:assert equal_tree(rows[f][pos],z['rows'][f]),('EXPORT_RESUME_DRIFT',f,pos)
                else:rows[f][pos]=z['rows'][f]
        if 'inside_paths' in z['rows']:
            value=z['rows']['inside_paths'];dest=DIRS['inside']/f'query{index:03d}'/'inside'/(value['reference_sha']+'.pt')
            if not dest.exists():save(dest,dict(authority=bind(AUTHS['inside']),**value,execution=bind(AUTH)))
    assert all(set(v)==set(range(128)) for v in rows.values())
    q,old,refs,profile,qgeom,qmeta,operator,opv=context(index,w)
    for f,d in DIRS.items():
        dest=d/f'query{index:03d}';dest.mkdir(exist_ok=True,parents=True)
        if (dest/'validation.json').exists():continue
        size=16 if f=='coordinate' else 8;parts=[]
        for part,start in enumerate(range(0,128,size)):
            p=dest/f'part{part:02d}.pt'
            if not p.exists():save(p,dict(authority=bind(AUTHS[f]),query_id=w['query_id'],part=part,
                                  pairs=[rows[f][i] for i in range(start,start+size)],execution=dict(authority=bind(AUTH),manifest=v['manifest'])))
            parts.append(bind(p))
        if f=='coordinate':
            raw=list(map(float,q['candidate_raw_scores']));ranked=sorted(range(128),key=lambda i:(-raw[i],q['candidate_physical_rows'][i]));win=ranked[0];challengers=[i for i in range(128) if i!=win]
            modes={}
            for level in C.LEVELS:
                evidence={i:rows[f][i]['arms'][level]['scores'] for i in range(128)}
                x=torch.stack([C.candidate_feature(raw,evidence,c,win) for c in challengers])
                modes[level]=dict(X=x.tolist(),scores=[evidence[i] for i in range(128)])
            payload=dict(authority=bind(AUTHS[f]),query_id=w['query_id'],execution_ordinal=index,source_image_sha256=w['source_image_sha256'],
                candidate_physical_rows=q['candidate_physical_rows'],raw_ranked_physical_rows=q['raw_ranked_physical_rows'],winner=win,challenger_positions=challengers,
                parts=parts,modes=modes,terminal_coordinate_only_control='NATIVE identical by unchanged overlaps and scorer dependency',label_reads=0,model_updates=0)
        else:
            arms=('NATIVE',) if f=='inside' else V.ARMS
            scores={arm:{st:[rows[f][i]['arms'][arm]['stages'][st]['M'] for i in range(128)] for st in V.STAGES} for arm in arms}
            inputs={k:b for pair in rows[f].values() for k,b in pair['inputs'].items()}
            payload=dict(status='M_ORIGIN_ALL128_SEALED',authority=bind(AUTHS[f]),query_id=w['query_id'],execution_ordinal=index,
                query_image=dict(path=q['query_source_path'],sha256=q['query_source_sha256']),query_geometry=qmeta,candidate_physical_rows=q['candidate_physical_rows'],
                raw_winner=operator['winner'],parts=parts,masses=scores,operator_source=opv['payload'],input_manifest=inputs,label_reads=0,model_updates=0)
        write(dest/'payload.json',dict(**payload,execution=dict(authority=bind(AUTH),manifest=v['manifest'])))
        subprocess.run([sys.executable,str(SCRIPTS[f]),'verify','--index',str(index)],check=True)
    write(folder/'export_validation.json',dict(status='UNIFIED_CPU_EXPORT_ORIGINAL_VALIDATORS_PASS',authority=bind(AUTH),index=index,
         validations={f:bind(d/f'query{index:03d}'/'validation.json') for f,d in DIRS.items()}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('prepare','pilot','collect','export'));ap.add_argument('--index',type=int,default=0);args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.stage=='prepare':prepare()
    elif args.stage=='export':export(args.index)
    else:collect(args.index,pilot=args.stage=='pilot')
